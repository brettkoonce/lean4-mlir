module @m {
  func.func @convnext_adamdrop_train_step(%x: tensor<32x150528xf32>, %psW: tensor<96x3x4x4xf32>, %psb: tensor<96xf32>, %psng: tensor<96xf32>, %psnbt: tensor<96xf32>, %s0b0dW: tensor<96x1x7x7xf32>, %s0b0db: tensor<96xf32>, %s0b0ng: tensor<96xf32>, %s0b0nbt: tensor<96xf32>, %s0b0eW: tensor<384x96x1x1xf32>, %s0b0eb: tensor<384xf32>, %s0b0pW: tensor<96x384x1x1xf32>, %s0b0pb: tensor<96xf32>, %s0b0lg: tensor<96xf32>, %s0b1dW: tensor<96x1x7x7xf32>, %s0b1db: tensor<96xf32>, %s0b1ng: tensor<96xf32>, %s0b1nbt: tensor<96xf32>, %s0b1eW: tensor<384x96x1x1xf32>, %s0b1eb: tensor<384xf32>, %s0b1pW: tensor<96x384x1x1xf32>, %s0b1pb: tensor<96xf32>, %s0b1lg: tensor<96xf32>, %s0b2dW: tensor<96x1x7x7xf32>, %s0b2db: tensor<96xf32>, %s0b2ng: tensor<96xf32>, %s0b2nbt: tensor<96xf32>, %s0b2eW: tensor<384x96x1x1xf32>, %s0b2eb: tensor<384xf32>, %s0b2pW: tensor<96x384x1x1xf32>, %s0b2pb: tensor<96xf32>, %s0b2lg: tensor<96xf32>, %d0ng: tensor<96xf32>, %d0nbt: tensor<96xf32>, %d0W: tensor<192x96x2x2xf32>, %d0b: tensor<192xf32>, %s1b0dW: tensor<192x1x7x7xf32>, %s1b0db: tensor<192xf32>, %s1b0ng: tensor<192xf32>, %s1b0nbt: tensor<192xf32>, %s1b0eW: tensor<768x192x1x1xf32>, %s1b0eb: tensor<768xf32>, %s1b0pW: tensor<192x768x1x1xf32>, %s1b0pb: tensor<192xf32>, %s1b0lg: tensor<192xf32>, %s1b1dW: tensor<192x1x7x7xf32>, %s1b1db: tensor<192xf32>, %s1b1ng: tensor<192xf32>, %s1b1nbt: tensor<192xf32>, %s1b1eW: tensor<768x192x1x1xf32>, %s1b1eb: tensor<768xf32>, %s1b1pW: tensor<192x768x1x1xf32>, %s1b1pb: tensor<192xf32>, %s1b1lg: tensor<192xf32>, %s1b2dW: tensor<192x1x7x7xf32>, %s1b2db: tensor<192xf32>, %s1b2ng: tensor<192xf32>, %s1b2nbt: tensor<192xf32>, %s1b2eW: tensor<768x192x1x1xf32>, %s1b2eb: tensor<768xf32>, %s1b2pW: tensor<192x768x1x1xf32>, %s1b2pb: tensor<192xf32>, %s1b2lg: tensor<192xf32>, %d1ng: tensor<192xf32>, %d1nbt: tensor<192xf32>, %d1W: tensor<384x192x2x2xf32>, %d1b: tensor<384xf32>, %s2b0dW: tensor<384x1x7x7xf32>, %s2b0db: tensor<384xf32>, %s2b0ng: tensor<384xf32>, %s2b0nbt: tensor<384xf32>, %s2b0eW: tensor<1536x384x1x1xf32>, %s2b0eb: tensor<1536xf32>, %s2b0pW: tensor<384x1536x1x1xf32>, %s2b0pb: tensor<384xf32>, %s2b0lg: tensor<384xf32>, %s2b1dW: tensor<384x1x7x7xf32>, %s2b1db: tensor<384xf32>, %s2b1ng: tensor<384xf32>, %s2b1nbt: tensor<384xf32>, %s2b1eW: tensor<1536x384x1x1xf32>, %s2b1eb: tensor<1536xf32>, %s2b1pW: tensor<384x1536x1x1xf32>, %s2b1pb: tensor<384xf32>, %s2b1lg: tensor<384xf32>, %s2b2dW: tensor<384x1x7x7xf32>, %s2b2db: tensor<384xf32>, %s2b2ng: tensor<384xf32>, %s2b2nbt: tensor<384xf32>, %s2b2eW: tensor<1536x384x1x1xf32>, %s2b2eb: tensor<1536xf32>, %s2b2pW: tensor<384x1536x1x1xf32>, %s2b2pb: tensor<384xf32>, %s2b2lg: tensor<384xf32>, %s2b3dW: tensor<384x1x7x7xf32>, %s2b3db: tensor<384xf32>, %s2b3ng: tensor<384xf32>, %s2b3nbt: tensor<384xf32>, %s2b3eW: tensor<1536x384x1x1xf32>, %s2b3eb: tensor<1536xf32>, %s2b3pW: tensor<384x1536x1x1xf32>, %s2b3pb: tensor<384xf32>, %s2b3lg: tensor<384xf32>, %s2b4dW: tensor<384x1x7x7xf32>, %s2b4db: tensor<384xf32>, %s2b4ng: tensor<384xf32>, %s2b4nbt: tensor<384xf32>, %s2b4eW: tensor<1536x384x1x1xf32>, %s2b4eb: tensor<1536xf32>, %s2b4pW: tensor<384x1536x1x1xf32>, %s2b4pb: tensor<384xf32>, %s2b4lg: tensor<384xf32>, %s2b5dW: tensor<384x1x7x7xf32>, %s2b5db: tensor<384xf32>, %s2b5ng: tensor<384xf32>, %s2b5nbt: tensor<384xf32>, %s2b5eW: tensor<1536x384x1x1xf32>, %s2b5eb: tensor<1536xf32>, %s2b5pW: tensor<384x1536x1x1xf32>, %s2b5pb: tensor<384xf32>, %s2b5lg: tensor<384xf32>, %s2b6dW: tensor<384x1x7x7xf32>, %s2b6db: tensor<384xf32>, %s2b6ng: tensor<384xf32>, %s2b6nbt: tensor<384xf32>, %s2b6eW: tensor<1536x384x1x1xf32>, %s2b6eb: tensor<1536xf32>, %s2b6pW: tensor<384x1536x1x1xf32>, %s2b6pb: tensor<384xf32>, %s2b6lg: tensor<384xf32>, %s2b7dW: tensor<384x1x7x7xf32>, %s2b7db: tensor<384xf32>, %s2b7ng: tensor<384xf32>, %s2b7nbt: tensor<384xf32>, %s2b7eW: tensor<1536x384x1x1xf32>, %s2b7eb: tensor<1536xf32>, %s2b7pW: tensor<384x1536x1x1xf32>, %s2b7pb: tensor<384xf32>, %s2b7lg: tensor<384xf32>, %s2b8dW: tensor<384x1x7x7xf32>, %s2b8db: tensor<384xf32>, %s2b8ng: tensor<384xf32>, %s2b8nbt: tensor<384xf32>, %s2b8eW: tensor<1536x384x1x1xf32>, %s2b8eb: tensor<1536xf32>, %s2b8pW: tensor<384x1536x1x1xf32>, %s2b8pb: tensor<384xf32>, %s2b8lg: tensor<384xf32>, %d2ng: tensor<384xf32>, %d2nbt: tensor<384xf32>, %d2W: tensor<768x384x2x2xf32>, %d2b: tensor<768xf32>, %s3b0dW: tensor<768x1x7x7xf32>, %s3b0db: tensor<768xf32>, %s3b0ng: tensor<768xf32>, %s3b0nbt: tensor<768xf32>, %s3b0eW: tensor<3072x768x1x1xf32>, %s3b0eb: tensor<3072xf32>, %s3b0pW: tensor<768x3072x1x1xf32>, %s3b0pb: tensor<768xf32>, %s3b0lg: tensor<768xf32>, %s3b1dW: tensor<768x1x7x7xf32>, %s3b1db: tensor<768xf32>, %s3b1ng: tensor<768xf32>, %s3b1nbt: tensor<768xf32>, %s3b1eW: tensor<3072x768x1x1xf32>, %s3b1eb: tensor<3072xf32>, %s3b1pW: tensor<768x3072x1x1xf32>, %s3b1pb: tensor<768xf32>, %s3b1lg: tensor<768xf32>, %s3b2dW: tensor<768x1x7x7xf32>, %s3b2db: tensor<768xf32>, %s3b2ng: tensor<768xf32>, %s3b2nbt: tensor<768xf32>, %s3b2eW: tensor<3072x768x1x1xf32>, %s3b2eb: tensor<3072xf32>, %s3b2pW: tensor<768x3072x1x1xf32>, %s3b2pb: tensor<768xf32>, %s3b2lg: tensor<768xf32>, %Wd: tensor<768x10xf32>, %bd: tensor<10xf32>, %psWm: tensor<96x3x4x4xf32>, %psbm: tensor<96xf32>, %psngm: tensor<96xf32>, %psnbtm: tensor<96xf32>, %s0b0dWm: tensor<96x1x7x7xf32>, %s0b0dbm: tensor<96xf32>, %s0b0ngm: tensor<96xf32>, %s0b0nbtm: tensor<96xf32>, %s0b0eWm: tensor<384x96x1x1xf32>, %s0b0ebm: tensor<384xf32>, %s0b0pWm: tensor<96x384x1x1xf32>, %s0b0pbm: tensor<96xf32>, %s0b0lgm: tensor<96xf32>, %s0b1dWm: tensor<96x1x7x7xf32>, %s0b1dbm: tensor<96xf32>, %s0b1ngm: tensor<96xf32>, %s0b1nbtm: tensor<96xf32>, %s0b1eWm: tensor<384x96x1x1xf32>, %s0b1ebm: tensor<384xf32>, %s0b1pWm: tensor<96x384x1x1xf32>, %s0b1pbm: tensor<96xf32>, %s0b1lgm: tensor<96xf32>, %s0b2dWm: tensor<96x1x7x7xf32>, %s0b2dbm: tensor<96xf32>, %s0b2ngm: tensor<96xf32>, %s0b2nbtm: tensor<96xf32>, %s0b2eWm: tensor<384x96x1x1xf32>, %s0b2ebm: tensor<384xf32>, %s0b2pWm: tensor<96x384x1x1xf32>, %s0b2pbm: tensor<96xf32>, %s0b2lgm: tensor<96xf32>, %d0ngm: tensor<96xf32>, %d0nbtm: tensor<96xf32>, %d0Wm: tensor<192x96x2x2xf32>, %d0bm: tensor<192xf32>, %s1b0dWm: tensor<192x1x7x7xf32>, %s1b0dbm: tensor<192xf32>, %s1b0ngm: tensor<192xf32>, %s1b0nbtm: tensor<192xf32>, %s1b0eWm: tensor<768x192x1x1xf32>, %s1b0ebm: tensor<768xf32>, %s1b0pWm: tensor<192x768x1x1xf32>, %s1b0pbm: tensor<192xf32>, %s1b0lgm: tensor<192xf32>, %s1b1dWm: tensor<192x1x7x7xf32>, %s1b1dbm: tensor<192xf32>, %s1b1ngm: tensor<192xf32>, %s1b1nbtm: tensor<192xf32>, %s1b1eWm: tensor<768x192x1x1xf32>, %s1b1ebm: tensor<768xf32>, %s1b1pWm: tensor<192x768x1x1xf32>, %s1b1pbm: tensor<192xf32>, %s1b1lgm: tensor<192xf32>, %s1b2dWm: tensor<192x1x7x7xf32>, %s1b2dbm: tensor<192xf32>, %s1b2ngm: tensor<192xf32>, %s1b2nbtm: tensor<192xf32>, %s1b2eWm: tensor<768x192x1x1xf32>, %s1b2ebm: tensor<768xf32>, %s1b2pWm: tensor<192x768x1x1xf32>, %s1b2pbm: tensor<192xf32>, %s1b2lgm: tensor<192xf32>, %d1ngm: tensor<192xf32>, %d1nbtm: tensor<192xf32>, %d1Wm: tensor<384x192x2x2xf32>, %d1bm: tensor<384xf32>, %s2b0dWm: tensor<384x1x7x7xf32>, %s2b0dbm: tensor<384xf32>, %s2b0ngm: tensor<384xf32>, %s2b0nbtm: tensor<384xf32>, %s2b0eWm: tensor<1536x384x1x1xf32>, %s2b0ebm: tensor<1536xf32>, %s2b0pWm: tensor<384x1536x1x1xf32>, %s2b0pbm: tensor<384xf32>, %s2b0lgm: tensor<384xf32>, %s2b1dWm: tensor<384x1x7x7xf32>, %s2b1dbm: tensor<384xf32>, %s2b1ngm: tensor<384xf32>, %s2b1nbtm: tensor<384xf32>, %s2b1eWm: tensor<1536x384x1x1xf32>, %s2b1ebm: tensor<1536xf32>, %s2b1pWm: tensor<384x1536x1x1xf32>, %s2b1pbm: tensor<384xf32>, %s2b1lgm: tensor<384xf32>, %s2b2dWm: tensor<384x1x7x7xf32>, %s2b2dbm: tensor<384xf32>, %s2b2ngm: tensor<384xf32>, %s2b2nbtm: tensor<384xf32>, %s2b2eWm: tensor<1536x384x1x1xf32>, %s2b2ebm: tensor<1536xf32>, %s2b2pWm: tensor<384x1536x1x1xf32>, %s2b2pbm: tensor<384xf32>, %s2b2lgm: tensor<384xf32>, %s2b3dWm: tensor<384x1x7x7xf32>, %s2b3dbm: tensor<384xf32>, %s2b3ngm: tensor<384xf32>, %s2b3nbtm: tensor<384xf32>, %s2b3eWm: tensor<1536x384x1x1xf32>, %s2b3ebm: tensor<1536xf32>, %s2b3pWm: tensor<384x1536x1x1xf32>, %s2b3pbm: tensor<384xf32>, %s2b3lgm: tensor<384xf32>, %s2b4dWm: tensor<384x1x7x7xf32>, %s2b4dbm: tensor<384xf32>, %s2b4ngm: tensor<384xf32>, %s2b4nbtm: tensor<384xf32>, %s2b4eWm: tensor<1536x384x1x1xf32>, %s2b4ebm: tensor<1536xf32>, %s2b4pWm: tensor<384x1536x1x1xf32>, %s2b4pbm: tensor<384xf32>, %s2b4lgm: tensor<384xf32>, %s2b5dWm: tensor<384x1x7x7xf32>, %s2b5dbm: tensor<384xf32>, %s2b5ngm: tensor<384xf32>, %s2b5nbtm: tensor<384xf32>, %s2b5eWm: tensor<1536x384x1x1xf32>, %s2b5ebm: tensor<1536xf32>, %s2b5pWm: tensor<384x1536x1x1xf32>, %s2b5pbm: tensor<384xf32>, %s2b5lgm: tensor<384xf32>, %s2b6dWm: tensor<384x1x7x7xf32>, %s2b6dbm: tensor<384xf32>, %s2b6ngm: tensor<384xf32>, %s2b6nbtm: tensor<384xf32>, %s2b6eWm: tensor<1536x384x1x1xf32>, %s2b6ebm: tensor<1536xf32>, %s2b6pWm: tensor<384x1536x1x1xf32>, %s2b6pbm: tensor<384xf32>, %s2b6lgm: tensor<384xf32>, %s2b7dWm: tensor<384x1x7x7xf32>, %s2b7dbm: tensor<384xf32>, %s2b7ngm: tensor<384xf32>, %s2b7nbtm: tensor<384xf32>, %s2b7eWm: tensor<1536x384x1x1xf32>, %s2b7ebm: tensor<1536xf32>, %s2b7pWm: tensor<384x1536x1x1xf32>, %s2b7pbm: tensor<384xf32>, %s2b7lgm: tensor<384xf32>, %s2b8dWm: tensor<384x1x7x7xf32>, %s2b8dbm: tensor<384xf32>, %s2b8ngm: tensor<384xf32>, %s2b8nbtm: tensor<384xf32>, %s2b8eWm: tensor<1536x384x1x1xf32>, %s2b8ebm: tensor<1536xf32>, %s2b8pWm: tensor<384x1536x1x1xf32>, %s2b8pbm: tensor<384xf32>, %s2b8lgm: tensor<384xf32>, %d2ngm: tensor<384xf32>, %d2nbtm: tensor<384xf32>, %d2Wm: tensor<768x384x2x2xf32>, %d2bm: tensor<768xf32>, %s3b0dWm: tensor<768x1x7x7xf32>, %s3b0dbm: tensor<768xf32>, %s3b0ngm: tensor<768xf32>, %s3b0nbtm: tensor<768xf32>, %s3b0eWm: tensor<3072x768x1x1xf32>, %s3b0ebm: tensor<3072xf32>, %s3b0pWm: tensor<768x3072x1x1xf32>, %s3b0pbm: tensor<768xf32>, %s3b0lgm: tensor<768xf32>, %s3b1dWm: tensor<768x1x7x7xf32>, %s3b1dbm: tensor<768xf32>, %s3b1ngm: tensor<768xf32>, %s3b1nbtm: tensor<768xf32>, %s3b1eWm: tensor<3072x768x1x1xf32>, %s3b1ebm: tensor<3072xf32>, %s3b1pWm: tensor<768x3072x1x1xf32>, %s3b1pbm: tensor<768xf32>, %s3b1lgm: tensor<768xf32>, %s3b2dWm: tensor<768x1x7x7xf32>, %s3b2dbm: tensor<768xf32>, %s3b2ngm: tensor<768xf32>, %s3b2nbtm: tensor<768xf32>, %s3b2eWm: tensor<3072x768x1x1xf32>, %s3b2ebm: tensor<3072xf32>, %s3b2pWm: tensor<768x3072x1x1xf32>, %s3b2pbm: tensor<768xf32>, %s3b2lgm: tensor<768xf32>, %Wdm: tensor<768x10xf32>, %bdm: tensor<10xf32>, %psWv: tensor<96x3x4x4xf32>, %psbv: tensor<96xf32>, %psngv: tensor<96xf32>, %psnbtv: tensor<96xf32>, %s0b0dWv: tensor<96x1x7x7xf32>, %s0b0dbv: tensor<96xf32>, %s0b0ngv: tensor<96xf32>, %s0b0nbtv: tensor<96xf32>, %s0b0eWv: tensor<384x96x1x1xf32>, %s0b0ebv: tensor<384xf32>, %s0b0pWv: tensor<96x384x1x1xf32>, %s0b0pbv: tensor<96xf32>, %s0b0lgv: tensor<96xf32>, %s0b1dWv: tensor<96x1x7x7xf32>, %s0b1dbv: tensor<96xf32>, %s0b1ngv: tensor<96xf32>, %s0b1nbtv: tensor<96xf32>, %s0b1eWv: tensor<384x96x1x1xf32>, %s0b1ebv: tensor<384xf32>, %s0b1pWv: tensor<96x384x1x1xf32>, %s0b1pbv: tensor<96xf32>, %s0b1lgv: tensor<96xf32>, %s0b2dWv: tensor<96x1x7x7xf32>, %s0b2dbv: tensor<96xf32>, %s0b2ngv: tensor<96xf32>, %s0b2nbtv: tensor<96xf32>, %s0b2eWv: tensor<384x96x1x1xf32>, %s0b2ebv: tensor<384xf32>, %s0b2pWv: tensor<96x384x1x1xf32>, %s0b2pbv: tensor<96xf32>, %s0b2lgv: tensor<96xf32>, %d0ngv: tensor<96xf32>, %d0nbtv: tensor<96xf32>, %d0Wv: tensor<192x96x2x2xf32>, %d0bv: tensor<192xf32>, %s1b0dWv: tensor<192x1x7x7xf32>, %s1b0dbv: tensor<192xf32>, %s1b0ngv: tensor<192xf32>, %s1b0nbtv: tensor<192xf32>, %s1b0eWv: tensor<768x192x1x1xf32>, %s1b0ebv: tensor<768xf32>, %s1b0pWv: tensor<192x768x1x1xf32>, %s1b0pbv: tensor<192xf32>, %s1b0lgv: tensor<192xf32>, %s1b1dWv: tensor<192x1x7x7xf32>, %s1b1dbv: tensor<192xf32>, %s1b1ngv: tensor<192xf32>, %s1b1nbtv: tensor<192xf32>, %s1b1eWv: tensor<768x192x1x1xf32>, %s1b1ebv: tensor<768xf32>, %s1b1pWv: tensor<192x768x1x1xf32>, %s1b1pbv: tensor<192xf32>, %s1b1lgv: tensor<192xf32>, %s1b2dWv: tensor<192x1x7x7xf32>, %s1b2dbv: tensor<192xf32>, %s1b2ngv: tensor<192xf32>, %s1b2nbtv: tensor<192xf32>, %s1b2eWv: tensor<768x192x1x1xf32>, %s1b2ebv: tensor<768xf32>, %s1b2pWv: tensor<192x768x1x1xf32>, %s1b2pbv: tensor<192xf32>, %s1b2lgv: tensor<192xf32>, %d1ngv: tensor<192xf32>, %d1nbtv: tensor<192xf32>, %d1Wv: tensor<384x192x2x2xf32>, %d1bv: tensor<384xf32>, %s2b0dWv: tensor<384x1x7x7xf32>, %s2b0dbv: tensor<384xf32>, %s2b0ngv: tensor<384xf32>, %s2b0nbtv: tensor<384xf32>, %s2b0eWv: tensor<1536x384x1x1xf32>, %s2b0ebv: tensor<1536xf32>, %s2b0pWv: tensor<384x1536x1x1xf32>, %s2b0pbv: tensor<384xf32>, %s2b0lgv: tensor<384xf32>, %s2b1dWv: tensor<384x1x7x7xf32>, %s2b1dbv: tensor<384xf32>, %s2b1ngv: tensor<384xf32>, %s2b1nbtv: tensor<384xf32>, %s2b1eWv: tensor<1536x384x1x1xf32>, %s2b1ebv: tensor<1536xf32>, %s2b1pWv: tensor<384x1536x1x1xf32>, %s2b1pbv: tensor<384xf32>, %s2b1lgv: tensor<384xf32>, %s2b2dWv: tensor<384x1x7x7xf32>, %s2b2dbv: tensor<384xf32>, %s2b2ngv: tensor<384xf32>, %s2b2nbtv: tensor<384xf32>, %s2b2eWv: tensor<1536x384x1x1xf32>, %s2b2ebv: tensor<1536xf32>, %s2b2pWv: tensor<384x1536x1x1xf32>, %s2b2pbv: tensor<384xf32>, %s2b2lgv: tensor<384xf32>, %s2b3dWv: tensor<384x1x7x7xf32>, %s2b3dbv: tensor<384xf32>, %s2b3ngv: tensor<384xf32>, %s2b3nbtv: tensor<384xf32>, %s2b3eWv: tensor<1536x384x1x1xf32>, %s2b3ebv: tensor<1536xf32>, %s2b3pWv: tensor<384x1536x1x1xf32>, %s2b3pbv: tensor<384xf32>, %s2b3lgv: tensor<384xf32>, %s2b4dWv: tensor<384x1x7x7xf32>, %s2b4dbv: tensor<384xf32>, %s2b4ngv: tensor<384xf32>, %s2b4nbtv: tensor<384xf32>, %s2b4eWv: tensor<1536x384x1x1xf32>, %s2b4ebv: tensor<1536xf32>, %s2b4pWv: tensor<384x1536x1x1xf32>, %s2b4pbv: tensor<384xf32>, %s2b4lgv: tensor<384xf32>, %s2b5dWv: tensor<384x1x7x7xf32>, %s2b5dbv: tensor<384xf32>, %s2b5ngv: tensor<384xf32>, %s2b5nbtv: tensor<384xf32>, %s2b5eWv: tensor<1536x384x1x1xf32>, %s2b5ebv: tensor<1536xf32>, %s2b5pWv: tensor<384x1536x1x1xf32>, %s2b5pbv: tensor<384xf32>, %s2b5lgv: tensor<384xf32>, %s2b6dWv: tensor<384x1x7x7xf32>, %s2b6dbv: tensor<384xf32>, %s2b6ngv: tensor<384xf32>, %s2b6nbtv: tensor<384xf32>, %s2b6eWv: tensor<1536x384x1x1xf32>, %s2b6ebv: tensor<1536xf32>, %s2b6pWv: tensor<384x1536x1x1xf32>, %s2b6pbv: tensor<384xf32>, %s2b6lgv: tensor<384xf32>, %s2b7dWv: tensor<384x1x7x7xf32>, %s2b7dbv: tensor<384xf32>, %s2b7ngv: tensor<384xf32>, %s2b7nbtv: tensor<384xf32>, %s2b7eWv: tensor<1536x384x1x1xf32>, %s2b7ebv: tensor<1536xf32>, %s2b7pWv: tensor<384x1536x1x1xf32>, %s2b7pbv: tensor<384xf32>, %s2b7lgv: tensor<384xf32>, %s2b8dWv: tensor<384x1x7x7xf32>, %s2b8dbv: tensor<384xf32>, %s2b8ngv: tensor<384xf32>, %s2b8nbtv: tensor<384xf32>, %s2b8eWv: tensor<1536x384x1x1xf32>, %s2b8ebv: tensor<1536xf32>, %s2b8pWv: tensor<384x1536x1x1xf32>, %s2b8pbv: tensor<384xf32>, %s2b8lgv: tensor<384xf32>, %d2ngv: tensor<384xf32>, %d2nbtv: tensor<384xf32>, %d2Wv: tensor<768x384x2x2xf32>, %d2bv: tensor<768xf32>, %s3b0dWv: tensor<768x1x7x7xf32>, %s3b0dbv: tensor<768xf32>, %s3b0ngv: tensor<768xf32>, %s3b0nbtv: tensor<768xf32>, %s3b0eWv: tensor<3072x768x1x1xf32>, %s3b0ebv: tensor<3072xf32>, %s3b0pWv: tensor<768x3072x1x1xf32>, %s3b0pbv: tensor<768xf32>, %s3b0lgv: tensor<768xf32>, %s3b1dWv: tensor<768x1x7x7xf32>, %s3b1dbv: tensor<768xf32>, %s3b1ngv: tensor<768xf32>, %s3b1nbtv: tensor<768xf32>, %s3b1eWv: tensor<3072x768x1x1xf32>, %s3b1ebv: tensor<3072xf32>, %s3b1pWv: tensor<768x3072x1x1xf32>, %s3b1pbv: tensor<768xf32>, %s3b1lgv: tensor<768xf32>, %s3b2dWv: tensor<768x1x7x7xf32>, %s3b2dbv: tensor<768xf32>, %s3b2ngv: tensor<768xf32>, %s3b2nbtv: tensor<768xf32>, %s3b2eWv: tensor<3072x768x1x1xf32>, %s3b2ebv: tensor<3072xf32>, %s3b2pWv: tensor<768x3072x1x1xf32>, %s3b2pbv: tensor<768xf32>, %s3b2lgv: tensor<768xf32>, %Wdv: tensor<768x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %dp0: tensor<32xf32>, %dp1: tensor<32xf32>, %dp2: tensor<32xf32>, %dp3: tensor<32xf32>, %dp4: tensor<32xf32>, %dp5: tensor<32xf32>, %dp6: tensor<32xf32>, %dp7: tensor<32xf32>, %dp8: tensor<32xf32>, %dp9: tensor<32xf32>, %dp10: tensor<32xf32>, %dp11: tensor<32xf32>, %dp12: tensor<32xf32>, %dp13: tensor<32xf32>, %dp14: tensor<32xf32>, %dp15: tensor<32xf32>, %dp16: tensor<32xf32>, %dp17: tensor<32xf32>, %onehot: tensor<32x10xf32>) -> (tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %bsc = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    // §2m: the channel-LN chain normalises with lnRowF at γ=1/β=0 and applies the REAL
    // per-channel affine with rowScaleF/rowBiasF, so these two are its scalar identities.
    %one = stablehlo.constant dense<1.0> : tensor<f32>
    %zero = stablehlo.constant dense<0.0> : tensor<f32>
    // ── ConvNeXt-T AdamW train step: gradients + optimizer are pretty(AST node) ──
    // All 180 params, including the stem 4x4/s4 patchify and the 2x2/s2 downsample
    // WEIGHT GRADIENTS — the two documented gaps, closed 2026-07-28 (new cert
    // flatConvStride4_weight_grad_has_vjp; emit-side odd/even split sWGradGeom).
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
    %v105 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v106 = stablehlo.multiply %v105, %v104 : tensor<32x301056xf32>
    %v107 = stablehlo.add %v106, %v38 : tensor<32x301056xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v109 = stablehlo.convolution(%v108, %s0b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v110 = stablehlo.broadcast_in_dim %s0b1db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v111 = stablehlo.add %v109, %v110 : tensor<32x96x56x56xf32>
    %v112 = stablehlo.reshape %v111 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v113 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v114 = stablehlo.transpose %v113, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v116 = stablehlo.reshape %v115 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v118 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v119 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v120 = stablehlo.reduce(%v116 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v121 = stablehlo.broadcast_in_dim %v120, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v122 = stablehlo.divide %v121, %v118 : tensor<32x3136x96xf32>
    %v123 = stablehlo.subtract %v116, %v122 : tensor<32x3136x96xf32>
    %v124 = stablehlo.multiply %v123, %v123 : tensor<32x3136x96xf32>
    %v125 = stablehlo.reduce(%v124 init: %v117) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v127 = stablehlo.divide %v126, %v118 : tensor<32x3136x96xf32>
    %v128 = stablehlo.add %v127, %v119 : tensor<32x3136x96xf32>
    %v129 = stablehlo.rsqrt %v128 : tensor<32x3136x96xf32>
    %v130 = stablehlo.multiply %v123, %v129 : tensor<32x3136x96xf32>
    %v131 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v132 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v133 = stablehlo.multiply %v130, %v131 : tensor<32x3136x96xf32>
    %v134 = stablehlo.add %v133, %v132 : tensor<32x3136x96xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v137 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v138 = stablehlo.multiply %v136, %v137 : tensor<32x3136x96xf32>
    %v139 = stablehlo.reshape %v138 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v141 = stablehlo.broadcast_in_dim %s0b1nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<32x3136x96xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v145 = stablehlo.transpose %v144, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v146 = stablehlo.reshape %v145 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v147 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v148 = stablehlo.convolution(%v147, %s0b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s0b1eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v150 = stablehlo.add %v148, %v149 : tensor<32x384x56x56xf32>
    %v151 = stablehlo.reshape %v150 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v152 = stablehlo.multiply %v151, %v151 : tensor<32x1204224xf32>
    %v153 = stablehlo.multiply %v152, %v151 : tensor<32x1204224xf32>
    %v154 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v155 = stablehlo.multiply %v154, %v153 : tensor<32x1204224xf32>
    %v156 = stablehlo.add %v151, %v155 : tensor<32x1204224xf32>
    %v157 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v158 = stablehlo.multiply %v157, %v156 : tensor<32x1204224xf32>
    %v159 = stablehlo.tanh %v158 : tensor<32x1204224xf32>
    %v160 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v161 = stablehlo.add %v160, %v159 : tensor<32x1204224xf32>
    %v162 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v163 = stablehlo.multiply %v162, %v151 : tensor<32x1204224xf32>
    %v164 = stablehlo.multiply %v163, %v161 : tensor<32x1204224xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v166 = stablehlo.convolution(%v165, %s0b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %s0b1pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<32x96x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v171 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v172 = stablehlo.multiply %v170, %v171 : tensor<32x96x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v174 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v175 = stablehlo.multiply %v174, %v173 : tensor<32x301056xf32>
    %v176 = stablehlo.add %v175, %v107 : tensor<32x301056xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v178 = stablehlo.convolution(%v177, %s0b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %s0b2db, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v180 = stablehlo.add %v178, %v179 : tensor<32x96x56x56xf32>
    %v181 = stablehlo.reshape %v180 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v183 = stablehlo.transpose %v182, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v184 = stablehlo.reshape %v183 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v187 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v188 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v189 = stablehlo.reduce(%v185 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v190 = stablehlo.broadcast_in_dim %v189, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v191 = stablehlo.divide %v190, %v187 : tensor<32x3136x96xf32>
    %v192 = stablehlo.subtract %v185, %v191 : tensor<32x3136x96xf32>
    %v193 = stablehlo.multiply %v192, %v192 : tensor<32x3136x96xf32>
    %v194 = stablehlo.reduce(%v193 init: %v186) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v195 = stablehlo.broadcast_in_dim %v194, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v196 = stablehlo.divide %v195, %v187 : tensor<32x3136x96xf32>
    %v197 = stablehlo.add %v196, %v188 : tensor<32x3136x96xf32>
    %v198 = stablehlo.rsqrt %v197 : tensor<32x3136x96xf32>
    %v199 = stablehlo.multiply %v192, %v198 : tensor<32x3136x96xf32>
    %v200 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v201 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<32x3136x96xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<32x3136x96xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v205 = stablehlo.reshape %v204 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v206 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v207 = stablehlo.multiply %v205, %v206 : tensor<32x3136x96xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v209 = stablehlo.reshape %v208 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v210 = stablehlo.broadcast_in_dim %s0b2nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v211 = stablehlo.add %v209, %v210 : tensor<32x3136x96xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v214 = stablehlo.transpose %v213, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v215 = stablehlo.reshape %v214 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v217 = stablehlo.convolution(%v216, %s0b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v218 = stablehlo.broadcast_in_dim %s0b2eb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x56x56xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<32x384x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v221 = stablehlo.multiply %v220, %v220 : tensor<32x1204224xf32>
    %v222 = stablehlo.multiply %v221, %v220 : tensor<32x1204224xf32>
    %v223 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v224 = stablehlo.multiply %v223, %v222 : tensor<32x1204224xf32>
    %v225 = stablehlo.add %v220, %v224 : tensor<32x1204224xf32>
    %v226 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v227 = stablehlo.multiply %v226, %v225 : tensor<32x1204224xf32>
    %v228 = stablehlo.tanh %v227 : tensor<32x1204224xf32>
    %v229 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<32x1204224xf32>
    %v231 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v232 = stablehlo.multiply %v231, %v220 : tensor<32x1204224xf32>
    %v233 = stablehlo.multiply %v232, %v230 : tensor<32x1204224xf32>
    %v234 = stablehlo.reshape %v233 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v235 = stablehlo.convolution(%v234, %s0b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s0b2pb, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v237 = stablehlo.add %v235, %v236 : tensor<32x96x56x56xf32>
    %v238 = stablehlo.reshape %v237 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v240 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v241 = stablehlo.multiply %v239, %v240 : tensor<32x96x56x56xf32>
    %v242 = stablehlo.reshape %v241 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v243 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v244 = stablehlo.multiply %v243, %v242 : tensor<32x301056xf32>
    %v245 = stablehlo.add %v244, %v176 : tensor<32x301056xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v247 = stablehlo.transpose %v246, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v248 = stablehlo.reshape %v247 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v251 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v252 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v253 = stablehlo.reduce(%v249 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v255 = stablehlo.divide %v254, %v251 : tensor<32x3136x96xf32>
    %v256 = stablehlo.subtract %v249, %v255 : tensor<32x3136x96xf32>
    %v257 = stablehlo.multiply %v256, %v256 : tensor<32x3136x96xf32>
    %v258 = stablehlo.reduce(%v257 init: %v250) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v259 = stablehlo.broadcast_in_dim %v258, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v260 = stablehlo.divide %v259, %v251 : tensor<32x3136x96xf32>
    %v261 = stablehlo.add %v260, %v252 : tensor<32x3136x96xf32>
    %v262 = stablehlo.rsqrt %v261 : tensor<32x3136x96xf32>
    %v263 = stablehlo.multiply %v256, %v262 : tensor<32x3136x96xf32>
    %v264 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v265 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v266 = stablehlo.multiply %v263, %v264 : tensor<32x3136x96xf32>
    %v267 = stablehlo.add %v266, %v265 : tensor<32x3136x96xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v270 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v271 = stablehlo.multiply %v269, %v270 : tensor<32x3136x96xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v274 = stablehlo.broadcast_in_dim %d0nbt, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v275 = stablehlo.add %v273, %v274 : tensor<32x3136x96xf32>
    %v276 = stablehlo.reshape %v275 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v278 = stablehlo.transpose %v277, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v281 = stablehlo.convolution(%v280, %d0W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<192x96x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v282 = stablehlo.broadcast_in_dim %d0b, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v283 = stablehlo.add %v281, %v282 : tensor<32x192x28x28xf32>
    %v284 = stablehlo.reshape %v283 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v285 = stablehlo.reshape %v284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v286 = stablehlo.convolution(%v285, %s1b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v287 = stablehlo.broadcast_in_dim %s1b0db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v288 = stablehlo.add %v286, %v287 : tensor<32x192x28x28xf32>
    %v289 = stablehlo.reshape %v288 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v291 = stablehlo.transpose %v290, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v296 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v297 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v298 = stablehlo.broadcast_in_dim %v297, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v299 = stablehlo.divide %v298, %v295 : tensor<32x784x192xf32>
    %v300 = stablehlo.subtract %v293, %v299 : tensor<32x784x192xf32>
    %v301 = stablehlo.multiply %v300, %v300 : tensor<32x784x192xf32>
    %v302 = stablehlo.reduce(%v301 init: %v294) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v303 = stablehlo.broadcast_in_dim %v302, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v304 = stablehlo.divide %v303, %v295 : tensor<32x784x192xf32>
    %v305 = stablehlo.add %v304, %v296 : tensor<32x784x192xf32>
    %v306 = stablehlo.rsqrt %v305 : tensor<32x784x192xf32>
    %v307 = stablehlo.multiply %v300, %v306 : tensor<32x784x192xf32>
    %v308 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v309 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v310 = stablehlo.multiply %v307, %v308 : tensor<32x784x192xf32>
    %v311 = stablehlo.add %v310, %v309 : tensor<32x784x192xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v314 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v315 = stablehlo.multiply %v313, %v314 : tensor<32x784x192xf32>
    %v316 = stablehlo.reshape %v315 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v317 = stablehlo.reshape %v316 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b0nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v319 = stablehlo.add %v317, %v318 : tensor<32x784x192xf32>
    %v320 = stablehlo.reshape %v319 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v322 = stablehlo.transpose %v321, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v325 = stablehlo.convolution(%v324, %s1b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v326 = stablehlo.broadcast_in_dim %s1b0eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v327 = stablehlo.add %v325, %v326 : tensor<32x768x28x28xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v329 = stablehlo.multiply %v328, %v328 : tensor<32x602112xf32>
    %v330 = stablehlo.multiply %v329, %v328 : tensor<32x602112xf32>
    %v331 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v332 = stablehlo.multiply %v331, %v330 : tensor<32x602112xf32>
    %v333 = stablehlo.add %v328, %v332 : tensor<32x602112xf32>
    %v334 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v335 = stablehlo.multiply %v334, %v333 : tensor<32x602112xf32>
    %v336 = stablehlo.tanh %v335 : tensor<32x602112xf32>
    %v337 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v338 = stablehlo.add %v337, %v336 : tensor<32x602112xf32>
    %v339 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v340 = stablehlo.multiply %v339, %v328 : tensor<32x602112xf32>
    %v341 = stablehlo.multiply %v340, %v338 : tensor<32x602112xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v343 = stablehlo.convolution(%v342, %s1b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v344 = stablehlo.broadcast_in_dim %s1b0pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v345 = stablehlo.add %v343, %v344 : tensor<32x192x28x28xf32>
    %v346 = stablehlo.reshape %v345 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v347 = stablehlo.reshape %v346 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v349 = stablehlo.multiply %v347, %v348 : tensor<32x192x28x28xf32>
    %v350 = stablehlo.reshape %v349 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v351 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v352 = stablehlo.multiply %v351, %v350 : tensor<32x150528xf32>
    %v353 = stablehlo.add %v352, %v284 : tensor<32x150528xf32>
    %v354 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v355 = stablehlo.convolution(%v354, %s1b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v356 = stablehlo.broadcast_in_dim %s1b1db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v357 = stablehlo.add %v355, %v356 : tensor<32x192x28x28xf32>
    %v358 = stablehlo.reshape %v357 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v360 = stablehlo.transpose %v359, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v364 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v365 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v366 = stablehlo.reduce(%v362 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v367 = stablehlo.broadcast_in_dim %v366, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v368 = stablehlo.divide %v367, %v364 : tensor<32x784x192xf32>
    %v369 = stablehlo.subtract %v362, %v368 : tensor<32x784x192xf32>
    %v370 = stablehlo.multiply %v369, %v369 : tensor<32x784x192xf32>
    %v371 = stablehlo.reduce(%v370 init: %v363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v372 = stablehlo.broadcast_in_dim %v371, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v373 = stablehlo.divide %v372, %v364 : tensor<32x784x192xf32>
    %v374 = stablehlo.add %v373, %v365 : tensor<32x784x192xf32>
    %v375 = stablehlo.rsqrt %v374 : tensor<32x784x192xf32>
    %v376 = stablehlo.multiply %v369, %v375 : tensor<32x784x192xf32>
    %v377 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v378 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v379 = stablehlo.multiply %v376, %v377 : tensor<32x784x192xf32>
    %v380 = stablehlo.add %v379, %v378 : tensor<32x784x192xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v383 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v384 = stablehlo.multiply %v382, %v383 : tensor<32x784x192xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v387 = stablehlo.broadcast_in_dim %s1b1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v388 = stablehlo.add %v386, %v387 : tensor<32x784x192xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v390 = stablehlo.reshape %v389 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v391 = stablehlo.transpose %v390, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v394 = stablehlo.convolution(%v393, %s1b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v395 = stablehlo.broadcast_in_dim %s1b1eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v396 = stablehlo.add %v394, %v395 : tensor<32x768x28x28xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<32x602112xf32>
    %v399 = stablehlo.multiply %v398, %v397 : tensor<32x602112xf32>
    %v400 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v401 = stablehlo.multiply %v400, %v399 : tensor<32x602112xf32>
    %v402 = stablehlo.add %v397, %v401 : tensor<32x602112xf32>
    %v403 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v404 = stablehlo.multiply %v403, %v402 : tensor<32x602112xf32>
    %v405 = stablehlo.tanh %v404 : tensor<32x602112xf32>
    %v406 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v407 = stablehlo.add %v406, %v405 : tensor<32x602112xf32>
    %v408 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v409 = stablehlo.multiply %v408, %v397 : tensor<32x602112xf32>
    %v410 = stablehlo.multiply %v409, %v407 : tensor<32x602112xf32>
    %v411 = stablehlo.reshape %v410 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v412 = stablehlo.convolution(%v411, %s1b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v413 = stablehlo.broadcast_in_dim %s1b1pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v414 = stablehlo.add %v412, %v413 : tensor<32x192x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v417 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v418 = stablehlo.multiply %v416, %v417 : tensor<32x192x28x28xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v420 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v421 = stablehlo.multiply %v420, %v419 : tensor<32x150528xf32>
    %v422 = stablehlo.add %v421, %v353 : tensor<32x150528xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v424 = stablehlo.convolution(%v423, %s1b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v425 = stablehlo.broadcast_in_dim %s1b2db, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v426 = stablehlo.add %v424, %v425 : tensor<32x192x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v429 = stablehlo.transpose %v428, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v434 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v435 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v437 = stablehlo.divide %v436, %v433 : tensor<32x784x192xf32>
    %v438 = stablehlo.subtract %v431, %v437 : tensor<32x784x192xf32>
    %v439 = stablehlo.multiply %v438, %v438 : tensor<32x784x192xf32>
    %v440 = stablehlo.reduce(%v439 init: %v432) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v442 = stablehlo.divide %v441, %v433 : tensor<32x784x192xf32>
    %v443 = stablehlo.add %v442, %v434 : tensor<32x784x192xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<32x784x192xf32>
    %v445 = stablehlo.multiply %v438, %v444 : tensor<32x784x192xf32>
    %v446 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v447 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v448 = stablehlo.multiply %v445, %v446 : tensor<32x784x192xf32>
    %v449 = stablehlo.add %v448, %v447 : tensor<32x784x192xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v452 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v453 = stablehlo.multiply %v451, %v452 : tensor<32x784x192xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v456 = stablehlo.broadcast_in_dim %s1b2nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v457 = stablehlo.add %v455, %v456 : tensor<32x784x192xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v459 = stablehlo.reshape %v458 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v460 = stablehlo.transpose %v459, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v463 = stablehlo.convolution(%v462, %s1b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v464 = stablehlo.broadcast_in_dim %s1b2eb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x28x28xf32>
    %v465 = stablehlo.add %v463, %v464 : tensor<32x768x28x28xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v467 = stablehlo.multiply %v466, %v466 : tensor<32x602112xf32>
    %v468 = stablehlo.multiply %v467, %v466 : tensor<32x602112xf32>
    %v469 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v470 = stablehlo.multiply %v469, %v468 : tensor<32x602112xf32>
    %v471 = stablehlo.add %v466, %v470 : tensor<32x602112xf32>
    %v472 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v473 = stablehlo.multiply %v472, %v471 : tensor<32x602112xf32>
    %v474 = stablehlo.tanh %v473 : tensor<32x602112xf32>
    %v475 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v476 = stablehlo.add %v475, %v474 : tensor<32x602112xf32>
    %v477 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v478 = stablehlo.multiply %v477, %v466 : tensor<32x602112xf32>
    %v479 = stablehlo.multiply %v478, %v476 : tensor<32x602112xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v481 = stablehlo.convolution(%v480, %s1b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %s1b2pb, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v483 = stablehlo.add %v481, %v482 : tensor<32x192x28x28xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v486 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v487 = stablehlo.multiply %v485, %v486 : tensor<32x192x28x28xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v489 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v490 = stablehlo.multiply %v489, %v488 : tensor<32x150528xf32>
    %v491 = stablehlo.add %v490, %v422 : tensor<32x150528xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v493 = stablehlo.transpose %v492, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v497 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v498 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v499 = stablehlo.reduce(%v495 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v500 = stablehlo.broadcast_in_dim %v499, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v501 = stablehlo.divide %v500, %v497 : tensor<32x784x192xf32>
    %v502 = stablehlo.subtract %v495, %v501 : tensor<32x784x192xf32>
    %v503 = stablehlo.multiply %v502, %v502 : tensor<32x784x192xf32>
    %v504 = stablehlo.reduce(%v503 init: %v496) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v505 = stablehlo.broadcast_in_dim %v504, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v506 = stablehlo.divide %v505, %v497 : tensor<32x784x192xf32>
    %v507 = stablehlo.add %v506, %v498 : tensor<32x784x192xf32>
    %v508 = stablehlo.rsqrt %v507 : tensor<32x784x192xf32>
    %v509 = stablehlo.multiply %v502, %v508 : tensor<32x784x192xf32>
    %v510 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v511 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v512 = stablehlo.multiply %v509, %v510 : tensor<32x784x192xf32>
    %v513 = stablehlo.add %v512, %v511 : tensor<32x784x192xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v516 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v517 = stablehlo.multiply %v515, %v516 : tensor<32x784x192xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v520 = stablehlo.broadcast_in_dim %d1nbt, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<32x784x192xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v524 = stablehlo.transpose %v523, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v525 = stablehlo.reshape %v524 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v527 = stablehlo.convolution(%v526, %d1W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<384x192x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v528 = stablehlo.broadcast_in_dim %d1b, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v529 = stablehlo.add %v527, %v528 : tensor<32x384x14x14xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v532 = stablehlo.convolution(%v531, %s2b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v533 = stablehlo.broadcast_in_dim %s2b0db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v534 = stablehlo.add %v532, %v533 : tensor<32x384x14x14xf32>
    %v535 = stablehlo.reshape %v534 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v536 = stablehlo.reshape %v535 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v537 = stablehlo.transpose %v536, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v539 = stablehlo.reshape %v538 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v541 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v542 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v543 = stablehlo.reduce(%v539 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v544 = stablehlo.broadcast_in_dim %v543, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v545 = stablehlo.divide %v544, %v541 : tensor<32x196x384xf32>
    %v546 = stablehlo.subtract %v539, %v545 : tensor<32x196x384xf32>
    %v547 = stablehlo.multiply %v546, %v546 : tensor<32x196x384xf32>
    %v548 = stablehlo.reduce(%v547 init: %v540) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v549 = stablehlo.broadcast_in_dim %v548, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v550 = stablehlo.divide %v549, %v541 : tensor<32x196x384xf32>
    %v551 = stablehlo.add %v550, %v542 : tensor<32x196x384xf32>
    %v552 = stablehlo.rsqrt %v551 : tensor<32x196x384xf32>
    %v553 = stablehlo.multiply %v546, %v552 : tensor<32x196x384xf32>
    %v554 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v555 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v556 = stablehlo.multiply %v553, %v554 : tensor<32x196x384xf32>
    %v557 = stablehlo.add %v556, %v555 : tensor<32x196x384xf32>
    %v558 = stablehlo.reshape %v557 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v559 = stablehlo.reshape %v558 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v560 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v561 = stablehlo.multiply %v559, %v560 : tensor<32x196x384xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v564 = stablehlo.broadcast_in_dim %s2b0nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v565 = stablehlo.add %v563, %v564 : tensor<32x196x384xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v568 = stablehlo.transpose %v567, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v570 = stablehlo.reshape %v569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v571 = stablehlo.convolution(%v570, %s2b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v572 = stablehlo.broadcast_in_dim %s2b0eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v573 = stablehlo.add %v571, %v572 : tensor<32x1536x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v575 = stablehlo.multiply %v574, %v574 : tensor<32x301056xf32>
    %v576 = stablehlo.multiply %v575, %v574 : tensor<32x301056xf32>
    %v577 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v578 = stablehlo.multiply %v577, %v576 : tensor<32x301056xf32>
    %v579 = stablehlo.add %v574, %v578 : tensor<32x301056xf32>
    %v580 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v581 = stablehlo.multiply %v580, %v579 : tensor<32x301056xf32>
    %v582 = stablehlo.tanh %v581 : tensor<32x301056xf32>
    %v583 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v584 = stablehlo.add %v583, %v582 : tensor<32x301056xf32>
    %v585 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v586 = stablehlo.multiply %v585, %v574 : tensor<32x301056xf32>
    %v587 = stablehlo.multiply %v586, %v584 : tensor<32x301056xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v589 = stablehlo.convolution(%v588, %s2b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v590 = stablehlo.broadcast_in_dim %s2b0pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v591 = stablehlo.add %v589, %v590 : tensor<32x384x14x14xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v593 = stablehlo.reshape %v592 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v595 = stablehlo.multiply %v593, %v594 : tensor<32x384x14x14xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v597 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v598 = stablehlo.multiply %v597, %v596 : tensor<32x75264xf32>
    %v599 = stablehlo.add %v598, %v530 : tensor<32x75264xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v601 = stablehlo.convolution(%v600, %s2b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %s2b1db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<32x384x14x14xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v606 = stablehlo.transpose %v605, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v610 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v611 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v612 = stablehlo.reduce(%v608 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v613 = stablehlo.broadcast_in_dim %v612, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v614 = stablehlo.divide %v613, %v610 : tensor<32x196x384xf32>
    %v615 = stablehlo.subtract %v608, %v614 : tensor<32x196x384xf32>
    %v616 = stablehlo.multiply %v615, %v615 : tensor<32x196x384xf32>
    %v617 = stablehlo.reduce(%v616 init: %v609) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v619 = stablehlo.divide %v618, %v610 : tensor<32x196x384xf32>
    %v620 = stablehlo.add %v619, %v611 : tensor<32x196x384xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<32x196x384xf32>
    %v622 = stablehlo.multiply %v615, %v621 : tensor<32x196x384xf32>
    %v623 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v624 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v625 = stablehlo.multiply %v622, %v623 : tensor<32x196x384xf32>
    %v626 = stablehlo.add %v625, %v624 : tensor<32x196x384xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v629 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v630 = stablehlo.multiply %v628, %v629 : tensor<32x196x384xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v633 = stablehlo.broadcast_in_dim %s2b1nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v634 = stablehlo.add %v632, %v633 : tensor<32x196x384xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v637 = stablehlo.transpose %v636, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v638 = stablehlo.reshape %v637 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v640 = stablehlo.convolution(%v639, %s2b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v641 = stablehlo.broadcast_in_dim %s2b1eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<32x1536x14x14xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v644 = stablehlo.multiply %v643, %v643 : tensor<32x301056xf32>
    %v645 = stablehlo.multiply %v644, %v643 : tensor<32x301056xf32>
    %v646 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v647 = stablehlo.multiply %v646, %v645 : tensor<32x301056xf32>
    %v648 = stablehlo.add %v643, %v647 : tensor<32x301056xf32>
    %v649 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v650 = stablehlo.multiply %v649, %v648 : tensor<32x301056xf32>
    %v651 = stablehlo.tanh %v650 : tensor<32x301056xf32>
    %v652 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v653 = stablehlo.add %v652, %v651 : tensor<32x301056xf32>
    %v654 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v655 = stablehlo.multiply %v654, %v643 : tensor<32x301056xf32>
    %v656 = stablehlo.multiply %v655, %v653 : tensor<32x301056xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v658 = stablehlo.convolution(%v657, %s2b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b1pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v660 = stablehlo.add %v658, %v659 : tensor<32x384x14x14xf32>
    %v661 = stablehlo.reshape %v660 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v664 = stablehlo.multiply %v662, %v663 : tensor<32x384x14x14xf32>
    %v665 = stablehlo.reshape %v664 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v666 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v667 = stablehlo.multiply %v666, %v665 : tensor<32x75264xf32>
    %v668 = stablehlo.add %v667, %v599 : tensor<32x75264xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v670 = stablehlo.convolution(%v669, %s2b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v671 = stablehlo.broadcast_in_dim %s2b2db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<32x384x14x14xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v675 = stablehlo.transpose %v674, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v680 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v681 = stablehlo.reduce(%v677 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v683 = stablehlo.divide %v682, %v679 : tensor<32x196x384xf32>
    %v684 = stablehlo.subtract %v677, %v683 : tensor<32x196x384xf32>
    %v685 = stablehlo.multiply %v684, %v684 : tensor<32x196x384xf32>
    %v686 = stablehlo.reduce(%v685 init: %v678) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v687 = stablehlo.broadcast_in_dim %v686, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v688 = stablehlo.divide %v687, %v679 : tensor<32x196x384xf32>
    %v689 = stablehlo.add %v688, %v680 : tensor<32x196x384xf32>
    %v690 = stablehlo.rsqrt %v689 : tensor<32x196x384xf32>
    %v691 = stablehlo.multiply %v684, %v690 : tensor<32x196x384xf32>
    %v692 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v693 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v694 = stablehlo.multiply %v691, %v692 : tensor<32x196x384xf32>
    %v695 = stablehlo.add %v694, %v693 : tensor<32x196x384xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v697 = stablehlo.reshape %v696 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v698 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v699 = stablehlo.multiply %v697, %v698 : tensor<32x196x384xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v702 = stablehlo.broadcast_in_dim %s2b2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v703 = stablehlo.add %v701, %v702 : tensor<32x196x384xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v705 = stablehlo.reshape %v704 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v706 = stablehlo.transpose %v705, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v709 = stablehlo.convolution(%v708, %s2b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v710 = stablehlo.broadcast_in_dim %s2b2eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v711 = stablehlo.add %v709, %v710 : tensor<32x1536x14x14xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v713 = stablehlo.multiply %v712, %v712 : tensor<32x301056xf32>
    %v714 = stablehlo.multiply %v713, %v712 : tensor<32x301056xf32>
    %v715 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v716 = stablehlo.multiply %v715, %v714 : tensor<32x301056xf32>
    %v717 = stablehlo.add %v712, %v716 : tensor<32x301056xf32>
    %v718 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v719 = stablehlo.multiply %v718, %v717 : tensor<32x301056xf32>
    %v720 = stablehlo.tanh %v719 : tensor<32x301056xf32>
    %v721 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v722 = stablehlo.add %v721, %v720 : tensor<32x301056xf32>
    %v723 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v724 = stablehlo.multiply %v723, %v712 : tensor<32x301056xf32>
    %v725 = stablehlo.multiply %v724, %v722 : tensor<32x301056xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v727 = stablehlo.convolution(%v726, %s2b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v728 = stablehlo.broadcast_in_dim %s2b2pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v729 = stablehlo.add %v727, %v728 : tensor<32x384x14x14xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v732 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v733 = stablehlo.multiply %v731, %v732 : tensor<32x384x14x14xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v735 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v736 = stablehlo.multiply %v735, %v734 : tensor<32x75264xf32>
    %v737 = stablehlo.add %v736, %v668 : tensor<32x75264xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v739 = stablehlo.convolution(%v738, %s2b3dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v740 = stablehlo.broadcast_in_dim %s2b3db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v741 = stablehlo.add %v739, %v740 : tensor<32x384x14x14xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v744 = stablehlo.transpose %v743, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v746 = stablehlo.reshape %v745 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v748 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v749 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v750 = stablehlo.reduce(%v746 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v751 = stablehlo.broadcast_in_dim %v750, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v752 = stablehlo.divide %v751, %v748 : tensor<32x196x384xf32>
    %v753 = stablehlo.subtract %v746, %v752 : tensor<32x196x384xf32>
    %v754 = stablehlo.multiply %v753, %v753 : tensor<32x196x384xf32>
    %v755 = stablehlo.reduce(%v754 init: %v747) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v756 = stablehlo.broadcast_in_dim %v755, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v757 = stablehlo.divide %v756, %v748 : tensor<32x196x384xf32>
    %v758 = stablehlo.add %v757, %v749 : tensor<32x196x384xf32>
    %v759 = stablehlo.rsqrt %v758 : tensor<32x196x384xf32>
    %v760 = stablehlo.multiply %v753, %v759 : tensor<32x196x384xf32>
    %v761 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v762 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v763 = stablehlo.multiply %v760, %v761 : tensor<32x196x384xf32>
    %v764 = stablehlo.add %v763, %v762 : tensor<32x196x384xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v767 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v768 = stablehlo.multiply %v766, %v767 : tensor<32x196x384xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v771 = stablehlo.broadcast_in_dim %s2b3nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v772 = stablehlo.add %v770, %v771 : tensor<32x196x384xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v775 = stablehlo.transpose %v774, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v776 = stablehlo.reshape %v775 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v778 = stablehlo.convolution(%v777, %s2b3eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v780 = stablehlo.add %v778, %v779 : tensor<32x1536x14x14xf32>
    %v781 = stablehlo.reshape %v780 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v782 = stablehlo.multiply %v781, %v781 : tensor<32x301056xf32>
    %v783 = stablehlo.multiply %v782, %v781 : tensor<32x301056xf32>
    %v784 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v785 = stablehlo.multiply %v784, %v783 : tensor<32x301056xf32>
    %v786 = stablehlo.add %v781, %v785 : tensor<32x301056xf32>
    %v787 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v788 = stablehlo.multiply %v787, %v786 : tensor<32x301056xf32>
    %v789 = stablehlo.tanh %v788 : tensor<32x301056xf32>
    %v790 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v791 = stablehlo.add %v790, %v789 : tensor<32x301056xf32>
    %v792 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v793 = stablehlo.multiply %v792, %v781 : tensor<32x301056xf32>
    %v794 = stablehlo.multiply %v793, %v791 : tensor<32x301056xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v796 = stablehlo.convolution(%v795, %s2b3pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v797 = stablehlo.broadcast_in_dim %s2b3pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<32x384x14x14xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v801 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v802 = stablehlo.multiply %v800, %v801 : tensor<32x384x14x14xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v804 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v805 = stablehlo.multiply %v804, %v803 : tensor<32x75264xf32>
    %v806 = stablehlo.add %v805, %v737 : tensor<32x75264xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v808 = stablehlo.convolution(%v807, %s2b4dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %s2b4db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v810 = stablehlo.add %v808, %v809 : tensor<32x384x14x14xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v813 = stablehlo.transpose %v812, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v815 = stablehlo.reshape %v814 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v816 = stablehlo.constant dense<0.0> : tensor<f32>
    %v817 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v818 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v819 = stablehlo.reduce(%v815 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v820 = stablehlo.broadcast_in_dim %v819, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v821 = stablehlo.divide %v820, %v817 : tensor<32x196x384xf32>
    %v822 = stablehlo.subtract %v815, %v821 : tensor<32x196x384xf32>
    %v823 = stablehlo.multiply %v822, %v822 : tensor<32x196x384xf32>
    %v824 = stablehlo.reduce(%v823 init: %v816) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v825 = stablehlo.broadcast_in_dim %v824, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v826 = stablehlo.divide %v825, %v817 : tensor<32x196x384xf32>
    %v827 = stablehlo.add %v826, %v818 : tensor<32x196x384xf32>
    %v828 = stablehlo.rsqrt %v827 : tensor<32x196x384xf32>
    %v829 = stablehlo.multiply %v822, %v828 : tensor<32x196x384xf32>
    %v830 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v831 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v832 = stablehlo.multiply %v829, %v830 : tensor<32x196x384xf32>
    %v833 = stablehlo.add %v832, %v831 : tensor<32x196x384xf32>
    %v834 = stablehlo.reshape %v833 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v836 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v837 = stablehlo.multiply %v835, %v836 : tensor<32x196x384xf32>
    %v838 = stablehlo.reshape %v837 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v839 = stablehlo.reshape %v838 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v840 = stablehlo.broadcast_in_dim %s2b4nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v841 = stablehlo.add %v839, %v840 : tensor<32x196x384xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v843 = stablehlo.reshape %v842 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v844 = stablehlo.transpose %v843, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v847 = stablehlo.convolution(%v846, %s2b4eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s2b4eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v849 = stablehlo.add %v847, %v848 : tensor<32x1536x14x14xf32>
    %v850 = stablehlo.reshape %v849 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v851 = stablehlo.multiply %v850, %v850 : tensor<32x301056xf32>
    %v852 = stablehlo.multiply %v851, %v850 : tensor<32x301056xf32>
    %v853 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v854 = stablehlo.multiply %v853, %v852 : tensor<32x301056xf32>
    %v855 = stablehlo.add %v850, %v854 : tensor<32x301056xf32>
    %v856 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v857 = stablehlo.multiply %v856, %v855 : tensor<32x301056xf32>
    %v858 = stablehlo.tanh %v857 : tensor<32x301056xf32>
    %v859 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v860 = stablehlo.add %v859, %v858 : tensor<32x301056xf32>
    %v861 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v862 = stablehlo.multiply %v861, %v850 : tensor<32x301056xf32>
    %v863 = stablehlo.multiply %v862, %v860 : tensor<32x301056xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v865 = stablehlo.convolution(%v864, %s2b4pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %s2b4pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v867 = stablehlo.add %v865, %v866 : tensor<32x384x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v871 = stablehlo.multiply %v869, %v870 : tensor<32x384x14x14xf32>
    %v872 = stablehlo.reshape %v871 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v873 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v874 = stablehlo.multiply %v873, %v872 : tensor<32x75264xf32>
    %v875 = stablehlo.add %v874, %v806 : tensor<32x75264xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v877 = stablehlo.convolution(%v876, %s2b5dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v878 = stablehlo.broadcast_in_dim %s2b5db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v879 = stablehlo.add %v877, %v878 : tensor<32x384x14x14xf32>
    %v880 = stablehlo.reshape %v879 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v882 = stablehlo.transpose %v881, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v887 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v888 = stablehlo.reduce(%v884 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v889 = stablehlo.broadcast_in_dim %v888, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v890 = stablehlo.divide %v889, %v886 : tensor<32x196x384xf32>
    %v891 = stablehlo.subtract %v884, %v890 : tensor<32x196x384xf32>
    %v892 = stablehlo.multiply %v891, %v891 : tensor<32x196x384xf32>
    %v893 = stablehlo.reduce(%v892 init: %v885) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v895 = stablehlo.divide %v894, %v886 : tensor<32x196x384xf32>
    %v896 = stablehlo.add %v895, %v887 : tensor<32x196x384xf32>
    %v897 = stablehlo.rsqrt %v896 : tensor<32x196x384xf32>
    %v898 = stablehlo.multiply %v891, %v897 : tensor<32x196x384xf32>
    %v899 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v900 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v901 = stablehlo.multiply %v898, %v899 : tensor<32x196x384xf32>
    %v902 = stablehlo.add %v901, %v900 : tensor<32x196x384xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v904 = stablehlo.reshape %v903 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v905 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v906 = stablehlo.multiply %v904, %v905 : tensor<32x196x384xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v909 = stablehlo.broadcast_in_dim %s2b5nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v910 = stablehlo.add %v908, %v909 : tensor<32x196x384xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v912 = stablehlo.reshape %v911 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v913 = stablehlo.transpose %v912, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v916 = stablehlo.convolution(%v915, %s2b5eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v917 = stablehlo.broadcast_in_dim %s2b5eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v918 = stablehlo.add %v916, %v917 : tensor<32x1536x14x14xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v920 = stablehlo.multiply %v919, %v919 : tensor<32x301056xf32>
    %v921 = stablehlo.multiply %v920, %v919 : tensor<32x301056xf32>
    %v922 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v923 = stablehlo.multiply %v922, %v921 : tensor<32x301056xf32>
    %v924 = stablehlo.add %v919, %v923 : tensor<32x301056xf32>
    %v925 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v926 = stablehlo.multiply %v925, %v924 : tensor<32x301056xf32>
    %v927 = stablehlo.tanh %v926 : tensor<32x301056xf32>
    %v928 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v929 = stablehlo.add %v928, %v927 : tensor<32x301056xf32>
    %v930 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v931 = stablehlo.multiply %v930, %v919 : tensor<32x301056xf32>
    %v932 = stablehlo.multiply %v931, %v929 : tensor<32x301056xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v934 = stablehlo.convolution(%v933, %s2b5pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v935 = stablehlo.broadcast_in_dim %s2b5pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v936 = stablehlo.add %v934, %v935 : tensor<32x384x14x14xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v938 = stablehlo.reshape %v937 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v939 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v940 = stablehlo.multiply %v938, %v939 : tensor<32x384x14x14xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v942 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v943 = stablehlo.multiply %v942, %v941 : tensor<32x75264xf32>
    %v944 = stablehlo.add %v943, %v875 : tensor<32x75264xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v946 = stablehlo.convolution(%v945, %s2b6dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s2b6db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v948 = stablehlo.add %v946, %v947 : tensor<32x384x14x14xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v951 = stablehlo.transpose %v950, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v955 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v956 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v957 = stablehlo.reduce(%v953 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v958 = stablehlo.broadcast_in_dim %v957, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v959 = stablehlo.divide %v958, %v955 : tensor<32x196x384xf32>
    %v960 = stablehlo.subtract %v953, %v959 : tensor<32x196x384xf32>
    %v961 = stablehlo.multiply %v960, %v960 : tensor<32x196x384xf32>
    %v962 = stablehlo.reduce(%v961 init: %v954) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v964 = stablehlo.divide %v963, %v955 : tensor<32x196x384xf32>
    %v965 = stablehlo.add %v964, %v956 : tensor<32x196x384xf32>
    %v966 = stablehlo.rsqrt %v965 : tensor<32x196x384xf32>
    %v967 = stablehlo.multiply %v960, %v966 : tensor<32x196x384xf32>
    %v968 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v969 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v970 = stablehlo.multiply %v967, %v968 : tensor<32x196x384xf32>
    %v971 = stablehlo.add %v970, %v969 : tensor<32x196x384xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v974 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v975 = stablehlo.multiply %v973, %v974 : tensor<32x196x384xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v978 = stablehlo.broadcast_in_dim %s2b6nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v979 = stablehlo.add %v977, %v978 : tensor<32x196x384xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v982 = stablehlo.transpose %v981, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v985 = stablehlo.convolution(%v984, %s2b6eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v986 = stablehlo.broadcast_in_dim %s2b6eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v987 = stablehlo.add %v985, %v986 : tensor<32x1536x14x14xf32>
    %v988 = stablehlo.reshape %v987 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v989 = stablehlo.multiply %v988, %v988 : tensor<32x301056xf32>
    %v990 = stablehlo.multiply %v989, %v988 : tensor<32x301056xf32>
    %v991 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v992 = stablehlo.multiply %v991, %v990 : tensor<32x301056xf32>
    %v993 = stablehlo.add %v988, %v992 : tensor<32x301056xf32>
    %v994 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v995 = stablehlo.multiply %v994, %v993 : tensor<32x301056xf32>
    %v996 = stablehlo.tanh %v995 : tensor<32x301056xf32>
    %v997 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<32x301056xf32>
    %v999 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1000 = stablehlo.multiply %v999, %v988 : tensor<32x301056xf32>
    %v1001 = stablehlo.multiply %v1000, %v998 : tensor<32x301056xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1003 = stablehlo.convolution(%v1002, %s2b6pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b6pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1005 = stablehlo.add %v1003, %v1004 : tensor<32x384x14x14xf32>
    %v1006 = stablehlo.reshape %v1005 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1008 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1009 = stablehlo.multiply %v1007, %v1008 : tensor<32x384x14x14xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1011 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1012 = stablehlo.multiply %v1011, %v1010 : tensor<32x75264xf32>
    %v1013 = stablehlo.add %v1012, %v944 : tensor<32x75264xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1015 = stablehlo.convolution(%v1014, %s2b7dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1016 = stablehlo.broadcast_in_dim %s2b7db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1017 = stablehlo.add %v1015, %v1016 : tensor<32x384x14x14xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1020 = stablehlo.transpose %v1019, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1024 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1025 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1026 = stablehlo.reduce(%v1022 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1027 = stablehlo.broadcast_in_dim %v1026, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1028 = stablehlo.divide %v1027, %v1024 : tensor<32x196x384xf32>
    %v1029 = stablehlo.subtract %v1022, %v1028 : tensor<32x196x384xf32>
    %v1030 = stablehlo.multiply %v1029, %v1029 : tensor<32x196x384xf32>
    %v1031 = stablehlo.reduce(%v1030 init: %v1023) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1032 = stablehlo.broadcast_in_dim %v1031, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1033 = stablehlo.divide %v1032, %v1024 : tensor<32x196x384xf32>
    %v1034 = stablehlo.add %v1033, %v1025 : tensor<32x196x384xf32>
    %v1035 = stablehlo.rsqrt %v1034 : tensor<32x196x384xf32>
    %v1036 = stablehlo.multiply %v1029, %v1035 : tensor<32x196x384xf32>
    %v1037 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1038 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1039 = stablehlo.multiply %v1036, %v1037 : tensor<32x196x384xf32>
    %v1040 = stablehlo.add %v1039, %v1038 : tensor<32x196x384xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1043 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1044 = stablehlo.multiply %v1042, %v1043 : tensor<32x196x384xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1047 = stablehlo.broadcast_in_dim %s2b7nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1048 = stablehlo.add %v1046, %v1047 : tensor<32x196x384xf32>
    %v1049 = stablehlo.reshape %v1048 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1050 = stablehlo.reshape %v1049 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1051 = stablehlo.transpose %v1050, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1054 = stablehlo.convolution(%v1053, %s2b7eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s2b7eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<32x1536x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1058 = stablehlo.multiply %v1057, %v1057 : tensor<32x301056xf32>
    %v1059 = stablehlo.multiply %v1058, %v1057 : tensor<32x301056xf32>
    %v1060 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1061 = stablehlo.multiply %v1060, %v1059 : tensor<32x301056xf32>
    %v1062 = stablehlo.add %v1057, %v1061 : tensor<32x301056xf32>
    %v1063 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1064 = stablehlo.multiply %v1063, %v1062 : tensor<32x301056xf32>
    %v1065 = stablehlo.tanh %v1064 : tensor<32x301056xf32>
    %v1066 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1067 = stablehlo.add %v1066, %v1065 : tensor<32x301056xf32>
    %v1068 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1069 = stablehlo.multiply %v1068, %v1057 : tensor<32x301056xf32>
    %v1070 = stablehlo.multiply %v1069, %v1067 : tensor<32x301056xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1072 = stablehlo.convolution(%v1071, %s2b7pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1073 = stablehlo.broadcast_in_dim %s2b7pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1074 = stablehlo.add %v1072, %v1073 : tensor<32x384x14x14xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1077 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1078 = stablehlo.multiply %v1076, %v1077 : tensor<32x384x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1080 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1081 = stablehlo.multiply %v1080, %v1079 : tensor<32x75264xf32>
    %v1082 = stablehlo.add %v1081, %v1013 : tensor<32x75264xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1084 = stablehlo.convolution(%v1083, %s2b8dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v1085 = stablehlo.broadcast_in_dim %s2b8db, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1086 = stablehlo.add %v1084, %v1085 : tensor<32x384x14x14xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1089 = stablehlo.transpose %v1088, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1093 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1094 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1095 = stablehlo.reduce(%v1091 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1096 = stablehlo.broadcast_in_dim %v1095, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1097 = stablehlo.divide %v1096, %v1093 : tensor<32x196x384xf32>
    %v1098 = stablehlo.subtract %v1091, %v1097 : tensor<32x196x384xf32>
    %v1099 = stablehlo.multiply %v1098, %v1098 : tensor<32x196x384xf32>
    %v1100 = stablehlo.reduce(%v1099 init: %v1092) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1102 = stablehlo.divide %v1101, %v1093 : tensor<32x196x384xf32>
    %v1103 = stablehlo.add %v1102, %v1094 : tensor<32x196x384xf32>
    %v1104 = stablehlo.rsqrt %v1103 : tensor<32x196x384xf32>
    %v1105 = stablehlo.multiply %v1098, %v1104 : tensor<32x196x384xf32>
    %v1106 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1107 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1108 = stablehlo.multiply %v1105, %v1106 : tensor<32x196x384xf32>
    %v1109 = stablehlo.add %v1108, %v1107 : tensor<32x196x384xf32>
    %v1110 = stablehlo.reshape %v1109 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1112 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1113 = stablehlo.multiply %v1111, %v1112 : tensor<32x196x384xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1116 = stablehlo.broadcast_in_dim %s2b8nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1117 = stablehlo.add %v1115, %v1116 : tensor<32x196x384xf32>
    %v1118 = stablehlo.reshape %v1117 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1120 = stablehlo.transpose %v1119, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1123 = stablehlo.convolution(%v1122, %s2b8eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %s2b8eb, dims = [1] : (tensor<1536xf32>) -> tensor<32x1536x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<32x1536x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1127 = stablehlo.multiply %v1126, %v1126 : tensor<32x301056xf32>
    %v1128 = stablehlo.multiply %v1127, %v1126 : tensor<32x301056xf32>
    %v1129 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1130 = stablehlo.multiply %v1129, %v1128 : tensor<32x301056xf32>
    %v1131 = stablehlo.add %v1126, %v1130 : tensor<32x301056xf32>
    %v1132 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1133 = stablehlo.multiply %v1132, %v1131 : tensor<32x301056xf32>
    %v1134 = stablehlo.tanh %v1133 : tensor<32x301056xf32>
    %v1135 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1136 = stablehlo.add %v1135, %v1134 : tensor<32x301056xf32>
    %v1137 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1138 = stablehlo.multiply %v1137, %v1126 : tensor<32x301056xf32>
    %v1139 = stablehlo.multiply %v1138, %v1136 : tensor<32x301056xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1141 = stablehlo.convolution(%v1140, %s2b8pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %s2b8pb, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1143 = stablehlo.add %v1141, %v1142 : tensor<32x384x14x14xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1145 = stablehlo.reshape %v1144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<32x384x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1149 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1150 = stablehlo.multiply %v1149, %v1148 : tensor<32x75264xf32>
    %v1151 = stablehlo.add %v1150, %v1082 : tensor<32x75264xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1153 = stablehlo.transpose %v1152, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1154 = stablehlo.reshape %v1153 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1157 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1158 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1159 = stablehlo.reduce(%v1155 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1160 = stablehlo.broadcast_in_dim %v1159, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1161 = stablehlo.divide %v1160, %v1157 : tensor<32x196x384xf32>
    %v1162 = stablehlo.subtract %v1155, %v1161 : tensor<32x196x384xf32>
    %v1163 = stablehlo.multiply %v1162, %v1162 : tensor<32x196x384xf32>
    %v1164 = stablehlo.reduce(%v1163 init: %v1156) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1166 = stablehlo.divide %v1165, %v1157 : tensor<32x196x384xf32>
    %v1167 = stablehlo.add %v1166, %v1158 : tensor<32x196x384xf32>
    %v1168 = stablehlo.rsqrt %v1167 : tensor<32x196x384xf32>
    %v1169 = stablehlo.multiply %v1162, %v1168 : tensor<32x196x384xf32>
    %v1170 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1171 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1172 = stablehlo.multiply %v1169, %v1170 : tensor<32x196x384xf32>
    %v1173 = stablehlo.add %v1172, %v1171 : tensor<32x196x384xf32>
    %v1174 = stablehlo.reshape %v1173 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1176 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1177 = stablehlo.multiply %v1175, %v1176 : tensor<32x196x384xf32>
    %v1178 = stablehlo.reshape %v1177 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1180 = stablehlo.broadcast_in_dim %d2nbt, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1181 = stablehlo.add %v1179, %v1180 : tensor<32x196x384xf32>
    %v1182 = stablehlo.reshape %v1181 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1184 = stablehlo.transpose %v1183, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1185 = stablehlo.reshape %v1184 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1187 = stablehlo.convolution(%v1186, %d2W)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<768x384x2x2xf32>) -> tensor<32x768x7x7xf32>
    %v1188 = stablehlo.broadcast_in_dim %d2b, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<32x768x7x7xf32>
    %v1190 = stablehlo.reshape %v1189 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1192 = stablehlo.convolution(%v1191, %s3b0dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1193 = stablehlo.broadcast_in_dim %s3b0db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<32x768x7x7xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1197 = stablehlo.transpose %v1196, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1198 = stablehlo.reshape %v1197 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1201 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1202 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1203 = stablehlo.reduce(%v1199 init: %v1200) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1204 = stablehlo.broadcast_in_dim %v1203, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1205 = stablehlo.divide %v1204, %v1201 : tensor<32x49x768xf32>
    %v1206 = stablehlo.subtract %v1199, %v1205 : tensor<32x49x768xf32>
    %v1207 = stablehlo.multiply %v1206, %v1206 : tensor<32x49x768xf32>
    %v1208 = stablehlo.reduce(%v1207 init: %v1200) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1210 = stablehlo.divide %v1209, %v1201 : tensor<32x49x768xf32>
    %v1211 = stablehlo.add %v1210, %v1202 : tensor<32x49x768xf32>
    %v1212 = stablehlo.rsqrt %v1211 : tensor<32x49x768xf32>
    %v1213 = stablehlo.multiply %v1206, %v1212 : tensor<32x49x768xf32>
    %v1214 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1215 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1216 = stablehlo.multiply %v1213, %v1214 : tensor<32x49x768xf32>
    %v1217 = stablehlo.add %v1216, %v1215 : tensor<32x49x768xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1221 = stablehlo.multiply %v1219, %v1220 : tensor<32x49x768xf32>
    %v1222 = stablehlo.reshape %v1221 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1224 = stablehlo.broadcast_in_dim %s3b0nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1225 = stablehlo.add %v1223, %v1224 : tensor<32x49x768xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1228 = stablehlo.transpose %v1227, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1231 = stablehlo.convolution(%v1230, %s3b0eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1232 = stablehlo.broadcast_in_dim %s3b0eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1233 = stablehlo.add %v1231, %v1232 : tensor<32x3072x7x7xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1235 = stablehlo.multiply %v1234, %v1234 : tensor<32x150528xf32>
    %v1236 = stablehlo.multiply %v1235, %v1234 : tensor<32x150528xf32>
    %v1237 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1238 = stablehlo.multiply %v1237, %v1236 : tensor<32x150528xf32>
    %v1239 = stablehlo.add %v1234, %v1238 : tensor<32x150528xf32>
    %v1240 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1241 = stablehlo.multiply %v1240, %v1239 : tensor<32x150528xf32>
    %v1242 = stablehlo.tanh %v1241 : tensor<32x150528xf32>
    %v1243 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1244 = stablehlo.add %v1243, %v1242 : tensor<32x150528xf32>
    %v1245 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1246 = stablehlo.multiply %v1245, %v1234 : tensor<32x150528xf32>
    %v1247 = stablehlo.multiply %v1246, %v1244 : tensor<32x150528xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1249 = stablehlo.convolution(%v1248, %s3b0pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1250 = stablehlo.broadcast_in_dim %s3b0pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<32x768x7x7xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1254 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1255 = stablehlo.multiply %v1253, %v1254 : tensor<32x768x7x7xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1257 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1258 = stablehlo.multiply %v1257, %v1256 : tensor<32x37632xf32>
    %v1259 = stablehlo.add %v1258, %v1190 : tensor<32x37632xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1261 = stablehlo.convolution(%v1260, %s3b1dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1262 = stablehlo.broadcast_in_dim %s3b1db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1263 = stablehlo.add %v1261, %v1262 : tensor<32x768x7x7xf32>
    %v1264 = stablehlo.reshape %v1263 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1266 = stablehlo.transpose %v1265, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1268 = stablehlo.reshape %v1267 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1269 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1270 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1271 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1272 = stablehlo.reduce(%v1268 init: %v1269) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1273 = stablehlo.broadcast_in_dim %v1272, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1274 = stablehlo.divide %v1273, %v1270 : tensor<32x49x768xf32>
    %v1275 = stablehlo.subtract %v1268, %v1274 : tensor<32x49x768xf32>
    %v1276 = stablehlo.multiply %v1275, %v1275 : tensor<32x49x768xf32>
    %v1277 = stablehlo.reduce(%v1276 init: %v1269) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1278 = stablehlo.broadcast_in_dim %v1277, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1279 = stablehlo.divide %v1278, %v1270 : tensor<32x49x768xf32>
    %v1280 = stablehlo.add %v1279, %v1271 : tensor<32x49x768xf32>
    %v1281 = stablehlo.rsqrt %v1280 : tensor<32x49x768xf32>
    %v1282 = stablehlo.multiply %v1275, %v1281 : tensor<32x49x768xf32>
    %v1283 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1284 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1285 = stablehlo.multiply %v1282, %v1283 : tensor<32x49x768xf32>
    %v1286 = stablehlo.add %v1285, %v1284 : tensor<32x49x768xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1288 = stablehlo.reshape %v1287 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1289 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<32x49x768xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1293 = stablehlo.broadcast_in_dim %s3b1nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<32x49x768xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1297 = stablehlo.transpose %v1296, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1298 = stablehlo.reshape %v1297 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1300 = stablehlo.convolution(%v1299, %s3b1eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1301 = stablehlo.broadcast_in_dim %s3b1eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1302 = stablehlo.add %v1300, %v1301 : tensor<32x3072x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<32x150528xf32>
    %v1305 = stablehlo.multiply %v1304, %v1303 : tensor<32x150528xf32>
    %v1306 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1307 = stablehlo.multiply %v1306, %v1305 : tensor<32x150528xf32>
    %v1308 = stablehlo.add %v1303, %v1307 : tensor<32x150528xf32>
    %v1309 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1310 = stablehlo.multiply %v1309, %v1308 : tensor<32x150528xf32>
    %v1311 = stablehlo.tanh %v1310 : tensor<32x150528xf32>
    %v1312 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1313 = stablehlo.add %v1312, %v1311 : tensor<32x150528xf32>
    %v1314 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1315 = stablehlo.multiply %v1314, %v1303 : tensor<32x150528xf32>
    %v1316 = stablehlo.multiply %v1315, %v1313 : tensor<32x150528xf32>
    %v1317 = stablehlo.reshape %v1316 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1318 = stablehlo.convolution(%v1317, %s3b1pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1319 = stablehlo.broadcast_in_dim %s3b1pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1320 = stablehlo.add %v1318, %v1319 : tensor<32x768x7x7xf32>
    %v1321 = stablehlo.reshape %v1320 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1322 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1323 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1324 = stablehlo.multiply %v1322, %v1323 : tensor<32x768x7x7xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1326 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1327 = stablehlo.multiply %v1326, %v1325 : tensor<32x37632xf32>
    %v1328 = stablehlo.add %v1327, %v1259 : tensor<32x37632xf32>
    %v1329 = stablehlo.reshape %v1328 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1330 = stablehlo.convolution(%v1329, %s3b2dW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1331 = stablehlo.broadcast_in_dim %s3b2db, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1332 = stablehlo.add %v1330, %v1331 : tensor<32x768x7x7xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1334 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1335 = stablehlo.transpose %v1334, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1338 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1339 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1340 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1341 = stablehlo.reduce(%v1337 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1342 = stablehlo.broadcast_in_dim %v1341, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1343 = stablehlo.divide %v1342, %v1339 : tensor<32x49x768xf32>
    %v1344 = stablehlo.subtract %v1337, %v1343 : tensor<32x49x768xf32>
    %v1345 = stablehlo.multiply %v1344, %v1344 : tensor<32x49x768xf32>
    %v1346 = stablehlo.reduce(%v1345 init: %v1338) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1347 = stablehlo.broadcast_in_dim %v1346, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1348 = stablehlo.divide %v1347, %v1339 : tensor<32x49x768xf32>
    %v1349 = stablehlo.add %v1348, %v1340 : tensor<32x49x768xf32>
    %v1350 = stablehlo.rsqrt %v1349 : tensor<32x49x768xf32>
    %v1351 = stablehlo.multiply %v1344, %v1350 : tensor<32x49x768xf32>
    %v1352 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1353 = stablehlo.broadcast_in_dim %zero, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1354 = stablehlo.multiply %v1351, %v1352 : tensor<32x49x768xf32>
    %v1355 = stablehlo.add %v1354, %v1353 : tensor<32x49x768xf32>
    %v1356 = stablehlo.reshape %v1355 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1358 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1359 = stablehlo.multiply %v1357, %v1358 : tensor<32x49x768xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1362 = stablehlo.broadcast_in_dim %s3b2nbt, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1363 = stablehlo.add %v1361, %v1362 : tensor<32x49x768xf32>
    %v1364 = stablehlo.reshape %v1363 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1366 = stablehlo.transpose %v1365, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1367 = stablehlo.reshape %v1366 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1369 = stablehlo.convolution(%v1368, %s3b2eW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1370 = stablehlo.broadcast_in_dim %s3b2eb, dims = [1] : (tensor<3072xf32>) -> tensor<32x3072x7x7xf32>
    %v1371 = stablehlo.add %v1369, %v1370 : tensor<32x3072x7x7xf32>
    %v1372 = stablehlo.reshape %v1371 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1373 = stablehlo.multiply %v1372, %v1372 : tensor<32x150528xf32>
    %v1374 = stablehlo.multiply %v1373, %v1372 : tensor<32x150528xf32>
    %v1375 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1376 = stablehlo.multiply %v1375, %v1374 : tensor<32x150528xf32>
    %v1377 = stablehlo.add %v1372, %v1376 : tensor<32x150528xf32>
    %v1378 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1379 = stablehlo.multiply %v1378, %v1377 : tensor<32x150528xf32>
    %v1380 = stablehlo.tanh %v1379 : tensor<32x150528xf32>
    %v1381 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1382 = stablehlo.add %v1381, %v1380 : tensor<32x150528xf32>
    %v1383 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1384 = stablehlo.multiply %v1383, %v1372 : tensor<32x150528xf32>
    %v1385 = stablehlo.multiply %v1384, %v1382 : tensor<32x150528xf32>
    %v1386 = stablehlo.reshape %v1385 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1387 = stablehlo.convolution(%v1386, %s3b2pW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1388 = stablehlo.broadcast_in_dim %s3b2pb, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1389 = stablehlo.add %v1387, %v1388 : tensor<32x768x7x7xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1392 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1393 = stablehlo.multiply %v1391, %v1392 : tensor<32x768x7x7xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1395 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1396 = stablehlo.multiply %v1395, %v1394 : tensor<32x37632xf32>
    %v1397 = stablehlo.add %v1396, %v1328 : tensor<32x37632xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1400 = stablehlo.reduce(%v1398 init: %v1399) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768xf32>
    %v1401 = stablehlo.constant dense<49.0> : tensor<32x768xf32>
    %v1402 = stablehlo.divide %v1400, %v1401 : tensor<32x768xf32>
    %v1403 = stablehlo.dot_general %v1402, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<768x10xf32>) -> tensor<32x10xf32>
    %v1404 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<10xf32>) -> tensor<32x10xf32>
    %v1405 = stablehlo.add %v1403, %v1404 : tensor<32x10xf32>
    %v1406 = stablehlo.exponential %v1405 : tensor<32x10xf32>
    %v1407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1408 = stablehlo.reduce(%v1406 init: %v1407) applies stablehlo.add across dimensions = [1] : (tensor<32x10xf32>, tensor<f32>) -> tensor<32xf32>
    %v1409 = stablehlo.broadcast_in_dim %v1408, dims = [0] : (tensor<32xf32>) -> tensor<32x10xf32>
    %v1410 = stablehlo.divide %v1406, %v1409 : tensor<32x10xf32>
    %v1411 = stablehlo.subtract %v1410, %onehot : tensor<32x10xf32>
    %v1412 = stablehlo.constant dense<0.100000> : tensor<32x10xf32>
    %v1413 = stablehlo.multiply %onehot, %v1412 : tensor<32x10xf32>
    %v1414 = stablehlo.add %v1411, %v1413 : tensor<32x10xf32>
    %v1415 = stablehlo.constant dense<-0.010000> : tensor<32x10xf32>
    %v1416 = stablehlo.add %v1414, %v1415 : tensor<32x10xf32>
    %v1417 = stablehlo.constant dense<32.0> : tensor<32x10xf32>
    %v1418 = stablehlo.divide %v1416, %v1417 : tensor<32x10xf32>
    %v1419 = stablehlo.dot_general %v1418, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<768x10xf32>) -> tensor<32x768xf32>
    %v1420 = stablehlo.dot_general %v1402, %v1418, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x768xf32>, tensor<32x10xf32>) -> tensor<768x10xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.reduce(%v1418 init: %v1421) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgi = stablehlo.reshape %v1419 : (tensor<32x768xf32>) -> tensor<32x768x1x1xf32>
    %dgb = stablehlo.broadcast_in_dim %dgi, dims = [0, 1, 2, 3] : (tensor<32x768x1x1xf32>) -> tensor<32x768x7x7xf32>
    %dgn = stablehlo.constant dense<49.0> : tensor<32x768x7x7xf32>
    %dgd = stablehlo.divide %dgb, %dgn : tensor<32x768x7x7xf32>
    %dgapf = stablehlo.reshape %dgd : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1423 = stablehlo.broadcast_in_dim %dp17, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1424 = stablehlo.multiply %v1423, %dgapf : tensor<32x37632xf32>
    %v1425 = stablehlo.reshape %v1424 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1426 = stablehlo.broadcast_in_dim %s3b2lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1427 = stablehlo.multiply %v1425, %v1426 : tensor<32x768x7x7xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1430 = stablehlo.reverse %s3b2pW, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1431 = stablehlo.transpose %v1430, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1432 = stablehlo.convolution(%v1429, %v1431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1434 = stablehlo.multiply %v1372, %v1372 : tensor<32x150528xf32>
    %v1435 = stablehlo.multiply %v1434, %v1372 : tensor<32x150528xf32>
    %v1436 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1437 = stablehlo.multiply %v1436, %v1435 : tensor<32x150528xf32>
    %v1438 = stablehlo.add %v1372, %v1437 : tensor<32x150528xf32>
    %v1439 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1440 = stablehlo.multiply %v1439, %v1438 : tensor<32x150528xf32>
    %v1441 = stablehlo.tanh %v1440 : tensor<32x150528xf32>
    %v1442 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1443 = stablehlo.add %v1442, %v1441 : tensor<32x150528xf32>
    %v1444 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1445 = stablehlo.multiply %v1444, %v1443 : tensor<32x150528xf32>
    %v1446 = stablehlo.multiply %v1441, %v1441 : tensor<32x150528xf32>
    %v1447 = stablehlo.subtract %v1442, %v1446 : tensor<32x150528xf32>
    %v1448 = stablehlo.multiply %v1444, %v1372 : tensor<32x150528xf32>
    %v1449 = stablehlo.multiply %v1448, %v1447 : tensor<32x150528xf32>
    %v1450 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1451 = stablehlo.multiply %v1450, %v1434 : tensor<32x150528xf32>
    %v1452 = stablehlo.add %v1442, %v1451 : tensor<32x150528xf32>
    %v1453 = stablehlo.multiply %v1439, %v1452 : tensor<32x150528xf32>
    %v1454 = stablehlo.multiply %v1449, %v1453 : tensor<32x150528xf32>
    %v1455 = stablehlo.add %v1445, %v1454 : tensor<32x150528xf32>
    %v1456 = stablehlo.multiply %v1433, %v1455 : tensor<32x150528xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1458 = stablehlo.reverse %s3b2eW, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1459 = stablehlo.transpose %v1458, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1460 = stablehlo.convolution(%v1457, %v1459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1462 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1463 = stablehlo.transpose %v1462, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1465 = stablehlo.reshape %v1461 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1466 = stablehlo.transpose %v1465, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1467 = stablehlo.reshape %v1466 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1469 = stablehlo.broadcast_in_dim %s3b2ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1470 = stablehlo.multiply %v1468, %v1469 : tensor<32x49x768xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1472 = stablehlo.reshape %v1471 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1473 = stablehlo.reshape %v1464 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1475 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1476 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1477 = stablehlo.reduce(%v1473 init: %v1474) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1478 = stablehlo.broadcast_in_dim %v1477, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1479 = stablehlo.divide %v1478, %v1475 : tensor<32x49x768xf32>
    %v1480 = stablehlo.subtract %v1473, %v1479 : tensor<32x49x768xf32>
    %v1481 = stablehlo.multiply %v1480, %v1480 : tensor<32x49x768xf32>
    %v1482 = stablehlo.reduce(%v1481 init: %v1474) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1483 = stablehlo.broadcast_in_dim %v1482, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1484 = stablehlo.divide %v1483, %v1475 : tensor<32x49x768xf32>
    %v1485 = stablehlo.add %v1484, %v1476 : tensor<32x49x768xf32>
    %v1486 = stablehlo.rsqrt %v1485 : tensor<32x49x768xf32>
    %v1487 = stablehlo.multiply %v1480, %v1486 : tensor<32x49x768xf32>
    %v1488 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1489 = stablehlo.multiply %v1488, %v1472 : tensor<32x49x768xf32>
    %v1490 = stablehlo.reduce(%v1489 init: %v1474) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1491 = stablehlo.broadcast_in_dim %v1490, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1492 = stablehlo.multiply %v1487, %v1489 : tensor<32x49x768xf32>
    %v1493 = stablehlo.reduce(%v1492 init: %v1474) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1494 = stablehlo.broadcast_in_dim %v1493, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1495 = stablehlo.multiply %v1489, %v1475 : tensor<32x49x768xf32>
    %v1496 = stablehlo.subtract %v1495, %v1491 : tensor<32x49x768xf32>
    %v1497 = stablehlo.multiply %v1487, %v1494 : tensor<32x49x768xf32>
    %v1498 = stablehlo.subtract %v1496, %v1497 : tensor<32x49x768xf32>
    %v1499 = stablehlo.divide %v1486, %v1475 : tensor<32x49x768xf32>
    %v1500 = stablehlo.multiply %v1499, %v1498 : tensor<32x49x768xf32>
    %v1501 = stablehlo.reshape %v1500 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1503 = stablehlo.transpose %v1502, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1504 = stablehlo.reshape %v1503 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1505 = stablehlo.reshape %v1504 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1506 = stablehlo.reverse %s3b2dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1507 = stablehlo.convolution(%v1505, %v1506)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1509 = stablehlo.add %v1508, %dgapf : tensor<32x37632xf32>
    %v1510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1511 = stablehlo.reshape %v1390 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1512 = stablehlo.reshape %v1424 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1513 = stablehlo.multiply %v1511, %v1512 : tensor<32x768x7x7xf32>
    %v1514 = stablehlo.reduce(%v1513 init: %v1510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1515 = stablehlo.reshape %v1385 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1516 = stablehlo.reshape %v1428 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1517 = stablehlo.transpose %v1515, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1518 = stablehlo.transpose %v1516, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1519 = stablehlo.convolution(%v1517, %v1518)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1520 = stablehlo.transpose %v1519, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1521 = stablehlo.reshape %v1428 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1523 = stablehlo.reduce(%v1521 init: %v1522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1524 = stablehlo.reshape %v1367 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1525 = stablehlo.reshape %v1456 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1526 = stablehlo.transpose %v1524, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1527 = stablehlo.transpose %v1525, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1528 = stablehlo.convolution(%v1526, %v1527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1529 = stablehlo.transpose %v1528, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1530 = stablehlo.reshape %v1456 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1532 = stablehlo.reduce(%v1530 init: %v1531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1533 = stablehlo.reshape %v1333 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1534 = stablehlo.transpose %v1533, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1536 = stablehlo.reshape %v1461 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1537 = stablehlo.transpose %v1536, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1539 = stablehlo.reshape %v1535 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1540 = stablehlo.reshape %v1538 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1541 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1542 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1543 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1544 = stablehlo.reduce(%v1539 init: %v1541) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1545 = stablehlo.broadcast_in_dim %v1544, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1546 = stablehlo.divide %v1545, %v1542 : tensor<32x49x768xf32>
    %v1547 = stablehlo.subtract %v1539, %v1546 : tensor<32x49x768xf32>
    %v1548 = stablehlo.multiply %v1547, %v1547 : tensor<32x49x768xf32>
    %v1549 = stablehlo.reduce(%v1548 init: %v1541) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1549, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1551 = stablehlo.divide %v1550, %v1542 : tensor<32x49x768xf32>
    %v1552 = stablehlo.add %v1551, %v1543 : tensor<32x49x768xf32>
    %v1553 = stablehlo.rsqrt %v1552 : tensor<32x49x768xf32>
    %v1554 = stablehlo.multiply %v1547, %v1553 : tensor<32x49x768xf32>
    %v1555 = stablehlo.multiply %v1540, %v1554 : tensor<32x49x768xf32>
    %v1556 = stablehlo.reduce(%v1555 init: %v1541) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1557 = stablehlo.reshape %v1461 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1558 = stablehlo.transpose %v1557, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1559 = stablehlo.reshape %v1558 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1561 = stablehlo.reshape %v1559 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1562 = stablehlo.reduce(%v1561 init: %v1560) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1563 = stablehlo.reshape %v1328 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1564 = stablehlo.reshape %v1504 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1565 = stablehlo.transpose %v1563, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1566 = stablehlo.transpose %v1564, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1567 = stablehlo.convolution(%v1565, %v1566)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1569 = stablehlo.reshape %v1504 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1571 = stablehlo.reduce(%v1569 init: %v1570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1572 = stablehlo.broadcast_in_dim %dp16, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1573 = stablehlo.multiply %v1572, %v1509 : tensor<32x37632xf32>
    %v1574 = stablehlo.reshape %v1573 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1575 = stablehlo.broadcast_in_dim %s3b1lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1576 = stablehlo.multiply %v1574, %v1575 : tensor<32x768x7x7xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1579 = stablehlo.reverse %s3b1pW, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1580 = stablehlo.transpose %v1579, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1581 = stablehlo.convolution(%v1578, %v1580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1582 = stablehlo.reshape %v1581 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1583 = stablehlo.multiply %v1303, %v1303 : tensor<32x150528xf32>
    %v1584 = stablehlo.multiply %v1583, %v1303 : tensor<32x150528xf32>
    %v1585 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1586 = stablehlo.multiply %v1585, %v1584 : tensor<32x150528xf32>
    %v1587 = stablehlo.add %v1303, %v1586 : tensor<32x150528xf32>
    %v1588 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1589 = stablehlo.multiply %v1588, %v1587 : tensor<32x150528xf32>
    %v1590 = stablehlo.tanh %v1589 : tensor<32x150528xf32>
    %v1591 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1592 = stablehlo.add %v1591, %v1590 : tensor<32x150528xf32>
    %v1593 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1594 = stablehlo.multiply %v1593, %v1592 : tensor<32x150528xf32>
    %v1595 = stablehlo.multiply %v1590, %v1590 : tensor<32x150528xf32>
    %v1596 = stablehlo.subtract %v1591, %v1595 : tensor<32x150528xf32>
    %v1597 = stablehlo.multiply %v1593, %v1303 : tensor<32x150528xf32>
    %v1598 = stablehlo.multiply %v1597, %v1596 : tensor<32x150528xf32>
    %v1599 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1600 = stablehlo.multiply %v1599, %v1583 : tensor<32x150528xf32>
    %v1601 = stablehlo.add %v1591, %v1600 : tensor<32x150528xf32>
    %v1602 = stablehlo.multiply %v1588, %v1601 : tensor<32x150528xf32>
    %v1603 = stablehlo.multiply %v1598, %v1602 : tensor<32x150528xf32>
    %v1604 = stablehlo.add %v1594, %v1603 : tensor<32x150528xf32>
    %v1605 = stablehlo.multiply %v1582, %v1604 : tensor<32x150528xf32>
    %v1606 = stablehlo.reshape %v1605 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1607 = stablehlo.reverse %s3b1eW, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1608 = stablehlo.transpose %v1607, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1609 = stablehlo.convolution(%v1606, %v1608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1611 = stablehlo.reshape %v1264 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1612 = stablehlo.transpose %v1611, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1614 = stablehlo.reshape %v1610 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1615 = stablehlo.transpose %v1614, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1617 = stablehlo.reshape %v1616 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1618 = stablehlo.broadcast_in_dim %s3b1ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1619 = stablehlo.multiply %v1617, %v1618 : tensor<32x49x768xf32>
    %v1620 = stablehlo.reshape %v1619 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1622 = stablehlo.reshape %v1613 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1624 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1625 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1626 = stablehlo.reduce(%v1622 init: %v1623) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1627 = stablehlo.broadcast_in_dim %v1626, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1628 = stablehlo.divide %v1627, %v1624 : tensor<32x49x768xf32>
    %v1629 = stablehlo.subtract %v1622, %v1628 : tensor<32x49x768xf32>
    %v1630 = stablehlo.multiply %v1629, %v1629 : tensor<32x49x768xf32>
    %v1631 = stablehlo.reduce(%v1630 init: %v1623) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1632 = stablehlo.broadcast_in_dim %v1631, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1633 = stablehlo.divide %v1632, %v1624 : tensor<32x49x768xf32>
    %v1634 = stablehlo.add %v1633, %v1625 : tensor<32x49x768xf32>
    %v1635 = stablehlo.rsqrt %v1634 : tensor<32x49x768xf32>
    %v1636 = stablehlo.multiply %v1629, %v1635 : tensor<32x49x768xf32>
    %v1637 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1638 = stablehlo.multiply %v1637, %v1621 : tensor<32x49x768xf32>
    %v1639 = stablehlo.reduce(%v1638 init: %v1623) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1640 = stablehlo.broadcast_in_dim %v1639, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1641 = stablehlo.multiply %v1636, %v1638 : tensor<32x49x768xf32>
    %v1642 = stablehlo.reduce(%v1641 init: %v1623) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1643 = stablehlo.broadcast_in_dim %v1642, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1644 = stablehlo.multiply %v1638, %v1624 : tensor<32x49x768xf32>
    %v1645 = stablehlo.subtract %v1644, %v1640 : tensor<32x49x768xf32>
    %v1646 = stablehlo.multiply %v1636, %v1643 : tensor<32x49x768xf32>
    %v1647 = stablehlo.subtract %v1645, %v1646 : tensor<32x49x768xf32>
    %v1648 = stablehlo.divide %v1635, %v1624 : tensor<32x49x768xf32>
    %v1649 = stablehlo.multiply %v1648, %v1647 : tensor<32x49x768xf32>
    %v1650 = stablehlo.reshape %v1649 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1651 = stablehlo.reshape %v1650 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1652 = stablehlo.transpose %v1651, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1654 = stablehlo.reshape %v1653 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1655 = stablehlo.reverse %s3b1dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1656 = stablehlo.convolution(%v1654, %v1655)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1658 = stablehlo.add %v1657, %v1509 : tensor<32x37632xf32>
    %v1659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1660 = stablehlo.reshape %v1321 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1661 = stablehlo.reshape %v1573 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1662 = stablehlo.multiply %v1660, %v1661 : tensor<32x768x7x7xf32>
    %v1663 = stablehlo.reduce(%v1662 init: %v1659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1664 = stablehlo.reshape %v1316 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1665 = stablehlo.reshape %v1577 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1666 = stablehlo.transpose %v1664, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1667 = stablehlo.transpose %v1665, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1668 = stablehlo.convolution(%v1666, %v1667)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1669 = stablehlo.transpose %v1668, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1670 = stablehlo.reshape %v1577 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1672 = stablehlo.reduce(%v1670 init: %v1671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1673 = stablehlo.reshape %v1298 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1674 = stablehlo.reshape %v1605 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1675 = stablehlo.transpose %v1673, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1676 = stablehlo.transpose %v1674, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1677 = stablehlo.convolution(%v1675, %v1676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1678 = stablehlo.transpose %v1677, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1679 = stablehlo.reshape %v1605 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1681 = stablehlo.reduce(%v1679 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1682 = stablehlo.reshape %v1264 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1683 = stablehlo.transpose %v1682, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1685 = stablehlo.reshape %v1610 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1686 = stablehlo.transpose %v1685, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1688 = stablehlo.reshape %v1684 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1689 = stablehlo.reshape %v1687 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1691 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1692 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1693 = stablehlo.reduce(%v1688 init: %v1690) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1694 = stablehlo.broadcast_in_dim %v1693, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1695 = stablehlo.divide %v1694, %v1691 : tensor<32x49x768xf32>
    %v1696 = stablehlo.subtract %v1688, %v1695 : tensor<32x49x768xf32>
    %v1697 = stablehlo.multiply %v1696, %v1696 : tensor<32x49x768xf32>
    %v1698 = stablehlo.reduce(%v1697 init: %v1690) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1699 = stablehlo.broadcast_in_dim %v1698, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1700 = stablehlo.divide %v1699, %v1691 : tensor<32x49x768xf32>
    %v1701 = stablehlo.add %v1700, %v1692 : tensor<32x49x768xf32>
    %v1702 = stablehlo.rsqrt %v1701 : tensor<32x49x768xf32>
    %v1703 = stablehlo.multiply %v1696, %v1702 : tensor<32x49x768xf32>
    %v1704 = stablehlo.multiply %v1689, %v1703 : tensor<32x49x768xf32>
    %v1705 = stablehlo.reduce(%v1704 init: %v1690) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1706 = stablehlo.reshape %v1610 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1710 = stablehlo.reshape %v1708 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1711 = stablehlo.reduce(%v1710 init: %v1709) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1712 = stablehlo.reshape %v1259 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1713 = stablehlo.reshape %v1653 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1714 = stablehlo.transpose %v1712, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1715 = stablehlo.transpose %v1713, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1716 = stablehlo.convolution(%v1714, %v1715)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1717 = stablehlo.reshape %v1716 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1718 = stablehlo.reshape %v1653 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1720 = stablehlo.reduce(%v1718 init: %v1719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1721 = stablehlo.broadcast_in_dim %dp15, dims = [0] : (tensor<32xf32>) -> tensor<32x37632xf32>
    %v1722 = stablehlo.multiply %v1721, %v1658 : tensor<32x37632xf32>
    %v1723 = stablehlo.reshape %v1722 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1724 = stablehlo.broadcast_in_dim %s3b0lg, dims = [1] : (tensor<768xf32>) -> tensor<32x768x7x7xf32>
    %v1725 = stablehlo.multiply %v1723, %v1724 : tensor<32x768x7x7xf32>
    %v1726 = stablehlo.reshape %v1725 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1727 = stablehlo.reshape %v1726 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1728 = stablehlo.reverse %s3b0pW, dims = [2, 3] : tensor<768x3072x1x1xf32>
    %v1729 = stablehlo.transpose %v1728, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1730 = stablehlo.convolution(%v1727, %v1729)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x7x7xf32>, tensor<3072x768x1x1xf32>) -> tensor<32x3072x7x7xf32>
    %v1731 = stablehlo.reshape %v1730 : (tensor<32x3072x7x7xf32>) -> tensor<32x150528xf32>
    %v1732 = stablehlo.multiply %v1234, %v1234 : tensor<32x150528xf32>
    %v1733 = stablehlo.multiply %v1732, %v1234 : tensor<32x150528xf32>
    %v1734 = stablehlo.constant dense<0.044715> : tensor<32x150528xf32>
    %v1735 = stablehlo.multiply %v1734, %v1733 : tensor<32x150528xf32>
    %v1736 = stablehlo.add %v1234, %v1735 : tensor<32x150528xf32>
    %v1737 = stablehlo.constant dense<0.7978845608028654> : tensor<32x150528xf32>
    %v1738 = stablehlo.multiply %v1737, %v1736 : tensor<32x150528xf32>
    %v1739 = stablehlo.tanh %v1738 : tensor<32x150528xf32>
    %v1740 = stablehlo.constant dense<1.0> : tensor<32x150528xf32>
    %v1741 = stablehlo.add %v1740, %v1739 : tensor<32x150528xf32>
    %v1742 = stablehlo.constant dense<0.5> : tensor<32x150528xf32>
    %v1743 = stablehlo.multiply %v1742, %v1741 : tensor<32x150528xf32>
    %v1744 = stablehlo.multiply %v1739, %v1739 : tensor<32x150528xf32>
    %v1745 = stablehlo.subtract %v1740, %v1744 : tensor<32x150528xf32>
    %v1746 = stablehlo.multiply %v1742, %v1234 : tensor<32x150528xf32>
    %v1747 = stablehlo.multiply %v1746, %v1745 : tensor<32x150528xf32>
    %v1748 = stablehlo.constant dense<0.134145> : tensor<32x150528xf32>
    %v1749 = stablehlo.multiply %v1748, %v1732 : tensor<32x150528xf32>
    %v1750 = stablehlo.add %v1740, %v1749 : tensor<32x150528xf32>
    %v1751 = stablehlo.multiply %v1737, %v1750 : tensor<32x150528xf32>
    %v1752 = stablehlo.multiply %v1747, %v1751 : tensor<32x150528xf32>
    %v1753 = stablehlo.add %v1743, %v1752 : tensor<32x150528xf32>
    %v1754 = stablehlo.multiply %v1731, %v1753 : tensor<32x150528xf32>
    %v1755 = stablehlo.reshape %v1754 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1756 = stablehlo.reverse %s3b0eW, dims = [2, 3] : tensor<3072x768x1x1xf32>
    %v1757 = stablehlo.transpose %v1756, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1758 = stablehlo.convolution(%v1755, %v1757)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3072x7x7xf32>, tensor<768x3072x1x1xf32>) -> tensor<32x768x7x7xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1760 = stablehlo.reshape %v1195 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1761 = stablehlo.transpose %v1760, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1763 = stablehlo.reshape %v1759 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1764 = stablehlo.transpose %v1763, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1766 = stablehlo.reshape %v1765 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1767 = stablehlo.broadcast_in_dim %s3b0ng, dims = [2] : (tensor<768xf32>) -> tensor<32x49x768xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<32x49x768xf32>
    %v1769 = stablehlo.reshape %v1768 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1770 = stablehlo.reshape %v1769 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1771 = stablehlo.reshape %v1762 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1773 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1774 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1775 = stablehlo.reduce(%v1771 init: %v1772) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1776 = stablehlo.broadcast_in_dim %v1775, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1777 = stablehlo.divide %v1776, %v1773 : tensor<32x49x768xf32>
    %v1778 = stablehlo.subtract %v1771, %v1777 : tensor<32x49x768xf32>
    %v1779 = stablehlo.multiply %v1778, %v1778 : tensor<32x49x768xf32>
    %v1780 = stablehlo.reduce(%v1779 init: %v1772) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1781 = stablehlo.broadcast_in_dim %v1780, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1782 = stablehlo.divide %v1781, %v1773 : tensor<32x49x768xf32>
    %v1783 = stablehlo.add %v1782, %v1774 : tensor<32x49x768xf32>
    %v1784 = stablehlo.rsqrt %v1783 : tensor<32x49x768xf32>
    %v1785 = stablehlo.multiply %v1778, %v1784 : tensor<32x49x768xf32>
    %v1786 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x49x768xf32>
    %v1787 = stablehlo.multiply %v1786, %v1770 : tensor<32x49x768xf32>
    %v1788 = stablehlo.reduce(%v1787 init: %v1772) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1789 = stablehlo.broadcast_in_dim %v1788, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1790 = stablehlo.multiply %v1785, %v1787 : tensor<32x49x768xf32>
    %v1791 = stablehlo.reduce(%v1790 init: %v1772) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1793 = stablehlo.multiply %v1787, %v1773 : tensor<32x49x768xf32>
    %v1794 = stablehlo.subtract %v1793, %v1789 : tensor<32x49x768xf32>
    %v1795 = stablehlo.multiply %v1785, %v1792 : tensor<32x49x768xf32>
    %v1796 = stablehlo.subtract %v1794, %v1795 : tensor<32x49x768xf32>
    %v1797 = stablehlo.divide %v1784, %v1773 : tensor<32x49x768xf32>
    %v1798 = stablehlo.multiply %v1797, %v1796 : tensor<32x49x768xf32>
    %v1799 = stablehlo.reshape %v1798 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1800 = stablehlo.reshape %v1799 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1801 = stablehlo.transpose %v1800, dims = [0, 2, 1] : (tensor<32x49x768xf32>) -> tensor<32x768x49xf32>
    %v1802 = stablehlo.reshape %v1801 : (tensor<32x768x49xf32>) -> tensor<32x37632xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1804 = stablehlo.reverse %s3b0dW, dims = [2, 3] : tensor<768x1x7x7xf32>
    %v1805 = stablehlo.convolution(%v1803, %v1804)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 768 : i64} : (tensor<32x768x7x7xf32>, tensor<768x1x7x7xf32>) -> tensor<32x768x7x7xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<32x768x7x7xf32>) -> tensor<32x37632xf32>
    %v1807 = stablehlo.add %v1806, %v1658 : tensor<32x37632xf32>
    %v1808 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1809 = stablehlo.reshape %v1252 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1810 = stablehlo.reshape %v1722 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1811 = stablehlo.multiply %v1809, %v1810 : tensor<32x768x7x7xf32>
    %v1812 = stablehlo.reduce(%v1811 init: %v1808) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1813 = stablehlo.reshape %v1247 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1814 = stablehlo.reshape %v1726 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1815 = stablehlo.transpose %v1813, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1816 = stablehlo.transpose %v1814, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1817 = stablehlo.convolution(%v1815, %v1816)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3072x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<3072x768x1x1xf32>
    %v1818 = stablehlo.transpose %v1817, dims = [1, 0, 2, 3] : (tensor<3072x768x1x1xf32>) -> tensor<768x3072x1x1xf32>
    %v1819 = stablehlo.reshape %v1726 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1821 = stablehlo.reduce(%v1819 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1822 = stablehlo.reshape %v1229 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1823 = stablehlo.reshape %v1754 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1824 = stablehlo.transpose %v1822, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1825 = stablehlo.transpose %v1823, dims = [1, 0, 2, 3] : (tensor<32x3072x7x7xf32>) -> tensor<3072x32x7x7xf32>
    %v1826 = stablehlo.convolution(%v1824, %v1825)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<3072x32x7x7xf32>) -> tensor<768x3072x1x1xf32>
    %v1827 = stablehlo.transpose %v1826, dims = [1, 0, 2, 3] : (tensor<768x3072x1x1xf32>) -> tensor<3072x768x1x1xf32>
    %v1828 = stablehlo.reshape %v1754 : (tensor<32x150528xf32>) -> tensor<32x3072x7x7xf32>
    %v1829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1830 = stablehlo.reduce(%v1828 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x3072x7x7xf32>, tensor<f32>) -> tensor<3072xf32>
    %v1831 = stablehlo.reshape %v1195 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1832 = stablehlo.transpose %v1831, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1833 = stablehlo.reshape %v1832 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1834 = stablehlo.reshape %v1759 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1835 = stablehlo.transpose %v1834, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1836 = stablehlo.reshape %v1835 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1837 = stablehlo.reshape %v1833 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1838 = stablehlo.reshape %v1836 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1840 = stablehlo.constant dense<768.0> : tensor<32x49x768xf32>
    %v1841 = stablehlo.constant dense<1.0e-6> : tensor<32x49x768xf32>
    %v1842 = stablehlo.reduce(%v1837 init: %v1839) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1843 = stablehlo.broadcast_in_dim %v1842, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1844 = stablehlo.divide %v1843, %v1840 : tensor<32x49x768xf32>
    %v1845 = stablehlo.subtract %v1837, %v1844 : tensor<32x49x768xf32>
    %v1846 = stablehlo.multiply %v1845, %v1845 : tensor<32x49x768xf32>
    %v1847 = stablehlo.reduce(%v1846 init: %v1839) applies stablehlo.add across dimensions = [2] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<32x49xf32>
    %v1848 = stablehlo.broadcast_in_dim %v1847, dims = [0, 1] : (tensor<32x49xf32>) -> tensor<32x49x768xf32>
    %v1849 = stablehlo.divide %v1848, %v1840 : tensor<32x49x768xf32>
    %v1850 = stablehlo.add %v1849, %v1841 : tensor<32x49x768xf32>
    %v1851 = stablehlo.rsqrt %v1850 : tensor<32x49x768xf32>
    %v1852 = stablehlo.multiply %v1845, %v1851 : tensor<32x49x768xf32>
    %v1853 = stablehlo.multiply %v1838, %v1852 : tensor<32x49x768xf32>
    %v1854 = stablehlo.reduce(%v1853 init: %v1839) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1855 = stablehlo.reshape %v1759 : (tensor<32x37632xf32>) -> tensor<32x768x49xf32>
    %v1856 = stablehlo.transpose %v1855, dims = [0, 2, 1] : (tensor<32x768x49xf32>) -> tensor<32x49x768xf32>
    %v1857 = stablehlo.reshape %v1856 : (tensor<32x49x768xf32>) -> tensor<32x37632xf32>
    %v1858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1859 = stablehlo.reshape %v1857 : (tensor<32x37632xf32>) -> tensor<32x49x768xf32>
    %v1860 = stablehlo.reduce(%v1859 init: %v1858) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x49x768xf32>, tensor<f32>) -> tensor<768xf32>
    %v1861 = stablehlo.reshape %v1190 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1862 = stablehlo.reshape %v1802 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1863 = stablehlo.transpose %v1861, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1864 = stablehlo.transpose %v1862, dims = [1, 0, 2, 3] : (tensor<32x768x7x7xf32>) -> tensor<768x32x7x7xf32>
    %v1865 = stablehlo.convolution(%v1863, %v1864)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 768 : i64, feature_group_count = 1 : i64} : (tensor<768x32x7x7xf32>, tensor<768x32x7x7xf32>) -> tensor<1x768x7x7xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<1x768x7x7xf32>) -> tensor<768x1x7x7xf32>
    %v1867 = stablehlo.reshape %v1802 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1870 = stablehlo.reshape %v1807 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1872 = stablehlo.pad %v1870, %v1871, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x14x14xf32>
    %v1873 = stablehlo.reverse %d2W, dims = [2, 3] : tensor<768x384x2x2xf32>
    %v1874 = stablehlo.transpose %v1873, dims = [1, 0, 2, 3] : (tensor<768x384x2x2xf32>) -> tensor<384x768x2x2xf32>
    %v1875 = stablehlo.convolution(%v1872, %v1874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x14x14xf32>, tensor<384x768x2x2xf32>) -> tensor<32x384x14x14xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1877 = stablehlo.reshape %v1151 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1878 = stablehlo.transpose %v1877, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1879 = stablehlo.reshape %v1878 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1880 = stablehlo.reshape %v1876 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1881 = stablehlo.transpose %v1880, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1883 = stablehlo.reshape %v1882 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1884 = stablehlo.broadcast_in_dim %d2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v1885 = stablehlo.multiply %v1883, %v1884 : tensor<32x196x384xf32>
    %v1886 = stablehlo.reshape %v1885 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1887 = stablehlo.reshape %v1886 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1888 = stablehlo.reshape %v1879 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1890 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1891 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1892 = stablehlo.reduce(%v1888 init: %v1889) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1893 = stablehlo.broadcast_in_dim %v1892, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1894 = stablehlo.divide %v1893, %v1890 : tensor<32x196x384xf32>
    %v1895 = stablehlo.subtract %v1888, %v1894 : tensor<32x196x384xf32>
    %v1896 = stablehlo.multiply %v1895, %v1895 : tensor<32x196x384xf32>
    %v1897 = stablehlo.reduce(%v1896 init: %v1889) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1898 = stablehlo.broadcast_in_dim %v1897, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1899 = stablehlo.divide %v1898, %v1890 : tensor<32x196x384xf32>
    %v1900 = stablehlo.add %v1899, %v1891 : tensor<32x196x384xf32>
    %v1901 = stablehlo.rsqrt %v1900 : tensor<32x196x384xf32>
    %v1902 = stablehlo.multiply %v1895, %v1901 : tensor<32x196x384xf32>
    %v1903 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v1904 = stablehlo.multiply %v1903, %v1887 : tensor<32x196x384xf32>
    %v1905 = stablehlo.reduce(%v1904 init: %v1889) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1907 = stablehlo.multiply %v1902, %v1904 : tensor<32x196x384xf32>
    %v1908 = stablehlo.reduce(%v1907 init: %v1889) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1909 = stablehlo.broadcast_in_dim %v1908, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1910 = stablehlo.multiply %v1904, %v1890 : tensor<32x196x384xf32>
    %v1911 = stablehlo.subtract %v1910, %v1906 : tensor<32x196x384xf32>
    %v1912 = stablehlo.multiply %v1902, %v1909 : tensor<32x196x384xf32>
    %v1913 = stablehlo.subtract %v1911, %v1912 : tensor<32x196x384xf32>
    %v1914 = stablehlo.divide %v1901, %v1890 : tensor<32x196x384xf32>
    %v1915 = stablehlo.multiply %v1914, %v1913 : tensor<32x196x384xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1918 = stablehlo.transpose %v1917, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v1919 = stablehlo.reshape %v1918 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v1920 = stablehlo.reshape %v1807 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1922 = stablehlo.reduce(%v1920 init: %v1921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<768xf32>
    %v1923 = stablehlo.reshape %v1151 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1924 = stablehlo.transpose %v1923, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1926 = stablehlo.reshape %v1876 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1927 = stablehlo.transpose %v1926, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1928 = stablehlo.reshape %v1927 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1929 = stablehlo.reshape %v1925 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1930 = stablehlo.reshape %v1928 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1932 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v1933 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v1934 = stablehlo.reduce(%v1929 init: %v1931) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1936 = stablehlo.divide %v1935, %v1932 : tensor<32x196x384xf32>
    %v1937 = stablehlo.subtract %v1929, %v1936 : tensor<32x196x384xf32>
    %v1938 = stablehlo.multiply %v1937, %v1937 : tensor<32x196x384xf32>
    %v1939 = stablehlo.reduce(%v1938 init: %v1931) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v1941 = stablehlo.divide %v1940, %v1932 : tensor<32x196x384xf32>
    %v1942 = stablehlo.add %v1941, %v1933 : tensor<32x196x384xf32>
    %v1943 = stablehlo.rsqrt %v1942 : tensor<32x196x384xf32>
    %v1944 = stablehlo.multiply %v1937, %v1943 : tensor<32x196x384xf32>
    %v1945 = stablehlo.multiply %v1930, %v1944 : tensor<32x196x384xf32>
    %v1946 = stablehlo.reduce(%v1945 init: %v1931) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v1947 = stablehlo.reshape %v1876 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v1948 = stablehlo.transpose %v1947, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v1950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1951 = stablehlo.reshape %v1949 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1950) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v1953 = stablehlo.reshape %v1185 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1954 = stablehlo.reshape %v1807 : (tensor<32x37632xf32>) -> tensor<32x768x7x7xf32>
    %v1955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1956 = stablehlo.pad %v1954, %v1955, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x768x7x7xf32>, tensor<f32>) -> tensor<32x768x13x13xf32>
    %v1957 = stablehlo.transpose %v1953, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v1958 = stablehlo.transpose %v1956, dims = [1, 0, 2, 3] : (tensor<32x768x13x13xf32>) -> tensor<768x32x13x13xf32>
    %v1959 = stablehlo.convolution(%v1957, %v1958)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<768x32x13x13xf32>) -> tensor<384x768x2x2xf32>
    %v1960 = stablehlo.transpose %v1959, dims = [1, 0, 2, 3] : (tensor<384x768x2x2xf32>) -> tensor<768x384x2x2xf32>
    %v1961 = stablehlo.broadcast_in_dim %dp14, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v1962 = stablehlo.multiply %v1961, %v1919 : tensor<32x75264xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1964 = stablehlo.broadcast_in_dim %s2b8lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v1965 = stablehlo.multiply %v1963, %v1964 : tensor<32x384x14x14xf32>
    %v1966 = stablehlo.reshape %v1965 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v1967 = stablehlo.reshape %v1966 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v1968 = stablehlo.reverse %s2b8pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v1969 = stablehlo.transpose %v1968, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v1970 = stablehlo.convolution(%v1967, %v1969)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v1972 = stablehlo.multiply %v1126, %v1126 : tensor<32x301056xf32>
    %v1973 = stablehlo.multiply %v1972, %v1126 : tensor<32x301056xf32>
    %v1974 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v1975 = stablehlo.multiply %v1974, %v1973 : tensor<32x301056xf32>
    %v1976 = stablehlo.add %v1126, %v1975 : tensor<32x301056xf32>
    %v1977 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<32x301056xf32>
    %v1979 = stablehlo.tanh %v1978 : tensor<32x301056xf32>
    %v1980 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v1981 = stablehlo.add %v1980, %v1979 : tensor<32x301056xf32>
    %v1982 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v1983 = stablehlo.multiply %v1982, %v1981 : tensor<32x301056xf32>
    %v1984 = stablehlo.multiply %v1979, %v1979 : tensor<32x301056xf32>
    %v1985 = stablehlo.subtract %v1980, %v1984 : tensor<32x301056xf32>
    %v1986 = stablehlo.multiply %v1982, %v1126 : tensor<32x301056xf32>
    %v1987 = stablehlo.multiply %v1986, %v1985 : tensor<32x301056xf32>
    %v1988 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v1989 = stablehlo.multiply %v1988, %v1972 : tensor<32x301056xf32>
    %v1990 = stablehlo.add %v1980, %v1989 : tensor<32x301056xf32>
    %v1991 = stablehlo.multiply %v1977, %v1990 : tensor<32x301056xf32>
    %v1992 = stablehlo.multiply %v1987, %v1991 : tensor<32x301056xf32>
    %v1993 = stablehlo.add %v1983, %v1992 : tensor<32x301056xf32>
    %v1994 = stablehlo.multiply %v1971, %v1993 : tensor<32x301056xf32>
    %v1995 = stablehlo.reshape %v1994 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v1996 = stablehlo.reverse %s2b8eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v1997 = stablehlo.transpose %v1996, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v1998 = stablehlo.convolution(%v1995, %v1997)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2000 = stablehlo.reshape %v1087 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2001 = stablehlo.transpose %v2000, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2002 = stablehlo.reshape %v2001 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2003 = stablehlo.reshape %v1999 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2004 = stablehlo.transpose %v2003, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2006 = stablehlo.reshape %v2005 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2007 = stablehlo.broadcast_in_dim %s2b8ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2008 = stablehlo.multiply %v2006, %v2007 : tensor<32x196x384xf32>
    %v2009 = stablehlo.reshape %v2008 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2010 = stablehlo.reshape %v2009 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2011 = stablehlo.reshape %v2002 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2013 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2014 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2015 = stablehlo.reduce(%v2011 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2016 = stablehlo.broadcast_in_dim %v2015, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2017 = stablehlo.divide %v2016, %v2013 : tensor<32x196x384xf32>
    %v2018 = stablehlo.subtract %v2011, %v2017 : tensor<32x196x384xf32>
    %v2019 = stablehlo.multiply %v2018, %v2018 : tensor<32x196x384xf32>
    %v2020 = stablehlo.reduce(%v2019 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2021 = stablehlo.broadcast_in_dim %v2020, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2022 = stablehlo.divide %v2021, %v2013 : tensor<32x196x384xf32>
    %v2023 = stablehlo.add %v2022, %v2014 : tensor<32x196x384xf32>
    %v2024 = stablehlo.rsqrt %v2023 : tensor<32x196x384xf32>
    %v2025 = stablehlo.multiply %v2018, %v2024 : tensor<32x196x384xf32>
    %v2026 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2027 = stablehlo.multiply %v2026, %v2010 : tensor<32x196x384xf32>
    %v2028 = stablehlo.reduce(%v2027 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2029 = stablehlo.broadcast_in_dim %v2028, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2030 = stablehlo.multiply %v2025, %v2027 : tensor<32x196x384xf32>
    %v2031 = stablehlo.reduce(%v2030 init: %v2012) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2032 = stablehlo.broadcast_in_dim %v2031, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2033 = stablehlo.multiply %v2027, %v2013 : tensor<32x196x384xf32>
    %v2034 = stablehlo.subtract %v2033, %v2029 : tensor<32x196x384xf32>
    %v2035 = stablehlo.multiply %v2025, %v2032 : tensor<32x196x384xf32>
    %v2036 = stablehlo.subtract %v2034, %v2035 : tensor<32x196x384xf32>
    %v2037 = stablehlo.divide %v2024, %v2013 : tensor<32x196x384xf32>
    %v2038 = stablehlo.multiply %v2037, %v2036 : tensor<32x196x384xf32>
    %v2039 = stablehlo.reshape %v2038 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2040 = stablehlo.reshape %v2039 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2041 = stablehlo.transpose %v2040, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2044 = stablehlo.reverse %s2b8dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2045 = stablehlo.convolution(%v2043, %v2044)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2046 = stablehlo.reshape %v2045 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2047 = stablehlo.add %v2046, %v1919 : tensor<32x75264xf32>
    %v2048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2049 = stablehlo.reshape %v1144 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2050 = stablehlo.reshape %v1962 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2051 = stablehlo.multiply %v2049, %v2050 : tensor<32x384x14x14xf32>
    %v2052 = stablehlo.reduce(%v2051 init: %v2048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2053 = stablehlo.reshape %v1139 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2054 = stablehlo.reshape %v1966 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2055 = stablehlo.transpose %v2053, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2056 = stablehlo.transpose %v2054, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2057 = stablehlo.convolution(%v2055, %v2056)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2058 = stablehlo.transpose %v2057, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2059 = stablehlo.reshape %v1966 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2061 = stablehlo.reduce(%v2059 init: %v2060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2062 = stablehlo.reshape %v1121 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2063 = stablehlo.reshape %v1994 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2064 = stablehlo.transpose %v2062, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2065 = stablehlo.transpose %v2063, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2066 = stablehlo.convolution(%v2064, %v2065)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2067 = stablehlo.transpose %v2066, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2068 = stablehlo.reshape %v1994 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2070 = stablehlo.reduce(%v2068 init: %v2069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2071 = stablehlo.reshape %v1087 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2072 = stablehlo.transpose %v2071, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2074 = stablehlo.reshape %v1999 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2075 = stablehlo.transpose %v2074, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2076 = stablehlo.reshape %v2075 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2077 = stablehlo.reshape %v2073 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2078 = stablehlo.reshape %v2076 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2080 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2081 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2082 = stablehlo.reduce(%v2077 init: %v2079) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2083 = stablehlo.broadcast_in_dim %v2082, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2084 = stablehlo.divide %v2083, %v2080 : tensor<32x196x384xf32>
    %v2085 = stablehlo.subtract %v2077, %v2084 : tensor<32x196x384xf32>
    %v2086 = stablehlo.multiply %v2085, %v2085 : tensor<32x196x384xf32>
    %v2087 = stablehlo.reduce(%v2086 init: %v2079) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2088 = stablehlo.broadcast_in_dim %v2087, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2089 = stablehlo.divide %v2088, %v2080 : tensor<32x196x384xf32>
    %v2090 = stablehlo.add %v2089, %v2081 : tensor<32x196x384xf32>
    %v2091 = stablehlo.rsqrt %v2090 : tensor<32x196x384xf32>
    %v2092 = stablehlo.multiply %v2085, %v2091 : tensor<32x196x384xf32>
    %v2093 = stablehlo.multiply %v2078, %v2092 : tensor<32x196x384xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2079) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2095 = stablehlo.reshape %v1999 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2096 = stablehlo.transpose %v2095, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2099 = stablehlo.reshape %v2097 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2100 = stablehlo.reduce(%v2099 init: %v2098) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2101 = stablehlo.reshape %v1082 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2102 = stablehlo.reshape %v2042 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2103 = stablehlo.transpose %v2101, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2104 = stablehlo.transpose %v2102, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2105 = stablehlo.convolution(%v2103, %v2104)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2106 = stablehlo.reshape %v2105 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2107 = stablehlo.reshape %v2042 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.reduce(%v2107 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2110 = stablehlo.broadcast_in_dim %dp13, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2111 = stablehlo.multiply %v2110, %v2047 : tensor<32x75264xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2113 = stablehlo.broadcast_in_dim %s2b7lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2114 = stablehlo.multiply %v2112, %v2113 : tensor<32x384x14x14xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2117 = stablehlo.reverse %s2b7pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2118 = stablehlo.transpose %v2117, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2119 = stablehlo.convolution(%v2116, %v2118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2121 = stablehlo.multiply %v1057, %v1057 : tensor<32x301056xf32>
    %v2122 = stablehlo.multiply %v2121, %v1057 : tensor<32x301056xf32>
    %v2123 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2124 = stablehlo.multiply %v2123, %v2122 : tensor<32x301056xf32>
    %v2125 = stablehlo.add %v1057, %v2124 : tensor<32x301056xf32>
    %v2126 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2127 = stablehlo.multiply %v2126, %v2125 : tensor<32x301056xf32>
    %v2128 = stablehlo.tanh %v2127 : tensor<32x301056xf32>
    %v2129 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2130 = stablehlo.add %v2129, %v2128 : tensor<32x301056xf32>
    %v2131 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2132 = stablehlo.multiply %v2131, %v2130 : tensor<32x301056xf32>
    %v2133 = stablehlo.multiply %v2128, %v2128 : tensor<32x301056xf32>
    %v2134 = stablehlo.subtract %v2129, %v2133 : tensor<32x301056xf32>
    %v2135 = stablehlo.multiply %v2131, %v1057 : tensor<32x301056xf32>
    %v2136 = stablehlo.multiply %v2135, %v2134 : tensor<32x301056xf32>
    %v2137 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2138 = stablehlo.multiply %v2137, %v2121 : tensor<32x301056xf32>
    %v2139 = stablehlo.add %v2129, %v2138 : tensor<32x301056xf32>
    %v2140 = stablehlo.multiply %v2126, %v2139 : tensor<32x301056xf32>
    %v2141 = stablehlo.multiply %v2136, %v2140 : tensor<32x301056xf32>
    %v2142 = stablehlo.add %v2132, %v2141 : tensor<32x301056xf32>
    %v2143 = stablehlo.multiply %v2120, %v2142 : tensor<32x301056xf32>
    %v2144 = stablehlo.reshape %v2143 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2145 = stablehlo.reverse %s2b7eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2146 = stablehlo.transpose %v2145, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2147 = stablehlo.convolution(%v2144, %v2146)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2149 = stablehlo.reshape %v1018 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2150 = stablehlo.transpose %v2149, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2151 = stablehlo.reshape %v2150 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2152 = stablehlo.reshape %v2148 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2153 = stablehlo.transpose %v2152, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2156 = stablehlo.broadcast_in_dim %s2b7ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2157 = stablehlo.multiply %v2155, %v2156 : tensor<32x196x384xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2159 = stablehlo.reshape %v2158 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2160 = stablehlo.reshape %v2151 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2162 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2163 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2164 = stablehlo.reduce(%v2160 init: %v2161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2165 = stablehlo.broadcast_in_dim %v2164, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2166 = stablehlo.divide %v2165, %v2162 : tensor<32x196x384xf32>
    %v2167 = stablehlo.subtract %v2160, %v2166 : tensor<32x196x384xf32>
    %v2168 = stablehlo.multiply %v2167, %v2167 : tensor<32x196x384xf32>
    %v2169 = stablehlo.reduce(%v2168 init: %v2161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2170 = stablehlo.broadcast_in_dim %v2169, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2171 = stablehlo.divide %v2170, %v2162 : tensor<32x196x384xf32>
    %v2172 = stablehlo.add %v2171, %v2163 : tensor<32x196x384xf32>
    %v2173 = stablehlo.rsqrt %v2172 : tensor<32x196x384xf32>
    %v2174 = stablehlo.multiply %v2167, %v2173 : tensor<32x196x384xf32>
    %v2175 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2176 = stablehlo.multiply %v2175, %v2159 : tensor<32x196x384xf32>
    %v2177 = stablehlo.reduce(%v2176 init: %v2161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2178 = stablehlo.broadcast_in_dim %v2177, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2179 = stablehlo.multiply %v2174, %v2176 : tensor<32x196x384xf32>
    %v2180 = stablehlo.reduce(%v2179 init: %v2161) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2181 = stablehlo.broadcast_in_dim %v2180, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2182 = stablehlo.multiply %v2176, %v2162 : tensor<32x196x384xf32>
    %v2183 = stablehlo.subtract %v2182, %v2178 : tensor<32x196x384xf32>
    %v2184 = stablehlo.multiply %v2174, %v2181 : tensor<32x196x384xf32>
    %v2185 = stablehlo.subtract %v2183, %v2184 : tensor<32x196x384xf32>
    %v2186 = stablehlo.divide %v2173, %v2162 : tensor<32x196x384xf32>
    %v2187 = stablehlo.multiply %v2186, %v2185 : tensor<32x196x384xf32>
    %v2188 = stablehlo.reshape %v2187 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2189 = stablehlo.reshape %v2188 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2190 = stablehlo.transpose %v2189, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2193 = stablehlo.reverse %s2b7dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2194 = stablehlo.convolution(%v2192, %v2193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2196 = stablehlo.add %v2195, %v2047 : tensor<32x75264xf32>
    %v2197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2198 = stablehlo.reshape %v1075 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2199 = stablehlo.reshape %v2111 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2200 = stablehlo.multiply %v2198, %v2199 : tensor<32x384x14x14xf32>
    %v2201 = stablehlo.reduce(%v2200 init: %v2197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2202 = stablehlo.reshape %v1070 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2203 = stablehlo.reshape %v2115 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2204 = stablehlo.transpose %v2202, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2205 = stablehlo.transpose %v2203, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2206 = stablehlo.convolution(%v2204, %v2205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2207 = stablehlo.transpose %v2206, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2208 = stablehlo.reshape %v2115 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.reduce(%v2208 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2211 = stablehlo.reshape %v1052 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2212 = stablehlo.reshape %v2143 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2213 = stablehlo.transpose %v2211, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2214 = stablehlo.transpose %v2212, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2215 = stablehlo.convolution(%v2213, %v2214)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2216 = stablehlo.transpose %v2215, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2217 = stablehlo.reshape %v2143 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2219 = stablehlo.reduce(%v2217 init: %v2218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2220 = stablehlo.reshape %v1018 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2221 = stablehlo.transpose %v2220, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2222 = stablehlo.reshape %v2221 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2223 = stablehlo.reshape %v2148 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2224 = stablehlo.transpose %v2223, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2225 = stablehlo.reshape %v2224 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2226 = stablehlo.reshape %v2222 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2227 = stablehlo.reshape %v2225 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2229 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2230 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2231 = stablehlo.reduce(%v2226 init: %v2228) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2233 = stablehlo.divide %v2232, %v2229 : tensor<32x196x384xf32>
    %v2234 = stablehlo.subtract %v2226, %v2233 : tensor<32x196x384xf32>
    %v2235 = stablehlo.multiply %v2234, %v2234 : tensor<32x196x384xf32>
    %v2236 = stablehlo.reduce(%v2235 init: %v2228) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2237 = stablehlo.broadcast_in_dim %v2236, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2238 = stablehlo.divide %v2237, %v2229 : tensor<32x196x384xf32>
    %v2239 = stablehlo.add %v2238, %v2230 : tensor<32x196x384xf32>
    %v2240 = stablehlo.rsqrt %v2239 : tensor<32x196x384xf32>
    %v2241 = stablehlo.multiply %v2234, %v2240 : tensor<32x196x384xf32>
    %v2242 = stablehlo.multiply %v2227, %v2241 : tensor<32x196x384xf32>
    %v2243 = stablehlo.reduce(%v2242 init: %v2228) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2244 = stablehlo.reshape %v2148 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2245 = stablehlo.transpose %v2244, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2246 = stablehlo.reshape %v2245 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2248 = stablehlo.reshape %v2246 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2249 = stablehlo.reduce(%v2248 init: %v2247) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2250 = stablehlo.reshape %v1013 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2251 = stablehlo.reshape %v2191 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2252 = stablehlo.transpose %v2250, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2253 = stablehlo.transpose %v2251, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2254 = stablehlo.convolution(%v2252, %v2253)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2255 = stablehlo.reshape %v2254 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2256 = stablehlo.reshape %v2191 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2258 = stablehlo.reduce(%v2256 init: %v2257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2259 = stablehlo.broadcast_in_dim %dp12, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2260 = stablehlo.multiply %v2259, %v2196 : tensor<32x75264xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2262 = stablehlo.broadcast_in_dim %s2b6lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2263 = stablehlo.multiply %v2261, %v2262 : tensor<32x384x14x14xf32>
    %v2264 = stablehlo.reshape %v2263 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2266 = stablehlo.reverse %s2b6pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2267 = stablehlo.transpose %v2266, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2268 = stablehlo.convolution(%v2265, %v2267)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2270 = stablehlo.multiply %v988, %v988 : tensor<32x301056xf32>
    %v2271 = stablehlo.multiply %v2270, %v988 : tensor<32x301056xf32>
    %v2272 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2273 = stablehlo.multiply %v2272, %v2271 : tensor<32x301056xf32>
    %v2274 = stablehlo.add %v988, %v2273 : tensor<32x301056xf32>
    %v2275 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2276 = stablehlo.multiply %v2275, %v2274 : tensor<32x301056xf32>
    %v2277 = stablehlo.tanh %v2276 : tensor<32x301056xf32>
    %v2278 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2279 = stablehlo.add %v2278, %v2277 : tensor<32x301056xf32>
    %v2280 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2281 = stablehlo.multiply %v2280, %v2279 : tensor<32x301056xf32>
    %v2282 = stablehlo.multiply %v2277, %v2277 : tensor<32x301056xf32>
    %v2283 = stablehlo.subtract %v2278, %v2282 : tensor<32x301056xf32>
    %v2284 = stablehlo.multiply %v2280, %v988 : tensor<32x301056xf32>
    %v2285 = stablehlo.multiply %v2284, %v2283 : tensor<32x301056xf32>
    %v2286 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2287 = stablehlo.multiply %v2286, %v2270 : tensor<32x301056xf32>
    %v2288 = stablehlo.add %v2278, %v2287 : tensor<32x301056xf32>
    %v2289 = stablehlo.multiply %v2275, %v2288 : tensor<32x301056xf32>
    %v2290 = stablehlo.multiply %v2285, %v2289 : tensor<32x301056xf32>
    %v2291 = stablehlo.add %v2281, %v2290 : tensor<32x301056xf32>
    %v2292 = stablehlo.multiply %v2269, %v2291 : tensor<32x301056xf32>
    %v2293 = stablehlo.reshape %v2292 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2294 = stablehlo.reverse %s2b6eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2295 = stablehlo.transpose %v2294, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2296 = stablehlo.convolution(%v2293, %v2295)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2297 = stablehlo.reshape %v2296 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2298 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2299 = stablehlo.transpose %v2298, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2300 = stablehlo.reshape %v2299 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2301 = stablehlo.reshape %v2297 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2302 = stablehlo.transpose %v2301, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2304 = stablehlo.reshape %v2303 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2305 = stablehlo.broadcast_in_dim %s2b6ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2306 = stablehlo.multiply %v2304, %v2305 : tensor<32x196x384xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2308 = stablehlo.reshape %v2307 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2309 = stablehlo.reshape %v2300 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2311 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2312 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2313 = stablehlo.reduce(%v2309 init: %v2310) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2314 = stablehlo.broadcast_in_dim %v2313, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2315 = stablehlo.divide %v2314, %v2311 : tensor<32x196x384xf32>
    %v2316 = stablehlo.subtract %v2309, %v2315 : tensor<32x196x384xf32>
    %v2317 = stablehlo.multiply %v2316, %v2316 : tensor<32x196x384xf32>
    %v2318 = stablehlo.reduce(%v2317 init: %v2310) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2319 = stablehlo.broadcast_in_dim %v2318, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2320 = stablehlo.divide %v2319, %v2311 : tensor<32x196x384xf32>
    %v2321 = stablehlo.add %v2320, %v2312 : tensor<32x196x384xf32>
    %v2322 = stablehlo.rsqrt %v2321 : tensor<32x196x384xf32>
    %v2323 = stablehlo.multiply %v2316, %v2322 : tensor<32x196x384xf32>
    %v2324 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2325 = stablehlo.multiply %v2324, %v2308 : tensor<32x196x384xf32>
    %v2326 = stablehlo.reduce(%v2325 init: %v2310) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2327 = stablehlo.broadcast_in_dim %v2326, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2328 = stablehlo.multiply %v2323, %v2325 : tensor<32x196x384xf32>
    %v2329 = stablehlo.reduce(%v2328 init: %v2310) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2330 = stablehlo.broadcast_in_dim %v2329, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2331 = stablehlo.multiply %v2325, %v2311 : tensor<32x196x384xf32>
    %v2332 = stablehlo.subtract %v2331, %v2327 : tensor<32x196x384xf32>
    %v2333 = stablehlo.multiply %v2323, %v2330 : tensor<32x196x384xf32>
    %v2334 = stablehlo.subtract %v2332, %v2333 : tensor<32x196x384xf32>
    %v2335 = stablehlo.divide %v2322, %v2311 : tensor<32x196x384xf32>
    %v2336 = stablehlo.multiply %v2335, %v2334 : tensor<32x196x384xf32>
    %v2337 = stablehlo.reshape %v2336 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2339 = stablehlo.transpose %v2338, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2340 = stablehlo.reshape %v2339 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2341 = stablehlo.reshape %v2340 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2342 = stablehlo.reverse %s2b6dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2343 = stablehlo.convolution(%v2341, %v2342)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2344 = stablehlo.reshape %v2343 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2345 = stablehlo.add %v2344, %v2196 : tensor<32x75264xf32>
    %v2346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2347 = stablehlo.reshape %v1006 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2348 = stablehlo.reshape %v2260 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2349 = stablehlo.multiply %v2347, %v2348 : tensor<32x384x14x14xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2351 = stablehlo.reshape %v1001 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2352 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2353 = stablehlo.transpose %v2351, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2354 = stablehlo.transpose %v2352, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2355 = stablehlo.convolution(%v2353, %v2354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2356 = stablehlo.transpose %v2355, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2357 = stablehlo.reshape %v2264 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2359 = stablehlo.reduce(%v2357 init: %v2358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2360 = stablehlo.reshape %v983 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2361 = stablehlo.reshape %v2292 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2362 = stablehlo.transpose %v2360, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2363 = stablehlo.transpose %v2361, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2364 = stablehlo.convolution(%v2362, %v2363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2365 = stablehlo.transpose %v2364, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2366 = stablehlo.reshape %v2292 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2368 = stablehlo.reduce(%v2366 init: %v2367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2369 = stablehlo.reshape %v949 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2370 = stablehlo.transpose %v2369, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2371 = stablehlo.reshape %v2370 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2372 = stablehlo.reshape %v2297 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2373 = stablehlo.transpose %v2372, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2374 = stablehlo.reshape %v2373 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2375 = stablehlo.reshape %v2371 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2376 = stablehlo.reshape %v2374 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2378 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2379 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2380 = stablehlo.reduce(%v2375 init: %v2377) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2381 = stablehlo.broadcast_in_dim %v2380, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2382 = stablehlo.divide %v2381, %v2378 : tensor<32x196x384xf32>
    %v2383 = stablehlo.subtract %v2375, %v2382 : tensor<32x196x384xf32>
    %v2384 = stablehlo.multiply %v2383, %v2383 : tensor<32x196x384xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2377) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2387 = stablehlo.divide %v2386, %v2378 : tensor<32x196x384xf32>
    %v2388 = stablehlo.add %v2387, %v2379 : tensor<32x196x384xf32>
    %v2389 = stablehlo.rsqrt %v2388 : tensor<32x196x384xf32>
    %v2390 = stablehlo.multiply %v2383, %v2389 : tensor<32x196x384xf32>
    %v2391 = stablehlo.multiply %v2376, %v2390 : tensor<32x196x384xf32>
    %v2392 = stablehlo.reduce(%v2391 init: %v2377) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2393 = stablehlo.reshape %v2297 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2394 = stablehlo.transpose %v2393, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2397 = stablehlo.reshape %v2395 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2398 = stablehlo.reduce(%v2397 init: %v2396) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2399 = stablehlo.reshape %v944 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2400 = stablehlo.reshape %v2340 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2401 = stablehlo.transpose %v2399, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2402 = stablehlo.transpose %v2400, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2403 = stablehlo.convolution(%v2401, %v2402)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2405 = stablehlo.reshape %v2340 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2407 = stablehlo.reduce(%v2405 init: %v2406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2408 = stablehlo.broadcast_in_dim %dp11, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2409 = stablehlo.multiply %v2408, %v2345 : tensor<32x75264xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2411 = stablehlo.broadcast_in_dim %s2b5lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2412 = stablehlo.multiply %v2410, %v2411 : tensor<32x384x14x14xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2414 = stablehlo.reshape %v2413 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2415 = stablehlo.reverse %s2b5pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2416 = stablehlo.transpose %v2415, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2417 = stablehlo.convolution(%v2414, %v2416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2419 = stablehlo.multiply %v919, %v919 : tensor<32x301056xf32>
    %v2420 = stablehlo.multiply %v2419, %v919 : tensor<32x301056xf32>
    %v2421 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2422 = stablehlo.multiply %v2421, %v2420 : tensor<32x301056xf32>
    %v2423 = stablehlo.add %v919, %v2422 : tensor<32x301056xf32>
    %v2424 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2425 = stablehlo.multiply %v2424, %v2423 : tensor<32x301056xf32>
    %v2426 = stablehlo.tanh %v2425 : tensor<32x301056xf32>
    %v2427 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2428 = stablehlo.add %v2427, %v2426 : tensor<32x301056xf32>
    %v2429 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2430 = stablehlo.multiply %v2429, %v2428 : tensor<32x301056xf32>
    %v2431 = stablehlo.multiply %v2426, %v2426 : tensor<32x301056xf32>
    %v2432 = stablehlo.subtract %v2427, %v2431 : tensor<32x301056xf32>
    %v2433 = stablehlo.multiply %v2429, %v919 : tensor<32x301056xf32>
    %v2434 = stablehlo.multiply %v2433, %v2432 : tensor<32x301056xf32>
    %v2435 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2436 = stablehlo.multiply %v2435, %v2419 : tensor<32x301056xf32>
    %v2437 = stablehlo.add %v2427, %v2436 : tensor<32x301056xf32>
    %v2438 = stablehlo.multiply %v2424, %v2437 : tensor<32x301056xf32>
    %v2439 = stablehlo.multiply %v2434, %v2438 : tensor<32x301056xf32>
    %v2440 = stablehlo.add %v2430, %v2439 : tensor<32x301056xf32>
    %v2441 = stablehlo.multiply %v2418, %v2440 : tensor<32x301056xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2443 = stablehlo.reverse %s2b5eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2444 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2445 = stablehlo.convolution(%v2442, %v2444)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2446 = stablehlo.reshape %v2445 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2447 = stablehlo.reshape %v880 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2448 = stablehlo.transpose %v2447, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2449 = stablehlo.reshape %v2448 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2450 = stablehlo.reshape %v2446 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2451 = stablehlo.transpose %v2450, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2452 = stablehlo.reshape %v2451 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2453 = stablehlo.reshape %v2452 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2454 = stablehlo.broadcast_in_dim %s2b5ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<32x196x384xf32>
    %v2456 = stablehlo.reshape %v2455 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2457 = stablehlo.reshape %v2456 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2458 = stablehlo.reshape %v2449 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2460 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2461 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2462 = stablehlo.reduce(%v2458 init: %v2459) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2463 = stablehlo.broadcast_in_dim %v2462, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2464 = stablehlo.divide %v2463, %v2460 : tensor<32x196x384xf32>
    %v2465 = stablehlo.subtract %v2458, %v2464 : tensor<32x196x384xf32>
    %v2466 = stablehlo.multiply %v2465, %v2465 : tensor<32x196x384xf32>
    %v2467 = stablehlo.reduce(%v2466 init: %v2459) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2468 = stablehlo.broadcast_in_dim %v2467, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2469 = stablehlo.divide %v2468, %v2460 : tensor<32x196x384xf32>
    %v2470 = stablehlo.add %v2469, %v2461 : tensor<32x196x384xf32>
    %v2471 = stablehlo.rsqrt %v2470 : tensor<32x196x384xf32>
    %v2472 = stablehlo.multiply %v2465, %v2471 : tensor<32x196x384xf32>
    %v2473 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2474 = stablehlo.multiply %v2473, %v2457 : tensor<32x196x384xf32>
    %v2475 = stablehlo.reduce(%v2474 init: %v2459) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2476 = stablehlo.broadcast_in_dim %v2475, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2477 = stablehlo.multiply %v2472, %v2474 : tensor<32x196x384xf32>
    %v2478 = stablehlo.reduce(%v2477 init: %v2459) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2480 = stablehlo.multiply %v2474, %v2460 : tensor<32x196x384xf32>
    %v2481 = stablehlo.subtract %v2480, %v2476 : tensor<32x196x384xf32>
    %v2482 = stablehlo.multiply %v2472, %v2479 : tensor<32x196x384xf32>
    %v2483 = stablehlo.subtract %v2481, %v2482 : tensor<32x196x384xf32>
    %v2484 = stablehlo.divide %v2471, %v2460 : tensor<32x196x384xf32>
    %v2485 = stablehlo.multiply %v2484, %v2483 : tensor<32x196x384xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2488 = stablehlo.transpose %v2487, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2489 = stablehlo.reshape %v2488 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2490 = stablehlo.reshape %v2489 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2491 = stablehlo.reverse %s2b5dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2492 = stablehlo.convolution(%v2490, %v2491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2493 = stablehlo.reshape %v2492 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2494 = stablehlo.add %v2493, %v2345 : tensor<32x75264xf32>
    %v2495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2496 = stablehlo.reshape %v937 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2497 = stablehlo.reshape %v2409 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2498 = stablehlo.multiply %v2496, %v2497 : tensor<32x384x14x14xf32>
    %v2499 = stablehlo.reduce(%v2498 init: %v2495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2500 = stablehlo.reshape %v932 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2501 = stablehlo.reshape %v2413 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2502 = stablehlo.transpose %v2500, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2503 = stablehlo.transpose %v2501, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2504 = stablehlo.convolution(%v2502, %v2503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2505 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2506 = stablehlo.reshape %v2413 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2508 = stablehlo.reduce(%v2506 init: %v2507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2509 = stablehlo.reshape %v914 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2510 = stablehlo.reshape %v2441 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2511 = stablehlo.transpose %v2509, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2512 = stablehlo.transpose %v2510, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2513 = stablehlo.convolution(%v2511, %v2512)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2514 = stablehlo.transpose %v2513, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2515 = stablehlo.reshape %v2441 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2517 = stablehlo.reduce(%v2515 init: %v2516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2518 = stablehlo.reshape %v880 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2519 = stablehlo.transpose %v2518, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2521 = stablehlo.reshape %v2446 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2522 = stablehlo.transpose %v2521, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2524 = stablehlo.reshape %v2520 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2525 = stablehlo.reshape %v2523 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2526 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2527 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2528 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2529 = stablehlo.reduce(%v2524 init: %v2526) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2530 = stablehlo.broadcast_in_dim %v2529, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2531 = stablehlo.divide %v2530, %v2527 : tensor<32x196x384xf32>
    %v2532 = stablehlo.subtract %v2524, %v2531 : tensor<32x196x384xf32>
    %v2533 = stablehlo.multiply %v2532, %v2532 : tensor<32x196x384xf32>
    %v2534 = stablehlo.reduce(%v2533 init: %v2526) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2535 = stablehlo.broadcast_in_dim %v2534, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2536 = stablehlo.divide %v2535, %v2527 : tensor<32x196x384xf32>
    %v2537 = stablehlo.add %v2536, %v2528 : tensor<32x196x384xf32>
    %v2538 = stablehlo.rsqrt %v2537 : tensor<32x196x384xf32>
    %v2539 = stablehlo.multiply %v2532, %v2538 : tensor<32x196x384xf32>
    %v2540 = stablehlo.multiply %v2525, %v2539 : tensor<32x196x384xf32>
    %v2541 = stablehlo.reduce(%v2540 init: %v2526) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2542 = stablehlo.reshape %v2446 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2543 = stablehlo.transpose %v2542, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2546 = stablehlo.reshape %v2544 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2545) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2548 = stablehlo.reshape %v875 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2549 = stablehlo.reshape %v2489 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2550 = stablehlo.transpose %v2548, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2551 = stablehlo.transpose %v2549, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2552 = stablehlo.convolution(%v2550, %v2551)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2554 = stablehlo.reshape %v2489 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2555 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2556 = stablehlo.reduce(%v2554 init: %v2555) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2557 = stablehlo.broadcast_in_dim %dp10, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2558 = stablehlo.multiply %v2557, %v2494 : tensor<32x75264xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2560 = stablehlo.broadcast_in_dim %s2b4lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2561 = stablehlo.multiply %v2559, %v2560 : tensor<32x384x14x14xf32>
    %v2562 = stablehlo.reshape %v2561 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2564 = stablehlo.reverse %s2b4pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2565 = stablehlo.transpose %v2564, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2566 = stablehlo.convolution(%v2563, %v2565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2567 = stablehlo.reshape %v2566 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2568 = stablehlo.multiply %v850, %v850 : tensor<32x301056xf32>
    %v2569 = stablehlo.multiply %v2568, %v850 : tensor<32x301056xf32>
    %v2570 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2571 = stablehlo.multiply %v2570, %v2569 : tensor<32x301056xf32>
    %v2572 = stablehlo.add %v850, %v2571 : tensor<32x301056xf32>
    %v2573 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2574 = stablehlo.multiply %v2573, %v2572 : tensor<32x301056xf32>
    %v2575 = stablehlo.tanh %v2574 : tensor<32x301056xf32>
    %v2576 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2577 = stablehlo.add %v2576, %v2575 : tensor<32x301056xf32>
    %v2578 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2579 = stablehlo.multiply %v2578, %v2577 : tensor<32x301056xf32>
    %v2580 = stablehlo.multiply %v2575, %v2575 : tensor<32x301056xf32>
    %v2581 = stablehlo.subtract %v2576, %v2580 : tensor<32x301056xf32>
    %v2582 = stablehlo.multiply %v2578, %v850 : tensor<32x301056xf32>
    %v2583 = stablehlo.multiply %v2582, %v2581 : tensor<32x301056xf32>
    %v2584 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2585 = stablehlo.multiply %v2584, %v2568 : tensor<32x301056xf32>
    %v2586 = stablehlo.add %v2576, %v2585 : tensor<32x301056xf32>
    %v2587 = stablehlo.multiply %v2573, %v2586 : tensor<32x301056xf32>
    %v2588 = stablehlo.multiply %v2583, %v2587 : tensor<32x301056xf32>
    %v2589 = stablehlo.add %v2579, %v2588 : tensor<32x301056xf32>
    %v2590 = stablehlo.multiply %v2567, %v2589 : tensor<32x301056xf32>
    %v2591 = stablehlo.reshape %v2590 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2592 = stablehlo.reverse %s2b4eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2593 = stablehlo.transpose %v2592, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2594 = stablehlo.convolution(%v2591, %v2593)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2596 = stablehlo.reshape %v811 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2597 = stablehlo.transpose %v2596, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2599 = stablehlo.reshape %v2595 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2600 = stablehlo.transpose %v2599, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2603 = stablehlo.broadcast_in_dim %s2b4ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2604 = stablehlo.multiply %v2602, %v2603 : tensor<32x196x384xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2607 = stablehlo.reshape %v2598 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2609 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2610 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2611 = stablehlo.reduce(%v2607 init: %v2608) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2612 = stablehlo.broadcast_in_dim %v2611, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2613 = stablehlo.divide %v2612, %v2609 : tensor<32x196x384xf32>
    %v2614 = stablehlo.subtract %v2607, %v2613 : tensor<32x196x384xf32>
    %v2615 = stablehlo.multiply %v2614, %v2614 : tensor<32x196x384xf32>
    %v2616 = stablehlo.reduce(%v2615 init: %v2608) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2618 = stablehlo.divide %v2617, %v2609 : tensor<32x196x384xf32>
    %v2619 = stablehlo.add %v2618, %v2610 : tensor<32x196x384xf32>
    %v2620 = stablehlo.rsqrt %v2619 : tensor<32x196x384xf32>
    %v2621 = stablehlo.multiply %v2614, %v2620 : tensor<32x196x384xf32>
    %v2622 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2623 = stablehlo.multiply %v2622, %v2606 : tensor<32x196x384xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2608) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2626 = stablehlo.multiply %v2621, %v2623 : tensor<32x196x384xf32>
    %v2627 = stablehlo.reduce(%v2626 init: %v2608) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2628 = stablehlo.broadcast_in_dim %v2627, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2629 = stablehlo.multiply %v2623, %v2609 : tensor<32x196x384xf32>
    %v2630 = stablehlo.subtract %v2629, %v2625 : tensor<32x196x384xf32>
    %v2631 = stablehlo.multiply %v2621, %v2628 : tensor<32x196x384xf32>
    %v2632 = stablehlo.subtract %v2630, %v2631 : tensor<32x196x384xf32>
    %v2633 = stablehlo.divide %v2620, %v2609 : tensor<32x196x384xf32>
    %v2634 = stablehlo.multiply %v2633, %v2632 : tensor<32x196x384xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2637 = stablehlo.transpose %v2636, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2638 = stablehlo.reshape %v2637 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2640 = stablehlo.reverse %s2b4dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2641 = stablehlo.convolution(%v2639, %v2640)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2643 = stablehlo.add %v2642, %v2494 : tensor<32x75264xf32>
    %v2644 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2645 = stablehlo.reshape %v868 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2646 = stablehlo.reshape %v2558 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2647 = stablehlo.multiply %v2645, %v2646 : tensor<32x384x14x14xf32>
    %v2648 = stablehlo.reduce(%v2647 init: %v2644) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2649 = stablehlo.reshape %v863 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2650 = stablehlo.reshape %v2562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2651 = stablehlo.transpose %v2649, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2652 = stablehlo.transpose %v2650, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2653 = stablehlo.convolution(%v2651, %v2652)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2654 = stablehlo.transpose %v2653, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2655 = stablehlo.reshape %v2562 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2657 = stablehlo.reduce(%v2655 init: %v2656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2658 = stablehlo.reshape %v845 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2659 = stablehlo.reshape %v2590 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2660 = stablehlo.transpose %v2658, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2661 = stablehlo.transpose %v2659, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2662 = stablehlo.convolution(%v2660, %v2661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2663 = stablehlo.transpose %v2662, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2664 = stablehlo.reshape %v2590 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2666 = stablehlo.reduce(%v2664 init: %v2665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2667 = stablehlo.reshape %v811 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2668 = stablehlo.transpose %v2667, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2669 = stablehlo.reshape %v2668 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2670 = stablehlo.reshape %v2595 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2671 = stablehlo.transpose %v2670, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2672 = stablehlo.reshape %v2671 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2673 = stablehlo.reshape %v2669 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2674 = stablehlo.reshape %v2672 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2676 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2677 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2678 = stablehlo.reduce(%v2673 init: %v2675) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2679 = stablehlo.broadcast_in_dim %v2678, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2680 = stablehlo.divide %v2679, %v2676 : tensor<32x196x384xf32>
    %v2681 = stablehlo.subtract %v2673, %v2680 : tensor<32x196x384xf32>
    %v2682 = stablehlo.multiply %v2681, %v2681 : tensor<32x196x384xf32>
    %v2683 = stablehlo.reduce(%v2682 init: %v2675) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2685 = stablehlo.divide %v2684, %v2676 : tensor<32x196x384xf32>
    %v2686 = stablehlo.add %v2685, %v2677 : tensor<32x196x384xf32>
    %v2687 = stablehlo.rsqrt %v2686 : tensor<32x196x384xf32>
    %v2688 = stablehlo.multiply %v2681, %v2687 : tensor<32x196x384xf32>
    %v2689 = stablehlo.multiply %v2674, %v2688 : tensor<32x196x384xf32>
    %v2690 = stablehlo.reduce(%v2689 init: %v2675) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2691 = stablehlo.reshape %v2595 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2692 = stablehlo.transpose %v2691, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2693 = stablehlo.reshape %v2692 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2695 = stablehlo.reshape %v2693 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2696 = stablehlo.reduce(%v2695 init: %v2694) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2697 = stablehlo.reshape %v806 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2698 = stablehlo.reshape %v2638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2699 = stablehlo.transpose %v2697, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2700 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2701 = stablehlo.convolution(%v2699, %v2700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2702 = stablehlo.reshape %v2701 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2703 = stablehlo.reshape %v2638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2705 = stablehlo.reduce(%v2703 init: %v2704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2706 = stablehlo.broadcast_in_dim %dp9, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2707 = stablehlo.multiply %v2706, %v2643 : tensor<32x75264xf32>
    %v2708 = stablehlo.reshape %v2707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2709 = stablehlo.broadcast_in_dim %s2b3lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2710 = stablehlo.multiply %v2708, %v2709 : tensor<32x384x14x14xf32>
    %v2711 = stablehlo.reshape %v2710 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2712 = stablehlo.reshape %v2711 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2713 = stablehlo.reverse %s2b3pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2714 = stablehlo.transpose %v2713, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2715 = stablehlo.convolution(%v2712, %v2714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2716 = stablehlo.reshape %v2715 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2717 = stablehlo.multiply %v781, %v781 : tensor<32x301056xf32>
    %v2718 = stablehlo.multiply %v2717, %v781 : tensor<32x301056xf32>
    %v2719 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2720 = stablehlo.multiply %v2719, %v2718 : tensor<32x301056xf32>
    %v2721 = stablehlo.add %v781, %v2720 : tensor<32x301056xf32>
    %v2722 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2723 = stablehlo.multiply %v2722, %v2721 : tensor<32x301056xf32>
    %v2724 = stablehlo.tanh %v2723 : tensor<32x301056xf32>
    %v2725 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2726 = stablehlo.add %v2725, %v2724 : tensor<32x301056xf32>
    %v2727 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2728 = stablehlo.multiply %v2727, %v2726 : tensor<32x301056xf32>
    %v2729 = stablehlo.multiply %v2724, %v2724 : tensor<32x301056xf32>
    %v2730 = stablehlo.subtract %v2725, %v2729 : tensor<32x301056xf32>
    %v2731 = stablehlo.multiply %v2727, %v781 : tensor<32x301056xf32>
    %v2732 = stablehlo.multiply %v2731, %v2730 : tensor<32x301056xf32>
    %v2733 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2734 = stablehlo.multiply %v2733, %v2717 : tensor<32x301056xf32>
    %v2735 = stablehlo.add %v2725, %v2734 : tensor<32x301056xf32>
    %v2736 = stablehlo.multiply %v2722, %v2735 : tensor<32x301056xf32>
    %v2737 = stablehlo.multiply %v2732, %v2736 : tensor<32x301056xf32>
    %v2738 = stablehlo.add %v2728, %v2737 : tensor<32x301056xf32>
    %v2739 = stablehlo.multiply %v2716, %v2738 : tensor<32x301056xf32>
    %v2740 = stablehlo.reshape %v2739 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2741 = stablehlo.reverse %s2b3eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2742 = stablehlo.transpose %v2741, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2743 = stablehlo.convolution(%v2740, %v2742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2744 = stablehlo.reshape %v2743 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2745 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2746 = stablehlo.transpose %v2745, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2747 = stablehlo.reshape %v2746 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2748 = stablehlo.reshape %v2744 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2749 = stablehlo.transpose %v2748, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2752 = stablehlo.broadcast_in_dim %s2b3ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2753 = stablehlo.multiply %v2751, %v2752 : tensor<32x196x384xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2756 = stablehlo.reshape %v2747 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2758 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2759 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2760 = stablehlo.reduce(%v2756 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2761 = stablehlo.broadcast_in_dim %v2760, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2762 = stablehlo.divide %v2761, %v2758 : tensor<32x196x384xf32>
    %v2763 = stablehlo.subtract %v2756, %v2762 : tensor<32x196x384xf32>
    %v2764 = stablehlo.multiply %v2763, %v2763 : tensor<32x196x384xf32>
    %v2765 = stablehlo.reduce(%v2764 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2766 = stablehlo.broadcast_in_dim %v2765, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2767 = stablehlo.divide %v2766, %v2758 : tensor<32x196x384xf32>
    %v2768 = stablehlo.add %v2767, %v2759 : tensor<32x196x384xf32>
    %v2769 = stablehlo.rsqrt %v2768 : tensor<32x196x384xf32>
    %v2770 = stablehlo.multiply %v2763, %v2769 : tensor<32x196x384xf32>
    %v2771 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2772 = stablehlo.multiply %v2771, %v2755 : tensor<32x196x384xf32>
    %v2773 = stablehlo.reduce(%v2772 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2773, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2775 = stablehlo.multiply %v2770, %v2772 : tensor<32x196x384xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2757) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2778 = stablehlo.multiply %v2772, %v2758 : tensor<32x196x384xf32>
    %v2779 = stablehlo.subtract %v2778, %v2774 : tensor<32x196x384xf32>
    %v2780 = stablehlo.multiply %v2770, %v2777 : tensor<32x196x384xf32>
    %v2781 = stablehlo.subtract %v2779, %v2780 : tensor<32x196x384xf32>
    %v2782 = stablehlo.divide %v2769, %v2758 : tensor<32x196x384xf32>
    %v2783 = stablehlo.multiply %v2782, %v2781 : tensor<32x196x384xf32>
    %v2784 = stablehlo.reshape %v2783 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2785 = stablehlo.reshape %v2784 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2786 = stablehlo.transpose %v2785, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2787 = stablehlo.reshape %v2786 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2788 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2789 = stablehlo.reverse %s2b3dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2790 = stablehlo.convolution(%v2788, %v2789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2791 = stablehlo.reshape %v2790 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2792 = stablehlo.add %v2791, %v2643 : tensor<32x75264xf32>
    %v2793 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2794 = stablehlo.reshape %v799 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2795 = stablehlo.reshape %v2707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2796 = stablehlo.multiply %v2794, %v2795 : tensor<32x384x14x14xf32>
    %v2797 = stablehlo.reduce(%v2796 init: %v2793) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2798 = stablehlo.reshape %v794 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2799 = stablehlo.reshape %v2711 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2800 = stablehlo.transpose %v2798, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2801 = stablehlo.transpose %v2799, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2802 = stablehlo.convolution(%v2800, %v2801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2803 = stablehlo.transpose %v2802, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2804 = stablehlo.reshape %v2711 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2806 = stablehlo.reduce(%v2804 init: %v2805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2807 = stablehlo.reshape %v776 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2808 = stablehlo.reshape %v2739 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2809 = stablehlo.transpose %v2807, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2810 = stablehlo.transpose %v2808, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2811 = stablehlo.convolution(%v2809, %v2810)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2812 = stablehlo.transpose %v2811, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2813 = stablehlo.reshape %v2739 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2815 = stablehlo.reduce(%v2813 init: %v2814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2816 = stablehlo.reshape %v742 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2817 = stablehlo.transpose %v2816, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2818 = stablehlo.reshape %v2817 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2819 = stablehlo.reshape %v2744 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2820 = stablehlo.transpose %v2819, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2822 = stablehlo.reshape %v2818 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2823 = stablehlo.reshape %v2821 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2825 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2826 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2827 = stablehlo.reduce(%v2822 init: %v2824) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2828 = stablehlo.broadcast_in_dim %v2827, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2829 = stablehlo.divide %v2828, %v2825 : tensor<32x196x384xf32>
    %v2830 = stablehlo.subtract %v2822, %v2829 : tensor<32x196x384xf32>
    %v2831 = stablehlo.multiply %v2830, %v2830 : tensor<32x196x384xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2824) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2834 = stablehlo.divide %v2833, %v2825 : tensor<32x196x384xf32>
    %v2835 = stablehlo.add %v2834, %v2826 : tensor<32x196x384xf32>
    %v2836 = stablehlo.rsqrt %v2835 : tensor<32x196x384xf32>
    %v2837 = stablehlo.multiply %v2830, %v2836 : tensor<32x196x384xf32>
    %v2838 = stablehlo.multiply %v2823, %v2837 : tensor<32x196x384xf32>
    %v2839 = stablehlo.reduce(%v2838 init: %v2824) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2840 = stablehlo.reshape %v2744 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2841 = stablehlo.transpose %v2840, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2842 = stablehlo.reshape %v2841 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2844 = stablehlo.reshape %v2842 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2845 = stablehlo.reduce(%v2844 init: %v2843) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2846 = stablehlo.reshape %v737 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2847 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2848 = stablehlo.transpose %v2846, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2849 = stablehlo.transpose %v2847, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2850 = stablehlo.convolution(%v2848, %v2849)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v2851 = stablehlo.reshape %v2850 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v2852 = stablehlo.reshape %v2787 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2854 = stablehlo.reduce(%v2852 init: %v2853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2855 = stablehlo.broadcast_in_dim %dp8, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v2856 = stablehlo.multiply %v2855, %v2792 : tensor<32x75264xf32>
    %v2857 = stablehlo.reshape %v2856 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2858 = stablehlo.broadcast_in_dim %s2b2lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v2859 = stablehlo.multiply %v2857, %v2858 : tensor<32x384x14x14xf32>
    %v2860 = stablehlo.reshape %v2859 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2861 = stablehlo.reshape %v2860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2862 = stablehlo.reverse %s2b2pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v2863 = stablehlo.transpose %v2862, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2864 = stablehlo.convolution(%v2861, %v2863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v2865 = stablehlo.reshape %v2864 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v2866 = stablehlo.multiply %v712, %v712 : tensor<32x301056xf32>
    %v2867 = stablehlo.multiply %v2866, %v712 : tensor<32x301056xf32>
    %v2868 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v2869 = stablehlo.multiply %v2868, %v2867 : tensor<32x301056xf32>
    %v2870 = stablehlo.add %v712, %v2869 : tensor<32x301056xf32>
    %v2871 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v2872 = stablehlo.multiply %v2871, %v2870 : tensor<32x301056xf32>
    %v2873 = stablehlo.tanh %v2872 : tensor<32x301056xf32>
    %v2874 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v2875 = stablehlo.add %v2874, %v2873 : tensor<32x301056xf32>
    %v2876 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v2877 = stablehlo.multiply %v2876, %v2875 : tensor<32x301056xf32>
    %v2878 = stablehlo.multiply %v2873, %v2873 : tensor<32x301056xf32>
    %v2879 = stablehlo.subtract %v2874, %v2878 : tensor<32x301056xf32>
    %v2880 = stablehlo.multiply %v2876, %v712 : tensor<32x301056xf32>
    %v2881 = stablehlo.multiply %v2880, %v2879 : tensor<32x301056xf32>
    %v2882 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v2883 = stablehlo.multiply %v2882, %v2866 : tensor<32x301056xf32>
    %v2884 = stablehlo.add %v2874, %v2883 : tensor<32x301056xf32>
    %v2885 = stablehlo.multiply %v2871, %v2884 : tensor<32x301056xf32>
    %v2886 = stablehlo.multiply %v2881, %v2885 : tensor<32x301056xf32>
    %v2887 = stablehlo.add %v2877, %v2886 : tensor<32x301056xf32>
    %v2888 = stablehlo.multiply %v2865, %v2887 : tensor<32x301056xf32>
    %v2889 = stablehlo.reshape %v2888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2890 = stablehlo.reverse %s2b2eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v2891 = stablehlo.transpose %v2890, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2892 = stablehlo.convolution(%v2889, %v2891)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2894 = stablehlo.reshape %v673 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2895 = stablehlo.transpose %v2894, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2897 = stablehlo.reshape %v2893 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2898 = stablehlo.transpose %v2897, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2899 = stablehlo.reshape %v2898 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2900 = stablehlo.reshape %v2899 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2901 = stablehlo.broadcast_in_dim %s2b2ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v2902 = stablehlo.multiply %v2900, %v2901 : tensor<32x196x384xf32>
    %v2903 = stablehlo.reshape %v2902 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2904 = stablehlo.reshape %v2903 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2905 = stablehlo.reshape %v2896 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2907 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2908 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2909 = stablehlo.reduce(%v2905 init: %v2906) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2910 = stablehlo.broadcast_in_dim %v2909, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2911 = stablehlo.divide %v2910, %v2907 : tensor<32x196x384xf32>
    %v2912 = stablehlo.subtract %v2905, %v2911 : tensor<32x196x384xf32>
    %v2913 = stablehlo.multiply %v2912, %v2912 : tensor<32x196x384xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2906) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2916 = stablehlo.divide %v2915, %v2907 : tensor<32x196x384xf32>
    %v2917 = stablehlo.add %v2916, %v2908 : tensor<32x196x384xf32>
    %v2918 = stablehlo.rsqrt %v2917 : tensor<32x196x384xf32>
    %v2919 = stablehlo.multiply %v2912, %v2918 : tensor<32x196x384xf32>
    %v2920 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v2921 = stablehlo.multiply %v2920, %v2904 : tensor<32x196x384xf32>
    %v2922 = stablehlo.reduce(%v2921 init: %v2906) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2923 = stablehlo.broadcast_in_dim %v2922, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2924 = stablehlo.multiply %v2919, %v2921 : tensor<32x196x384xf32>
    %v2925 = stablehlo.reduce(%v2924 init: %v2906) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2927 = stablehlo.multiply %v2921, %v2907 : tensor<32x196x384xf32>
    %v2928 = stablehlo.subtract %v2927, %v2923 : tensor<32x196x384xf32>
    %v2929 = stablehlo.multiply %v2919, %v2926 : tensor<32x196x384xf32>
    %v2930 = stablehlo.subtract %v2928, %v2929 : tensor<32x196x384xf32>
    %v2931 = stablehlo.divide %v2918, %v2907 : tensor<32x196x384xf32>
    %v2932 = stablehlo.multiply %v2931, %v2930 : tensor<32x196x384xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2934 = stablehlo.reshape %v2933 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2935 = stablehlo.transpose %v2934, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v2936 = stablehlo.reshape %v2935 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v2937 = stablehlo.reshape %v2936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2938 = stablehlo.reverse %s2b2dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v2939 = stablehlo.convolution(%v2937, %v2938)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v2940 = stablehlo.reshape %v2939 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v2941 = stablehlo.add %v2940, %v2792 : tensor<32x75264xf32>
    %v2942 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2943 = stablehlo.reshape %v730 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2944 = stablehlo.reshape %v2856 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2945 = stablehlo.multiply %v2943, %v2944 : tensor<32x384x14x14xf32>
    %v2946 = stablehlo.reduce(%v2945 init: %v2942) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2947 = stablehlo.reshape %v725 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2948 = stablehlo.reshape %v2860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2949 = stablehlo.transpose %v2947, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2950 = stablehlo.transpose %v2948, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2951 = stablehlo.convolution(%v2949, %v2950)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v2952 = stablehlo.transpose %v2951, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v2953 = stablehlo.reshape %v2860 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2955 = stablehlo.reduce(%v2953 init: %v2954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v2956 = stablehlo.reshape %v707 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2957 = stablehlo.reshape %v2888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2958 = stablehlo.transpose %v2956, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2959 = stablehlo.transpose %v2957, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v2960 = stablehlo.convolution(%v2958, %v2959)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v2961 = stablehlo.transpose %v2960, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v2962 = stablehlo.reshape %v2888 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v2963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2964 = stablehlo.reduce(%v2962 init: %v2963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v2965 = stablehlo.reshape %v673 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2966 = stablehlo.transpose %v2965, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2968 = stablehlo.reshape %v2893 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2969 = stablehlo.transpose %v2968, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2971 = stablehlo.reshape %v2967 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2972 = stablehlo.reshape %v2970 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2974 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v2975 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v2976 = stablehlo.reduce(%v2971 init: %v2973) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2977 = stablehlo.broadcast_in_dim %v2976, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2978 = stablehlo.divide %v2977, %v2974 : tensor<32x196x384xf32>
    %v2979 = stablehlo.subtract %v2971, %v2978 : tensor<32x196x384xf32>
    %v2980 = stablehlo.multiply %v2979, %v2979 : tensor<32x196x384xf32>
    %v2981 = stablehlo.reduce(%v2980 init: %v2973) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2981, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v2983 = stablehlo.divide %v2982, %v2974 : tensor<32x196x384xf32>
    %v2984 = stablehlo.add %v2983, %v2975 : tensor<32x196x384xf32>
    %v2985 = stablehlo.rsqrt %v2984 : tensor<32x196x384xf32>
    %v2986 = stablehlo.multiply %v2979, %v2985 : tensor<32x196x384xf32>
    %v2987 = stablehlo.multiply %v2972, %v2986 : tensor<32x196x384xf32>
    %v2988 = stablehlo.reduce(%v2987 init: %v2973) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2989 = stablehlo.reshape %v2893 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v2990 = stablehlo.transpose %v2989, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v2992 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2993 = stablehlo.reshape %v2991 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v2994 = stablehlo.reduce(%v2993 init: %v2992) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v2995 = stablehlo.reshape %v668 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2996 = stablehlo.reshape %v2936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v2997 = stablehlo.transpose %v2995, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2998 = stablehlo.transpose %v2996, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v2999 = stablehlo.convolution(%v2997, %v2998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3000 = stablehlo.reshape %v2999 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3001 = stablehlo.reshape %v2936 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3003 = stablehlo.reduce(%v3001 init: %v3002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3004 = stablehlo.broadcast_in_dim %dp7, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v3005 = stablehlo.multiply %v3004, %v2941 : tensor<32x75264xf32>
    %v3006 = stablehlo.reshape %v3005 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3007 = stablehlo.broadcast_in_dim %s2b1lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3008 = stablehlo.multiply %v3006, %v3007 : tensor<32x384x14x14xf32>
    %v3009 = stablehlo.reshape %v3008 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3010 = stablehlo.reshape %v3009 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3011 = stablehlo.reverse %s2b1pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3012 = stablehlo.transpose %v3011, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3013 = stablehlo.convolution(%v3010, %v3012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3014 = stablehlo.reshape %v3013 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3015 = stablehlo.multiply %v643, %v643 : tensor<32x301056xf32>
    %v3016 = stablehlo.multiply %v3015, %v643 : tensor<32x301056xf32>
    %v3017 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v3018 = stablehlo.multiply %v3017, %v3016 : tensor<32x301056xf32>
    %v3019 = stablehlo.add %v643, %v3018 : tensor<32x301056xf32>
    %v3020 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v3021 = stablehlo.multiply %v3020, %v3019 : tensor<32x301056xf32>
    %v3022 = stablehlo.tanh %v3021 : tensor<32x301056xf32>
    %v3023 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v3024 = stablehlo.add %v3023, %v3022 : tensor<32x301056xf32>
    %v3025 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v3026 = stablehlo.multiply %v3025, %v3024 : tensor<32x301056xf32>
    %v3027 = stablehlo.multiply %v3022, %v3022 : tensor<32x301056xf32>
    %v3028 = stablehlo.subtract %v3023, %v3027 : tensor<32x301056xf32>
    %v3029 = stablehlo.multiply %v3025, %v643 : tensor<32x301056xf32>
    %v3030 = stablehlo.multiply %v3029, %v3028 : tensor<32x301056xf32>
    %v3031 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v3032 = stablehlo.multiply %v3031, %v3015 : tensor<32x301056xf32>
    %v3033 = stablehlo.add %v3023, %v3032 : tensor<32x301056xf32>
    %v3034 = stablehlo.multiply %v3020, %v3033 : tensor<32x301056xf32>
    %v3035 = stablehlo.multiply %v3030, %v3034 : tensor<32x301056xf32>
    %v3036 = stablehlo.add %v3026, %v3035 : tensor<32x301056xf32>
    %v3037 = stablehlo.multiply %v3014, %v3036 : tensor<32x301056xf32>
    %v3038 = stablehlo.reshape %v3037 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3039 = stablehlo.reverse %s2b1eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3040 = stablehlo.transpose %v3039, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3041 = stablehlo.convolution(%v3038, %v3040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3042 = stablehlo.reshape %v3041 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3043 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3044 = stablehlo.transpose %v3043, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3045 = stablehlo.reshape %v3044 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3046 = stablehlo.reshape %v3042 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3047 = stablehlo.transpose %v3046, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3048 = stablehlo.reshape %v3047 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3050 = stablehlo.broadcast_in_dim %s2b1ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3051 = stablehlo.multiply %v3049, %v3050 : tensor<32x196x384xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3053 = stablehlo.reshape %v3052 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3054 = stablehlo.reshape %v3045 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3056 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3057 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3058 = stablehlo.reduce(%v3054 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3059 = stablehlo.broadcast_in_dim %v3058, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3060 = stablehlo.divide %v3059, %v3056 : tensor<32x196x384xf32>
    %v3061 = stablehlo.subtract %v3054, %v3060 : tensor<32x196x384xf32>
    %v3062 = stablehlo.multiply %v3061, %v3061 : tensor<32x196x384xf32>
    %v3063 = stablehlo.reduce(%v3062 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3064 = stablehlo.broadcast_in_dim %v3063, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3065 = stablehlo.divide %v3064, %v3056 : tensor<32x196x384xf32>
    %v3066 = stablehlo.add %v3065, %v3057 : tensor<32x196x384xf32>
    %v3067 = stablehlo.rsqrt %v3066 : tensor<32x196x384xf32>
    %v3068 = stablehlo.multiply %v3061, %v3067 : tensor<32x196x384xf32>
    %v3069 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3070 = stablehlo.multiply %v3069, %v3053 : tensor<32x196x384xf32>
    %v3071 = stablehlo.reduce(%v3070 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3072 = stablehlo.broadcast_in_dim %v3071, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3073 = stablehlo.multiply %v3068, %v3070 : tensor<32x196x384xf32>
    %v3074 = stablehlo.reduce(%v3073 init: %v3055) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3075 = stablehlo.broadcast_in_dim %v3074, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3076 = stablehlo.multiply %v3070, %v3056 : tensor<32x196x384xf32>
    %v3077 = stablehlo.subtract %v3076, %v3072 : tensor<32x196x384xf32>
    %v3078 = stablehlo.multiply %v3068, %v3075 : tensor<32x196x384xf32>
    %v3079 = stablehlo.subtract %v3077, %v3078 : tensor<32x196x384xf32>
    %v3080 = stablehlo.divide %v3067, %v3056 : tensor<32x196x384xf32>
    %v3081 = stablehlo.multiply %v3080, %v3079 : tensor<32x196x384xf32>
    %v3082 = stablehlo.reshape %v3081 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3083 = stablehlo.reshape %v3082 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3084 = stablehlo.transpose %v3083, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3085 = stablehlo.reshape %v3084 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3086 = stablehlo.reshape %v3085 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3087 = stablehlo.reverse %s2b1dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3088 = stablehlo.convolution(%v3086, %v3087)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3089 = stablehlo.reshape %v3088 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3090 = stablehlo.add %v3089, %v2941 : tensor<32x75264xf32>
    %v3091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3092 = stablehlo.reshape %v661 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3093 = stablehlo.reshape %v3005 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3094 = stablehlo.multiply %v3092, %v3093 : tensor<32x384x14x14xf32>
    %v3095 = stablehlo.reduce(%v3094 init: %v3091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3096 = stablehlo.reshape %v656 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3097 = stablehlo.reshape %v3009 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3098 = stablehlo.transpose %v3096, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3099 = stablehlo.transpose %v3097, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3100 = stablehlo.convolution(%v3098, %v3099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3101 = stablehlo.transpose %v3100, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3102 = stablehlo.reshape %v3009 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3104 = stablehlo.reduce(%v3102 init: %v3103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3105 = stablehlo.reshape %v638 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3106 = stablehlo.reshape %v3037 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3107 = stablehlo.transpose %v3105, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3108 = stablehlo.transpose %v3106, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3109 = stablehlo.convolution(%v3107, %v3108)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3110 = stablehlo.transpose %v3109, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3111 = stablehlo.reshape %v3037 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3113 = stablehlo.reduce(%v3111 init: %v3112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3114 = stablehlo.reshape %v604 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3115 = stablehlo.transpose %v3114, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3116 = stablehlo.reshape %v3115 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3117 = stablehlo.reshape %v3042 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3118 = stablehlo.transpose %v3117, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3120 = stablehlo.reshape %v3116 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3121 = stablehlo.reshape %v3119 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3123 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3124 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3125 = stablehlo.reduce(%v3120 init: %v3122) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3126 = stablehlo.broadcast_in_dim %v3125, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3127 = stablehlo.divide %v3126, %v3123 : tensor<32x196x384xf32>
    %v3128 = stablehlo.subtract %v3120, %v3127 : tensor<32x196x384xf32>
    %v3129 = stablehlo.multiply %v3128, %v3128 : tensor<32x196x384xf32>
    %v3130 = stablehlo.reduce(%v3129 init: %v3122) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3131 = stablehlo.broadcast_in_dim %v3130, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3132 = stablehlo.divide %v3131, %v3123 : tensor<32x196x384xf32>
    %v3133 = stablehlo.add %v3132, %v3124 : tensor<32x196x384xf32>
    %v3134 = stablehlo.rsqrt %v3133 : tensor<32x196x384xf32>
    %v3135 = stablehlo.multiply %v3128, %v3134 : tensor<32x196x384xf32>
    %v3136 = stablehlo.multiply %v3121, %v3135 : tensor<32x196x384xf32>
    %v3137 = stablehlo.reduce(%v3136 init: %v3122) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3138 = stablehlo.reshape %v3042 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3139 = stablehlo.transpose %v3138, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3140 = stablehlo.reshape %v3139 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3142 = stablehlo.reshape %v3140 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3143 = stablehlo.reduce(%v3142 init: %v3141) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3144 = stablehlo.reshape %v599 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3145 = stablehlo.reshape %v3085 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3146 = stablehlo.transpose %v3144, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3147 = stablehlo.transpose %v3145, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3148 = stablehlo.convolution(%v3146, %v3147)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3149 = stablehlo.reshape %v3148 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3150 = stablehlo.reshape %v3085 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3152 = stablehlo.reduce(%v3150 init: %v3151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3153 = stablehlo.broadcast_in_dim %dp6, dims = [0] : (tensor<32xf32>) -> tensor<32x75264xf32>
    %v3154 = stablehlo.multiply %v3153, %v3090 : tensor<32x75264xf32>
    %v3155 = stablehlo.reshape %v3154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3156 = stablehlo.broadcast_in_dim %s2b0lg, dims = [1] : (tensor<384xf32>) -> tensor<32x384x14x14xf32>
    %v3157 = stablehlo.multiply %v3155, %v3156 : tensor<32x384x14x14xf32>
    %v3158 = stablehlo.reshape %v3157 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3159 = stablehlo.reshape %v3158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3160 = stablehlo.reverse %s2b0pW, dims = [2, 3] : tensor<384x1536x1x1xf32>
    %v3161 = stablehlo.transpose %v3160, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3162 = stablehlo.convolution(%v3159, %v3161)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x14x14xf32>, tensor<1536x384x1x1xf32>) -> tensor<32x1536x14x14xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<32x1536x14x14xf32>) -> tensor<32x301056xf32>
    %v3164 = stablehlo.multiply %v574, %v574 : tensor<32x301056xf32>
    %v3165 = stablehlo.multiply %v3164, %v574 : tensor<32x301056xf32>
    %v3166 = stablehlo.constant dense<0.044715> : tensor<32x301056xf32>
    %v3167 = stablehlo.multiply %v3166, %v3165 : tensor<32x301056xf32>
    %v3168 = stablehlo.add %v574, %v3167 : tensor<32x301056xf32>
    %v3169 = stablehlo.constant dense<0.7978845608028654> : tensor<32x301056xf32>
    %v3170 = stablehlo.multiply %v3169, %v3168 : tensor<32x301056xf32>
    %v3171 = stablehlo.tanh %v3170 : tensor<32x301056xf32>
    %v3172 = stablehlo.constant dense<1.0> : tensor<32x301056xf32>
    %v3173 = stablehlo.add %v3172, %v3171 : tensor<32x301056xf32>
    %v3174 = stablehlo.constant dense<0.5> : tensor<32x301056xf32>
    %v3175 = stablehlo.multiply %v3174, %v3173 : tensor<32x301056xf32>
    %v3176 = stablehlo.multiply %v3171, %v3171 : tensor<32x301056xf32>
    %v3177 = stablehlo.subtract %v3172, %v3176 : tensor<32x301056xf32>
    %v3178 = stablehlo.multiply %v3174, %v574 : tensor<32x301056xf32>
    %v3179 = stablehlo.multiply %v3178, %v3177 : tensor<32x301056xf32>
    %v3180 = stablehlo.constant dense<0.134145> : tensor<32x301056xf32>
    %v3181 = stablehlo.multiply %v3180, %v3164 : tensor<32x301056xf32>
    %v3182 = stablehlo.add %v3172, %v3181 : tensor<32x301056xf32>
    %v3183 = stablehlo.multiply %v3169, %v3182 : tensor<32x301056xf32>
    %v3184 = stablehlo.multiply %v3179, %v3183 : tensor<32x301056xf32>
    %v3185 = stablehlo.add %v3175, %v3184 : tensor<32x301056xf32>
    %v3186 = stablehlo.multiply %v3163, %v3185 : tensor<32x301056xf32>
    %v3187 = stablehlo.reshape %v3186 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3188 = stablehlo.reverse %s2b0eW, dims = [2, 3] : tensor<1536x384x1x1xf32>
    %v3189 = stablehlo.transpose %v3188, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3190 = stablehlo.convolution(%v3187, %v3189)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x1536x14x14xf32>, tensor<384x1536x1x1xf32>) -> tensor<32x384x14x14xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3192 = stablehlo.reshape %v535 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3193 = stablehlo.transpose %v3192, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3194 = stablehlo.reshape %v3193 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3195 = stablehlo.reshape %v3191 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3196 = stablehlo.transpose %v3195, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3197 = stablehlo.reshape %v3196 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3198 = stablehlo.reshape %v3197 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3199 = stablehlo.broadcast_in_dim %s2b0ng, dims = [2] : (tensor<384xf32>) -> tensor<32x196x384xf32>
    %v3200 = stablehlo.multiply %v3198, %v3199 : tensor<32x196x384xf32>
    %v3201 = stablehlo.reshape %v3200 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3202 = stablehlo.reshape %v3201 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3203 = stablehlo.reshape %v3194 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3205 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3206 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3207 = stablehlo.reduce(%v3203 init: %v3204) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3208 = stablehlo.broadcast_in_dim %v3207, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3209 = stablehlo.divide %v3208, %v3205 : tensor<32x196x384xf32>
    %v3210 = stablehlo.subtract %v3203, %v3209 : tensor<32x196x384xf32>
    %v3211 = stablehlo.multiply %v3210, %v3210 : tensor<32x196x384xf32>
    %v3212 = stablehlo.reduce(%v3211 init: %v3204) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3213 = stablehlo.broadcast_in_dim %v3212, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3214 = stablehlo.divide %v3213, %v3205 : tensor<32x196x384xf32>
    %v3215 = stablehlo.add %v3214, %v3206 : tensor<32x196x384xf32>
    %v3216 = stablehlo.rsqrt %v3215 : tensor<32x196x384xf32>
    %v3217 = stablehlo.multiply %v3210, %v3216 : tensor<32x196x384xf32>
    %v3218 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x196x384xf32>
    %v3219 = stablehlo.multiply %v3218, %v3202 : tensor<32x196x384xf32>
    %v3220 = stablehlo.reduce(%v3219 init: %v3204) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3221 = stablehlo.broadcast_in_dim %v3220, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3222 = stablehlo.multiply %v3217, %v3219 : tensor<32x196x384xf32>
    %v3223 = stablehlo.reduce(%v3222 init: %v3204) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3224 = stablehlo.broadcast_in_dim %v3223, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3225 = stablehlo.multiply %v3219, %v3205 : tensor<32x196x384xf32>
    %v3226 = stablehlo.subtract %v3225, %v3221 : tensor<32x196x384xf32>
    %v3227 = stablehlo.multiply %v3217, %v3224 : tensor<32x196x384xf32>
    %v3228 = stablehlo.subtract %v3226, %v3227 : tensor<32x196x384xf32>
    %v3229 = stablehlo.divide %v3216, %v3205 : tensor<32x196x384xf32>
    %v3230 = stablehlo.multiply %v3229, %v3228 : tensor<32x196x384xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3232 = stablehlo.reshape %v3231 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3233 = stablehlo.transpose %v3232, dims = [0, 2, 1] : (tensor<32x196x384xf32>) -> tensor<32x384x196xf32>
    %v3234 = stablehlo.reshape %v3233 : (tensor<32x384x196xf32>) -> tensor<32x75264xf32>
    %v3235 = stablehlo.reshape %v3234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3236 = stablehlo.reverse %s2b0dW, dims = [2, 3] : tensor<384x1x7x7xf32>
    %v3237 = stablehlo.convolution(%v3235, %v3236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 384 : i64} : (tensor<32x384x14x14xf32>, tensor<384x1x7x7xf32>) -> tensor<32x384x14x14xf32>
    %v3238 = stablehlo.reshape %v3237 : (tensor<32x384x14x14xf32>) -> tensor<32x75264xf32>
    %v3239 = stablehlo.add %v3238, %v3090 : tensor<32x75264xf32>
    %v3240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3241 = stablehlo.reshape %v592 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3242 = stablehlo.reshape %v3154 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3243 = stablehlo.multiply %v3241, %v3242 : tensor<32x384x14x14xf32>
    %v3244 = stablehlo.reduce(%v3243 init: %v3240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3245 = stablehlo.reshape %v587 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3246 = stablehlo.reshape %v3158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3247 = stablehlo.transpose %v3245, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3248 = stablehlo.transpose %v3246, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3249 = stablehlo.convolution(%v3247, %v3248)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1536x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1536x384x1x1xf32>
    %v3250 = stablehlo.transpose %v3249, dims = [1, 0, 2, 3] : (tensor<1536x384x1x1xf32>) -> tensor<384x1536x1x1xf32>
    %v3251 = stablehlo.reshape %v3158 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3253 = stablehlo.reduce(%v3251 init: %v3252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3254 = stablehlo.reshape %v569 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3255 = stablehlo.reshape %v3186 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3256 = stablehlo.transpose %v3254, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3257 = stablehlo.transpose %v3255, dims = [1, 0, 2, 3] : (tensor<32x1536x14x14xf32>) -> tensor<1536x32x14x14xf32>
    %v3258 = stablehlo.convolution(%v3256, %v3257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<1536x32x14x14xf32>) -> tensor<384x1536x1x1xf32>
    %v3259 = stablehlo.transpose %v3258, dims = [1, 0, 2, 3] : (tensor<384x1536x1x1xf32>) -> tensor<1536x384x1x1xf32>
    %v3260 = stablehlo.reshape %v3186 : (tensor<32x301056xf32>) -> tensor<32x1536x14x14xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.reduce(%v3260 init: %v3261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x1536x14x14xf32>, tensor<f32>) -> tensor<1536xf32>
    %v3263 = stablehlo.reshape %v535 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3264 = stablehlo.transpose %v3263, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3266 = stablehlo.reshape %v3191 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3267 = stablehlo.transpose %v3266, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3268 = stablehlo.reshape %v3267 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3269 = stablehlo.reshape %v3265 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3270 = stablehlo.reshape %v3268 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3272 = stablehlo.constant dense<384.0> : tensor<32x196x384xf32>
    %v3273 = stablehlo.constant dense<1.0e-6> : tensor<32x196x384xf32>
    %v3274 = stablehlo.reduce(%v3269 init: %v3271) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3275 = stablehlo.broadcast_in_dim %v3274, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3276 = stablehlo.divide %v3275, %v3272 : tensor<32x196x384xf32>
    %v3277 = stablehlo.subtract %v3269, %v3276 : tensor<32x196x384xf32>
    %v3278 = stablehlo.multiply %v3277, %v3277 : tensor<32x196x384xf32>
    %v3279 = stablehlo.reduce(%v3278 init: %v3271) applies stablehlo.add across dimensions = [2] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<32x196xf32>
    %v3280 = stablehlo.broadcast_in_dim %v3279, dims = [0, 1] : (tensor<32x196xf32>) -> tensor<32x196x384xf32>
    %v3281 = stablehlo.divide %v3280, %v3272 : tensor<32x196x384xf32>
    %v3282 = stablehlo.add %v3281, %v3273 : tensor<32x196x384xf32>
    %v3283 = stablehlo.rsqrt %v3282 : tensor<32x196x384xf32>
    %v3284 = stablehlo.multiply %v3277, %v3283 : tensor<32x196x384xf32>
    %v3285 = stablehlo.multiply %v3270, %v3284 : tensor<32x196x384xf32>
    %v3286 = stablehlo.reduce(%v3285 init: %v3271) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3287 = stablehlo.reshape %v3191 : (tensor<32x75264xf32>) -> tensor<32x384x196xf32>
    %v3288 = stablehlo.transpose %v3287, dims = [0, 2, 1] : (tensor<32x384x196xf32>) -> tensor<32x196x384xf32>
    %v3289 = stablehlo.reshape %v3288 : (tensor<32x196x384xf32>) -> tensor<32x75264xf32>
    %v3290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3291 = stablehlo.reshape %v3289 : (tensor<32x75264xf32>) -> tensor<32x196x384xf32>
    %v3292 = stablehlo.reduce(%v3291 init: %v3290) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x196x384xf32>, tensor<f32>) -> tensor<384xf32>
    %v3293 = stablehlo.reshape %v530 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3294 = stablehlo.reshape %v3234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3295 = stablehlo.transpose %v3293, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3296 = stablehlo.transpose %v3294, dims = [1, 0, 2, 3] : (tensor<32x384x14x14xf32>) -> tensor<384x32x14x14xf32>
    %v3297 = stablehlo.convolution(%v3295, %v3296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 384 : i64, feature_group_count = 1 : i64} : (tensor<384x32x14x14xf32>, tensor<384x32x14x14xf32>) -> tensor<1x384x7x7xf32>
    %v3298 = stablehlo.reshape %v3297 : (tensor<1x384x7x7xf32>) -> tensor<384x1x7x7xf32>
    %v3299 = stablehlo.reshape %v3234 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3301 = stablehlo.reduce(%v3299 init: %v3300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3302 = stablehlo.reshape %v3239 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3304 = stablehlo.pad %v3302, %v3303, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x28x28xf32>
    %v3305 = stablehlo.reverse %d1W, dims = [2, 3] : tensor<384x192x2x2xf32>
    %v3306 = stablehlo.transpose %v3305, dims = [1, 0, 2, 3] : (tensor<384x192x2x2xf32>) -> tensor<192x384x2x2xf32>
    %v3307 = stablehlo.convolution(%v3304, %v3306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x28x28xf32>, tensor<192x384x2x2xf32>) -> tensor<32x192x28x28xf32>
    %v3308 = stablehlo.reshape %v3307 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3309 = stablehlo.reshape %v491 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3310 = stablehlo.transpose %v3309, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3311 = stablehlo.reshape %v3310 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3312 = stablehlo.reshape %v3308 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3313 = stablehlo.transpose %v3312, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3314 = stablehlo.reshape %v3313 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3315 = stablehlo.reshape %v3314 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3316 = stablehlo.broadcast_in_dim %d1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3317 = stablehlo.multiply %v3315, %v3316 : tensor<32x784x192xf32>
    %v3318 = stablehlo.reshape %v3317 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3319 = stablehlo.reshape %v3318 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3320 = stablehlo.reshape %v3311 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3322 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3323 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3324 = stablehlo.reduce(%v3320 init: %v3321) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3325 = stablehlo.broadcast_in_dim %v3324, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3326 = stablehlo.divide %v3325, %v3322 : tensor<32x784x192xf32>
    %v3327 = stablehlo.subtract %v3320, %v3326 : tensor<32x784x192xf32>
    %v3328 = stablehlo.multiply %v3327, %v3327 : tensor<32x784x192xf32>
    %v3329 = stablehlo.reduce(%v3328 init: %v3321) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3330 = stablehlo.broadcast_in_dim %v3329, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3331 = stablehlo.divide %v3330, %v3322 : tensor<32x784x192xf32>
    %v3332 = stablehlo.add %v3331, %v3323 : tensor<32x784x192xf32>
    %v3333 = stablehlo.rsqrt %v3332 : tensor<32x784x192xf32>
    %v3334 = stablehlo.multiply %v3327, %v3333 : tensor<32x784x192xf32>
    %v3335 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3336 = stablehlo.multiply %v3335, %v3319 : tensor<32x784x192xf32>
    %v3337 = stablehlo.reduce(%v3336 init: %v3321) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3339 = stablehlo.multiply %v3334, %v3336 : tensor<32x784x192xf32>
    %v3340 = stablehlo.reduce(%v3339 init: %v3321) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3341 = stablehlo.broadcast_in_dim %v3340, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3342 = stablehlo.multiply %v3336, %v3322 : tensor<32x784x192xf32>
    %v3343 = stablehlo.subtract %v3342, %v3338 : tensor<32x784x192xf32>
    %v3344 = stablehlo.multiply %v3334, %v3341 : tensor<32x784x192xf32>
    %v3345 = stablehlo.subtract %v3343, %v3344 : tensor<32x784x192xf32>
    %v3346 = stablehlo.divide %v3333, %v3322 : tensor<32x784x192xf32>
    %v3347 = stablehlo.multiply %v3346, %v3345 : tensor<32x784x192xf32>
    %v3348 = stablehlo.reshape %v3347 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3349 = stablehlo.reshape %v3348 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3350 = stablehlo.transpose %v3349, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3351 = stablehlo.reshape %v3350 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3352 = stablehlo.reshape %v3239 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3354 = stablehlo.reduce(%v3352 init: %v3353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<384xf32>
    %v3355 = stablehlo.reshape %v491 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3356 = stablehlo.transpose %v3355, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3357 = stablehlo.reshape %v3356 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3358 = stablehlo.reshape %v3308 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3359 = stablehlo.transpose %v3358, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3360 = stablehlo.reshape %v3359 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3361 = stablehlo.reshape %v3357 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3362 = stablehlo.reshape %v3360 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3364 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3365 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3366 = stablehlo.reduce(%v3361 init: %v3363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3367 = stablehlo.broadcast_in_dim %v3366, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3368 = stablehlo.divide %v3367, %v3364 : tensor<32x784x192xf32>
    %v3369 = stablehlo.subtract %v3361, %v3368 : tensor<32x784x192xf32>
    %v3370 = stablehlo.multiply %v3369, %v3369 : tensor<32x784x192xf32>
    %v3371 = stablehlo.reduce(%v3370 init: %v3363) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3372 = stablehlo.broadcast_in_dim %v3371, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3373 = stablehlo.divide %v3372, %v3364 : tensor<32x784x192xf32>
    %v3374 = stablehlo.add %v3373, %v3365 : tensor<32x784x192xf32>
    %v3375 = stablehlo.rsqrt %v3374 : tensor<32x784x192xf32>
    %v3376 = stablehlo.multiply %v3369, %v3375 : tensor<32x784x192xf32>
    %v3377 = stablehlo.multiply %v3362, %v3376 : tensor<32x784x192xf32>
    %v3378 = stablehlo.reduce(%v3377 init: %v3363) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3379 = stablehlo.reshape %v3308 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3380 = stablehlo.transpose %v3379, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3381 = stablehlo.reshape %v3380 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3382 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3383 = stablehlo.reshape %v3381 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3384 = stablehlo.reduce(%v3383 init: %v3382) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3385 = stablehlo.reshape %v525 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3386 = stablehlo.reshape %v3239 : (tensor<32x75264xf32>) -> tensor<32x384x14x14xf32>
    %v3387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3388 = stablehlo.pad %v3386, %v3387, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x384x14x14xf32>, tensor<f32>) -> tensor<32x384x27x27xf32>
    %v3389 = stablehlo.transpose %v3385, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3390 = stablehlo.transpose %v3388, dims = [1, 0, 2, 3] : (tensor<32x384x27x27xf32>) -> tensor<384x32x27x27xf32>
    %v3391 = stablehlo.convolution(%v3389, %v3390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<384x32x27x27xf32>) -> tensor<192x384x2x2xf32>
    %v3392 = stablehlo.transpose %v3391, dims = [1, 0, 2, 3] : (tensor<192x384x2x2xf32>) -> tensor<384x192x2x2xf32>
    %v3393 = stablehlo.broadcast_in_dim %dp5, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3394 = stablehlo.multiply %v3393, %v3351 : tensor<32x150528xf32>
    %v3395 = stablehlo.reshape %v3394 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3396 = stablehlo.broadcast_in_dim %s1b2lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3397 = stablehlo.multiply %v3395, %v3396 : tensor<32x192x28x28xf32>
    %v3398 = stablehlo.reshape %v3397 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3400 = stablehlo.reverse %s1b2pW, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3401 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3402 = stablehlo.convolution(%v3399, %v3401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3403 = stablehlo.reshape %v3402 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3404 = stablehlo.multiply %v466, %v466 : tensor<32x602112xf32>
    %v3405 = stablehlo.multiply %v3404, %v466 : tensor<32x602112xf32>
    %v3406 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3407 = stablehlo.multiply %v3406, %v3405 : tensor<32x602112xf32>
    %v3408 = stablehlo.add %v466, %v3407 : tensor<32x602112xf32>
    %v3409 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3410 = stablehlo.multiply %v3409, %v3408 : tensor<32x602112xf32>
    %v3411 = stablehlo.tanh %v3410 : tensor<32x602112xf32>
    %v3412 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3413 = stablehlo.add %v3412, %v3411 : tensor<32x602112xf32>
    %v3414 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3415 = stablehlo.multiply %v3414, %v3413 : tensor<32x602112xf32>
    %v3416 = stablehlo.multiply %v3411, %v3411 : tensor<32x602112xf32>
    %v3417 = stablehlo.subtract %v3412, %v3416 : tensor<32x602112xf32>
    %v3418 = stablehlo.multiply %v3414, %v466 : tensor<32x602112xf32>
    %v3419 = stablehlo.multiply %v3418, %v3417 : tensor<32x602112xf32>
    %v3420 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3421 = stablehlo.multiply %v3420, %v3404 : tensor<32x602112xf32>
    %v3422 = stablehlo.add %v3412, %v3421 : tensor<32x602112xf32>
    %v3423 = stablehlo.multiply %v3409, %v3422 : tensor<32x602112xf32>
    %v3424 = stablehlo.multiply %v3419, %v3423 : tensor<32x602112xf32>
    %v3425 = stablehlo.add %v3415, %v3424 : tensor<32x602112xf32>
    %v3426 = stablehlo.multiply %v3403, %v3425 : tensor<32x602112xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3428 = stablehlo.reverse %s1b2eW, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3429 = stablehlo.transpose %v3428, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3430 = stablehlo.convolution(%v3427, %v3429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3432 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3433 = stablehlo.transpose %v3432, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3434 = stablehlo.reshape %v3433 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3435 = stablehlo.reshape %v3431 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3436 = stablehlo.transpose %v3435, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3438 = stablehlo.reshape %v3437 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3439 = stablehlo.broadcast_in_dim %s1b2ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3440 = stablehlo.multiply %v3438, %v3439 : tensor<32x784x192xf32>
    %v3441 = stablehlo.reshape %v3440 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3442 = stablehlo.reshape %v3441 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3443 = stablehlo.reshape %v3434 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3445 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3446 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3447 = stablehlo.reduce(%v3443 init: %v3444) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3449 = stablehlo.divide %v3448, %v3445 : tensor<32x784x192xf32>
    %v3450 = stablehlo.subtract %v3443, %v3449 : tensor<32x784x192xf32>
    %v3451 = stablehlo.multiply %v3450, %v3450 : tensor<32x784x192xf32>
    %v3452 = stablehlo.reduce(%v3451 init: %v3444) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3454 = stablehlo.divide %v3453, %v3445 : tensor<32x784x192xf32>
    %v3455 = stablehlo.add %v3454, %v3446 : tensor<32x784x192xf32>
    %v3456 = stablehlo.rsqrt %v3455 : tensor<32x784x192xf32>
    %v3457 = stablehlo.multiply %v3450, %v3456 : tensor<32x784x192xf32>
    %v3458 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3459 = stablehlo.multiply %v3458, %v3442 : tensor<32x784x192xf32>
    %v3460 = stablehlo.reduce(%v3459 init: %v3444) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3461 = stablehlo.broadcast_in_dim %v3460, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3462 = stablehlo.multiply %v3457, %v3459 : tensor<32x784x192xf32>
    %v3463 = stablehlo.reduce(%v3462 init: %v3444) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3465 = stablehlo.multiply %v3459, %v3445 : tensor<32x784x192xf32>
    %v3466 = stablehlo.subtract %v3465, %v3461 : tensor<32x784x192xf32>
    %v3467 = stablehlo.multiply %v3457, %v3464 : tensor<32x784x192xf32>
    %v3468 = stablehlo.subtract %v3466, %v3467 : tensor<32x784x192xf32>
    %v3469 = stablehlo.divide %v3456, %v3445 : tensor<32x784x192xf32>
    %v3470 = stablehlo.multiply %v3469, %v3468 : tensor<32x784x192xf32>
    %v3471 = stablehlo.reshape %v3470 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3473 = stablehlo.transpose %v3472, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3474 = stablehlo.reshape %v3473 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3475 = stablehlo.reshape %v3474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3476 = stablehlo.reverse %s1b2dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3477 = stablehlo.convolution(%v3475, %v3476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3478 = stablehlo.reshape %v3477 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3479 = stablehlo.add %v3478, %v3351 : tensor<32x150528xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.reshape %v484 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3482 = stablehlo.reshape %v3394 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3483 = stablehlo.multiply %v3481, %v3482 : tensor<32x192x28x28xf32>
    %v3484 = stablehlo.reduce(%v3483 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3485 = stablehlo.reshape %v479 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3486 = stablehlo.reshape %v3398 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3487 = stablehlo.transpose %v3485, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3488 = stablehlo.transpose %v3486, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3489 = stablehlo.convolution(%v3487, %v3488)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3490 = stablehlo.transpose %v3489, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3491 = stablehlo.reshape %v3398 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3493 = stablehlo.reduce(%v3491 init: %v3492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3494 = stablehlo.reshape %v461 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3495 = stablehlo.reshape %v3426 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3496 = stablehlo.transpose %v3494, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3497 = stablehlo.transpose %v3495, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3498 = stablehlo.convolution(%v3496, %v3497)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3499 = stablehlo.transpose %v3498, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3500 = stablehlo.reshape %v3426 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3502 = stablehlo.reduce(%v3500 init: %v3501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3503 = stablehlo.reshape %v427 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3504 = stablehlo.transpose %v3503, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3505 = stablehlo.reshape %v3504 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3506 = stablehlo.reshape %v3431 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3507 = stablehlo.transpose %v3506, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3508 = stablehlo.reshape %v3507 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3509 = stablehlo.reshape %v3505 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3510 = stablehlo.reshape %v3508 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3512 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3513 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3514 = stablehlo.reduce(%v3509 init: %v3511) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3515 = stablehlo.broadcast_in_dim %v3514, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3516 = stablehlo.divide %v3515, %v3512 : tensor<32x784x192xf32>
    %v3517 = stablehlo.subtract %v3509, %v3516 : tensor<32x784x192xf32>
    %v3518 = stablehlo.multiply %v3517, %v3517 : tensor<32x784x192xf32>
    %v3519 = stablehlo.reduce(%v3518 init: %v3511) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3521 = stablehlo.divide %v3520, %v3512 : tensor<32x784x192xf32>
    %v3522 = stablehlo.add %v3521, %v3513 : tensor<32x784x192xf32>
    %v3523 = stablehlo.rsqrt %v3522 : tensor<32x784x192xf32>
    %v3524 = stablehlo.multiply %v3517, %v3523 : tensor<32x784x192xf32>
    %v3525 = stablehlo.multiply %v3510, %v3524 : tensor<32x784x192xf32>
    %v3526 = stablehlo.reduce(%v3525 init: %v3511) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3527 = stablehlo.reshape %v3431 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3528 = stablehlo.transpose %v3527, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3529 = stablehlo.reshape %v3528 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3531 = stablehlo.reshape %v3529 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3532 = stablehlo.reduce(%v3531 init: %v3530) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3533 = stablehlo.reshape %v422 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3534 = stablehlo.reshape %v3474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3535 = stablehlo.transpose %v3533, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3536 = stablehlo.transpose %v3534, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3537 = stablehlo.convolution(%v3535, %v3536)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3538 = stablehlo.reshape %v3537 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3539 = stablehlo.reshape %v3474 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3541 = stablehlo.reduce(%v3539 init: %v3540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3542 = stablehlo.broadcast_in_dim %dp4, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3543 = stablehlo.multiply %v3542, %v3479 : tensor<32x150528xf32>
    %v3544 = stablehlo.reshape %v3543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3545 = stablehlo.broadcast_in_dim %s1b1lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3546 = stablehlo.multiply %v3544, %v3545 : tensor<32x192x28x28xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3548 = stablehlo.reshape %v3547 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3549 = stablehlo.reverse %s1b1pW, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3550 = stablehlo.transpose %v3549, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3551 = stablehlo.convolution(%v3548, %v3550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3552 = stablehlo.reshape %v3551 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3553 = stablehlo.multiply %v397, %v397 : tensor<32x602112xf32>
    %v3554 = stablehlo.multiply %v3553, %v397 : tensor<32x602112xf32>
    %v3555 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3556 = stablehlo.multiply %v3555, %v3554 : tensor<32x602112xf32>
    %v3557 = stablehlo.add %v397, %v3556 : tensor<32x602112xf32>
    %v3558 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3559 = stablehlo.multiply %v3558, %v3557 : tensor<32x602112xf32>
    %v3560 = stablehlo.tanh %v3559 : tensor<32x602112xf32>
    %v3561 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3562 = stablehlo.add %v3561, %v3560 : tensor<32x602112xf32>
    %v3563 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3564 = stablehlo.multiply %v3563, %v3562 : tensor<32x602112xf32>
    %v3565 = stablehlo.multiply %v3560, %v3560 : tensor<32x602112xf32>
    %v3566 = stablehlo.subtract %v3561, %v3565 : tensor<32x602112xf32>
    %v3567 = stablehlo.multiply %v3563, %v397 : tensor<32x602112xf32>
    %v3568 = stablehlo.multiply %v3567, %v3566 : tensor<32x602112xf32>
    %v3569 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3570 = stablehlo.multiply %v3569, %v3553 : tensor<32x602112xf32>
    %v3571 = stablehlo.add %v3561, %v3570 : tensor<32x602112xf32>
    %v3572 = stablehlo.multiply %v3558, %v3571 : tensor<32x602112xf32>
    %v3573 = stablehlo.multiply %v3568, %v3572 : tensor<32x602112xf32>
    %v3574 = stablehlo.add %v3564, %v3573 : tensor<32x602112xf32>
    %v3575 = stablehlo.multiply %v3552, %v3574 : tensor<32x602112xf32>
    %v3576 = stablehlo.reshape %v3575 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3577 = stablehlo.reverse %s1b1eW, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3578 = stablehlo.transpose %v3577, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3579 = stablehlo.convolution(%v3576, %v3578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3580 = stablehlo.reshape %v3579 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3581 = stablehlo.reshape %v358 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3582 = stablehlo.transpose %v3581, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3583 = stablehlo.reshape %v3582 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3584 = stablehlo.reshape %v3580 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3585 = stablehlo.transpose %v3584, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3586 = stablehlo.reshape %v3585 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3588 = stablehlo.broadcast_in_dim %s1b1ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3589 = stablehlo.multiply %v3587, %v3588 : tensor<32x784x192xf32>
    %v3590 = stablehlo.reshape %v3589 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3591 = stablehlo.reshape %v3590 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3592 = stablehlo.reshape %v3583 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3594 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3595 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3596 = stablehlo.reduce(%v3592 init: %v3593) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3597 = stablehlo.broadcast_in_dim %v3596, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3598 = stablehlo.divide %v3597, %v3594 : tensor<32x784x192xf32>
    %v3599 = stablehlo.subtract %v3592, %v3598 : tensor<32x784x192xf32>
    %v3600 = stablehlo.multiply %v3599, %v3599 : tensor<32x784x192xf32>
    %v3601 = stablehlo.reduce(%v3600 init: %v3593) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3603 = stablehlo.divide %v3602, %v3594 : tensor<32x784x192xf32>
    %v3604 = stablehlo.add %v3603, %v3595 : tensor<32x784x192xf32>
    %v3605 = stablehlo.rsqrt %v3604 : tensor<32x784x192xf32>
    %v3606 = stablehlo.multiply %v3599, %v3605 : tensor<32x784x192xf32>
    %v3607 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3608 = stablehlo.multiply %v3607, %v3591 : tensor<32x784x192xf32>
    %v3609 = stablehlo.reduce(%v3608 init: %v3593) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3610 = stablehlo.broadcast_in_dim %v3609, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3611 = stablehlo.multiply %v3606, %v3608 : tensor<32x784x192xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3593) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3613 = stablehlo.broadcast_in_dim %v3612, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3614 = stablehlo.multiply %v3608, %v3594 : tensor<32x784x192xf32>
    %v3615 = stablehlo.subtract %v3614, %v3610 : tensor<32x784x192xf32>
    %v3616 = stablehlo.multiply %v3606, %v3613 : tensor<32x784x192xf32>
    %v3617 = stablehlo.subtract %v3615, %v3616 : tensor<32x784x192xf32>
    %v3618 = stablehlo.divide %v3605, %v3594 : tensor<32x784x192xf32>
    %v3619 = stablehlo.multiply %v3618, %v3617 : tensor<32x784x192xf32>
    %v3620 = stablehlo.reshape %v3619 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3621 = stablehlo.reshape %v3620 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3622 = stablehlo.transpose %v3621, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3623 = stablehlo.reshape %v3622 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3624 = stablehlo.reshape %v3623 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3625 = stablehlo.reverse %s1b1dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3626 = stablehlo.convolution(%v3624, %v3625)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3627 = stablehlo.reshape %v3626 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3628 = stablehlo.add %v3627, %v3479 : tensor<32x150528xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.reshape %v415 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3631 = stablehlo.reshape %v3543 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3632 = stablehlo.multiply %v3630, %v3631 : tensor<32x192x28x28xf32>
    %v3633 = stablehlo.reduce(%v3632 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3634 = stablehlo.reshape %v410 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3635 = stablehlo.reshape %v3547 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3636 = stablehlo.transpose %v3634, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3637 = stablehlo.transpose %v3635, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3638 = stablehlo.convolution(%v3636, %v3637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3639 = stablehlo.transpose %v3638, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3640 = stablehlo.reshape %v3547 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.reduce(%v3640 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3643 = stablehlo.reshape %v392 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3644 = stablehlo.reshape %v3575 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3645 = stablehlo.transpose %v3643, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3646 = stablehlo.transpose %v3644, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3647 = stablehlo.convolution(%v3645, %v3646)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3648 = stablehlo.transpose %v3647, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3649 = stablehlo.reshape %v3575 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3651 = stablehlo.reduce(%v3649 init: %v3650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3652 = stablehlo.reshape %v358 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3653 = stablehlo.transpose %v3652, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3655 = stablehlo.reshape %v3580 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3656 = stablehlo.transpose %v3655, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3657 = stablehlo.reshape %v3656 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3658 = stablehlo.reshape %v3654 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3659 = stablehlo.reshape %v3657 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3660 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3661 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3662 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3663 = stablehlo.reduce(%v3658 init: %v3660) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3664 = stablehlo.broadcast_in_dim %v3663, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3665 = stablehlo.divide %v3664, %v3661 : tensor<32x784x192xf32>
    %v3666 = stablehlo.subtract %v3658, %v3665 : tensor<32x784x192xf32>
    %v3667 = stablehlo.multiply %v3666, %v3666 : tensor<32x784x192xf32>
    %v3668 = stablehlo.reduce(%v3667 init: %v3660) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3670 = stablehlo.divide %v3669, %v3661 : tensor<32x784x192xf32>
    %v3671 = stablehlo.add %v3670, %v3662 : tensor<32x784x192xf32>
    %v3672 = stablehlo.rsqrt %v3671 : tensor<32x784x192xf32>
    %v3673 = stablehlo.multiply %v3666, %v3672 : tensor<32x784x192xf32>
    %v3674 = stablehlo.multiply %v3659, %v3673 : tensor<32x784x192xf32>
    %v3675 = stablehlo.reduce(%v3674 init: %v3660) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3676 = stablehlo.reshape %v3580 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3677 = stablehlo.transpose %v3676, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3678 = stablehlo.reshape %v3677 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3680 = stablehlo.reshape %v3678 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3681 = stablehlo.reduce(%v3680 init: %v3679) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3682 = stablehlo.reshape %v353 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3683 = stablehlo.reshape %v3623 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3684 = stablehlo.transpose %v3682, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3685 = stablehlo.transpose %v3683, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3686 = stablehlo.convolution(%v3684, %v3685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3687 = stablehlo.reshape %v3686 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3688 = stablehlo.reshape %v3623 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3690 = stablehlo.reduce(%v3688 init: %v3689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3691 = stablehlo.broadcast_in_dim %dp3, dims = [0] : (tensor<32xf32>) -> tensor<32x150528xf32>
    %v3692 = stablehlo.multiply %v3691, %v3628 : tensor<32x150528xf32>
    %v3693 = stablehlo.reshape %v3692 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3694 = stablehlo.broadcast_in_dim %s1b0lg, dims = [1] : (tensor<192xf32>) -> tensor<32x192x28x28xf32>
    %v3695 = stablehlo.multiply %v3693, %v3694 : tensor<32x192x28x28xf32>
    %v3696 = stablehlo.reshape %v3695 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3697 = stablehlo.reshape %v3696 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3698 = stablehlo.reverse %s1b0pW, dims = [2, 3] : tensor<192x768x1x1xf32>
    %v3699 = stablehlo.transpose %v3698, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3700 = stablehlo.convolution(%v3697, %v3699)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x28x28xf32>, tensor<768x192x1x1xf32>) -> tensor<32x768x28x28xf32>
    %v3701 = stablehlo.reshape %v3700 : (tensor<32x768x28x28xf32>) -> tensor<32x602112xf32>
    %v3702 = stablehlo.multiply %v328, %v328 : tensor<32x602112xf32>
    %v3703 = stablehlo.multiply %v3702, %v328 : tensor<32x602112xf32>
    %v3704 = stablehlo.constant dense<0.044715> : tensor<32x602112xf32>
    %v3705 = stablehlo.multiply %v3704, %v3703 : tensor<32x602112xf32>
    %v3706 = stablehlo.add %v328, %v3705 : tensor<32x602112xf32>
    %v3707 = stablehlo.constant dense<0.7978845608028654> : tensor<32x602112xf32>
    %v3708 = stablehlo.multiply %v3707, %v3706 : tensor<32x602112xf32>
    %v3709 = stablehlo.tanh %v3708 : tensor<32x602112xf32>
    %v3710 = stablehlo.constant dense<1.0> : tensor<32x602112xf32>
    %v3711 = stablehlo.add %v3710, %v3709 : tensor<32x602112xf32>
    %v3712 = stablehlo.constant dense<0.5> : tensor<32x602112xf32>
    %v3713 = stablehlo.multiply %v3712, %v3711 : tensor<32x602112xf32>
    %v3714 = stablehlo.multiply %v3709, %v3709 : tensor<32x602112xf32>
    %v3715 = stablehlo.subtract %v3710, %v3714 : tensor<32x602112xf32>
    %v3716 = stablehlo.multiply %v3712, %v328 : tensor<32x602112xf32>
    %v3717 = stablehlo.multiply %v3716, %v3715 : tensor<32x602112xf32>
    %v3718 = stablehlo.constant dense<0.134145> : tensor<32x602112xf32>
    %v3719 = stablehlo.multiply %v3718, %v3702 : tensor<32x602112xf32>
    %v3720 = stablehlo.add %v3710, %v3719 : tensor<32x602112xf32>
    %v3721 = stablehlo.multiply %v3707, %v3720 : tensor<32x602112xf32>
    %v3722 = stablehlo.multiply %v3717, %v3721 : tensor<32x602112xf32>
    %v3723 = stablehlo.add %v3713, %v3722 : tensor<32x602112xf32>
    %v3724 = stablehlo.multiply %v3701, %v3723 : tensor<32x602112xf32>
    %v3725 = stablehlo.reshape %v3724 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3726 = stablehlo.reverse %s1b0eW, dims = [2, 3] : tensor<768x192x1x1xf32>
    %v3727 = stablehlo.transpose %v3726, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3728 = stablehlo.convolution(%v3725, %v3727)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x768x28x28xf32>, tensor<192x768x1x1xf32>) -> tensor<32x192x28x28xf32>
    %v3729 = stablehlo.reshape %v3728 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3730 = stablehlo.reshape %v289 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3731 = stablehlo.transpose %v3730, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3732 = stablehlo.reshape %v3731 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3733 = stablehlo.reshape %v3729 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3734 = stablehlo.transpose %v3733, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3735 = stablehlo.reshape %v3734 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3736 = stablehlo.reshape %v3735 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3737 = stablehlo.broadcast_in_dim %s1b0ng, dims = [2] : (tensor<192xf32>) -> tensor<32x784x192xf32>
    %v3738 = stablehlo.multiply %v3736, %v3737 : tensor<32x784x192xf32>
    %v3739 = stablehlo.reshape %v3738 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3740 = stablehlo.reshape %v3739 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3741 = stablehlo.reshape %v3732 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3743 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3744 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3745 = stablehlo.reduce(%v3741 init: %v3742) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3746 = stablehlo.broadcast_in_dim %v3745, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3747 = stablehlo.divide %v3746, %v3743 : tensor<32x784x192xf32>
    %v3748 = stablehlo.subtract %v3741, %v3747 : tensor<32x784x192xf32>
    %v3749 = stablehlo.multiply %v3748, %v3748 : tensor<32x784x192xf32>
    %v3750 = stablehlo.reduce(%v3749 init: %v3742) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3751 = stablehlo.broadcast_in_dim %v3750, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3752 = stablehlo.divide %v3751, %v3743 : tensor<32x784x192xf32>
    %v3753 = stablehlo.add %v3752, %v3744 : tensor<32x784x192xf32>
    %v3754 = stablehlo.rsqrt %v3753 : tensor<32x784x192xf32>
    %v3755 = stablehlo.multiply %v3748, %v3754 : tensor<32x784x192xf32>
    %v3756 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x784x192xf32>
    %v3757 = stablehlo.multiply %v3756, %v3740 : tensor<32x784x192xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3742) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3760 = stablehlo.multiply %v3755, %v3757 : tensor<32x784x192xf32>
    %v3761 = stablehlo.reduce(%v3760 init: %v3742) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3763 = stablehlo.multiply %v3757, %v3743 : tensor<32x784x192xf32>
    %v3764 = stablehlo.subtract %v3763, %v3759 : tensor<32x784x192xf32>
    %v3765 = stablehlo.multiply %v3755, %v3762 : tensor<32x784x192xf32>
    %v3766 = stablehlo.subtract %v3764, %v3765 : tensor<32x784x192xf32>
    %v3767 = stablehlo.divide %v3754, %v3743 : tensor<32x784x192xf32>
    %v3768 = stablehlo.multiply %v3767, %v3766 : tensor<32x784x192xf32>
    %v3769 = stablehlo.reshape %v3768 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3770 = stablehlo.reshape %v3769 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3771 = stablehlo.transpose %v3770, dims = [0, 2, 1] : (tensor<32x784x192xf32>) -> tensor<32x192x784xf32>
    %v3772 = stablehlo.reshape %v3771 : (tensor<32x192x784xf32>) -> tensor<32x150528xf32>
    %v3773 = stablehlo.reshape %v3772 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3774 = stablehlo.reverse %s1b0dW, dims = [2, 3] : tensor<192x1x7x7xf32>
    %v3775 = stablehlo.convolution(%v3773, %v3774)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 192 : i64} : (tensor<32x192x28x28xf32>, tensor<192x1x7x7xf32>) -> tensor<32x192x28x28xf32>
    %v3776 = stablehlo.reshape %v3775 : (tensor<32x192x28x28xf32>) -> tensor<32x150528xf32>
    %v3777 = stablehlo.add %v3776, %v3628 : tensor<32x150528xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.reshape %v346 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3780 = stablehlo.reshape %v3692 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3781 = stablehlo.multiply %v3779, %v3780 : tensor<32x192x28x28xf32>
    %v3782 = stablehlo.reduce(%v3781 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3783 = stablehlo.reshape %v341 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3784 = stablehlo.reshape %v3696 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3785 = stablehlo.transpose %v3783, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3786 = stablehlo.transpose %v3784, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3787 = stablehlo.convolution(%v3785, %v3786)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<768x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<768x192x1x1xf32>
    %v3788 = stablehlo.transpose %v3787, dims = [1, 0, 2, 3] : (tensor<768x192x1x1xf32>) -> tensor<192x768x1x1xf32>
    %v3789 = stablehlo.reshape %v3696 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3791 = stablehlo.reduce(%v3789 init: %v3790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3792 = stablehlo.reshape %v323 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3793 = stablehlo.reshape %v3724 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3794 = stablehlo.transpose %v3792, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3795 = stablehlo.transpose %v3793, dims = [1, 0, 2, 3] : (tensor<32x768x28x28xf32>) -> tensor<768x32x28x28xf32>
    %v3796 = stablehlo.convolution(%v3794, %v3795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<768x32x28x28xf32>) -> tensor<192x768x1x1xf32>
    %v3797 = stablehlo.transpose %v3796, dims = [1, 0, 2, 3] : (tensor<192x768x1x1xf32>) -> tensor<768x192x1x1xf32>
    %v3798 = stablehlo.reshape %v3724 : (tensor<32x602112xf32>) -> tensor<32x768x28x28xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x768x28x28xf32>, tensor<f32>) -> tensor<768xf32>
    %v3801 = stablehlo.reshape %v289 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3802 = stablehlo.transpose %v3801, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3803 = stablehlo.reshape %v3802 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3804 = stablehlo.reshape %v3729 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3805 = stablehlo.transpose %v3804, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3806 = stablehlo.reshape %v3805 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3807 = stablehlo.reshape %v3803 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3808 = stablehlo.reshape %v3806 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3810 = stablehlo.constant dense<192.0> : tensor<32x784x192xf32>
    %v3811 = stablehlo.constant dense<1.0e-6> : tensor<32x784x192xf32>
    %v3812 = stablehlo.reduce(%v3807 init: %v3809) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3814 = stablehlo.divide %v3813, %v3810 : tensor<32x784x192xf32>
    %v3815 = stablehlo.subtract %v3807, %v3814 : tensor<32x784x192xf32>
    %v3816 = stablehlo.multiply %v3815, %v3815 : tensor<32x784x192xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3809) applies stablehlo.add across dimensions = [2] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<32x784xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [0, 1] : (tensor<32x784xf32>) -> tensor<32x784x192xf32>
    %v3819 = stablehlo.divide %v3818, %v3810 : tensor<32x784x192xf32>
    %v3820 = stablehlo.add %v3819, %v3811 : tensor<32x784x192xf32>
    %v3821 = stablehlo.rsqrt %v3820 : tensor<32x784x192xf32>
    %v3822 = stablehlo.multiply %v3815, %v3821 : tensor<32x784x192xf32>
    %v3823 = stablehlo.multiply %v3808, %v3822 : tensor<32x784x192xf32>
    %v3824 = stablehlo.reduce(%v3823 init: %v3809) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3825 = stablehlo.reshape %v3729 : (tensor<32x150528xf32>) -> tensor<32x192x784xf32>
    %v3826 = stablehlo.transpose %v3825, dims = [0, 2, 1] : (tensor<32x192x784xf32>) -> tensor<32x784x192xf32>
    %v3827 = stablehlo.reshape %v3826 : (tensor<32x784x192xf32>) -> tensor<32x150528xf32>
    %v3828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3829 = stablehlo.reshape %v3827 : (tensor<32x150528xf32>) -> tensor<32x784x192xf32>
    %v3830 = stablehlo.reduce(%v3829 init: %v3828) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x784x192xf32>, tensor<f32>) -> tensor<192xf32>
    %v3831 = stablehlo.reshape %v284 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3832 = stablehlo.reshape %v3772 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3833 = stablehlo.transpose %v3831, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3834 = stablehlo.transpose %v3832, dims = [1, 0, 2, 3] : (tensor<32x192x28x28xf32>) -> tensor<192x32x28x28xf32>
    %v3835 = stablehlo.convolution(%v3833, %v3834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 192 : i64, feature_group_count = 1 : i64} : (tensor<192x32x28x28xf32>, tensor<192x32x28x28xf32>) -> tensor<1x192x7x7xf32>
    %v3836 = stablehlo.reshape %v3835 : (tensor<1x192x7x7xf32>) -> tensor<192x1x7x7xf32>
    %v3837 = stablehlo.reshape %v3772 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3839 = stablehlo.reduce(%v3837 init: %v3838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3840 = stablehlo.reshape %v3777 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3842 = stablehlo.pad %v3840, %v3841, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x56x56xf32>
    %v3843 = stablehlo.reverse %d0W, dims = [2, 3] : tensor<192x96x2x2xf32>
    %v3844 = stablehlo.transpose %v3843, dims = [1, 0, 2, 3] : (tensor<192x96x2x2xf32>) -> tensor<96x192x2x2xf32>
    %v3845 = stablehlo.convolution(%v3842, %v3844)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x192x56x56xf32>, tensor<96x192x2x2xf32>) -> tensor<32x96x56x56xf32>
    %v3846 = stablehlo.reshape %v3845 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3847 = stablehlo.reshape %v245 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3848 = stablehlo.transpose %v3847, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3849 = stablehlo.reshape %v3848 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3850 = stablehlo.reshape %v3846 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3851 = stablehlo.transpose %v3850, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3852 = stablehlo.reshape %v3851 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3853 = stablehlo.reshape %v3852 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3854 = stablehlo.broadcast_in_dim %d0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v3855 = stablehlo.multiply %v3853, %v3854 : tensor<32x3136x96xf32>
    %v3856 = stablehlo.reshape %v3855 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3857 = stablehlo.reshape %v3856 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3858 = stablehlo.reshape %v3849 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3859 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3860 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v3861 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v3862 = stablehlo.reduce(%v3858 init: %v3859) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3863 = stablehlo.broadcast_in_dim %v3862, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3864 = stablehlo.divide %v3863, %v3860 : tensor<32x3136x96xf32>
    %v3865 = stablehlo.subtract %v3858, %v3864 : tensor<32x3136x96xf32>
    %v3866 = stablehlo.multiply %v3865, %v3865 : tensor<32x3136x96xf32>
    %v3867 = stablehlo.reduce(%v3866 init: %v3859) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3868 = stablehlo.broadcast_in_dim %v3867, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3869 = stablehlo.divide %v3868, %v3860 : tensor<32x3136x96xf32>
    %v3870 = stablehlo.add %v3869, %v3861 : tensor<32x3136x96xf32>
    %v3871 = stablehlo.rsqrt %v3870 : tensor<32x3136x96xf32>
    %v3872 = stablehlo.multiply %v3865, %v3871 : tensor<32x3136x96xf32>
    %v3873 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v3874 = stablehlo.multiply %v3873, %v3857 : tensor<32x3136x96xf32>
    %v3875 = stablehlo.reduce(%v3874 init: %v3859) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3876 = stablehlo.broadcast_in_dim %v3875, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3877 = stablehlo.multiply %v3872, %v3874 : tensor<32x3136x96xf32>
    %v3878 = stablehlo.reduce(%v3877 init: %v3859) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3879 = stablehlo.broadcast_in_dim %v3878, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3880 = stablehlo.multiply %v3874, %v3860 : tensor<32x3136x96xf32>
    %v3881 = stablehlo.subtract %v3880, %v3876 : tensor<32x3136x96xf32>
    %v3882 = stablehlo.multiply %v3872, %v3879 : tensor<32x3136x96xf32>
    %v3883 = stablehlo.subtract %v3881, %v3882 : tensor<32x3136x96xf32>
    %v3884 = stablehlo.divide %v3871, %v3860 : tensor<32x3136x96xf32>
    %v3885 = stablehlo.multiply %v3884, %v3883 : tensor<32x3136x96xf32>
    %v3886 = stablehlo.reshape %v3885 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3887 = stablehlo.reshape %v3886 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3888 = stablehlo.transpose %v3887, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v3890 = stablehlo.reshape %v3777 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3892 = stablehlo.reduce(%v3890 init: %v3891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<192xf32>
    %v3893 = stablehlo.reshape %v245 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3894 = stablehlo.transpose %v3893, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3895 = stablehlo.reshape %v3894 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3896 = stablehlo.reshape %v3846 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3897 = stablehlo.transpose %v3896, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3898 = stablehlo.reshape %v3897 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3899 = stablehlo.reshape %v3895 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3900 = stablehlo.reshape %v3898 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3902 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v3903 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v3904 = stablehlo.reduce(%v3899 init: %v3901) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3905 = stablehlo.broadcast_in_dim %v3904, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3906 = stablehlo.divide %v3905, %v3902 : tensor<32x3136x96xf32>
    %v3907 = stablehlo.subtract %v3899, %v3906 : tensor<32x3136x96xf32>
    %v3908 = stablehlo.multiply %v3907, %v3907 : tensor<32x3136x96xf32>
    %v3909 = stablehlo.reduce(%v3908 init: %v3901) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3909, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3911 = stablehlo.divide %v3910, %v3902 : tensor<32x3136x96xf32>
    %v3912 = stablehlo.add %v3911, %v3903 : tensor<32x3136x96xf32>
    %v3913 = stablehlo.rsqrt %v3912 : tensor<32x3136x96xf32>
    %v3914 = stablehlo.multiply %v3907, %v3913 : tensor<32x3136x96xf32>
    %v3915 = stablehlo.multiply %v3900, %v3914 : tensor<32x3136x96xf32>
    %v3916 = stablehlo.reduce(%v3915 init: %v3901) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v3917 = stablehlo.reshape %v3846 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3918 = stablehlo.transpose %v3917, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3919 = stablehlo.reshape %v3918 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3921 = stablehlo.reshape %v3919 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3920) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v3923 = stablehlo.reshape %v279 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3924 = stablehlo.reshape %v3777 : (tensor<32x150528xf32>) -> tensor<32x192x28x28xf32>
    %v3925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3926 = stablehlo.pad %v3924, %v3925, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 1, 1] : (tensor<32x192x28x28xf32>, tensor<f32>) -> tensor<32x192x55x55xf32>
    %v3927 = stablehlo.transpose %v3923, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v3928 = stablehlo.transpose %v3926, dims = [1, 0, 2, 3] : (tensor<32x192x55x55xf32>) -> tensor<192x32x55x55xf32>
    %v3929 = stablehlo.convolution(%v3927, %v3928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<192x32x55x55xf32>) -> tensor<96x192x2x2xf32>
    %v3930 = stablehlo.transpose %v3929, dims = [1, 0, 2, 3] : (tensor<96x192x2x2xf32>) -> tensor<192x96x2x2xf32>
    %v3931 = stablehlo.broadcast_in_dim %dp2, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v3932 = stablehlo.multiply %v3931, %v3889 : tensor<32x301056xf32>
    %v3933 = stablehlo.reshape %v3932 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3934 = stablehlo.broadcast_in_dim %s0b2lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v3935 = stablehlo.multiply %v3933, %v3934 : tensor<32x96x56x56xf32>
    %v3936 = stablehlo.reshape %v3935 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3937 = stablehlo.reshape %v3936 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v3938 = stablehlo.reverse %s0b2pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v3939 = stablehlo.transpose %v3938, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v3940 = stablehlo.convolution(%v3937, %v3939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v3941 = stablehlo.reshape %v3940 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v3942 = stablehlo.multiply %v220, %v220 : tensor<32x1204224xf32>
    %v3943 = stablehlo.multiply %v3942, %v220 : tensor<32x1204224xf32>
    %v3944 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v3945 = stablehlo.multiply %v3944, %v3943 : tensor<32x1204224xf32>
    %v3946 = stablehlo.add %v220, %v3945 : tensor<32x1204224xf32>
    %v3947 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v3948 = stablehlo.multiply %v3947, %v3946 : tensor<32x1204224xf32>
    %v3949 = stablehlo.tanh %v3948 : tensor<32x1204224xf32>
    %v3950 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v3951 = stablehlo.add %v3950, %v3949 : tensor<32x1204224xf32>
    %v3952 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v3953 = stablehlo.multiply %v3952, %v3951 : tensor<32x1204224xf32>
    %v3954 = stablehlo.multiply %v3949, %v3949 : tensor<32x1204224xf32>
    %v3955 = stablehlo.subtract %v3950, %v3954 : tensor<32x1204224xf32>
    %v3956 = stablehlo.multiply %v3952, %v220 : tensor<32x1204224xf32>
    %v3957 = stablehlo.multiply %v3956, %v3955 : tensor<32x1204224xf32>
    %v3958 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v3959 = stablehlo.multiply %v3958, %v3942 : tensor<32x1204224xf32>
    %v3960 = stablehlo.add %v3950, %v3959 : tensor<32x1204224xf32>
    %v3961 = stablehlo.multiply %v3947, %v3960 : tensor<32x1204224xf32>
    %v3962 = stablehlo.multiply %v3957, %v3961 : tensor<32x1204224xf32>
    %v3963 = stablehlo.add %v3953, %v3962 : tensor<32x1204224xf32>
    %v3964 = stablehlo.multiply %v3941, %v3963 : tensor<32x1204224xf32>
    %v3965 = stablehlo.reshape %v3964 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v3966 = stablehlo.reverse %s0b2eW, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v3967 = stablehlo.transpose %v3966, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v3968 = stablehlo.convolution(%v3965, %v3967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v3969 = stablehlo.reshape %v3968 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v3970 = stablehlo.reshape %v181 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3971 = stablehlo.transpose %v3970, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3972 = stablehlo.reshape %v3971 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3973 = stablehlo.reshape %v3969 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v3974 = stablehlo.transpose %v3973, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v3975 = stablehlo.reshape %v3974 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3976 = stablehlo.reshape %v3975 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3977 = stablehlo.broadcast_in_dim %s0b2ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v3978 = stablehlo.multiply %v3976, %v3977 : tensor<32x3136x96xf32>
    %v3979 = stablehlo.reshape %v3978 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v3980 = stablehlo.reshape %v3979 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3981 = stablehlo.reshape %v3972 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v3982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3983 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v3984 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v3985 = stablehlo.reduce(%v3981 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3986 = stablehlo.broadcast_in_dim %v3985, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3987 = stablehlo.divide %v3986, %v3983 : tensor<32x3136x96xf32>
    %v3988 = stablehlo.subtract %v3981, %v3987 : tensor<32x3136x96xf32>
    %v3989 = stablehlo.multiply %v3988, %v3988 : tensor<32x3136x96xf32>
    %v3990 = stablehlo.reduce(%v3989 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3991 = stablehlo.broadcast_in_dim %v3990, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v3992 = stablehlo.divide %v3991, %v3983 : tensor<32x3136x96xf32>
    %v3993 = stablehlo.add %v3992, %v3984 : tensor<32x3136x96xf32>
    %v3994 = stablehlo.rsqrt %v3993 : tensor<32x3136x96xf32>
    %v3995 = stablehlo.multiply %v3988, %v3994 : tensor<32x3136x96xf32>
    %v3996 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v3997 = stablehlo.multiply %v3996, %v3980 : tensor<32x3136x96xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3998, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4000 = stablehlo.multiply %v3995, %v3997 : tensor<32x3136x96xf32>
    %v4001 = stablehlo.reduce(%v4000 init: %v3982) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4002 = stablehlo.broadcast_in_dim %v4001, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4003 = stablehlo.multiply %v3997, %v3983 : tensor<32x3136x96xf32>
    %v4004 = stablehlo.subtract %v4003, %v3999 : tensor<32x3136x96xf32>
    %v4005 = stablehlo.multiply %v3995, %v4002 : tensor<32x3136x96xf32>
    %v4006 = stablehlo.subtract %v4004, %v4005 : tensor<32x3136x96xf32>
    %v4007 = stablehlo.divide %v3994, %v3983 : tensor<32x3136x96xf32>
    %v4008 = stablehlo.multiply %v4007, %v4006 : tensor<32x3136x96xf32>
    %v4009 = stablehlo.reshape %v4008 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4010 = stablehlo.reshape %v4009 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4011 = stablehlo.transpose %v4010, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4012 = stablehlo.reshape %v4011 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4013 = stablehlo.reshape %v4012 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4014 = stablehlo.reverse %s0b2dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4015 = stablehlo.convolution(%v4013, %v4014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4016 = stablehlo.reshape %v4015 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4017 = stablehlo.add %v4016, %v3889 : tensor<32x301056xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.reshape %v238 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4020 = stablehlo.reshape %v3932 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4021 = stablehlo.multiply %v4019, %v4020 : tensor<32x96x56x56xf32>
    %v4022 = stablehlo.reduce(%v4021 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4023 = stablehlo.reshape %v233 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4024 = stablehlo.reshape %v3936 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4025 = stablehlo.transpose %v4023, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4026 = stablehlo.transpose %v4024, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4027 = stablehlo.convolution(%v4025, %v4026)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4028 = stablehlo.transpose %v4027, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4029 = stablehlo.reshape %v3936 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4031 = stablehlo.reduce(%v4029 init: %v4030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4032 = stablehlo.reshape %v215 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4033 = stablehlo.reshape %v3964 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4034 = stablehlo.transpose %v4032, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4035 = stablehlo.transpose %v4033, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4036 = stablehlo.convolution(%v4034, %v4035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4037 = stablehlo.transpose %v4036, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4038 = stablehlo.reshape %v3964 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4041 = stablehlo.reshape %v181 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4042 = stablehlo.transpose %v4041, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4043 = stablehlo.reshape %v4042 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4044 = stablehlo.reshape %v3969 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4045 = stablehlo.transpose %v4044, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4047 = stablehlo.reshape %v4043 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4048 = stablehlo.reshape %v4046 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4049 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4050 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4051 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4052 = stablehlo.reduce(%v4047 init: %v4049) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4052, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4054 = stablehlo.divide %v4053, %v4050 : tensor<32x3136x96xf32>
    %v4055 = stablehlo.subtract %v4047, %v4054 : tensor<32x3136x96xf32>
    %v4056 = stablehlo.multiply %v4055, %v4055 : tensor<32x3136x96xf32>
    %v4057 = stablehlo.reduce(%v4056 init: %v4049) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4058 = stablehlo.broadcast_in_dim %v4057, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4059 = stablehlo.divide %v4058, %v4050 : tensor<32x3136x96xf32>
    %v4060 = stablehlo.add %v4059, %v4051 : tensor<32x3136x96xf32>
    %v4061 = stablehlo.rsqrt %v4060 : tensor<32x3136x96xf32>
    %v4062 = stablehlo.multiply %v4055, %v4061 : tensor<32x3136x96xf32>
    %v4063 = stablehlo.multiply %v4048, %v4062 : tensor<32x3136x96xf32>
    %v4064 = stablehlo.reduce(%v4063 init: %v4049) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4065 = stablehlo.reshape %v3969 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4066 = stablehlo.transpose %v4065, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4067 = stablehlo.reshape %v4066 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4069 = stablehlo.reshape %v4067 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4070 = stablehlo.reduce(%v4069 init: %v4068) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4071 = stablehlo.reshape %v176 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4072 = stablehlo.reshape %v4012 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4073 = stablehlo.transpose %v4071, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4074 = stablehlo.transpose %v4072, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4075 = stablehlo.convolution(%v4073, %v4074)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4076 = stablehlo.reshape %v4075 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4077 = stablehlo.reshape %v4012 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4079 = stablehlo.reduce(%v4077 init: %v4078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4080 = stablehlo.broadcast_in_dim %dp1, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v4081 = stablehlo.multiply %v4080, %v4017 : tensor<32x301056xf32>
    %v4082 = stablehlo.reshape %v4081 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4083 = stablehlo.broadcast_in_dim %s0b1lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4084 = stablehlo.multiply %v4082, %v4083 : tensor<32x96x56x56xf32>
    %v4085 = stablehlo.reshape %v4084 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4086 = stablehlo.reshape %v4085 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4087 = stablehlo.reverse %s0b1pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4088 = stablehlo.transpose %v4087, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4089 = stablehlo.convolution(%v4086, %v4088)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4090 = stablehlo.reshape %v4089 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4091 = stablehlo.multiply %v151, %v151 : tensor<32x1204224xf32>
    %v4092 = stablehlo.multiply %v4091, %v151 : tensor<32x1204224xf32>
    %v4093 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v4094 = stablehlo.multiply %v4093, %v4092 : tensor<32x1204224xf32>
    %v4095 = stablehlo.add %v151, %v4094 : tensor<32x1204224xf32>
    %v4096 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v4097 = stablehlo.multiply %v4096, %v4095 : tensor<32x1204224xf32>
    %v4098 = stablehlo.tanh %v4097 : tensor<32x1204224xf32>
    %v4099 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v4100 = stablehlo.add %v4099, %v4098 : tensor<32x1204224xf32>
    %v4101 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v4102 = stablehlo.multiply %v4101, %v4100 : tensor<32x1204224xf32>
    %v4103 = stablehlo.multiply %v4098, %v4098 : tensor<32x1204224xf32>
    %v4104 = stablehlo.subtract %v4099, %v4103 : tensor<32x1204224xf32>
    %v4105 = stablehlo.multiply %v4101, %v151 : tensor<32x1204224xf32>
    %v4106 = stablehlo.multiply %v4105, %v4104 : tensor<32x1204224xf32>
    %v4107 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v4108 = stablehlo.multiply %v4107, %v4091 : tensor<32x1204224xf32>
    %v4109 = stablehlo.add %v4099, %v4108 : tensor<32x1204224xf32>
    %v4110 = stablehlo.multiply %v4096, %v4109 : tensor<32x1204224xf32>
    %v4111 = stablehlo.multiply %v4106, %v4110 : tensor<32x1204224xf32>
    %v4112 = stablehlo.add %v4102, %v4111 : tensor<32x1204224xf32>
    %v4113 = stablehlo.multiply %v4090, %v4112 : tensor<32x1204224xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4115 = stablehlo.reverse %s0b1eW, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4116 = stablehlo.transpose %v4115, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4117 = stablehlo.convolution(%v4114, %v4116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4118 = stablehlo.reshape %v4117 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4119 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4120 = stablehlo.transpose %v4119, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4121 = stablehlo.reshape %v4120 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4122 = stablehlo.reshape %v4118 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4123 = stablehlo.transpose %v4122, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4124 = stablehlo.reshape %v4123 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4125 = stablehlo.reshape %v4124 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4126 = stablehlo.broadcast_in_dim %s0b1ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4127 = stablehlo.multiply %v4125, %v4126 : tensor<32x3136x96xf32>
    %v4128 = stablehlo.reshape %v4127 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4129 = stablehlo.reshape %v4128 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4130 = stablehlo.reshape %v4121 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4133 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4134 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4135 = stablehlo.broadcast_in_dim %v4134, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4136 = stablehlo.divide %v4135, %v4132 : tensor<32x3136x96xf32>
    %v4137 = stablehlo.subtract %v4130, %v4136 : tensor<32x3136x96xf32>
    %v4138 = stablehlo.multiply %v4137, %v4137 : tensor<32x3136x96xf32>
    %v4139 = stablehlo.reduce(%v4138 init: %v4131) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4140 = stablehlo.broadcast_in_dim %v4139, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4141 = stablehlo.divide %v4140, %v4132 : tensor<32x3136x96xf32>
    %v4142 = stablehlo.add %v4141, %v4133 : tensor<32x3136x96xf32>
    %v4143 = stablehlo.rsqrt %v4142 : tensor<32x3136x96xf32>
    %v4144 = stablehlo.multiply %v4137, %v4143 : tensor<32x3136x96xf32>
    %v4145 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4146 = stablehlo.multiply %v4145, %v4129 : tensor<32x3136x96xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4131) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4149 = stablehlo.multiply %v4144, %v4146 : tensor<32x3136x96xf32>
    %v4150 = stablehlo.reduce(%v4149 init: %v4131) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4151 = stablehlo.broadcast_in_dim %v4150, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4152 = stablehlo.multiply %v4146, %v4132 : tensor<32x3136x96xf32>
    %v4153 = stablehlo.subtract %v4152, %v4148 : tensor<32x3136x96xf32>
    %v4154 = stablehlo.multiply %v4144, %v4151 : tensor<32x3136x96xf32>
    %v4155 = stablehlo.subtract %v4153, %v4154 : tensor<32x3136x96xf32>
    %v4156 = stablehlo.divide %v4143, %v4132 : tensor<32x3136x96xf32>
    %v4157 = stablehlo.multiply %v4156, %v4155 : tensor<32x3136x96xf32>
    %v4158 = stablehlo.reshape %v4157 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4159 = stablehlo.reshape %v4158 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4160 = stablehlo.transpose %v4159, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4161 = stablehlo.reshape %v4160 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4162 = stablehlo.reshape %v4161 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4163 = stablehlo.reverse %s0b1dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4164 = stablehlo.convolution(%v4162, %v4163)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4165 = stablehlo.reshape %v4164 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4166 = stablehlo.add %v4165, %v4017 : tensor<32x301056xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.reshape %v169 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4169 = stablehlo.reshape %v4081 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4170 = stablehlo.multiply %v4168, %v4169 : tensor<32x96x56x56xf32>
    %v4171 = stablehlo.reduce(%v4170 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4172 = stablehlo.reshape %v164 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4173 = stablehlo.reshape %v4085 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4174 = stablehlo.transpose %v4172, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4175 = stablehlo.transpose %v4173, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4176 = stablehlo.convolution(%v4174, %v4175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4177 = stablehlo.transpose %v4176, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4178 = stablehlo.reshape %v4085 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4179 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4180 = stablehlo.reduce(%v4178 init: %v4179) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4181 = stablehlo.reshape %v146 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4182 = stablehlo.reshape %v4113 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4183 = stablehlo.transpose %v4181, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4184 = stablehlo.transpose %v4182, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4185 = stablehlo.convolution(%v4183, %v4184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4186 = stablehlo.transpose %v4185, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4187 = stablehlo.reshape %v4113 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4189 = stablehlo.reduce(%v4187 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4190 = stablehlo.reshape %v112 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4191 = stablehlo.transpose %v4190, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4192 = stablehlo.reshape %v4191 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4193 = stablehlo.reshape %v4118 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4194 = stablehlo.transpose %v4193, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4195 = stablehlo.reshape %v4194 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4196 = stablehlo.reshape %v4192 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4197 = stablehlo.reshape %v4195 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4199 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4200 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4201 = stablehlo.reduce(%v4196 init: %v4198) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4203 = stablehlo.divide %v4202, %v4199 : tensor<32x3136x96xf32>
    %v4204 = stablehlo.subtract %v4196, %v4203 : tensor<32x3136x96xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<32x3136x96xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4198) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4208 = stablehlo.divide %v4207, %v4199 : tensor<32x3136x96xf32>
    %v4209 = stablehlo.add %v4208, %v4200 : tensor<32x3136x96xf32>
    %v4210 = stablehlo.rsqrt %v4209 : tensor<32x3136x96xf32>
    %v4211 = stablehlo.multiply %v4204, %v4210 : tensor<32x3136x96xf32>
    %v4212 = stablehlo.multiply %v4197, %v4211 : tensor<32x3136x96xf32>
    %v4213 = stablehlo.reduce(%v4212 init: %v4198) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4214 = stablehlo.reshape %v4118 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4215 = stablehlo.transpose %v4214, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4216 = stablehlo.reshape %v4215 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4218 = stablehlo.reshape %v4216 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4219 = stablehlo.reduce(%v4218 init: %v4217) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4220 = stablehlo.reshape %v107 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4221 = stablehlo.reshape %v4161 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4222 = stablehlo.transpose %v4220, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4223 = stablehlo.transpose %v4221, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4224 = stablehlo.convolution(%v4222, %v4223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4225 = stablehlo.reshape %v4224 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4226 = stablehlo.reshape %v4161 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4228 = stablehlo.reduce(%v4226 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4229 = stablehlo.broadcast_in_dim %dp0, dims = [0] : (tensor<32xf32>) -> tensor<32x301056xf32>
    %v4230 = stablehlo.multiply %v4229, %v4166 : tensor<32x301056xf32>
    %v4231 = stablehlo.reshape %v4230 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4232 = stablehlo.broadcast_in_dim %s0b0lg, dims = [1] : (tensor<96xf32>) -> tensor<32x96x56x56xf32>
    %v4233 = stablehlo.multiply %v4231, %v4232 : tensor<32x96x56x56xf32>
    %v4234 = stablehlo.reshape %v4233 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4235 = stablehlo.reshape %v4234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4236 = stablehlo.reverse %s0b0pW, dims = [2, 3] : tensor<96x384x1x1xf32>
    %v4237 = stablehlo.transpose %v4236, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4238 = stablehlo.convolution(%v4235, %v4237)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x96x56x56xf32>, tensor<384x96x1x1xf32>) -> tensor<32x384x56x56xf32>
    %v4239 = stablehlo.reshape %v4238 : (tensor<32x384x56x56xf32>) -> tensor<32x1204224xf32>
    %v4240 = stablehlo.multiply %v82, %v82 : tensor<32x1204224xf32>
    %v4241 = stablehlo.multiply %v4240, %v82 : tensor<32x1204224xf32>
    %v4242 = stablehlo.constant dense<0.044715> : tensor<32x1204224xf32>
    %v4243 = stablehlo.multiply %v4242, %v4241 : tensor<32x1204224xf32>
    %v4244 = stablehlo.add %v82, %v4243 : tensor<32x1204224xf32>
    %v4245 = stablehlo.constant dense<0.7978845608028654> : tensor<32x1204224xf32>
    %v4246 = stablehlo.multiply %v4245, %v4244 : tensor<32x1204224xf32>
    %v4247 = stablehlo.tanh %v4246 : tensor<32x1204224xf32>
    %v4248 = stablehlo.constant dense<1.0> : tensor<32x1204224xf32>
    %v4249 = stablehlo.add %v4248, %v4247 : tensor<32x1204224xf32>
    %v4250 = stablehlo.constant dense<0.5> : tensor<32x1204224xf32>
    %v4251 = stablehlo.multiply %v4250, %v4249 : tensor<32x1204224xf32>
    %v4252 = stablehlo.multiply %v4247, %v4247 : tensor<32x1204224xf32>
    %v4253 = stablehlo.subtract %v4248, %v4252 : tensor<32x1204224xf32>
    %v4254 = stablehlo.multiply %v4250, %v82 : tensor<32x1204224xf32>
    %v4255 = stablehlo.multiply %v4254, %v4253 : tensor<32x1204224xf32>
    %v4256 = stablehlo.constant dense<0.134145> : tensor<32x1204224xf32>
    %v4257 = stablehlo.multiply %v4256, %v4240 : tensor<32x1204224xf32>
    %v4258 = stablehlo.add %v4248, %v4257 : tensor<32x1204224xf32>
    %v4259 = stablehlo.multiply %v4245, %v4258 : tensor<32x1204224xf32>
    %v4260 = stablehlo.multiply %v4255, %v4259 : tensor<32x1204224xf32>
    %v4261 = stablehlo.add %v4251, %v4260 : tensor<32x1204224xf32>
    %v4262 = stablehlo.multiply %v4239, %v4261 : tensor<32x1204224xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4264 = stablehlo.reverse %s0b0eW, dims = [2, 3] : tensor<384x96x1x1xf32>
    %v4265 = stablehlo.transpose %v4264, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4266 = stablehlo.convolution(%v4263, %v4265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x384x56x56xf32>, tensor<96x384x1x1xf32>) -> tensor<32x96x56x56xf32>
    %v4267 = stablehlo.reshape %v4266 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4268 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4269 = stablehlo.transpose %v4268, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4270 = stablehlo.reshape %v4269 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4271 = stablehlo.reshape %v4267 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4272 = stablehlo.transpose %v4271, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4273 = stablehlo.reshape %v4272 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4274 = stablehlo.reshape %v4273 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4275 = stablehlo.broadcast_in_dim %s0b0ng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4276 = stablehlo.multiply %v4274, %v4275 : tensor<32x3136x96xf32>
    %v4277 = stablehlo.reshape %v4276 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4278 = stablehlo.reshape %v4277 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4279 = stablehlo.reshape %v4270 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4281 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4282 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4283 = stablehlo.reduce(%v4279 init: %v4280) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4284 = stablehlo.broadcast_in_dim %v4283, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4285 = stablehlo.divide %v4284, %v4281 : tensor<32x3136x96xf32>
    %v4286 = stablehlo.subtract %v4279, %v4285 : tensor<32x3136x96xf32>
    %v4287 = stablehlo.multiply %v4286, %v4286 : tensor<32x3136x96xf32>
    %v4288 = stablehlo.reduce(%v4287 init: %v4280) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4289 = stablehlo.broadcast_in_dim %v4288, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4290 = stablehlo.divide %v4289, %v4281 : tensor<32x3136x96xf32>
    %v4291 = stablehlo.add %v4290, %v4282 : tensor<32x3136x96xf32>
    %v4292 = stablehlo.rsqrt %v4291 : tensor<32x3136x96xf32>
    %v4293 = stablehlo.multiply %v4286, %v4292 : tensor<32x3136x96xf32>
    %v4294 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4295 = stablehlo.multiply %v4294, %v4278 : tensor<32x3136x96xf32>
    %v4296 = stablehlo.reduce(%v4295 init: %v4280) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4297 = stablehlo.broadcast_in_dim %v4296, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4298 = stablehlo.multiply %v4293, %v4295 : tensor<32x3136x96xf32>
    %v4299 = stablehlo.reduce(%v4298 init: %v4280) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4300 = stablehlo.broadcast_in_dim %v4299, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4301 = stablehlo.multiply %v4295, %v4281 : tensor<32x3136x96xf32>
    %v4302 = stablehlo.subtract %v4301, %v4297 : tensor<32x3136x96xf32>
    %v4303 = stablehlo.multiply %v4293, %v4300 : tensor<32x3136x96xf32>
    %v4304 = stablehlo.subtract %v4302, %v4303 : tensor<32x3136x96xf32>
    %v4305 = stablehlo.divide %v4292, %v4281 : tensor<32x3136x96xf32>
    %v4306 = stablehlo.multiply %v4305, %v4304 : tensor<32x3136x96xf32>
    %v4307 = stablehlo.reshape %v4306 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4308 = stablehlo.reshape %v4307 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4309 = stablehlo.transpose %v4308, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4310 = stablehlo.reshape %v4309 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4311 = stablehlo.reshape %v4310 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4312 = stablehlo.reverse %s0b0dW, dims = [2, 3] : tensor<96x1x7x7xf32>
    %v4313 = stablehlo.convolution(%v4311, %v4312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 96 : i64} : (tensor<32x96x56x56xf32>, tensor<96x1x7x7xf32>) -> tensor<32x96x56x56xf32>
    %v4314 = stablehlo.reshape %v4313 : (tensor<32x96x56x56xf32>) -> tensor<32x301056xf32>
    %v4315 = stablehlo.add %v4314, %v4166 : tensor<32x301056xf32>
    %v4316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4317 = stablehlo.reshape %v100 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4318 = stablehlo.reshape %v4230 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4319 = stablehlo.multiply %v4317, %v4318 : tensor<32x96x56x56xf32>
    %v4320 = stablehlo.reduce(%v4319 init: %v4316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4321 = stablehlo.reshape %v95 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4322 = stablehlo.reshape %v4234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4323 = stablehlo.transpose %v4321, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4324 = stablehlo.transpose %v4322, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4325 = stablehlo.convolution(%v4323, %v4324)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<384x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<384x96x1x1xf32>
    %v4326 = stablehlo.transpose %v4325, dims = [1, 0, 2, 3] : (tensor<384x96x1x1xf32>) -> tensor<96x384x1x1xf32>
    %v4327 = stablehlo.reshape %v4234 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4329 = stablehlo.reduce(%v4327 init: %v4328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4330 = stablehlo.reshape %v77 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4331 = stablehlo.reshape %v4262 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4332 = stablehlo.transpose %v4330, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4333 = stablehlo.transpose %v4331, dims = [1, 0, 2, 3] : (tensor<32x384x56x56xf32>) -> tensor<384x32x56x56xf32>
    %v4334 = stablehlo.convolution(%v4332, %v4333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<384x32x56x56xf32>) -> tensor<96x384x1x1xf32>
    %v4335 = stablehlo.transpose %v4334, dims = [1, 0, 2, 3] : (tensor<96x384x1x1xf32>) -> tensor<384x96x1x1xf32>
    %v4336 = stablehlo.reshape %v4262 : (tensor<32x1204224xf32>) -> tensor<32x384x56x56xf32>
    %v4337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4338 = stablehlo.reduce(%v4336 init: %v4337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x384x56x56xf32>, tensor<f32>) -> tensor<384xf32>
    %v4339 = stablehlo.reshape %v43 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4340 = stablehlo.transpose %v4339, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4342 = stablehlo.reshape %v4267 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4343 = stablehlo.transpose %v4342, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4344 = stablehlo.reshape %v4343 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4345 = stablehlo.reshape %v4341 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4346 = stablehlo.reshape %v4344 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4347 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4348 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4349 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4350 = stablehlo.reduce(%v4345 init: %v4347) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4350, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4352 = stablehlo.divide %v4351, %v4348 : tensor<32x3136x96xf32>
    %v4353 = stablehlo.subtract %v4345, %v4352 : tensor<32x3136x96xf32>
    %v4354 = stablehlo.multiply %v4353, %v4353 : tensor<32x3136x96xf32>
    %v4355 = stablehlo.reduce(%v4354 init: %v4347) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4356 = stablehlo.broadcast_in_dim %v4355, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4357 = stablehlo.divide %v4356, %v4348 : tensor<32x3136x96xf32>
    %v4358 = stablehlo.add %v4357, %v4349 : tensor<32x3136x96xf32>
    %v4359 = stablehlo.rsqrt %v4358 : tensor<32x3136x96xf32>
    %v4360 = stablehlo.multiply %v4353, %v4359 : tensor<32x3136x96xf32>
    %v4361 = stablehlo.multiply %v4346, %v4360 : tensor<32x3136x96xf32>
    %v4362 = stablehlo.reduce(%v4361 init: %v4347) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4363 = stablehlo.reshape %v4267 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4364 = stablehlo.transpose %v4363, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4367 = stablehlo.reshape %v4365 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4368 = stablehlo.reduce(%v4367 init: %v4366) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4369 = stablehlo.reshape %v38 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4370 = stablehlo.reshape %v4310 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4371 = stablehlo.transpose %v4369, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4372 = stablehlo.transpose %v4370, dims = [1, 0, 2, 3] : (tensor<32x96x56x56xf32>) -> tensor<96x32x56x56xf32>
    %v4373 = stablehlo.convolution(%v4371, %v4372)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 96 : i64, feature_group_count = 1 : i64} : (tensor<96x32x56x56xf32>, tensor<96x32x56x56xf32>) -> tensor<1x96x7x7xf32>
    %v4374 = stablehlo.reshape %v4373 : (tensor<1x96x7x7xf32>) -> tensor<96x1x7x7xf32>
    %v4375 = stablehlo.reshape %v4310 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4377 = stablehlo.reduce(%v4375 init: %v4376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %v4378 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4379 = stablehlo.transpose %v4378, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4380 = stablehlo.reshape %v4379 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4381 = stablehlo.reshape %v4315 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4382 = stablehlo.transpose %v4381, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4383 = stablehlo.reshape %v4382 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4384 = stablehlo.reshape %v4380 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4385 = stablehlo.reshape %v4383 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4387 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4388 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4389 = stablehlo.reduce(%v4384 init: %v4386) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4390 = stablehlo.broadcast_in_dim %v4389, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4391 = stablehlo.divide %v4390, %v4387 : tensor<32x3136x96xf32>
    %v4392 = stablehlo.subtract %v4384, %v4391 : tensor<32x3136x96xf32>
    %v4393 = stablehlo.multiply %v4392, %v4392 : tensor<32x3136x96xf32>
    %v4394 = stablehlo.reduce(%v4393 init: %v4386) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4395 = stablehlo.broadcast_in_dim %v4394, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4396 = stablehlo.divide %v4395, %v4387 : tensor<32x3136x96xf32>
    %v4397 = stablehlo.add %v4396, %v4388 : tensor<32x3136x96xf32>
    %v4398 = stablehlo.rsqrt %v4397 : tensor<32x3136x96xf32>
    %v4399 = stablehlo.multiply %v4392, %v4398 : tensor<32x3136x96xf32>
    %v4400 = stablehlo.multiply %v4385, %v4399 : tensor<32x3136x96xf32>
    %v4401 = stablehlo.reduce(%v4400 init: %v4386) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4402 = stablehlo.reshape %v4315 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4403 = stablehlo.transpose %v4402, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4404 = stablehlo.reshape %v4403 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4406 = stablehlo.reshape %v4404 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4407 = stablehlo.reduce(%v4406 init: %v4405) applies stablehlo.add across dimensions = [0, 1] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<96xf32>
    %v4408 = stablehlo.reshape %v4 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4409 = stablehlo.transpose %v4408, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4410 = stablehlo.reshape %v4409 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4411 = stablehlo.reshape %v4315 : (tensor<32x301056xf32>) -> tensor<32x96x3136xf32>
    %v4412 = stablehlo.transpose %v4411, dims = [0, 2, 1] : (tensor<32x96x3136xf32>) -> tensor<32x3136x96xf32>
    %v4413 = stablehlo.reshape %v4412 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4414 = stablehlo.reshape %v4413 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4415 = stablehlo.broadcast_in_dim %psng, dims = [2] : (tensor<96xf32>) -> tensor<32x3136x96xf32>
    %v4416 = stablehlo.multiply %v4414, %v4415 : tensor<32x3136x96xf32>
    %v4417 = stablehlo.reshape %v4416 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4418 = stablehlo.reshape %v4417 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4419 = stablehlo.reshape %v4410 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4421 = stablehlo.constant dense<96.0> : tensor<32x3136x96xf32>
    %v4422 = stablehlo.constant dense<1.0e-6> : tensor<32x3136x96xf32>
    %v4423 = stablehlo.reduce(%v4419 init: %v4420) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4424 = stablehlo.broadcast_in_dim %v4423, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4425 = stablehlo.divide %v4424, %v4421 : tensor<32x3136x96xf32>
    %v4426 = stablehlo.subtract %v4419, %v4425 : tensor<32x3136x96xf32>
    %v4427 = stablehlo.multiply %v4426, %v4426 : tensor<32x3136x96xf32>
    %v4428 = stablehlo.reduce(%v4427 init: %v4420) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4429 = stablehlo.broadcast_in_dim %v4428, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4430 = stablehlo.divide %v4429, %v4421 : tensor<32x3136x96xf32>
    %v4431 = stablehlo.add %v4430, %v4422 : tensor<32x3136x96xf32>
    %v4432 = stablehlo.rsqrt %v4431 : tensor<32x3136x96xf32>
    %v4433 = stablehlo.multiply %v4426, %v4432 : tensor<32x3136x96xf32>
    %v4434 = stablehlo.broadcast_in_dim %one, dims = [] : (tensor<f32>) -> tensor<32x3136x96xf32>
    %v4435 = stablehlo.multiply %v4434, %v4418 : tensor<32x3136x96xf32>
    %v4436 = stablehlo.reduce(%v4435 init: %v4420) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4437 = stablehlo.broadcast_in_dim %v4436, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4438 = stablehlo.multiply %v4433, %v4435 : tensor<32x3136x96xf32>
    %v4439 = stablehlo.reduce(%v4438 init: %v4420) applies stablehlo.add across dimensions = [2] : (tensor<32x3136x96xf32>, tensor<f32>) -> tensor<32x3136xf32>
    %v4440 = stablehlo.broadcast_in_dim %v4439, dims = [0, 1] : (tensor<32x3136xf32>) -> tensor<32x3136x96xf32>
    %v4441 = stablehlo.multiply %v4435, %v4421 : tensor<32x3136x96xf32>
    %v4442 = stablehlo.subtract %v4441, %v4437 : tensor<32x3136x96xf32>
    %v4443 = stablehlo.multiply %v4433, %v4440 : tensor<32x3136x96xf32>
    %v4444 = stablehlo.subtract %v4442, %v4443 : tensor<32x3136x96xf32>
    %v4445 = stablehlo.divide %v4432, %v4421 : tensor<32x3136x96xf32>
    %v4446 = stablehlo.multiply %v4445, %v4444 : tensor<32x3136x96xf32>
    %v4447 = stablehlo.reshape %v4446 : (tensor<32x3136x96xf32>) -> tensor<32x301056xf32>
    %v4448 = stablehlo.reshape %v4447 : (tensor<32x301056xf32>) -> tensor<32x3136x96xf32>
    %v4449 = stablehlo.transpose %v4448, dims = [0, 2, 1] : (tensor<32x3136x96xf32>) -> tensor<32x96x3136xf32>
    %v4450 = stablehlo.reshape %v4449 : (tensor<32x96x3136xf32>) -> tensor<32x301056xf32>
    %v4454 = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %v4455 = stablehlo.reshape %v4450 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4457 = stablehlo.pad %v4455, %v4456, low = [0, 0, 0, 0], high = [0, 0, 0, 0], interior = [0, 0, 3, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<32x96x221x221xf32>
    %v4458 = stablehlo.transpose %v4454, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %v4459 = stablehlo.transpose %v4457, dims = [1, 0, 2, 3] : (tensor<32x96x221x221xf32>) -> tensor<96x32x221x221xf32>
    %v4460 = stablehlo.convolution(%v4458, %v4459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<96x32x221x221xf32>) -> tensor<3x96x4x4xf32>
    %v4461 = stablehlo.transpose %v4460, dims = [1, 0, 2, 3] : (tensor<3x96x4x4xf32>) -> tensor<96x3x4x4xf32>
    %v4451 = stablehlo.reshape %v4450 : (tensor<32x301056xf32>) -> tensor<32x96x56x56xf32>
    %v4452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4453 = stablehlo.reduce(%v4451 init: %v4452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x96x56x56xf32>, tensor<f32>) -> tensor<96xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v4462 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4463 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4464 = stablehlo.multiply %v4462, %psWm : tensor<96x3x4x4xf32>
    %v4465 = stablehlo.multiply %v4463, %v4461 : tensor<96x3x4x4xf32>
    %v4466 = stablehlo.add %v4464, %v4465 : tensor<96x3x4x4xf32>
    %v4467 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4468 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4469 = stablehlo.multiply %v4467, %psWv : tensor<96x3x4x4xf32>
    %v4470 = stablehlo.multiply %v4461, %v4461 : tensor<96x3x4x4xf32>
    %v4471 = stablehlo.multiply %v4468, %v4470 : tensor<96x3x4x4xf32>
    %v4472 = stablehlo.add %v4469, %v4471 : tensor<96x3x4x4xf32>
    %v4473 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4474 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4475 = stablehlo.multiply %v4473, %psWm : tensor<96x3x4x4xf32>
    %v4476 = stablehlo.multiply %v4474, %v4461 : tensor<96x3x4x4xf32>
    %v4477 = stablehlo.add %v4475, %v4476 : tensor<96x3x4x4xf32>
    %v4478 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4479 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4480 = stablehlo.multiply %v4478, %psWv : tensor<96x3x4x4xf32>
    %v4481 = stablehlo.multiply %v4461, %v4461 : tensor<96x3x4x4xf32>
    %v4482 = stablehlo.multiply %v4479, %v4481 : tensor<96x3x4x4xf32>
    %v4483 = stablehlo.add %v4480, %v4482 : tensor<96x3x4x4xf32>
    %v4484 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4485 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4486 = stablehlo.divide %v4477, %v4484 : tensor<96x3x4x4xf32>
    %v4487 = stablehlo.divide %v4483, %v4485 : tensor<96x3x4x4xf32>
    %v4488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4489 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4490 = stablehlo.sqrt %v4487 : tensor<96x3x4x4xf32>
    %v4491 = stablehlo.add %v4490, %v4489 : tensor<96x3x4x4xf32>
    %v4492 = stablehlo.divide %v4486, %v4491 : tensor<96x3x4x4xf32>
    %v4493 = stablehlo.multiply %v4488, %v4492 : tensor<96x3x4x4xf32>
    %v4494 = stablehlo.subtract %psW, %v4493 : tensor<96x3x4x4xf32>
    %v4495 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x3x4x4xf32>
    %v4496 = stablehlo.multiply %v4495, %v4488 : tensor<96x3x4x4xf32>
    %v4497 = stablehlo.multiply %v4496, %psW : tensor<96x3x4x4xf32>
    %v4498 = stablehlo.subtract %v4494, %v4497 : tensor<96x3x4x4xf32>
    %v4499 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4500 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4501 = stablehlo.multiply %v4499, %psbm : tensor<96xf32>
    %v4502 = stablehlo.multiply %v4500, %v4453 : tensor<96xf32>
    %v4503 = stablehlo.add %v4501, %v4502 : tensor<96xf32>
    %v4504 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4505 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4506 = stablehlo.multiply %v4504, %psbv : tensor<96xf32>
    %v4507 = stablehlo.multiply %v4453, %v4453 : tensor<96xf32>
    %v4508 = stablehlo.multiply %v4505, %v4507 : tensor<96xf32>
    %v4509 = stablehlo.add %v4506, %v4508 : tensor<96xf32>
    %v4510 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4511 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4512 = stablehlo.multiply %v4510, %psbm : tensor<96xf32>
    %v4513 = stablehlo.multiply %v4511, %v4453 : tensor<96xf32>
    %v4514 = stablehlo.add %v4512, %v4513 : tensor<96xf32>
    %v4515 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4516 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4517 = stablehlo.multiply %v4515, %psbv : tensor<96xf32>
    %v4518 = stablehlo.multiply %v4453, %v4453 : tensor<96xf32>
    %v4519 = stablehlo.multiply %v4516, %v4518 : tensor<96xf32>
    %v4520 = stablehlo.add %v4517, %v4519 : tensor<96xf32>
    %v4521 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4522 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4523 = stablehlo.divide %v4514, %v4521 : tensor<96xf32>
    %v4524 = stablehlo.divide %v4520, %v4522 : tensor<96xf32>
    %v4525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4526 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4527 = stablehlo.sqrt %v4524 : tensor<96xf32>
    %v4528 = stablehlo.add %v4527, %v4526 : tensor<96xf32>
    %v4529 = stablehlo.divide %v4523, %v4528 : tensor<96xf32>
    %v4530 = stablehlo.multiply %v4525, %v4529 : tensor<96xf32>
    %v4531 = stablehlo.subtract %psb, %v4530 : tensor<96xf32>
    %v4532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4533 = stablehlo.multiply %v4532, %v4525 : tensor<96xf32>
    %v4534 = stablehlo.multiply %v4533, %psb : tensor<96xf32>
    %v4535 = stablehlo.subtract %v4531, %v4534 : tensor<96xf32>
    %v4536 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4537 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4538 = stablehlo.multiply %v4536, %psngm : tensor<96xf32>
    %v4539 = stablehlo.multiply %v4537, %v4401 : tensor<96xf32>
    %v4540 = stablehlo.add %v4538, %v4539 : tensor<96xf32>
    %v4541 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4542 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4543 = stablehlo.multiply %v4541, %psngv : tensor<96xf32>
    %v4544 = stablehlo.multiply %v4401, %v4401 : tensor<96xf32>
    %v4545 = stablehlo.multiply %v4542, %v4544 : tensor<96xf32>
    %v4546 = stablehlo.add %v4543, %v4545 : tensor<96xf32>
    %v4547 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4548 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4549 = stablehlo.multiply %v4547, %psngm : tensor<96xf32>
    %v4550 = stablehlo.multiply %v4548, %v4401 : tensor<96xf32>
    %v4551 = stablehlo.add %v4549, %v4550 : tensor<96xf32>
    %v4552 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4553 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4554 = stablehlo.multiply %v4552, %psngv : tensor<96xf32>
    %v4555 = stablehlo.multiply %v4401, %v4401 : tensor<96xf32>
    %v4556 = stablehlo.multiply %v4553, %v4555 : tensor<96xf32>
    %v4557 = stablehlo.add %v4554, %v4556 : tensor<96xf32>
    %v4558 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4559 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4560 = stablehlo.divide %v4551, %v4558 : tensor<96xf32>
    %v4561 = stablehlo.divide %v4557, %v4559 : tensor<96xf32>
    %v4562 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4563 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4564 = stablehlo.sqrt %v4561 : tensor<96xf32>
    %v4565 = stablehlo.add %v4564, %v4563 : tensor<96xf32>
    %v4566 = stablehlo.divide %v4560, %v4565 : tensor<96xf32>
    %v4567 = stablehlo.multiply %v4562, %v4566 : tensor<96xf32>
    %v4568 = stablehlo.subtract %psng, %v4567 : tensor<96xf32>
    %v4569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4570 = stablehlo.multiply %v4569, %v4562 : tensor<96xf32>
    %v4571 = stablehlo.multiply %v4570, %psng : tensor<96xf32>
    %v4572 = stablehlo.subtract %v4568, %v4571 : tensor<96xf32>
    %v4573 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4574 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4575 = stablehlo.multiply %v4573, %psnbtm : tensor<96xf32>
    %v4576 = stablehlo.multiply %v4574, %v4407 : tensor<96xf32>
    %v4577 = stablehlo.add %v4575, %v4576 : tensor<96xf32>
    %v4578 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4579 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4580 = stablehlo.multiply %v4578, %psnbtv : tensor<96xf32>
    %v4581 = stablehlo.multiply %v4407, %v4407 : tensor<96xf32>
    %v4582 = stablehlo.multiply %v4579, %v4581 : tensor<96xf32>
    %v4583 = stablehlo.add %v4580, %v4582 : tensor<96xf32>
    %v4584 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4585 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4586 = stablehlo.multiply %v4584, %psnbtm : tensor<96xf32>
    %v4587 = stablehlo.multiply %v4585, %v4407 : tensor<96xf32>
    %v4588 = stablehlo.add %v4586, %v4587 : tensor<96xf32>
    %v4589 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4590 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4591 = stablehlo.multiply %v4589, %psnbtv : tensor<96xf32>
    %v4592 = stablehlo.multiply %v4407, %v4407 : tensor<96xf32>
    %v4593 = stablehlo.multiply %v4590, %v4592 : tensor<96xf32>
    %v4594 = stablehlo.add %v4591, %v4593 : tensor<96xf32>
    %v4595 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4596 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4597 = stablehlo.divide %v4588, %v4595 : tensor<96xf32>
    %v4598 = stablehlo.divide %v4594, %v4596 : tensor<96xf32>
    %v4599 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4600 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4601 = stablehlo.sqrt %v4598 : tensor<96xf32>
    %v4602 = stablehlo.add %v4601, %v4600 : tensor<96xf32>
    %v4603 = stablehlo.divide %v4597, %v4602 : tensor<96xf32>
    %v4604 = stablehlo.multiply %v4599, %v4603 : tensor<96xf32>
    %v4605 = stablehlo.subtract %psnbt, %v4604 : tensor<96xf32>
    %v4606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4607 = stablehlo.multiply %v4606, %v4599 : tensor<96xf32>
    %v4608 = stablehlo.multiply %v4607, %psnbt : tensor<96xf32>
    %v4609 = stablehlo.subtract %v4605, %v4608 : tensor<96xf32>
    %v4610 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4611 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4612 = stablehlo.multiply %v4610, %s0b0dWm : tensor<96x1x7x7xf32>
    %v4613 = stablehlo.multiply %v4611, %v4374 : tensor<96x1x7x7xf32>
    %v4614 = stablehlo.add %v4612, %v4613 : tensor<96x1x7x7xf32>
    %v4615 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4616 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4617 = stablehlo.multiply %v4615, %s0b0dWv : tensor<96x1x7x7xf32>
    %v4618 = stablehlo.multiply %v4374, %v4374 : tensor<96x1x7x7xf32>
    %v4619 = stablehlo.multiply %v4616, %v4618 : tensor<96x1x7x7xf32>
    %v4620 = stablehlo.add %v4617, %v4619 : tensor<96x1x7x7xf32>
    %v4621 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4622 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4623 = stablehlo.multiply %v4621, %s0b0dWm : tensor<96x1x7x7xf32>
    %v4624 = stablehlo.multiply %v4622, %v4374 : tensor<96x1x7x7xf32>
    %v4625 = stablehlo.add %v4623, %v4624 : tensor<96x1x7x7xf32>
    %v4626 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4627 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4628 = stablehlo.multiply %v4626, %s0b0dWv : tensor<96x1x7x7xf32>
    %v4629 = stablehlo.multiply %v4374, %v4374 : tensor<96x1x7x7xf32>
    %v4630 = stablehlo.multiply %v4627, %v4629 : tensor<96x1x7x7xf32>
    %v4631 = stablehlo.add %v4628, %v4630 : tensor<96x1x7x7xf32>
    %v4632 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4633 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4634 = stablehlo.divide %v4625, %v4632 : tensor<96x1x7x7xf32>
    %v4635 = stablehlo.divide %v4631, %v4633 : tensor<96x1x7x7xf32>
    %v4636 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4637 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4638 = stablehlo.sqrt %v4635 : tensor<96x1x7x7xf32>
    %v4639 = stablehlo.add %v4638, %v4637 : tensor<96x1x7x7xf32>
    %v4640 = stablehlo.divide %v4634, %v4639 : tensor<96x1x7x7xf32>
    %v4641 = stablehlo.multiply %v4636, %v4640 : tensor<96x1x7x7xf32>
    %v4642 = stablehlo.subtract %s0b0dW, %v4641 : tensor<96x1x7x7xf32>
    %v4643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4644 = stablehlo.multiply %v4643, %v4636 : tensor<96x1x7x7xf32>
    %v4645 = stablehlo.multiply %v4644, %s0b0dW : tensor<96x1x7x7xf32>
    %v4646 = stablehlo.subtract %v4642, %v4645 : tensor<96x1x7x7xf32>
    %v4647 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4648 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4649 = stablehlo.multiply %v4647, %s0b0dbm : tensor<96xf32>
    %v4650 = stablehlo.multiply %v4648, %v4377 : tensor<96xf32>
    %v4651 = stablehlo.add %v4649, %v4650 : tensor<96xf32>
    %v4652 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4653 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4654 = stablehlo.multiply %v4652, %s0b0dbv : tensor<96xf32>
    %v4655 = stablehlo.multiply %v4377, %v4377 : tensor<96xf32>
    %v4656 = stablehlo.multiply %v4653, %v4655 : tensor<96xf32>
    %v4657 = stablehlo.add %v4654, %v4656 : tensor<96xf32>
    %v4658 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4659 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4660 = stablehlo.multiply %v4658, %s0b0dbm : tensor<96xf32>
    %v4661 = stablehlo.multiply %v4659, %v4377 : tensor<96xf32>
    %v4662 = stablehlo.add %v4660, %v4661 : tensor<96xf32>
    %v4663 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4664 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4665 = stablehlo.multiply %v4663, %s0b0dbv : tensor<96xf32>
    %v4666 = stablehlo.multiply %v4377, %v4377 : tensor<96xf32>
    %v4667 = stablehlo.multiply %v4664, %v4666 : tensor<96xf32>
    %v4668 = stablehlo.add %v4665, %v4667 : tensor<96xf32>
    %v4669 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4670 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4671 = stablehlo.divide %v4662, %v4669 : tensor<96xf32>
    %v4672 = stablehlo.divide %v4668, %v4670 : tensor<96xf32>
    %v4673 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4674 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4675 = stablehlo.sqrt %v4672 : tensor<96xf32>
    %v4676 = stablehlo.add %v4675, %v4674 : tensor<96xf32>
    %v4677 = stablehlo.divide %v4671, %v4676 : tensor<96xf32>
    %v4678 = stablehlo.multiply %v4673, %v4677 : tensor<96xf32>
    %v4679 = stablehlo.subtract %s0b0db, %v4678 : tensor<96xf32>
    %v4680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4681 = stablehlo.multiply %v4680, %v4673 : tensor<96xf32>
    %v4682 = stablehlo.multiply %v4681, %s0b0db : tensor<96xf32>
    %v4683 = stablehlo.subtract %v4679, %v4682 : tensor<96xf32>
    %v4684 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4685 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4686 = stablehlo.multiply %v4684, %s0b0ngm : tensor<96xf32>
    %v4687 = stablehlo.multiply %v4685, %v4362 : tensor<96xf32>
    %v4688 = stablehlo.add %v4686, %v4687 : tensor<96xf32>
    %v4689 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4690 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4691 = stablehlo.multiply %v4689, %s0b0ngv : tensor<96xf32>
    %v4692 = stablehlo.multiply %v4362, %v4362 : tensor<96xf32>
    %v4693 = stablehlo.multiply %v4690, %v4692 : tensor<96xf32>
    %v4694 = stablehlo.add %v4691, %v4693 : tensor<96xf32>
    %v4695 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4696 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4697 = stablehlo.multiply %v4695, %s0b0ngm : tensor<96xf32>
    %v4698 = stablehlo.multiply %v4696, %v4362 : tensor<96xf32>
    %v4699 = stablehlo.add %v4697, %v4698 : tensor<96xf32>
    %v4700 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4701 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4702 = stablehlo.multiply %v4700, %s0b0ngv : tensor<96xf32>
    %v4703 = stablehlo.multiply %v4362, %v4362 : tensor<96xf32>
    %v4704 = stablehlo.multiply %v4701, %v4703 : tensor<96xf32>
    %v4705 = stablehlo.add %v4702, %v4704 : tensor<96xf32>
    %v4706 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4707 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4708 = stablehlo.divide %v4699, %v4706 : tensor<96xf32>
    %v4709 = stablehlo.divide %v4705, %v4707 : tensor<96xf32>
    %v4710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4711 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4712 = stablehlo.sqrt %v4709 : tensor<96xf32>
    %v4713 = stablehlo.add %v4712, %v4711 : tensor<96xf32>
    %v4714 = stablehlo.divide %v4708, %v4713 : tensor<96xf32>
    %v4715 = stablehlo.multiply %v4710, %v4714 : tensor<96xf32>
    %v4716 = stablehlo.subtract %s0b0ng, %v4715 : tensor<96xf32>
    %v4717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4718 = stablehlo.multiply %v4717, %v4710 : tensor<96xf32>
    %v4719 = stablehlo.multiply %v4718, %s0b0ng : tensor<96xf32>
    %v4720 = stablehlo.subtract %v4716, %v4719 : tensor<96xf32>
    %v4721 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4722 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4723 = stablehlo.multiply %v4721, %s0b0nbtm : tensor<96xf32>
    %v4724 = stablehlo.multiply %v4722, %v4368 : tensor<96xf32>
    %v4725 = stablehlo.add %v4723, %v4724 : tensor<96xf32>
    %v4726 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4727 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4728 = stablehlo.multiply %v4726, %s0b0nbtv : tensor<96xf32>
    %v4729 = stablehlo.multiply %v4368, %v4368 : tensor<96xf32>
    %v4730 = stablehlo.multiply %v4727, %v4729 : tensor<96xf32>
    %v4731 = stablehlo.add %v4728, %v4730 : tensor<96xf32>
    %v4732 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4733 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4734 = stablehlo.multiply %v4732, %s0b0nbtm : tensor<96xf32>
    %v4735 = stablehlo.multiply %v4733, %v4368 : tensor<96xf32>
    %v4736 = stablehlo.add %v4734, %v4735 : tensor<96xf32>
    %v4737 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4738 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4739 = stablehlo.multiply %v4737, %s0b0nbtv : tensor<96xf32>
    %v4740 = stablehlo.multiply %v4368, %v4368 : tensor<96xf32>
    %v4741 = stablehlo.multiply %v4738, %v4740 : tensor<96xf32>
    %v4742 = stablehlo.add %v4739, %v4741 : tensor<96xf32>
    %v4743 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4744 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4745 = stablehlo.divide %v4736, %v4743 : tensor<96xf32>
    %v4746 = stablehlo.divide %v4742, %v4744 : tensor<96xf32>
    %v4747 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4748 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4749 = stablehlo.sqrt %v4746 : tensor<96xf32>
    %v4750 = stablehlo.add %v4749, %v4748 : tensor<96xf32>
    %v4751 = stablehlo.divide %v4745, %v4750 : tensor<96xf32>
    %v4752 = stablehlo.multiply %v4747, %v4751 : tensor<96xf32>
    %v4753 = stablehlo.subtract %s0b0nbt, %v4752 : tensor<96xf32>
    %v4754 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4755 = stablehlo.multiply %v4754, %v4747 : tensor<96xf32>
    %v4756 = stablehlo.multiply %v4755, %s0b0nbt : tensor<96xf32>
    %v4757 = stablehlo.subtract %v4753, %v4756 : tensor<96xf32>
    %v4758 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4759 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4760 = stablehlo.multiply %v4758, %s0b0eWm : tensor<384x96x1x1xf32>
    %v4761 = stablehlo.multiply %v4759, %v4335 : tensor<384x96x1x1xf32>
    %v4762 = stablehlo.add %v4760, %v4761 : tensor<384x96x1x1xf32>
    %v4763 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4764 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4765 = stablehlo.multiply %v4763, %s0b0eWv : tensor<384x96x1x1xf32>
    %v4766 = stablehlo.multiply %v4335, %v4335 : tensor<384x96x1x1xf32>
    %v4767 = stablehlo.multiply %v4764, %v4766 : tensor<384x96x1x1xf32>
    %v4768 = stablehlo.add %v4765, %v4767 : tensor<384x96x1x1xf32>
    %v4769 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4770 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4771 = stablehlo.multiply %v4769, %s0b0eWm : tensor<384x96x1x1xf32>
    %v4772 = stablehlo.multiply %v4770, %v4335 : tensor<384x96x1x1xf32>
    %v4773 = stablehlo.add %v4771, %v4772 : tensor<384x96x1x1xf32>
    %v4774 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4775 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4776 = stablehlo.multiply %v4774, %s0b0eWv : tensor<384x96x1x1xf32>
    %v4777 = stablehlo.multiply %v4335, %v4335 : tensor<384x96x1x1xf32>
    %v4778 = stablehlo.multiply %v4775, %v4777 : tensor<384x96x1x1xf32>
    %v4779 = stablehlo.add %v4776, %v4778 : tensor<384x96x1x1xf32>
    %v4780 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4781 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4782 = stablehlo.divide %v4773, %v4780 : tensor<384x96x1x1xf32>
    %v4783 = stablehlo.divide %v4779, %v4781 : tensor<384x96x1x1xf32>
    %v4784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4785 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4786 = stablehlo.sqrt %v4783 : tensor<384x96x1x1xf32>
    %v4787 = stablehlo.add %v4786, %v4785 : tensor<384x96x1x1xf32>
    %v4788 = stablehlo.divide %v4782, %v4787 : tensor<384x96x1x1xf32>
    %v4789 = stablehlo.multiply %v4784, %v4788 : tensor<384x96x1x1xf32>
    %v4790 = stablehlo.subtract %s0b0eW, %v4789 : tensor<384x96x1x1xf32>
    %v4791 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v4792 = stablehlo.multiply %v4791, %v4784 : tensor<384x96x1x1xf32>
    %v4793 = stablehlo.multiply %v4792, %s0b0eW : tensor<384x96x1x1xf32>
    %v4794 = stablehlo.subtract %v4790, %v4793 : tensor<384x96x1x1xf32>
    %v4795 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4796 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4797 = stablehlo.multiply %v4795, %s0b0ebm : tensor<384xf32>
    %v4798 = stablehlo.multiply %v4796, %v4338 : tensor<384xf32>
    %v4799 = stablehlo.add %v4797, %v4798 : tensor<384xf32>
    %v4800 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4801 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4802 = stablehlo.multiply %v4800, %s0b0ebv : tensor<384xf32>
    %v4803 = stablehlo.multiply %v4338, %v4338 : tensor<384xf32>
    %v4804 = stablehlo.multiply %v4801, %v4803 : tensor<384xf32>
    %v4805 = stablehlo.add %v4802, %v4804 : tensor<384xf32>
    %v4806 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4807 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4808 = stablehlo.multiply %v4806, %s0b0ebm : tensor<384xf32>
    %v4809 = stablehlo.multiply %v4807, %v4338 : tensor<384xf32>
    %v4810 = stablehlo.add %v4808, %v4809 : tensor<384xf32>
    %v4811 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4812 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4813 = stablehlo.multiply %v4811, %s0b0ebv : tensor<384xf32>
    %v4814 = stablehlo.multiply %v4338, %v4338 : tensor<384xf32>
    %v4815 = stablehlo.multiply %v4812, %v4814 : tensor<384xf32>
    %v4816 = stablehlo.add %v4813, %v4815 : tensor<384xf32>
    %v4817 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4818 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4819 = stablehlo.divide %v4810, %v4817 : tensor<384xf32>
    %v4820 = stablehlo.divide %v4816, %v4818 : tensor<384xf32>
    %v4821 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4822 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4823 = stablehlo.sqrt %v4820 : tensor<384xf32>
    %v4824 = stablehlo.add %v4823, %v4822 : tensor<384xf32>
    %v4825 = stablehlo.divide %v4819, %v4824 : tensor<384xf32>
    %v4826 = stablehlo.multiply %v4821, %v4825 : tensor<384xf32>
    %v4827 = stablehlo.subtract %s0b0eb, %v4826 : tensor<384xf32>
    %v4828 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v4829 = stablehlo.multiply %v4828, %v4821 : tensor<384xf32>
    %v4830 = stablehlo.multiply %v4829, %s0b0eb : tensor<384xf32>
    %v4831 = stablehlo.subtract %v4827, %v4830 : tensor<384xf32>
    %v4832 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4833 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4834 = stablehlo.multiply %v4832, %s0b0pWm : tensor<96x384x1x1xf32>
    %v4835 = stablehlo.multiply %v4833, %v4326 : tensor<96x384x1x1xf32>
    %v4836 = stablehlo.add %v4834, %v4835 : tensor<96x384x1x1xf32>
    %v4837 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4838 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4839 = stablehlo.multiply %v4837, %s0b0pWv : tensor<96x384x1x1xf32>
    %v4840 = stablehlo.multiply %v4326, %v4326 : tensor<96x384x1x1xf32>
    %v4841 = stablehlo.multiply %v4838, %v4840 : tensor<96x384x1x1xf32>
    %v4842 = stablehlo.add %v4839, %v4841 : tensor<96x384x1x1xf32>
    %v4843 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4844 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4845 = stablehlo.multiply %v4843, %s0b0pWm : tensor<96x384x1x1xf32>
    %v4846 = stablehlo.multiply %v4844, %v4326 : tensor<96x384x1x1xf32>
    %v4847 = stablehlo.add %v4845, %v4846 : tensor<96x384x1x1xf32>
    %v4848 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4849 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4850 = stablehlo.multiply %v4848, %s0b0pWv : tensor<96x384x1x1xf32>
    %v4851 = stablehlo.multiply %v4326, %v4326 : tensor<96x384x1x1xf32>
    %v4852 = stablehlo.multiply %v4849, %v4851 : tensor<96x384x1x1xf32>
    %v4853 = stablehlo.add %v4850, %v4852 : tensor<96x384x1x1xf32>
    %v4854 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4855 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4856 = stablehlo.divide %v4847, %v4854 : tensor<96x384x1x1xf32>
    %v4857 = stablehlo.divide %v4853, %v4855 : tensor<96x384x1x1xf32>
    %v4858 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4859 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4860 = stablehlo.sqrt %v4857 : tensor<96x384x1x1xf32>
    %v4861 = stablehlo.add %v4860, %v4859 : tensor<96x384x1x1xf32>
    %v4862 = stablehlo.divide %v4856, %v4861 : tensor<96x384x1x1xf32>
    %v4863 = stablehlo.multiply %v4858, %v4862 : tensor<96x384x1x1xf32>
    %v4864 = stablehlo.subtract %s0b0pW, %v4863 : tensor<96x384x1x1xf32>
    %v4865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v4866 = stablehlo.multiply %v4865, %v4858 : tensor<96x384x1x1xf32>
    %v4867 = stablehlo.multiply %v4866, %s0b0pW : tensor<96x384x1x1xf32>
    %v4868 = stablehlo.subtract %v4864, %v4867 : tensor<96x384x1x1xf32>
    %v4869 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4870 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4871 = stablehlo.multiply %v4869, %s0b0pbm : tensor<96xf32>
    %v4872 = stablehlo.multiply %v4870, %v4329 : tensor<96xf32>
    %v4873 = stablehlo.add %v4871, %v4872 : tensor<96xf32>
    %v4874 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4875 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4876 = stablehlo.multiply %v4874, %s0b0pbv : tensor<96xf32>
    %v4877 = stablehlo.multiply %v4329, %v4329 : tensor<96xf32>
    %v4878 = stablehlo.multiply %v4875, %v4877 : tensor<96xf32>
    %v4879 = stablehlo.add %v4876, %v4878 : tensor<96xf32>
    %v4880 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4881 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4882 = stablehlo.multiply %v4880, %s0b0pbm : tensor<96xf32>
    %v4883 = stablehlo.multiply %v4881, %v4329 : tensor<96xf32>
    %v4884 = stablehlo.add %v4882, %v4883 : tensor<96xf32>
    %v4885 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4886 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4887 = stablehlo.multiply %v4885, %s0b0pbv : tensor<96xf32>
    %v4888 = stablehlo.multiply %v4329, %v4329 : tensor<96xf32>
    %v4889 = stablehlo.multiply %v4886, %v4888 : tensor<96xf32>
    %v4890 = stablehlo.add %v4887, %v4889 : tensor<96xf32>
    %v4891 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4892 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4893 = stablehlo.divide %v4884, %v4891 : tensor<96xf32>
    %v4894 = stablehlo.divide %v4890, %v4892 : tensor<96xf32>
    %v4895 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4896 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4897 = stablehlo.sqrt %v4894 : tensor<96xf32>
    %v4898 = stablehlo.add %v4897, %v4896 : tensor<96xf32>
    %v4899 = stablehlo.divide %v4893, %v4898 : tensor<96xf32>
    %v4900 = stablehlo.multiply %v4895, %v4899 : tensor<96xf32>
    %v4901 = stablehlo.subtract %s0b0pb, %v4900 : tensor<96xf32>
    %v4902 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4903 = stablehlo.multiply %v4902, %v4895 : tensor<96xf32>
    %v4904 = stablehlo.multiply %v4903, %s0b0pb : tensor<96xf32>
    %v4905 = stablehlo.subtract %v4901, %v4904 : tensor<96xf32>
    %v4906 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4907 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4908 = stablehlo.multiply %v4906, %s0b0lgm : tensor<96xf32>
    %v4909 = stablehlo.multiply %v4907, %v4320 : tensor<96xf32>
    %v4910 = stablehlo.add %v4908, %v4909 : tensor<96xf32>
    %v4911 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4912 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4913 = stablehlo.multiply %v4911, %s0b0lgv : tensor<96xf32>
    %v4914 = stablehlo.multiply %v4320, %v4320 : tensor<96xf32>
    %v4915 = stablehlo.multiply %v4912, %v4914 : tensor<96xf32>
    %v4916 = stablehlo.add %v4913, %v4915 : tensor<96xf32>
    %v4917 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4918 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4919 = stablehlo.multiply %v4917, %s0b0lgm : tensor<96xf32>
    %v4920 = stablehlo.multiply %v4918, %v4320 : tensor<96xf32>
    %v4921 = stablehlo.add %v4919, %v4920 : tensor<96xf32>
    %v4922 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4923 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4924 = stablehlo.multiply %v4922, %s0b0lgv : tensor<96xf32>
    %v4925 = stablehlo.multiply %v4320, %v4320 : tensor<96xf32>
    %v4926 = stablehlo.multiply %v4923, %v4925 : tensor<96xf32>
    %v4927 = stablehlo.add %v4924, %v4926 : tensor<96xf32>
    %v4928 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4929 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4930 = stablehlo.divide %v4921, %v4928 : tensor<96xf32>
    %v4931 = stablehlo.divide %v4927, %v4929 : tensor<96xf32>
    %v4932 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4933 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4934 = stablehlo.sqrt %v4931 : tensor<96xf32>
    %v4935 = stablehlo.add %v4934, %v4933 : tensor<96xf32>
    %v4936 = stablehlo.divide %v4930, %v4935 : tensor<96xf32>
    %v4937 = stablehlo.multiply %v4932, %v4936 : tensor<96xf32>
    %v4938 = stablehlo.subtract %s0b0lg, %v4937 : tensor<96xf32>
    %v4939 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4940 = stablehlo.multiply %v4939, %v4932 : tensor<96xf32>
    %v4941 = stablehlo.multiply %v4940, %s0b0lg : tensor<96xf32>
    %v4942 = stablehlo.subtract %v4938, %v4941 : tensor<96xf32>
    %v4943 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4944 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4945 = stablehlo.multiply %v4943, %s0b1dWm : tensor<96x1x7x7xf32>
    %v4946 = stablehlo.multiply %v4944, %v4225 : tensor<96x1x7x7xf32>
    %v4947 = stablehlo.add %v4945, %v4946 : tensor<96x1x7x7xf32>
    %v4948 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4949 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4950 = stablehlo.multiply %v4948, %s0b1dWv : tensor<96x1x7x7xf32>
    %v4951 = stablehlo.multiply %v4225, %v4225 : tensor<96x1x7x7xf32>
    %v4952 = stablehlo.multiply %v4949, %v4951 : tensor<96x1x7x7xf32>
    %v4953 = stablehlo.add %v4950, %v4952 : tensor<96x1x7x7xf32>
    %v4954 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4955 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4956 = stablehlo.multiply %v4954, %s0b1dWm : tensor<96x1x7x7xf32>
    %v4957 = stablehlo.multiply %v4955, %v4225 : tensor<96x1x7x7xf32>
    %v4958 = stablehlo.add %v4956, %v4957 : tensor<96x1x7x7xf32>
    %v4959 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4960 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4961 = stablehlo.multiply %v4959, %s0b1dWv : tensor<96x1x7x7xf32>
    %v4962 = stablehlo.multiply %v4225, %v4225 : tensor<96x1x7x7xf32>
    %v4963 = stablehlo.multiply %v4960, %v4962 : tensor<96x1x7x7xf32>
    %v4964 = stablehlo.add %v4961, %v4963 : tensor<96x1x7x7xf32>
    %v4965 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4966 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4967 = stablehlo.divide %v4958, %v4965 : tensor<96x1x7x7xf32>
    %v4968 = stablehlo.divide %v4964, %v4966 : tensor<96x1x7x7xf32>
    %v4969 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4970 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4971 = stablehlo.sqrt %v4968 : tensor<96x1x7x7xf32>
    %v4972 = stablehlo.add %v4971, %v4970 : tensor<96x1x7x7xf32>
    %v4973 = stablehlo.divide %v4967, %v4972 : tensor<96x1x7x7xf32>
    %v4974 = stablehlo.multiply %v4969, %v4973 : tensor<96x1x7x7xf32>
    %v4975 = stablehlo.subtract %s0b1dW, %v4974 : tensor<96x1x7x7xf32>
    %v4976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v4977 = stablehlo.multiply %v4976, %v4969 : tensor<96x1x7x7xf32>
    %v4978 = stablehlo.multiply %v4977, %s0b1dW : tensor<96x1x7x7xf32>
    %v4979 = stablehlo.subtract %v4975, %v4978 : tensor<96x1x7x7xf32>
    %v4980 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4981 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4982 = stablehlo.multiply %v4980, %s0b1dbm : tensor<96xf32>
    %v4983 = stablehlo.multiply %v4981, %v4228 : tensor<96xf32>
    %v4984 = stablehlo.add %v4982, %v4983 : tensor<96xf32>
    %v4985 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4986 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4987 = stablehlo.multiply %v4985, %s0b1dbv : tensor<96xf32>
    %v4988 = stablehlo.multiply %v4228, %v4228 : tensor<96xf32>
    %v4989 = stablehlo.multiply %v4986, %v4988 : tensor<96xf32>
    %v4990 = stablehlo.add %v4987, %v4989 : tensor<96xf32>
    %v4991 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4992 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4993 = stablehlo.multiply %v4991, %s0b1dbm : tensor<96xf32>
    %v4994 = stablehlo.multiply %v4992, %v4228 : tensor<96xf32>
    %v4995 = stablehlo.add %v4993, %v4994 : tensor<96xf32>
    %v4996 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4997 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v4998 = stablehlo.multiply %v4996, %s0b1dbv : tensor<96xf32>
    %v4999 = stablehlo.multiply %v4228, %v4228 : tensor<96xf32>
    %v5000 = stablehlo.multiply %v4997, %v4999 : tensor<96xf32>
    %v5001 = stablehlo.add %v4998, %v5000 : tensor<96xf32>
    %v5002 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5003 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5004 = stablehlo.divide %v4995, %v5002 : tensor<96xf32>
    %v5005 = stablehlo.divide %v5001, %v5003 : tensor<96xf32>
    %v5006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5007 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5008 = stablehlo.sqrt %v5005 : tensor<96xf32>
    %v5009 = stablehlo.add %v5008, %v5007 : tensor<96xf32>
    %v5010 = stablehlo.divide %v5004, %v5009 : tensor<96xf32>
    %v5011 = stablehlo.multiply %v5006, %v5010 : tensor<96xf32>
    %v5012 = stablehlo.subtract %s0b1db, %v5011 : tensor<96xf32>
    %v5013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5014 = stablehlo.multiply %v5013, %v5006 : tensor<96xf32>
    %v5015 = stablehlo.multiply %v5014, %s0b1db : tensor<96xf32>
    %v5016 = stablehlo.subtract %v5012, %v5015 : tensor<96xf32>
    %v5017 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5018 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5019 = stablehlo.multiply %v5017, %s0b1ngm : tensor<96xf32>
    %v5020 = stablehlo.multiply %v5018, %v4213 : tensor<96xf32>
    %v5021 = stablehlo.add %v5019, %v5020 : tensor<96xf32>
    %v5022 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5023 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5024 = stablehlo.multiply %v5022, %s0b1ngv : tensor<96xf32>
    %v5025 = stablehlo.multiply %v4213, %v4213 : tensor<96xf32>
    %v5026 = stablehlo.multiply %v5023, %v5025 : tensor<96xf32>
    %v5027 = stablehlo.add %v5024, %v5026 : tensor<96xf32>
    %v5028 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5029 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5030 = stablehlo.multiply %v5028, %s0b1ngm : tensor<96xf32>
    %v5031 = stablehlo.multiply %v5029, %v4213 : tensor<96xf32>
    %v5032 = stablehlo.add %v5030, %v5031 : tensor<96xf32>
    %v5033 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5034 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5035 = stablehlo.multiply %v5033, %s0b1ngv : tensor<96xf32>
    %v5036 = stablehlo.multiply %v4213, %v4213 : tensor<96xf32>
    %v5037 = stablehlo.multiply %v5034, %v5036 : tensor<96xf32>
    %v5038 = stablehlo.add %v5035, %v5037 : tensor<96xf32>
    %v5039 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5040 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5041 = stablehlo.divide %v5032, %v5039 : tensor<96xf32>
    %v5042 = stablehlo.divide %v5038, %v5040 : tensor<96xf32>
    %v5043 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5044 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5045 = stablehlo.sqrt %v5042 : tensor<96xf32>
    %v5046 = stablehlo.add %v5045, %v5044 : tensor<96xf32>
    %v5047 = stablehlo.divide %v5041, %v5046 : tensor<96xf32>
    %v5048 = stablehlo.multiply %v5043, %v5047 : tensor<96xf32>
    %v5049 = stablehlo.subtract %s0b1ng, %v5048 : tensor<96xf32>
    %v5050 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5051 = stablehlo.multiply %v5050, %v5043 : tensor<96xf32>
    %v5052 = stablehlo.multiply %v5051, %s0b1ng : tensor<96xf32>
    %v5053 = stablehlo.subtract %v5049, %v5052 : tensor<96xf32>
    %v5054 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5055 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5056 = stablehlo.multiply %v5054, %s0b1nbtm : tensor<96xf32>
    %v5057 = stablehlo.multiply %v5055, %v4219 : tensor<96xf32>
    %v5058 = stablehlo.add %v5056, %v5057 : tensor<96xf32>
    %v5059 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5060 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5061 = stablehlo.multiply %v5059, %s0b1nbtv : tensor<96xf32>
    %v5062 = stablehlo.multiply %v4219, %v4219 : tensor<96xf32>
    %v5063 = stablehlo.multiply %v5060, %v5062 : tensor<96xf32>
    %v5064 = stablehlo.add %v5061, %v5063 : tensor<96xf32>
    %v5065 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5066 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5067 = stablehlo.multiply %v5065, %s0b1nbtm : tensor<96xf32>
    %v5068 = stablehlo.multiply %v5066, %v4219 : tensor<96xf32>
    %v5069 = stablehlo.add %v5067, %v5068 : tensor<96xf32>
    %v5070 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5071 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5072 = stablehlo.multiply %v5070, %s0b1nbtv : tensor<96xf32>
    %v5073 = stablehlo.multiply %v4219, %v4219 : tensor<96xf32>
    %v5074 = stablehlo.multiply %v5071, %v5073 : tensor<96xf32>
    %v5075 = stablehlo.add %v5072, %v5074 : tensor<96xf32>
    %v5076 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5077 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5078 = stablehlo.divide %v5069, %v5076 : tensor<96xf32>
    %v5079 = stablehlo.divide %v5075, %v5077 : tensor<96xf32>
    %v5080 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5081 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5082 = stablehlo.sqrt %v5079 : tensor<96xf32>
    %v5083 = stablehlo.add %v5082, %v5081 : tensor<96xf32>
    %v5084 = stablehlo.divide %v5078, %v5083 : tensor<96xf32>
    %v5085 = stablehlo.multiply %v5080, %v5084 : tensor<96xf32>
    %v5086 = stablehlo.subtract %s0b1nbt, %v5085 : tensor<96xf32>
    %v5087 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5088 = stablehlo.multiply %v5087, %v5080 : tensor<96xf32>
    %v5089 = stablehlo.multiply %v5088, %s0b1nbt : tensor<96xf32>
    %v5090 = stablehlo.subtract %v5086, %v5089 : tensor<96xf32>
    %v5091 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5092 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5093 = stablehlo.multiply %v5091, %s0b1eWm : tensor<384x96x1x1xf32>
    %v5094 = stablehlo.multiply %v5092, %v4186 : tensor<384x96x1x1xf32>
    %v5095 = stablehlo.add %v5093, %v5094 : tensor<384x96x1x1xf32>
    %v5096 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5097 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5098 = stablehlo.multiply %v5096, %s0b1eWv : tensor<384x96x1x1xf32>
    %v5099 = stablehlo.multiply %v4186, %v4186 : tensor<384x96x1x1xf32>
    %v5100 = stablehlo.multiply %v5097, %v5099 : tensor<384x96x1x1xf32>
    %v5101 = stablehlo.add %v5098, %v5100 : tensor<384x96x1x1xf32>
    %v5102 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5103 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5104 = stablehlo.multiply %v5102, %s0b1eWm : tensor<384x96x1x1xf32>
    %v5105 = stablehlo.multiply %v5103, %v4186 : tensor<384x96x1x1xf32>
    %v5106 = stablehlo.add %v5104, %v5105 : tensor<384x96x1x1xf32>
    %v5107 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5108 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5109 = stablehlo.multiply %v5107, %s0b1eWv : tensor<384x96x1x1xf32>
    %v5110 = stablehlo.multiply %v4186, %v4186 : tensor<384x96x1x1xf32>
    %v5111 = stablehlo.multiply %v5108, %v5110 : tensor<384x96x1x1xf32>
    %v5112 = stablehlo.add %v5109, %v5111 : tensor<384x96x1x1xf32>
    %v5113 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5114 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5115 = stablehlo.divide %v5106, %v5113 : tensor<384x96x1x1xf32>
    %v5116 = stablehlo.divide %v5112, %v5114 : tensor<384x96x1x1xf32>
    %v5117 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5118 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5119 = stablehlo.sqrt %v5116 : tensor<384x96x1x1xf32>
    %v5120 = stablehlo.add %v5119, %v5118 : tensor<384x96x1x1xf32>
    %v5121 = stablehlo.divide %v5115, %v5120 : tensor<384x96x1x1xf32>
    %v5122 = stablehlo.multiply %v5117, %v5121 : tensor<384x96x1x1xf32>
    %v5123 = stablehlo.subtract %s0b1eW, %v5122 : tensor<384x96x1x1xf32>
    %v5124 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5125 = stablehlo.multiply %v5124, %v5117 : tensor<384x96x1x1xf32>
    %v5126 = stablehlo.multiply %v5125, %s0b1eW : tensor<384x96x1x1xf32>
    %v5127 = stablehlo.subtract %v5123, %v5126 : tensor<384x96x1x1xf32>
    %v5128 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5129 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5130 = stablehlo.multiply %v5128, %s0b1ebm : tensor<384xf32>
    %v5131 = stablehlo.multiply %v5129, %v4189 : tensor<384xf32>
    %v5132 = stablehlo.add %v5130, %v5131 : tensor<384xf32>
    %v5133 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5134 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5135 = stablehlo.multiply %v5133, %s0b1ebv : tensor<384xf32>
    %v5136 = stablehlo.multiply %v4189, %v4189 : tensor<384xf32>
    %v5137 = stablehlo.multiply %v5134, %v5136 : tensor<384xf32>
    %v5138 = stablehlo.add %v5135, %v5137 : tensor<384xf32>
    %v5139 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5140 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5141 = stablehlo.multiply %v5139, %s0b1ebm : tensor<384xf32>
    %v5142 = stablehlo.multiply %v5140, %v4189 : tensor<384xf32>
    %v5143 = stablehlo.add %v5141, %v5142 : tensor<384xf32>
    %v5144 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5145 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5146 = stablehlo.multiply %v5144, %s0b1ebv : tensor<384xf32>
    %v5147 = stablehlo.multiply %v4189, %v4189 : tensor<384xf32>
    %v5148 = stablehlo.multiply %v5145, %v5147 : tensor<384xf32>
    %v5149 = stablehlo.add %v5146, %v5148 : tensor<384xf32>
    %v5150 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5151 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5152 = stablehlo.divide %v5143, %v5150 : tensor<384xf32>
    %v5153 = stablehlo.divide %v5149, %v5151 : tensor<384xf32>
    %v5154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5155 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5156 = stablehlo.sqrt %v5153 : tensor<384xf32>
    %v5157 = stablehlo.add %v5156, %v5155 : tensor<384xf32>
    %v5158 = stablehlo.divide %v5152, %v5157 : tensor<384xf32>
    %v5159 = stablehlo.multiply %v5154, %v5158 : tensor<384xf32>
    %v5160 = stablehlo.subtract %s0b1eb, %v5159 : tensor<384xf32>
    %v5161 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5162 = stablehlo.multiply %v5161, %v5154 : tensor<384xf32>
    %v5163 = stablehlo.multiply %v5162, %s0b1eb : tensor<384xf32>
    %v5164 = stablehlo.subtract %v5160, %v5163 : tensor<384xf32>
    %v5165 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5166 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5167 = stablehlo.multiply %v5165, %s0b1pWm : tensor<96x384x1x1xf32>
    %v5168 = stablehlo.multiply %v5166, %v4177 : tensor<96x384x1x1xf32>
    %v5169 = stablehlo.add %v5167, %v5168 : tensor<96x384x1x1xf32>
    %v5170 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5171 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5172 = stablehlo.multiply %v5170, %s0b1pWv : tensor<96x384x1x1xf32>
    %v5173 = stablehlo.multiply %v4177, %v4177 : tensor<96x384x1x1xf32>
    %v5174 = stablehlo.multiply %v5171, %v5173 : tensor<96x384x1x1xf32>
    %v5175 = stablehlo.add %v5172, %v5174 : tensor<96x384x1x1xf32>
    %v5176 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5177 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5178 = stablehlo.multiply %v5176, %s0b1pWm : tensor<96x384x1x1xf32>
    %v5179 = stablehlo.multiply %v5177, %v4177 : tensor<96x384x1x1xf32>
    %v5180 = stablehlo.add %v5178, %v5179 : tensor<96x384x1x1xf32>
    %v5181 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5182 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5183 = stablehlo.multiply %v5181, %s0b1pWv : tensor<96x384x1x1xf32>
    %v5184 = stablehlo.multiply %v4177, %v4177 : tensor<96x384x1x1xf32>
    %v5185 = stablehlo.multiply %v5182, %v5184 : tensor<96x384x1x1xf32>
    %v5186 = stablehlo.add %v5183, %v5185 : tensor<96x384x1x1xf32>
    %v5187 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5188 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5189 = stablehlo.divide %v5180, %v5187 : tensor<96x384x1x1xf32>
    %v5190 = stablehlo.divide %v5186, %v5188 : tensor<96x384x1x1xf32>
    %v5191 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5192 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5193 = stablehlo.sqrt %v5190 : tensor<96x384x1x1xf32>
    %v5194 = stablehlo.add %v5193, %v5192 : tensor<96x384x1x1xf32>
    %v5195 = stablehlo.divide %v5189, %v5194 : tensor<96x384x1x1xf32>
    %v5196 = stablehlo.multiply %v5191, %v5195 : tensor<96x384x1x1xf32>
    %v5197 = stablehlo.subtract %s0b1pW, %v5196 : tensor<96x384x1x1xf32>
    %v5198 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5199 = stablehlo.multiply %v5198, %v5191 : tensor<96x384x1x1xf32>
    %v5200 = stablehlo.multiply %v5199, %s0b1pW : tensor<96x384x1x1xf32>
    %v5201 = stablehlo.subtract %v5197, %v5200 : tensor<96x384x1x1xf32>
    %v5202 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5203 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5204 = stablehlo.multiply %v5202, %s0b1pbm : tensor<96xf32>
    %v5205 = stablehlo.multiply %v5203, %v4180 : tensor<96xf32>
    %v5206 = stablehlo.add %v5204, %v5205 : tensor<96xf32>
    %v5207 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5208 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5209 = stablehlo.multiply %v5207, %s0b1pbv : tensor<96xf32>
    %v5210 = stablehlo.multiply %v4180, %v4180 : tensor<96xf32>
    %v5211 = stablehlo.multiply %v5208, %v5210 : tensor<96xf32>
    %v5212 = stablehlo.add %v5209, %v5211 : tensor<96xf32>
    %v5213 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5214 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5215 = stablehlo.multiply %v5213, %s0b1pbm : tensor<96xf32>
    %v5216 = stablehlo.multiply %v5214, %v4180 : tensor<96xf32>
    %v5217 = stablehlo.add %v5215, %v5216 : tensor<96xf32>
    %v5218 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5219 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5220 = stablehlo.multiply %v5218, %s0b1pbv : tensor<96xf32>
    %v5221 = stablehlo.multiply %v4180, %v4180 : tensor<96xf32>
    %v5222 = stablehlo.multiply %v5219, %v5221 : tensor<96xf32>
    %v5223 = stablehlo.add %v5220, %v5222 : tensor<96xf32>
    %v5224 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5225 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5226 = stablehlo.divide %v5217, %v5224 : tensor<96xf32>
    %v5227 = stablehlo.divide %v5223, %v5225 : tensor<96xf32>
    %v5228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5229 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5230 = stablehlo.sqrt %v5227 : tensor<96xf32>
    %v5231 = stablehlo.add %v5230, %v5229 : tensor<96xf32>
    %v5232 = stablehlo.divide %v5226, %v5231 : tensor<96xf32>
    %v5233 = stablehlo.multiply %v5228, %v5232 : tensor<96xf32>
    %v5234 = stablehlo.subtract %s0b1pb, %v5233 : tensor<96xf32>
    %v5235 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5236 = stablehlo.multiply %v5235, %v5228 : tensor<96xf32>
    %v5237 = stablehlo.multiply %v5236, %s0b1pb : tensor<96xf32>
    %v5238 = stablehlo.subtract %v5234, %v5237 : tensor<96xf32>
    %v5239 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5240 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5241 = stablehlo.multiply %v5239, %s0b1lgm : tensor<96xf32>
    %v5242 = stablehlo.multiply %v5240, %v4171 : tensor<96xf32>
    %v5243 = stablehlo.add %v5241, %v5242 : tensor<96xf32>
    %v5244 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5245 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5246 = stablehlo.multiply %v5244, %s0b1lgv : tensor<96xf32>
    %v5247 = stablehlo.multiply %v4171, %v4171 : tensor<96xf32>
    %v5248 = stablehlo.multiply %v5245, %v5247 : tensor<96xf32>
    %v5249 = stablehlo.add %v5246, %v5248 : tensor<96xf32>
    %v5250 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5251 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5252 = stablehlo.multiply %v5250, %s0b1lgm : tensor<96xf32>
    %v5253 = stablehlo.multiply %v5251, %v4171 : tensor<96xf32>
    %v5254 = stablehlo.add %v5252, %v5253 : tensor<96xf32>
    %v5255 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5256 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5257 = stablehlo.multiply %v5255, %s0b1lgv : tensor<96xf32>
    %v5258 = stablehlo.multiply %v4171, %v4171 : tensor<96xf32>
    %v5259 = stablehlo.multiply %v5256, %v5258 : tensor<96xf32>
    %v5260 = stablehlo.add %v5257, %v5259 : tensor<96xf32>
    %v5261 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5262 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5263 = stablehlo.divide %v5254, %v5261 : tensor<96xf32>
    %v5264 = stablehlo.divide %v5260, %v5262 : tensor<96xf32>
    %v5265 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5266 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5267 = stablehlo.sqrt %v5264 : tensor<96xf32>
    %v5268 = stablehlo.add %v5267, %v5266 : tensor<96xf32>
    %v5269 = stablehlo.divide %v5263, %v5268 : tensor<96xf32>
    %v5270 = stablehlo.multiply %v5265, %v5269 : tensor<96xf32>
    %v5271 = stablehlo.subtract %s0b1lg, %v5270 : tensor<96xf32>
    %v5272 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5273 = stablehlo.multiply %v5272, %v5265 : tensor<96xf32>
    %v5274 = stablehlo.multiply %v5273, %s0b1lg : tensor<96xf32>
    %v5275 = stablehlo.subtract %v5271, %v5274 : tensor<96xf32>
    %v5276 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5277 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5278 = stablehlo.multiply %v5276, %s0b2dWm : tensor<96x1x7x7xf32>
    %v5279 = stablehlo.multiply %v5277, %v4076 : tensor<96x1x7x7xf32>
    %v5280 = stablehlo.add %v5278, %v5279 : tensor<96x1x7x7xf32>
    %v5281 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5282 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5283 = stablehlo.multiply %v5281, %s0b2dWv : tensor<96x1x7x7xf32>
    %v5284 = stablehlo.multiply %v4076, %v4076 : tensor<96x1x7x7xf32>
    %v5285 = stablehlo.multiply %v5282, %v5284 : tensor<96x1x7x7xf32>
    %v5286 = stablehlo.add %v5283, %v5285 : tensor<96x1x7x7xf32>
    %v5287 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5288 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5289 = stablehlo.multiply %v5287, %s0b2dWm : tensor<96x1x7x7xf32>
    %v5290 = stablehlo.multiply %v5288, %v4076 : tensor<96x1x7x7xf32>
    %v5291 = stablehlo.add %v5289, %v5290 : tensor<96x1x7x7xf32>
    %v5292 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5293 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5294 = stablehlo.multiply %v5292, %s0b2dWv : tensor<96x1x7x7xf32>
    %v5295 = stablehlo.multiply %v4076, %v4076 : tensor<96x1x7x7xf32>
    %v5296 = stablehlo.multiply %v5293, %v5295 : tensor<96x1x7x7xf32>
    %v5297 = stablehlo.add %v5294, %v5296 : tensor<96x1x7x7xf32>
    %v5298 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5299 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5300 = stablehlo.divide %v5291, %v5298 : tensor<96x1x7x7xf32>
    %v5301 = stablehlo.divide %v5297, %v5299 : tensor<96x1x7x7xf32>
    %v5302 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5303 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5304 = stablehlo.sqrt %v5301 : tensor<96x1x7x7xf32>
    %v5305 = stablehlo.add %v5304, %v5303 : tensor<96x1x7x7xf32>
    %v5306 = stablehlo.divide %v5300, %v5305 : tensor<96x1x7x7xf32>
    %v5307 = stablehlo.multiply %v5302, %v5306 : tensor<96x1x7x7xf32>
    %v5308 = stablehlo.subtract %s0b2dW, %v5307 : tensor<96x1x7x7xf32>
    %v5309 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x1x7x7xf32>
    %v5310 = stablehlo.multiply %v5309, %v5302 : tensor<96x1x7x7xf32>
    %v5311 = stablehlo.multiply %v5310, %s0b2dW : tensor<96x1x7x7xf32>
    %v5312 = stablehlo.subtract %v5308, %v5311 : tensor<96x1x7x7xf32>
    %v5313 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5314 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5315 = stablehlo.multiply %v5313, %s0b2dbm : tensor<96xf32>
    %v5316 = stablehlo.multiply %v5314, %v4079 : tensor<96xf32>
    %v5317 = stablehlo.add %v5315, %v5316 : tensor<96xf32>
    %v5318 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5319 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5320 = stablehlo.multiply %v5318, %s0b2dbv : tensor<96xf32>
    %v5321 = stablehlo.multiply %v4079, %v4079 : tensor<96xf32>
    %v5322 = stablehlo.multiply %v5319, %v5321 : tensor<96xf32>
    %v5323 = stablehlo.add %v5320, %v5322 : tensor<96xf32>
    %v5324 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5325 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5326 = stablehlo.multiply %v5324, %s0b2dbm : tensor<96xf32>
    %v5327 = stablehlo.multiply %v5325, %v4079 : tensor<96xf32>
    %v5328 = stablehlo.add %v5326, %v5327 : tensor<96xf32>
    %v5329 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5330 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5331 = stablehlo.multiply %v5329, %s0b2dbv : tensor<96xf32>
    %v5332 = stablehlo.multiply %v4079, %v4079 : tensor<96xf32>
    %v5333 = stablehlo.multiply %v5330, %v5332 : tensor<96xf32>
    %v5334 = stablehlo.add %v5331, %v5333 : tensor<96xf32>
    %v5335 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5336 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5337 = stablehlo.divide %v5328, %v5335 : tensor<96xf32>
    %v5338 = stablehlo.divide %v5334, %v5336 : tensor<96xf32>
    %v5339 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5340 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5341 = stablehlo.sqrt %v5338 : tensor<96xf32>
    %v5342 = stablehlo.add %v5341, %v5340 : tensor<96xf32>
    %v5343 = stablehlo.divide %v5337, %v5342 : tensor<96xf32>
    %v5344 = stablehlo.multiply %v5339, %v5343 : tensor<96xf32>
    %v5345 = stablehlo.subtract %s0b2db, %v5344 : tensor<96xf32>
    %v5346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5347 = stablehlo.multiply %v5346, %v5339 : tensor<96xf32>
    %v5348 = stablehlo.multiply %v5347, %s0b2db : tensor<96xf32>
    %v5349 = stablehlo.subtract %v5345, %v5348 : tensor<96xf32>
    %v5350 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5351 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5352 = stablehlo.multiply %v5350, %s0b2ngm : tensor<96xf32>
    %v5353 = stablehlo.multiply %v5351, %v4064 : tensor<96xf32>
    %v5354 = stablehlo.add %v5352, %v5353 : tensor<96xf32>
    %v5355 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5356 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5357 = stablehlo.multiply %v5355, %s0b2ngv : tensor<96xf32>
    %v5358 = stablehlo.multiply %v4064, %v4064 : tensor<96xf32>
    %v5359 = stablehlo.multiply %v5356, %v5358 : tensor<96xf32>
    %v5360 = stablehlo.add %v5357, %v5359 : tensor<96xf32>
    %v5361 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5362 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5363 = stablehlo.multiply %v5361, %s0b2ngm : tensor<96xf32>
    %v5364 = stablehlo.multiply %v5362, %v4064 : tensor<96xf32>
    %v5365 = stablehlo.add %v5363, %v5364 : tensor<96xf32>
    %v5366 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5367 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5368 = stablehlo.multiply %v5366, %s0b2ngv : tensor<96xf32>
    %v5369 = stablehlo.multiply %v4064, %v4064 : tensor<96xf32>
    %v5370 = stablehlo.multiply %v5367, %v5369 : tensor<96xf32>
    %v5371 = stablehlo.add %v5368, %v5370 : tensor<96xf32>
    %v5372 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5373 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5374 = stablehlo.divide %v5365, %v5372 : tensor<96xf32>
    %v5375 = stablehlo.divide %v5371, %v5373 : tensor<96xf32>
    %v5376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5377 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5378 = stablehlo.sqrt %v5375 : tensor<96xf32>
    %v5379 = stablehlo.add %v5378, %v5377 : tensor<96xf32>
    %v5380 = stablehlo.divide %v5374, %v5379 : tensor<96xf32>
    %v5381 = stablehlo.multiply %v5376, %v5380 : tensor<96xf32>
    %v5382 = stablehlo.subtract %s0b2ng, %v5381 : tensor<96xf32>
    %v5383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5384 = stablehlo.multiply %v5383, %v5376 : tensor<96xf32>
    %v5385 = stablehlo.multiply %v5384, %s0b2ng : tensor<96xf32>
    %v5386 = stablehlo.subtract %v5382, %v5385 : tensor<96xf32>
    %v5387 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5388 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5389 = stablehlo.multiply %v5387, %s0b2nbtm : tensor<96xf32>
    %v5390 = stablehlo.multiply %v5388, %v4070 : tensor<96xf32>
    %v5391 = stablehlo.add %v5389, %v5390 : tensor<96xf32>
    %v5392 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5393 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5394 = stablehlo.multiply %v5392, %s0b2nbtv : tensor<96xf32>
    %v5395 = stablehlo.multiply %v4070, %v4070 : tensor<96xf32>
    %v5396 = stablehlo.multiply %v5393, %v5395 : tensor<96xf32>
    %v5397 = stablehlo.add %v5394, %v5396 : tensor<96xf32>
    %v5398 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5399 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5400 = stablehlo.multiply %v5398, %s0b2nbtm : tensor<96xf32>
    %v5401 = stablehlo.multiply %v5399, %v4070 : tensor<96xf32>
    %v5402 = stablehlo.add %v5400, %v5401 : tensor<96xf32>
    %v5403 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5404 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5405 = stablehlo.multiply %v5403, %s0b2nbtv : tensor<96xf32>
    %v5406 = stablehlo.multiply %v4070, %v4070 : tensor<96xf32>
    %v5407 = stablehlo.multiply %v5404, %v5406 : tensor<96xf32>
    %v5408 = stablehlo.add %v5405, %v5407 : tensor<96xf32>
    %v5409 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5410 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5411 = stablehlo.divide %v5402, %v5409 : tensor<96xf32>
    %v5412 = stablehlo.divide %v5408, %v5410 : tensor<96xf32>
    %v5413 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5414 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5415 = stablehlo.sqrt %v5412 : tensor<96xf32>
    %v5416 = stablehlo.add %v5415, %v5414 : tensor<96xf32>
    %v5417 = stablehlo.divide %v5411, %v5416 : tensor<96xf32>
    %v5418 = stablehlo.multiply %v5413, %v5417 : tensor<96xf32>
    %v5419 = stablehlo.subtract %s0b2nbt, %v5418 : tensor<96xf32>
    %v5420 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5421 = stablehlo.multiply %v5420, %v5413 : tensor<96xf32>
    %v5422 = stablehlo.multiply %v5421, %s0b2nbt : tensor<96xf32>
    %v5423 = stablehlo.subtract %v5419, %v5422 : tensor<96xf32>
    %v5424 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5425 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5426 = stablehlo.multiply %v5424, %s0b2eWm : tensor<384x96x1x1xf32>
    %v5427 = stablehlo.multiply %v5425, %v4037 : tensor<384x96x1x1xf32>
    %v5428 = stablehlo.add %v5426, %v5427 : tensor<384x96x1x1xf32>
    %v5429 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5430 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5431 = stablehlo.multiply %v5429, %s0b2eWv : tensor<384x96x1x1xf32>
    %v5432 = stablehlo.multiply %v4037, %v4037 : tensor<384x96x1x1xf32>
    %v5433 = stablehlo.multiply %v5430, %v5432 : tensor<384x96x1x1xf32>
    %v5434 = stablehlo.add %v5431, %v5433 : tensor<384x96x1x1xf32>
    %v5435 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5436 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5437 = stablehlo.multiply %v5435, %s0b2eWm : tensor<384x96x1x1xf32>
    %v5438 = stablehlo.multiply %v5436, %v4037 : tensor<384x96x1x1xf32>
    %v5439 = stablehlo.add %v5437, %v5438 : tensor<384x96x1x1xf32>
    %v5440 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5441 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5442 = stablehlo.multiply %v5440, %s0b2eWv : tensor<384x96x1x1xf32>
    %v5443 = stablehlo.multiply %v4037, %v4037 : tensor<384x96x1x1xf32>
    %v5444 = stablehlo.multiply %v5441, %v5443 : tensor<384x96x1x1xf32>
    %v5445 = stablehlo.add %v5442, %v5444 : tensor<384x96x1x1xf32>
    %v5446 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5447 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5448 = stablehlo.divide %v5439, %v5446 : tensor<384x96x1x1xf32>
    %v5449 = stablehlo.divide %v5445, %v5447 : tensor<384x96x1x1xf32>
    %v5450 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5451 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5452 = stablehlo.sqrt %v5449 : tensor<384x96x1x1xf32>
    %v5453 = stablehlo.add %v5452, %v5451 : tensor<384x96x1x1xf32>
    %v5454 = stablehlo.divide %v5448, %v5453 : tensor<384x96x1x1xf32>
    %v5455 = stablehlo.multiply %v5450, %v5454 : tensor<384x96x1x1xf32>
    %v5456 = stablehlo.subtract %s0b2eW, %v5455 : tensor<384x96x1x1xf32>
    %v5457 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x96x1x1xf32>
    %v5458 = stablehlo.multiply %v5457, %v5450 : tensor<384x96x1x1xf32>
    %v5459 = stablehlo.multiply %v5458, %s0b2eW : tensor<384x96x1x1xf32>
    %v5460 = stablehlo.subtract %v5456, %v5459 : tensor<384x96x1x1xf32>
    %v5461 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5462 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5463 = stablehlo.multiply %v5461, %s0b2ebm : tensor<384xf32>
    %v5464 = stablehlo.multiply %v5462, %v4040 : tensor<384xf32>
    %v5465 = stablehlo.add %v5463, %v5464 : tensor<384xf32>
    %v5466 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5467 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5468 = stablehlo.multiply %v5466, %s0b2ebv : tensor<384xf32>
    %v5469 = stablehlo.multiply %v4040, %v4040 : tensor<384xf32>
    %v5470 = stablehlo.multiply %v5467, %v5469 : tensor<384xf32>
    %v5471 = stablehlo.add %v5468, %v5470 : tensor<384xf32>
    %v5472 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5473 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5474 = stablehlo.multiply %v5472, %s0b2ebm : tensor<384xf32>
    %v5475 = stablehlo.multiply %v5473, %v4040 : tensor<384xf32>
    %v5476 = stablehlo.add %v5474, %v5475 : tensor<384xf32>
    %v5477 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5478 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5479 = stablehlo.multiply %v5477, %s0b2ebv : tensor<384xf32>
    %v5480 = stablehlo.multiply %v4040, %v4040 : tensor<384xf32>
    %v5481 = stablehlo.multiply %v5478, %v5480 : tensor<384xf32>
    %v5482 = stablehlo.add %v5479, %v5481 : tensor<384xf32>
    %v5483 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5484 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5485 = stablehlo.divide %v5476, %v5483 : tensor<384xf32>
    %v5486 = stablehlo.divide %v5482, %v5484 : tensor<384xf32>
    %v5487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5488 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5489 = stablehlo.sqrt %v5486 : tensor<384xf32>
    %v5490 = stablehlo.add %v5489, %v5488 : tensor<384xf32>
    %v5491 = stablehlo.divide %v5485, %v5490 : tensor<384xf32>
    %v5492 = stablehlo.multiply %v5487, %v5491 : tensor<384xf32>
    %v5493 = stablehlo.subtract %s0b2eb, %v5492 : tensor<384xf32>
    %v5494 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v5495 = stablehlo.multiply %v5494, %v5487 : tensor<384xf32>
    %v5496 = stablehlo.multiply %v5495, %s0b2eb : tensor<384xf32>
    %v5497 = stablehlo.subtract %v5493, %v5496 : tensor<384xf32>
    %v5498 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5499 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5500 = stablehlo.multiply %v5498, %s0b2pWm : tensor<96x384x1x1xf32>
    %v5501 = stablehlo.multiply %v5499, %v4028 : tensor<96x384x1x1xf32>
    %v5502 = stablehlo.add %v5500, %v5501 : tensor<96x384x1x1xf32>
    %v5503 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5504 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5505 = stablehlo.multiply %v5503, %s0b2pWv : tensor<96x384x1x1xf32>
    %v5506 = stablehlo.multiply %v4028, %v4028 : tensor<96x384x1x1xf32>
    %v5507 = stablehlo.multiply %v5504, %v5506 : tensor<96x384x1x1xf32>
    %v5508 = stablehlo.add %v5505, %v5507 : tensor<96x384x1x1xf32>
    %v5509 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5510 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5511 = stablehlo.multiply %v5509, %s0b2pWm : tensor<96x384x1x1xf32>
    %v5512 = stablehlo.multiply %v5510, %v4028 : tensor<96x384x1x1xf32>
    %v5513 = stablehlo.add %v5511, %v5512 : tensor<96x384x1x1xf32>
    %v5514 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5515 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5516 = stablehlo.multiply %v5514, %s0b2pWv : tensor<96x384x1x1xf32>
    %v5517 = stablehlo.multiply %v4028, %v4028 : tensor<96x384x1x1xf32>
    %v5518 = stablehlo.multiply %v5515, %v5517 : tensor<96x384x1x1xf32>
    %v5519 = stablehlo.add %v5516, %v5518 : tensor<96x384x1x1xf32>
    %v5520 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5521 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5522 = stablehlo.divide %v5513, %v5520 : tensor<96x384x1x1xf32>
    %v5523 = stablehlo.divide %v5519, %v5521 : tensor<96x384x1x1xf32>
    %v5524 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5525 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5526 = stablehlo.sqrt %v5523 : tensor<96x384x1x1xf32>
    %v5527 = stablehlo.add %v5526, %v5525 : tensor<96x384x1x1xf32>
    %v5528 = stablehlo.divide %v5522, %v5527 : tensor<96x384x1x1xf32>
    %v5529 = stablehlo.multiply %v5524, %v5528 : tensor<96x384x1x1xf32>
    %v5530 = stablehlo.subtract %s0b2pW, %v5529 : tensor<96x384x1x1xf32>
    %v5531 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96x384x1x1xf32>
    %v5532 = stablehlo.multiply %v5531, %v5524 : tensor<96x384x1x1xf32>
    %v5533 = stablehlo.multiply %v5532, %s0b2pW : tensor<96x384x1x1xf32>
    %v5534 = stablehlo.subtract %v5530, %v5533 : tensor<96x384x1x1xf32>
    %v5535 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5536 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5537 = stablehlo.multiply %v5535, %s0b2pbm : tensor<96xf32>
    %v5538 = stablehlo.multiply %v5536, %v4031 : tensor<96xf32>
    %v5539 = stablehlo.add %v5537, %v5538 : tensor<96xf32>
    %v5540 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5541 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5542 = stablehlo.multiply %v5540, %s0b2pbv : tensor<96xf32>
    %v5543 = stablehlo.multiply %v4031, %v4031 : tensor<96xf32>
    %v5544 = stablehlo.multiply %v5541, %v5543 : tensor<96xf32>
    %v5545 = stablehlo.add %v5542, %v5544 : tensor<96xf32>
    %v5546 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5547 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5548 = stablehlo.multiply %v5546, %s0b2pbm : tensor<96xf32>
    %v5549 = stablehlo.multiply %v5547, %v4031 : tensor<96xf32>
    %v5550 = stablehlo.add %v5548, %v5549 : tensor<96xf32>
    %v5551 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5552 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5553 = stablehlo.multiply %v5551, %s0b2pbv : tensor<96xf32>
    %v5554 = stablehlo.multiply %v4031, %v4031 : tensor<96xf32>
    %v5555 = stablehlo.multiply %v5552, %v5554 : tensor<96xf32>
    %v5556 = stablehlo.add %v5553, %v5555 : tensor<96xf32>
    %v5557 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5558 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5559 = stablehlo.divide %v5550, %v5557 : tensor<96xf32>
    %v5560 = stablehlo.divide %v5556, %v5558 : tensor<96xf32>
    %v5561 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5562 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5563 = stablehlo.sqrt %v5560 : tensor<96xf32>
    %v5564 = stablehlo.add %v5563, %v5562 : tensor<96xf32>
    %v5565 = stablehlo.divide %v5559, %v5564 : tensor<96xf32>
    %v5566 = stablehlo.multiply %v5561, %v5565 : tensor<96xf32>
    %v5567 = stablehlo.subtract %s0b2pb, %v5566 : tensor<96xf32>
    %v5568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5569 = stablehlo.multiply %v5568, %v5561 : tensor<96xf32>
    %v5570 = stablehlo.multiply %v5569, %s0b2pb : tensor<96xf32>
    %v5571 = stablehlo.subtract %v5567, %v5570 : tensor<96xf32>
    %v5572 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5573 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5574 = stablehlo.multiply %v5572, %s0b2lgm : tensor<96xf32>
    %v5575 = stablehlo.multiply %v5573, %v4022 : tensor<96xf32>
    %v5576 = stablehlo.add %v5574, %v5575 : tensor<96xf32>
    %v5577 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5578 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5579 = stablehlo.multiply %v5577, %s0b2lgv : tensor<96xf32>
    %v5580 = stablehlo.multiply %v4022, %v4022 : tensor<96xf32>
    %v5581 = stablehlo.multiply %v5578, %v5580 : tensor<96xf32>
    %v5582 = stablehlo.add %v5579, %v5581 : tensor<96xf32>
    %v5583 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5584 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5585 = stablehlo.multiply %v5583, %s0b2lgm : tensor<96xf32>
    %v5586 = stablehlo.multiply %v5584, %v4022 : tensor<96xf32>
    %v5587 = stablehlo.add %v5585, %v5586 : tensor<96xf32>
    %v5588 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5589 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5590 = stablehlo.multiply %v5588, %s0b2lgv : tensor<96xf32>
    %v5591 = stablehlo.multiply %v4022, %v4022 : tensor<96xf32>
    %v5592 = stablehlo.multiply %v5589, %v5591 : tensor<96xf32>
    %v5593 = stablehlo.add %v5590, %v5592 : tensor<96xf32>
    %v5594 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5595 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5596 = stablehlo.divide %v5587, %v5594 : tensor<96xf32>
    %v5597 = stablehlo.divide %v5593, %v5595 : tensor<96xf32>
    %v5598 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5599 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5600 = stablehlo.sqrt %v5597 : tensor<96xf32>
    %v5601 = stablehlo.add %v5600, %v5599 : tensor<96xf32>
    %v5602 = stablehlo.divide %v5596, %v5601 : tensor<96xf32>
    %v5603 = stablehlo.multiply %v5598, %v5602 : tensor<96xf32>
    %v5604 = stablehlo.subtract %s0b2lg, %v5603 : tensor<96xf32>
    %v5605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5606 = stablehlo.multiply %v5605, %v5598 : tensor<96xf32>
    %v5607 = stablehlo.multiply %v5606, %s0b2lg : tensor<96xf32>
    %v5608 = stablehlo.subtract %v5604, %v5607 : tensor<96xf32>
    %v5609 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5610 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5611 = stablehlo.multiply %v5609, %d0ngm : tensor<96xf32>
    %v5612 = stablehlo.multiply %v5610, %v3916 : tensor<96xf32>
    %v5613 = stablehlo.add %v5611, %v5612 : tensor<96xf32>
    %v5614 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5615 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5616 = stablehlo.multiply %v5614, %d0ngv : tensor<96xf32>
    %v5617 = stablehlo.multiply %v3916, %v3916 : tensor<96xf32>
    %v5618 = stablehlo.multiply %v5615, %v5617 : tensor<96xf32>
    %v5619 = stablehlo.add %v5616, %v5618 : tensor<96xf32>
    %v5620 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5621 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5622 = stablehlo.multiply %v5620, %d0ngm : tensor<96xf32>
    %v5623 = stablehlo.multiply %v5621, %v3916 : tensor<96xf32>
    %v5624 = stablehlo.add %v5622, %v5623 : tensor<96xf32>
    %v5625 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5626 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5627 = stablehlo.multiply %v5625, %d0ngv : tensor<96xf32>
    %v5628 = stablehlo.multiply %v3916, %v3916 : tensor<96xf32>
    %v5629 = stablehlo.multiply %v5626, %v5628 : tensor<96xf32>
    %v5630 = stablehlo.add %v5627, %v5629 : tensor<96xf32>
    %v5631 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5632 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5633 = stablehlo.divide %v5624, %v5631 : tensor<96xf32>
    %v5634 = stablehlo.divide %v5630, %v5632 : tensor<96xf32>
    %v5635 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5636 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5637 = stablehlo.sqrt %v5634 : tensor<96xf32>
    %v5638 = stablehlo.add %v5637, %v5636 : tensor<96xf32>
    %v5639 = stablehlo.divide %v5633, %v5638 : tensor<96xf32>
    %v5640 = stablehlo.multiply %v5635, %v5639 : tensor<96xf32>
    %v5641 = stablehlo.subtract %d0ng, %v5640 : tensor<96xf32>
    %v5642 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5643 = stablehlo.multiply %v5642, %v5635 : tensor<96xf32>
    %v5644 = stablehlo.multiply %v5643, %d0ng : tensor<96xf32>
    %v5645 = stablehlo.subtract %v5641, %v5644 : tensor<96xf32>
    %v5646 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5647 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5648 = stablehlo.multiply %v5646, %d0nbtm : tensor<96xf32>
    %v5649 = stablehlo.multiply %v5647, %v3922 : tensor<96xf32>
    %v5650 = stablehlo.add %v5648, %v5649 : tensor<96xf32>
    %v5651 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5652 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5653 = stablehlo.multiply %v5651, %d0nbtv : tensor<96xf32>
    %v5654 = stablehlo.multiply %v3922, %v3922 : tensor<96xf32>
    %v5655 = stablehlo.multiply %v5652, %v5654 : tensor<96xf32>
    %v5656 = stablehlo.add %v5653, %v5655 : tensor<96xf32>
    %v5657 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5658 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5659 = stablehlo.multiply %v5657, %d0nbtm : tensor<96xf32>
    %v5660 = stablehlo.multiply %v5658, %v3922 : tensor<96xf32>
    %v5661 = stablehlo.add %v5659, %v5660 : tensor<96xf32>
    %v5662 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5663 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5664 = stablehlo.multiply %v5662, %d0nbtv : tensor<96xf32>
    %v5665 = stablehlo.multiply %v3922, %v3922 : tensor<96xf32>
    %v5666 = stablehlo.multiply %v5663, %v5665 : tensor<96xf32>
    %v5667 = stablehlo.add %v5664, %v5666 : tensor<96xf32>
    %v5668 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5669 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5670 = stablehlo.divide %v5661, %v5668 : tensor<96xf32>
    %v5671 = stablehlo.divide %v5667, %v5669 : tensor<96xf32>
    %v5672 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5673 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5674 = stablehlo.sqrt %v5671 : tensor<96xf32>
    %v5675 = stablehlo.add %v5674, %v5673 : tensor<96xf32>
    %v5676 = stablehlo.divide %v5670, %v5675 : tensor<96xf32>
    %v5677 = stablehlo.multiply %v5672, %v5676 : tensor<96xf32>
    %v5678 = stablehlo.subtract %d0nbt, %v5677 : tensor<96xf32>
    %v5679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<96xf32>
    %v5680 = stablehlo.multiply %v5679, %v5672 : tensor<96xf32>
    %v5681 = stablehlo.multiply %v5680, %d0nbt : tensor<96xf32>
    %v5682 = stablehlo.subtract %v5678, %v5681 : tensor<96xf32>
    %v5683 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5684 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5685 = stablehlo.multiply %v5683, %d0Wm : tensor<192x96x2x2xf32>
    %v5686 = stablehlo.multiply %v5684, %v3930 : tensor<192x96x2x2xf32>
    %v5687 = stablehlo.add %v5685, %v5686 : tensor<192x96x2x2xf32>
    %v5688 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5689 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5690 = stablehlo.multiply %v5688, %d0Wv : tensor<192x96x2x2xf32>
    %v5691 = stablehlo.multiply %v3930, %v3930 : tensor<192x96x2x2xf32>
    %v5692 = stablehlo.multiply %v5689, %v5691 : tensor<192x96x2x2xf32>
    %v5693 = stablehlo.add %v5690, %v5692 : tensor<192x96x2x2xf32>
    %v5694 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5695 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5696 = stablehlo.multiply %v5694, %d0Wm : tensor<192x96x2x2xf32>
    %v5697 = stablehlo.multiply %v5695, %v3930 : tensor<192x96x2x2xf32>
    %v5698 = stablehlo.add %v5696, %v5697 : tensor<192x96x2x2xf32>
    %v5699 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5700 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5701 = stablehlo.multiply %v5699, %d0Wv : tensor<192x96x2x2xf32>
    %v5702 = stablehlo.multiply %v3930, %v3930 : tensor<192x96x2x2xf32>
    %v5703 = stablehlo.multiply %v5700, %v5702 : tensor<192x96x2x2xf32>
    %v5704 = stablehlo.add %v5701, %v5703 : tensor<192x96x2x2xf32>
    %v5705 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5706 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5707 = stablehlo.divide %v5698, %v5705 : tensor<192x96x2x2xf32>
    %v5708 = stablehlo.divide %v5704, %v5706 : tensor<192x96x2x2xf32>
    %v5709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5710 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5711 = stablehlo.sqrt %v5708 : tensor<192x96x2x2xf32>
    %v5712 = stablehlo.add %v5711, %v5710 : tensor<192x96x2x2xf32>
    %v5713 = stablehlo.divide %v5707, %v5712 : tensor<192x96x2x2xf32>
    %v5714 = stablehlo.multiply %v5709, %v5713 : tensor<192x96x2x2xf32>
    %v5715 = stablehlo.subtract %d0W, %v5714 : tensor<192x96x2x2xf32>
    %v5716 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x96x2x2xf32>
    %v5717 = stablehlo.multiply %v5716, %v5709 : tensor<192x96x2x2xf32>
    %v5718 = stablehlo.multiply %v5717, %d0W : tensor<192x96x2x2xf32>
    %v5719 = stablehlo.subtract %v5715, %v5718 : tensor<192x96x2x2xf32>
    %v5720 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5721 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5722 = stablehlo.multiply %v5720, %d0bm : tensor<192xf32>
    %v5723 = stablehlo.multiply %v5721, %v3892 : tensor<192xf32>
    %v5724 = stablehlo.add %v5722, %v5723 : tensor<192xf32>
    %v5725 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5726 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5727 = stablehlo.multiply %v5725, %d0bv : tensor<192xf32>
    %v5728 = stablehlo.multiply %v3892, %v3892 : tensor<192xf32>
    %v5729 = stablehlo.multiply %v5726, %v5728 : tensor<192xf32>
    %v5730 = stablehlo.add %v5727, %v5729 : tensor<192xf32>
    %v5731 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5732 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5733 = stablehlo.multiply %v5731, %d0bm : tensor<192xf32>
    %v5734 = stablehlo.multiply %v5732, %v3892 : tensor<192xf32>
    %v5735 = stablehlo.add %v5733, %v5734 : tensor<192xf32>
    %v5736 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5737 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5738 = stablehlo.multiply %v5736, %d0bv : tensor<192xf32>
    %v5739 = stablehlo.multiply %v3892, %v3892 : tensor<192xf32>
    %v5740 = stablehlo.multiply %v5737, %v5739 : tensor<192xf32>
    %v5741 = stablehlo.add %v5738, %v5740 : tensor<192xf32>
    %v5742 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5743 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5744 = stablehlo.divide %v5735, %v5742 : tensor<192xf32>
    %v5745 = stablehlo.divide %v5741, %v5743 : tensor<192xf32>
    %v5746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5747 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5748 = stablehlo.sqrt %v5745 : tensor<192xf32>
    %v5749 = stablehlo.add %v5748, %v5747 : tensor<192xf32>
    %v5750 = stablehlo.divide %v5744, %v5749 : tensor<192xf32>
    %v5751 = stablehlo.multiply %v5746, %v5750 : tensor<192xf32>
    %v5752 = stablehlo.subtract %d0b, %v5751 : tensor<192xf32>
    %v5753 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5754 = stablehlo.multiply %v5753, %v5746 : tensor<192xf32>
    %v5755 = stablehlo.multiply %v5754, %d0b : tensor<192xf32>
    %v5756 = stablehlo.subtract %v5752, %v5755 : tensor<192xf32>
    %v5757 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5758 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5759 = stablehlo.multiply %v5757, %s1b0dWm : tensor<192x1x7x7xf32>
    %v5760 = stablehlo.multiply %v5758, %v3836 : tensor<192x1x7x7xf32>
    %v5761 = stablehlo.add %v5759, %v5760 : tensor<192x1x7x7xf32>
    %v5762 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5763 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5764 = stablehlo.multiply %v5762, %s1b0dWv : tensor<192x1x7x7xf32>
    %v5765 = stablehlo.multiply %v3836, %v3836 : tensor<192x1x7x7xf32>
    %v5766 = stablehlo.multiply %v5763, %v5765 : tensor<192x1x7x7xf32>
    %v5767 = stablehlo.add %v5764, %v5766 : tensor<192x1x7x7xf32>
    %v5768 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5769 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5770 = stablehlo.multiply %v5768, %s1b0dWm : tensor<192x1x7x7xf32>
    %v5771 = stablehlo.multiply %v5769, %v3836 : tensor<192x1x7x7xf32>
    %v5772 = stablehlo.add %v5770, %v5771 : tensor<192x1x7x7xf32>
    %v5773 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5774 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5775 = stablehlo.multiply %v5773, %s1b0dWv : tensor<192x1x7x7xf32>
    %v5776 = stablehlo.multiply %v3836, %v3836 : tensor<192x1x7x7xf32>
    %v5777 = stablehlo.multiply %v5774, %v5776 : tensor<192x1x7x7xf32>
    %v5778 = stablehlo.add %v5775, %v5777 : tensor<192x1x7x7xf32>
    %v5779 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5780 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5781 = stablehlo.divide %v5772, %v5779 : tensor<192x1x7x7xf32>
    %v5782 = stablehlo.divide %v5778, %v5780 : tensor<192x1x7x7xf32>
    %v5783 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5784 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5785 = stablehlo.sqrt %v5782 : tensor<192x1x7x7xf32>
    %v5786 = stablehlo.add %v5785, %v5784 : tensor<192x1x7x7xf32>
    %v5787 = stablehlo.divide %v5781, %v5786 : tensor<192x1x7x7xf32>
    %v5788 = stablehlo.multiply %v5783, %v5787 : tensor<192x1x7x7xf32>
    %v5789 = stablehlo.subtract %s1b0dW, %v5788 : tensor<192x1x7x7xf32>
    %v5790 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v5791 = stablehlo.multiply %v5790, %v5783 : tensor<192x1x7x7xf32>
    %v5792 = stablehlo.multiply %v5791, %s1b0dW : tensor<192x1x7x7xf32>
    %v5793 = stablehlo.subtract %v5789, %v5792 : tensor<192x1x7x7xf32>
    %v5794 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5795 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5796 = stablehlo.multiply %v5794, %s1b0dbm : tensor<192xf32>
    %v5797 = stablehlo.multiply %v5795, %v3839 : tensor<192xf32>
    %v5798 = stablehlo.add %v5796, %v5797 : tensor<192xf32>
    %v5799 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5800 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5801 = stablehlo.multiply %v5799, %s1b0dbv : tensor<192xf32>
    %v5802 = stablehlo.multiply %v3839, %v3839 : tensor<192xf32>
    %v5803 = stablehlo.multiply %v5800, %v5802 : tensor<192xf32>
    %v5804 = stablehlo.add %v5801, %v5803 : tensor<192xf32>
    %v5805 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5806 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5807 = stablehlo.multiply %v5805, %s1b0dbm : tensor<192xf32>
    %v5808 = stablehlo.multiply %v5806, %v3839 : tensor<192xf32>
    %v5809 = stablehlo.add %v5807, %v5808 : tensor<192xf32>
    %v5810 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5811 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5812 = stablehlo.multiply %v5810, %s1b0dbv : tensor<192xf32>
    %v5813 = stablehlo.multiply %v3839, %v3839 : tensor<192xf32>
    %v5814 = stablehlo.multiply %v5811, %v5813 : tensor<192xf32>
    %v5815 = stablehlo.add %v5812, %v5814 : tensor<192xf32>
    %v5816 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5817 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5818 = stablehlo.divide %v5809, %v5816 : tensor<192xf32>
    %v5819 = stablehlo.divide %v5815, %v5817 : tensor<192xf32>
    %v5820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5821 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5822 = stablehlo.sqrt %v5819 : tensor<192xf32>
    %v5823 = stablehlo.add %v5822, %v5821 : tensor<192xf32>
    %v5824 = stablehlo.divide %v5818, %v5823 : tensor<192xf32>
    %v5825 = stablehlo.multiply %v5820, %v5824 : tensor<192xf32>
    %v5826 = stablehlo.subtract %s1b0db, %v5825 : tensor<192xf32>
    %v5827 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5828 = stablehlo.multiply %v5827, %v5820 : tensor<192xf32>
    %v5829 = stablehlo.multiply %v5828, %s1b0db : tensor<192xf32>
    %v5830 = stablehlo.subtract %v5826, %v5829 : tensor<192xf32>
    %v5831 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5832 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5833 = stablehlo.multiply %v5831, %s1b0ngm : tensor<192xf32>
    %v5834 = stablehlo.multiply %v5832, %v3824 : tensor<192xf32>
    %v5835 = stablehlo.add %v5833, %v5834 : tensor<192xf32>
    %v5836 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5837 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5838 = stablehlo.multiply %v5836, %s1b0ngv : tensor<192xf32>
    %v5839 = stablehlo.multiply %v3824, %v3824 : tensor<192xf32>
    %v5840 = stablehlo.multiply %v5837, %v5839 : tensor<192xf32>
    %v5841 = stablehlo.add %v5838, %v5840 : tensor<192xf32>
    %v5842 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5843 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5844 = stablehlo.multiply %v5842, %s1b0ngm : tensor<192xf32>
    %v5845 = stablehlo.multiply %v5843, %v3824 : tensor<192xf32>
    %v5846 = stablehlo.add %v5844, %v5845 : tensor<192xf32>
    %v5847 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5848 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5849 = stablehlo.multiply %v5847, %s1b0ngv : tensor<192xf32>
    %v5850 = stablehlo.multiply %v3824, %v3824 : tensor<192xf32>
    %v5851 = stablehlo.multiply %v5848, %v5850 : tensor<192xf32>
    %v5852 = stablehlo.add %v5849, %v5851 : tensor<192xf32>
    %v5853 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5854 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5855 = stablehlo.divide %v5846, %v5853 : tensor<192xf32>
    %v5856 = stablehlo.divide %v5852, %v5854 : tensor<192xf32>
    %v5857 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5858 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5859 = stablehlo.sqrt %v5856 : tensor<192xf32>
    %v5860 = stablehlo.add %v5859, %v5858 : tensor<192xf32>
    %v5861 = stablehlo.divide %v5855, %v5860 : tensor<192xf32>
    %v5862 = stablehlo.multiply %v5857, %v5861 : tensor<192xf32>
    %v5863 = stablehlo.subtract %s1b0ng, %v5862 : tensor<192xf32>
    %v5864 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5865 = stablehlo.multiply %v5864, %v5857 : tensor<192xf32>
    %v5866 = stablehlo.multiply %v5865, %s1b0ng : tensor<192xf32>
    %v5867 = stablehlo.subtract %v5863, %v5866 : tensor<192xf32>
    %v5868 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5869 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5870 = stablehlo.multiply %v5868, %s1b0nbtm : tensor<192xf32>
    %v5871 = stablehlo.multiply %v5869, %v3830 : tensor<192xf32>
    %v5872 = stablehlo.add %v5870, %v5871 : tensor<192xf32>
    %v5873 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5874 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5875 = stablehlo.multiply %v5873, %s1b0nbtv : tensor<192xf32>
    %v5876 = stablehlo.multiply %v3830, %v3830 : tensor<192xf32>
    %v5877 = stablehlo.multiply %v5874, %v5876 : tensor<192xf32>
    %v5878 = stablehlo.add %v5875, %v5877 : tensor<192xf32>
    %v5879 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5880 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5881 = stablehlo.multiply %v5879, %s1b0nbtm : tensor<192xf32>
    %v5882 = stablehlo.multiply %v5880, %v3830 : tensor<192xf32>
    %v5883 = stablehlo.add %v5881, %v5882 : tensor<192xf32>
    %v5884 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5885 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5886 = stablehlo.multiply %v5884, %s1b0nbtv : tensor<192xf32>
    %v5887 = stablehlo.multiply %v3830, %v3830 : tensor<192xf32>
    %v5888 = stablehlo.multiply %v5885, %v5887 : tensor<192xf32>
    %v5889 = stablehlo.add %v5886, %v5888 : tensor<192xf32>
    %v5890 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5891 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5892 = stablehlo.divide %v5883, %v5890 : tensor<192xf32>
    %v5893 = stablehlo.divide %v5889, %v5891 : tensor<192xf32>
    %v5894 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5895 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5896 = stablehlo.sqrt %v5893 : tensor<192xf32>
    %v5897 = stablehlo.add %v5896, %v5895 : tensor<192xf32>
    %v5898 = stablehlo.divide %v5892, %v5897 : tensor<192xf32>
    %v5899 = stablehlo.multiply %v5894, %v5898 : tensor<192xf32>
    %v5900 = stablehlo.subtract %s1b0nbt, %v5899 : tensor<192xf32>
    %v5901 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v5902 = stablehlo.multiply %v5901, %v5894 : tensor<192xf32>
    %v5903 = stablehlo.multiply %v5902, %s1b0nbt : tensor<192xf32>
    %v5904 = stablehlo.subtract %v5900, %v5903 : tensor<192xf32>
    %v5905 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5906 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5907 = stablehlo.multiply %v5905, %s1b0eWm : tensor<768x192x1x1xf32>
    %v5908 = stablehlo.multiply %v5906, %v3797 : tensor<768x192x1x1xf32>
    %v5909 = stablehlo.add %v5907, %v5908 : tensor<768x192x1x1xf32>
    %v5910 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5911 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5912 = stablehlo.multiply %v5910, %s1b0eWv : tensor<768x192x1x1xf32>
    %v5913 = stablehlo.multiply %v3797, %v3797 : tensor<768x192x1x1xf32>
    %v5914 = stablehlo.multiply %v5911, %v5913 : tensor<768x192x1x1xf32>
    %v5915 = stablehlo.add %v5912, %v5914 : tensor<768x192x1x1xf32>
    %v5916 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5917 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5918 = stablehlo.multiply %v5916, %s1b0eWm : tensor<768x192x1x1xf32>
    %v5919 = stablehlo.multiply %v5917, %v3797 : tensor<768x192x1x1xf32>
    %v5920 = stablehlo.add %v5918, %v5919 : tensor<768x192x1x1xf32>
    %v5921 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5922 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5923 = stablehlo.multiply %v5921, %s1b0eWv : tensor<768x192x1x1xf32>
    %v5924 = stablehlo.multiply %v3797, %v3797 : tensor<768x192x1x1xf32>
    %v5925 = stablehlo.multiply %v5922, %v5924 : tensor<768x192x1x1xf32>
    %v5926 = stablehlo.add %v5923, %v5925 : tensor<768x192x1x1xf32>
    %v5927 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5928 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5929 = stablehlo.divide %v5920, %v5927 : tensor<768x192x1x1xf32>
    %v5930 = stablehlo.divide %v5926, %v5928 : tensor<768x192x1x1xf32>
    %v5931 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5932 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5933 = stablehlo.sqrt %v5930 : tensor<768x192x1x1xf32>
    %v5934 = stablehlo.add %v5933, %v5932 : tensor<768x192x1x1xf32>
    %v5935 = stablehlo.divide %v5929, %v5934 : tensor<768x192x1x1xf32>
    %v5936 = stablehlo.multiply %v5931, %v5935 : tensor<768x192x1x1xf32>
    %v5937 = stablehlo.subtract %s1b0eW, %v5936 : tensor<768x192x1x1xf32>
    %v5938 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v5939 = stablehlo.multiply %v5938, %v5931 : tensor<768x192x1x1xf32>
    %v5940 = stablehlo.multiply %v5939, %s1b0eW : tensor<768x192x1x1xf32>
    %v5941 = stablehlo.subtract %v5937, %v5940 : tensor<768x192x1x1xf32>
    %v5942 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5943 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5944 = stablehlo.multiply %v5942, %s1b0ebm : tensor<768xf32>
    %v5945 = stablehlo.multiply %v5943, %v3800 : tensor<768xf32>
    %v5946 = stablehlo.add %v5944, %v5945 : tensor<768xf32>
    %v5947 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5948 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5949 = stablehlo.multiply %v5947, %s1b0ebv : tensor<768xf32>
    %v5950 = stablehlo.multiply %v3800, %v3800 : tensor<768xf32>
    %v5951 = stablehlo.multiply %v5948, %v5950 : tensor<768xf32>
    %v5952 = stablehlo.add %v5949, %v5951 : tensor<768xf32>
    %v5953 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5954 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5955 = stablehlo.multiply %v5953, %s1b0ebm : tensor<768xf32>
    %v5956 = stablehlo.multiply %v5954, %v3800 : tensor<768xf32>
    %v5957 = stablehlo.add %v5955, %v5956 : tensor<768xf32>
    %v5958 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5959 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5960 = stablehlo.multiply %v5958, %s1b0ebv : tensor<768xf32>
    %v5961 = stablehlo.multiply %v3800, %v3800 : tensor<768xf32>
    %v5962 = stablehlo.multiply %v5959, %v5961 : tensor<768xf32>
    %v5963 = stablehlo.add %v5960, %v5962 : tensor<768xf32>
    %v5964 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5965 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5966 = stablehlo.divide %v5957, %v5964 : tensor<768xf32>
    %v5967 = stablehlo.divide %v5963, %v5965 : tensor<768xf32>
    %v5968 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5969 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5970 = stablehlo.sqrt %v5967 : tensor<768xf32>
    %v5971 = stablehlo.add %v5970, %v5969 : tensor<768xf32>
    %v5972 = stablehlo.divide %v5966, %v5971 : tensor<768xf32>
    %v5973 = stablehlo.multiply %v5968, %v5972 : tensor<768xf32>
    %v5974 = stablehlo.subtract %s1b0eb, %v5973 : tensor<768xf32>
    %v5975 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v5976 = stablehlo.multiply %v5975, %v5968 : tensor<768xf32>
    %v5977 = stablehlo.multiply %v5976, %s1b0eb : tensor<768xf32>
    %v5978 = stablehlo.subtract %v5974, %v5977 : tensor<768xf32>
    %v5979 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5980 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5981 = stablehlo.multiply %v5979, %s1b0pWm : tensor<192x768x1x1xf32>
    %v5982 = stablehlo.multiply %v5980, %v3788 : tensor<192x768x1x1xf32>
    %v5983 = stablehlo.add %v5981, %v5982 : tensor<192x768x1x1xf32>
    %v5984 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5985 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5986 = stablehlo.multiply %v5984, %s1b0pWv : tensor<192x768x1x1xf32>
    %v5987 = stablehlo.multiply %v3788, %v3788 : tensor<192x768x1x1xf32>
    %v5988 = stablehlo.multiply %v5985, %v5987 : tensor<192x768x1x1xf32>
    %v5989 = stablehlo.add %v5986, %v5988 : tensor<192x768x1x1xf32>
    %v5990 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5991 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5992 = stablehlo.multiply %v5990, %s1b0pWm : tensor<192x768x1x1xf32>
    %v5993 = stablehlo.multiply %v5991, %v3788 : tensor<192x768x1x1xf32>
    %v5994 = stablehlo.add %v5992, %v5993 : tensor<192x768x1x1xf32>
    %v5995 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5996 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v5997 = stablehlo.multiply %v5995, %s1b0pWv : tensor<192x768x1x1xf32>
    %v5998 = stablehlo.multiply %v3788, %v3788 : tensor<192x768x1x1xf32>
    %v5999 = stablehlo.multiply %v5996, %v5998 : tensor<192x768x1x1xf32>
    %v6000 = stablehlo.add %v5997, %v5999 : tensor<192x768x1x1xf32>
    %v6001 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6002 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6003 = stablehlo.divide %v5994, %v6001 : tensor<192x768x1x1xf32>
    %v6004 = stablehlo.divide %v6000, %v6002 : tensor<192x768x1x1xf32>
    %v6005 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6006 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6007 = stablehlo.sqrt %v6004 : tensor<192x768x1x1xf32>
    %v6008 = stablehlo.add %v6007, %v6006 : tensor<192x768x1x1xf32>
    %v6009 = stablehlo.divide %v6003, %v6008 : tensor<192x768x1x1xf32>
    %v6010 = stablehlo.multiply %v6005, %v6009 : tensor<192x768x1x1xf32>
    %v6011 = stablehlo.subtract %s1b0pW, %v6010 : tensor<192x768x1x1xf32>
    %v6012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6013 = stablehlo.multiply %v6012, %v6005 : tensor<192x768x1x1xf32>
    %v6014 = stablehlo.multiply %v6013, %s1b0pW : tensor<192x768x1x1xf32>
    %v6015 = stablehlo.subtract %v6011, %v6014 : tensor<192x768x1x1xf32>
    %v6016 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6017 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6018 = stablehlo.multiply %v6016, %s1b0pbm : tensor<192xf32>
    %v6019 = stablehlo.multiply %v6017, %v3791 : tensor<192xf32>
    %v6020 = stablehlo.add %v6018, %v6019 : tensor<192xf32>
    %v6021 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6022 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6023 = stablehlo.multiply %v6021, %s1b0pbv : tensor<192xf32>
    %v6024 = stablehlo.multiply %v3791, %v3791 : tensor<192xf32>
    %v6025 = stablehlo.multiply %v6022, %v6024 : tensor<192xf32>
    %v6026 = stablehlo.add %v6023, %v6025 : tensor<192xf32>
    %v6027 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6028 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6029 = stablehlo.multiply %v6027, %s1b0pbm : tensor<192xf32>
    %v6030 = stablehlo.multiply %v6028, %v3791 : tensor<192xf32>
    %v6031 = stablehlo.add %v6029, %v6030 : tensor<192xf32>
    %v6032 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6033 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6034 = stablehlo.multiply %v6032, %s1b0pbv : tensor<192xf32>
    %v6035 = stablehlo.multiply %v3791, %v3791 : tensor<192xf32>
    %v6036 = stablehlo.multiply %v6033, %v6035 : tensor<192xf32>
    %v6037 = stablehlo.add %v6034, %v6036 : tensor<192xf32>
    %v6038 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6039 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6040 = stablehlo.divide %v6031, %v6038 : tensor<192xf32>
    %v6041 = stablehlo.divide %v6037, %v6039 : tensor<192xf32>
    %v6042 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6043 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6044 = stablehlo.sqrt %v6041 : tensor<192xf32>
    %v6045 = stablehlo.add %v6044, %v6043 : tensor<192xf32>
    %v6046 = stablehlo.divide %v6040, %v6045 : tensor<192xf32>
    %v6047 = stablehlo.multiply %v6042, %v6046 : tensor<192xf32>
    %v6048 = stablehlo.subtract %s1b0pb, %v6047 : tensor<192xf32>
    %v6049 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6050 = stablehlo.multiply %v6049, %v6042 : tensor<192xf32>
    %v6051 = stablehlo.multiply %v6050, %s1b0pb : tensor<192xf32>
    %v6052 = stablehlo.subtract %v6048, %v6051 : tensor<192xf32>
    %v6053 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6054 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6055 = stablehlo.multiply %v6053, %s1b0lgm : tensor<192xf32>
    %v6056 = stablehlo.multiply %v6054, %v3782 : tensor<192xf32>
    %v6057 = stablehlo.add %v6055, %v6056 : tensor<192xf32>
    %v6058 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6059 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6060 = stablehlo.multiply %v6058, %s1b0lgv : tensor<192xf32>
    %v6061 = stablehlo.multiply %v3782, %v3782 : tensor<192xf32>
    %v6062 = stablehlo.multiply %v6059, %v6061 : tensor<192xf32>
    %v6063 = stablehlo.add %v6060, %v6062 : tensor<192xf32>
    %v6064 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6065 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6066 = stablehlo.multiply %v6064, %s1b0lgm : tensor<192xf32>
    %v6067 = stablehlo.multiply %v6065, %v3782 : tensor<192xf32>
    %v6068 = stablehlo.add %v6066, %v6067 : tensor<192xf32>
    %v6069 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6070 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6071 = stablehlo.multiply %v6069, %s1b0lgv : tensor<192xf32>
    %v6072 = stablehlo.multiply %v3782, %v3782 : tensor<192xf32>
    %v6073 = stablehlo.multiply %v6070, %v6072 : tensor<192xf32>
    %v6074 = stablehlo.add %v6071, %v6073 : tensor<192xf32>
    %v6075 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6076 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6077 = stablehlo.divide %v6068, %v6075 : tensor<192xf32>
    %v6078 = stablehlo.divide %v6074, %v6076 : tensor<192xf32>
    %v6079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6080 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6081 = stablehlo.sqrt %v6078 : tensor<192xf32>
    %v6082 = stablehlo.add %v6081, %v6080 : tensor<192xf32>
    %v6083 = stablehlo.divide %v6077, %v6082 : tensor<192xf32>
    %v6084 = stablehlo.multiply %v6079, %v6083 : tensor<192xf32>
    %v6085 = stablehlo.subtract %s1b0lg, %v6084 : tensor<192xf32>
    %v6086 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6087 = stablehlo.multiply %v6086, %v6079 : tensor<192xf32>
    %v6088 = stablehlo.multiply %v6087, %s1b0lg : tensor<192xf32>
    %v6089 = stablehlo.subtract %v6085, %v6088 : tensor<192xf32>
    %v6090 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6091 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6092 = stablehlo.multiply %v6090, %s1b1dWm : tensor<192x1x7x7xf32>
    %v6093 = stablehlo.multiply %v6091, %v3687 : tensor<192x1x7x7xf32>
    %v6094 = stablehlo.add %v6092, %v6093 : tensor<192x1x7x7xf32>
    %v6095 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6096 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6097 = stablehlo.multiply %v6095, %s1b1dWv : tensor<192x1x7x7xf32>
    %v6098 = stablehlo.multiply %v3687, %v3687 : tensor<192x1x7x7xf32>
    %v6099 = stablehlo.multiply %v6096, %v6098 : tensor<192x1x7x7xf32>
    %v6100 = stablehlo.add %v6097, %v6099 : tensor<192x1x7x7xf32>
    %v6101 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6102 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6103 = stablehlo.multiply %v6101, %s1b1dWm : tensor<192x1x7x7xf32>
    %v6104 = stablehlo.multiply %v6102, %v3687 : tensor<192x1x7x7xf32>
    %v6105 = stablehlo.add %v6103, %v6104 : tensor<192x1x7x7xf32>
    %v6106 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6107 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6108 = stablehlo.multiply %v6106, %s1b1dWv : tensor<192x1x7x7xf32>
    %v6109 = stablehlo.multiply %v3687, %v3687 : tensor<192x1x7x7xf32>
    %v6110 = stablehlo.multiply %v6107, %v6109 : tensor<192x1x7x7xf32>
    %v6111 = stablehlo.add %v6108, %v6110 : tensor<192x1x7x7xf32>
    %v6112 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6113 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6114 = stablehlo.divide %v6105, %v6112 : tensor<192x1x7x7xf32>
    %v6115 = stablehlo.divide %v6111, %v6113 : tensor<192x1x7x7xf32>
    %v6116 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6117 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6118 = stablehlo.sqrt %v6115 : tensor<192x1x7x7xf32>
    %v6119 = stablehlo.add %v6118, %v6117 : tensor<192x1x7x7xf32>
    %v6120 = stablehlo.divide %v6114, %v6119 : tensor<192x1x7x7xf32>
    %v6121 = stablehlo.multiply %v6116, %v6120 : tensor<192x1x7x7xf32>
    %v6122 = stablehlo.subtract %s1b1dW, %v6121 : tensor<192x1x7x7xf32>
    %v6123 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6124 = stablehlo.multiply %v6123, %v6116 : tensor<192x1x7x7xf32>
    %v6125 = stablehlo.multiply %v6124, %s1b1dW : tensor<192x1x7x7xf32>
    %v6126 = stablehlo.subtract %v6122, %v6125 : tensor<192x1x7x7xf32>
    %v6127 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6128 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6129 = stablehlo.multiply %v6127, %s1b1dbm : tensor<192xf32>
    %v6130 = stablehlo.multiply %v6128, %v3690 : tensor<192xf32>
    %v6131 = stablehlo.add %v6129, %v6130 : tensor<192xf32>
    %v6132 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6133 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6134 = stablehlo.multiply %v6132, %s1b1dbv : tensor<192xf32>
    %v6135 = stablehlo.multiply %v3690, %v3690 : tensor<192xf32>
    %v6136 = stablehlo.multiply %v6133, %v6135 : tensor<192xf32>
    %v6137 = stablehlo.add %v6134, %v6136 : tensor<192xf32>
    %v6138 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6139 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6140 = stablehlo.multiply %v6138, %s1b1dbm : tensor<192xf32>
    %v6141 = stablehlo.multiply %v6139, %v3690 : tensor<192xf32>
    %v6142 = stablehlo.add %v6140, %v6141 : tensor<192xf32>
    %v6143 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6144 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6145 = stablehlo.multiply %v6143, %s1b1dbv : tensor<192xf32>
    %v6146 = stablehlo.multiply %v3690, %v3690 : tensor<192xf32>
    %v6147 = stablehlo.multiply %v6144, %v6146 : tensor<192xf32>
    %v6148 = stablehlo.add %v6145, %v6147 : tensor<192xf32>
    %v6149 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6150 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6151 = stablehlo.divide %v6142, %v6149 : tensor<192xf32>
    %v6152 = stablehlo.divide %v6148, %v6150 : tensor<192xf32>
    %v6153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6154 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6155 = stablehlo.sqrt %v6152 : tensor<192xf32>
    %v6156 = stablehlo.add %v6155, %v6154 : tensor<192xf32>
    %v6157 = stablehlo.divide %v6151, %v6156 : tensor<192xf32>
    %v6158 = stablehlo.multiply %v6153, %v6157 : tensor<192xf32>
    %v6159 = stablehlo.subtract %s1b1db, %v6158 : tensor<192xf32>
    %v6160 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6161 = stablehlo.multiply %v6160, %v6153 : tensor<192xf32>
    %v6162 = stablehlo.multiply %v6161, %s1b1db : tensor<192xf32>
    %v6163 = stablehlo.subtract %v6159, %v6162 : tensor<192xf32>
    %v6164 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6165 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6166 = stablehlo.multiply %v6164, %s1b1ngm : tensor<192xf32>
    %v6167 = stablehlo.multiply %v6165, %v3675 : tensor<192xf32>
    %v6168 = stablehlo.add %v6166, %v6167 : tensor<192xf32>
    %v6169 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6170 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6171 = stablehlo.multiply %v6169, %s1b1ngv : tensor<192xf32>
    %v6172 = stablehlo.multiply %v3675, %v3675 : tensor<192xf32>
    %v6173 = stablehlo.multiply %v6170, %v6172 : tensor<192xf32>
    %v6174 = stablehlo.add %v6171, %v6173 : tensor<192xf32>
    %v6175 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6176 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6177 = stablehlo.multiply %v6175, %s1b1ngm : tensor<192xf32>
    %v6178 = stablehlo.multiply %v6176, %v3675 : tensor<192xf32>
    %v6179 = stablehlo.add %v6177, %v6178 : tensor<192xf32>
    %v6180 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6181 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6182 = stablehlo.multiply %v6180, %s1b1ngv : tensor<192xf32>
    %v6183 = stablehlo.multiply %v3675, %v3675 : tensor<192xf32>
    %v6184 = stablehlo.multiply %v6181, %v6183 : tensor<192xf32>
    %v6185 = stablehlo.add %v6182, %v6184 : tensor<192xf32>
    %v6186 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6187 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6188 = stablehlo.divide %v6179, %v6186 : tensor<192xf32>
    %v6189 = stablehlo.divide %v6185, %v6187 : tensor<192xf32>
    %v6190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6191 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6192 = stablehlo.sqrt %v6189 : tensor<192xf32>
    %v6193 = stablehlo.add %v6192, %v6191 : tensor<192xf32>
    %v6194 = stablehlo.divide %v6188, %v6193 : tensor<192xf32>
    %v6195 = stablehlo.multiply %v6190, %v6194 : tensor<192xf32>
    %v6196 = stablehlo.subtract %s1b1ng, %v6195 : tensor<192xf32>
    %v6197 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6198 = stablehlo.multiply %v6197, %v6190 : tensor<192xf32>
    %v6199 = stablehlo.multiply %v6198, %s1b1ng : tensor<192xf32>
    %v6200 = stablehlo.subtract %v6196, %v6199 : tensor<192xf32>
    %v6201 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6202 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6203 = stablehlo.multiply %v6201, %s1b1nbtm : tensor<192xf32>
    %v6204 = stablehlo.multiply %v6202, %v3681 : tensor<192xf32>
    %v6205 = stablehlo.add %v6203, %v6204 : tensor<192xf32>
    %v6206 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6207 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6208 = stablehlo.multiply %v6206, %s1b1nbtv : tensor<192xf32>
    %v6209 = stablehlo.multiply %v3681, %v3681 : tensor<192xf32>
    %v6210 = stablehlo.multiply %v6207, %v6209 : tensor<192xf32>
    %v6211 = stablehlo.add %v6208, %v6210 : tensor<192xf32>
    %v6212 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6213 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6214 = stablehlo.multiply %v6212, %s1b1nbtm : tensor<192xf32>
    %v6215 = stablehlo.multiply %v6213, %v3681 : tensor<192xf32>
    %v6216 = stablehlo.add %v6214, %v6215 : tensor<192xf32>
    %v6217 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6218 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6219 = stablehlo.multiply %v6217, %s1b1nbtv : tensor<192xf32>
    %v6220 = stablehlo.multiply %v3681, %v3681 : tensor<192xf32>
    %v6221 = stablehlo.multiply %v6218, %v6220 : tensor<192xf32>
    %v6222 = stablehlo.add %v6219, %v6221 : tensor<192xf32>
    %v6223 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6224 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6225 = stablehlo.divide %v6216, %v6223 : tensor<192xf32>
    %v6226 = stablehlo.divide %v6222, %v6224 : tensor<192xf32>
    %v6227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6228 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6229 = stablehlo.sqrt %v6226 : tensor<192xf32>
    %v6230 = stablehlo.add %v6229, %v6228 : tensor<192xf32>
    %v6231 = stablehlo.divide %v6225, %v6230 : tensor<192xf32>
    %v6232 = stablehlo.multiply %v6227, %v6231 : tensor<192xf32>
    %v6233 = stablehlo.subtract %s1b1nbt, %v6232 : tensor<192xf32>
    %v6234 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6235 = stablehlo.multiply %v6234, %v6227 : tensor<192xf32>
    %v6236 = stablehlo.multiply %v6235, %s1b1nbt : tensor<192xf32>
    %v6237 = stablehlo.subtract %v6233, %v6236 : tensor<192xf32>
    %v6238 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6239 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6240 = stablehlo.multiply %v6238, %s1b1eWm : tensor<768x192x1x1xf32>
    %v6241 = stablehlo.multiply %v6239, %v3648 : tensor<768x192x1x1xf32>
    %v6242 = stablehlo.add %v6240, %v6241 : tensor<768x192x1x1xf32>
    %v6243 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6244 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6245 = stablehlo.multiply %v6243, %s1b1eWv : tensor<768x192x1x1xf32>
    %v6246 = stablehlo.multiply %v3648, %v3648 : tensor<768x192x1x1xf32>
    %v6247 = stablehlo.multiply %v6244, %v6246 : tensor<768x192x1x1xf32>
    %v6248 = stablehlo.add %v6245, %v6247 : tensor<768x192x1x1xf32>
    %v6249 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6250 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6251 = stablehlo.multiply %v6249, %s1b1eWm : tensor<768x192x1x1xf32>
    %v6252 = stablehlo.multiply %v6250, %v3648 : tensor<768x192x1x1xf32>
    %v6253 = stablehlo.add %v6251, %v6252 : tensor<768x192x1x1xf32>
    %v6254 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6255 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6256 = stablehlo.multiply %v6254, %s1b1eWv : tensor<768x192x1x1xf32>
    %v6257 = stablehlo.multiply %v3648, %v3648 : tensor<768x192x1x1xf32>
    %v6258 = stablehlo.multiply %v6255, %v6257 : tensor<768x192x1x1xf32>
    %v6259 = stablehlo.add %v6256, %v6258 : tensor<768x192x1x1xf32>
    %v6260 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6261 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6262 = stablehlo.divide %v6253, %v6260 : tensor<768x192x1x1xf32>
    %v6263 = stablehlo.divide %v6259, %v6261 : tensor<768x192x1x1xf32>
    %v6264 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6265 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6266 = stablehlo.sqrt %v6263 : tensor<768x192x1x1xf32>
    %v6267 = stablehlo.add %v6266, %v6265 : tensor<768x192x1x1xf32>
    %v6268 = stablehlo.divide %v6262, %v6267 : tensor<768x192x1x1xf32>
    %v6269 = stablehlo.multiply %v6264, %v6268 : tensor<768x192x1x1xf32>
    %v6270 = stablehlo.subtract %s1b1eW, %v6269 : tensor<768x192x1x1xf32>
    %v6271 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6272 = stablehlo.multiply %v6271, %v6264 : tensor<768x192x1x1xf32>
    %v6273 = stablehlo.multiply %v6272, %s1b1eW : tensor<768x192x1x1xf32>
    %v6274 = stablehlo.subtract %v6270, %v6273 : tensor<768x192x1x1xf32>
    %v6275 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6276 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6277 = stablehlo.multiply %v6275, %s1b1ebm : tensor<768xf32>
    %v6278 = stablehlo.multiply %v6276, %v3651 : tensor<768xf32>
    %v6279 = stablehlo.add %v6277, %v6278 : tensor<768xf32>
    %v6280 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6281 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6282 = stablehlo.multiply %v6280, %s1b1ebv : tensor<768xf32>
    %v6283 = stablehlo.multiply %v3651, %v3651 : tensor<768xf32>
    %v6284 = stablehlo.multiply %v6281, %v6283 : tensor<768xf32>
    %v6285 = stablehlo.add %v6282, %v6284 : tensor<768xf32>
    %v6286 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6287 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6288 = stablehlo.multiply %v6286, %s1b1ebm : tensor<768xf32>
    %v6289 = stablehlo.multiply %v6287, %v3651 : tensor<768xf32>
    %v6290 = stablehlo.add %v6288, %v6289 : tensor<768xf32>
    %v6291 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6292 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6293 = stablehlo.multiply %v6291, %s1b1ebv : tensor<768xf32>
    %v6294 = stablehlo.multiply %v3651, %v3651 : tensor<768xf32>
    %v6295 = stablehlo.multiply %v6292, %v6294 : tensor<768xf32>
    %v6296 = stablehlo.add %v6293, %v6295 : tensor<768xf32>
    %v6297 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6298 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6299 = stablehlo.divide %v6290, %v6297 : tensor<768xf32>
    %v6300 = stablehlo.divide %v6296, %v6298 : tensor<768xf32>
    %v6301 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6302 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6303 = stablehlo.sqrt %v6300 : tensor<768xf32>
    %v6304 = stablehlo.add %v6303, %v6302 : tensor<768xf32>
    %v6305 = stablehlo.divide %v6299, %v6304 : tensor<768xf32>
    %v6306 = stablehlo.multiply %v6301, %v6305 : tensor<768xf32>
    %v6307 = stablehlo.subtract %s1b1eb, %v6306 : tensor<768xf32>
    %v6308 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6309 = stablehlo.multiply %v6308, %v6301 : tensor<768xf32>
    %v6310 = stablehlo.multiply %v6309, %s1b1eb : tensor<768xf32>
    %v6311 = stablehlo.subtract %v6307, %v6310 : tensor<768xf32>
    %v6312 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6313 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6314 = stablehlo.multiply %v6312, %s1b1pWm : tensor<192x768x1x1xf32>
    %v6315 = stablehlo.multiply %v6313, %v3639 : tensor<192x768x1x1xf32>
    %v6316 = stablehlo.add %v6314, %v6315 : tensor<192x768x1x1xf32>
    %v6317 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6318 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6319 = stablehlo.multiply %v6317, %s1b1pWv : tensor<192x768x1x1xf32>
    %v6320 = stablehlo.multiply %v3639, %v3639 : tensor<192x768x1x1xf32>
    %v6321 = stablehlo.multiply %v6318, %v6320 : tensor<192x768x1x1xf32>
    %v6322 = stablehlo.add %v6319, %v6321 : tensor<192x768x1x1xf32>
    %v6323 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6324 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6325 = stablehlo.multiply %v6323, %s1b1pWm : tensor<192x768x1x1xf32>
    %v6326 = stablehlo.multiply %v6324, %v3639 : tensor<192x768x1x1xf32>
    %v6327 = stablehlo.add %v6325, %v6326 : tensor<192x768x1x1xf32>
    %v6328 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6329 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6330 = stablehlo.multiply %v6328, %s1b1pWv : tensor<192x768x1x1xf32>
    %v6331 = stablehlo.multiply %v3639, %v3639 : tensor<192x768x1x1xf32>
    %v6332 = stablehlo.multiply %v6329, %v6331 : tensor<192x768x1x1xf32>
    %v6333 = stablehlo.add %v6330, %v6332 : tensor<192x768x1x1xf32>
    %v6334 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6335 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6336 = stablehlo.divide %v6327, %v6334 : tensor<192x768x1x1xf32>
    %v6337 = stablehlo.divide %v6333, %v6335 : tensor<192x768x1x1xf32>
    %v6338 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6339 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6340 = stablehlo.sqrt %v6337 : tensor<192x768x1x1xf32>
    %v6341 = stablehlo.add %v6340, %v6339 : tensor<192x768x1x1xf32>
    %v6342 = stablehlo.divide %v6336, %v6341 : tensor<192x768x1x1xf32>
    %v6343 = stablehlo.multiply %v6338, %v6342 : tensor<192x768x1x1xf32>
    %v6344 = stablehlo.subtract %s1b1pW, %v6343 : tensor<192x768x1x1xf32>
    %v6345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6346 = stablehlo.multiply %v6345, %v6338 : tensor<192x768x1x1xf32>
    %v6347 = stablehlo.multiply %v6346, %s1b1pW : tensor<192x768x1x1xf32>
    %v6348 = stablehlo.subtract %v6344, %v6347 : tensor<192x768x1x1xf32>
    %v6349 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6350 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6351 = stablehlo.multiply %v6349, %s1b1pbm : tensor<192xf32>
    %v6352 = stablehlo.multiply %v6350, %v3642 : tensor<192xf32>
    %v6353 = stablehlo.add %v6351, %v6352 : tensor<192xf32>
    %v6354 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6355 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6356 = stablehlo.multiply %v6354, %s1b1pbv : tensor<192xf32>
    %v6357 = stablehlo.multiply %v3642, %v3642 : tensor<192xf32>
    %v6358 = stablehlo.multiply %v6355, %v6357 : tensor<192xf32>
    %v6359 = stablehlo.add %v6356, %v6358 : tensor<192xf32>
    %v6360 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6361 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6362 = stablehlo.multiply %v6360, %s1b1pbm : tensor<192xf32>
    %v6363 = stablehlo.multiply %v6361, %v3642 : tensor<192xf32>
    %v6364 = stablehlo.add %v6362, %v6363 : tensor<192xf32>
    %v6365 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6366 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6367 = stablehlo.multiply %v6365, %s1b1pbv : tensor<192xf32>
    %v6368 = stablehlo.multiply %v3642, %v3642 : tensor<192xf32>
    %v6369 = stablehlo.multiply %v6366, %v6368 : tensor<192xf32>
    %v6370 = stablehlo.add %v6367, %v6369 : tensor<192xf32>
    %v6371 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6372 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6373 = stablehlo.divide %v6364, %v6371 : tensor<192xf32>
    %v6374 = stablehlo.divide %v6370, %v6372 : tensor<192xf32>
    %v6375 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6376 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6377 = stablehlo.sqrt %v6374 : tensor<192xf32>
    %v6378 = stablehlo.add %v6377, %v6376 : tensor<192xf32>
    %v6379 = stablehlo.divide %v6373, %v6378 : tensor<192xf32>
    %v6380 = stablehlo.multiply %v6375, %v6379 : tensor<192xf32>
    %v6381 = stablehlo.subtract %s1b1pb, %v6380 : tensor<192xf32>
    %v6382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6383 = stablehlo.multiply %v6382, %v6375 : tensor<192xf32>
    %v6384 = stablehlo.multiply %v6383, %s1b1pb : tensor<192xf32>
    %v6385 = stablehlo.subtract %v6381, %v6384 : tensor<192xf32>
    %v6386 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6387 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6388 = stablehlo.multiply %v6386, %s1b1lgm : tensor<192xf32>
    %v6389 = stablehlo.multiply %v6387, %v3633 : tensor<192xf32>
    %v6390 = stablehlo.add %v6388, %v6389 : tensor<192xf32>
    %v6391 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6392 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6393 = stablehlo.multiply %v6391, %s1b1lgv : tensor<192xf32>
    %v6394 = stablehlo.multiply %v3633, %v3633 : tensor<192xf32>
    %v6395 = stablehlo.multiply %v6392, %v6394 : tensor<192xf32>
    %v6396 = stablehlo.add %v6393, %v6395 : tensor<192xf32>
    %v6397 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6398 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6399 = stablehlo.multiply %v6397, %s1b1lgm : tensor<192xf32>
    %v6400 = stablehlo.multiply %v6398, %v3633 : tensor<192xf32>
    %v6401 = stablehlo.add %v6399, %v6400 : tensor<192xf32>
    %v6402 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6403 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6404 = stablehlo.multiply %v6402, %s1b1lgv : tensor<192xf32>
    %v6405 = stablehlo.multiply %v3633, %v3633 : tensor<192xf32>
    %v6406 = stablehlo.multiply %v6403, %v6405 : tensor<192xf32>
    %v6407 = stablehlo.add %v6404, %v6406 : tensor<192xf32>
    %v6408 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6409 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6410 = stablehlo.divide %v6401, %v6408 : tensor<192xf32>
    %v6411 = stablehlo.divide %v6407, %v6409 : tensor<192xf32>
    %v6412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6413 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6414 = stablehlo.sqrt %v6411 : tensor<192xf32>
    %v6415 = stablehlo.add %v6414, %v6413 : tensor<192xf32>
    %v6416 = stablehlo.divide %v6410, %v6415 : tensor<192xf32>
    %v6417 = stablehlo.multiply %v6412, %v6416 : tensor<192xf32>
    %v6418 = stablehlo.subtract %s1b1lg, %v6417 : tensor<192xf32>
    %v6419 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6420 = stablehlo.multiply %v6419, %v6412 : tensor<192xf32>
    %v6421 = stablehlo.multiply %v6420, %s1b1lg : tensor<192xf32>
    %v6422 = stablehlo.subtract %v6418, %v6421 : tensor<192xf32>
    %v6423 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6424 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6425 = stablehlo.multiply %v6423, %s1b2dWm : tensor<192x1x7x7xf32>
    %v6426 = stablehlo.multiply %v6424, %v3538 : tensor<192x1x7x7xf32>
    %v6427 = stablehlo.add %v6425, %v6426 : tensor<192x1x7x7xf32>
    %v6428 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6429 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6430 = stablehlo.multiply %v6428, %s1b2dWv : tensor<192x1x7x7xf32>
    %v6431 = stablehlo.multiply %v3538, %v3538 : tensor<192x1x7x7xf32>
    %v6432 = stablehlo.multiply %v6429, %v6431 : tensor<192x1x7x7xf32>
    %v6433 = stablehlo.add %v6430, %v6432 : tensor<192x1x7x7xf32>
    %v6434 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6435 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6436 = stablehlo.multiply %v6434, %s1b2dWm : tensor<192x1x7x7xf32>
    %v6437 = stablehlo.multiply %v6435, %v3538 : tensor<192x1x7x7xf32>
    %v6438 = stablehlo.add %v6436, %v6437 : tensor<192x1x7x7xf32>
    %v6439 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6440 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6441 = stablehlo.multiply %v6439, %s1b2dWv : tensor<192x1x7x7xf32>
    %v6442 = stablehlo.multiply %v3538, %v3538 : tensor<192x1x7x7xf32>
    %v6443 = stablehlo.multiply %v6440, %v6442 : tensor<192x1x7x7xf32>
    %v6444 = stablehlo.add %v6441, %v6443 : tensor<192x1x7x7xf32>
    %v6445 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6446 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6447 = stablehlo.divide %v6438, %v6445 : tensor<192x1x7x7xf32>
    %v6448 = stablehlo.divide %v6444, %v6446 : tensor<192x1x7x7xf32>
    %v6449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6450 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6451 = stablehlo.sqrt %v6448 : tensor<192x1x7x7xf32>
    %v6452 = stablehlo.add %v6451, %v6450 : tensor<192x1x7x7xf32>
    %v6453 = stablehlo.divide %v6447, %v6452 : tensor<192x1x7x7xf32>
    %v6454 = stablehlo.multiply %v6449, %v6453 : tensor<192x1x7x7xf32>
    %v6455 = stablehlo.subtract %s1b2dW, %v6454 : tensor<192x1x7x7xf32>
    %v6456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x1x7x7xf32>
    %v6457 = stablehlo.multiply %v6456, %v6449 : tensor<192x1x7x7xf32>
    %v6458 = stablehlo.multiply %v6457, %s1b2dW : tensor<192x1x7x7xf32>
    %v6459 = stablehlo.subtract %v6455, %v6458 : tensor<192x1x7x7xf32>
    %v6460 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6461 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6462 = stablehlo.multiply %v6460, %s1b2dbm : tensor<192xf32>
    %v6463 = stablehlo.multiply %v6461, %v3541 : tensor<192xf32>
    %v6464 = stablehlo.add %v6462, %v6463 : tensor<192xf32>
    %v6465 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6466 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6467 = stablehlo.multiply %v6465, %s1b2dbv : tensor<192xf32>
    %v6468 = stablehlo.multiply %v3541, %v3541 : tensor<192xf32>
    %v6469 = stablehlo.multiply %v6466, %v6468 : tensor<192xf32>
    %v6470 = stablehlo.add %v6467, %v6469 : tensor<192xf32>
    %v6471 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6472 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6473 = stablehlo.multiply %v6471, %s1b2dbm : tensor<192xf32>
    %v6474 = stablehlo.multiply %v6472, %v3541 : tensor<192xf32>
    %v6475 = stablehlo.add %v6473, %v6474 : tensor<192xf32>
    %v6476 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6477 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6478 = stablehlo.multiply %v6476, %s1b2dbv : tensor<192xf32>
    %v6479 = stablehlo.multiply %v3541, %v3541 : tensor<192xf32>
    %v6480 = stablehlo.multiply %v6477, %v6479 : tensor<192xf32>
    %v6481 = stablehlo.add %v6478, %v6480 : tensor<192xf32>
    %v6482 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6483 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6484 = stablehlo.divide %v6475, %v6482 : tensor<192xf32>
    %v6485 = stablehlo.divide %v6481, %v6483 : tensor<192xf32>
    %v6486 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6487 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6488 = stablehlo.sqrt %v6485 : tensor<192xf32>
    %v6489 = stablehlo.add %v6488, %v6487 : tensor<192xf32>
    %v6490 = stablehlo.divide %v6484, %v6489 : tensor<192xf32>
    %v6491 = stablehlo.multiply %v6486, %v6490 : tensor<192xf32>
    %v6492 = stablehlo.subtract %s1b2db, %v6491 : tensor<192xf32>
    %v6493 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6494 = stablehlo.multiply %v6493, %v6486 : tensor<192xf32>
    %v6495 = stablehlo.multiply %v6494, %s1b2db : tensor<192xf32>
    %v6496 = stablehlo.subtract %v6492, %v6495 : tensor<192xf32>
    %v6497 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6498 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6499 = stablehlo.multiply %v6497, %s1b2ngm : tensor<192xf32>
    %v6500 = stablehlo.multiply %v6498, %v3526 : tensor<192xf32>
    %v6501 = stablehlo.add %v6499, %v6500 : tensor<192xf32>
    %v6502 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6503 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6504 = stablehlo.multiply %v6502, %s1b2ngv : tensor<192xf32>
    %v6505 = stablehlo.multiply %v3526, %v3526 : tensor<192xf32>
    %v6506 = stablehlo.multiply %v6503, %v6505 : tensor<192xf32>
    %v6507 = stablehlo.add %v6504, %v6506 : tensor<192xf32>
    %v6508 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6509 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6510 = stablehlo.multiply %v6508, %s1b2ngm : tensor<192xf32>
    %v6511 = stablehlo.multiply %v6509, %v3526 : tensor<192xf32>
    %v6512 = stablehlo.add %v6510, %v6511 : tensor<192xf32>
    %v6513 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6514 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6515 = stablehlo.multiply %v6513, %s1b2ngv : tensor<192xf32>
    %v6516 = stablehlo.multiply %v3526, %v3526 : tensor<192xf32>
    %v6517 = stablehlo.multiply %v6514, %v6516 : tensor<192xf32>
    %v6518 = stablehlo.add %v6515, %v6517 : tensor<192xf32>
    %v6519 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6520 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6521 = stablehlo.divide %v6512, %v6519 : tensor<192xf32>
    %v6522 = stablehlo.divide %v6518, %v6520 : tensor<192xf32>
    %v6523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6524 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6525 = stablehlo.sqrt %v6522 : tensor<192xf32>
    %v6526 = stablehlo.add %v6525, %v6524 : tensor<192xf32>
    %v6527 = stablehlo.divide %v6521, %v6526 : tensor<192xf32>
    %v6528 = stablehlo.multiply %v6523, %v6527 : tensor<192xf32>
    %v6529 = stablehlo.subtract %s1b2ng, %v6528 : tensor<192xf32>
    %v6530 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6531 = stablehlo.multiply %v6530, %v6523 : tensor<192xf32>
    %v6532 = stablehlo.multiply %v6531, %s1b2ng : tensor<192xf32>
    %v6533 = stablehlo.subtract %v6529, %v6532 : tensor<192xf32>
    %v6534 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6535 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6536 = stablehlo.multiply %v6534, %s1b2nbtm : tensor<192xf32>
    %v6537 = stablehlo.multiply %v6535, %v3532 : tensor<192xf32>
    %v6538 = stablehlo.add %v6536, %v6537 : tensor<192xf32>
    %v6539 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6540 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6541 = stablehlo.multiply %v6539, %s1b2nbtv : tensor<192xf32>
    %v6542 = stablehlo.multiply %v3532, %v3532 : tensor<192xf32>
    %v6543 = stablehlo.multiply %v6540, %v6542 : tensor<192xf32>
    %v6544 = stablehlo.add %v6541, %v6543 : tensor<192xf32>
    %v6545 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6546 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6547 = stablehlo.multiply %v6545, %s1b2nbtm : tensor<192xf32>
    %v6548 = stablehlo.multiply %v6546, %v3532 : tensor<192xf32>
    %v6549 = stablehlo.add %v6547, %v6548 : tensor<192xf32>
    %v6550 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6551 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6552 = stablehlo.multiply %v6550, %s1b2nbtv : tensor<192xf32>
    %v6553 = stablehlo.multiply %v3532, %v3532 : tensor<192xf32>
    %v6554 = stablehlo.multiply %v6551, %v6553 : tensor<192xf32>
    %v6555 = stablehlo.add %v6552, %v6554 : tensor<192xf32>
    %v6556 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6557 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6558 = stablehlo.divide %v6549, %v6556 : tensor<192xf32>
    %v6559 = stablehlo.divide %v6555, %v6557 : tensor<192xf32>
    %v6560 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6561 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6562 = stablehlo.sqrt %v6559 : tensor<192xf32>
    %v6563 = stablehlo.add %v6562, %v6561 : tensor<192xf32>
    %v6564 = stablehlo.divide %v6558, %v6563 : tensor<192xf32>
    %v6565 = stablehlo.multiply %v6560, %v6564 : tensor<192xf32>
    %v6566 = stablehlo.subtract %s1b2nbt, %v6565 : tensor<192xf32>
    %v6567 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6568 = stablehlo.multiply %v6567, %v6560 : tensor<192xf32>
    %v6569 = stablehlo.multiply %v6568, %s1b2nbt : tensor<192xf32>
    %v6570 = stablehlo.subtract %v6566, %v6569 : tensor<192xf32>
    %v6571 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6572 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6573 = stablehlo.multiply %v6571, %s1b2eWm : tensor<768x192x1x1xf32>
    %v6574 = stablehlo.multiply %v6572, %v3499 : tensor<768x192x1x1xf32>
    %v6575 = stablehlo.add %v6573, %v6574 : tensor<768x192x1x1xf32>
    %v6576 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6577 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6578 = stablehlo.multiply %v6576, %s1b2eWv : tensor<768x192x1x1xf32>
    %v6579 = stablehlo.multiply %v3499, %v3499 : tensor<768x192x1x1xf32>
    %v6580 = stablehlo.multiply %v6577, %v6579 : tensor<768x192x1x1xf32>
    %v6581 = stablehlo.add %v6578, %v6580 : tensor<768x192x1x1xf32>
    %v6582 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6583 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6584 = stablehlo.multiply %v6582, %s1b2eWm : tensor<768x192x1x1xf32>
    %v6585 = stablehlo.multiply %v6583, %v3499 : tensor<768x192x1x1xf32>
    %v6586 = stablehlo.add %v6584, %v6585 : tensor<768x192x1x1xf32>
    %v6587 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6588 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6589 = stablehlo.multiply %v6587, %s1b2eWv : tensor<768x192x1x1xf32>
    %v6590 = stablehlo.multiply %v3499, %v3499 : tensor<768x192x1x1xf32>
    %v6591 = stablehlo.multiply %v6588, %v6590 : tensor<768x192x1x1xf32>
    %v6592 = stablehlo.add %v6589, %v6591 : tensor<768x192x1x1xf32>
    %v6593 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6594 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6595 = stablehlo.divide %v6586, %v6593 : tensor<768x192x1x1xf32>
    %v6596 = stablehlo.divide %v6592, %v6594 : tensor<768x192x1x1xf32>
    %v6597 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6598 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6599 = stablehlo.sqrt %v6596 : tensor<768x192x1x1xf32>
    %v6600 = stablehlo.add %v6599, %v6598 : tensor<768x192x1x1xf32>
    %v6601 = stablehlo.divide %v6595, %v6600 : tensor<768x192x1x1xf32>
    %v6602 = stablehlo.multiply %v6597, %v6601 : tensor<768x192x1x1xf32>
    %v6603 = stablehlo.subtract %s1b2eW, %v6602 : tensor<768x192x1x1xf32>
    %v6604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x192x1x1xf32>
    %v6605 = stablehlo.multiply %v6604, %v6597 : tensor<768x192x1x1xf32>
    %v6606 = stablehlo.multiply %v6605, %s1b2eW : tensor<768x192x1x1xf32>
    %v6607 = stablehlo.subtract %v6603, %v6606 : tensor<768x192x1x1xf32>
    %v6608 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6609 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6610 = stablehlo.multiply %v6608, %s1b2ebm : tensor<768xf32>
    %v6611 = stablehlo.multiply %v6609, %v3502 : tensor<768xf32>
    %v6612 = stablehlo.add %v6610, %v6611 : tensor<768xf32>
    %v6613 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6614 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6615 = stablehlo.multiply %v6613, %s1b2ebv : tensor<768xf32>
    %v6616 = stablehlo.multiply %v3502, %v3502 : tensor<768xf32>
    %v6617 = stablehlo.multiply %v6614, %v6616 : tensor<768xf32>
    %v6618 = stablehlo.add %v6615, %v6617 : tensor<768xf32>
    %v6619 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6620 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6621 = stablehlo.multiply %v6619, %s1b2ebm : tensor<768xf32>
    %v6622 = stablehlo.multiply %v6620, %v3502 : tensor<768xf32>
    %v6623 = stablehlo.add %v6621, %v6622 : tensor<768xf32>
    %v6624 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6625 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6626 = stablehlo.multiply %v6624, %s1b2ebv : tensor<768xf32>
    %v6627 = stablehlo.multiply %v3502, %v3502 : tensor<768xf32>
    %v6628 = stablehlo.multiply %v6625, %v6627 : tensor<768xf32>
    %v6629 = stablehlo.add %v6626, %v6628 : tensor<768xf32>
    %v6630 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6631 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6632 = stablehlo.divide %v6623, %v6630 : tensor<768xf32>
    %v6633 = stablehlo.divide %v6629, %v6631 : tensor<768xf32>
    %v6634 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6635 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6636 = stablehlo.sqrt %v6633 : tensor<768xf32>
    %v6637 = stablehlo.add %v6636, %v6635 : tensor<768xf32>
    %v6638 = stablehlo.divide %v6632, %v6637 : tensor<768xf32>
    %v6639 = stablehlo.multiply %v6634, %v6638 : tensor<768xf32>
    %v6640 = stablehlo.subtract %s1b2eb, %v6639 : tensor<768xf32>
    %v6641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v6642 = stablehlo.multiply %v6641, %v6634 : tensor<768xf32>
    %v6643 = stablehlo.multiply %v6642, %s1b2eb : tensor<768xf32>
    %v6644 = stablehlo.subtract %v6640, %v6643 : tensor<768xf32>
    %v6645 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6646 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6647 = stablehlo.multiply %v6645, %s1b2pWm : tensor<192x768x1x1xf32>
    %v6648 = stablehlo.multiply %v6646, %v3490 : tensor<192x768x1x1xf32>
    %v6649 = stablehlo.add %v6647, %v6648 : tensor<192x768x1x1xf32>
    %v6650 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6651 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6652 = stablehlo.multiply %v6650, %s1b2pWv : tensor<192x768x1x1xf32>
    %v6653 = stablehlo.multiply %v3490, %v3490 : tensor<192x768x1x1xf32>
    %v6654 = stablehlo.multiply %v6651, %v6653 : tensor<192x768x1x1xf32>
    %v6655 = stablehlo.add %v6652, %v6654 : tensor<192x768x1x1xf32>
    %v6656 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6657 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6658 = stablehlo.multiply %v6656, %s1b2pWm : tensor<192x768x1x1xf32>
    %v6659 = stablehlo.multiply %v6657, %v3490 : tensor<192x768x1x1xf32>
    %v6660 = stablehlo.add %v6658, %v6659 : tensor<192x768x1x1xf32>
    %v6661 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6662 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6663 = stablehlo.multiply %v6661, %s1b2pWv : tensor<192x768x1x1xf32>
    %v6664 = stablehlo.multiply %v3490, %v3490 : tensor<192x768x1x1xf32>
    %v6665 = stablehlo.multiply %v6662, %v6664 : tensor<192x768x1x1xf32>
    %v6666 = stablehlo.add %v6663, %v6665 : tensor<192x768x1x1xf32>
    %v6667 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6668 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6669 = stablehlo.divide %v6660, %v6667 : tensor<192x768x1x1xf32>
    %v6670 = stablehlo.divide %v6666, %v6668 : tensor<192x768x1x1xf32>
    %v6671 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6672 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6673 = stablehlo.sqrt %v6670 : tensor<192x768x1x1xf32>
    %v6674 = stablehlo.add %v6673, %v6672 : tensor<192x768x1x1xf32>
    %v6675 = stablehlo.divide %v6669, %v6674 : tensor<192x768x1x1xf32>
    %v6676 = stablehlo.multiply %v6671, %v6675 : tensor<192x768x1x1xf32>
    %v6677 = stablehlo.subtract %s1b2pW, %v6676 : tensor<192x768x1x1xf32>
    %v6678 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192x768x1x1xf32>
    %v6679 = stablehlo.multiply %v6678, %v6671 : tensor<192x768x1x1xf32>
    %v6680 = stablehlo.multiply %v6679, %s1b2pW : tensor<192x768x1x1xf32>
    %v6681 = stablehlo.subtract %v6677, %v6680 : tensor<192x768x1x1xf32>
    %v6682 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6683 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6684 = stablehlo.multiply %v6682, %s1b2pbm : tensor<192xf32>
    %v6685 = stablehlo.multiply %v6683, %v3493 : tensor<192xf32>
    %v6686 = stablehlo.add %v6684, %v6685 : tensor<192xf32>
    %v6687 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6688 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6689 = stablehlo.multiply %v6687, %s1b2pbv : tensor<192xf32>
    %v6690 = stablehlo.multiply %v3493, %v3493 : tensor<192xf32>
    %v6691 = stablehlo.multiply %v6688, %v6690 : tensor<192xf32>
    %v6692 = stablehlo.add %v6689, %v6691 : tensor<192xf32>
    %v6693 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6694 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6695 = stablehlo.multiply %v6693, %s1b2pbm : tensor<192xf32>
    %v6696 = stablehlo.multiply %v6694, %v3493 : tensor<192xf32>
    %v6697 = stablehlo.add %v6695, %v6696 : tensor<192xf32>
    %v6698 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6699 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6700 = stablehlo.multiply %v6698, %s1b2pbv : tensor<192xf32>
    %v6701 = stablehlo.multiply %v3493, %v3493 : tensor<192xf32>
    %v6702 = stablehlo.multiply %v6699, %v6701 : tensor<192xf32>
    %v6703 = stablehlo.add %v6700, %v6702 : tensor<192xf32>
    %v6704 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6705 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6706 = stablehlo.divide %v6697, %v6704 : tensor<192xf32>
    %v6707 = stablehlo.divide %v6703, %v6705 : tensor<192xf32>
    %v6708 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6709 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6710 = stablehlo.sqrt %v6707 : tensor<192xf32>
    %v6711 = stablehlo.add %v6710, %v6709 : tensor<192xf32>
    %v6712 = stablehlo.divide %v6706, %v6711 : tensor<192xf32>
    %v6713 = stablehlo.multiply %v6708, %v6712 : tensor<192xf32>
    %v6714 = stablehlo.subtract %s1b2pb, %v6713 : tensor<192xf32>
    %v6715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6716 = stablehlo.multiply %v6715, %v6708 : tensor<192xf32>
    %v6717 = stablehlo.multiply %v6716, %s1b2pb : tensor<192xf32>
    %v6718 = stablehlo.subtract %v6714, %v6717 : tensor<192xf32>
    %v6719 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6720 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6721 = stablehlo.multiply %v6719, %s1b2lgm : tensor<192xf32>
    %v6722 = stablehlo.multiply %v6720, %v3484 : tensor<192xf32>
    %v6723 = stablehlo.add %v6721, %v6722 : tensor<192xf32>
    %v6724 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6725 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6726 = stablehlo.multiply %v6724, %s1b2lgv : tensor<192xf32>
    %v6727 = stablehlo.multiply %v3484, %v3484 : tensor<192xf32>
    %v6728 = stablehlo.multiply %v6725, %v6727 : tensor<192xf32>
    %v6729 = stablehlo.add %v6726, %v6728 : tensor<192xf32>
    %v6730 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6731 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6732 = stablehlo.multiply %v6730, %s1b2lgm : tensor<192xf32>
    %v6733 = stablehlo.multiply %v6731, %v3484 : tensor<192xf32>
    %v6734 = stablehlo.add %v6732, %v6733 : tensor<192xf32>
    %v6735 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6736 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6737 = stablehlo.multiply %v6735, %s1b2lgv : tensor<192xf32>
    %v6738 = stablehlo.multiply %v3484, %v3484 : tensor<192xf32>
    %v6739 = stablehlo.multiply %v6736, %v6738 : tensor<192xf32>
    %v6740 = stablehlo.add %v6737, %v6739 : tensor<192xf32>
    %v6741 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6742 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6743 = stablehlo.divide %v6734, %v6741 : tensor<192xf32>
    %v6744 = stablehlo.divide %v6740, %v6742 : tensor<192xf32>
    %v6745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6746 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6747 = stablehlo.sqrt %v6744 : tensor<192xf32>
    %v6748 = stablehlo.add %v6747, %v6746 : tensor<192xf32>
    %v6749 = stablehlo.divide %v6743, %v6748 : tensor<192xf32>
    %v6750 = stablehlo.multiply %v6745, %v6749 : tensor<192xf32>
    %v6751 = stablehlo.subtract %s1b2lg, %v6750 : tensor<192xf32>
    %v6752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6753 = stablehlo.multiply %v6752, %v6745 : tensor<192xf32>
    %v6754 = stablehlo.multiply %v6753, %s1b2lg : tensor<192xf32>
    %v6755 = stablehlo.subtract %v6751, %v6754 : tensor<192xf32>
    %v6756 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6757 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6758 = stablehlo.multiply %v6756, %d1ngm : tensor<192xf32>
    %v6759 = stablehlo.multiply %v6757, %v3378 : tensor<192xf32>
    %v6760 = stablehlo.add %v6758, %v6759 : tensor<192xf32>
    %v6761 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6762 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6763 = stablehlo.multiply %v6761, %d1ngv : tensor<192xf32>
    %v6764 = stablehlo.multiply %v3378, %v3378 : tensor<192xf32>
    %v6765 = stablehlo.multiply %v6762, %v6764 : tensor<192xf32>
    %v6766 = stablehlo.add %v6763, %v6765 : tensor<192xf32>
    %v6767 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6768 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6769 = stablehlo.multiply %v6767, %d1ngm : tensor<192xf32>
    %v6770 = stablehlo.multiply %v6768, %v3378 : tensor<192xf32>
    %v6771 = stablehlo.add %v6769, %v6770 : tensor<192xf32>
    %v6772 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6773 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6774 = stablehlo.multiply %v6772, %d1ngv : tensor<192xf32>
    %v6775 = stablehlo.multiply %v3378, %v3378 : tensor<192xf32>
    %v6776 = stablehlo.multiply %v6773, %v6775 : tensor<192xf32>
    %v6777 = stablehlo.add %v6774, %v6776 : tensor<192xf32>
    %v6778 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6779 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6780 = stablehlo.divide %v6771, %v6778 : tensor<192xf32>
    %v6781 = stablehlo.divide %v6777, %v6779 : tensor<192xf32>
    %v6782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6783 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6784 = stablehlo.sqrt %v6781 : tensor<192xf32>
    %v6785 = stablehlo.add %v6784, %v6783 : tensor<192xf32>
    %v6786 = stablehlo.divide %v6780, %v6785 : tensor<192xf32>
    %v6787 = stablehlo.multiply %v6782, %v6786 : tensor<192xf32>
    %v6788 = stablehlo.subtract %d1ng, %v6787 : tensor<192xf32>
    %v6789 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6790 = stablehlo.multiply %v6789, %v6782 : tensor<192xf32>
    %v6791 = stablehlo.multiply %v6790, %d1ng : tensor<192xf32>
    %v6792 = stablehlo.subtract %v6788, %v6791 : tensor<192xf32>
    %v6793 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6794 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6795 = stablehlo.multiply %v6793, %d1nbtm : tensor<192xf32>
    %v6796 = stablehlo.multiply %v6794, %v3384 : tensor<192xf32>
    %v6797 = stablehlo.add %v6795, %v6796 : tensor<192xf32>
    %v6798 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6799 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6800 = stablehlo.multiply %v6798, %d1nbtv : tensor<192xf32>
    %v6801 = stablehlo.multiply %v3384, %v3384 : tensor<192xf32>
    %v6802 = stablehlo.multiply %v6799, %v6801 : tensor<192xf32>
    %v6803 = stablehlo.add %v6800, %v6802 : tensor<192xf32>
    %v6804 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6805 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6806 = stablehlo.multiply %v6804, %d1nbtm : tensor<192xf32>
    %v6807 = stablehlo.multiply %v6805, %v3384 : tensor<192xf32>
    %v6808 = stablehlo.add %v6806, %v6807 : tensor<192xf32>
    %v6809 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6810 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6811 = stablehlo.multiply %v6809, %d1nbtv : tensor<192xf32>
    %v6812 = stablehlo.multiply %v3384, %v3384 : tensor<192xf32>
    %v6813 = stablehlo.multiply %v6810, %v6812 : tensor<192xf32>
    %v6814 = stablehlo.add %v6811, %v6813 : tensor<192xf32>
    %v6815 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6816 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6817 = stablehlo.divide %v6808, %v6815 : tensor<192xf32>
    %v6818 = stablehlo.divide %v6814, %v6816 : tensor<192xf32>
    %v6819 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6820 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6821 = stablehlo.sqrt %v6818 : tensor<192xf32>
    %v6822 = stablehlo.add %v6821, %v6820 : tensor<192xf32>
    %v6823 = stablehlo.divide %v6817, %v6822 : tensor<192xf32>
    %v6824 = stablehlo.multiply %v6819, %v6823 : tensor<192xf32>
    %v6825 = stablehlo.subtract %d1nbt, %v6824 : tensor<192xf32>
    %v6826 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<192xf32>
    %v6827 = stablehlo.multiply %v6826, %v6819 : tensor<192xf32>
    %v6828 = stablehlo.multiply %v6827, %d1nbt : tensor<192xf32>
    %v6829 = stablehlo.subtract %v6825, %v6828 : tensor<192xf32>
    %v6830 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6831 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6832 = stablehlo.multiply %v6830, %d1Wm : tensor<384x192x2x2xf32>
    %v6833 = stablehlo.multiply %v6831, %v3392 : tensor<384x192x2x2xf32>
    %v6834 = stablehlo.add %v6832, %v6833 : tensor<384x192x2x2xf32>
    %v6835 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6836 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6837 = stablehlo.multiply %v6835, %d1Wv : tensor<384x192x2x2xf32>
    %v6838 = stablehlo.multiply %v3392, %v3392 : tensor<384x192x2x2xf32>
    %v6839 = stablehlo.multiply %v6836, %v6838 : tensor<384x192x2x2xf32>
    %v6840 = stablehlo.add %v6837, %v6839 : tensor<384x192x2x2xf32>
    %v6841 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6842 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6843 = stablehlo.multiply %v6841, %d1Wm : tensor<384x192x2x2xf32>
    %v6844 = stablehlo.multiply %v6842, %v3392 : tensor<384x192x2x2xf32>
    %v6845 = stablehlo.add %v6843, %v6844 : tensor<384x192x2x2xf32>
    %v6846 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6847 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6848 = stablehlo.multiply %v6846, %d1Wv : tensor<384x192x2x2xf32>
    %v6849 = stablehlo.multiply %v3392, %v3392 : tensor<384x192x2x2xf32>
    %v6850 = stablehlo.multiply %v6847, %v6849 : tensor<384x192x2x2xf32>
    %v6851 = stablehlo.add %v6848, %v6850 : tensor<384x192x2x2xf32>
    %v6852 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6853 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6854 = stablehlo.divide %v6845, %v6852 : tensor<384x192x2x2xf32>
    %v6855 = stablehlo.divide %v6851, %v6853 : tensor<384x192x2x2xf32>
    %v6856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6857 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6858 = stablehlo.sqrt %v6855 : tensor<384x192x2x2xf32>
    %v6859 = stablehlo.add %v6858, %v6857 : tensor<384x192x2x2xf32>
    %v6860 = stablehlo.divide %v6854, %v6859 : tensor<384x192x2x2xf32>
    %v6861 = stablehlo.multiply %v6856, %v6860 : tensor<384x192x2x2xf32>
    %v6862 = stablehlo.subtract %d1W, %v6861 : tensor<384x192x2x2xf32>
    %v6863 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x192x2x2xf32>
    %v6864 = stablehlo.multiply %v6863, %v6856 : tensor<384x192x2x2xf32>
    %v6865 = stablehlo.multiply %v6864, %d1W : tensor<384x192x2x2xf32>
    %v6866 = stablehlo.subtract %v6862, %v6865 : tensor<384x192x2x2xf32>
    %v6867 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6868 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6869 = stablehlo.multiply %v6867, %d1bm : tensor<384xf32>
    %v6870 = stablehlo.multiply %v6868, %v3354 : tensor<384xf32>
    %v6871 = stablehlo.add %v6869, %v6870 : tensor<384xf32>
    %v6872 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6873 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6874 = stablehlo.multiply %v6872, %d1bv : tensor<384xf32>
    %v6875 = stablehlo.multiply %v3354, %v3354 : tensor<384xf32>
    %v6876 = stablehlo.multiply %v6873, %v6875 : tensor<384xf32>
    %v6877 = stablehlo.add %v6874, %v6876 : tensor<384xf32>
    %v6878 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6879 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6880 = stablehlo.multiply %v6878, %d1bm : tensor<384xf32>
    %v6881 = stablehlo.multiply %v6879, %v3354 : tensor<384xf32>
    %v6882 = stablehlo.add %v6880, %v6881 : tensor<384xf32>
    %v6883 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6884 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6885 = stablehlo.multiply %v6883, %d1bv : tensor<384xf32>
    %v6886 = stablehlo.multiply %v3354, %v3354 : tensor<384xf32>
    %v6887 = stablehlo.multiply %v6884, %v6886 : tensor<384xf32>
    %v6888 = stablehlo.add %v6885, %v6887 : tensor<384xf32>
    %v6889 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6890 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6891 = stablehlo.divide %v6882, %v6889 : tensor<384xf32>
    %v6892 = stablehlo.divide %v6888, %v6890 : tensor<384xf32>
    %v6893 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6894 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6895 = stablehlo.sqrt %v6892 : tensor<384xf32>
    %v6896 = stablehlo.add %v6895, %v6894 : tensor<384xf32>
    %v6897 = stablehlo.divide %v6891, %v6896 : tensor<384xf32>
    %v6898 = stablehlo.multiply %v6893, %v6897 : tensor<384xf32>
    %v6899 = stablehlo.subtract %d1b, %v6898 : tensor<384xf32>
    %v6900 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6901 = stablehlo.multiply %v6900, %v6893 : tensor<384xf32>
    %v6902 = stablehlo.multiply %v6901, %d1b : tensor<384xf32>
    %v6903 = stablehlo.subtract %v6899, %v6902 : tensor<384xf32>
    %v6904 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6905 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6906 = stablehlo.multiply %v6904, %s2b0dWm : tensor<384x1x7x7xf32>
    %v6907 = stablehlo.multiply %v6905, %v3298 : tensor<384x1x7x7xf32>
    %v6908 = stablehlo.add %v6906, %v6907 : tensor<384x1x7x7xf32>
    %v6909 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6910 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6911 = stablehlo.multiply %v6909, %s2b0dWv : tensor<384x1x7x7xf32>
    %v6912 = stablehlo.multiply %v3298, %v3298 : tensor<384x1x7x7xf32>
    %v6913 = stablehlo.multiply %v6910, %v6912 : tensor<384x1x7x7xf32>
    %v6914 = stablehlo.add %v6911, %v6913 : tensor<384x1x7x7xf32>
    %v6915 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6916 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6917 = stablehlo.multiply %v6915, %s2b0dWm : tensor<384x1x7x7xf32>
    %v6918 = stablehlo.multiply %v6916, %v3298 : tensor<384x1x7x7xf32>
    %v6919 = stablehlo.add %v6917, %v6918 : tensor<384x1x7x7xf32>
    %v6920 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6921 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6922 = stablehlo.multiply %v6920, %s2b0dWv : tensor<384x1x7x7xf32>
    %v6923 = stablehlo.multiply %v3298, %v3298 : tensor<384x1x7x7xf32>
    %v6924 = stablehlo.multiply %v6921, %v6923 : tensor<384x1x7x7xf32>
    %v6925 = stablehlo.add %v6922, %v6924 : tensor<384x1x7x7xf32>
    %v6926 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6927 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6928 = stablehlo.divide %v6919, %v6926 : tensor<384x1x7x7xf32>
    %v6929 = stablehlo.divide %v6925, %v6927 : tensor<384x1x7x7xf32>
    %v6930 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6931 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6932 = stablehlo.sqrt %v6929 : tensor<384x1x7x7xf32>
    %v6933 = stablehlo.add %v6932, %v6931 : tensor<384x1x7x7xf32>
    %v6934 = stablehlo.divide %v6928, %v6933 : tensor<384x1x7x7xf32>
    %v6935 = stablehlo.multiply %v6930, %v6934 : tensor<384x1x7x7xf32>
    %v6936 = stablehlo.subtract %s2b0dW, %v6935 : tensor<384x1x7x7xf32>
    %v6937 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v6938 = stablehlo.multiply %v6937, %v6930 : tensor<384x1x7x7xf32>
    %v6939 = stablehlo.multiply %v6938, %s2b0dW : tensor<384x1x7x7xf32>
    %v6940 = stablehlo.subtract %v6936, %v6939 : tensor<384x1x7x7xf32>
    %v6941 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6942 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6943 = stablehlo.multiply %v6941, %s2b0dbm : tensor<384xf32>
    %v6944 = stablehlo.multiply %v6942, %v3301 : tensor<384xf32>
    %v6945 = stablehlo.add %v6943, %v6944 : tensor<384xf32>
    %v6946 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6947 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6948 = stablehlo.multiply %v6946, %s2b0dbv : tensor<384xf32>
    %v6949 = stablehlo.multiply %v3301, %v3301 : tensor<384xf32>
    %v6950 = stablehlo.multiply %v6947, %v6949 : tensor<384xf32>
    %v6951 = stablehlo.add %v6948, %v6950 : tensor<384xf32>
    %v6952 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6953 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6954 = stablehlo.multiply %v6952, %s2b0dbm : tensor<384xf32>
    %v6955 = stablehlo.multiply %v6953, %v3301 : tensor<384xf32>
    %v6956 = stablehlo.add %v6954, %v6955 : tensor<384xf32>
    %v6957 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6958 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6959 = stablehlo.multiply %v6957, %s2b0dbv : tensor<384xf32>
    %v6960 = stablehlo.multiply %v3301, %v3301 : tensor<384xf32>
    %v6961 = stablehlo.multiply %v6958, %v6960 : tensor<384xf32>
    %v6962 = stablehlo.add %v6959, %v6961 : tensor<384xf32>
    %v6963 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6964 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6965 = stablehlo.divide %v6956, %v6963 : tensor<384xf32>
    %v6966 = stablehlo.divide %v6962, %v6964 : tensor<384xf32>
    %v6967 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6968 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6969 = stablehlo.sqrt %v6966 : tensor<384xf32>
    %v6970 = stablehlo.add %v6969, %v6968 : tensor<384xf32>
    %v6971 = stablehlo.divide %v6965, %v6970 : tensor<384xf32>
    %v6972 = stablehlo.multiply %v6967, %v6971 : tensor<384xf32>
    %v6973 = stablehlo.subtract %s2b0db, %v6972 : tensor<384xf32>
    %v6974 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6975 = stablehlo.multiply %v6974, %v6967 : tensor<384xf32>
    %v6976 = stablehlo.multiply %v6975, %s2b0db : tensor<384xf32>
    %v6977 = stablehlo.subtract %v6973, %v6976 : tensor<384xf32>
    %v6978 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6979 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6980 = stablehlo.multiply %v6978, %s2b0ngm : tensor<384xf32>
    %v6981 = stablehlo.multiply %v6979, %v3286 : tensor<384xf32>
    %v6982 = stablehlo.add %v6980, %v6981 : tensor<384xf32>
    %v6983 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6984 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6985 = stablehlo.multiply %v6983, %s2b0ngv : tensor<384xf32>
    %v6986 = stablehlo.multiply %v3286, %v3286 : tensor<384xf32>
    %v6987 = stablehlo.multiply %v6984, %v6986 : tensor<384xf32>
    %v6988 = stablehlo.add %v6985, %v6987 : tensor<384xf32>
    %v6989 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6990 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6991 = stablehlo.multiply %v6989, %s2b0ngm : tensor<384xf32>
    %v6992 = stablehlo.multiply %v6990, %v3286 : tensor<384xf32>
    %v6993 = stablehlo.add %v6991, %v6992 : tensor<384xf32>
    %v6994 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6995 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v6996 = stablehlo.multiply %v6994, %s2b0ngv : tensor<384xf32>
    %v6997 = stablehlo.multiply %v3286, %v3286 : tensor<384xf32>
    %v6998 = stablehlo.multiply %v6995, %v6997 : tensor<384xf32>
    %v6999 = stablehlo.add %v6996, %v6998 : tensor<384xf32>
    %v7000 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7001 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7002 = stablehlo.divide %v6993, %v7000 : tensor<384xf32>
    %v7003 = stablehlo.divide %v6999, %v7001 : tensor<384xf32>
    %v7004 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7005 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7006 = stablehlo.sqrt %v7003 : tensor<384xf32>
    %v7007 = stablehlo.add %v7006, %v7005 : tensor<384xf32>
    %v7008 = stablehlo.divide %v7002, %v7007 : tensor<384xf32>
    %v7009 = stablehlo.multiply %v7004, %v7008 : tensor<384xf32>
    %v7010 = stablehlo.subtract %s2b0ng, %v7009 : tensor<384xf32>
    %v7011 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7012 = stablehlo.multiply %v7011, %v7004 : tensor<384xf32>
    %v7013 = stablehlo.multiply %v7012, %s2b0ng : tensor<384xf32>
    %v7014 = stablehlo.subtract %v7010, %v7013 : tensor<384xf32>
    %v7015 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7016 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7017 = stablehlo.multiply %v7015, %s2b0nbtm : tensor<384xf32>
    %v7018 = stablehlo.multiply %v7016, %v3292 : tensor<384xf32>
    %v7019 = stablehlo.add %v7017, %v7018 : tensor<384xf32>
    %v7020 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7021 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7022 = stablehlo.multiply %v7020, %s2b0nbtv : tensor<384xf32>
    %v7023 = stablehlo.multiply %v3292, %v3292 : tensor<384xf32>
    %v7024 = stablehlo.multiply %v7021, %v7023 : tensor<384xf32>
    %v7025 = stablehlo.add %v7022, %v7024 : tensor<384xf32>
    %v7026 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7027 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7028 = stablehlo.multiply %v7026, %s2b0nbtm : tensor<384xf32>
    %v7029 = stablehlo.multiply %v7027, %v3292 : tensor<384xf32>
    %v7030 = stablehlo.add %v7028, %v7029 : tensor<384xf32>
    %v7031 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7032 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7033 = stablehlo.multiply %v7031, %s2b0nbtv : tensor<384xf32>
    %v7034 = stablehlo.multiply %v3292, %v3292 : tensor<384xf32>
    %v7035 = stablehlo.multiply %v7032, %v7034 : tensor<384xf32>
    %v7036 = stablehlo.add %v7033, %v7035 : tensor<384xf32>
    %v7037 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7038 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7039 = stablehlo.divide %v7030, %v7037 : tensor<384xf32>
    %v7040 = stablehlo.divide %v7036, %v7038 : tensor<384xf32>
    %v7041 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7042 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7043 = stablehlo.sqrt %v7040 : tensor<384xf32>
    %v7044 = stablehlo.add %v7043, %v7042 : tensor<384xf32>
    %v7045 = stablehlo.divide %v7039, %v7044 : tensor<384xf32>
    %v7046 = stablehlo.multiply %v7041, %v7045 : tensor<384xf32>
    %v7047 = stablehlo.subtract %s2b0nbt, %v7046 : tensor<384xf32>
    %v7048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7049 = stablehlo.multiply %v7048, %v7041 : tensor<384xf32>
    %v7050 = stablehlo.multiply %v7049, %s2b0nbt : tensor<384xf32>
    %v7051 = stablehlo.subtract %v7047, %v7050 : tensor<384xf32>
    %v7052 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7053 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7054 = stablehlo.multiply %v7052, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v7055 = stablehlo.multiply %v7053, %v3259 : tensor<1536x384x1x1xf32>
    %v7056 = stablehlo.add %v7054, %v7055 : tensor<1536x384x1x1xf32>
    %v7057 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7058 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7059 = stablehlo.multiply %v7057, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v7060 = stablehlo.multiply %v3259, %v3259 : tensor<1536x384x1x1xf32>
    %v7061 = stablehlo.multiply %v7058, %v7060 : tensor<1536x384x1x1xf32>
    %v7062 = stablehlo.add %v7059, %v7061 : tensor<1536x384x1x1xf32>
    %v7063 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7064 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7065 = stablehlo.multiply %v7063, %s2b0eWm : tensor<1536x384x1x1xf32>
    %v7066 = stablehlo.multiply %v7064, %v3259 : tensor<1536x384x1x1xf32>
    %v7067 = stablehlo.add %v7065, %v7066 : tensor<1536x384x1x1xf32>
    %v7068 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7069 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7070 = stablehlo.multiply %v7068, %s2b0eWv : tensor<1536x384x1x1xf32>
    %v7071 = stablehlo.multiply %v3259, %v3259 : tensor<1536x384x1x1xf32>
    %v7072 = stablehlo.multiply %v7069, %v7071 : tensor<1536x384x1x1xf32>
    %v7073 = stablehlo.add %v7070, %v7072 : tensor<1536x384x1x1xf32>
    %v7074 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7075 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7076 = stablehlo.divide %v7067, %v7074 : tensor<1536x384x1x1xf32>
    %v7077 = stablehlo.divide %v7073, %v7075 : tensor<1536x384x1x1xf32>
    %v7078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7079 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7080 = stablehlo.sqrt %v7077 : tensor<1536x384x1x1xf32>
    %v7081 = stablehlo.add %v7080, %v7079 : tensor<1536x384x1x1xf32>
    %v7082 = stablehlo.divide %v7076, %v7081 : tensor<1536x384x1x1xf32>
    %v7083 = stablehlo.multiply %v7078, %v7082 : tensor<1536x384x1x1xf32>
    %v7084 = stablehlo.subtract %s2b0eW, %v7083 : tensor<1536x384x1x1xf32>
    %v7085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7086 = stablehlo.multiply %v7085, %v7078 : tensor<1536x384x1x1xf32>
    %v7087 = stablehlo.multiply %v7086, %s2b0eW : tensor<1536x384x1x1xf32>
    %v7088 = stablehlo.subtract %v7084, %v7087 : tensor<1536x384x1x1xf32>
    %v7089 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7090 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7091 = stablehlo.multiply %v7089, %s2b0ebm : tensor<1536xf32>
    %v7092 = stablehlo.multiply %v7090, %v3262 : tensor<1536xf32>
    %v7093 = stablehlo.add %v7091, %v7092 : tensor<1536xf32>
    %v7094 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7095 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7096 = stablehlo.multiply %v7094, %s2b0ebv : tensor<1536xf32>
    %v7097 = stablehlo.multiply %v3262, %v3262 : tensor<1536xf32>
    %v7098 = stablehlo.multiply %v7095, %v7097 : tensor<1536xf32>
    %v7099 = stablehlo.add %v7096, %v7098 : tensor<1536xf32>
    %v7100 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7101 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7102 = stablehlo.multiply %v7100, %s2b0ebm : tensor<1536xf32>
    %v7103 = stablehlo.multiply %v7101, %v3262 : tensor<1536xf32>
    %v7104 = stablehlo.add %v7102, %v7103 : tensor<1536xf32>
    %v7105 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7106 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7107 = stablehlo.multiply %v7105, %s2b0ebv : tensor<1536xf32>
    %v7108 = stablehlo.multiply %v3262, %v3262 : tensor<1536xf32>
    %v7109 = stablehlo.multiply %v7106, %v7108 : tensor<1536xf32>
    %v7110 = stablehlo.add %v7107, %v7109 : tensor<1536xf32>
    %v7111 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7112 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7113 = stablehlo.divide %v7104, %v7111 : tensor<1536xf32>
    %v7114 = stablehlo.divide %v7110, %v7112 : tensor<1536xf32>
    %v7115 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7116 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7117 = stablehlo.sqrt %v7114 : tensor<1536xf32>
    %v7118 = stablehlo.add %v7117, %v7116 : tensor<1536xf32>
    %v7119 = stablehlo.divide %v7113, %v7118 : tensor<1536xf32>
    %v7120 = stablehlo.multiply %v7115, %v7119 : tensor<1536xf32>
    %v7121 = stablehlo.subtract %s2b0eb, %v7120 : tensor<1536xf32>
    %v7122 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7123 = stablehlo.multiply %v7122, %v7115 : tensor<1536xf32>
    %v7124 = stablehlo.multiply %v7123, %s2b0eb : tensor<1536xf32>
    %v7125 = stablehlo.subtract %v7121, %v7124 : tensor<1536xf32>
    %v7126 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7127 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7128 = stablehlo.multiply %v7126, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v7129 = stablehlo.multiply %v7127, %v3250 : tensor<384x1536x1x1xf32>
    %v7130 = stablehlo.add %v7128, %v7129 : tensor<384x1536x1x1xf32>
    %v7131 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7132 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7133 = stablehlo.multiply %v7131, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v7134 = stablehlo.multiply %v3250, %v3250 : tensor<384x1536x1x1xf32>
    %v7135 = stablehlo.multiply %v7132, %v7134 : tensor<384x1536x1x1xf32>
    %v7136 = stablehlo.add %v7133, %v7135 : tensor<384x1536x1x1xf32>
    %v7137 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7138 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7139 = stablehlo.multiply %v7137, %s2b0pWm : tensor<384x1536x1x1xf32>
    %v7140 = stablehlo.multiply %v7138, %v3250 : tensor<384x1536x1x1xf32>
    %v7141 = stablehlo.add %v7139, %v7140 : tensor<384x1536x1x1xf32>
    %v7142 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7143 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7144 = stablehlo.multiply %v7142, %s2b0pWv : tensor<384x1536x1x1xf32>
    %v7145 = stablehlo.multiply %v3250, %v3250 : tensor<384x1536x1x1xf32>
    %v7146 = stablehlo.multiply %v7143, %v7145 : tensor<384x1536x1x1xf32>
    %v7147 = stablehlo.add %v7144, %v7146 : tensor<384x1536x1x1xf32>
    %v7148 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7149 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7150 = stablehlo.divide %v7141, %v7148 : tensor<384x1536x1x1xf32>
    %v7151 = stablehlo.divide %v7147, %v7149 : tensor<384x1536x1x1xf32>
    %v7152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7153 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7154 = stablehlo.sqrt %v7151 : tensor<384x1536x1x1xf32>
    %v7155 = stablehlo.add %v7154, %v7153 : tensor<384x1536x1x1xf32>
    %v7156 = stablehlo.divide %v7150, %v7155 : tensor<384x1536x1x1xf32>
    %v7157 = stablehlo.multiply %v7152, %v7156 : tensor<384x1536x1x1xf32>
    %v7158 = stablehlo.subtract %s2b0pW, %v7157 : tensor<384x1536x1x1xf32>
    %v7159 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7160 = stablehlo.multiply %v7159, %v7152 : tensor<384x1536x1x1xf32>
    %v7161 = stablehlo.multiply %v7160, %s2b0pW : tensor<384x1536x1x1xf32>
    %v7162 = stablehlo.subtract %v7158, %v7161 : tensor<384x1536x1x1xf32>
    %v7163 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7164 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7165 = stablehlo.multiply %v7163, %s2b0pbm : tensor<384xf32>
    %v7166 = stablehlo.multiply %v7164, %v3253 : tensor<384xf32>
    %v7167 = stablehlo.add %v7165, %v7166 : tensor<384xf32>
    %v7168 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7169 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7170 = stablehlo.multiply %v7168, %s2b0pbv : tensor<384xf32>
    %v7171 = stablehlo.multiply %v3253, %v3253 : tensor<384xf32>
    %v7172 = stablehlo.multiply %v7169, %v7171 : tensor<384xf32>
    %v7173 = stablehlo.add %v7170, %v7172 : tensor<384xf32>
    %v7174 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7175 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7176 = stablehlo.multiply %v7174, %s2b0pbm : tensor<384xf32>
    %v7177 = stablehlo.multiply %v7175, %v3253 : tensor<384xf32>
    %v7178 = stablehlo.add %v7176, %v7177 : tensor<384xf32>
    %v7179 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7180 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7181 = stablehlo.multiply %v7179, %s2b0pbv : tensor<384xf32>
    %v7182 = stablehlo.multiply %v3253, %v3253 : tensor<384xf32>
    %v7183 = stablehlo.multiply %v7180, %v7182 : tensor<384xf32>
    %v7184 = stablehlo.add %v7181, %v7183 : tensor<384xf32>
    %v7185 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7186 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7187 = stablehlo.divide %v7178, %v7185 : tensor<384xf32>
    %v7188 = stablehlo.divide %v7184, %v7186 : tensor<384xf32>
    %v7189 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7190 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7191 = stablehlo.sqrt %v7188 : tensor<384xf32>
    %v7192 = stablehlo.add %v7191, %v7190 : tensor<384xf32>
    %v7193 = stablehlo.divide %v7187, %v7192 : tensor<384xf32>
    %v7194 = stablehlo.multiply %v7189, %v7193 : tensor<384xf32>
    %v7195 = stablehlo.subtract %s2b0pb, %v7194 : tensor<384xf32>
    %v7196 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7197 = stablehlo.multiply %v7196, %v7189 : tensor<384xf32>
    %v7198 = stablehlo.multiply %v7197, %s2b0pb : tensor<384xf32>
    %v7199 = stablehlo.subtract %v7195, %v7198 : tensor<384xf32>
    %v7200 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7201 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7202 = stablehlo.multiply %v7200, %s2b0lgm : tensor<384xf32>
    %v7203 = stablehlo.multiply %v7201, %v3244 : tensor<384xf32>
    %v7204 = stablehlo.add %v7202, %v7203 : tensor<384xf32>
    %v7205 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7206 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7207 = stablehlo.multiply %v7205, %s2b0lgv : tensor<384xf32>
    %v7208 = stablehlo.multiply %v3244, %v3244 : tensor<384xf32>
    %v7209 = stablehlo.multiply %v7206, %v7208 : tensor<384xf32>
    %v7210 = stablehlo.add %v7207, %v7209 : tensor<384xf32>
    %v7211 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7212 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7213 = stablehlo.multiply %v7211, %s2b0lgm : tensor<384xf32>
    %v7214 = stablehlo.multiply %v7212, %v3244 : tensor<384xf32>
    %v7215 = stablehlo.add %v7213, %v7214 : tensor<384xf32>
    %v7216 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7217 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7218 = stablehlo.multiply %v7216, %s2b0lgv : tensor<384xf32>
    %v7219 = stablehlo.multiply %v3244, %v3244 : tensor<384xf32>
    %v7220 = stablehlo.multiply %v7217, %v7219 : tensor<384xf32>
    %v7221 = stablehlo.add %v7218, %v7220 : tensor<384xf32>
    %v7222 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7223 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7224 = stablehlo.divide %v7215, %v7222 : tensor<384xf32>
    %v7225 = stablehlo.divide %v7221, %v7223 : tensor<384xf32>
    %v7226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7227 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7228 = stablehlo.sqrt %v7225 : tensor<384xf32>
    %v7229 = stablehlo.add %v7228, %v7227 : tensor<384xf32>
    %v7230 = stablehlo.divide %v7224, %v7229 : tensor<384xf32>
    %v7231 = stablehlo.multiply %v7226, %v7230 : tensor<384xf32>
    %v7232 = stablehlo.subtract %s2b0lg, %v7231 : tensor<384xf32>
    %v7233 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7234 = stablehlo.multiply %v7233, %v7226 : tensor<384xf32>
    %v7235 = stablehlo.multiply %v7234, %s2b0lg : tensor<384xf32>
    %v7236 = stablehlo.subtract %v7232, %v7235 : tensor<384xf32>
    %v7237 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7238 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7239 = stablehlo.multiply %v7237, %s2b1dWm : tensor<384x1x7x7xf32>
    %v7240 = stablehlo.multiply %v7238, %v3149 : tensor<384x1x7x7xf32>
    %v7241 = stablehlo.add %v7239, %v7240 : tensor<384x1x7x7xf32>
    %v7242 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7243 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7244 = stablehlo.multiply %v7242, %s2b1dWv : tensor<384x1x7x7xf32>
    %v7245 = stablehlo.multiply %v3149, %v3149 : tensor<384x1x7x7xf32>
    %v7246 = stablehlo.multiply %v7243, %v7245 : tensor<384x1x7x7xf32>
    %v7247 = stablehlo.add %v7244, %v7246 : tensor<384x1x7x7xf32>
    %v7248 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7249 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7250 = stablehlo.multiply %v7248, %s2b1dWm : tensor<384x1x7x7xf32>
    %v7251 = stablehlo.multiply %v7249, %v3149 : tensor<384x1x7x7xf32>
    %v7252 = stablehlo.add %v7250, %v7251 : tensor<384x1x7x7xf32>
    %v7253 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7254 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7255 = stablehlo.multiply %v7253, %s2b1dWv : tensor<384x1x7x7xf32>
    %v7256 = stablehlo.multiply %v3149, %v3149 : tensor<384x1x7x7xf32>
    %v7257 = stablehlo.multiply %v7254, %v7256 : tensor<384x1x7x7xf32>
    %v7258 = stablehlo.add %v7255, %v7257 : tensor<384x1x7x7xf32>
    %v7259 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7260 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7261 = stablehlo.divide %v7252, %v7259 : tensor<384x1x7x7xf32>
    %v7262 = stablehlo.divide %v7258, %v7260 : tensor<384x1x7x7xf32>
    %v7263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7264 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7265 = stablehlo.sqrt %v7262 : tensor<384x1x7x7xf32>
    %v7266 = stablehlo.add %v7265, %v7264 : tensor<384x1x7x7xf32>
    %v7267 = stablehlo.divide %v7261, %v7266 : tensor<384x1x7x7xf32>
    %v7268 = stablehlo.multiply %v7263, %v7267 : tensor<384x1x7x7xf32>
    %v7269 = stablehlo.subtract %s2b1dW, %v7268 : tensor<384x1x7x7xf32>
    %v7270 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7271 = stablehlo.multiply %v7270, %v7263 : tensor<384x1x7x7xf32>
    %v7272 = stablehlo.multiply %v7271, %s2b1dW : tensor<384x1x7x7xf32>
    %v7273 = stablehlo.subtract %v7269, %v7272 : tensor<384x1x7x7xf32>
    %v7274 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7275 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7276 = stablehlo.multiply %v7274, %s2b1dbm : tensor<384xf32>
    %v7277 = stablehlo.multiply %v7275, %v3152 : tensor<384xf32>
    %v7278 = stablehlo.add %v7276, %v7277 : tensor<384xf32>
    %v7279 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7280 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7281 = stablehlo.multiply %v7279, %s2b1dbv : tensor<384xf32>
    %v7282 = stablehlo.multiply %v3152, %v3152 : tensor<384xf32>
    %v7283 = stablehlo.multiply %v7280, %v7282 : tensor<384xf32>
    %v7284 = stablehlo.add %v7281, %v7283 : tensor<384xf32>
    %v7285 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7286 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7287 = stablehlo.multiply %v7285, %s2b1dbm : tensor<384xf32>
    %v7288 = stablehlo.multiply %v7286, %v3152 : tensor<384xf32>
    %v7289 = stablehlo.add %v7287, %v7288 : tensor<384xf32>
    %v7290 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7291 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7292 = stablehlo.multiply %v7290, %s2b1dbv : tensor<384xf32>
    %v7293 = stablehlo.multiply %v3152, %v3152 : tensor<384xf32>
    %v7294 = stablehlo.multiply %v7291, %v7293 : tensor<384xf32>
    %v7295 = stablehlo.add %v7292, %v7294 : tensor<384xf32>
    %v7296 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7297 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7298 = stablehlo.divide %v7289, %v7296 : tensor<384xf32>
    %v7299 = stablehlo.divide %v7295, %v7297 : tensor<384xf32>
    %v7300 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7301 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7302 = stablehlo.sqrt %v7299 : tensor<384xf32>
    %v7303 = stablehlo.add %v7302, %v7301 : tensor<384xf32>
    %v7304 = stablehlo.divide %v7298, %v7303 : tensor<384xf32>
    %v7305 = stablehlo.multiply %v7300, %v7304 : tensor<384xf32>
    %v7306 = stablehlo.subtract %s2b1db, %v7305 : tensor<384xf32>
    %v7307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7308 = stablehlo.multiply %v7307, %v7300 : tensor<384xf32>
    %v7309 = stablehlo.multiply %v7308, %s2b1db : tensor<384xf32>
    %v7310 = stablehlo.subtract %v7306, %v7309 : tensor<384xf32>
    %v7311 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7312 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7313 = stablehlo.multiply %v7311, %s2b1ngm : tensor<384xf32>
    %v7314 = stablehlo.multiply %v7312, %v3137 : tensor<384xf32>
    %v7315 = stablehlo.add %v7313, %v7314 : tensor<384xf32>
    %v7316 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7317 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7318 = stablehlo.multiply %v7316, %s2b1ngv : tensor<384xf32>
    %v7319 = stablehlo.multiply %v3137, %v3137 : tensor<384xf32>
    %v7320 = stablehlo.multiply %v7317, %v7319 : tensor<384xf32>
    %v7321 = stablehlo.add %v7318, %v7320 : tensor<384xf32>
    %v7322 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7323 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7324 = stablehlo.multiply %v7322, %s2b1ngm : tensor<384xf32>
    %v7325 = stablehlo.multiply %v7323, %v3137 : tensor<384xf32>
    %v7326 = stablehlo.add %v7324, %v7325 : tensor<384xf32>
    %v7327 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7328 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7329 = stablehlo.multiply %v7327, %s2b1ngv : tensor<384xf32>
    %v7330 = stablehlo.multiply %v3137, %v3137 : tensor<384xf32>
    %v7331 = stablehlo.multiply %v7328, %v7330 : tensor<384xf32>
    %v7332 = stablehlo.add %v7329, %v7331 : tensor<384xf32>
    %v7333 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7334 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7335 = stablehlo.divide %v7326, %v7333 : tensor<384xf32>
    %v7336 = stablehlo.divide %v7332, %v7334 : tensor<384xf32>
    %v7337 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7338 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7339 = stablehlo.sqrt %v7336 : tensor<384xf32>
    %v7340 = stablehlo.add %v7339, %v7338 : tensor<384xf32>
    %v7341 = stablehlo.divide %v7335, %v7340 : tensor<384xf32>
    %v7342 = stablehlo.multiply %v7337, %v7341 : tensor<384xf32>
    %v7343 = stablehlo.subtract %s2b1ng, %v7342 : tensor<384xf32>
    %v7344 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7345 = stablehlo.multiply %v7344, %v7337 : tensor<384xf32>
    %v7346 = stablehlo.multiply %v7345, %s2b1ng : tensor<384xf32>
    %v7347 = stablehlo.subtract %v7343, %v7346 : tensor<384xf32>
    %v7348 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7349 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7350 = stablehlo.multiply %v7348, %s2b1nbtm : tensor<384xf32>
    %v7351 = stablehlo.multiply %v7349, %v3143 : tensor<384xf32>
    %v7352 = stablehlo.add %v7350, %v7351 : tensor<384xf32>
    %v7353 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7354 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7355 = stablehlo.multiply %v7353, %s2b1nbtv : tensor<384xf32>
    %v7356 = stablehlo.multiply %v3143, %v3143 : tensor<384xf32>
    %v7357 = stablehlo.multiply %v7354, %v7356 : tensor<384xf32>
    %v7358 = stablehlo.add %v7355, %v7357 : tensor<384xf32>
    %v7359 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7360 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7361 = stablehlo.multiply %v7359, %s2b1nbtm : tensor<384xf32>
    %v7362 = stablehlo.multiply %v7360, %v3143 : tensor<384xf32>
    %v7363 = stablehlo.add %v7361, %v7362 : tensor<384xf32>
    %v7364 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7365 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7366 = stablehlo.multiply %v7364, %s2b1nbtv : tensor<384xf32>
    %v7367 = stablehlo.multiply %v3143, %v3143 : tensor<384xf32>
    %v7368 = stablehlo.multiply %v7365, %v7367 : tensor<384xf32>
    %v7369 = stablehlo.add %v7366, %v7368 : tensor<384xf32>
    %v7370 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7371 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7372 = stablehlo.divide %v7363, %v7370 : tensor<384xf32>
    %v7373 = stablehlo.divide %v7369, %v7371 : tensor<384xf32>
    %v7374 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7375 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7376 = stablehlo.sqrt %v7373 : tensor<384xf32>
    %v7377 = stablehlo.add %v7376, %v7375 : tensor<384xf32>
    %v7378 = stablehlo.divide %v7372, %v7377 : tensor<384xf32>
    %v7379 = stablehlo.multiply %v7374, %v7378 : tensor<384xf32>
    %v7380 = stablehlo.subtract %s2b1nbt, %v7379 : tensor<384xf32>
    %v7381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7382 = stablehlo.multiply %v7381, %v7374 : tensor<384xf32>
    %v7383 = stablehlo.multiply %v7382, %s2b1nbt : tensor<384xf32>
    %v7384 = stablehlo.subtract %v7380, %v7383 : tensor<384xf32>
    %v7385 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7386 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7387 = stablehlo.multiply %v7385, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v7388 = stablehlo.multiply %v7386, %v3110 : tensor<1536x384x1x1xf32>
    %v7389 = stablehlo.add %v7387, %v7388 : tensor<1536x384x1x1xf32>
    %v7390 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7391 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7392 = stablehlo.multiply %v7390, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v7393 = stablehlo.multiply %v3110, %v3110 : tensor<1536x384x1x1xf32>
    %v7394 = stablehlo.multiply %v7391, %v7393 : tensor<1536x384x1x1xf32>
    %v7395 = stablehlo.add %v7392, %v7394 : tensor<1536x384x1x1xf32>
    %v7396 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7397 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7398 = stablehlo.multiply %v7396, %s2b1eWm : tensor<1536x384x1x1xf32>
    %v7399 = stablehlo.multiply %v7397, %v3110 : tensor<1536x384x1x1xf32>
    %v7400 = stablehlo.add %v7398, %v7399 : tensor<1536x384x1x1xf32>
    %v7401 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7402 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7403 = stablehlo.multiply %v7401, %s2b1eWv : tensor<1536x384x1x1xf32>
    %v7404 = stablehlo.multiply %v3110, %v3110 : tensor<1536x384x1x1xf32>
    %v7405 = stablehlo.multiply %v7402, %v7404 : tensor<1536x384x1x1xf32>
    %v7406 = stablehlo.add %v7403, %v7405 : tensor<1536x384x1x1xf32>
    %v7407 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7408 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7409 = stablehlo.divide %v7400, %v7407 : tensor<1536x384x1x1xf32>
    %v7410 = stablehlo.divide %v7406, %v7408 : tensor<1536x384x1x1xf32>
    %v7411 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7412 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7413 = stablehlo.sqrt %v7410 : tensor<1536x384x1x1xf32>
    %v7414 = stablehlo.add %v7413, %v7412 : tensor<1536x384x1x1xf32>
    %v7415 = stablehlo.divide %v7409, %v7414 : tensor<1536x384x1x1xf32>
    %v7416 = stablehlo.multiply %v7411, %v7415 : tensor<1536x384x1x1xf32>
    %v7417 = stablehlo.subtract %s2b1eW, %v7416 : tensor<1536x384x1x1xf32>
    %v7418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7419 = stablehlo.multiply %v7418, %v7411 : tensor<1536x384x1x1xf32>
    %v7420 = stablehlo.multiply %v7419, %s2b1eW : tensor<1536x384x1x1xf32>
    %v7421 = stablehlo.subtract %v7417, %v7420 : tensor<1536x384x1x1xf32>
    %v7422 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7423 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7424 = stablehlo.multiply %v7422, %s2b1ebm : tensor<1536xf32>
    %v7425 = stablehlo.multiply %v7423, %v3113 : tensor<1536xf32>
    %v7426 = stablehlo.add %v7424, %v7425 : tensor<1536xf32>
    %v7427 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7428 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7429 = stablehlo.multiply %v7427, %s2b1ebv : tensor<1536xf32>
    %v7430 = stablehlo.multiply %v3113, %v3113 : tensor<1536xf32>
    %v7431 = stablehlo.multiply %v7428, %v7430 : tensor<1536xf32>
    %v7432 = stablehlo.add %v7429, %v7431 : tensor<1536xf32>
    %v7433 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7434 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7435 = stablehlo.multiply %v7433, %s2b1ebm : tensor<1536xf32>
    %v7436 = stablehlo.multiply %v7434, %v3113 : tensor<1536xf32>
    %v7437 = stablehlo.add %v7435, %v7436 : tensor<1536xf32>
    %v7438 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7439 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7440 = stablehlo.multiply %v7438, %s2b1ebv : tensor<1536xf32>
    %v7441 = stablehlo.multiply %v3113, %v3113 : tensor<1536xf32>
    %v7442 = stablehlo.multiply %v7439, %v7441 : tensor<1536xf32>
    %v7443 = stablehlo.add %v7440, %v7442 : tensor<1536xf32>
    %v7444 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7445 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7446 = stablehlo.divide %v7437, %v7444 : tensor<1536xf32>
    %v7447 = stablehlo.divide %v7443, %v7445 : tensor<1536xf32>
    %v7448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7449 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7450 = stablehlo.sqrt %v7447 : tensor<1536xf32>
    %v7451 = stablehlo.add %v7450, %v7449 : tensor<1536xf32>
    %v7452 = stablehlo.divide %v7446, %v7451 : tensor<1536xf32>
    %v7453 = stablehlo.multiply %v7448, %v7452 : tensor<1536xf32>
    %v7454 = stablehlo.subtract %s2b1eb, %v7453 : tensor<1536xf32>
    %v7455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7456 = stablehlo.multiply %v7455, %v7448 : tensor<1536xf32>
    %v7457 = stablehlo.multiply %v7456, %s2b1eb : tensor<1536xf32>
    %v7458 = stablehlo.subtract %v7454, %v7457 : tensor<1536xf32>
    %v7459 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7460 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7461 = stablehlo.multiply %v7459, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v7462 = stablehlo.multiply %v7460, %v3101 : tensor<384x1536x1x1xf32>
    %v7463 = stablehlo.add %v7461, %v7462 : tensor<384x1536x1x1xf32>
    %v7464 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7465 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7466 = stablehlo.multiply %v7464, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v7467 = stablehlo.multiply %v3101, %v3101 : tensor<384x1536x1x1xf32>
    %v7468 = stablehlo.multiply %v7465, %v7467 : tensor<384x1536x1x1xf32>
    %v7469 = stablehlo.add %v7466, %v7468 : tensor<384x1536x1x1xf32>
    %v7470 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7471 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7472 = stablehlo.multiply %v7470, %s2b1pWm : tensor<384x1536x1x1xf32>
    %v7473 = stablehlo.multiply %v7471, %v3101 : tensor<384x1536x1x1xf32>
    %v7474 = stablehlo.add %v7472, %v7473 : tensor<384x1536x1x1xf32>
    %v7475 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7476 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7477 = stablehlo.multiply %v7475, %s2b1pWv : tensor<384x1536x1x1xf32>
    %v7478 = stablehlo.multiply %v3101, %v3101 : tensor<384x1536x1x1xf32>
    %v7479 = stablehlo.multiply %v7476, %v7478 : tensor<384x1536x1x1xf32>
    %v7480 = stablehlo.add %v7477, %v7479 : tensor<384x1536x1x1xf32>
    %v7481 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7482 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7483 = stablehlo.divide %v7474, %v7481 : tensor<384x1536x1x1xf32>
    %v7484 = stablehlo.divide %v7480, %v7482 : tensor<384x1536x1x1xf32>
    %v7485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7486 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7487 = stablehlo.sqrt %v7484 : tensor<384x1536x1x1xf32>
    %v7488 = stablehlo.add %v7487, %v7486 : tensor<384x1536x1x1xf32>
    %v7489 = stablehlo.divide %v7483, %v7488 : tensor<384x1536x1x1xf32>
    %v7490 = stablehlo.multiply %v7485, %v7489 : tensor<384x1536x1x1xf32>
    %v7491 = stablehlo.subtract %s2b1pW, %v7490 : tensor<384x1536x1x1xf32>
    %v7492 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7493 = stablehlo.multiply %v7492, %v7485 : tensor<384x1536x1x1xf32>
    %v7494 = stablehlo.multiply %v7493, %s2b1pW : tensor<384x1536x1x1xf32>
    %v7495 = stablehlo.subtract %v7491, %v7494 : tensor<384x1536x1x1xf32>
    %v7496 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7497 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7498 = stablehlo.multiply %v7496, %s2b1pbm : tensor<384xf32>
    %v7499 = stablehlo.multiply %v7497, %v3104 : tensor<384xf32>
    %v7500 = stablehlo.add %v7498, %v7499 : tensor<384xf32>
    %v7501 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7502 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7503 = stablehlo.multiply %v7501, %s2b1pbv : tensor<384xf32>
    %v7504 = stablehlo.multiply %v3104, %v3104 : tensor<384xf32>
    %v7505 = stablehlo.multiply %v7502, %v7504 : tensor<384xf32>
    %v7506 = stablehlo.add %v7503, %v7505 : tensor<384xf32>
    %v7507 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7508 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7509 = stablehlo.multiply %v7507, %s2b1pbm : tensor<384xf32>
    %v7510 = stablehlo.multiply %v7508, %v3104 : tensor<384xf32>
    %v7511 = stablehlo.add %v7509, %v7510 : tensor<384xf32>
    %v7512 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7513 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7514 = stablehlo.multiply %v7512, %s2b1pbv : tensor<384xf32>
    %v7515 = stablehlo.multiply %v3104, %v3104 : tensor<384xf32>
    %v7516 = stablehlo.multiply %v7513, %v7515 : tensor<384xf32>
    %v7517 = stablehlo.add %v7514, %v7516 : tensor<384xf32>
    %v7518 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7519 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7520 = stablehlo.divide %v7511, %v7518 : tensor<384xf32>
    %v7521 = stablehlo.divide %v7517, %v7519 : tensor<384xf32>
    %v7522 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7523 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7524 = stablehlo.sqrt %v7521 : tensor<384xf32>
    %v7525 = stablehlo.add %v7524, %v7523 : tensor<384xf32>
    %v7526 = stablehlo.divide %v7520, %v7525 : tensor<384xf32>
    %v7527 = stablehlo.multiply %v7522, %v7526 : tensor<384xf32>
    %v7528 = stablehlo.subtract %s2b1pb, %v7527 : tensor<384xf32>
    %v7529 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7530 = stablehlo.multiply %v7529, %v7522 : tensor<384xf32>
    %v7531 = stablehlo.multiply %v7530, %s2b1pb : tensor<384xf32>
    %v7532 = stablehlo.subtract %v7528, %v7531 : tensor<384xf32>
    %v7533 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7534 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7535 = stablehlo.multiply %v7533, %s2b1lgm : tensor<384xf32>
    %v7536 = stablehlo.multiply %v7534, %v3095 : tensor<384xf32>
    %v7537 = stablehlo.add %v7535, %v7536 : tensor<384xf32>
    %v7538 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7539 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7540 = stablehlo.multiply %v7538, %s2b1lgv : tensor<384xf32>
    %v7541 = stablehlo.multiply %v3095, %v3095 : tensor<384xf32>
    %v7542 = stablehlo.multiply %v7539, %v7541 : tensor<384xf32>
    %v7543 = stablehlo.add %v7540, %v7542 : tensor<384xf32>
    %v7544 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7545 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7546 = stablehlo.multiply %v7544, %s2b1lgm : tensor<384xf32>
    %v7547 = stablehlo.multiply %v7545, %v3095 : tensor<384xf32>
    %v7548 = stablehlo.add %v7546, %v7547 : tensor<384xf32>
    %v7549 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7550 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7551 = stablehlo.multiply %v7549, %s2b1lgv : tensor<384xf32>
    %v7552 = stablehlo.multiply %v3095, %v3095 : tensor<384xf32>
    %v7553 = stablehlo.multiply %v7550, %v7552 : tensor<384xf32>
    %v7554 = stablehlo.add %v7551, %v7553 : tensor<384xf32>
    %v7555 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7556 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7557 = stablehlo.divide %v7548, %v7555 : tensor<384xf32>
    %v7558 = stablehlo.divide %v7554, %v7556 : tensor<384xf32>
    %v7559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7560 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7561 = stablehlo.sqrt %v7558 : tensor<384xf32>
    %v7562 = stablehlo.add %v7561, %v7560 : tensor<384xf32>
    %v7563 = stablehlo.divide %v7557, %v7562 : tensor<384xf32>
    %v7564 = stablehlo.multiply %v7559, %v7563 : tensor<384xf32>
    %v7565 = stablehlo.subtract %s2b1lg, %v7564 : tensor<384xf32>
    %v7566 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7567 = stablehlo.multiply %v7566, %v7559 : tensor<384xf32>
    %v7568 = stablehlo.multiply %v7567, %s2b1lg : tensor<384xf32>
    %v7569 = stablehlo.subtract %v7565, %v7568 : tensor<384xf32>
    %v7570 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7571 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7572 = stablehlo.multiply %v7570, %s2b2dWm : tensor<384x1x7x7xf32>
    %v7573 = stablehlo.multiply %v7571, %v3000 : tensor<384x1x7x7xf32>
    %v7574 = stablehlo.add %v7572, %v7573 : tensor<384x1x7x7xf32>
    %v7575 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7576 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7577 = stablehlo.multiply %v7575, %s2b2dWv : tensor<384x1x7x7xf32>
    %v7578 = stablehlo.multiply %v3000, %v3000 : tensor<384x1x7x7xf32>
    %v7579 = stablehlo.multiply %v7576, %v7578 : tensor<384x1x7x7xf32>
    %v7580 = stablehlo.add %v7577, %v7579 : tensor<384x1x7x7xf32>
    %v7581 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7582 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7583 = stablehlo.multiply %v7581, %s2b2dWm : tensor<384x1x7x7xf32>
    %v7584 = stablehlo.multiply %v7582, %v3000 : tensor<384x1x7x7xf32>
    %v7585 = stablehlo.add %v7583, %v7584 : tensor<384x1x7x7xf32>
    %v7586 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7587 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7588 = stablehlo.multiply %v7586, %s2b2dWv : tensor<384x1x7x7xf32>
    %v7589 = stablehlo.multiply %v3000, %v3000 : tensor<384x1x7x7xf32>
    %v7590 = stablehlo.multiply %v7587, %v7589 : tensor<384x1x7x7xf32>
    %v7591 = stablehlo.add %v7588, %v7590 : tensor<384x1x7x7xf32>
    %v7592 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7593 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7594 = stablehlo.divide %v7585, %v7592 : tensor<384x1x7x7xf32>
    %v7595 = stablehlo.divide %v7591, %v7593 : tensor<384x1x7x7xf32>
    %v7596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7597 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7598 = stablehlo.sqrt %v7595 : tensor<384x1x7x7xf32>
    %v7599 = stablehlo.add %v7598, %v7597 : tensor<384x1x7x7xf32>
    %v7600 = stablehlo.divide %v7594, %v7599 : tensor<384x1x7x7xf32>
    %v7601 = stablehlo.multiply %v7596, %v7600 : tensor<384x1x7x7xf32>
    %v7602 = stablehlo.subtract %s2b2dW, %v7601 : tensor<384x1x7x7xf32>
    %v7603 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7604 = stablehlo.multiply %v7603, %v7596 : tensor<384x1x7x7xf32>
    %v7605 = stablehlo.multiply %v7604, %s2b2dW : tensor<384x1x7x7xf32>
    %v7606 = stablehlo.subtract %v7602, %v7605 : tensor<384x1x7x7xf32>
    %v7607 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7608 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7609 = stablehlo.multiply %v7607, %s2b2dbm : tensor<384xf32>
    %v7610 = stablehlo.multiply %v7608, %v3003 : tensor<384xf32>
    %v7611 = stablehlo.add %v7609, %v7610 : tensor<384xf32>
    %v7612 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7613 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7614 = stablehlo.multiply %v7612, %s2b2dbv : tensor<384xf32>
    %v7615 = stablehlo.multiply %v3003, %v3003 : tensor<384xf32>
    %v7616 = stablehlo.multiply %v7613, %v7615 : tensor<384xf32>
    %v7617 = stablehlo.add %v7614, %v7616 : tensor<384xf32>
    %v7618 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7619 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7620 = stablehlo.multiply %v7618, %s2b2dbm : tensor<384xf32>
    %v7621 = stablehlo.multiply %v7619, %v3003 : tensor<384xf32>
    %v7622 = stablehlo.add %v7620, %v7621 : tensor<384xf32>
    %v7623 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7624 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7625 = stablehlo.multiply %v7623, %s2b2dbv : tensor<384xf32>
    %v7626 = stablehlo.multiply %v3003, %v3003 : tensor<384xf32>
    %v7627 = stablehlo.multiply %v7624, %v7626 : tensor<384xf32>
    %v7628 = stablehlo.add %v7625, %v7627 : tensor<384xf32>
    %v7629 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7630 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7631 = stablehlo.divide %v7622, %v7629 : tensor<384xf32>
    %v7632 = stablehlo.divide %v7628, %v7630 : tensor<384xf32>
    %v7633 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7634 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7635 = stablehlo.sqrt %v7632 : tensor<384xf32>
    %v7636 = stablehlo.add %v7635, %v7634 : tensor<384xf32>
    %v7637 = stablehlo.divide %v7631, %v7636 : tensor<384xf32>
    %v7638 = stablehlo.multiply %v7633, %v7637 : tensor<384xf32>
    %v7639 = stablehlo.subtract %s2b2db, %v7638 : tensor<384xf32>
    %v7640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7641 = stablehlo.multiply %v7640, %v7633 : tensor<384xf32>
    %v7642 = stablehlo.multiply %v7641, %s2b2db : tensor<384xf32>
    %v7643 = stablehlo.subtract %v7639, %v7642 : tensor<384xf32>
    %v7644 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7645 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7646 = stablehlo.multiply %v7644, %s2b2ngm : tensor<384xf32>
    %v7647 = stablehlo.multiply %v7645, %v2988 : tensor<384xf32>
    %v7648 = stablehlo.add %v7646, %v7647 : tensor<384xf32>
    %v7649 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7650 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7651 = stablehlo.multiply %v7649, %s2b2ngv : tensor<384xf32>
    %v7652 = stablehlo.multiply %v2988, %v2988 : tensor<384xf32>
    %v7653 = stablehlo.multiply %v7650, %v7652 : tensor<384xf32>
    %v7654 = stablehlo.add %v7651, %v7653 : tensor<384xf32>
    %v7655 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7656 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7657 = stablehlo.multiply %v7655, %s2b2ngm : tensor<384xf32>
    %v7658 = stablehlo.multiply %v7656, %v2988 : tensor<384xf32>
    %v7659 = stablehlo.add %v7657, %v7658 : tensor<384xf32>
    %v7660 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7661 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7662 = stablehlo.multiply %v7660, %s2b2ngv : tensor<384xf32>
    %v7663 = stablehlo.multiply %v2988, %v2988 : tensor<384xf32>
    %v7664 = stablehlo.multiply %v7661, %v7663 : tensor<384xf32>
    %v7665 = stablehlo.add %v7662, %v7664 : tensor<384xf32>
    %v7666 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7667 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7668 = stablehlo.divide %v7659, %v7666 : tensor<384xf32>
    %v7669 = stablehlo.divide %v7665, %v7667 : tensor<384xf32>
    %v7670 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7671 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7672 = stablehlo.sqrt %v7669 : tensor<384xf32>
    %v7673 = stablehlo.add %v7672, %v7671 : tensor<384xf32>
    %v7674 = stablehlo.divide %v7668, %v7673 : tensor<384xf32>
    %v7675 = stablehlo.multiply %v7670, %v7674 : tensor<384xf32>
    %v7676 = stablehlo.subtract %s2b2ng, %v7675 : tensor<384xf32>
    %v7677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7678 = stablehlo.multiply %v7677, %v7670 : tensor<384xf32>
    %v7679 = stablehlo.multiply %v7678, %s2b2ng : tensor<384xf32>
    %v7680 = stablehlo.subtract %v7676, %v7679 : tensor<384xf32>
    %v7681 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7682 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7683 = stablehlo.multiply %v7681, %s2b2nbtm : tensor<384xf32>
    %v7684 = stablehlo.multiply %v7682, %v2994 : tensor<384xf32>
    %v7685 = stablehlo.add %v7683, %v7684 : tensor<384xf32>
    %v7686 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7687 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7688 = stablehlo.multiply %v7686, %s2b2nbtv : tensor<384xf32>
    %v7689 = stablehlo.multiply %v2994, %v2994 : tensor<384xf32>
    %v7690 = stablehlo.multiply %v7687, %v7689 : tensor<384xf32>
    %v7691 = stablehlo.add %v7688, %v7690 : tensor<384xf32>
    %v7692 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7693 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7694 = stablehlo.multiply %v7692, %s2b2nbtm : tensor<384xf32>
    %v7695 = stablehlo.multiply %v7693, %v2994 : tensor<384xf32>
    %v7696 = stablehlo.add %v7694, %v7695 : tensor<384xf32>
    %v7697 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7698 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7699 = stablehlo.multiply %v7697, %s2b2nbtv : tensor<384xf32>
    %v7700 = stablehlo.multiply %v2994, %v2994 : tensor<384xf32>
    %v7701 = stablehlo.multiply %v7698, %v7700 : tensor<384xf32>
    %v7702 = stablehlo.add %v7699, %v7701 : tensor<384xf32>
    %v7703 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7704 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7705 = stablehlo.divide %v7696, %v7703 : tensor<384xf32>
    %v7706 = stablehlo.divide %v7702, %v7704 : tensor<384xf32>
    %v7707 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7708 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7709 = stablehlo.sqrt %v7706 : tensor<384xf32>
    %v7710 = stablehlo.add %v7709, %v7708 : tensor<384xf32>
    %v7711 = stablehlo.divide %v7705, %v7710 : tensor<384xf32>
    %v7712 = stablehlo.multiply %v7707, %v7711 : tensor<384xf32>
    %v7713 = stablehlo.subtract %s2b2nbt, %v7712 : tensor<384xf32>
    %v7714 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7715 = stablehlo.multiply %v7714, %v7707 : tensor<384xf32>
    %v7716 = stablehlo.multiply %v7715, %s2b2nbt : tensor<384xf32>
    %v7717 = stablehlo.subtract %v7713, %v7716 : tensor<384xf32>
    %v7718 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7719 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7720 = stablehlo.multiply %v7718, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v7721 = stablehlo.multiply %v7719, %v2961 : tensor<1536x384x1x1xf32>
    %v7722 = stablehlo.add %v7720, %v7721 : tensor<1536x384x1x1xf32>
    %v7723 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7724 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7725 = stablehlo.multiply %v7723, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v7726 = stablehlo.multiply %v2961, %v2961 : tensor<1536x384x1x1xf32>
    %v7727 = stablehlo.multiply %v7724, %v7726 : tensor<1536x384x1x1xf32>
    %v7728 = stablehlo.add %v7725, %v7727 : tensor<1536x384x1x1xf32>
    %v7729 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7730 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7731 = stablehlo.multiply %v7729, %s2b2eWm : tensor<1536x384x1x1xf32>
    %v7732 = stablehlo.multiply %v7730, %v2961 : tensor<1536x384x1x1xf32>
    %v7733 = stablehlo.add %v7731, %v7732 : tensor<1536x384x1x1xf32>
    %v7734 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7735 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7736 = stablehlo.multiply %v7734, %s2b2eWv : tensor<1536x384x1x1xf32>
    %v7737 = stablehlo.multiply %v2961, %v2961 : tensor<1536x384x1x1xf32>
    %v7738 = stablehlo.multiply %v7735, %v7737 : tensor<1536x384x1x1xf32>
    %v7739 = stablehlo.add %v7736, %v7738 : tensor<1536x384x1x1xf32>
    %v7740 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7741 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7742 = stablehlo.divide %v7733, %v7740 : tensor<1536x384x1x1xf32>
    %v7743 = stablehlo.divide %v7739, %v7741 : tensor<1536x384x1x1xf32>
    %v7744 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7745 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7746 = stablehlo.sqrt %v7743 : tensor<1536x384x1x1xf32>
    %v7747 = stablehlo.add %v7746, %v7745 : tensor<1536x384x1x1xf32>
    %v7748 = stablehlo.divide %v7742, %v7747 : tensor<1536x384x1x1xf32>
    %v7749 = stablehlo.multiply %v7744, %v7748 : tensor<1536x384x1x1xf32>
    %v7750 = stablehlo.subtract %s2b2eW, %v7749 : tensor<1536x384x1x1xf32>
    %v7751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v7752 = stablehlo.multiply %v7751, %v7744 : tensor<1536x384x1x1xf32>
    %v7753 = stablehlo.multiply %v7752, %s2b2eW : tensor<1536x384x1x1xf32>
    %v7754 = stablehlo.subtract %v7750, %v7753 : tensor<1536x384x1x1xf32>
    %v7755 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7756 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7757 = stablehlo.multiply %v7755, %s2b2ebm : tensor<1536xf32>
    %v7758 = stablehlo.multiply %v7756, %v2964 : tensor<1536xf32>
    %v7759 = stablehlo.add %v7757, %v7758 : tensor<1536xf32>
    %v7760 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7761 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7762 = stablehlo.multiply %v7760, %s2b2ebv : tensor<1536xf32>
    %v7763 = stablehlo.multiply %v2964, %v2964 : tensor<1536xf32>
    %v7764 = stablehlo.multiply %v7761, %v7763 : tensor<1536xf32>
    %v7765 = stablehlo.add %v7762, %v7764 : tensor<1536xf32>
    %v7766 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7767 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7768 = stablehlo.multiply %v7766, %s2b2ebm : tensor<1536xf32>
    %v7769 = stablehlo.multiply %v7767, %v2964 : tensor<1536xf32>
    %v7770 = stablehlo.add %v7768, %v7769 : tensor<1536xf32>
    %v7771 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7772 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7773 = stablehlo.multiply %v7771, %s2b2ebv : tensor<1536xf32>
    %v7774 = stablehlo.multiply %v2964, %v2964 : tensor<1536xf32>
    %v7775 = stablehlo.multiply %v7772, %v7774 : tensor<1536xf32>
    %v7776 = stablehlo.add %v7773, %v7775 : tensor<1536xf32>
    %v7777 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7778 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7779 = stablehlo.divide %v7770, %v7777 : tensor<1536xf32>
    %v7780 = stablehlo.divide %v7776, %v7778 : tensor<1536xf32>
    %v7781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7782 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7783 = stablehlo.sqrt %v7780 : tensor<1536xf32>
    %v7784 = stablehlo.add %v7783, %v7782 : tensor<1536xf32>
    %v7785 = stablehlo.divide %v7779, %v7784 : tensor<1536xf32>
    %v7786 = stablehlo.multiply %v7781, %v7785 : tensor<1536xf32>
    %v7787 = stablehlo.subtract %s2b2eb, %v7786 : tensor<1536xf32>
    %v7788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v7789 = stablehlo.multiply %v7788, %v7781 : tensor<1536xf32>
    %v7790 = stablehlo.multiply %v7789, %s2b2eb : tensor<1536xf32>
    %v7791 = stablehlo.subtract %v7787, %v7790 : tensor<1536xf32>
    %v7792 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7793 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7794 = stablehlo.multiply %v7792, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v7795 = stablehlo.multiply %v7793, %v2952 : tensor<384x1536x1x1xf32>
    %v7796 = stablehlo.add %v7794, %v7795 : tensor<384x1536x1x1xf32>
    %v7797 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7798 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7799 = stablehlo.multiply %v7797, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v7800 = stablehlo.multiply %v2952, %v2952 : tensor<384x1536x1x1xf32>
    %v7801 = stablehlo.multiply %v7798, %v7800 : tensor<384x1536x1x1xf32>
    %v7802 = stablehlo.add %v7799, %v7801 : tensor<384x1536x1x1xf32>
    %v7803 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7804 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7805 = stablehlo.multiply %v7803, %s2b2pWm : tensor<384x1536x1x1xf32>
    %v7806 = stablehlo.multiply %v7804, %v2952 : tensor<384x1536x1x1xf32>
    %v7807 = stablehlo.add %v7805, %v7806 : tensor<384x1536x1x1xf32>
    %v7808 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7809 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7810 = stablehlo.multiply %v7808, %s2b2pWv : tensor<384x1536x1x1xf32>
    %v7811 = stablehlo.multiply %v2952, %v2952 : tensor<384x1536x1x1xf32>
    %v7812 = stablehlo.multiply %v7809, %v7811 : tensor<384x1536x1x1xf32>
    %v7813 = stablehlo.add %v7810, %v7812 : tensor<384x1536x1x1xf32>
    %v7814 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7815 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7816 = stablehlo.divide %v7807, %v7814 : tensor<384x1536x1x1xf32>
    %v7817 = stablehlo.divide %v7813, %v7815 : tensor<384x1536x1x1xf32>
    %v7818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7819 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7820 = stablehlo.sqrt %v7817 : tensor<384x1536x1x1xf32>
    %v7821 = stablehlo.add %v7820, %v7819 : tensor<384x1536x1x1xf32>
    %v7822 = stablehlo.divide %v7816, %v7821 : tensor<384x1536x1x1xf32>
    %v7823 = stablehlo.multiply %v7818, %v7822 : tensor<384x1536x1x1xf32>
    %v7824 = stablehlo.subtract %s2b2pW, %v7823 : tensor<384x1536x1x1xf32>
    %v7825 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v7826 = stablehlo.multiply %v7825, %v7818 : tensor<384x1536x1x1xf32>
    %v7827 = stablehlo.multiply %v7826, %s2b2pW : tensor<384x1536x1x1xf32>
    %v7828 = stablehlo.subtract %v7824, %v7827 : tensor<384x1536x1x1xf32>
    %v7829 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7830 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7831 = stablehlo.multiply %v7829, %s2b2pbm : tensor<384xf32>
    %v7832 = stablehlo.multiply %v7830, %v2955 : tensor<384xf32>
    %v7833 = stablehlo.add %v7831, %v7832 : tensor<384xf32>
    %v7834 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7835 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7836 = stablehlo.multiply %v7834, %s2b2pbv : tensor<384xf32>
    %v7837 = stablehlo.multiply %v2955, %v2955 : tensor<384xf32>
    %v7838 = stablehlo.multiply %v7835, %v7837 : tensor<384xf32>
    %v7839 = stablehlo.add %v7836, %v7838 : tensor<384xf32>
    %v7840 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7841 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7842 = stablehlo.multiply %v7840, %s2b2pbm : tensor<384xf32>
    %v7843 = stablehlo.multiply %v7841, %v2955 : tensor<384xf32>
    %v7844 = stablehlo.add %v7842, %v7843 : tensor<384xf32>
    %v7845 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7846 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7847 = stablehlo.multiply %v7845, %s2b2pbv : tensor<384xf32>
    %v7848 = stablehlo.multiply %v2955, %v2955 : tensor<384xf32>
    %v7849 = stablehlo.multiply %v7846, %v7848 : tensor<384xf32>
    %v7850 = stablehlo.add %v7847, %v7849 : tensor<384xf32>
    %v7851 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7852 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7853 = stablehlo.divide %v7844, %v7851 : tensor<384xf32>
    %v7854 = stablehlo.divide %v7850, %v7852 : tensor<384xf32>
    %v7855 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7856 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7857 = stablehlo.sqrt %v7854 : tensor<384xf32>
    %v7858 = stablehlo.add %v7857, %v7856 : tensor<384xf32>
    %v7859 = stablehlo.divide %v7853, %v7858 : tensor<384xf32>
    %v7860 = stablehlo.multiply %v7855, %v7859 : tensor<384xf32>
    %v7861 = stablehlo.subtract %s2b2pb, %v7860 : tensor<384xf32>
    %v7862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7863 = stablehlo.multiply %v7862, %v7855 : tensor<384xf32>
    %v7864 = stablehlo.multiply %v7863, %s2b2pb : tensor<384xf32>
    %v7865 = stablehlo.subtract %v7861, %v7864 : tensor<384xf32>
    %v7866 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7867 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7868 = stablehlo.multiply %v7866, %s2b2lgm : tensor<384xf32>
    %v7869 = stablehlo.multiply %v7867, %v2946 : tensor<384xf32>
    %v7870 = stablehlo.add %v7868, %v7869 : tensor<384xf32>
    %v7871 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7872 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7873 = stablehlo.multiply %v7871, %s2b2lgv : tensor<384xf32>
    %v7874 = stablehlo.multiply %v2946, %v2946 : tensor<384xf32>
    %v7875 = stablehlo.multiply %v7872, %v7874 : tensor<384xf32>
    %v7876 = stablehlo.add %v7873, %v7875 : tensor<384xf32>
    %v7877 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7878 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7879 = stablehlo.multiply %v7877, %s2b2lgm : tensor<384xf32>
    %v7880 = stablehlo.multiply %v7878, %v2946 : tensor<384xf32>
    %v7881 = stablehlo.add %v7879, %v7880 : tensor<384xf32>
    %v7882 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7883 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7884 = stablehlo.multiply %v7882, %s2b2lgv : tensor<384xf32>
    %v7885 = stablehlo.multiply %v2946, %v2946 : tensor<384xf32>
    %v7886 = stablehlo.multiply %v7883, %v7885 : tensor<384xf32>
    %v7887 = stablehlo.add %v7884, %v7886 : tensor<384xf32>
    %v7888 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7889 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7890 = stablehlo.divide %v7881, %v7888 : tensor<384xf32>
    %v7891 = stablehlo.divide %v7887, %v7889 : tensor<384xf32>
    %v7892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7893 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7894 = stablehlo.sqrt %v7891 : tensor<384xf32>
    %v7895 = stablehlo.add %v7894, %v7893 : tensor<384xf32>
    %v7896 = stablehlo.divide %v7890, %v7895 : tensor<384xf32>
    %v7897 = stablehlo.multiply %v7892, %v7896 : tensor<384xf32>
    %v7898 = stablehlo.subtract %s2b2lg, %v7897 : tensor<384xf32>
    %v7899 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7900 = stablehlo.multiply %v7899, %v7892 : tensor<384xf32>
    %v7901 = stablehlo.multiply %v7900, %s2b2lg : tensor<384xf32>
    %v7902 = stablehlo.subtract %v7898, %v7901 : tensor<384xf32>
    %v7903 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7904 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7905 = stablehlo.multiply %v7903, %s2b3dWm : tensor<384x1x7x7xf32>
    %v7906 = stablehlo.multiply %v7904, %v2851 : tensor<384x1x7x7xf32>
    %v7907 = stablehlo.add %v7905, %v7906 : tensor<384x1x7x7xf32>
    %v7908 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7909 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7910 = stablehlo.multiply %v7908, %s2b3dWv : tensor<384x1x7x7xf32>
    %v7911 = stablehlo.multiply %v2851, %v2851 : tensor<384x1x7x7xf32>
    %v7912 = stablehlo.multiply %v7909, %v7911 : tensor<384x1x7x7xf32>
    %v7913 = stablehlo.add %v7910, %v7912 : tensor<384x1x7x7xf32>
    %v7914 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7915 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7916 = stablehlo.multiply %v7914, %s2b3dWm : tensor<384x1x7x7xf32>
    %v7917 = stablehlo.multiply %v7915, %v2851 : tensor<384x1x7x7xf32>
    %v7918 = stablehlo.add %v7916, %v7917 : tensor<384x1x7x7xf32>
    %v7919 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7920 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7921 = stablehlo.multiply %v7919, %s2b3dWv : tensor<384x1x7x7xf32>
    %v7922 = stablehlo.multiply %v2851, %v2851 : tensor<384x1x7x7xf32>
    %v7923 = stablehlo.multiply %v7920, %v7922 : tensor<384x1x7x7xf32>
    %v7924 = stablehlo.add %v7921, %v7923 : tensor<384x1x7x7xf32>
    %v7925 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7926 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7927 = stablehlo.divide %v7918, %v7925 : tensor<384x1x7x7xf32>
    %v7928 = stablehlo.divide %v7924, %v7926 : tensor<384x1x7x7xf32>
    %v7929 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7930 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7931 = stablehlo.sqrt %v7928 : tensor<384x1x7x7xf32>
    %v7932 = stablehlo.add %v7931, %v7930 : tensor<384x1x7x7xf32>
    %v7933 = stablehlo.divide %v7927, %v7932 : tensor<384x1x7x7xf32>
    %v7934 = stablehlo.multiply %v7929, %v7933 : tensor<384x1x7x7xf32>
    %v7935 = stablehlo.subtract %s2b3dW, %v7934 : tensor<384x1x7x7xf32>
    %v7936 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v7937 = stablehlo.multiply %v7936, %v7929 : tensor<384x1x7x7xf32>
    %v7938 = stablehlo.multiply %v7937, %s2b3dW : tensor<384x1x7x7xf32>
    %v7939 = stablehlo.subtract %v7935, %v7938 : tensor<384x1x7x7xf32>
    %v7940 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7941 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7942 = stablehlo.multiply %v7940, %s2b3dbm : tensor<384xf32>
    %v7943 = stablehlo.multiply %v7941, %v2854 : tensor<384xf32>
    %v7944 = stablehlo.add %v7942, %v7943 : tensor<384xf32>
    %v7945 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7946 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7947 = stablehlo.multiply %v7945, %s2b3dbv : tensor<384xf32>
    %v7948 = stablehlo.multiply %v2854, %v2854 : tensor<384xf32>
    %v7949 = stablehlo.multiply %v7946, %v7948 : tensor<384xf32>
    %v7950 = stablehlo.add %v7947, %v7949 : tensor<384xf32>
    %v7951 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7952 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7953 = stablehlo.multiply %v7951, %s2b3dbm : tensor<384xf32>
    %v7954 = stablehlo.multiply %v7952, %v2854 : tensor<384xf32>
    %v7955 = stablehlo.add %v7953, %v7954 : tensor<384xf32>
    %v7956 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7957 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7958 = stablehlo.multiply %v7956, %s2b3dbv : tensor<384xf32>
    %v7959 = stablehlo.multiply %v2854, %v2854 : tensor<384xf32>
    %v7960 = stablehlo.multiply %v7957, %v7959 : tensor<384xf32>
    %v7961 = stablehlo.add %v7958, %v7960 : tensor<384xf32>
    %v7962 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7963 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7964 = stablehlo.divide %v7955, %v7962 : tensor<384xf32>
    %v7965 = stablehlo.divide %v7961, %v7963 : tensor<384xf32>
    %v7966 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7967 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7968 = stablehlo.sqrt %v7965 : tensor<384xf32>
    %v7969 = stablehlo.add %v7968, %v7967 : tensor<384xf32>
    %v7970 = stablehlo.divide %v7964, %v7969 : tensor<384xf32>
    %v7971 = stablehlo.multiply %v7966, %v7970 : tensor<384xf32>
    %v7972 = stablehlo.subtract %s2b3db, %v7971 : tensor<384xf32>
    %v7973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7974 = stablehlo.multiply %v7973, %v7966 : tensor<384xf32>
    %v7975 = stablehlo.multiply %v7974, %s2b3db : tensor<384xf32>
    %v7976 = stablehlo.subtract %v7972, %v7975 : tensor<384xf32>
    %v7977 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7978 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7979 = stablehlo.multiply %v7977, %s2b3ngm : tensor<384xf32>
    %v7980 = stablehlo.multiply %v7978, %v2839 : tensor<384xf32>
    %v7981 = stablehlo.add %v7979, %v7980 : tensor<384xf32>
    %v7982 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7983 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7984 = stablehlo.multiply %v7982, %s2b3ngv : tensor<384xf32>
    %v7985 = stablehlo.multiply %v2839, %v2839 : tensor<384xf32>
    %v7986 = stablehlo.multiply %v7983, %v7985 : tensor<384xf32>
    %v7987 = stablehlo.add %v7984, %v7986 : tensor<384xf32>
    %v7988 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7989 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7990 = stablehlo.multiply %v7988, %s2b3ngm : tensor<384xf32>
    %v7991 = stablehlo.multiply %v7989, %v2839 : tensor<384xf32>
    %v7992 = stablehlo.add %v7990, %v7991 : tensor<384xf32>
    %v7993 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7994 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v7995 = stablehlo.multiply %v7993, %s2b3ngv : tensor<384xf32>
    %v7996 = stablehlo.multiply %v2839, %v2839 : tensor<384xf32>
    %v7997 = stablehlo.multiply %v7994, %v7996 : tensor<384xf32>
    %v7998 = stablehlo.add %v7995, %v7997 : tensor<384xf32>
    %v7999 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8000 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8001 = stablehlo.divide %v7992, %v7999 : tensor<384xf32>
    %v8002 = stablehlo.divide %v7998, %v8000 : tensor<384xf32>
    %v8003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8004 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8005 = stablehlo.sqrt %v8002 : tensor<384xf32>
    %v8006 = stablehlo.add %v8005, %v8004 : tensor<384xf32>
    %v8007 = stablehlo.divide %v8001, %v8006 : tensor<384xf32>
    %v8008 = stablehlo.multiply %v8003, %v8007 : tensor<384xf32>
    %v8009 = stablehlo.subtract %s2b3ng, %v8008 : tensor<384xf32>
    %v8010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8011 = stablehlo.multiply %v8010, %v8003 : tensor<384xf32>
    %v8012 = stablehlo.multiply %v8011, %s2b3ng : tensor<384xf32>
    %v8013 = stablehlo.subtract %v8009, %v8012 : tensor<384xf32>
    %v8014 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8015 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8016 = stablehlo.multiply %v8014, %s2b3nbtm : tensor<384xf32>
    %v8017 = stablehlo.multiply %v8015, %v2845 : tensor<384xf32>
    %v8018 = stablehlo.add %v8016, %v8017 : tensor<384xf32>
    %v8019 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8020 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8021 = stablehlo.multiply %v8019, %s2b3nbtv : tensor<384xf32>
    %v8022 = stablehlo.multiply %v2845, %v2845 : tensor<384xf32>
    %v8023 = stablehlo.multiply %v8020, %v8022 : tensor<384xf32>
    %v8024 = stablehlo.add %v8021, %v8023 : tensor<384xf32>
    %v8025 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8026 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8027 = stablehlo.multiply %v8025, %s2b3nbtm : tensor<384xf32>
    %v8028 = stablehlo.multiply %v8026, %v2845 : tensor<384xf32>
    %v8029 = stablehlo.add %v8027, %v8028 : tensor<384xf32>
    %v8030 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8031 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8032 = stablehlo.multiply %v8030, %s2b3nbtv : tensor<384xf32>
    %v8033 = stablehlo.multiply %v2845, %v2845 : tensor<384xf32>
    %v8034 = stablehlo.multiply %v8031, %v8033 : tensor<384xf32>
    %v8035 = stablehlo.add %v8032, %v8034 : tensor<384xf32>
    %v8036 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8037 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8038 = stablehlo.divide %v8029, %v8036 : tensor<384xf32>
    %v8039 = stablehlo.divide %v8035, %v8037 : tensor<384xf32>
    %v8040 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8041 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8042 = stablehlo.sqrt %v8039 : tensor<384xf32>
    %v8043 = stablehlo.add %v8042, %v8041 : tensor<384xf32>
    %v8044 = stablehlo.divide %v8038, %v8043 : tensor<384xf32>
    %v8045 = stablehlo.multiply %v8040, %v8044 : tensor<384xf32>
    %v8046 = stablehlo.subtract %s2b3nbt, %v8045 : tensor<384xf32>
    %v8047 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8048 = stablehlo.multiply %v8047, %v8040 : tensor<384xf32>
    %v8049 = stablehlo.multiply %v8048, %s2b3nbt : tensor<384xf32>
    %v8050 = stablehlo.subtract %v8046, %v8049 : tensor<384xf32>
    %v8051 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8052 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8053 = stablehlo.multiply %v8051, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v8054 = stablehlo.multiply %v8052, %v2812 : tensor<1536x384x1x1xf32>
    %v8055 = stablehlo.add %v8053, %v8054 : tensor<1536x384x1x1xf32>
    %v8056 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8057 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8058 = stablehlo.multiply %v8056, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v8059 = stablehlo.multiply %v2812, %v2812 : tensor<1536x384x1x1xf32>
    %v8060 = stablehlo.multiply %v8057, %v8059 : tensor<1536x384x1x1xf32>
    %v8061 = stablehlo.add %v8058, %v8060 : tensor<1536x384x1x1xf32>
    %v8062 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8063 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8064 = stablehlo.multiply %v8062, %s2b3eWm : tensor<1536x384x1x1xf32>
    %v8065 = stablehlo.multiply %v8063, %v2812 : tensor<1536x384x1x1xf32>
    %v8066 = stablehlo.add %v8064, %v8065 : tensor<1536x384x1x1xf32>
    %v8067 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8068 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8069 = stablehlo.multiply %v8067, %s2b3eWv : tensor<1536x384x1x1xf32>
    %v8070 = stablehlo.multiply %v2812, %v2812 : tensor<1536x384x1x1xf32>
    %v8071 = stablehlo.multiply %v8068, %v8070 : tensor<1536x384x1x1xf32>
    %v8072 = stablehlo.add %v8069, %v8071 : tensor<1536x384x1x1xf32>
    %v8073 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8074 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8075 = stablehlo.divide %v8066, %v8073 : tensor<1536x384x1x1xf32>
    %v8076 = stablehlo.divide %v8072, %v8074 : tensor<1536x384x1x1xf32>
    %v8077 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8078 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8079 = stablehlo.sqrt %v8076 : tensor<1536x384x1x1xf32>
    %v8080 = stablehlo.add %v8079, %v8078 : tensor<1536x384x1x1xf32>
    %v8081 = stablehlo.divide %v8075, %v8080 : tensor<1536x384x1x1xf32>
    %v8082 = stablehlo.multiply %v8077, %v8081 : tensor<1536x384x1x1xf32>
    %v8083 = stablehlo.subtract %s2b3eW, %v8082 : tensor<1536x384x1x1xf32>
    %v8084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8085 = stablehlo.multiply %v8084, %v8077 : tensor<1536x384x1x1xf32>
    %v8086 = stablehlo.multiply %v8085, %s2b3eW : tensor<1536x384x1x1xf32>
    %v8087 = stablehlo.subtract %v8083, %v8086 : tensor<1536x384x1x1xf32>
    %v8088 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8089 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8090 = stablehlo.multiply %v8088, %s2b3ebm : tensor<1536xf32>
    %v8091 = stablehlo.multiply %v8089, %v2815 : tensor<1536xf32>
    %v8092 = stablehlo.add %v8090, %v8091 : tensor<1536xf32>
    %v8093 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8094 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8095 = stablehlo.multiply %v8093, %s2b3ebv : tensor<1536xf32>
    %v8096 = stablehlo.multiply %v2815, %v2815 : tensor<1536xf32>
    %v8097 = stablehlo.multiply %v8094, %v8096 : tensor<1536xf32>
    %v8098 = stablehlo.add %v8095, %v8097 : tensor<1536xf32>
    %v8099 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8100 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8101 = stablehlo.multiply %v8099, %s2b3ebm : tensor<1536xf32>
    %v8102 = stablehlo.multiply %v8100, %v2815 : tensor<1536xf32>
    %v8103 = stablehlo.add %v8101, %v8102 : tensor<1536xf32>
    %v8104 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8105 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8106 = stablehlo.multiply %v8104, %s2b3ebv : tensor<1536xf32>
    %v8107 = stablehlo.multiply %v2815, %v2815 : tensor<1536xf32>
    %v8108 = stablehlo.multiply %v8105, %v8107 : tensor<1536xf32>
    %v8109 = stablehlo.add %v8106, %v8108 : tensor<1536xf32>
    %v8110 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8111 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8112 = stablehlo.divide %v8103, %v8110 : tensor<1536xf32>
    %v8113 = stablehlo.divide %v8109, %v8111 : tensor<1536xf32>
    %v8114 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8115 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8116 = stablehlo.sqrt %v8113 : tensor<1536xf32>
    %v8117 = stablehlo.add %v8116, %v8115 : tensor<1536xf32>
    %v8118 = stablehlo.divide %v8112, %v8117 : tensor<1536xf32>
    %v8119 = stablehlo.multiply %v8114, %v8118 : tensor<1536xf32>
    %v8120 = stablehlo.subtract %s2b3eb, %v8119 : tensor<1536xf32>
    %v8121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8122 = stablehlo.multiply %v8121, %v8114 : tensor<1536xf32>
    %v8123 = stablehlo.multiply %v8122, %s2b3eb : tensor<1536xf32>
    %v8124 = stablehlo.subtract %v8120, %v8123 : tensor<1536xf32>
    %v8125 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8126 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8127 = stablehlo.multiply %v8125, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v8128 = stablehlo.multiply %v8126, %v2803 : tensor<384x1536x1x1xf32>
    %v8129 = stablehlo.add %v8127, %v8128 : tensor<384x1536x1x1xf32>
    %v8130 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8131 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8132 = stablehlo.multiply %v8130, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v8133 = stablehlo.multiply %v2803, %v2803 : tensor<384x1536x1x1xf32>
    %v8134 = stablehlo.multiply %v8131, %v8133 : tensor<384x1536x1x1xf32>
    %v8135 = stablehlo.add %v8132, %v8134 : tensor<384x1536x1x1xf32>
    %v8136 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8137 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8138 = stablehlo.multiply %v8136, %s2b3pWm : tensor<384x1536x1x1xf32>
    %v8139 = stablehlo.multiply %v8137, %v2803 : tensor<384x1536x1x1xf32>
    %v8140 = stablehlo.add %v8138, %v8139 : tensor<384x1536x1x1xf32>
    %v8141 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8142 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8143 = stablehlo.multiply %v8141, %s2b3pWv : tensor<384x1536x1x1xf32>
    %v8144 = stablehlo.multiply %v2803, %v2803 : tensor<384x1536x1x1xf32>
    %v8145 = stablehlo.multiply %v8142, %v8144 : tensor<384x1536x1x1xf32>
    %v8146 = stablehlo.add %v8143, %v8145 : tensor<384x1536x1x1xf32>
    %v8147 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8148 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8149 = stablehlo.divide %v8140, %v8147 : tensor<384x1536x1x1xf32>
    %v8150 = stablehlo.divide %v8146, %v8148 : tensor<384x1536x1x1xf32>
    %v8151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8152 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8153 = stablehlo.sqrt %v8150 : tensor<384x1536x1x1xf32>
    %v8154 = stablehlo.add %v8153, %v8152 : tensor<384x1536x1x1xf32>
    %v8155 = stablehlo.divide %v8149, %v8154 : tensor<384x1536x1x1xf32>
    %v8156 = stablehlo.multiply %v8151, %v8155 : tensor<384x1536x1x1xf32>
    %v8157 = stablehlo.subtract %s2b3pW, %v8156 : tensor<384x1536x1x1xf32>
    %v8158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8159 = stablehlo.multiply %v8158, %v8151 : tensor<384x1536x1x1xf32>
    %v8160 = stablehlo.multiply %v8159, %s2b3pW : tensor<384x1536x1x1xf32>
    %v8161 = stablehlo.subtract %v8157, %v8160 : tensor<384x1536x1x1xf32>
    %v8162 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8163 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8164 = stablehlo.multiply %v8162, %s2b3pbm : tensor<384xf32>
    %v8165 = stablehlo.multiply %v8163, %v2806 : tensor<384xf32>
    %v8166 = stablehlo.add %v8164, %v8165 : tensor<384xf32>
    %v8167 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8168 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8169 = stablehlo.multiply %v8167, %s2b3pbv : tensor<384xf32>
    %v8170 = stablehlo.multiply %v2806, %v2806 : tensor<384xf32>
    %v8171 = stablehlo.multiply %v8168, %v8170 : tensor<384xf32>
    %v8172 = stablehlo.add %v8169, %v8171 : tensor<384xf32>
    %v8173 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8174 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8175 = stablehlo.multiply %v8173, %s2b3pbm : tensor<384xf32>
    %v8176 = stablehlo.multiply %v8174, %v2806 : tensor<384xf32>
    %v8177 = stablehlo.add %v8175, %v8176 : tensor<384xf32>
    %v8178 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8179 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8180 = stablehlo.multiply %v8178, %s2b3pbv : tensor<384xf32>
    %v8181 = stablehlo.multiply %v2806, %v2806 : tensor<384xf32>
    %v8182 = stablehlo.multiply %v8179, %v8181 : tensor<384xf32>
    %v8183 = stablehlo.add %v8180, %v8182 : tensor<384xf32>
    %v8184 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8185 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8186 = stablehlo.divide %v8177, %v8184 : tensor<384xf32>
    %v8187 = stablehlo.divide %v8183, %v8185 : tensor<384xf32>
    %v8188 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8189 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8190 = stablehlo.sqrt %v8187 : tensor<384xf32>
    %v8191 = stablehlo.add %v8190, %v8189 : tensor<384xf32>
    %v8192 = stablehlo.divide %v8186, %v8191 : tensor<384xf32>
    %v8193 = stablehlo.multiply %v8188, %v8192 : tensor<384xf32>
    %v8194 = stablehlo.subtract %s2b3pb, %v8193 : tensor<384xf32>
    %v8195 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8196 = stablehlo.multiply %v8195, %v8188 : tensor<384xf32>
    %v8197 = stablehlo.multiply %v8196, %s2b3pb : tensor<384xf32>
    %v8198 = stablehlo.subtract %v8194, %v8197 : tensor<384xf32>
    %v8199 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8200 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8201 = stablehlo.multiply %v8199, %s2b3lgm : tensor<384xf32>
    %v8202 = stablehlo.multiply %v8200, %v2797 : tensor<384xf32>
    %v8203 = stablehlo.add %v8201, %v8202 : tensor<384xf32>
    %v8204 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8205 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8206 = stablehlo.multiply %v8204, %s2b3lgv : tensor<384xf32>
    %v8207 = stablehlo.multiply %v2797, %v2797 : tensor<384xf32>
    %v8208 = stablehlo.multiply %v8205, %v8207 : tensor<384xf32>
    %v8209 = stablehlo.add %v8206, %v8208 : tensor<384xf32>
    %v8210 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8211 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8212 = stablehlo.multiply %v8210, %s2b3lgm : tensor<384xf32>
    %v8213 = stablehlo.multiply %v8211, %v2797 : tensor<384xf32>
    %v8214 = stablehlo.add %v8212, %v8213 : tensor<384xf32>
    %v8215 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8216 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8217 = stablehlo.multiply %v8215, %s2b3lgv : tensor<384xf32>
    %v8218 = stablehlo.multiply %v2797, %v2797 : tensor<384xf32>
    %v8219 = stablehlo.multiply %v8216, %v8218 : tensor<384xf32>
    %v8220 = stablehlo.add %v8217, %v8219 : tensor<384xf32>
    %v8221 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8222 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8223 = stablehlo.divide %v8214, %v8221 : tensor<384xf32>
    %v8224 = stablehlo.divide %v8220, %v8222 : tensor<384xf32>
    %v8225 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8226 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8227 = stablehlo.sqrt %v8224 : tensor<384xf32>
    %v8228 = stablehlo.add %v8227, %v8226 : tensor<384xf32>
    %v8229 = stablehlo.divide %v8223, %v8228 : tensor<384xf32>
    %v8230 = stablehlo.multiply %v8225, %v8229 : tensor<384xf32>
    %v8231 = stablehlo.subtract %s2b3lg, %v8230 : tensor<384xf32>
    %v8232 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8233 = stablehlo.multiply %v8232, %v8225 : tensor<384xf32>
    %v8234 = stablehlo.multiply %v8233, %s2b3lg : tensor<384xf32>
    %v8235 = stablehlo.subtract %v8231, %v8234 : tensor<384xf32>
    %v8236 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8237 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8238 = stablehlo.multiply %v8236, %s2b4dWm : tensor<384x1x7x7xf32>
    %v8239 = stablehlo.multiply %v8237, %v2702 : tensor<384x1x7x7xf32>
    %v8240 = stablehlo.add %v8238, %v8239 : tensor<384x1x7x7xf32>
    %v8241 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8242 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8243 = stablehlo.multiply %v8241, %s2b4dWv : tensor<384x1x7x7xf32>
    %v8244 = stablehlo.multiply %v2702, %v2702 : tensor<384x1x7x7xf32>
    %v8245 = stablehlo.multiply %v8242, %v8244 : tensor<384x1x7x7xf32>
    %v8246 = stablehlo.add %v8243, %v8245 : tensor<384x1x7x7xf32>
    %v8247 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8248 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8249 = stablehlo.multiply %v8247, %s2b4dWm : tensor<384x1x7x7xf32>
    %v8250 = stablehlo.multiply %v8248, %v2702 : tensor<384x1x7x7xf32>
    %v8251 = stablehlo.add %v8249, %v8250 : tensor<384x1x7x7xf32>
    %v8252 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8253 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8254 = stablehlo.multiply %v8252, %s2b4dWv : tensor<384x1x7x7xf32>
    %v8255 = stablehlo.multiply %v2702, %v2702 : tensor<384x1x7x7xf32>
    %v8256 = stablehlo.multiply %v8253, %v8255 : tensor<384x1x7x7xf32>
    %v8257 = stablehlo.add %v8254, %v8256 : tensor<384x1x7x7xf32>
    %v8258 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8259 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8260 = stablehlo.divide %v8251, %v8258 : tensor<384x1x7x7xf32>
    %v8261 = stablehlo.divide %v8257, %v8259 : tensor<384x1x7x7xf32>
    %v8262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8263 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8264 = stablehlo.sqrt %v8261 : tensor<384x1x7x7xf32>
    %v8265 = stablehlo.add %v8264, %v8263 : tensor<384x1x7x7xf32>
    %v8266 = stablehlo.divide %v8260, %v8265 : tensor<384x1x7x7xf32>
    %v8267 = stablehlo.multiply %v8262, %v8266 : tensor<384x1x7x7xf32>
    %v8268 = stablehlo.subtract %s2b4dW, %v8267 : tensor<384x1x7x7xf32>
    %v8269 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8270 = stablehlo.multiply %v8269, %v8262 : tensor<384x1x7x7xf32>
    %v8271 = stablehlo.multiply %v8270, %s2b4dW : tensor<384x1x7x7xf32>
    %v8272 = stablehlo.subtract %v8268, %v8271 : tensor<384x1x7x7xf32>
    %v8273 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8274 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8275 = stablehlo.multiply %v8273, %s2b4dbm : tensor<384xf32>
    %v8276 = stablehlo.multiply %v8274, %v2705 : tensor<384xf32>
    %v8277 = stablehlo.add %v8275, %v8276 : tensor<384xf32>
    %v8278 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8279 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8280 = stablehlo.multiply %v8278, %s2b4dbv : tensor<384xf32>
    %v8281 = stablehlo.multiply %v2705, %v2705 : tensor<384xf32>
    %v8282 = stablehlo.multiply %v8279, %v8281 : tensor<384xf32>
    %v8283 = stablehlo.add %v8280, %v8282 : tensor<384xf32>
    %v8284 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8285 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8286 = stablehlo.multiply %v8284, %s2b4dbm : tensor<384xf32>
    %v8287 = stablehlo.multiply %v8285, %v2705 : tensor<384xf32>
    %v8288 = stablehlo.add %v8286, %v8287 : tensor<384xf32>
    %v8289 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8290 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8291 = stablehlo.multiply %v8289, %s2b4dbv : tensor<384xf32>
    %v8292 = stablehlo.multiply %v2705, %v2705 : tensor<384xf32>
    %v8293 = stablehlo.multiply %v8290, %v8292 : tensor<384xf32>
    %v8294 = stablehlo.add %v8291, %v8293 : tensor<384xf32>
    %v8295 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8296 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8297 = stablehlo.divide %v8288, %v8295 : tensor<384xf32>
    %v8298 = stablehlo.divide %v8294, %v8296 : tensor<384xf32>
    %v8299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8300 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8301 = stablehlo.sqrt %v8298 : tensor<384xf32>
    %v8302 = stablehlo.add %v8301, %v8300 : tensor<384xf32>
    %v8303 = stablehlo.divide %v8297, %v8302 : tensor<384xf32>
    %v8304 = stablehlo.multiply %v8299, %v8303 : tensor<384xf32>
    %v8305 = stablehlo.subtract %s2b4db, %v8304 : tensor<384xf32>
    %v8306 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8307 = stablehlo.multiply %v8306, %v8299 : tensor<384xf32>
    %v8308 = stablehlo.multiply %v8307, %s2b4db : tensor<384xf32>
    %v8309 = stablehlo.subtract %v8305, %v8308 : tensor<384xf32>
    %v8310 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8311 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8312 = stablehlo.multiply %v8310, %s2b4ngm : tensor<384xf32>
    %v8313 = stablehlo.multiply %v8311, %v2690 : tensor<384xf32>
    %v8314 = stablehlo.add %v8312, %v8313 : tensor<384xf32>
    %v8315 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8316 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8317 = stablehlo.multiply %v8315, %s2b4ngv : tensor<384xf32>
    %v8318 = stablehlo.multiply %v2690, %v2690 : tensor<384xf32>
    %v8319 = stablehlo.multiply %v8316, %v8318 : tensor<384xf32>
    %v8320 = stablehlo.add %v8317, %v8319 : tensor<384xf32>
    %v8321 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8322 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8323 = stablehlo.multiply %v8321, %s2b4ngm : tensor<384xf32>
    %v8324 = stablehlo.multiply %v8322, %v2690 : tensor<384xf32>
    %v8325 = stablehlo.add %v8323, %v8324 : tensor<384xf32>
    %v8326 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8327 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8328 = stablehlo.multiply %v8326, %s2b4ngv : tensor<384xf32>
    %v8329 = stablehlo.multiply %v2690, %v2690 : tensor<384xf32>
    %v8330 = stablehlo.multiply %v8327, %v8329 : tensor<384xf32>
    %v8331 = stablehlo.add %v8328, %v8330 : tensor<384xf32>
    %v8332 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8333 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8334 = stablehlo.divide %v8325, %v8332 : tensor<384xf32>
    %v8335 = stablehlo.divide %v8331, %v8333 : tensor<384xf32>
    %v8336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8337 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8338 = stablehlo.sqrt %v8335 : tensor<384xf32>
    %v8339 = stablehlo.add %v8338, %v8337 : tensor<384xf32>
    %v8340 = stablehlo.divide %v8334, %v8339 : tensor<384xf32>
    %v8341 = stablehlo.multiply %v8336, %v8340 : tensor<384xf32>
    %v8342 = stablehlo.subtract %s2b4ng, %v8341 : tensor<384xf32>
    %v8343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8344 = stablehlo.multiply %v8343, %v8336 : tensor<384xf32>
    %v8345 = stablehlo.multiply %v8344, %s2b4ng : tensor<384xf32>
    %v8346 = stablehlo.subtract %v8342, %v8345 : tensor<384xf32>
    %v8347 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8348 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8349 = stablehlo.multiply %v8347, %s2b4nbtm : tensor<384xf32>
    %v8350 = stablehlo.multiply %v8348, %v2696 : tensor<384xf32>
    %v8351 = stablehlo.add %v8349, %v8350 : tensor<384xf32>
    %v8352 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8353 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8354 = stablehlo.multiply %v8352, %s2b4nbtv : tensor<384xf32>
    %v8355 = stablehlo.multiply %v2696, %v2696 : tensor<384xf32>
    %v8356 = stablehlo.multiply %v8353, %v8355 : tensor<384xf32>
    %v8357 = stablehlo.add %v8354, %v8356 : tensor<384xf32>
    %v8358 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8359 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8360 = stablehlo.multiply %v8358, %s2b4nbtm : tensor<384xf32>
    %v8361 = stablehlo.multiply %v8359, %v2696 : tensor<384xf32>
    %v8362 = stablehlo.add %v8360, %v8361 : tensor<384xf32>
    %v8363 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8364 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8365 = stablehlo.multiply %v8363, %s2b4nbtv : tensor<384xf32>
    %v8366 = stablehlo.multiply %v2696, %v2696 : tensor<384xf32>
    %v8367 = stablehlo.multiply %v8364, %v8366 : tensor<384xf32>
    %v8368 = stablehlo.add %v8365, %v8367 : tensor<384xf32>
    %v8369 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8370 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8371 = stablehlo.divide %v8362, %v8369 : tensor<384xf32>
    %v8372 = stablehlo.divide %v8368, %v8370 : tensor<384xf32>
    %v8373 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8374 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8375 = stablehlo.sqrt %v8372 : tensor<384xf32>
    %v8376 = stablehlo.add %v8375, %v8374 : tensor<384xf32>
    %v8377 = stablehlo.divide %v8371, %v8376 : tensor<384xf32>
    %v8378 = stablehlo.multiply %v8373, %v8377 : tensor<384xf32>
    %v8379 = stablehlo.subtract %s2b4nbt, %v8378 : tensor<384xf32>
    %v8380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8381 = stablehlo.multiply %v8380, %v8373 : tensor<384xf32>
    %v8382 = stablehlo.multiply %v8381, %s2b4nbt : tensor<384xf32>
    %v8383 = stablehlo.subtract %v8379, %v8382 : tensor<384xf32>
    %v8384 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8385 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8386 = stablehlo.multiply %v8384, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v8387 = stablehlo.multiply %v8385, %v2663 : tensor<1536x384x1x1xf32>
    %v8388 = stablehlo.add %v8386, %v8387 : tensor<1536x384x1x1xf32>
    %v8389 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8390 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8391 = stablehlo.multiply %v8389, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v8392 = stablehlo.multiply %v2663, %v2663 : tensor<1536x384x1x1xf32>
    %v8393 = stablehlo.multiply %v8390, %v8392 : tensor<1536x384x1x1xf32>
    %v8394 = stablehlo.add %v8391, %v8393 : tensor<1536x384x1x1xf32>
    %v8395 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8396 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8397 = stablehlo.multiply %v8395, %s2b4eWm : tensor<1536x384x1x1xf32>
    %v8398 = stablehlo.multiply %v8396, %v2663 : tensor<1536x384x1x1xf32>
    %v8399 = stablehlo.add %v8397, %v8398 : tensor<1536x384x1x1xf32>
    %v8400 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8401 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8402 = stablehlo.multiply %v8400, %s2b4eWv : tensor<1536x384x1x1xf32>
    %v8403 = stablehlo.multiply %v2663, %v2663 : tensor<1536x384x1x1xf32>
    %v8404 = stablehlo.multiply %v8401, %v8403 : tensor<1536x384x1x1xf32>
    %v8405 = stablehlo.add %v8402, %v8404 : tensor<1536x384x1x1xf32>
    %v8406 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8407 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8408 = stablehlo.divide %v8399, %v8406 : tensor<1536x384x1x1xf32>
    %v8409 = stablehlo.divide %v8405, %v8407 : tensor<1536x384x1x1xf32>
    %v8410 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8411 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8412 = stablehlo.sqrt %v8409 : tensor<1536x384x1x1xf32>
    %v8413 = stablehlo.add %v8412, %v8411 : tensor<1536x384x1x1xf32>
    %v8414 = stablehlo.divide %v8408, %v8413 : tensor<1536x384x1x1xf32>
    %v8415 = stablehlo.multiply %v8410, %v8414 : tensor<1536x384x1x1xf32>
    %v8416 = stablehlo.subtract %s2b4eW, %v8415 : tensor<1536x384x1x1xf32>
    %v8417 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8418 = stablehlo.multiply %v8417, %v8410 : tensor<1536x384x1x1xf32>
    %v8419 = stablehlo.multiply %v8418, %s2b4eW : tensor<1536x384x1x1xf32>
    %v8420 = stablehlo.subtract %v8416, %v8419 : tensor<1536x384x1x1xf32>
    %v8421 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8422 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8423 = stablehlo.multiply %v8421, %s2b4ebm : tensor<1536xf32>
    %v8424 = stablehlo.multiply %v8422, %v2666 : tensor<1536xf32>
    %v8425 = stablehlo.add %v8423, %v8424 : tensor<1536xf32>
    %v8426 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8427 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8428 = stablehlo.multiply %v8426, %s2b4ebv : tensor<1536xf32>
    %v8429 = stablehlo.multiply %v2666, %v2666 : tensor<1536xf32>
    %v8430 = stablehlo.multiply %v8427, %v8429 : tensor<1536xf32>
    %v8431 = stablehlo.add %v8428, %v8430 : tensor<1536xf32>
    %v8432 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8433 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8434 = stablehlo.multiply %v8432, %s2b4ebm : tensor<1536xf32>
    %v8435 = stablehlo.multiply %v8433, %v2666 : tensor<1536xf32>
    %v8436 = stablehlo.add %v8434, %v8435 : tensor<1536xf32>
    %v8437 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8438 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8439 = stablehlo.multiply %v8437, %s2b4ebv : tensor<1536xf32>
    %v8440 = stablehlo.multiply %v2666, %v2666 : tensor<1536xf32>
    %v8441 = stablehlo.multiply %v8438, %v8440 : tensor<1536xf32>
    %v8442 = stablehlo.add %v8439, %v8441 : tensor<1536xf32>
    %v8443 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8444 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8445 = stablehlo.divide %v8436, %v8443 : tensor<1536xf32>
    %v8446 = stablehlo.divide %v8442, %v8444 : tensor<1536xf32>
    %v8447 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8448 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8449 = stablehlo.sqrt %v8446 : tensor<1536xf32>
    %v8450 = stablehlo.add %v8449, %v8448 : tensor<1536xf32>
    %v8451 = stablehlo.divide %v8445, %v8450 : tensor<1536xf32>
    %v8452 = stablehlo.multiply %v8447, %v8451 : tensor<1536xf32>
    %v8453 = stablehlo.subtract %s2b4eb, %v8452 : tensor<1536xf32>
    %v8454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8455 = stablehlo.multiply %v8454, %v8447 : tensor<1536xf32>
    %v8456 = stablehlo.multiply %v8455, %s2b4eb : tensor<1536xf32>
    %v8457 = stablehlo.subtract %v8453, %v8456 : tensor<1536xf32>
    %v8458 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8459 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8460 = stablehlo.multiply %v8458, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v8461 = stablehlo.multiply %v8459, %v2654 : tensor<384x1536x1x1xf32>
    %v8462 = stablehlo.add %v8460, %v8461 : tensor<384x1536x1x1xf32>
    %v8463 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8464 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8465 = stablehlo.multiply %v8463, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v8466 = stablehlo.multiply %v2654, %v2654 : tensor<384x1536x1x1xf32>
    %v8467 = stablehlo.multiply %v8464, %v8466 : tensor<384x1536x1x1xf32>
    %v8468 = stablehlo.add %v8465, %v8467 : tensor<384x1536x1x1xf32>
    %v8469 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8470 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8471 = stablehlo.multiply %v8469, %s2b4pWm : tensor<384x1536x1x1xf32>
    %v8472 = stablehlo.multiply %v8470, %v2654 : tensor<384x1536x1x1xf32>
    %v8473 = stablehlo.add %v8471, %v8472 : tensor<384x1536x1x1xf32>
    %v8474 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8475 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8476 = stablehlo.multiply %v8474, %s2b4pWv : tensor<384x1536x1x1xf32>
    %v8477 = stablehlo.multiply %v2654, %v2654 : tensor<384x1536x1x1xf32>
    %v8478 = stablehlo.multiply %v8475, %v8477 : tensor<384x1536x1x1xf32>
    %v8479 = stablehlo.add %v8476, %v8478 : tensor<384x1536x1x1xf32>
    %v8480 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8481 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8482 = stablehlo.divide %v8473, %v8480 : tensor<384x1536x1x1xf32>
    %v8483 = stablehlo.divide %v8479, %v8481 : tensor<384x1536x1x1xf32>
    %v8484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8485 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8486 = stablehlo.sqrt %v8483 : tensor<384x1536x1x1xf32>
    %v8487 = stablehlo.add %v8486, %v8485 : tensor<384x1536x1x1xf32>
    %v8488 = stablehlo.divide %v8482, %v8487 : tensor<384x1536x1x1xf32>
    %v8489 = stablehlo.multiply %v8484, %v8488 : tensor<384x1536x1x1xf32>
    %v8490 = stablehlo.subtract %s2b4pW, %v8489 : tensor<384x1536x1x1xf32>
    %v8491 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8492 = stablehlo.multiply %v8491, %v8484 : tensor<384x1536x1x1xf32>
    %v8493 = stablehlo.multiply %v8492, %s2b4pW : tensor<384x1536x1x1xf32>
    %v8494 = stablehlo.subtract %v8490, %v8493 : tensor<384x1536x1x1xf32>
    %v8495 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8496 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8497 = stablehlo.multiply %v8495, %s2b4pbm : tensor<384xf32>
    %v8498 = stablehlo.multiply %v8496, %v2657 : tensor<384xf32>
    %v8499 = stablehlo.add %v8497, %v8498 : tensor<384xf32>
    %v8500 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8501 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8502 = stablehlo.multiply %v8500, %s2b4pbv : tensor<384xf32>
    %v8503 = stablehlo.multiply %v2657, %v2657 : tensor<384xf32>
    %v8504 = stablehlo.multiply %v8501, %v8503 : tensor<384xf32>
    %v8505 = stablehlo.add %v8502, %v8504 : tensor<384xf32>
    %v8506 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8507 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8508 = stablehlo.multiply %v8506, %s2b4pbm : tensor<384xf32>
    %v8509 = stablehlo.multiply %v8507, %v2657 : tensor<384xf32>
    %v8510 = stablehlo.add %v8508, %v8509 : tensor<384xf32>
    %v8511 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8512 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8513 = stablehlo.multiply %v8511, %s2b4pbv : tensor<384xf32>
    %v8514 = stablehlo.multiply %v2657, %v2657 : tensor<384xf32>
    %v8515 = stablehlo.multiply %v8512, %v8514 : tensor<384xf32>
    %v8516 = stablehlo.add %v8513, %v8515 : tensor<384xf32>
    %v8517 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8518 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8519 = stablehlo.divide %v8510, %v8517 : tensor<384xf32>
    %v8520 = stablehlo.divide %v8516, %v8518 : tensor<384xf32>
    %v8521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8522 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8523 = stablehlo.sqrt %v8520 : tensor<384xf32>
    %v8524 = stablehlo.add %v8523, %v8522 : tensor<384xf32>
    %v8525 = stablehlo.divide %v8519, %v8524 : tensor<384xf32>
    %v8526 = stablehlo.multiply %v8521, %v8525 : tensor<384xf32>
    %v8527 = stablehlo.subtract %s2b4pb, %v8526 : tensor<384xf32>
    %v8528 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8529 = stablehlo.multiply %v8528, %v8521 : tensor<384xf32>
    %v8530 = stablehlo.multiply %v8529, %s2b4pb : tensor<384xf32>
    %v8531 = stablehlo.subtract %v8527, %v8530 : tensor<384xf32>
    %v8532 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8533 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8534 = stablehlo.multiply %v8532, %s2b4lgm : tensor<384xf32>
    %v8535 = stablehlo.multiply %v8533, %v2648 : tensor<384xf32>
    %v8536 = stablehlo.add %v8534, %v8535 : tensor<384xf32>
    %v8537 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8538 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8539 = stablehlo.multiply %v8537, %s2b4lgv : tensor<384xf32>
    %v8540 = stablehlo.multiply %v2648, %v2648 : tensor<384xf32>
    %v8541 = stablehlo.multiply %v8538, %v8540 : tensor<384xf32>
    %v8542 = stablehlo.add %v8539, %v8541 : tensor<384xf32>
    %v8543 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8544 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8545 = stablehlo.multiply %v8543, %s2b4lgm : tensor<384xf32>
    %v8546 = stablehlo.multiply %v8544, %v2648 : tensor<384xf32>
    %v8547 = stablehlo.add %v8545, %v8546 : tensor<384xf32>
    %v8548 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8549 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8550 = stablehlo.multiply %v8548, %s2b4lgv : tensor<384xf32>
    %v8551 = stablehlo.multiply %v2648, %v2648 : tensor<384xf32>
    %v8552 = stablehlo.multiply %v8549, %v8551 : tensor<384xf32>
    %v8553 = stablehlo.add %v8550, %v8552 : tensor<384xf32>
    %v8554 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8555 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8556 = stablehlo.divide %v8547, %v8554 : tensor<384xf32>
    %v8557 = stablehlo.divide %v8553, %v8555 : tensor<384xf32>
    %v8558 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8559 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8560 = stablehlo.sqrt %v8557 : tensor<384xf32>
    %v8561 = stablehlo.add %v8560, %v8559 : tensor<384xf32>
    %v8562 = stablehlo.divide %v8556, %v8561 : tensor<384xf32>
    %v8563 = stablehlo.multiply %v8558, %v8562 : tensor<384xf32>
    %v8564 = stablehlo.subtract %s2b4lg, %v8563 : tensor<384xf32>
    %v8565 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8566 = stablehlo.multiply %v8565, %v8558 : tensor<384xf32>
    %v8567 = stablehlo.multiply %v8566, %s2b4lg : tensor<384xf32>
    %v8568 = stablehlo.subtract %v8564, %v8567 : tensor<384xf32>
    %v8569 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8570 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8571 = stablehlo.multiply %v8569, %s2b5dWm : tensor<384x1x7x7xf32>
    %v8572 = stablehlo.multiply %v8570, %v2553 : tensor<384x1x7x7xf32>
    %v8573 = stablehlo.add %v8571, %v8572 : tensor<384x1x7x7xf32>
    %v8574 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8575 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8576 = stablehlo.multiply %v8574, %s2b5dWv : tensor<384x1x7x7xf32>
    %v8577 = stablehlo.multiply %v2553, %v2553 : tensor<384x1x7x7xf32>
    %v8578 = stablehlo.multiply %v8575, %v8577 : tensor<384x1x7x7xf32>
    %v8579 = stablehlo.add %v8576, %v8578 : tensor<384x1x7x7xf32>
    %v8580 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8581 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8582 = stablehlo.multiply %v8580, %s2b5dWm : tensor<384x1x7x7xf32>
    %v8583 = stablehlo.multiply %v8581, %v2553 : tensor<384x1x7x7xf32>
    %v8584 = stablehlo.add %v8582, %v8583 : tensor<384x1x7x7xf32>
    %v8585 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8586 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8587 = stablehlo.multiply %v8585, %s2b5dWv : tensor<384x1x7x7xf32>
    %v8588 = stablehlo.multiply %v2553, %v2553 : tensor<384x1x7x7xf32>
    %v8589 = stablehlo.multiply %v8586, %v8588 : tensor<384x1x7x7xf32>
    %v8590 = stablehlo.add %v8587, %v8589 : tensor<384x1x7x7xf32>
    %v8591 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8592 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8593 = stablehlo.divide %v8584, %v8591 : tensor<384x1x7x7xf32>
    %v8594 = stablehlo.divide %v8590, %v8592 : tensor<384x1x7x7xf32>
    %v8595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8596 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8597 = stablehlo.sqrt %v8594 : tensor<384x1x7x7xf32>
    %v8598 = stablehlo.add %v8597, %v8596 : tensor<384x1x7x7xf32>
    %v8599 = stablehlo.divide %v8593, %v8598 : tensor<384x1x7x7xf32>
    %v8600 = stablehlo.multiply %v8595, %v8599 : tensor<384x1x7x7xf32>
    %v8601 = stablehlo.subtract %s2b5dW, %v8600 : tensor<384x1x7x7xf32>
    %v8602 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8603 = stablehlo.multiply %v8602, %v8595 : tensor<384x1x7x7xf32>
    %v8604 = stablehlo.multiply %v8603, %s2b5dW : tensor<384x1x7x7xf32>
    %v8605 = stablehlo.subtract %v8601, %v8604 : tensor<384x1x7x7xf32>
    %v8606 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8607 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8608 = stablehlo.multiply %v8606, %s2b5dbm : tensor<384xf32>
    %v8609 = stablehlo.multiply %v8607, %v2556 : tensor<384xf32>
    %v8610 = stablehlo.add %v8608, %v8609 : tensor<384xf32>
    %v8611 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8612 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8613 = stablehlo.multiply %v8611, %s2b5dbv : tensor<384xf32>
    %v8614 = stablehlo.multiply %v2556, %v2556 : tensor<384xf32>
    %v8615 = stablehlo.multiply %v8612, %v8614 : tensor<384xf32>
    %v8616 = stablehlo.add %v8613, %v8615 : tensor<384xf32>
    %v8617 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8618 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8619 = stablehlo.multiply %v8617, %s2b5dbm : tensor<384xf32>
    %v8620 = stablehlo.multiply %v8618, %v2556 : tensor<384xf32>
    %v8621 = stablehlo.add %v8619, %v8620 : tensor<384xf32>
    %v8622 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8623 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8624 = stablehlo.multiply %v8622, %s2b5dbv : tensor<384xf32>
    %v8625 = stablehlo.multiply %v2556, %v2556 : tensor<384xf32>
    %v8626 = stablehlo.multiply %v8623, %v8625 : tensor<384xf32>
    %v8627 = stablehlo.add %v8624, %v8626 : tensor<384xf32>
    %v8628 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8629 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8630 = stablehlo.divide %v8621, %v8628 : tensor<384xf32>
    %v8631 = stablehlo.divide %v8627, %v8629 : tensor<384xf32>
    %v8632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8633 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8634 = stablehlo.sqrt %v8631 : tensor<384xf32>
    %v8635 = stablehlo.add %v8634, %v8633 : tensor<384xf32>
    %v8636 = stablehlo.divide %v8630, %v8635 : tensor<384xf32>
    %v8637 = stablehlo.multiply %v8632, %v8636 : tensor<384xf32>
    %v8638 = stablehlo.subtract %s2b5db, %v8637 : tensor<384xf32>
    %v8639 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8640 = stablehlo.multiply %v8639, %v8632 : tensor<384xf32>
    %v8641 = stablehlo.multiply %v8640, %s2b5db : tensor<384xf32>
    %v8642 = stablehlo.subtract %v8638, %v8641 : tensor<384xf32>
    %v8643 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8644 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8645 = stablehlo.multiply %v8643, %s2b5ngm : tensor<384xf32>
    %v8646 = stablehlo.multiply %v8644, %v2541 : tensor<384xf32>
    %v8647 = stablehlo.add %v8645, %v8646 : tensor<384xf32>
    %v8648 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8649 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8650 = stablehlo.multiply %v8648, %s2b5ngv : tensor<384xf32>
    %v8651 = stablehlo.multiply %v2541, %v2541 : tensor<384xf32>
    %v8652 = stablehlo.multiply %v8649, %v8651 : tensor<384xf32>
    %v8653 = stablehlo.add %v8650, %v8652 : tensor<384xf32>
    %v8654 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8655 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8656 = stablehlo.multiply %v8654, %s2b5ngm : tensor<384xf32>
    %v8657 = stablehlo.multiply %v8655, %v2541 : tensor<384xf32>
    %v8658 = stablehlo.add %v8656, %v8657 : tensor<384xf32>
    %v8659 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8660 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8661 = stablehlo.multiply %v8659, %s2b5ngv : tensor<384xf32>
    %v8662 = stablehlo.multiply %v2541, %v2541 : tensor<384xf32>
    %v8663 = stablehlo.multiply %v8660, %v8662 : tensor<384xf32>
    %v8664 = stablehlo.add %v8661, %v8663 : tensor<384xf32>
    %v8665 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8666 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8667 = stablehlo.divide %v8658, %v8665 : tensor<384xf32>
    %v8668 = stablehlo.divide %v8664, %v8666 : tensor<384xf32>
    %v8669 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8670 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8671 = stablehlo.sqrt %v8668 : tensor<384xf32>
    %v8672 = stablehlo.add %v8671, %v8670 : tensor<384xf32>
    %v8673 = stablehlo.divide %v8667, %v8672 : tensor<384xf32>
    %v8674 = stablehlo.multiply %v8669, %v8673 : tensor<384xf32>
    %v8675 = stablehlo.subtract %s2b5ng, %v8674 : tensor<384xf32>
    %v8676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8677 = stablehlo.multiply %v8676, %v8669 : tensor<384xf32>
    %v8678 = stablehlo.multiply %v8677, %s2b5ng : tensor<384xf32>
    %v8679 = stablehlo.subtract %v8675, %v8678 : tensor<384xf32>
    %v8680 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8681 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8682 = stablehlo.multiply %v8680, %s2b5nbtm : tensor<384xf32>
    %v8683 = stablehlo.multiply %v8681, %v2547 : tensor<384xf32>
    %v8684 = stablehlo.add %v8682, %v8683 : tensor<384xf32>
    %v8685 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8686 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8687 = stablehlo.multiply %v8685, %s2b5nbtv : tensor<384xf32>
    %v8688 = stablehlo.multiply %v2547, %v2547 : tensor<384xf32>
    %v8689 = stablehlo.multiply %v8686, %v8688 : tensor<384xf32>
    %v8690 = stablehlo.add %v8687, %v8689 : tensor<384xf32>
    %v8691 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8692 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8693 = stablehlo.multiply %v8691, %s2b5nbtm : tensor<384xf32>
    %v8694 = stablehlo.multiply %v8692, %v2547 : tensor<384xf32>
    %v8695 = stablehlo.add %v8693, %v8694 : tensor<384xf32>
    %v8696 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8697 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8698 = stablehlo.multiply %v8696, %s2b5nbtv : tensor<384xf32>
    %v8699 = stablehlo.multiply %v2547, %v2547 : tensor<384xf32>
    %v8700 = stablehlo.multiply %v8697, %v8699 : tensor<384xf32>
    %v8701 = stablehlo.add %v8698, %v8700 : tensor<384xf32>
    %v8702 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8703 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8704 = stablehlo.divide %v8695, %v8702 : tensor<384xf32>
    %v8705 = stablehlo.divide %v8701, %v8703 : tensor<384xf32>
    %v8706 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8707 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8708 = stablehlo.sqrt %v8705 : tensor<384xf32>
    %v8709 = stablehlo.add %v8708, %v8707 : tensor<384xf32>
    %v8710 = stablehlo.divide %v8704, %v8709 : tensor<384xf32>
    %v8711 = stablehlo.multiply %v8706, %v8710 : tensor<384xf32>
    %v8712 = stablehlo.subtract %s2b5nbt, %v8711 : tensor<384xf32>
    %v8713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8714 = stablehlo.multiply %v8713, %v8706 : tensor<384xf32>
    %v8715 = stablehlo.multiply %v8714, %s2b5nbt : tensor<384xf32>
    %v8716 = stablehlo.subtract %v8712, %v8715 : tensor<384xf32>
    %v8717 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8718 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8719 = stablehlo.multiply %v8717, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v8720 = stablehlo.multiply %v8718, %v2514 : tensor<1536x384x1x1xf32>
    %v8721 = stablehlo.add %v8719, %v8720 : tensor<1536x384x1x1xf32>
    %v8722 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8723 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8724 = stablehlo.multiply %v8722, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v8725 = stablehlo.multiply %v2514, %v2514 : tensor<1536x384x1x1xf32>
    %v8726 = stablehlo.multiply %v8723, %v8725 : tensor<1536x384x1x1xf32>
    %v8727 = stablehlo.add %v8724, %v8726 : tensor<1536x384x1x1xf32>
    %v8728 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8729 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8730 = stablehlo.multiply %v8728, %s2b5eWm : tensor<1536x384x1x1xf32>
    %v8731 = stablehlo.multiply %v8729, %v2514 : tensor<1536x384x1x1xf32>
    %v8732 = stablehlo.add %v8730, %v8731 : tensor<1536x384x1x1xf32>
    %v8733 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8734 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8735 = stablehlo.multiply %v8733, %s2b5eWv : tensor<1536x384x1x1xf32>
    %v8736 = stablehlo.multiply %v2514, %v2514 : tensor<1536x384x1x1xf32>
    %v8737 = stablehlo.multiply %v8734, %v8736 : tensor<1536x384x1x1xf32>
    %v8738 = stablehlo.add %v8735, %v8737 : tensor<1536x384x1x1xf32>
    %v8739 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8740 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8741 = stablehlo.divide %v8732, %v8739 : tensor<1536x384x1x1xf32>
    %v8742 = stablehlo.divide %v8738, %v8740 : tensor<1536x384x1x1xf32>
    %v8743 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8744 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8745 = stablehlo.sqrt %v8742 : tensor<1536x384x1x1xf32>
    %v8746 = stablehlo.add %v8745, %v8744 : tensor<1536x384x1x1xf32>
    %v8747 = stablehlo.divide %v8741, %v8746 : tensor<1536x384x1x1xf32>
    %v8748 = stablehlo.multiply %v8743, %v8747 : tensor<1536x384x1x1xf32>
    %v8749 = stablehlo.subtract %s2b5eW, %v8748 : tensor<1536x384x1x1xf32>
    %v8750 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v8751 = stablehlo.multiply %v8750, %v8743 : tensor<1536x384x1x1xf32>
    %v8752 = stablehlo.multiply %v8751, %s2b5eW : tensor<1536x384x1x1xf32>
    %v8753 = stablehlo.subtract %v8749, %v8752 : tensor<1536x384x1x1xf32>
    %v8754 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8755 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8756 = stablehlo.multiply %v8754, %s2b5ebm : tensor<1536xf32>
    %v8757 = stablehlo.multiply %v8755, %v2517 : tensor<1536xf32>
    %v8758 = stablehlo.add %v8756, %v8757 : tensor<1536xf32>
    %v8759 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8760 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8761 = stablehlo.multiply %v8759, %s2b5ebv : tensor<1536xf32>
    %v8762 = stablehlo.multiply %v2517, %v2517 : tensor<1536xf32>
    %v8763 = stablehlo.multiply %v8760, %v8762 : tensor<1536xf32>
    %v8764 = stablehlo.add %v8761, %v8763 : tensor<1536xf32>
    %v8765 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8766 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8767 = stablehlo.multiply %v8765, %s2b5ebm : tensor<1536xf32>
    %v8768 = stablehlo.multiply %v8766, %v2517 : tensor<1536xf32>
    %v8769 = stablehlo.add %v8767, %v8768 : tensor<1536xf32>
    %v8770 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8771 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8772 = stablehlo.multiply %v8770, %s2b5ebv : tensor<1536xf32>
    %v8773 = stablehlo.multiply %v2517, %v2517 : tensor<1536xf32>
    %v8774 = stablehlo.multiply %v8771, %v8773 : tensor<1536xf32>
    %v8775 = stablehlo.add %v8772, %v8774 : tensor<1536xf32>
    %v8776 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8777 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8778 = stablehlo.divide %v8769, %v8776 : tensor<1536xf32>
    %v8779 = stablehlo.divide %v8775, %v8777 : tensor<1536xf32>
    %v8780 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8781 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8782 = stablehlo.sqrt %v8779 : tensor<1536xf32>
    %v8783 = stablehlo.add %v8782, %v8781 : tensor<1536xf32>
    %v8784 = stablehlo.divide %v8778, %v8783 : tensor<1536xf32>
    %v8785 = stablehlo.multiply %v8780, %v8784 : tensor<1536xf32>
    %v8786 = stablehlo.subtract %s2b5eb, %v8785 : tensor<1536xf32>
    %v8787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v8788 = stablehlo.multiply %v8787, %v8780 : tensor<1536xf32>
    %v8789 = stablehlo.multiply %v8788, %s2b5eb : tensor<1536xf32>
    %v8790 = stablehlo.subtract %v8786, %v8789 : tensor<1536xf32>
    %v8791 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8792 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8793 = stablehlo.multiply %v8791, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v8794 = stablehlo.multiply %v8792, %v2505 : tensor<384x1536x1x1xf32>
    %v8795 = stablehlo.add %v8793, %v8794 : tensor<384x1536x1x1xf32>
    %v8796 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8797 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8798 = stablehlo.multiply %v8796, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v8799 = stablehlo.multiply %v2505, %v2505 : tensor<384x1536x1x1xf32>
    %v8800 = stablehlo.multiply %v8797, %v8799 : tensor<384x1536x1x1xf32>
    %v8801 = stablehlo.add %v8798, %v8800 : tensor<384x1536x1x1xf32>
    %v8802 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8803 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8804 = stablehlo.multiply %v8802, %s2b5pWm : tensor<384x1536x1x1xf32>
    %v8805 = stablehlo.multiply %v8803, %v2505 : tensor<384x1536x1x1xf32>
    %v8806 = stablehlo.add %v8804, %v8805 : tensor<384x1536x1x1xf32>
    %v8807 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8808 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8809 = stablehlo.multiply %v8807, %s2b5pWv : tensor<384x1536x1x1xf32>
    %v8810 = stablehlo.multiply %v2505, %v2505 : tensor<384x1536x1x1xf32>
    %v8811 = stablehlo.multiply %v8808, %v8810 : tensor<384x1536x1x1xf32>
    %v8812 = stablehlo.add %v8809, %v8811 : tensor<384x1536x1x1xf32>
    %v8813 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8814 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8815 = stablehlo.divide %v8806, %v8813 : tensor<384x1536x1x1xf32>
    %v8816 = stablehlo.divide %v8812, %v8814 : tensor<384x1536x1x1xf32>
    %v8817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8818 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8819 = stablehlo.sqrt %v8816 : tensor<384x1536x1x1xf32>
    %v8820 = stablehlo.add %v8819, %v8818 : tensor<384x1536x1x1xf32>
    %v8821 = stablehlo.divide %v8815, %v8820 : tensor<384x1536x1x1xf32>
    %v8822 = stablehlo.multiply %v8817, %v8821 : tensor<384x1536x1x1xf32>
    %v8823 = stablehlo.subtract %s2b5pW, %v8822 : tensor<384x1536x1x1xf32>
    %v8824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v8825 = stablehlo.multiply %v8824, %v8817 : tensor<384x1536x1x1xf32>
    %v8826 = stablehlo.multiply %v8825, %s2b5pW : tensor<384x1536x1x1xf32>
    %v8827 = stablehlo.subtract %v8823, %v8826 : tensor<384x1536x1x1xf32>
    %v8828 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8829 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8830 = stablehlo.multiply %v8828, %s2b5pbm : tensor<384xf32>
    %v8831 = stablehlo.multiply %v8829, %v2508 : tensor<384xf32>
    %v8832 = stablehlo.add %v8830, %v8831 : tensor<384xf32>
    %v8833 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8834 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8835 = stablehlo.multiply %v8833, %s2b5pbv : tensor<384xf32>
    %v8836 = stablehlo.multiply %v2508, %v2508 : tensor<384xf32>
    %v8837 = stablehlo.multiply %v8834, %v8836 : tensor<384xf32>
    %v8838 = stablehlo.add %v8835, %v8837 : tensor<384xf32>
    %v8839 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8840 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8841 = stablehlo.multiply %v8839, %s2b5pbm : tensor<384xf32>
    %v8842 = stablehlo.multiply %v8840, %v2508 : tensor<384xf32>
    %v8843 = stablehlo.add %v8841, %v8842 : tensor<384xf32>
    %v8844 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8845 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8846 = stablehlo.multiply %v8844, %s2b5pbv : tensor<384xf32>
    %v8847 = stablehlo.multiply %v2508, %v2508 : tensor<384xf32>
    %v8848 = stablehlo.multiply %v8845, %v8847 : tensor<384xf32>
    %v8849 = stablehlo.add %v8846, %v8848 : tensor<384xf32>
    %v8850 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8851 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8852 = stablehlo.divide %v8843, %v8850 : tensor<384xf32>
    %v8853 = stablehlo.divide %v8849, %v8851 : tensor<384xf32>
    %v8854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8855 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8856 = stablehlo.sqrt %v8853 : tensor<384xf32>
    %v8857 = stablehlo.add %v8856, %v8855 : tensor<384xf32>
    %v8858 = stablehlo.divide %v8852, %v8857 : tensor<384xf32>
    %v8859 = stablehlo.multiply %v8854, %v8858 : tensor<384xf32>
    %v8860 = stablehlo.subtract %s2b5pb, %v8859 : tensor<384xf32>
    %v8861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8862 = stablehlo.multiply %v8861, %v8854 : tensor<384xf32>
    %v8863 = stablehlo.multiply %v8862, %s2b5pb : tensor<384xf32>
    %v8864 = stablehlo.subtract %v8860, %v8863 : tensor<384xf32>
    %v8865 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8866 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8867 = stablehlo.multiply %v8865, %s2b5lgm : tensor<384xf32>
    %v8868 = stablehlo.multiply %v8866, %v2499 : tensor<384xf32>
    %v8869 = stablehlo.add %v8867, %v8868 : tensor<384xf32>
    %v8870 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8871 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8872 = stablehlo.multiply %v8870, %s2b5lgv : tensor<384xf32>
    %v8873 = stablehlo.multiply %v2499, %v2499 : tensor<384xf32>
    %v8874 = stablehlo.multiply %v8871, %v8873 : tensor<384xf32>
    %v8875 = stablehlo.add %v8872, %v8874 : tensor<384xf32>
    %v8876 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8877 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8878 = stablehlo.multiply %v8876, %s2b5lgm : tensor<384xf32>
    %v8879 = stablehlo.multiply %v8877, %v2499 : tensor<384xf32>
    %v8880 = stablehlo.add %v8878, %v8879 : tensor<384xf32>
    %v8881 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8882 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8883 = stablehlo.multiply %v8881, %s2b5lgv : tensor<384xf32>
    %v8884 = stablehlo.multiply %v2499, %v2499 : tensor<384xf32>
    %v8885 = stablehlo.multiply %v8882, %v8884 : tensor<384xf32>
    %v8886 = stablehlo.add %v8883, %v8885 : tensor<384xf32>
    %v8887 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8888 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8889 = stablehlo.divide %v8880, %v8887 : tensor<384xf32>
    %v8890 = stablehlo.divide %v8886, %v8888 : tensor<384xf32>
    %v8891 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8892 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8893 = stablehlo.sqrt %v8890 : tensor<384xf32>
    %v8894 = stablehlo.add %v8893, %v8892 : tensor<384xf32>
    %v8895 = stablehlo.divide %v8889, %v8894 : tensor<384xf32>
    %v8896 = stablehlo.multiply %v8891, %v8895 : tensor<384xf32>
    %v8897 = stablehlo.subtract %s2b5lg, %v8896 : tensor<384xf32>
    %v8898 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8899 = stablehlo.multiply %v8898, %v8891 : tensor<384xf32>
    %v8900 = stablehlo.multiply %v8899, %s2b5lg : tensor<384xf32>
    %v8901 = stablehlo.subtract %v8897, %v8900 : tensor<384xf32>
    %v8902 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8903 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8904 = stablehlo.multiply %v8902, %s2b6dWm : tensor<384x1x7x7xf32>
    %v8905 = stablehlo.multiply %v8903, %v2404 : tensor<384x1x7x7xf32>
    %v8906 = stablehlo.add %v8904, %v8905 : tensor<384x1x7x7xf32>
    %v8907 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8908 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8909 = stablehlo.multiply %v8907, %s2b6dWv : tensor<384x1x7x7xf32>
    %v8910 = stablehlo.multiply %v2404, %v2404 : tensor<384x1x7x7xf32>
    %v8911 = stablehlo.multiply %v8908, %v8910 : tensor<384x1x7x7xf32>
    %v8912 = stablehlo.add %v8909, %v8911 : tensor<384x1x7x7xf32>
    %v8913 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8914 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8915 = stablehlo.multiply %v8913, %s2b6dWm : tensor<384x1x7x7xf32>
    %v8916 = stablehlo.multiply %v8914, %v2404 : tensor<384x1x7x7xf32>
    %v8917 = stablehlo.add %v8915, %v8916 : tensor<384x1x7x7xf32>
    %v8918 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8919 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8920 = stablehlo.multiply %v8918, %s2b6dWv : tensor<384x1x7x7xf32>
    %v8921 = stablehlo.multiply %v2404, %v2404 : tensor<384x1x7x7xf32>
    %v8922 = stablehlo.multiply %v8919, %v8921 : tensor<384x1x7x7xf32>
    %v8923 = stablehlo.add %v8920, %v8922 : tensor<384x1x7x7xf32>
    %v8924 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8925 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8926 = stablehlo.divide %v8917, %v8924 : tensor<384x1x7x7xf32>
    %v8927 = stablehlo.divide %v8923, %v8925 : tensor<384x1x7x7xf32>
    %v8928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8929 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8930 = stablehlo.sqrt %v8927 : tensor<384x1x7x7xf32>
    %v8931 = stablehlo.add %v8930, %v8929 : tensor<384x1x7x7xf32>
    %v8932 = stablehlo.divide %v8926, %v8931 : tensor<384x1x7x7xf32>
    %v8933 = stablehlo.multiply %v8928, %v8932 : tensor<384x1x7x7xf32>
    %v8934 = stablehlo.subtract %s2b6dW, %v8933 : tensor<384x1x7x7xf32>
    %v8935 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v8936 = stablehlo.multiply %v8935, %v8928 : tensor<384x1x7x7xf32>
    %v8937 = stablehlo.multiply %v8936, %s2b6dW : tensor<384x1x7x7xf32>
    %v8938 = stablehlo.subtract %v8934, %v8937 : tensor<384x1x7x7xf32>
    %v8939 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8940 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8941 = stablehlo.multiply %v8939, %s2b6dbm : tensor<384xf32>
    %v8942 = stablehlo.multiply %v8940, %v2407 : tensor<384xf32>
    %v8943 = stablehlo.add %v8941, %v8942 : tensor<384xf32>
    %v8944 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8945 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8946 = stablehlo.multiply %v8944, %s2b6dbv : tensor<384xf32>
    %v8947 = stablehlo.multiply %v2407, %v2407 : tensor<384xf32>
    %v8948 = stablehlo.multiply %v8945, %v8947 : tensor<384xf32>
    %v8949 = stablehlo.add %v8946, %v8948 : tensor<384xf32>
    %v8950 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8951 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8952 = stablehlo.multiply %v8950, %s2b6dbm : tensor<384xf32>
    %v8953 = stablehlo.multiply %v8951, %v2407 : tensor<384xf32>
    %v8954 = stablehlo.add %v8952, %v8953 : tensor<384xf32>
    %v8955 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8956 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8957 = stablehlo.multiply %v8955, %s2b6dbv : tensor<384xf32>
    %v8958 = stablehlo.multiply %v2407, %v2407 : tensor<384xf32>
    %v8959 = stablehlo.multiply %v8956, %v8958 : tensor<384xf32>
    %v8960 = stablehlo.add %v8957, %v8959 : tensor<384xf32>
    %v8961 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8962 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8963 = stablehlo.divide %v8954, %v8961 : tensor<384xf32>
    %v8964 = stablehlo.divide %v8960, %v8962 : tensor<384xf32>
    %v8965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8966 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8967 = stablehlo.sqrt %v8964 : tensor<384xf32>
    %v8968 = stablehlo.add %v8967, %v8966 : tensor<384xf32>
    %v8969 = stablehlo.divide %v8963, %v8968 : tensor<384xf32>
    %v8970 = stablehlo.multiply %v8965, %v8969 : tensor<384xf32>
    %v8971 = stablehlo.subtract %s2b6db, %v8970 : tensor<384xf32>
    %v8972 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8973 = stablehlo.multiply %v8972, %v8965 : tensor<384xf32>
    %v8974 = stablehlo.multiply %v8973, %s2b6db : tensor<384xf32>
    %v8975 = stablehlo.subtract %v8971, %v8974 : tensor<384xf32>
    %v8976 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8977 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8978 = stablehlo.multiply %v8976, %s2b6ngm : tensor<384xf32>
    %v8979 = stablehlo.multiply %v8977, %v2392 : tensor<384xf32>
    %v8980 = stablehlo.add %v8978, %v8979 : tensor<384xf32>
    %v8981 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8982 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8983 = stablehlo.multiply %v8981, %s2b6ngv : tensor<384xf32>
    %v8984 = stablehlo.multiply %v2392, %v2392 : tensor<384xf32>
    %v8985 = stablehlo.multiply %v8982, %v8984 : tensor<384xf32>
    %v8986 = stablehlo.add %v8983, %v8985 : tensor<384xf32>
    %v8987 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8988 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8989 = stablehlo.multiply %v8987, %s2b6ngm : tensor<384xf32>
    %v8990 = stablehlo.multiply %v8988, %v2392 : tensor<384xf32>
    %v8991 = stablehlo.add %v8989, %v8990 : tensor<384xf32>
    %v8992 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8993 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8994 = stablehlo.multiply %v8992, %s2b6ngv : tensor<384xf32>
    %v8995 = stablehlo.multiply %v2392, %v2392 : tensor<384xf32>
    %v8996 = stablehlo.multiply %v8993, %v8995 : tensor<384xf32>
    %v8997 = stablehlo.add %v8994, %v8996 : tensor<384xf32>
    %v8998 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v8999 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9000 = stablehlo.divide %v8991, %v8998 : tensor<384xf32>
    %v9001 = stablehlo.divide %v8997, %v8999 : tensor<384xf32>
    %v9002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9003 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9004 = stablehlo.sqrt %v9001 : tensor<384xf32>
    %v9005 = stablehlo.add %v9004, %v9003 : tensor<384xf32>
    %v9006 = stablehlo.divide %v9000, %v9005 : tensor<384xf32>
    %v9007 = stablehlo.multiply %v9002, %v9006 : tensor<384xf32>
    %v9008 = stablehlo.subtract %s2b6ng, %v9007 : tensor<384xf32>
    %v9009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9010 = stablehlo.multiply %v9009, %v9002 : tensor<384xf32>
    %v9011 = stablehlo.multiply %v9010, %s2b6ng : tensor<384xf32>
    %v9012 = stablehlo.subtract %v9008, %v9011 : tensor<384xf32>
    %v9013 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9014 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9015 = stablehlo.multiply %v9013, %s2b6nbtm : tensor<384xf32>
    %v9016 = stablehlo.multiply %v9014, %v2398 : tensor<384xf32>
    %v9017 = stablehlo.add %v9015, %v9016 : tensor<384xf32>
    %v9018 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9019 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9020 = stablehlo.multiply %v9018, %s2b6nbtv : tensor<384xf32>
    %v9021 = stablehlo.multiply %v2398, %v2398 : tensor<384xf32>
    %v9022 = stablehlo.multiply %v9019, %v9021 : tensor<384xf32>
    %v9023 = stablehlo.add %v9020, %v9022 : tensor<384xf32>
    %v9024 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9025 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9026 = stablehlo.multiply %v9024, %s2b6nbtm : tensor<384xf32>
    %v9027 = stablehlo.multiply %v9025, %v2398 : tensor<384xf32>
    %v9028 = stablehlo.add %v9026, %v9027 : tensor<384xf32>
    %v9029 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9030 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9031 = stablehlo.multiply %v9029, %s2b6nbtv : tensor<384xf32>
    %v9032 = stablehlo.multiply %v2398, %v2398 : tensor<384xf32>
    %v9033 = stablehlo.multiply %v9030, %v9032 : tensor<384xf32>
    %v9034 = stablehlo.add %v9031, %v9033 : tensor<384xf32>
    %v9035 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9036 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9037 = stablehlo.divide %v9028, %v9035 : tensor<384xf32>
    %v9038 = stablehlo.divide %v9034, %v9036 : tensor<384xf32>
    %v9039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9040 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9041 = stablehlo.sqrt %v9038 : tensor<384xf32>
    %v9042 = stablehlo.add %v9041, %v9040 : tensor<384xf32>
    %v9043 = stablehlo.divide %v9037, %v9042 : tensor<384xf32>
    %v9044 = stablehlo.multiply %v9039, %v9043 : tensor<384xf32>
    %v9045 = stablehlo.subtract %s2b6nbt, %v9044 : tensor<384xf32>
    %v9046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9047 = stablehlo.multiply %v9046, %v9039 : tensor<384xf32>
    %v9048 = stablehlo.multiply %v9047, %s2b6nbt : tensor<384xf32>
    %v9049 = stablehlo.subtract %v9045, %v9048 : tensor<384xf32>
    %v9050 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9051 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9052 = stablehlo.multiply %v9050, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v9053 = stablehlo.multiply %v9051, %v2365 : tensor<1536x384x1x1xf32>
    %v9054 = stablehlo.add %v9052, %v9053 : tensor<1536x384x1x1xf32>
    %v9055 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9056 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9057 = stablehlo.multiply %v9055, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v9058 = stablehlo.multiply %v2365, %v2365 : tensor<1536x384x1x1xf32>
    %v9059 = stablehlo.multiply %v9056, %v9058 : tensor<1536x384x1x1xf32>
    %v9060 = stablehlo.add %v9057, %v9059 : tensor<1536x384x1x1xf32>
    %v9061 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9062 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9063 = stablehlo.multiply %v9061, %s2b6eWm : tensor<1536x384x1x1xf32>
    %v9064 = stablehlo.multiply %v9062, %v2365 : tensor<1536x384x1x1xf32>
    %v9065 = stablehlo.add %v9063, %v9064 : tensor<1536x384x1x1xf32>
    %v9066 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9067 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9068 = stablehlo.multiply %v9066, %s2b6eWv : tensor<1536x384x1x1xf32>
    %v9069 = stablehlo.multiply %v2365, %v2365 : tensor<1536x384x1x1xf32>
    %v9070 = stablehlo.multiply %v9067, %v9069 : tensor<1536x384x1x1xf32>
    %v9071 = stablehlo.add %v9068, %v9070 : tensor<1536x384x1x1xf32>
    %v9072 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9073 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9074 = stablehlo.divide %v9065, %v9072 : tensor<1536x384x1x1xf32>
    %v9075 = stablehlo.divide %v9071, %v9073 : tensor<1536x384x1x1xf32>
    %v9076 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9077 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9078 = stablehlo.sqrt %v9075 : tensor<1536x384x1x1xf32>
    %v9079 = stablehlo.add %v9078, %v9077 : tensor<1536x384x1x1xf32>
    %v9080 = stablehlo.divide %v9074, %v9079 : tensor<1536x384x1x1xf32>
    %v9081 = stablehlo.multiply %v9076, %v9080 : tensor<1536x384x1x1xf32>
    %v9082 = stablehlo.subtract %s2b6eW, %v9081 : tensor<1536x384x1x1xf32>
    %v9083 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9084 = stablehlo.multiply %v9083, %v9076 : tensor<1536x384x1x1xf32>
    %v9085 = stablehlo.multiply %v9084, %s2b6eW : tensor<1536x384x1x1xf32>
    %v9086 = stablehlo.subtract %v9082, %v9085 : tensor<1536x384x1x1xf32>
    %v9087 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9088 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9089 = stablehlo.multiply %v9087, %s2b6ebm : tensor<1536xf32>
    %v9090 = stablehlo.multiply %v9088, %v2368 : tensor<1536xf32>
    %v9091 = stablehlo.add %v9089, %v9090 : tensor<1536xf32>
    %v9092 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9093 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9094 = stablehlo.multiply %v9092, %s2b6ebv : tensor<1536xf32>
    %v9095 = stablehlo.multiply %v2368, %v2368 : tensor<1536xf32>
    %v9096 = stablehlo.multiply %v9093, %v9095 : tensor<1536xf32>
    %v9097 = stablehlo.add %v9094, %v9096 : tensor<1536xf32>
    %v9098 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9099 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9100 = stablehlo.multiply %v9098, %s2b6ebm : tensor<1536xf32>
    %v9101 = stablehlo.multiply %v9099, %v2368 : tensor<1536xf32>
    %v9102 = stablehlo.add %v9100, %v9101 : tensor<1536xf32>
    %v9103 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9104 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9105 = stablehlo.multiply %v9103, %s2b6ebv : tensor<1536xf32>
    %v9106 = stablehlo.multiply %v2368, %v2368 : tensor<1536xf32>
    %v9107 = stablehlo.multiply %v9104, %v9106 : tensor<1536xf32>
    %v9108 = stablehlo.add %v9105, %v9107 : tensor<1536xf32>
    %v9109 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9110 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9111 = stablehlo.divide %v9102, %v9109 : tensor<1536xf32>
    %v9112 = stablehlo.divide %v9108, %v9110 : tensor<1536xf32>
    %v9113 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9114 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9115 = stablehlo.sqrt %v9112 : tensor<1536xf32>
    %v9116 = stablehlo.add %v9115, %v9114 : tensor<1536xf32>
    %v9117 = stablehlo.divide %v9111, %v9116 : tensor<1536xf32>
    %v9118 = stablehlo.multiply %v9113, %v9117 : tensor<1536xf32>
    %v9119 = stablehlo.subtract %s2b6eb, %v9118 : tensor<1536xf32>
    %v9120 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9121 = stablehlo.multiply %v9120, %v9113 : tensor<1536xf32>
    %v9122 = stablehlo.multiply %v9121, %s2b6eb : tensor<1536xf32>
    %v9123 = stablehlo.subtract %v9119, %v9122 : tensor<1536xf32>
    %v9124 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9125 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9126 = stablehlo.multiply %v9124, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v9127 = stablehlo.multiply %v9125, %v2356 : tensor<384x1536x1x1xf32>
    %v9128 = stablehlo.add %v9126, %v9127 : tensor<384x1536x1x1xf32>
    %v9129 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9130 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9131 = stablehlo.multiply %v9129, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v9132 = stablehlo.multiply %v2356, %v2356 : tensor<384x1536x1x1xf32>
    %v9133 = stablehlo.multiply %v9130, %v9132 : tensor<384x1536x1x1xf32>
    %v9134 = stablehlo.add %v9131, %v9133 : tensor<384x1536x1x1xf32>
    %v9135 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9136 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9137 = stablehlo.multiply %v9135, %s2b6pWm : tensor<384x1536x1x1xf32>
    %v9138 = stablehlo.multiply %v9136, %v2356 : tensor<384x1536x1x1xf32>
    %v9139 = stablehlo.add %v9137, %v9138 : tensor<384x1536x1x1xf32>
    %v9140 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9141 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9142 = stablehlo.multiply %v9140, %s2b6pWv : tensor<384x1536x1x1xf32>
    %v9143 = stablehlo.multiply %v2356, %v2356 : tensor<384x1536x1x1xf32>
    %v9144 = stablehlo.multiply %v9141, %v9143 : tensor<384x1536x1x1xf32>
    %v9145 = stablehlo.add %v9142, %v9144 : tensor<384x1536x1x1xf32>
    %v9146 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9147 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9148 = stablehlo.divide %v9139, %v9146 : tensor<384x1536x1x1xf32>
    %v9149 = stablehlo.divide %v9145, %v9147 : tensor<384x1536x1x1xf32>
    %v9150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9151 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9152 = stablehlo.sqrt %v9149 : tensor<384x1536x1x1xf32>
    %v9153 = stablehlo.add %v9152, %v9151 : tensor<384x1536x1x1xf32>
    %v9154 = stablehlo.divide %v9148, %v9153 : tensor<384x1536x1x1xf32>
    %v9155 = stablehlo.multiply %v9150, %v9154 : tensor<384x1536x1x1xf32>
    %v9156 = stablehlo.subtract %s2b6pW, %v9155 : tensor<384x1536x1x1xf32>
    %v9157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9158 = stablehlo.multiply %v9157, %v9150 : tensor<384x1536x1x1xf32>
    %v9159 = stablehlo.multiply %v9158, %s2b6pW : tensor<384x1536x1x1xf32>
    %v9160 = stablehlo.subtract %v9156, %v9159 : tensor<384x1536x1x1xf32>
    %v9161 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9162 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9163 = stablehlo.multiply %v9161, %s2b6pbm : tensor<384xf32>
    %v9164 = stablehlo.multiply %v9162, %v2359 : tensor<384xf32>
    %v9165 = stablehlo.add %v9163, %v9164 : tensor<384xf32>
    %v9166 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9167 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9168 = stablehlo.multiply %v9166, %s2b6pbv : tensor<384xf32>
    %v9169 = stablehlo.multiply %v2359, %v2359 : tensor<384xf32>
    %v9170 = stablehlo.multiply %v9167, %v9169 : tensor<384xf32>
    %v9171 = stablehlo.add %v9168, %v9170 : tensor<384xf32>
    %v9172 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9173 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9174 = stablehlo.multiply %v9172, %s2b6pbm : tensor<384xf32>
    %v9175 = stablehlo.multiply %v9173, %v2359 : tensor<384xf32>
    %v9176 = stablehlo.add %v9174, %v9175 : tensor<384xf32>
    %v9177 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9178 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9179 = stablehlo.multiply %v9177, %s2b6pbv : tensor<384xf32>
    %v9180 = stablehlo.multiply %v2359, %v2359 : tensor<384xf32>
    %v9181 = stablehlo.multiply %v9178, %v9180 : tensor<384xf32>
    %v9182 = stablehlo.add %v9179, %v9181 : tensor<384xf32>
    %v9183 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9184 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9185 = stablehlo.divide %v9176, %v9183 : tensor<384xf32>
    %v9186 = stablehlo.divide %v9182, %v9184 : tensor<384xf32>
    %v9187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9188 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9189 = stablehlo.sqrt %v9186 : tensor<384xf32>
    %v9190 = stablehlo.add %v9189, %v9188 : tensor<384xf32>
    %v9191 = stablehlo.divide %v9185, %v9190 : tensor<384xf32>
    %v9192 = stablehlo.multiply %v9187, %v9191 : tensor<384xf32>
    %v9193 = stablehlo.subtract %s2b6pb, %v9192 : tensor<384xf32>
    %v9194 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9195 = stablehlo.multiply %v9194, %v9187 : tensor<384xf32>
    %v9196 = stablehlo.multiply %v9195, %s2b6pb : tensor<384xf32>
    %v9197 = stablehlo.subtract %v9193, %v9196 : tensor<384xf32>
    %v9198 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9199 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9200 = stablehlo.multiply %v9198, %s2b6lgm : tensor<384xf32>
    %v9201 = stablehlo.multiply %v9199, %v2350 : tensor<384xf32>
    %v9202 = stablehlo.add %v9200, %v9201 : tensor<384xf32>
    %v9203 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9204 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9205 = stablehlo.multiply %v9203, %s2b6lgv : tensor<384xf32>
    %v9206 = stablehlo.multiply %v2350, %v2350 : tensor<384xf32>
    %v9207 = stablehlo.multiply %v9204, %v9206 : tensor<384xf32>
    %v9208 = stablehlo.add %v9205, %v9207 : tensor<384xf32>
    %v9209 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9210 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9211 = stablehlo.multiply %v9209, %s2b6lgm : tensor<384xf32>
    %v9212 = stablehlo.multiply %v9210, %v2350 : tensor<384xf32>
    %v9213 = stablehlo.add %v9211, %v9212 : tensor<384xf32>
    %v9214 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9215 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9216 = stablehlo.multiply %v9214, %s2b6lgv : tensor<384xf32>
    %v9217 = stablehlo.multiply %v2350, %v2350 : tensor<384xf32>
    %v9218 = stablehlo.multiply %v9215, %v9217 : tensor<384xf32>
    %v9219 = stablehlo.add %v9216, %v9218 : tensor<384xf32>
    %v9220 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9221 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9222 = stablehlo.divide %v9213, %v9220 : tensor<384xf32>
    %v9223 = stablehlo.divide %v9219, %v9221 : tensor<384xf32>
    %v9224 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9225 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9226 = stablehlo.sqrt %v9223 : tensor<384xf32>
    %v9227 = stablehlo.add %v9226, %v9225 : tensor<384xf32>
    %v9228 = stablehlo.divide %v9222, %v9227 : tensor<384xf32>
    %v9229 = stablehlo.multiply %v9224, %v9228 : tensor<384xf32>
    %v9230 = stablehlo.subtract %s2b6lg, %v9229 : tensor<384xf32>
    %v9231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9232 = stablehlo.multiply %v9231, %v9224 : tensor<384xf32>
    %v9233 = stablehlo.multiply %v9232, %s2b6lg : tensor<384xf32>
    %v9234 = stablehlo.subtract %v9230, %v9233 : tensor<384xf32>
    %v9235 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9236 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9237 = stablehlo.multiply %v9235, %s2b7dWm : tensor<384x1x7x7xf32>
    %v9238 = stablehlo.multiply %v9236, %v2255 : tensor<384x1x7x7xf32>
    %v9239 = stablehlo.add %v9237, %v9238 : tensor<384x1x7x7xf32>
    %v9240 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9241 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9242 = stablehlo.multiply %v9240, %s2b7dWv : tensor<384x1x7x7xf32>
    %v9243 = stablehlo.multiply %v2255, %v2255 : tensor<384x1x7x7xf32>
    %v9244 = stablehlo.multiply %v9241, %v9243 : tensor<384x1x7x7xf32>
    %v9245 = stablehlo.add %v9242, %v9244 : tensor<384x1x7x7xf32>
    %v9246 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9247 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9248 = stablehlo.multiply %v9246, %s2b7dWm : tensor<384x1x7x7xf32>
    %v9249 = stablehlo.multiply %v9247, %v2255 : tensor<384x1x7x7xf32>
    %v9250 = stablehlo.add %v9248, %v9249 : tensor<384x1x7x7xf32>
    %v9251 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9252 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9253 = stablehlo.multiply %v9251, %s2b7dWv : tensor<384x1x7x7xf32>
    %v9254 = stablehlo.multiply %v2255, %v2255 : tensor<384x1x7x7xf32>
    %v9255 = stablehlo.multiply %v9252, %v9254 : tensor<384x1x7x7xf32>
    %v9256 = stablehlo.add %v9253, %v9255 : tensor<384x1x7x7xf32>
    %v9257 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9258 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9259 = stablehlo.divide %v9250, %v9257 : tensor<384x1x7x7xf32>
    %v9260 = stablehlo.divide %v9256, %v9258 : tensor<384x1x7x7xf32>
    %v9261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9262 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9263 = stablehlo.sqrt %v9260 : tensor<384x1x7x7xf32>
    %v9264 = stablehlo.add %v9263, %v9262 : tensor<384x1x7x7xf32>
    %v9265 = stablehlo.divide %v9259, %v9264 : tensor<384x1x7x7xf32>
    %v9266 = stablehlo.multiply %v9261, %v9265 : tensor<384x1x7x7xf32>
    %v9267 = stablehlo.subtract %s2b7dW, %v9266 : tensor<384x1x7x7xf32>
    %v9268 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9269 = stablehlo.multiply %v9268, %v9261 : tensor<384x1x7x7xf32>
    %v9270 = stablehlo.multiply %v9269, %s2b7dW : tensor<384x1x7x7xf32>
    %v9271 = stablehlo.subtract %v9267, %v9270 : tensor<384x1x7x7xf32>
    %v9272 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9273 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9274 = stablehlo.multiply %v9272, %s2b7dbm : tensor<384xf32>
    %v9275 = stablehlo.multiply %v9273, %v2258 : tensor<384xf32>
    %v9276 = stablehlo.add %v9274, %v9275 : tensor<384xf32>
    %v9277 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9278 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9279 = stablehlo.multiply %v9277, %s2b7dbv : tensor<384xf32>
    %v9280 = stablehlo.multiply %v2258, %v2258 : tensor<384xf32>
    %v9281 = stablehlo.multiply %v9278, %v9280 : tensor<384xf32>
    %v9282 = stablehlo.add %v9279, %v9281 : tensor<384xf32>
    %v9283 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9284 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9285 = stablehlo.multiply %v9283, %s2b7dbm : tensor<384xf32>
    %v9286 = stablehlo.multiply %v9284, %v2258 : tensor<384xf32>
    %v9287 = stablehlo.add %v9285, %v9286 : tensor<384xf32>
    %v9288 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9289 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9290 = stablehlo.multiply %v9288, %s2b7dbv : tensor<384xf32>
    %v9291 = stablehlo.multiply %v2258, %v2258 : tensor<384xf32>
    %v9292 = stablehlo.multiply %v9289, %v9291 : tensor<384xf32>
    %v9293 = stablehlo.add %v9290, %v9292 : tensor<384xf32>
    %v9294 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9295 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9296 = stablehlo.divide %v9287, %v9294 : tensor<384xf32>
    %v9297 = stablehlo.divide %v9293, %v9295 : tensor<384xf32>
    %v9298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9299 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9300 = stablehlo.sqrt %v9297 : tensor<384xf32>
    %v9301 = stablehlo.add %v9300, %v9299 : tensor<384xf32>
    %v9302 = stablehlo.divide %v9296, %v9301 : tensor<384xf32>
    %v9303 = stablehlo.multiply %v9298, %v9302 : tensor<384xf32>
    %v9304 = stablehlo.subtract %s2b7db, %v9303 : tensor<384xf32>
    %v9305 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9306 = stablehlo.multiply %v9305, %v9298 : tensor<384xf32>
    %v9307 = stablehlo.multiply %v9306, %s2b7db : tensor<384xf32>
    %v9308 = stablehlo.subtract %v9304, %v9307 : tensor<384xf32>
    %v9309 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9310 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9311 = stablehlo.multiply %v9309, %s2b7ngm : tensor<384xf32>
    %v9312 = stablehlo.multiply %v9310, %v2243 : tensor<384xf32>
    %v9313 = stablehlo.add %v9311, %v9312 : tensor<384xf32>
    %v9314 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9315 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9316 = stablehlo.multiply %v9314, %s2b7ngv : tensor<384xf32>
    %v9317 = stablehlo.multiply %v2243, %v2243 : tensor<384xf32>
    %v9318 = stablehlo.multiply %v9315, %v9317 : tensor<384xf32>
    %v9319 = stablehlo.add %v9316, %v9318 : tensor<384xf32>
    %v9320 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9321 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9322 = stablehlo.multiply %v9320, %s2b7ngm : tensor<384xf32>
    %v9323 = stablehlo.multiply %v9321, %v2243 : tensor<384xf32>
    %v9324 = stablehlo.add %v9322, %v9323 : tensor<384xf32>
    %v9325 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9326 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9327 = stablehlo.multiply %v9325, %s2b7ngv : tensor<384xf32>
    %v9328 = stablehlo.multiply %v2243, %v2243 : tensor<384xf32>
    %v9329 = stablehlo.multiply %v9326, %v9328 : tensor<384xf32>
    %v9330 = stablehlo.add %v9327, %v9329 : tensor<384xf32>
    %v9331 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9332 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9333 = stablehlo.divide %v9324, %v9331 : tensor<384xf32>
    %v9334 = stablehlo.divide %v9330, %v9332 : tensor<384xf32>
    %v9335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9336 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9337 = stablehlo.sqrt %v9334 : tensor<384xf32>
    %v9338 = stablehlo.add %v9337, %v9336 : tensor<384xf32>
    %v9339 = stablehlo.divide %v9333, %v9338 : tensor<384xf32>
    %v9340 = stablehlo.multiply %v9335, %v9339 : tensor<384xf32>
    %v9341 = stablehlo.subtract %s2b7ng, %v9340 : tensor<384xf32>
    %v9342 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9343 = stablehlo.multiply %v9342, %v9335 : tensor<384xf32>
    %v9344 = stablehlo.multiply %v9343, %s2b7ng : tensor<384xf32>
    %v9345 = stablehlo.subtract %v9341, %v9344 : tensor<384xf32>
    %v9346 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9347 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9348 = stablehlo.multiply %v9346, %s2b7nbtm : tensor<384xf32>
    %v9349 = stablehlo.multiply %v9347, %v2249 : tensor<384xf32>
    %v9350 = stablehlo.add %v9348, %v9349 : tensor<384xf32>
    %v9351 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9352 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9353 = stablehlo.multiply %v9351, %s2b7nbtv : tensor<384xf32>
    %v9354 = stablehlo.multiply %v2249, %v2249 : tensor<384xf32>
    %v9355 = stablehlo.multiply %v9352, %v9354 : tensor<384xf32>
    %v9356 = stablehlo.add %v9353, %v9355 : tensor<384xf32>
    %v9357 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9358 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9359 = stablehlo.multiply %v9357, %s2b7nbtm : tensor<384xf32>
    %v9360 = stablehlo.multiply %v9358, %v2249 : tensor<384xf32>
    %v9361 = stablehlo.add %v9359, %v9360 : tensor<384xf32>
    %v9362 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9363 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9364 = stablehlo.multiply %v9362, %s2b7nbtv : tensor<384xf32>
    %v9365 = stablehlo.multiply %v2249, %v2249 : tensor<384xf32>
    %v9366 = stablehlo.multiply %v9363, %v9365 : tensor<384xf32>
    %v9367 = stablehlo.add %v9364, %v9366 : tensor<384xf32>
    %v9368 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9369 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9370 = stablehlo.divide %v9361, %v9368 : tensor<384xf32>
    %v9371 = stablehlo.divide %v9367, %v9369 : tensor<384xf32>
    %v9372 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9373 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9374 = stablehlo.sqrt %v9371 : tensor<384xf32>
    %v9375 = stablehlo.add %v9374, %v9373 : tensor<384xf32>
    %v9376 = stablehlo.divide %v9370, %v9375 : tensor<384xf32>
    %v9377 = stablehlo.multiply %v9372, %v9376 : tensor<384xf32>
    %v9378 = stablehlo.subtract %s2b7nbt, %v9377 : tensor<384xf32>
    %v9379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9380 = stablehlo.multiply %v9379, %v9372 : tensor<384xf32>
    %v9381 = stablehlo.multiply %v9380, %s2b7nbt : tensor<384xf32>
    %v9382 = stablehlo.subtract %v9378, %v9381 : tensor<384xf32>
    %v9383 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9384 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9385 = stablehlo.multiply %v9383, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v9386 = stablehlo.multiply %v9384, %v2216 : tensor<1536x384x1x1xf32>
    %v9387 = stablehlo.add %v9385, %v9386 : tensor<1536x384x1x1xf32>
    %v9388 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9389 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9390 = stablehlo.multiply %v9388, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v9391 = stablehlo.multiply %v2216, %v2216 : tensor<1536x384x1x1xf32>
    %v9392 = stablehlo.multiply %v9389, %v9391 : tensor<1536x384x1x1xf32>
    %v9393 = stablehlo.add %v9390, %v9392 : tensor<1536x384x1x1xf32>
    %v9394 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9395 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9396 = stablehlo.multiply %v9394, %s2b7eWm : tensor<1536x384x1x1xf32>
    %v9397 = stablehlo.multiply %v9395, %v2216 : tensor<1536x384x1x1xf32>
    %v9398 = stablehlo.add %v9396, %v9397 : tensor<1536x384x1x1xf32>
    %v9399 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9400 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9401 = stablehlo.multiply %v9399, %s2b7eWv : tensor<1536x384x1x1xf32>
    %v9402 = stablehlo.multiply %v2216, %v2216 : tensor<1536x384x1x1xf32>
    %v9403 = stablehlo.multiply %v9400, %v9402 : tensor<1536x384x1x1xf32>
    %v9404 = stablehlo.add %v9401, %v9403 : tensor<1536x384x1x1xf32>
    %v9405 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9406 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9407 = stablehlo.divide %v9398, %v9405 : tensor<1536x384x1x1xf32>
    %v9408 = stablehlo.divide %v9404, %v9406 : tensor<1536x384x1x1xf32>
    %v9409 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9410 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9411 = stablehlo.sqrt %v9408 : tensor<1536x384x1x1xf32>
    %v9412 = stablehlo.add %v9411, %v9410 : tensor<1536x384x1x1xf32>
    %v9413 = stablehlo.divide %v9407, %v9412 : tensor<1536x384x1x1xf32>
    %v9414 = stablehlo.multiply %v9409, %v9413 : tensor<1536x384x1x1xf32>
    %v9415 = stablehlo.subtract %s2b7eW, %v9414 : tensor<1536x384x1x1xf32>
    %v9416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9417 = stablehlo.multiply %v9416, %v9409 : tensor<1536x384x1x1xf32>
    %v9418 = stablehlo.multiply %v9417, %s2b7eW : tensor<1536x384x1x1xf32>
    %v9419 = stablehlo.subtract %v9415, %v9418 : tensor<1536x384x1x1xf32>
    %v9420 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9421 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9422 = stablehlo.multiply %v9420, %s2b7ebm : tensor<1536xf32>
    %v9423 = stablehlo.multiply %v9421, %v2219 : tensor<1536xf32>
    %v9424 = stablehlo.add %v9422, %v9423 : tensor<1536xf32>
    %v9425 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9426 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9427 = stablehlo.multiply %v9425, %s2b7ebv : tensor<1536xf32>
    %v9428 = stablehlo.multiply %v2219, %v2219 : tensor<1536xf32>
    %v9429 = stablehlo.multiply %v9426, %v9428 : tensor<1536xf32>
    %v9430 = stablehlo.add %v9427, %v9429 : tensor<1536xf32>
    %v9431 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9432 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9433 = stablehlo.multiply %v9431, %s2b7ebm : tensor<1536xf32>
    %v9434 = stablehlo.multiply %v9432, %v2219 : tensor<1536xf32>
    %v9435 = stablehlo.add %v9433, %v9434 : tensor<1536xf32>
    %v9436 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9437 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9438 = stablehlo.multiply %v9436, %s2b7ebv : tensor<1536xf32>
    %v9439 = stablehlo.multiply %v2219, %v2219 : tensor<1536xf32>
    %v9440 = stablehlo.multiply %v9437, %v9439 : tensor<1536xf32>
    %v9441 = stablehlo.add %v9438, %v9440 : tensor<1536xf32>
    %v9442 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9443 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9444 = stablehlo.divide %v9435, %v9442 : tensor<1536xf32>
    %v9445 = stablehlo.divide %v9441, %v9443 : tensor<1536xf32>
    %v9446 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9447 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9448 = stablehlo.sqrt %v9445 : tensor<1536xf32>
    %v9449 = stablehlo.add %v9448, %v9447 : tensor<1536xf32>
    %v9450 = stablehlo.divide %v9444, %v9449 : tensor<1536xf32>
    %v9451 = stablehlo.multiply %v9446, %v9450 : tensor<1536xf32>
    %v9452 = stablehlo.subtract %s2b7eb, %v9451 : tensor<1536xf32>
    %v9453 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9454 = stablehlo.multiply %v9453, %v9446 : tensor<1536xf32>
    %v9455 = stablehlo.multiply %v9454, %s2b7eb : tensor<1536xf32>
    %v9456 = stablehlo.subtract %v9452, %v9455 : tensor<1536xf32>
    %v9457 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9458 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9459 = stablehlo.multiply %v9457, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v9460 = stablehlo.multiply %v9458, %v2207 : tensor<384x1536x1x1xf32>
    %v9461 = stablehlo.add %v9459, %v9460 : tensor<384x1536x1x1xf32>
    %v9462 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9463 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9464 = stablehlo.multiply %v9462, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v9465 = stablehlo.multiply %v2207, %v2207 : tensor<384x1536x1x1xf32>
    %v9466 = stablehlo.multiply %v9463, %v9465 : tensor<384x1536x1x1xf32>
    %v9467 = stablehlo.add %v9464, %v9466 : tensor<384x1536x1x1xf32>
    %v9468 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9469 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9470 = stablehlo.multiply %v9468, %s2b7pWm : tensor<384x1536x1x1xf32>
    %v9471 = stablehlo.multiply %v9469, %v2207 : tensor<384x1536x1x1xf32>
    %v9472 = stablehlo.add %v9470, %v9471 : tensor<384x1536x1x1xf32>
    %v9473 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9474 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9475 = stablehlo.multiply %v9473, %s2b7pWv : tensor<384x1536x1x1xf32>
    %v9476 = stablehlo.multiply %v2207, %v2207 : tensor<384x1536x1x1xf32>
    %v9477 = stablehlo.multiply %v9474, %v9476 : tensor<384x1536x1x1xf32>
    %v9478 = stablehlo.add %v9475, %v9477 : tensor<384x1536x1x1xf32>
    %v9479 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9480 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9481 = stablehlo.divide %v9472, %v9479 : tensor<384x1536x1x1xf32>
    %v9482 = stablehlo.divide %v9478, %v9480 : tensor<384x1536x1x1xf32>
    %v9483 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9484 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9485 = stablehlo.sqrt %v9482 : tensor<384x1536x1x1xf32>
    %v9486 = stablehlo.add %v9485, %v9484 : tensor<384x1536x1x1xf32>
    %v9487 = stablehlo.divide %v9481, %v9486 : tensor<384x1536x1x1xf32>
    %v9488 = stablehlo.multiply %v9483, %v9487 : tensor<384x1536x1x1xf32>
    %v9489 = stablehlo.subtract %s2b7pW, %v9488 : tensor<384x1536x1x1xf32>
    %v9490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9491 = stablehlo.multiply %v9490, %v9483 : tensor<384x1536x1x1xf32>
    %v9492 = stablehlo.multiply %v9491, %s2b7pW : tensor<384x1536x1x1xf32>
    %v9493 = stablehlo.subtract %v9489, %v9492 : tensor<384x1536x1x1xf32>
    %v9494 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9495 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9496 = stablehlo.multiply %v9494, %s2b7pbm : tensor<384xf32>
    %v9497 = stablehlo.multiply %v9495, %v2210 : tensor<384xf32>
    %v9498 = stablehlo.add %v9496, %v9497 : tensor<384xf32>
    %v9499 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9500 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9501 = stablehlo.multiply %v9499, %s2b7pbv : tensor<384xf32>
    %v9502 = stablehlo.multiply %v2210, %v2210 : tensor<384xf32>
    %v9503 = stablehlo.multiply %v9500, %v9502 : tensor<384xf32>
    %v9504 = stablehlo.add %v9501, %v9503 : tensor<384xf32>
    %v9505 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9506 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9507 = stablehlo.multiply %v9505, %s2b7pbm : tensor<384xf32>
    %v9508 = stablehlo.multiply %v9506, %v2210 : tensor<384xf32>
    %v9509 = stablehlo.add %v9507, %v9508 : tensor<384xf32>
    %v9510 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9511 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9512 = stablehlo.multiply %v9510, %s2b7pbv : tensor<384xf32>
    %v9513 = stablehlo.multiply %v2210, %v2210 : tensor<384xf32>
    %v9514 = stablehlo.multiply %v9511, %v9513 : tensor<384xf32>
    %v9515 = stablehlo.add %v9512, %v9514 : tensor<384xf32>
    %v9516 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9517 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9518 = stablehlo.divide %v9509, %v9516 : tensor<384xf32>
    %v9519 = stablehlo.divide %v9515, %v9517 : tensor<384xf32>
    %v9520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9521 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9522 = stablehlo.sqrt %v9519 : tensor<384xf32>
    %v9523 = stablehlo.add %v9522, %v9521 : tensor<384xf32>
    %v9524 = stablehlo.divide %v9518, %v9523 : tensor<384xf32>
    %v9525 = stablehlo.multiply %v9520, %v9524 : tensor<384xf32>
    %v9526 = stablehlo.subtract %s2b7pb, %v9525 : tensor<384xf32>
    %v9527 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9528 = stablehlo.multiply %v9527, %v9520 : tensor<384xf32>
    %v9529 = stablehlo.multiply %v9528, %s2b7pb : tensor<384xf32>
    %v9530 = stablehlo.subtract %v9526, %v9529 : tensor<384xf32>
    %v9531 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9532 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9533 = stablehlo.multiply %v9531, %s2b7lgm : tensor<384xf32>
    %v9534 = stablehlo.multiply %v9532, %v2201 : tensor<384xf32>
    %v9535 = stablehlo.add %v9533, %v9534 : tensor<384xf32>
    %v9536 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9537 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9538 = stablehlo.multiply %v9536, %s2b7lgv : tensor<384xf32>
    %v9539 = stablehlo.multiply %v2201, %v2201 : tensor<384xf32>
    %v9540 = stablehlo.multiply %v9537, %v9539 : tensor<384xf32>
    %v9541 = stablehlo.add %v9538, %v9540 : tensor<384xf32>
    %v9542 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9543 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9544 = stablehlo.multiply %v9542, %s2b7lgm : tensor<384xf32>
    %v9545 = stablehlo.multiply %v9543, %v2201 : tensor<384xf32>
    %v9546 = stablehlo.add %v9544, %v9545 : tensor<384xf32>
    %v9547 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9548 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9549 = stablehlo.multiply %v9547, %s2b7lgv : tensor<384xf32>
    %v9550 = stablehlo.multiply %v2201, %v2201 : tensor<384xf32>
    %v9551 = stablehlo.multiply %v9548, %v9550 : tensor<384xf32>
    %v9552 = stablehlo.add %v9549, %v9551 : tensor<384xf32>
    %v9553 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9554 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9555 = stablehlo.divide %v9546, %v9553 : tensor<384xf32>
    %v9556 = stablehlo.divide %v9552, %v9554 : tensor<384xf32>
    %v9557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9558 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9559 = stablehlo.sqrt %v9556 : tensor<384xf32>
    %v9560 = stablehlo.add %v9559, %v9558 : tensor<384xf32>
    %v9561 = stablehlo.divide %v9555, %v9560 : tensor<384xf32>
    %v9562 = stablehlo.multiply %v9557, %v9561 : tensor<384xf32>
    %v9563 = stablehlo.subtract %s2b7lg, %v9562 : tensor<384xf32>
    %v9564 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9565 = stablehlo.multiply %v9564, %v9557 : tensor<384xf32>
    %v9566 = stablehlo.multiply %v9565, %s2b7lg : tensor<384xf32>
    %v9567 = stablehlo.subtract %v9563, %v9566 : tensor<384xf32>
    %v9568 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9569 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9570 = stablehlo.multiply %v9568, %s2b8dWm : tensor<384x1x7x7xf32>
    %v9571 = stablehlo.multiply %v9569, %v2106 : tensor<384x1x7x7xf32>
    %v9572 = stablehlo.add %v9570, %v9571 : tensor<384x1x7x7xf32>
    %v9573 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9574 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9575 = stablehlo.multiply %v9573, %s2b8dWv : tensor<384x1x7x7xf32>
    %v9576 = stablehlo.multiply %v2106, %v2106 : tensor<384x1x7x7xf32>
    %v9577 = stablehlo.multiply %v9574, %v9576 : tensor<384x1x7x7xf32>
    %v9578 = stablehlo.add %v9575, %v9577 : tensor<384x1x7x7xf32>
    %v9579 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9580 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9581 = stablehlo.multiply %v9579, %s2b8dWm : tensor<384x1x7x7xf32>
    %v9582 = stablehlo.multiply %v9580, %v2106 : tensor<384x1x7x7xf32>
    %v9583 = stablehlo.add %v9581, %v9582 : tensor<384x1x7x7xf32>
    %v9584 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9585 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9586 = stablehlo.multiply %v9584, %s2b8dWv : tensor<384x1x7x7xf32>
    %v9587 = stablehlo.multiply %v2106, %v2106 : tensor<384x1x7x7xf32>
    %v9588 = stablehlo.multiply %v9585, %v9587 : tensor<384x1x7x7xf32>
    %v9589 = stablehlo.add %v9586, %v9588 : tensor<384x1x7x7xf32>
    %v9590 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9591 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9592 = stablehlo.divide %v9583, %v9590 : tensor<384x1x7x7xf32>
    %v9593 = stablehlo.divide %v9589, %v9591 : tensor<384x1x7x7xf32>
    %v9594 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9595 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9596 = stablehlo.sqrt %v9593 : tensor<384x1x7x7xf32>
    %v9597 = stablehlo.add %v9596, %v9595 : tensor<384x1x7x7xf32>
    %v9598 = stablehlo.divide %v9592, %v9597 : tensor<384x1x7x7xf32>
    %v9599 = stablehlo.multiply %v9594, %v9598 : tensor<384x1x7x7xf32>
    %v9600 = stablehlo.subtract %s2b8dW, %v9599 : tensor<384x1x7x7xf32>
    %v9601 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1x7x7xf32>
    %v9602 = stablehlo.multiply %v9601, %v9594 : tensor<384x1x7x7xf32>
    %v9603 = stablehlo.multiply %v9602, %s2b8dW : tensor<384x1x7x7xf32>
    %v9604 = stablehlo.subtract %v9600, %v9603 : tensor<384x1x7x7xf32>
    %v9605 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9606 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9607 = stablehlo.multiply %v9605, %s2b8dbm : tensor<384xf32>
    %v9608 = stablehlo.multiply %v9606, %v2109 : tensor<384xf32>
    %v9609 = stablehlo.add %v9607, %v9608 : tensor<384xf32>
    %v9610 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9611 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9612 = stablehlo.multiply %v9610, %s2b8dbv : tensor<384xf32>
    %v9613 = stablehlo.multiply %v2109, %v2109 : tensor<384xf32>
    %v9614 = stablehlo.multiply %v9611, %v9613 : tensor<384xf32>
    %v9615 = stablehlo.add %v9612, %v9614 : tensor<384xf32>
    %v9616 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9617 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9618 = stablehlo.multiply %v9616, %s2b8dbm : tensor<384xf32>
    %v9619 = stablehlo.multiply %v9617, %v2109 : tensor<384xf32>
    %v9620 = stablehlo.add %v9618, %v9619 : tensor<384xf32>
    %v9621 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9622 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9623 = stablehlo.multiply %v9621, %s2b8dbv : tensor<384xf32>
    %v9624 = stablehlo.multiply %v2109, %v2109 : tensor<384xf32>
    %v9625 = stablehlo.multiply %v9622, %v9624 : tensor<384xf32>
    %v9626 = stablehlo.add %v9623, %v9625 : tensor<384xf32>
    %v9627 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9628 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9629 = stablehlo.divide %v9620, %v9627 : tensor<384xf32>
    %v9630 = stablehlo.divide %v9626, %v9628 : tensor<384xf32>
    %v9631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9632 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9633 = stablehlo.sqrt %v9630 : tensor<384xf32>
    %v9634 = stablehlo.add %v9633, %v9632 : tensor<384xf32>
    %v9635 = stablehlo.divide %v9629, %v9634 : tensor<384xf32>
    %v9636 = stablehlo.multiply %v9631, %v9635 : tensor<384xf32>
    %v9637 = stablehlo.subtract %s2b8db, %v9636 : tensor<384xf32>
    %v9638 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9639 = stablehlo.multiply %v9638, %v9631 : tensor<384xf32>
    %v9640 = stablehlo.multiply %v9639, %s2b8db : tensor<384xf32>
    %v9641 = stablehlo.subtract %v9637, %v9640 : tensor<384xf32>
    %v9642 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9643 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9644 = stablehlo.multiply %v9642, %s2b8ngm : tensor<384xf32>
    %v9645 = stablehlo.multiply %v9643, %v2094 : tensor<384xf32>
    %v9646 = stablehlo.add %v9644, %v9645 : tensor<384xf32>
    %v9647 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9648 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9649 = stablehlo.multiply %v9647, %s2b8ngv : tensor<384xf32>
    %v9650 = stablehlo.multiply %v2094, %v2094 : tensor<384xf32>
    %v9651 = stablehlo.multiply %v9648, %v9650 : tensor<384xf32>
    %v9652 = stablehlo.add %v9649, %v9651 : tensor<384xf32>
    %v9653 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9654 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9655 = stablehlo.multiply %v9653, %s2b8ngm : tensor<384xf32>
    %v9656 = stablehlo.multiply %v9654, %v2094 : tensor<384xf32>
    %v9657 = stablehlo.add %v9655, %v9656 : tensor<384xf32>
    %v9658 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9659 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9660 = stablehlo.multiply %v9658, %s2b8ngv : tensor<384xf32>
    %v9661 = stablehlo.multiply %v2094, %v2094 : tensor<384xf32>
    %v9662 = stablehlo.multiply %v9659, %v9661 : tensor<384xf32>
    %v9663 = stablehlo.add %v9660, %v9662 : tensor<384xf32>
    %v9664 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9665 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9666 = stablehlo.divide %v9657, %v9664 : tensor<384xf32>
    %v9667 = stablehlo.divide %v9663, %v9665 : tensor<384xf32>
    %v9668 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9669 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9670 = stablehlo.sqrt %v9667 : tensor<384xf32>
    %v9671 = stablehlo.add %v9670, %v9669 : tensor<384xf32>
    %v9672 = stablehlo.divide %v9666, %v9671 : tensor<384xf32>
    %v9673 = stablehlo.multiply %v9668, %v9672 : tensor<384xf32>
    %v9674 = stablehlo.subtract %s2b8ng, %v9673 : tensor<384xf32>
    %v9675 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9676 = stablehlo.multiply %v9675, %v9668 : tensor<384xf32>
    %v9677 = stablehlo.multiply %v9676, %s2b8ng : tensor<384xf32>
    %v9678 = stablehlo.subtract %v9674, %v9677 : tensor<384xf32>
    %v9679 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9680 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9681 = stablehlo.multiply %v9679, %s2b8nbtm : tensor<384xf32>
    %v9682 = stablehlo.multiply %v9680, %v2100 : tensor<384xf32>
    %v9683 = stablehlo.add %v9681, %v9682 : tensor<384xf32>
    %v9684 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9685 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9686 = stablehlo.multiply %v9684, %s2b8nbtv : tensor<384xf32>
    %v9687 = stablehlo.multiply %v2100, %v2100 : tensor<384xf32>
    %v9688 = stablehlo.multiply %v9685, %v9687 : tensor<384xf32>
    %v9689 = stablehlo.add %v9686, %v9688 : tensor<384xf32>
    %v9690 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9691 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9692 = stablehlo.multiply %v9690, %s2b8nbtm : tensor<384xf32>
    %v9693 = stablehlo.multiply %v9691, %v2100 : tensor<384xf32>
    %v9694 = stablehlo.add %v9692, %v9693 : tensor<384xf32>
    %v9695 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9696 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9697 = stablehlo.multiply %v9695, %s2b8nbtv : tensor<384xf32>
    %v9698 = stablehlo.multiply %v2100, %v2100 : tensor<384xf32>
    %v9699 = stablehlo.multiply %v9696, %v9698 : tensor<384xf32>
    %v9700 = stablehlo.add %v9697, %v9699 : tensor<384xf32>
    %v9701 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9702 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9703 = stablehlo.divide %v9694, %v9701 : tensor<384xf32>
    %v9704 = stablehlo.divide %v9700, %v9702 : tensor<384xf32>
    %v9705 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9706 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9707 = stablehlo.sqrt %v9704 : tensor<384xf32>
    %v9708 = stablehlo.add %v9707, %v9706 : tensor<384xf32>
    %v9709 = stablehlo.divide %v9703, %v9708 : tensor<384xf32>
    %v9710 = stablehlo.multiply %v9705, %v9709 : tensor<384xf32>
    %v9711 = stablehlo.subtract %s2b8nbt, %v9710 : tensor<384xf32>
    %v9712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9713 = stablehlo.multiply %v9712, %v9705 : tensor<384xf32>
    %v9714 = stablehlo.multiply %v9713, %s2b8nbt : tensor<384xf32>
    %v9715 = stablehlo.subtract %v9711, %v9714 : tensor<384xf32>
    %v9716 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9717 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9718 = stablehlo.multiply %v9716, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v9719 = stablehlo.multiply %v9717, %v2067 : tensor<1536x384x1x1xf32>
    %v9720 = stablehlo.add %v9718, %v9719 : tensor<1536x384x1x1xf32>
    %v9721 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9722 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9723 = stablehlo.multiply %v9721, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v9724 = stablehlo.multiply %v2067, %v2067 : tensor<1536x384x1x1xf32>
    %v9725 = stablehlo.multiply %v9722, %v9724 : tensor<1536x384x1x1xf32>
    %v9726 = stablehlo.add %v9723, %v9725 : tensor<1536x384x1x1xf32>
    %v9727 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9728 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9729 = stablehlo.multiply %v9727, %s2b8eWm : tensor<1536x384x1x1xf32>
    %v9730 = stablehlo.multiply %v9728, %v2067 : tensor<1536x384x1x1xf32>
    %v9731 = stablehlo.add %v9729, %v9730 : tensor<1536x384x1x1xf32>
    %v9732 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9733 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9734 = stablehlo.multiply %v9732, %s2b8eWv : tensor<1536x384x1x1xf32>
    %v9735 = stablehlo.multiply %v2067, %v2067 : tensor<1536x384x1x1xf32>
    %v9736 = stablehlo.multiply %v9733, %v9735 : tensor<1536x384x1x1xf32>
    %v9737 = stablehlo.add %v9734, %v9736 : tensor<1536x384x1x1xf32>
    %v9738 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9739 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9740 = stablehlo.divide %v9731, %v9738 : tensor<1536x384x1x1xf32>
    %v9741 = stablehlo.divide %v9737, %v9739 : tensor<1536x384x1x1xf32>
    %v9742 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9743 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9744 = stablehlo.sqrt %v9741 : tensor<1536x384x1x1xf32>
    %v9745 = stablehlo.add %v9744, %v9743 : tensor<1536x384x1x1xf32>
    %v9746 = stablehlo.divide %v9740, %v9745 : tensor<1536x384x1x1xf32>
    %v9747 = stablehlo.multiply %v9742, %v9746 : tensor<1536x384x1x1xf32>
    %v9748 = stablehlo.subtract %s2b8eW, %v9747 : tensor<1536x384x1x1xf32>
    %v9749 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536x384x1x1xf32>
    %v9750 = stablehlo.multiply %v9749, %v9742 : tensor<1536x384x1x1xf32>
    %v9751 = stablehlo.multiply %v9750, %s2b8eW : tensor<1536x384x1x1xf32>
    %v9752 = stablehlo.subtract %v9748, %v9751 : tensor<1536x384x1x1xf32>
    %v9753 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9754 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9755 = stablehlo.multiply %v9753, %s2b8ebm : tensor<1536xf32>
    %v9756 = stablehlo.multiply %v9754, %v2070 : tensor<1536xf32>
    %v9757 = stablehlo.add %v9755, %v9756 : tensor<1536xf32>
    %v9758 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9759 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9760 = stablehlo.multiply %v9758, %s2b8ebv : tensor<1536xf32>
    %v9761 = stablehlo.multiply %v2070, %v2070 : tensor<1536xf32>
    %v9762 = stablehlo.multiply %v9759, %v9761 : tensor<1536xf32>
    %v9763 = stablehlo.add %v9760, %v9762 : tensor<1536xf32>
    %v9764 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9765 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9766 = stablehlo.multiply %v9764, %s2b8ebm : tensor<1536xf32>
    %v9767 = stablehlo.multiply %v9765, %v2070 : tensor<1536xf32>
    %v9768 = stablehlo.add %v9766, %v9767 : tensor<1536xf32>
    %v9769 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9770 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9771 = stablehlo.multiply %v9769, %s2b8ebv : tensor<1536xf32>
    %v9772 = stablehlo.multiply %v2070, %v2070 : tensor<1536xf32>
    %v9773 = stablehlo.multiply %v9770, %v9772 : tensor<1536xf32>
    %v9774 = stablehlo.add %v9771, %v9773 : tensor<1536xf32>
    %v9775 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9776 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9777 = stablehlo.divide %v9768, %v9775 : tensor<1536xf32>
    %v9778 = stablehlo.divide %v9774, %v9776 : tensor<1536xf32>
    %v9779 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9780 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9781 = stablehlo.sqrt %v9778 : tensor<1536xf32>
    %v9782 = stablehlo.add %v9781, %v9780 : tensor<1536xf32>
    %v9783 = stablehlo.divide %v9777, %v9782 : tensor<1536xf32>
    %v9784 = stablehlo.multiply %v9779, %v9783 : tensor<1536xf32>
    %v9785 = stablehlo.subtract %s2b8eb, %v9784 : tensor<1536xf32>
    %v9786 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1536xf32>
    %v9787 = stablehlo.multiply %v9786, %v9779 : tensor<1536xf32>
    %v9788 = stablehlo.multiply %v9787, %s2b8eb : tensor<1536xf32>
    %v9789 = stablehlo.subtract %v9785, %v9788 : tensor<1536xf32>
    %v9790 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9791 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9792 = stablehlo.multiply %v9790, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v9793 = stablehlo.multiply %v9791, %v2058 : tensor<384x1536x1x1xf32>
    %v9794 = stablehlo.add %v9792, %v9793 : tensor<384x1536x1x1xf32>
    %v9795 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9796 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9797 = stablehlo.multiply %v9795, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v9798 = stablehlo.multiply %v2058, %v2058 : tensor<384x1536x1x1xf32>
    %v9799 = stablehlo.multiply %v9796, %v9798 : tensor<384x1536x1x1xf32>
    %v9800 = stablehlo.add %v9797, %v9799 : tensor<384x1536x1x1xf32>
    %v9801 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9802 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9803 = stablehlo.multiply %v9801, %s2b8pWm : tensor<384x1536x1x1xf32>
    %v9804 = stablehlo.multiply %v9802, %v2058 : tensor<384x1536x1x1xf32>
    %v9805 = stablehlo.add %v9803, %v9804 : tensor<384x1536x1x1xf32>
    %v9806 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9807 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9808 = stablehlo.multiply %v9806, %s2b8pWv : tensor<384x1536x1x1xf32>
    %v9809 = stablehlo.multiply %v2058, %v2058 : tensor<384x1536x1x1xf32>
    %v9810 = stablehlo.multiply %v9807, %v9809 : tensor<384x1536x1x1xf32>
    %v9811 = stablehlo.add %v9808, %v9810 : tensor<384x1536x1x1xf32>
    %v9812 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9813 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9814 = stablehlo.divide %v9805, %v9812 : tensor<384x1536x1x1xf32>
    %v9815 = stablehlo.divide %v9811, %v9813 : tensor<384x1536x1x1xf32>
    %v9816 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9817 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9818 = stablehlo.sqrt %v9815 : tensor<384x1536x1x1xf32>
    %v9819 = stablehlo.add %v9818, %v9817 : tensor<384x1536x1x1xf32>
    %v9820 = stablehlo.divide %v9814, %v9819 : tensor<384x1536x1x1xf32>
    %v9821 = stablehlo.multiply %v9816, %v9820 : tensor<384x1536x1x1xf32>
    %v9822 = stablehlo.subtract %s2b8pW, %v9821 : tensor<384x1536x1x1xf32>
    %v9823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384x1536x1x1xf32>
    %v9824 = stablehlo.multiply %v9823, %v9816 : tensor<384x1536x1x1xf32>
    %v9825 = stablehlo.multiply %v9824, %s2b8pW : tensor<384x1536x1x1xf32>
    %v9826 = stablehlo.subtract %v9822, %v9825 : tensor<384x1536x1x1xf32>
    %v9827 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9828 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9829 = stablehlo.multiply %v9827, %s2b8pbm : tensor<384xf32>
    %v9830 = stablehlo.multiply %v9828, %v2061 : tensor<384xf32>
    %v9831 = stablehlo.add %v9829, %v9830 : tensor<384xf32>
    %v9832 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9833 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9834 = stablehlo.multiply %v9832, %s2b8pbv : tensor<384xf32>
    %v9835 = stablehlo.multiply %v2061, %v2061 : tensor<384xf32>
    %v9836 = stablehlo.multiply %v9833, %v9835 : tensor<384xf32>
    %v9837 = stablehlo.add %v9834, %v9836 : tensor<384xf32>
    %v9838 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9839 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9840 = stablehlo.multiply %v9838, %s2b8pbm : tensor<384xf32>
    %v9841 = stablehlo.multiply %v9839, %v2061 : tensor<384xf32>
    %v9842 = stablehlo.add %v9840, %v9841 : tensor<384xf32>
    %v9843 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9844 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9845 = stablehlo.multiply %v9843, %s2b8pbv : tensor<384xf32>
    %v9846 = stablehlo.multiply %v2061, %v2061 : tensor<384xf32>
    %v9847 = stablehlo.multiply %v9844, %v9846 : tensor<384xf32>
    %v9848 = stablehlo.add %v9845, %v9847 : tensor<384xf32>
    %v9849 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9850 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9851 = stablehlo.divide %v9842, %v9849 : tensor<384xf32>
    %v9852 = stablehlo.divide %v9848, %v9850 : tensor<384xf32>
    %v9853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9854 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9855 = stablehlo.sqrt %v9852 : tensor<384xf32>
    %v9856 = stablehlo.add %v9855, %v9854 : tensor<384xf32>
    %v9857 = stablehlo.divide %v9851, %v9856 : tensor<384xf32>
    %v9858 = stablehlo.multiply %v9853, %v9857 : tensor<384xf32>
    %v9859 = stablehlo.subtract %s2b8pb, %v9858 : tensor<384xf32>
    %v9860 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9861 = stablehlo.multiply %v9860, %v9853 : tensor<384xf32>
    %v9862 = stablehlo.multiply %v9861, %s2b8pb : tensor<384xf32>
    %v9863 = stablehlo.subtract %v9859, %v9862 : tensor<384xf32>
    %v9864 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9865 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9866 = stablehlo.multiply %v9864, %s2b8lgm : tensor<384xf32>
    %v9867 = stablehlo.multiply %v9865, %v2052 : tensor<384xf32>
    %v9868 = stablehlo.add %v9866, %v9867 : tensor<384xf32>
    %v9869 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9870 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9871 = stablehlo.multiply %v9869, %s2b8lgv : tensor<384xf32>
    %v9872 = stablehlo.multiply %v2052, %v2052 : tensor<384xf32>
    %v9873 = stablehlo.multiply %v9870, %v9872 : tensor<384xf32>
    %v9874 = stablehlo.add %v9871, %v9873 : tensor<384xf32>
    %v9875 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9876 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9877 = stablehlo.multiply %v9875, %s2b8lgm : tensor<384xf32>
    %v9878 = stablehlo.multiply %v9876, %v2052 : tensor<384xf32>
    %v9879 = stablehlo.add %v9877, %v9878 : tensor<384xf32>
    %v9880 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9881 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9882 = stablehlo.multiply %v9880, %s2b8lgv : tensor<384xf32>
    %v9883 = stablehlo.multiply %v2052, %v2052 : tensor<384xf32>
    %v9884 = stablehlo.multiply %v9881, %v9883 : tensor<384xf32>
    %v9885 = stablehlo.add %v9882, %v9884 : tensor<384xf32>
    %v9886 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9887 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9888 = stablehlo.divide %v9879, %v9886 : tensor<384xf32>
    %v9889 = stablehlo.divide %v9885, %v9887 : tensor<384xf32>
    %v9890 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9891 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9892 = stablehlo.sqrt %v9889 : tensor<384xf32>
    %v9893 = stablehlo.add %v9892, %v9891 : tensor<384xf32>
    %v9894 = stablehlo.divide %v9888, %v9893 : tensor<384xf32>
    %v9895 = stablehlo.multiply %v9890, %v9894 : tensor<384xf32>
    %v9896 = stablehlo.subtract %s2b8lg, %v9895 : tensor<384xf32>
    %v9897 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9898 = stablehlo.multiply %v9897, %v9890 : tensor<384xf32>
    %v9899 = stablehlo.multiply %v9898, %s2b8lg : tensor<384xf32>
    %v9900 = stablehlo.subtract %v9896, %v9899 : tensor<384xf32>
    %v9901 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9902 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9903 = stablehlo.multiply %v9901, %d2ngm : tensor<384xf32>
    %v9904 = stablehlo.multiply %v9902, %v1946 : tensor<384xf32>
    %v9905 = stablehlo.add %v9903, %v9904 : tensor<384xf32>
    %v9906 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9907 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9908 = stablehlo.multiply %v9906, %d2ngv : tensor<384xf32>
    %v9909 = stablehlo.multiply %v1946, %v1946 : tensor<384xf32>
    %v9910 = stablehlo.multiply %v9907, %v9909 : tensor<384xf32>
    %v9911 = stablehlo.add %v9908, %v9910 : tensor<384xf32>
    %v9912 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9913 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9914 = stablehlo.multiply %v9912, %d2ngm : tensor<384xf32>
    %v9915 = stablehlo.multiply %v9913, %v1946 : tensor<384xf32>
    %v9916 = stablehlo.add %v9914, %v9915 : tensor<384xf32>
    %v9917 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9918 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9919 = stablehlo.multiply %v9917, %d2ngv : tensor<384xf32>
    %v9920 = stablehlo.multiply %v1946, %v1946 : tensor<384xf32>
    %v9921 = stablehlo.multiply %v9918, %v9920 : tensor<384xf32>
    %v9922 = stablehlo.add %v9919, %v9921 : tensor<384xf32>
    %v9923 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9924 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9925 = stablehlo.divide %v9916, %v9923 : tensor<384xf32>
    %v9926 = stablehlo.divide %v9922, %v9924 : tensor<384xf32>
    %v9927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9928 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9929 = stablehlo.sqrt %v9926 : tensor<384xf32>
    %v9930 = stablehlo.add %v9929, %v9928 : tensor<384xf32>
    %v9931 = stablehlo.divide %v9925, %v9930 : tensor<384xf32>
    %v9932 = stablehlo.multiply %v9927, %v9931 : tensor<384xf32>
    %v9933 = stablehlo.subtract %d2ng, %v9932 : tensor<384xf32>
    %v9934 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9935 = stablehlo.multiply %v9934, %v9927 : tensor<384xf32>
    %v9936 = stablehlo.multiply %v9935, %d2ng : tensor<384xf32>
    %v9937 = stablehlo.subtract %v9933, %v9936 : tensor<384xf32>
    %v9938 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9939 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9940 = stablehlo.multiply %v9938, %d2nbtm : tensor<384xf32>
    %v9941 = stablehlo.multiply %v9939, %v1952 : tensor<384xf32>
    %v9942 = stablehlo.add %v9940, %v9941 : tensor<384xf32>
    %v9943 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9944 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9945 = stablehlo.multiply %v9943, %d2nbtv : tensor<384xf32>
    %v9946 = stablehlo.multiply %v1952, %v1952 : tensor<384xf32>
    %v9947 = stablehlo.multiply %v9944, %v9946 : tensor<384xf32>
    %v9948 = stablehlo.add %v9945, %v9947 : tensor<384xf32>
    %v9949 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9950 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9951 = stablehlo.multiply %v9949, %d2nbtm : tensor<384xf32>
    %v9952 = stablehlo.multiply %v9950, %v1952 : tensor<384xf32>
    %v9953 = stablehlo.add %v9951, %v9952 : tensor<384xf32>
    %v9954 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9955 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9956 = stablehlo.multiply %v9954, %d2nbtv : tensor<384xf32>
    %v9957 = stablehlo.multiply %v1952, %v1952 : tensor<384xf32>
    %v9958 = stablehlo.multiply %v9955, %v9957 : tensor<384xf32>
    %v9959 = stablehlo.add %v9956, %v9958 : tensor<384xf32>
    %v9960 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9961 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9962 = stablehlo.divide %v9953, %v9960 : tensor<384xf32>
    %v9963 = stablehlo.divide %v9959, %v9961 : tensor<384xf32>
    %v9964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9965 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9966 = stablehlo.sqrt %v9963 : tensor<384xf32>
    %v9967 = stablehlo.add %v9966, %v9965 : tensor<384xf32>
    %v9968 = stablehlo.divide %v9962, %v9967 : tensor<384xf32>
    %v9969 = stablehlo.multiply %v9964, %v9968 : tensor<384xf32>
    %v9970 = stablehlo.subtract %d2nbt, %v9969 : tensor<384xf32>
    %v9971 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<384xf32>
    %v9972 = stablehlo.multiply %v9971, %v9964 : tensor<384xf32>
    %v9973 = stablehlo.multiply %v9972, %d2nbt : tensor<384xf32>
    %v9974 = stablehlo.subtract %v9970, %v9973 : tensor<384xf32>
    %v9975 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9976 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9977 = stablehlo.multiply %v9975, %d2Wm : tensor<768x384x2x2xf32>
    %v9978 = stablehlo.multiply %v9976, %v1960 : tensor<768x384x2x2xf32>
    %v9979 = stablehlo.add %v9977, %v9978 : tensor<768x384x2x2xf32>
    %v9980 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9981 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9982 = stablehlo.multiply %v9980, %d2Wv : tensor<768x384x2x2xf32>
    %v9983 = stablehlo.multiply %v1960, %v1960 : tensor<768x384x2x2xf32>
    %v9984 = stablehlo.multiply %v9981, %v9983 : tensor<768x384x2x2xf32>
    %v9985 = stablehlo.add %v9982, %v9984 : tensor<768x384x2x2xf32>
    %v9986 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9987 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9988 = stablehlo.multiply %v9986, %d2Wm : tensor<768x384x2x2xf32>
    %v9989 = stablehlo.multiply %v9987, %v1960 : tensor<768x384x2x2xf32>
    %v9990 = stablehlo.add %v9988, %v9989 : tensor<768x384x2x2xf32>
    %v9991 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9992 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9993 = stablehlo.multiply %v9991, %d2Wv : tensor<768x384x2x2xf32>
    %v9994 = stablehlo.multiply %v1960, %v1960 : tensor<768x384x2x2xf32>
    %v9995 = stablehlo.multiply %v9992, %v9994 : tensor<768x384x2x2xf32>
    %v9996 = stablehlo.add %v9993, %v9995 : tensor<768x384x2x2xf32>
    %v9997 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9998 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v9999 = stablehlo.divide %v9990, %v9997 : tensor<768x384x2x2xf32>
    %v10000 = stablehlo.divide %v9996, %v9998 : tensor<768x384x2x2xf32>
    %v10001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v10002 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v10003 = stablehlo.sqrt %v10000 : tensor<768x384x2x2xf32>
    %v10004 = stablehlo.add %v10003, %v10002 : tensor<768x384x2x2xf32>
    %v10005 = stablehlo.divide %v9999, %v10004 : tensor<768x384x2x2xf32>
    %v10006 = stablehlo.multiply %v10001, %v10005 : tensor<768x384x2x2xf32>
    %v10007 = stablehlo.subtract %d2W, %v10006 : tensor<768x384x2x2xf32>
    %v10008 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x384x2x2xf32>
    %v10009 = stablehlo.multiply %v10008, %v10001 : tensor<768x384x2x2xf32>
    %v10010 = stablehlo.multiply %v10009, %d2W : tensor<768x384x2x2xf32>
    %v10011 = stablehlo.subtract %v10007, %v10010 : tensor<768x384x2x2xf32>
    %v10012 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10013 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10014 = stablehlo.multiply %v10012, %d2bm : tensor<768xf32>
    %v10015 = stablehlo.multiply %v10013, %v1922 : tensor<768xf32>
    %v10016 = stablehlo.add %v10014, %v10015 : tensor<768xf32>
    %v10017 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10018 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10019 = stablehlo.multiply %v10017, %d2bv : tensor<768xf32>
    %v10020 = stablehlo.multiply %v1922, %v1922 : tensor<768xf32>
    %v10021 = stablehlo.multiply %v10018, %v10020 : tensor<768xf32>
    %v10022 = stablehlo.add %v10019, %v10021 : tensor<768xf32>
    %v10023 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10024 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10025 = stablehlo.multiply %v10023, %d2bm : tensor<768xf32>
    %v10026 = stablehlo.multiply %v10024, %v1922 : tensor<768xf32>
    %v10027 = stablehlo.add %v10025, %v10026 : tensor<768xf32>
    %v10028 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10029 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10030 = stablehlo.multiply %v10028, %d2bv : tensor<768xf32>
    %v10031 = stablehlo.multiply %v1922, %v1922 : tensor<768xf32>
    %v10032 = stablehlo.multiply %v10029, %v10031 : tensor<768xf32>
    %v10033 = stablehlo.add %v10030, %v10032 : tensor<768xf32>
    %v10034 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10035 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10036 = stablehlo.divide %v10027, %v10034 : tensor<768xf32>
    %v10037 = stablehlo.divide %v10033, %v10035 : tensor<768xf32>
    %v10038 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10039 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10040 = stablehlo.sqrt %v10037 : tensor<768xf32>
    %v10041 = stablehlo.add %v10040, %v10039 : tensor<768xf32>
    %v10042 = stablehlo.divide %v10036, %v10041 : tensor<768xf32>
    %v10043 = stablehlo.multiply %v10038, %v10042 : tensor<768xf32>
    %v10044 = stablehlo.subtract %d2b, %v10043 : tensor<768xf32>
    %v10045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10046 = stablehlo.multiply %v10045, %v10038 : tensor<768xf32>
    %v10047 = stablehlo.multiply %v10046, %d2b : tensor<768xf32>
    %v10048 = stablehlo.subtract %v10044, %v10047 : tensor<768xf32>
    %v10049 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10050 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10051 = stablehlo.multiply %v10049, %s3b0dWm : tensor<768x1x7x7xf32>
    %v10052 = stablehlo.multiply %v10050, %v1866 : tensor<768x1x7x7xf32>
    %v10053 = stablehlo.add %v10051, %v10052 : tensor<768x1x7x7xf32>
    %v10054 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10055 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10056 = stablehlo.multiply %v10054, %s3b0dWv : tensor<768x1x7x7xf32>
    %v10057 = stablehlo.multiply %v1866, %v1866 : tensor<768x1x7x7xf32>
    %v10058 = stablehlo.multiply %v10055, %v10057 : tensor<768x1x7x7xf32>
    %v10059 = stablehlo.add %v10056, %v10058 : tensor<768x1x7x7xf32>
    %v10060 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10061 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10062 = stablehlo.multiply %v10060, %s3b0dWm : tensor<768x1x7x7xf32>
    %v10063 = stablehlo.multiply %v10061, %v1866 : tensor<768x1x7x7xf32>
    %v10064 = stablehlo.add %v10062, %v10063 : tensor<768x1x7x7xf32>
    %v10065 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10066 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10067 = stablehlo.multiply %v10065, %s3b0dWv : tensor<768x1x7x7xf32>
    %v10068 = stablehlo.multiply %v1866, %v1866 : tensor<768x1x7x7xf32>
    %v10069 = stablehlo.multiply %v10066, %v10068 : tensor<768x1x7x7xf32>
    %v10070 = stablehlo.add %v10067, %v10069 : tensor<768x1x7x7xf32>
    %v10071 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10072 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10073 = stablehlo.divide %v10064, %v10071 : tensor<768x1x7x7xf32>
    %v10074 = stablehlo.divide %v10070, %v10072 : tensor<768x1x7x7xf32>
    %v10075 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10076 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10077 = stablehlo.sqrt %v10074 : tensor<768x1x7x7xf32>
    %v10078 = stablehlo.add %v10077, %v10076 : tensor<768x1x7x7xf32>
    %v10079 = stablehlo.divide %v10073, %v10078 : tensor<768x1x7x7xf32>
    %v10080 = stablehlo.multiply %v10075, %v10079 : tensor<768x1x7x7xf32>
    %v10081 = stablehlo.subtract %s3b0dW, %v10080 : tensor<768x1x7x7xf32>
    %v10082 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10083 = stablehlo.multiply %v10082, %v10075 : tensor<768x1x7x7xf32>
    %v10084 = stablehlo.multiply %v10083, %s3b0dW : tensor<768x1x7x7xf32>
    %v10085 = stablehlo.subtract %v10081, %v10084 : tensor<768x1x7x7xf32>
    %v10086 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10087 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10088 = stablehlo.multiply %v10086, %s3b0dbm : tensor<768xf32>
    %v10089 = stablehlo.multiply %v10087, %v1869 : tensor<768xf32>
    %v10090 = stablehlo.add %v10088, %v10089 : tensor<768xf32>
    %v10091 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10092 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10093 = stablehlo.multiply %v10091, %s3b0dbv : tensor<768xf32>
    %v10094 = stablehlo.multiply %v1869, %v1869 : tensor<768xf32>
    %v10095 = stablehlo.multiply %v10092, %v10094 : tensor<768xf32>
    %v10096 = stablehlo.add %v10093, %v10095 : tensor<768xf32>
    %v10097 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10098 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10099 = stablehlo.multiply %v10097, %s3b0dbm : tensor<768xf32>
    %v10100 = stablehlo.multiply %v10098, %v1869 : tensor<768xf32>
    %v10101 = stablehlo.add %v10099, %v10100 : tensor<768xf32>
    %v10102 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10103 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10104 = stablehlo.multiply %v10102, %s3b0dbv : tensor<768xf32>
    %v10105 = stablehlo.multiply %v1869, %v1869 : tensor<768xf32>
    %v10106 = stablehlo.multiply %v10103, %v10105 : tensor<768xf32>
    %v10107 = stablehlo.add %v10104, %v10106 : tensor<768xf32>
    %v10108 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10109 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10110 = stablehlo.divide %v10101, %v10108 : tensor<768xf32>
    %v10111 = stablehlo.divide %v10107, %v10109 : tensor<768xf32>
    %v10112 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10113 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10114 = stablehlo.sqrt %v10111 : tensor<768xf32>
    %v10115 = stablehlo.add %v10114, %v10113 : tensor<768xf32>
    %v10116 = stablehlo.divide %v10110, %v10115 : tensor<768xf32>
    %v10117 = stablehlo.multiply %v10112, %v10116 : tensor<768xf32>
    %v10118 = stablehlo.subtract %s3b0db, %v10117 : tensor<768xf32>
    %v10119 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10120 = stablehlo.multiply %v10119, %v10112 : tensor<768xf32>
    %v10121 = stablehlo.multiply %v10120, %s3b0db : tensor<768xf32>
    %v10122 = stablehlo.subtract %v10118, %v10121 : tensor<768xf32>
    %v10123 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10124 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10125 = stablehlo.multiply %v10123, %s3b0ngm : tensor<768xf32>
    %v10126 = stablehlo.multiply %v10124, %v1854 : tensor<768xf32>
    %v10127 = stablehlo.add %v10125, %v10126 : tensor<768xf32>
    %v10128 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10129 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10130 = stablehlo.multiply %v10128, %s3b0ngv : tensor<768xf32>
    %v10131 = stablehlo.multiply %v1854, %v1854 : tensor<768xf32>
    %v10132 = stablehlo.multiply %v10129, %v10131 : tensor<768xf32>
    %v10133 = stablehlo.add %v10130, %v10132 : tensor<768xf32>
    %v10134 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10135 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10136 = stablehlo.multiply %v10134, %s3b0ngm : tensor<768xf32>
    %v10137 = stablehlo.multiply %v10135, %v1854 : tensor<768xf32>
    %v10138 = stablehlo.add %v10136, %v10137 : tensor<768xf32>
    %v10139 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10140 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10141 = stablehlo.multiply %v10139, %s3b0ngv : tensor<768xf32>
    %v10142 = stablehlo.multiply %v1854, %v1854 : tensor<768xf32>
    %v10143 = stablehlo.multiply %v10140, %v10142 : tensor<768xf32>
    %v10144 = stablehlo.add %v10141, %v10143 : tensor<768xf32>
    %v10145 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10146 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10147 = stablehlo.divide %v10138, %v10145 : tensor<768xf32>
    %v10148 = stablehlo.divide %v10144, %v10146 : tensor<768xf32>
    %v10149 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10150 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10151 = stablehlo.sqrt %v10148 : tensor<768xf32>
    %v10152 = stablehlo.add %v10151, %v10150 : tensor<768xf32>
    %v10153 = stablehlo.divide %v10147, %v10152 : tensor<768xf32>
    %v10154 = stablehlo.multiply %v10149, %v10153 : tensor<768xf32>
    %v10155 = stablehlo.subtract %s3b0ng, %v10154 : tensor<768xf32>
    %v10156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10157 = stablehlo.multiply %v10156, %v10149 : tensor<768xf32>
    %v10158 = stablehlo.multiply %v10157, %s3b0ng : tensor<768xf32>
    %v10159 = stablehlo.subtract %v10155, %v10158 : tensor<768xf32>
    %v10160 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10161 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10162 = stablehlo.multiply %v10160, %s3b0nbtm : tensor<768xf32>
    %v10163 = stablehlo.multiply %v10161, %v1860 : tensor<768xf32>
    %v10164 = stablehlo.add %v10162, %v10163 : tensor<768xf32>
    %v10165 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10166 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10167 = stablehlo.multiply %v10165, %s3b0nbtv : tensor<768xf32>
    %v10168 = stablehlo.multiply %v1860, %v1860 : tensor<768xf32>
    %v10169 = stablehlo.multiply %v10166, %v10168 : tensor<768xf32>
    %v10170 = stablehlo.add %v10167, %v10169 : tensor<768xf32>
    %v10171 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10172 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10173 = stablehlo.multiply %v10171, %s3b0nbtm : tensor<768xf32>
    %v10174 = stablehlo.multiply %v10172, %v1860 : tensor<768xf32>
    %v10175 = stablehlo.add %v10173, %v10174 : tensor<768xf32>
    %v10176 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10177 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10178 = stablehlo.multiply %v10176, %s3b0nbtv : tensor<768xf32>
    %v10179 = stablehlo.multiply %v1860, %v1860 : tensor<768xf32>
    %v10180 = stablehlo.multiply %v10177, %v10179 : tensor<768xf32>
    %v10181 = stablehlo.add %v10178, %v10180 : tensor<768xf32>
    %v10182 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10183 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10184 = stablehlo.divide %v10175, %v10182 : tensor<768xf32>
    %v10185 = stablehlo.divide %v10181, %v10183 : tensor<768xf32>
    %v10186 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10187 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10188 = stablehlo.sqrt %v10185 : tensor<768xf32>
    %v10189 = stablehlo.add %v10188, %v10187 : tensor<768xf32>
    %v10190 = stablehlo.divide %v10184, %v10189 : tensor<768xf32>
    %v10191 = stablehlo.multiply %v10186, %v10190 : tensor<768xf32>
    %v10192 = stablehlo.subtract %s3b0nbt, %v10191 : tensor<768xf32>
    %v10193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10194 = stablehlo.multiply %v10193, %v10186 : tensor<768xf32>
    %v10195 = stablehlo.multiply %v10194, %s3b0nbt : tensor<768xf32>
    %v10196 = stablehlo.subtract %v10192, %v10195 : tensor<768xf32>
    %v10197 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10198 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10199 = stablehlo.multiply %v10197, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v10200 = stablehlo.multiply %v10198, %v1827 : tensor<3072x768x1x1xf32>
    %v10201 = stablehlo.add %v10199, %v10200 : tensor<3072x768x1x1xf32>
    %v10202 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10203 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10204 = stablehlo.multiply %v10202, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v10205 = stablehlo.multiply %v1827, %v1827 : tensor<3072x768x1x1xf32>
    %v10206 = stablehlo.multiply %v10203, %v10205 : tensor<3072x768x1x1xf32>
    %v10207 = stablehlo.add %v10204, %v10206 : tensor<3072x768x1x1xf32>
    %v10208 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10209 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10210 = stablehlo.multiply %v10208, %s3b0eWm : tensor<3072x768x1x1xf32>
    %v10211 = stablehlo.multiply %v10209, %v1827 : tensor<3072x768x1x1xf32>
    %v10212 = stablehlo.add %v10210, %v10211 : tensor<3072x768x1x1xf32>
    %v10213 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10214 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10215 = stablehlo.multiply %v10213, %s3b0eWv : tensor<3072x768x1x1xf32>
    %v10216 = stablehlo.multiply %v1827, %v1827 : tensor<3072x768x1x1xf32>
    %v10217 = stablehlo.multiply %v10214, %v10216 : tensor<3072x768x1x1xf32>
    %v10218 = stablehlo.add %v10215, %v10217 : tensor<3072x768x1x1xf32>
    %v10219 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10220 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10221 = stablehlo.divide %v10212, %v10219 : tensor<3072x768x1x1xf32>
    %v10222 = stablehlo.divide %v10218, %v10220 : tensor<3072x768x1x1xf32>
    %v10223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10224 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10225 = stablehlo.sqrt %v10222 : tensor<3072x768x1x1xf32>
    %v10226 = stablehlo.add %v10225, %v10224 : tensor<3072x768x1x1xf32>
    %v10227 = stablehlo.divide %v10221, %v10226 : tensor<3072x768x1x1xf32>
    %v10228 = stablehlo.multiply %v10223, %v10227 : tensor<3072x768x1x1xf32>
    %v10229 = stablehlo.subtract %s3b0eW, %v10228 : tensor<3072x768x1x1xf32>
    %v10230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10231 = stablehlo.multiply %v10230, %v10223 : tensor<3072x768x1x1xf32>
    %v10232 = stablehlo.multiply %v10231, %s3b0eW : tensor<3072x768x1x1xf32>
    %v10233 = stablehlo.subtract %v10229, %v10232 : tensor<3072x768x1x1xf32>
    %v10234 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10235 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10236 = stablehlo.multiply %v10234, %s3b0ebm : tensor<3072xf32>
    %v10237 = stablehlo.multiply %v10235, %v1830 : tensor<3072xf32>
    %v10238 = stablehlo.add %v10236, %v10237 : tensor<3072xf32>
    %v10239 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10240 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10241 = stablehlo.multiply %v10239, %s3b0ebv : tensor<3072xf32>
    %v10242 = stablehlo.multiply %v1830, %v1830 : tensor<3072xf32>
    %v10243 = stablehlo.multiply %v10240, %v10242 : tensor<3072xf32>
    %v10244 = stablehlo.add %v10241, %v10243 : tensor<3072xf32>
    %v10245 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10246 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10247 = stablehlo.multiply %v10245, %s3b0ebm : tensor<3072xf32>
    %v10248 = stablehlo.multiply %v10246, %v1830 : tensor<3072xf32>
    %v10249 = stablehlo.add %v10247, %v10248 : tensor<3072xf32>
    %v10250 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10251 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10252 = stablehlo.multiply %v10250, %s3b0ebv : tensor<3072xf32>
    %v10253 = stablehlo.multiply %v1830, %v1830 : tensor<3072xf32>
    %v10254 = stablehlo.multiply %v10251, %v10253 : tensor<3072xf32>
    %v10255 = stablehlo.add %v10252, %v10254 : tensor<3072xf32>
    %v10256 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10257 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10258 = stablehlo.divide %v10249, %v10256 : tensor<3072xf32>
    %v10259 = stablehlo.divide %v10255, %v10257 : tensor<3072xf32>
    %v10260 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10261 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10262 = stablehlo.sqrt %v10259 : tensor<3072xf32>
    %v10263 = stablehlo.add %v10262, %v10261 : tensor<3072xf32>
    %v10264 = stablehlo.divide %v10258, %v10263 : tensor<3072xf32>
    %v10265 = stablehlo.multiply %v10260, %v10264 : tensor<3072xf32>
    %v10266 = stablehlo.subtract %s3b0eb, %v10265 : tensor<3072xf32>
    %v10267 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10268 = stablehlo.multiply %v10267, %v10260 : tensor<3072xf32>
    %v10269 = stablehlo.multiply %v10268, %s3b0eb : tensor<3072xf32>
    %v10270 = stablehlo.subtract %v10266, %v10269 : tensor<3072xf32>
    %v10271 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10272 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10273 = stablehlo.multiply %v10271, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v10274 = stablehlo.multiply %v10272, %v1818 : tensor<768x3072x1x1xf32>
    %v10275 = stablehlo.add %v10273, %v10274 : tensor<768x3072x1x1xf32>
    %v10276 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10277 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10278 = stablehlo.multiply %v10276, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v10279 = stablehlo.multiply %v1818, %v1818 : tensor<768x3072x1x1xf32>
    %v10280 = stablehlo.multiply %v10277, %v10279 : tensor<768x3072x1x1xf32>
    %v10281 = stablehlo.add %v10278, %v10280 : tensor<768x3072x1x1xf32>
    %v10282 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10283 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10284 = stablehlo.multiply %v10282, %s3b0pWm : tensor<768x3072x1x1xf32>
    %v10285 = stablehlo.multiply %v10283, %v1818 : tensor<768x3072x1x1xf32>
    %v10286 = stablehlo.add %v10284, %v10285 : tensor<768x3072x1x1xf32>
    %v10287 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10288 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10289 = stablehlo.multiply %v10287, %s3b0pWv : tensor<768x3072x1x1xf32>
    %v10290 = stablehlo.multiply %v1818, %v1818 : tensor<768x3072x1x1xf32>
    %v10291 = stablehlo.multiply %v10288, %v10290 : tensor<768x3072x1x1xf32>
    %v10292 = stablehlo.add %v10289, %v10291 : tensor<768x3072x1x1xf32>
    %v10293 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10294 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10295 = stablehlo.divide %v10286, %v10293 : tensor<768x3072x1x1xf32>
    %v10296 = stablehlo.divide %v10292, %v10294 : tensor<768x3072x1x1xf32>
    %v10297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10298 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10299 = stablehlo.sqrt %v10296 : tensor<768x3072x1x1xf32>
    %v10300 = stablehlo.add %v10299, %v10298 : tensor<768x3072x1x1xf32>
    %v10301 = stablehlo.divide %v10295, %v10300 : tensor<768x3072x1x1xf32>
    %v10302 = stablehlo.multiply %v10297, %v10301 : tensor<768x3072x1x1xf32>
    %v10303 = stablehlo.subtract %s3b0pW, %v10302 : tensor<768x3072x1x1xf32>
    %v10304 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10305 = stablehlo.multiply %v10304, %v10297 : tensor<768x3072x1x1xf32>
    %v10306 = stablehlo.multiply %v10305, %s3b0pW : tensor<768x3072x1x1xf32>
    %v10307 = stablehlo.subtract %v10303, %v10306 : tensor<768x3072x1x1xf32>
    %v10308 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10309 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10310 = stablehlo.multiply %v10308, %s3b0pbm : tensor<768xf32>
    %v10311 = stablehlo.multiply %v10309, %v1821 : tensor<768xf32>
    %v10312 = stablehlo.add %v10310, %v10311 : tensor<768xf32>
    %v10313 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10314 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10315 = stablehlo.multiply %v10313, %s3b0pbv : tensor<768xf32>
    %v10316 = stablehlo.multiply %v1821, %v1821 : tensor<768xf32>
    %v10317 = stablehlo.multiply %v10314, %v10316 : tensor<768xf32>
    %v10318 = stablehlo.add %v10315, %v10317 : tensor<768xf32>
    %v10319 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10320 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10321 = stablehlo.multiply %v10319, %s3b0pbm : tensor<768xf32>
    %v10322 = stablehlo.multiply %v10320, %v1821 : tensor<768xf32>
    %v10323 = stablehlo.add %v10321, %v10322 : tensor<768xf32>
    %v10324 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10325 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10326 = stablehlo.multiply %v10324, %s3b0pbv : tensor<768xf32>
    %v10327 = stablehlo.multiply %v1821, %v1821 : tensor<768xf32>
    %v10328 = stablehlo.multiply %v10325, %v10327 : tensor<768xf32>
    %v10329 = stablehlo.add %v10326, %v10328 : tensor<768xf32>
    %v10330 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10331 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10332 = stablehlo.divide %v10323, %v10330 : tensor<768xf32>
    %v10333 = stablehlo.divide %v10329, %v10331 : tensor<768xf32>
    %v10334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10335 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10336 = stablehlo.sqrt %v10333 : tensor<768xf32>
    %v10337 = stablehlo.add %v10336, %v10335 : tensor<768xf32>
    %v10338 = stablehlo.divide %v10332, %v10337 : tensor<768xf32>
    %v10339 = stablehlo.multiply %v10334, %v10338 : tensor<768xf32>
    %v10340 = stablehlo.subtract %s3b0pb, %v10339 : tensor<768xf32>
    %v10341 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10342 = stablehlo.multiply %v10341, %v10334 : tensor<768xf32>
    %v10343 = stablehlo.multiply %v10342, %s3b0pb : tensor<768xf32>
    %v10344 = stablehlo.subtract %v10340, %v10343 : tensor<768xf32>
    %v10345 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10346 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10347 = stablehlo.multiply %v10345, %s3b0lgm : tensor<768xf32>
    %v10348 = stablehlo.multiply %v10346, %v1812 : tensor<768xf32>
    %v10349 = stablehlo.add %v10347, %v10348 : tensor<768xf32>
    %v10350 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10351 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10352 = stablehlo.multiply %v10350, %s3b0lgv : tensor<768xf32>
    %v10353 = stablehlo.multiply %v1812, %v1812 : tensor<768xf32>
    %v10354 = stablehlo.multiply %v10351, %v10353 : tensor<768xf32>
    %v10355 = stablehlo.add %v10352, %v10354 : tensor<768xf32>
    %v10356 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10357 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10358 = stablehlo.multiply %v10356, %s3b0lgm : tensor<768xf32>
    %v10359 = stablehlo.multiply %v10357, %v1812 : tensor<768xf32>
    %v10360 = stablehlo.add %v10358, %v10359 : tensor<768xf32>
    %v10361 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10362 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10363 = stablehlo.multiply %v10361, %s3b0lgv : tensor<768xf32>
    %v10364 = stablehlo.multiply %v1812, %v1812 : tensor<768xf32>
    %v10365 = stablehlo.multiply %v10362, %v10364 : tensor<768xf32>
    %v10366 = stablehlo.add %v10363, %v10365 : tensor<768xf32>
    %v10367 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10368 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10369 = stablehlo.divide %v10360, %v10367 : tensor<768xf32>
    %v10370 = stablehlo.divide %v10366, %v10368 : tensor<768xf32>
    %v10371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10372 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10373 = stablehlo.sqrt %v10370 : tensor<768xf32>
    %v10374 = stablehlo.add %v10373, %v10372 : tensor<768xf32>
    %v10375 = stablehlo.divide %v10369, %v10374 : tensor<768xf32>
    %v10376 = stablehlo.multiply %v10371, %v10375 : tensor<768xf32>
    %v10377 = stablehlo.subtract %s3b0lg, %v10376 : tensor<768xf32>
    %v10378 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10379 = stablehlo.multiply %v10378, %v10371 : tensor<768xf32>
    %v10380 = stablehlo.multiply %v10379, %s3b0lg : tensor<768xf32>
    %v10381 = stablehlo.subtract %v10377, %v10380 : tensor<768xf32>
    %v10382 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10383 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10384 = stablehlo.multiply %v10382, %s3b1dWm : tensor<768x1x7x7xf32>
    %v10385 = stablehlo.multiply %v10383, %v1717 : tensor<768x1x7x7xf32>
    %v10386 = stablehlo.add %v10384, %v10385 : tensor<768x1x7x7xf32>
    %v10387 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10388 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10389 = stablehlo.multiply %v10387, %s3b1dWv : tensor<768x1x7x7xf32>
    %v10390 = stablehlo.multiply %v1717, %v1717 : tensor<768x1x7x7xf32>
    %v10391 = stablehlo.multiply %v10388, %v10390 : tensor<768x1x7x7xf32>
    %v10392 = stablehlo.add %v10389, %v10391 : tensor<768x1x7x7xf32>
    %v10393 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10394 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10395 = stablehlo.multiply %v10393, %s3b1dWm : tensor<768x1x7x7xf32>
    %v10396 = stablehlo.multiply %v10394, %v1717 : tensor<768x1x7x7xf32>
    %v10397 = stablehlo.add %v10395, %v10396 : tensor<768x1x7x7xf32>
    %v10398 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10399 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10400 = stablehlo.multiply %v10398, %s3b1dWv : tensor<768x1x7x7xf32>
    %v10401 = stablehlo.multiply %v1717, %v1717 : tensor<768x1x7x7xf32>
    %v10402 = stablehlo.multiply %v10399, %v10401 : tensor<768x1x7x7xf32>
    %v10403 = stablehlo.add %v10400, %v10402 : tensor<768x1x7x7xf32>
    %v10404 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10405 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10406 = stablehlo.divide %v10397, %v10404 : tensor<768x1x7x7xf32>
    %v10407 = stablehlo.divide %v10403, %v10405 : tensor<768x1x7x7xf32>
    %v10408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10409 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10410 = stablehlo.sqrt %v10407 : tensor<768x1x7x7xf32>
    %v10411 = stablehlo.add %v10410, %v10409 : tensor<768x1x7x7xf32>
    %v10412 = stablehlo.divide %v10406, %v10411 : tensor<768x1x7x7xf32>
    %v10413 = stablehlo.multiply %v10408, %v10412 : tensor<768x1x7x7xf32>
    %v10414 = stablehlo.subtract %s3b1dW, %v10413 : tensor<768x1x7x7xf32>
    %v10415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10416 = stablehlo.multiply %v10415, %v10408 : tensor<768x1x7x7xf32>
    %v10417 = stablehlo.multiply %v10416, %s3b1dW : tensor<768x1x7x7xf32>
    %v10418 = stablehlo.subtract %v10414, %v10417 : tensor<768x1x7x7xf32>
    %v10419 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10420 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10421 = stablehlo.multiply %v10419, %s3b1dbm : tensor<768xf32>
    %v10422 = stablehlo.multiply %v10420, %v1720 : tensor<768xf32>
    %v10423 = stablehlo.add %v10421, %v10422 : tensor<768xf32>
    %v10424 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10425 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10426 = stablehlo.multiply %v10424, %s3b1dbv : tensor<768xf32>
    %v10427 = stablehlo.multiply %v1720, %v1720 : tensor<768xf32>
    %v10428 = stablehlo.multiply %v10425, %v10427 : tensor<768xf32>
    %v10429 = stablehlo.add %v10426, %v10428 : tensor<768xf32>
    %v10430 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10431 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10432 = stablehlo.multiply %v10430, %s3b1dbm : tensor<768xf32>
    %v10433 = stablehlo.multiply %v10431, %v1720 : tensor<768xf32>
    %v10434 = stablehlo.add %v10432, %v10433 : tensor<768xf32>
    %v10435 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10436 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10437 = stablehlo.multiply %v10435, %s3b1dbv : tensor<768xf32>
    %v10438 = stablehlo.multiply %v1720, %v1720 : tensor<768xf32>
    %v10439 = stablehlo.multiply %v10436, %v10438 : tensor<768xf32>
    %v10440 = stablehlo.add %v10437, %v10439 : tensor<768xf32>
    %v10441 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10442 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10443 = stablehlo.divide %v10434, %v10441 : tensor<768xf32>
    %v10444 = stablehlo.divide %v10440, %v10442 : tensor<768xf32>
    %v10445 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10446 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10447 = stablehlo.sqrt %v10444 : tensor<768xf32>
    %v10448 = stablehlo.add %v10447, %v10446 : tensor<768xf32>
    %v10449 = stablehlo.divide %v10443, %v10448 : tensor<768xf32>
    %v10450 = stablehlo.multiply %v10445, %v10449 : tensor<768xf32>
    %v10451 = stablehlo.subtract %s3b1db, %v10450 : tensor<768xf32>
    %v10452 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10453 = stablehlo.multiply %v10452, %v10445 : tensor<768xf32>
    %v10454 = stablehlo.multiply %v10453, %s3b1db : tensor<768xf32>
    %v10455 = stablehlo.subtract %v10451, %v10454 : tensor<768xf32>
    %v10456 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10457 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10458 = stablehlo.multiply %v10456, %s3b1ngm : tensor<768xf32>
    %v10459 = stablehlo.multiply %v10457, %v1705 : tensor<768xf32>
    %v10460 = stablehlo.add %v10458, %v10459 : tensor<768xf32>
    %v10461 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10462 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10463 = stablehlo.multiply %v10461, %s3b1ngv : tensor<768xf32>
    %v10464 = stablehlo.multiply %v1705, %v1705 : tensor<768xf32>
    %v10465 = stablehlo.multiply %v10462, %v10464 : tensor<768xf32>
    %v10466 = stablehlo.add %v10463, %v10465 : tensor<768xf32>
    %v10467 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10468 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10469 = stablehlo.multiply %v10467, %s3b1ngm : tensor<768xf32>
    %v10470 = stablehlo.multiply %v10468, %v1705 : tensor<768xf32>
    %v10471 = stablehlo.add %v10469, %v10470 : tensor<768xf32>
    %v10472 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10473 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10474 = stablehlo.multiply %v10472, %s3b1ngv : tensor<768xf32>
    %v10475 = stablehlo.multiply %v1705, %v1705 : tensor<768xf32>
    %v10476 = stablehlo.multiply %v10473, %v10475 : tensor<768xf32>
    %v10477 = stablehlo.add %v10474, %v10476 : tensor<768xf32>
    %v10478 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10479 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10480 = stablehlo.divide %v10471, %v10478 : tensor<768xf32>
    %v10481 = stablehlo.divide %v10477, %v10479 : tensor<768xf32>
    %v10482 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10483 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10484 = stablehlo.sqrt %v10481 : tensor<768xf32>
    %v10485 = stablehlo.add %v10484, %v10483 : tensor<768xf32>
    %v10486 = stablehlo.divide %v10480, %v10485 : tensor<768xf32>
    %v10487 = stablehlo.multiply %v10482, %v10486 : tensor<768xf32>
    %v10488 = stablehlo.subtract %s3b1ng, %v10487 : tensor<768xf32>
    %v10489 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10490 = stablehlo.multiply %v10489, %v10482 : tensor<768xf32>
    %v10491 = stablehlo.multiply %v10490, %s3b1ng : tensor<768xf32>
    %v10492 = stablehlo.subtract %v10488, %v10491 : tensor<768xf32>
    %v10493 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10494 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10495 = stablehlo.multiply %v10493, %s3b1nbtm : tensor<768xf32>
    %v10496 = stablehlo.multiply %v10494, %v1711 : tensor<768xf32>
    %v10497 = stablehlo.add %v10495, %v10496 : tensor<768xf32>
    %v10498 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10499 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10500 = stablehlo.multiply %v10498, %s3b1nbtv : tensor<768xf32>
    %v10501 = stablehlo.multiply %v1711, %v1711 : tensor<768xf32>
    %v10502 = stablehlo.multiply %v10499, %v10501 : tensor<768xf32>
    %v10503 = stablehlo.add %v10500, %v10502 : tensor<768xf32>
    %v10504 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10505 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10506 = stablehlo.multiply %v10504, %s3b1nbtm : tensor<768xf32>
    %v10507 = stablehlo.multiply %v10505, %v1711 : tensor<768xf32>
    %v10508 = stablehlo.add %v10506, %v10507 : tensor<768xf32>
    %v10509 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10510 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10511 = stablehlo.multiply %v10509, %s3b1nbtv : tensor<768xf32>
    %v10512 = stablehlo.multiply %v1711, %v1711 : tensor<768xf32>
    %v10513 = stablehlo.multiply %v10510, %v10512 : tensor<768xf32>
    %v10514 = stablehlo.add %v10511, %v10513 : tensor<768xf32>
    %v10515 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10516 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10517 = stablehlo.divide %v10508, %v10515 : tensor<768xf32>
    %v10518 = stablehlo.divide %v10514, %v10516 : tensor<768xf32>
    %v10519 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10520 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10521 = stablehlo.sqrt %v10518 : tensor<768xf32>
    %v10522 = stablehlo.add %v10521, %v10520 : tensor<768xf32>
    %v10523 = stablehlo.divide %v10517, %v10522 : tensor<768xf32>
    %v10524 = stablehlo.multiply %v10519, %v10523 : tensor<768xf32>
    %v10525 = stablehlo.subtract %s3b1nbt, %v10524 : tensor<768xf32>
    %v10526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10527 = stablehlo.multiply %v10526, %v10519 : tensor<768xf32>
    %v10528 = stablehlo.multiply %v10527, %s3b1nbt : tensor<768xf32>
    %v10529 = stablehlo.subtract %v10525, %v10528 : tensor<768xf32>
    %v10530 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10531 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10532 = stablehlo.multiply %v10530, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v10533 = stablehlo.multiply %v10531, %v1678 : tensor<3072x768x1x1xf32>
    %v10534 = stablehlo.add %v10532, %v10533 : tensor<3072x768x1x1xf32>
    %v10535 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10536 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10537 = stablehlo.multiply %v10535, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v10538 = stablehlo.multiply %v1678, %v1678 : tensor<3072x768x1x1xf32>
    %v10539 = stablehlo.multiply %v10536, %v10538 : tensor<3072x768x1x1xf32>
    %v10540 = stablehlo.add %v10537, %v10539 : tensor<3072x768x1x1xf32>
    %v10541 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10542 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10543 = stablehlo.multiply %v10541, %s3b1eWm : tensor<3072x768x1x1xf32>
    %v10544 = stablehlo.multiply %v10542, %v1678 : tensor<3072x768x1x1xf32>
    %v10545 = stablehlo.add %v10543, %v10544 : tensor<3072x768x1x1xf32>
    %v10546 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10547 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10548 = stablehlo.multiply %v10546, %s3b1eWv : tensor<3072x768x1x1xf32>
    %v10549 = stablehlo.multiply %v1678, %v1678 : tensor<3072x768x1x1xf32>
    %v10550 = stablehlo.multiply %v10547, %v10549 : tensor<3072x768x1x1xf32>
    %v10551 = stablehlo.add %v10548, %v10550 : tensor<3072x768x1x1xf32>
    %v10552 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10553 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10554 = stablehlo.divide %v10545, %v10552 : tensor<3072x768x1x1xf32>
    %v10555 = stablehlo.divide %v10551, %v10553 : tensor<3072x768x1x1xf32>
    %v10556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10557 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10558 = stablehlo.sqrt %v10555 : tensor<3072x768x1x1xf32>
    %v10559 = stablehlo.add %v10558, %v10557 : tensor<3072x768x1x1xf32>
    %v10560 = stablehlo.divide %v10554, %v10559 : tensor<3072x768x1x1xf32>
    %v10561 = stablehlo.multiply %v10556, %v10560 : tensor<3072x768x1x1xf32>
    %v10562 = stablehlo.subtract %s3b1eW, %v10561 : tensor<3072x768x1x1xf32>
    %v10563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10564 = stablehlo.multiply %v10563, %v10556 : tensor<3072x768x1x1xf32>
    %v10565 = stablehlo.multiply %v10564, %s3b1eW : tensor<3072x768x1x1xf32>
    %v10566 = stablehlo.subtract %v10562, %v10565 : tensor<3072x768x1x1xf32>
    %v10567 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10568 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10569 = stablehlo.multiply %v10567, %s3b1ebm : tensor<3072xf32>
    %v10570 = stablehlo.multiply %v10568, %v1681 : tensor<3072xf32>
    %v10571 = stablehlo.add %v10569, %v10570 : tensor<3072xf32>
    %v10572 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10573 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10574 = stablehlo.multiply %v10572, %s3b1ebv : tensor<3072xf32>
    %v10575 = stablehlo.multiply %v1681, %v1681 : tensor<3072xf32>
    %v10576 = stablehlo.multiply %v10573, %v10575 : tensor<3072xf32>
    %v10577 = stablehlo.add %v10574, %v10576 : tensor<3072xf32>
    %v10578 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10579 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10580 = stablehlo.multiply %v10578, %s3b1ebm : tensor<3072xf32>
    %v10581 = stablehlo.multiply %v10579, %v1681 : tensor<3072xf32>
    %v10582 = stablehlo.add %v10580, %v10581 : tensor<3072xf32>
    %v10583 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10584 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10585 = stablehlo.multiply %v10583, %s3b1ebv : tensor<3072xf32>
    %v10586 = stablehlo.multiply %v1681, %v1681 : tensor<3072xf32>
    %v10587 = stablehlo.multiply %v10584, %v10586 : tensor<3072xf32>
    %v10588 = stablehlo.add %v10585, %v10587 : tensor<3072xf32>
    %v10589 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10590 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10591 = stablehlo.divide %v10582, %v10589 : tensor<3072xf32>
    %v10592 = stablehlo.divide %v10588, %v10590 : tensor<3072xf32>
    %v10593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10594 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10595 = stablehlo.sqrt %v10592 : tensor<3072xf32>
    %v10596 = stablehlo.add %v10595, %v10594 : tensor<3072xf32>
    %v10597 = stablehlo.divide %v10591, %v10596 : tensor<3072xf32>
    %v10598 = stablehlo.multiply %v10593, %v10597 : tensor<3072xf32>
    %v10599 = stablehlo.subtract %s3b1eb, %v10598 : tensor<3072xf32>
    %v10600 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10601 = stablehlo.multiply %v10600, %v10593 : tensor<3072xf32>
    %v10602 = stablehlo.multiply %v10601, %s3b1eb : tensor<3072xf32>
    %v10603 = stablehlo.subtract %v10599, %v10602 : tensor<3072xf32>
    %v10604 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10605 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10606 = stablehlo.multiply %v10604, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v10607 = stablehlo.multiply %v10605, %v1669 : tensor<768x3072x1x1xf32>
    %v10608 = stablehlo.add %v10606, %v10607 : tensor<768x3072x1x1xf32>
    %v10609 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10610 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10611 = stablehlo.multiply %v10609, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v10612 = stablehlo.multiply %v1669, %v1669 : tensor<768x3072x1x1xf32>
    %v10613 = stablehlo.multiply %v10610, %v10612 : tensor<768x3072x1x1xf32>
    %v10614 = stablehlo.add %v10611, %v10613 : tensor<768x3072x1x1xf32>
    %v10615 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10616 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10617 = stablehlo.multiply %v10615, %s3b1pWm : tensor<768x3072x1x1xf32>
    %v10618 = stablehlo.multiply %v10616, %v1669 : tensor<768x3072x1x1xf32>
    %v10619 = stablehlo.add %v10617, %v10618 : tensor<768x3072x1x1xf32>
    %v10620 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10621 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10622 = stablehlo.multiply %v10620, %s3b1pWv : tensor<768x3072x1x1xf32>
    %v10623 = stablehlo.multiply %v1669, %v1669 : tensor<768x3072x1x1xf32>
    %v10624 = stablehlo.multiply %v10621, %v10623 : tensor<768x3072x1x1xf32>
    %v10625 = stablehlo.add %v10622, %v10624 : tensor<768x3072x1x1xf32>
    %v10626 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10627 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10628 = stablehlo.divide %v10619, %v10626 : tensor<768x3072x1x1xf32>
    %v10629 = stablehlo.divide %v10625, %v10627 : tensor<768x3072x1x1xf32>
    %v10630 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10631 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10632 = stablehlo.sqrt %v10629 : tensor<768x3072x1x1xf32>
    %v10633 = stablehlo.add %v10632, %v10631 : tensor<768x3072x1x1xf32>
    %v10634 = stablehlo.divide %v10628, %v10633 : tensor<768x3072x1x1xf32>
    %v10635 = stablehlo.multiply %v10630, %v10634 : tensor<768x3072x1x1xf32>
    %v10636 = stablehlo.subtract %s3b1pW, %v10635 : tensor<768x3072x1x1xf32>
    %v10637 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10638 = stablehlo.multiply %v10637, %v10630 : tensor<768x3072x1x1xf32>
    %v10639 = stablehlo.multiply %v10638, %s3b1pW : tensor<768x3072x1x1xf32>
    %v10640 = stablehlo.subtract %v10636, %v10639 : tensor<768x3072x1x1xf32>
    %v10641 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10642 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10643 = stablehlo.multiply %v10641, %s3b1pbm : tensor<768xf32>
    %v10644 = stablehlo.multiply %v10642, %v1672 : tensor<768xf32>
    %v10645 = stablehlo.add %v10643, %v10644 : tensor<768xf32>
    %v10646 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10647 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10648 = stablehlo.multiply %v10646, %s3b1pbv : tensor<768xf32>
    %v10649 = stablehlo.multiply %v1672, %v1672 : tensor<768xf32>
    %v10650 = stablehlo.multiply %v10647, %v10649 : tensor<768xf32>
    %v10651 = stablehlo.add %v10648, %v10650 : tensor<768xf32>
    %v10652 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10653 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10654 = stablehlo.multiply %v10652, %s3b1pbm : tensor<768xf32>
    %v10655 = stablehlo.multiply %v10653, %v1672 : tensor<768xf32>
    %v10656 = stablehlo.add %v10654, %v10655 : tensor<768xf32>
    %v10657 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10658 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10659 = stablehlo.multiply %v10657, %s3b1pbv : tensor<768xf32>
    %v10660 = stablehlo.multiply %v1672, %v1672 : tensor<768xf32>
    %v10661 = stablehlo.multiply %v10658, %v10660 : tensor<768xf32>
    %v10662 = stablehlo.add %v10659, %v10661 : tensor<768xf32>
    %v10663 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10664 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10665 = stablehlo.divide %v10656, %v10663 : tensor<768xf32>
    %v10666 = stablehlo.divide %v10662, %v10664 : tensor<768xf32>
    %v10667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10668 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10669 = stablehlo.sqrt %v10666 : tensor<768xf32>
    %v10670 = stablehlo.add %v10669, %v10668 : tensor<768xf32>
    %v10671 = stablehlo.divide %v10665, %v10670 : tensor<768xf32>
    %v10672 = stablehlo.multiply %v10667, %v10671 : tensor<768xf32>
    %v10673 = stablehlo.subtract %s3b1pb, %v10672 : tensor<768xf32>
    %v10674 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10675 = stablehlo.multiply %v10674, %v10667 : tensor<768xf32>
    %v10676 = stablehlo.multiply %v10675, %s3b1pb : tensor<768xf32>
    %v10677 = stablehlo.subtract %v10673, %v10676 : tensor<768xf32>
    %v10678 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10679 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10680 = stablehlo.multiply %v10678, %s3b1lgm : tensor<768xf32>
    %v10681 = stablehlo.multiply %v10679, %v1663 : tensor<768xf32>
    %v10682 = stablehlo.add %v10680, %v10681 : tensor<768xf32>
    %v10683 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10684 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10685 = stablehlo.multiply %v10683, %s3b1lgv : tensor<768xf32>
    %v10686 = stablehlo.multiply %v1663, %v1663 : tensor<768xf32>
    %v10687 = stablehlo.multiply %v10684, %v10686 : tensor<768xf32>
    %v10688 = stablehlo.add %v10685, %v10687 : tensor<768xf32>
    %v10689 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10690 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10691 = stablehlo.multiply %v10689, %s3b1lgm : tensor<768xf32>
    %v10692 = stablehlo.multiply %v10690, %v1663 : tensor<768xf32>
    %v10693 = stablehlo.add %v10691, %v10692 : tensor<768xf32>
    %v10694 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10695 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10696 = stablehlo.multiply %v10694, %s3b1lgv : tensor<768xf32>
    %v10697 = stablehlo.multiply %v1663, %v1663 : tensor<768xf32>
    %v10698 = stablehlo.multiply %v10695, %v10697 : tensor<768xf32>
    %v10699 = stablehlo.add %v10696, %v10698 : tensor<768xf32>
    %v10700 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10701 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10702 = stablehlo.divide %v10693, %v10700 : tensor<768xf32>
    %v10703 = stablehlo.divide %v10699, %v10701 : tensor<768xf32>
    %v10704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10705 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10706 = stablehlo.sqrt %v10703 : tensor<768xf32>
    %v10707 = stablehlo.add %v10706, %v10705 : tensor<768xf32>
    %v10708 = stablehlo.divide %v10702, %v10707 : tensor<768xf32>
    %v10709 = stablehlo.multiply %v10704, %v10708 : tensor<768xf32>
    %v10710 = stablehlo.subtract %s3b1lg, %v10709 : tensor<768xf32>
    %v10711 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10712 = stablehlo.multiply %v10711, %v10704 : tensor<768xf32>
    %v10713 = stablehlo.multiply %v10712, %s3b1lg : tensor<768xf32>
    %v10714 = stablehlo.subtract %v10710, %v10713 : tensor<768xf32>
    %v10715 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10716 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10717 = stablehlo.multiply %v10715, %s3b2dWm : tensor<768x1x7x7xf32>
    %v10718 = stablehlo.multiply %v10716, %v1568 : tensor<768x1x7x7xf32>
    %v10719 = stablehlo.add %v10717, %v10718 : tensor<768x1x7x7xf32>
    %v10720 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10721 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10722 = stablehlo.multiply %v10720, %s3b2dWv : tensor<768x1x7x7xf32>
    %v10723 = stablehlo.multiply %v1568, %v1568 : tensor<768x1x7x7xf32>
    %v10724 = stablehlo.multiply %v10721, %v10723 : tensor<768x1x7x7xf32>
    %v10725 = stablehlo.add %v10722, %v10724 : tensor<768x1x7x7xf32>
    %v10726 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10727 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10728 = stablehlo.multiply %v10726, %s3b2dWm : tensor<768x1x7x7xf32>
    %v10729 = stablehlo.multiply %v10727, %v1568 : tensor<768x1x7x7xf32>
    %v10730 = stablehlo.add %v10728, %v10729 : tensor<768x1x7x7xf32>
    %v10731 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10732 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10733 = stablehlo.multiply %v10731, %s3b2dWv : tensor<768x1x7x7xf32>
    %v10734 = stablehlo.multiply %v1568, %v1568 : tensor<768x1x7x7xf32>
    %v10735 = stablehlo.multiply %v10732, %v10734 : tensor<768x1x7x7xf32>
    %v10736 = stablehlo.add %v10733, %v10735 : tensor<768x1x7x7xf32>
    %v10737 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10738 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10739 = stablehlo.divide %v10730, %v10737 : tensor<768x1x7x7xf32>
    %v10740 = stablehlo.divide %v10736, %v10738 : tensor<768x1x7x7xf32>
    %v10741 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10742 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10743 = stablehlo.sqrt %v10740 : tensor<768x1x7x7xf32>
    %v10744 = stablehlo.add %v10743, %v10742 : tensor<768x1x7x7xf32>
    %v10745 = stablehlo.divide %v10739, %v10744 : tensor<768x1x7x7xf32>
    %v10746 = stablehlo.multiply %v10741, %v10745 : tensor<768x1x7x7xf32>
    %v10747 = stablehlo.subtract %s3b2dW, %v10746 : tensor<768x1x7x7xf32>
    %v10748 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x1x7x7xf32>
    %v10749 = stablehlo.multiply %v10748, %v10741 : tensor<768x1x7x7xf32>
    %v10750 = stablehlo.multiply %v10749, %s3b2dW : tensor<768x1x7x7xf32>
    %v10751 = stablehlo.subtract %v10747, %v10750 : tensor<768x1x7x7xf32>
    %v10752 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10753 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10754 = stablehlo.multiply %v10752, %s3b2dbm : tensor<768xf32>
    %v10755 = stablehlo.multiply %v10753, %v1571 : tensor<768xf32>
    %v10756 = stablehlo.add %v10754, %v10755 : tensor<768xf32>
    %v10757 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10758 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10759 = stablehlo.multiply %v10757, %s3b2dbv : tensor<768xf32>
    %v10760 = stablehlo.multiply %v1571, %v1571 : tensor<768xf32>
    %v10761 = stablehlo.multiply %v10758, %v10760 : tensor<768xf32>
    %v10762 = stablehlo.add %v10759, %v10761 : tensor<768xf32>
    %v10763 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10764 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10765 = stablehlo.multiply %v10763, %s3b2dbm : tensor<768xf32>
    %v10766 = stablehlo.multiply %v10764, %v1571 : tensor<768xf32>
    %v10767 = stablehlo.add %v10765, %v10766 : tensor<768xf32>
    %v10768 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10769 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10770 = stablehlo.multiply %v10768, %s3b2dbv : tensor<768xf32>
    %v10771 = stablehlo.multiply %v1571, %v1571 : tensor<768xf32>
    %v10772 = stablehlo.multiply %v10769, %v10771 : tensor<768xf32>
    %v10773 = stablehlo.add %v10770, %v10772 : tensor<768xf32>
    %v10774 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10775 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10776 = stablehlo.divide %v10767, %v10774 : tensor<768xf32>
    %v10777 = stablehlo.divide %v10773, %v10775 : tensor<768xf32>
    %v10778 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10779 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10780 = stablehlo.sqrt %v10777 : tensor<768xf32>
    %v10781 = stablehlo.add %v10780, %v10779 : tensor<768xf32>
    %v10782 = stablehlo.divide %v10776, %v10781 : tensor<768xf32>
    %v10783 = stablehlo.multiply %v10778, %v10782 : tensor<768xf32>
    %v10784 = stablehlo.subtract %s3b2db, %v10783 : tensor<768xf32>
    %v10785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10786 = stablehlo.multiply %v10785, %v10778 : tensor<768xf32>
    %v10787 = stablehlo.multiply %v10786, %s3b2db : tensor<768xf32>
    %v10788 = stablehlo.subtract %v10784, %v10787 : tensor<768xf32>
    %v10789 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10790 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10791 = stablehlo.multiply %v10789, %s3b2ngm : tensor<768xf32>
    %v10792 = stablehlo.multiply %v10790, %v1556 : tensor<768xf32>
    %v10793 = stablehlo.add %v10791, %v10792 : tensor<768xf32>
    %v10794 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10795 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10796 = stablehlo.multiply %v10794, %s3b2ngv : tensor<768xf32>
    %v10797 = stablehlo.multiply %v1556, %v1556 : tensor<768xf32>
    %v10798 = stablehlo.multiply %v10795, %v10797 : tensor<768xf32>
    %v10799 = stablehlo.add %v10796, %v10798 : tensor<768xf32>
    %v10800 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10801 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10802 = stablehlo.multiply %v10800, %s3b2ngm : tensor<768xf32>
    %v10803 = stablehlo.multiply %v10801, %v1556 : tensor<768xf32>
    %v10804 = stablehlo.add %v10802, %v10803 : tensor<768xf32>
    %v10805 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10806 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10807 = stablehlo.multiply %v10805, %s3b2ngv : tensor<768xf32>
    %v10808 = stablehlo.multiply %v1556, %v1556 : tensor<768xf32>
    %v10809 = stablehlo.multiply %v10806, %v10808 : tensor<768xf32>
    %v10810 = stablehlo.add %v10807, %v10809 : tensor<768xf32>
    %v10811 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10812 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10813 = stablehlo.divide %v10804, %v10811 : tensor<768xf32>
    %v10814 = stablehlo.divide %v10810, %v10812 : tensor<768xf32>
    %v10815 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10816 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10817 = stablehlo.sqrt %v10814 : tensor<768xf32>
    %v10818 = stablehlo.add %v10817, %v10816 : tensor<768xf32>
    %v10819 = stablehlo.divide %v10813, %v10818 : tensor<768xf32>
    %v10820 = stablehlo.multiply %v10815, %v10819 : tensor<768xf32>
    %v10821 = stablehlo.subtract %s3b2ng, %v10820 : tensor<768xf32>
    %v10822 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10823 = stablehlo.multiply %v10822, %v10815 : tensor<768xf32>
    %v10824 = stablehlo.multiply %v10823, %s3b2ng : tensor<768xf32>
    %v10825 = stablehlo.subtract %v10821, %v10824 : tensor<768xf32>
    %v10826 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10827 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10828 = stablehlo.multiply %v10826, %s3b2nbtm : tensor<768xf32>
    %v10829 = stablehlo.multiply %v10827, %v1562 : tensor<768xf32>
    %v10830 = stablehlo.add %v10828, %v10829 : tensor<768xf32>
    %v10831 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10832 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10833 = stablehlo.multiply %v10831, %s3b2nbtv : tensor<768xf32>
    %v10834 = stablehlo.multiply %v1562, %v1562 : tensor<768xf32>
    %v10835 = stablehlo.multiply %v10832, %v10834 : tensor<768xf32>
    %v10836 = stablehlo.add %v10833, %v10835 : tensor<768xf32>
    %v10837 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10838 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10839 = stablehlo.multiply %v10837, %s3b2nbtm : tensor<768xf32>
    %v10840 = stablehlo.multiply %v10838, %v1562 : tensor<768xf32>
    %v10841 = stablehlo.add %v10839, %v10840 : tensor<768xf32>
    %v10842 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10843 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10844 = stablehlo.multiply %v10842, %s3b2nbtv : tensor<768xf32>
    %v10845 = stablehlo.multiply %v1562, %v1562 : tensor<768xf32>
    %v10846 = stablehlo.multiply %v10843, %v10845 : tensor<768xf32>
    %v10847 = stablehlo.add %v10844, %v10846 : tensor<768xf32>
    %v10848 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10849 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10850 = stablehlo.divide %v10841, %v10848 : tensor<768xf32>
    %v10851 = stablehlo.divide %v10847, %v10849 : tensor<768xf32>
    %v10852 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10853 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10854 = stablehlo.sqrt %v10851 : tensor<768xf32>
    %v10855 = stablehlo.add %v10854, %v10853 : tensor<768xf32>
    %v10856 = stablehlo.divide %v10850, %v10855 : tensor<768xf32>
    %v10857 = stablehlo.multiply %v10852, %v10856 : tensor<768xf32>
    %v10858 = stablehlo.subtract %s3b2nbt, %v10857 : tensor<768xf32>
    %v10859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10860 = stablehlo.multiply %v10859, %v10852 : tensor<768xf32>
    %v10861 = stablehlo.multiply %v10860, %s3b2nbt : tensor<768xf32>
    %v10862 = stablehlo.subtract %v10858, %v10861 : tensor<768xf32>
    %v10863 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10864 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10865 = stablehlo.multiply %v10863, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v10866 = stablehlo.multiply %v10864, %v1529 : tensor<3072x768x1x1xf32>
    %v10867 = stablehlo.add %v10865, %v10866 : tensor<3072x768x1x1xf32>
    %v10868 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10869 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10870 = stablehlo.multiply %v10868, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v10871 = stablehlo.multiply %v1529, %v1529 : tensor<3072x768x1x1xf32>
    %v10872 = stablehlo.multiply %v10869, %v10871 : tensor<3072x768x1x1xf32>
    %v10873 = stablehlo.add %v10870, %v10872 : tensor<3072x768x1x1xf32>
    %v10874 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10875 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10876 = stablehlo.multiply %v10874, %s3b2eWm : tensor<3072x768x1x1xf32>
    %v10877 = stablehlo.multiply %v10875, %v1529 : tensor<3072x768x1x1xf32>
    %v10878 = stablehlo.add %v10876, %v10877 : tensor<3072x768x1x1xf32>
    %v10879 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10880 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10881 = stablehlo.multiply %v10879, %s3b2eWv : tensor<3072x768x1x1xf32>
    %v10882 = stablehlo.multiply %v1529, %v1529 : tensor<3072x768x1x1xf32>
    %v10883 = stablehlo.multiply %v10880, %v10882 : tensor<3072x768x1x1xf32>
    %v10884 = stablehlo.add %v10881, %v10883 : tensor<3072x768x1x1xf32>
    %v10885 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10886 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10887 = stablehlo.divide %v10878, %v10885 : tensor<3072x768x1x1xf32>
    %v10888 = stablehlo.divide %v10884, %v10886 : tensor<3072x768x1x1xf32>
    %v10889 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10890 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10891 = stablehlo.sqrt %v10888 : tensor<3072x768x1x1xf32>
    %v10892 = stablehlo.add %v10891, %v10890 : tensor<3072x768x1x1xf32>
    %v10893 = stablehlo.divide %v10887, %v10892 : tensor<3072x768x1x1xf32>
    %v10894 = stablehlo.multiply %v10889, %v10893 : tensor<3072x768x1x1xf32>
    %v10895 = stablehlo.subtract %s3b2eW, %v10894 : tensor<3072x768x1x1xf32>
    %v10896 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072x768x1x1xf32>
    %v10897 = stablehlo.multiply %v10896, %v10889 : tensor<3072x768x1x1xf32>
    %v10898 = stablehlo.multiply %v10897, %s3b2eW : tensor<3072x768x1x1xf32>
    %v10899 = stablehlo.subtract %v10895, %v10898 : tensor<3072x768x1x1xf32>
    %v10900 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10901 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10902 = stablehlo.multiply %v10900, %s3b2ebm : tensor<3072xf32>
    %v10903 = stablehlo.multiply %v10901, %v1532 : tensor<3072xf32>
    %v10904 = stablehlo.add %v10902, %v10903 : tensor<3072xf32>
    %v10905 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10906 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10907 = stablehlo.multiply %v10905, %s3b2ebv : tensor<3072xf32>
    %v10908 = stablehlo.multiply %v1532, %v1532 : tensor<3072xf32>
    %v10909 = stablehlo.multiply %v10906, %v10908 : tensor<3072xf32>
    %v10910 = stablehlo.add %v10907, %v10909 : tensor<3072xf32>
    %v10911 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10912 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10913 = stablehlo.multiply %v10911, %s3b2ebm : tensor<3072xf32>
    %v10914 = stablehlo.multiply %v10912, %v1532 : tensor<3072xf32>
    %v10915 = stablehlo.add %v10913, %v10914 : tensor<3072xf32>
    %v10916 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10917 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10918 = stablehlo.multiply %v10916, %s3b2ebv : tensor<3072xf32>
    %v10919 = stablehlo.multiply %v1532, %v1532 : tensor<3072xf32>
    %v10920 = stablehlo.multiply %v10917, %v10919 : tensor<3072xf32>
    %v10921 = stablehlo.add %v10918, %v10920 : tensor<3072xf32>
    %v10922 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10923 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10924 = stablehlo.divide %v10915, %v10922 : tensor<3072xf32>
    %v10925 = stablehlo.divide %v10921, %v10923 : tensor<3072xf32>
    %v10926 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10927 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10928 = stablehlo.sqrt %v10925 : tensor<3072xf32>
    %v10929 = stablehlo.add %v10928, %v10927 : tensor<3072xf32>
    %v10930 = stablehlo.divide %v10924, %v10929 : tensor<3072xf32>
    %v10931 = stablehlo.multiply %v10926, %v10930 : tensor<3072xf32>
    %v10932 = stablehlo.subtract %s3b2eb, %v10931 : tensor<3072xf32>
    %v10933 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<3072xf32>
    %v10934 = stablehlo.multiply %v10933, %v10926 : tensor<3072xf32>
    %v10935 = stablehlo.multiply %v10934, %s3b2eb : tensor<3072xf32>
    %v10936 = stablehlo.subtract %v10932, %v10935 : tensor<3072xf32>
    %v10937 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10938 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10939 = stablehlo.multiply %v10937, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v10940 = stablehlo.multiply %v10938, %v1520 : tensor<768x3072x1x1xf32>
    %v10941 = stablehlo.add %v10939, %v10940 : tensor<768x3072x1x1xf32>
    %v10942 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10943 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10944 = stablehlo.multiply %v10942, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v10945 = stablehlo.multiply %v1520, %v1520 : tensor<768x3072x1x1xf32>
    %v10946 = stablehlo.multiply %v10943, %v10945 : tensor<768x3072x1x1xf32>
    %v10947 = stablehlo.add %v10944, %v10946 : tensor<768x3072x1x1xf32>
    %v10948 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10949 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10950 = stablehlo.multiply %v10948, %s3b2pWm : tensor<768x3072x1x1xf32>
    %v10951 = stablehlo.multiply %v10949, %v1520 : tensor<768x3072x1x1xf32>
    %v10952 = stablehlo.add %v10950, %v10951 : tensor<768x3072x1x1xf32>
    %v10953 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10954 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10955 = stablehlo.multiply %v10953, %s3b2pWv : tensor<768x3072x1x1xf32>
    %v10956 = stablehlo.multiply %v1520, %v1520 : tensor<768x3072x1x1xf32>
    %v10957 = stablehlo.multiply %v10954, %v10956 : tensor<768x3072x1x1xf32>
    %v10958 = stablehlo.add %v10955, %v10957 : tensor<768x3072x1x1xf32>
    %v10959 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10960 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10961 = stablehlo.divide %v10952, %v10959 : tensor<768x3072x1x1xf32>
    %v10962 = stablehlo.divide %v10958, %v10960 : tensor<768x3072x1x1xf32>
    %v10963 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10964 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10965 = stablehlo.sqrt %v10962 : tensor<768x3072x1x1xf32>
    %v10966 = stablehlo.add %v10965, %v10964 : tensor<768x3072x1x1xf32>
    %v10967 = stablehlo.divide %v10961, %v10966 : tensor<768x3072x1x1xf32>
    %v10968 = stablehlo.multiply %v10963, %v10967 : tensor<768x3072x1x1xf32>
    %v10969 = stablehlo.subtract %s3b2pW, %v10968 : tensor<768x3072x1x1xf32>
    %v10970 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x3072x1x1xf32>
    %v10971 = stablehlo.multiply %v10970, %v10963 : tensor<768x3072x1x1xf32>
    %v10972 = stablehlo.multiply %v10971, %s3b2pW : tensor<768x3072x1x1xf32>
    %v10973 = stablehlo.subtract %v10969, %v10972 : tensor<768x3072x1x1xf32>
    %v10974 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10975 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10976 = stablehlo.multiply %v10974, %s3b2pbm : tensor<768xf32>
    %v10977 = stablehlo.multiply %v10975, %v1523 : tensor<768xf32>
    %v10978 = stablehlo.add %v10976, %v10977 : tensor<768xf32>
    %v10979 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10980 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10981 = stablehlo.multiply %v10979, %s3b2pbv : tensor<768xf32>
    %v10982 = stablehlo.multiply %v1523, %v1523 : tensor<768xf32>
    %v10983 = stablehlo.multiply %v10980, %v10982 : tensor<768xf32>
    %v10984 = stablehlo.add %v10981, %v10983 : tensor<768xf32>
    %v10985 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10986 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10987 = stablehlo.multiply %v10985, %s3b2pbm : tensor<768xf32>
    %v10988 = stablehlo.multiply %v10986, %v1523 : tensor<768xf32>
    %v10989 = stablehlo.add %v10987, %v10988 : tensor<768xf32>
    %v10990 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10991 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10992 = stablehlo.multiply %v10990, %s3b2pbv : tensor<768xf32>
    %v10993 = stablehlo.multiply %v1523, %v1523 : tensor<768xf32>
    %v10994 = stablehlo.multiply %v10991, %v10993 : tensor<768xf32>
    %v10995 = stablehlo.add %v10992, %v10994 : tensor<768xf32>
    %v10996 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10997 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v10998 = stablehlo.divide %v10989, %v10996 : tensor<768xf32>
    %v10999 = stablehlo.divide %v10995, %v10997 : tensor<768xf32>
    %v11000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11001 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11002 = stablehlo.sqrt %v10999 : tensor<768xf32>
    %v11003 = stablehlo.add %v11002, %v11001 : tensor<768xf32>
    %v11004 = stablehlo.divide %v10998, %v11003 : tensor<768xf32>
    %v11005 = stablehlo.multiply %v11000, %v11004 : tensor<768xf32>
    %v11006 = stablehlo.subtract %s3b2pb, %v11005 : tensor<768xf32>
    %v11007 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11008 = stablehlo.multiply %v11007, %v11000 : tensor<768xf32>
    %v11009 = stablehlo.multiply %v11008, %s3b2pb : tensor<768xf32>
    %v11010 = stablehlo.subtract %v11006, %v11009 : tensor<768xf32>
    %v11011 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11012 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11013 = stablehlo.multiply %v11011, %s3b2lgm : tensor<768xf32>
    %v11014 = stablehlo.multiply %v11012, %v1514 : tensor<768xf32>
    %v11015 = stablehlo.add %v11013, %v11014 : tensor<768xf32>
    %v11016 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11017 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11018 = stablehlo.multiply %v11016, %s3b2lgv : tensor<768xf32>
    %v11019 = stablehlo.multiply %v1514, %v1514 : tensor<768xf32>
    %v11020 = stablehlo.multiply %v11017, %v11019 : tensor<768xf32>
    %v11021 = stablehlo.add %v11018, %v11020 : tensor<768xf32>
    %v11022 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11023 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11024 = stablehlo.multiply %v11022, %s3b2lgm : tensor<768xf32>
    %v11025 = stablehlo.multiply %v11023, %v1514 : tensor<768xf32>
    %v11026 = stablehlo.add %v11024, %v11025 : tensor<768xf32>
    %v11027 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11028 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11029 = stablehlo.multiply %v11027, %s3b2lgv : tensor<768xf32>
    %v11030 = stablehlo.multiply %v1514, %v1514 : tensor<768xf32>
    %v11031 = stablehlo.multiply %v11028, %v11030 : tensor<768xf32>
    %v11032 = stablehlo.add %v11029, %v11031 : tensor<768xf32>
    %v11033 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11034 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11035 = stablehlo.divide %v11026, %v11033 : tensor<768xf32>
    %v11036 = stablehlo.divide %v11032, %v11034 : tensor<768xf32>
    %v11037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11038 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11039 = stablehlo.sqrt %v11036 : tensor<768xf32>
    %v11040 = stablehlo.add %v11039, %v11038 : tensor<768xf32>
    %v11041 = stablehlo.divide %v11035, %v11040 : tensor<768xf32>
    %v11042 = stablehlo.multiply %v11037, %v11041 : tensor<768xf32>
    %v11043 = stablehlo.subtract %s3b2lg, %v11042 : tensor<768xf32>
    %v11044 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768xf32>
    %v11045 = stablehlo.multiply %v11044, %v11037 : tensor<768xf32>
    %v11046 = stablehlo.multiply %v11045, %s3b2lg : tensor<768xf32>
    %v11047 = stablehlo.subtract %v11043, %v11046 : tensor<768xf32>
    %v11048 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11049 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11050 = stablehlo.multiply %v11048, %Wdm : tensor<768x10xf32>
    %v11051 = stablehlo.multiply %v11049, %v1420 : tensor<768x10xf32>
    %v11052 = stablehlo.add %v11050, %v11051 : tensor<768x10xf32>
    %v11053 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11054 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11055 = stablehlo.multiply %v11053, %Wdv : tensor<768x10xf32>
    %v11056 = stablehlo.multiply %v1420, %v1420 : tensor<768x10xf32>
    %v11057 = stablehlo.multiply %v11054, %v11056 : tensor<768x10xf32>
    %v11058 = stablehlo.add %v11055, %v11057 : tensor<768x10xf32>
    %v11059 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11060 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11061 = stablehlo.multiply %v11059, %Wdm : tensor<768x10xf32>
    %v11062 = stablehlo.multiply %v11060, %v1420 : tensor<768x10xf32>
    %v11063 = stablehlo.add %v11061, %v11062 : tensor<768x10xf32>
    %v11064 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11065 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11066 = stablehlo.multiply %v11064, %Wdv : tensor<768x10xf32>
    %v11067 = stablehlo.multiply %v1420, %v1420 : tensor<768x10xf32>
    %v11068 = stablehlo.multiply %v11065, %v11067 : tensor<768x10xf32>
    %v11069 = stablehlo.add %v11066, %v11068 : tensor<768x10xf32>
    %v11070 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11071 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11072 = stablehlo.divide %v11063, %v11070 : tensor<768x10xf32>
    %v11073 = stablehlo.divide %v11069, %v11071 : tensor<768x10xf32>
    %v11074 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11075 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11076 = stablehlo.sqrt %v11073 : tensor<768x10xf32>
    %v11077 = stablehlo.add %v11076, %v11075 : tensor<768x10xf32>
    %v11078 = stablehlo.divide %v11072, %v11077 : tensor<768x10xf32>
    %v11079 = stablehlo.multiply %v11074, %v11078 : tensor<768x10xf32>
    %v11080 = stablehlo.subtract %Wd, %v11079 : tensor<768x10xf32>
    %v11081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<768x10xf32>
    %v11082 = stablehlo.multiply %v11081, %v11074 : tensor<768x10xf32>
    %v11083 = stablehlo.multiply %v11082, %Wd : tensor<768x10xf32>
    %v11084 = stablehlo.subtract %v11080, %v11083 : tensor<768x10xf32>
    %v11085 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11086 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11087 = stablehlo.multiply %v11085, %bdm : tensor<10xf32>
    %v11088 = stablehlo.multiply %v11086, %v1422 : tensor<10xf32>
    %v11089 = stablehlo.add %v11087, %v11088 : tensor<10xf32>
    %v11090 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11091 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11092 = stablehlo.multiply %v11090, %bdv : tensor<10xf32>
    %v11093 = stablehlo.multiply %v1422, %v1422 : tensor<10xf32>
    %v11094 = stablehlo.multiply %v11091, %v11093 : tensor<10xf32>
    %v11095 = stablehlo.add %v11092, %v11094 : tensor<10xf32>
    %v11096 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11097 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11098 = stablehlo.multiply %v11096, %bdm : tensor<10xf32>
    %v11099 = stablehlo.multiply %v11097, %v1422 : tensor<10xf32>
    %v11100 = stablehlo.add %v11098, %v11099 : tensor<10xf32>
    %v11101 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11102 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11103 = stablehlo.multiply %v11101, %bdv : tensor<10xf32>
    %v11104 = stablehlo.multiply %v1422, %v1422 : tensor<10xf32>
    %v11105 = stablehlo.multiply %v11102, %v11104 : tensor<10xf32>
    %v11106 = stablehlo.add %v11103, %v11105 : tensor<10xf32>
    %v11107 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11108 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11109 = stablehlo.divide %v11100, %v11107 : tensor<10xf32>
    %v11110 = stablehlo.divide %v11106, %v11108 : tensor<10xf32>
    %v11111 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11112 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11113 = stablehlo.sqrt %v11110 : tensor<10xf32>
    %v11114 = stablehlo.add %v11113, %v11112 : tensor<10xf32>
    %v11115 = stablehlo.divide %v11109, %v11114 : tensor<10xf32>
    %v11116 = stablehlo.multiply %v11111, %v11115 : tensor<10xf32>
    %v11117 = stablehlo.subtract %bd, %v11116 : tensor<10xf32>
    %v11118 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<10xf32>
    %v11119 = stablehlo.multiply %v11118, %v11111 : tensor<10xf32>
    %v11120 = stablehlo.multiply %v11119, %bd : tensor<10xf32>
    %v11121 = stablehlo.subtract %v11117, %v11120 : tensor<10xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1410 : tensor<32x10xf32>
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
    return %v4498, %v4535, %v4572, %v4609, %v4646, %v4683, %v4720, %v4757, %v4794, %v4831, %v4868, %v4905, %v4942, %v4979, %v5016, %v5053, %v5090, %v5127, %v5164, %v5201, %v5238, %v5275, %v5312, %v5349, %v5386, %v5423, %v5460, %v5497, %v5534, %v5571, %v5608, %v5645, %v5682, %v5719, %v5756, %v5793, %v5830, %v5867, %v5904, %v5941, %v5978, %v6015, %v6052, %v6089, %v6126, %v6163, %v6200, %v6237, %v6274, %v6311, %v6348, %v6385, %v6422, %v6459, %v6496, %v6533, %v6570, %v6607, %v6644, %v6681, %v6718, %v6755, %v6792, %v6829, %v6866, %v6903, %v6940, %v6977, %v7014, %v7051, %v7088, %v7125, %v7162, %v7199, %v7236, %v7273, %v7310, %v7347, %v7384, %v7421, %v7458, %v7495, %v7532, %v7569, %v7606, %v7643, %v7680, %v7717, %v7754, %v7791, %v7828, %v7865, %v7902, %v7939, %v7976, %v8013, %v8050, %v8087, %v8124, %v8161, %v8198, %v8235, %v8272, %v8309, %v8346, %v8383, %v8420, %v8457, %v8494, %v8531, %v8568, %v8605, %v8642, %v8679, %v8716, %v8753, %v8790, %v8827, %v8864, %v8901, %v8938, %v8975, %v9012, %v9049, %v9086, %v9123, %v9160, %v9197, %v9234, %v9271, %v9308, %v9345, %v9382, %v9419, %v9456, %v9493, %v9530, %v9567, %v9604, %v9641, %v9678, %v9715, %v9752, %v9789, %v9826, %v9863, %v9900, %v9937, %v9974, %v10011, %v10048, %v10085, %v10122, %v10159, %v10196, %v10233, %v10270, %v10307, %v10344, %v10381, %v10418, %v10455, %v10492, %v10529, %v10566, %v10603, %v10640, %v10677, %v10714, %v10751, %v10788, %v10825, %v10862, %v10899, %v10936, %v10973, %v11010, %v11047, %v11084, %v11121, %v4466, %v4503, %v4540, %v4577, %v4614, %v4651, %v4688, %v4725, %v4762, %v4799, %v4836, %v4873, %v4910, %v4947, %v4984, %v5021, %v5058, %v5095, %v5132, %v5169, %v5206, %v5243, %v5280, %v5317, %v5354, %v5391, %v5428, %v5465, %v5502, %v5539, %v5576, %v5613, %v5650, %v5687, %v5724, %v5761, %v5798, %v5835, %v5872, %v5909, %v5946, %v5983, %v6020, %v6057, %v6094, %v6131, %v6168, %v6205, %v6242, %v6279, %v6316, %v6353, %v6390, %v6427, %v6464, %v6501, %v6538, %v6575, %v6612, %v6649, %v6686, %v6723, %v6760, %v6797, %v6834, %v6871, %v6908, %v6945, %v6982, %v7019, %v7056, %v7093, %v7130, %v7167, %v7204, %v7241, %v7278, %v7315, %v7352, %v7389, %v7426, %v7463, %v7500, %v7537, %v7574, %v7611, %v7648, %v7685, %v7722, %v7759, %v7796, %v7833, %v7870, %v7907, %v7944, %v7981, %v8018, %v8055, %v8092, %v8129, %v8166, %v8203, %v8240, %v8277, %v8314, %v8351, %v8388, %v8425, %v8462, %v8499, %v8536, %v8573, %v8610, %v8647, %v8684, %v8721, %v8758, %v8795, %v8832, %v8869, %v8906, %v8943, %v8980, %v9017, %v9054, %v9091, %v9128, %v9165, %v9202, %v9239, %v9276, %v9313, %v9350, %v9387, %v9424, %v9461, %v9498, %v9535, %v9572, %v9609, %v9646, %v9683, %v9720, %v9757, %v9794, %v9831, %v9868, %v9905, %v9942, %v9979, %v10016, %v10053, %v10090, %v10127, %v10164, %v10201, %v10238, %v10275, %v10312, %v10349, %v10386, %v10423, %v10460, %v10497, %v10534, %v10571, %v10608, %v10645, %v10682, %v10719, %v10756, %v10793, %v10830, %v10867, %v10904, %v10941, %v10978, %v11015, %v11052, %v11089, %v4472, %v4509, %v4546, %v4583, %v4620, %v4657, %v4694, %v4731, %v4768, %v4805, %v4842, %v4879, %v4916, %v4953, %v4990, %v5027, %v5064, %v5101, %v5138, %v5175, %v5212, %v5249, %v5286, %v5323, %v5360, %v5397, %v5434, %v5471, %v5508, %v5545, %v5582, %v5619, %v5656, %v5693, %v5730, %v5767, %v5804, %v5841, %v5878, %v5915, %v5952, %v5989, %v6026, %v6063, %v6100, %v6137, %v6174, %v6211, %v6248, %v6285, %v6322, %v6359, %v6396, %v6433, %v6470, %v6507, %v6544, %v6581, %v6618, %v6655, %v6692, %v6729, %v6766, %v6803, %v6840, %v6877, %v6914, %v6951, %v6988, %v7025, %v7062, %v7099, %v7136, %v7173, %v7210, %v7247, %v7284, %v7321, %v7358, %v7395, %v7432, %v7469, %v7506, %v7543, %v7580, %v7617, %v7654, %v7691, %v7728, %v7765, %v7802, %v7839, %v7876, %v7913, %v7950, %v7987, %v8024, %v8061, %v8098, %v8135, %v8172, %v8209, %v8246, %v8283, %v8320, %v8357, %v8394, %v8431, %v8468, %v8505, %v8542, %v8579, %v8616, %v8653, %v8690, %v8727, %v8764, %v8801, %v8838, %v8875, %v8912, %v8949, %v8986, %v9023, %v9060, %v9097, %v9134, %v9171, %v9208, %v9245, %v9282, %v9319, %v9356, %v9393, %v9430, %v9467, %v9504, %v9541, %v9578, %v9615, %v9652, %v9689, %v9726, %v9763, %v9800, %v9837, %v9874, %v9911, %v9948, %v9985, %v10022, %v10059, %v10096, %v10133, %v10170, %v10207, %v10244, %v10281, %v10318, %v10355, %v10392, %v10429, %v10466, %v10503, %v10540, %v10577, %v10614, %v10651, %v10688, %v10725, %v10762, %v10799, %v10836, %v10873, %v10910, %v10947, %v10984, %v11021, %v11058, %v11095, %loss, %bc1, %bc2, %dp0, %dp1, %dp2, %dp3, %dp4, %dp5, %dp6, %dp7, %dp8, %dp9, %dp10, %dp11, %dp12, %dp13, %dp14, %dp15, %dp16, %dp17 : tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<96x3x4x4xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96x1x7x7xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<384x96x1x1xf32>, tensor<384xf32>, tensor<96x384x1x1xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<96xf32>, tensor<192x96x2x2xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192x1x7x7xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<768x192x1x1xf32>, tensor<768xf32>, tensor<192x768x1x1xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<192xf32>, tensor<384x192x2x2xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384x1x7x7xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<1536x384x1x1xf32>, tensor<1536xf32>, tensor<384x1536x1x1xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<384xf32>, tensor<768x384x2x2xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x1x7x7xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768xf32>, tensor<3072x768x1x1xf32>, tensor<3072xf32>, tensor<768x3072x1x1xf32>, tensor<768xf32>, tensor<768xf32>, tensor<768x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>, tensor<32xf32>
  }
}
