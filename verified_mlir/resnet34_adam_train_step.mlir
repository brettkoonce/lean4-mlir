module @m {
  func.func @resnet34_adam_train_step(%x: tensor<32x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sb: tensor<64xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0b1: tensor<64xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0b2: tensor<64xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1b1: tensor<64xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1b2: tensor<64xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2b1: tensor<64xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2b2: tensor<64xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2b1: tensor<128xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2b2: tensor<128xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x3x3xf32>, %d2bp: tensor<128xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0b1: tensor<128xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0b2: tensor<128xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1b1: tensor<128xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1b2: tensor<128xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2b1: tensor<128xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2b2: tensor<128xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3b1: tensor<256xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3b2: tensor<256xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x3x3xf32>, %d3bp: tensor<256xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0b1: tensor<256xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0b2: tensor<256xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1b1: tensor<256xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1b2: tensor<256xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2b1: tensor<256xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2b2: tensor<256xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3b1: tensor<256xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3b2: tensor<256xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4b1: tensor<256xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4b2: tensor<256xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4b1: tensor<512xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4b2: tensor<512xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x3x3xf32>, %d4bp: tensor<512xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0b1: tensor<512xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0b2: tensor<512xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1b1: tensor<512xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1b2: tensor<512xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x10xf32>, %bd: tensor<10xf32>, %sWm: tensor<64x3x7x7xf32>, %sbm: tensor<64xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0b1m: tensor<64xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0b2m: tensor<64xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1b1m: tensor<64xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1b2m: tensor<64xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2b1m: tensor<64xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2b2m: tensor<64xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2b1m: tensor<128xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2b2m: tensor<128xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x3x3xf32>, %d2bpm: tensor<128xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0b1m: tensor<128xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0b2m: tensor<128xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1b1m: tensor<128xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1b2m: tensor<128xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2b1m: tensor<128xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2b2m: tensor<128xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3b1m: tensor<256xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3b2m: tensor<256xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x3x3xf32>, %d3bpm: tensor<256xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0b1m: tensor<256xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0b2m: tensor<256xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1b1m: tensor<256xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1b2m: tensor<256xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2b1m: tensor<256xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2b2m: tensor<256xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3b1m: tensor<256xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3b2m: tensor<256xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4b1m: tensor<256xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4b2m: tensor<256xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4b1m: tensor<512xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4b2m: tensor<512xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x3x3xf32>, %d4bpm: tensor<512xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0b1m: tensor<512xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0b2m: tensor<512xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1b1m: tensor<512xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1b2m: tensor<512xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x10xf32>, %bdm: tensor<10xf32>, %sWv: tensor<64x3x7x7xf32>, %sbv: tensor<64xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0b1v: tensor<64xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0b2v: tensor<64xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1b1v: tensor<64xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1b2v: tensor<64xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2b1v: tensor<64xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2b2v: tensor<64xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2b1v: tensor<128xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2b2v: tensor<128xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x3x3xf32>, %d2bpv: tensor<128xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0b1v: tensor<128xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0b2v: tensor<128xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1b1v: tensor<128xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1b2v: tensor<128xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2b1v: tensor<128xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2b2v: tensor<128xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3b1v: tensor<256xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3b2v: tensor<256xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x3x3xf32>, %d3bpv: tensor<256xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0b1v: tensor<256xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0b2v: tensor<256xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1b1v: tensor<256xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1b2v: tensor<256xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2b1v: tensor<256xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2b2v: tensor<256xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3b1v: tensor<256xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3b2v: tensor<256xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4b1v: tensor<256xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4b2v: tensor<256xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4b1v: tensor<512xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4b2v: tensor<512xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x3x3xf32>, %d4bpv: tensor<512xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0b1v: tensor<512xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0b2v: tensor<512xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1b1v: tensor<512xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1b2v: tensor<512xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x10xf32>, %bdv: tensor<10xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %onehot: tensor<32x10xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>) {
    %sc = stablehlo.constant dense<0.0> : tensor<f32>
    %xr = stablehlo.reshape %x : (tensor<32x150528xf32>) -> tensor<32x3x224x224xf32>
    %stcc = stablehlo.convolution(%xr, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<32x64x112x112xf32>
    %stcbb = stablehlo.broadcast_in_dim %sb, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stc = stablehlo.add %stcc, %stcbb : tensor<32x64x112x112xf32>
    %stnnf = stablehlo.constant dense<12544.0> : tensor<32x64x112x112xf32>
    %stnep = stablehlo.constant dense<1.0e-5> : tensor<32x64x112x112xf32>
    %stnsmr = stablehlo.reduce(%stc init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %stnsm = stablehlo.broadcast_in_dim %stnsmr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %stnmu = stablehlo.divide %stnsm, %stnnf : tensor<32x64x112x112xf32>
    %stnxc = stablehlo.subtract %stc, %stnmu : tensor<32x64x112x112xf32>
    %stnsq = stablehlo.multiply %stnxc, %stnxc : tensor<32x64x112x112xf32>
    %stnvsr = stablehlo.reduce(%stnsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %stnvs = stablehlo.broadcast_in_dim %stnvsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %stnvr = stablehlo.divide %stnvs, %stnnf : tensor<32x64x112x112xf32>
    %stnve = stablehlo.add %stnvr, %stnep : tensor<32x64x112x112xf32>
    %stnistd = stablehlo.rsqrt %stnve : tensor<32x64x112x112xf32>
    %stnxh = stablehlo.multiply %stnxc, %stnistd : tensor<32x64x112x112xf32>
    %stngb = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stnbtb = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<32x64x112x112xf32>
    %stngx = stablehlo.multiply %stnxh, %stngb : tensor<32x64x112x112xf32>
    %stn = stablehlo.add %stngx, %stnbtb : tensor<32x64x112x112xf32>
    %strz = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %str = stablehlo.maximum %stn, %strz : tensor<32x64x112x112xf32>
    %stpni = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %stp = "stablehlo.reduce_window"(%str, %stpni) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x56x56xf32>
    %s1b0c1c = stablehlo.convolution(%stp, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c1bb = stablehlo.broadcast_in_dim %s1b0b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c1 = stablehlo.add %s1b0c1c, %s1b0c1bb : tensor<32x64x56x56xf32>
    %s1b0n1nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n1smr = stablehlo.reduce(%s1b0c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0n1sm = stablehlo.broadcast_in_dim %s1b0n1smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1mu = stablehlo.divide %s1b0n1sm, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0n1xc = stablehlo.subtract %s1b0c1, %s1b0n1mu : tensor<32x64x56x56xf32>
    %s1b0n1sq = stablehlo.multiply %s1b0n1xc, %s1b0n1xc : tensor<32x64x56x56xf32>
    %s1b0n1vsr = stablehlo.reduce(%s1b0n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0n1vs = stablehlo.broadcast_in_dim %s1b0n1vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1vr = stablehlo.divide %s1b0n1vs, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0n1ve = stablehlo.add %s1b0n1vr, %s1b0n1ep : tensor<32x64x56x56xf32>
    %s1b0n1istd = stablehlo.rsqrt %s1b0n1ve : tensor<32x64x56x56xf32>
    %s1b0n1xh = stablehlo.multiply %s1b0n1xc, %s1b0n1istd : tensor<32x64x56x56xf32>
    %s1b0n1gb = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1btb = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n1gx = stablehlo.multiply %s1b0n1xh, %s1b0n1gb : tensor<32x64x56x56xf32>
    %s1b0n1 = stablehlo.add %s1b0n1gx, %s1b0n1btb : tensor<32x64x56x56xf32>
    %s1b0r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0r1 = stablehlo.maximum %s1b0n1, %s1b0r1z : tensor<32x64x56x56xf32>
    %s1b0c2c = stablehlo.convolution(%s1b0r1, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c2bb = stablehlo.broadcast_in_dim %s1b0b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0c2 = stablehlo.add %s1b0c2c, %s1b0c2bb : tensor<32x64x56x56xf32>
    %s1b0n2nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b0n2smr = stablehlo.reduce(%s1b0c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0n2sm = stablehlo.broadcast_in_dim %s1b0n2smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2mu = stablehlo.divide %s1b0n2sm, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0n2xc = stablehlo.subtract %s1b0c2, %s1b0n2mu : tensor<32x64x56x56xf32>
    %s1b0n2sq = stablehlo.multiply %s1b0n2xc, %s1b0n2xc : tensor<32x64x56x56xf32>
    %s1b0n2vsr = stablehlo.reduce(%s1b0n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0n2vs = stablehlo.broadcast_in_dim %s1b0n2vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2vr = stablehlo.divide %s1b0n2vs, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0n2ve = stablehlo.add %s1b0n2vr, %s1b0n2ep : tensor<32x64x56x56xf32>
    %s1b0n2istd = stablehlo.rsqrt %s1b0n2ve : tensor<32x64x56x56xf32>
    %s1b0n2xh = stablehlo.multiply %s1b0n2xc, %s1b0n2istd : tensor<32x64x56x56xf32>
    %s1b0n2gb = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2btb = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0n2gx = stablehlo.multiply %s1b0n2xh, %s1b0n2gb : tensor<32x64x56x56xf32>
    %s1b0n2 = stablehlo.add %s1b0n2gx, %s1b0n2btb : tensor<32x64x56x56xf32>
    %s1b0a = stablehlo.add %s1b0n2, %stp : tensor<32x64x56x56xf32>
    %s1b0oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0o = stablehlo.maximum %s1b0a, %s1b0oz : tensor<32x64x56x56xf32>
    %s1b1c1c = stablehlo.convolution(%s1b0o, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c1bb = stablehlo.broadcast_in_dim %s1b1b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c1 = stablehlo.add %s1b1c1c, %s1b1c1bb : tensor<32x64x56x56xf32>
    %s1b1n1nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n1smr = stablehlo.reduce(%s1b1c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1n1sm = stablehlo.broadcast_in_dim %s1b1n1smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1mu = stablehlo.divide %s1b1n1sm, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1n1xc = stablehlo.subtract %s1b1c1, %s1b1n1mu : tensor<32x64x56x56xf32>
    %s1b1n1sq = stablehlo.multiply %s1b1n1xc, %s1b1n1xc : tensor<32x64x56x56xf32>
    %s1b1n1vsr = stablehlo.reduce(%s1b1n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1n1vs = stablehlo.broadcast_in_dim %s1b1n1vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1vr = stablehlo.divide %s1b1n1vs, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1n1ve = stablehlo.add %s1b1n1vr, %s1b1n1ep : tensor<32x64x56x56xf32>
    %s1b1n1istd = stablehlo.rsqrt %s1b1n1ve : tensor<32x64x56x56xf32>
    %s1b1n1xh = stablehlo.multiply %s1b1n1xc, %s1b1n1istd : tensor<32x64x56x56xf32>
    %s1b1n1gb = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1btb = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n1gx = stablehlo.multiply %s1b1n1xh, %s1b1n1gb : tensor<32x64x56x56xf32>
    %s1b1n1 = stablehlo.add %s1b1n1gx, %s1b1n1btb : tensor<32x64x56x56xf32>
    %s1b1r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1r1 = stablehlo.maximum %s1b1n1, %s1b1r1z : tensor<32x64x56x56xf32>
    %s1b1c2c = stablehlo.convolution(%s1b1r1, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c2bb = stablehlo.broadcast_in_dim %s1b1b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1c2 = stablehlo.add %s1b1c2c, %s1b1c2bb : tensor<32x64x56x56xf32>
    %s1b1n2nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b1n2smr = stablehlo.reduce(%s1b1c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1n2sm = stablehlo.broadcast_in_dim %s1b1n2smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2mu = stablehlo.divide %s1b1n2sm, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1n2xc = stablehlo.subtract %s1b1c2, %s1b1n2mu : tensor<32x64x56x56xf32>
    %s1b1n2sq = stablehlo.multiply %s1b1n2xc, %s1b1n2xc : tensor<32x64x56x56xf32>
    %s1b1n2vsr = stablehlo.reduce(%s1b1n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1n2vs = stablehlo.broadcast_in_dim %s1b1n2vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2vr = stablehlo.divide %s1b1n2vs, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1n2ve = stablehlo.add %s1b1n2vr, %s1b1n2ep : tensor<32x64x56x56xf32>
    %s1b1n2istd = stablehlo.rsqrt %s1b1n2ve : tensor<32x64x56x56xf32>
    %s1b1n2xh = stablehlo.multiply %s1b1n2xc, %s1b1n2istd : tensor<32x64x56x56xf32>
    %s1b1n2gb = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2btb = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1n2gx = stablehlo.multiply %s1b1n2xh, %s1b1n2gb : tensor<32x64x56x56xf32>
    %s1b1n2 = stablehlo.add %s1b1n2gx, %s1b1n2btb : tensor<32x64x56x56xf32>
    %s1b1a = stablehlo.add %s1b1n2, %s1b0o : tensor<32x64x56x56xf32>
    %s1b1oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1o = stablehlo.maximum %s1b1a, %s1b1oz : tensor<32x64x56x56xf32>
    %s1b2c1c = stablehlo.convolution(%s1b1o, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c1bb = stablehlo.broadcast_in_dim %s1b2b1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c1 = stablehlo.add %s1b2c1c, %s1b2c1bb : tensor<32x64x56x56xf32>
    %s1b2n1nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n1smr = stablehlo.reduce(%s1b2c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2n1sm = stablehlo.broadcast_in_dim %s1b2n1smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1mu = stablehlo.divide %s1b2n1sm, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2n1xc = stablehlo.subtract %s1b2c1, %s1b2n1mu : tensor<32x64x56x56xf32>
    %s1b2n1sq = stablehlo.multiply %s1b2n1xc, %s1b2n1xc : tensor<32x64x56x56xf32>
    %s1b2n1vsr = stablehlo.reduce(%s1b2n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2n1vs = stablehlo.broadcast_in_dim %s1b2n1vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1vr = stablehlo.divide %s1b2n1vs, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2n1ve = stablehlo.add %s1b2n1vr, %s1b2n1ep : tensor<32x64x56x56xf32>
    %s1b2n1istd = stablehlo.rsqrt %s1b2n1ve : tensor<32x64x56x56xf32>
    %s1b2n1xh = stablehlo.multiply %s1b2n1xc, %s1b2n1istd : tensor<32x64x56x56xf32>
    %s1b2n1gb = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1btb = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n1gx = stablehlo.multiply %s1b2n1xh, %s1b2n1gb : tensor<32x64x56x56xf32>
    %s1b2n1 = stablehlo.add %s1b2n1gx, %s1b2n1btb : tensor<32x64x56x56xf32>
    %s1b2r1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2r1 = stablehlo.maximum %s1b2n1, %s1b2r1z : tensor<32x64x56x56xf32>
    %s1b2c2c = stablehlo.convolution(%s1b2r1, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c2bb = stablehlo.broadcast_in_dim %s1b2b2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2c2 = stablehlo.add %s1b2c2c, %s1b2c2bb : tensor<32x64x56x56xf32>
    %s1b2n2nf = stablehlo.constant dense<3136.0> : tensor<32x64x56x56xf32>
    %s1b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x64x56x56xf32>
    %s1b2n2smr = stablehlo.reduce(%s1b2c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2n2sm = stablehlo.broadcast_in_dim %s1b2n2smr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2mu = stablehlo.divide %s1b2n2sm, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2n2xc = stablehlo.subtract %s1b2c2, %s1b2n2mu : tensor<32x64x56x56xf32>
    %s1b2n2sq = stablehlo.multiply %s1b2n2xc, %s1b2n2xc : tensor<32x64x56x56xf32>
    %s1b2n2vsr = stablehlo.reduce(%s1b2n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2n2vs = stablehlo.broadcast_in_dim %s1b2n2vsr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2vr = stablehlo.divide %s1b2n2vs, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2n2ve = stablehlo.add %s1b2n2vr, %s1b2n2ep : tensor<32x64x56x56xf32>
    %s1b2n2istd = stablehlo.rsqrt %s1b2n2ve : tensor<32x64x56x56xf32>
    %s1b2n2xh = stablehlo.multiply %s1b2n2xc, %s1b2n2istd : tensor<32x64x56x56xf32>
    %s1b2n2gb = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2btb = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2n2gx = stablehlo.multiply %s1b2n2xh, %s1b2n2gb : tensor<32x64x56x56xf32>
    %s1b2n2 = stablehlo.add %s1b2n2gx, %s1b2n2btb : tensor<32x64x56x56xf32>
    %s1b2a = stablehlo.add %s1b2n2, %s1b1o : tensor<32x64x56x56xf32>
    %s1b2oz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2o = stablehlo.maximum %s1b2a, %s1b2oz : tensor<32x64x56x56xf32>
    %d2c1c = stablehlo.convolution(%s1b2o, %d2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2c1bb = stablehlo.broadcast_in_dim %d2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2c1 = stablehlo.add %d2c1c, %d2c1bb : tensor<32x128x28x28xf32>
    %d2n1nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %d2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n1smr = stablehlo.reduce(%d2c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2n1sm = stablehlo.broadcast_in_dim %d2n1smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1mu = stablehlo.divide %d2n1sm, %d2n1nf : tensor<32x128x28x28xf32>
    %d2n1xc = stablehlo.subtract %d2c1, %d2n1mu : tensor<32x128x28x28xf32>
    %d2n1sq = stablehlo.multiply %d2n1xc, %d2n1xc : tensor<32x128x28x28xf32>
    %d2n1vsr = stablehlo.reduce(%d2n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2n1vs = stablehlo.broadcast_in_dim %d2n1vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1vr = stablehlo.divide %d2n1vs, %d2n1nf : tensor<32x128x28x28xf32>
    %d2n1ve = stablehlo.add %d2n1vr, %d2n1ep : tensor<32x128x28x28xf32>
    %d2n1istd = stablehlo.rsqrt %d2n1ve : tensor<32x128x28x28xf32>
    %d2n1xh = stablehlo.multiply %d2n1xc, %d2n1istd : tensor<32x128x28x28xf32>
    %d2n1gb = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1btb = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n1gx = stablehlo.multiply %d2n1xh, %d2n1gb : tensor<32x128x28x28xf32>
    %d2n1 = stablehlo.add %d2n1gx, %d2n1btb : tensor<32x128x28x28xf32>
    %d2r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2r1 = stablehlo.maximum %d2n1, %d2r1z : tensor<32x128x28x28xf32>
    %d2c2c = stablehlo.convolution(%d2r1, %d2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2c2bb = stablehlo.broadcast_in_dim %d2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2c2 = stablehlo.add %d2c2c, %d2c2bb : tensor<32x128x28x28xf32>
    %d2n2nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %d2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2n2smr = stablehlo.reduce(%d2c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2n2sm = stablehlo.broadcast_in_dim %d2n2smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2mu = stablehlo.divide %d2n2sm, %d2n2nf : tensor<32x128x28x28xf32>
    %d2n2xc = stablehlo.subtract %d2c2, %d2n2mu : tensor<32x128x28x28xf32>
    %d2n2sq = stablehlo.multiply %d2n2xc, %d2n2xc : tensor<32x128x28x28xf32>
    %d2n2vsr = stablehlo.reduce(%d2n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2n2vs = stablehlo.broadcast_in_dim %d2n2vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2vr = stablehlo.divide %d2n2vs, %d2n2nf : tensor<32x128x28x28xf32>
    %d2n2ve = stablehlo.add %d2n2vr, %d2n2ep : tensor<32x128x28x28xf32>
    %d2n2istd = stablehlo.rsqrt %d2n2ve : tensor<32x128x28x28xf32>
    %d2n2xh = stablehlo.multiply %d2n2xc, %d2n2istd : tensor<32x128x28x28xf32>
    %d2n2gb = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2btb = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2n2gx = stablehlo.multiply %d2n2xh, %d2n2gb : tensor<32x128x28x28xf32>
    %d2n2 = stablehlo.add %d2n2gx, %d2n2btb : tensor<32x128x28x28xf32>
    %d2cpc = stablehlo.convolution(%s1b2o, %d2Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<128x64x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2cpbb = stablehlo.broadcast_in_dim %d2bp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2cp = stablehlo.add %d2cpc, %d2cpbb : tensor<32x128x28x28xf32>
    %d2npnf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %d2npep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %d2npsmr = stablehlo.reduce(%d2cp init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2npsm = stablehlo.broadcast_in_dim %d2npsmr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2npmu = stablehlo.divide %d2npsm, %d2npnf : tensor<32x128x28x28xf32>
    %d2npxc = stablehlo.subtract %d2cp, %d2npmu : tensor<32x128x28x28xf32>
    %d2npsq = stablehlo.multiply %d2npxc, %d2npxc : tensor<32x128x28x28xf32>
    %d2npvsr = stablehlo.reduce(%d2npsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2npvs = stablehlo.broadcast_in_dim %d2npvsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2npvr = stablehlo.divide %d2npvs, %d2npnf : tensor<32x128x28x28xf32>
    %d2npve = stablehlo.add %d2npvr, %d2npep : tensor<32x128x28x28xf32>
    %d2npistd = stablehlo.rsqrt %d2npve : tensor<32x128x28x28xf32>
    %d2npxh = stablehlo.multiply %d2npxc, %d2npistd : tensor<32x128x28x28xf32>
    %d2npgb = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npbtb = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %d2npgx = stablehlo.multiply %d2npxh, %d2npgb : tensor<32x128x28x28xf32>
    %d2np = stablehlo.add %d2npgx, %d2npbtb : tensor<32x128x28x28xf32>
    %d2a = stablehlo.add %d2n2, %d2np : tensor<32x128x28x28xf32>
    %d2oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2o = stablehlo.maximum %d2a, %d2oz : tensor<32x128x28x28xf32>
    %s2b0c1c = stablehlo.convolution(%d2o, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c1bb = stablehlo.broadcast_in_dim %s2b0b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c1 = stablehlo.add %s2b0c1c, %s2b0c1bb : tensor<32x128x28x28xf32>
    %s2b0n1nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n1smr = stablehlo.reduce(%s2b0c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0n1sm = stablehlo.broadcast_in_dim %s2b0n1smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1mu = stablehlo.divide %s2b0n1sm, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0n1xc = stablehlo.subtract %s2b0c1, %s2b0n1mu : tensor<32x128x28x28xf32>
    %s2b0n1sq = stablehlo.multiply %s2b0n1xc, %s2b0n1xc : tensor<32x128x28x28xf32>
    %s2b0n1vsr = stablehlo.reduce(%s2b0n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0n1vs = stablehlo.broadcast_in_dim %s2b0n1vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1vr = stablehlo.divide %s2b0n1vs, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0n1ve = stablehlo.add %s2b0n1vr, %s2b0n1ep : tensor<32x128x28x28xf32>
    %s2b0n1istd = stablehlo.rsqrt %s2b0n1ve : tensor<32x128x28x28xf32>
    %s2b0n1xh = stablehlo.multiply %s2b0n1xc, %s2b0n1istd : tensor<32x128x28x28xf32>
    %s2b0n1gb = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1btb = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n1gx = stablehlo.multiply %s2b0n1xh, %s2b0n1gb : tensor<32x128x28x28xf32>
    %s2b0n1 = stablehlo.add %s2b0n1gx, %s2b0n1btb : tensor<32x128x28x28xf32>
    %s2b0r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0r1 = stablehlo.maximum %s2b0n1, %s2b0r1z : tensor<32x128x28x28xf32>
    %s2b0c2c = stablehlo.convolution(%s2b0r1, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c2bb = stablehlo.broadcast_in_dim %s2b0b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0c2 = stablehlo.add %s2b0c2c, %s2b0c2bb : tensor<32x128x28x28xf32>
    %s2b0n2nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b0n2smr = stablehlo.reduce(%s2b0c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0n2sm = stablehlo.broadcast_in_dim %s2b0n2smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2mu = stablehlo.divide %s2b0n2sm, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0n2xc = stablehlo.subtract %s2b0c2, %s2b0n2mu : tensor<32x128x28x28xf32>
    %s2b0n2sq = stablehlo.multiply %s2b0n2xc, %s2b0n2xc : tensor<32x128x28x28xf32>
    %s2b0n2vsr = stablehlo.reduce(%s2b0n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0n2vs = stablehlo.broadcast_in_dim %s2b0n2vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2vr = stablehlo.divide %s2b0n2vs, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0n2ve = stablehlo.add %s2b0n2vr, %s2b0n2ep : tensor<32x128x28x28xf32>
    %s2b0n2istd = stablehlo.rsqrt %s2b0n2ve : tensor<32x128x28x28xf32>
    %s2b0n2xh = stablehlo.multiply %s2b0n2xc, %s2b0n2istd : tensor<32x128x28x28xf32>
    %s2b0n2gb = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2btb = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0n2gx = stablehlo.multiply %s2b0n2xh, %s2b0n2gb : tensor<32x128x28x28xf32>
    %s2b0n2 = stablehlo.add %s2b0n2gx, %s2b0n2btb : tensor<32x128x28x28xf32>
    %s2b0a = stablehlo.add %s2b0n2, %d2o : tensor<32x128x28x28xf32>
    %s2b0oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0o = stablehlo.maximum %s2b0a, %s2b0oz : tensor<32x128x28x28xf32>
    %s2b1c1c = stablehlo.convolution(%s2b0o, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c1bb = stablehlo.broadcast_in_dim %s2b1b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c1 = stablehlo.add %s2b1c1c, %s2b1c1bb : tensor<32x128x28x28xf32>
    %s2b1n1nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n1smr = stablehlo.reduce(%s2b1c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1n1sm = stablehlo.broadcast_in_dim %s2b1n1smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1mu = stablehlo.divide %s2b1n1sm, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1n1xc = stablehlo.subtract %s2b1c1, %s2b1n1mu : tensor<32x128x28x28xf32>
    %s2b1n1sq = stablehlo.multiply %s2b1n1xc, %s2b1n1xc : tensor<32x128x28x28xf32>
    %s2b1n1vsr = stablehlo.reduce(%s2b1n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1n1vs = stablehlo.broadcast_in_dim %s2b1n1vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1vr = stablehlo.divide %s2b1n1vs, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1n1ve = stablehlo.add %s2b1n1vr, %s2b1n1ep : tensor<32x128x28x28xf32>
    %s2b1n1istd = stablehlo.rsqrt %s2b1n1ve : tensor<32x128x28x28xf32>
    %s2b1n1xh = stablehlo.multiply %s2b1n1xc, %s2b1n1istd : tensor<32x128x28x28xf32>
    %s2b1n1gb = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1btb = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n1gx = stablehlo.multiply %s2b1n1xh, %s2b1n1gb : tensor<32x128x28x28xf32>
    %s2b1n1 = stablehlo.add %s2b1n1gx, %s2b1n1btb : tensor<32x128x28x28xf32>
    %s2b1r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1r1 = stablehlo.maximum %s2b1n1, %s2b1r1z : tensor<32x128x28x28xf32>
    %s2b1c2c = stablehlo.convolution(%s2b1r1, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c2bb = stablehlo.broadcast_in_dim %s2b1b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1c2 = stablehlo.add %s2b1c2c, %s2b1c2bb : tensor<32x128x28x28xf32>
    %s2b1n2nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b1n2smr = stablehlo.reduce(%s2b1c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1n2sm = stablehlo.broadcast_in_dim %s2b1n2smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2mu = stablehlo.divide %s2b1n2sm, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1n2xc = stablehlo.subtract %s2b1c2, %s2b1n2mu : tensor<32x128x28x28xf32>
    %s2b1n2sq = stablehlo.multiply %s2b1n2xc, %s2b1n2xc : tensor<32x128x28x28xf32>
    %s2b1n2vsr = stablehlo.reduce(%s2b1n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1n2vs = stablehlo.broadcast_in_dim %s2b1n2vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2vr = stablehlo.divide %s2b1n2vs, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1n2ve = stablehlo.add %s2b1n2vr, %s2b1n2ep : tensor<32x128x28x28xf32>
    %s2b1n2istd = stablehlo.rsqrt %s2b1n2ve : tensor<32x128x28x28xf32>
    %s2b1n2xh = stablehlo.multiply %s2b1n2xc, %s2b1n2istd : tensor<32x128x28x28xf32>
    %s2b1n2gb = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2btb = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1n2gx = stablehlo.multiply %s2b1n2xh, %s2b1n2gb : tensor<32x128x28x28xf32>
    %s2b1n2 = stablehlo.add %s2b1n2gx, %s2b1n2btb : tensor<32x128x28x28xf32>
    %s2b1a = stablehlo.add %s2b1n2, %s2b0o : tensor<32x128x28x28xf32>
    %s2b1oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1o = stablehlo.maximum %s2b1a, %s2b1oz : tensor<32x128x28x28xf32>
    %s2b2c1c = stablehlo.convolution(%s2b1o, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c1bb = stablehlo.broadcast_in_dim %s2b2b1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c1 = stablehlo.add %s2b2c1c, %s2b2c1bb : tensor<32x128x28x28xf32>
    %s2b2n1nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n1smr = stablehlo.reduce(%s2b2c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2n1sm = stablehlo.broadcast_in_dim %s2b2n1smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1mu = stablehlo.divide %s2b2n1sm, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2n1xc = stablehlo.subtract %s2b2c1, %s2b2n1mu : tensor<32x128x28x28xf32>
    %s2b2n1sq = stablehlo.multiply %s2b2n1xc, %s2b2n1xc : tensor<32x128x28x28xf32>
    %s2b2n1vsr = stablehlo.reduce(%s2b2n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2n1vs = stablehlo.broadcast_in_dim %s2b2n1vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1vr = stablehlo.divide %s2b2n1vs, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2n1ve = stablehlo.add %s2b2n1vr, %s2b2n1ep : tensor<32x128x28x28xf32>
    %s2b2n1istd = stablehlo.rsqrt %s2b2n1ve : tensor<32x128x28x28xf32>
    %s2b2n1xh = stablehlo.multiply %s2b2n1xc, %s2b2n1istd : tensor<32x128x28x28xf32>
    %s2b2n1gb = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1btb = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n1gx = stablehlo.multiply %s2b2n1xh, %s2b2n1gb : tensor<32x128x28x28xf32>
    %s2b2n1 = stablehlo.add %s2b2n1gx, %s2b2n1btb : tensor<32x128x28x28xf32>
    %s2b2r1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2r1 = stablehlo.maximum %s2b2n1, %s2b2r1z : tensor<32x128x28x28xf32>
    %s2b2c2c = stablehlo.convolution(%s2b2r1, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c2bb = stablehlo.broadcast_in_dim %s2b2b2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2c2 = stablehlo.add %s2b2c2c, %s2b2c2bb : tensor<32x128x28x28xf32>
    %s2b2n2nf = stablehlo.constant dense<784.0> : tensor<32x128x28x28xf32>
    %s2b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x128x28x28xf32>
    %s2b2n2smr = stablehlo.reduce(%s2b2c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2n2sm = stablehlo.broadcast_in_dim %s2b2n2smr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2mu = stablehlo.divide %s2b2n2sm, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2n2xc = stablehlo.subtract %s2b2c2, %s2b2n2mu : tensor<32x128x28x28xf32>
    %s2b2n2sq = stablehlo.multiply %s2b2n2xc, %s2b2n2xc : tensor<32x128x28x28xf32>
    %s2b2n2vsr = stablehlo.reduce(%s2b2n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2n2vs = stablehlo.broadcast_in_dim %s2b2n2vsr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2vr = stablehlo.divide %s2b2n2vs, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2n2ve = stablehlo.add %s2b2n2vr, %s2b2n2ep : tensor<32x128x28x28xf32>
    %s2b2n2istd = stablehlo.rsqrt %s2b2n2ve : tensor<32x128x28x28xf32>
    %s2b2n2xh = stablehlo.multiply %s2b2n2xc, %s2b2n2istd : tensor<32x128x28x28xf32>
    %s2b2n2gb = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2btb = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2n2gx = stablehlo.multiply %s2b2n2xh, %s2b2n2gb : tensor<32x128x28x28xf32>
    %s2b2n2 = stablehlo.add %s2b2n2gx, %s2b2n2btb : tensor<32x128x28x28xf32>
    %s2b2a = stablehlo.add %s2b2n2, %s2b1o : tensor<32x128x28x28xf32>
    %s2b2oz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2o = stablehlo.maximum %s2b2a, %s2b2oz : tensor<32x128x28x28xf32>
    %d3c1c = stablehlo.convolution(%s2b2o, %d3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3c1bb = stablehlo.broadcast_in_dim %d3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3c1 = stablehlo.add %d3c1c, %d3c1bb : tensor<32x256x14x14xf32>
    %d3n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %d3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n1smr = stablehlo.reduce(%d3c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3n1sm = stablehlo.broadcast_in_dim %d3n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1mu = stablehlo.divide %d3n1sm, %d3n1nf : tensor<32x256x14x14xf32>
    %d3n1xc = stablehlo.subtract %d3c1, %d3n1mu : tensor<32x256x14x14xf32>
    %d3n1sq = stablehlo.multiply %d3n1xc, %d3n1xc : tensor<32x256x14x14xf32>
    %d3n1vsr = stablehlo.reduce(%d3n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3n1vs = stablehlo.broadcast_in_dim %d3n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1vr = stablehlo.divide %d3n1vs, %d3n1nf : tensor<32x256x14x14xf32>
    %d3n1ve = stablehlo.add %d3n1vr, %d3n1ep : tensor<32x256x14x14xf32>
    %d3n1istd = stablehlo.rsqrt %d3n1ve : tensor<32x256x14x14xf32>
    %d3n1xh = stablehlo.multiply %d3n1xc, %d3n1istd : tensor<32x256x14x14xf32>
    %d3n1gb = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1btb = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n1gx = stablehlo.multiply %d3n1xh, %d3n1gb : tensor<32x256x14x14xf32>
    %d3n1 = stablehlo.add %d3n1gx, %d3n1btb : tensor<32x256x14x14xf32>
    %d3r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3r1 = stablehlo.maximum %d3n1, %d3r1z : tensor<32x256x14x14xf32>
    %d3c2c = stablehlo.convolution(%d3r1, %d3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3c2bb = stablehlo.broadcast_in_dim %d3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3c2 = stablehlo.add %d3c2c, %d3c2bb : tensor<32x256x14x14xf32>
    %d3n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %d3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3n2smr = stablehlo.reduce(%d3c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3n2sm = stablehlo.broadcast_in_dim %d3n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2mu = stablehlo.divide %d3n2sm, %d3n2nf : tensor<32x256x14x14xf32>
    %d3n2xc = stablehlo.subtract %d3c2, %d3n2mu : tensor<32x256x14x14xf32>
    %d3n2sq = stablehlo.multiply %d3n2xc, %d3n2xc : tensor<32x256x14x14xf32>
    %d3n2vsr = stablehlo.reduce(%d3n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3n2vs = stablehlo.broadcast_in_dim %d3n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2vr = stablehlo.divide %d3n2vs, %d3n2nf : tensor<32x256x14x14xf32>
    %d3n2ve = stablehlo.add %d3n2vr, %d3n2ep : tensor<32x256x14x14xf32>
    %d3n2istd = stablehlo.rsqrt %d3n2ve : tensor<32x256x14x14xf32>
    %d3n2xh = stablehlo.multiply %d3n2xc, %d3n2istd : tensor<32x256x14x14xf32>
    %d3n2gb = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2btb = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3n2gx = stablehlo.multiply %d3n2xh, %d3n2gb : tensor<32x256x14x14xf32>
    %d3n2 = stablehlo.add %d3n2gx, %d3n2btb : tensor<32x256x14x14xf32>
    %d3cpc = stablehlo.convolution(%s2b2o, %d3Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<256x128x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3cpbb = stablehlo.broadcast_in_dim %d3bp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3cp = stablehlo.add %d3cpc, %d3cpbb : tensor<32x256x14x14xf32>
    %d3npnf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %d3npep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %d3npsmr = stablehlo.reduce(%d3cp init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3npsm = stablehlo.broadcast_in_dim %d3npsmr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3npmu = stablehlo.divide %d3npsm, %d3npnf : tensor<32x256x14x14xf32>
    %d3npxc = stablehlo.subtract %d3cp, %d3npmu : tensor<32x256x14x14xf32>
    %d3npsq = stablehlo.multiply %d3npxc, %d3npxc : tensor<32x256x14x14xf32>
    %d3npvsr = stablehlo.reduce(%d3npsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3npvs = stablehlo.broadcast_in_dim %d3npvsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3npvr = stablehlo.divide %d3npvs, %d3npnf : tensor<32x256x14x14xf32>
    %d3npve = stablehlo.add %d3npvr, %d3npep : tensor<32x256x14x14xf32>
    %d3npistd = stablehlo.rsqrt %d3npve : tensor<32x256x14x14xf32>
    %d3npxh = stablehlo.multiply %d3npxc, %d3npistd : tensor<32x256x14x14xf32>
    %d3npgb = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npbtb = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %d3npgx = stablehlo.multiply %d3npxh, %d3npgb : tensor<32x256x14x14xf32>
    %d3np = stablehlo.add %d3npgx, %d3npbtb : tensor<32x256x14x14xf32>
    %d3a = stablehlo.add %d3n2, %d3np : tensor<32x256x14x14xf32>
    %d3oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3o = stablehlo.maximum %d3a, %d3oz : tensor<32x256x14x14xf32>
    %s3b0c1c = stablehlo.convolution(%d3o, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c1bb = stablehlo.broadcast_in_dim %s3b0b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c1 = stablehlo.add %s3b0c1c, %s3b0c1bb : tensor<32x256x14x14xf32>
    %s3b0n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n1smr = stablehlo.reduce(%s3b0c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0n1sm = stablehlo.broadcast_in_dim %s3b0n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1mu = stablehlo.divide %s3b0n1sm, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0n1xc = stablehlo.subtract %s3b0c1, %s3b0n1mu : tensor<32x256x14x14xf32>
    %s3b0n1sq = stablehlo.multiply %s3b0n1xc, %s3b0n1xc : tensor<32x256x14x14xf32>
    %s3b0n1vsr = stablehlo.reduce(%s3b0n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0n1vs = stablehlo.broadcast_in_dim %s3b0n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1vr = stablehlo.divide %s3b0n1vs, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0n1ve = stablehlo.add %s3b0n1vr, %s3b0n1ep : tensor<32x256x14x14xf32>
    %s3b0n1istd = stablehlo.rsqrt %s3b0n1ve : tensor<32x256x14x14xf32>
    %s3b0n1xh = stablehlo.multiply %s3b0n1xc, %s3b0n1istd : tensor<32x256x14x14xf32>
    %s3b0n1gb = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1btb = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n1gx = stablehlo.multiply %s3b0n1xh, %s3b0n1gb : tensor<32x256x14x14xf32>
    %s3b0n1 = stablehlo.add %s3b0n1gx, %s3b0n1btb : tensor<32x256x14x14xf32>
    %s3b0r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0r1 = stablehlo.maximum %s3b0n1, %s3b0r1z : tensor<32x256x14x14xf32>
    %s3b0c2c = stablehlo.convolution(%s3b0r1, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c2bb = stablehlo.broadcast_in_dim %s3b0b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0c2 = stablehlo.add %s3b0c2c, %s3b0c2bb : tensor<32x256x14x14xf32>
    %s3b0n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b0n2smr = stablehlo.reduce(%s3b0c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0n2sm = stablehlo.broadcast_in_dim %s3b0n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2mu = stablehlo.divide %s3b0n2sm, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0n2xc = stablehlo.subtract %s3b0c2, %s3b0n2mu : tensor<32x256x14x14xf32>
    %s3b0n2sq = stablehlo.multiply %s3b0n2xc, %s3b0n2xc : tensor<32x256x14x14xf32>
    %s3b0n2vsr = stablehlo.reduce(%s3b0n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0n2vs = stablehlo.broadcast_in_dim %s3b0n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2vr = stablehlo.divide %s3b0n2vs, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0n2ve = stablehlo.add %s3b0n2vr, %s3b0n2ep : tensor<32x256x14x14xf32>
    %s3b0n2istd = stablehlo.rsqrt %s3b0n2ve : tensor<32x256x14x14xf32>
    %s3b0n2xh = stablehlo.multiply %s3b0n2xc, %s3b0n2istd : tensor<32x256x14x14xf32>
    %s3b0n2gb = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2btb = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0n2gx = stablehlo.multiply %s3b0n2xh, %s3b0n2gb : tensor<32x256x14x14xf32>
    %s3b0n2 = stablehlo.add %s3b0n2gx, %s3b0n2btb : tensor<32x256x14x14xf32>
    %s3b0a = stablehlo.add %s3b0n2, %d3o : tensor<32x256x14x14xf32>
    %s3b0oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0o = stablehlo.maximum %s3b0a, %s3b0oz : tensor<32x256x14x14xf32>
    %s3b1c1c = stablehlo.convolution(%s3b0o, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c1bb = stablehlo.broadcast_in_dim %s3b1b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c1 = stablehlo.add %s3b1c1c, %s3b1c1bb : tensor<32x256x14x14xf32>
    %s3b1n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n1smr = stablehlo.reduce(%s3b1c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1n1sm = stablehlo.broadcast_in_dim %s3b1n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1mu = stablehlo.divide %s3b1n1sm, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1n1xc = stablehlo.subtract %s3b1c1, %s3b1n1mu : tensor<32x256x14x14xf32>
    %s3b1n1sq = stablehlo.multiply %s3b1n1xc, %s3b1n1xc : tensor<32x256x14x14xf32>
    %s3b1n1vsr = stablehlo.reduce(%s3b1n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1n1vs = stablehlo.broadcast_in_dim %s3b1n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1vr = stablehlo.divide %s3b1n1vs, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1n1ve = stablehlo.add %s3b1n1vr, %s3b1n1ep : tensor<32x256x14x14xf32>
    %s3b1n1istd = stablehlo.rsqrt %s3b1n1ve : tensor<32x256x14x14xf32>
    %s3b1n1xh = stablehlo.multiply %s3b1n1xc, %s3b1n1istd : tensor<32x256x14x14xf32>
    %s3b1n1gb = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1btb = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n1gx = stablehlo.multiply %s3b1n1xh, %s3b1n1gb : tensor<32x256x14x14xf32>
    %s3b1n1 = stablehlo.add %s3b1n1gx, %s3b1n1btb : tensor<32x256x14x14xf32>
    %s3b1r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1r1 = stablehlo.maximum %s3b1n1, %s3b1r1z : tensor<32x256x14x14xf32>
    %s3b1c2c = stablehlo.convolution(%s3b1r1, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c2bb = stablehlo.broadcast_in_dim %s3b1b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1c2 = stablehlo.add %s3b1c2c, %s3b1c2bb : tensor<32x256x14x14xf32>
    %s3b1n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b1n2smr = stablehlo.reduce(%s3b1c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1n2sm = stablehlo.broadcast_in_dim %s3b1n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2mu = stablehlo.divide %s3b1n2sm, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1n2xc = stablehlo.subtract %s3b1c2, %s3b1n2mu : tensor<32x256x14x14xf32>
    %s3b1n2sq = stablehlo.multiply %s3b1n2xc, %s3b1n2xc : tensor<32x256x14x14xf32>
    %s3b1n2vsr = stablehlo.reduce(%s3b1n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1n2vs = stablehlo.broadcast_in_dim %s3b1n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2vr = stablehlo.divide %s3b1n2vs, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1n2ve = stablehlo.add %s3b1n2vr, %s3b1n2ep : tensor<32x256x14x14xf32>
    %s3b1n2istd = stablehlo.rsqrt %s3b1n2ve : tensor<32x256x14x14xf32>
    %s3b1n2xh = stablehlo.multiply %s3b1n2xc, %s3b1n2istd : tensor<32x256x14x14xf32>
    %s3b1n2gb = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2btb = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1n2gx = stablehlo.multiply %s3b1n2xh, %s3b1n2gb : tensor<32x256x14x14xf32>
    %s3b1n2 = stablehlo.add %s3b1n2gx, %s3b1n2btb : tensor<32x256x14x14xf32>
    %s3b1a = stablehlo.add %s3b1n2, %s3b0o : tensor<32x256x14x14xf32>
    %s3b1oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1o = stablehlo.maximum %s3b1a, %s3b1oz : tensor<32x256x14x14xf32>
    %s3b2c1c = stablehlo.convolution(%s3b1o, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c1bb = stablehlo.broadcast_in_dim %s3b2b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c1 = stablehlo.add %s3b2c1c, %s3b2c1bb : tensor<32x256x14x14xf32>
    %s3b2n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b2n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n1smr = stablehlo.reduce(%s3b2c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2n1sm = stablehlo.broadcast_in_dim %s3b2n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1mu = stablehlo.divide %s3b2n1sm, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2n1xc = stablehlo.subtract %s3b2c1, %s3b2n1mu : tensor<32x256x14x14xf32>
    %s3b2n1sq = stablehlo.multiply %s3b2n1xc, %s3b2n1xc : tensor<32x256x14x14xf32>
    %s3b2n1vsr = stablehlo.reduce(%s3b2n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2n1vs = stablehlo.broadcast_in_dim %s3b2n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1vr = stablehlo.divide %s3b2n1vs, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2n1ve = stablehlo.add %s3b2n1vr, %s3b2n1ep : tensor<32x256x14x14xf32>
    %s3b2n1istd = stablehlo.rsqrt %s3b2n1ve : tensor<32x256x14x14xf32>
    %s3b2n1xh = stablehlo.multiply %s3b2n1xc, %s3b2n1istd : tensor<32x256x14x14xf32>
    %s3b2n1gb = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1btb = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n1gx = stablehlo.multiply %s3b2n1xh, %s3b2n1gb : tensor<32x256x14x14xf32>
    %s3b2n1 = stablehlo.add %s3b2n1gx, %s3b2n1btb : tensor<32x256x14x14xf32>
    %s3b2r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2r1 = stablehlo.maximum %s3b2n1, %s3b2r1z : tensor<32x256x14x14xf32>
    %s3b2c2c = stablehlo.convolution(%s3b2r1, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c2bb = stablehlo.broadcast_in_dim %s3b2b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2c2 = stablehlo.add %s3b2c2c, %s3b2c2bb : tensor<32x256x14x14xf32>
    %s3b2n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b2n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b2n2smr = stablehlo.reduce(%s3b2c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2n2sm = stablehlo.broadcast_in_dim %s3b2n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2mu = stablehlo.divide %s3b2n2sm, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2n2xc = stablehlo.subtract %s3b2c2, %s3b2n2mu : tensor<32x256x14x14xf32>
    %s3b2n2sq = stablehlo.multiply %s3b2n2xc, %s3b2n2xc : tensor<32x256x14x14xf32>
    %s3b2n2vsr = stablehlo.reduce(%s3b2n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2n2vs = stablehlo.broadcast_in_dim %s3b2n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2vr = stablehlo.divide %s3b2n2vs, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2n2ve = stablehlo.add %s3b2n2vr, %s3b2n2ep : tensor<32x256x14x14xf32>
    %s3b2n2istd = stablehlo.rsqrt %s3b2n2ve : tensor<32x256x14x14xf32>
    %s3b2n2xh = stablehlo.multiply %s3b2n2xc, %s3b2n2istd : tensor<32x256x14x14xf32>
    %s3b2n2gb = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2btb = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2n2gx = stablehlo.multiply %s3b2n2xh, %s3b2n2gb : tensor<32x256x14x14xf32>
    %s3b2n2 = stablehlo.add %s3b2n2gx, %s3b2n2btb : tensor<32x256x14x14xf32>
    %s3b2a = stablehlo.add %s3b2n2, %s3b1o : tensor<32x256x14x14xf32>
    %s3b2oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2o = stablehlo.maximum %s3b2a, %s3b2oz : tensor<32x256x14x14xf32>
    %s3b3c1c = stablehlo.convolution(%s3b2o, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c1bb = stablehlo.broadcast_in_dim %s3b3b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c1 = stablehlo.add %s3b3c1c, %s3b3c1bb : tensor<32x256x14x14xf32>
    %s3b3n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b3n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n1smr = stablehlo.reduce(%s3b3c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3n1sm = stablehlo.broadcast_in_dim %s3b3n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1mu = stablehlo.divide %s3b3n1sm, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3n1xc = stablehlo.subtract %s3b3c1, %s3b3n1mu : tensor<32x256x14x14xf32>
    %s3b3n1sq = stablehlo.multiply %s3b3n1xc, %s3b3n1xc : tensor<32x256x14x14xf32>
    %s3b3n1vsr = stablehlo.reduce(%s3b3n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3n1vs = stablehlo.broadcast_in_dim %s3b3n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1vr = stablehlo.divide %s3b3n1vs, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3n1ve = stablehlo.add %s3b3n1vr, %s3b3n1ep : tensor<32x256x14x14xf32>
    %s3b3n1istd = stablehlo.rsqrt %s3b3n1ve : tensor<32x256x14x14xf32>
    %s3b3n1xh = stablehlo.multiply %s3b3n1xc, %s3b3n1istd : tensor<32x256x14x14xf32>
    %s3b3n1gb = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1btb = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n1gx = stablehlo.multiply %s3b3n1xh, %s3b3n1gb : tensor<32x256x14x14xf32>
    %s3b3n1 = stablehlo.add %s3b3n1gx, %s3b3n1btb : tensor<32x256x14x14xf32>
    %s3b3r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3r1 = stablehlo.maximum %s3b3n1, %s3b3r1z : tensor<32x256x14x14xf32>
    %s3b3c2c = stablehlo.convolution(%s3b3r1, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c2bb = stablehlo.broadcast_in_dim %s3b3b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3c2 = stablehlo.add %s3b3c2c, %s3b3c2bb : tensor<32x256x14x14xf32>
    %s3b3n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b3n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b3n2smr = stablehlo.reduce(%s3b3c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3n2sm = stablehlo.broadcast_in_dim %s3b3n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2mu = stablehlo.divide %s3b3n2sm, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3n2xc = stablehlo.subtract %s3b3c2, %s3b3n2mu : tensor<32x256x14x14xf32>
    %s3b3n2sq = stablehlo.multiply %s3b3n2xc, %s3b3n2xc : tensor<32x256x14x14xf32>
    %s3b3n2vsr = stablehlo.reduce(%s3b3n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3n2vs = stablehlo.broadcast_in_dim %s3b3n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2vr = stablehlo.divide %s3b3n2vs, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3n2ve = stablehlo.add %s3b3n2vr, %s3b3n2ep : tensor<32x256x14x14xf32>
    %s3b3n2istd = stablehlo.rsqrt %s3b3n2ve : tensor<32x256x14x14xf32>
    %s3b3n2xh = stablehlo.multiply %s3b3n2xc, %s3b3n2istd : tensor<32x256x14x14xf32>
    %s3b3n2gb = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2btb = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3n2gx = stablehlo.multiply %s3b3n2xh, %s3b3n2gb : tensor<32x256x14x14xf32>
    %s3b3n2 = stablehlo.add %s3b3n2gx, %s3b3n2btb : tensor<32x256x14x14xf32>
    %s3b3a = stablehlo.add %s3b3n2, %s3b2o : tensor<32x256x14x14xf32>
    %s3b3oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3o = stablehlo.maximum %s3b3a, %s3b3oz : tensor<32x256x14x14xf32>
    %s3b4c1c = stablehlo.convolution(%s3b3o, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c1bb = stablehlo.broadcast_in_dim %s3b4b1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c1 = stablehlo.add %s3b4c1c, %s3b4c1bb : tensor<32x256x14x14xf32>
    %s3b4n1nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n1smr = stablehlo.reduce(%s3b4c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4n1sm = stablehlo.broadcast_in_dim %s3b4n1smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1mu = stablehlo.divide %s3b4n1sm, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4n1xc = stablehlo.subtract %s3b4c1, %s3b4n1mu : tensor<32x256x14x14xf32>
    %s3b4n1sq = stablehlo.multiply %s3b4n1xc, %s3b4n1xc : tensor<32x256x14x14xf32>
    %s3b4n1vsr = stablehlo.reduce(%s3b4n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4n1vs = stablehlo.broadcast_in_dim %s3b4n1vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1vr = stablehlo.divide %s3b4n1vs, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4n1ve = stablehlo.add %s3b4n1vr, %s3b4n1ep : tensor<32x256x14x14xf32>
    %s3b4n1istd = stablehlo.rsqrt %s3b4n1ve : tensor<32x256x14x14xf32>
    %s3b4n1xh = stablehlo.multiply %s3b4n1xc, %s3b4n1istd : tensor<32x256x14x14xf32>
    %s3b4n1gb = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1btb = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n1gx = stablehlo.multiply %s3b4n1xh, %s3b4n1gb : tensor<32x256x14x14xf32>
    %s3b4n1 = stablehlo.add %s3b4n1gx, %s3b4n1btb : tensor<32x256x14x14xf32>
    %s3b4r1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4r1 = stablehlo.maximum %s3b4n1, %s3b4r1z : tensor<32x256x14x14xf32>
    %s3b4c2c = stablehlo.convolution(%s3b4r1, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c2bb = stablehlo.broadcast_in_dim %s3b4b2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4c2 = stablehlo.add %s3b4c2c, %s3b4c2bb : tensor<32x256x14x14xf32>
    %s3b4n2nf = stablehlo.constant dense<196.0> : tensor<32x256x14x14xf32>
    %s3b4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x256x14x14xf32>
    %s3b4n2smr = stablehlo.reduce(%s3b4c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4n2sm = stablehlo.broadcast_in_dim %s3b4n2smr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2mu = stablehlo.divide %s3b4n2sm, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4n2xc = stablehlo.subtract %s3b4c2, %s3b4n2mu : tensor<32x256x14x14xf32>
    %s3b4n2sq = stablehlo.multiply %s3b4n2xc, %s3b4n2xc : tensor<32x256x14x14xf32>
    %s3b4n2vsr = stablehlo.reduce(%s3b4n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4n2vs = stablehlo.broadcast_in_dim %s3b4n2vsr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2vr = stablehlo.divide %s3b4n2vs, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4n2ve = stablehlo.add %s3b4n2vr, %s3b4n2ep : tensor<32x256x14x14xf32>
    %s3b4n2istd = stablehlo.rsqrt %s3b4n2ve : tensor<32x256x14x14xf32>
    %s3b4n2xh = stablehlo.multiply %s3b4n2xc, %s3b4n2istd : tensor<32x256x14x14xf32>
    %s3b4n2gb = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2btb = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4n2gx = stablehlo.multiply %s3b4n2xh, %s3b4n2gb : tensor<32x256x14x14xf32>
    %s3b4n2 = stablehlo.add %s3b4n2gx, %s3b4n2btb : tensor<32x256x14x14xf32>
    %s3b4a = stablehlo.add %s3b4n2, %s3b3o : tensor<32x256x14x14xf32>
    %s3b4oz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4o = stablehlo.maximum %s3b4a, %s3b4oz : tensor<32x256x14x14xf32>
    %d4c1c = stablehlo.convolution(%s3b4o, %d4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4c1bb = stablehlo.broadcast_in_dim %d4b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4c1 = stablehlo.add %d4c1c, %d4c1bb : tensor<32x512x7x7xf32>
    %d4n1nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %d4n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n1smr = stablehlo.reduce(%d4c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4n1sm = stablehlo.broadcast_in_dim %d4n1smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1mu = stablehlo.divide %d4n1sm, %d4n1nf : tensor<32x512x7x7xf32>
    %d4n1xc = stablehlo.subtract %d4c1, %d4n1mu : tensor<32x512x7x7xf32>
    %d4n1sq = stablehlo.multiply %d4n1xc, %d4n1xc : tensor<32x512x7x7xf32>
    %d4n1vsr = stablehlo.reduce(%d4n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4n1vs = stablehlo.broadcast_in_dim %d4n1vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1vr = stablehlo.divide %d4n1vs, %d4n1nf : tensor<32x512x7x7xf32>
    %d4n1ve = stablehlo.add %d4n1vr, %d4n1ep : tensor<32x512x7x7xf32>
    %d4n1istd = stablehlo.rsqrt %d4n1ve : tensor<32x512x7x7xf32>
    %d4n1xh = stablehlo.multiply %d4n1xc, %d4n1istd : tensor<32x512x7x7xf32>
    %d4n1gb = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1btb = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n1gx = stablehlo.multiply %d4n1xh, %d4n1gb : tensor<32x512x7x7xf32>
    %d4n1 = stablehlo.add %d4n1gx, %d4n1btb : tensor<32x512x7x7xf32>
    %d4r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4r1 = stablehlo.maximum %d4n1, %d4r1z : tensor<32x512x7x7xf32>
    %d4c2c = stablehlo.convolution(%d4r1, %d4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4c2bb = stablehlo.broadcast_in_dim %d4b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4c2 = stablehlo.add %d4c2c, %d4c2bb : tensor<32x512x7x7xf32>
    %d4n2nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %d4n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4n2smr = stablehlo.reduce(%d4c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4n2sm = stablehlo.broadcast_in_dim %d4n2smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2mu = stablehlo.divide %d4n2sm, %d4n2nf : tensor<32x512x7x7xf32>
    %d4n2xc = stablehlo.subtract %d4c2, %d4n2mu : tensor<32x512x7x7xf32>
    %d4n2sq = stablehlo.multiply %d4n2xc, %d4n2xc : tensor<32x512x7x7xf32>
    %d4n2vsr = stablehlo.reduce(%d4n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4n2vs = stablehlo.broadcast_in_dim %d4n2vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2vr = stablehlo.divide %d4n2vs, %d4n2nf : tensor<32x512x7x7xf32>
    %d4n2ve = stablehlo.add %d4n2vr, %d4n2ep : tensor<32x512x7x7xf32>
    %d4n2istd = stablehlo.rsqrt %d4n2ve : tensor<32x512x7x7xf32>
    %d4n2xh = stablehlo.multiply %d4n2xc, %d4n2istd : tensor<32x512x7x7xf32>
    %d4n2gb = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2btb = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4n2gx = stablehlo.multiply %d4n2xh, %d4n2gb : tensor<32x512x7x7xf32>
    %d4n2 = stablehlo.add %d4n2gx, %d4n2btb : tensor<32x512x7x7xf32>
    %d4cpc = stablehlo.convolution(%s3b4o, %d4Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<512x256x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4cpbb = stablehlo.broadcast_in_dim %d4bp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4cp = stablehlo.add %d4cpc, %d4cpbb : tensor<32x512x7x7xf32>
    %d4npnf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %d4npep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %d4npsmr = stablehlo.reduce(%d4cp init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4npsm = stablehlo.broadcast_in_dim %d4npsmr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4npmu = stablehlo.divide %d4npsm, %d4npnf : tensor<32x512x7x7xf32>
    %d4npxc = stablehlo.subtract %d4cp, %d4npmu : tensor<32x512x7x7xf32>
    %d4npsq = stablehlo.multiply %d4npxc, %d4npxc : tensor<32x512x7x7xf32>
    %d4npvsr = stablehlo.reduce(%d4npsq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4npvs = stablehlo.broadcast_in_dim %d4npvsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4npvr = stablehlo.divide %d4npvs, %d4npnf : tensor<32x512x7x7xf32>
    %d4npve = stablehlo.add %d4npvr, %d4npep : tensor<32x512x7x7xf32>
    %d4npistd = stablehlo.rsqrt %d4npve : tensor<32x512x7x7xf32>
    %d4npxh = stablehlo.multiply %d4npxc, %d4npistd : tensor<32x512x7x7xf32>
    %d4npgb = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npbtb = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %d4npgx = stablehlo.multiply %d4npxh, %d4npgb : tensor<32x512x7x7xf32>
    %d4np = stablehlo.add %d4npgx, %d4npbtb : tensor<32x512x7x7xf32>
    %d4a = stablehlo.add %d4n2, %d4np : tensor<32x512x7x7xf32>
    %d4oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4o = stablehlo.maximum %d4a, %d4oz : tensor<32x512x7x7xf32>
    %s4b0c1c = stablehlo.convolution(%d4o, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c1bb = stablehlo.broadcast_in_dim %s4b0b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c1 = stablehlo.add %s4b0c1c, %s4b0c1bb : tensor<32x512x7x7xf32>
    %s4b0n1nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %s4b0n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n1smr = stablehlo.reduce(%s4b0c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0n1sm = stablehlo.broadcast_in_dim %s4b0n1smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1mu = stablehlo.divide %s4b0n1sm, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0n1xc = stablehlo.subtract %s4b0c1, %s4b0n1mu : tensor<32x512x7x7xf32>
    %s4b0n1sq = stablehlo.multiply %s4b0n1xc, %s4b0n1xc : tensor<32x512x7x7xf32>
    %s4b0n1vsr = stablehlo.reduce(%s4b0n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0n1vs = stablehlo.broadcast_in_dim %s4b0n1vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1vr = stablehlo.divide %s4b0n1vs, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0n1ve = stablehlo.add %s4b0n1vr, %s4b0n1ep : tensor<32x512x7x7xf32>
    %s4b0n1istd = stablehlo.rsqrt %s4b0n1ve : tensor<32x512x7x7xf32>
    %s4b0n1xh = stablehlo.multiply %s4b0n1xc, %s4b0n1istd : tensor<32x512x7x7xf32>
    %s4b0n1gb = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1btb = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n1gx = stablehlo.multiply %s4b0n1xh, %s4b0n1gb : tensor<32x512x7x7xf32>
    %s4b0n1 = stablehlo.add %s4b0n1gx, %s4b0n1btb : tensor<32x512x7x7xf32>
    %s4b0r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0r1 = stablehlo.maximum %s4b0n1, %s4b0r1z : tensor<32x512x7x7xf32>
    %s4b0c2c = stablehlo.convolution(%s4b0r1, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c2bb = stablehlo.broadcast_in_dim %s4b0b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0c2 = stablehlo.add %s4b0c2c, %s4b0c2bb : tensor<32x512x7x7xf32>
    %s4b0n2nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %s4b0n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b0n2smr = stablehlo.reduce(%s4b0c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0n2sm = stablehlo.broadcast_in_dim %s4b0n2smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2mu = stablehlo.divide %s4b0n2sm, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0n2xc = stablehlo.subtract %s4b0c2, %s4b0n2mu : tensor<32x512x7x7xf32>
    %s4b0n2sq = stablehlo.multiply %s4b0n2xc, %s4b0n2xc : tensor<32x512x7x7xf32>
    %s4b0n2vsr = stablehlo.reduce(%s4b0n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0n2vs = stablehlo.broadcast_in_dim %s4b0n2vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2vr = stablehlo.divide %s4b0n2vs, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0n2ve = stablehlo.add %s4b0n2vr, %s4b0n2ep : tensor<32x512x7x7xf32>
    %s4b0n2istd = stablehlo.rsqrt %s4b0n2ve : tensor<32x512x7x7xf32>
    %s4b0n2xh = stablehlo.multiply %s4b0n2xc, %s4b0n2istd : tensor<32x512x7x7xf32>
    %s4b0n2gb = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2btb = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0n2gx = stablehlo.multiply %s4b0n2xh, %s4b0n2gb : tensor<32x512x7x7xf32>
    %s4b0n2 = stablehlo.add %s4b0n2gx, %s4b0n2btb : tensor<32x512x7x7xf32>
    %s4b0a = stablehlo.add %s4b0n2, %d4o : tensor<32x512x7x7xf32>
    %s4b0oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0o = stablehlo.maximum %s4b0a, %s4b0oz : tensor<32x512x7x7xf32>
    %s4b1c1c = stablehlo.convolution(%s4b0o, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c1bb = stablehlo.broadcast_in_dim %s4b1b1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c1 = stablehlo.add %s4b1c1c, %s4b1c1bb : tensor<32x512x7x7xf32>
    %s4b1n1nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %s4b1n1ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n1smr = stablehlo.reduce(%s4b1c1 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1n1sm = stablehlo.broadcast_in_dim %s4b1n1smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1mu = stablehlo.divide %s4b1n1sm, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1n1xc = stablehlo.subtract %s4b1c1, %s4b1n1mu : tensor<32x512x7x7xf32>
    %s4b1n1sq = stablehlo.multiply %s4b1n1xc, %s4b1n1xc : tensor<32x512x7x7xf32>
    %s4b1n1vsr = stablehlo.reduce(%s4b1n1sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1n1vs = stablehlo.broadcast_in_dim %s4b1n1vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1vr = stablehlo.divide %s4b1n1vs, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1n1ve = stablehlo.add %s4b1n1vr, %s4b1n1ep : tensor<32x512x7x7xf32>
    %s4b1n1istd = stablehlo.rsqrt %s4b1n1ve : tensor<32x512x7x7xf32>
    %s4b1n1xh = stablehlo.multiply %s4b1n1xc, %s4b1n1istd : tensor<32x512x7x7xf32>
    %s4b1n1gb = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1btb = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n1gx = stablehlo.multiply %s4b1n1xh, %s4b1n1gb : tensor<32x512x7x7xf32>
    %s4b1n1 = stablehlo.add %s4b1n1gx, %s4b1n1btb : tensor<32x512x7x7xf32>
    %s4b1r1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1r1 = stablehlo.maximum %s4b1n1, %s4b1r1z : tensor<32x512x7x7xf32>
    %s4b1c2c = stablehlo.convolution(%s4b1r1, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c2bb = stablehlo.broadcast_in_dim %s4b1b2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1c2 = stablehlo.add %s4b1c2c, %s4b1c2bb : tensor<32x512x7x7xf32>
    %s4b1n2nf = stablehlo.constant dense<49.0> : tensor<32x512x7x7xf32>
    %s4b1n2ep = stablehlo.constant dense<1.0e-5> : tensor<32x512x7x7xf32>
    %s4b1n2smr = stablehlo.reduce(%s4b1c2 init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1n2sm = stablehlo.broadcast_in_dim %s4b1n2smr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2mu = stablehlo.divide %s4b1n2sm, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1n2xc = stablehlo.subtract %s4b1c2, %s4b1n2mu : tensor<32x512x7x7xf32>
    %s4b1n2sq = stablehlo.multiply %s4b1n2xc, %s4b1n2xc : tensor<32x512x7x7xf32>
    %s4b1n2vsr = stablehlo.reduce(%s4b1n2sq init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1n2vs = stablehlo.broadcast_in_dim %s4b1n2vsr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2vr = stablehlo.divide %s4b1n2vs, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1n2ve = stablehlo.add %s4b1n2vr, %s4b1n2ep : tensor<32x512x7x7xf32>
    %s4b1n2istd = stablehlo.rsqrt %s4b1n2ve : tensor<32x512x7x7xf32>
    %s4b1n2xh = stablehlo.multiply %s4b1n2xc, %s4b1n2istd : tensor<32x512x7x7xf32>
    %s4b1n2gb = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2btb = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1n2gx = stablehlo.multiply %s4b1n2xh, %s4b1n2gb : tensor<32x512x7x7xf32>
    %s4b1n2 = stablehlo.add %s4b1n2gx, %s4b1n2btb : tensor<32x512x7x7xf32>
    %s4b1a = stablehlo.add %s4b1n2, %s4b0o : tensor<32x512x7x7xf32>
    %s4b1oz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1o = stablehlo.maximum %s4b1a, %s4b1oz : tensor<32x512x7x7xf32>
    %gaps = stablehlo.reduce(%s4b1o init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %gapnf = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %gap = stablehlo.divide %gaps, %gapnf : tensor<32x512xf32>
    %ld = stablehlo.dot_general %gap, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<512x10xf32>) -> tensor<32x10xf32>
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
    %dgap = stablehlo.dot_general %dy, %Wd, contracting_dims = [1] x [1], precision = [DEFAULT, DEFAULT] : (tensor<32x10xf32>, tensor<512x10xf32>) -> tensor<32x512xf32>
    %dWd = stablehlo.dot_general %gap, %dy, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<32x512xf32>, tensor<32x10xf32>) -> tensor<512x10xf32>
    %dbd = stablehlo.reduce(%dy init: %sc) applies stablehlo.add across dimensions = [0] : (tensor<32x10xf32>, tensor<f32>) -> tensor<10xf32>
    %dgnf = stablehlo.constant dense<49.0> : tensor<32x512xf32>
    %dgs = stablehlo.divide %dgap, %dgnf : tensor<32x512xf32>
    %dgapin = stablehlo.broadcast_in_dim %dgs, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1dam = stablehlo.compare GT, %s4b1a, %s4b1daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b1da = stablehlo.select %s4b1dam, %dgapin, %s4b1daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b1dn2dxh = stablehlo.multiply %s4b1n2gb, %s4b1da : tensor<32x512x7x7xf32>
    %s4b1dn2sdxr = stablehlo.reduce(%s4b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1dn2sdx = stablehlo.broadcast_in_dim %s4b1dn2sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn2xd = stablehlo.multiply %s4b1n2xh, %s4b1dn2dxh : tensor<32x512x7x7xf32>
    %s4b1dn2sxdr = stablehlo.reduce(%s4b1dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1dn2sxd = stablehlo.broadcast_in_dim %s4b1dn2sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn2t1 = stablehlo.multiply %s4b1dn2dxh, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1dn2i1 = stablehlo.subtract %s4b1dn2t1, %s4b1dn2sdx : tensor<32x512x7x7xf32>
    %s4b1dn2xs = stablehlo.multiply %s4b1n2xh, %s4b1dn2sxd : tensor<32x512x7x7xf32>
    %s4b1dn2i2 = stablehlo.subtract %s4b1dn2i1, %s4b1dn2xs : tensor<32x512x7x7xf32>
    %s4b1dn2sN = stablehlo.divide %s4b1n2istd, %s4b1n2nf : tensor<32x512x7x7xf32>
    %s4b1dn2 = stablehlo.multiply %s4b1dn2sN, %s4b1dn2i2 : tensor<32x512x7x7xf32>
    %s4b1dn2dgp = stablehlo.multiply %s4b1da, %s4b1n2xh : tensor<32x512x7x7xf32>
    %s4b1dn2dg = stablehlo.reduce(%s4b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn2db = stablehlo.reduce(%s4b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dc2t = stablehlo.transpose %s4b1W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dc2r = stablehlo.reverse %s4b1dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b1dc2 = stablehlo.convolution(%s4b1dn2, %s4b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dW2xt = stablehlo.transpose %s4b1r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW2dt = stablehlo.transpose %s4b1dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW2raw = stablehlo.convolution(%s4b1dW2xt, %s4b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dW2 = stablehlo.transpose %s4b1dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1db2 = stablehlo.reduce(%s4b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b1dr1m = stablehlo.compare GT, %s4b1n1, %s4b1dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b1dr1 = stablehlo.select %s4b1dr1m, %s4b1dc2, %s4b1dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b1dn1dxh = stablehlo.multiply %s4b1n1gb, %s4b1dr1 : tensor<32x512x7x7xf32>
    %s4b1dn1sdxr = stablehlo.reduce(%s4b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1dn1sdx = stablehlo.broadcast_in_dim %s4b1dn1sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn1xd = stablehlo.multiply %s4b1n1xh, %s4b1dn1dxh : tensor<32x512x7x7xf32>
    %s4b1dn1sxdr = stablehlo.reduce(%s4b1dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b1dn1sxd = stablehlo.broadcast_in_dim %s4b1dn1sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dn1t1 = stablehlo.multiply %s4b1dn1dxh, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1dn1i1 = stablehlo.subtract %s4b1dn1t1, %s4b1dn1sdx : tensor<32x512x7x7xf32>
    %s4b1dn1xs = stablehlo.multiply %s4b1n1xh, %s4b1dn1sxd : tensor<32x512x7x7xf32>
    %s4b1dn1i2 = stablehlo.subtract %s4b1dn1i1, %s4b1dn1xs : tensor<32x512x7x7xf32>
    %s4b1dn1sN = stablehlo.divide %s4b1n1istd, %s4b1n1nf : tensor<32x512x7x7xf32>
    %s4b1dn1 = stablehlo.multiply %s4b1dn1sN, %s4b1dn1i2 : tensor<32x512x7x7xf32>
    %s4b1dn1dgp = stablehlo.multiply %s4b1dr1, %s4b1n1xh : tensor<32x512x7x7xf32>
    %s4b1dn1dg = stablehlo.reduce(%s4b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dn1db = stablehlo.reduce(%s4b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dc1t = stablehlo.transpose %s4b1W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dc1r = stablehlo.reverse %s4b1dc1t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b1dc1 = stablehlo.convolution(%s4b1dn1, %s4b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b1dW1xt = stablehlo.transpose %s4b0o, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW1dt = stablehlo.transpose %s4b1dn1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b1dW1raw = stablehlo.convolution(%s4b1dW1xt, %s4b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b1dW1 = stablehlo.transpose %s4b1dW1raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b1db1 = stablehlo.reduce(%s4b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b1dx = stablehlo.add %s4b1dc1, %s4b1da : tensor<32x512x7x7xf32>
    %s4b0daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0dam = stablehlo.compare GT, %s4b0a, %s4b0daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b0da = stablehlo.select %s4b0dam, %s4b1dx, %s4b0daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b0dn2dxh = stablehlo.multiply %s4b0n2gb, %s4b0da : tensor<32x512x7x7xf32>
    %s4b0dn2sdxr = stablehlo.reduce(%s4b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0dn2sdx = stablehlo.broadcast_in_dim %s4b0dn2sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn2xd = stablehlo.multiply %s4b0n2xh, %s4b0dn2dxh : tensor<32x512x7x7xf32>
    %s4b0dn2sxdr = stablehlo.reduce(%s4b0dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0dn2sxd = stablehlo.broadcast_in_dim %s4b0dn2sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn2t1 = stablehlo.multiply %s4b0dn2dxh, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0dn2i1 = stablehlo.subtract %s4b0dn2t1, %s4b0dn2sdx : tensor<32x512x7x7xf32>
    %s4b0dn2xs = stablehlo.multiply %s4b0n2xh, %s4b0dn2sxd : tensor<32x512x7x7xf32>
    %s4b0dn2i2 = stablehlo.subtract %s4b0dn2i1, %s4b0dn2xs : tensor<32x512x7x7xf32>
    %s4b0dn2sN = stablehlo.divide %s4b0n2istd, %s4b0n2nf : tensor<32x512x7x7xf32>
    %s4b0dn2 = stablehlo.multiply %s4b0dn2sN, %s4b0dn2i2 : tensor<32x512x7x7xf32>
    %s4b0dn2dgp = stablehlo.multiply %s4b0da, %s4b0n2xh : tensor<32x512x7x7xf32>
    %s4b0dn2dg = stablehlo.reduce(%s4b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn2db = stablehlo.reduce(%s4b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dc2t = stablehlo.transpose %s4b0W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dc2r = stablehlo.reverse %s4b0dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b0dc2 = stablehlo.convolution(%s4b0dn2, %s4b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dW2xt = stablehlo.transpose %s4b0r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW2dt = stablehlo.transpose %s4b0dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW2raw = stablehlo.convolution(%s4b0dW2xt, %s4b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dW2 = stablehlo.transpose %s4b0dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0db2 = stablehlo.reduce(%s4b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %s4b0dr1m = stablehlo.compare GT, %s4b0n1, %s4b0dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %s4b0dr1 = stablehlo.select %s4b0dr1m, %s4b0dc2, %s4b0dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %s4b0dn1dxh = stablehlo.multiply %s4b0n1gb, %s4b0dr1 : tensor<32x512x7x7xf32>
    %s4b0dn1sdxr = stablehlo.reduce(%s4b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0dn1sdx = stablehlo.broadcast_in_dim %s4b0dn1sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn1xd = stablehlo.multiply %s4b0n1xh, %s4b0dn1dxh : tensor<32x512x7x7xf32>
    %s4b0dn1sxdr = stablehlo.reduce(%s4b0dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %s4b0dn1sxd = stablehlo.broadcast_in_dim %s4b0dn1sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dn1t1 = stablehlo.multiply %s4b0dn1dxh, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0dn1i1 = stablehlo.subtract %s4b0dn1t1, %s4b0dn1sdx : tensor<32x512x7x7xf32>
    %s4b0dn1xs = stablehlo.multiply %s4b0n1xh, %s4b0dn1sxd : tensor<32x512x7x7xf32>
    %s4b0dn1i2 = stablehlo.subtract %s4b0dn1i1, %s4b0dn1xs : tensor<32x512x7x7xf32>
    %s4b0dn1sN = stablehlo.divide %s4b0n1istd, %s4b0n1nf : tensor<32x512x7x7xf32>
    %s4b0dn1 = stablehlo.multiply %s4b0dn1sN, %s4b0dn1i2 : tensor<32x512x7x7xf32>
    %s4b0dn1dgp = stablehlo.multiply %s4b0dr1, %s4b0n1xh : tensor<32x512x7x7xf32>
    %s4b0dn1dg = stablehlo.reduce(%s4b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dn1db = stablehlo.reduce(%s4b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dc1t = stablehlo.transpose %s4b0W1, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dc1r = stablehlo.reverse %s4b0dc1t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %s4b0dc1 = stablehlo.convolution(%s4b0dn1, %s4b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %s4b0dW1xt = stablehlo.transpose %d4o, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW1dt = stablehlo.transpose %s4b0dn1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %s4b0dW1raw = stablehlo.convolution(%s4b0dW1xt, %s4b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %s4b0dW1 = stablehlo.transpose %s4b0dW1raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %s4b0db1 = stablehlo.reduce(%s4b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %s4b0dx = stablehlo.add %s4b0dc1, %s4b0da : tensor<32x512x7x7xf32>
    %d4daz = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4dam = stablehlo.compare GT, %d4a, %d4daz : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %d4da = stablehlo.select %d4dam, %s4b0dx, %d4daz : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %d4dn2dxh = stablehlo.multiply %d4n2gb, %d4da : tensor<32x512x7x7xf32>
    %d4dn2sdxr = stablehlo.reduce(%d4dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dn2sdx = stablehlo.broadcast_in_dim %d4dn2sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn2xd = stablehlo.multiply %d4n2xh, %d4dn2dxh : tensor<32x512x7x7xf32>
    %d4dn2sxdr = stablehlo.reduce(%d4dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dn2sxd = stablehlo.broadcast_in_dim %d4dn2sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn2t1 = stablehlo.multiply %d4dn2dxh, %d4n2nf : tensor<32x512x7x7xf32>
    %d4dn2i1 = stablehlo.subtract %d4dn2t1, %d4dn2sdx : tensor<32x512x7x7xf32>
    %d4dn2xs = stablehlo.multiply %d4n2xh, %d4dn2sxd : tensor<32x512x7x7xf32>
    %d4dn2i2 = stablehlo.subtract %d4dn2i1, %d4dn2xs : tensor<32x512x7x7xf32>
    %d4dn2sN = stablehlo.divide %d4n2istd, %d4n2nf : tensor<32x512x7x7xf32>
    %d4dn2 = stablehlo.multiply %d4dn2sN, %d4dn2i2 : tensor<32x512x7x7xf32>
    %d4dn2dgp = stablehlo.multiply %d4da, %d4n2xh : tensor<32x512x7x7xf32>
    %d4dn2dg = stablehlo.reduce(%d4dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn2db = stablehlo.reduce(%d4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dc2t = stablehlo.transpose %d4W2, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %d4dc2r = stablehlo.reverse %d4dc2t, dims = [2, 3] : tensor<512x512x3x3xf32>
    %d4dc2 = stablehlo.convolution(%d4dn2, %d4dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<32x512x7x7xf32>
    %d4dW2xt = stablehlo.transpose %d4r1, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %d4dW2dt = stablehlo.transpose %d4dn2, dims = [1, 0, 2, 3] : (tensor<32x512x7x7xf32>) -> tensor<512x32x7x7xf32>
    %d4dW2raw = stablehlo.convolution(%d4dW2xt, %d4dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x32x7x7xf32>, tensor<512x32x7x7xf32>) -> tensor<512x512x3x3xf32>
    %d4dW2 = stablehlo.transpose %d4dW2raw, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %d4db2 = stablehlo.reduce(%d4dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dr1z = stablehlo.constant dense<0.0> : tensor<32x512x7x7xf32>
    %d4dr1m = stablehlo.compare GT, %d4n1, %d4dr1z : (tensor<32x512x7x7xf32>, tensor<32x512x7x7xf32>) -> tensor<32x512x7x7xi1>
    %d4dr1 = stablehlo.select %d4dr1m, %d4dc2, %d4dr1z : tensor<32x512x7x7xi1>, tensor<32x512x7x7xf32>
    %d4dn1dxh = stablehlo.multiply %d4n1gb, %d4dr1 : tensor<32x512x7x7xf32>
    %d4dn1sdxr = stablehlo.reduce(%d4dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dn1sdx = stablehlo.broadcast_in_dim %d4dn1sdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn1xd = stablehlo.multiply %d4n1xh, %d4dn1dxh : tensor<32x512x7x7xf32>
    %d4dn1sxdr = stablehlo.reduce(%d4dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dn1sxd = stablehlo.broadcast_in_dim %d4dn1sxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dn1t1 = stablehlo.multiply %d4dn1dxh, %d4n1nf : tensor<32x512x7x7xf32>
    %d4dn1i1 = stablehlo.subtract %d4dn1t1, %d4dn1sdx : tensor<32x512x7x7xf32>
    %d4dn1xs = stablehlo.multiply %d4n1xh, %d4dn1sxd : tensor<32x512x7x7xf32>
    %d4dn1i2 = stablehlo.subtract %d4dn1i1, %d4dn1xs : tensor<32x512x7x7xf32>
    %d4dn1sN = stablehlo.divide %d4n1istd, %d4n1nf : tensor<32x512x7x7xf32>
    %d4dn1 = stablehlo.multiply %d4dn1sN, %d4dn1i2 : tensor<32x512x7x7xf32>
    %d4dn1dgp = stablehlo.multiply %d4dr1, %d4n1xh : tensor<32x512x7x7xf32>
    %d4dn1dg = stablehlo.reduce(%d4dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dn1db = stablehlo.reduce(%d4dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dc1u = stablehlo.pad %d4dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dc1t = stablehlo.transpose %d4W1, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %d4dc1r = stablehlo.reverse %d4dc1t, dims = [2, 3] : tensor<256x512x3x3xf32>
    %d4dc1 = stablehlo.convolution(%d4dc1u, %d4dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d4dW1u = stablehlo.pad %d4dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dW1xt = stablehlo.transpose %s3b4o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d4dW1dt = stablehlo.transpose %d4dW1u, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %d4dW1raw = stablehlo.convolution(%d4dW1xt, %d4dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %d4dW1 = stablehlo.transpose %d4dW1raw, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %d4db1 = stablehlo.reduce(%d4dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpdxh = stablehlo.multiply %d4npgb, %d4da : tensor<32x512x7x7xf32>
    %d4dnpsdxr = stablehlo.reduce(%d4dnpdxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dnpsdx = stablehlo.broadcast_in_dim %d4dnpsdxr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dnpxd = stablehlo.multiply %d4npxh, %d4dnpdxh : tensor<32x512x7x7xf32>
    %d4dnpsxdr = stablehlo.reduce(%d4dnpxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512xf32>
    %d4dnpsxd = stablehlo.broadcast_in_dim %d4dnpsxdr, dims = [0, 1] : (tensor<32x512xf32>) -> tensor<32x512x7x7xf32>
    %d4dnpt1 = stablehlo.multiply %d4dnpdxh, %d4npnf : tensor<32x512x7x7xf32>
    %d4dnpi1 = stablehlo.subtract %d4dnpt1, %d4dnpsdx : tensor<32x512x7x7xf32>
    %d4dnpxs = stablehlo.multiply %d4npxh, %d4dnpsxd : tensor<32x512x7x7xf32>
    %d4dnpi2 = stablehlo.subtract %d4dnpi1, %d4dnpxs : tensor<32x512x7x7xf32>
    %d4dnpsN = stablehlo.divide %d4npistd, %d4npnf : tensor<32x512x7x7xf32>
    %d4dnp = stablehlo.multiply %d4dnpsN, %d4dnpi2 : tensor<32x512x7x7xf32>
    %d4dnpdgp = stablehlo.multiply %d4da, %d4npxh : tensor<32x512x7x7xf32>
    %d4dnpdg = stablehlo.reduce(%d4dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dnpdb = stablehlo.reduce(%d4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dcpu = stablehlo.pad %d4dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dcpt = stablehlo.transpose %d4Wp, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %d4dcpr = stablehlo.reverse %d4dcpt, dims = [2, 3] : tensor<256x512x3x3xf32>
    %d4dcp = stablehlo.convolution(%d4dcpu, %d4dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x512x14x14xf32>, tensor<256x512x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d4dWpu = stablehlo.pad %d4dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<32x512x14x14xf32>
    %d4dWpxt = stablehlo.transpose %s3b4o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d4dWpdt = stablehlo.transpose %d4dWpu, dims = [1, 0, 2, 3] : (tensor<32x512x14x14xf32>) -> tensor<512x32x14x14xf32>
    %d4dWpraw = stablehlo.convolution(%d4dWpxt, %d4dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<512x32x14x14xf32>) -> tensor<256x512x3x3xf32>
    %d4dWp = stablehlo.transpose %d4dWpraw, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %d4dbp = stablehlo.reduce(%d4dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %d4dx = stablehlo.add %d4dc1, %d4dcp : tensor<32x256x14x14xf32>
    %s3b4daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4dam = stablehlo.compare GT, %s3b4a, %s3b4daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b4da = stablehlo.select %s3b4dam, %d4dx, %s3b4daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b4dn2dxh = stablehlo.multiply %s3b4n2gb, %s3b4da : tensor<32x256x14x14xf32>
    %s3b4dn2sdxr = stablehlo.reduce(%s3b4dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4dn2sdx = stablehlo.broadcast_in_dim %s3b4dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn2xd = stablehlo.multiply %s3b4n2xh, %s3b4dn2dxh : tensor<32x256x14x14xf32>
    %s3b4dn2sxdr = stablehlo.reduce(%s3b4dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4dn2sxd = stablehlo.broadcast_in_dim %s3b4dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn2t1 = stablehlo.multiply %s3b4dn2dxh, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4dn2i1 = stablehlo.subtract %s3b4dn2t1, %s3b4dn2sdx : tensor<32x256x14x14xf32>
    %s3b4dn2xs = stablehlo.multiply %s3b4n2xh, %s3b4dn2sxd : tensor<32x256x14x14xf32>
    %s3b4dn2i2 = stablehlo.subtract %s3b4dn2i1, %s3b4dn2xs : tensor<32x256x14x14xf32>
    %s3b4dn2sN = stablehlo.divide %s3b4n2istd, %s3b4n2nf : tensor<32x256x14x14xf32>
    %s3b4dn2 = stablehlo.multiply %s3b4dn2sN, %s3b4dn2i2 : tensor<32x256x14x14xf32>
    %s3b4dn2dgp = stablehlo.multiply %s3b4da, %s3b4n2xh : tensor<32x256x14x14xf32>
    %s3b4dn2dg = stablehlo.reduce(%s3b4dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn2db = stablehlo.reduce(%s3b4da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dc2t = stablehlo.transpose %s3b4W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dc2r = stablehlo.reverse %s3b4dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b4dc2 = stablehlo.convolution(%s3b4dn2, %s3b4dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dW2xt = stablehlo.transpose %s3b4r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW2dt = stablehlo.transpose %s3b4dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW2raw = stablehlo.convolution(%s3b4dW2xt, %s3b4dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dW2 = stablehlo.transpose %s3b4dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4db2 = stablehlo.reduce(%s3b4dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b4dr1m = stablehlo.compare GT, %s3b4n1, %s3b4dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b4dr1 = stablehlo.select %s3b4dr1m, %s3b4dc2, %s3b4dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b4dn1dxh = stablehlo.multiply %s3b4n1gb, %s3b4dr1 : tensor<32x256x14x14xf32>
    %s3b4dn1sdxr = stablehlo.reduce(%s3b4dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4dn1sdx = stablehlo.broadcast_in_dim %s3b4dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn1xd = stablehlo.multiply %s3b4n1xh, %s3b4dn1dxh : tensor<32x256x14x14xf32>
    %s3b4dn1sxdr = stablehlo.reduce(%s3b4dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b4dn1sxd = stablehlo.broadcast_in_dim %s3b4dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dn1t1 = stablehlo.multiply %s3b4dn1dxh, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4dn1i1 = stablehlo.subtract %s3b4dn1t1, %s3b4dn1sdx : tensor<32x256x14x14xf32>
    %s3b4dn1xs = stablehlo.multiply %s3b4n1xh, %s3b4dn1sxd : tensor<32x256x14x14xf32>
    %s3b4dn1i2 = stablehlo.subtract %s3b4dn1i1, %s3b4dn1xs : tensor<32x256x14x14xf32>
    %s3b4dn1sN = stablehlo.divide %s3b4n1istd, %s3b4n1nf : tensor<32x256x14x14xf32>
    %s3b4dn1 = stablehlo.multiply %s3b4dn1sN, %s3b4dn1i2 : tensor<32x256x14x14xf32>
    %s3b4dn1dgp = stablehlo.multiply %s3b4dr1, %s3b4n1xh : tensor<32x256x14x14xf32>
    %s3b4dn1dg = stablehlo.reduce(%s3b4dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dn1db = stablehlo.reduce(%s3b4dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dc1t = stablehlo.transpose %s3b4W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dc1r = stablehlo.reverse %s3b4dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b4dc1 = stablehlo.convolution(%s3b4dn1, %s3b4dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b4dW1xt = stablehlo.transpose %s3b3o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW1dt = stablehlo.transpose %s3b4dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b4dW1raw = stablehlo.convolution(%s3b4dW1xt, %s3b4dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b4dW1 = stablehlo.transpose %s3b4dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b4db1 = stablehlo.reduce(%s3b4dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b4dx = stablehlo.add %s3b4dc1, %s3b4da : tensor<32x256x14x14xf32>
    %s3b3daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3dam = stablehlo.compare GT, %s3b3a, %s3b3daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b3da = stablehlo.select %s3b3dam, %s3b4dx, %s3b3daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b3dn2dxh = stablehlo.multiply %s3b3n2gb, %s3b3da : tensor<32x256x14x14xf32>
    %s3b3dn2sdxr = stablehlo.reduce(%s3b3dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3dn2sdx = stablehlo.broadcast_in_dim %s3b3dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn2xd = stablehlo.multiply %s3b3n2xh, %s3b3dn2dxh : tensor<32x256x14x14xf32>
    %s3b3dn2sxdr = stablehlo.reduce(%s3b3dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3dn2sxd = stablehlo.broadcast_in_dim %s3b3dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn2t1 = stablehlo.multiply %s3b3dn2dxh, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3dn2i1 = stablehlo.subtract %s3b3dn2t1, %s3b3dn2sdx : tensor<32x256x14x14xf32>
    %s3b3dn2xs = stablehlo.multiply %s3b3n2xh, %s3b3dn2sxd : tensor<32x256x14x14xf32>
    %s3b3dn2i2 = stablehlo.subtract %s3b3dn2i1, %s3b3dn2xs : tensor<32x256x14x14xf32>
    %s3b3dn2sN = stablehlo.divide %s3b3n2istd, %s3b3n2nf : tensor<32x256x14x14xf32>
    %s3b3dn2 = stablehlo.multiply %s3b3dn2sN, %s3b3dn2i2 : tensor<32x256x14x14xf32>
    %s3b3dn2dgp = stablehlo.multiply %s3b3da, %s3b3n2xh : tensor<32x256x14x14xf32>
    %s3b3dn2dg = stablehlo.reduce(%s3b3dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn2db = stablehlo.reduce(%s3b3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dc2t = stablehlo.transpose %s3b3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dc2r = stablehlo.reverse %s3b3dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b3dc2 = stablehlo.convolution(%s3b3dn2, %s3b3dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dW2xt = stablehlo.transpose %s3b3r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW2dt = stablehlo.transpose %s3b3dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW2raw = stablehlo.convolution(%s3b3dW2xt, %s3b3dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dW2 = stablehlo.transpose %s3b3dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3db2 = stablehlo.reduce(%s3b3dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b3dr1m = stablehlo.compare GT, %s3b3n1, %s3b3dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b3dr1 = stablehlo.select %s3b3dr1m, %s3b3dc2, %s3b3dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b3dn1dxh = stablehlo.multiply %s3b3n1gb, %s3b3dr1 : tensor<32x256x14x14xf32>
    %s3b3dn1sdxr = stablehlo.reduce(%s3b3dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3dn1sdx = stablehlo.broadcast_in_dim %s3b3dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn1xd = stablehlo.multiply %s3b3n1xh, %s3b3dn1dxh : tensor<32x256x14x14xf32>
    %s3b3dn1sxdr = stablehlo.reduce(%s3b3dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b3dn1sxd = stablehlo.broadcast_in_dim %s3b3dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dn1t1 = stablehlo.multiply %s3b3dn1dxh, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3dn1i1 = stablehlo.subtract %s3b3dn1t1, %s3b3dn1sdx : tensor<32x256x14x14xf32>
    %s3b3dn1xs = stablehlo.multiply %s3b3n1xh, %s3b3dn1sxd : tensor<32x256x14x14xf32>
    %s3b3dn1i2 = stablehlo.subtract %s3b3dn1i1, %s3b3dn1xs : tensor<32x256x14x14xf32>
    %s3b3dn1sN = stablehlo.divide %s3b3n1istd, %s3b3n1nf : tensor<32x256x14x14xf32>
    %s3b3dn1 = stablehlo.multiply %s3b3dn1sN, %s3b3dn1i2 : tensor<32x256x14x14xf32>
    %s3b3dn1dgp = stablehlo.multiply %s3b3dr1, %s3b3n1xh : tensor<32x256x14x14xf32>
    %s3b3dn1dg = stablehlo.reduce(%s3b3dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dn1db = stablehlo.reduce(%s3b3dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dc1t = stablehlo.transpose %s3b3W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dc1r = stablehlo.reverse %s3b3dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b3dc1 = stablehlo.convolution(%s3b3dn1, %s3b3dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b3dW1xt = stablehlo.transpose %s3b2o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW1dt = stablehlo.transpose %s3b3dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b3dW1raw = stablehlo.convolution(%s3b3dW1xt, %s3b3dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b3dW1 = stablehlo.transpose %s3b3dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b3db1 = stablehlo.reduce(%s3b3dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b3dx = stablehlo.add %s3b3dc1, %s3b3da : tensor<32x256x14x14xf32>
    %s3b2daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2dam = stablehlo.compare GT, %s3b2a, %s3b2daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b2da = stablehlo.select %s3b2dam, %s3b3dx, %s3b2daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b2dn2dxh = stablehlo.multiply %s3b2n2gb, %s3b2da : tensor<32x256x14x14xf32>
    %s3b2dn2sdxr = stablehlo.reduce(%s3b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2dn2sdx = stablehlo.broadcast_in_dim %s3b2dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn2xd = stablehlo.multiply %s3b2n2xh, %s3b2dn2dxh : tensor<32x256x14x14xf32>
    %s3b2dn2sxdr = stablehlo.reduce(%s3b2dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2dn2sxd = stablehlo.broadcast_in_dim %s3b2dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn2t1 = stablehlo.multiply %s3b2dn2dxh, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2dn2i1 = stablehlo.subtract %s3b2dn2t1, %s3b2dn2sdx : tensor<32x256x14x14xf32>
    %s3b2dn2xs = stablehlo.multiply %s3b2n2xh, %s3b2dn2sxd : tensor<32x256x14x14xf32>
    %s3b2dn2i2 = stablehlo.subtract %s3b2dn2i1, %s3b2dn2xs : tensor<32x256x14x14xf32>
    %s3b2dn2sN = stablehlo.divide %s3b2n2istd, %s3b2n2nf : tensor<32x256x14x14xf32>
    %s3b2dn2 = stablehlo.multiply %s3b2dn2sN, %s3b2dn2i2 : tensor<32x256x14x14xf32>
    %s3b2dn2dgp = stablehlo.multiply %s3b2da, %s3b2n2xh : tensor<32x256x14x14xf32>
    %s3b2dn2dg = stablehlo.reduce(%s3b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn2db = stablehlo.reduce(%s3b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dc2t = stablehlo.transpose %s3b2W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dc2r = stablehlo.reverse %s3b2dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b2dc2 = stablehlo.convolution(%s3b2dn2, %s3b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dW2xt = stablehlo.transpose %s3b2r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW2dt = stablehlo.transpose %s3b2dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW2raw = stablehlo.convolution(%s3b2dW2xt, %s3b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dW2 = stablehlo.transpose %s3b2dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2db2 = stablehlo.reduce(%s3b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b2dr1m = stablehlo.compare GT, %s3b2n1, %s3b2dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b2dr1 = stablehlo.select %s3b2dr1m, %s3b2dc2, %s3b2dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b2dn1dxh = stablehlo.multiply %s3b2n1gb, %s3b2dr1 : tensor<32x256x14x14xf32>
    %s3b2dn1sdxr = stablehlo.reduce(%s3b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2dn1sdx = stablehlo.broadcast_in_dim %s3b2dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn1xd = stablehlo.multiply %s3b2n1xh, %s3b2dn1dxh : tensor<32x256x14x14xf32>
    %s3b2dn1sxdr = stablehlo.reduce(%s3b2dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b2dn1sxd = stablehlo.broadcast_in_dim %s3b2dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dn1t1 = stablehlo.multiply %s3b2dn1dxh, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2dn1i1 = stablehlo.subtract %s3b2dn1t1, %s3b2dn1sdx : tensor<32x256x14x14xf32>
    %s3b2dn1xs = stablehlo.multiply %s3b2n1xh, %s3b2dn1sxd : tensor<32x256x14x14xf32>
    %s3b2dn1i2 = stablehlo.subtract %s3b2dn1i1, %s3b2dn1xs : tensor<32x256x14x14xf32>
    %s3b2dn1sN = stablehlo.divide %s3b2n1istd, %s3b2n1nf : tensor<32x256x14x14xf32>
    %s3b2dn1 = stablehlo.multiply %s3b2dn1sN, %s3b2dn1i2 : tensor<32x256x14x14xf32>
    %s3b2dn1dgp = stablehlo.multiply %s3b2dr1, %s3b2n1xh : tensor<32x256x14x14xf32>
    %s3b2dn1dg = stablehlo.reduce(%s3b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dn1db = stablehlo.reduce(%s3b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dc1t = stablehlo.transpose %s3b2W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dc1r = stablehlo.reverse %s3b2dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b2dc1 = stablehlo.convolution(%s3b2dn1, %s3b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b2dW1xt = stablehlo.transpose %s3b1o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW1dt = stablehlo.transpose %s3b2dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b2dW1raw = stablehlo.convolution(%s3b2dW1xt, %s3b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b2dW1 = stablehlo.transpose %s3b2dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b2db1 = stablehlo.reduce(%s3b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b2dx = stablehlo.add %s3b2dc1, %s3b2da : tensor<32x256x14x14xf32>
    %s3b1daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1dam = stablehlo.compare GT, %s3b1a, %s3b1daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b1da = stablehlo.select %s3b1dam, %s3b2dx, %s3b1daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b1dn2dxh = stablehlo.multiply %s3b1n2gb, %s3b1da : tensor<32x256x14x14xf32>
    %s3b1dn2sdxr = stablehlo.reduce(%s3b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1dn2sdx = stablehlo.broadcast_in_dim %s3b1dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn2xd = stablehlo.multiply %s3b1n2xh, %s3b1dn2dxh : tensor<32x256x14x14xf32>
    %s3b1dn2sxdr = stablehlo.reduce(%s3b1dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1dn2sxd = stablehlo.broadcast_in_dim %s3b1dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn2t1 = stablehlo.multiply %s3b1dn2dxh, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1dn2i1 = stablehlo.subtract %s3b1dn2t1, %s3b1dn2sdx : tensor<32x256x14x14xf32>
    %s3b1dn2xs = stablehlo.multiply %s3b1n2xh, %s3b1dn2sxd : tensor<32x256x14x14xf32>
    %s3b1dn2i2 = stablehlo.subtract %s3b1dn2i1, %s3b1dn2xs : tensor<32x256x14x14xf32>
    %s3b1dn2sN = stablehlo.divide %s3b1n2istd, %s3b1n2nf : tensor<32x256x14x14xf32>
    %s3b1dn2 = stablehlo.multiply %s3b1dn2sN, %s3b1dn2i2 : tensor<32x256x14x14xf32>
    %s3b1dn2dgp = stablehlo.multiply %s3b1da, %s3b1n2xh : tensor<32x256x14x14xf32>
    %s3b1dn2dg = stablehlo.reduce(%s3b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn2db = stablehlo.reduce(%s3b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dc2t = stablehlo.transpose %s3b1W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dc2r = stablehlo.reverse %s3b1dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b1dc2 = stablehlo.convolution(%s3b1dn2, %s3b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dW2xt = stablehlo.transpose %s3b1r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW2dt = stablehlo.transpose %s3b1dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW2raw = stablehlo.convolution(%s3b1dW2xt, %s3b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dW2 = stablehlo.transpose %s3b1dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1db2 = stablehlo.reduce(%s3b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b1dr1m = stablehlo.compare GT, %s3b1n1, %s3b1dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b1dr1 = stablehlo.select %s3b1dr1m, %s3b1dc2, %s3b1dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b1dn1dxh = stablehlo.multiply %s3b1n1gb, %s3b1dr1 : tensor<32x256x14x14xf32>
    %s3b1dn1sdxr = stablehlo.reduce(%s3b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1dn1sdx = stablehlo.broadcast_in_dim %s3b1dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn1xd = stablehlo.multiply %s3b1n1xh, %s3b1dn1dxh : tensor<32x256x14x14xf32>
    %s3b1dn1sxdr = stablehlo.reduce(%s3b1dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b1dn1sxd = stablehlo.broadcast_in_dim %s3b1dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dn1t1 = stablehlo.multiply %s3b1dn1dxh, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1dn1i1 = stablehlo.subtract %s3b1dn1t1, %s3b1dn1sdx : tensor<32x256x14x14xf32>
    %s3b1dn1xs = stablehlo.multiply %s3b1n1xh, %s3b1dn1sxd : tensor<32x256x14x14xf32>
    %s3b1dn1i2 = stablehlo.subtract %s3b1dn1i1, %s3b1dn1xs : tensor<32x256x14x14xf32>
    %s3b1dn1sN = stablehlo.divide %s3b1n1istd, %s3b1n1nf : tensor<32x256x14x14xf32>
    %s3b1dn1 = stablehlo.multiply %s3b1dn1sN, %s3b1dn1i2 : tensor<32x256x14x14xf32>
    %s3b1dn1dgp = stablehlo.multiply %s3b1dr1, %s3b1n1xh : tensor<32x256x14x14xf32>
    %s3b1dn1dg = stablehlo.reduce(%s3b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dn1db = stablehlo.reduce(%s3b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dc1t = stablehlo.transpose %s3b1W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dc1r = stablehlo.reverse %s3b1dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b1dc1 = stablehlo.convolution(%s3b1dn1, %s3b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b1dW1xt = stablehlo.transpose %s3b0o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW1dt = stablehlo.transpose %s3b1dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b1dW1raw = stablehlo.convolution(%s3b1dW1xt, %s3b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b1dW1 = stablehlo.transpose %s3b1dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b1db1 = stablehlo.reduce(%s3b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b1dx = stablehlo.add %s3b1dc1, %s3b1da : tensor<32x256x14x14xf32>
    %s3b0daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0dam = stablehlo.compare GT, %s3b0a, %s3b0daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b0da = stablehlo.select %s3b0dam, %s3b1dx, %s3b0daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b0dn2dxh = stablehlo.multiply %s3b0n2gb, %s3b0da : tensor<32x256x14x14xf32>
    %s3b0dn2sdxr = stablehlo.reduce(%s3b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0dn2sdx = stablehlo.broadcast_in_dim %s3b0dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn2xd = stablehlo.multiply %s3b0n2xh, %s3b0dn2dxh : tensor<32x256x14x14xf32>
    %s3b0dn2sxdr = stablehlo.reduce(%s3b0dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0dn2sxd = stablehlo.broadcast_in_dim %s3b0dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn2t1 = stablehlo.multiply %s3b0dn2dxh, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0dn2i1 = stablehlo.subtract %s3b0dn2t1, %s3b0dn2sdx : tensor<32x256x14x14xf32>
    %s3b0dn2xs = stablehlo.multiply %s3b0n2xh, %s3b0dn2sxd : tensor<32x256x14x14xf32>
    %s3b0dn2i2 = stablehlo.subtract %s3b0dn2i1, %s3b0dn2xs : tensor<32x256x14x14xf32>
    %s3b0dn2sN = stablehlo.divide %s3b0n2istd, %s3b0n2nf : tensor<32x256x14x14xf32>
    %s3b0dn2 = stablehlo.multiply %s3b0dn2sN, %s3b0dn2i2 : tensor<32x256x14x14xf32>
    %s3b0dn2dgp = stablehlo.multiply %s3b0da, %s3b0n2xh : tensor<32x256x14x14xf32>
    %s3b0dn2dg = stablehlo.reduce(%s3b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn2db = stablehlo.reduce(%s3b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dc2t = stablehlo.transpose %s3b0W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dc2r = stablehlo.reverse %s3b0dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b0dc2 = stablehlo.convolution(%s3b0dn2, %s3b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dW2xt = stablehlo.transpose %s3b0r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW2dt = stablehlo.transpose %s3b0dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW2raw = stablehlo.convolution(%s3b0dW2xt, %s3b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dW2 = stablehlo.transpose %s3b0dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0db2 = stablehlo.reduce(%s3b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %s3b0dr1m = stablehlo.compare GT, %s3b0n1, %s3b0dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %s3b0dr1 = stablehlo.select %s3b0dr1m, %s3b0dc2, %s3b0dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %s3b0dn1dxh = stablehlo.multiply %s3b0n1gb, %s3b0dr1 : tensor<32x256x14x14xf32>
    %s3b0dn1sdxr = stablehlo.reduce(%s3b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0dn1sdx = stablehlo.broadcast_in_dim %s3b0dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn1xd = stablehlo.multiply %s3b0n1xh, %s3b0dn1dxh : tensor<32x256x14x14xf32>
    %s3b0dn1sxdr = stablehlo.reduce(%s3b0dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %s3b0dn1sxd = stablehlo.broadcast_in_dim %s3b0dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dn1t1 = stablehlo.multiply %s3b0dn1dxh, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0dn1i1 = stablehlo.subtract %s3b0dn1t1, %s3b0dn1sdx : tensor<32x256x14x14xf32>
    %s3b0dn1xs = stablehlo.multiply %s3b0n1xh, %s3b0dn1sxd : tensor<32x256x14x14xf32>
    %s3b0dn1i2 = stablehlo.subtract %s3b0dn1i1, %s3b0dn1xs : tensor<32x256x14x14xf32>
    %s3b0dn1sN = stablehlo.divide %s3b0n1istd, %s3b0n1nf : tensor<32x256x14x14xf32>
    %s3b0dn1 = stablehlo.multiply %s3b0dn1sN, %s3b0dn1i2 : tensor<32x256x14x14xf32>
    %s3b0dn1dgp = stablehlo.multiply %s3b0dr1, %s3b0n1xh : tensor<32x256x14x14xf32>
    %s3b0dn1dg = stablehlo.reduce(%s3b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dn1db = stablehlo.reduce(%s3b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dc1t = stablehlo.transpose %s3b0W1, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dc1r = stablehlo.reverse %s3b0dc1t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %s3b0dc1 = stablehlo.convolution(%s3b0dn1, %s3b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %s3b0dW1xt = stablehlo.transpose %d3o, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW1dt = stablehlo.transpose %s3b0dn1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %s3b0dW1raw = stablehlo.convolution(%s3b0dW1xt, %s3b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %s3b0dW1 = stablehlo.transpose %s3b0dW1raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %s3b0db1 = stablehlo.reduce(%s3b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %s3b0dx = stablehlo.add %s3b0dc1, %s3b0da : tensor<32x256x14x14xf32>
    %d3daz = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3dam = stablehlo.compare GT, %d3a, %d3daz : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %d3da = stablehlo.select %d3dam, %s3b0dx, %d3daz : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %d3dn2dxh = stablehlo.multiply %d3n2gb, %d3da : tensor<32x256x14x14xf32>
    %d3dn2sdxr = stablehlo.reduce(%d3dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dn2sdx = stablehlo.broadcast_in_dim %d3dn2sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn2xd = stablehlo.multiply %d3n2xh, %d3dn2dxh : tensor<32x256x14x14xf32>
    %d3dn2sxdr = stablehlo.reduce(%d3dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dn2sxd = stablehlo.broadcast_in_dim %d3dn2sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn2t1 = stablehlo.multiply %d3dn2dxh, %d3n2nf : tensor<32x256x14x14xf32>
    %d3dn2i1 = stablehlo.subtract %d3dn2t1, %d3dn2sdx : tensor<32x256x14x14xf32>
    %d3dn2xs = stablehlo.multiply %d3n2xh, %d3dn2sxd : tensor<32x256x14x14xf32>
    %d3dn2i2 = stablehlo.subtract %d3dn2i1, %d3dn2xs : tensor<32x256x14x14xf32>
    %d3dn2sN = stablehlo.divide %d3n2istd, %d3n2nf : tensor<32x256x14x14xf32>
    %d3dn2 = stablehlo.multiply %d3dn2sN, %d3dn2i2 : tensor<32x256x14x14xf32>
    %d3dn2dgp = stablehlo.multiply %d3da, %d3n2xh : tensor<32x256x14x14xf32>
    %d3dn2dg = stablehlo.reduce(%d3dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn2db = stablehlo.reduce(%d3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dc2t = stablehlo.transpose %d3W2, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %d3dc2r = stablehlo.reverse %d3dc2t, dims = [2, 3] : tensor<256x256x3x3xf32>
    %d3dc2 = stablehlo.convolution(%d3dn2, %d3dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<32x256x14x14xf32>
    %d3dW2xt = stablehlo.transpose %d3r1, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d3dW2dt = stablehlo.transpose %d3dn2, dims = [1, 0, 2, 3] : (tensor<32x256x14x14xf32>) -> tensor<256x32x14x14xf32>
    %d3dW2raw = stablehlo.convolution(%d3dW2xt, %d3dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x32x14x14xf32>, tensor<256x32x14x14xf32>) -> tensor<256x256x3x3xf32>
    %d3dW2 = stablehlo.transpose %d3dW2raw, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %d3db2 = stablehlo.reduce(%d3dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dr1z = stablehlo.constant dense<0.0> : tensor<32x256x14x14xf32>
    %d3dr1m = stablehlo.compare GT, %d3n1, %d3dr1z : (tensor<32x256x14x14xf32>, tensor<32x256x14x14xf32>) -> tensor<32x256x14x14xi1>
    %d3dr1 = stablehlo.select %d3dr1m, %d3dc2, %d3dr1z : tensor<32x256x14x14xi1>, tensor<32x256x14x14xf32>
    %d3dn1dxh = stablehlo.multiply %d3n1gb, %d3dr1 : tensor<32x256x14x14xf32>
    %d3dn1sdxr = stablehlo.reduce(%d3dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dn1sdx = stablehlo.broadcast_in_dim %d3dn1sdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn1xd = stablehlo.multiply %d3n1xh, %d3dn1dxh : tensor<32x256x14x14xf32>
    %d3dn1sxdr = stablehlo.reduce(%d3dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dn1sxd = stablehlo.broadcast_in_dim %d3dn1sxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dn1t1 = stablehlo.multiply %d3dn1dxh, %d3n1nf : tensor<32x256x14x14xf32>
    %d3dn1i1 = stablehlo.subtract %d3dn1t1, %d3dn1sdx : tensor<32x256x14x14xf32>
    %d3dn1xs = stablehlo.multiply %d3n1xh, %d3dn1sxd : tensor<32x256x14x14xf32>
    %d3dn1i2 = stablehlo.subtract %d3dn1i1, %d3dn1xs : tensor<32x256x14x14xf32>
    %d3dn1sN = stablehlo.divide %d3n1istd, %d3n1nf : tensor<32x256x14x14xf32>
    %d3dn1 = stablehlo.multiply %d3dn1sN, %d3dn1i2 : tensor<32x256x14x14xf32>
    %d3dn1dgp = stablehlo.multiply %d3dr1, %d3n1xh : tensor<32x256x14x14xf32>
    %d3dn1dg = stablehlo.reduce(%d3dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dn1db = stablehlo.reduce(%d3dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dc1u = stablehlo.pad %d3dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dc1t = stablehlo.transpose %d3W1, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %d3dc1r = stablehlo.reverse %d3dc1t, dims = [2, 3] : tensor<128x256x3x3xf32>
    %d3dc1 = stablehlo.convolution(%d3dc1u, %d3dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d3dW1u = stablehlo.pad %d3dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dW1xt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d3dW1dt = stablehlo.transpose %d3dW1u, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %d3dW1raw = stablehlo.convolution(%d3dW1xt, %d3dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %d3dW1 = stablehlo.transpose %d3dW1raw, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %d3db1 = stablehlo.reduce(%d3dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpdxh = stablehlo.multiply %d3npgb, %d3da : tensor<32x256x14x14xf32>
    %d3dnpsdxr = stablehlo.reduce(%d3dnpdxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dnpsdx = stablehlo.broadcast_in_dim %d3dnpsdxr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dnpxd = stablehlo.multiply %d3npxh, %d3dnpdxh : tensor<32x256x14x14xf32>
    %d3dnpsxdr = stablehlo.reduce(%d3dnpxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256xf32>
    %d3dnpsxd = stablehlo.broadcast_in_dim %d3dnpsxdr, dims = [0, 1] : (tensor<32x256xf32>) -> tensor<32x256x14x14xf32>
    %d3dnpt1 = stablehlo.multiply %d3dnpdxh, %d3npnf : tensor<32x256x14x14xf32>
    %d3dnpi1 = stablehlo.subtract %d3dnpt1, %d3dnpsdx : tensor<32x256x14x14xf32>
    %d3dnpxs = stablehlo.multiply %d3npxh, %d3dnpsxd : tensor<32x256x14x14xf32>
    %d3dnpi2 = stablehlo.subtract %d3dnpi1, %d3dnpxs : tensor<32x256x14x14xf32>
    %d3dnpsN = stablehlo.divide %d3npistd, %d3npnf : tensor<32x256x14x14xf32>
    %d3dnp = stablehlo.multiply %d3dnpsN, %d3dnpi2 : tensor<32x256x14x14xf32>
    %d3dnpdgp = stablehlo.multiply %d3da, %d3npxh : tensor<32x256x14x14xf32>
    %d3dnpdg = stablehlo.reduce(%d3dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dnpdb = stablehlo.reduce(%d3da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dcpu = stablehlo.pad %d3dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dcpt = stablehlo.transpose %d3Wp, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %d3dcpr = stablehlo.reverse %d3dcpt, dims = [2, 3] : tensor<128x256x3x3xf32>
    %d3dcp = stablehlo.convolution(%d3dcpu, %d3dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x256x28x28xf32>, tensor<128x256x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d3dWpu = stablehlo.pad %d3dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<32x256x28x28xf32>
    %d3dWpxt = stablehlo.transpose %s2b2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d3dWpdt = stablehlo.transpose %d3dWpu, dims = [1, 0, 2, 3] : (tensor<32x256x28x28xf32>) -> tensor<256x32x28x28xf32>
    %d3dWpraw = stablehlo.convolution(%d3dWpxt, %d3dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<256x32x28x28xf32>) -> tensor<128x256x3x3xf32>
    %d3dWp = stablehlo.transpose %d3dWpraw, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %d3dbp = stablehlo.reduce(%d3dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %d3dx = stablehlo.add %d3dc1, %d3dcp : tensor<32x128x28x28xf32>
    %s2b2daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2dam = stablehlo.compare GT, %s2b2a, %s2b2daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b2da = stablehlo.select %s2b2dam, %d3dx, %s2b2daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b2dn2dxh = stablehlo.multiply %s2b2n2gb, %s2b2da : tensor<32x128x28x28xf32>
    %s2b2dn2sdxr = stablehlo.reduce(%s2b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2dn2sdx = stablehlo.broadcast_in_dim %s2b2dn2sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn2xd = stablehlo.multiply %s2b2n2xh, %s2b2dn2dxh : tensor<32x128x28x28xf32>
    %s2b2dn2sxdr = stablehlo.reduce(%s2b2dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2dn2sxd = stablehlo.broadcast_in_dim %s2b2dn2sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn2t1 = stablehlo.multiply %s2b2dn2dxh, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2dn2i1 = stablehlo.subtract %s2b2dn2t1, %s2b2dn2sdx : tensor<32x128x28x28xf32>
    %s2b2dn2xs = stablehlo.multiply %s2b2n2xh, %s2b2dn2sxd : tensor<32x128x28x28xf32>
    %s2b2dn2i2 = stablehlo.subtract %s2b2dn2i1, %s2b2dn2xs : tensor<32x128x28x28xf32>
    %s2b2dn2sN = stablehlo.divide %s2b2n2istd, %s2b2n2nf : tensor<32x128x28x28xf32>
    %s2b2dn2 = stablehlo.multiply %s2b2dn2sN, %s2b2dn2i2 : tensor<32x128x28x28xf32>
    %s2b2dn2dgp = stablehlo.multiply %s2b2da, %s2b2n2xh : tensor<32x128x28x28xf32>
    %s2b2dn2dg = stablehlo.reduce(%s2b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn2db = stablehlo.reduce(%s2b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dc2t = stablehlo.transpose %s2b2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dc2r = stablehlo.reverse %s2b2dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b2dc2 = stablehlo.convolution(%s2b2dn2, %s2b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dW2xt = stablehlo.transpose %s2b2r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW2dt = stablehlo.transpose %s2b2dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW2raw = stablehlo.convolution(%s2b2dW2xt, %s2b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dW2 = stablehlo.transpose %s2b2dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2db2 = stablehlo.reduce(%s2b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b2dr1m = stablehlo.compare GT, %s2b2n1, %s2b2dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b2dr1 = stablehlo.select %s2b2dr1m, %s2b2dc2, %s2b2dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b2dn1dxh = stablehlo.multiply %s2b2n1gb, %s2b2dr1 : tensor<32x128x28x28xf32>
    %s2b2dn1sdxr = stablehlo.reduce(%s2b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2dn1sdx = stablehlo.broadcast_in_dim %s2b2dn1sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn1xd = stablehlo.multiply %s2b2n1xh, %s2b2dn1dxh : tensor<32x128x28x28xf32>
    %s2b2dn1sxdr = stablehlo.reduce(%s2b2dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b2dn1sxd = stablehlo.broadcast_in_dim %s2b2dn1sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dn1t1 = stablehlo.multiply %s2b2dn1dxh, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2dn1i1 = stablehlo.subtract %s2b2dn1t1, %s2b2dn1sdx : tensor<32x128x28x28xf32>
    %s2b2dn1xs = stablehlo.multiply %s2b2n1xh, %s2b2dn1sxd : tensor<32x128x28x28xf32>
    %s2b2dn1i2 = stablehlo.subtract %s2b2dn1i1, %s2b2dn1xs : tensor<32x128x28x28xf32>
    %s2b2dn1sN = stablehlo.divide %s2b2n1istd, %s2b2n1nf : tensor<32x128x28x28xf32>
    %s2b2dn1 = stablehlo.multiply %s2b2dn1sN, %s2b2dn1i2 : tensor<32x128x28x28xf32>
    %s2b2dn1dgp = stablehlo.multiply %s2b2dr1, %s2b2n1xh : tensor<32x128x28x28xf32>
    %s2b2dn1dg = stablehlo.reduce(%s2b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dn1db = stablehlo.reduce(%s2b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dc1t = stablehlo.transpose %s2b2W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dc1r = stablehlo.reverse %s2b2dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b2dc1 = stablehlo.convolution(%s2b2dn1, %s2b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b2dW1xt = stablehlo.transpose %s2b1o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW1dt = stablehlo.transpose %s2b2dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b2dW1raw = stablehlo.convolution(%s2b2dW1xt, %s2b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b2dW1 = stablehlo.transpose %s2b2dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b2db1 = stablehlo.reduce(%s2b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b2dx = stablehlo.add %s2b2dc1, %s2b2da : tensor<32x128x28x28xf32>
    %s2b1daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1dam = stablehlo.compare GT, %s2b1a, %s2b1daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b1da = stablehlo.select %s2b1dam, %s2b2dx, %s2b1daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b1dn2dxh = stablehlo.multiply %s2b1n2gb, %s2b1da : tensor<32x128x28x28xf32>
    %s2b1dn2sdxr = stablehlo.reduce(%s2b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1dn2sdx = stablehlo.broadcast_in_dim %s2b1dn2sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn2xd = stablehlo.multiply %s2b1n2xh, %s2b1dn2dxh : tensor<32x128x28x28xf32>
    %s2b1dn2sxdr = stablehlo.reduce(%s2b1dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1dn2sxd = stablehlo.broadcast_in_dim %s2b1dn2sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn2t1 = stablehlo.multiply %s2b1dn2dxh, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1dn2i1 = stablehlo.subtract %s2b1dn2t1, %s2b1dn2sdx : tensor<32x128x28x28xf32>
    %s2b1dn2xs = stablehlo.multiply %s2b1n2xh, %s2b1dn2sxd : tensor<32x128x28x28xf32>
    %s2b1dn2i2 = stablehlo.subtract %s2b1dn2i1, %s2b1dn2xs : tensor<32x128x28x28xf32>
    %s2b1dn2sN = stablehlo.divide %s2b1n2istd, %s2b1n2nf : tensor<32x128x28x28xf32>
    %s2b1dn2 = stablehlo.multiply %s2b1dn2sN, %s2b1dn2i2 : tensor<32x128x28x28xf32>
    %s2b1dn2dgp = stablehlo.multiply %s2b1da, %s2b1n2xh : tensor<32x128x28x28xf32>
    %s2b1dn2dg = stablehlo.reduce(%s2b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn2db = stablehlo.reduce(%s2b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dc2t = stablehlo.transpose %s2b1W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dc2r = stablehlo.reverse %s2b1dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b1dc2 = stablehlo.convolution(%s2b1dn2, %s2b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dW2xt = stablehlo.transpose %s2b1r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW2dt = stablehlo.transpose %s2b1dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW2raw = stablehlo.convolution(%s2b1dW2xt, %s2b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dW2 = stablehlo.transpose %s2b1dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1db2 = stablehlo.reduce(%s2b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b1dr1m = stablehlo.compare GT, %s2b1n1, %s2b1dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b1dr1 = stablehlo.select %s2b1dr1m, %s2b1dc2, %s2b1dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b1dn1dxh = stablehlo.multiply %s2b1n1gb, %s2b1dr1 : tensor<32x128x28x28xf32>
    %s2b1dn1sdxr = stablehlo.reduce(%s2b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1dn1sdx = stablehlo.broadcast_in_dim %s2b1dn1sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn1xd = stablehlo.multiply %s2b1n1xh, %s2b1dn1dxh : tensor<32x128x28x28xf32>
    %s2b1dn1sxdr = stablehlo.reduce(%s2b1dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b1dn1sxd = stablehlo.broadcast_in_dim %s2b1dn1sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dn1t1 = stablehlo.multiply %s2b1dn1dxh, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1dn1i1 = stablehlo.subtract %s2b1dn1t1, %s2b1dn1sdx : tensor<32x128x28x28xf32>
    %s2b1dn1xs = stablehlo.multiply %s2b1n1xh, %s2b1dn1sxd : tensor<32x128x28x28xf32>
    %s2b1dn1i2 = stablehlo.subtract %s2b1dn1i1, %s2b1dn1xs : tensor<32x128x28x28xf32>
    %s2b1dn1sN = stablehlo.divide %s2b1n1istd, %s2b1n1nf : tensor<32x128x28x28xf32>
    %s2b1dn1 = stablehlo.multiply %s2b1dn1sN, %s2b1dn1i2 : tensor<32x128x28x28xf32>
    %s2b1dn1dgp = stablehlo.multiply %s2b1dr1, %s2b1n1xh : tensor<32x128x28x28xf32>
    %s2b1dn1dg = stablehlo.reduce(%s2b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dn1db = stablehlo.reduce(%s2b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dc1t = stablehlo.transpose %s2b1W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dc1r = stablehlo.reverse %s2b1dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b1dc1 = stablehlo.convolution(%s2b1dn1, %s2b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b1dW1xt = stablehlo.transpose %s2b0o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW1dt = stablehlo.transpose %s2b1dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b1dW1raw = stablehlo.convolution(%s2b1dW1xt, %s2b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b1dW1 = stablehlo.transpose %s2b1dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b1db1 = stablehlo.reduce(%s2b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b1dx = stablehlo.add %s2b1dc1, %s2b1da : tensor<32x128x28x28xf32>
    %s2b0daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0dam = stablehlo.compare GT, %s2b0a, %s2b0daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b0da = stablehlo.select %s2b0dam, %s2b1dx, %s2b0daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b0dn2dxh = stablehlo.multiply %s2b0n2gb, %s2b0da : tensor<32x128x28x28xf32>
    %s2b0dn2sdxr = stablehlo.reduce(%s2b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0dn2sdx = stablehlo.broadcast_in_dim %s2b0dn2sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn2xd = stablehlo.multiply %s2b0n2xh, %s2b0dn2dxh : tensor<32x128x28x28xf32>
    %s2b0dn2sxdr = stablehlo.reduce(%s2b0dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0dn2sxd = stablehlo.broadcast_in_dim %s2b0dn2sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn2t1 = stablehlo.multiply %s2b0dn2dxh, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0dn2i1 = stablehlo.subtract %s2b0dn2t1, %s2b0dn2sdx : tensor<32x128x28x28xf32>
    %s2b0dn2xs = stablehlo.multiply %s2b0n2xh, %s2b0dn2sxd : tensor<32x128x28x28xf32>
    %s2b0dn2i2 = stablehlo.subtract %s2b0dn2i1, %s2b0dn2xs : tensor<32x128x28x28xf32>
    %s2b0dn2sN = stablehlo.divide %s2b0n2istd, %s2b0n2nf : tensor<32x128x28x28xf32>
    %s2b0dn2 = stablehlo.multiply %s2b0dn2sN, %s2b0dn2i2 : tensor<32x128x28x28xf32>
    %s2b0dn2dgp = stablehlo.multiply %s2b0da, %s2b0n2xh : tensor<32x128x28x28xf32>
    %s2b0dn2dg = stablehlo.reduce(%s2b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn2db = stablehlo.reduce(%s2b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dc2t = stablehlo.transpose %s2b0W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dc2r = stablehlo.reverse %s2b0dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b0dc2 = stablehlo.convolution(%s2b0dn2, %s2b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dW2xt = stablehlo.transpose %s2b0r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW2dt = stablehlo.transpose %s2b0dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW2raw = stablehlo.convolution(%s2b0dW2xt, %s2b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dW2 = stablehlo.transpose %s2b0dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0db2 = stablehlo.reduce(%s2b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %s2b0dr1m = stablehlo.compare GT, %s2b0n1, %s2b0dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %s2b0dr1 = stablehlo.select %s2b0dr1m, %s2b0dc2, %s2b0dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %s2b0dn1dxh = stablehlo.multiply %s2b0n1gb, %s2b0dr1 : tensor<32x128x28x28xf32>
    %s2b0dn1sdxr = stablehlo.reduce(%s2b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0dn1sdx = stablehlo.broadcast_in_dim %s2b0dn1sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn1xd = stablehlo.multiply %s2b0n1xh, %s2b0dn1dxh : tensor<32x128x28x28xf32>
    %s2b0dn1sxdr = stablehlo.reduce(%s2b0dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %s2b0dn1sxd = stablehlo.broadcast_in_dim %s2b0dn1sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dn1t1 = stablehlo.multiply %s2b0dn1dxh, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0dn1i1 = stablehlo.subtract %s2b0dn1t1, %s2b0dn1sdx : tensor<32x128x28x28xf32>
    %s2b0dn1xs = stablehlo.multiply %s2b0n1xh, %s2b0dn1sxd : tensor<32x128x28x28xf32>
    %s2b0dn1i2 = stablehlo.subtract %s2b0dn1i1, %s2b0dn1xs : tensor<32x128x28x28xf32>
    %s2b0dn1sN = stablehlo.divide %s2b0n1istd, %s2b0n1nf : tensor<32x128x28x28xf32>
    %s2b0dn1 = stablehlo.multiply %s2b0dn1sN, %s2b0dn1i2 : tensor<32x128x28x28xf32>
    %s2b0dn1dgp = stablehlo.multiply %s2b0dr1, %s2b0n1xh : tensor<32x128x28x28xf32>
    %s2b0dn1dg = stablehlo.reduce(%s2b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dn1db = stablehlo.reduce(%s2b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dc1t = stablehlo.transpose %s2b0W1, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dc1r = stablehlo.reverse %s2b0dc1t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %s2b0dc1 = stablehlo.convolution(%s2b0dn1, %s2b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %s2b0dW1xt = stablehlo.transpose %d2o, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW1dt = stablehlo.transpose %s2b0dn1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %s2b0dW1raw = stablehlo.convolution(%s2b0dW1xt, %s2b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %s2b0dW1 = stablehlo.transpose %s2b0dW1raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %s2b0db1 = stablehlo.reduce(%s2b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %s2b0dx = stablehlo.add %s2b0dc1, %s2b0da : tensor<32x128x28x28xf32>
    %d2daz = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2dam = stablehlo.compare GT, %d2a, %d2daz : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %d2da = stablehlo.select %d2dam, %s2b0dx, %d2daz : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %d2dn2dxh = stablehlo.multiply %d2n2gb, %d2da : tensor<32x128x28x28xf32>
    %d2dn2sdxr = stablehlo.reduce(%d2dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dn2sdx = stablehlo.broadcast_in_dim %d2dn2sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn2xd = stablehlo.multiply %d2n2xh, %d2dn2dxh : tensor<32x128x28x28xf32>
    %d2dn2sxdr = stablehlo.reduce(%d2dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dn2sxd = stablehlo.broadcast_in_dim %d2dn2sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn2t1 = stablehlo.multiply %d2dn2dxh, %d2n2nf : tensor<32x128x28x28xf32>
    %d2dn2i1 = stablehlo.subtract %d2dn2t1, %d2dn2sdx : tensor<32x128x28x28xf32>
    %d2dn2xs = stablehlo.multiply %d2n2xh, %d2dn2sxd : tensor<32x128x28x28xf32>
    %d2dn2i2 = stablehlo.subtract %d2dn2i1, %d2dn2xs : tensor<32x128x28x28xf32>
    %d2dn2sN = stablehlo.divide %d2n2istd, %d2n2nf : tensor<32x128x28x28xf32>
    %d2dn2 = stablehlo.multiply %d2dn2sN, %d2dn2i2 : tensor<32x128x28x28xf32>
    %d2dn2dgp = stablehlo.multiply %d2da, %d2n2xh : tensor<32x128x28x28xf32>
    %d2dn2dg = stablehlo.reduce(%d2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn2db = stablehlo.reduce(%d2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dc2t = stablehlo.transpose %d2W2, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %d2dc2r = stablehlo.reverse %d2dc2t, dims = [2, 3] : tensor<128x128x3x3xf32>
    %d2dc2 = stablehlo.convolution(%d2dn2, %d2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<32x128x28x28xf32>
    %d2dW2xt = stablehlo.transpose %d2r1, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d2dW2dt = stablehlo.transpose %d2dn2, dims = [1, 0, 2, 3] : (tensor<32x128x28x28xf32>) -> tensor<128x32x28x28xf32>
    %d2dW2raw = stablehlo.convolution(%d2dW2xt, %d2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x32x28x28xf32>, tensor<128x32x28x28xf32>) -> tensor<128x128x3x3xf32>
    %d2dW2 = stablehlo.transpose %d2dW2raw, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %d2db2 = stablehlo.reduce(%d2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dr1z = stablehlo.constant dense<0.0> : tensor<32x128x28x28xf32>
    %d2dr1m = stablehlo.compare GT, %d2n1, %d2dr1z : (tensor<32x128x28x28xf32>, tensor<32x128x28x28xf32>) -> tensor<32x128x28x28xi1>
    %d2dr1 = stablehlo.select %d2dr1m, %d2dc2, %d2dr1z : tensor<32x128x28x28xi1>, tensor<32x128x28x28xf32>
    %d2dn1dxh = stablehlo.multiply %d2n1gb, %d2dr1 : tensor<32x128x28x28xf32>
    %d2dn1sdxr = stablehlo.reduce(%d2dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dn1sdx = stablehlo.broadcast_in_dim %d2dn1sdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn1xd = stablehlo.multiply %d2n1xh, %d2dn1dxh : tensor<32x128x28x28xf32>
    %d2dn1sxdr = stablehlo.reduce(%d2dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dn1sxd = stablehlo.broadcast_in_dim %d2dn1sxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dn1t1 = stablehlo.multiply %d2dn1dxh, %d2n1nf : tensor<32x128x28x28xf32>
    %d2dn1i1 = stablehlo.subtract %d2dn1t1, %d2dn1sdx : tensor<32x128x28x28xf32>
    %d2dn1xs = stablehlo.multiply %d2n1xh, %d2dn1sxd : tensor<32x128x28x28xf32>
    %d2dn1i2 = stablehlo.subtract %d2dn1i1, %d2dn1xs : tensor<32x128x28x28xf32>
    %d2dn1sN = stablehlo.divide %d2n1istd, %d2n1nf : tensor<32x128x28x28xf32>
    %d2dn1 = stablehlo.multiply %d2dn1sN, %d2dn1i2 : tensor<32x128x28x28xf32>
    %d2dn1dgp = stablehlo.multiply %d2dr1, %d2n1xh : tensor<32x128x28x28xf32>
    %d2dn1dg = stablehlo.reduce(%d2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dn1db = stablehlo.reduce(%d2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dc1u = stablehlo.pad %d2dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dc1t = stablehlo.transpose %d2W1, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %d2dc1r = stablehlo.reverse %d2dc1t, dims = [2, 3] : tensor<64x128x3x3xf32>
    %d2dc1 = stablehlo.convolution(%d2dc1u, %d2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %d2dW1u = stablehlo.pad %d2dn1, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dW1xt = stablehlo.transpose %s1b2o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %d2dW1dt = stablehlo.transpose %d2dW1u, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %d2dW1raw = stablehlo.convolution(%d2dW1xt, %d2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %d2dW1 = stablehlo.transpose %d2dW1raw, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %d2db1 = stablehlo.reduce(%d2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpdxh = stablehlo.multiply %d2npgb, %d2da : tensor<32x128x28x28xf32>
    %d2dnpsdxr = stablehlo.reduce(%d2dnpdxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dnpsdx = stablehlo.broadcast_in_dim %d2dnpsdxr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dnpxd = stablehlo.multiply %d2npxh, %d2dnpdxh : tensor<32x128x28x28xf32>
    %d2dnpsxdr = stablehlo.reduce(%d2dnpxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128xf32>
    %d2dnpsxd = stablehlo.broadcast_in_dim %d2dnpsxdr, dims = [0, 1] : (tensor<32x128xf32>) -> tensor<32x128x28x28xf32>
    %d2dnpt1 = stablehlo.multiply %d2dnpdxh, %d2npnf : tensor<32x128x28x28xf32>
    %d2dnpi1 = stablehlo.subtract %d2dnpt1, %d2dnpsdx : tensor<32x128x28x28xf32>
    %d2dnpxs = stablehlo.multiply %d2npxh, %d2dnpsxd : tensor<32x128x28x28xf32>
    %d2dnpi2 = stablehlo.subtract %d2dnpi1, %d2dnpxs : tensor<32x128x28x28xf32>
    %d2dnpsN = stablehlo.divide %d2npistd, %d2npnf : tensor<32x128x28x28xf32>
    %d2dnp = stablehlo.multiply %d2dnpsN, %d2dnpi2 : tensor<32x128x28x28xf32>
    %d2dnpdgp = stablehlo.multiply %d2da, %d2npxh : tensor<32x128x28x28xf32>
    %d2dnpdg = stablehlo.reduce(%d2dnpdgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dnpdb = stablehlo.reduce(%d2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dcpu = stablehlo.pad %d2dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dcpt = stablehlo.transpose %d2Wp, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %d2dcpr = stablehlo.reverse %d2dcpt, dims = [2, 3] : tensor<64x128x3x3xf32>
    %d2dcp = stablehlo.convolution(%d2dcpu, %d2dcpr)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x128x56x56xf32>, tensor<64x128x3x3xf32>) -> tensor<32x64x56x56xf32>
    %d2dWpu = stablehlo.pad %d2dnp, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<32x128x56x56xf32>
    %d2dWpxt = stablehlo.transpose %s1b2o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %d2dWpdt = stablehlo.transpose %d2dWpu, dims = [1, 0, 2, 3] : (tensor<32x128x56x56xf32>) -> tensor<128x32x56x56xf32>
    %d2dWpraw = stablehlo.convolution(%d2dWpxt, %d2dWpdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<128x32x56x56xf32>) -> tensor<64x128x3x3xf32>
    %d2dWp = stablehlo.transpose %d2dWpraw, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %d2dbp = stablehlo.reduce(%d2dnp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %d2dx = stablehlo.add %d2dc1, %d2dcp : tensor<32x64x56x56xf32>
    %s1b2daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2dam = stablehlo.compare GT, %s1b2a, %s1b2daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b2da = stablehlo.select %s1b2dam, %d2dx, %s1b2daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b2dn2dxh = stablehlo.multiply %s1b2n2gb, %s1b2da : tensor<32x64x56x56xf32>
    %s1b2dn2sdxr = stablehlo.reduce(%s1b2dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2dn2sdx = stablehlo.broadcast_in_dim %s1b2dn2sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn2xd = stablehlo.multiply %s1b2n2xh, %s1b2dn2dxh : tensor<32x64x56x56xf32>
    %s1b2dn2sxdr = stablehlo.reduce(%s1b2dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2dn2sxd = stablehlo.broadcast_in_dim %s1b2dn2sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn2t1 = stablehlo.multiply %s1b2dn2dxh, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2dn2i1 = stablehlo.subtract %s1b2dn2t1, %s1b2dn2sdx : tensor<32x64x56x56xf32>
    %s1b2dn2xs = stablehlo.multiply %s1b2n2xh, %s1b2dn2sxd : tensor<32x64x56x56xf32>
    %s1b2dn2i2 = stablehlo.subtract %s1b2dn2i1, %s1b2dn2xs : tensor<32x64x56x56xf32>
    %s1b2dn2sN = stablehlo.divide %s1b2n2istd, %s1b2n2nf : tensor<32x64x56x56xf32>
    %s1b2dn2 = stablehlo.multiply %s1b2dn2sN, %s1b2dn2i2 : tensor<32x64x56x56xf32>
    %s1b2dn2dgp = stablehlo.multiply %s1b2da, %s1b2n2xh : tensor<32x64x56x56xf32>
    %s1b2dn2dg = stablehlo.reduce(%s1b2dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn2db = stablehlo.reduce(%s1b2da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dc2t = stablehlo.transpose %s1b2W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dc2r = stablehlo.reverse %s1b2dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b2dc2 = stablehlo.convolution(%s1b2dn2, %s1b2dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dW2xt = stablehlo.transpose %s1b2r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW2dt = stablehlo.transpose %s1b2dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW2raw = stablehlo.convolution(%s1b2dW2xt, %s1b2dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dW2 = stablehlo.transpose %s1b2dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2db2 = stablehlo.reduce(%s1b2dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b2dr1m = stablehlo.compare GT, %s1b2n1, %s1b2dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b2dr1 = stablehlo.select %s1b2dr1m, %s1b2dc2, %s1b2dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b2dn1dxh = stablehlo.multiply %s1b2n1gb, %s1b2dr1 : tensor<32x64x56x56xf32>
    %s1b2dn1sdxr = stablehlo.reduce(%s1b2dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2dn1sdx = stablehlo.broadcast_in_dim %s1b2dn1sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn1xd = stablehlo.multiply %s1b2n1xh, %s1b2dn1dxh : tensor<32x64x56x56xf32>
    %s1b2dn1sxdr = stablehlo.reduce(%s1b2dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b2dn1sxd = stablehlo.broadcast_in_dim %s1b2dn1sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dn1t1 = stablehlo.multiply %s1b2dn1dxh, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2dn1i1 = stablehlo.subtract %s1b2dn1t1, %s1b2dn1sdx : tensor<32x64x56x56xf32>
    %s1b2dn1xs = stablehlo.multiply %s1b2n1xh, %s1b2dn1sxd : tensor<32x64x56x56xf32>
    %s1b2dn1i2 = stablehlo.subtract %s1b2dn1i1, %s1b2dn1xs : tensor<32x64x56x56xf32>
    %s1b2dn1sN = stablehlo.divide %s1b2n1istd, %s1b2n1nf : tensor<32x64x56x56xf32>
    %s1b2dn1 = stablehlo.multiply %s1b2dn1sN, %s1b2dn1i2 : tensor<32x64x56x56xf32>
    %s1b2dn1dgp = stablehlo.multiply %s1b2dr1, %s1b2n1xh : tensor<32x64x56x56xf32>
    %s1b2dn1dg = stablehlo.reduce(%s1b2dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dn1db = stablehlo.reduce(%s1b2dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dc1t = stablehlo.transpose %s1b2W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dc1r = stablehlo.reverse %s1b2dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b2dc1 = stablehlo.convolution(%s1b2dn1, %s1b2dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b2dW1xt = stablehlo.transpose %s1b1o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW1dt = stablehlo.transpose %s1b2dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b2dW1raw = stablehlo.convolution(%s1b2dW1xt, %s1b2dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b2dW1 = stablehlo.transpose %s1b2dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b2db1 = stablehlo.reduce(%s1b2dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b2dx = stablehlo.add %s1b2dc1, %s1b2da : tensor<32x64x56x56xf32>
    %s1b1daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1dam = stablehlo.compare GT, %s1b1a, %s1b1daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b1da = stablehlo.select %s1b1dam, %s1b2dx, %s1b1daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b1dn2dxh = stablehlo.multiply %s1b1n2gb, %s1b1da : tensor<32x64x56x56xf32>
    %s1b1dn2sdxr = stablehlo.reduce(%s1b1dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1dn2sdx = stablehlo.broadcast_in_dim %s1b1dn2sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn2xd = stablehlo.multiply %s1b1n2xh, %s1b1dn2dxh : tensor<32x64x56x56xf32>
    %s1b1dn2sxdr = stablehlo.reduce(%s1b1dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1dn2sxd = stablehlo.broadcast_in_dim %s1b1dn2sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn2t1 = stablehlo.multiply %s1b1dn2dxh, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1dn2i1 = stablehlo.subtract %s1b1dn2t1, %s1b1dn2sdx : tensor<32x64x56x56xf32>
    %s1b1dn2xs = stablehlo.multiply %s1b1n2xh, %s1b1dn2sxd : tensor<32x64x56x56xf32>
    %s1b1dn2i2 = stablehlo.subtract %s1b1dn2i1, %s1b1dn2xs : tensor<32x64x56x56xf32>
    %s1b1dn2sN = stablehlo.divide %s1b1n2istd, %s1b1n2nf : tensor<32x64x56x56xf32>
    %s1b1dn2 = stablehlo.multiply %s1b1dn2sN, %s1b1dn2i2 : tensor<32x64x56x56xf32>
    %s1b1dn2dgp = stablehlo.multiply %s1b1da, %s1b1n2xh : tensor<32x64x56x56xf32>
    %s1b1dn2dg = stablehlo.reduce(%s1b1dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn2db = stablehlo.reduce(%s1b1da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dc2t = stablehlo.transpose %s1b1W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dc2r = stablehlo.reverse %s1b1dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b1dc2 = stablehlo.convolution(%s1b1dn2, %s1b1dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dW2xt = stablehlo.transpose %s1b1r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW2dt = stablehlo.transpose %s1b1dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW2raw = stablehlo.convolution(%s1b1dW2xt, %s1b1dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dW2 = stablehlo.transpose %s1b1dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1db2 = stablehlo.reduce(%s1b1dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b1dr1m = stablehlo.compare GT, %s1b1n1, %s1b1dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b1dr1 = stablehlo.select %s1b1dr1m, %s1b1dc2, %s1b1dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b1dn1dxh = stablehlo.multiply %s1b1n1gb, %s1b1dr1 : tensor<32x64x56x56xf32>
    %s1b1dn1sdxr = stablehlo.reduce(%s1b1dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1dn1sdx = stablehlo.broadcast_in_dim %s1b1dn1sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn1xd = stablehlo.multiply %s1b1n1xh, %s1b1dn1dxh : tensor<32x64x56x56xf32>
    %s1b1dn1sxdr = stablehlo.reduce(%s1b1dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b1dn1sxd = stablehlo.broadcast_in_dim %s1b1dn1sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dn1t1 = stablehlo.multiply %s1b1dn1dxh, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1dn1i1 = stablehlo.subtract %s1b1dn1t1, %s1b1dn1sdx : tensor<32x64x56x56xf32>
    %s1b1dn1xs = stablehlo.multiply %s1b1n1xh, %s1b1dn1sxd : tensor<32x64x56x56xf32>
    %s1b1dn1i2 = stablehlo.subtract %s1b1dn1i1, %s1b1dn1xs : tensor<32x64x56x56xf32>
    %s1b1dn1sN = stablehlo.divide %s1b1n1istd, %s1b1n1nf : tensor<32x64x56x56xf32>
    %s1b1dn1 = stablehlo.multiply %s1b1dn1sN, %s1b1dn1i2 : tensor<32x64x56x56xf32>
    %s1b1dn1dgp = stablehlo.multiply %s1b1dr1, %s1b1n1xh : tensor<32x64x56x56xf32>
    %s1b1dn1dg = stablehlo.reduce(%s1b1dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dn1db = stablehlo.reduce(%s1b1dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dc1t = stablehlo.transpose %s1b1W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dc1r = stablehlo.reverse %s1b1dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b1dc1 = stablehlo.convolution(%s1b1dn1, %s1b1dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b1dW1xt = stablehlo.transpose %s1b0o, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW1dt = stablehlo.transpose %s1b1dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b1dW1raw = stablehlo.convolution(%s1b1dW1xt, %s1b1dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b1dW1 = stablehlo.transpose %s1b1dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b1db1 = stablehlo.reduce(%s1b1dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b1dx = stablehlo.add %s1b1dc1, %s1b1da : tensor<32x64x56x56xf32>
    %s1b0daz = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0dam = stablehlo.compare GT, %s1b0a, %s1b0daz : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b0da = stablehlo.select %s1b0dam, %s1b1dx, %s1b0daz : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b0dn2dxh = stablehlo.multiply %s1b0n2gb, %s1b0da : tensor<32x64x56x56xf32>
    %s1b0dn2sdxr = stablehlo.reduce(%s1b0dn2dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0dn2sdx = stablehlo.broadcast_in_dim %s1b0dn2sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn2xd = stablehlo.multiply %s1b0n2xh, %s1b0dn2dxh : tensor<32x64x56x56xf32>
    %s1b0dn2sxdr = stablehlo.reduce(%s1b0dn2xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0dn2sxd = stablehlo.broadcast_in_dim %s1b0dn2sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn2t1 = stablehlo.multiply %s1b0dn2dxh, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0dn2i1 = stablehlo.subtract %s1b0dn2t1, %s1b0dn2sdx : tensor<32x64x56x56xf32>
    %s1b0dn2xs = stablehlo.multiply %s1b0n2xh, %s1b0dn2sxd : tensor<32x64x56x56xf32>
    %s1b0dn2i2 = stablehlo.subtract %s1b0dn2i1, %s1b0dn2xs : tensor<32x64x56x56xf32>
    %s1b0dn2sN = stablehlo.divide %s1b0n2istd, %s1b0n2nf : tensor<32x64x56x56xf32>
    %s1b0dn2 = stablehlo.multiply %s1b0dn2sN, %s1b0dn2i2 : tensor<32x64x56x56xf32>
    %s1b0dn2dgp = stablehlo.multiply %s1b0da, %s1b0n2xh : tensor<32x64x56x56xf32>
    %s1b0dn2dg = stablehlo.reduce(%s1b0dn2dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn2db = stablehlo.reduce(%s1b0da init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dc2t = stablehlo.transpose %s1b0W2, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dc2r = stablehlo.reverse %s1b0dc2t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b0dc2 = stablehlo.convolution(%s1b0dn2, %s1b0dc2r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dW2xt = stablehlo.transpose %s1b0r1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW2dt = stablehlo.transpose %s1b0dn2, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW2raw = stablehlo.convolution(%s1b0dW2xt, %s1b0dW2dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dW2 = stablehlo.transpose %s1b0dW2raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0db2 = stablehlo.reduce(%s1b0dn2 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dr1z = stablehlo.constant dense<0.0> : tensor<32x64x56x56xf32>
    %s1b0dr1m = stablehlo.compare GT, %s1b0n1, %s1b0dr1z : (tensor<32x64x56x56xf32>, tensor<32x64x56x56xf32>) -> tensor<32x64x56x56xi1>
    %s1b0dr1 = stablehlo.select %s1b0dr1m, %s1b0dc2, %s1b0dr1z : tensor<32x64x56x56xi1>, tensor<32x64x56x56xf32>
    %s1b0dn1dxh = stablehlo.multiply %s1b0n1gb, %s1b0dr1 : tensor<32x64x56x56xf32>
    %s1b0dn1sdxr = stablehlo.reduce(%s1b0dn1dxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0dn1sdx = stablehlo.broadcast_in_dim %s1b0dn1sdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn1xd = stablehlo.multiply %s1b0n1xh, %s1b0dn1dxh : tensor<32x64x56x56xf32>
    %s1b0dn1sxdr = stablehlo.reduce(%s1b0dn1xd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64xf32>
    %s1b0dn1sxd = stablehlo.broadcast_in_dim %s1b0dn1sxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dn1t1 = stablehlo.multiply %s1b0dn1dxh, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0dn1i1 = stablehlo.subtract %s1b0dn1t1, %s1b0dn1sdx : tensor<32x64x56x56xf32>
    %s1b0dn1xs = stablehlo.multiply %s1b0n1xh, %s1b0dn1sxd : tensor<32x64x56x56xf32>
    %s1b0dn1i2 = stablehlo.subtract %s1b0dn1i1, %s1b0dn1xs : tensor<32x64x56x56xf32>
    %s1b0dn1sN = stablehlo.divide %s1b0n1istd, %s1b0n1nf : tensor<32x64x56x56xf32>
    %s1b0dn1 = stablehlo.multiply %s1b0dn1sN, %s1b0dn1i2 : tensor<32x64x56x56xf32>
    %s1b0dn1dgp = stablehlo.multiply %s1b0dr1, %s1b0n1xh : tensor<32x64x56x56xf32>
    %s1b0dn1dg = stablehlo.reduce(%s1b0dn1dgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dn1db = stablehlo.reduce(%s1b0dr1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dc1t = stablehlo.transpose %s1b0W1, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dc1r = stablehlo.reverse %s1b0dc1t, dims = [2, 3] : tensor<64x64x3x3xf32>
    %s1b0dc1 = stablehlo.convolution(%s1b0dn1, %s1b0dc1r)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<32x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<32x64x56x56xf32>
    %s1b0dW1xt = stablehlo.transpose %stp, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW1dt = stablehlo.transpose %s1b0dn1, dims = [1, 0, 2, 3] : (tensor<32x64x56x56xf32>) -> tensor<64x32x56x56xf32>
    %s1b0dW1raw = stablehlo.convolution(%s1b0dW1xt, %s1b0dW1dt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x32x56x56xf32>, tensor<64x32x56x56xf32>) -> tensor<64x64x3x3xf32>
    %s1b0dW1 = stablehlo.transpose %s1b0dW1raw, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %s1b0db1 = stablehlo.reduce(%s1b0dn1 init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %s1b0dx = stablehlo.add %s1b0dc1, %s1b0da : tensor<32x64x56x56xf32>
    %dmp = "stablehlo.select_and_scatter"(%str, %s1b0dx, %sc) ({
      ^bb0(%qa: tensor<f32>, %qb: tensor<f32>):
        %qge = stablehlo.compare GE, %qa, %qb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %qge : tensor<i1>
    }, {
      ^bb0(%qc: tensor<f32>, %qd: tensor<f32>):
        %qs = stablehlo.add %qc, %qd : tensor<f32>
        stablehlo.return %qs : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 2, 2>, window_strides = array<i64: 1, 1, 2, 2>} : (tensor<32x64x112x112xf32>, tensor<32x64x56x56xf32>, tensor<f32>) -> tensor<32x64x112x112xf32>
    %dstrz = stablehlo.constant dense<0.0> : tensor<32x64x112x112xf32>
    %dstrm = stablehlo.compare GT, %stn, %dstrz : (tensor<32x64x112x112xf32>, tensor<32x64x112x112xf32>) -> tensor<32x64x112x112xi1>
    %dstr = stablehlo.select %dstrm, %dmp, %dstrz : tensor<32x64x112x112xi1>, tensor<32x64x112x112xf32>
    %dstndxh = stablehlo.multiply %stngb, %dstr : tensor<32x64x112x112xf32>
    %dstnsdxr = stablehlo.reduce(%dstndxh init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %dstnsdx = stablehlo.broadcast_in_dim %dstnsdxr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %dstnxd = stablehlo.multiply %stnxh, %dstndxh : tensor<32x64x112x112xf32>
    %dstnsxdr = stablehlo.reduce(%dstnxd init: %sc) applies stablehlo.add across dimensions = [2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64xf32>
    %dstnsxd = stablehlo.broadcast_in_dim %dstnsxdr, dims = [0, 1] : (tensor<32x64xf32>) -> tensor<32x64x112x112xf32>
    %dstnt1 = stablehlo.multiply %dstndxh, %stnnf : tensor<32x64x112x112xf32>
    %dstni1 = stablehlo.subtract %dstnt1, %dstnsdx : tensor<32x64x112x112xf32>
    %dstnxs = stablehlo.multiply %stnxh, %dstnsxd : tensor<32x64x112x112xf32>
    %dstni2 = stablehlo.subtract %dstni1, %dstnxs : tensor<32x64x112x112xf32>
    %dstnsN = stablehlo.divide %stnistd, %stnnf : tensor<32x64x112x112xf32>
    %dstn = stablehlo.multiply %dstnsN, %dstni2 : tensor<32x64x112x112xf32>
    %dstndgp = stablehlo.multiply %dstr, %stnxh : tensor<32x64x112x112xf32>
    %dstndg = stablehlo.reduce(%dstndgp init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dstndb = stablehlo.reduce(%dstr init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dsb = stablehlo.reduce(%dstn init: %sc) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %dsWu = stablehlo.pad %dstn, %sc, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<32x64x112x112xf32>, tensor<f32>) -> tensor<32x64x224x224xf32>
    %dsWxt = stablehlo.transpose %xr, dims = [1, 0, 2, 3] : (tensor<32x3x224x224xf32>) -> tensor<3x32x224x224xf32>
    %dsWdt = stablehlo.transpose %dsWu, dims = [1, 0, 2, 3] : (tensor<32x64x224x224xf32>) -> tensor<64x32x224x224xf32>
    %dsWraw = stablehlo.convolution(%dsWxt, %dsWdt)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x32x224x224xf32>, tensor<64x32x224x224xf32>) -> tensor<3x64x7x7xf32>
    %dsW = stablehlo.transpose %dsWraw, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %b1 = stablehlo.constant dense<0.9> : tensor<f32>
    %ob1 = stablehlo.constant dense<0.1> : tensor<f32>
    %b2 = stablehlo.constant dense<0.999> : tensor<f32>
    %ob2 = stablehlo.constant dense<0.001> : tensor<f32>
    %eps = stablehlo.constant dense<1.0e-8> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %adb1sW = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adob1sW = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %admssW = stablehlo.multiply %adb1sW, %sWm : tensor<64x3x7x7xf32>
    %admgsW = stablehlo.multiply %adob1sW, %dsW : tensor<64x3x7x7xf32>
    %admnsW = stablehlo.add %admssW, %admgsW : tensor<64x3x7x7xf32>
    %adb2sW = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adob2sW = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %advssW = stablehlo.multiply %adb2sW, %sWv : tensor<64x3x7x7xf32>
    %adg2sW = stablehlo.multiply %dsW, %dsW : tensor<64x3x7x7xf32>
    %advgsW = stablehlo.multiply %adob2sW, %adg2sW : tensor<64x3x7x7xf32>
    %advnsW = stablehlo.add %advssW, %advgsW : tensor<64x3x7x7xf32>
    %adbc1sW = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adbc2sW = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %admhsW = stablehlo.divide %admnsW, %adbc1sW : tensor<64x3x7x7xf32>
    %advhsW = stablehlo.divide %advnsW, %adbc2sW : tensor<64x3x7x7xf32>
    %adlrsW = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adepssW = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adsqsW = stablehlo.sqrt %advhsW : tensor<64x3x7x7xf32>
    %addensW = stablehlo.add %adsqsW, %adepssW : tensor<64x3x7x7xf32>
    %adratsW = stablehlo.divide %admhsW, %addensW : tensor<64x3x7x7xf32>
    %adstsW = stablehlo.multiply %adlrsW, %adratsW : tensor<64x3x7x7xf32>
    %adsubsW = stablehlo.subtract %sW, %adstsW : tensor<64x3x7x7xf32>
    %adwdsW = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %adwdlrsW = stablehlo.multiply %adwdsW, %adlrsW : tensor<64x3x7x7xf32>
    %adwdpsW = stablehlo.multiply %adwdlrsW, %sW : tensor<64x3x7x7xf32>
    %adnewsW = stablehlo.subtract %adsubsW, %adwdpsW : tensor<64x3x7x7xf32>
    %adb1sb = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1sb = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admssb = stablehlo.multiply %adb1sb, %sbm : tensor<64xf32>
    %admgsb = stablehlo.multiply %adob1sb, %dsb : tensor<64xf32>
    %admnsb = stablehlo.add %admssb, %admgsb : tensor<64xf32>
    %adb2sb = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2sb = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advssb = stablehlo.multiply %adb2sb, %sbv : tensor<64xf32>
    %adg2sb = stablehlo.multiply %dsb, %dsb : tensor<64xf32>
    %advgsb = stablehlo.multiply %adob2sb, %adg2sb : tensor<64xf32>
    %advnsb = stablehlo.add %advssb, %advgsb : tensor<64xf32>
    %adbc1sb = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2sb = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhsb = stablehlo.divide %admnsb, %adbc1sb : tensor<64xf32>
    %advhsb = stablehlo.divide %advnsb, %adbc2sb : tensor<64xf32>
    %adlrsb = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepssb = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqsb = stablehlo.sqrt %advhsb : tensor<64xf32>
    %addensb = stablehlo.add %adsqsb, %adepssb : tensor<64xf32>
    %adratsb = stablehlo.divide %admhsb, %addensb : tensor<64xf32>
    %adstsb = stablehlo.multiply %adlrsb, %adratsb : tensor<64xf32>
    %adsubsb = stablehlo.subtract %sb, %adstsb : tensor<64xf32>
    %adwdsb = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrsb = stablehlo.multiply %adwdsb, %adlrsb : tensor<64xf32>
    %adwdpsb = stablehlo.multiply %adwdlrsb, %sb : tensor<64xf32>
    %adnewsb = stablehlo.subtract %adsubsb, %adwdpsb : tensor<64xf32>
    %adb1sg = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1sg = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admssg = stablehlo.multiply %adb1sg, %sgm : tensor<64xf32>
    %admgsg = stablehlo.multiply %adob1sg, %dstndg : tensor<64xf32>
    %admnsg = stablehlo.add %admssg, %admgsg : tensor<64xf32>
    %adb2sg = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2sg = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advssg = stablehlo.multiply %adb2sg, %sgv : tensor<64xf32>
    %adg2sg = stablehlo.multiply %dstndg, %dstndg : tensor<64xf32>
    %advgsg = stablehlo.multiply %adob2sg, %adg2sg : tensor<64xf32>
    %advnsg = stablehlo.add %advssg, %advgsg : tensor<64xf32>
    %adbc1sg = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2sg = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhsg = stablehlo.divide %admnsg, %adbc1sg : tensor<64xf32>
    %advhsg = stablehlo.divide %advnsg, %adbc2sg : tensor<64xf32>
    %adlrsg = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepssg = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqsg = stablehlo.sqrt %advhsg : tensor<64xf32>
    %addensg = stablehlo.add %adsqsg, %adepssg : tensor<64xf32>
    %adratsg = stablehlo.divide %admhsg, %addensg : tensor<64xf32>
    %adstsg = stablehlo.multiply %adlrsg, %adratsg : tensor<64xf32>
    %adsubsg = stablehlo.subtract %sg, %adstsg : tensor<64xf32>
    %adwdsg = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrsg = stablehlo.multiply %adwdsg, %adlrsg : tensor<64xf32>
    %adwdpsg = stablehlo.multiply %adwdlrsg, %sg : tensor<64xf32>
    %adnewsg = stablehlo.subtract %adsubsg, %adwdpsg : tensor<64xf32>
    %adb1sbt = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1sbt = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admssbt = stablehlo.multiply %adb1sbt, %sbtm : tensor<64xf32>
    %admgsbt = stablehlo.multiply %adob1sbt, %dstndb : tensor<64xf32>
    %admnsbt = stablehlo.add %admssbt, %admgsbt : tensor<64xf32>
    %adb2sbt = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2sbt = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advssbt = stablehlo.multiply %adb2sbt, %sbtv : tensor<64xf32>
    %adg2sbt = stablehlo.multiply %dstndb, %dstndb : tensor<64xf32>
    %advgsbt = stablehlo.multiply %adob2sbt, %adg2sbt : tensor<64xf32>
    %advnsbt = stablehlo.add %advssbt, %advgsbt : tensor<64xf32>
    %adbc1sbt = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2sbt = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhsbt = stablehlo.divide %admnsbt, %adbc1sbt : tensor<64xf32>
    %advhsbt = stablehlo.divide %advnsbt, %adbc2sbt : tensor<64xf32>
    %adlrsbt = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepssbt = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqsbt = stablehlo.sqrt %advhsbt : tensor<64xf32>
    %addensbt = stablehlo.add %adsqsbt, %adepssbt : tensor<64xf32>
    %adratsbt = stablehlo.divide %admhsbt, %addensbt : tensor<64xf32>
    %adstsbt = stablehlo.multiply %adlrsbt, %adratsbt : tensor<64xf32>
    %adsubsbt = stablehlo.subtract %sbt, %adstsbt : tensor<64xf32>
    %adwdsbt = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrsbt = stablehlo.multiply %adwdsbt, %adlrsbt : tensor<64xf32>
    %adwdpsbt = stablehlo.multiply %adwdlrsbt, %sbt : tensor<64xf32>
    %adnewsbt = stablehlo.subtract %adsubsbt, %adwdpsbt : tensor<64xf32>
    %adb1s1b0W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b0W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b0W1 = stablehlo.multiply %adb1s1b0W1, %s1b0W1m : tensor<64x64x3x3xf32>
    %admgs1b0W1 = stablehlo.multiply %adob1s1b0W1, %s1b0dW1 : tensor<64x64x3x3xf32>
    %admns1b0W1 = stablehlo.add %admss1b0W1, %admgs1b0W1 : tensor<64x64x3x3xf32>
    %adb2s1b0W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b0W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b0W1 = stablehlo.multiply %adb2s1b0W1, %s1b0W1v : tensor<64x64x3x3xf32>
    %adg2s1b0W1 = stablehlo.multiply %s1b0dW1, %s1b0dW1 : tensor<64x64x3x3xf32>
    %advgs1b0W1 = stablehlo.multiply %adob2s1b0W1, %adg2s1b0W1 : tensor<64x64x3x3xf32>
    %advns1b0W1 = stablehlo.add %advss1b0W1, %advgs1b0W1 : tensor<64x64x3x3xf32>
    %adbc1s1b0W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b0W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b0W1 = stablehlo.divide %admns1b0W1, %adbc1s1b0W1 : tensor<64x64x3x3xf32>
    %advhs1b0W1 = stablehlo.divide %advns1b0W1, %adbc2s1b0W1 : tensor<64x64x3x3xf32>
    %adlrs1b0W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b0W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b0W1 = stablehlo.sqrt %advhs1b0W1 : tensor<64x64x3x3xf32>
    %addens1b0W1 = stablehlo.add %adsqs1b0W1, %adepss1b0W1 : tensor<64x64x3x3xf32>
    %adrats1b0W1 = stablehlo.divide %admhs1b0W1, %addens1b0W1 : tensor<64x64x3x3xf32>
    %adsts1b0W1 = stablehlo.multiply %adlrs1b0W1, %adrats1b0W1 : tensor<64x64x3x3xf32>
    %adsubs1b0W1 = stablehlo.subtract %s1b0W1, %adsts1b0W1 : tensor<64x64x3x3xf32>
    %adwds1b0W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b0W1 = stablehlo.multiply %adwds1b0W1, %adlrs1b0W1 : tensor<64x64x3x3xf32>
    %adwdps1b0W1 = stablehlo.multiply %adwdlrs1b0W1, %s1b0W1 : tensor<64x64x3x3xf32>
    %adnews1b0W1 = stablehlo.subtract %adsubs1b0W1, %adwdps1b0W1 : tensor<64x64x3x3xf32>
    %adb1s1b0b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0b1 = stablehlo.multiply %adb1s1b0b1, %s1b0b1m : tensor<64xf32>
    %admgs1b0b1 = stablehlo.multiply %adob1s1b0b1, %s1b0db1 : tensor<64xf32>
    %admns1b0b1 = stablehlo.add %admss1b0b1, %admgs1b0b1 : tensor<64xf32>
    %adb2s1b0b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0b1 = stablehlo.multiply %adb2s1b0b1, %s1b0b1v : tensor<64xf32>
    %adg2s1b0b1 = stablehlo.multiply %s1b0db1, %s1b0db1 : tensor<64xf32>
    %advgs1b0b1 = stablehlo.multiply %adob2s1b0b1, %adg2s1b0b1 : tensor<64xf32>
    %advns1b0b1 = stablehlo.add %advss1b0b1, %advgs1b0b1 : tensor<64xf32>
    %adbc1s1b0b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0b1 = stablehlo.divide %admns1b0b1, %adbc1s1b0b1 : tensor<64xf32>
    %advhs1b0b1 = stablehlo.divide %advns1b0b1, %adbc2s1b0b1 : tensor<64xf32>
    %adlrs1b0b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0b1 = stablehlo.sqrt %advhs1b0b1 : tensor<64xf32>
    %addens1b0b1 = stablehlo.add %adsqs1b0b1, %adepss1b0b1 : tensor<64xf32>
    %adrats1b0b1 = stablehlo.divide %admhs1b0b1, %addens1b0b1 : tensor<64xf32>
    %adsts1b0b1 = stablehlo.multiply %adlrs1b0b1, %adrats1b0b1 : tensor<64xf32>
    %adsubs1b0b1 = stablehlo.subtract %s1b0b1, %adsts1b0b1 : tensor<64xf32>
    %adwds1b0b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0b1 = stablehlo.multiply %adwds1b0b1, %adlrs1b0b1 : tensor<64xf32>
    %adwdps1b0b1 = stablehlo.multiply %adwdlrs1b0b1, %s1b0b1 : tensor<64xf32>
    %adnews1b0b1 = stablehlo.subtract %adsubs1b0b1, %adwdps1b0b1 : tensor<64xf32>
    %adb1s1b0g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0g1 = stablehlo.multiply %adb1s1b0g1, %s1b0g1m : tensor<64xf32>
    %admgs1b0g1 = stablehlo.multiply %adob1s1b0g1, %s1b0dn1dg : tensor<64xf32>
    %admns1b0g1 = stablehlo.add %admss1b0g1, %admgs1b0g1 : tensor<64xf32>
    %adb2s1b0g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0g1 = stablehlo.multiply %adb2s1b0g1, %s1b0g1v : tensor<64xf32>
    %adg2s1b0g1 = stablehlo.multiply %s1b0dn1dg, %s1b0dn1dg : tensor<64xf32>
    %advgs1b0g1 = stablehlo.multiply %adob2s1b0g1, %adg2s1b0g1 : tensor<64xf32>
    %advns1b0g1 = stablehlo.add %advss1b0g1, %advgs1b0g1 : tensor<64xf32>
    %adbc1s1b0g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0g1 = stablehlo.divide %admns1b0g1, %adbc1s1b0g1 : tensor<64xf32>
    %advhs1b0g1 = stablehlo.divide %advns1b0g1, %adbc2s1b0g1 : tensor<64xf32>
    %adlrs1b0g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0g1 = stablehlo.sqrt %advhs1b0g1 : tensor<64xf32>
    %addens1b0g1 = stablehlo.add %adsqs1b0g1, %adepss1b0g1 : tensor<64xf32>
    %adrats1b0g1 = stablehlo.divide %admhs1b0g1, %addens1b0g1 : tensor<64xf32>
    %adsts1b0g1 = stablehlo.multiply %adlrs1b0g1, %adrats1b0g1 : tensor<64xf32>
    %adsubs1b0g1 = stablehlo.subtract %s1b0g1, %adsts1b0g1 : tensor<64xf32>
    %adwds1b0g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0g1 = stablehlo.multiply %adwds1b0g1, %adlrs1b0g1 : tensor<64xf32>
    %adwdps1b0g1 = stablehlo.multiply %adwdlrs1b0g1, %s1b0g1 : tensor<64xf32>
    %adnews1b0g1 = stablehlo.subtract %adsubs1b0g1, %adwdps1b0g1 : tensor<64xf32>
    %adb1s1b0bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0bt1 = stablehlo.multiply %adb1s1b0bt1, %s1b0bt1m : tensor<64xf32>
    %admgs1b0bt1 = stablehlo.multiply %adob1s1b0bt1, %s1b0dn1db : tensor<64xf32>
    %admns1b0bt1 = stablehlo.add %admss1b0bt1, %admgs1b0bt1 : tensor<64xf32>
    %adb2s1b0bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0bt1 = stablehlo.multiply %adb2s1b0bt1, %s1b0bt1v : tensor<64xf32>
    %adg2s1b0bt1 = stablehlo.multiply %s1b0dn1db, %s1b0dn1db : tensor<64xf32>
    %advgs1b0bt1 = stablehlo.multiply %adob2s1b0bt1, %adg2s1b0bt1 : tensor<64xf32>
    %advns1b0bt1 = stablehlo.add %advss1b0bt1, %advgs1b0bt1 : tensor<64xf32>
    %adbc1s1b0bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0bt1 = stablehlo.divide %admns1b0bt1, %adbc1s1b0bt1 : tensor<64xf32>
    %advhs1b0bt1 = stablehlo.divide %advns1b0bt1, %adbc2s1b0bt1 : tensor<64xf32>
    %adlrs1b0bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0bt1 = stablehlo.sqrt %advhs1b0bt1 : tensor<64xf32>
    %addens1b0bt1 = stablehlo.add %adsqs1b0bt1, %adepss1b0bt1 : tensor<64xf32>
    %adrats1b0bt1 = stablehlo.divide %admhs1b0bt1, %addens1b0bt1 : tensor<64xf32>
    %adsts1b0bt1 = stablehlo.multiply %adlrs1b0bt1, %adrats1b0bt1 : tensor<64xf32>
    %adsubs1b0bt1 = stablehlo.subtract %s1b0bt1, %adsts1b0bt1 : tensor<64xf32>
    %adwds1b0bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0bt1 = stablehlo.multiply %adwds1b0bt1, %adlrs1b0bt1 : tensor<64xf32>
    %adwdps1b0bt1 = stablehlo.multiply %adwdlrs1b0bt1, %s1b0bt1 : tensor<64xf32>
    %adnews1b0bt1 = stablehlo.subtract %adsubs1b0bt1, %adwdps1b0bt1 : tensor<64xf32>
    %adb1s1b0W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b0W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b0W2 = stablehlo.multiply %adb1s1b0W2, %s1b0W2m : tensor<64x64x3x3xf32>
    %admgs1b0W2 = stablehlo.multiply %adob1s1b0W2, %s1b0dW2 : tensor<64x64x3x3xf32>
    %admns1b0W2 = stablehlo.add %admss1b0W2, %admgs1b0W2 : tensor<64x64x3x3xf32>
    %adb2s1b0W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b0W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b0W2 = stablehlo.multiply %adb2s1b0W2, %s1b0W2v : tensor<64x64x3x3xf32>
    %adg2s1b0W2 = stablehlo.multiply %s1b0dW2, %s1b0dW2 : tensor<64x64x3x3xf32>
    %advgs1b0W2 = stablehlo.multiply %adob2s1b0W2, %adg2s1b0W2 : tensor<64x64x3x3xf32>
    %advns1b0W2 = stablehlo.add %advss1b0W2, %advgs1b0W2 : tensor<64x64x3x3xf32>
    %adbc1s1b0W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b0W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b0W2 = stablehlo.divide %admns1b0W2, %adbc1s1b0W2 : tensor<64x64x3x3xf32>
    %advhs1b0W2 = stablehlo.divide %advns1b0W2, %adbc2s1b0W2 : tensor<64x64x3x3xf32>
    %adlrs1b0W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b0W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b0W2 = stablehlo.sqrt %advhs1b0W2 : tensor<64x64x3x3xf32>
    %addens1b0W2 = stablehlo.add %adsqs1b0W2, %adepss1b0W2 : tensor<64x64x3x3xf32>
    %adrats1b0W2 = stablehlo.divide %admhs1b0W2, %addens1b0W2 : tensor<64x64x3x3xf32>
    %adsts1b0W2 = stablehlo.multiply %adlrs1b0W2, %adrats1b0W2 : tensor<64x64x3x3xf32>
    %adsubs1b0W2 = stablehlo.subtract %s1b0W2, %adsts1b0W2 : tensor<64x64x3x3xf32>
    %adwds1b0W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b0W2 = stablehlo.multiply %adwds1b0W2, %adlrs1b0W2 : tensor<64x64x3x3xf32>
    %adwdps1b0W2 = stablehlo.multiply %adwdlrs1b0W2, %s1b0W2 : tensor<64x64x3x3xf32>
    %adnews1b0W2 = stablehlo.subtract %adsubs1b0W2, %adwdps1b0W2 : tensor<64x64x3x3xf32>
    %adb1s1b0b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0b2 = stablehlo.multiply %adb1s1b0b2, %s1b0b2m : tensor<64xf32>
    %admgs1b0b2 = stablehlo.multiply %adob1s1b0b2, %s1b0db2 : tensor<64xf32>
    %admns1b0b2 = stablehlo.add %admss1b0b2, %admgs1b0b2 : tensor<64xf32>
    %adb2s1b0b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0b2 = stablehlo.multiply %adb2s1b0b2, %s1b0b2v : tensor<64xf32>
    %adg2s1b0b2 = stablehlo.multiply %s1b0db2, %s1b0db2 : tensor<64xf32>
    %advgs1b0b2 = stablehlo.multiply %adob2s1b0b2, %adg2s1b0b2 : tensor<64xf32>
    %advns1b0b2 = stablehlo.add %advss1b0b2, %advgs1b0b2 : tensor<64xf32>
    %adbc1s1b0b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0b2 = stablehlo.divide %admns1b0b2, %adbc1s1b0b2 : tensor<64xf32>
    %advhs1b0b2 = stablehlo.divide %advns1b0b2, %adbc2s1b0b2 : tensor<64xf32>
    %adlrs1b0b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0b2 = stablehlo.sqrt %advhs1b0b2 : tensor<64xf32>
    %addens1b0b2 = stablehlo.add %adsqs1b0b2, %adepss1b0b2 : tensor<64xf32>
    %adrats1b0b2 = stablehlo.divide %admhs1b0b2, %addens1b0b2 : tensor<64xf32>
    %adsts1b0b2 = stablehlo.multiply %adlrs1b0b2, %adrats1b0b2 : tensor<64xf32>
    %adsubs1b0b2 = stablehlo.subtract %s1b0b2, %adsts1b0b2 : tensor<64xf32>
    %adwds1b0b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0b2 = stablehlo.multiply %adwds1b0b2, %adlrs1b0b2 : tensor<64xf32>
    %adwdps1b0b2 = stablehlo.multiply %adwdlrs1b0b2, %s1b0b2 : tensor<64xf32>
    %adnews1b0b2 = stablehlo.subtract %adsubs1b0b2, %adwdps1b0b2 : tensor<64xf32>
    %adb1s1b0g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0g2 = stablehlo.multiply %adb1s1b0g2, %s1b0g2m : tensor<64xf32>
    %admgs1b0g2 = stablehlo.multiply %adob1s1b0g2, %s1b0dn2dg : tensor<64xf32>
    %admns1b0g2 = stablehlo.add %admss1b0g2, %admgs1b0g2 : tensor<64xf32>
    %adb2s1b0g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0g2 = stablehlo.multiply %adb2s1b0g2, %s1b0g2v : tensor<64xf32>
    %adg2s1b0g2 = stablehlo.multiply %s1b0dn2dg, %s1b0dn2dg : tensor<64xf32>
    %advgs1b0g2 = stablehlo.multiply %adob2s1b0g2, %adg2s1b0g2 : tensor<64xf32>
    %advns1b0g2 = stablehlo.add %advss1b0g2, %advgs1b0g2 : tensor<64xf32>
    %adbc1s1b0g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0g2 = stablehlo.divide %admns1b0g2, %adbc1s1b0g2 : tensor<64xf32>
    %advhs1b0g2 = stablehlo.divide %advns1b0g2, %adbc2s1b0g2 : tensor<64xf32>
    %adlrs1b0g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0g2 = stablehlo.sqrt %advhs1b0g2 : tensor<64xf32>
    %addens1b0g2 = stablehlo.add %adsqs1b0g2, %adepss1b0g2 : tensor<64xf32>
    %adrats1b0g2 = stablehlo.divide %admhs1b0g2, %addens1b0g2 : tensor<64xf32>
    %adsts1b0g2 = stablehlo.multiply %adlrs1b0g2, %adrats1b0g2 : tensor<64xf32>
    %adsubs1b0g2 = stablehlo.subtract %s1b0g2, %adsts1b0g2 : tensor<64xf32>
    %adwds1b0g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0g2 = stablehlo.multiply %adwds1b0g2, %adlrs1b0g2 : tensor<64xf32>
    %adwdps1b0g2 = stablehlo.multiply %adwdlrs1b0g2, %s1b0g2 : tensor<64xf32>
    %adnews1b0g2 = stablehlo.subtract %adsubs1b0g2, %adwdps1b0g2 : tensor<64xf32>
    %adb1s1b0bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b0bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b0bt2 = stablehlo.multiply %adb1s1b0bt2, %s1b0bt2m : tensor<64xf32>
    %admgs1b0bt2 = stablehlo.multiply %adob1s1b0bt2, %s1b0dn2db : tensor<64xf32>
    %admns1b0bt2 = stablehlo.add %admss1b0bt2, %admgs1b0bt2 : tensor<64xf32>
    %adb2s1b0bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b0bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b0bt2 = stablehlo.multiply %adb2s1b0bt2, %s1b0bt2v : tensor<64xf32>
    %adg2s1b0bt2 = stablehlo.multiply %s1b0dn2db, %s1b0dn2db : tensor<64xf32>
    %advgs1b0bt2 = stablehlo.multiply %adob2s1b0bt2, %adg2s1b0bt2 : tensor<64xf32>
    %advns1b0bt2 = stablehlo.add %advss1b0bt2, %advgs1b0bt2 : tensor<64xf32>
    %adbc1s1b0bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b0bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b0bt2 = stablehlo.divide %admns1b0bt2, %adbc1s1b0bt2 : tensor<64xf32>
    %advhs1b0bt2 = stablehlo.divide %advns1b0bt2, %adbc2s1b0bt2 : tensor<64xf32>
    %adlrs1b0bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b0bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b0bt2 = stablehlo.sqrt %advhs1b0bt2 : tensor<64xf32>
    %addens1b0bt2 = stablehlo.add %adsqs1b0bt2, %adepss1b0bt2 : tensor<64xf32>
    %adrats1b0bt2 = stablehlo.divide %admhs1b0bt2, %addens1b0bt2 : tensor<64xf32>
    %adsts1b0bt2 = stablehlo.multiply %adlrs1b0bt2, %adrats1b0bt2 : tensor<64xf32>
    %adsubs1b0bt2 = stablehlo.subtract %s1b0bt2, %adsts1b0bt2 : tensor<64xf32>
    %adwds1b0bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b0bt2 = stablehlo.multiply %adwds1b0bt2, %adlrs1b0bt2 : tensor<64xf32>
    %adwdps1b0bt2 = stablehlo.multiply %adwdlrs1b0bt2, %s1b0bt2 : tensor<64xf32>
    %adnews1b0bt2 = stablehlo.subtract %adsubs1b0bt2, %adwdps1b0bt2 : tensor<64xf32>
    %adb1s1b1W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b1W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b1W1 = stablehlo.multiply %adb1s1b1W1, %s1b1W1m : tensor<64x64x3x3xf32>
    %admgs1b1W1 = stablehlo.multiply %adob1s1b1W1, %s1b1dW1 : tensor<64x64x3x3xf32>
    %admns1b1W1 = stablehlo.add %admss1b1W1, %admgs1b1W1 : tensor<64x64x3x3xf32>
    %adb2s1b1W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b1W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b1W1 = stablehlo.multiply %adb2s1b1W1, %s1b1W1v : tensor<64x64x3x3xf32>
    %adg2s1b1W1 = stablehlo.multiply %s1b1dW1, %s1b1dW1 : tensor<64x64x3x3xf32>
    %advgs1b1W1 = stablehlo.multiply %adob2s1b1W1, %adg2s1b1W1 : tensor<64x64x3x3xf32>
    %advns1b1W1 = stablehlo.add %advss1b1W1, %advgs1b1W1 : tensor<64x64x3x3xf32>
    %adbc1s1b1W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b1W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b1W1 = stablehlo.divide %admns1b1W1, %adbc1s1b1W1 : tensor<64x64x3x3xf32>
    %advhs1b1W1 = stablehlo.divide %advns1b1W1, %adbc2s1b1W1 : tensor<64x64x3x3xf32>
    %adlrs1b1W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b1W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b1W1 = stablehlo.sqrt %advhs1b1W1 : tensor<64x64x3x3xf32>
    %addens1b1W1 = stablehlo.add %adsqs1b1W1, %adepss1b1W1 : tensor<64x64x3x3xf32>
    %adrats1b1W1 = stablehlo.divide %admhs1b1W1, %addens1b1W1 : tensor<64x64x3x3xf32>
    %adsts1b1W1 = stablehlo.multiply %adlrs1b1W1, %adrats1b1W1 : tensor<64x64x3x3xf32>
    %adsubs1b1W1 = stablehlo.subtract %s1b1W1, %adsts1b1W1 : tensor<64x64x3x3xf32>
    %adwds1b1W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b1W1 = stablehlo.multiply %adwds1b1W1, %adlrs1b1W1 : tensor<64x64x3x3xf32>
    %adwdps1b1W1 = stablehlo.multiply %adwdlrs1b1W1, %s1b1W1 : tensor<64x64x3x3xf32>
    %adnews1b1W1 = stablehlo.subtract %adsubs1b1W1, %adwdps1b1W1 : tensor<64x64x3x3xf32>
    %adb1s1b1b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1b1 = stablehlo.multiply %adb1s1b1b1, %s1b1b1m : tensor<64xf32>
    %admgs1b1b1 = stablehlo.multiply %adob1s1b1b1, %s1b1db1 : tensor<64xf32>
    %admns1b1b1 = stablehlo.add %admss1b1b1, %admgs1b1b1 : tensor<64xf32>
    %adb2s1b1b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1b1 = stablehlo.multiply %adb2s1b1b1, %s1b1b1v : tensor<64xf32>
    %adg2s1b1b1 = stablehlo.multiply %s1b1db1, %s1b1db1 : tensor<64xf32>
    %advgs1b1b1 = stablehlo.multiply %adob2s1b1b1, %adg2s1b1b1 : tensor<64xf32>
    %advns1b1b1 = stablehlo.add %advss1b1b1, %advgs1b1b1 : tensor<64xf32>
    %adbc1s1b1b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1b1 = stablehlo.divide %admns1b1b1, %adbc1s1b1b1 : tensor<64xf32>
    %advhs1b1b1 = stablehlo.divide %advns1b1b1, %adbc2s1b1b1 : tensor<64xf32>
    %adlrs1b1b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1b1 = stablehlo.sqrt %advhs1b1b1 : tensor<64xf32>
    %addens1b1b1 = stablehlo.add %adsqs1b1b1, %adepss1b1b1 : tensor<64xf32>
    %adrats1b1b1 = stablehlo.divide %admhs1b1b1, %addens1b1b1 : tensor<64xf32>
    %adsts1b1b1 = stablehlo.multiply %adlrs1b1b1, %adrats1b1b1 : tensor<64xf32>
    %adsubs1b1b1 = stablehlo.subtract %s1b1b1, %adsts1b1b1 : tensor<64xf32>
    %adwds1b1b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1b1 = stablehlo.multiply %adwds1b1b1, %adlrs1b1b1 : tensor<64xf32>
    %adwdps1b1b1 = stablehlo.multiply %adwdlrs1b1b1, %s1b1b1 : tensor<64xf32>
    %adnews1b1b1 = stablehlo.subtract %adsubs1b1b1, %adwdps1b1b1 : tensor<64xf32>
    %adb1s1b1g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1g1 = stablehlo.multiply %adb1s1b1g1, %s1b1g1m : tensor<64xf32>
    %admgs1b1g1 = stablehlo.multiply %adob1s1b1g1, %s1b1dn1dg : tensor<64xf32>
    %admns1b1g1 = stablehlo.add %admss1b1g1, %admgs1b1g1 : tensor<64xf32>
    %adb2s1b1g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1g1 = stablehlo.multiply %adb2s1b1g1, %s1b1g1v : tensor<64xf32>
    %adg2s1b1g1 = stablehlo.multiply %s1b1dn1dg, %s1b1dn1dg : tensor<64xf32>
    %advgs1b1g1 = stablehlo.multiply %adob2s1b1g1, %adg2s1b1g1 : tensor<64xf32>
    %advns1b1g1 = stablehlo.add %advss1b1g1, %advgs1b1g1 : tensor<64xf32>
    %adbc1s1b1g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1g1 = stablehlo.divide %admns1b1g1, %adbc1s1b1g1 : tensor<64xf32>
    %advhs1b1g1 = stablehlo.divide %advns1b1g1, %adbc2s1b1g1 : tensor<64xf32>
    %adlrs1b1g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1g1 = stablehlo.sqrt %advhs1b1g1 : tensor<64xf32>
    %addens1b1g1 = stablehlo.add %adsqs1b1g1, %adepss1b1g1 : tensor<64xf32>
    %adrats1b1g1 = stablehlo.divide %admhs1b1g1, %addens1b1g1 : tensor<64xf32>
    %adsts1b1g1 = stablehlo.multiply %adlrs1b1g1, %adrats1b1g1 : tensor<64xf32>
    %adsubs1b1g1 = stablehlo.subtract %s1b1g1, %adsts1b1g1 : tensor<64xf32>
    %adwds1b1g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1g1 = stablehlo.multiply %adwds1b1g1, %adlrs1b1g1 : tensor<64xf32>
    %adwdps1b1g1 = stablehlo.multiply %adwdlrs1b1g1, %s1b1g1 : tensor<64xf32>
    %adnews1b1g1 = stablehlo.subtract %adsubs1b1g1, %adwdps1b1g1 : tensor<64xf32>
    %adb1s1b1bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1bt1 = stablehlo.multiply %adb1s1b1bt1, %s1b1bt1m : tensor<64xf32>
    %admgs1b1bt1 = stablehlo.multiply %adob1s1b1bt1, %s1b1dn1db : tensor<64xf32>
    %admns1b1bt1 = stablehlo.add %admss1b1bt1, %admgs1b1bt1 : tensor<64xf32>
    %adb2s1b1bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1bt1 = stablehlo.multiply %adb2s1b1bt1, %s1b1bt1v : tensor<64xf32>
    %adg2s1b1bt1 = stablehlo.multiply %s1b1dn1db, %s1b1dn1db : tensor<64xf32>
    %advgs1b1bt1 = stablehlo.multiply %adob2s1b1bt1, %adg2s1b1bt1 : tensor<64xf32>
    %advns1b1bt1 = stablehlo.add %advss1b1bt1, %advgs1b1bt1 : tensor<64xf32>
    %adbc1s1b1bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1bt1 = stablehlo.divide %admns1b1bt1, %adbc1s1b1bt1 : tensor<64xf32>
    %advhs1b1bt1 = stablehlo.divide %advns1b1bt1, %adbc2s1b1bt1 : tensor<64xf32>
    %adlrs1b1bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1bt1 = stablehlo.sqrt %advhs1b1bt1 : tensor<64xf32>
    %addens1b1bt1 = stablehlo.add %adsqs1b1bt1, %adepss1b1bt1 : tensor<64xf32>
    %adrats1b1bt1 = stablehlo.divide %admhs1b1bt1, %addens1b1bt1 : tensor<64xf32>
    %adsts1b1bt1 = stablehlo.multiply %adlrs1b1bt1, %adrats1b1bt1 : tensor<64xf32>
    %adsubs1b1bt1 = stablehlo.subtract %s1b1bt1, %adsts1b1bt1 : tensor<64xf32>
    %adwds1b1bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1bt1 = stablehlo.multiply %adwds1b1bt1, %adlrs1b1bt1 : tensor<64xf32>
    %adwdps1b1bt1 = stablehlo.multiply %adwdlrs1b1bt1, %s1b1bt1 : tensor<64xf32>
    %adnews1b1bt1 = stablehlo.subtract %adsubs1b1bt1, %adwdps1b1bt1 : tensor<64xf32>
    %adb1s1b1W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b1W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b1W2 = stablehlo.multiply %adb1s1b1W2, %s1b1W2m : tensor<64x64x3x3xf32>
    %admgs1b1W2 = stablehlo.multiply %adob1s1b1W2, %s1b1dW2 : tensor<64x64x3x3xf32>
    %admns1b1W2 = stablehlo.add %admss1b1W2, %admgs1b1W2 : tensor<64x64x3x3xf32>
    %adb2s1b1W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b1W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b1W2 = stablehlo.multiply %adb2s1b1W2, %s1b1W2v : tensor<64x64x3x3xf32>
    %adg2s1b1W2 = stablehlo.multiply %s1b1dW2, %s1b1dW2 : tensor<64x64x3x3xf32>
    %advgs1b1W2 = stablehlo.multiply %adob2s1b1W2, %adg2s1b1W2 : tensor<64x64x3x3xf32>
    %advns1b1W2 = stablehlo.add %advss1b1W2, %advgs1b1W2 : tensor<64x64x3x3xf32>
    %adbc1s1b1W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b1W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b1W2 = stablehlo.divide %admns1b1W2, %adbc1s1b1W2 : tensor<64x64x3x3xf32>
    %advhs1b1W2 = stablehlo.divide %advns1b1W2, %adbc2s1b1W2 : tensor<64x64x3x3xf32>
    %adlrs1b1W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b1W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b1W2 = stablehlo.sqrt %advhs1b1W2 : tensor<64x64x3x3xf32>
    %addens1b1W2 = stablehlo.add %adsqs1b1W2, %adepss1b1W2 : tensor<64x64x3x3xf32>
    %adrats1b1W2 = stablehlo.divide %admhs1b1W2, %addens1b1W2 : tensor<64x64x3x3xf32>
    %adsts1b1W2 = stablehlo.multiply %adlrs1b1W2, %adrats1b1W2 : tensor<64x64x3x3xf32>
    %adsubs1b1W2 = stablehlo.subtract %s1b1W2, %adsts1b1W2 : tensor<64x64x3x3xf32>
    %adwds1b1W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b1W2 = stablehlo.multiply %adwds1b1W2, %adlrs1b1W2 : tensor<64x64x3x3xf32>
    %adwdps1b1W2 = stablehlo.multiply %adwdlrs1b1W2, %s1b1W2 : tensor<64x64x3x3xf32>
    %adnews1b1W2 = stablehlo.subtract %adsubs1b1W2, %adwdps1b1W2 : tensor<64x64x3x3xf32>
    %adb1s1b1b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1b2 = stablehlo.multiply %adb1s1b1b2, %s1b1b2m : tensor<64xf32>
    %admgs1b1b2 = stablehlo.multiply %adob1s1b1b2, %s1b1db2 : tensor<64xf32>
    %admns1b1b2 = stablehlo.add %admss1b1b2, %admgs1b1b2 : tensor<64xf32>
    %adb2s1b1b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1b2 = stablehlo.multiply %adb2s1b1b2, %s1b1b2v : tensor<64xf32>
    %adg2s1b1b2 = stablehlo.multiply %s1b1db2, %s1b1db2 : tensor<64xf32>
    %advgs1b1b2 = stablehlo.multiply %adob2s1b1b2, %adg2s1b1b2 : tensor<64xf32>
    %advns1b1b2 = stablehlo.add %advss1b1b2, %advgs1b1b2 : tensor<64xf32>
    %adbc1s1b1b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1b2 = stablehlo.divide %admns1b1b2, %adbc1s1b1b2 : tensor<64xf32>
    %advhs1b1b2 = stablehlo.divide %advns1b1b2, %adbc2s1b1b2 : tensor<64xf32>
    %adlrs1b1b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1b2 = stablehlo.sqrt %advhs1b1b2 : tensor<64xf32>
    %addens1b1b2 = stablehlo.add %adsqs1b1b2, %adepss1b1b2 : tensor<64xf32>
    %adrats1b1b2 = stablehlo.divide %admhs1b1b2, %addens1b1b2 : tensor<64xf32>
    %adsts1b1b2 = stablehlo.multiply %adlrs1b1b2, %adrats1b1b2 : tensor<64xf32>
    %adsubs1b1b2 = stablehlo.subtract %s1b1b2, %adsts1b1b2 : tensor<64xf32>
    %adwds1b1b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1b2 = stablehlo.multiply %adwds1b1b2, %adlrs1b1b2 : tensor<64xf32>
    %adwdps1b1b2 = stablehlo.multiply %adwdlrs1b1b2, %s1b1b2 : tensor<64xf32>
    %adnews1b1b2 = stablehlo.subtract %adsubs1b1b2, %adwdps1b1b2 : tensor<64xf32>
    %adb1s1b1g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1g2 = stablehlo.multiply %adb1s1b1g2, %s1b1g2m : tensor<64xf32>
    %admgs1b1g2 = stablehlo.multiply %adob1s1b1g2, %s1b1dn2dg : tensor<64xf32>
    %admns1b1g2 = stablehlo.add %admss1b1g2, %admgs1b1g2 : tensor<64xf32>
    %adb2s1b1g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1g2 = stablehlo.multiply %adb2s1b1g2, %s1b1g2v : tensor<64xf32>
    %adg2s1b1g2 = stablehlo.multiply %s1b1dn2dg, %s1b1dn2dg : tensor<64xf32>
    %advgs1b1g2 = stablehlo.multiply %adob2s1b1g2, %adg2s1b1g2 : tensor<64xf32>
    %advns1b1g2 = stablehlo.add %advss1b1g2, %advgs1b1g2 : tensor<64xf32>
    %adbc1s1b1g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1g2 = stablehlo.divide %admns1b1g2, %adbc1s1b1g2 : tensor<64xf32>
    %advhs1b1g2 = stablehlo.divide %advns1b1g2, %adbc2s1b1g2 : tensor<64xf32>
    %adlrs1b1g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1g2 = stablehlo.sqrt %advhs1b1g2 : tensor<64xf32>
    %addens1b1g2 = stablehlo.add %adsqs1b1g2, %adepss1b1g2 : tensor<64xf32>
    %adrats1b1g2 = stablehlo.divide %admhs1b1g2, %addens1b1g2 : tensor<64xf32>
    %adsts1b1g2 = stablehlo.multiply %adlrs1b1g2, %adrats1b1g2 : tensor<64xf32>
    %adsubs1b1g2 = stablehlo.subtract %s1b1g2, %adsts1b1g2 : tensor<64xf32>
    %adwds1b1g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1g2 = stablehlo.multiply %adwds1b1g2, %adlrs1b1g2 : tensor<64xf32>
    %adwdps1b1g2 = stablehlo.multiply %adwdlrs1b1g2, %s1b1g2 : tensor<64xf32>
    %adnews1b1g2 = stablehlo.subtract %adsubs1b1g2, %adwdps1b1g2 : tensor<64xf32>
    %adb1s1b1bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b1bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b1bt2 = stablehlo.multiply %adb1s1b1bt2, %s1b1bt2m : tensor<64xf32>
    %admgs1b1bt2 = stablehlo.multiply %adob1s1b1bt2, %s1b1dn2db : tensor<64xf32>
    %admns1b1bt2 = stablehlo.add %admss1b1bt2, %admgs1b1bt2 : tensor<64xf32>
    %adb2s1b1bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b1bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b1bt2 = stablehlo.multiply %adb2s1b1bt2, %s1b1bt2v : tensor<64xf32>
    %adg2s1b1bt2 = stablehlo.multiply %s1b1dn2db, %s1b1dn2db : tensor<64xf32>
    %advgs1b1bt2 = stablehlo.multiply %adob2s1b1bt2, %adg2s1b1bt2 : tensor<64xf32>
    %advns1b1bt2 = stablehlo.add %advss1b1bt2, %advgs1b1bt2 : tensor<64xf32>
    %adbc1s1b1bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b1bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b1bt2 = stablehlo.divide %admns1b1bt2, %adbc1s1b1bt2 : tensor<64xf32>
    %advhs1b1bt2 = stablehlo.divide %advns1b1bt2, %adbc2s1b1bt2 : tensor<64xf32>
    %adlrs1b1bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b1bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b1bt2 = stablehlo.sqrt %advhs1b1bt2 : tensor<64xf32>
    %addens1b1bt2 = stablehlo.add %adsqs1b1bt2, %adepss1b1bt2 : tensor<64xf32>
    %adrats1b1bt2 = stablehlo.divide %admhs1b1bt2, %addens1b1bt2 : tensor<64xf32>
    %adsts1b1bt2 = stablehlo.multiply %adlrs1b1bt2, %adrats1b1bt2 : tensor<64xf32>
    %adsubs1b1bt2 = stablehlo.subtract %s1b1bt2, %adsts1b1bt2 : tensor<64xf32>
    %adwds1b1bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b1bt2 = stablehlo.multiply %adwds1b1bt2, %adlrs1b1bt2 : tensor<64xf32>
    %adwdps1b1bt2 = stablehlo.multiply %adwdlrs1b1bt2, %s1b1bt2 : tensor<64xf32>
    %adnews1b1bt2 = stablehlo.subtract %adsubs1b1bt2, %adwdps1b1bt2 : tensor<64xf32>
    %adb1s1b2W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b2W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b2W1 = stablehlo.multiply %adb1s1b2W1, %s1b2W1m : tensor<64x64x3x3xf32>
    %admgs1b2W1 = stablehlo.multiply %adob1s1b2W1, %s1b2dW1 : tensor<64x64x3x3xf32>
    %admns1b2W1 = stablehlo.add %admss1b2W1, %admgs1b2W1 : tensor<64x64x3x3xf32>
    %adb2s1b2W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b2W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b2W1 = stablehlo.multiply %adb2s1b2W1, %s1b2W1v : tensor<64x64x3x3xf32>
    %adg2s1b2W1 = stablehlo.multiply %s1b2dW1, %s1b2dW1 : tensor<64x64x3x3xf32>
    %advgs1b2W1 = stablehlo.multiply %adob2s1b2W1, %adg2s1b2W1 : tensor<64x64x3x3xf32>
    %advns1b2W1 = stablehlo.add %advss1b2W1, %advgs1b2W1 : tensor<64x64x3x3xf32>
    %adbc1s1b2W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b2W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b2W1 = stablehlo.divide %admns1b2W1, %adbc1s1b2W1 : tensor<64x64x3x3xf32>
    %advhs1b2W1 = stablehlo.divide %advns1b2W1, %adbc2s1b2W1 : tensor<64x64x3x3xf32>
    %adlrs1b2W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b2W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b2W1 = stablehlo.sqrt %advhs1b2W1 : tensor<64x64x3x3xf32>
    %addens1b2W1 = stablehlo.add %adsqs1b2W1, %adepss1b2W1 : tensor<64x64x3x3xf32>
    %adrats1b2W1 = stablehlo.divide %admhs1b2W1, %addens1b2W1 : tensor<64x64x3x3xf32>
    %adsts1b2W1 = stablehlo.multiply %adlrs1b2W1, %adrats1b2W1 : tensor<64x64x3x3xf32>
    %adsubs1b2W1 = stablehlo.subtract %s1b2W1, %adsts1b2W1 : tensor<64x64x3x3xf32>
    %adwds1b2W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b2W1 = stablehlo.multiply %adwds1b2W1, %adlrs1b2W1 : tensor<64x64x3x3xf32>
    %adwdps1b2W1 = stablehlo.multiply %adwdlrs1b2W1, %s1b2W1 : tensor<64x64x3x3xf32>
    %adnews1b2W1 = stablehlo.subtract %adsubs1b2W1, %adwdps1b2W1 : tensor<64x64x3x3xf32>
    %adb1s1b2b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2b1 = stablehlo.multiply %adb1s1b2b1, %s1b2b1m : tensor<64xf32>
    %admgs1b2b1 = stablehlo.multiply %adob1s1b2b1, %s1b2db1 : tensor<64xf32>
    %admns1b2b1 = stablehlo.add %admss1b2b1, %admgs1b2b1 : tensor<64xf32>
    %adb2s1b2b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2b1 = stablehlo.multiply %adb2s1b2b1, %s1b2b1v : tensor<64xf32>
    %adg2s1b2b1 = stablehlo.multiply %s1b2db1, %s1b2db1 : tensor<64xf32>
    %advgs1b2b1 = stablehlo.multiply %adob2s1b2b1, %adg2s1b2b1 : tensor<64xf32>
    %advns1b2b1 = stablehlo.add %advss1b2b1, %advgs1b2b1 : tensor<64xf32>
    %adbc1s1b2b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2b1 = stablehlo.divide %admns1b2b1, %adbc1s1b2b1 : tensor<64xf32>
    %advhs1b2b1 = stablehlo.divide %advns1b2b1, %adbc2s1b2b1 : tensor<64xf32>
    %adlrs1b2b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2b1 = stablehlo.sqrt %advhs1b2b1 : tensor<64xf32>
    %addens1b2b1 = stablehlo.add %adsqs1b2b1, %adepss1b2b1 : tensor<64xf32>
    %adrats1b2b1 = stablehlo.divide %admhs1b2b1, %addens1b2b1 : tensor<64xf32>
    %adsts1b2b1 = stablehlo.multiply %adlrs1b2b1, %adrats1b2b1 : tensor<64xf32>
    %adsubs1b2b1 = stablehlo.subtract %s1b2b1, %adsts1b2b1 : tensor<64xf32>
    %adwds1b2b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2b1 = stablehlo.multiply %adwds1b2b1, %adlrs1b2b1 : tensor<64xf32>
    %adwdps1b2b1 = stablehlo.multiply %adwdlrs1b2b1, %s1b2b1 : tensor<64xf32>
    %adnews1b2b1 = stablehlo.subtract %adsubs1b2b1, %adwdps1b2b1 : tensor<64xf32>
    %adb1s1b2g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2g1 = stablehlo.multiply %adb1s1b2g1, %s1b2g1m : tensor<64xf32>
    %admgs1b2g1 = stablehlo.multiply %adob1s1b2g1, %s1b2dn1dg : tensor<64xf32>
    %admns1b2g1 = stablehlo.add %admss1b2g1, %admgs1b2g1 : tensor<64xf32>
    %adb2s1b2g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2g1 = stablehlo.multiply %adb2s1b2g1, %s1b2g1v : tensor<64xf32>
    %adg2s1b2g1 = stablehlo.multiply %s1b2dn1dg, %s1b2dn1dg : tensor<64xf32>
    %advgs1b2g1 = stablehlo.multiply %adob2s1b2g1, %adg2s1b2g1 : tensor<64xf32>
    %advns1b2g1 = stablehlo.add %advss1b2g1, %advgs1b2g1 : tensor<64xf32>
    %adbc1s1b2g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2g1 = stablehlo.divide %admns1b2g1, %adbc1s1b2g1 : tensor<64xf32>
    %advhs1b2g1 = stablehlo.divide %advns1b2g1, %adbc2s1b2g1 : tensor<64xf32>
    %adlrs1b2g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2g1 = stablehlo.sqrt %advhs1b2g1 : tensor<64xf32>
    %addens1b2g1 = stablehlo.add %adsqs1b2g1, %adepss1b2g1 : tensor<64xf32>
    %adrats1b2g1 = stablehlo.divide %admhs1b2g1, %addens1b2g1 : tensor<64xf32>
    %adsts1b2g1 = stablehlo.multiply %adlrs1b2g1, %adrats1b2g1 : tensor<64xf32>
    %adsubs1b2g1 = stablehlo.subtract %s1b2g1, %adsts1b2g1 : tensor<64xf32>
    %adwds1b2g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2g1 = stablehlo.multiply %adwds1b2g1, %adlrs1b2g1 : tensor<64xf32>
    %adwdps1b2g1 = stablehlo.multiply %adwdlrs1b2g1, %s1b2g1 : tensor<64xf32>
    %adnews1b2g1 = stablehlo.subtract %adsubs1b2g1, %adwdps1b2g1 : tensor<64xf32>
    %adb1s1b2bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2bt1 = stablehlo.multiply %adb1s1b2bt1, %s1b2bt1m : tensor<64xf32>
    %admgs1b2bt1 = stablehlo.multiply %adob1s1b2bt1, %s1b2dn1db : tensor<64xf32>
    %admns1b2bt1 = stablehlo.add %admss1b2bt1, %admgs1b2bt1 : tensor<64xf32>
    %adb2s1b2bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2bt1 = stablehlo.multiply %adb2s1b2bt1, %s1b2bt1v : tensor<64xf32>
    %adg2s1b2bt1 = stablehlo.multiply %s1b2dn1db, %s1b2dn1db : tensor<64xf32>
    %advgs1b2bt1 = stablehlo.multiply %adob2s1b2bt1, %adg2s1b2bt1 : tensor<64xf32>
    %advns1b2bt1 = stablehlo.add %advss1b2bt1, %advgs1b2bt1 : tensor<64xf32>
    %adbc1s1b2bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2bt1 = stablehlo.divide %admns1b2bt1, %adbc1s1b2bt1 : tensor<64xf32>
    %advhs1b2bt1 = stablehlo.divide %advns1b2bt1, %adbc2s1b2bt1 : tensor<64xf32>
    %adlrs1b2bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2bt1 = stablehlo.sqrt %advhs1b2bt1 : tensor<64xf32>
    %addens1b2bt1 = stablehlo.add %adsqs1b2bt1, %adepss1b2bt1 : tensor<64xf32>
    %adrats1b2bt1 = stablehlo.divide %admhs1b2bt1, %addens1b2bt1 : tensor<64xf32>
    %adsts1b2bt1 = stablehlo.multiply %adlrs1b2bt1, %adrats1b2bt1 : tensor<64xf32>
    %adsubs1b2bt1 = stablehlo.subtract %s1b2bt1, %adsts1b2bt1 : tensor<64xf32>
    %adwds1b2bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2bt1 = stablehlo.multiply %adwds1b2bt1, %adlrs1b2bt1 : tensor<64xf32>
    %adwdps1b2bt1 = stablehlo.multiply %adwdlrs1b2bt1, %s1b2bt1 : tensor<64xf32>
    %adnews1b2bt1 = stablehlo.subtract %adsubs1b2bt1, %adwdps1b2bt1 : tensor<64xf32>
    %adb1s1b2W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob1s1b2W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admss1b2W2 = stablehlo.multiply %adb1s1b2W2, %s1b2W2m : tensor<64x64x3x3xf32>
    %admgs1b2W2 = stablehlo.multiply %adob1s1b2W2, %s1b2dW2 : tensor<64x64x3x3xf32>
    %admns1b2W2 = stablehlo.add %admss1b2W2, %admgs1b2W2 : tensor<64x64x3x3xf32>
    %adb2s1b2W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adob2s1b2W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %advss1b2W2 = stablehlo.multiply %adb2s1b2W2, %s1b2W2v : tensor<64x64x3x3xf32>
    %adg2s1b2W2 = stablehlo.multiply %s1b2dW2, %s1b2dW2 : tensor<64x64x3x3xf32>
    %advgs1b2W2 = stablehlo.multiply %adob2s1b2W2, %adg2s1b2W2 : tensor<64x64x3x3xf32>
    %advns1b2W2 = stablehlo.add %advss1b2W2, %advgs1b2W2 : tensor<64x64x3x3xf32>
    %adbc1s1b2W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adbc2s1b2W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %admhs1b2W2 = stablehlo.divide %admns1b2W2, %adbc1s1b2W2 : tensor<64x64x3x3xf32>
    %advhs1b2W2 = stablehlo.divide %advns1b2W2, %adbc2s1b2W2 : tensor<64x64x3x3xf32>
    %adlrs1b2W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adepss1b2W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adsqs1b2W2 = stablehlo.sqrt %advhs1b2W2 : tensor<64x64x3x3xf32>
    %addens1b2W2 = stablehlo.add %adsqs1b2W2, %adepss1b2W2 : tensor<64x64x3x3xf32>
    %adrats1b2W2 = stablehlo.divide %admhs1b2W2, %addens1b2W2 : tensor<64x64x3x3xf32>
    %adsts1b2W2 = stablehlo.multiply %adlrs1b2W2, %adrats1b2W2 : tensor<64x64x3x3xf32>
    %adsubs1b2W2 = stablehlo.subtract %s1b2W2, %adsts1b2W2 : tensor<64x64x3x3xf32>
    %adwds1b2W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %adwdlrs1b2W2 = stablehlo.multiply %adwds1b2W2, %adlrs1b2W2 : tensor<64x64x3x3xf32>
    %adwdps1b2W2 = stablehlo.multiply %adwdlrs1b2W2, %s1b2W2 : tensor<64x64x3x3xf32>
    %adnews1b2W2 = stablehlo.subtract %adsubs1b2W2, %adwdps1b2W2 : tensor<64x64x3x3xf32>
    %adb1s1b2b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2b2 = stablehlo.multiply %adb1s1b2b2, %s1b2b2m : tensor<64xf32>
    %admgs1b2b2 = stablehlo.multiply %adob1s1b2b2, %s1b2db2 : tensor<64xf32>
    %admns1b2b2 = stablehlo.add %admss1b2b2, %admgs1b2b2 : tensor<64xf32>
    %adb2s1b2b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2b2 = stablehlo.multiply %adb2s1b2b2, %s1b2b2v : tensor<64xf32>
    %adg2s1b2b2 = stablehlo.multiply %s1b2db2, %s1b2db2 : tensor<64xf32>
    %advgs1b2b2 = stablehlo.multiply %adob2s1b2b2, %adg2s1b2b2 : tensor<64xf32>
    %advns1b2b2 = stablehlo.add %advss1b2b2, %advgs1b2b2 : tensor<64xf32>
    %adbc1s1b2b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2b2 = stablehlo.divide %admns1b2b2, %adbc1s1b2b2 : tensor<64xf32>
    %advhs1b2b2 = stablehlo.divide %advns1b2b2, %adbc2s1b2b2 : tensor<64xf32>
    %adlrs1b2b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2b2 = stablehlo.sqrt %advhs1b2b2 : tensor<64xf32>
    %addens1b2b2 = stablehlo.add %adsqs1b2b2, %adepss1b2b2 : tensor<64xf32>
    %adrats1b2b2 = stablehlo.divide %admhs1b2b2, %addens1b2b2 : tensor<64xf32>
    %adsts1b2b2 = stablehlo.multiply %adlrs1b2b2, %adrats1b2b2 : tensor<64xf32>
    %adsubs1b2b2 = stablehlo.subtract %s1b2b2, %adsts1b2b2 : tensor<64xf32>
    %adwds1b2b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2b2 = stablehlo.multiply %adwds1b2b2, %adlrs1b2b2 : tensor<64xf32>
    %adwdps1b2b2 = stablehlo.multiply %adwdlrs1b2b2, %s1b2b2 : tensor<64xf32>
    %adnews1b2b2 = stablehlo.subtract %adsubs1b2b2, %adwdps1b2b2 : tensor<64xf32>
    %adb1s1b2g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2g2 = stablehlo.multiply %adb1s1b2g2, %s1b2g2m : tensor<64xf32>
    %admgs1b2g2 = stablehlo.multiply %adob1s1b2g2, %s1b2dn2dg : tensor<64xf32>
    %admns1b2g2 = stablehlo.add %admss1b2g2, %admgs1b2g2 : tensor<64xf32>
    %adb2s1b2g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2g2 = stablehlo.multiply %adb2s1b2g2, %s1b2g2v : tensor<64xf32>
    %adg2s1b2g2 = stablehlo.multiply %s1b2dn2dg, %s1b2dn2dg : tensor<64xf32>
    %advgs1b2g2 = stablehlo.multiply %adob2s1b2g2, %adg2s1b2g2 : tensor<64xf32>
    %advns1b2g2 = stablehlo.add %advss1b2g2, %advgs1b2g2 : tensor<64xf32>
    %adbc1s1b2g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2g2 = stablehlo.divide %admns1b2g2, %adbc1s1b2g2 : tensor<64xf32>
    %advhs1b2g2 = stablehlo.divide %advns1b2g2, %adbc2s1b2g2 : tensor<64xf32>
    %adlrs1b2g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2g2 = stablehlo.sqrt %advhs1b2g2 : tensor<64xf32>
    %addens1b2g2 = stablehlo.add %adsqs1b2g2, %adepss1b2g2 : tensor<64xf32>
    %adrats1b2g2 = stablehlo.divide %admhs1b2g2, %addens1b2g2 : tensor<64xf32>
    %adsts1b2g2 = stablehlo.multiply %adlrs1b2g2, %adrats1b2g2 : tensor<64xf32>
    %adsubs1b2g2 = stablehlo.subtract %s1b2g2, %adsts1b2g2 : tensor<64xf32>
    %adwds1b2g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2g2 = stablehlo.multiply %adwds1b2g2, %adlrs1b2g2 : tensor<64xf32>
    %adwdps1b2g2 = stablehlo.multiply %adwdlrs1b2g2, %s1b2g2 : tensor<64xf32>
    %adnews1b2g2 = stablehlo.subtract %adsubs1b2g2, %adwdps1b2g2 : tensor<64xf32>
    %adb1s1b2bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob1s1b2bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admss1b2bt2 = stablehlo.multiply %adb1s1b2bt2, %s1b2bt2m : tensor<64xf32>
    %admgs1b2bt2 = stablehlo.multiply %adob1s1b2bt2, %s1b2dn2db : tensor<64xf32>
    %admns1b2bt2 = stablehlo.add %admss1b2bt2, %admgs1b2bt2 : tensor<64xf32>
    %adb2s1b2bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adob2s1b2bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %advss1b2bt2 = stablehlo.multiply %adb2s1b2bt2, %s1b2bt2v : tensor<64xf32>
    %adg2s1b2bt2 = stablehlo.multiply %s1b2dn2db, %s1b2dn2db : tensor<64xf32>
    %advgs1b2bt2 = stablehlo.multiply %adob2s1b2bt2, %adg2s1b2bt2 : tensor<64xf32>
    %advns1b2bt2 = stablehlo.add %advss1b2bt2, %advgs1b2bt2 : tensor<64xf32>
    %adbc1s1b2bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adbc2s1b2bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %admhs1b2bt2 = stablehlo.divide %admns1b2bt2, %adbc1s1b2bt2 : tensor<64xf32>
    %advhs1b2bt2 = stablehlo.divide %advns1b2bt2, %adbc2s1b2bt2 : tensor<64xf32>
    %adlrs1b2bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adepss1b2bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adsqs1b2bt2 = stablehlo.sqrt %advhs1b2bt2 : tensor<64xf32>
    %addens1b2bt2 = stablehlo.add %adsqs1b2bt2, %adepss1b2bt2 : tensor<64xf32>
    %adrats1b2bt2 = stablehlo.divide %admhs1b2bt2, %addens1b2bt2 : tensor<64xf32>
    %adsts1b2bt2 = stablehlo.multiply %adlrs1b2bt2, %adrats1b2bt2 : tensor<64xf32>
    %adsubs1b2bt2 = stablehlo.subtract %s1b2bt2, %adsts1b2bt2 : tensor<64xf32>
    %adwds1b2bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %adwdlrs1b2bt2 = stablehlo.multiply %adwds1b2bt2, %adlrs1b2bt2 : tensor<64xf32>
    %adwdps1b2bt2 = stablehlo.multiply %adwdlrs1b2bt2, %s1b2bt2 : tensor<64xf32>
    %adnews1b2bt2 = stablehlo.subtract %adsubs1b2bt2, %adwdps1b2bt2 : tensor<64xf32>
    %adb1d2W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adob1d2W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %admsd2W1 = stablehlo.multiply %adb1d2W1, %d2W1m : tensor<128x64x3x3xf32>
    %admgd2W1 = stablehlo.multiply %adob1d2W1, %d2dW1 : tensor<128x64x3x3xf32>
    %admnd2W1 = stablehlo.add %admsd2W1, %admgd2W1 : tensor<128x64x3x3xf32>
    %adb2d2W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adob2d2W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %advsd2W1 = stablehlo.multiply %adb2d2W1, %d2W1v : tensor<128x64x3x3xf32>
    %adg2d2W1 = stablehlo.multiply %d2dW1, %d2dW1 : tensor<128x64x3x3xf32>
    %advgd2W1 = stablehlo.multiply %adob2d2W1, %adg2d2W1 : tensor<128x64x3x3xf32>
    %advnd2W1 = stablehlo.add %advsd2W1, %advgd2W1 : tensor<128x64x3x3xf32>
    %adbc1d2W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adbc2d2W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %admhd2W1 = stablehlo.divide %admnd2W1, %adbc1d2W1 : tensor<128x64x3x3xf32>
    %advhd2W1 = stablehlo.divide %advnd2W1, %adbc2d2W1 : tensor<128x64x3x3xf32>
    %adlrd2W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adepsd2W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adsqd2W1 = stablehlo.sqrt %advhd2W1 : tensor<128x64x3x3xf32>
    %addend2W1 = stablehlo.add %adsqd2W1, %adepsd2W1 : tensor<128x64x3x3xf32>
    %adratd2W1 = stablehlo.divide %admhd2W1, %addend2W1 : tensor<128x64x3x3xf32>
    %adstd2W1 = stablehlo.multiply %adlrd2W1, %adratd2W1 : tensor<128x64x3x3xf32>
    %adsubd2W1 = stablehlo.subtract %d2W1, %adstd2W1 : tensor<128x64x3x3xf32>
    %adwdd2W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adwdlrd2W1 = stablehlo.multiply %adwdd2W1, %adlrd2W1 : tensor<128x64x3x3xf32>
    %adwdpd2W1 = stablehlo.multiply %adwdlrd2W1, %d2W1 : tensor<128x64x3x3xf32>
    %adnewd2W1 = stablehlo.subtract %adsubd2W1, %adwdpd2W1 : tensor<128x64x3x3xf32>
    %adb1d2b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2b1 = stablehlo.multiply %adb1d2b1, %d2b1m : tensor<128xf32>
    %admgd2b1 = stablehlo.multiply %adob1d2b1, %d2db1 : tensor<128xf32>
    %admnd2b1 = stablehlo.add %admsd2b1, %admgd2b1 : tensor<128xf32>
    %adb2d2b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2b1 = stablehlo.multiply %adb2d2b1, %d2b1v : tensor<128xf32>
    %adg2d2b1 = stablehlo.multiply %d2db1, %d2db1 : tensor<128xf32>
    %advgd2b1 = stablehlo.multiply %adob2d2b1, %adg2d2b1 : tensor<128xf32>
    %advnd2b1 = stablehlo.add %advsd2b1, %advgd2b1 : tensor<128xf32>
    %adbc1d2b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2b1 = stablehlo.divide %admnd2b1, %adbc1d2b1 : tensor<128xf32>
    %advhd2b1 = stablehlo.divide %advnd2b1, %adbc2d2b1 : tensor<128xf32>
    %adlrd2b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2b1 = stablehlo.sqrt %advhd2b1 : tensor<128xf32>
    %addend2b1 = stablehlo.add %adsqd2b1, %adepsd2b1 : tensor<128xf32>
    %adratd2b1 = stablehlo.divide %admhd2b1, %addend2b1 : tensor<128xf32>
    %adstd2b1 = stablehlo.multiply %adlrd2b1, %adratd2b1 : tensor<128xf32>
    %adsubd2b1 = stablehlo.subtract %d2b1, %adstd2b1 : tensor<128xf32>
    %adwdd2b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2b1 = stablehlo.multiply %adwdd2b1, %adlrd2b1 : tensor<128xf32>
    %adwdpd2b1 = stablehlo.multiply %adwdlrd2b1, %d2b1 : tensor<128xf32>
    %adnewd2b1 = stablehlo.subtract %adsubd2b1, %adwdpd2b1 : tensor<128xf32>
    %adb1d2g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2g1 = stablehlo.multiply %adb1d2g1, %d2g1m : tensor<128xf32>
    %admgd2g1 = stablehlo.multiply %adob1d2g1, %d2dn1dg : tensor<128xf32>
    %admnd2g1 = stablehlo.add %admsd2g1, %admgd2g1 : tensor<128xf32>
    %adb2d2g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2g1 = stablehlo.multiply %adb2d2g1, %d2g1v : tensor<128xf32>
    %adg2d2g1 = stablehlo.multiply %d2dn1dg, %d2dn1dg : tensor<128xf32>
    %advgd2g1 = stablehlo.multiply %adob2d2g1, %adg2d2g1 : tensor<128xf32>
    %advnd2g1 = stablehlo.add %advsd2g1, %advgd2g1 : tensor<128xf32>
    %adbc1d2g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2g1 = stablehlo.divide %admnd2g1, %adbc1d2g1 : tensor<128xf32>
    %advhd2g1 = stablehlo.divide %advnd2g1, %adbc2d2g1 : tensor<128xf32>
    %adlrd2g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2g1 = stablehlo.sqrt %advhd2g1 : tensor<128xf32>
    %addend2g1 = stablehlo.add %adsqd2g1, %adepsd2g1 : tensor<128xf32>
    %adratd2g1 = stablehlo.divide %admhd2g1, %addend2g1 : tensor<128xf32>
    %adstd2g1 = stablehlo.multiply %adlrd2g1, %adratd2g1 : tensor<128xf32>
    %adsubd2g1 = stablehlo.subtract %d2g1, %adstd2g1 : tensor<128xf32>
    %adwdd2g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2g1 = stablehlo.multiply %adwdd2g1, %adlrd2g1 : tensor<128xf32>
    %adwdpd2g1 = stablehlo.multiply %adwdlrd2g1, %d2g1 : tensor<128xf32>
    %adnewd2g1 = stablehlo.subtract %adsubd2g1, %adwdpd2g1 : tensor<128xf32>
    %adb1d2bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2bt1 = stablehlo.multiply %adb1d2bt1, %d2bt1m : tensor<128xf32>
    %admgd2bt1 = stablehlo.multiply %adob1d2bt1, %d2dn1db : tensor<128xf32>
    %admnd2bt1 = stablehlo.add %admsd2bt1, %admgd2bt1 : tensor<128xf32>
    %adb2d2bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2bt1 = stablehlo.multiply %adb2d2bt1, %d2bt1v : tensor<128xf32>
    %adg2d2bt1 = stablehlo.multiply %d2dn1db, %d2dn1db : tensor<128xf32>
    %advgd2bt1 = stablehlo.multiply %adob2d2bt1, %adg2d2bt1 : tensor<128xf32>
    %advnd2bt1 = stablehlo.add %advsd2bt1, %advgd2bt1 : tensor<128xf32>
    %adbc1d2bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2bt1 = stablehlo.divide %admnd2bt1, %adbc1d2bt1 : tensor<128xf32>
    %advhd2bt1 = stablehlo.divide %advnd2bt1, %adbc2d2bt1 : tensor<128xf32>
    %adlrd2bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2bt1 = stablehlo.sqrt %advhd2bt1 : tensor<128xf32>
    %addend2bt1 = stablehlo.add %adsqd2bt1, %adepsd2bt1 : tensor<128xf32>
    %adratd2bt1 = stablehlo.divide %admhd2bt1, %addend2bt1 : tensor<128xf32>
    %adstd2bt1 = stablehlo.multiply %adlrd2bt1, %adratd2bt1 : tensor<128xf32>
    %adsubd2bt1 = stablehlo.subtract %d2bt1, %adstd2bt1 : tensor<128xf32>
    %adwdd2bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2bt1 = stablehlo.multiply %adwdd2bt1, %adlrd2bt1 : tensor<128xf32>
    %adwdpd2bt1 = stablehlo.multiply %adwdlrd2bt1, %d2bt1 : tensor<128xf32>
    %adnewd2bt1 = stablehlo.subtract %adsubd2bt1, %adwdpd2bt1 : tensor<128xf32>
    %adb1d2W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1d2W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admsd2W2 = stablehlo.multiply %adb1d2W2, %d2W2m : tensor<128x128x3x3xf32>
    %admgd2W2 = stablehlo.multiply %adob1d2W2, %d2dW2 : tensor<128x128x3x3xf32>
    %admnd2W2 = stablehlo.add %admsd2W2, %admgd2W2 : tensor<128x128x3x3xf32>
    %adb2d2W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2d2W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advsd2W2 = stablehlo.multiply %adb2d2W2, %d2W2v : tensor<128x128x3x3xf32>
    %adg2d2W2 = stablehlo.multiply %d2dW2, %d2dW2 : tensor<128x128x3x3xf32>
    %advgd2W2 = stablehlo.multiply %adob2d2W2, %adg2d2W2 : tensor<128x128x3x3xf32>
    %advnd2W2 = stablehlo.add %advsd2W2, %advgd2W2 : tensor<128x128x3x3xf32>
    %adbc1d2W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2d2W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhd2W2 = stablehlo.divide %admnd2W2, %adbc1d2W2 : tensor<128x128x3x3xf32>
    %advhd2W2 = stablehlo.divide %advnd2W2, %adbc2d2W2 : tensor<128x128x3x3xf32>
    %adlrd2W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepsd2W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqd2W2 = stablehlo.sqrt %advhd2W2 : tensor<128x128x3x3xf32>
    %addend2W2 = stablehlo.add %adsqd2W2, %adepsd2W2 : tensor<128x128x3x3xf32>
    %adratd2W2 = stablehlo.divide %admhd2W2, %addend2W2 : tensor<128x128x3x3xf32>
    %adstd2W2 = stablehlo.multiply %adlrd2W2, %adratd2W2 : tensor<128x128x3x3xf32>
    %adsubd2W2 = stablehlo.subtract %d2W2, %adstd2W2 : tensor<128x128x3x3xf32>
    %adwdd2W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrd2W2 = stablehlo.multiply %adwdd2W2, %adlrd2W2 : tensor<128x128x3x3xf32>
    %adwdpd2W2 = stablehlo.multiply %adwdlrd2W2, %d2W2 : tensor<128x128x3x3xf32>
    %adnewd2W2 = stablehlo.subtract %adsubd2W2, %adwdpd2W2 : tensor<128x128x3x3xf32>
    %adb1d2b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2b2 = stablehlo.multiply %adb1d2b2, %d2b2m : tensor<128xf32>
    %admgd2b2 = stablehlo.multiply %adob1d2b2, %d2db2 : tensor<128xf32>
    %admnd2b2 = stablehlo.add %admsd2b2, %admgd2b2 : tensor<128xf32>
    %adb2d2b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2b2 = stablehlo.multiply %adb2d2b2, %d2b2v : tensor<128xf32>
    %adg2d2b2 = stablehlo.multiply %d2db2, %d2db2 : tensor<128xf32>
    %advgd2b2 = stablehlo.multiply %adob2d2b2, %adg2d2b2 : tensor<128xf32>
    %advnd2b2 = stablehlo.add %advsd2b2, %advgd2b2 : tensor<128xf32>
    %adbc1d2b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2b2 = stablehlo.divide %admnd2b2, %adbc1d2b2 : tensor<128xf32>
    %advhd2b2 = stablehlo.divide %advnd2b2, %adbc2d2b2 : tensor<128xf32>
    %adlrd2b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2b2 = stablehlo.sqrt %advhd2b2 : tensor<128xf32>
    %addend2b2 = stablehlo.add %adsqd2b2, %adepsd2b2 : tensor<128xf32>
    %adratd2b2 = stablehlo.divide %admhd2b2, %addend2b2 : tensor<128xf32>
    %adstd2b2 = stablehlo.multiply %adlrd2b2, %adratd2b2 : tensor<128xf32>
    %adsubd2b2 = stablehlo.subtract %d2b2, %adstd2b2 : tensor<128xf32>
    %adwdd2b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2b2 = stablehlo.multiply %adwdd2b2, %adlrd2b2 : tensor<128xf32>
    %adwdpd2b2 = stablehlo.multiply %adwdlrd2b2, %d2b2 : tensor<128xf32>
    %adnewd2b2 = stablehlo.subtract %adsubd2b2, %adwdpd2b2 : tensor<128xf32>
    %adb1d2g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2g2 = stablehlo.multiply %adb1d2g2, %d2g2m : tensor<128xf32>
    %admgd2g2 = stablehlo.multiply %adob1d2g2, %d2dn2dg : tensor<128xf32>
    %admnd2g2 = stablehlo.add %admsd2g2, %admgd2g2 : tensor<128xf32>
    %adb2d2g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2g2 = stablehlo.multiply %adb2d2g2, %d2g2v : tensor<128xf32>
    %adg2d2g2 = stablehlo.multiply %d2dn2dg, %d2dn2dg : tensor<128xf32>
    %advgd2g2 = stablehlo.multiply %adob2d2g2, %adg2d2g2 : tensor<128xf32>
    %advnd2g2 = stablehlo.add %advsd2g2, %advgd2g2 : tensor<128xf32>
    %adbc1d2g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2g2 = stablehlo.divide %admnd2g2, %adbc1d2g2 : tensor<128xf32>
    %advhd2g2 = stablehlo.divide %advnd2g2, %adbc2d2g2 : tensor<128xf32>
    %adlrd2g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2g2 = stablehlo.sqrt %advhd2g2 : tensor<128xf32>
    %addend2g2 = stablehlo.add %adsqd2g2, %adepsd2g2 : tensor<128xf32>
    %adratd2g2 = stablehlo.divide %admhd2g2, %addend2g2 : tensor<128xf32>
    %adstd2g2 = stablehlo.multiply %adlrd2g2, %adratd2g2 : tensor<128xf32>
    %adsubd2g2 = stablehlo.subtract %d2g2, %adstd2g2 : tensor<128xf32>
    %adwdd2g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2g2 = stablehlo.multiply %adwdd2g2, %adlrd2g2 : tensor<128xf32>
    %adwdpd2g2 = stablehlo.multiply %adwdlrd2g2, %d2g2 : tensor<128xf32>
    %adnewd2g2 = stablehlo.subtract %adsubd2g2, %adwdpd2g2 : tensor<128xf32>
    %adb1d2bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2bt2 = stablehlo.multiply %adb1d2bt2, %d2bt2m : tensor<128xf32>
    %admgd2bt2 = stablehlo.multiply %adob1d2bt2, %d2dn2db : tensor<128xf32>
    %admnd2bt2 = stablehlo.add %admsd2bt2, %admgd2bt2 : tensor<128xf32>
    %adb2d2bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2bt2 = stablehlo.multiply %adb2d2bt2, %d2bt2v : tensor<128xf32>
    %adg2d2bt2 = stablehlo.multiply %d2dn2db, %d2dn2db : tensor<128xf32>
    %advgd2bt2 = stablehlo.multiply %adob2d2bt2, %adg2d2bt2 : tensor<128xf32>
    %advnd2bt2 = stablehlo.add %advsd2bt2, %advgd2bt2 : tensor<128xf32>
    %adbc1d2bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2bt2 = stablehlo.divide %admnd2bt2, %adbc1d2bt2 : tensor<128xf32>
    %advhd2bt2 = stablehlo.divide %advnd2bt2, %adbc2d2bt2 : tensor<128xf32>
    %adlrd2bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2bt2 = stablehlo.sqrt %advhd2bt2 : tensor<128xf32>
    %addend2bt2 = stablehlo.add %adsqd2bt2, %adepsd2bt2 : tensor<128xf32>
    %adratd2bt2 = stablehlo.divide %admhd2bt2, %addend2bt2 : tensor<128xf32>
    %adstd2bt2 = stablehlo.multiply %adlrd2bt2, %adratd2bt2 : tensor<128xf32>
    %adsubd2bt2 = stablehlo.subtract %d2bt2, %adstd2bt2 : tensor<128xf32>
    %adwdd2bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2bt2 = stablehlo.multiply %adwdd2bt2, %adlrd2bt2 : tensor<128xf32>
    %adwdpd2bt2 = stablehlo.multiply %adwdlrd2bt2, %d2bt2 : tensor<128xf32>
    %adnewd2bt2 = stablehlo.subtract %adsubd2bt2, %adwdpd2bt2 : tensor<128xf32>
    %adb1d2Wp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adob1d2Wp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %admsd2Wp = stablehlo.multiply %adb1d2Wp, %d2Wpm : tensor<128x64x3x3xf32>
    %admgd2Wp = stablehlo.multiply %adob1d2Wp, %d2dWp : tensor<128x64x3x3xf32>
    %admnd2Wp = stablehlo.add %admsd2Wp, %admgd2Wp : tensor<128x64x3x3xf32>
    %adb2d2Wp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adob2d2Wp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %advsd2Wp = stablehlo.multiply %adb2d2Wp, %d2Wpv : tensor<128x64x3x3xf32>
    %adg2d2Wp = stablehlo.multiply %d2dWp, %d2dWp : tensor<128x64x3x3xf32>
    %advgd2Wp = stablehlo.multiply %adob2d2Wp, %adg2d2Wp : tensor<128x64x3x3xf32>
    %advnd2Wp = stablehlo.add %advsd2Wp, %advgd2Wp : tensor<128x64x3x3xf32>
    %adbc1d2Wp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adbc2d2Wp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %admhd2Wp = stablehlo.divide %admnd2Wp, %adbc1d2Wp : tensor<128x64x3x3xf32>
    %advhd2Wp = stablehlo.divide %advnd2Wp, %adbc2d2Wp : tensor<128x64x3x3xf32>
    %adlrd2Wp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adepsd2Wp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adsqd2Wp = stablehlo.sqrt %advhd2Wp : tensor<128x64x3x3xf32>
    %addend2Wp = stablehlo.add %adsqd2Wp, %adepsd2Wp : tensor<128x64x3x3xf32>
    %adratd2Wp = stablehlo.divide %admhd2Wp, %addend2Wp : tensor<128x64x3x3xf32>
    %adstd2Wp = stablehlo.multiply %adlrd2Wp, %adratd2Wp : tensor<128x64x3x3xf32>
    %adsubd2Wp = stablehlo.subtract %d2Wp, %adstd2Wp : tensor<128x64x3x3xf32>
    %adwdd2Wp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %adwdlrd2Wp = stablehlo.multiply %adwdd2Wp, %adlrd2Wp : tensor<128x64x3x3xf32>
    %adwdpd2Wp = stablehlo.multiply %adwdlrd2Wp, %d2Wp : tensor<128x64x3x3xf32>
    %adnewd2Wp = stablehlo.subtract %adsubd2Wp, %adwdpd2Wp : tensor<128x64x3x3xf32>
    %adb1d2bp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2bp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2bp = stablehlo.multiply %adb1d2bp, %d2bpm : tensor<128xf32>
    %admgd2bp = stablehlo.multiply %adob1d2bp, %d2dbp : tensor<128xf32>
    %admnd2bp = stablehlo.add %admsd2bp, %admgd2bp : tensor<128xf32>
    %adb2d2bp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2bp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2bp = stablehlo.multiply %adb2d2bp, %d2bpv : tensor<128xf32>
    %adg2d2bp = stablehlo.multiply %d2dbp, %d2dbp : tensor<128xf32>
    %advgd2bp = stablehlo.multiply %adob2d2bp, %adg2d2bp : tensor<128xf32>
    %advnd2bp = stablehlo.add %advsd2bp, %advgd2bp : tensor<128xf32>
    %adbc1d2bp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2bp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2bp = stablehlo.divide %admnd2bp, %adbc1d2bp : tensor<128xf32>
    %advhd2bp = stablehlo.divide %advnd2bp, %adbc2d2bp : tensor<128xf32>
    %adlrd2bp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2bp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2bp = stablehlo.sqrt %advhd2bp : tensor<128xf32>
    %addend2bp = stablehlo.add %adsqd2bp, %adepsd2bp : tensor<128xf32>
    %adratd2bp = stablehlo.divide %admhd2bp, %addend2bp : tensor<128xf32>
    %adstd2bp = stablehlo.multiply %adlrd2bp, %adratd2bp : tensor<128xf32>
    %adsubd2bp = stablehlo.subtract %d2bp, %adstd2bp : tensor<128xf32>
    %adwdd2bp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2bp = stablehlo.multiply %adwdd2bp, %adlrd2bp : tensor<128xf32>
    %adwdpd2bp = stablehlo.multiply %adwdlrd2bp, %d2bp : tensor<128xf32>
    %adnewd2bp = stablehlo.subtract %adsubd2bp, %adwdpd2bp : tensor<128xf32>
    %adb1d2gp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2gp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2gp = stablehlo.multiply %adb1d2gp, %d2gpm : tensor<128xf32>
    %admgd2gp = stablehlo.multiply %adob1d2gp, %d2dnpdg : tensor<128xf32>
    %admnd2gp = stablehlo.add %admsd2gp, %admgd2gp : tensor<128xf32>
    %adb2d2gp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2gp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2gp = stablehlo.multiply %adb2d2gp, %d2gpv : tensor<128xf32>
    %adg2d2gp = stablehlo.multiply %d2dnpdg, %d2dnpdg : tensor<128xf32>
    %advgd2gp = stablehlo.multiply %adob2d2gp, %adg2d2gp : tensor<128xf32>
    %advnd2gp = stablehlo.add %advsd2gp, %advgd2gp : tensor<128xf32>
    %adbc1d2gp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2gp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2gp = stablehlo.divide %admnd2gp, %adbc1d2gp : tensor<128xf32>
    %advhd2gp = stablehlo.divide %advnd2gp, %adbc2d2gp : tensor<128xf32>
    %adlrd2gp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2gp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2gp = stablehlo.sqrt %advhd2gp : tensor<128xf32>
    %addend2gp = stablehlo.add %adsqd2gp, %adepsd2gp : tensor<128xf32>
    %adratd2gp = stablehlo.divide %admhd2gp, %addend2gp : tensor<128xf32>
    %adstd2gp = stablehlo.multiply %adlrd2gp, %adratd2gp : tensor<128xf32>
    %adsubd2gp = stablehlo.subtract %d2gp, %adstd2gp : tensor<128xf32>
    %adwdd2gp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2gp = stablehlo.multiply %adwdd2gp, %adlrd2gp : tensor<128xf32>
    %adwdpd2gp = stablehlo.multiply %adwdlrd2gp, %d2gp : tensor<128xf32>
    %adnewd2gp = stablehlo.subtract %adsubd2gp, %adwdpd2gp : tensor<128xf32>
    %adb1d2btp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1d2btp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admsd2btp = stablehlo.multiply %adb1d2btp, %d2btpm : tensor<128xf32>
    %admgd2btp = stablehlo.multiply %adob1d2btp, %d2dnpdb : tensor<128xf32>
    %admnd2btp = stablehlo.add %admsd2btp, %admgd2btp : tensor<128xf32>
    %adb2d2btp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2d2btp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advsd2btp = stablehlo.multiply %adb2d2btp, %d2btpv : tensor<128xf32>
    %adg2d2btp = stablehlo.multiply %d2dnpdb, %d2dnpdb : tensor<128xf32>
    %advgd2btp = stablehlo.multiply %adob2d2btp, %adg2d2btp : tensor<128xf32>
    %advnd2btp = stablehlo.add %advsd2btp, %advgd2btp : tensor<128xf32>
    %adbc1d2btp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2d2btp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhd2btp = stablehlo.divide %admnd2btp, %adbc1d2btp : tensor<128xf32>
    %advhd2btp = stablehlo.divide %advnd2btp, %adbc2d2btp : tensor<128xf32>
    %adlrd2btp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepsd2btp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqd2btp = stablehlo.sqrt %advhd2btp : tensor<128xf32>
    %addend2btp = stablehlo.add %adsqd2btp, %adepsd2btp : tensor<128xf32>
    %adratd2btp = stablehlo.divide %admhd2btp, %addend2btp : tensor<128xf32>
    %adstd2btp = stablehlo.multiply %adlrd2btp, %adratd2btp : tensor<128xf32>
    %adsubd2btp = stablehlo.subtract %d2btp, %adstd2btp : tensor<128xf32>
    %adwdd2btp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrd2btp = stablehlo.multiply %adwdd2btp, %adlrd2btp : tensor<128xf32>
    %adwdpd2btp = stablehlo.multiply %adwdlrd2btp, %d2btp : tensor<128xf32>
    %adnewd2btp = stablehlo.subtract %adsubd2btp, %adwdpd2btp : tensor<128xf32>
    %adb1s2b0W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b0W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b0W1 = stablehlo.multiply %adb1s2b0W1, %s2b0W1m : tensor<128x128x3x3xf32>
    %admgs2b0W1 = stablehlo.multiply %adob1s2b0W1, %s2b0dW1 : tensor<128x128x3x3xf32>
    %admns2b0W1 = stablehlo.add %admss2b0W1, %admgs2b0W1 : tensor<128x128x3x3xf32>
    %adb2s2b0W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b0W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b0W1 = stablehlo.multiply %adb2s2b0W1, %s2b0W1v : tensor<128x128x3x3xf32>
    %adg2s2b0W1 = stablehlo.multiply %s2b0dW1, %s2b0dW1 : tensor<128x128x3x3xf32>
    %advgs2b0W1 = stablehlo.multiply %adob2s2b0W1, %adg2s2b0W1 : tensor<128x128x3x3xf32>
    %advns2b0W1 = stablehlo.add %advss2b0W1, %advgs2b0W1 : tensor<128x128x3x3xf32>
    %adbc1s2b0W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b0W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b0W1 = stablehlo.divide %admns2b0W1, %adbc1s2b0W1 : tensor<128x128x3x3xf32>
    %advhs2b0W1 = stablehlo.divide %advns2b0W1, %adbc2s2b0W1 : tensor<128x128x3x3xf32>
    %adlrs2b0W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b0W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b0W1 = stablehlo.sqrt %advhs2b0W1 : tensor<128x128x3x3xf32>
    %addens2b0W1 = stablehlo.add %adsqs2b0W1, %adepss2b0W1 : tensor<128x128x3x3xf32>
    %adrats2b0W1 = stablehlo.divide %admhs2b0W1, %addens2b0W1 : tensor<128x128x3x3xf32>
    %adsts2b0W1 = stablehlo.multiply %adlrs2b0W1, %adrats2b0W1 : tensor<128x128x3x3xf32>
    %adsubs2b0W1 = stablehlo.subtract %s2b0W1, %adsts2b0W1 : tensor<128x128x3x3xf32>
    %adwds2b0W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b0W1 = stablehlo.multiply %adwds2b0W1, %adlrs2b0W1 : tensor<128x128x3x3xf32>
    %adwdps2b0W1 = stablehlo.multiply %adwdlrs2b0W1, %s2b0W1 : tensor<128x128x3x3xf32>
    %adnews2b0W1 = stablehlo.subtract %adsubs2b0W1, %adwdps2b0W1 : tensor<128x128x3x3xf32>
    %adb1s2b0b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0b1 = stablehlo.multiply %adb1s2b0b1, %s2b0b1m : tensor<128xf32>
    %admgs2b0b1 = stablehlo.multiply %adob1s2b0b1, %s2b0db1 : tensor<128xf32>
    %admns2b0b1 = stablehlo.add %admss2b0b1, %admgs2b0b1 : tensor<128xf32>
    %adb2s2b0b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0b1 = stablehlo.multiply %adb2s2b0b1, %s2b0b1v : tensor<128xf32>
    %adg2s2b0b1 = stablehlo.multiply %s2b0db1, %s2b0db1 : tensor<128xf32>
    %advgs2b0b1 = stablehlo.multiply %adob2s2b0b1, %adg2s2b0b1 : tensor<128xf32>
    %advns2b0b1 = stablehlo.add %advss2b0b1, %advgs2b0b1 : tensor<128xf32>
    %adbc1s2b0b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0b1 = stablehlo.divide %admns2b0b1, %adbc1s2b0b1 : tensor<128xf32>
    %advhs2b0b1 = stablehlo.divide %advns2b0b1, %adbc2s2b0b1 : tensor<128xf32>
    %adlrs2b0b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0b1 = stablehlo.sqrt %advhs2b0b1 : tensor<128xf32>
    %addens2b0b1 = stablehlo.add %adsqs2b0b1, %adepss2b0b1 : tensor<128xf32>
    %adrats2b0b1 = stablehlo.divide %admhs2b0b1, %addens2b0b1 : tensor<128xf32>
    %adsts2b0b1 = stablehlo.multiply %adlrs2b0b1, %adrats2b0b1 : tensor<128xf32>
    %adsubs2b0b1 = stablehlo.subtract %s2b0b1, %adsts2b0b1 : tensor<128xf32>
    %adwds2b0b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0b1 = stablehlo.multiply %adwds2b0b1, %adlrs2b0b1 : tensor<128xf32>
    %adwdps2b0b1 = stablehlo.multiply %adwdlrs2b0b1, %s2b0b1 : tensor<128xf32>
    %adnews2b0b1 = stablehlo.subtract %adsubs2b0b1, %adwdps2b0b1 : tensor<128xf32>
    %adb1s2b0g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0g1 = stablehlo.multiply %adb1s2b0g1, %s2b0g1m : tensor<128xf32>
    %admgs2b0g1 = stablehlo.multiply %adob1s2b0g1, %s2b0dn1dg : tensor<128xf32>
    %admns2b0g1 = stablehlo.add %admss2b0g1, %admgs2b0g1 : tensor<128xf32>
    %adb2s2b0g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0g1 = stablehlo.multiply %adb2s2b0g1, %s2b0g1v : tensor<128xf32>
    %adg2s2b0g1 = stablehlo.multiply %s2b0dn1dg, %s2b0dn1dg : tensor<128xf32>
    %advgs2b0g1 = stablehlo.multiply %adob2s2b0g1, %adg2s2b0g1 : tensor<128xf32>
    %advns2b0g1 = stablehlo.add %advss2b0g1, %advgs2b0g1 : tensor<128xf32>
    %adbc1s2b0g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0g1 = stablehlo.divide %admns2b0g1, %adbc1s2b0g1 : tensor<128xf32>
    %advhs2b0g1 = stablehlo.divide %advns2b0g1, %adbc2s2b0g1 : tensor<128xf32>
    %adlrs2b0g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0g1 = stablehlo.sqrt %advhs2b0g1 : tensor<128xf32>
    %addens2b0g1 = stablehlo.add %adsqs2b0g1, %adepss2b0g1 : tensor<128xf32>
    %adrats2b0g1 = stablehlo.divide %admhs2b0g1, %addens2b0g1 : tensor<128xf32>
    %adsts2b0g1 = stablehlo.multiply %adlrs2b0g1, %adrats2b0g1 : tensor<128xf32>
    %adsubs2b0g1 = stablehlo.subtract %s2b0g1, %adsts2b0g1 : tensor<128xf32>
    %adwds2b0g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0g1 = stablehlo.multiply %adwds2b0g1, %adlrs2b0g1 : tensor<128xf32>
    %adwdps2b0g1 = stablehlo.multiply %adwdlrs2b0g1, %s2b0g1 : tensor<128xf32>
    %adnews2b0g1 = stablehlo.subtract %adsubs2b0g1, %adwdps2b0g1 : tensor<128xf32>
    %adb1s2b0bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0bt1 = stablehlo.multiply %adb1s2b0bt1, %s2b0bt1m : tensor<128xf32>
    %admgs2b0bt1 = stablehlo.multiply %adob1s2b0bt1, %s2b0dn1db : tensor<128xf32>
    %admns2b0bt1 = stablehlo.add %admss2b0bt1, %admgs2b0bt1 : tensor<128xf32>
    %adb2s2b0bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0bt1 = stablehlo.multiply %adb2s2b0bt1, %s2b0bt1v : tensor<128xf32>
    %adg2s2b0bt1 = stablehlo.multiply %s2b0dn1db, %s2b0dn1db : tensor<128xf32>
    %advgs2b0bt1 = stablehlo.multiply %adob2s2b0bt1, %adg2s2b0bt1 : tensor<128xf32>
    %advns2b0bt1 = stablehlo.add %advss2b0bt1, %advgs2b0bt1 : tensor<128xf32>
    %adbc1s2b0bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0bt1 = stablehlo.divide %admns2b0bt1, %adbc1s2b0bt1 : tensor<128xf32>
    %advhs2b0bt1 = stablehlo.divide %advns2b0bt1, %adbc2s2b0bt1 : tensor<128xf32>
    %adlrs2b0bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0bt1 = stablehlo.sqrt %advhs2b0bt1 : tensor<128xf32>
    %addens2b0bt1 = stablehlo.add %adsqs2b0bt1, %adepss2b0bt1 : tensor<128xf32>
    %adrats2b0bt1 = stablehlo.divide %admhs2b0bt1, %addens2b0bt1 : tensor<128xf32>
    %adsts2b0bt1 = stablehlo.multiply %adlrs2b0bt1, %adrats2b0bt1 : tensor<128xf32>
    %adsubs2b0bt1 = stablehlo.subtract %s2b0bt1, %adsts2b0bt1 : tensor<128xf32>
    %adwds2b0bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0bt1 = stablehlo.multiply %adwds2b0bt1, %adlrs2b0bt1 : tensor<128xf32>
    %adwdps2b0bt1 = stablehlo.multiply %adwdlrs2b0bt1, %s2b0bt1 : tensor<128xf32>
    %adnews2b0bt1 = stablehlo.subtract %adsubs2b0bt1, %adwdps2b0bt1 : tensor<128xf32>
    %adb1s2b0W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b0W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b0W2 = stablehlo.multiply %adb1s2b0W2, %s2b0W2m : tensor<128x128x3x3xf32>
    %admgs2b0W2 = stablehlo.multiply %adob1s2b0W2, %s2b0dW2 : tensor<128x128x3x3xf32>
    %admns2b0W2 = stablehlo.add %admss2b0W2, %admgs2b0W2 : tensor<128x128x3x3xf32>
    %adb2s2b0W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b0W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b0W2 = stablehlo.multiply %adb2s2b0W2, %s2b0W2v : tensor<128x128x3x3xf32>
    %adg2s2b0W2 = stablehlo.multiply %s2b0dW2, %s2b0dW2 : tensor<128x128x3x3xf32>
    %advgs2b0W2 = stablehlo.multiply %adob2s2b0W2, %adg2s2b0W2 : tensor<128x128x3x3xf32>
    %advns2b0W2 = stablehlo.add %advss2b0W2, %advgs2b0W2 : tensor<128x128x3x3xf32>
    %adbc1s2b0W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b0W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b0W2 = stablehlo.divide %admns2b0W2, %adbc1s2b0W2 : tensor<128x128x3x3xf32>
    %advhs2b0W2 = stablehlo.divide %advns2b0W2, %adbc2s2b0W2 : tensor<128x128x3x3xf32>
    %adlrs2b0W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b0W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b0W2 = stablehlo.sqrt %advhs2b0W2 : tensor<128x128x3x3xf32>
    %addens2b0W2 = stablehlo.add %adsqs2b0W2, %adepss2b0W2 : tensor<128x128x3x3xf32>
    %adrats2b0W2 = stablehlo.divide %admhs2b0W2, %addens2b0W2 : tensor<128x128x3x3xf32>
    %adsts2b0W2 = stablehlo.multiply %adlrs2b0W2, %adrats2b0W2 : tensor<128x128x3x3xf32>
    %adsubs2b0W2 = stablehlo.subtract %s2b0W2, %adsts2b0W2 : tensor<128x128x3x3xf32>
    %adwds2b0W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b0W2 = stablehlo.multiply %adwds2b0W2, %adlrs2b0W2 : tensor<128x128x3x3xf32>
    %adwdps2b0W2 = stablehlo.multiply %adwdlrs2b0W2, %s2b0W2 : tensor<128x128x3x3xf32>
    %adnews2b0W2 = stablehlo.subtract %adsubs2b0W2, %adwdps2b0W2 : tensor<128x128x3x3xf32>
    %adb1s2b0b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0b2 = stablehlo.multiply %adb1s2b0b2, %s2b0b2m : tensor<128xf32>
    %admgs2b0b2 = stablehlo.multiply %adob1s2b0b2, %s2b0db2 : tensor<128xf32>
    %admns2b0b2 = stablehlo.add %admss2b0b2, %admgs2b0b2 : tensor<128xf32>
    %adb2s2b0b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0b2 = stablehlo.multiply %adb2s2b0b2, %s2b0b2v : tensor<128xf32>
    %adg2s2b0b2 = stablehlo.multiply %s2b0db2, %s2b0db2 : tensor<128xf32>
    %advgs2b0b2 = stablehlo.multiply %adob2s2b0b2, %adg2s2b0b2 : tensor<128xf32>
    %advns2b0b2 = stablehlo.add %advss2b0b2, %advgs2b0b2 : tensor<128xf32>
    %adbc1s2b0b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0b2 = stablehlo.divide %admns2b0b2, %adbc1s2b0b2 : tensor<128xf32>
    %advhs2b0b2 = stablehlo.divide %advns2b0b2, %adbc2s2b0b2 : tensor<128xf32>
    %adlrs2b0b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0b2 = stablehlo.sqrt %advhs2b0b2 : tensor<128xf32>
    %addens2b0b2 = stablehlo.add %adsqs2b0b2, %adepss2b0b2 : tensor<128xf32>
    %adrats2b0b2 = stablehlo.divide %admhs2b0b2, %addens2b0b2 : tensor<128xf32>
    %adsts2b0b2 = stablehlo.multiply %adlrs2b0b2, %adrats2b0b2 : tensor<128xf32>
    %adsubs2b0b2 = stablehlo.subtract %s2b0b2, %adsts2b0b2 : tensor<128xf32>
    %adwds2b0b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0b2 = stablehlo.multiply %adwds2b0b2, %adlrs2b0b2 : tensor<128xf32>
    %adwdps2b0b2 = stablehlo.multiply %adwdlrs2b0b2, %s2b0b2 : tensor<128xf32>
    %adnews2b0b2 = stablehlo.subtract %adsubs2b0b2, %adwdps2b0b2 : tensor<128xf32>
    %adb1s2b0g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0g2 = stablehlo.multiply %adb1s2b0g2, %s2b0g2m : tensor<128xf32>
    %admgs2b0g2 = stablehlo.multiply %adob1s2b0g2, %s2b0dn2dg : tensor<128xf32>
    %admns2b0g2 = stablehlo.add %admss2b0g2, %admgs2b0g2 : tensor<128xf32>
    %adb2s2b0g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0g2 = stablehlo.multiply %adb2s2b0g2, %s2b0g2v : tensor<128xf32>
    %adg2s2b0g2 = stablehlo.multiply %s2b0dn2dg, %s2b0dn2dg : tensor<128xf32>
    %advgs2b0g2 = stablehlo.multiply %adob2s2b0g2, %adg2s2b0g2 : tensor<128xf32>
    %advns2b0g2 = stablehlo.add %advss2b0g2, %advgs2b0g2 : tensor<128xf32>
    %adbc1s2b0g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0g2 = stablehlo.divide %admns2b0g2, %adbc1s2b0g2 : tensor<128xf32>
    %advhs2b0g2 = stablehlo.divide %advns2b0g2, %adbc2s2b0g2 : tensor<128xf32>
    %adlrs2b0g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0g2 = stablehlo.sqrt %advhs2b0g2 : tensor<128xf32>
    %addens2b0g2 = stablehlo.add %adsqs2b0g2, %adepss2b0g2 : tensor<128xf32>
    %adrats2b0g2 = stablehlo.divide %admhs2b0g2, %addens2b0g2 : tensor<128xf32>
    %adsts2b0g2 = stablehlo.multiply %adlrs2b0g2, %adrats2b0g2 : tensor<128xf32>
    %adsubs2b0g2 = stablehlo.subtract %s2b0g2, %adsts2b0g2 : tensor<128xf32>
    %adwds2b0g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0g2 = stablehlo.multiply %adwds2b0g2, %adlrs2b0g2 : tensor<128xf32>
    %adwdps2b0g2 = stablehlo.multiply %adwdlrs2b0g2, %s2b0g2 : tensor<128xf32>
    %adnews2b0g2 = stablehlo.subtract %adsubs2b0g2, %adwdps2b0g2 : tensor<128xf32>
    %adb1s2b0bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b0bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b0bt2 = stablehlo.multiply %adb1s2b0bt2, %s2b0bt2m : tensor<128xf32>
    %admgs2b0bt2 = stablehlo.multiply %adob1s2b0bt2, %s2b0dn2db : tensor<128xf32>
    %admns2b0bt2 = stablehlo.add %admss2b0bt2, %admgs2b0bt2 : tensor<128xf32>
    %adb2s2b0bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b0bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b0bt2 = stablehlo.multiply %adb2s2b0bt2, %s2b0bt2v : tensor<128xf32>
    %adg2s2b0bt2 = stablehlo.multiply %s2b0dn2db, %s2b0dn2db : tensor<128xf32>
    %advgs2b0bt2 = stablehlo.multiply %adob2s2b0bt2, %adg2s2b0bt2 : tensor<128xf32>
    %advns2b0bt2 = stablehlo.add %advss2b0bt2, %advgs2b0bt2 : tensor<128xf32>
    %adbc1s2b0bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b0bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b0bt2 = stablehlo.divide %admns2b0bt2, %adbc1s2b0bt2 : tensor<128xf32>
    %advhs2b0bt2 = stablehlo.divide %advns2b0bt2, %adbc2s2b0bt2 : tensor<128xf32>
    %adlrs2b0bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b0bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b0bt2 = stablehlo.sqrt %advhs2b0bt2 : tensor<128xf32>
    %addens2b0bt2 = stablehlo.add %adsqs2b0bt2, %adepss2b0bt2 : tensor<128xf32>
    %adrats2b0bt2 = stablehlo.divide %admhs2b0bt2, %addens2b0bt2 : tensor<128xf32>
    %adsts2b0bt2 = stablehlo.multiply %adlrs2b0bt2, %adrats2b0bt2 : tensor<128xf32>
    %adsubs2b0bt2 = stablehlo.subtract %s2b0bt2, %adsts2b0bt2 : tensor<128xf32>
    %adwds2b0bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b0bt2 = stablehlo.multiply %adwds2b0bt2, %adlrs2b0bt2 : tensor<128xf32>
    %adwdps2b0bt2 = stablehlo.multiply %adwdlrs2b0bt2, %s2b0bt2 : tensor<128xf32>
    %adnews2b0bt2 = stablehlo.subtract %adsubs2b0bt2, %adwdps2b0bt2 : tensor<128xf32>
    %adb1s2b1W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b1W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b1W1 = stablehlo.multiply %adb1s2b1W1, %s2b1W1m : tensor<128x128x3x3xf32>
    %admgs2b1W1 = stablehlo.multiply %adob1s2b1W1, %s2b1dW1 : tensor<128x128x3x3xf32>
    %admns2b1W1 = stablehlo.add %admss2b1W1, %admgs2b1W1 : tensor<128x128x3x3xf32>
    %adb2s2b1W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b1W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b1W1 = stablehlo.multiply %adb2s2b1W1, %s2b1W1v : tensor<128x128x3x3xf32>
    %adg2s2b1W1 = stablehlo.multiply %s2b1dW1, %s2b1dW1 : tensor<128x128x3x3xf32>
    %advgs2b1W1 = stablehlo.multiply %adob2s2b1W1, %adg2s2b1W1 : tensor<128x128x3x3xf32>
    %advns2b1W1 = stablehlo.add %advss2b1W1, %advgs2b1W1 : tensor<128x128x3x3xf32>
    %adbc1s2b1W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b1W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b1W1 = stablehlo.divide %admns2b1W1, %adbc1s2b1W1 : tensor<128x128x3x3xf32>
    %advhs2b1W1 = stablehlo.divide %advns2b1W1, %adbc2s2b1W1 : tensor<128x128x3x3xf32>
    %adlrs2b1W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b1W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b1W1 = stablehlo.sqrt %advhs2b1W1 : tensor<128x128x3x3xf32>
    %addens2b1W1 = stablehlo.add %adsqs2b1W1, %adepss2b1W1 : tensor<128x128x3x3xf32>
    %adrats2b1W1 = stablehlo.divide %admhs2b1W1, %addens2b1W1 : tensor<128x128x3x3xf32>
    %adsts2b1W1 = stablehlo.multiply %adlrs2b1W1, %adrats2b1W1 : tensor<128x128x3x3xf32>
    %adsubs2b1W1 = stablehlo.subtract %s2b1W1, %adsts2b1W1 : tensor<128x128x3x3xf32>
    %adwds2b1W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b1W1 = stablehlo.multiply %adwds2b1W1, %adlrs2b1W1 : tensor<128x128x3x3xf32>
    %adwdps2b1W1 = stablehlo.multiply %adwdlrs2b1W1, %s2b1W1 : tensor<128x128x3x3xf32>
    %adnews2b1W1 = stablehlo.subtract %adsubs2b1W1, %adwdps2b1W1 : tensor<128x128x3x3xf32>
    %adb1s2b1b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1b1 = stablehlo.multiply %adb1s2b1b1, %s2b1b1m : tensor<128xf32>
    %admgs2b1b1 = stablehlo.multiply %adob1s2b1b1, %s2b1db1 : tensor<128xf32>
    %admns2b1b1 = stablehlo.add %admss2b1b1, %admgs2b1b1 : tensor<128xf32>
    %adb2s2b1b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1b1 = stablehlo.multiply %adb2s2b1b1, %s2b1b1v : tensor<128xf32>
    %adg2s2b1b1 = stablehlo.multiply %s2b1db1, %s2b1db1 : tensor<128xf32>
    %advgs2b1b1 = stablehlo.multiply %adob2s2b1b1, %adg2s2b1b1 : tensor<128xf32>
    %advns2b1b1 = stablehlo.add %advss2b1b1, %advgs2b1b1 : tensor<128xf32>
    %adbc1s2b1b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1b1 = stablehlo.divide %admns2b1b1, %adbc1s2b1b1 : tensor<128xf32>
    %advhs2b1b1 = stablehlo.divide %advns2b1b1, %adbc2s2b1b1 : tensor<128xf32>
    %adlrs2b1b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1b1 = stablehlo.sqrt %advhs2b1b1 : tensor<128xf32>
    %addens2b1b1 = stablehlo.add %adsqs2b1b1, %adepss2b1b1 : tensor<128xf32>
    %adrats2b1b1 = stablehlo.divide %admhs2b1b1, %addens2b1b1 : tensor<128xf32>
    %adsts2b1b1 = stablehlo.multiply %adlrs2b1b1, %adrats2b1b1 : tensor<128xf32>
    %adsubs2b1b1 = stablehlo.subtract %s2b1b1, %adsts2b1b1 : tensor<128xf32>
    %adwds2b1b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1b1 = stablehlo.multiply %adwds2b1b1, %adlrs2b1b1 : tensor<128xf32>
    %adwdps2b1b1 = stablehlo.multiply %adwdlrs2b1b1, %s2b1b1 : tensor<128xf32>
    %adnews2b1b1 = stablehlo.subtract %adsubs2b1b1, %adwdps2b1b1 : tensor<128xf32>
    %adb1s2b1g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1g1 = stablehlo.multiply %adb1s2b1g1, %s2b1g1m : tensor<128xf32>
    %admgs2b1g1 = stablehlo.multiply %adob1s2b1g1, %s2b1dn1dg : tensor<128xf32>
    %admns2b1g1 = stablehlo.add %admss2b1g1, %admgs2b1g1 : tensor<128xf32>
    %adb2s2b1g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1g1 = stablehlo.multiply %adb2s2b1g1, %s2b1g1v : tensor<128xf32>
    %adg2s2b1g1 = stablehlo.multiply %s2b1dn1dg, %s2b1dn1dg : tensor<128xf32>
    %advgs2b1g1 = stablehlo.multiply %adob2s2b1g1, %adg2s2b1g1 : tensor<128xf32>
    %advns2b1g1 = stablehlo.add %advss2b1g1, %advgs2b1g1 : tensor<128xf32>
    %adbc1s2b1g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1g1 = stablehlo.divide %admns2b1g1, %adbc1s2b1g1 : tensor<128xf32>
    %advhs2b1g1 = stablehlo.divide %advns2b1g1, %adbc2s2b1g1 : tensor<128xf32>
    %adlrs2b1g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1g1 = stablehlo.sqrt %advhs2b1g1 : tensor<128xf32>
    %addens2b1g1 = stablehlo.add %adsqs2b1g1, %adepss2b1g1 : tensor<128xf32>
    %adrats2b1g1 = stablehlo.divide %admhs2b1g1, %addens2b1g1 : tensor<128xf32>
    %adsts2b1g1 = stablehlo.multiply %adlrs2b1g1, %adrats2b1g1 : tensor<128xf32>
    %adsubs2b1g1 = stablehlo.subtract %s2b1g1, %adsts2b1g1 : tensor<128xf32>
    %adwds2b1g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1g1 = stablehlo.multiply %adwds2b1g1, %adlrs2b1g1 : tensor<128xf32>
    %adwdps2b1g1 = stablehlo.multiply %adwdlrs2b1g1, %s2b1g1 : tensor<128xf32>
    %adnews2b1g1 = stablehlo.subtract %adsubs2b1g1, %adwdps2b1g1 : tensor<128xf32>
    %adb1s2b1bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1bt1 = stablehlo.multiply %adb1s2b1bt1, %s2b1bt1m : tensor<128xf32>
    %admgs2b1bt1 = stablehlo.multiply %adob1s2b1bt1, %s2b1dn1db : tensor<128xf32>
    %admns2b1bt1 = stablehlo.add %admss2b1bt1, %admgs2b1bt1 : tensor<128xf32>
    %adb2s2b1bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1bt1 = stablehlo.multiply %adb2s2b1bt1, %s2b1bt1v : tensor<128xf32>
    %adg2s2b1bt1 = stablehlo.multiply %s2b1dn1db, %s2b1dn1db : tensor<128xf32>
    %advgs2b1bt1 = stablehlo.multiply %adob2s2b1bt1, %adg2s2b1bt1 : tensor<128xf32>
    %advns2b1bt1 = stablehlo.add %advss2b1bt1, %advgs2b1bt1 : tensor<128xf32>
    %adbc1s2b1bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1bt1 = stablehlo.divide %admns2b1bt1, %adbc1s2b1bt1 : tensor<128xf32>
    %advhs2b1bt1 = stablehlo.divide %advns2b1bt1, %adbc2s2b1bt1 : tensor<128xf32>
    %adlrs2b1bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1bt1 = stablehlo.sqrt %advhs2b1bt1 : tensor<128xf32>
    %addens2b1bt1 = stablehlo.add %adsqs2b1bt1, %adepss2b1bt1 : tensor<128xf32>
    %adrats2b1bt1 = stablehlo.divide %admhs2b1bt1, %addens2b1bt1 : tensor<128xf32>
    %adsts2b1bt1 = stablehlo.multiply %adlrs2b1bt1, %adrats2b1bt1 : tensor<128xf32>
    %adsubs2b1bt1 = stablehlo.subtract %s2b1bt1, %adsts2b1bt1 : tensor<128xf32>
    %adwds2b1bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1bt1 = stablehlo.multiply %adwds2b1bt1, %adlrs2b1bt1 : tensor<128xf32>
    %adwdps2b1bt1 = stablehlo.multiply %adwdlrs2b1bt1, %s2b1bt1 : tensor<128xf32>
    %adnews2b1bt1 = stablehlo.subtract %adsubs2b1bt1, %adwdps2b1bt1 : tensor<128xf32>
    %adb1s2b1W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b1W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b1W2 = stablehlo.multiply %adb1s2b1W2, %s2b1W2m : tensor<128x128x3x3xf32>
    %admgs2b1W2 = stablehlo.multiply %adob1s2b1W2, %s2b1dW2 : tensor<128x128x3x3xf32>
    %admns2b1W2 = stablehlo.add %admss2b1W2, %admgs2b1W2 : tensor<128x128x3x3xf32>
    %adb2s2b1W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b1W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b1W2 = stablehlo.multiply %adb2s2b1W2, %s2b1W2v : tensor<128x128x3x3xf32>
    %adg2s2b1W2 = stablehlo.multiply %s2b1dW2, %s2b1dW2 : tensor<128x128x3x3xf32>
    %advgs2b1W2 = stablehlo.multiply %adob2s2b1W2, %adg2s2b1W2 : tensor<128x128x3x3xf32>
    %advns2b1W2 = stablehlo.add %advss2b1W2, %advgs2b1W2 : tensor<128x128x3x3xf32>
    %adbc1s2b1W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b1W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b1W2 = stablehlo.divide %admns2b1W2, %adbc1s2b1W2 : tensor<128x128x3x3xf32>
    %advhs2b1W2 = stablehlo.divide %advns2b1W2, %adbc2s2b1W2 : tensor<128x128x3x3xf32>
    %adlrs2b1W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b1W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b1W2 = stablehlo.sqrt %advhs2b1W2 : tensor<128x128x3x3xf32>
    %addens2b1W2 = stablehlo.add %adsqs2b1W2, %adepss2b1W2 : tensor<128x128x3x3xf32>
    %adrats2b1W2 = stablehlo.divide %admhs2b1W2, %addens2b1W2 : tensor<128x128x3x3xf32>
    %adsts2b1W2 = stablehlo.multiply %adlrs2b1W2, %adrats2b1W2 : tensor<128x128x3x3xf32>
    %adsubs2b1W2 = stablehlo.subtract %s2b1W2, %adsts2b1W2 : tensor<128x128x3x3xf32>
    %adwds2b1W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b1W2 = stablehlo.multiply %adwds2b1W2, %adlrs2b1W2 : tensor<128x128x3x3xf32>
    %adwdps2b1W2 = stablehlo.multiply %adwdlrs2b1W2, %s2b1W2 : tensor<128x128x3x3xf32>
    %adnews2b1W2 = stablehlo.subtract %adsubs2b1W2, %adwdps2b1W2 : tensor<128x128x3x3xf32>
    %adb1s2b1b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1b2 = stablehlo.multiply %adb1s2b1b2, %s2b1b2m : tensor<128xf32>
    %admgs2b1b2 = stablehlo.multiply %adob1s2b1b2, %s2b1db2 : tensor<128xf32>
    %admns2b1b2 = stablehlo.add %admss2b1b2, %admgs2b1b2 : tensor<128xf32>
    %adb2s2b1b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1b2 = stablehlo.multiply %adb2s2b1b2, %s2b1b2v : tensor<128xf32>
    %adg2s2b1b2 = stablehlo.multiply %s2b1db2, %s2b1db2 : tensor<128xf32>
    %advgs2b1b2 = stablehlo.multiply %adob2s2b1b2, %adg2s2b1b2 : tensor<128xf32>
    %advns2b1b2 = stablehlo.add %advss2b1b2, %advgs2b1b2 : tensor<128xf32>
    %adbc1s2b1b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1b2 = stablehlo.divide %admns2b1b2, %adbc1s2b1b2 : tensor<128xf32>
    %advhs2b1b2 = stablehlo.divide %advns2b1b2, %adbc2s2b1b2 : tensor<128xf32>
    %adlrs2b1b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1b2 = stablehlo.sqrt %advhs2b1b2 : tensor<128xf32>
    %addens2b1b2 = stablehlo.add %adsqs2b1b2, %adepss2b1b2 : tensor<128xf32>
    %adrats2b1b2 = stablehlo.divide %admhs2b1b2, %addens2b1b2 : tensor<128xf32>
    %adsts2b1b2 = stablehlo.multiply %adlrs2b1b2, %adrats2b1b2 : tensor<128xf32>
    %adsubs2b1b2 = stablehlo.subtract %s2b1b2, %adsts2b1b2 : tensor<128xf32>
    %adwds2b1b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1b2 = stablehlo.multiply %adwds2b1b2, %adlrs2b1b2 : tensor<128xf32>
    %adwdps2b1b2 = stablehlo.multiply %adwdlrs2b1b2, %s2b1b2 : tensor<128xf32>
    %adnews2b1b2 = stablehlo.subtract %adsubs2b1b2, %adwdps2b1b2 : tensor<128xf32>
    %adb1s2b1g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1g2 = stablehlo.multiply %adb1s2b1g2, %s2b1g2m : tensor<128xf32>
    %admgs2b1g2 = stablehlo.multiply %adob1s2b1g2, %s2b1dn2dg : tensor<128xf32>
    %admns2b1g2 = stablehlo.add %admss2b1g2, %admgs2b1g2 : tensor<128xf32>
    %adb2s2b1g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1g2 = stablehlo.multiply %adb2s2b1g2, %s2b1g2v : tensor<128xf32>
    %adg2s2b1g2 = stablehlo.multiply %s2b1dn2dg, %s2b1dn2dg : tensor<128xf32>
    %advgs2b1g2 = stablehlo.multiply %adob2s2b1g2, %adg2s2b1g2 : tensor<128xf32>
    %advns2b1g2 = stablehlo.add %advss2b1g2, %advgs2b1g2 : tensor<128xf32>
    %adbc1s2b1g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1g2 = stablehlo.divide %admns2b1g2, %adbc1s2b1g2 : tensor<128xf32>
    %advhs2b1g2 = stablehlo.divide %advns2b1g2, %adbc2s2b1g2 : tensor<128xf32>
    %adlrs2b1g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1g2 = stablehlo.sqrt %advhs2b1g2 : tensor<128xf32>
    %addens2b1g2 = stablehlo.add %adsqs2b1g2, %adepss2b1g2 : tensor<128xf32>
    %adrats2b1g2 = stablehlo.divide %admhs2b1g2, %addens2b1g2 : tensor<128xf32>
    %adsts2b1g2 = stablehlo.multiply %adlrs2b1g2, %adrats2b1g2 : tensor<128xf32>
    %adsubs2b1g2 = stablehlo.subtract %s2b1g2, %adsts2b1g2 : tensor<128xf32>
    %adwds2b1g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1g2 = stablehlo.multiply %adwds2b1g2, %adlrs2b1g2 : tensor<128xf32>
    %adwdps2b1g2 = stablehlo.multiply %adwdlrs2b1g2, %s2b1g2 : tensor<128xf32>
    %adnews2b1g2 = stablehlo.subtract %adsubs2b1g2, %adwdps2b1g2 : tensor<128xf32>
    %adb1s2b1bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b1bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b1bt2 = stablehlo.multiply %adb1s2b1bt2, %s2b1bt2m : tensor<128xf32>
    %admgs2b1bt2 = stablehlo.multiply %adob1s2b1bt2, %s2b1dn2db : tensor<128xf32>
    %admns2b1bt2 = stablehlo.add %admss2b1bt2, %admgs2b1bt2 : tensor<128xf32>
    %adb2s2b1bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b1bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b1bt2 = stablehlo.multiply %adb2s2b1bt2, %s2b1bt2v : tensor<128xf32>
    %adg2s2b1bt2 = stablehlo.multiply %s2b1dn2db, %s2b1dn2db : tensor<128xf32>
    %advgs2b1bt2 = stablehlo.multiply %adob2s2b1bt2, %adg2s2b1bt2 : tensor<128xf32>
    %advns2b1bt2 = stablehlo.add %advss2b1bt2, %advgs2b1bt2 : tensor<128xf32>
    %adbc1s2b1bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b1bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b1bt2 = stablehlo.divide %admns2b1bt2, %adbc1s2b1bt2 : tensor<128xf32>
    %advhs2b1bt2 = stablehlo.divide %advns2b1bt2, %adbc2s2b1bt2 : tensor<128xf32>
    %adlrs2b1bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b1bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b1bt2 = stablehlo.sqrt %advhs2b1bt2 : tensor<128xf32>
    %addens2b1bt2 = stablehlo.add %adsqs2b1bt2, %adepss2b1bt2 : tensor<128xf32>
    %adrats2b1bt2 = stablehlo.divide %admhs2b1bt2, %addens2b1bt2 : tensor<128xf32>
    %adsts2b1bt2 = stablehlo.multiply %adlrs2b1bt2, %adrats2b1bt2 : tensor<128xf32>
    %adsubs2b1bt2 = stablehlo.subtract %s2b1bt2, %adsts2b1bt2 : tensor<128xf32>
    %adwds2b1bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b1bt2 = stablehlo.multiply %adwds2b1bt2, %adlrs2b1bt2 : tensor<128xf32>
    %adwdps2b1bt2 = stablehlo.multiply %adwdlrs2b1bt2, %s2b1bt2 : tensor<128xf32>
    %adnews2b1bt2 = stablehlo.subtract %adsubs2b1bt2, %adwdps2b1bt2 : tensor<128xf32>
    %adb1s2b2W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b2W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b2W1 = stablehlo.multiply %adb1s2b2W1, %s2b2W1m : tensor<128x128x3x3xf32>
    %admgs2b2W1 = stablehlo.multiply %adob1s2b2W1, %s2b2dW1 : tensor<128x128x3x3xf32>
    %admns2b2W1 = stablehlo.add %admss2b2W1, %admgs2b2W1 : tensor<128x128x3x3xf32>
    %adb2s2b2W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b2W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b2W1 = stablehlo.multiply %adb2s2b2W1, %s2b2W1v : tensor<128x128x3x3xf32>
    %adg2s2b2W1 = stablehlo.multiply %s2b2dW1, %s2b2dW1 : tensor<128x128x3x3xf32>
    %advgs2b2W1 = stablehlo.multiply %adob2s2b2W1, %adg2s2b2W1 : tensor<128x128x3x3xf32>
    %advns2b2W1 = stablehlo.add %advss2b2W1, %advgs2b2W1 : tensor<128x128x3x3xf32>
    %adbc1s2b2W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b2W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b2W1 = stablehlo.divide %admns2b2W1, %adbc1s2b2W1 : tensor<128x128x3x3xf32>
    %advhs2b2W1 = stablehlo.divide %advns2b2W1, %adbc2s2b2W1 : tensor<128x128x3x3xf32>
    %adlrs2b2W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b2W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b2W1 = stablehlo.sqrt %advhs2b2W1 : tensor<128x128x3x3xf32>
    %addens2b2W1 = stablehlo.add %adsqs2b2W1, %adepss2b2W1 : tensor<128x128x3x3xf32>
    %adrats2b2W1 = stablehlo.divide %admhs2b2W1, %addens2b2W1 : tensor<128x128x3x3xf32>
    %adsts2b2W1 = stablehlo.multiply %adlrs2b2W1, %adrats2b2W1 : tensor<128x128x3x3xf32>
    %adsubs2b2W1 = stablehlo.subtract %s2b2W1, %adsts2b2W1 : tensor<128x128x3x3xf32>
    %adwds2b2W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b2W1 = stablehlo.multiply %adwds2b2W1, %adlrs2b2W1 : tensor<128x128x3x3xf32>
    %adwdps2b2W1 = stablehlo.multiply %adwdlrs2b2W1, %s2b2W1 : tensor<128x128x3x3xf32>
    %adnews2b2W1 = stablehlo.subtract %adsubs2b2W1, %adwdps2b2W1 : tensor<128x128x3x3xf32>
    %adb1s2b2b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2b1 = stablehlo.multiply %adb1s2b2b1, %s2b2b1m : tensor<128xf32>
    %admgs2b2b1 = stablehlo.multiply %adob1s2b2b1, %s2b2db1 : tensor<128xf32>
    %admns2b2b1 = stablehlo.add %admss2b2b1, %admgs2b2b1 : tensor<128xf32>
    %adb2s2b2b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2b1 = stablehlo.multiply %adb2s2b2b1, %s2b2b1v : tensor<128xf32>
    %adg2s2b2b1 = stablehlo.multiply %s2b2db1, %s2b2db1 : tensor<128xf32>
    %advgs2b2b1 = stablehlo.multiply %adob2s2b2b1, %adg2s2b2b1 : tensor<128xf32>
    %advns2b2b1 = stablehlo.add %advss2b2b1, %advgs2b2b1 : tensor<128xf32>
    %adbc1s2b2b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2b1 = stablehlo.divide %admns2b2b1, %adbc1s2b2b1 : tensor<128xf32>
    %advhs2b2b1 = stablehlo.divide %advns2b2b1, %adbc2s2b2b1 : tensor<128xf32>
    %adlrs2b2b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2b1 = stablehlo.sqrt %advhs2b2b1 : tensor<128xf32>
    %addens2b2b1 = stablehlo.add %adsqs2b2b1, %adepss2b2b1 : tensor<128xf32>
    %adrats2b2b1 = stablehlo.divide %admhs2b2b1, %addens2b2b1 : tensor<128xf32>
    %adsts2b2b1 = stablehlo.multiply %adlrs2b2b1, %adrats2b2b1 : tensor<128xf32>
    %adsubs2b2b1 = stablehlo.subtract %s2b2b1, %adsts2b2b1 : tensor<128xf32>
    %adwds2b2b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2b1 = stablehlo.multiply %adwds2b2b1, %adlrs2b2b1 : tensor<128xf32>
    %adwdps2b2b1 = stablehlo.multiply %adwdlrs2b2b1, %s2b2b1 : tensor<128xf32>
    %adnews2b2b1 = stablehlo.subtract %adsubs2b2b1, %adwdps2b2b1 : tensor<128xf32>
    %adb1s2b2g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2g1 = stablehlo.multiply %adb1s2b2g1, %s2b2g1m : tensor<128xf32>
    %admgs2b2g1 = stablehlo.multiply %adob1s2b2g1, %s2b2dn1dg : tensor<128xf32>
    %admns2b2g1 = stablehlo.add %admss2b2g1, %admgs2b2g1 : tensor<128xf32>
    %adb2s2b2g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2g1 = stablehlo.multiply %adb2s2b2g1, %s2b2g1v : tensor<128xf32>
    %adg2s2b2g1 = stablehlo.multiply %s2b2dn1dg, %s2b2dn1dg : tensor<128xf32>
    %advgs2b2g1 = stablehlo.multiply %adob2s2b2g1, %adg2s2b2g1 : tensor<128xf32>
    %advns2b2g1 = stablehlo.add %advss2b2g1, %advgs2b2g1 : tensor<128xf32>
    %adbc1s2b2g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2g1 = stablehlo.divide %admns2b2g1, %adbc1s2b2g1 : tensor<128xf32>
    %advhs2b2g1 = stablehlo.divide %advns2b2g1, %adbc2s2b2g1 : tensor<128xf32>
    %adlrs2b2g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2g1 = stablehlo.sqrt %advhs2b2g1 : tensor<128xf32>
    %addens2b2g1 = stablehlo.add %adsqs2b2g1, %adepss2b2g1 : tensor<128xf32>
    %adrats2b2g1 = stablehlo.divide %admhs2b2g1, %addens2b2g1 : tensor<128xf32>
    %adsts2b2g1 = stablehlo.multiply %adlrs2b2g1, %adrats2b2g1 : tensor<128xf32>
    %adsubs2b2g1 = stablehlo.subtract %s2b2g1, %adsts2b2g1 : tensor<128xf32>
    %adwds2b2g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2g1 = stablehlo.multiply %adwds2b2g1, %adlrs2b2g1 : tensor<128xf32>
    %adwdps2b2g1 = stablehlo.multiply %adwdlrs2b2g1, %s2b2g1 : tensor<128xf32>
    %adnews2b2g1 = stablehlo.subtract %adsubs2b2g1, %adwdps2b2g1 : tensor<128xf32>
    %adb1s2b2bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2bt1 = stablehlo.multiply %adb1s2b2bt1, %s2b2bt1m : tensor<128xf32>
    %admgs2b2bt1 = stablehlo.multiply %adob1s2b2bt1, %s2b2dn1db : tensor<128xf32>
    %admns2b2bt1 = stablehlo.add %admss2b2bt1, %admgs2b2bt1 : tensor<128xf32>
    %adb2s2b2bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2bt1 = stablehlo.multiply %adb2s2b2bt1, %s2b2bt1v : tensor<128xf32>
    %adg2s2b2bt1 = stablehlo.multiply %s2b2dn1db, %s2b2dn1db : tensor<128xf32>
    %advgs2b2bt1 = stablehlo.multiply %adob2s2b2bt1, %adg2s2b2bt1 : tensor<128xf32>
    %advns2b2bt1 = stablehlo.add %advss2b2bt1, %advgs2b2bt1 : tensor<128xf32>
    %adbc1s2b2bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2bt1 = stablehlo.divide %admns2b2bt1, %adbc1s2b2bt1 : tensor<128xf32>
    %advhs2b2bt1 = stablehlo.divide %advns2b2bt1, %adbc2s2b2bt1 : tensor<128xf32>
    %adlrs2b2bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2bt1 = stablehlo.sqrt %advhs2b2bt1 : tensor<128xf32>
    %addens2b2bt1 = stablehlo.add %adsqs2b2bt1, %adepss2b2bt1 : tensor<128xf32>
    %adrats2b2bt1 = stablehlo.divide %admhs2b2bt1, %addens2b2bt1 : tensor<128xf32>
    %adsts2b2bt1 = stablehlo.multiply %adlrs2b2bt1, %adrats2b2bt1 : tensor<128xf32>
    %adsubs2b2bt1 = stablehlo.subtract %s2b2bt1, %adsts2b2bt1 : tensor<128xf32>
    %adwds2b2bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2bt1 = stablehlo.multiply %adwds2b2bt1, %adlrs2b2bt1 : tensor<128xf32>
    %adwdps2b2bt1 = stablehlo.multiply %adwdlrs2b2bt1, %s2b2bt1 : tensor<128xf32>
    %adnews2b2bt1 = stablehlo.subtract %adsubs2b2bt1, %adwdps2b2bt1 : tensor<128xf32>
    %adb1s2b2W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob1s2b2W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admss2b2W2 = stablehlo.multiply %adb1s2b2W2, %s2b2W2m : tensor<128x128x3x3xf32>
    %admgs2b2W2 = stablehlo.multiply %adob1s2b2W2, %s2b2dW2 : tensor<128x128x3x3xf32>
    %admns2b2W2 = stablehlo.add %admss2b2W2, %admgs2b2W2 : tensor<128x128x3x3xf32>
    %adb2s2b2W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adob2s2b2W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %advss2b2W2 = stablehlo.multiply %adb2s2b2W2, %s2b2W2v : tensor<128x128x3x3xf32>
    %adg2s2b2W2 = stablehlo.multiply %s2b2dW2, %s2b2dW2 : tensor<128x128x3x3xf32>
    %advgs2b2W2 = stablehlo.multiply %adob2s2b2W2, %adg2s2b2W2 : tensor<128x128x3x3xf32>
    %advns2b2W2 = stablehlo.add %advss2b2W2, %advgs2b2W2 : tensor<128x128x3x3xf32>
    %adbc1s2b2W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adbc2s2b2W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %admhs2b2W2 = stablehlo.divide %admns2b2W2, %adbc1s2b2W2 : tensor<128x128x3x3xf32>
    %advhs2b2W2 = stablehlo.divide %advns2b2W2, %adbc2s2b2W2 : tensor<128x128x3x3xf32>
    %adlrs2b2W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adepss2b2W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adsqs2b2W2 = stablehlo.sqrt %advhs2b2W2 : tensor<128x128x3x3xf32>
    %addens2b2W2 = stablehlo.add %adsqs2b2W2, %adepss2b2W2 : tensor<128x128x3x3xf32>
    %adrats2b2W2 = stablehlo.divide %admhs2b2W2, %addens2b2W2 : tensor<128x128x3x3xf32>
    %adsts2b2W2 = stablehlo.multiply %adlrs2b2W2, %adrats2b2W2 : tensor<128x128x3x3xf32>
    %adsubs2b2W2 = stablehlo.subtract %s2b2W2, %adsts2b2W2 : tensor<128x128x3x3xf32>
    %adwds2b2W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %adwdlrs2b2W2 = stablehlo.multiply %adwds2b2W2, %adlrs2b2W2 : tensor<128x128x3x3xf32>
    %adwdps2b2W2 = stablehlo.multiply %adwdlrs2b2W2, %s2b2W2 : tensor<128x128x3x3xf32>
    %adnews2b2W2 = stablehlo.subtract %adsubs2b2W2, %adwdps2b2W2 : tensor<128x128x3x3xf32>
    %adb1s2b2b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2b2 = stablehlo.multiply %adb1s2b2b2, %s2b2b2m : tensor<128xf32>
    %admgs2b2b2 = stablehlo.multiply %adob1s2b2b2, %s2b2db2 : tensor<128xf32>
    %admns2b2b2 = stablehlo.add %admss2b2b2, %admgs2b2b2 : tensor<128xf32>
    %adb2s2b2b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2b2 = stablehlo.multiply %adb2s2b2b2, %s2b2b2v : tensor<128xf32>
    %adg2s2b2b2 = stablehlo.multiply %s2b2db2, %s2b2db2 : tensor<128xf32>
    %advgs2b2b2 = stablehlo.multiply %adob2s2b2b2, %adg2s2b2b2 : tensor<128xf32>
    %advns2b2b2 = stablehlo.add %advss2b2b2, %advgs2b2b2 : tensor<128xf32>
    %adbc1s2b2b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2b2 = stablehlo.divide %admns2b2b2, %adbc1s2b2b2 : tensor<128xf32>
    %advhs2b2b2 = stablehlo.divide %advns2b2b2, %adbc2s2b2b2 : tensor<128xf32>
    %adlrs2b2b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2b2 = stablehlo.sqrt %advhs2b2b2 : tensor<128xf32>
    %addens2b2b2 = stablehlo.add %adsqs2b2b2, %adepss2b2b2 : tensor<128xf32>
    %adrats2b2b2 = stablehlo.divide %admhs2b2b2, %addens2b2b2 : tensor<128xf32>
    %adsts2b2b2 = stablehlo.multiply %adlrs2b2b2, %adrats2b2b2 : tensor<128xf32>
    %adsubs2b2b2 = stablehlo.subtract %s2b2b2, %adsts2b2b2 : tensor<128xf32>
    %adwds2b2b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2b2 = stablehlo.multiply %adwds2b2b2, %adlrs2b2b2 : tensor<128xf32>
    %adwdps2b2b2 = stablehlo.multiply %adwdlrs2b2b2, %s2b2b2 : tensor<128xf32>
    %adnews2b2b2 = stablehlo.subtract %adsubs2b2b2, %adwdps2b2b2 : tensor<128xf32>
    %adb1s2b2g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2g2 = stablehlo.multiply %adb1s2b2g2, %s2b2g2m : tensor<128xf32>
    %admgs2b2g2 = stablehlo.multiply %adob1s2b2g2, %s2b2dn2dg : tensor<128xf32>
    %admns2b2g2 = stablehlo.add %admss2b2g2, %admgs2b2g2 : tensor<128xf32>
    %adb2s2b2g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2g2 = stablehlo.multiply %adb2s2b2g2, %s2b2g2v : tensor<128xf32>
    %adg2s2b2g2 = stablehlo.multiply %s2b2dn2dg, %s2b2dn2dg : tensor<128xf32>
    %advgs2b2g2 = stablehlo.multiply %adob2s2b2g2, %adg2s2b2g2 : tensor<128xf32>
    %advns2b2g2 = stablehlo.add %advss2b2g2, %advgs2b2g2 : tensor<128xf32>
    %adbc1s2b2g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2g2 = stablehlo.divide %admns2b2g2, %adbc1s2b2g2 : tensor<128xf32>
    %advhs2b2g2 = stablehlo.divide %advns2b2g2, %adbc2s2b2g2 : tensor<128xf32>
    %adlrs2b2g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2g2 = stablehlo.sqrt %advhs2b2g2 : tensor<128xf32>
    %addens2b2g2 = stablehlo.add %adsqs2b2g2, %adepss2b2g2 : tensor<128xf32>
    %adrats2b2g2 = stablehlo.divide %admhs2b2g2, %addens2b2g2 : tensor<128xf32>
    %adsts2b2g2 = stablehlo.multiply %adlrs2b2g2, %adrats2b2g2 : tensor<128xf32>
    %adsubs2b2g2 = stablehlo.subtract %s2b2g2, %adsts2b2g2 : tensor<128xf32>
    %adwds2b2g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2g2 = stablehlo.multiply %adwds2b2g2, %adlrs2b2g2 : tensor<128xf32>
    %adwdps2b2g2 = stablehlo.multiply %adwdlrs2b2g2, %s2b2g2 : tensor<128xf32>
    %adnews2b2g2 = stablehlo.subtract %adsubs2b2g2, %adwdps2b2g2 : tensor<128xf32>
    %adb1s2b2bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob1s2b2bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admss2b2bt2 = stablehlo.multiply %adb1s2b2bt2, %s2b2bt2m : tensor<128xf32>
    %admgs2b2bt2 = stablehlo.multiply %adob1s2b2bt2, %s2b2dn2db : tensor<128xf32>
    %admns2b2bt2 = stablehlo.add %admss2b2bt2, %admgs2b2bt2 : tensor<128xf32>
    %adb2s2b2bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adob2s2b2bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %advss2b2bt2 = stablehlo.multiply %adb2s2b2bt2, %s2b2bt2v : tensor<128xf32>
    %adg2s2b2bt2 = stablehlo.multiply %s2b2dn2db, %s2b2dn2db : tensor<128xf32>
    %advgs2b2bt2 = stablehlo.multiply %adob2s2b2bt2, %adg2s2b2bt2 : tensor<128xf32>
    %advns2b2bt2 = stablehlo.add %advss2b2bt2, %advgs2b2bt2 : tensor<128xf32>
    %adbc1s2b2bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adbc2s2b2bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %admhs2b2bt2 = stablehlo.divide %admns2b2bt2, %adbc1s2b2bt2 : tensor<128xf32>
    %advhs2b2bt2 = stablehlo.divide %advns2b2bt2, %adbc2s2b2bt2 : tensor<128xf32>
    %adlrs2b2bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adepss2b2bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adsqs2b2bt2 = stablehlo.sqrt %advhs2b2bt2 : tensor<128xf32>
    %addens2b2bt2 = stablehlo.add %adsqs2b2bt2, %adepss2b2bt2 : tensor<128xf32>
    %adrats2b2bt2 = stablehlo.divide %admhs2b2bt2, %addens2b2bt2 : tensor<128xf32>
    %adsts2b2bt2 = stablehlo.multiply %adlrs2b2bt2, %adrats2b2bt2 : tensor<128xf32>
    %adsubs2b2bt2 = stablehlo.subtract %s2b2bt2, %adsts2b2bt2 : tensor<128xf32>
    %adwds2b2bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %adwdlrs2b2bt2 = stablehlo.multiply %adwds2b2bt2, %adlrs2b2bt2 : tensor<128xf32>
    %adwdps2b2bt2 = stablehlo.multiply %adwdlrs2b2bt2, %s2b2bt2 : tensor<128xf32>
    %adnews2b2bt2 = stablehlo.subtract %adsubs2b2bt2, %adwdps2b2bt2 : tensor<128xf32>
    %adb1d3W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adob1d3W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %admsd3W1 = stablehlo.multiply %adb1d3W1, %d3W1m : tensor<256x128x3x3xf32>
    %admgd3W1 = stablehlo.multiply %adob1d3W1, %d3dW1 : tensor<256x128x3x3xf32>
    %admnd3W1 = stablehlo.add %admsd3W1, %admgd3W1 : tensor<256x128x3x3xf32>
    %adb2d3W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adob2d3W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %advsd3W1 = stablehlo.multiply %adb2d3W1, %d3W1v : tensor<256x128x3x3xf32>
    %adg2d3W1 = stablehlo.multiply %d3dW1, %d3dW1 : tensor<256x128x3x3xf32>
    %advgd3W1 = stablehlo.multiply %adob2d3W1, %adg2d3W1 : tensor<256x128x3x3xf32>
    %advnd3W1 = stablehlo.add %advsd3W1, %advgd3W1 : tensor<256x128x3x3xf32>
    %adbc1d3W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adbc2d3W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %admhd3W1 = stablehlo.divide %admnd3W1, %adbc1d3W1 : tensor<256x128x3x3xf32>
    %advhd3W1 = stablehlo.divide %advnd3W1, %adbc2d3W1 : tensor<256x128x3x3xf32>
    %adlrd3W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adepsd3W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adsqd3W1 = stablehlo.sqrt %advhd3W1 : tensor<256x128x3x3xf32>
    %addend3W1 = stablehlo.add %adsqd3W1, %adepsd3W1 : tensor<256x128x3x3xf32>
    %adratd3W1 = stablehlo.divide %admhd3W1, %addend3W1 : tensor<256x128x3x3xf32>
    %adstd3W1 = stablehlo.multiply %adlrd3W1, %adratd3W1 : tensor<256x128x3x3xf32>
    %adsubd3W1 = stablehlo.subtract %d3W1, %adstd3W1 : tensor<256x128x3x3xf32>
    %adwdd3W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adwdlrd3W1 = stablehlo.multiply %adwdd3W1, %adlrd3W1 : tensor<256x128x3x3xf32>
    %adwdpd3W1 = stablehlo.multiply %adwdlrd3W1, %d3W1 : tensor<256x128x3x3xf32>
    %adnewd3W1 = stablehlo.subtract %adsubd3W1, %adwdpd3W1 : tensor<256x128x3x3xf32>
    %adb1d3b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3b1 = stablehlo.multiply %adb1d3b1, %d3b1m : tensor<256xf32>
    %admgd3b1 = stablehlo.multiply %adob1d3b1, %d3db1 : tensor<256xf32>
    %admnd3b1 = stablehlo.add %admsd3b1, %admgd3b1 : tensor<256xf32>
    %adb2d3b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3b1 = stablehlo.multiply %adb2d3b1, %d3b1v : tensor<256xf32>
    %adg2d3b1 = stablehlo.multiply %d3db1, %d3db1 : tensor<256xf32>
    %advgd3b1 = stablehlo.multiply %adob2d3b1, %adg2d3b1 : tensor<256xf32>
    %advnd3b1 = stablehlo.add %advsd3b1, %advgd3b1 : tensor<256xf32>
    %adbc1d3b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3b1 = stablehlo.divide %admnd3b1, %adbc1d3b1 : tensor<256xf32>
    %advhd3b1 = stablehlo.divide %advnd3b1, %adbc2d3b1 : tensor<256xf32>
    %adlrd3b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3b1 = stablehlo.sqrt %advhd3b1 : tensor<256xf32>
    %addend3b1 = stablehlo.add %adsqd3b1, %adepsd3b1 : tensor<256xf32>
    %adratd3b1 = stablehlo.divide %admhd3b1, %addend3b1 : tensor<256xf32>
    %adstd3b1 = stablehlo.multiply %adlrd3b1, %adratd3b1 : tensor<256xf32>
    %adsubd3b1 = stablehlo.subtract %d3b1, %adstd3b1 : tensor<256xf32>
    %adwdd3b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3b1 = stablehlo.multiply %adwdd3b1, %adlrd3b1 : tensor<256xf32>
    %adwdpd3b1 = stablehlo.multiply %adwdlrd3b1, %d3b1 : tensor<256xf32>
    %adnewd3b1 = stablehlo.subtract %adsubd3b1, %adwdpd3b1 : tensor<256xf32>
    %adb1d3g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3g1 = stablehlo.multiply %adb1d3g1, %d3g1m : tensor<256xf32>
    %admgd3g1 = stablehlo.multiply %adob1d3g1, %d3dn1dg : tensor<256xf32>
    %admnd3g1 = stablehlo.add %admsd3g1, %admgd3g1 : tensor<256xf32>
    %adb2d3g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3g1 = stablehlo.multiply %adb2d3g1, %d3g1v : tensor<256xf32>
    %adg2d3g1 = stablehlo.multiply %d3dn1dg, %d3dn1dg : tensor<256xf32>
    %advgd3g1 = stablehlo.multiply %adob2d3g1, %adg2d3g1 : tensor<256xf32>
    %advnd3g1 = stablehlo.add %advsd3g1, %advgd3g1 : tensor<256xf32>
    %adbc1d3g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3g1 = stablehlo.divide %admnd3g1, %adbc1d3g1 : tensor<256xf32>
    %advhd3g1 = stablehlo.divide %advnd3g1, %adbc2d3g1 : tensor<256xf32>
    %adlrd3g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3g1 = stablehlo.sqrt %advhd3g1 : tensor<256xf32>
    %addend3g1 = stablehlo.add %adsqd3g1, %adepsd3g1 : tensor<256xf32>
    %adratd3g1 = stablehlo.divide %admhd3g1, %addend3g1 : tensor<256xf32>
    %adstd3g1 = stablehlo.multiply %adlrd3g1, %adratd3g1 : tensor<256xf32>
    %adsubd3g1 = stablehlo.subtract %d3g1, %adstd3g1 : tensor<256xf32>
    %adwdd3g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3g1 = stablehlo.multiply %adwdd3g1, %adlrd3g1 : tensor<256xf32>
    %adwdpd3g1 = stablehlo.multiply %adwdlrd3g1, %d3g1 : tensor<256xf32>
    %adnewd3g1 = stablehlo.subtract %adsubd3g1, %adwdpd3g1 : tensor<256xf32>
    %adb1d3bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3bt1 = stablehlo.multiply %adb1d3bt1, %d3bt1m : tensor<256xf32>
    %admgd3bt1 = stablehlo.multiply %adob1d3bt1, %d3dn1db : tensor<256xf32>
    %admnd3bt1 = stablehlo.add %admsd3bt1, %admgd3bt1 : tensor<256xf32>
    %adb2d3bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3bt1 = stablehlo.multiply %adb2d3bt1, %d3bt1v : tensor<256xf32>
    %adg2d3bt1 = stablehlo.multiply %d3dn1db, %d3dn1db : tensor<256xf32>
    %advgd3bt1 = stablehlo.multiply %adob2d3bt1, %adg2d3bt1 : tensor<256xf32>
    %advnd3bt1 = stablehlo.add %advsd3bt1, %advgd3bt1 : tensor<256xf32>
    %adbc1d3bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3bt1 = stablehlo.divide %admnd3bt1, %adbc1d3bt1 : tensor<256xf32>
    %advhd3bt1 = stablehlo.divide %advnd3bt1, %adbc2d3bt1 : tensor<256xf32>
    %adlrd3bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3bt1 = stablehlo.sqrt %advhd3bt1 : tensor<256xf32>
    %addend3bt1 = stablehlo.add %adsqd3bt1, %adepsd3bt1 : tensor<256xf32>
    %adratd3bt1 = stablehlo.divide %admhd3bt1, %addend3bt1 : tensor<256xf32>
    %adstd3bt1 = stablehlo.multiply %adlrd3bt1, %adratd3bt1 : tensor<256xf32>
    %adsubd3bt1 = stablehlo.subtract %d3bt1, %adstd3bt1 : tensor<256xf32>
    %adwdd3bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3bt1 = stablehlo.multiply %adwdd3bt1, %adlrd3bt1 : tensor<256xf32>
    %adwdpd3bt1 = stablehlo.multiply %adwdlrd3bt1, %d3bt1 : tensor<256xf32>
    %adnewd3bt1 = stablehlo.subtract %adsubd3bt1, %adwdpd3bt1 : tensor<256xf32>
    %adb1d3W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1d3W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admsd3W2 = stablehlo.multiply %adb1d3W2, %d3W2m : tensor<256x256x3x3xf32>
    %admgd3W2 = stablehlo.multiply %adob1d3W2, %d3dW2 : tensor<256x256x3x3xf32>
    %admnd3W2 = stablehlo.add %admsd3W2, %admgd3W2 : tensor<256x256x3x3xf32>
    %adb2d3W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2d3W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advsd3W2 = stablehlo.multiply %adb2d3W2, %d3W2v : tensor<256x256x3x3xf32>
    %adg2d3W2 = stablehlo.multiply %d3dW2, %d3dW2 : tensor<256x256x3x3xf32>
    %advgd3W2 = stablehlo.multiply %adob2d3W2, %adg2d3W2 : tensor<256x256x3x3xf32>
    %advnd3W2 = stablehlo.add %advsd3W2, %advgd3W2 : tensor<256x256x3x3xf32>
    %adbc1d3W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2d3W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhd3W2 = stablehlo.divide %admnd3W2, %adbc1d3W2 : tensor<256x256x3x3xf32>
    %advhd3W2 = stablehlo.divide %advnd3W2, %adbc2d3W2 : tensor<256x256x3x3xf32>
    %adlrd3W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepsd3W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqd3W2 = stablehlo.sqrt %advhd3W2 : tensor<256x256x3x3xf32>
    %addend3W2 = stablehlo.add %adsqd3W2, %adepsd3W2 : tensor<256x256x3x3xf32>
    %adratd3W2 = stablehlo.divide %admhd3W2, %addend3W2 : tensor<256x256x3x3xf32>
    %adstd3W2 = stablehlo.multiply %adlrd3W2, %adratd3W2 : tensor<256x256x3x3xf32>
    %adsubd3W2 = stablehlo.subtract %d3W2, %adstd3W2 : tensor<256x256x3x3xf32>
    %adwdd3W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrd3W2 = stablehlo.multiply %adwdd3W2, %adlrd3W2 : tensor<256x256x3x3xf32>
    %adwdpd3W2 = stablehlo.multiply %adwdlrd3W2, %d3W2 : tensor<256x256x3x3xf32>
    %adnewd3W2 = stablehlo.subtract %adsubd3W2, %adwdpd3W2 : tensor<256x256x3x3xf32>
    %adb1d3b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3b2 = stablehlo.multiply %adb1d3b2, %d3b2m : tensor<256xf32>
    %admgd3b2 = stablehlo.multiply %adob1d3b2, %d3db2 : tensor<256xf32>
    %admnd3b2 = stablehlo.add %admsd3b2, %admgd3b2 : tensor<256xf32>
    %adb2d3b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3b2 = stablehlo.multiply %adb2d3b2, %d3b2v : tensor<256xf32>
    %adg2d3b2 = stablehlo.multiply %d3db2, %d3db2 : tensor<256xf32>
    %advgd3b2 = stablehlo.multiply %adob2d3b2, %adg2d3b2 : tensor<256xf32>
    %advnd3b2 = stablehlo.add %advsd3b2, %advgd3b2 : tensor<256xf32>
    %adbc1d3b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3b2 = stablehlo.divide %admnd3b2, %adbc1d3b2 : tensor<256xf32>
    %advhd3b2 = stablehlo.divide %advnd3b2, %adbc2d3b2 : tensor<256xf32>
    %adlrd3b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3b2 = stablehlo.sqrt %advhd3b2 : tensor<256xf32>
    %addend3b2 = stablehlo.add %adsqd3b2, %adepsd3b2 : tensor<256xf32>
    %adratd3b2 = stablehlo.divide %admhd3b2, %addend3b2 : tensor<256xf32>
    %adstd3b2 = stablehlo.multiply %adlrd3b2, %adratd3b2 : tensor<256xf32>
    %adsubd3b2 = stablehlo.subtract %d3b2, %adstd3b2 : tensor<256xf32>
    %adwdd3b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3b2 = stablehlo.multiply %adwdd3b2, %adlrd3b2 : tensor<256xf32>
    %adwdpd3b2 = stablehlo.multiply %adwdlrd3b2, %d3b2 : tensor<256xf32>
    %adnewd3b2 = stablehlo.subtract %adsubd3b2, %adwdpd3b2 : tensor<256xf32>
    %adb1d3g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3g2 = stablehlo.multiply %adb1d3g2, %d3g2m : tensor<256xf32>
    %admgd3g2 = stablehlo.multiply %adob1d3g2, %d3dn2dg : tensor<256xf32>
    %admnd3g2 = stablehlo.add %admsd3g2, %admgd3g2 : tensor<256xf32>
    %adb2d3g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3g2 = stablehlo.multiply %adb2d3g2, %d3g2v : tensor<256xf32>
    %adg2d3g2 = stablehlo.multiply %d3dn2dg, %d3dn2dg : tensor<256xf32>
    %advgd3g2 = stablehlo.multiply %adob2d3g2, %adg2d3g2 : tensor<256xf32>
    %advnd3g2 = stablehlo.add %advsd3g2, %advgd3g2 : tensor<256xf32>
    %adbc1d3g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3g2 = stablehlo.divide %admnd3g2, %adbc1d3g2 : tensor<256xf32>
    %advhd3g2 = stablehlo.divide %advnd3g2, %adbc2d3g2 : tensor<256xf32>
    %adlrd3g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3g2 = stablehlo.sqrt %advhd3g2 : tensor<256xf32>
    %addend3g2 = stablehlo.add %adsqd3g2, %adepsd3g2 : tensor<256xf32>
    %adratd3g2 = stablehlo.divide %admhd3g2, %addend3g2 : tensor<256xf32>
    %adstd3g2 = stablehlo.multiply %adlrd3g2, %adratd3g2 : tensor<256xf32>
    %adsubd3g2 = stablehlo.subtract %d3g2, %adstd3g2 : tensor<256xf32>
    %adwdd3g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3g2 = stablehlo.multiply %adwdd3g2, %adlrd3g2 : tensor<256xf32>
    %adwdpd3g2 = stablehlo.multiply %adwdlrd3g2, %d3g2 : tensor<256xf32>
    %adnewd3g2 = stablehlo.subtract %adsubd3g2, %adwdpd3g2 : tensor<256xf32>
    %adb1d3bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3bt2 = stablehlo.multiply %adb1d3bt2, %d3bt2m : tensor<256xf32>
    %admgd3bt2 = stablehlo.multiply %adob1d3bt2, %d3dn2db : tensor<256xf32>
    %admnd3bt2 = stablehlo.add %admsd3bt2, %admgd3bt2 : tensor<256xf32>
    %adb2d3bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3bt2 = stablehlo.multiply %adb2d3bt2, %d3bt2v : tensor<256xf32>
    %adg2d3bt2 = stablehlo.multiply %d3dn2db, %d3dn2db : tensor<256xf32>
    %advgd3bt2 = stablehlo.multiply %adob2d3bt2, %adg2d3bt2 : tensor<256xf32>
    %advnd3bt2 = stablehlo.add %advsd3bt2, %advgd3bt2 : tensor<256xf32>
    %adbc1d3bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3bt2 = stablehlo.divide %admnd3bt2, %adbc1d3bt2 : tensor<256xf32>
    %advhd3bt2 = stablehlo.divide %advnd3bt2, %adbc2d3bt2 : tensor<256xf32>
    %adlrd3bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3bt2 = stablehlo.sqrt %advhd3bt2 : tensor<256xf32>
    %addend3bt2 = stablehlo.add %adsqd3bt2, %adepsd3bt2 : tensor<256xf32>
    %adratd3bt2 = stablehlo.divide %admhd3bt2, %addend3bt2 : tensor<256xf32>
    %adstd3bt2 = stablehlo.multiply %adlrd3bt2, %adratd3bt2 : tensor<256xf32>
    %adsubd3bt2 = stablehlo.subtract %d3bt2, %adstd3bt2 : tensor<256xf32>
    %adwdd3bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3bt2 = stablehlo.multiply %adwdd3bt2, %adlrd3bt2 : tensor<256xf32>
    %adwdpd3bt2 = stablehlo.multiply %adwdlrd3bt2, %d3bt2 : tensor<256xf32>
    %adnewd3bt2 = stablehlo.subtract %adsubd3bt2, %adwdpd3bt2 : tensor<256xf32>
    %adb1d3Wp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adob1d3Wp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %admsd3Wp = stablehlo.multiply %adb1d3Wp, %d3Wpm : tensor<256x128x3x3xf32>
    %admgd3Wp = stablehlo.multiply %adob1d3Wp, %d3dWp : tensor<256x128x3x3xf32>
    %admnd3Wp = stablehlo.add %admsd3Wp, %admgd3Wp : tensor<256x128x3x3xf32>
    %adb2d3Wp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adob2d3Wp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %advsd3Wp = stablehlo.multiply %adb2d3Wp, %d3Wpv : tensor<256x128x3x3xf32>
    %adg2d3Wp = stablehlo.multiply %d3dWp, %d3dWp : tensor<256x128x3x3xf32>
    %advgd3Wp = stablehlo.multiply %adob2d3Wp, %adg2d3Wp : tensor<256x128x3x3xf32>
    %advnd3Wp = stablehlo.add %advsd3Wp, %advgd3Wp : tensor<256x128x3x3xf32>
    %adbc1d3Wp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adbc2d3Wp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %admhd3Wp = stablehlo.divide %admnd3Wp, %adbc1d3Wp : tensor<256x128x3x3xf32>
    %advhd3Wp = stablehlo.divide %advnd3Wp, %adbc2d3Wp : tensor<256x128x3x3xf32>
    %adlrd3Wp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adepsd3Wp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adsqd3Wp = stablehlo.sqrt %advhd3Wp : tensor<256x128x3x3xf32>
    %addend3Wp = stablehlo.add %adsqd3Wp, %adepsd3Wp : tensor<256x128x3x3xf32>
    %adratd3Wp = stablehlo.divide %admhd3Wp, %addend3Wp : tensor<256x128x3x3xf32>
    %adstd3Wp = stablehlo.multiply %adlrd3Wp, %adratd3Wp : tensor<256x128x3x3xf32>
    %adsubd3Wp = stablehlo.subtract %d3Wp, %adstd3Wp : tensor<256x128x3x3xf32>
    %adwdd3Wp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %adwdlrd3Wp = stablehlo.multiply %adwdd3Wp, %adlrd3Wp : tensor<256x128x3x3xf32>
    %adwdpd3Wp = stablehlo.multiply %adwdlrd3Wp, %d3Wp : tensor<256x128x3x3xf32>
    %adnewd3Wp = stablehlo.subtract %adsubd3Wp, %adwdpd3Wp : tensor<256x128x3x3xf32>
    %adb1d3bp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3bp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3bp = stablehlo.multiply %adb1d3bp, %d3bpm : tensor<256xf32>
    %admgd3bp = stablehlo.multiply %adob1d3bp, %d3dbp : tensor<256xf32>
    %admnd3bp = stablehlo.add %admsd3bp, %admgd3bp : tensor<256xf32>
    %adb2d3bp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3bp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3bp = stablehlo.multiply %adb2d3bp, %d3bpv : tensor<256xf32>
    %adg2d3bp = stablehlo.multiply %d3dbp, %d3dbp : tensor<256xf32>
    %advgd3bp = stablehlo.multiply %adob2d3bp, %adg2d3bp : tensor<256xf32>
    %advnd3bp = stablehlo.add %advsd3bp, %advgd3bp : tensor<256xf32>
    %adbc1d3bp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3bp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3bp = stablehlo.divide %admnd3bp, %adbc1d3bp : tensor<256xf32>
    %advhd3bp = stablehlo.divide %advnd3bp, %adbc2d3bp : tensor<256xf32>
    %adlrd3bp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3bp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3bp = stablehlo.sqrt %advhd3bp : tensor<256xf32>
    %addend3bp = stablehlo.add %adsqd3bp, %adepsd3bp : tensor<256xf32>
    %adratd3bp = stablehlo.divide %admhd3bp, %addend3bp : tensor<256xf32>
    %adstd3bp = stablehlo.multiply %adlrd3bp, %adratd3bp : tensor<256xf32>
    %adsubd3bp = stablehlo.subtract %d3bp, %adstd3bp : tensor<256xf32>
    %adwdd3bp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3bp = stablehlo.multiply %adwdd3bp, %adlrd3bp : tensor<256xf32>
    %adwdpd3bp = stablehlo.multiply %adwdlrd3bp, %d3bp : tensor<256xf32>
    %adnewd3bp = stablehlo.subtract %adsubd3bp, %adwdpd3bp : tensor<256xf32>
    %adb1d3gp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3gp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3gp = stablehlo.multiply %adb1d3gp, %d3gpm : tensor<256xf32>
    %admgd3gp = stablehlo.multiply %adob1d3gp, %d3dnpdg : tensor<256xf32>
    %admnd3gp = stablehlo.add %admsd3gp, %admgd3gp : tensor<256xf32>
    %adb2d3gp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3gp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3gp = stablehlo.multiply %adb2d3gp, %d3gpv : tensor<256xf32>
    %adg2d3gp = stablehlo.multiply %d3dnpdg, %d3dnpdg : tensor<256xf32>
    %advgd3gp = stablehlo.multiply %adob2d3gp, %adg2d3gp : tensor<256xf32>
    %advnd3gp = stablehlo.add %advsd3gp, %advgd3gp : tensor<256xf32>
    %adbc1d3gp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3gp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3gp = stablehlo.divide %admnd3gp, %adbc1d3gp : tensor<256xf32>
    %advhd3gp = stablehlo.divide %advnd3gp, %adbc2d3gp : tensor<256xf32>
    %adlrd3gp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3gp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3gp = stablehlo.sqrt %advhd3gp : tensor<256xf32>
    %addend3gp = stablehlo.add %adsqd3gp, %adepsd3gp : tensor<256xf32>
    %adratd3gp = stablehlo.divide %admhd3gp, %addend3gp : tensor<256xf32>
    %adstd3gp = stablehlo.multiply %adlrd3gp, %adratd3gp : tensor<256xf32>
    %adsubd3gp = stablehlo.subtract %d3gp, %adstd3gp : tensor<256xf32>
    %adwdd3gp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3gp = stablehlo.multiply %adwdd3gp, %adlrd3gp : tensor<256xf32>
    %adwdpd3gp = stablehlo.multiply %adwdlrd3gp, %d3gp : tensor<256xf32>
    %adnewd3gp = stablehlo.subtract %adsubd3gp, %adwdpd3gp : tensor<256xf32>
    %adb1d3btp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1d3btp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admsd3btp = stablehlo.multiply %adb1d3btp, %d3btpm : tensor<256xf32>
    %admgd3btp = stablehlo.multiply %adob1d3btp, %d3dnpdb : tensor<256xf32>
    %admnd3btp = stablehlo.add %admsd3btp, %admgd3btp : tensor<256xf32>
    %adb2d3btp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2d3btp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advsd3btp = stablehlo.multiply %adb2d3btp, %d3btpv : tensor<256xf32>
    %adg2d3btp = stablehlo.multiply %d3dnpdb, %d3dnpdb : tensor<256xf32>
    %advgd3btp = stablehlo.multiply %adob2d3btp, %adg2d3btp : tensor<256xf32>
    %advnd3btp = stablehlo.add %advsd3btp, %advgd3btp : tensor<256xf32>
    %adbc1d3btp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2d3btp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhd3btp = stablehlo.divide %admnd3btp, %adbc1d3btp : tensor<256xf32>
    %advhd3btp = stablehlo.divide %advnd3btp, %adbc2d3btp : tensor<256xf32>
    %adlrd3btp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepsd3btp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqd3btp = stablehlo.sqrt %advhd3btp : tensor<256xf32>
    %addend3btp = stablehlo.add %adsqd3btp, %adepsd3btp : tensor<256xf32>
    %adratd3btp = stablehlo.divide %admhd3btp, %addend3btp : tensor<256xf32>
    %adstd3btp = stablehlo.multiply %adlrd3btp, %adratd3btp : tensor<256xf32>
    %adsubd3btp = stablehlo.subtract %d3btp, %adstd3btp : tensor<256xf32>
    %adwdd3btp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrd3btp = stablehlo.multiply %adwdd3btp, %adlrd3btp : tensor<256xf32>
    %adwdpd3btp = stablehlo.multiply %adwdlrd3btp, %d3btp : tensor<256xf32>
    %adnewd3btp = stablehlo.subtract %adsubd3btp, %adwdpd3btp : tensor<256xf32>
    %adb1s3b0W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b0W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b0W1 = stablehlo.multiply %adb1s3b0W1, %s3b0W1m : tensor<256x256x3x3xf32>
    %admgs3b0W1 = stablehlo.multiply %adob1s3b0W1, %s3b0dW1 : tensor<256x256x3x3xf32>
    %admns3b0W1 = stablehlo.add %admss3b0W1, %admgs3b0W1 : tensor<256x256x3x3xf32>
    %adb2s3b0W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b0W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b0W1 = stablehlo.multiply %adb2s3b0W1, %s3b0W1v : tensor<256x256x3x3xf32>
    %adg2s3b0W1 = stablehlo.multiply %s3b0dW1, %s3b0dW1 : tensor<256x256x3x3xf32>
    %advgs3b0W1 = stablehlo.multiply %adob2s3b0W1, %adg2s3b0W1 : tensor<256x256x3x3xf32>
    %advns3b0W1 = stablehlo.add %advss3b0W1, %advgs3b0W1 : tensor<256x256x3x3xf32>
    %adbc1s3b0W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b0W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b0W1 = stablehlo.divide %admns3b0W1, %adbc1s3b0W1 : tensor<256x256x3x3xf32>
    %advhs3b0W1 = stablehlo.divide %advns3b0W1, %adbc2s3b0W1 : tensor<256x256x3x3xf32>
    %adlrs3b0W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b0W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b0W1 = stablehlo.sqrt %advhs3b0W1 : tensor<256x256x3x3xf32>
    %addens3b0W1 = stablehlo.add %adsqs3b0W1, %adepss3b0W1 : tensor<256x256x3x3xf32>
    %adrats3b0W1 = stablehlo.divide %admhs3b0W1, %addens3b0W1 : tensor<256x256x3x3xf32>
    %adsts3b0W1 = stablehlo.multiply %adlrs3b0W1, %adrats3b0W1 : tensor<256x256x3x3xf32>
    %adsubs3b0W1 = stablehlo.subtract %s3b0W1, %adsts3b0W1 : tensor<256x256x3x3xf32>
    %adwds3b0W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b0W1 = stablehlo.multiply %adwds3b0W1, %adlrs3b0W1 : tensor<256x256x3x3xf32>
    %adwdps3b0W1 = stablehlo.multiply %adwdlrs3b0W1, %s3b0W1 : tensor<256x256x3x3xf32>
    %adnews3b0W1 = stablehlo.subtract %adsubs3b0W1, %adwdps3b0W1 : tensor<256x256x3x3xf32>
    %adb1s3b0b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0b1 = stablehlo.multiply %adb1s3b0b1, %s3b0b1m : tensor<256xf32>
    %admgs3b0b1 = stablehlo.multiply %adob1s3b0b1, %s3b0db1 : tensor<256xf32>
    %admns3b0b1 = stablehlo.add %admss3b0b1, %admgs3b0b1 : tensor<256xf32>
    %adb2s3b0b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0b1 = stablehlo.multiply %adb2s3b0b1, %s3b0b1v : tensor<256xf32>
    %adg2s3b0b1 = stablehlo.multiply %s3b0db1, %s3b0db1 : tensor<256xf32>
    %advgs3b0b1 = stablehlo.multiply %adob2s3b0b1, %adg2s3b0b1 : tensor<256xf32>
    %advns3b0b1 = stablehlo.add %advss3b0b1, %advgs3b0b1 : tensor<256xf32>
    %adbc1s3b0b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0b1 = stablehlo.divide %admns3b0b1, %adbc1s3b0b1 : tensor<256xf32>
    %advhs3b0b1 = stablehlo.divide %advns3b0b1, %adbc2s3b0b1 : tensor<256xf32>
    %adlrs3b0b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0b1 = stablehlo.sqrt %advhs3b0b1 : tensor<256xf32>
    %addens3b0b1 = stablehlo.add %adsqs3b0b1, %adepss3b0b1 : tensor<256xf32>
    %adrats3b0b1 = stablehlo.divide %admhs3b0b1, %addens3b0b1 : tensor<256xf32>
    %adsts3b0b1 = stablehlo.multiply %adlrs3b0b1, %adrats3b0b1 : tensor<256xf32>
    %adsubs3b0b1 = stablehlo.subtract %s3b0b1, %adsts3b0b1 : tensor<256xf32>
    %adwds3b0b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0b1 = stablehlo.multiply %adwds3b0b1, %adlrs3b0b1 : tensor<256xf32>
    %adwdps3b0b1 = stablehlo.multiply %adwdlrs3b0b1, %s3b0b1 : tensor<256xf32>
    %adnews3b0b1 = stablehlo.subtract %adsubs3b0b1, %adwdps3b0b1 : tensor<256xf32>
    %adb1s3b0g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0g1 = stablehlo.multiply %adb1s3b0g1, %s3b0g1m : tensor<256xf32>
    %admgs3b0g1 = stablehlo.multiply %adob1s3b0g1, %s3b0dn1dg : tensor<256xf32>
    %admns3b0g1 = stablehlo.add %admss3b0g1, %admgs3b0g1 : tensor<256xf32>
    %adb2s3b0g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0g1 = stablehlo.multiply %adb2s3b0g1, %s3b0g1v : tensor<256xf32>
    %adg2s3b0g1 = stablehlo.multiply %s3b0dn1dg, %s3b0dn1dg : tensor<256xf32>
    %advgs3b0g1 = stablehlo.multiply %adob2s3b0g1, %adg2s3b0g1 : tensor<256xf32>
    %advns3b0g1 = stablehlo.add %advss3b0g1, %advgs3b0g1 : tensor<256xf32>
    %adbc1s3b0g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0g1 = stablehlo.divide %admns3b0g1, %adbc1s3b0g1 : tensor<256xf32>
    %advhs3b0g1 = stablehlo.divide %advns3b0g1, %adbc2s3b0g1 : tensor<256xf32>
    %adlrs3b0g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0g1 = stablehlo.sqrt %advhs3b0g1 : tensor<256xf32>
    %addens3b0g1 = stablehlo.add %adsqs3b0g1, %adepss3b0g1 : tensor<256xf32>
    %adrats3b0g1 = stablehlo.divide %admhs3b0g1, %addens3b0g1 : tensor<256xf32>
    %adsts3b0g1 = stablehlo.multiply %adlrs3b0g1, %adrats3b0g1 : tensor<256xf32>
    %adsubs3b0g1 = stablehlo.subtract %s3b0g1, %adsts3b0g1 : tensor<256xf32>
    %adwds3b0g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0g1 = stablehlo.multiply %adwds3b0g1, %adlrs3b0g1 : tensor<256xf32>
    %adwdps3b0g1 = stablehlo.multiply %adwdlrs3b0g1, %s3b0g1 : tensor<256xf32>
    %adnews3b0g1 = stablehlo.subtract %adsubs3b0g1, %adwdps3b0g1 : tensor<256xf32>
    %adb1s3b0bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0bt1 = stablehlo.multiply %adb1s3b0bt1, %s3b0bt1m : tensor<256xf32>
    %admgs3b0bt1 = stablehlo.multiply %adob1s3b0bt1, %s3b0dn1db : tensor<256xf32>
    %admns3b0bt1 = stablehlo.add %admss3b0bt1, %admgs3b0bt1 : tensor<256xf32>
    %adb2s3b0bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0bt1 = stablehlo.multiply %adb2s3b0bt1, %s3b0bt1v : tensor<256xf32>
    %adg2s3b0bt1 = stablehlo.multiply %s3b0dn1db, %s3b0dn1db : tensor<256xf32>
    %advgs3b0bt1 = stablehlo.multiply %adob2s3b0bt1, %adg2s3b0bt1 : tensor<256xf32>
    %advns3b0bt1 = stablehlo.add %advss3b0bt1, %advgs3b0bt1 : tensor<256xf32>
    %adbc1s3b0bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0bt1 = stablehlo.divide %admns3b0bt1, %adbc1s3b0bt1 : tensor<256xf32>
    %advhs3b0bt1 = stablehlo.divide %advns3b0bt1, %adbc2s3b0bt1 : tensor<256xf32>
    %adlrs3b0bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0bt1 = stablehlo.sqrt %advhs3b0bt1 : tensor<256xf32>
    %addens3b0bt1 = stablehlo.add %adsqs3b0bt1, %adepss3b0bt1 : tensor<256xf32>
    %adrats3b0bt1 = stablehlo.divide %admhs3b0bt1, %addens3b0bt1 : tensor<256xf32>
    %adsts3b0bt1 = stablehlo.multiply %adlrs3b0bt1, %adrats3b0bt1 : tensor<256xf32>
    %adsubs3b0bt1 = stablehlo.subtract %s3b0bt1, %adsts3b0bt1 : tensor<256xf32>
    %adwds3b0bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0bt1 = stablehlo.multiply %adwds3b0bt1, %adlrs3b0bt1 : tensor<256xf32>
    %adwdps3b0bt1 = stablehlo.multiply %adwdlrs3b0bt1, %s3b0bt1 : tensor<256xf32>
    %adnews3b0bt1 = stablehlo.subtract %adsubs3b0bt1, %adwdps3b0bt1 : tensor<256xf32>
    %adb1s3b0W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b0W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b0W2 = stablehlo.multiply %adb1s3b0W2, %s3b0W2m : tensor<256x256x3x3xf32>
    %admgs3b0W2 = stablehlo.multiply %adob1s3b0W2, %s3b0dW2 : tensor<256x256x3x3xf32>
    %admns3b0W2 = stablehlo.add %admss3b0W2, %admgs3b0W2 : tensor<256x256x3x3xf32>
    %adb2s3b0W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b0W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b0W2 = stablehlo.multiply %adb2s3b0W2, %s3b0W2v : tensor<256x256x3x3xf32>
    %adg2s3b0W2 = stablehlo.multiply %s3b0dW2, %s3b0dW2 : tensor<256x256x3x3xf32>
    %advgs3b0W2 = stablehlo.multiply %adob2s3b0W2, %adg2s3b0W2 : tensor<256x256x3x3xf32>
    %advns3b0W2 = stablehlo.add %advss3b0W2, %advgs3b0W2 : tensor<256x256x3x3xf32>
    %adbc1s3b0W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b0W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b0W2 = stablehlo.divide %admns3b0W2, %adbc1s3b0W2 : tensor<256x256x3x3xf32>
    %advhs3b0W2 = stablehlo.divide %advns3b0W2, %adbc2s3b0W2 : tensor<256x256x3x3xf32>
    %adlrs3b0W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b0W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b0W2 = stablehlo.sqrt %advhs3b0W2 : tensor<256x256x3x3xf32>
    %addens3b0W2 = stablehlo.add %adsqs3b0W2, %adepss3b0W2 : tensor<256x256x3x3xf32>
    %adrats3b0W2 = stablehlo.divide %admhs3b0W2, %addens3b0W2 : tensor<256x256x3x3xf32>
    %adsts3b0W2 = stablehlo.multiply %adlrs3b0W2, %adrats3b0W2 : tensor<256x256x3x3xf32>
    %adsubs3b0W2 = stablehlo.subtract %s3b0W2, %adsts3b0W2 : tensor<256x256x3x3xf32>
    %adwds3b0W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b0W2 = stablehlo.multiply %adwds3b0W2, %adlrs3b0W2 : tensor<256x256x3x3xf32>
    %adwdps3b0W2 = stablehlo.multiply %adwdlrs3b0W2, %s3b0W2 : tensor<256x256x3x3xf32>
    %adnews3b0W2 = stablehlo.subtract %adsubs3b0W2, %adwdps3b0W2 : tensor<256x256x3x3xf32>
    %adb1s3b0b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0b2 = stablehlo.multiply %adb1s3b0b2, %s3b0b2m : tensor<256xf32>
    %admgs3b0b2 = stablehlo.multiply %adob1s3b0b2, %s3b0db2 : tensor<256xf32>
    %admns3b0b2 = stablehlo.add %admss3b0b2, %admgs3b0b2 : tensor<256xf32>
    %adb2s3b0b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0b2 = stablehlo.multiply %adb2s3b0b2, %s3b0b2v : tensor<256xf32>
    %adg2s3b0b2 = stablehlo.multiply %s3b0db2, %s3b0db2 : tensor<256xf32>
    %advgs3b0b2 = stablehlo.multiply %adob2s3b0b2, %adg2s3b0b2 : tensor<256xf32>
    %advns3b0b2 = stablehlo.add %advss3b0b2, %advgs3b0b2 : tensor<256xf32>
    %adbc1s3b0b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0b2 = stablehlo.divide %admns3b0b2, %adbc1s3b0b2 : tensor<256xf32>
    %advhs3b0b2 = stablehlo.divide %advns3b0b2, %adbc2s3b0b2 : tensor<256xf32>
    %adlrs3b0b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0b2 = stablehlo.sqrt %advhs3b0b2 : tensor<256xf32>
    %addens3b0b2 = stablehlo.add %adsqs3b0b2, %adepss3b0b2 : tensor<256xf32>
    %adrats3b0b2 = stablehlo.divide %admhs3b0b2, %addens3b0b2 : tensor<256xf32>
    %adsts3b0b2 = stablehlo.multiply %adlrs3b0b2, %adrats3b0b2 : tensor<256xf32>
    %adsubs3b0b2 = stablehlo.subtract %s3b0b2, %adsts3b0b2 : tensor<256xf32>
    %adwds3b0b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0b2 = stablehlo.multiply %adwds3b0b2, %adlrs3b0b2 : tensor<256xf32>
    %adwdps3b0b2 = stablehlo.multiply %adwdlrs3b0b2, %s3b0b2 : tensor<256xf32>
    %adnews3b0b2 = stablehlo.subtract %adsubs3b0b2, %adwdps3b0b2 : tensor<256xf32>
    %adb1s3b0g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0g2 = stablehlo.multiply %adb1s3b0g2, %s3b0g2m : tensor<256xf32>
    %admgs3b0g2 = stablehlo.multiply %adob1s3b0g2, %s3b0dn2dg : tensor<256xf32>
    %admns3b0g2 = stablehlo.add %admss3b0g2, %admgs3b0g2 : tensor<256xf32>
    %adb2s3b0g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0g2 = stablehlo.multiply %adb2s3b0g2, %s3b0g2v : tensor<256xf32>
    %adg2s3b0g2 = stablehlo.multiply %s3b0dn2dg, %s3b0dn2dg : tensor<256xf32>
    %advgs3b0g2 = stablehlo.multiply %adob2s3b0g2, %adg2s3b0g2 : tensor<256xf32>
    %advns3b0g2 = stablehlo.add %advss3b0g2, %advgs3b0g2 : tensor<256xf32>
    %adbc1s3b0g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0g2 = stablehlo.divide %admns3b0g2, %adbc1s3b0g2 : tensor<256xf32>
    %advhs3b0g2 = stablehlo.divide %advns3b0g2, %adbc2s3b0g2 : tensor<256xf32>
    %adlrs3b0g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0g2 = stablehlo.sqrt %advhs3b0g2 : tensor<256xf32>
    %addens3b0g2 = stablehlo.add %adsqs3b0g2, %adepss3b0g2 : tensor<256xf32>
    %adrats3b0g2 = stablehlo.divide %admhs3b0g2, %addens3b0g2 : tensor<256xf32>
    %adsts3b0g2 = stablehlo.multiply %adlrs3b0g2, %adrats3b0g2 : tensor<256xf32>
    %adsubs3b0g2 = stablehlo.subtract %s3b0g2, %adsts3b0g2 : tensor<256xf32>
    %adwds3b0g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0g2 = stablehlo.multiply %adwds3b0g2, %adlrs3b0g2 : tensor<256xf32>
    %adwdps3b0g2 = stablehlo.multiply %adwdlrs3b0g2, %s3b0g2 : tensor<256xf32>
    %adnews3b0g2 = stablehlo.subtract %adsubs3b0g2, %adwdps3b0g2 : tensor<256xf32>
    %adb1s3b0bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b0bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b0bt2 = stablehlo.multiply %adb1s3b0bt2, %s3b0bt2m : tensor<256xf32>
    %admgs3b0bt2 = stablehlo.multiply %adob1s3b0bt2, %s3b0dn2db : tensor<256xf32>
    %admns3b0bt2 = stablehlo.add %admss3b0bt2, %admgs3b0bt2 : tensor<256xf32>
    %adb2s3b0bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b0bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b0bt2 = stablehlo.multiply %adb2s3b0bt2, %s3b0bt2v : tensor<256xf32>
    %adg2s3b0bt2 = stablehlo.multiply %s3b0dn2db, %s3b0dn2db : tensor<256xf32>
    %advgs3b0bt2 = stablehlo.multiply %adob2s3b0bt2, %adg2s3b0bt2 : tensor<256xf32>
    %advns3b0bt2 = stablehlo.add %advss3b0bt2, %advgs3b0bt2 : tensor<256xf32>
    %adbc1s3b0bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b0bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b0bt2 = stablehlo.divide %admns3b0bt2, %adbc1s3b0bt2 : tensor<256xf32>
    %advhs3b0bt2 = stablehlo.divide %advns3b0bt2, %adbc2s3b0bt2 : tensor<256xf32>
    %adlrs3b0bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b0bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b0bt2 = stablehlo.sqrt %advhs3b0bt2 : tensor<256xf32>
    %addens3b0bt2 = stablehlo.add %adsqs3b0bt2, %adepss3b0bt2 : tensor<256xf32>
    %adrats3b0bt2 = stablehlo.divide %admhs3b0bt2, %addens3b0bt2 : tensor<256xf32>
    %adsts3b0bt2 = stablehlo.multiply %adlrs3b0bt2, %adrats3b0bt2 : tensor<256xf32>
    %adsubs3b0bt2 = stablehlo.subtract %s3b0bt2, %adsts3b0bt2 : tensor<256xf32>
    %adwds3b0bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b0bt2 = stablehlo.multiply %adwds3b0bt2, %adlrs3b0bt2 : tensor<256xf32>
    %adwdps3b0bt2 = stablehlo.multiply %adwdlrs3b0bt2, %s3b0bt2 : tensor<256xf32>
    %adnews3b0bt2 = stablehlo.subtract %adsubs3b0bt2, %adwdps3b0bt2 : tensor<256xf32>
    %adb1s3b1W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b1W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b1W1 = stablehlo.multiply %adb1s3b1W1, %s3b1W1m : tensor<256x256x3x3xf32>
    %admgs3b1W1 = stablehlo.multiply %adob1s3b1W1, %s3b1dW1 : tensor<256x256x3x3xf32>
    %admns3b1W1 = stablehlo.add %admss3b1W1, %admgs3b1W1 : tensor<256x256x3x3xf32>
    %adb2s3b1W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b1W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b1W1 = stablehlo.multiply %adb2s3b1W1, %s3b1W1v : tensor<256x256x3x3xf32>
    %adg2s3b1W1 = stablehlo.multiply %s3b1dW1, %s3b1dW1 : tensor<256x256x3x3xf32>
    %advgs3b1W1 = stablehlo.multiply %adob2s3b1W1, %adg2s3b1W1 : tensor<256x256x3x3xf32>
    %advns3b1W1 = stablehlo.add %advss3b1W1, %advgs3b1W1 : tensor<256x256x3x3xf32>
    %adbc1s3b1W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b1W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b1W1 = stablehlo.divide %admns3b1W1, %adbc1s3b1W1 : tensor<256x256x3x3xf32>
    %advhs3b1W1 = stablehlo.divide %advns3b1W1, %adbc2s3b1W1 : tensor<256x256x3x3xf32>
    %adlrs3b1W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b1W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b1W1 = stablehlo.sqrt %advhs3b1W1 : tensor<256x256x3x3xf32>
    %addens3b1W1 = stablehlo.add %adsqs3b1W1, %adepss3b1W1 : tensor<256x256x3x3xf32>
    %adrats3b1W1 = stablehlo.divide %admhs3b1W1, %addens3b1W1 : tensor<256x256x3x3xf32>
    %adsts3b1W1 = stablehlo.multiply %adlrs3b1W1, %adrats3b1W1 : tensor<256x256x3x3xf32>
    %adsubs3b1W1 = stablehlo.subtract %s3b1W1, %adsts3b1W1 : tensor<256x256x3x3xf32>
    %adwds3b1W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b1W1 = stablehlo.multiply %adwds3b1W1, %adlrs3b1W1 : tensor<256x256x3x3xf32>
    %adwdps3b1W1 = stablehlo.multiply %adwdlrs3b1W1, %s3b1W1 : tensor<256x256x3x3xf32>
    %adnews3b1W1 = stablehlo.subtract %adsubs3b1W1, %adwdps3b1W1 : tensor<256x256x3x3xf32>
    %adb1s3b1b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1b1 = stablehlo.multiply %adb1s3b1b1, %s3b1b1m : tensor<256xf32>
    %admgs3b1b1 = stablehlo.multiply %adob1s3b1b1, %s3b1db1 : tensor<256xf32>
    %admns3b1b1 = stablehlo.add %admss3b1b1, %admgs3b1b1 : tensor<256xf32>
    %adb2s3b1b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1b1 = stablehlo.multiply %adb2s3b1b1, %s3b1b1v : tensor<256xf32>
    %adg2s3b1b1 = stablehlo.multiply %s3b1db1, %s3b1db1 : tensor<256xf32>
    %advgs3b1b1 = stablehlo.multiply %adob2s3b1b1, %adg2s3b1b1 : tensor<256xf32>
    %advns3b1b1 = stablehlo.add %advss3b1b1, %advgs3b1b1 : tensor<256xf32>
    %adbc1s3b1b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1b1 = stablehlo.divide %admns3b1b1, %adbc1s3b1b1 : tensor<256xf32>
    %advhs3b1b1 = stablehlo.divide %advns3b1b1, %adbc2s3b1b1 : tensor<256xf32>
    %adlrs3b1b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1b1 = stablehlo.sqrt %advhs3b1b1 : tensor<256xf32>
    %addens3b1b1 = stablehlo.add %adsqs3b1b1, %adepss3b1b1 : tensor<256xf32>
    %adrats3b1b1 = stablehlo.divide %admhs3b1b1, %addens3b1b1 : tensor<256xf32>
    %adsts3b1b1 = stablehlo.multiply %adlrs3b1b1, %adrats3b1b1 : tensor<256xf32>
    %adsubs3b1b1 = stablehlo.subtract %s3b1b1, %adsts3b1b1 : tensor<256xf32>
    %adwds3b1b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1b1 = stablehlo.multiply %adwds3b1b1, %adlrs3b1b1 : tensor<256xf32>
    %adwdps3b1b1 = stablehlo.multiply %adwdlrs3b1b1, %s3b1b1 : tensor<256xf32>
    %adnews3b1b1 = stablehlo.subtract %adsubs3b1b1, %adwdps3b1b1 : tensor<256xf32>
    %adb1s3b1g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1g1 = stablehlo.multiply %adb1s3b1g1, %s3b1g1m : tensor<256xf32>
    %admgs3b1g1 = stablehlo.multiply %adob1s3b1g1, %s3b1dn1dg : tensor<256xf32>
    %admns3b1g1 = stablehlo.add %admss3b1g1, %admgs3b1g1 : tensor<256xf32>
    %adb2s3b1g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1g1 = stablehlo.multiply %adb2s3b1g1, %s3b1g1v : tensor<256xf32>
    %adg2s3b1g1 = stablehlo.multiply %s3b1dn1dg, %s3b1dn1dg : tensor<256xf32>
    %advgs3b1g1 = stablehlo.multiply %adob2s3b1g1, %adg2s3b1g1 : tensor<256xf32>
    %advns3b1g1 = stablehlo.add %advss3b1g1, %advgs3b1g1 : tensor<256xf32>
    %adbc1s3b1g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1g1 = stablehlo.divide %admns3b1g1, %adbc1s3b1g1 : tensor<256xf32>
    %advhs3b1g1 = stablehlo.divide %advns3b1g1, %adbc2s3b1g1 : tensor<256xf32>
    %adlrs3b1g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1g1 = stablehlo.sqrt %advhs3b1g1 : tensor<256xf32>
    %addens3b1g1 = stablehlo.add %adsqs3b1g1, %adepss3b1g1 : tensor<256xf32>
    %adrats3b1g1 = stablehlo.divide %admhs3b1g1, %addens3b1g1 : tensor<256xf32>
    %adsts3b1g1 = stablehlo.multiply %adlrs3b1g1, %adrats3b1g1 : tensor<256xf32>
    %adsubs3b1g1 = stablehlo.subtract %s3b1g1, %adsts3b1g1 : tensor<256xf32>
    %adwds3b1g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1g1 = stablehlo.multiply %adwds3b1g1, %adlrs3b1g1 : tensor<256xf32>
    %adwdps3b1g1 = stablehlo.multiply %adwdlrs3b1g1, %s3b1g1 : tensor<256xf32>
    %adnews3b1g1 = stablehlo.subtract %adsubs3b1g1, %adwdps3b1g1 : tensor<256xf32>
    %adb1s3b1bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1bt1 = stablehlo.multiply %adb1s3b1bt1, %s3b1bt1m : tensor<256xf32>
    %admgs3b1bt1 = stablehlo.multiply %adob1s3b1bt1, %s3b1dn1db : tensor<256xf32>
    %admns3b1bt1 = stablehlo.add %admss3b1bt1, %admgs3b1bt1 : tensor<256xf32>
    %adb2s3b1bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1bt1 = stablehlo.multiply %adb2s3b1bt1, %s3b1bt1v : tensor<256xf32>
    %adg2s3b1bt1 = stablehlo.multiply %s3b1dn1db, %s3b1dn1db : tensor<256xf32>
    %advgs3b1bt1 = stablehlo.multiply %adob2s3b1bt1, %adg2s3b1bt1 : tensor<256xf32>
    %advns3b1bt1 = stablehlo.add %advss3b1bt1, %advgs3b1bt1 : tensor<256xf32>
    %adbc1s3b1bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1bt1 = stablehlo.divide %admns3b1bt1, %adbc1s3b1bt1 : tensor<256xf32>
    %advhs3b1bt1 = stablehlo.divide %advns3b1bt1, %adbc2s3b1bt1 : tensor<256xf32>
    %adlrs3b1bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1bt1 = stablehlo.sqrt %advhs3b1bt1 : tensor<256xf32>
    %addens3b1bt1 = stablehlo.add %adsqs3b1bt1, %adepss3b1bt1 : tensor<256xf32>
    %adrats3b1bt1 = stablehlo.divide %admhs3b1bt1, %addens3b1bt1 : tensor<256xf32>
    %adsts3b1bt1 = stablehlo.multiply %adlrs3b1bt1, %adrats3b1bt1 : tensor<256xf32>
    %adsubs3b1bt1 = stablehlo.subtract %s3b1bt1, %adsts3b1bt1 : tensor<256xf32>
    %adwds3b1bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1bt1 = stablehlo.multiply %adwds3b1bt1, %adlrs3b1bt1 : tensor<256xf32>
    %adwdps3b1bt1 = stablehlo.multiply %adwdlrs3b1bt1, %s3b1bt1 : tensor<256xf32>
    %adnews3b1bt1 = stablehlo.subtract %adsubs3b1bt1, %adwdps3b1bt1 : tensor<256xf32>
    %adb1s3b1W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b1W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b1W2 = stablehlo.multiply %adb1s3b1W2, %s3b1W2m : tensor<256x256x3x3xf32>
    %admgs3b1W2 = stablehlo.multiply %adob1s3b1W2, %s3b1dW2 : tensor<256x256x3x3xf32>
    %admns3b1W2 = stablehlo.add %admss3b1W2, %admgs3b1W2 : tensor<256x256x3x3xf32>
    %adb2s3b1W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b1W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b1W2 = stablehlo.multiply %adb2s3b1W2, %s3b1W2v : tensor<256x256x3x3xf32>
    %adg2s3b1W2 = stablehlo.multiply %s3b1dW2, %s3b1dW2 : tensor<256x256x3x3xf32>
    %advgs3b1W2 = stablehlo.multiply %adob2s3b1W2, %adg2s3b1W2 : tensor<256x256x3x3xf32>
    %advns3b1W2 = stablehlo.add %advss3b1W2, %advgs3b1W2 : tensor<256x256x3x3xf32>
    %adbc1s3b1W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b1W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b1W2 = stablehlo.divide %admns3b1W2, %adbc1s3b1W2 : tensor<256x256x3x3xf32>
    %advhs3b1W2 = stablehlo.divide %advns3b1W2, %adbc2s3b1W2 : tensor<256x256x3x3xf32>
    %adlrs3b1W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b1W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b1W2 = stablehlo.sqrt %advhs3b1W2 : tensor<256x256x3x3xf32>
    %addens3b1W2 = stablehlo.add %adsqs3b1W2, %adepss3b1W2 : tensor<256x256x3x3xf32>
    %adrats3b1W2 = stablehlo.divide %admhs3b1W2, %addens3b1W2 : tensor<256x256x3x3xf32>
    %adsts3b1W2 = stablehlo.multiply %adlrs3b1W2, %adrats3b1W2 : tensor<256x256x3x3xf32>
    %adsubs3b1W2 = stablehlo.subtract %s3b1W2, %adsts3b1W2 : tensor<256x256x3x3xf32>
    %adwds3b1W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b1W2 = stablehlo.multiply %adwds3b1W2, %adlrs3b1W2 : tensor<256x256x3x3xf32>
    %adwdps3b1W2 = stablehlo.multiply %adwdlrs3b1W2, %s3b1W2 : tensor<256x256x3x3xf32>
    %adnews3b1W2 = stablehlo.subtract %adsubs3b1W2, %adwdps3b1W2 : tensor<256x256x3x3xf32>
    %adb1s3b1b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1b2 = stablehlo.multiply %adb1s3b1b2, %s3b1b2m : tensor<256xf32>
    %admgs3b1b2 = stablehlo.multiply %adob1s3b1b2, %s3b1db2 : tensor<256xf32>
    %admns3b1b2 = stablehlo.add %admss3b1b2, %admgs3b1b2 : tensor<256xf32>
    %adb2s3b1b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1b2 = stablehlo.multiply %adb2s3b1b2, %s3b1b2v : tensor<256xf32>
    %adg2s3b1b2 = stablehlo.multiply %s3b1db2, %s3b1db2 : tensor<256xf32>
    %advgs3b1b2 = stablehlo.multiply %adob2s3b1b2, %adg2s3b1b2 : tensor<256xf32>
    %advns3b1b2 = stablehlo.add %advss3b1b2, %advgs3b1b2 : tensor<256xf32>
    %adbc1s3b1b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1b2 = stablehlo.divide %admns3b1b2, %adbc1s3b1b2 : tensor<256xf32>
    %advhs3b1b2 = stablehlo.divide %advns3b1b2, %adbc2s3b1b2 : tensor<256xf32>
    %adlrs3b1b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1b2 = stablehlo.sqrt %advhs3b1b2 : tensor<256xf32>
    %addens3b1b2 = stablehlo.add %adsqs3b1b2, %adepss3b1b2 : tensor<256xf32>
    %adrats3b1b2 = stablehlo.divide %admhs3b1b2, %addens3b1b2 : tensor<256xf32>
    %adsts3b1b2 = stablehlo.multiply %adlrs3b1b2, %adrats3b1b2 : tensor<256xf32>
    %adsubs3b1b2 = stablehlo.subtract %s3b1b2, %adsts3b1b2 : tensor<256xf32>
    %adwds3b1b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1b2 = stablehlo.multiply %adwds3b1b2, %adlrs3b1b2 : tensor<256xf32>
    %adwdps3b1b2 = stablehlo.multiply %adwdlrs3b1b2, %s3b1b2 : tensor<256xf32>
    %adnews3b1b2 = stablehlo.subtract %adsubs3b1b2, %adwdps3b1b2 : tensor<256xf32>
    %adb1s3b1g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1g2 = stablehlo.multiply %adb1s3b1g2, %s3b1g2m : tensor<256xf32>
    %admgs3b1g2 = stablehlo.multiply %adob1s3b1g2, %s3b1dn2dg : tensor<256xf32>
    %admns3b1g2 = stablehlo.add %admss3b1g2, %admgs3b1g2 : tensor<256xf32>
    %adb2s3b1g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1g2 = stablehlo.multiply %adb2s3b1g2, %s3b1g2v : tensor<256xf32>
    %adg2s3b1g2 = stablehlo.multiply %s3b1dn2dg, %s3b1dn2dg : tensor<256xf32>
    %advgs3b1g2 = stablehlo.multiply %adob2s3b1g2, %adg2s3b1g2 : tensor<256xf32>
    %advns3b1g2 = stablehlo.add %advss3b1g2, %advgs3b1g2 : tensor<256xf32>
    %adbc1s3b1g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1g2 = stablehlo.divide %admns3b1g2, %adbc1s3b1g2 : tensor<256xf32>
    %advhs3b1g2 = stablehlo.divide %advns3b1g2, %adbc2s3b1g2 : tensor<256xf32>
    %adlrs3b1g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1g2 = stablehlo.sqrt %advhs3b1g2 : tensor<256xf32>
    %addens3b1g2 = stablehlo.add %adsqs3b1g2, %adepss3b1g2 : tensor<256xf32>
    %adrats3b1g2 = stablehlo.divide %admhs3b1g2, %addens3b1g2 : tensor<256xf32>
    %adsts3b1g2 = stablehlo.multiply %adlrs3b1g2, %adrats3b1g2 : tensor<256xf32>
    %adsubs3b1g2 = stablehlo.subtract %s3b1g2, %adsts3b1g2 : tensor<256xf32>
    %adwds3b1g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1g2 = stablehlo.multiply %adwds3b1g2, %adlrs3b1g2 : tensor<256xf32>
    %adwdps3b1g2 = stablehlo.multiply %adwdlrs3b1g2, %s3b1g2 : tensor<256xf32>
    %adnews3b1g2 = stablehlo.subtract %adsubs3b1g2, %adwdps3b1g2 : tensor<256xf32>
    %adb1s3b1bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b1bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b1bt2 = stablehlo.multiply %adb1s3b1bt2, %s3b1bt2m : tensor<256xf32>
    %admgs3b1bt2 = stablehlo.multiply %adob1s3b1bt2, %s3b1dn2db : tensor<256xf32>
    %admns3b1bt2 = stablehlo.add %admss3b1bt2, %admgs3b1bt2 : tensor<256xf32>
    %adb2s3b1bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b1bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b1bt2 = stablehlo.multiply %adb2s3b1bt2, %s3b1bt2v : tensor<256xf32>
    %adg2s3b1bt2 = stablehlo.multiply %s3b1dn2db, %s3b1dn2db : tensor<256xf32>
    %advgs3b1bt2 = stablehlo.multiply %adob2s3b1bt2, %adg2s3b1bt2 : tensor<256xf32>
    %advns3b1bt2 = stablehlo.add %advss3b1bt2, %advgs3b1bt2 : tensor<256xf32>
    %adbc1s3b1bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b1bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b1bt2 = stablehlo.divide %admns3b1bt2, %adbc1s3b1bt2 : tensor<256xf32>
    %advhs3b1bt2 = stablehlo.divide %advns3b1bt2, %adbc2s3b1bt2 : tensor<256xf32>
    %adlrs3b1bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b1bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b1bt2 = stablehlo.sqrt %advhs3b1bt2 : tensor<256xf32>
    %addens3b1bt2 = stablehlo.add %adsqs3b1bt2, %adepss3b1bt2 : tensor<256xf32>
    %adrats3b1bt2 = stablehlo.divide %admhs3b1bt2, %addens3b1bt2 : tensor<256xf32>
    %adsts3b1bt2 = stablehlo.multiply %adlrs3b1bt2, %adrats3b1bt2 : tensor<256xf32>
    %adsubs3b1bt2 = stablehlo.subtract %s3b1bt2, %adsts3b1bt2 : tensor<256xf32>
    %adwds3b1bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b1bt2 = stablehlo.multiply %adwds3b1bt2, %adlrs3b1bt2 : tensor<256xf32>
    %adwdps3b1bt2 = stablehlo.multiply %adwdlrs3b1bt2, %s3b1bt2 : tensor<256xf32>
    %adnews3b1bt2 = stablehlo.subtract %adsubs3b1bt2, %adwdps3b1bt2 : tensor<256xf32>
    %adb1s3b2W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b2W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b2W1 = stablehlo.multiply %adb1s3b2W1, %s3b2W1m : tensor<256x256x3x3xf32>
    %admgs3b2W1 = stablehlo.multiply %adob1s3b2W1, %s3b2dW1 : tensor<256x256x3x3xf32>
    %admns3b2W1 = stablehlo.add %admss3b2W1, %admgs3b2W1 : tensor<256x256x3x3xf32>
    %adb2s3b2W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b2W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b2W1 = stablehlo.multiply %adb2s3b2W1, %s3b2W1v : tensor<256x256x3x3xf32>
    %adg2s3b2W1 = stablehlo.multiply %s3b2dW1, %s3b2dW1 : tensor<256x256x3x3xf32>
    %advgs3b2W1 = stablehlo.multiply %adob2s3b2W1, %adg2s3b2W1 : tensor<256x256x3x3xf32>
    %advns3b2W1 = stablehlo.add %advss3b2W1, %advgs3b2W1 : tensor<256x256x3x3xf32>
    %adbc1s3b2W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b2W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b2W1 = stablehlo.divide %admns3b2W1, %adbc1s3b2W1 : tensor<256x256x3x3xf32>
    %advhs3b2W1 = stablehlo.divide %advns3b2W1, %adbc2s3b2W1 : tensor<256x256x3x3xf32>
    %adlrs3b2W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b2W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b2W1 = stablehlo.sqrt %advhs3b2W1 : tensor<256x256x3x3xf32>
    %addens3b2W1 = stablehlo.add %adsqs3b2W1, %adepss3b2W1 : tensor<256x256x3x3xf32>
    %adrats3b2W1 = stablehlo.divide %admhs3b2W1, %addens3b2W1 : tensor<256x256x3x3xf32>
    %adsts3b2W1 = stablehlo.multiply %adlrs3b2W1, %adrats3b2W1 : tensor<256x256x3x3xf32>
    %adsubs3b2W1 = stablehlo.subtract %s3b2W1, %adsts3b2W1 : tensor<256x256x3x3xf32>
    %adwds3b2W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b2W1 = stablehlo.multiply %adwds3b2W1, %adlrs3b2W1 : tensor<256x256x3x3xf32>
    %adwdps3b2W1 = stablehlo.multiply %adwdlrs3b2W1, %s3b2W1 : tensor<256x256x3x3xf32>
    %adnews3b2W1 = stablehlo.subtract %adsubs3b2W1, %adwdps3b2W1 : tensor<256x256x3x3xf32>
    %adb1s3b2b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2b1 = stablehlo.multiply %adb1s3b2b1, %s3b2b1m : tensor<256xf32>
    %admgs3b2b1 = stablehlo.multiply %adob1s3b2b1, %s3b2db1 : tensor<256xf32>
    %admns3b2b1 = stablehlo.add %admss3b2b1, %admgs3b2b1 : tensor<256xf32>
    %adb2s3b2b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2b1 = stablehlo.multiply %adb2s3b2b1, %s3b2b1v : tensor<256xf32>
    %adg2s3b2b1 = stablehlo.multiply %s3b2db1, %s3b2db1 : tensor<256xf32>
    %advgs3b2b1 = stablehlo.multiply %adob2s3b2b1, %adg2s3b2b1 : tensor<256xf32>
    %advns3b2b1 = stablehlo.add %advss3b2b1, %advgs3b2b1 : tensor<256xf32>
    %adbc1s3b2b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2b1 = stablehlo.divide %admns3b2b1, %adbc1s3b2b1 : tensor<256xf32>
    %advhs3b2b1 = stablehlo.divide %advns3b2b1, %adbc2s3b2b1 : tensor<256xf32>
    %adlrs3b2b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2b1 = stablehlo.sqrt %advhs3b2b1 : tensor<256xf32>
    %addens3b2b1 = stablehlo.add %adsqs3b2b1, %adepss3b2b1 : tensor<256xf32>
    %adrats3b2b1 = stablehlo.divide %admhs3b2b1, %addens3b2b1 : tensor<256xf32>
    %adsts3b2b1 = stablehlo.multiply %adlrs3b2b1, %adrats3b2b1 : tensor<256xf32>
    %adsubs3b2b1 = stablehlo.subtract %s3b2b1, %adsts3b2b1 : tensor<256xf32>
    %adwds3b2b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2b1 = stablehlo.multiply %adwds3b2b1, %adlrs3b2b1 : tensor<256xf32>
    %adwdps3b2b1 = stablehlo.multiply %adwdlrs3b2b1, %s3b2b1 : tensor<256xf32>
    %adnews3b2b1 = stablehlo.subtract %adsubs3b2b1, %adwdps3b2b1 : tensor<256xf32>
    %adb1s3b2g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2g1 = stablehlo.multiply %adb1s3b2g1, %s3b2g1m : tensor<256xf32>
    %admgs3b2g1 = stablehlo.multiply %adob1s3b2g1, %s3b2dn1dg : tensor<256xf32>
    %admns3b2g1 = stablehlo.add %admss3b2g1, %admgs3b2g1 : tensor<256xf32>
    %adb2s3b2g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2g1 = stablehlo.multiply %adb2s3b2g1, %s3b2g1v : tensor<256xf32>
    %adg2s3b2g1 = stablehlo.multiply %s3b2dn1dg, %s3b2dn1dg : tensor<256xf32>
    %advgs3b2g1 = stablehlo.multiply %adob2s3b2g1, %adg2s3b2g1 : tensor<256xf32>
    %advns3b2g1 = stablehlo.add %advss3b2g1, %advgs3b2g1 : tensor<256xf32>
    %adbc1s3b2g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2g1 = stablehlo.divide %admns3b2g1, %adbc1s3b2g1 : tensor<256xf32>
    %advhs3b2g1 = stablehlo.divide %advns3b2g1, %adbc2s3b2g1 : tensor<256xf32>
    %adlrs3b2g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2g1 = stablehlo.sqrt %advhs3b2g1 : tensor<256xf32>
    %addens3b2g1 = stablehlo.add %adsqs3b2g1, %adepss3b2g1 : tensor<256xf32>
    %adrats3b2g1 = stablehlo.divide %admhs3b2g1, %addens3b2g1 : tensor<256xf32>
    %adsts3b2g1 = stablehlo.multiply %adlrs3b2g1, %adrats3b2g1 : tensor<256xf32>
    %adsubs3b2g1 = stablehlo.subtract %s3b2g1, %adsts3b2g1 : tensor<256xf32>
    %adwds3b2g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2g1 = stablehlo.multiply %adwds3b2g1, %adlrs3b2g1 : tensor<256xf32>
    %adwdps3b2g1 = stablehlo.multiply %adwdlrs3b2g1, %s3b2g1 : tensor<256xf32>
    %adnews3b2g1 = stablehlo.subtract %adsubs3b2g1, %adwdps3b2g1 : tensor<256xf32>
    %adb1s3b2bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2bt1 = stablehlo.multiply %adb1s3b2bt1, %s3b2bt1m : tensor<256xf32>
    %admgs3b2bt1 = stablehlo.multiply %adob1s3b2bt1, %s3b2dn1db : tensor<256xf32>
    %admns3b2bt1 = stablehlo.add %admss3b2bt1, %admgs3b2bt1 : tensor<256xf32>
    %adb2s3b2bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2bt1 = stablehlo.multiply %adb2s3b2bt1, %s3b2bt1v : tensor<256xf32>
    %adg2s3b2bt1 = stablehlo.multiply %s3b2dn1db, %s3b2dn1db : tensor<256xf32>
    %advgs3b2bt1 = stablehlo.multiply %adob2s3b2bt1, %adg2s3b2bt1 : tensor<256xf32>
    %advns3b2bt1 = stablehlo.add %advss3b2bt1, %advgs3b2bt1 : tensor<256xf32>
    %adbc1s3b2bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2bt1 = stablehlo.divide %admns3b2bt1, %adbc1s3b2bt1 : tensor<256xf32>
    %advhs3b2bt1 = stablehlo.divide %advns3b2bt1, %adbc2s3b2bt1 : tensor<256xf32>
    %adlrs3b2bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2bt1 = stablehlo.sqrt %advhs3b2bt1 : tensor<256xf32>
    %addens3b2bt1 = stablehlo.add %adsqs3b2bt1, %adepss3b2bt1 : tensor<256xf32>
    %adrats3b2bt1 = stablehlo.divide %admhs3b2bt1, %addens3b2bt1 : tensor<256xf32>
    %adsts3b2bt1 = stablehlo.multiply %adlrs3b2bt1, %adrats3b2bt1 : tensor<256xf32>
    %adsubs3b2bt1 = stablehlo.subtract %s3b2bt1, %adsts3b2bt1 : tensor<256xf32>
    %adwds3b2bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2bt1 = stablehlo.multiply %adwds3b2bt1, %adlrs3b2bt1 : tensor<256xf32>
    %adwdps3b2bt1 = stablehlo.multiply %adwdlrs3b2bt1, %s3b2bt1 : tensor<256xf32>
    %adnews3b2bt1 = stablehlo.subtract %adsubs3b2bt1, %adwdps3b2bt1 : tensor<256xf32>
    %adb1s3b2W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b2W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b2W2 = stablehlo.multiply %adb1s3b2W2, %s3b2W2m : tensor<256x256x3x3xf32>
    %admgs3b2W2 = stablehlo.multiply %adob1s3b2W2, %s3b2dW2 : tensor<256x256x3x3xf32>
    %admns3b2W2 = stablehlo.add %admss3b2W2, %admgs3b2W2 : tensor<256x256x3x3xf32>
    %adb2s3b2W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b2W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b2W2 = stablehlo.multiply %adb2s3b2W2, %s3b2W2v : tensor<256x256x3x3xf32>
    %adg2s3b2W2 = stablehlo.multiply %s3b2dW2, %s3b2dW2 : tensor<256x256x3x3xf32>
    %advgs3b2W2 = stablehlo.multiply %adob2s3b2W2, %adg2s3b2W2 : tensor<256x256x3x3xf32>
    %advns3b2W2 = stablehlo.add %advss3b2W2, %advgs3b2W2 : tensor<256x256x3x3xf32>
    %adbc1s3b2W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b2W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b2W2 = stablehlo.divide %admns3b2W2, %adbc1s3b2W2 : tensor<256x256x3x3xf32>
    %advhs3b2W2 = stablehlo.divide %advns3b2W2, %adbc2s3b2W2 : tensor<256x256x3x3xf32>
    %adlrs3b2W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b2W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b2W2 = stablehlo.sqrt %advhs3b2W2 : tensor<256x256x3x3xf32>
    %addens3b2W2 = stablehlo.add %adsqs3b2W2, %adepss3b2W2 : tensor<256x256x3x3xf32>
    %adrats3b2W2 = stablehlo.divide %admhs3b2W2, %addens3b2W2 : tensor<256x256x3x3xf32>
    %adsts3b2W2 = stablehlo.multiply %adlrs3b2W2, %adrats3b2W2 : tensor<256x256x3x3xf32>
    %adsubs3b2W2 = stablehlo.subtract %s3b2W2, %adsts3b2W2 : tensor<256x256x3x3xf32>
    %adwds3b2W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b2W2 = stablehlo.multiply %adwds3b2W2, %adlrs3b2W2 : tensor<256x256x3x3xf32>
    %adwdps3b2W2 = stablehlo.multiply %adwdlrs3b2W2, %s3b2W2 : tensor<256x256x3x3xf32>
    %adnews3b2W2 = stablehlo.subtract %adsubs3b2W2, %adwdps3b2W2 : tensor<256x256x3x3xf32>
    %adb1s3b2b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2b2 = stablehlo.multiply %adb1s3b2b2, %s3b2b2m : tensor<256xf32>
    %admgs3b2b2 = stablehlo.multiply %adob1s3b2b2, %s3b2db2 : tensor<256xf32>
    %admns3b2b2 = stablehlo.add %admss3b2b2, %admgs3b2b2 : tensor<256xf32>
    %adb2s3b2b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2b2 = stablehlo.multiply %adb2s3b2b2, %s3b2b2v : tensor<256xf32>
    %adg2s3b2b2 = stablehlo.multiply %s3b2db2, %s3b2db2 : tensor<256xf32>
    %advgs3b2b2 = stablehlo.multiply %adob2s3b2b2, %adg2s3b2b2 : tensor<256xf32>
    %advns3b2b2 = stablehlo.add %advss3b2b2, %advgs3b2b2 : tensor<256xf32>
    %adbc1s3b2b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2b2 = stablehlo.divide %admns3b2b2, %adbc1s3b2b2 : tensor<256xf32>
    %advhs3b2b2 = stablehlo.divide %advns3b2b2, %adbc2s3b2b2 : tensor<256xf32>
    %adlrs3b2b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2b2 = stablehlo.sqrt %advhs3b2b2 : tensor<256xf32>
    %addens3b2b2 = stablehlo.add %adsqs3b2b2, %adepss3b2b2 : tensor<256xf32>
    %adrats3b2b2 = stablehlo.divide %admhs3b2b2, %addens3b2b2 : tensor<256xf32>
    %adsts3b2b2 = stablehlo.multiply %adlrs3b2b2, %adrats3b2b2 : tensor<256xf32>
    %adsubs3b2b2 = stablehlo.subtract %s3b2b2, %adsts3b2b2 : tensor<256xf32>
    %adwds3b2b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2b2 = stablehlo.multiply %adwds3b2b2, %adlrs3b2b2 : tensor<256xf32>
    %adwdps3b2b2 = stablehlo.multiply %adwdlrs3b2b2, %s3b2b2 : tensor<256xf32>
    %adnews3b2b2 = stablehlo.subtract %adsubs3b2b2, %adwdps3b2b2 : tensor<256xf32>
    %adb1s3b2g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2g2 = stablehlo.multiply %adb1s3b2g2, %s3b2g2m : tensor<256xf32>
    %admgs3b2g2 = stablehlo.multiply %adob1s3b2g2, %s3b2dn2dg : tensor<256xf32>
    %admns3b2g2 = stablehlo.add %admss3b2g2, %admgs3b2g2 : tensor<256xf32>
    %adb2s3b2g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2g2 = stablehlo.multiply %adb2s3b2g2, %s3b2g2v : tensor<256xf32>
    %adg2s3b2g2 = stablehlo.multiply %s3b2dn2dg, %s3b2dn2dg : tensor<256xf32>
    %advgs3b2g2 = stablehlo.multiply %adob2s3b2g2, %adg2s3b2g2 : tensor<256xf32>
    %advns3b2g2 = stablehlo.add %advss3b2g2, %advgs3b2g2 : tensor<256xf32>
    %adbc1s3b2g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2g2 = stablehlo.divide %admns3b2g2, %adbc1s3b2g2 : tensor<256xf32>
    %advhs3b2g2 = stablehlo.divide %advns3b2g2, %adbc2s3b2g2 : tensor<256xf32>
    %adlrs3b2g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2g2 = stablehlo.sqrt %advhs3b2g2 : tensor<256xf32>
    %addens3b2g2 = stablehlo.add %adsqs3b2g2, %adepss3b2g2 : tensor<256xf32>
    %adrats3b2g2 = stablehlo.divide %admhs3b2g2, %addens3b2g2 : tensor<256xf32>
    %adsts3b2g2 = stablehlo.multiply %adlrs3b2g2, %adrats3b2g2 : tensor<256xf32>
    %adsubs3b2g2 = stablehlo.subtract %s3b2g2, %adsts3b2g2 : tensor<256xf32>
    %adwds3b2g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2g2 = stablehlo.multiply %adwds3b2g2, %adlrs3b2g2 : tensor<256xf32>
    %adwdps3b2g2 = stablehlo.multiply %adwdlrs3b2g2, %s3b2g2 : tensor<256xf32>
    %adnews3b2g2 = stablehlo.subtract %adsubs3b2g2, %adwdps3b2g2 : tensor<256xf32>
    %adb1s3b2bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b2bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b2bt2 = stablehlo.multiply %adb1s3b2bt2, %s3b2bt2m : tensor<256xf32>
    %admgs3b2bt2 = stablehlo.multiply %adob1s3b2bt2, %s3b2dn2db : tensor<256xf32>
    %admns3b2bt2 = stablehlo.add %admss3b2bt2, %admgs3b2bt2 : tensor<256xf32>
    %adb2s3b2bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b2bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b2bt2 = stablehlo.multiply %adb2s3b2bt2, %s3b2bt2v : tensor<256xf32>
    %adg2s3b2bt2 = stablehlo.multiply %s3b2dn2db, %s3b2dn2db : tensor<256xf32>
    %advgs3b2bt2 = stablehlo.multiply %adob2s3b2bt2, %adg2s3b2bt2 : tensor<256xf32>
    %advns3b2bt2 = stablehlo.add %advss3b2bt2, %advgs3b2bt2 : tensor<256xf32>
    %adbc1s3b2bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b2bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b2bt2 = stablehlo.divide %admns3b2bt2, %adbc1s3b2bt2 : tensor<256xf32>
    %advhs3b2bt2 = stablehlo.divide %advns3b2bt2, %adbc2s3b2bt2 : tensor<256xf32>
    %adlrs3b2bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b2bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b2bt2 = stablehlo.sqrt %advhs3b2bt2 : tensor<256xf32>
    %addens3b2bt2 = stablehlo.add %adsqs3b2bt2, %adepss3b2bt2 : tensor<256xf32>
    %adrats3b2bt2 = stablehlo.divide %admhs3b2bt2, %addens3b2bt2 : tensor<256xf32>
    %adsts3b2bt2 = stablehlo.multiply %adlrs3b2bt2, %adrats3b2bt2 : tensor<256xf32>
    %adsubs3b2bt2 = stablehlo.subtract %s3b2bt2, %adsts3b2bt2 : tensor<256xf32>
    %adwds3b2bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b2bt2 = stablehlo.multiply %adwds3b2bt2, %adlrs3b2bt2 : tensor<256xf32>
    %adwdps3b2bt2 = stablehlo.multiply %adwdlrs3b2bt2, %s3b2bt2 : tensor<256xf32>
    %adnews3b2bt2 = stablehlo.subtract %adsubs3b2bt2, %adwdps3b2bt2 : tensor<256xf32>
    %adb1s3b3W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b3W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b3W1 = stablehlo.multiply %adb1s3b3W1, %s3b3W1m : tensor<256x256x3x3xf32>
    %admgs3b3W1 = stablehlo.multiply %adob1s3b3W1, %s3b3dW1 : tensor<256x256x3x3xf32>
    %admns3b3W1 = stablehlo.add %admss3b3W1, %admgs3b3W1 : tensor<256x256x3x3xf32>
    %adb2s3b3W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b3W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b3W1 = stablehlo.multiply %adb2s3b3W1, %s3b3W1v : tensor<256x256x3x3xf32>
    %adg2s3b3W1 = stablehlo.multiply %s3b3dW1, %s3b3dW1 : tensor<256x256x3x3xf32>
    %advgs3b3W1 = stablehlo.multiply %adob2s3b3W1, %adg2s3b3W1 : tensor<256x256x3x3xf32>
    %advns3b3W1 = stablehlo.add %advss3b3W1, %advgs3b3W1 : tensor<256x256x3x3xf32>
    %adbc1s3b3W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b3W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b3W1 = stablehlo.divide %admns3b3W1, %adbc1s3b3W1 : tensor<256x256x3x3xf32>
    %advhs3b3W1 = stablehlo.divide %advns3b3W1, %adbc2s3b3W1 : tensor<256x256x3x3xf32>
    %adlrs3b3W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b3W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b3W1 = stablehlo.sqrt %advhs3b3W1 : tensor<256x256x3x3xf32>
    %addens3b3W1 = stablehlo.add %adsqs3b3W1, %adepss3b3W1 : tensor<256x256x3x3xf32>
    %adrats3b3W1 = stablehlo.divide %admhs3b3W1, %addens3b3W1 : tensor<256x256x3x3xf32>
    %adsts3b3W1 = stablehlo.multiply %adlrs3b3W1, %adrats3b3W1 : tensor<256x256x3x3xf32>
    %adsubs3b3W1 = stablehlo.subtract %s3b3W1, %adsts3b3W1 : tensor<256x256x3x3xf32>
    %adwds3b3W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b3W1 = stablehlo.multiply %adwds3b3W1, %adlrs3b3W1 : tensor<256x256x3x3xf32>
    %adwdps3b3W1 = stablehlo.multiply %adwdlrs3b3W1, %s3b3W1 : tensor<256x256x3x3xf32>
    %adnews3b3W1 = stablehlo.subtract %adsubs3b3W1, %adwdps3b3W1 : tensor<256x256x3x3xf32>
    %adb1s3b3b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3b1 = stablehlo.multiply %adb1s3b3b1, %s3b3b1m : tensor<256xf32>
    %admgs3b3b1 = stablehlo.multiply %adob1s3b3b1, %s3b3db1 : tensor<256xf32>
    %admns3b3b1 = stablehlo.add %admss3b3b1, %admgs3b3b1 : tensor<256xf32>
    %adb2s3b3b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3b1 = stablehlo.multiply %adb2s3b3b1, %s3b3b1v : tensor<256xf32>
    %adg2s3b3b1 = stablehlo.multiply %s3b3db1, %s3b3db1 : tensor<256xf32>
    %advgs3b3b1 = stablehlo.multiply %adob2s3b3b1, %adg2s3b3b1 : tensor<256xf32>
    %advns3b3b1 = stablehlo.add %advss3b3b1, %advgs3b3b1 : tensor<256xf32>
    %adbc1s3b3b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3b1 = stablehlo.divide %admns3b3b1, %adbc1s3b3b1 : tensor<256xf32>
    %advhs3b3b1 = stablehlo.divide %advns3b3b1, %adbc2s3b3b1 : tensor<256xf32>
    %adlrs3b3b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3b1 = stablehlo.sqrt %advhs3b3b1 : tensor<256xf32>
    %addens3b3b1 = stablehlo.add %adsqs3b3b1, %adepss3b3b1 : tensor<256xf32>
    %adrats3b3b1 = stablehlo.divide %admhs3b3b1, %addens3b3b1 : tensor<256xf32>
    %adsts3b3b1 = stablehlo.multiply %adlrs3b3b1, %adrats3b3b1 : tensor<256xf32>
    %adsubs3b3b1 = stablehlo.subtract %s3b3b1, %adsts3b3b1 : tensor<256xf32>
    %adwds3b3b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3b1 = stablehlo.multiply %adwds3b3b1, %adlrs3b3b1 : tensor<256xf32>
    %adwdps3b3b1 = stablehlo.multiply %adwdlrs3b3b1, %s3b3b1 : tensor<256xf32>
    %adnews3b3b1 = stablehlo.subtract %adsubs3b3b1, %adwdps3b3b1 : tensor<256xf32>
    %adb1s3b3g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3g1 = stablehlo.multiply %adb1s3b3g1, %s3b3g1m : tensor<256xf32>
    %admgs3b3g1 = stablehlo.multiply %adob1s3b3g1, %s3b3dn1dg : tensor<256xf32>
    %admns3b3g1 = stablehlo.add %admss3b3g1, %admgs3b3g1 : tensor<256xf32>
    %adb2s3b3g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3g1 = stablehlo.multiply %adb2s3b3g1, %s3b3g1v : tensor<256xf32>
    %adg2s3b3g1 = stablehlo.multiply %s3b3dn1dg, %s3b3dn1dg : tensor<256xf32>
    %advgs3b3g1 = stablehlo.multiply %adob2s3b3g1, %adg2s3b3g1 : tensor<256xf32>
    %advns3b3g1 = stablehlo.add %advss3b3g1, %advgs3b3g1 : tensor<256xf32>
    %adbc1s3b3g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3g1 = stablehlo.divide %admns3b3g1, %adbc1s3b3g1 : tensor<256xf32>
    %advhs3b3g1 = stablehlo.divide %advns3b3g1, %adbc2s3b3g1 : tensor<256xf32>
    %adlrs3b3g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3g1 = stablehlo.sqrt %advhs3b3g1 : tensor<256xf32>
    %addens3b3g1 = stablehlo.add %adsqs3b3g1, %adepss3b3g1 : tensor<256xf32>
    %adrats3b3g1 = stablehlo.divide %admhs3b3g1, %addens3b3g1 : tensor<256xf32>
    %adsts3b3g1 = stablehlo.multiply %adlrs3b3g1, %adrats3b3g1 : tensor<256xf32>
    %adsubs3b3g1 = stablehlo.subtract %s3b3g1, %adsts3b3g1 : tensor<256xf32>
    %adwds3b3g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3g1 = stablehlo.multiply %adwds3b3g1, %adlrs3b3g1 : tensor<256xf32>
    %adwdps3b3g1 = stablehlo.multiply %adwdlrs3b3g1, %s3b3g1 : tensor<256xf32>
    %adnews3b3g1 = stablehlo.subtract %adsubs3b3g1, %adwdps3b3g1 : tensor<256xf32>
    %adb1s3b3bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3bt1 = stablehlo.multiply %adb1s3b3bt1, %s3b3bt1m : tensor<256xf32>
    %admgs3b3bt1 = stablehlo.multiply %adob1s3b3bt1, %s3b3dn1db : tensor<256xf32>
    %admns3b3bt1 = stablehlo.add %admss3b3bt1, %admgs3b3bt1 : tensor<256xf32>
    %adb2s3b3bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3bt1 = stablehlo.multiply %adb2s3b3bt1, %s3b3bt1v : tensor<256xf32>
    %adg2s3b3bt1 = stablehlo.multiply %s3b3dn1db, %s3b3dn1db : tensor<256xf32>
    %advgs3b3bt1 = stablehlo.multiply %adob2s3b3bt1, %adg2s3b3bt1 : tensor<256xf32>
    %advns3b3bt1 = stablehlo.add %advss3b3bt1, %advgs3b3bt1 : tensor<256xf32>
    %adbc1s3b3bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3bt1 = stablehlo.divide %admns3b3bt1, %adbc1s3b3bt1 : tensor<256xf32>
    %advhs3b3bt1 = stablehlo.divide %advns3b3bt1, %adbc2s3b3bt1 : tensor<256xf32>
    %adlrs3b3bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3bt1 = stablehlo.sqrt %advhs3b3bt1 : tensor<256xf32>
    %addens3b3bt1 = stablehlo.add %adsqs3b3bt1, %adepss3b3bt1 : tensor<256xf32>
    %adrats3b3bt1 = stablehlo.divide %admhs3b3bt1, %addens3b3bt1 : tensor<256xf32>
    %adsts3b3bt1 = stablehlo.multiply %adlrs3b3bt1, %adrats3b3bt1 : tensor<256xf32>
    %adsubs3b3bt1 = stablehlo.subtract %s3b3bt1, %adsts3b3bt1 : tensor<256xf32>
    %adwds3b3bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3bt1 = stablehlo.multiply %adwds3b3bt1, %adlrs3b3bt1 : tensor<256xf32>
    %adwdps3b3bt1 = stablehlo.multiply %adwdlrs3b3bt1, %s3b3bt1 : tensor<256xf32>
    %adnews3b3bt1 = stablehlo.subtract %adsubs3b3bt1, %adwdps3b3bt1 : tensor<256xf32>
    %adb1s3b3W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b3W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b3W2 = stablehlo.multiply %adb1s3b3W2, %s3b3W2m : tensor<256x256x3x3xf32>
    %admgs3b3W2 = stablehlo.multiply %adob1s3b3W2, %s3b3dW2 : tensor<256x256x3x3xf32>
    %admns3b3W2 = stablehlo.add %admss3b3W2, %admgs3b3W2 : tensor<256x256x3x3xf32>
    %adb2s3b3W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b3W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b3W2 = stablehlo.multiply %adb2s3b3W2, %s3b3W2v : tensor<256x256x3x3xf32>
    %adg2s3b3W2 = stablehlo.multiply %s3b3dW2, %s3b3dW2 : tensor<256x256x3x3xf32>
    %advgs3b3W2 = stablehlo.multiply %adob2s3b3W2, %adg2s3b3W2 : tensor<256x256x3x3xf32>
    %advns3b3W2 = stablehlo.add %advss3b3W2, %advgs3b3W2 : tensor<256x256x3x3xf32>
    %adbc1s3b3W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b3W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b3W2 = stablehlo.divide %admns3b3W2, %adbc1s3b3W2 : tensor<256x256x3x3xf32>
    %advhs3b3W2 = stablehlo.divide %advns3b3W2, %adbc2s3b3W2 : tensor<256x256x3x3xf32>
    %adlrs3b3W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b3W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b3W2 = stablehlo.sqrt %advhs3b3W2 : tensor<256x256x3x3xf32>
    %addens3b3W2 = stablehlo.add %adsqs3b3W2, %adepss3b3W2 : tensor<256x256x3x3xf32>
    %adrats3b3W2 = stablehlo.divide %admhs3b3W2, %addens3b3W2 : tensor<256x256x3x3xf32>
    %adsts3b3W2 = stablehlo.multiply %adlrs3b3W2, %adrats3b3W2 : tensor<256x256x3x3xf32>
    %adsubs3b3W2 = stablehlo.subtract %s3b3W2, %adsts3b3W2 : tensor<256x256x3x3xf32>
    %adwds3b3W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b3W2 = stablehlo.multiply %adwds3b3W2, %adlrs3b3W2 : tensor<256x256x3x3xf32>
    %adwdps3b3W2 = stablehlo.multiply %adwdlrs3b3W2, %s3b3W2 : tensor<256x256x3x3xf32>
    %adnews3b3W2 = stablehlo.subtract %adsubs3b3W2, %adwdps3b3W2 : tensor<256x256x3x3xf32>
    %adb1s3b3b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3b2 = stablehlo.multiply %adb1s3b3b2, %s3b3b2m : tensor<256xf32>
    %admgs3b3b2 = stablehlo.multiply %adob1s3b3b2, %s3b3db2 : tensor<256xf32>
    %admns3b3b2 = stablehlo.add %admss3b3b2, %admgs3b3b2 : tensor<256xf32>
    %adb2s3b3b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3b2 = stablehlo.multiply %adb2s3b3b2, %s3b3b2v : tensor<256xf32>
    %adg2s3b3b2 = stablehlo.multiply %s3b3db2, %s3b3db2 : tensor<256xf32>
    %advgs3b3b2 = stablehlo.multiply %adob2s3b3b2, %adg2s3b3b2 : tensor<256xf32>
    %advns3b3b2 = stablehlo.add %advss3b3b2, %advgs3b3b2 : tensor<256xf32>
    %adbc1s3b3b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3b2 = stablehlo.divide %admns3b3b2, %adbc1s3b3b2 : tensor<256xf32>
    %advhs3b3b2 = stablehlo.divide %advns3b3b2, %adbc2s3b3b2 : tensor<256xf32>
    %adlrs3b3b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3b2 = stablehlo.sqrt %advhs3b3b2 : tensor<256xf32>
    %addens3b3b2 = stablehlo.add %adsqs3b3b2, %adepss3b3b2 : tensor<256xf32>
    %adrats3b3b2 = stablehlo.divide %admhs3b3b2, %addens3b3b2 : tensor<256xf32>
    %adsts3b3b2 = stablehlo.multiply %adlrs3b3b2, %adrats3b3b2 : tensor<256xf32>
    %adsubs3b3b2 = stablehlo.subtract %s3b3b2, %adsts3b3b2 : tensor<256xf32>
    %adwds3b3b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3b2 = stablehlo.multiply %adwds3b3b2, %adlrs3b3b2 : tensor<256xf32>
    %adwdps3b3b2 = stablehlo.multiply %adwdlrs3b3b2, %s3b3b2 : tensor<256xf32>
    %adnews3b3b2 = stablehlo.subtract %adsubs3b3b2, %adwdps3b3b2 : tensor<256xf32>
    %adb1s3b3g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3g2 = stablehlo.multiply %adb1s3b3g2, %s3b3g2m : tensor<256xf32>
    %admgs3b3g2 = stablehlo.multiply %adob1s3b3g2, %s3b3dn2dg : tensor<256xf32>
    %admns3b3g2 = stablehlo.add %admss3b3g2, %admgs3b3g2 : tensor<256xf32>
    %adb2s3b3g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3g2 = stablehlo.multiply %adb2s3b3g2, %s3b3g2v : tensor<256xf32>
    %adg2s3b3g2 = stablehlo.multiply %s3b3dn2dg, %s3b3dn2dg : tensor<256xf32>
    %advgs3b3g2 = stablehlo.multiply %adob2s3b3g2, %adg2s3b3g2 : tensor<256xf32>
    %advns3b3g2 = stablehlo.add %advss3b3g2, %advgs3b3g2 : tensor<256xf32>
    %adbc1s3b3g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3g2 = stablehlo.divide %admns3b3g2, %adbc1s3b3g2 : tensor<256xf32>
    %advhs3b3g2 = stablehlo.divide %advns3b3g2, %adbc2s3b3g2 : tensor<256xf32>
    %adlrs3b3g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3g2 = stablehlo.sqrt %advhs3b3g2 : tensor<256xf32>
    %addens3b3g2 = stablehlo.add %adsqs3b3g2, %adepss3b3g2 : tensor<256xf32>
    %adrats3b3g2 = stablehlo.divide %admhs3b3g2, %addens3b3g2 : tensor<256xf32>
    %adsts3b3g2 = stablehlo.multiply %adlrs3b3g2, %adrats3b3g2 : tensor<256xf32>
    %adsubs3b3g2 = stablehlo.subtract %s3b3g2, %adsts3b3g2 : tensor<256xf32>
    %adwds3b3g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3g2 = stablehlo.multiply %adwds3b3g2, %adlrs3b3g2 : tensor<256xf32>
    %adwdps3b3g2 = stablehlo.multiply %adwdlrs3b3g2, %s3b3g2 : tensor<256xf32>
    %adnews3b3g2 = stablehlo.subtract %adsubs3b3g2, %adwdps3b3g2 : tensor<256xf32>
    %adb1s3b3bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b3bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b3bt2 = stablehlo.multiply %adb1s3b3bt2, %s3b3bt2m : tensor<256xf32>
    %admgs3b3bt2 = stablehlo.multiply %adob1s3b3bt2, %s3b3dn2db : tensor<256xf32>
    %admns3b3bt2 = stablehlo.add %admss3b3bt2, %admgs3b3bt2 : tensor<256xf32>
    %adb2s3b3bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b3bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b3bt2 = stablehlo.multiply %adb2s3b3bt2, %s3b3bt2v : tensor<256xf32>
    %adg2s3b3bt2 = stablehlo.multiply %s3b3dn2db, %s3b3dn2db : tensor<256xf32>
    %advgs3b3bt2 = stablehlo.multiply %adob2s3b3bt2, %adg2s3b3bt2 : tensor<256xf32>
    %advns3b3bt2 = stablehlo.add %advss3b3bt2, %advgs3b3bt2 : tensor<256xf32>
    %adbc1s3b3bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b3bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b3bt2 = stablehlo.divide %admns3b3bt2, %adbc1s3b3bt2 : tensor<256xf32>
    %advhs3b3bt2 = stablehlo.divide %advns3b3bt2, %adbc2s3b3bt2 : tensor<256xf32>
    %adlrs3b3bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b3bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b3bt2 = stablehlo.sqrt %advhs3b3bt2 : tensor<256xf32>
    %addens3b3bt2 = stablehlo.add %adsqs3b3bt2, %adepss3b3bt2 : tensor<256xf32>
    %adrats3b3bt2 = stablehlo.divide %admhs3b3bt2, %addens3b3bt2 : tensor<256xf32>
    %adsts3b3bt2 = stablehlo.multiply %adlrs3b3bt2, %adrats3b3bt2 : tensor<256xf32>
    %adsubs3b3bt2 = stablehlo.subtract %s3b3bt2, %adsts3b3bt2 : tensor<256xf32>
    %adwds3b3bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b3bt2 = stablehlo.multiply %adwds3b3bt2, %adlrs3b3bt2 : tensor<256xf32>
    %adwdps3b3bt2 = stablehlo.multiply %adwdlrs3b3bt2, %s3b3bt2 : tensor<256xf32>
    %adnews3b3bt2 = stablehlo.subtract %adsubs3b3bt2, %adwdps3b3bt2 : tensor<256xf32>
    %adb1s3b4W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b4W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b4W1 = stablehlo.multiply %adb1s3b4W1, %s3b4W1m : tensor<256x256x3x3xf32>
    %admgs3b4W1 = stablehlo.multiply %adob1s3b4W1, %s3b4dW1 : tensor<256x256x3x3xf32>
    %admns3b4W1 = stablehlo.add %admss3b4W1, %admgs3b4W1 : tensor<256x256x3x3xf32>
    %adb2s3b4W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b4W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b4W1 = stablehlo.multiply %adb2s3b4W1, %s3b4W1v : tensor<256x256x3x3xf32>
    %adg2s3b4W1 = stablehlo.multiply %s3b4dW1, %s3b4dW1 : tensor<256x256x3x3xf32>
    %advgs3b4W1 = stablehlo.multiply %adob2s3b4W1, %adg2s3b4W1 : tensor<256x256x3x3xf32>
    %advns3b4W1 = stablehlo.add %advss3b4W1, %advgs3b4W1 : tensor<256x256x3x3xf32>
    %adbc1s3b4W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b4W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b4W1 = stablehlo.divide %admns3b4W1, %adbc1s3b4W1 : tensor<256x256x3x3xf32>
    %advhs3b4W1 = stablehlo.divide %advns3b4W1, %adbc2s3b4W1 : tensor<256x256x3x3xf32>
    %adlrs3b4W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b4W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b4W1 = stablehlo.sqrt %advhs3b4W1 : tensor<256x256x3x3xf32>
    %addens3b4W1 = stablehlo.add %adsqs3b4W1, %adepss3b4W1 : tensor<256x256x3x3xf32>
    %adrats3b4W1 = stablehlo.divide %admhs3b4W1, %addens3b4W1 : tensor<256x256x3x3xf32>
    %adsts3b4W1 = stablehlo.multiply %adlrs3b4W1, %adrats3b4W1 : tensor<256x256x3x3xf32>
    %adsubs3b4W1 = stablehlo.subtract %s3b4W1, %adsts3b4W1 : tensor<256x256x3x3xf32>
    %adwds3b4W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b4W1 = stablehlo.multiply %adwds3b4W1, %adlrs3b4W1 : tensor<256x256x3x3xf32>
    %adwdps3b4W1 = stablehlo.multiply %adwdlrs3b4W1, %s3b4W1 : tensor<256x256x3x3xf32>
    %adnews3b4W1 = stablehlo.subtract %adsubs3b4W1, %adwdps3b4W1 : tensor<256x256x3x3xf32>
    %adb1s3b4b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4b1 = stablehlo.multiply %adb1s3b4b1, %s3b4b1m : tensor<256xf32>
    %admgs3b4b1 = stablehlo.multiply %adob1s3b4b1, %s3b4db1 : tensor<256xf32>
    %admns3b4b1 = stablehlo.add %admss3b4b1, %admgs3b4b1 : tensor<256xf32>
    %adb2s3b4b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4b1 = stablehlo.multiply %adb2s3b4b1, %s3b4b1v : tensor<256xf32>
    %adg2s3b4b1 = stablehlo.multiply %s3b4db1, %s3b4db1 : tensor<256xf32>
    %advgs3b4b1 = stablehlo.multiply %adob2s3b4b1, %adg2s3b4b1 : tensor<256xf32>
    %advns3b4b1 = stablehlo.add %advss3b4b1, %advgs3b4b1 : tensor<256xf32>
    %adbc1s3b4b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4b1 = stablehlo.divide %admns3b4b1, %adbc1s3b4b1 : tensor<256xf32>
    %advhs3b4b1 = stablehlo.divide %advns3b4b1, %adbc2s3b4b1 : tensor<256xf32>
    %adlrs3b4b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4b1 = stablehlo.sqrt %advhs3b4b1 : tensor<256xf32>
    %addens3b4b1 = stablehlo.add %adsqs3b4b1, %adepss3b4b1 : tensor<256xf32>
    %adrats3b4b1 = stablehlo.divide %admhs3b4b1, %addens3b4b1 : tensor<256xf32>
    %adsts3b4b1 = stablehlo.multiply %adlrs3b4b1, %adrats3b4b1 : tensor<256xf32>
    %adsubs3b4b1 = stablehlo.subtract %s3b4b1, %adsts3b4b1 : tensor<256xf32>
    %adwds3b4b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4b1 = stablehlo.multiply %adwds3b4b1, %adlrs3b4b1 : tensor<256xf32>
    %adwdps3b4b1 = stablehlo.multiply %adwdlrs3b4b1, %s3b4b1 : tensor<256xf32>
    %adnews3b4b1 = stablehlo.subtract %adsubs3b4b1, %adwdps3b4b1 : tensor<256xf32>
    %adb1s3b4g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4g1 = stablehlo.multiply %adb1s3b4g1, %s3b4g1m : tensor<256xf32>
    %admgs3b4g1 = stablehlo.multiply %adob1s3b4g1, %s3b4dn1dg : tensor<256xf32>
    %admns3b4g1 = stablehlo.add %admss3b4g1, %admgs3b4g1 : tensor<256xf32>
    %adb2s3b4g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4g1 = stablehlo.multiply %adb2s3b4g1, %s3b4g1v : tensor<256xf32>
    %adg2s3b4g1 = stablehlo.multiply %s3b4dn1dg, %s3b4dn1dg : tensor<256xf32>
    %advgs3b4g1 = stablehlo.multiply %adob2s3b4g1, %adg2s3b4g1 : tensor<256xf32>
    %advns3b4g1 = stablehlo.add %advss3b4g1, %advgs3b4g1 : tensor<256xf32>
    %adbc1s3b4g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4g1 = stablehlo.divide %admns3b4g1, %adbc1s3b4g1 : tensor<256xf32>
    %advhs3b4g1 = stablehlo.divide %advns3b4g1, %adbc2s3b4g1 : tensor<256xf32>
    %adlrs3b4g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4g1 = stablehlo.sqrt %advhs3b4g1 : tensor<256xf32>
    %addens3b4g1 = stablehlo.add %adsqs3b4g1, %adepss3b4g1 : tensor<256xf32>
    %adrats3b4g1 = stablehlo.divide %admhs3b4g1, %addens3b4g1 : tensor<256xf32>
    %adsts3b4g1 = stablehlo.multiply %adlrs3b4g1, %adrats3b4g1 : tensor<256xf32>
    %adsubs3b4g1 = stablehlo.subtract %s3b4g1, %adsts3b4g1 : tensor<256xf32>
    %adwds3b4g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4g1 = stablehlo.multiply %adwds3b4g1, %adlrs3b4g1 : tensor<256xf32>
    %adwdps3b4g1 = stablehlo.multiply %adwdlrs3b4g1, %s3b4g1 : tensor<256xf32>
    %adnews3b4g1 = stablehlo.subtract %adsubs3b4g1, %adwdps3b4g1 : tensor<256xf32>
    %adb1s3b4bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4bt1 = stablehlo.multiply %adb1s3b4bt1, %s3b4bt1m : tensor<256xf32>
    %admgs3b4bt1 = stablehlo.multiply %adob1s3b4bt1, %s3b4dn1db : tensor<256xf32>
    %admns3b4bt1 = stablehlo.add %admss3b4bt1, %admgs3b4bt1 : tensor<256xf32>
    %adb2s3b4bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4bt1 = stablehlo.multiply %adb2s3b4bt1, %s3b4bt1v : tensor<256xf32>
    %adg2s3b4bt1 = stablehlo.multiply %s3b4dn1db, %s3b4dn1db : tensor<256xf32>
    %advgs3b4bt1 = stablehlo.multiply %adob2s3b4bt1, %adg2s3b4bt1 : tensor<256xf32>
    %advns3b4bt1 = stablehlo.add %advss3b4bt1, %advgs3b4bt1 : tensor<256xf32>
    %adbc1s3b4bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4bt1 = stablehlo.divide %admns3b4bt1, %adbc1s3b4bt1 : tensor<256xf32>
    %advhs3b4bt1 = stablehlo.divide %advns3b4bt1, %adbc2s3b4bt1 : tensor<256xf32>
    %adlrs3b4bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4bt1 = stablehlo.sqrt %advhs3b4bt1 : tensor<256xf32>
    %addens3b4bt1 = stablehlo.add %adsqs3b4bt1, %adepss3b4bt1 : tensor<256xf32>
    %adrats3b4bt1 = stablehlo.divide %admhs3b4bt1, %addens3b4bt1 : tensor<256xf32>
    %adsts3b4bt1 = stablehlo.multiply %adlrs3b4bt1, %adrats3b4bt1 : tensor<256xf32>
    %adsubs3b4bt1 = stablehlo.subtract %s3b4bt1, %adsts3b4bt1 : tensor<256xf32>
    %adwds3b4bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4bt1 = stablehlo.multiply %adwds3b4bt1, %adlrs3b4bt1 : tensor<256xf32>
    %adwdps3b4bt1 = stablehlo.multiply %adwdlrs3b4bt1, %s3b4bt1 : tensor<256xf32>
    %adnews3b4bt1 = stablehlo.subtract %adsubs3b4bt1, %adwdps3b4bt1 : tensor<256xf32>
    %adb1s3b4W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob1s3b4W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admss3b4W2 = stablehlo.multiply %adb1s3b4W2, %s3b4W2m : tensor<256x256x3x3xf32>
    %admgs3b4W2 = stablehlo.multiply %adob1s3b4W2, %s3b4dW2 : tensor<256x256x3x3xf32>
    %admns3b4W2 = stablehlo.add %admss3b4W2, %admgs3b4W2 : tensor<256x256x3x3xf32>
    %adb2s3b4W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adob2s3b4W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %advss3b4W2 = stablehlo.multiply %adb2s3b4W2, %s3b4W2v : tensor<256x256x3x3xf32>
    %adg2s3b4W2 = stablehlo.multiply %s3b4dW2, %s3b4dW2 : tensor<256x256x3x3xf32>
    %advgs3b4W2 = stablehlo.multiply %adob2s3b4W2, %adg2s3b4W2 : tensor<256x256x3x3xf32>
    %advns3b4W2 = stablehlo.add %advss3b4W2, %advgs3b4W2 : tensor<256x256x3x3xf32>
    %adbc1s3b4W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adbc2s3b4W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %admhs3b4W2 = stablehlo.divide %admns3b4W2, %adbc1s3b4W2 : tensor<256x256x3x3xf32>
    %advhs3b4W2 = stablehlo.divide %advns3b4W2, %adbc2s3b4W2 : tensor<256x256x3x3xf32>
    %adlrs3b4W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adepss3b4W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adsqs3b4W2 = stablehlo.sqrt %advhs3b4W2 : tensor<256x256x3x3xf32>
    %addens3b4W2 = stablehlo.add %adsqs3b4W2, %adepss3b4W2 : tensor<256x256x3x3xf32>
    %adrats3b4W2 = stablehlo.divide %admhs3b4W2, %addens3b4W2 : tensor<256x256x3x3xf32>
    %adsts3b4W2 = stablehlo.multiply %adlrs3b4W2, %adrats3b4W2 : tensor<256x256x3x3xf32>
    %adsubs3b4W2 = stablehlo.subtract %s3b4W2, %adsts3b4W2 : tensor<256x256x3x3xf32>
    %adwds3b4W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %adwdlrs3b4W2 = stablehlo.multiply %adwds3b4W2, %adlrs3b4W2 : tensor<256x256x3x3xf32>
    %adwdps3b4W2 = stablehlo.multiply %adwdlrs3b4W2, %s3b4W2 : tensor<256x256x3x3xf32>
    %adnews3b4W2 = stablehlo.subtract %adsubs3b4W2, %adwdps3b4W2 : tensor<256x256x3x3xf32>
    %adb1s3b4b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4b2 = stablehlo.multiply %adb1s3b4b2, %s3b4b2m : tensor<256xf32>
    %admgs3b4b2 = stablehlo.multiply %adob1s3b4b2, %s3b4db2 : tensor<256xf32>
    %admns3b4b2 = stablehlo.add %admss3b4b2, %admgs3b4b2 : tensor<256xf32>
    %adb2s3b4b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4b2 = stablehlo.multiply %adb2s3b4b2, %s3b4b2v : tensor<256xf32>
    %adg2s3b4b2 = stablehlo.multiply %s3b4db2, %s3b4db2 : tensor<256xf32>
    %advgs3b4b2 = stablehlo.multiply %adob2s3b4b2, %adg2s3b4b2 : tensor<256xf32>
    %advns3b4b2 = stablehlo.add %advss3b4b2, %advgs3b4b2 : tensor<256xf32>
    %adbc1s3b4b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4b2 = stablehlo.divide %admns3b4b2, %adbc1s3b4b2 : tensor<256xf32>
    %advhs3b4b2 = stablehlo.divide %advns3b4b2, %adbc2s3b4b2 : tensor<256xf32>
    %adlrs3b4b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4b2 = stablehlo.sqrt %advhs3b4b2 : tensor<256xf32>
    %addens3b4b2 = stablehlo.add %adsqs3b4b2, %adepss3b4b2 : tensor<256xf32>
    %adrats3b4b2 = stablehlo.divide %admhs3b4b2, %addens3b4b2 : tensor<256xf32>
    %adsts3b4b2 = stablehlo.multiply %adlrs3b4b2, %adrats3b4b2 : tensor<256xf32>
    %adsubs3b4b2 = stablehlo.subtract %s3b4b2, %adsts3b4b2 : tensor<256xf32>
    %adwds3b4b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4b2 = stablehlo.multiply %adwds3b4b2, %adlrs3b4b2 : tensor<256xf32>
    %adwdps3b4b2 = stablehlo.multiply %adwdlrs3b4b2, %s3b4b2 : tensor<256xf32>
    %adnews3b4b2 = stablehlo.subtract %adsubs3b4b2, %adwdps3b4b2 : tensor<256xf32>
    %adb1s3b4g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4g2 = stablehlo.multiply %adb1s3b4g2, %s3b4g2m : tensor<256xf32>
    %admgs3b4g2 = stablehlo.multiply %adob1s3b4g2, %s3b4dn2dg : tensor<256xf32>
    %admns3b4g2 = stablehlo.add %admss3b4g2, %admgs3b4g2 : tensor<256xf32>
    %adb2s3b4g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4g2 = stablehlo.multiply %adb2s3b4g2, %s3b4g2v : tensor<256xf32>
    %adg2s3b4g2 = stablehlo.multiply %s3b4dn2dg, %s3b4dn2dg : tensor<256xf32>
    %advgs3b4g2 = stablehlo.multiply %adob2s3b4g2, %adg2s3b4g2 : tensor<256xf32>
    %advns3b4g2 = stablehlo.add %advss3b4g2, %advgs3b4g2 : tensor<256xf32>
    %adbc1s3b4g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4g2 = stablehlo.divide %admns3b4g2, %adbc1s3b4g2 : tensor<256xf32>
    %advhs3b4g2 = stablehlo.divide %advns3b4g2, %adbc2s3b4g2 : tensor<256xf32>
    %adlrs3b4g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4g2 = stablehlo.sqrt %advhs3b4g2 : tensor<256xf32>
    %addens3b4g2 = stablehlo.add %adsqs3b4g2, %adepss3b4g2 : tensor<256xf32>
    %adrats3b4g2 = stablehlo.divide %admhs3b4g2, %addens3b4g2 : tensor<256xf32>
    %adsts3b4g2 = stablehlo.multiply %adlrs3b4g2, %adrats3b4g2 : tensor<256xf32>
    %adsubs3b4g2 = stablehlo.subtract %s3b4g2, %adsts3b4g2 : tensor<256xf32>
    %adwds3b4g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4g2 = stablehlo.multiply %adwds3b4g2, %adlrs3b4g2 : tensor<256xf32>
    %adwdps3b4g2 = stablehlo.multiply %adwdlrs3b4g2, %s3b4g2 : tensor<256xf32>
    %adnews3b4g2 = stablehlo.subtract %adsubs3b4g2, %adwdps3b4g2 : tensor<256xf32>
    %adb1s3b4bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob1s3b4bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admss3b4bt2 = stablehlo.multiply %adb1s3b4bt2, %s3b4bt2m : tensor<256xf32>
    %admgs3b4bt2 = stablehlo.multiply %adob1s3b4bt2, %s3b4dn2db : tensor<256xf32>
    %admns3b4bt2 = stablehlo.add %admss3b4bt2, %admgs3b4bt2 : tensor<256xf32>
    %adb2s3b4bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adob2s3b4bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %advss3b4bt2 = stablehlo.multiply %adb2s3b4bt2, %s3b4bt2v : tensor<256xf32>
    %adg2s3b4bt2 = stablehlo.multiply %s3b4dn2db, %s3b4dn2db : tensor<256xf32>
    %advgs3b4bt2 = stablehlo.multiply %adob2s3b4bt2, %adg2s3b4bt2 : tensor<256xf32>
    %advns3b4bt2 = stablehlo.add %advss3b4bt2, %advgs3b4bt2 : tensor<256xf32>
    %adbc1s3b4bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adbc2s3b4bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %admhs3b4bt2 = stablehlo.divide %admns3b4bt2, %adbc1s3b4bt2 : tensor<256xf32>
    %advhs3b4bt2 = stablehlo.divide %advns3b4bt2, %adbc2s3b4bt2 : tensor<256xf32>
    %adlrs3b4bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adepss3b4bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adsqs3b4bt2 = stablehlo.sqrt %advhs3b4bt2 : tensor<256xf32>
    %addens3b4bt2 = stablehlo.add %adsqs3b4bt2, %adepss3b4bt2 : tensor<256xf32>
    %adrats3b4bt2 = stablehlo.divide %admhs3b4bt2, %addens3b4bt2 : tensor<256xf32>
    %adsts3b4bt2 = stablehlo.multiply %adlrs3b4bt2, %adrats3b4bt2 : tensor<256xf32>
    %adsubs3b4bt2 = stablehlo.subtract %s3b4bt2, %adsts3b4bt2 : tensor<256xf32>
    %adwds3b4bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %adwdlrs3b4bt2 = stablehlo.multiply %adwds3b4bt2, %adlrs3b4bt2 : tensor<256xf32>
    %adwdps3b4bt2 = stablehlo.multiply %adwdlrs3b4bt2, %s3b4bt2 : tensor<256xf32>
    %adnews3b4bt2 = stablehlo.subtract %adsubs3b4bt2, %adwdps3b4bt2 : tensor<256xf32>
    %adb1d4W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adob1d4W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %admsd4W1 = stablehlo.multiply %adb1d4W1, %d4W1m : tensor<512x256x3x3xf32>
    %admgd4W1 = stablehlo.multiply %adob1d4W1, %d4dW1 : tensor<512x256x3x3xf32>
    %admnd4W1 = stablehlo.add %admsd4W1, %admgd4W1 : tensor<512x256x3x3xf32>
    %adb2d4W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adob2d4W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %advsd4W1 = stablehlo.multiply %adb2d4W1, %d4W1v : tensor<512x256x3x3xf32>
    %adg2d4W1 = stablehlo.multiply %d4dW1, %d4dW1 : tensor<512x256x3x3xf32>
    %advgd4W1 = stablehlo.multiply %adob2d4W1, %adg2d4W1 : tensor<512x256x3x3xf32>
    %advnd4W1 = stablehlo.add %advsd4W1, %advgd4W1 : tensor<512x256x3x3xf32>
    %adbc1d4W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adbc2d4W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %admhd4W1 = stablehlo.divide %admnd4W1, %adbc1d4W1 : tensor<512x256x3x3xf32>
    %advhd4W1 = stablehlo.divide %advnd4W1, %adbc2d4W1 : tensor<512x256x3x3xf32>
    %adlrd4W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adepsd4W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adsqd4W1 = stablehlo.sqrt %advhd4W1 : tensor<512x256x3x3xf32>
    %addend4W1 = stablehlo.add %adsqd4W1, %adepsd4W1 : tensor<512x256x3x3xf32>
    %adratd4W1 = stablehlo.divide %admhd4W1, %addend4W1 : tensor<512x256x3x3xf32>
    %adstd4W1 = stablehlo.multiply %adlrd4W1, %adratd4W1 : tensor<512x256x3x3xf32>
    %adsubd4W1 = stablehlo.subtract %d4W1, %adstd4W1 : tensor<512x256x3x3xf32>
    %adwdd4W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adwdlrd4W1 = stablehlo.multiply %adwdd4W1, %adlrd4W1 : tensor<512x256x3x3xf32>
    %adwdpd4W1 = stablehlo.multiply %adwdlrd4W1, %d4W1 : tensor<512x256x3x3xf32>
    %adnewd4W1 = stablehlo.subtract %adsubd4W1, %adwdpd4W1 : tensor<512x256x3x3xf32>
    %adb1d4b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4b1 = stablehlo.multiply %adb1d4b1, %d4b1m : tensor<512xf32>
    %admgd4b1 = stablehlo.multiply %adob1d4b1, %d4db1 : tensor<512xf32>
    %admnd4b1 = stablehlo.add %admsd4b1, %admgd4b1 : tensor<512xf32>
    %adb2d4b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4b1 = stablehlo.multiply %adb2d4b1, %d4b1v : tensor<512xf32>
    %adg2d4b1 = stablehlo.multiply %d4db1, %d4db1 : tensor<512xf32>
    %advgd4b1 = stablehlo.multiply %adob2d4b1, %adg2d4b1 : tensor<512xf32>
    %advnd4b1 = stablehlo.add %advsd4b1, %advgd4b1 : tensor<512xf32>
    %adbc1d4b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4b1 = stablehlo.divide %admnd4b1, %adbc1d4b1 : tensor<512xf32>
    %advhd4b1 = stablehlo.divide %advnd4b1, %adbc2d4b1 : tensor<512xf32>
    %adlrd4b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4b1 = stablehlo.sqrt %advhd4b1 : tensor<512xf32>
    %addend4b1 = stablehlo.add %adsqd4b1, %adepsd4b1 : tensor<512xf32>
    %adratd4b1 = stablehlo.divide %admhd4b1, %addend4b1 : tensor<512xf32>
    %adstd4b1 = stablehlo.multiply %adlrd4b1, %adratd4b1 : tensor<512xf32>
    %adsubd4b1 = stablehlo.subtract %d4b1, %adstd4b1 : tensor<512xf32>
    %adwdd4b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4b1 = stablehlo.multiply %adwdd4b1, %adlrd4b1 : tensor<512xf32>
    %adwdpd4b1 = stablehlo.multiply %adwdlrd4b1, %d4b1 : tensor<512xf32>
    %adnewd4b1 = stablehlo.subtract %adsubd4b1, %adwdpd4b1 : tensor<512xf32>
    %adb1d4g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4g1 = stablehlo.multiply %adb1d4g1, %d4g1m : tensor<512xf32>
    %admgd4g1 = stablehlo.multiply %adob1d4g1, %d4dn1dg : tensor<512xf32>
    %admnd4g1 = stablehlo.add %admsd4g1, %admgd4g1 : tensor<512xf32>
    %adb2d4g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4g1 = stablehlo.multiply %adb2d4g1, %d4g1v : tensor<512xf32>
    %adg2d4g1 = stablehlo.multiply %d4dn1dg, %d4dn1dg : tensor<512xf32>
    %advgd4g1 = stablehlo.multiply %adob2d4g1, %adg2d4g1 : tensor<512xf32>
    %advnd4g1 = stablehlo.add %advsd4g1, %advgd4g1 : tensor<512xf32>
    %adbc1d4g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4g1 = stablehlo.divide %admnd4g1, %adbc1d4g1 : tensor<512xf32>
    %advhd4g1 = stablehlo.divide %advnd4g1, %adbc2d4g1 : tensor<512xf32>
    %adlrd4g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4g1 = stablehlo.sqrt %advhd4g1 : tensor<512xf32>
    %addend4g1 = stablehlo.add %adsqd4g1, %adepsd4g1 : tensor<512xf32>
    %adratd4g1 = stablehlo.divide %admhd4g1, %addend4g1 : tensor<512xf32>
    %adstd4g1 = stablehlo.multiply %adlrd4g1, %adratd4g1 : tensor<512xf32>
    %adsubd4g1 = stablehlo.subtract %d4g1, %adstd4g1 : tensor<512xf32>
    %adwdd4g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4g1 = stablehlo.multiply %adwdd4g1, %adlrd4g1 : tensor<512xf32>
    %adwdpd4g1 = stablehlo.multiply %adwdlrd4g1, %d4g1 : tensor<512xf32>
    %adnewd4g1 = stablehlo.subtract %adsubd4g1, %adwdpd4g1 : tensor<512xf32>
    %adb1d4bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4bt1 = stablehlo.multiply %adb1d4bt1, %d4bt1m : tensor<512xf32>
    %admgd4bt1 = stablehlo.multiply %adob1d4bt1, %d4dn1db : tensor<512xf32>
    %admnd4bt1 = stablehlo.add %admsd4bt1, %admgd4bt1 : tensor<512xf32>
    %adb2d4bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4bt1 = stablehlo.multiply %adb2d4bt1, %d4bt1v : tensor<512xf32>
    %adg2d4bt1 = stablehlo.multiply %d4dn1db, %d4dn1db : tensor<512xf32>
    %advgd4bt1 = stablehlo.multiply %adob2d4bt1, %adg2d4bt1 : tensor<512xf32>
    %advnd4bt1 = stablehlo.add %advsd4bt1, %advgd4bt1 : tensor<512xf32>
    %adbc1d4bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4bt1 = stablehlo.divide %admnd4bt1, %adbc1d4bt1 : tensor<512xf32>
    %advhd4bt1 = stablehlo.divide %advnd4bt1, %adbc2d4bt1 : tensor<512xf32>
    %adlrd4bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4bt1 = stablehlo.sqrt %advhd4bt1 : tensor<512xf32>
    %addend4bt1 = stablehlo.add %adsqd4bt1, %adepsd4bt1 : tensor<512xf32>
    %adratd4bt1 = stablehlo.divide %admhd4bt1, %addend4bt1 : tensor<512xf32>
    %adstd4bt1 = stablehlo.multiply %adlrd4bt1, %adratd4bt1 : tensor<512xf32>
    %adsubd4bt1 = stablehlo.subtract %d4bt1, %adstd4bt1 : tensor<512xf32>
    %adwdd4bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4bt1 = stablehlo.multiply %adwdd4bt1, %adlrd4bt1 : tensor<512xf32>
    %adwdpd4bt1 = stablehlo.multiply %adwdlrd4bt1, %d4bt1 : tensor<512xf32>
    %adnewd4bt1 = stablehlo.subtract %adsubd4bt1, %adwdpd4bt1 : tensor<512xf32>
    %adb1d4W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob1d4W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admsd4W2 = stablehlo.multiply %adb1d4W2, %d4W2m : tensor<512x512x3x3xf32>
    %admgd4W2 = stablehlo.multiply %adob1d4W2, %d4dW2 : tensor<512x512x3x3xf32>
    %admnd4W2 = stablehlo.add %admsd4W2, %admgd4W2 : tensor<512x512x3x3xf32>
    %adb2d4W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob2d4W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %advsd4W2 = stablehlo.multiply %adb2d4W2, %d4W2v : tensor<512x512x3x3xf32>
    %adg2d4W2 = stablehlo.multiply %d4dW2, %d4dW2 : tensor<512x512x3x3xf32>
    %advgd4W2 = stablehlo.multiply %adob2d4W2, %adg2d4W2 : tensor<512x512x3x3xf32>
    %advnd4W2 = stablehlo.add %advsd4W2, %advgd4W2 : tensor<512x512x3x3xf32>
    %adbc1d4W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adbc2d4W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admhd4W2 = stablehlo.divide %admnd4W2, %adbc1d4W2 : tensor<512x512x3x3xf32>
    %advhd4W2 = stablehlo.divide %advnd4W2, %adbc2d4W2 : tensor<512x512x3x3xf32>
    %adlrd4W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adepsd4W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adsqd4W2 = stablehlo.sqrt %advhd4W2 : tensor<512x512x3x3xf32>
    %addend4W2 = stablehlo.add %adsqd4W2, %adepsd4W2 : tensor<512x512x3x3xf32>
    %adratd4W2 = stablehlo.divide %admhd4W2, %addend4W2 : tensor<512x512x3x3xf32>
    %adstd4W2 = stablehlo.multiply %adlrd4W2, %adratd4W2 : tensor<512x512x3x3xf32>
    %adsubd4W2 = stablehlo.subtract %d4W2, %adstd4W2 : tensor<512x512x3x3xf32>
    %adwdd4W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adwdlrd4W2 = stablehlo.multiply %adwdd4W2, %adlrd4W2 : tensor<512x512x3x3xf32>
    %adwdpd4W2 = stablehlo.multiply %adwdlrd4W2, %d4W2 : tensor<512x512x3x3xf32>
    %adnewd4W2 = stablehlo.subtract %adsubd4W2, %adwdpd4W2 : tensor<512x512x3x3xf32>
    %adb1d4b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4b2 = stablehlo.multiply %adb1d4b2, %d4b2m : tensor<512xf32>
    %admgd4b2 = stablehlo.multiply %adob1d4b2, %d4db2 : tensor<512xf32>
    %admnd4b2 = stablehlo.add %admsd4b2, %admgd4b2 : tensor<512xf32>
    %adb2d4b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4b2 = stablehlo.multiply %adb2d4b2, %d4b2v : tensor<512xf32>
    %adg2d4b2 = stablehlo.multiply %d4db2, %d4db2 : tensor<512xf32>
    %advgd4b2 = stablehlo.multiply %adob2d4b2, %adg2d4b2 : tensor<512xf32>
    %advnd4b2 = stablehlo.add %advsd4b2, %advgd4b2 : tensor<512xf32>
    %adbc1d4b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4b2 = stablehlo.divide %admnd4b2, %adbc1d4b2 : tensor<512xf32>
    %advhd4b2 = stablehlo.divide %advnd4b2, %adbc2d4b2 : tensor<512xf32>
    %adlrd4b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4b2 = stablehlo.sqrt %advhd4b2 : tensor<512xf32>
    %addend4b2 = stablehlo.add %adsqd4b2, %adepsd4b2 : tensor<512xf32>
    %adratd4b2 = stablehlo.divide %admhd4b2, %addend4b2 : tensor<512xf32>
    %adstd4b2 = stablehlo.multiply %adlrd4b2, %adratd4b2 : tensor<512xf32>
    %adsubd4b2 = stablehlo.subtract %d4b2, %adstd4b2 : tensor<512xf32>
    %adwdd4b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4b2 = stablehlo.multiply %adwdd4b2, %adlrd4b2 : tensor<512xf32>
    %adwdpd4b2 = stablehlo.multiply %adwdlrd4b2, %d4b2 : tensor<512xf32>
    %adnewd4b2 = stablehlo.subtract %adsubd4b2, %adwdpd4b2 : tensor<512xf32>
    %adb1d4g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4g2 = stablehlo.multiply %adb1d4g2, %d4g2m : tensor<512xf32>
    %admgd4g2 = stablehlo.multiply %adob1d4g2, %d4dn2dg : tensor<512xf32>
    %admnd4g2 = stablehlo.add %admsd4g2, %admgd4g2 : tensor<512xf32>
    %adb2d4g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4g2 = stablehlo.multiply %adb2d4g2, %d4g2v : tensor<512xf32>
    %adg2d4g2 = stablehlo.multiply %d4dn2dg, %d4dn2dg : tensor<512xf32>
    %advgd4g2 = stablehlo.multiply %adob2d4g2, %adg2d4g2 : tensor<512xf32>
    %advnd4g2 = stablehlo.add %advsd4g2, %advgd4g2 : tensor<512xf32>
    %adbc1d4g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4g2 = stablehlo.divide %admnd4g2, %adbc1d4g2 : tensor<512xf32>
    %advhd4g2 = stablehlo.divide %advnd4g2, %adbc2d4g2 : tensor<512xf32>
    %adlrd4g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4g2 = stablehlo.sqrt %advhd4g2 : tensor<512xf32>
    %addend4g2 = stablehlo.add %adsqd4g2, %adepsd4g2 : tensor<512xf32>
    %adratd4g2 = stablehlo.divide %admhd4g2, %addend4g2 : tensor<512xf32>
    %adstd4g2 = stablehlo.multiply %adlrd4g2, %adratd4g2 : tensor<512xf32>
    %adsubd4g2 = stablehlo.subtract %d4g2, %adstd4g2 : tensor<512xf32>
    %adwdd4g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4g2 = stablehlo.multiply %adwdd4g2, %adlrd4g2 : tensor<512xf32>
    %adwdpd4g2 = stablehlo.multiply %adwdlrd4g2, %d4g2 : tensor<512xf32>
    %adnewd4g2 = stablehlo.subtract %adsubd4g2, %adwdpd4g2 : tensor<512xf32>
    %adb1d4bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4bt2 = stablehlo.multiply %adb1d4bt2, %d4bt2m : tensor<512xf32>
    %admgd4bt2 = stablehlo.multiply %adob1d4bt2, %d4dn2db : tensor<512xf32>
    %admnd4bt2 = stablehlo.add %admsd4bt2, %admgd4bt2 : tensor<512xf32>
    %adb2d4bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4bt2 = stablehlo.multiply %adb2d4bt2, %d4bt2v : tensor<512xf32>
    %adg2d4bt2 = stablehlo.multiply %d4dn2db, %d4dn2db : tensor<512xf32>
    %advgd4bt2 = stablehlo.multiply %adob2d4bt2, %adg2d4bt2 : tensor<512xf32>
    %advnd4bt2 = stablehlo.add %advsd4bt2, %advgd4bt2 : tensor<512xf32>
    %adbc1d4bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4bt2 = stablehlo.divide %admnd4bt2, %adbc1d4bt2 : tensor<512xf32>
    %advhd4bt2 = stablehlo.divide %advnd4bt2, %adbc2d4bt2 : tensor<512xf32>
    %adlrd4bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4bt2 = stablehlo.sqrt %advhd4bt2 : tensor<512xf32>
    %addend4bt2 = stablehlo.add %adsqd4bt2, %adepsd4bt2 : tensor<512xf32>
    %adratd4bt2 = stablehlo.divide %admhd4bt2, %addend4bt2 : tensor<512xf32>
    %adstd4bt2 = stablehlo.multiply %adlrd4bt2, %adratd4bt2 : tensor<512xf32>
    %adsubd4bt2 = stablehlo.subtract %d4bt2, %adstd4bt2 : tensor<512xf32>
    %adwdd4bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4bt2 = stablehlo.multiply %adwdd4bt2, %adlrd4bt2 : tensor<512xf32>
    %adwdpd4bt2 = stablehlo.multiply %adwdlrd4bt2, %d4bt2 : tensor<512xf32>
    %adnewd4bt2 = stablehlo.subtract %adsubd4bt2, %adwdpd4bt2 : tensor<512xf32>
    %adb1d4Wp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adob1d4Wp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %admsd4Wp = stablehlo.multiply %adb1d4Wp, %d4Wpm : tensor<512x256x3x3xf32>
    %admgd4Wp = stablehlo.multiply %adob1d4Wp, %d4dWp : tensor<512x256x3x3xf32>
    %admnd4Wp = stablehlo.add %admsd4Wp, %admgd4Wp : tensor<512x256x3x3xf32>
    %adb2d4Wp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adob2d4Wp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %advsd4Wp = stablehlo.multiply %adb2d4Wp, %d4Wpv : tensor<512x256x3x3xf32>
    %adg2d4Wp = stablehlo.multiply %d4dWp, %d4dWp : tensor<512x256x3x3xf32>
    %advgd4Wp = stablehlo.multiply %adob2d4Wp, %adg2d4Wp : tensor<512x256x3x3xf32>
    %advnd4Wp = stablehlo.add %advsd4Wp, %advgd4Wp : tensor<512x256x3x3xf32>
    %adbc1d4Wp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adbc2d4Wp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %admhd4Wp = stablehlo.divide %admnd4Wp, %adbc1d4Wp : tensor<512x256x3x3xf32>
    %advhd4Wp = stablehlo.divide %advnd4Wp, %adbc2d4Wp : tensor<512x256x3x3xf32>
    %adlrd4Wp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adepsd4Wp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adsqd4Wp = stablehlo.sqrt %advhd4Wp : tensor<512x256x3x3xf32>
    %addend4Wp = stablehlo.add %adsqd4Wp, %adepsd4Wp : tensor<512x256x3x3xf32>
    %adratd4Wp = stablehlo.divide %admhd4Wp, %addend4Wp : tensor<512x256x3x3xf32>
    %adstd4Wp = stablehlo.multiply %adlrd4Wp, %adratd4Wp : tensor<512x256x3x3xf32>
    %adsubd4Wp = stablehlo.subtract %d4Wp, %adstd4Wp : tensor<512x256x3x3xf32>
    %adwdd4Wp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %adwdlrd4Wp = stablehlo.multiply %adwdd4Wp, %adlrd4Wp : tensor<512x256x3x3xf32>
    %adwdpd4Wp = stablehlo.multiply %adwdlrd4Wp, %d4Wp : tensor<512x256x3x3xf32>
    %adnewd4Wp = stablehlo.subtract %adsubd4Wp, %adwdpd4Wp : tensor<512x256x3x3xf32>
    %adb1d4bp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4bp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4bp = stablehlo.multiply %adb1d4bp, %d4bpm : tensor<512xf32>
    %admgd4bp = stablehlo.multiply %adob1d4bp, %d4dbp : tensor<512xf32>
    %admnd4bp = stablehlo.add %admsd4bp, %admgd4bp : tensor<512xf32>
    %adb2d4bp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4bp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4bp = stablehlo.multiply %adb2d4bp, %d4bpv : tensor<512xf32>
    %adg2d4bp = stablehlo.multiply %d4dbp, %d4dbp : tensor<512xf32>
    %advgd4bp = stablehlo.multiply %adob2d4bp, %adg2d4bp : tensor<512xf32>
    %advnd4bp = stablehlo.add %advsd4bp, %advgd4bp : tensor<512xf32>
    %adbc1d4bp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4bp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4bp = stablehlo.divide %admnd4bp, %adbc1d4bp : tensor<512xf32>
    %advhd4bp = stablehlo.divide %advnd4bp, %adbc2d4bp : tensor<512xf32>
    %adlrd4bp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4bp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4bp = stablehlo.sqrt %advhd4bp : tensor<512xf32>
    %addend4bp = stablehlo.add %adsqd4bp, %adepsd4bp : tensor<512xf32>
    %adratd4bp = stablehlo.divide %admhd4bp, %addend4bp : tensor<512xf32>
    %adstd4bp = stablehlo.multiply %adlrd4bp, %adratd4bp : tensor<512xf32>
    %adsubd4bp = stablehlo.subtract %d4bp, %adstd4bp : tensor<512xf32>
    %adwdd4bp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4bp = stablehlo.multiply %adwdd4bp, %adlrd4bp : tensor<512xf32>
    %adwdpd4bp = stablehlo.multiply %adwdlrd4bp, %d4bp : tensor<512xf32>
    %adnewd4bp = stablehlo.subtract %adsubd4bp, %adwdpd4bp : tensor<512xf32>
    %adb1d4gp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4gp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4gp = stablehlo.multiply %adb1d4gp, %d4gpm : tensor<512xf32>
    %admgd4gp = stablehlo.multiply %adob1d4gp, %d4dnpdg : tensor<512xf32>
    %admnd4gp = stablehlo.add %admsd4gp, %admgd4gp : tensor<512xf32>
    %adb2d4gp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4gp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4gp = stablehlo.multiply %adb2d4gp, %d4gpv : tensor<512xf32>
    %adg2d4gp = stablehlo.multiply %d4dnpdg, %d4dnpdg : tensor<512xf32>
    %advgd4gp = stablehlo.multiply %adob2d4gp, %adg2d4gp : tensor<512xf32>
    %advnd4gp = stablehlo.add %advsd4gp, %advgd4gp : tensor<512xf32>
    %adbc1d4gp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4gp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4gp = stablehlo.divide %admnd4gp, %adbc1d4gp : tensor<512xf32>
    %advhd4gp = stablehlo.divide %advnd4gp, %adbc2d4gp : tensor<512xf32>
    %adlrd4gp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4gp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4gp = stablehlo.sqrt %advhd4gp : tensor<512xf32>
    %addend4gp = stablehlo.add %adsqd4gp, %adepsd4gp : tensor<512xf32>
    %adratd4gp = stablehlo.divide %admhd4gp, %addend4gp : tensor<512xf32>
    %adstd4gp = stablehlo.multiply %adlrd4gp, %adratd4gp : tensor<512xf32>
    %adsubd4gp = stablehlo.subtract %d4gp, %adstd4gp : tensor<512xf32>
    %adwdd4gp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4gp = stablehlo.multiply %adwdd4gp, %adlrd4gp : tensor<512xf32>
    %adwdpd4gp = stablehlo.multiply %adwdlrd4gp, %d4gp : tensor<512xf32>
    %adnewd4gp = stablehlo.subtract %adsubd4gp, %adwdpd4gp : tensor<512xf32>
    %adb1d4btp = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1d4btp = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admsd4btp = stablehlo.multiply %adb1d4btp, %d4btpm : tensor<512xf32>
    %admgd4btp = stablehlo.multiply %adob1d4btp, %d4dnpdb : tensor<512xf32>
    %admnd4btp = stablehlo.add %admsd4btp, %admgd4btp : tensor<512xf32>
    %adb2d4btp = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2d4btp = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advsd4btp = stablehlo.multiply %adb2d4btp, %d4btpv : tensor<512xf32>
    %adg2d4btp = stablehlo.multiply %d4dnpdb, %d4dnpdb : tensor<512xf32>
    %advgd4btp = stablehlo.multiply %adob2d4btp, %adg2d4btp : tensor<512xf32>
    %advnd4btp = stablehlo.add %advsd4btp, %advgd4btp : tensor<512xf32>
    %adbc1d4btp = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2d4btp = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhd4btp = stablehlo.divide %admnd4btp, %adbc1d4btp : tensor<512xf32>
    %advhd4btp = stablehlo.divide %advnd4btp, %adbc2d4btp : tensor<512xf32>
    %adlrd4btp = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepsd4btp = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqd4btp = stablehlo.sqrt %advhd4btp : tensor<512xf32>
    %addend4btp = stablehlo.add %adsqd4btp, %adepsd4btp : tensor<512xf32>
    %adratd4btp = stablehlo.divide %admhd4btp, %addend4btp : tensor<512xf32>
    %adstd4btp = stablehlo.multiply %adlrd4btp, %adratd4btp : tensor<512xf32>
    %adsubd4btp = stablehlo.subtract %d4btp, %adstd4btp : tensor<512xf32>
    %adwdd4btp = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrd4btp = stablehlo.multiply %adwdd4btp, %adlrd4btp : tensor<512xf32>
    %adwdpd4btp = stablehlo.multiply %adwdlrd4btp, %d4btp : tensor<512xf32>
    %adnewd4btp = stablehlo.subtract %adsubd4btp, %adwdpd4btp : tensor<512xf32>
    %adb1s4b0W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob1s4b0W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admss4b0W1 = stablehlo.multiply %adb1s4b0W1, %s4b0W1m : tensor<512x512x3x3xf32>
    %admgs4b0W1 = stablehlo.multiply %adob1s4b0W1, %s4b0dW1 : tensor<512x512x3x3xf32>
    %admns4b0W1 = stablehlo.add %admss4b0W1, %admgs4b0W1 : tensor<512x512x3x3xf32>
    %adb2s4b0W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob2s4b0W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %advss4b0W1 = stablehlo.multiply %adb2s4b0W1, %s4b0W1v : tensor<512x512x3x3xf32>
    %adg2s4b0W1 = stablehlo.multiply %s4b0dW1, %s4b0dW1 : tensor<512x512x3x3xf32>
    %advgs4b0W1 = stablehlo.multiply %adob2s4b0W1, %adg2s4b0W1 : tensor<512x512x3x3xf32>
    %advns4b0W1 = stablehlo.add %advss4b0W1, %advgs4b0W1 : tensor<512x512x3x3xf32>
    %adbc1s4b0W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adbc2s4b0W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admhs4b0W1 = stablehlo.divide %admns4b0W1, %adbc1s4b0W1 : tensor<512x512x3x3xf32>
    %advhs4b0W1 = stablehlo.divide %advns4b0W1, %adbc2s4b0W1 : tensor<512x512x3x3xf32>
    %adlrs4b0W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adepss4b0W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adsqs4b0W1 = stablehlo.sqrt %advhs4b0W1 : tensor<512x512x3x3xf32>
    %addens4b0W1 = stablehlo.add %adsqs4b0W1, %adepss4b0W1 : tensor<512x512x3x3xf32>
    %adrats4b0W1 = stablehlo.divide %admhs4b0W1, %addens4b0W1 : tensor<512x512x3x3xf32>
    %adsts4b0W1 = stablehlo.multiply %adlrs4b0W1, %adrats4b0W1 : tensor<512x512x3x3xf32>
    %adsubs4b0W1 = stablehlo.subtract %s4b0W1, %adsts4b0W1 : tensor<512x512x3x3xf32>
    %adwds4b0W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adwdlrs4b0W1 = stablehlo.multiply %adwds4b0W1, %adlrs4b0W1 : tensor<512x512x3x3xf32>
    %adwdps4b0W1 = stablehlo.multiply %adwdlrs4b0W1, %s4b0W1 : tensor<512x512x3x3xf32>
    %adnews4b0W1 = stablehlo.subtract %adsubs4b0W1, %adwdps4b0W1 : tensor<512x512x3x3xf32>
    %adb1s4b0b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0b1 = stablehlo.multiply %adb1s4b0b1, %s4b0b1m : tensor<512xf32>
    %admgs4b0b1 = stablehlo.multiply %adob1s4b0b1, %s4b0db1 : tensor<512xf32>
    %admns4b0b1 = stablehlo.add %admss4b0b1, %admgs4b0b1 : tensor<512xf32>
    %adb2s4b0b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0b1 = stablehlo.multiply %adb2s4b0b1, %s4b0b1v : tensor<512xf32>
    %adg2s4b0b1 = stablehlo.multiply %s4b0db1, %s4b0db1 : tensor<512xf32>
    %advgs4b0b1 = stablehlo.multiply %adob2s4b0b1, %adg2s4b0b1 : tensor<512xf32>
    %advns4b0b1 = stablehlo.add %advss4b0b1, %advgs4b0b1 : tensor<512xf32>
    %adbc1s4b0b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0b1 = stablehlo.divide %admns4b0b1, %adbc1s4b0b1 : tensor<512xf32>
    %advhs4b0b1 = stablehlo.divide %advns4b0b1, %adbc2s4b0b1 : tensor<512xf32>
    %adlrs4b0b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0b1 = stablehlo.sqrt %advhs4b0b1 : tensor<512xf32>
    %addens4b0b1 = stablehlo.add %adsqs4b0b1, %adepss4b0b1 : tensor<512xf32>
    %adrats4b0b1 = stablehlo.divide %admhs4b0b1, %addens4b0b1 : tensor<512xf32>
    %adsts4b0b1 = stablehlo.multiply %adlrs4b0b1, %adrats4b0b1 : tensor<512xf32>
    %adsubs4b0b1 = stablehlo.subtract %s4b0b1, %adsts4b0b1 : tensor<512xf32>
    %adwds4b0b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0b1 = stablehlo.multiply %adwds4b0b1, %adlrs4b0b1 : tensor<512xf32>
    %adwdps4b0b1 = stablehlo.multiply %adwdlrs4b0b1, %s4b0b1 : tensor<512xf32>
    %adnews4b0b1 = stablehlo.subtract %adsubs4b0b1, %adwdps4b0b1 : tensor<512xf32>
    %adb1s4b0g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0g1 = stablehlo.multiply %adb1s4b0g1, %s4b0g1m : tensor<512xf32>
    %admgs4b0g1 = stablehlo.multiply %adob1s4b0g1, %s4b0dn1dg : tensor<512xf32>
    %admns4b0g1 = stablehlo.add %admss4b0g1, %admgs4b0g1 : tensor<512xf32>
    %adb2s4b0g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0g1 = stablehlo.multiply %adb2s4b0g1, %s4b0g1v : tensor<512xf32>
    %adg2s4b0g1 = stablehlo.multiply %s4b0dn1dg, %s4b0dn1dg : tensor<512xf32>
    %advgs4b0g1 = stablehlo.multiply %adob2s4b0g1, %adg2s4b0g1 : tensor<512xf32>
    %advns4b0g1 = stablehlo.add %advss4b0g1, %advgs4b0g1 : tensor<512xf32>
    %adbc1s4b0g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0g1 = stablehlo.divide %admns4b0g1, %adbc1s4b0g1 : tensor<512xf32>
    %advhs4b0g1 = stablehlo.divide %advns4b0g1, %adbc2s4b0g1 : tensor<512xf32>
    %adlrs4b0g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0g1 = stablehlo.sqrt %advhs4b0g1 : tensor<512xf32>
    %addens4b0g1 = stablehlo.add %adsqs4b0g1, %adepss4b0g1 : tensor<512xf32>
    %adrats4b0g1 = stablehlo.divide %admhs4b0g1, %addens4b0g1 : tensor<512xf32>
    %adsts4b0g1 = stablehlo.multiply %adlrs4b0g1, %adrats4b0g1 : tensor<512xf32>
    %adsubs4b0g1 = stablehlo.subtract %s4b0g1, %adsts4b0g1 : tensor<512xf32>
    %adwds4b0g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0g1 = stablehlo.multiply %adwds4b0g1, %adlrs4b0g1 : tensor<512xf32>
    %adwdps4b0g1 = stablehlo.multiply %adwdlrs4b0g1, %s4b0g1 : tensor<512xf32>
    %adnews4b0g1 = stablehlo.subtract %adsubs4b0g1, %adwdps4b0g1 : tensor<512xf32>
    %adb1s4b0bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0bt1 = stablehlo.multiply %adb1s4b0bt1, %s4b0bt1m : tensor<512xf32>
    %admgs4b0bt1 = stablehlo.multiply %adob1s4b0bt1, %s4b0dn1db : tensor<512xf32>
    %admns4b0bt1 = stablehlo.add %admss4b0bt1, %admgs4b0bt1 : tensor<512xf32>
    %adb2s4b0bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0bt1 = stablehlo.multiply %adb2s4b0bt1, %s4b0bt1v : tensor<512xf32>
    %adg2s4b0bt1 = stablehlo.multiply %s4b0dn1db, %s4b0dn1db : tensor<512xf32>
    %advgs4b0bt1 = stablehlo.multiply %adob2s4b0bt1, %adg2s4b0bt1 : tensor<512xf32>
    %advns4b0bt1 = stablehlo.add %advss4b0bt1, %advgs4b0bt1 : tensor<512xf32>
    %adbc1s4b0bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0bt1 = stablehlo.divide %admns4b0bt1, %adbc1s4b0bt1 : tensor<512xf32>
    %advhs4b0bt1 = stablehlo.divide %advns4b0bt1, %adbc2s4b0bt1 : tensor<512xf32>
    %adlrs4b0bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0bt1 = stablehlo.sqrt %advhs4b0bt1 : tensor<512xf32>
    %addens4b0bt1 = stablehlo.add %adsqs4b0bt1, %adepss4b0bt1 : tensor<512xf32>
    %adrats4b0bt1 = stablehlo.divide %admhs4b0bt1, %addens4b0bt1 : tensor<512xf32>
    %adsts4b0bt1 = stablehlo.multiply %adlrs4b0bt1, %adrats4b0bt1 : tensor<512xf32>
    %adsubs4b0bt1 = stablehlo.subtract %s4b0bt1, %adsts4b0bt1 : tensor<512xf32>
    %adwds4b0bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0bt1 = stablehlo.multiply %adwds4b0bt1, %adlrs4b0bt1 : tensor<512xf32>
    %adwdps4b0bt1 = stablehlo.multiply %adwdlrs4b0bt1, %s4b0bt1 : tensor<512xf32>
    %adnews4b0bt1 = stablehlo.subtract %adsubs4b0bt1, %adwdps4b0bt1 : tensor<512xf32>
    %adb1s4b0W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob1s4b0W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admss4b0W2 = stablehlo.multiply %adb1s4b0W2, %s4b0W2m : tensor<512x512x3x3xf32>
    %admgs4b0W2 = stablehlo.multiply %adob1s4b0W2, %s4b0dW2 : tensor<512x512x3x3xf32>
    %admns4b0W2 = stablehlo.add %admss4b0W2, %admgs4b0W2 : tensor<512x512x3x3xf32>
    %adb2s4b0W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob2s4b0W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %advss4b0W2 = stablehlo.multiply %adb2s4b0W2, %s4b0W2v : tensor<512x512x3x3xf32>
    %adg2s4b0W2 = stablehlo.multiply %s4b0dW2, %s4b0dW2 : tensor<512x512x3x3xf32>
    %advgs4b0W2 = stablehlo.multiply %adob2s4b0W2, %adg2s4b0W2 : tensor<512x512x3x3xf32>
    %advns4b0W2 = stablehlo.add %advss4b0W2, %advgs4b0W2 : tensor<512x512x3x3xf32>
    %adbc1s4b0W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adbc2s4b0W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admhs4b0W2 = stablehlo.divide %admns4b0W2, %adbc1s4b0W2 : tensor<512x512x3x3xf32>
    %advhs4b0W2 = stablehlo.divide %advns4b0W2, %adbc2s4b0W2 : tensor<512x512x3x3xf32>
    %adlrs4b0W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adepss4b0W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adsqs4b0W2 = stablehlo.sqrt %advhs4b0W2 : tensor<512x512x3x3xf32>
    %addens4b0W2 = stablehlo.add %adsqs4b0W2, %adepss4b0W2 : tensor<512x512x3x3xf32>
    %adrats4b0W2 = stablehlo.divide %admhs4b0W2, %addens4b0W2 : tensor<512x512x3x3xf32>
    %adsts4b0W2 = stablehlo.multiply %adlrs4b0W2, %adrats4b0W2 : tensor<512x512x3x3xf32>
    %adsubs4b0W2 = stablehlo.subtract %s4b0W2, %adsts4b0W2 : tensor<512x512x3x3xf32>
    %adwds4b0W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adwdlrs4b0W2 = stablehlo.multiply %adwds4b0W2, %adlrs4b0W2 : tensor<512x512x3x3xf32>
    %adwdps4b0W2 = stablehlo.multiply %adwdlrs4b0W2, %s4b0W2 : tensor<512x512x3x3xf32>
    %adnews4b0W2 = stablehlo.subtract %adsubs4b0W2, %adwdps4b0W2 : tensor<512x512x3x3xf32>
    %adb1s4b0b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0b2 = stablehlo.multiply %adb1s4b0b2, %s4b0b2m : tensor<512xf32>
    %admgs4b0b2 = stablehlo.multiply %adob1s4b0b2, %s4b0db2 : tensor<512xf32>
    %admns4b0b2 = stablehlo.add %admss4b0b2, %admgs4b0b2 : tensor<512xf32>
    %adb2s4b0b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0b2 = stablehlo.multiply %adb2s4b0b2, %s4b0b2v : tensor<512xf32>
    %adg2s4b0b2 = stablehlo.multiply %s4b0db2, %s4b0db2 : tensor<512xf32>
    %advgs4b0b2 = stablehlo.multiply %adob2s4b0b2, %adg2s4b0b2 : tensor<512xf32>
    %advns4b0b2 = stablehlo.add %advss4b0b2, %advgs4b0b2 : tensor<512xf32>
    %adbc1s4b0b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0b2 = stablehlo.divide %admns4b0b2, %adbc1s4b0b2 : tensor<512xf32>
    %advhs4b0b2 = stablehlo.divide %advns4b0b2, %adbc2s4b0b2 : tensor<512xf32>
    %adlrs4b0b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0b2 = stablehlo.sqrt %advhs4b0b2 : tensor<512xf32>
    %addens4b0b2 = stablehlo.add %adsqs4b0b2, %adepss4b0b2 : tensor<512xf32>
    %adrats4b0b2 = stablehlo.divide %admhs4b0b2, %addens4b0b2 : tensor<512xf32>
    %adsts4b0b2 = stablehlo.multiply %adlrs4b0b2, %adrats4b0b2 : tensor<512xf32>
    %adsubs4b0b2 = stablehlo.subtract %s4b0b2, %adsts4b0b2 : tensor<512xf32>
    %adwds4b0b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0b2 = stablehlo.multiply %adwds4b0b2, %adlrs4b0b2 : tensor<512xf32>
    %adwdps4b0b2 = stablehlo.multiply %adwdlrs4b0b2, %s4b0b2 : tensor<512xf32>
    %adnews4b0b2 = stablehlo.subtract %adsubs4b0b2, %adwdps4b0b2 : tensor<512xf32>
    %adb1s4b0g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0g2 = stablehlo.multiply %adb1s4b0g2, %s4b0g2m : tensor<512xf32>
    %admgs4b0g2 = stablehlo.multiply %adob1s4b0g2, %s4b0dn2dg : tensor<512xf32>
    %admns4b0g2 = stablehlo.add %admss4b0g2, %admgs4b0g2 : tensor<512xf32>
    %adb2s4b0g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0g2 = stablehlo.multiply %adb2s4b0g2, %s4b0g2v : tensor<512xf32>
    %adg2s4b0g2 = stablehlo.multiply %s4b0dn2dg, %s4b0dn2dg : tensor<512xf32>
    %advgs4b0g2 = stablehlo.multiply %adob2s4b0g2, %adg2s4b0g2 : tensor<512xf32>
    %advns4b0g2 = stablehlo.add %advss4b0g2, %advgs4b0g2 : tensor<512xf32>
    %adbc1s4b0g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0g2 = stablehlo.divide %admns4b0g2, %adbc1s4b0g2 : tensor<512xf32>
    %advhs4b0g2 = stablehlo.divide %advns4b0g2, %adbc2s4b0g2 : tensor<512xf32>
    %adlrs4b0g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0g2 = stablehlo.sqrt %advhs4b0g2 : tensor<512xf32>
    %addens4b0g2 = stablehlo.add %adsqs4b0g2, %adepss4b0g2 : tensor<512xf32>
    %adrats4b0g2 = stablehlo.divide %admhs4b0g2, %addens4b0g2 : tensor<512xf32>
    %adsts4b0g2 = stablehlo.multiply %adlrs4b0g2, %adrats4b0g2 : tensor<512xf32>
    %adsubs4b0g2 = stablehlo.subtract %s4b0g2, %adsts4b0g2 : tensor<512xf32>
    %adwds4b0g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0g2 = stablehlo.multiply %adwds4b0g2, %adlrs4b0g2 : tensor<512xf32>
    %adwdps4b0g2 = stablehlo.multiply %adwdlrs4b0g2, %s4b0g2 : tensor<512xf32>
    %adnews4b0g2 = stablehlo.subtract %adsubs4b0g2, %adwdps4b0g2 : tensor<512xf32>
    %adb1s4b0bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b0bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b0bt2 = stablehlo.multiply %adb1s4b0bt2, %s4b0bt2m : tensor<512xf32>
    %admgs4b0bt2 = stablehlo.multiply %adob1s4b0bt2, %s4b0dn2db : tensor<512xf32>
    %admns4b0bt2 = stablehlo.add %admss4b0bt2, %admgs4b0bt2 : tensor<512xf32>
    %adb2s4b0bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b0bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b0bt2 = stablehlo.multiply %adb2s4b0bt2, %s4b0bt2v : tensor<512xf32>
    %adg2s4b0bt2 = stablehlo.multiply %s4b0dn2db, %s4b0dn2db : tensor<512xf32>
    %advgs4b0bt2 = stablehlo.multiply %adob2s4b0bt2, %adg2s4b0bt2 : tensor<512xf32>
    %advns4b0bt2 = stablehlo.add %advss4b0bt2, %advgs4b0bt2 : tensor<512xf32>
    %adbc1s4b0bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b0bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b0bt2 = stablehlo.divide %admns4b0bt2, %adbc1s4b0bt2 : tensor<512xf32>
    %advhs4b0bt2 = stablehlo.divide %advns4b0bt2, %adbc2s4b0bt2 : tensor<512xf32>
    %adlrs4b0bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b0bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b0bt2 = stablehlo.sqrt %advhs4b0bt2 : tensor<512xf32>
    %addens4b0bt2 = stablehlo.add %adsqs4b0bt2, %adepss4b0bt2 : tensor<512xf32>
    %adrats4b0bt2 = stablehlo.divide %admhs4b0bt2, %addens4b0bt2 : tensor<512xf32>
    %adsts4b0bt2 = stablehlo.multiply %adlrs4b0bt2, %adrats4b0bt2 : tensor<512xf32>
    %adsubs4b0bt2 = stablehlo.subtract %s4b0bt2, %adsts4b0bt2 : tensor<512xf32>
    %adwds4b0bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b0bt2 = stablehlo.multiply %adwds4b0bt2, %adlrs4b0bt2 : tensor<512xf32>
    %adwdps4b0bt2 = stablehlo.multiply %adwdlrs4b0bt2, %s4b0bt2 : tensor<512xf32>
    %adnews4b0bt2 = stablehlo.subtract %adsubs4b0bt2, %adwdps4b0bt2 : tensor<512xf32>
    %adb1s4b1W1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob1s4b1W1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admss4b1W1 = stablehlo.multiply %adb1s4b1W1, %s4b1W1m : tensor<512x512x3x3xf32>
    %admgs4b1W1 = stablehlo.multiply %adob1s4b1W1, %s4b1dW1 : tensor<512x512x3x3xf32>
    %admns4b1W1 = stablehlo.add %admss4b1W1, %admgs4b1W1 : tensor<512x512x3x3xf32>
    %adb2s4b1W1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob2s4b1W1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %advss4b1W1 = stablehlo.multiply %adb2s4b1W1, %s4b1W1v : tensor<512x512x3x3xf32>
    %adg2s4b1W1 = stablehlo.multiply %s4b1dW1, %s4b1dW1 : tensor<512x512x3x3xf32>
    %advgs4b1W1 = stablehlo.multiply %adob2s4b1W1, %adg2s4b1W1 : tensor<512x512x3x3xf32>
    %advns4b1W1 = stablehlo.add %advss4b1W1, %advgs4b1W1 : tensor<512x512x3x3xf32>
    %adbc1s4b1W1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adbc2s4b1W1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admhs4b1W1 = stablehlo.divide %admns4b1W1, %adbc1s4b1W1 : tensor<512x512x3x3xf32>
    %advhs4b1W1 = stablehlo.divide %advns4b1W1, %adbc2s4b1W1 : tensor<512x512x3x3xf32>
    %adlrs4b1W1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adepss4b1W1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adsqs4b1W1 = stablehlo.sqrt %advhs4b1W1 : tensor<512x512x3x3xf32>
    %addens4b1W1 = stablehlo.add %adsqs4b1W1, %adepss4b1W1 : tensor<512x512x3x3xf32>
    %adrats4b1W1 = stablehlo.divide %admhs4b1W1, %addens4b1W1 : tensor<512x512x3x3xf32>
    %adsts4b1W1 = stablehlo.multiply %adlrs4b1W1, %adrats4b1W1 : tensor<512x512x3x3xf32>
    %adsubs4b1W1 = stablehlo.subtract %s4b1W1, %adsts4b1W1 : tensor<512x512x3x3xf32>
    %adwds4b1W1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adwdlrs4b1W1 = stablehlo.multiply %adwds4b1W1, %adlrs4b1W1 : tensor<512x512x3x3xf32>
    %adwdps4b1W1 = stablehlo.multiply %adwdlrs4b1W1, %s4b1W1 : tensor<512x512x3x3xf32>
    %adnews4b1W1 = stablehlo.subtract %adsubs4b1W1, %adwdps4b1W1 : tensor<512x512x3x3xf32>
    %adb1s4b1b1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1b1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1b1 = stablehlo.multiply %adb1s4b1b1, %s4b1b1m : tensor<512xf32>
    %admgs4b1b1 = stablehlo.multiply %adob1s4b1b1, %s4b1db1 : tensor<512xf32>
    %admns4b1b1 = stablehlo.add %admss4b1b1, %admgs4b1b1 : tensor<512xf32>
    %adb2s4b1b1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1b1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1b1 = stablehlo.multiply %adb2s4b1b1, %s4b1b1v : tensor<512xf32>
    %adg2s4b1b1 = stablehlo.multiply %s4b1db1, %s4b1db1 : tensor<512xf32>
    %advgs4b1b1 = stablehlo.multiply %adob2s4b1b1, %adg2s4b1b1 : tensor<512xf32>
    %advns4b1b1 = stablehlo.add %advss4b1b1, %advgs4b1b1 : tensor<512xf32>
    %adbc1s4b1b1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1b1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1b1 = stablehlo.divide %admns4b1b1, %adbc1s4b1b1 : tensor<512xf32>
    %advhs4b1b1 = stablehlo.divide %advns4b1b1, %adbc2s4b1b1 : tensor<512xf32>
    %adlrs4b1b1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1b1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1b1 = stablehlo.sqrt %advhs4b1b1 : tensor<512xf32>
    %addens4b1b1 = stablehlo.add %adsqs4b1b1, %adepss4b1b1 : tensor<512xf32>
    %adrats4b1b1 = stablehlo.divide %admhs4b1b1, %addens4b1b1 : tensor<512xf32>
    %adsts4b1b1 = stablehlo.multiply %adlrs4b1b1, %adrats4b1b1 : tensor<512xf32>
    %adsubs4b1b1 = stablehlo.subtract %s4b1b1, %adsts4b1b1 : tensor<512xf32>
    %adwds4b1b1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1b1 = stablehlo.multiply %adwds4b1b1, %adlrs4b1b1 : tensor<512xf32>
    %adwdps4b1b1 = stablehlo.multiply %adwdlrs4b1b1, %s4b1b1 : tensor<512xf32>
    %adnews4b1b1 = stablehlo.subtract %adsubs4b1b1, %adwdps4b1b1 : tensor<512xf32>
    %adb1s4b1g1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1g1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1g1 = stablehlo.multiply %adb1s4b1g1, %s4b1g1m : tensor<512xf32>
    %admgs4b1g1 = stablehlo.multiply %adob1s4b1g1, %s4b1dn1dg : tensor<512xf32>
    %admns4b1g1 = stablehlo.add %admss4b1g1, %admgs4b1g1 : tensor<512xf32>
    %adb2s4b1g1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1g1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1g1 = stablehlo.multiply %adb2s4b1g1, %s4b1g1v : tensor<512xf32>
    %adg2s4b1g1 = stablehlo.multiply %s4b1dn1dg, %s4b1dn1dg : tensor<512xf32>
    %advgs4b1g1 = stablehlo.multiply %adob2s4b1g1, %adg2s4b1g1 : tensor<512xf32>
    %advns4b1g1 = stablehlo.add %advss4b1g1, %advgs4b1g1 : tensor<512xf32>
    %adbc1s4b1g1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1g1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1g1 = stablehlo.divide %admns4b1g1, %adbc1s4b1g1 : tensor<512xf32>
    %advhs4b1g1 = stablehlo.divide %advns4b1g1, %adbc2s4b1g1 : tensor<512xf32>
    %adlrs4b1g1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1g1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1g1 = stablehlo.sqrt %advhs4b1g1 : tensor<512xf32>
    %addens4b1g1 = stablehlo.add %adsqs4b1g1, %adepss4b1g1 : tensor<512xf32>
    %adrats4b1g1 = stablehlo.divide %admhs4b1g1, %addens4b1g1 : tensor<512xf32>
    %adsts4b1g1 = stablehlo.multiply %adlrs4b1g1, %adrats4b1g1 : tensor<512xf32>
    %adsubs4b1g1 = stablehlo.subtract %s4b1g1, %adsts4b1g1 : tensor<512xf32>
    %adwds4b1g1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1g1 = stablehlo.multiply %adwds4b1g1, %adlrs4b1g1 : tensor<512xf32>
    %adwdps4b1g1 = stablehlo.multiply %adwdlrs4b1g1, %s4b1g1 : tensor<512xf32>
    %adnews4b1g1 = stablehlo.subtract %adsubs4b1g1, %adwdps4b1g1 : tensor<512xf32>
    %adb1s4b1bt1 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1bt1 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1bt1 = stablehlo.multiply %adb1s4b1bt1, %s4b1bt1m : tensor<512xf32>
    %admgs4b1bt1 = stablehlo.multiply %adob1s4b1bt1, %s4b1dn1db : tensor<512xf32>
    %admns4b1bt1 = stablehlo.add %admss4b1bt1, %admgs4b1bt1 : tensor<512xf32>
    %adb2s4b1bt1 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1bt1 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1bt1 = stablehlo.multiply %adb2s4b1bt1, %s4b1bt1v : tensor<512xf32>
    %adg2s4b1bt1 = stablehlo.multiply %s4b1dn1db, %s4b1dn1db : tensor<512xf32>
    %advgs4b1bt1 = stablehlo.multiply %adob2s4b1bt1, %adg2s4b1bt1 : tensor<512xf32>
    %advns4b1bt1 = stablehlo.add %advss4b1bt1, %advgs4b1bt1 : tensor<512xf32>
    %adbc1s4b1bt1 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1bt1 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1bt1 = stablehlo.divide %admns4b1bt1, %adbc1s4b1bt1 : tensor<512xf32>
    %advhs4b1bt1 = stablehlo.divide %advns4b1bt1, %adbc2s4b1bt1 : tensor<512xf32>
    %adlrs4b1bt1 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1bt1 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1bt1 = stablehlo.sqrt %advhs4b1bt1 : tensor<512xf32>
    %addens4b1bt1 = stablehlo.add %adsqs4b1bt1, %adepss4b1bt1 : tensor<512xf32>
    %adrats4b1bt1 = stablehlo.divide %admhs4b1bt1, %addens4b1bt1 : tensor<512xf32>
    %adsts4b1bt1 = stablehlo.multiply %adlrs4b1bt1, %adrats4b1bt1 : tensor<512xf32>
    %adsubs4b1bt1 = stablehlo.subtract %s4b1bt1, %adsts4b1bt1 : tensor<512xf32>
    %adwds4b1bt1 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1bt1 = stablehlo.multiply %adwds4b1bt1, %adlrs4b1bt1 : tensor<512xf32>
    %adwdps4b1bt1 = stablehlo.multiply %adwdlrs4b1bt1, %s4b1bt1 : tensor<512xf32>
    %adnews4b1bt1 = stablehlo.subtract %adsubs4b1bt1, %adwdps4b1bt1 : tensor<512xf32>
    %adb1s4b1W2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob1s4b1W2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admss4b1W2 = stablehlo.multiply %adb1s4b1W2, %s4b1W2m : tensor<512x512x3x3xf32>
    %admgs4b1W2 = stablehlo.multiply %adob1s4b1W2, %s4b1dW2 : tensor<512x512x3x3xf32>
    %admns4b1W2 = stablehlo.add %admss4b1W2, %admgs4b1W2 : tensor<512x512x3x3xf32>
    %adb2s4b1W2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adob2s4b1W2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %advss4b1W2 = stablehlo.multiply %adb2s4b1W2, %s4b1W2v : tensor<512x512x3x3xf32>
    %adg2s4b1W2 = stablehlo.multiply %s4b1dW2, %s4b1dW2 : tensor<512x512x3x3xf32>
    %advgs4b1W2 = stablehlo.multiply %adob2s4b1W2, %adg2s4b1W2 : tensor<512x512x3x3xf32>
    %advns4b1W2 = stablehlo.add %advss4b1W2, %advgs4b1W2 : tensor<512x512x3x3xf32>
    %adbc1s4b1W2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adbc2s4b1W2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %admhs4b1W2 = stablehlo.divide %admns4b1W2, %adbc1s4b1W2 : tensor<512x512x3x3xf32>
    %advhs4b1W2 = stablehlo.divide %advns4b1W2, %adbc2s4b1W2 : tensor<512x512x3x3xf32>
    %adlrs4b1W2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adepss4b1W2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adsqs4b1W2 = stablehlo.sqrt %advhs4b1W2 : tensor<512x512x3x3xf32>
    %addens4b1W2 = stablehlo.add %adsqs4b1W2, %adepss4b1W2 : tensor<512x512x3x3xf32>
    %adrats4b1W2 = stablehlo.divide %admhs4b1W2, %addens4b1W2 : tensor<512x512x3x3xf32>
    %adsts4b1W2 = stablehlo.multiply %adlrs4b1W2, %adrats4b1W2 : tensor<512x512x3x3xf32>
    %adsubs4b1W2 = stablehlo.subtract %s4b1W2, %adsts4b1W2 : tensor<512x512x3x3xf32>
    %adwds4b1W2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %adwdlrs4b1W2 = stablehlo.multiply %adwds4b1W2, %adlrs4b1W2 : tensor<512x512x3x3xf32>
    %adwdps4b1W2 = stablehlo.multiply %adwdlrs4b1W2, %s4b1W2 : tensor<512x512x3x3xf32>
    %adnews4b1W2 = stablehlo.subtract %adsubs4b1W2, %adwdps4b1W2 : tensor<512x512x3x3xf32>
    %adb1s4b1b2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1b2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1b2 = stablehlo.multiply %adb1s4b1b2, %s4b1b2m : tensor<512xf32>
    %admgs4b1b2 = stablehlo.multiply %adob1s4b1b2, %s4b1db2 : tensor<512xf32>
    %admns4b1b2 = stablehlo.add %admss4b1b2, %admgs4b1b2 : tensor<512xf32>
    %adb2s4b1b2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1b2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1b2 = stablehlo.multiply %adb2s4b1b2, %s4b1b2v : tensor<512xf32>
    %adg2s4b1b2 = stablehlo.multiply %s4b1db2, %s4b1db2 : tensor<512xf32>
    %advgs4b1b2 = stablehlo.multiply %adob2s4b1b2, %adg2s4b1b2 : tensor<512xf32>
    %advns4b1b2 = stablehlo.add %advss4b1b2, %advgs4b1b2 : tensor<512xf32>
    %adbc1s4b1b2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1b2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1b2 = stablehlo.divide %admns4b1b2, %adbc1s4b1b2 : tensor<512xf32>
    %advhs4b1b2 = stablehlo.divide %advns4b1b2, %adbc2s4b1b2 : tensor<512xf32>
    %adlrs4b1b2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1b2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1b2 = stablehlo.sqrt %advhs4b1b2 : tensor<512xf32>
    %addens4b1b2 = stablehlo.add %adsqs4b1b2, %adepss4b1b2 : tensor<512xf32>
    %adrats4b1b2 = stablehlo.divide %admhs4b1b2, %addens4b1b2 : tensor<512xf32>
    %adsts4b1b2 = stablehlo.multiply %adlrs4b1b2, %adrats4b1b2 : tensor<512xf32>
    %adsubs4b1b2 = stablehlo.subtract %s4b1b2, %adsts4b1b2 : tensor<512xf32>
    %adwds4b1b2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1b2 = stablehlo.multiply %adwds4b1b2, %adlrs4b1b2 : tensor<512xf32>
    %adwdps4b1b2 = stablehlo.multiply %adwdlrs4b1b2, %s4b1b2 : tensor<512xf32>
    %adnews4b1b2 = stablehlo.subtract %adsubs4b1b2, %adwdps4b1b2 : tensor<512xf32>
    %adb1s4b1g2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1g2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1g2 = stablehlo.multiply %adb1s4b1g2, %s4b1g2m : tensor<512xf32>
    %admgs4b1g2 = stablehlo.multiply %adob1s4b1g2, %s4b1dn2dg : tensor<512xf32>
    %admns4b1g2 = stablehlo.add %admss4b1g2, %admgs4b1g2 : tensor<512xf32>
    %adb2s4b1g2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1g2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1g2 = stablehlo.multiply %adb2s4b1g2, %s4b1g2v : tensor<512xf32>
    %adg2s4b1g2 = stablehlo.multiply %s4b1dn2dg, %s4b1dn2dg : tensor<512xf32>
    %advgs4b1g2 = stablehlo.multiply %adob2s4b1g2, %adg2s4b1g2 : tensor<512xf32>
    %advns4b1g2 = stablehlo.add %advss4b1g2, %advgs4b1g2 : tensor<512xf32>
    %adbc1s4b1g2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1g2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1g2 = stablehlo.divide %admns4b1g2, %adbc1s4b1g2 : tensor<512xf32>
    %advhs4b1g2 = stablehlo.divide %advns4b1g2, %adbc2s4b1g2 : tensor<512xf32>
    %adlrs4b1g2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1g2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1g2 = stablehlo.sqrt %advhs4b1g2 : tensor<512xf32>
    %addens4b1g2 = stablehlo.add %adsqs4b1g2, %adepss4b1g2 : tensor<512xf32>
    %adrats4b1g2 = stablehlo.divide %admhs4b1g2, %addens4b1g2 : tensor<512xf32>
    %adsts4b1g2 = stablehlo.multiply %adlrs4b1g2, %adrats4b1g2 : tensor<512xf32>
    %adsubs4b1g2 = stablehlo.subtract %s4b1g2, %adsts4b1g2 : tensor<512xf32>
    %adwds4b1g2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1g2 = stablehlo.multiply %adwds4b1g2, %adlrs4b1g2 : tensor<512xf32>
    %adwdps4b1g2 = stablehlo.multiply %adwdlrs4b1g2, %s4b1g2 : tensor<512xf32>
    %adnews4b1g2 = stablehlo.subtract %adsubs4b1g2, %adwdps4b1g2 : tensor<512xf32>
    %adb1s4b1bt2 = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob1s4b1bt2 = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admss4b1bt2 = stablehlo.multiply %adb1s4b1bt2, %s4b1bt2m : tensor<512xf32>
    %admgs4b1bt2 = stablehlo.multiply %adob1s4b1bt2, %s4b1dn2db : tensor<512xf32>
    %admns4b1bt2 = stablehlo.add %admss4b1bt2, %admgs4b1bt2 : tensor<512xf32>
    %adb2s4b1bt2 = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adob2s4b1bt2 = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %advss4b1bt2 = stablehlo.multiply %adb2s4b1bt2, %s4b1bt2v : tensor<512xf32>
    %adg2s4b1bt2 = stablehlo.multiply %s4b1dn2db, %s4b1dn2db : tensor<512xf32>
    %advgs4b1bt2 = stablehlo.multiply %adob2s4b1bt2, %adg2s4b1bt2 : tensor<512xf32>
    %advns4b1bt2 = stablehlo.add %advss4b1bt2, %advgs4b1bt2 : tensor<512xf32>
    %adbc1s4b1bt2 = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adbc2s4b1bt2 = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %admhs4b1bt2 = stablehlo.divide %admns4b1bt2, %adbc1s4b1bt2 : tensor<512xf32>
    %advhs4b1bt2 = stablehlo.divide %advns4b1bt2, %adbc2s4b1bt2 : tensor<512xf32>
    %adlrs4b1bt2 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adepss4b1bt2 = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adsqs4b1bt2 = stablehlo.sqrt %advhs4b1bt2 : tensor<512xf32>
    %addens4b1bt2 = stablehlo.add %adsqs4b1bt2, %adepss4b1bt2 : tensor<512xf32>
    %adrats4b1bt2 = stablehlo.divide %admhs4b1bt2, %addens4b1bt2 : tensor<512xf32>
    %adsts4b1bt2 = stablehlo.multiply %adlrs4b1bt2, %adrats4b1bt2 : tensor<512xf32>
    %adsubs4b1bt2 = stablehlo.subtract %s4b1bt2, %adsts4b1bt2 : tensor<512xf32>
    %adwds4b1bt2 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %adwdlrs4b1bt2 = stablehlo.multiply %adwds4b1bt2, %adlrs4b1bt2 : tensor<512xf32>
    %adwdps4b1bt2 = stablehlo.multiply %adwdlrs4b1bt2, %s4b1bt2 : tensor<512xf32>
    %adnews4b1bt2 = stablehlo.subtract %adsubs4b1bt2, %adwdps4b1bt2 : tensor<512xf32>
    %adb1Wd = stablehlo.broadcast_in_dim %b1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adob1Wd = stablehlo.broadcast_in_dim %ob1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %admsWd = stablehlo.multiply %adb1Wd, %Wdm : tensor<512x10xf32>
    %admgWd = stablehlo.multiply %adob1Wd, %dWd : tensor<512x10xf32>
    %admnWd = stablehlo.add %admsWd, %admgWd : tensor<512x10xf32>
    %adb2Wd = stablehlo.broadcast_in_dim %b2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adob2Wd = stablehlo.broadcast_in_dim %ob2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %advsWd = stablehlo.multiply %adb2Wd, %Wdv : tensor<512x10xf32>
    %adg2Wd = stablehlo.multiply %dWd, %dWd : tensor<512x10xf32>
    %advgWd = stablehlo.multiply %adob2Wd, %adg2Wd : tensor<512x10xf32>
    %advnWd = stablehlo.add %advsWd, %advgWd : tensor<512x10xf32>
    %adbc1Wd = stablehlo.broadcast_in_dim %bc1, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adbc2Wd = stablehlo.broadcast_in_dim %bc2, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %admhWd = stablehlo.divide %admnWd, %adbc1Wd : tensor<512x10xf32>
    %advhWd = stablehlo.divide %advnWd, %adbc2Wd : tensor<512x10xf32>
    %adlrWd = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adepsWd = stablehlo.broadcast_in_dim %eps, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adsqWd = stablehlo.sqrt %advhWd : tensor<512x10xf32>
    %addenWd = stablehlo.add %adsqWd, %adepsWd : tensor<512x10xf32>
    %adratWd = stablehlo.divide %admhWd, %addenWd : tensor<512x10xf32>
    %adstWd = stablehlo.multiply %adlrWd, %adratWd : tensor<512x10xf32>
    %adsubWd = stablehlo.subtract %Wd, %adstWd : tensor<512x10xf32>
    %adwdWd = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x10xf32>
    %adwdlrWd = stablehlo.multiply %adwdWd, %adlrWd : tensor<512x10xf32>
    %adwdpWd = stablehlo.multiply %adwdlrWd, %Wd : tensor<512x10xf32>
    %adnewWd = stablehlo.subtract %adsubWd, %adwdpWd : tensor<512x10xf32>
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
    return %adnewsW, %adnewsb, %adnewsg, %adnewsbt, %adnews1b0W1, %adnews1b0b1, %adnews1b0g1, %adnews1b0bt1, %adnews1b0W2, %adnews1b0b2, %adnews1b0g2, %adnews1b0bt2, %adnews1b1W1, %adnews1b1b1, %adnews1b1g1, %adnews1b1bt1, %adnews1b1W2, %adnews1b1b2, %adnews1b1g2, %adnews1b1bt2, %adnews1b2W1, %adnews1b2b1, %adnews1b2g1, %adnews1b2bt1, %adnews1b2W2, %adnews1b2b2, %adnews1b2g2, %adnews1b2bt2, %adnewd2W1, %adnewd2b1, %adnewd2g1, %adnewd2bt1, %adnewd2W2, %adnewd2b2, %adnewd2g2, %adnewd2bt2, %adnewd2Wp, %adnewd2bp, %adnewd2gp, %adnewd2btp, %adnews2b0W1, %adnews2b0b1, %adnews2b0g1, %adnews2b0bt1, %adnews2b0W2, %adnews2b0b2, %adnews2b0g2, %adnews2b0bt2, %adnews2b1W1, %adnews2b1b1, %adnews2b1g1, %adnews2b1bt1, %adnews2b1W2, %adnews2b1b2, %adnews2b1g2, %adnews2b1bt2, %adnews2b2W1, %adnews2b2b1, %adnews2b2g1, %adnews2b2bt1, %adnews2b2W2, %adnews2b2b2, %adnews2b2g2, %adnews2b2bt2, %adnewd3W1, %adnewd3b1, %adnewd3g1, %adnewd3bt1, %adnewd3W2, %adnewd3b2, %adnewd3g2, %adnewd3bt2, %adnewd3Wp, %adnewd3bp, %adnewd3gp, %adnewd3btp, %adnews3b0W1, %adnews3b0b1, %adnews3b0g1, %adnews3b0bt1, %adnews3b0W2, %adnews3b0b2, %adnews3b0g2, %adnews3b0bt2, %adnews3b1W1, %adnews3b1b1, %adnews3b1g1, %adnews3b1bt1, %adnews3b1W2, %adnews3b1b2, %adnews3b1g2, %adnews3b1bt2, %adnews3b2W1, %adnews3b2b1, %adnews3b2g1, %adnews3b2bt1, %adnews3b2W2, %adnews3b2b2, %adnews3b2g2, %adnews3b2bt2, %adnews3b3W1, %adnews3b3b1, %adnews3b3g1, %adnews3b3bt1, %adnews3b3W2, %adnews3b3b2, %adnews3b3g2, %adnews3b3bt2, %adnews3b4W1, %adnews3b4b1, %adnews3b4g1, %adnews3b4bt1, %adnews3b4W2, %adnews3b4b2, %adnews3b4g2, %adnews3b4bt2, %adnewd4W1, %adnewd4b1, %adnewd4g1, %adnewd4bt1, %adnewd4W2, %adnewd4b2, %adnewd4g2, %adnewd4bt2, %adnewd4Wp, %adnewd4bp, %adnewd4gp, %adnewd4btp, %adnews4b0W1, %adnews4b0b1, %adnews4b0g1, %adnews4b0bt1, %adnews4b0W2, %adnews4b0b2, %adnews4b0g2, %adnews4b0bt2, %adnews4b1W1, %adnews4b1b1, %adnews4b1g1, %adnews4b1bt1, %adnews4b1W2, %adnews4b1b2, %adnews4b1g2, %adnews4b1bt2, %adnewWd, %adnewbd, %admnsW, %admnsb, %admnsg, %admnsbt, %admns1b0W1, %admns1b0b1, %admns1b0g1, %admns1b0bt1, %admns1b0W2, %admns1b0b2, %admns1b0g2, %admns1b0bt2, %admns1b1W1, %admns1b1b1, %admns1b1g1, %admns1b1bt1, %admns1b1W2, %admns1b1b2, %admns1b1g2, %admns1b1bt2, %admns1b2W1, %admns1b2b1, %admns1b2g1, %admns1b2bt1, %admns1b2W2, %admns1b2b2, %admns1b2g2, %admns1b2bt2, %admnd2W1, %admnd2b1, %admnd2g1, %admnd2bt1, %admnd2W2, %admnd2b2, %admnd2g2, %admnd2bt2, %admnd2Wp, %admnd2bp, %admnd2gp, %admnd2btp, %admns2b0W1, %admns2b0b1, %admns2b0g1, %admns2b0bt1, %admns2b0W2, %admns2b0b2, %admns2b0g2, %admns2b0bt2, %admns2b1W1, %admns2b1b1, %admns2b1g1, %admns2b1bt1, %admns2b1W2, %admns2b1b2, %admns2b1g2, %admns2b1bt2, %admns2b2W1, %admns2b2b1, %admns2b2g1, %admns2b2bt1, %admns2b2W2, %admns2b2b2, %admns2b2g2, %admns2b2bt2, %admnd3W1, %admnd3b1, %admnd3g1, %admnd3bt1, %admnd3W2, %admnd3b2, %admnd3g2, %admnd3bt2, %admnd3Wp, %admnd3bp, %admnd3gp, %admnd3btp, %admns3b0W1, %admns3b0b1, %admns3b0g1, %admns3b0bt1, %admns3b0W2, %admns3b0b2, %admns3b0g2, %admns3b0bt2, %admns3b1W1, %admns3b1b1, %admns3b1g1, %admns3b1bt1, %admns3b1W2, %admns3b1b2, %admns3b1g2, %admns3b1bt2, %admns3b2W1, %admns3b2b1, %admns3b2g1, %admns3b2bt1, %admns3b2W2, %admns3b2b2, %admns3b2g2, %admns3b2bt2, %admns3b3W1, %admns3b3b1, %admns3b3g1, %admns3b3bt1, %admns3b3W2, %admns3b3b2, %admns3b3g2, %admns3b3bt2, %admns3b4W1, %admns3b4b1, %admns3b4g1, %admns3b4bt1, %admns3b4W2, %admns3b4b2, %admns3b4g2, %admns3b4bt2, %admnd4W1, %admnd4b1, %admnd4g1, %admnd4bt1, %admnd4W2, %admnd4b2, %admnd4g2, %admnd4bt2, %admnd4Wp, %admnd4bp, %admnd4gp, %admnd4btp, %admns4b0W1, %admns4b0b1, %admns4b0g1, %admns4b0bt1, %admns4b0W2, %admns4b0b2, %admns4b0g2, %admns4b0bt2, %admns4b1W1, %admns4b1b1, %admns4b1g1, %admns4b1bt1, %admns4b1W2, %admns4b1b2, %admns4b1g2, %admns4b1bt2, %admnWd, %admnbd, %advnsW, %advnsb, %advnsg, %advnsbt, %advns1b0W1, %advns1b0b1, %advns1b0g1, %advns1b0bt1, %advns1b0W2, %advns1b0b2, %advns1b0g2, %advns1b0bt2, %advns1b1W1, %advns1b1b1, %advns1b1g1, %advns1b1bt1, %advns1b1W2, %advns1b1b2, %advns1b1g2, %advns1b1bt2, %advns1b2W1, %advns1b2b1, %advns1b2g1, %advns1b2bt1, %advns1b2W2, %advns1b2b2, %advns1b2g2, %advns1b2bt2, %advnd2W1, %advnd2b1, %advnd2g1, %advnd2bt1, %advnd2W2, %advnd2b2, %advnd2g2, %advnd2bt2, %advnd2Wp, %advnd2bp, %advnd2gp, %advnd2btp, %advns2b0W1, %advns2b0b1, %advns2b0g1, %advns2b0bt1, %advns2b0W2, %advns2b0b2, %advns2b0g2, %advns2b0bt2, %advns2b1W1, %advns2b1b1, %advns2b1g1, %advns2b1bt1, %advns2b1W2, %advns2b1b2, %advns2b1g2, %advns2b1bt2, %advns2b2W1, %advns2b2b1, %advns2b2g1, %advns2b2bt1, %advns2b2W2, %advns2b2b2, %advns2b2g2, %advns2b2bt2, %advnd3W1, %advnd3b1, %advnd3g1, %advnd3bt1, %advnd3W2, %advnd3b2, %advnd3g2, %advnd3bt2, %advnd3Wp, %advnd3bp, %advnd3gp, %advnd3btp, %advns3b0W1, %advns3b0b1, %advns3b0g1, %advns3b0bt1, %advns3b0W2, %advns3b0b2, %advns3b0g2, %advns3b0bt2, %advns3b1W1, %advns3b1b1, %advns3b1g1, %advns3b1bt1, %advns3b1W2, %advns3b1b2, %advns3b1g2, %advns3b1bt2, %advns3b2W1, %advns3b2b1, %advns3b2g1, %advns3b2bt1, %advns3b2W2, %advns3b2b2, %advns3b2g2, %advns3b2bt2, %advns3b3W1, %advns3b3b1, %advns3b3g1, %advns3b3bt1, %advns3b3W2, %advns3b3b2, %advns3b3g2, %advns3b3bt2, %advns3b4W1, %advns3b4b1, %advns3b4g1, %advns3b4bt1, %advns3b4W2, %advns3b4b2, %advns3b4g2, %advns3b4bt2, %advnd4W1, %advnd4b1, %advnd4g1, %advnd4bt1, %advnd4W2, %advnd4b2, %advnd4g2, %advnd4bt2, %advnd4Wp, %advnd4bp, %advnd4gp, %advnd4btp, %advns4b0W1, %advns4b0b1, %advns4b0g1, %advns4b0bt1, %advns4b0W2, %advns4b0b2, %advns4b0g2, %advns4b0bt2, %advns4b1W1, %advns4b1b1, %advns4b1g1, %advns4b1bt1, %advns4b1W2, %advns4b1b2, %advns4b1g2, %advns4b1bt2, %advnWd, %advnbd, %loss, %bc1, %bc2 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x10xf32>, tensor<10xf32>, tensor<f32>, tensor<f32>, tensor<f32>
  }
}
