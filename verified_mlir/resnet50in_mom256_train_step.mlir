module @m {
  func.func @resnet50in_mom256_train_step(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x1x1xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b0W3m: tensor<256x64x1x1xf32>, %s1b0g3m: tensor<256xf32>, %s1b0bt3m: tensor<256xf32>, %s1b0Wpm: tensor<256x64x1x1xf32>, %s1b0gpm: tensor<256xf32>, %s1b0btpm: tensor<256xf32>, %s1b1W1m: tensor<64x256x1x1xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b1W3m: tensor<256x64x1x1xf32>, %s1b1g3m: tensor<256xf32>, %s1b1bt3m: tensor<256xf32>, %s1b2W1m: tensor<64x256x1x1xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %s1b2W3m: tensor<256x64x1x1xf32>, %s1b2g3m: tensor<256xf32>, %s1b2bt3m: tensor<256xf32>, %s2b0W1m: tensor<128x256x1x1xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b0W3m: tensor<512x128x1x1xf32>, %s2b0g3m: tensor<512xf32>, %s2b0bt3m: tensor<512xf32>, %s2b0Wpm: tensor<512x256x1x1xf32>, %s2b0gpm: tensor<512xf32>, %s2b0btpm: tensor<512xf32>, %s2b1W1m: tensor<128x512x1x1xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b1W3m: tensor<512x128x1x1xf32>, %s2b1g3m: tensor<512xf32>, %s2b1bt3m: tensor<512xf32>, %s2b2W1m: tensor<128x512x1x1xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %s2b2W3m: tensor<512x128x1x1xf32>, %s2b2g3m: tensor<512xf32>, %s2b2bt3m: tensor<512xf32>, %s2b3W1m: tensor<128x512x1x1xf32>, %s2b3g1m: tensor<128xf32>, %s2b3bt1m: tensor<128xf32>, %s2b3W2m: tensor<128x128x3x3xf32>, %s2b3g2m: tensor<128xf32>, %s2b3bt2m: tensor<128xf32>, %s2b3W3m: tensor<512x128x1x1xf32>, %s2b3g3m: tensor<512xf32>, %s2b3bt3m: tensor<512xf32>, %s3b0W1m: tensor<256x512x1x1xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b0W3m: tensor<1024x256x1x1xf32>, %s3b0g3m: tensor<1024xf32>, %s3b0bt3m: tensor<1024xf32>, %s3b0Wpm: tensor<1024x512x1x1xf32>, %s3b0gpm: tensor<1024xf32>, %s3b0btpm: tensor<1024xf32>, %s3b1W1m: tensor<256x1024x1x1xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b1W3m: tensor<1024x256x1x1xf32>, %s3b1g3m: tensor<1024xf32>, %s3b1bt3m: tensor<1024xf32>, %s3b2W1m: tensor<256x1024x1x1xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b2W3m: tensor<1024x256x1x1xf32>, %s3b2g3m: tensor<1024xf32>, %s3b2bt3m: tensor<1024xf32>, %s3b3W1m: tensor<256x1024x1x1xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b3W3m: tensor<1024x256x1x1xf32>, %s3b3g3m: tensor<1024xf32>, %s3b3bt3m: tensor<1024xf32>, %s3b4W1m: tensor<256x1024x1x1xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %s3b4W3m: tensor<1024x256x1x1xf32>, %s3b4g3m: tensor<1024xf32>, %s3b4bt3m: tensor<1024xf32>, %s3b5W1m: tensor<256x1024x1x1xf32>, %s3b5g1m: tensor<256xf32>, %s3b5bt1m: tensor<256xf32>, %s3b5W2m: tensor<256x256x3x3xf32>, %s3b5g2m: tensor<256xf32>, %s3b5bt2m: tensor<256xf32>, %s3b5W3m: tensor<1024x256x1x1xf32>, %s3b5g3m: tensor<1024xf32>, %s3b5bt3m: tensor<1024xf32>, %s4b0W1m: tensor<512x1024x1x1xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b0W3m: tensor<2048x512x1x1xf32>, %s4b0g3m: tensor<2048xf32>, %s4b0bt3m: tensor<2048xf32>, %s4b0Wpm: tensor<2048x1024x1x1xf32>, %s4b0gpm: tensor<2048xf32>, %s4b0btpm: tensor<2048xf32>, %s4b1W1m: tensor<512x2048x1x1xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %s4b1W3m: tensor<2048x512x1x1xf32>, %s4b1g3m: tensor<2048xf32>, %s4b1bt3m: tensor<2048xf32>, %s4b2W1m: tensor<512x2048x1x1xf32>, %s4b2g1m: tensor<512xf32>, %s4b2bt1m: tensor<512xf32>, %s4b2W2m: tensor<512x512x3x3xf32>, %s4b2g2m: tensor<512xf32>, %s4b2bt2m: tensor<512xf32>, %s4b2W3m: tensor<2048x512x1x1xf32>, %s4b2g3m: tensor<2048xf32>, %s4b2bt3m: tensor<2048xf32>, %Wdm: tensor<2048x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x1x1xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b0W3v: tensor<256x64x1x1xf32>, %s1b0g3v: tensor<256xf32>, %s1b0bt3v: tensor<256xf32>, %s1b0Wpv: tensor<256x64x1x1xf32>, %s1b0gpv: tensor<256xf32>, %s1b0btpv: tensor<256xf32>, %s1b1W1v: tensor<64x256x1x1xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b1W3v: tensor<256x64x1x1xf32>, %s1b1g3v: tensor<256xf32>, %s1b1bt3v: tensor<256xf32>, %s1b2W1v: tensor<64x256x1x1xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %s1b2W3v: tensor<256x64x1x1xf32>, %s1b2g3v: tensor<256xf32>, %s1b2bt3v: tensor<256xf32>, %s2b0W1v: tensor<128x256x1x1xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b0W3v: tensor<512x128x1x1xf32>, %s2b0g3v: tensor<512xf32>, %s2b0bt3v: tensor<512xf32>, %s2b0Wpv: tensor<512x256x1x1xf32>, %s2b0gpv: tensor<512xf32>, %s2b0btpv: tensor<512xf32>, %s2b1W1v: tensor<128x512x1x1xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b1W3v: tensor<512x128x1x1xf32>, %s2b1g3v: tensor<512xf32>, %s2b1bt3v: tensor<512xf32>, %s2b2W1v: tensor<128x512x1x1xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %s2b2W3v: tensor<512x128x1x1xf32>, %s2b2g3v: tensor<512xf32>, %s2b2bt3v: tensor<512xf32>, %s2b3W1v: tensor<128x512x1x1xf32>, %s2b3g1v: tensor<128xf32>, %s2b3bt1v: tensor<128xf32>, %s2b3W2v: tensor<128x128x3x3xf32>, %s2b3g2v: tensor<128xf32>, %s2b3bt2v: tensor<128xf32>, %s2b3W3v: tensor<512x128x1x1xf32>, %s2b3g3v: tensor<512xf32>, %s2b3bt3v: tensor<512xf32>, %s3b0W1v: tensor<256x512x1x1xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b0W3v: tensor<1024x256x1x1xf32>, %s3b0g3v: tensor<1024xf32>, %s3b0bt3v: tensor<1024xf32>, %s3b0Wpv: tensor<1024x512x1x1xf32>, %s3b0gpv: tensor<1024xf32>, %s3b0btpv: tensor<1024xf32>, %s3b1W1v: tensor<256x1024x1x1xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b1W3v: tensor<1024x256x1x1xf32>, %s3b1g3v: tensor<1024xf32>, %s3b1bt3v: tensor<1024xf32>, %s3b2W1v: tensor<256x1024x1x1xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b2W3v: tensor<1024x256x1x1xf32>, %s3b2g3v: tensor<1024xf32>, %s3b2bt3v: tensor<1024xf32>, %s3b3W1v: tensor<256x1024x1x1xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b3W3v: tensor<1024x256x1x1xf32>, %s3b3g3v: tensor<1024xf32>, %s3b3bt3v: tensor<1024xf32>, %s3b4W1v: tensor<256x1024x1x1xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %s3b4W3v: tensor<1024x256x1x1xf32>, %s3b4g3v: tensor<1024xf32>, %s3b4bt3v: tensor<1024xf32>, %s3b5W1v: tensor<256x1024x1x1xf32>, %s3b5g1v: tensor<256xf32>, %s3b5bt1v: tensor<256xf32>, %s3b5W2v: tensor<256x256x3x3xf32>, %s3b5g2v: tensor<256xf32>, %s3b5bt2v: tensor<256xf32>, %s3b5W3v: tensor<1024x256x1x1xf32>, %s3b5g3v: tensor<1024xf32>, %s3b5bt3v: tensor<1024xf32>, %s4b0W1v: tensor<512x1024x1x1xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b0W3v: tensor<2048x512x1x1xf32>, %s4b0g3v: tensor<2048xf32>, %s4b0bt3v: tensor<2048xf32>, %s4b0Wpv: tensor<2048x1024x1x1xf32>, %s4b0gpv: tensor<2048xf32>, %s4b0btpv: tensor<2048xf32>, %s4b1W1v: tensor<512x2048x1x1xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %s4b1W3v: tensor<2048x512x1x1xf32>, %s4b1g3v: tensor<2048xf32>, %s4b1bt3v: tensor<2048xf32>, %s4b2W1v: tensor<512x2048x1x1xf32>, %s4b2g1v: tensor<512xf32>, %s4b2bt1v: tensor<512xf32>, %s4b2W2v: tensor<512x512x3x3xf32>, %s4b2g2v: tensor<512xf32>, %s4b2bt2v: tensor<512xf32>, %s4b2W3v: tensor<2048x512x1x1xf32>, %s4b2g3v: tensor<2048xf32>, %s4b2bt3v: tensor<2048xf32>, %Wdv: tensor<2048x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b0n3mui: tensor<256xf32>, %s1b0n3vari: tensor<256xf32>, %s1b0npmui: tensor<256xf32>, %s1b0npvari: tensor<256xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b1n3mui: tensor<256xf32>, %s1b1n3vari: tensor<256xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %s1b2n3mui: tensor<256xf32>, %s1b2n3vari: tensor<256xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b0n3mui: tensor<512xf32>, %s2b0n3vari: tensor<512xf32>, %s2b0npmui: tensor<512xf32>, %s2b0npvari: tensor<512xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b1n3mui: tensor<512xf32>, %s2b1n3vari: tensor<512xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %s2b2n3mui: tensor<512xf32>, %s2b2n3vari: tensor<512xf32>, %s2b3n1mui: tensor<128xf32>, %s2b3n1vari: tensor<128xf32>, %s2b3n2mui: tensor<128xf32>, %s2b3n2vari: tensor<128xf32>, %s2b3n3mui: tensor<512xf32>, %s2b3n3vari: tensor<512xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b0n3mui: tensor<1024xf32>, %s3b0n3vari: tensor<1024xf32>, %s3b0npmui: tensor<1024xf32>, %s3b0npvari: tensor<1024xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b1n3mui: tensor<1024xf32>, %s3b1n3vari: tensor<1024xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b2n3mui: tensor<1024xf32>, %s3b2n3vari: tensor<1024xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b3n3mui: tensor<1024xf32>, %s3b3n3vari: tensor<1024xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %s3b4n3mui: tensor<1024xf32>, %s3b4n3vari: tensor<1024xf32>, %s3b5n1mui: tensor<256xf32>, %s3b5n1vari: tensor<256xf32>, %s3b5n2mui: tensor<256xf32>, %s3b5n2vari: tensor<256xf32>, %s3b5n3mui: tensor<1024xf32>, %s3b5n3vari: tensor<1024xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b0n3mui: tensor<2048xf32>, %s4b0n3vari: tensor<2048xf32>, %s4b0npmui: tensor<2048xf32>, %s4b0npvari: tensor<2048xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %s4b1n3mui: tensor<2048xf32>, %s4b1n3vari: tensor<2048xf32>, %s4b2n1mui: tensor<512xf32>, %s4b2n1vari: tensor<512xf32>, %s4b2n2mui: tensor<512xf32>, %s4b2n2vari: tensor<512xf32>, %s4b2n3mui: tensor<2048xf32>, %s4b2n3vari: tensor<2048xf32>, %onehot: tensor<256x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>) {
    // ── ResNet-50 bottleneck batch-BN heavy-ball momentum + coupled L2 train step: every op is pretty(verified AST node) except the constants, the input passthroughs and the marked report-only %loss ──
    // The conv biases are not in the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb2048 = stablehlo.constant dense<0.0> : tensor<2048xf32>
    %v0 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<256x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<256x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v8 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v9 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v10 = stablehlo.broadcast_in_dim %v9, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v11 = stablehlo.divide %v10, %v7 : tensor<256x64x112x112xf32>
    %v12 = stablehlo.subtract %v5, %v11 : tensor<256x64x112x112xf32>
    %v13 = stablehlo.multiply %v12, %v12 : tensor<256x64x112x112xf32>
    %v14 = stablehlo.reduce(%v13 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v16 = stablehlo.divide %v15, %v7 : tensor<256x64x112x112xf32>
    %v17 = stablehlo.add %v16, %v8 : tensor<256x64x112x112xf32>
    %v18 = stablehlo.rsqrt %v17 : tensor<256x64x112x112xf32>
    %v19 = stablehlo.multiply %v12, %v18 : tensor<256x64x112x112xf32>
    %v20 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v21 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v22 = stablehlo.multiply %v19, %v20 : tensor<256x64x112x112xf32>
    %v23 = stablehlo.add %v22, %v21 : tensor<256x64x112x112xf32>
    %v24 = stablehlo.reshape %v23 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v25 = stablehlo.reshape %v24 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v26 = stablehlo.constant dense<0.0> : tensor<256x64x112x112xf32>
    %v27 = stablehlo.maximum %v25, %v26 : tensor<256x64x112x112xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v29 = stablehlo.reshape %v28 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v30 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v31 = "stablehlo.reduce_window"(%v29, %v30) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v34 = stablehlo.convolution(%v33, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v35 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v36 = stablehlo.add %v34, %v35 : tensor<256x64x56x56xf32>
    %v37 = stablehlo.reshape %v36 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<f32>
    %v40 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v41 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v42 = stablehlo.reduce(%v38 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v43 = stablehlo.broadcast_in_dim %v42, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v44 = stablehlo.divide %v43, %v40 : tensor<256x64x56x56xf32>
    %v45 = stablehlo.subtract %v38, %v44 : tensor<256x64x56x56xf32>
    %v46 = stablehlo.multiply %v45, %v45 : tensor<256x64x56x56xf32>
    %v47 = stablehlo.reduce(%v46 init: %v39) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v48 = stablehlo.broadcast_in_dim %v47, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v49 = stablehlo.divide %v48, %v40 : tensor<256x64x56x56xf32>
    %v50 = stablehlo.add %v49, %v41 : tensor<256x64x56x56xf32>
    %v51 = stablehlo.rsqrt %v50 : tensor<256x64x56x56xf32>
    %v52 = stablehlo.multiply %v45, %v51 : tensor<256x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v54 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v55 = stablehlo.multiply %v52, %v53 : tensor<256x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v54 : tensor<256x64x56x56xf32>
    %v57 = stablehlo.reshape %v56 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v60 = stablehlo.maximum %v58, %v59 : tensor<256x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v63 = stablehlo.convolution(%v62, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v64 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v65 = stablehlo.add %v63, %v64 : tensor<256x64x56x56xf32>
    %v66 = stablehlo.reshape %v65 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v68 = stablehlo.constant dense<0.0> : tensor<f32>
    %v69 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v70 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v71 = stablehlo.reduce(%v67 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v73 = stablehlo.divide %v72, %v69 : tensor<256x64x56x56xf32>
    %v74 = stablehlo.subtract %v67, %v73 : tensor<256x64x56x56xf32>
    %v75 = stablehlo.multiply %v74, %v74 : tensor<256x64x56x56xf32>
    %v76 = stablehlo.reduce(%v75 init: %v68) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v69 : tensor<256x64x56x56xf32>
    %v79 = stablehlo.add %v78, %v70 : tensor<256x64x56x56xf32>
    %v80 = stablehlo.rsqrt %v79 : tensor<256x64x56x56xf32>
    %v81 = stablehlo.multiply %v74, %v80 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v83 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v84 = stablehlo.multiply %v81, %v82 : tensor<256x64x56x56xf32>
    %v85 = stablehlo.add %v84, %v83 : tensor<256x64x56x56xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v87 = stablehlo.reshape %v86 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v88 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v89 = stablehlo.maximum %v87, %v88 : tensor<256x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v92 = stablehlo.convolution(%v91, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v93 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v94 = stablehlo.add %v92, %v93 : tensor<256x256x56x56xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v99 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v100 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v102 = stablehlo.divide %v101, %v98 : tensor<256x256x56x56xf32>
    %v103 = stablehlo.subtract %v96, %v102 : tensor<256x256x56x56xf32>
    %v104 = stablehlo.multiply %v103, %v103 : tensor<256x256x56x56xf32>
    %v105 = stablehlo.reduce(%v104 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v106 = stablehlo.broadcast_in_dim %v105, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v107 = stablehlo.divide %v106, %v98 : tensor<256x256x56x56xf32>
    %v108 = stablehlo.add %v107, %v99 : tensor<256x256x56x56xf32>
    %v109 = stablehlo.rsqrt %v108 : tensor<256x256x56x56xf32>
    %v110 = stablehlo.multiply %v103, %v109 : tensor<256x256x56x56xf32>
    %v111 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v112 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v113 = stablehlo.multiply %v110, %v111 : tensor<256x256x56x56xf32>
    %v114 = stablehlo.add %v113, %v112 : tensor<256x256x56x56xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v116 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v117 = stablehlo.convolution(%v116, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v118 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v119 = stablehlo.add %v117, %v118 : tensor<256x256x56x56xf32>
    %v120 = stablehlo.reshape %v119 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v123 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v124 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v125 = stablehlo.reduce(%v121 init: %v122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v126 = stablehlo.broadcast_in_dim %v125, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v127 = stablehlo.divide %v126, %v123 : tensor<256x256x56x56xf32>
    %v128 = stablehlo.subtract %v121, %v127 : tensor<256x256x56x56xf32>
    %v129 = stablehlo.multiply %v128, %v128 : tensor<256x256x56x56xf32>
    %v130 = stablehlo.reduce(%v129 init: %v122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v131 = stablehlo.broadcast_in_dim %v130, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v132 = stablehlo.divide %v131, %v123 : tensor<256x256x56x56xf32>
    %v133 = stablehlo.add %v132, %v124 : tensor<256x256x56x56xf32>
    %v134 = stablehlo.rsqrt %v133 : tensor<256x256x56x56xf32>
    %v135 = stablehlo.multiply %v128, %v134 : tensor<256x256x56x56xf32>
    %v136 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v137 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v138 = stablehlo.multiply %v135, %v136 : tensor<256x256x56x56xf32>
    %v139 = stablehlo.add %v138, %v137 : tensor<256x256x56x56xf32>
    %v140 = stablehlo.reshape %v139 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v141 = stablehlo.reshape %v115 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v142 = stablehlo.reshape %v140 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v143 = stablehlo.add %v141, %v142 : tensor<256x256x56x56xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v145 = stablehlo.reshape %v144 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v146 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v147 = stablehlo.maximum %v145, %v146 : tensor<256x256x56x56xf32>
    %v148 = stablehlo.reshape %v147 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v150 = stablehlo.convolution(%v149, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v152 = stablehlo.add %v150, %v151 : tensor<256x64x56x56xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v154 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v156 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v157 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v158 = stablehlo.reduce(%v154 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v159 = stablehlo.broadcast_in_dim %v158, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v160 = stablehlo.divide %v159, %v156 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.subtract %v154, %v160 : tensor<256x64x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<256x64x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v164 = stablehlo.broadcast_in_dim %v163, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v165 = stablehlo.divide %v164, %v156 : tensor<256x64x56x56xf32>
    %v166 = stablehlo.add %v165, %v157 : tensor<256x64x56x56xf32>
    %v167 = stablehlo.rsqrt %v166 : tensor<256x64x56x56xf32>
    %v168 = stablehlo.multiply %v161, %v167 : tensor<256x64x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v170 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.multiply %v168, %v169 : tensor<256x64x56x56xf32>
    %v172 = stablehlo.add %v171, %v170 : tensor<256x64x56x56xf32>
    %v173 = stablehlo.reshape %v172 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v174 = stablehlo.reshape %v173 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v175 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v176 = stablehlo.maximum %v174, %v175 : tensor<256x64x56x56xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v178 = stablehlo.reshape %v177 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v179 = stablehlo.convolution(%v178, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v181 = stablehlo.add %v179, %v180 : tensor<256x64x56x56xf32>
    %v182 = stablehlo.reshape %v181 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v185 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v186 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v187 = stablehlo.reduce(%v183 init: %v184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v188 = stablehlo.broadcast_in_dim %v187, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v189 = stablehlo.divide %v188, %v185 : tensor<256x64x56x56xf32>
    %v190 = stablehlo.subtract %v183, %v189 : tensor<256x64x56x56xf32>
    %v191 = stablehlo.multiply %v190, %v190 : tensor<256x64x56x56xf32>
    %v192 = stablehlo.reduce(%v191 init: %v184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v193 = stablehlo.broadcast_in_dim %v192, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v194 = stablehlo.divide %v193, %v185 : tensor<256x64x56x56xf32>
    %v195 = stablehlo.add %v194, %v186 : tensor<256x64x56x56xf32>
    %v196 = stablehlo.rsqrt %v195 : tensor<256x64x56x56xf32>
    %v197 = stablehlo.multiply %v190, %v196 : tensor<256x64x56x56xf32>
    %v198 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v199 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v200 = stablehlo.multiply %v197, %v198 : tensor<256x64x56x56xf32>
    %v201 = stablehlo.add %v200, %v199 : tensor<256x64x56x56xf32>
    %v202 = stablehlo.reshape %v201 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v204 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v205 = stablehlo.maximum %v203, %v204 : tensor<256x64x56x56xf32>
    %v206 = stablehlo.reshape %v205 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v207 = stablehlo.reshape %v206 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v208 = stablehlo.convolution(%v207, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v210 = stablehlo.add %v208, %v209 : tensor<256x256x56x56xf32>
    %v211 = stablehlo.reshape %v210 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v214 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v215 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v216 = stablehlo.reduce(%v212 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v217 = stablehlo.broadcast_in_dim %v216, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v218 = stablehlo.divide %v217, %v214 : tensor<256x256x56x56xf32>
    %v219 = stablehlo.subtract %v212, %v218 : tensor<256x256x56x56xf32>
    %v220 = stablehlo.multiply %v219, %v219 : tensor<256x256x56x56xf32>
    %v221 = stablehlo.reduce(%v220 init: %v213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v222 = stablehlo.broadcast_in_dim %v221, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v223 = stablehlo.divide %v222, %v214 : tensor<256x256x56x56xf32>
    %v224 = stablehlo.add %v223, %v215 : tensor<256x256x56x56xf32>
    %v225 = stablehlo.rsqrt %v224 : tensor<256x256x56x56xf32>
    %v226 = stablehlo.multiply %v219, %v225 : tensor<256x256x56x56xf32>
    %v227 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v229 = stablehlo.multiply %v226, %v227 : tensor<256x256x56x56xf32>
    %v230 = stablehlo.add %v229, %v228 : tensor<256x256x56x56xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v233 = stablehlo.reshape %v148 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<256x256x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v238 = stablehlo.maximum %v236, %v237 : tensor<256x256x56x56xf32>
    %v239 = stablehlo.reshape %v238 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v241 = stablehlo.convolution(%v240, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v242 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v243 = stablehlo.add %v241, %v242 : tensor<256x64x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v245 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v247 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v248 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v249 = stablehlo.reduce(%v245 init: %v246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v250 = stablehlo.broadcast_in_dim %v249, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v251 = stablehlo.divide %v250, %v247 : tensor<256x64x56x56xf32>
    %v252 = stablehlo.subtract %v245, %v251 : tensor<256x64x56x56xf32>
    %v253 = stablehlo.multiply %v252, %v252 : tensor<256x64x56x56xf32>
    %v254 = stablehlo.reduce(%v253 init: %v246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v255 = stablehlo.broadcast_in_dim %v254, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v256 = stablehlo.divide %v255, %v247 : tensor<256x64x56x56xf32>
    %v257 = stablehlo.add %v256, %v248 : tensor<256x64x56x56xf32>
    %v258 = stablehlo.rsqrt %v257 : tensor<256x64x56x56xf32>
    %v259 = stablehlo.multiply %v252, %v258 : tensor<256x64x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v261 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<256x64x56x56xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<256x64x56x56xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v265 = stablehlo.reshape %v264 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v266 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v267 = stablehlo.maximum %v265, %v266 : tensor<256x64x56x56xf32>
    %v268 = stablehlo.reshape %v267 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v269 = stablehlo.reshape %v268 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v270 = stablehlo.convolution(%v269, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v271 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v272 = stablehlo.add %v270, %v271 : tensor<256x64x56x56xf32>
    %v273 = stablehlo.reshape %v272 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v276 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v277 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v278 = stablehlo.reduce(%v274 init: %v275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v279 = stablehlo.broadcast_in_dim %v278, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v280 = stablehlo.divide %v279, %v276 : tensor<256x64x56x56xf32>
    %v281 = stablehlo.subtract %v274, %v280 : tensor<256x64x56x56xf32>
    %v282 = stablehlo.multiply %v281, %v281 : tensor<256x64x56x56xf32>
    %v283 = stablehlo.reduce(%v282 init: %v275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v284 = stablehlo.broadcast_in_dim %v283, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v285 = stablehlo.divide %v284, %v276 : tensor<256x64x56x56xf32>
    %v286 = stablehlo.add %v285, %v277 : tensor<256x64x56x56xf32>
    %v287 = stablehlo.rsqrt %v286 : tensor<256x64x56x56xf32>
    %v288 = stablehlo.multiply %v281, %v287 : tensor<256x64x56x56xf32>
    %v289 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v290 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v291 = stablehlo.multiply %v288, %v289 : tensor<256x64x56x56xf32>
    %v292 = stablehlo.add %v291, %v290 : tensor<256x64x56x56xf32>
    %v293 = stablehlo.reshape %v292 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v295 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v296 = stablehlo.maximum %v294, %v295 : tensor<256x64x56x56xf32>
    %v297 = stablehlo.reshape %v296 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v299 = stablehlo.convolution(%v298, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v300 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v301 = stablehlo.add %v299, %v300 : tensor<256x256x56x56xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v305 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v306 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v307 = stablehlo.reduce(%v303 init: %v304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v308 = stablehlo.broadcast_in_dim %v307, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v309 = stablehlo.divide %v308, %v305 : tensor<256x256x56x56xf32>
    %v310 = stablehlo.subtract %v303, %v309 : tensor<256x256x56x56xf32>
    %v311 = stablehlo.multiply %v310, %v310 : tensor<256x256x56x56xf32>
    %v312 = stablehlo.reduce(%v311 init: %v304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v313 = stablehlo.broadcast_in_dim %v312, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v314 = stablehlo.divide %v313, %v305 : tensor<256x256x56x56xf32>
    %v315 = stablehlo.add %v314, %v306 : tensor<256x256x56x56xf32>
    %v316 = stablehlo.rsqrt %v315 : tensor<256x256x56x56xf32>
    %v317 = stablehlo.multiply %v310, %v316 : tensor<256x256x56x56xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v319 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v320 = stablehlo.multiply %v317, %v318 : tensor<256x256x56x56xf32>
    %v321 = stablehlo.add %v320, %v319 : tensor<256x256x56x56xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v324 = stablehlo.reshape %v239 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v325 = stablehlo.add %v323, %v324 : tensor<256x256x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v328 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v329 = stablehlo.maximum %v327, %v328 : tensor<256x256x56x56xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v332 = stablehlo.convolution(%v331, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<256x128x56x56xf32>
    %v333 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<256x128x56x56xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<256x128x56x56xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<256x128x56x56xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<256x128x56x56xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<256x128x56x56xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<256x128x56x56xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<256x128x56x56xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<256x128x56x56xf32>
    %v351 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v352 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<256x128x56x56xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<256x128x56x56xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v357 = stablehlo.constant dense<0.0> : tensor<256x128x56x56xf32>
    %v358 = stablehlo.maximum %v356, %v357 : tensor<256x128x56x56xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v361 = stablehlo.convolution(%v360, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v362 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<256x128x28x28xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v366 = stablehlo.constant dense<0.0> : tensor<f32>
    %v367 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v368 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v369 = stablehlo.reduce(%v365 init: %v366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v370 = stablehlo.broadcast_in_dim %v369, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v371 = stablehlo.divide %v370, %v367 : tensor<256x128x28x28xf32>
    %v372 = stablehlo.subtract %v365, %v371 : tensor<256x128x28x28xf32>
    %v373 = stablehlo.multiply %v372, %v372 : tensor<256x128x28x28xf32>
    %v374 = stablehlo.reduce(%v373 init: %v366) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v375 = stablehlo.broadcast_in_dim %v374, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v376 = stablehlo.divide %v375, %v367 : tensor<256x128x28x28xf32>
    %v377 = stablehlo.add %v376, %v368 : tensor<256x128x28x28xf32>
    %v378 = stablehlo.rsqrt %v377 : tensor<256x128x28x28xf32>
    %v379 = stablehlo.multiply %v372, %v378 : tensor<256x128x28x28xf32>
    %v380 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v381 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v382 = stablehlo.multiply %v379, %v380 : tensor<256x128x28x28xf32>
    %v383 = stablehlo.add %v382, %v381 : tensor<256x128x28x28xf32>
    %v384 = stablehlo.reshape %v383 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v386 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v387 = stablehlo.maximum %v385, %v386 : tensor<256x128x28x28xf32>
    %v388 = stablehlo.reshape %v387 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v389 = stablehlo.reshape %v388 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v390 = stablehlo.convolution(%v389, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v392 = stablehlo.add %v390, %v391 : tensor<256x512x28x28xf32>
    %v393 = stablehlo.reshape %v392 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v396 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v397 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v398 = stablehlo.reduce(%v394 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v399 = stablehlo.broadcast_in_dim %v398, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v400 = stablehlo.divide %v399, %v396 : tensor<256x512x28x28xf32>
    %v401 = stablehlo.subtract %v394, %v400 : tensor<256x512x28x28xf32>
    %v402 = stablehlo.multiply %v401, %v401 : tensor<256x512x28x28xf32>
    %v403 = stablehlo.reduce(%v402 init: %v395) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v404 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v405 = stablehlo.divide %v404, %v396 : tensor<256x512x28x28xf32>
    %v406 = stablehlo.add %v405, %v397 : tensor<256x512x28x28xf32>
    %v407 = stablehlo.rsqrt %v406 : tensor<256x512x28x28xf32>
    %v408 = stablehlo.multiply %v401, %v407 : tensor<256x512x28x28xf32>
    %v409 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v410 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v411 = stablehlo.multiply %v408, %v409 : tensor<256x512x28x28xf32>
    %v412 = stablehlo.add %v411, %v410 : tensor<256x512x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v414 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v415 = stablehlo.convolution(%v414, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v416 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v417 = stablehlo.add %v415, %v416 : tensor<256x512x28x28xf32>
    %v418 = stablehlo.reshape %v417 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v421 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v422 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v423 = stablehlo.reduce(%v419 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v425 = stablehlo.divide %v424, %v421 : tensor<256x512x28x28xf32>
    %v426 = stablehlo.subtract %v419, %v425 : tensor<256x512x28x28xf32>
    %v427 = stablehlo.multiply %v426, %v426 : tensor<256x512x28x28xf32>
    %v428 = stablehlo.reduce(%v427 init: %v420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v429 = stablehlo.broadcast_in_dim %v428, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v430 = stablehlo.divide %v429, %v421 : tensor<256x512x28x28xf32>
    %v431 = stablehlo.add %v430, %v422 : tensor<256x512x28x28xf32>
    %v432 = stablehlo.rsqrt %v431 : tensor<256x512x28x28xf32>
    %v433 = stablehlo.multiply %v426, %v432 : tensor<256x512x28x28xf32>
    %v434 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v435 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v436 = stablehlo.multiply %v433, %v434 : tensor<256x512x28x28xf32>
    %v437 = stablehlo.add %v436, %v435 : tensor<256x512x28x28xf32>
    %v438 = stablehlo.reshape %v437 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v439 = stablehlo.reshape %v413 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v440 = stablehlo.reshape %v438 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v441 = stablehlo.add %v439, %v440 : tensor<256x512x28x28xf32>
    %v442 = stablehlo.reshape %v441 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v443 = stablehlo.reshape %v442 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v444 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v445 = stablehlo.maximum %v443, %v444 : tensor<256x512x28x28xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v447 = stablehlo.reshape %v446 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v448 = stablehlo.convolution(%v447, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v449 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v450 = stablehlo.add %v448, %v449 : tensor<256x128x28x28xf32>
    %v451 = stablehlo.reshape %v450 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v452 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v454 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v455 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v456 = stablehlo.reduce(%v452 init: %v453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v457 = stablehlo.broadcast_in_dim %v456, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v458 = stablehlo.divide %v457, %v454 : tensor<256x128x28x28xf32>
    %v459 = stablehlo.subtract %v452, %v458 : tensor<256x128x28x28xf32>
    %v460 = stablehlo.multiply %v459, %v459 : tensor<256x128x28x28xf32>
    %v461 = stablehlo.reduce(%v460 init: %v453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v462 = stablehlo.broadcast_in_dim %v461, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v463 = stablehlo.divide %v462, %v454 : tensor<256x128x28x28xf32>
    %v464 = stablehlo.add %v463, %v455 : tensor<256x128x28x28xf32>
    %v465 = stablehlo.rsqrt %v464 : tensor<256x128x28x28xf32>
    %v466 = stablehlo.multiply %v459, %v465 : tensor<256x128x28x28xf32>
    %v467 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v468 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v469 = stablehlo.multiply %v466, %v467 : tensor<256x128x28x28xf32>
    %v470 = stablehlo.add %v469, %v468 : tensor<256x128x28x28xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v472 = stablehlo.reshape %v471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v473 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v474 = stablehlo.maximum %v472, %v473 : tensor<256x128x28x28xf32>
    %v475 = stablehlo.reshape %v474 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v476 = stablehlo.reshape %v475 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v477 = stablehlo.convolution(%v476, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v478 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v479 = stablehlo.add %v477, %v478 : tensor<256x128x28x28xf32>
    %v480 = stablehlo.reshape %v479 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v483 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v484 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v485 = stablehlo.reduce(%v481 init: %v482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v486 = stablehlo.broadcast_in_dim %v485, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v487 = stablehlo.divide %v486, %v483 : tensor<256x128x28x28xf32>
    %v488 = stablehlo.subtract %v481, %v487 : tensor<256x128x28x28xf32>
    %v489 = stablehlo.multiply %v488, %v488 : tensor<256x128x28x28xf32>
    %v490 = stablehlo.reduce(%v489 init: %v482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v491 = stablehlo.broadcast_in_dim %v490, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v492 = stablehlo.divide %v491, %v483 : tensor<256x128x28x28xf32>
    %v493 = stablehlo.add %v492, %v484 : tensor<256x128x28x28xf32>
    %v494 = stablehlo.rsqrt %v493 : tensor<256x128x28x28xf32>
    %v495 = stablehlo.multiply %v488, %v494 : tensor<256x128x28x28xf32>
    %v496 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v497 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v498 = stablehlo.multiply %v495, %v496 : tensor<256x128x28x28xf32>
    %v499 = stablehlo.add %v498, %v497 : tensor<256x128x28x28xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v501 = stablehlo.reshape %v500 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v502 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v503 = stablehlo.maximum %v501, %v502 : tensor<256x128x28x28xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v506 = stablehlo.convolution(%v505, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v508 = stablehlo.add %v506, %v507 : tensor<256x512x28x28xf32>
    %v509 = stablehlo.reshape %v508 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v510 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v512 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v513 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v514 = stablehlo.reduce(%v510 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v515 = stablehlo.broadcast_in_dim %v514, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v516 = stablehlo.divide %v515, %v512 : tensor<256x512x28x28xf32>
    %v517 = stablehlo.subtract %v510, %v516 : tensor<256x512x28x28xf32>
    %v518 = stablehlo.multiply %v517, %v517 : tensor<256x512x28x28xf32>
    %v519 = stablehlo.reduce(%v518 init: %v511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v520 = stablehlo.broadcast_in_dim %v519, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v521 = stablehlo.divide %v520, %v512 : tensor<256x512x28x28xf32>
    %v522 = stablehlo.add %v521, %v513 : tensor<256x512x28x28xf32>
    %v523 = stablehlo.rsqrt %v522 : tensor<256x512x28x28xf32>
    %v524 = stablehlo.multiply %v517, %v523 : tensor<256x512x28x28xf32>
    %v525 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v526 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v527 = stablehlo.multiply %v524, %v525 : tensor<256x512x28x28xf32>
    %v528 = stablehlo.add %v527, %v526 : tensor<256x512x28x28xf32>
    %v529 = stablehlo.reshape %v528 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v530 = stablehlo.reshape %v529 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v531 = stablehlo.reshape %v446 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<256x512x28x28xf32>
    %v533 = stablehlo.reshape %v532 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v534 = stablehlo.reshape %v533 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v535 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v536 = stablehlo.maximum %v534, %v535 : tensor<256x512x28x28xf32>
    %v537 = stablehlo.reshape %v536 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v538 = stablehlo.reshape %v537 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v539 = stablehlo.convolution(%v538, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v540 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v541 = stablehlo.add %v539, %v540 : tensor<256x128x28x28xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v545 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v546 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v547 = stablehlo.reduce(%v543 init: %v544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v548 = stablehlo.broadcast_in_dim %v547, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v549 = stablehlo.divide %v548, %v545 : tensor<256x128x28x28xf32>
    %v550 = stablehlo.subtract %v543, %v549 : tensor<256x128x28x28xf32>
    %v551 = stablehlo.multiply %v550, %v550 : tensor<256x128x28x28xf32>
    %v552 = stablehlo.reduce(%v551 init: %v544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v553 = stablehlo.broadcast_in_dim %v552, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v554 = stablehlo.divide %v553, %v545 : tensor<256x128x28x28xf32>
    %v555 = stablehlo.add %v554, %v546 : tensor<256x128x28x28xf32>
    %v556 = stablehlo.rsqrt %v555 : tensor<256x128x28x28xf32>
    %v557 = stablehlo.multiply %v550, %v556 : tensor<256x128x28x28xf32>
    %v558 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v559 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v560 = stablehlo.multiply %v557, %v558 : tensor<256x128x28x28xf32>
    %v561 = stablehlo.add %v560, %v559 : tensor<256x128x28x28xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v564 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v565 = stablehlo.maximum %v563, %v564 : tensor<256x128x28x28xf32>
    %v566 = stablehlo.reshape %v565 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v567 = stablehlo.reshape %v566 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v568 = stablehlo.convolution(%v567, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v569 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v570 = stablehlo.add %v568, %v569 : tensor<256x128x28x28xf32>
    %v571 = stablehlo.reshape %v570 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v573 = stablehlo.constant dense<0.0> : tensor<f32>
    %v574 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v575 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v576 = stablehlo.reduce(%v572 init: %v573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v577 = stablehlo.broadcast_in_dim %v576, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v578 = stablehlo.divide %v577, %v574 : tensor<256x128x28x28xf32>
    %v579 = stablehlo.subtract %v572, %v578 : tensor<256x128x28x28xf32>
    %v580 = stablehlo.multiply %v579, %v579 : tensor<256x128x28x28xf32>
    %v581 = stablehlo.reduce(%v580 init: %v573) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v582 = stablehlo.broadcast_in_dim %v581, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v583 = stablehlo.divide %v582, %v574 : tensor<256x128x28x28xf32>
    %v584 = stablehlo.add %v583, %v575 : tensor<256x128x28x28xf32>
    %v585 = stablehlo.rsqrt %v584 : tensor<256x128x28x28xf32>
    %v586 = stablehlo.multiply %v579, %v585 : tensor<256x128x28x28xf32>
    %v587 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v588 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v589 = stablehlo.multiply %v586, %v587 : tensor<256x128x28x28xf32>
    %v590 = stablehlo.add %v589, %v588 : tensor<256x128x28x28xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v594 = stablehlo.maximum %v592, %v593 : tensor<256x128x28x28xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v597 = stablehlo.convolution(%v596, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v598 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v599 = stablehlo.add %v597, %v598 : tensor<256x512x28x28xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v601 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v603 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v604 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v605 = stablehlo.reduce(%v601 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v606 = stablehlo.broadcast_in_dim %v605, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v607 = stablehlo.divide %v606, %v603 : tensor<256x512x28x28xf32>
    %v608 = stablehlo.subtract %v601, %v607 : tensor<256x512x28x28xf32>
    %v609 = stablehlo.multiply %v608, %v608 : tensor<256x512x28x28xf32>
    %v610 = stablehlo.reduce(%v609 init: %v602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v611 = stablehlo.broadcast_in_dim %v610, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v612 = stablehlo.divide %v611, %v603 : tensor<256x512x28x28xf32>
    %v613 = stablehlo.add %v612, %v604 : tensor<256x512x28x28xf32>
    %v614 = stablehlo.rsqrt %v613 : tensor<256x512x28x28xf32>
    %v615 = stablehlo.multiply %v608, %v614 : tensor<256x512x28x28xf32>
    %v616 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v617 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v618 = stablehlo.multiply %v615, %v616 : tensor<256x512x28x28xf32>
    %v619 = stablehlo.add %v618, %v617 : tensor<256x512x28x28xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v621 = stablehlo.reshape %v620 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v622 = stablehlo.reshape %v537 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v623 = stablehlo.add %v621, %v622 : tensor<256x512x28x28xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v626 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v627 = stablehlo.maximum %v625, %v626 : tensor<256x512x28x28xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v630 = stablehlo.convolution(%v629, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v631 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<256x128x28x28xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v636 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v637 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v638 = stablehlo.reduce(%v634 init: %v635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v639 = stablehlo.broadcast_in_dim %v638, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v640 = stablehlo.divide %v639, %v636 : tensor<256x128x28x28xf32>
    %v641 = stablehlo.subtract %v634, %v640 : tensor<256x128x28x28xf32>
    %v642 = stablehlo.multiply %v641, %v641 : tensor<256x128x28x28xf32>
    %v643 = stablehlo.reduce(%v642 init: %v635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v644 = stablehlo.broadcast_in_dim %v643, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v645 = stablehlo.divide %v644, %v636 : tensor<256x128x28x28xf32>
    %v646 = stablehlo.add %v645, %v637 : tensor<256x128x28x28xf32>
    %v647 = stablehlo.rsqrt %v646 : tensor<256x128x28x28xf32>
    %v648 = stablehlo.multiply %v641, %v647 : tensor<256x128x28x28xf32>
    %v649 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v650 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v651 = stablehlo.multiply %v648, %v649 : tensor<256x128x28x28xf32>
    %v652 = stablehlo.add %v651, %v650 : tensor<256x128x28x28xf32>
    %v653 = stablehlo.reshape %v652 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v654 = stablehlo.reshape %v653 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v655 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v656 = stablehlo.maximum %v654, %v655 : tensor<256x128x28x28xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v658 = stablehlo.reshape %v657 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v659 = stablehlo.convolution(%v658, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v660 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v661 = stablehlo.add %v659, %v660 : tensor<256x128x28x28xf32>
    %v662 = stablehlo.reshape %v661 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v665 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v666 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v667 = stablehlo.reduce(%v663 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v668 = stablehlo.broadcast_in_dim %v667, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v669 = stablehlo.divide %v668, %v665 : tensor<256x128x28x28xf32>
    %v670 = stablehlo.subtract %v663, %v669 : tensor<256x128x28x28xf32>
    %v671 = stablehlo.multiply %v670, %v670 : tensor<256x128x28x28xf32>
    %v672 = stablehlo.reduce(%v671 init: %v664) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v673 = stablehlo.broadcast_in_dim %v672, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v674 = stablehlo.divide %v673, %v665 : tensor<256x128x28x28xf32>
    %v675 = stablehlo.add %v674, %v666 : tensor<256x128x28x28xf32>
    %v676 = stablehlo.rsqrt %v675 : tensor<256x128x28x28xf32>
    %v677 = stablehlo.multiply %v670, %v676 : tensor<256x128x28x28xf32>
    %v678 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v679 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v680 = stablehlo.multiply %v677, %v678 : tensor<256x128x28x28xf32>
    %v681 = stablehlo.add %v680, %v679 : tensor<256x128x28x28xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v684 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v685 = stablehlo.maximum %v683, %v684 : tensor<256x128x28x28xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v688 = stablehlo.convolution(%v687, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v689 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v690 = stablehlo.add %v688, %v689 : tensor<256x512x28x28xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v694 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v695 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v696 = stablehlo.reduce(%v692 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v698 = stablehlo.divide %v697, %v694 : tensor<256x512x28x28xf32>
    %v699 = stablehlo.subtract %v692, %v698 : tensor<256x512x28x28xf32>
    %v700 = stablehlo.multiply %v699, %v699 : tensor<256x512x28x28xf32>
    %v701 = stablehlo.reduce(%v700 init: %v693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v702 = stablehlo.broadcast_in_dim %v701, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v703 = stablehlo.divide %v702, %v694 : tensor<256x512x28x28xf32>
    %v704 = stablehlo.add %v703, %v695 : tensor<256x512x28x28xf32>
    %v705 = stablehlo.rsqrt %v704 : tensor<256x512x28x28xf32>
    %v706 = stablehlo.multiply %v699, %v705 : tensor<256x512x28x28xf32>
    %v707 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v708 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v709 = stablehlo.multiply %v706, %v707 : tensor<256x512x28x28xf32>
    %v710 = stablehlo.add %v709, %v708 : tensor<256x512x28x28xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v712 = stablehlo.reshape %v711 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v713 = stablehlo.reshape %v628 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<256x512x28x28xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v718 = stablehlo.maximum %v716, %v717 : tensor<256x512x28x28xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v721 = stablehlo.convolution(%v720, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x28x28xf32>
    %v722 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v723 = stablehlo.add %v721, %v722 : tensor<256x256x28x28xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v727 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v728 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v729 = stablehlo.reduce(%v725 init: %v726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v730 = stablehlo.broadcast_in_dim %v729, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v731 = stablehlo.divide %v730, %v727 : tensor<256x256x28x28xf32>
    %v732 = stablehlo.subtract %v725, %v731 : tensor<256x256x28x28xf32>
    %v733 = stablehlo.multiply %v732, %v732 : tensor<256x256x28x28xf32>
    %v734 = stablehlo.reduce(%v733 init: %v726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v735 = stablehlo.broadcast_in_dim %v734, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v736 = stablehlo.divide %v735, %v727 : tensor<256x256x28x28xf32>
    %v737 = stablehlo.add %v736, %v728 : tensor<256x256x28x28xf32>
    %v738 = stablehlo.rsqrt %v737 : tensor<256x256x28x28xf32>
    %v739 = stablehlo.multiply %v732, %v738 : tensor<256x256x28x28xf32>
    %v740 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v741 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v742 = stablehlo.multiply %v739, %v740 : tensor<256x256x28x28xf32>
    %v743 = stablehlo.add %v742, %v741 : tensor<256x256x28x28xf32>
    %v744 = stablehlo.reshape %v743 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v745 = stablehlo.reshape %v744 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v746 = stablehlo.constant dense<0.0> : tensor<256x256x28x28xf32>
    %v747 = stablehlo.maximum %v745, %v746 : tensor<256x256x28x28xf32>
    %v748 = stablehlo.reshape %v747 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v750 = stablehlo.convolution(%v749, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v751 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<256x256x14x14xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v756 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v757 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v758 = stablehlo.reduce(%v754 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v759 = stablehlo.broadcast_in_dim %v758, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v760 = stablehlo.divide %v759, %v756 : tensor<256x256x14x14xf32>
    %v761 = stablehlo.subtract %v754, %v760 : tensor<256x256x14x14xf32>
    %v762 = stablehlo.multiply %v761, %v761 : tensor<256x256x14x14xf32>
    %v763 = stablehlo.reduce(%v762 init: %v755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v764 = stablehlo.broadcast_in_dim %v763, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v765 = stablehlo.divide %v764, %v756 : tensor<256x256x14x14xf32>
    %v766 = stablehlo.add %v765, %v757 : tensor<256x256x14x14xf32>
    %v767 = stablehlo.rsqrt %v766 : tensor<256x256x14x14xf32>
    %v768 = stablehlo.multiply %v761, %v767 : tensor<256x256x14x14xf32>
    %v769 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v770 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v771 = stablehlo.multiply %v768, %v769 : tensor<256x256x14x14xf32>
    %v772 = stablehlo.add %v771, %v770 : tensor<256x256x14x14xf32>
    %v773 = stablehlo.reshape %v772 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v774 = stablehlo.reshape %v773 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v775 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v776 = stablehlo.maximum %v774, %v775 : tensor<256x256x14x14xf32>
    %v777 = stablehlo.reshape %v776 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v778 = stablehlo.reshape %v777 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v779 = stablehlo.convolution(%v778, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v781 = stablehlo.add %v779, %v780 : tensor<256x1024x14x14xf32>
    %v782 = stablehlo.reshape %v781 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v785 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v786 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v787 = stablehlo.reduce(%v783 init: %v784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v788 = stablehlo.broadcast_in_dim %v787, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v789 = stablehlo.divide %v788, %v785 : tensor<256x1024x14x14xf32>
    %v790 = stablehlo.subtract %v783, %v789 : tensor<256x1024x14x14xf32>
    %v791 = stablehlo.multiply %v790, %v790 : tensor<256x1024x14x14xf32>
    %v792 = stablehlo.reduce(%v791 init: %v784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v793 = stablehlo.broadcast_in_dim %v792, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v794 = stablehlo.divide %v793, %v785 : tensor<256x1024x14x14xf32>
    %v795 = stablehlo.add %v794, %v786 : tensor<256x1024x14x14xf32>
    %v796 = stablehlo.rsqrt %v795 : tensor<256x1024x14x14xf32>
    %v797 = stablehlo.multiply %v790, %v796 : tensor<256x1024x14x14xf32>
    %v798 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v799 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v800 = stablehlo.multiply %v797, %v798 : tensor<256x1024x14x14xf32>
    %v801 = stablehlo.add %v800, %v799 : tensor<256x1024x14x14xf32>
    %v802 = stablehlo.reshape %v801 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v803 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v804 = stablehlo.convolution(%v803, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v805 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<256x1024x14x14xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v810 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v811 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v812 = stablehlo.reduce(%v808 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v813 = stablehlo.broadcast_in_dim %v812, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v814 = stablehlo.divide %v813, %v810 : tensor<256x1024x14x14xf32>
    %v815 = stablehlo.subtract %v808, %v814 : tensor<256x1024x14x14xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<256x1024x14x14xf32>
    %v817 = stablehlo.reduce(%v816 init: %v809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v818 = stablehlo.broadcast_in_dim %v817, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v819 = stablehlo.divide %v818, %v810 : tensor<256x1024x14x14xf32>
    %v820 = stablehlo.add %v819, %v811 : tensor<256x1024x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<256x1024x14x14xf32>
    %v822 = stablehlo.multiply %v815, %v821 : tensor<256x1024x14x14xf32>
    %v823 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v825 = stablehlo.multiply %v822, %v823 : tensor<256x1024x14x14xf32>
    %v826 = stablehlo.add %v825, %v824 : tensor<256x1024x14x14xf32>
    %v827 = stablehlo.reshape %v826 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v828 = stablehlo.reshape %v802 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v829 = stablehlo.reshape %v827 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<256x1024x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v834 = stablehlo.maximum %v832, %v833 : tensor<256x1024x14x14xf32>
    %v835 = stablehlo.reshape %v834 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v837 = stablehlo.convolution(%v836, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v839 = stablehlo.add %v837, %v838 : tensor<256x256x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v841 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v843 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v844 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v845 = stablehlo.reduce(%v841 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v846 = stablehlo.broadcast_in_dim %v845, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v847 = stablehlo.divide %v846, %v843 : tensor<256x256x14x14xf32>
    %v848 = stablehlo.subtract %v841, %v847 : tensor<256x256x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v848 : tensor<256x256x14x14xf32>
    %v850 = stablehlo.reduce(%v849 init: %v842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v851 = stablehlo.broadcast_in_dim %v850, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v852 = stablehlo.divide %v851, %v843 : tensor<256x256x14x14xf32>
    %v853 = stablehlo.add %v852, %v844 : tensor<256x256x14x14xf32>
    %v854 = stablehlo.rsqrt %v853 : tensor<256x256x14x14xf32>
    %v855 = stablehlo.multiply %v848, %v854 : tensor<256x256x14x14xf32>
    %v856 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v858 = stablehlo.multiply %v855, %v856 : tensor<256x256x14x14xf32>
    %v859 = stablehlo.add %v858, %v857 : tensor<256x256x14x14xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v861 = stablehlo.reshape %v860 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v862 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v863 = stablehlo.maximum %v861, %v862 : tensor<256x256x14x14xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v865 = stablehlo.reshape %v864 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v866 = stablehlo.convolution(%v865, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v868 = stablehlo.add %v866, %v867 : tensor<256x256x14x14xf32>
    %v869 = stablehlo.reshape %v868 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v872 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v873 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v874 = stablehlo.reduce(%v870 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v875 = stablehlo.broadcast_in_dim %v874, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v876 = stablehlo.divide %v875, %v872 : tensor<256x256x14x14xf32>
    %v877 = stablehlo.subtract %v870, %v876 : tensor<256x256x14x14xf32>
    %v878 = stablehlo.multiply %v877, %v877 : tensor<256x256x14x14xf32>
    %v879 = stablehlo.reduce(%v878 init: %v871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v880 = stablehlo.broadcast_in_dim %v879, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v881 = stablehlo.divide %v880, %v872 : tensor<256x256x14x14xf32>
    %v882 = stablehlo.add %v881, %v873 : tensor<256x256x14x14xf32>
    %v883 = stablehlo.rsqrt %v882 : tensor<256x256x14x14xf32>
    %v884 = stablehlo.multiply %v877, %v883 : tensor<256x256x14x14xf32>
    %v885 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v886 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v887 = stablehlo.multiply %v884, %v885 : tensor<256x256x14x14xf32>
    %v888 = stablehlo.add %v887, %v886 : tensor<256x256x14x14xf32>
    %v889 = stablehlo.reshape %v888 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v891 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v892 = stablehlo.maximum %v890, %v891 : tensor<256x256x14x14xf32>
    %v893 = stablehlo.reshape %v892 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v894 = stablehlo.reshape %v893 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v895 = stablehlo.convolution(%v894, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v896 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v897 = stablehlo.add %v895, %v896 : tensor<256x1024x14x14xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v899 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v901 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v902 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v903 = stablehlo.reduce(%v899 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v904 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v905 = stablehlo.divide %v904, %v901 : tensor<256x1024x14x14xf32>
    %v906 = stablehlo.subtract %v899, %v905 : tensor<256x1024x14x14xf32>
    %v907 = stablehlo.multiply %v906, %v906 : tensor<256x1024x14x14xf32>
    %v908 = stablehlo.reduce(%v907 init: %v900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v909 = stablehlo.broadcast_in_dim %v908, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v910 = stablehlo.divide %v909, %v901 : tensor<256x1024x14x14xf32>
    %v911 = stablehlo.add %v910, %v902 : tensor<256x1024x14x14xf32>
    %v912 = stablehlo.rsqrt %v911 : tensor<256x1024x14x14xf32>
    %v913 = stablehlo.multiply %v906, %v912 : tensor<256x1024x14x14xf32>
    %v914 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v915 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v916 = stablehlo.multiply %v913, %v914 : tensor<256x1024x14x14xf32>
    %v917 = stablehlo.add %v916, %v915 : tensor<256x1024x14x14xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v920 = stablehlo.reshape %v835 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v921 = stablehlo.add %v919, %v920 : tensor<256x1024x14x14xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v923 = stablehlo.reshape %v922 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v925 = stablehlo.maximum %v923, %v924 : tensor<256x1024x14x14xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v928 = stablehlo.convolution(%v927, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v929 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v930 = stablehlo.add %v928, %v929 : tensor<256x256x14x14xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v934 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v935 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v936 = stablehlo.reduce(%v932 init: %v933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v937 = stablehlo.broadcast_in_dim %v936, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v938 = stablehlo.divide %v937, %v934 : tensor<256x256x14x14xf32>
    %v939 = stablehlo.subtract %v932, %v938 : tensor<256x256x14x14xf32>
    %v940 = stablehlo.multiply %v939, %v939 : tensor<256x256x14x14xf32>
    %v941 = stablehlo.reduce(%v940 init: %v933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v942 = stablehlo.broadcast_in_dim %v941, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v943 = stablehlo.divide %v942, %v934 : tensor<256x256x14x14xf32>
    %v944 = stablehlo.add %v943, %v935 : tensor<256x256x14x14xf32>
    %v945 = stablehlo.rsqrt %v944 : tensor<256x256x14x14xf32>
    %v946 = stablehlo.multiply %v939, %v945 : tensor<256x256x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v948 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v949 = stablehlo.multiply %v946, %v947 : tensor<256x256x14x14xf32>
    %v950 = stablehlo.add %v949, %v948 : tensor<256x256x14x14xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v953 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v954 = stablehlo.maximum %v952, %v953 : tensor<256x256x14x14xf32>
    %v955 = stablehlo.reshape %v954 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v956 = stablehlo.reshape %v955 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v957 = stablehlo.convolution(%v956, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v958 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v959 = stablehlo.add %v957, %v958 : tensor<256x256x14x14xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v961 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v963 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v964 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v965 = stablehlo.reduce(%v961 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v966 = stablehlo.broadcast_in_dim %v965, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v967 = stablehlo.divide %v966, %v963 : tensor<256x256x14x14xf32>
    %v968 = stablehlo.subtract %v961, %v967 : tensor<256x256x14x14xf32>
    %v969 = stablehlo.multiply %v968, %v968 : tensor<256x256x14x14xf32>
    %v970 = stablehlo.reduce(%v969 init: %v962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v971 = stablehlo.broadcast_in_dim %v970, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v972 = stablehlo.divide %v971, %v963 : tensor<256x256x14x14xf32>
    %v973 = stablehlo.add %v972, %v964 : tensor<256x256x14x14xf32>
    %v974 = stablehlo.rsqrt %v973 : tensor<256x256x14x14xf32>
    %v975 = stablehlo.multiply %v968, %v974 : tensor<256x256x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v977 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v978 = stablehlo.multiply %v975, %v976 : tensor<256x256x14x14xf32>
    %v979 = stablehlo.add %v978, %v977 : tensor<256x256x14x14xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v981 = stablehlo.reshape %v980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v983 = stablehlo.maximum %v981, %v982 : tensor<256x256x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v986 = stablehlo.convolution(%v985, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v987 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v988 = stablehlo.add %v986, %v987 : tensor<256x1024x14x14xf32>
    %v989 = stablehlo.reshape %v988 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v990 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v992 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v993 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v994 = stablehlo.reduce(%v990 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v996 = stablehlo.divide %v995, %v992 : tensor<256x1024x14x14xf32>
    %v997 = stablehlo.subtract %v990, %v996 : tensor<256x1024x14x14xf32>
    %v998 = stablehlo.multiply %v997, %v997 : tensor<256x1024x14x14xf32>
    %v999 = stablehlo.reduce(%v998 init: %v991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1000 = stablehlo.broadcast_in_dim %v999, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1001 = stablehlo.divide %v1000, %v992 : tensor<256x1024x14x14xf32>
    %v1002 = stablehlo.add %v1001, %v993 : tensor<256x1024x14x14xf32>
    %v1003 = stablehlo.rsqrt %v1002 : tensor<256x1024x14x14xf32>
    %v1004 = stablehlo.multiply %v997, %v1003 : tensor<256x1024x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1006 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<256x1024x14x14xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<256x1024x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1011 = stablehlo.reshape %v926 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1012 = stablehlo.add %v1010, %v1011 : tensor<256x1024x14x14xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1014 = stablehlo.reshape %v1013 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1015 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v1016 = stablehlo.maximum %v1014, %v1015 : tensor<256x1024x14x14xf32>
    %v1017 = stablehlo.reshape %v1016 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1018 = stablehlo.reshape %v1017 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1019 = stablehlo.convolution(%v1018, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1021 = stablehlo.add %v1019, %v1020 : tensor<256x256x14x14xf32>
    %v1022 = stablehlo.reshape %v1021 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1023 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1024 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1025 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1026 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1027 = stablehlo.reduce(%v1023 init: %v1024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1028 = stablehlo.broadcast_in_dim %v1027, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1029 = stablehlo.divide %v1028, %v1025 : tensor<256x256x14x14xf32>
    %v1030 = stablehlo.subtract %v1023, %v1029 : tensor<256x256x14x14xf32>
    %v1031 = stablehlo.multiply %v1030, %v1030 : tensor<256x256x14x14xf32>
    %v1032 = stablehlo.reduce(%v1031 init: %v1024) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1033 = stablehlo.broadcast_in_dim %v1032, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1034 = stablehlo.divide %v1033, %v1025 : tensor<256x256x14x14xf32>
    %v1035 = stablehlo.add %v1034, %v1026 : tensor<256x256x14x14xf32>
    %v1036 = stablehlo.rsqrt %v1035 : tensor<256x256x14x14xf32>
    %v1037 = stablehlo.multiply %v1030, %v1036 : tensor<256x256x14x14xf32>
    %v1038 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1039 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1040 = stablehlo.multiply %v1037, %v1038 : tensor<256x256x14x14xf32>
    %v1041 = stablehlo.add %v1040, %v1039 : tensor<256x256x14x14xf32>
    %v1042 = stablehlo.reshape %v1041 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1043 = stablehlo.reshape %v1042 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1044 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1045 = stablehlo.maximum %v1043, %v1044 : tensor<256x256x14x14xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1048 = stablehlo.convolution(%v1047, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1049 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1050 = stablehlo.add %v1048, %v1049 : tensor<256x256x14x14xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1054 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1055 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1056 = stablehlo.reduce(%v1052 init: %v1053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1057 = stablehlo.broadcast_in_dim %v1056, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1058 = stablehlo.divide %v1057, %v1054 : tensor<256x256x14x14xf32>
    %v1059 = stablehlo.subtract %v1052, %v1058 : tensor<256x256x14x14xf32>
    %v1060 = stablehlo.multiply %v1059, %v1059 : tensor<256x256x14x14xf32>
    %v1061 = stablehlo.reduce(%v1060 init: %v1053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1062 = stablehlo.broadcast_in_dim %v1061, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1063 = stablehlo.divide %v1062, %v1054 : tensor<256x256x14x14xf32>
    %v1064 = stablehlo.add %v1063, %v1055 : tensor<256x256x14x14xf32>
    %v1065 = stablehlo.rsqrt %v1064 : tensor<256x256x14x14xf32>
    %v1066 = stablehlo.multiply %v1059, %v1065 : tensor<256x256x14x14xf32>
    %v1067 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1068 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1069 = stablehlo.multiply %v1066, %v1067 : tensor<256x256x14x14xf32>
    %v1070 = stablehlo.add %v1069, %v1068 : tensor<256x256x14x14xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1072 = stablehlo.reshape %v1071 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1073 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1074 = stablehlo.maximum %v1072, %v1073 : tensor<256x256x14x14xf32>
    %v1075 = stablehlo.reshape %v1074 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1076 = stablehlo.reshape %v1075 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1077 = stablehlo.convolution(%v1076, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1078 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1079 = stablehlo.add %v1077, %v1078 : tensor<256x1024x14x14xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1081 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1083 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v1084 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v1085 = stablehlo.reduce(%v1081 init: %v1082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1086 = stablehlo.broadcast_in_dim %v1085, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1087 = stablehlo.divide %v1086, %v1083 : tensor<256x1024x14x14xf32>
    %v1088 = stablehlo.subtract %v1081, %v1087 : tensor<256x1024x14x14xf32>
    %v1089 = stablehlo.multiply %v1088, %v1088 : tensor<256x1024x14x14xf32>
    %v1090 = stablehlo.reduce(%v1089 init: %v1082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1091 = stablehlo.broadcast_in_dim %v1090, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1092 = stablehlo.divide %v1091, %v1083 : tensor<256x1024x14x14xf32>
    %v1093 = stablehlo.add %v1092, %v1084 : tensor<256x1024x14x14xf32>
    %v1094 = stablehlo.rsqrt %v1093 : tensor<256x1024x14x14xf32>
    %v1095 = stablehlo.multiply %v1088, %v1094 : tensor<256x1024x14x14xf32>
    %v1096 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1097 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1098 = stablehlo.multiply %v1095, %v1096 : tensor<256x1024x14x14xf32>
    %v1099 = stablehlo.add %v1098, %v1097 : tensor<256x1024x14x14xf32>
    %v1100 = stablehlo.reshape %v1099 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1101 = stablehlo.reshape %v1100 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1102 = stablehlo.reshape %v1017 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1103 = stablehlo.add %v1101, %v1102 : tensor<256x1024x14x14xf32>
    %v1104 = stablehlo.reshape %v1103 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1105 = stablehlo.reshape %v1104 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1106 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v1107 = stablehlo.maximum %v1105, %v1106 : tensor<256x1024x14x14xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1109 = stablehlo.reshape %v1108 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1110 = stablehlo.convolution(%v1109, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1112 = stablehlo.add %v1110, %v1111 : tensor<256x256x14x14xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1116 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1117 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1118 = stablehlo.reduce(%v1114 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1119 = stablehlo.broadcast_in_dim %v1118, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1120 = stablehlo.divide %v1119, %v1116 : tensor<256x256x14x14xf32>
    %v1121 = stablehlo.subtract %v1114, %v1120 : tensor<256x256x14x14xf32>
    %v1122 = stablehlo.multiply %v1121, %v1121 : tensor<256x256x14x14xf32>
    %v1123 = stablehlo.reduce(%v1122 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1124 = stablehlo.broadcast_in_dim %v1123, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1125 = stablehlo.divide %v1124, %v1116 : tensor<256x256x14x14xf32>
    %v1126 = stablehlo.add %v1125, %v1117 : tensor<256x256x14x14xf32>
    %v1127 = stablehlo.rsqrt %v1126 : tensor<256x256x14x14xf32>
    %v1128 = stablehlo.multiply %v1121, %v1127 : tensor<256x256x14x14xf32>
    %v1129 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1130 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1131 = stablehlo.multiply %v1128, %v1129 : tensor<256x256x14x14xf32>
    %v1132 = stablehlo.add %v1131, %v1130 : tensor<256x256x14x14xf32>
    %v1133 = stablehlo.reshape %v1132 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1134 = stablehlo.reshape %v1133 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1135 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1136 = stablehlo.maximum %v1134, %v1135 : tensor<256x256x14x14xf32>
    %v1137 = stablehlo.reshape %v1136 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1138 = stablehlo.reshape %v1137 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1139 = stablehlo.convolution(%v1138, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1140 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1141 = stablehlo.add %v1139, %v1140 : tensor<256x256x14x14xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1143 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1145 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1146 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1147 = stablehlo.reduce(%v1143 init: %v1144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1148 = stablehlo.broadcast_in_dim %v1147, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1149 = stablehlo.divide %v1148, %v1145 : tensor<256x256x14x14xf32>
    %v1150 = stablehlo.subtract %v1143, %v1149 : tensor<256x256x14x14xf32>
    %v1151 = stablehlo.multiply %v1150, %v1150 : tensor<256x256x14x14xf32>
    %v1152 = stablehlo.reduce(%v1151 init: %v1144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1153 = stablehlo.broadcast_in_dim %v1152, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1154 = stablehlo.divide %v1153, %v1145 : tensor<256x256x14x14xf32>
    %v1155 = stablehlo.add %v1154, %v1146 : tensor<256x256x14x14xf32>
    %v1156 = stablehlo.rsqrt %v1155 : tensor<256x256x14x14xf32>
    %v1157 = stablehlo.multiply %v1150, %v1156 : tensor<256x256x14x14xf32>
    %v1158 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1159 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1160 = stablehlo.multiply %v1157, %v1158 : tensor<256x256x14x14xf32>
    %v1161 = stablehlo.add %v1160, %v1159 : tensor<256x256x14x14xf32>
    %v1162 = stablehlo.reshape %v1161 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1164 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1165 = stablehlo.maximum %v1163, %v1164 : tensor<256x256x14x14xf32>
    %v1166 = stablehlo.reshape %v1165 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1168 = stablehlo.convolution(%v1167, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1169 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1170 = stablehlo.add %v1168, %v1169 : tensor<256x1024x14x14xf32>
    %v1171 = stablehlo.reshape %v1170 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1174 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v1175 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v1176 = stablehlo.reduce(%v1172 init: %v1173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1177 = stablehlo.broadcast_in_dim %v1176, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1178 = stablehlo.divide %v1177, %v1174 : tensor<256x1024x14x14xf32>
    %v1179 = stablehlo.subtract %v1172, %v1178 : tensor<256x1024x14x14xf32>
    %v1180 = stablehlo.multiply %v1179, %v1179 : tensor<256x1024x14x14xf32>
    %v1181 = stablehlo.reduce(%v1180 init: %v1173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1183 = stablehlo.divide %v1182, %v1174 : tensor<256x1024x14x14xf32>
    %v1184 = stablehlo.add %v1183, %v1175 : tensor<256x1024x14x14xf32>
    %v1185 = stablehlo.rsqrt %v1184 : tensor<256x1024x14x14xf32>
    %v1186 = stablehlo.multiply %v1179, %v1185 : tensor<256x1024x14x14xf32>
    %v1187 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1188 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1189 = stablehlo.multiply %v1186, %v1187 : tensor<256x1024x14x14xf32>
    %v1190 = stablehlo.add %v1189, %v1188 : tensor<256x1024x14x14xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1193 = stablehlo.reshape %v1108 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1194 = stablehlo.add %v1192, %v1193 : tensor<256x1024x14x14xf32>
    %v1195 = stablehlo.reshape %v1194 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<256x1024x14x14xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1201 = stablehlo.convolution(%v1200, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1202 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1203 = stablehlo.add %v1201, %v1202 : tensor<256x256x14x14xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1205 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1206 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1207 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1208 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1209 = stablehlo.reduce(%v1205 init: %v1206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1210 = stablehlo.broadcast_in_dim %v1209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1211 = stablehlo.divide %v1210, %v1207 : tensor<256x256x14x14xf32>
    %v1212 = stablehlo.subtract %v1205, %v1211 : tensor<256x256x14x14xf32>
    %v1213 = stablehlo.multiply %v1212, %v1212 : tensor<256x256x14x14xf32>
    %v1214 = stablehlo.reduce(%v1213 init: %v1206) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1215 = stablehlo.broadcast_in_dim %v1214, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1216 = stablehlo.divide %v1215, %v1207 : tensor<256x256x14x14xf32>
    %v1217 = stablehlo.add %v1216, %v1208 : tensor<256x256x14x14xf32>
    %v1218 = stablehlo.rsqrt %v1217 : tensor<256x256x14x14xf32>
    %v1219 = stablehlo.multiply %v1212, %v1218 : tensor<256x256x14x14xf32>
    %v1220 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1221 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1222 = stablehlo.multiply %v1219, %v1220 : tensor<256x256x14x14xf32>
    %v1223 = stablehlo.add %v1222, %v1221 : tensor<256x256x14x14xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1227 = stablehlo.maximum %v1225, %v1226 : tensor<256x256x14x14xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1229 = stablehlo.reshape %v1228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1230 = stablehlo.convolution(%v1229, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1231 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1232 = stablehlo.add %v1230, %v1231 : tensor<256x256x14x14xf32>
    %v1233 = stablehlo.reshape %v1232 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1234 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1236 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1237 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1238 = stablehlo.reduce(%v1234 init: %v1235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1239 = stablehlo.broadcast_in_dim %v1238, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1240 = stablehlo.divide %v1239, %v1236 : tensor<256x256x14x14xf32>
    %v1241 = stablehlo.subtract %v1234, %v1240 : tensor<256x256x14x14xf32>
    %v1242 = stablehlo.multiply %v1241, %v1241 : tensor<256x256x14x14xf32>
    %v1243 = stablehlo.reduce(%v1242 init: %v1235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1244 = stablehlo.broadcast_in_dim %v1243, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1245 = stablehlo.divide %v1244, %v1236 : tensor<256x256x14x14xf32>
    %v1246 = stablehlo.add %v1245, %v1237 : tensor<256x256x14x14xf32>
    %v1247 = stablehlo.rsqrt %v1246 : tensor<256x256x14x14xf32>
    %v1248 = stablehlo.multiply %v1241, %v1247 : tensor<256x256x14x14xf32>
    %v1249 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1250 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1251 = stablehlo.multiply %v1248, %v1249 : tensor<256x256x14x14xf32>
    %v1252 = stablehlo.add %v1251, %v1250 : tensor<256x256x14x14xf32>
    %v1253 = stablehlo.reshape %v1252 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1254 = stablehlo.reshape %v1253 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1255 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v1256 = stablehlo.maximum %v1254, %v1255 : tensor<256x256x14x14xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1259 = stablehlo.convolution(%v1258, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1260 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1261 = stablehlo.add %v1259, %v1260 : tensor<256x1024x14x14xf32>
    %v1262 = stablehlo.reshape %v1261 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1265 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v1266 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v1267 = stablehlo.reduce(%v1263 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1269 = stablehlo.divide %v1268, %v1265 : tensor<256x1024x14x14xf32>
    %v1270 = stablehlo.subtract %v1263, %v1269 : tensor<256x1024x14x14xf32>
    %v1271 = stablehlo.multiply %v1270, %v1270 : tensor<256x1024x14x14xf32>
    %v1272 = stablehlo.reduce(%v1271 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1273 = stablehlo.broadcast_in_dim %v1272, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1274 = stablehlo.divide %v1273, %v1265 : tensor<256x1024x14x14xf32>
    %v1275 = stablehlo.add %v1274, %v1266 : tensor<256x1024x14x14xf32>
    %v1276 = stablehlo.rsqrt %v1275 : tensor<256x1024x14x14xf32>
    %v1277 = stablehlo.multiply %v1270, %v1276 : tensor<256x1024x14x14xf32>
    %v1278 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1279 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1280 = stablehlo.multiply %v1277, %v1278 : tensor<256x1024x14x14xf32>
    %v1281 = stablehlo.add %v1280, %v1279 : tensor<256x1024x14x14xf32>
    %v1282 = stablehlo.reshape %v1281 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1284 = stablehlo.reshape %v1199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1285 = stablehlo.add %v1283, %v1284 : tensor<256x1024x14x14xf32>
    %v1286 = stablehlo.reshape %v1285 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1287 = stablehlo.reshape %v1286 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1288 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v1289 = stablehlo.maximum %v1287, %v1288 : tensor<256x1024x14x14xf32>
    %v1290 = stablehlo.reshape %v1289 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1292 = stablehlo.convolution(%v1291, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x14x14xf32>
    %v1293 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1294 = stablehlo.add %v1292, %v1293 : tensor<256x512x14x14xf32>
    %v1295 = stablehlo.reshape %v1294 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1296 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1298 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v1299 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v1300 = stablehlo.reduce(%v1296 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1301 = stablehlo.broadcast_in_dim %v1300, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1302 = stablehlo.divide %v1301, %v1298 : tensor<256x512x14x14xf32>
    %v1303 = stablehlo.subtract %v1296, %v1302 : tensor<256x512x14x14xf32>
    %v1304 = stablehlo.multiply %v1303, %v1303 : tensor<256x512x14x14xf32>
    %v1305 = stablehlo.reduce(%v1304 init: %v1297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1306 = stablehlo.broadcast_in_dim %v1305, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1307 = stablehlo.divide %v1306, %v1298 : tensor<256x512x14x14xf32>
    %v1308 = stablehlo.add %v1307, %v1299 : tensor<256x512x14x14xf32>
    %v1309 = stablehlo.rsqrt %v1308 : tensor<256x512x14x14xf32>
    %v1310 = stablehlo.multiply %v1303, %v1309 : tensor<256x512x14x14xf32>
    %v1311 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1312 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1313 = stablehlo.multiply %v1310, %v1311 : tensor<256x512x14x14xf32>
    %v1314 = stablehlo.add %v1313, %v1312 : tensor<256x512x14x14xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<256x512x14x14xf32>
    %v1318 = stablehlo.maximum %v1316, %v1317 : tensor<256x512x14x14xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1321 = stablehlo.convolution(%v1320, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1322 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1323 = stablehlo.add %v1321, %v1322 : tensor<256x512x7x7xf32>
    %v1324 = stablehlo.reshape %v1323 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1325 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1326 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1327 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1328 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1329 = stablehlo.reduce(%v1325 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1330 = stablehlo.broadcast_in_dim %v1329, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1331 = stablehlo.divide %v1330, %v1327 : tensor<256x512x7x7xf32>
    %v1332 = stablehlo.subtract %v1325, %v1331 : tensor<256x512x7x7xf32>
    %v1333 = stablehlo.multiply %v1332, %v1332 : tensor<256x512x7x7xf32>
    %v1334 = stablehlo.reduce(%v1333 init: %v1326) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1335 = stablehlo.broadcast_in_dim %v1334, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1336 = stablehlo.divide %v1335, %v1327 : tensor<256x512x7x7xf32>
    %v1337 = stablehlo.add %v1336, %v1328 : tensor<256x512x7x7xf32>
    %v1338 = stablehlo.rsqrt %v1337 : tensor<256x512x7x7xf32>
    %v1339 = stablehlo.multiply %v1332, %v1338 : tensor<256x512x7x7xf32>
    %v1340 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1341 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1339, %v1340 : tensor<256x512x7x7xf32>
    %v1343 = stablehlo.add %v1342, %v1341 : tensor<256x512x7x7xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1345 = stablehlo.reshape %v1344 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1346 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1347 = stablehlo.maximum %v1345, %v1346 : tensor<256x512x7x7xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1350 = stablehlo.convolution(%v1349, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1351 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1352 = stablehlo.add %v1350, %v1351 : tensor<256x2048x7x7xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1357 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1358 = stablehlo.reduce(%v1354 init: %v1355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1359 = stablehlo.broadcast_in_dim %v1358, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1360 = stablehlo.divide %v1359, %v1356 : tensor<256x2048x7x7xf32>
    %v1361 = stablehlo.subtract %v1354, %v1360 : tensor<256x2048x7x7xf32>
    %v1362 = stablehlo.multiply %v1361, %v1361 : tensor<256x2048x7x7xf32>
    %v1363 = stablehlo.reduce(%v1362 init: %v1355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1365 = stablehlo.divide %v1364, %v1356 : tensor<256x2048x7x7xf32>
    %v1366 = stablehlo.add %v1365, %v1357 : tensor<256x2048x7x7xf32>
    %v1367 = stablehlo.rsqrt %v1366 : tensor<256x2048x7x7xf32>
    %v1368 = stablehlo.multiply %v1361, %v1367 : tensor<256x2048x7x7xf32>
    %v1369 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1370 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1371 = stablehlo.multiply %v1368, %v1369 : tensor<256x2048x7x7xf32>
    %v1372 = stablehlo.add %v1371, %v1370 : tensor<256x2048x7x7xf32>
    %v1373 = stablehlo.reshape %v1372 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1374 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1375 = stablehlo.convolution(%v1374, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1376 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<256x2048x7x7xf32>
    %v1378 = stablehlo.reshape %v1377 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1381 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1382 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1383 = stablehlo.reduce(%v1379 init: %v1380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1384 = stablehlo.broadcast_in_dim %v1383, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1385 = stablehlo.divide %v1384, %v1381 : tensor<256x2048x7x7xf32>
    %v1386 = stablehlo.subtract %v1379, %v1385 : tensor<256x2048x7x7xf32>
    %v1387 = stablehlo.multiply %v1386, %v1386 : tensor<256x2048x7x7xf32>
    %v1388 = stablehlo.reduce(%v1387 init: %v1380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1389 = stablehlo.broadcast_in_dim %v1388, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1390 = stablehlo.divide %v1389, %v1381 : tensor<256x2048x7x7xf32>
    %v1391 = stablehlo.add %v1390, %v1382 : tensor<256x2048x7x7xf32>
    %v1392 = stablehlo.rsqrt %v1391 : tensor<256x2048x7x7xf32>
    %v1393 = stablehlo.multiply %v1386, %v1392 : tensor<256x2048x7x7xf32>
    %v1394 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1395 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1396 = stablehlo.multiply %v1393, %v1394 : tensor<256x2048x7x7xf32>
    %v1397 = stablehlo.add %v1396, %v1395 : tensor<256x2048x7x7xf32>
    %v1398 = stablehlo.reshape %v1397 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1399 = stablehlo.reshape %v1373 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1400 = stablehlo.reshape %v1398 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1401 = stablehlo.add %v1399, %v1400 : tensor<256x2048x7x7xf32>
    %v1402 = stablehlo.reshape %v1401 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1403 = stablehlo.reshape %v1402 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1404 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1405 = stablehlo.maximum %v1403, %v1404 : tensor<256x2048x7x7xf32>
    %v1406 = stablehlo.reshape %v1405 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1407 = stablehlo.reshape %v1406 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1408 = stablehlo.convolution(%v1407, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1409 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1410 = stablehlo.add %v1408, %v1409 : tensor<256x512x7x7xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1414 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1415 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1416 = stablehlo.reduce(%v1412 init: %v1413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1417 = stablehlo.broadcast_in_dim %v1416, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1418 = stablehlo.divide %v1417, %v1414 : tensor<256x512x7x7xf32>
    %v1419 = stablehlo.subtract %v1412, %v1418 : tensor<256x512x7x7xf32>
    %v1420 = stablehlo.multiply %v1419, %v1419 : tensor<256x512x7x7xf32>
    %v1421 = stablehlo.reduce(%v1420 init: %v1413) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1422 = stablehlo.broadcast_in_dim %v1421, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1423 = stablehlo.divide %v1422, %v1414 : tensor<256x512x7x7xf32>
    %v1424 = stablehlo.add %v1423, %v1415 : tensor<256x512x7x7xf32>
    %v1425 = stablehlo.rsqrt %v1424 : tensor<256x512x7x7xf32>
    %v1426 = stablehlo.multiply %v1419, %v1425 : tensor<256x512x7x7xf32>
    %v1427 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1428 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1429 = stablehlo.multiply %v1426, %v1427 : tensor<256x512x7x7xf32>
    %v1430 = stablehlo.add %v1429, %v1428 : tensor<256x512x7x7xf32>
    %v1431 = stablehlo.reshape %v1430 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1433 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1434 = stablehlo.maximum %v1432, %v1433 : tensor<256x512x7x7xf32>
    %v1435 = stablehlo.reshape %v1434 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1437 = stablehlo.convolution(%v1436, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1438 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1439 = stablehlo.add %v1437, %v1438 : tensor<256x512x7x7xf32>
    %v1440 = stablehlo.reshape %v1439 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1443 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1444 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1445 = stablehlo.reduce(%v1441 init: %v1442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1446 = stablehlo.broadcast_in_dim %v1445, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1447 = stablehlo.divide %v1446, %v1443 : tensor<256x512x7x7xf32>
    %v1448 = stablehlo.subtract %v1441, %v1447 : tensor<256x512x7x7xf32>
    %v1449 = stablehlo.multiply %v1448, %v1448 : tensor<256x512x7x7xf32>
    %v1450 = stablehlo.reduce(%v1449 init: %v1442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1451 = stablehlo.broadcast_in_dim %v1450, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1452 = stablehlo.divide %v1451, %v1443 : tensor<256x512x7x7xf32>
    %v1453 = stablehlo.add %v1452, %v1444 : tensor<256x512x7x7xf32>
    %v1454 = stablehlo.rsqrt %v1453 : tensor<256x512x7x7xf32>
    %v1455 = stablehlo.multiply %v1448, %v1454 : tensor<256x512x7x7xf32>
    %v1456 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1457 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1458 = stablehlo.multiply %v1455, %v1456 : tensor<256x512x7x7xf32>
    %v1459 = stablehlo.add %v1458, %v1457 : tensor<256x512x7x7xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1461 = stablehlo.reshape %v1460 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1462 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1463 = stablehlo.maximum %v1461, %v1462 : tensor<256x512x7x7xf32>
    %v1464 = stablehlo.reshape %v1463 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1465 = stablehlo.reshape %v1464 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1466 = stablehlo.convolution(%v1465, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1467 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1468 = stablehlo.add %v1466, %v1467 : tensor<256x2048x7x7xf32>
    %v1469 = stablehlo.reshape %v1468 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1470 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1472 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1473 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1474 = stablehlo.reduce(%v1470 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1475 = stablehlo.broadcast_in_dim %v1474, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1476 = stablehlo.divide %v1475, %v1472 : tensor<256x2048x7x7xf32>
    %v1477 = stablehlo.subtract %v1470, %v1476 : tensor<256x2048x7x7xf32>
    %v1478 = stablehlo.multiply %v1477, %v1477 : tensor<256x2048x7x7xf32>
    %v1479 = stablehlo.reduce(%v1478 init: %v1471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1480 = stablehlo.broadcast_in_dim %v1479, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1481 = stablehlo.divide %v1480, %v1472 : tensor<256x2048x7x7xf32>
    %v1482 = stablehlo.add %v1481, %v1473 : tensor<256x2048x7x7xf32>
    %v1483 = stablehlo.rsqrt %v1482 : tensor<256x2048x7x7xf32>
    %v1484 = stablehlo.multiply %v1477, %v1483 : tensor<256x2048x7x7xf32>
    %v1485 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1486 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1487 = stablehlo.multiply %v1484, %v1485 : tensor<256x2048x7x7xf32>
    %v1488 = stablehlo.add %v1487, %v1486 : tensor<256x2048x7x7xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1490 = stablehlo.reshape %v1489 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1491 = stablehlo.reshape %v1406 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1492 = stablehlo.add %v1490, %v1491 : tensor<256x2048x7x7xf32>
    %v1493 = stablehlo.reshape %v1492 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1494 = stablehlo.reshape %v1493 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1495 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1496 = stablehlo.maximum %v1494, %v1495 : tensor<256x2048x7x7xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1498 = stablehlo.reshape %v1497 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1499 = stablehlo.convolution(%v1498, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1500 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1501 = stablehlo.add %v1499, %v1500 : tensor<256x512x7x7xf32>
    %v1502 = stablehlo.reshape %v1501 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1503 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1505 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1506 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1507 = stablehlo.reduce(%v1503 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1508 = stablehlo.broadcast_in_dim %v1507, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1509 = stablehlo.divide %v1508, %v1505 : tensor<256x512x7x7xf32>
    %v1510 = stablehlo.subtract %v1503, %v1509 : tensor<256x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1510, %v1510 : tensor<256x512x7x7xf32>
    %v1512 = stablehlo.reduce(%v1511 init: %v1504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1513 = stablehlo.broadcast_in_dim %v1512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1514 = stablehlo.divide %v1513, %v1505 : tensor<256x512x7x7xf32>
    %v1515 = stablehlo.add %v1514, %v1506 : tensor<256x512x7x7xf32>
    %v1516 = stablehlo.rsqrt %v1515 : tensor<256x512x7x7xf32>
    %v1517 = stablehlo.multiply %v1510, %v1516 : tensor<256x512x7x7xf32>
    %v1518 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1519 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1517, %v1518 : tensor<256x512x7x7xf32>
    %v1521 = stablehlo.add %v1520, %v1519 : tensor<256x512x7x7xf32>
    %v1522 = stablehlo.reshape %v1521 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1524 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1525 = stablehlo.maximum %v1523, %v1524 : tensor<256x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1528 = stablehlo.convolution(%v1527, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1529 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1530 = stablehlo.add %v1528, %v1529 : tensor<256x512x7x7xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1532 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1535 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1536 = stablehlo.reduce(%v1532 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1538 = stablehlo.divide %v1537, %v1534 : tensor<256x512x7x7xf32>
    %v1539 = stablehlo.subtract %v1532, %v1538 : tensor<256x512x7x7xf32>
    %v1540 = stablehlo.multiply %v1539, %v1539 : tensor<256x512x7x7xf32>
    %v1541 = stablehlo.reduce(%v1540 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1542 = stablehlo.broadcast_in_dim %v1541, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1543 = stablehlo.divide %v1542, %v1534 : tensor<256x512x7x7xf32>
    %v1544 = stablehlo.add %v1543, %v1535 : tensor<256x512x7x7xf32>
    %v1545 = stablehlo.rsqrt %v1544 : tensor<256x512x7x7xf32>
    %v1546 = stablehlo.multiply %v1539, %v1545 : tensor<256x512x7x7xf32>
    %v1547 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1548 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1549 = stablehlo.multiply %v1546, %v1547 : tensor<256x512x7x7xf32>
    %v1550 = stablehlo.add %v1549, %v1548 : tensor<256x512x7x7xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1552 = stablehlo.reshape %v1551 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1553 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1554 = stablehlo.maximum %v1552, %v1553 : tensor<256x512x7x7xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1557 = stablehlo.convolution(%v1556, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1558 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1559 = stablehlo.add %v1557, %v1558 : tensor<256x2048x7x7xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1562 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1563 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1564 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1565 = stablehlo.reduce(%v1561 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1567 = stablehlo.divide %v1566, %v1563 : tensor<256x2048x7x7xf32>
    %v1568 = stablehlo.subtract %v1561, %v1567 : tensor<256x2048x7x7xf32>
    %v1569 = stablehlo.multiply %v1568, %v1568 : tensor<256x2048x7x7xf32>
    %v1570 = stablehlo.reduce(%v1569 init: %v1562) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1571 = stablehlo.broadcast_in_dim %v1570, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1572 = stablehlo.divide %v1571, %v1563 : tensor<256x2048x7x7xf32>
    %v1573 = stablehlo.add %v1572, %v1564 : tensor<256x2048x7x7xf32>
    %v1574 = stablehlo.rsqrt %v1573 : tensor<256x2048x7x7xf32>
    %v1575 = stablehlo.multiply %v1568, %v1574 : tensor<256x2048x7x7xf32>
    %v1576 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1577 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1578 = stablehlo.multiply %v1575, %v1576 : tensor<256x2048x7x7xf32>
    %v1579 = stablehlo.add %v1578, %v1577 : tensor<256x2048x7x7xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1581 = stablehlo.reshape %v1580 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1582 = stablehlo.reshape %v1497 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1583 = stablehlo.add %v1581, %v1582 : tensor<256x2048x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1586 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1587 = stablehlo.maximum %v1585, %v1586 : tensor<256x2048x7x7xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1591 = stablehlo.reduce(%v1589 init: %v1590) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %v1592 = stablehlo.constant dense<49.0> : tensor<256x2048xf32>
    %v1593 = stablehlo.divide %v1591, %v1592 : tensor<256x2048xf32>
    %v1594 = stablehlo.dot_general %v1593, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1000xf32>) -> tensor<256x1000xf32>
    %v1595 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v1596 = stablehlo.add %v1594, %v1595 : tensor<256x1000xf32>
    %v1597 = stablehlo.reshape %v1596 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1599 = stablehlo.exponential %v1597 : tensor<256x1x1000xf32>
    %v1600 = stablehlo.reduce(%v1599 init: %v1598) applies stablehlo.add across dimensions = [2] : (tensor<256x1x1000xf32>, tensor<f32>) -> tensor<256x1xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x1x1000xf32>
    %v1602 = stablehlo.divide %v1599, %v1601 : tensor<256x1x1000xf32>
    %v1603 = stablehlo.reshape %v1602 : (tensor<256x1x1000xf32>) -> tensor<256x1000xf32>
    %v1604 = stablehlo.subtract %v1603, %onehot : tensor<256x1000xf32>
    %v1605 = stablehlo.constant dense<0.100000> : tensor<256x1000xf32>
    %v1606 = stablehlo.multiply %onehot, %v1605 : tensor<256x1000xf32>
    %v1607 = stablehlo.add %v1604, %v1606 : tensor<256x1000xf32>
    %v1608 = stablehlo.constant dense<-0.000100> : tensor<256x1000xf32>
    %v1609 = stablehlo.add %v1607, %v1608 : tensor<256x1000xf32>
    %v1610 = stablehlo.constant dense<256.0> : tensor<256x1000xf32>
    %v1611 = stablehlo.divide %v1609, %v1610 : tensor<256x1000xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1613 = stablehlo.dot_general %v1612, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<256x1x2048xf32>
    %v1614 = stablehlo.reshape %v1613 : (tensor<256x1x2048xf32>) -> tensor<256x2048xf32>
    %v1615 = stablehlo.dot_general %v1593, %v1611, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<256x1000xf32>) -> tensor<2048x1000xf32>
    %v1616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1617 = stablehlo.reduce(%v1611 init: %v1616) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1618 = stablehlo.broadcast_in_dim %v1614, dims = [0, 1] : (tensor<256x2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1619 = stablehlo.constant dense<49.0> : tensor<256x2048x7x7xf32>
    %v1620 = stablehlo.divide %v1618, %v1619 : tensor<256x2048x7x7xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1622 = stablehlo.reshape %v1621 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1623 = stablehlo.reshape %v1584 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1624 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1625 = stablehlo.compare GT, %v1623, %v1624 : (tensor<256x2048x7x7xf32>, tensor<256x2048x7x7xf32>) -> tensor<256x2048x7x7xi1>
    %v1626 = stablehlo.select %v1625, %v1622, %v1624 : tensor<256x2048x7x7xi1>, tensor<256x2048x7x7xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1628 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1630 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1631 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1632 = stablehlo.reduce(%v1628 init: %v1629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1634 = stablehlo.divide %v1633, %v1630 : tensor<256x2048x7x7xf32>
    %v1635 = stablehlo.subtract %v1628, %v1634 : tensor<256x2048x7x7xf32>
    %v1636 = stablehlo.multiply %v1635, %v1635 : tensor<256x2048x7x7xf32>
    %v1637 = stablehlo.reduce(%v1636 init: %v1629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1638 = stablehlo.broadcast_in_dim %v1637, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1639 = stablehlo.divide %v1638, %v1630 : tensor<256x2048x7x7xf32>
    %v1640 = stablehlo.add %v1639, %v1631 : tensor<256x2048x7x7xf32>
    %v1641 = stablehlo.rsqrt %v1640 : tensor<256x2048x7x7xf32>
    %v1642 = stablehlo.multiply %v1635, %v1641 : tensor<256x2048x7x7xf32>
    %v1643 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1644 = stablehlo.reshape %v1627 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1645 = stablehlo.multiply %v1643, %v1644 : tensor<256x2048x7x7xf32>
    %v1646 = stablehlo.reduce(%v1645 init: %v1629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1647 = stablehlo.broadcast_in_dim %v1646, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1648 = stablehlo.multiply %v1642, %v1645 : tensor<256x2048x7x7xf32>
    %v1649 = stablehlo.reduce(%v1648 init: %v1629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1650 = stablehlo.broadcast_in_dim %v1649, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1651 = stablehlo.multiply %v1645, %v1630 : tensor<256x2048x7x7xf32>
    %v1652 = stablehlo.subtract %v1651, %v1647 : tensor<256x2048x7x7xf32>
    %v1653 = stablehlo.multiply %v1642, %v1650 : tensor<256x2048x7x7xf32>
    %v1654 = stablehlo.subtract %v1652, %v1653 : tensor<256x2048x7x7xf32>
    %v1655 = stablehlo.divide %v1641, %v1630 : tensor<256x2048x7x7xf32>
    %v1656 = stablehlo.multiply %v1655, %v1654 : tensor<256x2048x7x7xf32>
    %v1657 = stablehlo.reshape %v1656 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1659 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1660 = stablehlo.transpose %v1659, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1661 = stablehlo.convolution(%v1658, %v1660)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1664 = stablehlo.reshape %v1551 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1665 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1666 = stablehlo.compare GT, %v1664, %v1665 : (tensor<256x512x7x7xf32>, tensor<256x512x7x7xf32>) -> tensor<256x512x7x7xi1>
    %v1667 = stablehlo.select %v1666, %v1663, %v1665 : tensor<256x512x7x7xi1>, tensor<256x512x7x7xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1669 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1671 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1672 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1673 = stablehlo.reduce(%v1669 init: %v1670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1674 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1675 = stablehlo.divide %v1674, %v1671 : tensor<256x512x7x7xf32>
    %v1676 = stablehlo.subtract %v1669, %v1675 : tensor<256x512x7x7xf32>
    %v1677 = stablehlo.multiply %v1676, %v1676 : tensor<256x512x7x7xf32>
    %v1678 = stablehlo.reduce(%v1677 init: %v1670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1679 = stablehlo.broadcast_in_dim %v1678, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1680 = stablehlo.divide %v1679, %v1671 : tensor<256x512x7x7xf32>
    %v1681 = stablehlo.add %v1680, %v1672 : tensor<256x512x7x7xf32>
    %v1682 = stablehlo.rsqrt %v1681 : tensor<256x512x7x7xf32>
    %v1683 = stablehlo.multiply %v1676, %v1682 : tensor<256x512x7x7xf32>
    %v1684 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1685 = stablehlo.reshape %v1668 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1686 = stablehlo.multiply %v1684, %v1685 : tensor<256x512x7x7xf32>
    %v1687 = stablehlo.reduce(%v1686 init: %v1670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1688 = stablehlo.broadcast_in_dim %v1687, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1689 = stablehlo.multiply %v1683, %v1686 : tensor<256x512x7x7xf32>
    %v1690 = stablehlo.reduce(%v1689 init: %v1670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1691 = stablehlo.broadcast_in_dim %v1690, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1692 = stablehlo.multiply %v1686, %v1671 : tensor<256x512x7x7xf32>
    %v1693 = stablehlo.subtract %v1692, %v1688 : tensor<256x512x7x7xf32>
    %v1694 = stablehlo.multiply %v1683, %v1691 : tensor<256x512x7x7xf32>
    %v1695 = stablehlo.subtract %v1693, %v1694 : tensor<256x512x7x7xf32>
    %v1696 = stablehlo.divide %v1682, %v1671 : tensor<256x512x7x7xf32>
    %v1697 = stablehlo.multiply %v1696, %v1695 : tensor<256x512x7x7xf32>
    %v1698 = stablehlo.reshape %v1697 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1700 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1701 = stablehlo.transpose %v1700, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1702 = stablehlo.convolution(%v1699, %v1701)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1703 = stablehlo.reshape %v1702 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1705 = stablehlo.reshape %v1522 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1706 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1707 = stablehlo.compare GT, %v1705, %v1706 : (tensor<256x512x7x7xf32>, tensor<256x512x7x7xf32>) -> tensor<256x512x7x7xi1>
    %v1708 = stablehlo.select %v1707, %v1704, %v1706 : tensor<256x512x7x7xi1>, tensor<256x512x7x7xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1710 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1712 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1713 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1714 = stablehlo.reduce(%v1710 init: %v1711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1715 = stablehlo.broadcast_in_dim %v1714, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1716 = stablehlo.divide %v1715, %v1712 : tensor<256x512x7x7xf32>
    %v1717 = stablehlo.subtract %v1710, %v1716 : tensor<256x512x7x7xf32>
    %v1718 = stablehlo.multiply %v1717, %v1717 : tensor<256x512x7x7xf32>
    %v1719 = stablehlo.reduce(%v1718 init: %v1711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1720 = stablehlo.broadcast_in_dim %v1719, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1721 = stablehlo.divide %v1720, %v1712 : tensor<256x512x7x7xf32>
    %v1722 = stablehlo.add %v1721, %v1713 : tensor<256x512x7x7xf32>
    %v1723 = stablehlo.rsqrt %v1722 : tensor<256x512x7x7xf32>
    %v1724 = stablehlo.multiply %v1717, %v1723 : tensor<256x512x7x7xf32>
    %v1725 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1726 = stablehlo.reshape %v1709 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1727 = stablehlo.multiply %v1725, %v1726 : tensor<256x512x7x7xf32>
    %v1728 = stablehlo.reduce(%v1727 init: %v1711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1729 = stablehlo.broadcast_in_dim %v1728, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1730 = stablehlo.multiply %v1724, %v1727 : tensor<256x512x7x7xf32>
    %v1731 = stablehlo.reduce(%v1730 init: %v1711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1731, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1733 = stablehlo.multiply %v1727, %v1712 : tensor<256x512x7x7xf32>
    %v1734 = stablehlo.subtract %v1733, %v1729 : tensor<256x512x7x7xf32>
    %v1735 = stablehlo.multiply %v1724, %v1732 : tensor<256x512x7x7xf32>
    %v1736 = stablehlo.subtract %v1734, %v1735 : tensor<256x512x7x7xf32>
    %v1737 = stablehlo.divide %v1723, %v1712 : tensor<256x512x7x7xf32>
    %v1738 = stablehlo.multiply %v1737, %v1736 : tensor<256x512x7x7xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1741 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1742 = stablehlo.transpose %v1741, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1743 = stablehlo.convolution(%v1740, %v1742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1746 = stablehlo.reshape %v1627 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1747 = stablehlo.add %v1745, %v1746 : tensor<256x2048x7x7xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1749 = stablehlo.reshape %v1497 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1750 = stablehlo.reshape %v1739 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1751 = stablehlo.transpose %v1749, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1752 = stablehlo.transpose %v1750, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1753 = stablehlo.convolution(%v1751, %v1752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v1754 = stablehlo.transpose %v1753, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1755 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1757 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1758 = stablehlo.reduce(%v1755 init: %v1756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1759 = stablehlo.broadcast_in_dim %v1758, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1760 = stablehlo.divide %v1759, %v1757 : tensor<256x512x7x7xf32>
    %v1761 = stablehlo.subtract %v1755, %v1760 : tensor<256x512x7x7xf32>
    %v1762 = stablehlo.multiply %v1761, %v1761 : tensor<256x512x7x7xf32>
    %v1763 = stablehlo.reduce(%v1762 init: %v1756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1764 = stablehlo.broadcast_in_dim %v1763, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1765 = stablehlo.divide %v1764, %v1757 : tensor<256x512x7x7xf32>
    %v1766 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1767 = stablehlo.add %v1765, %v1766 : tensor<256x512x7x7xf32>
    %v1768 = stablehlo.rsqrt %v1767 : tensor<256x512x7x7xf32>
    %v1769 = stablehlo.multiply %v1761, %v1768 : tensor<256x512x7x7xf32>
    %v1770 = stablehlo.reshape %v1709 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1771 = stablehlo.multiply %v1770, %v1769 : tensor<256x512x7x7xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1773 = stablehlo.reshape %v1709 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1775 = stablehlo.reduce(%v1773 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1776 = stablehlo.reshape %v1526 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1777 = stablehlo.reshape %v1698 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1778 = stablehlo.transpose %v1776, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1779 = stablehlo.transpose %v1777, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1780 = stablehlo.convolution(%v1778, %v1779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1781 = stablehlo.transpose %v1780, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1782 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1784 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1785 = stablehlo.reduce(%v1782 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1786 = stablehlo.broadcast_in_dim %v1785, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1787 = stablehlo.divide %v1786, %v1784 : tensor<256x512x7x7xf32>
    %v1788 = stablehlo.subtract %v1782, %v1787 : tensor<256x512x7x7xf32>
    %v1789 = stablehlo.multiply %v1788, %v1788 : tensor<256x512x7x7xf32>
    %v1790 = stablehlo.reduce(%v1789 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1791 = stablehlo.broadcast_in_dim %v1790, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1792 = stablehlo.divide %v1791, %v1784 : tensor<256x512x7x7xf32>
    %v1793 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1794 = stablehlo.add %v1792, %v1793 : tensor<256x512x7x7xf32>
    %v1795 = stablehlo.rsqrt %v1794 : tensor<256x512x7x7xf32>
    %v1796 = stablehlo.multiply %v1788, %v1795 : tensor<256x512x7x7xf32>
    %v1797 = stablehlo.reshape %v1668 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1798 = stablehlo.multiply %v1797, %v1796 : tensor<256x512x7x7xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1800 = stablehlo.reshape %v1668 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1802 = stablehlo.reduce(%v1800 init: %v1801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1803 = stablehlo.reshape %v1555 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1804 = stablehlo.reshape %v1657 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1805 = stablehlo.transpose %v1803, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1806 = stablehlo.transpose %v1804, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1807 = stablehlo.convolution(%v1805, %v1806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v1808 = stablehlo.transpose %v1807, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1809 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1811 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1812 = stablehlo.reduce(%v1809 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1813 = stablehlo.broadcast_in_dim %v1812, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1814 = stablehlo.divide %v1813, %v1811 : tensor<256x2048x7x7xf32>
    %v1815 = stablehlo.subtract %v1809, %v1814 : tensor<256x2048x7x7xf32>
    %v1816 = stablehlo.multiply %v1815, %v1815 : tensor<256x2048x7x7xf32>
    %v1817 = stablehlo.reduce(%v1816 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1818 = stablehlo.broadcast_in_dim %v1817, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1819 = stablehlo.divide %v1818, %v1811 : tensor<256x2048x7x7xf32>
    %v1820 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1821 = stablehlo.add %v1819, %v1820 : tensor<256x2048x7x7xf32>
    %v1822 = stablehlo.rsqrt %v1821 : tensor<256x2048x7x7xf32>
    %v1823 = stablehlo.multiply %v1815, %v1822 : tensor<256x2048x7x7xf32>
    %v1824 = stablehlo.reshape %v1627 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1825 = stablehlo.multiply %v1824, %v1823 : tensor<256x2048x7x7xf32>
    %v1826 = stablehlo.reduce(%v1825 init: %v1810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1827 = stablehlo.reshape %v1627 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1829 = stablehlo.reduce(%v1827 init: %v1828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1830 = stablehlo.reshape %v1748 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1831 = stablehlo.reshape %v1493 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1832 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v1833 = stablehlo.compare GT, %v1831, %v1832 : (tensor<256x2048x7x7xf32>, tensor<256x2048x7x7xf32>) -> tensor<256x2048x7x7xi1>
    %v1834 = stablehlo.select %v1833, %v1830, %v1832 : tensor<256x2048x7x7xi1>, tensor<256x2048x7x7xf32>
    %v1835 = stablehlo.reshape %v1834 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1836 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1838 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1839 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1840 = stablehlo.reduce(%v1836 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1841 = stablehlo.broadcast_in_dim %v1840, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1842 = stablehlo.divide %v1841, %v1838 : tensor<256x2048x7x7xf32>
    %v1843 = stablehlo.subtract %v1836, %v1842 : tensor<256x2048x7x7xf32>
    %v1844 = stablehlo.multiply %v1843, %v1843 : tensor<256x2048x7x7xf32>
    %v1845 = stablehlo.reduce(%v1844 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1846 = stablehlo.broadcast_in_dim %v1845, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1847 = stablehlo.divide %v1846, %v1838 : tensor<256x2048x7x7xf32>
    %v1848 = stablehlo.add %v1847, %v1839 : tensor<256x2048x7x7xf32>
    %v1849 = stablehlo.rsqrt %v1848 : tensor<256x2048x7x7xf32>
    %v1850 = stablehlo.multiply %v1843, %v1849 : tensor<256x2048x7x7xf32>
    %v1851 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1852 = stablehlo.reshape %v1835 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1853 = stablehlo.multiply %v1851, %v1852 : tensor<256x2048x7x7xf32>
    %v1854 = stablehlo.reduce(%v1853 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1854, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1856 = stablehlo.multiply %v1850, %v1853 : tensor<256x2048x7x7xf32>
    %v1857 = stablehlo.reduce(%v1856 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1858 = stablehlo.broadcast_in_dim %v1857, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1859 = stablehlo.multiply %v1853, %v1838 : tensor<256x2048x7x7xf32>
    %v1860 = stablehlo.subtract %v1859, %v1855 : tensor<256x2048x7x7xf32>
    %v1861 = stablehlo.multiply %v1850, %v1858 : tensor<256x2048x7x7xf32>
    %v1862 = stablehlo.subtract %v1860, %v1861 : tensor<256x2048x7x7xf32>
    %v1863 = stablehlo.divide %v1849, %v1838 : tensor<256x2048x7x7xf32>
    %v1864 = stablehlo.multiply %v1863, %v1862 : tensor<256x2048x7x7xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1867 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1868 = stablehlo.transpose %v1867, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1869 = stablehlo.convolution(%v1866, %v1868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1871 = stablehlo.reshape %v1870 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1872 = stablehlo.reshape %v1460 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1873 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1874 = stablehlo.compare GT, %v1872, %v1873 : (tensor<256x512x7x7xf32>, tensor<256x512x7x7xf32>) -> tensor<256x512x7x7xi1>
    %v1875 = stablehlo.select %v1874, %v1871, %v1873 : tensor<256x512x7x7xi1>, tensor<256x512x7x7xf32>
    %v1876 = stablehlo.reshape %v1875 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1877 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1879 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1880 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1881 = stablehlo.reduce(%v1877 init: %v1878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1882 = stablehlo.broadcast_in_dim %v1881, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1883 = stablehlo.divide %v1882, %v1879 : tensor<256x512x7x7xf32>
    %v1884 = stablehlo.subtract %v1877, %v1883 : tensor<256x512x7x7xf32>
    %v1885 = stablehlo.multiply %v1884, %v1884 : tensor<256x512x7x7xf32>
    %v1886 = stablehlo.reduce(%v1885 init: %v1878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1887 = stablehlo.broadcast_in_dim %v1886, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1888 = stablehlo.divide %v1887, %v1879 : tensor<256x512x7x7xf32>
    %v1889 = stablehlo.add %v1888, %v1880 : tensor<256x512x7x7xf32>
    %v1890 = stablehlo.rsqrt %v1889 : tensor<256x512x7x7xf32>
    %v1891 = stablehlo.multiply %v1884, %v1890 : tensor<256x512x7x7xf32>
    %v1892 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1893 = stablehlo.reshape %v1876 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1894 = stablehlo.multiply %v1892, %v1893 : tensor<256x512x7x7xf32>
    %v1895 = stablehlo.reduce(%v1894 init: %v1878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1896 = stablehlo.broadcast_in_dim %v1895, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1897 = stablehlo.multiply %v1891, %v1894 : tensor<256x512x7x7xf32>
    %v1898 = stablehlo.reduce(%v1897 init: %v1878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1899 = stablehlo.broadcast_in_dim %v1898, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1900 = stablehlo.multiply %v1894, %v1879 : tensor<256x512x7x7xf32>
    %v1901 = stablehlo.subtract %v1900, %v1896 : tensor<256x512x7x7xf32>
    %v1902 = stablehlo.multiply %v1891, %v1899 : tensor<256x512x7x7xf32>
    %v1903 = stablehlo.subtract %v1901, %v1902 : tensor<256x512x7x7xf32>
    %v1904 = stablehlo.divide %v1890, %v1879 : tensor<256x512x7x7xf32>
    %v1905 = stablehlo.multiply %v1904, %v1903 : tensor<256x512x7x7xf32>
    %v1906 = stablehlo.reshape %v1905 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1907 = stablehlo.reshape %v1906 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1908 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1909 = stablehlo.transpose %v1908, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1910 = stablehlo.convolution(%v1907, %v1909)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1913 = stablehlo.reshape %v1431 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1914 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v1915 = stablehlo.compare GT, %v1913, %v1914 : (tensor<256x512x7x7xf32>, tensor<256x512x7x7xf32>) -> tensor<256x512x7x7xi1>
    %v1916 = stablehlo.select %v1915, %v1912, %v1914 : tensor<256x512x7x7xi1>, tensor<256x512x7x7xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1918 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1920 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1921 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1922 = stablehlo.reduce(%v1918 init: %v1919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1923 = stablehlo.broadcast_in_dim %v1922, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1924 = stablehlo.divide %v1923, %v1920 : tensor<256x512x7x7xf32>
    %v1925 = stablehlo.subtract %v1918, %v1924 : tensor<256x512x7x7xf32>
    %v1926 = stablehlo.multiply %v1925, %v1925 : tensor<256x512x7x7xf32>
    %v1927 = stablehlo.reduce(%v1926 init: %v1919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1928 = stablehlo.broadcast_in_dim %v1927, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1929 = stablehlo.divide %v1928, %v1920 : tensor<256x512x7x7xf32>
    %v1930 = stablehlo.add %v1929, %v1921 : tensor<256x512x7x7xf32>
    %v1931 = stablehlo.rsqrt %v1930 : tensor<256x512x7x7xf32>
    %v1932 = stablehlo.multiply %v1925, %v1931 : tensor<256x512x7x7xf32>
    %v1933 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1934 = stablehlo.reshape %v1917 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1935 = stablehlo.multiply %v1933, %v1934 : tensor<256x512x7x7xf32>
    %v1936 = stablehlo.reduce(%v1935 init: %v1919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1937 = stablehlo.broadcast_in_dim %v1936, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1938 = stablehlo.multiply %v1932, %v1935 : tensor<256x512x7x7xf32>
    %v1939 = stablehlo.reduce(%v1938 init: %v1919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1941 = stablehlo.multiply %v1935, %v1920 : tensor<256x512x7x7xf32>
    %v1942 = stablehlo.subtract %v1941, %v1937 : tensor<256x512x7x7xf32>
    %v1943 = stablehlo.multiply %v1932, %v1940 : tensor<256x512x7x7xf32>
    %v1944 = stablehlo.subtract %v1942, %v1943 : tensor<256x512x7x7xf32>
    %v1945 = stablehlo.divide %v1931, %v1920 : tensor<256x512x7x7xf32>
    %v1946 = stablehlo.multiply %v1945, %v1944 : tensor<256x512x7x7xf32>
    %v1947 = stablehlo.reshape %v1946 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1949 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1950 = stablehlo.transpose %v1949, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1951 = stablehlo.convolution(%v1948, %v1950)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1954 = stablehlo.reshape %v1835 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1955 = stablehlo.add %v1953, %v1954 : tensor<256x2048x7x7xf32>
    %v1956 = stablehlo.reshape %v1955 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1957 = stablehlo.reshape %v1406 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1958 = stablehlo.reshape %v1947 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1959 = stablehlo.transpose %v1957, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1960 = stablehlo.transpose %v1958, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1961 = stablehlo.convolution(%v1959, %v1960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v1962 = stablehlo.transpose %v1961, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1963 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1965 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1966 = stablehlo.reduce(%v1963 init: %v1964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1968 = stablehlo.divide %v1967, %v1965 : tensor<256x512x7x7xf32>
    %v1969 = stablehlo.subtract %v1963, %v1968 : tensor<256x512x7x7xf32>
    %v1970 = stablehlo.multiply %v1969, %v1969 : tensor<256x512x7x7xf32>
    %v1971 = stablehlo.reduce(%v1970 init: %v1964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1973 = stablehlo.divide %v1972, %v1965 : tensor<256x512x7x7xf32>
    %v1974 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1975 = stablehlo.add %v1973, %v1974 : tensor<256x512x7x7xf32>
    %v1976 = stablehlo.rsqrt %v1975 : tensor<256x512x7x7xf32>
    %v1977 = stablehlo.multiply %v1969, %v1976 : tensor<256x512x7x7xf32>
    %v1978 = stablehlo.reshape %v1917 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1979 = stablehlo.multiply %v1978, %v1977 : tensor<256x512x7x7xf32>
    %v1980 = stablehlo.reduce(%v1979 init: %v1964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1981 = stablehlo.reshape %v1917 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1983 = stablehlo.reduce(%v1981 init: %v1982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1984 = stablehlo.reshape %v1435 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1985 = stablehlo.reshape %v1906 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1986 = stablehlo.transpose %v1984, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1987 = stablehlo.transpose %v1985, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1988 = stablehlo.convolution(%v1986, %v1987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1989 = stablehlo.transpose %v1988, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1990 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1992 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1993 = stablehlo.reduce(%v1990 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1995 = stablehlo.divide %v1994, %v1992 : tensor<256x512x7x7xf32>
    %v1996 = stablehlo.subtract %v1990, %v1995 : tensor<256x512x7x7xf32>
    %v1997 = stablehlo.multiply %v1996, %v1996 : tensor<256x512x7x7xf32>
    %v1998 = stablehlo.reduce(%v1997 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1999 = stablehlo.broadcast_in_dim %v1998, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2000 = stablehlo.divide %v1999, %v1992 : tensor<256x512x7x7xf32>
    %v2001 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v2002 = stablehlo.add %v2000, %v2001 : tensor<256x512x7x7xf32>
    %v2003 = stablehlo.rsqrt %v2002 : tensor<256x512x7x7xf32>
    %v2004 = stablehlo.multiply %v1996, %v2003 : tensor<256x512x7x7xf32>
    %v2005 = stablehlo.reshape %v1876 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2006 = stablehlo.multiply %v2005, %v2004 : tensor<256x512x7x7xf32>
    %v2007 = stablehlo.reduce(%v2006 init: %v1991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2008 = stablehlo.reshape %v1876 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2010 = stablehlo.reduce(%v2008 init: %v2009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2011 = stablehlo.reshape %v1464 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2012 = stablehlo.reshape %v1865 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2013 = stablehlo.transpose %v2011, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v2014 = stablehlo.transpose %v2012, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v2015 = stablehlo.convolution(%v2013, %v2014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2016 = stablehlo.transpose %v2015, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2017 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2019 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2020 = stablehlo.reduce(%v2017 init: %v2018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2021 = stablehlo.broadcast_in_dim %v2020, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2022 = stablehlo.divide %v2021, %v2019 : tensor<256x2048x7x7xf32>
    %v2023 = stablehlo.subtract %v2017, %v2022 : tensor<256x2048x7x7xf32>
    %v2024 = stablehlo.multiply %v2023, %v2023 : tensor<256x2048x7x7xf32>
    %v2025 = stablehlo.reduce(%v2024 init: %v2018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2026 = stablehlo.broadcast_in_dim %v2025, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2027 = stablehlo.divide %v2026, %v2019 : tensor<256x2048x7x7xf32>
    %v2028 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2029 = stablehlo.add %v2027, %v2028 : tensor<256x2048x7x7xf32>
    %v2030 = stablehlo.rsqrt %v2029 : tensor<256x2048x7x7xf32>
    %v2031 = stablehlo.multiply %v2023, %v2030 : tensor<256x2048x7x7xf32>
    %v2032 = stablehlo.reshape %v1835 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2033 = stablehlo.multiply %v2032, %v2031 : tensor<256x2048x7x7xf32>
    %v2034 = stablehlo.reduce(%v2033 init: %v2018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2035 = stablehlo.reshape %v1835 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2037 = stablehlo.reduce(%v2035 init: %v2036) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2038 = stablehlo.reshape %v1956 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2039 = stablehlo.reshape %v1402 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2040 = stablehlo.constant dense<0.0> : tensor<256x2048x7x7xf32>
    %v2041 = stablehlo.compare GT, %v2039, %v2040 : (tensor<256x2048x7x7xf32>, tensor<256x2048x7x7xf32>) -> tensor<256x2048x7x7xi1>
    %v2042 = stablehlo.select %v2041, %v2038, %v2040 : tensor<256x2048x7x7xi1>, tensor<256x2048x7x7xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v2044 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2046 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2047 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2048 = stablehlo.reduce(%v2044 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2050 = stablehlo.divide %v2049, %v2046 : tensor<256x2048x7x7xf32>
    %v2051 = stablehlo.subtract %v2044, %v2050 : tensor<256x2048x7x7xf32>
    %v2052 = stablehlo.multiply %v2051, %v2051 : tensor<256x2048x7x7xf32>
    %v2053 = stablehlo.reduce(%v2052 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2054 = stablehlo.broadcast_in_dim %v2053, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2055 = stablehlo.divide %v2054, %v2046 : tensor<256x2048x7x7xf32>
    %v2056 = stablehlo.add %v2055, %v2047 : tensor<256x2048x7x7xf32>
    %v2057 = stablehlo.rsqrt %v2056 : tensor<256x2048x7x7xf32>
    %v2058 = stablehlo.multiply %v2051, %v2057 : tensor<256x2048x7x7xf32>
    %v2059 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2060 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2061 = stablehlo.multiply %v2059, %v2060 : tensor<256x2048x7x7xf32>
    %v2062 = stablehlo.reduce(%v2061 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2063 = stablehlo.broadcast_in_dim %v2062, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2064 = stablehlo.multiply %v2058, %v2061 : tensor<256x2048x7x7xf32>
    %v2065 = stablehlo.reduce(%v2064 init: %v2045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2067 = stablehlo.multiply %v2061, %v2046 : tensor<256x2048x7x7xf32>
    %v2068 = stablehlo.subtract %v2067, %v2063 : tensor<256x2048x7x7xf32>
    %v2069 = stablehlo.multiply %v2058, %v2066 : tensor<256x2048x7x7xf32>
    %v2070 = stablehlo.subtract %v2068, %v2069 : tensor<256x2048x7x7xf32>
    %v2071 = stablehlo.divide %v2057, %v2046 : tensor<256x2048x7x7xf32>
    %v2072 = stablehlo.multiply %v2071, %v2070 : tensor<256x2048x7x7xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2075 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2076 = stablehlo.transpose %v2075, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2077 = stablehlo.convolution(%v2074, %v2076)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v2078 = stablehlo.reshape %v2077 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v2079 = stablehlo.reshape %v2078 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2080 = stablehlo.reshape %v1344 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2081 = stablehlo.constant dense<0.0> : tensor<256x512x7x7xf32>
    %v2082 = stablehlo.compare GT, %v2080, %v2081 : (tensor<256x512x7x7xf32>, tensor<256x512x7x7xf32>) -> tensor<256x512x7x7xi1>
    %v2083 = stablehlo.select %v2082, %v2079, %v2081 : tensor<256x512x7x7xi1>, tensor<256x512x7x7xf32>
    %v2084 = stablehlo.reshape %v2083 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v2085 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2087 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v2088 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v2089 = stablehlo.reduce(%v2085 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2090 = stablehlo.broadcast_in_dim %v2089, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2091 = stablehlo.divide %v2090, %v2087 : tensor<256x512x7x7xf32>
    %v2092 = stablehlo.subtract %v2085, %v2091 : tensor<256x512x7x7xf32>
    %v2093 = stablehlo.multiply %v2092, %v2092 : tensor<256x512x7x7xf32>
    %v2094 = stablehlo.reduce(%v2093 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2095 = stablehlo.broadcast_in_dim %v2094, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2096 = stablehlo.divide %v2095, %v2087 : tensor<256x512x7x7xf32>
    %v2097 = stablehlo.add %v2096, %v2088 : tensor<256x512x7x7xf32>
    %v2098 = stablehlo.rsqrt %v2097 : tensor<256x512x7x7xf32>
    %v2099 = stablehlo.multiply %v2092, %v2098 : tensor<256x512x7x7xf32>
    %v2100 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2101 = stablehlo.reshape %v2084 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2102 = stablehlo.multiply %v2100, %v2101 : tensor<256x512x7x7xf32>
    %v2103 = stablehlo.reduce(%v2102 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2104 = stablehlo.broadcast_in_dim %v2103, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2105 = stablehlo.multiply %v2099, %v2102 : tensor<256x512x7x7xf32>
    %v2106 = stablehlo.reduce(%v2105 init: %v2086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2107 = stablehlo.broadcast_in_dim %v2106, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2108 = stablehlo.multiply %v2102, %v2087 : tensor<256x512x7x7xf32>
    %v2109 = stablehlo.subtract %v2108, %v2104 : tensor<256x512x7x7xf32>
    %v2110 = stablehlo.multiply %v2099, %v2107 : tensor<256x512x7x7xf32>
    %v2111 = stablehlo.subtract %v2109, %v2110 : tensor<256x512x7x7xf32>
    %v2112 = stablehlo.divide %v2098, %v2087 : tensor<256x512x7x7xf32>
    %v2113 = stablehlo.multiply %v2112, %v2111 : tensor<256x512x7x7xf32>
    %v2114 = stablehlo.reshape %v2113 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2116 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2117 = stablehlo.transpose %v2116, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2118 = stablehlo.convolution(%v2115, %v2117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x14x14xf32>
    %v2119 = stablehlo.reshape %v2118 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2121 = stablehlo.reshape %v1315 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2122 = stablehlo.constant dense<0.0> : tensor<256x512x14x14xf32>
    %v2123 = stablehlo.compare GT, %v2121, %v2122 : (tensor<256x512x14x14xf32>, tensor<256x512x14x14xf32>) -> tensor<256x512x14x14xi1>
    %v2124 = stablehlo.select %v2123, %v2120, %v2122 : tensor<256x512x14x14xi1>, tensor<256x512x14x14xf32>
    %v2125 = stablehlo.reshape %v2124 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2126 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2128 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v2129 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v2130 = stablehlo.reduce(%v2126 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2130, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2132 = stablehlo.divide %v2131, %v2128 : tensor<256x512x14x14xf32>
    %v2133 = stablehlo.subtract %v2126, %v2132 : tensor<256x512x14x14xf32>
    %v2134 = stablehlo.multiply %v2133, %v2133 : tensor<256x512x14x14xf32>
    %v2135 = stablehlo.reduce(%v2134 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2136 = stablehlo.broadcast_in_dim %v2135, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2137 = stablehlo.divide %v2136, %v2128 : tensor<256x512x14x14xf32>
    %v2138 = stablehlo.add %v2137, %v2129 : tensor<256x512x14x14xf32>
    %v2139 = stablehlo.rsqrt %v2138 : tensor<256x512x14x14xf32>
    %v2140 = stablehlo.multiply %v2133, %v2139 : tensor<256x512x14x14xf32>
    %v2141 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2142 = stablehlo.reshape %v2125 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2143 = stablehlo.multiply %v2141, %v2142 : tensor<256x512x14x14xf32>
    %v2144 = stablehlo.reduce(%v2143 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2145 = stablehlo.broadcast_in_dim %v2144, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2146 = stablehlo.multiply %v2140, %v2143 : tensor<256x512x14x14xf32>
    %v2147 = stablehlo.reduce(%v2146 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2148 = stablehlo.broadcast_in_dim %v2147, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2149 = stablehlo.multiply %v2143, %v2128 : tensor<256x512x14x14xf32>
    %v2150 = stablehlo.subtract %v2149, %v2145 : tensor<256x512x14x14xf32>
    %v2151 = stablehlo.multiply %v2140, %v2148 : tensor<256x512x14x14xf32>
    %v2152 = stablehlo.subtract %v2150, %v2151 : tensor<256x512x14x14xf32>
    %v2153 = stablehlo.divide %v2139, %v2128 : tensor<256x512x14x14xf32>
    %v2154 = stablehlo.multiply %v2153, %v2152 : tensor<256x512x14x14xf32>
    %v2155 = stablehlo.reshape %v2154 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2156 = stablehlo.reshape %v2155 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2157 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v2158 = stablehlo.transpose %v2157, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v2159 = stablehlo.convolution(%v2156, %v2158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2160 = stablehlo.reshape %v2159 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2161 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2163 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2164 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2165 = stablehlo.reduce(%v2161 init: %v2162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2166 = stablehlo.broadcast_in_dim %v2165, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2167 = stablehlo.divide %v2166, %v2163 : tensor<256x2048x7x7xf32>
    %v2168 = stablehlo.subtract %v2161, %v2167 : tensor<256x2048x7x7xf32>
    %v2169 = stablehlo.multiply %v2168, %v2168 : tensor<256x2048x7x7xf32>
    %v2170 = stablehlo.reduce(%v2169 init: %v2162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2171 = stablehlo.broadcast_in_dim %v2170, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2172 = stablehlo.divide %v2171, %v2163 : tensor<256x2048x7x7xf32>
    %v2173 = stablehlo.add %v2172, %v2164 : tensor<256x2048x7x7xf32>
    %v2174 = stablehlo.rsqrt %v2173 : tensor<256x2048x7x7xf32>
    %v2175 = stablehlo.multiply %v2168, %v2174 : tensor<256x2048x7x7xf32>
    %v2176 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2177 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2178 = stablehlo.multiply %v2176, %v2177 : tensor<256x2048x7x7xf32>
    %v2179 = stablehlo.reduce(%v2178 init: %v2162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2180 = stablehlo.broadcast_in_dim %v2179, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2181 = stablehlo.multiply %v2175, %v2178 : tensor<256x2048x7x7xf32>
    %v2182 = stablehlo.reduce(%v2181 init: %v2162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2183 = stablehlo.broadcast_in_dim %v2182, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2184 = stablehlo.multiply %v2178, %v2163 : tensor<256x2048x7x7xf32>
    %v2185 = stablehlo.subtract %v2184, %v2180 : tensor<256x2048x7x7xf32>
    %v2186 = stablehlo.multiply %v2175, %v2183 : tensor<256x2048x7x7xf32>
    %v2187 = stablehlo.subtract %v2185, %v2186 : tensor<256x2048x7x7xf32>
    %v2188 = stablehlo.divide %v2174, %v2163 : tensor<256x2048x7x7xf32>
    %v2189 = stablehlo.multiply %v2188, %v2187 : tensor<256x2048x7x7xf32>
    %v2190 = stablehlo.reshape %v2189 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v2191 = stablehlo.reshape %v2190 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2192 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2193 = stablehlo.transpose %v2192, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2194 = stablehlo.convolution(%v2191, %v2193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<1024x2048x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2196 = stablehlo.reshape %v2160 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2197 = stablehlo.reshape %v2195 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2198 = stablehlo.add %v2196, %v2197 : tensor<256x1024x14x14xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2200 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2201 = stablehlo.reshape %v2155 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2202 = stablehlo.transpose %v2200, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2203 = stablehlo.transpose %v2201, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2204 = stablehlo.convolution(%v2202, %v2203)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<1024x512x1x1xf32>
    %v2205 = stablehlo.transpose %v2204, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2206 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2208 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v2209 = stablehlo.reduce(%v2206 init: %v2207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2210 = stablehlo.broadcast_in_dim %v2209, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2211 = stablehlo.divide %v2210, %v2208 : tensor<256x512x14x14xf32>
    %v2212 = stablehlo.subtract %v2206, %v2211 : tensor<256x512x14x14xf32>
    %v2213 = stablehlo.multiply %v2212, %v2212 : tensor<256x512x14x14xf32>
    %v2214 = stablehlo.reduce(%v2213 init: %v2207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2215 = stablehlo.broadcast_in_dim %v2214, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2216 = stablehlo.divide %v2215, %v2208 : tensor<256x512x14x14xf32>
    %v2217 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v2218 = stablehlo.add %v2216, %v2217 : tensor<256x512x14x14xf32>
    %v2219 = stablehlo.rsqrt %v2218 : tensor<256x512x14x14xf32>
    %v2220 = stablehlo.multiply %v2212, %v2219 : tensor<256x512x14x14xf32>
    %v2221 = stablehlo.reshape %v2125 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2222 = stablehlo.multiply %v2221, %v2220 : tensor<256x512x14x14xf32>
    %v2223 = stablehlo.reduce(%v2222 init: %v2207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2224 = stablehlo.reshape %v2125 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2226 = stablehlo.reduce(%v2224 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2227 = stablehlo.reshape %v1319 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2228 = stablehlo.reshape %v2114 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2229 = stablehlo.transpose %v2227, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2231 = stablehlo.transpose %v2228, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v2232 = stablehlo.convolution(%v2229, %v2231)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x14x14xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v2233 = stablehlo.transpose %v2232, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2234 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2236 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v2237 = stablehlo.reduce(%v2234 init: %v2235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2238 = stablehlo.broadcast_in_dim %v2237, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2239 = stablehlo.divide %v2238, %v2236 : tensor<256x512x7x7xf32>
    %v2240 = stablehlo.subtract %v2234, %v2239 : tensor<256x512x7x7xf32>
    %v2241 = stablehlo.multiply %v2240, %v2240 : tensor<256x512x7x7xf32>
    %v2242 = stablehlo.reduce(%v2241 init: %v2235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2244 = stablehlo.divide %v2243, %v2236 : tensor<256x512x7x7xf32>
    %v2245 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v2246 = stablehlo.add %v2244, %v2245 : tensor<256x512x7x7xf32>
    %v2247 = stablehlo.rsqrt %v2246 : tensor<256x512x7x7xf32>
    %v2248 = stablehlo.multiply %v2240, %v2247 : tensor<256x512x7x7xf32>
    %v2249 = stablehlo.reshape %v2084 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2250 = stablehlo.multiply %v2249, %v2248 : tensor<256x512x7x7xf32>
    %v2251 = stablehlo.reduce(%v2250 init: %v2235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2252 = stablehlo.reshape %v2084 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2254 = stablehlo.reduce(%v2252 init: %v2253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2255 = stablehlo.reshape %v1348 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2256 = stablehlo.reshape %v2073 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2257 = stablehlo.transpose %v2255, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v2258 = stablehlo.transpose %v2256, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v2259 = stablehlo.convolution(%v2257, %v2258)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2260 = stablehlo.transpose %v2259, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2261 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2262 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2263 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2264 = stablehlo.reduce(%v2261 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2265 = stablehlo.broadcast_in_dim %v2264, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2266 = stablehlo.divide %v2265, %v2263 : tensor<256x2048x7x7xf32>
    %v2267 = stablehlo.subtract %v2261, %v2266 : tensor<256x2048x7x7xf32>
    %v2268 = stablehlo.multiply %v2267, %v2267 : tensor<256x2048x7x7xf32>
    %v2269 = stablehlo.reduce(%v2268 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2271 = stablehlo.divide %v2270, %v2263 : tensor<256x2048x7x7xf32>
    %v2272 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2273 = stablehlo.add %v2271, %v2272 : tensor<256x2048x7x7xf32>
    %v2274 = stablehlo.rsqrt %v2273 : tensor<256x2048x7x7xf32>
    %v2275 = stablehlo.multiply %v2267, %v2274 : tensor<256x2048x7x7xf32>
    %v2276 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2277 = stablehlo.multiply %v2276, %v2275 : tensor<256x2048x7x7xf32>
    %v2278 = stablehlo.reduce(%v2277 init: %v2262) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2279 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2282 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2283 = stablehlo.reshape %v2190 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2284 = stablehlo.transpose %v2282, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2286 = stablehlo.transpose %v2283, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v2285 = stablehlo.slice %v2284 [0:1024, 0:256, 0:13, 0:13] : (tensor<1024x256x14x14xf32>) -> tensor<1024x256x13x13xf32>
    %v2287 = stablehlo.convolution(%v2285, %v2286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x13x13xf32>, tensor<2048x256x7x7xf32>) -> tensor<1024x2048x1x1xf32>
    %v2288 = stablehlo.transpose %v2287, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v2289 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2291 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2292 = stablehlo.reduce(%v2289 init: %v2290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2293 = stablehlo.broadcast_in_dim %v2292, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2294 = stablehlo.divide %v2293, %v2291 : tensor<256x2048x7x7xf32>
    %v2295 = stablehlo.subtract %v2289, %v2294 : tensor<256x2048x7x7xf32>
    %v2296 = stablehlo.multiply %v2295, %v2295 : tensor<256x2048x7x7xf32>
    %v2297 = stablehlo.reduce(%v2296 init: %v2290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2298 = stablehlo.broadcast_in_dim %v2297, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2299 = stablehlo.divide %v2298, %v2291 : tensor<256x2048x7x7xf32>
    %v2300 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2301 = stablehlo.add %v2299, %v2300 : tensor<256x2048x7x7xf32>
    %v2302 = stablehlo.rsqrt %v2301 : tensor<256x2048x7x7xf32>
    %v2303 = stablehlo.multiply %v2295, %v2302 : tensor<256x2048x7x7xf32>
    %v2304 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2305 = stablehlo.multiply %v2304, %v2303 : tensor<256x2048x7x7xf32>
    %v2306 = stablehlo.reduce(%v2305 init: %v2290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2307 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2309 = stablehlo.reduce(%v2307 init: %v2308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2310 = stablehlo.reshape %v2199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2311 = stablehlo.reshape %v1286 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2312 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2313 = stablehlo.compare GT, %v2311, %v2312 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2314 = stablehlo.select %v2313, %v2310, %v2312 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2315 = stablehlo.reshape %v2314 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2316 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2318 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2319 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2320 = stablehlo.reduce(%v2316 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2321 = stablehlo.broadcast_in_dim %v2320, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2322 = stablehlo.divide %v2321, %v2318 : tensor<256x1024x14x14xf32>
    %v2323 = stablehlo.subtract %v2316, %v2322 : tensor<256x1024x14x14xf32>
    %v2324 = stablehlo.multiply %v2323, %v2323 : tensor<256x1024x14x14xf32>
    %v2325 = stablehlo.reduce(%v2324 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2326 = stablehlo.broadcast_in_dim %v2325, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2327 = stablehlo.divide %v2326, %v2318 : tensor<256x1024x14x14xf32>
    %v2328 = stablehlo.add %v2327, %v2319 : tensor<256x1024x14x14xf32>
    %v2329 = stablehlo.rsqrt %v2328 : tensor<256x1024x14x14xf32>
    %v2330 = stablehlo.multiply %v2323, %v2329 : tensor<256x1024x14x14xf32>
    %v2331 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2332 = stablehlo.reshape %v2315 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2333 = stablehlo.multiply %v2331, %v2332 : tensor<256x1024x14x14xf32>
    %v2334 = stablehlo.reduce(%v2333 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2335 = stablehlo.broadcast_in_dim %v2334, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2336 = stablehlo.multiply %v2330, %v2333 : tensor<256x1024x14x14xf32>
    %v2337 = stablehlo.reduce(%v2336 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2338 = stablehlo.broadcast_in_dim %v2337, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2339 = stablehlo.multiply %v2333, %v2318 : tensor<256x1024x14x14xf32>
    %v2340 = stablehlo.subtract %v2339, %v2335 : tensor<256x1024x14x14xf32>
    %v2341 = stablehlo.multiply %v2330, %v2338 : tensor<256x1024x14x14xf32>
    %v2342 = stablehlo.subtract %v2340, %v2341 : tensor<256x1024x14x14xf32>
    %v2343 = stablehlo.divide %v2329, %v2318 : tensor<256x1024x14x14xf32>
    %v2344 = stablehlo.multiply %v2343, %v2342 : tensor<256x1024x14x14xf32>
    %v2345 = stablehlo.reshape %v2344 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2346 = stablehlo.reshape %v2345 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2347 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2348 = stablehlo.transpose %v2347, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2349 = stablehlo.convolution(%v2346, %v2348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2350 = stablehlo.reshape %v2349 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2352 = stablehlo.reshape %v1253 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2353 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2354 = stablehlo.compare GT, %v2352, %v2353 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2355 = stablehlo.select %v2354, %v2351, %v2353 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2357 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2359 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2360 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2361 = stablehlo.reduce(%v2357 init: %v2358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2362 = stablehlo.broadcast_in_dim %v2361, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2363 = stablehlo.divide %v2362, %v2359 : tensor<256x256x14x14xf32>
    %v2364 = stablehlo.subtract %v2357, %v2363 : tensor<256x256x14x14xf32>
    %v2365 = stablehlo.multiply %v2364, %v2364 : tensor<256x256x14x14xf32>
    %v2366 = stablehlo.reduce(%v2365 init: %v2358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2366, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2368 = stablehlo.divide %v2367, %v2359 : tensor<256x256x14x14xf32>
    %v2369 = stablehlo.add %v2368, %v2360 : tensor<256x256x14x14xf32>
    %v2370 = stablehlo.rsqrt %v2369 : tensor<256x256x14x14xf32>
    %v2371 = stablehlo.multiply %v2364, %v2370 : tensor<256x256x14x14xf32>
    %v2372 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2373 = stablehlo.reshape %v2356 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2374 = stablehlo.multiply %v2372, %v2373 : tensor<256x256x14x14xf32>
    %v2375 = stablehlo.reduce(%v2374 init: %v2358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2376 = stablehlo.broadcast_in_dim %v2375, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2377 = stablehlo.multiply %v2371, %v2374 : tensor<256x256x14x14xf32>
    %v2378 = stablehlo.reduce(%v2377 init: %v2358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2379 = stablehlo.broadcast_in_dim %v2378, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2380 = stablehlo.multiply %v2374, %v2359 : tensor<256x256x14x14xf32>
    %v2381 = stablehlo.subtract %v2380, %v2376 : tensor<256x256x14x14xf32>
    %v2382 = stablehlo.multiply %v2371, %v2379 : tensor<256x256x14x14xf32>
    %v2383 = stablehlo.subtract %v2381, %v2382 : tensor<256x256x14x14xf32>
    %v2384 = stablehlo.divide %v2370, %v2359 : tensor<256x256x14x14xf32>
    %v2385 = stablehlo.multiply %v2384, %v2383 : tensor<256x256x14x14xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2388 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2389 = stablehlo.transpose %v2388, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2390 = stablehlo.convolution(%v2387, %v2389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2391 = stablehlo.reshape %v2390 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2393 = stablehlo.reshape %v1224 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2394 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2395 = stablehlo.compare GT, %v2393, %v2394 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2396 = stablehlo.select %v2395, %v2392, %v2394 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2398 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2400 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2401 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2402 = stablehlo.reduce(%v2398 init: %v2399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2403 = stablehlo.broadcast_in_dim %v2402, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2404 = stablehlo.divide %v2403, %v2400 : tensor<256x256x14x14xf32>
    %v2405 = stablehlo.subtract %v2398, %v2404 : tensor<256x256x14x14xf32>
    %v2406 = stablehlo.multiply %v2405, %v2405 : tensor<256x256x14x14xf32>
    %v2407 = stablehlo.reduce(%v2406 init: %v2399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2408 = stablehlo.broadcast_in_dim %v2407, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2409 = stablehlo.divide %v2408, %v2400 : tensor<256x256x14x14xf32>
    %v2410 = stablehlo.add %v2409, %v2401 : tensor<256x256x14x14xf32>
    %v2411 = stablehlo.rsqrt %v2410 : tensor<256x256x14x14xf32>
    %v2412 = stablehlo.multiply %v2405, %v2411 : tensor<256x256x14x14xf32>
    %v2413 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2414 = stablehlo.reshape %v2397 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2415 = stablehlo.multiply %v2413, %v2414 : tensor<256x256x14x14xf32>
    %v2416 = stablehlo.reduce(%v2415 init: %v2399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2417 = stablehlo.broadcast_in_dim %v2416, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2418 = stablehlo.multiply %v2412, %v2415 : tensor<256x256x14x14xf32>
    %v2419 = stablehlo.reduce(%v2418 init: %v2399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2420 = stablehlo.broadcast_in_dim %v2419, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2421 = stablehlo.multiply %v2415, %v2400 : tensor<256x256x14x14xf32>
    %v2422 = stablehlo.subtract %v2421, %v2417 : tensor<256x256x14x14xf32>
    %v2423 = stablehlo.multiply %v2412, %v2420 : tensor<256x256x14x14xf32>
    %v2424 = stablehlo.subtract %v2422, %v2423 : tensor<256x256x14x14xf32>
    %v2425 = stablehlo.divide %v2411, %v2400 : tensor<256x256x14x14xf32>
    %v2426 = stablehlo.multiply %v2425, %v2424 : tensor<256x256x14x14xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2428 = stablehlo.reshape %v2427 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2429 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2430 = stablehlo.transpose %v2429, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2431 = stablehlo.convolution(%v2428, %v2430)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2432 = stablehlo.reshape %v2431 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2434 = stablehlo.reshape %v2315 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2435 = stablehlo.add %v2433, %v2434 : tensor<256x1024x14x14xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2437 = stablehlo.reshape %v1199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2438 = stablehlo.reshape %v2427 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2439 = stablehlo.transpose %v2437, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2440 = stablehlo.transpose %v2438, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2441 = stablehlo.convolution(%v2439, %v2440)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2442 = stablehlo.transpose %v2441, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2443 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2445 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2446 = stablehlo.reduce(%v2443 init: %v2444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2446, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2448 = stablehlo.divide %v2447, %v2445 : tensor<256x256x14x14xf32>
    %v2449 = stablehlo.subtract %v2443, %v2448 : tensor<256x256x14x14xf32>
    %v2450 = stablehlo.multiply %v2449, %v2449 : tensor<256x256x14x14xf32>
    %v2451 = stablehlo.reduce(%v2450 init: %v2444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2452 = stablehlo.broadcast_in_dim %v2451, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2453 = stablehlo.divide %v2452, %v2445 : tensor<256x256x14x14xf32>
    %v2454 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2455 = stablehlo.add %v2453, %v2454 : tensor<256x256x14x14xf32>
    %v2456 = stablehlo.rsqrt %v2455 : tensor<256x256x14x14xf32>
    %v2457 = stablehlo.multiply %v2449, %v2456 : tensor<256x256x14x14xf32>
    %v2458 = stablehlo.reshape %v2397 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2459 = stablehlo.multiply %v2458, %v2457 : tensor<256x256x14x14xf32>
    %v2460 = stablehlo.reduce(%v2459 init: %v2444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2461 = stablehlo.reshape %v2397 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2463 = stablehlo.reduce(%v2461 init: %v2462) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2464 = stablehlo.reshape %v1228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2465 = stablehlo.reshape %v2386 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2466 = stablehlo.transpose %v2464, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2467 = stablehlo.transpose %v2465, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2468 = stablehlo.convolution(%v2466, %v2467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2469 = stablehlo.transpose %v2468, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2470 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2472 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2473 = stablehlo.reduce(%v2470 init: %v2471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2474 = stablehlo.broadcast_in_dim %v2473, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2475 = stablehlo.divide %v2474, %v2472 : tensor<256x256x14x14xf32>
    %v2476 = stablehlo.subtract %v2470, %v2475 : tensor<256x256x14x14xf32>
    %v2477 = stablehlo.multiply %v2476, %v2476 : tensor<256x256x14x14xf32>
    %v2478 = stablehlo.reduce(%v2477 init: %v2471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2478, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2480 = stablehlo.divide %v2479, %v2472 : tensor<256x256x14x14xf32>
    %v2481 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2482 = stablehlo.add %v2480, %v2481 : tensor<256x256x14x14xf32>
    %v2483 = stablehlo.rsqrt %v2482 : tensor<256x256x14x14xf32>
    %v2484 = stablehlo.multiply %v2476, %v2483 : tensor<256x256x14x14xf32>
    %v2485 = stablehlo.reshape %v2356 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2486 = stablehlo.multiply %v2485, %v2484 : tensor<256x256x14x14xf32>
    %v2487 = stablehlo.reduce(%v2486 init: %v2471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2488 = stablehlo.reshape %v2356 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.reduce(%v2488 init: %v2489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2491 = stablehlo.reshape %v1257 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2492 = stablehlo.reshape %v2345 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2493 = stablehlo.transpose %v2491, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2494 = stablehlo.transpose %v2492, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2495 = stablehlo.convolution(%v2493, %v2494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2496 = stablehlo.transpose %v2495, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2497 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2499 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2500 = stablehlo.reduce(%v2497 init: %v2498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2501 = stablehlo.broadcast_in_dim %v2500, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2502 = stablehlo.divide %v2501, %v2499 : tensor<256x1024x14x14xf32>
    %v2503 = stablehlo.subtract %v2497, %v2502 : tensor<256x1024x14x14xf32>
    %v2504 = stablehlo.multiply %v2503, %v2503 : tensor<256x1024x14x14xf32>
    %v2505 = stablehlo.reduce(%v2504 init: %v2498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2506 = stablehlo.broadcast_in_dim %v2505, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2507 = stablehlo.divide %v2506, %v2499 : tensor<256x1024x14x14xf32>
    %v2508 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2509 = stablehlo.add %v2507, %v2508 : tensor<256x1024x14x14xf32>
    %v2510 = stablehlo.rsqrt %v2509 : tensor<256x1024x14x14xf32>
    %v2511 = stablehlo.multiply %v2503, %v2510 : tensor<256x1024x14x14xf32>
    %v2512 = stablehlo.reshape %v2315 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2513 = stablehlo.multiply %v2512, %v2511 : tensor<256x1024x14x14xf32>
    %v2514 = stablehlo.reduce(%v2513 init: %v2498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2515 = stablehlo.reshape %v2315 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2517 = stablehlo.reduce(%v2515 init: %v2516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2518 = stablehlo.reshape %v2436 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2519 = stablehlo.reshape %v1195 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2520 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2521 = stablehlo.compare GT, %v2519, %v2520 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2522 = stablehlo.select %v2521, %v2518, %v2520 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2523 = stablehlo.reshape %v2522 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2524 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2526 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2527 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2528 = stablehlo.reduce(%v2524 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2528, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2530 = stablehlo.divide %v2529, %v2526 : tensor<256x1024x14x14xf32>
    %v2531 = stablehlo.subtract %v2524, %v2530 : tensor<256x1024x14x14xf32>
    %v2532 = stablehlo.multiply %v2531, %v2531 : tensor<256x1024x14x14xf32>
    %v2533 = stablehlo.reduce(%v2532 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2534 = stablehlo.broadcast_in_dim %v2533, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2535 = stablehlo.divide %v2534, %v2526 : tensor<256x1024x14x14xf32>
    %v2536 = stablehlo.add %v2535, %v2527 : tensor<256x1024x14x14xf32>
    %v2537 = stablehlo.rsqrt %v2536 : tensor<256x1024x14x14xf32>
    %v2538 = stablehlo.multiply %v2531, %v2537 : tensor<256x1024x14x14xf32>
    %v2539 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2540 = stablehlo.reshape %v2523 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2541 = stablehlo.multiply %v2539, %v2540 : tensor<256x1024x14x14xf32>
    %v2542 = stablehlo.reduce(%v2541 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2543 = stablehlo.broadcast_in_dim %v2542, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2544 = stablehlo.multiply %v2538, %v2541 : tensor<256x1024x14x14xf32>
    %v2545 = stablehlo.reduce(%v2544 init: %v2525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2547 = stablehlo.multiply %v2541, %v2526 : tensor<256x1024x14x14xf32>
    %v2548 = stablehlo.subtract %v2547, %v2543 : tensor<256x1024x14x14xf32>
    %v2549 = stablehlo.multiply %v2538, %v2546 : tensor<256x1024x14x14xf32>
    %v2550 = stablehlo.subtract %v2548, %v2549 : tensor<256x1024x14x14xf32>
    %v2551 = stablehlo.divide %v2537, %v2526 : tensor<256x1024x14x14xf32>
    %v2552 = stablehlo.multiply %v2551, %v2550 : tensor<256x1024x14x14xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2554 = stablehlo.reshape %v2553 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2555 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2556 = stablehlo.transpose %v2555, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2557 = stablehlo.convolution(%v2554, %v2556)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2560 = stablehlo.reshape %v1162 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2561 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2562 = stablehlo.compare GT, %v2560, %v2561 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2563 = stablehlo.select %v2562, %v2559, %v2561 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2565 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2567 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2568 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2569 = stablehlo.reduce(%v2565 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2570 = stablehlo.broadcast_in_dim %v2569, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2571 = stablehlo.divide %v2570, %v2567 : tensor<256x256x14x14xf32>
    %v2572 = stablehlo.subtract %v2565, %v2571 : tensor<256x256x14x14xf32>
    %v2573 = stablehlo.multiply %v2572, %v2572 : tensor<256x256x14x14xf32>
    %v2574 = stablehlo.reduce(%v2573 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2575 = stablehlo.broadcast_in_dim %v2574, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2576 = stablehlo.divide %v2575, %v2567 : tensor<256x256x14x14xf32>
    %v2577 = stablehlo.add %v2576, %v2568 : tensor<256x256x14x14xf32>
    %v2578 = stablehlo.rsqrt %v2577 : tensor<256x256x14x14xf32>
    %v2579 = stablehlo.multiply %v2572, %v2578 : tensor<256x256x14x14xf32>
    %v2580 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2581 = stablehlo.reshape %v2564 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2582 = stablehlo.multiply %v2580, %v2581 : tensor<256x256x14x14xf32>
    %v2583 = stablehlo.reduce(%v2582 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2584 = stablehlo.broadcast_in_dim %v2583, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2585 = stablehlo.multiply %v2579, %v2582 : tensor<256x256x14x14xf32>
    %v2586 = stablehlo.reduce(%v2585 init: %v2566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2586, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2588 = stablehlo.multiply %v2582, %v2567 : tensor<256x256x14x14xf32>
    %v2589 = stablehlo.subtract %v2588, %v2584 : tensor<256x256x14x14xf32>
    %v2590 = stablehlo.multiply %v2579, %v2587 : tensor<256x256x14x14xf32>
    %v2591 = stablehlo.subtract %v2589, %v2590 : tensor<256x256x14x14xf32>
    %v2592 = stablehlo.divide %v2578, %v2567 : tensor<256x256x14x14xf32>
    %v2593 = stablehlo.multiply %v2592, %v2591 : tensor<256x256x14x14xf32>
    %v2594 = stablehlo.reshape %v2593 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2595 = stablehlo.reshape %v2594 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2596 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2597 = stablehlo.transpose %v2596, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2598 = stablehlo.convolution(%v2595, %v2597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2601 = stablehlo.reshape %v1133 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2602 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2603 = stablehlo.compare GT, %v2601, %v2602 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2604 = stablehlo.select %v2603, %v2600, %v2602 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2606 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2608 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2609 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2610 = stablehlo.reduce(%v2606 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2611 = stablehlo.broadcast_in_dim %v2610, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2612 = stablehlo.divide %v2611, %v2608 : tensor<256x256x14x14xf32>
    %v2613 = stablehlo.subtract %v2606, %v2612 : tensor<256x256x14x14xf32>
    %v2614 = stablehlo.multiply %v2613, %v2613 : tensor<256x256x14x14xf32>
    %v2615 = stablehlo.reduce(%v2614 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2616 = stablehlo.broadcast_in_dim %v2615, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2617 = stablehlo.divide %v2616, %v2608 : tensor<256x256x14x14xf32>
    %v2618 = stablehlo.add %v2617, %v2609 : tensor<256x256x14x14xf32>
    %v2619 = stablehlo.rsqrt %v2618 : tensor<256x256x14x14xf32>
    %v2620 = stablehlo.multiply %v2613, %v2619 : tensor<256x256x14x14xf32>
    %v2621 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2622 = stablehlo.reshape %v2605 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2623 = stablehlo.multiply %v2621, %v2622 : tensor<256x256x14x14xf32>
    %v2624 = stablehlo.reduce(%v2623 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2625 = stablehlo.broadcast_in_dim %v2624, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2626 = stablehlo.multiply %v2620, %v2623 : tensor<256x256x14x14xf32>
    %v2627 = stablehlo.reduce(%v2626 init: %v2607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2628 = stablehlo.broadcast_in_dim %v2627, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2629 = stablehlo.multiply %v2623, %v2608 : tensor<256x256x14x14xf32>
    %v2630 = stablehlo.subtract %v2629, %v2625 : tensor<256x256x14x14xf32>
    %v2631 = stablehlo.multiply %v2620, %v2628 : tensor<256x256x14x14xf32>
    %v2632 = stablehlo.subtract %v2630, %v2631 : tensor<256x256x14x14xf32>
    %v2633 = stablehlo.divide %v2619, %v2608 : tensor<256x256x14x14xf32>
    %v2634 = stablehlo.multiply %v2633, %v2632 : tensor<256x256x14x14xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2636 = stablehlo.reshape %v2635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2637 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2638 = stablehlo.transpose %v2637, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2639 = stablehlo.convolution(%v2636, %v2638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2640 = stablehlo.reshape %v2639 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2642 = stablehlo.reshape %v2523 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2643 = stablehlo.add %v2641, %v2642 : tensor<256x1024x14x14xf32>
    %v2644 = stablehlo.reshape %v2643 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2645 = stablehlo.reshape %v1108 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2646 = stablehlo.reshape %v2635 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2647 = stablehlo.transpose %v2645, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2648 = stablehlo.transpose %v2646, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2649 = stablehlo.convolution(%v2647, %v2648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2650 = stablehlo.transpose %v2649, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2651 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2653 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2654 = stablehlo.reduce(%v2651 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2655 = stablehlo.broadcast_in_dim %v2654, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2656 = stablehlo.divide %v2655, %v2653 : tensor<256x256x14x14xf32>
    %v2657 = stablehlo.subtract %v2651, %v2656 : tensor<256x256x14x14xf32>
    %v2658 = stablehlo.multiply %v2657, %v2657 : tensor<256x256x14x14xf32>
    %v2659 = stablehlo.reduce(%v2658 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2659, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2661 = stablehlo.divide %v2660, %v2653 : tensor<256x256x14x14xf32>
    %v2662 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2663 = stablehlo.add %v2661, %v2662 : tensor<256x256x14x14xf32>
    %v2664 = stablehlo.rsqrt %v2663 : tensor<256x256x14x14xf32>
    %v2665 = stablehlo.multiply %v2657, %v2664 : tensor<256x256x14x14xf32>
    %v2666 = stablehlo.reshape %v2605 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2667 = stablehlo.multiply %v2666, %v2665 : tensor<256x256x14x14xf32>
    %v2668 = stablehlo.reduce(%v2667 init: %v2652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2669 = stablehlo.reshape %v2605 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2671 = stablehlo.reduce(%v2669 init: %v2670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2672 = stablehlo.reshape %v1137 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2673 = stablehlo.reshape %v2594 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2674 = stablehlo.transpose %v2672, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2675 = stablehlo.transpose %v2673, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2676 = stablehlo.convolution(%v2674, %v2675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2677 = stablehlo.transpose %v2676, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2678 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2680 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2681 = stablehlo.reduce(%v2678 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2682 = stablehlo.broadcast_in_dim %v2681, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2683 = stablehlo.divide %v2682, %v2680 : tensor<256x256x14x14xf32>
    %v2684 = stablehlo.subtract %v2678, %v2683 : tensor<256x256x14x14xf32>
    %v2685 = stablehlo.multiply %v2684, %v2684 : tensor<256x256x14x14xf32>
    %v2686 = stablehlo.reduce(%v2685 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2687 = stablehlo.broadcast_in_dim %v2686, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2688 = stablehlo.divide %v2687, %v2680 : tensor<256x256x14x14xf32>
    %v2689 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2690 = stablehlo.add %v2688, %v2689 : tensor<256x256x14x14xf32>
    %v2691 = stablehlo.rsqrt %v2690 : tensor<256x256x14x14xf32>
    %v2692 = stablehlo.multiply %v2684, %v2691 : tensor<256x256x14x14xf32>
    %v2693 = stablehlo.reshape %v2564 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2694 = stablehlo.multiply %v2693, %v2692 : tensor<256x256x14x14xf32>
    %v2695 = stablehlo.reduce(%v2694 init: %v2679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2696 = stablehlo.reshape %v2564 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2698 = stablehlo.reduce(%v2696 init: %v2697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2699 = stablehlo.reshape %v1166 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2700 = stablehlo.reshape %v2553 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2701 = stablehlo.transpose %v2699, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2702 = stablehlo.transpose %v2700, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2703 = stablehlo.convolution(%v2701, %v2702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2704 = stablehlo.transpose %v2703, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2705 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2707 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2708 = stablehlo.reduce(%v2705 init: %v2706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2710 = stablehlo.divide %v2709, %v2707 : tensor<256x1024x14x14xf32>
    %v2711 = stablehlo.subtract %v2705, %v2710 : tensor<256x1024x14x14xf32>
    %v2712 = stablehlo.multiply %v2711, %v2711 : tensor<256x1024x14x14xf32>
    %v2713 = stablehlo.reduce(%v2712 init: %v2706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2715 = stablehlo.divide %v2714, %v2707 : tensor<256x1024x14x14xf32>
    %v2716 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2717 = stablehlo.add %v2715, %v2716 : tensor<256x1024x14x14xf32>
    %v2718 = stablehlo.rsqrt %v2717 : tensor<256x1024x14x14xf32>
    %v2719 = stablehlo.multiply %v2711, %v2718 : tensor<256x1024x14x14xf32>
    %v2720 = stablehlo.reshape %v2523 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2721 = stablehlo.multiply %v2720, %v2719 : tensor<256x1024x14x14xf32>
    %v2722 = stablehlo.reduce(%v2721 init: %v2706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2723 = stablehlo.reshape %v2523 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2725 = stablehlo.reduce(%v2723 init: %v2724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2726 = stablehlo.reshape %v2644 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2727 = stablehlo.reshape %v1104 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2728 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2729 = stablehlo.compare GT, %v2727, %v2728 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2730 = stablehlo.select %v2729, %v2726, %v2728 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2731 = stablehlo.reshape %v2730 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2732 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2734 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2735 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2736 = stablehlo.reduce(%v2732 init: %v2733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2737 = stablehlo.broadcast_in_dim %v2736, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2738 = stablehlo.divide %v2737, %v2734 : tensor<256x1024x14x14xf32>
    %v2739 = stablehlo.subtract %v2732, %v2738 : tensor<256x1024x14x14xf32>
    %v2740 = stablehlo.multiply %v2739, %v2739 : tensor<256x1024x14x14xf32>
    %v2741 = stablehlo.reduce(%v2740 init: %v2733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2742 = stablehlo.broadcast_in_dim %v2741, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2743 = stablehlo.divide %v2742, %v2734 : tensor<256x1024x14x14xf32>
    %v2744 = stablehlo.add %v2743, %v2735 : tensor<256x1024x14x14xf32>
    %v2745 = stablehlo.rsqrt %v2744 : tensor<256x1024x14x14xf32>
    %v2746 = stablehlo.multiply %v2739, %v2745 : tensor<256x1024x14x14xf32>
    %v2747 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2748 = stablehlo.reshape %v2731 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2749 = stablehlo.multiply %v2747, %v2748 : tensor<256x1024x14x14xf32>
    %v2750 = stablehlo.reduce(%v2749 init: %v2733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2751 = stablehlo.broadcast_in_dim %v2750, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2752 = stablehlo.multiply %v2746, %v2749 : tensor<256x1024x14x14xf32>
    %v2753 = stablehlo.reduce(%v2752 init: %v2733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2754 = stablehlo.broadcast_in_dim %v2753, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2755 = stablehlo.multiply %v2749, %v2734 : tensor<256x1024x14x14xf32>
    %v2756 = stablehlo.subtract %v2755, %v2751 : tensor<256x1024x14x14xf32>
    %v2757 = stablehlo.multiply %v2746, %v2754 : tensor<256x1024x14x14xf32>
    %v2758 = stablehlo.subtract %v2756, %v2757 : tensor<256x1024x14x14xf32>
    %v2759 = stablehlo.divide %v2745, %v2734 : tensor<256x1024x14x14xf32>
    %v2760 = stablehlo.multiply %v2759, %v2758 : tensor<256x1024x14x14xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2762 = stablehlo.reshape %v2761 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2763 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2764 = stablehlo.transpose %v2763, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2765 = stablehlo.convolution(%v2762, %v2764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2766 = stablehlo.reshape %v2765 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2767 = stablehlo.reshape %v2766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2768 = stablehlo.reshape %v1071 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2769 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2770 = stablehlo.compare GT, %v2768, %v2769 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2771 = stablehlo.select %v2770, %v2767, %v2769 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2773 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2775 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2776 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2777 = stablehlo.reduce(%v2773 init: %v2774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2778 = stablehlo.broadcast_in_dim %v2777, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2779 = stablehlo.divide %v2778, %v2775 : tensor<256x256x14x14xf32>
    %v2780 = stablehlo.subtract %v2773, %v2779 : tensor<256x256x14x14xf32>
    %v2781 = stablehlo.multiply %v2780, %v2780 : tensor<256x256x14x14xf32>
    %v2782 = stablehlo.reduce(%v2781 init: %v2774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2783 = stablehlo.broadcast_in_dim %v2782, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2784 = stablehlo.divide %v2783, %v2775 : tensor<256x256x14x14xf32>
    %v2785 = stablehlo.add %v2784, %v2776 : tensor<256x256x14x14xf32>
    %v2786 = stablehlo.rsqrt %v2785 : tensor<256x256x14x14xf32>
    %v2787 = stablehlo.multiply %v2780, %v2786 : tensor<256x256x14x14xf32>
    %v2788 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2789 = stablehlo.reshape %v2772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2790 = stablehlo.multiply %v2788, %v2789 : tensor<256x256x14x14xf32>
    %v2791 = stablehlo.reduce(%v2790 init: %v2774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2792 = stablehlo.broadcast_in_dim %v2791, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2793 = stablehlo.multiply %v2787, %v2790 : tensor<256x256x14x14xf32>
    %v2794 = stablehlo.reduce(%v2793 init: %v2774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2795 = stablehlo.broadcast_in_dim %v2794, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2796 = stablehlo.multiply %v2790, %v2775 : tensor<256x256x14x14xf32>
    %v2797 = stablehlo.subtract %v2796, %v2792 : tensor<256x256x14x14xf32>
    %v2798 = stablehlo.multiply %v2787, %v2795 : tensor<256x256x14x14xf32>
    %v2799 = stablehlo.subtract %v2797, %v2798 : tensor<256x256x14x14xf32>
    %v2800 = stablehlo.divide %v2786, %v2775 : tensor<256x256x14x14xf32>
    %v2801 = stablehlo.multiply %v2800, %v2799 : tensor<256x256x14x14xf32>
    %v2802 = stablehlo.reshape %v2801 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2804 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2805 = stablehlo.transpose %v2804, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2806 = stablehlo.convolution(%v2803, %v2805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2807 = stablehlo.reshape %v2806 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2809 = stablehlo.reshape %v1042 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2810 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2811 = stablehlo.compare GT, %v2809, %v2810 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2812 = stablehlo.select %v2811, %v2808, %v2810 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2813 = stablehlo.reshape %v2812 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2814 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2816 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2817 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2818 = stablehlo.reduce(%v2814 init: %v2815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2818, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2820 = stablehlo.divide %v2819, %v2816 : tensor<256x256x14x14xf32>
    %v2821 = stablehlo.subtract %v2814, %v2820 : tensor<256x256x14x14xf32>
    %v2822 = stablehlo.multiply %v2821, %v2821 : tensor<256x256x14x14xf32>
    %v2823 = stablehlo.reduce(%v2822 init: %v2815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2823, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2825 = stablehlo.divide %v2824, %v2816 : tensor<256x256x14x14xf32>
    %v2826 = stablehlo.add %v2825, %v2817 : tensor<256x256x14x14xf32>
    %v2827 = stablehlo.rsqrt %v2826 : tensor<256x256x14x14xf32>
    %v2828 = stablehlo.multiply %v2821, %v2827 : tensor<256x256x14x14xf32>
    %v2829 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2830 = stablehlo.reshape %v2813 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2831 = stablehlo.multiply %v2829, %v2830 : tensor<256x256x14x14xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2834 = stablehlo.multiply %v2828, %v2831 : tensor<256x256x14x14xf32>
    %v2835 = stablehlo.reduce(%v2834 init: %v2815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2836 = stablehlo.broadcast_in_dim %v2835, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2837 = stablehlo.multiply %v2831, %v2816 : tensor<256x256x14x14xf32>
    %v2838 = stablehlo.subtract %v2837, %v2833 : tensor<256x256x14x14xf32>
    %v2839 = stablehlo.multiply %v2828, %v2836 : tensor<256x256x14x14xf32>
    %v2840 = stablehlo.subtract %v2838, %v2839 : tensor<256x256x14x14xf32>
    %v2841 = stablehlo.divide %v2827, %v2816 : tensor<256x256x14x14xf32>
    %v2842 = stablehlo.multiply %v2841, %v2840 : tensor<256x256x14x14xf32>
    %v2843 = stablehlo.reshape %v2842 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2844 = stablehlo.reshape %v2843 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2845 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2846 = stablehlo.transpose %v2845, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2847 = stablehlo.convolution(%v2844, %v2846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2850 = stablehlo.reshape %v2731 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2851 = stablehlo.add %v2849, %v2850 : tensor<256x1024x14x14xf32>
    %v2852 = stablehlo.reshape %v2851 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2853 = stablehlo.reshape %v1017 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2854 = stablehlo.reshape %v2843 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2855 = stablehlo.transpose %v2853, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2856 = stablehlo.transpose %v2854, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2857 = stablehlo.convolution(%v2855, %v2856)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2858 = stablehlo.transpose %v2857, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2859 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2861 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2862 = stablehlo.reduce(%v2859 init: %v2860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2863 = stablehlo.broadcast_in_dim %v2862, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2864 = stablehlo.divide %v2863, %v2861 : tensor<256x256x14x14xf32>
    %v2865 = stablehlo.subtract %v2859, %v2864 : tensor<256x256x14x14xf32>
    %v2866 = stablehlo.multiply %v2865, %v2865 : tensor<256x256x14x14xf32>
    %v2867 = stablehlo.reduce(%v2866 init: %v2860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2868 = stablehlo.broadcast_in_dim %v2867, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2869 = stablehlo.divide %v2868, %v2861 : tensor<256x256x14x14xf32>
    %v2870 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2871 = stablehlo.add %v2869, %v2870 : tensor<256x256x14x14xf32>
    %v2872 = stablehlo.rsqrt %v2871 : tensor<256x256x14x14xf32>
    %v2873 = stablehlo.multiply %v2865, %v2872 : tensor<256x256x14x14xf32>
    %v2874 = stablehlo.reshape %v2813 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2875 = stablehlo.multiply %v2874, %v2873 : tensor<256x256x14x14xf32>
    %v2876 = stablehlo.reduce(%v2875 init: %v2860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2877 = stablehlo.reshape %v2813 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2879 = stablehlo.reduce(%v2877 init: %v2878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2880 = stablehlo.reshape %v1046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2881 = stablehlo.reshape %v2802 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2882 = stablehlo.transpose %v2880, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2883 = stablehlo.transpose %v2881, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2884 = stablehlo.convolution(%v2882, %v2883)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2885 = stablehlo.transpose %v2884, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2886 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2888 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2889 = stablehlo.reduce(%v2886 init: %v2887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2890 = stablehlo.broadcast_in_dim %v2889, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2891 = stablehlo.divide %v2890, %v2888 : tensor<256x256x14x14xf32>
    %v2892 = stablehlo.subtract %v2886, %v2891 : tensor<256x256x14x14xf32>
    %v2893 = stablehlo.multiply %v2892, %v2892 : tensor<256x256x14x14xf32>
    %v2894 = stablehlo.reduce(%v2893 init: %v2887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2895 = stablehlo.broadcast_in_dim %v2894, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2896 = stablehlo.divide %v2895, %v2888 : tensor<256x256x14x14xf32>
    %v2897 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2898 = stablehlo.add %v2896, %v2897 : tensor<256x256x14x14xf32>
    %v2899 = stablehlo.rsqrt %v2898 : tensor<256x256x14x14xf32>
    %v2900 = stablehlo.multiply %v2892, %v2899 : tensor<256x256x14x14xf32>
    %v2901 = stablehlo.reshape %v2772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2902 = stablehlo.multiply %v2901, %v2900 : tensor<256x256x14x14xf32>
    %v2903 = stablehlo.reduce(%v2902 init: %v2887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2904 = stablehlo.reshape %v2772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2906 = stablehlo.reduce(%v2904 init: %v2905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2907 = stablehlo.reshape %v1075 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2908 = stablehlo.reshape %v2761 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2909 = stablehlo.transpose %v2907, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2910 = stablehlo.transpose %v2908, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2911 = stablehlo.convolution(%v2909, %v2910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2912 = stablehlo.transpose %v2911, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2913 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2915 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2916 = stablehlo.reduce(%v2913 init: %v2914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2917 = stablehlo.broadcast_in_dim %v2916, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2918 = stablehlo.divide %v2917, %v2915 : tensor<256x1024x14x14xf32>
    %v2919 = stablehlo.subtract %v2913, %v2918 : tensor<256x1024x14x14xf32>
    %v2920 = stablehlo.multiply %v2919, %v2919 : tensor<256x1024x14x14xf32>
    %v2921 = stablehlo.reduce(%v2920 init: %v2914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2922 = stablehlo.broadcast_in_dim %v2921, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2923 = stablehlo.divide %v2922, %v2915 : tensor<256x1024x14x14xf32>
    %v2924 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2925 = stablehlo.add %v2923, %v2924 : tensor<256x1024x14x14xf32>
    %v2926 = stablehlo.rsqrt %v2925 : tensor<256x1024x14x14xf32>
    %v2927 = stablehlo.multiply %v2919, %v2926 : tensor<256x1024x14x14xf32>
    %v2928 = stablehlo.reshape %v2731 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2929 = stablehlo.multiply %v2928, %v2927 : tensor<256x1024x14x14xf32>
    %v2930 = stablehlo.reduce(%v2929 init: %v2914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2931 = stablehlo.reshape %v2731 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2933 = stablehlo.reduce(%v2931 init: %v2932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2934 = stablehlo.reshape %v2852 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2935 = stablehlo.reshape %v1013 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2936 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2937 = stablehlo.compare GT, %v2935, %v2936 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2938 = stablehlo.select %v2937, %v2934, %v2936 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2939 = stablehlo.reshape %v2938 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2940 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2942 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2943 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2944 = stablehlo.reduce(%v2940 init: %v2941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2946 = stablehlo.divide %v2945, %v2942 : tensor<256x1024x14x14xf32>
    %v2947 = stablehlo.subtract %v2940, %v2946 : tensor<256x1024x14x14xf32>
    %v2948 = stablehlo.multiply %v2947, %v2947 : tensor<256x1024x14x14xf32>
    %v2949 = stablehlo.reduce(%v2948 init: %v2941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2950 = stablehlo.broadcast_in_dim %v2949, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2951 = stablehlo.divide %v2950, %v2942 : tensor<256x1024x14x14xf32>
    %v2952 = stablehlo.add %v2951, %v2943 : tensor<256x1024x14x14xf32>
    %v2953 = stablehlo.rsqrt %v2952 : tensor<256x1024x14x14xf32>
    %v2954 = stablehlo.multiply %v2947, %v2953 : tensor<256x1024x14x14xf32>
    %v2955 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2956 = stablehlo.reshape %v2939 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2957 = stablehlo.multiply %v2955, %v2956 : tensor<256x1024x14x14xf32>
    %v2958 = stablehlo.reduce(%v2957 init: %v2941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2959 = stablehlo.broadcast_in_dim %v2958, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2960 = stablehlo.multiply %v2954, %v2957 : tensor<256x1024x14x14xf32>
    %v2961 = stablehlo.reduce(%v2960 init: %v2941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2962 = stablehlo.broadcast_in_dim %v2961, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2963 = stablehlo.multiply %v2957, %v2942 : tensor<256x1024x14x14xf32>
    %v2964 = stablehlo.subtract %v2963, %v2959 : tensor<256x1024x14x14xf32>
    %v2965 = stablehlo.multiply %v2954, %v2962 : tensor<256x1024x14x14xf32>
    %v2966 = stablehlo.subtract %v2964, %v2965 : tensor<256x1024x14x14xf32>
    %v2967 = stablehlo.divide %v2953, %v2942 : tensor<256x1024x14x14xf32>
    %v2968 = stablehlo.multiply %v2967, %v2966 : tensor<256x1024x14x14xf32>
    %v2969 = stablehlo.reshape %v2968 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2970 = stablehlo.reshape %v2969 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2971 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2972 = stablehlo.transpose %v2971, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2973 = stablehlo.convolution(%v2970, %v2972)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2974 = stablehlo.reshape %v2973 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2975 = stablehlo.reshape %v2974 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2976 = stablehlo.reshape %v980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2977 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2978 = stablehlo.compare GT, %v2976, %v2977 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2979 = stablehlo.select %v2978, %v2975, %v2977 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2980 = stablehlo.reshape %v2979 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2981 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2983 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2984 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2985 = stablehlo.reduce(%v2981 init: %v2982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2986 = stablehlo.broadcast_in_dim %v2985, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2987 = stablehlo.divide %v2986, %v2983 : tensor<256x256x14x14xf32>
    %v2988 = stablehlo.subtract %v2981, %v2987 : tensor<256x256x14x14xf32>
    %v2989 = stablehlo.multiply %v2988, %v2988 : tensor<256x256x14x14xf32>
    %v2990 = stablehlo.reduce(%v2989 init: %v2982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2991 = stablehlo.broadcast_in_dim %v2990, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2992 = stablehlo.divide %v2991, %v2983 : tensor<256x256x14x14xf32>
    %v2993 = stablehlo.add %v2992, %v2984 : tensor<256x256x14x14xf32>
    %v2994 = stablehlo.rsqrt %v2993 : tensor<256x256x14x14xf32>
    %v2995 = stablehlo.multiply %v2988, %v2994 : tensor<256x256x14x14xf32>
    %v2996 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2997 = stablehlo.reshape %v2980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2998 = stablehlo.multiply %v2996, %v2997 : tensor<256x256x14x14xf32>
    %v2999 = stablehlo.reduce(%v2998 init: %v2982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3000 = stablehlo.broadcast_in_dim %v2999, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3001 = stablehlo.multiply %v2995, %v2998 : tensor<256x256x14x14xf32>
    %v3002 = stablehlo.reduce(%v3001 init: %v2982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3003 = stablehlo.broadcast_in_dim %v3002, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3004 = stablehlo.multiply %v2998, %v2983 : tensor<256x256x14x14xf32>
    %v3005 = stablehlo.subtract %v3004, %v3000 : tensor<256x256x14x14xf32>
    %v3006 = stablehlo.multiply %v2995, %v3003 : tensor<256x256x14x14xf32>
    %v3007 = stablehlo.subtract %v3005, %v3006 : tensor<256x256x14x14xf32>
    %v3008 = stablehlo.divide %v2994, %v2983 : tensor<256x256x14x14xf32>
    %v3009 = stablehlo.multiply %v3008, %v3007 : tensor<256x256x14x14xf32>
    %v3010 = stablehlo.reshape %v3009 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3011 = stablehlo.reshape %v3010 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3012 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3013 = stablehlo.transpose %v3012, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3014 = stablehlo.convolution(%v3011, %v3013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v3015 = stablehlo.reshape %v3014 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3016 = stablehlo.reshape %v3015 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3017 = stablehlo.reshape %v951 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3018 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3019 = stablehlo.compare GT, %v3017, %v3018 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3020 = stablehlo.select %v3019, %v3016, %v3018 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3021 = stablehlo.reshape %v3020 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3022 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3024 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3025 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3026 = stablehlo.reduce(%v3022 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3027 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3028 = stablehlo.divide %v3027, %v3024 : tensor<256x256x14x14xf32>
    %v3029 = stablehlo.subtract %v3022, %v3028 : tensor<256x256x14x14xf32>
    %v3030 = stablehlo.multiply %v3029, %v3029 : tensor<256x256x14x14xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3032 = stablehlo.broadcast_in_dim %v3031, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3033 = stablehlo.divide %v3032, %v3024 : tensor<256x256x14x14xf32>
    %v3034 = stablehlo.add %v3033, %v3025 : tensor<256x256x14x14xf32>
    %v3035 = stablehlo.rsqrt %v3034 : tensor<256x256x14x14xf32>
    %v3036 = stablehlo.multiply %v3029, %v3035 : tensor<256x256x14x14xf32>
    %v3037 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3038 = stablehlo.reshape %v3021 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3039 = stablehlo.multiply %v3037, %v3038 : tensor<256x256x14x14xf32>
    %v3040 = stablehlo.reduce(%v3039 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3040, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3042 = stablehlo.multiply %v3036, %v3039 : tensor<256x256x14x14xf32>
    %v3043 = stablehlo.reduce(%v3042 init: %v3023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3045 = stablehlo.multiply %v3039, %v3024 : tensor<256x256x14x14xf32>
    %v3046 = stablehlo.subtract %v3045, %v3041 : tensor<256x256x14x14xf32>
    %v3047 = stablehlo.multiply %v3036, %v3044 : tensor<256x256x14x14xf32>
    %v3048 = stablehlo.subtract %v3046, %v3047 : tensor<256x256x14x14xf32>
    %v3049 = stablehlo.divide %v3035, %v3024 : tensor<256x256x14x14xf32>
    %v3050 = stablehlo.multiply %v3049, %v3048 : tensor<256x256x14x14xf32>
    %v3051 = stablehlo.reshape %v3050 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3052 = stablehlo.reshape %v3051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3053 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3054 = stablehlo.transpose %v3053, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3055 = stablehlo.convolution(%v3052, %v3054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3058 = stablehlo.reshape %v2939 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3059 = stablehlo.add %v3057, %v3058 : tensor<256x1024x14x14xf32>
    %v3060 = stablehlo.reshape %v3059 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3061 = stablehlo.reshape %v926 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3062 = stablehlo.reshape %v3051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3063 = stablehlo.transpose %v3061, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3064 = stablehlo.transpose %v3062, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3065 = stablehlo.convolution(%v3063, %v3064)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3066 = stablehlo.transpose %v3065, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3067 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3069 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3070 = stablehlo.reduce(%v3067 init: %v3068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3071 = stablehlo.broadcast_in_dim %v3070, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3072 = stablehlo.divide %v3071, %v3069 : tensor<256x256x14x14xf32>
    %v3073 = stablehlo.subtract %v3067, %v3072 : tensor<256x256x14x14xf32>
    %v3074 = stablehlo.multiply %v3073, %v3073 : tensor<256x256x14x14xf32>
    %v3075 = stablehlo.reduce(%v3074 init: %v3068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3077 = stablehlo.divide %v3076, %v3069 : tensor<256x256x14x14xf32>
    %v3078 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3079 = stablehlo.add %v3077, %v3078 : tensor<256x256x14x14xf32>
    %v3080 = stablehlo.rsqrt %v3079 : tensor<256x256x14x14xf32>
    %v3081 = stablehlo.multiply %v3073, %v3080 : tensor<256x256x14x14xf32>
    %v3082 = stablehlo.reshape %v3021 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3083 = stablehlo.multiply %v3082, %v3081 : tensor<256x256x14x14xf32>
    %v3084 = stablehlo.reduce(%v3083 init: %v3068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3085 = stablehlo.reshape %v3021 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3087 = stablehlo.reduce(%v3085 init: %v3086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3088 = stablehlo.reshape %v955 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3089 = stablehlo.reshape %v3010 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3090 = stablehlo.transpose %v3088, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3091 = stablehlo.transpose %v3089, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3092 = stablehlo.convolution(%v3090, %v3091)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3093 = stablehlo.transpose %v3092, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3094 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3096 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3097 = stablehlo.reduce(%v3094 init: %v3095) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3098 = stablehlo.broadcast_in_dim %v3097, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3099 = stablehlo.divide %v3098, %v3096 : tensor<256x256x14x14xf32>
    %v3100 = stablehlo.subtract %v3094, %v3099 : tensor<256x256x14x14xf32>
    %v3101 = stablehlo.multiply %v3100, %v3100 : tensor<256x256x14x14xf32>
    %v3102 = stablehlo.reduce(%v3101 init: %v3095) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3103 = stablehlo.broadcast_in_dim %v3102, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3104 = stablehlo.divide %v3103, %v3096 : tensor<256x256x14x14xf32>
    %v3105 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3106 = stablehlo.add %v3104, %v3105 : tensor<256x256x14x14xf32>
    %v3107 = stablehlo.rsqrt %v3106 : tensor<256x256x14x14xf32>
    %v3108 = stablehlo.multiply %v3100, %v3107 : tensor<256x256x14x14xf32>
    %v3109 = stablehlo.reshape %v2980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3110 = stablehlo.multiply %v3109, %v3108 : tensor<256x256x14x14xf32>
    %v3111 = stablehlo.reduce(%v3110 init: %v3095) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3112 = stablehlo.reshape %v2980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3114 = stablehlo.reduce(%v3112 init: %v3113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3115 = stablehlo.reshape %v984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3116 = stablehlo.reshape %v2969 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3117 = stablehlo.transpose %v3115, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3118 = stablehlo.transpose %v3116, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3119 = stablehlo.convolution(%v3117, %v3118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3120 = stablehlo.transpose %v3119, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3121 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3123 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3124 = stablehlo.reduce(%v3121 init: %v3122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3125 = stablehlo.broadcast_in_dim %v3124, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3126 = stablehlo.divide %v3125, %v3123 : tensor<256x1024x14x14xf32>
    %v3127 = stablehlo.subtract %v3121, %v3126 : tensor<256x1024x14x14xf32>
    %v3128 = stablehlo.multiply %v3127, %v3127 : tensor<256x1024x14x14xf32>
    %v3129 = stablehlo.reduce(%v3128 init: %v3122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3130 = stablehlo.broadcast_in_dim %v3129, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3131 = stablehlo.divide %v3130, %v3123 : tensor<256x1024x14x14xf32>
    %v3132 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3133 = stablehlo.add %v3131, %v3132 : tensor<256x1024x14x14xf32>
    %v3134 = stablehlo.rsqrt %v3133 : tensor<256x1024x14x14xf32>
    %v3135 = stablehlo.multiply %v3127, %v3134 : tensor<256x1024x14x14xf32>
    %v3136 = stablehlo.reshape %v2939 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3137 = stablehlo.multiply %v3136, %v3135 : tensor<256x1024x14x14xf32>
    %v3138 = stablehlo.reduce(%v3137 init: %v3122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3139 = stablehlo.reshape %v2939 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.reduce(%v3139 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3142 = stablehlo.reshape %v3060 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3143 = stablehlo.reshape %v922 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3144 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v3145 = stablehlo.compare GT, %v3143, %v3144 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v3146 = stablehlo.select %v3145, %v3142, %v3144 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3148 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3150 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3151 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3152 = stablehlo.reduce(%v3148 init: %v3149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3153 = stablehlo.broadcast_in_dim %v3152, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3154 = stablehlo.divide %v3153, %v3150 : tensor<256x1024x14x14xf32>
    %v3155 = stablehlo.subtract %v3148, %v3154 : tensor<256x1024x14x14xf32>
    %v3156 = stablehlo.multiply %v3155, %v3155 : tensor<256x1024x14x14xf32>
    %v3157 = stablehlo.reduce(%v3156 init: %v3149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3158 = stablehlo.broadcast_in_dim %v3157, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3159 = stablehlo.divide %v3158, %v3150 : tensor<256x1024x14x14xf32>
    %v3160 = stablehlo.add %v3159, %v3151 : tensor<256x1024x14x14xf32>
    %v3161 = stablehlo.rsqrt %v3160 : tensor<256x1024x14x14xf32>
    %v3162 = stablehlo.multiply %v3155, %v3161 : tensor<256x1024x14x14xf32>
    %v3163 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3164 = stablehlo.reshape %v3147 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3165 = stablehlo.multiply %v3163, %v3164 : tensor<256x1024x14x14xf32>
    %v3166 = stablehlo.reduce(%v3165 init: %v3149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3167 = stablehlo.broadcast_in_dim %v3166, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3168 = stablehlo.multiply %v3162, %v3165 : tensor<256x1024x14x14xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3171 = stablehlo.multiply %v3165, %v3150 : tensor<256x1024x14x14xf32>
    %v3172 = stablehlo.subtract %v3171, %v3167 : tensor<256x1024x14x14xf32>
    %v3173 = stablehlo.multiply %v3162, %v3170 : tensor<256x1024x14x14xf32>
    %v3174 = stablehlo.subtract %v3172, %v3173 : tensor<256x1024x14x14xf32>
    %v3175 = stablehlo.divide %v3161, %v3150 : tensor<256x1024x14x14xf32>
    %v3176 = stablehlo.multiply %v3175, %v3174 : tensor<256x1024x14x14xf32>
    %v3177 = stablehlo.reshape %v3176 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3178 = stablehlo.reshape %v3177 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3179 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3180 = stablehlo.transpose %v3179, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3181 = stablehlo.convolution(%v3178, %v3180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v3182 = stablehlo.reshape %v3181 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3183 = stablehlo.reshape %v3182 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3184 = stablehlo.reshape %v889 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3185 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3186 = stablehlo.compare GT, %v3184, %v3185 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3187 = stablehlo.select %v3186, %v3183, %v3185 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3189 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3191 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3192 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3193 = stablehlo.reduce(%v3189 init: %v3190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3193, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3195 = stablehlo.divide %v3194, %v3191 : tensor<256x256x14x14xf32>
    %v3196 = stablehlo.subtract %v3189, %v3195 : tensor<256x256x14x14xf32>
    %v3197 = stablehlo.multiply %v3196, %v3196 : tensor<256x256x14x14xf32>
    %v3198 = stablehlo.reduce(%v3197 init: %v3190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3199 = stablehlo.broadcast_in_dim %v3198, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3200 = stablehlo.divide %v3199, %v3191 : tensor<256x256x14x14xf32>
    %v3201 = stablehlo.add %v3200, %v3192 : tensor<256x256x14x14xf32>
    %v3202 = stablehlo.rsqrt %v3201 : tensor<256x256x14x14xf32>
    %v3203 = stablehlo.multiply %v3196, %v3202 : tensor<256x256x14x14xf32>
    %v3204 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3205 = stablehlo.reshape %v3188 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3206 = stablehlo.multiply %v3204, %v3205 : tensor<256x256x14x14xf32>
    %v3207 = stablehlo.reduce(%v3206 init: %v3190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3208 = stablehlo.broadcast_in_dim %v3207, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3209 = stablehlo.multiply %v3203, %v3206 : tensor<256x256x14x14xf32>
    %v3210 = stablehlo.reduce(%v3209 init: %v3190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3211 = stablehlo.broadcast_in_dim %v3210, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3212 = stablehlo.multiply %v3206, %v3191 : tensor<256x256x14x14xf32>
    %v3213 = stablehlo.subtract %v3212, %v3208 : tensor<256x256x14x14xf32>
    %v3214 = stablehlo.multiply %v3203, %v3211 : tensor<256x256x14x14xf32>
    %v3215 = stablehlo.subtract %v3213, %v3214 : tensor<256x256x14x14xf32>
    %v3216 = stablehlo.divide %v3202, %v3191 : tensor<256x256x14x14xf32>
    %v3217 = stablehlo.multiply %v3216, %v3215 : tensor<256x256x14x14xf32>
    %v3218 = stablehlo.reshape %v3217 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3219 = stablehlo.reshape %v3218 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3220 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3221 = stablehlo.transpose %v3220, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3222 = stablehlo.convolution(%v3219, %v3221)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v3223 = stablehlo.reshape %v3222 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3224 = stablehlo.reshape %v3223 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3225 = stablehlo.reshape %v860 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3226 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3227 = stablehlo.compare GT, %v3225, %v3226 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3228 = stablehlo.select %v3227, %v3224, %v3226 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3230 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3233 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3234 = stablehlo.reduce(%v3230 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3235 = stablehlo.broadcast_in_dim %v3234, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3236 = stablehlo.divide %v3235, %v3232 : tensor<256x256x14x14xf32>
    %v3237 = stablehlo.subtract %v3230, %v3236 : tensor<256x256x14x14xf32>
    %v3238 = stablehlo.multiply %v3237, %v3237 : tensor<256x256x14x14xf32>
    %v3239 = stablehlo.reduce(%v3238 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3241 = stablehlo.divide %v3240, %v3232 : tensor<256x256x14x14xf32>
    %v3242 = stablehlo.add %v3241, %v3233 : tensor<256x256x14x14xf32>
    %v3243 = stablehlo.rsqrt %v3242 : tensor<256x256x14x14xf32>
    %v3244 = stablehlo.multiply %v3237, %v3243 : tensor<256x256x14x14xf32>
    %v3245 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3246 = stablehlo.reshape %v3229 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3247 = stablehlo.multiply %v3245, %v3246 : tensor<256x256x14x14xf32>
    %v3248 = stablehlo.reduce(%v3247 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3248, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3250 = stablehlo.multiply %v3244, %v3247 : tensor<256x256x14x14xf32>
    %v3251 = stablehlo.reduce(%v3250 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3252 = stablehlo.broadcast_in_dim %v3251, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3253 = stablehlo.multiply %v3247, %v3232 : tensor<256x256x14x14xf32>
    %v3254 = stablehlo.subtract %v3253, %v3249 : tensor<256x256x14x14xf32>
    %v3255 = stablehlo.multiply %v3244, %v3252 : tensor<256x256x14x14xf32>
    %v3256 = stablehlo.subtract %v3254, %v3255 : tensor<256x256x14x14xf32>
    %v3257 = stablehlo.divide %v3243, %v3232 : tensor<256x256x14x14xf32>
    %v3258 = stablehlo.multiply %v3257, %v3256 : tensor<256x256x14x14xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3261 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3262 = stablehlo.transpose %v3261, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3263 = stablehlo.convolution(%v3260, %v3262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v3264 = stablehlo.reshape %v3263 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3266 = stablehlo.reshape %v3147 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3267 = stablehlo.add %v3265, %v3266 : tensor<256x1024x14x14xf32>
    %v3268 = stablehlo.reshape %v3267 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3269 = stablehlo.reshape %v835 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3270 = stablehlo.reshape %v3259 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3271 = stablehlo.transpose %v3269, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3272 = stablehlo.transpose %v3270, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3273 = stablehlo.convolution(%v3271, %v3272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3274 = stablehlo.transpose %v3273, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3275 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3277 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3278 = stablehlo.reduce(%v3275 init: %v3276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3279 = stablehlo.broadcast_in_dim %v3278, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3280 = stablehlo.divide %v3279, %v3277 : tensor<256x256x14x14xf32>
    %v3281 = stablehlo.subtract %v3275, %v3280 : tensor<256x256x14x14xf32>
    %v3282 = stablehlo.multiply %v3281, %v3281 : tensor<256x256x14x14xf32>
    %v3283 = stablehlo.reduce(%v3282 init: %v3276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3284 = stablehlo.broadcast_in_dim %v3283, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3285 = stablehlo.divide %v3284, %v3277 : tensor<256x256x14x14xf32>
    %v3286 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3287 = stablehlo.add %v3285, %v3286 : tensor<256x256x14x14xf32>
    %v3288 = stablehlo.rsqrt %v3287 : tensor<256x256x14x14xf32>
    %v3289 = stablehlo.multiply %v3281, %v3288 : tensor<256x256x14x14xf32>
    %v3290 = stablehlo.reshape %v3229 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3291 = stablehlo.multiply %v3290, %v3289 : tensor<256x256x14x14xf32>
    %v3292 = stablehlo.reduce(%v3291 init: %v3276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3293 = stablehlo.reshape %v3229 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3295 = stablehlo.reduce(%v3293 init: %v3294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3296 = stablehlo.reshape %v864 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3297 = stablehlo.reshape %v3218 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3298 = stablehlo.transpose %v3296, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3299 = stablehlo.transpose %v3297, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3300 = stablehlo.convolution(%v3298, %v3299)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3301 = stablehlo.transpose %v3300, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3302 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3304 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3305 = stablehlo.reduce(%v3302 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3306 = stablehlo.broadcast_in_dim %v3305, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3307 = stablehlo.divide %v3306, %v3304 : tensor<256x256x14x14xf32>
    %v3308 = stablehlo.subtract %v3302, %v3307 : tensor<256x256x14x14xf32>
    %v3309 = stablehlo.multiply %v3308, %v3308 : tensor<256x256x14x14xf32>
    %v3310 = stablehlo.reduce(%v3309 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3311 = stablehlo.broadcast_in_dim %v3310, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3312 = stablehlo.divide %v3311, %v3304 : tensor<256x256x14x14xf32>
    %v3313 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3314 = stablehlo.add %v3312, %v3313 : tensor<256x256x14x14xf32>
    %v3315 = stablehlo.rsqrt %v3314 : tensor<256x256x14x14xf32>
    %v3316 = stablehlo.multiply %v3308, %v3315 : tensor<256x256x14x14xf32>
    %v3317 = stablehlo.reshape %v3188 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3318 = stablehlo.multiply %v3317, %v3316 : tensor<256x256x14x14xf32>
    %v3319 = stablehlo.reduce(%v3318 init: %v3303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3320 = stablehlo.reshape %v3188 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3322 = stablehlo.reduce(%v3320 init: %v3321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3323 = stablehlo.reshape %v893 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3324 = stablehlo.reshape %v3177 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3325 = stablehlo.transpose %v3323, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3326 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3327 = stablehlo.convolution(%v3325, %v3326)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3328 = stablehlo.transpose %v3327, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3329 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3331 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3332 = stablehlo.reduce(%v3329 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3333 = stablehlo.broadcast_in_dim %v3332, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3334 = stablehlo.divide %v3333, %v3331 : tensor<256x1024x14x14xf32>
    %v3335 = stablehlo.subtract %v3329, %v3334 : tensor<256x1024x14x14xf32>
    %v3336 = stablehlo.multiply %v3335, %v3335 : tensor<256x1024x14x14xf32>
    %v3337 = stablehlo.reduce(%v3336 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3337, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3339 = stablehlo.divide %v3338, %v3331 : tensor<256x1024x14x14xf32>
    %v3340 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3341 = stablehlo.add %v3339, %v3340 : tensor<256x1024x14x14xf32>
    %v3342 = stablehlo.rsqrt %v3341 : tensor<256x1024x14x14xf32>
    %v3343 = stablehlo.multiply %v3335, %v3342 : tensor<256x1024x14x14xf32>
    %v3344 = stablehlo.reshape %v3147 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3345 = stablehlo.multiply %v3344, %v3343 : tensor<256x1024x14x14xf32>
    %v3346 = stablehlo.reduce(%v3345 init: %v3330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3347 = stablehlo.reshape %v3147 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3349 = stablehlo.reduce(%v3347 init: %v3348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3350 = stablehlo.reshape %v3268 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3351 = stablehlo.reshape %v831 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3352 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v3353 = stablehlo.compare GT, %v3351, %v3352 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v3354 = stablehlo.select %v3353, %v3350, %v3352 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v3355 = stablehlo.reshape %v3354 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3356 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3358 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3359 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3360 = stablehlo.reduce(%v3356 init: %v3357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3361 = stablehlo.broadcast_in_dim %v3360, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3362 = stablehlo.divide %v3361, %v3358 : tensor<256x1024x14x14xf32>
    %v3363 = stablehlo.subtract %v3356, %v3362 : tensor<256x1024x14x14xf32>
    %v3364 = stablehlo.multiply %v3363, %v3363 : tensor<256x1024x14x14xf32>
    %v3365 = stablehlo.reduce(%v3364 init: %v3357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3366 = stablehlo.broadcast_in_dim %v3365, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3367 = stablehlo.divide %v3366, %v3358 : tensor<256x1024x14x14xf32>
    %v3368 = stablehlo.add %v3367, %v3359 : tensor<256x1024x14x14xf32>
    %v3369 = stablehlo.rsqrt %v3368 : tensor<256x1024x14x14xf32>
    %v3370 = stablehlo.multiply %v3363, %v3369 : tensor<256x1024x14x14xf32>
    %v3371 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3372 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3373 = stablehlo.multiply %v3371, %v3372 : tensor<256x1024x14x14xf32>
    %v3374 = stablehlo.reduce(%v3373 init: %v3357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3374, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3376 = stablehlo.multiply %v3370, %v3373 : tensor<256x1024x14x14xf32>
    %v3377 = stablehlo.reduce(%v3376 init: %v3357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3378 = stablehlo.broadcast_in_dim %v3377, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3379 = stablehlo.multiply %v3373, %v3358 : tensor<256x1024x14x14xf32>
    %v3380 = stablehlo.subtract %v3379, %v3375 : tensor<256x1024x14x14xf32>
    %v3381 = stablehlo.multiply %v3370, %v3378 : tensor<256x1024x14x14xf32>
    %v3382 = stablehlo.subtract %v3380, %v3381 : tensor<256x1024x14x14xf32>
    %v3383 = stablehlo.divide %v3369, %v3358 : tensor<256x1024x14x14xf32>
    %v3384 = stablehlo.multiply %v3383, %v3382 : tensor<256x1024x14x14xf32>
    %v3385 = stablehlo.reshape %v3384 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3386 = stablehlo.reshape %v3385 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3387 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3388 = stablehlo.transpose %v3387, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3389 = stablehlo.convolution(%v3386, %v3388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3391 = stablehlo.reshape %v3390 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3392 = stablehlo.reshape %v773 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3393 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3394 = stablehlo.compare GT, %v3392, %v3393 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3395 = stablehlo.select %v3394, %v3391, %v3393 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3396 = stablehlo.reshape %v3395 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3397 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3399 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3400 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3401 = stablehlo.reduce(%v3397 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3402 = stablehlo.broadcast_in_dim %v3401, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3403 = stablehlo.divide %v3402, %v3399 : tensor<256x256x14x14xf32>
    %v3404 = stablehlo.subtract %v3397, %v3403 : tensor<256x256x14x14xf32>
    %v3405 = stablehlo.multiply %v3404, %v3404 : tensor<256x256x14x14xf32>
    %v3406 = stablehlo.reduce(%v3405 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3407 = stablehlo.broadcast_in_dim %v3406, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3408 = stablehlo.divide %v3407, %v3399 : tensor<256x256x14x14xf32>
    %v3409 = stablehlo.add %v3408, %v3400 : tensor<256x256x14x14xf32>
    %v3410 = stablehlo.rsqrt %v3409 : tensor<256x256x14x14xf32>
    %v3411 = stablehlo.multiply %v3404, %v3410 : tensor<256x256x14x14xf32>
    %v3412 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3413 = stablehlo.reshape %v3396 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3414 = stablehlo.multiply %v3412, %v3413 : tensor<256x256x14x14xf32>
    %v3415 = stablehlo.reduce(%v3414 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3416 = stablehlo.broadcast_in_dim %v3415, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3417 = stablehlo.multiply %v3411, %v3414 : tensor<256x256x14x14xf32>
    %v3418 = stablehlo.reduce(%v3417 init: %v3398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3419 = stablehlo.broadcast_in_dim %v3418, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3420 = stablehlo.multiply %v3414, %v3399 : tensor<256x256x14x14xf32>
    %v3421 = stablehlo.subtract %v3420, %v3416 : tensor<256x256x14x14xf32>
    %v3422 = stablehlo.multiply %v3411, %v3419 : tensor<256x256x14x14xf32>
    %v3423 = stablehlo.subtract %v3421, %v3422 : tensor<256x256x14x14xf32>
    %v3424 = stablehlo.divide %v3410, %v3399 : tensor<256x256x14x14xf32>
    %v3425 = stablehlo.multiply %v3424, %v3423 : tensor<256x256x14x14xf32>
    %v3426 = stablehlo.reshape %v3425 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3427 = stablehlo.reshape %v3426 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3428 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3429 = stablehlo.transpose %v3428, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3430 = stablehlo.convolution(%v3427, %v3429)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x28x28xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3433 = stablehlo.reshape %v744 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3434 = stablehlo.constant dense<0.0> : tensor<256x256x28x28xf32>
    %v3435 = stablehlo.compare GT, %v3433, %v3434 : (tensor<256x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xi1>
    %v3436 = stablehlo.select %v3435, %v3432, %v3434 : tensor<256x256x28x28xi1>, tensor<256x256x28x28xf32>
    %v3437 = stablehlo.reshape %v3436 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3438 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3440 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3441 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3442 = stablehlo.reduce(%v3438 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3444 = stablehlo.divide %v3443, %v3440 : tensor<256x256x28x28xf32>
    %v3445 = stablehlo.subtract %v3438, %v3444 : tensor<256x256x28x28xf32>
    %v3446 = stablehlo.multiply %v3445, %v3445 : tensor<256x256x28x28xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3449 = stablehlo.divide %v3448, %v3440 : tensor<256x256x28x28xf32>
    %v3450 = stablehlo.add %v3449, %v3441 : tensor<256x256x28x28xf32>
    %v3451 = stablehlo.rsqrt %v3450 : tensor<256x256x28x28xf32>
    %v3452 = stablehlo.multiply %v3445, %v3451 : tensor<256x256x28x28xf32>
    %v3453 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3454 = stablehlo.reshape %v3437 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3455 = stablehlo.multiply %v3453, %v3454 : tensor<256x256x28x28xf32>
    %v3456 = stablehlo.reduce(%v3455 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3457 = stablehlo.broadcast_in_dim %v3456, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3458 = stablehlo.multiply %v3452, %v3455 : tensor<256x256x28x28xf32>
    %v3459 = stablehlo.reduce(%v3458 init: %v3439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3460 = stablehlo.broadcast_in_dim %v3459, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3461 = stablehlo.multiply %v3455, %v3440 : tensor<256x256x28x28xf32>
    %v3462 = stablehlo.subtract %v3461, %v3457 : tensor<256x256x28x28xf32>
    %v3463 = stablehlo.multiply %v3452, %v3460 : tensor<256x256x28x28xf32>
    %v3464 = stablehlo.subtract %v3462, %v3463 : tensor<256x256x28x28xf32>
    %v3465 = stablehlo.divide %v3451, %v3440 : tensor<256x256x28x28xf32>
    %v3466 = stablehlo.multiply %v3465, %v3464 : tensor<256x256x28x28xf32>
    %v3467 = stablehlo.reshape %v3466 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3468 = stablehlo.reshape %v3467 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3469 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v3470 = stablehlo.transpose %v3469, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v3471 = stablehlo.convolution(%v3468, %v3470)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3473 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3475 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3476 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3477 = stablehlo.reduce(%v3473 init: %v3474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3478 = stablehlo.broadcast_in_dim %v3477, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3479 = stablehlo.divide %v3478, %v3475 : tensor<256x1024x14x14xf32>
    %v3480 = stablehlo.subtract %v3473, %v3479 : tensor<256x1024x14x14xf32>
    %v3481 = stablehlo.multiply %v3480, %v3480 : tensor<256x1024x14x14xf32>
    %v3482 = stablehlo.reduce(%v3481 init: %v3474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3482, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3484 = stablehlo.divide %v3483, %v3475 : tensor<256x1024x14x14xf32>
    %v3485 = stablehlo.add %v3484, %v3476 : tensor<256x1024x14x14xf32>
    %v3486 = stablehlo.rsqrt %v3485 : tensor<256x1024x14x14xf32>
    %v3487 = stablehlo.multiply %v3480, %v3486 : tensor<256x1024x14x14xf32>
    %v3488 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3489 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3490 = stablehlo.multiply %v3488, %v3489 : tensor<256x1024x14x14xf32>
    %v3491 = stablehlo.reduce(%v3490 init: %v3474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3492 = stablehlo.broadcast_in_dim %v3491, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3493 = stablehlo.multiply %v3487, %v3490 : tensor<256x1024x14x14xf32>
    %v3494 = stablehlo.reduce(%v3493 init: %v3474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3495 = stablehlo.broadcast_in_dim %v3494, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3496 = stablehlo.multiply %v3490, %v3475 : tensor<256x1024x14x14xf32>
    %v3497 = stablehlo.subtract %v3496, %v3492 : tensor<256x1024x14x14xf32>
    %v3498 = stablehlo.multiply %v3487, %v3495 : tensor<256x1024x14x14xf32>
    %v3499 = stablehlo.subtract %v3497, %v3498 : tensor<256x1024x14x14xf32>
    %v3500 = stablehlo.divide %v3486, %v3475 : tensor<256x1024x14x14xf32>
    %v3501 = stablehlo.multiply %v3500, %v3499 : tensor<256x1024x14x14xf32>
    %v3502 = stablehlo.reshape %v3501 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3503 = stablehlo.reshape %v3502 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3504 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v3505 = stablehlo.transpose %v3504, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3506 = stablehlo.convolution(%v3503, %v3505)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3507 = stablehlo.reshape %v3506 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3508 = stablehlo.reshape %v3472 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3509 = stablehlo.reshape %v3507 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3510 = stablehlo.add %v3508, %v3509 : tensor<256x512x28x28xf32>
    %v3511 = stablehlo.reshape %v3510 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3512 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3513 = stablehlo.reshape %v3467 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3514 = stablehlo.transpose %v3512, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3515 = stablehlo.transpose %v3513, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3516 = stablehlo.convolution(%v3514, %v3515)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<512x256x1x1xf32>
    %v3517 = stablehlo.transpose %v3516, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v3518 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3520 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3521 = stablehlo.reduce(%v3518 init: %v3519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3522 = stablehlo.broadcast_in_dim %v3521, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3523 = stablehlo.divide %v3522, %v3520 : tensor<256x256x28x28xf32>
    %v3524 = stablehlo.subtract %v3518, %v3523 : tensor<256x256x28x28xf32>
    %v3525 = stablehlo.multiply %v3524, %v3524 : tensor<256x256x28x28xf32>
    %v3526 = stablehlo.reduce(%v3525 init: %v3519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3527 = stablehlo.broadcast_in_dim %v3526, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3528 = stablehlo.divide %v3527, %v3520 : tensor<256x256x28x28xf32>
    %v3529 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3530 = stablehlo.add %v3528, %v3529 : tensor<256x256x28x28xf32>
    %v3531 = stablehlo.rsqrt %v3530 : tensor<256x256x28x28xf32>
    %v3532 = stablehlo.multiply %v3524, %v3531 : tensor<256x256x28x28xf32>
    %v3533 = stablehlo.reshape %v3437 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3534 = stablehlo.multiply %v3533, %v3532 : tensor<256x256x28x28xf32>
    %v3535 = stablehlo.reduce(%v3534 init: %v3519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3536 = stablehlo.reshape %v3437 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.reduce(%v3536 init: %v3537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3539 = stablehlo.reshape %v748 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3540 = stablehlo.reshape %v3426 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3541 = stablehlo.transpose %v3539, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3543 = stablehlo.transpose %v3540, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3544 = stablehlo.convolution(%v3541, %v3543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3545 = stablehlo.transpose %v3544, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3546 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3548 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3549 = stablehlo.reduce(%v3546 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3550 = stablehlo.broadcast_in_dim %v3549, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3551 = stablehlo.divide %v3550, %v3548 : tensor<256x256x14x14xf32>
    %v3552 = stablehlo.subtract %v3546, %v3551 : tensor<256x256x14x14xf32>
    %v3553 = stablehlo.multiply %v3552, %v3552 : tensor<256x256x14x14xf32>
    %v3554 = stablehlo.reduce(%v3553 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3555 = stablehlo.broadcast_in_dim %v3554, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3556 = stablehlo.divide %v3555, %v3548 : tensor<256x256x14x14xf32>
    %v3557 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3558 = stablehlo.add %v3556, %v3557 : tensor<256x256x14x14xf32>
    %v3559 = stablehlo.rsqrt %v3558 : tensor<256x256x14x14xf32>
    %v3560 = stablehlo.multiply %v3552, %v3559 : tensor<256x256x14x14xf32>
    %v3561 = stablehlo.reshape %v3396 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3562 = stablehlo.multiply %v3561, %v3560 : tensor<256x256x14x14xf32>
    %v3563 = stablehlo.reduce(%v3562 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3564 = stablehlo.reshape %v3396 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3566 = stablehlo.reduce(%v3564 init: %v3565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3567 = stablehlo.reshape %v777 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3568 = stablehlo.reshape %v3385 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3569 = stablehlo.transpose %v3567, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3570 = stablehlo.transpose %v3568, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3571 = stablehlo.convolution(%v3569, %v3570)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3572 = stablehlo.transpose %v3571, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3573 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3576 = stablehlo.reduce(%v3573 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3576, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3578 = stablehlo.divide %v3577, %v3575 : tensor<256x1024x14x14xf32>
    %v3579 = stablehlo.subtract %v3573, %v3578 : tensor<256x1024x14x14xf32>
    %v3580 = stablehlo.multiply %v3579, %v3579 : tensor<256x1024x14x14xf32>
    %v3581 = stablehlo.reduce(%v3580 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3582 = stablehlo.broadcast_in_dim %v3581, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3583 = stablehlo.divide %v3582, %v3575 : tensor<256x1024x14x14xf32>
    %v3584 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3585 = stablehlo.add %v3583, %v3584 : tensor<256x1024x14x14xf32>
    %v3586 = stablehlo.rsqrt %v3585 : tensor<256x1024x14x14xf32>
    %v3587 = stablehlo.multiply %v3579, %v3586 : tensor<256x1024x14x14xf32>
    %v3588 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3589 = stablehlo.multiply %v3588, %v3587 : tensor<256x1024x14x14xf32>
    %v3590 = stablehlo.reduce(%v3589 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3591 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3593 = stablehlo.reduce(%v3591 init: %v3592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3594 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3595 = stablehlo.reshape %v3502 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3596 = stablehlo.transpose %v3594, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3598 = stablehlo.transpose %v3595, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3597 = stablehlo.slice %v3596 [0:512, 0:256, 0:27, 0:27] : (tensor<512x256x28x28xf32>) -> tensor<512x256x27x27xf32>
    %v3599 = stablehlo.convolution(%v3597, %v3598)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x27x27xf32>, tensor<1024x256x14x14xf32>) -> tensor<512x1024x1x1xf32>
    %v3600 = stablehlo.transpose %v3599, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3601 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3603 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3604 = stablehlo.reduce(%v3601 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3605 = stablehlo.broadcast_in_dim %v3604, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3606 = stablehlo.divide %v3605, %v3603 : tensor<256x1024x14x14xf32>
    %v3607 = stablehlo.subtract %v3601, %v3606 : tensor<256x1024x14x14xf32>
    %v3608 = stablehlo.multiply %v3607, %v3607 : tensor<256x1024x14x14xf32>
    %v3609 = stablehlo.reduce(%v3608 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3610 = stablehlo.broadcast_in_dim %v3609, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3611 = stablehlo.divide %v3610, %v3603 : tensor<256x1024x14x14xf32>
    %v3612 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3613 = stablehlo.add %v3611, %v3612 : tensor<256x1024x14x14xf32>
    %v3614 = stablehlo.rsqrt %v3613 : tensor<256x1024x14x14xf32>
    %v3615 = stablehlo.multiply %v3607, %v3614 : tensor<256x1024x14x14xf32>
    %v3616 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3617 = stablehlo.multiply %v3616, %v3615 : tensor<256x1024x14x14xf32>
    %v3618 = stablehlo.reduce(%v3617 init: %v3602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3619 = stablehlo.reshape %v3355 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3620 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3621 = stablehlo.reduce(%v3619 init: %v3620) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3622 = stablehlo.reshape %v3511 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3623 = stablehlo.reshape %v715 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3624 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v3625 = stablehlo.compare GT, %v3623, %v3624 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v3626 = stablehlo.select %v3625, %v3622, %v3624 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v3627 = stablehlo.reshape %v3626 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3628 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3630 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3631 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3632 = stablehlo.reduce(%v3628 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3633 = stablehlo.broadcast_in_dim %v3632, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3634 = stablehlo.divide %v3633, %v3630 : tensor<256x512x28x28xf32>
    %v3635 = stablehlo.subtract %v3628, %v3634 : tensor<256x512x28x28xf32>
    %v3636 = stablehlo.multiply %v3635, %v3635 : tensor<256x512x28x28xf32>
    %v3637 = stablehlo.reduce(%v3636 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3638 = stablehlo.broadcast_in_dim %v3637, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3639 = stablehlo.divide %v3638, %v3630 : tensor<256x512x28x28xf32>
    %v3640 = stablehlo.add %v3639, %v3631 : tensor<256x512x28x28xf32>
    %v3641 = stablehlo.rsqrt %v3640 : tensor<256x512x28x28xf32>
    %v3642 = stablehlo.multiply %v3635, %v3641 : tensor<256x512x28x28xf32>
    %v3643 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3644 = stablehlo.reshape %v3627 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3645 = stablehlo.multiply %v3643, %v3644 : tensor<256x512x28x28xf32>
    %v3646 = stablehlo.reduce(%v3645 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3647 = stablehlo.broadcast_in_dim %v3646, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3648 = stablehlo.multiply %v3642, %v3645 : tensor<256x512x28x28xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3650 = stablehlo.broadcast_in_dim %v3649, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3651 = stablehlo.multiply %v3645, %v3630 : tensor<256x512x28x28xf32>
    %v3652 = stablehlo.subtract %v3651, %v3647 : tensor<256x512x28x28xf32>
    %v3653 = stablehlo.multiply %v3642, %v3650 : tensor<256x512x28x28xf32>
    %v3654 = stablehlo.subtract %v3652, %v3653 : tensor<256x512x28x28xf32>
    %v3655 = stablehlo.divide %v3641, %v3630 : tensor<256x512x28x28xf32>
    %v3656 = stablehlo.multiply %v3655, %v3654 : tensor<256x512x28x28xf32>
    %v3657 = stablehlo.reshape %v3656 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3658 = stablehlo.reshape %v3657 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3659 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3660 = stablehlo.transpose %v3659, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3661 = stablehlo.convolution(%v3658, %v3660)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3663 = stablehlo.reshape %v3662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3664 = stablehlo.reshape %v682 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3666 = stablehlo.compare GT, %v3664, %v3665 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3667 = stablehlo.select %v3666, %v3663, %v3665 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3668 = stablehlo.reshape %v3667 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3669 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3671 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3672 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3673 = stablehlo.reduce(%v3669 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3674 = stablehlo.broadcast_in_dim %v3673, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3675 = stablehlo.divide %v3674, %v3671 : tensor<256x128x28x28xf32>
    %v3676 = stablehlo.subtract %v3669, %v3675 : tensor<256x128x28x28xf32>
    %v3677 = stablehlo.multiply %v3676, %v3676 : tensor<256x128x28x28xf32>
    %v3678 = stablehlo.reduce(%v3677 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3679 = stablehlo.broadcast_in_dim %v3678, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3680 = stablehlo.divide %v3679, %v3671 : tensor<256x128x28x28xf32>
    %v3681 = stablehlo.add %v3680, %v3672 : tensor<256x128x28x28xf32>
    %v3682 = stablehlo.rsqrt %v3681 : tensor<256x128x28x28xf32>
    %v3683 = stablehlo.multiply %v3676, %v3682 : tensor<256x128x28x28xf32>
    %v3684 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3685 = stablehlo.reshape %v3668 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3686 = stablehlo.multiply %v3684, %v3685 : tensor<256x128x28x28xf32>
    %v3687 = stablehlo.reduce(%v3686 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3688 = stablehlo.broadcast_in_dim %v3687, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3689 = stablehlo.multiply %v3683, %v3686 : tensor<256x128x28x28xf32>
    %v3690 = stablehlo.reduce(%v3689 init: %v3670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3691 = stablehlo.broadcast_in_dim %v3690, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3692 = stablehlo.multiply %v3686, %v3671 : tensor<256x128x28x28xf32>
    %v3693 = stablehlo.subtract %v3692, %v3688 : tensor<256x128x28x28xf32>
    %v3694 = stablehlo.multiply %v3683, %v3691 : tensor<256x128x28x28xf32>
    %v3695 = stablehlo.subtract %v3693, %v3694 : tensor<256x128x28x28xf32>
    %v3696 = stablehlo.divide %v3682, %v3671 : tensor<256x128x28x28xf32>
    %v3697 = stablehlo.multiply %v3696, %v3695 : tensor<256x128x28x28xf32>
    %v3698 = stablehlo.reshape %v3697 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3699 = stablehlo.reshape %v3698 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3700 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3701 = stablehlo.transpose %v3700, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3702 = stablehlo.convolution(%v3699, %v3701)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3703 = stablehlo.reshape %v3702 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3704 = stablehlo.reshape %v3703 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3705 = stablehlo.reshape %v653 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3706 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3707 = stablehlo.compare GT, %v3705, %v3706 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3708 = stablehlo.select %v3707, %v3704, %v3706 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3709 = stablehlo.reshape %v3708 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3710 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3713 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3714 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3715 = stablehlo.broadcast_in_dim %v3714, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3716 = stablehlo.divide %v3715, %v3712 : tensor<256x128x28x28xf32>
    %v3717 = stablehlo.subtract %v3710, %v3716 : tensor<256x128x28x28xf32>
    %v3718 = stablehlo.multiply %v3717, %v3717 : tensor<256x128x28x28xf32>
    %v3719 = stablehlo.reduce(%v3718 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3721 = stablehlo.divide %v3720, %v3712 : tensor<256x128x28x28xf32>
    %v3722 = stablehlo.add %v3721, %v3713 : tensor<256x128x28x28xf32>
    %v3723 = stablehlo.rsqrt %v3722 : tensor<256x128x28x28xf32>
    %v3724 = stablehlo.multiply %v3717, %v3723 : tensor<256x128x28x28xf32>
    %v3725 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3726 = stablehlo.reshape %v3709 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3727 = stablehlo.multiply %v3725, %v3726 : tensor<256x128x28x28xf32>
    %v3728 = stablehlo.reduce(%v3727 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3729 = stablehlo.broadcast_in_dim %v3728, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3730 = stablehlo.multiply %v3724, %v3727 : tensor<256x128x28x28xf32>
    %v3731 = stablehlo.reduce(%v3730 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3732 = stablehlo.broadcast_in_dim %v3731, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3733 = stablehlo.multiply %v3727, %v3712 : tensor<256x128x28x28xf32>
    %v3734 = stablehlo.subtract %v3733, %v3729 : tensor<256x128x28x28xf32>
    %v3735 = stablehlo.multiply %v3724, %v3732 : tensor<256x128x28x28xf32>
    %v3736 = stablehlo.subtract %v3734, %v3735 : tensor<256x128x28x28xf32>
    %v3737 = stablehlo.divide %v3723, %v3712 : tensor<256x128x28x28xf32>
    %v3738 = stablehlo.multiply %v3737, %v3736 : tensor<256x128x28x28xf32>
    %v3739 = stablehlo.reshape %v3738 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3740 = stablehlo.reshape %v3739 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3741 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3742 = stablehlo.transpose %v3741, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3743 = stablehlo.convolution(%v3740, %v3742)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3744 = stablehlo.reshape %v3743 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3746 = stablehlo.reshape %v3627 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3747 = stablehlo.add %v3745, %v3746 : tensor<256x512x28x28xf32>
    %v3748 = stablehlo.reshape %v3747 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3749 = stablehlo.reshape %v628 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3750 = stablehlo.reshape %v3739 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3751 = stablehlo.transpose %v3749, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3752 = stablehlo.transpose %v3750, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3753 = stablehlo.convolution(%v3751, %v3752)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3754 = stablehlo.transpose %v3753, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3755 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3756 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3757 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3758 = stablehlo.reduce(%v3755 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3759 = stablehlo.broadcast_in_dim %v3758, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3760 = stablehlo.divide %v3759, %v3757 : tensor<256x128x28x28xf32>
    %v3761 = stablehlo.subtract %v3755, %v3760 : tensor<256x128x28x28xf32>
    %v3762 = stablehlo.multiply %v3761, %v3761 : tensor<256x128x28x28xf32>
    %v3763 = stablehlo.reduce(%v3762 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3764 = stablehlo.broadcast_in_dim %v3763, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3765 = stablehlo.divide %v3764, %v3757 : tensor<256x128x28x28xf32>
    %v3766 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3767 = stablehlo.add %v3765, %v3766 : tensor<256x128x28x28xf32>
    %v3768 = stablehlo.rsqrt %v3767 : tensor<256x128x28x28xf32>
    %v3769 = stablehlo.multiply %v3761, %v3768 : tensor<256x128x28x28xf32>
    %v3770 = stablehlo.reshape %v3709 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3771 = stablehlo.multiply %v3770, %v3769 : tensor<256x128x28x28xf32>
    %v3772 = stablehlo.reduce(%v3771 init: %v3756) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3773 = stablehlo.reshape %v3709 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3775 = stablehlo.reduce(%v3773 init: %v3774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3776 = stablehlo.reshape %v657 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3777 = stablehlo.reshape %v3698 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3778 = stablehlo.transpose %v3776, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3779 = stablehlo.transpose %v3777, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3780 = stablehlo.convolution(%v3778, %v3779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3781 = stablehlo.transpose %v3780, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3782 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3784 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3785 = stablehlo.reduce(%v3782 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3787 = stablehlo.divide %v3786, %v3784 : tensor<256x128x28x28xf32>
    %v3788 = stablehlo.subtract %v3782, %v3787 : tensor<256x128x28x28xf32>
    %v3789 = stablehlo.multiply %v3788, %v3788 : tensor<256x128x28x28xf32>
    %v3790 = stablehlo.reduce(%v3789 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3791 = stablehlo.broadcast_in_dim %v3790, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3792 = stablehlo.divide %v3791, %v3784 : tensor<256x128x28x28xf32>
    %v3793 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3794 = stablehlo.add %v3792, %v3793 : tensor<256x128x28x28xf32>
    %v3795 = stablehlo.rsqrt %v3794 : tensor<256x128x28x28xf32>
    %v3796 = stablehlo.multiply %v3788, %v3795 : tensor<256x128x28x28xf32>
    %v3797 = stablehlo.reshape %v3668 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3798 = stablehlo.multiply %v3797, %v3796 : tensor<256x128x28x28xf32>
    %v3799 = stablehlo.reduce(%v3798 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3800 = stablehlo.reshape %v3668 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3802 = stablehlo.reduce(%v3800 init: %v3801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3803 = stablehlo.reshape %v686 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3804 = stablehlo.reshape %v3657 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3805 = stablehlo.transpose %v3803, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3806 = stablehlo.transpose %v3804, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3807 = stablehlo.convolution(%v3805, %v3806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v3808 = stablehlo.transpose %v3807, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3809 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3812, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3814 = stablehlo.divide %v3813, %v3811 : tensor<256x512x28x28xf32>
    %v3815 = stablehlo.subtract %v3809, %v3814 : tensor<256x512x28x28xf32>
    %v3816 = stablehlo.multiply %v3815, %v3815 : tensor<256x512x28x28xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3819 = stablehlo.divide %v3818, %v3811 : tensor<256x512x28x28xf32>
    %v3820 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3821 = stablehlo.add %v3819, %v3820 : tensor<256x512x28x28xf32>
    %v3822 = stablehlo.rsqrt %v3821 : tensor<256x512x28x28xf32>
    %v3823 = stablehlo.multiply %v3815, %v3822 : tensor<256x512x28x28xf32>
    %v3824 = stablehlo.reshape %v3627 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3825 = stablehlo.multiply %v3824, %v3823 : tensor<256x512x28x28xf32>
    %v3826 = stablehlo.reduce(%v3825 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3827 = stablehlo.reshape %v3627 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3829 = stablehlo.reduce(%v3827 init: %v3828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3830 = stablehlo.reshape %v3748 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3831 = stablehlo.reshape %v624 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3832 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v3833 = stablehlo.compare GT, %v3831, %v3832 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v3834 = stablehlo.select %v3833, %v3830, %v3832 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3836 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3838 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3839 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3840 = stablehlo.reduce(%v3836 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3841 = stablehlo.broadcast_in_dim %v3840, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3842 = stablehlo.divide %v3841, %v3838 : tensor<256x512x28x28xf32>
    %v3843 = stablehlo.subtract %v3836, %v3842 : tensor<256x512x28x28xf32>
    %v3844 = stablehlo.multiply %v3843, %v3843 : tensor<256x512x28x28xf32>
    %v3845 = stablehlo.reduce(%v3844 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3846 = stablehlo.broadcast_in_dim %v3845, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3847 = stablehlo.divide %v3846, %v3838 : tensor<256x512x28x28xf32>
    %v3848 = stablehlo.add %v3847, %v3839 : tensor<256x512x28x28xf32>
    %v3849 = stablehlo.rsqrt %v3848 : tensor<256x512x28x28xf32>
    %v3850 = stablehlo.multiply %v3843, %v3849 : tensor<256x512x28x28xf32>
    %v3851 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3852 = stablehlo.reshape %v3835 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3853 = stablehlo.multiply %v3851, %v3852 : tensor<256x512x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3855 = stablehlo.broadcast_in_dim %v3854, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3856 = stablehlo.multiply %v3850, %v3853 : tensor<256x512x28x28xf32>
    %v3857 = stablehlo.reduce(%v3856 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3858 = stablehlo.broadcast_in_dim %v3857, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3859 = stablehlo.multiply %v3853, %v3838 : tensor<256x512x28x28xf32>
    %v3860 = stablehlo.subtract %v3859, %v3855 : tensor<256x512x28x28xf32>
    %v3861 = stablehlo.multiply %v3850, %v3858 : tensor<256x512x28x28xf32>
    %v3862 = stablehlo.subtract %v3860, %v3861 : tensor<256x512x28x28xf32>
    %v3863 = stablehlo.divide %v3849, %v3838 : tensor<256x512x28x28xf32>
    %v3864 = stablehlo.multiply %v3863, %v3862 : tensor<256x512x28x28xf32>
    %v3865 = stablehlo.reshape %v3864 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3866 = stablehlo.reshape %v3865 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3867 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3868 = stablehlo.transpose %v3867, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3869 = stablehlo.convolution(%v3866, %v3868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3870 = stablehlo.reshape %v3869 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3871 = stablehlo.reshape %v3870 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3872 = stablehlo.reshape %v591 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3873 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3874 = stablehlo.compare GT, %v3872, %v3873 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3875 = stablehlo.select %v3874, %v3871, %v3873 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3876 = stablehlo.reshape %v3875 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3877 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3879 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3880 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3881 = stablehlo.reduce(%v3877 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3882 = stablehlo.broadcast_in_dim %v3881, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3883 = stablehlo.divide %v3882, %v3879 : tensor<256x128x28x28xf32>
    %v3884 = stablehlo.subtract %v3877, %v3883 : tensor<256x128x28x28xf32>
    %v3885 = stablehlo.multiply %v3884, %v3884 : tensor<256x128x28x28xf32>
    %v3886 = stablehlo.reduce(%v3885 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3887 = stablehlo.broadcast_in_dim %v3886, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3888 = stablehlo.divide %v3887, %v3879 : tensor<256x128x28x28xf32>
    %v3889 = stablehlo.add %v3888, %v3880 : tensor<256x128x28x28xf32>
    %v3890 = stablehlo.rsqrt %v3889 : tensor<256x128x28x28xf32>
    %v3891 = stablehlo.multiply %v3884, %v3890 : tensor<256x128x28x28xf32>
    %v3892 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3893 = stablehlo.reshape %v3876 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3894 = stablehlo.multiply %v3892, %v3893 : tensor<256x128x28x28xf32>
    %v3895 = stablehlo.reduce(%v3894 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3896 = stablehlo.broadcast_in_dim %v3895, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3897 = stablehlo.multiply %v3891, %v3894 : tensor<256x128x28x28xf32>
    %v3898 = stablehlo.reduce(%v3897 init: %v3878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3899 = stablehlo.broadcast_in_dim %v3898, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3900 = stablehlo.multiply %v3894, %v3879 : tensor<256x128x28x28xf32>
    %v3901 = stablehlo.subtract %v3900, %v3896 : tensor<256x128x28x28xf32>
    %v3902 = stablehlo.multiply %v3891, %v3899 : tensor<256x128x28x28xf32>
    %v3903 = stablehlo.subtract %v3901, %v3902 : tensor<256x128x28x28xf32>
    %v3904 = stablehlo.divide %v3890, %v3879 : tensor<256x128x28x28xf32>
    %v3905 = stablehlo.multiply %v3904, %v3903 : tensor<256x128x28x28xf32>
    %v3906 = stablehlo.reshape %v3905 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3907 = stablehlo.reshape %v3906 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3908 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3909 = stablehlo.transpose %v3908, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3910 = stablehlo.convolution(%v3907, %v3909)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3911 = stablehlo.reshape %v3910 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3912 = stablehlo.reshape %v3911 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3913 = stablehlo.reshape %v562 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3914 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3915 = stablehlo.compare GT, %v3913, %v3914 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3916 = stablehlo.select %v3915, %v3912, %v3914 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3917 = stablehlo.reshape %v3916 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3918 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3919 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3920 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3921 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3922 = stablehlo.reduce(%v3918 init: %v3919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3922, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3924 = stablehlo.divide %v3923, %v3920 : tensor<256x128x28x28xf32>
    %v3925 = stablehlo.subtract %v3918, %v3924 : tensor<256x128x28x28xf32>
    %v3926 = stablehlo.multiply %v3925, %v3925 : tensor<256x128x28x28xf32>
    %v3927 = stablehlo.reduce(%v3926 init: %v3919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3928 = stablehlo.broadcast_in_dim %v3927, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3929 = stablehlo.divide %v3928, %v3920 : tensor<256x128x28x28xf32>
    %v3930 = stablehlo.add %v3929, %v3921 : tensor<256x128x28x28xf32>
    %v3931 = stablehlo.rsqrt %v3930 : tensor<256x128x28x28xf32>
    %v3932 = stablehlo.multiply %v3925, %v3931 : tensor<256x128x28x28xf32>
    %v3933 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3934 = stablehlo.reshape %v3917 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3935 = stablehlo.multiply %v3933, %v3934 : tensor<256x128x28x28xf32>
    %v3936 = stablehlo.reduce(%v3935 init: %v3919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3937 = stablehlo.broadcast_in_dim %v3936, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3938 = stablehlo.multiply %v3932, %v3935 : tensor<256x128x28x28xf32>
    %v3939 = stablehlo.reduce(%v3938 init: %v3919) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3940 = stablehlo.broadcast_in_dim %v3939, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3941 = stablehlo.multiply %v3935, %v3920 : tensor<256x128x28x28xf32>
    %v3942 = stablehlo.subtract %v3941, %v3937 : tensor<256x128x28x28xf32>
    %v3943 = stablehlo.multiply %v3932, %v3940 : tensor<256x128x28x28xf32>
    %v3944 = stablehlo.subtract %v3942, %v3943 : tensor<256x128x28x28xf32>
    %v3945 = stablehlo.divide %v3931, %v3920 : tensor<256x128x28x28xf32>
    %v3946 = stablehlo.multiply %v3945, %v3944 : tensor<256x128x28x28xf32>
    %v3947 = stablehlo.reshape %v3946 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3948 = stablehlo.reshape %v3947 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3949 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3950 = stablehlo.transpose %v3949, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3951 = stablehlo.convolution(%v3948, %v3950)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3952 = stablehlo.reshape %v3951 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3953 = stablehlo.reshape %v3952 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3954 = stablehlo.reshape %v3835 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3955 = stablehlo.add %v3953, %v3954 : tensor<256x512x28x28xf32>
    %v3956 = stablehlo.reshape %v3955 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3957 = stablehlo.reshape %v537 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3958 = stablehlo.reshape %v3947 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3959 = stablehlo.transpose %v3957, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3960 = stablehlo.transpose %v3958, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3961 = stablehlo.convolution(%v3959, %v3960)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3962 = stablehlo.transpose %v3961, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3963 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3965 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3966 = stablehlo.reduce(%v3963 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3967 = stablehlo.broadcast_in_dim %v3966, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3968 = stablehlo.divide %v3967, %v3965 : tensor<256x128x28x28xf32>
    %v3969 = stablehlo.subtract %v3963, %v3968 : tensor<256x128x28x28xf32>
    %v3970 = stablehlo.multiply %v3969, %v3969 : tensor<256x128x28x28xf32>
    %v3971 = stablehlo.reduce(%v3970 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3972 = stablehlo.broadcast_in_dim %v3971, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3973 = stablehlo.divide %v3972, %v3965 : tensor<256x128x28x28xf32>
    %v3974 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3975 = stablehlo.add %v3973, %v3974 : tensor<256x128x28x28xf32>
    %v3976 = stablehlo.rsqrt %v3975 : tensor<256x128x28x28xf32>
    %v3977 = stablehlo.multiply %v3969, %v3976 : tensor<256x128x28x28xf32>
    %v3978 = stablehlo.reshape %v3917 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3979 = stablehlo.multiply %v3978, %v3977 : tensor<256x128x28x28xf32>
    %v3980 = stablehlo.reduce(%v3979 init: %v3964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3981 = stablehlo.reshape %v3917 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3983 = stablehlo.reduce(%v3981 init: %v3982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3984 = stablehlo.reshape %v566 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3985 = stablehlo.reshape %v3906 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3986 = stablehlo.transpose %v3984, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3987 = stablehlo.transpose %v3985, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3988 = stablehlo.convolution(%v3986, %v3987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3989 = stablehlo.transpose %v3988, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3990 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3993 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3995 = stablehlo.divide %v3994, %v3992 : tensor<256x128x28x28xf32>
    %v3996 = stablehlo.subtract %v3990, %v3995 : tensor<256x128x28x28xf32>
    %v3997 = stablehlo.multiply %v3996, %v3996 : tensor<256x128x28x28xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3998, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4000 = stablehlo.divide %v3999, %v3992 : tensor<256x128x28x28xf32>
    %v4001 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4002 = stablehlo.add %v4000, %v4001 : tensor<256x128x28x28xf32>
    %v4003 = stablehlo.rsqrt %v4002 : tensor<256x128x28x28xf32>
    %v4004 = stablehlo.multiply %v3996, %v4003 : tensor<256x128x28x28xf32>
    %v4005 = stablehlo.reshape %v3876 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4006 = stablehlo.multiply %v4005, %v4004 : tensor<256x128x28x28xf32>
    %v4007 = stablehlo.reduce(%v4006 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4008 = stablehlo.reshape %v3876 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4010 = stablehlo.reduce(%v4008 init: %v4009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4011 = stablehlo.reshape %v595 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4012 = stablehlo.reshape %v3865 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4013 = stablehlo.transpose %v4011, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4014 = stablehlo.transpose %v4012, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4015 = stablehlo.convolution(%v4013, %v4014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4016 = stablehlo.transpose %v4015, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4017 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4021 = stablehlo.broadcast_in_dim %v4020, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4022 = stablehlo.divide %v4021, %v4019 : tensor<256x512x28x28xf32>
    %v4023 = stablehlo.subtract %v4017, %v4022 : tensor<256x512x28x28xf32>
    %v4024 = stablehlo.multiply %v4023, %v4023 : tensor<256x512x28x28xf32>
    %v4025 = stablehlo.reduce(%v4024 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4027 = stablehlo.divide %v4026, %v4019 : tensor<256x512x28x28xf32>
    %v4028 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4029 = stablehlo.add %v4027, %v4028 : tensor<256x512x28x28xf32>
    %v4030 = stablehlo.rsqrt %v4029 : tensor<256x512x28x28xf32>
    %v4031 = stablehlo.multiply %v4023, %v4030 : tensor<256x512x28x28xf32>
    %v4032 = stablehlo.reshape %v3835 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4033 = stablehlo.multiply %v4032, %v4031 : tensor<256x512x28x28xf32>
    %v4034 = stablehlo.reduce(%v4033 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4035 = stablehlo.reshape %v3835 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4036 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4037 = stablehlo.reduce(%v4035 init: %v4036) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4038 = stablehlo.reshape %v3956 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4039 = stablehlo.reshape %v533 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v4041 = stablehlo.compare GT, %v4039, %v4040 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v4042 = stablehlo.select %v4041, %v4038, %v4040 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v4043 = stablehlo.reshape %v4042 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4044 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4046 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4047 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4048 = stablehlo.reduce(%v4044 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4049 = stablehlo.broadcast_in_dim %v4048, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4050 = stablehlo.divide %v4049, %v4046 : tensor<256x512x28x28xf32>
    %v4051 = stablehlo.subtract %v4044, %v4050 : tensor<256x512x28x28xf32>
    %v4052 = stablehlo.multiply %v4051, %v4051 : tensor<256x512x28x28xf32>
    %v4053 = stablehlo.reduce(%v4052 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4053, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4055 = stablehlo.divide %v4054, %v4046 : tensor<256x512x28x28xf32>
    %v4056 = stablehlo.add %v4055, %v4047 : tensor<256x512x28x28xf32>
    %v4057 = stablehlo.rsqrt %v4056 : tensor<256x512x28x28xf32>
    %v4058 = stablehlo.multiply %v4051, %v4057 : tensor<256x512x28x28xf32>
    %v4059 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4060 = stablehlo.reshape %v4043 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4061 = stablehlo.multiply %v4059, %v4060 : tensor<256x512x28x28xf32>
    %v4062 = stablehlo.reduce(%v4061 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4063 = stablehlo.broadcast_in_dim %v4062, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4064 = stablehlo.multiply %v4058, %v4061 : tensor<256x512x28x28xf32>
    %v4065 = stablehlo.reduce(%v4064 init: %v4045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4066 = stablehlo.broadcast_in_dim %v4065, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4067 = stablehlo.multiply %v4061, %v4046 : tensor<256x512x28x28xf32>
    %v4068 = stablehlo.subtract %v4067, %v4063 : tensor<256x512x28x28xf32>
    %v4069 = stablehlo.multiply %v4058, %v4066 : tensor<256x512x28x28xf32>
    %v4070 = stablehlo.subtract %v4068, %v4069 : tensor<256x512x28x28xf32>
    %v4071 = stablehlo.divide %v4057, %v4046 : tensor<256x512x28x28xf32>
    %v4072 = stablehlo.multiply %v4071, %v4070 : tensor<256x512x28x28xf32>
    %v4073 = stablehlo.reshape %v4072 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4074 = stablehlo.reshape %v4073 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4075 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4076 = stablehlo.transpose %v4075, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4077 = stablehlo.convolution(%v4074, %v4076)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4079 = stablehlo.reshape %v4078 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4080 = stablehlo.reshape %v500 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4081 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4082 = stablehlo.compare GT, %v4080, %v4081 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4083 = stablehlo.select %v4082, %v4079, %v4081 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4084 = stablehlo.reshape %v4083 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4085 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4087 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4088 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4089 = stablehlo.reduce(%v4085 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4091 = stablehlo.divide %v4090, %v4087 : tensor<256x128x28x28xf32>
    %v4092 = stablehlo.subtract %v4085, %v4091 : tensor<256x128x28x28xf32>
    %v4093 = stablehlo.multiply %v4092, %v4092 : tensor<256x128x28x28xf32>
    %v4094 = stablehlo.reduce(%v4093 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4095 = stablehlo.broadcast_in_dim %v4094, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4096 = stablehlo.divide %v4095, %v4087 : tensor<256x128x28x28xf32>
    %v4097 = stablehlo.add %v4096, %v4088 : tensor<256x128x28x28xf32>
    %v4098 = stablehlo.rsqrt %v4097 : tensor<256x128x28x28xf32>
    %v4099 = stablehlo.multiply %v4092, %v4098 : tensor<256x128x28x28xf32>
    %v4100 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4101 = stablehlo.reshape %v4084 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4102 = stablehlo.multiply %v4100, %v4101 : tensor<256x128x28x28xf32>
    %v4103 = stablehlo.reduce(%v4102 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4104 = stablehlo.broadcast_in_dim %v4103, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4105 = stablehlo.multiply %v4099, %v4102 : tensor<256x128x28x28xf32>
    %v4106 = stablehlo.reduce(%v4105 init: %v4086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4107 = stablehlo.broadcast_in_dim %v4106, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4108 = stablehlo.multiply %v4102, %v4087 : tensor<256x128x28x28xf32>
    %v4109 = stablehlo.subtract %v4108, %v4104 : tensor<256x128x28x28xf32>
    %v4110 = stablehlo.multiply %v4099, %v4107 : tensor<256x128x28x28xf32>
    %v4111 = stablehlo.subtract %v4109, %v4110 : tensor<256x128x28x28xf32>
    %v4112 = stablehlo.divide %v4098, %v4087 : tensor<256x128x28x28xf32>
    %v4113 = stablehlo.multiply %v4112, %v4111 : tensor<256x128x28x28xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4115 = stablehlo.reshape %v4114 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4116 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4117 = stablehlo.transpose %v4116, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4118 = stablehlo.convolution(%v4115, %v4117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v4119 = stablehlo.reshape %v4118 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4120 = stablehlo.reshape %v4119 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4121 = stablehlo.reshape %v471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4122 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4123 = stablehlo.compare GT, %v4121, %v4122 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4124 = stablehlo.select %v4123, %v4120, %v4122 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4125 = stablehlo.reshape %v4124 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4126 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4128 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4129 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4130 = stablehlo.reduce(%v4126 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4131 = stablehlo.broadcast_in_dim %v4130, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4132 = stablehlo.divide %v4131, %v4128 : tensor<256x128x28x28xf32>
    %v4133 = stablehlo.subtract %v4126, %v4132 : tensor<256x128x28x28xf32>
    %v4134 = stablehlo.multiply %v4133, %v4133 : tensor<256x128x28x28xf32>
    %v4135 = stablehlo.reduce(%v4134 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4136 = stablehlo.broadcast_in_dim %v4135, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4137 = stablehlo.divide %v4136, %v4128 : tensor<256x128x28x28xf32>
    %v4138 = stablehlo.add %v4137, %v4129 : tensor<256x128x28x28xf32>
    %v4139 = stablehlo.rsqrt %v4138 : tensor<256x128x28x28xf32>
    %v4140 = stablehlo.multiply %v4133, %v4139 : tensor<256x128x28x28xf32>
    %v4141 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4142 = stablehlo.reshape %v4125 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4143 = stablehlo.multiply %v4141, %v4142 : tensor<256x128x28x28xf32>
    %v4144 = stablehlo.reduce(%v4143 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4145 = stablehlo.broadcast_in_dim %v4144, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4146 = stablehlo.multiply %v4140, %v4143 : tensor<256x128x28x28xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4149 = stablehlo.multiply %v4143, %v4128 : tensor<256x128x28x28xf32>
    %v4150 = stablehlo.subtract %v4149, %v4145 : tensor<256x128x28x28xf32>
    %v4151 = stablehlo.multiply %v4140, %v4148 : tensor<256x128x28x28xf32>
    %v4152 = stablehlo.subtract %v4150, %v4151 : tensor<256x128x28x28xf32>
    %v4153 = stablehlo.divide %v4139, %v4128 : tensor<256x128x28x28xf32>
    %v4154 = stablehlo.multiply %v4153, %v4152 : tensor<256x128x28x28xf32>
    %v4155 = stablehlo.reshape %v4154 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4156 = stablehlo.reshape %v4155 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4157 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4158 = stablehlo.transpose %v4157, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4159 = stablehlo.convolution(%v4156, %v4158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v4160 = stablehlo.reshape %v4159 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4161 = stablehlo.reshape %v4160 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4162 = stablehlo.reshape %v4043 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4163 = stablehlo.add %v4161, %v4162 : tensor<256x512x28x28xf32>
    %v4164 = stablehlo.reshape %v4163 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4165 = stablehlo.reshape %v446 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4166 = stablehlo.reshape %v4155 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4167 = stablehlo.transpose %v4165, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4168 = stablehlo.transpose %v4166, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4169 = stablehlo.convolution(%v4167, %v4168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v4170 = stablehlo.transpose %v4169, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4171 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4173 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4174 = stablehlo.reduce(%v4171 init: %v4172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4175 = stablehlo.broadcast_in_dim %v4174, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4176 = stablehlo.divide %v4175, %v4173 : tensor<256x128x28x28xf32>
    %v4177 = stablehlo.subtract %v4171, %v4176 : tensor<256x128x28x28xf32>
    %v4178 = stablehlo.multiply %v4177, %v4177 : tensor<256x128x28x28xf32>
    %v4179 = stablehlo.reduce(%v4178 init: %v4172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4180 = stablehlo.broadcast_in_dim %v4179, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4181 = stablehlo.divide %v4180, %v4173 : tensor<256x128x28x28xf32>
    %v4182 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4183 = stablehlo.add %v4181, %v4182 : tensor<256x128x28x28xf32>
    %v4184 = stablehlo.rsqrt %v4183 : tensor<256x128x28x28xf32>
    %v4185 = stablehlo.multiply %v4177, %v4184 : tensor<256x128x28x28xf32>
    %v4186 = stablehlo.reshape %v4125 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4187 = stablehlo.multiply %v4186, %v4185 : tensor<256x128x28x28xf32>
    %v4188 = stablehlo.reduce(%v4187 init: %v4172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4189 = stablehlo.reshape %v4125 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4191 = stablehlo.reduce(%v4189 init: %v4190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4192 = stablehlo.reshape %v475 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4193 = stablehlo.reshape %v4114 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4194 = stablehlo.transpose %v4192, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4195 = stablehlo.transpose %v4193, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4196 = stablehlo.convolution(%v4194, %v4195)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v4197 = stablehlo.transpose %v4196, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4198 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4200 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4203 = stablehlo.divide %v4202, %v4200 : tensor<256x128x28x28xf32>
    %v4204 = stablehlo.subtract %v4198, %v4203 : tensor<256x128x28x28xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<256x128x28x28xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4208 = stablehlo.divide %v4207, %v4200 : tensor<256x128x28x28xf32>
    %v4209 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4210 = stablehlo.add %v4208, %v4209 : tensor<256x128x28x28xf32>
    %v4211 = stablehlo.rsqrt %v4210 : tensor<256x128x28x28xf32>
    %v4212 = stablehlo.multiply %v4204, %v4211 : tensor<256x128x28x28xf32>
    %v4213 = stablehlo.reshape %v4084 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4214 = stablehlo.multiply %v4213, %v4212 : tensor<256x128x28x28xf32>
    %v4215 = stablehlo.reduce(%v4214 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4216 = stablehlo.reshape %v4084 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4217 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4218 = stablehlo.reduce(%v4216 init: %v4217) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4219 = stablehlo.reshape %v504 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4220 = stablehlo.reshape %v4073 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4221 = stablehlo.transpose %v4219, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4222 = stablehlo.transpose %v4220, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4223 = stablehlo.convolution(%v4221, %v4222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4224 = stablehlo.transpose %v4223, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4225 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4228 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4229 = stablehlo.broadcast_in_dim %v4228, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4230 = stablehlo.divide %v4229, %v4227 : tensor<256x512x28x28xf32>
    %v4231 = stablehlo.subtract %v4225, %v4230 : tensor<256x512x28x28xf32>
    %v4232 = stablehlo.multiply %v4231, %v4231 : tensor<256x512x28x28xf32>
    %v4233 = stablehlo.reduce(%v4232 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4234 = stablehlo.broadcast_in_dim %v4233, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4235 = stablehlo.divide %v4234, %v4227 : tensor<256x512x28x28xf32>
    %v4236 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4237 = stablehlo.add %v4235, %v4236 : tensor<256x512x28x28xf32>
    %v4238 = stablehlo.rsqrt %v4237 : tensor<256x512x28x28xf32>
    %v4239 = stablehlo.multiply %v4231, %v4238 : tensor<256x512x28x28xf32>
    %v4240 = stablehlo.reshape %v4043 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4241 = stablehlo.multiply %v4240, %v4239 : tensor<256x512x28x28xf32>
    %v4242 = stablehlo.reduce(%v4241 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4243 = stablehlo.reshape %v4043 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4245 = stablehlo.reduce(%v4243 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4246 = stablehlo.reshape %v4164 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4247 = stablehlo.reshape %v442 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4248 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v4249 = stablehlo.compare GT, %v4247, %v4248 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v4250 = stablehlo.select %v4249, %v4246, %v4248 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v4251 = stablehlo.reshape %v4250 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4252 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4254 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4255 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4256 = stablehlo.reduce(%v4252 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4257 = stablehlo.broadcast_in_dim %v4256, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4258 = stablehlo.divide %v4257, %v4254 : tensor<256x512x28x28xf32>
    %v4259 = stablehlo.subtract %v4252, %v4258 : tensor<256x512x28x28xf32>
    %v4260 = stablehlo.multiply %v4259, %v4259 : tensor<256x512x28x28xf32>
    %v4261 = stablehlo.reduce(%v4260 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4262 = stablehlo.broadcast_in_dim %v4261, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4263 = stablehlo.divide %v4262, %v4254 : tensor<256x512x28x28xf32>
    %v4264 = stablehlo.add %v4263, %v4255 : tensor<256x512x28x28xf32>
    %v4265 = stablehlo.rsqrt %v4264 : tensor<256x512x28x28xf32>
    %v4266 = stablehlo.multiply %v4259, %v4265 : tensor<256x512x28x28xf32>
    %v4267 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4268 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4269 = stablehlo.multiply %v4267, %v4268 : tensor<256x512x28x28xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4271 = stablehlo.broadcast_in_dim %v4270, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4272 = stablehlo.multiply %v4266, %v4269 : tensor<256x512x28x28xf32>
    %v4273 = stablehlo.reduce(%v4272 init: %v4253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4274 = stablehlo.broadcast_in_dim %v4273, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4275 = stablehlo.multiply %v4269, %v4254 : tensor<256x512x28x28xf32>
    %v4276 = stablehlo.subtract %v4275, %v4271 : tensor<256x512x28x28xf32>
    %v4277 = stablehlo.multiply %v4266, %v4274 : tensor<256x512x28x28xf32>
    %v4278 = stablehlo.subtract %v4276, %v4277 : tensor<256x512x28x28xf32>
    %v4279 = stablehlo.divide %v4265, %v4254 : tensor<256x512x28x28xf32>
    %v4280 = stablehlo.multiply %v4279, %v4278 : tensor<256x512x28x28xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4282 = stablehlo.reshape %v4281 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4283 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4284 = stablehlo.transpose %v4283, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4285 = stablehlo.convolution(%v4282, %v4284)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v4286 = stablehlo.reshape %v4285 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4287 = stablehlo.reshape %v4286 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4288 = stablehlo.reshape %v384 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4289 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4290 = stablehlo.compare GT, %v4288, %v4289 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4291 = stablehlo.select %v4290, %v4287, %v4289 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4292 = stablehlo.reshape %v4291 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4293 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4295 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4296 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4297 = stablehlo.reduce(%v4293 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4298 = stablehlo.broadcast_in_dim %v4297, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4299 = stablehlo.divide %v4298, %v4295 : tensor<256x128x28x28xf32>
    %v4300 = stablehlo.subtract %v4293, %v4299 : tensor<256x128x28x28xf32>
    %v4301 = stablehlo.multiply %v4300, %v4300 : tensor<256x128x28x28xf32>
    %v4302 = stablehlo.reduce(%v4301 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4303 = stablehlo.broadcast_in_dim %v4302, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4304 = stablehlo.divide %v4303, %v4295 : tensor<256x128x28x28xf32>
    %v4305 = stablehlo.add %v4304, %v4296 : tensor<256x128x28x28xf32>
    %v4306 = stablehlo.rsqrt %v4305 : tensor<256x128x28x28xf32>
    %v4307 = stablehlo.multiply %v4300, %v4306 : tensor<256x128x28x28xf32>
    %v4308 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4309 = stablehlo.reshape %v4292 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4310 = stablehlo.multiply %v4308, %v4309 : tensor<256x128x28x28xf32>
    %v4311 = stablehlo.reduce(%v4310 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4312 = stablehlo.broadcast_in_dim %v4311, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4313 = stablehlo.multiply %v4307, %v4310 : tensor<256x128x28x28xf32>
    %v4314 = stablehlo.reduce(%v4313 init: %v4294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4315 = stablehlo.broadcast_in_dim %v4314, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4316 = stablehlo.multiply %v4310, %v4295 : tensor<256x128x28x28xf32>
    %v4317 = stablehlo.subtract %v4316, %v4312 : tensor<256x128x28x28xf32>
    %v4318 = stablehlo.multiply %v4307, %v4315 : tensor<256x128x28x28xf32>
    %v4319 = stablehlo.subtract %v4317, %v4318 : tensor<256x128x28x28xf32>
    %v4320 = stablehlo.divide %v4306, %v4295 : tensor<256x128x28x28xf32>
    %v4321 = stablehlo.multiply %v4320, %v4319 : tensor<256x128x28x28xf32>
    %v4322 = stablehlo.reshape %v4321 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4323 = stablehlo.reshape %v4322 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4324 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4325 = stablehlo.transpose %v4324, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4326 = stablehlo.convolution(%v4323, %v4325)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x56x56xf32>
    %v4327 = stablehlo.reshape %v4326 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4328 = stablehlo.reshape %v4327 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4329 = stablehlo.reshape %v355 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4330 = stablehlo.constant dense<0.0> : tensor<256x128x56x56xf32>
    %v4331 = stablehlo.compare GT, %v4329, %v4330 : (tensor<256x128x56x56xf32>, tensor<256x128x56x56xf32>) -> tensor<256x128x56x56xi1>
    %v4332 = stablehlo.select %v4331, %v4328, %v4330 : tensor<256x128x56x56xi1>, tensor<256x128x56x56xf32>
    %v4333 = stablehlo.reshape %v4332 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4334 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4336 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4337 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4338 = stablehlo.reduce(%v4334 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4339 = stablehlo.broadcast_in_dim %v4338, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4340 = stablehlo.divide %v4339, %v4336 : tensor<256x128x56x56xf32>
    %v4341 = stablehlo.subtract %v4334, %v4340 : tensor<256x128x56x56xf32>
    %v4342 = stablehlo.multiply %v4341, %v4341 : tensor<256x128x56x56xf32>
    %v4343 = stablehlo.reduce(%v4342 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4344 = stablehlo.broadcast_in_dim %v4343, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4345 = stablehlo.divide %v4344, %v4336 : tensor<256x128x56x56xf32>
    %v4346 = stablehlo.add %v4345, %v4337 : tensor<256x128x56x56xf32>
    %v4347 = stablehlo.rsqrt %v4346 : tensor<256x128x56x56xf32>
    %v4348 = stablehlo.multiply %v4341, %v4347 : tensor<256x128x56x56xf32>
    %v4349 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4350 = stablehlo.reshape %v4333 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4351 = stablehlo.multiply %v4349, %v4350 : tensor<256x128x56x56xf32>
    %v4352 = stablehlo.reduce(%v4351 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4353 = stablehlo.broadcast_in_dim %v4352, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4354 = stablehlo.multiply %v4348, %v4351 : tensor<256x128x56x56xf32>
    %v4355 = stablehlo.reduce(%v4354 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4356 = stablehlo.broadcast_in_dim %v4355, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4357 = stablehlo.multiply %v4351, %v4336 : tensor<256x128x56x56xf32>
    %v4358 = stablehlo.subtract %v4357, %v4353 : tensor<256x128x56x56xf32>
    %v4359 = stablehlo.multiply %v4348, %v4356 : tensor<256x128x56x56xf32>
    %v4360 = stablehlo.subtract %v4358, %v4359 : tensor<256x128x56x56xf32>
    %v4361 = stablehlo.divide %v4347, %v4336 : tensor<256x128x56x56xf32>
    %v4362 = stablehlo.multiply %v4361, %v4360 : tensor<256x128x56x56xf32>
    %v4363 = stablehlo.reshape %v4362 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4364 = stablehlo.reshape %v4363 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4365 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v4366 = stablehlo.transpose %v4365, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v4367 = stablehlo.convolution(%v4364, %v4366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4368 = stablehlo.reshape %v4367 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4369 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4371 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4372 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4373 = stablehlo.reduce(%v4369 init: %v4370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4374 = stablehlo.broadcast_in_dim %v4373, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4375 = stablehlo.divide %v4374, %v4371 : tensor<256x512x28x28xf32>
    %v4376 = stablehlo.subtract %v4369, %v4375 : tensor<256x512x28x28xf32>
    %v4377 = stablehlo.multiply %v4376, %v4376 : tensor<256x512x28x28xf32>
    %v4378 = stablehlo.reduce(%v4377 init: %v4370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4379 = stablehlo.broadcast_in_dim %v4378, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4380 = stablehlo.divide %v4379, %v4371 : tensor<256x512x28x28xf32>
    %v4381 = stablehlo.add %v4380, %v4372 : tensor<256x512x28x28xf32>
    %v4382 = stablehlo.rsqrt %v4381 : tensor<256x512x28x28xf32>
    %v4383 = stablehlo.multiply %v4376, %v4382 : tensor<256x512x28x28xf32>
    %v4384 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4385 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4386 = stablehlo.multiply %v4384, %v4385 : tensor<256x512x28x28xf32>
    %v4387 = stablehlo.reduce(%v4386 init: %v4370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4388 = stablehlo.broadcast_in_dim %v4387, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4389 = stablehlo.multiply %v4383, %v4386 : tensor<256x512x28x28xf32>
    %v4390 = stablehlo.reduce(%v4389 init: %v4370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4391 = stablehlo.broadcast_in_dim %v4390, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4392 = stablehlo.multiply %v4386, %v4371 : tensor<256x512x28x28xf32>
    %v4393 = stablehlo.subtract %v4392, %v4388 : tensor<256x512x28x28xf32>
    %v4394 = stablehlo.multiply %v4383, %v4391 : tensor<256x512x28x28xf32>
    %v4395 = stablehlo.subtract %v4393, %v4394 : tensor<256x512x28x28xf32>
    %v4396 = stablehlo.divide %v4382, %v4371 : tensor<256x512x28x28xf32>
    %v4397 = stablehlo.multiply %v4396, %v4395 : tensor<256x512x28x28xf32>
    %v4398 = stablehlo.reshape %v4397 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4399 = stablehlo.reshape %v4398 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4400 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v4401 = stablehlo.transpose %v4400, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4402 = stablehlo.convolution(%v4399, %v4401)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4403 = stablehlo.reshape %v4402 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4404 = stablehlo.reshape %v4368 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4405 = stablehlo.reshape %v4403 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4406 = stablehlo.add %v4404, %v4405 : tensor<256x256x56x56xf32>
    %v4407 = stablehlo.reshape %v4406 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4408 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4409 = stablehlo.reshape %v4363 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4410 = stablehlo.transpose %v4408, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4411 = stablehlo.transpose %v4409, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4412 = stablehlo.convolution(%v4410, %v4411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<256x128x1x1xf32>
    %v4413 = stablehlo.transpose %v4412, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v4414 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4416 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4417 = stablehlo.reduce(%v4414 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4418 = stablehlo.broadcast_in_dim %v4417, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4419 = stablehlo.divide %v4418, %v4416 : tensor<256x128x56x56xf32>
    %v4420 = stablehlo.subtract %v4414, %v4419 : tensor<256x128x56x56xf32>
    %v4421 = stablehlo.multiply %v4420, %v4420 : tensor<256x128x56x56xf32>
    %v4422 = stablehlo.reduce(%v4421 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4423 = stablehlo.broadcast_in_dim %v4422, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4424 = stablehlo.divide %v4423, %v4416 : tensor<256x128x56x56xf32>
    %v4425 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4426 = stablehlo.add %v4424, %v4425 : tensor<256x128x56x56xf32>
    %v4427 = stablehlo.rsqrt %v4426 : tensor<256x128x56x56xf32>
    %v4428 = stablehlo.multiply %v4420, %v4427 : tensor<256x128x56x56xf32>
    %v4429 = stablehlo.reshape %v4333 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4430 = stablehlo.multiply %v4429, %v4428 : tensor<256x128x56x56xf32>
    %v4431 = stablehlo.reduce(%v4430 init: %v4415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4432 = stablehlo.reshape %v4333 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4434 = stablehlo.reduce(%v4432 init: %v4433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4435 = stablehlo.reshape %v359 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4436 = stablehlo.reshape %v4322 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4437 = stablehlo.transpose %v4435, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4439 = stablehlo.transpose %v4436, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4440 = stablehlo.convolution(%v4437, %v4439)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x56x56xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v4441 = stablehlo.transpose %v4440, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4442 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4444 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4445 = stablehlo.reduce(%v4442 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4446 = stablehlo.broadcast_in_dim %v4445, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4447 = stablehlo.divide %v4446, %v4444 : tensor<256x128x28x28xf32>
    %v4448 = stablehlo.subtract %v4442, %v4447 : tensor<256x128x28x28xf32>
    %v4449 = stablehlo.multiply %v4448, %v4448 : tensor<256x128x28x28xf32>
    %v4450 = stablehlo.reduce(%v4449 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4451 = stablehlo.broadcast_in_dim %v4450, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4452 = stablehlo.divide %v4451, %v4444 : tensor<256x128x28x28xf32>
    %v4453 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4454 = stablehlo.add %v4452, %v4453 : tensor<256x128x28x28xf32>
    %v4455 = stablehlo.rsqrt %v4454 : tensor<256x128x28x28xf32>
    %v4456 = stablehlo.multiply %v4448, %v4455 : tensor<256x128x28x28xf32>
    %v4457 = stablehlo.reshape %v4292 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4458 = stablehlo.multiply %v4457, %v4456 : tensor<256x128x28x28xf32>
    %v4459 = stablehlo.reduce(%v4458 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4460 = stablehlo.reshape %v4292 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4462 = stablehlo.reduce(%v4460 init: %v4461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4463 = stablehlo.reshape %v388 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4464 = stablehlo.reshape %v4281 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4465 = stablehlo.transpose %v4463, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4466 = stablehlo.transpose %v4464, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4467 = stablehlo.convolution(%v4465, %v4466)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4468 = stablehlo.transpose %v4467, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4469 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4471 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4472 = stablehlo.reduce(%v4469 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4473 = stablehlo.broadcast_in_dim %v4472, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4474 = stablehlo.divide %v4473, %v4471 : tensor<256x512x28x28xf32>
    %v4475 = stablehlo.subtract %v4469, %v4474 : tensor<256x512x28x28xf32>
    %v4476 = stablehlo.multiply %v4475, %v4475 : tensor<256x512x28x28xf32>
    %v4477 = stablehlo.reduce(%v4476 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4478 = stablehlo.broadcast_in_dim %v4477, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4479 = stablehlo.divide %v4478, %v4471 : tensor<256x512x28x28xf32>
    %v4480 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4481 = stablehlo.add %v4479, %v4480 : tensor<256x512x28x28xf32>
    %v4482 = stablehlo.rsqrt %v4481 : tensor<256x512x28x28xf32>
    %v4483 = stablehlo.multiply %v4475, %v4482 : tensor<256x512x28x28xf32>
    %v4484 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4485 = stablehlo.multiply %v4484, %v4483 : tensor<256x512x28x28xf32>
    %v4486 = stablehlo.reduce(%v4485 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4487 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4489 = stablehlo.reduce(%v4487 init: %v4488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4490 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4491 = stablehlo.reshape %v4398 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4492 = stablehlo.transpose %v4490, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4494 = stablehlo.transpose %v4491, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4493 = stablehlo.slice %v4492 [0:256, 0:256, 0:55, 0:55] : (tensor<256x256x56x56xf32>) -> tensor<256x256x55x55xf32>
    %v4495 = stablehlo.convolution(%v4493, %v4494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x55x55xf32>, tensor<512x256x28x28xf32>) -> tensor<256x512x1x1xf32>
    %v4496 = stablehlo.transpose %v4495, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4497 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4499 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4500 = stablehlo.reduce(%v4497 init: %v4498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4501 = stablehlo.broadcast_in_dim %v4500, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4502 = stablehlo.divide %v4501, %v4499 : tensor<256x512x28x28xf32>
    %v4503 = stablehlo.subtract %v4497, %v4502 : tensor<256x512x28x28xf32>
    %v4504 = stablehlo.multiply %v4503, %v4503 : tensor<256x512x28x28xf32>
    %v4505 = stablehlo.reduce(%v4504 init: %v4498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4506 = stablehlo.broadcast_in_dim %v4505, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4507 = stablehlo.divide %v4506, %v4499 : tensor<256x512x28x28xf32>
    %v4508 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4509 = stablehlo.add %v4507, %v4508 : tensor<256x512x28x28xf32>
    %v4510 = stablehlo.rsqrt %v4509 : tensor<256x512x28x28xf32>
    %v4511 = stablehlo.multiply %v4503, %v4510 : tensor<256x512x28x28xf32>
    %v4512 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4513 = stablehlo.multiply %v4512, %v4511 : tensor<256x512x28x28xf32>
    %v4514 = stablehlo.reduce(%v4513 init: %v4498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4515 = stablehlo.reshape %v4251 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4517 = stablehlo.reduce(%v4515 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4518 = stablehlo.reshape %v4407 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4519 = stablehlo.reshape %v326 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4520 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4521 = stablehlo.compare GT, %v4519, %v4520 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4522 = stablehlo.select %v4521, %v4518, %v4520 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4523 = stablehlo.reshape %v4522 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4524 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4526 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4527 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4528 = stablehlo.reduce(%v4524 init: %v4525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4529 = stablehlo.broadcast_in_dim %v4528, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4530 = stablehlo.divide %v4529, %v4526 : tensor<256x256x56x56xf32>
    %v4531 = stablehlo.subtract %v4524, %v4530 : tensor<256x256x56x56xf32>
    %v4532 = stablehlo.multiply %v4531, %v4531 : tensor<256x256x56x56xf32>
    %v4533 = stablehlo.reduce(%v4532 init: %v4525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4534 = stablehlo.broadcast_in_dim %v4533, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4535 = stablehlo.divide %v4534, %v4526 : tensor<256x256x56x56xf32>
    %v4536 = stablehlo.add %v4535, %v4527 : tensor<256x256x56x56xf32>
    %v4537 = stablehlo.rsqrt %v4536 : tensor<256x256x56x56xf32>
    %v4538 = stablehlo.multiply %v4531, %v4537 : tensor<256x256x56x56xf32>
    %v4539 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4540 = stablehlo.reshape %v4523 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4541 = stablehlo.multiply %v4539, %v4540 : tensor<256x256x56x56xf32>
    %v4542 = stablehlo.reduce(%v4541 init: %v4525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4543 = stablehlo.broadcast_in_dim %v4542, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4544 = stablehlo.multiply %v4538, %v4541 : tensor<256x256x56x56xf32>
    %v4545 = stablehlo.reduce(%v4544 init: %v4525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4546 = stablehlo.broadcast_in_dim %v4545, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4547 = stablehlo.multiply %v4541, %v4526 : tensor<256x256x56x56xf32>
    %v4548 = stablehlo.subtract %v4547, %v4543 : tensor<256x256x56x56xf32>
    %v4549 = stablehlo.multiply %v4538, %v4546 : tensor<256x256x56x56xf32>
    %v4550 = stablehlo.subtract %v4548, %v4549 : tensor<256x256x56x56xf32>
    %v4551 = stablehlo.divide %v4537, %v4526 : tensor<256x256x56x56xf32>
    %v4552 = stablehlo.multiply %v4551, %v4550 : tensor<256x256x56x56xf32>
    %v4553 = stablehlo.reshape %v4552 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4554 = stablehlo.reshape %v4553 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4555 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4556 = stablehlo.transpose %v4555, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4557 = stablehlo.convolution(%v4554, %v4556)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4558 = stablehlo.reshape %v4557 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4559 = stablehlo.reshape %v4558 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4560 = stablehlo.reshape %v293 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4561 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4562 = stablehlo.compare GT, %v4560, %v4561 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4563 = stablehlo.select %v4562, %v4559, %v4561 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4564 = stablehlo.reshape %v4563 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4565 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4567 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4568 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4569 = stablehlo.reduce(%v4565 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4570 = stablehlo.broadcast_in_dim %v4569, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4571 = stablehlo.divide %v4570, %v4567 : tensor<256x64x56x56xf32>
    %v4572 = stablehlo.subtract %v4565, %v4571 : tensor<256x64x56x56xf32>
    %v4573 = stablehlo.multiply %v4572, %v4572 : tensor<256x64x56x56xf32>
    %v4574 = stablehlo.reduce(%v4573 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4575 = stablehlo.broadcast_in_dim %v4574, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4576 = stablehlo.divide %v4575, %v4567 : tensor<256x64x56x56xf32>
    %v4577 = stablehlo.add %v4576, %v4568 : tensor<256x64x56x56xf32>
    %v4578 = stablehlo.rsqrt %v4577 : tensor<256x64x56x56xf32>
    %v4579 = stablehlo.multiply %v4572, %v4578 : tensor<256x64x56x56xf32>
    %v4580 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4581 = stablehlo.reshape %v4564 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4582 = stablehlo.multiply %v4580, %v4581 : tensor<256x64x56x56xf32>
    %v4583 = stablehlo.reduce(%v4582 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4584 = stablehlo.broadcast_in_dim %v4583, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4585 = stablehlo.multiply %v4579, %v4582 : tensor<256x64x56x56xf32>
    %v4586 = stablehlo.reduce(%v4585 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4587 = stablehlo.broadcast_in_dim %v4586, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4588 = stablehlo.multiply %v4582, %v4567 : tensor<256x64x56x56xf32>
    %v4589 = stablehlo.subtract %v4588, %v4584 : tensor<256x64x56x56xf32>
    %v4590 = stablehlo.multiply %v4579, %v4587 : tensor<256x64x56x56xf32>
    %v4591 = stablehlo.subtract %v4589, %v4590 : tensor<256x64x56x56xf32>
    %v4592 = stablehlo.divide %v4578, %v4567 : tensor<256x64x56x56xf32>
    %v4593 = stablehlo.multiply %v4592, %v4591 : tensor<256x64x56x56xf32>
    %v4594 = stablehlo.reshape %v4593 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4595 = stablehlo.reshape %v4594 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4596 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4597 = stablehlo.transpose %v4596, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4598 = stablehlo.convolution(%v4595, %v4597)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4599 = stablehlo.reshape %v4598 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4600 = stablehlo.reshape %v4599 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4601 = stablehlo.reshape %v264 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4602 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4603 = stablehlo.compare GT, %v4601, %v4602 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4604 = stablehlo.select %v4603, %v4600, %v4602 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4605 = stablehlo.reshape %v4604 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4606 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4608 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4609 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4610 = stablehlo.reduce(%v4606 init: %v4607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4611 = stablehlo.broadcast_in_dim %v4610, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4612 = stablehlo.divide %v4611, %v4608 : tensor<256x64x56x56xf32>
    %v4613 = stablehlo.subtract %v4606, %v4612 : tensor<256x64x56x56xf32>
    %v4614 = stablehlo.multiply %v4613, %v4613 : tensor<256x64x56x56xf32>
    %v4615 = stablehlo.reduce(%v4614 init: %v4607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4616 = stablehlo.broadcast_in_dim %v4615, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4617 = stablehlo.divide %v4616, %v4608 : tensor<256x64x56x56xf32>
    %v4618 = stablehlo.add %v4617, %v4609 : tensor<256x64x56x56xf32>
    %v4619 = stablehlo.rsqrt %v4618 : tensor<256x64x56x56xf32>
    %v4620 = stablehlo.multiply %v4613, %v4619 : tensor<256x64x56x56xf32>
    %v4621 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4622 = stablehlo.reshape %v4605 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4623 = stablehlo.multiply %v4621, %v4622 : tensor<256x64x56x56xf32>
    %v4624 = stablehlo.reduce(%v4623 init: %v4607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %v4624, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4626 = stablehlo.multiply %v4620, %v4623 : tensor<256x64x56x56xf32>
    %v4627 = stablehlo.reduce(%v4626 init: %v4607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4628 = stablehlo.broadcast_in_dim %v4627, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4629 = stablehlo.multiply %v4623, %v4608 : tensor<256x64x56x56xf32>
    %v4630 = stablehlo.subtract %v4629, %v4625 : tensor<256x64x56x56xf32>
    %v4631 = stablehlo.multiply %v4620, %v4628 : tensor<256x64x56x56xf32>
    %v4632 = stablehlo.subtract %v4630, %v4631 : tensor<256x64x56x56xf32>
    %v4633 = stablehlo.divide %v4619, %v4608 : tensor<256x64x56x56xf32>
    %v4634 = stablehlo.multiply %v4633, %v4632 : tensor<256x64x56x56xf32>
    %v4635 = stablehlo.reshape %v4634 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4636 = stablehlo.reshape %v4635 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4637 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4638 = stablehlo.transpose %v4637, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4639 = stablehlo.convolution(%v4636, %v4638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4640 = stablehlo.reshape %v4639 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4642 = stablehlo.reshape %v4523 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4643 = stablehlo.add %v4641, %v4642 : tensor<256x256x56x56xf32>
    %v4644 = stablehlo.reshape %v4643 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4645 = stablehlo.reshape %v239 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4646 = stablehlo.reshape %v4635 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4647 = stablehlo.transpose %v4645, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4648 = stablehlo.transpose %v4646, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4649 = stablehlo.convolution(%v4647, %v4648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4650 = stablehlo.transpose %v4649, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4651 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4653 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4654 = stablehlo.reduce(%v4651 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4655 = stablehlo.broadcast_in_dim %v4654, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4656 = stablehlo.divide %v4655, %v4653 : tensor<256x64x56x56xf32>
    %v4657 = stablehlo.subtract %v4651, %v4656 : tensor<256x64x56x56xf32>
    %v4658 = stablehlo.multiply %v4657, %v4657 : tensor<256x64x56x56xf32>
    %v4659 = stablehlo.reduce(%v4658 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4660 = stablehlo.broadcast_in_dim %v4659, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4661 = stablehlo.divide %v4660, %v4653 : tensor<256x64x56x56xf32>
    %v4662 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4663 = stablehlo.add %v4661, %v4662 : tensor<256x64x56x56xf32>
    %v4664 = stablehlo.rsqrt %v4663 : tensor<256x64x56x56xf32>
    %v4665 = stablehlo.multiply %v4657, %v4664 : tensor<256x64x56x56xf32>
    %v4666 = stablehlo.reshape %v4605 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4667 = stablehlo.multiply %v4666, %v4665 : tensor<256x64x56x56xf32>
    %v4668 = stablehlo.reduce(%v4667 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4669 = stablehlo.reshape %v4605 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4671 = stablehlo.reduce(%v4669 init: %v4670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4672 = stablehlo.reshape %v268 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4673 = stablehlo.reshape %v4594 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4674 = stablehlo.transpose %v4672, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4675 = stablehlo.transpose %v4673, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4676 = stablehlo.convolution(%v4674, %v4675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4677 = stablehlo.transpose %v4676, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4678 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4680 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4681 = stablehlo.reduce(%v4678 init: %v4679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4682 = stablehlo.broadcast_in_dim %v4681, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4683 = stablehlo.divide %v4682, %v4680 : tensor<256x64x56x56xf32>
    %v4684 = stablehlo.subtract %v4678, %v4683 : tensor<256x64x56x56xf32>
    %v4685 = stablehlo.multiply %v4684, %v4684 : tensor<256x64x56x56xf32>
    %v4686 = stablehlo.reduce(%v4685 init: %v4679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4687 = stablehlo.broadcast_in_dim %v4686, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4688 = stablehlo.divide %v4687, %v4680 : tensor<256x64x56x56xf32>
    %v4689 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4690 = stablehlo.add %v4688, %v4689 : tensor<256x64x56x56xf32>
    %v4691 = stablehlo.rsqrt %v4690 : tensor<256x64x56x56xf32>
    %v4692 = stablehlo.multiply %v4684, %v4691 : tensor<256x64x56x56xf32>
    %v4693 = stablehlo.reshape %v4564 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4694 = stablehlo.multiply %v4693, %v4692 : tensor<256x64x56x56xf32>
    %v4695 = stablehlo.reduce(%v4694 init: %v4679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4696 = stablehlo.reshape %v4564 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4698 = stablehlo.reduce(%v4696 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4699 = stablehlo.reshape %v297 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4700 = stablehlo.reshape %v4553 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4701 = stablehlo.transpose %v4699, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4702 = stablehlo.transpose %v4700, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4703 = stablehlo.convolution(%v4701, %v4702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4704 = stablehlo.transpose %v4703, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4705 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4707 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4708 = stablehlo.reduce(%v4705 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4709 = stablehlo.broadcast_in_dim %v4708, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4710 = stablehlo.divide %v4709, %v4707 : tensor<256x256x56x56xf32>
    %v4711 = stablehlo.subtract %v4705, %v4710 : tensor<256x256x56x56xf32>
    %v4712 = stablehlo.multiply %v4711, %v4711 : tensor<256x256x56x56xf32>
    %v4713 = stablehlo.reduce(%v4712 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4714 = stablehlo.broadcast_in_dim %v4713, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4715 = stablehlo.divide %v4714, %v4707 : tensor<256x256x56x56xf32>
    %v4716 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4717 = stablehlo.add %v4715, %v4716 : tensor<256x256x56x56xf32>
    %v4718 = stablehlo.rsqrt %v4717 : tensor<256x256x56x56xf32>
    %v4719 = stablehlo.multiply %v4711, %v4718 : tensor<256x256x56x56xf32>
    %v4720 = stablehlo.reshape %v4523 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4721 = stablehlo.multiply %v4720, %v4719 : tensor<256x256x56x56xf32>
    %v4722 = stablehlo.reduce(%v4721 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4723 = stablehlo.reshape %v4523 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4725 = stablehlo.reduce(%v4723 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4726 = stablehlo.reshape %v4644 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4727 = stablehlo.reshape %v235 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4728 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4729 = stablehlo.compare GT, %v4727, %v4728 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4730 = stablehlo.select %v4729, %v4726, %v4728 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4731 = stablehlo.reshape %v4730 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4732 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4734 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4735 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4736 = stablehlo.reduce(%v4732 init: %v4733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.broadcast_in_dim %v4736, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4738 = stablehlo.divide %v4737, %v4734 : tensor<256x256x56x56xf32>
    %v4739 = stablehlo.subtract %v4732, %v4738 : tensor<256x256x56x56xf32>
    %v4740 = stablehlo.multiply %v4739, %v4739 : tensor<256x256x56x56xf32>
    %v4741 = stablehlo.reduce(%v4740 init: %v4733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %v4741, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4743 = stablehlo.divide %v4742, %v4734 : tensor<256x256x56x56xf32>
    %v4744 = stablehlo.add %v4743, %v4735 : tensor<256x256x56x56xf32>
    %v4745 = stablehlo.rsqrt %v4744 : tensor<256x256x56x56xf32>
    %v4746 = stablehlo.multiply %v4739, %v4745 : tensor<256x256x56x56xf32>
    %v4747 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4748 = stablehlo.reshape %v4731 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4749 = stablehlo.multiply %v4747, %v4748 : tensor<256x256x56x56xf32>
    %v4750 = stablehlo.reduce(%v4749 init: %v4733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4751 = stablehlo.broadcast_in_dim %v4750, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4752 = stablehlo.multiply %v4746, %v4749 : tensor<256x256x56x56xf32>
    %v4753 = stablehlo.reduce(%v4752 init: %v4733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4754 = stablehlo.broadcast_in_dim %v4753, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4755 = stablehlo.multiply %v4749, %v4734 : tensor<256x256x56x56xf32>
    %v4756 = stablehlo.subtract %v4755, %v4751 : tensor<256x256x56x56xf32>
    %v4757 = stablehlo.multiply %v4746, %v4754 : tensor<256x256x56x56xf32>
    %v4758 = stablehlo.subtract %v4756, %v4757 : tensor<256x256x56x56xf32>
    %v4759 = stablehlo.divide %v4745, %v4734 : tensor<256x256x56x56xf32>
    %v4760 = stablehlo.multiply %v4759, %v4758 : tensor<256x256x56x56xf32>
    %v4761 = stablehlo.reshape %v4760 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4762 = stablehlo.reshape %v4761 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4763 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4764 = stablehlo.transpose %v4763, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4765 = stablehlo.convolution(%v4762, %v4764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4766 = stablehlo.reshape %v4765 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4767 = stablehlo.reshape %v4766 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4768 = stablehlo.reshape %v202 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4769 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4770 = stablehlo.compare GT, %v4768, %v4769 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4771 = stablehlo.select %v4770, %v4767, %v4769 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4772 = stablehlo.reshape %v4771 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4773 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4775 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4776 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4777 = stablehlo.reduce(%v4773 init: %v4774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4778 = stablehlo.broadcast_in_dim %v4777, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4779 = stablehlo.divide %v4778, %v4775 : tensor<256x64x56x56xf32>
    %v4780 = stablehlo.subtract %v4773, %v4779 : tensor<256x64x56x56xf32>
    %v4781 = stablehlo.multiply %v4780, %v4780 : tensor<256x64x56x56xf32>
    %v4782 = stablehlo.reduce(%v4781 init: %v4774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4783 = stablehlo.broadcast_in_dim %v4782, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4784 = stablehlo.divide %v4783, %v4775 : tensor<256x64x56x56xf32>
    %v4785 = stablehlo.add %v4784, %v4776 : tensor<256x64x56x56xf32>
    %v4786 = stablehlo.rsqrt %v4785 : tensor<256x64x56x56xf32>
    %v4787 = stablehlo.multiply %v4780, %v4786 : tensor<256x64x56x56xf32>
    %v4788 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4789 = stablehlo.reshape %v4772 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4790 = stablehlo.multiply %v4788, %v4789 : tensor<256x64x56x56xf32>
    %v4791 = stablehlo.reduce(%v4790 init: %v4774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4792 = stablehlo.broadcast_in_dim %v4791, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4793 = stablehlo.multiply %v4787, %v4790 : tensor<256x64x56x56xf32>
    %v4794 = stablehlo.reduce(%v4793 init: %v4774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4795 = stablehlo.broadcast_in_dim %v4794, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4796 = stablehlo.multiply %v4790, %v4775 : tensor<256x64x56x56xf32>
    %v4797 = stablehlo.subtract %v4796, %v4792 : tensor<256x64x56x56xf32>
    %v4798 = stablehlo.multiply %v4787, %v4795 : tensor<256x64x56x56xf32>
    %v4799 = stablehlo.subtract %v4797, %v4798 : tensor<256x64x56x56xf32>
    %v4800 = stablehlo.divide %v4786, %v4775 : tensor<256x64x56x56xf32>
    %v4801 = stablehlo.multiply %v4800, %v4799 : tensor<256x64x56x56xf32>
    %v4802 = stablehlo.reshape %v4801 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4803 = stablehlo.reshape %v4802 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4804 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4805 = stablehlo.transpose %v4804, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4806 = stablehlo.convolution(%v4803, %v4805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4807 = stablehlo.reshape %v4806 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4808 = stablehlo.reshape %v4807 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4809 = stablehlo.reshape %v173 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4810 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4811 = stablehlo.compare GT, %v4809, %v4810 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4812 = stablehlo.select %v4811, %v4808, %v4810 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4813 = stablehlo.reshape %v4812 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4814 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4816 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4817 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4818 = stablehlo.reduce(%v4814 init: %v4815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4819 = stablehlo.broadcast_in_dim %v4818, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4820 = stablehlo.divide %v4819, %v4816 : tensor<256x64x56x56xf32>
    %v4821 = stablehlo.subtract %v4814, %v4820 : tensor<256x64x56x56xf32>
    %v4822 = stablehlo.multiply %v4821, %v4821 : tensor<256x64x56x56xf32>
    %v4823 = stablehlo.reduce(%v4822 init: %v4815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4824 = stablehlo.broadcast_in_dim %v4823, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4825 = stablehlo.divide %v4824, %v4816 : tensor<256x64x56x56xf32>
    %v4826 = stablehlo.add %v4825, %v4817 : tensor<256x64x56x56xf32>
    %v4827 = stablehlo.rsqrt %v4826 : tensor<256x64x56x56xf32>
    %v4828 = stablehlo.multiply %v4821, %v4827 : tensor<256x64x56x56xf32>
    %v4829 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4830 = stablehlo.reshape %v4813 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4831 = stablehlo.multiply %v4829, %v4830 : tensor<256x64x56x56xf32>
    %v4832 = stablehlo.reduce(%v4831 init: %v4815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4833 = stablehlo.broadcast_in_dim %v4832, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4834 = stablehlo.multiply %v4828, %v4831 : tensor<256x64x56x56xf32>
    %v4835 = stablehlo.reduce(%v4834 init: %v4815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4836 = stablehlo.broadcast_in_dim %v4835, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4837 = stablehlo.multiply %v4831, %v4816 : tensor<256x64x56x56xf32>
    %v4838 = stablehlo.subtract %v4837, %v4833 : tensor<256x64x56x56xf32>
    %v4839 = stablehlo.multiply %v4828, %v4836 : tensor<256x64x56x56xf32>
    %v4840 = stablehlo.subtract %v4838, %v4839 : tensor<256x64x56x56xf32>
    %v4841 = stablehlo.divide %v4827, %v4816 : tensor<256x64x56x56xf32>
    %v4842 = stablehlo.multiply %v4841, %v4840 : tensor<256x64x56x56xf32>
    %v4843 = stablehlo.reshape %v4842 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4844 = stablehlo.reshape %v4843 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4845 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4846 = stablehlo.transpose %v4845, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4847 = stablehlo.convolution(%v4844, %v4846)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4848 = stablehlo.reshape %v4847 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4849 = stablehlo.reshape %v4848 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4850 = stablehlo.reshape %v4731 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4851 = stablehlo.add %v4849, %v4850 : tensor<256x256x56x56xf32>
    %v4852 = stablehlo.reshape %v4851 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4853 = stablehlo.reshape %v148 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4854 = stablehlo.reshape %v4843 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4855 = stablehlo.transpose %v4853, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4856 = stablehlo.transpose %v4854, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4857 = stablehlo.convolution(%v4855, %v4856)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4858 = stablehlo.transpose %v4857, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4859 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4860 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4861 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4862 = stablehlo.reduce(%v4859 init: %v4860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4863 = stablehlo.broadcast_in_dim %v4862, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4864 = stablehlo.divide %v4863, %v4861 : tensor<256x64x56x56xf32>
    %v4865 = stablehlo.subtract %v4859, %v4864 : tensor<256x64x56x56xf32>
    %v4866 = stablehlo.multiply %v4865, %v4865 : tensor<256x64x56x56xf32>
    %v4867 = stablehlo.reduce(%v4866 init: %v4860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4868 = stablehlo.broadcast_in_dim %v4867, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4869 = stablehlo.divide %v4868, %v4861 : tensor<256x64x56x56xf32>
    %v4870 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4871 = stablehlo.add %v4869, %v4870 : tensor<256x64x56x56xf32>
    %v4872 = stablehlo.rsqrt %v4871 : tensor<256x64x56x56xf32>
    %v4873 = stablehlo.multiply %v4865, %v4872 : tensor<256x64x56x56xf32>
    %v4874 = stablehlo.reshape %v4813 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4875 = stablehlo.multiply %v4874, %v4873 : tensor<256x64x56x56xf32>
    %v4876 = stablehlo.reduce(%v4875 init: %v4860) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4877 = stablehlo.reshape %v4813 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4879 = stablehlo.reduce(%v4877 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4880 = stablehlo.reshape %v177 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4881 = stablehlo.reshape %v4802 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4882 = stablehlo.transpose %v4880, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4883 = stablehlo.transpose %v4881, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4884 = stablehlo.convolution(%v4882, %v4883)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4885 = stablehlo.transpose %v4884, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4886 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4888 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4889 = stablehlo.reduce(%v4886 init: %v4887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4890 = stablehlo.broadcast_in_dim %v4889, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4891 = stablehlo.divide %v4890, %v4888 : tensor<256x64x56x56xf32>
    %v4892 = stablehlo.subtract %v4886, %v4891 : tensor<256x64x56x56xf32>
    %v4893 = stablehlo.multiply %v4892, %v4892 : tensor<256x64x56x56xf32>
    %v4894 = stablehlo.reduce(%v4893 init: %v4887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4895 = stablehlo.broadcast_in_dim %v4894, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4896 = stablehlo.divide %v4895, %v4888 : tensor<256x64x56x56xf32>
    %v4897 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4898 = stablehlo.add %v4896, %v4897 : tensor<256x64x56x56xf32>
    %v4899 = stablehlo.rsqrt %v4898 : tensor<256x64x56x56xf32>
    %v4900 = stablehlo.multiply %v4892, %v4899 : tensor<256x64x56x56xf32>
    %v4901 = stablehlo.reshape %v4772 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4902 = stablehlo.multiply %v4901, %v4900 : tensor<256x64x56x56xf32>
    %v4903 = stablehlo.reduce(%v4902 init: %v4887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4904 = stablehlo.reshape %v4772 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4906 = stablehlo.reduce(%v4904 init: %v4905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4907 = stablehlo.reshape %v206 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4908 = stablehlo.reshape %v4761 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4909 = stablehlo.transpose %v4907, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4910 = stablehlo.transpose %v4908, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4911 = stablehlo.convolution(%v4909, %v4910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4912 = stablehlo.transpose %v4911, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4913 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4915 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4916 = stablehlo.reduce(%v4913 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4917 = stablehlo.broadcast_in_dim %v4916, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4918 = stablehlo.divide %v4917, %v4915 : tensor<256x256x56x56xf32>
    %v4919 = stablehlo.subtract %v4913, %v4918 : tensor<256x256x56x56xf32>
    %v4920 = stablehlo.multiply %v4919, %v4919 : tensor<256x256x56x56xf32>
    %v4921 = stablehlo.reduce(%v4920 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4922 = stablehlo.broadcast_in_dim %v4921, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4923 = stablehlo.divide %v4922, %v4915 : tensor<256x256x56x56xf32>
    %v4924 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4925 = stablehlo.add %v4923, %v4924 : tensor<256x256x56x56xf32>
    %v4926 = stablehlo.rsqrt %v4925 : tensor<256x256x56x56xf32>
    %v4927 = stablehlo.multiply %v4919, %v4926 : tensor<256x256x56x56xf32>
    %v4928 = stablehlo.reshape %v4731 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4929 = stablehlo.multiply %v4928, %v4927 : tensor<256x256x56x56xf32>
    %v4930 = stablehlo.reduce(%v4929 init: %v4914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4931 = stablehlo.reshape %v4731 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4933 = stablehlo.reduce(%v4931 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4934 = stablehlo.reshape %v4852 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4935 = stablehlo.reshape %v144 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4936 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4937 = stablehlo.compare GT, %v4935, %v4936 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4938 = stablehlo.select %v4937, %v4934, %v4936 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4939 = stablehlo.reshape %v4938 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4940 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4941 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4942 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4943 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4944 = stablehlo.reduce(%v4940 init: %v4941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4945 = stablehlo.broadcast_in_dim %v4944, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4946 = stablehlo.divide %v4945, %v4942 : tensor<256x256x56x56xf32>
    %v4947 = stablehlo.subtract %v4940, %v4946 : tensor<256x256x56x56xf32>
    %v4948 = stablehlo.multiply %v4947, %v4947 : tensor<256x256x56x56xf32>
    %v4949 = stablehlo.reduce(%v4948 init: %v4941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4950 = stablehlo.broadcast_in_dim %v4949, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4951 = stablehlo.divide %v4950, %v4942 : tensor<256x256x56x56xf32>
    %v4952 = stablehlo.add %v4951, %v4943 : tensor<256x256x56x56xf32>
    %v4953 = stablehlo.rsqrt %v4952 : tensor<256x256x56x56xf32>
    %v4954 = stablehlo.multiply %v4947, %v4953 : tensor<256x256x56x56xf32>
    %v4955 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4956 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4957 = stablehlo.multiply %v4955, %v4956 : tensor<256x256x56x56xf32>
    %v4958 = stablehlo.reduce(%v4957 init: %v4941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4959 = stablehlo.broadcast_in_dim %v4958, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4960 = stablehlo.multiply %v4954, %v4957 : tensor<256x256x56x56xf32>
    %v4961 = stablehlo.reduce(%v4960 init: %v4941) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4962 = stablehlo.broadcast_in_dim %v4961, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4963 = stablehlo.multiply %v4957, %v4942 : tensor<256x256x56x56xf32>
    %v4964 = stablehlo.subtract %v4963, %v4959 : tensor<256x256x56x56xf32>
    %v4965 = stablehlo.multiply %v4954, %v4962 : tensor<256x256x56x56xf32>
    %v4966 = stablehlo.subtract %v4964, %v4965 : tensor<256x256x56x56xf32>
    %v4967 = stablehlo.divide %v4953, %v4942 : tensor<256x256x56x56xf32>
    %v4968 = stablehlo.multiply %v4967, %v4966 : tensor<256x256x56x56xf32>
    %v4969 = stablehlo.reshape %v4968 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4971 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4972 = stablehlo.transpose %v4971, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4973 = stablehlo.convolution(%v4970, %v4972)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4974 = stablehlo.reshape %v4973 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4975 = stablehlo.reshape %v4974 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4976 = stablehlo.reshape %v86 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4977 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4978 = stablehlo.compare GT, %v4976, %v4977 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4979 = stablehlo.select %v4978, %v4975, %v4977 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4980 = stablehlo.reshape %v4979 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4981 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4983 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4984 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4985 = stablehlo.reduce(%v4981 init: %v4982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4986 = stablehlo.broadcast_in_dim %v4985, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4987 = stablehlo.divide %v4986, %v4983 : tensor<256x64x56x56xf32>
    %v4988 = stablehlo.subtract %v4981, %v4987 : tensor<256x64x56x56xf32>
    %v4989 = stablehlo.multiply %v4988, %v4988 : tensor<256x64x56x56xf32>
    %v4990 = stablehlo.reduce(%v4989 init: %v4982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4991 = stablehlo.broadcast_in_dim %v4990, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4992 = stablehlo.divide %v4991, %v4983 : tensor<256x64x56x56xf32>
    %v4993 = stablehlo.add %v4992, %v4984 : tensor<256x64x56x56xf32>
    %v4994 = stablehlo.rsqrt %v4993 : tensor<256x64x56x56xf32>
    %v4995 = stablehlo.multiply %v4988, %v4994 : tensor<256x64x56x56xf32>
    %v4996 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4997 = stablehlo.reshape %v4980 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4998 = stablehlo.multiply %v4996, %v4997 : tensor<256x64x56x56xf32>
    %v4999 = stablehlo.reduce(%v4998 init: %v4982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5000 = stablehlo.broadcast_in_dim %v4999, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5001 = stablehlo.multiply %v4995, %v4998 : tensor<256x64x56x56xf32>
    %v5002 = stablehlo.reduce(%v5001 init: %v4982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5003 = stablehlo.broadcast_in_dim %v5002, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5004 = stablehlo.multiply %v4998, %v4983 : tensor<256x64x56x56xf32>
    %v5005 = stablehlo.subtract %v5004, %v5000 : tensor<256x64x56x56xf32>
    %v5006 = stablehlo.multiply %v4995, %v5003 : tensor<256x64x56x56xf32>
    %v5007 = stablehlo.subtract %v5005, %v5006 : tensor<256x64x56x56xf32>
    %v5008 = stablehlo.divide %v4994, %v4983 : tensor<256x64x56x56xf32>
    %v5009 = stablehlo.multiply %v5008, %v5007 : tensor<256x64x56x56xf32>
    %v5010 = stablehlo.reshape %v5009 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5011 = stablehlo.reshape %v5010 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5012 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5013 = stablehlo.transpose %v5012, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5014 = stablehlo.convolution(%v5011, %v5013)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v5015 = stablehlo.reshape %v5014 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5016 = stablehlo.reshape %v5015 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5017 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5018 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v5019 = stablehlo.compare GT, %v5017, %v5018 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5020 = stablehlo.select %v5019, %v5016, %v5018 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5021 = stablehlo.reshape %v5020 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5022 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5024 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5025 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5026 = stablehlo.reduce(%v5022 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5027 = stablehlo.broadcast_in_dim %v5026, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5028 = stablehlo.divide %v5027, %v5024 : tensor<256x64x56x56xf32>
    %v5029 = stablehlo.subtract %v5022, %v5028 : tensor<256x64x56x56xf32>
    %v5030 = stablehlo.multiply %v5029, %v5029 : tensor<256x64x56x56xf32>
    %v5031 = stablehlo.reduce(%v5030 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5032 = stablehlo.broadcast_in_dim %v5031, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5033 = stablehlo.divide %v5032, %v5024 : tensor<256x64x56x56xf32>
    %v5034 = stablehlo.add %v5033, %v5025 : tensor<256x64x56x56xf32>
    %v5035 = stablehlo.rsqrt %v5034 : tensor<256x64x56x56xf32>
    %v5036 = stablehlo.multiply %v5029, %v5035 : tensor<256x64x56x56xf32>
    %v5037 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5038 = stablehlo.reshape %v5021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5039 = stablehlo.multiply %v5037, %v5038 : tensor<256x64x56x56xf32>
    %v5040 = stablehlo.reduce(%v5039 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5041 = stablehlo.broadcast_in_dim %v5040, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5042 = stablehlo.multiply %v5036, %v5039 : tensor<256x64x56x56xf32>
    %v5043 = stablehlo.reduce(%v5042 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5044 = stablehlo.broadcast_in_dim %v5043, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5045 = stablehlo.multiply %v5039, %v5024 : tensor<256x64x56x56xf32>
    %v5046 = stablehlo.subtract %v5045, %v5041 : tensor<256x64x56x56xf32>
    %v5047 = stablehlo.multiply %v5036, %v5044 : tensor<256x64x56x56xf32>
    %v5048 = stablehlo.subtract %v5046, %v5047 : tensor<256x64x56x56xf32>
    %v5049 = stablehlo.divide %v5035, %v5024 : tensor<256x64x56x56xf32>
    %v5050 = stablehlo.multiply %v5049, %v5048 : tensor<256x64x56x56xf32>
    %v5051 = stablehlo.reshape %v5050 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5052 = stablehlo.reshape %v5051 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5053 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v5054 = stablehlo.transpose %v5053, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5055 = stablehlo.convolution(%v5052, %v5054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v5056 = stablehlo.reshape %v5055 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5057 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5059 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5060 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5061 = stablehlo.reduce(%v5057 init: %v5058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5062 = stablehlo.broadcast_in_dim %v5061, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5063 = stablehlo.divide %v5062, %v5059 : tensor<256x256x56x56xf32>
    %v5064 = stablehlo.subtract %v5057, %v5063 : tensor<256x256x56x56xf32>
    %v5065 = stablehlo.multiply %v5064, %v5064 : tensor<256x256x56x56xf32>
    %v5066 = stablehlo.reduce(%v5065 init: %v5058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.broadcast_in_dim %v5066, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5068 = stablehlo.divide %v5067, %v5059 : tensor<256x256x56x56xf32>
    %v5069 = stablehlo.add %v5068, %v5060 : tensor<256x256x56x56xf32>
    %v5070 = stablehlo.rsqrt %v5069 : tensor<256x256x56x56xf32>
    %v5071 = stablehlo.multiply %v5064, %v5070 : tensor<256x256x56x56xf32>
    %v5072 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5073 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5074 = stablehlo.multiply %v5072, %v5073 : tensor<256x256x56x56xf32>
    %v5075 = stablehlo.reduce(%v5074 init: %v5058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5076 = stablehlo.broadcast_in_dim %v5075, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5077 = stablehlo.multiply %v5071, %v5074 : tensor<256x256x56x56xf32>
    %v5078 = stablehlo.reduce(%v5077 init: %v5058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5079 = stablehlo.broadcast_in_dim %v5078, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5080 = stablehlo.multiply %v5074, %v5059 : tensor<256x256x56x56xf32>
    %v5081 = stablehlo.subtract %v5080, %v5076 : tensor<256x256x56x56xf32>
    %v5082 = stablehlo.multiply %v5071, %v5079 : tensor<256x256x56x56xf32>
    %v5083 = stablehlo.subtract %v5081, %v5082 : tensor<256x256x56x56xf32>
    %v5084 = stablehlo.divide %v5070, %v5059 : tensor<256x256x56x56xf32>
    %v5085 = stablehlo.multiply %v5084, %v5083 : tensor<256x256x56x56xf32>
    %v5086 = stablehlo.reshape %v5085 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v5087 = stablehlo.reshape %v5086 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5088 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5089 = stablehlo.transpose %v5088, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5090 = stablehlo.convolution(%v5087, %v5089)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v5091 = stablehlo.reshape %v5090 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5092 = stablehlo.reshape %v5056 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5093 = stablehlo.reshape %v5091 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5094 = stablehlo.add %v5092, %v5093 : tensor<256x64x56x56xf32>
    %v5095 = stablehlo.reshape %v5094 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5096 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5097 = stablehlo.reshape %v5051 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5098 = stablehlo.transpose %v5096, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5099 = stablehlo.transpose %v5097, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5100 = stablehlo.convolution(%v5098, %v5099)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x1x1xf32>
    %v5101 = stablehlo.transpose %v5100, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5102 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5104 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5105 = stablehlo.reduce(%v5102 init: %v5103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5106 = stablehlo.broadcast_in_dim %v5105, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5107 = stablehlo.divide %v5106, %v5104 : tensor<256x64x56x56xf32>
    %v5108 = stablehlo.subtract %v5102, %v5107 : tensor<256x64x56x56xf32>
    %v5109 = stablehlo.multiply %v5108, %v5108 : tensor<256x64x56x56xf32>
    %v5110 = stablehlo.reduce(%v5109 init: %v5103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5111 = stablehlo.broadcast_in_dim %v5110, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5112 = stablehlo.divide %v5111, %v5104 : tensor<256x64x56x56xf32>
    %v5113 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5114 = stablehlo.add %v5112, %v5113 : tensor<256x64x56x56xf32>
    %v5115 = stablehlo.rsqrt %v5114 : tensor<256x64x56x56xf32>
    %v5116 = stablehlo.multiply %v5108, %v5115 : tensor<256x64x56x56xf32>
    %v5117 = stablehlo.reshape %v5021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5118 = stablehlo.multiply %v5117, %v5116 : tensor<256x64x56x56xf32>
    %v5119 = stablehlo.reduce(%v5118 init: %v5103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5120 = stablehlo.reshape %v5021 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5122 = stablehlo.reduce(%v5120 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5123 = stablehlo.reshape %v61 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5124 = stablehlo.reshape %v5010 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5125 = stablehlo.transpose %v5123, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5126 = stablehlo.transpose %v5124, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5127 = stablehlo.convolution(%v5125, %v5126)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v5128 = stablehlo.transpose %v5127, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5129 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5131 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5132 = stablehlo.reduce(%v5129 init: %v5130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5133 = stablehlo.broadcast_in_dim %v5132, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5134 = stablehlo.divide %v5133, %v5131 : tensor<256x64x56x56xf32>
    %v5135 = stablehlo.subtract %v5129, %v5134 : tensor<256x64x56x56xf32>
    %v5136 = stablehlo.multiply %v5135, %v5135 : tensor<256x64x56x56xf32>
    %v5137 = stablehlo.reduce(%v5136 init: %v5130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5138 = stablehlo.broadcast_in_dim %v5137, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5139 = stablehlo.divide %v5138, %v5131 : tensor<256x64x56x56xf32>
    %v5140 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5141 = stablehlo.add %v5139, %v5140 : tensor<256x64x56x56xf32>
    %v5142 = stablehlo.rsqrt %v5141 : tensor<256x64x56x56xf32>
    %v5143 = stablehlo.multiply %v5135, %v5142 : tensor<256x64x56x56xf32>
    %v5144 = stablehlo.reshape %v4980 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5145 = stablehlo.multiply %v5144, %v5143 : tensor<256x64x56x56xf32>
    %v5146 = stablehlo.reduce(%v5145 init: %v5130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5147 = stablehlo.reshape %v4980 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5149 = stablehlo.reduce(%v5147 init: %v5148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5150 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5151 = stablehlo.reshape %v4969 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5152 = stablehlo.transpose %v5150, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5153 = stablehlo.transpose %v5151, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v5154 = stablehlo.convolution(%v5152, %v5153)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v5155 = stablehlo.transpose %v5154, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5156 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5158 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5159 = stablehlo.reduce(%v5156 init: %v5157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5160 = stablehlo.broadcast_in_dim %v5159, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5161 = stablehlo.divide %v5160, %v5158 : tensor<256x256x56x56xf32>
    %v5162 = stablehlo.subtract %v5156, %v5161 : tensor<256x256x56x56xf32>
    %v5163 = stablehlo.multiply %v5162, %v5162 : tensor<256x256x56x56xf32>
    %v5164 = stablehlo.reduce(%v5163 init: %v5157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5165 = stablehlo.broadcast_in_dim %v5164, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5166 = stablehlo.divide %v5165, %v5158 : tensor<256x256x56x56xf32>
    %v5167 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5168 = stablehlo.add %v5166, %v5167 : tensor<256x256x56x56xf32>
    %v5169 = stablehlo.rsqrt %v5168 : tensor<256x256x56x56xf32>
    %v5170 = stablehlo.multiply %v5162, %v5169 : tensor<256x256x56x56xf32>
    %v5171 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5172 = stablehlo.multiply %v5171, %v5170 : tensor<256x256x56x56xf32>
    %v5173 = stablehlo.reduce(%v5172 init: %v5157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5174 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5176 = stablehlo.reduce(%v5174 init: %v5175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5177 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5178 = stablehlo.reshape %v5086 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5179 = stablehlo.transpose %v5177, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5180 = stablehlo.transpose %v5178, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v5181 = stablehlo.convolution(%v5179, %v5180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v5182 = stablehlo.transpose %v5181, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5183 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5185 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5186 = stablehlo.reduce(%v5183 init: %v5184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5187 = stablehlo.broadcast_in_dim %v5186, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5188 = stablehlo.divide %v5187, %v5185 : tensor<256x256x56x56xf32>
    %v5189 = stablehlo.subtract %v5183, %v5188 : tensor<256x256x56x56xf32>
    %v5190 = stablehlo.multiply %v5189, %v5189 : tensor<256x256x56x56xf32>
    %v5191 = stablehlo.reduce(%v5190 init: %v5184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5192 = stablehlo.broadcast_in_dim %v5191, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5193 = stablehlo.divide %v5192, %v5185 : tensor<256x256x56x56xf32>
    %v5194 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5195 = stablehlo.add %v5193, %v5194 : tensor<256x256x56x56xf32>
    %v5196 = stablehlo.rsqrt %v5195 : tensor<256x256x56x56xf32>
    %v5197 = stablehlo.multiply %v5189, %v5196 : tensor<256x256x56x56xf32>
    %v5198 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5199 = stablehlo.multiply %v5198, %v5197 : tensor<256x256x56x56xf32>
    %v5200 = stablehlo.reduce(%v5199 init: %v5184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5201 = stablehlo.reshape %v4939 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5203 = stablehlo.reduce(%v5201 init: %v5202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5204 = stablehlo.reshape %v28 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5205 = stablehlo.reshape %v5095 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5206 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v5207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5208 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v5209 = stablehlo.pad %v5204, %v5206, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5210 = stablehlo.slice %v5209 [0:256, 0:64, 0:112:2, 0:112:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5211 = stablehlo.slice %v5209 [0:256, 0:64, 0:112:2, 1:113:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5212 = stablehlo.slice %v5209 [0:256, 0:64, 0:112:2, 2:114:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5213 = stablehlo.slice %v5209 [0:256, 0:64, 1:113:2, 0:112:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5214 = stablehlo.slice %v5209 [0:256, 0:64, 1:113:2, 1:113:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5215 = stablehlo.slice %v5209 [0:256, 0:64, 1:113:2, 2:114:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5216 = stablehlo.slice %v5209 [0:256, 0:64, 2:114:2, 0:112:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5217 = stablehlo.slice %v5209 [0:256, 0:64, 2:114:2, 1:113:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5218 = stablehlo.slice %v5209 [0:256, 0:64, 2:114:2, 2:114:2] : (tensor<256x64x114x114xf32>) -> tensor<256x64x56x56xf32>
    %v5219 = stablehlo.maximum %v5210, %v5211 : tensor<256x64x56x56xf32>
    %v5220 = stablehlo.maximum %v5219, %v5212 : tensor<256x64x56x56xf32>
    %v5221 = stablehlo.maximum %v5220, %v5213 : tensor<256x64x56x56xf32>
    %v5222 = stablehlo.maximum %v5221, %v5214 : tensor<256x64x56x56xf32>
    %v5223 = stablehlo.maximum %v5222, %v5215 : tensor<256x64x56x56xf32>
    %v5224 = stablehlo.maximum %v5223, %v5216 : tensor<256x64x56x56xf32>
    %v5225 = stablehlo.maximum %v5224, %v5217 : tensor<256x64x56x56xf32>
    %v5226 = stablehlo.maximum %v5225, %v5218 : tensor<256x64x56x56xf32>
    %v5227 = stablehlo.compare EQ, %v5210, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5228 = stablehlo.select %v5227, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5229 = stablehlo.pad %v5228, %v5207, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5230 = stablehlo.compare EQ, %v5211, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5231 = stablehlo.not %v5227 : tensor<256x64x56x56xi1>
    %v5232 = stablehlo.and %v5230, %v5231 : tensor<256x64x56x56xi1>
    %v5233 = stablehlo.or %v5227, %v5230 : tensor<256x64x56x56xi1>
    %v5234 = stablehlo.select %v5232, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5235 = stablehlo.pad %v5234, %v5207, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5236 = stablehlo.add %v5229, %v5235 : tensor<256x64x114x114xf32>
    %v5237 = stablehlo.compare EQ, %v5212, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5238 = stablehlo.not %v5233 : tensor<256x64x56x56xi1>
    %v5239 = stablehlo.and %v5237, %v5238 : tensor<256x64x56x56xi1>
    %v5240 = stablehlo.or %v5233, %v5237 : tensor<256x64x56x56xi1>
    %v5241 = stablehlo.select %v5239, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5242 = stablehlo.pad %v5241, %v5207, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5243 = stablehlo.add %v5236, %v5242 : tensor<256x64x114x114xf32>
    %v5244 = stablehlo.compare EQ, %v5213, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5245 = stablehlo.not %v5240 : tensor<256x64x56x56xi1>
    %v5246 = stablehlo.and %v5244, %v5245 : tensor<256x64x56x56xi1>
    %v5247 = stablehlo.or %v5240, %v5244 : tensor<256x64x56x56xi1>
    %v5248 = stablehlo.select %v5246, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5249 = stablehlo.pad %v5248, %v5207, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5250 = stablehlo.add %v5243, %v5249 : tensor<256x64x114x114xf32>
    %v5251 = stablehlo.compare EQ, %v5214, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5252 = stablehlo.not %v5247 : tensor<256x64x56x56xi1>
    %v5253 = stablehlo.and %v5251, %v5252 : tensor<256x64x56x56xi1>
    %v5254 = stablehlo.or %v5247, %v5251 : tensor<256x64x56x56xi1>
    %v5255 = stablehlo.select %v5253, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5256 = stablehlo.pad %v5255, %v5207, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5257 = stablehlo.add %v5250, %v5256 : tensor<256x64x114x114xf32>
    %v5258 = stablehlo.compare EQ, %v5215, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5259 = stablehlo.not %v5254 : tensor<256x64x56x56xi1>
    %v5260 = stablehlo.and %v5258, %v5259 : tensor<256x64x56x56xi1>
    %v5261 = stablehlo.or %v5254, %v5258 : tensor<256x64x56x56xi1>
    %v5262 = stablehlo.select %v5260, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5263 = stablehlo.pad %v5262, %v5207, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5264 = stablehlo.add %v5257, %v5263 : tensor<256x64x114x114xf32>
    %v5265 = stablehlo.compare EQ, %v5216, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5266 = stablehlo.not %v5261 : tensor<256x64x56x56xi1>
    %v5267 = stablehlo.and %v5265, %v5266 : tensor<256x64x56x56xi1>
    %v5268 = stablehlo.or %v5261, %v5265 : tensor<256x64x56x56xi1>
    %v5269 = stablehlo.select %v5267, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5270 = stablehlo.pad %v5269, %v5207, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5271 = stablehlo.add %v5264, %v5270 : tensor<256x64x114x114xf32>
    %v5272 = stablehlo.compare EQ, %v5217, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5273 = stablehlo.not %v5268 : tensor<256x64x56x56xi1>
    %v5274 = stablehlo.and %v5272, %v5273 : tensor<256x64x56x56xi1>
    %v5275 = stablehlo.or %v5268, %v5272 : tensor<256x64x56x56xi1>
    %v5276 = stablehlo.select %v5274, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5277 = stablehlo.pad %v5276, %v5207, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5278 = stablehlo.add %v5271, %v5277 : tensor<256x64x114x114xf32>
    %v5279 = stablehlo.compare EQ, %v5218, %v5226 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5280 = stablehlo.not %v5275 : tensor<256x64x56x56xi1>
    %v5281 = stablehlo.and %v5279, %v5280 : tensor<256x64x56x56xi1>
    %v5282 = stablehlo.or %v5275, %v5279 : tensor<256x64x56x56xi1>
    %v5283 = stablehlo.select %v5281, %v5205, %v5208 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5284 = stablehlo.pad %v5283, %v5207, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x114x114xf32>
    %v5285 = stablehlo.add %v5278, %v5284 : tensor<256x64x114x114xf32>
    %v5286 = stablehlo.slice %v5285 [0:256, 0:64, 1:113, 1:113] : (tensor<256x64x114x114xf32>) -> tensor<256x64x112x112xf32>
    %v5287 = stablehlo.reshape %v5286 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5288 = stablehlo.reshape %v5287 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5289 = stablehlo.reshape %v24 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5290 = stablehlo.constant dense<0.0> : tensor<256x64x112x112xf32>
    %v5291 = stablehlo.compare GT, %v5289, %v5290 : (tensor<256x64x112x112xf32>, tensor<256x64x112x112xf32>) -> tensor<256x64x112x112xi1>
    %v5292 = stablehlo.select %v5291, %v5288, %v5290 : tensor<256x64x112x112xi1>, tensor<256x64x112x112xf32>
    %v5293 = stablehlo.reshape %v5292 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5294 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5296 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5297 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v5298 = stablehlo.reduce(%v5294 init: %v5295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5299 = stablehlo.broadcast_in_dim %v5298, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5300 = stablehlo.divide %v5299, %v5296 : tensor<256x64x112x112xf32>
    %v5301 = stablehlo.subtract %v5294, %v5300 : tensor<256x64x112x112xf32>
    %v5302 = stablehlo.multiply %v5301, %v5301 : tensor<256x64x112x112xf32>
    %v5303 = stablehlo.reduce(%v5302 init: %v5295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5304 = stablehlo.broadcast_in_dim %v5303, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5305 = stablehlo.divide %v5304, %v5296 : tensor<256x64x112x112xf32>
    %v5306 = stablehlo.add %v5305, %v5297 : tensor<256x64x112x112xf32>
    %v5307 = stablehlo.rsqrt %v5306 : tensor<256x64x112x112xf32>
    %v5308 = stablehlo.multiply %v5301, %v5307 : tensor<256x64x112x112xf32>
    %v5309 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5310 = stablehlo.reshape %v5293 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5311 = stablehlo.multiply %v5309, %v5310 : tensor<256x64x112x112xf32>
    %v5312 = stablehlo.reduce(%v5311 init: %v5295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5313 = stablehlo.broadcast_in_dim %v5312, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5314 = stablehlo.multiply %v5308, %v5311 : tensor<256x64x112x112xf32>
    %v5315 = stablehlo.reduce(%v5314 init: %v5295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5316 = stablehlo.broadcast_in_dim %v5315, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5317 = stablehlo.multiply %v5311, %v5296 : tensor<256x64x112x112xf32>
    %v5318 = stablehlo.subtract %v5317, %v5313 : tensor<256x64x112x112xf32>
    %v5319 = stablehlo.multiply %v5308, %v5316 : tensor<256x64x112x112xf32>
    %v5320 = stablehlo.subtract %v5318, %v5319 : tensor<256x64x112x112xf32>
    %v5321 = stablehlo.divide %v5307, %v5296 : tensor<256x64x112x112xf32>
    %v5322 = stablehlo.multiply %v5321, %v5320 : tensor<256x64x112x112xf32>
    %v5323 = stablehlo.reshape %v5322 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5324 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v5325 = stablehlo.reshape %v5323 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5326 = stablehlo.transpose %v5324, dims = [1, 0, 2, 3] : (tensor<256x3x224x224xf32>) -> tensor<3x256x224x224xf32>
    %v5328 = stablehlo.transpose %v5325, dims = [1, 0, 2, 3] : (tensor<256x64x112x112xf32>) -> tensor<64x256x112x112xf32>
    %v5329 = stablehlo.convolution(%v5326, %v5328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x224x224xf32>, tensor<64x256x112x112xf32>) -> tensor<3x64x7x7xf32>
    %v5330 = stablehlo.transpose %v5329, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v5331 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5333 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5334 = stablehlo.reduce(%v5331 init: %v5332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5335 = stablehlo.broadcast_in_dim %v5334, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5336 = stablehlo.divide %v5335, %v5333 : tensor<256x64x112x112xf32>
    %v5337 = stablehlo.subtract %v5331, %v5336 : tensor<256x64x112x112xf32>
    %v5338 = stablehlo.multiply %v5337, %v5337 : tensor<256x64x112x112xf32>
    %v5339 = stablehlo.reduce(%v5338 init: %v5332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5340 = stablehlo.broadcast_in_dim %v5339, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5341 = stablehlo.divide %v5340, %v5333 : tensor<256x64x112x112xf32>
    %v5342 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v5343 = stablehlo.add %v5341, %v5342 : tensor<256x64x112x112xf32>
    %v5344 = stablehlo.rsqrt %v5343 : tensor<256x64x112x112xf32>
    %v5345 = stablehlo.multiply %v5337, %v5344 : tensor<256x64x112x112xf32>
    %v5346 = stablehlo.reshape %v5293 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5347 = stablehlo.multiply %v5346, %v5345 : tensor<256x64x112x112xf32>
    %v5348 = stablehlo.reduce(%v5347 init: %v5332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5349 = stablehlo.reshape %v5293 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5350 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5351 = stablehlo.reduce(%v5349 init: %v5350) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5352 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5354 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v5355 = stablehlo.reduce(%v5352 init: %v5353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5356 = stablehlo.divide %v5355, %v5354 : tensor<64xf32>
    %v5357 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5359 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5360 = stablehlo.reduce(%v5357 init: %v5358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5361 = stablehlo.broadcast_in_dim %v5360, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5362 = stablehlo.divide %v5361, %v5359 : tensor<256x64x112x112xf32>
    %v5363 = stablehlo.subtract %v5357, %v5362 : tensor<256x64x112x112xf32>
    %v5364 = stablehlo.multiply %v5363, %v5363 : tensor<256x64x112x112xf32>
    %v5365 = stablehlo.reduce(%v5364 init: %v5358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5366 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v5367 = stablehlo.divide %v5365, %v5366 : tensor<64xf32>
    %v5368 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5370 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5371 = stablehlo.reduce(%v5368 init: %v5369) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5372 = stablehlo.divide %v5371, %v5370 : tensor<64xf32>
    %v5373 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5375 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5376 = stablehlo.reduce(%v5373 init: %v5374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5377 = stablehlo.broadcast_in_dim %v5376, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5378 = stablehlo.divide %v5377, %v5375 : tensor<256x64x56x56xf32>
    %v5379 = stablehlo.subtract %v5373, %v5378 : tensor<256x64x56x56xf32>
    %v5380 = stablehlo.multiply %v5379, %v5379 : tensor<256x64x56x56xf32>
    %v5381 = stablehlo.reduce(%v5380 init: %v5374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5382 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5383 = stablehlo.divide %v5381, %v5382 : tensor<64xf32>
    %v5384 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5386 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5387 = stablehlo.reduce(%v5384 init: %v5385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5388 = stablehlo.divide %v5387, %v5386 : tensor<64xf32>
    %v5389 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5391 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5392 = stablehlo.reduce(%v5389 init: %v5390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5393 = stablehlo.broadcast_in_dim %v5392, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5394 = stablehlo.divide %v5393, %v5391 : tensor<256x64x56x56xf32>
    %v5395 = stablehlo.subtract %v5389, %v5394 : tensor<256x64x56x56xf32>
    %v5396 = stablehlo.multiply %v5395, %v5395 : tensor<256x64x56x56xf32>
    %v5397 = stablehlo.reduce(%v5396 init: %v5390) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5398 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5399 = stablehlo.divide %v5397, %v5398 : tensor<64xf32>
    %v5400 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5402 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5403 = stablehlo.reduce(%v5400 init: %v5401) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5404 = stablehlo.divide %v5403, %v5402 : tensor<256xf32>
    %v5405 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5407 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5408 = stablehlo.reduce(%v5405 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5409 = stablehlo.broadcast_in_dim %v5408, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5410 = stablehlo.divide %v5409, %v5407 : tensor<256x256x56x56xf32>
    %v5411 = stablehlo.subtract %v5405, %v5410 : tensor<256x256x56x56xf32>
    %v5412 = stablehlo.multiply %v5411, %v5411 : tensor<256x256x56x56xf32>
    %v5413 = stablehlo.reduce(%v5412 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5414 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5415 = stablehlo.divide %v5413, %v5414 : tensor<256xf32>
    %v5416 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5418 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5419 = stablehlo.reduce(%v5416 init: %v5417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5420 = stablehlo.divide %v5419, %v5418 : tensor<256xf32>
    %v5421 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5423 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5424 = stablehlo.reduce(%v5421 init: %v5422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5425 = stablehlo.broadcast_in_dim %v5424, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5426 = stablehlo.divide %v5425, %v5423 : tensor<256x256x56x56xf32>
    %v5427 = stablehlo.subtract %v5421, %v5426 : tensor<256x256x56x56xf32>
    %v5428 = stablehlo.multiply %v5427, %v5427 : tensor<256x256x56x56xf32>
    %v5429 = stablehlo.reduce(%v5428 init: %v5422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5430 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5431 = stablehlo.divide %v5429, %v5430 : tensor<256xf32>
    %v5432 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5434 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5435 = stablehlo.reduce(%v5432 init: %v5433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5436 = stablehlo.divide %v5435, %v5434 : tensor<64xf32>
    %v5437 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5438 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5439 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5440 = stablehlo.reduce(%v5437 init: %v5438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5441 = stablehlo.broadcast_in_dim %v5440, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5442 = stablehlo.divide %v5441, %v5439 : tensor<256x64x56x56xf32>
    %v5443 = stablehlo.subtract %v5437, %v5442 : tensor<256x64x56x56xf32>
    %v5444 = stablehlo.multiply %v5443, %v5443 : tensor<256x64x56x56xf32>
    %v5445 = stablehlo.reduce(%v5444 init: %v5438) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5446 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5447 = stablehlo.divide %v5445, %v5446 : tensor<64xf32>
    %v5448 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5450 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5451 = stablehlo.reduce(%v5448 init: %v5449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5452 = stablehlo.divide %v5451, %v5450 : tensor<64xf32>
    %v5453 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5455 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5456 = stablehlo.reduce(%v5453 init: %v5454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5457 = stablehlo.broadcast_in_dim %v5456, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5458 = stablehlo.divide %v5457, %v5455 : tensor<256x64x56x56xf32>
    %v5459 = stablehlo.subtract %v5453, %v5458 : tensor<256x64x56x56xf32>
    %v5460 = stablehlo.multiply %v5459, %v5459 : tensor<256x64x56x56xf32>
    %v5461 = stablehlo.reduce(%v5460 init: %v5454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5462 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5463 = stablehlo.divide %v5461, %v5462 : tensor<64xf32>
    %v5464 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5466 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5467 = stablehlo.reduce(%v5464 init: %v5465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5468 = stablehlo.divide %v5467, %v5466 : tensor<256xf32>
    %v5469 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5471 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5472 = stablehlo.reduce(%v5469 init: %v5470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5473 = stablehlo.broadcast_in_dim %v5472, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5474 = stablehlo.divide %v5473, %v5471 : tensor<256x256x56x56xf32>
    %v5475 = stablehlo.subtract %v5469, %v5474 : tensor<256x256x56x56xf32>
    %v5476 = stablehlo.multiply %v5475, %v5475 : tensor<256x256x56x56xf32>
    %v5477 = stablehlo.reduce(%v5476 init: %v5470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5478 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5479 = stablehlo.divide %v5477, %v5478 : tensor<256xf32>
    %v5480 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5482 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5483 = stablehlo.reduce(%v5480 init: %v5481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5484 = stablehlo.divide %v5483, %v5482 : tensor<64xf32>
    %v5485 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5486 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5487 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5488 = stablehlo.reduce(%v5485 init: %v5486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5489 = stablehlo.broadcast_in_dim %v5488, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5490 = stablehlo.divide %v5489, %v5487 : tensor<256x64x56x56xf32>
    %v5491 = stablehlo.subtract %v5485, %v5490 : tensor<256x64x56x56xf32>
    %v5492 = stablehlo.multiply %v5491, %v5491 : tensor<256x64x56x56xf32>
    %v5493 = stablehlo.reduce(%v5492 init: %v5486) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5494 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5495 = stablehlo.divide %v5493, %v5494 : tensor<64xf32>
    %v5496 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5498 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5499 = stablehlo.reduce(%v5496 init: %v5497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5500 = stablehlo.divide %v5499, %v5498 : tensor<64xf32>
    %v5501 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5503 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5504 = stablehlo.reduce(%v5501 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5505 = stablehlo.broadcast_in_dim %v5504, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5506 = stablehlo.divide %v5505, %v5503 : tensor<256x64x56x56xf32>
    %v5507 = stablehlo.subtract %v5501, %v5506 : tensor<256x64x56x56xf32>
    %v5508 = stablehlo.multiply %v5507, %v5507 : tensor<256x64x56x56xf32>
    %v5509 = stablehlo.reduce(%v5508 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5510 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5511 = stablehlo.divide %v5509, %v5510 : tensor<64xf32>
    %v5512 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5514 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5515 = stablehlo.reduce(%v5512 init: %v5513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5516 = stablehlo.divide %v5515, %v5514 : tensor<256xf32>
    %v5517 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5519 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5520 = stablehlo.reduce(%v5517 init: %v5518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5521 = stablehlo.broadcast_in_dim %v5520, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5522 = stablehlo.divide %v5521, %v5519 : tensor<256x256x56x56xf32>
    %v5523 = stablehlo.subtract %v5517, %v5522 : tensor<256x256x56x56xf32>
    %v5524 = stablehlo.multiply %v5523, %v5523 : tensor<256x256x56x56xf32>
    %v5525 = stablehlo.reduce(%v5524 init: %v5518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5526 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5527 = stablehlo.divide %v5525, %v5526 : tensor<256xf32>
    %v5528 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5530 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5531 = stablehlo.reduce(%v5528 init: %v5529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5532 = stablehlo.divide %v5531, %v5530 : tensor<128xf32>
    %v5533 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5535 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v5536 = stablehlo.reduce(%v5533 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5537 = stablehlo.broadcast_in_dim %v5536, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v5538 = stablehlo.divide %v5537, %v5535 : tensor<256x128x56x56xf32>
    %v5539 = stablehlo.subtract %v5533, %v5538 : tensor<256x128x56x56xf32>
    %v5540 = stablehlo.multiply %v5539, %v5539 : tensor<256x128x56x56xf32>
    %v5541 = stablehlo.reduce(%v5540 init: %v5534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5542 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5543 = stablehlo.divide %v5541, %v5542 : tensor<128xf32>
    %v5544 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5546 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5547 = stablehlo.reduce(%v5544 init: %v5545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5548 = stablehlo.divide %v5547, %v5546 : tensor<128xf32>
    %v5549 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5551 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5552 = stablehlo.reduce(%v5549 init: %v5550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5553 = stablehlo.broadcast_in_dim %v5552, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5554 = stablehlo.divide %v5553, %v5551 : tensor<256x128x28x28xf32>
    %v5555 = stablehlo.subtract %v5549, %v5554 : tensor<256x128x28x28xf32>
    %v5556 = stablehlo.multiply %v5555, %v5555 : tensor<256x128x28x28xf32>
    %v5557 = stablehlo.reduce(%v5556 init: %v5550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5558 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5559 = stablehlo.divide %v5557, %v5558 : tensor<128xf32>
    %v5560 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5562 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5563 = stablehlo.reduce(%v5560 init: %v5561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5564 = stablehlo.divide %v5563, %v5562 : tensor<512xf32>
    %v5565 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5567 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5568 = stablehlo.reduce(%v5565 init: %v5566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5569 = stablehlo.broadcast_in_dim %v5568, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5570 = stablehlo.divide %v5569, %v5567 : tensor<256x512x28x28xf32>
    %v5571 = stablehlo.subtract %v5565, %v5570 : tensor<256x512x28x28xf32>
    %v5572 = stablehlo.multiply %v5571, %v5571 : tensor<256x512x28x28xf32>
    %v5573 = stablehlo.reduce(%v5572 init: %v5566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5574 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5575 = stablehlo.divide %v5573, %v5574 : tensor<512xf32>
    %v5576 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5578 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5579 = stablehlo.reduce(%v5576 init: %v5577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5580 = stablehlo.divide %v5579, %v5578 : tensor<512xf32>
    %v5581 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5582 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5583 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5584 = stablehlo.reduce(%v5581 init: %v5582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5585 = stablehlo.broadcast_in_dim %v5584, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5586 = stablehlo.divide %v5585, %v5583 : tensor<256x512x28x28xf32>
    %v5587 = stablehlo.subtract %v5581, %v5586 : tensor<256x512x28x28xf32>
    %v5588 = stablehlo.multiply %v5587, %v5587 : tensor<256x512x28x28xf32>
    %v5589 = stablehlo.reduce(%v5588 init: %v5582) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5590 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5591 = stablehlo.divide %v5589, %v5590 : tensor<512xf32>
    %v5592 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5594 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5595 = stablehlo.reduce(%v5592 init: %v5593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5596 = stablehlo.divide %v5595, %v5594 : tensor<128xf32>
    %v5597 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5599 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5600 = stablehlo.reduce(%v5597 init: %v5598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5601 = stablehlo.broadcast_in_dim %v5600, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5602 = stablehlo.divide %v5601, %v5599 : tensor<256x128x28x28xf32>
    %v5603 = stablehlo.subtract %v5597, %v5602 : tensor<256x128x28x28xf32>
    %v5604 = stablehlo.multiply %v5603, %v5603 : tensor<256x128x28x28xf32>
    %v5605 = stablehlo.reduce(%v5604 init: %v5598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5606 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5607 = stablehlo.divide %v5605, %v5606 : tensor<128xf32>
    %v5608 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5609 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5610 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5611 = stablehlo.reduce(%v5608 init: %v5609) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5612 = stablehlo.divide %v5611, %v5610 : tensor<128xf32>
    %v5613 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5615 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5616 = stablehlo.reduce(%v5613 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5617 = stablehlo.broadcast_in_dim %v5616, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5618 = stablehlo.divide %v5617, %v5615 : tensor<256x128x28x28xf32>
    %v5619 = stablehlo.subtract %v5613, %v5618 : tensor<256x128x28x28xf32>
    %v5620 = stablehlo.multiply %v5619, %v5619 : tensor<256x128x28x28xf32>
    %v5621 = stablehlo.reduce(%v5620 init: %v5614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5622 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5623 = stablehlo.divide %v5621, %v5622 : tensor<128xf32>
    %v5624 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5626 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5627 = stablehlo.reduce(%v5624 init: %v5625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5628 = stablehlo.divide %v5627, %v5626 : tensor<512xf32>
    %v5629 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5631 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5632 = stablehlo.reduce(%v5629 init: %v5630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5633 = stablehlo.broadcast_in_dim %v5632, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5634 = stablehlo.divide %v5633, %v5631 : tensor<256x512x28x28xf32>
    %v5635 = stablehlo.subtract %v5629, %v5634 : tensor<256x512x28x28xf32>
    %v5636 = stablehlo.multiply %v5635, %v5635 : tensor<256x512x28x28xf32>
    %v5637 = stablehlo.reduce(%v5636 init: %v5630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5638 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5639 = stablehlo.divide %v5637, %v5638 : tensor<512xf32>
    %v5640 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5642 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5643 = stablehlo.reduce(%v5640 init: %v5641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5644 = stablehlo.divide %v5643, %v5642 : tensor<128xf32>
    %v5645 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5646 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5647 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5648 = stablehlo.reduce(%v5645 init: %v5646) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5649 = stablehlo.broadcast_in_dim %v5648, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5650 = stablehlo.divide %v5649, %v5647 : tensor<256x128x28x28xf32>
    %v5651 = stablehlo.subtract %v5645, %v5650 : tensor<256x128x28x28xf32>
    %v5652 = stablehlo.multiply %v5651, %v5651 : tensor<256x128x28x28xf32>
    %v5653 = stablehlo.reduce(%v5652 init: %v5646) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5654 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5655 = stablehlo.divide %v5653, %v5654 : tensor<128xf32>
    %v5656 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5658 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5659 = stablehlo.reduce(%v5656 init: %v5657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5660 = stablehlo.divide %v5659, %v5658 : tensor<128xf32>
    %v5661 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5663 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5664 = stablehlo.reduce(%v5661 init: %v5662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5665 = stablehlo.broadcast_in_dim %v5664, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5666 = stablehlo.divide %v5665, %v5663 : tensor<256x128x28x28xf32>
    %v5667 = stablehlo.subtract %v5661, %v5666 : tensor<256x128x28x28xf32>
    %v5668 = stablehlo.multiply %v5667, %v5667 : tensor<256x128x28x28xf32>
    %v5669 = stablehlo.reduce(%v5668 init: %v5662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5670 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5671 = stablehlo.divide %v5669, %v5670 : tensor<128xf32>
    %v5672 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5674 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5675 = stablehlo.reduce(%v5672 init: %v5673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5676 = stablehlo.divide %v5675, %v5674 : tensor<512xf32>
    %v5677 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5679 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5680 = stablehlo.reduce(%v5677 init: %v5678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5681 = stablehlo.broadcast_in_dim %v5680, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5682 = stablehlo.divide %v5681, %v5679 : tensor<256x512x28x28xf32>
    %v5683 = stablehlo.subtract %v5677, %v5682 : tensor<256x512x28x28xf32>
    %v5684 = stablehlo.multiply %v5683, %v5683 : tensor<256x512x28x28xf32>
    %v5685 = stablehlo.reduce(%v5684 init: %v5678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5686 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5687 = stablehlo.divide %v5685, %v5686 : tensor<512xf32>
    %v5688 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5690 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5691 = stablehlo.reduce(%v5688 init: %v5689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5692 = stablehlo.divide %v5691, %v5690 : tensor<128xf32>
    %v5693 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5694 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5695 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5696 = stablehlo.reduce(%v5693 init: %v5694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5697 = stablehlo.broadcast_in_dim %v5696, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5698 = stablehlo.divide %v5697, %v5695 : tensor<256x128x28x28xf32>
    %v5699 = stablehlo.subtract %v5693, %v5698 : tensor<256x128x28x28xf32>
    %v5700 = stablehlo.multiply %v5699, %v5699 : tensor<256x128x28x28xf32>
    %v5701 = stablehlo.reduce(%v5700 init: %v5694) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5702 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5703 = stablehlo.divide %v5701, %v5702 : tensor<128xf32>
    %v5704 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5706 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5707 = stablehlo.reduce(%v5704 init: %v5705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5708 = stablehlo.divide %v5707, %v5706 : tensor<128xf32>
    %v5709 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5710 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5711 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5712 = stablehlo.reduce(%v5709 init: %v5710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5713 = stablehlo.broadcast_in_dim %v5712, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5714 = stablehlo.divide %v5713, %v5711 : tensor<256x128x28x28xf32>
    %v5715 = stablehlo.subtract %v5709, %v5714 : tensor<256x128x28x28xf32>
    %v5716 = stablehlo.multiply %v5715, %v5715 : tensor<256x128x28x28xf32>
    %v5717 = stablehlo.reduce(%v5716 init: %v5710) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5718 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5719 = stablehlo.divide %v5717, %v5718 : tensor<128xf32>
    %v5720 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5722 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5723 = stablehlo.reduce(%v5720 init: %v5721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5724 = stablehlo.divide %v5723, %v5722 : tensor<512xf32>
    %v5725 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5727 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5728 = stablehlo.reduce(%v5725 init: %v5726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5729 = stablehlo.broadcast_in_dim %v5728, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5730 = stablehlo.divide %v5729, %v5727 : tensor<256x512x28x28xf32>
    %v5731 = stablehlo.subtract %v5725, %v5730 : tensor<256x512x28x28xf32>
    %v5732 = stablehlo.multiply %v5731, %v5731 : tensor<256x512x28x28xf32>
    %v5733 = stablehlo.reduce(%v5732 init: %v5726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5734 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5735 = stablehlo.divide %v5733, %v5734 : tensor<512xf32>
    %v5736 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5737 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5738 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5739 = stablehlo.reduce(%v5736 init: %v5737) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5740 = stablehlo.divide %v5739, %v5738 : tensor<256xf32>
    %v5741 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5743 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v5744 = stablehlo.reduce(%v5741 init: %v5742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5745 = stablehlo.broadcast_in_dim %v5744, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v5746 = stablehlo.divide %v5745, %v5743 : tensor<256x256x28x28xf32>
    %v5747 = stablehlo.subtract %v5741, %v5746 : tensor<256x256x28x28xf32>
    %v5748 = stablehlo.multiply %v5747, %v5747 : tensor<256x256x28x28xf32>
    %v5749 = stablehlo.reduce(%v5748 init: %v5742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5750 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5751 = stablehlo.divide %v5749, %v5750 : tensor<256xf32>
    %v5752 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5753 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5754 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5755 = stablehlo.reduce(%v5752 init: %v5753) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5756 = stablehlo.divide %v5755, %v5754 : tensor<256xf32>
    %v5757 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5759 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5760 = stablehlo.reduce(%v5757 init: %v5758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5761 = stablehlo.broadcast_in_dim %v5760, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5762 = stablehlo.divide %v5761, %v5759 : tensor<256x256x14x14xf32>
    %v5763 = stablehlo.subtract %v5757, %v5762 : tensor<256x256x14x14xf32>
    %v5764 = stablehlo.multiply %v5763, %v5763 : tensor<256x256x14x14xf32>
    %v5765 = stablehlo.reduce(%v5764 init: %v5758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5766 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5767 = stablehlo.divide %v5765, %v5766 : tensor<256xf32>
    %v5768 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5770 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5771 = stablehlo.reduce(%v5768 init: %v5769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5772 = stablehlo.divide %v5771, %v5770 : tensor<1024xf32>
    %v5773 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5775 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5776 = stablehlo.reduce(%v5773 init: %v5774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5777 = stablehlo.broadcast_in_dim %v5776, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5778 = stablehlo.divide %v5777, %v5775 : tensor<256x1024x14x14xf32>
    %v5779 = stablehlo.subtract %v5773, %v5778 : tensor<256x1024x14x14xf32>
    %v5780 = stablehlo.multiply %v5779, %v5779 : tensor<256x1024x14x14xf32>
    %v5781 = stablehlo.reduce(%v5780 init: %v5774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5782 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5783 = stablehlo.divide %v5781, %v5782 : tensor<1024xf32>
    %v5784 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5786 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5787 = stablehlo.reduce(%v5784 init: %v5785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5788 = stablehlo.divide %v5787, %v5786 : tensor<1024xf32>
    %v5789 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5791 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5792 = stablehlo.reduce(%v5789 init: %v5790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5793 = stablehlo.broadcast_in_dim %v5792, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5794 = stablehlo.divide %v5793, %v5791 : tensor<256x1024x14x14xf32>
    %v5795 = stablehlo.subtract %v5789, %v5794 : tensor<256x1024x14x14xf32>
    %v5796 = stablehlo.multiply %v5795, %v5795 : tensor<256x1024x14x14xf32>
    %v5797 = stablehlo.reduce(%v5796 init: %v5790) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5798 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5799 = stablehlo.divide %v5797, %v5798 : tensor<1024xf32>
    %v5800 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5802 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5803 = stablehlo.reduce(%v5800 init: %v5801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5804 = stablehlo.divide %v5803, %v5802 : tensor<256xf32>
    %v5805 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5806 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5807 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5808 = stablehlo.reduce(%v5805 init: %v5806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5809 = stablehlo.broadcast_in_dim %v5808, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5810 = stablehlo.divide %v5809, %v5807 : tensor<256x256x14x14xf32>
    %v5811 = stablehlo.subtract %v5805, %v5810 : tensor<256x256x14x14xf32>
    %v5812 = stablehlo.multiply %v5811, %v5811 : tensor<256x256x14x14xf32>
    %v5813 = stablehlo.reduce(%v5812 init: %v5806) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5814 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5815 = stablehlo.divide %v5813, %v5814 : tensor<256xf32>
    %v5816 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5817 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5818 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5819 = stablehlo.reduce(%v5816 init: %v5817) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5820 = stablehlo.divide %v5819, %v5818 : tensor<256xf32>
    %v5821 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5823 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5824 = stablehlo.reduce(%v5821 init: %v5822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5825 = stablehlo.broadcast_in_dim %v5824, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5826 = stablehlo.divide %v5825, %v5823 : tensor<256x256x14x14xf32>
    %v5827 = stablehlo.subtract %v5821, %v5826 : tensor<256x256x14x14xf32>
    %v5828 = stablehlo.multiply %v5827, %v5827 : tensor<256x256x14x14xf32>
    %v5829 = stablehlo.reduce(%v5828 init: %v5822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5830 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5831 = stablehlo.divide %v5829, %v5830 : tensor<256xf32>
    %v5832 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5834 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5835 = stablehlo.reduce(%v5832 init: %v5833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5836 = stablehlo.divide %v5835, %v5834 : tensor<1024xf32>
    %v5837 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5839 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5840 = stablehlo.reduce(%v5837 init: %v5838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5841 = stablehlo.broadcast_in_dim %v5840, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5842 = stablehlo.divide %v5841, %v5839 : tensor<256x1024x14x14xf32>
    %v5843 = stablehlo.subtract %v5837, %v5842 : tensor<256x1024x14x14xf32>
    %v5844 = stablehlo.multiply %v5843, %v5843 : tensor<256x1024x14x14xf32>
    %v5845 = stablehlo.reduce(%v5844 init: %v5838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5846 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5847 = stablehlo.divide %v5845, %v5846 : tensor<1024xf32>
    %v5848 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5850 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5851 = stablehlo.reduce(%v5848 init: %v5849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5852 = stablehlo.divide %v5851, %v5850 : tensor<256xf32>
    %v5853 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5855 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5856 = stablehlo.reduce(%v5853 init: %v5854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5857 = stablehlo.broadcast_in_dim %v5856, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5858 = stablehlo.divide %v5857, %v5855 : tensor<256x256x14x14xf32>
    %v5859 = stablehlo.subtract %v5853, %v5858 : tensor<256x256x14x14xf32>
    %v5860 = stablehlo.multiply %v5859, %v5859 : tensor<256x256x14x14xf32>
    %v5861 = stablehlo.reduce(%v5860 init: %v5854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5862 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5863 = stablehlo.divide %v5861, %v5862 : tensor<256xf32>
    %v5864 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5866 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5867 = stablehlo.reduce(%v5864 init: %v5865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5868 = stablehlo.divide %v5867, %v5866 : tensor<256xf32>
    %v5869 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5871 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5872 = stablehlo.reduce(%v5869 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5873 = stablehlo.broadcast_in_dim %v5872, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5874 = stablehlo.divide %v5873, %v5871 : tensor<256x256x14x14xf32>
    %v5875 = stablehlo.subtract %v5869, %v5874 : tensor<256x256x14x14xf32>
    %v5876 = stablehlo.multiply %v5875, %v5875 : tensor<256x256x14x14xf32>
    %v5877 = stablehlo.reduce(%v5876 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5878 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5879 = stablehlo.divide %v5877, %v5878 : tensor<256xf32>
    %v5880 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5881 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5882 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5883 = stablehlo.reduce(%v5880 init: %v5881) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5884 = stablehlo.divide %v5883, %v5882 : tensor<1024xf32>
    %v5885 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5887 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5888 = stablehlo.reduce(%v5885 init: %v5886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5889 = stablehlo.broadcast_in_dim %v5888, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5890 = stablehlo.divide %v5889, %v5887 : tensor<256x1024x14x14xf32>
    %v5891 = stablehlo.subtract %v5885, %v5890 : tensor<256x1024x14x14xf32>
    %v5892 = stablehlo.multiply %v5891, %v5891 : tensor<256x1024x14x14xf32>
    %v5893 = stablehlo.reduce(%v5892 init: %v5886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5894 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5895 = stablehlo.divide %v5893, %v5894 : tensor<1024xf32>
    %v5896 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5898 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5899 = stablehlo.reduce(%v5896 init: %v5897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5900 = stablehlo.divide %v5899, %v5898 : tensor<256xf32>
    %v5901 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5903 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5904 = stablehlo.reduce(%v5901 init: %v5902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5905 = stablehlo.broadcast_in_dim %v5904, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5906 = stablehlo.divide %v5905, %v5903 : tensor<256x256x14x14xf32>
    %v5907 = stablehlo.subtract %v5901, %v5906 : tensor<256x256x14x14xf32>
    %v5908 = stablehlo.multiply %v5907, %v5907 : tensor<256x256x14x14xf32>
    %v5909 = stablehlo.reduce(%v5908 init: %v5902) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5910 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5911 = stablehlo.divide %v5909, %v5910 : tensor<256xf32>
    %v5912 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5914 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5915 = stablehlo.reduce(%v5912 init: %v5913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5916 = stablehlo.divide %v5915, %v5914 : tensor<256xf32>
    %v5917 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5919 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5920 = stablehlo.reduce(%v5917 init: %v5918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5921 = stablehlo.broadcast_in_dim %v5920, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5922 = stablehlo.divide %v5921, %v5919 : tensor<256x256x14x14xf32>
    %v5923 = stablehlo.subtract %v5917, %v5922 : tensor<256x256x14x14xf32>
    %v5924 = stablehlo.multiply %v5923, %v5923 : tensor<256x256x14x14xf32>
    %v5925 = stablehlo.reduce(%v5924 init: %v5918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5926 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5927 = stablehlo.divide %v5925, %v5926 : tensor<256xf32>
    %v5928 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5930 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5931 = stablehlo.reduce(%v5928 init: %v5929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5932 = stablehlo.divide %v5931, %v5930 : tensor<1024xf32>
    %v5933 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5935 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5936 = stablehlo.reduce(%v5933 init: %v5934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5937 = stablehlo.broadcast_in_dim %v5936, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5938 = stablehlo.divide %v5937, %v5935 : tensor<256x1024x14x14xf32>
    %v5939 = stablehlo.subtract %v5933, %v5938 : tensor<256x1024x14x14xf32>
    %v5940 = stablehlo.multiply %v5939, %v5939 : tensor<256x1024x14x14xf32>
    %v5941 = stablehlo.reduce(%v5940 init: %v5934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5942 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5943 = stablehlo.divide %v5941, %v5942 : tensor<1024xf32>
    %v5944 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5946 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5947 = stablehlo.reduce(%v5944 init: %v5945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5948 = stablehlo.divide %v5947, %v5946 : tensor<256xf32>
    %v5949 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5951 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5952 = stablehlo.reduce(%v5949 init: %v5950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5953 = stablehlo.broadcast_in_dim %v5952, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5954 = stablehlo.divide %v5953, %v5951 : tensor<256x256x14x14xf32>
    %v5955 = stablehlo.subtract %v5949, %v5954 : tensor<256x256x14x14xf32>
    %v5956 = stablehlo.multiply %v5955, %v5955 : tensor<256x256x14x14xf32>
    %v5957 = stablehlo.reduce(%v5956 init: %v5950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5958 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5959 = stablehlo.divide %v5957, %v5958 : tensor<256xf32>
    %v5960 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5962 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5963 = stablehlo.reduce(%v5960 init: %v5961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5964 = stablehlo.divide %v5963, %v5962 : tensor<256xf32>
    %v5965 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5967 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5968 = stablehlo.reduce(%v5965 init: %v5966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5969 = stablehlo.broadcast_in_dim %v5968, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5970 = stablehlo.divide %v5969, %v5967 : tensor<256x256x14x14xf32>
    %v5971 = stablehlo.subtract %v5965, %v5970 : tensor<256x256x14x14xf32>
    %v5972 = stablehlo.multiply %v5971, %v5971 : tensor<256x256x14x14xf32>
    %v5973 = stablehlo.reduce(%v5972 init: %v5966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5974 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5975 = stablehlo.divide %v5973, %v5974 : tensor<256xf32>
    %v5976 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5978 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5979 = stablehlo.reduce(%v5976 init: %v5977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5980 = stablehlo.divide %v5979, %v5978 : tensor<1024xf32>
    %v5981 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5983 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5984 = stablehlo.reduce(%v5981 init: %v5982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5985 = stablehlo.broadcast_in_dim %v5984, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5986 = stablehlo.divide %v5985, %v5983 : tensor<256x1024x14x14xf32>
    %v5987 = stablehlo.subtract %v5981, %v5986 : tensor<256x1024x14x14xf32>
    %v5988 = stablehlo.multiply %v5987, %v5987 : tensor<256x1024x14x14xf32>
    %v5989 = stablehlo.reduce(%v5988 init: %v5982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5990 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5991 = stablehlo.divide %v5989, %v5990 : tensor<1024xf32>
    %v5992 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5994 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5995 = stablehlo.reduce(%v5992 init: %v5993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5996 = stablehlo.divide %v5995, %v5994 : tensor<256xf32>
    %v5997 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5999 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v6000 = stablehlo.reduce(%v5997 init: %v5998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6001 = stablehlo.broadcast_in_dim %v6000, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v6002 = stablehlo.divide %v6001, %v5999 : tensor<256x256x14x14xf32>
    %v6003 = stablehlo.subtract %v5997, %v6002 : tensor<256x256x14x14xf32>
    %v6004 = stablehlo.multiply %v6003, %v6003 : tensor<256x256x14x14xf32>
    %v6005 = stablehlo.reduce(%v6004 init: %v5998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6006 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v6007 = stablehlo.divide %v6005, %v6006 : tensor<256xf32>
    %v6008 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v6009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6010 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v6011 = stablehlo.reduce(%v6008 init: %v6009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6012 = stablehlo.divide %v6011, %v6010 : tensor<256xf32>
    %v6013 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v6014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6015 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v6016 = stablehlo.reduce(%v6013 init: %v6014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6017 = stablehlo.broadcast_in_dim %v6016, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v6018 = stablehlo.divide %v6017, %v6015 : tensor<256x256x14x14xf32>
    %v6019 = stablehlo.subtract %v6013, %v6018 : tensor<256x256x14x14xf32>
    %v6020 = stablehlo.multiply %v6019, %v6019 : tensor<256x256x14x14xf32>
    %v6021 = stablehlo.reduce(%v6020 init: %v6014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6022 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v6023 = stablehlo.divide %v6021, %v6022 : tensor<256xf32>
    %v6024 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v6025 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6026 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v6027 = stablehlo.reduce(%v6024 init: %v6025) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6028 = stablehlo.divide %v6027, %v6026 : tensor<1024xf32>
    %v6029 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v6030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6031 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v6032 = stablehlo.reduce(%v6029 init: %v6030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6033 = stablehlo.broadcast_in_dim %v6032, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v6034 = stablehlo.divide %v6033, %v6031 : tensor<256x1024x14x14xf32>
    %v6035 = stablehlo.subtract %v6029, %v6034 : tensor<256x1024x14x14xf32>
    %v6036 = stablehlo.multiply %v6035, %v6035 : tensor<256x1024x14x14xf32>
    %v6037 = stablehlo.reduce(%v6036 init: %v6030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6038 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v6039 = stablehlo.divide %v6037, %v6038 : tensor<1024xf32>
    %v6040 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v6041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6042 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6043 = stablehlo.reduce(%v6040 init: %v6041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6044 = stablehlo.divide %v6043, %v6042 : tensor<512xf32>
    %v6045 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v6046 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6047 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v6048 = stablehlo.reduce(%v6045 init: %v6046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6049 = stablehlo.broadcast_in_dim %v6048, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v6050 = stablehlo.divide %v6049, %v6047 : tensor<256x512x14x14xf32>
    %v6051 = stablehlo.subtract %v6045, %v6050 : tensor<256x512x14x14xf32>
    %v6052 = stablehlo.multiply %v6051, %v6051 : tensor<256x512x14x14xf32>
    %v6053 = stablehlo.reduce(%v6052 init: %v6046) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6054 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6055 = stablehlo.divide %v6053, %v6054 : tensor<512xf32>
    %v6056 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6058 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6059 = stablehlo.reduce(%v6056 init: %v6057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6060 = stablehlo.divide %v6059, %v6058 : tensor<512xf32>
    %v6061 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6063 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6064 = stablehlo.reduce(%v6061 init: %v6062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6065 = stablehlo.broadcast_in_dim %v6064, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6066 = stablehlo.divide %v6065, %v6063 : tensor<256x512x7x7xf32>
    %v6067 = stablehlo.subtract %v6061, %v6066 : tensor<256x512x7x7xf32>
    %v6068 = stablehlo.multiply %v6067, %v6067 : tensor<256x512x7x7xf32>
    %v6069 = stablehlo.reduce(%v6068 init: %v6062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6070 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6071 = stablehlo.divide %v6069, %v6070 : tensor<512xf32>
    %v6072 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6073 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6074 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6075 = stablehlo.reduce(%v6072 init: %v6073) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6076 = stablehlo.divide %v6075, %v6074 : tensor<2048xf32>
    %v6077 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6079 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6080 = stablehlo.reduce(%v6077 init: %v6078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6081 = stablehlo.broadcast_in_dim %v6080, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6082 = stablehlo.divide %v6081, %v6079 : tensor<256x2048x7x7xf32>
    %v6083 = stablehlo.subtract %v6077, %v6082 : tensor<256x2048x7x7xf32>
    %v6084 = stablehlo.multiply %v6083, %v6083 : tensor<256x2048x7x7xf32>
    %v6085 = stablehlo.reduce(%v6084 init: %v6078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6086 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6087 = stablehlo.divide %v6085, %v6086 : tensor<2048xf32>
    %v6088 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6090 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6091 = stablehlo.reduce(%v6088 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6092 = stablehlo.divide %v6091, %v6090 : tensor<2048xf32>
    %v6093 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6095 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6096 = stablehlo.reduce(%v6093 init: %v6094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6097 = stablehlo.broadcast_in_dim %v6096, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6098 = stablehlo.divide %v6097, %v6095 : tensor<256x2048x7x7xf32>
    %v6099 = stablehlo.subtract %v6093, %v6098 : tensor<256x2048x7x7xf32>
    %v6100 = stablehlo.multiply %v6099, %v6099 : tensor<256x2048x7x7xf32>
    %v6101 = stablehlo.reduce(%v6100 init: %v6094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6102 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6103 = stablehlo.divide %v6101, %v6102 : tensor<2048xf32>
    %v6104 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6105 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6106 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6107 = stablehlo.reduce(%v6104 init: %v6105) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6108 = stablehlo.divide %v6107, %v6106 : tensor<512xf32>
    %v6109 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6111 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6112 = stablehlo.reduce(%v6109 init: %v6110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6113 = stablehlo.broadcast_in_dim %v6112, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6114 = stablehlo.divide %v6113, %v6111 : tensor<256x512x7x7xf32>
    %v6115 = stablehlo.subtract %v6109, %v6114 : tensor<256x512x7x7xf32>
    %v6116 = stablehlo.multiply %v6115, %v6115 : tensor<256x512x7x7xf32>
    %v6117 = stablehlo.reduce(%v6116 init: %v6110) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6118 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6119 = stablehlo.divide %v6117, %v6118 : tensor<512xf32>
    %v6120 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6122 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6123 = stablehlo.reduce(%v6120 init: %v6121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6124 = stablehlo.divide %v6123, %v6122 : tensor<512xf32>
    %v6125 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6126 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6127 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6128 = stablehlo.reduce(%v6125 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6129 = stablehlo.broadcast_in_dim %v6128, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6130 = stablehlo.divide %v6129, %v6127 : tensor<256x512x7x7xf32>
    %v6131 = stablehlo.subtract %v6125, %v6130 : tensor<256x512x7x7xf32>
    %v6132 = stablehlo.multiply %v6131, %v6131 : tensor<256x512x7x7xf32>
    %v6133 = stablehlo.reduce(%v6132 init: %v6126) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6134 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6135 = stablehlo.divide %v6133, %v6134 : tensor<512xf32>
    %v6136 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6138 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6139 = stablehlo.reduce(%v6136 init: %v6137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6140 = stablehlo.divide %v6139, %v6138 : tensor<2048xf32>
    %v6141 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6143 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6144 = stablehlo.reduce(%v6141 init: %v6142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6145 = stablehlo.broadcast_in_dim %v6144, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6146 = stablehlo.divide %v6145, %v6143 : tensor<256x2048x7x7xf32>
    %v6147 = stablehlo.subtract %v6141, %v6146 : tensor<256x2048x7x7xf32>
    %v6148 = stablehlo.multiply %v6147, %v6147 : tensor<256x2048x7x7xf32>
    %v6149 = stablehlo.reduce(%v6148 init: %v6142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6150 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6151 = stablehlo.divide %v6149, %v6150 : tensor<2048xf32>
    %v6152 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6154 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6155 = stablehlo.reduce(%v6152 init: %v6153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6156 = stablehlo.divide %v6155, %v6154 : tensor<512xf32>
    %v6157 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6159 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6160 = stablehlo.reduce(%v6157 init: %v6158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6161 = stablehlo.broadcast_in_dim %v6160, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6162 = stablehlo.divide %v6161, %v6159 : tensor<256x512x7x7xf32>
    %v6163 = stablehlo.subtract %v6157, %v6162 : tensor<256x512x7x7xf32>
    %v6164 = stablehlo.multiply %v6163, %v6163 : tensor<256x512x7x7xf32>
    %v6165 = stablehlo.reduce(%v6164 init: %v6158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6166 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6167 = stablehlo.divide %v6165, %v6166 : tensor<512xf32>
    %v6168 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6169 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6170 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6171 = stablehlo.reduce(%v6168 init: %v6169) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6172 = stablehlo.divide %v6171, %v6170 : tensor<512xf32>
    %v6173 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6174 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6175 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6176 = stablehlo.reduce(%v6173 init: %v6174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6177 = stablehlo.broadcast_in_dim %v6176, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6178 = stablehlo.divide %v6177, %v6175 : tensor<256x512x7x7xf32>
    %v6179 = stablehlo.subtract %v6173, %v6178 : tensor<256x512x7x7xf32>
    %v6180 = stablehlo.multiply %v6179, %v6179 : tensor<256x512x7x7xf32>
    %v6181 = stablehlo.reduce(%v6180 init: %v6174) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6182 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6183 = stablehlo.divide %v6181, %v6182 : tensor<512xf32>
    %v6184 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6186 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6187 = stablehlo.reduce(%v6184 init: %v6185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6188 = stablehlo.divide %v6187, %v6186 : tensor<2048xf32>
    %v6189 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6191 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6192 = stablehlo.reduce(%v6189 init: %v6190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6193 = stablehlo.broadcast_in_dim %v6192, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6194 = stablehlo.divide %v6193, %v6191 : tensor<256x2048x7x7xf32>
    %v6195 = stablehlo.subtract %v6189, %v6194 : tensor<256x2048x7x7xf32>
    %v6196 = stablehlo.multiply %v6195, %v6195 : tensor<256x2048x7x7xf32>
    %v6197 = stablehlo.reduce(%v6196 init: %v6190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6198 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6199 = stablehlo.divide %v6197, %v6198 : tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v6200 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6201 = stablehlo.multiply %v6200, %sW : tensor<64x3x7x7xf32>
    %v6202 = stablehlo.add %v6201, %v5330 : tensor<64x3x7x7xf32>
    %v6203 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6204 = stablehlo.multiply %v6203, %sWv : tensor<64x3x7x7xf32>
    %v6205 = stablehlo.add %v6204, %v6202 : tensor<64x3x7x7xf32>
    %v6206 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6207 = stablehlo.multiply %v6206, %v6205 : tensor<64x3x7x7xf32>
    %v6208 = stablehlo.subtract %sW, %v6207 : tensor<64x3x7x7xf32>
    %v6209 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6210 = stablehlo.multiply %v6209, %sg : tensor<64xf32>
    %v6211 = stablehlo.add %v6210, %v5348 : tensor<64xf32>
    %v6212 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6213 = stablehlo.multiply %v6212, %sgv : tensor<64xf32>
    %v6214 = stablehlo.add %v6213, %v6211 : tensor<64xf32>
    %v6215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6216 = stablehlo.multiply %v6215, %v6214 : tensor<64xf32>
    %v6217 = stablehlo.subtract %sg, %v6216 : tensor<64xf32>
    %v6218 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6219 = stablehlo.multiply %v6218, %sbt : tensor<64xf32>
    %v6220 = stablehlo.add %v6219, %v5351 : tensor<64xf32>
    %v6221 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6222 = stablehlo.multiply %v6221, %sbtv : tensor<64xf32>
    %v6223 = stablehlo.add %v6222, %v6220 : tensor<64xf32>
    %v6224 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6225 = stablehlo.multiply %v6224, %v6223 : tensor<64xf32>
    %v6226 = stablehlo.subtract %sbt, %v6225 : tensor<64xf32>
    %v6227 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6228 = stablehlo.multiply %v6227, %s1b0W1 : tensor<64x64x1x1xf32>
    %v6229 = stablehlo.add %v6228, %v5101 : tensor<64x64x1x1xf32>
    %v6230 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6231 = stablehlo.multiply %v6230, %s1b0W1v : tensor<64x64x1x1xf32>
    %v6232 = stablehlo.add %v6231, %v6229 : tensor<64x64x1x1xf32>
    %v6233 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6234 = stablehlo.multiply %v6233, %v6232 : tensor<64x64x1x1xf32>
    %v6235 = stablehlo.subtract %s1b0W1, %v6234 : tensor<64x64x1x1xf32>
    %v6236 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6237 = stablehlo.multiply %v6236, %s1b0g1 : tensor<64xf32>
    %v6238 = stablehlo.add %v6237, %v5119 : tensor<64xf32>
    %v6239 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6240 = stablehlo.multiply %v6239, %s1b0g1v : tensor<64xf32>
    %v6241 = stablehlo.add %v6240, %v6238 : tensor<64xf32>
    %v6242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6243 = stablehlo.multiply %v6242, %v6241 : tensor<64xf32>
    %v6244 = stablehlo.subtract %s1b0g1, %v6243 : tensor<64xf32>
    %v6245 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6246 = stablehlo.multiply %v6245, %s1b0bt1 : tensor<64xf32>
    %v6247 = stablehlo.add %v6246, %v5122 : tensor<64xf32>
    %v6248 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6249 = stablehlo.multiply %v6248, %s1b0bt1v : tensor<64xf32>
    %v6250 = stablehlo.add %v6249, %v6247 : tensor<64xf32>
    %v6251 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6252 = stablehlo.multiply %v6251, %v6250 : tensor<64xf32>
    %v6253 = stablehlo.subtract %s1b0bt1, %v6252 : tensor<64xf32>
    %v6254 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6255 = stablehlo.multiply %v6254, %s1b0W2 : tensor<64x64x3x3xf32>
    %v6256 = stablehlo.add %v6255, %v5128 : tensor<64x64x3x3xf32>
    %v6257 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6258 = stablehlo.multiply %v6257, %s1b0W2v : tensor<64x64x3x3xf32>
    %v6259 = stablehlo.add %v6258, %v6256 : tensor<64x64x3x3xf32>
    %v6260 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6261 = stablehlo.multiply %v6260, %v6259 : tensor<64x64x3x3xf32>
    %v6262 = stablehlo.subtract %s1b0W2, %v6261 : tensor<64x64x3x3xf32>
    %v6263 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6264 = stablehlo.multiply %v6263, %s1b0g2 : tensor<64xf32>
    %v6265 = stablehlo.add %v6264, %v5146 : tensor<64xf32>
    %v6266 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6267 = stablehlo.multiply %v6266, %s1b0g2v : tensor<64xf32>
    %v6268 = stablehlo.add %v6267, %v6265 : tensor<64xf32>
    %v6269 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6270 = stablehlo.multiply %v6269, %v6268 : tensor<64xf32>
    %v6271 = stablehlo.subtract %s1b0g2, %v6270 : tensor<64xf32>
    %v6272 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6273 = stablehlo.multiply %v6272, %s1b0bt2 : tensor<64xf32>
    %v6274 = stablehlo.add %v6273, %v5149 : tensor<64xf32>
    %v6275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6276 = stablehlo.multiply %v6275, %s1b0bt2v : tensor<64xf32>
    %v6277 = stablehlo.add %v6276, %v6274 : tensor<64xf32>
    %v6278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6279 = stablehlo.multiply %v6278, %v6277 : tensor<64xf32>
    %v6280 = stablehlo.subtract %s1b0bt2, %v6279 : tensor<64xf32>
    %v6281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6282 = stablehlo.multiply %v6281, %s1b0W3 : tensor<256x64x1x1xf32>
    %v6283 = stablehlo.add %v6282, %v5155 : tensor<256x64x1x1xf32>
    %v6284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6285 = stablehlo.multiply %v6284, %s1b0W3v : tensor<256x64x1x1xf32>
    %v6286 = stablehlo.add %v6285, %v6283 : tensor<256x64x1x1xf32>
    %v6287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6288 = stablehlo.multiply %v6287, %v6286 : tensor<256x64x1x1xf32>
    %v6289 = stablehlo.subtract %s1b0W3, %v6288 : tensor<256x64x1x1xf32>
    %v6290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6291 = stablehlo.multiply %v6290, %s1b0g3 : tensor<256xf32>
    %v6292 = stablehlo.add %v6291, %v5173 : tensor<256xf32>
    %v6293 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6294 = stablehlo.multiply %v6293, %s1b0g3v : tensor<256xf32>
    %v6295 = stablehlo.add %v6294, %v6292 : tensor<256xf32>
    %v6296 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6297 = stablehlo.multiply %v6296, %v6295 : tensor<256xf32>
    %v6298 = stablehlo.subtract %s1b0g3, %v6297 : tensor<256xf32>
    %v6299 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6300 = stablehlo.multiply %v6299, %s1b0bt3 : tensor<256xf32>
    %v6301 = stablehlo.add %v6300, %v5176 : tensor<256xf32>
    %v6302 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6303 = stablehlo.multiply %v6302, %s1b0bt3v : tensor<256xf32>
    %v6304 = stablehlo.add %v6303, %v6301 : tensor<256xf32>
    %v6305 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6306 = stablehlo.multiply %v6305, %v6304 : tensor<256xf32>
    %v6307 = stablehlo.subtract %s1b0bt3, %v6306 : tensor<256xf32>
    %v6308 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6309 = stablehlo.multiply %v6308, %s1b0Wp : tensor<256x64x1x1xf32>
    %v6310 = stablehlo.add %v6309, %v5182 : tensor<256x64x1x1xf32>
    %v6311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6312 = stablehlo.multiply %v6311, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v6313 = stablehlo.add %v6312, %v6310 : tensor<256x64x1x1xf32>
    %v6314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6315 = stablehlo.multiply %v6314, %v6313 : tensor<256x64x1x1xf32>
    %v6316 = stablehlo.subtract %s1b0Wp, %v6315 : tensor<256x64x1x1xf32>
    %v6317 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6318 = stablehlo.multiply %v6317, %s1b0gp : tensor<256xf32>
    %v6319 = stablehlo.add %v6318, %v5200 : tensor<256xf32>
    %v6320 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6321 = stablehlo.multiply %v6320, %s1b0gpv : tensor<256xf32>
    %v6322 = stablehlo.add %v6321, %v6319 : tensor<256xf32>
    %v6323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6324 = stablehlo.multiply %v6323, %v6322 : tensor<256xf32>
    %v6325 = stablehlo.subtract %s1b0gp, %v6324 : tensor<256xf32>
    %v6326 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6327 = stablehlo.multiply %v6326, %s1b0btp : tensor<256xf32>
    %v6328 = stablehlo.add %v6327, %v5203 : tensor<256xf32>
    %v6329 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6330 = stablehlo.multiply %v6329, %s1b0btpv : tensor<256xf32>
    %v6331 = stablehlo.add %v6330, %v6328 : tensor<256xf32>
    %v6332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6333 = stablehlo.multiply %v6332, %v6331 : tensor<256xf32>
    %v6334 = stablehlo.subtract %s1b0btp, %v6333 : tensor<256xf32>
    %v6335 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6336 = stablehlo.multiply %v6335, %s1b1W1 : tensor<64x256x1x1xf32>
    %v6337 = stablehlo.add %v6336, %v4858 : tensor<64x256x1x1xf32>
    %v6338 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6339 = stablehlo.multiply %v6338, %s1b1W1v : tensor<64x256x1x1xf32>
    %v6340 = stablehlo.add %v6339, %v6337 : tensor<64x256x1x1xf32>
    %v6341 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6342 = stablehlo.multiply %v6341, %v6340 : tensor<64x256x1x1xf32>
    %v6343 = stablehlo.subtract %s1b1W1, %v6342 : tensor<64x256x1x1xf32>
    %v6344 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6345 = stablehlo.multiply %v6344, %s1b1g1 : tensor<64xf32>
    %v6346 = stablehlo.add %v6345, %v4876 : tensor<64xf32>
    %v6347 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6348 = stablehlo.multiply %v6347, %s1b1g1v : tensor<64xf32>
    %v6349 = stablehlo.add %v6348, %v6346 : tensor<64xf32>
    %v6350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6351 = stablehlo.multiply %v6350, %v6349 : tensor<64xf32>
    %v6352 = stablehlo.subtract %s1b1g1, %v6351 : tensor<64xf32>
    %v6353 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6354 = stablehlo.multiply %v6353, %s1b1bt1 : tensor<64xf32>
    %v6355 = stablehlo.add %v6354, %v4879 : tensor<64xf32>
    %v6356 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6357 = stablehlo.multiply %v6356, %s1b1bt1v : tensor<64xf32>
    %v6358 = stablehlo.add %v6357, %v6355 : tensor<64xf32>
    %v6359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6360 = stablehlo.multiply %v6359, %v6358 : tensor<64xf32>
    %v6361 = stablehlo.subtract %s1b1bt1, %v6360 : tensor<64xf32>
    %v6362 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6363 = stablehlo.multiply %v6362, %s1b1W2 : tensor<64x64x3x3xf32>
    %v6364 = stablehlo.add %v6363, %v4885 : tensor<64x64x3x3xf32>
    %v6365 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6366 = stablehlo.multiply %v6365, %s1b1W2v : tensor<64x64x3x3xf32>
    %v6367 = stablehlo.add %v6366, %v6364 : tensor<64x64x3x3xf32>
    %v6368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6369 = stablehlo.multiply %v6368, %v6367 : tensor<64x64x3x3xf32>
    %v6370 = stablehlo.subtract %s1b1W2, %v6369 : tensor<64x64x3x3xf32>
    %v6371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6372 = stablehlo.multiply %v6371, %s1b1g2 : tensor<64xf32>
    %v6373 = stablehlo.add %v6372, %v4903 : tensor<64xf32>
    %v6374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6375 = stablehlo.multiply %v6374, %s1b1g2v : tensor<64xf32>
    %v6376 = stablehlo.add %v6375, %v6373 : tensor<64xf32>
    %v6377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6378 = stablehlo.multiply %v6377, %v6376 : tensor<64xf32>
    %v6379 = stablehlo.subtract %s1b1g2, %v6378 : tensor<64xf32>
    %v6380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6381 = stablehlo.multiply %v6380, %s1b1bt2 : tensor<64xf32>
    %v6382 = stablehlo.add %v6381, %v4906 : tensor<64xf32>
    %v6383 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6384 = stablehlo.multiply %v6383, %s1b1bt2v : tensor<64xf32>
    %v6385 = stablehlo.add %v6384, %v6382 : tensor<64xf32>
    %v6386 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6387 = stablehlo.multiply %v6386, %v6385 : tensor<64xf32>
    %v6388 = stablehlo.subtract %s1b1bt2, %v6387 : tensor<64xf32>
    %v6389 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6390 = stablehlo.multiply %v6389, %s1b1W3 : tensor<256x64x1x1xf32>
    %v6391 = stablehlo.add %v6390, %v4912 : tensor<256x64x1x1xf32>
    %v6392 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6393 = stablehlo.multiply %v6392, %s1b1W3v : tensor<256x64x1x1xf32>
    %v6394 = stablehlo.add %v6393, %v6391 : tensor<256x64x1x1xf32>
    %v6395 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6396 = stablehlo.multiply %v6395, %v6394 : tensor<256x64x1x1xf32>
    %v6397 = stablehlo.subtract %s1b1W3, %v6396 : tensor<256x64x1x1xf32>
    %v6398 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6399 = stablehlo.multiply %v6398, %s1b1g3 : tensor<256xf32>
    %v6400 = stablehlo.add %v6399, %v4930 : tensor<256xf32>
    %v6401 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6402 = stablehlo.multiply %v6401, %s1b1g3v : tensor<256xf32>
    %v6403 = stablehlo.add %v6402, %v6400 : tensor<256xf32>
    %v6404 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6405 = stablehlo.multiply %v6404, %v6403 : tensor<256xf32>
    %v6406 = stablehlo.subtract %s1b1g3, %v6405 : tensor<256xf32>
    %v6407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6408 = stablehlo.multiply %v6407, %s1b1bt3 : tensor<256xf32>
    %v6409 = stablehlo.add %v6408, %v4933 : tensor<256xf32>
    %v6410 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6411 = stablehlo.multiply %v6410, %s1b1bt3v : tensor<256xf32>
    %v6412 = stablehlo.add %v6411, %v6409 : tensor<256xf32>
    %v6413 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6414 = stablehlo.multiply %v6413, %v6412 : tensor<256xf32>
    %v6415 = stablehlo.subtract %s1b1bt3, %v6414 : tensor<256xf32>
    %v6416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6417 = stablehlo.multiply %v6416, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6418 = stablehlo.add %v6417, %v4650 : tensor<64x256x1x1xf32>
    %v6419 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6420 = stablehlo.multiply %v6419, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6421 = stablehlo.add %v6420, %v6418 : tensor<64x256x1x1xf32>
    %v6422 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6423 = stablehlo.multiply %v6422, %v6421 : tensor<64x256x1x1xf32>
    %v6424 = stablehlo.subtract %s1b2W1, %v6423 : tensor<64x256x1x1xf32>
    %v6425 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6426 = stablehlo.multiply %v6425, %s1b2g1 : tensor<64xf32>
    %v6427 = stablehlo.add %v6426, %v4668 : tensor<64xf32>
    %v6428 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6429 = stablehlo.multiply %v6428, %s1b2g1v : tensor<64xf32>
    %v6430 = stablehlo.add %v6429, %v6427 : tensor<64xf32>
    %v6431 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6432 = stablehlo.multiply %v6431, %v6430 : tensor<64xf32>
    %v6433 = stablehlo.subtract %s1b2g1, %v6432 : tensor<64xf32>
    %v6434 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6435 = stablehlo.multiply %v6434, %s1b2bt1 : tensor<64xf32>
    %v6436 = stablehlo.add %v6435, %v4671 : tensor<64xf32>
    %v6437 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6438 = stablehlo.multiply %v6437, %s1b2bt1v : tensor<64xf32>
    %v6439 = stablehlo.add %v6438, %v6436 : tensor<64xf32>
    %v6440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6441 = stablehlo.multiply %v6440, %v6439 : tensor<64xf32>
    %v6442 = stablehlo.subtract %s1b2bt1, %v6441 : tensor<64xf32>
    %v6443 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6444 = stablehlo.multiply %v6443, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6445 = stablehlo.add %v6444, %v4677 : tensor<64x64x3x3xf32>
    %v6446 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6447 = stablehlo.multiply %v6446, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6448 = stablehlo.add %v6447, %v6445 : tensor<64x64x3x3xf32>
    %v6449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6450 = stablehlo.multiply %v6449, %v6448 : tensor<64x64x3x3xf32>
    %v6451 = stablehlo.subtract %s1b2W2, %v6450 : tensor<64x64x3x3xf32>
    %v6452 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6453 = stablehlo.multiply %v6452, %s1b2g2 : tensor<64xf32>
    %v6454 = stablehlo.add %v6453, %v4695 : tensor<64xf32>
    %v6455 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6456 = stablehlo.multiply %v6455, %s1b2g2v : tensor<64xf32>
    %v6457 = stablehlo.add %v6456, %v6454 : tensor<64xf32>
    %v6458 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6459 = stablehlo.multiply %v6458, %v6457 : tensor<64xf32>
    %v6460 = stablehlo.subtract %s1b2g2, %v6459 : tensor<64xf32>
    %v6461 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6462 = stablehlo.multiply %v6461, %s1b2bt2 : tensor<64xf32>
    %v6463 = stablehlo.add %v6462, %v4698 : tensor<64xf32>
    %v6464 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6465 = stablehlo.multiply %v6464, %s1b2bt2v : tensor<64xf32>
    %v6466 = stablehlo.add %v6465, %v6463 : tensor<64xf32>
    %v6467 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6468 = stablehlo.multiply %v6467, %v6466 : tensor<64xf32>
    %v6469 = stablehlo.subtract %s1b2bt2, %v6468 : tensor<64xf32>
    %v6470 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6471 = stablehlo.multiply %v6470, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6472 = stablehlo.add %v6471, %v4704 : tensor<256x64x1x1xf32>
    %v6473 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6474 = stablehlo.multiply %v6473, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6475 = stablehlo.add %v6474, %v6472 : tensor<256x64x1x1xf32>
    %v6476 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6477 = stablehlo.multiply %v6476, %v6475 : tensor<256x64x1x1xf32>
    %v6478 = stablehlo.subtract %s1b2W3, %v6477 : tensor<256x64x1x1xf32>
    %v6479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6480 = stablehlo.multiply %v6479, %s1b2g3 : tensor<256xf32>
    %v6481 = stablehlo.add %v6480, %v4722 : tensor<256xf32>
    %v6482 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6483 = stablehlo.multiply %v6482, %s1b2g3v : tensor<256xf32>
    %v6484 = stablehlo.add %v6483, %v6481 : tensor<256xf32>
    %v6485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6486 = stablehlo.multiply %v6485, %v6484 : tensor<256xf32>
    %v6487 = stablehlo.subtract %s1b2g3, %v6486 : tensor<256xf32>
    %v6488 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6489 = stablehlo.multiply %v6488, %s1b2bt3 : tensor<256xf32>
    %v6490 = stablehlo.add %v6489, %v4725 : tensor<256xf32>
    %v6491 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6492 = stablehlo.multiply %v6491, %s1b2bt3v : tensor<256xf32>
    %v6493 = stablehlo.add %v6492, %v6490 : tensor<256xf32>
    %v6494 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6495 = stablehlo.multiply %v6494, %v6493 : tensor<256xf32>
    %v6496 = stablehlo.subtract %s1b2bt3, %v6495 : tensor<256xf32>
    %v6497 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6498 = stablehlo.multiply %v6497, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6499 = stablehlo.add %v6498, %v4413 : tensor<128x256x1x1xf32>
    %v6500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6501 = stablehlo.multiply %v6500, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6502 = stablehlo.add %v6501, %v6499 : tensor<128x256x1x1xf32>
    %v6503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6504 = stablehlo.multiply %v6503, %v6502 : tensor<128x256x1x1xf32>
    %v6505 = stablehlo.subtract %s2b0W1, %v6504 : tensor<128x256x1x1xf32>
    %v6506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6507 = stablehlo.multiply %v6506, %s2b0g1 : tensor<128xf32>
    %v6508 = stablehlo.add %v6507, %v4431 : tensor<128xf32>
    %v6509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6510 = stablehlo.multiply %v6509, %s2b0g1v : tensor<128xf32>
    %v6511 = stablehlo.add %v6510, %v6508 : tensor<128xf32>
    %v6512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6513 = stablehlo.multiply %v6512, %v6511 : tensor<128xf32>
    %v6514 = stablehlo.subtract %s2b0g1, %v6513 : tensor<128xf32>
    %v6515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6516 = stablehlo.multiply %v6515, %s2b0bt1 : tensor<128xf32>
    %v6517 = stablehlo.add %v6516, %v4434 : tensor<128xf32>
    %v6518 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6519 = stablehlo.multiply %v6518, %s2b0bt1v : tensor<128xf32>
    %v6520 = stablehlo.add %v6519, %v6517 : tensor<128xf32>
    %v6521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6522 = stablehlo.multiply %v6521, %v6520 : tensor<128xf32>
    %v6523 = stablehlo.subtract %s2b0bt1, %v6522 : tensor<128xf32>
    %v6524 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6525 = stablehlo.multiply %v6524, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6526 = stablehlo.add %v6525, %v4441 : tensor<128x128x3x3xf32>
    %v6527 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6528 = stablehlo.multiply %v6527, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6529 = stablehlo.add %v6528, %v6526 : tensor<128x128x3x3xf32>
    %v6530 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6531 = stablehlo.multiply %v6530, %v6529 : tensor<128x128x3x3xf32>
    %v6532 = stablehlo.subtract %s2b0W2, %v6531 : tensor<128x128x3x3xf32>
    %v6533 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6534 = stablehlo.multiply %v6533, %s2b0g2 : tensor<128xf32>
    %v6535 = stablehlo.add %v6534, %v4459 : tensor<128xf32>
    %v6536 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6537 = stablehlo.multiply %v6536, %s2b0g2v : tensor<128xf32>
    %v6538 = stablehlo.add %v6537, %v6535 : tensor<128xf32>
    %v6539 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6540 = stablehlo.multiply %v6539, %v6538 : tensor<128xf32>
    %v6541 = stablehlo.subtract %s2b0g2, %v6540 : tensor<128xf32>
    %v6542 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6543 = stablehlo.multiply %v6542, %s2b0bt2 : tensor<128xf32>
    %v6544 = stablehlo.add %v6543, %v4462 : tensor<128xf32>
    %v6545 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6546 = stablehlo.multiply %v6545, %s2b0bt2v : tensor<128xf32>
    %v6547 = stablehlo.add %v6546, %v6544 : tensor<128xf32>
    %v6548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6549 = stablehlo.multiply %v6548, %v6547 : tensor<128xf32>
    %v6550 = stablehlo.subtract %s2b0bt2, %v6549 : tensor<128xf32>
    %v6551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6552 = stablehlo.multiply %v6551, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6553 = stablehlo.add %v6552, %v4468 : tensor<512x128x1x1xf32>
    %v6554 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6555 = stablehlo.multiply %v6554, %s2b0W3v : tensor<512x128x1x1xf32>
    %v6556 = stablehlo.add %v6555, %v6553 : tensor<512x128x1x1xf32>
    %v6557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6558 = stablehlo.multiply %v6557, %v6556 : tensor<512x128x1x1xf32>
    %v6559 = stablehlo.subtract %s2b0W3, %v6558 : tensor<512x128x1x1xf32>
    %v6560 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6561 = stablehlo.multiply %v6560, %s2b0g3 : tensor<512xf32>
    %v6562 = stablehlo.add %v6561, %v4486 : tensor<512xf32>
    %v6563 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6564 = stablehlo.multiply %v6563, %s2b0g3v : tensor<512xf32>
    %v6565 = stablehlo.add %v6564, %v6562 : tensor<512xf32>
    %v6566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6567 = stablehlo.multiply %v6566, %v6565 : tensor<512xf32>
    %v6568 = stablehlo.subtract %s2b0g3, %v6567 : tensor<512xf32>
    %v6569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6570 = stablehlo.multiply %v6569, %s2b0bt3 : tensor<512xf32>
    %v6571 = stablehlo.add %v6570, %v4489 : tensor<512xf32>
    %v6572 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6573 = stablehlo.multiply %v6572, %s2b0bt3v : tensor<512xf32>
    %v6574 = stablehlo.add %v6573, %v6571 : tensor<512xf32>
    %v6575 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6576 = stablehlo.multiply %v6575, %v6574 : tensor<512xf32>
    %v6577 = stablehlo.subtract %s2b0bt3, %v6576 : tensor<512xf32>
    %v6578 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6579 = stablehlo.multiply %v6578, %s2b0Wp : tensor<512x256x1x1xf32>
    %v6580 = stablehlo.add %v6579, %v4496 : tensor<512x256x1x1xf32>
    %v6581 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6582 = stablehlo.multiply %v6581, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v6583 = stablehlo.add %v6582, %v6580 : tensor<512x256x1x1xf32>
    %v6584 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6585 = stablehlo.multiply %v6584, %v6583 : tensor<512x256x1x1xf32>
    %v6586 = stablehlo.subtract %s2b0Wp, %v6585 : tensor<512x256x1x1xf32>
    %v6587 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6588 = stablehlo.multiply %v6587, %s2b0gp : tensor<512xf32>
    %v6589 = stablehlo.add %v6588, %v4514 : tensor<512xf32>
    %v6590 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6591 = stablehlo.multiply %v6590, %s2b0gpv : tensor<512xf32>
    %v6592 = stablehlo.add %v6591, %v6589 : tensor<512xf32>
    %v6593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6594 = stablehlo.multiply %v6593, %v6592 : tensor<512xf32>
    %v6595 = stablehlo.subtract %s2b0gp, %v6594 : tensor<512xf32>
    %v6596 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6597 = stablehlo.multiply %v6596, %s2b0btp : tensor<512xf32>
    %v6598 = stablehlo.add %v6597, %v4517 : tensor<512xf32>
    %v6599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6600 = stablehlo.multiply %v6599, %s2b0btpv : tensor<512xf32>
    %v6601 = stablehlo.add %v6600, %v6598 : tensor<512xf32>
    %v6602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6603 = stablehlo.multiply %v6602, %v6601 : tensor<512xf32>
    %v6604 = stablehlo.subtract %s2b0btp, %v6603 : tensor<512xf32>
    %v6605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6606 = stablehlo.multiply %v6605, %s2b1W1 : tensor<128x512x1x1xf32>
    %v6607 = stablehlo.add %v6606, %v4170 : tensor<128x512x1x1xf32>
    %v6608 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6609 = stablehlo.multiply %v6608, %s2b1W1v : tensor<128x512x1x1xf32>
    %v6610 = stablehlo.add %v6609, %v6607 : tensor<128x512x1x1xf32>
    %v6611 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6612 = stablehlo.multiply %v6611, %v6610 : tensor<128x512x1x1xf32>
    %v6613 = stablehlo.subtract %s2b1W1, %v6612 : tensor<128x512x1x1xf32>
    %v6614 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6615 = stablehlo.multiply %v6614, %s2b1g1 : tensor<128xf32>
    %v6616 = stablehlo.add %v6615, %v4188 : tensor<128xf32>
    %v6617 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6618 = stablehlo.multiply %v6617, %s2b1g1v : tensor<128xf32>
    %v6619 = stablehlo.add %v6618, %v6616 : tensor<128xf32>
    %v6620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6621 = stablehlo.multiply %v6620, %v6619 : tensor<128xf32>
    %v6622 = stablehlo.subtract %s2b1g1, %v6621 : tensor<128xf32>
    %v6623 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6624 = stablehlo.multiply %v6623, %s2b1bt1 : tensor<128xf32>
    %v6625 = stablehlo.add %v6624, %v4191 : tensor<128xf32>
    %v6626 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6627 = stablehlo.multiply %v6626, %s2b1bt1v : tensor<128xf32>
    %v6628 = stablehlo.add %v6627, %v6625 : tensor<128xf32>
    %v6629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6630 = stablehlo.multiply %v6629, %v6628 : tensor<128xf32>
    %v6631 = stablehlo.subtract %s2b1bt1, %v6630 : tensor<128xf32>
    %v6632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6633 = stablehlo.multiply %v6632, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6634 = stablehlo.add %v6633, %v4197 : tensor<128x128x3x3xf32>
    %v6635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6636 = stablehlo.multiply %v6635, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6637 = stablehlo.add %v6636, %v6634 : tensor<128x128x3x3xf32>
    %v6638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6639 = stablehlo.multiply %v6638, %v6637 : tensor<128x128x3x3xf32>
    %v6640 = stablehlo.subtract %s2b1W2, %v6639 : tensor<128x128x3x3xf32>
    %v6641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6642 = stablehlo.multiply %v6641, %s2b1g2 : tensor<128xf32>
    %v6643 = stablehlo.add %v6642, %v4215 : tensor<128xf32>
    %v6644 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6645 = stablehlo.multiply %v6644, %s2b1g2v : tensor<128xf32>
    %v6646 = stablehlo.add %v6645, %v6643 : tensor<128xf32>
    %v6647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6648 = stablehlo.multiply %v6647, %v6646 : tensor<128xf32>
    %v6649 = stablehlo.subtract %s2b1g2, %v6648 : tensor<128xf32>
    %v6650 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6651 = stablehlo.multiply %v6650, %s2b1bt2 : tensor<128xf32>
    %v6652 = stablehlo.add %v6651, %v4218 : tensor<128xf32>
    %v6653 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6654 = stablehlo.multiply %v6653, %s2b1bt2v : tensor<128xf32>
    %v6655 = stablehlo.add %v6654, %v6652 : tensor<128xf32>
    %v6656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6657 = stablehlo.multiply %v6656, %v6655 : tensor<128xf32>
    %v6658 = stablehlo.subtract %s2b1bt2, %v6657 : tensor<128xf32>
    %v6659 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6660 = stablehlo.multiply %v6659, %s2b1W3 : tensor<512x128x1x1xf32>
    %v6661 = stablehlo.add %v6660, %v4224 : tensor<512x128x1x1xf32>
    %v6662 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6663 = stablehlo.multiply %v6662, %s2b1W3v : tensor<512x128x1x1xf32>
    %v6664 = stablehlo.add %v6663, %v6661 : tensor<512x128x1x1xf32>
    %v6665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6666 = stablehlo.multiply %v6665, %v6664 : tensor<512x128x1x1xf32>
    %v6667 = stablehlo.subtract %s2b1W3, %v6666 : tensor<512x128x1x1xf32>
    %v6668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6669 = stablehlo.multiply %v6668, %s2b1g3 : tensor<512xf32>
    %v6670 = stablehlo.add %v6669, %v4242 : tensor<512xf32>
    %v6671 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6672 = stablehlo.multiply %v6671, %s2b1g3v : tensor<512xf32>
    %v6673 = stablehlo.add %v6672, %v6670 : tensor<512xf32>
    %v6674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6675 = stablehlo.multiply %v6674, %v6673 : tensor<512xf32>
    %v6676 = stablehlo.subtract %s2b1g3, %v6675 : tensor<512xf32>
    %v6677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6678 = stablehlo.multiply %v6677, %s2b1bt3 : tensor<512xf32>
    %v6679 = stablehlo.add %v6678, %v4245 : tensor<512xf32>
    %v6680 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6681 = stablehlo.multiply %v6680, %s2b1bt3v : tensor<512xf32>
    %v6682 = stablehlo.add %v6681, %v6679 : tensor<512xf32>
    %v6683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6684 = stablehlo.multiply %v6683, %v6682 : tensor<512xf32>
    %v6685 = stablehlo.subtract %s2b1bt3, %v6684 : tensor<512xf32>
    %v6686 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6687 = stablehlo.multiply %v6686, %s2b2W1 : tensor<128x512x1x1xf32>
    %v6688 = stablehlo.add %v6687, %v3962 : tensor<128x512x1x1xf32>
    %v6689 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6690 = stablehlo.multiply %v6689, %s2b2W1v : tensor<128x512x1x1xf32>
    %v6691 = stablehlo.add %v6690, %v6688 : tensor<128x512x1x1xf32>
    %v6692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6693 = stablehlo.multiply %v6692, %v6691 : tensor<128x512x1x1xf32>
    %v6694 = stablehlo.subtract %s2b2W1, %v6693 : tensor<128x512x1x1xf32>
    %v6695 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6696 = stablehlo.multiply %v6695, %s2b2g1 : tensor<128xf32>
    %v6697 = stablehlo.add %v6696, %v3980 : tensor<128xf32>
    %v6698 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6699 = stablehlo.multiply %v6698, %s2b2g1v : tensor<128xf32>
    %v6700 = stablehlo.add %v6699, %v6697 : tensor<128xf32>
    %v6701 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6702 = stablehlo.multiply %v6701, %v6700 : tensor<128xf32>
    %v6703 = stablehlo.subtract %s2b2g1, %v6702 : tensor<128xf32>
    %v6704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6705 = stablehlo.multiply %v6704, %s2b2bt1 : tensor<128xf32>
    %v6706 = stablehlo.add %v6705, %v3983 : tensor<128xf32>
    %v6707 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6708 = stablehlo.multiply %v6707, %s2b2bt1v : tensor<128xf32>
    %v6709 = stablehlo.add %v6708, %v6706 : tensor<128xf32>
    %v6710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6711 = stablehlo.multiply %v6710, %v6709 : tensor<128xf32>
    %v6712 = stablehlo.subtract %s2b2bt1, %v6711 : tensor<128xf32>
    %v6713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6714 = stablehlo.multiply %v6713, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6715 = stablehlo.add %v6714, %v3989 : tensor<128x128x3x3xf32>
    %v6716 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6717 = stablehlo.multiply %v6716, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6718 = stablehlo.add %v6717, %v6715 : tensor<128x128x3x3xf32>
    %v6719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6720 = stablehlo.multiply %v6719, %v6718 : tensor<128x128x3x3xf32>
    %v6721 = stablehlo.subtract %s2b2W2, %v6720 : tensor<128x128x3x3xf32>
    %v6722 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6723 = stablehlo.multiply %v6722, %s2b2g2 : tensor<128xf32>
    %v6724 = stablehlo.add %v6723, %v4007 : tensor<128xf32>
    %v6725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6726 = stablehlo.multiply %v6725, %s2b2g2v : tensor<128xf32>
    %v6727 = stablehlo.add %v6726, %v6724 : tensor<128xf32>
    %v6728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6729 = stablehlo.multiply %v6728, %v6727 : tensor<128xf32>
    %v6730 = stablehlo.subtract %s2b2g2, %v6729 : tensor<128xf32>
    %v6731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6732 = stablehlo.multiply %v6731, %s2b2bt2 : tensor<128xf32>
    %v6733 = stablehlo.add %v6732, %v4010 : tensor<128xf32>
    %v6734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6735 = stablehlo.multiply %v6734, %s2b2bt2v : tensor<128xf32>
    %v6736 = stablehlo.add %v6735, %v6733 : tensor<128xf32>
    %v6737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6738 = stablehlo.multiply %v6737, %v6736 : tensor<128xf32>
    %v6739 = stablehlo.subtract %s2b2bt2, %v6738 : tensor<128xf32>
    %v6740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6741 = stablehlo.multiply %v6740, %s2b2W3 : tensor<512x128x1x1xf32>
    %v6742 = stablehlo.add %v6741, %v4016 : tensor<512x128x1x1xf32>
    %v6743 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6744 = stablehlo.multiply %v6743, %s2b2W3v : tensor<512x128x1x1xf32>
    %v6745 = stablehlo.add %v6744, %v6742 : tensor<512x128x1x1xf32>
    %v6746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6747 = stablehlo.multiply %v6746, %v6745 : tensor<512x128x1x1xf32>
    %v6748 = stablehlo.subtract %s2b2W3, %v6747 : tensor<512x128x1x1xf32>
    %v6749 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6750 = stablehlo.multiply %v6749, %s2b2g3 : tensor<512xf32>
    %v6751 = stablehlo.add %v6750, %v4034 : tensor<512xf32>
    %v6752 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6753 = stablehlo.multiply %v6752, %s2b2g3v : tensor<512xf32>
    %v6754 = stablehlo.add %v6753, %v6751 : tensor<512xf32>
    %v6755 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6756 = stablehlo.multiply %v6755, %v6754 : tensor<512xf32>
    %v6757 = stablehlo.subtract %s2b2g3, %v6756 : tensor<512xf32>
    %v6758 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6759 = stablehlo.multiply %v6758, %s2b2bt3 : tensor<512xf32>
    %v6760 = stablehlo.add %v6759, %v4037 : tensor<512xf32>
    %v6761 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6762 = stablehlo.multiply %v6761, %s2b2bt3v : tensor<512xf32>
    %v6763 = stablehlo.add %v6762, %v6760 : tensor<512xf32>
    %v6764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6765 = stablehlo.multiply %v6764, %v6763 : tensor<512xf32>
    %v6766 = stablehlo.subtract %s2b2bt3, %v6765 : tensor<512xf32>
    %v6767 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6768 = stablehlo.multiply %v6767, %s2b3W1 : tensor<128x512x1x1xf32>
    %v6769 = stablehlo.add %v6768, %v3754 : tensor<128x512x1x1xf32>
    %v6770 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6771 = stablehlo.multiply %v6770, %s2b3W1v : tensor<128x512x1x1xf32>
    %v6772 = stablehlo.add %v6771, %v6769 : tensor<128x512x1x1xf32>
    %v6773 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6774 = stablehlo.multiply %v6773, %v6772 : tensor<128x512x1x1xf32>
    %v6775 = stablehlo.subtract %s2b3W1, %v6774 : tensor<128x512x1x1xf32>
    %v6776 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6777 = stablehlo.multiply %v6776, %s2b3g1 : tensor<128xf32>
    %v6778 = stablehlo.add %v6777, %v3772 : tensor<128xf32>
    %v6779 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6780 = stablehlo.multiply %v6779, %s2b3g1v : tensor<128xf32>
    %v6781 = stablehlo.add %v6780, %v6778 : tensor<128xf32>
    %v6782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6783 = stablehlo.multiply %v6782, %v6781 : tensor<128xf32>
    %v6784 = stablehlo.subtract %s2b3g1, %v6783 : tensor<128xf32>
    %v6785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6786 = stablehlo.multiply %v6785, %s2b3bt1 : tensor<128xf32>
    %v6787 = stablehlo.add %v6786, %v3775 : tensor<128xf32>
    %v6788 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6789 = stablehlo.multiply %v6788, %s2b3bt1v : tensor<128xf32>
    %v6790 = stablehlo.add %v6789, %v6787 : tensor<128xf32>
    %v6791 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6792 = stablehlo.multiply %v6791, %v6790 : tensor<128xf32>
    %v6793 = stablehlo.subtract %s2b3bt1, %v6792 : tensor<128xf32>
    %v6794 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6795 = stablehlo.multiply %v6794, %s2b3W2 : tensor<128x128x3x3xf32>
    %v6796 = stablehlo.add %v6795, %v3781 : tensor<128x128x3x3xf32>
    %v6797 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6798 = stablehlo.multiply %v6797, %s2b3W2v : tensor<128x128x3x3xf32>
    %v6799 = stablehlo.add %v6798, %v6796 : tensor<128x128x3x3xf32>
    %v6800 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6801 = stablehlo.multiply %v6800, %v6799 : tensor<128x128x3x3xf32>
    %v6802 = stablehlo.subtract %s2b3W2, %v6801 : tensor<128x128x3x3xf32>
    %v6803 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6804 = stablehlo.multiply %v6803, %s2b3g2 : tensor<128xf32>
    %v6805 = stablehlo.add %v6804, %v3799 : tensor<128xf32>
    %v6806 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6807 = stablehlo.multiply %v6806, %s2b3g2v : tensor<128xf32>
    %v6808 = stablehlo.add %v6807, %v6805 : tensor<128xf32>
    %v6809 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6810 = stablehlo.multiply %v6809, %v6808 : tensor<128xf32>
    %v6811 = stablehlo.subtract %s2b3g2, %v6810 : tensor<128xf32>
    %v6812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6813 = stablehlo.multiply %v6812, %s2b3bt2 : tensor<128xf32>
    %v6814 = stablehlo.add %v6813, %v3802 : tensor<128xf32>
    %v6815 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6816 = stablehlo.multiply %v6815, %s2b3bt2v : tensor<128xf32>
    %v6817 = stablehlo.add %v6816, %v6814 : tensor<128xf32>
    %v6818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6819 = stablehlo.multiply %v6818, %v6817 : tensor<128xf32>
    %v6820 = stablehlo.subtract %s2b3bt2, %v6819 : tensor<128xf32>
    %v6821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6822 = stablehlo.multiply %v6821, %s2b3W3 : tensor<512x128x1x1xf32>
    %v6823 = stablehlo.add %v6822, %v3808 : tensor<512x128x1x1xf32>
    %v6824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6825 = stablehlo.multiply %v6824, %s2b3W3v : tensor<512x128x1x1xf32>
    %v6826 = stablehlo.add %v6825, %v6823 : tensor<512x128x1x1xf32>
    %v6827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6828 = stablehlo.multiply %v6827, %v6826 : tensor<512x128x1x1xf32>
    %v6829 = stablehlo.subtract %s2b3W3, %v6828 : tensor<512x128x1x1xf32>
    %v6830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6831 = stablehlo.multiply %v6830, %s2b3g3 : tensor<512xf32>
    %v6832 = stablehlo.add %v6831, %v3826 : tensor<512xf32>
    %v6833 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6834 = stablehlo.multiply %v6833, %s2b3g3v : tensor<512xf32>
    %v6835 = stablehlo.add %v6834, %v6832 : tensor<512xf32>
    %v6836 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6837 = stablehlo.multiply %v6836, %v6835 : tensor<512xf32>
    %v6838 = stablehlo.subtract %s2b3g3, %v6837 : tensor<512xf32>
    %v6839 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6840 = stablehlo.multiply %v6839, %s2b3bt3 : tensor<512xf32>
    %v6841 = stablehlo.add %v6840, %v3829 : tensor<512xf32>
    %v6842 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6843 = stablehlo.multiply %v6842, %s2b3bt3v : tensor<512xf32>
    %v6844 = stablehlo.add %v6843, %v6841 : tensor<512xf32>
    %v6845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6846 = stablehlo.multiply %v6845, %v6844 : tensor<512xf32>
    %v6847 = stablehlo.subtract %s2b3bt3, %v6846 : tensor<512xf32>
    %v6848 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6849 = stablehlo.multiply %v6848, %s3b0W1 : tensor<256x512x1x1xf32>
    %v6850 = stablehlo.add %v6849, %v3517 : tensor<256x512x1x1xf32>
    %v6851 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6852 = stablehlo.multiply %v6851, %s3b0W1v : tensor<256x512x1x1xf32>
    %v6853 = stablehlo.add %v6852, %v6850 : tensor<256x512x1x1xf32>
    %v6854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6855 = stablehlo.multiply %v6854, %v6853 : tensor<256x512x1x1xf32>
    %v6856 = stablehlo.subtract %s3b0W1, %v6855 : tensor<256x512x1x1xf32>
    %v6857 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.multiply %v6857, %s3b0g1 : tensor<256xf32>
    %v6859 = stablehlo.add %v6858, %v3535 : tensor<256xf32>
    %v6860 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6861 = stablehlo.multiply %v6860, %s3b0g1v : tensor<256xf32>
    %v6862 = stablehlo.add %v6861, %v6859 : tensor<256xf32>
    %v6863 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6864 = stablehlo.multiply %v6863, %v6862 : tensor<256xf32>
    %v6865 = stablehlo.subtract %s3b0g1, %v6864 : tensor<256xf32>
    %v6866 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6867 = stablehlo.multiply %v6866, %s3b0bt1 : tensor<256xf32>
    %v6868 = stablehlo.add %v6867, %v3538 : tensor<256xf32>
    %v6869 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6870 = stablehlo.multiply %v6869, %s3b0bt1v : tensor<256xf32>
    %v6871 = stablehlo.add %v6870, %v6868 : tensor<256xf32>
    %v6872 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6873 = stablehlo.multiply %v6872, %v6871 : tensor<256xf32>
    %v6874 = stablehlo.subtract %s3b0bt1, %v6873 : tensor<256xf32>
    %v6875 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6876 = stablehlo.multiply %v6875, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6877 = stablehlo.add %v6876, %v3545 : tensor<256x256x3x3xf32>
    %v6878 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6879 = stablehlo.multiply %v6878, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6880 = stablehlo.add %v6879, %v6877 : tensor<256x256x3x3xf32>
    %v6881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6882 = stablehlo.multiply %v6881, %v6880 : tensor<256x256x3x3xf32>
    %v6883 = stablehlo.subtract %s3b0W2, %v6882 : tensor<256x256x3x3xf32>
    %v6884 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6885 = stablehlo.multiply %v6884, %s3b0g2 : tensor<256xf32>
    %v6886 = stablehlo.add %v6885, %v3563 : tensor<256xf32>
    %v6887 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6888 = stablehlo.multiply %v6887, %s3b0g2v : tensor<256xf32>
    %v6889 = stablehlo.add %v6888, %v6886 : tensor<256xf32>
    %v6890 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6891 = stablehlo.multiply %v6890, %v6889 : tensor<256xf32>
    %v6892 = stablehlo.subtract %s3b0g2, %v6891 : tensor<256xf32>
    %v6893 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6894 = stablehlo.multiply %v6893, %s3b0bt2 : tensor<256xf32>
    %v6895 = stablehlo.add %v6894, %v3566 : tensor<256xf32>
    %v6896 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6897 = stablehlo.multiply %v6896, %s3b0bt2v : tensor<256xf32>
    %v6898 = stablehlo.add %v6897, %v6895 : tensor<256xf32>
    %v6899 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6900 = stablehlo.multiply %v6899, %v6898 : tensor<256xf32>
    %v6901 = stablehlo.subtract %s3b0bt2, %v6900 : tensor<256xf32>
    %v6902 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6903 = stablehlo.multiply %v6902, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v6904 = stablehlo.add %v6903, %v3572 : tensor<1024x256x1x1xf32>
    %v6905 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6906 = stablehlo.multiply %v6905, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v6907 = stablehlo.add %v6906, %v6904 : tensor<1024x256x1x1xf32>
    %v6908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6909 = stablehlo.multiply %v6908, %v6907 : tensor<1024x256x1x1xf32>
    %v6910 = stablehlo.subtract %s3b0W3, %v6909 : tensor<1024x256x1x1xf32>
    %v6911 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6912 = stablehlo.multiply %v6911, %s3b0g3 : tensor<1024xf32>
    %v6913 = stablehlo.add %v6912, %v3590 : tensor<1024xf32>
    %v6914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6915 = stablehlo.multiply %v6914, %s3b0g3v : tensor<1024xf32>
    %v6916 = stablehlo.add %v6915, %v6913 : tensor<1024xf32>
    %v6917 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6918 = stablehlo.multiply %v6917, %v6916 : tensor<1024xf32>
    %v6919 = stablehlo.subtract %s3b0g3, %v6918 : tensor<1024xf32>
    %v6920 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6921 = stablehlo.multiply %v6920, %s3b0bt3 : tensor<1024xf32>
    %v6922 = stablehlo.add %v6921, %v3593 : tensor<1024xf32>
    %v6923 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6924 = stablehlo.multiply %v6923, %s3b0bt3v : tensor<1024xf32>
    %v6925 = stablehlo.add %v6924, %v6922 : tensor<1024xf32>
    %v6926 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6927 = stablehlo.multiply %v6926, %v6925 : tensor<1024xf32>
    %v6928 = stablehlo.subtract %s3b0bt3, %v6927 : tensor<1024xf32>
    %v6929 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6930 = stablehlo.multiply %v6929, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v6931 = stablehlo.add %v6930, %v3600 : tensor<1024x512x1x1xf32>
    %v6932 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6933 = stablehlo.multiply %v6932, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v6934 = stablehlo.add %v6933, %v6931 : tensor<1024x512x1x1xf32>
    %v6935 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6936 = stablehlo.multiply %v6935, %v6934 : tensor<1024x512x1x1xf32>
    %v6937 = stablehlo.subtract %s3b0Wp, %v6936 : tensor<1024x512x1x1xf32>
    %v6938 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6939 = stablehlo.multiply %v6938, %s3b0gp : tensor<1024xf32>
    %v6940 = stablehlo.add %v6939, %v3618 : tensor<1024xf32>
    %v6941 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6942 = stablehlo.multiply %v6941, %s3b0gpv : tensor<1024xf32>
    %v6943 = stablehlo.add %v6942, %v6940 : tensor<1024xf32>
    %v6944 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6945 = stablehlo.multiply %v6944, %v6943 : tensor<1024xf32>
    %v6946 = stablehlo.subtract %s3b0gp, %v6945 : tensor<1024xf32>
    %v6947 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6948 = stablehlo.multiply %v6947, %s3b0btp : tensor<1024xf32>
    %v6949 = stablehlo.add %v6948, %v3621 : tensor<1024xf32>
    %v6950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6951 = stablehlo.multiply %v6950, %s3b0btpv : tensor<1024xf32>
    %v6952 = stablehlo.add %v6951, %v6949 : tensor<1024xf32>
    %v6953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6954 = stablehlo.multiply %v6953, %v6952 : tensor<1024xf32>
    %v6955 = stablehlo.subtract %s3b0btp, %v6954 : tensor<1024xf32>
    %v6956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6957 = stablehlo.multiply %v6956, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v6958 = stablehlo.add %v6957, %v3274 : tensor<256x1024x1x1xf32>
    %v6959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6960 = stablehlo.multiply %v6959, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v6961 = stablehlo.add %v6960, %v6958 : tensor<256x1024x1x1xf32>
    %v6962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6963 = stablehlo.multiply %v6962, %v6961 : tensor<256x1024x1x1xf32>
    %v6964 = stablehlo.subtract %s3b1W1, %v6963 : tensor<256x1024x1x1xf32>
    %v6965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6966 = stablehlo.multiply %v6965, %s3b1g1 : tensor<256xf32>
    %v6967 = stablehlo.add %v6966, %v3292 : tensor<256xf32>
    %v6968 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6969 = stablehlo.multiply %v6968, %s3b1g1v : tensor<256xf32>
    %v6970 = stablehlo.add %v6969, %v6967 : tensor<256xf32>
    %v6971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6972 = stablehlo.multiply %v6971, %v6970 : tensor<256xf32>
    %v6973 = stablehlo.subtract %s3b1g1, %v6972 : tensor<256xf32>
    %v6974 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6975 = stablehlo.multiply %v6974, %s3b1bt1 : tensor<256xf32>
    %v6976 = stablehlo.add %v6975, %v3295 : tensor<256xf32>
    %v6977 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6978 = stablehlo.multiply %v6977, %s3b1bt1v : tensor<256xf32>
    %v6979 = stablehlo.add %v6978, %v6976 : tensor<256xf32>
    %v6980 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6981 = stablehlo.multiply %v6980, %v6979 : tensor<256xf32>
    %v6982 = stablehlo.subtract %s3b1bt1, %v6981 : tensor<256xf32>
    %v6983 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6984 = stablehlo.multiply %v6983, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6985 = stablehlo.add %v6984, %v3301 : tensor<256x256x3x3xf32>
    %v6986 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6987 = stablehlo.multiply %v6986, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6988 = stablehlo.add %v6987, %v6985 : tensor<256x256x3x3xf32>
    %v6989 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6990 = stablehlo.multiply %v6989, %v6988 : tensor<256x256x3x3xf32>
    %v6991 = stablehlo.subtract %s3b1W2, %v6990 : tensor<256x256x3x3xf32>
    %v6992 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6993 = stablehlo.multiply %v6992, %s3b1g2 : tensor<256xf32>
    %v6994 = stablehlo.add %v6993, %v3319 : tensor<256xf32>
    %v6995 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6996 = stablehlo.multiply %v6995, %s3b1g2v : tensor<256xf32>
    %v6997 = stablehlo.add %v6996, %v6994 : tensor<256xf32>
    %v6998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6999 = stablehlo.multiply %v6998, %v6997 : tensor<256xf32>
    %v7000 = stablehlo.subtract %s3b1g2, %v6999 : tensor<256xf32>
    %v7001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7002 = stablehlo.multiply %v7001, %s3b1bt2 : tensor<256xf32>
    %v7003 = stablehlo.add %v7002, %v3322 : tensor<256xf32>
    %v7004 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7005 = stablehlo.multiply %v7004, %s3b1bt2v : tensor<256xf32>
    %v7006 = stablehlo.add %v7005, %v7003 : tensor<256xf32>
    %v7007 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7008 = stablehlo.multiply %v7007, %v7006 : tensor<256xf32>
    %v7009 = stablehlo.subtract %s3b1bt2, %v7008 : tensor<256xf32>
    %v7010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7011 = stablehlo.multiply %v7010, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v7012 = stablehlo.add %v7011, %v3328 : tensor<1024x256x1x1xf32>
    %v7013 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7014 = stablehlo.multiply %v7013, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v7015 = stablehlo.add %v7014, %v7012 : tensor<1024x256x1x1xf32>
    %v7016 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7017 = stablehlo.multiply %v7016, %v7015 : tensor<1024x256x1x1xf32>
    %v7018 = stablehlo.subtract %s3b1W3, %v7017 : tensor<1024x256x1x1xf32>
    %v7019 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7020 = stablehlo.multiply %v7019, %s3b1g3 : tensor<1024xf32>
    %v7021 = stablehlo.add %v7020, %v3346 : tensor<1024xf32>
    %v7022 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7023 = stablehlo.multiply %v7022, %s3b1g3v : tensor<1024xf32>
    %v7024 = stablehlo.add %v7023, %v7021 : tensor<1024xf32>
    %v7025 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7026 = stablehlo.multiply %v7025, %v7024 : tensor<1024xf32>
    %v7027 = stablehlo.subtract %s3b1g3, %v7026 : tensor<1024xf32>
    %v7028 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7029 = stablehlo.multiply %v7028, %s3b1bt3 : tensor<1024xf32>
    %v7030 = stablehlo.add %v7029, %v3349 : tensor<1024xf32>
    %v7031 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7032 = stablehlo.multiply %v7031, %s3b1bt3v : tensor<1024xf32>
    %v7033 = stablehlo.add %v7032, %v7030 : tensor<1024xf32>
    %v7034 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7035 = stablehlo.multiply %v7034, %v7033 : tensor<1024xf32>
    %v7036 = stablehlo.subtract %s3b1bt3, %v7035 : tensor<1024xf32>
    %v7037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7038 = stablehlo.multiply %v7037, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v7039 = stablehlo.add %v7038, %v3066 : tensor<256x1024x1x1xf32>
    %v7040 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7041 = stablehlo.multiply %v7040, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v7042 = stablehlo.add %v7041, %v7039 : tensor<256x1024x1x1xf32>
    %v7043 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7044 = stablehlo.multiply %v7043, %v7042 : tensor<256x1024x1x1xf32>
    %v7045 = stablehlo.subtract %s3b2W1, %v7044 : tensor<256x1024x1x1xf32>
    %v7046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7047 = stablehlo.multiply %v7046, %s3b2g1 : tensor<256xf32>
    %v7048 = stablehlo.add %v7047, %v3084 : tensor<256xf32>
    %v7049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7050 = stablehlo.multiply %v7049, %s3b2g1v : tensor<256xf32>
    %v7051 = stablehlo.add %v7050, %v7048 : tensor<256xf32>
    %v7052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7053 = stablehlo.multiply %v7052, %v7051 : tensor<256xf32>
    %v7054 = stablehlo.subtract %s3b2g1, %v7053 : tensor<256xf32>
    %v7055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7056 = stablehlo.multiply %v7055, %s3b2bt1 : tensor<256xf32>
    %v7057 = stablehlo.add %v7056, %v3087 : tensor<256xf32>
    %v7058 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7059 = stablehlo.multiply %v7058, %s3b2bt1v : tensor<256xf32>
    %v7060 = stablehlo.add %v7059, %v7057 : tensor<256xf32>
    %v7061 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7062 = stablehlo.multiply %v7061, %v7060 : tensor<256xf32>
    %v7063 = stablehlo.subtract %s3b2bt1, %v7062 : tensor<256xf32>
    %v7064 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7065 = stablehlo.multiply %v7064, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7066 = stablehlo.add %v7065, %v3093 : tensor<256x256x3x3xf32>
    %v7067 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7068 = stablehlo.multiply %v7067, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7069 = stablehlo.add %v7068, %v7066 : tensor<256x256x3x3xf32>
    %v7070 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7071 = stablehlo.multiply %v7070, %v7069 : tensor<256x256x3x3xf32>
    %v7072 = stablehlo.subtract %s3b2W2, %v7071 : tensor<256x256x3x3xf32>
    %v7073 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7074 = stablehlo.multiply %v7073, %s3b2g2 : tensor<256xf32>
    %v7075 = stablehlo.add %v7074, %v3111 : tensor<256xf32>
    %v7076 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7077 = stablehlo.multiply %v7076, %s3b2g2v : tensor<256xf32>
    %v7078 = stablehlo.add %v7077, %v7075 : tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.multiply %v7079, %v7078 : tensor<256xf32>
    %v7081 = stablehlo.subtract %s3b2g2, %v7080 : tensor<256xf32>
    %v7082 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7083 = stablehlo.multiply %v7082, %s3b2bt2 : tensor<256xf32>
    %v7084 = stablehlo.add %v7083, %v3114 : tensor<256xf32>
    %v7085 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7086 = stablehlo.multiply %v7085, %s3b2bt2v : tensor<256xf32>
    %v7087 = stablehlo.add %v7086, %v7084 : tensor<256xf32>
    %v7088 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7089 = stablehlo.multiply %v7088, %v7087 : tensor<256xf32>
    %v7090 = stablehlo.subtract %s3b2bt2, %v7089 : tensor<256xf32>
    %v7091 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7092 = stablehlo.multiply %v7091, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7093 = stablehlo.add %v7092, %v3120 : tensor<1024x256x1x1xf32>
    %v7094 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7095 = stablehlo.multiply %v7094, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7096 = stablehlo.add %v7095, %v7093 : tensor<1024x256x1x1xf32>
    %v7097 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7098 = stablehlo.multiply %v7097, %v7096 : tensor<1024x256x1x1xf32>
    %v7099 = stablehlo.subtract %s3b2W3, %v7098 : tensor<1024x256x1x1xf32>
    %v7100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7101 = stablehlo.multiply %v7100, %s3b2g3 : tensor<1024xf32>
    %v7102 = stablehlo.add %v7101, %v3138 : tensor<1024xf32>
    %v7103 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7104 = stablehlo.multiply %v7103, %s3b2g3v : tensor<1024xf32>
    %v7105 = stablehlo.add %v7104, %v7102 : tensor<1024xf32>
    %v7106 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7107 = stablehlo.multiply %v7106, %v7105 : tensor<1024xf32>
    %v7108 = stablehlo.subtract %s3b2g3, %v7107 : tensor<1024xf32>
    %v7109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7110 = stablehlo.multiply %v7109, %s3b2bt3 : tensor<1024xf32>
    %v7111 = stablehlo.add %v7110, %v3141 : tensor<1024xf32>
    %v7112 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7113 = stablehlo.multiply %v7112, %s3b2bt3v : tensor<1024xf32>
    %v7114 = stablehlo.add %v7113, %v7111 : tensor<1024xf32>
    %v7115 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7116 = stablehlo.multiply %v7115, %v7114 : tensor<1024xf32>
    %v7117 = stablehlo.subtract %s3b2bt3, %v7116 : tensor<1024xf32>
    %v7118 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7119 = stablehlo.multiply %v7118, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7120 = stablehlo.add %v7119, %v2858 : tensor<256x1024x1x1xf32>
    %v7121 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7122 = stablehlo.multiply %v7121, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7123 = stablehlo.add %v7122, %v7120 : tensor<256x1024x1x1xf32>
    %v7124 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7125 = stablehlo.multiply %v7124, %v7123 : tensor<256x1024x1x1xf32>
    %v7126 = stablehlo.subtract %s3b3W1, %v7125 : tensor<256x1024x1x1xf32>
    %v7127 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7128 = stablehlo.multiply %v7127, %s3b3g1 : tensor<256xf32>
    %v7129 = stablehlo.add %v7128, %v2876 : tensor<256xf32>
    %v7130 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7131 = stablehlo.multiply %v7130, %s3b3g1v : tensor<256xf32>
    %v7132 = stablehlo.add %v7131, %v7129 : tensor<256xf32>
    %v7133 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7134 = stablehlo.multiply %v7133, %v7132 : tensor<256xf32>
    %v7135 = stablehlo.subtract %s3b3g1, %v7134 : tensor<256xf32>
    %v7136 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7137 = stablehlo.multiply %v7136, %s3b3bt1 : tensor<256xf32>
    %v7138 = stablehlo.add %v7137, %v2879 : tensor<256xf32>
    %v7139 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7140 = stablehlo.multiply %v7139, %s3b3bt1v : tensor<256xf32>
    %v7141 = stablehlo.add %v7140, %v7138 : tensor<256xf32>
    %v7142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7143 = stablehlo.multiply %v7142, %v7141 : tensor<256xf32>
    %v7144 = stablehlo.subtract %s3b3bt1, %v7143 : tensor<256xf32>
    %v7145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7146 = stablehlo.multiply %v7145, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7147 = stablehlo.add %v7146, %v2885 : tensor<256x256x3x3xf32>
    %v7148 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7149 = stablehlo.multiply %v7148, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7150 = stablehlo.add %v7149, %v7147 : tensor<256x256x3x3xf32>
    %v7151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7152 = stablehlo.multiply %v7151, %v7150 : tensor<256x256x3x3xf32>
    %v7153 = stablehlo.subtract %s3b3W2, %v7152 : tensor<256x256x3x3xf32>
    %v7154 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7155 = stablehlo.multiply %v7154, %s3b3g2 : tensor<256xf32>
    %v7156 = stablehlo.add %v7155, %v2903 : tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.multiply %v7157, %s3b3g2v : tensor<256xf32>
    %v7159 = stablehlo.add %v7158, %v7156 : tensor<256xf32>
    %v7160 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7161 = stablehlo.multiply %v7160, %v7159 : tensor<256xf32>
    %v7162 = stablehlo.subtract %s3b3g2, %v7161 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.multiply %v7163, %s3b3bt2 : tensor<256xf32>
    %v7165 = stablehlo.add %v7164, %v2906 : tensor<256xf32>
    %v7166 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7167 = stablehlo.multiply %v7166, %s3b3bt2v : tensor<256xf32>
    %v7168 = stablehlo.add %v7167, %v7165 : tensor<256xf32>
    %v7169 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7170 = stablehlo.multiply %v7169, %v7168 : tensor<256xf32>
    %v7171 = stablehlo.subtract %s3b3bt2, %v7170 : tensor<256xf32>
    %v7172 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7173 = stablehlo.multiply %v7172, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v7174 = stablehlo.add %v7173, %v2912 : tensor<1024x256x1x1xf32>
    %v7175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7176 = stablehlo.multiply %v7175, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v7177 = stablehlo.add %v7176, %v7174 : tensor<1024x256x1x1xf32>
    %v7178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7179 = stablehlo.multiply %v7178, %v7177 : tensor<1024x256x1x1xf32>
    %v7180 = stablehlo.subtract %s3b3W3, %v7179 : tensor<1024x256x1x1xf32>
    %v7181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7182 = stablehlo.multiply %v7181, %s3b3g3 : tensor<1024xf32>
    %v7183 = stablehlo.add %v7182, %v2930 : tensor<1024xf32>
    %v7184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7185 = stablehlo.multiply %v7184, %s3b3g3v : tensor<1024xf32>
    %v7186 = stablehlo.add %v7185, %v7183 : tensor<1024xf32>
    %v7187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7188 = stablehlo.multiply %v7187, %v7186 : tensor<1024xf32>
    %v7189 = stablehlo.subtract %s3b3g3, %v7188 : tensor<1024xf32>
    %v7190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7191 = stablehlo.multiply %v7190, %s3b3bt3 : tensor<1024xf32>
    %v7192 = stablehlo.add %v7191, %v2933 : tensor<1024xf32>
    %v7193 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7194 = stablehlo.multiply %v7193, %s3b3bt3v : tensor<1024xf32>
    %v7195 = stablehlo.add %v7194, %v7192 : tensor<1024xf32>
    %v7196 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7197 = stablehlo.multiply %v7196, %v7195 : tensor<1024xf32>
    %v7198 = stablehlo.subtract %s3b3bt3, %v7197 : tensor<1024xf32>
    %v7199 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7200 = stablehlo.multiply %v7199, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v7201 = stablehlo.add %v7200, %v2650 : tensor<256x1024x1x1xf32>
    %v7202 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7203 = stablehlo.multiply %v7202, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v7204 = stablehlo.add %v7203, %v7201 : tensor<256x1024x1x1xf32>
    %v7205 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7206 = stablehlo.multiply %v7205, %v7204 : tensor<256x1024x1x1xf32>
    %v7207 = stablehlo.subtract %s3b4W1, %v7206 : tensor<256x1024x1x1xf32>
    %v7208 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7209 = stablehlo.multiply %v7208, %s3b4g1 : tensor<256xf32>
    %v7210 = stablehlo.add %v7209, %v2668 : tensor<256xf32>
    %v7211 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7212 = stablehlo.multiply %v7211, %s3b4g1v : tensor<256xf32>
    %v7213 = stablehlo.add %v7212, %v7210 : tensor<256xf32>
    %v7214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7215 = stablehlo.multiply %v7214, %v7213 : tensor<256xf32>
    %v7216 = stablehlo.subtract %s3b4g1, %v7215 : tensor<256xf32>
    %v7217 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7218 = stablehlo.multiply %v7217, %s3b4bt1 : tensor<256xf32>
    %v7219 = stablehlo.add %v7218, %v2671 : tensor<256xf32>
    %v7220 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7221 = stablehlo.multiply %v7220, %s3b4bt1v : tensor<256xf32>
    %v7222 = stablehlo.add %v7221, %v7219 : tensor<256xf32>
    %v7223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7224 = stablehlo.multiply %v7223, %v7222 : tensor<256xf32>
    %v7225 = stablehlo.subtract %s3b4bt1, %v7224 : tensor<256xf32>
    %v7226 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7227 = stablehlo.multiply %v7226, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7228 = stablehlo.add %v7227, %v2677 : tensor<256x256x3x3xf32>
    %v7229 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7230 = stablehlo.multiply %v7229, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7231 = stablehlo.add %v7230, %v7228 : tensor<256x256x3x3xf32>
    %v7232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7233 = stablehlo.multiply %v7232, %v7231 : tensor<256x256x3x3xf32>
    %v7234 = stablehlo.subtract %s3b4W2, %v7233 : tensor<256x256x3x3xf32>
    %v7235 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7236 = stablehlo.multiply %v7235, %s3b4g2 : tensor<256xf32>
    %v7237 = stablehlo.add %v7236, %v2695 : tensor<256xf32>
    %v7238 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7239 = stablehlo.multiply %v7238, %s3b4g2v : tensor<256xf32>
    %v7240 = stablehlo.add %v7239, %v7237 : tensor<256xf32>
    %v7241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7242 = stablehlo.multiply %v7241, %v7240 : tensor<256xf32>
    %v7243 = stablehlo.subtract %s3b4g2, %v7242 : tensor<256xf32>
    %v7244 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7245 = stablehlo.multiply %v7244, %s3b4bt2 : tensor<256xf32>
    %v7246 = stablehlo.add %v7245, %v2698 : tensor<256xf32>
    %v7247 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7248 = stablehlo.multiply %v7247, %s3b4bt2v : tensor<256xf32>
    %v7249 = stablehlo.add %v7248, %v7246 : tensor<256xf32>
    %v7250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7251 = stablehlo.multiply %v7250, %v7249 : tensor<256xf32>
    %v7252 = stablehlo.subtract %s3b4bt2, %v7251 : tensor<256xf32>
    %v7253 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7254 = stablehlo.multiply %v7253, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v7255 = stablehlo.add %v7254, %v2704 : tensor<1024x256x1x1xf32>
    %v7256 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7257 = stablehlo.multiply %v7256, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v7258 = stablehlo.add %v7257, %v7255 : tensor<1024x256x1x1xf32>
    %v7259 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7260 = stablehlo.multiply %v7259, %v7258 : tensor<1024x256x1x1xf32>
    %v7261 = stablehlo.subtract %s3b4W3, %v7260 : tensor<1024x256x1x1xf32>
    %v7262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7263 = stablehlo.multiply %v7262, %s3b4g3 : tensor<1024xf32>
    %v7264 = stablehlo.add %v7263, %v2722 : tensor<1024xf32>
    %v7265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7266 = stablehlo.multiply %v7265, %s3b4g3v : tensor<1024xf32>
    %v7267 = stablehlo.add %v7266, %v7264 : tensor<1024xf32>
    %v7268 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7269 = stablehlo.multiply %v7268, %v7267 : tensor<1024xf32>
    %v7270 = stablehlo.subtract %s3b4g3, %v7269 : tensor<1024xf32>
    %v7271 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7272 = stablehlo.multiply %v7271, %s3b4bt3 : tensor<1024xf32>
    %v7273 = stablehlo.add %v7272, %v2725 : tensor<1024xf32>
    %v7274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7275 = stablehlo.multiply %v7274, %s3b4bt3v : tensor<1024xf32>
    %v7276 = stablehlo.add %v7275, %v7273 : tensor<1024xf32>
    %v7277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7278 = stablehlo.multiply %v7277, %v7276 : tensor<1024xf32>
    %v7279 = stablehlo.subtract %s3b4bt3, %v7278 : tensor<1024xf32>
    %v7280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7281 = stablehlo.multiply %v7280, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v7282 = stablehlo.add %v7281, %v2442 : tensor<256x1024x1x1xf32>
    %v7283 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7284 = stablehlo.multiply %v7283, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v7285 = stablehlo.add %v7284, %v7282 : tensor<256x1024x1x1xf32>
    %v7286 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7287 = stablehlo.multiply %v7286, %v7285 : tensor<256x1024x1x1xf32>
    %v7288 = stablehlo.subtract %s3b5W1, %v7287 : tensor<256x1024x1x1xf32>
    %v7289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7290 = stablehlo.multiply %v7289, %s3b5g1 : tensor<256xf32>
    %v7291 = stablehlo.add %v7290, %v2460 : tensor<256xf32>
    %v7292 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7293 = stablehlo.multiply %v7292, %s3b5g1v : tensor<256xf32>
    %v7294 = stablehlo.add %v7293, %v7291 : tensor<256xf32>
    %v7295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7296 = stablehlo.multiply %v7295, %v7294 : tensor<256xf32>
    %v7297 = stablehlo.subtract %s3b5g1, %v7296 : tensor<256xf32>
    %v7298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7299 = stablehlo.multiply %v7298, %s3b5bt1 : tensor<256xf32>
    %v7300 = stablehlo.add %v7299, %v2463 : tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.multiply %v7301, %s3b5bt1v : tensor<256xf32>
    %v7303 = stablehlo.add %v7302, %v7300 : tensor<256xf32>
    %v7304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7305 = stablehlo.multiply %v7304, %v7303 : tensor<256xf32>
    %v7306 = stablehlo.subtract %s3b5bt1, %v7305 : tensor<256xf32>
    %v7307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7308 = stablehlo.multiply %v7307, %s3b5W2 : tensor<256x256x3x3xf32>
    %v7309 = stablehlo.add %v7308, %v2469 : tensor<256x256x3x3xf32>
    %v7310 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7311 = stablehlo.multiply %v7310, %s3b5W2v : tensor<256x256x3x3xf32>
    %v7312 = stablehlo.add %v7311, %v7309 : tensor<256x256x3x3xf32>
    %v7313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7314 = stablehlo.multiply %v7313, %v7312 : tensor<256x256x3x3xf32>
    %v7315 = stablehlo.subtract %s3b5W2, %v7314 : tensor<256x256x3x3xf32>
    %v7316 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7317 = stablehlo.multiply %v7316, %s3b5g2 : tensor<256xf32>
    %v7318 = stablehlo.add %v7317, %v2487 : tensor<256xf32>
    %v7319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7320 = stablehlo.multiply %v7319, %s3b5g2v : tensor<256xf32>
    %v7321 = stablehlo.add %v7320, %v7318 : tensor<256xf32>
    %v7322 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7323 = stablehlo.multiply %v7322, %v7321 : tensor<256xf32>
    %v7324 = stablehlo.subtract %s3b5g2, %v7323 : tensor<256xf32>
    %v7325 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7326 = stablehlo.multiply %v7325, %s3b5bt2 : tensor<256xf32>
    %v7327 = stablehlo.add %v7326, %v2490 : tensor<256xf32>
    %v7328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7329 = stablehlo.multiply %v7328, %s3b5bt2v : tensor<256xf32>
    %v7330 = stablehlo.add %v7329, %v7327 : tensor<256xf32>
    %v7331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7332 = stablehlo.multiply %v7331, %v7330 : tensor<256xf32>
    %v7333 = stablehlo.subtract %s3b5bt2, %v7332 : tensor<256xf32>
    %v7334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7335 = stablehlo.multiply %v7334, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v7336 = stablehlo.add %v7335, %v2496 : tensor<1024x256x1x1xf32>
    %v7337 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7338 = stablehlo.multiply %v7337, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v7339 = stablehlo.add %v7338, %v7336 : tensor<1024x256x1x1xf32>
    %v7340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7341 = stablehlo.multiply %v7340, %v7339 : tensor<1024x256x1x1xf32>
    %v7342 = stablehlo.subtract %s3b5W3, %v7341 : tensor<1024x256x1x1xf32>
    %v7343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7344 = stablehlo.multiply %v7343, %s3b5g3 : tensor<1024xf32>
    %v7345 = stablehlo.add %v7344, %v2514 : tensor<1024xf32>
    %v7346 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7347 = stablehlo.multiply %v7346, %s3b5g3v : tensor<1024xf32>
    %v7348 = stablehlo.add %v7347, %v7345 : tensor<1024xf32>
    %v7349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7350 = stablehlo.multiply %v7349, %v7348 : tensor<1024xf32>
    %v7351 = stablehlo.subtract %s3b5g3, %v7350 : tensor<1024xf32>
    %v7352 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7353 = stablehlo.multiply %v7352, %s3b5bt3 : tensor<1024xf32>
    %v7354 = stablehlo.add %v7353, %v2517 : tensor<1024xf32>
    %v7355 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7356 = stablehlo.multiply %v7355, %s3b5bt3v : tensor<1024xf32>
    %v7357 = stablehlo.add %v7356, %v7354 : tensor<1024xf32>
    %v7358 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7359 = stablehlo.multiply %v7358, %v7357 : tensor<1024xf32>
    %v7360 = stablehlo.subtract %s3b5bt3, %v7359 : tensor<1024xf32>
    %v7361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7362 = stablehlo.multiply %v7361, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v7363 = stablehlo.add %v7362, %v2205 : tensor<512x1024x1x1xf32>
    %v7364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7365 = stablehlo.multiply %v7364, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v7366 = stablehlo.add %v7365, %v7363 : tensor<512x1024x1x1xf32>
    %v7367 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7368 = stablehlo.multiply %v7367, %v7366 : tensor<512x1024x1x1xf32>
    %v7369 = stablehlo.subtract %s4b0W1, %v7368 : tensor<512x1024x1x1xf32>
    %v7370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7371 = stablehlo.multiply %v7370, %s4b0g1 : tensor<512xf32>
    %v7372 = stablehlo.add %v7371, %v2223 : tensor<512xf32>
    %v7373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7374 = stablehlo.multiply %v7373, %s4b0g1v : tensor<512xf32>
    %v7375 = stablehlo.add %v7374, %v7372 : tensor<512xf32>
    %v7376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7377 = stablehlo.multiply %v7376, %v7375 : tensor<512xf32>
    %v7378 = stablehlo.subtract %s4b0g1, %v7377 : tensor<512xf32>
    %v7379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7380 = stablehlo.multiply %v7379, %s4b0bt1 : tensor<512xf32>
    %v7381 = stablehlo.add %v7380, %v2226 : tensor<512xf32>
    %v7382 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7383 = stablehlo.multiply %v7382, %s4b0bt1v : tensor<512xf32>
    %v7384 = stablehlo.add %v7383, %v7381 : tensor<512xf32>
    %v7385 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7386 = stablehlo.multiply %v7385, %v7384 : tensor<512xf32>
    %v7387 = stablehlo.subtract %s4b0bt1, %v7386 : tensor<512xf32>
    %v7388 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7389 = stablehlo.multiply %v7388, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7390 = stablehlo.add %v7389, %v2233 : tensor<512x512x3x3xf32>
    %v7391 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7392 = stablehlo.multiply %v7391, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7393 = stablehlo.add %v7392, %v7390 : tensor<512x512x3x3xf32>
    %v7394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7395 = stablehlo.multiply %v7394, %v7393 : tensor<512x512x3x3xf32>
    %v7396 = stablehlo.subtract %s4b0W2, %v7395 : tensor<512x512x3x3xf32>
    %v7397 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7398 = stablehlo.multiply %v7397, %s4b0g2 : tensor<512xf32>
    %v7399 = stablehlo.add %v7398, %v2251 : tensor<512xf32>
    %v7400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7401 = stablehlo.multiply %v7400, %s4b0g2v : tensor<512xf32>
    %v7402 = stablehlo.add %v7401, %v7399 : tensor<512xf32>
    %v7403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7404 = stablehlo.multiply %v7403, %v7402 : tensor<512xf32>
    %v7405 = stablehlo.subtract %s4b0g2, %v7404 : tensor<512xf32>
    %v7406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7407 = stablehlo.multiply %v7406, %s4b0bt2 : tensor<512xf32>
    %v7408 = stablehlo.add %v7407, %v2254 : tensor<512xf32>
    %v7409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7410 = stablehlo.multiply %v7409, %s4b0bt2v : tensor<512xf32>
    %v7411 = stablehlo.add %v7410, %v7408 : tensor<512xf32>
    %v7412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7413 = stablehlo.multiply %v7412, %v7411 : tensor<512xf32>
    %v7414 = stablehlo.subtract %s4b0bt2, %v7413 : tensor<512xf32>
    %v7415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7416 = stablehlo.multiply %v7415, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7417 = stablehlo.add %v7416, %v2260 : tensor<2048x512x1x1xf32>
    %v7418 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7419 = stablehlo.multiply %v7418, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7420 = stablehlo.add %v7419, %v7417 : tensor<2048x512x1x1xf32>
    %v7421 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7422 = stablehlo.multiply %v7421, %v7420 : tensor<2048x512x1x1xf32>
    %v7423 = stablehlo.subtract %s4b0W3, %v7422 : tensor<2048x512x1x1xf32>
    %v7424 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7425 = stablehlo.multiply %v7424, %s4b0g3 : tensor<2048xf32>
    %v7426 = stablehlo.add %v7425, %v2278 : tensor<2048xf32>
    %v7427 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7428 = stablehlo.multiply %v7427, %s4b0g3v : tensor<2048xf32>
    %v7429 = stablehlo.add %v7428, %v7426 : tensor<2048xf32>
    %v7430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7431 = stablehlo.multiply %v7430, %v7429 : tensor<2048xf32>
    %v7432 = stablehlo.subtract %s4b0g3, %v7431 : tensor<2048xf32>
    %v7433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7434 = stablehlo.multiply %v7433, %s4b0bt3 : tensor<2048xf32>
    %v7435 = stablehlo.add %v7434, %v2281 : tensor<2048xf32>
    %v7436 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7437 = stablehlo.multiply %v7436, %s4b0bt3v : tensor<2048xf32>
    %v7438 = stablehlo.add %v7437, %v7435 : tensor<2048xf32>
    %v7439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7440 = stablehlo.multiply %v7439, %v7438 : tensor<2048xf32>
    %v7441 = stablehlo.subtract %s4b0bt3, %v7440 : tensor<2048xf32>
    %v7442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7443 = stablehlo.multiply %v7442, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7444 = stablehlo.add %v7443, %v2288 : tensor<2048x1024x1x1xf32>
    %v7445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7446 = stablehlo.multiply %v7445, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7447 = stablehlo.add %v7446, %v7444 : tensor<2048x1024x1x1xf32>
    %v7448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7449 = stablehlo.multiply %v7448, %v7447 : tensor<2048x1024x1x1xf32>
    %v7450 = stablehlo.subtract %s4b0Wp, %v7449 : tensor<2048x1024x1x1xf32>
    %v7451 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7452 = stablehlo.multiply %v7451, %s4b0gp : tensor<2048xf32>
    %v7453 = stablehlo.add %v7452, %v2306 : tensor<2048xf32>
    %v7454 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7455 = stablehlo.multiply %v7454, %s4b0gpv : tensor<2048xf32>
    %v7456 = stablehlo.add %v7455, %v7453 : tensor<2048xf32>
    %v7457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7458 = stablehlo.multiply %v7457, %v7456 : tensor<2048xf32>
    %v7459 = stablehlo.subtract %s4b0gp, %v7458 : tensor<2048xf32>
    %v7460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7461 = stablehlo.multiply %v7460, %s4b0btp : tensor<2048xf32>
    %v7462 = stablehlo.add %v7461, %v2309 : tensor<2048xf32>
    %v7463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7464 = stablehlo.multiply %v7463, %s4b0btpv : tensor<2048xf32>
    %v7465 = stablehlo.add %v7464, %v7462 : tensor<2048xf32>
    %v7466 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7467 = stablehlo.multiply %v7466, %v7465 : tensor<2048xf32>
    %v7468 = stablehlo.subtract %s4b0btp, %v7467 : tensor<2048xf32>
    %v7469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7470 = stablehlo.multiply %v7469, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7471 = stablehlo.add %v7470, %v1962 : tensor<512x2048x1x1xf32>
    %v7472 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7473 = stablehlo.multiply %v7472, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7474 = stablehlo.add %v7473, %v7471 : tensor<512x2048x1x1xf32>
    %v7475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7476 = stablehlo.multiply %v7475, %v7474 : tensor<512x2048x1x1xf32>
    %v7477 = stablehlo.subtract %s4b1W1, %v7476 : tensor<512x2048x1x1xf32>
    %v7478 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7479 = stablehlo.multiply %v7478, %s4b1g1 : tensor<512xf32>
    %v7480 = stablehlo.add %v7479, %v1980 : tensor<512xf32>
    %v7481 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7482 = stablehlo.multiply %v7481, %s4b1g1v : tensor<512xf32>
    %v7483 = stablehlo.add %v7482, %v7480 : tensor<512xf32>
    %v7484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7485 = stablehlo.multiply %v7484, %v7483 : tensor<512xf32>
    %v7486 = stablehlo.subtract %s4b1g1, %v7485 : tensor<512xf32>
    %v7487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7488 = stablehlo.multiply %v7487, %s4b1bt1 : tensor<512xf32>
    %v7489 = stablehlo.add %v7488, %v1983 : tensor<512xf32>
    %v7490 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7491 = stablehlo.multiply %v7490, %s4b1bt1v : tensor<512xf32>
    %v7492 = stablehlo.add %v7491, %v7489 : tensor<512xf32>
    %v7493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7494 = stablehlo.multiply %v7493, %v7492 : tensor<512xf32>
    %v7495 = stablehlo.subtract %s4b1bt1, %v7494 : tensor<512xf32>
    %v7496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7497 = stablehlo.multiply %v7496, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7498 = stablehlo.add %v7497, %v1989 : tensor<512x512x3x3xf32>
    %v7499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7500 = stablehlo.multiply %v7499, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7501 = stablehlo.add %v7500, %v7498 : tensor<512x512x3x3xf32>
    %v7502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7503 = stablehlo.multiply %v7502, %v7501 : tensor<512x512x3x3xf32>
    %v7504 = stablehlo.subtract %s4b1W2, %v7503 : tensor<512x512x3x3xf32>
    %v7505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7506 = stablehlo.multiply %v7505, %s4b1g2 : tensor<512xf32>
    %v7507 = stablehlo.add %v7506, %v2007 : tensor<512xf32>
    %v7508 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7509 = stablehlo.multiply %v7508, %s4b1g2v : tensor<512xf32>
    %v7510 = stablehlo.add %v7509, %v7507 : tensor<512xf32>
    %v7511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7512 = stablehlo.multiply %v7511, %v7510 : tensor<512xf32>
    %v7513 = stablehlo.subtract %s4b1g2, %v7512 : tensor<512xf32>
    %v7514 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7515 = stablehlo.multiply %v7514, %s4b1bt2 : tensor<512xf32>
    %v7516 = stablehlo.add %v7515, %v2010 : tensor<512xf32>
    %v7517 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7518 = stablehlo.multiply %v7517, %s4b1bt2v : tensor<512xf32>
    %v7519 = stablehlo.add %v7518, %v7516 : tensor<512xf32>
    %v7520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7521 = stablehlo.multiply %v7520, %v7519 : tensor<512xf32>
    %v7522 = stablehlo.subtract %s4b1bt2, %v7521 : tensor<512xf32>
    %v7523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7524 = stablehlo.multiply %v7523, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7525 = stablehlo.add %v7524, %v2016 : tensor<2048x512x1x1xf32>
    %v7526 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7527 = stablehlo.multiply %v7526, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7528 = stablehlo.add %v7527, %v7525 : tensor<2048x512x1x1xf32>
    %v7529 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7530 = stablehlo.multiply %v7529, %v7528 : tensor<2048x512x1x1xf32>
    %v7531 = stablehlo.subtract %s4b1W3, %v7530 : tensor<2048x512x1x1xf32>
    %v7532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7533 = stablehlo.multiply %v7532, %s4b1g3 : tensor<2048xf32>
    %v7534 = stablehlo.add %v7533, %v2034 : tensor<2048xf32>
    %v7535 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7536 = stablehlo.multiply %v7535, %s4b1g3v : tensor<2048xf32>
    %v7537 = stablehlo.add %v7536, %v7534 : tensor<2048xf32>
    %v7538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7539 = stablehlo.multiply %v7538, %v7537 : tensor<2048xf32>
    %v7540 = stablehlo.subtract %s4b1g3, %v7539 : tensor<2048xf32>
    %v7541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7542 = stablehlo.multiply %v7541, %s4b1bt3 : tensor<2048xf32>
    %v7543 = stablehlo.add %v7542, %v2037 : tensor<2048xf32>
    %v7544 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7545 = stablehlo.multiply %v7544, %s4b1bt3v : tensor<2048xf32>
    %v7546 = stablehlo.add %v7545, %v7543 : tensor<2048xf32>
    %v7547 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7548 = stablehlo.multiply %v7547, %v7546 : tensor<2048xf32>
    %v7549 = stablehlo.subtract %s4b1bt3, %v7548 : tensor<2048xf32>
    %v7550 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7551 = stablehlo.multiply %v7550, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7552 = stablehlo.add %v7551, %v1754 : tensor<512x2048x1x1xf32>
    %v7553 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7554 = stablehlo.multiply %v7553, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7555 = stablehlo.add %v7554, %v7552 : tensor<512x2048x1x1xf32>
    %v7556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7557 = stablehlo.multiply %v7556, %v7555 : tensor<512x2048x1x1xf32>
    %v7558 = stablehlo.subtract %s4b2W1, %v7557 : tensor<512x2048x1x1xf32>
    %v7559 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7560 = stablehlo.multiply %v7559, %s4b2g1 : tensor<512xf32>
    %v7561 = stablehlo.add %v7560, %v1772 : tensor<512xf32>
    %v7562 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7563 = stablehlo.multiply %v7562, %s4b2g1v : tensor<512xf32>
    %v7564 = stablehlo.add %v7563, %v7561 : tensor<512xf32>
    %v7565 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7566 = stablehlo.multiply %v7565, %v7564 : tensor<512xf32>
    %v7567 = stablehlo.subtract %s4b2g1, %v7566 : tensor<512xf32>
    %v7568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7569 = stablehlo.multiply %v7568, %s4b2bt1 : tensor<512xf32>
    %v7570 = stablehlo.add %v7569, %v1775 : tensor<512xf32>
    %v7571 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7572 = stablehlo.multiply %v7571, %s4b2bt1v : tensor<512xf32>
    %v7573 = stablehlo.add %v7572, %v7570 : tensor<512xf32>
    %v7574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7575 = stablehlo.multiply %v7574, %v7573 : tensor<512xf32>
    %v7576 = stablehlo.subtract %s4b2bt1, %v7575 : tensor<512xf32>
    %v7577 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7578 = stablehlo.multiply %v7577, %s4b2W2 : tensor<512x512x3x3xf32>
    %v7579 = stablehlo.add %v7578, %v1781 : tensor<512x512x3x3xf32>
    %v7580 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7581 = stablehlo.multiply %v7580, %s4b2W2v : tensor<512x512x3x3xf32>
    %v7582 = stablehlo.add %v7581, %v7579 : tensor<512x512x3x3xf32>
    %v7583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7584 = stablehlo.multiply %v7583, %v7582 : tensor<512x512x3x3xf32>
    %v7585 = stablehlo.subtract %s4b2W2, %v7584 : tensor<512x512x3x3xf32>
    %v7586 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7587 = stablehlo.multiply %v7586, %s4b2g2 : tensor<512xf32>
    %v7588 = stablehlo.add %v7587, %v1799 : tensor<512xf32>
    %v7589 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7590 = stablehlo.multiply %v7589, %s4b2g2v : tensor<512xf32>
    %v7591 = stablehlo.add %v7590, %v7588 : tensor<512xf32>
    %v7592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7593 = stablehlo.multiply %v7592, %v7591 : tensor<512xf32>
    %v7594 = stablehlo.subtract %s4b2g2, %v7593 : tensor<512xf32>
    %v7595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7596 = stablehlo.multiply %v7595, %s4b2bt2 : tensor<512xf32>
    %v7597 = stablehlo.add %v7596, %v1802 : tensor<512xf32>
    %v7598 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7599 = stablehlo.multiply %v7598, %s4b2bt2v : tensor<512xf32>
    %v7600 = stablehlo.add %v7599, %v7597 : tensor<512xf32>
    %v7601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7602 = stablehlo.multiply %v7601, %v7600 : tensor<512xf32>
    %v7603 = stablehlo.subtract %s4b2bt2, %v7602 : tensor<512xf32>
    %v7604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7605 = stablehlo.multiply %v7604, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v7606 = stablehlo.add %v7605, %v1808 : tensor<2048x512x1x1xf32>
    %v7607 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7608 = stablehlo.multiply %v7607, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v7609 = stablehlo.add %v7608, %v7606 : tensor<2048x512x1x1xf32>
    %v7610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7611 = stablehlo.multiply %v7610, %v7609 : tensor<2048x512x1x1xf32>
    %v7612 = stablehlo.subtract %s4b2W3, %v7611 : tensor<2048x512x1x1xf32>
    %v7613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7614 = stablehlo.multiply %v7613, %s4b2g3 : tensor<2048xf32>
    %v7615 = stablehlo.add %v7614, %v1826 : tensor<2048xf32>
    %v7616 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7617 = stablehlo.multiply %v7616, %s4b2g3v : tensor<2048xf32>
    %v7618 = stablehlo.add %v7617, %v7615 : tensor<2048xf32>
    %v7619 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7620 = stablehlo.multiply %v7619, %v7618 : tensor<2048xf32>
    %v7621 = stablehlo.subtract %s4b2g3, %v7620 : tensor<2048xf32>
    %v7622 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7623 = stablehlo.multiply %v7622, %s4b2bt3 : tensor<2048xf32>
    %v7624 = stablehlo.add %v7623, %v1829 : tensor<2048xf32>
    %v7625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7626 = stablehlo.multiply %v7625, %s4b2bt3v : tensor<2048xf32>
    %v7627 = stablehlo.add %v7626, %v7624 : tensor<2048xf32>
    %v7628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7629 = stablehlo.multiply %v7628, %v7627 : tensor<2048xf32>
    %v7630 = stablehlo.subtract %s4b2bt3, %v7629 : tensor<2048xf32>
    %v7631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7632 = stablehlo.multiply %v7631, %Wd : tensor<2048x1000xf32>
    %v7633 = stablehlo.add %v7632, %v1615 : tensor<2048x1000xf32>
    %v7634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7635 = stablehlo.multiply %v7634, %Wdv : tensor<2048x1000xf32>
    %v7636 = stablehlo.add %v7635, %v7633 : tensor<2048x1000xf32>
    %v7637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7638 = stablehlo.multiply %v7637, %v7636 : tensor<2048x1000xf32>
    %v7639 = stablehlo.subtract %Wd, %v7638 : tensor<2048x1000xf32>
    %v7640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7641 = stablehlo.multiply %v7640, %bd : tensor<1000xf32>
    %v7642 = stablehlo.add %v7641, %v1617 : tensor<1000xf32>
    %v7643 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7644 = stablehlo.multiply %v7643, %bdv : tensor<1000xf32>
    %v7645 = stablehlo.add %v7644, %v7642 : tensor<1000xf32>
    %v7646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7647 = stablehlo.multiply %v7646, %v7645 : tensor<1000xf32>
    %v7648 = stablehlo.subtract %bd, %v7647 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1603 : tensor<256x1000xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<256x1000xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<256xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<256xf32>
    %laKc = stablehlo.constant dense<0.000100> : tensor<256xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<256xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<256xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<256xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<256xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<256.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v6208, %v6217, %v6226, %v6235, %v6244, %v6253, %v6262, %v6271, %v6280, %v6289, %v6298, %v6307, %v6316, %v6325, %v6334, %v6343, %v6352, %v6361, %v6370, %v6379, %v6388, %v6397, %v6406, %v6415, %v6424, %v6433, %v6442, %v6451, %v6460, %v6469, %v6478, %v6487, %v6496, %v6505, %v6514, %v6523, %v6532, %v6541, %v6550, %v6559, %v6568, %v6577, %v6586, %v6595, %v6604, %v6613, %v6622, %v6631, %v6640, %v6649, %v6658, %v6667, %v6676, %v6685, %v6694, %v6703, %v6712, %v6721, %v6730, %v6739, %v6748, %v6757, %v6766, %v6775, %v6784, %v6793, %v6802, %v6811, %v6820, %v6829, %v6838, %v6847, %v6856, %v6865, %v6874, %v6883, %v6892, %v6901, %v6910, %v6919, %v6928, %v6937, %v6946, %v6955, %v6964, %v6973, %v6982, %v6991, %v7000, %v7009, %v7018, %v7027, %v7036, %v7045, %v7054, %v7063, %v7072, %v7081, %v7090, %v7099, %v7108, %v7117, %v7126, %v7135, %v7144, %v7153, %v7162, %v7171, %v7180, %v7189, %v7198, %v7207, %v7216, %v7225, %v7234, %v7243, %v7252, %v7261, %v7270, %v7279, %v7288, %v7297, %v7306, %v7315, %v7324, %v7333, %v7342, %v7351, %v7360, %v7369, %v7378, %v7387, %v7396, %v7405, %v7414, %v7423, %v7432, %v7441, %v7450, %v7459, %v7468, %v7477, %v7486, %v7495, %v7504, %v7513, %v7522, %v7531, %v7540, %v7549, %v7558, %v7567, %v7576, %v7585, %v7594, %v7603, %v7612, %v7621, %v7630, %v7639, %v7648, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v6205, %v6214, %v6223, %v6232, %v6241, %v6250, %v6259, %v6268, %v6277, %v6286, %v6295, %v6304, %v6313, %v6322, %v6331, %v6340, %v6349, %v6358, %v6367, %v6376, %v6385, %v6394, %v6403, %v6412, %v6421, %v6430, %v6439, %v6448, %v6457, %v6466, %v6475, %v6484, %v6493, %v6502, %v6511, %v6520, %v6529, %v6538, %v6547, %v6556, %v6565, %v6574, %v6583, %v6592, %v6601, %v6610, %v6619, %v6628, %v6637, %v6646, %v6655, %v6664, %v6673, %v6682, %v6691, %v6700, %v6709, %v6718, %v6727, %v6736, %v6745, %v6754, %v6763, %v6772, %v6781, %v6790, %v6799, %v6808, %v6817, %v6826, %v6835, %v6844, %v6853, %v6862, %v6871, %v6880, %v6889, %v6898, %v6907, %v6916, %v6925, %v6934, %v6943, %v6952, %v6961, %v6970, %v6979, %v6988, %v6997, %v7006, %v7015, %v7024, %v7033, %v7042, %v7051, %v7060, %v7069, %v7078, %v7087, %v7096, %v7105, %v7114, %v7123, %v7132, %v7141, %v7150, %v7159, %v7168, %v7177, %v7186, %v7195, %v7204, %v7213, %v7222, %v7231, %v7240, %v7249, %v7258, %v7267, %v7276, %v7285, %v7294, %v7303, %v7312, %v7321, %v7330, %v7339, %v7348, %v7357, %v7366, %v7375, %v7384, %v7393, %v7402, %v7411, %v7420, %v7429, %v7438, %v7447, %v7456, %v7465, %v7474, %v7483, %v7492, %v7501, %v7510, %v7519, %v7528, %v7537, %v7546, %v7555, %v7564, %v7573, %v7582, %v7591, %v7600, %v7609, %v7618, %v7627, %v7636, %v7645, %loss, %bc1, %bc2, %v5356, %v5367, %v5372, %v5383, %v5388, %v5399, %v5404, %v5415, %v5420, %v5431, %v5436, %v5447, %v5452, %v5463, %v5468, %v5479, %v5484, %v5495, %v5500, %v5511, %v5516, %v5527, %v5532, %v5543, %v5548, %v5559, %v5564, %v5575, %v5580, %v5591, %v5596, %v5607, %v5612, %v5623, %v5628, %v5639, %v5644, %v5655, %v5660, %v5671, %v5676, %v5687, %v5692, %v5703, %v5708, %v5719, %v5724, %v5735, %v5740, %v5751, %v5756, %v5767, %v5772, %v5783, %v5788, %v5799, %v5804, %v5815, %v5820, %v5831, %v5836, %v5847, %v5852, %v5863, %v5868, %v5879, %v5884, %v5895, %v5900, %v5911, %v5916, %v5927, %v5932, %v5943, %v5948, %v5959, %v5964, %v5975, %v5980, %v5991, %v5996, %v6007, %v6012, %v6023, %v6028, %v6039, %v6044, %v6055, %v6060, %v6071, %v6076, %v6087, %v6092, %v6103, %v6108, %v6119, %v6124, %v6135, %v6140, %v6151, %v6156, %v6167, %v6172, %v6183, %v6188, %v6199 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
