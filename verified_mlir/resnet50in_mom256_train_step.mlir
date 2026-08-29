module @m {
  func.func @resnet50in_mom256_train_step(%x: tensor<256x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x1x1xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b0W3m: tensor<256x64x1x1xf32>, %s1b0g3m: tensor<256xf32>, %s1b0bt3m: tensor<256xf32>, %s1b0Wpm: tensor<256x64x1x1xf32>, %s1b0gpm: tensor<256xf32>, %s1b0btpm: tensor<256xf32>, %s1b1W1m: tensor<64x256x1x1xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b1W3m: tensor<256x64x1x1xf32>, %s1b1g3m: tensor<256xf32>, %s1b1bt3m: tensor<256xf32>, %s1b2W1m: tensor<64x256x1x1xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %s1b2W3m: tensor<256x64x1x1xf32>, %s1b2g3m: tensor<256xf32>, %s1b2bt3m: tensor<256xf32>, %s2b0W1m: tensor<128x256x1x1xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b0W3m: tensor<512x128x1x1xf32>, %s2b0g3m: tensor<512xf32>, %s2b0bt3m: tensor<512xf32>, %s2b0Wpm: tensor<512x256x1x1xf32>, %s2b0gpm: tensor<512xf32>, %s2b0btpm: tensor<512xf32>, %s2b1W1m: tensor<128x512x1x1xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b1W3m: tensor<512x128x1x1xf32>, %s2b1g3m: tensor<512xf32>, %s2b1bt3m: tensor<512xf32>, %s2b2W1m: tensor<128x512x1x1xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %s2b2W3m: tensor<512x128x1x1xf32>, %s2b2g3m: tensor<512xf32>, %s2b2bt3m: tensor<512xf32>, %s2b3W1m: tensor<128x512x1x1xf32>, %s2b3g1m: tensor<128xf32>, %s2b3bt1m: tensor<128xf32>, %s2b3W2m: tensor<128x128x3x3xf32>, %s2b3g2m: tensor<128xf32>, %s2b3bt2m: tensor<128xf32>, %s2b3W3m: tensor<512x128x1x1xf32>, %s2b3g3m: tensor<512xf32>, %s2b3bt3m: tensor<512xf32>, %s3b0W1m: tensor<256x512x1x1xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b0W3m: tensor<1024x256x1x1xf32>, %s3b0g3m: tensor<1024xf32>, %s3b0bt3m: tensor<1024xf32>, %s3b0Wpm: tensor<1024x512x1x1xf32>, %s3b0gpm: tensor<1024xf32>, %s3b0btpm: tensor<1024xf32>, %s3b1W1m: tensor<256x1024x1x1xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b1W3m: tensor<1024x256x1x1xf32>, %s3b1g3m: tensor<1024xf32>, %s3b1bt3m: tensor<1024xf32>, %s3b2W1m: tensor<256x1024x1x1xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b2W3m: tensor<1024x256x1x1xf32>, %s3b2g3m: tensor<1024xf32>, %s3b2bt3m: tensor<1024xf32>, %s3b3W1m: tensor<256x1024x1x1xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b3W3m: tensor<1024x256x1x1xf32>, %s3b3g3m: tensor<1024xf32>, %s3b3bt3m: tensor<1024xf32>, %s3b4W1m: tensor<256x1024x1x1xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %s3b4W3m: tensor<1024x256x1x1xf32>, %s3b4g3m: tensor<1024xf32>, %s3b4bt3m: tensor<1024xf32>, %s3b5W1m: tensor<256x1024x1x1xf32>, %s3b5g1m: tensor<256xf32>, %s3b5bt1m: tensor<256xf32>, %s3b5W2m: tensor<256x256x3x3xf32>, %s3b5g2m: tensor<256xf32>, %s3b5bt2m: tensor<256xf32>, %s3b5W3m: tensor<1024x256x1x1xf32>, %s3b5g3m: tensor<1024xf32>, %s3b5bt3m: tensor<1024xf32>, %s4b0W1m: tensor<512x1024x1x1xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b0W3m: tensor<2048x512x1x1xf32>, %s4b0g3m: tensor<2048xf32>, %s4b0bt3m: tensor<2048xf32>, %s4b0Wpm: tensor<2048x1024x1x1xf32>, %s4b0gpm: tensor<2048xf32>, %s4b0btpm: tensor<2048xf32>, %s4b1W1m: tensor<512x2048x1x1xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %s4b1W3m: tensor<2048x512x1x1xf32>, %s4b1g3m: tensor<2048xf32>, %s4b1bt3m: tensor<2048xf32>, %s4b2W1m: tensor<512x2048x1x1xf32>, %s4b2g1m: tensor<512xf32>, %s4b2bt1m: tensor<512xf32>, %s4b2W2m: tensor<512x512x3x3xf32>, %s4b2g2m: tensor<512xf32>, %s4b2bt2m: tensor<512xf32>, %s4b2W3m: tensor<2048x512x1x1xf32>, %s4b2g3m: tensor<2048xf32>, %s4b2bt3m: tensor<2048xf32>, %Wdm: tensor<2048x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x1x1xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b0W3v: tensor<256x64x1x1xf32>, %s1b0g3v: tensor<256xf32>, %s1b0bt3v: tensor<256xf32>, %s1b0Wpv: tensor<256x64x1x1xf32>, %s1b0gpv: tensor<256xf32>, %s1b0btpv: tensor<256xf32>, %s1b1W1v: tensor<64x256x1x1xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b1W3v: tensor<256x64x1x1xf32>, %s1b1g3v: tensor<256xf32>, %s1b1bt3v: tensor<256xf32>, %s1b2W1v: tensor<64x256x1x1xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %s1b2W3v: tensor<256x64x1x1xf32>, %s1b2g3v: tensor<256xf32>, %s1b2bt3v: tensor<256xf32>, %s2b0W1v: tensor<128x256x1x1xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b0W3v: tensor<512x128x1x1xf32>, %s2b0g3v: tensor<512xf32>, %s2b0bt3v: tensor<512xf32>, %s2b0Wpv: tensor<512x256x1x1xf32>, %s2b0gpv: tensor<512xf32>, %s2b0btpv: tensor<512xf32>, %s2b1W1v: tensor<128x512x1x1xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b1W3v: tensor<512x128x1x1xf32>, %s2b1g3v: tensor<512xf32>, %s2b1bt3v: tensor<512xf32>, %s2b2W1v: tensor<128x512x1x1xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %s2b2W3v: tensor<512x128x1x1xf32>, %s2b2g3v: tensor<512xf32>, %s2b2bt3v: tensor<512xf32>, %s2b3W1v: tensor<128x512x1x1xf32>, %s2b3g1v: tensor<128xf32>, %s2b3bt1v: tensor<128xf32>, %s2b3W2v: tensor<128x128x3x3xf32>, %s2b3g2v: tensor<128xf32>, %s2b3bt2v: tensor<128xf32>, %s2b3W3v: tensor<512x128x1x1xf32>, %s2b3g3v: tensor<512xf32>, %s2b3bt3v: tensor<512xf32>, %s3b0W1v: tensor<256x512x1x1xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b0W3v: tensor<1024x256x1x1xf32>, %s3b0g3v: tensor<1024xf32>, %s3b0bt3v: tensor<1024xf32>, %s3b0Wpv: tensor<1024x512x1x1xf32>, %s3b0gpv: tensor<1024xf32>, %s3b0btpv: tensor<1024xf32>, %s3b1W1v: tensor<256x1024x1x1xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b1W3v: tensor<1024x256x1x1xf32>, %s3b1g3v: tensor<1024xf32>, %s3b1bt3v: tensor<1024xf32>, %s3b2W1v: tensor<256x1024x1x1xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b2W3v: tensor<1024x256x1x1xf32>, %s3b2g3v: tensor<1024xf32>, %s3b2bt3v: tensor<1024xf32>, %s3b3W1v: tensor<256x1024x1x1xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b3W3v: tensor<1024x256x1x1xf32>, %s3b3g3v: tensor<1024xf32>, %s3b3bt3v: tensor<1024xf32>, %s3b4W1v: tensor<256x1024x1x1xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %s3b4W3v: tensor<1024x256x1x1xf32>, %s3b4g3v: tensor<1024xf32>, %s3b4bt3v: tensor<1024xf32>, %s3b5W1v: tensor<256x1024x1x1xf32>, %s3b5g1v: tensor<256xf32>, %s3b5bt1v: tensor<256xf32>, %s3b5W2v: tensor<256x256x3x3xf32>, %s3b5g2v: tensor<256xf32>, %s3b5bt2v: tensor<256xf32>, %s3b5W3v: tensor<1024x256x1x1xf32>, %s3b5g3v: tensor<1024xf32>, %s3b5bt3v: tensor<1024xf32>, %s4b0W1v: tensor<512x1024x1x1xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b0W3v: tensor<2048x512x1x1xf32>, %s4b0g3v: tensor<2048xf32>, %s4b0bt3v: tensor<2048xf32>, %s4b0Wpv: tensor<2048x1024x1x1xf32>, %s4b0gpv: tensor<2048xf32>, %s4b0btpv: tensor<2048xf32>, %s4b1W1v: tensor<512x2048x1x1xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %s4b1W3v: tensor<2048x512x1x1xf32>, %s4b1g3v: tensor<2048xf32>, %s4b1bt3v: tensor<2048xf32>, %s4b2W1v: tensor<512x2048x1x1xf32>, %s4b2g1v: tensor<512xf32>, %s4b2bt1v: tensor<512xf32>, %s4b2W2v: tensor<512x512x3x3xf32>, %s4b2g2v: tensor<512xf32>, %s4b2bt2v: tensor<512xf32>, %s4b2W3v: tensor<2048x512x1x1xf32>, %s4b2g3v: tensor<2048xf32>, %s4b2bt3v: tensor<2048xf32>, %Wdv: tensor<2048x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b0n3mui: tensor<256xf32>, %s1b0n3vari: tensor<256xf32>, %s1b0npmui: tensor<256xf32>, %s1b0npvari: tensor<256xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b1n3mui: tensor<256xf32>, %s1b1n3vari: tensor<256xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %s1b2n3mui: tensor<256xf32>, %s1b2n3vari: tensor<256xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b0n3mui: tensor<512xf32>, %s2b0n3vari: tensor<512xf32>, %s2b0npmui: tensor<512xf32>, %s2b0npvari: tensor<512xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b1n3mui: tensor<512xf32>, %s2b1n3vari: tensor<512xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %s2b2n3mui: tensor<512xf32>, %s2b2n3vari: tensor<512xf32>, %s2b3n1mui: tensor<128xf32>, %s2b3n1vari: tensor<128xf32>, %s2b3n2mui: tensor<128xf32>, %s2b3n2vari: tensor<128xf32>, %s2b3n3mui: tensor<512xf32>, %s2b3n3vari: tensor<512xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b0n3mui: tensor<1024xf32>, %s3b0n3vari: tensor<1024xf32>, %s3b0npmui: tensor<1024xf32>, %s3b0npvari: tensor<1024xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b1n3mui: tensor<1024xf32>, %s3b1n3vari: tensor<1024xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b2n3mui: tensor<1024xf32>, %s3b2n3vari: tensor<1024xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b3n3mui: tensor<1024xf32>, %s3b3n3vari: tensor<1024xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %s3b4n3mui: tensor<1024xf32>, %s3b4n3vari: tensor<1024xf32>, %s3b5n1mui: tensor<256xf32>, %s3b5n1vari: tensor<256xf32>, %s3b5n2mui: tensor<256xf32>, %s3b5n2vari: tensor<256xf32>, %s3b5n3mui: tensor<1024xf32>, %s3b5n3vari: tensor<1024xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b0n3mui: tensor<2048xf32>, %s4b0n3vari: tensor<2048xf32>, %s4b0npmui: tensor<2048xf32>, %s4b0npvari: tensor<2048xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %s4b1n3mui: tensor<2048xf32>, %s4b1n3vari: tensor<2048xf32>, %s4b2n1mui: tensor<512xf32>, %s4b2n1vari: tensor<512xf32>, %s4b2n2mui: tensor<512xf32>, %s4b2n2vari: tensor<512xf32>, %s4b2n3mui: tensor<2048xf32>, %s4b2n3vari: tensor<2048xf32>, %onehot: tensor<256x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>) {
    // ── ResNet-50 bottleneck batch-BN heavy-ball momentum + coupled L2 train step: every line is pretty(verified AST node) ──
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
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
    %v2116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2117 = stablehlo.pad %v2115, %v2116, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v2118 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2119 = stablehlo.transpose %v2118, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2120 = stablehlo.convolution(%v2117, %v2119)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x14x14xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2122 = stablehlo.reshape %v2121 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2123 = stablehlo.reshape %v1315 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2124 = stablehlo.constant dense<0.0> : tensor<256x512x14x14xf32>
    %v2125 = stablehlo.compare GT, %v2123, %v2124 : (tensor<256x512x14x14xf32>, tensor<256x512x14x14xf32>) -> tensor<256x512x14x14xi1>
    %v2126 = stablehlo.select %v2125, %v2122, %v2124 : tensor<256x512x14x14xi1>, tensor<256x512x14x14xf32>
    %v2127 = stablehlo.reshape %v2126 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2128 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2130 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v2131 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v2132 = stablehlo.reduce(%v2128 init: %v2129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2132, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2134 = stablehlo.divide %v2133, %v2130 : tensor<256x512x14x14xf32>
    %v2135 = stablehlo.subtract %v2128, %v2134 : tensor<256x512x14x14xf32>
    %v2136 = stablehlo.multiply %v2135, %v2135 : tensor<256x512x14x14xf32>
    %v2137 = stablehlo.reduce(%v2136 init: %v2129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2138 = stablehlo.broadcast_in_dim %v2137, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2139 = stablehlo.divide %v2138, %v2130 : tensor<256x512x14x14xf32>
    %v2140 = stablehlo.add %v2139, %v2131 : tensor<256x512x14x14xf32>
    %v2141 = stablehlo.rsqrt %v2140 : tensor<256x512x14x14xf32>
    %v2142 = stablehlo.multiply %v2135, %v2141 : tensor<256x512x14x14xf32>
    %v2143 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2144 = stablehlo.reshape %v2127 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2145 = stablehlo.multiply %v2143, %v2144 : tensor<256x512x14x14xf32>
    %v2146 = stablehlo.reduce(%v2145 init: %v2129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2147 = stablehlo.broadcast_in_dim %v2146, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2148 = stablehlo.multiply %v2142, %v2145 : tensor<256x512x14x14xf32>
    %v2149 = stablehlo.reduce(%v2148 init: %v2129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2150 = stablehlo.broadcast_in_dim %v2149, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2151 = stablehlo.multiply %v2145, %v2130 : tensor<256x512x14x14xf32>
    %v2152 = stablehlo.subtract %v2151, %v2147 : tensor<256x512x14x14xf32>
    %v2153 = stablehlo.multiply %v2142, %v2150 : tensor<256x512x14x14xf32>
    %v2154 = stablehlo.subtract %v2152, %v2153 : tensor<256x512x14x14xf32>
    %v2155 = stablehlo.divide %v2141, %v2130 : tensor<256x512x14x14xf32>
    %v2156 = stablehlo.multiply %v2155, %v2154 : tensor<256x512x14x14xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v2158 = stablehlo.reshape %v2157 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2159 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v2160 = stablehlo.transpose %v2159, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v2161 = stablehlo.convolution(%v2158, %v2160)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2163 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2165 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2166 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2167 = stablehlo.reduce(%v2163 init: %v2164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2168 = stablehlo.broadcast_in_dim %v2167, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2169 = stablehlo.divide %v2168, %v2165 : tensor<256x2048x7x7xf32>
    %v2170 = stablehlo.subtract %v2163, %v2169 : tensor<256x2048x7x7xf32>
    %v2171 = stablehlo.multiply %v2170, %v2170 : tensor<256x2048x7x7xf32>
    %v2172 = stablehlo.reduce(%v2171 init: %v2164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2173 = stablehlo.broadcast_in_dim %v2172, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2174 = stablehlo.divide %v2173, %v2165 : tensor<256x2048x7x7xf32>
    %v2175 = stablehlo.add %v2174, %v2166 : tensor<256x2048x7x7xf32>
    %v2176 = stablehlo.rsqrt %v2175 : tensor<256x2048x7x7xf32>
    %v2177 = stablehlo.multiply %v2170, %v2176 : tensor<256x2048x7x7xf32>
    %v2178 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2179 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2180 = stablehlo.multiply %v2178, %v2179 : tensor<256x2048x7x7xf32>
    %v2181 = stablehlo.reduce(%v2180 init: %v2164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2182 = stablehlo.broadcast_in_dim %v2181, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2183 = stablehlo.multiply %v2177, %v2180 : tensor<256x2048x7x7xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2184, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2186 = stablehlo.multiply %v2180, %v2165 : tensor<256x2048x7x7xf32>
    %v2187 = stablehlo.subtract %v2186, %v2182 : tensor<256x2048x7x7xf32>
    %v2188 = stablehlo.multiply %v2177, %v2185 : tensor<256x2048x7x7xf32>
    %v2189 = stablehlo.subtract %v2187, %v2188 : tensor<256x2048x7x7xf32>
    %v2190 = stablehlo.divide %v2176, %v2165 : tensor<256x2048x7x7xf32>
    %v2191 = stablehlo.multiply %v2190, %v2189 : tensor<256x2048x7x7xf32>
    %v2192 = stablehlo.reshape %v2191 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v2193 = stablehlo.reshape %v2192 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2195 = stablehlo.pad %v2193, %v2194, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048x14x14xf32>
    %v2196 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2197 = stablehlo.transpose %v2196, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2198 = stablehlo.convolution(%v2195, %v2197)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x14x14xf32>, tensor<1024x2048x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2199 = stablehlo.reshape %v2198 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2200 = stablehlo.reshape %v2162 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2201 = stablehlo.reshape %v2199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2202 = stablehlo.add %v2200, %v2201 : tensor<256x1024x14x14xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2204 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2205 = stablehlo.reshape %v2157 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2206 = stablehlo.transpose %v2204, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2207 = stablehlo.transpose %v2205, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2208 = stablehlo.convolution(%v2206, %v2207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<1024x512x1x1xf32>
    %v2209 = stablehlo.transpose %v2208, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2210 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2212 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v2213 = stablehlo.reduce(%v2210 init: %v2211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2214 = stablehlo.broadcast_in_dim %v2213, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2215 = stablehlo.divide %v2214, %v2212 : tensor<256x512x14x14xf32>
    %v2216 = stablehlo.subtract %v2210, %v2215 : tensor<256x512x14x14xf32>
    %v2217 = stablehlo.multiply %v2216, %v2216 : tensor<256x512x14x14xf32>
    %v2218 = stablehlo.reduce(%v2217 init: %v2211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2219 = stablehlo.broadcast_in_dim %v2218, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2220 = stablehlo.divide %v2219, %v2212 : tensor<256x512x14x14xf32>
    %v2221 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v2222 = stablehlo.add %v2220, %v2221 : tensor<256x512x14x14xf32>
    %v2223 = stablehlo.rsqrt %v2222 : tensor<256x512x14x14xf32>
    %v2224 = stablehlo.multiply %v2216, %v2223 : tensor<256x512x14x14xf32>
    %v2225 = stablehlo.reshape %v2127 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2226 = stablehlo.multiply %v2225, %v2224 : tensor<256x512x14x14xf32>
    %v2227 = stablehlo.reduce(%v2226 init: %v2211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2228 = stablehlo.reshape %v2127 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2230 = stablehlo.reduce(%v2228 init: %v2229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2231 = stablehlo.reshape %v1319 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2232 = stablehlo.reshape %v2114 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2233 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2234 = stablehlo.pad %v2232, %v2233, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v2235 = stablehlo.transpose %v2231, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2236 = stablehlo.transpose %v2234, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2237 = stablehlo.convolution(%v2235, %v2236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<512x512x3x3xf32>
    %v2238 = stablehlo.transpose %v2237, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2239 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v2242 = stablehlo.reduce(%v2239 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2243 = stablehlo.broadcast_in_dim %v2242, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2244 = stablehlo.divide %v2243, %v2241 : tensor<256x512x7x7xf32>
    %v2245 = stablehlo.subtract %v2239, %v2244 : tensor<256x512x7x7xf32>
    %v2246 = stablehlo.multiply %v2245, %v2245 : tensor<256x512x7x7xf32>
    %v2247 = stablehlo.reduce(%v2246 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2248 = stablehlo.broadcast_in_dim %v2247, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2249 = stablehlo.divide %v2248, %v2241 : tensor<256x512x7x7xf32>
    %v2250 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v2251 = stablehlo.add %v2249, %v2250 : tensor<256x512x7x7xf32>
    %v2252 = stablehlo.rsqrt %v2251 : tensor<256x512x7x7xf32>
    %v2253 = stablehlo.multiply %v2245, %v2252 : tensor<256x512x7x7xf32>
    %v2254 = stablehlo.reshape %v2084 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2255 = stablehlo.multiply %v2254, %v2253 : tensor<256x512x7x7xf32>
    %v2256 = stablehlo.reduce(%v2255 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2257 = stablehlo.reshape %v2084 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2259 = stablehlo.reduce(%v2257 init: %v2258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2260 = stablehlo.reshape %v1348 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2261 = stablehlo.reshape %v2073 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2262 = stablehlo.transpose %v2260, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v2263 = stablehlo.transpose %v2261, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v2264 = stablehlo.convolution(%v2262, %v2263)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2265 = stablehlo.transpose %v2264, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2266 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2268 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2269 = stablehlo.reduce(%v2266 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2271 = stablehlo.divide %v2270, %v2268 : tensor<256x2048x7x7xf32>
    %v2272 = stablehlo.subtract %v2266, %v2271 : tensor<256x2048x7x7xf32>
    %v2273 = stablehlo.multiply %v2272, %v2272 : tensor<256x2048x7x7xf32>
    %v2274 = stablehlo.reduce(%v2273 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2275 = stablehlo.broadcast_in_dim %v2274, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2276 = stablehlo.divide %v2275, %v2268 : tensor<256x2048x7x7xf32>
    %v2277 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2278 = stablehlo.add %v2276, %v2277 : tensor<256x2048x7x7xf32>
    %v2279 = stablehlo.rsqrt %v2278 : tensor<256x2048x7x7xf32>
    %v2280 = stablehlo.multiply %v2272, %v2279 : tensor<256x2048x7x7xf32>
    %v2281 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2282 = stablehlo.multiply %v2281, %v2280 : tensor<256x2048x7x7xf32>
    %v2283 = stablehlo.reduce(%v2282 init: %v2267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2284 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2286 = stablehlo.reduce(%v2284 init: %v2285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2287 = stablehlo.reshape %v1290 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2288 = stablehlo.reshape %v2192 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2290 = stablehlo.pad %v2288, %v2289, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048x14x14xf32>
    %v2291 = stablehlo.transpose %v2287, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2292 = stablehlo.transpose %v2290, dims = [1, 0, 2, 3] : (tensor<256x2048x14x14xf32>) -> tensor<2048x256x14x14xf32>
    %v2293 = stablehlo.convolution(%v2291, %v2292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<2048x256x14x14xf32>) -> tensor<1024x2048x1x1xf32>
    %v2294 = stablehlo.transpose %v2293, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v2295 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2298 = stablehlo.reduce(%v2295 init: %v2296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2299 = stablehlo.broadcast_in_dim %v2298, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2300 = stablehlo.divide %v2299, %v2297 : tensor<256x2048x7x7xf32>
    %v2301 = stablehlo.subtract %v2295, %v2300 : tensor<256x2048x7x7xf32>
    %v2302 = stablehlo.multiply %v2301, %v2301 : tensor<256x2048x7x7xf32>
    %v2303 = stablehlo.reduce(%v2302 init: %v2296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2304 = stablehlo.broadcast_in_dim %v2303, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2305 = stablehlo.divide %v2304, %v2297 : tensor<256x2048x7x7xf32>
    %v2306 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2307 = stablehlo.add %v2305, %v2306 : tensor<256x2048x7x7xf32>
    %v2308 = stablehlo.rsqrt %v2307 : tensor<256x2048x7x7xf32>
    %v2309 = stablehlo.multiply %v2301, %v2308 : tensor<256x2048x7x7xf32>
    %v2310 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2311 = stablehlo.multiply %v2310, %v2309 : tensor<256x2048x7x7xf32>
    %v2312 = stablehlo.reduce(%v2311 init: %v2296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2313 = stablehlo.reshape %v2043 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.reduce(%v2313 init: %v2314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2316 = stablehlo.reshape %v2203 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2317 = stablehlo.reshape %v1286 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2318 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2319 = stablehlo.compare GT, %v2317, %v2318 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2320 = stablehlo.select %v2319, %v2316, %v2318 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2321 = stablehlo.reshape %v2320 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2322 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2324 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2325 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2326 = stablehlo.reduce(%v2322 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2327 = stablehlo.broadcast_in_dim %v2326, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2328 = stablehlo.divide %v2327, %v2324 : tensor<256x1024x14x14xf32>
    %v2329 = stablehlo.subtract %v2322, %v2328 : tensor<256x1024x14x14xf32>
    %v2330 = stablehlo.multiply %v2329, %v2329 : tensor<256x1024x14x14xf32>
    %v2331 = stablehlo.reduce(%v2330 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2332 = stablehlo.broadcast_in_dim %v2331, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2333 = stablehlo.divide %v2332, %v2324 : tensor<256x1024x14x14xf32>
    %v2334 = stablehlo.add %v2333, %v2325 : tensor<256x1024x14x14xf32>
    %v2335 = stablehlo.rsqrt %v2334 : tensor<256x1024x14x14xf32>
    %v2336 = stablehlo.multiply %v2329, %v2335 : tensor<256x1024x14x14xf32>
    %v2337 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2338 = stablehlo.reshape %v2321 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2339 = stablehlo.multiply %v2337, %v2338 : tensor<256x1024x14x14xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2340, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2342 = stablehlo.multiply %v2336, %v2339 : tensor<256x1024x14x14xf32>
    %v2343 = stablehlo.reduce(%v2342 init: %v2323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2344 = stablehlo.broadcast_in_dim %v2343, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2345 = stablehlo.multiply %v2339, %v2324 : tensor<256x1024x14x14xf32>
    %v2346 = stablehlo.subtract %v2345, %v2341 : tensor<256x1024x14x14xf32>
    %v2347 = stablehlo.multiply %v2336, %v2344 : tensor<256x1024x14x14xf32>
    %v2348 = stablehlo.subtract %v2346, %v2347 : tensor<256x1024x14x14xf32>
    %v2349 = stablehlo.divide %v2335, %v2324 : tensor<256x1024x14x14xf32>
    %v2350 = stablehlo.multiply %v2349, %v2348 : tensor<256x1024x14x14xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2352 = stablehlo.reshape %v2351 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2353 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2354 = stablehlo.transpose %v2353, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2355 = stablehlo.convolution(%v2352, %v2354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2357 = stablehlo.reshape %v2356 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2358 = stablehlo.reshape %v1253 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2359 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2360 = stablehlo.compare GT, %v2358, %v2359 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2361 = stablehlo.select %v2360, %v2357, %v2359 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2363 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2365 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2366 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2367 = stablehlo.reduce(%v2363 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2367, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2369 = stablehlo.divide %v2368, %v2365 : tensor<256x256x14x14xf32>
    %v2370 = stablehlo.subtract %v2363, %v2369 : tensor<256x256x14x14xf32>
    %v2371 = stablehlo.multiply %v2370, %v2370 : tensor<256x256x14x14xf32>
    %v2372 = stablehlo.reduce(%v2371 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2373 = stablehlo.broadcast_in_dim %v2372, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2374 = stablehlo.divide %v2373, %v2365 : tensor<256x256x14x14xf32>
    %v2375 = stablehlo.add %v2374, %v2366 : tensor<256x256x14x14xf32>
    %v2376 = stablehlo.rsqrt %v2375 : tensor<256x256x14x14xf32>
    %v2377 = stablehlo.multiply %v2370, %v2376 : tensor<256x256x14x14xf32>
    %v2378 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2379 = stablehlo.reshape %v2362 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2380 = stablehlo.multiply %v2378, %v2379 : tensor<256x256x14x14xf32>
    %v2381 = stablehlo.reduce(%v2380 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2382 = stablehlo.broadcast_in_dim %v2381, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2383 = stablehlo.multiply %v2377, %v2380 : tensor<256x256x14x14xf32>
    %v2384 = stablehlo.reduce(%v2383 init: %v2364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2385 = stablehlo.broadcast_in_dim %v2384, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2386 = stablehlo.multiply %v2380, %v2365 : tensor<256x256x14x14xf32>
    %v2387 = stablehlo.subtract %v2386, %v2382 : tensor<256x256x14x14xf32>
    %v2388 = stablehlo.multiply %v2377, %v2385 : tensor<256x256x14x14xf32>
    %v2389 = stablehlo.subtract %v2387, %v2388 : tensor<256x256x14x14xf32>
    %v2390 = stablehlo.divide %v2376, %v2365 : tensor<256x256x14x14xf32>
    %v2391 = stablehlo.multiply %v2390, %v2389 : tensor<256x256x14x14xf32>
    %v2392 = stablehlo.reshape %v2391 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2394 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2395 = stablehlo.transpose %v2394, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2396 = stablehlo.convolution(%v2393, %v2395)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2398 = stablehlo.reshape %v2397 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2399 = stablehlo.reshape %v1224 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2400 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2401 = stablehlo.compare GT, %v2399, %v2400 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2402 = stablehlo.select %v2401, %v2398, %v2400 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2404 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2406 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2407 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2408 = stablehlo.reduce(%v2404 init: %v2405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2409 = stablehlo.broadcast_in_dim %v2408, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2410 = stablehlo.divide %v2409, %v2406 : tensor<256x256x14x14xf32>
    %v2411 = stablehlo.subtract %v2404, %v2410 : tensor<256x256x14x14xf32>
    %v2412 = stablehlo.multiply %v2411, %v2411 : tensor<256x256x14x14xf32>
    %v2413 = stablehlo.reduce(%v2412 init: %v2405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2414 = stablehlo.broadcast_in_dim %v2413, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2415 = stablehlo.divide %v2414, %v2406 : tensor<256x256x14x14xf32>
    %v2416 = stablehlo.add %v2415, %v2407 : tensor<256x256x14x14xf32>
    %v2417 = stablehlo.rsqrt %v2416 : tensor<256x256x14x14xf32>
    %v2418 = stablehlo.multiply %v2411, %v2417 : tensor<256x256x14x14xf32>
    %v2419 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2420 = stablehlo.reshape %v2403 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2421 = stablehlo.multiply %v2419, %v2420 : tensor<256x256x14x14xf32>
    %v2422 = stablehlo.reduce(%v2421 init: %v2405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2423 = stablehlo.broadcast_in_dim %v2422, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2424 = stablehlo.multiply %v2418, %v2421 : tensor<256x256x14x14xf32>
    %v2425 = stablehlo.reduce(%v2424 init: %v2405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2426 = stablehlo.broadcast_in_dim %v2425, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2427 = stablehlo.multiply %v2421, %v2406 : tensor<256x256x14x14xf32>
    %v2428 = stablehlo.subtract %v2427, %v2423 : tensor<256x256x14x14xf32>
    %v2429 = stablehlo.multiply %v2418, %v2426 : tensor<256x256x14x14xf32>
    %v2430 = stablehlo.subtract %v2428, %v2429 : tensor<256x256x14x14xf32>
    %v2431 = stablehlo.divide %v2417, %v2406 : tensor<256x256x14x14xf32>
    %v2432 = stablehlo.multiply %v2431, %v2430 : tensor<256x256x14x14xf32>
    %v2433 = stablehlo.reshape %v2432 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2435 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2436 = stablehlo.transpose %v2435, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2437 = stablehlo.convolution(%v2434, %v2436)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2440 = stablehlo.reshape %v2321 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2441 = stablehlo.add %v2439, %v2440 : tensor<256x1024x14x14xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2443 = stablehlo.reshape %v1199 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2444 = stablehlo.reshape %v2433 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2445 = stablehlo.transpose %v2443, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2446 = stablehlo.transpose %v2444, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2447 = stablehlo.convolution(%v2445, %v2446)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2448 = stablehlo.transpose %v2447, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2449 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2451 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2452 = stablehlo.reduce(%v2449 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2453 = stablehlo.broadcast_in_dim %v2452, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2454 = stablehlo.divide %v2453, %v2451 : tensor<256x256x14x14xf32>
    %v2455 = stablehlo.subtract %v2449, %v2454 : tensor<256x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2455, %v2455 : tensor<256x256x14x14xf32>
    %v2457 = stablehlo.reduce(%v2456 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2458 = stablehlo.broadcast_in_dim %v2457, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2459 = stablehlo.divide %v2458, %v2451 : tensor<256x256x14x14xf32>
    %v2460 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2461 = stablehlo.add %v2459, %v2460 : tensor<256x256x14x14xf32>
    %v2462 = stablehlo.rsqrt %v2461 : tensor<256x256x14x14xf32>
    %v2463 = stablehlo.multiply %v2455, %v2462 : tensor<256x256x14x14xf32>
    %v2464 = stablehlo.reshape %v2403 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2465 = stablehlo.multiply %v2464, %v2463 : tensor<256x256x14x14xf32>
    %v2466 = stablehlo.reduce(%v2465 init: %v2450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2467 = stablehlo.reshape %v2403 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2469 = stablehlo.reduce(%v2467 init: %v2468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2470 = stablehlo.reshape %v1228 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2471 = stablehlo.reshape %v2392 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2472 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2473 = stablehlo.transpose %v2471, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2474 = stablehlo.convolution(%v2472, %v2473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2475 = stablehlo.transpose %v2474, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2476 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2478 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2479 = stablehlo.reduce(%v2476 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2479, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2481 = stablehlo.divide %v2480, %v2478 : tensor<256x256x14x14xf32>
    %v2482 = stablehlo.subtract %v2476, %v2481 : tensor<256x256x14x14xf32>
    %v2483 = stablehlo.multiply %v2482, %v2482 : tensor<256x256x14x14xf32>
    %v2484 = stablehlo.reduce(%v2483 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2485 = stablehlo.broadcast_in_dim %v2484, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2486 = stablehlo.divide %v2485, %v2478 : tensor<256x256x14x14xf32>
    %v2487 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2488 = stablehlo.add %v2486, %v2487 : tensor<256x256x14x14xf32>
    %v2489 = stablehlo.rsqrt %v2488 : tensor<256x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2482, %v2489 : tensor<256x256x14x14xf32>
    %v2491 = stablehlo.reshape %v2362 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2492 = stablehlo.multiply %v2491, %v2490 : tensor<256x256x14x14xf32>
    %v2493 = stablehlo.reduce(%v2492 init: %v2477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2494 = stablehlo.reshape %v2362 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2496 = stablehlo.reduce(%v2494 init: %v2495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2497 = stablehlo.reshape %v1257 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2498 = stablehlo.reshape %v2351 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2499 = stablehlo.transpose %v2497, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2500 = stablehlo.transpose %v2498, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2501 = stablehlo.convolution(%v2499, %v2500)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2502 = stablehlo.transpose %v2501, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2503 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2505 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2506 = stablehlo.reduce(%v2503 init: %v2504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2507 = stablehlo.broadcast_in_dim %v2506, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2508 = stablehlo.divide %v2507, %v2505 : tensor<256x1024x14x14xf32>
    %v2509 = stablehlo.subtract %v2503, %v2508 : tensor<256x1024x14x14xf32>
    %v2510 = stablehlo.multiply %v2509, %v2509 : tensor<256x1024x14x14xf32>
    %v2511 = stablehlo.reduce(%v2510 init: %v2504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2512 = stablehlo.broadcast_in_dim %v2511, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2513 = stablehlo.divide %v2512, %v2505 : tensor<256x1024x14x14xf32>
    %v2514 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2515 = stablehlo.add %v2513, %v2514 : tensor<256x1024x14x14xf32>
    %v2516 = stablehlo.rsqrt %v2515 : tensor<256x1024x14x14xf32>
    %v2517 = stablehlo.multiply %v2509, %v2516 : tensor<256x1024x14x14xf32>
    %v2518 = stablehlo.reshape %v2321 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2519 = stablehlo.multiply %v2518, %v2517 : tensor<256x1024x14x14xf32>
    %v2520 = stablehlo.reduce(%v2519 init: %v2504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2521 = stablehlo.reshape %v2321 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2523 = stablehlo.reduce(%v2521 init: %v2522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2524 = stablehlo.reshape %v2442 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2525 = stablehlo.reshape %v1195 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2526 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2527 = stablehlo.compare GT, %v2525, %v2526 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2528 = stablehlo.select %v2527, %v2524, %v2526 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2529 = stablehlo.reshape %v2528 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2530 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2532 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2533 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2534 = stablehlo.reduce(%v2530 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2535 = stablehlo.broadcast_in_dim %v2534, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2536 = stablehlo.divide %v2535, %v2532 : tensor<256x1024x14x14xf32>
    %v2537 = stablehlo.subtract %v2530, %v2536 : tensor<256x1024x14x14xf32>
    %v2538 = stablehlo.multiply %v2537, %v2537 : tensor<256x1024x14x14xf32>
    %v2539 = stablehlo.reduce(%v2538 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2540 = stablehlo.broadcast_in_dim %v2539, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2541 = stablehlo.divide %v2540, %v2532 : tensor<256x1024x14x14xf32>
    %v2542 = stablehlo.add %v2541, %v2533 : tensor<256x1024x14x14xf32>
    %v2543 = stablehlo.rsqrt %v2542 : tensor<256x1024x14x14xf32>
    %v2544 = stablehlo.multiply %v2537, %v2543 : tensor<256x1024x14x14xf32>
    %v2545 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2546 = stablehlo.reshape %v2529 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2547 = stablehlo.multiply %v2545, %v2546 : tensor<256x1024x14x14xf32>
    %v2548 = stablehlo.reduce(%v2547 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2550 = stablehlo.multiply %v2544, %v2547 : tensor<256x1024x14x14xf32>
    %v2551 = stablehlo.reduce(%v2550 init: %v2531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2552 = stablehlo.broadcast_in_dim %v2551, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2553 = stablehlo.multiply %v2547, %v2532 : tensor<256x1024x14x14xf32>
    %v2554 = stablehlo.subtract %v2553, %v2549 : tensor<256x1024x14x14xf32>
    %v2555 = stablehlo.multiply %v2544, %v2552 : tensor<256x1024x14x14xf32>
    %v2556 = stablehlo.subtract %v2554, %v2555 : tensor<256x1024x14x14xf32>
    %v2557 = stablehlo.divide %v2543, %v2532 : tensor<256x1024x14x14xf32>
    %v2558 = stablehlo.multiply %v2557, %v2556 : tensor<256x1024x14x14xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2560 = stablehlo.reshape %v2559 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2561 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2562 = stablehlo.transpose %v2561, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2563 = stablehlo.convolution(%v2560, %v2562)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2564 = stablehlo.reshape %v2563 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2566 = stablehlo.reshape %v1162 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2567 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2568 = stablehlo.compare GT, %v2566, %v2567 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2569 = stablehlo.select %v2568, %v2565, %v2567 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2571 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2573 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2574 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2575 = stablehlo.reduce(%v2571 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2576 = stablehlo.broadcast_in_dim %v2575, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2577 = stablehlo.divide %v2576, %v2573 : tensor<256x256x14x14xf32>
    %v2578 = stablehlo.subtract %v2571, %v2577 : tensor<256x256x14x14xf32>
    %v2579 = stablehlo.multiply %v2578, %v2578 : tensor<256x256x14x14xf32>
    %v2580 = stablehlo.reduce(%v2579 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2581 = stablehlo.broadcast_in_dim %v2580, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2582 = stablehlo.divide %v2581, %v2573 : tensor<256x256x14x14xf32>
    %v2583 = stablehlo.add %v2582, %v2574 : tensor<256x256x14x14xf32>
    %v2584 = stablehlo.rsqrt %v2583 : tensor<256x256x14x14xf32>
    %v2585 = stablehlo.multiply %v2578, %v2584 : tensor<256x256x14x14xf32>
    %v2586 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2587 = stablehlo.reshape %v2570 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2588 = stablehlo.multiply %v2586, %v2587 : tensor<256x256x14x14xf32>
    %v2589 = stablehlo.reduce(%v2588 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2591 = stablehlo.multiply %v2585, %v2588 : tensor<256x256x14x14xf32>
    %v2592 = stablehlo.reduce(%v2591 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2593 = stablehlo.broadcast_in_dim %v2592, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2594 = stablehlo.multiply %v2588, %v2573 : tensor<256x256x14x14xf32>
    %v2595 = stablehlo.subtract %v2594, %v2590 : tensor<256x256x14x14xf32>
    %v2596 = stablehlo.multiply %v2585, %v2593 : tensor<256x256x14x14xf32>
    %v2597 = stablehlo.subtract %v2595, %v2596 : tensor<256x256x14x14xf32>
    %v2598 = stablehlo.divide %v2584, %v2573 : tensor<256x256x14x14xf32>
    %v2599 = stablehlo.multiply %v2598, %v2597 : tensor<256x256x14x14xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2602 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2603 = stablehlo.transpose %v2602, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2604 = stablehlo.convolution(%v2601, %v2603)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2605 = stablehlo.reshape %v2604 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2607 = stablehlo.reshape %v1133 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2608 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2609 = stablehlo.compare GT, %v2607, %v2608 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2610 = stablehlo.select %v2609, %v2606, %v2608 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2612 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2614 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2615 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2616 = stablehlo.reduce(%v2612 init: %v2613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2617 = stablehlo.broadcast_in_dim %v2616, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2618 = stablehlo.divide %v2617, %v2614 : tensor<256x256x14x14xf32>
    %v2619 = stablehlo.subtract %v2612, %v2618 : tensor<256x256x14x14xf32>
    %v2620 = stablehlo.multiply %v2619, %v2619 : tensor<256x256x14x14xf32>
    %v2621 = stablehlo.reduce(%v2620 init: %v2613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2622 = stablehlo.broadcast_in_dim %v2621, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2623 = stablehlo.divide %v2622, %v2614 : tensor<256x256x14x14xf32>
    %v2624 = stablehlo.add %v2623, %v2615 : tensor<256x256x14x14xf32>
    %v2625 = stablehlo.rsqrt %v2624 : tensor<256x256x14x14xf32>
    %v2626 = stablehlo.multiply %v2619, %v2625 : tensor<256x256x14x14xf32>
    %v2627 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2628 = stablehlo.reshape %v2611 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2629 = stablehlo.multiply %v2627, %v2628 : tensor<256x256x14x14xf32>
    %v2630 = stablehlo.reduce(%v2629 init: %v2613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2631 = stablehlo.broadcast_in_dim %v2630, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2632 = stablehlo.multiply %v2626, %v2629 : tensor<256x256x14x14xf32>
    %v2633 = stablehlo.reduce(%v2632 init: %v2613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2635 = stablehlo.multiply %v2629, %v2614 : tensor<256x256x14x14xf32>
    %v2636 = stablehlo.subtract %v2635, %v2631 : tensor<256x256x14x14xf32>
    %v2637 = stablehlo.multiply %v2626, %v2634 : tensor<256x256x14x14xf32>
    %v2638 = stablehlo.subtract %v2636, %v2637 : tensor<256x256x14x14xf32>
    %v2639 = stablehlo.divide %v2625, %v2614 : tensor<256x256x14x14xf32>
    %v2640 = stablehlo.multiply %v2639, %v2638 : tensor<256x256x14x14xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2642 = stablehlo.reshape %v2641 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2643 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2644 = stablehlo.transpose %v2643, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2645 = stablehlo.convolution(%v2642, %v2644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2646 = stablehlo.reshape %v2645 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2648 = stablehlo.reshape %v2529 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2649 = stablehlo.add %v2647, %v2648 : tensor<256x1024x14x14xf32>
    %v2650 = stablehlo.reshape %v2649 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2651 = stablehlo.reshape %v1108 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2652 = stablehlo.reshape %v2641 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2653 = stablehlo.transpose %v2651, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2654 = stablehlo.transpose %v2652, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2655 = stablehlo.convolution(%v2653, %v2654)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2656 = stablehlo.transpose %v2655, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2657 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2659 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2660 = stablehlo.reduce(%v2657 init: %v2658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2660, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2662 = stablehlo.divide %v2661, %v2659 : tensor<256x256x14x14xf32>
    %v2663 = stablehlo.subtract %v2657, %v2662 : tensor<256x256x14x14xf32>
    %v2664 = stablehlo.multiply %v2663, %v2663 : tensor<256x256x14x14xf32>
    %v2665 = stablehlo.reduce(%v2664 init: %v2658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2666 = stablehlo.broadcast_in_dim %v2665, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2667 = stablehlo.divide %v2666, %v2659 : tensor<256x256x14x14xf32>
    %v2668 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2669 = stablehlo.add %v2667, %v2668 : tensor<256x256x14x14xf32>
    %v2670 = stablehlo.rsqrt %v2669 : tensor<256x256x14x14xf32>
    %v2671 = stablehlo.multiply %v2663, %v2670 : tensor<256x256x14x14xf32>
    %v2672 = stablehlo.reshape %v2611 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2673 = stablehlo.multiply %v2672, %v2671 : tensor<256x256x14x14xf32>
    %v2674 = stablehlo.reduce(%v2673 init: %v2658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2675 = stablehlo.reshape %v2611 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2677 = stablehlo.reduce(%v2675 init: %v2676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2678 = stablehlo.reshape %v1137 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2679 = stablehlo.reshape %v2600 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2680 = stablehlo.transpose %v2678, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2681 = stablehlo.transpose %v2679, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2682 = stablehlo.convolution(%v2680, %v2681)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2683 = stablehlo.transpose %v2682, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2684 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2685 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2686 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2687 = stablehlo.reduce(%v2684 init: %v2685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2688 = stablehlo.broadcast_in_dim %v2687, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2689 = stablehlo.divide %v2688, %v2686 : tensor<256x256x14x14xf32>
    %v2690 = stablehlo.subtract %v2684, %v2689 : tensor<256x256x14x14xf32>
    %v2691 = stablehlo.multiply %v2690, %v2690 : tensor<256x256x14x14xf32>
    %v2692 = stablehlo.reduce(%v2691 init: %v2685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2693 = stablehlo.broadcast_in_dim %v2692, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2694 = stablehlo.divide %v2693, %v2686 : tensor<256x256x14x14xf32>
    %v2695 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2696 = stablehlo.add %v2694, %v2695 : tensor<256x256x14x14xf32>
    %v2697 = stablehlo.rsqrt %v2696 : tensor<256x256x14x14xf32>
    %v2698 = stablehlo.multiply %v2690, %v2697 : tensor<256x256x14x14xf32>
    %v2699 = stablehlo.reshape %v2570 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2700 = stablehlo.multiply %v2699, %v2698 : tensor<256x256x14x14xf32>
    %v2701 = stablehlo.reduce(%v2700 init: %v2685) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2702 = stablehlo.reshape %v2570 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2704 = stablehlo.reduce(%v2702 init: %v2703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2705 = stablehlo.reshape %v1166 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2706 = stablehlo.reshape %v2559 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2707 = stablehlo.transpose %v2705, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2708 = stablehlo.transpose %v2706, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2709 = stablehlo.convolution(%v2707, %v2708)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2710 = stablehlo.transpose %v2709, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2711 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2713 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2714 = stablehlo.reduce(%v2711 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2715 = stablehlo.broadcast_in_dim %v2714, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2716 = stablehlo.divide %v2715, %v2713 : tensor<256x1024x14x14xf32>
    %v2717 = stablehlo.subtract %v2711, %v2716 : tensor<256x1024x14x14xf32>
    %v2718 = stablehlo.multiply %v2717, %v2717 : tensor<256x1024x14x14xf32>
    %v2719 = stablehlo.reduce(%v2718 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2720 = stablehlo.broadcast_in_dim %v2719, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2721 = stablehlo.divide %v2720, %v2713 : tensor<256x1024x14x14xf32>
    %v2722 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2723 = stablehlo.add %v2721, %v2722 : tensor<256x1024x14x14xf32>
    %v2724 = stablehlo.rsqrt %v2723 : tensor<256x1024x14x14xf32>
    %v2725 = stablehlo.multiply %v2717, %v2724 : tensor<256x1024x14x14xf32>
    %v2726 = stablehlo.reshape %v2529 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2727 = stablehlo.multiply %v2726, %v2725 : tensor<256x1024x14x14xf32>
    %v2728 = stablehlo.reduce(%v2727 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2729 = stablehlo.reshape %v2529 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2731 = stablehlo.reduce(%v2729 init: %v2730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2732 = stablehlo.reshape %v2650 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2733 = stablehlo.reshape %v1104 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2734 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2735 = stablehlo.compare GT, %v2733, %v2734 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2736 = stablehlo.select %v2735, %v2732, %v2734 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2737 = stablehlo.reshape %v2736 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2738 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2740 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2741 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2742 = stablehlo.reduce(%v2738 init: %v2739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2743 = stablehlo.broadcast_in_dim %v2742, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2744 = stablehlo.divide %v2743, %v2740 : tensor<256x1024x14x14xf32>
    %v2745 = stablehlo.subtract %v2738, %v2744 : tensor<256x1024x14x14xf32>
    %v2746 = stablehlo.multiply %v2745, %v2745 : tensor<256x1024x14x14xf32>
    %v2747 = stablehlo.reduce(%v2746 init: %v2739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2748 = stablehlo.broadcast_in_dim %v2747, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2749 = stablehlo.divide %v2748, %v2740 : tensor<256x1024x14x14xf32>
    %v2750 = stablehlo.add %v2749, %v2741 : tensor<256x1024x14x14xf32>
    %v2751 = stablehlo.rsqrt %v2750 : tensor<256x1024x14x14xf32>
    %v2752 = stablehlo.multiply %v2745, %v2751 : tensor<256x1024x14x14xf32>
    %v2753 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2754 = stablehlo.reshape %v2737 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2755 = stablehlo.multiply %v2753, %v2754 : tensor<256x1024x14x14xf32>
    %v2756 = stablehlo.reduce(%v2755 init: %v2739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2757 = stablehlo.broadcast_in_dim %v2756, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2758 = stablehlo.multiply %v2752, %v2755 : tensor<256x1024x14x14xf32>
    %v2759 = stablehlo.reduce(%v2758 init: %v2739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2759, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2761 = stablehlo.multiply %v2755, %v2740 : tensor<256x1024x14x14xf32>
    %v2762 = stablehlo.subtract %v2761, %v2757 : tensor<256x1024x14x14xf32>
    %v2763 = stablehlo.multiply %v2752, %v2760 : tensor<256x1024x14x14xf32>
    %v2764 = stablehlo.subtract %v2762, %v2763 : tensor<256x1024x14x14xf32>
    %v2765 = stablehlo.divide %v2751, %v2740 : tensor<256x1024x14x14xf32>
    %v2766 = stablehlo.multiply %v2765, %v2764 : tensor<256x1024x14x14xf32>
    %v2767 = stablehlo.reshape %v2766 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2768 = stablehlo.reshape %v2767 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2769 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2770 = stablehlo.transpose %v2769, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2771 = stablehlo.convolution(%v2768, %v2770)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2772 = stablehlo.reshape %v2771 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2773 = stablehlo.reshape %v2772 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2774 = stablehlo.reshape %v1071 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2775 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2776 = stablehlo.compare GT, %v2774, %v2775 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2777 = stablehlo.select %v2776, %v2773, %v2775 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2778 = stablehlo.reshape %v2777 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2779 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2781 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2782 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2783 = stablehlo.reduce(%v2779 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2784 = stablehlo.broadcast_in_dim %v2783, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2785 = stablehlo.divide %v2784, %v2781 : tensor<256x256x14x14xf32>
    %v2786 = stablehlo.subtract %v2779, %v2785 : tensor<256x256x14x14xf32>
    %v2787 = stablehlo.multiply %v2786, %v2786 : tensor<256x256x14x14xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2790 = stablehlo.divide %v2789, %v2781 : tensor<256x256x14x14xf32>
    %v2791 = stablehlo.add %v2790, %v2782 : tensor<256x256x14x14xf32>
    %v2792 = stablehlo.rsqrt %v2791 : tensor<256x256x14x14xf32>
    %v2793 = stablehlo.multiply %v2786, %v2792 : tensor<256x256x14x14xf32>
    %v2794 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2795 = stablehlo.reshape %v2778 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2796 = stablehlo.multiply %v2794, %v2795 : tensor<256x256x14x14xf32>
    %v2797 = stablehlo.reduce(%v2796 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2798 = stablehlo.broadcast_in_dim %v2797, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2799 = stablehlo.multiply %v2793, %v2796 : tensor<256x256x14x14xf32>
    %v2800 = stablehlo.reduce(%v2799 init: %v2780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2800, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2802 = stablehlo.multiply %v2796, %v2781 : tensor<256x256x14x14xf32>
    %v2803 = stablehlo.subtract %v2802, %v2798 : tensor<256x256x14x14xf32>
    %v2804 = stablehlo.multiply %v2793, %v2801 : tensor<256x256x14x14xf32>
    %v2805 = stablehlo.subtract %v2803, %v2804 : tensor<256x256x14x14xf32>
    %v2806 = stablehlo.divide %v2792, %v2781 : tensor<256x256x14x14xf32>
    %v2807 = stablehlo.multiply %v2806, %v2805 : tensor<256x256x14x14xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2810 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2811 = stablehlo.transpose %v2810, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2812 = stablehlo.convolution(%v2809, %v2811)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2813 = stablehlo.reshape %v2812 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2815 = stablehlo.reshape %v1042 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2817 = stablehlo.compare GT, %v2815, %v2816 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2818 = stablehlo.select %v2817, %v2814, %v2816 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2820 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2822 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2823 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2824 = stablehlo.reduce(%v2820 init: %v2821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2825 = stablehlo.broadcast_in_dim %v2824, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2826 = stablehlo.divide %v2825, %v2822 : tensor<256x256x14x14xf32>
    %v2827 = stablehlo.subtract %v2820, %v2826 : tensor<256x256x14x14xf32>
    %v2828 = stablehlo.multiply %v2827, %v2827 : tensor<256x256x14x14xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2831 = stablehlo.divide %v2830, %v2822 : tensor<256x256x14x14xf32>
    %v2832 = stablehlo.add %v2831, %v2823 : tensor<256x256x14x14xf32>
    %v2833 = stablehlo.rsqrt %v2832 : tensor<256x256x14x14xf32>
    %v2834 = stablehlo.multiply %v2827, %v2833 : tensor<256x256x14x14xf32>
    %v2835 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2836 = stablehlo.reshape %v2819 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2837 = stablehlo.multiply %v2835, %v2836 : tensor<256x256x14x14xf32>
    %v2838 = stablehlo.reduce(%v2837 init: %v2821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2839 = stablehlo.broadcast_in_dim %v2838, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2840 = stablehlo.multiply %v2834, %v2837 : tensor<256x256x14x14xf32>
    %v2841 = stablehlo.reduce(%v2840 init: %v2821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2842 = stablehlo.broadcast_in_dim %v2841, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2843 = stablehlo.multiply %v2837, %v2822 : tensor<256x256x14x14xf32>
    %v2844 = stablehlo.subtract %v2843, %v2839 : tensor<256x256x14x14xf32>
    %v2845 = stablehlo.multiply %v2834, %v2842 : tensor<256x256x14x14xf32>
    %v2846 = stablehlo.subtract %v2844, %v2845 : tensor<256x256x14x14xf32>
    %v2847 = stablehlo.divide %v2833, %v2822 : tensor<256x256x14x14xf32>
    %v2848 = stablehlo.multiply %v2847, %v2846 : tensor<256x256x14x14xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2850 = stablehlo.reshape %v2849 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2851 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2852 = stablehlo.transpose %v2851, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2853 = stablehlo.convolution(%v2850, %v2852)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2856 = stablehlo.reshape %v2737 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2857 = stablehlo.add %v2855, %v2856 : tensor<256x1024x14x14xf32>
    %v2858 = stablehlo.reshape %v2857 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2859 = stablehlo.reshape %v1017 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2860 = stablehlo.reshape %v2849 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2861 = stablehlo.transpose %v2859, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2862 = stablehlo.transpose %v2860, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2863 = stablehlo.convolution(%v2861, %v2862)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2864 = stablehlo.transpose %v2863, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2865 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2867 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2868 = stablehlo.reduce(%v2865 init: %v2866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2869 = stablehlo.broadcast_in_dim %v2868, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2870 = stablehlo.divide %v2869, %v2867 : tensor<256x256x14x14xf32>
    %v2871 = stablehlo.subtract %v2865, %v2870 : tensor<256x256x14x14xf32>
    %v2872 = stablehlo.multiply %v2871, %v2871 : tensor<256x256x14x14xf32>
    %v2873 = stablehlo.reduce(%v2872 init: %v2866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2875 = stablehlo.divide %v2874, %v2867 : tensor<256x256x14x14xf32>
    %v2876 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2877 = stablehlo.add %v2875, %v2876 : tensor<256x256x14x14xf32>
    %v2878 = stablehlo.rsqrt %v2877 : tensor<256x256x14x14xf32>
    %v2879 = stablehlo.multiply %v2871, %v2878 : tensor<256x256x14x14xf32>
    %v2880 = stablehlo.reshape %v2819 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2881 = stablehlo.multiply %v2880, %v2879 : tensor<256x256x14x14xf32>
    %v2882 = stablehlo.reduce(%v2881 init: %v2866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2883 = stablehlo.reshape %v2819 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2885 = stablehlo.reduce(%v2883 init: %v2884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2886 = stablehlo.reshape %v1046 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2887 = stablehlo.reshape %v2808 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2888 = stablehlo.transpose %v2886, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2889 = stablehlo.transpose %v2887, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2890 = stablehlo.convolution(%v2888, %v2889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2891 = stablehlo.transpose %v2890, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2892 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2894 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2895 = stablehlo.reduce(%v2892 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2896 = stablehlo.broadcast_in_dim %v2895, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2897 = stablehlo.divide %v2896, %v2894 : tensor<256x256x14x14xf32>
    %v2898 = stablehlo.subtract %v2892, %v2897 : tensor<256x256x14x14xf32>
    %v2899 = stablehlo.multiply %v2898, %v2898 : tensor<256x256x14x14xf32>
    %v2900 = stablehlo.reduce(%v2899 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2901 = stablehlo.broadcast_in_dim %v2900, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2902 = stablehlo.divide %v2901, %v2894 : tensor<256x256x14x14xf32>
    %v2903 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2904 = stablehlo.add %v2902, %v2903 : tensor<256x256x14x14xf32>
    %v2905 = stablehlo.rsqrt %v2904 : tensor<256x256x14x14xf32>
    %v2906 = stablehlo.multiply %v2898, %v2905 : tensor<256x256x14x14xf32>
    %v2907 = stablehlo.reshape %v2778 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2908 = stablehlo.multiply %v2907, %v2906 : tensor<256x256x14x14xf32>
    %v2909 = stablehlo.reduce(%v2908 init: %v2893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2910 = stablehlo.reshape %v2778 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2912 = stablehlo.reduce(%v2910 init: %v2911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2913 = stablehlo.reshape %v1075 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2914 = stablehlo.reshape %v2767 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2915 = stablehlo.transpose %v2913, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2916 = stablehlo.transpose %v2914, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2917 = stablehlo.convolution(%v2915, %v2916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2918 = stablehlo.transpose %v2917, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2919 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2920 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2921 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2922 = stablehlo.reduce(%v2919 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2923 = stablehlo.broadcast_in_dim %v2922, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2924 = stablehlo.divide %v2923, %v2921 : tensor<256x1024x14x14xf32>
    %v2925 = stablehlo.subtract %v2919, %v2924 : tensor<256x1024x14x14xf32>
    %v2926 = stablehlo.multiply %v2925, %v2925 : tensor<256x1024x14x14xf32>
    %v2927 = stablehlo.reduce(%v2926 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2928 = stablehlo.broadcast_in_dim %v2927, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2929 = stablehlo.divide %v2928, %v2921 : tensor<256x1024x14x14xf32>
    %v2930 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2931 = stablehlo.add %v2929, %v2930 : tensor<256x1024x14x14xf32>
    %v2932 = stablehlo.rsqrt %v2931 : tensor<256x1024x14x14xf32>
    %v2933 = stablehlo.multiply %v2925, %v2932 : tensor<256x1024x14x14xf32>
    %v2934 = stablehlo.reshape %v2737 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2935 = stablehlo.multiply %v2934, %v2933 : tensor<256x1024x14x14xf32>
    %v2936 = stablehlo.reduce(%v2935 init: %v2920) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2937 = stablehlo.reshape %v2737 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2939 = stablehlo.reduce(%v2937 init: %v2938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2940 = stablehlo.reshape %v2858 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2941 = stablehlo.reshape %v1013 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2942 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v2943 = stablehlo.compare GT, %v2941, %v2942 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v2944 = stablehlo.select %v2943, %v2940, %v2942 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v2945 = stablehlo.reshape %v2944 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2946 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2948 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2949 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2950 = stablehlo.reduce(%v2946 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2951 = stablehlo.broadcast_in_dim %v2950, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2952 = stablehlo.divide %v2951, %v2948 : tensor<256x1024x14x14xf32>
    %v2953 = stablehlo.subtract %v2946, %v2952 : tensor<256x1024x14x14xf32>
    %v2954 = stablehlo.multiply %v2953, %v2953 : tensor<256x1024x14x14xf32>
    %v2955 = stablehlo.reduce(%v2954 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2956 = stablehlo.broadcast_in_dim %v2955, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2957 = stablehlo.divide %v2956, %v2948 : tensor<256x1024x14x14xf32>
    %v2958 = stablehlo.add %v2957, %v2949 : tensor<256x1024x14x14xf32>
    %v2959 = stablehlo.rsqrt %v2958 : tensor<256x1024x14x14xf32>
    %v2960 = stablehlo.multiply %v2953, %v2959 : tensor<256x1024x14x14xf32>
    %v2961 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2962 = stablehlo.reshape %v2945 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2963 = stablehlo.multiply %v2961, %v2962 : tensor<256x1024x14x14xf32>
    %v2964 = stablehlo.reduce(%v2963 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2965 = stablehlo.broadcast_in_dim %v2964, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2966 = stablehlo.multiply %v2960, %v2963 : tensor<256x1024x14x14xf32>
    %v2967 = stablehlo.reduce(%v2966 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2968 = stablehlo.broadcast_in_dim %v2967, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2969 = stablehlo.multiply %v2963, %v2948 : tensor<256x1024x14x14xf32>
    %v2970 = stablehlo.subtract %v2969, %v2965 : tensor<256x1024x14x14xf32>
    %v2971 = stablehlo.multiply %v2960, %v2968 : tensor<256x1024x14x14xf32>
    %v2972 = stablehlo.subtract %v2970, %v2971 : tensor<256x1024x14x14xf32>
    %v2973 = stablehlo.divide %v2959, %v2948 : tensor<256x1024x14x14xf32>
    %v2974 = stablehlo.multiply %v2973, %v2972 : tensor<256x1024x14x14xf32>
    %v2975 = stablehlo.reshape %v2974 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2976 = stablehlo.reshape %v2975 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2977 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2978 = stablehlo.transpose %v2977, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2979 = stablehlo.convolution(%v2976, %v2978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2980 = stablehlo.reshape %v2979 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2981 = stablehlo.reshape %v2980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2982 = stablehlo.reshape %v980 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2983 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v2984 = stablehlo.compare GT, %v2982, %v2983 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v2985 = stablehlo.select %v2984, %v2981, %v2983 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v2986 = stablehlo.reshape %v2985 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2987 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2988 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2989 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2990 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2991 = stablehlo.reduce(%v2987 init: %v2988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2992 = stablehlo.broadcast_in_dim %v2991, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2993 = stablehlo.divide %v2992, %v2989 : tensor<256x256x14x14xf32>
    %v2994 = stablehlo.subtract %v2987, %v2993 : tensor<256x256x14x14xf32>
    %v2995 = stablehlo.multiply %v2994, %v2994 : tensor<256x256x14x14xf32>
    %v2996 = stablehlo.reduce(%v2995 init: %v2988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2997 = stablehlo.broadcast_in_dim %v2996, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2998 = stablehlo.divide %v2997, %v2989 : tensor<256x256x14x14xf32>
    %v2999 = stablehlo.add %v2998, %v2990 : tensor<256x256x14x14xf32>
    %v3000 = stablehlo.rsqrt %v2999 : tensor<256x256x14x14xf32>
    %v3001 = stablehlo.multiply %v2994, %v3000 : tensor<256x256x14x14xf32>
    %v3002 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3003 = stablehlo.reshape %v2986 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3004 = stablehlo.multiply %v3002, %v3003 : tensor<256x256x14x14xf32>
    %v3005 = stablehlo.reduce(%v3004 init: %v2988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3006 = stablehlo.broadcast_in_dim %v3005, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3007 = stablehlo.multiply %v3001, %v3004 : tensor<256x256x14x14xf32>
    %v3008 = stablehlo.reduce(%v3007 init: %v2988) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3009 = stablehlo.broadcast_in_dim %v3008, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3010 = stablehlo.multiply %v3004, %v2989 : tensor<256x256x14x14xf32>
    %v3011 = stablehlo.subtract %v3010, %v3006 : tensor<256x256x14x14xf32>
    %v3012 = stablehlo.multiply %v3001, %v3009 : tensor<256x256x14x14xf32>
    %v3013 = stablehlo.subtract %v3011, %v3012 : tensor<256x256x14x14xf32>
    %v3014 = stablehlo.divide %v3000, %v2989 : tensor<256x256x14x14xf32>
    %v3015 = stablehlo.multiply %v3014, %v3013 : tensor<256x256x14x14xf32>
    %v3016 = stablehlo.reshape %v3015 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3017 = stablehlo.reshape %v3016 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3018 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3019 = stablehlo.transpose %v3018, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3020 = stablehlo.convolution(%v3017, %v3019)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v3021 = stablehlo.reshape %v3020 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3023 = stablehlo.reshape %v951 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3024 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3025 = stablehlo.compare GT, %v3023, %v3024 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3026 = stablehlo.select %v3025, %v3022, %v3024 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3028 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3030 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3031 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3032 = stablehlo.reduce(%v3028 init: %v3029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3033 = stablehlo.broadcast_in_dim %v3032, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3034 = stablehlo.divide %v3033, %v3030 : tensor<256x256x14x14xf32>
    %v3035 = stablehlo.subtract %v3028, %v3034 : tensor<256x256x14x14xf32>
    %v3036 = stablehlo.multiply %v3035, %v3035 : tensor<256x256x14x14xf32>
    %v3037 = stablehlo.reduce(%v3036 init: %v3029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3038 = stablehlo.broadcast_in_dim %v3037, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3039 = stablehlo.divide %v3038, %v3030 : tensor<256x256x14x14xf32>
    %v3040 = stablehlo.add %v3039, %v3031 : tensor<256x256x14x14xf32>
    %v3041 = stablehlo.rsqrt %v3040 : tensor<256x256x14x14xf32>
    %v3042 = stablehlo.multiply %v3035, %v3041 : tensor<256x256x14x14xf32>
    %v3043 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3044 = stablehlo.reshape %v3027 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3045 = stablehlo.multiply %v3043, %v3044 : tensor<256x256x14x14xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3048 = stablehlo.multiply %v3042, %v3045 : tensor<256x256x14x14xf32>
    %v3049 = stablehlo.reduce(%v3048 init: %v3029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3050 = stablehlo.broadcast_in_dim %v3049, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3051 = stablehlo.multiply %v3045, %v3030 : tensor<256x256x14x14xf32>
    %v3052 = stablehlo.subtract %v3051, %v3047 : tensor<256x256x14x14xf32>
    %v3053 = stablehlo.multiply %v3042, %v3050 : tensor<256x256x14x14xf32>
    %v3054 = stablehlo.subtract %v3052, %v3053 : tensor<256x256x14x14xf32>
    %v3055 = stablehlo.divide %v3041, %v3030 : tensor<256x256x14x14xf32>
    %v3056 = stablehlo.multiply %v3055, %v3054 : tensor<256x256x14x14xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3058 = stablehlo.reshape %v3057 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3059 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3060 = stablehlo.transpose %v3059, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3061 = stablehlo.convolution(%v3058, %v3060)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v3062 = stablehlo.reshape %v3061 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3063 = stablehlo.reshape %v3062 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3064 = stablehlo.reshape %v2945 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3065 = stablehlo.add %v3063, %v3064 : tensor<256x1024x14x14xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3067 = stablehlo.reshape %v926 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3068 = stablehlo.reshape %v3057 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3069 = stablehlo.transpose %v3067, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3070 = stablehlo.transpose %v3068, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3071 = stablehlo.convolution(%v3069, %v3070)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3072 = stablehlo.transpose %v3071, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3073 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3074 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3075 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3076 = stablehlo.reduce(%v3073 init: %v3074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3077 = stablehlo.broadcast_in_dim %v3076, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3078 = stablehlo.divide %v3077, %v3075 : tensor<256x256x14x14xf32>
    %v3079 = stablehlo.subtract %v3073, %v3078 : tensor<256x256x14x14xf32>
    %v3080 = stablehlo.multiply %v3079, %v3079 : tensor<256x256x14x14xf32>
    %v3081 = stablehlo.reduce(%v3080 init: %v3074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3082 = stablehlo.broadcast_in_dim %v3081, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3083 = stablehlo.divide %v3082, %v3075 : tensor<256x256x14x14xf32>
    %v3084 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3085 = stablehlo.add %v3083, %v3084 : tensor<256x256x14x14xf32>
    %v3086 = stablehlo.rsqrt %v3085 : tensor<256x256x14x14xf32>
    %v3087 = stablehlo.multiply %v3079, %v3086 : tensor<256x256x14x14xf32>
    %v3088 = stablehlo.reshape %v3027 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3089 = stablehlo.multiply %v3088, %v3087 : tensor<256x256x14x14xf32>
    %v3090 = stablehlo.reduce(%v3089 init: %v3074) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3091 = stablehlo.reshape %v3027 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3093 = stablehlo.reduce(%v3091 init: %v3092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3094 = stablehlo.reshape %v955 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3095 = stablehlo.reshape %v3016 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3096 = stablehlo.transpose %v3094, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3097 = stablehlo.transpose %v3095, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3098 = stablehlo.convolution(%v3096, %v3097)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3099 = stablehlo.transpose %v3098, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3100 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3102 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3103 = stablehlo.reduce(%v3100 init: %v3101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3104 = stablehlo.broadcast_in_dim %v3103, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3105 = stablehlo.divide %v3104, %v3102 : tensor<256x256x14x14xf32>
    %v3106 = stablehlo.subtract %v3100, %v3105 : tensor<256x256x14x14xf32>
    %v3107 = stablehlo.multiply %v3106, %v3106 : tensor<256x256x14x14xf32>
    %v3108 = stablehlo.reduce(%v3107 init: %v3101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3109 = stablehlo.broadcast_in_dim %v3108, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3110 = stablehlo.divide %v3109, %v3102 : tensor<256x256x14x14xf32>
    %v3111 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3112 = stablehlo.add %v3110, %v3111 : tensor<256x256x14x14xf32>
    %v3113 = stablehlo.rsqrt %v3112 : tensor<256x256x14x14xf32>
    %v3114 = stablehlo.multiply %v3106, %v3113 : tensor<256x256x14x14xf32>
    %v3115 = stablehlo.reshape %v2986 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3116 = stablehlo.multiply %v3115, %v3114 : tensor<256x256x14x14xf32>
    %v3117 = stablehlo.reduce(%v3116 init: %v3101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3118 = stablehlo.reshape %v2986 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3120 = stablehlo.reduce(%v3118 init: %v3119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3121 = stablehlo.reshape %v984 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3122 = stablehlo.reshape %v2975 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3123 = stablehlo.transpose %v3121, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3124 = stablehlo.transpose %v3122, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3125 = stablehlo.convolution(%v3123, %v3124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3126 = stablehlo.transpose %v3125, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3127 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3129 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3130 = stablehlo.reduce(%v3127 init: %v3128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3131 = stablehlo.broadcast_in_dim %v3130, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3132 = stablehlo.divide %v3131, %v3129 : tensor<256x1024x14x14xf32>
    %v3133 = stablehlo.subtract %v3127, %v3132 : tensor<256x1024x14x14xf32>
    %v3134 = stablehlo.multiply %v3133, %v3133 : tensor<256x1024x14x14xf32>
    %v3135 = stablehlo.reduce(%v3134 init: %v3128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3136 = stablehlo.broadcast_in_dim %v3135, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3137 = stablehlo.divide %v3136, %v3129 : tensor<256x1024x14x14xf32>
    %v3138 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3139 = stablehlo.add %v3137, %v3138 : tensor<256x1024x14x14xf32>
    %v3140 = stablehlo.rsqrt %v3139 : tensor<256x1024x14x14xf32>
    %v3141 = stablehlo.multiply %v3133, %v3140 : tensor<256x1024x14x14xf32>
    %v3142 = stablehlo.reshape %v2945 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3143 = stablehlo.multiply %v3142, %v3141 : tensor<256x1024x14x14xf32>
    %v3144 = stablehlo.reduce(%v3143 init: %v3128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3145 = stablehlo.reshape %v2945 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3147 = stablehlo.reduce(%v3145 init: %v3146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3148 = stablehlo.reshape %v3066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3149 = stablehlo.reshape %v922 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3150 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v3151 = stablehlo.compare GT, %v3149, %v3150 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v3152 = stablehlo.select %v3151, %v3148, %v3150 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v3153 = stablehlo.reshape %v3152 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3154 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3155 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3156 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3157 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3158 = stablehlo.reduce(%v3154 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3159 = stablehlo.broadcast_in_dim %v3158, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3160 = stablehlo.divide %v3159, %v3156 : tensor<256x1024x14x14xf32>
    %v3161 = stablehlo.subtract %v3154, %v3160 : tensor<256x1024x14x14xf32>
    %v3162 = stablehlo.multiply %v3161, %v3161 : tensor<256x1024x14x14xf32>
    %v3163 = stablehlo.reduce(%v3162 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3164 = stablehlo.broadcast_in_dim %v3163, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3165 = stablehlo.divide %v3164, %v3156 : tensor<256x1024x14x14xf32>
    %v3166 = stablehlo.add %v3165, %v3157 : tensor<256x1024x14x14xf32>
    %v3167 = stablehlo.rsqrt %v3166 : tensor<256x1024x14x14xf32>
    %v3168 = stablehlo.multiply %v3161, %v3167 : tensor<256x1024x14x14xf32>
    %v3169 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3170 = stablehlo.reshape %v3153 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3171 = stablehlo.multiply %v3169, %v3170 : tensor<256x1024x14x14xf32>
    %v3172 = stablehlo.reduce(%v3171 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3173 = stablehlo.broadcast_in_dim %v3172, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3174 = stablehlo.multiply %v3168, %v3171 : tensor<256x1024x14x14xf32>
    %v3175 = stablehlo.reduce(%v3174 init: %v3155) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3176 = stablehlo.broadcast_in_dim %v3175, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3177 = stablehlo.multiply %v3171, %v3156 : tensor<256x1024x14x14xf32>
    %v3178 = stablehlo.subtract %v3177, %v3173 : tensor<256x1024x14x14xf32>
    %v3179 = stablehlo.multiply %v3168, %v3176 : tensor<256x1024x14x14xf32>
    %v3180 = stablehlo.subtract %v3178, %v3179 : tensor<256x1024x14x14xf32>
    %v3181 = stablehlo.divide %v3167, %v3156 : tensor<256x1024x14x14xf32>
    %v3182 = stablehlo.multiply %v3181, %v3180 : tensor<256x1024x14x14xf32>
    %v3183 = stablehlo.reshape %v3182 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3184 = stablehlo.reshape %v3183 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3185 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3186 = stablehlo.transpose %v3185, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3187 = stablehlo.convolution(%v3184, %v3186)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v3188 = stablehlo.reshape %v3187 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3189 = stablehlo.reshape %v3188 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3190 = stablehlo.reshape %v889 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3191 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3192 = stablehlo.compare GT, %v3190, %v3191 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3193 = stablehlo.select %v3192, %v3189, %v3191 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3194 = stablehlo.reshape %v3193 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3195 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3197 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3198 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3199 = stablehlo.reduce(%v3195 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3200 = stablehlo.broadcast_in_dim %v3199, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3201 = stablehlo.divide %v3200, %v3197 : tensor<256x256x14x14xf32>
    %v3202 = stablehlo.subtract %v3195, %v3201 : tensor<256x256x14x14xf32>
    %v3203 = stablehlo.multiply %v3202, %v3202 : tensor<256x256x14x14xf32>
    %v3204 = stablehlo.reduce(%v3203 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3205 = stablehlo.broadcast_in_dim %v3204, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3206 = stablehlo.divide %v3205, %v3197 : tensor<256x256x14x14xf32>
    %v3207 = stablehlo.add %v3206, %v3198 : tensor<256x256x14x14xf32>
    %v3208 = stablehlo.rsqrt %v3207 : tensor<256x256x14x14xf32>
    %v3209 = stablehlo.multiply %v3202, %v3208 : tensor<256x256x14x14xf32>
    %v3210 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3211 = stablehlo.reshape %v3194 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3212 = stablehlo.multiply %v3210, %v3211 : tensor<256x256x14x14xf32>
    %v3213 = stablehlo.reduce(%v3212 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3215 = stablehlo.multiply %v3209, %v3212 : tensor<256x256x14x14xf32>
    %v3216 = stablehlo.reduce(%v3215 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3217 = stablehlo.broadcast_in_dim %v3216, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3218 = stablehlo.multiply %v3212, %v3197 : tensor<256x256x14x14xf32>
    %v3219 = stablehlo.subtract %v3218, %v3214 : tensor<256x256x14x14xf32>
    %v3220 = stablehlo.multiply %v3209, %v3217 : tensor<256x256x14x14xf32>
    %v3221 = stablehlo.subtract %v3219, %v3220 : tensor<256x256x14x14xf32>
    %v3222 = stablehlo.divide %v3208, %v3197 : tensor<256x256x14x14xf32>
    %v3223 = stablehlo.multiply %v3222, %v3221 : tensor<256x256x14x14xf32>
    %v3224 = stablehlo.reshape %v3223 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3226 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3227 = stablehlo.transpose %v3226, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3228 = stablehlo.convolution(%v3225, %v3227)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3231 = stablehlo.reshape %v860 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3233 = stablehlo.compare GT, %v3231, %v3232 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3234 = stablehlo.select %v3233, %v3230, %v3232 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3235 = stablehlo.reshape %v3234 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3236 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3238 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3239 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3240 = stablehlo.reduce(%v3236 init: %v3237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3242 = stablehlo.divide %v3241, %v3238 : tensor<256x256x14x14xf32>
    %v3243 = stablehlo.subtract %v3236, %v3242 : tensor<256x256x14x14xf32>
    %v3244 = stablehlo.multiply %v3243, %v3243 : tensor<256x256x14x14xf32>
    %v3245 = stablehlo.reduce(%v3244 init: %v3237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3246 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3247 = stablehlo.divide %v3246, %v3238 : tensor<256x256x14x14xf32>
    %v3248 = stablehlo.add %v3247, %v3239 : tensor<256x256x14x14xf32>
    %v3249 = stablehlo.rsqrt %v3248 : tensor<256x256x14x14xf32>
    %v3250 = stablehlo.multiply %v3243, %v3249 : tensor<256x256x14x14xf32>
    %v3251 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3252 = stablehlo.reshape %v3235 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3253 = stablehlo.multiply %v3251, %v3252 : tensor<256x256x14x14xf32>
    %v3254 = stablehlo.reduce(%v3253 init: %v3237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3255 = stablehlo.broadcast_in_dim %v3254, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3256 = stablehlo.multiply %v3250, %v3253 : tensor<256x256x14x14xf32>
    %v3257 = stablehlo.reduce(%v3256 init: %v3237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3258 = stablehlo.broadcast_in_dim %v3257, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3259 = stablehlo.multiply %v3253, %v3238 : tensor<256x256x14x14xf32>
    %v3260 = stablehlo.subtract %v3259, %v3255 : tensor<256x256x14x14xf32>
    %v3261 = stablehlo.multiply %v3250, %v3258 : tensor<256x256x14x14xf32>
    %v3262 = stablehlo.subtract %v3260, %v3261 : tensor<256x256x14x14xf32>
    %v3263 = stablehlo.divide %v3249, %v3238 : tensor<256x256x14x14xf32>
    %v3264 = stablehlo.multiply %v3263, %v3262 : tensor<256x256x14x14xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3267 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3268 = stablehlo.transpose %v3267, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3269 = stablehlo.convolution(%v3266, %v3268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v3270 = stablehlo.reshape %v3269 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3271 = stablehlo.reshape %v3270 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3272 = stablehlo.reshape %v3153 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3273 = stablehlo.add %v3271, %v3272 : tensor<256x1024x14x14xf32>
    %v3274 = stablehlo.reshape %v3273 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3275 = stablehlo.reshape %v835 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3276 = stablehlo.reshape %v3265 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3277 = stablehlo.transpose %v3275, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3278 = stablehlo.transpose %v3276, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3279 = stablehlo.convolution(%v3277, %v3278)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3280 = stablehlo.transpose %v3279, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3281 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3282 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3283 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3284 = stablehlo.reduce(%v3281 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3286 = stablehlo.divide %v3285, %v3283 : tensor<256x256x14x14xf32>
    %v3287 = stablehlo.subtract %v3281, %v3286 : tensor<256x256x14x14xf32>
    %v3288 = stablehlo.multiply %v3287, %v3287 : tensor<256x256x14x14xf32>
    %v3289 = stablehlo.reduce(%v3288 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3290 = stablehlo.broadcast_in_dim %v3289, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3291 = stablehlo.divide %v3290, %v3283 : tensor<256x256x14x14xf32>
    %v3292 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3293 = stablehlo.add %v3291, %v3292 : tensor<256x256x14x14xf32>
    %v3294 = stablehlo.rsqrt %v3293 : tensor<256x256x14x14xf32>
    %v3295 = stablehlo.multiply %v3287, %v3294 : tensor<256x256x14x14xf32>
    %v3296 = stablehlo.reshape %v3235 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3297 = stablehlo.multiply %v3296, %v3295 : tensor<256x256x14x14xf32>
    %v3298 = stablehlo.reduce(%v3297 init: %v3282) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3299 = stablehlo.reshape %v3235 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3301 = stablehlo.reduce(%v3299 init: %v3300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3302 = stablehlo.reshape %v864 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3303 = stablehlo.reshape %v3224 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3304 = stablehlo.transpose %v3302, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3305 = stablehlo.transpose %v3303, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3306 = stablehlo.convolution(%v3304, %v3305)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3307 = stablehlo.transpose %v3306, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3308 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3310 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3311 = stablehlo.reduce(%v3308 init: %v3309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3313 = stablehlo.divide %v3312, %v3310 : tensor<256x256x14x14xf32>
    %v3314 = stablehlo.subtract %v3308, %v3313 : tensor<256x256x14x14xf32>
    %v3315 = stablehlo.multiply %v3314, %v3314 : tensor<256x256x14x14xf32>
    %v3316 = stablehlo.reduce(%v3315 init: %v3309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3317 = stablehlo.broadcast_in_dim %v3316, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3318 = stablehlo.divide %v3317, %v3310 : tensor<256x256x14x14xf32>
    %v3319 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3320 = stablehlo.add %v3318, %v3319 : tensor<256x256x14x14xf32>
    %v3321 = stablehlo.rsqrt %v3320 : tensor<256x256x14x14xf32>
    %v3322 = stablehlo.multiply %v3314, %v3321 : tensor<256x256x14x14xf32>
    %v3323 = stablehlo.reshape %v3194 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3324 = stablehlo.multiply %v3323, %v3322 : tensor<256x256x14x14xf32>
    %v3325 = stablehlo.reduce(%v3324 init: %v3309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3326 = stablehlo.reshape %v3194 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3328 = stablehlo.reduce(%v3326 init: %v3327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3329 = stablehlo.reshape %v893 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3330 = stablehlo.reshape %v3183 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3331 = stablehlo.transpose %v3329, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3332 = stablehlo.transpose %v3330, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3333 = stablehlo.convolution(%v3331, %v3332)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3334 = stablehlo.transpose %v3333, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3335 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3337 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3338 = stablehlo.reduce(%v3335 init: %v3336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3340 = stablehlo.divide %v3339, %v3337 : tensor<256x1024x14x14xf32>
    %v3341 = stablehlo.subtract %v3335, %v3340 : tensor<256x1024x14x14xf32>
    %v3342 = stablehlo.multiply %v3341, %v3341 : tensor<256x1024x14x14xf32>
    %v3343 = stablehlo.reduce(%v3342 init: %v3336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3344 = stablehlo.broadcast_in_dim %v3343, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3345 = stablehlo.divide %v3344, %v3337 : tensor<256x1024x14x14xf32>
    %v3346 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3347 = stablehlo.add %v3345, %v3346 : tensor<256x1024x14x14xf32>
    %v3348 = stablehlo.rsqrt %v3347 : tensor<256x1024x14x14xf32>
    %v3349 = stablehlo.multiply %v3341, %v3348 : tensor<256x1024x14x14xf32>
    %v3350 = stablehlo.reshape %v3153 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3351 = stablehlo.multiply %v3350, %v3349 : tensor<256x1024x14x14xf32>
    %v3352 = stablehlo.reduce(%v3351 init: %v3336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3353 = stablehlo.reshape %v3153 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3354 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3355 = stablehlo.reduce(%v3353 init: %v3354) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3356 = stablehlo.reshape %v3274 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3357 = stablehlo.reshape %v831 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3358 = stablehlo.constant dense<0.0> : tensor<256x1024x14x14xf32>
    %v3359 = stablehlo.compare GT, %v3357, %v3358 : (tensor<256x1024x14x14xf32>, tensor<256x1024x14x14xf32>) -> tensor<256x1024x14x14xi1>
    %v3360 = stablehlo.select %v3359, %v3356, %v3358 : tensor<256x1024x14x14xi1>, tensor<256x1024x14x14xf32>
    %v3361 = stablehlo.reshape %v3360 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3362 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3364 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3365 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3366 = stablehlo.reduce(%v3362 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3367 = stablehlo.broadcast_in_dim %v3366, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3368 = stablehlo.divide %v3367, %v3364 : tensor<256x1024x14x14xf32>
    %v3369 = stablehlo.subtract %v3362, %v3368 : tensor<256x1024x14x14xf32>
    %v3370 = stablehlo.multiply %v3369, %v3369 : tensor<256x1024x14x14xf32>
    %v3371 = stablehlo.reduce(%v3370 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3372 = stablehlo.broadcast_in_dim %v3371, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3373 = stablehlo.divide %v3372, %v3364 : tensor<256x1024x14x14xf32>
    %v3374 = stablehlo.add %v3373, %v3365 : tensor<256x1024x14x14xf32>
    %v3375 = stablehlo.rsqrt %v3374 : tensor<256x1024x14x14xf32>
    %v3376 = stablehlo.multiply %v3369, %v3375 : tensor<256x1024x14x14xf32>
    %v3377 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3378 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3379 = stablehlo.multiply %v3377, %v3378 : tensor<256x1024x14x14xf32>
    %v3380 = stablehlo.reduce(%v3379 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3382 = stablehlo.multiply %v3376, %v3379 : tensor<256x1024x14x14xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3383, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3385 = stablehlo.multiply %v3379, %v3364 : tensor<256x1024x14x14xf32>
    %v3386 = stablehlo.subtract %v3385, %v3381 : tensor<256x1024x14x14xf32>
    %v3387 = stablehlo.multiply %v3376, %v3384 : tensor<256x1024x14x14xf32>
    %v3388 = stablehlo.subtract %v3386, %v3387 : tensor<256x1024x14x14xf32>
    %v3389 = stablehlo.divide %v3375, %v3364 : tensor<256x1024x14x14xf32>
    %v3390 = stablehlo.multiply %v3389, %v3388 : tensor<256x1024x14x14xf32>
    %v3391 = stablehlo.reshape %v3390 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3392 = stablehlo.reshape %v3391 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3393 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3394 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3395 = stablehlo.convolution(%v3392, %v3394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v3396 = stablehlo.reshape %v3395 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3398 = stablehlo.reshape %v773 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3399 = stablehlo.constant dense<0.0> : tensor<256x256x14x14xf32>
    %v3400 = stablehlo.compare GT, %v3398, %v3399 : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xi1>
    %v3401 = stablehlo.select %v3400, %v3397, %v3399 : tensor<256x256x14x14xi1>, tensor<256x256x14x14xf32>
    %v3402 = stablehlo.reshape %v3401 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3403 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3405 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3406 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3407 = stablehlo.reduce(%v3403 init: %v3404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3408 = stablehlo.broadcast_in_dim %v3407, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3409 = stablehlo.divide %v3408, %v3405 : tensor<256x256x14x14xf32>
    %v3410 = stablehlo.subtract %v3403, %v3409 : tensor<256x256x14x14xf32>
    %v3411 = stablehlo.multiply %v3410, %v3410 : tensor<256x256x14x14xf32>
    %v3412 = stablehlo.reduce(%v3411 init: %v3404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3413 = stablehlo.broadcast_in_dim %v3412, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3414 = stablehlo.divide %v3413, %v3405 : tensor<256x256x14x14xf32>
    %v3415 = stablehlo.add %v3414, %v3406 : tensor<256x256x14x14xf32>
    %v3416 = stablehlo.rsqrt %v3415 : tensor<256x256x14x14xf32>
    %v3417 = stablehlo.multiply %v3410, %v3416 : tensor<256x256x14x14xf32>
    %v3418 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3419 = stablehlo.reshape %v3402 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3420 = stablehlo.multiply %v3418, %v3419 : tensor<256x256x14x14xf32>
    %v3421 = stablehlo.reduce(%v3420 init: %v3404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3422 = stablehlo.broadcast_in_dim %v3421, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3423 = stablehlo.multiply %v3417, %v3420 : tensor<256x256x14x14xf32>
    %v3424 = stablehlo.reduce(%v3423 init: %v3404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3425 = stablehlo.broadcast_in_dim %v3424, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3426 = stablehlo.multiply %v3420, %v3405 : tensor<256x256x14x14xf32>
    %v3427 = stablehlo.subtract %v3426, %v3422 : tensor<256x256x14x14xf32>
    %v3428 = stablehlo.multiply %v3417, %v3425 : tensor<256x256x14x14xf32>
    %v3429 = stablehlo.subtract %v3427, %v3428 : tensor<256x256x14x14xf32>
    %v3430 = stablehlo.divide %v3416, %v3405 : tensor<256x256x14x14xf32>
    %v3431 = stablehlo.multiply %v3430, %v3429 : tensor<256x256x14x14xf32>
    %v3432 = stablehlo.reshape %v3431 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3433 = stablehlo.reshape %v3432 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3435 = stablehlo.pad %v3433, %v3434, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v3436 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3437 = stablehlo.transpose %v3436, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3438 = stablehlo.convolution(%v3435, %v3437)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x28x28xf32>
    %v3439 = stablehlo.reshape %v3438 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3440 = stablehlo.reshape %v3439 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3441 = stablehlo.reshape %v744 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3442 = stablehlo.constant dense<0.0> : tensor<256x256x28x28xf32>
    %v3443 = stablehlo.compare GT, %v3441, %v3442 : (tensor<256x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xi1>
    %v3444 = stablehlo.select %v3443, %v3440, %v3442 : tensor<256x256x28x28xi1>, tensor<256x256x28x28xf32>
    %v3445 = stablehlo.reshape %v3444 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3446 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3448 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3449 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3450 = stablehlo.reduce(%v3446 init: %v3447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3451 = stablehlo.broadcast_in_dim %v3450, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3452 = stablehlo.divide %v3451, %v3448 : tensor<256x256x28x28xf32>
    %v3453 = stablehlo.subtract %v3446, %v3452 : tensor<256x256x28x28xf32>
    %v3454 = stablehlo.multiply %v3453, %v3453 : tensor<256x256x28x28xf32>
    %v3455 = stablehlo.reduce(%v3454 init: %v3447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3456 = stablehlo.broadcast_in_dim %v3455, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3457 = stablehlo.divide %v3456, %v3448 : tensor<256x256x28x28xf32>
    %v3458 = stablehlo.add %v3457, %v3449 : tensor<256x256x28x28xf32>
    %v3459 = stablehlo.rsqrt %v3458 : tensor<256x256x28x28xf32>
    %v3460 = stablehlo.multiply %v3453, %v3459 : tensor<256x256x28x28xf32>
    %v3461 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3462 = stablehlo.reshape %v3445 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3463 = stablehlo.multiply %v3461, %v3462 : tensor<256x256x28x28xf32>
    %v3464 = stablehlo.reduce(%v3463 init: %v3447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3465 = stablehlo.broadcast_in_dim %v3464, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3466 = stablehlo.multiply %v3460, %v3463 : tensor<256x256x28x28xf32>
    %v3467 = stablehlo.reduce(%v3466 init: %v3447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3468 = stablehlo.broadcast_in_dim %v3467, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3469 = stablehlo.multiply %v3463, %v3448 : tensor<256x256x28x28xf32>
    %v3470 = stablehlo.subtract %v3469, %v3465 : tensor<256x256x28x28xf32>
    %v3471 = stablehlo.multiply %v3460, %v3468 : tensor<256x256x28x28xf32>
    %v3472 = stablehlo.subtract %v3470, %v3471 : tensor<256x256x28x28xf32>
    %v3473 = stablehlo.divide %v3459, %v3448 : tensor<256x256x28x28xf32>
    %v3474 = stablehlo.multiply %v3473, %v3472 : tensor<256x256x28x28xf32>
    %v3475 = stablehlo.reshape %v3474 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3476 = stablehlo.reshape %v3475 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3477 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v3478 = stablehlo.transpose %v3477, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v3479 = stablehlo.convolution(%v3476, %v3478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3480 = stablehlo.reshape %v3479 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3481 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3482 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3483 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3484 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3485 = stablehlo.reduce(%v3481 init: %v3482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3486 = stablehlo.broadcast_in_dim %v3485, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3487 = stablehlo.divide %v3486, %v3483 : tensor<256x1024x14x14xf32>
    %v3488 = stablehlo.subtract %v3481, %v3487 : tensor<256x1024x14x14xf32>
    %v3489 = stablehlo.multiply %v3488, %v3488 : tensor<256x1024x14x14xf32>
    %v3490 = stablehlo.reduce(%v3489 init: %v3482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3491 = stablehlo.broadcast_in_dim %v3490, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3492 = stablehlo.divide %v3491, %v3483 : tensor<256x1024x14x14xf32>
    %v3493 = stablehlo.add %v3492, %v3484 : tensor<256x1024x14x14xf32>
    %v3494 = stablehlo.rsqrt %v3493 : tensor<256x1024x14x14xf32>
    %v3495 = stablehlo.multiply %v3488, %v3494 : tensor<256x1024x14x14xf32>
    %v3496 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3497 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3498 = stablehlo.multiply %v3496, %v3497 : tensor<256x1024x14x14xf32>
    %v3499 = stablehlo.reduce(%v3498 init: %v3482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3500 = stablehlo.broadcast_in_dim %v3499, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3501 = stablehlo.multiply %v3495, %v3498 : tensor<256x1024x14x14xf32>
    %v3502 = stablehlo.reduce(%v3501 init: %v3482) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3503 = stablehlo.broadcast_in_dim %v3502, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3504 = stablehlo.multiply %v3498, %v3483 : tensor<256x1024x14x14xf32>
    %v3505 = stablehlo.subtract %v3504, %v3500 : tensor<256x1024x14x14xf32>
    %v3506 = stablehlo.multiply %v3495, %v3503 : tensor<256x1024x14x14xf32>
    %v3507 = stablehlo.subtract %v3505, %v3506 : tensor<256x1024x14x14xf32>
    %v3508 = stablehlo.divide %v3494, %v3483 : tensor<256x1024x14x14xf32>
    %v3509 = stablehlo.multiply %v3508, %v3507 : tensor<256x1024x14x14xf32>
    %v3510 = stablehlo.reshape %v3509 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3511 = stablehlo.reshape %v3510 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3513 = stablehlo.pad %v3511, %v3512, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<256x1024x28x28xf32>
    %v3514 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v3515 = stablehlo.transpose %v3514, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3516 = stablehlo.convolution(%v3513, %v3515)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x28x28xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3517 = stablehlo.reshape %v3516 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3518 = stablehlo.reshape %v3480 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3519 = stablehlo.reshape %v3517 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3520 = stablehlo.add %v3518, %v3519 : tensor<256x512x28x28xf32>
    %v3521 = stablehlo.reshape %v3520 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3522 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3523 = stablehlo.reshape %v3475 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3524 = stablehlo.transpose %v3522, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3525 = stablehlo.transpose %v3523, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3526 = stablehlo.convolution(%v3524, %v3525)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<512x256x1x1xf32>
    %v3527 = stablehlo.transpose %v3526, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v3528 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3530 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3531 = stablehlo.reduce(%v3528 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3531, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3533 = stablehlo.divide %v3532, %v3530 : tensor<256x256x28x28xf32>
    %v3534 = stablehlo.subtract %v3528, %v3533 : tensor<256x256x28x28xf32>
    %v3535 = stablehlo.multiply %v3534, %v3534 : tensor<256x256x28x28xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3538 = stablehlo.divide %v3537, %v3530 : tensor<256x256x28x28xf32>
    %v3539 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3540 = stablehlo.add %v3538, %v3539 : tensor<256x256x28x28xf32>
    %v3541 = stablehlo.rsqrt %v3540 : tensor<256x256x28x28xf32>
    %v3542 = stablehlo.multiply %v3534, %v3541 : tensor<256x256x28x28xf32>
    %v3543 = stablehlo.reshape %v3445 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3544 = stablehlo.multiply %v3543, %v3542 : tensor<256x256x28x28xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3546 = stablehlo.reshape %v3445 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3548 = stablehlo.reduce(%v3546 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3549 = stablehlo.reshape %v748 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3550 = stablehlo.reshape %v3432 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3552 = stablehlo.pad %v3550, %v3551, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v3553 = stablehlo.transpose %v3549, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3554 = stablehlo.transpose %v3552, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3555 = stablehlo.convolution(%v3553, %v3554)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<256x256x3x3xf32>
    %v3556 = stablehlo.transpose %v3555, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3557 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3558 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3559 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3560 = stablehlo.reduce(%v3557 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3561 = stablehlo.broadcast_in_dim %v3560, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3562 = stablehlo.divide %v3561, %v3559 : tensor<256x256x14x14xf32>
    %v3563 = stablehlo.subtract %v3557, %v3562 : tensor<256x256x14x14xf32>
    %v3564 = stablehlo.multiply %v3563, %v3563 : tensor<256x256x14x14xf32>
    %v3565 = stablehlo.reduce(%v3564 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3566 = stablehlo.broadcast_in_dim %v3565, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3567 = stablehlo.divide %v3566, %v3559 : tensor<256x256x14x14xf32>
    %v3568 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3569 = stablehlo.add %v3567, %v3568 : tensor<256x256x14x14xf32>
    %v3570 = stablehlo.rsqrt %v3569 : tensor<256x256x14x14xf32>
    %v3571 = stablehlo.multiply %v3563, %v3570 : tensor<256x256x14x14xf32>
    %v3572 = stablehlo.reshape %v3402 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3573 = stablehlo.multiply %v3572, %v3571 : tensor<256x256x14x14xf32>
    %v3574 = stablehlo.reduce(%v3573 init: %v3558) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3575 = stablehlo.reshape %v3402 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3577 = stablehlo.reduce(%v3575 init: %v3576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3578 = stablehlo.reshape %v777 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3579 = stablehlo.reshape %v3391 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3580 = stablehlo.transpose %v3578, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3581 = stablehlo.transpose %v3579, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3582 = stablehlo.convolution(%v3580, %v3581)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3583 = stablehlo.transpose %v3582, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3584 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3587 = stablehlo.reduce(%v3584 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3588 = stablehlo.broadcast_in_dim %v3587, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3589 = stablehlo.divide %v3588, %v3586 : tensor<256x1024x14x14xf32>
    %v3590 = stablehlo.subtract %v3584, %v3589 : tensor<256x1024x14x14xf32>
    %v3591 = stablehlo.multiply %v3590, %v3590 : tensor<256x1024x14x14xf32>
    %v3592 = stablehlo.reduce(%v3591 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3594 = stablehlo.divide %v3593, %v3586 : tensor<256x1024x14x14xf32>
    %v3595 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3596 = stablehlo.add %v3594, %v3595 : tensor<256x1024x14x14xf32>
    %v3597 = stablehlo.rsqrt %v3596 : tensor<256x1024x14x14xf32>
    %v3598 = stablehlo.multiply %v3590, %v3597 : tensor<256x1024x14x14xf32>
    %v3599 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3600 = stablehlo.multiply %v3599, %v3598 : tensor<256x1024x14x14xf32>
    %v3601 = stablehlo.reduce(%v3600 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3602 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3604 = stablehlo.reduce(%v3602 init: %v3603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3605 = stablehlo.reshape %v719 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3606 = stablehlo.reshape %v3510 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3608 = stablehlo.pad %v3606, %v3607, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<256x1024x28x28xf32>
    %v3609 = stablehlo.transpose %v3605, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3610 = stablehlo.transpose %v3608, dims = [1, 0, 2, 3] : (tensor<256x1024x28x28xf32>) -> tensor<1024x256x28x28xf32>
    %v3611 = stablehlo.convolution(%v3609, %v3610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<1024x256x28x28xf32>) -> tensor<512x1024x1x1xf32>
    %v3612 = stablehlo.transpose %v3611, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3613 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3615 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3616 = stablehlo.reduce(%v3613 init: %v3614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3617 = stablehlo.broadcast_in_dim %v3616, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3618 = stablehlo.divide %v3617, %v3615 : tensor<256x1024x14x14xf32>
    %v3619 = stablehlo.subtract %v3613, %v3618 : tensor<256x1024x14x14xf32>
    %v3620 = stablehlo.multiply %v3619, %v3619 : tensor<256x1024x14x14xf32>
    %v3621 = stablehlo.reduce(%v3620 init: %v3614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3622 = stablehlo.broadcast_in_dim %v3621, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3623 = stablehlo.divide %v3622, %v3615 : tensor<256x1024x14x14xf32>
    %v3624 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3625 = stablehlo.add %v3623, %v3624 : tensor<256x1024x14x14xf32>
    %v3626 = stablehlo.rsqrt %v3625 : tensor<256x1024x14x14xf32>
    %v3627 = stablehlo.multiply %v3619, %v3626 : tensor<256x1024x14x14xf32>
    %v3628 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3629 = stablehlo.multiply %v3628, %v3627 : tensor<256x1024x14x14xf32>
    %v3630 = stablehlo.reduce(%v3629 init: %v3614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3631 = stablehlo.reshape %v3361 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3632 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3633 = stablehlo.reduce(%v3631 init: %v3632) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3634 = stablehlo.reshape %v3521 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3635 = stablehlo.reshape %v715 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3636 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v3637 = stablehlo.compare GT, %v3635, %v3636 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v3638 = stablehlo.select %v3637, %v3634, %v3636 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v3639 = stablehlo.reshape %v3638 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3640 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3643 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3644 = stablehlo.reduce(%v3640 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3645 = stablehlo.broadcast_in_dim %v3644, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3646 = stablehlo.divide %v3645, %v3642 : tensor<256x512x28x28xf32>
    %v3647 = stablehlo.subtract %v3640, %v3646 : tensor<256x512x28x28xf32>
    %v3648 = stablehlo.multiply %v3647, %v3647 : tensor<256x512x28x28xf32>
    %v3649 = stablehlo.reduce(%v3648 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3650 = stablehlo.broadcast_in_dim %v3649, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3651 = stablehlo.divide %v3650, %v3642 : tensor<256x512x28x28xf32>
    %v3652 = stablehlo.add %v3651, %v3643 : tensor<256x512x28x28xf32>
    %v3653 = stablehlo.rsqrt %v3652 : tensor<256x512x28x28xf32>
    %v3654 = stablehlo.multiply %v3647, %v3653 : tensor<256x512x28x28xf32>
    %v3655 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3656 = stablehlo.reshape %v3639 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3657 = stablehlo.multiply %v3655, %v3656 : tensor<256x512x28x28xf32>
    %v3658 = stablehlo.reduce(%v3657 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3659 = stablehlo.broadcast_in_dim %v3658, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3660 = stablehlo.multiply %v3654, %v3657 : tensor<256x512x28x28xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3661, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3663 = stablehlo.multiply %v3657, %v3642 : tensor<256x512x28x28xf32>
    %v3664 = stablehlo.subtract %v3663, %v3659 : tensor<256x512x28x28xf32>
    %v3665 = stablehlo.multiply %v3654, %v3662 : tensor<256x512x28x28xf32>
    %v3666 = stablehlo.subtract %v3664, %v3665 : tensor<256x512x28x28xf32>
    %v3667 = stablehlo.divide %v3653, %v3642 : tensor<256x512x28x28xf32>
    %v3668 = stablehlo.multiply %v3667, %v3666 : tensor<256x512x28x28xf32>
    %v3669 = stablehlo.reshape %v3668 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3670 = stablehlo.reshape %v3669 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3671 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3672 = stablehlo.transpose %v3671, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3673 = stablehlo.convolution(%v3670, %v3672)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3674 = stablehlo.reshape %v3673 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3675 = stablehlo.reshape %v3674 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3676 = stablehlo.reshape %v682 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3677 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3678 = stablehlo.compare GT, %v3676, %v3677 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3679 = stablehlo.select %v3678, %v3675, %v3677 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3681 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3683 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3684 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3685 = stablehlo.reduce(%v3681 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3686 = stablehlo.broadcast_in_dim %v3685, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3687 = stablehlo.divide %v3686, %v3683 : tensor<256x128x28x28xf32>
    %v3688 = stablehlo.subtract %v3681, %v3687 : tensor<256x128x28x28xf32>
    %v3689 = stablehlo.multiply %v3688, %v3688 : tensor<256x128x28x28xf32>
    %v3690 = stablehlo.reduce(%v3689 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3691 = stablehlo.broadcast_in_dim %v3690, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3692 = stablehlo.divide %v3691, %v3683 : tensor<256x128x28x28xf32>
    %v3693 = stablehlo.add %v3692, %v3684 : tensor<256x128x28x28xf32>
    %v3694 = stablehlo.rsqrt %v3693 : tensor<256x128x28x28xf32>
    %v3695 = stablehlo.multiply %v3688, %v3694 : tensor<256x128x28x28xf32>
    %v3696 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3697 = stablehlo.reshape %v3680 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3698 = stablehlo.multiply %v3696, %v3697 : tensor<256x128x28x28xf32>
    %v3699 = stablehlo.reduce(%v3698 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3700 = stablehlo.broadcast_in_dim %v3699, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3701 = stablehlo.multiply %v3695, %v3698 : tensor<256x128x28x28xf32>
    %v3702 = stablehlo.reduce(%v3701 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3703 = stablehlo.broadcast_in_dim %v3702, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3704 = stablehlo.multiply %v3698, %v3683 : tensor<256x128x28x28xf32>
    %v3705 = stablehlo.subtract %v3704, %v3700 : tensor<256x128x28x28xf32>
    %v3706 = stablehlo.multiply %v3695, %v3703 : tensor<256x128x28x28xf32>
    %v3707 = stablehlo.subtract %v3705, %v3706 : tensor<256x128x28x28xf32>
    %v3708 = stablehlo.divide %v3694, %v3683 : tensor<256x128x28x28xf32>
    %v3709 = stablehlo.multiply %v3708, %v3707 : tensor<256x128x28x28xf32>
    %v3710 = stablehlo.reshape %v3709 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3711 = stablehlo.reshape %v3710 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3712 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3713 = stablehlo.transpose %v3712, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3714 = stablehlo.convolution(%v3711, %v3713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3715 = stablehlo.reshape %v3714 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3716 = stablehlo.reshape %v3715 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3717 = stablehlo.reshape %v653 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3718 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3719 = stablehlo.compare GT, %v3717, %v3718 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3720 = stablehlo.select %v3719, %v3716, %v3718 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3721 = stablehlo.reshape %v3720 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3722 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3724 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3725 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3726 = stablehlo.reduce(%v3722 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3727 = stablehlo.broadcast_in_dim %v3726, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3728 = stablehlo.divide %v3727, %v3724 : tensor<256x128x28x28xf32>
    %v3729 = stablehlo.subtract %v3722, %v3728 : tensor<256x128x28x28xf32>
    %v3730 = stablehlo.multiply %v3729, %v3729 : tensor<256x128x28x28xf32>
    %v3731 = stablehlo.reduce(%v3730 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3732 = stablehlo.broadcast_in_dim %v3731, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3733 = stablehlo.divide %v3732, %v3724 : tensor<256x128x28x28xf32>
    %v3734 = stablehlo.add %v3733, %v3725 : tensor<256x128x28x28xf32>
    %v3735 = stablehlo.rsqrt %v3734 : tensor<256x128x28x28xf32>
    %v3736 = stablehlo.multiply %v3729, %v3735 : tensor<256x128x28x28xf32>
    %v3737 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3738 = stablehlo.reshape %v3721 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3739 = stablehlo.multiply %v3737, %v3738 : tensor<256x128x28x28xf32>
    %v3740 = stablehlo.reduce(%v3739 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3741 = stablehlo.broadcast_in_dim %v3740, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3742 = stablehlo.multiply %v3736, %v3739 : tensor<256x128x28x28xf32>
    %v3743 = stablehlo.reduce(%v3742 init: %v3723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3744 = stablehlo.broadcast_in_dim %v3743, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3745 = stablehlo.multiply %v3739, %v3724 : tensor<256x128x28x28xf32>
    %v3746 = stablehlo.subtract %v3745, %v3741 : tensor<256x128x28x28xf32>
    %v3747 = stablehlo.multiply %v3736, %v3744 : tensor<256x128x28x28xf32>
    %v3748 = stablehlo.subtract %v3746, %v3747 : tensor<256x128x28x28xf32>
    %v3749 = stablehlo.divide %v3735, %v3724 : tensor<256x128x28x28xf32>
    %v3750 = stablehlo.multiply %v3749, %v3748 : tensor<256x128x28x28xf32>
    %v3751 = stablehlo.reshape %v3750 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3752 = stablehlo.reshape %v3751 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3753 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3754 = stablehlo.transpose %v3753, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3755 = stablehlo.convolution(%v3752, %v3754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3756 = stablehlo.reshape %v3755 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3757 = stablehlo.reshape %v3756 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3758 = stablehlo.reshape %v3639 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3759 = stablehlo.add %v3757, %v3758 : tensor<256x512x28x28xf32>
    %v3760 = stablehlo.reshape %v3759 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3761 = stablehlo.reshape %v628 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3762 = stablehlo.reshape %v3751 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3763 = stablehlo.transpose %v3761, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3764 = stablehlo.transpose %v3762, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3765 = stablehlo.convolution(%v3763, %v3764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3766 = stablehlo.transpose %v3765, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3767 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3769 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3770 = stablehlo.reduce(%v3767 init: %v3768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3771 = stablehlo.broadcast_in_dim %v3770, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3772 = stablehlo.divide %v3771, %v3769 : tensor<256x128x28x28xf32>
    %v3773 = stablehlo.subtract %v3767, %v3772 : tensor<256x128x28x28xf32>
    %v3774 = stablehlo.multiply %v3773, %v3773 : tensor<256x128x28x28xf32>
    %v3775 = stablehlo.reduce(%v3774 init: %v3768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3776 = stablehlo.broadcast_in_dim %v3775, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3777 = stablehlo.divide %v3776, %v3769 : tensor<256x128x28x28xf32>
    %v3778 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3779 = stablehlo.add %v3777, %v3778 : tensor<256x128x28x28xf32>
    %v3780 = stablehlo.rsqrt %v3779 : tensor<256x128x28x28xf32>
    %v3781 = stablehlo.multiply %v3773, %v3780 : tensor<256x128x28x28xf32>
    %v3782 = stablehlo.reshape %v3721 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3783 = stablehlo.multiply %v3782, %v3781 : tensor<256x128x28x28xf32>
    %v3784 = stablehlo.reduce(%v3783 init: %v3768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3785 = stablehlo.reshape %v3721 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3787 = stablehlo.reduce(%v3785 init: %v3786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3788 = stablehlo.reshape %v657 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3789 = stablehlo.reshape %v3710 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3790 = stablehlo.transpose %v3788, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3791 = stablehlo.transpose %v3789, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3792 = stablehlo.convolution(%v3790, %v3791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3793 = stablehlo.transpose %v3792, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3794 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3796 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3797 = stablehlo.reduce(%v3794 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3798 = stablehlo.broadcast_in_dim %v3797, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3799 = stablehlo.divide %v3798, %v3796 : tensor<256x128x28x28xf32>
    %v3800 = stablehlo.subtract %v3794, %v3799 : tensor<256x128x28x28xf32>
    %v3801 = stablehlo.multiply %v3800, %v3800 : tensor<256x128x28x28xf32>
    %v3802 = stablehlo.reduce(%v3801 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3803 = stablehlo.broadcast_in_dim %v3802, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3804 = stablehlo.divide %v3803, %v3796 : tensor<256x128x28x28xf32>
    %v3805 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3806 = stablehlo.add %v3804, %v3805 : tensor<256x128x28x28xf32>
    %v3807 = stablehlo.rsqrt %v3806 : tensor<256x128x28x28xf32>
    %v3808 = stablehlo.multiply %v3800, %v3807 : tensor<256x128x28x28xf32>
    %v3809 = stablehlo.reshape %v3680 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3810 = stablehlo.multiply %v3809, %v3808 : tensor<256x128x28x28xf32>
    %v3811 = stablehlo.reduce(%v3810 init: %v3795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3812 = stablehlo.reshape %v3680 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3813 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3814 = stablehlo.reduce(%v3812 init: %v3813) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3815 = stablehlo.reshape %v686 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3816 = stablehlo.reshape %v3669 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3817 = stablehlo.transpose %v3815, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3818 = stablehlo.transpose %v3816, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3819 = stablehlo.convolution(%v3817, %v3818)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v3820 = stablehlo.transpose %v3819, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3821 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3823 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3824 = stablehlo.reduce(%v3821 init: %v3822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3825 = stablehlo.broadcast_in_dim %v3824, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3826 = stablehlo.divide %v3825, %v3823 : tensor<256x512x28x28xf32>
    %v3827 = stablehlo.subtract %v3821, %v3826 : tensor<256x512x28x28xf32>
    %v3828 = stablehlo.multiply %v3827, %v3827 : tensor<256x512x28x28xf32>
    %v3829 = stablehlo.reduce(%v3828 init: %v3822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3830 = stablehlo.broadcast_in_dim %v3829, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3831 = stablehlo.divide %v3830, %v3823 : tensor<256x512x28x28xf32>
    %v3832 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3833 = stablehlo.add %v3831, %v3832 : tensor<256x512x28x28xf32>
    %v3834 = stablehlo.rsqrt %v3833 : tensor<256x512x28x28xf32>
    %v3835 = stablehlo.multiply %v3827, %v3834 : tensor<256x512x28x28xf32>
    %v3836 = stablehlo.reshape %v3639 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3837 = stablehlo.multiply %v3836, %v3835 : tensor<256x512x28x28xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3839 = stablehlo.reshape %v3639 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3841 = stablehlo.reduce(%v3839 init: %v3840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3842 = stablehlo.reshape %v3760 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3843 = stablehlo.reshape %v624 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3844 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v3845 = stablehlo.compare GT, %v3843, %v3844 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v3846 = stablehlo.select %v3845, %v3842, %v3844 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3848 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3849 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3850 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3851 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3852 = stablehlo.reduce(%v3848 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3852, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3854 = stablehlo.divide %v3853, %v3850 : tensor<256x512x28x28xf32>
    %v3855 = stablehlo.subtract %v3848, %v3854 : tensor<256x512x28x28xf32>
    %v3856 = stablehlo.multiply %v3855, %v3855 : tensor<256x512x28x28xf32>
    %v3857 = stablehlo.reduce(%v3856 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3858 = stablehlo.broadcast_in_dim %v3857, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3859 = stablehlo.divide %v3858, %v3850 : tensor<256x512x28x28xf32>
    %v3860 = stablehlo.add %v3859, %v3851 : tensor<256x512x28x28xf32>
    %v3861 = stablehlo.rsqrt %v3860 : tensor<256x512x28x28xf32>
    %v3862 = stablehlo.multiply %v3855, %v3861 : tensor<256x512x28x28xf32>
    %v3863 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3864 = stablehlo.reshape %v3847 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3865 = stablehlo.multiply %v3863, %v3864 : tensor<256x512x28x28xf32>
    %v3866 = stablehlo.reduce(%v3865 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3867 = stablehlo.broadcast_in_dim %v3866, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3868 = stablehlo.multiply %v3862, %v3865 : tensor<256x512x28x28xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3849) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3871 = stablehlo.multiply %v3865, %v3850 : tensor<256x512x28x28xf32>
    %v3872 = stablehlo.subtract %v3871, %v3867 : tensor<256x512x28x28xf32>
    %v3873 = stablehlo.multiply %v3862, %v3870 : tensor<256x512x28x28xf32>
    %v3874 = stablehlo.subtract %v3872, %v3873 : tensor<256x512x28x28xf32>
    %v3875 = stablehlo.divide %v3861, %v3850 : tensor<256x512x28x28xf32>
    %v3876 = stablehlo.multiply %v3875, %v3874 : tensor<256x512x28x28xf32>
    %v3877 = stablehlo.reshape %v3876 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3878 = stablehlo.reshape %v3877 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3879 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3880 = stablehlo.transpose %v3879, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3881 = stablehlo.convolution(%v3878, %v3880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3882 = stablehlo.reshape %v3881 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3883 = stablehlo.reshape %v3882 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3884 = stablehlo.reshape %v591 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3886 = stablehlo.compare GT, %v3884, %v3885 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3887 = stablehlo.select %v3886, %v3883, %v3885 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3888 = stablehlo.reshape %v3887 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3889 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3892 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3893 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3894 = stablehlo.broadcast_in_dim %v3893, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3895 = stablehlo.divide %v3894, %v3891 : tensor<256x128x28x28xf32>
    %v3896 = stablehlo.subtract %v3889, %v3895 : tensor<256x128x28x28xf32>
    %v3897 = stablehlo.multiply %v3896, %v3896 : tensor<256x128x28x28xf32>
    %v3898 = stablehlo.reduce(%v3897 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3899 = stablehlo.broadcast_in_dim %v3898, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3900 = stablehlo.divide %v3899, %v3891 : tensor<256x128x28x28xf32>
    %v3901 = stablehlo.add %v3900, %v3892 : tensor<256x128x28x28xf32>
    %v3902 = stablehlo.rsqrt %v3901 : tensor<256x128x28x28xf32>
    %v3903 = stablehlo.multiply %v3896, %v3902 : tensor<256x128x28x28xf32>
    %v3904 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3905 = stablehlo.reshape %v3888 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3906 = stablehlo.multiply %v3904, %v3905 : tensor<256x128x28x28xf32>
    %v3907 = stablehlo.reduce(%v3906 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3908 = stablehlo.broadcast_in_dim %v3907, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3909 = stablehlo.multiply %v3903, %v3906 : tensor<256x128x28x28xf32>
    %v3910 = stablehlo.reduce(%v3909 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3911 = stablehlo.broadcast_in_dim %v3910, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3912 = stablehlo.multiply %v3906, %v3891 : tensor<256x128x28x28xf32>
    %v3913 = stablehlo.subtract %v3912, %v3908 : tensor<256x128x28x28xf32>
    %v3914 = stablehlo.multiply %v3903, %v3911 : tensor<256x128x28x28xf32>
    %v3915 = stablehlo.subtract %v3913, %v3914 : tensor<256x128x28x28xf32>
    %v3916 = stablehlo.divide %v3902, %v3891 : tensor<256x128x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3915 : tensor<256x128x28x28xf32>
    %v3918 = stablehlo.reshape %v3917 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3919 = stablehlo.reshape %v3918 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3920 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3921 = stablehlo.transpose %v3920, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3922 = stablehlo.convolution(%v3919, %v3921)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3923 = stablehlo.reshape %v3922 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3924 = stablehlo.reshape %v3923 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3925 = stablehlo.reshape %v562 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3926 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v3927 = stablehlo.compare GT, %v3925, %v3926 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v3928 = stablehlo.select %v3927, %v3924, %v3926 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v3929 = stablehlo.reshape %v3928 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3930 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3933 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3934 = stablehlo.reduce(%v3930 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3935 = stablehlo.broadcast_in_dim %v3934, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3936 = stablehlo.divide %v3935, %v3932 : tensor<256x128x28x28xf32>
    %v3937 = stablehlo.subtract %v3930, %v3936 : tensor<256x128x28x28xf32>
    %v3938 = stablehlo.multiply %v3937, %v3937 : tensor<256x128x28x28xf32>
    %v3939 = stablehlo.reduce(%v3938 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3940 = stablehlo.broadcast_in_dim %v3939, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3941 = stablehlo.divide %v3940, %v3932 : tensor<256x128x28x28xf32>
    %v3942 = stablehlo.add %v3941, %v3933 : tensor<256x128x28x28xf32>
    %v3943 = stablehlo.rsqrt %v3942 : tensor<256x128x28x28xf32>
    %v3944 = stablehlo.multiply %v3937, %v3943 : tensor<256x128x28x28xf32>
    %v3945 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3946 = stablehlo.reshape %v3929 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3947 = stablehlo.multiply %v3945, %v3946 : tensor<256x128x28x28xf32>
    %v3948 = stablehlo.reduce(%v3947 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3949 = stablehlo.broadcast_in_dim %v3948, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3950 = stablehlo.multiply %v3944, %v3947 : tensor<256x128x28x28xf32>
    %v3951 = stablehlo.reduce(%v3950 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3952 = stablehlo.broadcast_in_dim %v3951, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3953 = stablehlo.multiply %v3947, %v3932 : tensor<256x128x28x28xf32>
    %v3954 = stablehlo.subtract %v3953, %v3949 : tensor<256x128x28x28xf32>
    %v3955 = stablehlo.multiply %v3944, %v3952 : tensor<256x128x28x28xf32>
    %v3956 = stablehlo.subtract %v3954, %v3955 : tensor<256x128x28x28xf32>
    %v3957 = stablehlo.divide %v3943, %v3932 : tensor<256x128x28x28xf32>
    %v3958 = stablehlo.multiply %v3957, %v3956 : tensor<256x128x28x28xf32>
    %v3959 = stablehlo.reshape %v3958 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3960 = stablehlo.reshape %v3959 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3961 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3962 = stablehlo.transpose %v3961, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3963 = stablehlo.convolution(%v3960, %v3962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3964 = stablehlo.reshape %v3963 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3965 = stablehlo.reshape %v3964 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3966 = stablehlo.reshape %v3847 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3967 = stablehlo.add %v3965, %v3966 : tensor<256x512x28x28xf32>
    %v3968 = stablehlo.reshape %v3967 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3969 = stablehlo.reshape %v537 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3970 = stablehlo.reshape %v3959 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3971 = stablehlo.transpose %v3969, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3972 = stablehlo.transpose %v3970, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3973 = stablehlo.convolution(%v3971, %v3972)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3974 = stablehlo.transpose %v3973, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3975 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3977 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3978 = stablehlo.reduce(%v3975 init: %v3976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3979 = stablehlo.broadcast_in_dim %v3978, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3980 = stablehlo.divide %v3979, %v3977 : tensor<256x128x28x28xf32>
    %v3981 = stablehlo.subtract %v3975, %v3980 : tensor<256x128x28x28xf32>
    %v3982 = stablehlo.multiply %v3981, %v3981 : tensor<256x128x28x28xf32>
    %v3983 = stablehlo.reduce(%v3982 init: %v3976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3984 = stablehlo.broadcast_in_dim %v3983, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3985 = stablehlo.divide %v3984, %v3977 : tensor<256x128x28x28xf32>
    %v3986 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3987 = stablehlo.add %v3985, %v3986 : tensor<256x128x28x28xf32>
    %v3988 = stablehlo.rsqrt %v3987 : tensor<256x128x28x28xf32>
    %v3989 = stablehlo.multiply %v3981, %v3988 : tensor<256x128x28x28xf32>
    %v3990 = stablehlo.reshape %v3929 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3991 = stablehlo.multiply %v3990, %v3989 : tensor<256x128x28x28xf32>
    %v3992 = stablehlo.reduce(%v3991 init: %v3976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3993 = stablehlo.reshape %v3929 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3995 = stablehlo.reduce(%v3993 init: %v3994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3996 = stablehlo.reshape %v566 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3997 = stablehlo.reshape %v3918 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3998 = stablehlo.transpose %v3996, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3999 = stablehlo.transpose %v3997, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4000 = stablehlo.convolution(%v3998, %v3999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v4001 = stablehlo.transpose %v4000, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4002 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4004 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4005 = stablehlo.reduce(%v4002 init: %v4003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4006 = stablehlo.broadcast_in_dim %v4005, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4007 = stablehlo.divide %v4006, %v4004 : tensor<256x128x28x28xf32>
    %v4008 = stablehlo.subtract %v4002, %v4007 : tensor<256x128x28x28xf32>
    %v4009 = stablehlo.multiply %v4008, %v4008 : tensor<256x128x28x28xf32>
    %v4010 = stablehlo.reduce(%v4009 init: %v4003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4011 = stablehlo.broadcast_in_dim %v4010, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4012 = stablehlo.divide %v4011, %v4004 : tensor<256x128x28x28xf32>
    %v4013 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4014 = stablehlo.add %v4012, %v4013 : tensor<256x128x28x28xf32>
    %v4015 = stablehlo.rsqrt %v4014 : tensor<256x128x28x28xf32>
    %v4016 = stablehlo.multiply %v4008, %v4015 : tensor<256x128x28x28xf32>
    %v4017 = stablehlo.reshape %v3888 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4018 = stablehlo.multiply %v4017, %v4016 : tensor<256x128x28x28xf32>
    %v4019 = stablehlo.reduce(%v4018 init: %v4003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4020 = stablehlo.reshape %v3888 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4022 = stablehlo.reduce(%v4020 init: %v4021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4023 = stablehlo.reshape %v595 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4024 = stablehlo.reshape %v3877 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4025 = stablehlo.transpose %v4023, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4026 = stablehlo.transpose %v4024, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4027 = stablehlo.convolution(%v4025, %v4026)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4028 = stablehlo.transpose %v4027, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4029 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4030 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4031 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4032 = stablehlo.reduce(%v4029 init: %v4030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4033 = stablehlo.broadcast_in_dim %v4032, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4034 = stablehlo.divide %v4033, %v4031 : tensor<256x512x28x28xf32>
    %v4035 = stablehlo.subtract %v4029, %v4034 : tensor<256x512x28x28xf32>
    %v4036 = stablehlo.multiply %v4035, %v4035 : tensor<256x512x28x28xf32>
    %v4037 = stablehlo.reduce(%v4036 init: %v4030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4038 = stablehlo.broadcast_in_dim %v4037, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4039 = stablehlo.divide %v4038, %v4031 : tensor<256x512x28x28xf32>
    %v4040 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4041 = stablehlo.add %v4039, %v4040 : tensor<256x512x28x28xf32>
    %v4042 = stablehlo.rsqrt %v4041 : tensor<256x512x28x28xf32>
    %v4043 = stablehlo.multiply %v4035, %v4042 : tensor<256x512x28x28xf32>
    %v4044 = stablehlo.reshape %v3847 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4045 = stablehlo.multiply %v4044, %v4043 : tensor<256x512x28x28xf32>
    %v4046 = stablehlo.reduce(%v4045 init: %v4030) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4047 = stablehlo.reshape %v3847 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4049 = stablehlo.reduce(%v4047 init: %v4048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4050 = stablehlo.reshape %v3968 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4051 = stablehlo.reshape %v533 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4052 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v4053 = stablehlo.compare GT, %v4051, %v4052 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v4054 = stablehlo.select %v4053, %v4050, %v4052 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v4055 = stablehlo.reshape %v4054 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4056 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4058 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4059 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4060 = stablehlo.reduce(%v4056 init: %v4057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4061 = stablehlo.broadcast_in_dim %v4060, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4062 = stablehlo.divide %v4061, %v4058 : tensor<256x512x28x28xf32>
    %v4063 = stablehlo.subtract %v4056, %v4062 : tensor<256x512x28x28xf32>
    %v4064 = stablehlo.multiply %v4063, %v4063 : tensor<256x512x28x28xf32>
    %v4065 = stablehlo.reduce(%v4064 init: %v4057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4066 = stablehlo.broadcast_in_dim %v4065, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4067 = stablehlo.divide %v4066, %v4058 : tensor<256x512x28x28xf32>
    %v4068 = stablehlo.add %v4067, %v4059 : tensor<256x512x28x28xf32>
    %v4069 = stablehlo.rsqrt %v4068 : tensor<256x512x28x28xf32>
    %v4070 = stablehlo.multiply %v4063, %v4069 : tensor<256x512x28x28xf32>
    %v4071 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4072 = stablehlo.reshape %v4055 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4073 = stablehlo.multiply %v4071, %v4072 : tensor<256x512x28x28xf32>
    %v4074 = stablehlo.reduce(%v4073 init: %v4057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4075 = stablehlo.broadcast_in_dim %v4074, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4076 = stablehlo.multiply %v4070, %v4073 : tensor<256x512x28x28xf32>
    %v4077 = stablehlo.reduce(%v4076 init: %v4057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4077, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4079 = stablehlo.multiply %v4073, %v4058 : tensor<256x512x28x28xf32>
    %v4080 = stablehlo.subtract %v4079, %v4075 : tensor<256x512x28x28xf32>
    %v4081 = stablehlo.multiply %v4070, %v4078 : tensor<256x512x28x28xf32>
    %v4082 = stablehlo.subtract %v4080, %v4081 : tensor<256x512x28x28xf32>
    %v4083 = stablehlo.divide %v4069, %v4058 : tensor<256x512x28x28xf32>
    %v4084 = stablehlo.multiply %v4083, %v4082 : tensor<256x512x28x28xf32>
    %v4085 = stablehlo.reshape %v4084 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4086 = stablehlo.reshape %v4085 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4087 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4088 = stablehlo.transpose %v4087, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4089 = stablehlo.convolution(%v4086, %v4088)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v4090 = stablehlo.reshape %v4089 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4091 = stablehlo.reshape %v4090 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4092 = stablehlo.reshape %v500 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4093 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4094 = stablehlo.compare GT, %v4092, %v4093 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4095 = stablehlo.select %v4094, %v4091, %v4093 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4096 = stablehlo.reshape %v4095 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4097 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4099 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4100 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4101 = stablehlo.reduce(%v4097 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4102 = stablehlo.broadcast_in_dim %v4101, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4103 = stablehlo.divide %v4102, %v4099 : tensor<256x128x28x28xf32>
    %v4104 = stablehlo.subtract %v4097, %v4103 : tensor<256x128x28x28xf32>
    %v4105 = stablehlo.multiply %v4104, %v4104 : tensor<256x128x28x28xf32>
    %v4106 = stablehlo.reduce(%v4105 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4107 = stablehlo.broadcast_in_dim %v4106, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4108 = stablehlo.divide %v4107, %v4099 : tensor<256x128x28x28xf32>
    %v4109 = stablehlo.add %v4108, %v4100 : tensor<256x128x28x28xf32>
    %v4110 = stablehlo.rsqrt %v4109 : tensor<256x128x28x28xf32>
    %v4111 = stablehlo.multiply %v4104, %v4110 : tensor<256x128x28x28xf32>
    %v4112 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4113 = stablehlo.reshape %v4096 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4114 = stablehlo.multiply %v4112, %v4113 : tensor<256x128x28x28xf32>
    %v4115 = stablehlo.reduce(%v4114 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4116 = stablehlo.broadcast_in_dim %v4115, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4117 = stablehlo.multiply %v4111, %v4114 : tensor<256x128x28x28xf32>
    %v4118 = stablehlo.reduce(%v4117 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4119 = stablehlo.broadcast_in_dim %v4118, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4120 = stablehlo.multiply %v4114, %v4099 : tensor<256x128x28x28xf32>
    %v4121 = stablehlo.subtract %v4120, %v4116 : tensor<256x128x28x28xf32>
    %v4122 = stablehlo.multiply %v4111, %v4119 : tensor<256x128x28x28xf32>
    %v4123 = stablehlo.subtract %v4121, %v4122 : tensor<256x128x28x28xf32>
    %v4124 = stablehlo.divide %v4110, %v4099 : tensor<256x128x28x28xf32>
    %v4125 = stablehlo.multiply %v4124, %v4123 : tensor<256x128x28x28xf32>
    %v4126 = stablehlo.reshape %v4125 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4127 = stablehlo.reshape %v4126 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4128 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4129 = stablehlo.transpose %v4128, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4130 = stablehlo.convolution(%v4127, %v4129)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v4131 = stablehlo.reshape %v4130 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4132 = stablehlo.reshape %v4131 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4133 = stablehlo.reshape %v471 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4134 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4135 = stablehlo.compare GT, %v4133, %v4134 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4136 = stablehlo.select %v4135, %v4132, %v4134 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4137 = stablehlo.reshape %v4136 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4138 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4140 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4141 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4142 = stablehlo.reduce(%v4138 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4143 = stablehlo.broadcast_in_dim %v4142, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4144 = stablehlo.divide %v4143, %v4140 : tensor<256x128x28x28xf32>
    %v4145 = stablehlo.subtract %v4138, %v4144 : tensor<256x128x28x28xf32>
    %v4146 = stablehlo.multiply %v4145, %v4145 : tensor<256x128x28x28xf32>
    %v4147 = stablehlo.reduce(%v4146 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4149 = stablehlo.divide %v4148, %v4140 : tensor<256x128x28x28xf32>
    %v4150 = stablehlo.add %v4149, %v4141 : tensor<256x128x28x28xf32>
    %v4151 = stablehlo.rsqrt %v4150 : tensor<256x128x28x28xf32>
    %v4152 = stablehlo.multiply %v4145, %v4151 : tensor<256x128x28x28xf32>
    %v4153 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4154 = stablehlo.reshape %v4137 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4155 = stablehlo.multiply %v4153, %v4154 : tensor<256x128x28x28xf32>
    %v4156 = stablehlo.reduce(%v4155 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4157 = stablehlo.broadcast_in_dim %v4156, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4158 = stablehlo.multiply %v4152, %v4155 : tensor<256x128x28x28xf32>
    %v4159 = stablehlo.reduce(%v4158 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4160 = stablehlo.broadcast_in_dim %v4159, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4161 = stablehlo.multiply %v4155, %v4140 : tensor<256x128x28x28xf32>
    %v4162 = stablehlo.subtract %v4161, %v4157 : tensor<256x128x28x28xf32>
    %v4163 = stablehlo.multiply %v4152, %v4160 : tensor<256x128x28x28xf32>
    %v4164 = stablehlo.subtract %v4162, %v4163 : tensor<256x128x28x28xf32>
    %v4165 = stablehlo.divide %v4151, %v4140 : tensor<256x128x28x28xf32>
    %v4166 = stablehlo.multiply %v4165, %v4164 : tensor<256x128x28x28xf32>
    %v4167 = stablehlo.reshape %v4166 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4168 = stablehlo.reshape %v4167 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4169 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4170 = stablehlo.transpose %v4169, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4171 = stablehlo.convolution(%v4168, %v4170)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v4172 = stablehlo.reshape %v4171 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4174 = stablehlo.reshape %v4055 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4175 = stablehlo.add %v4173, %v4174 : tensor<256x512x28x28xf32>
    %v4176 = stablehlo.reshape %v4175 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4177 = stablehlo.reshape %v446 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4178 = stablehlo.reshape %v4167 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4179 = stablehlo.transpose %v4177, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4180 = stablehlo.transpose %v4178, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4181 = stablehlo.convolution(%v4179, %v4180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v4182 = stablehlo.transpose %v4181, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4183 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4185 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4186 = stablehlo.reduce(%v4183 init: %v4184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4187 = stablehlo.broadcast_in_dim %v4186, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4188 = stablehlo.divide %v4187, %v4185 : tensor<256x128x28x28xf32>
    %v4189 = stablehlo.subtract %v4183, %v4188 : tensor<256x128x28x28xf32>
    %v4190 = stablehlo.multiply %v4189, %v4189 : tensor<256x128x28x28xf32>
    %v4191 = stablehlo.reduce(%v4190 init: %v4184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4192 = stablehlo.broadcast_in_dim %v4191, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4193 = stablehlo.divide %v4192, %v4185 : tensor<256x128x28x28xf32>
    %v4194 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4195 = stablehlo.add %v4193, %v4194 : tensor<256x128x28x28xf32>
    %v4196 = stablehlo.rsqrt %v4195 : tensor<256x128x28x28xf32>
    %v4197 = stablehlo.multiply %v4189, %v4196 : tensor<256x128x28x28xf32>
    %v4198 = stablehlo.reshape %v4137 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4199 = stablehlo.multiply %v4198, %v4197 : tensor<256x128x28x28xf32>
    %v4200 = stablehlo.reduce(%v4199 init: %v4184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4201 = stablehlo.reshape %v4137 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4203 = stablehlo.reduce(%v4201 init: %v4202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4204 = stablehlo.reshape %v475 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4205 = stablehlo.reshape %v4126 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4206 = stablehlo.transpose %v4204, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4207 = stablehlo.transpose %v4205, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4208 = stablehlo.convolution(%v4206, %v4207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v4209 = stablehlo.transpose %v4208, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4210 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4212 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4213 = stablehlo.reduce(%v4210 init: %v4211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4214 = stablehlo.broadcast_in_dim %v4213, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4215 = stablehlo.divide %v4214, %v4212 : tensor<256x128x28x28xf32>
    %v4216 = stablehlo.subtract %v4210, %v4215 : tensor<256x128x28x28xf32>
    %v4217 = stablehlo.multiply %v4216, %v4216 : tensor<256x128x28x28xf32>
    %v4218 = stablehlo.reduce(%v4217 init: %v4211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4219 = stablehlo.broadcast_in_dim %v4218, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4220 = stablehlo.divide %v4219, %v4212 : tensor<256x128x28x28xf32>
    %v4221 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4222 = stablehlo.add %v4220, %v4221 : tensor<256x128x28x28xf32>
    %v4223 = stablehlo.rsqrt %v4222 : tensor<256x128x28x28xf32>
    %v4224 = stablehlo.multiply %v4216, %v4223 : tensor<256x128x28x28xf32>
    %v4225 = stablehlo.reshape %v4096 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4226 = stablehlo.multiply %v4225, %v4224 : tensor<256x128x28x28xf32>
    %v4227 = stablehlo.reduce(%v4226 init: %v4211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4228 = stablehlo.reshape %v4096 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4230 = stablehlo.reduce(%v4228 init: %v4229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4231 = stablehlo.reshape %v504 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4232 = stablehlo.reshape %v4085 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4233 = stablehlo.transpose %v4231, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4234 = stablehlo.transpose %v4232, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4235 = stablehlo.convolution(%v4233, %v4234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4236 = stablehlo.transpose %v4235, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4237 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4239 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4240 = stablehlo.reduce(%v4237 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4241 = stablehlo.broadcast_in_dim %v4240, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4242 = stablehlo.divide %v4241, %v4239 : tensor<256x512x28x28xf32>
    %v4243 = stablehlo.subtract %v4237, %v4242 : tensor<256x512x28x28xf32>
    %v4244 = stablehlo.multiply %v4243, %v4243 : tensor<256x512x28x28xf32>
    %v4245 = stablehlo.reduce(%v4244 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4246 = stablehlo.broadcast_in_dim %v4245, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4247 = stablehlo.divide %v4246, %v4239 : tensor<256x512x28x28xf32>
    %v4248 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4249 = stablehlo.add %v4247, %v4248 : tensor<256x512x28x28xf32>
    %v4250 = stablehlo.rsqrt %v4249 : tensor<256x512x28x28xf32>
    %v4251 = stablehlo.multiply %v4243, %v4250 : tensor<256x512x28x28xf32>
    %v4252 = stablehlo.reshape %v4055 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4253 = stablehlo.multiply %v4252, %v4251 : tensor<256x512x28x28xf32>
    %v4254 = stablehlo.reduce(%v4253 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4255 = stablehlo.reshape %v4055 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4257 = stablehlo.reduce(%v4255 init: %v4256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4258 = stablehlo.reshape %v4176 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4259 = stablehlo.reshape %v442 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4260 = stablehlo.constant dense<0.0> : tensor<256x512x28x28xf32>
    %v4261 = stablehlo.compare GT, %v4259, %v4260 : (tensor<256x512x28x28xf32>, tensor<256x512x28x28xf32>) -> tensor<256x512x28x28xi1>
    %v4262 = stablehlo.select %v4261, %v4258, %v4260 : tensor<256x512x28x28xi1>, tensor<256x512x28x28xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4264 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4266 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4267 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4268 = stablehlo.reduce(%v4264 init: %v4265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4269 = stablehlo.broadcast_in_dim %v4268, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4270 = stablehlo.divide %v4269, %v4266 : tensor<256x512x28x28xf32>
    %v4271 = stablehlo.subtract %v4264, %v4270 : tensor<256x512x28x28xf32>
    %v4272 = stablehlo.multiply %v4271, %v4271 : tensor<256x512x28x28xf32>
    %v4273 = stablehlo.reduce(%v4272 init: %v4265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4274 = stablehlo.broadcast_in_dim %v4273, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4275 = stablehlo.divide %v4274, %v4266 : tensor<256x512x28x28xf32>
    %v4276 = stablehlo.add %v4275, %v4267 : tensor<256x512x28x28xf32>
    %v4277 = stablehlo.rsqrt %v4276 : tensor<256x512x28x28xf32>
    %v4278 = stablehlo.multiply %v4271, %v4277 : tensor<256x512x28x28xf32>
    %v4279 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4280 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4281 = stablehlo.multiply %v4279, %v4280 : tensor<256x512x28x28xf32>
    %v4282 = stablehlo.reduce(%v4281 init: %v4265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4283 = stablehlo.broadcast_in_dim %v4282, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4284 = stablehlo.multiply %v4278, %v4281 : tensor<256x512x28x28xf32>
    %v4285 = stablehlo.reduce(%v4284 init: %v4265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4286 = stablehlo.broadcast_in_dim %v4285, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4287 = stablehlo.multiply %v4281, %v4266 : tensor<256x512x28x28xf32>
    %v4288 = stablehlo.subtract %v4287, %v4283 : tensor<256x512x28x28xf32>
    %v4289 = stablehlo.multiply %v4278, %v4286 : tensor<256x512x28x28xf32>
    %v4290 = stablehlo.subtract %v4288, %v4289 : tensor<256x512x28x28xf32>
    %v4291 = stablehlo.divide %v4277, %v4266 : tensor<256x512x28x28xf32>
    %v4292 = stablehlo.multiply %v4291, %v4290 : tensor<256x512x28x28xf32>
    %v4293 = stablehlo.reshape %v4292 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4294 = stablehlo.reshape %v4293 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4295 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4296 = stablehlo.transpose %v4295, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4297 = stablehlo.convolution(%v4294, %v4296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4299 = stablehlo.reshape %v4298 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4300 = stablehlo.reshape %v384 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4301 = stablehlo.constant dense<0.0> : tensor<256x128x28x28xf32>
    %v4302 = stablehlo.compare GT, %v4300, %v4301 : (tensor<256x128x28x28xf32>, tensor<256x128x28x28xf32>) -> tensor<256x128x28x28xi1>
    %v4303 = stablehlo.select %v4302, %v4299, %v4301 : tensor<256x128x28x28xi1>, tensor<256x128x28x28xf32>
    %v4304 = stablehlo.reshape %v4303 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4305 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4307 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4308 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4309 = stablehlo.reduce(%v4305 init: %v4306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4310 = stablehlo.broadcast_in_dim %v4309, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4311 = stablehlo.divide %v4310, %v4307 : tensor<256x128x28x28xf32>
    %v4312 = stablehlo.subtract %v4305, %v4311 : tensor<256x128x28x28xf32>
    %v4313 = stablehlo.multiply %v4312, %v4312 : tensor<256x128x28x28xf32>
    %v4314 = stablehlo.reduce(%v4313 init: %v4306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4315 = stablehlo.broadcast_in_dim %v4314, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4316 = stablehlo.divide %v4315, %v4307 : tensor<256x128x28x28xf32>
    %v4317 = stablehlo.add %v4316, %v4308 : tensor<256x128x28x28xf32>
    %v4318 = stablehlo.rsqrt %v4317 : tensor<256x128x28x28xf32>
    %v4319 = stablehlo.multiply %v4312, %v4318 : tensor<256x128x28x28xf32>
    %v4320 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4321 = stablehlo.reshape %v4304 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4322 = stablehlo.multiply %v4320, %v4321 : tensor<256x128x28x28xf32>
    %v4323 = stablehlo.reduce(%v4322 init: %v4306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4324 = stablehlo.broadcast_in_dim %v4323, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4325 = stablehlo.multiply %v4319, %v4322 : tensor<256x128x28x28xf32>
    %v4326 = stablehlo.reduce(%v4325 init: %v4306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4327 = stablehlo.broadcast_in_dim %v4326, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4328 = stablehlo.multiply %v4322, %v4307 : tensor<256x128x28x28xf32>
    %v4329 = stablehlo.subtract %v4328, %v4324 : tensor<256x128x28x28xf32>
    %v4330 = stablehlo.multiply %v4319, %v4327 : tensor<256x128x28x28xf32>
    %v4331 = stablehlo.subtract %v4329, %v4330 : tensor<256x128x28x28xf32>
    %v4332 = stablehlo.divide %v4318, %v4307 : tensor<256x128x28x28xf32>
    %v4333 = stablehlo.multiply %v4332, %v4331 : tensor<256x128x28x28xf32>
    %v4334 = stablehlo.reshape %v4333 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4335 = stablehlo.reshape %v4334 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4337 = stablehlo.pad %v4335, %v4336, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v4338 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4339 = stablehlo.transpose %v4338, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4340 = stablehlo.convolution(%v4337, %v4339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x56x56xf32>
    %v4341 = stablehlo.reshape %v4340 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4342 = stablehlo.reshape %v4341 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4343 = stablehlo.reshape %v355 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4344 = stablehlo.constant dense<0.0> : tensor<256x128x56x56xf32>
    %v4345 = stablehlo.compare GT, %v4343, %v4344 : (tensor<256x128x56x56xf32>, tensor<256x128x56x56xf32>) -> tensor<256x128x56x56xi1>
    %v4346 = stablehlo.select %v4345, %v4342, %v4344 : tensor<256x128x56x56xi1>, tensor<256x128x56x56xf32>
    %v4347 = stablehlo.reshape %v4346 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4348 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4350 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4351 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4352 = stablehlo.reduce(%v4348 init: %v4349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4353 = stablehlo.broadcast_in_dim %v4352, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4354 = stablehlo.divide %v4353, %v4350 : tensor<256x128x56x56xf32>
    %v4355 = stablehlo.subtract %v4348, %v4354 : tensor<256x128x56x56xf32>
    %v4356 = stablehlo.multiply %v4355, %v4355 : tensor<256x128x56x56xf32>
    %v4357 = stablehlo.reduce(%v4356 init: %v4349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4358 = stablehlo.broadcast_in_dim %v4357, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4359 = stablehlo.divide %v4358, %v4350 : tensor<256x128x56x56xf32>
    %v4360 = stablehlo.add %v4359, %v4351 : tensor<256x128x56x56xf32>
    %v4361 = stablehlo.rsqrt %v4360 : tensor<256x128x56x56xf32>
    %v4362 = stablehlo.multiply %v4355, %v4361 : tensor<256x128x56x56xf32>
    %v4363 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4364 = stablehlo.reshape %v4347 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4365 = stablehlo.multiply %v4363, %v4364 : tensor<256x128x56x56xf32>
    %v4366 = stablehlo.reduce(%v4365 init: %v4349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4367 = stablehlo.broadcast_in_dim %v4366, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4368 = stablehlo.multiply %v4362, %v4365 : tensor<256x128x56x56xf32>
    %v4369 = stablehlo.reduce(%v4368 init: %v4349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4370 = stablehlo.broadcast_in_dim %v4369, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4371 = stablehlo.multiply %v4365, %v4350 : tensor<256x128x56x56xf32>
    %v4372 = stablehlo.subtract %v4371, %v4367 : tensor<256x128x56x56xf32>
    %v4373 = stablehlo.multiply %v4362, %v4370 : tensor<256x128x56x56xf32>
    %v4374 = stablehlo.subtract %v4372, %v4373 : tensor<256x128x56x56xf32>
    %v4375 = stablehlo.divide %v4361, %v4350 : tensor<256x128x56x56xf32>
    %v4376 = stablehlo.multiply %v4375, %v4374 : tensor<256x128x56x56xf32>
    %v4377 = stablehlo.reshape %v4376 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4378 = stablehlo.reshape %v4377 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4379 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v4380 = stablehlo.transpose %v4379, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v4381 = stablehlo.convolution(%v4378, %v4380)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4382 = stablehlo.reshape %v4381 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4383 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4385 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4386 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4387 = stablehlo.reduce(%v4383 init: %v4384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4388 = stablehlo.broadcast_in_dim %v4387, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4389 = stablehlo.divide %v4388, %v4385 : tensor<256x512x28x28xf32>
    %v4390 = stablehlo.subtract %v4383, %v4389 : tensor<256x512x28x28xf32>
    %v4391 = stablehlo.multiply %v4390, %v4390 : tensor<256x512x28x28xf32>
    %v4392 = stablehlo.reduce(%v4391 init: %v4384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4393 = stablehlo.broadcast_in_dim %v4392, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4394 = stablehlo.divide %v4393, %v4385 : tensor<256x512x28x28xf32>
    %v4395 = stablehlo.add %v4394, %v4386 : tensor<256x512x28x28xf32>
    %v4396 = stablehlo.rsqrt %v4395 : tensor<256x512x28x28xf32>
    %v4397 = stablehlo.multiply %v4390, %v4396 : tensor<256x512x28x28xf32>
    %v4398 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4399 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4400 = stablehlo.multiply %v4398, %v4399 : tensor<256x512x28x28xf32>
    %v4401 = stablehlo.reduce(%v4400 init: %v4384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4402 = stablehlo.broadcast_in_dim %v4401, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4403 = stablehlo.multiply %v4397, %v4400 : tensor<256x512x28x28xf32>
    %v4404 = stablehlo.reduce(%v4403 init: %v4384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4405 = stablehlo.broadcast_in_dim %v4404, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4406 = stablehlo.multiply %v4400, %v4385 : tensor<256x512x28x28xf32>
    %v4407 = stablehlo.subtract %v4406, %v4402 : tensor<256x512x28x28xf32>
    %v4408 = stablehlo.multiply %v4397, %v4405 : tensor<256x512x28x28xf32>
    %v4409 = stablehlo.subtract %v4407, %v4408 : tensor<256x512x28x28xf32>
    %v4410 = stablehlo.divide %v4396, %v4385 : tensor<256x512x28x28xf32>
    %v4411 = stablehlo.multiply %v4410, %v4409 : tensor<256x512x28x28xf32>
    %v4412 = stablehlo.reshape %v4411 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4413 = stablehlo.reshape %v4412 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4415 = stablehlo.pad %v4413, %v4414, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<256x512x56x56xf32>
    %v4416 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v4417 = stablehlo.transpose %v4416, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4418 = stablehlo.convolution(%v4415, %v4417)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x56x56xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4419 = stablehlo.reshape %v4418 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4420 = stablehlo.reshape %v4382 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4421 = stablehlo.reshape %v4419 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4422 = stablehlo.add %v4420, %v4421 : tensor<256x256x56x56xf32>
    %v4423 = stablehlo.reshape %v4422 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4424 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4425 = stablehlo.reshape %v4377 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4426 = stablehlo.transpose %v4424, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4427 = stablehlo.transpose %v4425, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4428 = stablehlo.convolution(%v4426, %v4427)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<256x128x1x1xf32>
    %v4429 = stablehlo.transpose %v4428, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v4430 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4432 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4433 = stablehlo.reduce(%v4430 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4434 = stablehlo.broadcast_in_dim %v4433, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4435 = stablehlo.divide %v4434, %v4432 : tensor<256x128x56x56xf32>
    %v4436 = stablehlo.subtract %v4430, %v4435 : tensor<256x128x56x56xf32>
    %v4437 = stablehlo.multiply %v4436, %v4436 : tensor<256x128x56x56xf32>
    %v4438 = stablehlo.reduce(%v4437 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4439 = stablehlo.broadcast_in_dim %v4438, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4440 = stablehlo.divide %v4439, %v4432 : tensor<256x128x56x56xf32>
    %v4441 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4442 = stablehlo.add %v4440, %v4441 : tensor<256x128x56x56xf32>
    %v4443 = stablehlo.rsqrt %v4442 : tensor<256x128x56x56xf32>
    %v4444 = stablehlo.multiply %v4436, %v4443 : tensor<256x128x56x56xf32>
    %v4445 = stablehlo.reshape %v4347 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4446 = stablehlo.multiply %v4445, %v4444 : tensor<256x128x56x56xf32>
    %v4447 = stablehlo.reduce(%v4446 init: %v4431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4448 = stablehlo.reshape %v4347 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4449 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4450 = stablehlo.reduce(%v4448 init: %v4449) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4451 = stablehlo.reshape %v359 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4452 = stablehlo.reshape %v4334 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4454 = stablehlo.pad %v4452, %v4453, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v4455 = stablehlo.transpose %v4451, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4456 = stablehlo.transpose %v4454, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4457 = stablehlo.convolution(%v4455, %v4456)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<128x128x3x3xf32>
    %v4458 = stablehlo.transpose %v4457, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4459 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4461 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4462 = stablehlo.reduce(%v4459 init: %v4460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4463 = stablehlo.broadcast_in_dim %v4462, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4464 = stablehlo.divide %v4463, %v4461 : tensor<256x128x28x28xf32>
    %v4465 = stablehlo.subtract %v4459, %v4464 : tensor<256x128x28x28xf32>
    %v4466 = stablehlo.multiply %v4465, %v4465 : tensor<256x128x28x28xf32>
    %v4467 = stablehlo.reduce(%v4466 init: %v4460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4468 = stablehlo.broadcast_in_dim %v4467, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4469 = stablehlo.divide %v4468, %v4461 : tensor<256x128x28x28xf32>
    %v4470 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4471 = stablehlo.add %v4469, %v4470 : tensor<256x128x28x28xf32>
    %v4472 = stablehlo.rsqrt %v4471 : tensor<256x128x28x28xf32>
    %v4473 = stablehlo.multiply %v4465, %v4472 : tensor<256x128x28x28xf32>
    %v4474 = stablehlo.reshape %v4304 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4475 = stablehlo.multiply %v4474, %v4473 : tensor<256x128x28x28xf32>
    %v4476 = stablehlo.reduce(%v4475 init: %v4460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4477 = stablehlo.reshape %v4304 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4479 = stablehlo.reduce(%v4477 init: %v4478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4480 = stablehlo.reshape %v388 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4481 = stablehlo.reshape %v4293 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4482 = stablehlo.transpose %v4480, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4483 = stablehlo.transpose %v4481, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4484 = stablehlo.convolution(%v4482, %v4483)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4485 = stablehlo.transpose %v4484, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4486 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4488 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4489 = stablehlo.reduce(%v4486 init: %v4487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4490 = stablehlo.broadcast_in_dim %v4489, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4491 = stablehlo.divide %v4490, %v4488 : tensor<256x512x28x28xf32>
    %v4492 = stablehlo.subtract %v4486, %v4491 : tensor<256x512x28x28xf32>
    %v4493 = stablehlo.multiply %v4492, %v4492 : tensor<256x512x28x28xf32>
    %v4494 = stablehlo.reduce(%v4493 init: %v4487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4495 = stablehlo.broadcast_in_dim %v4494, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4496 = stablehlo.divide %v4495, %v4488 : tensor<256x512x28x28xf32>
    %v4497 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4498 = stablehlo.add %v4496, %v4497 : tensor<256x512x28x28xf32>
    %v4499 = stablehlo.rsqrt %v4498 : tensor<256x512x28x28xf32>
    %v4500 = stablehlo.multiply %v4492, %v4499 : tensor<256x512x28x28xf32>
    %v4501 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4502 = stablehlo.multiply %v4501, %v4500 : tensor<256x512x28x28xf32>
    %v4503 = stablehlo.reduce(%v4502 init: %v4487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4504 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4506 = stablehlo.reduce(%v4504 init: %v4505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4507 = stablehlo.reshape %v330 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4508 = stablehlo.reshape %v4412 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4509 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4510 = stablehlo.pad %v4508, %v4509, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<256x512x56x56xf32>
    %v4511 = stablehlo.transpose %v4507, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4512 = stablehlo.transpose %v4510, dims = [1, 0, 2, 3] : (tensor<256x512x56x56xf32>) -> tensor<512x256x56x56xf32>
    %v4513 = stablehlo.convolution(%v4511, %v4512)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x56x56xf32>) -> tensor<256x512x1x1xf32>
    %v4514 = stablehlo.transpose %v4513, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4515 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4517 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4518 = stablehlo.reduce(%v4515 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4519 = stablehlo.broadcast_in_dim %v4518, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4520 = stablehlo.divide %v4519, %v4517 : tensor<256x512x28x28xf32>
    %v4521 = stablehlo.subtract %v4515, %v4520 : tensor<256x512x28x28xf32>
    %v4522 = stablehlo.multiply %v4521, %v4521 : tensor<256x512x28x28xf32>
    %v4523 = stablehlo.reduce(%v4522 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4524 = stablehlo.broadcast_in_dim %v4523, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4525 = stablehlo.divide %v4524, %v4517 : tensor<256x512x28x28xf32>
    %v4526 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4527 = stablehlo.add %v4525, %v4526 : tensor<256x512x28x28xf32>
    %v4528 = stablehlo.rsqrt %v4527 : tensor<256x512x28x28xf32>
    %v4529 = stablehlo.multiply %v4521, %v4528 : tensor<256x512x28x28xf32>
    %v4530 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4531 = stablehlo.multiply %v4530, %v4529 : tensor<256x512x28x28xf32>
    %v4532 = stablehlo.reduce(%v4531 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4533 = stablehlo.reshape %v4263 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4535 = stablehlo.reduce(%v4533 init: %v4534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4536 = stablehlo.reshape %v4423 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4537 = stablehlo.reshape %v326 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4538 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4539 = stablehlo.compare GT, %v4537, %v4538 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4540 = stablehlo.select %v4539, %v4536, %v4538 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4541 = stablehlo.reshape %v4540 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4542 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4544 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4545 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4546 = stablehlo.reduce(%v4542 init: %v4543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4547 = stablehlo.broadcast_in_dim %v4546, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4548 = stablehlo.divide %v4547, %v4544 : tensor<256x256x56x56xf32>
    %v4549 = stablehlo.subtract %v4542, %v4548 : tensor<256x256x56x56xf32>
    %v4550 = stablehlo.multiply %v4549, %v4549 : tensor<256x256x56x56xf32>
    %v4551 = stablehlo.reduce(%v4550 init: %v4543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4552 = stablehlo.broadcast_in_dim %v4551, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4553 = stablehlo.divide %v4552, %v4544 : tensor<256x256x56x56xf32>
    %v4554 = stablehlo.add %v4553, %v4545 : tensor<256x256x56x56xf32>
    %v4555 = stablehlo.rsqrt %v4554 : tensor<256x256x56x56xf32>
    %v4556 = stablehlo.multiply %v4549, %v4555 : tensor<256x256x56x56xf32>
    %v4557 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4558 = stablehlo.reshape %v4541 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4559 = stablehlo.multiply %v4557, %v4558 : tensor<256x256x56x56xf32>
    %v4560 = stablehlo.reduce(%v4559 init: %v4543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4561 = stablehlo.broadcast_in_dim %v4560, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4562 = stablehlo.multiply %v4556, %v4559 : tensor<256x256x56x56xf32>
    %v4563 = stablehlo.reduce(%v4562 init: %v4543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4564 = stablehlo.broadcast_in_dim %v4563, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4565 = stablehlo.multiply %v4559, %v4544 : tensor<256x256x56x56xf32>
    %v4566 = stablehlo.subtract %v4565, %v4561 : tensor<256x256x56x56xf32>
    %v4567 = stablehlo.multiply %v4556, %v4564 : tensor<256x256x56x56xf32>
    %v4568 = stablehlo.subtract %v4566, %v4567 : tensor<256x256x56x56xf32>
    %v4569 = stablehlo.divide %v4555, %v4544 : tensor<256x256x56x56xf32>
    %v4570 = stablehlo.multiply %v4569, %v4568 : tensor<256x256x56x56xf32>
    %v4571 = stablehlo.reshape %v4570 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4572 = stablehlo.reshape %v4571 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4573 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4574 = stablehlo.transpose %v4573, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4575 = stablehlo.convolution(%v4572, %v4574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4576 = stablehlo.reshape %v4575 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4577 = stablehlo.reshape %v4576 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4578 = stablehlo.reshape %v293 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4579 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4580 = stablehlo.compare GT, %v4578, %v4579 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4581 = stablehlo.select %v4580, %v4577, %v4579 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4582 = stablehlo.reshape %v4581 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4583 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4585 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4586 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4587 = stablehlo.reduce(%v4583 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4588 = stablehlo.broadcast_in_dim %v4587, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4589 = stablehlo.divide %v4588, %v4585 : tensor<256x64x56x56xf32>
    %v4590 = stablehlo.subtract %v4583, %v4589 : tensor<256x64x56x56xf32>
    %v4591 = stablehlo.multiply %v4590, %v4590 : tensor<256x64x56x56xf32>
    %v4592 = stablehlo.reduce(%v4591 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4593 = stablehlo.broadcast_in_dim %v4592, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4594 = stablehlo.divide %v4593, %v4585 : tensor<256x64x56x56xf32>
    %v4595 = stablehlo.add %v4594, %v4586 : tensor<256x64x56x56xf32>
    %v4596 = stablehlo.rsqrt %v4595 : tensor<256x64x56x56xf32>
    %v4597 = stablehlo.multiply %v4590, %v4596 : tensor<256x64x56x56xf32>
    %v4598 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4599 = stablehlo.reshape %v4582 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4600 = stablehlo.multiply %v4598, %v4599 : tensor<256x64x56x56xf32>
    %v4601 = stablehlo.reduce(%v4600 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4602 = stablehlo.broadcast_in_dim %v4601, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4603 = stablehlo.multiply %v4597, %v4600 : tensor<256x64x56x56xf32>
    %v4604 = stablehlo.reduce(%v4603 init: %v4584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4605 = stablehlo.broadcast_in_dim %v4604, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4606 = stablehlo.multiply %v4600, %v4585 : tensor<256x64x56x56xf32>
    %v4607 = stablehlo.subtract %v4606, %v4602 : tensor<256x64x56x56xf32>
    %v4608 = stablehlo.multiply %v4597, %v4605 : tensor<256x64x56x56xf32>
    %v4609 = stablehlo.subtract %v4607, %v4608 : tensor<256x64x56x56xf32>
    %v4610 = stablehlo.divide %v4596, %v4585 : tensor<256x64x56x56xf32>
    %v4611 = stablehlo.multiply %v4610, %v4609 : tensor<256x64x56x56xf32>
    %v4612 = stablehlo.reshape %v4611 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4613 = stablehlo.reshape %v4612 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4614 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4615 = stablehlo.transpose %v4614, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4616 = stablehlo.convolution(%v4613, %v4615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4617 = stablehlo.reshape %v4616 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4618 = stablehlo.reshape %v4617 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4619 = stablehlo.reshape %v264 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4620 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4621 = stablehlo.compare GT, %v4619, %v4620 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4622 = stablehlo.select %v4621, %v4618, %v4620 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4623 = stablehlo.reshape %v4622 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4624 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4626 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4627 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4628 = stablehlo.reduce(%v4624 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4629 = stablehlo.broadcast_in_dim %v4628, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4630 = stablehlo.divide %v4629, %v4626 : tensor<256x64x56x56xf32>
    %v4631 = stablehlo.subtract %v4624, %v4630 : tensor<256x64x56x56xf32>
    %v4632 = stablehlo.multiply %v4631, %v4631 : tensor<256x64x56x56xf32>
    %v4633 = stablehlo.reduce(%v4632 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4634 = stablehlo.broadcast_in_dim %v4633, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4635 = stablehlo.divide %v4634, %v4626 : tensor<256x64x56x56xf32>
    %v4636 = stablehlo.add %v4635, %v4627 : tensor<256x64x56x56xf32>
    %v4637 = stablehlo.rsqrt %v4636 : tensor<256x64x56x56xf32>
    %v4638 = stablehlo.multiply %v4631, %v4637 : tensor<256x64x56x56xf32>
    %v4639 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4640 = stablehlo.reshape %v4623 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4641 = stablehlo.multiply %v4639, %v4640 : tensor<256x64x56x56xf32>
    %v4642 = stablehlo.reduce(%v4641 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4643 = stablehlo.broadcast_in_dim %v4642, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4644 = stablehlo.multiply %v4638, %v4641 : tensor<256x64x56x56xf32>
    %v4645 = stablehlo.reduce(%v4644 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4646 = stablehlo.broadcast_in_dim %v4645, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4647 = stablehlo.multiply %v4641, %v4626 : tensor<256x64x56x56xf32>
    %v4648 = stablehlo.subtract %v4647, %v4643 : tensor<256x64x56x56xf32>
    %v4649 = stablehlo.multiply %v4638, %v4646 : tensor<256x64x56x56xf32>
    %v4650 = stablehlo.subtract %v4648, %v4649 : tensor<256x64x56x56xf32>
    %v4651 = stablehlo.divide %v4637, %v4626 : tensor<256x64x56x56xf32>
    %v4652 = stablehlo.multiply %v4651, %v4650 : tensor<256x64x56x56xf32>
    %v4653 = stablehlo.reshape %v4652 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4654 = stablehlo.reshape %v4653 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4655 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4656 = stablehlo.transpose %v4655, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4657 = stablehlo.convolution(%v4654, %v4656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4658 = stablehlo.reshape %v4657 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4659 = stablehlo.reshape %v4658 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4660 = stablehlo.reshape %v4541 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4661 = stablehlo.add %v4659, %v4660 : tensor<256x256x56x56xf32>
    %v4662 = stablehlo.reshape %v4661 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4663 = stablehlo.reshape %v239 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4664 = stablehlo.reshape %v4653 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4665 = stablehlo.transpose %v4663, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4666 = stablehlo.transpose %v4664, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4667 = stablehlo.convolution(%v4665, %v4666)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4668 = stablehlo.transpose %v4667, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4669 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4671 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4672 = stablehlo.reduce(%v4669 init: %v4670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4673 = stablehlo.broadcast_in_dim %v4672, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4674 = stablehlo.divide %v4673, %v4671 : tensor<256x64x56x56xf32>
    %v4675 = stablehlo.subtract %v4669, %v4674 : tensor<256x64x56x56xf32>
    %v4676 = stablehlo.multiply %v4675, %v4675 : tensor<256x64x56x56xf32>
    %v4677 = stablehlo.reduce(%v4676 init: %v4670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4678 = stablehlo.broadcast_in_dim %v4677, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4679 = stablehlo.divide %v4678, %v4671 : tensor<256x64x56x56xf32>
    %v4680 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4681 = stablehlo.add %v4679, %v4680 : tensor<256x64x56x56xf32>
    %v4682 = stablehlo.rsqrt %v4681 : tensor<256x64x56x56xf32>
    %v4683 = stablehlo.multiply %v4675, %v4682 : tensor<256x64x56x56xf32>
    %v4684 = stablehlo.reshape %v4623 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4685 = stablehlo.multiply %v4684, %v4683 : tensor<256x64x56x56xf32>
    %v4686 = stablehlo.reduce(%v4685 init: %v4670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4687 = stablehlo.reshape %v4623 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4689 = stablehlo.reduce(%v4687 init: %v4688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4690 = stablehlo.reshape %v268 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4691 = stablehlo.reshape %v4612 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4692 = stablehlo.transpose %v4690, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4693 = stablehlo.transpose %v4691, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4694 = stablehlo.convolution(%v4692, %v4693)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4695 = stablehlo.transpose %v4694, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4696 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4698 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4699 = stablehlo.reduce(%v4696 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4700 = stablehlo.broadcast_in_dim %v4699, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4701 = stablehlo.divide %v4700, %v4698 : tensor<256x64x56x56xf32>
    %v4702 = stablehlo.subtract %v4696, %v4701 : tensor<256x64x56x56xf32>
    %v4703 = stablehlo.multiply %v4702, %v4702 : tensor<256x64x56x56xf32>
    %v4704 = stablehlo.reduce(%v4703 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4705 = stablehlo.broadcast_in_dim %v4704, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4706 = stablehlo.divide %v4705, %v4698 : tensor<256x64x56x56xf32>
    %v4707 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4708 = stablehlo.add %v4706, %v4707 : tensor<256x64x56x56xf32>
    %v4709 = stablehlo.rsqrt %v4708 : tensor<256x64x56x56xf32>
    %v4710 = stablehlo.multiply %v4702, %v4709 : tensor<256x64x56x56xf32>
    %v4711 = stablehlo.reshape %v4582 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4712 = stablehlo.multiply %v4711, %v4710 : tensor<256x64x56x56xf32>
    %v4713 = stablehlo.reduce(%v4712 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4714 = stablehlo.reshape %v4582 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4716 = stablehlo.reduce(%v4714 init: %v4715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4717 = stablehlo.reshape %v297 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4718 = stablehlo.reshape %v4571 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4719 = stablehlo.transpose %v4717, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4720 = stablehlo.transpose %v4718, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4721 = stablehlo.convolution(%v4719, %v4720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4722 = stablehlo.transpose %v4721, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4723 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4724 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4725 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4726 = stablehlo.reduce(%v4723 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4727 = stablehlo.broadcast_in_dim %v4726, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4728 = stablehlo.divide %v4727, %v4725 : tensor<256x256x56x56xf32>
    %v4729 = stablehlo.subtract %v4723, %v4728 : tensor<256x256x56x56xf32>
    %v4730 = stablehlo.multiply %v4729, %v4729 : tensor<256x256x56x56xf32>
    %v4731 = stablehlo.reduce(%v4730 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4732 = stablehlo.broadcast_in_dim %v4731, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4733 = stablehlo.divide %v4732, %v4725 : tensor<256x256x56x56xf32>
    %v4734 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4735 = stablehlo.add %v4733, %v4734 : tensor<256x256x56x56xf32>
    %v4736 = stablehlo.rsqrt %v4735 : tensor<256x256x56x56xf32>
    %v4737 = stablehlo.multiply %v4729, %v4736 : tensor<256x256x56x56xf32>
    %v4738 = stablehlo.reshape %v4541 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4739 = stablehlo.multiply %v4738, %v4737 : tensor<256x256x56x56xf32>
    %v4740 = stablehlo.reduce(%v4739 init: %v4724) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4741 = stablehlo.reshape %v4541 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4743 = stablehlo.reduce(%v4741 init: %v4742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4744 = stablehlo.reshape %v4662 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4745 = stablehlo.reshape %v235 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4746 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4747 = stablehlo.compare GT, %v4745, %v4746 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4748 = stablehlo.select %v4747, %v4744, %v4746 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4749 = stablehlo.reshape %v4748 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4750 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4752 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4753 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4754 = stablehlo.reduce(%v4750 init: %v4751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4755 = stablehlo.broadcast_in_dim %v4754, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4756 = stablehlo.divide %v4755, %v4752 : tensor<256x256x56x56xf32>
    %v4757 = stablehlo.subtract %v4750, %v4756 : tensor<256x256x56x56xf32>
    %v4758 = stablehlo.multiply %v4757, %v4757 : tensor<256x256x56x56xf32>
    %v4759 = stablehlo.reduce(%v4758 init: %v4751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4760 = stablehlo.broadcast_in_dim %v4759, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4761 = stablehlo.divide %v4760, %v4752 : tensor<256x256x56x56xf32>
    %v4762 = stablehlo.add %v4761, %v4753 : tensor<256x256x56x56xf32>
    %v4763 = stablehlo.rsqrt %v4762 : tensor<256x256x56x56xf32>
    %v4764 = stablehlo.multiply %v4757, %v4763 : tensor<256x256x56x56xf32>
    %v4765 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4766 = stablehlo.reshape %v4749 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4767 = stablehlo.multiply %v4765, %v4766 : tensor<256x256x56x56xf32>
    %v4768 = stablehlo.reduce(%v4767 init: %v4751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4769 = stablehlo.broadcast_in_dim %v4768, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4770 = stablehlo.multiply %v4764, %v4767 : tensor<256x256x56x56xf32>
    %v4771 = stablehlo.reduce(%v4770 init: %v4751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4772 = stablehlo.broadcast_in_dim %v4771, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4773 = stablehlo.multiply %v4767, %v4752 : tensor<256x256x56x56xf32>
    %v4774 = stablehlo.subtract %v4773, %v4769 : tensor<256x256x56x56xf32>
    %v4775 = stablehlo.multiply %v4764, %v4772 : tensor<256x256x56x56xf32>
    %v4776 = stablehlo.subtract %v4774, %v4775 : tensor<256x256x56x56xf32>
    %v4777 = stablehlo.divide %v4763, %v4752 : tensor<256x256x56x56xf32>
    %v4778 = stablehlo.multiply %v4777, %v4776 : tensor<256x256x56x56xf32>
    %v4779 = stablehlo.reshape %v4778 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4780 = stablehlo.reshape %v4779 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4781 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4782 = stablehlo.transpose %v4781, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4783 = stablehlo.convolution(%v4780, %v4782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4784 = stablehlo.reshape %v4783 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4785 = stablehlo.reshape %v4784 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4786 = stablehlo.reshape %v202 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4787 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4788 = stablehlo.compare GT, %v4786, %v4787 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4789 = stablehlo.select %v4788, %v4785, %v4787 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4790 = stablehlo.reshape %v4789 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4791 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4793 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4794 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4795 = stablehlo.reduce(%v4791 init: %v4792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4796 = stablehlo.broadcast_in_dim %v4795, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4797 = stablehlo.divide %v4796, %v4793 : tensor<256x64x56x56xf32>
    %v4798 = stablehlo.subtract %v4791, %v4797 : tensor<256x64x56x56xf32>
    %v4799 = stablehlo.multiply %v4798, %v4798 : tensor<256x64x56x56xf32>
    %v4800 = stablehlo.reduce(%v4799 init: %v4792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4801 = stablehlo.broadcast_in_dim %v4800, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4802 = stablehlo.divide %v4801, %v4793 : tensor<256x64x56x56xf32>
    %v4803 = stablehlo.add %v4802, %v4794 : tensor<256x64x56x56xf32>
    %v4804 = stablehlo.rsqrt %v4803 : tensor<256x64x56x56xf32>
    %v4805 = stablehlo.multiply %v4798, %v4804 : tensor<256x64x56x56xf32>
    %v4806 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4807 = stablehlo.reshape %v4790 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4808 = stablehlo.multiply %v4806, %v4807 : tensor<256x64x56x56xf32>
    %v4809 = stablehlo.reduce(%v4808 init: %v4792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4810 = stablehlo.broadcast_in_dim %v4809, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4811 = stablehlo.multiply %v4805, %v4808 : tensor<256x64x56x56xf32>
    %v4812 = stablehlo.reduce(%v4811 init: %v4792) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4813 = stablehlo.broadcast_in_dim %v4812, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4814 = stablehlo.multiply %v4808, %v4793 : tensor<256x64x56x56xf32>
    %v4815 = stablehlo.subtract %v4814, %v4810 : tensor<256x64x56x56xf32>
    %v4816 = stablehlo.multiply %v4805, %v4813 : tensor<256x64x56x56xf32>
    %v4817 = stablehlo.subtract %v4815, %v4816 : tensor<256x64x56x56xf32>
    %v4818 = stablehlo.divide %v4804, %v4793 : tensor<256x64x56x56xf32>
    %v4819 = stablehlo.multiply %v4818, %v4817 : tensor<256x64x56x56xf32>
    %v4820 = stablehlo.reshape %v4819 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4821 = stablehlo.reshape %v4820 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4822 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4823 = stablehlo.transpose %v4822, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4824 = stablehlo.convolution(%v4821, %v4823)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4825 = stablehlo.reshape %v4824 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4826 = stablehlo.reshape %v4825 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4827 = stablehlo.reshape %v173 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4828 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4829 = stablehlo.compare GT, %v4827, %v4828 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4830 = stablehlo.select %v4829, %v4826, %v4828 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4831 = stablehlo.reshape %v4830 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4832 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4834 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4835 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4836 = stablehlo.reduce(%v4832 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4837 = stablehlo.broadcast_in_dim %v4836, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4838 = stablehlo.divide %v4837, %v4834 : tensor<256x64x56x56xf32>
    %v4839 = stablehlo.subtract %v4832, %v4838 : tensor<256x64x56x56xf32>
    %v4840 = stablehlo.multiply %v4839, %v4839 : tensor<256x64x56x56xf32>
    %v4841 = stablehlo.reduce(%v4840 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4842 = stablehlo.broadcast_in_dim %v4841, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4843 = stablehlo.divide %v4842, %v4834 : tensor<256x64x56x56xf32>
    %v4844 = stablehlo.add %v4843, %v4835 : tensor<256x64x56x56xf32>
    %v4845 = stablehlo.rsqrt %v4844 : tensor<256x64x56x56xf32>
    %v4846 = stablehlo.multiply %v4839, %v4845 : tensor<256x64x56x56xf32>
    %v4847 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4848 = stablehlo.reshape %v4831 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4849 = stablehlo.multiply %v4847, %v4848 : tensor<256x64x56x56xf32>
    %v4850 = stablehlo.reduce(%v4849 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4851 = stablehlo.broadcast_in_dim %v4850, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4852 = stablehlo.multiply %v4846, %v4849 : tensor<256x64x56x56xf32>
    %v4853 = stablehlo.reduce(%v4852 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4854 = stablehlo.broadcast_in_dim %v4853, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4855 = stablehlo.multiply %v4849, %v4834 : tensor<256x64x56x56xf32>
    %v4856 = stablehlo.subtract %v4855, %v4851 : tensor<256x64x56x56xf32>
    %v4857 = stablehlo.multiply %v4846, %v4854 : tensor<256x64x56x56xf32>
    %v4858 = stablehlo.subtract %v4856, %v4857 : tensor<256x64x56x56xf32>
    %v4859 = stablehlo.divide %v4845, %v4834 : tensor<256x64x56x56xf32>
    %v4860 = stablehlo.multiply %v4859, %v4858 : tensor<256x64x56x56xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4862 = stablehlo.reshape %v4861 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4863 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4864 = stablehlo.transpose %v4863, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4865 = stablehlo.convolution(%v4862, %v4864)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4866 = stablehlo.reshape %v4865 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4867 = stablehlo.reshape %v4866 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4868 = stablehlo.reshape %v4749 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4869 = stablehlo.add %v4867, %v4868 : tensor<256x256x56x56xf32>
    %v4870 = stablehlo.reshape %v4869 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4871 = stablehlo.reshape %v148 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4872 = stablehlo.reshape %v4861 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4873 = stablehlo.transpose %v4871, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4874 = stablehlo.transpose %v4872, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4875 = stablehlo.convolution(%v4873, %v4874)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4876 = stablehlo.transpose %v4875, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4877 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4878 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4879 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4880 = stablehlo.reduce(%v4877 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4881 = stablehlo.broadcast_in_dim %v4880, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4882 = stablehlo.divide %v4881, %v4879 : tensor<256x64x56x56xf32>
    %v4883 = stablehlo.subtract %v4877, %v4882 : tensor<256x64x56x56xf32>
    %v4884 = stablehlo.multiply %v4883, %v4883 : tensor<256x64x56x56xf32>
    %v4885 = stablehlo.reduce(%v4884 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4886 = stablehlo.broadcast_in_dim %v4885, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4887 = stablehlo.divide %v4886, %v4879 : tensor<256x64x56x56xf32>
    %v4888 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4889 = stablehlo.add %v4887, %v4888 : tensor<256x64x56x56xf32>
    %v4890 = stablehlo.rsqrt %v4889 : tensor<256x64x56x56xf32>
    %v4891 = stablehlo.multiply %v4883, %v4890 : tensor<256x64x56x56xf32>
    %v4892 = stablehlo.reshape %v4831 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4893 = stablehlo.multiply %v4892, %v4891 : tensor<256x64x56x56xf32>
    %v4894 = stablehlo.reduce(%v4893 init: %v4878) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4895 = stablehlo.reshape %v4831 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4896 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4897 = stablehlo.reduce(%v4895 init: %v4896) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4898 = stablehlo.reshape %v177 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4899 = stablehlo.reshape %v4820 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4900 = stablehlo.transpose %v4898, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4901 = stablehlo.transpose %v4899, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4902 = stablehlo.convolution(%v4900, %v4901)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4903 = stablehlo.transpose %v4902, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4904 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4905 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4906 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4907 = stablehlo.reduce(%v4904 init: %v4905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4908 = stablehlo.broadcast_in_dim %v4907, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4909 = stablehlo.divide %v4908, %v4906 : tensor<256x64x56x56xf32>
    %v4910 = stablehlo.subtract %v4904, %v4909 : tensor<256x64x56x56xf32>
    %v4911 = stablehlo.multiply %v4910, %v4910 : tensor<256x64x56x56xf32>
    %v4912 = stablehlo.reduce(%v4911 init: %v4905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4913 = stablehlo.broadcast_in_dim %v4912, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4914 = stablehlo.divide %v4913, %v4906 : tensor<256x64x56x56xf32>
    %v4915 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4916 = stablehlo.add %v4914, %v4915 : tensor<256x64x56x56xf32>
    %v4917 = stablehlo.rsqrt %v4916 : tensor<256x64x56x56xf32>
    %v4918 = stablehlo.multiply %v4910, %v4917 : tensor<256x64x56x56xf32>
    %v4919 = stablehlo.reshape %v4790 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4920 = stablehlo.multiply %v4919, %v4918 : tensor<256x64x56x56xf32>
    %v4921 = stablehlo.reduce(%v4920 init: %v4905) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4922 = stablehlo.reshape %v4790 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4923 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4924 = stablehlo.reduce(%v4922 init: %v4923) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4925 = stablehlo.reshape %v206 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4926 = stablehlo.reshape %v4779 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4927 = stablehlo.transpose %v4925, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4928 = stablehlo.transpose %v4926, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4929 = stablehlo.convolution(%v4927, %v4928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4930 = stablehlo.transpose %v4929, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4931 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4933 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4934 = stablehlo.reduce(%v4931 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4935 = stablehlo.broadcast_in_dim %v4934, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4936 = stablehlo.divide %v4935, %v4933 : tensor<256x256x56x56xf32>
    %v4937 = stablehlo.subtract %v4931, %v4936 : tensor<256x256x56x56xf32>
    %v4938 = stablehlo.multiply %v4937, %v4937 : tensor<256x256x56x56xf32>
    %v4939 = stablehlo.reduce(%v4938 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4940 = stablehlo.broadcast_in_dim %v4939, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4941 = stablehlo.divide %v4940, %v4933 : tensor<256x256x56x56xf32>
    %v4942 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4943 = stablehlo.add %v4941, %v4942 : tensor<256x256x56x56xf32>
    %v4944 = stablehlo.rsqrt %v4943 : tensor<256x256x56x56xf32>
    %v4945 = stablehlo.multiply %v4937, %v4944 : tensor<256x256x56x56xf32>
    %v4946 = stablehlo.reshape %v4749 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4947 = stablehlo.multiply %v4946, %v4945 : tensor<256x256x56x56xf32>
    %v4948 = stablehlo.reduce(%v4947 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4949 = stablehlo.reshape %v4749 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4951 = stablehlo.reduce(%v4949 init: %v4950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4952 = stablehlo.reshape %v4870 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4953 = stablehlo.reshape %v144 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4954 = stablehlo.constant dense<0.0> : tensor<256x256x56x56xf32>
    %v4955 = stablehlo.compare GT, %v4953, %v4954 : (tensor<256x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xi1>
    %v4956 = stablehlo.select %v4955, %v4952, %v4954 : tensor<256x256x56x56xi1>, tensor<256x256x56x56xf32>
    %v4957 = stablehlo.reshape %v4956 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4958 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4960 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4961 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4962 = stablehlo.reduce(%v4958 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4963 = stablehlo.broadcast_in_dim %v4962, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4964 = stablehlo.divide %v4963, %v4960 : tensor<256x256x56x56xf32>
    %v4965 = stablehlo.subtract %v4958, %v4964 : tensor<256x256x56x56xf32>
    %v4966 = stablehlo.multiply %v4965, %v4965 : tensor<256x256x56x56xf32>
    %v4967 = stablehlo.reduce(%v4966 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4968 = stablehlo.broadcast_in_dim %v4967, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4969 = stablehlo.divide %v4968, %v4960 : tensor<256x256x56x56xf32>
    %v4970 = stablehlo.add %v4969, %v4961 : tensor<256x256x56x56xf32>
    %v4971 = stablehlo.rsqrt %v4970 : tensor<256x256x56x56xf32>
    %v4972 = stablehlo.multiply %v4965, %v4971 : tensor<256x256x56x56xf32>
    %v4973 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4974 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4975 = stablehlo.multiply %v4973, %v4974 : tensor<256x256x56x56xf32>
    %v4976 = stablehlo.reduce(%v4975 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4977 = stablehlo.broadcast_in_dim %v4976, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4978 = stablehlo.multiply %v4972, %v4975 : tensor<256x256x56x56xf32>
    %v4979 = stablehlo.reduce(%v4978 init: %v4959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4980 = stablehlo.broadcast_in_dim %v4979, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4981 = stablehlo.multiply %v4975, %v4960 : tensor<256x256x56x56xf32>
    %v4982 = stablehlo.subtract %v4981, %v4977 : tensor<256x256x56x56xf32>
    %v4983 = stablehlo.multiply %v4972, %v4980 : tensor<256x256x56x56xf32>
    %v4984 = stablehlo.subtract %v4982, %v4983 : tensor<256x256x56x56xf32>
    %v4985 = stablehlo.divide %v4971, %v4960 : tensor<256x256x56x56xf32>
    %v4986 = stablehlo.multiply %v4985, %v4984 : tensor<256x256x56x56xf32>
    %v4987 = stablehlo.reshape %v4986 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4988 = stablehlo.reshape %v4987 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4989 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4990 = stablehlo.transpose %v4989, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4991 = stablehlo.convolution(%v4988, %v4990)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4992 = stablehlo.reshape %v4991 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4993 = stablehlo.reshape %v4992 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4994 = stablehlo.reshape %v86 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4995 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v4996 = stablehlo.compare GT, %v4994, %v4995 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v4997 = stablehlo.select %v4996, %v4993, %v4995 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v4998 = stablehlo.reshape %v4997 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4999 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5001 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5002 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5003 = stablehlo.reduce(%v4999 init: %v5000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5004 = stablehlo.broadcast_in_dim %v5003, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5005 = stablehlo.divide %v5004, %v5001 : tensor<256x64x56x56xf32>
    %v5006 = stablehlo.subtract %v4999, %v5005 : tensor<256x64x56x56xf32>
    %v5007 = stablehlo.multiply %v5006, %v5006 : tensor<256x64x56x56xf32>
    %v5008 = stablehlo.reduce(%v5007 init: %v5000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5009 = stablehlo.broadcast_in_dim %v5008, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5010 = stablehlo.divide %v5009, %v5001 : tensor<256x64x56x56xf32>
    %v5011 = stablehlo.add %v5010, %v5002 : tensor<256x64x56x56xf32>
    %v5012 = stablehlo.rsqrt %v5011 : tensor<256x64x56x56xf32>
    %v5013 = stablehlo.multiply %v5006, %v5012 : tensor<256x64x56x56xf32>
    %v5014 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5015 = stablehlo.reshape %v4998 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5016 = stablehlo.multiply %v5014, %v5015 : tensor<256x64x56x56xf32>
    %v5017 = stablehlo.reduce(%v5016 init: %v5000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5018 = stablehlo.broadcast_in_dim %v5017, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5019 = stablehlo.multiply %v5013, %v5016 : tensor<256x64x56x56xf32>
    %v5020 = stablehlo.reduce(%v5019 init: %v5000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5021 = stablehlo.broadcast_in_dim %v5020, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5022 = stablehlo.multiply %v5016, %v5001 : tensor<256x64x56x56xf32>
    %v5023 = stablehlo.subtract %v5022, %v5018 : tensor<256x64x56x56xf32>
    %v5024 = stablehlo.multiply %v5013, %v5021 : tensor<256x64x56x56xf32>
    %v5025 = stablehlo.subtract %v5023, %v5024 : tensor<256x64x56x56xf32>
    %v5026 = stablehlo.divide %v5012, %v5001 : tensor<256x64x56x56xf32>
    %v5027 = stablehlo.multiply %v5026, %v5025 : tensor<256x64x56x56xf32>
    %v5028 = stablehlo.reshape %v5027 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5029 = stablehlo.reshape %v5028 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5030 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5031 = stablehlo.transpose %v5030, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5032 = stablehlo.convolution(%v5029, %v5031)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v5033 = stablehlo.reshape %v5032 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5034 = stablehlo.reshape %v5033 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5035 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5036 = stablehlo.constant dense<0.0> : tensor<256x64x56x56xf32>
    %v5037 = stablehlo.compare GT, %v5035, %v5036 : (tensor<256x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xi1>
    %v5038 = stablehlo.select %v5037, %v5034, %v5036 : tensor<256x64x56x56xi1>, tensor<256x64x56x56xf32>
    %v5039 = stablehlo.reshape %v5038 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5040 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5042 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5043 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5044 = stablehlo.reduce(%v5040 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5045 = stablehlo.broadcast_in_dim %v5044, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5046 = stablehlo.divide %v5045, %v5042 : tensor<256x64x56x56xf32>
    %v5047 = stablehlo.subtract %v5040, %v5046 : tensor<256x64x56x56xf32>
    %v5048 = stablehlo.multiply %v5047, %v5047 : tensor<256x64x56x56xf32>
    %v5049 = stablehlo.reduce(%v5048 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5050 = stablehlo.broadcast_in_dim %v5049, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5051 = stablehlo.divide %v5050, %v5042 : tensor<256x64x56x56xf32>
    %v5052 = stablehlo.add %v5051, %v5043 : tensor<256x64x56x56xf32>
    %v5053 = stablehlo.rsqrt %v5052 : tensor<256x64x56x56xf32>
    %v5054 = stablehlo.multiply %v5047, %v5053 : tensor<256x64x56x56xf32>
    %v5055 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5056 = stablehlo.reshape %v5039 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5057 = stablehlo.multiply %v5055, %v5056 : tensor<256x64x56x56xf32>
    %v5058 = stablehlo.reduce(%v5057 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5059 = stablehlo.broadcast_in_dim %v5058, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5060 = stablehlo.multiply %v5054, %v5057 : tensor<256x64x56x56xf32>
    %v5061 = stablehlo.reduce(%v5060 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5062 = stablehlo.broadcast_in_dim %v5061, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5063 = stablehlo.multiply %v5057, %v5042 : tensor<256x64x56x56xf32>
    %v5064 = stablehlo.subtract %v5063, %v5059 : tensor<256x64x56x56xf32>
    %v5065 = stablehlo.multiply %v5054, %v5062 : tensor<256x64x56x56xf32>
    %v5066 = stablehlo.subtract %v5064, %v5065 : tensor<256x64x56x56xf32>
    %v5067 = stablehlo.divide %v5053, %v5042 : tensor<256x64x56x56xf32>
    %v5068 = stablehlo.multiply %v5067, %v5066 : tensor<256x64x56x56xf32>
    %v5069 = stablehlo.reshape %v5068 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5070 = stablehlo.reshape %v5069 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5071 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v5072 = stablehlo.transpose %v5071, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5073 = stablehlo.convolution(%v5070, %v5072)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v5074 = stablehlo.reshape %v5073 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5075 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5077 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5078 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5079 = stablehlo.reduce(%v5075 init: %v5076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5080 = stablehlo.broadcast_in_dim %v5079, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5081 = stablehlo.divide %v5080, %v5077 : tensor<256x256x56x56xf32>
    %v5082 = stablehlo.subtract %v5075, %v5081 : tensor<256x256x56x56xf32>
    %v5083 = stablehlo.multiply %v5082, %v5082 : tensor<256x256x56x56xf32>
    %v5084 = stablehlo.reduce(%v5083 init: %v5076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5085 = stablehlo.broadcast_in_dim %v5084, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5086 = stablehlo.divide %v5085, %v5077 : tensor<256x256x56x56xf32>
    %v5087 = stablehlo.add %v5086, %v5078 : tensor<256x256x56x56xf32>
    %v5088 = stablehlo.rsqrt %v5087 : tensor<256x256x56x56xf32>
    %v5089 = stablehlo.multiply %v5082, %v5088 : tensor<256x256x56x56xf32>
    %v5090 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5091 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5092 = stablehlo.multiply %v5090, %v5091 : tensor<256x256x56x56xf32>
    %v5093 = stablehlo.reduce(%v5092 init: %v5076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5094 = stablehlo.broadcast_in_dim %v5093, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5095 = stablehlo.multiply %v5089, %v5092 : tensor<256x256x56x56xf32>
    %v5096 = stablehlo.reduce(%v5095 init: %v5076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5097 = stablehlo.broadcast_in_dim %v5096, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5098 = stablehlo.multiply %v5092, %v5077 : tensor<256x256x56x56xf32>
    %v5099 = stablehlo.subtract %v5098, %v5094 : tensor<256x256x56x56xf32>
    %v5100 = stablehlo.multiply %v5089, %v5097 : tensor<256x256x56x56xf32>
    %v5101 = stablehlo.subtract %v5099, %v5100 : tensor<256x256x56x56xf32>
    %v5102 = stablehlo.divide %v5088, %v5077 : tensor<256x256x56x56xf32>
    %v5103 = stablehlo.multiply %v5102, %v5101 : tensor<256x256x56x56xf32>
    %v5104 = stablehlo.reshape %v5103 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v5105 = stablehlo.reshape %v5104 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5106 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5107 = stablehlo.transpose %v5106, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5108 = stablehlo.convolution(%v5105, %v5107)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v5109 = stablehlo.reshape %v5108 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5110 = stablehlo.reshape %v5074 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5111 = stablehlo.reshape %v5109 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5112 = stablehlo.add %v5110, %v5111 : tensor<256x64x56x56xf32>
    %v5113 = stablehlo.reshape %v5112 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v5114 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5115 = stablehlo.reshape %v5069 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5116 = stablehlo.transpose %v5114, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5117 = stablehlo.transpose %v5115, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5118 = stablehlo.convolution(%v5116, %v5117)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x1x1xf32>
    %v5119 = stablehlo.transpose %v5118, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5120 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5122 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5123 = stablehlo.reduce(%v5120 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5124 = stablehlo.broadcast_in_dim %v5123, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5125 = stablehlo.divide %v5124, %v5122 : tensor<256x64x56x56xf32>
    %v5126 = stablehlo.subtract %v5120, %v5125 : tensor<256x64x56x56xf32>
    %v5127 = stablehlo.multiply %v5126, %v5126 : tensor<256x64x56x56xf32>
    %v5128 = stablehlo.reduce(%v5127 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5129 = stablehlo.broadcast_in_dim %v5128, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5130 = stablehlo.divide %v5129, %v5122 : tensor<256x64x56x56xf32>
    %v5131 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5132 = stablehlo.add %v5130, %v5131 : tensor<256x64x56x56xf32>
    %v5133 = stablehlo.rsqrt %v5132 : tensor<256x64x56x56xf32>
    %v5134 = stablehlo.multiply %v5126, %v5133 : tensor<256x64x56x56xf32>
    %v5135 = stablehlo.reshape %v5039 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5136 = stablehlo.multiply %v5135, %v5134 : tensor<256x64x56x56xf32>
    %v5137 = stablehlo.reduce(%v5136 init: %v5121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5138 = stablehlo.reshape %v5039 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5140 = stablehlo.reduce(%v5138 init: %v5139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5141 = stablehlo.reshape %v61 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5142 = stablehlo.reshape %v5028 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5143 = stablehlo.transpose %v5141, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5144 = stablehlo.transpose %v5142, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5145 = stablehlo.convolution(%v5143, %v5144)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v5146 = stablehlo.transpose %v5145, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5147 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5149 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5150 = stablehlo.reduce(%v5147 init: %v5148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5151 = stablehlo.broadcast_in_dim %v5150, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5152 = stablehlo.divide %v5151, %v5149 : tensor<256x64x56x56xf32>
    %v5153 = stablehlo.subtract %v5147, %v5152 : tensor<256x64x56x56xf32>
    %v5154 = stablehlo.multiply %v5153, %v5153 : tensor<256x64x56x56xf32>
    %v5155 = stablehlo.reduce(%v5154 init: %v5148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5156 = stablehlo.broadcast_in_dim %v5155, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5157 = stablehlo.divide %v5156, %v5149 : tensor<256x64x56x56xf32>
    %v5158 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v5159 = stablehlo.add %v5157, %v5158 : tensor<256x64x56x56xf32>
    %v5160 = stablehlo.rsqrt %v5159 : tensor<256x64x56x56xf32>
    %v5161 = stablehlo.multiply %v5153, %v5160 : tensor<256x64x56x56xf32>
    %v5162 = stablehlo.reshape %v4998 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5163 = stablehlo.multiply %v5162, %v5161 : tensor<256x64x56x56xf32>
    %v5164 = stablehlo.reduce(%v5163 init: %v5148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5165 = stablehlo.reshape %v4998 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5166 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5167 = stablehlo.reduce(%v5165 init: %v5166) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5168 = stablehlo.reshape %v90 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5169 = stablehlo.reshape %v4987 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5170 = stablehlo.transpose %v5168, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5171 = stablehlo.transpose %v5169, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v5172 = stablehlo.convolution(%v5170, %v5171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v5173 = stablehlo.transpose %v5172, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5174 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5175 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5176 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5177 = stablehlo.reduce(%v5174 init: %v5175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5178 = stablehlo.broadcast_in_dim %v5177, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5179 = stablehlo.divide %v5178, %v5176 : tensor<256x256x56x56xf32>
    %v5180 = stablehlo.subtract %v5174, %v5179 : tensor<256x256x56x56xf32>
    %v5181 = stablehlo.multiply %v5180, %v5180 : tensor<256x256x56x56xf32>
    %v5182 = stablehlo.reduce(%v5181 init: %v5175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5183 = stablehlo.broadcast_in_dim %v5182, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5184 = stablehlo.divide %v5183, %v5176 : tensor<256x256x56x56xf32>
    %v5185 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5186 = stablehlo.add %v5184, %v5185 : tensor<256x256x56x56xf32>
    %v5187 = stablehlo.rsqrt %v5186 : tensor<256x256x56x56xf32>
    %v5188 = stablehlo.multiply %v5180, %v5187 : tensor<256x256x56x56xf32>
    %v5189 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5190 = stablehlo.multiply %v5189, %v5188 : tensor<256x256x56x56xf32>
    %v5191 = stablehlo.reduce(%v5190 init: %v5175) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5192 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5194 = stablehlo.reduce(%v5192 init: %v5193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5195 = stablehlo.reshape %v32 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5196 = stablehlo.reshape %v5104 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5197 = stablehlo.transpose %v5195, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v5198 = stablehlo.transpose %v5196, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v5199 = stablehlo.convolution(%v5197, %v5198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v5200 = stablehlo.transpose %v5199, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5201 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5203 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5204 = stablehlo.reduce(%v5201 init: %v5202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5205 = stablehlo.broadcast_in_dim %v5204, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5206 = stablehlo.divide %v5205, %v5203 : tensor<256x256x56x56xf32>
    %v5207 = stablehlo.subtract %v5201, %v5206 : tensor<256x256x56x56xf32>
    %v5208 = stablehlo.multiply %v5207, %v5207 : tensor<256x256x56x56xf32>
    %v5209 = stablehlo.reduce(%v5208 init: %v5202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5210 = stablehlo.broadcast_in_dim %v5209, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5211 = stablehlo.divide %v5210, %v5203 : tensor<256x256x56x56xf32>
    %v5212 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v5213 = stablehlo.add %v5211, %v5212 : tensor<256x256x56x56xf32>
    %v5214 = stablehlo.rsqrt %v5213 : tensor<256x256x56x56xf32>
    %v5215 = stablehlo.multiply %v5207, %v5214 : tensor<256x256x56x56xf32>
    %v5216 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5217 = stablehlo.multiply %v5216, %v5215 : tensor<256x256x56x56xf32>
    %v5218 = stablehlo.reduce(%v5217 init: %v5202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5219 = stablehlo.reshape %v4957 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5221 = stablehlo.reduce(%v5219 init: %v5220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5222 = stablehlo.reshape %v28 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5223 = stablehlo.reshape %v5113 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5225 = "stablehlo.select_and_scatter"(%v5222, %v5223, %v5224) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x112x112xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5227 = stablehlo.reshape %v5226 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5228 = stablehlo.reshape %v24 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5229 = stablehlo.constant dense<0.0> : tensor<256x64x112x112xf32>
    %v5230 = stablehlo.compare GT, %v5228, %v5229 : (tensor<256x64x112x112xf32>, tensor<256x64x112x112xf32>) -> tensor<256x64x112x112xi1>
    %v5231 = stablehlo.select %v5230, %v5227, %v5229 : tensor<256x64x112x112xi1>, tensor<256x64x112x112xf32>
    %v5232 = stablehlo.reshape %v5231 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5233 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5235 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5236 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v5237 = stablehlo.reduce(%v5233 init: %v5234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5238 = stablehlo.broadcast_in_dim %v5237, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5239 = stablehlo.divide %v5238, %v5235 : tensor<256x64x112x112xf32>
    %v5240 = stablehlo.subtract %v5233, %v5239 : tensor<256x64x112x112xf32>
    %v5241 = stablehlo.multiply %v5240, %v5240 : tensor<256x64x112x112xf32>
    %v5242 = stablehlo.reduce(%v5241 init: %v5234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5243 = stablehlo.broadcast_in_dim %v5242, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5244 = stablehlo.divide %v5243, %v5235 : tensor<256x64x112x112xf32>
    %v5245 = stablehlo.add %v5244, %v5236 : tensor<256x64x112x112xf32>
    %v5246 = stablehlo.rsqrt %v5245 : tensor<256x64x112x112xf32>
    %v5247 = stablehlo.multiply %v5240, %v5246 : tensor<256x64x112x112xf32>
    %v5248 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5249 = stablehlo.reshape %v5232 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5250 = stablehlo.multiply %v5248, %v5249 : tensor<256x64x112x112xf32>
    %v5251 = stablehlo.reduce(%v5250 init: %v5234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5252 = stablehlo.broadcast_in_dim %v5251, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5253 = stablehlo.multiply %v5247, %v5250 : tensor<256x64x112x112xf32>
    %v5254 = stablehlo.reduce(%v5253 init: %v5234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5255 = stablehlo.broadcast_in_dim %v5254, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5256 = stablehlo.multiply %v5250, %v5235 : tensor<256x64x112x112xf32>
    %v5257 = stablehlo.subtract %v5256, %v5252 : tensor<256x64x112x112xf32>
    %v5258 = stablehlo.multiply %v5247, %v5255 : tensor<256x64x112x112xf32>
    %v5259 = stablehlo.subtract %v5257, %v5258 : tensor<256x64x112x112xf32>
    %v5260 = stablehlo.divide %v5246, %v5235 : tensor<256x64x112x112xf32>
    %v5261 = stablehlo.multiply %v5260, %v5259 : tensor<256x64x112x112xf32>
    %v5262 = stablehlo.reshape %v5261 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v5263 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v5264 = stablehlo.reshape %v5262 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5266 = stablehlo.pad %v5264, %v5265, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x224x224xf32>
    %v5267 = stablehlo.transpose %v5263, dims = [1, 0, 2, 3] : (tensor<256x3x224x224xf32>) -> tensor<3x256x224x224xf32>
    %v5268 = stablehlo.transpose %v5266, dims = [1, 0, 2, 3] : (tensor<256x64x224x224xf32>) -> tensor<64x256x224x224xf32>
    %v5269 = stablehlo.convolution(%v5267, %v5268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x224x224xf32>, tensor<64x256x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v5270 = stablehlo.transpose %v5269, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v5271 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5273 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5274 = stablehlo.reduce(%v5271 init: %v5272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5275 = stablehlo.broadcast_in_dim %v5274, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5276 = stablehlo.divide %v5275, %v5273 : tensor<256x64x112x112xf32>
    %v5277 = stablehlo.subtract %v5271, %v5276 : tensor<256x64x112x112xf32>
    %v5278 = stablehlo.multiply %v5277, %v5277 : tensor<256x64x112x112xf32>
    %v5279 = stablehlo.reduce(%v5278 init: %v5272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5280 = stablehlo.broadcast_in_dim %v5279, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5281 = stablehlo.divide %v5280, %v5273 : tensor<256x64x112x112xf32>
    %v5282 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v5283 = stablehlo.add %v5281, %v5282 : tensor<256x64x112x112xf32>
    %v5284 = stablehlo.rsqrt %v5283 : tensor<256x64x112x112xf32>
    %v5285 = stablehlo.multiply %v5277, %v5284 : tensor<256x64x112x112xf32>
    %v5286 = stablehlo.reshape %v5232 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5287 = stablehlo.multiply %v5286, %v5285 : tensor<256x64x112x112xf32>
    %v5288 = stablehlo.reduce(%v5287 init: %v5272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5289 = stablehlo.reshape %v5232 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5290 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5291 = stablehlo.reduce(%v5289 init: %v5290) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5292 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5294 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v5295 = stablehlo.reduce(%v5292 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5296 = stablehlo.divide %v5295, %v5294 : tensor<64xf32>
    %v5297 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v5298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5299 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v5300 = stablehlo.reduce(%v5297 init: %v5298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5301 = stablehlo.broadcast_in_dim %v5300, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v5302 = stablehlo.divide %v5301, %v5299 : tensor<256x64x112x112xf32>
    %v5303 = stablehlo.subtract %v5297, %v5302 : tensor<256x64x112x112xf32>
    %v5304 = stablehlo.multiply %v5303, %v5303 : tensor<256x64x112x112xf32>
    %v5305 = stablehlo.reduce(%v5304 init: %v5298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5306 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v5307 = stablehlo.divide %v5305, %v5306 : tensor<64xf32>
    %v5308 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5310 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5311 = stablehlo.reduce(%v5308 init: %v5309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5312 = stablehlo.divide %v5311, %v5310 : tensor<64xf32>
    %v5313 = stablehlo.reshape %v37 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5315 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5316 = stablehlo.reduce(%v5313 init: %v5314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5317 = stablehlo.broadcast_in_dim %v5316, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5318 = stablehlo.divide %v5317, %v5315 : tensor<256x64x56x56xf32>
    %v5319 = stablehlo.subtract %v5313, %v5318 : tensor<256x64x56x56xf32>
    %v5320 = stablehlo.multiply %v5319, %v5319 : tensor<256x64x56x56xf32>
    %v5321 = stablehlo.reduce(%v5320 init: %v5314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5322 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5323 = stablehlo.divide %v5321, %v5322 : tensor<64xf32>
    %v5324 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5326 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5327 = stablehlo.reduce(%v5324 init: %v5325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5328 = stablehlo.divide %v5327, %v5326 : tensor<64xf32>
    %v5329 = stablehlo.reshape %v66 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5331 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5332 = stablehlo.reduce(%v5329 init: %v5330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5333 = stablehlo.broadcast_in_dim %v5332, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5334 = stablehlo.divide %v5333, %v5331 : tensor<256x64x56x56xf32>
    %v5335 = stablehlo.subtract %v5329, %v5334 : tensor<256x64x56x56xf32>
    %v5336 = stablehlo.multiply %v5335, %v5335 : tensor<256x64x56x56xf32>
    %v5337 = stablehlo.reduce(%v5336 init: %v5330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5338 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5339 = stablehlo.divide %v5337, %v5338 : tensor<64xf32>
    %v5340 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5342 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5343 = stablehlo.reduce(%v5340 init: %v5341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5344 = stablehlo.divide %v5343, %v5342 : tensor<256xf32>
    %v5345 = stablehlo.reshape %v95 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5346 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5347 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5348 = stablehlo.reduce(%v5345 init: %v5346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5349 = stablehlo.broadcast_in_dim %v5348, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5350 = stablehlo.divide %v5349, %v5347 : tensor<256x256x56x56xf32>
    %v5351 = stablehlo.subtract %v5345, %v5350 : tensor<256x256x56x56xf32>
    %v5352 = stablehlo.multiply %v5351, %v5351 : tensor<256x256x56x56xf32>
    %v5353 = stablehlo.reduce(%v5352 init: %v5346) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5354 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5355 = stablehlo.divide %v5353, %v5354 : tensor<256xf32>
    %v5356 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5358 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5359 = stablehlo.reduce(%v5356 init: %v5357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5360 = stablehlo.divide %v5359, %v5358 : tensor<256xf32>
    %v5361 = stablehlo.reshape %v120 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5363 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5364 = stablehlo.reduce(%v5361 init: %v5362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5365 = stablehlo.broadcast_in_dim %v5364, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5366 = stablehlo.divide %v5365, %v5363 : tensor<256x256x56x56xf32>
    %v5367 = stablehlo.subtract %v5361, %v5366 : tensor<256x256x56x56xf32>
    %v5368 = stablehlo.multiply %v5367, %v5367 : tensor<256x256x56x56xf32>
    %v5369 = stablehlo.reduce(%v5368 init: %v5362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5370 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5371 = stablehlo.divide %v5369, %v5370 : tensor<256xf32>
    %v5372 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5374 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5375 = stablehlo.reduce(%v5372 init: %v5373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5376 = stablehlo.divide %v5375, %v5374 : tensor<64xf32>
    %v5377 = stablehlo.reshape %v153 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5379 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5380 = stablehlo.reduce(%v5377 init: %v5378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5381 = stablehlo.broadcast_in_dim %v5380, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5382 = stablehlo.divide %v5381, %v5379 : tensor<256x64x56x56xf32>
    %v5383 = stablehlo.subtract %v5377, %v5382 : tensor<256x64x56x56xf32>
    %v5384 = stablehlo.multiply %v5383, %v5383 : tensor<256x64x56x56xf32>
    %v5385 = stablehlo.reduce(%v5384 init: %v5378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5386 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5387 = stablehlo.divide %v5385, %v5386 : tensor<64xf32>
    %v5388 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5390 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5391 = stablehlo.reduce(%v5388 init: %v5389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5392 = stablehlo.divide %v5391, %v5390 : tensor<64xf32>
    %v5393 = stablehlo.reshape %v182 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5395 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5396 = stablehlo.reduce(%v5393 init: %v5394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5397 = stablehlo.broadcast_in_dim %v5396, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5398 = stablehlo.divide %v5397, %v5395 : tensor<256x64x56x56xf32>
    %v5399 = stablehlo.subtract %v5393, %v5398 : tensor<256x64x56x56xf32>
    %v5400 = stablehlo.multiply %v5399, %v5399 : tensor<256x64x56x56xf32>
    %v5401 = stablehlo.reduce(%v5400 init: %v5394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5402 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5403 = stablehlo.divide %v5401, %v5402 : tensor<64xf32>
    %v5404 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5406 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5407 = stablehlo.reduce(%v5404 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5408 = stablehlo.divide %v5407, %v5406 : tensor<256xf32>
    %v5409 = stablehlo.reshape %v211 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5411 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5412 = stablehlo.reduce(%v5409 init: %v5410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5413 = stablehlo.broadcast_in_dim %v5412, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5414 = stablehlo.divide %v5413, %v5411 : tensor<256x256x56x56xf32>
    %v5415 = stablehlo.subtract %v5409, %v5414 : tensor<256x256x56x56xf32>
    %v5416 = stablehlo.multiply %v5415, %v5415 : tensor<256x256x56x56xf32>
    %v5417 = stablehlo.reduce(%v5416 init: %v5410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5418 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5419 = stablehlo.divide %v5417, %v5418 : tensor<256xf32>
    %v5420 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5422 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5423 = stablehlo.reduce(%v5420 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5424 = stablehlo.divide %v5423, %v5422 : tensor<64xf32>
    %v5425 = stablehlo.reshape %v244 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5427 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5428 = stablehlo.reduce(%v5425 init: %v5426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5429 = stablehlo.broadcast_in_dim %v5428, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5430 = stablehlo.divide %v5429, %v5427 : tensor<256x64x56x56xf32>
    %v5431 = stablehlo.subtract %v5425, %v5430 : tensor<256x64x56x56xf32>
    %v5432 = stablehlo.multiply %v5431, %v5431 : tensor<256x64x56x56xf32>
    %v5433 = stablehlo.reduce(%v5432 init: %v5426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5434 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5435 = stablehlo.divide %v5433, %v5434 : tensor<64xf32>
    %v5436 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5438 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5439 = stablehlo.reduce(%v5436 init: %v5437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5440 = stablehlo.divide %v5439, %v5438 : tensor<64xf32>
    %v5441 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5443 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5444 = stablehlo.reduce(%v5441 init: %v5442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5445 = stablehlo.broadcast_in_dim %v5444, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5446 = stablehlo.divide %v5445, %v5443 : tensor<256x64x56x56xf32>
    %v5447 = stablehlo.subtract %v5441, %v5446 : tensor<256x64x56x56xf32>
    %v5448 = stablehlo.multiply %v5447, %v5447 : tensor<256x64x56x56xf32>
    %v5449 = stablehlo.reduce(%v5448 init: %v5442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5450 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5451 = stablehlo.divide %v5449, %v5450 : tensor<64xf32>
    %v5452 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5454 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5455 = stablehlo.reduce(%v5452 init: %v5453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5456 = stablehlo.divide %v5455, %v5454 : tensor<256xf32>
    %v5457 = stablehlo.reshape %v302 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5459 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5460 = stablehlo.reduce(%v5457 init: %v5458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5461 = stablehlo.broadcast_in_dim %v5460, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5462 = stablehlo.divide %v5461, %v5459 : tensor<256x256x56x56xf32>
    %v5463 = stablehlo.subtract %v5457, %v5462 : tensor<256x256x56x56xf32>
    %v5464 = stablehlo.multiply %v5463, %v5463 : tensor<256x256x56x56xf32>
    %v5465 = stablehlo.reduce(%v5464 init: %v5458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5466 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5467 = stablehlo.divide %v5465, %v5466 : tensor<256xf32>
    %v5468 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5470 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5471 = stablehlo.reduce(%v5468 init: %v5469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5472 = stablehlo.divide %v5471, %v5470 : tensor<128xf32>
    %v5473 = stablehlo.reshape %v335 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5475 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v5476 = stablehlo.reduce(%v5473 init: %v5474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5477 = stablehlo.broadcast_in_dim %v5476, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v5478 = stablehlo.divide %v5477, %v5475 : tensor<256x128x56x56xf32>
    %v5479 = stablehlo.subtract %v5473, %v5478 : tensor<256x128x56x56xf32>
    %v5480 = stablehlo.multiply %v5479, %v5479 : tensor<256x128x56x56xf32>
    %v5481 = stablehlo.reduce(%v5480 init: %v5474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5482 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5483 = stablehlo.divide %v5481, %v5482 : tensor<128xf32>
    %v5484 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5486 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5487 = stablehlo.reduce(%v5484 init: %v5485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5488 = stablehlo.divide %v5487, %v5486 : tensor<128xf32>
    %v5489 = stablehlo.reshape %v364 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5491 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5492 = stablehlo.reduce(%v5489 init: %v5490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5493 = stablehlo.broadcast_in_dim %v5492, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5494 = stablehlo.divide %v5493, %v5491 : tensor<256x128x28x28xf32>
    %v5495 = stablehlo.subtract %v5489, %v5494 : tensor<256x128x28x28xf32>
    %v5496 = stablehlo.multiply %v5495, %v5495 : tensor<256x128x28x28xf32>
    %v5497 = stablehlo.reduce(%v5496 init: %v5490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5498 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5499 = stablehlo.divide %v5497, %v5498 : tensor<128xf32>
    %v5500 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5502 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5503 = stablehlo.reduce(%v5500 init: %v5501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5504 = stablehlo.divide %v5503, %v5502 : tensor<512xf32>
    %v5505 = stablehlo.reshape %v393 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5507 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5508 = stablehlo.reduce(%v5505 init: %v5506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5509 = stablehlo.broadcast_in_dim %v5508, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5510 = stablehlo.divide %v5509, %v5507 : tensor<256x512x28x28xf32>
    %v5511 = stablehlo.subtract %v5505, %v5510 : tensor<256x512x28x28xf32>
    %v5512 = stablehlo.multiply %v5511, %v5511 : tensor<256x512x28x28xf32>
    %v5513 = stablehlo.reduce(%v5512 init: %v5506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5514 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5515 = stablehlo.divide %v5513, %v5514 : tensor<512xf32>
    %v5516 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5518 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5519 = stablehlo.reduce(%v5516 init: %v5517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5520 = stablehlo.divide %v5519, %v5518 : tensor<512xf32>
    %v5521 = stablehlo.reshape %v418 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5523 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5524 = stablehlo.reduce(%v5521 init: %v5522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5525 = stablehlo.broadcast_in_dim %v5524, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5526 = stablehlo.divide %v5525, %v5523 : tensor<256x512x28x28xf32>
    %v5527 = stablehlo.subtract %v5521, %v5526 : tensor<256x512x28x28xf32>
    %v5528 = stablehlo.multiply %v5527, %v5527 : tensor<256x512x28x28xf32>
    %v5529 = stablehlo.reduce(%v5528 init: %v5522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5530 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5531 = stablehlo.divide %v5529, %v5530 : tensor<512xf32>
    %v5532 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5534 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5535 = stablehlo.reduce(%v5532 init: %v5533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5536 = stablehlo.divide %v5535, %v5534 : tensor<128xf32>
    %v5537 = stablehlo.reshape %v451 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5539 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5540 = stablehlo.reduce(%v5537 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5541 = stablehlo.broadcast_in_dim %v5540, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5542 = stablehlo.divide %v5541, %v5539 : tensor<256x128x28x28xf32>
    %v5543 = stablehlo.subtract %v5537, %v5542 : tensor<256x128x28x28xf32>
    %v5544 = stablehlo.multiply %v5543, %v5543 : tensor<256x128x28x28xf32>
    %v5545 = stablehlo.reduce(%v5544 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5546 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5547 = stablehlo.divide %v5545, %v5546 : tensor<128xf32>
    %v5548 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5550 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5551 = stablehlo.reduce(%v5548 init: %v5549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5552 = stablehlo.divide %v5551, %v5550 : tensor<128xf32>
    %v5553 = stablehlo.reshape %v480 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5555 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5556 = stablehlo.reduce(%v5553 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5557 = stablehlo.broadcast_in_dim %v5556, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5558 = stablehlo.divide %v5557, %v5555 : tensor<256x128x28x28xf32>
    %v5559 = stablehlo.subtract %v5553, %v5558 : tensor<256x128x28x28xf32>
    %v5560 = stablehlo.multiply %v5559, %v5559 : tensor<256x128x28x28xf32>
    %v5561 = stablehlo.reduce(%v5560 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5562 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5563 = stablehlo.divide %v5561, %v5562 : tensor<128xf32>
    %v5564 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5566 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5567 = stablehlo.reduce(%v5564 init: %v5565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5568 = stablehlo.divide %v5567, %v5566 : tensor<512xf32>
    %v5569 = stablehlo.reshape %v509 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5571 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5572 = stablehlo.reduce(%v5569 init: %v5570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5573 = stablehlo.broadcast_in_dim %v5572, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5574 = stablehlo.divide %v5573, %v5571 : tensor<256x512x28x28xf32>
    %v5575 = stablehlo.subtract %v5569, %v5574 : tensor<256x512x28x28xf32>
    %v5576 = stablehlo.multiply %v5575, %v5575 : tensor<256x512x28x28xf32>
    %v5577 = stablehlo.reduce(%v5576 init: %v5570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5578 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5579 = stablehlo.divide %v5577, %v5578 : tensor<512xf32>
    %v5580 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5582 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5583 = stablehlo.reduce(%v5580 init: %v5581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5584 = stablehlo.divide %v5583, %v5582 : tensor<128xf32>
    %v5585 = stablehlo.reshape %v542 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5587 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5588 = stablehlo.reduce(%v5585 init: %v5586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5589 = stablehlo.broadcast_in_dim %v5588, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5590 = stablehlo.divide %v5589, %v5587 : tensor<256x128x28x28xf32>
    %v5591 = stablehlo.subtract %v5585, %v5590 : tensor<256x128x28x28xf32>
    %v5592 = stablehlo.multiply %v5591, %v5591 : tensor<256x128x28x28xf32>
    %v5593 = stablehlo.reduce(%v5592 init: %v5586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5594 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5595 = stablehlo.divide %v5593, %v5594 : tensor<128xf32>
    %v5596 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5598 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5599 = stablehlo.reduce(%v5596 init: %v5597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5600 = stablehlo.divide %v5599, %v5598 : tensor<128xf32>
    %v5601 = stablehlo.reshape %v571 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5603 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5604 = stablehlo.reduce(%v5601 init: %v5602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5605 = stablehlo.broadcast_in_dim %v5604, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5606 = stablehlo.divide %v5605, %v5603 : tensor<256x128x28x28xf32>
    %v5607 = stablehlo.subtract %v5601, %v5606 : tensor<256x128x28x28xf32>
    %v5608 = stablehlo.multiply %v5607, %v5607 : tensor<256x128x28x28xf32>
    %v5609 = stablehlo.reduce(%v5608 init: %v5602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5611 = stablehlo.divide %v5609, %v5610 : tensor<128xf32>
    %v5612 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5614 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5615 = stablehlo.reduce(%v5612 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5616 = stablehlo.divide %v5615, %v5614 : tensor<512xf32>
    %v5617 = stablehlo.reshape %v600 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5619 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5620 = stablehlo.reduce(%v5617 init: %v5618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5621 = stablehlo.broadcast_in_dim %v5620, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5622 = stablehlo.divide %v5621, %v5619 : tensor<256x512x28x28xf32>
    %v5623 = stablehlo.subtract %v5617, %v5622 : tensor<256x512x28x28xf32>
    %v5624 = stablehlo.multiply %v5623, %v5623 : tensor<256x512x28x28xf32>
    %v5625 = stablehlo.reduce(%v5624 init: %v5618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5626 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5627 = stablehlo.divide %v5625, %v5626 : tensor<512xf32>
    %v5628 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5630 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5631 = stablehlo.reduce(%v5628 init: %v5629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5632 = stablehlo.divide %v5631, %v5630 : tensor<128xf32>
    %v5633 = stablehlo.reshape %v633 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5635 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5636 = stablehlo.reduce(%v5633 init: %v5634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5637 = stablehlo.broadcast_in_dim %v5636, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5638 = stablehlo.divide %v5637, %v5635 : tensor<256x128x28x28xf32>
    %v5639 = stablehlo.subtract %v5633, %v5638 : tensor<256x128x28x28xf32>
    %v5640 = stablehlo.multiply %v5639, %v5639 : tensor<256x128x28x28xf32>
    %v5641 = stablehlo.reduce(%v5640 init: %v5634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5642 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5643 = stablehlo.divide %v5641, %v5642 : tensor<128xf32>
    %v5644 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5646 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5647 = stablehlo.reduce(%v5644 init: %v5645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5648 = stablehlo.divide %v5647, %v5646 : tensor<128xf32>
    %v5649 = stablehlo.reshape %v662 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5651 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5652 = stablehlo.reduce(%v5649 init: %v5650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5653 = stablehlo.broadcast_in_dim %v5652, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5654 = stablehlo.divide %v5653, %v5651 : tensor<256x128x28x28xf32>
    %v5655 = stablehlo.subtract %v5649, %v5654 : tensor<256x128x28x28xf32>
    %v5656 = stablehlo.multiply %v5655, %v5655 : tensor<256x128x28x28xf32>
    %v5657 = stablehlo.reduce(%v5656 init: %v5650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5658 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5659 = stablehlo.divide %v5657, %v5658 : tensor<128xf32>
    %v5660 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5662 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5663 = stablehlo.reduce(%v5660 init: %v5661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5664 = stablehlo.divide %v5663, %v5662 : tensor<512xf32>
    %v5665 = stablehlo.reshape %v691 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5667 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5668 = stablehlo.reduce(%v5665 init: %v5666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5669 = stablehlo.broadcast_in_dim %v5668, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5670 = stablehlo.divide %v5669, %v5667 : tensor<256x512x28x28xf32>
    %v5671 = stablehlo.subtract %v5665, %v5670 : tensor<256x512x28x28xf32>
    %v5672 = stablehlo.multiply %v5671, %v5671 : tensor<256x512x28x28xf32>
    %v5673 = stablehlo.reduce(%v5672 init: %v5666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5674 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5675 = stablehlo.divide %v5673, %v5674 : tensor<512xf32>
    %v5676 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5678 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5679 = stablehlo.reduce(%v5676 init: %v5677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5680 = stablehlo.divide %v5679, %v5678 : tensor<256xf32>
    %v5681 = stablehlo.reshape %v724 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5683 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v5684 = stablehlo.reduce(%v5681 init: %v5682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5685 = stablehlo.broadcast_in_dim %v5684, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v5686 = stablehlo.divide %v5685, %v5683 : tensor<256x256x28x28xf32>
    %v5687 = stablehlo.subtract %v5681, %v5686 : tensor<256x256x28x28xf32>
    %v5688 = stablehlo.multiply %v5687, %v5687 : tensor<256x256x28x28xf32>
    %v5689 = stablehlo.reduce(%v5688 init: %v5682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5690 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5691 = stablehlo.divide %v5689, %v5690 : tensor<256xf32>
    %v5692 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5694 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5695 = stablehlo.reduce(%v5692 init: %v5693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5696 = stablehlo.divide %v5695, %v5694 : tensor<256xf32>
    %v5697 = stablehlo.reshape %v753 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5699 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5700 = stablehlo.reduce(%v5697 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5701 = stablehlo.broadcast_in_dim %v5700, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5702 = stablehlo.divide %v5701, %v5699 : tensor<256x256x14x14xf32>
    %v5703 = stablehlo.subtract %v5697, %v5702 : tensor<256x256x14x14xf32>
    %v5704 = stablehlo.multiply %v5703, %v5703 : tensor<256x256x14x14xf32>
    %v5705 = stablehlo.reduce(%v5704 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5706 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5707 = stablehlo.divide %v5705, %v5706 : tensor<256xf32>
    %v5708 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5710 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5711 = stablehlo.reduce(%v5708 init: %v5709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5712 = stablehlo.divide %v5711, %v5710 : tensor<1024xf32>
    %v5713 = stablehlo.reshape %v782 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5715 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5716 = stablehlo.reduce(%v5713 init: %v5714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5717 = stablehlo.broadcast_in_dim %v5716, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5718 = stablehlo.divide %v5717, %v5715 : tensor<256x1024x14x14xf32>
    %v5719 = stablehlo.subtract %v5713, %v5718 : tensor<256x1024x14x14xf32>
    %v5720 = stablehlo.multiply %v5719, %v5719 : tensor<256x1024x14x14xf32>
    %v5721 = stablehlo.reduce(%v5720 init: %v5714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5722 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5723 = stablehlo.divide %v5721, %v5722 : tensor<1024xf32>
    %v5724 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5726 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5727 = stablehlo.reduce(%v5724 init: %v5725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5728 = stablehlo.divide %v5727, %v5726 : tensor<1024xf32>
    %v5729 = stablehlo.reshape %v807 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5731 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5732 = stablehlo.reduce(%v5729 init: %v5730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5733 = stablehlo.broadcast_in_dim %v5732, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5734 = stablehlo.divide %v5733, %v5731 : tensor<256x1024x14x14xf32>
    %v5735 = stablehlo.subtract %v5729, %v5734 : tensor<256x1024x14x14xf32>
    %v5736 = stablehlo.multiply %v5735, %v5735 : tensor<256x1024x14x14xf32>
    %v5737 = stablehlo.reduce(%v5736 init: %v5730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5738 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5739 = stablehlo.divide %v5737, %v5738 : tensor<1024xf32>
    %v5740 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5742 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5743 = stablehlo.reduce(%v5740 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5744 = stablehlo.divide %v5743, %v5742 : tensor<256xf32>
    %v5745 = stablehlo.reshape %v840 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5747 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5748 = stablehlo.reduce(%v5745 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5749 = stablehlo.broadcast_in_dim %v5748, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5750 = stablehlo.divide %v5749, %v5747 : tensor<256x256x14x14xf32>
    %v5751 = stablehlo.subtract %v5745, %v5750 : tensor<256x256x14x14xf32>
    %v5752 = stablehlo.multiply %v5751, %v5751 : tensor<256x256x14x14xf32>
    %v5753 = stablehlo.reduce(%v5752 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5754 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5755 = stablehlo.divide %v5753, %v5754 : tensor<256xf32>
    %v5756 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5758 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5759 = stablehlo.reduce(%v5756 init: %v5757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5760 = stablehlo.divide %v5759, %v5758 : tensor<256xf32>
    %v5761 = stablehlo.reshape %v869 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5763 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5764 = stablehlo.reduce(%v5761 init: %v5762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5765 = stablehlo.broadcast_in_dim %v5764, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5766 = stablehlo.divide %v5765, %v5763 : tensor<256x256x14x14xf32>
    %v5767 = stablehlo.subtract %v5761, %v5766 : tensor<256x256x14x14xf32>
    %v5768 = stablehlo.multiply %v5767, %v5767 : tensor<256x256x14x14xf32>
    %v5769 = stablehlo.reduce(%v5768 init: %v5762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5770 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5771 = stablehlo.divide %v5769, %v5770 : tensor<256xf32>
    %v5772 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5774 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5775 = stablehlo.reduce(%v5772 init: %v5773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5776 = stablehlo.divide %v5775, %v5774 : tensor<1024xf32>
    %v5777 = stablehlo.reshape %v898 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5779 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5780 = stablehlo.reduce(%v5777 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5781 = stablehlo.broadcast_in_dim %v5780, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5782 = stablehlo.divide %v5781, %v5779 : tensor<256x1024x14x14xf32>
    %v5783 = stablehlo.subtract %v5777, %v5782 : tensor<256x1024x14x14xf32>
    %v5784 = stablehlo.multiply %v5783, %v5783 : tensor<256x1024x14x14xf32>
    %v5785 = stablehlo.reduce(%v5784 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5786 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5787 = stablehlo.divide %v5785, %v5786 : tensor<1024xf32>
    %v5788 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5790 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5791 = stablehlo.reduce(%v5788 init: %v5789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5792 = stablehlo.divide %v5791, %v5790 : tensor<256xf32>
    %v5793 = stablehlo.reshape %v931 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5795 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5796 = stablehlo.reduce(%v5793 init: %v5794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5797 = stablehlo.broadcast_in_dim %v5796, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5798 = stablehlo.divide %v5797, %v5795 : tensor<256x256x14x14xf32>
    %v5799 = stablehlo.subtract %v5793, %v5798 : tensor<256x256x14x14xf32>
    %v5800 = stablehlo.multiply %v5799, %v5799 : tensor<256x256x14x14xf32>
    %v5801 = stablehlo.reduce(%v5800 init: %v5794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5802 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5803 = stablehlo.divide %v5801, %v5802 : tensor<256xf32>
    %v5804 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5806 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5807 = stablehlo.reduce(%v5804 init: %v5805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5808 = stablehlo.divide %v5807, %v5806 : tensor<256xf32>
    %v5809 = stablehlo.reshape %v960 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5811 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5812 = stablehlo.reduce(%v5809 init: %v5810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5813 = stablehlo.broadcast_in_dim %v5812, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5814 = stablehlo.divide %v5813, %v5811 : tensor<256x256x14x14xf32>
    %v5815 = stablehlo.subtract %v5809, %v5814 : tensor<256x256x14x14xf32>
    %v5816 = stablehlo.multiply %v5815, %v5815 : tensor<256x256x14x14xf32>
    %v5817 = stablehlo.reduce(%v5816 init: %v5810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5818 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5819 = stablehlo.divide %v5817, %v5818 : tensor<256xf32>
    %v5820 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5822 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5823 = stablehlo.reduce(%v5820 init: %v5821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5824 = stablehlo.divide %v5823, %v5822 : tensor<1024xf32>
    %v5825 = stablehlo.reshape %v989 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5827 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5828 = stablehlo.reduce(%v5825 init: %v5826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5829 = stablehlo.broadcast_in_dim %v5828, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5830 = stablehlo.divide %v5829, %v5827 : tensor<256x1024x14x14xf32>
    %v5831 = stablehlo.subtract %v5825, %v5830 : tensor<256x1024x14x14xf32>
    %v5832 = stablehlo.multiply %v5831, %v5831 : tensor<256x1024x14x14xf32>
    %v5833 = stablehlo.reduce(%v5832 init: %v5826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5834 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5835 = stablehlo.divide %v5833, %v5834 : tensor<1024xf32>
    %v5836 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5838 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5839 = stablehlo.reduce(%v5836 init: %v5837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5840 = stablehlo.divide %v5839, %v5838 : tensor<256xf32>
    %v5841 = stablehlo.reshape %v1022 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5843 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5844 = stablehlo.reduce(%v5841 init: %v5842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5845 = stablehlo.broadcast_in_dim %v5844, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5846 = stablehlo.divide %v5845, %v5843 : tensor<256x256x14x14xf32>
    %v5847 = stablehlo.subtract %v5841, %v5846 : tensor<256x256x14x14xf32>
    %v5848 = stablehlo.multiply %v5847, %v5847 : tensor<256x256x14x14xf32>
    %v5849 = stablehlo.reduce(%v5848 init: %v5842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5850 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5851 = stablehlo.divide %v5849, %v5850 : tensor<256xf32>
    %v5852 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5854 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5855 = stablehlo.reduce(%v5852 init: %v5853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5856 = stablehlo.divide %v5855, %v5854 : tensor<256xf32>
    %v5857 = stablehlo.reshape %v1051 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5859 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5860 = stablehlo.reduce(%v5857 init: %v5858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5861 = stablehlo.broadcast_in_dim %v5860, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5862 = stablehlo.divide %v5861, %v5859 : tensor<256x256x14x14xf32>
    %v5863 = stablehlo.subtract %v5857, %v5862 : tensor<256x256x14x14xf32>
    %v5864 = stablehlo.multiply %v5863, %v5863 : tensor<256x256x14x14xf32>
    %v5865 = stablehlo.reduce(%v5864 init: %v5858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5866 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5867 = stablehlo.divide %v5865, %v5866 : tensor<256xf32>
    %v5868 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5869 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5870 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5871 = stablehlo.reduce(%v5868 init: %v5869) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5872 = stablehlo.divide %v5871, %v5870 : tensor<1024xf32>
    %v5873 = stablehlo.reshape %v1080 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5875 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5876 = stablehlo.reduce(%v5873 init: %v5874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5877 = stablehlo.broadcast_in_dim %v5876, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5878 = stablehlo.divide %v5877, %v5875 : tensor<256x1024x14x14xf32>
    %v5879 = stablehlo.subtract %v5873, %v5878 : tensor<256x1024x14x14xf32>
    %v5880 = stablehlo.multiply %v5879, %v5879 : tensor<256x1024x14x14xf32>
    %v5881 = stablehlo.reduce(%v5880 init: %v5874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5882 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5883 = stablehlo.divide %v5881, %v5882 : tensor<1024xf32>
    %v5884 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5886 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5887 = stablehlo.reduce(%v5884 init: %v5885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5888 = stablehlo.divide %v5887, %v5886 : tensor<256xf32>
    %v5889 = stablehlo.reshape %v1113 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5891 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5892 = stablehlo.reduce(%v5889 init: %v5890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5893 = stablehlo.broadcast_in_dim %v5892, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5894 = stablehlo.divide %v5893, %v5891 : tensor<256x256x14x14xf32>
    %v5895 = stablehlo.subtract %v5889, %v5894 : tensor<256x256x14x14xf32>
    %v5896 = stablehlo.multiply %v5895, %v5895 : tensor<256x256x14x14xf32>
    %v5897 = stablehlo.reduce(%v5896 init: %v5890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5898 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5899 = stablehlo.divide %v5897, %v5898 : tensor<256xf32>
    %v5900 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5902 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5903 = stablehlo.reduce(%v5900 init: %v5901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5904 = stablehlo.divide %v5903, %v5902 : tensor<256xf32>
    %v5905 = stablehlo.reshape %v1142 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5907 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5908 = stablehlo.reduce(%v5905 init: %v5906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5909 = stablehlo.broadcast_in_dim %v5908, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5910 = stablehlo.divide %v5909, %v5907 : tensor<256x256x14x14xf32>
    %v5911 = stablehlo.subtract %v5905, %v5910 : tensor<256x256x14x14xf32>
    %v5912 = stablehlo.multiply %v5911, %v5911 : tensor<256x256x14x14xf32>
    %v5913 = stablehlo.reduce(%v5912 init: %v5906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5914 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5915 = stablehlo.divide %v5913, %v5914 : tensor<256xf32>
    %v5916 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5918 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5919 = stablehlo.reduce(%v5916 init: %v5917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5920 = stablehlo.divide %v5919, %v5918 : tensor<1024xf32>
    %v5921 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5923 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5924 = stablehlo.reduce(%v5921 init: %v5922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5925 = stablehlo.broadcast_in_dim %v5924, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5926 = stablehlo.divide %v5925, %v5923 : tensor<256x1024x14x14xf32>
    %v5927 = stablehlo.subtract %v5921, %v5926 : tensor<256x1024x14x14xf32>
    %v5928 = stablehlo.multiply %v5927, %v5927 : tensor<256x1024x14x14xf32>
    %v5929 = stablehlo.reduce(%v5928 init: %v5922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5930 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5931 = stablehlo.divide %v5929, %v5930 : tensor<1024xf32>
    %v5932 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5934 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5935 = stablehlo.reduce(%v5932 init: %v5933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5936 = stablehlo.divide %v5935, %v5934 : tensor<256xf32>
    %v5937 = stablehlo.reshape %v1204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5939 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5940 = stablehlo.reduce(%v5937 init: %v5938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5941 = stablehlo.broadcast_in_dim %v5940, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5942 = stablehlo.divide %v5941, %v5939 : tensor<256x256x14x14xf32>
    %v5943 = stablehlo.subtract %v5937, %v5942 : tensor<256x256x14x14xf32>
    %v5944 = stablehlo.multiply %v5943, %v5943 : tensor<256x256x14x14xf32>
    %v5945 = stablehlo.reduce(%v5944 init: %v5938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5946 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5947 = stablehlo.divide %v5945, %v5946 : tensor<256xf32>
    %v5948 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5950 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5951 = stablehlo.reduce(%v5948 init: %v5949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5952 = stablehlo.divide %v5951, %v5950 : tensor<256xf32>
    %v5953 = stablehlo.reshape %v1233 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5955 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5956 = stablehlo.reduce(%v5953 init: %v5954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5957 = stablehlo.broadcast_in_dim %v5956, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5958 = stablehlo.divide %v5957, %v5955 : tensor<256x256x14x14xf32>
    %v5959 = stablehlo.subtract %v5953, %v5958 : tensor<256x256x14x14xf32>
    %v5960 = stablehlo.multiply %v5959, %v5959 : tensor<256x256x14x14xf32>
    %v5961 = stablehlo.reduce(%v5960 init: %v5954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5962 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5963 = stablehlo.divide %v5961, %v5962 : tensor<256xf32>
    %v5964 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5966 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5967 = stablehlo.reduce(%v5964 init: %v5965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5968 = stablehlo.divide %v5967, %v5966 : tensor<1024xf32>
    %v5969 = stablehlo.reshape %v1262 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5971 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5972 = stablehlo.reduce(%v5969 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5973 = stablehlo.broadcast_in_dim %v5972, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5974 = stablehlo.divide %v5973, %v5971 : tensor<256x1024x14x14xf32>
    %v5975 = stablehlo.subtract %v5969, %v5974 : tensor<256x1024x14x14xf32>
    %v5976 = stablehlo.multiply %v5975, %v5975 : tensor<256x1024x14x14xf32>
    %v5977 = stablehlo.reduce(%v5976 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5978 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5979 = stablehlo.divide %v5977, %v5978 : tensor<1024xf32>
    %v5980 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v5981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5982 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5983 = stablehlo.reduce(%v5980 init: %v5981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5984 = stablehlo.divide %v5983, %v5982 : tensor<512xf32>
    %v5985 = stablehlo.reshape %v1295 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v5986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5987 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v5988 = stablehlo.reduce(%v5985 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5989 = stablehlo.broadcast_in_dim %v5988, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v5990 = stablehlo.divide %v5989, %v5987 : tensor<256x512x14x14xf32>
    %v5991 = stablehlo.subtract %v5985, %v5990 : tensor<256x512x14x14xf32>
    %v5992 = stablehlo.multiply %v5991, %v5991 : tensor<256x512x14x14xf32>
    %v5993 = stablehlo.reduce(%v5992 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5994 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5995 = stablehlo.divide %v5993, %v5994 : tensor<512xf32>
    %v5996 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5998 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5999 = stablehlo.reduce(%v5996 init: %v5997) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6000 = stablehlo.divide %v5999, %v5998 : tensor<512xf32>
    %v6001 = stablehlo.reshape %v1324 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6003 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6004 = stablehlo.reduce(%v6001 init: %v6002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6005 = stablehlo.broadcast_in_dim %v6004, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6006 = stablehlo.divide %v6005, %v6003 : tensor<256x512x7x7xf32>
    %v6007 = stablehlo.subtract %v6001, %v6006 : tensor<256x512x7x7xf32>
    %v6008 = stablehlo.multiply %v6007, %v6007 : tensor<256x512x7x7xf32>
    %v6009 = stablehlo.reduce(%v6008 init: %v6002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6010 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6011 = stablehlo.divide %v6009, %v6010 : tensor<512xf32>
    %v6012 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6014 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6015 = stablehlo.reduce(%v6012 init: %v6013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6016 = stablehlo.divide %v6015, %v6014 : tensor<2048xf32>
    %v6017 = stablehlo.reshape %v1353 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6019 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6020 = stablehlo.reduce(%v6017 init: %v6018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6021 = stablehlo.broadcast_in_dim %v6020, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6022 = stablehlo.divide %v6021, %v6019 : tensor<256x2048x7x7xf32>
    %v6023 = stablehlo.subtract %v6017, %v6022 : tensor<256x2048x7x7xf32>
    %v6024 = stablehlo.multiply %v6023, %v6023 : tensor<256x2048x7x7xf32>
    %v6025 = stablehlo.reduce(%v6024 init: %v6018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6026 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6027 = stablehlo.divide %v6025, %v6026 : tensor<2048xf32>
    %v6028 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6030 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6031 = stablehlo.reduce(%v6028 init: %v6029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6032 = stablehlo.divide %v6031, %v6030 : tensor<2048xf32>
    %v6033 = stablehlo.reshape %v1378 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6035 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6036 = stablehlo.reduce(%v6033 init: %v6034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6037 = stablehlo.broadcast_in_dim %v6036, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6038 = stablehlo.divide %v6037, %v6035 : tensor<256x2048x7x7xf32>
    %v6039 = stablehlo.subtract %v6033, %v6038 : tensor<256x2048x7x7xf32>
    %v6040 = stablehlo.multiply %v6039, %v6039 : tensor<256x2048x7x7xf32>
    %v6041 = stablehlo.reduce(%v6040 init: %v6034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6042 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6043 = stablehlo.divide %v6041, %v6042 : tensor<2048xf32>
    %v6044 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6046 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6047 = stablehlo.reduce(%v6044 init: %v6045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6048 = stablehlo.divide %v6047, %v6046 : tensor<512xf32>
    %v6049 = stablehlo.reshape %v1411 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6051 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6052 = stablehlo.reduce(%v6049 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6053 = stablehlo.broadcast_in_dim %v6052, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6054 = stablehlo.divide %v6053, %v6051 : tensor<256x512x7x7xf32>
    %v6055 = stablehlo.subtract %v6049, %v6054 : tensor<256x512x7x7xf32>
    %v6056 = stablehlo.multiply %v6055, %v6055 : tensor<256x512x7x7xf32>
    %v6057 = stablehlo.reduce(%v6056 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6058 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6059 = stablehlo.divide %v6057, %v6058 : tensor<512xf32>
    %v6060 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6062 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6063 = stablehlo.reduce(%v6060 init: %v6061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6064 = stablehlo.divide %v6063, %v6062 : tensor<512xf32>
    %v6065 = stablehlo.reshape %v1440 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6067 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6068 = stablehlo.reduce(%v6065 init: %v6066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6069 = stablehlo.broadcast_in_dim %v6068, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6070 = stablehlo.divide %v6069, %v6067 : tensor<256x512x7x7xf32>
    %v6071 = stablehlo.subtract %v6065, %v6070 : tensor<256x512x7x7xf32>
    %v6072 = stablehlo.multiply %v6071, %v6071 : tensor<256x512x7x7xf32>
    %v6073 = stablehlo.reduce(%v6072 init: %v6066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6074 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6075 = stablehlo.divide %v6073, %v6074 : tensor<512xf32>
    %v6076 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6078 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6079 = stablehlo.reduce(%v6076 init: %v6077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6080 = stablehlo.divide %v6079, %v6078 : tensor<2048xf32>
    %v6081 = stablehlo.reshape %v1469 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6083 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6084 = stablehlo.reduce(%v6081 init: %v6082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6085 = stablehlo.broadcast_in_dim %v6084, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6086 = stablehlo.divide %v6085, %v6083 : tensor<256x2048x7x7xf32>
    %v6087 = stablehlo.subtract %v6081, %v6086 : tensor<256x2048x7x7xf32>
    %v6088 = stablehlo.multiply %v6087, %v6087 : tensor<256x2048x7x7xf32>
    %v6089 = stablehlo.reduce(%v6088 init: %v6082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6090 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6091 = stablehlo.divide %v6089, %v6090 : tensor<2048xf32>
    %v6092 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6094 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6095 = stablehlo.reduce(%v6092 init: %v6093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6096 = stablehlo.divide %v6095, %v6094 : tensor<512xf32>
    %v6097 = stablehlo.reshape %v1502 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6099 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6100 = stablehlo.reduce(%v6097 init: %v6098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6101 = stablehlo.broadcast_in_dim %v6100, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6102 = stablehlo.divide %v6101, %v6099 : tensor<256x512x7x7xf32>
    %v6103 = stablehlo.subtract %v6097, %v6102 : tensor<256x512x7x7xf32>
    %v6104 = stablehlo.multiply %v6103, %v6103 : tensor<256x512x7x7xf32>
    %v6105 = stablehlo.reduce(%v6104 init: %v6098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6106 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6107 = stablehlo.divide %v6105, %v6106 : tensor<512xf32>
    %v6108 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6110 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6111 = stablehlo.reduce(%v6108 init: %v6109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6112 = stablehlo.divide %v6111, %v6110 : tensor<512xf32>
    %v6113 = stablehlo.reshape %v1531 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v6114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6115 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v6116 = stablehlo.reduce(%v6113 init: %v6114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6117 = stablehlo.broadcast_in_dim %v6116, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v6118 = stablehlo.divide %v6117, %v6115 : tensor<256x512x7x7xf32>
    %v6119 = stablehlo.subtract %v6113, %v6118 : tensor<256x512x7x7xf32>
    %v6120 = stablehlo.multiply %v6119, %v6119 : tensor<256x512x7x7xf32>
    %v6121 = stablehlo.reduce(%v6120 init: %v6114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6122 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6123 = stablehlo.divide %v6121, %v6122 : tensor<512xf32>
    %v6124 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6125 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6126 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6127 = stablehlo.reduce(%v6124 init: %v6125) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6128 = stablehlo.divide %v6127, %v6126 : tensor<2048xf32>
    %v6129 = stablehlo.reshape %v1560 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v6130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6131 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v6132 = stablehlo.reduce(%v6129 init: %v6130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6133 = stablehlo.broadcast_in_dim %v6132, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v6134 = stablehlo.divide %v6133, %v6131 : tensor<256x2048x7x7xf32>
    %v6135 = stablehlo.subtract %v6129, %v6134 : tensor<256x2048x7x7xf32>
    %v6136 = stablehlo.multiply %v6135, %v6135 : tensor<256x2048x7x7xf32>
    %v6137 = stablehlo.reduce(%v6136 init: %v6130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6138 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v6139 = stablehlo.divide %v6137, %v6138 : tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v6140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6141 = stablehlo.multiply %v6140, %sW : tensor<64x3x7x7xf32>
    %v6142 = stablehlo.add %v6141, %v5270 : tensor<64x3x7x7xf32>
    %v6143 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6144 = stablehlo.multiply %v6143, %sWv : tensor<64x3x7x7xf32>
    %v6145 = stablehlo.add %v6144, %v6142 : tensor<64x3x7x7xf32>
    %v6146 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6147 = stablehlo.multiply %v6146, %v6145 : tensor<64x3x7x7xf32>
    %v6148 = stablehlo.subtract %sW, %v6147 : tensor<64x3x7x7xf32>
    %v6149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6150 = stablehlo.multiply %v6149, %sg : tensor<64xf32>
    %v6151 = stablehlo.add %v6150, %v5288 : tensor<64xf32>
    %v6152 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6153 = stablehlo.multiply %v6152, %sgv : tensor<64xf32>
    %v6154 = stablehlo.add %v6153, %v6151 : tensor<64xf32>
    %v6155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6156 = stablehlo.multiply %v6155, %v6154 : tensor<64xf32>
    %v6157 = stablehlo.subtract %sg, %v6156 : tensor<64xf32>
    %v6158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6159 = stablehlo.multiply %v6158, %sbt : tensor<64xf32>
    %v6160 = stablehlo.add %v6159, %v5291 : tensor<64xf32>
    %v6161 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6162 = stablehlo.multiply %v6161, %sbtv : tensor<64xf32>
    %v6163 = stablehlo.add %v6162, %v6160 : tensor<64xf32>
    %v6164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6165 = stablehlo.multiply %v6164, %v6163 : tensor<64xf32>
    %v6166 = stablehlo.subtract %sbt, %v6165 : tensor<64xf32>
    %v6167 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6168 = stablehlo.multiply %v6167, %s1b0W1 : tensor<64x64x1x1xf32>
    %v6169 = stablehlo.add %v6168, %v5119 : tensor<64x64x1x1xf32>
    %v6170 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6171 = stablehlo.multiply %v6170, %s1b0W1v : tensor<64x64x1x1xf32>
    %v6172 = stablehlo.add %v6171, %v6169 : tensor<64x64x1x1xf32>
    %v6173 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6174 = stablehlo.multiply %v6173, %v6172 : tensor<64x64x1x1xf32>
    %v6175 = stablehlo.subtract %s1b0W1, %v6174 : tensor<64x64x1x1xf32>
    %v6176 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6177 = stablehlo.multiply %v6176, %s1b0g1 : tensor<64xf32>
    %v6178 = stablehlo.add %v6177, %v5137 : tensor<64xf32>
    %v6179 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6180 = stablehlo.multiply %v6179, %s1b0g1v : tensor<64xf32>
    %v6181 = stablehlo.add %v6180, %v6178 : tensor<64xf32>
    %v6182 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6183 = stablehlo.multiply %v6182, %v6181 : tensor<64xf32>
    %v6184 = stablehlo.subtract %s1b0g1, %v6183 : tensor<64xf32>
    %v6185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6186 = stablehlo.multiply %v6185, %s1b0bt1 : tensor<64xf32>
    %v6187 = stablehlo.add %v6186, %v5140 : tensor<64xf32>
    %v6188 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6189 = stablehlo.multiply %v6188, %s1b0bt1v : tensor<64xf32>
    %v6190 = stablehlo.add %v6189, %v6187 : tensor<64xf32>
    %v6191 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6192 = stablehlo.multiply %v6191, %v6190 : tensor<64xf32>
    %v6193 = stablehlo.subtract %s1b0bt1, %v6192 : tensor<64xf32>
    %v6194 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6195 = stablehlo.multiply %v6194, %s1b0W2 : tensor<64x64x3x3xf32>
    %v6196 = stablehlo.add %v6195, %v5146 : tensor<64x64x3x3xf32>
    %v6197 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6198 = stablehlo.multiply %v6197, %s1b0W2v : tensor<64x64x3x3xf32>
    %v6199 = stablehlo.add %v6198, %v6196 : tensor<64x64x3x3xf32>
    %v6200 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6201 = stablehlo.multiply %v6200, %v6199 : tensor<64x64x3x3xf32>
    %v6202 = stablehlo.subtract %s1b0W2, %v6201 : tensor<64x64x3x3xf32>
    %v6203 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6204 = stablehlo.multiply %v6203, %s1b0g2 : tensor<64xf32>
    %v6205 = stablehlo.add %v6204, %v5164 : tensor<64xf32>
    %v6206 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6207 = stablehlo.multiply %v6206, %s1b0g2v : tensor<64xf32>
    %v6208 = stablehlo.add %v6207, %v6205 : tensor<64xf32>
    %v6209 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6210 = stablehlo.multiply %v6209, %v6208 : tensor<64xf32>
    %v6211 = stablehlo.subtract %s1b0g2, %v6210 : tensor<64xf32>
    %v6212 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6213 = stablehlo.multiply %v6212, %s1b0bt2 : tensor<64xf32>
    %v6214 = stablehlo.add %v6213, %v5167 : tensor<64xf32>
    %v6215 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6216 = stablehlo.multiply %v6215, %s1b0bt2v : tensor<64xf32>
    %v6217 = stablehlo.add %v6216, %v6214 : tensor<64xf32>
    %v6218 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6219 = stablehlo.multiply %v6218, %v6217 : tensor<64xf32>
    %v6220 = stablehlo.subtract %s1b0bt2, %v6219 : tensor<64xf32>
    %v6221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6222 = stablehlo.multiply %v6221, %s1b0W3 : tensor<256x64x1x1xf32>
    %v6223 = stablehlo.add %v6222, %v5173 : tensor<256x64x1x1xf32>
    %v6224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6225 = stablehlo.multiply %v6224, %s1b0W3v : tensor<256x64x1x1xf32>
    %v6226 = stablehlo.add %v6225, %v6223 : tensor<256x64x1x1xf32>
    %v6227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6228 = stablehlo.multiply %v6227, %v6226 : tensor<256x64x1x1xf32>
    %v6229 = stablehlo.subtract %s1b0W3, %v6228 : tensor<256x64x1x1xf32>
    %v6230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6231 = stablehlo.multiply %v6230, %s1b0g3 : tensor<256xf32>
    %v6232 = stablehlo.add %v6231, %v5191 : tensor<256xf32>
    %v6233 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6234 = stablehlo.multiply %v6233, %s1b0g3v : tensor<256xf32>
    %v6235 = stablehlo.add %v6234, %v6232 : tensor<256xf32>
    %v6236 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6237 = stablehlo.multiply %v6236, %v6235 : tensor<256xf32>
    %v6238 = stablehlo.subtract %s1b0g3, %v6237 : tensor<256xf32>
    %v6239 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6240 = stablehlo.multiply %v6239, %s1b0bt3 : tensor<256xf32>
    %v6241 = stablehlo.add %v6240, %v5194 : tensor<256xf32>
    %v6242 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6243 = stablehlo.multiply %v6242, %s1b0bt3v : tensor<256xf32>
    %v6244 = stablehlo.add %v6243, %v6241 : tensor<256xf32>
    %v6245 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6246 = stablehlo.multiply %v6245, %v6244 : tensor<256xf32>
    %v6247 = stablehlo.subtract %s1b0bt3, %v6246 : tensor<256xf32>
    %v6248 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6249 = stablehlo.multiply %v6248, %s1b0Wp : tensor<256x64x1x1xf32>
    %v6250 = stablehlo.add %v6249, %v5200 : tensor<256x64x1x1xf32>
    %v6251 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6252 = stablehlo.multiply %v6251, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v6253 = stablehlo.add %v6252, %v6250 : tensor<256x64x1x1xf32>
    %v6254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6255 = stablehlo.multiply %v6254, %v6253 : tensor<256x64x1x1xf32>
    %v6256 = stablehlo.subtract %s1b0Wp, %v6255 : tensor<256x64x1x1xf32>
    %v6257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6258 = stablehlo.multiply %v6257, %s1b0gp : tensor<256xf32>
    %v6259 = stablehlo.add %v6258, %v5218 : tensor<256xf32>
    %v6260 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6261 = stablehlo.multiply %v6260, %s1b0gpv : tensor<256xf32>
    %v6262 = stablehlo.add %v6261, %v6259 : tensor<256xf32>
    %v6263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6264 = stablehlo.multiply %v6263, %v6262 : tensor<256xf32>
    %v6265 = stablehlo.subtract %s1b0gp, %v6264 : tensor<256xf32>
    %v6266 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6267 = stablehlo.multiply %v6266, %s1b0btp : tensor<256xf32>
    %v6268 = stablehlo.add %v6267, %v5221 : tensor<256xf32>
    %v6269 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6270 = stablehlo.multiply %v6269, %s1b0btpv : tensor<256xf32>
    %v6271 = stablehlo.add %v6270, %v6268 : tensor<256xf32>
    %v6272 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6273 = stablehlo.multiply %v6272, %v6271 : tensor<256xf32>
    %v6274 = stablehlo.subtract %s1b0btp, %v6273 : tensor<256xf32>
    %v6275 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6276 = stablehlo.multiply %v6275, %s1b1W1 : tensor<64x256x1x1xf32>
    %v6277 = stablehlo.add %v6276, %v4876 : tensor<64x256x1x1xf32>
    %v6278 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6279 = stablehlo.multiply %v6278, %s1b1W1v : tensor<64x256x1x1xf32>
    %v6280 = stablehlo.add %v6279, %v6277 : tensor<64x256x1x1xf32>
    %v6281 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6282 = stablehlo.multiply %v6281, %v6280 : tensor<64x256x1x1xf32>
    %v6283 = stablehlo.subtract %s1b1W1, %v6282 : tensor<64x256x1x1xf32>
    %v6284 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6285 = stablehlo.multiply %v6284, %s1b1g1 : tensor<64xf32>
    %v6286 = stablehlo.add %v6285, %v4894 : tensor<64xf32>
    %v6287 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6288 = stablehlo.multiply %v6287, %s1b1g1v : tensor<64xf32>
    %v6289 = stablehlo.add %v6288, %v6286 : tensor<64xf32>
    %v6290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6291 = stablehlo.multiply %v6290, %v6289 : tensor<64xf32>
    %v6292 = stablehlo.subtract %s1b1g1, %v6291 : tensor<64xf32>
    %v6293 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6294 = stablehlo.multiply %v6293, %s1b1bt1 : tensor<64xf32>
    %v6295 = stablehlo.add %v6294, %v4897 : tensor<64xf32>
    %v6296 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6297 = stablehlo.multiply %v6296, %s1b1bt1v : tensor<64xf32>
    %v6298 = stablehlo.add %v6297, %v6295 : tensor<64xf32>
    %v6299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6300 = stablehlo.multiply %v6299, %v6298 : tensor<64xf32>
    %v6301 = stablehlo.subtract %s1b1bt1, %v6300 : tensor<64xf32>
    %v6302 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6303 = stablehlo.multiply %v6302, %s1b1W2 : tensor<64x64x3x3xf32>
    %v6304 = stablehlo.add %v6303, %v4903 : tensor<64x64x3x3xf32>
    %v6305 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6306 = stablehlo.multiply %v6305, %s1b1W2v : tensor<64x64x3x3xf32>
    %v6307 = stablehlo.add %v6306, %v6304 : tensor<64x64x3x3xf32>
    %v6308 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6309 = stablehlo.multiply %v6308, %v6307 : tensor<64x64x3x3xf32>
    %v6310 = stablehlo.subtract %s1b1W2, %v6309 : tensor<64x64x3x3xf32>
    %v6311 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6312 = stablehlo.multiply %v6311, %s1b1g2 : tensor<64xf32>
    %v6313 = stablehlo.add %v6312, %v4921 : tensor<64xf32>
    %v6314 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6315 = stablehlo.multiply %v6314, %s1b1g2v : tensor<64xf32>
    %v6316 = stablehlo.add %v6315, %v6313 : tensor<64xf32>
    %v6317 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6318 = stablehlo.multiply %v6317, %v6316 : tensor<64xf32>
    %v6319 = stablehlo.subtract %s1b1g2, %v6318 : tensor<64xf32>
    %v6320 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6321 = stablehlo.multiply %v6320, %s1b1bt2 : tensor<64xf32>
    %v6322 = stablehlo.add %v6321, %v4924 : tensor<64xf32>
    %v6323 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6324 = stablehlo.multiply %v6323, %s1b1bt2v : tensor<64xf32>
    %v6325 = stablehlo.add %v6324, %v6322 : tensor<64xf32>
    %v6326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6327 = stablehlo.multiply %v6326, %v6325 : tensor<64xf32>
    %v6328 = stablehlo.subtract %s1b1bt2, %v6327 : tensor<64xf32>
    %v6329 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6330 = stablehlo.multiply %v6329, %s1b1W3 : tensor<256x64x1x1xf32>
    %v6331 = stablehlo.add %v6330, %v4930 : tensor<256x64x1x1xf32>
    %v6332 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6333 = stablehlo.multiply %v6332, %s1b1W3v : tensor<256x64x1x1xf32>
    %v6334 = stablehlo.add %v6333, %v6331 : tensor<256x64x1x1xf32>
    %v6335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6336 = stablehlo.multiply %v6335, %v6334 : tensor<256x64x1x1xf32>
    %v6337 = stablehlo.subtract %s1b1W3, %v6336 : tensor<256x64x1x1xf32>
    %v6338 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6339 = stablehlo.multiply %v6338, %s1b1g3 : tensor<256xf32>
    %v6340 = stablehlo.add %v6339, %v4948 : tensor<256xf32>
    %v6341 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6342 = stablehlo.multiply %v6341, %s1b1g3v : tensor<256xf32>
    %v6343 = stablehlo.add %v6342, %v6340 : tensor<256xf32>
    %v6344 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6345 = stablehlo.multiply %v6344, %v6343 : tensor<256xf32>
    %v6346 = stablehlo.subtract %s1b1g3, %v6345 : tensor<256xf32>
    %v6347 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6348 = stablehlo.multiply %v6347, %s1b1bt3 : tensor<256xf32>
    %v6349 = stablehlo.add %v6348, %v4951 : tensor<256xf32>
    %v6350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6351 = stablehlo.multiply %v6350, %s1b1bt3v : tensor<256xf32>
    %v6352 = stablehlo.add %v6351, %v6349 : tensor<256xf32>
    %v6353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6354 = stablehlo.multiply %v6353, %v6352 : tensor<256xf32>
    %v6355 = stablehlo.subtract %s1b1bt3, %v6354 : tensor<256xf32>
    %v6356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6357 = stablehlo.multiply %v6356, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6358 = stablehlo.add %v6357, %v4668 : tensor<64x256x1x1xf32>
    %v6359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6360 = stablehlo.multiply %v6359, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6361 = stablehlo.add %v6360, %v6358 : tensor<64x256x1x1xf32>
    %v6362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6363 = stablehlo.multiply %v6362, %v6361 : tensor<64x256x1x1xf32>
    %v6364 = stablehlo.subtract %s1b2W1, %v6363 : tensor<64x256x1x1xf32>
    %v6365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6366 = stablehlo.multiply %v6365, %s1b2g1 : tensor<64xf32>
    %v6367 = stablehlo.add %v6366, %v4686 : tensor<64xf32>
    %v6368 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6369 = stablehlo.multiply %v6368, %s1b2g1v : tensor<64xf32>
    %v6370 = stablehlo.add %v6369, %v6367 : tensor<64xf32>
    %v6371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6372 = stablehlo.multiply %v6371, %v6370 : tensor<64xf32>
    %v6373 = stablehlo.subtract %s1b2g1, %v6372 : tensor<64xf32>
    %v6374 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6375 = stablehlo.multiply %v6374, %s1b2bt1 : tensor<64xf32>
    %v6376 = stablehlo.add %v6375, %v4689 : tensor<64xf32>
    %v6377 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6378 = stablehlo.multiply %v6377, %s1b2bt1v : tensor<64xf32>
    %v6379 = stablehlo.add %v6378, %v6376 : tensor<64xf32>
    %v6380 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6381 = stablehlo.multiply %v6380, %v6379 : tensor<64xf32>
    %v6382 = stablehlo.subtract %s1b2bt1, %v6381 : tensor<64xf32>
    %v6383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6384 = stablehlo.multiply %v6383, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6385 = stablehlo.add %v6384, %v4695 : tensor<64x64x3x3xf32>
    %v6386 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6387 = stablehlo.multiply %v6386, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6388 = stablehlo.add %v6387, %v6385 : tensor<64x64x3x3xf32>
    %v6389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6390 = stablehlo.multiply %v6389, %v6388 : tensor<64x64x3x3xf32>
    %v6391 = stablehlo.subtract %s1b2W2, %v6390 : tensor<64x64x3x3xf32>
    %v6392 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6393 = stablehlo.multiply %v6392, %s1b2g2 : tensor<64xf32>
    %v6394 = stablehlo.add %v6393, %v4713 : tensor<64xf32>
    %v6395 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6396 = stablehlo.multiply %v6395, %s1b2g2v : tensor<64xf32>
    %v6397 = stablehlo.add %v6396, %v6394 : tensor<64xf32>
    %v6398 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6399 = stablehlo.multiply %v6398, %v6397 : tensor<64xf32>
    %v6400 = stablehlo.subtract %s1b2g2, %v6399 : tensor<64xf32>
    %v6401 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6402 = stablehlo.multiply %v6401, %s1b2bt2 : tensor<64xf32>
    %v6403 = stablehlo.add %v6402, %v4716 : tensor<64xf32>
    %v6404 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6405 = stablehlo.multiply %v6404, %s1b2bt2v : tensor<64xf32>
    %v6406 = stablehlo.add %v6405, %v6403 : tensor<64xf32>
    %v6407 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6408 = stablehlo.multiply %v6407, %v6406 : tensor<64xf32>
    %v6409 = stablehlo.subtract %s1b2bt2, %v6408 : tensor<64xf32>
    %v6410 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6411 = stablehlo.multiply %v6410, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6412 = stablehlo.add %v6411, %v4722 : tensor<256x64x1x1xf32>
    %v6413 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6414 = stablehlo.multiply %v6413, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6415 = stablehlo.add %v6414, %v6412 : tensor<256x64x1x1xf32>
    %v6416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6417 = stablehlo.multiply %v6416, %v6415 : tensor<256x64x1x1xf32>
    %v6418 = stablehlo.subtract %s1b2W3, %v6417 : tensor<256x64x1x1xf32>
    %v6419 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6420 = stablehlo.multiply %v6419, %s1b2g3 : tensor<256xf32>
    %v6421 = stablehlo.add %v6420, %v4740 : tensor<256xf32>
    %v6422 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6423 = stablehlo.multiply %v6422, %s1b2g3v : tensor<256xf32>
    %v6424 = stablehlo.add %v6423, %v6421 : tensor<256xf32>
    %v6425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6426 = stablehlo.multiply %v6425, %v6424 : tensor<256xf32>
    %v6427 = stablehlo.subtract %s1b2g3, %v6426 : tensor<256xf32>
    %v6428 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6429 = stablehlo.multiply %v6428, %s1b2bt3 : tensor<256xf32>
    %v6430 = stablehlo.add %v6429, %v4743 : tensor<256xf32>
    %v6431 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6432 = stablehlo.multiply %v6431, %s1b2bt3v : tensor<256xf32>
    %v6433 = stablehlo.add %v6432, %v6430 : tensor<256xf32>
    %v6434 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6435 = stablehlo.multiply %v6434, %v6433 : tensor<256xf32>
    %v6436 = stablehlo.subtract %s1b2bt3, %v6435 : tensor<256xf32>
    %v6437 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6438 = stablehlo.multiply %v6437, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6439 = stablehlo.add %v6438, %v4429 : tensor<128x256x1x1xf32>
    %v6440 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6441 = stablehlo.multiply %v6440, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6442 = stablehlo.add %v6441, %v6439 : tensor<128x256x1x1xf32>
    %v6443 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6444 = stablehlo.multiply %v6443, %v6442 : tensor<128x256x1x1xf32>
    %v6445 = stablehlo.subtract %s2b0W1, %v6444 : tensor<128x256x1x1xf32>
    %v6446 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6447 = stablehlo.multiply %v6446, %s2b0g1 : tensor<128xf32>
    %v6448 = stablehlo.add %v6447, %v4447 : tensor<128xf32>
    %v6449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6450 = stablehlo.multiply %v6449, %s2b0g1v : tensor<128xf32>
    %v6451 = stablehlo.add %v6450, %v6448 : tensor<128xf32>
    %v6452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6453 = stablehlo.multiply %v6452, %v6451 : tensor<128xf32>
    %v6454 = stablehlo.subtract %s2b0g1, %v6453 : tensor<128xf32>
    %v6455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6456 = stablehlo.multiply %v6455, %s2b0bt1 : tensor<128xf32>
    %v6457 = stablehlo.add %v6456, %v4450 : tensor<128xf32>
    %v6458 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6459 = stablehlo.multiply %v6458, %s2b0bt1v : tensor<128xf32>
    %v6460 = stablehlo.add %v6459, %v6457 : tensor<128xf32>
    %v6461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6462 = stablehlo.multiply %v6461, %v6460 : tensor<128xf32>
    %v6463 = stablehlo.subtract %s2b0bt1, %v6462 : tensor<128xf32>
    %v6464 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6465 = stablehlo.multiply %v6464, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6466 = stablehlo.add %v6465, %v4458 : tensor<128x128x3x3xf32>
    %v6467 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6468 = stablehlo.multiply %v6467, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6469 = stablehlo.add %v6468, %v6466 : tensor<128x128x3x3xf32>
    %v6470 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6471 = stablehlo.multiply %v6470, %v6469 : tensor<128x128x3x3xf32>
    %v6472 = stablehlo.subtract %s2b0W2, %v6471 : tensor<128x128x3x3xf32>
    %v6473 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6474 = stablehlo.multiply %v6473, %s2b0g2 : tensor<128xf32>
    %v6475 = stablehlo.add %v6474, %v4476 : tensor<128xf32>
    %v6476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6477 = stablehlo.multiply %v6476, %s2b0g2v : tensor<128xf32>
    %v6478 = stablehlo.add %v6477, %v6475 : tensor<128xf32>
    %v6479 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6480 = stablehlo.multiply %v6479, %v6478 : tensor<128xf32>
    %v6481 = stablehlo.subtract %s2b0g2, %v6480 : tensor<128xf32>
    %v6482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6483 = stablehlo.multiply %v6482, %s2b0bt2 : tensor<128xf32>
    %v6484 = stablehlo.add %v6483, %v4479 : tensor<128xf32>
    %v6485 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6486 = stablehlo.multiply %v6485, %s2b0bt2v : tensor<128xf32>
    %v6487 = stablehlo.add %v6486, %v6484 : tensor<128xf32>
    %v6488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6489 = stablehlo.multiply %v6488, %v6487 : tensor<128xf32>
    %v6490 = stablehlo.subtract %s2b0bt2, %v6489 : tensor<128xf32>
    %v6491 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6492 = stablehlo.multiply %v6491, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6493 = stablehlo.add %v6492, %v4485 : tensor<512x128x1x1xf32>
    %v6494 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6495 = stablehlo.multiply %v6494, %s2b0W3v : tensor<512x128x1x1xf32>
    %v6496 = stablehlo.add %v6495, %v6493 : tensor<512x128x1x1xf32>
    %v6497 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6498 = stablehlo.multiply %v6497, %v6496 : tensor<512x128x1x1xf32>
    %v6499 = stablehlo.subtract %s2b0W3, %v6498 : tensor<512x128x1x1xf32>
    %v6500 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6501 = stablehlo.multiply %v6500, %s2b0g3 : tensor<512xf32>
    %v6502 = stablehlo.add %v6501, %v4503 : tensor<512xf32>
    %v6503 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6504 = stablehlo.multiply %v6503, %s2b0g3v : tensor<512xf32>
    %v6505 = stablehlo.add %v6504, %v6502 : tensor<512xf32>
    %v6506 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6507 = stablehlo.multiply %v6506, %v6505 : tensor<512xf32>
    %v6508 = stablehlo.subtract %s2b0g3, %v6507 : tensor<512xf32>
    %v6509 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6510 = stablehlo.multiply %v6509, %s2b0bt3 : tensor<512xf32>
    %v6511 = stablehlo.add %v6510, %v4506 : tensor<512xf32>
    %v6512 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6513 = stablehlo.multiply %v6512, %s2b0bt3v : tensor<512xf32>
    %v6514 = stablehlo.add %v6513, %v6511 : tensor<512xf32>
    %v6515 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6516 = stablehlo.multiply %v6515, %v6514 : tensor<512xf32>
    %v6517 = stablehlo.subtract %s2b0bt3, %v6516 : tensor<512xf32>
    %v6518 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6519 = stablehlo.multiply %v6518, %s2b0Wp : tensor<512x256x1x1xf32>
    %v6520 = stablehlo.add %v6519, %v4514 : tensor<512x256x1x1xf32>
    %v6521 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6522 = stablehlo.multiply %v6521, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v6523 = stablehlo.add %v6522, %v6520 : tensor<512x256x1x1xf32>
    %v6524 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6525 = stablehlo.multiply %v6524, %v6523 : tensor<512x256x1x1xf32>
    %v6526 = stablehlo.subtract %s2b0Wp, %v6525 : tensor<512x256x1x1xf32>
    %v6527 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6528 = stablehlo.multiply %v6527, %s2b0gp : tensor<512xf32>
    %v6529 = stablehlo.add %v6528, %v4532 : tensor<512xf32>
    %v6530 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6531 = stablehlo.multiply %v6530, %s2b0gpv : tensor<512xf32>
    %v6532 = stablehlo.add %v6531, %v6529 : tensor<512xf32>
    %v6533 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6534 = stablehlo.multiply %v6533, %v6532 : tensor<512xf32>
    %v6535 = stablehlo.subtract %s2b0gp, %v6534 : tensor<512xf32>
    %v6536 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6537 = stablehlo.multiply %v6536, %s2b0btp : tensor<512xf32>
    %v6538 = stablehlo.add %v6537, %v4535 : tensor<512xf32>
    %v6539 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6540 = stablehlo.multiply %v6539, %s2b0btpv : tensor<512xf32>
    %v6541 = stablehlo.add %v6540, %v6538 : tensor<512xf32>
    %v6542 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6543 = stablehlo.multiply %v6542, %v6541 : tensor<512xf32>
    %v6544 = stablehlo.subtract %s2b0btp, %v6543 : tensor<512xf32>
    %v6545 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6546 = stablehlo.multiply %v6545, %s2b1W1 : tensor<128x512x1x1xf32>
    %v6547 = stablehlo.add %v6546, %v4182 : tensor<128x512x1x1xf32>
    %v6548 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6549 = stablehlo.multiply %v6548, %s2b1W1v : tensor<128x512x1x1xf32>
    %v6550 = stablehlo.add %v6549, %v6547 : tensor<128x512x1x1xf32>
    %v6551 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6552 = stablehlo.multiply %v6551, %v6550 : tensor<128x512x1x1xf32>
    %v6553 = stablehlo.subtract %s2b1W1, %v6552 : tensor<128x512x1x1xf32>
    %v6554 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6555 = stablehlo.multiply %v6554, %s2b1g1 : tensor<128xf32>
    %v6556 = stablehlo.add %v6555, %v4200 : tensor<128xf32>
    %v6557 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6558 = stablehlo.multiply %v6557, %s2b1g1v : tensor<128xf32>
    %v6559 = stablehlo.add %v6558, %v6556 : tensor<128xf32>
    %v6560 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6561 = stablehlo.multiply %v6560, %v6559 : tensor<128xf32>
    %v6562 = stablehlo.subtract %s2b1g1, %v6561 : tensor<128xf32>
    %v6563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6564 = stablehlo.multiply %v6563, %s2b1bt1 : tensor<128xf32>
    %v6565 = stablehlo.add %v6564, %v4203 : tensor<128xf32>
    %v6566 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6567 = stablehlo.multiply %v6566, %s2b1bt1v : tensor<128xf32>
    %v6568 = stablehlo.add %v6567, %v6565 : tensor<128xf32>
    %v6569 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6570 = stablehlo.multiply %v6569, %v6568 : tensor<128xf32>
    %v6571 = stablehlo.subtract %s2b1bt1, %v6570 : tensor<128xf32>
    %v6572 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6573 = stablehlo.multiply %v6572, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6574 = stablehlo.add %v6573, %v4209 : tensor<128x128x3x3xf32>
    %v6575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6576 = stablehlo.multiply %v6575, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6577 = stablehlo.add %v6576, %v6574 : tensor<128x128x3x3xf32>
    %v6578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6579 = stablehlo.multiply %v6578, %v6577 : tensor<128x128x3x3xf32>
    %v6580 = stablehlo.subtract %s2b1W2, %v6579 : tensor<128x128x3x3xf32>
    %v6581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6582 = stablehlo.multiply %v6581, %s2b1g2 : tensor<128xf32>
    %v6583 = stablehlo.add %v6582, %v4227 : tensor<128xf32>
    %v6584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6585 = stablehlo.multiply %v6584, %s2b1g2v : tensor<128xf32>
    %v6586 = stablehlo.add %v6585, %v6583 : tensor<128xf32>
    %v6587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6588 = stablehlo.multiply %v6587, %v6586 : tensor<128xf32>
    %v6589 = stablehlo.subtract %s2b1g2, %v6588 : tensor<128xf32>
    %v6590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6591 = stablehlo.multiply %v6590, %s2b1bt2 : tensor<128xf32>
    %v6592 = stablehlo.add %v6591, %v4230 : tensor<128xf32>
    %v6593 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6594 = stablehlo.multiply %v6593, %s2b1bt2v : tensor<128xf32>
    %v6595 = stablehlo.add %v6594, %v6592 : tensor<128xf32>
    %v6596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6597 = stablehlo.multiply %v6596, %v6595 : tensor<128xf32>
    %v6598 = stablehlo.subtract %s2b1bt2, %v6597 : tensor<128xf32>
    %v6599 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6600 = stablehlo.multiply %v6599, %s2b1W3 : tensor<512x128x1x1xf32>
    %v6601 = stablehlo.add %v6600, %v4236 : tensor<512x128x1x1xf32>
    %v6602 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6603 = stablehlo.multiply %v6602, %s2b1W3v : tensor<512x128x1x1xf32>
    %v6604 = stablehlo.add %v6603, %v6601 : tensor<512x128x1x1xf32>
    %v6605 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6606 = stablehlo.multiply %v6605, %v6604 : tensor<512x128x1x1xf32>
    %v6607 = stablehlo.subtract %s2b1W3, %v6606 : tensor<512x128x1x1xf32>
    %v6608 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6609 = stablehlo.multiply %v6608, %s2b1g3 : tensor<512xf32>
    %v6610 = stablehlo.add %v6609, %v4254 : tensor<512xf32>
    %v6611 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6612 = stablehlo.multiply %v6611, %s2b1g3v : tensor<512xf32>
    %v6613 = stablehlo.add %v6612, %v6610 : tensor<512xf32>
    %v6614 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6615 = stablehlo.multiply %v6614, %v6613 : tensor<512xf32>
    %v6616 = stablehlo.subtract %s2b1g3, %v6615 : tensor<512xf32>
    %v6617 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6618 = stablehlo.multiply %v6617, %s2b1bt3 : tensor<512xf32>
    %v6619 = stablehlo.add %v6618, %v4257 : tensor<512xf32>
    %v6620 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6621 = stablehlo.multiply %v6620, %s2b1bt3v : tensor<512xf32>
    %v6622 = stablehlo.add %v6621, %v6619 : tensor<512xf32>
    %v6623 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6624 = stablehlo.multiply %v6623, %v6622 : tensor<512xf32>
    %v6625 = stablehlo.subtract %s2b1bt3, %v6624 : tensor<512xf32>
    %v6626 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6627 = stablehlo.multiply %v6626, %s2b2W1 : tensor<128x512x1x1xf32>
    %v6628 = stablehlo.add %v6627, %v3974 : tensor<128x512x1x1xf32>
    %v6629 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6630 = stablehlo.multiply %v6629, %s2b2W1v : tensor<128x512x1x1xf32>
    %v6631 = stablehlo.add %v6630, %v6628 : tensor<128x512x1x1xf32>
    %v6632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6633 = stablehlo.multiply %v6632, %v6631 : tensor<128x512x1x1xf32>
    %v6634 = stablehlo.subtract %s2b2W1, %v6633 : tensor<128x512x1x1xf32>
    %v6635 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6636 = stablehlo.multiply %v6635, %s2b2g1 : tensor<128xf32>
    %v6637 = stablehlo.add %v6636, %v3992 : tensor<128xf32>
    %v6638 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6639 = stablehlo.multiply %v6638, %s2b2g1v : tensor<128xf32>
    %v6640 = stablehlo.add %v6639, %v6637 : tensor<128xf32>
    %v6641 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6642 = stablehlo.multiply %v6641, %v6640 : tensor<128xf32>
    %v6643 = stablehlo.subtract %s2b2g1, %v6642 : tensor<128xf32>
    %v6644 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6645 = stablehlo.multiply %v6644, %s2b2bt1 : tensor<128xf32>
    %v6646 = stablehlo.add %v6645, %v3995 : tensor<128xf32>
    %v6647 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6648 = stablehlo.multiply %v6647, %s2b2bt1v : tensor<128xf32>
    %v6649 = stablehlo.add %v6648, %v6646 : tensor<128xf32>
    %v6650 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6651 = stablehlo.multiply %v6650, %v6649 : tensor<128xf32>
    %v6652 = stablehlo.subtract %s2b2bt1, %v6651 : tensor<128xf32>
    %v6653 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6654 = stablehlo.multiply %v6653, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6655 = stablehlo.add %v6654, %v4001 : tensor<128x128x3x3xf32>
    %v6656 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6657 = stablehlo.multiply %v6656, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6658 = stablehlo.add %v6657, %v6655 : tensor<128x128x3x3xf32>
    %v6659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6660 = stablehlo.multiply %v6659, %v6658 : tensor<128x128x3x3xf32>
    %v6661 = stablehlo.subtract %s2b2W2, %v6660 : tensor<128x128x3x3xf32>
    %v6662 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6663 = stablehlo.multiply %v6662, %s2b2g2 : tensor<128xf32>
    %v6664 = stablehlo.add %v6663, %v4019 : tensor<128xf32>
    %v6665 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6666 = stablehlo.multiply %v6665, %s2b2g2v : tensor<128xf32>
    %v6667 = stablehlo.add %v6666, %v6664 : tensor<128xf32>
    %v6668 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6669 = stablehlo.multiply %v6668, %v6667 : tensor<128xf32>
    %v6670 = stablehlo.subtract %s2b2g2, %v6669 : tensor<128xf32>
    %v6671 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6672 = stablehlo.multiply %v6671, %s2b2bt2 : tensor<128xf32>
    %v6673 = stablehlo.add %v6672, %v4022 : tensor<128xf32>
    %v6674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6675 = stablehlo.multiply %v6674, %s2b2bt2v : tensor<128xf32>
    %v6676 = stablehlo.add %v6675, %v6673 : tensor<128xf32>
    %v6677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6678 = stablehlo.multiply %v6677, %v6676 : tensor<128xf32>
    %v6679 = stablehlo.subtract %s2b2bt2, %v6678 : tensor<128xf32>
    %v6680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6681 = stablehlo.multiply %v6680, %s2b2W3 : tensor<512x128x1x1xf32>
    %v6682 = stablehlo.add %v6681, %v4028 : tensor<512x128x1x1xf32>
    %v6683 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6684 = stablehlo.multiply %v6683, %s2b2W3v : tensor<512x128x1x1xf32>
    %v6685 = stablehlo.add %v6684, %v6682 : tensor<512x128x1x1xf32>
    %v6686 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6687 = stablehlo.multiply %v6686, %v6685 : tensor<512x128x1x1xf32>
    %v6688 = stablehlo.subtract %s2b2W3, %v6687 : tensor<512x128x1x1xf32>
    %v6689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6690 = stablehlo.multiply %v6689, %s2b2g3 : tensor<512xf32>
    %v6691 = stablehlo.add %v6690, %v4046 : tensor<512xf32>
    %v6692 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6693 = stablehlo.multiply %v6692, %s2b2g3v : tensor<512xf32>
    %v6694 = stablehlo.add %v6693, %v6691 : tensor<512xf32>
    %v6695 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6696 = stablehlo.multiply %v6695, %v6694 : tensor<512xf32>
    %v6697 = stablehlo.subtract %s2b2g3, %v6696 : tensor<512xf32>
    %v6698 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6699 = stablehlo.multiply %v6698, %s2b2bt3 : tensor<512xf32>
    %v6700 = stablehlo.add %v6699, %v4049 : tensor<512xf32>
    %v6701 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6702 = stablehlo.multiply %v6701, %s2b2bt3v : tensor<512xf32>
    %v6703 = stablehlo.add %v6702, %v6700 : tensor<512xf32>
    %v6704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6705 = stablehlo.multiply %v6704, %v6703 : tensor<512xf32>
    %v6706 = stablehlo.subtract %s2b2bt3, %v6705 : tensor<512xf32>
    %v6707 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6708 = stablehlo.multiply %v6707, %s2b3W1 : tensor<128x512x1x1xf32>
    %v6709 = stablehlo.add %v6708, %v3766 : tensor<128x512x1x1xf32>
    %v6710 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6711 = stablehlo.multiply %v6710, %s2b3W1v : tensor<128x512x1x1xf32>
    %v6712 = stablehlo.add %v6711, %v6709 : tensor<128x512x1x1xf32>
    %v6713 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6714 = stablehlo.multiply %v6713, %v6712 : tensor<128x512x1x1xf32>
    %v6715 = stablehlo.subtract %s2b3W1, %v6714 : tensor<128x512x1x1xf32>
    %v6716 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6717 = stablehlo.multiply %v6716, %s2b3g1 : tensor<128xf32>
    %v6718 = stablehlo.add %v6717, %v3784 : tensor<128xf32>
    %v6719 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6720 = stablehlo.multiply %v6719, %s2b3g1v : tensor<128xf32>
    %v6721 = stablehlo.add %v6720, %v6718 : tensor<128xf32>
    %v6722 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6723 = stablehlo.multiply %v6722, %v6721 : tensor<128xf32>
    %v6724 = stablehlo.subtract %s2b3g1, %v6723 : tensor<128xf32>
    %v6725 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6726 = stablehlo.multiply %v6725, %s2b3bt1 : tensor<128xf32>
    %v6727 = stablehlo.add %v6726, %v3787 : tensor<128xf32>
    %v6728 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6729 = stablehlo.multiply %v6728, %s2b3bt1v : tensor<128xf32>
    %v6730 = stablehlo.add %v6729, %v6727 : tensor<128xf32>
    %v6731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6732 = stablehlo.multiply %v6731, %v6730 : tensor<128xf32>
    %v6733 = stablehlo.subtract %s2b3bt1, %v6732 : tensor<128xf32>
    %v6734 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6735 = stablehlo.multiply %v6734, %s2b3W2 : tensor<128x128x3x3xf32>
    %v6736 = stablehlo.add %v6735, %v3793 : tensor<128x128x3x3xf32>
    %v6737 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6738 = stablehlo.multiply %v6737, %s2b3W2v : tensor<128x128x3x3xf32>
    %v6739 = stablehlo.add %v6738, %v6736 : tensor<128x128x3x3xf32>
    %v6740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6741 = stablehlo.multiply %v6740, %v6739 : tensor<128x128x3x3xf32>
    %v6742 = stablehlo.subtract %s2b3W2, %v6741 : tensor<128x128x3x3xf32>
    %v6743 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6744 = stablehlo.multiply %v6743, %s2b3g2 : tensor<128xf32>
    %v6745 = stablehlo.add %v6744, %v3811 : tensor<128xf32>
    %v6746 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6747 = stablehlo.multiply %v6746, %s2b3g2v : tensor<128xf32>
    %v6748 = stablehlo.add %v6747, %v6745 : tensor<128xf32>
    %v6749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6750 = stablehlo.multiply %v6749, %v6748 : tensor<128xf32>
    %v6751 = stablehlo.subtract %s2b3g2, %v6750 : tensor<128xf32>
    %v6752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6753 = stablehlo.multiply %v6752, %s2b3bt2 : tensor<128xf32>
    %v6754 = stablehlo.add %v6753, %v3814 : tensor<128xf32>
    %v6755 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6756 = stablehlo.multiply %v6755, %s2b3bt2v : tensor<128xf32>
    %v6757 = stablehlo.add %v6756, %v6754 : tensor<128xf32>
    %v6758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6759 = stablehlo.multiply %v6758, %v6757 : tensor<128xf32>
    %v6760 = stablehlo.subtract %s2b3bt2, %v6759 : tensor<128xf32>
    %v6761 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6762 = stablehlo.multiply %v6761, %s2b3W3 : tensor<512x128x1x1xf32>
    %v6763 = stablehlo.add %v6762, %v3820 : tensor<512x128x1x1xf32>
    %v6764 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6765 = stablehlo.multiply %v6764, %s2b3W3v : tensor<512x128x1x1xf32>
    %v6766 = stablehlo.add %v6765, %v6763 : tensor<512x128x1x1xf32>
    %v6767 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6768 = stablehlo.multiply %v6767, %v6766 : tensor<512x128x1x1xf32>
    %v6769 = stablehlo.subtract %s2b3W3, %v6768 : tensor<512x128x1x1xf32>
    %v6770 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6771 = stablehlo.multiply %v6770, %s2b3g3 : tensor<512xf32>
    %v6772 = stablehlo.add %v6771, %v3838 : tensor<512xf32>
    %v6773 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6774 = stablehlo.multiply %v6773, %s2b3g3v : tensor<512xf32>
    %v6775 = stablehlo.add %v6774, %v6772 : tensor<512xf32>
    %v6776 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6777 = stablehlo.multiply %v6776, %v6775 : tensor<512xf32>
    %v6778 = stablehlo.subtract %s2b3g3, %v6777 : tensor<512xf32>
    %v6779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6780 = stablehlo.multiply %v6779, %s2b3bt3 : tensor<512xf32>
    %v6781 = stablehlo.add %v6780, %v3841 : tensor<512xf32>
    %v6782 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6783 = stablehlo.multiply %v6782, %s2b3bt3v : tensor<512xf32>
    %v6784 = stablehlo.add %v6783, %v6781 : tensor<512xf32>
    %v6785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6786 = stablehlo.multiply %v6785, %v6784 : tensor<512xf32>
    %v6787 = stablehlo.subtract %s2b3bt3, %v6786 : tensor<512xf32>
    %v6788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6789 = stablehlo.multiply %v6788, %s3b0W1 : tensor<256x512x1x1xf32>
    %v6790 = stablehlo.add %v6789, %v3527 : tensor<256x512x1x1xf32>
    %v6791 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6792 = stablehlo.multiply %v6791, %s3b0W1v : tensor<256x512x1x1xf32>
    %v6793 = stablehlo.add %v6792, %v6790 : tensor<256x512x1x1xf32>
    %v6794 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6795 = stablehlo.multiply %v6794, %v6793 : tensor<256x512x1x1xf32>
    %v6796 = stablehlo.subtract %s3b0W1, %v6795 : tensor<256x512x1x1xf32>
    %v6797 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6798 = stablehlo.multiply %v6797, %s3b0g1 : tensor<256xf32>
    %v6799 = stablehlo.add %v6798, %v3545 : tensor<256xf32>
    %v6800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6801 = stablehlo.multiply %v6800, %s3b0g1v : tensor<256xf32>
    %v6802 = stablehlo.add %v6801, %v6799 : tensor<256xf32>
    %v6803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6804 = stablehlo.multiply %v6803, %v6802 : tensor<256xf32>
    %v6805 = stablehlo.subtract %s3b0g1, %v6804 : tensor<256xf32>
    %v6806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6807 = stablehlo.multiply %v6806, %s3b0bt1 : tensor<256xf32>
    %v6808 = stablehlo.add %v6807, %v3548 : tensor<256xf32>
    %v6809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6810 = stablehlo.multiply %v6809, %s3b0bt1v : tensor<256xf32>
    %v6811 = stablehlo.add %v6810, %v6808 : tensor<256xf32>
    %v6812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6813 = stablehlo.multiply %v6812, %v6811 : tensor<256xf32>
    %v6814 = stablehlo.subtract %s3b0bt1, %v6813 : tensor<256xf32>
    %v6815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6816 = stablehlo.multiply %v6815, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6817 = stablehlo.add %v6816, %v3556 : tensor<256x256x3x3xf32>
    %v6818 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6819 = stablehlo.multiply %v6818, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6820 = stablehlo.add %v6819, %v6817 : tensor<256x256x3x3xf32>
    %v6821 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6822 = stablehlo.multiply %v6821, %v6820 : tensor<256x256x3x3xf32>
    %v6823 = stablehlo.subtract %s3b0W2, %v6822 : tensor<256x256x3x3xf32>
    %v6824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6825 = stablehlo.multiply %v6824, %s3b0g2 : tensor<256xf32>
    %v6826 = stablehlo.add %v6825, %v3574 : tensor<256xf32>
    %v6827 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6828 = stablehlo.multiply %v6827, %s3b0g2v : tensor<256xf32>
    %v6829 = stablehlo.add %v6828, %v6826 : tensor<256xf32>
    %v6830 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6831 = stablehlo.multiply %v6830, %v6829 : tensor<256xf32>
    %v6832 = stablehlo.subtract %s3b0g2, %v6831 : tensor<256xf32>
    %v6833 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6834 = stablehlo.multiply %v6833, %s3b0bt2 : tensor<256xf32>
    %v6835 = stablehlo.add %v6834, %v3577 : tensor<256xf32>
    %v6836 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6837 = stablehlo.multiply %v6836, %s3b0bt2v : tensor<256xf32>
    %v6838 = stablehlo.add %v6837, %v6835 : tensor<256xf32>
    %v6839 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6840 = stablehlo.multiply %v6839, %v6838 : tensor<256xf32>
    %v6841 = stablehlo.subtract %s3b0bt2, %v6840 : tensor<256xf32>
    %v6842 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6843 = stablehlo.multiply %v6842, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v6844 = stablehlo.add %v6843, %v3583 : tensor<1024x256x1x1xf32>
    %v6845 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6846 = stablehlo.multiply %v6845, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v6847 = stablehlo.add %v6846, %v6844 : tensor<1024x256x1x1xf32>
    %v6848 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6849 = stablehlo.multiply %v6848, %v6847 : tensor<1024x256x1x1xf32>
    %v6850 = stablehlo.subtract %s3b0W3, %v6849 : tensor<1024x256x1x1xf32>
    %v6851 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6852 = stablehlo.multiply %v6851, %s3b0g3 : tensor<1024xf32>
    %v6853 = stablehlo.add %v6852, %v3601 : tensor<1024xf32>
    %v6854 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6855 = stablehlo.multiply %v6854, %s3b0g3v : tensor<1024xf32>
    %v6856 = stablehlo.add %v6855, %v6853 : tensor<1024xf32>
    %v6857 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6858 = stablehlo.multiply %v6857, %v6856 : tensor<1024xf32>
    %v6859 = stablehlo.subtract %s3b0g3, %v6858 : tensor<1024xf32>
    %v6860 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6861 = stablehlo.multiply %v6860, %s3b0bt3 : tensor<1024xf32>
    %v6862 = stablehlo.add %v6861, %v3604 : tensor<1024xf32>
    %v6863 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6864 = stablehlo.multiply %v6863, %s3b0bt3v : tensor<1024xf32>
    %v6865 = stablehlo.add %v6864, %v6862 : tensor<1024xf32>
    %v6866 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6867 = stablehlo.multiply %v6866, %v6865 : tensor<1024xf32>
    %v6868 = stablehlo.subtract %s3b0bt3, %v6867 : tensor<1024xf32>
    %v6869 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6870 = stablehlo.multiply %v6869, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v6871 = stablehlo.add %v6870, %v3612 : tensor<1024x512x1x1xf32>
    %v6872 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6873 = stablehlo.multiply %v6872, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v6874 = stablehlo.add %v6873, %v6871 : tensor<1024x512x1x1xf32>
    %v6875 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6876 = stablehlo.multiply %v6875, %v6874 : tensor<1024x512x1x1xf32>
    %v6877 = stablehlo.subtract %s3b0Wp, %v6876 : tensor<1024x512x1x1xf32>
    %v6878 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6879 = stablehlo.multiply %v6878, %s3b0gp : tensor<1024xf32>
    %v6880 = stablehlo.add %v6879, %v3630 : tensor<1024xf32>
    %v6881 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6882 = stablehlo.multiply %v6881, %s3b0gpv : tensor<1024xf32>
    %v6883 = stablehlo.add %v6882, %v6880 : tensor<1024xf32>
    %v6884 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6885 = stablehlo.multiply %v6884, %v6883 : tensor<1024xf32>
    %v6886 = stablehlo.subtract %s3b0gp, %v6885 : tensor<1024xf32>
    %v6887 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6888 = stablehlo.multiply %v6887, %s3b0btp : tensor<1024xf32>
    %v6889 = stablehlo.add %v6888, %v3633 : tensor<1024xf32>
    %v6890 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6891 = stablehlo.multiply %v6890, %s3b0btpv : tensor<1024xf32>
    %v6892 = stablehlo.add %v6891, %v6889 : tensor<1024xf32>
    %v6893 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6894 = stablehlo.multiply %v6893, %v6892 : tensor<1024xf32>
    %v6895 = stablehlo.subtract %s3b0btp, %v6894 : tensor<1024xf32>
    %v6896 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6897 = stablehlo.multiply %v6896, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v6898 = stablehlo.add %v6897, %v3280 : tensor<256x1024x1x1xf32>
    %v6899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6900 = stablehlo.multiply %v6899, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v6901 = stablehlo.add %v6900, %v6898 : tensor<256x1024x1x1xf32>
    %v6902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6903 = stablehlo.multiply %v6902, %v6901 : tensor<256x1024x1x1xf32>
    %v6904 = stablehlo.subtract %s3b1W1, %v6903 : tensor<256x1024x1x1xf32>
    %v6905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6906 = stablehlo.multiply %v6905, %s3b1g1 : tensor<256xf32>
    %v6907 = stablehlo.add %v6906, %v3298 : tensor<256xf32>
    %v6908 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6909 = stablehlo.multiply %v6908, %s3b1g1v : tensor<256xf32>
    %v6910 = stablehlo.add %v6909, %v6907 : tensor<256xf32>
    %v6911 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6912 = stablehlo.multiply %v6911, %v6910 : tensor<256xf32>
    %v6913 = stablehlo.subtract %s3b1g1, %v6912 : tensor<256xf32>
    %v6914 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6915 = stablehlo.multiply %v6914, %s3b1bt1 : tensor<256xf32>
    %v6916 = stablehlo.add %v6915, %v3301 : tensor<256xf32>
    %v6917 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6918 = stablehlo.multiply %v6917, %s3b1bt1v : tensor<256xf32>
    %v6919 = stablehlo.add %v6918, %v6916 : tensor<256xf32>
    %v6920 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6921 = stablehlo.multiply %v6920, %v6919 : tensor<256xf32>
    %v6922 = stablehlo.subtract %s3b1bt1, %v6921 : tensor<256xf32>
    %v6923 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6924 = stablehlo.multiply %v6923, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6925 = stablehlo.add %v6924, %v3307 : tensor<256x256x3x3xf32>
    %v6926 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6927 = stablehlo.multiply %v6926, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6928 = stablehlo.add %v6927, %v6925 : tensor<256x256x3x3xf32>
    %v6929 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6930 = stablehlo.multiply %v6929, %v6928 : tensor<256x256x3x3xf32>
    %v6931 = stablehlo.subtract %s3b1W2, %v6930 : tensor<256x256x3x3xf32>
    %v6932 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6933 = stablehlo.multiply %v6932, %s3b1g2 : tensor<256xf32>
    %v6934 = stablehlo.add %v6933, %v3325 : tensor<256xf32>
    %v6935 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6936 = stablehlo.multiply %v6935, %s3b1g2v : tensor<256xf32>
    %v6937 = stablehlo.add %v6936, %v6934 : tensor<256xf32>
    %v6938 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6939 = stablehlo.multiply %v6938, %v6937 : tensor<256xf32>
    %v6940 = stablehlo.subtract %s3b1g2, %v6939 : tensor<256xf32>
    %v6941 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6942 = stablehlo.multiply %v6941, %s3b1bt2 : tensor<256xf32>
    %v6943 = stablehlo.add %v6942, %v3328 : tensor<256xf32>
    %v6944 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6945 = stablehlo.multiply %v6944, %s3b1bt2v : tensor<256xf32>
    %v6946 = stablehlo.add %v6945, %v6943 : tensor<256xf32>
    %v6947 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6948 = stablehlo.multiply %v6947, %v6946 : tensor<256xf32>
    %v6949 = stablehlo.subtract %s3b1bt2, %v6948 : tensor<256xf32>
    %v6950 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6951 = stablehlo.multiply %v6950, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v6952 = stablehlo.add %v6951, %v3334 : tensor<1024x256x1x1xf32>
    %v6953 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6954 = stablehlo.multiply %v6953, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v6955 = stablehlo.add %v6954, %v6952 : tensor<1024x256x1x1xf32>
    %v6956 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6957 = stablehlo.multiply %v6956, %v6955 : tensor<1024x256x1x1xf32>
    %v6958 = stablehlo.subtract %s3b1W3, %v6957 : tensor<1024x256x1x1xf32>
    %v6959 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6960 = stablehlo.multiply %v6959, %s3b1g3 : tensor<1024xf32>
    %v6961 = stablehlo.add %v6960, %v3352 : tensor<1024xf32>
    %v6962 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6963 = stablehlo.multiply %v6962, %s3b1g3v : tensor<1024xf32>
    %v6964 = stablehlo.add %v6963, %v6961 : tensor<1024xf32>
    %v6965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6966 = stablehlo.multiply %v6965, %v6964 : tensor<1024xf32>
    %v6967 = stablehlo.subtract %s3b1g3, %v6966 : tensor<1024xf32>
    %v6968 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6969 = stablehlo.multiply %v6968, %s3b1bt3 : tensor<1024xf32>
    %v6970 = stablehlo.add %v6969, %v3355 : tensor<1024xf32>
    %v6971 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6972 = stablehlo.multiply %v6971, %s3b1bt3v : tensor<1024xf32>
    %v6973 = stablehlo.add %v6972, %v6970 : tensor<1024xf32>
    %v6974 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6975 = stablehlo.multiply %v6974, %v6973 : tensor<1024xf32>
    %v6976 = stablehlo.subtract %s3b1bt3, %v6975 : tensor<1024xf32>
    %v6977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6978 = stablehlo.multiply %v6977, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v6979 = stablehlo.add %v6978, %v3072 : tensor<256x1024x1x1xf32>
    %v6980 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6981 = stablehlo.multiply %v6980, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v6982 = stablehlo.add %v6981, %v6979 : tensor<256x1024x1x1xf32>
    %v6983 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6984 = stablehlo.multiply %v6983, %v6982 : tensor<256x1024x1x1xf32>
    %v6985 = stablehlo.subtract %s3b2W1, %v6984 : tensor<256x1024x1x1xf32>
    %v6986 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6987 = stablehlo.multiply %v6986, %s3b2g1 : tensor<256xf32>
    %v6988 = stablehlo.add %v6987, %v3090 : tensor<256xf32>
    %v6989 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6990 = stablehlo.multiply %v6989, %s3b2g1v : tensor<256xf32>
    %v6991 = stablehlo.add %v6990, %v6988 : tensor<256xf32>
    %v6992 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6993 = stablehlo.multiply %v6992, %v6991 : tensor<256xf32>
    %v6994 = stablehlo.subtract %s3b2g1, %v6993 : tensor<256xf32>
    %v6995 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6996 = stablehlo.multiply %v6995, %s3b2bt1 : tensor<256xf32>
    %v6997 = stablehlo.add %v6996, %v3093 : tensor<256xf32>
    %v6998 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6999 = stablehlo.multiply %v6998, %s3b2bt1v : tensor<256xf32>
    %v7000 = stablehlo.add %v6999, %v6997 : tensor<256xf32>
    %v7001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7002 = stablehlo.multiply %v7001, %v7000 : tensor<256xf32>
    %v7003 = stablehlo.subtract %s3b2bt1, %v7002 : tensor<256xf32>
    %v7004 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7005 = stablehlo.multiply %v7004, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7006 = stablehlo.add %v7005, %v3099 : tensor<256x256x3x3xf32>
    %v7007 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7008 = stablehlo.multiply %v7007, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7009 = stablehlo.add %v7008, %v7006 : tensor<256x256x3x3xf32>
    %v7010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7011 = stablehlo.multiply %v7010, %v7009 : tensor<256x256x3x3xf32>
    %v7012 = stablehlo.subtract %s3b2W2, %v7011 : tensor<256x256x3x3xf32>
    %v7013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7014 = stablehlo.multiply %v7013, %s3b2g2 : tensor<256xf32>
    %v7015 = stablehlo.add %v7014, %v3117 : tensor<256xf32>
    %v7016 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7017 = stablehlo.multiply %v7016, %s3b2g2v : tensor<256xf32>
    %v7018 = stablehlo.add %v7017, %v7015 : tensor<256xf32>
    %v7019 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7020 = stablehlo.multiply %v7019, %v7018 : tensor<256xf32>
    %v7021 = stablehlo.subtract %s3b2g2, %v7020 : tensor<256xf32>
    %v7022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7023 = stablehlo.multiply %v7022, %s3b2bt2 : tensor<256xf32>
    %v7024 = stablehlo.add %v7023, %v3120 : tensor<256xf32>
    %v7025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7026 = stablehlo.multiply %v7025, %s3b2bt2v : tensor<256xf32>
    %v7027 = stablehlo.add %v7026, %v7024 : tensor<256xf32>
    %v7028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7029 = stablehlo.multiply %v7028, %v7027 : tensor<256xf32>
    %v7030 = stablehlo.subtract %s3b2bt2, %v7029 : tensor<256xf32>
    %v7031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7032 = stablehlo.multiply %v7031, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7033 = stablehlo.add %v7032, %v3126 : tensor<1024x256x1x1xf32>
    %v7034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7035 = stablehlo.multiply %v7034, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7036 = stablehlo.add %v7035, %v7033 : tensor<1024x256x1x1xf32>
    %v7037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7038 = stablehlo.multiply %v7037, %v7036 : tensor<1024x256x1x1xf32>
    %v7039 = stablehlo.subtract %s3b2W3, %v7038 : tensor<1024x256x1x1xf32>
    %v7040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7041 = stablehlo.multiply %v7040, %s3b2g3 : tensor<1024xf32>
    %v7042 = stablehlo.add %v7041, %v3144 : tensor<1024xf32>
    %v7043 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7044 = stablehlo.multiply %v7043, %s3b2g3v : tensor<1024xf32>
    %v7045 = stablehlo.add %v7044, %v7042 : tensor<1024xf32>
    %v7046 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7047 = stablehlo.multiply %v7046, %v7045 : tensor<1024xf32>
    %v7048 = stablehlo.subtract %s3b2g3, %v7047 : tensor<1024xf32>
    %v7049 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7050 = stablehlo.multiply %v7049, %s3b2bt3 : tensor<1024xf32>
    %v7051 = stablehlo.add %v7050, %v3147 : tensor<1024xf32>
    %v7052 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7053 = stablehlo.multiply %v7052, %s3b2bt3v : tensor<1024xf32>
    %v7054 = stablehlo.add %v7053, %v7051 : tensor<1024xf32>
    %v7055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7056 = stablehlo.multiply %v7055, %v7054 : tensor<1024xf32>
    %v7057 = stablehlo.subtract %s3b2bt3, %v7056 : tensor<1024xf32>
    %v7058 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7059 = stablehlo.multiply %v7058, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7060 = stablehlo.add %v7059, %v2864 : tensor<256x1024x1x1xf32>
    %v7061 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7062 = stablehlo.multiply %v7061, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7063 = stablehlo.add %v7062, %v7060 : tensor<256x1024x1x1xf32>
    %v7064 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7065 = stablehlo.multiply %v7064, %v7063 : tensor<256x1024x1x1xf32>
    %v7066 = stablehlo.subtract %s3b3W1, %v7065 : tensor<256x1024x1x1xf32>
    %v7067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7068 = stablehlo.multiply %v7067, %s3b3g1 : tensor<256xf32>
    %v7069 = stablehlo.add %v7068, %v2882 : tensor<256xf32>
    %v7070 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7071 = stablehlo.multiply %v7070, %s3b3g1v : tensor<256xf32>
    %v7072 = stablehlo.add %v7071, %v7069 : tensor<256xf32>
    %v7073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7074 = stablehlo.multiply %v7073, %v7072 : tensor<256xf32>
    %v7075 = stablehlo.subtract %s3b3g1, %v7074 : tensor<256xf32>
    %v7076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7077 = stablehlo.multiply %v7076, %s3b3bt1 : tensor<256xf32>
    %v7078 = stablehlo.add %v7077, %v2885 : tensor<256xf32>
    %v7079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7080 = stablehlo.multiply %v7079, %s3b3bt1v : tensor<256xf32>
    %v7081 = stablehlo.add %v7080, %v7078 : tensor<256xf32>
    %v7082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7083 = stablehlo.multiply %v7082, %v7081 : tensor<256xf32>
    %v7084 = stablehlo.subtract %s3b3bt1, %v7083 : tensor<256xf32>
    %v7085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7086 = stablehlo.multiply %v7085, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7087 = stablehlo.add %v7086, %v2891 : tensor<256x256x3x3xf32>
    %v7088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7089 = stablehlo.multiply %v7088, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7090 = stablehlo.add %v7089, %v7087 : tensor<256x256x3x3xf32>
    %v7091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7092 = stablehlo.multiply %v7091, %v7090 : tensor<256x256x3x3xf32>
    %v7093 = stablehlo.subtract %s3b3W2, %v7092 : tensor<256x256x3x3xf32>
    %v7094 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7095 = stablehlo.multiply %v7094, %s3b3g2 : tensor<256xf32>
    %v7096 = stablehlo.add %v7095, %v2909 : tensor<256xf32>
    %v7097 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7098 = stablehlo.multiply %v7097, %s3b3g2v : tensor<256xf32>
    %v7099 = stablehlo.add %v7098, %v7096 : tensor<256xf32>
    %v7100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7101 = stablehlo.multiply %v7100, %v7099 : tensor<256xf32>
    %v7102 = stablehlo.subtract %s3b3g2, %v7101 : tensor<256xf32>
    %v7103 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7104 = stablehlo.multiply %v7103, %s3b3bt2 : tensor<256xf32>
    %v7105 = stablehlo.add %v7104, %v2912 : tensor<256xf32>
    %v7106 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7107 = stablehlo.multiply %v7106, %s3b3bt2v : tensor<256xf32>
    %v7108 = stablehlo.add %v7107, %v7105 : tensor<256xf32>
    %v7109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7110 = stablehlo.multiply %v7109, %v7108 : tensor<256xf32>
    %v7111 = stablehlo.subtract %s3b3bt2, %v7110 : tensor<256xf32>
    %v7112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7113 = stablehlo.multiply %v7112, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v7114 = stablehlo.add %v7113, %v2918 : tensor<1024x256x1x1xf32>
    %v7115 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7116 = stablehlo.multiply %v7115, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v7117 = stablehlo.add %v7116, %v7114 : tensor<1024x256x1x1xf32>
    %v7118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7119 = stablehlo.multiply %v7118, %v7117 : tensor<1024x256x1x1xf32>
    %v7120 = stablehlo.subtract %s3b3W3, %v7119 : tensor<1024x256x1x1xf32>
    %v7121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7122 = stablehlo.multiply %v7121, %s3b3g3 : tensor<1024xf32>
    %v7123 = stablehlo.add %v7122, %v2936 : tensor<1024xf32>
    %v7124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7125 = stablehlo.multiply %v7124, %s3b3g3v : tensor<1024xf32>
    %v7126 = stablehlo.add %v7125, %v7123 : tensor<1024xf32>
    %v7127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7128 = stablehlo.multiply %v7127, %v7126 : tensor<1024xf32>
    %v7129 = stablehlo.subtract %s3b3g3, %v7128 : tensor<1024xf32>
    %v7130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7131 = stablehlo.multiply %v7130, %s3b3bt3 : tensor<1024xf32>
    %v7132 = stablehlo.add %v7131, %v2939 : tensor<1024xf32>
    %v7133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7134 = stablehlo.multiply %v7133, %s3b3bt3v : tensor<1024xf32>
    %v7135 = stablehlo.add %v7134, %v7132 : tensor<1024xf32>
    %v7136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7137 = stablehlo.multiply %v7136, %v7135 : tensor<1024xf32>
    %v7138 = stablehlo.subtract %s3b3bt3, %v7137 : tensor<1024xf32>
    %v7139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7140 = stablehlo.multiply %v7139, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v7141 = stablehlo.add %v7140, %v2656 : tensor<256x1024x1x1xf32>
    %v7142 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7143 = stablehlo.multiply %v7142, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v7144 = stablehlo.add %v7143, %v7141 : tensor<256x1024x1x1xf32>
    %v7145 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7146 = stablehlo.multiply %v7145, %v7144 : tensor<256x1024x1x1xf32>
    %v7147 = stablehlo.subtract %s3b4W1, %v7146 : tensor<256x1024x1x1xf32>
    %v7148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7149 = stablehlo.multiply %v7148, %s3b4g1 : tensor<256xf32>
    %v7150 = stablehlo.add %v7149, %v2674 : tensor<256xf32>
    %v7151 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7152 = stablehlo.multiply %v7151, %s3b4g1v : tensor<256xf32>
    %v7153 = stablehlo.add %v7152, %v7150 : tensor<256xf32>
    %v7154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7155 = stablehlo.multiply %v7154, %v7153 : tensor<256xf32>
    %v7156 = stablehlo.subtract %s3b4g1, %v7155 : tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.multiply %v7157, %s3b4bt1 : tensor<256xf32>
    %v7159 = stablehlo.add %v7158, %v2677 : tensor<256xf32>
    %v7160 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7161 = stablehlo.multiply %v7160, %s3b4bt1v : tensor<256xf32>
    %v7162 = stablehlo.add %v7161, %v7159 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.multiply %v7163, %v7162 : tensor<256xf32>
    %v7165 = stablehlo.subtract %s3b4bt1, %v7164 : tensor<256xf32>
    %v7166 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7167 = stablehlo.multiply %v7166, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7168 = stablehlo.add %v7167, %v2683 : tensor<256x256x3x3xf32>
    %v7169 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7170 = stablehlo.multiply %v7169, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7171 = stablehlo.add %v7170, %v7168 : tensor<256x256x3x3xf32>
    %v7172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7173 = stablehlo.multiply %v7172, %v7171 : tensor<256x256x3x3xf32>
    %v7174 = stablehlo.subtract %s3b4W2, %v7173 : tensor<256x256x3x3xf32>
    %v7175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7176 = stablehlo.multiply %v7175, %s3b4g2 : tensor<256xf32>
    %v7177 = stablehlo.add %v7176, %v2701 : tensor<256xf32>
    %v7178 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7179 = stablehlo.multiply %v7178, %s3b4g2v : tensor<256xf32>
    %v7180 = stablehlo.add %v7179, %v7177 : tensor<256xf32>
    %v7181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7182 = stablehlo.multiply %v7181, %v7180 : tensor<256xf32>
    %v7183 = stablehlo.subtract %s3b4g2, %v7182 : tensor<256xf32>
    %v7184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7185 = stablehlo.multiply %v7184, %s3b4bt2 : tensor<256xf32>
    %v7186 = stablehlo.add %v7185, %v2704 : tensor<256xf32>
    %v7187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7188 = stablehlo.multiply %v7187, %s3b4bt2v : tensor<256xf32>
    %v7189 = stablehlo.add %v7188, %v7186 : tensor<256xf32>
    %v7190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7191 = stablehlo.multiply %v7190, %v7189 : tensor<256xf32>
    %v7192 = stablehlo.subtract %s3b4bt2, %v7191 : tensor<256xf32>
    %v7193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7194 = stablehlo.multiply %v7193, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v7195 = stablehlo.add %v7194, %v2710 : tensor<1024x256x1x1xf32>
    %v7196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7197 = stablehlo.multiply %v7196, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v7198 = stablehlo.add %v7197, %v7195 : tensor<1024x256x1x1xf32>
    %v7199 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7200 = stablehlo.multiply %v7199, %v7198 : tensor<1024x256x1x1xf32>
    %v7201 = stablehlo.subtract %s3b4W3, %v7200 : tensor<1024x256x1x1xf32>
    %v7202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7203 = stablehlo.multiply %v7202, %s3b4g3 : tensor<1024xf32>
    %v7204 = stablehlo.add %v7203, %v2728 : tensor<1024xf32>
    %v7205 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7206 = stablehlo.multiply %v7205, %s3b4g3v : tensor<1024xf32>
    %v7207 = stablehlo.add %v7206, %v7204 : tensor<1024xf32>
    %v7208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7209 = stablehlo.multiply %v7208, %v7207 : tensor<1024xf32>
    %v7210 = stablehlo.subtract %s3b4g3, %v7209 : tensor<1024xf32>
    %v7211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7212 = stablehlo.multiply %v7211, %s3b4bt3 : tensor<1024xf32>
    %v7213 = stablehlo.add %v7212, %v2731 : tensor<1024xf32>
    %v7214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7215 = stablehlo.multiply %v7214, %s3b4bt3v : tensor<1024xf32>
    %v7216 = stablehlo.add %v7215, %v7213 : tensor<1024xf32>
    %v7217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7218 = stablehlo.multiply %v7217, %v7216 : tensor<1024xf32>
    %v7219 = stablehlo.subtract %s3b4bt3, %v7218 : tensor<1024xf32>
    %v7220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7221 = stablehlo.multiply %v7220, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v7222 = stablehlo.add %v7221, %v2448 : tensor<256x1024x1x1xf32>
    %v7223 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7224 = stablehlo.multiply %v7223, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v7225 = stablehlo.add %v7224, %v7222 : tensor<256x1024x1x1xf32>
    %v7226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7227 = stablehlo.multiply %v7226, %v7225 : tensor<256x1024x1x1xf32>
    %v7228 = stablehlo.subtract %s3b5W1, %v7227 : tensor<256x1024x1x1xf32>
    %v7229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7230 = stablehlo.multiply %v7229, %s3b5g1 : tensor<256xf32>
    %v7231 = stablehlo.add %v7230, %v2466 : tensor<256xf32>
    %v7232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7233 = stablehlo.multiply %v7232, %s3b5g1v : tensor<256xf32>
    %v7234 = stablehlo.add %v7233, %v7231 : tensor<256xf32>
    %v7235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7236 = stablehlo.multiply %v7235, %v7234 : tensor<256xf32>
    %v7237 = stablehlo.subtract %s3b5g1, %v7236 : tensor<256xf32>
    %v7238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7239 = stablehlo.multiply %v7238, %s3b5bt1 : tensor<256xf32>
    %v7240 = stablehlo.add %v7239, %v2469 : tensor<256xf32>
    %v7241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7242 = stablehlo.multiply %v7241, %s3b5bt1v : tensor<256xf32>
    %v7243 = stablehlo.add %v7242, %v7240 : tensor<256xf32>
    %v7244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7245 = stablehlo.multiply %v7244, %v7243 : tensor<256xf32>
    %v7246 = stablehlo.subtract %s3b5bt1, %v7245 : tensor<256xf32>
    %v7247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7248 = stablehlo.multiply %v7247, %s3b5W2 : tensor<256x256x3x3xf32>
    %v7249 = stablehlo.add %v7248, %v2475 : tensor<256x256x3x3xf32>
    %v7250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7251 = stablehlo.multiply %v7250, %s3b5W2v : tensor<256x256x3x3xf32>
    %v7252 = stablehlo.add %v7251, %v7249 : tensor<256x256x3x3xf32>
    %v7253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7254 = stablehlo.multiply %v7253, %v7252 : tensor<256x256x3x3xf32>
    %v7255 = stablehlo.subtract %s3b5W2, %v7254 : tensor<256x256x3x3xf32>
    %v7256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7257 = stablehlo.multiply %v7256, %s3b5g2 : tensor<256xf32>
    %v7258 = stablehlo.add %v7257, %v2493 : tensor<256xf32>
    %v7259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7260 = stablehlo.multiply %v7259, %s3b5g2v : tensor<256xf32>
    %v7261 = stablehlo.add %v7260, %v7258 : tensor<256xf32>
    %v7262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7263 = stablehlo.multiply %v7262, %v7261 : tensor<256xf32>
    %v7264 = stablehlo.subtract %s3b5g2, %v7263 : tensor<256xf32>
    %v7265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7266 = stablehlo.multiply %v7265, %s3b5bt2 : tensor<256xf32>
    %v7267 = stablehlo.add %v7266, %v2496 : tensor<256xf32>
    %v7268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7269 = stablehlo.multiply %v7268, %s3b5bt2v : tensor<256xf32>
    %v7270 = stablehlo.add %v7269, %v7267 : tensor<256xf32>
    %v7271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7272 = stablehlo.multiply %v7271, %v7270 : tensor<256xf32>
    %v7273 = stablehlo.subtract %s3b5bt2, %v7272 : tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7275 = stablehlo.multiply %v7274, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v7276 = stablehlo.add %v7275, %v2502 : tensor<1024x256x1x1xf32>
    %v7277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7278 = stablehlo.multiply %v7277, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v7279 = stablehlo.add %v7278, %v7276 : tensor<1024x256x1x1xf32>
    %v7280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7281 = stablehlo.multiply %v7280, %v7279 : tensor<1024x256x1x1xf32>
    %v7282 = stablehlo.subtract %s3b5W3, %v7281 : tensor<1024x256x1x1xf32>
    %v7283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7284 = stablehlo.multiply %v7283, %s3b5g3 : tensor<1024xf32>
    %v7285 = stablehlo.add %v7284, %v2520 : tensor<1024xf32>
    %v7286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7287 = stablehlo.multiply %v7286, %s3b5g3v : tensor<1024xf32>
    %v7288 = stablehlo.add %v7287, %v7285 : tensor<1024xf32>
    %v7289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7290 = stablehlo.multiply %v7289, %v7288 : tensor<1024xf32>
    %v7291 = stablehlo.subtract %s3b5g3, %v7290 : tensor<1024xf32>
    %v7292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7293 = stablehlo.multiply %v7292, %s3b5bt3 : tensor<1024xf32>
    %v7294 = stablehlo.add %v7293, %v2523 : tensor<1024xf32>
    %v7295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7296 = stablehlo.multiply %v7295, %s3b5bt3v : tensor<1024xf32>
    %v7297 = stablehlo.add %v7296, %v7294 : tensor<1024xf32>
    %v7298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7299 = stablehlo.multiply %v7298, %v7297 : tensor<1024xf32>
    %v7300 = stablehlo.subtract %s3b5bt3, %v7299 : tensor<1024xf32>
    %v7301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7302 = stablehlo.multiply %v7301, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v7303 = stablehlo.add %v7302, %v2209 : tensor<512x1024x1x1xf32>
    %v7304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7305 = stablehlo.multiply %v7304, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v7306 = stablehlo.add %v7305, %v7303 : tensor<512x1024x1x1xf32>
    %v7307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7308 = stablehlo.multiply %v7307, %v7306 : tensor<512x1024x1x1xf32>
    %v7309 = stablehlo.subtract %s4b0W1, %v7308 : tensor<512x1024x1x1xf32>
    %v7310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7311 = stablehlo.multiply %v7310, %s4b0g1 : tensor<512xf32>
    %v7312 = stablehlo.add %v7311, %v2227 : tensor<512xf32>
    %v7313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7314 = stablehlo.multiply %v7313, %s4b0g1v : tensor<512xf32>
    %v7315 = stablehlo.add %v7314, %v7312 : tensor<512xf32>
    %v7316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7317 = stablehlo.multiply %v7316, %v7315 : tensor<512xf32>
    %v7318 = stablehlo.subtract %s4b0g1, %v7317 : tensor<512xf32>
    %v7319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7320 = stablehlo.multiply %v7319, %s4b0bt1 : tensor<512xf32>
    %v7321 = stablehlo.add %v7320, %v2230 : tensor<512xf32>
    %v7322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7323 = stablehlo.multiply %v7322, %s4b0bt1v : tensor<512xf32>
    %v7324 = stablehlo.add %v7323, %v7321 : tensor<512xf32>
    %v7325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7326 = stablehlo.multiply %v7325, %v7324 : tensor<512xf32>
    %v7327 = stablehlo.subtract %s4b0bt1, %v7326 : tensor<512xf32>
    %v7328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7329 = stablehlo.multiply %v7328, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7330 = stablehlo.add %v7329, %v2238 : tensor<512x512x3x3xf32>
    %v7331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7332 = stablehlo.multiply %v7331, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7333 = stablehlo.add %v7332, %v7330 : tensor<512x512x3x3xf32>
    %v7334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7335 = stablehlo.multiply %v7334, %v7333 : tensor<512x512x3x3xf32>
    %v7336 = stablehlo.subtract %s4b0W2, %v7335 : tensor<512x512x3x3xf32>
    %v7337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7338 = stablehlo.multiply %v7337, %s4b0g2 : tensor<512xf32>
    %v7339 = stablehlo.add %v7338, %v2256 : tensor<512xf32>
    %v7340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7341 = stablehlo.multiply %v7340, %s4b0g2v : tensor<512xf32>
    %v7342 = stablehlo.add %v7341, %v7339 : tensor<512xf32>
    %v7343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7344 = stablehlo.multiply %v7343, %v7342 : tensor<512xf32>
    %v7345 = stablehlo.subtract %s4b0g2, %v7344 : tensor<512xf32>
    %v7346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7347 = stablehlo.multiply %v7346, %s4b0bt2 : tensor<512xf32>
    %v7348 = stablehlo.add %v7347, %v2259 : tensor<512xf32>
    %v7349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7350 = stablehlo.multiply %v7349, %s4b0bt2v : tensor<512xf32>
    %v7351 = stablehlo.add %v7350, %v7348 : tensor<512xf32>
    %v7352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7353 = stablehlo.multiply %v7352, %v7351 : tensor<512xf32>
    %v7354 = stablehlo.subtract %s4b0bt2, %v7353 : tensor<512xf32>
    %v7355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7356 = stablehlo.multiply %v7355, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7357 = stablehlo.add %v7356, %v2265 : tensor<2048x512x1x1xf32>
    %v7358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7359 = stablehlo.multiply %v7358, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7360 = stablehlo.add %v7359, %v7357 : tensor<2048x512x1x1xf32>
    %v7361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7362 = stablehlo.multiply %v7361, %v7360 : tensor<2048x512x1x1xf32>
    %v7363 = stablehlo.subtract %s4b0W3, %v7362 : tensor<2048x512x1x1xf32>
    %v7364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7365 = stablehlo.multiply %v7364, %s4b0g3 : tensor<2048xf32>
    %v7366 = stablehlo.add %v7365, %v2283 : tensor<2048xf32>
    %v7367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7368 = stablehlo.multiply %v7367, %s4b0g3v : tensor<2048xf32>
    %v7369 = stablehlo.add %v7368, %v7366 : tensor<2048xf32>
    %v7370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7371 = stablehlo.multiply %v7370, %v7369 : tensor<2048xf32>
    %v7372 = stablehlo.subtract %s4b0g3, %v7371 : tensor<2048xf32>
    %v7373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7374 = stablehlo.multiply %v7373, %s4b0bt3 : tensor<2048xf32>
    %v7375 = stablehlo.add %v7374, %v2286 : tensor<2048xf32>
    %v7376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7377 = stablehlo.multiply %v7376, %s4b0bt3v : tensor<2048xf32>
    %v7378 = stablehlo.add %v7377, %v7375 : tensor<2048xf32>
    %v7379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7380 = stablehlo.multiply %v7379, %v7378 : tensor<2048xf32>
    %v7381 = stablehlo.subtract %s4b0bt3, %v7380 : tensor<2048xf32>
    %v7382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7383 = stablehlo.multiply %v7382, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7384 = stablehlo.add %v7383, %v2294 : tensor<2048x1024x1x1xf32>
    %v7385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7386 = stablehlo.multiply %v7385, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7387 = stablehlo.add %v7386, %v7384 : tensor<2048x1024x1x1xf32>
    %v7388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7389 = stablehlo.multiply %v7388, %v7387 : tensor<2048x1024x1x1xf32>
    %v7390 = stablehlo.subtract %s4b0Wp, %v7389 : tensor<2048x1024x1x1xf32>
    %v7391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7392 = stablehlo.multiply %v7391, %s4b0gp : tensor<2048xf32>
    %v7393 = stablehlo.add %v7392, %v2312 : tensor<2048xf32>
    %v7394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7395 = stablehlo.multiply %v7394, %s4b0gpv : tensor<2048xf32>
    %v7396 = stablehlo.add %v7395, %v7393 : tensor<2048xf32>
    %v7397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7398 = stablehlo.multiply %v7397, %v7396 : tensor<2048xf32>
    %v7399 = stablehlo.subtract %s4b0gp, %v7398 : tensor<2048xf32>
    %v7400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7401 = stablehlo.multiply %v7400, %s4b0btp : tensor<2048xf32>
    %v7402 = stablehlo.add %v7401, %v2315 : tensor<2048xf32>
    %v7403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7404 = stablehlo.multiply %v7403, %s4b0btpv : tensor<2048xf32>
    %v7405 = stablehlo.add %v7404, %v7402 : tensor<2048xf32>
    %v7406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7407 = stablehlo.multiply %v7406, %v7405 : tensor<2048xf32>
    %v7408 = stablehlo.subtract %s4b0btp, %v7407 : tensor<2048xf32>
    %v7409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7410 = stablehlo.multiply %v7409, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7411 = stablehlo.add %v7410, %v1962 : tensor<512x2048x1x1xf32>
    %v7412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7413 = stablehlo.multiply %v7412, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7414 = stablehlo.add %v7413, %v7411 : tensor<512x2048x1x1xf32>
    %v7415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7416 = stablehlo.multiply %v7415, %v7414 : tensor<512x2048x1x1xf32>
    %v7417 = stablehlo.subtract %s4b1W1, %v7416 : tensor<512x2048x1x1xf32>
    %v7418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7419 = stablehlo.multiply %v7418, %s4b1g1 : tensor<512xf32>
    %v7420 = stablehlo.add %v7419, %v1980 : tensor<512xf32>
    %v7421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7422 = stablehlo.multiply %v7421, %s4b1g1v : tensor<512xf32>
    %v7423 = stablehlo.add %v7422, %v7420 : tensor<512xf32>
    %v7424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7425 = stablehlo.multiply %v7424, %v7423 : tensor<512xf32>
    %v7426 = stablehlo.subtract %s4b1g1, %v7425 : tensor<512xf32>
    %v7427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7428 = stablehlo.multiply %v7427, %s4b1bt1 : tensor<512xf32>
    %v7429 = stablehlo.add %v7428, %v1983 : tensor<512xf32>
    %v7430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7431 = stablehlo.multiply %v7430, %s4b1bt1v : tensor<512xf32>
    %v7432 = stablehlo.add %v7431, %v7429 : tensor<512xf32>
    %v7433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7434 = stablehlo.multiply %v7433, %v7432 : tensor<512xf32>
    %v7435 = stablehlo.subtract %s4b1bt1, %v7434 : tensor<512xf32>
    %v7436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7437 = stablehlo.multiply %v7436, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7438 = stablehlo.add %v7437, %v1989 : tensor<512x512x3x3xf32>
    %v7439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7440 = stablehlo.multiply %v7439, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7441 = stablehlo.add %v7440, %v7438 : tensor<512x512x3x3xf32>
    %v7442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7443 = stablehlo.multiply %v7442, %v7441 : tensor<512x512x3x3xf32>
    %v7444 = stablehlo.subtract %s4b1W2, %v7443 : tensor<512x512x3x3xf32>
    %v7445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7446 = stablehlo.multiply %v7445, %s4b1g2 : tensor<512xf32>
    %v7447 = stablehlo.add %v7446, %v2007 : tensor<512xf32>
    %v7448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7449 = stablehlo.multiply %v7448, %s4b1g2v : tensor<512xf32>
    %v7450 = stablehlo.add %v7449, %v7447 : tensor<512xf32>
    %v7451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7452 = stablehlo.multiply %v7451, %v7450 : tensor<512xf32>
    %v7453 = stablehlo.subtract %s4b1g2, %v7452 : tensor<512xf32>
    %v7454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7455 = stablehlo.multiply %v7454, %s4b1bt2 : tensor<512xf32>
    %v7456 = stablehlo.add %v7455, %v2010 : tensor<512xf32>
    %v7457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7458 = stablehlo.multiply %v7457, %s4b1bt2v : tensor<512xf32>
    %v7459 = stablehlo.add %v7458, %v7456 : tensor<512xf32>
    %v7460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7461 = stablehlo.multiply %v7460, %v7459 : tensor<512xf32>
    %v7462 = stablehlo.subtract %s4b1bt2, %v7461 : tensor<512xf32>
    %v7463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7464 = stablehlo.multiply %v7463, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7465 = stablehlo.add %v7464, %v2016 : tensor<2048x512x1x1xf32>
    %v7466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7467 = stablehlo.multiply %v7466, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7468 = stablehlo.add %v7467, %v7465 : tensor<2048x512x1x1xf32>
    %v7469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7470 = stablehlo.multiply %v7469, %v7468 : tensor<2048x512x1x1xf32>
    %v7471 = stablehlo.subtract %s4b1W3, %v7470 : tensor<2048x512x1x1xf32>
    %v7472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7473 = stablehlo.multiply %v7472, %s4b1g3 : tensor<2048xf32>
    %v7474 = stablehlo.add %v7473, %v2034 : tensor<2048xf32>
    %v7475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7476 = stablehlo.multiply %v7475, %s4b1g3v : tensor<2048xf32>
    %v7477 = stablehlo.add %v7476, %v7474 : tensor<2048xf32>
    %v7478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7479 = stablehlo.multiply %v7478, %v7477 : tensor<2048xf32>
    %v7480 = stablehlo.subtract %s4b1g3, %v7479 : tensor<2048xf32>
    %v7481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7482 = stablehlo.multiply %v7481, %s4b1bt3 : tensor<2048xf32>
    %v7483 = stablehlo.add %v7482, %v2037 : tensor<2048xf32>
    %v7484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7485 = stablehlo.multiply %v7484, %s4b1bt3v : tensor<2048xf32>
    %v7486 = stablehlo.add %v7485, %v7483 : tensor<2048xf32>
    %v7487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7488 = stablehlo.multiply %v7487, %v7486 : tensor<2048xf32>
    %v7489 = stablehlo.subtract %s4b1bt3, %v7488 : tensor<2048xf32>
    %v7490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7491 = stablehlo.multiply %v7490, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7492 = stablehlo.add %v7491, %v1754 : tensor<512x2048x1x1xf32>
    %v7493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7494 = stablehlo.multiply %v7493, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7495 = stablehlo.add %v7494, %v7492 : tensor<512x2048x1x1xf32>
    %v7496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7497 = stablehlo.multiply %v7496, %v7495 : tensor<512x2048x1x1xf32>
    %v7498 = stablehlo.subtract %s4b2W1, %v7497 : tensor<512x2048x1x1xf32>
    %v7499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7500 = stablehlo.multiply %v7499, %s4b2g1 : tensor<512xf32>
    %v7501 = stablehlo.add %v7500, %v1772 : tensor<512xf32>
    %v7502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7503 = stablehlo.multiply %v7502, %s4b2g1v : tensor<512xf32>
    %v7504 = stablehlo.add %v7503, %v7501 : tensor<512xf32>
    %v7505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7506 = stablehlo.multiply %v7505, %v7504 : tensor<512xf32>
    %v7507 = stablehlo.subtract %s4b2g1, %v7506 : tensor<512xf32>
    %v7508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7509 = stablehlo.multiply %v7508, %s4b2bt1 : tensor<512xf32>
    %v7510 = stablehlo.add %v7509, %v1775 : tensor<512xf32>
    %v7511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7512 = stablehlo.multiply %v7511, %s4b2bt1v : tensor<512xf32>
    %v7513 = stablehlo.add %v7512, %v7510 : tensor<512xf32>
    %v7514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7515 = stablehlo.multiply %v7514, %v7513 : tensor<512xf32>
    %v7516 = stablehlo.subtract %s4b2bt1, %v7515 : tensor<512xf32>
    %v7517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7518 = stablehlo.multiply %v7517, %s4b2W2 : tensor<512x512x3x3xf32>
    %v7519 = stablehlo.add %v7518, %v1781 : tensor<512x512x3x3xf32>
    %v7520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7521 = stablehlo.multiply %v7520, %s4b2W2v : tensor<512x512x3x3xf32>
    %v7522 = stablehlo.add %v7521, %v7519 : tensor<512x512x3x3xf32>
    %v7523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7524 = stablehlo.multiply %v7523, %v7522 : tensor<512x512x3x3xf32>
    %v7525 = stablehlo.subtract %s4b2W2, %v7524 : tensor<512x512x3x3xf32>
    %v7526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7527 = stablehlo.multiply %v7526, %s4b2g2 : tensor<512xf32>
    %v7528 = stablehlo.add %v7527, %v1799 : tensor<512xf32>
    %v7529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7530 = stablehlo.multiply %v7529, %s4b2g2v : tensor<512xf32>
    %v7531 = stablehlo.add %v7530, %v7528 : tensor<512xf32>
    %v7532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7533 = stablehlo.multiply %v7532, %v7531 : tensor<512xf32>
    %v7534 = stablehlo.subtract %s4b2g2, %v7533 : tensor<512xf32>
    %v7535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7536 = stablehlo.multiply %v7535, %s4b2bt2 : tensor<512xf32>
    %v7537 = stablehlo.add %v7536, %v1802 : tensor<512xf32>
    %v7538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7539 = stablehlo.multiply %v7538, %s4b2bt2v : tensor<512xf32>
    %v7540 = stablehlo.add %v7539, %v7537 : tensor<512xf32>
    %v7541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7542 = stablehlo.multiply %v7541, %v7540 : tensor<512xf32>
    %v7543 = stablehlo.subtract %s4b2bt2, %v7542 : tensor<512xf32>
    %v7544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7545 = stablehlo.multiply %v7544, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v7546 = stablehlo.add %v7545, %v1808 : tensor<2048x512x1x1xf32>
    %v7547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7548 = stablehlo.multiply %v7547, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v7549 = stablehlo.add %v7548, %v7546 : tensor<2048x512x1x1xf32>
    %v7550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7551 = stablehlo.multiply %v7550, %v7549 : tensor<2048x512x1x1xf32>
    %v7552 = stablehlo.subtract %s4b2W3, %v7551 : tensor<2048x512x1x1xf32>
    %v7553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7554 = stablehlo.multiply %v7553, %s4b2g3 : tensor<2048xf32>
    %v7555 = stablehlo.add %v7554, %v1826 : tensor<2048xf32>
    %v7556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7557 = stablehlo.multiply %v7556, %s4b2g3v : tensor<2048xf32>
    %v7558 = stablehlo.add %v7557, %v7555 : tensor<2048xf32>
    %v7559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7560 = stablehlo.multiply %v7559, %v7558 : tensor<2048xf32>
    %v7561 = stablehlo.subtract %s4b2g3, %v7560 : tensor<2048xf32>
    %v7562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7563 = stablehlo.multiply %v7562, %s4b2bt3 : tensor<2048xf32>
    %v7564 = stablehlo.add %v7563, %v1829 : tensor<2048xf32>
    %v7565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7566 = stablehlo.multiply %v7565, %s4b2bt3v : tensor<2048xf32>
    %v7567 = stablehlo.add %v7566, %v7564 : tensor<2048xf32>
    %v7568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7569 = stablehlo.multiply %v7568, %v7567 : tensor<2048xf32>
    %v7570 = stablehlo.subtract %s4b2bt3, %v7569 : tensor<2048xf32>
    %v7571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7572 = stablehlo.multiply %v7571, %Wd : tensor<2048x1000xf32>
    %v7573 = stablehlo.add %v7572, %v1615 : tensor<2048x1000xf32>
    %v7574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7575 = stablehlo.multiply %v7574, %Wdv : tensor<2048x1000xf32>
    %v7576 = stablehlo.add %v7575, %v7573 : tensor<2048x1000xf32>
    %v7577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7578 = stablehlo.multiply %v7577, %v7576 : tensor<2048x1000xf32>
    %v7579 = stablehlo.subtract %Wd, %v7578 : tensor<2048x1000xf32>
    %v7580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7581 = stablehlo.multiply %v7580, %bd : tensor<1000xf32>
    %v7582 = stablehlo.add %v7581, %v1617 : tensor<1000xf32>
    %v7583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7584 = stablehlo.multiply %v7583, %bdv : tensor<1000xf32>
    %v7585 = stablehlo.add %v7584, %v7582 : tensor<1000xf32>
    %v7586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7587 = stablehlo.multiply %v7586, %v7585 : tensor<1000xf32>
    %v7588 = stablehlo.subtract %bd, %v7587 : tensor<1000xf32>
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
    return %v6148, %v6157, %v6166, %v6175, %v6184, %v6193, %v6202, %v6211, %v6220, %v6229, %v6238, %v6247, %v6256, %v6265, %v6274, %v6283, %v6292, %v6301, %v6310, %v6319, %v6328, %v6337, %v6346, %v6355, %v6364, %v6373, %v6382, %v6391, %v6400, %v6409, %v6418, %v6427, %v6436, %v6445, %v6454, %v6463, %v6472, %v6481, %v6490, %v6499, %v6508, %v6517, %v6526, %v6535, %v6544, %v6553, %v6562, %v6571, %v6580, %v6589, %v6598, %v6607, %v6616, %v6625, %v6634, %v6643, %v6652, %v6661, %v6670, %v6679, %v6688, %v6697, %v6706, %v6715, %v6724, %v6733, %v6742, %v6751, %v6760, %v6769, %v6778, %v6787, %v6796, %v6805, %v6814, %v6823, %v6832, %v6841, %v6850, %v6859, %v6868, %v6877, %v6886, %v6895, %v6904, %v6913, %v6922, %v6931, %v6940, %v6949, %v6958, %v6967, %v6976, %v6985, %v6994, %v7003, %v7012, %v7021, %v7030, %v7039, %v7048, %v7057, %v7066, %v7075, %v7084, %v7093, %v7102, %v7111, %v7120, %v7129, %v7138, %v7147, %v7156, %v7165, %v7174, %v7183, %v7192, %v7201, %v7210, %v7219, %v7228, %v7237, %v7246, %v7255, %v7264, %v7273, %v7282, %v7291, %v7300, %v7309, %v7318, %v7327, %v7336, %v7345, %v7354, %v7363, %v7372, %v7381, %v7390, %v7399, %v7408, %v7417, %v7426, %v7435, %v7444, %v7453, %v7462, %v7471, %v7480, %v7489, %v7498, %v7507, %v7516, %v7525, %v7534, %v7543, %v7552, %v7561, %v7570, %v7579, %v7588, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v6145, %v6154, %v6163, %v6172, %v6181, %v6190, %v6199, %v6208, %v6217, %v6226, %v6235, %v6244, %v6253, %v6262, %v6271, %v6280, %v6289, %v6298, %v6307, %v6316, %v6325, %v6334, %v6343, %v6352, %v6361, %v6370, %v6379, %v6388, %v6397, %v6406, %v6415, %v6424, %v6433, %v6442, %v6451, %v6460, %v6469, %v6478, %v6487, %v6496, %v6505, %v6514, %v6523, %v6532, %v6541, %v6550, %v6559, %v6568, %v6577, %v6586, %v6595, %v6604, %v6613, %v6622, %v6631, %v6640, %v6649, %v6658, %v6667, %v6676, %v6685, %v6694, %v6703, %v6712, %v6721, %v6730, %v6739, %v6748, %v6757, %v6766, %v6775, %v6784, %v6793, %v6802, %v6811, %v6820, %v6829, %v6838, %v6847, %v6856, %v6865, %v6874, %v6883, %v6892, %v6901, %v6910, %v6919, %v6928, %v6937, %v6946, %v6955, %v6964, %v6973, %v6982, %v6991, %v7000, %v7009, %v7018, %v7027, %v7036, %v7045, %v7054, %v7063, %v7072, %v7081, %v7090, %v7099, %v7108, %v7117, %v7126, %v7135, %v7144, %v7153, %v7162, %v7171, %v7180, %v7189, %v7198, %v7207, %v7216, %v7225, %v7234, %v7243, %v7252, %v7261, %v7270, %v7279, %v7288, %v7297, %v7306, %v7315, %v7324, %v7333, %v7342, %v7351, %v7360, %v7369, %v7378, %v7387, %v7396, %v7405, %v7414, %v7423, %v7432, %v7441, %v7450, %v7459, %v7468, %v7477, %v7486, %v7495, %v7504, %v7513, %v7522, %v7531, %v7540, %v7549, %v7558, %v7567, %v7576, %v7585, %loss, %bc1, %bc2, %v5296, %v5307, %v5312, %v5323, %v5328, %v5339, %v5344, %v5355, %v5360, %v5371, %v5376, %v5387, %v5392, %v5403, %v5408, %v5419, %v5424, %v5435, %v5440, %v5451, %v5456, %v5467, %v5472, %v5483, %v5488, %v5499, %v5504, %v5515, %v5520, %v5531, %v5536, %v5547, %v5552, %v5563, %v5568, %v5579, %v5584, %v5595, %v5600, %v5611, %v5616, %v5627, %v5632, %v5643, %v5648, %v5659, %v5664, %v5675, %v5680, %v5691, %v5696, %v5707, %v5712, %v5723, %v5728, %v5739, %v5744, %v5755, %v5760, %v5771, %v5776, %v5787, %v5792, %v5803, %v5808, %v5819, %v5824, %v5835, %v5840, %v5851, %v5856, %v5867, %v5872, %v5883, %v5888, %v5899, %v5904, %v5915, %v5920, %v5931, %v5936, %v5947, %v5952, %v5963, %v5968, %v5979, %v5984, %v5995, %v6000, %v6011, %v6016, %v6027, %v6032, %v6043, %v6048, %v6059, %v6064, %v6075, %v6080, %v6091, %v6096, %v6107, %v6112, %v6123, %v6128, %v6139 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
