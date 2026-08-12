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
    %v25 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v26 = stablehlo.maximum %v24, %v25 : tensor<256x802816xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v28 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v29 = "stablehlo.reduce_window"(%v27, %v28) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x56x56xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v32 = stablehlo.convolution(%v31, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v33 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v34 = stablehlo.add %v32, %v33 : tensor<256x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v37 = stablehlo.constant dense<0.0> : tensor<f32>
    %v38 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v39 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v40 = stablehlo.reduce(%v36 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v41 = stablehlo.broadcast_in_dim %v40, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v42 = stablehlo.divide %v41, %v38 : tensor<256x64x56x56xf32>
    %v43 = stablehlo.subtract %v36, %v42 : tensor<256x64x56x56xf32>
    %v44 = stablehlo.multiply %v43, %v43 : tensor<256x64x56x56xf32>
    %v45 = stablehlo.reduce(%v44 init: %v37) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v46 = stablehlo.broadcast_in_dim %v45, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v47 = stablehlo.divide %v46, %v38 : tensor<256x64x56x56xf32>
    %v48 = stablehlo.add %v47, %v39 : tensor<256x64x56x56xf32>
    %v49 = stablehlo.rsqrt %v48 : tensor<256x64x56x56xf32>
    %v50 = stablehlo.multiply %v43, %v49 : tensor<256x64x56x56xf32>
    %v51 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v52 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v53 = stablehlo.multiply %v50, %v51 : tensor<256x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v52 : tensor<256x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v57 = stablehlo.maximum %v55, %v56 : tensor<256x200704xf32>
    %v58 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v59 = stablehlo.convolution(%v58, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v61 = stablehlo.add %v59, %v60 : tensor<256x64x56x56xf32>
    %v62 = stablehlo.reshape %v61 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v64 = stablehlo.constant dense<0.0> : tensor<f32>
    %v65 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v66 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v67 = stablehlo.reduce(%v63 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v68 = stablehlo.broadcast_in_dim %v67, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v69 = stablehlo.divide %v68, %v65 : tensor<256x64x56x56xf32>
    %v70 = stablehlo.subtract %v63, %v69 : tensor<256x64x56x56xf32>
    %v71 = stablehlo.multiply %v70, %v70 : tensor<256x64x56x56xf32>
    %v72 = stablehlo.reduce(%v71 init: %v64) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v73 = stablehlo.broadcast_in_dim %v72, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v74 = stablehlo.divide %v73, %v65 : tensor<256x64x56x56xf32>
    %v75 = stablehlo.add %v74, %v66 : tensor<256x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<256x64x56x56xf32>
    %v77 = stablehlo.multiply %v70, %v76 : tensor<256x64x56x56xf32>
    %v78 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v80 = stablehlo.multiply %v77, %v78 : tensor<256x64x56x56xf32>
    %v81 = stablehlo.add %v80, %v79 : tensor<256x64x56x56xf32>
    %v82 = stablehlo.reshape %v81 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v83 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v84 = stablehlo.maximum %v82, %v83 : tensor<256x200704xf32>
    %v85 = stablehlo.reshape %v84 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v86 = stablehlo.convolution(%v85, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v87 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v88 = stablehlo.add %v86, %v87 : tensor<256x256x56x56xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v91 = stablehlo.constant dense<0.0> : tensor<f32>
    %v92 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v93 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v94 = stablehlo.reduce(%v90 init: %v91) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v95 = stablehlo.broadcast_in_dim %v94, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v96 = stablehlo.divide %v95, %v92 : tensor<256x256x56x56xf32>
    %v97 = stablehlo.subtract %v90, %v96 : tensor<256x256x56x56xf32>
    %v98 = stablehlo.multiply %v97, %v97 : tensor<256x256x56x56xf32>
    %v99 = stablehlo.reduce(%v98 init: %v91) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v100 = stablehlo.broadcast_in_dim %v99, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v101 = stablehlo.divide %v100, %v92 : tensor<256x256x56x56xf32>
    %v102 = stablehlo.add %v101, %v93 : tensor<256x256x56x56xf32>
    %v103 = stablehlo.rsqrt %v102 : tensor<256x256x56x56xf32>
    %v104 = stablehlo.multiply %v97, %v103 : tensor<256x256x56x56xf32>
    %v105 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v106 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v107 = stablehlo.multiply %v104, %v105 : tensor<256x256x56x56xf32>
    %v108 = stablehlo.add %v107, %v106 : tensor<256x256x56x56xf32>
    %v109 = stablehlo.reshape %v108 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v110 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v111 = stablehlo.convolution(%v110, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v112 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v113 = stablehlo.add %v111, %v112 : tensor<256x256x56x56xf32>
    %v114 = stablehlo.reshape %v113 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v115 = stablehlo.reshape %v114 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v116 = stablehlo.constant dense<0.0> : tensor<f32>
    %v117 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v118 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v119 = stablehlo.reduce(%v115 init: %v116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v120 = stablehlo.broadcast_in_dim %v119, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v121 = stablehlo.divide %v120, %v117 : tensor<256x256x56x56xf32>
    %v122 = stablehlo.subtract %v115, %v121 : tensor<256x256x56x56xf32>
    %v123 = stablehlo.multiply %v122, %v122 : tensor<256x256x56x56xf32>
    %v124 = stablehlo.reduce(%v123 init: %v116) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v125 = stablehlo.broadcast_in_dim %v124, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v126 = stablehlo.divide %v125, %v117 : tensor<256x256x56x56xf32>
    %v127 = stablehlo.add %v126, %v118 : tensor<256x256x56x56xf32>
    %v128 = stablehlo.rsqrt %v127 : tensor<256x256x56x56xf32>
    %v129 = stablehlo.multiply %v122, %v128 : tensor<256x256x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v131 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v132 = stablehlo.multiply %v129, %v130 : tensor<256x256x56x56xf32>
    %v133 = stablehlo.add %v132, %v131 : tensor<256x256x56x56xf32>
    %v134 = stablehlo.reshape %v133 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v135 = stablehlo.add %v109, %v134 : tensor<256x802816xf32>
    %v136 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v137 = stablehlo.maximum %v135, %v136 : tensor<256x802816xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v139 = stablehlo.convolution(%v138, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v140 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v141 = stablehlo.add %v139, %v140 : tensor<256x64x56x56xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v145 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v146 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v147 = stablehlo.reduce(%v143 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v148 = stablehlo.broadcast_in_dim %v147, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v149 = stablehlo.divide %v148, %v145 : tensor<256x64x56x56xf32>
    %v150 = stablehlo.subtract %v143, %v149 : tensor<256x64x56x56xf32>
    %v151 = stablehlo.multiply %v150, %v150 : tensor<256x64x56x56xf32>
    %v152 = stablehlo.reduce(%v151 init: %v144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v153 = stablehlo.broadcast_in_dim %v152, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v154 = stablehlo.divide %v153, %v145 : tensor<256x64x56x56xf32>
    %v155 = stablehlo.add %v154, %v146 : tensor<256x64x56x56xf32>
    %v156 = stablehlo.rsqrt %v155 : tensor<256x64x56x56xf32>
    %v157 = stablehlo.multiply %v150, %v156 : tensor<256x64x56x56xf32>
    %v158 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v159 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v160 = stablehlo.multiply %v157, %v158 : tensor<256x64x56x56xf32>
    %v161 = stablehlo.add %v160, %v159 : tensor<256x64x56x56xf32>
    %v162 = stablehlo.reshape %v161 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v163 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v164 = stablehlo.maximum %v162, %v163 : tensor<256x200704xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v166 = stablehlo.convolution(%v165, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v168 = stablehlo.add %v166, %v167 : tensor<256x64x56x56xf32>
    %v169 = stablehlo.reshape %v168 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v172 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v173 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v174 = stablehlo.reduce(%v170 init: %v171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v175 = stablehlo.broadcast_in_dim %v174, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v176 = stablehlo.divide %v175, %v172 : tensor<256x64x56x56xf32>
    %v177 = stablehlo.subtract %v170, %v176 : tensor<256x64x56x56xf32>
    %v178 = stablehlo.multiply %v177, %v177 : tensor<256x64x56x56xf32>
    %v179 = stablehlo.reduce(%v178 init: %v171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v180 = stablehlo.broadcast_in_dim %v179, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v181 = stablehlo.divide %v180, %v172 : tensor<256x64x56x56xf32>
    %v182 = stablehlo.add %v181, %v173 : tensor<256x64x56x56xf32>
    %v183 = stablehlo.rsqrt %v182 : tensor<256x64x56x56xf32>
    %v184 = stablehlo.multiply %v177, %v183 : tensor<256x64x56x56xf32>
    %v185 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v186 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v187 = stablehlo.multiply %v184, %v185 : tensor<256x64x56x56xf32>
    %v188 = stablehlo.add %v187, %v186 : tensor<256x64x56x56xf32>
    %v189 = stablehlo.reshape %v188 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v190 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v191 = stablehlo.maximum %v189, %v190 : tensor<256x200704xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v193 = stablehlo.convolution(%v192, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v195 = stablehlo.add %v193, %v194 : tensor<256x256x56x56xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v197 = stablehlo.reshape %v196 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v198 = stablehlo.constant dense<0.0> : tensor<f32>
    %v199 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v200 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v201 = stablehlo.reduce(%v197 init: %v198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v202 = stablehlo.broadcast_in_dim %v201, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v203 = stablehlo.divide %v202, %v199 : tensor<256x256x56x56xf32>
    %v204 = stablehlo.subtract %v197, %v203 : tensor<256x256x56x56xf32>
    %v205 = stablehlo.multiply %v204, %v204 : tensor<256x256x56x56xf32>
    %v206 = stablehlo.reduce(%v205 init: %v198) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v207 = stablehlo.broadcast_in_dim %v206, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v208 = stablehlo.divide %v207, %v199 : tensor<256x256x56x56xf32>
    %v209 = stablehlo.add %v208, %v200 : tensor<256x256x56x56xf32>
    %v210 = stablehlo.rsqrt %v209 : tensor<256x256x56x56xf32>
    %v211 = stablehlo.multiply %v204, %v210 : tensor<256x256x56x56xf32>
    %v212 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v213 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v214 = stablehlo.multiply %v211, %v212 : tensor<256x256x56x56xf32>
    %v215 = stablehlo.add %v214, %v213 : tensor<256x256x56x56xf32>
    %v216 = stablehlo.reshape %v215 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v217 = stablehlo.add %v216, %v137 : tensor<256x802816xf32>
    %v218 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v219 = stablehlo.maximum %v217, %v218 : tensor<256x802816xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v221 = stablehlo.convolution(%v220, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v222 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<256x64x56x56xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v227 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v228 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v229 = stablehlo.reduce(%v225 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v230 = stablehlo.broadcast_in_dim %v229, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v231 = stablehlo.divide %v230, %v227 : tensor<256x64x56x56xf32>
    %v232 = stablehlo.subtract %v225, %v231 : tensor<256x64x56x56xf32>
    %v233 = stablehlo.multiply %v232, %v232 : tensor<256x64x56x56xf32>
    %v234 = stablehlo.reduce(%v233 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v235 = stablehlo.broadcast_in_dim %v234, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v236 = stablehlo.divide %v235, %v227 : tensor<256x64x56x56xf32>
    %v237 = stablehlo.add %v236, %v228 : tensor<256x64x56x56xf32>
    %v238 = stablehlo.rsqrt %v237 : tensor<256x64x56x56xf32>
    %v239 = stablehlo.multiply %v232, %v238 : tensor<256x64x56x56xf32>
    %v240 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v241 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v242 = stablehlo.multiply %v239, %v240 : tensor<256x64x56x56xf32>
    %v243 = stablehlo.add %v242, %v241 : tensor<256x64x56x56xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<256x200704xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v248 = stablehlo.convolution(%v247, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<256x64x56x56xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v256 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v257 = stablehlo.broadcast_in_dim %v256, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v258 = stablehlo.divide %v257, %v254 : tensor<256x64x56x56xf32>
    %v259 = stablehlo.subtract %v252, %v258 : tensor<256x64x56x56xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<256x64x56x56xf32>
    %v261 = stablehlo.reduce(%v260 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v262 = stablehlo.broadcast_in_dim %v261, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v263 = stablehlo.divide %v262, %v254 : tensor<256x64x56x56xf32>
    %v264 = stablehlo.add %v263, %v255 : tensor<256x64x56x56xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<256x64x56x56xf32>
    %v266 = stablehlo.multiply %v259, %v265 : tensor<256x64x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v268 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<256x64x56x56xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<256x64x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v273 = stablehlo.maximum %v271, %v272 : tensor<256x200704xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v275 = stablehlo.convolution(%v274, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v276 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v277 = stablehlo.add %v275, %v276 : tensor<256x256x56x56xf32>
    %v278 = stablehlo.reshape %v277 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v279 = stablehlo.reshape %v278 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v281 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v282 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v283 = stablehlo.reduce(%v279 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v284 = stablehlo.broadcast_in_dim %v283, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v285 = stablehlo.divide %v284, %v281 : tensor<256x256x56x56xf32>
    %v286 = stablehlo.subtract %v279, %v285 : tensor<256x256x56x56xf32>
    %v287 = stablehlo.multiply %v286, %v286 : tensor<256x256x56x56xf32>
    %v288 = stablehlo.reduce(%v287 init: %v280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v289 = stablehlo.broadcast_in_dim %v288, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v290 = stablehlo.divide %v289, %v281 : tensor<256x256x56x56xf32>
    %v291 = stablehlo.add %v290, %v282 : tensor<256x256x56x56xf32>
    %v292 = stablehlo.rsqrt %v291 : tensor<256x256x56x56xf32>
    %v293 = stablehlo.multiply %v286, %v292 : tensor<256x256x56x56xf32>
    %v294 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v295 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v296 = stablehlo.multiply %v293, %v294 : tensor<256x256x56x56xf32>
    %v297 = stablehlo.add %v296, %v295 : tensor<256x256x56x56xf32>
    %v298 = stablehlo.reshape %v297 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v299 = stablehlo.add %v298, %v219 : tensor<256x802816xf32>
    %v300 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v301 = stablehlo.maximum %v299, %v300 : tensor<256x802816xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v303 = stablehlo.convolution(%v302, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<256x128x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v305 = stablehlo.add %v303, %v304 : tensor<256x128x56x56xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v307 = stablehlo.reshape %v306 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v309 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v310 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v311 = stablehlo.reduce(%v307 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v312 = stablehlo.broadcast_in_dim %v311, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v313 = stablehlo.divide %v312, %v309 : tensor<256x128x56x56xf32>
    %v314 = stablehlo.subtract %v307, %v313 : tensor<256x128x56x56xf32>
    %v315 = stablehlo.multiply %v314, %v314 : tensor<256x128x56x56xf32>
    %v316 = stablehlo.reduce(%v315 init: %v308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v318 = stablehlo.divide %v317, %v309 : tensor<256x128x56x56xf32>
    %v319 = stablehlo.add %v318, %v310 : tensor<256x128x56x56xf32>
    %v320 = stablehlo.rsqrt %v319 : tensor<256x128x56x56xf32>
    %v321 = stablehlo.multiply %v314, %v320 : tensor<256x128x56x56xf32>
    %v322 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v323 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v324 = stablehlo.multiply %v321, %v322 : tensor<256x128x56x56xf32>
    %v325 = stablehlo.add %v324, %v323 : tensor<256x128x56x56xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v327 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v328 = stablehlo.maximum %v326, %v327 : tensor<256x401408xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v330 = stablehlo.convolution(%v329, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v331 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v332 = stablehlo.add %v330, %v331 : tensor<256x128x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v334 = stablehlo.reshape %v333 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v336 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v337 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v338 = stablehlo.reduce(%v334 init: %v335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v339 = stablehlo.broadcast_in_dim %v338, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v340 = stablehlo.divide %v339, %v336 : tensor<256x128x28x28xf32>
    %v341 = stablehlo.subtract %v334, %v340 : tensor<256x128x28x28xf32>
    %v342 = stablehlo.multiply %v341, %v341 : tensor<256x128x28x28xf32>
    %v343 = stablehlo.reduce(%v342 init: %v335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v344 = stablehlo.broadcast_in_dim %v343, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v345 = stablehlo.divide %v344, %v336 : tensor<256x128x28x28xf32>
    %v346 = stablehlo.add %v345, %v337 : tensor<256x128x28x28xf32>
    %v347 = stablehlo.rsqrt %v346 : tensor<256x128x28x28xf32>
    %v348 = stablehlo.multiply %v341, %v347 : tensor<256x128x28x28xf32>
    %v349 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v350 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v351 = stablehlo.multiply %v348, %v349 : tensor<256x128x28x28xf32>
    %v352 = stablehlo.add %v351, %v350 : tensor<256x128x28x28xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v354 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v355 = stablehlo.maximum %v353, %v354 : tensor<256x100352xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v357 = stablehlo.convolution(%v356, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v358 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v359 = stablehlo.add %v357, %v358 : tensor<256x512x28x28xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v362 = stablehlo.constant dense<0.0> : tensor<f32>
    %v363 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v364 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v365 = stablehlo.reduce(%v361 init: %v362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v366 = stablehlo.broadcast_in_dim %v365, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v367 = stablehlo.divide %v366, %v363 : tensor<256x512x28x28xf32>
    %v368 = stablehlo.subtract %v361, %v367 : tensor<256x512x28x28xf32>
    %v369 = stablehlo.multiply %v368, %v368 : tensor<256x512x28x28xf32>
    %v370 = stablehlo.reduce(%v369 init: %v362) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v371 = stablehlo.broadcast_in_dim %v370, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v372 = stablehlo.divide %v371, %v363 : tensor<256x512x28x28xf32>
    %v373 = stablehlo.add %v372, %v364 : tensor<256x512x28x28xf32>
    %v374 = stablehlo.rsqrt %v373 : tensor<256x512x28x28xf32>
    %v375 = stablehlo.multiply %v368, %v374 : tensor<256x512x28x28xf32>
    %v376 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v377 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v378 = stablehlo.multiply %v375, %v376 : tensor<256x512x28x28xf32>
    %v379 = stablehlo.add %v378, %v377 : tensor<256x512x28x28xf32>
    %v380 = stablehlo.reshape %v379 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v381 = stablehlo.reshape %v301 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v382 = stablehlo.convolution(%v381, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v383 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v384 = stablehlo.add %v382, %v383 : tensor<256x512x28x28xf32>
    %v385 = stablehlo.reshape %v384 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v386 = stablehlo.reshape %v385 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v388 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v389 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v390 = stablehlo.reduce(%v386 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v391 = stablehlo.broadcast_in_dim %v390, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v392 = stablehlo.divide %v391, %v388 : tensor<256x512x28x28xf32>
    %v393 = stablehlo.subtract %v386, %v392 : tensor<256x512x28x28xf32>
    %v394 = stablehlo.multiply %v393, %v393 : tensor<256x512x28x28xf32>
    %v395 = stablehlo.reduce(%v394 init: %v387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v396 = stablehlo.broadcast_in_dim %v395, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v397 = stablehlo.divide %v396, %v388 : tensor<256x512x28x28xf32>
    %v398 = stablehlo.add %v397, %v389 : tensor<256x512x28x28xf32>
    %v399 = stablehlo.rsqrt %v398 : tensor<256x512x28x28xf32>
    %v400 = stablehlo.multiply %v393, %v399 : tensor<256x512x28x28xf32>
    %v401 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v402 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v403 = stablehlo.multiply %v400, %v401 : tensor<256x512x28x28xf32>
    %v404 = stablehlo.add %v403, %v402 : tensor<256x512x28x28xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v406 = stablehlo.add %v380, %v405 : tensor<256x401408xf32>
    %v407 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v408 = stablehlo.maximum %v406, %v407 : tensor<256x401408xf32>
    %v409 = stablehlo.reshape %v408 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v410 = stablehlo.convolution(%v409, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v412 = stablehlo.add %v410, %v411 : tensor<256x128x28x28xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v414 = stablehlo.reshape %v413 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v416 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v417 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v418 = stablehlo.reduce(%v414 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v419 = stablehlo.broadcast_in_dim %v418, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v420 = stablehlo.divide %v419, %v416 : tensor<256x128x28x28xf32>
    %v421 = stablehlo.subtract %v414, %v420 : tensor<256x128x28x28xf32>
    %v422 = stablehlo.multiply %v421, %v421 : tensor<256x128x28x28xf32>
    %v423 = stablehlo.reduce(%v422 init: %v415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v424 = stablehlo.broadcast_in_dim %v423, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v425 = stablehlo.divide %v424, %v416 : tensor<256x128x28x28xf32>
    %v426 = stablehlo.add %v425, %v417 : tensor<256x128x28x28xf32>
    %v427 = stablehlo.rsqrt %v426 : tensor<256x128x28x28xf32>
    %v428 = stablehlo.multiply %v421, %v427 : tensor<256x128x28x28xf32>
    %v429 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v430 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v431 = stablehlo.multiply %v428, %v429 : tensor<256x128x28x28xf32>
    %v432 = stablehlo.add %v431, %v430 : tensor<256x128x28x28xf32>
    %v433 = stablehlo.reshape %v432 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v434 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v435 = stablehlo.maximum %v433, %v434 : tensor<256x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v437 = stablehlo.convolution(%v436, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v438 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v439 = stablehlo.add %v437, %v438 : tensor<256x128x28x28xf32>
    %v440 = stablehlo.reshape %v439 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v441 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v443 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v444 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v445 = stablehlo.reduce(%v441 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v447 = stablehlo.divide %v446, %v443 : tensor<256x128x28x28xf32>
    %v448 = stablehlo.subtract %v441, %v447 : tensor<256x128x28x28xf32>
    %v449 = stablehlo.multiply %v448, %v448 : tensor<256x128x28x28xf32>
    %v450 = stablehlo.reduce(%v449 init: %v442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v451 = stablehlo.broadcast_in_dim %v450, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v452 = stablehlo.divide %v451, %v443 : tensor<256x128x28x28xf32>
    %v453 = stablehlo.add %v452, %v444 : tensor<256x128x28x28xf32>
    %v454 = stablehlo.rsqrt %v453 : tensor<256x128x28x28xf32>
    %v455 = stablehlo.multiply %v448, %v454 : tensor<256x128x28x28xf32>
    %v456 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v457 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v458 = stablehlo.multiply %v455, %v456 : tensor<256x128x28x28xf32>
    %v459 = stablehlo.add %v458, %v457 : tensor<256x128x28x28xf32>
    %v460 = stablehlo.reshape %v459 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v461 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v462 = stablehlo.maximum %v460, %v461 : tensor<256x100352xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v464 = stablehlo.convolution(%v463, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v465 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v466 = stablehlo.add %v464, %v465 : tensor<256x512x28x28xf32>
    %v467 = stablehlo.reshape %v466 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v470 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v471 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v472 = stablehlo.reduce(%v468 init: %v469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v473 = stablehlo.broadcast_in_dim %v472, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v474 = stablehlo.divide %v473, %v470 : tensor<256x512x28x28xf32>
    %v475 = stablehlo.subtract %v468, %v474 : tensor<256x512x28x28xf32>
    %v476 = stablehlo.multiply %v475, %v475 : tensor<256x512x28x28xf32>
    %v477 = stablehlo.reduce(%v476 init: %v469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v478 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v479 = stablehlo.divide %v478, %v470 : tensor<256x512x28x28xf32>
    %v480 = stablehlo.add %v479, %v471 : tensor<256x512x28x28xf32>
    %v481 = stablehlo.rsqrt %v480 : tensor<256x512x28x28xf32>
    %v482 = stablehlo.multiply %v475, %v481 : tensor<256x512x28x28xf32>
    %v483 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v484 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v485 = stablehlo.multiply %v482, %v483 : tensor<256x512x28x28xf32>
    %v486 = stablehlo.add %v485, %v484 : tensor<256x512x28x28xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v488 = stablehlo.add %v487, %v408 : tensor<256x401408xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<256x401408xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v492 = stablehlo.convolution(%v491, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v493 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v494 = stablehlo.add %v492, %v493 : tensor<256x128x28x28xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v498 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v499 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v500 = stablehlo.reduce(%v496 init: %v497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v501 = stablehlo.broadcast_in_dim %v500, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v502 = stablehlo.divide %v501, %v498 : tensor<256x128x28x28xf32>
    %v503 = stablehlo.subtract %v496, %v502 : tensor<256x128x28x28xf32>
    %v504 = stablehlo.multiply %v503, %v503 : tensor<256x128x28x28xf32>
    %v505 = stablehlo.reduce(%v504 init: %v497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v506 = stablehlo.broadcast_in_dim %v505, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v507 = stablehlo.divide %v506, %v498 : tensor<256x128x28x28xf32>
    %v508 = stablehlo.add %v507, %v499 : tensor<256x128x28x28xf32>
    %v509 = stablehlo.rsqrt %v508 : tensor<256x128x28x28xf32>
    %v510 = stablehlo.multiply %v503, %v509 : tensor<256x128x28x28xf32>
    %v511 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v512 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v513 = stablehlo.multiply %v510, %v511 : tensor<256x128x28x28xf32>
    %v514 = stablehlo.add %v513, %v512 : tensor<256x128x28x28xf32>
    %v515 = stablehlo.reshape %v514 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v516 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v517 = stablehlo.maximum %v515, %v516 : tensor<256x100352xf32>
    %v518 = stablehlo.reshape %v517 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v519 = stablehlo.convolution(%v518, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v520 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v521 = stablehlo.add %v519, %v520 : tensor<256x128x28x28xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v525 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v526 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v527 = stablehlo.reduce(%v523 init: %v524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v528 = stablehlo.broadcast_in_dim %v527, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v529 = stablehlo.divide %v528, %v525 : tensor<256x128x28x28xf32>
    %v530 = stablehlo.subtract %v523, %v529 : tensor<256x128x28x28xf32>
    %v531 = stablehlo.multiply %v530, %v530 : tensor<256x128x28x28xf32>
    %v532 = stablehlo.reduce(%v531 init: %v524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v533 = stablehlo.broadcast_in_dim %v532, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v534 = stablehlo.divide %v533, %v525 : tensor<256x128x28x28xf32>
    %v535 = stablehlo.add %v534, %v526 : tensor<256x128x28x28xf32>
    %v536 = stablehlo.rsqrt %v535 : tensor<256x128x28x28xf32>
    %v537 = stablehlo.multiply %v530, %v536 : tensor<256x128x28x28xf32>
    %v538 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v539 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v540 = stablehlo.multiply %v537, %v538 : tensor<256x128x28x28xf32>
    %v541 = stablehlo.add %v540, %v539 : tensor<256x128x28x28xf32>
    %v542 = stablehlo.reshape %v541 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v543 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v544 = stablehlo.maximum %v542, %v543 : tensor<256x100352xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v546 = stablehlo.convolution(%v545, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v547 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v548 = stablehlo.add %v546, %v547 : tensor<256x512x28x28xf32>
    %v549 = stablehlo.reshape %v548 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v552 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v553 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v554 = stablehlo.reduce(%v550 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v555 = stablehlo.broadcast_in_dim %v554, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v556 = stablehlo.divide %v555, %v552 : tensor<256x512x28x28xf32>
    %v557 = stablehlo.subtract %v550, %v556 : tensor<256x512x28x28xf32>
    %v558 = stablehlo.multiply %v557, %v557 : tensor<256x512x28x28xf32>
    %v559 = stablehlo.reduce(%v558 init: %v551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v560 = stablehlo.broadcast_in_dim %v559, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v561 = stablehlo.divide %v560, %v552 : tensor<256x512x28x28xf32>
    %v562 = stablehlo.add %v561, %v553 : tensor<256x512x28x28xf32>
    %v563 = stablehlo.rsqrt %v562 : tensor<256x512x28x28xf32>
    %v564 = stablehlo.multiply %v557, %v563 : tensor<256x512x28x28xf32>
    %v565 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v566 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v567 = stablehlo.multiply %v564, %v565 : tensor<256x512x28x28xf32>
    %v568 = stablehlo.add %v567, %v566 : tensor<256x512x28x28xf32>
    %v569 = stablehlo.reshape %v568 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v570 = stablehlo.add %v569, %v490 : tensor<256x401408xf32>
    %v571 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v572 = stablehlo.maximum %v570, %v571 : tensor<256x401408xf32>
    %v573 = stablehlo.reshape %v572 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v574 = stablehlo.convolution(%v573, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v575 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<256x128x28x28xf32>
    %v577 = stablehlo.reshape %v576 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v580 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v581 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v582 = stablehlo.reduce(%v578 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v583 = stablehlo.broadcast_in_dim %v582, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v584 = stablehlo.divide %v583, %v580 : tensor<256x128x28x28xf32>
    %v585 = stablehlo.subtract %v578, %v584 : tensor<256x128x28x28xf32>
    %v586 = stablehlo.multiply %v585, %v585 : tensor<256x128x28x28xf32>
    %v587 = stablehlo.reduce(%v586 init: %v579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v589 = stablehlo.divide %v588, %v580 : tensor<256x128x28x28xf32>
    %v590 = stablehlo.add %v589, %v581 : tensor<256x128x28x28xf32>
    %v591 = stablehlo.rsqrt %v590 : tensor<256x128x28x28xf32>
    %v592 = stablehlo.multiply %v585, %v591 : tensor<256x128x28x28xf32>
    %v593 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v594 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v595 = stablehlo.multiply %v592, %v593 : tensor<256x128x28x28xf32>
    %v596 = stablehlo.add %v595, %v594 : tensor<256x128x28x28xf32>
    %v597 = stablehlo.reshape %v596 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v598 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v599 = stablehlo.maximum %v597, %v598 : tensor<256x100352xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v601 = stablehlo.convolution(%v600, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v602 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v603 = stablehlo.add %v601, %v602 : tensor<256x128x28x28xf32>
    %v604 = stablehlo.reshape %v603 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v607 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v608 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v609 = stablehlo.reduce(%v605 init: %v606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v610 = stablehlo.broadcast_in_dim %v609, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v611 = stablehlo.divide %v610, %v607 : tensor<256x128x28x28xf32>
    %v612 = stablehlo.subtract %v605, %v611 : tensor<256x128x28x28xf32>
    %v613 = stablehlo.multiply %v612, %v612 : tensor<256x128x28x28xf32>
    %v614 = stablehlo.reduce(%v613 init: %v606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v615 = stablehlo.broadcast_in_dim %v614, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v616 = stablehlo.divide %v615, %v607 : tensor<256x128x28x28xf32>
    %v617 = stablehlo.add %v616, %v608 : tensor<256x128x28x28xf32>
    %v618 = stablehlo.rsqrt %v617 : tensor<256x128x28x28xf32>
    %v619 = stablehlo.multiply %v612, %v618 : tensor<256x128x28x28xf32>
    %v620 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v621 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v622 = stablehlo.multiply %v619, %v620 : tensor<256x128x28x28xf32>
    %v623 = stablehlo.add %v622, %v621 : tensor<256x128x28x28xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v625 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v626 = stablehlo.maximum %v624, %v625 : tensor<256x100352xf32>
    %v627 = stablehlo.reshape %v626 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v628 = stablehlo.convolution(%v627, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v629 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<256x512x28x28xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v634 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v635 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v636 = stablehlo.reduce(%v632 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v637 = stablehlo.broadcast_in_dim %v636, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v638 = stablehlo.divide %v637, %v634 : tensor<256x512x28x28xf32>
    %v639 = stablehlo.subtract %v632, %v638 : tensor<256x512x28x28xf32>
    %v640 = stablehlo.multiply %v639, %v639 : tensor<256x512x28x28xf32>
    %v641 = stablehlo.reduce(%v640 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v642 = stablehlo.broadcast_in_dim %v641, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v643 = stablehlo.divide %v642, %v634 : tensor<256x512x28x28xf32>
    %v644 = stablehlo.add %v643, %v635 : tensor<256x512x28x28xf32>
    %v645 = stablehlo.rsqrt %v644 : tensor<256x512x28x28xf32>
    %v646 = stablehlo.multiply %v639, %v645 : tensor<256x512x28x28xf32>
    %v647 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v648 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v649 = stablehlo.multiply %v646, %v647 : tensor<256x512x28x28xf32>
    %v650 = stablehlo.add %v649, %v648 : tensor<256x512x28x28xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v652 = stablehlo.add %v651, %v572 : tensor<256x401408xf32>
    %v653 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v654 = stablehlo.maximum %v652, %v653 : tensor<256x401408xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v656 = stablehlo.convolution(%v655, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x28x28xf32>
    %v657 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v658 = stablehlo.add %v656, %v657 : tensor<256x256x28x28xf32>
    %v659 = stablehlo.reshape %v658 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v660 = stablehlo.reshape %v659 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v662 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v663 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v664 = stablehlo.reduce(%v660 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v665 = stablehlo.broadcast_in_dim %v664, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v666 = stablehlo.divide %v665, %v662 : tensor<256x256x28x28xf32>
    %v667 = stablehlo.subtract %v660, %v666 : tensor<256x256x28x28xf32>
    %v668 = stablehlo.multiply %v667, %v667 : tensor<256x256x28x28xf32>
    %v669 = stablehlo.reduce(%v668 init: %v661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v670 = stablehlo.broadcast_in_dim %v669, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v671 = stablehlo.divide %v670, %v662 : tensor<256x256x28x28xf32>
    %v672 = stablehlo.add %v671, %v663 : tensor<256x256x28x28xf32>
    %v673 = stablehlo.rsqrt %v672 : tensor<256x256x28x28xf32>
    %v674 = stablehlo.multiply %v667, %v673 : tensor<256x256x28x28xf32>
    %v675 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v676 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v677 = stablehlo.multiply %v674, %v675 : tensor<256x256x28x28xf32>
    %v678 = stablehlo.add %v677, %v676 : tensor<256x256x28x28xf32>
    %v679 = stablehlo.reshape %v678 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v680 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v681 = stablehlo.maximum %v679, %v680 : tensor<256x200704xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v683 = stablehlo.convolution(%v682, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v684 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v685 = stablehlo.add %v683, %v684 : tensor<256x256x14x14xf32>
    %v686 = stablehlo.reshape %v685 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v689 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v690 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v691 = stablehlo.reduce(%v687 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v692 = stablehlo.broadcast_in_dim %v691, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v693 = stablehlo.divide %v692, %v689 : tensor<256x256x14x14xf32>
    %v694 = stablehlo.subtract %v687, %v693 : tensor<256x256x14x14xf32>
    %v695 = stablehlo.multiply %v694, %v694 : tensor<256x256x14x14xf32>
    %v696 = stablehlo.reduce(%v695 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v697 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v698 = stablehlo.divide %v697, %v689 : tensor<256x256x14x14xf32>
    %v699 = stablehlo.add %v698, %v690 : tensor<256x256x14x14xf32>
    %v700 = stablehlo.rsqrt %v699 : tensor<256x256x14x14xf32>
    %v701 = stablehlo.multiply %v694, %v700 : tensor<256x256x14x14xf32>
    %v702 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v703 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v704 = stablehlo.multiply %v701, %v702 : tensor<256x256x14x14xf32>
    %v705 = stablehlo.add %v704, %v703 : tensor<256x256x14x14xf32>
    %v706 = stablehlo.reshape %v705 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v707 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v708 = stablehlo.maximum %v706, %v707 : tensor<256x50176xf32>
    %v709 = stablehlo.reshape %v708 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v710 = stablehlo.convolution(%v709, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v711 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v712 = stablehlo.add %v710, %v711 : tensor<256x1024x14x14xf32>
    %v713 = stablehlo.reshape %v712 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v714 = stablehlo.reshape %v713 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v716 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v717 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v718 = stablehlo.reduce(%v714 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v719 = stablehlo.broadcast_in_dim %v718, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v720 = stablehlo.divide %v719, %v716 : tensor<256x1024x14x14xf32>
    %v721 = stablehlo.subtract %v714, %v720 : tensor<256x1024x14x14xf32>
    %v722 = stablehlo.multiply %v721, %v721 : tensor<256x1024x14x14xf32>
    %v723 = stablehlo.reduce(%v722 init: %v715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v724 = stablehlo.broadcast_in_dim %v723, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v725 = stablehlo.divide %v724, %v716 : tensor<256x1024x14x14xf32>
    %v726 = stablehlo.add %v725, %v717 : tensor<256x1024x14x14xf32>
    %v727 = stablehlo.rsqrt %v726 : tensor<256x1024x14x14xf32>
    %v728 = stablehlo.multiply %v721, %v727 : tensor<256x1024x14x14xf32>
    %v729 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v730 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v731 = stablehlo.multiply %v728, %v729 : tensor<256x1024x14x14xf32>
    %v732 = stablehlo.add %v731, %v730 : tensor<256x1024x14x14xf32>
    %v733 = stablehlo.reshape %v732 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v734 = stablehlo.reshape %v654 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v735 = stablehlo.convolution(%v734, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v736 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<256x1024x14x14xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<f32>
    %v741 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v742 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v743 = stablehlo.reduce(%v739 init: %v740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v744 = stablehlo.broadcast_in_dim %v743, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v745 = stablehlo.divide %v744, %v741 : tensor<256x1024x14x14xf32>
    %v746 = stablehlo.subtract %v739, %v745 : tensor<256x1024x14x14xf32>
    %v747 = stablehlo.multiply %v746, %v746 : tensor<256x1024x14x14xf32>
    %v748 = stablehlo.reduce(%v747 init: %v740) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v749 = stablehlo.broadcast_in_dim %v748, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v750 = stablehlo.divide %v749, %v741 : tensor<256x1024x14x14xf32>
    %v751 = stablehlo.add %v750, %v742 : tensor<256x1024x14x14xf32>
    %v752 = stablehlo.rsqrt %v751 : tensor<256x1024x14x14xf32>
    %v753 = stablehlo.multiply %v746, %v752 : tensor<256x1024x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v755 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v756 = stablehlo.multiply %v753, %v754 : tensor<256x1024x14x14xf32>
    %v757 = stablehlo.add %v756, %v755 : tensor<256x1024x14x14xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v759 = stablehlo.add %v733, %v758 : tensor<256x200704xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v761 = stablehlo.maximum %v759, %v760 : tensor<256x200704xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v763 = stablehlo.convolution(%v762, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v764 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v765 = stablehlo.add %v763, %v764 : tensor<256x256x14x14xf32>
    %v766 = stablehlo.reshape %v765 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v767 = stablehlo.reshape %v766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v769 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v770 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v771 = stablehlo.reduce(%v767 init: %v768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v772 = stablehlo.broadcast_in_dim %v771, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v773 = stablehlo.divide %v772, %v769 : tensor<256x256x14x14xf32>
    %v774 = stablehlo.subtract %v767, %v773 : tensor<256x256x14x14xf32>
    %v775 = stablehlo.multiply %v774, %v774 : tensor<256x256x14x14xf32>
    %v776 = stablehlo.reduce(%v775 init: %v768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v777 = stablehlo.broadcast_in_dim %v776, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v778 = stablehlo.divide %v777, %v769 : tensor<256x256x14x14xf32>
    %v779 = stablehlo.add %v778, %v770 : tensor<256x256x14x14xf32>
    %v780 = stablehlo.rsqrt %v779 : tensor<256x256x14x14xf32>
    %v781 = stablehlo.multiply %v774, %v780 : tensor<256x256x14x14xf32>
    %v782 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v783 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v784 = stablehlo.multiply %v781, %v782 : tensor<256x256x14x14xf32>
    %v785 = stablehlo.add %v784, %v783 : tensor<256x256x14x14xf32>
    %v786 = stablehlo.reshape %v785 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v787 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v788 = stablehlo.maximum %v786, %v787 : tensor<256x50176xf32>
    %v789 = stablehlo.reshape %v788 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v790 = stablehlo.convolution(%v789, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v791 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v792 = stablehlo.add %v790, %v791 : tensor<256x256x14x14xf32>
    %v793 = stablehlo.reshape %v792 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v795 = stablehlo.constant dense<0.0> : tensor<f32>
    %v796 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v797 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v798 = stablehlo.reduce(%v794 init: %v795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v799 = stablehlo.broadcast_in_dim %v798, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v800 = stablehlo.divide %v799, %v796 : tensor<256x256x14x14xf32>
    %v801 = stablehlo.subtract %v794, %v800 : tensor<256x256x14x14xf32>
    %v802 = stablehlo.multiply %v801, %v801 : tensor<256x256x14x14xf32>
    %v803 = stablehlo.reduce(%v802 init: %v795) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v796 : tensor<256x256x14x14xf32>
    %v806 = stablehlo.add %v805, %v797 : tensor<256x256x14x14xf32>
    %v807 = stablehlo.rsqrt %v806 : tensor<256x256x14x14xf32>
    %v808 = stablehlo.multiply %v801, %v807 : tensor<256x256x14x14xf32>
    %v809 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v811 = stablehlo.multiply %v808, %v809 : tensor<256x256x14x14xf32>
    %v812 = stablehlo.add %v811, %v810 : tensor<256x256x14x14xf32>
    %v813 = stablehlo.reshape %v812 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v814 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v815 = stablehlo.maximum %v813, %v814 : tensor<256x50176xf32>
    %v816 = stablehlo.reshape %v815 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v817 = stablehlo.convolution(%v816, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v819 = stablehlo.add %v817, %v818 : tensor<256x1024x14x14xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v821 = stablehlo.reshape %v820 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v823 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v824 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v825 = stablehlo.reduce(%v821 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v826 = stablehlo.broadcast_in_dim %v825, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v827 = stablehlo.divide %v826, %v823 : tensor<256x1024x14x14xf32>
    %v828 = stablehlo.subtract %v821, %v827 : tensor<256x1024x14x14xf32>
    %v829 = stablehlo.multiply %v828, %v828 : tensor<256x1024x14x14xf32>
    %v830 = stablehlo.reduce(%v829 init: %v822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v831 = stablehlo.broadcast_in_dim %v830, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v832 = stablehlo.divide %v831, %v823 : tensor<256x1024x14x14xf32>
    %v833 = stablehlo.add %v832, %v824 : tensor<256x1024x14x14xf32>
    %v834 = stablehlo.rsqrt %v833 : tensor<256x1024x14x14xf32>
    %v835 = stablehlo.multiply %v828, %v834 : tensor<256x1024x14x14xf32>
    %v836 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v837 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v838 = stablehlo.multiply %v835, %v836 : tensor<256x1024x14x14xf32>
    %v839 = stablehlo.add %v838, %v837 : tensor<256x1024x14x14xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v841 = stablehlo.add %v840, %v761 : tensor<256x200704xf32>
    %v842 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v843 = stablehlo.maximum %v841, %v842 : tensor<256x200704xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v845 = stablehlo.convolution(%v844, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v847 = stablehlo.add %v845, %v846 : tensor<256x256x14x14xf32>
    %v848 = stablehlo.reshape %v847 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v851 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v852 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v853 = stablehlo.reduce(%v849 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v854 = stablehlo.broadcast_in_dim %v853, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v855 = stablehlo.divide %v854, %v851 : tensor<256x256x14x14xf32>
    %v856 = stablehlo.subtract %v849, %v855 : tensor<256x256x14x14xf32>
    %v857 = stablehlo.multiply %v856, %v856 : tensor<256x256x14x14xf32>
    %v858 = stablehlo.reduce(%v857 init: %v850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v859 = stablehlo.broadcast_in_dim %v858, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v860 = stablehlo.divide %v859, %v851 : tensor<256x256x14x14xf32>
    %v861 = stablehlo.add %v860, %v852 : tensor<256x256x14x14xf32>
    %v862 = stablehlo.rsqrt %v861 : tensor<256x256x14x14xf32>
    %v863 = stablehlo.multiply %v856, %v862 : tensor<256x256x14x14xf32>
    %v864 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v865 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v866 = stablehlo.multiply %v863, %v864 : tensor<256x256x14x14xf32>
    %v867 = stablehlo.add %v866, %v865 : tensor<256x256x14x14xf32>
    %v868 = stablehlo.reshape %v867 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v869 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v870 = stablehlo.maximum %v868, %v869 : tensor<256x50176xf32>
    %v871 = stablehlo.reshape %v870 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v872 = stablehlo.convolution(%v871, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v873 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v874 = stablehlo.add %v872, %v873 : tensor<256x256x14x14xf32>
    %v875 = stablehlo.reshape %v874 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v878 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v879 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v880 = stablehlo.reduce(%v876 init: %v877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v881 = stablehlo.broadcast_in_dim %v880, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v882 = stablehlo.divide %v881, %v878 : tensor<256x256x14x14xf32>
    %v883 = stablehlo.subtract %v876, %v882 : tensor<256x256x14x14xf32>
    %v884 = stablehlo.multiply %v883, %v883 : tensor<256x256x14x14xf32>
    %v885 = stablehlo.reduce(%v884 init: %v877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v886 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v887 = stablehlo.divide %v886, %v878 : tensor<256x256x14x14xf32>
    %v888 = stablehlo.add %v887, %v879 : tensor<256x256x14x14xf32>
    %v889 = stablehlo.rsqrt %v888 : tensor<256x256x14x14xf32>
    %v890 = stablehlo.multiply %v883, %v889 : tensor<256x256x14x14xf32>
    %v891 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v892 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v893 = stablehlo.multiply %v890, %v891 : tensor<256x256x14x14xf32>
    %v894 = stablehlo.add %v893, %v892 : tensor<256x256x14x14xf32>
    %v895 = stablehlo.reshape %v894 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v896 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v897 = stablehlo.maximum %v895, %v896 : tensor<256x50176xf32>
    %v898 = stablehlo.reshape %v897 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v899 = stablehlo.convolution(%v898, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v900 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v901 = stablehlo.add %v899, %v900 : tensor<256x1024x14x14xf32>
    %v902 = stablehlo.reshape %v901 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v905 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v906 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v907 = stablehlo.reduce(%v903 init: %v904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v908 = stablehlo.broadcast_in_dim %v907, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v909 = stablehlo.divide %v908, %v905 : tensor<256x1024x14x14xf32>
    %v910 = stablehlo.subtract %v903, %v909 : tensor<256x1024x14x14xf32>
    %v911 = stablehlo.multiply %v910, %v910 : tensor<256x1024x14x14xf32>
    %v912 = stablehlo.reduce(%v911 init: %v904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v913 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v914 = stablehlo.divide %v913, %v905 : tensor<256x1024x14x14xf32>
    %v915 = stablehlo.add %v914, %v906 : tensor<256x1024x14x14xf32>
    %v916 = stablehlo.rsqrt %v915 : tensor<256x1024x14x14xf32>
    %v917 = stablehlo.multiply %v910, %v916 : tensor<256x1024x14x14xf32>
    %v918 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v919 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v920 = stablehlo.multiply %v917, %v918 : tensor<256x1024x14x14xf32>
    %v921 = stablehlo.add %v920, %v919 : tensor<256x1024x14x14xf32>
    %v922 = stablehlo.reshape %v921 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v923 = stablehlo.add %v922, %v843 : tensor<256x200704xf32>
    %v924 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v925 = stablehlo.maximum %v923, %v924 : tensor<256x200704xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v927 = stablehlo.convolution(%v926, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v928 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v929 = stablehlo.add %v927, %v928 : tensor<256x256x14x14xf32>
    %v930 = stablehlo.reshape %v929 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v931 = stablehlo.reshape %v930 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v933 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v934 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v935 = stablehlo.reduce(%v931 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v936 = stablehlo.broadcast_in_dim %v935, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v937 = stablehlo.divide %v936, %v933 : tensor<256x256x14x14xf32>
    %v938 = stablehlo.subtract %v931, %v937 : tensor<256x256x14x14xf32>
    %v939 = stablehlo.multiply %v938, %v938 : tensor<256x256x14x14xf32>
    %v940 = stablehlo.reduce(%v939 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v941 = stablehlo.broadcast_in_dim %v940, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v942 = stablehlo.divide %v941, %v933 : tensor<256x256x14x14xf32>
    %v943 = stablehlo.add %v942, %v934 : tensor<256x256x14x14xf32>
    %v944 = stablehlo.rsqrt %v943 : tensor<256x256x14x14xf32>
    %v945 = stablehlo.multiply %v938, %v944 : tensor<256x256x14x14xf32>
    %v946 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v947 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v948 = stablehlo.multiply %v945, %v946 : tensor<256x256x14x14xf32>
    %v949 = stablehlo.add %v948, %v947 : tensor<256x256x14x14xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v951 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v952 = stablehlo.maximum %v950, %v951 : tensor<256x50176xf32>
    %v953 = stablehlo.reshape %v952 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v954 = stablehlo.convolution(%v953, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v956 = stablehlo.add %v954, %v955 : tensor<256x256x14x14xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v960 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v961 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v962 = stablehlo.reduce(%v958 init: %v959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v963 = stablehlo.broadcast_in_dim %v962, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v964 = stablehlo.divide %v963, %v960 : tensor<256x256x14x14xf32>
    %v965 = stablehlo.subtract %v958, %v964 : tensor<256x256x14x14xf32>
    %v966 = stablehlo.multiply %v965, %v965 : tensor<256x256x14x14xf32>
    %v967 = stablehlo.reduce(%v966 init: %v959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v969 = stablehlo.divide %v968, %v960 : tensor<256x256x14x14xf32>
    %v970 = stablehlo.add %v969, %v961 : tensor<256x256x14x14xf32>
    %v971 = stablehlo.rsqrt %v970 : tensor<256x256x14x14xf32>
    %v972 = stablehlo.multiply %v965, %v971 : tensor<256x256x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v974 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v975 = stablehlo.multiply %v972, %v973 : tensor<256x256x14x14xf32>
    %v976 = stablehlo.add %v975, %v974 : tensor<256x256x14x14xf32>
    %v977 = stablehlo.reshape %v976 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v978 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v979 = stablehlo.maximum %v977, %v978 : tensor<256x50176xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v981 = stablehlo.convolution(%v980, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v982 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v983 = stablehlo.add %v981, %v982 : tensor<256x1024x14x14xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v985 = stablehlo.reshape %v984 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v987 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v988 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v989 = stablehlo.reduce(%v985 init: %v986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v990 = stablehlo.broadcast_in_dim %v989, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v991 = stablehlo.divide %v990, %v987 : tensor<256x1024x14x14xf32>
    %v992 = stablehlo.subtract %v985, %v991 : tensor<256x1024x14x14xf32>
    %v993 = stablehlo.multiply %v992, %v992 : tensor<256x1024x14x14xf32>
    %v994 = stablehlo.reduce(%v993 init: %v986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v995 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v996 = stablehlo.divide %v995, %v987 : tensor<256x1024x14x14xf32>
    %v997 = stablehlo.add %v996, %v988 : tensor<256x1024x14x14xf32>
    %v998 = stablehlo.rsqrt %v997 : tensor<256x1024x14x14xf32>
    %v999 = stablehlo.multiply %v992, %v998 : tensor<256x1024x14x14xf32>
    %v1000 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1001 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1002 = stablehlo.multiply %v999, %v1000 : tensor<256x1024x14x14xf32>
    %v1003 = stablehlo.add %v1002, %v1001 : tensor<256x1024x14x14xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1005 = stablehlo.add %v1004, %v925 : tensor<256x200704xf32>
    %v1006 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v1007 = stablehlo.maximum %v1005, %v1006 : tensor<256x200704xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1009 = stablehlo.convolution(%v1008, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1010 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1011 = stablehlo.add %v1009, %v1010 : tensor<256x256x14x14xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1013 = stablehlo.reshape %v1012 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1015 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1016 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1017 = stablehlo.reduce(%v1013 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1018 = stablehlo.broadcast_in_dim %v1017, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1019 = stablehlo.divide %v1018, %v1015 : tensor<256x256x14x14xf32>
    %v1020 = stablehlo.subtract %v1013, %v1019 : tensor<256x256x14x14xf32>
    %v1021 = stablehlo.multiply %v1020, %v1020 : tensor<256x256x14x14xf32>
    %v1022 = stablehlo.reduce(%v1021 init: %v1014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1023 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1024 = stablehlo.divide %v1023, %v1015 : tensor<256x256x14x14xf32>
    %v1025 = stablehlo.add %v1024, %v1016 : tensor<256x256x14x14xf32>
    %v1026 = stablehlo.rsqrt %v1025 : tensor<256x256x14x14xf32>
    %v1027 = stablehlo.multiply %v1020, %v1026 : tensor<256x256x14x14xf32>
    %v1028 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1029 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1030 = stablehlo.multiply %v1027, %v1028 : tensor<256x256x14x14xf32>
    %v1031 = stablehlo.add %v1030, %v1029 : tensor<256x256x14x14xf32>
    %v1032 = stablehlo.reshape %v1031 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1033 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1034 = stablehlo.maximum %v1032, %v1033 : tensor<256x50176xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1036 = stablehlo.convolution(%v1035, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1037 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1038 = stablehlo.add %v1036, %v1037 : tensor<256x256x14x14xf32>
    %v1039 = stablehlo.reshape %v1038 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1042 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1043 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1044 = stablehlo.reduce(%v1040 init: %v1041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1045 = stablehlo.broadcast_in_dim %v1044, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1046 = stablehlo.divide %v1045, %v1042 : tensor<256x256x14x14xf32>
    %v1047 = stablehlo.subtract %v1040, %v1046 : tensor<256x256x14x14xf32>
    %v1048 = stablehlo.multiply %v1047, %v1047 : tensor<256x256x14x14xf32>
    %v1049 = stablehlo.reduce(%v1048 init: %v1041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1050 = stablehlo.broadcast_in_dim %v1049, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1051 = stablehlo.divide %v1050, %v1042 : tensor<256x256x14x14xf32>
    %v1052 = stablehlo.add %v1051, %v1043 : tensor<256x256x14x14xf32>
    %v1053 = stablehlo.rsqrt %v1052 : tensor<256x256x14x14xf32>
    %v1054 = stablehlo.multiply %v1047, %v1053 : tensor<256x256x14x14xf32>
    %v1055 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1056 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1057 = stablehlo.multiply %v1054, %v1055 : tensor<256x256x14x14xf32>
    %v1058 = stablehlo.add %v1057, %v1056 : tensor<256x256x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1060 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1061 = stablehlo.maximum %v1059, %v1060 : tensor<256x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1063 = stablehlo.convolution(%v1062, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1064 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1065 = stablehlo.add %v1063, %v1064 : tensor<256x1024x14x14xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1067 = stablehlo.reshape %v1066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v1070 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v1071 = stablehlo.reduce(%v1067 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1073 = stablehlo.divide %v1072, %v1069 : tensor<256x1024x14x14xf32>
    %v1074 = stablehlo.subtract %v1067, %v1073 : tensor<256x1024x14x14xf32>
    %v1075 = stablehlo.multiply %v1074, %v1074 : tensor<256x1024x14x14xf32>
    %v1076 = stablehlo.reduce(%v1075 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1077 = stablehlo.broadcast_in_dim %v1076, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1078 = stablehlo.divide %v1077, %v1069 : tensor<256x1024x14x14xf32>
    %v1079 = stablehlo.add %v1078, %v1070 : tensor<256x1024x14x14xf32>
    %v1080 = stablehlo.rsqrt %v1079 : tensor<256x1024x14x14xf32>
    %v1081 = stablehlo.multiply %v1074, %v1080 : tensor<256x1024x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1083 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1084 = stablehlo.multiply %v1081, %v1082 : tensor<256x1024x14x14xf32>
    %v1085 = stablehlo.add %v1084, %v1083 : tensor<256x1024x14x14xf32>
    %v1086 = stablehlo.reshape %v1085 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1087 = stablehlo.add %v1086, %v1007 : tensor<256x200704xf32>
    %v1088 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v1089 = stablehlo.maximum %v1087, %v1088 : tensor<256x200704xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1091 = stablehlo.convolution(%v1090, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1093 = stablehlo.add %v1091, %v1092 : tensor<256x256x14x14xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1097 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1098 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1099 = stablehlo.reduce(%v1095 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1100 = stablehlo.broadcast_in_dim %v1099, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1101 = stablehlo.divide %v1100, %v1097 : tensor<256x256x14x14xf32>
    %v1102 = stablehlo.subtract %v1095, %v1101 : tensor<256x256x14x14xf32>
    %v1103 = stablehlo.multiply %v1102, %v1102 : tensor<256x256x14x14xf32>
    %v1104 = stablehlo.reduce(%v1103 init: %v1096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1105 = stablehlo.broadcast_in_dim %v1104, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1106 = stablehlo.divide %v1105, %v1097 : tensor<256x256x14x14xf32>
    %v1107 = stablehlo.add %v1106, %v1098 : tensor<256x256x14x14xf32>
    %v1108 = stablehlo.rsqrt %v1107 : tensor<256x256x14x14xf32>
    %v1109 = stablehlo.multiply %v1102, %v1108 : tensor<256x256x14x14xf32>
    %v1110 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1112 = stablehlo.multiply %v1109, %v1110 : tensor<256x256x14x14xf32>
    %v1113 = stablehlo.add %v1112, %v1111 : tensor<256x256x14x14xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1116 = stablehlo.maximum %v1114, %v1115 : tensor<256x50176xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1118 = stablehlo.convolution(%v1117, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v1119 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1120 = stablehlo.add %v1118, %v1119 : tensor<256x256x14x14xf32>
    %v1121 = stablehlo.reshape %v1120 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1122 = stablehlo.reshape %v1121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1124 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v1125 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v1126 = stablehlo.reduce(%v1122 init: %v1123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1127 = stablehlo.broadcast_in_dim %v1126, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1128 = stablehlo.divide %v1127, %v1124 : tensor<256x256x14x14xf32>
    %v1129 = stablehlo.subtract %v1122, %v1128 : tensor<256x256x14x14xf32>
    %v1130 = stablehlo.multiply %v1129, %v1129 : tensor<256x256x14x14xf32>
    %v1131 = stablehlo.reduce(%v1130 init: %v1123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1133 = stablehlo.divide %v1132, %v1124 : tensor<256x256x14x14xf32>
    %v1134 = stablehlo.add %v1133, %v1125 : tensor<256x256x14x14xf32>
    %v1135 = stablehlo.rsqrt %v1134 : tensor<256x256x14x14xf32>
    %v1136 = stablehlo.multiply %v1129, %v1135 : tensor<256x256x14x14xf32>
    %v1137 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1138 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v1139 = stablehlo.multiply %v1136, %v1137 : tensor<256x256x14x14xf32>
    %v1140 = stablehlo.add %v1139, %v1138 : tensor<256x256x14x14xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v1142 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v1143 = stablehlo.maximum %v1141, %v1142 : tensor<256x50176xf32>
    %v1144 = stablehlo.reshape %v1143 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v1145 = stablehlo.convolution(%v1144, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1146 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1147 = stablehlo.add %v1145, %v1146 : tensor<256x1024x14x14xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1151 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v1152 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v1153 = stablehlo.reduce(%v1149 init: %v1150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1154 = stablehlo.broadcast_in_dim %v1153, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1155 = stablehlo.divide %v1154, %v1151 : tensor<256x1024x14x14xf32>
    %v1156 = stablehlo.subtract %v1149, %v1155 : tensor<256x1024x14x14xf32>
    %v1157 = stablehlo.multiply %v1156, %v1156 : tensor<256x1024x14x14xf32>
    %v1158 = stablehlo.reduce(%v1157 init: %v1150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1159 = stablehlo.broadcast_in_dim %v1158, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1160 = stablehlo.divide %v1159, %v1151 : tensor<256x1024x14x14xf32>
    %v1161 = stablehlo.add %v1160, %v1152 : tensor<256x1024x14x14xf32>
    %v1162 = stablehlo.rsqrt %v1161 : tensor<256x1024x14x14xf32>
    %v1163 = stablehlo.multiply %v1156, %v1162 : tensor<256x1024x14x14xf32>
    %v1164 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v1166 = stablehlo.multiply %v1163, %v1164 : tensor<256x1024x14x14xf32>
    %v1167 = stablehlo.add %v1166, %v1165 : tensor<256x1024x14x14xf32>
    %v1168 = stablehlo.reshape %v1167 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1169 = stablehlo.add %v1168, %v1089 : tensor<256x200704xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v1171 = stablehlo.maximum %v1169, %v1170 : tensor<256x200704xf32>
    %v1172 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1173 = stablehlo.convolution(%v1172, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x14x14xf32>
    %v1174 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1175 = stablehlo.add %v1173, %v1174 : tensor<256x512x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1177 = stablehlo.reshape %v1176 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1179 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v1180 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v1181 = stablehlo.reduce(%v1177 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1182 = stablehlo.broadcast_in_dim %v1181, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1183 = stablehlo.divide %v1182, %v1179 : tensor<256x512x14x14xf32>
    %v1184 = stablehlo.subtract %v1177, %v1183 : tensor<256x512x14x14xf32>
    %v1185 = stablehlo.multiply %v1184, %v1184 : tensor<256x512x14x14xf32>
    %v1186 = stablehlo.reduce(%v1185 init: %v1178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1186, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1188 = stablehlo.divide %v1187, %v1179 : tensor<256x512x14x14xf32>
    %v1189 = stablehlo.add %v1188, %v1180 : tensor<256x512x14x14xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<256x512x14x14xf32>
    %v1191 = stablehlo.multiply %v1184, %v1190 : tensor<256x512x14x14xf32>
    %v1192 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1193 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1194 = stablehlo.multiply %v1191, %v1192 : tensor<256x512x14x14xf32>
    %v1195 = stablehlo.add %v1194, %v1193 : tensor<256x512x14x14xf32>
    %v1196 = stablehlo.reshape %v1195 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1197 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1198 = stablehlo.maximum %v1196, %v1197 : tensor<256x100352xf32>
    %v1199 = stablehlo.reshape %v1198 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1200 = stablehlo.convolution(%v1199, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1201 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1202 = stablehlo.add %v1200, %v1201 : tensor<256x512x7x7xf32>
    %v1203 = stablehlo.reshape %v1202 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1204 = stablehlo.reshape %v1203 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1206 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1207 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1208 = stablehlo.reduce(%v1204 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1209 = stablehlo.broadcast_in_dim %v1208, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1210 = stablehlo.divide %v1209, %v1206 : tensor<256x512x7x7xf32>
    %v1211 = stablehlo.subtract %v1204, %v1210 : tensor<256x512x7x7xf32>
    %v1212 = stablehlo.multiply %v1211, %v1211 : tensor<256x512x7x7xf32>
    %v1213 = stablehlo.reduce(%v1212 init: %v1205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1214 = stablehlo.broadcast_in_dim %v1213, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1215 = stablehlo.divide %v1214, %v1206 : tensor<256x512x7x7xf32>
    %v1216 = stablehlo.add %v1215, %v1207 : tensor<256x512x7x7xf32>
    %v1217 = stablehlo.rsqrt %v1216 : tensor<256x512x7x7xf32>
    %v1218 = stablehlo.multiply %v1211, %v1217 : tensor<256x512x7x7xf32>
    %v1219 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1220 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1221 = stablehlo.multiply %v1218, %v1219 : tensor<256x512x7x7xf32>
    %v1222 = stablehlo.add %v1221, %v1220 : tensor<256x512x7x7xf32>
    %v1223 = stablehlo.reshape %v1222 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1224 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1225 = stablehlo.maximum %v1223, %v1224 : tensor<256x25088xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1227 = stablehlo.convolution(%v1226, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1228 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1229 = stablehlo.add %v1227, %v1228 : tensor<256x2048x7x7xf32>
    %v1230 = stablehlo.reshape %v1229 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1231 = stablehlo.reshape %v1230 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1233 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1234 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1235 = stablehlo.reduce(%v1231 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1236 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1237 = stablehlo.divide %v1236, %v1233 : tensor<256x2048x7x7xf32>
    %v1238 = stablehlo.subtract %v1231, %v1237 : tensor<256x2048x7x7xf32>
    %v1239 = stablehlo.multiply %v1238, %v1238 : tensor<256x2048x7x7xf32>
    %v1240 = stablehlo.reduce(%v1239 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1241 = stablehlo.broadcast_in_dim %v1240, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1242 = stablehlo.divide %v1241, %v1233 : tensor<256x2048x7x7xf32>
    %v1243 = stablehlo.add %v1242, %v1234 : tensor<256x2048x7x7xf32>
    %v1244 = stablehlo.rsqrt %v1243 : tensor<256x2048x7x7xf32>
    %v1245 = stablehlo.multiply %v1238, %v1244 : tensor<256x2048x7x7xf32>
    %v1246 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1247 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1248 = stablehlo.multiply %v1245, %v1246 : tensor<256x2048x7x7xf32>
    %v1249 = stablehlo.add %v1248, %v1247 : tensor<256x2048x7x7xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1251 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v1252 = stablehlo.convolution(%v1251, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1253 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1254 = stablehlo.add %v1252, %v1253 : tensor<256x2048x7x7xf32>
    %v1255 = stablehlo.reshape %v1254 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1256 = stablehlo.reshape %v1255 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1258 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1259 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1260 = stablehlo.reduce(%v1256 init: %v1257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1261 = stablehlo.broadcast_in_dim %v1260, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1262 = stablehlo.divide %v1261, %v1258 : tensor<256x2048x7x7xf32>
    %v1263 = stablehlo.subtract %v1256, %v1262 : tensor<256x2048x7x7xf32>
    %v1264 = stablehlo.multiply %v1263, %v1263 : tensor<256x2048x7x7xf32>
    %v1265 = stablehlo.reduce(%v1264 init: %v1257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1266 = stablehlo.broadcast_in_dim %v1265, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1267 = stablehlo.divide %v1266, %v1258 : tensor<256x2048x7x7xf32>
    %v1268 = stablehlo.add %v1267, %v1259 : tensor<256x2048x7x7xf32>
    %v1269 = stablehlo.rsqrt %v1268 : tensor<256x2048x7x7xf32>
    %v1270 = stablehlo.multiply %v1263, %v1269 : tensor<256x2048x7x7xf32>
    %v1271 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1272 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1273 = stablehlo.multiply %v1270, %v1271 : tensor<256x2048x7x7xf32>
    %v1274 = stablehlo.add %v1273, %v1272 : tensor<256x2048x7x7xf32>
    %v1275 = stablehlo.reshape %v1274 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1276 = stablehlo.add %v1250, %v1275 : tensor<256x100352xf32>
    %v1277 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1278 = stablehlo.maximum %v1276, %v1277 : tensor<256x100352xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1280 = stablehlo.convolution(%v1279, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1281 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1282 = stablehlo.add %v1280, %v1281 : tensor<256x512x7x7xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1285 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1286 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1287 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1288 = stablehlo.reduce(%v1284 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1288, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1290 = stablehlo.divide %v1289, %v1286 : tensor<256x512x7x7xf32>
    %v1291 = stablehlo.subtract %v1284, %v1290 : tensor<256x512x7x7xf32>
    %v1292 = stablehlo.multiply %v1291, %v1291 : tensor<256x512x7x7xf32>
    %v1293 = stablehlo.reduce(%v1292 init: %v1285) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1293, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1295 = stablehlo.divide %v1294, %v1286 : tensor<256x512x7x7xf32>
    %v1296 = stablehlo.add %v1295, %v1287 : tensor<256x512x7x7xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<256x512x7x7xf32>
    %v1298 = stablehlo.multiply %v1291, %v1297 : tensor<256x512x7x7xf32>
    %v1299 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1300 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1298, %v1299 : tensor<256x512x7x7xf32>
    %v1302 = stablehlo.add %v1301, %v1300 : tensor<256x512x7x7xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1304 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1305 = stablehlo.maximum %v1303, %v1304 : tensor<256x25088xf32>
    %v1306 = stablehlo.reshape %v1305 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1307 = stablehlo.convolution(%v1306, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1308 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1309 = stablehlo.add %v1307, %v1308 : tensor<256x512x7x7xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1312 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1313 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1314 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1315 = stablehlo.reduce(%v1311 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1316 = stablehlo.broadcast_in_dim %v1315, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1317 = stablehlo.divide %v1316, %v1313 : tensor<256x512x7x7xf32>
    %v1318 = stablehlo.subtract %v1311, %v1317 : tensor<256x512x7x7xf32>
    %v1319 = stablehlo.multiply %v1318, %v1318 : tensor<256x512x7x7xf32>
    %v1320 = stablehlo.reduce(%v1319 init: %v1312) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1321 = stablehlo.broadcast_in_dim %v1320, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1322 = stablehlo.divide %v1321, %v1313 : tensor<256x512x7x7xf32>
    %v1323 = stablehlo.add %v1322, %v1314 : tensor<256x512x7x7xf32>
    %v1324 = stablehlo.rsqrt %v1323 : tensor<256x512x7x7xf32>
    %v1325 = stablehlo.multiply %v1318, %v1324 : tensor<256x512x7x7xf32>
    %v1326 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1327 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1325, %v1326 : tensor<256x512x7x7xf32>
    %v1329 = stablehlo.add %v1328, %v1327 : tensor<256x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1329 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1331 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1332 = stablehlo.maximum %v1330, %v1331 : tensor<256x25088xf32>
    %v1333 = stablehlo.reshape %v1332 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1334 = stablehlo.convolution(%v1333, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1335 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1336 = stablehlo.add %v1334, %v1335 : tensor<256x2048x7x7xf32>
    %v1337 = stablehlo.reshape %v1336 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1340 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1341 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1342 = stablehlo.reduce(%v1338 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1343 = stablehlo.broadcast_in_dim %v1342, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1344 = stablehlo.divide %v1343, %v1340 : tensor<256x2048x7x7xf32>
    %v1345 = stablehlo.subtract %v1338, %v1344 : tensor<256x2048x7x7xf32>
    %v1346 = stablehlo.multiply %v1345, %v1345 : tensor<256x2048x7x7xf32>
    %v1347 = stablehlo.reduce(%v1346 init: %v1339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1348 = stablehlo.broadcast_in_dim %v1347, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1349 = stablehlo.divide %v1348, %v1340 : tensor<256x2048x7x7xf32>
    %v1350 = stablehlo.add %v1349, %v1341 : tensor<256x2048x7x7xf32>
    %v1351 = stablehlo.rsqrt %v1350 : tensor<256x2048x7x7xf32>
    %v1352 = stablehlo.multiply %v1345, %v1351 : tensor<256x2048x7x7xf32>
    %v1353 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1355 = stablehlo.multiply %v1352, %v1353 : tensor<256x2048x7x7xf32>
    %v1356 = stablehlo.add %v1355, %v1354 : tensor<256x2048x7x7xf32>
    %v1357 = stablehlo.reshape %v1356 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1358 = stablehlo.add %v1357, %v1278 : tensor<256x100352xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1360 = stablehlo.maximum %v1358, %v1359 : tensor<256x100352xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1362 = stablehlo.convolution(%v1361, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1363 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1364 = stablehlo.add %v1362, %v1363 : tensor<256x512x7x7xf32>
    %v1365 = stablehlo.reshape %v1364 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1366 = stablehlo.reshape %v1365 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1368 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1369 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1370 = stablehlo.reduce(%v1366 init: %v1367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1371 = stablehlo.broadcast_in_dim %v1370, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1372 = stablehlo.divide %v1371, %v1368 : tensor<256x512x7x7xf32>
    %v1373 = stablehlo.subtract %v1366, %v1372 : tensor<256x512x7x7xf32>
    %v1374 = stablehlo.multiply %v1373, %v1373 : tensor<256x512x7x7xf32>
    %v1375 = stablehlo.reduce(%v1374 init: %v1367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1377 = stablehlo.divide %v1376, %v1368 : tensor<256x512x7x7xf32>
    %v1378 = stablehlo.add %v1377, %v1369 : tensor<256x512x7x7xf32>
    %v1379 = stablehlo.rsqrt %v1378 : tensor<256x512x7x7xf32>
    %v1380 = stablehlo.multiply %v1373, %v1379 : tensor<256x512x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1383 = stablehlo.multiply %v1380, %v1381 : tensor<256x512x7x7xf32>
    %v1384 = stablehlo.add %v1383, %v1382 : tensor<256x512x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1387 = stablehlo.maximum %v1385, %v1386 : tensor<256x25088xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1389 = stablehlo.convolution(%v1388, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1390 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1391 = stablehlo.add %v1389, %v1390 : tensor<256x512x7x7xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1395 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1396 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1397 = stablehlo.reduce(%v1393 init: %v1394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1398 = stablehlo.broadcast_in_dim %v1397, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1399 = stablehlo.divide %v1398, %v1395 : tensor<256x512x7x7xf32>
    %v1400 = stablehlo.subtract %v1393, %v1399 : tensor<256x512x7x7xf32>
    %v1401 = stablehlo.multiply %v1400, %v1400 : tensor<256x512x7x7xf32>
    %v1402 = stablehlo.reduce(%v1401 init: %v1394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1403 = stablehlo.broadcast_in_dim %v1402, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1404 = stablehlo.divide %v1403, %v1395 : tensor<256x512x7x7xf32>
    %v1405 = stablehlo.add %v1404, %v1396 : tensor<256x512x7x7xf32>
    %v1406 = stablehlo.rsqrt %v1405 : tensor<256x512x7x7xf32>
    %v1407 = stablehlo.multiply %v1400, %v1406 : tensor<256x512x7x7xf32>
    %v1408 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1409 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1410 = stablehlo.multiply %v1407, %v1408 : tensor<256x512x7x7xf32>
    %v1411 = stablehlo.add %v1410, %v1409 : tensor<256x512x7x7xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1414 = stablehlo.maximum %v1412, %v1413 : tensor<256x25088xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1416 = stablehlo.convolution(%v1415, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1417 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1418 = stablehlo.add %v1416, %v1417 : tensor<256x2048x7x7xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1422 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1423 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1424 = stablehlo.reduce(%v1420 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1425 = stablehlo.broadcast_in_dim %v1424, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1426 = stablehlo.divide %v1425, %v1422 : tensor<256x2048x7x7xf32>
    %v1427 = stablehlo.subtract %v1420, %v1426 : tensor<256x2048x7x7xf32>
    %v1428 = stablehlo.multiply %v1427, %v1427 : tensor<256x2048x7x7xf32>
    %v1429 = stablehlo.reduce(%v1428 init: %v1421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1430 = stablehlo.broadcast_in_dim %v1429, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1431 = stablehlo.divide %v1430, %v1422 : tensor<256x2048x7x7xf32>
    %v1432 = stablehlo.add %v1431, %v1423 : tensor<256x2048x7x7xf32>
    %v1433 = stablehlo.rsqrt %v1432 : tensor<256x2048x7x7xf32>
    %v1434 = stablehlo.multiply %v1427, %v1433 : tensor<256x2048x7x7xf32>
    %v1435 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1436 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1437 = stablehlo.multiply %v1434, %v1435 : tensor<256x2048x7x7xf32>
    %v1438 = stablehlo.add %v1437, %v1436 : tensor<256x2048x7x7xf32>
    %v1439 = stablehlo.reshape %v1438 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1440 = stablehlo.add %v1439, %v1360 : tensor<256x100352xf32>
    %v1441 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1442 = stablehlo.maximum %v1440, %v1441 : tensor<256x100352xf32>
    %v1443 = stablehlo.reshape %v1442 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.reduce(%v1443 init: %v1444) applies stablehlo.add across dimensions = [2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048xf32>
    %v1446 = stablehlo.constant dense<49.0> : tensor<256x2048xf32>
    %v1447 = stablehlo.divide %v1445, %v1446 : tensor<256x2048xf32>
    %v1448 = stablehlo.dot_general %v1447, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<2048x1000xf32>) -> tensor<256x1000xf32>
    %v1449 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<256x1000xf32>
    %v1450 = stablehlo.add %v1448, %v1449 : tensor<256x1000xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1453 = stablehlo.exponential %v1451 : tensor<256x1x1000xf32>
    %v1454 = stablehlo.reduce(%v1453 init: %v1452) applies stablehlo.add across dimensions = [2] : (tensor<256x1x1000xf32>, tensor<f32>) -> tensor<256x1xf32>
    %v1455 = stablehlo.broadcast_in_dim %v1454, dims = [0, 1] : (tensor<256x1xf32>) -> tensor<256x1x1000xf32>
    %v1456 = stablehlo.divide %v1453, %v1455 : tensor<256x1x1000xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<256x1x1000xf32>) -> tensor<256x1000xf32>
    %v1458 = stablehlo.subtract %v1457, %onehot : tensor<256x1000xf32>
    %v1459 = stablehlo.constant dense<0.100000> : tensor<256x1000xf32>
    %v1460 = stablehlo.multiply %onehot, %v1459 : tensor<256x1000xf32>
    %v1461 = stablehlo.add %v1458, %v1460 : tensor<256x1000xf32>
    %v1462 = stablehlo.constant dense<-0.000100> : tensor<256x1000xf32>
    %v1463 = stablehlo.add %v1461, %v1462 : tensor<256x1000xf32>
    %v1464 = stablehlo.constant dense<256.0> : tensor<256x1000xf32>
    %v1465 = stablehlo.divide %v1463, %v1464 : tensor<256x1000xf32>
    %v1466 = stablehlo.reshape %v1465 : (tensor<256x1000xf32>) -> tensor<256x1x1000xf32>
    %v1467 = stablehlo.dot_general %v1466, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<256x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<256x1x2048xf32>
    %v1468 = stablehlo.reshape %v1467 : (tensor<256x1x2048xf32>) -> tensor<256x2048xf32>
    %v1469 = stablehlo.dot_general %v1447, %v1465, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<256x2048xf32>, tensor<256x1000xf32>) -> tensor<2048x1000xf32>
    %v1470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1471 = stablehlo.reduce(%v1465 init: %v1470) applies stablehlo.add across dimensions = [0] : (tensor<256x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1472 = stablehlo.broadcast_in_dim %v1468, dims = [0, 1] : (tensor<256x2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1473 = stablehlo.constant dense<49.0> : tensor<256x2048x7x7xf32>
    %v1474 = stablehlo.divide %v1472, %v1473 : tensor<256x2048x7x7xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1476 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1477 = stablehlo.compare GT, %v1440, %v1476 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v1478 = stablehlo.select %v1477, %v1475, %v1476 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v1479 = stablehlo.reshape %v1419 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1481 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1482 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1483 = stablehlo.reduce(%v1479 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1484 = stablehlo.broadcast_in_dim %v1483, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1485 = stablehlo.divide %v1484, %v1481 : tensor<256x2048x7x7xf32>
    %v1486 = stablehlo.subtract %v1479, %v1485 : tensor<256x2048x7x7xf32>
    %v1487 = stablehlo.multiply %v1486, %v1486 : tensor<256x2048x7x7xf32>
    %v1488 = stablehlo.reduce(%v1487 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1489 = stablehlo.broadcast_in_dim %v1488, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1490 = stablehlo.divide %v1489, %v1481 : tensor<256x2048x7x7xf32>
    %v1491 = stablehlo.add %v1490, %v1482 : tensor<256x2048x7x7xf32>
    %v1492 = stablehlo.rsqrt %v1491 : tensor<256x2048x7x7xf32>
    %v1493 = stablehlo.multiply %v1486, %v1492 : tensor<256x2048x7x7xf32>
    %v1494 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1495 = stablehlo.reshape %v1478 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1496 = stablehlo.multiply %v1494, %v1495 : tensor<256x2048x7x7xf32>
    %v1497 = stablehlo.reduce(%v1496 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1499 = stablehlo.multiply %v1493, %v1496 : tensor<256x2048x7x7xf32>
    %v1500 = stablehlo.reduce(%v1499 init: %v1480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1501 = stablehlo.broadcast_in_dim %v1500, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1502 = stablehlo.multiply %v1496, %v1481 : tensor<256x2048x7x7xf32>
    %v1503 = stablehlo.subtract %v1502, %v1498 : tensor<256x2048x7x7xf32>
    %v1504 = stablehlo.multiply %v1493, %v1501 : tensor<256x2048x7x7xf32>
    %v1505 = stablehlo.subtract %v1503, %v1504 : tensor<256x2048x7x7xf32>
    %v1506 = stablehlo.divide %v1492, %v1481 : tensor<256x2048x7x7xf32>
    %v1507 = stablehlo.multiply %v1506, %v1505 : tensor<256x2048x7x7xf32>
    %v1508 = stablehlo.reshape %v1507 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1509 = stablehlo.reshape %v1508 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1510 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1511 = stablehlo.transpose %v1510, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1512 = stablehlo.convolution(%v1509, %v1511)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1513 = stablehlo.reshape %v1512 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1514 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1515 = stablehlo.compare GT, %v1412, %v1514 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1516 = stablehlo.select %v1515, %v1513, %v1514 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1517 = stablehlo.reshape %v1392 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1519 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1520 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1521 = stablehlo.reduce(%v1517 init: %v1518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1522 = stablehlo.broadcast_in_dim %v1521, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1523 = stablehlo.divide %v1522, %v1519 : tensor<256x512x7x7xf32>
    %v1524 = stablehlo.subtract %v1517, %v1523 : tensor<256x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1524 : tensor<256x512x7x7xf32>
    %v1526 = stablehlo.reduce(%v1525 init: %v1518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1527 = stablehlo.broadcast_in_dim %v1526, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1528 = stablehlo.divide %v1527, %v1519 : tensor<256x512x7x7xf32>
    %v1529 = stablehlo.add %v1528, %v1520 : tensor<256x512x7x7xf32>
    %v1530 = stablehlo.rsqrt %v1529 : tensor<256x512x7x7xf32>
    %v1531 = stablehlo.multiply %v1524, %v1530 : tensor<256x512x7x7xf32>
    %v1532 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1533 = stablehlo.reshape %v1516 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1534 = stablehlo.multiply %v1532, %v1533 : tensor<256x512x7x7xf32>
    %v1535 = stablehlo.reduce(%v1534 init: %v1518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1536 = stablehlo.broadcast_in_dim %v1535, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1537 = stablehlo.multiply %v1531, %v1534 : tensor<256x512x7x7xf32>
    %v1538 = stablehlo.reduce(%v1537 init: %v1518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1539 = stablehlo.broadcast_in_dim %v1538, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1540 = stablehlo.multiply %v1534, %v1519 : tensor<256x512x7x7xf32>
    %v1541 = stablehlo.subtract %v1540, %v1536 : tensor<256x512x7x7xf32>
    %v1542 = stablehlo.multiply %v1531, %v1539 : tensor<256x512x7x7xf32>
    %v1543 = stablehlo.subtract %v1541, %v1542 : tensor<256x512x7x7xf32>
    %v1544 = stablehlo.divide %v1530, %v1519 : tensor<256x512x7x7xf32>
    %v1545 = stablehlo.multiply %v1544, %v1543 : tensor<256x512x7x7xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1548 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1549 = stablehlo.transpose %v1548, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1550 = stablehlo.convolution(%v1547, %v1549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1551 = stablehlo.reshape %v1550 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1552 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1553 = stablehlo.compare GT, %v1385, %v1552 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1554 = stablehlo.select %v1553, %v1551, %v1552 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1555 = stablehlo.reshape %v1365 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1557 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1558 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1559 = stablehlo.reduce(%v1555 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1560 = stablehlo.broadcast_in_dim %v1559, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1561 = stablehlo.divide %v1560, %v1557 : tensor<256x512x7x7xf32>
    %v1562 = stablehlo.subtract %v1555, %v1561 : tensor<256x512x7x7xf32>
    %v1563 = stablehlo.multiply %v1562, %v1562 : tensor<256x512x7x7xf32>
    %v1564 = stablehlo.reduce(%v1563 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1565 = stablehlo.broadcast_in_dim %v1564, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1566 = stablehlo.divide %v1565, %v1557 : tensor<256x512x7x7xf32>
    %v1567 = stablehlo.add %v1566, %v1558 : tensor<256x512x7x7xf32>
    %v1568 = stablehlo.rsqrt %v1567 : tensor<256x512x7x7xf32>
    %v1569 = stablehlo.multiply %v1562, %v1568 : tensor<256x512x7x7xf32>
    %v1570 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1571 = stablehlo.reshape %v1554 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1572 = stablehlo.multiply %v1570, %v1571 : tensor<256x512x7x7xf32>
    %v1573 = stablehlo.reduce(%v1572 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1574 = stablehlo.broadcast_in_dim %v1573, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1575 = stablehlo.multiply %v1569, %v1572 : tensor<256x512x7x7xf32>
    %v1576 = stablehlo.reduce(%v1575 init: %v1556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1577 = stablehlo.broadcast_in_dim %v1576, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1578 = stablehlo.multiply %v1572, %v1557 : tensor<256x512x7x7xf32>
    %v1579 = stablehlo.subtract %v1578, %v1574 : tensor<256x512x7x7xf32>
    %v1580 = stablehlo.multiply %v1569, %v1577 : tensor<256x512x7x7xf32>
    %v1581 = stablehlo.subtract %v1579, %v1580 : tensor<256x512x7x7xf32>
    %v1582 = stablehlo.divide %v1568, %v1557 : tensor<256x512x7x7xf32>
    %v1583 = stablehlo.multiply %v1582, %v1581 : tensor<256x512x7x7xf32>
    %v1584 = stablehlo.reshape %v1583 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1585 = stablehlo.reshape %v1584 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1586 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1587 = stablehlo.transpose %v1586, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1588 = stablehlo.convolution(%v1585, %v1587)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1589 = stablehlo.reshape %v1588 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1590 = stablehlo.add %v1589, %v1478 : tensor<256x100352xf32>
    %v1591 = stablehlo.reshape %v1360 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1592 = stablehlo.reshape %v1584 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1593 = stablehlo.transpose %v1591, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1594 = stablehlo.transpose %v1592, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1595 = stablehlo.convolution(%v1593, %v1594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v1596 = stablehlo.transpose %v1595, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1597 = stablehlo.reshape %v1365 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1599 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1600 = stablehlo.reduce(%v1597 init: %v1598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1602 = stablehlo.divide %v1601, %v1599 : tensor<256x512x7x7xf32>
    %v1603 = stablehlo.subtract %v1597, %v1602 : tensor<256x512x7x7xf32>
    %v1604 = stablehlo.multiply %v1603, %v1603 : tensor<256x512x7x7xf32>
    %v1605 = stablehlo.reduce(%v1604 init: %v1598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1606 = stablehlo.broadcast_in_dim %v1605, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1607 = stablehlo.divide %v1606, %v1599 : tensor<256x512x7x7xf32>
    %v1608 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1609 = stablehlo.add %v1607, %v1608 : tensor<256x512x7x7xf32>
    %v1610 = stablehlo.rsqrt %v1609 : tensor<256x512x7x7xf32>
    %v1611 = stablehlo.multiply %v1603, %v1610 : tensor<256x512x7x7xf32>
    %v1612 = stablehlo.reshape %v1554 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1613 = stablehlo.multiply %v1612, %v1611 : tensor<256x512x7x7xf32>
    %v1614 = stablehlo.reduce(%v1613 init: %v1598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1615 = stablehlo.reshape %v1554 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1617 = stablehlo.reduce(%v1615 init: %v1616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1618 = stablehlo.reshape %v1387 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1619 = stablehlo.reshape %v1546 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1620 = stablehlo.transpose %v1618, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1621 = stablehlo.transpose %v1619, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1622 = stablehlo.convolution(%v1620, %v1621)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1623 = stablehlo.transpose %v1622, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1624 = stablehlo.reshape %v1392 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1626 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1627 = stablehlo.reduce(%v1624 init: %v1625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1628 = stablehlo.broadcast_in_dim %v1627, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1629 = stablehlo.divide %v1628, %v1626 : tensor<256x512x7x7xf32>
    %v1630 = stablehlo.subtract %v1624, %v1629 : tensor<256x512x7x7xf32>
    %v1631 = stablehlo.multiply %v1630, %v1630 : tensor<256x512x7x7xf32>
    %v1632 = stablehlo.reduce(%v1631 init: %v1625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1633 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1634 = stablehlo.divide %v1633, %v1626 : tensor<256x512x7x7xf32>
    %v1635 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1636 = stablehlo.add %v1634, %v1635 : tensor<256x512x7x7xf32>
    %v1637 = stablehlo.rsqrt %v1636 : tensor<256x512x7x7xf32>
    %v1638 = stablehlo.multiply %v1630, %v1637 : tensor<256x512x7x7xf32>
    %v1639 = stablehlo.reshape %v1516 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1640 = stablehlo.multiply %v1639, %v1638 : tensor<256x512x7x7xf32>
    %v1641 = stablehlo.reduce(%v1640 init: %v1625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1642 = stablehlo.reshape %v1516 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1644 = stablehlo.reduce(%v1642 init: %v1643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1645 = stablehlo.reshape %v1414 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1646 = stablehlo.reshape %v1508 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1647 = stablehlo.transpose %v1645, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1648 = stablehlo.transpose %v1646, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1649 = stablehlo.convolution(%v1647, %v1648)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v1650 = stablehlo.transpose %v1649, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1651 = stablehlo.reshape %v1419 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1653 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1654 = stablehlo.reduce(%v1651 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1655 = stablehlo.broadcast_in_dim %v1654, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1656 = stablehlo.divide %v1655, %v1653 : tensor<256x2048x7x7xf32>
    %v1657 = stablehlo.subtract %v1651, %v1656 : tensor<256x2048x7x7xf32>
    %v1658 = stablehlo.multiply %v1657, %v1657 : tensor<256x2048x7x7xf32>
    %v1659 = stablehlo.reduce(%v1658 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1660 = stablehlo.broadcast_in_dim %v1659, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1661 = stablehlo.divide %v1660, %v1653 : tensor<256x2048x7x7xf32>
    %v1662 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1663 = stablehlo.add %v1661, %v1662 : tensor<256x2048x7x7xf32>
    %v1664 = stablehlo.rsqrt %v1663 : tensor<256x2048x7x7xf32>
    %v1665 = stablehlo.multiply %v1657, %v1664 : tensor<256x2048x7x7xf32>
    %v1666 = stablehlo.reshape %v1478 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1667 = stablehlo.multiply %v1666, %v1665 : tensor<256x2048x7x7xf32>
    %v1668 = stablehlo.reduce(%v1667 init: %v1652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1669 = stablehlo.reshape %v1478 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1671 = stablehlo.reduce(%v1669 init: %v1670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1672 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1673 = stablehlo.compare GT, %v1358, %v1672 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v1674 = stablehlo.select %v1673, %v1590, %v1672 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v1675 = stablehlo.reshape %v1337 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1677 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1678 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1679 = stablehlo.reduce(%v1675 init: %v1676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1680 = stablehlo.broadcast_in_dim %v1679, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1681 = stablehlo.divide %v1680, %v1677 : tensor<256x2048x7x7xf32>
    %v1682 = stablehlo.subtract %v1675, %v1681 : tensor<256x2048x7x7xf32>
    %v1683 = stablehlo.multiply %v1682, %v1682 : tensor<256x2048x7x7xf32>
    %v1684 = stablehlo.reduce(%v1683 init: %v1676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1685 = stablehlo.broadcast_in_dim %v1684, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1686 = stablehlo.divide %v1685, %v1677 : tensor<256x2048x7x7xf32>
    %v1687 = stablehlo.add %v1686, %v1678 : tensor<256x2048x7x7xf32>
    %v1688 = stablehlo.rsqrt %v1687 : tensor<256x2048x7x7xf32>
    %v1689 = stablehlo.multiply %v1682, %v1688 : tensor<256x2048x7x7xf32>
    %v1690 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1691 = stablehlo.reshape %v1674 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1692 = stablehlo.multiply %v1690, %v1691 : tensor<256x2048x7x7xf32>
    %v1693 = stablehlo.reduce(%v1692 init: %v1676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1694 = stablehlo.broadcast_in_dim %v1693, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1695 = stablehlo.multiply %v1689, %v1692 : tensor<256x2048x7x7xf32>
    %v1696 = stablehlo.reduce(%v1695 init: %v1676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1697 = stablehlo.broadcast_in_dim %v1696, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1698 = stablehlo.multiply %v1692, %v1677 : tensor<256x2048x7x7xf32>
    %v1699 = stablehlo.subtract %v1698, %v1694 : tensor<256x2048x7x7xf32>
    %v1700 = stablehlo.multiply %v1689, %v1697 : tensor<256x2048x7x7xf32>
    %v1701 = stablehlo.subtract %v1699, %v1700 : tensor<256x2048x7x7xf32>
    %v1702 = stablehlo.divide %v1688, %v1677 : tensor<256x2048x7x7xf32>
    %v1703 = stablehlo.multiply %v1702, %v1701 : tensor<256x2048x7x7xf32>
    %v1704 = stablehlo.reshape %v1703 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1706 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1707 = stablehlo.transpose %v1706, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1708 = stablehlo.convolution(%v1705, %v1707)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1710 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1711 = stablehlo.compare GT, %v1330, %v1710 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1712 = stablehlo.select %v1711, %v1709, %v1710 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1713 = stablehlo.reshape %v1310 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1715 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1716 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1717 = stablehlo.reduce(%v1713 init: %v1714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1718 = stablehlo.broadcast_in_dim %v1717, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1719 = stablehlo.divide %v1718, %v1715 : tensor<256x512x7x7xf32>
    %v1720 = stablehlo.subtract %v1713, %v1719 : tensor<256x512x7x7xf32>
    %v1721 = stablehlo.multiply %v1720, %v1720 : tensor<256x512x7x7xf32>
    %v1722 = stablehlo.reduce(%v1721 init: %v1714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1723 = stablehlo.broadcast_in_dim %v1722, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1724 = stablehlo.divide %v1723, %v1715 : tensor<256x512x7x7xf32>
    %v1725 = stablehlo.add %v1724, %v1716 : tensor<256x512x7x7xf32>
    %v1726 = stablehlo.rsqrt %v1725 : tensor<256x512x7x7xf32>
    %v1727 = stablehlo.multiply %v1720, %v1726 : tensor<256x512x7x7xf32>
    %v1728 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1729 = stablehlo.reshape %v1712 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1730 = stablehlo.multiply %v1728, %v1729 : tensor<256x512x7x7xf32>
    %v1731 = stablehlo.reduce(%v1730 init: %v1714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1731, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1733 = stablehlo.multiply %v1727, %v1730 : tensor<256x512x7x7xf32>
    %v1734 = stablehlo.reduce(%v1733 init: %v1714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1735 = stablehlo.broadcast_in_dim %v1734, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1736 = stablehlo.multiply %v1730, %v1715 : tensor<256x512x7x7xf32>
    %v1737 = stablehlo.subtract %v1736, %v1732 : tensor<256x512x7x7xf32>
    %v1738 = stablehlo.multiply %v1727, %v1735 : tensor<256x512x7x7xf32>
    %v1739 = stablehlo.subtract %v1737, %v1738 : tensor<256x512x7x7xf32>
    %v1740 = stablehlo.divide %v1726, %v1715 : tensor<256x512x7x7xf32>
    %v1741 = stablehlo.multiply %v1740, %v1739 : tensor<256x512x7x7xf32>
    %v1742 = stablehlo.reshape %v1741 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1744 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1745 = stablehlo.transpose %v1744, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1746 = stablehlo.convolution(%v1743, %v1745)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x7x7xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1748 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1749 = stablehlo.compare GT, %v1303, %v1748 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1750 = stablehlo.select %v1749, %v1747, %v1748 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1751 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1753 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1754 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1755 = stablehlo.reduce(%v1751 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1756 = stablehlo.broadcast_in_dim %v1755, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1757 = stablehlo.divide %v1756, %v1753 : tensor<256x512x7x7xf32>
    %v1758 = stablehlo.subtract %v1751, %v1757 : tensor<256x512x7x7xf32>
    %v1759 = stablehlo.multiply %v1758, %v1758 : tensor<256x512x7x7xf32>
    %v1760 = stablehlo.reduce(%v1759 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1761 = stablehlo.broadcast_in_dim %v1760, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1762 = stablehlo.divide %v1761, %v1753 : tensor<256x512x7x7xf32>
    %v1763 = stablehlo.add %v1762, %v1754 : tensor<256x512x7x7xf32>
    %v1764 = stablehlo.rsqrt %v1763 : tensor<256x512x7x7xf32>
    %v1765 = stablehlo.multiply %v1758, %v1764 : tensor<256x512x7x7xf32>
    %v1766 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1767 = stablehlo.reshape %v1750 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1768 = stablehlo.multiply %v1766, %v1767 : tensor<256x512x7x7xf32>
    %v1769 = stablehlo.reduce(%v1768 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1770 = stablehlo.broadcast_in_dim %v1769, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1771 = stablehlo.multiply %v1765, %v1768 : tensor<256x512x7x7xf32>
    %v1772 = stablehlo.reduce(%v1771 init: %v1752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1773 = stablehlo.broadcast_in_dim %v1772, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1774 = stablehlo.multiply %v1768, %v1753 : tensor<256x512x7x7xf32>
    %v1775 = stablehlo.subtract %v1774, %v1770 : tensor<256x512x7x7xf32>
    %v1776 = stablehlo.multiply %v1765, %v1773 : tensor<256x512x7x7xf32>
    %v1777 = stablehlo.subtract %v1775, %v1776 : tensor<256x512x7x7xf32>
    %v1778 = stablehlo.divide %v1764, %v1753 : tensor<256x512x7x7xf32>
    %v1779 = stablehlo.multiply %v1778, %v1777 : tensor<256x512x7x7xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1782 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1783 = stablehlo.transpose %v1782, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1784 = stablehlo.convolution(%v1781, %v1783)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<256x2048x7x7xf32>
    %v1785 = stablehlo.reshape %v1784 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1786 = stablehlo.add %v1785, %v1674 : tensor<256x100352xf32>
    %v1787 = stablehlo.reshape %v1278 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1788 = stablehlo.reshape %v1780 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1789 = stablehlo.transpose %v1787, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1790 = stablehlo.transpose %v1788, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1791 = stablehlo.convolution(%v1789, %v1790)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v1792 = stablehlo.transpose %v1791, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1793 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1795 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1796 = stablehlo.reduce(%v1793 init: %v1794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1798 = stablehlo.divide %v1797, %v1795 : tensor<256x512x7x7xf32>
    %v1799 = stablehlo.subtract %v1793, %v1798 : tensor<256x512x7x7xf32>
    %v1800 = stablehlo.multiply %v1799, %v1799 : tensor<256x512x7x7xf32>
    %v1801 = stablehlo.reduce(%v1800 init: %v1794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1803 = stablehlo.divide %v1802, %v1795 : tensor<256x512x7x7xf32>
    %v1804 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1805 = stablehlo.add %v1803, %v1804 : tensor<256x512x7x7xf32>
    %v1806 = stablehlo.rsqrt %v1805 : tensor<256x512x7x7xf32>
    %v1807 = stablehlo.multiply %v1799, %v1806 : tensor<256x512x7x7xf32>
    %v1808 = stablehlo.reshape %v1750 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1809 = stablehlo.multiply %v1808, %v1807 : tensor<256x512x7x7xf32>
    %v1810 = stablehlo.reduce(%v1809 init: %v1794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1811 = stablehlo.reshape %v1750 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1813 = stablehlo.reduce(%v1811 init: %v1812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1814 = stablehlo.reshape %v1305 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1815 = stablehlo.reshape %v1742 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1816 = stablehlo.transpose %v1814, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1817 = stablehlo.transpose %v1815, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1818 = stablehlo.convolution(%v1816, %v1817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<512x256x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v1819 = stablehlo.transpose %v1818, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1820 = stablehlo.reshape %v1310 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1822 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1823 = stablehlo.reduce(%v1820 init: %v1821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1824 = stablehlo.broadcast_in_dim %v1823, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1825 = stablehlo.divide %v1824, %v1822 : tensor<256x512x7x7xf32>
    %v1826 = stablehlo.subtract %v1820, %v1825 : tensor<256x512x7x7xf32>
    %v1827 = stablehlo.multiply %v1826, %v1826 : tensor<256x512x7x7xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1830 = stablehlo.divide %v1829, %v1822 : tensor<256x512x7x7xf32>
    %v1831 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1832 = stablehlo.add %v1830, %v1831 : tensor<256x512x7x7xf32>
    %v1833 = stablehlo.rsqrt %v1832 : tensor<256x512x7x7xf32>
    %v1834 = stablehlo.multiply %v1826, %v1833 : tensor<256x512x7x7xf32>
    %v1835 = stablehlo.reshape %v1712 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1836 = stablehlo.multiply %v1835, %v1834 : tensor<256x512x7x7xf32>
    %v1837 = stablehlo.reduce(%v1836 init: %v1821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1838 = stablehlo.reshape %v1712 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1840 = stablehlo.reduce(%v1838 init: %v1839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1841 = stablehlo.reshape %v1332 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1842 = stablehlo.reshape %v1704 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1843 = stablehlo.transpose %v1841, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v1844 = stablehlo.transpose %v1842, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v1845 = stablehlo.convolution(%v1843, %v1844)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v1846 = stablehlo.transpose %v1845, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1847 = stablehlo.reshape %v1337 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1848 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1849 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1850 = stablehlo.reduce(%v1847 init: %v1848) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1850, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1852 = stablehlo.divide %v1851, %v1849 : tensor<256x2048x7x7xf32>
    %v1853 = stablehlo.subtract %v1847, %v1852 : tensor<256x2048x7x7xf32>
    %v1854 = stablehlo.multiply %v1853, %v1853 : tensor<256x2048x7x7xf32>
    %v1855 = stablehlo.reduce(%v1854 init: %v1848) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1856 = stablehlo.broadcast_in_dim %v1855, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1857 = stablehlo.divide %v1856, %v1849 : tensor<256x2048x7x7xf32>
    %v1858 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1859 = stablehlo.add %v1857, %v1858 : tensor<256x2048x7x7xf32>
    %v1860 = stablehlo.rsqrt %v1859 : tensor<256x2048x7x7xf32>
    %v1861 = stablehlo.multiply %v1853, %v1860 : tensor<256x2048x7x7xf32>
    %v1862 = stablehlo.reshape %v1674 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1863 = stablehlo.multiply %v1862, %v1861 : tensor<256x2048x7x7xf32>
    %v1864 = stablehlo.reduce(%v1863 init: %v1848) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1865 = stablehlo.reshape %v1674 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1867 = stablehlo.reduce(%v1865 init: %v1866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1869 = stablehlo.compare GT, %v1276, %v1868 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v1870 = stablehlo.select %v1869, %v1786, %v1868 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v1871 = stablehlo.reshape %v1230 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1872 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1873 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1874 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1875 = stablehlo.reduce(%v1871 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1877 = stablehlo.divide %v1876, %v1873 : tensor<256x2048x7x7xf32>
    %v1878 = stablehlo.subtract %v1871, %v1877 : tensor<256x2048x7x7xf32>
    %v1879 = stablehlo.multiply %v1878, %v1878 : tensor<256x2048x7x7xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1881 = stablehlo.broadcast_in_dim %v1880, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1882 = stablehlo.divide %v1881, %v1873 : tensor<256x2048x7x7xf32>
    %v1883 = stablehlo.add %v1882, %v1874 : tensor<256x2048x7x7xf32>
    %v1884 = stablehlo.rsqrt %v1883 : tensor<256x2048x7x7xf32>
    %v1885 = stablehlo.multiply %v1878, %v1884 : tensor<256x2048x7x7xf32>
    %v1886 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1887 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1888 = stablehlo.multiply %v1886, %v1887 : tensor<256x2048x7x7xf32>
    %v1889 = stablehlo.reduce(%v1888 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1889, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1891 = stablehlo.multiply %v1885, %v1888 : tensor<256x2048x7x7xf32>
    %v1892 = stablehlo.reduce(%v1891 init: %v1872) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1893 = stablehlo.broadcast_in_dim %v1892, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1894 = stablehlo.multiply %v1888, %v1873 : tensor<256x2048x7x7xf32>
    %v1895 = stablehlo.subtract %v1894, %v1890 : tensor<256x2048x7x7xf32>
    %v1896 = stablehlo.multiply %v1885, %v1893 : tensor<256x2048x7x7xf32>
    %v1897 = stablehlo.subtract %v1895, %v1896 : tensor<256x2048x7x7xf32>
    %v1898 = stablehlo.divide %v1884, %v1873 : tensor<256x2048x7x7xf32>
    %v1899 = stablehlo.multiply %v1898, %v1897 : tensor<256x2048x7x7xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v1901 = stablehlo.reshape %v1900 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1902 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1903 = stablehlo.transpose %v1902, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1904 = stablehlo.convolution(%v1901, %v1903)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<256x512x7x7xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1906 = stablehlo.constant dense<0.0> : tensor<256x25088xf32>
    %v1907 = stablehlo.compare GT, %v1223, %v1906 : (tensor<256x25088xf32>, tensor<256x25088xf32>) -> tensor<256x25088xi1>
    %v1908 = stablehlo.select %v1907, %v1905, %v1906 : tensor<256x25088xi1>, tensor<256x25088xf32>
    %v1909 = stablehlo.reshape %v1203 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1910 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1911 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v1912 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v1913 = stablehlo.reduce(%v1909 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1914 = stablehlo.broadcast_in_dim %v1913, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1915 = stablehlo.divide %v1914, %v1911 : tensor<256x512x7x7xf32>
    %v1916 = stablehlo.subtract %v1909, %v1915 : tensor<256x512x7x7xf32>
    %v1917 = stablehlo.multiply %v1916, %v1916 : tensor<256x512x7x7xf32>
    %v1918 = stablehlo.reduce(%v1917 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1919 = stablehlo.broadcast_in_dim %v1918, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1920 = stablehlo.divide %v1919, %v1911 : tensor<256x512x7x7xf32>
    %v1921 = stablehlo.add %v1920, %v1912 : tensor<256x512x7x7xf32>
    %v1922 = stablehlo.rsqrt %v1921 : tensor<256x512x7x7xf32>
    %v1923 = stablehlo.multiply %v1916, %v1922 : tensor<256x512x7x7xf32>
    %v1924 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1925 = stablehlo.reshape %v1908 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1926 = stablehlo.multiply %v1924, %v1925 : tensor<256x512x7x7xf32>
    %v1927 = stablehlo.reduce(%v1926 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1928 = stablehlo.broadcast_in_dim %v1927, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1929 = stablehlo.multiply %v1923, %v1926 : tensor<256x512x7x7xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1910) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v1932 = stablehlo.multiply %v1926, %v1911 : tensor<256x512x7x7xf32>
    %v1933 = stablehlo.subtract %v1932, %v1928 : tensor<256x512x7x7xf32>
    %v1934 = stablehlo.multiply %v1923, %v1931 : tensor<256x512x7x7xf32>
    %v1935 = stablehlo.subtract %v1933, %v1934 : tensor<256x512x7x7xf32>
    %v1936 = stablehlo.divide %v1922, %v1911 : tensor<256x512x7x7xf32>
    %v1937 = stablehlo.multiply %v1936, %v1935 : tensor<256x512x7x7xf32>
    %v1938 = stablehlo.reshape %v1937 : (tensor<256x512x7x7xf32>) -> tensor<256x25088xf32>
    %v1939 = stablehlo.reshape %v1938 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v1940 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1941 = stablehlo.pad %v1939, %v1940, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v1942 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1943 = stablehlo.transpose %v1942, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1944 = stablehlo.convolution(%v1941, %v1943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<256x512x14x14xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1946 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v1947 = stablehlo.compare GT, %v1196, %v1946 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v1948 = stablehlo.select %v1947, %v1945, %v1946 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v1949 = stablehlo.reshape %v1176 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1951 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v1952 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v1953 = stablehlo.reduce(%v1949 init: %v1950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1954 = stablehlo.broadcast_in_dim %v1953, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1955 = stablehlo.divide %v1954, %v1951 : tensor<256x512x14x14xf32>
    %v1956 = stablehlo.subtract %v1949, %v1955 : tensor<256x512x14x14xf32>
    %v1957 = stablehlo.multiply %v1956, %v1956 : tensor<256x512x14x14xf32>
    %v1958 = stablehlo.reduce(%v1957 init: %v1950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1959 = stablehlo.broadcast_in_dim %v1958, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1960 = stablehlo.divide %v1959, %v1951 : tensor<256x512x14x14xf32>
    %v1961 = stablehlo.add %v1960, %v1952 : tensor<256x512x14x14xf32>
    %v1962 = stablehlo.rsqrt %v1961 : tensor<256x512x14x14xf32>
    %v1963 = stablehlo.multiply %v1956, %v1962 : tensor<256x512x14x14xf32>
    %v1964 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1965 = stablehlo.reshape %v1948 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1966 = stablehlo.multiply %v1964, %v1965 : tensor<256x512x14x14xf32>
    %v1967 = stablehlo.reduce(%v1966 init: %v1950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1968 = stablehlo.broadcast_in_dim %v1967, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1969 = stablehlo.multiply %v1963, %v1966 : tensor<256x512x14x14xf32>
    %v1970 = stablehlo.reduce(%v1969 init: %v1950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1971 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v1972 = stablehlo.multiply %v1966, %v1951 : tensor<256x512x14x14xf32>
    %v1973 = stablehlo.subtract %v1972, %v1968 : tensor<256x512x14x14xf32>
    %v1974 = stablehlo.multiply %v1963, %v1971 : tensor<256x512x14x14xf32>
    %v1975 = stablehlo.subtract %v1973, %v1974 : tensor<256x512x14x14xf32>
    %v1976 = stablehlo.divide %v1962, %v1951 : tensor<256x512x14x14xf32>
    %v1977 = stablehlo.multiply %v1976, %v1975 : tensor<256x512x14x14xf32>
    %v1978 = stablehlo.reshape %v1977 : (tensor<256x512x14x14xf32>) -> tensor<256x100352xf32>
    %v1979 = stablehlo.reshape %v1978 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v1980 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v1981 = stablehlo.transpose %v1980, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v1982 = stablehlo.convolution(%v1979, %v1981)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x14x14xf32>, tensor<1024x512x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v1984 = stablehlo.reshape %v1255 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v1985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1986 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v1987 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v1988 = stablehlo.reduce(%v1984 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1989 = stablehlo.broadcast_in_dim %v1988, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1990 = stablehlo.divide %v1989, %v1986 : tensor<256x2048x7x7xf32>
    %v1991 = stablehlo.subtract %v1984, %v1990 : tensor<256x2048x7x7xf32>
    %v1992 = stablehlo.multiply %v1991, %v1991 : tensor<256x2048x7x7xf32>
    %v1993 = stablehlo.reduce(%v1992 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1994 = stablehlo.broadcast_in_dim %v1993, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v1995 = stablehlo.divide %v1994, %v1986 : tensor<256x2048x7x7xf32>
    %v1996 = stablehlo.add %v1995, %v1987 : tensor<256x2048x7x7xf32>
    %v1997 = stablehlo.rsqrt %v1996 : tensor<256x2048x7x7xf32>
    %v1998 = stablehlo.multiply %v1991, %v1997 : tensor<256x2048x7x7xf32>
    %v1999 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2000 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2001 = stablehlo.multiply %v1999, %v2000 : tensor<256x2048x7x7xf32>
    %v2002 = stablehlo.reduce(%v2001 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2003 = stablehlo.broadcast_in_dim %v2002, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2004 = stablehlo.multiply %v1998, %v2001 : tensor<256x2048x7x7xf32>
    %v2005 = stablehlo.reduce(%v2004 init: %v1985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2006 = stablehlo.broadcast_in_dim %v2005, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2007 = stablehlo.multiply %v2001, %v1986 : tensor<256x2048x7x7xf32>
    %v2008 = stablehlo.subtract %v2007, %v2003 : tensor<256x2048x7x7xf32>
    %v2009 = stablehlo.multiply %v1998, %v2006 : tensor<256x2048x7x7xf32>
    %v2010 = stablehlo.subtract %v2008, %v2009 : tensor<256x2048x7x7xf32>
    %v2011 = stablehlo.divide %v1997, %v1986 : tensor<256x2048x7x7xf32>
    %v2012 = stablehlo.multiply %v2011, %v2010 : tensor<256x2048x7x7xf32>
    %v2013 = stablehlo.reshape %v2012 : (tensor<256x2048x7x7xf32>) -> tensor<256x100352xf32>
    %v2014 = stablehlo.reshape %v2013 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2015 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2016 = stablehlo.pad %v2014, %v2015, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048x14x14xf32>
    %v2017 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2018 = stablehlo.transpose %v2017, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2019 = stablehlo.convolution(%v2016, %v2018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x2048x14x14xf32>, tensor<1024x2048x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2020 = stablehlo.reshape %v2019 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2021 = stablehlo.add %v1983, %v2020 : tensor<256x200704xf32>
    %v2022 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2023 = stablehlo.reshape %v1978 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2024 = stablehlo.transpose %v2022, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2025 = stablehlo.transpose %v2023, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2026 = stablehlo.convolution(%v2024, %v2025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<1024x512x1x1xf32>
    %v2027 = stablehlo.transpose %v2026, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2028 = stablehlo.reshape %v1176 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2030 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v2031 = stablehlo.reduce(%v2028 init: %v2029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2032 = stablehlo.broadcast_in_dim %v2031, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2033 = stablehlo.divide %v2032, %v2030 : tensor<256x512x14x14xf32>
    %v2034 = stablehlo.subtract %v2028, %v2033 : tensor<256x512x14x14xf32>
    %v2035 = stablehlo.multiply %v2034, %v2034 : tensor<256x512x14x14xf32>
    %v2036 = stablehlo.reduce(%v2035 init: %v2029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2037 = stablehlo.broadcast_in_dim %v2036, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v2038 = stablehlo.divide %v2037, %v2030 : tensor<256x512x14x14xf32>
    %v2039 = stablehlo.constant dense<1.0e-05> : tensor<256x512x14x14xf32>
    %v2040 = stablehlo.add %v2038, %v2039 : tensor<256x512x14x14xf32>
    %v2041 = stablehlo.rsqrt %v2040 : tensor<256x512x14x14xf32>
    %v2042 = stablehlo.multiply %v2034, %v2041 : tensor<256x512x14x14xf32>
    %v2043 = stablehlo.reshape %v1948 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2044 = stablehlo.multiply %v2043, %v2042 : tensor<256x512x14x14xf32>
    %v2045 = stablehlo.reduce(%v2044 init: %v2029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2046 = stablehlo.reshape %v1948 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2048 = stablehlo.reduce(%v2046 init: %v2047) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2049 = stablehlo.reshape %v1198 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v2050 = stablehlo.reshape %v1938 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2051 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2052 = stablehlo.pad %v2050, %v2051, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<256x512x14x14xf32>
    %v2053 = stablehlo.transpose %v2049, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2054 = stablehlo.transpose %v2052, dims = [1, 0, 2, 3] : (tensor<256x512x14x14xf32>) -> tensor<512x256x14x14xf32>
    %v2055 = stablehlo.convolution(%v2053, %v2054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x14x14xf32>, tensor<512x256x14x14xf32>) -> tensor<512x512x3x3xf32>
    %v2056 = stablehlo.transpose %v2055, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2057 = stablehlo.reshape %v1203 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2059 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v2060 = stablehlo.reduce(%v2057 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2061 = stablehlo.broadcast_in_dim %v2060, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2062 = stablehlo.divide %v2061, %v2059 : tensor<256x512x7x7xf32>
    %v2063 = stablehlo.subtract %v2057, %v2062 : tensor<256x512x7x7xf32>
    %v2064 = stablehlo.multiply %v2063, %v2063 : tensor<256x512x7x7xf32>
    %v2065 = stablehlo.reduce(%v2064 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2066 = stablehlo.broadcast_in_dim %v2065, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v2067 = stablehlo.divide %v2066, %v2059 : tensor<256x512x7x7xf32>
    %v2068 = stablehlo.constant dense<1.0e-05> : tensor<256x512x7x7xf32>
    %v2069 = stablehlo.add %v2067, %v2068 : tensor<256x512x7x7xf32>
    %v2070 = stablehlo.rsqrt %v2069 : tensor<256x512x7x7xf32>
    %v2071 = stablehlo.multiply %v2063, %v2070 : tensor<256x512x7x7xf32>
    %v2072 = stablehlo.reshape %v1908 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2073 = stablehlo.multiply %v2072, %v2071 : tensor<256x512x7x7xf32>
    %v2074 = stablehlo.reduce(%v2073 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2075 = stablehlo.reshape %v1908 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2077 = stablehlo.reduce(%v2075 init: %v2076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2078 = stablehlo.reshape %v1225 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v2079 = stablehlo.reshape %v1900 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2080 = stablehlo.transpose %v2078, dims = [1, 0, 2, 3] : (tensor<256x512x7x7xf32>) -> tensor<512x256x7x7xf32>
    %v2081 = stablehlo.transpose %v2079, dims = [1, 0, 2, 3] : (tensor<256x2048x7x7xf32>) -> tensor<2048x256x7x7xf32>
    %v2082 = stablehlo.convolution(%v2080, %v2081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x7x7xf32>, tensor<2048x256x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2083 = stablehlo.transpose %v2082, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2084 = stablehlo.reshape %v1230 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2086 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2087 = stablehlo.reduce(%v2084 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2088 = stablehlo.broadcast_in_dim %v2087, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2089 = stablehlo.divide %v2088, %v2086 : tensor<256x2048x7x7xf32>
    %v2090 = stablehlo.subtract %v2084, %v2089 : tensor<256x2048x7x7xf32>
    %v2091 = stablehlo.multiply %v2090, %v2090 : tensor<256x2048x7x7xf32>
    %v2092 = stablehlo.reduce(%v2091 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2093 = stablehlo.broadcast_in_dim %v2092, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2094 = stablehlo.divide %v2093, %v2086 : tensor<256x2048x7x7xf32>
    %v2095 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2096 = stablehlo.add %v2094, %v2095 : tensor<256x2048x7x7xf32>
    %v2097 = stablehlo.rsqrt %v2096 : tensor<256x2048x7x7xf32>
    %v2098 = stablehlo.multiply %v2090, %v2097 : tensor<256x2048x7x7xf32>
    %v2099 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2100 = stablehlo.multiply %v2099, %v2098 : tensor<256x2048x7x7xf32>
    %v2101 = stablehlo.reduce(%v2100 init: %v2085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2102 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2104 = stablehlo.reduce(%v2102 init: %v2103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2105 = stablehlo.reshape %v1171 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2106 = stablehlo.reshape %v2013 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2107 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2108 = stablehlo.pad %v2106, %v2107, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<256x2048x14x14xf32>
    %v2109 = stablehlo.transpose %v2105, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2110 = stablehlo.transpose %v2108, dims = [1, 0, 2, 3] : (tensor<256x2048x14x14xf32>) -> tensor<2048x256x14x14xf32>
    %v2111 = stablehlo.convolution(%v2109, %v2110)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<2048x256x14x14xf32>) -> tensor<1024x2048x1x1xf32>
    %v2112 = stablehlo.transpose %v2111, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v2113 = stablehlo.reshape %v1255 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2115 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v2116 = stablehlo.reduce(%v2113 init: %v2114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2117 = stablehlo.broadcast_in_dim %v2116, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2118 = stablehlo.divide %v2117, %v2115 : tensor<256x2048x7x7xf32>
    %v2119 = stablehlo.subtract %v2113, %v2118 : tensor<256x2048x7x7xf32>
    %v2120 = stablehlo.multiply %v2119, %v2119 : tensor<256x2048x7x7xf32>
    %v2121 = stablehlo.reduce(%v2120 init: %v2114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2122 = stablehlo.broadcast_in_dim %v2121, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v2123 = stablehlo.divide %v2122, %v2115 : tensor<256x2048x7x7xf32>
    %v2124 = stablehlo.constant dense<1.0e-05> : tensor<256x2048x7x7xf32>
    %v2125 = stablehlo.add %v2123, %v2124 : tensor<256x2048x7x7xf32>
    %v2126 = stablehlo.rsqrt %v2125 : tensor<256x2048x7x7xf32>
    %v2127 = stablehlo.multiply %v2119, %v2126 : tensor<256x2048x7x7xf32>
    %v2128 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2129 = stablehlo.multiply %v2128, %v2127 : tensor<256x2048x7x7xf32>
    %v2130 = stablehlo.reduce(%v2129 init: %v2114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2131 = stablehlo.reshape %v1870 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v2132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2133 = stablehlo.reduce(%v2131 init: %v2132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2134 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2135 = stablehlo.compare GT, %v1169, %v2134 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2136 = stablehlo.select %v2135, %v2021, %v2134 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2137 = stablehlo.reshape %v1148 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2140 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2141 = stablehlo.reduce(%v2137 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2142 = stablehlo.broadcast_in_dim %v2141, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2143 = stablehlo.divide %v2142, %v2139 : tensor<256x1024x14x14xf32>
    %v2144 = stablehlo.subtract %v2137, %v2143 : tensor<256x1024x14x14xf32>
    %v2145 = stablehlo.multiply %v2144, %v2144 : tensor<256x1024x14x14xf32>
    %v2146 = stablehlo.reduce(%v2145 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2147 = stablehlo.broadcast_in_dim %v2146, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2148 = stablehlo.divide %v2147, %v2139 : tensor<256x1024x14x14xf32>
    %v2149 = stablehlo.add %v2148, %v2140 : tensor<256x1024x14x14xf32>
    %v2150 = stablehlo.rsqrt %v2149 : tensor<256x1024x14x14xf32>
    %v2151 = stablehlo.multiply %v2144, %v2150 : tensor<256x1024x14x14xf32>
    %v2152 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2153 = stablehlo.reshape %v2136 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2154 = stablehlo.multiply %v2152, %v2153 : tensor<256x1024x14x14xf32>
    %v2155 = stablehlo.reduce(%v2154 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2156 = stablehlo.broadcast_in_dim %v2155, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2157 = stablehlo.multiply %v2151, %v2154 : tensor<256x1024x14x14xf32>
    %v2158 = stablehlo.reduce(%v2157 init: %v2138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2158, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2160 = stablehlo.multiply %v2154, %v2139 : tensor<256x1024x14x14xf32>
    %v2161 = stablehlo.subtract %v2160, %v2156 : tensor<256x1024x14x14xf32>
    %v2162 = stablehlo.multiply %v2151, %v2159 : tensor<256x1024x14x14xf32>
    %v2163 = stablehlo.subtract %v2161, %v2162 : tensor<256x1024x14x14xf32>
    %v2164 = stablehlo.divide %v2150, %v2139 : tensor<256x1024x14x14xf32>
    %v2165 = stablehlo.multiply %v2164, %v2163 : tensor<256x1024x14x14xf32>
    %v2166 = stablehlo.reshape %v2165 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2167 = stablehlo.reshape %v2166 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2168 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2169 = stablehlo.transpose %v2168, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2170 = stablehlo.convolution(%v2167, %v2169)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2171 = stablehlo.reshape %v2170 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2172 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2173 = stablehlo.compare GT, %v1141, %v2172 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2174 = stablehlo.select %v2173, %v2171, %v2172 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2175 = stablehlo.reshape %v1121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2177 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2178 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2179 = stablehlo.reduce(%v2175 init: %v2176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2180 = stablehlo.broadcast_in_dim %v2179, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2181 = stablehlo.divide %v2180, %v2177 : tensor<256x256x14x14xf32>
    %v2182 = stablehlo.subtract %v2175, %v2181 : tensor<256x256x14x14xf32>
    %v2183 = stablehlo.multiply %v2182, %v2182 : tensor<256x256x14x14xf32>
    %v2184 = stablehlo.reduce(%v2183 init: %v2176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2184, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2186 = stablehlo.divide %v2185, %v2177 : tensor<256x256x14x14xf32>
    %v2187 = stablehlo.add %v2186, %v2178 : tensor<256x256x14x14xf32>
    %v2188 = stablehlo.rsqrt %v2187 : tensor<256x256x14x14xf32>
    %v2189 = stablehlo.multiply %v2182, %v2188 : tensor<256x256x14x14xf32>
    %v2190 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2191 = stablehlo.reshape %v2174 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2192 = stablehlo.multiply %v2190, %v2191 : tensor<256x256x14x14xf32>
    %v2193 = stablehlo.reduce(%v2192 init: %v2176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2194 = stablehlo.broadcast_in_dim %v2193, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2195 = stablehlo.multiply %v2189, %v2192 : tensor<256x256x14x14xf32>
    %v2196 = stablehlo.reduce(%v2195 init: %v2176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2197 = stablehlo.broadcast_in_dim %v2196, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2198 = stablehlo.multiply %v2192, %v2177 : tensor<256x256x14x14xf32>
    %v2199 = stablehlo.subtract %v2198, %v2194 : tensor<256x256x14x14xf32>
    %v2200 = stablehlo.multiply %v2189, %v2197 : tensor<256x256x14x14xf32>
    %v2201 = stablehlo.subtract %v2199, %v2200 : tensor<256x256x14x14xf32>
    %v2202 = stablehlo.divide %v2188, %v2177 : tensor<256x256x14x14xf32>
    %v2203 = stablehlo.multiply %v2202, %v2201 : tensor<256x256x14x14xf32>
    %v2204 = stablehlo.reshape %v2203 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2205 = stablehlo.reshape %v2204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2206 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2207 = stablehlo.transpose %v2206, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2208 = stablehlo.convolution(%v2205, %v2207)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2209 = stablehlo.reshape %v2208 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2210 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2211 = stablehlo.compare GT, %v1114, %v2210 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2212 = stablehlo.select %v2211, %v2209, %v2210 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2213 = stablehlo.reshape %v1094 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2215 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2216 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2217 = stablehlo.reduce(%v2213 init: %v2214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2218 = stablehlo.broadcast_in_dim %v2217, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2219 = stablehlo.divide %v2218, %v2215 : tensor<256x256x14x14xf32>
    %v2220 = stablehlo.subtract %v2213, %v2219 : tensor<256x256x14x14xf32>
    %v2221 = stablehlo.multiply %v2220, %v2220 : tensor<256x256x14x14xf32>
    %v2222 = stablehlo.reduce(%v2221 init: %v2214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2223 = stablehlo.broadcast_in_dim %v2222, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2224 = stablehlo.divide %v2223, %v2215 : tensor<256x256x14x14xf32>
    %v2225 = stablehlo.add %v2224, %v2216 : tensor<256x256x14x14xf32>
    %v2226 = stablehlo.rsqrt %v2225 : tensor<256x256x14x14xf32>
    %v2227 = stablehlo.multiply %v2220, %v2226 : tensor<256x256x14x14xf32>
    %v2228 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2229 = stablehlo.reshape %v2212 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2230 = stablehlo.multiply %v2228, %v2229 : tensor<256x256x14x14xf32>
    %v2231 = stablehlo.reduce(%v2230 init: %v2214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2232 = stablehlo.broadcast_in_dim %v2231, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2233 = stablehlo.multiply %v2227, %v2230 : tensor<256x256x14x14xf32>
    %v2234 = stablehlo.reduce(%v2233 init: %v2214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2235 = stablehlo.broadcast_in_dim %v2234, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2236 = stablehlo.multiply %v2230, %v2215 : tensor<256x256x14x14xf32>
    %v2237 = stablehlo.subtract %v2236, %v2232 : tensor<256x256x14x14xf32>
    %v2238 = stablehlo.multiply %v2227, %v2235 : tensor<256x256x14x14xf32>
    %v2239 = stablehlo.subtract %v2237, %v2238 : tensor<256x256x14x14xf32>
    %v2240 = stablehlo.divide %v2226, %v2215 : tensor<256x256x14x14xf32>
    %v2241 = stablehlo.multiply %v2240, %v2239 : tensor<256x256x14x14xf32>
    %v2242 = stablehlo.reshape %v2241 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2243 = stablehlo.reshape %v2242 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2244 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2245 = stablehlo.transpose %v2244, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2246 = stablehlo.convolution(%v2243, %v2245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2247 = stablehlo.reshape %v2246 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2248 = stablehlo.add %v2247, %v2136 : tensor<256x200704xf32>
    %v2249 = stablehlo.reshape %v1089 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2250 = stablehlo.reshape %v2242 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2251 = stablehlo.transpose %v2249, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2252 = stablehlo.transpose %v2250, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2253 = stablehlo.convolution(%v2251, %v2252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2254 = stablehlo.transpose %v2253, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2255 = stablehlo.reshape %v1094 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2257 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2258 = stablehlo.reduce(%v2255 init: %v2256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2259 = stablehlo.broadcast_in_dim %v2258, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2260 = stablehlo.divide %v2259, %v2257 : tensor<256x256x14x14xf32>
    %v2261 = stablehlo.subtract %v2255, %v2260 : tensor<256x256x14x14xf32>
    %v2262 = stablehlo.multiply %v2261, %v2261 : tensor<256x256x14x14xf32>
    %v2263 = stablehlo.reduce(%v2262 init: %v2256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2264 = stablehlo.broadcast_in_dim %v2263, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2265 = stablehlo.divide %v2264, %v2257 : tensor<256x256x14x14xf32>
    %v2266 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2267 = stablehlo.add %v2265, %v2266 : tensor<256x256x14x14xf32>
    %v2268 = stablehlo.rsqrt %v2267 : tensor<256x256x14x14xf32>
    %v2269 = stablehlo.multiply %v2261, %v2268 : tensor<256x256x14x14xf32>
    %v2270 = stablehlo.reshape %v2212 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2271 = stablehlo.multiply %v2270, %v2269 : tensor<256x256x14x14xf32>
    %v2272 = stablehlo.reduce(%v2271 init: %v2256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2273 = stablehlo.reshape %v2212 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2275 = stablehlo.reduce(%v2273 init: %v2274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2276 = stablehlo.reshape %v1116 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2277 = stablehlo.reshape %v2204 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2278 = stablehlo.transpose %v2276, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2279 = stablehlo.transpose %v2277, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2280 = stablehlo.convolution(%v2278, %v2279)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2281 = stablehlo.transpose %v2280, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2282 = stablehlo.reshape %v1121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2284 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2285 = stablehlo.reduce(%v2282 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2286 = stablehlo.broadcast_in_dim %v2285, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2287 = stablehlo.divide %v2286, %v2284 : tensor<256x256x14x14xf32>
    %v2288 = stablehlo.subtract %v2282, %v2287 : tensor<256x256x14x14xf32>
    %v2289 = stablehlo.multiply %v2288, %v2288 : tensor<256x256x14x14xf32>
    %v2290 = stablehlo.reduce(%v2289 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2291 = stablehlo.broadcast_in_dim %v2290, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2292 = stablehlo.divide %v2291, %v2284 : tensor<256x256x14x14xf32>
    %v2293 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2294 = stablehlo.add %v2292, %v2293 : tensor<256x256x14x14xf32>
    %v2295 = stablehlo.rsqrt %v2294 : tensor<256x256x14x14xf32>
    %v2296 = stablehlo.multiply %v2288, %v2295 : tensor<256x256x14x14xf32>
    %v2297 = stablehlo.reshape %v2174 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2298 = stablehlo.multiply %v2297, %v2296 : tensor<256x256x14x14xf32>
    %v2299 = stablehlo.reduce(%v2298 init: %v2283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2300 = stablehlo.reshape %v2174 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2302 = stablehlo.reduce(%v2300 init: %v2301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2303 = stablehlo.reshape %v1143 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2304 = stablehlo.reshape %v2166 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2305 = stablehlo.transpose %v2303, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2306 = stablehlo.transpose %v2304, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2307 = stablehlo.convolution(%v2305, %v2306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2308 = stablehlo.transpose %v2307, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2309 = stablehlo.reshape %v1148 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2311 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2312 = stablehlo.reduce(%v2309 init: %v2310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2313 = stablehlo.broadcast_in_dim %v2312, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2314 = stablehlo.divide %v2313, %v2311 : tensor<256x1024x14x14xf32>
    %v2315 = stablehlo.subtract %v2309, %v2314 : tensor<256x1024x14x14xf32>
    %v2316 = stablehlo.multiply %v2315, %v2315 : tensor<256x1024x14x14xf32>
    %v2317 = stablehlo.reduce(%v2316 init: %v2310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2318 = stablehlo.broadcast_in_dim %v2317, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2319 = stablehlo.divide %v2318, %v2311 : tensor<256x1024x14x14xf32>
    %v2320 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2321 = stablehlo.add %v2319, %v2320 : tensor<256x1024x14x14xf32>
    %v2322 = stablehlo.rsqrt %v2321 : tensor<256x1024x14x14xf32>
    %v2323 = stablehlo.multiply %v2315, %v2322 : tensor<256x1024x14x14xf32>
    %v2324 = stablehlo.reshape %v2136 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2325 = stablehlo.multiply %v2324, %v2323 : tensor<256x1024x14x14xf32>
    %v2326 = stablehlo.reduce(%v2325 init: %v2310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2327 = stablehlo.reshape %v2136 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2328 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2329 = stablehlo.reduce(%v2327 init: %v2328) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2331 = stablehlo.compare GT, %v1087, %v2330 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2332 = stablehlo.select %v2331, %v2248, %v2330 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2333 = stablehlo.reshape %v1066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2335 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2336 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2337 = stablehlo.reduce(%v2333 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2338 = stablehlo.broadcast_in_dim %v2337, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2339 = stablehlo.divide %v2338, %v2335 : tensor<256x1024x14x14xf32>
    %v2340 = stablehlo.subtract %v2333, %v2339 : tensor<256x1024x14x14xf32>
    %v2341 = stablehlo.multiply %v2340, %v2340 : tensor<256x1024x14x14xf32>
    %v2342 = stablehlo.reduce(%v2341 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2342, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2344 = stablehlo.divide %v2343, %v2335 : tensor<256x1024x14x14xf32>
    %v2345 = stablehlo.add %v2344, %v2336 : tensor<256x1024x14x14xf32>
    %v2346 = stablehlo.rsqrt %v2345 : tensor<256x1024x14x14xf32>
    %v2347 = stablehlo.multiply %v2340, %v2346 : tensor<256x1024x14x14xf32>
    %v2348 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2349 = stablehlo.reshape %v2332 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2350 = stablehlo.multiply %v2348, %v2349 : tensor<256x1024x14x14xf32>
    %v2351 = stablehlo.reduce(%v2350 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2352 = stablehlo.broadcast_in_dim %v2351, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2353 = stablehlo.multiply %v2347, %v2350 : tensor<256x1024x14x14xf32>
    %v2354 = stablehlo.reduce(%v2353 init: %v2334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2356 = stablehlo.multiply %v2350, %v2335 : tensor<256x1024x14x14xf32>
    %v2357 = stablehlo.subtract %v2356, %v2352 : tensor<256x1024x14x14xf32>
    %v2358 = stablehlo.multiply %v2347, %v2355 : tensor<256x1024x14x14xf32>
    %v2359 = stablehlo.subtract %v2357, %v2358 : tensor<256x1024x14x14xf32>
    %v2360 = stablehlo.divide %v2346, %v2335 : tensor<256x1024x14x14xf32>
    %v2361 = stablehlo.multiply %v2360, %v2359 : tensor<256x1024x14x14xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2363 = stablehlo.reshape %v2362 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2364 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2365 = stablehlo.transpose %v2364, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2366 = stablehlo.convolution(%v2363, %v2365)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2367 = stablehlo.reshape %v2366 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2369 = stablehlo.compare GT, %v1059, %v2368 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2370 = stablehlo.select %v2369, %v2367, %v2368 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2371 = stablehlo.reshape %v1039 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2372 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2373 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2374 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2375 = stablehlo.reduce(%v2371 init: %v2372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2376 = stablehlo.broadcast_in_dim %v2375, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2377 = stablehlo.divide %v2376, %v2373 : tensor<256x256x14x14xf32>
    %v2378 = stablehlo.subtract %v2371, %v2377 : tensor<256x256x14x14xf32>
    %v2379 = stablehlo.multiply %v2378, %v2378 : tensor<256x256x14x14xf32>
    %v2380 = stablehlo.reduce(%v2379 init: %v2372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2381 = stablehlo.broadcast_in_dim %v2380, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2382 = stablehlo.divide %v2381, %v2373 : tensor<256x256x14x14xf32>
    %v2383 = stablehlo.add %v2382, %v2374 : tensor<256x256x14x14xf32>
    %v2384 = stablehlo.rsqrt %v2383 : tensor<256x256x14x14xf32>
    %v2385 = stablehlo.multiply %v2378, %v2384 : tensor<256x256x14x14xf32>
    %v2386 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2387 = stablehlo.reshape %v2370 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2388 = stablehlo.multiply %v2386, %v2387 : tensor<256x256x14x14xf32>
    %v2389 = stablehlo.reduce(%v2388 init: %v2372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2390 = stablehlo.broadcast_in_dim %v2389, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2391 = stablehlo.multiply %v2385, %v2388 : tensor<256x256x14x14xf32>
    %v2392 = stablehlo.reduce(%v2391 init: %v2372) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2393 = stablehlo.broadcast_in_dim %v2392, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2394 = stablehlo.multiply %v2388, %v2373 : tensor<256x256x14x14xf32>
    %v2395 = stablehlo.subtract %v2394, %v2390 : tensor<256x256x14x14xf32>
    %v2396 = stablehlo.multiply %v2385, %v2393 : tensor<256x256x14x14xf32>
    %v2397 = stablehlo.subtract %v2395, %v2396 : tensor<256x256x14x14xf32>
    %v2398 = stablehlo.divide %v2384, %v2373 : tensor<256x256x14x14xf32>
    %v2399 = stablehlo.multiply %v2398, %v2397 : tensor<256x256x14x14xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2401 = stablehlo.reshape %v2400 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2402 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2403 = stablehlo.transpose %v2402, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2404 = stablehlo.convolution(%v2401, %v2403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2406 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2407 = stablehlo.compare GT, %v1032, %v2406 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2408 = stablehlo.select %v2407, %v2405, %v2406 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2409 = stablehlo.reshape %v1012 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2411 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2412 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2413 = stablehlo.reduce(%v2409 init: %v2410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2414 = stablehlo.broadcast_in_dim %v2413, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2415 = stablehlo.divide %v2414, %v2411 : tensor<256x256x14x14xf32>
    %v2416 = stablehlo.subtract %v2409, %v2415 : tensor<256x256x14x14xf32>
    %v2417 = stablehlo.multiply %v2416, %v2416 : tensor<256x256x14x14xf32>
    %v2418 = stablehlo.reduce(%v2417 init: %v2410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2419 = stablehlo.broadcast_in_dim %v2418, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2420 = stablehlo.divide %v2419, %v2411 : tensor<256x256x14x14xf32>
    %v2421 = stablehlo.add %v2420, %v2412 : tensor<256x256x14x14xf32>
    %v2422 = stablehlo.rsqrt %v2421 : tensor<256x256x14x14xf32>
    %v2423 = stablehlo.multiply %v2416, %v2422 : tensor<256x256x14x14xf32>
    %v2424 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2425 = stablehlo.reshape %v2408 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2426 = stablehlo.multiply %v2424, %v2425 : tensor<256x256x14x14xf32>
    %v2427 = stablehlo.reduce(%v2426 init: %v2410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2428 = stablehlo.broadcast_in_dim %v2427, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2429 = stablehlo.multiply %v2423, %v2426 : tensor<256x256x14x14xf32>
    %v2430 = stablehlo.reduce(%v2429 init: %v2410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2430, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2432 = stablehlo.multiply %v2426, %v2411 : tensor<256x256x14x14xf32>
    %v2433 = stablehlo.subtract %v2432, %v2428 : tensor<256x256x14x14xf32>
    %v2434 = stablehlo.multiply %v2423, %v2431 : tensor<256x256x14x14xf32>
    %v2435 = stablehlo.subtract %v2433, %v2434 : tensor<256x256x14x14xf32>
    %v2436 = stablehlo.divide %v2422, %v2411 : tensor<256x256x14x14xf32>
    %v2437 = stablehlo.multiply %v2436, %v2435 : tensor<256x256x14x14xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2439 = stablehlo.reshape %v2438 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2440 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2441 = stablehlo.transpose %v2440, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2442 = stablehlo.convolution(%v2439, %v2441)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2444 = stablehlo.add %v2443, %v2332 : tensor<256x200704xf32>
    %v2445 = stablehlo.reshape %v1007 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2446 = stablehlo.reshape %v2438 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2447 = stablehlo.transpose %v2445, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2448 = stablehlo.transpose %v2446, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2449 = stablehlo.convolution(%v2447, %v2448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2450 = stablehlo.transpose %v2449, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2451 = stablehlo.reshape %v1012 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2452 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2453 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2454 = stablehlo.reduce(%v2451 init: %v2452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2455 = stablehlo.broadcast_in_dim %v2454, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2456 = stablehlo.divide %v2455, %v2453 : tensor<256x256x14x14xf32>
    %v2457 = stablehlo.subtract %v2451, %v2456 : tensor<256x256x14x14xf32>
    %v2458 = stablehlo.multiply %v2457, %v2457 : tensor<256x256x14x14xf32>
    %v2459 = stablehlo.reduce(%v2458 init: %v2452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2461 = stablehlo.divide %v2460, %v2453 : tensor<256x256x14x14xf32>
    %v2462 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2463 = stablehlo.add %v2461, %v2462 : tensor<256x256x14x14xf32>
    %v2464 = stablehlo.rsqrt %v2463 : tensor<256x256x14x14xf32>
    %v2465 = stablehlo.multiply %v2457, %v2464 : tensor<256x256x14x14xf32>
    %v2466 = stablehlo.reshape %v2408 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2467 = stablehlo.multiply %v2466, %v2465 : tensor<256x256x14x14xf32>
    %v2468 = stablehlo.reduce(%v2467 init: %v2452) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2469 = stablehlo.reshape %v2408 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2471 = stablehlo.reduce(%v2469 init: %v2470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2472 = stablehlo.reshape %v1034 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2473 = stablehlo.reshape %v2400 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2474 = stablehlo.transpose %v2472, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2475 = stablehlo.transpose %v2473, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2476 = stablehlo.convolution(%v2474, %v2475)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2477 = stablehlo.transpose %v2476, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2478 = stablehlo.reshape %v1039 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2480 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2481 = stablehlo.reduce(%v2478 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2482 = stablehlo.broadcast_in_dim %v2481, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2483 = stablehlo.divide %v2482, %v2480 : tensor<256x256x14x14xf32>
    %v2484 = stablehlo.subtract %v2478, %v2483 : tensor<256x256x14x14xf32>
    %v2485 = stablehlo.multiply %v2484, %v2484 : tensor<256x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2485 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2488 = stablehlo.divide %v2487, %v2480 : tensor<256x256x14x14xf32>
    %v2489 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2490 = stablehlo.add %v2488, %v2489 : tensor<256x256x14x14xf32>
    %v2491 = stablehlo.rsqrt %v2490 : tensor<256x256x14x14xf32>
    %v2492 = stablehlo.multiply %v2484, %v2491 : tensor<256x256x14x14xf32>
    %v2493 = stablehlo.reshape %v2370 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2494 = stablehlo.multiply %v2493, %v2492 : tensor<256x256x14x14xf32>
    %v2495 = stablehlo.reduce(%v2494 init: %v2479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2496 = stablehlo.reshape %v2370 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2497 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2498 = stablehlo.reduce(%v2496 init: %v2497) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2499 = stablehlo.reshape %v1061 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2500 = stablehlo.reshape %v2362 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2501 = stablehlo.transpose %v2499, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2502 = stablehlo.transpose %v2500, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2503 = stablehlo.convolution(%v2501, %v2502)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2504 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2505 = stablehlo.reshape %v1066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2507 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2508 = stablehlo.reduce(%v2505 init: %v2506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2509 = stablehlo.broadcast_in_dim %v2508, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2510 = stablehlo.divide %v2509, %v2507 : tensor<256x1024x14x14xf32>
    %v2511 = stablehlo.subtract %v2505, %v2510 : tensor<256x1024x14x14xf32>
    %v2512 = stablehlo.multiply %v2511, %v2511 : tensor<256x1024x14x14xf32>
    %v2513 = stablehlo.reduce(%v2512 init: %v2506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2514 = stablehlo.broadcast_in_dim %v2513, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2515 = stablehlo.divide %v2514, %v2507 : tensor<256x1024x14x14xf32>
    %v2516 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2517 = stablehlo.add %v2515, %v2516 : tensor<256x1024x14x14xf32>
    %v2518 = stablehlo.rsqrt %v2517 : tensor<256x1024x14x14xf32>
    %v2519 = stablehlo.multiply %v2511, %v2518 : tensor<256x1024x14x14xf32>
    %v2520 = stablehlo.reshape %v2332 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2521 = stablehlo.multiply %v2520, %v2519 : tensor<256x1024x14x14xf32>
    %v2522 = stablehlo.reduce(%v2521 init: %v2506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2523 = stablehlo.reshape %v2332 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2525 = stablehlo.reduce(%v2523 init: %v2524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2526 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2527 = stablehlo.compare GT, %v1005, %v2526 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2528 = stablehlo.select %v2527, %v2444, %v2526 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2529 = stablehlo.reshape %v984 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2531 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2532 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2533 = stablehlo.reduce(%v2529 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2534 = stablehlo.broadcast_in_dim %v2533, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2535 = stablehlo.divide %v2534, %v2531 : tensor<256x1024x14x14xf32>
    %v2536 = stablehlo.subtract %v2529, %v2535 : tensor<256x1024x14x14xf32>
    %v2537 = stablehlo.multiply %v2536, %v2536 : tensor<256x1024x14x14xf32>
    %v2538 = stablehlo.reduce(%v2537 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2539 = stablehlo.broadcast_in_dim %v2538, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2540 = stablehlo.divide %v2539, %v2531 : tensor<256x1024x14x14xf32>
    %v2541 = stablehlo.add %v2540, %v2532 : tensor<256x1024x14x14xf32>
    %v2542 = stablehlo.rsqrt %v2541 : tensor<256x1024x14x14xf32>
    %v2543 = stablehlo.multiply %v2536, %v2542 : tensor<256x1024x14x14xf32>
    %v2544 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2545 = stablehlo.reshape %v2528 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2546 = stablehlo.multiply %v2544, %v2545 : tensor<256x1024x14x14xf32>
    %v2547 = stablehlo.reduce(%v2546 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2548 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2549 = stablehlo.multiply %v2543, %v2546 : tensor<256x1024x14x14xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2550, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2552 = stablehlo.multiply %v2546, %v2531 : tensor<256x1024x14x14xf32>
    %v2553 = stablehlo.subtract %v2552, %v2548 : tensor<256x1024x14x14xf32>
    %v2554 = stablehlo.multiply %v2543, %v2551 : tensor<256x1024x14x14xf32>
    %v2555 = stablehlo.subtract %v2553, %v2554 : tensor<256x1024x14x14xf32>
    %v2556 = stablehlo.divide %v2542, %v2531 : tensor<256x1024x14x14xf32>
    %v2557 = stablehlo.multiply %v2556, %v2555 : tensor<256x1024x14x14xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2559 = stablehlo.reshape %v2558 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2560 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2561 = stablehlo.transpose %v2560, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2562 = stablehlo.convolution(%v2559, %v2561)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2563 = stablehlo.reshape %v2562 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2565 = stablehlo.compare GT, %v977, %v2564 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2566 = stablehlo.select %v2565, %v2563, %v2564 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2567 = stablehlo.reshape %v957 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2569 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2570 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2571 = stablehlo.reduce(%v2567 init: %v2568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2572 = stablehlo.broadcast_in_dim %v2571, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2573 = stablehlo.divide %v2572, %v2569 : tensor<256x256x14x14xf32>
    %v2574 = stablehlo.subtract %v2567, %v2573 : tensor<256x256x14x14xf32>
    %v2575 = stablehlo.multiply %v2574, %v2574 : tensor<256x256x14x14xf32>
    %v2576 = stablehlo.reduce(%v2575 init: %v2568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2577 = stablehlo.broadcast_in_dim %v2576, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2578 = stablehlo.divide %v2577, %v2569 : tensor<256x256x14x14xf32>
    %v2579 = stablehlo.add %v2578, %v2570 : tensor<256x256x14x14xf32>
    %v2580 = stablehlo.rsqrt %v2579 : tensor<256x256x14x14xf32>
    %v2581 = stablehlo.multiply %v2574, %v2580 : tensor<256x256x14x14xf32>
    %v2582 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2583 = stablehlo.reshape %v2566 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2584 = stablehlo.multiply %v2582, %v2583 : tensor<256x256x14x14xf32>
    %v2585 = stablehlo.reduce(%v2584 init: %v2568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2585, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2587 = stablehlo.multiply %v2581, %v2584 : tensor<256x256x14x14xf32>
    %v2588 = stablehlo.reduce(%v2587 init: %v2568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2589 = stablehlo.broadcast_in_dim %v2588, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2590 = stablehlo.multiply %v2584, %v2569 : tensor<256x256x14x14xf32>
    %v2591 = stablehlo.subtract %v2590, %v2586 : tensor<256x256x14x14xf32>
    %v2592 = stablehlo.multiply %v2581, %v2589 : tensor<256x256x14x14xf32>
    %v2593 = stablehlo.subtract %v2591, %v2592 : tensor<256x256x14x14xf32>
    %v2594 = stablehlo.divide %v2580, %v2569 : tensor<256x256x14x14xf32>
    %v2595 = stablehlo.multiply %v2594, %v2593 : tensor<256x256x14x14xf32>
    %v2596 = stablehlo.reshape %v2595 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2597 = stablehlo.reshape %v2596 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2598 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2599 = stablehlo.transpose %v2598, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2600 = stablehlo.convolution(%v2597, %v2599)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2601 = stablehlo.reshape %v2600 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2602 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2603 = stablehlo.compare GT, %v950, %v2602 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2604 = stablehlo.select %v2603, %v2601, %v2602 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2605 = stablehlo.reshape %v930 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2607 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2608 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2609 = stablehlo.reduce(%v2605 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2610 = stablehlo.broadcast_in_dim %v2609, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2611 = stablehlo.divide %v2610, %v2607 : tensor<256x256x14x14xf32>
    %v2612 = stablehlo.subtract %v2605, %v2611 : tensor<256x256x14x14xf32>
    %v2613 = stablehlo.multiply %v2612, %v2612 : tensor<256x256x14x14xf32>
    %v2614 = stablehlo.reduce(%v2613 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2615 = stablehlo.broadcast_in_dim %v2614, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2616 = stablehlo.divide %v2615, %v2607 : tensor<256x256x14x14xf32>
    %v2617 = stablehlo.add %v2616, %v2608 : tensor<256x256x14x14xf32>
    %v2618 = stablehlo.rsqrt %v2617 : tensor<256x256x14x14xf32>
    %v2619 = stablehlo.multiply %v2612, %v2618 : tensor<256x256x14x14xf32>
    %v2620 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2621 = stablehlo.reshape %v2604 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2622 = stablehlo.multiply %v2620, %v2621 : tensor<256x256x14x14xf32>
    %v2623 = stablehlo.reduce(%v2622 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2624 = stablehlo.broadcast_in_dim %v2623, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2625 = stablehlo.multiply %v2619, %v2622 : tensor<256x256x14x14xf32>
    %v2626 = stablehlo.reduce(%v2625 init: %v2606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2627 = stablehlo.broadcast_in_dim %v2626, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2628 = stablehlo.multiply %v2622, %v2607 : tensor<256x256x14x14xf32>
    %v2629 = stablehlo.subtract %v2628, %v2624 : tensor<256x256x14x14xf32>
    %v2630 = stablehlo.multiply %v2619, %v2627 : tensor<256x256x14x14xf32>
    %v2631 = stablehlo.subtract %v2629, %v2630 : tensor<256x256x14x14xf32>
    %v2632 = stablehlo.divide %v2618, %v2607 : tensor<256x256x14x14xf32>
    %v2633 = stablehlo.multiply %v2632, %v2631 : tensor<256x256x14x14xf32>
    %v2634 = stablehlo.reshape %v2633 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2635 = stablehlo.reshape %v2634 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2636 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2637 = stablehlo.transpose %v2636, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2638 = stablehlo.convolution(%v2635, %v2637)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2640 = stablehlo.add %v2639, %v2528 : tensor<256x200704xf32>
    %v2641 = stablehlo.reshape %v925 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2642 = stablehlo.reshape %v2634 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2643 = stablehlo.transpose %v2641, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2644 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2645 = stablehlo.convolution(%v2643, %v2644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2646 = stablehlo.transpose %v2645, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2647 = stablehlo.reshape %v930 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2649 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2650 = stablehlo.reduce(%v2647 init: %v2648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2652 = stablehlo.divide %v2651, %v2649 : tensor<256x256x14x14xf32>
    %v2653 = stablehlo.subtract %v2647, %v2652 : tensor<256x256x14x14xf32>
    %v2654 = stablehlo.multiply %v2653, %v2653 : tensor<256x256x14x14xf32>
    %v2655 = stablehlo.reduce(%v2654 init: %v2648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2656 = stablehlo.broadcast_in_dim %v2655, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2657 = stablehlo.divide %v2656, %v2649 : tensor<256x256x14x14xf32>
    %v2658 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2659 = stablehlo.add %v2657, %v2658 : tensor<256x256x14x14xf32>
    %v2660 = stablehlo.rsqrt %v2659 : tensor<256x256x14x14xf32>
    %v2661 = stablehlo.multiply %v2653, %v2660 : tensor<256x256x14x14xf32>
    %v2662 = stablehlo.reshape %v2604 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2663 = stablehlo.multiply %v2662, %v2661 : tensor<256x256x14x14xf32>
    %v2664 = stablehlo.reduce(%v2663 init: %v2648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2665 = stablehlo.reshape %v2604 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2667 = stablehlo.reduce(%v2665 init: %v2666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2668 = stablehlo.reshape %v952 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2669 = stablehlo.reshape %v2596 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2670 = stablehlo.transpose %v2668, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2671 = stablehlo.transpose %v2669, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2672 = stablehlo.convolution(%v2670, %v2671)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2673 = stablehlo.transpose %v2672, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2674 = stablehlo.reshape %v957 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2676 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2677 = stablehlo.reduce(%v2674 init: %v2675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2678 = stablehlo.broadcast_in_dim %v2677, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2679 = stablehlo.divide %v2678, %v2676 : tensor<256x256x14x14xf32>
    %v2680 = stablehlo.subtract %v2674, %v2679 : tensor<256x256x14x14xf32>
    %v2681 = stablehlo.multiply %v2680, %v2680 : tensor<256x256x14x14xf32>
    %v2682 = stablehlo.reduce(%v2681 init: %v2675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2683 = stablehlo.broadcast_in_dim %v2682, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2684 = stablehlo.divide %v2683, %v2676 : tensor<256x256x14x14xf32>
    %v2685 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2686 = stablehlo.add %v2684, %v2685 : tensor<256x256x14x14xf32>
    %v2687 = stablehlo.rsqrt %v2686 : tensor<256x256x14x14xf32>
    %v2688 = stablehlo.multiply %v2680, %v2687 : tensor<256x256x14x14xf32>
    %v2689 = stablehlo.reshape %v2566 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2690 = stablehlo.multiply %v2689, %v2688 : tensor<256x256x14x14xf32>
    %v2691 = stablehlo.reduce(%v2690 init: %v2675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2692 = stablehlo.reshape %v2566 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2694 = stablehlo.reduce(%v2692 init: %v2693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2695 = stablehlo.reshape %v979 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2696 = stablehlo.reshape %v2558 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2697 = stablehlo.transpose %v2695, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2698 = stablehlo.transpose %v2696, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2699 = stablehlo.convolution(%v2697, %v2698)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2700 = stablehlo.transpose %v2699, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2701 = stablehlo.reshape %v984 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2703 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2704 = stablehlo.reduce(%v2701 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2705 = stablehlo.broadcast_in_dim %v2704, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2706 = stablehlo.divide %v2705, %v2703 : tensor<256x1024x14x14xf32>
    %v2707 = stablehlo.subtract %v2701, %v2706 : tensor<256x1024x14x14xf32>
    %v2708 = stablehlo.multiply %v2707, %v2707 : tensor<256x1024x14x14xf32>
    %v2709 = stablehlo.reduce(%v2708 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2710 = stablehlo.broadcast_in_dim %v2709, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2711 = stablehlo.divide %v2710, %v2703 : tensor<256x1024x14x14xf32>
    %v2712 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2713 = stablehlo.add %v2711, %v2712 : tensor<256x1024x14x14xf32>
    %v2714 = stablehlo.rsqrt %v2713 : tensor<256x1024x14x14xf32>
    %v2715 = stablehlo.multiply %v2707, %v2714 : tensor<256x1024x14x14xf32>
    %v2716 = stablehlo.reshape %v2528 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2717 = stablehlo.multiply %v2716, %v2715 : tensor<256x1024x14x14xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2719 = stablehlo.reshape %v2528 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2721 = stablehlo.reduce(%v2719 init: %v2720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2722 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2723 = stablehlo.compare GT, %v923, %v2722 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2724 = stablehlo.select %v2723, %v2640, %v2722 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2725 = stablehlo.reshape %v902 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2727 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2728 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2729 = stablehlo.reduce(%v2725 init: %v2726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2730 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2731 = stablehlo.divide %v2730, %v2727 : tensor<256x1024x14x14xf32>
    %v2732 = stablehlo.subtract %v2725, %v2731 : tensor<256x1024x14x14xf32>
    %v2733 = stablehlo.multiply %v2732, %v2732 : tensor<256x1024x14x14xf32>
    %v2734 = stablehlo.reduce(%v2733 init: %v2726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2734, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2736 = stablehlo.divide %v2735, %v2727 : tensor<256x1024x14x14xf32>
    %v2737 = stablehlo.add %v2736, %v2728 : tensor<256x1024x14x14xf32>
    %v2738 = stablehlo.rsqrt %v2737 : tensor<256x1024x14x14xf32>
    %v2739 = stablehlo.multiply %v2732, %v2738 : tensor<256x1024x14x14xf32>
    %v2740 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2741 = stablehlo.reshape %v2724 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2742 = stablehlo.multiply %v2740, %v2741 : tensor<256x1024x14x14xf32>
    %v2743 = stablehlo.reduce(%v2742 init: %v2726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2744 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2745 = stablehlo.multiply %v2739, %v2742 : tensor<256x1024x14x14xf32>
    %v2746 = stablehlo.reduce(%v2745 init: %v2726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2747 = stablehlo.broadcast_in_dim %v2746, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2748 = stablehlo.multiply %v2742, %v2727 : tensor<256x1024x14x14xf32>
    %v2749 = stablehlo.subtract %v2748, %v2744 : tensor<256x1024x14x14xf32>
    %v2750 = stablehlo.multiply %v2739, %v2747 : tensor<256x1024x14x14xf32>
    %v2751 = stablehlo.subtract %v2749, %v2750 : tensor<256x1024x14x14xf32>
    %v2752 = stablehlo.divide %v2738, %v2727 : tensor<256x1024x14x14xf32>
    %v2753 = stablehlo.multiply %v2752, %v2751 : tensor<256x1024x14x14xf32>
    %v2754 = stablehlo.reshape %v2753 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2755 = stablehlo.reshape %v2754 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2756 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2757 = stablehlo.transpose %v2756, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2758 = stablehlo.convolution(%v2755, %v2757)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2760 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2761 = stablehlo.compare GT, %v895, %v2760 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2762 = stablehlo.select %v2761, %v2759, %v2760 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2763 = stablehlo.reshape %v875 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2765 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2766 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2767 = stablehlo.reduce(%v2763 init: %v2764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2768 = stablehlo.broadcast_in_dim %v2767, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2769 = stablehlo.divide %v2768, %v2765 : tensor<256x256x14x14xf32>
    %v2770 = stablehlo.subtract %v2763, %v2769 : tensor<256x256x14x14xf32>
    %v2771 = stablehlo.multiply %v2770, %v2770 : tensor<256x256x14x14xf32>
    %v2772 = stablehlo.reduce(%v2771 init: %v2764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2773 = stablehlo.broadcast_in_dim %v2772, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2774 = stablehlo.divide %v2773, %v2765 : tensor<256x256x14x14xf32>
    %v2775 = stablehlo.add %v2774, %v2766 : tensor<256x256x14x14xf32>
    %v2776 = stablehlo.rsqrt %v2775 : tensor<256x256x14x14xf32>
    %v2777 = stablehlo.multiply %v2770, %v2776 : tensor<256x256x14x14xf32>
    %v2778 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2779 = stablehlo.reshape %v2762 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2780 = stablehlo.multiply %v2778, %v2779 : tensor<256x256x14x14xf32>
    %v2781 = stablehlo.reduce(%v2780 init: %v2764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2782 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2783 = stablehlo.multiply %v2777, %v2780 : tensor<256x256x14x14xf32>
    %v2784 = stablehlo.reduce(%v2783 init: %v2764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2785 = stablehlo.broadcast_in_dim %v2784, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2786 = stablehlo.multiply %v2780, %v2765 : tensor<256x256x14x14xf32>
    %v2787 = stablehlo.subtract %v2786, %v2782 : tensor<256x256x14x14xf32>
    %v2788 = stablehlo.multiply %v2777, %v2785 : tensor<256x256x14x14xf32>
    %v2789 = stablehlo.subtract %v2787, %v2788 : tensor<256x256x14x14xf32>
    %v2790 = stablehlo.divide %v2776, %v2765 : tensor<256x256x14x14xf32>
    %v2791 = stablehlo.multiply %v2790, %v2789 : tensor<256x256x14x14xf32>
    %v2792 = stablehlo.reshape %v2791 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2793 = stablehlo.reshape %v2792 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2794 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2795 = stablehlo.transpose %v2794, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2796 = stablehlo.convolution(%v2793, %v2795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2798 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2799 = stablehlo.compare GT, %v868, %v2798 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2800 = stablehlo.select %v2799, %v2797, %v2798 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2801 = stablehlo.reshape %v848 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2803 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2804 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2805 = stablehlo.reduce(%v2801 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2806 = stablehlo.broadcast_in_dim %v2805, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2807 = stablehlo.divide %v2806, %v2803 : tensor<256x256x14x14xf32>
    %v2808 = stablehlo.subtract %v2801, %v2807 : tensor<256x256x14x14xf32>
    %v2809 = stablehlo.multiply %v2808, %v2808 : tensor<256x256x14x14xf32>
    %v2810 = stablehlo.reduce(%v2809 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2811 = stablehlo.broadcast_in_dim %v2810, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2812 = stablehlo.divide %v2811, %v2803 : tensor<256x256x14x14xf32>
    %v2813 = stablehlo.add %v2812, %v2804 : tensor<256x256x14x14xf32>
    %v2814 = stablehlo.rsqrt %v2813 : tensor<256x256x14x14xf32>
    %v2815 = stablehlo.multiply %v2808, %v2814 : tensor<256x256x14x14xf32>
    %v2816 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2817 = stablehlo.reshape %v2800 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2818 = stablehlo.multiply %v2816, %v2817 : tensor<256x256x14x14xf32>
    %v2819 = stablehlo.reduce(%v2818 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2820 = stablehlo.broadcast_in_dim %v2819, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2821 = stablehlo.multiply %v2815, %v2818 : tensor<256x256x14x14xf32>
    %v2822 = stablehlo.reduce(%v2821 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2823 = stablehlo.broadcast_in_dim %v2822, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2824 = stablehlo.multiply %v2818, %v2803 : tensor<256x256x14x14xf32>
    %v2825 = stablehlo.subtract %v2824, %v2820 : tensor<256x256x14x14xf32>
    %v2826 = stablehlo.multiply %v2815, %v2823 : tensor<256x256x14x14xf32>
    %v2827 = stablehlo.subtract %v2825, %v2826 : tensor<256x256x14x14xf32>
    %v2828 = stablehlo.divide %v2814, %v2803 : tensor<256x256x14x14xf32>
    %v2829 = stablehlo.multiply %v2828, %v2827 : tensor<256x256x14x14xf32>
    %v2830 = stablehlo.reshape %v2829 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2831 = stablehlo.reshape %v2830 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2832 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2833 = stablehlo.transpose %v2832, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2834 = stablehlo.convolution(%v2831, %v2833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v2835 = stablehlo.reshape %v2834 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2836 = stablehlo.add %v2835, %v2724 : tensor<256x200704xf32>
    %v2837 = stablehlo.reshape %v843 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2838 = stablehlo.reshape %v2830 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2839 = stablehlo.transpose %v2837, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2840 = stablehlo.transpose %v2838, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2841 = stablehlo.convolution(%v2839, %v2840)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v2842 = stablehlo.transpose %v2841, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2843 = stablehlo.reshape %v848 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2844 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2845 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2846 = stablehlo.reduce(%v2843 init: %v2844) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2847 = stablehlo.broadcast_in_dim %v2846, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2848 = stablehlo.divide %v2847, %v2845 : tensor<256x256x14x14xf32>
    %v2849 = stablehlo.subtract %v2843, %v2848 : tensor<256x256x14x14xf32>
    %v2850 = stablehlo.multiply %v2849, %v2849 : tensor<256x256x14x14xf32>
    %v2851 = stablehlo.reduce(%v2850 init: %v2844) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2852 = stablehlo.broadcast_in_dim %v2851, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2853 = stablehlo.divide %v2852, %v2845 : tensor<256x256x14x14xf32>
    %v2854 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2855 = stablehlo.add %v2853, %v2854 : tensor<256x256x14x14xf32>
    %v2856 = stablehlo.rsqrt %v2855 : tensor<256x256x14x14xf32>
    %v2857 = stablehlo.multiply %v2849, %v2856 : tensor<256x256x14x14xf32>
    %v2858 = stablehlo.reshape %v2800 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2859 = stablehlo.multiply %v2858, %v2857 : tensor<256x256x14x14xf32>
    %v2860 = stablehlo.reduce(%v2859 init: %v2844) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2861 = stablehlo.reshape %v2800 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2862 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2863 = stablehlo.reduce(%v2861 init: %v2862) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2864 = stablehlo.reshape %v870 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2865 = stablehlo.reshape %v2792 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2866 = stablehlo.transpose %v2864, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2867 = stablehlo.transpose %v2865, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2868 = stablehlo.convolution(%v2866, %v2867)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v2869 = stablehlo.transpose %v2868, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2870 = stablehlo.reshape %v875 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2872 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2873 = stablehlo.reduce(%v2870 init: %v2871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2875 = stablehlo.divide %v2874, %v2872 : tensor<256x256x14x14xf32>
    %v2876 = stablehlo.subtract %v2870, %v2875 : tensor<256x256x14x14xf32>
    %v2877 = stablehlo.multiply %v2876, %v2876 : tensor<256x256x14x14xf32>
    %v2878 = stablehlo.reduce(%v2877 init: %v2871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2879 = stablehlo.broadcast_in_dim %v2878, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2880 = stablehlo.divide %v2879, %v2872 : tensor<256x256x14x14xf32>
    %v2881 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2882 = stablehlo.add %v2880, %v2881 : tensor<256x256x14x14xf32>
    %v2883 = stablehlo.rsqrt %v2882 : tensor<256x256x14x14xf32>
    %v2884 = stablehlo.multiply %v2876, %v2883 : tensor<256x256x14x14xf32>
    %v2885 = stablehlo.reshape %v2762 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2886 = stablehlo.multiply %v2885, %v2884 : tensor<256x256x14x14xf32>
    %v2887 = stablehlo.reduce(%v2886 init: %v2871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2888 = stablehlo.reshape %v2762 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2889 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2890 = stablehlo.reduce(%v2888 init: %v2889) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2891 = stablehlo.reshape %v897 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2892 = stablehlo.reshape %v2754 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2893 = stablehlo.transpose %v2891, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v2894 = stablehlo.transpose %v2892, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v2895 = stablehlo.convolution(%v2893, %v2894)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v2896 = stablehlo.transpose %v2895, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2897 = stablehlo.reshape %v902 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2899 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2900 = stablehlo.reduce(%v2897 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2901 = stablehlo.broadcast_in_dim %v2900, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2902 = stablehlo.divide %v2901, %v2899 : tensor<256x1024x14x14xf32>
    %v2903 = stablehlo.subtract %v2897, %v2902 : tensor<256x1024x14x14xf32>
    %v2904 = stablehlo.multiply %v2903, %v2903 : tensor<256x1024x14x14xf32>
    %v2905 = stablehlo.reduce(%v2904 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2905, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2907 = stablehlo.divide %v2906, %v2899 : tensor<256x1024x14x14xf32>
    %v2908 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2909 = stablehlo.add %v2907, %v2908 : tensor<256x1024x14x14xf32>
    %v2910 = stablehlo.rsqrt %v2909 : tensor<256x1024x14x14xf32>
    %v2911 = stablehlo.multiply %v2903, %v2910 : tensor<256x1024x14x14xf32>
    %v2912 = stablehlo.reshape %v2724 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2913 = stablehlo.multiply %v2912, %v2911 : tensor<256x1024x14x14xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2915 = stablehlo.reshape %v2724 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2917 = stablehlo.reduce(%v2915 init: %v2916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v2919 = stablehlo.compare GT, %v841, %v2918 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v2920 = stablehlo.select %v2919, %v2836, %v2918 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v2921 = stablehlo.reshape %v820 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2923 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v2924 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v2925 = stablehlo.reduce(%v2921 init: %v2922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2926 = stablehlo.broadcast_in_dim %v2925, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2927 = stablehlo.divide %v2926, %v2923 : tensor<256x1024x14x14xf32>
    %v2928 = stablehlo.subtract %v2921, %v2927 : tensor<256x1024x14x14xf32>
    %v2929 = stablehlo.multiply %v2928, %v2928 : tensor<256x1024x14x14xf32>
    %v2930 = stablehlo.reduce(%v2929 init: %v2922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2931 = stablehlo.broadcast_in_dim %v2930, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2932 = stablehlo.divide %v2931, %v2923 : tensor<256x1024x14x14xf32>
    %v2933 = stablehlo.add %v2932, %v2924 : tensor<256x1024x14x14xf32>
    %v2934 = stablehlo.rsqrt %v2933 : tensor<256x1024x14x14xf32>
    %v2935 = stablehlo.multiply %v2928, %v2934 : tensor<256x1024x14x14xf32>
    %v2936 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2937 = stablehlo.reshape %v2920 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2938 = stablehlo.multiply %v2936, %v2937 : tensor<256x1024x14x14xf32>
    %v2939 = stablehlo.reduce(%v2938 init: %v2922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2941 = stablehlo.multiply %v2935, %v2938 : tensor<256x1024x14x14xf32>
    %v2942 = stablehlo.reduce(%v2941 init: %v2922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2943 = stablehlo.broadcast_in_dim %v2942, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v2944 = stablehlo.multiply %v2938, %v2923 : tensor<256x1024x14x14xf32>
    %v2945 = stablehlo.subtract %v2944, %v2940 : tensor<256x1024x14x14xf32>
    %v2946 = stablehlo.multiply %v2935, %v2943 : tensor<256x1024x14x14xf32>
    %v2947 = stablehlo.subtract %v2945, %v2946 : tensor<256x1024x14x14xf32>
    %v2948 = stablehlo.divide %v2934, %v2923 : tensor<256x1024x14x14xf32>
    %v2949 = stablehlo.multiply %v2948, %v2947 : tensor<256x1024x14x14xf32>
    %v2950 = stablehlo.reshape %v2949 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v2951 = stablehlo.reshape %v2950 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v2952 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2953 = stablehlo.transpose %v2952, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2954 = stablehlo.convolution(%v2951, %v2953)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v2955 = stablehlo.reshape %v2954 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2956 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2957 = stablehlo.compare GT, %v813, %v2956 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2958 = stablehlo.select %v2957, %v2955, %v2956 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2959 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2960 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2961 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v2962 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v2963 = stablehlo.reduce(%v2959 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2964 = stablehlo.broadcast_in_dim %v2963, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2965 = stablehlo.divide %v2964, %v2961 : tensor<256x256x14x14xf32>
    %v2966 = stablehlo.subtract %v2959, %v2965 : tensor<256x256x14x14xf32>
    %v2967 = stablehlo.multiply %v2966, %v2966 : tensor<256x256x14x14xf32>
    %v2968 = stablehlo.reduce(%v2967 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2969 = stablehlo.broadcast_in_dim %v2968, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2970 = stablehlo.divide %v2969, %v2961 : tensor<256x256x14x14xf32>
    %v2971 = stablehlo.add %v2970, %v2962 : tensor<256x256x14x14xf32>
    %v2972 = stablehlo.rsqrt %v2971 : tensor<256x256x14x14xf32>
    %v2973 = stablehlo.multiply %v2966, %v2972 : tensor<256x256x14x14xf32>
    %v2974 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2975 = stablehlo.reshape %v2958 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2976 = stablehlo.multiply %v2974, %v2975 : tensor<256x256x14x14xf32>
    %v2977 = stablehlo.reduce(%v2976 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2979 = stablehlo.multiply %v2973, %v2976 : tensor<256x256x14x14xf32>
    %v2980 = stablehlo.reduce(%v2979 init: %v2960) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v2982 = stablehlo.multiply %v2976, %v2961 : tensor<256x256x14x14xf32>
    %v2983 = stablehlo.subtract %v2982, %v2978 : tensor<256x256x14x14xf32>
    %v2984 = stablehlo.multiply %v2973, %v2981 : tensor<256x256x14x14xf32>
    %v2985 = stablehlo.subtract %v2983, %v2984 : tensor<256x256x14x14xf32>
    %v2986 = stablehlo.divide %v2972, %v2961 : tensor<256x256x14x14xf32>
    %v2987 = stablehlo.multiply %v2986, %v2985 : tensor<256x256x14x14xf32>
    %v2988 = stablehlo.reshape %v2987 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2989 = stablehlo.reshape %v2988 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2990 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2991 = stablehlo.transpose %v2990, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2992 = stablehlo.convolution(%v2989, %v2991)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x14x14xf32>
    %v2993 = stablehlo.reshape %v2992 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v2994 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v2995 = stablehlo.compare GT, %v786, %v2994 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v2996 = stablehlo.select %v2995, %v2993, %v2994 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v2997 = stablehlo.reshape %v766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v2998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2999 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3000 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3001 = stablehlo.reduce(%v2997 init: %v2998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3002 = stablehlo.broadcast_in_dim %v3001, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3003 = stablehlo.divide %v3002, %v2999 : tensor<256x256x14x14xf32>
    %v3004 = stablehlo.subtract %v2997, %v3003 : tensor<256x256x14x14xf32>
    %v3005 = stablehlo.multiply %v3004, %v3004 : tensor<256x256x14x14xf32>
    %v3006 = stablehlo.reduce(%v3005 init: %v2998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3007 = stablehlo.broadcast_in_dim %v3006, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3008 = stablehlo.divide %v3007, %v2999 : tensor<256x256x14x14xf32>
    %v3009 = stablehlo.add %v3008, %v3000 : tensor<256x256x14x14xf32>
    %v3010 = stablehlo.rsqrt %v3009 : tensor<256x256x14x14xf32>
    %v3011 = stablehlo.multiply %v3004, %v3010 : tensor<256x256x14x14xf32>
    %v3012 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3013 = stablehlo.reshape %v2996 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3014 = stablehlo.multiply %v3012, %v3013 : tensor<256x256x14x14xf32>
    %v3015 = stablehlo.reduce(%v3014 init: %v2998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3016 = stablehlo.broadcast_in_dim %v3015, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3017 = stablehlo.multiply %v3011, %v3014 : tensor<256x256x14x14xf32>
    %v3018 = stablehlo.reduce(%v3017 init: %v2998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3020 = stablehlo.multiply %v3014, %v2999 : tensor<256x256x14x14xf32>
    %v3021 = stablehlo.subtract %v3020, %v3016 : tensor<256x256x14x14xf32>
    %v3022 = stablehlo.multiply %v3011, %v3019 : tensor<256x256x14x14xf32>
    %v3023 = stablehlo.subtract %v3021, %v3022 : tensor<256x256x14x14xf32>
    %v3024 = stablehlo.divide %v3010, %v2999 : tensor<256x256x14x14xf32>
    %v3025 = stablehlo.multiply %v3024, %v3023 : tensor<256x256x14x14xf32>
    %v3026 = stablehlo.reshape %v3025 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3028 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3029 = stablehlo.transpose %v3028, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3030 = stablehlo.convolution(%v3027, %v3029)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<256x1024x14x14xf32>
    %v3031 = stablehlo.reshape %v3030 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3032 = stablehlo.add %v3031, %v2920 : tensor<256x200704xf32>
    %v3033 = stablehlo.reshape %v761 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3034 = stablehlo.reshape %v3026 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3035 = stablehlo.transpose %v3033, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3036 = stablehlo.transpose %v3034, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3037 = stablehlo.convolution(%v3035, %v3036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3038 = stablehlo.transpose %v3037, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3039 = stablehlo.reshape %v766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3041 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3042 = stablehlo.reduce(%v3039 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3043 = stablehlo.broadcast_in_dim %v3042, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3044 = stablehlo.divide %v3043, %v3041 : tensor<256x256x14x14xf32>
    %v3045 = stablehlo.subtract %v3039, %v3044 : tensor<256x256x14x14xf32>
    %v3046 = stablehlo.multiply %v3045, %v3045 : tensor<256x256x14x14xf32>
    %v3047 = stablehlo.reduce(%v3046 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3048 = stablehlo.broadcast_in_dim %v3047, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3049 = stablehlo.divide %v3048, %v3041 : tensor<256x256x14x14xf32>
    %v3050 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3051 = stablehlo.add %v3049, %v3050 : tensor<256x256x14x14xf32>
    %v3052 = stablehlo.rsqrt %v3051 : tensor<256x256x14x14xf32>
    %v3053 = stablehlo.multiply %v3045, %v3052 : tensor<256x256x14x14xf32>
    %v3054 = stablehlo.reshape %v2996 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3055 = stablehlo.multiply %v3054, %v3053 : tensor<256x256x14x14xf32>
    %v3056 = stablehlo.reduce(%v3055 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3057 = stablehlo.reshape %v2996 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3059 = stablehlo.reduce(%v3057 init: %v3058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3060 = stablehlo.reshape %v788 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3061 = stablehlo.reshape %v2988 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3062 = stablehlo.transpose %v3060, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3063 = stablehlo.transpose %v3061, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3064 = stablehlo.convolution(%v3062, %v3063)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<256x256x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3065 = stablehlo.transpose %v3064, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3066 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3068 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3069 = stablehlo.reduce(%v3066 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3070 = stablehlo.broadcast_in_dim %v3069, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3071 = stablehlo.divide %v3070, %v3068 : tensor<256x256x14x14xf32>
    %v3072 = stablehlo.subtract %v3066, %v3071 : tensor<256x256x14x14xf32>
    %v3073 = stablehlo.multiply %v3072, %v3072 : tensor<256x256x14x14xf32>
    %v3074 = stablehlo.reduce(%v3073 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3075 = stablehlo.broadcast_in_dim %v3074, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3076 = stablehlo.divide %v3075, %v3068 : tensor<256x256x14x14xf32>
    %v3077 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3078 = stablehlo.add %v3076, %v3077 : tensor<256x256x14x14xf32>
    %v3079 = stablehlo.rsqrt %v3078 : tensor<256x256x14x14xf32>
    %v3080 = stablehlo.multiply %v3072, %v3079 : tensor<256x256x14x14xf32>
    %v3081 = stablehlo.reshape %v2958 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3082 = stablehlo.multiply %v3081, %v3080 : tensor<256x256x14x14xf32>
    %v3083 = stablehlo.reduce(%v3082 init: %v3067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3084 = stablehlo.reshape %v2958 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3086 = stablehlo.reduce(%v3084 init: %v3085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3087 = stablehlo.reshape %v815 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3088 = stablehlo.reshape %v2950 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3089 = stablehlo.transpose %v3087, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3090 = stablehlo.transpose %v3088, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3091 = stablehlo.convolution(%v3089, %v3090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3092 = stablehlo.transpose %v3091, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3093 = stablehlo.reshape %v820 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3094 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3095 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3096 = stablehlo.reduce(%v3093 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3097 = stablehlo.broadcast_in_dim %v3096, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3098 = stablehlo.divide %v3097, %v3095 : tensor<256x1024x14x14xf32>
    %v3099 = stablehlo.subtract %v3093, %v3098 : tensor<256x1024x14x14xf32>
    %v3100 = stablehlo.multiply %v3099, %v3099 : tensor<256x1024x14x14xf32>
    %v3101 = stablehlo.reduce(%v3100 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3101, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3103 = stablehlo.divide %v3102, %v3095 : tensor<256x1024x14x14xf32>
    %v3104 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3105 = stablehlo.add %v3103, %v3104 : tensor<256x1024x14x14xf32>
    %v3106 = stablehlo.rsqrt %v3105 : tensor<256x1024x14x14xf32>
    %v3107 = stablehlo.multiply %v3099, %v3106 : tensor<256x1024x14x14xf32>
    %v3108 = stablehlo.reshape %v2920 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3109 = stablehlo.multiply %v3108, %v3107 : tensor<256x1024x14x14xf32>
    %v3110 = stablehlo.reduce(%v3109 init: %v3094) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3111 = stablehlo.reshape %v2920 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3113 = stablehlo.reduce(%v3111 init: %v3112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3114 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3115 = stablehlo.compare GT, %v759, %v3114 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3116 = stablehlo.select %v3115, %v3032, %v3114 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3117 = stablehlo.reshape %v713 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3119 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3120 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3121 = stablehlo.reduce(%v3117 init: %v3118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3122 = stablehlo.broadcast_in_dim %v3121, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3123 = stablehlo.divide %v3122, %v3119 : tensor<256x1024x14x14xf32>
    %v3124 = stablehlo.subtract %v3117, %v3123 : tensor<256x1024x14x14xf32>
    %v3125 = stablehlo.multiply %v3124, %v3124 : tensor<256x1024x14x14xf32>
    %v3126 = stablehlo.reduce(%v3125 init: %v3118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3127 = stablehlo.broadcast_in_dim %v3126, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3128 = stablehlo.divide %v3127, %v3119 : tensor<256x1024x14x14xf32>
    %v3129 = stablehlo.add %v3128, %v3120 : tensor<256x1024x14x14xf32>
    %v3130 = stablehlo.rsqrt %v3129 : tensor<256x1024x14x14xf32>
    %v3131 = stablehlo.multiply %v3124, %v3130 : tensor<256x1024x14x14xf32>
    %v3132 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3133 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3134 = stablehlo.multiply %v3132, %v3133 : tensor<256x1024x14x14xf32>
    %v3135 = stablehlo.reduce(%v3134 init: %v3118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3136 = stablehlo.broadcast_in_dim %v3135, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3137 = stablehlo.multiply %v3131, %v3134 : tensor<256x1024x14x14xf32>
    %v3138 = stablehlo.reduce(%v3137 init: %v3118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3139 = stablehlo.broadcast_in_dim %v3138, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3140 = stablehlo.multiply %v3134, %v3119 : tensor<256x1024x14x14xf32>
    %v3141 = stablehlo.subtract %v3140, %v3136 : tensor<256x1024x14x14xf32>
    %v3142 = stablehlo.multiply %v3131, %v3139 : tensor<256x1024x14x14xf32>
    %v3143 = stablehlo.subtract %v3141, %v3142 : tensor<256x1024x14x14xf32>
    %v3144 = stablehlo.divide %v3130, %v3119 : tensor<256x1024x14x14xf32>
    %v3145 = stablehlo.multiply %v3144, %v3143 : tensor<256x1024x14x14xf32>
    %v3146 = stablehlo.reshape %v3145 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3147 = stablehlo.reshape %v3146 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3148 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3149 = stablehlo.transpose %v3148, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3150 = stablehlo.convolution(%v3147, %v3149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<256x256x14x14xf32>
    %v3151 = stablehlo.reshape %v3150 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3152 = stablehlo.constant dense<0.0> : tensor<256x50176xf32>
    %v3153 = stablehlo.compare GT, %v706, %v3152 : (tensor<256x50176xf32>, tensor<256x50176xf32>) -> tensor<256x50176xi1>
    %v3154 = stablehlo.select %v3153, %v3151, %v3152 : tensor<256x50176xi1>, tensor<256x50176xf32>
    %v3155 = stablehlo.reshape %v686 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3157 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3158 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3159 = stablehlo.reduce(%v3155 init: %v3156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3160 = stablehlo.broadcast_in_dim %v3159, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3161 = stablehlo.divide %v3160, %v3157 : tensor<256x256x14x14xf32>
    %v3162 = stablehlo.subtract %v3155, %v3161 : tensor<256x256x14x14xf32>
    %v3163 = stablehlo.multiply %v3162, %v3162 : tensor<256x256x14x14xf32>
    %v3164 = stablehlo.reduce(%v3163 init: %v3156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3165 = stablehlo.broadcast_in_dim %v3164, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3166 = stablehlo.divide %v3165, %v3157 : tensor<256x256x14x14xf32>
    %v3167 = stablehlo.add %v3166, %v3158 : tensor<256x256x14x14xf32>
    %v3168 = stablehlo.rsqrt %v3167 : tensor<256x256x14x14xf32>
    %v3169 = stablehlo.multiply %v3162, %v3168 : tensor<256x256x14x14xf32>
    %v3170 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3171 = stablehlo.reshape %v3154 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3172 = stablehlo.multiply %v3170, %v3171 : tensor<256x256x14x14xf32>
    %v3173 = stablehlo.reduce(%v3172 init: %v3156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3174 = stablehlo.broadcast_in_dim %v3173, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3175 = stablehlo.multiply %v3169, %v3172 : tensor<256x256x14x14xf32>
    %v3176 = stablehlo.reduce(%v3175 init: %v3156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3177 = stablehlo.broadcast_in_dim %v3176, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3178 = stablehlo.multiply %v3172, %v3157 : tensor<256x256x14x14xf32>
    %v3179 = stablehlo.subtract %v3178, %v3174 : tensor<256x256x14x14xf32>
    %v3180 = stablehlo.multiply %v3169, %v3177 : tensor<256x256x14x14xf32>
    %v3181 = stablehlo.subtract %v3179, %v3180 : tensor<256x256x14x14xf32>
    %v3182 = stablehlo.divide %v3168, %v3157 : tensor<256x256x14x14xf32>
    %v3183 = stablehlo.multiply %v3182, %v3181 : tensor<256x256x14x14xf32>
    %v3184 = stablehlo.reshape %v3183 : (tensor<256x256x14x14xf32>) -> tensor<256x50176xf32>
    %v3185 = stablehlo.reshape %v3184 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3187 = stablehlo.pad %v3185, %v3186, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v3188 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3189 = stablehlo.transpose %v3188, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3190 = stablehlo.convolution(%v3187, %v3189)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<256x256x28x28xf32>
    %v3191 = stablehlo.reshape %v3190 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3192 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v3193 = stablehlo.compare GT, %v679, %v3192 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v3194 = stablehlo.select %v3193, %v3191, %v3192 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v3195 = stablehlo.reshape %v659 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3196 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3197 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3198 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3199 = stablehlo.reduce(%v3195 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3200 = stablehlo.broadcast_in_dim %v3199, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3201 = stablehlo.divide %v3200, %v3197 : tensor<256x256x28x28xf32>
    %v3202 = stablehlo.subtract %v3195, %v3201 : tensor<256x256x28x28xf32>
    %v3203 = stablehlo.multiply %v3202, %v3202 : tensor<256x256x28x28xf32>
    %v3204 = stablehlo.reduce(%v3203 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3205 = stablehlo.broadcast_in_dim %v3204, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3206 = stablehlo.divide %v3205, %v3197 : tensor<256x256x28x28xf32>
    %v3207 = stablehlo.add %v3206, %v3198 : tensor<256x256x28x28xf32>
    %v3208 = stablehlo.rsqrt %v3207 : tensor<256x256x28x28xf32>
    %v3209 = stablehlo.multiply %v3202, %v3208 : tensor<256x256x28x28xf32>
    %v3210 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3211 = stablehlo.reshape %v3194 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3212 = stablehlo.multiply %v3210, %v3211 : tensor<256x256x28x28xf32>
    %v3213 = stablehlo.reduce(%v3212 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3214 = stablehlo.broadcast_in_dim %v3213, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3215 = stablehlo.multiply %v3209, %v3212 : tensor<256x256x28x28xf32>
    %v3216 = stablehlo.reduce(%v3215 init: %v3196) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3217 = stablehlo.broadcast_in_dim %v3216, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3218 = stablehlo.multiply %v3212, %v3197 : tensor<256x256x28x28xf32>
    %v3219 = stablehlo.subtract %v3218, %v3214 : tensor<256x256x28x28xf32>
    %v3220 = stablehlo.multiply %v3209, %v3217 : tensor<256x256x28x28xf32>
    %v3221 = stablehlo.subtract %v3219, %v3220 : tensor<256x256x28x28xf32>
    %v3222 = stablehlo.divide %v3208, %v3197 : tensor<256x256x28x28xf32>
    %v3223 = stablehlo.multiply %v3222, %v3221 : tensor<256x256x28x28xf32>
    %v3224 = stablehlo.reshape %v3223 : (tensor<256x256x28x28xf32>) -> tensor<256x200704xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3226 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v3227 = stablehlo.transpose %v3226, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v3228 = stablehlo.convolution(%v3225, %v3227)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<512x256x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3229 = stablehlo.reshape %v3228 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3230 = stablehlo.reshape %v738 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3232 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3233 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3234 = stablehlo.reduce(%v3230 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3235 = stablehlo.broadcast_in_dim %v3234, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3236 = stablehlo.divide %v3235, %v3232 : tensor<256x1024x14x14xf32>
    %v3237 = stablehlo.subtract %v3230, %v3236 : tensor<256x1024x14x14xf32>
    %v3238 = stablehlo.multiply %v3237, %v3237 : tensor<256x1024x14x14xf32>
    %v3239 = stablehlo.reduce(%v3238 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3240 = stablehlo.broadcast_in_dim %v3239, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3241 = stablehlo.divide %v3240, %v3232 : tensor<256x1024x14x14xf32>
    %v3242 = stablehlo.add %v3241, %v3233 : tensor<256x1024x14x14xf32>
    %v3243 = stablehlo.rsqrt %v3242 : tensor<256x1024x14x14xf32>
    %v3244 = stablehlo.multiply %v3237, %v3243 : tensor<256x1024x14x14xf32>
    %v3245 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3246 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3247 = stablehlo.multiply %v3245, %v3246 : tensor<256x1024x14x14xf32>
    %v3248 = stablehlo.reduce(%v3247 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3248, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3250 = stablehlo.multiply %v3244, %v3247 : tensor<256x1024x14x14xf32>
    %v3251 = stablehlo.reduce(%v3250 init: %v3231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3252 = stablehlo.broadcast_in_dim %v3251, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3253 = stablehlo.multiply %v3247, %v3232 : tensor<256x1024x14x14xf32>
    %v3254 = stablehlo.subtract %v3253, %v3249 : tensor<256x1024x14x14xf32>
    %v3255 = stablehlo.multiply %v3244, %v3252 : tensor<256x1024x14x14xf32>
    %v3256 = stablehlo.subtract %v3254, %v3255 : tensor<256x1024x14x14xf32>
    %v3257 = stablehlo.divide %v3243, %v3232 : tensor<256x1024x14x14xf32>
    %v3258 = stablehlo.multiply %v3257, %v3256 : tensor<256x1024x14x14xf32>
    %v3259 = stablehlo.reshape %v3258 : (tensor<256x1024x14x14xf32>) -> tensor<256x200704xf32>
    %v3260 = stablehlo.reshape %v3259 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.pad %v3260, %v3261, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<256x1024x28x28xf32>
    %v3263 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v3264 = stablehlo.transpose %v3263, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3265 = stablehlo.convolution(%v3262, %v3264)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x1024x28x28xf32>, tensor<512x1024x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3267 = stablehlo.add %v3229, %v3266 : tensor<256x401408xf32>
    %v3268 = stablehlo.reshape %v654 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3269 = stablehlo.reshape %v3224 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3270 = stablehlo.transpose %v3268, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3271 = stablehlo.transpose %v3269, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3272 = stablehlo.convolution(%v3270, %v3271)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<512x256x1x1xf32>
    %v3273 = stablehlo.transpose %v3272, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v3274 = stablehlo.reshape %v659 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3276 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v3277 = stablehlo.reduce(%v3274 init: %v3275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3278 = stablehlo.broadcast_in_dim %v3277, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3279 = stablehlo.divide %v3278, %v3276 : tensor<256x256x28x28xf32>
    %v3280 = stablehlo.subtract %v3274, %v3279 : tensor<256x256x28x28xf32>
    %v3281 = stablehlo.multiply %v3280, %v3280 : tensor<256x256x28x28xf32>
    %v3282 = stablehlo.reduce(%v3281 init: %v3275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3283 = stablehlo.broadcast_in_dim %v3282, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v3284 = stablehlo.divide %v3283, %v3276 : tensor<256x256x28x28xf32>
    %v3285 = stablehlo.constant dense<1.0e-05> : tensor<256x256x28x28xf32>
    %v3286 = stablehlo.add %v3284, %v3285 : tensor<256x256x28x28xf32>
    %v3287 = stablehlo.rsqrt %v3286 : tensor<256x256x28x28xf32>
    %v3288 = stablehlo.multiply %v3280, %v3287 : tensor<256x256x28x28xf32>
    %v3289 = stablehlo.reshape %v3194 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3290 = stablehlo.multiply %v3289, %v3288 : tensor<256x256x28x28xf32>
    %v3291 = stablehlo.reduce(%v3290 init: %v3275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3292 = stablehlo.reshape %v3194 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3294 = stablehlo.reduce(%v3292 init: %v3293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3295 = stablehlo.reshape %v681 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v3296 = stablehlo.reshape %v3184 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3298 = stablehlo.pad %v3296, %v3297, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256x256x28x28xf32>
    %v3299 = stablehlo.transpose %v3295, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3300 = stablehlo.transpose %v3298, dims = [1, 0, 2, 3] : (tensor<256x256x28x28xf32>) -> tensor<256x256x28x28xf32>
    %v3301 = stablehlo.convolution(%v3299, %v3300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x28x28xf32>, tensor<256x256x28x28xf32>) -> tensor<256x256x3x3xf32>
    %v3302 = stablehlo.transpose %v3301, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3303 = stablehlo.reshape %v686 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3305 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v3306 = stablehlo.reduce(%v3303 init: %v3304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3307 = stablehlo.broadcast_in_dim %v3306, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3308 = stablehlo.divide %v3307, %v3305 : tensor<256x256x14x14xf32>
    %v3309 = stablehlo.subtract %v3303, %v3308 : tensor<256x256x14x14xf32>
    %v3310 = stablehlo.multiply %v3309, %v3309 : tensor<256x256x14x14xf32>
    %v3311 = stablehlo.reduce(%v3310 init: %v3304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v3313 = stablehlo.divide %v3312, %v3305 : tensor<256x256x14x14xf32>
    %v3314 = stablehlo.constant dense<1.0e-05> : tensor<256x256x14x14xf32>
    %v3315 = stablehlo.add %v3313, %v3314 : tensor<256x256x14x14xf32>
    %v3316 = stablehlo.rsqrt %v3315 : tensor<256x256x14x14xf32>
    %v3317 = stablehlo.multiply %v3309, %v3316 : tensor<256x256x14x14xf32>
    %v3318 = stablehlo.reshape %v3154 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3319 = stablehlo.multiply %v3318, %v3317 : tensor<256x256x14x14xf32>
    %v3320 = stablehlo.reduce(%v3319 init: %v3304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3321 = stablehlo.reshape %v3154 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3322 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3323 = stablehlo.reduce(%v3321 init: %v3322) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3324 = stablehlo.reshape %v708 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v3325 = stablehlo.reshape %v3146 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3326 = stablehlo.transpose %v3324, dims = [1, 0, 2, 3] : (tensor<256x256x14x14xf32>) -> tensor<256x256x14x14xf32>
    %v3327 = stablehlo.transpose %v3325, dims = [1, 0, 2, 3] : (tensor<256x1024x14x14xf32>) -> tensor<1024x256x14x14xf32>
    %v3328 = stablehlo.convolution(%v3326, %v3327)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x14x14xf32>, tensor<1024x256x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3329 = stablehlo.transpose %v3328, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3330 = stablehlo.reshape %v713 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3332 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3333 = stablehlo.reduce(%v3330 init: %v3331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3334 = stablehlo.broadcast_in_dim %v3333, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3335 = stablehlo.divide %v3334, %v3332 : tensor<256x1024x14x14xf32>
    %v3336 = stablehlo.subtract %v3330, %v3335 : tensor<256x1024x14x14xf32>
    %v3337 = stablehlo.multiply %v3336, %v3336 : tensor<256x1024x14x14xf32>
    %v3338 = stablehlo.reduce(%v3337 init: %v3331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3338, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3340 = stablehlo.divide %v3339, %v3332 : tensor<256x1024x14x14xf32>
    %v3341 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3342 = stablehlo.add %v3340, %v3341 : tensor<256x1024x14x14xf32>
    %v3343 = stablehlo.rsqrt %v3342 : tensor<256x1024x14x14xf32>
    %v3344 = stablehlo.multiply %v3336, %v3343 : tensor<256x1024x14x14xf32>
    %v3345 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3346 = stablehlo.multiply %v3345, %v3344 : tensor<256x1024x14x14xf32>
    %v3347 = stablehlo.reduce(%v3346 init: %v3331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3348 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3350 = stablehlo.reduce(%v3348 init: %v3349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3351 = stablehlo.reshape %v654 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3352 = stablehlo.reshape %v3259 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3354 = stablehlo.pad %v3352, %v3353, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<256x1024x28x28xf32>
    %v3355 = stablehlo.transpose %v3351, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3356 = stablehlo.transpose %v3354, dims = [1, 0, 2, 3] : (tensor<256x1024x28x28xf32>) -> tensor<1024x256x28x28xf32>
    %v3357 = stablehlo.convolution(%v3355, %v3356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<1024x256x28x28xf32>) -> tensor<512x1024x1x1xf32>
    %v3358 = stablehlo.transpose %v3357, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3359 = stablehlo.reshape %v738 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3361 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v3362 = stablehlo.reduce(%v3359 init: %v3360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3363 = stablehlo.broadcast_in_dim %v3362, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3364 = stablehlo.divide %v3363, %v3361 : tensor<256x1024x14x14xf32>
    %v3365 = stablehlo.subtract %v3359, %v3364 : tensor<256x1024x14x14xf32>
    %v3366 = stablehlo.multiply %v3365, %v3365 : tensor<256x1024x14x14xf32>
    %v3367 = stablehlo.reduce(%v3366 init: %v3360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3367, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v3369 = stablehlo.divide %v3368, %v3361 : tensor<256x1024x14x14xf32>
    %v3370 = stablehlo.constant dense<1.0e-05> : tensor<256x1024x14x14xf32>
    %v3371 = stablehlo.add %v3369, %v3370 : tensor<256x1024x14x14xf32>
    %v3372 = stablehlo.rsqrt %v3371 : tensor<256x1024x14x14xf32>
    %v3373 = stablehlo.multiply %v3365, %v3372 : tensor<256x1024x14x14xf32>
    %v3374 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3375 = stablehlo.multiply %v3374, %v3373 : tensor<256x1024x14x14xf32>
    %v3376 = stablehlo.reduce(%v3375 init: %v3360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3377 = stablehlo.reshape %v3116 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v3378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3379 = stablehlo.reduce(%v3377 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3380 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v3381 = stablehlo.compare GT, %v652, %v3380 : (tensor<256x401408xf32>, tensor<256x401408xf32>) -> tensor<256x401408xi1>
    %v3382 = stablehlo.select %v3381, %v3267, %v3380 : tensor<256x401408xi1>, tensor<256x401408xf32>
    %v3383 = stablehlo.reshape %v631 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3385 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3386 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3387 = stablehlo.reduce(%v3383 init: %v3384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3388 = stablehlo.broadcast_in_dim %v3387, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3389 = stablehlo.divide %v3388, %v3385 : tensor<256x512x28x28xf32>
    %v3390 = stablehlo.subtract %v3383, %v3389 : tensor<256x512x28x28xf32>
    %v3391 = stablehlo.multiply %v3390, %v3390 : tensor<256x512x28x28xf32>
    %v3392 = stablehlo.reduce(%v3391 init: %v3384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3393 = stablehlo.broadcast_in_dim %v3392, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3394 = stablehlo.divide %v3393, %v3385 : tensor<256x512x28x28xf32>
    %v3395 = stablehlo.add %v3394, %v3386 : tensor<256x512x28x28xf32>
    %v3396 = stablehlo.rsqrt %v3395 : tensor<256x512x28x28xf32>
    %v3397 = stablehlo.multiply %v3390, %v3396 : tensor<256x512x28x28xf32>
    %v3398 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3399 = stablehlo.reshape %v3382 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3400 = stablehlo.multiply %v3398, %v3399 : tensor<256x512x28x28xf32>
    %v3401 = stablehlo.reduce(%v3400 init: %v3384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3402 = stablehlo.broadcast_in_dim %v3401, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3403 = stablehlo.multiply %v3397, %v3400 : tensor<256x512x28x28xf32>
    %v3404 = stablehlo.reduce(%v3403 init: %v3384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3405 = stablehlo.broadcast_in_dim %v3404, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3406 = stablehlo.multiply %v3400, %v3385 : tensor<256x512x28x28xf32>
    %v3407 = stablehlo.subtract %v3406, %v3402 : tensor<256x512x28x28xf32>
    %v3408 = stablehlo.multiply %v3397, %v3405 : tensor<256x512x28x28xf32>
    %v3409 = stablehlo.subtract %v3407, %v3408 : tensor<256x512x28x28xf32>
    %v3410 = stablehlo.divide %v3396, %v3385 : tensor<256x512x28x28xf32>
    %v3411 = stablehlo.multiply %v3410, %v3409 : tensor<256x512x28x28xf32>
    %v3412 = stablehlo.reshape %v3411 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3414 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3415 = stablehlo.transpose %v3414, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3416 = stablehlo.convolution(%v3413, %v3415)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3417 = stablehlo.reshape %v3416 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3418 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3419 = stablehlo.compare GT, %v624, %v3418 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3420 = stablehlo.select %v3419, %v3417, %v3418 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3421 = stablehlo.reshape %v604 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3423 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3424 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3425 = stablehlo.reduce(%v3421 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3425, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3427 = stablehlo.divide %v3426, %v3423 : tensor<256x128x28x28xf32>
    %v3428 = stablehlo.subtract %v3421, %v3427 : tensor<256x128x28x28xf32>
    %v3429 = stablehlo.multiply %v3428, %v3428 : tensor<256x128x28x28xf32>
    %v3430 = stablehlo.reduce(%v3429 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3432 = stablehlo.divide %v3431, %v3423 : tensor<256x128x28x28xf32>
    %v3433 = stablehlo.add %v3432, %v3424 : tensor<256x128x28x28xf32>
    %v3434 = stablehlo.rsqrt %v3433 : tensor<256x128x28x28xf32>
    %v3435 = stablehlo.multiply %v3428, %v3434 : tensor<256x128x28x28xf32>
    %v3436 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3437 = stablehlo.reshape %v3420 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3438 = stablehlo.multiply %v3436, %v3437 : tensor<256x128x28x28xf32>
    %v3439 = stablehlo.reduce(%v3438 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3440 = stablehlo.broadcast_in_dim %v3439, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3441 = stablehlo.multiply %v3435, %v3438 : tensor<256x128x28x28xf32>
    %v3442 = stablehlo.reduce(%v3441 init: %v3422) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3442, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3444 = stablehlo.multiply %v3438, %v3423 : tensor<256x128x28x28xf32>
    %v3445 = stablehlo.subtract %v3444, %v3440 : tensor<256x128x28x28xf32>
    %v3446 = stablehlo.multiply %v3435, %v3443 : tensor<256x128x28x28xf32>
    %v3447 = stablehlo.subtract %v3445, %v3446 : tensor<256x128x28x28xf32>
    %v3448 = stablehlo.divide %v3434, %v3423 : tensor<256x128x28x28xf32>
    %v3449 = stablehlo.multiply %v3448, %v3447 : tensor<256x128x28x28xf32>
    %v3450 = stablehlo.reshape %v3449 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3451 = stablehlo.reshape %v3450 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3452 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3453 = stablehlo.transpose %v3452, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3454 = stablehlo.convolution(%v3451, %v3453)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3456 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3457 = stablehlo.compare GT, %v597, %v3456 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3458 = stablehlo.select %v3457, %v3455, %v3456 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3459 = stablehlo.reshape %v577 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3461 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3462 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3463 = stablehlo.reduce(%v3459 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3465 = stablehlo.divide %v3464, %v3461 : tensor<256x128x28x28xf32>
    %v3466 = stablehlo.subtract %v3459, %v3465 : tensor<256x128x28x28xf32>
    %v3467 = stablehlo.multiply %v3466, %v3466 : tensor<256x128x28x28xf32>
    %v3468 = stablehlo.reduce(%v3467 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3469 = stablehlo.broadcast_in_dim %v3468, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3470 = stablehlo.divide %v3469, %v3461 : tensor<256x128x28x28xf32>
    %v3471 = stablehlo.add %v3470, %v3462 : tensor<256x128x28x28xf32>
    %v3472 = stablehlo.rsqrt %v3471 : tensor<256x128x28x28xf32>
    %v3473 = stablehlo.multiply %v3466, %v3472 : tensor<256x128x28x28xf32>
    %v3474 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3475 = stablehlo.reshape %v3458 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3476 = stablehlo.multiply %v3474, %v3475 : tensor<256x128x28x28xf32>
    %v3477 = stablehlo.reduce(%v3476 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3478 = stablehlo.broadcast_in_dim %v3477, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3479 = stablehlo.multiply %v3473, %v3476 : tensor<256x128x28x28xf32>
    %v3480 = stablehlo.reduce(%v3479 init: %v3460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3481 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3482 = stablehlo.multiply %v3476, %v3461 : tensor<256x128x28x28xf32>
    %v3483 = stablehlo.subtract %v3482, %v3478 : tensor<256x128x28x28xf32>
    %v3484 = stablehlo.multiply %v3473, %v3481 : tensor<256x128x28x28xf32>
    %v3485 = stablehlo.subtract %v3483, %v3484 : tensor<256x128x28x28xf32>
    %v3486 = stablehlo.divide %v3472, %v3461 : tensor<256x128x28x28xf32>
    %v3487 = stablehlo.multiply %v3486, %v3485 : tensor<256x128x28x28xf32>
    %v3488 = stablehlo.reshape %v3487 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3489 = stablehlo.reshape %v3488 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3490 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3491 = stablehlo.transpose %v3490, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3492 = stablehlo.convolution(%v3489, %v3491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3493 = stablehlo.reshape %v3492 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3494 = stablehlo.add %v3493, %v3382 : tensor<256x401408xf32>
    %v3495 = stablehlo.reshape %v572 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3496 = stablehlo.reshape %v3488 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3497 = stablehlo.transpose %v3495, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3498 = stablehlo.transpose %v3496, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3499 = stablehlo.convolution(%v3497, %v3498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3500 = stablehlo.transpose %v3499, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3501 = stablehlo.reshape %v577 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3503 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3504 = stablehlo.reduce(%v3501 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3506 = stablehlo.divide %v3505, %v3503 : tensor<256x128x28x28xf32>
    %v3507 = stablehlo.subtract %v3501, %v3506 : tensor<256x128x28x28xf32>
    %v3508 = stablehlo.multiply %v3507, %v3507 : tensor<256x128x28x28xf32>
    %v3509 = stablehlo.reduce(%v3508 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3510 = stablehlo.broadcast_in_dim %v3509, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3511 = stablehlo.divide %v3510, %v3503 : tensor<256x128x28x28xf32>
    %v3512 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3513 = stablehlo.add %v3511, %v3512 : tensor<256x128x28x28xf32>
    %v3514 = stablehlo.rsqrt %v3513 : tensor<256x128x28x28xf32>
    %v3515 = stablehlo.multiply %v3507, %v3514 : tensor<256x128x28x28xf32>
    %v3516 = stablehlo.reshape %v3458 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3517 = stablehlo.multiply %v3516, %v3515 : tensor<256x128x28x28xf32>
    %v3518 = stablehlo.reduce(%v3517 init: %v3502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3519 = stablehlo.reshape %v3458 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3521 = stablehlo.reduce(%v3519 init: %v3520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3522 = stablehlo.reshape %v599 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3523 = stablehlo.reshape %v3450 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3524 = stablehlo.transpose %v3522, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3525 = stablehlo.transpose %v3523, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3526 = stablehlo.convolution(%v3524, %v3525)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3527 = stablehlo.transpose %v3526, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3528 = stablehlo.reshape %v604 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3530 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3531 = stablehlo.reduce(%v3528 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3531, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3533 = stablehlo.divide %v3532, %v3530 : tensor<256x128x28x28xf32>
    %v3534 = stablehlo.subtract %v3528, %v3533 : tensor<256x128x28x28xf32>
    %v3535 = stablehlo.multiply %v3534, %v3534 : tensor<256x128x28x28xf32>
    %v3536 = stablehlo.reduce(%v3535 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3537 = stablehlo.broadcast_in_dim %v3536, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3538 = stablehlo.divide %v3537, %v3530 : tensor<256x128x28x28xf32>
    %v3539 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3540 = stablehlo.add %v3538, %v3539 : tensor<256x128x28x28xf32>
    %v3541 = stablehlo.rsqrt %v3540 : tensor<256x128x28x28xf32>
    %v3542 = stablehlo.multiply %v3534, %v3541 : tensor<256x128x28x28xf32>
    %v3543 = stablehlo.reshape %v3420 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3544 = stablehlo.multiply %v3543, %v3542 : tensor<256x128x28x28xf32>
    %v3545 = stablehlo.reduce(%v3544 init: %v3529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3546 = stablehlo.reshape %v3420 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3548 = stablehlo.reduce(%v3546 init: %v3547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3549 = stablehlo.reshape %v626 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3550 = stablehlo.reshape %v3412 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3551 = stablehlo.transpose %v3549, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3552 = stablehlo.transpose %v3550, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3553 = stablehlo.convolution(%v3551, %v3552)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v3554 = stablehlo.transpose %v3553, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3555 = stablehlo.reshape %v631 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3557 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3558 = stablehlo.reduce(%v3555 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3559 = stablehlo.broadcast_in_dim %v3558, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3560 = stablehlo.divide %v3559, %v3557 : tensor<256x512x28x28xf32>
    %v3561 = stablehlo.subtract %v3555, %v3560 : tensor<256x512x28x28xf32>
    %v3562 = stablehlo.multiply %v3561, %v3561 : tensor<256x512x28x28xf32>
    %v3563 = stablehlo.reduce(%v3562 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3564 = stablehlo.broadcast_in_dim %v3563, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3565 = stablehlo.divide %v3564, %v3557 : tensor<256x512x28x28xf32>
    %v3566 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3567 = stablehlo.add %v3565, %v3566 : tensor<256x512x28x28xf32>
    %v3568 = stablehlo.rsqrt %v3567 : tensor<256x512x28x28xf32>
    %v3569 = stablehlo.multiply %v3561, %v3568 : tensor<256x512x28x28xf32>
    %v3570 = stablehlo.reshape %v3382 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3571 = stablehlo.multiply %v3570, %v3569 : tensor<256x512x28x28xf32>
    %v3572 = stablehlo.reduce(%v3571 init: %v3556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3573 = stablehlo.reshape %v3382 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3574 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3575 = stablehlo.reduce(%v3573 init: %v3574) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3576 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v3577 = stablehlo.compare GT, %v570, %v3576 : (tensor<256x401408xf32>, tensor<256x401408xf32>) -> tensor<256x401408xi1>
    %v3578 = stablehlo.select %v3577, %v3494, %v3576 : tensor<256x401408xi1>, tensor<256x401408xf32>
    %v3579 = stablehlo.reshape %v549 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3581 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3582 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3583 = stablehlo.reduce(%v3579 init: %v3580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3584 = stablehlo.broadcast_in_dim %v3583, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3585 = stablehlo.divide %v3584, %v3581 : tensor<256x512x28x28xf32>
    %v3586 = stablehlo.subtract %v3579, %v3585 : tensor<256x512x28x28xf32>
    %v3587 = stablehlo.multiply %v3586, %v3586 : tensor<256x512x28x28xf32>
    %v3588 = stablehlo.reduce(%v3587 init: %v3580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3589 = stablehlo.broadcast_in_dim %v3588, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3590 = stablehlo.divide %v3589, %v3581 : tensor<256x512x28x28xf32>
    %v3591 = stablehlo.add %v3590, %v3582 : tensor<256x512x28x28xf32>
    %v3592 = stablehlo.rsqrt %v3591 : tensor<256x512x28x28xf32>
    %v3593 = stablehlo.multiply %v3586, %v3592 : tensor<256x512x28x28xf32>
    %v3594 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3595 = stablehlo.reshape %v3578 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3596 = stablehlo.multiply %v3594, %v3595 : tensor<256x512x28x28xf32>
    %v3597 = stablehlo.reduce(%v3596 init: %v3580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3597, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3599 = stablehlo.multiply %v3593, %v3596 : tensor<256x512x28x28xf32>
    %v3600 = stablehlo.reduce(%v3599 init: %v3580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3601 = stablehlo.broadcast_in_dim %v3600, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3602 = stablehlo.multiply %v3596, %v3581 : tensor<256x512x28x28xf32>
    %v3603 = stablehlo.subtract %v3602, %v3598 : tensor<256x512x28x28xf32>
    %v3604 = stablehlo.multiply %v3593, %v3601 : tensor<256x512x28x28xf32>
    %v3605 = stablehlo.subtract %v3603, %v3604 : tensor<256x512x28x28xf32>
    %v3606 = stablehlo.divide %v3592, %v3581 : tensor<256x512x28x28xf32>
    %v3607 = stablehlo.multiply %v3606, %v3605 : tensor<256x512x28x28xf32>
    %v3608 = stablehlo.reshape %v3607 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3609 = stablehlo.reshape %v3608 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3610 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3611 = stablehlo.transpose %v3610, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3612 = stablehlo.convolution(%v3609, %v3611)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3613 = stablehlo.reshape %v3612 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3614 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3615 = stablehlo.compare GT, %v542, %v3614 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3616 = stablehlo.select %v3615, %v3613, %v3614 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3617 = stablehlo.reshape %v522 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3619 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3620 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3621 = stablehlo.reduce(%v3617 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3622 = stablehlo.broadcast_in_dim %v3621, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3623 = stablehlo.divide %v3622, %v3619 : tensor<256x128x28x28xf32>
    %v3624 = stablehlo.subtract %v3617, %v3623 : tensor<256x128x28x28xf32>
    %v3625 = stablehlo.multiply %v3624, %v3624 : tensor<256x128x28x28xf32>
    %v3626 = stablehlo.reduce(%v3625 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3627 = stablehlo.broadcast_in_dim %v3626, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3628 = stablehlo.divide %v3627, %v3619 : tensor<256x128x28x28xf32>
    %v3629 = stablehlo.add %v3628, %v3620 : tensor<256x128x28x28xf32>
    %v3630 = stablehlo.rsqrt %v3629 : tensor<256x128x28x28xf32>
    %v3631 = stablehlo.multiply %v3624, %v3630 : tensor<256x128x28x28xf32>
    %v3632 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3633 = stablehlo.reshape %v3616 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3634 = stablehlo.multiply %v3632, %v3633 : tensor<256x128x28x28xf32>
    %v3635 = stablehlo.reduce(%v3634 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3636 = stablehlo.broadcast_in_dim %v3635, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3637 = stablehlo.multiply %v3631, %v3634 : tensor<256x128x28x28xf32>
    %v3638 = stablehlo.reduce(%v3637 init: %v3618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3639 = stablehlo.broadcast_in_dim %v3638, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3640 = stablehlo.multiply %v3634, %v3619 : tensor<256x128x28x28xf32>
    %v3641 = stablehlo.subtract %v3640, %v3636 : tensor<256x128x28x28xf32>
    %v3642 = stablehlo.multiply %v3631, %v3639 : tensor<256x128x28x28xf32>
    %v3643 = stablehlo.subtract %v3641, %v3642 : tensor<256x128x28x28xf32>
    %v3644 = stablehlo.divide %v3630, %v3619 : tensor<256x128x28x28xf32>
    %v3645 = stablehlo.multiply %v3644, %v3643 : tensor<256x128x28x28xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3647 = stablehlo.reshape %v3646 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3648 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3649 = stablehlo.transpose %v3648, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3650 = stablehlo.convolution(%v3647, %v3649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3652 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3653 = stablehlo.compare GT, %v515, %v3652 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3654 = stablehlo.select %v3653, %v3651, %v3652 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3655 = stablehlo.reshape %v495 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3657 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3658 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3659 = stablehlo.reduce(%v3655 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3660 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3661 = stablehlo.divide %v3660, %v3657 : tensor<256x128x28x28xf32>
    %v3662 = stablehlo.subtract %v3655, %v3661 : tensor<256x128x28x28xf32>
    %v3663 = stablehlo.multiply %v3662, %v3662 : tensor<256x128x28x28xf32>
    %v3664 = stablehlo.reduce(%v3663 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3665 = stablehlo.broadcast_in_dim %v3664, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3666 = stablehlo.divide %v3665, %v3657 : tensor<256x128x28x28xf32>
    %v3667 = stablehlo.add %v3666, %v3658 : tensor<256x128x28x28xf32>
    %v3668 = stablehlo.rsqrt %v3667 : tensor<256x128x28x28xf32>
    %v3669 = stablehlo.multiply %v3662, %v3668 : tensor<256x128x28x28xf32>
    %v3670 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3671 = stablehlo.reshape %v3654 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3672 = stablehlo.multiply %v3670, %v3671 : tensor<256x128x28x28xf32>
    %v3673 = stablehlo.reduce(%v3672 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3674 = stablehlo.broadcast_in_dim %v3673, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3675 = stablehlo.multiply %v3669, %v3672 : tensor<256x128x28x28xf32>
    %v3676 = stablehlo.reduce(%v3675 init: %v3656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3677 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3678 = stablehlo.multiply %v3672, %v3657 : tensor<256x128x28x28xf32>
    %v3679 = stablehlo.subtract %v3678, %v3674 : tensor<256x128x28x28xf32>
    %v3680 = stablehlo.multiply %v3669, %v3677 : tensor<256x128x28x28xf32>
    %v3681 = stablehlo.subtract %v3679, %v3680 : tensor<256x128x28x28xf32>
    %v3682 = stablehlo.divide %v3668, %v3657 : tensor<256x128x28x28xf32>
    %v3683 = stablehlo.multiply %v3682, %v3681 : tensor<256x128x28x28xf32>
    %v3684 = stablehlo.reshape %v3683 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3685 = stablehlo.reshape %v3684 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3686 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3687 = stablehlo.transpose %v3686, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3688 = stablehlo.convolution(%v3685, %v3687)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3689 = stablehlo.reshape %v3688 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3690 = stablehlo.add %v3689, %v3578 : tensor<256x401408xf32>
    %v3691 = stablehlo.reshape %v490 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3692 = stablehlo.reshape %v3684 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3693 = stablehlo.transpose %v3691, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3694 = stablehlo.transpose %v3692, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3695 = stablehlo.convolution(%v3693, %v3694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3696 = stablehlo.transpose %v3695, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3697 = stablehlo.reshape %v495 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3699 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3700 = stablehlo.reduce(%v3697 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3702 = stablehlo.divide %v3701, %v3699 : tensor<256x128x28x28xf32>
    %v3703 = stablehlo.subtract %v3697, %v3702 : tensor<256x128x28x28xf32>
    %v3704 = stablehlo.multiply %v3703, %v3703 : tensor<256x128x28x28xf32>
    %v3705 = stablehlo.reduce(%v3704 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3706 = stablehlo.broadcast_in_dim %v3705, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3707 = stablehlo.divide %v3706, %v3699 : tensor<256x128x28x28xf32>
    %v3708 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3709 = stablehlo.add %v3707, %v3708 : tensor<256x128x28x28xf32>
    %v3710 = stablehlo.rsqrt %v3709 : tensor<256x128x28x28xf32>
    %v3711 = stablehlo.multiply %v3703, %v3710 : tensor<256x128x28x28xf32>
    %v3712 = stablehlo.reshape %v3654 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3713 = stablehlo.multiply %v3712, %v3711 : tensor<256x128x28x28xf32>
    %v3714 = stablehlo.reduce(%v3713 init: %v3698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3715 = stablehlo.reshape %v3654 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3717 = stablehlo.reduce(%v3715 init: %v3716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3718 = stablehlo.reshape %v517 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3719 = stablehlo.reshape %v3646 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3720 = stablehlo.transpose %v3718, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3721 = stablehlo.transpose %v3719, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3722 = stablehlo.convolution(%v3720, %v3721)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3723 = stablehlo.transpose %v3722, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3724 = stablehlo.reshape %v522 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3726 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3727 = stablehlo.reduce(%v3724 init: %v3725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3728 = stablehlo.broadcast_in_dim %v3727, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3729 = stablehlo.divide %v3728, %v3726 : tensor<256x128x28x28xf32>
    %v3730 = stablehlo.subtract %v3724, %v3729 : tensor<256x128x28x28xf32>
    %v3731 = stablehlo.multiply %v3730, %v3730 : tensor<256x128x28x28xf32>
    %v3732 = stablehlo.reduce(%v3731 init: %v3725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3733 = stablehlo.broadcast_in_dim %v3732, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3734 = stablehlo.divide %v3733, %v3726 : tensor<256x128x28x28xf32>
    %v3735 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3736 = stablehlo.add %v3734, %v3735 : tensor<256x128x28x28xf32>
    %v3737 = stablehlo.rsqrt %v3736 : tensor<256x128x28x28xf32>
    %v3738 = stablehlo.multiply %v3730, %v3737 : tensor<256x128x28x28xf32>
    %v3739 = stablehlo.reshape %v3616 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3740 = stablehlo.multiply %v3739, %v3738 : tensor<256x128x28x28xf32>
    %v3741 = stablehlo.reduce(%v3740 init: %v3725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3742 = stablehlo.reshape %v3616 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3744 = stablehlo.reduce(%v3742 init: %v3743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3745 = stablehlo.reshape %v544 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3746 = stablehlo.reshape %v3608 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3747 = stablehlo.transpose %v3745, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3748 = stablehlo.transpose %v3746, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3749 = stablehlo.convolution(%v3747, %v3748)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v3750 = stablehlo.transpose %v3749, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3751 = stablehlo.reshape %v549 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3753 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3754 = stablehlo.reduce(%v3751 init: %v3752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3755 = stablehlo.broadcast_in_dim %v3754, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3756 = stablehlo.divide %v3755, %v3753 : tensor<256x512x28x28xf32>
    %v3757 = stablehlo.subtract %v3751, %v3756 : tensor<256x512x28x28xf32>
    %v3758 = stablehlo.multiply %v3757, %v3757 : tensor<256x512x28x28xf32>
    %v3759 = stablehlo.reduce(%v3758 init: %v3752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3760 = stablehlo.broadcast_in_dim %v3759, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3761 = stablehlo.divide %v3760, %v3753 : tensor<256x512x28x28xf32>
    %v3762 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3763 = stablehlo.add %v3761, %v3762 : tensor<256x512x28x28xf32>
    %v3764 = stablehlo.rsqrt %v3763 : tensor<256x512x28x28xf32>
    %v3765 = stablehlo.multiply %v3757, %v3764 : tensor<256x512x28x28xf32>
    %v3766 = stablehlo.reshape %v3578 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3767 = stablehlo.multiply %v3766, %v3765 : tensor<256x512x28x28xf32>
    %v3768 = stablehlo.reduce(%v3767 init: %v3752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3769 = stablehlo.reshape %v3578 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3771 = stablehlo.reduce(%v3769 init: %v3770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3772 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v3773 = stablehlo.compare GT, %v488, %v3772 : (tensor<256x401408xf32>, tensor<256x401408xf32>) -> tensor<256x401408xi1>
    %v3774 = stablehlo.select %v3773, %v3690, %v3772 : tensor<256x401408xi1>, tensor<256x401408xf32>
    %v3775 = stablehlo.reshape %v467 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3777 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3778 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3779 = stablehlo.reduce(%v3775 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3780 = stablehlo.broadcast_in_dim %v3779, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3781 = stablehlo.divide %v3780, %v3777 : tensor<256x512x28x28xf32>
    %v3782 = stablehlo.subtract %v3775, %v3781 : tensor<256x512x28x28xf32>
    %v3783 = stablehlo.multiply %v3782, %v3782 : tensor<256x512x28x28xf32>
    %v3784 = stablehlo.reduce(%v3783 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3785 = stablehlo.broadcast_in_dim %v3784, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3786 = stablehlo.divide %v3785, %v3777 : tensor<256x512x28x28xf32>
    %v3787 = stablehlo.add %v3786, %v3778 : tensor<256x512x28x28xf32>
    %v3788 = stablehlo.rsqrt %v3787 : tensor<256x512x28x28xf32>
    %v3789 = stablehlo.multiply %v3782, %v3788 : tensor<256x512x28x28xf32>
    %v3790 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3791 = stablehlo.reshape %v3774 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3792 = stablehlo.multiply %v3790, %v3791 : tensor<256x512x28x28xf32>
    %v3793 = stablehlo.reduce(%v3792 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3794 = stablehlo.broadcast_in_dim %v3793, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3795 = stablehlo.multiply %v3789, %v3792 : tensor<256x512x28x28xf32>
    %v3796 = stablehlo.reduce(%v3795 init: %v3776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3797 = stablehlo.broadcast_in_dim %v3796, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3798 = stablehlo.multiply %v3792, %v3777 : tensor<256x512x28x28xf32>
    %v3799 = stablehlo.subtract %v3798, %v3794 : tensor<256x512x28x28xf32>
    %v3800 = stablehlo.multiply %v3789, %v3797 : tensor<256x512x28x28xf32>
    %v3801 = stablehlo.subtract %v3799, %v3800 : tensor<256x512x28x28xf32>
    %v3802 = stablehlo.divide %v3788, %v3777 : tensor<256x512x28x28xf32>
    %v3803 = stablehlo.multiply %v3802, %v3801 : tensor<256x512x28x28xf32>
    %v3804 = stablehlo.reshape %v3803 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3805 = stablehlo.reshape %v3804 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3806 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3807 = stablehlo.transpose %v3806, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3808 = stablehlo.convolution(%v3805, %v3807)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v3809 = stablehlo.reshape %v3808 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3811 = stablehlo.compare GT, %v460, %v3810 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3812 = stablehlo.select %v3811, %v3809, %v3810 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3813 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3814 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3815 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3816 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3817 = stablehlo.reduce(%v3813 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3819 = stablehlo.divide %v3818, %v3815 : tensor<256x128x28x28xf32>
    %v3820 = stablehlo.subtract %v3813, %v3819 : tensor<256x128x28x28xf32>
    %v3821 = stablehlo.multiply %v3820, %v3820 : tensor<256x128x28x28xf32>
    %v3822 = stablehlo.reduce(%v3821 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3823 = stablehlo.broadcast_in_dim %v3822, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3824 = stablehlo.divide %v3823, %v3815 : tensor<256x128x28x28xf32>
    %v3825 = stablehlo.add %v3824, %v3816 : tensor<256x128x28x28xf32>
    %v3826 = stablehlo.rsqrt %v3825 : tensor<256x128x28x28xf32>
    %v3827 = stablehlo.multiply %v3820, %v3826 : tensor<256x128x28x28xf32>
    %v3828 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3829 = stablehlo.reshape %v3812 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3830 = stablehlo.multiply %v3828, %v3829 : tensor<256x128x28x28xf32>
    %v3831 = stablehlo.reduce(%v3830 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3832 = stablehlo.broadcast_in_dim %v3831, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3833 = stablehlo.multiply %v3827, %v3830 : tensor<256x128x28x28xf32>
    %v3834 = stablehlo.reduce(%v3833 init: %v3814) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3835 = stablehlo.broadcast_in_dim %v3834, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3836 = stablehlo.multiply %v3830, %v3815 : tensor<256x128x28x28xf32>
    %v3837 = stablehlo.subtract %v3836, %v3832 : tensor<256x128x28x28xf32>
    %v3838 = stablehlo.multiply %v3827, %v3835 : tensor<256x128x28x28xf32>
    %v3839 = stablehlo.subtract %v3837, %v3838 : tensor<256x128x28x28xf32>
    %v3840 = stablehlo.divide %v3826, %v3815 : tensor<256x128x28x28xf32>
    %v3841 = stablehlo.multiply %v3840, %v3839 : tensor<256x128x28x28xf32>
    %v3842 = stablehlo.reshape %v3841 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3843 = stablehlo.reshape %v3842 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3844 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3845 = stablehlo.transpose %v3844, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3846 = stablehlo.convolution(%v3843, %v3845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x28x28xf32>
    %v3847 = stablehlo.reshape %v3846 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3848 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v3849 = stablehlo.compare GT, %v433, %v3848 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v3850 = stablehlo.select %v3849, %v3847, %v3848 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v3851 = stablehlo.reshape %v413 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3853 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3854 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3855 = stablehlo.reduce(%v3851 init: %v3852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3856 = stablehlo.broadcast_in_dim %v3855, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3857 = stablehlo.divide %v3856, %v3853 : tensor<256x128x28x28xf32>
    %v3858 = stablehlo.subtract %v3851, %v3857 : tensor<256x128x28x28xf32>
    %v3859 = stablehlo.multiply %v3858, %v3858 : tensor<256x128x28x28xf32>
    %v3860 = stablehlo.reduce(%v3859 init: %v3852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3861 = stablehlo.broadcast_in_dim %v3860, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3862 = stablehlo.divide %v3861, %v3853 : tensor<256x128x28x28xf32>
    %v3863 = stablehlo.add %v3862, %v3854 : tensor<256x128x28x28xf32>
    %v3864 = stablehlo.rsqrt %v3863 : tensor<256x128x28x28xf32>
    %v3865 = stablehlo.multiply %v3858, %v3864 : tensor<256x128x28x28xf32>
    %v3866 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3867 = stablehlo.reshape %v3850 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3868 = stablehlo.multiply %v3866, %v3867 : tensor<256x128x28x28xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3871 = stablehlo.multiply %v3865, %v3868 : tensor<256x128x28x28xf32>
    %v3872 = stablehlo.reduce(%v3871 init: %v3852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3873 = stablehlo.broadcast_in_dim %v3872, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3874 = stablehlo.multiply %v3868, %v3853 : tensor<256x128x28x28xf32>
    %v3875 = stablehlo.subtract %v3874, %v3870 : tensor<256x128x28x28xf32>
    %v3876 = stablehlo.multiply %v3865, %v3873 : tensor<256x128x28x28xf32>
    %v3877 = stablehlo.subtract %v3875, %v3876 : tensor<256x128x28x28xf32>
    %v3878 = stablehlo.divide %v3864, %v3853 : tensor<256x128x28x28xf32>
    %v3879 = stablehlo.multiply %v3878, %v3877 : tensor<256x128x28x28xf32>
    %v3880 = stablehlo.reshape %v3879 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v3881 = stablehlo.reshape %v3880 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3882 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3883 = stablehlo.transpose %v3882, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3884 = stablehlo.convolution(%v3881, %v3883)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<256x512x28x28xf32>
    %v3885 = stablehlo.reshape %v3884 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v3886 = stablehlo.add %v3885, %v3774 : tensor<256x401408xf32>
    %v3887 = stablehlo.reshape %v408 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3888 = stablehlo.reshape %v3880 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3889 = stablehlo.transpose %v3887, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3890 = stablehlo.transpose %v3888, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3891 = stablehlo.convolution(%v3889, %v3890)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v3892 = stablehlo.transpose %v3891, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3893 = stablehlo.reshape %v413 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3895 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3896 = stablehlo.reduce(%v3893 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3897 = stablehlo.broadcast_in_dim %v3896, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3898 = stablehlo.divide %v3897, %v3895 : tensor<256x128x28x28xf32>
    %v3899 = stablehlo.subtract %v3893, %v3898 : tensor<256x128x28x28xf32>
    %v3900 = stablehlo.multiply %v3899, %v3899 : tensor<256x128x28x28xf32>
    %v3901 = stablehlo.reduce(%v3900 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3902 = stablehlo.broadcast_in_dim %v3901, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3903 = stablehlo.divide %v3902, %v3895 : tensor<256x128x28x28xf32>
    %v3904 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3905 = stablehlo.add %v3903, %v3904 : tensor<256x128x28x28xf32>
    %v3906 = stablehlo.rsqrt %v3905 : tensor<256x128x28x28xf32>
    %v3907 = stablehlo.multiply %v3899, %v3906 : tensor<256x128x28x28xf32>
    %v3908 = stablehlo.reshape %v3850 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3909 = stablehlo.multiply %v3908, %v3907 : tensor<256x128x28x28xf32>
    %v3910 = stablehlo.reduce(%v3909 init: %v3894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3911 = stablehlo.reshape %v3850 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3913 = stablehlo.reduce(%v3911 init: %v3912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3914 = stablehlo.reshape %v435 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3915 = stablehlo.reshape %v3842 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3916 = stablehlo.transpose %v3914, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3917 = stablehlo.transpose %v3915, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3918 = stablehlo.convolution(%v3916, %v3917)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<128x256x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v3919 = stablehlo.transpose %v3918, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3920 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3922 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v3923 = stablehlo.reduce(%v3920 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3924 = stablehlo.broadcast_in_dim %v3923, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3925 = stablehlo.divide %v3924, %v3922 : tensor<256x128x28x28xf32>
    %v3926 = stablehlo.subtract %v3920, %v3925 : tensor<256x128x28x28xf32>
    %v3927 = stablehlo.multiply %v3926, %v3926 : tensor<256x128x28x28xf32>
    %v3928 = stablehlo.reduce(%v3927 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3928, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v3930 = stablehlo.divide %v3929, %v3922 : tensor<256x128x28x28xf32>
    %v3931 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v3932 = stablehlo.add %v3930, %v3931 : tensor<256x128x28x28xf32>
    %v3933 = stablehlo.rsqrt %v3932 : tensor<256x128x28x28xf32>
    %v3934 = stablehlo.multiply %v3926, %v3933 : tensor<256x128x28x28xf32>
    %v3935 = stablehlo.reshape %v3812 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3936 = stablehlo.multiply %v3935, %v3934 : tensor<256x128x28x28xf32>
    %v3937 = stablehlo.reduce(%v3936 init: %v3921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3938 = stablehlo.reshape %v3812 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3940 = stablehlo.reduce(%v3938 init: %v3939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3941 = stablehlo.reshape %v462 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v3942 = stablehlo.reshape %v3804 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3943 = stablehlo.transpose %v3941, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v3944 = stablehlo.transpose %v3942, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v3945 = stablehlo.convolution(%v3943, %v3944)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v3946 = stablehlo.transpose %v3945, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3947 = stablehlo.reshape %v467 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3949 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3950 = stablehlo.reduce(%v3947 init: %v3948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3951 = stablehlo.broadcast_in_dim %v3950, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3952 = stablehlo.divide %v3951, %v3949 : tensor<256x512x28x28xf32>
    %v3953 = stablehlo.subtract %v3947, %v3952 : tensor<256x512x28x28xf32>
    %v3954 = stablehlo.multiply %v3953, %v3953 : tensor<256x512x28x28xf32>
    %v3955 = stablehlo.reduce(%v3954 init: %v3948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3956 = stablehlo.broadcast_in_dim %v3955, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3957 = stablehlo.divide %v3956, %v3949 : tensor<256x512x28x28xf32>
    %v3958 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3959 = stablehlo.add %v3957, %v3958 : tensor<256x512x28x28xf32>
    %v3960 = stablehlo.rsqrt %v3959 : tensor<256x512x28x28xf32>
    %v3961 = stablehlo.multiply %v3953, %v3960 : tensor<256x512x28x28xf32>
    %v3962 = stablehlo.reshape %v3774 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3963 = stablehlo.multiply %v3962, %v3961 : tensor<256x512x28x28xf32>
    %v3964 = stablehlo.reduce(%v3963 init: %v3948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3965 = stablehlo.reshape %v3774 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3966 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3967 = stablehlo.reduce(%v3965 init: %v3966) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3968 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v3969 = stablehlo.compare GT, %v406, %v3968 : (tensor<256x401408xf32>, tensor<256x401408xf32>) -> tensor<256x401408xi1>
    %v3970 = stablehlo.select %v3969, %v3886, %v3968 : tensor<256x401408xi1>, tensor<256x401408xf32>
    %v3971 = stablehlo.reshape %v360 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3973 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v3974 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v3975 = stablehlo.reduce(%v3971 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3976 = stablehlo.broadcast_in_dim %v3975, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3977 = stablehlo.divide %v3976, %v3973 : tensor<256x512x28x28xf32>
    %v3978 = stablehlo.subtract %v3971, %v3977 : tensor<256x512x28x28xf32>
    %v3979 = stablehlo.multiply %v3978, %v3978 : tensor<256x512x28x28xf32>
    %v3980 = stablehlo.reduce(%v3979 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3980, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3982 = stablehlo.divide %v3981, %v3973 : tensor<256x512x28x28xf32>
    %v3983 = stablehlo.add %v3982, %v3974 : tensor<256x512x28x28xf32>
    %v3984 = stablehlo.rsqrt %v3983 : tensor<256x512x28x28xf32>
    %v3985 = stablehlo.multiply %v3978, %v3984 : tensor<256x512x28x28xf32>
    %v3986 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3987 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v3988 = stablehlo.multiply %v3986, %v3987 : tensor<256x512x28x28xf32>
    %v3989 = stablehlo.reduce(%v3988 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3990 = stablehlo.broadcast_in_dim %v3989, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3991 = stablehlo.multiply %v3985, %v3988 : tensor<256x512x28x28xf32>
    %v3992 = stablehlo.reduce(%v3991 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3993 = stablehlo.broadcast_in_dim %v3992, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v3994 = stablehlo.multiply %v3988, %v3973 : tensor<256x512x28x28xf32>
    %v3995 = stablehlo.subtract %v3994, %v3990 : tensor<256x512x28x28xf32>
    %v3996 = stablehlo.multiply %v3985, %v3993 : tensor<256x512x28x28xf32>
    %v3997 = stablehlo.subtract %v3995, %v3996 : tensor<256x512x28x28xf32>
    %v3998 = stablehlo.divide %v3984, %v3973 : tensor<256x512x28x28xf32>
    %v3999 = stablehlo.multiply %v3998, %v3997 : tensor<256x512x28x28xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4002 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4003 = stablehlo.transpose %v4002, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4004 = stablehlo.convolution(%v4001, %v4003)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<256x128x28x28xf32>
    %v4005 = stablehlo.reshape %v4004 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4006 = stablehlo.constant dense<0.0> : tensor<256x100352xf32>
    %v4007 = stablehlo.compare GT, %v353, %v4006 : (tensor<256x100352xf32>, tensor<256x100352xf32>) -> tensor<256x100352xi1>
    %v4008 = stablehlo.select %v4007, %v4005, %v4006 : tensor<256x100352xi1>, tensor<256x100352xf32>
    %v4009 = stablehlo.reshape %v333 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4010 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4011 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4012 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4013 = stablehlo.reduce(%v4009 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4014 = stablehlo.broadcast_in_dim %v4013, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4015 = stablehlo.divide %v4014, %v4011 : tensor<256x128x28x28xf32>
    %v4016 = stablehlo.subtract %v4009, %v4015 : tensor<256x128x28x28xf32>
    %v4017 = stablehlo.multiply %v4016, %v4016 : tensor<256x128x28x28xf32>
    %v4018 = stablehlo.reduce(%v4017 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4019 = stablehlo.broadcast_in_dim %v4018, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4020 = stablehlo.divide %v4019, %v4011 : tensor<256x128x28x28xf32>
    %v4021 = stablehlo.add %v4020, %v4012 : tensor<256x128x28x28xf32>
    %v4022 = stablehlo.rsqrt %v4021 : tensor<256x128x28x28xf32>
    %v4023 = stablehlo.multiply %v4016, %v4022 : tensor<256x128x28x28xf32>
    %v4024 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4025 = stablehlo.reshape %v4008 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4026 = stablehlo.multiply %v4024, %v4025 : tensor<256x128x28x28xf32>
    %v4027 = stablehlo.reduce(%v4026 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4028 = stablehlo.broadcast_in_dim %v4027, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4029 = stablehlo.multiply %v4023, %v4026 : tensor<256x128x28x28xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4010) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4031 = stablehlo.broadcast_in_dim %v4030, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4032 = stablehlo.multiply %v4026, %v4011 : tensor<256x128x28x28xf32>
    %v4033 = stablehlo.subtract %v4032, %v4028 : tensor<256x128x28x28xf32>
    %v4034 = stablehlo.multiply %v4023, %v4031 : tensor<256x128x28x28xf32>
    %v4035 = stablehlo.subtract %v4033, %v4034 : tensor<256x128x28x28xf32>
    %v4036 = stablehlo.divide %v4022, %v4011 : tensor<256x128x28x28xf32>
    %v4037 = stablehlo.multiply %v4036, %v4035 : tensor<256x128x28x28xf32>
    %v4038 = stablehlo.reshape %v4037 : (tensor<256x128x28x28xf32>) -> tensor<256x100352xf32>
    %v4039 = stablehlo.reshape %v4038 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4041 = stablehlo.pad %v4039, %v4040, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v4042 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4043 = stablehlo.transpose %v4042, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4044 = stablehlo.convolution(%v4041, %v4043)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<256x128x56x56xf32>
    %v4045 = stablehlo.reshape %v4044 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4046 = stablehlo.constant dense<0.0> : tensor<256x401408xf32>
    %v4047 = stablehlo.compare GT, %v326, %v4046 : (tensor<256x401408xf32>, tensor<256x401408xf32>) -> tensor<256x401408xi1>
    %v4048 = stablehlo.select %v4047, %v4045, %v4046 : tensor<256x401408xi1>, tensor<256x401408xf32>
    %v4049 = stablehlo.reshape %v306 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4052 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4053 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4053, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4055 = stablehlo.divide %v4054, %v4051 : tensor<256x128x56x56xf32>
    %v4056 = stablehlo.subtract %v4049, %v4055 : tensor<256x128x56x56xf32>
    %v4057 = stablehlo.multiply %v4056, %v4056 : tensor<256x128x56x56xf32>
    %v4058 = stablehlo.reduce(%v4057 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4059 = stablehlo.broadcast_in_dim %v4058, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4060 = stablehlo.divide %v4059, %v4051 : tensor<256x128x56x56xf32>
    %v4061 = stablehlo.add %v4060, %v4052 : tensor<256x128x56x56xf32>
    %v4062 = stablehlo.rsqrt %v4061 : tensor<256x128x56x56xf32>
    %v4063 = stablehlo.multiply %v4056, %v4062 : tensor<256x128x56x56xf32>
    %v4064 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4065 = stablehlo.reshape %v4048 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4066 = stablehlo.multiply %v4064, %v4065 : tensor<256x128x56x56xf32>
    %v4067 = stablehlo.reduce(%v4066 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4068 = stablehlo.broadcast_in_dim %v4067, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4069 = stablehlo.multiply %v4063, %v4066 : tensor<256x128x56x56xf32>
    %v4070 = stablehlo.reduce(%v4069 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4071 = stablehlo.broadcast_in_dim %v4070, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4072 = stablehlo.multiply %v4066, %v4051 : tensor<256x128x56x56xf32>
    %v4073 = stablehlo.subtract %v4072, %v4068 : tensor<256x128x56x56xf32>
    %v4074 = stablehlo.multiply %v4063, %v4071 : tensor<256x128x56x56xf32>
    %v4075 = stablehlo.subtract %v4073, %v4074 : tensor<256x128x56x56xf32>
    %v4076 = stablehlo.divide %v4062, %v4051 : tensor<256x128x56x56xf32>
    %v4077 = stablehlo.multiply %v4076, %v4075 : tensor<256x128x56x56xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<256x128x56x56xf32>) -> tensor<256x401408xf32>
    %v4079 = stablehlo.reshape %v4078 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4080 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v4081 = stablehlo.transpose %v4080, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v4082 = stablehlo.convolution(%v4079, %v4081)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x128x56x56xf32>, tensor<256x128x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4083 = stablehlo.reshape %v4082 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4084 = stablehlo.reshape %v385 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4086 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4087 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4088 = stablehlo.reduce(%v4084 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4089 = stablehlo.broadcast_in_dim %v4088, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4090 = stablehlo.divide %v4089, %v4086 : tensor<256x512x28x28xf32>
    %v4091 = stablehlo.subtract %v4084, %v4090 : tensor<256x512x28x28xf32>
    %v4092 = stablehlo.multiply %v4091, %v4091 : tensor<256x512x28x28xf32>
    %v4093 = stablehlo.reduce(%v4092 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4095 = stablehlo.divide %v4094, %v4086 : tensor<256x512x28x28xf32>
    %v4096 = stablehlo.add %v4095, %v4087 : tensor<256x512x28x28xf32>
    %v4097 = stablehlo.rsqrt %v4096 : tensor<256x512x28x28xf32>
    %v4098 = stablehlo.multiply %v4091, %v4097 : tensor<256x512x28x28xf32>
    %v4099 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4100 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4101 = stablehlo.multiply %v4099, %v4100 : tensor<256x512x28x28xf32>
    %v4102 = stablehlo.reduce(%v4101 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4103 = stablehlo.broadcast_in_dim %v4102, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4104 = stablehlo.multiply %v4098, %v4101 : tensor<256x512x28x28xf32>
    %v4105 = stablehlo.reduce(%v4104 init: %v4085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4107 = stablehlo.multiply %v4101, %v4086 : tensor<256x512x28x28xf32>
    %v4108 = stablehlo.subtract %v4107, %v4103 : tensor<256x512x28x28xf32>
    %v4109 = stablehlo.multiply %v4098, %v4106 : tensor<256x512x28x28xf32>
    %v4110 = stablehlo.subtract %v4108, %v4109 : tensor<256x512x28x28xf32>
    %v4111 = stablehlo.divide %v4097, %v4086 : tensor<256x512x28x28xf32>
    %v4112 = stablehlo.multiply %v4111, %v4110 : tensor<256x512x28x28xf32>
    %v4113 = stablehlo.reshape %v4112 : (tensor<256x512x28x28xf32>) -> tensor<256x401408xf32>
    %v4114 = stablehlo.reshape %v4113 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4116 = stablehlo.pad %v4114, %v4115, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<256x512x56x56xf32>
    %v4117 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v4118 = stablehlo.transpose %v4117, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4119 = stablehlo.convolution(%v4116, %v4118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x512x56x56xf32>, tensor<256x512x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4120 = stablehlo.reshape %v4119 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4121 = stablehlo.add %v4083, %v4120 : tensor<256x802816xf32>
    %v4122 = stablehlo.reshape %v301 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4123 = stablehlo.reshape %v4078 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4124 = stablehlo.transpose %v4122, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4125 = stablehlo.transpose %v4123, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4126 = stablehlo.convolution(%v4124, %v4125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<256x128x1x1xf32>
    %v4127 = stablehlo.transpose %v4126, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v4128 = stablehlo.reshape %v306 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4130 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v4131 = stablehlo.reduce(%v4128 init: %v4129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4132 = stablehlo.broadcast_in_dim %v4131, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4133 = stablehlo.divide %v4132, %v4130 : tensor<256x128x56x56xf32>
    %v4134 = stablehlo.subtract %v4128, %v4133 : tensor<256x128x56x56xf32>
    %v4135 = stablehlo.multiply %v4134, %v4134 : tensor<256x128x56x56xf32>
    %v4136 = stablehlo.reduce(%v4135 init: %v4129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4137 = stablehlo.broadcast_in_dim %v4136, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v4138 = stablehlo.divide %v4137, %v4130 : tensor<256x128x56x56xf32>
    %v4139 = stablehlo.constant dense<1.0e-05> : tensor<256x128x56x56xf32>
    %v4140 = stablehlo.add %v4138, %v4139 : tensor<256x128x56x56xf32>
    %v4141 = stablehlo.rsqrt %v4140 : tensor<256x128x56x56xf32>
    %v4142 = stablehlo.multiply %v4134, %v4141 : tensor<256x128x56x56xf32>
    %v4143 = stablehlo.reshape %v4048 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4144 = stablehlo.multiply %v4143, %v4142 : tensor<256x128x56x56xf32>
    %v4145 = stablehlo.reduce(%v4144 init: %v4129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4146 = stablehlo.reshape %v4048 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4148 = stablehlo.reduce(%v4146 init: %v4147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4149 = stablehlo.reshape %v328 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v4150 = stablehlo.reshape %v4038 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4152 = stablehlo.pad %v4150, %v4151, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<256x128x56x56xf32>
    %v4153 = stablehlo.transpose %v4149, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4154 = stablehlo.transpose %v4152, dims = [1, 0, 2, 3] : (tensor<256x128x56x56xf32>) -> tensor<128x256x56x56xf32>
    %v4155 = stablehlo.convolution(%v4153, %v4154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x56x56xf32>, tensor<128x256x56x56xf32>) -> tensor<128x128x3x3xf32>
    %v4156 = stablehlo.transpose %v4155, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4157 = stablehlo.reshape %v333 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4159 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v4160 = stablehlo.reduce(%v4157 init: %v4158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4161 = stablehlo.broadcast_in_dim %v4160, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4162 = stablehlo.divide %v4161, %v4159 : tensor<256x128x28x28xf32>
    %v4163 = stablehlo.subtract %v4157, %v4162 : tensor<256x128x28x28xf32>
    %v4164 = stablehlo.multiply %v4163, %v4163 : tensor<256x128x28x28xf32>
    %v4165 = stablehlo.reduce(%v4164 init: %v4158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4166 = stablehlo.broadcast_in_dim %v4165, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v4167 = stablehlo.divide %v4166, %v4159 : tensor<256x128x28x28xf32>
    %v4168 = stablehlo.constant dense<1.0e-05> : tensor<256x128x28x28xf32>
    %v4169 = stablehlo.add %v4167, %v4168 : tensor<256x128x28x28xf32>
    %v4170 = stablehlo.rsqrt %v4169 : tensor<256x128x28x28xf32>
    %v4171 = stablehlo.multiply %v4163, %v4170 : tensor<256x128x28x28xf32>
    %v4172 = stablehlo.reshape %v4008 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4173 = stablehlo.multiply %v4172, %v4171 : tensor<256x128x28x28xf32>
    %v4174 = stablehlo.reduce(%v4173 init: %v4158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4175 = stablehlo.reshape %v4008 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4177 = stablehlo.reduce(%v4175 init: %v4176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4178 = stablehlo.reshape %v355 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v4179 = stablehlo.reshape %v4000 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4180 = stablehlo.transpose %v4178, dims = [1, 0, 2, 3] : (tensor<256x128x28x28xf32>) -> tensor<128x256x28x28xf32>
    %v4181 = stablehlo.transpose %v4179, dims = [1, 0, 2, 3] : (tensor<256x512x28x28xf32>) -> tensor<512x256x28x28xf32>
    %v4182 = stablehlo.convolution(%v4180, %v4181)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x256x28x28xf32>, tensor<512x256x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4183 = stablehlo.transpose %v4182, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4184 = stablehlo.reshape %v360 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4186 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4187 = stablehlo.reduce(%v4184 init: %v4185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4188 = stablehlo.broadcast_in_dim %v4187, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4189 = stablehlo.divide %v4188, %v4186 : tensor<256x512x28x28xf32>
    %v4190 = stablehlo.subtract %v4184, %v4189 : tensor<256x512x28x28xf32>
    %v4191 = stablehlo.multiply %v4190, %v4190 : tensor<256x512x28x28xf32>
    %v4192 = stablehlo.reduce(%v4191 init: %v4185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4193 = stablehlo.broadcast_in_dim %v4192, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4194 = stablehlo.divide %v4193, %v4186 : tensor<256x512x28x28xf32>
    %v4195 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4196 = stablehlo.add %v4194, %v4195 : tensor<256x512x28x28xf32>
    %v4197 = stablehlo.rsqrt %v4196 : tensor<256x512x28x28xf32>
    %v4198 = stablehlo.multiply %v4190, %v4197 : tensor<256x512x28x28xf32>
    %v4199 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4200 = stablehlo.multiply %v4199, %v4198 : tensor<256x512x28x28xf32>
    %v4201 = stablehlo.reduce(%v4200 init: %v4185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4202 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4204 = stablehlo.reduce(%v4202 init: %v4203) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4205 = stablehlo.reshape %v301 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4206 = stablehlo.reshape %v4113 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4208 = stablehlo.pad %v4206, %v4207, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<256x512x56x56xf32>
    %v4209 = stablehlo.transpose %v4205, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4210 = stablehlo.transpose %v4208, dims = [1, 0, 2, 3] : (tensor<256x512x56x56xf32>) -> tensor<512x256x56x56xf32>
    %v4211 = stablehlo.convolution(%v4209, %v4210)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<512x256x56x56xf32>) -> tensor<256x512x1x1xf32>
    %v4212 = stablehlo.transpose %v4211, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4213 = stablehlo.reshape %v385 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4215 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v4216 = stablehlo.reduce(%v4213 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4217 = stablehlo.broadcast_in_dim %v4216, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4218 = stablehlo.divide %v4217, %v4215 : tensor<256x512x28x28xf32>
    %v4219 = stablehlo.subtract %v4213, %v4218 : tensor<256x512x28x28xf32>
    %v4220 = stablehlo.multiply %v4219, %v4219 : tensor<256x512x28x28xf32>
    %v4221 = stablehlo.reduce(%v4220 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4222 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v4223 = stablehlo.divide %v4222, %v4215 : tensor<256x512x28x28xf32>
    %v4224 = stablehlo.constant dense<1.0e-05> : tensor<256x512x28x28xf32>
    %v4225 = stablehlo.add %v4223, %v4224 : tensor<256x512x28x28xf32>
    %v4226 = stablehlo.rsqrt %v4225 : tensor<256x512x28x28xf32>
    %v4227 = stablehlo.multiply %v4219, %v4226 : tensor<256x512x28x28xf32>
    %v4228 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4229 = stablehlo.multiply %v4228, %v4227 : tensor<256x512x28x28xf32>
    %v4230 = stablehlo.reduce(%v4229 init: %v4214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4231 = stablehlo.reshape %v3970 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v4232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4233 = stablehlo.reduce(%v4231 init: %v4232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4234 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v4235 = stablehlo.compare GT, %v299, %v4234 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v4236 = stablehlo.select %v4235, %v4121, %v4234 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v4237 = stablehlo.reshape %v278 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4239 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4240 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4241 = stablehlo.reduce(%v4237 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4242 = stablehlo.broadcast_in_dim %v4241, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4243 = stablehlo.divide %v4242, %v4239 : tensor<256x256x56x56xf32>
    %v4244 = stablehlo.subtract %v4237, %v4243 : tensor<256x256x56x56xf32>
    %v4245 = stablehlo.multiply %v4244, %v4244 : tensor<256x256x56x56xf32>
    %v4246 = stablehlo.reduce(%v4245 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4247 = stablehlo.broadcast_in_dim %v4246, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4248 = stablehlo.divide %v4247, %v4239 : tensor<256x256x56x56xf32>
    %v4249 = stablehlo.add %v4248, %v4240 : tensor<256x256x56x56xf32>
    %v4250 = stablehlo.rsqrt %v4249 : tensor<256x256x56x56xf32>
    %v4251 = stablehlo.multiply %v4244, %v4250 : tensor<256x256x56x56xf32>
    %v4252 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4253 = stablehlo.reshape %v4236 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4254 = stablehlo.multiply %v4252, %v4253 : tensor<256x256x56x56xf32>
    %v4255 = stablehlo.reduce(%v4254 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4256 = stablehlo.broadcast_in_dim %v4255, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4257 = stablehlo.multiply %v4251, %v4254 : tensor<256x256x56x56xf32>
    %v4258 = stablehlo.reduce(%v4257 init: %v4238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4259 = stablehlo.broadcast_in_dim %v4258, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4260 = stablehlo.multiply %v4254, %v4239 : tensor<256x256x56x56xf32>
    %v4261 = stablehlo.subtract %v4260, %v4256 : tensor<256x256x56x56xf32>
    %v4262 = stablehlo.multiply %v4251, %v4259 : tensor<256x256x56x56xf32>
    %v4263 = stablehlo.subtract %v4261, %v4262 : tensor<256x256x56x56xf32>
    %v4264 = stablehlo.divide %v4250, %v4239 : tensor<256x256x56x56xf32>
    %v4265 = stablehlo.multiply %v4264, %v4263 : tensor<256x256x56x56xf32>
    %v4266 = stablehlo.reshape %v4265 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4267 = stablehlo.reshape %v4266 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4268 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4269 = stablehlo.transpose %v4268, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4270 = stablehlo.convolution(%v4267, %v4269)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4271 = stablehlo.reshape %v4270 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4272 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4273 = stablehlo.compare GT, %v271, %v4272 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4274 = stablehlo.select %v4273, %v4271, %v4272 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4275 = stablehlo.reshape %v251 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4277 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4278 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4279 = stablehlo.reduce(%v4275 init: %v4276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4280 = stablehlo.broadcast_in_dim %v4279, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4281 = stablehlo.divide %v4280, %v4277 : tensor<256x64x56x56xf32>
    %v4282 = stablehlo.subtract %v4275, %v4281 : tensor<256x64x56x56xf32>
    %v4283 = stablehlo.multiply %v4282, %v4282 : tensor<256x64x56x56xf32>
    %v4284 = stablehlo.reduce(%v4283 init: %v4276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4285 = stablehlo.broadcast_in_dim %v4284, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4286 = stablehlo.divide %v4285, %v4277 : tensor<256x64x56x56xf32>
    %v4287 = stablehlo.add %v4286, %v4278 : tensor<256x64x56x56xf32>
    %v4288 = stablehlo.rsqrt %v4287 : tensor<256x64x56x56xf32>
    %v4289 = stablehlo.multiply %v4282, %v4288 : tensor<256x64x56x56xf32>
    %v4290 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4291 = stablehlo.reshape %v4274 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4292 = stablehlo.multiply %v4290, %v4291 : tensor<256x64x56x56xf32>
    %v4293 = stablehlo.reduce(%v4292 init: %v4276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4294 = stablehlo.broadcast_in_dim %v4293, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4295 = stablehlo.multiply %v4289, %v4292 : tensor<256x64x56x56xf32>
    %v4296 = stablehlo.reduce(%v4295 init: %v4276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4297 = stablehlo.broadcast_in_dim %v4296, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4298 = stablehlo.multiply %v4292, %v4277 : tensor<256x64x56x56xf32>
    %v4299 = stablehlo.subtract %v4298, %v4294 : tensor<256x64x56x56xf32>
    %v4300 = stablehlo.multiply %v4289, %v4297 : tensor<256x64x56x56xf32>
    %v4301 = stablehlo.subtract %v4299, %v4300 : tensor<256x64x56x56xf32>
    %v4302 = stablehlo.divide %v4288, %v4277 : tensor<256x64x56x56xf32>
    %v4303 = stablehlo.multiply %v4302, %v4301 : tensor<256x64x56x56xf32>
    %v4304 = stablehlo.reshape %v4303 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4305 = stablehlo.reshape %v4304 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4306 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4307 = stablehlo.transpose %v4306, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4308 = stablehlo.convolution(%v4305, %v4307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4310 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4311 = stablehlo.compare GT, %v244, %v4310 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4312 = stablehlo.select %v4311, %v4309, %v4310 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4313 = stablehlo.reshape %v224 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4315 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4316 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4317 = stablehlo.reduce(%v4313 init: %v4314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4318 = stablehlo.broadcast_in_dim %v4317, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4319 = stablehlo.divide %v4318, %v4315 : tensor<256x64x56x56xf32>
    %v4320 = stablehlo.subtract %v4313, %v4319 : tensor<256x64x56x56xf32>
    %v4321 = stablehlo.multiply %v4320, %v4320 : tensor<256x64x56x56xf32>
    %v4322 = stablehlo.reduce(%v4321 init: %v4314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4323 = stablehlo.broadcast_in_dim %v4322, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4324 = stablehlo.divide %v4323, %v4315 : tensor<256x64x56x56xf32>
    %v4325 = stablehlo.add %v4324, %v4316 : tensor<256x64x56x56xf32>
    %v4326 = stablehlo.rsqrt %v4325 : tensor<256x64x56x56xf32>
    %v4327 = stablehlo.multiply %v4320, %v4326 : tensor<256x64x56x56xf32>
    %v4328 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4329 = stablehlo.reshape %v4312 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4330 = stablehlo.multiply %v4328, %v4329 : tensor<256x64x56x56xf32>
    %v4331 = stablehlo.reduce(%v4330 init: %v4314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.broadcast_in_dim %v4331, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4333 = stablehlo.multiply %v4327, %v4330 : tensor<256x64x56x56xf32>
    %v4334 = stablehlo.reduce(%v4333 init: %v4314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4335 = stablehlo.broadcast_in_dim %v4334, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4336 = stablehlo.multiply %v4330, %v4315 : tensor<256x64x56x56xf32>
    %v4337 = stablehlo.subtract %v4336, %v4332 : tensor<256x64x56x56xf32>
    %v4338 = stablehlo.multiply %v4327, %v4335 : tensor<256x64x56x56xf32>
    %v4339 = stablehlo.subtract %v4337, %v4338 : tensor<256x64x56x56xf32>
    %v4340 = stablehlo.divide %v4326, %v4315 : tensor<256x64x56x56xf32>
    %v4341 = stablehlo.multiply %v4340, %v4339 : tensor<256x64x56x56xf32>
    %v4342 = stablehlo.reshape %v4341 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4343 = stablehlo.reshape %v4342 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4344 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4345 = stablehlo.transpose %v4344, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4346 = stablehlo.convolution(%v4343, %v4345)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4347 = stablehlo.reshape %v4346 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4348 = stablehlo.add %v4347, %v4236 : tensor<256x802816xf32>
    %v4349 = stablehlo.reshape %v219 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4350 = stablehlo.reshape %v4342 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4351 = stablehlo.transpose %v4349, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4352 = stablehlo.transpose %v4350, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4353 = stablehlo.convolution(%v4351, %v4352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4354 = stablehlo.transpose %v4353, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4355 = stablehlo.reshape %v224 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4357 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4358 = stablehlo.reduce(%v4355 init: %v4356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.broadcast_in_dim %v4358, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4360 = stablehlo.divide %v4359, %v4357 : tensor<256x64x56x56xf32>
    %v4361 = stablehlo.subtract %v4355, %v4360 : tensor<256x64x56x56xf32>
    %v4362 = stablehlo.multiply %v4361, %v4361 : tensor<256x64x56x56xf32>
    %v4363 = stablehlo.reduce(%v4362 init: %v4356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %v4363, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4365 = stablehlo.divide %v4364, %v4357 : tensor<256x64x56x56xf32>
    %v4366 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4367 = stablehlo.add %v4365, %v4366 : tensor<256x64x56x56xf32>
    %v4368 = stablehlo.rsqrt %v4367 : tensor<256x64x56x56xf32>
    %v4369 = stablehlo.multiply %v4361, %v4368 : tensor<256x64x56x56xf32>
    %v4370 = stablehlo.reshape %v4312 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4371 = stablehlo.multiply %v4370, %v4369 : tensor<256x64x56x56xf32>
    %v4372 = stablehlo.reduce(%v4371 init: %v4356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4373 = stablehlo.reshape %v4312 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4375 = stablehlo.reduce(%v4373 init: %v4374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4376 = stablehlo.reshape %v246 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4377 = stablehlo.reshape %v4304 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4378 = stablehlo.transpose %v4376, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4379 = stablehlo.transpose %v4377, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4380 = stablehlo.convolution(%v4378, %v4379)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4381 = stablehlo.transpose %v4380, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4382 = stablehlo.reshape %v251 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4384 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4385 = stablehlo.reduce(%v4382 init: %v4383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4386 = stablehlo.broadcast_in_dim %v4385, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4387 = stablehlo.divide %v4386, %v4384 : tensor<256x64x56x56xf32>
    %v4388 = stablehlo.subtract %v4382, %v4387 : tensor<256x64x56x56xf32>
    %v4389 = stablehlo.multiply %v4388, %v4388 : tensor<256x64x56x56xf32>
    %v4390 = stablehlo.reduce(%v4389 init: %v4383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4391 = stablehlo.broadcast_in_dim %v4390, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4392 = stablehlo.divide %v4391, %v4384 : tensor<256x64x56x56xf32>
    %v4393 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4394 = stablehlo.add %v4392, %v4393 : tensor<256x64x56x56xf32>
    %v4395 = stablehlo.rsqrt %v4394 : tensor<256x64x56x56xf32>
    %v4396 = stablehlo.multiply %v4388, %v4395 : tensor<256x64x56x56xf32>
    %v4397 = stablehlo.reshape %v4274 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4398 = stablehlo.multiply %v4397, %v4396 : tensor<256x64x56x56xf32>
    %v4399 = stablehlo.reduce(%v4398 init: %v4383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4400 = stablehlo.reshape %v4274 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4402 = stablehlo.reduce(%v4400 init: %v4401) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4403 = stablehlo.reshape %v273 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4404 = stablehlo.reshape %v4266 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4405 = stablehlo.transpose %v4403, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4406 = stablehlo.transpose %v4404, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4407 = stablehlo.convolution(%v4405, %v4406)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4408 = stablehlo.transpose %v4407, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4409 = stablehlo.reshape %v278 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4411 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4412 = stablehlo.reduce(%v4409 init: %v4410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4413 = stablehlo.broadcast_in_dim %v4412, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4414 = stablehlo.divide %v4413, %v4411 : tensor<256x256x56x56xf32>
    %v4415 = stablehlo.subtract %v4409, %v4414 : tensor<256x256x56x56xf32>
    %v4416 = stablehlo.multiply %v4415, %v4415 : tensor<256x256x56x56xf32>
    %v4417 = stablehlo.reduce(%v4416 init: %v4410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4418 = stablehlo.broadcast_in_dim %v4417, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4419 = stablehlo.divide %v4418, %v4411 : tensor<256x256x56x56xf32>
    %v4420 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4421 = stablehlo.add %v4419, %v4420 : tensor<256x256x56x56xf32>
    %v4422 = stablehlo.rsqrt %v4421 : tensor<256x256x56x56xf32>
    %v4423 = stablehlo.multiply %v4415, %v4422 : tensor<256x256x56x56xf32>
    %v4424 = stablehlo.reshape %v4236 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4425 = stablehlo.multiply %v4424, %v4423 : tensor<256x256x56x56xf32>
    %v4426 = stablehlo.reduce(%v4425 init: %v4410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4427 = stablehlo.reshape %v4236 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4429 = stablehlo.reduce(%v4427 init: %v4428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4430 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v4431 = stablehlo.compare GT, %v217, %v4430 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v4432 = stablehlo.select %v4431, %v4348, %v4430 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v4433 = stablehlo.reshape %v196 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4435 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4436 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4437 = stablehlo.reduce(%v4433 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4438 = stablehlo.broadcast_in_dim %v4437, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4439 = stablehlo.divide %v4438, %v4435 : tensor<256x256x56x56xf32>
    %v4440 = stablehlo.subtract %v4433, %v4439 : tensor<256x256x56x56xf32>
    %v4441 = stablehlo.multiply %v4440, %v4440 : tensor<256x256x56x56xf32>
    %v4442 = stablehlo.reduce(%v4441 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4443 = stablehlo.broadcast_in_dim %v4442, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4444 = stablehlo.divide %v4443, %v4435 : tensor<256x256x56x56xf32>
    %v4445 = stablehlo.add %v4444, %v4436 : tensor<256x256x56x56xf32>
    %v4446 = stablehlo.rsqrt %v4445 : tensor<256x256x56x56xf32>
    %v4447 = stablehlo.multiply %v4440, %v4446 : tensor<256x256x56x56xf32>
    %v4448 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4449 = stablehlo.reshape %v4432 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4450 = stablehlo.multiply %v4448, %v4449 : tensor<256x256x56x56xf32>
    %v4451 = stablehlo.reduce(%v4450 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4452 = stablehlo.broadcast_in_dim %v4451, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4453 = stablehlo.multiply %v4447, %v4450 : tensor<256x256x56x56xf32>
    %v4454 = stablehlo.reduce(%v4453 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4455 = stablehlo.broadcast_in_dim %v4454, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4456 = stablehlo.multiply %v4450, %v4435 : tensor<256x256x56x56xf32>
    %v4457 = stablehlo.subtract %v4456, %v4452 : tensor<256x256x56x56xf32>
    %v4458 = stablehlo.multiply %v4447, %v4455 : tensor<256x256x56x56xf32>
    %v4459 = stablehlo.subtract %v4457, %v4458 : tensor<256x256x56x56xf32>
    %v4460 = stablehlo.divide %v4446, %v4435 : tensor<256x256x56x56xf32>
    %v4461 = stablehlo.multiply %v4460, %v4459 : tensor<256x256x56x56xf32>
    %v4462 = stablehlo.reshape %v4461 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4463 = stablehlo.reshape %v4462 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4464 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4465 = stablehlo.transpose %v4464, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4466 = stablehlo.convolution(%v4463, %v4465)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4467 = stablehlo.reshape %v4466 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4468 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4469 = stablehlo.compare GT, %v189, %v4468 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4470 = stablehlo.select %v4469, %v4467, %v4468 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4471 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4473 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4474 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4475 = stablehlo.reduce(%v4471 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.broadcast_in_dim %v4475, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4477 = stablehlo.divide %v4476, %v4473 : tensor<256x64x56x56xf32>
    %v4478 = stablehlo.subtract %v4471, %v4477 : tensor<256x64x56x56xf32>
    %v4479 = stablehlo.multiply %v4478, %v4478 : tensor<256x64x56x56xf32>
    %v4480 = stablehlo.reduce(%v4479 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4481 = stablehlo.broadcast_in_dim %v4480, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4482 = stablehlo.divide %v4481, %v4473 : tensor<256x64x56x56xf32>
    %v4483 = stablehlo.add %v4482, %v4474 : tensor<256x64x56x56xf32>
    %v4484 = stablehlo.rsqrt %v4483 : tensor<256x64x56x56xf32>
    %v4485 = stablehlo.multiply %v4478, %v4484 : tensor<256x64x56x56xf32>
    %v4486 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4487 = stablehlo.reshape %v4470 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4488 = stablehlo.multiply %v4486, %v4487 : tensor<256x64x56x56xf32>
    %v4489 = stablehlo.reduce(%v4488 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4490 = stablehlo.broadcast_in_dim %v4489, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4491 = stablehlo.multiply %v4485, %v4488 : tensor<256x64x56x56xf32>
    %v4492 = stablehlo.reduce(%v4491 init: %v4472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4493 = stablehlo.broadcast_in_dim %v4492, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4494 = stablehlo.multiply %v4488, %v4473 : tensor<256x64x56x56xf32>
    %v4495 = stablehlo.subtract %v4494, %v4490 : tensor<256x64x56x56xf32>
    %v4496 = stablehlo.multiply %v4485, %v4493 : tensor<256x64x56x56xf32>
    %v4497 = stablehlo.subtract %v4495, %v4496 : tensor<256x64x56x56xf32>
    %v4498 = stablehlo.divide %v4484, %v4473 : tensor<256x64x56x56xf32>
    %v4499 = stablehlo.multiply %v4498, %v4497 : tensor<256x64x56x56xf32>
    %v4500 = stablehlo.reshape %v4499 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4501 = stablehlo.reshape %v4500 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4502 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4503 = stablehlo.transpose %v4502, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4504 = stablehlo.convolution(%v4501, %v4503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4505 = stablehlo.reshape %v4504 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4506 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4507 = stablehlo.compare GT, %v162, %v4506 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4508 = stablehlo.select %v4507, %v4505, %v4506 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4509 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4510 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4511 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4512 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4513 = stablehlo.reduce(%v4509 init: %v4510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4514 = stablehlo.broadcast_in_dim %v4513, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4515 = stablehlo.divide %v4514, %v4511 : tensor<256x64x56x56xf32>
    %v4516 = stablehlo.subtract %v4509, %v4515 : tensor<256x64x56x56xf32>
    %v4517 = stablehlo.multiply %v4516, %v4516 : tensor<256x64x56x56xf32>
    %v4518 = stablehlo.reduce(%v4517 init: %v4510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4519 = stablehlo.broadcast_in_dim %v4518, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4520 = stablehlo.divide %v4519, %v4511 : tensor<256x64x56x56xf32>
    %v4521 = stablehlo.add %v4520, %v4512 : tensor<256x64x56x56xf32>
    %v4522 = stablehlo.rsqrt %v4521 : tensor<256x64x56x56xf32>
    %v4523 = stablehlo.multiply %v4516, %v4522 : tensor<256x64x56x56xf32>
    %v4524 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4525 = stablehlo.reshape %v4508 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4526 = stablehlo.multiply %v4524, %v4525 : tensor<256x64x56x56xf32>
    %v4527 = stablehlo.reduce(%v4526 init: %v4510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4528 = stablehlo.broadcast_in_dim %v4527, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4529 = stablehlo.multiply %v4523, %v4526 : tensor<256x64x56x56xf32>
    %v4530 = stablehlo.reduce(%v4529 init: %v4510) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4531 = stablehlo.broadcast_in_dim %v4530, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4532 = stablehlo.multiply %v4526, %v4511 : tensor<256x64x56x56xf32>
    %v4533 = stablehlo.subtract %v4532, %v4528 : tensor<256x64x56x56xf32>
    %v4534 = stablehlo.multiply %v4523, %v4531 : tensor<256x64x56x56xf32>
    %v4535 = stablehlo.subtract %v4533, %v4534 : tensor<256x64x56x56xf32>
    %v4536 = stablehlo.divide %v4522, %v4511 : tensor<256x64x56x56xf32>
    %v4537 = stablehlo.multiply %v4536, %v4535 : tensor<256x64x56x56xf32>
    %v4538 = stablehlo.reshape %v4537 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4539 = stablehlo.reshape %v4538 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4540 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4541 = stablehlo.transpose %v4540, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4542 = stablehlo.convolution(%v4539, %v4541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<256x256x56x56xf32>
    %v4543 = stablehlo.reshape %v4542 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4544 = stablehlo.add %v4543, %v4432 : tensor<256x802816xf32>
    %v4545 = stablehlo.reshape %v137 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4546 = stablehlo.reshape %v4538 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4547 = stablehlo.transpose %v4545, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4548 = stablehlo.transpose %v4546, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4549 = stablehlo.convolution(%v4547, %v4548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v4550 = stablehlo.transpose %v4549, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4551 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4553 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4554 = stablehlo.reduce(%v4551 init: %v4552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4555 = stablehlo.broadcast_in_dim %v4554, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4556 = stablehlo.divide %v4555, %v4553 : tensor<256x64x56x56xf32>
    %v4557 = stablehlo.subtract %v4551, %v4556 : tensor<256x64x56x56xf32>
    %v4558 = stablehlo.multiply %v4557, %v4557 : tensor<256x64x56x56xf32>
    %v4559 = stablehlo.reduce(%v4558 init: %v4552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4560 = stablehlo.broadcast_in_dim %v4559, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4561 = stablehlo.divide %v4560, %v4553 : tensor<256x64x56x56xf32>
    %v4562 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4563 = stablehlo.add %v4561, %v4562 : tensor<256x64x56x56xf32>
    %v4564 = stablehlo.rsqrt %v4563 : tensor<256x64x56x56xf32>
    %v4565 = stablehlo.multiply %v4557, %v4564 : tensor<256x64x56x56xf32>
    %v4566 = stablehlo.reshape %v4508 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4567 = stablehlo.multiply %v4566, %v4565 : tensor<256x64x56x56xf32>
    %v4568 = stablehlo.reduce(%v4567 init: %v4552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4569 = stablehlo.reshape %v4508 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4571 = stablehlo.reduce(%v4569 init: %v4570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4572 = stablehlo.reshape %v164 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4573 = stablehlo.reshape %v4500 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4574 = stablehlo.transpose %v4572, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4575 = stablehlo.transpose %v4573, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4576 = stablehlo.convolution(%v4574, %v4575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4577 = stablehlo.transpose %v4576, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4578 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4580 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4581 = stablehlo.reduce(%v4578 init: %v4579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4582 = stablehlo.broadcast_in_dim %v4581, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4583 = stablehlo.divide %v4582, %v4580 : tensor<256x64x56x56xf32>
    %v4584 = stablehlo.subtract %v4578, %v4583 : tensor<256x64x56x56xf32>
    %v4585 = stablehlo.multiply %v4584, %v4584 : tensor<256x64x56x56xf32>
    %v4586 = stablehlo.reduce(%v4585 init: %v4579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4587 = stablehlo.broadcast_in_dim %v4586, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4588 = stablehlo.divide %v4587, %v4580 : tensor<256x64x56x56xf32>
    %v4589 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4590 = stablehlo.add %v4588, %v4589 : tensor<256x64x56x56xf32>
    %v4591 = stablehlo.rsqrt %v4590 : tensor<256x64x56x56xf32>
    %v4592 = stablehlo.multiply %v4584, %v4591 : tensor<256x64x56x56xf32>
    %v4593 = stablehlo.reshape %v4470 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4594 = stablehlo.multiply %v4593, %v4592 : tensor<256x64x56x56xf32>
    %v4595 = stablehlo.reduce(%v4594 init: %v4579) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4596 = stablehlo.reshape %v4470 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4598 = stablehlo.reduce(%v4596 init: %v4597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4599 = stablehlo.reshape %v191 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4600 = stablehlo.reshape %v4462 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4601 = stablehlo.transpose %v4599, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4602 = stablehlo.transpose %v4600, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4603 = stablehlo.convolution(%v4601, %v4602)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4604 = stablehlo.transpose %v4603, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4605 = stablehlo.reshape %v196 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4606 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4607 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4608 = stablehlo.reduce(%v4605 init: %v4606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4609 = stablehlo.broadcast_in_dim %v4608, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4610 = stablehlo.divide %v4609, %v4607 : tensor<256x256x56x56xf32>
    %v4611 = stablehlo.subtract %v4605, %v4610 : tensor<256x256x56x56xf32>
    %v4612 = stablehlo.multiply %v4611, %v4611 : tensor<256x256x56x56xf32>
    %v4613 = stablehlo.reduce(%v4612 init: %v4606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4614 = stablehlo.broadcast_in_dim %v4613, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4615 = stablehlo.divide %v4614, %v4607 : tensor<256x256x56x56xf32>
    %v4616 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4617 = stablehlo.add %v4615, %v4616 : tensor<256x256x56x56xf32>
    %v4618 = stablehlo.rsqrt %v4617 : tensor<256x256x56x56xf32>
    %v4619 = stablehlo.multiply %v4611, %v4618 : tensor<256x256x56x56xf32>
    %v4620 = stablehlo.reshape %v4432 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4621 = stablehlo.multiply %v4620, %v4619 : tensor<256x256x56x56xf32>
    %v4622 = stablehlo.reduce(%v4621 init: %v4606) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4623 = stablehlo.reshape %v4432 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4625 = stablehlo.reduce(%v4623 init: %v4624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4626 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v4627 = stablehlo.compare GT, %v135, %v4626 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v4628 = stablehlo.select %v4627, %v4544, %v4626 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v4629 = stablehlo.reshape %v89 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4631 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4632 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4633 = stablehlo.reduce(%v4629 init: %v4630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4634 = stablehlo.broadcast_in_dim %v4633, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4635 = stablehlo.divide %v4634, %v4631 : tensor<256x256x56x56xf32>
    %v4636 = stablehlo.subtract %v4629, %v4635 : tensor<256x256x56x56xf32>
    %v4637 = stablehlo.multiply %v4636, %v4636 : tensor<256x256x56x56xf32>
    %v4638 = stablehlo.reduce(%v4637 init: %v4630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4639 = stablehlo.broadcast_in_dim %v4638, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4640 = stablehlo.divide %v4639, %v4631 : tensor<256x256x56x56xf32>
    %v4641 = stablehlo.add %v4640, %v4632 : tensor<256x256x56x56xf32>
    %v4642 = stablehlo.rsqrt %v4641 : tensor<256x256x56x56xf32>
    %v4643 = stablehlo.multiply %v4636, %v4642 : tensor<256x256x56x56xf32>
    %v4644 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4645 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4646 = stablehlo.multiply %v4644, %v4645 : tensor<256x256x56x56xf32>
    %v4647 = stablehlo.reduce(%v4646 init: %v4630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4648 = stablehlo.broadcast_in_dim %v4647, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4649 = stablehlo.multiply %v4643, %v4646 : tensor<256x256x56x56xf32>
    %v4650 = stablehlo.reduce(%v4649 init: %v4630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4651 = stablehlo.broadcast_in_dim %v4650, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4652 = stablehlo.multiply %v4646, %v4631 : tensor<256x256x56x56xf32>
    %v4653 = stablehlo.subtract %v4652, %v4648 : tensor<256x256x56x56xf32>
    %v4654 = stablehlo.multiply %v4643, %v4651 : tensor<256x256x56x56xf32>
    %v4655 = stablehlo.subtract %v4653, %v4654 : tensor<256x256x56x56xf32>
    %v4656 = stablehlo.divide %v4642, %v4631 : tensor<256x256x56x56xf32>
    %v4657 = stablehlo.multiply %v4656, %v4655 : tensor<256x256x56x56xf32>
    %v4658 = stablehlo.reshape %v4657 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4659 = stablehlo.reshape %v4658 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4660 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4661 = stablehlo.transpose %v4660, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4662 = stablehlo.convolution(%v4659, %v4661)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4663 = stablehlo.reshape %v4662 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4664 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4665 = stablehlo.compare GT, %v82, %v4664 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4666 = stablehlo.select %v4665, %v4663, %v4664 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4667 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4669 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4670 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4671 = stablehlo.reduce(%v4667 init: %v4668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4672 = stablehlo.broadcast_in_dim %v4671, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4673 = stablehlo.divide %v4672, %v4669 : tensor<256x64x56x56xf32>
    %v4674 = stablehlo.subtract %v4667, %v4673 : tensor<256x64x56x56xf32>
    %v4675 = stablehlo.multiply %v4674, %v4674 : tensor<256x64x56x56xf32>
    %v4676 = stablehlo.reduce(%v4675 init: %v4668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4677 = stablehlo.broadcast_in_dim %v4676, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4678 = stablehlo.divide %v4677, %v4669 : tensor<256x64x56x56xf32>
    %v4679 = stablehlo.add %v4678, %v4670 : tensor<256x64x56x56xf32>
    %v4680 = stablehlo.rsqrt %v4679 : tensor<256x64x56x56xf32>
    %v4681 = stablehlo.multiply %v4674, %v4680 : tensor<256x64x56x56xf32>
    %v4682 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4683 = stablehlo.reshape %v4666 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4684 = stablehlo.multiply %v4682, %v4683 : tensor<256x64x56x56xf32>
    %v4685 = stablehlo.reduce(%v4684 init: %v4668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4686 = stablehlo.broadcast_in_dim %v4685, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4687 = stablehlo.multiply %v4681, %v4684 : tensor<256x64x56x56xf32>
    %v4688 = stablehlo.reduce(%v4687 init: %v4668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4689 = stablehlo.broadcast_in_dim %v4688, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4690 = stablehlo.multiply %v4684, %v4669 : tensor<256x64x56x56xf32>
    %v4691 = stablehlo.subtract %v4690, %v4686 : tensor<256x64x56x56xf32>
    %v4692 = stablehlo.multiply %v4681, %v4689 : tensor<256x64x56x56xf32>
    %v4693 = stablehlo.subtract %v4691, %v4692 : tensor<256x64x56x56xf32>
    %v4694 = stablehlo.divide %v4680, %v4669 : tensor<256x64x56x56xf32>
    %v4695 = stablehlo.multiply %v4694, %v4693 : tensor<256x64x56x56xf32>
    %v4696 = stablehlo.reshape %v4695 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4697 = stablehlo.reshape %v4696 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4698 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4699 = stablehlo.transpose %v4698, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4700 = stablehlo.convolution(%v4697, %v4699)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<256x64x56x56xf32>
    %v4701 = stablehlo.reshape %v4700 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4702 = stablehlo.constant dense<0.0> : tensor<256x200704xf32>
    %v4703 = stablehlo.compare GT, %v55, %v4702 : (tensor<256x200704xf32>, tensor<256x200704xf32>) -> tensor<256x200704xi1>
    %v4704 = stablehlo.select %v4703, %v4701, %v4702 : tensor<256x200704xi1>, tensor<256x200704xf32>
    %v4705 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4707 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4708 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4709 = stablehlo.reduce(%v4705 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4710 = stablehlo.broadcast_in_dim %v4709, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4711 = stablehlo.divide %v4710, %v4707 : tensor<256x64x56x56xf32>
    %v4712 = stablehlo.subtract %v4705, %v4711 : tensor<256x64x56x56xf32>
    %v4713 = stablehlo.multiply %v4712, %v4712 : tensor<256x64x56x56xf32>
    %v4714 = stablehlo.reduce(%v4713 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4715 = stablehlo.broadcast_in_dim %v4714, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4716 = stablehlo.divide %v4715, %v4707 : tensor<256x64x56x56xf32>
    %v4717 = stablehlo.add %v4716, %v4708 : tensor<256x64x56x56xf32>
    %v4718 = stablehlo.rsqrt %v4717 : tensor<256x64x56x56xf32>
    %v4719 = stablehlo.multiply %v4712, %v4718 : tensor<256x64x56x56xf32>
    %v4720 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4721 = stablehlo.reshape %v4704 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4722 = stablehlo.multiply %v4720, %v4721 : tensor<256x64x56x56xf32>
    %v4723 = stablehlo.reduce(%v4722 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4724 = stablehlo.broadcast_in_dim %v4723, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4725 = stablehlo.multiply %v4719, %v4722 : tensor<256x64x56x56xf32>
    %v4726 = stablehlo.reduce(%v4725 init: %v4706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4727 = stablehlo.broadcast_in_dim %v4726, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4728 = stablehlo.multiply %v4722, %v4707 : tensor<256x64x56x56xf32>
    %v4729 = stablehlo.subtract %v4728, %v4724 : tensor<256x64x56x56xf32>
    %v4730 = stablehlo.multiply %v4719, %v4727 : tensor<256x64x56x56xf32>
    %v4731 = stablehlo.subtract %v4729, %v4730 : tensor<256x64x56x56xf32>
    %v4732 = stablehlo.divide %v4718, %v4707 : tensor<256x64x56x56xf32>
    %v4733 = stablehlo.multiply %v4732, %v4731 : tensor<256x64x56x56xf32>
    %v4734 = stablehlo.reshape %v4733 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4735 = stablehlo.reshape %v4734 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4736 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v4737 = stablehlo.transpose %v4736, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v4738 = stablehlo.convolution(%v4735, %v4737)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4739 = stablehlo.reshape %v4738 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4740 = stablehlo.reshape %v114 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4742 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4743 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4744 = stablehlo.reduce(%v4740 init: %v4741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4745 = stablehlo.broadcast_in_dim %v4744, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4746 = stablehlo.divide %v4745, %v4742 : tensor<256x256x56x56xf32>
    %v4747 = stablehlo.subtract %v4740, %v4746 : tensor<256x256x56x56xf32>
    %v4748 = stablehlo.multiply %v4747, %v4747 : tensor<256x256x56x56xf32>
    %v4749 = stablehlo.reduce(%v4748 init: %v4741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4750 = stablehlo.broadcast_in_dim %v4749, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4751 = stablehlo.divide %v4750, %v4742 : tensor<256x256x56x56xf32>
    %v4752 = stablehlo.add %v4751, %v4743 : tensor<256x256x56x56xf32>
    %v4753 = stablehlo.rsqrt %v4752 : tensor<256x256x56x56xf32>
    %v4754 = stablehlo.multiply %v4747, %v4753 : tensor<256x256x56x56xf32>
    %v4755 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4756 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4757 = stablehlo.multiply %v4755, %v4756 : tensor<256x256x56x56xf32>
    %v4758 = stablehlo.reduce(%v4757 init: %v4741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4759 = stablehlo.broadcast_in_dim %v4758, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4760 = stablehlo.multiply %v4754, %v4757 : tensor<256x256x56x56xf32>
    %v4761 = stablehlo.reduce(%v4760 init: %v4741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4762 = stablehlo.broadcast_in_dim %v4761, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4763 = stablehlo.multiply %v4757, %v4742 : tensor<256x256x56x56xf32>
    %v4764 = stablehlo.subtract %v4763, %v4759 : tensor<256x256x56x56xf32>
    %v4765 = stablehlo.multiply %v4754, %v4762 : tensor<256x256x56x56xf32>
    %v4766 = stablehlo.subtract %v4764, %v4765 : tensor<256x256x56x56xf32>
    %v4767 = stablehlo.divide %v4753, %v4742 : tensor<256x256x56x56xf32>
    %v4768 = stablehlo.multiply %v4767, %v4766 : tensor<256x256x56x56xf32>
    %v4769 = stablehlo.reshape %v4768 : (tensor<256x256x56x56xf32>) -> tensor<256x802816xf32>
    %v4770 = stablehlo.reshape %v4769 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4771 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4772 = stablehlo.transpose %v4771, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4773 = stablehlo.convolution(%v4770, %v4772)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<256x64x56x56xf32>
    %v4774 = stablehlo.reshape %v4773 : (tensor<256x64x56x56xf32>) -> tensor<256x200704xf32>
    %v4775 = stablehlo.add %v4739, %v4774 : tensor<256x200704xf32>
    %v4776 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4777 = stablehlo.reshape %v4734 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4778 = stablehlo.transpose %v4776, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4779 = stablehlo.transpose %v4777, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4780 = stablehlo.convolution(%v4778, %v4779)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x1x1xf32>
    %v4781 = stablehlo.transpose %v4780, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v4782 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4784 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4785 = stablehlo.reduce(%v4782 init: %v4783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4786 = stablehlo.broadcast_in_dim %v4785, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4787 = stablehlo.divide %v4786, %v4784 : tensor<256x64x56x56xf32>
    %v4788 = stablehlo.subtract %v4782, %v4787 : tensor<256x64x56x56xf32>
    %v4789 = stablehlo.multiply %v4788, %v4788 : tensor<256x64x56x56xf32>
    %v4790 = stablehlo.reduce(%v4789 init: %v4783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4791 = stablehlo.broadcast_in_dim %v4790, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4792 = stablehlo.divide %v4791, %v4784 : tensor<256x64x56x56xf32>
    %v4793 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4794 = stablehlo.add %v4792, %v4793 : tensor<256x64x56x56xf32>
    %v4795 = stablehlo.rsqrt %v4794 : tensor<256x64x56x56xf32>
    %v4796 = stablehlo.multiply %v4788, %v4795 : tensor<256x64x56x56xf32>
    %v4797 = stablehlo.reshape %v4704 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4798 = stablehlo.multiply %v4797, %v4796 : tensor<256x64x56x56xf32>
    %v4799 = stablehlo.reduce(%v4798 init: %v4783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4800 = stablehlo.reshape %v4704 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4802 = stablehlo.reduce(%v4800 init: %v4801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4803 = stablehlo.reshape %v57 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4804 = stablehlo.reshape %v4696 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4805 = stablehlo.transpose %v4803, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4806 = stablehlo.transpose %v4804, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4807 = stablehlo.convolution(%v4805, %v4806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v4808 = stablehlo.transpose %v4807, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4809 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4811 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4812 = stablehlo.reduce(%v4809 init: %v4810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4813 = stablehlo.broadcast_in_dim %v4812, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4814 = stablehlo.divide %v4813, %v4811 : tensor<256x64x56x56xf32>
    %v4815 = stablehlo.subtract %v4809, %v4814 : tensor<256x64x56x56xf32>
    %v4816 = stablehlo.multiply %v4815, %v4815 : tensor<256x64x56x56xf32>
    %v4817 = stablehlo.reduce(%v4816 init: %v4810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4818 = stablehlo.broadcast_in_dim %v4817, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4819 = stablehlo.divide %v4818, %v4811 : tensor<256x64x56x56xf32>
    %v4820 = stablehlo.constant dense<1.0e-05> : tensor<256x64x56x56xf32>
    %v4821 = stablehlo.add %v4819, %v4820 : tensor<256x64x56x56xf32>
    %v4822 = stablehlo.rsqrt %v4821 : tensor<256x64x56x56xf32>
    %v4823 = stablehlo.multiply %v4815, %v4822 : tensor<256x64x56x56xf32>
    %v4824 = stablehlo.reshape %v4666 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4825 = stablehlo.multiply %v4824, %v4823 : tensor<256x64x56x56xf32>
    %v4826 = stablehlo.reduce(%v4825 init: %v4810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4827 = stablehlo.reshape %v4666 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4829 = stablehlo.reduce(%v4827 init: %v4828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4830 = stablehlo.reshape %v84 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4831 = stablehlo.reshape %v4658 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4832 = stablehlo.transpose %v4830, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4833 = stablehlo.transpose %v4831, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4834 = stablehlo.convolution(%v4832, %v4833)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4835 = stablehlo.transpose %v4834, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4836 = stablehlo.reshape %v89 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4838 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4839 = stablehlo.reduce(%v4836 init: %v4837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4840 = stablehlo.broadcast_in_dim %v4839, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4841 = stablehlo.divide %v4840, %v4838 : tensor<256x256x56x56xf32>
    %v4842 = stablehlo.subtract %v4836, %v4841 : tensor<256x256x56x56xf32>
    %v4843 = stablehlo.multiply %v4842, %v4842 : tensor<256x256x56x56xf32>
    %v4844 = stablehlo.reduce(%v4843 init: %v4837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4845 = stablehlo.broadcast_in_dim %v4844, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4846 = stablehlo.divide %v4845, %v4838 : tensor<256x256x56x56xf32>
    %v4847 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4848 = stablehlo.add %v4846, %v4847 : tensor<256x256x56x56xf32>
    %v4849 = stablehlo.rsqrt %v4848 : tensor<256x256x56x56xf32>
    %v4850 = stablehlo.multiply %v4842, %v4849 : tensor<256x256x56x56xf32>
    %v4851 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4852 = stablehlo.multiply %v4851, %v4850 : tensor<256x256x56x56xf32>
    %v4853 = stablehlo.reduce(%v4852 init: %v4837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4854 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4856 = stablehlo.reduce(%v4854 init: %v4855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4857 = stablehlo.reshape %v30 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4858 = stablehlo.reshape %v4769 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4859 = stablehlo.transpose %v4857, dims = [1, 0, 2, 3] : (tensor<256x64x56x56xf32>) -> tensor<64x256x56x56xf32>
    %v4860 = stablehlo.transpose %v4858, dims = [1, 0, 2, 3] : (tensor<256x256x56x56xf32>) -> tensor<256x256x56x56xf32>
    %v4861 = stablehlo.convolution(%v4859, %v4860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<256x256x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v4862 = stablehlo.transpose %v4861, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4863 = stablehlo.reshape %v114 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4865 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v4866 = stablehlo.reduce(%v4863 init: %v4864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4867 = stablehlo.broadcast_in_dim %v4866, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4868 = stablehlo.divide %v4867, %v4865 : tensor<256x256x56x56xf32>
    %v4869 = stablehlo.subtract %v4863, %v4868 : tensor<256x256x56x56xf32>
    %v4870 = stablehlo.multiply %v4869, %v4869 : tensor<256x256x56x56xf32>
    %v4871 = stablehlo.reduce(%v4870 init: %v4864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4872 = stablehlo.broadcast_in_dim %v4871, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v4873 = stablehlo.divide %v4872, %v4865 : tensor<256x256x56x56xf32>
    %v4874 = stablehlo.constant dense<1.0e-05> : tensor<256x256x56x56xf32>
    %v4875 = stablehlo.add %v4873, %v4874 : tensor<256x256x56x56xf32>
    %v4876 = stablehlo.rsqrt %v4875 : tensor<256x256x56x56xf32>
    %v4877 = stablehlo.multiply %v4869, %v4876 : tensor<256x256x56x56xf32>
    %v4878 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4879 = stablehlo.multiply %v4878, %v4877 : tensor<256x256x56x56xf32>
    %v4880 = stablehlo.reduce(%v4879 init: %v4864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4881 = stablehlo.reshape %v4628 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v4882 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4883 = stablehlo.reduce(%v4881 init: %v4882) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4884 = stablehlo.reshape %v26 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4885 = stablehlo.reshape %v4775 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4887 = "stablehlo.select_and_scatter"(%v4884, %v4885, %v4886) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<256x64x112x112xf32>, tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<256x64x112x112xf32>
    %v4888 = stablehlo.reshape %v4887 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v4889 = stablehlo.constant dense<0.0> : tensor<256x802816xf32>
    %v4890 = stablehlo.compare GT, %v24, %v4889 : (tensor<256x802816xf32>, tensor<256x802816xf32>) -> tensor<256x802816xi1>
    %v4891 = stablehlo.select %v4890, %v4888, %v4889 : tensor<256x802816xi1>, tensor<256x802816xf32>
    %v4892 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4894 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v4895 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v4896 = stablehlo.reduce(%v4892 init: %v4893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4897 = stablehlo.broadcast_in_dim %v4896, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4898 = stablehlo.divide %v4897, %v4894 : tensor<256x64x112x112xf32>
    %v4899 = stablehlo.subtract %v4892, %v4898 : tensor<256x64x112x112xf32>
    %v4900 = stablehlo.multiply %v4899, %v4899 : tensor<256x64x112x112xf32>
    %v4901 = stablehlo.reduce(%v4900 init: %v4893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4902 = stablehlo.broadcast_in_dim %v4901, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4903 = stablehlo.divide %v4902, %v4894 : tensor<256x64x112x112xf32>
    %v4904 = stablehlo.add %v4903, %v4895 : tensor<256x64x112x112xf32>
    %v4905 = stablehlo.rsqrt %v4904 : tensor<256x64x112x112xf32>
    %v4906 = stablehlo.multiply %v4899, %v4905 : tensor<256x64x112x112xf32>
    %v4907 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4908 = stablehlo.reshape %v4891 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4909 = stablehlo.multiply %v4907, %v4908 : tensor<256x64x112x112xf32>
    %v4910 = stablehlo.reduce(%v4909 init: %v4893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4911 = stablehlo.broadcast_in_dim %v4910, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4912 = stablehlo.multiply %v4906, %v4909 : tensor<256x64x112x112xf32>
    %v4913 = stablehlo.reduce(%v4912 init: %v4893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4914 = stablehlo.broadcast_in_dim %v4913, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4915 = stablehlo.multiply %v4909, %v4894 : tensor<256x64x112x112xf32>
    %v4916 = stablehlo.subtract %v4915, %v4911 : tensor<256x64x112x112xf32>
    %v4917 = stablehlo.multiply %v4906, %v4914 : tensor<256x64x112x112xf32>
    %v4918 = stablehlo.subtract %v4916, %v4917 : tensor<256x64x112x112xf32>
    %v4919 = stablehlo.divide %v4905, %v4894 : tensor<256x64x112x112xf32>
    %v4920 = stablehlo.multiply %v4919, %v4918 : tensor<256x64x112x112xf32>
    %v4921 = stablehlo.reshape %v4920 : (tensor<256x64x112x112xf32>) -> tensor<256x802816xf32>
    %v4922 = stablehlo.reshape %x : (tensor<256x150528xf32>) -> tensor<256x3x224x224xf32>
    %v4923 = stablehlo.reshape %v4921 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4924 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4925 = stablehlo.pad %v4923, %v4924, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<256x64x224x224xf32>
    %v4926 = stablehlo.transpose %v4922, dims = [1, 0, 2, 3] : (tensor<256x3x224x224xf32>) -> tensor<3x256x224x224xf32>
    %v4927 = stablehlo.transpose %v4925, dims = [1, 0, 2, 3] : (tensor<256x64x224x224xf32>) -> tensor<64x256x224x224xf32>
    %v4928 = stablehlo.convolution(%v4926, %v4927)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x256x224x224xf32>, tensor<64x256x224x224xf32>) -> tensor<3x64x7x7xf32>
    %v4929 = stablehlo.transpose %v4928, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v4930 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4932 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v4933 = stablehlo.reduce(%v4930 init: %v4931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4934 = stablehlo.broadcast_in_dim %v4933, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4935 = stablehlo.divide %v4934, %v4932 : tensor<256x64x112x112xf32>
    %v4936 = stablehlo.subtract %v4930, %v4935 : tensor<256x64x112x112xf32>
    %v4937 = stablehlo.multiply %v4936, %v4936 : tensor<256x64x112x112xf32>
    %v4938 = stablehlo.reduce(%v4937 init: %v4931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4939 = stablehlo.broadcast_in_dim %v4938, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4940 = stablehlo.divide %v4939, %v4932 : tensor<256x64x112x112xf32>
    %v4941 = stablehlo.constant dense<1.0e-05> : tensor<256x64x112x112xf32>
    %v4942 = stablehlo.add %v4940, %v4941 : tensor<256x64x112x112xf32>
    %v4943 = stablehlo.rsqrt %v4942 : tensor<256x64x112x112xf32>
    %v4944 = stablehlo.multiply %v4936, %v4943 : tensor<256x64x112x112xf32>
    %v4945 = stablehlo.reshape %v4891 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4946 = stablehlo.multiply %v4945, %v4944 : tensor<256x64x112x112xf32>
    %v4947 = stablehlo.reduce(%v4946 init: %v4931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4948 = stablehlo.reshape %v4891 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4949 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4950 = stablehlo.reduce(%v4948 init: %v4949) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4951 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4952 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4953 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v4954 = stablehlo.reduce(%v4951 init: %v4952) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4955 = stablehlo.divide %v4954, %v4953 : tensor<64xf32>
    %v4956 = stablehlo.reshape %v4 : (tensor<256x802816xf32>) -> tensor<256x64x112x112xf32>
    %v4957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4958 = stablehlo.constant dense<3211264.0> : tensor<256x64x112x112xf32>
    %v4959 = stablehlo.reduce(%v4956 init: %v4957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4960 = stablehlo.broadcast_in_dim %v4959, dims = [1] : (tensor<64xf32>) -> tensor<256x64x112x112xf32>
    %v4961 = stablehlo.divide %v4960, %v4958 : tensor<256x64x112x112xf32>
    %v4962 = stablehlo.subtract %v4956, %v4961 : tensor<256x64x112x112xf32>
    %v4963 = stablehlo.multiply %v4962, %v4962 : tensor<256x64x112x112xf32>
    %v4964 = stablehlo.reduce(%v4963 init: %v4957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4965 = stablehlo.constant dense<3211264.0> : tensor<64xf32>
    %v4966 = stablehlo.divide %v4964, %v4965 : tensor<64xf32>
    %v4967 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4969 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v4970 = stablehlo.reduce(%v4967 init: %v4968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4971 = stablehlo.divide %v4970, %v4969 : tensor<64xf32>
    %v4972 = stablehlo.reshape %v35 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4973 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4974 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4975 = stablehlo.reduce(%v4972 init: %v4973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4976 = stablehlo.broadcast_in_dim %v4975, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4977 = stablehlo.divide %v4976, %v4974 : tensor<256x64x56x56xf32>
    %v4978 = stablehlo.subtract %v4972, %v4977 : tensor<256x64x56x56xf32>
    %v4979 = stablehlo.multiply %v4978, %v4978 : tensor<256x64x56x56xf32>
    %v4980 = stablehlo.reduce(%v4979 init: %v4973) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4981 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v4982 = stablehlo.divide %v4980, %v4981 : tensor<64xf32>
    %v4983 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4985 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v4986 = stablehlo.reduce(%v4983 init: %v4984) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4987 = stablehlo.divide %v4986, %v4985 : tensor<64xf32>
    %v4988 = stablehlo.reshape %v62 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v4989 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4990 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v4991 = stablehlo.reduce(%v4988 init: %v4989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4992 = stablehlo.broadcast_in_dim %v4991, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v4993 = stablehlo.divide %v4992, %v4990 : tensor<256x64x56x56xf32>
    %v4994 = stablehlo.subtract %v4988, %v4993 : tensor<256x64x56x56xf32>
    %v4995 = stablehlo.multiply %v4994, %v4994 : tensor<256x64x56x56xf32>
    %v4996 = stablehlo.reduce(%v4995 init: %v4989) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4997 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v4998 = stablehlo.divide %v4996, %v4997 : tensor<64xf32>
    %v4999 = stablehlo.reshape %v89 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5000 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5001 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5002 = stablehlo.reduce(%v4999 init: %v5000) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5003 = stablehlo.divide %v5002, %v5001 : tensor<256xf32>
    %v5004 = stablehlo.reshape %v89 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5006 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5007 = stablehlo.reduce(%v5004 init: %v5005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5008 = stablehlo.broadcast_in_dim %v5007, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5009 = stablehlo.divide %v5008, %v5006 : tensor<256x256x56x56xf32>
    %v5010 = stablehlo.subtract %v5004, %v5009 : tensor<256x256x56x56xf32>
    %v5011 = stablehlo.multiply %v5010, %v5010 : tensor<256x256x56x56xf32>
    %v5012 = stablehlo.reduce(%v5011 init: %v5005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5013 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5014 = stablehlo.divide %v5012, %v5013 : tensor<256xf32>
    %v5015 = stablehlo.reshape %v114 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5017 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5018 = stablehlo.reduce(%v5015 init: %v5016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5019 = stablehlo.divide %v5018, %v5017 : tensor<256xf32>
    %v5020 = stablehlo.reshape %v114 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5022 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5023 = stablehlo.reduce(%v5020 init: %v5021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5024 = stablehlo.broadcast_in_dim %v5023, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5025 = stablehlo.divide %v5024, %v5022 : tensor<256x256x56x56xf32>
    %v5026 = stablehlo.subtract %v5020, %v5025 : tensor<256x256x56x56xf32>
    %v5027 = stablehlo.multiply %v5026, %v5026 : tensor<256x256x56x56xf32>
    %v5028 = stablehlo.reduce(%v5027 init: %v5021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5029 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5030 = stablehlo.divide %v5028, %v5029 : tensor<256xf32>
    %v5031 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5032 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5033 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5034 = stablehlo.reduce(%v5031 init: %v5032) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5035 = stablehlo.divide %v5034, %v5033 : tensor<64xf32>
    %v5036 = stablehlo.reshape %v142 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5038 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5039 = stablehlo.reduce(%v5036 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5040 = stablehlo.broadcast_in_dim %v5039, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5041 = stablehlo.divide %v5040, %v5038 : tensor<256x64x56x56xf32>
    %v5042 = stablehlo.subtract %v5036, %v5041 : tensor<256x64x56x56xf32>
    %v5043 = stablehlo.multiply %v5042, %v5042 : tensor<256x64x56x56xf32>
    %v5044 = stablehlo.reduce(%v5043 init: %v5037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5045 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5046 = stablehlo.divide %v5044, %v5045 : tensor<64xf32>
    %v5047 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5049 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5050 = stablehlo.reduce(%v5047 init: %v5048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5051 = stablehlo.divide %v5050, %v5049 : tensor<64xf32>
    %v5052 = stablehlo.reshape %v169 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5054 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5055 = stablehlo.reduce(%v5052 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5056 = stablehlo.broadcast_in_dim %v5055, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5057 = stablehlo.divide %v5056, %v5054 : tensor<256x64x56x56xf32>
    %v5058 = stablehlo.subtract %v5052, %v5057 : tensor<256x64x56x56xf32>
    %v5059 = stablehlo.multiply %v5058, %v5058 : tensor<256x64x56x56xf32>
    %v5060 = stablehlo.reduce(%v5059 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5061 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5062 = stablehlo.divide %v5060, %v5061 : tensor<64xf32>
    %v5063 = stablehlo.reshape %v196 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5065 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5066 = stablehlo.reduce(%v5063 init: %v5064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.divide %v5066, %v5065 : tensor<256xf32>
    %v5068 = stablehlo.reshape %v196 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5069 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5070 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5071 = stablehlo.reduce(%v5068 init: %v5069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5072 = stablehlo.broadcast_in_dim %v5071, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5073 = stablehlo.divide %v5072, %v5070 : tensor<256x256x56x56xf32>
    %v5074 = stablehlo.subtract %v5068, %v5073 : tensor<256x256x56x56xf32>
    %v5075 = stablehlo.multiply %v5074, %v5074 : tensor<256x256x56x56xf32>
    %v5076 = stablehlo.reduce(%v5075 init: %v5069) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5077 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5078 = stablehlo.divide %v5076, %v5077 : tensor<256xf32>
    %v5079 = stablehlo.reshape %v224 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5081 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5082 = stablehlo.reduce(%v5079 init: %v5080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5083 = stablehlo.divide %v5082, %v5081 : tensor<64xf32>
    %v5084 = stablehlo.reshape %v224 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5086 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5087 = stablehlo.reduce(%v5084 init: %v5085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5088 = stablehlo.broadcast_in_dim %v5087, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5089 = stablehlo.divide %v5088, %v5086 : tensor<256x64x56x56xf32>
    %v5090 = stablehlo.subtract %v5084, %v5089 : tensor<256x64x56x56xf32>
    %v5091 = stablehlo.multiply %v5090, %v5090 : tensor<256x64x56x56xf32>
    %v5092 = stablehlo.reduce(%v5091 init: %v5085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5093 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5094 = stablehlo.divide %v5092, %v5093 : tensor<64xf32>
    %v5095 = stablehlo.reshape %v251 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5096 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5097 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5098 = stablehlo.reduce(%v5095 init: %v5096) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5099 = stablehlo.divide %v5098, %v5097 : tensor<64xf32>
    %v5100 = stablehlo.reshape %v251 : (tensor<256x200704xf32>) -> tensor<256x64x56x56xf32>
    %v5101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5102 = stablehlo.constant dense<802816.0> : tensor<256x64x56x56xf32>
    %v5103 = stablehlo.reduce(%v5100 init: %v5101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5104 = stablehlo.broadcast_in_dim %v5103, dims = [1] : (tensor<64xf32>) -> tensor<256x64x56x56xf32>
    %v5105 = stablehlo.divide %v5104, %v5102 : tensor<256x64x56x56xf32>
    %v5106 = stablehlo.subtract %v5100, %v5105 : tensor<256x64x56x56xf32>
    %v5107 = stablehlo.multiply %v5106, %v5106 : tensor<256x64x56x56xf32>
    %v5108 = stablehlo.reduce(%v5107 init: %v5101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5109 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5110 = stablehlo.divide %v5108, %v5109 : tensor<64xf32>
    %v5111 = stablehlo.reshape %v278 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5113 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5114 = stablehlo.reduce(%v5111 init: %v5112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5115 = stablehlo.divide %v5114, %v5113 : tensor<256xf32>
    %v5116 = stablehlo.reshape %v278 : (tensor<256x802816xf32>) -> tensor<256x256x56x56xf32>
    %v5117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5118 = stablehlo.constant dense<802816.0> : tensor<256x256x56x56xf32>
    %v5119 = stablehlo.reduce(%v5116 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5120 = stablehlo.broadcast_in_dim %v5119, dims = [1] : (tensor<256xf32>) -> tensor<256x256x56x56xf32>
    %v5121 = stablehlo.divide %v5120, %v5118 : tensor<256x256x56x56xf32>
    %v5122 = stablehlo.subtract %v5116, %v5121 : tensor<256x256x56x56xf32>
    %v5123 = stablehlo.multiply %v5122, %v5122 : tensor<256x256x56x56xf32>
    %v5124 = stablehlo.reduce(%v5123 init: %v5117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5125 = stablehlo.constant dense<802816.0> : tensor<256xf32>
    %v5126 = stablehlo.divide %v5124, %v5125 : tensor<256xf32>
    %v5127 = stablehlo.reshape %v306 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5129 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5130 = stablehlo.reduce(%v5127 init: %v5128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5131 = stablehlo.divide %v5130, %v5129 : tensor<128xf32>
    %v5132 = stablehlo.reshape %v306 : (tensor<256x401408xf32>) -> tensor<256x128x56x56xf32>
    %v5133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5134 = stablehlo.constant dense<802816.0> : tensor<256x128x56x56xf32>
    %v5135 = stablehlo.reduce(%v5132 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5136 = stablehlo.broadcast_in_dim %v5135, dims = [1] : (tensor<128xf32>) -> tensor<256x128x56x56xf32>
    %v5137 = stablehlo.divide %v5136, %v5134 : tensor<256x128x56x56xf32>
    %v5138 = stablehlo.subtract %v5132, %v5137 : tensor<256x128x56x56xf32>
    %v5139 = stablehlo.multiply %v5138, %v5138 : tensor<256x128x56x56xf32>
    %v5140 = stablehlo.reduce(%v5139 init: %v5133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5141 = stablehlo.constant dense<802816.0> : tensor<128xf32>
    %v5142 = stablehlo.divide %v5140, %v5141 : tensor<128xf32>
    %v5143 = stablehlo.reshape %v333 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5145 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5146 = stablehlo.reduce(%v5143 init: %v5144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5147 = stablehlo.divide %v5146, %v5145 : tensor<128xf32>
    %v5148 = stablehlo.reshape %v333 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5150 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5151 = stablehlo.reduce(%v5148 init: %v5149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5152 = stablehlo.broadcast_in_dim %v5151, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5153 = stablehlo.divide %v5152, %v5150 : tensor<256x128x28x28xf32>
    %v5154 = stablehlo.subtract %v5148, %v5153 : tensor<256x128x28x28xf32>
    %v5155 = stablehlo.multiply %v5154, %v5154 : tensor<256x128x28x28xf32>
    %v5156 = stablehlo.reduce(%v5155 init: %v5149) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5157 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5158 = stablehlo.divide %v5156, %v5157 : tensor<128xf32>
    %v5159 = stablehlo.reshape %v360 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5161 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5162 = stablehlo.reduce(%v5159 init: %v5160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5163 = stablehlo.divide %v5162, %v5161 : tensor<512xf32>
    %v5164 = stablehlo.reshape %v360 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5166 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5167 = stablehlo.reduce(%v5164 init: %v5165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5168 = stablehlo.broadcast_in_dim %v5167, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5169 = stablehlo.divide %v5168, %v5166 : tensor<256x512x28x28xf32>
    %v5170 = stablehlo.subtract %v5164, %v5169 : tensor<256x512x28x28xf32>
    %v5171 = stablehlo.multiply %v5170, %v5170 : tensor<256x512x28x28xf32>
    %v5172 = stablehlo.reduce(%v5171 init: %v5165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5173 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5174 = stablehlo.divide %v5172, %v5173 : tensor<512xf32>
    %v5175 = stablehlo.reshape %v385 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5176 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5177 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5178 = stablehlo.reduce(%v5175 init: %v5176) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5179 = stablehlo.divide %v5178, %v5177 : tensor<512xf32>
    %v5180 = stablehlo.reshape %v385 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5182 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5183 = stablehlo.reduce(%v5180 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5184 = stablehlo.broadcast_in_dim %v5183, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5185 = stablehlo.divide %v5184, %v5182 : tensor<256x512x28x28xf32>
    %v5186 = stablehlo.subtract %v5180, %v5185 : tensor<256x512x28x28xf32>
    %v5187 = stablehlo.multiply %v5186, %v5186 : tensor<256x512x28x28xf32>
    %v5188 = stablehlo.reduce(%v5187 init: %v5181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5189 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5190 = stablehlo.divide %v5188, %v5189 : tensor<512xf32>
    %v5191 = stablehlo.reshape %v413 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5193 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5194 = stablehlo.reduce(%v5191 init: %v5192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5195 = stablehlo.divide %v5194, %v5193 : tensor<128xf32>
    %v5196 = stablehlo.reshape %v413 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5198 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5199 = stablehlo.reduce(%v5196 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5200 = stablehlo.broadcast_in_dim %v5199, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5201 = stablehlo.divide %v5200, %v5198 : tensor<256x128x28x28xf32>
    %v5202 = stablehlo.subtract %v5196, %v5201 : tensor<256x128x28x28xf32>
    %v5203 = stablehlo.multiply %v5202, %v5202 : tensor<256x128x28x28xf32>
    %v5204 = stablehlo.reduce(%v5203 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5205 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5206 = stablehlo.divide %v5204, %v5205 : tensor<128xf32>
    %v5207 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5209 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5210 = stablehlo.reduce(%v5207 init: %v5208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5211 = stablehlo.divide %v5210, %v5209 : tensor<128xf32>
    %v5212 = stablehlo.reshape %v440 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5214 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5215 = stablehlo.reduce(%v5212 init: %v5213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5216 = stablehlo.broadcast_in_dim %v5215, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5217 = stablehlo.divide %v5216, %v5214 : tensor<256x128x28x28xf32>
    %v5218 = stablehlo.subtract %v5212, %v5217 : tensor<256x128x28x28xf32>
    %v5219 = stablehlo.multiply %v5218, %v5218 : tensor<256x128x28x28xf32>
    %v5220 = stablehlo.reduce(%v5219 init: %v5213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5221 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5222 = stablehlo.divide %v5220, %v5221 : tensor<128xf32>
    %v5223 = stablehlo.reshape %v467 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5225 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5226 = stablehlo.reduce(%v5223 init: %v5224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5227 = stablehlo.divide %v5226, %v5225 : tensor<512xf32>
    %v5228 = stablehlo.reshape %v467 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5230 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5231 = stablehlo.reduce(%v5228 init: %v5229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5232 = stablehlo.broadcast_in_dim %v5231, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5233 = stablehlo.divide %v5232, %v5230 : tensor<256x512x28x28xf32>
    %v5234 = stablehlo.subtract %v5228, %v5233 : tensor<256x512x28x28xf32>
    %v5235 = stablehlo.multiply %v5234, %v5234 : tensor<256x512x28x28xf32>
    %v5236 = stablehlo.reduce(%v5235 init: %v5229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5237 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5238 = stablehlo.divide %v5236, %v5237 : tensor<512xf32>
    %v5239 = stablehlo.reshape %v495 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5241 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5242 = stablehlo.reduce(%v5239 init: %v5240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5243 = stablehlo.divide %v5242, %v5241 : tensor<128xf32>
    %v5244 = stablehlo.reshape %v495 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5246 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5247 = stablehlo.reduce(%v5244 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5248 = stablehlo.broadcast_in_dim %v5247, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5249 = stablehlo.divide %v5248, %v5246 : tensor<256x128x28x28xf32>
    %v5250 = stablehlo.subtract %v5244, %v5249 : tensor<256x128x28x28xf32>
    %v5251 = stablehlo.multiply %v5250, %v5250 : tensor<256x128x28x28xf32>
    %v5252 = stablehlo.reduce(%v5251 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5253 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5254 = stablehlo.divide %v5252, %v5253 : tensor<128xf32>
    %v5255 = stablehlo.reshape %v522 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5257 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5258 = stablehlo.reduce(%v5255 init: %v5256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5259 = stablehlo.divide %v5258, %v5257 : tensor<128xf32>
    %v5260 = stablehlo.reshape %v522 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5262 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5263 = stablehlo.reduce(%v5260 init: %v5261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5264 = stablehlo.broadcast_in_dim %v5263, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5265 = stablehlo.divide %v5264, %v5262 : tensor<256x128x28x28xf32>
    %v5266 = stablehlo.subtract %v5260, %v5265 : tensor<256x128x28x28xf32>
    %v5267 = stablehlo.multiply %v5266, %v5266 : tensor<256x128x28x28xf32>
    %v5268 = stablehlo.reduce(%v5267 init: %v5261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5269 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5270 = stablehlo.divide %v5268, %v5269 : tensor<128xf32>
    %v5271 = stablehlo.reshape %v549 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5273 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5274 = stablehlo.reduce(%v5271 init: %v5272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5275 = stablehlo.divide %v5274, %v5273 : tensor<512xf32>
    %v5276 = stablehlo.reshape %v549 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5278 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5279 = stablehlo.reduce(%v5276 init: %v5277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5280 = stablehlo.broadcast_in_dim %v5279, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5281 = stablehlo.divide %v5280, %v5278 : tensor<256x512x28x28xf32>
    %v5282 = stablehlo.subtract %v5276, %v5281 : tensor<256x512x28x28xf32>
    %v5283 = stablehlo.multiply %v5282, %v5282 : tensor<256x512x28x28xf32>
    %v5284 = stablehlo.reduce(%v5283 init: %v5277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5285 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5286 = stablehlo.divide %v5284, %v5285 : tensor<512xf32>
    %v5287 = stablehlo.reshape %v577 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5289 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5290 = stablehlo.reduce(%v5287 init: %v5288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5291 = stablehlo.divide %v5290, %v5289 : tensor<128xf32>
    %v5292 = stablehlo.reshape %v577 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5294 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5295 = stablehlo.reduce(%v5292 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5296 = stablehlo.broadcast_in_dim %v5295, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5297 = stablehlo.divide %v5296, %v5294 : tensor<256x128x28x28xf32>
    %v5298 = stablehlo.subtract %v5292, %v5297 : tensor<256x128x28x28xf32>
    %v5299 = stablehlo.multiply %v5298, %v5298 : tensor<256x128x28x28xf32>
    %v5300 = stablehlo.reduce(%v5299 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5301 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5302 = stablehlo.divide %v5300, %v5301 : tensor<128xf32>
    %v5303 = stablehlo.reshape %v604 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5305 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5306 = stablehlo.reduce(%v5303 init: %v5304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5307 = stablehlo.divide %v5306, %v5305 : tensor<128xf32>
    %v5308 = stablehlo.reshape %v604 : (tensor<256x100352xf32>) -> tensor<256x128x28x28xf32>
    %v5309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5310 = stablehlo.constant dense<200704.0> : tensor<256x128x28x28xf32>
    %v5311 = stablehlo.reduce(%v5308 init: %v5309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5312 = stablehlo.broadcast_in_dim %v5311, dims = [1] : (tensor<128xf32>) -> tensor<256x128x28x28xf32>
    %v5313 = stablehlo.divide %v5312, %v5310 : tensor<256x128x28x28xf32>
    %v5314 = stablehlo.subtract %v5308, %v5313 : tensor<256x128x28x28xf32>
    %v5315 = stablehlo.multiply %v5314, %v5314 : tensor<256x128x28x28xf32>
    %v5316 = stablehlo.reduce(%v5315 init: %v5309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5317 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5318 = stablehlo.divide %v5316, %v5317 : tensor<128xf32>
    %v5319 = stablehlo.reshape %v631 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5321 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5322 = stablehlo.reduce(%v5319 init: %v5320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5323 = stablehlo.divide %v5322, %v5321 : tensor<512xf32>
    %v5324 = stablehlo.reshape %v631 : (tensor<256x401408xf32>) -> tensor<256x512x28x28xf32>
    %v5325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5326 = stablehlo.constant dense<200704.0> : tensor<256x512x28x28xf32>
    %v5327 = stablehlo.reduce(%v5324 init: %v5325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5328 = stablehlo.broadcast_in_dim %v5327, dims = [1] : (tensor<512xf32>) -> tensor<256x512x28x28xf32>
    %v5329 = stablehlo.divide %v5328, %v5326 : tensor<256x512x28x28xf32>
    %v5330 = stablehlo.subtract %v5324, %v5329 : tensor<256x512x28x28xf32>
    %v5331 = stablehlo.multiply %v5330, %v5330 : tensor<256x512x28x28xf32>
    %v5332 = stablehlo.reduce(%v5331 init: %v5325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5333 = stablehlo.constant dense<200704.0> : tensor<512xf32>
    %v5334 = stablehlo.divide %v5332, %v5333 : tensor<512xf32>
    %v5335 = stablehlo.reshape %v659 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5337 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5338 = stablehlo.reduce(%v5335 init: %v5336) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5339 = stablehlo.divide %v5338, %v5337 : tensor<256xf32>
    %v5340 = stablehlo.reshape %v659 : (tensor<256x200704xf32>) -> tensor<256x256x28x28xf32>
    %v5341 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5342 = stablehlo.constant dense<200704.0> : tensor<256x256x28x28xf32>
    %v5343 = stablehlo.reduce(%v5340 init: %v5341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5344 = stablehlo.broadcast_in_dim %v5343, dims = [1] : (tensor<256xf32>) -> tensor<256x256x28x28xf32>
    %v5345 = stablehlo.divide %v5344, %v5342 : tensor<256x256x28x28xf32>
    %v5346 = stablehlo.subtract %v5340, %v5345 : tensor<256x256x28x28xf32>
    %v5347 = stablehlo.multiply %v5346, %v5346 : tensor<256x256x28x28xf32>
    %v5348 = stablehlo.reduce(%v5347 init: %v5341) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5349 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5350 = stablehlo.divide %v5348, %v5349 : tensor<256xf32>
    %v5351 = stablehlo.reshape %v686 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5353 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5354 = stablehlo.reduce(%v5351 init: %v5352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5355 = stablehlo.divide %v5354, %v5353 : tensor<256xf32>
    %v5356 = stablehlo.reshape %v686 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5358 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5359 = stablehlo.reduce(%v5356 init: %v5357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5360 = stablehlo.broadcast_in_dim %v5359, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5361 = stablehlo.divide %v5360, %v5358 : tensor<256x256x14x14xf32>
    %v5362 = stablehlo.subtract %v5356, %v5361 : tensor<256x256x14x14xf32>
    %v5363 = stablehlo.multiply %v5362, %v5362 : tensor<256x256x14x14xf32>
    %v5364 = stablehlo.reduce(%v5363 init: %v5357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5365 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5366 = stablehlo.divide %v5364, %v5365 : tensor<256xf32>
    %v5367 = stablehlo.reshape %v713 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5369 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5370 = stablehlo.reduce(%v5367 init: %v5368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5371 = stablehlo.divide %v5370, %v5369 : tensor<1024xf32>
    %v5372 = stablehlo.reshape %v713 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5374 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5375 = stablehlo.reduce(%v5372 init: %v5373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5376 = stablehlo.broadcast_in_dim %v5375, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5377 = stablehlo.divide %v5376, %v5374 : tensor<256x1024x14x14xf32>
    %v5378 = stablehlo.subtract %v5372, %v5377 : tensor<256x1024x14x14xf32>
    %v5379 = stablehlo.multiply %v5378, %v5378 : tensor<256x1024x14x14xf32>
    %v5380 = stablehlo.reduce(%v5379 init: %v5373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5381 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5382 = stablehlo.divide %v5380, %v5381 : tensor<1024xf32>
    %v5383 = stablehlo.reshape %v738 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5385 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5386 = stablehlo.reduce(%v5383 init: %v5384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5387 = stablehlo.divide %v5386, %v5385 : tensor<1024xf32>
    %v5388 = stablehlo.reshape %v738 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5390 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5391 = stablehlo.reduce(%v5388 init: %v5389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5392 = stablehlo.broadcast_in_dim %v5391, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5393 = stablehlo.divide %v5392, %v5390 : tensor<256x1024x14x14xf32>
    %v5394 = stablehlo.subtract %v5388, %v5393 : tensor<256x1024x14x14xf32>
    %v5395 = stablehlo.multiply %v5394, %v5394 : tensor<256x1024x14x14xf32>
    %v5396 = stablehlo.reduce(%v5395 init: %v5389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5397 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5398 = stablehlo.divide %v5396, %v5397 : tensor<1024xf32>
    %v5399 = stablehlo.reshape %v766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5401 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5402 = stablehlo.reduce(%v5399 init: %v5400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5403 = stablehlo.divide %v5402, %v5401 : tensor<256xf32>
    %v5404 = stablehlo.reshape %v766 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5406 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5407 = stablehlo.reduce(%v5404 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5408 = stablehlo.broadcast_in_dim %v5407, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5409 = stablehlo.divide %v5408, %v5406 : tensor<256x256x14x14xf32>
    %v5410 = stablehlo.subtract %v5404, %v5409 : tensor<256x256x14x14xf32>
    %v5411 = stablehlo.multiply %v5410, %v5410 : tensor<256x256x14x14xf32>
    %v5412 = stablehlo.reduce(%v5411 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5413 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5414 = stablehlo.divide %v5412, %v5413 : tensor<256xf32>
    %v5415 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5416 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5417 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5418 = stablehlo.reduce(%v5415 init: %v5416) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5419 = stablehlo.divide %v5418, %v5417 : tensor<256xf32>
    %v5420 = stablehlo.reshape %v793 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5422 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5423 = stablehlo.reduce(%v5420 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5424 = stablehlo.broadcast_in_dim %v5423, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5425 = stablehlo.divide %v5424, %v5422 : tensor<256x256x14x14xf32>
    %v5426 = stablehlo.subtract %v5420, %v5425 : tensor<256x256x14x14xf32>
    %v5427 = stablehlo.multiply %v5426, %v5426 : tensor<256x256x14x14xf32>
    %v5428 = stablehlo.reduce(%v5427 init: %v5421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5429 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5430 = stablehlo.divide %v5428, %v5429 : tensor<256xf32>
    %v5431 = stablehlo.reshape %v820 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5433 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5434 = stablehlo.reduce(%v5431 init: %v5432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5435 = stablehlo.divide %v5434, %v5433 : tensor<1024xf32>
    %v5436 = stablehlo.reshape %v820 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5438 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5439 = stablehlo.reduce(%v5436 init: %v5437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5440 = stablehlo.broadcast_in_dim %v5439, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5441 = stablehlo.divide %v5440, %v5438 : tensor<256x1024x14x14xf32>
    %v5442 = stablehlo.subtract %v5436, %v5441 : tensor<256x1024x14x14xf32>
    %v5443 = stablehlo.multiply %v5442, %v5442 : tensor<256x1024x14x14xf32>
    %v5444 = stablehlo.reduce(%v5443 init: %v5437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5445 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5446 = stablehlo.divide %v5444, %v5445 : tensor<1024xf32>
    %v5447 = stablehlo.reshape %v848 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5449 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5450 = stablehlo.reduce(%v5447 init: %v5448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5451 = stablehlo.divide %v5450, %v5449 : tensor<256xf32>
    %v5452 = stablehlo.reshape %v848 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5454 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5455 = stablehlo.reduce(%v5452 init: %v5453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5456 = stablehlo.broadcast_in_dim %v5455, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5457 = stablehlo.divide %v5456, %v5454 : tensor<256x256x14x14xf32>
    %v5458 = stablehlo.subtract %v5452, %v5457 : tensor<256x256x14x14xf32>
    %v5459 = stablehlo.multiply %v5458, %v5458 : tensor<256x256x14x14xf32>
    %v5460 = stablehlo.reduce(%v5459 init: %v5453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5461 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5462 = stablehlo.divide %v5460, %v5461 : tensor<256xf32>
    %v5463 = stablehlo.reshape %v875 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5464 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5465 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5466 = stablehlo.reduce(%v5463 init: %v5464) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5467 = stablehlo.divide %v5466, %v5465 : tensor<256xf32>
    %v5468 = stablehlo.reshape %v875 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5469 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5470 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5471 = stablehlo.reduce(%v5468 init: %v5469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5472 = stablehlo.broadcast_in_dim %v5471, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5473 = stablehlo.divide %v5472, %v5470 : tensor<256x256x14x14xf32>
    %v5474 = stablehlo.subtract %v5468, %v5473 : tensor<256x256x14x14xf32>
    %v5475 = stablehlo.multiply %v5474, %v5474 : tensor<256x256x14x14xf32>
    %v5476 = stablehlo.reduce(%v5475 init: %v5469) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5477 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5478 = stablehlo.divide %v5476, %v5477 : tensor<256xf32>
    %v5479 = stablehlo.reshape %v902 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5481 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5482 = stablehlo.reduce(%v5479 init: %v5480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5483 = stablehlo.divide %v5482, %v5481 : tensor<1024xf32>
    %v5484 = stablehlo.reshape %v902 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5485 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5486 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5487 = stablehlo.reduce(%v5484 init: %v5485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5488 = stablehlo.broadcast_in_dim %v5487, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5489 = stablehlo.divide %v5488, %v5486 : tensor<256x1024x14x14xf32>
    %v5490 = stablehlo.subtract %v5484, %v5489 : tensor<256x1024x14x14xf32>
    %v5491 = stablehlo.multiply %v5490, %v5490 : tensor<256x1024x14x14xf32>
    %v5492 = stablehlo.reduce(%v5491 init: %v5485) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5493 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5494 = stablehlo.divide %v5492, %v5493 : tensor<1024xf32>
    %v5495 = stablehlo.reshape %v930 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5496 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5497 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5498 = stablehlo.reduce(%v5495 init: %v5496) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5499 = stablehlo.divide %v5498, %v5497 : tensor<256xf32>
    %v5500 = stablehlo.reshape %v930 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5502 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5503 = stablehlo.reduce(%v5500 init: %v5501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5504 = stablehlo.broadcast_in_dim %v5503, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5505 = stablehlo.divide %v5504, %v5502 : tensor<256x256x14x14xf32>
    %v5506 = stablehlo.subtract %v5500, %v5505 : tensor<256x256x14x14xf32>
    %v5507 = stablehlo.multiply %v5506, %v5506 : tensor<256x256x14x14xf32>
    %v5508 = stablehlo.reduce(%v5507 init: %v5501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5509 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5510 = stablehlo.divide %v5508, %v5509 : tensor<256xf32>
    %v5511 = stablehlo.reshape %v957 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5512 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5513 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5514 = stablehlo.reduce(%v5511 init: %v5512) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5515 = stablehlo.divide %v5514, %v5513 : tensor<256xf32>
    %v5516 = stablehlo.reshape %v957 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5518 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5519 = stablehlo.reduce(%v5516 init: %v5517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5520 = stablehlo.broadcast_in_dim %v5519, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5521 = stablehlo.divide %v5520, %v5518 : tensor<256x256x14x14xf32>
    %v5522 = stablehlo.subtract %v5516, %v5521 : tensor<256x256x14x14xf32>
    %v5523 = stablehlo.multiply %v5522, %v5522 : tensor<256x256x14x14xf32>
    %v5524 = stablehlo.reduce(%v5523 init: %v5517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5525 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5526 = stablehlo.divide %v5524, %v5525 : tensor<256xf32>
    %v5527 = stablehlo.reshape %v984 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5529 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5530 = stablehlo.reduce(%v5527 init: %v5528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5531 = stablehlo.divide %v5530, %v5529 : tensor<1024xf32>
    %v5532 = stablehlo.reshape %v984 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5534 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5535 = stablehlo.reduce(%v5532 init: %v5533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5536 = stablehlo.broadcast_in_dim %v5535, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5537 = stablehlo.divide %v5536, %v5534 : tensor<256x1024x14x14xf32>
    %v5538 = stablehlo.subtract %v5532, %v5537 : tensor<256x1024x14x14xf32>
    %v5539 = stablehlo.multiply %v5538, %v5538 : tensor<256x1024x14x14xf32>
    %v5540 = stablehlo.reduce(%v5539 init: %v5533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5541 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5542 = stablehlo.divide %v5540, %v5541 : tensor<1024xf32>
    %v5543 = stablehlo.reshape %v1012 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5544 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5545 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5546 = stablehlo.reduce(%v5543 init: %v5544) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5547 = stablehlo.divide %v5546, %v5545 : tensor<256xf32>
    %v5548 = stablehlo.reshape %v1012 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5550 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5551 = stablehlo.reduce(%v5548 init: %v5549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5552 = stablehlo.broadcast_in_dim %v5551, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5553 = stablehlo.divide %v5552, %v5550 : tensor<256x256x14x14xf32>
    %v5554 = stablehlo.subtract %v5548, %v5553 : tensor<256x256x14x14xf32>
    %v5555 = stablehlo.multiply %v5554, %v5554 : tensor<256x256x14x14xf32>
    %v5556 = stablehlo.reduce(%v5555 init: %v5549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5557 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5558 = stablehlo.divide %v5556, %v5557 : tensor<256xf32>
    %v5559 = stablehlo.reshape %v1039 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5561 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5562 = stablehlo.reduce(%v5559 init: %v5560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5563 = stablehlo.divide %v5562, %v5561 : tensor<256xf32>
    %v5564 = stablehlo.reshape %v1039 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5566 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5567 = stablehlo.reduce(%v5564 init: %v5565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5568 = stablehlo.broadcast_in_dim %v5567, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5569 = stablehlo.divide %v5568, %v5566 : tensor<256x256x14x14xf32>
    %v5570 = stablehlo.subtract %v5564, %v5569 : tensor<256x256x14x14xf32>
    %v5571 = stablehlo.multiply %v5570, %v5570 : tensor<256x256x14x14xf32>
    %v5572 = stablehlo.reduce(%v5571 init: %v5565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5573 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5574 = stablehlo.divide %v5572, %v5573 : tensor<256xf32>
    %v5575 = stablehlo.reshape %v1066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5576 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5577 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5578 = stablehlo.reduce(%v5575 init: %v5576) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5579 = stablehlo.divide %v5578, %v5577 : tensor<1024xf32>
    %v5580 = stablehlo.reshape %v1066 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5582 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5583 = stablehlo.reduce(%v5580 init: %v5581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5584 = stablehlo.broadcast_in_dim %v5583, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5585 = stablehlo.divide %v5584, %v5582 : tensor<256x1024x14x14xf32>
    %v5586 = stablehlo.subtract %v5580, %v5585 : tensor<256x1024x14x14xf32>
    %v5587 = stablehlo.multiply %v5586, %v5586 : tensor<256x1024x14x14xf32>
    %v5588 = stablehlo.reduce(%v5587 init: %v5581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5589 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5590 = stablehlo.divide %v5588, %v5589 : tensor<1024xf32>
    %v5591 = stablehlo.reshape %v1094 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5593 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5594 = stablehlo.reduce(%v5591 init: %v5592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5595 = stablehlo.divide %v5594, %v5593 : tensor<256xf32>
    %v5596 = stablehlo.reshape %v1094 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5598 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5599 = stablehlo.reduce(%v5596 init: %v5597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5600 = stablehlo.broadcast_in_dim %v5599, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5601 = stablehlo.divide %v5600, %v5598 : tensor<256x256x14x14xf32>
    %v5602 = stablehlo.subtract %v5596, %v5601 : tensor<256x256x14x14xf32>
    %v5603 = stablehlo.multiply %v5602, %v5602 : tensor<256x256x14x14xf32>
    %v5604 = stablehlo.reduce(%v5603 init: %v5597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5605 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5606 = stablehlo.divide %v5604, %v5605 : tensor<256xf32>
    %v5607 = stablehlo.reshape %v1121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5609 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5610 = stablehlo.reduce(%v5607 init: %v5608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5611 = stablehlo.divide %v5610, %v5609 : tensor<256xf32>
    %v5612 = stablehlo.reshape %v1121 : (tensor<256x50176xf32>) -> tensor<256x256x14x14xf32>
    %v5613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5614 = stablehlo.constant dense<50176.0> : tensor<256x256x14x14xf32>
    %v5615 = stablehlo.reduce(%v5612 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5616 = stablehlo.broadcast_in_dim %v5615, dims = [1] : (tensor<256xf32>) -> tensor<256x256x14x14xf32>
    %v5617 = stablehlo.divide %v5616, %v5614 : tensor<256x256x14x14xf32>
    %v5618 = stablehlo.subtract %v5612, %v5617 : tensor<256x256x14x14xf32>
    %v5619 = stablehlo.multiply %v5618, %v5618 : tensor<256x256x14x14xf32>
    %v5620 = stablehlo.reduce(%v5619 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5621 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5622 = stablehlo.divide %v5620, %v5621 : tensor<256xf32>
    %v5623 = stablehlo.reshape %v1148 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5625 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5626 = stablehlo.reduce(%v5623 init: %v5624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5627 = stablehlo.divide %v5626, %v5625 : tensor<1024xf32>
    %v5628 = stablehlo.reshape %v1148 : (tensor<256x200704xf32>) -> tensor<256x1024x14x14xf32>
    %v5629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5630 = stablehlo.constant dense<50176.0> : tensor<256x1024x14x14xf32>
    %v5631 = stablehlo.reduce(%v5628 init: %v5629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5632 = stablehlo.broadcast_in_dim %v5631, dims = [1] : (tensor<1024xf32>) -> tensor<256x1024x14x14xf32>
    %v5633 = stablehlo.divide %v5632, %v5630 : tensor<256x1024x14x14xf32>
    %v5634 = stablehlo.subtract %v5628, %v5633 : tensor<256x1024x14x14xf32>
    %v5635 = stablehlo.multiply %v5634, %v5634 : tensor<256x1024x14x14xf32>
    %v5636 = stablehlo.reduce(%v5635 init: %v5629) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5637 = stablehlo.constant dense<50176.0> : tensor<1024xf32>
    %v5638 = stablehlo.divide %v5636, %v5637 : tensor<1024xf32>
    %v5639 = stablehlo.reshape %v1176 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v5640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5641 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5642 = stablehlo.reduce(%v5639 init: %v5640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5643 = stablehlo.divide %v5642, %v5641 : tensor<512xf32>
    %v5644 = stablehlo.reshape %v1176 : (tensor<256x100352xf32>) -> tensor<256x512x14x14xf32>
    %v5645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5646 = stablehlo.constant dense<50176.0> : tensor<256x512x14x14xf32>
    %v5647 = stablehlo.reduce(%v5644 init: %v5645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5648 = stablehlo.broadcast_in_dim %v5647, dims = [1] : (tensor<512xf32>) -> tensor<256x512x14x14xf32>
    %v5649 = stablehlo.divide %v5648, %v5646 : tensor<256x512x14x14xf32>
    %v5650 = stablehlo.subtract %v5644, %v5649 : tensor<256x512x14x14xf32>
    %v5651 = stablehlo.multiply %v5650, %v5650 : tensor<256x512x14x14xf32>
    %v5652 = stablehlo.reduce(%v5651 init: %v5645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v5653 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5654 = stablehlo.divide %v5652, %v5653 : tensor<512xf32>
    %v5655 = stablehlo.reshape %v1203 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5656 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5657 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5658 = stablehlo.reduce(%v5655 init: %v5656) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5659 = stablehlo.divide %v5658, %v5657 : tensor<512xf32>
    %v5660 = stablehlo.reshape %v1203 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5662 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v5663 = stablehlo.reduce(%v5660 init: %v5661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5664 = stablehlo.broadcast_in_dim %v5663, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v5665 = stablehlo.divide %v5664, %v5662 : tensor<256x512x7x7xf32>
    %v5666 = stablehlo.subtract %v5660, %v5665 : tensor<256x512x7x7xf32>
    %v5667 = stablehlo.multiply %v5666, %v5666 : tensor<256x512x7x7xf32>
    %v5668 = stablehlo.reduce(%v5667 init: %v5661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5669 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5670 = stablehlo.divide %v5668, %v5669 : tensor<512xf32>
    %v5671 = stablehlo.reshape %v1230 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5673 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5674 = stablehlo.reduce(%v5671 init: %v5672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5675 = stablehlo.divide %v5674, %v5673 : tensor<2048xf32>
    %v5676 = stablehlo.reshape %v1230 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5678 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v5679 = stablehlo.reduce(%v5676 init: %v5677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5680 = stablehlo.broadcast_in_dim %v5679, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v5681 = stablehlo.divide %v5680, %v5678 : tensor<256x2048x7x7xf32>
    %v5682 = stablehlo.subtract %v5676, %v5681 : tensor<256x2048x7x7xf32>
    %v5683 = stablehlo.multiply %v5682, %v5682 : tensor<256x2048x7x7xf32>
    %v5684 = stablehlo.reduce(%v5683 init: %v5677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5685 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5686 = stablehlo.divide %v5684, %v5685 : tensor<2048xf32>
    %v5687 = stablehlo.reshape %v1255 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5689 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5690 = stablehlo.reduce(%v5687 init: %v5688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5691 = stablehlo.divide %v5690, %v5689 : tensor<2048xf32>
    %v5692 = stablehlo.reshape %v1255 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5694 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v5695 = stablehlo.reduce(%v5692 init: %v5693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5696 = stablehlo.broadcast_in_dim %v5695, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v5697 = stablehlo.divide %v5696, %v5694 : tensor<256x2048x7x7xf32>
    %v5698 = stablehlo.subtract %v5692, %v5697 : tensor<256x2048x7x7xf32>
    %v5699 = stablehlo.multiply %v5698, %v5698 : tensor<256x2048x7x7xf32>
    %v5700 = stablehlo.reduce(%v5699 init: %v5693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5701 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5702 = stablehlo.divide %v5700, %v5701 : tensor<2048xf32>
    %v5703 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5705 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5706 = stablehlo.reduce(%v5703 init: %v5704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5707 = stablehlo.divide %v5706, %v5705 : tensor<512xf32>
    %v5708 = stablehlo.reshape %v1283 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5710 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v5711 = stablehlo.reduce(%v5708 init: %v5709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5712 = stablehlo.broadcast_in_dim %v5711, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v5713 = stablehlo.divide %v5712, %v5710 : tensor<256x512x7x7xf32>
    %v5714 = stablehlo.subtract %v5708, %v5713 : tensor<256x512x7x7xf32>
    %v5715 = stablehlo.multiply %v5714, %v5714 : tensor<256x512x7x7xf32>
    %v5716 = stablehlo.reduce(%v5715 init: %v5709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5717 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5718 = stablehlo.divide %v5716, %v5717 : tensor<512xf32>
    %v5719 = stablehlo.reshape %v1310 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5720 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5721 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5722 = stablehlo.reduce(%v5719 init: %v5720) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5723 = stablehlo.divide %v5722, %v5721 : tensor<512xf32>
    %v5724 = stablehlo.reshape %v1310 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5726 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v5727 = stablehlo.reduce(%v5724 init: %v5725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5728 = stablehlo.broadcast_in_dim %v5727, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v5729 = stablehlo.divide %v5728, %v5726 : tensor<256x512x7x7xf32>
    %v5730 = stablehlo.subtract %v5724, %v5729 : tensor<256x512x7x7xf32>
    %v5731 = stablehlo.multiply %v5730, %v5730 : tensor<256x512x7x7xf32>
    %v5732 = stablehlo.reduce(%v5731 init: %v5725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5733 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5734 = stablehlo.divide %v5732, %v5733 : tensor<512xf32>
    %v5735 = stablehlo.reshape %v1337 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5737 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5738 = stablehlo.reduce(%v5735 init: %v5736) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5739 = stablehlo.divide %v5738, %v5737 : tensor<2048xf32>
    %v5740 = stablehlo.reshape %v1337 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5742 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v5743 = stablehlo.reduce(%v5740 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5744 = stablehlo.broadcast_in_dim %v5743, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v5745 = stablehlo.divide %v5744, %v5742 : tensor<256x2048x7x7xf32>
    %v5746 = stablehlo.subtract %v5740, %v5745 : tensor<256x2048x7x7xf32>
    %v5747 = stablehlo.multiply %v5746, %v5746 : tensor<256x2048x7x7xf32>
    %v5748 = stablehlo.reduce(%v5747 init: %v5741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5749 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5750 = stablehlo.divide %v5748, %v5749 : tensor<2048xf32>
    %v5751 = stablehlo.reshape %v1365 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5753 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5754 = stablehlo.reduce(%v5751 init: %v5752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5755 = stablehlo.divide %v5754, %v5753 : tensor<512xf32>
    %v5756 = stablehlo.reshape %v1365 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5758 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v5759 = stablehlo.reduce(%v5756 init: %v5757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5760 = stablehlo.broadcast_in_dim %v5759, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v5761 = stablehlo.divide %v5760, %v5758 : tensor<256x512x7x7xf32>
    %v5762 = stablehlo.subtract %v5756, %v5761 : tensor<256x512x7x7xf32>
    %v5763 = stablehlo.multiply %v5762, %v5762 : tensor<256x512x7x7xf32>
    %v5764 = stablehlo.reduce(%v5763 init: %v5757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5765 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5766 = stablehlo.divide %v5764, %v5765 : tensor<512xf32>
    %v5767 = stablehlo.reshape %v1392 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5769 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5770 = stablehlo.reduce(%v5767 init: %v5768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5771 = stablehlo.divide %v5770, %v5769 : tensor<512xf32>
    %v5772 = stablehlo.reshape %v1392 : (tensor<256x25088xf32>) -> tensor<256x512x7x7xf32>
    %v5773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5774 = stablehlo.constant dense<12544.0> : tensor<256x512x7x7xf32>
    %v5775 = stablehlo.reduce(%v5772 init: %v5773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5776 = stablehlo.broadcast_in_dim %v5775, dims = [1] : (tensor<512xf32>) -> tensor<256x512x7x7xf32>
    %v5777 = stablehlo.divide %v5776, %v5774 : tensor<256x512x7x7xf32>
    %v5778 = stablehlo.subtract %v5772, %v5777 : tensor<256x512x7x7xf32>
    %v5779 = stablehlo.multiply %v5778, %v5778 : tensor<256x512x7x7xf32>
    %v5780 = stablehlo.reduce(%v5779 init: %v5773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v5781 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v5782 = stablehlo.divide %v5780, %v5781 : tensor<512xf32>
    %v5783 = stablehlo.reshape %v1419 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5785 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5786 = stablehlo.reduce(%v5783 init: %v5784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5787 = stablehlo.divide %v5786, %v5785 : tensor<2048xf32>
    %v5788 = stablehlo.reshape %v1419 : (tensor<256x100352xf32>) -> tensor<256x2048x7x7xf32>
    %v5789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5790 = stablehlo.constant dense<12544.0> : tensor<256x2048x7x7xf32>
    %v5791 = stablehlo.reduce(%v5788 init: %v5789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5792 = stablehlo.broadcast_in_dim %v5791, dims = [1] : (tensor<2048xf32>) -> tensor<256x2048x7x7xf32>
    %v5793 = stablehlo.divide %v5792, %v5790 : tensor<256x2048x7x7xf32>
    %v5794 = stablehlo.subtract %v5788, %v5793 : tensor<256x2048x7x7xf32>
    %v5795 = stablehlo.multiply %v5794, %v5794 : tensor<256x2048x7x7xf32>
    %v5796 = stablehlo.reduce(%v5795 init: %v5789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<256x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v5797 = stablehlo.constant dense<12544.0> : tensor<2048xf32>
    %v5798 = stablehlo.divide %v5796, %v5797 : tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %v5799 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v5800 = stablehlo.multiply %v5799, %sW : tensor<64x3x7x7xf32>
    %v5801 = stablehlo.add %v5800, %v4929 : tensor<64x3x7x7xf32>
    %v5802 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v5803 = stablehlo.multiply %v5802, %sWv : tensor<64x3x7x7xf32>
    %v5804 = stablehlo.add %v5803, %v5801 : tensor<64x3x7x7xf32>
    %v5805 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v5806 = stablehlo.multiply %v5805, %v5804 : tensor<64x3x7x7xf32>
    %v5807 = stablehlo.subtract %sW, %v5806 : tensor<64x3x7x7xf32>
    %v5808 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5809 = stablehlo.multiply %v5808, %sg : tensor<64xf32>
    %v5810 = stablehlo.add %v5809, %v4947 : tensor<64xf32>
    %v5811 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5812 = stablehlo.multiply %v5811, %sgv : tensor<64xf32>
    %v5813 = stablehlo.add %v5812, %v5810 : tensor<64xf32>
    %v5814 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5815 = stablehlo.multiply %v5814, %v5813 : tensor<64xf32>
    %v5816 = stablehlo.subtract %sg, %v5815 : tensor<64xf32>
    %v5817 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5818 = stablehlo.multiply %v5817, %sbt : tensor<64xf32>
    %v5819 = stablehlo.add %v5818, %v4950 : tensor<64xf32>
    %v5820 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5821 = stablehlo.multiply %v5820, %sbtv : tensor<64xf32>
    %v5822 = stablehlo.add %v5821, %v5819 : tensor<64xf32>
    %v5823 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5824 = stablehlo.multiply %v5823, %v5822 : tensor<64xf32>
    %v5825 = stablehlo.subtract %sbt, %v5824 : tensor<64xf32>
    %v5826 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v5827 = stablehlo.multiply %v5826, %s1b0W1 : tensor<64x64x1x1xf32>
    %v5828 = stablehlo.add %v5827, %v4781 : tensor<64x64x1x1xf32>
    %v5829 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v5830 = stablehlo.multiply %v5829, %s1b0W1v : tensor<64x64x1x1xf32>
    %v5831 = stablehlo.add %v5830, %v5828 : tensor<64x64x1x1xf32>
    %v5832 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v5833 = stablehlo.multiply %v5832, %v5831 : tensor<64x64x1x1xf32>
    %v5834 = stablehlo.subtract %s1b0W1, %v5833 : tensor<64x64x1x1xf32>
    %v5835 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5836 = stablehlo.multiply %v5835, %s1b0g1 : tensor<64xf32>
    %v5837 = stablehlo.add %v5836, %v4799 : tensor<64xf32>
    %v5838 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5839 = stablehlo.multiply %v5838, %s1b0g1v : tensor<64xf32>
    %v5840 = stablehlo.add %v5839, %v5837 : tensor<64xf32>
    %v5841 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5842 = stablehlo.multiply %v5841, %v5840 : tensor<64xf32>
    %v5843 = stablehlo.subtract %s1b0g1, %v5842 : tensor<64xf32>
    %v5844 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5845 = stablehlo.multiply %v5844, %s1b0bt1 : tensor<64xf32>
    %v5846 = stablehlo.add %v5845, %v4802 : tensor<64xf32>
    %v5847 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5848 = stablehlo.multiply %v5847, %s1b0bt1v : tensor<64xf32>
    %v5849 = stablehlo.add %v5848, %v5846 : tensor<64xf32>
    %v5850 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5851 = stablehlo.multiply %v5850, %v5849 : tensor<64xf32>
    %v5852 = stablehlo.subtract %s1b0bt1, %v5851 : tensor<64xf32>
    %v5853 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5854 = stablehlo.multiply %v5853, %s1b0W2 : tensor<64x64x3x3xf32>
    %v5855 = stablehlo.add %v5854, %v4808 : tensor<64x64x3x3xf32>
    %v5856 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5857 = stablehlo.multiply %v5856, %s1b0W2v : tensor<64x64x3x3xf32>
    %v5858 = stablehlo.add %v5857, %v5855 : tensor<64x64x3x3xf32>
    %v5859 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5860 = stablehlo.multiply %v5859, %v5858 : tensor<64x64x3x3xf32>
    %v5861 = stablehlo.subtract %s1b0W2, %v5860 : tensor<64x64x3x3xf32>
    %v5862 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5863 = stablehlo.multiply %v5862, %s1b0g2 : tensor<64xf32>
    %v5864 = stablehlo.add %v5863, %v4826 : tensor<64xf32>
    %v5865 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5866 = stablehlo.multiply %v5865, %s1b0g2v : tensor<64xf32>
    %v5867 = stablehlo.add %v5866, %v5864 : tensor<64xf32>
    %v5868 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5869 = stablehlo.multiply %v5868, %v5867 : tensor<64xf32>
    %v5870 = stablehlo.subtract %s1b0g2, %v5869 : tensor<64xf32>
    %v5871 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5872 = stablehlo.multiply %v5871, %s1b0bt2 : tensor<64xf32>
    %v5873 = stablehlo.add %v5872, %v4829 : tensor<64xf32>
    %v5874 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5875 = stablehlo.multiply %v5874, %s1b0bt2v : tensor<64xf32>
    %v5876 = stablehlo.add %v5875, %v5873 : tensor<64xf32>
    %v5877 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5878 = stablehlo.multiply %v5877, %v5876 : tensor<64xf32>
    %v5879 = stablehlo.subtract %s1b0bt2, %v5878 : tensor<64xf32>
    %v5880 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5881 = stablehlo.multiply %v5880, %s1b0W3 : tensor<256x64x1x1xf32>
    %v5882 = stablehlo.add %v5881, %v4835 : tensor<256x64x1x1xf32>
    %v5883 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5884 = stablehlo.multiply %v5883, %s1b0W3v : tensor<256x64x1x1xf32>
    %v5885 = stablehlo.add %v5884, %v5882 : tensor<256x64x1x1xf32>
    %v5886 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5887 = stablehlo.multiply %v5886, %v5885 : tensor<256x64x1x1xf32>
    %v5888 = stablehlo.subtract %s1b0W3, %v5887 : tensor<256x64x1x1xf32>
    %v5889 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5890 = stablehlo.multiply %v5889, %s1b0g3 : tensor<256xf32>
    %v5891 = stablehlo.add %v5890, %v4853 : tensor<256xf32>
    %v5892 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5893 = stablehlo.multiply %v5892, %s1b0g3v : tensor<256xf32>
    %v5894 = stablehlo.add %v5893, %v5891 : tensor<256xf32>
    %v5895 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5896 = stablehlo.multiply %v5895, %v5894 : tensor<256xf32>
    %v5897 = stablehlo.subtract %s1b0g3, %v5896 : tensor<256xf32>
    %v5898 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5899 = stablehlo.multiply %v5898, %s1b0bt3 : tensor<256xf32>
    %v5900 = stablehlo.add %v5899, %v4856 : tensor<256xf32>
    %v5901 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5902 = stablehlo.multiply %v5901, %s1b0bt3v : tensor<256xf32>
    %v5903 = stablehlo.add %v5902, %v5900 : tensor<256xf32>
    %v5904 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5905 = stablehlo.multiply %v5904, %v5903 : tensor<256xf32>
    %v5906 = stablehlo.subtract %s1b0bt3, %v5905 : tensor<256xf32>
    %v5907 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5908 = stablehlo.multiply %v5907, %s1b0Wp : tensor<256x64x1x1xf32>
    %v5909 = stablehlo.add %v5908, %v4862 : tensor<256x64x1x1xf32>
    %v5910 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5911 = stablehlo.multiply %v5910, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v5912 = stablehlo.add %v5911, %v5909 : tensor<256x64x1x1xf32>
    %v5913 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5914 = stablehlo.multiply %v5913, %v5912 : tensor<256x64x1x1xf32>
    %v5915 = stablehlo.subtract %s1b0Wp, %v5914 : tensor<256x64x1x1xf32>
    %v5916 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5917 = stablehlo.multiply %v5916, %s1b0gp : tensor<256xf32>
    %v5918 = stablehlo.add %v5917, %v4880 : tensor<256xf32>
    %v5919 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5920 = stablehlo.multiply %v5919, %s1b0gpv : tensor<256xf32>
    %v5921 = stablehlo.add %v5920, %v5918 : tensor<256xf32>
    %v5922 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5923 = stablehlo.multiply %v5922, %v5921 : tensor<256xf32>
    %v5924 = stablehlo.subtract %s1b0gp, %v5923 : tensor<256xf32>
    %v5925 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5926 = stablehlo.multiply %v5925, %s1b0btp : tensor<256xf32>
    %v5927 = stablehlo.add %v5926, %v4883 : tensor<256xf32>
    %v5928 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5929 = stablehlo.multiply %v5928, %s1b0btpv : tensor<256xf32>
    %v5930 = stablehlo.add %v5929, %v5927 : tensor<256xf32>
    %v5931 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5932 = stablehlo.multiply %v5931, %v5930 : tensor<256xf32>
    %v5933 = stablehlo.subtract %s1b0btp, %v5932 : tensor<256xf32>
    %v5934 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v5935 = stablehlo.multiply %v5934, %s1b1W1 : tensor<64x256x1x1xf32>
    %v5936 = stablehlo.add %v5935, %v4550 : tensor<64x256x1x1xf32>
    %v5937 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v5938 = stablehlo.multiply %v5937, %s1b1W1v : tensor<64x256x1x1xf32>
    %v5939 = stablehlo.add %v5938, %v5936 : tensor<64x256x1x1xf32>
    %v5940 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v5941 = stablehlo.multiply %v5940, %v5939 : tensor<64x256x1x1xf32>
    %v5942 = stablehlo.subtract %s1b1W1, %v5941 : tensor<64x256x1x1xf32>
    %v5943 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5944 = stablehlo.multiply %v5943, %s1b1g1 : tensor<64xf32>
    %v5945 = stablehlo.add %v5944, %v4568 : tensor<64xf32>
    %v5946 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5947 = stablehlo.multiply %v5946, %s1b1g1v : tensor<64xf32>
    %v5948 = stablehlo.add %v5947, %v5945 : tensor<64xf32>
    %v5949 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5950 = stablehlo.multiply %v5949, %v5948 : tensor<64xf32>
    %v5951 = stablehlo.subtract %s1b1g1, %v5950 : tensor<64xf32>
    %v5952 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5953 = stablehlo.multiply %v5952, %s1b1bt1 : tensor<64xf32>
    %v5954 = stablehlo.add %v5953, %v4571 : tensor<64xf32>
    %v5955 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5956 = stablehlo.multiply %v5955, %s1b1bt1v : tensor<64xf32>
    %v5957 = stablehlo.add %v5956, %v5954 : tensor<64xf32>
    %v5958 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5959 = stablehlo.multiply %v5958, %v5957 : tensor<64xf32>
    %v5960 = stablehlo.subtract %s1b1bt1, %v5959 : tensor<64xf32>
    %v5961 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5962 = stablehlo.multiply %v5961, %s1b1W2 : tensor<64x64x3x3xf32>
    %v5963 = stablehlo.add %v5962, %v4577 : tensor<64x64x3x3xf32>
    %v5964 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5965 = stablehlo.multiply %v5964, %s1b1W2v : tensor<64x64x3x3xf32>
    %v5966 = stablehlo.add %v5965, %v5963 : tensor<64x64x3x3xf32>
    %v5967 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5968 = stablehlo.multiply %v5967, %v5966 : tensor<64x64x3x3xf32>
    %v5969 = stablehlo.subtract %s1b1W2, %v5968 : tensor<64x64x3x3xf32>
    %v5970 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5971 = stablehlo.multiply %v5970, %s1b1g2 : tensor<64xf32>
    %v5972 = stablehlo.add %v5971, %v4595 : tensor<64xf32>
    %v5973 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5974 = stablehlo.multiply %v5973, %s1b1g2v : tensor<64xf32>
    %v5975 = stablehlo.add %v5974, %v5972 : tensor<64xf32>
    %v5976 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5977 = stablehlo.multiply %v5976, %v5975 : tensor<64xf32>
    %v5978 = stablehlo.subtract %s1b1g2, %v5977 : tensor<64xf32>
    %v5979 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5980 = stablehlo.multiply %v5979, %s1b1bt2 : tensor<64xf32>
    %v5981 = stablehlo.add %v5980, %v4598 : tensor<64xf32>
    %v5982 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5983 = stablehlo.multiply %v5982, %s1b1bt2v : tensor<64xf32>
    %v5984 = stablehlo.add %v5983, %v5981 : tensor<64xf32>
    %v5985 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5986 = stablehlo.multiply %v5985, %v5984 : tensor<64xf32>
    %v5987 = stablehlo.subtract %s1b1bt2, %v5986 : tensor<64xf32>
    %v5988 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5989 = stablehlo.multiply %v5988, %s1b1W3 : tensor<256x64x1x1xf32>
    %v5990 = stablehlo.add %v5989, %v4604 : tensor<256x64x1x1xf32>
    %v5991 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5992 = stablehlo.multiply %v5991, %s1b1W3v : tensor<256x64x1x1xf32>
    %v5993 = stablehlo.add %v5992, %v5990 : tensor<256x64x1x1xf32>
    %v5994 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v5995 = stablehlo.multiply %v5994, %v5993 : tensor<256x64x1x1xf32>
    %v5996 = stablehlo.subtract %s1b1W3, %v5995 : tensor<256x64x1x1xf32>
    %v5997 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5998 = stablehlo.multiply %v5997, %s1b1g3 : tensor<256xf32>
    %v5999 = stablehlo.add %v5998, %v4622 : tensor<256xf32>
    %v6000 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6001 = stablehlo.multiply %v6000, %s1b1g3v : tensor<256xf32>
    %v6002 = stablehlo.add %v6001, %v5999 : tensor<256xf32>
    %v6003 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6004 = stablehlo.multiply %v6003, %v6002 : tensor<256xf32>
    %v6005 = stablehlo.subtract %s1b1g3, %v6004 : tensor<256xf32>
    %v6006 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6007 = stablehlo.multiply %v6006, %s1b1bt3 : tensor<256xf32>
    %v6008 = stablehlo.add %v6007, %v4625 : tensor<256xf32>
    %v6009 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6010 = stablehlo.multiply %v6009, %s1b1bt3v : tensor<256xf32>
    %v6011 = stablehlo.add %v6010, %v6008 : tensor<256xf32>
    %v6012 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6013 = stablehlo.multiply %v6012, %v6011 : tensor<256xf32>
    %v6014 = stablehlo.subtract %s1b1bt3, %v6013 : tensor<256xf32>
    %v6015 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6016 = stablehlo.multiply %v6015, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6017 = stablehlo.add %v6016, %v4354 : tensor<64x256x1x1xf32>
    %v6018 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6019 = stablehlo.multiply %v6018, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6020 = stablehlo.add %v6019, %v6017 : tensor<64x256x1x1xf32>
    %v6021 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6022 = stablehlo.multiply %v6021, %v6020 : tensor<64x256x1x1xf32>
    %v6023 = stablehlo.subtract %s1b2W1, %v6022 : tensor<64x256x1x1xf32>
    %v6024 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6025 = stablehlo.multiply %v6024, %s1b2g1 : tensor<64xf32>
    %v6026 = stablehlo.add %v6025, %v4372 : tensor<64xf32>
    %v6027 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6028 = stablehlo.multiply %v6027, %s1b2g1v : tensor<64xf32>
    %v6029 = stablehlo.add %v6028, %v6026 : tensor<64xf32>
    %v6030 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6031 = stablehlo.multiply %v6030, %v6029 : tensor<64xf32>
    %v6032 = stablehlo.subtract %s1b2g1, %v6031 : tensor<64xf32>
    %v6033 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6034 = stablehlo.multiply %v6033, %s1b2bt1 : tensor<64xf32>
    %v6035 = stablehlo.add %v6034, %v4375 : tensor<64xf32>
    %v6036 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6037 = stablehlo.multiply %v6036, %s1b2bt1v : tensor<64xf32>
    %v6038 = stablehlo.add %v6037, %v6035 : tensor<64xf32>
    %v6039 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6040 = stablehlo.multiply %v6039, %v6038 : tensor<64xf32>
    %v6041 = stablehlo.subtract %s1b2bt1, %v6040 : tensor<64xf32>
    %v6042 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6043 = stablehlo.multiply %v6042, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6044 = stablehlo.add %v6043, %v4381 : tensor<64x64x3x3xf32>
    %v6045 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6046 = stablehlo.multiply %v6045, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6047 = stablehlo.add %v6046, %v6044 : tensor<64x64x3x3xf32>
    %v6048 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6049 = stablehlo.multiply %v6048, %v6047 : tensor<64x64x3x3xf32>
    %v6050 = stablehlo.subtract %s1b2W2, %v6049 : tensor<64x64x3x3xf32>
    %v6051 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6052 = stablehlo.multiply %v6051, %s1b2g2 : tensor<64xf32>
    %v6053 = stablehlo.add %v6052, %v4399 : tensor<64xf32>
    %v6054 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6055 = stablehlo.multiply %v6054, %s1b2g2v : tensor<64xf32>
    %v6056 = stablehlo.add %v6055, %v6053 : tensor<64xf32>
    %v6057 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6058 = stablehlo.multiply %v6057, %v6056 : tensor<64xf32>
    %v6059 = stablehlo.subtract %s1b2g2, %v6058 : tensor<64xf32>
    %v6060 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6061 = stablehlo.multiply %v6060, %s1b2bt2 : tensor<64xf32>
    %v6062 = stablehlo.add %v6061, %v4402 : tensor<64xf32>
    %v6063 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6064 = stablehlo.multiply %v6063, %s1b2bt2v : tensor<64xf32>
    %v6065 = stablehlo.add %v6064, %v6062 : tensor<64xf32>
    %v6066 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6067 = stablehlo.multiply %v6066, %v6065 : tensor<64xf32>
    %v6068 = stablehlo.subtract %s1b2bt2, %v6067 : tensor<64xf32>
    %v6069 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6070 = stablehlo.multiply %v6069, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6071 = stablehlo.add %v6070, %v4408 : tensor<256x64x1x1xf32>
    %v6072 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6073 = stablehlo.multiply %v6072, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6074 = stablehlo.add %v6073, %v6071 : tensor<256x64x1x1xf32>
    %v6075 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6076 = stablehlo.multiply %v6075, %v6074 : tensor<256x64x1x1xf32>
    %v6077 = stablehlo.subtract %s1b2W3, %v6076 : tensor<256x64x1x1xf32>
    %v6078 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6079 = stablehlo.multiply %v6078, %s1b2g3 : tensor<256xf32>
    %v6080 = stablehlo.add %v6079, %v4426 : tensor<256xf32>
    %v6081 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6082 = stablehlo.multiply %v6081, %s1b2g3v : tensor<256xf32>
    %v6083 = stablehlo.add %v6082, %v6080 : tensor<256xf32>
    %v6084 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6085 = stablehlo.multiply %v6084, %v6083 : tensor<256xf32>
    %v6086 = stablehlo.subtract %s1b2g3, %v6085 : tensor<256xf32>
    %v6087 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6088 = stablehlo.multiply %v6087, %s1b2bt3 : tensor<256xf32>
    %v6089 = stablehlo.add %v6088, %v4429 : tensor<256xf32>
    %v6090 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6091 = stablehlo.multiply %v6090, %s1b2bt3v : tensor<256xf32>
    %v6092 = stablehlo.add %v6091, %v6089 : tensor<256xf32>
    %v6093 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6094 = stablehlo.multiply %v6093, %v6092 : tensor<256xf32>
    %v6095 = stablehlo.subtract %s1b2bt3, %v6094 : tensor<256xf32>
    %v6096 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6097 = stablehlo.multiply %v6096, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6098 = stablehlo.add %v6097, %v4127 : tensor<128x256x1x1xf32>
    %v6099 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6100 = stablehlo.multiply %v6099, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6101 = stablehlo.add %v6100, %v6098 : tensor<128x256x1x1xf32>
    %v6102 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6103 = stablehlo.multiply %v6102, %v6101 : tensor<128x256x1x1xf32>
    %v6104 = stablehlo.subtract %s2b0W1, %v6103 : tensor<128x256x1x1xf32>
    %v6105 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6106 = stablehlo.multiply %v6105, %s2b0g1 : tensor<128xf32>
    %v6107 = stablehlo.add %v6106, %v4145 : tensor<128xf32>
    %v6108 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6109 = stablehlo.multiply %v6108, %s2b0g1v : tensor<128xf32>
    %v6110 = stablehlo.add %v6109, %v6107 : tensor<128xf32>
    %v6111 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6112 = stablehlo.multiply %v6111, %v6110 : tensor<128xf32>
    %v6113 = stablehlo.subtract %s2b0g1, %v6112 : tensor<128xf32>
    %v6114 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6115 = stablehlo.multiply %v6114, %s2b0bt1 : tensor<128xf32>
    %v6116 = stablehlo.add %v6115, %v4148 : tensor<128xf32>
    %v6117 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6118 = stablehlo.multiply %v6117, %s2b0bt1v : tensor<128xf32>
    %v6119 = stablehlo.add %v6118, %v6116 : tensor<128xf32>
    %v6120 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6121 = stablehlo.multiply %v6120, %v6119 : tensor<128xf32>
    %v6122 = stablehlo.subtract %s2b0bt1, %v6121 : tensor<128xf32>
    %v6123 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6124 = stablehlo.multiply %v6123, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6125 = stablehlo.add %v6124, %v4156 : tensor<128x128x3x3xf32>
    %v6126 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6127 = stablehlo.multiply %v6126, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6128 = stablehlo.add %v6127, %v6125 : tensor<128x128x3x3xf32>
    %v6129 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6130 = stablehlo.multiply %v6129, %v6128 : tensor<128x128x3x3xf32>
    %v6131 = stablehlo.subtract %s2b0W2, %v6130 : tensor<128x128x3x3xf32>
    %v6132 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6133 = stablehlo.multiply %v6132, %s2b0g2 : tensor<128xf32>
    %v6134 = stablehlo.add %v6133, %v4174 : tensor<128xf32>
    %v6135 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6136 = stablehlo.multiply %v6135, %s2b0g2v : tensor<128xf32>
    %v6137 = stablehlo.add %v6136, %v6134 : tensor<128xf32>
    %v6138 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6139 = stablehlo.multiply %v6138, %v6137 : tensor<128xf32>
    %v6140 = stablehlo.subtract %s2b0g2, %v6139 : tensor<128xf32>
    %v6141 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6142 = stablehlo.multiply %v6141, %s2b0bt2 : tensor<128xf32>
    %v6143 = stablehlo.add %v6142, %v4177 : tensor<128xf32>
    %v6144 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6145 = stablehlo.multiply %v6144, %s2b0bt2v : tensor<128xf32>
    %v6146 = stablehlo.add %v6145, %v6143 : tensor<128xf32>
    %v6147 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6148 = stablehlo.multiply %v6147, %v6146 : tensor<128xf32>
    %v6149 = stablehlo.subtract %s2b0bt2, %v6148 : tensor<128xf32>
    %v6150 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6151 = stablehlo.multiply %v6150, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6152 = stablehlo.add %v6151, %v4183 : tensor<512x128x1x1xf32>
    %v6153 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6154 = stablehlo.multiply %v6153, %s2b0W3v : tensor<512x128x1x1xf32>
    %v6155 = stablehlo.add %v6154, %v6152 : tensor<512x128x1x1xf32>
    %v6156 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6157 = stablehlo.multiply %v6156, %v6155 : tensor<512x128x1x1xf32>
    %v6158 = stablehlo.subtract %s2b0W3, %v6157 : tensor<512x128x1x1xf32>
    %v6159 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6160 = stablehlo.multiply %v6159, %s2b0g3 : tensor<512xf32>
    %v6161 = stablehlo.add %v6160, %v4201 : tensor<512xf32>
    %v6162 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6163 = stablehlo.multiply %v6162, %s2b0g3v : tensor<512xf32>
    %v6164 = stablehlo.add %v6163, %v6161 : tensor<512xf32>
    %v6165 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6166 = stablehlo.multiply %v6165, %v6164 : tensor<512xf32>
    %v6167 = stablehlo.subtract %s2b0g3, %v6166 : tensor<512xf32>
    %v6168 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6169 = stablehlo.multiply %v6168, %s2b0bt3 : tensor<512xf32>
    %v6170 = stablehlo.add %v6169, %v4204 : tensor<512xf32>
    %v6171 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6172 = stablehlo.multiply %v6171, %s2b0bt3v : tensor<512xf32>
    %v6173 = stablehlo.add %v6172, %v6170 : tensor<512xf32>
    %v6174 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6175 = stablehlo.multiply %v6174, %v6173 : tensor<512xf32>
    %v6176 = stablehlo.subtract %s2b0bt3, %v6175 : tensor<512xf32>
    %v6177 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6178 = stablehlo.multiply %v6177, %s2b0Wp : tensor<512x256x1x1xf32>
    %v6179 = stablehlo.add %v6178, %v4212 : tensor<512x256x1x1xf32>
    %v6180 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6181 = stablehlo.multiply %v6180, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v6182 = stablehlo.add %v6181, %v6179 : tensor<512x256x1x1xf32>
    %v6183 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6184 = stablehlo.multiply %v6183, %v6182 : tensor<512x256x1x1xf32>
    %v6185 = stablehlo.subtract %s2b0Wp, %v6184 : tensor<512x256x1x1xf32>
    %v6186 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6187 = stablehlo.multiply %v6186, %s2b0gp : tensor<512xf32>
    %v6188 = stablehlo.add %v6187, %v4230 : tensor<512xf32>
    %v6189 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6190 = stablehlo.multiply %v6189, %s2b0gpv : tensor<512xf32>
    %v6191 = stablehlo.add %v6190, %v6188 : tensor<512xf32>
    %v6192 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6193 = stablehlo.multiply %v6192, %v6191 : tensor<512xf32>
    %v6194 = stablehlo.subtract %s2b0gp, %v6193 : tensor<512xf32>
    %v6195 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6196 = stablehlo.multiply %v6195, %s2b0btp : tensor<512xf32>
    %v6197 = stablehlo.add %v6196, %v4233 : tensor<512xf32>
    %v6198 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6199 = stablehlo.multiply %v6198, %s2b0btpv : tensor<512xf32>
    %v6200 = stablehlo.add %v6199, %v6197 : tensor<512xf32>
    %v6201 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6202 = stablehlo.multiply %v6201, %v6200 : tensor<512xf32>
    %v6203 = stablehlo.subtract %s2b0btp, %v6202 : tensor<512xf32>
    %v6204 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6205 = stablehlo.multiply %v6204, %s2b1W1 : tensor<128x512x1x1xf32>
    %v6206 = stablehlo.add %v6205, %v3892 : tensor<128x512x1x1xf32>
    %v6207 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6208 = stablehlo.multiply %v6207, %s2b1W1v : tensor<128x512x1x1xf32>
    %v6209 = stablehlo.add %v6208, %v6206 : tensor<128x512x1x1xf32>
    %v6210 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6211 = stablehlo.multiply %v6210, %v6209 : tensor<128x512x1x1xf32>
    %v6212 = stablehlo.subtract %s2b1W1, %v6211 : tensor<128x512x1x1xf32>
    %v6213 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6214 = stablehlo.multiply %v6213, %s2b1g1 : tensor<128xf32>
    %v6215 = stablehlo.add %v6214, %v3910 : tensor<128xf32>
    %v6216 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6217 = stablehlo.multiply %v6216, %s2b1g1v : tensor<128xf32>
    %v6218 = stablehlo.add %v6217, %v6215 : tensor<128xf32>
    %v6219 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6220 = stablehlo.multiply %v6219, %v6218 : tensor<128xf32>
    %v6221 = stablehlo.subtract %s2b1g1, %v6220 : tensor<128xf32>
    %v6222 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6223 = stablehlo.multiply %v6222, %s2b1bt1 : tensor<128xf32>
    %v6224 = stablehlo.add %v6223, %v3913 : tensor<128xf32>
    %v6225 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6226 = stablehlo.multiply %v6225, %s2b1bt1v : tensor<128xf32>
    %v6227 = stablehlo.add %v6226, %v6224 : tensor<128xf32>
    %v6228 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6229 = stablehlo.multiply %v6228, %v6227 : tensor<128xf32>
    %v6230 = stablehlo.subtract %s2b1bt1, %v6229 : tensor<128xf32>
    %v6231 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6232 = stablehlo.multiply %v6231, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6233 = stablehlo.add %v6232, %v3919 : tensor<128x128x3x3xf32>
    %v6234 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6235 = stablehlo.multiply %v6234, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6236 = stablehlo.add %v6235, %v6233 : tensor<128x128x3x3xf32>
    %v6237 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6238 = stablehlo.multiply %v6237, %v6236 : tensor<128x128x3x3xf32>
    %v6239 = stablehlo.subtract %s2b1W2, %v6238 : tensor<128x128x3x3xf32>
    %v6240 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6241 = stablehlo.multiply %v6240, %s2b1g2 : tensor<128xf32>
    %v6242 = stablehlo.add %v6241, %v3937 : tensor<128xf32>
    %v6243 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6244 = stablehlo.multiply %v6243, %s2b1g2v : tensor<128xf32>
    %v6245 = stablehlo.add %v6244, %v6242 : tensor<128xf32>
    %v6246 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6247 = stablehlo.multiply %v6246, %v6245 : tensor<128xf32>
    %v6248 = stablehlo.subtract %s2b1g2, %v6247 : tensor<128xf32>
    %v6249 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6250 = stablehlo.multiply %v6249, %s2b1bt2 : tensor<128xf32>
    %v6251 = stablehlo.add %v6250, %v3940 : tensor<128xf32>
    %v6252 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6253 = stablehlo.multiply %v6252, %s2b1bt2v : tensor<128xf32>
    %v6254 = stablehlo.add %v6253, %v6251 : tensor<128xf32>
    %v6255 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6256 = stablehlo.multiply %v6255, %v6254 : tensor<128xf32>
    %v6257 = stablehlo.subtract %s2b1bt2, %v6256 : tensor<128xf32>
    %v6258 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6259 = stablehlo.multiply %v6258, %s2b1W3 : tensor<512x128x1x1xf32>
    %v6260 = stablehlo.add %v6259, %v3946 : tensor<512x128x1x1xf32>
    %v6261 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6262 = stablehlo.multiply %v6261, %s2b1W3v : tensor<512x128x1x1xf32>
    %v6263 = stablehlo.add %v6262, %v6260 : tensor<512x128x1x1xf32>
    %v6264 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6265 = stablehlo.multiply %v6264, %v6263 : tensor<512x128x1x1xf32>
    %v6266 = stablehlo.subtract %s2b1W3, %v6265 : tensor<512x128x1x1xf32>
    %v6267 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6268 = stablehlo.multiply %v6267, %s2b1g3 : tensor<512xf32>
    %v6269 = stablehlo.add %v6268, %v3964 : tensor<512xf32>
    %v6270 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6271 = stablehlo.multiply %v6270, %s2b1g3v : tensor<512xf32>
    %v6272 = stablehlo.add %v6271, %v6269 : tensor<512xf32>
    %v6273 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6274 = stablehlo.multiply %v6273, %v6272 : tensor<512xf32>
    %v6275 = stablehlo.subtract %s2b1g3, %v6274 : tensor<512xf32>
    %v6276 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6277 = stablehlo.multiply %v6276, %s2b1bt3 : tensor<512xf32>
    %v6278 = stablehlo.add %v6277, %v3967 : tensor<512xf32>
    %v6279 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6280 = stablehlo.multiply %v6279, %s2b1bt3v : tensor<512xf32>
    %v6281 = stablehlo.add %v6280, %v6278 : tensor<512xf32>
    %v6282 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6283 = stablehlo.multiply %v6282, %v6281 : tensor<512xf32>
    %v6284 = stablehlo.subtract %s2b1bt3, %v6283 : tensor<512xf32>
    %v6285 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6286 = stablehlo.multiply %v6285, %s2b2W1 : tensor<128x512x1x1xf32>
    %v6287 = stablehlo.add %v6286, %v3696 : tensor<128x512x1x1xf32>
    %v6288 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6289 = stablehlo.multiply %v6288, %s2b2W1v : tensor<128x512x1x1xf32>
    %v6290 = stablehlo.add %v6289, %v6287 : tensor<128x512x1x1xf32>
    %v6291 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6292 = stablehlo.multiply %v6291, %v6290 : tensor<128x512x1x1xf32>
    %v6293 = stablehlo.subtract %s2b2W1, %v6292 : tensor<128x512x1x1xf32>
    %v6294 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6295 = stablehlo.multiply %v6294, %s2b2g1 : tensor<128xf32>
    %v6296 = stablehlo.add %v6295, %v3714 : tensor<128xf32>
    %v6297 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6298 = stablehlo.multiply %v6297, %s2b2g1v : tensor<128xf32>
    %v6299 = stablehlo.add %v6298, %v6296 : tensor<128xf32>
    %v6300 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6301 = stablehlo.multiply %v6300, %v6299 : tensor<128xf32>
    %v6302 = stablehlo.subtract %s2b2g1, %v6301 : tensor<128xf32>
    %v6303 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6304 = stablehlo.multiply %v6303, %s2b2bt1 : tensor<128xf32>
    %v6305 = stablehlo.add %v6304, %v3717 : tensor<128xf32>
    %v6306 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6307 = stablehlo.multiply %v6306, %s2b2bt1v : tensor<128xf32>
    %v6308 = stablehlo.add %v6307, %v6305 : tensor<128xf32>
    %v6309 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6310 = stablehlo.multiply %v6309, %v6308 : tensor<128xf32>
    %v6311 = stablehlo.subtract %s2b2bt1, %v6310 : tensor<128xf32>
    %v6312 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6313 = stablehlo.multiply %v6312, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6314 = stablehlo.add %v6313, %v3723 : tensor<128x128x3x3xf32>
    %v6315 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6316 = stablehlo.multiply %v6315, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6317 = stablehlo.add %v6316, %v6314 : tensor<128x128x3x3xf32>
    %v6318 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6319 = stablehlo.multiply %v6318, %v6317 : tensor<128x128x3x3xf32>
    %v6320 = stablehlo.subtract %s2b2W2, %v6319 : tensor<128x128x3x3xf32>
    %v6321 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6322 = stablehlo.multiply %v6321, %s2b2g2 : tensor<128xf32>
    %v6323 = stablehlo.add %v6322, %v3741 : tensor<128xf32>
    %v6324 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6325 = stablehlo.multiply %v6324, %s2b2g2v : tensor<128xf32>
    %v6326 = stablehlo.add %v6325, %v6323 : tensor<128xf32>
    %v6327 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6328 = stablehlo.multiply %v6327, %v6326 : tensor<128xf32>
    %v6329 = stablehlo.subtract %s2b2g2, %v6328 : tensor<128xf32>
    %v6330 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6331 = stablehlo.multiply %v6330, %s2b2bt2 : tensor<128xf32>
    %v6332 = stablehlo.add %v6331, %v3744 : tensor<128xf32>
    %v6333 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6334 = stablehlo.multiply %v6333, %s2b2bt2v : tensor<128xf32>
    %v6335 = stablehlo.add %v6334, %v6332 : tensor<128xf32>
    %v6336 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6337 = stablehlo.multiply %v6336, %v6335 : tensor<128xf32>
    %v6338 = stablehlo.subtract %s2b2bt2, %v6337 : tensor<128xf32>
    %v6339 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6340 = stablehlo.multiply %v6339, %s2b2W3 : tensor<512x128x1x1xf32>
    %v6341 = stablehlo.add %v6340, %v3750 : tensor<512x128x1x1xf32>
    %v6342 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6343 = stablehlo.multiply %v6342, %s2b2W3v : tensor<512x128x1x1xf32>
    %v6344 = stablehlo.add %v6343, %v6341 : tensor<512x128x1x1xf32>
    %v6345 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6346 = stablehlo.multiply %v6345, %v6344 : tensor<512x128x1x1xf32>
    %v6347 = stablehlo.subtract %s2b2W3, %v6346 : tensor<512x128x1x1xf32>
    %v6348 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6349 = stablehlo.multiply %v6348, %s2b2g3 : tensor<512xf32>
    %v6350 = stablehlo.add %v6349, %v3768 : tensor<512xf32>
    %v6351 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6352 = stablehlo.multiply %v6351, %s2b2g3v : tensor<512xf32>
    %v6353 = stablehlo.add %v6352, %v6350 : tensor<512xf32>
    %v6354 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6355 = stablehlo.multiply %v6354, %v6353 : tensor<512xf32>
    %v6356 = stablehlo.subtract %s2b2g3, %v6355 : tensor<512xf32>
    %v6357 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6358 = stablehlo.multiply %v6357, %s2b2bt3 : tensor<512xf32>
    %v6359 = stablehlo.add %v6358, %v3771 : tensor<512xf32>
    %v6360 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6361 = stablehlo.multiply %v6360, %s2b2bt3v : tensor<512xf32>
    %v6362 = stablehlo.add %v6361, %v6359 : tensor<512xf32>
    %v6363 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6364 = stablehlo.multiply %v6363, %v6362 : tensor<512xf32>
    %v6365 = stablehlo.subtract %s2b2bt3, %v6364 : tensor<512xf32>
    %v6366 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6367 = stablehlo.multiply %v6366, %s2b3W1 : tensor<128x512x1x1xf32>
    %v6368 = stablehlo.add %v6367, %v3500 : tensor<128x512x1x1xf32>
    %v6369 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6370 = stablehlo.multiply %v6369, %s2b3W1v : tensor<128x512x1x1xf32>
    %v6371 = stablehlo.add %v6370, %v6368 : tensor<128x512x1x1xf32>
    %v6372 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6373 = stablehlo.multiply %v6372, %v6371 : tensor<128x512x1x1xf32>
    %v6374 = stablehlo.subtract %s2b3W1, %v6373 : tensor<128x512x1x1xf32>
    %v6375 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6376 = stablehlo.multiply %v6375, %s2b3g1 : tensor<128xf32>
    %v6377 = stablehlo.add %v6376, %v3518 : tensor<128xf32>
    %v6378 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6379 = stablehlo.multiply %v6378, %s2b3g1v : tensor<128xf32>
    %v6380 = stablehlo.add %v6379, %v6377 : tensor<128xf32>
    %v6381 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6382 = stablehlo.multiply %v6381, %v6380 : tensor<128xf32>
    %v6383 = stablehlo.subtract %s2b3g1, %v6382 : tensor<128xf32>
    %v6384 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6385 = stablehlo.multiply %v6384, %s2b3bt1 : tensor<128xf32>
    %v6386 = stablehlo.add %v6385, %v3521 : tensor<128xf32>
    %v6387 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6388 = stablehlo.multiply %v6387, %s2b3bt1v : tensor<128xf32>
    %v6389 = stablehlo.add %v6388, %v6386 : tensor<128xf32>
    %v6390 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6391 = stablehlo.multiply %v6390, %v6389 : tensor<128xf32>
    %v6392 = stablehlo.subtract %s2b3bt1, %v6391 : tensor<128xf32>
    %v6393 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6394 = stablehlo.multiply %v6393, %s2b3W2 : tensor<128x128x3x3xf32>
    %v6395 = stablehlo.add %v6394, %v3527 : tensor<128x128x3x3xf32>
    %v6396 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6397 = stablehlo.multiply %v6396, %s2b3W2v : tensor<128x128x3x3xf32>
    %v6398 = stablehlo.add %v6397, %v6395 : tensor<128x128x3x3xf32>
    %v6399 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6400 = stablehlo.multiply %v6399, %v6398 : tensor<128x128x3x3xf32>
    %v6401 = stablehlo.subtract %s2b3W2, %v6400 : tensor<128x128x3x3xf32>
    %v6402 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6403 = stablehlo.multiply %v6402, %s2b3g2 : tensor<128xf32>
    %v6404 = stablehlo.add %v6403, %v3545 : tensor<128xf32>
    %v6405 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6406 = stablehlo.multiply %v6405, %s2b3g2v : tensor<128xf32>
    %v6407 = stablehlo.add %v6406, %v6404 : tensor<128xf32>
    %v6408 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6409 = stablehlo.multiply %v6408, %v6407 : tensor<128xf32>
    %v6410 = stablehlo.subtract %s2b3g2, %v6409 : tensor<128xf32>
    %v6411 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6412 = stablehlo.multiply %v6411, %s2b3bt2 : tensor<128xf32>
    %v6413 = stablehlo.add %v6412, %v3548 : tensor<128xf32>
    %v6414 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6415 = stablehlo.multiply %v6414, %s2b3bt2v : tensor<128xf32>
    %v6416 = stablehlo.add %v6415, %v6413 : tensor<128xf32>
    %v6417 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6418 = stablehlo.multiply %v6417, %v6416 : tensor<128xf32>
    %v6419 = stablehlo.subtract %s2b3bt2, %v6418 : tensor<128xf32>
    %v6420 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6421 = stablehlo.multiply %v6420, %s2b3W3 : tensor<512x128x1x1xf32>
    %v6422 = stablehlo.add %v6421, %v3554 : tensor<512x128x1x1xf32>
    %v6423 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6424 = stablehlo.multiply %v6423, %s2b3W3v : tensor<512x128x1x1xf32>
    %v6425 = stablehlo.add %v6424, %v6422 : tensor<512x128x1x1xf32>
    %v6426 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6427 = stablehlo.multiply %v6426, %v6425 : tensor<512x128x1x1xf32>
    %v6428 = stablehlo.subtract %s2b3W3, %v6427 : tensor<512x128x1x1xf32>
    %v6429 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6430 = stablehlo.multiply %v6429, %s2b3g3 : tensor<512xf32>
    %v6431 = stablehlo.add %v6430, %v3572 : tensor<512xf32>
    %v6432 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6433 = stablehlo.multiply %v6432, %s2b3g3v : tensor<512xf32>
    %v6434 = stablehlo.add %v6433, %v6431 : tensor<512xf32>
    %v6435 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6436 = stablehlo.multiply %v6435, %v6434 : tensor<512xf32>
    %v6437 = stablehlo.subtract %s2b3g3, %v6436 : tensor<512xf32>
    %v6438 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6439 = stablehlo.multiply %v6438, %s2b3bt3 : tensor<512xf32>
    %v6440 = stablehlo.add %v6439, %v3575 : tensor<512xf32>
    %v6441 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6442 = stablehlo.multiply %v6441, %s2b3bt3v : tensor<512xf32>
    %v6443 = stablehlo.add %v6442, %v6440 : tensor<512xf32>
    %v6444 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6445 = stablehlo.multiply %v6444, %v6443 : tensor<512xf32>
    %v6446 = stablehlo.subtract %s2b3bt3, %v6445 : tensor<512xf32>
    %v6447 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6448 = stablehlo.multiply %v6447, %s3b0W1 : tensor<256x512x1x1xf32>
    %v6449 = stablehlo.add %v6448, %v3273 : tensor<256x512x1x1xf32>
    %v6450 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6451 = stablehlo.multiply %v6450, %s3b0W1v : tensor<256x512x1x1xf32>
    %v6452 = stablehlo.add %v6451, %v6449 : tensor<256x512x1x1xf32>
    %v6453 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6454 = stablehlo.multiply %v6453, %v6452 : tensor<256x512x1x1xf32>
    %v6455 = stablehlo.subtract %s3b0W1, %v6454 : tensor<256x512x1x1xf32>
    %v6456 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6457 = stablehlo.multiply %v6456, %s3b0g1 : tensor<256xf32>
    %v6458 = stablehlo.add %v6457, %v3291 : tensor<256xf32>
    %v6459 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6460 = stablehlo.multiply %v6459, %s3b0g1v : tensor<256xf32>
    %v6461 = stablehlo.add %v6460, %v6458 : tensor<256xf32>
    %v6462 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6463 = stablehlo.multiply %v6462, %v6461 : tensor<256xf32>
    %v6464 = stablehlo.subtract %s3b0g1, %v6463 : tensor<256xf32>
    %v6465 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6466 = stablehlo.multiply %v6465, %s3b0bt1 : tensor<256xf32>
    %v6467 = stablehlo.add %v6466, %v3294 : tensor<256xf32>
    %v6468 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6469 = stablehlo.multiply %v6468, %s3b0bt1v : tensor<256xf32>
    %v6470 = stablehlo.add %v6469, %v6467 : tensor<256xf32>
    %v6471 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6472 = stablehlo.multiply %v6471, %v6470 : tensor<256xf32>
    %v6473 = stablehlo.subtract %s3b0bt1, %v6472 : tensor<256xf32>
    %v6474 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6475 = stablehlo.multiply %v6474, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6476 = stablehlo.add %v6475, %v3302 : tensor<256x256x3x3xf32>
    %v6477 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6478 = stablehlo.multiply %v6477, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6479 = stablehlo.add %v6478, %v6476 : tensor<256x256x3x3xf32>
    %v6480 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6481 = stablehlo.multiply %v6480, %v6479 : tensor<256x256x3x3xf32>
    %v6482 = stablehlo.subtract %s3b0W2, %v6481 : tensor<256x256x3x3xf32>
    %v6483 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6484 = stablehlo.multiply %v6483, %s3b0g2 : tensor<256xf32>
    %v6485 = stablehlo.add %v6484, %v3320 : tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.multiply %v6486, %s3b0g2v : tensor<256xf32>
    %v6488 = stablehlo.add %v6487, %v6485 : tensor<256xf32>
    %v6489 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6490 = stablehlo.multiply %v6489, %v6488 : tensor<256xf32>
    %v6491 = stablehlo.subtract %s3b0g2, %v6490 : tensor<256xf32>
    %v6492 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6493 = stablehlo.multiply %v6492, %s3b0bt2 : tensor<256xf32>
    %v6494 = stablehlo.add %v6493, %v3323 : tensor<256xf32>
    %v6495 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6496 = stablehlo.multiply %v6495, %s3b0bt2v : tensor<256xf32>
    %v6497 = stablehlo.add %v6496, %v6494 : tensor<256xf32>
    %v6498 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6499 = stablehlo.multiply %v6498, %v6497 : tensor<256xf32>
    %v6500 = stablehlo.subtract %s3b0bt2, %v6499 : tensor<256xf32>
    %v6501 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6502 = stablehlo.multiply %v6501, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v6503 = stablehlo.add %v6502, %v3329 : tensor<1024x256x1x1xf32>
    %v6504 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6505 = stablehlo.multiply %v6504, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v6506 = stablehlo.add %v6505, %v6503 : tensor<1024x256x1x1xf32>
    %v6507 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6508 = stablehlo.multiply %v6507, %v6506 : tensor<1024x256x1x1xf32>
    %v6509 = stablehlo.subtract %s3b0W3, %v6508 : tensor<1024x256x1x1xf32>
    %v6510 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6511 = stablehlo.multiply %v6510, %s3b0g3 : tensor<1024xf32>
    %v6512 = stablehlo.add %v6511, %v3347 : tensor<1024xf32>
    %v6513 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6514 = stablehlo.multiply %v6513, %s3b0g3v : tensor<1024xf32>
    %v6515 = stablehlo.add %v6514, %v6512 : tensor<1024xf32>
    %v6516 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6517 = stablehlo.multiply %v6516, %v6515 : tensor<1024xf32>
    %v6518 = stablehlo.subtract %s3b0g3, %v6517 : tensor<1024xf32>
    %v6519 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6520 = stablehlo.multiply %v6519, %s3b0bt3 : tensor<1024xf32>
    %v6521 = stablehlo.add %v6520, %v3350 : tensor<1024xf32>
    %v6522 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6523 = stablehlo.multiply %v6522, %s3b0bt3v : tensor<1024xf32>
    %v6524 = stablehlo.add %v6523, %v6521 : tensor<1024xf32>
    %v6525 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6526 = stablehlo.multiply %v6525, %v6524 : tensor<1024xf32>
    %v6527 = stablehlo.subtract %s3b0bt3, %v6526 : tensor<1024xf32>
    %v6528 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6529 = stablehlo.multiply %v6528, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v6530 = stablehlo.add %v6529, %v3358 : tensor<1024x512x1x1xf32>
    %v6531 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6532 = stablehlo.multiply %v6531, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v6533 = stablehlo.add %v6532, %v6530 : tensor<1024x512x1x1xf32>
    %v6534 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v6535 = stablehlo.multiply %v6534, %v6533 : tensor<1024x512x1x1xf32>
    %v6536 = stablehlo.subtract %s3b0Wp, %v6535 : tensor<1024x512x1x1xf32>
    %v6537 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6538 = stablehlo.multiply %v6537, %s3b0gp : tensor<1024xf32>
    %v6539 = stablehlo.add %v6538, %v3376 : tensor<1024xf32>
    %v6540 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6541 = stablehlo.multiply %v6540, %s3b0gpv : tensor<1024xf32>
    %v6542 = stablehlo.add %v6541, %v6539 : tensor<1024xf32>
    %v6543 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6544 = stablehlo.multiply %v6543, %v6542 : tensor<1024xf32>
    %v6545 = stablehlo.subtract %s3b0gp, %v6544 : tensor<1024xf32>
    %v6546 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6547 = stablehlo.multiply %v6546, %s3b0btp : tensor<1024xf32>
    %v6548 = stablehlo.add %v6547, %v3379 : tensor<1024xf32>
    %v6549 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6550 = stablehlo.multiply %v6549, %s3b0btpv : tensor<1024xf32>
    %v6551 = stablehlo.add %v6550, %v6548 : tensor<1024xf32>
    %v6552 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6553 = stablehlo.multiply %v6552, %v6551 : tensor<1024xf32>
    %v6554 = stablehlo.subtract %s3b0btp, %v6553 : tensor<1024xf32>
    %v6555 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6556 = stablehlo.multiply %v6555, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v6557 = stablehlo.add %v6556, %v3038 : tensor<256x1024x1x1xf32>
    %v6558 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6559 = stablehlo.multiply %v6558, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v6560 = stablehlo.add %v6559, %v6557 : tensor<256x1024x1x1xf32>
    %v6561 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6562 = stablehlo.multiply %v6561, %v6560 : tensor<256x1024x1x1xf32>
    %v6563 = stablehlo.subtract %s3b1W1, %v6562 : tensor<256x1024x1x1xf32>
    %v6564 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6565 = stablehlo.multiply %v6564, %s3b1g1 : tensor<256xf32>
    %v6566 = stablehlo.add %v6565, %v3056 : tensor<256xf32>
    %v6567 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6568 = stablehlo.multiply %v6567, %s3b1g1v : tensor<256xf32>
    %v6569 = stablehlo.add %v6568, %v6566 : tensor<256xf32>
    %v6570 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6571 = stablehlo.multiply %v6570, %v6569 : tensor<256xf32>
    %v6572 = stablehlo.subtract %s3b1g1, %v6571 : tensor<256xf32>
    %v6573 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6574 = stablehlo.multiply %v6573, %s3b1bt1 : tensor<256xf32>
    %v6575 = stablehlo.add %v6574, %v3059 : tensor<256xf32>
    %v6576 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6577 = stablehlo.multiply %v6576, %s3b1bt1v : tensor<256xf32>
    %v6578 = stablehlo.add %v6577, %v6575 : tensor<256xf32>
    %v6579 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6580 = stablehlo.multiply %v6579, %v6578 : tensor<256xf32>
    %v6581 = stablehlo.subtract %s3b1bt1, %v6580 : tensor<256xf32>
    %v6582 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6583 = stablehlo.multiply %v6582, %s3b1W2 : tensor<256x256x3x3xf32>
    %v6584 = stablehlo.add %v6583, %v3065 : tensor<256x256x3x3xf32>
    %v6585 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6586 = stablehlo.multiply %v6585, %s3b1W2v : tensor<256x256x3x3xf32>
    %v6587 = stablehlo.add %v6586, %v6584 : tensor<256x256x3x3xf32>
    %v6588 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6589 = stablehlo.multiply %v6588, %v6587 : tensor<256x256x3x3xf32>
    %v6590 = stablehlo.subtract %s3b1W2, %v6589 : tensor<256x256x3x3xf32>
    %v6591 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6592 = stablehlo.multiply %v6591, %s3b1g2 : tensor<256xf32>
    %v6593 = stablehlo.add %v6592, %v3083 : tensor<256xf32>
    %v6594 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6595 = stablehlo.multiply %v6594, %s3b1g2v : tensor<256xf32>
    %v6596 = stablehlo.add %v6595, %v6593 : tensor<256xf32>
    %v6597 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6598 = stablehlo.multiply %v6597, %v6596 : tensor<256xf32>
    %v6599 = stablehlo.subtract %s3b1g2, %v6598 : tensor<256xf32>
    %v6600 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6601 = stablehlo.multiply %v6600, %s3b1bt2 : tensor<256xf32>
    %v6602 = stablehlo.add %v6601, %v3086 : tensor<256xf32>
    %v6603 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6604 = stablehlo.multiply %v6603, %s3b1bt2v : tensor<256xf32>
    %v6605 = stablehlo.add %v6604, %v6602 : tensor<256xf32>
    %v6606 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6607 = stablehlo.multiply %v6606, %v6605 : tensor<256xf32>
    %v6608 = stablehlo.subtract %s3b1bt2, %v6607 : tensor<256xf32>
    %v6609 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6610 = stablehlo.multiply %v6609, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v6611 = stablehlo.add %v6610, %v3092 : tensor<1024x256x1x1xf32>
    %v6612 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6613 = stablehlo.multiply %v6612, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v6614 = stablehlo.add %v6613, %v6611 : tensor<1024x256x1x1xf32>
    %v6615 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6616 = stablehlo.multiply %v6615, %v6614 : tensor<1024x256x1x1xf32>
    %v6617 = stablehlo.subtract %s3b1W3, %v6616 : tensor<1024x256x1x1xf32>
    %v6618 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6619 = stablehlo.multiply %v6618, %s3b1g3 : tensor<1024xf32>
    %v6620 = stablehlo.add %v6619, %v3110 : tensor<1024xf32>
    %v6621 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6622 = stablehlo.multiply %v6621, %s3b1g3v : tensor<1024xf32>
    %v6623 = stablehlo.add %v6622, %v6620 : tensor<1024xf32>
    %v6624 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6625 = stablehlo.multiply %v6624, %v6623 : tensor<1024xf32>
    %v6626 = stablehlo.subtract %s3b1g3, %v6625 : tensor<1024xf32>
    %v6627 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6628 = stablehlo.multiply %v6627, %s3b1bt3 : tensor<1024xf32>
    %v6629 = stablehlo.add %v6628, %v3113 : tensor<1024xf32>
    %v6630 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6631 = stablehlo.multiply %v6630, %s3b1bt3v : tensor<1024xf32>
    %v6632 = stablehlo.add %v6631, %v6629 : tensor<1024xf32>
    %v6633 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6634 = stablehlo.multiply %v6633, %v6632 : tensor<1024xf32>
    %v6635 = stablehlo.subtract %s3b1bt3, %v6634 : tensor<1024xf32>
    %v6636 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6637 = stablehlo.multiply %v6636, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v6638 = stablehlo.add %v6637, %v2842 : tensor<256x1024x1x1xf32>
    %v6639 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6640 = stablehlo.multiply %v6639, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v6641 = stablehlo.add %v6640, %v6638 : tensor<256x1024x1x1xf32>
    %v6642 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6643 = stablehlo.multiply %v6642, %v6641 : tensor<256x1024x1x1xf32>
    %v6644 = stablehlo.subtract %s3b2W1, %v6643 : tensor<256x1024x1x1xf32>
    %v6645 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6646 = stablehlo.multiply %v6645, %s3b2g1 : tensor<256xf32>
    %v6647 = stablehlo.add %v6646, %v2860 : tensor<256xf32>
    %v6648 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6649 = stablehlo.multiply %v6648, %s3b2g1v : tensor<256xf32>
    %v6650 = stablehlo.add %v6649, %v6647 : tensor<256xf32>
    %v6651 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6652 = stablehlo.multiply %v6651, %v6650 : tensor<256xf32>
    %v6653 = stablehlo.subtract %s3b2g1, %v6652 : tensor<256xf32>
    %v6654 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6655 = stablehlo.multiply %v6654, %s3b2bt1 : tensor<256xf32>
    %v6656 = stablehlo.add %v6655, %v2863 : tensor<256xf32>
    %v6657 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6658 = stablehlo.multiply %v6657, %s3b2bt1v : tensor<256xf32>
    %v6659 = stablehlo.add %v6658, %v6656 : tensor<256xf32>
    %v6660 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6661 = stablehlo.multiply %v6660, %v6659 : tensor<256xf32>
    %v6662 = stablehlo.subtract %s3b2bt1, %v6661 : tensor<256xf32>
    %v6663 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6664 = stablehlo.multiply %v6663, %s3b2W2 : tensor<256x256x3x3xf32>
    %v6665 = stablehlo.add %v6664, %v2869 : tensor<256x256x3x3xf32>
    %v6666 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6667 = stablehlo.multiply %v6666, %s3b2W2v : tensor<256x256x3x3xf32>
    %v6668 = stablehlo.add %v6667, %v6665 : tensor<256x256x3x3xf32>
    %v6669 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6670 = stablehlo.multiply %v6669, %v6668 : tensor<256x256x3x3xf32>
    %v6671 = stablehlo.subtract %s3b2W2, %v6670 : tensor<256x256x3x3xf32>
    %v6672 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6673 = stablehlo.multiply %v6672, %s3b2g2 : tensor<256xf32>
    %v6674 = stablehlo.add %v6673, %v2887 : tensor<256xf32>
    %v6675 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6676 = stablehlo.multiply %v6675, %s3b2g2v : tensor<256xf32>
    %v6677 = stablehlo.add %v6676, %v6674 : tensor<256xf32>
    %v6678 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6679 = stablehlo.multiply %v6678, %v6677 : tensor<256xf32>
    %v6680 = stablehlo.subtract %s3b2g2, %v6679 : tensor<256xf32>
    %v6681 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6682 = stablehlo.multiply %v6681, %s3b2bt2 : tensor<256xf32>
    %v6683 = stablehlo.add %v6682, %v2890 : tensor<256xf32>
    %v6684 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6685 = stablehlo.multiply %v6684, %s3b2bt2v : tensor<256xf32>
    %v6686 = stablehlo.add %v6685, %v6683 : tensor<256xf32>
    %v6687 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6688 = stablehlo.multiply %v6687, %v6686 : tensor<256xf32>
    %v6689 = stablehlo.subtract %s3b2bt2, %v6688 : tensor<256xf32>
    %v6690 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6691 = stablehlo.multiply %v6690, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v6692 = stablehlo.add %v6691, %v2896 : tensor<1024x256x1x1xf32>
    %v6693 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6694 = stablehlo.multiply %v6693, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v6695 = stablehlo.add %v6694, %v6692 : tensor<1024x256x1x1xf32>
    %v6696 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6697 = stablehlo.multiply %v6696, %v6695 : tensor<1024x256x1x1xf32>
    %v6698 = stablehlo.subtract %s3b2W3, %v6697 : tensor<1024x256x1x1xf32>
    %v6699 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6700 = stablehlo.multiply %v6699, %s3b2g3 : tensor<1024xf32>
    %v6701 = stablehlo.add %v6700, %v2914 : tensor<1024xf32>
    %v6702 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6703 = stablehlo.multiply %v6702, %s3b2g3v : tensor<1024xf32>
    %v6704 = stablehlo.add %v6703, %v6701 : tensor<1024xf32>
    %v6705 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6706 = stablehlo.multiply %v6705, %v6704 : tensor<1024xf32>
    %v6707 = stablehlo.subtract %s3b2g3, %v6706 : tensor<1024xf32>
    %v6708 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6709 = stablehlo.multiply %v6708, %s3b2bt3 : tensor<1024xf32>
    %v6710 = stablehlo.add %v6709, %v2917 : tensor<1024xf32>
    %v6711 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6712 = stablehlo.multiply %v6711, %s3b2bt3v : tensor<1024xf32>
    %v6713 = stablehlo.add %v6712, %v6710 : tensor<1024xf32>
    %v6714 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6715 = stablehlo.multiply %v6714, %v6713 : tensor<1024xf32>
    %v6716 = stablehlo.subtract %s3b2bt3, %v6715 : tensor<1024xf32>
    %v6717 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6718 = stablehlo.multiply %v6717, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v6719 = stablehlo.add %v6718, %v2646 : tensor<256x1024x1x1xf32>
    %v6720 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6721 = stablehlo.multiply %v6720, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v6722 = stablehlo.add %v6721, %v6719 : tensor<256x1024x1x1xf32>
    %v6723 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6724 = stablehlo.multiply %v6723, %v6722 : tensor<256x1024x1x1xf32>
    %v6725 = stablehlo.subtract %s3b3W1, %v6724 : tensor<256x1024x1x1xf32>
    %v6726 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6727 = stablehlo.multiply %v6726, %s3b3g1 : tensor<256xf32>
    %v6728 = stablehlo.add %v6727, %v2664 : tensor<256xf32>
    %v6729 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6730 = stablehlo.multiply %v6729, %s3b3g1v : tensor<256xf32>
    %v6731 = stablehlo.add %v6730, %v6728 : tensor<256xf32>
    %v6732 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6733 = stablehlo.multiply %v6732, %v6731 : tensor<256xf32>
    %v6734 = stablehlo.subtract %s3b3g1, %v6733 : tensor<256xf32>
    %v6735 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6736 = stablehlo.multiply %v6735, %s3b3bt1 : tensor<256xf32>
    %v6737 = stablehlo.add %v6736, %v2667 : tensor<256xf32>
    %v6738 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6739 = stablehlo.multiply %v6738, %s3b3bt1v : tensor<256xf32>
    %v6740 = stablehlo.add %v6739, %v6737 : tensor<256xf32>
    %v6741 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6742 = stablehlo.multiply %v6741, %v6740 : tensor<256xf32>
    %v6743 = stablehlo.subtract %s3b3bt1, %v6742 : tensor<256xf32>
    %v6744 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6745 = stablehlo.multiply %v6744, %s3b3W2 : tensor<256x256x3x3xf32>
    %v6746 = stablehlo.add %v6745, %v2673 : tensor<256x256x3x3xf32>
    %v6747 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6748 = stablehlo.multiply %v6747, %s3b3W2v : tensor<256x256x3x3xf32>
    %v6749 = stablehlo.add %v6748, %v6746 : tensor<256x256x3x3xf32>
    %v6750 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6751 = stablehlo.multiply %v6750, %v6749 : tensor<256x256x3x3xf32>
    %v6752 = stablehlo.subtract %s3b3W2, %v6751 : tensor<256x256x3x3xf32>
    %v6753 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6754 = stablehlo.multiply %v6753, %s3b3g2 : tensor<256xf32>
    %v6755 = stablehlo.add %v6754, %v2691 : tensor<256xf32>
    %v6756 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6757 = stablehlo.multiply %v6756, %s3b3g2v : tensor<256xf32>
    %v6758 = stablehlo.add %v6757, %v6755 : tensor<256xf32>
    %v6759 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6760 = stablehlo.multiply %v6759, %v6758 : tensor<256xf32>
    %v6761 = stablehlo.subtract %s3b3g2, %v6760 : tensor<256xf32>
    %v6762 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6763 = stablehlo.multiply %v6762, %s3b3bt2 : tensor<256xf32>
    %v6764 = stablehlo.add %v6763, %v2694 : tensor<256xf32>
    %v6765 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6766 = stablehlo.multiply %v6765, %s3b3bt2v : tensor<256xf32>
    %v6767 = stablehlo.add %v6766, %v6764 : tensor<256xf32>
    %v6768 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6769 = stablehlo.multiply %v6768, %v6767 : tensor<256xf32>
    %v6770 = stablehlo.subtract %s3b3bt2, %v6769 : tensor<256xf32>
    %v6771 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6772 = stablehlo.multiply %v6771, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v6773 = stablehlo.add %v6772, %v2700 : tensor<1024x256x1x1xf32>
    %v6774 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6775 = stablehlo.multiply %v6774, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v6776 = stablehlo.add %v6775, %v6773 : tensor<1024x256x1x1xf32>
    %v6777 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6778 = stablehlo.multiply %v6777, %v6776 : tensor<1024x256x1x1xf32>
    %v6779 = stablehlo.subtract %s3b3W3, %v6778 : tensor<1024x256x1x1xf32>
    %v6780 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6781 = stablehlo.multiply %v6780, %s3b3g3 : tensor<1024xf32>
    %v6782 = stablehlo.add %v6781, %v2718 : tensor<1024xf32>
    %v6783 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6784 = stablehlo.multiply %v6783, %s3b3g3v : tensor<1024xf32>
    %v6785 = stablehlo.add %v6784, %v6782 : tensor<1024xf32>
    %v6786 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6787 = stablehlo.multiply %v6786, %v6785 : tensor<1024xf32>
    %v6788 = stablehlo.subtract %s3b3g3, %v6787 : tensor<1024xf32>
    %v6789 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6790 = stablehlo.multiply %v6789, %s3b3bt3 : tensor<1024xf32>
    %v6791 = stablehlo.add %v6790, %v2721 : tensor<1024xf32>
    %v6792 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6793 = stablehlo.multiply %v6792, %s3b3bt3v : tensor<1024xf32>
    %v6794 = stablehlo.add %v6793, %v6791 : tensor<1024xf32>
    %v6795 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6796 = stablehlo.multiply %v6795, %v6794 : tensor<1024xf32>
    %v6797 = stablehlo.subtract %s3b3bt3, %v6796 : tensor<1024xf32>
    %v6798 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6799 = stablehlo.multiply %v6798, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v6800 = stablehlo.add %v6799, %v2450 : tensor<256x1024x1x1xf32>
    %v6801 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6802 = stablehlo.multiply %v6801, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v6803 = stablehlo.add %v6802, %v6800 : tensor<256x1024x1x1xf32>
    %v6804 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6805 = stablehlo.multiply %v6804, %v6803 : tensor<256x1024x1x1xf32>
    %v6806 = stablehlo.subtract %s3b4W1, %v6805 : tensor<256x1024x1x1xf32>
    %v6807 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6808 = stablehlo.multiply %v6807, %s3b4g1 : tensor<256xf32>
    %v6809 = stablehlo.add %v6808, %v2468 : tensor<256xf32>
    %v6810 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6811 = stablehlo.multiply %v6810, %s3b4g1v : tensor<256xf32>
    %v6812 = stablehlo.add %v6811, %v6809 : tensor<256xf32>
    %v6813 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6814 = stablehlo.multiply %v6813, %v6812 : tensor<256xf32>
    %v6815 = stablehlo.subtract %s3b4g1, %v6814 : tensor<256xf32>
    %v6816 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6817 = stablehlo.multiply %v6816, %s3b4bt1 : tensor<256xf32>
    %v6818 = stablehlo.add %v6817, %v2471 : tensor<256xf32>
    %v6819 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6820 = stablehlo.multiply %v6819, %s3b4bt1v : tensor<256xf32>
    %v6821 = stablehlo.add %v6820, %v6818 : tensor<256xf32>
    %v6822 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6823 = stablehlo.multiply %v6822, %v6821 : tensor<256xf32>
    %v6824 = stablehlo.subtract %s3b4bt1, %v6823 : tensor<256xf32>
    %v6825 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6826 = stablehlo.multiply %v6825, %s3b4W2 : tensor<256x256x3x3xf32>
    %v6827 = stablehlo.add %v6826, %v2477 : tensor<256x256x3x3xf32>
    %v6828 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6829 = stablehlo.multiply %v6828, %s3b4W2v : tensor<256x256x3x3xf32>
    %v6830 = stablehlo.add %v6829, %v6827 : tensor<256x256x3x3xf32>
    %v6831 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6832 = stablehlo.multiply %v6831, %v6830 : tensor<256x256x3x3xf32>
    %v6833 = stablehlo.subtract %s3b4W2, %v6832 : tensor<256x256x3x3xf32>
    %v6834 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6835 = stablehlo.multiply %v6834, %s3b4g2 : tensor<256xf32>
    %v6836 = stablehlo.add %v6835, %v2495 : tensor<256xf32>
    %v6837 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6838 = stablehlo.multiply %v6837, %s3b4g2v : tensor<256xf32>
    %v6839 = stablehlo.add %v6838, %v6836 : tensor<256xf32>
    %v6840 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6841 = stablehlo.multiply %v6840, %v6839 : tensor<256xf32>
    %v6842 = stablehlo.subtract %s3b4g2, %v6841 : tensor<256xf32>
    %v6843 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6844 = stablehlo.multiply %v6843, %s3b4bt2 : tensor<256xf32>
    %v6845 = stablehlo.add %v6844, %v2498 : tensor<256xf32>
    %v6846 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6847 = stablehlo.multiply %v6846, %s3b4bt2v : tensor<256xf32>
    %v6848 = stablehlo.add %v6847, %v6845 : tensor<256xf32>
    %v6849 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6850 = stablehlo.multiply %v6849, %v6848 : tensor<256xf32>
    %v6851 = stablehlo.subtract %s3b4bt2, %v6850 : tensor<256xf32>
    %v6852 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6853 = stablehlo.multiply %v6852, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v6854 = stablehlo.add %v6853, %v2504 : tensor<1024x256x1x1xf32>
    %v6855 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6856 = stablehlo.multiply %v6855, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v6857 = stablehlo.add %v6856, %v6854 : tensor<1024x256x1x1xf32>
    %v6858 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6859 = stablehlo.multiply %v6858, %v6857 : tensor<1024x256x1x1xf32>
    %v6860 = stablehlo.subtract %s3b4W3, %v6859 : tensor<1024x256x1x1xf32>
    %v6861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6862 = stablehlo.multiply %v6861, %s3b4g3 : tensor<1024xf32>
    %v6863 = stablehlo.add %v6862, %v2522 : tensor<1024xf32>
    %v6864 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6865 = stablehlo.multiply %v6864, %s3b4g3v : tensor<1024xf32>
    %v6866 = stablehlo.add %v6865, %v6863 : tensor<1024xf32>
    %v6867 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6868 = stablehlo.multiply %v6867, %v6866 : tensor<1024xf32>
    %v6869 = stablehlo.subtract %s3b4g3, %v6868 : tensor<1024xf32>
    %v6870 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6871 = stablehlo.multiply %v6870, %s3b4bt3 : tensor<1024xf32>
    %v6872 = stablehlo.add %v6871, %v2525 : tensor<1024xf32>
    %v6873 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6874 = stablehlo.multiply %v6873, %s3b4bt3v : tensor<1024xf32>
    %v6875 = stablehlo.add %v6874, %v6872 : tensor<1024xf32>
    %v6876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6877 = stablehlo.multiply %v6876, %v6875 : tensor<1024xf32>
    %v6878 = stablehlo.subtract %s3b4bt3, %v6877 : tensor<1024xf32>
    %v6879 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6880 = stablehlo.multiply %v6879, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v6881 = stablehlo.add %v6880, %v2254 : tensor<256x1024x1x1xf32>
    %v6882 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6883 = stablehlo.multiply %v6882, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v6884 = stablehlo.add %v6883, %v6881 : tensor<256x1024x1x1xf32>
    %v6885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v6886 = stablehlo.multiply %v6885, %v6884 : tensor<256x1024x1x1xf32>
    %v6887 = stablehlo.subtract %s3b5W1, %v6886 : tensor<256x1024x1x1xf32>
    %v6888 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6889 = stablehlo.multiply %v6888, %s3b5g1 : tensor<256xf32>
    %v6890 = stablehlo.add %v6889, %v2272 : tensor<256xf32>
    %v6891 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6892 = stablehlo.multiply %v6891, %s3b5g1v : tensor<256xf32>
    %v6893 = stablehlo.add %v6892, %v6890 : tensor<256xf32>
    %v6894 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6895 = stablehlo.multiply %v6894, %v6893 : tensor<256xf32>
    %v6896 = stablehlo.subtract %s3b5g1, %v6895 : tensor<256xf32>
    %v6897 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6898 = stablehlo.multiply %v6897, %s3b5bt1 : tensor<256xf32>
    %v6899 = stablehlo.add %v6898, %v2275 : tensor<256xf32>
    %v6900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6901 = stablehlo.multiply %v6900, %s3b5bt1v : tensor<256xf32>
    %v6902 = stablehlo.add %v6901, %v6899 : tensor<256xf32>
    %v6903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6904 = stablehlo.multiply %v6903, %v6902 : tensor<256xf32>
    %v6905 = stablehlo.subtract %s3b5bt1, %v6904 : tensor<256xf32>
    %v6906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6907 = stablehlo.multiply %v6906, %s3b5W2 : tensor<256x256x3x3xf32>
    %v6908 = stablehlo.add %v6907, %v2281 : tensor<256x256x3x3xf32>
    %v6909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6910 = stablehlo.multiply %v6909, %s3b5W2v : tensor<256x256x3x3xf32>
    %v6911 = stablehlo.add %v6910, %v6908 : tensor<256x256x3x3xf32>
    %v6912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6913 = stablehlo.multiply %v6912, %v6911 : tensor<256x256x3x3xf32>
    %v6914 = stablehlo.subtract %s3b5W2, %v6913 : tensor<256x256x3x3xf32>
    %v6915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6916 = stablehlo.multiply %v6915, %s3b5g2 : tensor<256xf32>
    %v6917 = stablehlo.add %v6916, %v2299 : tensor<256xf32>
    %v6918 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6919 = stablehlo.multiply %v6918, %s3b5g2v : tensor<256xf32>
    %v6920 = stablehlo.add %v6919, %v6917 : tensor<256xf32>
    %v6921 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6922 = stablehlo.multiply %v6921, %v6920 : tensor<256xf32>
    %v6923 = stablehlo.subtract %s3b5g2, %v6922 : tensor<256xf32>
    %v6924 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6925 = stablehlo.multiply %v6924, %s3b5bt2 : tensor<256xf32>
    %v6926 = stablehlo.add %v6925, %v2302 : tensor<256xf32>
    %v6927 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6928 = stablehlo.multiply %v6927, %s3b5bt2v : tensor<256xf32>
    %v6929 = stablehlo.add %v6928, %v6926 : tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.multiply %v6930, %v6929 : tensor<256xf32>
    %v6932 = stablehlo.subtract %s3b5bt2, %v6931 : tensor<256xf32>
    %v6933 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6934 = stablehlo.multiply %v6933, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v6935 = stablehlo.add %v6934, %v2308 : tensor<1024x256x1x1xf32>
    %v6936 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6937 = stablehlo.multiply %v6936, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v6938 = stablehlo.add %v6937, %v6935 : tensor<1024x256x1x1xf32>
    %v6939 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6940 = stablehlo.multiply %v6939, %v6938 : tensor<1024x256x1x1xf32>
    %v6941 = stablehlo.subtract %s3b5W3, %v6940 : tensor<1024x256x1x1xf32>
    %v6942 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6943 = stablehlo.multiply %v6942, %s3b5g3 : tensor<1024xf32>
    %v6944 = stablehlo.add %v6943, %v2326 : tensor<1024xf32>
    %v6945 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6946 = stablehlo.multiply %v6945, %s3b5g3v : tensor<1024xf32>
    %v6947 = stablehlo.add %v6946, %v6944 : tensor<1024xf32>
    %v6948 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6949 = stablehlo.multiply %v6948, %v6947 : tensor<1024xf32>
    %v6950 = stablehlo.subtract %s3b5g3, %v6949 : tensor<1024xf32>
    %v6951 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6952 = stablehlo.multiply %v6951, %s3b5bt3 : tensor<1024xf32>
    %v6953 = stablehlo.add %v6952, %v2329 : tensor<1024xf32>
    %v6954 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6955 = stablehlo.multiply %v6954, %s3b5bt3v : tensor<1024xf32>
    %v6956 = stablehlo.add %v6955, %v6953 : tensor<1024xf32>
    %v6957 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6958 = stablehlo.multiply %v6957, %v6956 : tensor<1024xf32>
    %v6959 = stablehlo.subtract %s3b5bt3, %v6958 : tensor<1024xf32>
    %v6960 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v6961 = stablehlo.multiply %v6960, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v6962 = stablehlo.add %v6961, %v2027 : tensor<512x1024x1x1xf32>
    %v6963 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v6964 = stablehlo.multiply %v6963, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v6965 = stablehlo.add %v6964, %v6962 : tensor<512x1024x1x1xf32>
    %v6966 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v6967 = stablehlo.multiply %v6966, %v6965 : tensor<512x1024x1x1xf32>
    %v6968 = stablehlo.subtract %s4b0W1, %v6967 : tensor<512x1024x1x1xf32>
    %v6969 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6970 = stablehlo.multiply %v6969, %s4b0g1 : tensor<512xf32>
    %v6971 = stablehlo.add %v6970, %v2045 : tensor<512xf32>
    %v6972 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6973 = stablehlo.multiply %v6972, %s4b0g1v : tensor<512xf32>
    %v6974 = stablehlo.add %v6973, %v6971 : tensor<512xf32>
    %v6975 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6976 = stablehlo.multiply %v6975, %v6974 : tensor<512xf32>
    %v6977 = stablehlo.subtract %s4b0g1, %v6976 : tensor<512xf32>
    %v6978 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6979 = stablehlo.multiply %v6978, %s4b0bt1 : tensor<512xf32>
    %v6980 = stablehlo.add %v6979, %v2048 : tensor<512xf32>
    %v6981 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6982 = stablehlo.multiply %v6981, %s4b0bt1v : tensor<512xf32>
    %v6983 = stablehlo.add %v6982, %v6980 : tensor<512xf32>
    %v6984 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6985 = stablehlo.multiply %v6984, %v6983 : tensor<512xf32>
    %v6986 = stablehlo.subtract %s4b0bt1, %v6985 : tensor<512xf32>
    %v6987 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v6988 = stablehlo.multiply %v6987, %s4b0W2 : tensor<512x512x3x3xf32>
    %v6989 = stablehlo.add %v6988, %v2056 : tensor<512x512x3x3xf32>
    %v6990 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v6991 = stablehlo.multiply %v6990, %s4b0W2v : tensor<512x512x3x3xf32>
    %v6992 = stablehlo.add %v6991, %v6989 : tensor<512x512x3x3xf32>
    %v6993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v6994 = stablehlo.multiply %v6993, %v6992 : tensor<512x512x3x3xf32>
    %v6995 = stablehlo.subtract %s4b0W2, %v6994 : tensor<512x512x3x3xf32>
    %v6996 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6997 = stablehlo.multiply %v6996, %s4b0g2 : tensor<512xf32>
    %v6998 = stablehlo.add %v6997, %v2074 : tensor<512xf32>
    %v6999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7000 = stablehlo.multiply %v6999, %s4b0g2v : tensor<512xf32>
    %v7001 = stablehlo.add %v7000, %v6998 : tensor<512xf32>
    %v7002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7003 = stablehlo.multiply %v7002, %v7001 : tensor<512xf32>
    %v7004 = stablehlo.subtract %s4b0g2, %v7003 : tensor<512xf32>
    %v7005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7006 = stablehlo.multiply %v7005, %s4b0bt2 : tensor<512xf32>
    %v7007 = stablehlo.add %v7006, %v2077 : tensor<512xf32>
    %v7008 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7009 = stablehlo.multiply %v7008, %s4b0bt2v : tensor<512xf32>
    %v7010 = stablehlo.add %v7009, %v7007 : tensor<512xf32>
    %v7011 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7012 = stablehlo.multiply %v7011, %v7010 : tensor<512xf32>
    %v7013 = stablehlo.subtract %s4b0bt2, %v7012 : tensor<512xf32>
    %v7014 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7015 = stablehlo.multiply %v7014, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7016 = stablehlo.add %v7015, %v2083 : tensor<2048x512x1x1xf32>
    %v7017 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7018 = stablehlo.multiply %v7017, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7019 = stablehlo.add %v7018, %v7016 : tensor<2048x512x1x1xf32>
    %v7020 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7021 = stablehlo.multiply %v7020, %v7019 : tensor<2048x512x1x1xf32>
    %v7022 = stablehlo.subtract %s4b0W3, %v7021 : tensor<2048x512x1x1xf32>
    %v7023 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7024 = stablehlo.multiply %v7023, %s4b0g3 : tensor<2048xf32>
    %v7025 = stablehlo.add %v7024, %v2101 : tensor<2048xf32>
    %v7026 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7027 = stablehlo.multiply %v7026, %s4b0g3v : tensor<2048xf32>
    %v7028 = stablehlo.add %v7027, %v7025 : tensor<2048xf32>
    %v7029 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7030 = stablehlo.multiply %v7029, %v7028 : tensor<2048xf32>
    %v7031 = stablehlo.subtract %s4b0g3, %v7030 : tensor<2048xf32>
    %v7032 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7033 = stablehlo.multiply %v7032, %s4b0bt3 : tensor<2048xf32>
    %v7034 = stablehlo.add %v7033, %v2104 : tensor<2048xf32>
    %v7035 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7036 = stablehlo.multiply %v7035, %s4b0bt3v : tensor<2048xf32>
    %v7037 = stablehlo.add %v7036, %v7034 : tensor<2048xf32>
    %v7038 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7039 = stablehlo.multiply %v7038, %v7037 : tensor<2048xf32>
    %v7040 = stablehlo.subtract %s4b0bt3, %v7039 : tensor<2048xf32>
    %v7041 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7042 = stablehlo.multiply %v7041, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7043 = stablehlo.add %v7042, %v2112 : tensor<2048x1024x1x1xf32>
    %v7044 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7045 = stablehlo.multiply %v7044, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7046 = stablehlo.add %v7045, %v7043 : tensor<2048x1024x1x1xf32>
    %v7047 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7048 = stablehlo.multiply %v7047, %v7046 : tensor<2048x1024x1x1xf32>
    %v7049 = stablehlo.subtract %s4b0Wp, %v7048 : tensor<2048x1024x1x1xf32>
    %v7050 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7051 = stablehlo.multiply %v7050, %s4b0gp : tensor<2048xf32>
    %v7052 = stablehlo.add %v7051, %v2130 : tensor<2048xf32>
    %v7053 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7054 = stablehlo.multiply %v7053, %s4b0gpv : tensor<2048xf32>
    %v7055 = stablehlo.add %v7054, %v7052 : tensor<2048xf32>
    %v7056 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7057 = stablehlo.multiply %v7056, %v7055 : tensor<2048xf32>
    %v7058 = stablehlo.subtract %s4b0gp, %v7057 : tensor<2048xf32>
    %v7059 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7060 = stablehlo.multiply %v7059, %s4b0btp : tensor<2048xf32>
    %v7061 = stablehlo.add %v7060, %v2133 : tensor<2048xf32>
    %v7062 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7063 = stablehlo.multiply %v7062, %s4b0btpv : tensor<2048xf32>
    %v7064 = stablehlo.add %v7063, %v7061 : tensor<2048xf32>
    %v7065 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7066 = stablehlo.multiply %v7065, %v7064 : tensor<2048xf32>
    %v7067 = stablehlo.subtract %s4b0btp, %v7066 : tensor<2048xf32>
    %v7068 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7069 = stablehlo.multiply %v7068, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7070 = stablehlo.add %v7069, %v1792 : tensor<512x2048x1x1xf32>
    %v7071 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7072 = stablehlo.multiply %v7071, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7073 = stablehlo.add %v7072, %v7070 : tensor<512x2048x1x1xf32>
    %v7074 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7075 = stablehlo.multiply %v7074, %v7073 : tensor<512x2048x1x1xf32>
    %v7076 = stablehlo.subtract %s4b1W1, %v7075 : tensor<512x2048x1x1xf32>
    %v7077 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7078 = stablehlo.multiply %v7077, %s4b1g1 : tensor<512xf32>
    %v7079 = stablehlo.add %v7078, %v1810 : tensor<512xf32>
    %v7080 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7081 = stablehlo.multiply %v7080, %s4b1g1v : tensor<512xf32>
    %v7082 = stablehlo.add %v7081, %v7079 : tensor<512xf32>
    %v7083 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7084 = stablehlo.multiply %v7083, %v7082 : tensor<512xf32>
    %v7085 = stablehlo.subtract %s4b1g1, %v7084 : tensor<512xf32>
    %v7086 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7087 = stablehlo.multiply %v7086, %s4b1bt1 : tensor<512xf32>
    %v7088 = stablehlo.add %v7087, %v1813 : tensor<512xf32>
    %v7089 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7090 = stablehlo.multiply %v7089, %s4b1bt1v : tensor<512xf32>
    %v7091 = stablehlo.add %v7090, %v7088 : tensor<512xf32>
    %v7092 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7093 = stablehlo.multiply %v7092, %v7091 : tensor<512xf32>
    %v7094 = stablehlo.subtract %s4b1bt1, %v7093 : tensor<512xf32>
    %v7095 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7096 = stablehlo.multiply %v7095, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7097 = stablehlo.add %v7096, %v1819 : tensor<512x512x3x3xf32>
    %v7098 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7099 = stablehlo.multiply %v7098, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7100 = stablehlo.add %v7099, %v7097 : tensor<512x512x3x3xf32>
    %v7101 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7102 = stablehlo.multiply %v7101, %v7100 : tensor<512x512x3x3xf32>
    %v7103 = stablehlo.subtract %s4b1W2, %v7102 : tensor<512x512x3x3xf32>
    %v7104 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7105 = stablehlo.multiply %v7104, %s4b1g2 : tensor<512xf32>
    %v7106 = stablehlo.add %v7105, %v1837 : tensor<512xf32>
    %v7107 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7108 = stablehlo.multiply %v7107, %s4b1g2v : tensor<512xf32>
    %v7109 = stablehlo.add %v7108, %v7106 : tensor<512xf32>
    %v7110 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7111 = stablehlo.multiply %v7110, %v7109 : tensor<512xf32>
    %v7112 = stablehlo.subtract %s4b1g2, %v7111 : tensor<512xf32>
    %v7113 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7114 = stablehlo.multiply %v7113, %s4b1bt2 : tensor<512xf32>
    %v7115 = stablehlo.add %v7114, %v1840 : tensor<512xf32>
    %v7116 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7117 = stablehlo.multiply %v7116, %s4b1bt2v : tensor<512xf32>
    %v7118 = stablehlo.add %v7117, %v7115 : tensor<512xf32>
    %v7119 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7120 = stablehlo.multiply %v7119, %v7118 : tensor<512xf32>
    %v7121 = stablehlo.subtract %s4b1bt2, %v7120 : tensor<512xf32>
    %v7122 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7123 = stablehlo.multiply %v7122, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7124 = stablehlo.add %v7123, %v1846 : tensor<2048x512x1x1xf32>
    %v7125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7126 = stablehlo.multiply %v7125, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7127 = stablehlo.add %v7126, %v7124 : tensor<2048x512x1x1xf32>
    %v7128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7129 = stablehlo.multiply %v7128, %v7127 : tensor<2048x512x1x1xf32>
    %v7130 = stablehlo.subtract %s4b1W3, %v7129 : tensor<2048x512x1x1xf32>
    %v7131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7132 = stablehlo.multiply %v7131, %s4b1g3 : tensor<2048xf32>
    %v7133 = stablehlo.add %v7132, %v1864 : tensor<2048xf32>
    %v7134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7135 = stablehlo.multiply %v7134, %s4b1g3v : tensor<2048xf32>
    %v7136 = stablehlo.add %v7135, %v7133 : tensor<2048xf32>
    %v7137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7138 = stablehlo.multiply %v7137, %v7136 : tensor<2048xf32>
    %v7139 = stablehlo.subtract %s4b1g3, %v7138 : tensor<2048xf32>
    %v7140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7141 = stablehlo.multiply %v7140, %s4b1bt3 : tensor<2048xf32>
    %v7142 = stablehlo.add %v7141, %v1867 : tensor<2048xf32>
    %v7143 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7144 = stablehlo.multiply %v7143, %s4b1bt3v : tensor<2048xf32>
    %v7145 = stablehlo.add %v7144, %v7142 : tensor<2048xf32>
    %v7146 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7147 = stablehlo.multiply %v7146, %v7145 : tensor<2048xf32>
    %v7148 = stablehlo.subtract %s4b1bt3, %v7147 : tensor<2048xf32>
    %v7149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7150 = stablehlo.multiply %v7149, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7151 = stablehlo.add %v7150, %v1596 : tensor<512x2048x1x1xf32>
    %v7152 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7153 = stablehlo.multiply %v7152, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7154 = stablehlo.add %v7153, %v7151 : tensor<512x2048x1x1xf32>
    %v7155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7156 = stablehlo.multiply %v7155, %v7154 : tensor<512x2048x1x1xf32>
    %v7157 = stablehlo.subtract %s4b2W1, %v7156 : tensor<512x2048x1x1xf32>
    %v7158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7159 = stablehlo.multiply %v7158, %s4b2g1 : tensor<512xf32>
    %v7160 = stablehlo.add %v7159, %v1614 : tensor<512xf32>
    %v7161 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7162 = stablehlo.multiply %v7161, %s4b2g1v : tensor<512xf32>
    %v7163 = stablehlo.add %v7162, %v7160 : tensor<512xf32>
    %v7164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7165 = stablehlo.multiply %v7164, %v7163 : tensor<512xf32>
    %v7166 = stablehlo.subtract %s4b2g1, %v7165 : tensor<512xf32>
    %v7167 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7168 = stablehlo.multiply %v7167, %s4b2bt1 : tensor<512xf32>
    %v7169 = stablehlo.add %v7168, %v1617 : tensor<512xf32>
    %v7170 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7171 = stablehlo.multiply %v7170, %s4b2bt1v : tensor<512xf32>
    %v7172 = stablehlo.add %v7171, %v7169 : tensor<512xf32>
    %v7173 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7174 = stablehlo.multiply %v7173, %v7172 : tensor<512xf32>
    %v7175 = stablehlo.subtract %s4b2bt1, %v7174 : tensor<512xf32>
    %v7176 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7177 = stablehlo.multiply %v7176, %s4b2W2 : tensor<512x512x3x3xf32>
    %v7178 = stablehlo.add %v7177, %v1623 : tensor<512x512x3x3xf32>
    %v7179 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7180 = stablehlo.multiply %v7179, %s4b2W2v : tensor<512x512x3x3xf32>
    %v7181 = stablehlo.add %v7180, %v7178 : tensor<512x512x3x3xf32>
    %v7182 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7183 = stablehlo.multiply %v7182, %v7181 : tensor<512x512x3x3xf32>
    %v7184 = stablehlo.subtract %s4b2W2, %v7183 : tensor<512x512x3x3xf32>
    %v7185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7186 = stablehlo.multiply %v7185, %s4b2g2 : tensor<512xf32>
    %v7187 = stablehlo.add %v7186, %v1641 : tensor<512xf32>
    %v7188 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7189 = stablehlo.multiply %v7188, %s4b2g2v : tensor<512xf32>
    %v7190 = stablehlo.add %v7189, %v7187 : tensor<512xf32>
    %v7191 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7192 = stablehlo.multiply %v7191, %v7190 : tensor<512xf32>
    %v7193 = stablehlo.subtract %s4b2g2, %v7192 : tensor<512xf32>
    %v7194 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7195 = stablehlo.multiply %v7194, %s4b2bt2 : tensor<512xf32>
    %v7196 = stablehlo.add %v7195, %v1644 : tensor<512xf32>
    %v7197 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7198 = stablehlo.multiply %v7197, %s4b2bt2v : tensor<512xf32>
    %v7199 = stablehlo.add %v7198, %v7196 : tensor<512xf32>
    %v7200 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7201 = stablehlo.multiply %v7200, %v7199 : tensor<512xf32>
    %v7202 = stablehlo.subtract %s4b2bt2, %v7201 : tensor<512xf32>
    %v7203 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7204 = stablehlo.multiply %v7203, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v7205 = stablehlo.add %v7204, %v1650 : tensor<2048x512x1x1xf32>
    %v7206 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7207 = stablehlo.multiply %v7206, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v7208 = stablehlo.add %v7207, %v7205 : tensor<2048x512x1x1xf32>
    %v7209 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7210 = stablehlo.multiply %v7209, %v7208 : tensor<2048x512x1x1xf32>
    %v7211 = stablehlo.subtract %s4b2W3, %v7210 : tensor<2048x512x1x1xf32>
    %v7212 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7213 = stablehlo.multiply %v7212, %s4b2g3 : tensor<2048xf32>
    %v7214 = stablehlo.add %v7213, %v1668 : tensor<2048xf32>
    %v7215 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7216 = stablehlo.multiply %v7215, %s4b2g3v : tensor<2048xf32>
    %v7217 = stablehlo.add %v7216, %v7214 : tensor<2048xf32>
    %v7218 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7219 = stablehlo.multiply %v7218, %v7217 : tensor<2048xf32>
    %v7220 = stablehlo.subtract %s4b2g3, %v7219 : tensor<2048xf32>
    %v7221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7222 = stablehlo.multiply %v7221, %s4b2bt3 : tensor<2048xf32>
    %v7223 = stablehlo.add %v7222, %v1671 : tensor<2048xf32>
    %v7224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7225 = stablehlo.multiply %v7224, %s4b2bt3v : tensor<2048xf32>
    %v7226 = stablehlo.add %v7225, %v7223 : tensor<2048xf32>
    %v7227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7228 = stablehlo.multiply %v7227, %v7226 : tensor<2048xf32>
    %v7229 = stablehlo.subtract %s4b2bt3, %v7228 : tensor<2048xf32>
    %v7230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7231 = stablehlo.multiply %v7230, %Wd : tensor<2048x1000xf32>
    %v7232 = stablehlo.add %v7231, %v1469 : tensor<2048x1000xf32>
    %v7233 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7234 = stablehlo.multiply %v7233, %Wdv : tensor<2048x1000xf32>
    %v7235 = stablehlo.add %v7234, %v7232 : tensor<2048x1000xf32>
    %v7236 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7237 = stablehlo.multiply %v7236, %v7235 : tensor<2048x1000xf32>
    %v7238 = stablehlo.subtract %Wd, %v7237 : tensor<2048x1000xf32>
    %v7239 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7240 = stablehlo.multiply %v7239, %bd : tensor<1000xf32>
    %v7241 = stablehlo.add %v7240, %v1471 : tensor<1000xf32>
    %v7242 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7243 = stablehlo.multiply %v7242, %bdv : tensor<1000xf32>
    %v7244 = stablehlo.add %v7243, %v7241 : tensor<1000xf32>
    %v7245 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7246 = stablehlo.multiply %v7245, %v7244 : tensor<1000xf32>
    %v7247 = stablehlo.subtract %bd, %v7246 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1457 : tensor<256x1000xf32>
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
    return %v5807, %v5816, %v5825, %v5834, %v5843, %v5852, %v5861, %v5870, %v5879, %v5888, %v5897, %v5906, %v5915, %v5924, %v5933, %v5942, %v5951, %v5960, %v5969, %v5978, %v5987, %v5996, %v6005, %v6014, %v6023, %v6032, %v6041, %v6050, %v6059, %v6068, %v6077, %v6086, %v6095, %v6104, %v6113, %v6122, %v6131, %v6140, %v6149, %v6158, %v6167, %v6176, %v6185, %v6194, %v6203, %v6212, %v6221, %v6230, %v6239, %v6248, %v6257, %v6266, %v6275, %v6284, %v6293, %v6302, %v6311, %v6320, %v6329, %v6338, %v6347, %v6356, %v6365, %v6374, %v6383, %v6392, %v6401, %v6410, %v6419, %v6428, %v6437, %v6446, %v6455, %v6464, %v6473, %v6482, %v6491, %v6500, %v6509, %v6518, %v6527, %v6536, %v6545, %v6554, %v6563, %v6572, %v6581, %v6590, %v6599, %v6608, %v6617, %v6626, %v6635, %v6644, %v6653, %v6662, %v6671, %v6680, %v6689, %v6698, %v6707, %v6716, %v6725, %v6734, %v6743, %v6752, %v6761, %v6770, %v6779, %v6788, %v6797, %v6806, %v6815, %v6824, %v6833, %v6842, %v6851, %v6860, %v6869, %v6878, %v6887, %v6896, %v6905, %v6914, %v6923, %v6932, %v6941, %v6950, %v6959, %v6968, %v6977, %v6986, %v6995, %v7004, %v7013, %v7022, %v7031, %v7040, %v7049, %v7058, %v7067, %v7076, %v7085, %v7094, %v7103, %v7112, %v7121, %v7130, %v7139, %v7148, %v7157, %v7166, %v7175, %v7184, %v7193, %v7202, %v7211, %v7220, %v7229, %v7238, %v7247, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v5804, %v5813, %v5822, %v5831, %v5840, %v5849, %v5858, %v5867, %v5876, %v5885, %v5894, %v5903, %v5912, %v5921, %v5930, %v5939, %v5948, %v5957, %v5966, %v5975, %v5984, %v5993, %v6002, %v6011, %v6020, %v6029, %v6038, %v6047, %v6056, %v6065, %v6074, %v6083, %v6092, %v6101, %v6110, %v6119, %v6128, %v6137, %v6146, %v6155, %v6164, %v6173, %v6182, %v6191, %v6200, %v6209, %v6218, %v6227, %v6236, %v6245, %v6254, %v6263, %v6272, %v6281, %v6290, %v6299, %v6308, %v6317, %v6326, %v6335, %v6344, %v6353, %v6362, %v6371, %v6380, %v6389, %v6398, %v6407, %v6416, %v6425, %v6434, %v6443, %v6452, %v6461, %v6470, %v6479, %v6488, %v6497, %v6506, %v6515, %v6524, %v6533, %v6542, %v6551, %v6560, %v6569, %v6578, %v6587, %v6596, %v6605, %v6614, %v6623, %v6632, %v6641, %v6650, %v6659, %v6668, %v6677, %v6686, %v6695, %v6704, %v6713, %v6722, %v6731, %v6740, %v6749, %v6758, %v6767, %v6776, %v6785, %v6794, %v6803, %v6812, %v6821, %v6830, %v6839, %v6848, %v6857, %v6866, %v6875, %v6884, %v6893, %v6902, %v6911, %v6920, %v6929, %v6938, %v6947, %v6956, %v6965, %v6974, %v6983, %v6992, %v7001, %v7010, %v7019, %v7028, %v7037, %v7046, %v7055, %v7064, %v7073, %v7082, %v7091, %v7100, %v7109, %v7118, %v7127, %v7136, %v7145, %v7154, %v7163, %v7172, %v7181, %v7190, %v7199, %v7208, %v7217, %v7226, %v7235, %v7244, %loss, %bc1, %bc2, %v4955, %v4966, %v4971, %v4982, %v4987, %v4998, %v5003, %v5014, %v5019, %v5030, %v5035, %v5046, %v5051, %v5062, %v5067, %v5078, %v5083, %v5094, %v5099, %v5110, %v5115, %v5126, %v5131, %v5142, %v5147, %v5158, %v5163, %v5174, %v5179, %v5190, %v5195, %v5206, %v5211, %v5222, %v5227, %v5238, %v5243, %v5254, %v5259, %v5270, %v5275, %v5286, %v5291, %v5302, %v5307, %v5318, %v5323, %v5334, %v5339, %v5350, %v5355, %v5366, %v5371, %v5382, %v5387, %v5398, %v5403, %v5414, %v5419, %v5430, %v5435, %v5446, %v5451, %v5462, %v5467, %v5478, %v5483, %v5494, %v5499, %v5510, %v5515, %v5526, %v5531, %v5542, %v5547, %v5558, %v5563, %v5574, %v5579, %v5590, %v5595, %v5606, %v5611, %v5622, %v5627, %v5638, %v5643, %v5654, %v5659, %v5670, %v5675, %v5686, %v5691, %v5702, %v5707, %v5718, %v5723, %v5734, %v5739, %v5750, %v5755, %v5766, %v5771, %v5782, %v5787, %v5798 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
