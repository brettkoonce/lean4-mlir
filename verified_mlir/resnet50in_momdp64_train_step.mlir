module @m {
  func.func @resnet50in_momdp64_train_step(%x: tensor<64x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x1x1xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b0W3m: tensor<256x64x1x1xf32>, %s1b0g3m: tensor<256xf32>, %s1b0bt3m: tensor<256xf32>, %s1b0Wpm: tensor<256x64x1x1xf32>, %s1b0gpm: tensor<256xf32>, %s1b0btpm: tensor<256xf32>, %s1b1W1m: tensor<64x256x1x1xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b1W3m: tensor<256x64x1x1xf32>, %s1b1g3m: tensor<256xf32>, %s1b1bt3m: tensor<256xf32>, %s1b2W1m: tensor<64x256x1x1xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %s1b2W3m: tensor<256x64x1x1xf32>, %s1b2g3m: tensor<256xf32>, %s1b2bt3m: tensor<256xf32>, %s2b0W1m: tensor<128x256x1x1xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b0W3m: tensor<512x128x1x1xf32>, %s2b0g3m: tensor<512xf32>, %s2b0bt3m: tensor<512xf32>, %s2b0Wpm: tensor<512x256x1x1xf32>, %s2b0gpm: tensor<512xf32>, %s2b0btpm: tensor<512xf32>, %s2b1W1m: tensor<128x512x1x1xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b1W3m: tensor<512x128x1x1xf32>, %s2b1g3m: tensor<512xf32>, %s2b1bt3m: tensor<512xf32>, %s2b2W1m: tensor<128x512x1x1xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %s2b2W3m: tensor<512x128x1x1xf32>, %s2b2g3m: tensor<512xf32>, %s2b2bt3m: tensor<512xf32>, %s2b3W1m: tensor<128x512x1x1xf32>, %s2b3g1m: tensor<128xf32>, %s2b3bt1m: tensor<128xf32>, %s2b3W2m: tensor<128x128x3x3xf32>, %s2b3g2m: tensor<128xf32>, %s2b3bt2m: tensor<128xf32>, %s2b3W3m: tensor<512x128x1x1xf32>, %s2b3g3m: tensor<512xf32>, %s2b3bt3m: tensor<512xf32>, %s3b0W1m: tensor<256x512x1x1xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b0W3m: tensor<1024x256x1x1xf32>, %s3b0g3m: tensor<1024xf32>, %s3b0bt3m: tensor<1024xf32>, %s3b0Wpm: tensor<1024x512x1x1xf32>, %s3b0gpm: tensor<1024xf32>, %s3b0btpm: tensor<1024xf32>, %s3b1W1m: tensor<256x1024x1x1xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b1W3m: tensor<1024x256x1x1xf32>, %s3b1g3m: tensor<1024xf32>, %s3b1bt3m: tensor<1024xf32>, %s3b2W1m: tensor<256x1024x1x1xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b2W3m: tensor<1024x256x1x1xf32>, %s3b2g3m: tensor<1024xf32>, %s3b2bt3m: tensor<1024xf32>, %s3b3W1m: tensor<256x1024x1x1xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b3W3m: tensor<1024x256x1x1xf32>, %s3b3g3m: tensor<1024xf32>, %s3b3bt3m: tensor<1024xf32>, %s3b4W1m: tensor<256x1024x1x1xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %s3b4W3m: tensor<1024x256x1x1xf32>, %s3b4g3m: tensor<1024xf32>, %s3b4bt3m: tensor<1024xf32>, %s3b5W1m: tensor<256x1024x1x1xf32>, %s3b5g1m: tensor<256xf32>, %s3b5bt1m: tensor<256xf32>, %s3b5W2m: tensor<256x256x3x3xf32>, %s3b5g2m: tensor<256xf32>, %s3b5bt2m: tensor<256xf32>, %s3b5W3m: tensor<1024x256x1x1xf32>, %s3b5g3m: tensor<1024xf32>, %s3b5bt3m: tensor<1024xf32>, %s4b0W1m: tensor<512x1024x1x1xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b0W3m: tensor<2048x512x1x1xf32>, %s4b0g3m: tensor<2048xf32>, %s4b0bt3m: tensor<2048xf32>, %s4b0Wpm: tensor<2048x1024x1x1xf32>, %s4b0gpm: tensor<2048xf32>, %s4b0btpm: tensor<2048xf32>, %s4b1W1m: tensor<512x2048x1x1xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %s4b1W3m: tensor<2048x512x1x1xf32>, %s4b1g3m: tensor<2048xf32>, %s4b1bt3m: tensor<2048xf32>, %s4b2W1m: tensor<512x2048x1x1xf32>, %s4b2g1m: tensor<512xf32>, %s4b2bt1m: tensor<512xf32>, %s4b2W2m: tensor<512x512x3x3xf32>, %s4b2g2m: tensor<512xf32>, %s4b2bt2m: tensor<512xf32>, %s4b2W3m: tensor<2048x512x1x1xf32>, %s4b2g3m: tensor<2048xf32>, %s4b2bt3m: tensor<2048xf32>, %Wdm: tensor<2048x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x1x1xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b0W3v: tensor<256x64x1x1xf32>, %s1b0g3v: tensor<256xf32>, %s1b0bt3v: tensor<256xf32>, %s1b0Wpv: tensor<256x64x1x1xf32>, %s1b0gpv: tensor<256xf32>, %s1b0btpv: tensor<256xf32>, %s1b1W1v: tensor<64x256x1x1xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b1W3v: tensor<256x64x1x1xf32>, %s1b1g3v: tensor<256xf32>, %s1b1bt3v: tensor<256xf32>, %s1b2W1v: tensor<64x256x1x1xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %s1b2W3v: tensor<256x64x1x1xf32>, %s1b2g3v: tensor<256xf32>, %s1b2bt3v: tensor<256xf32>, %s2b0W1v: tensor<128x256x1x1xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b0W3v: tensor<512x128x1x1xf32>, %s2b0g3v: tensor<512xf32>, %s2b0bt3v: tensor<512xf32>, %s2b0Wpv: tensor<512x256x1x1xf32>, %s2b0gpv: tensor<512xf32>, %s2b0btpv: tensor<512xf32>, %s2b1W1v: tensor<128x512x1x1xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b1W3v: tensor<512x128x1x1xf32>, %s2b1g3v: tensor<512xf32>, %s2b1bt3v: tensor<512xf32>, %s2b2W1v: tensor<128x512x1x1xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %s2b2W3v: tensor<512x128x1x1xf32>, %s2b2g3v: tensor<512xf32>, %s2b2bt3v: tensor<512xf32>, %s2b3W1v: tensor<128x512x1x1xf32>, %s2b3g1v: tensor<128xf32>, %s2b3bt1v: tensor<128xf32>, %s2b3W2v: tensor<128x128x3x3xf32>, %s2b3g2v: tensor<128xf32>, %s2b3bt2v: tensor<128xf32>, %s2b3W3v: tensor<512x128x1x1xf32>, %s2b3g3v: tensor<512xf32>, %s2b3bt3v: tensor<512xf32>, %s3b0W1v: tensor<256x512x1x1xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b0W3v: tensor<1024x256x1x1xf32>, %s3b0g3v: tensor<1024xf32>, %s3b0bt3v: tensor<1024xf32>, %s3b0Wpv: tensor<1024x512x1x1xf32>, %s3b0gpv: tensor<1024xf32>, %s3b0btpv: tensor<1024xf32>, %s3b1W1v: tensor<256x1024x1x1xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b1W3v: tensor<1024x256x1x1xf32>, %s3b1g3v: tensor<1024xf32>, %s3b1bt3v: tensor<1024xf32>, %s3b2W1v: tensor<256x1024x1x1xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b2W3v: tensor<1024x256x1x1xf32>, %s3b2g3v: tensor<1024xf32>, %s3b2bt3v: tensor<1024xf32>, %s3b3W1v: tensor<256x1024x1x1xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b3W3v: tensor<1024x256x1x1xf32>, %s3b3g3v: tensor<1024xf32>, %s3b3bt3v: tensor<1024xf32>, %s3b4W1v: tensor<256x1024x1x1xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %s3b4W3v: tensor<1024x256x1x1xf32>, %s3b4g3v: tensor<1024xf32>, %s3b4bt3v: tensor<1024xf32>, %s3b5W1v: tensor<256x1024x1x1xf32>, %s3b5g1v: tensor<256xf32>, %s3b5bt1v: tensor<256xf32>, %s3b5W2v: tensor<256x256x3x3xf32>, %s3b5g2v: tensor<256xf32>, %s3b5bt2v: tensor<256xf32>, %s3b5W3v: tensor<1024x256x1x1xf32>, %s3b5g3v: tensor<1024xf32>, %s3b5bt3v: tensor<1024xf32>, %s4b0W1v: tensor<512x1024x1x1xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b0W3v: tensor<2048x512x1x1xf32>, %s4b0g3v: tensor<2048xf32>, %s4b0bt3v: tensor<2048xf32>, %s4b0Wpv: tensor<2048x1024x1x1xf32>, %s4b0gpv: tensor<2048xf32>, %s4b0btpv: tensor<2048xf32>, %s4b1W1v: tensor<512x2048x1x1xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %s4b1W3v: tensor<2048x512x1x1xf32>, %s4b1g3v: tensor<2048xf32>, %s4b1bt3v: tensor<2048xf32>, %s4b2W1v: tensor<512x2048x1x1xf32>, %s4b2g1v: tensor<512xf32>, %s4b2bt1v: tensor<512xf32>, %s4b2W2v: tensor<512x512x3x3xf32>, %s4b2g2v: tensor<512xf32>, %s4b2bt2v: tensor<512xf32>, %s4b2W3v: tensor<2048x512x1x1xf32>, %s4b2g3v: tensor<2048xf32>, %s4b2bt3v: tensor<2048xf32>, %Wdv: tensor<2048x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b0n3mui: tensor<256xf32>, %s1b0n3vari: tensor<256xf32>, %s1b0npmui: tensor<256xf32>, %s1b0npvari: tensor<256xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b1n3mui: tensor<256xf32>, %s1b1n3vari: tensor<256xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %s1b2n3mui: tensor<256xf32>, %s1b2n3vari: tensor<256xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b0n3mui: tensor<512xf32>, %s2b0n3vari: tensor<512xf32>, %s2b0npmui: tensor<512xf32>, %s2b0npvari: tensor<512xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b1n3mui: tensor<512xf32>, %s2b1n3vari: tensor<512xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %s2b2n3mui: tensor<512xf32>, %s2b2n3vari: tensor<512xf32>, %s2b3n1mui: tensor<128xf32>, %s2b3n1vari: tensor<128xf32>, %s2b3n2mui: tensor<128xf32>, %s2b3n2vari: tensor<128xf32>, %s2b3n3mui: tensor<512xf32>, %s2b3n3vari: tensor<512xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b0n3mui: tensor<1024xf32>, %s3b0n3vari: tensor<1024xf32>, %s3b0npmui: tensor<1024xf32>, %s3b0npvari: tensor<1024xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b1n3mui: tensor<1024xf32>, %s3b1n3vari: tensor<1024xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b2n3mui: tensor<1024xf32>, %s3b2n3vari: tensor<1024xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b3n3mui: tensor<1024xf32>, %s3b3n3vari: tensor<1024xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %s3b4n3mui: tensor<1024xf32>, %s3b4n3vari: tensor<1024xf32>, %s3b5n1mui: tensor<256xf32>, %s3b5n1vari: tensor<256xf32>, %s3b5n2mui: tensor<256xf32>, %s3b5n2vari: tensor<256xf32>, %s3b5n3mui: tensor<1024xf32>, %s3b5n3vari: tensor<1024xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b0n3mui: tensor<2048xf32>, %s4b0n3vari: tensor<2048xf32>, %s4b0npmui: tensor<2048xf32>, %s4b0npvari: tensor<2048xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %s4b1n3mui: tensor<2048xf32>, %s4b1n3vari: tensor<2048xf32>, %s4b2n1mui: tensor<512xf32>, %s4b2n1vari: tensor<512xf32>, %s4b2n2mui: tensor<512xf32>, %s4b2n2vari: tensor<512xf32>, %s4b2n3mui: tensor<2048xf32>, %s4b2n3vari: tensor<2048xf32>, %onehot: tensor<64x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>) {
    // ── ResNet-50 bottleneck batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 4 replicas ──
    // Every gradient and update op is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes. BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet50SyncTieB.r50_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet50FwdGraphSyncFull_shard (the forward), both in
    // LeanMlir/Proofs/Nets/ResNet/.
    // The conv biases are not in the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb2048 = stablehlo.constant dense<0.0> : tensor<2048xf32>
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convolution(%v0, %sW)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xf32>, tensor<64x3x7x7xf32>) -> tensor<64x64x112x112xf32>
    %v2 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3 = stablehlo.add %v1, %v2 : tensor<64x64x112x112xf32>
    %v4 = stablehlo.reshape %v3 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6 = stablehlo.constant dense<0.0> : tensor<f32>
    %v7 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v8 = stablehlo.reduce(%v5 init: %v6) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v9 = stablehlo.divide %v8, %v7 : tensor<64xf32>
    %arsumsgmu = "stablehlo.all_reduce"(%v9) ({
    ^bb0(%arasgmu: tensor<f32>, %arbsgmu: tensor<f32>):
      %araddsgmu = stablehlo.add %arasgmu, %arbsgmu : tensor<f32>
      stablehlo.return %araddsgmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgmu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansgmu = stablehlo.divide %arsumsgmu, %arnsgmu : tensor<64xf32>
    %v10 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v11 = stablehlo.constant dense<0.0> : tensor<f32>
    %v12 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v13 = stablehlo.reduce(%v10 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v14 = stablehlo.divide %v13, %v12 : tensor<64xf32>
    %v15 = stablehlo.broadcast_in_dim %v14, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v16 = stablehlo.subtract %v10, %v15 : tensor<64x64x112x112xf32>
    %v17 = stablehlo.multiply %v16, %v16 : tensor<64x64x112x112xf32>
    %v18 = stablehlo.reduce(%v17 init: %v11) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v19 = stablehlo.divide %v18, %v12 : tensor<64xf32>
    %v20 = stablehlo.subtract %v14, %armeansgmu : tensor<64xf32>
    %v21 = stablehlo.multiply %v20, %v20 : tensor<64xf32>
    %v22 = stablehlo.add %v19, %v21 : tensor<64xf32>
    %arsumsgvar = "stablehlo.all_reduce"(%v22) ({
    ^bb0(%arasgvar: tensor<f32>, %arbsgvar: tensor<f32>):
      %araddsgvar = stablehlo.add %arasgvar, %arbsgvar : tensor<f32>
      stablehlo.return %araddsgvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgvar = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansgvar = stablehlo.divide %arsumsgvar, %arnsgvar : tensor<64xf32>
    %v23 = stablehlo.concatenate %armeansgmu, %armeansgvar, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v24 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v25 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v26 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v27 = stablehlo.broadcast_in_dim %v25, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v28 = stablehlo.broadcast_in_dim %v26, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v29 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v30 = stablehlo.add %v28, %v29 : tensor<64x64x112x112xf32>
    %v31 = stablehlo.rsqrt %v30 : tensor<64x64x112x112xf32>
    %v32 = stablehlo.subtract %v24, %v27 : tensor<64x64x112x112xf32>
    %v33 = stablehlo.multiply %v32, %v31 : tensor<64x64x112x112xf32>
    %v34 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v35 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v36 = stablehlo.multiply %v33, %v34 : tensor<64x64x112x112xf32>
    %v37 = stablehlo.add %v36, %v35 : tensor<64x64x112x112xf32>
    %v38 = stablehlo.reshape %v37 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v39 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v40 = stablehlo.maximum %v38, %v39 : tensor<64x802816xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v42 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v43 = "stablehlo.reduce_window"(%v41, %v42) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v45 = stablehlo.reshape %v44 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v46 = stablehlo.convolution(%v45, %s1b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v47 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v48 = stablehlo.add %v46, %v47 : tensor<64x64x56x56xf32>
    %v49 = stablehlo.reshape %v48 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v50 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v51 = stablehlo.constant dense<0.0> : tensor<f32>
    %v52 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v53 = stablehlo.reduce(%v50 init: %v51) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v54 = stablehlo.divide %v53, %v52 : tensor<64xf32>
    %arsums1b0g1mu = "stablehlo.all_reduce"(%v54) ({
    ^bb0(%aras1b0g1mu: tensor<f32>, %arbs1b0g1mu: tensor<f32>):
      %aradds1b0g1mu = stablehlo.add %aras1b0g1mu, %arbs1b0g1mu : tensor<f32>
      stablehlo.return %aradds1b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1mu = stablehlo.divide %arsums1b0g1mu, %arns1b0g1mu : tensor<64xf32>
    %v55 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v56 = stablehlo.constant dense<0.0> : tensor<f32>
    %v57 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v58 = stablehlo.reduce(%v55 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v59 = stablehlo.divide %v58, %v57 : tensor<64xf32>
    %v60 = stablehlo.broadcast_in_dim %v59, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v61 = stablehlo.subtract %v55, %v60 : tensor<64x64x56x56xf32>
    %v62 = stablehlo.multiply %v61, %v61 : tensor<64x64x56x56xf32>
    %v63 = stablehlo.reduce(%v62 init: %v56) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v64 = stablehlo.divide %v63, %v57 : tensor<64xf32>
    %v65 = stablehlo.subtract %v59, %armeans1b0g1mu : tensor<64xf32>
    %v66 = stablehlo.multiply %v65, %v65 : tensor<64xf32>
    %v67 = stablehlo.add %v64, %v66 : tensor<64xf32>
    %arsums1b0g1var = "stablehlo.all_reduce"(%v67) ({
    ^bb0(%aras1b0g1var: tensor<f32>, %arbs1b0g1var: tensor<f32>):
      %aradds1b0g1var = stablehlo.add %aras1b0g1var, %arbs1b0g1var : tensor<f32>
      stablehlo.return %aradds1b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1var = stablehlo.divide %arsums1b0g1var, %arns1b0g1var : tensor<64xf32>
    %v68 = stablehlo.concatenate %armeans1b0g1mu, %armeans1b0g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v69 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v70 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v71 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v72 = stablehlo.broadcast_in_dim %v70, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %v71, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v74 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v75 = stablehlo.add %v73, %v74 : tensor<64x64x56x56xf32>
    %v76 = stablehlo.rsqrt %v75 : tensor<64x64x56x56xf32>
    %v77 = stablehlo.subtract %v69, %v72 : tensor<64x64x56x56xf32>
    %v78 = stablehlo.multiply %v77, %v76 : tensor<64x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v80 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v81 = stablehlo.multiply %v78, %v79 : tensor<64x64x56x56xf32>
    %v82 = stablehlo.add %v81, %v80 : tensor<64x64x56x56xf32>
    %v83 = stablehlo.reshape %v82 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v84 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v85 = stablehlo.maximum %v83, %v84 : tensor<64x200704xf32>
    %v86 = stablehlo.reshape %v85 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v87 = stablehlo.convolution(%v86, %s1b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v89 = stablehlo.add %v87, %v88 : tensor<64x64x56x56xf32>
    %v90 = stablehlo.reshape %v89 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<f32>
    %v93 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v94 = stablehlo.reduce(%v91 init: %v92) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v95 = stablehlo.divide %v94, %v93 : tensor<64xf32>
    %arsums1b0g2mu = "stablehlo.all_reduce"(%v95) ({
    ^bb0(%aras1b0g2mu: tensor<f32>, %arbs1b0g2mu: tensor<f32>):
      %aradds1b0g2mu = stablehlo.add %aras1b0g2mu, %arbs1b0g2mu : tensor<f32>
      stablehlo.return %aradds1b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2mu = stablehlo.divide %arsums1b0g2mu, %arns1b0g2mu : tensor<64xf32>
    %v96 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<f32>
    %v98 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v99 = stablehlo.reduce(%v96 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v100 = stablehlo.divide %v99, %v98 : tensor<64xf32>
    %v101 = stablehlo.broadcast_in_dim %v100, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v102 = stablehlo.subtract %v96, %v101 : tensor<64x64x56x56xf32>
    %v103 = stablehlo.multiply %v102, %v102 : tensor<64x64x56x56xf32>
    %v104 = stablehlo.reduce(%v103 init: %v97) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v105 = stablehlo.divide %v104, %v98 : tensor<64xf32>
    %v106 = stablehlo.subtract %v100, %armeans1b0g2mu : tensor<64xf32>
    %v107 = stablehlo.multiply %v106, %v106 : tensor<64xf32>
    %v108 = stablehlo.add %v105, %v107 : tensor<64xf32>
    %arsums1b0g2var = "stablehlo.all_reduce"(%v108) ({
    ^bb0(%aras1b0g2var: tensor<f32>, %arbs1b0g2var: tensor<f32>):
      %aradds1b0g2var = stablehlo.add %aras1b0g2var, %arbs1b0g2var : tensor<f32>
      stablehlo.return %aradds1b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2var = stablehlo.divide %arsums1b0g2var, %arns1b0g2var : tensor<64xf32>
    %v109 = stablehlo.concatenate %armeans1b0g2mu, %armeans1b0g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v110 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v111 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v112 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v113 = stablehlo.broadcast_in_dim %v111, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v114 = stablehlo.broadcast_in_dim %v112, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v115 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v116 = stablehlo.add %v114, %v115 : tensor<64x64x56x56xf32>
    %v117 = stablehlo.rsqrt %v116 : tensor<64x64x56x56xf32>
    %v118 = stablehlo.subtract %v110, %v113 : tensor<64x64x56x56xf32>
    %v119 = stablehlo.multiply %v118, %v117 : tensor<64x64x56x56xf32>
    %v120 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v121 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v122 = stablehlo.multiply %v119, %v120 : tensor<64x64x56x56xf32>
    %v123 = stablehlo.add %v122, %v121 : tensor<64x64x56x56xf32>
    %v124 = stablehlo.reshape %v123 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v125 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v126 = stablehlo.maximum %v124, %v125 : tensor<64x200704xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v128 = stablehlo.convolution(%v127, %s1b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v129 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v130 = stablehlo.add %v128, %v129 : tensor<64x256x56x56xf32>
    %v131 = stablehlo.reshape %v130 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v134 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v135 = stablehlo.reduce(%v132 init: %v133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v136 = stablehlo.divide %v135, %v134 : tensor<256xf32>
    %arsums1b0g3mu = "stablehlo.all_reduce"(%v136) ({
    ^bb0(%aras1b0g3mu: tensor<f32>, %arbs1b0g3mu: tensor<f32>):
      %aradds1b0g3mu = stablehlo.add %aras1b0g3mu, %arbs1b0g3mu : tensor<f32>
      stablehlo.return %aradds1b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3mu = stablehlo.divide %arsums1b0g3mu, %arns1b0g3mu : tensor<256xf32>
    %v137 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v139 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v140 = stablehlo.reduce(%v137 init: %v138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v141 = stablehlo.divide %v140, %v139 : tensor<256xf32>
    %v142 = stablehlo.broadcast_in_dim %v141, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v143 = stablehlo.subtract %v137, %v142 : tensor<64x256x56x56xf32>
    %v144 = stablehlo.multiply %v143, %v143 : tensor<64x256x56x56xf32>
    %v145 = stablehlo.reduce(%v144 init: %v138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v146 = stablehlo.divide %v145, %v139 : tensor<256xf32>
    %v147 = stablehlo.subtract %v141, %armeans1b0g3mu : tensor<256xf32>
    %v148 = stablehlo.multiply %v147, %v147 : tensor<256xf32>
    %v149 = stablehlo.add %v146, %v148 : tensor<256xf32>
    %arsums1b0g3var = "stablehlo.all_reduce"(%v149) ({
    ^bb0(%aras1b0g3var: tensor<f32>, %arbs1b0g3var: tensor<f32>):
      %aradds1b0g3var = stablehlo.add %aras1b0g3var, %arbs1b0g3var : tensor<f32>
      stablehlo.return %aradds1b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3var = stablehlo.divide %arsums1b0g3var, %arns1b0g3var : tensor<256xf32>
    %v150 = stablehlo.concatenate %armeans1b0g3mu, %armeans1b0g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v151 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v152 = stablehlo.slice %v150 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v153 = stablehlo.slice %v150 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v154 = stablehlo.broadcast_in_dim %v152, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v155 = stablehlo.broadcast_in_dim %v153, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v156 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v157 = stablehlo.add %v155, %v156 : tensor<64x256x56x56xf32>
    %v158 = stablehlo.rsqrt %v157 : tensor<64x256x56x56xf32>
    %v159 = stablehlo.subtract %v151, %v154 : tensor<64x256x56x56xf32>
    %v160 = stablehlo.multiply %v159, %v158 : tensor<64x256x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v162 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v163 = stablehlo.multiply %v160, %v161 : tensor<64x256x56x56xf32>
    %v164 = stablehlo.add %v163, %v162 : tensor<64x256x56x56xf32>
    %v165 = stablehlo.reshape %v164 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v166 = stablehlo.reshape %v44 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v167 = stablehlo.convolution(%v166, %s1b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v168 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<64x256x56x56xf32>
    %v170 = stablehlo.reshape %v169 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v173 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v174 = stablehlo.reduce(%v171 init: %v172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v175 = stablehlo.divide %v174, %v173 : tensor<256xf32>
    %arsums1b0gpmu = "stablehlo.all_reduce"(%v175) ({
    ^bb0(%aras1b0gpmu: tensor<f32>, %arbs1b0gpmu: tensor<f32>):
      %aradds1b0gpmu = stablehlo.add %aras1b0gpmu, %arbs1b0gpmu : tensor<f32>
      stablehlo.return %aradds1b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gpmu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gpmu = stablehlo.divide %arsums1b0gpmu, %arns1b0gpmu : tensor<256xf32>
    %v176 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v178 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v179 = stablehlo.reduce(%v176 init: %v177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v180 = stablehlo.divide %v179, %v178 : tensor<256xf32>
    %v181 = stablehlo.broadcast_in_dim %v180, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v182 = stablehlo.subtract %v176, %v181 : tensor<64x256x56x56xf32>
    %v183 = stablehlo.multiply %v182, %v182 : tensor<64x256x56x56xf32>
    %v184 = stablehlo.reduce(%v183 init: %v177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v185 = stablehlo.divide %v184, %v178 : tensor<256xf32>
    %v186 = stablehlo.subtract %v180, %armeans1b0gpmu : tensor<256xf32>
    %v187 = stablehlo.multiply %v186, %v186 : tensor<256xf32>
    %v188 = stablehlo.add %v185, %v187 : tensor<256xf32>
    %arsums1b0gpvar = "stablehlo.all_reduce"(%v188) ({
    ^bb0(%aras1b0gpvar: tensor<f32>, %arbs1b0gpvar: tensor<f32>):
      %aradds1b0gpvar = stablehlo.add %aras1b0gpvar, %arbs1b0gpvar : tensor<f32>
      stablehlo.return %aradds1b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gpvar = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gpvar = stablehlo.divide %arsums1b0gpvar, %arns1b0gpvar : tensor<256xf32>
    %v189 = stablehlo.concatenate %armeans1b0gpmu, %armeans1b0gpvar, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v190 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v191 = stablehlo.slice %v189 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v192 = stablehlo.slice %v189 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v193 = stablehlo.broadcast_in_dim %v191, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v194 = stablehlo.broadcast_in_dim %v192, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v195 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v196 = stablehlo.add %v194, %v195 : tensor<64x256x56x56xf32>
    %v197 = stablehlo.rsqrt %v196 : tensor<64x256x56x56xf32>
    %v198 = stablehlo.subtract %v190, %v193 : tensor<64x256x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v197 : tensor<64x256x56x56xf32>
    %v200 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v201 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v202 = stablehlo.multiply %v199, %v200 : tensor<64x256x56x56xf32>
    %v203 = stablehlo.add %v202, %v201 : tensor<64x256x56x56xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v205 = stablehlo.add %v165, %v204 : tensor<64x802816xf32>
    %v206 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v207 = stablehlo.maximum %v205, %v206 : tensor<64x802816xf32>
    %v208 = stablehlo.reshape %v207 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v209 = stablehlo.convolution(%v208, %s1b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v211 = stablehlo.add %v209, %v210 : tensor<64x64x56x56xf32>
    %v212 = stablehlo.reshape %v211 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v215 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v216 = stablehlo.reduce(%v213 init: %v214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v217 = stablehlo.divide %v216, %v215 : tensor<64xf32>
    %arsums1b1g1mu = "stablehlo.all_reduce"(%v217) ({
    ^bb0(%aras1b1g1mu: tensor<f32>, %arbs1b1g1mu: tensor<f32>):
      %aradds1b1g1mu = stablehlo.add %aras1b1g1mu, %arbs1b1g1mu : tensor<f32>
      stablehlo.return %aradds1b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1mu = stablehlo.divide %arsums1b1g1mu, %arns1b1g1mu : tensor<64xf32>
    %v218 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v220 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v221 = stablehlo.reduce(%v218 init: %v219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v222 = stablehlo.divide %v221, %v220 : tensor<64xf32>
    %v223 = stablehlo.broadcast_in_dim %v222, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v224 = stablehlo.subtract %v218, %v223 : tensor<64x64x56x56xf32>
    %v225 = stablehlo.multiply %v224, %v224 : tensor<64x64x56x56xf32>
    %v226 = stablehlo.reduce(%v225 init: %v219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v227 = stablehlo.divide %v226, %v220 : tensor<64xf32>
    %v228 = stablehlo.subtract %v222, %armeans1b1g1mu : tensor<64xf32>
    %v229 = stablehlo.multiply %v228, %v228 : tensor<64xf32>
    %v230 = stablehlo.add %v227, %v229 : tensor<64xf32>
    %arsums1b1g1var = "stablehlo.all_reduce"(%v230) ({
    ^bb0(%aras1b1g1var: tensor<f32>, %arbs1b1g1var: tensor<f32>):
      %aradds1b1g1var = stablehlo.add %aras1b1g1var, %arbs1b1g1var : tensor<f32>
      stablehlo.return %aradds1b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1var = stablehlo.divide %arsums1b1g1var, %arns1b1g1var : tensor<64xf32>
    %v231 = stablehlo.concatenate %armeans1b1g1mu, %armeans1b1g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v232 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v233 = stablehlo.slice %v231 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v234 = stablehlo.slice %v231 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v235 = stablehlo.broadcast_in_dim %v233, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %v234, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v237 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v238 = stablehlo.add %v236, %v237 : tensor<64x64x56x56xf32>
    %v239 = stablehlo.rsqrt %v238 : tensor<64x64x56x56xf32>
    %v240 = stablehlo.subtract %v232, %v235 : tensor<64x64x56x56xf32>
    %v241 = stablehlo.multiply %v240, %v239 : tensor<64x64x56x56xf32>
    %v242 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v243 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v244 = stablehlo.multiply %v241, %v242 : tensor<64x64x56x56xf32>
    %v245 = stablehlo.add %v244, %v243 : tensor<64x64x56x56xf32>
    %v246 = stablehlo.reshape %v245 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v247 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v248 = stablehlo.maximum %v246, %v247 : tensor<64x200704xf32>
    %v249 = stablehlo.reshape %v248 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v250 = stablehlo.convolution(%v249, %s1b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v251 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v252 = stablehlo.add %v250, %v251 : tensor<64x64x56x56xf32>
    %v253 = stablehlo.reshape %v252 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v255 = stablehlo.constant dense<0.0> : tensor<f32>
    %v256 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v257 = stablehlo.reduce(%v254 init: %v255) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v258 = stablehlo.divide %v257, %v256 : tensor<64xf32>
    %arsums1b1g2mu = "stablehlo.all_reduce"(%v258) ({
    ^bb0(%aras1b1g2mu: tensor<f32>, %arbs1b1g2mu: tensor<f32>):
      %aradds1b1g2mu = stablehlo.add %aras1b1g2mu, %arbs1b1g2mu : tensor<f32>
      stablehlo.return %aradds1b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2mu = stablehlo.divide %arsums1b1g2mu, %arns1b1g2mu : tensor<64xf32>
    %v259 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v260 = stablehlo.constant dense<0.0> : tensor<f32>
    %v261 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v262 = stablehlo.reduce(%v259 init: %v260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v263 = stablehlo.divide %v262, %v261 : tensor<64xf32>
    %v264 = stablehlo.broadcast_in_dim %v263, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v265 = stablehlo.subtract %v259, %v264 : tensor<64x64x56x56xf32>
    %v266 = stablehlo.multiply %v265, %v265 : tensor<64x64x56x56xf32>
    %v267 = stablehlo.reduce(%v266 init: %v260) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v268 = stablehlo.divide %v267, %v261 : tensor<64xf32>
    %v269 = stablehlo.subtract %v263, %armeans1b1g2mu : tensor<64xf32>
    %v270 = stablehlo.multiply %v269, %v269 : tensor<64xf32>
    %v271 = stablehlo.add %v268, %v270 : tensor<64xf32>
    %arsums1b1g2var = "stablehlo.all_reduce"(%v271) ({
    ^bb0(%aras1b1g2var: tensor<f32>, %arbs1b1g2var: tensor<f32>):
      %aradds1b1g2var = stablehlo.add %aras1b1g2var, %arbs1b1g2var : tensor<f32>
      stablehlo.return %aradds1b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2var = stablehlo.divide %arsums1b1g2var, %arns1b1g2var : tensor<64xf32>
    %v272 = stablehlo.concatenate %armeans1b1g2mu, %armeans1b1g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v273 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v274 = stablehlo.slice %v272 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v275 = stablehlo.slice %v272 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v276 = stablehlo.broadcast_in_dim %v274, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v277 = stablehlo.broadcast_in_dim %v275, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v278 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v279 = stablehlo.add %v277, %v278 : tensor<64x64x56x56xf32>
    %v280 = stablehlo.rsqrt %v279 : tensor<64x64x56x56xf32>
    %v281 = stablehlo.subtract %v273, %v276 : tensor<64x64x56x56xf32>
    %v282 = stablehlo.multiply %v281, %v280 : tensor<64x64x56x56xf32>
    %v283 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v284 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v285 = stablehlo.multiply %v282, %v283 : tensor<64x64x56x56xf32>
    %v286 = stablehlo.add %v285, %v284 : tensor<64x64x56x56xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v288 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v289 = stablehlo.maximum %v287, %v288 : tensor<64x200704xf32>
    %v290 = stablehlo.reshape %v289 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v291 = stablehlo.convolution(%v290, %s1b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v292 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v293 = stablehlo.add %v291, %v292 : tensor<64x256x56x56xf32>
    %v294 = stablehlo.reshape %v293 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v297 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v298 = stablehlo.reduce(%v295 init: %v296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v299 = stablehlo.divide %v298, %v297 : tensor<256xf32>
    %arsums1b1g3mu = "stablehlo.all_reduce"(%v299) ({
    ^bb0(%aras1b1g3mu: tensor<f32>, %arbs1b1g3mu: tensor<f32>):
      %aradds1b1g3mu = stablehlo.add %aras1b1g3mu, %arbs1b1g3mu : tensor<f32>
      stablehlo.return %aradds1b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3mu = stablehlo.divide %arsums1b1g3mu, %arns1b1g3mu : tensor<256xf32>
    %v300 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v302 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v303 = stablehlo.reduce(%v300 init: %v301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v304 = stablehlo.divide %v303, %v302 : tensor<256xf32>
    %v305 = stablehlo.broadcast_in_dim %v304, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v306 = stablehlo.subtract %v300, %v305 : tensor<64x256x56x56xf32>
    %v307 = stablehlo.multiply %v306, %v306 : tensor<64x256x56x56xf32>
    %v308 = stablehlo.reduce(%v307 init: %v301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v309 = stablehlo.divide %v308, %v302 : tensor<256xf32>
    %v310 = stablehlo.subtract %v304, %armeans1b1g3mu : tensor<256xf32>
    %v311 = stablehlo.multiply %v310, %v310 : tensor<256xf32>
    %v312 = stablehlo.add %v309, %v311 : tensor<256xf32>
    %arsums1b1g3var = "stablehlo.all_reduce"(%v312) ({
    ^bb0(%aras1b1g3var: tensor<f32>, %arbs1b1g3var: tensor<f32>):
      %aradds1b1g3var = stablehlo.add %aras1b1g3var, %arbs1b1g3var : tensor<f32>
      stablehlo.return %aradds1b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3var = stablehlo.divide %arsums1b1g3var, %arns1b1g3var : tensor<256xf32>
    %v313 = stablehlo.concatenate %armeans1b1g3mu, %armeans1b1g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v314 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v315 = stablehlo.slice %v313 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v316 = stablehlo.slice %v313 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v317 = stablehlo.broadcast_in_dim %v315, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v318 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v319 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v320 = stablehlo.add %v318, %v319 : tensor<64x256x56x56xf32>
    %v321 = stablehlo.rsqrt %v320 : tensor<64x256x56x56xf32>
    %v322 = stablehlo.subtract %v314, %v317 : tensor<64x256x56x56xf32>
    %v323 = stablehlo.multiply %v322, %v321 : tensor<64x256x56x56xf32>
    %v324 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v325 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v326 = stablehlo.multiply %v323, %v324 : tensor<64x256x56x56xf32>
    %v327 = stablehlo.add %v326, %v325 : tensor<64x256x56x56xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v330 = stablehlo.reshape %v207 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v331 = stablehlo.add %v329, %v330 : tensor<64x256x56x56xf32>
    %v332 = stablehlo.reshape %v331 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v335 = stablehlo.maximum %v333, %v334 : tensor<64x256x56x56xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v338 = stablehlo.convolution(%v337, %s1b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v339 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v340 = stablehlo.add %v338, %v339 : tensor<64x64x56x56xf32>
    %v341 = stablehlo.reshape %v340 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v344 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v345 = stablehlo.reduce(%v342 init: %v343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v346 = stablehlo.divide %v345, %v344 : tensor<64xf32>
    %arsums1b2g1mu = "stablehlo.all_reduce"(%v346) ({
    ^bb0(%aras1b2g1mu: tensor<f32>, %arbs1b2g1mu: tensor<f32>):
      %aradds1b2g1mu = stablehlo.add %aras1b2g1mu, %arbs1b2g1mu : tensor<f32>
      stablehlo.return %aradds1b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1mu = stablehlo.divide %arsums1b2g1mu, %arns1b2g1mu : tensor<64xf32>
    %v347 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v349 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v350 = stablehlo.reduce(%v347 init: %v348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v351 = stablehlo.divide %v350, %v349 : tensor<64xf32>
    %v352 = stablehlo.broadcast_in_dim %v351, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v353 = stablehlo.subtract %v347, %v352 : tensor<64x64x56x56xf32>
    %v354 = stablehlo.multiply %v353, %v353 : tensor<64x64x56x56xf32>
    %v355 = stablehlo.reduce(%v354 init: %v348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v356 = stablehlo.divide %v355, %v349 : tensor<64xf32>
    %v357 = stablehlo.subtract %v351, %armeans1b2g1mu : tensor<64xf32>
    %v358 = stablehlo.multiply %v357, %v357 : tensor<64xf32>
    %v359 = stablehlo.add %v356, %v358 : tensor<64xf32>
    %arsums1b2g1var = "stablehlo.all_reduce"(%v359) ({
    ^bb0(%aras1b2g1var: tensor<f32>, %arbs1b2g1var: tensor<f32>):
      %aradds1b2g1var = stablehlo.add %aras1b2g1var, %arbs1b2g1var : tensor<f32>
      stablehlo.return %aradds1b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1var = stablehlo.divide %arsums1b2g1var, %arns1b2g1var : tensor<64xf32>
    %v360 = stablehlo.concatenate %armeans1b2g1mu, %armeans1b2g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v361 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v362 = stablehlo.slice %v360 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v363 = stablehlo.slice %v360 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v364 = stablehlo.broadcast_in_dim %v362, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v365 = stablehlo.broadcast_in_dim %v363, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v366 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v367 = stablehlo.add %v365, %v366 : tensor<64x64x56x56xf32>
    %v368 = stablehlo.rsqrt %v367 : tensor<64x64x56x56xf32>
    %v369 = stablehlo.subtract %v361, %v364 : tensor<64x64x56x56xf32>
    %v370 = stablehlo.multiply %v369, %v368 : tensor<64x64x56x56xf32>
    %v371 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v372 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v373 = stablehlo.multiply %v370, %v371 : tensor<64x64x56x56xf32>
    %v374 = stablehlo.add %v373, %v372 : tensor<64x64x56x56xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v376 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v377 = stablehlo.maximum %v375, %v376 : tensor<64x200704xf32>
    %v378 = stablehlo.reshape %v377 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v379 = stablehlo.convolution(%v378, %s1b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v380 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v381 = stablehlo.add %v379, %v380 : tensor<64x64x56x56xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v383 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v384 = stablehlo.constant dense<0.0> : tensor<f32>
    %v385 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v386 = stablehlo.reduce(%v383 init: %v384) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v387 = stablehlo.divide %v386, %v385 : tensor<64xf32>
    %arsums1b2g2mu = "stablehlo.all_reduce"(%v387) ({
    ^bb0(%aras1b2g2mu: tensor<f32>, %arbs1b2g2mu: tensor<f32>):
      %aradds1b2g2mu = stablehlo.add %aras1b2g2mu, %arbs1b2g2mu : tensor<f32>
      stablehlo.return %aradds1b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2mu = stablehlo.divide %arsums1b2g2mu, %arns1b2g2mu : tensor<64xf32>
    %v388 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v390 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v391 = stablehlo.reduce(%v388 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v392 = stablehlo.divide %v391, %v390 : tensor<64xf32>
    %v393 = stablehlo.broadcast_in_dim %v392, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v394 = stablehlo.subtract %v388, %v393 : tensor<64x64x56x56xf32>
    %v395 = stablehlo.multiply %v394, %v394 : tensor<64x64x56x56xf32>
    %v396 = stablehlo.reduce(%v395 init: %v389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v397 = stablehlo.divide %v396, %v390 : tensor<64xf32>
    %v398 = stablehlo.subtract %v392, %armeans1b2g2mu : tensor<64xf32>
    %v399 = stablehlo.multiply %v398, %v398 : tensor<64xf32>
    %v400 = stablehlo.add %v397, %v399 : tensor<64xf32>
    %arsums1b2g2var = "stablehlo.all_reduce"(%v400) ({
    ^bb0(%aras1b2g2var: tensor<f32>, %arbs1b2g2var: tensor<f32>):
      %aradds1b2g2var = stablehlo.add %aras1b2g2var, %arbs1b2g2var : tensor<f32>
      stablehlo.return %aradds1b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2var = stablehlo.divide %arsums1b2g2var, %arns1b2g2var : tensor<64xf32>
    %v401 = stablehlo.concatenate %armeans1b2g2mu, %armeans1b2g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v402 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v403 = stablehlo.slice %v401 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v404 = stablehlo.slice %v401 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v405 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v406 = stablehlo.broadcast_in_dim %v404, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v407 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v408 = stablehlo.add %v406, %v407 : tensor<64x64x56x56xf32>
    %v409 = stablehlo.rsqrt %v408 : tensor<64x64x56x56xf32>
    %v410 = stablehlo.subtract %v402, %v405 : tensor<64x64x56x56xf32>
    %v411 = stablehlo.multiply %v410, %v409 : tensor<64x64x56x56xf32>
    %v412 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v413 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v414 = stablehlo.multiply %v411, %v412 : tensor<64x64x56x56xf32>
    %v415 = stablehlo.add %v414, %v413 : tensor<64x64x56x56xf32>
    %v416 = stablehlo.reshape %v415 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v417 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v418 = stablehlo.maximum %v416, %v417 : tensor<64x200704xf32>
    %v419 = stablehlo.reshape %v418 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v420 = stablehlo.convolution(%v419, %s1b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v421 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<64x256x56x56xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v426 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v427 = stablehlo.reduce(%v424 init: %v425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v428 = stablehlo.divide %v427, %v426 : tensor<256xf32>
    %arsums1b2g3mu = "stablehlo.all_reduce"(%v428) ({
    ^bb0(%aras1b2g3mu: tensor<f32>, %arbs1b2g3mu: tensor<f32>):
      %aradds1b2g3mu = stablehlo.add %aras1b2g3mu, %arbs1b2g3mu : tensor<f32>
      stablehlo.return %aradds1b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3mu = stablehlo.divide %arsums1b2g3mu, %arns1b2g3mu : tensor<256xf32>
    %v429 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v431 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v432 = stablehlo.reduce(%v429 init: %v430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v433 = stablehlo.divide %v432, %v431 : tensor<256xf32>
    %v434 = stablehlo.broadcast_in_dim %v433, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v435 = stablehlo.subtract %v429, %v434 : tensor<64x256x56x56xf32>
    %v436 = stablehlo.multiply %v435, %v435 : tensor<64x256x56x56xf32>
    %v437 = stablehlo.reduce(%v436 init: %v430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v438 = stablehlo.divide %v437, %v431 : tensor<256xf32>
    %v439 = stablehlo.subtract %v433, %armeans1b2g3mu : tensor<256xf32>
    %v440 = stablehlo.multiply %v439, %v439 : tensor<256xf32>
    %v441 = stablehlo.add %v438, %v440 : tensor<256xf32>
    %arsums1b2g3var = "stablehlo.all_reduce"(%v441) ({
    ^bb0(%aras1b2g3var: tensor<f32>, %arbs1b2g3var: tensor<f32>):
      %aradds1b2g3var = stablehlo.add %aras1b2g3var, %arbs1b2g3var : tensor<f32>
      stablehlo.return %aradds1b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3var = stablehlo.divide %arsums1b2g3var, %arns1b2g3var : tensor<256xf32>
    %v442 = stablehlo.concatenate %armeans1b2g3mu, %armeans1b2g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v443 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v444 = stablehlo.slice %v442 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v445 = stablehlo.slice %v442 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v446 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v447 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v448 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<64x256x56x56xf32>
    %v450 = stablehlo.rsqrt %v449 : tensor<64x256x56x56xf32>
    %v451 = stablehlo.subtract %v443, %v446 : tensor<64x256x56x56xf32>
    %v452 = stablehlo.multiply %v451, %v450 : tensor<64x256x56x56xf32>
    %v453 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v454 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v455 = stablehlo.multiply %v452, %v453 : tensor<64x256x56x56xf32>
    %v456 = stablehlo.add %v455, %v454 : tensor<64x256x56x56xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v459 = stablehlo.reshape %v336 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v460 = stablehlo.add %v458, %v459 : tensor<64x256x56x56xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v464 = stablehlo.maximum %v462, %v463 : tensor<64x256x56x56xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v467 = stablehlo.convolution(%v466, %s2b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<128x256x1x1xf32>) -> tensor<64x128x56x56xf32>
    %v468 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v469 = stablehlo.add %v467, %v468 : tensor<64x128x56x56xf32>
    %v470 = stablehlo.reshape %v469 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v471 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v472 = stablehlo.constant dense<0.0> : tensor<f32>
    %v473 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v474 = stablehlo.reduce(%v471 init: %v472) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v475 = stablehlo.divide %v474, %v473 : tensor<128xf32>
    %arsums2b0g1mu = "stablehlo.all_reduce"(%v475) ({
    ^bb0(%aras2b0g1mu: tensor<f32>, %arbs2b0g1mu: tensor<f32>):
      %aradds2b0g1mu = stablehlo.add %aras2b0g1mu, %arbs2b0g1mu : tensor<f32>
      stablehlo.return %aradds2b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1mu = stablehlo.divide %arsums2b0g1mu, %arns2b0g1mu : tensor<128xf32>
    %v476 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v477 = stablehlo.constant dense<0.0> : tensor<f32>
    %v478 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v479 = stablehlo.reduce(%v476 init: %v477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v480 = stablehlo.divide %v479, %v478 : tensor<128xf32>
    %v481 = stablehlo.broadcast_in_dim %v480, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v482 = stablehlo.subtract %v476, %v481 : tensor<64x128x56x56xf32>
    %v483 = stablehlo.multiply %v482, %v482 : tensor<64x128x56x56xf32>
    %v484 = stablehlo.reduce(%v483 init: %v477) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v485 = stablehlo.divide %v484, %v478 : tensor<128xf32>
    %v486 = stablehlo.subtract %v480, %armeans2b0g1mu : tensor<128xf32>
    %v487 = stablehlo.multiply %v486, %v486 : tensor<128xf32>
    %v488 = stablehlo.add %v485, %v487 : tensor<128xf32>
    %arsums2b0g1var = "stablehlo.all_reduce"(%v488) ({
    ^bb0(%aras2b0g1var: tensor<f32>, %arbs2b0g1var: tensor<f32>):
      %aradds2b0g1var = stablehlo.add %aras2b0g1var, %arbs2b0g1var : tensor<f32>
      stablehlo.return %aradds2b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1var = stablehlo.divide %arsums2b0g1var, %arns2b0g1var : tensor<128xf32>
    %v489 = stablehlo.concatenate %armeans2b0g1mu, %armeans2b0g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v490 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v491 = stablehlo.slice %v489 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v492 = stablehlo.slice %v489 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v493 = stablehlo.broadcast_in_dim %v491, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v494 = stablehlo.broadcast_in_dim %v492, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v495 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v496 = stablehlo.add %v494, %v495 : tensor<64x128x56x56xf32>
    %v497 = stablehlo.rsqrt %v496 : tensor<64x128x56x56xf32>
    %v498 = stablehlo.subtract %v490, %v493 : tensor<64x128x56x56xf32>
    %v499 = stablehlo.multiply %v498, %v497 : tensor<64x128x56x56xf32>
    %v500 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v501 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v502 = stablehlo.multiply %v499, %v500 : tensor<64x128x56x56xf32>
    %v503 = stablehlo.add %v502, %v501 : tensor<64x128x56x56xf32>
    %v504 = stablehlo.reshape %v503 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v505 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v506 = stablehlo.maximum %v504, %v505 : tensor<64x401408xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v508 = stablehlo.convolution(%v507, %s2b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v509 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v510 = stablehlo.add %v508, %v509 : tensor<64x128x28x28xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v514 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v515 = stablehlo.reduce(%v512 init: %v513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v516 = stablehlo.divide %v515, %v514 : tensor<128xf32>
    %arsums2b0g2mu = "stablehlo.all_reduce"(%v516) ({
    ^bb0(%aras2b0g2mu: tensor<f32>, %arbs2b0g2mu: tensor<f32>):
      %aradds2b0g2mu = stablehlo.add %aras2b0g2mu, %arbs2b0g2mu : tensor<f32>
      stablehlo.return %aradds2b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2mu = stablehlo.divide %arsums2b0g2mu, %arns2b0g2mu : tensor<128xf32>
    %v517 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v519 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v520 = stablehlo.reduce(%v517 init: %v518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v521 = stablehlo.divide %v520, %v519 : tensor<128xf32>
    %v522 = stablehlo.broadcast_in_dim %v521, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v523 = stablehlo.subtract %v517, %v522 : tensor<64x128x28x28xf32>
    %v524 = stablehlo.multiply %v523, %v523 : tensor<64x128x28x28xf32>
    %v525 = stablehlo.reduce(%v524 init: %v518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v526 = stablehlo.divide %v525, %v519 : tensor<128xf32>
    %v527 = stablehlo.subtract %v521, %armeans2b0g2mu : tensor<128xf32>
    %v528 = stablehlo.multiply %v527, %v527 : tensor<128xf32>
    %v529 = stablehlo.add %v526, %v528 : tensor<128xf32>
    %arsums2b0g2var = "stablehlo.all_reduce"(%v529) ({
    ^bb0(%aras2b0g2var: tensor<f32>, %arbs2b0g2var: tensor<f32>):
      %aradds2b0g2var = stablehlo.add %aras2b0g2var, %arbs2b0g2var : tensor<f32>
      stablehlo.return %aradds2b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2var = stablehlo.divide %arsums2b0g2var, %arns2b0g2var : tensor<128xf32>
    %v530 = stablehlo.concatenate %armeans2b0g2mu, %armeans2b0g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v531 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v532 = stablehlo.slice %v530 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v533 = stablehlo.slice %v530 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v534 = stablehlo.broadcast_in_dim %v532, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v535 = stablehlo.broadcast_in_dim %v533, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v536 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v537 = stablehlo.add %v535, %v536 : tensor<64x128x28x28xf32>
    %v538 = stablehlo.rsqrt %v537 : tensor<64x128x28x28xf32>
    %v539 = stablehlo.subtract %v531, %v534 : tensor<64x128x28x28xf32>
    %v540 = stablehlo.multiply %v539, %v538 : tensor<64x128x28x28xf32>
    %v541 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v542 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v543 = stablehlo.multiply %v540, %v541 : tensor<64x128x28x28xf32>
    %v544 = stablehlo.add %v543, %v542 : tensor<64x128x28x28xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v546 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v547 = stablehlo.maximum %v545, %v546 : tensor<64x100352xf32>
    %v548 = stablehlo.reshape %v547 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v549 = stablehlo.convolution(%v548, %s2b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v550 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<64x512x28x28xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v556 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v557 = stablehlo.divide %v556, %v555 : tensor<512xf32>
    %arsums2b0g3mu = "stablehlo.all_reduce"(%v557) ({
    ^bb0(%aras2b0g3mu: tensor<f32>, %arbs2b0g3mu: tensor<f32>):
      %aradds2b0g3mu = stablehlo.add %aras2b0g3mu, %arbs2b0g3mu : tensor<f32>
      stablehlo.return %aradds2b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3mu = stablehlo.divide %arsums2b0g3mu, %arns2b0g3mu : tensor<512xf32>
    %v558 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v560 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v561 = stablehlo.reduce(%v558 init: %v559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v562 = stablehlo.divide %v561, %v560 : tensor<512xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v564 = stablehlo.subtract %v558, %v563 : tensor<64x512x28x28xf32>
    %v565 = stablehlo.multiply %v564, %v564 : tensor<64x512x28x28xf32>
    %v566 = stablehlo.reduce(%v565 init: %v559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v567 = stablehlo.divide %v566, %v560 : tensor<512xf32>
    %v568 = stablehlo.subtract %v562, %armeans2b0g3mu : tensor<512xf32>
    %v569 = stablehlo.multiply %v568, %v568 : tensor<512xf32>
    %v570 = stablehlo.add %v567, %v569 : tensor<512xf32>
    %arsums2b0g3var = "stablehlo.all_reduce"(%v570) ({
    ^bb0(%aras2b0g3var: tensor<f32>, %arbs2b0g3var: tensor<f32>):
      %aradds2b0g3var = stablehlo.add %aras2b0g3var, %arbs2b0g3var : tensor<f32>
      stablehlo.return %aradds2b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3var = stablehlo.divide %arsums2b0g3var, %arns2b0g3var : tensor<512xf32>
    %v571 = stablehlo.concatenate %armeans2b0g3mu, %armeans2b0g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v572 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v573 = stablehlo.slice %v571 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v574 = stablehlo.slice %v571 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v575 = stablehlo.broadcast_in_dim %v573, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v576 = stablehlo.broadcast_in_dim %v574, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v577 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v578 = stablehlo.add %v576, %v577 : tensor<64x512x28x28xf32>
    %v579 = stablehlo.rsqrt %v578 : tensor<64x512x28x28xf32>
    %v580 = stablehlo.subtract %v572, %v575 : tensor<64x512x28x28xf32>
    %v581 = stablehlo.multiply %v580, %v579 : tensor<64x512x28x28xf32>
    %v582 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v583 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v584 = stablehlo.multiply %v581, %v582 : tensor<64x512x28x28xf32>
    %v585 = stablehlo.add %v584, %v583 : tensor<64x512x28x28xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v587 = stablehlo.reshape %v465 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v588 = stablehlo.convolution(%v587, %s2b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v589 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v590 = stablehlo.add %v588, %v589 : tensor<64x512x28x28xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v594 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v595 = stablehlo.reduce(%v592 init: %v593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v596 = stablehlo.divide %v595, %v594 : tensor<512xf32>
    %arsums2b0gpmu = "stablehlo.all_reduce"(%v596) ({
    ^bb0(%aras2b0gpmu: tensor<f32>, %arbs2b0gpmu: tensor<f32>):
      %aradds2b0gpmu = stablehlo.add %aras2b0gpmu, %arbs2b0gpmu : tensor<f32>
      stablehlo.return %aradds2b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gpmu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gpmu = stablehlo.divide %arsums2b0gpmu, %arns2b0gpmu : tensor<512xf32>
    %v597 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v599 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v600 = stablehlo.reduce(%v597 init: %v598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v601 = stablehlo.divide %v600, %v599 : tensor<512xf32>
    %v602 = stablehlo.broadcast_in_dim %v601, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v603 = stablehlo.subtract %v597, %v602 : tensor<64x512x28x28xf32>
    %v604 = stablehlo.multiply %v603, %v603 : tensor<64x512x28x28xf32>
    %v605 = stablehlo.reduce(%v604 init: %v598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v606 = stablehlo.divide %v605, %v599 : tensor<512xf32>
    %v607 = stablehlo.subtract %v601, %armeans2b0gpmu : tensor<512xf32>
    %v608 = stablehlo.multiply %v607, %v607 : tensor<512xf32>
    %v609 = stablehlo.add %v606, %v608 : tensor<512xf32>
    %arsums2b0gpvar = "stablehlo.all_reduce"(%v609) ({
    ^bb0(%aras2b0gpvar: tensor<f32>, %arbs2b0gpvar: tensor<f32>):
      %aradds2b0gpvar = stablehlo.add %aras2b0gpvar, %arbs2b0gpvar : tensor<f32>
      stablehlo.return %aradds2b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gpvar = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gpvar = stablehlo.divide %arsums2b0gpvar, %arns2b0gpvar : tensor<512xf32>
    %v610 = stablehlo.concatenate %armeans2b0gpmu, %armeans2b0gpvar, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v611 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v612 = stablehlo.slice %v610 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v613 = stablehlo.slice %v610 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v614 = stablehlo.broadcast_in_dim %v612, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v615 = stablehlo.broadcast_in_dim %v613, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v617 = stablehlo.add %v615, %v616 : tensor<64x512x28x28xf32>
    %v618 = stablehlo.rsqrt %v617 : tensor<64x512x28x28xf32>
    %v619 = stablehlo.subtract %v611, %v614 : tensor<64x512x28x28xf32>
    %v620 = stablehlo.multiply %v619, %v618 : tensor<64x512x28x28xf32>
    %v621 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v622 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v623 = stablehlo.multiply %v620, %v621 : tensor<64x512x28x28xf32>
    %v624 = stablehlo.add %v623, %v622 : tensor<64x512x28x28xf32>
    %v625 = stablehlo.reshape %v624 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v626 = stablehlo.add %v586, %v625 : tensor<64x401408xf32>
    %v627 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v628 = stablehlo.maximum %v626, %v627 : tensor<64x401408xf32>
    %v629 = stablehlo.reshape %v628 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v630 = stablehlo.convolution(%v629, %s2b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v631 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<64x128x28x28xf32>
    %v633 = stablehlo.reshape %v632 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v634 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v635 = stablehlo.constant dense<0.0> : tensor<f32>
    %v636 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v637 = stablehlo.reduce(%v634 init: %v635) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v638 = stablehlo.divide %v637, %v636 : tensor<128xf32>
    %arsums2b1g1mu = "stablehlo.all_reduce"(%v638) ({
    ^bb0(%aras2b1g1mu: tensor<f32>, %arbs2b1g1mu: tensor<f32>):
      %aradds2b1g1mu = stablehlo.add %aras2b1g1mu, %arbs2b1g1mu : tensor<f32>
      stablehlo.return %aradds2b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1mu = stablehlo.divide %arsums2b1g1mu, %arns2b1g1mu : tensor<128xf32>
    %v639 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v641 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v642 = stablehlo.reduce(%v639 init: %v640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v643 = stablehlo.divide %v642, %v641 : tensor<128xf32>
    %v644 = stablehlo.broadcast_in_dim %v643, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v645 = stablehlo.subtract %v639, %v644 : tensor<64x128x28x28xf32>
    %v646 = stablehlo.multiply %v645, %v645 : tensor<64x128x28x28xf32>
    %v647 = stablehlo.reduce(%v646 init: %v640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v648 = stablehlo.divide %v647, %v641 : tensor<128xf32>
    %v649 = stablehlo.subtract %v643, %armeans2b1g1mu : tensor<128xf32>
    %v650 = stablehlo.multiply %v649, %v649 : tensor<128xf32>
    %v651 = stablehlo.add %v648, %v650 : tensor<128xf32>
    %arsums2b1g1var = "stablehlo.all_reduce"(%v651) ({
    ^bb0(%aras2b1g1var: tensor<f32>, %arbs2b1g1var: tensor<f32>):
      %aradds2b1g1var = stablehlo.add %aras2b1g1var, %arbs2b1g1var : tensor<f32>
      stablehlo.return %aradds2b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1var = stablehlo.divide %arsums2b1g1var, %arns2b1g1var : tensor<128xf32>
    %v652 = stablehlo.concatenate %armeans2b1g1mu, %armeans2b1g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v653 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v654 = stablehlo.slice %v652 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v655 = stablehlo.slice %v652 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v656 = stablehlo.broadcast_in_dim %v654, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v657 = stablehlo.broadcast_in_dim %v655, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v658 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v659 = stablehlo.add %v657, %v658 : tensor<64x128x28x28xf32>
    %v660 = stablehlo.rsqrt %v659 : tensor<64x128x28x28xf32>
    %v661 = stablehlo.subtract %v653, %v656 : tensor<64x128x28x28xf32>
    %v662 = stablehlo.multiply %v661, %v660 : tensor<64x128x28x28xf32>
    %v663 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v664 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v665 = stablehlo.multiply %v662, %v663 : tensor<64x128x28x28xf32>
    %v666 = stablehlo.add %v665, %v664 : tensor<64x128x28x28xf32>
    %v667 = stablehlo.reshape %v666 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v668 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v669 = stablehlo.maximum %v667, %v668 : tensor<64x100352xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v671 = stablehlo.convolution(%v670, %s2b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v672 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v673 = stablehlo.add %v671, %v672 : tensor<64x128x28x28xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v675 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v677 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v678 = stablehlo.reduce(%v675 init: %v676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v679 = stablehlo.divide %v678, %v677 : tensor<128xf32>
    %arsums2b1g2mu = "stablehlo.all_reduce"(%v679) ({
    ^bb0(%aras2b1g2mu: tensor<f32>, %arbs2b1g2mu: tensor<f32>):
      %aradds2b1g2mu = stablehlo.add %aras2b1g2mu, %arbs2b1g2mu : tensor<f32>
      stablehlo.return %aradds2b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2mu = stablehlo.divide %arsums2b1g2mu, %arns2b1g2mu : tensor<128xf32>
    %v680 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v682 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v683 = stablehlo.reduce(%v680 init: %v681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v684 = stablehlo.divide %v683, %v682 : tensor<128xf32>
    %v685 = stablehlo.broadcast_in_dim %v684, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v686 = stablehlo.subtract %v680, %v685 : tensor<64x128x28x28xf32>
    %v687 = stablehlo.multiply %v686, %v686 : tensor<64x128x28x28xf32>
    %v688 = stablehlo.reduce(%v687 init: %v681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v689 = stablehlo.divide %v688, %v682 : tensor<128xf32>
    %v690 = stablehlo.subtract %v684, %armeans2b1g2mu : tensor<128xf32>
    %v691 = stablehlo.multiply %v690, %v690 : tensor<128xf32>
    %v692 = stablehlo.add %v689, %v691 : tensor<128xf32>
    %arsums2b1g2var = "stablehlo.all_reduce"(%v692) ({
    ^bb0(%aras2b1g2var: tensor<f32>, %arbs2b1g2var: tensor<f32>):
      %aradds2b1g2var = stablehlo.add %aras2b1g2var, %arbs2b1g2var : tensor<f32>
      stablehlo.return %aradds2b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2var = stablehlo.divide %arsums2b1g2var, %arns2b1g2var : tensor<128xf32>
    %v693 = stablehlo.concatenate %armeans2b1g2mu, %armeans2b1g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v694 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v695 = stablehlo.slice %v693 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v696 = stablehlo.slice %v693 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v697 = stablehlo.broadcast_in_dim %v695, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v698 = stablehlo.broadcast_in_dim %v696, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v699 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v700 = stablehlo.add %v698, %v699 : tensor<64x128x28x28xf32>
    %v701 = stablehlo.rsqrt %v700 : tensor<64x128x28x28xf32>
    %v702 = stablehlo.subtract %v694, %v697 : tensor<64x128x28x28xf32>
    %v703 = stablehlo.multiply %v702, %v701 : tensor<64x128x28x28xf32>
    %v704 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v705 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v706 = stablehlo.multiply %v703, %v704 : tensor<64x128x28x28xf32>
    %v707 = stablehlo.add %v706, %v705 : tensor<64x128x28x28xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v710 = stablehlo.maximum %v708, %v709 : tensor<64x100352xf32>
    %v711 = stablehlo.reshape %v710 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v712 = stablehlo.convolution(%v711, %s2b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v713 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v714 = stablehlo.add %v712, %v713 : tensor<64x512x28x28xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v716 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v718 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v719 = stablehlo.reduce(%v716 init: %v717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v720 = stablehlo.divide %v719, %v718 : tensor<512xf32>
    %arsums2b1g3mu = "stablehlo.all_reduce"(%v720) ({
    ^bb0(%aras2b1g3mu: tensor<f32>, %arbs2b1g3mu: tensor<f32>):
      %aradds2b1g3mu = stablehlo.add %aras2b1g3mu, %arbs2b1g3mu : tensor<f32>
      stablehlo.return %aradds2b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3mu = stablehlo.divide %arsums2b1g3mu, %arns2b1g3mu : tensor<512xf32>
    %v721 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v723 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v724 = stablehlo.reduce(%v721 init: %v722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v725 = stablehlo.divide %v724, %v723 : tensor<512xf32>
    %v726 = stablehlo.broadcast_in_dim %v725, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v727 = stablehlo.subtract %v721, %v726 : tensor<64x512x28x28xf32>
    %v728 = stablehlo.multiply %v727, %v727 : tensor<64x512x28x28xf32>
    %v729 = stablehlo.reduce(%v728 init: %v722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v730 = stablehlo.divide %v729, %v723 : tensor<512xf32>
    %v731 = stablehlo.subtract %v725, %armeans2b1g3mu : tensor<512xf32>
    %v732 = stablehlo.multiply %v731, %v731 : tensor<512xf32>
    %v733 = stablehlo.add %v730, %v732 : tensor<512xf32>
    %arsums2b1g3var = "stablehlo.all_reduce"(%v733) ({
    ^bb0(%aras2b1g3var: tensor<f32>, %arbs2b1g3var: tensor<f32>):
      %aradds2b1g3var = stablehlo.add %aras2b1g3var, %arbs2b1g3var : tensor<f32>
      stablehlo.return %aradds2b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3var = stablehlo.divide %arsums2b1g3var, %arns2b1g3var : tensor<512xf32>
    %v734 = stablehlo.concatenate %armeans2b1g3mu, %armeans2b1g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v735 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v736 = stablehlo.slice %v734 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v737 = stablehlo.slice %v734 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v738 = stablehlo.broadcast_in_dim %v736, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v739 = stablehlo.broadcast_in_dim %v737, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v740 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v741 = stablehlo.add %v739, %v740 : tensor<64x512x28x28xf32>
    %v742 = stablehlo.rsqrt %v741 : tensor<64x512x28x28xf32>
    %v743 = stablehlo.subtract %v735, %v738 : tensor<64x512x28x28xf32>
    %v744 = stablehlo.multiply %v743, %v742 : tensor<64x512x28x28xf32>
    %v745 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v746 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v747 = stablehlo.multiply %v744, %v745 : tensor<64x512x28x28xf32>
    %v748 = stablehlo.add %v747, %v746 : tensor<64x512x28x28xf32>
    %v749 = stablehlo.reshape %v748 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v751 = stablehlo.reshape %v628 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v752 = stablehlo.add %v750, %v751 : tensor<64x512x28x28xf32>
    %v753 = stablehlo.reshape %v752 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v756 = stablehlo.maximum %v754, %v755 : tensor<64x512x28x28xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v758 = stablehlo.reshape %v757 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v759 = stablehlo.convolution(%v758, %s2b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v760 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v761 = stablehlo.add %v759, %v760 : tensor<64x128x28x28xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v765 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v766 = stablehlo.reduce(%v763 init: %v764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v767 = stablehlo.divide %v766, %v765 : tensor<128xf32>
    %arsums2b2g1mu = "stablehlo.all_reduce"(%v767) ({
    ^bb0(%aras2b2g1mu: tensor<f32>, %arbs2b2g1mu: tensor<f32>):
      %aradds2b2g1mu = stablehlo.add %aras2b2g1mu, %arbs2b2g1mu : tensor<f32>
      stablehlo.return %aradds2b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1mu = stablehlo.divide %arsums2b2g1mu, %arns2b2g1mu : tensor<128xf32>
    %v768 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v770 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v771 = stablehlo.reduce(%v768 init: %v769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v772 = stablehlo.divide %v771, %v770 : tensor<128xf32>
    %v773 = stablehlo.broadcast_in_dim %v772, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v774 = stablehlo.subtract %v768, %v773 : tensor<64x128x28x28xf32>
    %v775 = stablehlo.multiply %v774, %v774 : tensor<64x128x28x28xf32>
    %v776 = stablehlo.reduce(%v775 init: %v769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v777 = stablehlo.divide %v776, %v770 : tensor<128xf32>
    %v778 = stablehlo.subtract %v772, %armeans2b2g1mu : tensor<128xf32>
    %v779 = stablehlo.multiply %v778, %v778 : tensor<128xf32>
    %v780 = stablehlo.add %v777, %v779 : tensor<128xf32>
    %arsums2b2g1var = "stablehlo.all_reduce"(%v780) ({
    ^bb0(%aras2b2g1var: tensor<f32>, %arbs2b2g1var: tensor<f32>):
      %aradds2b2g1var = stablehlo.add %aras2b2g1var, %arbs2b2g1var : tensor<f32>
      stablehlo.return %aradds2b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1var = stablehlo.divide %arsums2b2g1var, %arns2b2g1var : tensor<128xf32>
    %v781 = stablehlo.concatenate %armeans2b2g1mu, %armeans2b2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v782 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v783 = stablehlo.slice %v781 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v784 = stablehlo.slice %v781 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v785 = stablehlo.broadcast_in_dim %v783, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v786 = stablehlo.broadcast_in_dim %v784, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v787 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v788 = stablehlo.add %v786, %v787 : tensor<64x128x28x28xf32>
    %v789 = stablehlo.rsqrt %v788 : tensor<64x128x28x28xf32>
    %v790 = stablehlo.subtract %v782, %v785 : tensor<64x128x28x28xf32>
    %v791 = stablehlo.multiply %v790, %v789 : tensor<64x128x28x28xf32>
    %v792 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v793 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v794 = stablehlo.multiply %v791, %v792 : tensor<64x128x28x28xf32>
    %v795 = stablehlo.add %v794, %v793 : tensor<64x128x28x28xf32>
    %v796 = stablehlo.reshape %v795 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v797 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v798 = stablehlo.maximum %v796, %v797 : tensor<64x100352xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v800 = stablehlo.convolution(%v799, %s2b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v801 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v802 = stablehlo.add %v800, %v801 : tensor<64x128x28x28xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v805 = stablehlo.constant dense<0.0> : tensor<f32>
    %v806 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v807 = stablehlo.reduce(%v804 init: %v805) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v808 = stablehlo.divide %v807, %v806 : tensor<128xf32>
    %arsums2b2g2mu = "stablehlo.all_reduce"(%v808) ({
    ^bb0(%aras2b2g2mu: tensor<f32>, %arbs2b2g2mu: tensor<f32>):
      %aradds2b2g2mu = stablehlo.add %aras2b2g2mu, %arbs2b2g2mu : tensor<f32>
      stablehlo.return %aradds2b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2mu = stablehlo.divide %arsums2b2g2mu, %arns2b2g2mu : tensor<128xf32>
    %v809 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v811 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v812 = stablehlo.reduce(%v809 init: %v810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v813 = stablehlo.divide %v812, %v811 : tensor<128xf32>
    %v814 = stablehlo.broadcast_in_dim %v813, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v815 = stablehlo.subtract %v809, %v814 : tensor<64x128x28x28xf32>
    %v816 = stablehlo.multiply %v815, %v815 : tensor<64x128x28x28xf32>
    %v817 = stablehlo.reduce(%v816 init: %v810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v818 = stablehlo.divide %v817, %v811 : tensor<128xf32>
    %v819 = stablehlo.subtract %v813, %armeans2b2g2mu : tensor<128xf32>
    %v820 = stablehlo.multiply %v819, %v819 : tensor<128xf32>
    %v821 = stablehlo.add %v818, %v820 : tensor<128xf32>
    %arsums2b2g2var = "stablehlo.all_reduce"(%v821) ({
    ^bb0(%aras2b2g2var: tensor<f32>, %arbs2b2g2var: tensor<f32>):
      %aradds2b2g2var = stablehlo.add %aras2b2g2var, %arbs2b2g2var : tensor<f32>
      stablehlo.return %aradds2b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2var = stablehlo.divide %arsums2b2g2var, %arns2b2g2var : tensor<128xf32>
    %v822 = stablehlo.concatenate %armeans2b2g2mu, %armeans2b2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v823 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v824 = stablehlo.slice %v822 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v825 = stablehlo.slice %v822 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v826 = stablehlo.broadcast_in_dim %v824, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v827 = stablehlo.broadcast_in_dim %v825, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v828 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v829 = stablehlo.add %v827, %v828 : tensor<64x128x28x28xf32>
    %v830 = stablehlo.rsqrt %v829 : tensor<64x128x28x28xf32>
    %v831 = stablehlo.subtract %v823, %v826 : tensor<64x128x28x28xf32>
    %v832 = stablehlo.multiply %v831, %v830 : tensor<64x128x28x28xf32>
    %v833 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v834 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v835 = stablehlo.multiply %v832, %v833 : tensor<64x128x28x28xf32>
    %v836 = stablehlo.add %v835, %v834 : tensor<64x128x28x28xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v839 = stablehlo.maximum %v837, %v838 : tensor<64x100352xf32>
    %v840 = stablehlo.reshape %v839 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v841 = stablehlo.convolution(%v840, %s2b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v842 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v843 = stablehlo.add %v841, %v842 : tensor<64x512x28x28xf32>
    %v844 = stablehlo.reshape %v843 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v845 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v846 = stablehlo.constant dense<0.0> : tensor<f32>
    %v847 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v848 = stablehlo.reduce(%v845 init: %v846) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v849 = stablehlo.divide %v848, %v847 : tensor<512xf32>
    %arsums2b2g3mu = "stablehlo.all_reduce"(%v849) ({
    ^bb0(%aras2b2g3mu: tensor<f32>, %arbs2b2g3mu: tensor<f32>):
      %aradds2b2g3mu = stablehlo.add %aras2b2g3mu, %arbs2b2g3mu : tensor<f32>
      stablehlo.return %aradds2b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3mu = stablehlo.divide %arsums2b2g3mu, %arns2b2g3mu : tensor<512xf32>
    %v850 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v852 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v853 = stablehlo.reduce(%v850 init: %v851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v854 = stablehlo.divide %v853, %v852 : tensor<512xf32>
    %v855 = stablehlo.broadcast_in_dim %v854, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v856 = stablehlo.subtract %v850, %v855 : tensor<64x512x28x28xf32>
    %v857 = stablehlo.multiply %v856, %v856 : tensor<64x512x28x28xf32>
    %v858 = stablehlo.reduce(%v857 init: %v851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v859 = stablehlo.divide %v858, %v852 : tensor<512xf32>
    %v860 = stablehlo.subtract %v854, %armeans2b2g3mu : tensor<512xf32>
    %v861 = stablehlo.multiply %v860, %v860 : tensor<512xf32>
    %v862 = stablehlo.add %v859, %v861 : tensor<512xf32>
    %arsums2b2g3var = "stablehlo.all_reduce"(%v862) ({
    ^bb0(%aras2b2g3var: tensor<f32>, %arbs2b2g3var: tensor<f32>):
      %aradds2b2g3var = stablehlo.add %aras2b2g3var, %arbs2b2g3var : tensor<f32>
      stablehlo.return %aradds2b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3var = stablehlo.divide %arsums2b2g3var, %arns2b2g3var : tensor<512xf32>
    %v863 = stablehlo.concatenate %armeans2b2g3mu, %armeans2b2g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v864 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v865 = stablehlo.slice %v863 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v866 = stablehlo.slice %v863 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v867 = stablehlo.broadcast_in_dim %v865, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v868 = stablehlo.broadcast_in_dim %v866, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v869 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v870 = stablehlo.add %v868, %v869 : tensor<64x512x28x28xf32>
    %v871 = stablehlo.rsqrt %v870 : tensor<64x512x28x28xf32>
    %v872 = stablehlo.subtract %v864, %v867 : tensor<64x512x28x28xf32>
    %v873 = stablehlo.multiply %v872, %v871 : tensor<64x512x28x28xf32>
    %v874 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v875 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v876 = stablehlo.multiply %v873, %v874 : tensor<64x512x28x28xf32>
    %v877 = stablehlo.add %v876, %v875 : tensor<64x512x28x28xf32>
    %v878 = stablehlo.reshape %v877 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v880 = stablehlo.reshape %v757 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v881 = stablehlo.add %v879, %v880 : tensor<64x512x28x28xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v884 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v885 = stablehlo.maximum %v883, %v884 : tensor<64x512x28x28xf32>
    %v886 = stablehlo.reshape %v885 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v887 = stablehlo.reshape %v886 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v888 = stablehlo.convolution(%v887, %s2b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v889 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<64x128x28x28xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v894 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v895 = stablehlo.reduce(%v892 init: %v893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v896 = stablehlo.divide %v895, %v894 : tensor<128xf32>
    %arsums2b3g1mu = "stablehlo.all_reduce"(%v896) ({
    ^bb0(%aras2b3g1mu: tensor<f32>, %arbs2b3g1mu: tensor<f32>):
      %aradds2b3g1mu = stablehlo.add %aras2b3g1mu, %arbs2b3g1mu : tensor<f32>
      stablehlo.return %aradds2b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1mu = stablehlo.divide %arsums2b3g1mu, %arns2b3g1mu : tensor<128xf32>
    %v897 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v899 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v900 = stablehlo.reduce(%v897 init: %v898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v901 = stablehlo.divide %v900, %v899 : tensor<128xf32>
    %v902 = stablehlo.broadcast_in_dim %v901, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v903 = stablehlo.subtract %v897, %v902 : tensor<64x128x28x28xf32>
    %v904 = stablehlo.multiply %v903, %v903 : tensor<64x128x28x28xf32>
    %v905 = stablehlo.reduce(%v904 init: %v898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v906 = stablehlo.divide %v905, %v899 : tensor<128xf32>
    %v907 = stablehlo.subtract %v901, %armeans2b3g1mu : tensor<128xf32>
    %v908 = stablehlo.multiply %v907, %v907 : tensor<128xf32>
    %v909 = stablehlo.add %v906, %v908 : tensor<128xf32>
    %arsums2b3g1var = "stablehlo.all_reduce"(%v909) ({
    ^bb0(%aras2b3g1var: tensor<f32>, %arbs2b3g1var: tensor<f32>):
      %aradds2b3g1var = stablehlo.add %aras2b3g1var, %arbs2b3g1var : tensor<f32>
      stablehlo.return %aradds2b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1var = stablehlo.divide %arsums2b3g1var, %arns2b3g1var : tensor<128xf32>
    %v910 = stablehlo.concatenate %armeans2b3g1mu, %armeans2b3g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v911 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v912 = stablehlo.slice %v910 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v913 = stablehlo.slice %v910 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v914 = stablehlo.broadcast_in_dim %v912, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v915 = stablehlo.broadcast_in_dim %v913, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v916 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v917 = stablehlo.add %v915, %v916 : tensor<64x128x28x28xf32>
    %v918 = stablehlo.rsqrt %v917 : tensor<64x128x28x28xf32>
    %v919 = stablehlo.subtract %v911, %v914 : tensor<64x128x28x28xf32>
    %v920 = stablehlo.multiply %v919, %v918 : tensor<64x128x28x28xf32>
    %v921 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v922 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v923 = stablehlo.multiply %v920, %v921 : tensor<64x128x28x28xf32>
    %v924 = stablehlo.add %v923, %v922 : tensor<64x128x28x28xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v926 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v927 = stablehlo.maximum %v925, %v926 : tensor<64x100352xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v929 = stablehlo.convolution(%v928, %s2b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v930 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v931 = stablehlo.add %v929, %v930 : tensor<64x128x28x28xf32>
    %v932 = stablehlo.reshape %v931 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v934 = stablehlo.constant dense<0.0> : tensor<f32>
    %v935 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v936 = stablehlo.reduce(%v933 init: %v934) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v937 = stablehlo.divide %v936, %v935 : tensor<128xf32>
    %arsums2b3g2mu = "stablehlo.all_reduce"(%v937) ({
    ^bb0(%aras2b3g2mu: tensor<f32>, %arbs2b3g2mu: tensor<f32>):
      %aradds2b3g2mu = stablehlo.add %aras2b3g2mu, %arbs2b3g2mu : tensor<f32>
      stablehlo.return %aradds2b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2mu = stablehlo.divide %arsums2b3g2mu, %arns2b3g2mu : tensor<128xf32>
    %v938 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v940 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v941 = stablehlo.reduce(%v938 init: %v939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v942 = stablehlo.divide %v941, %v940 : tensor<128xf32>
    %v943 = stablehlo.broadcast_in_dim %v942, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v944 = stablehlo.subtract %v938, %v943 : tensor<64x128x28x28xf32>
    %v945 = stablehlo.multiply %v944, %v944 : tensor<64x128x28x28xf32>
    %v946 = stablehlo.reduce(%v945 init: %v939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v947 = stablehlo.divide %v946, %v940 : tensor<128xf32>
    %v948 = stablehlo.subtract %v942, %armeans2b3g2mu : tensor<128xf32>
    %v949 = stablehlo.multiply %v948, %v948 : tensor<128xf32>
    %v950 = stablehlo.add %v947, %v949 : tensor<128xf32>
    %arsums2b3g2var = "stablehlo.all_reduce"(%v950) ({
    ^bb0(%aras2b3g2var: tensor<f32>, %arbs2b3g2var: tensor<f32>):
      %aradds2b3g2var = stablehlo.add %aras2b3g2var, %arbs2b3g2var : tensor<f32>
      stablehlo.return %aradds2b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2var = stablehlo.divide %arsums2b3g2var, %arns2b3g2var : tensor<128xf32>
    %v951 = stablehlo.concatenate %armeans2b3g2mu, %armeans2b3g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v952 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v953 = stablehlo.slice %v951 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v954 = stablehlo.slice %v951 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v955 = stablehlo.broadcast_in_dim %v953, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v956 = stablehlo.broadcast_in_dim %v954, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v957 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<64x128x28x28xf32>
    %v959 = stablehlo.rsqrt %v958 : tensor<64x128x28x28xf32>
    %v960 = stablehlo.subtract %v952, %v955 : tensor<64x128x28x28xf32>
    %v961 = stablehlo.multiply %v960, %v959 : tensor<64x128x28x28xf32>
    %v962 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v963 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v964 = stablehlo.multiply %v961, %v962 : tensor<64x128x28x28xf32>
    %v965 = stablehlo.add %v964, %v963 : tensor<64x128x28x28xf32>
    %v966 = stablehlo.reshape %v965 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v967 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v968 = stablehlo.maximum %v966, %v967 : tensor<64x100352xf32>
    %v969 = stablehlo.reshape %v968 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v970 = stablehlo.convolution(%v969, %s2b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v971 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v972 = stablehlo.add %v970, %v971 : tensor<64x512x28x28xf32>
    %v973 = stablehlo.reshape %v972 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v976 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v977 = stablehlo.reduce(%v974 init: %v975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v978 = stablehlo.divide %v977, %v976 : tensor<512xf32>
    %arsums2b3g3mu = "stablehlo.all_reduce"(%v978) ({
    ^bb0(%aras2b3g3mu: tensor<f32>, %arbs2b3g3mu: tensor<f32>):
      %aradds2b3g3mu = stablehlo.add %aras2b3g3mu, %arbs2b3g3mu : tensor<f32>
      stablehlo.return %aradds2b3g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3mu = stablehlo.divide %arsums2b3g3mu, %arns2b3g3mu : tensor<512xf32>
    %v979 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v981 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v982 = stablehlo.reduce(%v979 init: %v980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v983 = stablehlo.divide %v982, %v981 : tensor<512xf32>
    %v984 = stablehlo.broadcast_in_dim %v983, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v985 = stablehlo.subtract %v979, %v984 : tensor<64x512x28x28xf32>
    %v986 = stablehlo.multiply %v985, %v985 : tensor<64x512x28x28xf32>
    %v987 = stablehlo.reduce(%v986 init: %v980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v988 = stablehlo.divide %v987, %v981 : tensor<512xf32>
    %v989 = stablehlo.subtract %v983, %armeans2b3g3mu : tensor<512xf32>
    %v990 = stablehlo.multiply %v989, %v989 : tensor<512xf32>
    %v991 = stablehlo.add %v988, %v990 : tensor<512xf32>
    %arsums2b3g3var = "stablehlo.all_reduce"(%v991) ({
    ^bb0(%aras2b3g3var: tensor<f32>, %arbs2b3g3var: tensor<f32>):
      %aradds2b3g3var = stablehlo.add %aras2b3g3var, %arbs2b3g3var : tensor<f32>
      stablehlo.return %aradds2b3g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3var = stablehlo.divide %arsums2b3g3var, %arns2b3g3var : tensor<512xf32>
    %v992 = stablehlo.concatenate %armeans2b3g3mu, %armeans2b3g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v993 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v994 = stablehlo.slice %v992 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v995 = stablehlo.slice %v992 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v996 = stablehlo.broadcast_in_dim %v994, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v997 = stablehlo.broadcast_in_dim %v995, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v998 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v999 = stablehlo.add %v997, %v998 : tensor<64x512x28x28xf32>
    %v1000 = stablehlo.rsqrt %v999 : tensor<64x512x28x28xf32>
    %v1001 = stablehlo.subtract %v993, %v996 : tensor<64x512x28x28xf32>
    %v1002 = stablehlo.multiply %v1001, %v1000 : tensor<64x512x28x28xf32>
    %v1003 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1004 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1005 = stablehlo.multiply %v1002, %v1003 : tensor<64x512x28x28xf32>
    %v1006 = stablehlo.add %v1005, %v1004 : tensor<64x512x28x28xf32>
    %v1007 = stablehlo.reshape %v1006 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1008 = stablehlo.reshape %v1007 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1009 = stablehlo.reshape %v886 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1010 = stablehlo.add %v1008, %v1009 : tensor<64x512x28x28xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1013 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v1014 = stablehlo.maximum %v1012, %v1013 : tensor<64x512x28x28xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1017 = stablehlo.convolution(%v1016, %s3b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x28x28xf32>
    %v1018 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1019 = stablehlo.add %v1017, %v1018 : tensor<64x256x28x28xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v1021 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1023 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v1024 = stablehlo.reduce(%v1021 init: %v1022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1025 = stablehlo.divide %v1024, %v1023 : tensor<256xf32>
    %arsums3b0g1mu = "stablehlo.all_reduce"(%v1025) ({
    ^bb0(%aras3b0g1mu: tensor<f32>, %arbs3b0g1mu: tensor<f32>):
      %aradds3b0g1mu = stablehlo.add %aras3b0g1mu, %arbs3b0g1mu : tensor<f32>
      stablehlo.return %aradds3b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1mu = stablehlo.divide %arsums3b0g1mu, %arns3b0g1mu : tensor<256xf32>
    %v1026 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1027 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1028 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v1029 = stablehlo.reduce(%v1026 init: %v1027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1030 = stablehlo.divide %v1029, %v1028 : tensor<256xf32>
    %v1031 = stablehlo.broadcast_in_dim %v1030, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1032 = stablehlo.subtract %v1026, %v1031 : tensor<64x256x28x28xf32>
    %v1033 = stablehlo.multiply %v1032, %v1032 : tensor<64x256x28x28xf32>
    %v1034 = stablehlo.reduce(%v1033 init: %v1027) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1035 = stablehlo.divide %v1034, %v1028 : tensor<256xf32>
    %v1036 = stablehlo.subtract %v1030, %armeans3b0g1mu : tensor<256xf32>
    %v1037 = stablehlo.multiply %v1036, %v1036 : tensor<256xf32>
    %v1038 = stablehlo.add %v1035, %v1037 : tensor<256xf32>
    %arsums3b0g1var = "stablehlo.all_reduce"(%v1038) ({
    ^bb0(%aras3b0g1var: tensor<f32>, %arbs3b0g1var: tensor<f32>):
      %aradds3b0g1var = stablehlo.add %aras3b0g1var, %arbs3b0g1var : tensor<f32>
      stablehlo.return %aradds3b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1var = stablehlo.divide %arsums3b0g1var, %arns3b0g1var : tensor<256xf32>
    %v1039 = stablehlo.concatenate %armeans3b0g1mu, %armeans3b0g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1040 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1041 = stablehlo.slice %v1039 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1042 = stablehlo.slice %v1039 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1043 = stablehlo.broadcast_in_dim %v1041, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1044 = stablehlo.broadcast_in_dim %v1042, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1045 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v1046 = stablehlo.add %v1044, %v1045 : tensor<64x256x28x28xf32>
    %v1047 = stablehlo.rsqrt %v1046 : tensor<64x256x28x28xf32>
    %v1048 = stablehlo.subtract %v1040, %v1043 : tensor<64x256x28x28xf32>
    %v1049 = stablehlo.multiply %v1048, %v1047 : tensor<64x256x28x28xf32>
    %v1050 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1051 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1052 = stablehlo.multiply %v1049, %v1050 : tensor<64x256x28x28xf32>
    %v1053 = stablehlo.add %v1052, %v1051 : tensor<64x256x28x28xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v1055 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1056 = stablehlo.maximum %v1054, %v1055 : tensor<64x200704xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1058 = stablehlo.convolution(%v1057, %s3b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1059 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1060 = stablehlo.add %v1058, %v1059 : tensor<64x256x14x14xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1064 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1065 = stablehlo.reduce(%v1062 init: %v1063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1066 = stablehlo.divide %v1065, %v1064 : tensor<256xf32>
    %arsums3b0g2mu = "stablehlo.all_reduce"(%v1066) ({
    ^bb0(%aras3b0g2mu: tensor<f32>, %arbs3b0g2mu: tensor<f32>):
      %aradds3b0g2mu = stablehlo.add %aras3b0g2mu, %arbs3b0g2mu : tensor<f32>
      stablehlo.return %aradds3b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2mu = stablehlo.divide %arsums3b0g2mu, %arns3b0g2mu : tensor<256xf32>
    %v1067 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1069 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1070 = stablehlo.reduce(%v1067 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1071 = stablehlo.divide %v1070, %v1069 : tensor<256xf32>
    %v1072 = stablehlo.broadcast_in_dim %v1071, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1073 = stablehlo.subtract %v1067, %v1072 : tensor<64x256x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<64x256x14x14xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1076 = stablehlo.divide %v1075, %v1069 : tensor<256xf32>
    %v1077 = stablehlo.subtract %v1071, %armeans3b0g2mu : tensor<256xf32>
    %v1078 = stablehlo.multiply %v1077, %v1077 : tensor<256xf32>
    %v1079 = stablehlo.add %v1076, %v1078 : tensor<256xf32>
    %arsums3b0g2var = "stablehlo.all_reduce"(%v1079) ({
    ^bb0(%aras3b0g2var: tensor<f32>, %arbs3b0g2var: tensor<f32>):
      %aradds3b0g2var = stablehlo.add %aras3b0g2var, %arbs3b0g2var : tensor<f32>
      stablehlo.return %aradds3b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2var = stablehlo.divide %arsums3b0g2var, %arns3b0g2var : tensor<256xf32>
    %v1080 = stablehlo.concatenate %armeans3b0g2mu, %armeans3b0g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1081 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1082 = stablehlo.slice %v1080 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1083 = stablehlo.slice %v1080 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1084 = stablehlo.broadcast_in_dim %v1082, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1085 = stablehlo.broadcast_in_dim %v1083, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1086 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1087 = stablehlo.add %v1085, %v1086 : tensor<64x256x14x14xf32>
    %v1088 = stablehlo.rsqrt %v1087 : tensor<64x256x14x14xf32>
    %v1089 = stablehlo.subtract %v1081, %v1084 : tensor<64x256x14x14xf32>
    %v1090 = stablehlo.multiply %v1089, %v1088 : tensor<64x256x14x14xf32>
    %v1091 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1092 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1093 = stablehlo.multiply %v1090, %v1091 : tensor<64x256x14x14xf32>
    %v1094 = stablehlo.add %v1093, %v1092 : tensor<64x256x14x14xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1096 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1097 = stablehlo.maximum %v1095, %v1096 : tensor<64x50176xf32>
    %v1098 = stablehlo.reshape %v1097 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1099 = stablehlo.convolution(%v1098, %s3b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1100 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1101 = stablehlo.add %v1099, %v1100 : tensor<64x1024x14x14xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1105 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1106 = stablehlo.reduce(%v1103 init: %v1104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1107 = stablehlo.divide %v1106, %v1105 : tensor<1024xf32>
    %arsums3b0g3mu = "stablehlo.all_reduce"(%v1107) ({
    ^bb0(%aras3b0g3mu: tensor<f32>, %arbs3b0g3mu: tensor<f32>):
      %aradds3b0g3mu = stablehlo.add %aras3b0g3mu, %arbs3b0g3mu : tensor<f32>
      stablehlo.return %aradds3b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3mu = stablehlo.divide %arsums3b0g3mu, %arns3b0g3mu : tensor<1024xf32>
    %v1108 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1110 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1111 = stablehlo.reduce(%v1108 init: %v1109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1112 = stablehlo.divide %v1111, %v1110 : tensor<1024xf32>
    %v1113 = stablehlo.broadcast_in_dim %v1112, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1114 = stablehlo.subtract %v1108, %v1113 : tensor<64x1024x14x14xf32>
    %v1115 = stablehlo.multiply %v1114, %v1114 : tensor<64x1024x14x14xf32>
    %v1116 = stablehlo.reduce(%v1115 init: %v1109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1117 = stablehlo.divide %v1116, %v1110 : tensor<1024xf32>
    %v1118 = stablehlo.subtract %v1112, %armeans3b0g3mu : tensor<1024xf32>
    %v1119 = stablehlo.multiply %v1118, %v1118 : tensor<1024xf32>
    %v1120 = stablehlo.add %v1117, %v1119 : tensor<1024xf32>
    %arsums3b0g3var = "stablehlo.all_reduce"(%v1120) ({
    ^bb0(%aras3b0g3var: tensor<f32>, %arbs3b0g3var: tensor<f32>):
      %aradds3b0g3var = stablehlo.add %aras3b0g3var, %arbs3b0g3var : tensor<f32>
      stablehlo.return %aradds3b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3var = stablehlo.divide %arsums3b0g3var, %arns3b0g3var : tensor<1024xf32>
    %v1121 = stablehlo.concatenate %armeans3b0g3mu, %armeans3b0g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1122 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1123 = stablehlo.slice %v1121 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1124 = stablehlo.slice %v1121 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1125 = stablehlo.broadcast_in_dim %v1123, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1126 = stablehlo.broadcast_in_dim %v1124, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1127 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1128 = stablehlo.add %v1126, %v1127 : tensor<64x1024x14x14xf32>
    %v1129 = stablehlo.rsqrt %v1128 : tensor<64x1024x14x14xf32>
    %v1130 = stablehlo.subtract %v1122, %v1125 : tensor<64x1024x14x14xf32>
    %v1131 = stablehlo.multiply %v1130, %v1129 : tensor<64x1024x14x14xf32>
    %v1132 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1133 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1134 = stablehlo.multiply %v1131, %v1132 : tensor<64x1024x14x14xf32>
    %v1135 = stablehlo.add %v1134, %v1133 : tensor<64x1024x14x14xf32>
    %v1136 = stablehlo.reshape %v1135 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1137 = stablehlo.reshape %v1015 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1138 = stablehlo.convolution(%v1137, %s3b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<1024x512x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1139 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1140 = stablehlo.add %v1138, %v1139 : tensor<64x1024x14x14xf32>
    %v1141 = stablehlo.reshape %v1140 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1142 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1144 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1145 = stablehlo.reduce(%v1142 init: %v1143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1146 = stablehlo.divide %v1145, %v1144 : tensor<1024xf32>
    %arsums3b0gpmu = "stablehlo.all_reduce"(%v1146) ({
    ^bb0(%aras3b0gpmu: tensor<f32>, %arbs3b0gpmu: tensor<f32>):
      %aradds3b0gpmu = stablehlo.add %aras3b0gpmu, %arbs3b0gpmu : tensor<f32>
      stablehlo.return %aradds3b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gpmu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gpmu = stablehlo.divide %arsums3b0gpmu, %arns3b0gpmu : tensor<1024xf32>
    %v1147 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1148 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1149 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1150 = stablehlo.reduce(%v1147 init: %v1148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1151 = stablehlo.divide %v1150, %v1149 : tensor<1024xf32>
    %v1152 = stablehlo.broadcast_in_dim %v1151, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1153 = stablehlo.subtract %v1147, %v1152 : tensor<64x1024x14x14xf32>
    %v1154 = stablehlo.multiply %v1153, %v1153 : tensor<64x1024x14x14xf32>
    %v1155 = stablehlo.reduce(%v1154 init: %v1148) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1156 = stablehlo.divide %v1155, %v1149 : tensor<1024xf32>
    %v1157 = stablehlo.subtract %v1151, %armeans3b0gpmu : tensor<1024xf32>
    %v1158 = stablehlo.multiply %v1157, %v1157 : tensor<1024xf32>
    %v1159 = stablehlo.add %v1156, %v1158 : tensor<1024xf32>
    %arsums3b0gpvar = "stablehlo.all_reduce"(%v1159) ({
    ^bb0(%aras3b0gpvar: tensor<f32>, %arbs3b0gpvar: tensor<f32>):
      %aradds3b0gpvar = stablehlo.add %aras3b0gpvar, %arbs3b0gpvar : tensor<f32>
      stablehlo.return %aradds3b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gpvar = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gpvar = stablehlo.divide %arsums3b0gpvar, %arns3b0gpvar : tensor<1024xf32>
    %v1160 = stablehlo.concatenate %armeans3b0gpmu, %armeans3b0gpvar, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1161 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1162 = stablehlo.slice %v1160 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1163 = stablehlo.slice %v1160 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1164 = stablehlo.broadcast_in_dim %v1162, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1163, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1166 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1167 = stablehlo.add %v1165, %v1166 : tensor<64x1024x14x14xf32>
    %v1168 = stablehlo.rsqrt %v1167 : tensor<64x1024x14x14xf32>
    %v1169 = stablehlo.subtract %v1161, %v1164 : tensor<64x1024x14x14xf32>
    %v1170 = stablehlo.multiply %v1169, %v1168 : tensor<64x1024x14x14xf32>
    %v1171 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1172 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1173 = stablehlo.multiply %v1170, %v1171 : tensor<64x1024x14x14xf32>
    %v1174 = stablehlo.add %v1173, %v1172 : tensor<64x1024x14x14xf32>
    %v1175 = stablehlo.reshape %v1174 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1176 = stablehlo.add %v1136, %v1175 : tensor<64x200704xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1178 = stablehlo.maximum %v1176, %v1177 : tensor<64x200704xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1180 = stablehlo.convolution(%v1179, %s3b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<64x256x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1187 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1188 = stablehlo.divide %v1187, %v1186 : tensor<256xf32>
    %arsums3b1g1mu = "stablehlo.all_reduce"(%v1188) ({
    ^bb0(%aras3b1g1mu: tensor<f32>, %arbs3b1g1mu: tensor<f32>):
      %aradds3b1g1mu = stablehlo.add %aras3b1g1mu, %arbs3b1g1mu : tensor<f32>
      stablehlo.return %aradds3b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1mu = stablehlo.divide %arsums3b1g1mu, %arns3b1g1mu : tensor<256xf32>
    %v1189 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1192 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1193 = stablehlo.divide %v1192, %v1191 : tensor<256xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1195 = stablehlo.subtract %v1189, %v1194 : tensor<64x256x14x14xf32>
    %v1196 = stablehlo.multiply %v1195, %v1195 : tensor<64x256x14x14xf32>
    %v1197 = stablehlo.reduce(%v1196 init: %v1190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1198 = stablehlo.divide %v1197, %v1191 : tensor<256xf32>
    %v1199 = stablehlo.subtract %v1193, %armeans3b1g1mu : tensor<256xf32>
    %v1200 = stablehlo.multiply %v1199, %v1199 : tensor<256xf32>
    %v1201 = stablehlo.add %v1198, %v1200 : tensor<256xf32>
    %arsums3b1g1var = "stablehlo.all_reduce"(%v1201) ({
    ^bb0(%aras3b1g1var: tensor<f32>, %arbs3b1g1var: tensor<f32>):
      %aradds3b1g1var = stablehlo.add %aras3b1g1var, %arbs3b1g1var : tensor<f32>
      stablehlo.return %aradds3b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1var = stablehlo.divide %arsums3b1g1var, %arns3b1g1var : tensor<256xf32>
    %v1202 = stablehlo.concatenate %armeans3b1g1mu, %armeans3b1g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1203 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1204 = stablehlo.slice %v1202 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1205 = stablehlo.slice %v1202 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1206 = stablehlo.broadcast_in_dim %v1204, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1205, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1208 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<64x256x14x14xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<64x256x14x14xf32>
    %v1211 = stablehlo.subtract %v1203, %v1206 : tensor<64x256x14x14xf32>
    %v1212 = stablehlo.multiply %v1211, %v1210 : tensor<64x256x14x14xf32>
    %v1213 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1214 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1215 = stablehlo.multiply %v1212, %v1213 : tensor<64x256x14x14xf32>
    %v1216 = stablehlo.add %v1215, %v1214 : tensor<64x256x14x14xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1218 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1219 = stablehlo.maximum %v1217, %v1218 : tensor<64x50176xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1221 = stablehlo.convolution(%v1220, %s3b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1222 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1223 = stablehlo.add %v1221, %v1222 : tensor<64x256x14x14xf32>
    %v1224 = stablehlo.reshape %v1223 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1227 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1228 = stablehlo.reduce(%v1225 init: %v1226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1229 = stablehlo.divide %v1228, %v1227 : tensor<256xf32>
    %arsums3b1g2mu = "stablehlo.all_reduce"(%v1229) ({
    ^bb0(%aras3b1g2mu: tensor<f32>, %arbs3b1g2mu: tensor<f32>):
      %aradds3b1g2mu = stablehlo.add %aras3b1g2mu, %arbs3b1g2mu : tensor<f32>
      stablehlo.return %aradds3b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2mu = stablehlo.divide %arsums3b1g2mu, %arns3b1g2mu : tensor<256xf32>
    %v1230 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1232 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1233 = stablehlo.reduce(%v1230 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1234 = stablehlo.divide %v1233, %v1232 : tensor<256xf32>
    %v1235 = stablehlo.broadcast_in_dim %v1234, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1236 = stablehlo.subtract %v1230, %v1235 : tensor<64x256x14x14xf32>
    %v1237 = stablehlo.multiply %v1236, %v1236 : tensor<64x256x14x14xf32>
    %v1238 = stablehlo.reduce(%v1237 init: %v1231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1239 = stablehlo.divide %v1238, %v1232 : tensor<256xf32>
    %v1240 = stablehlo.subtract %v1234, %armeans3b1g2mu : tensor<256xf32>
    %v1241 = stablehlo.multiply %v1240, %v1240 : tensor<256xf32>
    %v1242 = stablehlo.add %v1239, %v1241 : tensor<256xf32>
    %arsums3b1g2var = "stablehlo.all_reduce"(%v1242) ({
    ^bb0(%aras3b1g2var: tensor<f32>, %arbs3b1g2var: tensor<f32>):
      %aradds3b1g2var = stablehlo.add %aras3b1g2var, %arbs3b1g2var : tensor<f32>
      stablehlo.return %aradds3b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2var = stablehlo.divide %arsums3b1g2var, %arns3b1g2var : tensor<256xf32>
    %v1243 = stablehlo.concatenate %armeans3b1g2mu, %armeans3b1g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1244 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1245 = stablehlo.slice %v1243 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1246 = stablehlo.slice %v1243 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1247 = stablehlo.broadcast_in_dim %v1245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1249 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1250 = stablehlo.add %v1248, %v1249 : tensor<64x256x14x14xf32>
    %v1251 = stablehlo.rsqrt %v1250 : tensor<64x256x14x14xf32>
    %v1252 = stablehlo.subtract %v1244, %v1247 : tensor<64x256x14x14xf32>
    %v1253 = stablehlo.multiply %v1252, %v1251 : tensor<64x256x14x14xf32>
    %v1254 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1255 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1256 = stablehlo.multiply %v1253, %v1254 : tensor<64x256x14x14xf32>
    %v1257 = stablehlo.add %v1256, %v1255 : tensor<64x256x14x14xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1259 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1260 = stablehlo.maximum %v1258, %v1259 : tensor<64x50176xf32>
    %v1261 = stablehlo.reshape %v1260 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1262 = stablehlo.convolution(%v1261, %s3b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1263 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1264 = stablehlo.add %v1262, %v1263 : tensor<64x1024x14x14xf32>
    %v1265 = stablehlo.reshape %v1264 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1266 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1268 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1269 = stablehlo.reduce(%v1266 init: %v1267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1270 = stablehlo.divide %v1269, %v1268 : tensor<1024xf32>
    %arsums3b1g3mu = "stablehlo.all_reduce"(%v1270) ({
    ^bb0(%aras3b1g3mu: tensor<f32>, %arbs3b1g3mu: tensor<f32>):
      %aradds3b1g3mu = stablehlo.add %aras3b1g3mu, %arbs3b1g3mu : tensor<f32>
      stablehlo.return %aradds3b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3mu = stablehlo.divide %arsums3b1g3mu, %arns3b1g3mu : tensor<1024xf32>
    %v1271 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1273 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1274 = stablehlo.reduce(%v1271 init: %v1272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1275 = stablehlo.divide %v1274, %v1273 : tensor<1024xf32>
    %v1276 = stablehlo.broadcast_in_dim %v1275, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1277 = stablehlo.subtract %v1271, %v1276 : tensor<64x1024x14x14xf32>
    %v1278 = stablehlo.multiply %v1277, %v1277 : tensor<64x1024x14x14xf32>
    %v1279 = stablehlo.reduce(%v1278 init: %v1272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1280 = stablehlo.divide %v1279, %v1273 : tensor<1024xf32>
    %v1281 = stablehlo.subtract %v1275, %armeans3b1g3mu : tensor<1024xf32>
    %v1282 = stablehlo.multiply %v1281, %v1281 : tensor<1024xf32>
    %v1283 = stablehlo.add %v1280, %v1282 : tensor<1024xf32>
    %arsums3b1g3var = "stablehlo.all_reduce"(%v1283) ({
    ^bb0(%aras3b1g3var: tensor<f32>, %arbs3b1g3var: tensor<f32>):
      %aradds3b1g3var = stablehlo.add %aras3b1g3var, %arbs3b1g3var : tensor<f32>
      stablehlo.return %aradds3b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3var = stablehlo.divide %arsums3b1g3var, %arns3b1g3var : tensor<1024xf32>
    %v1284 = stablehlo.concatenate %armeans3b1g3mu, %armeans3b1g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1285 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1286 = stablehlo.slice %v1284 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1287 = stablehlo.slice %v1284 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1286, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1289 = stablehlo.broadcast_in_dim %v1287, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1290 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1291 = stablehlo.add %v1289, %v1290 : tensor<64x1024x14x14xf32>
    %v1292 = stablehlo.rsqrt %v1291 : tensor<64x1024x14x14xf32>
    %v1293 = stablehlo.subtract %v1285, %v1288 : tensor<64x1024x14x14xf32>
    %v1294 = stablehlo.multiply %v1293, %v1292 : tensor<64x1024x14x14xf32>
    %v1295 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1296 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1297 = stablehlo.multiply %v1294, %v1295 : tensor<64x1024x14x14xf32>
    %v1298 = stablehlo.add %v1297, %v1296 : tensor<64x1024x14x14xf32>
    %v1299 = stablehlo.reshape %v1298 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1300 = stablehlo.reshape %v1299 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1301 = stablehlo.reshape %v1178 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1302 = stablehlo.add %v1300, %v1301 : tensor<64x1024x14x14xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1305 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1306 = stablehlo.maximum %v1304, %v1305 : tensor<64x1024x14x14xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1309 = stablehlo.convolution(%v1308, %s3b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v1310 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1311 = stablehlo.add %v1309, %v1310 : tensor<64x256x14x14xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1313 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1315 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1316 = stablehlo.reduce(%v1313 init: %v1314) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1317 = stablehlo.divide %v1316, %v1315 : tensor<256xf32>
    %arsums3b2g1mu = "stablehlo.all_reduce"(%v1317) ({
    ^bb0(%aras3b2g1mu: tensor<f32>, %arbs3b2g1mu: tensor<f32>):
      %aradds3b2g1mu = stablehlo.add %aras3b2g1mu, %arbs3b2g1mu : tensor<f32>
      stablehlo.return %aradds3b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1mu = stablehlo.divide %arsums3b2g1mu, %arns3b2g1mu : tensor<256xf32>
    %v1318 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1320 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1321 = stablehlo.reduce(%v1318 init: %v1319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1322 = stablehlo.divide %v1321, %v1320 : tensor<256xf32>
    %v1323 = stablehlo.broadcast_in_dim %v1322, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1324 = stablehlo.subtract %v1318, %v1323 : tensor<64x256x14x14xf32>
    %v1325 = stablehlo.multiply %v1324, %v1324 : tensor<64x256x14x14xf32>
    %v1326 = stablehlo.reduce(%v1325 init: %v1319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1327 = stablehlo.divide %v1326, %v1320 : tensor<256xf32>
    %v1328 = stablehlo.subtract %v1322, %armeans3b2g1mu : tensor<256xf32>
    %v1329 = stablehlo.multiply %v1328, %v1328 : tensor<256xf32>
    %v1330 = stablehlo.add %v1327, %v1329 : tensor<256xf32>
    %arsums3b2g1var = "stablehlo.all_reduce"(%v1330) ({
    ^bb0(%aras3b2g1var: tensor<f32>, %arbs3b2g1var: tensor<f32>):
      %aradds3b2g1var = stablehlo.add %aras3b2g1var, %arbs3b2g1var : tensor<f32>
      stablehlo.return %aradds3b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1var = stablehlo.divide %arsums3b2g1var, %arns3b2g1var : tensor<256xf32>
    %v1331 = stablehlo.concatenate %armeans3b2g1mu, %armeans3b2g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1332 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1333 = stablehlo.slice %v1331 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1334 = stablehlo.slice %v1331 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1335 = stablehlo.broadcast_in_dim %v1333, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1336 = stablehlo.broadcast_in_dim %v1334, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1337 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1338 = stablehlo.add %v1336, %v1337 : tensor<64x256x14x14xf32>
    %v1339 = stablehlo.rsqrt %v1338 : tensor<64x256x14x14xf32>
    %v1340 = stablehlo.subtract %v1332, %v1335 : tensor<64x256x14x14xf32>
    %v1341 = stablehlo.multiply %v1340, %v1339 : tensor<64x256x14x14xf32>
    %v1342 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1343 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1344 = stablehlo.multiply %v1341, %v1342 : tensor<64x256x14x14xf32>
    %v1345 = stablehlo.add %v1344, %v1343 : tensor<64x256x14x14xf32>
    %v1346 = stablehlo.reshape %v1345 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1347 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1348 = stablehlo.maximum %v1346, %v1347 : tensor<64x50176xf32>
    %v1349 = stablehlo.reshape %v1348 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1350 = stablehlo.convolution(%v1349, %s3b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1351 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1352 = stablehlo.add %v1350, %v1351 : tensor<64x256x14x14xf32>
    %v1353 = stablehlo.reshape %v1352 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1354 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1356 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1357 = stablehlo.reduce(%v1354 init: %v1355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1358 = stablehlo.divide %v1357, %v1356 : tensor<256xf32>
    %arsums3b2g2mu = "stablehlo.all_reduce"(%v1358) ({
    ^bb0(%aras3b2g2mu: tensor<f32>, %arbs3b2g2mu: tensor<f32>):
      %aradds3b2g2mu = stablehlo.add %aras3b2g2mu, %arbs3b2g2mu : tensor<f32>
      stablehlo.return %aradds3b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2mu = stablehlo.divide %arsums3b2g2mu, %arns3b2g2mu : tensor<256xf32>
    %v1359 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1362 = stablehlo.reduce(%v1359 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1363 = stablehlo.divide %v1362, %v1361 : tensor<256xf32>
    %v1364 = stablehlo.broadcast_in_dim %v1363, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1365 = stablehlo.subtract %v1359, %v1364 : tensor<64x256x14x14xf32>
    %v1366 = stablehlo.multiply %v1365, %v1365 : tensor<64x256x14x14xf32>
    %v1367 = stablehlo.reduce(%v1366 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1368 = stablehlo.divide %v1367, %v1361 : tensor<256xf32>
    %v1369 = stablehlo.subtract %v1363, %armeans3b2g2mu : tensor<256xf32>
    %v1370 = stablehlo.multiply %v1369, %v1369 : tensor<256xf32>
    %v1371 = stablehlo.add %v1368, %v1370 : tensor<256xf32>
    %arsums3b2g2var = "stablehlo.all_reduce"(%v1371) ({
    ^bb0(%aras3b2g2var: tensor<f32>, %arbs3b2g2var: tensor<f32>):
      %aradds3b2g2var = stablehlo.add %aras3b2g2var, %arbs3b2g2var : tensor<f32>
      stablehlo.return %aradds3b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2var = stablehlo.divide %arsums3b2g2var, %arns3b2g2var : tensor<256xf32>
    %v1372 = stablehlo.concatenate %armeans3b2g2mu, %armeans3b2g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1373 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1374 = stablehlo.slice %v1372 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1375 = stablehlo.slice %v1372 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1376 = stablehlo.broadcast_in_dim %v1374, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1377 = stablehlo.broadcast_in_dim %v1375, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1378 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1379 = stablehlo.add %v1377, %v1378 : tensor<64x256x14x14xf32>
    %v1380 = stablehlo.rsqrt %v1379 : tensor<64x256x14x14xf32>
    %v1381 = stablehlo.subtract %v1373, %v1376 : tensor<64x256x14x14xf32>
    %v1382 = stablehlo.multiply %v1381, %v1380 : tensor<64x256x14x14xf32>
    %v1383 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1384 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1385 = stablehlo.multiply %v1382, %v1383 : tensor<64x256x14x14xf32>
    %v1386 = stablehlo.add %v1385, %v1384 : tensor<64x256x14x14xf32>
    %v1387 = stablehlo.reshape %v1386 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1388 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1389 = stablehlo.maximum %v1387, %v1388 : tensor<64x50176xf32>
    %v1390 = stablehlo.reshape %v1389 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1391 = stablehlo.convolution(%v1390, %s3b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1392 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1393 = stablehlo.add %v1391, %v1392 : tensor<64x1024x14x14xf32>
    %v1394 = stablehlo.reshape %v1393 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1397 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1398 = stablehlo.reduce(%v1395 init: %v1396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1399 = stablehlo.divide %v1398, %v1397 : tensor<1024xf32>
    %arsums3b2g3mu = "stablehlo.all_reduce"(%v1399) ({
    ^bb0(%aras3b2g3mu: tensor<f32>, %arbs3b2g3mu: tensor<f32>):
      %aradds3b2g3mu = stablehlo.add %aras3b2g3mu, %arbs3b2g3mu : tensor<f32>
      stablehlo.return %aradds3b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3mu = stablehlo.divide %arsums3b2g3mu, %arns3b2g3mu : tensor<1024xf32>
    %v1400 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1401 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1402 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1403 = stablehlo.reduce(%v1400 init: %v1401) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1404 = stablehlo.divide %v1403, %v1402 : tensor<1024xf32>
    %v1405 = stablehlo.broadcast_in_dim %v1404, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1406 = stablehlo.subtract %v1400, %v1405 : tensor<64x1024x14x14xf32>
    %v1407 = stablehlo.multiply %v1406, %v1406 : tensor<64x1024x14x14xf32>
    %v1408 = stablehlo.reduce(%v1407 init: %v1401) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1409 = stablehlo.divide %v1408, %v1402 : tensor<1024xf32>
    %v1410 = stablehlo.subtract %v1404, %armeans3b2g3mu : tensor<1024xf32>
    %v1411 = stablehlo.multiply %v1410, %v1410 : tensor<1024xf32>
    %v1412 = stablehlo.add %v1409, %v1411 : tensor<1024xf32>
    %arsums3b2g3var = "stablehlo.all_reduce"(%v1412) ({
    ^bb0(%aras3b2g3var: tensor<f32>, %arbs3b2g3var: tensor<f32>):
      %aradds3b2g3var = stablehlo.add %aras3b2g3var, %arbs3b2g3var : tensor<f32>
      stablehlo.return %aradds3b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3var = stablehlo.divide %arsums3b2g3var, %arns3b2g3var : tensor<1024xf32>
    %v1413 = stablehlo.concatenate %armeans3b2g3mu, %armeans3b2g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1414 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1415 = stablehlo.slice %v1413 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1416 = stablehlo.slice %v1413 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1417 = stablehlo.broadcast_in_dim %v1415, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1416, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1419 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1420 = stablehlo.add %v1418, %v1419 : tensor<64x1024x14x14xf32>
    %v1421 = stablehlo.rsqrt %v1420 : tensor<64x1024x14x14xf32>
    %v1422 = stablehlo.subtract %v1414, %v1417 : tensor<64x1024x14x14xf32>
    %v1423 = stablehlo.multiply %v1422, %v1421 : tensor<64x1024x14x14xf32>
    %v1424 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1425 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1426 = stablehlo.multiply %v1423, %v1424 : tensor<64x1024x14x14xf32>
    %v1427 = stablehlo.add %v1426, %v1425 : tensor<64x1024x14x14xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1430 = stablehlo.reshape %v1307 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1431 = stablehlo.add %v1429, %v1430 : tensor<64x1024x14x14xf32>
    %v1432 = stablehlo.reshape %v1431 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1433 = stablehlo.reshape %v1432 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1434 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1435 = stablehlo.maximum %v1433, %v1434 : tensor<64x1024x14x14xf32>
    %v1436 = stablehlo.reshape %v1435 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1438 = stablehlo.convolution(%v1437, %s3b3W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v1439 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1440 = stablehlo.add %v1438, %v1439 : tensor<64x256x14x14xf32>
    %v1441 = stablehlo.reshape %v1440 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1444 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1445 = stablehlo.reduce(%v1442 init: %v1443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1446 = stablehlo.divide %v1445, %v1444 : tensor<256xf32>
    %arsums3b3g1mu = "stablehlo.all_reduce"(%v1446) ({
    ^bb0(%aras3b3g1mu: tensor<f32>, %arbs3b3g1mu: tensor<f32>):
      %aradds3b3g1mu = stablehlo.add %aras3b3g1mu, %arbs3b3g1mu : tensor<f32>
      stablehlo.return %aradds3b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1mu = stablehlo.divide %arsums3b3g1mu, %arns3b3g1mu : tensor<256xf32>
    %v1447 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1448 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1449 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1450 = stablehlo.reduce(%v1447 init: %v1448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1451 = stablehlo.divide %v1450, %v1449 : tensor<256xf32>
    %v1452 = stablehlo.broadcast_in_dim %v1451, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1453 = stablehlo.subtract %v1447, %v1452 : tensor<64x256x14x14xf32>
    %v1454 = stablehlo.multiply %v1453, %v1453 : tensor<64x256x14x14xf32>
    %v1455 = stablehlo.reduce(%v1454 init: %v1448) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1456 = stablehlo.divide %v1455, %v1449 : tensor<256xf32>
    %v1457 = stablehlo.subtract %v1451, %armeans3b3g1mu : tensor<256xf32>
    %v1458 = stablehlo.multiply %v1457, %v1457 : tensor<256xf32>
    %v1459 = stablehlo.add %v1456, %v1458 : tensor<256xf32>
    %arsums3b3g1var = "stablehlo.all_reduce"(%v1459) ({
    ^bb0(%aras3b3g1var: tensor<f32>, %arbs3b3g1var: tensor<f32>):
      %aradds3b3g1var = stablehlo.add %aras3b3g1var, %arbs3b3g1var : tensor<f32>
      stablehlo.return %aradds3b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1var = stablehlo.divide %arsums3b3g1var, %arns3b3g1var : tensor<256xf32>
    %v1460 = stablehlo.concatenate %armeans3b3g1mu, %armeans3b3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1461 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1462 = stablehlo.slice %v1460 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1463 = stablehlo.slice %v1460 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1464 = stablehlo.broadcast_in_dim %v1462, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1463, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1466 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1467 = stablehlo.add %v1465, %v1466 : tensor<64x256x14x14xf32>
    %v1468 = stablehlo.rsqrt %v1467 : tensor<64x256x14x14xf32>
    %v1469 = stablehlo.subtract %v1461, %v1464 : tensor<64x256x14x14xf32>
    %v1470 = stablehlo.multiply %v1469, %v1468 : tensor<64x256x14x14xf32>
    %v1471 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1472 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1473 = stablehlo.multiply %v1470, %v1471 : tensor<64x256x14x14xf32>
    %v1474 = stablehlo.add %v1473, %v1472 : tensor<64x256x14x14xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1476 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1477 = stablehlo.maximum %v1475, %v1476 : tensor<64x50176xf32>
    %v1478 = stablehlo.reshape %v1477 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1479 = stablehlo.convolution(%v1478, %s3b3W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1480 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1481 = stablehlo.add %v1479, %v1480 : tensor<64x256x14x14xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1486 = stablehlo.reduce(%v1483 init: %v1484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1487 = stablehlo.divide %v1486, %v1485 : tensor<256xf32>
    %arsums3b3g2mu = "stablehlo.all_reduce"(%v1487) ({
    ^bb0(%aras3b3g2mu: tensor<f32>, %arbs3b3g2mu: tensor<f32>):
      %aradds3b3g2mu = stablehlo.add %aras3b3g2mu, %arbs3b3g2mu : tensor<f32>
      stablehlo.return %aradds3b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2mu = stablehlo.divide %arsums3b3g2mu, %arns3b3g2mu : tensor<256xf32>
    %v1488 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1491 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1492 = stablehlo.divide %v1491, %v1490 : tensor<256xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1494 = stablehlo.subtract %v1488, %v1493 : tensor<64x256x14x14xf32>
    %v1495 = stablehlo.multiply %v1494, %v1494 : tensor<64x256x14x14xf32>
    %v1496 = stablehlo.reduce(%v1495 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1497 = stablehlo.divide %v1496, %v1490 : tensor<256xf32>
    %v1498 = stablehlo.subtract %v1492, %armeans3b3g2mu : tensor<256xf32>
    %v1499 = stablehlo.multiply %v1498, %v1498 : tensor<256xf32>
    %v1500 = stablehlo.add %v1497, %v1499 : tensor<256xf32>
    %arsums3b3g2var = "stablehlo.all_reduce"(%v1500) ({
    ^bb0(%aras3b3g2var: tensor<f32>, %arbs3b3g2var: tensor<f32>):
      %aradds3b3g2var = stablehlo.add %aras3b3g2var, %arbs3b3g2var : tensor<f32>
      stablehlo.return %aradds3b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2var = stablehlo.divide %arsums3b3g2var, %arns3b3g2var : tensor<256xf32>
    %v1501 = stablehlo.concatenate %armeans3b3g2mu, %armeans3b3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1502 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1503 = stablehlo.slice %v1501 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1504 = stablehlo.slice %v1501 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1503, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1506 = stablehlo.broadcast_in_dim %v1504, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1507 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1508 = stablehlo.add %v1506, %v1507 : tensor<64x256x14x14xf32>
    %v1509 = stablehlo.rsqrt %v1508 : tensor<64x256x14x14xf32>
    %v1510 = stablehlo.subtract %v1502, %v1505 : tensor<64x256x14x14xf32>
    %v1511 = stablehlo.multiply %v1510, %v1509 : tensor<64x256x14x14xf32>
    %v1512 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1513 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1514 = stablehlo.multiply %v1511, %v1512 : tensor<64x256x14x14xf32>
    %v1515 = stablehlo.add %v1514, %v1513 : tensor<64x256x14x14xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1517 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1518 = stablehlo.maximum %v1516, %v1517 : tensor<64x50176xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1520 = stablehlo.convolution(%v1519, %s3b3W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1521 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<64x1024x14x14xf32>
    %v1523 = stablehlo.reshape %v1522 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1524 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1525 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1526 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1527 = stablehlo.reduce(%v1524 init: %v1525) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1528 = stablehlo.divide %v1527, %v1526 : tensor<1024xf32>
    %arsums3b3g3mu = "stablehlo.all_reduce"(%v1528) ({
    ^bb0(%aras3b3g3mu: tensor<f32>, %arbs3b3g3mu: tensor<f32>):
      %aradds3b3g3mu = stablehlo.add %aras3b3g3mu, %arbs3b3g3mu : tensor<f32>
      stablehlo.return %aradds3b3g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3mu = stablehlo.divide %arsums3b3g3mu, %arns3b3g3mu : tensor<1024xf32>
    %v1529 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1531 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1532 = stablehlo.reduce(%v1529 init: %v1530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1533 = stablehlo.divide %v1532, %v1531 : tensor<1024xf32>
    %v1534 = stablehlo.broadcast_in_dim %v1533, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1535 = stablehlo.subtract %v1529, %v1534 : tensor<64x1024x14x14xf32>
    %v1536 = stablehlo.multiply %v1535, %v1535 : tensor<64x1024x14x14xf32>
    %v1537 = stablehlo.reduce(%v1536 init: %v1530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1538 = stablehlo.divide %v1537, %v1531 : tensor<1024xf32>
    %v1539 = stablehlo.subtract %v1533, %armeans3b3g3mu : tensor<1024xf32>
    %v1540 = stablehlo.multiply %v1539, %v1539 : tensor<1024xf32>
    %v1541 = stablehlo.add %v1538, %v1540 : tensor<1024xf32>
    %arsums3b3g3var = "stablehlo.all_reduce"(%v1541) ({
    ^bb0(%aras3b3g3var: tensor<f32>, %arbs3b3g3var: tensor<f32>):
      %aradds3b3g3var = stablehlo.add %aras3b3g3var, %arbs3b3g3var : tensor<f32>
      stablehlo.return %aradds3b3g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3var = stablehlo.divide %arsums3b3g3var, %arns3b3g3var : tensor<1024xf32>
    %v1542 = stablehlo.concatenate %armeans3b3g3mu, %armeans3b3g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1543 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1544 = stablehlo.slice %v1542 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1545 = stablehlo.slice %v1542 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1546 = stablehlo.broadcast_in_dim %v1544, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1547 = stablehlo.broadcast_in_dim %v1545, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1548 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1549 = stablehlo.add %v1547, %v1548 : tensor<64x1024x14x14xf32>
    %v1550 = stablehlo.rsqrt %v1549 : tensor<64x1024x14x14xf32>
    %v1551 = stablehlo.subtract %v1543, %v1546 : tensor<64x1024x14x14xf32>
    %v1552 = stablehlo.multiply %v1551, %v1550 : tensor<64x1024x14x14xf32>
    %v1553 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1554 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1555 = stablehlo.multiply %v1552, %v1553 : tensor<64x1024x14x14xf32>
    %v1556 = stablehlo.add %v1555, %v1554 : tensor<64x1024x14x14xf32>
    %v1557 = stablehlo.reshape %v1556 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1558 = stablehlo.reshape %v1557 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1559 = stablehlo.reshape %v1436 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1560 = stablehlo.add %v1558, %v1559 : tensor<64x1024x14x14xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1562 = stablehlo.reshape %v1561 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1563 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1564 = stablehlo.maximum %v1562, %v1563 : tensor<64x1024x14x14xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1566 = stablehlo.reshape %v1565 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1567 = stablehlo.convolution(%v1566, %s3b4W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v1568 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1569 = stablehlo.add %v1567, %v1568 : tensor<64x256x14x14xf32>
    %v1570 = stablehlo.reshape %v1569 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1573 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1574 = stablehlo.reduce(%v1571 init: %v1572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1575 = stablehlo.divide %v1574, %v1573 : tensor<256xf32>
    %arsums3b4g1mu = "stablehlo.all_reduce"(%v1575) ({
    ^bb0(%aras3b4g1mu: tensor<f32>, %arbs3b4g1mu: tensor<f32>):
      %aradds3b4g1mu = stablehlo.add %aras3b4g1mu, %arbs3b4g1mu : tensor<f32>
      stablehlo.return %aradds3b4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1mu = stablehlo.divide %arsums3b4g1mu, %arns3b4g1mu : tensor<256xf32>
    %v1576 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1577 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1578 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1579 = stablehlo.reduce(%v1576 init: %v1577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1580 = stablehlo.divide %v1579, %v1578 : tensor<256xf32>
    %v1581 = stablehlo.broadcast_in_dim %v1580, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1582 = stablehlo.subtract %v1576, %v1581 : tensor<64x256x14x14xf32>
    %v1583 = stablehlo.multiply %v1582, %v1582 : tensor<64x256x14x14xf32>
    %v1584 = stablehlo.reduce(%v1583 init: %v1577) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1585 = stablehlo.divide %v1584, %v1578 : tensor<256xf32>
    %v1586 = stablehlo.subtract %v1580, %armeans3b4g1mu : tensor<256xf32>
    %v1587 = stablehlo.multiply %v1586, %v1586 : tensor<256xf32>
    %v1588 = stablehlo.add %v1585, %v1587 : tensor<256xf32>
    %arsums3b4g1var = "stablehlo.all_reduce"(%v1588) ({
    ^bb0(%aras3b4g1var: tensor<f32>, %arbs3b4g1var: tensor<f32>):
      %aradds3b4g1var = stablehlo.add %aras3b4g1var, %arbs3b4g1var : tensor<f32>
      stablehlo.return %aradds3b4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1var = stablehlo.divide %arsums3b4g1var, %arns3b4g1var : tensor<256xf32>
    %v1589 = stablehlo.concatenate %armeans3b4g1mu, %armeans3b4g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1590 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1591 = stablehlo.slice %v1589 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1592 = stablehlo.slice %v1589 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1591, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1594 = stablehlo.broadcast_in_dim %v1592, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1595 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1596 = stablehlo.add %v1594, %v1595 : tensor<64x256x14x14xf32>
    %v1597 = stablehlo.rsqrt %v1596 : tensor<64x256x14x14xf32>
    %v1598 = stablehlo.subtract %v1590, %v1593 : tensor<64x256x14x14xf32>
    %v1599 = stablehlo.multiply %v1598, %v1597 : tensor<64x256x14x14xf32>
    %v1600 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1601 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1602 = stablehlo.multiply %v1599, %v1600 : tensor<64x256x14x14xf32>
    %v1603 = stablehlo.add %v1602, %v1601 : tensor<64x256x14x14xf32>
    %v1604 = stablehlo.reshape %v1603 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1605 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1606 = stablehlo.maximum %v1604, %v1605 : tensor<64x50176xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1608 = stablehlo.convolution(%v1607, %s3b4W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1609 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1610 = stablehlo.add %v1608, %v1609 : tensor<64x256x14x14xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1614 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1615 = stablehlo.reduce(%v1612 init: %v1613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1616 = stablehlo.divide %v1615, %v1614 : tensor<256xf32>
    %arsums3b4g2mu = "stablehlo.all_reduce"(%v1616) ({
    ^bb0(%aras3b4g2mu: tensor<f32>, %arbs3b4g2mu: tensor<f32>):
      %aradds3b4g2mu = stablehlo.add %aras3b4g2mu, %arbs3b4g2mu : tensor<f32>
      stablehlo.return %aradds3b4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2mu = stablehlo.divide %arsums3b4g2mu, %arns3b4g2mu : tensor<256xf32>
    %v1617 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1619 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1620 = stablehlo.reduce(%v1617 init: %v1618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1621 = stablehlo.divide %v1620, %v1619 : tensor<256xf32>
    %v1622 = stablehlo.broadcast_in_dim %v1621, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1623 = stablehlo.subtract %v1617, %v1622 : tensor<64x256x14x14xf32>
    %v1624 = stablehlo.multiply %v1623, %v1623 : tensor<64x256x14x14xf32>
    %v1625 = stablehlo.reduce(%v1624 init: %v1618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1626 = stablehlo.divide %v1625, %v1619 : tensor<256xf32>
    %v1627 = stablehlo.subtract %v1621, %armeans3b4g2mu : tensor<256xf32>
    %v1628 = stablehlo.multiply %v1627, %v1627 : tensor<256xf32>
    %v1629 = stablehlo.add %v1626, %v1628 : tensor<256xf32>
    %arsums3b4g2var = "stablehlo.all_reduce"(%v1629) ({
    ^bb0(%aras3b4g2var: tensor<f32>, %arbs3b4g2var: tensor<f32>):
      %aradds3b4g2var = stablehlo.add %aras3b4g2var, %arbs3b4g2var : tensor<f32>
      stablehlo.return %aradds3b4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2var = stablehlo.divide %arsums3b4g2var, %arns3b4g2var : tensor<256xf32>
    %v1630 = stablehlo.concatenate %armeans3b4g2mu, %armeans3b4g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1631 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1632 = stablehlo.slice %v1630 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1633 = stablehlo.slice %v1630 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1634 = stablehlo.broadcast_in_dim %v1632, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1635 = stablehlo.broadcast_in_dim %v1633, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1636 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1637 = stablehlo.add %v1635, %v1636 : tensor<64x256x14x14xf32>
    %v1638 = stablehlo.rsqrt %v1637 : tensor<64x256x14x14xf32>
    %v1639 = stablehlo.subtract %v1631, %v1634 : tensor<64x256x14x14xf32>
    %v1640 = stablehlo.multiply %v1639, %v1638 : tensor<64x256x14x14xf32>
    %v1641 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1642 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1640, %v1641 : tensor<64x256x14x14xf32>
    %v1644 = stablehlo.add %v1643, %v1642 : tensor<64x256x14x14xf32>
    %v1645 = stablehlo.reshape %v1644 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1646 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1647 = stablehlo.maximum %v1645, %v1646 : tensor<64x50176xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1649 = stablehlo.convolution(%v1648, %s3b4W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1650 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1651 = stablehlo.add %v1649, %v1650 : tensor<64x1024x14x14xf32>
    %v1652 = stablehlo.reshape %v1651 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1653 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1654 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1655 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1656 = stablehlo.reduce(%v1653 init: %v1654) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1657 = stablehlo.divide %v1656, %v1655 : tensor<1024xf32>
    %arsums3b4g3mu = "stablehlo.all_reduce"(%v1657) ({
    ^bb0(%aras3b4g3mu: tensor<f32>, %arbs3b4g3mu: tensor<f32>):
      %aradds3b4g3mu = stablehlo.add %aras3b4g3mu, %arbs3b4g3mu : tensor<f32>
      stablehlo.return %aradds3b4g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3mu = stablehlo.divide %arsums3b4g3mu, %arns3b4g3mu : tensor<1024xf32>
    %v1658 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1660 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1661 = stablehlo.reduce(%v1658 init: %v1659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1662 = stablehlo.divide %v1661, %v1660 : tensor<1024xf32>
    %v1663 = stablehlo.broadcast_in_dim %v1662, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1664 = stablehlo.subtract %v1658, %v1663 : tensor<64x1024x14x14xf32>
    %v1665 = stablehlo.multiply %v1664, %v1664 : tensor<64x1024x14x14xf32>
    %v1666 = stablehlo.reduce(%v1665 init: %v1659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1667 = stablehlo.divide %v1666, %v1660 : tensor<1024xf32>
    %v1668 = stablehlo.subtract %v1662, %armeans3b4g3mu : tensor<1024xf32>
    %v1669 = stablehlo.multiply %v1668, %v1668 : tensor<1024xf32>
    %v1670 = stablehlo.add %v1667, %v1669 : tensor<1024xf32>
    %arsums3b4g3var = "stablehlo.all_reduce"(%v1670) ({
    ^bb0(%aras3b4g3var: tensor<f32>, %arbs3b4g3var: tensor<f32>):
      %aradds3b4g3var = stablehlo.add %aras3b4g3var, %arbs3b4g3var : tensor<f32>
      stablehlo.return %aradds3b4g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3var = stablehlo.divide %arsums3b4g3var, %arns3b4g3var : tensor<1024xf32>
    %v1671 = stablehlo.concatenate %armeans3b4g3mu, %armeans3b4g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1672 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1673 = stablehlo.slice %v1671 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1674 = stablehlo.slice %v1671 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1673, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1676 = stablehlo.broadcast_in_dim %v1674, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1677 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1678 = stablehlo.add %v1676, %v1677 : tensor<64x1024x14x14xf32>
    %v1679 = stablehlo.rsqrt %v1678 : tensor<64x1024x14x14xf32>
    %v1680 = stablehlo.subtract %v1672, %v1675 : tensor<64x1024x14x14xf32>
    %v1681 = stablehlo.multiply %v1680, %v1679 : tensor<64x1024x14x14xf32>
    %v1682 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1683 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1684 = stablehlo.multiply %v1681, %v1682 : tensor<64x1024x14x14xf32>
    %v1685 = stablehlo.add %v1684, %v1683 : tensor<64x1024x14x14xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1688 = stablehlo.reshape %v1565 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1689 = stablehlo.add %v1687, %v1688 : tensor<64x1024x14x14xf32>
    %v1690 = stablehlo.reshape %v1689 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1691 = stablehlo.reshape %v1690 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1692 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1693 = stablehlo.maximum %v1691, %v1692 : tensor<64x1024x14x14xf32>
    %v1694 = stablehlo.reshape %v1693 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1696 = stablehlo.convolution(%v1695, %s3b5W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v1697 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1698 = stablehlo.add %v1696, %v1697 : tensor<64x256x14x14xf32>
    %v1699 = stablehlo.reshape %v1698 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1700 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1702 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1703 = stablehlo.reduce(%v1700 init: %v1701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1704 = stablehlo.divide %v1703, %v1702 : tensor<256xf32>
    %arsums3b5g1mu = "stablehlo.all_reduce"(%v1704) ({
    ^bb0(%aras3b5g1mu: tensor<f32>, %arbs3b5g1mu: tensor<f32>):
      %aradds3b5g1mu = stablehlo.add %aras3b5g1mu, %arbs3b5g1mu : tensor<f32>
      stablehlo.return %aradds3b5g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1mu = stablehlo.divide %arsums3b5g1mu, %arns3b5g1mu : tensor<256xf32>
    %v1705 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1707 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1708 = stablehlo.reduce(%v1705 init: %v1706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1709 = stablehlo.divide %v1708, %v1707 : tensor<256xf32>
    %v1710 = stablehlo.broadcast_in_dim %v1709, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1711 = stablehlo.subtract %v1705, %v1710 : tensor<64x256x14x14xf32>
    %v1712 = stablehlo.multiply %v1711, %v1711 : tensor<64x256x14x14xf32>
    %v1713 = stablehlo.reduce(%v1712 init: %v1706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1714 = stablehlo.divide %v1713, %v1707 : tensor<256xf32>
    %v1715 = stablehlo.subtract %v1709, %armeans3b5g1mu : tensor<256xf32>
    %v1716 = stablehlo.multiply %v1715, %v1715 : tensor<256xf32>
    %v1717 = stablehlo.add %v1714, %v1716 : tensor<256xf32>
    %arsums3b5g1var = "stablehlo.all_reduce"(%v1717) ({
    ^bb0(%aras3b5g1var: tensor<f32>, %arbs3b5g1var: tensor<f32>):
      %aradds3b5g1var = stablehlo.add %aras3b5g1var, %arbs3b5g1var : tensor<f32>
      stablehlo.return %aradds3b5g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1var = stablehlo.divide %arsums3b5g1var, %arns3b5g1var : tensor<256xf32>
    %v1718 = stablehlo.concatenate %armeans3b5g1mu, %armeans3b5g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1719 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1720 = stablehlo.slice %v1718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1721 = stablehlo.slice %v1718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1722 = stablehlo.broadcast_in_dim %v1720, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1723 = stablehlo.broadcast_in_dim %v1721, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1724 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1725 = stablehlo.add %v1723, %v1724 : tensor<64x256x14x14xf32>
    %v1726 = stablehlo.rsqrt %v1725 : tensor<64x256x14x14xf32>
    %v1727 = stablehlo.subtract %v1719, %v1722 : tensor<64x256x14x14xf32>
    %v1728 = stablehlo.multiply %v1727, %v1726 : tensor<64x256x14x14xf32>
    %v1729 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1730 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1731 = stablehlo.multiply %v1728, %v1729 : tensor<64x256x14x14xf32>
    %v1732 = stablehlo.add %v1731, %v1730 : tensor<64x256x14x14xf32>
    %v1733 = stablehlo.reshape %v1732 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1734 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1735 = stablehlo.maximum %v1733, %v1734 : tensor<64x50176xf32>
    %v1736 = stablehlo.reshape %v1735 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1737 = stablehlo.convolution(%v1736, %s3b5W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v1738 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1739 = stablehlo.add %v1737, %v1738 : tensor<64x256x14x14xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1741 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1743 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1744 = stablehlo.reduce(%v1741 init: %v1742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1745 = stablehlo.divide %v1744, %v1743 : tensor<256xf32>
    %arsums3b5g2mu = "stablehlo.all_reduce"(%v1745) ({
    ^bb0(%aras3b5g2mu: tensor<f32>, %arbs3b5g2mu: tensor<f32>):
      %aradds3b5g2mu = stablehlo.add %aras3b5g2mu, %arbs3b5g2mu : tensor<f32>
      stablehlo.return %aradds3b5g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2mu = stablehlo.divide %arsums3b5g2mu, %arns3b5g2mu : tensor<256xf32>
    %v1746 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1747 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1748 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1749 = stablehlo.reduce(%v1746 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1750 = stablehlo.divide %v1749, %v1748 : tensor<256xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1752 = stablehlo.subtract %v1746, %v1751 : tensor<64x256x14x14xf32>
    %v1753 = stablehlo.multiply %v1752, %v1752 : tensor<64x256x14x14xf32>
    %v1754 = stablehlo.reduce(%v1753 init: %v1747) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1755 = stablehlo.divide %v1754, %v1748 : tensor<256xf32>
    %v1756 = stablehlo.subtract %v1750, %armeans3b5g2mu : tensor<256xf32>
    %v1757 = stablehlo.multiply %v1756, %v1756 : tensor<256xf32>
    %v1758 = stablehlo.add %v1755, %v1757 : tensor<256xf32>
    %arsums3b5g2var = "stablehlo.all_reduce"(%v1758) ({
    ^bb0(%aras3b5g2var: tensor<f32>, %arbs3b5g2var: tensor<f32>):
      %aradds3b5g2var = stablehlo.add %aras3b5g2var, %arbs3b5g2var : tensor<f32>
      stablehlo.return %aradds3b5g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2var = stablehlo.divide %arsums3b5g2var, %arns3b5g2var : tensor<256xf32>
    %v1759 = stablehlo.concatenate %armeans3b5g2mu, %armeans3b5g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1760 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1761 = stablehlo.slice %v1759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1762 = stablehlo.slice %v1759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1761, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1764 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1765 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1766 = stablehlo.add %v1764, %v1765 : tensor<64x256x14x14xf32>
    %v1767 = stablehlo.rsqrt %v1766 : tensor<64x256x14x14xf32>
    %v1768 = stablehlo.subtract %v1760, %v1763 : tensor<64x256x14x14xf32>
    %v1769 = stablehlo.multiply %v1768, %v1767 : tensor<64x256x14x14xf32>
    %v1770 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1771 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1772 = stablehlo.multiply %v1769, %v1770 : tensor<64x256x14x14xf32>
    %v1773 = stablehlo.add %v1772, %v1771 : tensor<64x256x14x14xf32>
    %v1774 = stablehlo.reshape %v1773 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1776 = stablehlo.maximum %v1774, %v1775 : tensor<64x50176xf32>
    %v1777 = stablehlo.reshape %v1776 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1778 = stablehlo.convolution(%v1777, %s3b5W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v1779 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1780 = stablehlo.add %v1778, %v1779 : tensor<64x1024x14x14xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1782 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1784 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1785 = stablehlo.reduce(%v1782 init: %v1783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1786 = stablehlo.divide %v1785, %v1784 : tensor<1024xf32>
    %arsums3b5g3mu = "stablehlo.all_reduce"(%v1786) ({
    ^bb0(%aras3b5g3mu: tensor<f32>, %arbs3b5g3mu: tensor<f32>):
      %aradds3b5g3mu = stablehlo.add %aras3b5g3mu, %arbs3b5g3mu : tensor<f32>
      stablehlo.return %aradds3b5g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3mu = stablehlo.divide %arsums3b5g3mu, %arns3b5g3mu : tensor<1024xf32>
    %v1787 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1789 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1790 = stablehlo.reduce(%v1787 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1791 = stablehlo.divide %v1790, %v1789 : tensor<1024xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1793 = stablehlo.subtract %v1787, %v1792 : tensor<64x1024x14x14xf32>
    %v1794 = stablehlo.multiply %v1793, %v1793 : tensor<64x1024x14x14xf32>
    %v1795 = stablehlo.reduce(%v1794 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1796 = stablehlo.divide %v1795, %v1789 : tensor<1024xf32>
    %v1797 = stablehlo.subtract %v1791, %armeans3b5g3mu : tensor<1024xf32>
    %v1798 = stablehlo.multiply %v1797, %v1797 : tensor<1024xf32>
    %v1799 = stablehlo.add %v1796, %v1798 : tensor<1024xf32>
    %arsums3b5g3var = "stablehlo.all_reduce"(%v1799) ({
    ^bb0(%aras3b5g3var: tensor<f32>, %arbs3b5g3var: tensor<f32>):
      %aradds3b5g3var = stablehlo.add %aras3b5g3var, %arbs3b5g3var : tensor<f32>
      stablehlo.return %aradds3b5g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3var = stablehlo.divide %arsums3b5g3var, %arns3b5g3var : tensor<1024xf32>
    %v1800 = stablehlo.concatenate %armeans3b5g3mu, %armeans3b5g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1801 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1802 = stablehlo.slice %v1800 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1803 = stablehlo.slice %v1800 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1804 = stablehlo.broadcast_in_dim %v1802, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1805 = stablehlo.broadcast_in_dim %v1803, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1806 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1807 = stablehlo.add %v1805, %v1806 : tensor<64x1024x14x14xf32>
    %v1808 = stablehlo.rsqrt %v1807 : tensor<64x1024x14x14xf32>
    %v1809 = stablehlo.subtract %v1801, %v1804 : tensor<64x1024x14x14xf32>
    %v1810 = stablehlo.multiply %v1809, %v1808 : tensor<64x1024x14x14xf32>
    %v1811 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1812 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1813 = stablehlo.multiply %v1810, %v1811 : tensor<64x1024x14x14xf32>
    %v1814 = stablehlo.add %v1813, %v1812 : tensor<64x1024x14x14xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1817 = stablehlo.reshape %v1694 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1818 = stablehlo.add %v1816, %v1817 : tensor<64x1024x14x14xf32>
    %v1819 = stablehlo.reshape %v1818 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1820 = stablehlo.reshape %v1819 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1821 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1822 = stablehlo.maximum %v1820, %v1821 : tensor<64x1024x14x14xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1825 = stablehlo.convolution(%v1824, %s4b0W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<64x512x14x14xf32>
    %v1826 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1827 = stablehlo.add %v1825, %v1826 : tensor<64x512x14x14xf32>
    %v1828 = stablehlo.reshape %v1827 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1829 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1831 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v1832 = stablehlo.reduce(%v1829 init: %v1830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1833 = stablehlo.divide %v1832, %v1831 : tensor<512xf32>
    %arsums4b0g1mu = "stablehlo.all_reduce"(%v1833) ({
    ^bb0(%aras4b0g1mu: tensor<f32>, %arbs4b0g1mu: tensor<f32>):
      %aradds4b0g1mu = stablehlo.add %aras4b0g1mu, %arbs4b0g1mu : tensor<f32>
      stablehlo.return %aradds4b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1mu = stablehlo.divide %arsums4b0g1mu, %arns4b0g1mu : tensor<512xf32>
    %v1834 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1835 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1836 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v1837 = stablehlo.reduce(%v1834 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1838 = stablehlo.divide %v1837, %v1836 : tensor<512xf32>
    %v1839 = stablehlo.broadcast_in_dim %v1838, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1840 = stablehlo.subtract %v1834, %v1839 : tensor<64x512x14x14xf32>
    %v1841 = stablehlo.multiply %v1840, %v1840 : tensor<64x512x14x14xf32>
    %v1842 = stablehlo.reduce(%v1841 init: %v1835) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1843 = stablehlo.divide %v1842, %v1836 : tensor<512xf32>
    %v1844 = stablehlo.subtract %v1838, %armeans4b0g1mu : tensor<512xf32>
    %v1845 = stablehlo.multiply %v1844, %v1844 : tensor<512xf32>
    %v1846 = stablehlo.add %v1843, %v1845 : tensor<512xf32>
    %arsums4b0g1var = "stablehlo.all_reduce"(%v1846) ({
    ^bb0(%aras4b0g1var: tensor<f32>, %arbs4b0g1var: tensor<f32>):
      %aradds4b0g1var = stablehlo.add %aras4b0g1var, %arbs4b0g1var : tensor<f32>
      stablehlo.return %aradds4b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1var = stablehlo.divide %arsums4b0g1var, %arns4b0g1var : tensor<512xf32>
    %v1847 = stablehlo.concatenate %armeans4b0g1mu, %armeans4b0g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1848 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1849 = stablehlo.slice %v1847 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1850 = stablehlo.slice %v1847 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1851 = stablehlo.broadcast_in_dim %v1849, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1852 = stablehlo.broadcast_in_dim %v1850, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1853 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v1854 = stablehlo.add %v1852, %v1853 : tensor<64x512x14x14xf32>
    %v1855 = stablehlo.rsqrt %v1854 : tensor<64x512x14x14xf32>
    %v1856 = stablehlo.subtract %v1848, %v1851 : tensor<64x512x14x14xf32>
    %v1857 = stablehlo.multiply %v1856, %v1855 : tensor<64x512x14x14xf32>
    %v1858 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1859 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1860 = stablehlo.multiply %v1857, %v1858 : tensor<64x512x14x14xf32>
    %v1861 = stablehlo.add %v1860, %v1859 : tensor<64x512x14x14xf32>
    %v1862 = stablehlo.reshape %v1861 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1863 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1864 = stablehlo.maximum %v1862, %v1863 : tensor<64x100352xf32>
    %v1865 = stablehlo.reshape %v1864 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1866 = stablehlo.convolution(%v1865, %s4b0W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x7x7xf32>
    %v1867 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1868 = stablehlo.add %v1866, %v1867 : tensor<64x512x7x7xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1870 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1872 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1873 = stablehlo.reduce(%v1870 init: %v1871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1874 = stablehlo.divide %v1873, %v1872 : tensor<512xf32>
    %arsums4b0g2mu = "stablehlo.all_reduce"(%v1874) ({
    ^bb0(%aras4b0g2mu: tensor<f32>, %arbs4b0g2mu: tensor<f32>):
      %aradds4b0g2mu = stablehlo.add %aras4b0g2mu, %arbs4b0g2mu : tensor<f32>
      stablehlo.return %aradds4b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2mu = stablehlo.divide %arsums4b0g2mu, %arns4b0g2mu : tensor<512xf32>
    %v1875 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1877 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1878 = stablehlo.reduce(%v1875 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1879 = stablehlo.divide %v1878, %v1877 : tensor<512xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1881 = stablehlo.subtract %v1875, %v1880 : tensor<64x512x7x7xf32>
    %v1882 = stablehlo.multiply %v1881, %v1881 : tensor<64x512x7x7xf32>
    %v1883 = stablehlo.reduce(%v1882 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1884 = stablehlo.divide %v1883, %v1877 : tensor<512xf32>
    %v1885 = stablehlo.subtract %v1879, %armeans4b0g2mu : tensor<512xf32>
    %v1886 = stablehlo.multiply %v1885, %v1885 : tensor<512xf32>
    %v1887 = stablehlo.add %v1884, %v1886 : tensor<512xf32>
    %arsums4b0g2var = "stablehlo.all_reduce"(%v1887) ({
    ^bb0(%aras4b0g2var: tensor<f32>, %arbs4b0g2var: tensor<f32>):
      %aradds4b0g2var = stablehlo.add %aras4b0g2var, %arbs4b0g2var : tensor<f32>
      stablehlo.return %aradds4b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2var = stablehlo.divide %arsums4b0g2var, %arns4b0g2var : tensor<512xf32>
    %v1888 = stablehlo.concatenate %armeans4b0g2mu, %armeans4b0g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1889 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1890 = stablehlo.slice %v1888 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1891 = stablehlo.slice %v1888 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1892 = stablehlo.broadcast_in_dim %v1890, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1893 = stablehlo.broadcast_in_dim %v1891, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1894 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1895 = stablehlo.add %v1893, %v1894 : tensor<64x512x7x7xf32>
    %v1896 = stablehlo.rsqrt %v1895 : tensor<64x512x7x7xf32>
    %v1897 = stablehlo.subtract %v1889, %v1892 : tensor<64x512x7x7xf32>
    %v1898 = stablehlo.multiply %v1897, %v1896 : tensor<64x512x7x7xf32>
    %v1899 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1900 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1901 = stablehlo.multiply %v1898, %v1899 : tensor<64x512x7x7xf32>
    %v1902 = stablehlo.add %v1901, %v1900 : tensor<64x512x7x7xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1904 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1905 = stablehlo.maximum %v1903, %v1904 : tensor<64x25088xf32>
    %v1906 = stablehlo.reshape %v1905 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1907 = stablehlo.convolution(%v1906, %s4b0W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v1908 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1909 = stablehlo.add %v1907, %v1908 : tensor<64x2048x7x7xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1913 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v1914 = stablehlo.reduce(%v1911 init: %v1912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1915 = stablehlo.divide %v1914, %v1913 : tensor<2048xf32>
    %arsums4b0g3mu = "stablehlo.all_reduce"(%v1915) ({
    ^bb0(%aras4b0g3mu: tensor<f32>, %arbs4b0g3mu: tensor<f32>):
      %aradds4b0g3mu = stablehlo.add %aras4b0g3mu, %arbs4b0g3mu : tensor<f32>
      stablehlo.return %aradds4b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3mu = stablehlo.divide %arsums4b0g3mu, %arns4b0g3mu : tensor<2048xf32>
    %v1916 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v1919 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1920 = stablehlo.divide %v1919, %v1918 : tensor<2048xf32>
    %v1921 = stablehlo.broadcast_in_dim %v1920, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1922 = stablehlo.subtract %v1916, %v1921 : tensor<64x2048x7x7xf32>
    %v1923 = stablehlo.multiply %v1922, %v1922 : tensor<64x2048x7x7xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1925 = stablehlo.divide %v1924, %v1918 : tensor<2048xf32>
    %v1926 = stablehlo.subtract %v1920, %armeans4b0g3mu : tensor<2048xf32>
    %v1927 = stablehlo.multiply %v1926, %v1926 : tensor<2048xf32>
    %v1928 = stablehlo.add %v1925, %v1927 : tensor<2048xf32>
    %arsums4b0g3var = "stablehlo.all_reduce"(%v1928) ({
    ^bb0(%aras4b0g3var: tensor<f32>, %arbs4b0g3var: tensor<f32>):
      %aradds4b0g3var = stablehlo.add %aras4b0g3var, %arbs4b0g3var : tensor<f32>
      stablehlo.return %aradds4b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3var = stablehlo.divide %arsums4b0g3var, %arns4b0g3var : tensor<2048xf32>
    %v1929 = stablehlo.concatenate %armeans4b0g3mu, %armeans4b0g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v1930 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1931 = stablehlo.slice %v1929 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v1932 = stablehlo.slice %v1929 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v1933 = stablehlo.broadcast_in_dim %v1931, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1932, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1935 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1936 = stablehlo.add %v1934, %v1935 : tensor<64x2048x7x7xf32>
    %v1937 = stablehlo.rsqrt %v1936 : tensor<64x2048x7x7xf32>
    %v1938 = stablehlo.subtract %v1930, %v1933 : tensor<64x2048x7x7xf32>
    %v1939 = stablehlo.multiply %v1938, %v1937 : tensor<64x2048x7x7xf32>
    %v1940 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1941 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1942 = stablehlo.multiply %v1939, %v1940 : tensor<64x2048x7x7xf32>
    %v1943 = stablehlo.add %v1942, %v1941 : tensor<64x2048x7x7xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1945 = stablehlo.reshape %v1823 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1946 = stablehlo.convolution(%v1945, %s4b0Wp)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<2048x1024x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v1947 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1948 = stablehlo.add %v1946, %v1947 : tensor<64x2048x7x7xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1952 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v1953 = stablehlo.reduce(%v1950 init: %v1951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1954 = stablehlo.divide %v1953, %v1952 : tensor<2048xf32>
    %arsums4b0gpmu = "stablehlo.all_reduce"(%v1954) ({
    ^bb0(%aras4b0gpmu: tensor<f32>, %arbs4b0gpmu: tensor<f32>):
      %aradds4b0gpmu = stablehlo.add %aras4b0gpmu, %arbs4b0gpmu : tensor<f32>
      stablehlo.return %aradds4b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gpmu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gpmu = stablehlo.divide %arsums4b0gpmu, %arns4b0gpmu : tensor<2048xf32>
    %v1955 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1956 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1957 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v1958 = stablehlo.reduce(%v1955 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1959 = stablehlo.divide %v1958, %v1957 : tensor<2048xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1961 = stablehlo.subtract %v1955, %v1960 : tensor<64x2048x7x7xf32>
    %v1962 = stablehlo.multiply %v1961, %v1961 : tensor<64x2048x7x7xf32>
    %v1963 = stablehlo.reduce(%v1962 init: %v1956) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1964 = stablehlo.divide %v1963, %v1957 : tensor<2048xf32>
    %v1965 = stablehlo.subtract %v1959, %armeans4b0gpmu : tensor<2048xf32>
    %v1966 = stablehlo.multiply %v1965, %v1965 : tensor<2048xf32>
    %v1967 = stablehlo.add %v1964, %v1966 : tensor<2048xf32>
    %arsums4b0gpvar = "stablehlo.all_reduce"(%v1967) ({
    ^bb0(%aras4b0gpvar: tensor<f32>, %arbs4b0gpvar: tensor<f32>):
      %aradds4b0gpvar = stablehlo.add %aras4b0gpvar, %arbs4b0gpvar : tensor<f32>
      stablehlo.return %aradds4b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gpvar = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gpvar = stablehlo.divide %arsums4b0gpvar, %arns4b0gpvar : tensor<2048xf32>
    %v1968 = stablehlo.concatenate %armeans4b0gpmu, %armeans4b0gpvar, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v1969 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1970 = stablehlo.slice %v1968 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v1971 = stablehlo.slice %v1968 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1973 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1974 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1975 = stablehlo.add %v1973, %v1974 : tensor<64x2048x7x7xf32>
    %v1976 = stablehlo.rsqrt %v1975 : tensor<64x2048x7x7xf32>
    %v1977 = stablehlo.subtract %v1969, %v1972 : tensor<64x2048x7x7xf32>
    %v1978 = stablehlo.multiply %v1977, %v1976 : tensor<64x2048x7x7xf32>
    %v1979 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1980 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1981 = stablehlo.multiply %v1978, %v1979 : tensor<64x2048x7x7xf32>
    %v1982 = stablehlo.add %v1981, %v1980 : tensor<64x2048x7x7xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1984 = stablehlo.add %v1944, %v1983 : tensor<64x100352xf32>
    %v1985 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1986 = stablehlo.maximum %v1984, %v1985 : tensor<64x100352xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1988 = stablehlo.convolution(%v1987, %s4b1W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v1989 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1990 = stablehlo.add %v1988, %v1989 : tensor<64x512x7x7xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1992 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1994 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1995 = stablehlo.reduce(%v1992 init: %v1993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1996 = stablehlo.divide %v1995, %v1994 : tensor<512xf32>
    %arsums4b1g1mu = "stablehlo.all_reduce"(%v1996) ({
    ^bb0(%aras4b1g1mu: tensor<f32>, %arbs4b1g1mu: tensor<f32>):
      %aradds4b1g1mu = stablehlo.add %aras4b1g1mu, %arbs4b1g1mu : tensor<f32>
      stablehlo.return %aradds4b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1mu = stablehlo.divide %arsums4b1g1mu, %arns4b1g1mu : tensor<512xf32>
    %v1997 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1999 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2000 = stablehlo.reduce(%v1997 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2001 = stablehlo.divide %v2000, %v1999 : tensor<512xf32>
    %v2002 = stablehlo.broadcast_in_dim %v2001, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2003 = stablehlo.subtract %v1997, %v2002 : tensor<64x512x7x7xf32>
    %v2004 = stablehlo.multiply %v2003, %v2003 : tensor<64x512x7x7xf32>
    %v2005 = stablehlo.reduce(%v2004 init: %v1998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2006 = stablehlo.divide %v2005, %v1999 : tensor<512xf32>
    %v2007 = stablehlo.subtract %v2001, %armeans4b1g1mu : tensor<512xf32>
    %v2008 = stablehlo.multiply %v2007, %v2007 : tensor<512xf32>
    %v2009 = stablehlo.add %v2006, %v2008 : tensor<512xf32>
    %arsums4b1g1var = "stablehlo.all_reduce"(%v2009) ({
    ^bb0(%aras4b1g1var: tensor<f32>, %arbs4b1g1var: tensor<f32>):
      %aradds4b1g1var = stablehlo.add %aras4b1g1var, %arbs4b1g1var : tensor<f32>
      stablehlo.return %aradds4b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1var = stablehlo.divide %arsums4b1g1var, %arns4b1g1var : tensor<512xf32>
    %v2010 = stablehlo.concatenate %armeans4b1g1mu, %armeans4b1g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2011 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2012 = stablehlo.slice %v2010 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2013 = stablehlo.slice %v2010 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2012, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2015 = stablehlo.broadcast_in_dim %v2013, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2016 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2017 = stablehlo.add %v2015, %v2016 : tensor<64x512x7x7xf32>
    %v2018 = stablehlo.rsqrt %v2017 : tensor<64x512x7x7xf32>
    %v2019 = stablehlo.subtract %v2011, %v2014 : tensor<64x512x7x7xf32>
    %v2020 = stablehlo.multiply %v2019, %v2018 : tensor<64x512x7x7xf32>
    %v2021 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2022 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2023 = stablehlo.multiply %v2020, %v2021 : tensor<64x512x7x7xf32>
    %v2024 = stablehlo.add %v2023, %v2022 : tensor<64x512x7x7xf32>
    %v2025 = stablehlo.reshape %v2024 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2026 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2027 = stablehlo.maximum %v2025, %v2026 : tensor<64x25088xf32>
    %v2028 = stablehlo.reshape %v2027 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2029 = stablehlo.convolution(%v2028, %s4b1W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x7x7xf32>
    %v2030 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2031 = stablehlo.add %v2029, %v2030 : tensor<64x512x7x7xf32>
    %v2032 = stablehlo.reshape %v2031 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2033 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2035 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2036 = stablehlo.reduce(%v2033 init: %v2034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2037 = stablehlo.divide %v2036, %v2035 : tensor<512xf32>
    %arsums4b1g2mu = "stablehlo.all_reduce"(%v2037) ({
    ^bb0(%aras4b1g2mu: tensor<f32>, %arbs4b1g2mu: tensor<f32>):
      %aradds4b1g2mu = stablehlo.add %aras4b1g2mu, %arbs4b1g2mu : tensor<f32>
      stablehlo.return %aradds4b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2mu = stablehlo.divide %arsums4b1g2mu, %arns4b1g2mu : tensor<512xf32>
    %v2038 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2040 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2041 = stablehlo.reduce(%v2038 init: %v2039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2042 = stablehlo.divide %v2041, %v2040 : tensor<512xf32>
    %v2043 = stablehlo.broadcast_in_dim %v2042, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2044 = stablehlo.subtract %v2038, %v2043 : tensor<64x512x7x7xf32>
    %v2045 = stablehlo.multiply %v2044, %v2044 : tensor<64x512x7x7xf32>
    %v2046 = stablehlo.reduce(%v2045 init: %v2039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2047 = stablehlo.divide %v2046, %v2040 : tensor<512xf32>
    %v2048 = stablehlo.subtract %v2042, %armeans4b1g2mu : tensor<512xf32>
    %v2049 = stablehlo.multiply %v2048, %v2048 : tensor<512xf32>
    %v2050 = stablehlo.add %v2047, %v2049 : tensor<512xf32>
    %arsums4b1g2var = "stablehlo.all_reduce"(%v2050) ({
    ^bb0(%aras4b1g2var: tensor<f32>, %arbs4b1g2var: tensor<f32>):
      %aradds4b1g2var = stablehlo.add %aras4b1g2var, %arbs4b1g2var : tensor<f32>
      stablehlo.return %aradds4b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2var = stablehlo.divide %arsums4b1g2var, %arns4b1g2var : tensor<512xf32>
    %v2051 = stablehlo.concatenate %armeans4b1g2mu, %armeans4b1g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2052 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2053 = stablehlo.slice %v2051 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2054 = stablehlo.slice %v2051 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2055 = stablehlo.broadcast_in_dim %v2053, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2056 = stablehlo.broadcast_in_dim %v2054, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2057 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2058 = stablehlo.add %v2056, %v2057 : tensor<64x512x7x7xf32>
    %v2059 = stablehlo.rsqrt %v2058 : tensor<64x512x7x7xf32>
    %v2060 = stablehlo.subtract %v2052, %v2055 : tensor<64x512x7x7xf32>
    %v2061 = stablehlo.multiply %v2060, %v2059 : tensor<64x512x7x7xf32>
    %v2062 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2063 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2064 = stablehlo.multiply %v2061, %v2062 : tensor<64x512x7x7xf32>
    %v2065 = stablehlo.add %v2064, %v2063 : tensor<64x512x7x7xf32>
    %v2066 = stablehlo.reshape %v2065 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2067 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2068 = stablehlo.maximum %v2066, %v2067 : tensor<64x25088xf32>
    %v2069 = stablehlo.reshape %v2068 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2070 = stablehlo.convolution(%v2069, %s4b1W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v2071 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2072 = stablehlo.add %v2070, %v2071 : tensor<64x2048x7x7xf32>
    %v2073 = stablehlo.reshape %v2072 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2074 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2075 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2076 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2077 = stablehlo.reduce(%v2074 init: %v2075) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2078 = stablehlo.divide %v2077, %v2076 : tensor<2048xf32>
    %arsums4b1g3mu = "stablehlo.all_reduce"(%v2078) ({
    ^bb0(%aras4b1g3mu: tensor<f32>, %arbs4b1g3mu: tensor<f32>):
      %aradds4b1g3mu = stablehlo.add %aras4b1g3mu, %arbs4b1g3mu : tensor<f32>
      stablehlo.return %aradds4b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3mu = stablehlo.divide %arsums4b1g3mu, %arns4b1g3mu : tensor<2048xf32>
    %v2079 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2081 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2082 = stablehlo.reduce(%v2079 init: %v2080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2083 = stablehlo.divide %v2082, %v2081 : tensor<2048xf32>
    %v2084 = stablehlo.broadcast_in_dim %v2083, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2085 = stablehlo.subtract %v2079, %v2084 : tensor<64x2048x7x7xf32>
    %v2086 = stablehlo.multiply %v2085, %v2085 : tensor<64x2048x7x7xf32>
    %v2087 = stablehlo.reduce(%v2086 init: %v2080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2088 = stablehlo.divide %v2087, %v2081 : tensor<2048xf32>
    %v2089 = stablehlo.subtract %v2083, %armeans4b1g3mu : tensor<2048xf32>
    %v2090 = stablehlo.multiply %v2089, %v2089 : tensor<2048xf32>
    %v2091 = stablehlo.add %v2088, %v2090 : tensor<2048xf32>
    %arsums4b1g3var = "stablehlo.all_reduce"(%v2091) ({
    ^bb0(%aras4b1g3var: tensor<f32>, %arbs4b1g3var: tensor<f32>):
      %aradds4b1g3var = stablehlo.add %aras4b1g3var, %arbs4b1g3var : tensor<f32>
      stablehlo.return %aradds4b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3var = stablehlo.divide %arsums4b1g3var, %arns4b1g3var : tensor<2048xf32>
    %v2092 = stablehlo.concatenate %armeans4b1g3mu, %armeans4b1g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2093 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2094 = stablehlo.slice %v2092 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2095 = stablehlo.slice %v2092 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2096 = stablehlo.broadcast_in_dim %v2094, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2097 = stablehlo.broadcast_in_dim %v2095, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2098 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2099 = stablehlo.add %v2097, %v2098 : tensor<64x2048x7x7xf32>
    %v2100 = stablehlo.rsqrt %v2099 : tensor<64x2048x7x7xf32>
    %v2101 = stablehlo.subtract %v2093, %v2096 : tensor<64x2048x7x7xf32>
    %v2102 = stablehlo.multiply %v2101, %v2100 : tensor<64x2048x7x7xf32>
    %v2103 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2104 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2105 = stablehlo.multiply %v2102, %v2103 : tensor<64x2048x7x7xf32>
    %v2106 = stablehlo.add %v2105, %v2104 : tensor<64x2048x7x7xf32>
    %v2107 = stablehlo.reshape %v2106 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2108 = stablehlo.reshape %v2107 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2109 = stablehlo.reshape %v1986 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2110 = stablehlo.add %v2108, %v2109 : tensor<64x2048x7x7xf32>
    %v2111 = stablehlo.reshape %v2110 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2112 = stablehlo.reshape %v2111 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2113 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2114 = stablehlo.maximum %v2112, %v2113 : tensor<64x2048x7x7xf32>
    %v2115 = stablehlo.reshape %v2114 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2116 = stablehlo.reshape %v2115 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2117 = stablehlo.convolution(%v2116, %s4b2W1)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v2118 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2119 = stablehlo.add %v2117, %v2118 : tensor<64x512x7x7xf32>
    %v2120 = stablehlo.reshape %v2119 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2121 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2123 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2124 = stablehlo.reduce(%v2121 init: %v2122) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2125 = stablehlo.divide %v2124, %v2123 : tensor<512xf32>
    %arsums4b2g1mu = "stablehlo.all_reduce"(%v2125) ({
    ^bb0(%aras4b2g1mu: tensor<f32>, %arbs4b2g1mu: tensor<f32>):
      %aradds4b2g1mu = stablehlo.add %aras4b2g1mu, %arbs4b2g1mu : tensor<f32>
      stablehlo.return %aradds4b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1mu = stablehlo.divide %arsums4b2g1mu, %arns4b2g1mu : tensor<512xf32>
    %v2126 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2128 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2129 = stablehlo.reduce(%v2126 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2130 = stablehlo.divide %v2129, %v2128 : tensor<512xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2130, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2132 = stablehlo.subtract %v2126, %v2131 : tensor<64x512x7x7xf32>
    %v2133 = stablehlo.multiply %v2132, %v2132 : tensor<64x512x7x7xf32>
    %v2134 = stablehlo.reduce(%v2133 init: %v2127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2135 = stablehlo.divide %v2134, %v2128 : tensor<512xf32>
    %v2136 = stablehlo.subtract %v2130, %armeans4b2g1mu : tensor<512xf32>
    %v2137 = stablehlo.multiply %v2136, %v2136 : tensor<512xf32>
    %v2138 = stablehlo.add %v2135, %v2137 : tensor<512xf32>
    %arsums4b2g1var = "stablehlo.all_reduce"(%v2138) ({
    ^bb0(%aras4b2g1var: tensor<f32>, %arbs4b2g1var: tensor<f32>):
      %aradds4b2g1var = stablehlo.add %aras4b2g1var, %arbs4b2g1var : tensor<f32>
      stablehlo.return %aradds4b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1var = stablehlo.divide %arsums4b2g1var, %arns4b2g1var : tensor<512xf32>
    %v2139 = stablehlo.concatenate %armeans4b2g1mu, %armeans4b2g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2140 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2141 = stablehlo.slice %v2139 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2142 = stablehlo.slice %v2139 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2143 = stablehlo.broadcast_in_dim %v2141, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2144 = stablehlo.broadcast_in_dim %v2142, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2145 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2146 = stablehlo.add %v2144, %v2145 : tensor<64x512x7x7xf32>
    %v2147 = stablehlo.rsqrt %v2146 : tensor<64x512x7x7xf32>
    %v2148 = stablehlo.subtract %v2140, %v2143 : tensor<64x512x7x7xf32>
    %v2149 = stablehlo.multiply %v2148, %v2147 : tensor<64x512x7x7xf32>
    %v2150 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2151 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2152 = stablehlo.multiply %v2149, %v2150 : tensor<64x512x7x7xf32>
    %v2153 = stablehlo.add %v2152, %v2151 : tensor<64x512x7x7xf32>
    %v2154 = stablehlo.reshape %v2153 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2155 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2156 = stablehlo.maximum %v2154, %v2155 : tensor<64x25088xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2158 = stablehlo.convolution(%v2157, %s4b2W2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x7x7xf32>
    %v2159 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2160 = stablehlo.add %v2158, %v2159 : tensor<64x512x7x7xf32>
    %v2161 = stablehlo.reshape %v2160 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2162 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2164 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2165 = stablehlo.reduce(%v2162 init: %v2163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2166 = stablehlo.divide %v2165, %v2164 : tensor<512xf32>
    %arsums4b2g2mu = "stablehlo.all_reduce"(%v2166) ({
    ^bb0(%aras4b2g2mu: tensor<f32>, %arbs4b2g2mu: tensor<f32>):
      %aradds4b2g2mu = stablehlo.add %aras4b2g2mu, %arbs4b2g2mu : tensor<f32>
      stablehlo.return %aradds4b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2mu = stablehlo.divide %arsums4b2g2mu, %arns4b2g2mu : tensor<512xf32>
    %v2167 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2168 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2169 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2170 = stablehlo.reduce(%v2167 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2171 = stablehlo.divide %v2170, %v2169 : tensor<512xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2173 = stablehlo.subtract %v2167, %v2172 : tensor<64x512x7x7xf32>
    %v2174 = stablehlo.multiply %v2173, %v2173 : tensor<64x512x7x7xf32>
    %v2175 = stablehlo.reduce(%v2174 init: %v2168) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2176 = stablehlo.divide %v2175, %v2169 : tensor<512xf32>
    %v2177 = stablehlo.subtract %v2171, %armeans4b2g2mu : tensor<512xf32>
    %v2178 = stablehlo.multiply %v2177, %v2177 : tensor<512xf32>
    %v2179 = stablehlo.add %v2176, %v2178 : tensor<512xf32>
    %arsums4b2g2var = "stablehlo.all_reduce"(%v2179) ({
    ^bb0(%aras4b2g2var: tensor<f32>, %arbs4b2g2var: tensor<f32>):
      %aradds4b2g2var = stablehlo.add %aras4b2g2var, %arbs4b2g2var : tensor<f32>
      stablehlo.return %aradds4b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2var = stablehlo.divide %arsums4b2g2var, %arns4b2g2var : tensor<512xf32>
    %v2180 = stablehlo.concatenate %armeans4b2g2mu, %armeans4b2g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2181 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2182 = stablehlo.slice %v2180 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2183 = stablehlo.slice %v2180 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2184 = stablehlo.broadcast_in_dim %v2182, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2183, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2186 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2187 = stablehlo.add %v2185, %v2186 : tensor<64x512x7x7xf32>
    %v2188 = stablehlo.rsqrt %v2187 : tensor<64x512x7x7xf32>
    %v2189 = stablehlo.subtract %v2181, %v2184 : tensor<64x512x7x7xf32>
    %v2190 = stablehlo.multiply %v2189, %v2188 : tensor<64x512x7x7xf32>
    %v2191 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2192 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2193 = stablehlo.multiply %v2190, %v2191 : tensor<64x512x7x7xf32>
    %v2194 = stablehlo.add %v2193, %v2192 : tensor<64x512x7x7xf32>
    %v2195 = stablehlo.reshape %v2194 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2196 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2197 = stablehlo.maximum %v2195, %v2196 : tensor<64x25088xf32>
    %v2198 = stablehlo.reshape %v2197 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2199 = stablehlo.convolution(%v2198, %s4b2W3)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v2200 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2201 = stablehlo.add %v2199, %v2200 : tensor<64x2048x7x7xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2203 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2205 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2206 = stablehlo.reduce(%v2203 init: %v2204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2207 = stablehlo.divide %v2206, %v2205 : tensor<2048xf32>
    %arsums4b2g3mu = "stablehlo.all_reduce"(%v2207) ({
    ^bb0(%aras4b2g3mu: tensor<f32>, %arbs4b2g3mu: tensor<f32>):
      %aradds4b2g3mu = stablehlo.add %aras4b2g3mu, %arbs4b2g3mu : tensor<f32>
      stablehlo.return %aradds4b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3mu = stablehlo.divide %arsums4b2g3mu, %arns4b2g3mu : tensor<2048xf32>
    %v2208 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2210 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2211 = stablehlo.reduce(%v2208 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2212 = stablehlo.divide %v2211, %v2210 : tensor<2048xf32>
    %v2213 = stablehlo.broadcast_in_dim %v2212, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2214 = stablehlo.subtract %v2208, %v2213 : tensor<64x2048x7x7xf32>
    %v2215 = stablehlo.multiply %v2214, %v2214 : tensor<64x2048x7x7xf32>
    %v2216 = stablehlo.reduce(%v2215 init: %v2209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2217 = stablehlo.divide %v2216, %v2210 : tensor<2048xf32>
    %v2218 = stablehlo.subtract %v2212, %armeans4b2g3mu : tensor<2048xf32>
    %v2219 = stablehlo.multiply %v2218, %v2218 : tensor<2048xf32>
    %v2220 = stablehlo.add %v2217, %v2219 : tensor<2048xf32>
    %arsums4b2g3var = "stablehlo.all_reduce"(%v2220) ({
    ^bb0(%aras4b2g3var: tensor<f32>, %arbs4b2g3var: tensor<f32>):
      %aradds4b2g3var = stablehlo.add %aras4b2g3var, %arbs4b2g3var : tensor<f32>
      stablehlo.return %aradds4b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3var = stablehlo.divide %arsums4b2g3var, %arns4b2g3var : tensor<2048xf32>
    %v2221 = stablehlo.concatenate %armeans4b2g3mu, %armeans4b2g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2222 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2223 = stablehlo.slice %v2221 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2224 = stablehlo.slice %v2221 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2225 = stablehlo.broadcast_in_dim %v2223, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2226 = stablehlo.broadcast_in_dim %v2224, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2227 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2228 = stablehlo.add %v2226, %v2227 : tensor<64x2048x7x7xf32>
    %v2229 = stablehlo.rsqrt %v2228 : tensor<64x2048x7x7xf32>
    %v2230 = stablehlo.subtract %v2222, %v2225 : tensor<64x2048x7x7xf32>
    %v2231 = stablehlo.multiply %v2230, %v2229 : tensor<64x2048x7x7xf32>
    %v2232 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2233 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2234 = stablehlo.multiply %v2231, %v2232 : tensor<64x2048x7x7xf32>
    %v2235 = stablehlo.add %v2234, %v2233 : tensor<64x2048x7x7xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2238 = stablehlo.reshape %v2115 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2239 = stablehlo.add %v2237, %v2238 : tensor<64x2048x7x7xf32>
    %v2240 = stablehlo.reshape %v2239 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2241 = stablehlo.reshape %v2240 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2242 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2243 = stablehlo.maximum %v2241, %v2242 : tensor<64x2048x7x7xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2245 = stablehlo.reshape %v2244 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2247 = stablehlo.reduce(%v2245 init: %v2246) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048xf32>
    %v2248 = stablehlo.constant dense<49.0> : tensor<64x2048xf32>
    %v2249 = stablehlo.divide %v2247, %v2248 : tensor<64x2048xf32>
    %v2250 = stablehlo.dot_general %v2249, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<2048x1000xf32>) -> tensor<64x1000xf32>
    %v2251 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v2252 = stablehlo.add %v2250, %v2251 : tensor<64x1000xf32>
    %v2253 = stablehlo.reshape %v2252 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v2254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2255 = stablehlo.exponential %v2253 : tensor<64x1x1000xf32>
    %v2256 = stablehlo.reduce(%v2255 init: %v2254) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v2257 = stablehlo.broadcast_in_dim %v2256, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v2258 = stablehlo.divide %v2255, %v2257 : tensor<64x1x1000xf32>
    %v2259 = stablehlo.reshape %v2258 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v2260 = stablehlo.subtract %v2259, %onehot : tensor<64x1000xf32>
    %v2261 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v2262 = stablehlo.multiply %onehot, %v2261 : tensor<64x1000xf32>
    %v2263 = stablehlo.add %v2260, %v2262 : tensor<64x1000xf32>
    %v2264 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v2265 = stablehlo.add %v2263, %v2264 : tensor<64x1000xf32>
    %v2266 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v2267 = stablehlo.divide %v2265, %v2266 : tensor<64x1000xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v2269 = stablehlo.dot_general %v2268, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<64x1x2048xf32>
    %v2270 = stablehlo.reshape %v2269 : (tensor<64x1x2048xf32>) -> tensor<64x2048xf32>
    %v2271 = stablehlo.dot_general %v2249, %v2267, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<64x1000xf32>) -> tensor<2048x1000xf32>
    %v2272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2273 = stablehlo.reduce(%v2267 init: %v2272) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v2274 = stablehlo.broadcast_in_dim %v2270, dims = [0, 1] : (tensor<64x2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2275 = stablehlo.constant dense<49.0> : tensor<64x2048x7x7xf32>
    %v2276 = stablehlo.divide %v2274, %v2275 : tensor<64x2048x7x7xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2279 = stablehlo.reshape %v2240 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2281 = stablehlo.compare GT, %v2279, %v2280 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2282 = stablehlo.select %v2281, %v2278, %v2280 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2283 = stablehlo.reshape %v2282 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2284 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2285 = stablehlo.slice %v2221 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2286 = stablehlo.slice %v2221 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2287 = stablehlo.broadcast_in_dim %v2285, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2288 = stablehlo.broadcast_in_dim %v2286, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2289 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2290 = stablehlo.add %v2288, %v2289 : tensor<64x2048x7x7xf32>
    %v2291 = stablehlo.rsqrt %v2290 : tensor<64x2048x7x7xf32>
    %v2292 = stablehlo.subtract %v2284, %v2287 : tensor<64x2048x7x7xf32>
    %v2293 = stablehlo.multiply %v2292, %v2291 : tensor<64x2048x7x7xf32>
    %v2294 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2295 = stablehlo.reshape %v2283 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2296 = stablehlo.multiply %v2294, %v2295 : tensor<64x2048x7x7xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2299 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2300 = stablehlo.divide %v2299, %v2298 : tensor<2048xf32>
    %v2301 = stablehlo.multiply %v2293, %v2296 : tensor<64x2048x7x7xf32>
    %v2302 = stablehlo.reduce(%v2301 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2303 = stablehlo.divide %v2302, %v2298 : tensor<2048xf32>
    %v2304 = stablehlo.concatenate %v2300, %v2303, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2305 = stablehlo.concatenate %v2221, %v2304, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b2g3dst = "stablehlo.all_reduce"(%v2305) ({
    ^bb0(%aras4b2g3dst: tensor<f32>, %arbs4b2g3dst: tensor<f32>):
      %aradds4b2g3dst = stablehlo.add %aras4b2g3dst, %arbs4b2g3dst : tensor<f32>
      stablehlo.return %aradds4b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b2g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b2g3dst = stablehlo.divide %arsums4b2g3dst, %arns4b2g3dst : tensor<8192xf32>
    %v2306 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2307 = stablehlo.slice %armeans4b2g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2308 = stablehlo.slice %armeans4b2g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2309 = stablehlo.slice %armeans4b2g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2310 = stablehlo.slice %armeans4b2g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2311 = stablehlo.broadcast_in_dim %v2307, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2312 = stablehlo.broadcast_in_dim %v2308, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2313 = stablehlo.broadcast_in_dim %v2309, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2314 = stablehlo.broadcast_in_dim %v2310, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2315 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2316 = stablehlo.add %v2312, %v2315 : tensor<64x2048x7x7xf32>
    %v2317 = stablehlo.rsqrt %v2316 : tensor<64x2048x7x7xf32>
    %v2318 = stablehlo.subtract %v2306, %v2311 : tensor<64x2048x7x7xf32>
    %v2319 = stablehlo.multiply %v2318, %v2317 : tensor<64x2048x7x7xf32>
    %v2320 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2321 = stablehlo.reshape %v2283 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2322 = stablehlo.multiply %v2320, %v2321 : tensor<64x2048x7x7xf32>
    %v2323 = stablehlo.subtract %v2322, %v2313 : tensor<64x2048x7x7xf32>
    %v2324 = stablehlo.multiply %v2319, %v2314 : tensor<64x2048x7x7xf32>
    %v2325 = stablehlo.subtract %v2323, %v2324 : tensor<64x2048x7x7xf32>
    %v2326 = stablehlo.multiply %v2317, %v2325 : tensor<64x2048x7x7xf32>
    %v2327 = stablehlo.reshape %v2326 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2329 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2330 = stablehlo.transpose %v2329, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2331 = stablehlo.convolution(%v2328, %v2330)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v2332 = stablehlo.reshape %v2331 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2333 = stablehlo.reshape %v2332 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2334 = stablehlo.reshape %v2195 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2335 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2336 = stablehlo.compare GT, %v2334, %v2335 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2337 = stablehlo.select %v2336, %v2333, %v2335 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2338 = stablehlo.reshape %v2337 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2339 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2340 = stablehlo.slice %v2180 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2341 = stablehlo.slice %v2180 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2342 = stablehlo.broadcast_in_dim %v2340, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2343 = stablehlo.broadcast_in_dim %v2341, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2344 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2345 = stablehlo.add %v2343, %v2344 : tensor<64x512x7x7xf32>
    %v2346 = stablehlo.rsqrt %v2345 : tensor<64x512x7x7xf32>
    %v2347 = stablehlo.subtract %v2339, %v2342 : tensor<64x512x7x7xf32>
    %v2348 = stablehlo.multiply %v2347, %v2346 : tensor<64x512x7x7xf32>
    %v2349 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2350 = stablehlo.reshape %v2338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2351 = stablehlo.multiply %v2349, %v2350 : tensor<64x512x7x7xf32>
    %v2352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2353 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2354 = stablehlo.reduce(%v2351 init: %v2352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2355 = stablehlo.divide %v2354, %v2353 : tensor<512xf32>
    %v2356 = stablehlo.multiply %v2348, %v2351 : tensor<64x512x7x7xf32>
    %v2357 = stablehlo.reduce(%v2356 init: %v2352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2358 = stablehlo.divide %v2357, %v2353 : tensor<512xf32>
    %v2359 = stablehlo.concatenate %v2355, %v2358, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2360 = stablehlo.concatenate %v2180, %v2359, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b2g2dst = "stablehlo.all_reduce"(%v2360) ({
    ^bb0(%aras4b2g2dst: tensor<f32>, %arbs4b2g2dst: tensor<f32>):
      %aradds4b2g2dst = stablehlo.add %aras4b2g2dst, %arbs4b2g2dst : tensor<f32>
      stablehlo.return %aradds4b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g2dst = stablehlo.divide %arsums4b2g2dst, %arns4b2g2dst : tensor<2048xf32>
    %v2361 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2362 = stablehlo.slice %armeans4b2g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2363 = stablehlo.slice %armeans4b2g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2364 = stablehlo.slice %armeans4b2g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2365 = stablehlo.slice %armeans4b2g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2362, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2367 = stablehlo.broadcast_in_dim %v2363, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2368 = stablehlo.broadcast_in_dim %v2364, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2365, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2370 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2371 = stablehlo.add %v2367, %v2370 : tensor<64x512x7x7xf32>
    %v2372 = stablehlo.rsqrt %v2371 : tensor<64x512x7x7xf32>
    %v2373 = stablehlo.subtract %v2361, %v2366 : tensor<64x512x7x7xf32>
    %v2374 = stablehlo.multiply %v2373, %v2372 : tensor<64x512x7x7xf32>
    %v2375 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2376 = stablehlo.reshape %v2338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2377 = stablehlo.multiply %v2375, %v2376 : tensor<64x512x7x7xf32>
    %v2378 = stablehlo.subtract %v2377, %v2368 : tensor<64x512x7x7xf32>
    %v2379 = stablehlo.multiply %v2374, %v2369 : tensor<64x512x7x7xf32>
    %v2380 = stablehlo.subtract %v2378, %v2379 : tensor<64x512x7x7xf32>
    %v2381 = stablehlo.multiply %v2372, %v2380 : tensor<64x512x7x7xf32>
    %v2382 = stablehlo.reshape %v2381 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2383 = stablehlo.reshape %v2382 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2384 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2385 = stablehlo.transpose %v2384, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2386 = stablehlo.convolution(%v2383, %v2385)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x7x7xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2388 = stablehlo.reshape %v2387 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2389 = stablehlo.reshape %v2154 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2390 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2391 = stablehlo.compare GT, %v2389, %v2390 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2392 = stablehlo.select %v2391, %v2388, %v2390 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2394 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2395 = stablehlo.slice %v2139 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2396 = stablehlo.slice %v2139 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2397 = stablehlo.broadcast_in_dim %v2395, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2398 = stablehlo.broadcast_in_dim %v2396, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2399 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2400 = stablehlo.add %v2398, %v2399 : tensor<64x512x7x7xf32>
    %v2401 = stablehlo.rsqrt %v2400 : tensor<64x512x7x7xf32>
    %v2402 = stablehlo.subtract %v2394, %v2397 : tensor<64x512x7x7xf32>
    %v2403 = stablehlo.multiply %v2402, %v2401 : tensor<64x512x7x7xf32>
    %v2404 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2405 = stablehlo.reshape %v2393 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2406 = stablehlo.multiply %v2404, %v2405 : tensor<64x512x7x7xf32>
    %v2407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2408 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2409 = stablehlo.reduce(%v2406 init: %v2407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2410 = stablehlo.divide %v2409, %v2408 : tensor<512xf32>
    %v2411 = stablehlo.multiply %v2403, %v2406 : tensor<64x512x7x7xf32>
    %v2412 = stablehlo.reduce(%v2411 init: %v2407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2413 = stablehlo.divide %v2412, %v2408 : tensor<512xf32>
    %v2414 = stablehlo.concatenate %v2410, %v2413, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2415 = stablehlo.concatenate %v2139, %v2414, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b2g1dst = "stablehlo.all_reduce"(%v2415) ({
    ^bb0(%aras4b2g1dst: tensor<f32>, %arbs4b2g1dst: tensor<f32>):
      %aradds4b2g1dst = stablehlo.add %aras4b2g1dst, %arbs4b2g1dst : tensor<f32>
      stablehlo.return %aradds4b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g1dst = stablehlo.divide %arsums4b2g1dst, %arns4b2g1dst : tensor<2048xf32>
    %v2416 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2417 = stablehlo.slice %armeans4b2g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2418 = stablehlo.slice %armeans4b2g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2419 = stablehlo.slice %armeans4b2g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2420 = stablehlo.slice %armeans4b2g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2421 = stablehlo.broadcast_in_dim %v2417, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2422 = stablehlo.broadcast_in_dim %v2418, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2423 = stablehlo.broadcast_in_dim %v2419, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2424 = stablehlo.broadcast_in_dim %v2420, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2425 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2426 = stablehlo.add %v2422, %v2425 : tensor<64x512x7x7xf32>
    %v2427 = stablehlo.rsqrt %v2426 : tensor<64x512x7x7xf32>
    %v2428 = stablehlo.subtract %v2416, %v2421 : tensor<64x512x7x7xf32>
    %v2429 = stablehlo.multiply %v2428, %v2427 : tensor<64x512x7x7xf32>
    %v2430 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2431 = stablehlo.reshape %v2393 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2432 = stablehlo.multiply %v2430, %v2431 : tensor<64x512x7x7xf32>
    %v2433 = stablehlo.subtract %v2432, %v2423 : tensor<64x512x7x7xf32>
    %v2434 = stablehlo.multiply %v2429, %v2424 : tensor<64x512x7x7xf32>
    %v2435 = stablehlo.subtract %v2433, %v2434 : tensor<64x512x7x7xf32>
    %v2436 = stablehlo.multiply %v2427, %v2435 : tensor<64x512x7x7xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2438 = stablehlo.reshape %v2437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2439 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v2440 = stablehlo.transpose %v2439, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2441 = stablehlo.convolution(%v2438, %v2440)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2443 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2444 = stablehlo.reshape %v2283 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2445 = stablehlo.add %v2443, %v2444 : tensor<64x2048x7x7xf32>
    %v2446 = stablehlo.reshape %v2445 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2447 = stablehlo.reshape %v2115 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2448 = stablehlo.reshape %v2437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2449 = stablehlo.transpose %v2447, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2450 = stablehlo.transpose %v2448, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2451 = stablehlo.convolution(%v2449, %v2450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xf32>, tensor<512x64x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v2452 = stablehlo.transpose %v2451, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2453 = stablehlo.reshape %v2120 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2454 = stablehlo.slice %v2139 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2455 = stablehlo.slice %v2139 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2456 = stablehlo.broadcast_in_dim %v2454, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2455, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2458 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2459 = stablehlo.add %v2457, %v2458 : tensor<64x512x7x7xf32>
    %v2460 = stablehlo.rsqrt %v2459 : tensor<64x512x7x7xf32>
    %v2461 = stablehlo.subtract %v2453, %v2456 : tensor<64x512x7x7xf32>
    %v2462 = stablehlo.multiply %v2461, %v2460 : tensor<64x512x7x7xf32>
    %v2463 = stablehlo.reshape %v2393 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<64x512x7x7xf32>
    %v2465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2466 = stablehlo.reduce(%v2464 init: %v2465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2467 = stablehlo.reshape %v2393 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2469 = stablehlo.reduce(%v2467 init: %v2468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2470 = stablehlo.reshape %v2156 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2471 = stablehlo.reshape %v2382 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2472 = stablehlo.transpose %v2470, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2473 = stablehlo.transpose %v2471, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2474 = stablehlo.convolution(%v2472, %v2473)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xf32>, tensor<512x64x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v2475 = stablehlo.transpose %v2474, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2476 = stablehlo.reshape %v2161 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2477 = stablehlo.slice %v2180 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2478 = stablehlo.slice %v2180 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2477, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2478, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2481 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2482 = stablehlo.add %v2480, %v2481 : tensor<64x512x7x7xf32>
    %v2483 = stablehlo.rsqrt %v2482 : tensor<64x512x7x7xf32>
    %v2484 = stablehlo.subtract %v2476, %v2479 : tensor<64x512x7x7xf32>
    %v2485 = stablehlo.multiply %v2484, %v2483 : tensor<64x512x7x7xf32>
    %v2486 = stablehlo.reshape %v2338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2487 = stablehlo.multiply %v2486, %v2485 : tensor<64x512x7x7xf32>
    %v2488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2489 = stablehlo.reduce(%v2487 init: %v2488) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2490 = stablehlo.reshape %v2338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2492 = stablehlo.reduce(%v2490 init: %v2491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2493 = stablehlo.reshape %v2197 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2494 = stablehlo.reshape %v2327 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2495 = stablehlo.transpose %v2493, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2496 = stablehlo.transpose %v2494, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2497 = stablehlo.convolution(%v2495, %v2496)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xf32>, tensor<2048x64x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2498 = stablehlo.transpose %v2497, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2499 = stablehlo.reshape %v2202 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2500 = stablehlo.slice %v2221 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2501 = stablehlo.slice %v2221 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2502 = stablehlo.broadcast_in_dim %v2500, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2503 = stablehlo.broadcast_in_dim %v2501, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2504 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2505 = stablehlo.add %v2503, %v2504 : tensor<64x2048x7x7xf32>
    %v2506 = stablehlo.rsqrt %v2505 : tensor<64x2048x7x7xf32>
    %v2507 = stablehlo.subtract %v2499, %v2502 : tensor<64x2048x7x7xf32>
    %v2508 = stablehlo.multiply %v2507, %v2506 : tensor<64x2048x7x7xf32>
    %v2509 = stablehlo.reshape %v2283 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2510 = stablehlo.multiply %v2509, %v2508 : tensor<64x2048x7x7xf32>
    %v2511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2512 = stablehlo.reduce(%v2510 init: %v2511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2513 = stablehlo.reshape %v2283 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2515 = stablehlo.reduce(%v2513 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2516 = stablehlo.reshape %v2446 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2517 = stablehlo.reshape %v2111 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2518 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2519 = stablehlo.compare GT, %v2517, %v2518 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2520 = stablehlo.select %v2519, %v2516, %v2518 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2521 = stablehlo.reshape %v2520 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2522 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2523 = stablehlo.slice %v2092 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2524 = stablehlo.slice %v2092 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2525 = stablehlo.broadcast_in_dim %v2523, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2526 = stablehlo.broadcast_in_dim %v2524, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2527 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2528 = stablehlo.add %v2526, %v2527 : tensor<64x2048x7x7xf32>
    %v2529 = stablehlo.rsqrt %v2528 : tensor<64x2048x7x7xf32>
    %v2530 = stablehlo.subtract %v2522, %v2525 : tensor<64x2048x7x7xf32>
    %v2531 = stablehlo.multiply %v2530, %v2529 : tensor<64x2048x7x7xf32>
    %v2532 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2533 = stablehlo.reshape %v2521 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2534 = stablehlo.multiply %v2532, %v2533 : tensor<64x2048x7x7xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2536 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2537 = stablehlo.reduce(%v2534 init: %v2535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2538 = stablehlo.divide %v2537, %v2536 : tensor<2048xf32>
    %v2539 = stablehlo.multiply %v2531, %v2534 : tensor<64x2048x7x7xf32>
    %v2540 = stablehlo.reduce(%v2539 init: %v2535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2541 = stablehlo.divide %v2540, %v2536 : tensor<2048xf32>
    %v2542 = stablehlo.concatenate %v2538, %v2541, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2543 = stablehlo.concatenate %v2092, %v2542, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b1g3dst = "stablehlo.all_reduce"(%v2543) ({
    ^bb0(%aras4b1g3dst: tensor<f32>, %arbs4b1g3dst: tensor<f32>):
      %aradds4b1g3dst = stablehlo.add %aras4b1g3dst, %arbs4b1g3dst : tensor<f32>
      stablehlo.return %aradds4b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b1g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b1g3dst = stablehlo.divide %arsums4b1g3dst, %arns4b1g3dst : tensor<8192xf32>
    %v2544 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2545 = stablehlo.slice %armeans4b1g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2546 = stablehlo.slice %armeans4b1g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2547 = stablehlo.slice %armeans4b1g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2548 = stablehlo.slice %armeans4b1g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2545, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2550 = stablehlo.broadcast_in_dim %v2546, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2547, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2552 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2553 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2554 = stablehlo.add %v2550, %v2553 : tensor<64x2048x7x7xf32>
    %v2555 = stablehlo.rsqrt %v2554 : tensor<64x2048x7x7xf32>
    %v2556 = stablehlo.subtract %v2544, %v2549 : tensor<64x2048x7x7xf32>
    %v2557 = stablehlo.multiply %v2556, %v2555 : tensor<64x2048x7x7xf32>
    %v2558 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2559 = stablehlo.reshape %v2521 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2560 = stablehlo.multiply %v2558, %v2559 : tensor<64x2048x7x7xf32>
    %v2561 = stablehlo.subtract %v2560, %v2551 : tensor<64x2048x7x7xf32>
    %v2562 = stablehlo.multiply %v2557, %v2552 : tensor<64x2048x7x7xf32>
    %v2563 = stablehlo.subtract %v2561, %v2562 : tensor<64x2048x7x7xf32>
    %v2564 = stablehlo.multiply %v2555, %v2563 : tensor<64x2048x7x7xf32>
    %v2565 = stablehlo.reshape %v2564 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2566 = stablehlo.reshape %v2565 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2567 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2568 = stablehlo.transpose %v2567, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2569 = stablehlo.convolution(%v2566, %v2568)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2572 = stablehlo.reshape %v2066 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2573 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2574 = stablehlo.compare GT, %v2572, %v2573 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2575 = stablehlo.select %v2574, %v2571, %v2573 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2576 = stablehlo.reshape %v2575 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2577 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2578 = stablehlo.slice %v2051 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2579 = stablehlo.slice %v2051 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2580 = stablehlo.broadcast_in_dim %v2578, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2581 = stablehlo.broadcast_in_dim %v2579, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2582 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2583 = stablehlo.add %v2581, %v2582 : tensor<64x512x7x7xf32>
    %v2584 = stablehlo.rsqrt %v2583 : tensor<64x512x7x7xf32>
    %v2585 = stablehlo.subtract %v2577, %v2580 : tensor<64x512x7x7xf32>
    %v2586 = stablehlo.multiply %v2585, %v2584 : tensor<64x512x7x7xf32>
    %v2587 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2588 = stablehlo.reshape %v2576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2589 = stablehlo.multiply %v2587, %v2588 : tensor<64x512x7x7xf32>
    %v2590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2591 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2592 = stablehlo.reduce(%v2589 init: %v2590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2593 = stablehlo.divide %v2592, %v2591 : tensor<512xf32>
    %v2594 = stablehlo.multiply %v2586, %v2589 : tensor<64x512x7x7xf32>
    %v2595 = stablehlo.reduce(%v2594 init: %v2590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2596 = stablehlo.divide %v2595, %v2591 : tensor<512xf32>
    %v2597 = stablehlo.concatenate %v2593, %v2596, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2598 = stablehlo.concatenate %v2051, %v2597, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g2dst = "stablehlo.all_reduce"(%v2598) ({
    ^bb0(%aras4b1g2dst: tensor<f32>, %arbs4b1g2dst: tensor<f32>):
      %aradds4b1g2dst = stablehlo.add %aras4b1g2dst, %arbs4b1g2dst : tensor<f32>
      stablehlo.return %aradds4b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g2dst = stablehlo.divide %arsums4b1g2dst, %arns4b1g2dst : tensor<2048xf32>
    %v2599 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2600 = stablehlo.slice %armeans4b1g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2601 = stablehlo.slice %armeans4b1g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2602 = stablehlo.slice %armeans4b1g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2603 = stablehlo.slice %armeans4b1g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2604 = stablehlo.broadcast_in_dim %v2600, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2605 = stablehlo.broadcast_in_dim %v2601, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2606 = stablehlo.broadcast_in_dim %v2602, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2607 = stablehlo.broadcast_in_dim %v2603, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2608 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2609 = stablehlo.add %v2605, %v2608 : tensor<64x512x7x7xf32>
    %v2610 = stablehlo.rsqrt %v2609 : tensor<64x512x7x7xf32>
    %v2611 = stablehlo.subtract %v2599, %v2604 : tensor<64x512x7x7xf32>
    %v2612 = stablehlo.multiply %v2611, %v2610 : tensor<64x512x7x7xf32>
    %v2613 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2614 = stablehlo.reshape %v2576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2615 = stablehlo.multiply %v2613, %v2614 : tensor<64x512x7x7xf32>
    %v2616 = stablehlo.subtract %v2615, %v2606 : tensor<64x512x7x7xf32>
    %v2617 = stablehlo.multiply %v2612, %v2607 : tensor<64x512x7x7xf32>
    %v2618 = stablehlo.subtract %v2616, %v2617 : tensor<64x512x7x7xf32>
    %v2619 = stablehlo.multiply %v2610, %v2618 : tensor<64x512x7x7xf32>
    %v2620 = stablehlo.reshape %v2619 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2621 = stablehlo.reshape %v2620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2622 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2623 = stablehlo.transpose %v2622, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2624 = stablehlo.convolution(%v2621, %v2623)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x7x7xf32>
    %v2625 = stablehlo.reshape %v2624 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2626 = stablehlo.reshape %v2625 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2627 = stablehlo.reshape %v2025 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2628 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2629 = stablehlo.compare GT, %v2627, %v2628 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2630 = stablehlo.select %v2629, %v2626, %v2628 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2631 = stablehlo.reshape %v2630 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2632 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2633 = stablehlo.slice %v2010 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2634 = stablehlo.slice %v2010 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2635 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2636 = stablehlo.broadcast_in_dim %v2634, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2637 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2638 = stablehlo.add %v2636, %v2637 : tensor<64x512x7x7xf32>
    %v2639 = stablehlo.rsqrt %v2638 : tensor<64x512x7x7xf32>
    %v2640 = stablehlo.subtract %v2632, %v2635 : tensor<64x512x7x7xf32>
    %v2641 = stablehlo.multiply %v2640, %v2639 : tensor<64x512x7x7xf32>
    %v2642 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2643 = stablehlo.reshape %v2631 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2644 = stablehlo.multiply %v2642, %v2643 : tensor<64x512x7x7xf32>
    %v2645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2646 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2647 = stablehlo.reduce(%v2644 init: %v2645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2648 = stablehlo.divide %v2647, %v2646 : tensor<512xf32>
    %v2649 = stablehlo.multiply %v2641, %v2644 : tensor<64x512x7x7xf32>
    %v2650 = stablehlo.reduce(%v2649 init: %v2645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2651 = stablehlo.divide %v2650, %v2646 : tensor<512xf32>
    %v2652 = stablehlo.concatenate %v2648, %v2651, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2653 = stablehlo.concatenate %v2010, %v2652, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g1dst = "stablehlo.all_reduce"(%v2653) ({
    ^bb0(%aras4b1g1dst: tensor<f32>, %arbs4b1g1dst: tensor<f32>):
      %aradds4b1g1dst = stablehlo.add %aras4b1g1dst, %arbs4b1g1dst : tensor<f32>
      stablehlo.return %aradds4b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g1dst = stablehlo.divide %arsums4b1g1dst, %arns4b1g1dst : tensor<2048xf32>
    %v2654 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2655 = stablehlo.slice %armeans4b1g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2656 = stablehlo.slice %armeans4b1g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2657 = stablehlo.slice %armeans4b1g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2658 = stablehlo.slice %armeans4b1g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2659 = stablehlo.broadcast_in_dim %v2655, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2660 = stablehlo.broadcast_in_dim %v2656, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2661 = stablehlo.broadcast_in_dim %v2657, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2662 = stablehlo.broadcast_in_dim %v2658, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2663 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2664 = stablehlo.add %v2660, %v2663 : tensor<64x512x7x7xf32>
    %v2665 = stablehlo.rsqrt %v2664 : tensor<64x512x7x7xf32>
    %v2666 = stablehlo.subtract %v2654, %v2659 : tensor<64x512x7x7xf32>
    %v2667 = stablehlo.multiply %v2666, %v2665 : tensor<64x512x7x7xf32>
    %v2668 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2669 = stablehlo.reshape %v2631 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2670 = stablehlo.multiply %v2668, %v2669 : tensor<64x512x7x7xf32>
    %v2671 = stablehlo.subtract %v2670, %v2661 : tensor<64x512x7x7xf32>
    %v2672 = stablehlo.multiply %v2667, %v2662 : tensor<64x512x7x7xf32>
    %v2673 = stablehlo.subtract %v2671, %v2672 : tensor<64x512x7x7xf32>
    %v2674 = stablehlo.multiply %v2665, %v2673 : tensor<64x512x7x7xf32>
    %v2675 = stablehlo.reshape %v2674 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2676 = stablehlo.reshape %v2675 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2677 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v2678 = stablehlo.transpose %v2677, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2679 = stablehlo.convolution(%v2676, %v2678)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<2048x512x1x1xf32>) -> tensor<64x2048x7x7xf32>
    %v2680 = stablehlo.reshape %v2679 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2681 = stablehlo.reshape %v2680 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2682 = stablehlo.reshape %v2521 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2683 = stablehlo.add %v2681, %v2682 : tensor<64x2048x7x7xf32>
    %v2684 = stablehlo.reshape %v2683 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2685 = stablehlo.reshape %v1986 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2686 = stablehlo.reshape %v2675 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2687 = stablehlo.transpose %v2685, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2688 = stablehlo.transpose %v2686, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2689 = stablehlo.convolution(%v2687, %v2688)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xf32>, tensor<512x64x7x7xf32>) -> tensor<2048x512x1x1xf32>
    %v2690 = stablehlo.transpose %v2689, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2691 = stablehlo.reshape %v1991 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2692 = stablehlo.slice %v2010 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2693 = stablehlo.slice %v2010 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2694 = stablehlo.broadcast_in_dim %v2692, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2695 = stablehlo.broadcast_in_dim %v2693, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2696 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2697 = stablehlo.add %v2695, %v2696 : tensor<64x512x7x7xf32>
    %v2698 = stablehlo.rsqrt %v2697 : tensor<64x512x7x7xf32>
    %v2699 = stablehlo.subtract %v2691, %v2694 : tensor<64x512x7x7xf32>
    %v2700 = stablehlo.multiply %v2699, %v2698 : tensor<64x512x7x7xf32>
    %v2701 = stablehlo.reshape %v2631 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2702 = stablehlo.multiply %v2701, %v2700 : tensor<64x512x7x7xf32>
    %v2703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2704 = stablehlo.reduce(%v2702 init: %v2703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2705 = stablehlo.reshape %v2631 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2706 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2707 = stablehlo.reduce(%v2705 init: %v2706) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2708 = stablehlo.reshape %v2027 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2709 = stablehlo.reshape %v2620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2710 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2711 = stablehlo.transpose %v2709, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2712 = stablehlo.convolution(%v2710, %v2711)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xf32>, tensor<512x64x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v2713 = stablehlo.transpose %v2712, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2714 = stablehlo.reshape %v2032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2715 = stablehlo.slice %v2051 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2716 = stablehlo.slice %v2051 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2715, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2718 = stablehlo.broadcast_in_dim %v2716, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2719 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2720 = stablehlo.add %v2718, %v2719 : tensor<64x512x7x7xf32>
    %v2721 = stablehlo.rsqrt %v2720 : tensor<64x512x7x7xf32>
    %v2722 = stablehlo.subtract %v2714, %v2717 : tensor<64x512x7x7xf32>
    %v2723 = stablehlo.multiply %v2722, %v2721 : tensor<64x512x7x7xf32>
    %v2724 = stablehlo.reshape %v2576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2725 = stablehlo.multiply %v2724, %v2723 : tensor<64x512x7x7xf32>
    %v2726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2727 = stablehlo.reduce(%v2725 init: %v2726) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2728 = stablehlo.reshape %v2576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2731 = stablehlo.reshape %v2068 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2732 = stablehlo.reshape %v2565 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2733 = stablehlo.transpose %v2731, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2734 = stablehlo.transpose %v2732, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2735 = stablehlo.convolution(%v2733, %v2734)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xf32>, tensor<2048x64x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v2736 = stablehlo.transpose %v2735, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2737 = stablehlo.reshape %v2073 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2738 = stablehlo.slice %v2092 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2739 = stablehlo.slice %v2092 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2740 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2741 = stablehlo.broadcast_in_dim %v2739, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2742 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2743 = stablehlo.add %v2741, %v2742 : tensor<64x2048x7x7xf32>
    %v2744 = stablehlo.rsqrt %v2743 : tensor<64x2048x7x7xf32>
    %v2745 = stablehlo.subtract %v2737, %v2740 : tensor<64x2048x7x7xf32>
    %v2746 = stablehlo.multiply %v2745, %v2744 : tensor<64x2048x7x7xf32>
    %v2747 = stablehlo.reshape %v2521 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2748 = stablehlo.multiply %v2747, %v2746 : tensor<64x2048x7x7xf32>
    %v2749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2750 = stablehlo.reduce(%v2748 init: %v2749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2751 = stablehlo.reshape %v2521 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2753 = stablehlo.reduce(%v2751 init: %v2752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2754 = stablehlo.reshape %v2684 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2755 = stablehlo.reshape %v1984 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2756 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2757 = stablehlo.compare GT, %v2755, %v2756 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2758 = stablehlo.select %v2757, %v2754, %v2756 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2759 = stablehlo.reshape %v2758 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2760 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2761 = stablehlo.slice %v1929 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2762 = stablehlo.slice %v1929 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2763 = stablehlo.broadcast_in_dim %v2761, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2764 = stablehlo.broadcast_in_dim %v2762, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2765 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2766 = stablehlo.add %v2764, %v2765 : tensor<64x2048x7x7xf32>
    %v2767 = stablehlo.rsqrt %v2766 : tensor<64x2048x7x7xf32>
    %v2768 = stablehlo.subtract %v2760, %v2763 : tensor<64x2048x7x7xf32>
    %v2769 = stablehlo.multiply %v2768, %v2767 : tensor<64x2048x7x7xf32>
    %v2770 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2771 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2772 = stablehlo.multiply %v2770, %v2771 : tensor<64x2048x7x7xf32>
    %v2773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2774 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2775 = stablehlo.reduce(%v2772 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2776 = stablehlo.divide %v2775, %v2774 : tensor<2048xf32>
    %v2777 = stablehlo.multiply %v2769, %v2772 : tensor<64x2048x7x7xf32>
    %v2778 = stablehlo.reduce(%v2777 init: %v2773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2779 = stablehlo.divide %v2778, %v2774 : tensor<2048xf32>
    %v2780 = stablehlo.concatenate %v2776, %v2779, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2781 = stablehlo.concatenate %v1929, %v2780, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b0g3dst = "stablehlo.all_reduce"(%v2781) ({
    ^bb0(%aras4b0g3dst: tensor<f32>, %arbs4b0g3dst: tensor<f32>):
      %aradds4b0g3dst = stablehlo.add %aras4b0g3dst, %arbs4b0g3dst : tensor<f32>
      stablehlo.return %aradds4b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b0g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b0g3dst = stablehlo.divide %arsums4b0g3dst, %arns4b0g3dst : tensor<8192xf32>
    %v2782 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2783 = stablehlo.slice %armeans4b0g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2784 = stablehlo.slice %armeans4b0g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2785 = stablehlo.slice %armeans4b0g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2786 = stablehlo.slice %armeans4b0g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2787 = stablehlo.broadcast_in_dim %v2783, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2788 = stablehlo.broadcast_in_dim %v2784, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2785, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2790 = stablehlo.broadcast_in_dim %v2786, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2791 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2792 = stablehlo.add %v2788, %v2791 : tensor<64x2048x7x7xf32>
    %v2793 = stablehlo.rsqrt %v2792 : tensor<64x2048x7x7xf32>
    %v2794 = stablehlo.subtract %v2782, %v2787 : tensor<64x2048x7x7xf32>
    %v2795 = stablehlo.multiply %v2794, %v2793 : tensor<64x2048x7x7xf32>
    %v2796 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2797 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2798 = stablehlo.multiply %v2796, %v2797 : tensor<64x2048x7x7xf32>
    %v2799 = stablehlo.subtract %v2798, %v2789 : tensor<64x2048x7x7xf32>
    %v2800 = stablehlo.multiply %v2795, %v2790 : tensor<64x2048x7x7xf32>
    %v2801 = stablehlo.subtract %v2799, %v2800 : tensor<64x2048x7x7xf32>
    %v2802 = stablehlo.multiply %v2793, %v2801 : tensor<64x2048x7x7xf32>
    %v2803 = stablehlo.reshape %v2802 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2805 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2806 = stablehlo.transpose %v2805, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2807 = stablehlo.convolution(%v2804, %v2806)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<512x2048x1x1xf32>) -> tensor<64x512x7x7xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2810 = stablehlo.reshape %v1903 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2811 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2812 = stablehlo.compare GT, %v2810, %v2811 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2813 = stablehlo.select %v2812, %v2809, %v2811 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2815 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2816 = stablehlo.slice %v1888 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2817 = stablehlo.slice %v1888 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2818 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2817, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2820 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2821 = stablehlo.add %v2819, %v2820 : tensor<64x512x7x7xf32>
    %v2822 = stablehlo.rsqrt %v2821 : tensor<64x512x7x7xf32>
    %v2823 = stablehlo.subtract %v2815, %v2818 : tensor<64x512x7x7xf32>
    %v2824 = stablehlo.multiply %v2823, %v2822 : tensor<64x512x7x7xf32>
    %v2825 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2826 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2827 = stablehlo.multiply %v2825, %v2826 : tensor<64x512x7x7xf32>
    %v2828 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2829 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2830 = stablehlo.reduce(%v2827 init: %v2828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2831 = stablehlo.divide %v2830, %v2829 : tensor<512xf32>
    %v2832 = stablehlo.multiply %v2824, %v2827 : tensor<64x512x7x7xf32>
    %v2833 = stablehlo.reduce(%v2832 init: %v2828) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2834 = stablehlo.divide %v2833, %v2829 : tensor<512xf32>
    %v2835 = stablehlo.concatenate %v2831, %v2834, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2836 = stablehlo.concatenate %v1888, %v2835, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g2dst = "stablehlo.all_reduce"(%v2836) ({
    ^bb0(%aras4b0g2dst: tensor<f32>, %arbs4b0g2dst: tensor<f32>):
      %aradds4b0g2dst = stablehlo.add %aras4b0g2dst, %arbs4b0g2dst : tensor<f32>
      stablehlo.return %aradds4b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g2dst = stablehlo.divide %arsums4b0g2dst, %arns4b0g2dst : tensor<2048xf32>
    %v2837 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2838 = stablehlo.slice %armeans4b0g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2839 = stablehlo.slice %armeans4b0g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2840 = stablehlo.slice %armeans4b0g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2841 = stablehlo.slice %armeans4b0g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2842 = stablehlo.broadcast_in_dim %v2838, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2843 = stablehlo.broadcast_in_dim %v2839, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2844 = stablehlo.broadcast_in_dim %v2840, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2845 = stablehlo.broadcast_in_dim %v2841, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2846 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2847 = stablehlo.add %v2843, %v2846 : tensor<64x512x7x7xf32>
    %v2848 = stablehlo.rsqrt %v2847 : tensor<64x512x7x7xf32>
    %v2849 = stablehlo.subtract %v2837, %v2842 : tensor<64x512x7x7xf32>
    %v2850 = stablehlo.multiply %v2849, %v2848 : tensor<64x512x7x7xf32>
    %v2851 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2852 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2853 = stablehlo.multiply %v2851, %v2852 : tensor<64x512x7x7xf32>
    %v2854 = stablehlo.subtract %v2853, %v2844 : tensor<64x512x7x7xf32>
    %v2855 = stablehlo.multiply %v2850, %v2845 : tensor<64x512x7x7xf32>
    %v2856 = stablehlo.subtract %v2854, %v2855 : tensor<64x512x7x7xf32>
    %v2857 = stablehlo.multiply %v2848, %v2856 : tensor<64x512x7x7xf32>
    %v2858 = stablehlo.reshape %v2857 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2860 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2861 = stablehlo.transpose %v2860, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2862 = stablehlo.convolution(%v2859, %v2861)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xf32>, tensor<512x512x3x3xf32>) -> tensor<64x512x14x14xf32>
    %v2863 = stablehlo.reshape %v2862 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2865 = stablehlo.reshape %v1862 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2866 = stablehlo.constant dense<0.0> : tensor<64x512x14x14xf32>
    %v2867 = stablehlo.compare GT, %v2865, %v2866 : (tensor<64x512x14x14xf32>, tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xi1>
    %v2868 = stablehlo.select %v2867, %v2864, %v2866 : tensor<64x512x14x14xi1>, tensor<64x512x14x14xf32>
    %v2869 = stablehlo.reshape %v2868 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2870 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2871 = stablehlo.slice %v1847 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2872 = stablehlo.slice %v1847 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2873 = stablehlo.broadcast_in_dim %v2871, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2872, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2875 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2876 = stablehlo.add %v2874, %v2875 : tensor<64x512x14x14xf32>
    %v2877 = stablehlo.rsqrt %v2876 : tensor<64x512x14x14xf32>
    %v2878 = stablehlo.subtract %v2870, %v2873 : tensor<64x512x14x14xf32>
    %v2879 = stablehlo.multiply %v2878, %v2877 : tensor<64x512x14x14xf32>
    %v2880 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2881 = stablehlo.reshape %v2869 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2882 = stablehlo.multiply %v2880, %v2881 : tensor<64x512x14x14xf32>
    %v2883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2884 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v2885 = stablehlo.reduce(%v2882 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2886 = stablehlo.divide %v2885, %v2884 : tensor<512xf32>
    %v2887 = stablehlo.multiply %v2879, %v2882 : tensor<64x512x14x14xf32>
    %v2888 = stablehlo.reduce(%v2887 init: %v2883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2889 = stablehlo.divide %v2888, %v2884 : tensor<512xf32>
    %v2890 = stablehlo.concatenate %v2886, %v2889, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2891 = stablehlo.concatenate %v1847, %v2890, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g1dst = "stablehlo.all_reduce"(%v2891) ({
    ^bb0(%aras4b0g1dst: tensor<f32>, %arbs4b0g1dst: tensor<f32>):
      %aradds4b0g1dst = stablehlo.add %aras4b0g1dst, %arbs4b0g1dst : tensor<f32>
      stablehlo.return %aradds4b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g1dst = stablehlo.divide %arsums4b0g1dst, %arns4b0g1dst : tensor<2048xf32>
    %v2892 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2893 = stablehlo.slice %armeans4b0g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2894 = stablehlo.slice %armeans4b0g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2895 = stablehlo.slice %armeans4b0g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2896 = stablehlo.slice %armeans4b0g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2897 = stablehlo.broadcast_in_dim %v2893, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2898 = stablehlo.broadcast_in_dim %v2894, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2899 = stablehlo.broadcast_in_dim %v2895, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2900 = stablehlo.broadcast_in_dim %v2896, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2901 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2902 = stablehlo.add %v2898, %v2901 : tensor<64x512x14x14xf32>
    %v2903 = stablehlo.rsqrt %v2902 : tensor<64x512x14x14xf32>
    %v2904 = stablehlo.subtract %v2892, %v2897 : tensor<64x512x14x14xf32>
    %v2905 = stablehlo.multiply %v2904, %v2903 : tensor<64x512x14x14xf32>
    %v2906 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2907 = stablehlo.reshape %v2869 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2908 = stablehlo.multiply %v2906, %v2907 : tensor<64x512x14x14xf32>
    %v2909 = stablehlo.subtract %v2908, %v2899 : tensor<64x512x14x14xf32>
    %v2910 = stablehlo.multiply %v2905, %v2900 : tensor<64x512x14x14xf32>
    %v2911 = stablehlo.subtract %v2909, %v2910 : tensor<64x512x14x14xf32>
    %v2912 = stablehlo.multiply %v2903, %v2911 : tensor<64x512x14x14xf32>
    %v2913 = stablehlo.reshape %v2912 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2914 = stablehlo.reshape %v2913 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2915 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v2916 = stablehlo.transpose %v2915, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v2917 = stablehlo.convolution(%v2914, %v2916)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xf32>, tensor<1024x512x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v2918 = stablehlo.reshape %v2917 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2919 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2920 = stablehlo.slice %v1968 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2921 = stablehlo.slice %v1968 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2922 = stablehlo.broadcast_in_dim %v2920, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2923 = stablehlo.broadcast_in_dim %v2921, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2924 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2925 = stablehlo.add %v2923, %v2924 : tensor<64x2048x7x7xf32>
    %v2926 = stablehlo.rsqrt %v2925 : tensor<64x2048x7x7xf32>
    %v2927 = stablehlo.subtract %v2919, %v2922 : tensor<64x2048x7x7xf32>
    %v2928 = stablehlo.multiply %v2927, %v2926 : tensor<64x2048x7x7xf32>
    %v2929 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2930 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2931 = stablehlo.multiply %v2929, %v2930 : tensor<64x2048x7x7xf32>
    %v2932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2933 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2934 = stablehlo.reduce(%v2931 init: %v2932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2935 = stablehlo.divide %v2934, %v2933 : tensor<2048xf32>
    %v2936 = stablehlo.multiply %v2928, %v2931 : tensor<64x2048x7x7xf32>
    %v2937 = stablehlo.reduce(%v2936 init: %v2932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2938 = stablehlo.divide %v2937, %v2933 : tensor<2048xf32>
    %v2939 = stablehlo.concatenate %v2935, %v2938, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2940 = stablehlo.concatenate %v1968, %v2939, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b0gpdst = "stablehlo.all_reduce"(%v2940) ({
    ^bb0(%aras4b0gpdst: tensor<f32>, %arbs4b0gpdst: tensor<f32>):
      %aradds4b0gpdst = stablehlo.add %aras4b0gpdst, %arbs4b0gpdst : tensor<f32>
      stablehlo.return %aradds4b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b0gpdst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b0gpdst = stablehlo.divide %arsums4b0gpdst, %arns4b0gpdst : tensor<8192xf32>
    %v2941 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2942 = stablehlo.slice %armeans4b0gpdst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2943 = stablehlo.slice %armeans4b0gpdst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2944 = stablehlo.slice %armeans4b0gpdst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2945 = stablehlo.slice %armeans4b0gpdst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2946 = stablehlo.broadcast_in_dim %v2942, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2947 = stablehlo.broadcast_in_dim %v2943, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2948 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2949 = stablehlo.broadcast_in_dim %v2945, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2950 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2951 = stablehlo.add %v2947, %v2950 : tensor<64x2048x7x7xf32>
    %v2952 = stablehlo.rsqrt %v2951 : tensor<64x2048x7x7xf32>
    %v2953 = stablehlo.subtract %v2941, %v2946 : tensor<64x2048x7x7xf32>
    %v2954 = stablehlo.multiply %v2953, %v2952 : tensor<64x2048x7x7xf32>
    %v2955 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2956 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2957 = stablehlo.multiply %v2955, %v2956 : tensor<64x2048x7x7xf32>
    %v2958 = stablehlo.subtract %v2957, %v2948 : tensor<64x2048x7x7xf32>
    %v2959 = stablehlo.multiply %v2954, %v2949 : tensor<64x2048x7x7xf32>
    %v2960 = stablehlo.subtract %v2958, %v2959 : tensor<64x2048x7x7xf32>
    %v2961 = stablehlo.multiply %v2952, %v2960 : tensor<64x2048x7x7xf32>
    %v2962 = stablehlo.reshape %v2961 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2963 = stablehlo.reshape %v2962 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2964 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2965 = stablehlo.transpose %v2964, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2966 = stablehlo.convolution(%v2963, %v2965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xf32>, tensor<1024x2048x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v2967 = stablehlo.reshape %v2966 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2968 = stablehlo.reshape %v2918 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2969 = stablehlo.reshape %v2967 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2970 = stablehlo.add %v2968, %v2969 : tensor<64x1024x14x14xf32>
    %v2971 = stablehlo.reshape %v2970 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2972 = stablehlo.reshape %v1823 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2973 = stablehlo.reshape %v2913 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2974 = stablehlo.transpose %v2972, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2975 = stablehlo.transpose %v2973, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2976 = stablehlo.convolution(%v2974, %v2975)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<512x64x14x14xf32>) -> tensor<1024x512x1x1xf32>
    %v2977 = stablehlo.transpose %v2976, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2978 = stablehlo.reshape %v1828 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2979 = stablehlo.slice %v1847 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2980 = stablehlo.slice %v1847 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2981 = stablehlo.broadcast_in_dim %v2979, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2983 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2984 = stablehlo.add %v2982, %v2983 : tensor<64x512x14x14xf32>
    %v2985 = stablehlo.rsqrt %v2984 : tensor<64x512x14x14xf32>
    %v2986 = stablehlo.subtract %v2978, %v2981 : tensor<64x512x14x14xf32>
    %v2987 = stablehlo.multiply %v2986, %v2985 : tensor<64x512x14x14xf32>
    %v2988 = stablehlo.reshape %v2869 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2989 = stablehlo.multiply %v2988, %v2987 : tensor<64x512x14x14xf32>
    %v2990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2991 = stablehlo.reduce(%v2989 init: %v2990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2992 = stablehlo.reshape %v2869 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2994 = stablehlo.reduce(%v2992 init: %v2993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2995 = stablehlo.reshape %v1864 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2996 = stablehlo.reshape %v2858 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2997 = stablehlo.transpose %v2995, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2999 = stablehlo.transpose %v2996, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v3000 = stablehlo.convolution(%v2997, %v2999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x14x14xf32>, tensor<512x64x7x7xf32>) -> tensor<512x512x3x3xf32>
    %v3001 = stablehlo.transpose %v3000, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v3002 = stablehlo.reshape %v1869 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3003 = stablehlo.slice %v1888 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3004 = stablehlo.slice %v1888 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3005 = stablehlo.broadcast_in_dim %v3003, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3006 = stablehlo.broadcast_in_dim %v3004, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3007 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v3008 = stablehlo.add %v3006, %v3007 : tensor<64x512x7x7xf32>
    %v3009 = stablehlo.rsqrt %v3008 : tensor<64x512x7x7xf32>
    %v3010 = stablehlo.subtract %v3002, %v3005 : tensor<64x512x7x7xf32>
    %v3011 = stablehlo.multiply %v3010, %v3009 : tensor<64x512x7x7xf32>
    %v3012 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3013 = stablehlo.multiply %v3012, %v3011 : tensor<64x512x7x7xf32>
    %v3014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3015 = stablehlo.reduce(%v3013 init: %v3014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3016 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3017 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3018 = stablehlo.reduce(%v3016 init: %v3017) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3019 = stablehlo.reshape %v1905 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3020 = stablehlo.reshape %v2803 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3021 = stablehlo.transpose %v3019, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v3022 = stablehlo.transpose %v3020, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v3023 = stablehlo.convolution(%v3021, %v3022)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xf32>, tensor<2048x64x7x7xf32>) -> tensor<512x2048x1x1xf32>
    %v3024 = stablehlo.transpose %v3023, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v3025 = stablehlo.reshape %v1910 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3026 = stablehlo.slice %v1929 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3027 = stablehlo.slice %v1929 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3028 = stablehlo.broadcast_in_dim %v3026, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3029 = stablehlo.broadcast_in_dim %v3027, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3030 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3031 = stablehlo.add %v3029, %v3030 : tensor<64x2048x7x7xf32>
    %v3032 = stablehlo.rsqrt %v3031 : tensor<64x2048x7x7xf32>
    %v3033 = stablehlo.subtract %v3025, %v3028 : tensor<64x2048x7x7xf32>
    %v3034 = stablehlo.multiply %v3033, %v3032 : tensor<64x2048x7x7xf32>
    %v3035 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3036 = stablehlo.multiply %v3035, %v3034 : tensor<64x2048x7x7xf32>
    %v3037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3038 = stablehlo.reduce(%v3036 init: %v3037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3039 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3041 = stablehlo.reduce(%v3039 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3042 = stablehlo.reshape %v1823 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3043 = stablehlo.reshape %v2962 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3044 = stablehlo.transpose %v3042, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3046 = stablehlo.transpose %v3043, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v3045 = stablehlo.slice %v3044 [0:1024, 0:64, 0:13, 0:13] : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x13x13xf32>
    %v3047 = stablehlo.convolution(%v3045, %v3046)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x13x13xf32>, tensor<2048x64x7x7xf32>) -> tensor<1024x2048x1x1xf32>
    %v3048 = stablehlo.transpose %v3047, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v3049 = stablehlo.reshape %v1949 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3050 = stablehlo.slice %v1968 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3051 = stablehlo.slice %v1968 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3052 = stablehlo.broadcast_in_dim %v3050, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3053 = stablehlo.broadcast_in_dim %v3051, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3054 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3055 = stablehlo.add %v3053, %v3054 : tensor<64x2048x7x7xf32>
    %v3056 = stablehlo.rsqrt %v3055 : tensor<64x2048x7x7xf32>
    %v3057 = stablehlo.subtract %v3049, %v3052 : tensor<64x2048x7x7xf32>
    %v3058 = stablehlo.multiply %v3057, %v3056 : tensor<64x2048x7x7xf32>
    %v3059 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3060 = stablehlo.multiply %v3059, %v3058 : tensor<64x2048x7x7xf32>
    %v3061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3062 = stablehlo.reduce(%v3060 init: %v3061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3063 = stablehlo.reshape %v2759 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3064 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3065 = stablehlo.reduce(%v3063 init: %v3064) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3066 = stablehlo.reshape %v2971 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3067 = stablehlo.reshape %v1819 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3068 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3069 = stablehlo.compare GT, %v3067, %v3068 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3070 = stablehlo.select %v3069, %v3066, %v3068 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3071 = stablehlo.reshape %v3070 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3072 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3073 = stablehlo.slice %v1800 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3074 = stablehlo.slice %v1800 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3075 = stablehlo.broadcast_in_dim %v3073, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3074, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3077 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3078 = stablehlo.add %v3076, %v3077 : tensor<64x1024x14x14xf32>
    %v3079 = stablehlo.rsqrt %v3078 : tensor<64x1024x14x14xf32>
    %v3080 = stablehlo.subtract %v3072, %v3075 : tensor<64x1024x14x14xf32>
    %v3081 = stablehlo.multiply %v3080, %v3079 : tensor<64x1024x14x14xf32>
    %v3082 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3083 = stablehlo.reshape %v3071 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3084 = stablehlo.multiply %v3082, %v3083 : tensor<64x1024x14x14xf32>
    %v3085 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3086 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3087 = stablehlo.reduce(%v3084 init: %v3085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3088 = stablehlo.divide %v3087, %v3086 : tensor<1024xf32>
    %v3089 = stablehlo.multiply %v3081, %v3084 : tensor<64x1024x14x14xf32>
    %v3090 = stablehlo.reduce(%v3089 init: %v3085) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3091 = stablehlo.divide %v3090, %v3086 : tensor<1024xf32>
    %v3092 = stablehlo.concatenate %v3088, %v3091, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3093 = stablehlo.concatenate %v1800, %v3092, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b5g3dst = "stablehlo.all_reduce"(%v3093) ({
    ^bb0(%aras3b5g3dst: tensor<f32>, %arbs3b5g3dst: tensor<f32>):
      %aradds3b5g3dst = stablehlo.add %aras3b5g3dst, %arbs3b5g3dst : tensor<f32>
      stablehlo.return %aradds3b5g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b5g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b5g3dst = stablehlo.divide %arsums3b5g3dst, %arns3b5g3dst : tensor<4096xf32>
    %v3094 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3095 = stablehlo.slice %armeans3b5g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3096 = stablehlo.slice %armeans3b5g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3097 = stablehlo.slice %armeans3b5g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3098 = stablehlo.slice %armeans3b5g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3099 = stablehlo.broadcast_in_dim %v3095, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3096, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3101 = stablehlo.broadcast_in_dim %v3097, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3098, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3103 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3104 = stablehlo.add %v3100, %v3103 : tensor<64x1024x14x14xf32>
    %v3105 = stablehlo.rsqrt %v3104 : tensor<64x1024x14x14xf32>
    %v3106 = stablehlo.subtract %v3094, %v3099 : tensor<64x1024x14x14xf32>
    %v3107 = stablehlo.multiply %v3106, %v3105 : tensor<64x1024x14x14xf32>
    %v3108 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3109 = stablehlo.reshape %v3071 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3110 = stablehlo.multiply %v3108, %v3109 : tensor<64x1024x14x14xf32>
    %v3111 = stablehlo.subtract %v3110, %v3101 : tensor<64x1024x14x14xf32>
    %v3112 = stablehlo.multiply %v3107, %v3102 : tensor<64x1024x14x14xf32>
    %v3113 = stablehlo.subtract %v3111, %v3112 : tensor<64x1024x14x14xf32>
    %v3114 = stablehlo.multiply %v3105, %v3113 : tensor<64x1024x14x14xf32>
    %v3115 = stablehlo.reshape %v3114 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3116 = stablehlo.reshape %v3115 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3117 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3118 = stablehlo.transpose %v3117, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3119 = stablehlo.convolution(%v3116, %v3118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v3120 = stablehlo.reshape %v3119 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3121 = stablehlo.reshape %v3120 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3122 = stablehlo.reshape %v1774 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3123 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3124 = stablehlo.compare GT, %v3122, %v3123 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3125 = stablehlo.select %v3124, %v3121, %v3123 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3126 = stablehlo.reshape %v3125 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3127 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3128 = stablehlo.slice %v1759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3129 = stablehlo.slice %v1759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3130 = stablehlo.broadcast_in_dim %v3128, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3131 = stablehlo.broadcast_in_dim %v3129, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3132 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3133 = stablehlo.add %v3131, %v3132 : tensor<64x256x14x14xf32>
    %v3134 = stablehlo.rsqrt %v3133 : tensor<64x256x14x14xf32>
    %v3135 = stablehlo.subtract %v3127, %v3130 : tensor<64x256x14x14xf32>
    %v3136 = stablehlo.multiply %v3135, %v3134 : tensor<64x256x14x14xf32>
    %v3137 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3138 = stablehlo.reshape %v3126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3139 = stablehlo.multiply %v3137, %v3138 : tensor<64x256x14x14xf32>
    %v3140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3141 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3142 = stablehlo.reduce(%v3139 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3143 = stablehlo.divide %v3142, %v3141 : tensor<256xf32>
    %v3144 = stablehlo.multiply %v3136, %v3139 : tensor<64x256x14x14xf32>
    %v3145 = stablehlo.reduce(%v3144 init: %v3140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3146 = stablehlo.divide %v3145, %v3141 : tensor<256xf32>
    %v3147 = stablehlo.concatenate %v3143, %v3146, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3148 = stablehlo.concatenate %v1759, %v3147, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b5g2dst = "stablehlo.all_reduce"(%v3148) ({
    ^bb0(%aras3b5g2dst: tensor<f32>, %arbs3b5g2dst: tensor<f32>):
      %aradds3b5g2dst = stablehlo.add %aras3b5g2dst, %arbs3b5g2dst : tensor<f32>
      stablehlo.return %aradds3b5g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g2dst = stablehlo.divide %arsums3b5g2dst, %arns3b5g2dst : tensor<1024xf32>
    %v3149 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3150 = stablehlo.slice %armeans3b5g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3151 = stablehlo.slice %armeans3b5g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3152 = stablehlo.slice %armeans3b5g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3153 = stablehlo.slice %armeans3b5g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3154 = stablehlo.broadcast_in_dim %v3150, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3151, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3156 = stablehlo.broadcast_in_dim %v3152, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3157 = stablehlo.broadcast_in_dim %v3153, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3158 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3159 = stablehlo.add %v3155, %v3158 : tensor<64x256x14x14xf32>
    %v3160 = stablehlo.rsqrt %v3159 : tensor<64x256x14x14xf32>
    %v3161 = stablehlo.subtract %v3149, %v3154 : tensor<64x256x14x14xf32>
    %v3162 = stablehlo.multiply %v3161, %v3160 : tensor<64x256x14x14xf32>
    %v3163 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3164 = stablehlo.reshape %v3126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3165 = stablehlo.multiply %v3163, %v3164 : tensor<64x256x14x14xf32>
    %v3166 = stablehlo.subtract %v3165, %v3156 : tensor<64x256x14x14xf32>
    %v3167 = stablehlo.multiply %v3162, %v3157 : tensor<64x256x14x14xf32>
    %v3168 = stablehlo.subtract %v3166, %v3167 : tensor<64x256x14x14xf32>
    %v3169 = stablehlo.multiply %v3160, %v3168 : tensor<64x256x14x14xf32>
    %v3170 = stablehlo.reshape %v3169 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3171 = stablehlo.reshape %v3170 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3172 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3173 = stablehlo.transpose %v3172, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3174 = stablehlo.convolution(%v3171, %v3173)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v3175 = stablehlo.reshape %v3174 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3176 = stablehlo.reshape %v3175 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3177 = stablehlo.reshape %v1733 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3178 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3179 = stablehlo.compare GT, %v3177, %v3178 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3180 = stablehlo.select %v3179, %v3176, %v3178 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3181 = stablehlo.reshape %v3180 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3182 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3183 = stablehlo.slice %v1718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3184 = stablehlo.slice %v1718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3185 = stablehlo.broadcast_in_dim %v3183, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3186 = stablehlo.broadcast_in_dim %v3184, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3187 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3188 = stablehlo.add %v3186, %v3187 : tensor<64x256x14x14xf32>
    %v3189 = stablehlo.rsqrt %v3188 : tensor<64x256x14x14xf32>
    %v3190 = stablehlo.subtract %v3182, %v3185 : tensor<64x256x14x14xf32>
    %v3191 = stablehlo.multiply %v3190, %v3189 : tensor<64x256x14x14xf32>
    %v3192 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3193 = stablehlo.reshape %v3181 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3194 = stablehlo.multiply %v3192, %v3193 : tensor<64x256x14x14xf32>
    %v3195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3196 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3197 = stablehlo.reduce(%v3194 init: %v3195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3198 = stablehlo.divide %v3197, %v3196 : tensor<256xf32>
    %v3199 = stablehlo.multiply %v3191, %v3194 : tensor<64x256x14x14xf32>
    %v3200 = stablehlo.reduce(%v3199 init: %v3195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3201 = stablehlo.divide %v3200, %v3196 : tensor<256xf32>
    %v3202 = stablehlo.concatenate %v3198, %v3201, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3203 = stablehlo.concatenate %v1718, %v3202, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b5g1dst = "stablehlo.all_reduce"(%v3203) ({
    ^bb0(%aras3b5g1dst: tensor<f32>, %arbs3b5g1dst: tensor<f32>):
      %aradds3b5g1dst = stablehlo.add %aras3b5g1dst, %arbs3b5g1dst : tensor<f32>
      stablehlo.return %aradds3b5g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g1dst = stablehlo.divide %arsums3b5g1dst, %arns3b5g1dst : tensor<1024xf32>
    %v3204 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3205 = stablehlo.slice %armeans3b5g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3206 = stablehlo.slice %armeans3b5g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3207 = stablehlo.slice %armeans3b5g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3208 = stablehlo.slice %armeans3b5g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3209 = stablehlo.broadcast_in_dim %v3205, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3210 = stablehlo.broadcast_in_dim %v3206, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3211 = stablehlo.broadcast_in_dim %v3207, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3212 = stablehlo.broadcast_in_dim %v3208, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3213 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3214 = stablehlo.add %v3210, %v3213 : tensor<64x256x14x14xf32>
    %v3215 = stablehlo.rsqrt %v3214 : tensor<64x256x14x14xf32>
    %v3216 = stablehlo.subtract %v3204, %v3209 : tensor<64x256x14x14xf32>
    %v3217 = stablehlo.multiply %v3216, %v3215 : tensor<64x256x14x14xf32>
    %v3218 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3219 = stablehlo.reshape %v3181 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3220 = stablehlo.multiply %v3218, %v3219 : tensor<64x256x14x14xf32>
    %v3221 = stablehlo.subtract %v3220, %v3211 : tensor<64x256x14x14xf32>
    %v3222 = stablehlo.multiply %v3217, %v3212 : tensor<64x256x14x14xf32>
    %v3223 = stablehlo.subtract %v3221, %v3222 : tensor<64x256x14x14xf32>
    %v3224 = stablehlo.multiply %v3215, %v3223 : tensor<64x256x14x14xf32>
    %v3225 = stablehlo.reshape %v3224 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3226 = stablehlo.reshape %v3225 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3227 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3228 = stablehlo.transpose %v3227, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3229 = stablehlo.convolution(%v3226, %v3228)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v3230 = stablehlo.reshape %v3229 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3231 = stablehlo.reshape %v3230 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3232 = stablehlo.reshape %v3071 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3233 = stablehlo.add %v3231, %v3232 : tensor<64x1024x14x14xf32>
    %v3234 = stablehlo.reshape %v3233 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3235 = stablehlo.reshape %v1694 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3236 = stablehlo.reshape %v3225 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3237 = stablehlo.transpose %v3235, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3238 = stablehlo.transpose %v3236, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3239 = stablehlo.convolution(%v3237, %v3238)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3240 = stablehlo.transpose %v3239, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3241 = stablehlo.reshape %v1699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3242 = stablehlo.slice %v1718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3243 = stablehlo.slice %v1718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3244 = stablehlo.broadcast_in_dim %v3242, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3245 = stablehlo.broadcast_in_dim %v3243, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3246 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3247 = stablehlo.add %v3245, %v3246 : tensor<64x256x14x14xf32>
    %v3248 = stablehlo.rsqrt %v3247 : tensor<64x256x14x14xf32>
    %v3249 = stablehlo.subtract %v3241, %v3244 : tensor<64x256x14x14xf32>
    %v3250 = stablehlo.multiply %v3249, %v3248 : tensor<64x256x14x14xf32>
    %v3251 = stablehlo.reshape %v3181 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3252 = stablehlo.multiply %v3251, %v3250 : tensor<64x256x14x14xf32>
    %v3253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3254 = stablehlo.reduce(%v3252 init: %v3253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3255 = stablehlo.reshape %v3181 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3257 = stablehlo.reduce(%v3255 init: %v3256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3258 = stablehlo.reshape %v1735 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3259 = stablehlo.reshape %v3170 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3260 = stablehlo.transpose %v3258, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3261 = stablehlo.transpose %v3259, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3262 = stablehlo.convolution(%v3260, %v3261)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3263 = stablehlo.transpose %v3262, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3264 = stablehlo.reshape %v1740 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3265 = stablehlo.slice %v1759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3266 = stablehlo.slice %v1759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3267 = stablehlo.broadcast_in_dim %v3265, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3266, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3269 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3270 = stablehlo.add %v3268, %v3269 : tensor<64x256x14x14xf32>
    %v3271 = stablehlo.rsqrt %v3270 : tensor<64x256x14x14xf32>
    %v3272 = stablehlo.subtract %v3264, %v3267 : tensor<64x256x14x14xf32>
    %v3273 = stablehlo.multiply %v3272, %v3271 : tensor<64x256x14x14xf32>
    %v3274 = stablehlo.reshape %v3126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3275 = stablehlo.multiply %v3274, %v3273 : tensor<64x256x14x14xf32>
    %v3276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3277 = stablehlo.reduce(%v3275 init: %v3276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3278 = stablehlo.reshape %v3126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3280 = stablehlo.reduce(%v3278 init: %v3279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3281 = stablehlo.reshape %v1776 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3282 = stablehlo.reshape %v3115 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3283 = stablehlo.transpose %v3281, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3284 = stablehlo.transpose %v3282, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3285 = stablehlo.convolution(%v3283, %v3284)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3286 = stablehlo.transpose %v3285, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3287 = stablehlo.reshape %v1781 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3288 = stablehlo.slice %v1800 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3289 = stablehlo.slice %v1800 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3290 = stablehlo.broadcast_in_dim %v3288, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3291 = stablehlo.broadcast_in_dim %v3289, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3292 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3293 = stablehlo.add %v3291, %v3292 : tensor<64x1024x14x14xf32>
    %v3294 = stablehlo.rsqrt %v3293 : tensor<64x1024x14x14xf32>
    %v3295 = stablehlo.subtract %v3287, %v3290 : tensor<64x1024x14x14xf32>
    %v3296 = stablehlo.multiply %v3295, %v3294 : tensor<64x1024x14x14xf32>
    %v3297 = stablehlo.reshape %v3071 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3298 = stablehlo.multiply %v3297, %v3296 : tensor<64x1024x14x14xf32>
    %v3299 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3300 = stablehlo.reduce(%v3298 init: %v3299) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3301 = stablehlo.reshape %v3071 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3302 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3303 = stablehlo.reduce(%v3301 init: %v3302) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3304 = stablehlo.reshape %v3234 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3305 = stablehlo.reshape %v1690 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3306 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3307 = stablehlo.compare GT, %v3305, %v3306 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3308 = stablehlo.select %v3307, %v3304, %v3306 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3309 = stablehlo.reshape %v3308 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3310 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3311 = stablehlo.slice %v1671 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3312 = stablehlo.slice %v1671 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3314 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3315 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3316 = stablehlo.add %v3314, %v3315 : tensor<64x1024x14x14xf32>
    %v3317 = stablehlo.rsqrt %v3316 : tensor<64x1024x14x14xf32>
    %v3318 = stablehlo.subtract %v3310, %v3313 : tensor<64x1024x14x14xf32>
    %v3319 = stablehlo.multiply %v3318, %v3317 : tensor<64x1024x14x14xf32>
    %v3320 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3321 = stablehlo.reshape %v3309 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3322 = stablehlo.multiply %v3320, %v3321 : tensor<64x1024x14x14xf32>
    %v3323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3324 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3325 = stablehlo.reduce(%v3322 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3326 = stablehlo.divide %v3325, %v3324 : tensor<1024xf32>
    %v3327 = stablehlo.multiply %v3319, %v3322 : tensor<64x1024x14x14xf32>
    %v3328 = stablehlo.reduce(%v3327 init: %v3323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3329 = stablehlo.divide %v3328, %v3324 : tensor<1024xf32>
    %v3330 = stablehlo.concatenate %v3326, %v3329, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3331 = stablehlo.concatenate %v1671, %v3330, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b4g3dst = "stablehlo.all_reduce"(%v3331) ({
    ^bb0(%aras3b4g3dst: tensor<f32>, %arbs3b4g3dst: tensor<f32>):
      %aradds3b4g3dst = stablehlo.add %aras3b4g3dst, %arbs3b4g3dst : tensor<f32>
      stablehlo.return %aradds3b4g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b4g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b4g3dst = stablehlo.divide %arsums3b4g3dst, %arns3b4g3dst : tensor<4096xf32>
    %v3332 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3333 = stablehlo.slice %armeans3b4g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3334 = stablehlo.slice %armeans3b4g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3335 = stablehlo.slice %armeans3b4g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3336 = stablehlo.slice %armeans3b4g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3337 = stablehlo.broadcast_in_dim %v3333, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3338 = stablehlo.broadcast_in_dim %v3334, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3339 = stablehlo.broadcast_in_dim %v3335, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3340 = stablehlo.broadcast_in_dim %v3336, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3341 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3342 = stablehlo.add %v3338, %v3341 : tensor<64x1024x14x14xf32>
    %v3343 = stablehlo.rsqrt %v3342 : tensor<64x1024x14x14xf32>
    %v3344 = stablehlo.subtract %v3332, %v3337 : tensor<64x1024x14x14xf32>
    %v3345 = stablehlo.multiply %v3344, %v3343 : tensor<64x1024x14x14xf32>
    %v3346 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3347 = stablehlo.reshape %v3309 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3348 = stablehlo.multiply %v3346, %v3347 : tensor<64x1024x14x14xf32>
    %v3349 = stablehlo.subtract %v3348, %v3339 : tensor<64x1024x14x14xf32>
    %v3350 = stablehlo.multiply %v3345, %v3340 : tensor<64x1024x14x14xf32>
    %v3351 = stablehlo.subtract %v3349, %v3350 : tensor<64x1024x14x14xf32>
    %v3352 = stablehlo.multiply %v3343, %v3351 : tensor<64x1024x14x14xf32>
    %v3353 = stablehlo.reshape %v3352 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3354 = stablehlo.reshape %v3353 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3355 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3356 = stablehlo.transpose %v3355, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3357 = stablehlo.convolution(%v3354, %v3356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v3358 = stablehlo.reshape %v3357 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3359 = stablehlo.reshape %v3358 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3360 = stablehlo.reshape %v1645 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3361 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3362 = stablehlo.compare GT, %v3360, %v3361 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3363 = stablehlo.select %v3362, %v3359, %v3361 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3364 = stablehlo.reshape %v3363 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3365 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3366 = stablehlo.slice %v1630 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3367 = stablehlo.slice %v1630 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3368 = stablehlo.broadcast_in_dim %v3366, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3369 = stablehlo.broadcast_in_dim %v3367, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3370 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3371 = stablehlo.add %v3369, %v3370 : tensor<64x256x14x14xf32>
    %v3372 = stablehlo.rsqrt %v3371 : tensor<64x256x14x14xf32>
    %v3373 = stablehlo.subtract %v3365, %v3368 : tensor<64x256x14x14xf32>
    %v3374 = stablehlo.multiply %v3373, %v3372 : tensor<64x256x14x14xf32>
    %v3375 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3376 = stablehlo.reshape %v3364 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3377 = stablehlo.multiply %v3375, %v3376 : tensor<64x256x14x14xf32>
    %v3378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3379 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3380 = stablehlo.reduce(%v3377 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3381 = stablehlo.divide %v3380, %v3379 : tensor<256xf32>
    %v3382 = stablehlo.multiply %v3374, %v3377 : tensor<64x256x14x14xf32>
    %v3383 = stablehlo.reduce(%v3382 init: %v3378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3384 = stablehlo.divide %v3383, %v3379 : tensor<256xf32>
    %v3385 = stablehlo.concatenate %v3381, %v3384, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3386 = stablehlo.concatenate %v1630, %v3385, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g2dst = "stablehlo.all_reduce"(%v3386) ({
    ^bb0(%aras3b4g2dst: tensor<f32>, %arbs3b4g2dst: tensor<f32>):
      %aradds3b4g2dst = stablehlo.add %aras3b4g2dst, %arbs3b4g2dst : tensor<f32>
      stablehlo.return %aradds3b4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g2dst = stablehlo.divide %arsums3b4g2dst, %arns3b4g2dst : tensor<1024xf32>
    %v3387 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3388 = stablehlo.slice %armeans3b4g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3389 = stablehlo.slice %armeans3b4g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3390 = stablehlo.slice %armeans3b4g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3391 = stablehlo.slice %armeans3b4g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3392 = stablehlo.broadcast_in_dim %v3388, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3393 = stablehlo.broadcast_in_dim %v3389, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3394 = stablehlo.broadcast_in_dim %v3390, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3395 = stablehlo.broadcast_in_dim %v3391, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3396 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3397 = stablehlo.add %v3393, %v3396 : tensor<64x256x14x14xf32>
    %v3398 = stablehlo.rsqrt %v3397 : tensor<64x256x14x14xf32>
    %v3399 = stablehlo.subtract %v3387, %v3392 : tensor<64x256x14x14xf32>
    %v3400 = stablehlo.multiply %v3399, %v3398 : tensor<64x256x14x14xf32>
    %v3401 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3402 = stablehlo.reshape %v3364 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3403 = stablehlo.multiply %v3401, %v3402 : tensor<64x256x14x14xf32>
    %v3404 = stablehlo.subtract %v3403, %v3394 : tensor<64x256x14x14xf32>
    %v3405 = stablehlo.multiply %v3400, %v3395 : tensor<64x256x14x14xf32>
    %v3406 = stablehlo.subtract %v3404, %v3405 : tensor<64x256x14x14xf32>
    %v3407 = stablehlo.multiply %v3398, %v3406 : tensor<64x256x14x14xf32>
    %v3408 = stablehlo.reshape %v3407 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3409 = stablehlo.reshape %v3408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3410 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3411 = stablehlo.transpose %v3410, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3412 = stablehlo.convolution(%v3409, %v3411)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v3413 = stablehlo.reshape %v3412 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3414 = stablehlo.reshape %v3413 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3415 = stablehlo.reshape %v1604 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3416 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3417 = stablehlo.compare GT, %v3415, %v3416 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3418 = stablehlo.select %v3417, %v3414, %v3416 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3419 = stablehlo.reshape %v3418 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3420 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3421 = stablehlo.slice %v1589 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3422 = stablehlo.slice %v1589 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3423 = stablehlo.broadcast_in_dim %v3421, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3424 = stablehlo.broadcast_in_dim %v3422, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3425 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3426 = stablehlo.add %v3424, %v3425 : tensor<64x256x14x14xf32>
    %v3427 = stablehlo.rsqrt %v3426 : tensor<64x256x14x14xf32>
    %v3428 = stablehlo.subtract %v3420, %v3423 : tensor<64x256x14x14xf32>
    %v3429 = stablehlo.multiply %v3428, %v3427 : tensor<64x256x14x14xf32>
    %v3430 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3431 = stablehlo.reshape %v3419 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3432 = stablehlo.multiply %v3430, %v3431 : tensor<64x256x14x14xf32>
    %v3433 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3434 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3435 = stablehlo.reduce(%v3432 init: %v3433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3436 = stablehlo.divide %v3435, %v3434 : tensor<256xf32>
    %v3437 = stablehlo.multiply %v3429, %v3432 : tensor<64x256x14x14xf32>
    %v3438 = stablehlo.reduce(%v3437 init: %v3433) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3439 = stablehlo.divide %v3438, %v3434 : tensor<256xf32>
    %v3440 = stablehlo.concatenate %v3436, %v3439, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3441 = stablehlo.concatenate %v1589, %v3440, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g1dst = "stablehlo.all_reduce"(%v3441) ({
    ^bb0(%aras3b4g1dst: tensor<f32>, %arbs3b4g1dst: tensor<f32>):
      %aradds3b4g1dst = stablehlo.add %aras3b4g1dst, %arbs3b4g1dst : tensor<f32>
      stablehlo.return %aradds3b4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g1dst = stablehlo.divide %arsums3b4g1dst, %arns3b4g1dst : tensor<1024xf32>
    %v3442 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3443 = stablehlo.slice %armeans3b4g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3444 = stablehlo.slice %armeans3b4g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3445 = stablehlo.slice %armeans3b4g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3446 = stablehlo.slice %armeans3b4g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3447 = stablehlo.broadcast_in_dim %v3443, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3444, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3449 = stablehlo.broadcast_in_dim %v3445, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3446, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3451 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3452 = stablehlo.add %v3448, %v3451 : tensor<64x256x14x14xf32>
    %v3453 = stablehlo.rsqrt %v3452 : tensor<64x256x14x14xf32>
    %v3454 = stablehlo.subtract %v3442, %v3447 : tensor<64x256x14x14xf32>
    %v3455 = stablehlo.multiply %v3454, %v3453 : tensor<64x256x14x14xf32>
    %v3456 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3457 = stablehlo.reshape %v3419 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3458 = stablehlo.multiply %v3456, %v3457 : tensor<64x256x14x14xf32>
    %v3459 = stablehlo.subtract %v3458, %v3449 : tensor<64x256x14x14xf32>
    %v3460 = stablehlo.multiply %v3455, %v3450 : tensor<64x256x14x14xf32>
    %v3461 = stablehlo.subtract %v3459, %v3460 : tensor<64x256x14x14xf32>
    %v3462 = stablehlo.multiply %v3453, %v3461 : tensor<64x256x14x14xf32>
    %v3463 = stablehlo.reshape %v3462 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3464 = stablehlo.reshape %v3463 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3465 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3466 = stablehlo.transpose %v3465, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3467 = stablehlo.convolution(%v3464, %v3466)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v3468 = stablehlo.reshape %v3467 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3469 = stablehlo.reshape %v3468 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3470 = stablehlo.reshape %v3309 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3471 = stablehlo.add %v3469, %v3470 : tensor<64x1024x14x14xf32>
    %v3472 = stablehlo.reshape %v3471 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3473 = stablehlo.reshape %v1565 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3474 = stablehlo.reshape %v3463 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3475 = stablehlo.transpose %v3473, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3476 = stablehlo.transpose %v3474, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3477 = stablehlo.convolution(%v3475, %v3476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3478 = stablehlo.transpose %v3477, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3479 = stablehlo.reshape %v1570 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3480 = stablehlo.slice %v1589 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3481 = stablehlo.slice %v1589 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3484 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3485 = stablehlo.add %v3483, %v3484 : tensor<64x256x14x14xf32>
    %v3486 = stablehlo.rsqrt %v3485 : tensor<64x256x14x14xf32>
    %v3487 = stablehlo.subtract %v3479, %v3482 : tensor<64x256x14x14xf32>
    %v3488 = stablehlo.multiply %v3487, %v3486 : tensor<64x256x14x14xf32>
    %v3489 = stablehlo.reshape %v3419 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3490 = stablehlo.multiply %v3489, %v3488 : tensor<64x256x14x14xf32>
    %v3491 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3492 = stablehlo.reduce(%v3490 init: %v3491) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3493 = stablehlo.reshape %v3419 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3495 = stablehlo.reduce(%v3493 init: %v3494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3496 = stablehlo.reshape %v1606 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3497 = stablehlo.reshape %v3408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3498 = stablehlo.transpose %v3496, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3499 = stablehlo.transpose %v3497, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3500 = stablehlo.convolution(%v3498, %v3499)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3501 = stablehlo.transpose %v3500, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3502 = stablehlo.reshape %v1611 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3503 = stablehlo.slice %v1630 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3504 = stablehlo.slice %v1630 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3505 = stablehlo.broadcast_in_dim %v3503, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3506 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3507 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3508 = stablehlo.add %v3506, %v3507 : tensor<64x256x14x14xf32>
    %v3509 = stablehlo.rsqrt %v3508 : tensor<64x256x14x14xf32>
    %v3510 = stablehlo.subtract %v3502, %v3505 : tensor<64x256x14x14xf32>
    %v3511 = stablehlo.multiply %v3510, %v3509 : tensor<64x256x14x14xf32>
    %v3512 = stablehlo.reshape %v3364 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3513 = stablehlo.multiply %v3512, %v3511 : tensor<64x256x14x14xf32>
    %v3514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3515 = stablehlo.reduce(%v3513 init: %v3514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3516 = stablehlo.reshape %v3364 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3518 = stablehlo.reduce(%v3516 init: %v3517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3519 = stablehlo.reshape %v1647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3520 = stablehlo.reshape %v3353 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3521 = stablehlo.transpose %v3519, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3522 = stablehlo.transpose %v3520, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3523 = stablehlo.convolution(%v3521, %v3522)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3524 = stablehlo.transpose %v3523, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3525 = stablehlo.reshape %v1652 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3526 = stablehlo.slice %v1671 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3527 = stablehlo.slice %v1671 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3528 = stablehlo.broadcast_in_dim %v3526, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3527, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3530 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3531 = stablehlo.add %v3529, %v3530 : tensor<64x1024x14x14xf32>
    %v3532 = stablehlo.rsqrt %v3531 : tensor<64x1024x14x14xf32>
    %v3533 = stablehlo.subtract %v3525, %v3528 : tensor<64x1024x14x14xf32>
    %v3534 = stablehlo.multiply %v3533, %v3532 : tensor<64x1024x14x14xf32>
    %v3535 = stablehlo.reshape %v3309 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3536 = stablehlo.multiply %v3535, %v3534 : tensor<64x1024x14x14xf32>
    %v3537 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3538 = stablehlo.reduce(%v3536 init: %v3537) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3539 = stablehlo.reshape %v3309 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3541 = stablehlo.reduce(%v3539 init: %v3540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3542 = stablehlo.reshape %v3472 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3543 = stablehlo.reshape %v1561 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3544 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3545 = stablehlo.compare GT, %v3543, %v3544 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3546 = stablehlo.select %v3545, %v3542, %v3544 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3548 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3549 = stablehlo.slice %v1542 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3550 = stablehlo.slice %v1542 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3551 = stablehlo.broadcast_in_dim %v3549, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3552 = stablehlo.broadcast_in_dim %v3550, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3553 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3554 = stablehlo.add %v3552, %v3553 : tensor<64x1024x14x14xf32>
    %v3555 = stablehlo.rsqrt %v3554 : tensor<64x1024x14x14xf32>
    %v3556 = stablehlo.subtract %v3548, %v3551 : tensor<64x1024x14x14xf32>
    %v3557 = stablehlo.multiply %v3556, %v3555 : tensor<64x1024x14x14xf32>
    %v3558 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3559 = stablehlo.reshape %v3547 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3560 = stablehlo.multiply %v3558, %v3559 : tensor<64x1024x14x14xf32>
    %v3561 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3562 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3563 = stablehlo.reduce(%v3560 init: %v3561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3564 = stablehlo.divide %v3563, %v3562 : tensor<1024xf32>
    %v3565 = stablehlo.multiply %v3557, %v3560 : tensor<64x1024x14x14xf32>
    %v3566 = stablehlo.reduce(%v3565 init: %v3561) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3567 = stablehlo.divide %v3566, %v3562 : tensor<1024xf32>
    %v3568 = stablehlo.concatenate %v3564, %v3567, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3569 = stablehlo.concatenate %v1542, %v3568, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b3g3dst = "stablehlo.all_reduce"(%v3569) ({
    ^bb0(%aras3b3g3dst: tensor<f32>, %arbs3b3g3dst: tensor<f32>):
      %aradds3b3g3dst = stablehlo.add %aras3b3g3dst, %arbs3b3g3dst : tensor<f32>
      stablehlo.return %aradds3b3g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b3g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b3g3dst = stablehlo.divide %arsums3b3g3dst, %arns3b3g3dst : tensor<4096xf32>
    %v3570 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3571 = stablehlo.slice %armeans3b3g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3572 = stablehlo.slice %armeans3b3g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3573 = stablehlo.slice %armeans3b3g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3574 = stablehlo.slice %armeans3b3g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3575 = stablehlo.broadcast_in_dim %v3571, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3576 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3577 = stablehlo.broadcast_in_dim %v3573, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3578 = stablehlo.broadcast_in_dim %v3574, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3579 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3580 = stablehlo.add %v3576, %v3579 : tensor<64x1024x14x14xf32>
    %v3581 = stablehlo.rsqrt %v3580 : tensor<64x1024x14x14xf32>
    %v3582 = stablehlo.subtract %v3570, %v3575 : tensor<64x1024x14x14xf32>
    %v3583 = stablehlo.multiply %v3582, %v3581 : tensor<64x1024x14x14xf32>
    %v3584 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3585 = stablehlo.reshape %v3547 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3586 = stablehlo.multiply %v3584, %v3585 : tensor<64x1024x14x14xf32>
    %v3587 = stablehlo.subtract %v3586, %v3577 : tensor<64x1024x14x14xf32>
    %v3588 = stablehlo.multiply %v3583, %v3578 : tensor<64x1024x14x14xf32>
    %v3589 = stablehlo.subtract %v3587, %v3588 : tensor<64x1024x14x14xf32>
    %v3590 = stablehlo.multiply %v3581, %v3589 : tensor<64x1024x14x14xf32>
    %v3591 = stablehlo.reshape %v3590 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3592 = stablehlo.reshape %v3591 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3593 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3594 = stablehlo.transpose %v3593, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3595 = stablehlo.convolution(%v3592, %v3594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3597 = stablehlo.reshape %v3596 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3598 = stablehlo.reshape %v1516 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3599 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3600 = stablehlo.compare GT, %v3598, %v3599 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3601 = stablehlo.select %v3600, %v3597, %v3599 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3602 = stablehlo.reshape %v3601 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3603 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3604 = stablehlo.slice %v1501 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3605 = stablehlo.slice %v1501 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3606 = stablehlo.broadcast_in_dim %v3604, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3607 = stablehlo.broadcast_in_dim %v3605, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3608 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3609 = stablehlo.add %v3607, %v3608 : tensor<64x256x14x14xf32>
    %v3610 = stablehlo.rsqrt %v3609 : tensor<64x256x14x14xf32>
    %v3611 = stablehlo.subtract %v3603, %v3606 : tensor<64x256x14x14xf32>
    %v3612 = stablehlo.multiply %v3611, %v3610 : tensor<64x256x14x14xf32>
    %v3613 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3614 = stablehlo.reshape %v3602 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3615 = stablehlo.multiply %v3613, %v3614 : tensor<64x256x14x14xf32>
    %v3616 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3617 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3618 = stablehlo.reduce(%v3615 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3619 = stablehlo.divide %v3618, %v3617 : tensor<256xf32>
    %v3620 = stablehlo.multiply %v3612, %v3615 : tensor<64x256x14x14xf32>
    %v3621 = stablehlo.reduce(%v3620 init: %v3616) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3622 = stablehlo.divide %v3621, %v3617 : tensor<256xf32>
    %v3623 = stablehlo.concatenate %v3619, %v3622, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3624 = stablehlo.concatenate %v1501, %v3623, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g2dst = "stablehlo.all_reduce"(%v3624) ({
    ^bb0(%aras3b3g2dst: tensor<f32>, %arbs3b3g2dst: tensor<f32>):
      %aradds3b3g2dst = stablehlo.add %aras3b3g2dst, %arbs3b3g2dst : tensor<f32>
      stablehlo.return %aradds3b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g2dst = stablehlo.divide %arsums3b3g2dst, %arns3b3g2dst : tensor<1024xf32>
    %v3625 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3626 = stablehlo.slice %armeans3b3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3627 = stablehlo.slice %armeans3b3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3628 = stablehlo.slice %armeans3b3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3629 = stablehlo.slice %armeans3b3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3630 = stablehlo.broadcast_in_dim %v3626, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3631 = stablehlo.broadcast_in_dim %v3627, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3632 = stablehlo.broadcast_in_dim %v3628, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3633 = stablehlo.broadcast_in_dim %v3629, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3634 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3635 = stablehlo.add %v3631, %v3634 : tensor<64x256x14x14xf32>
    %v3636 = stablehlo.rsqrt %v3635 : tensor<64x256x14x14xf32>
    %v3637 = stablehlo.subtract %v3625, %v3630 : tensor<64x256x14x14xf32>
    %v3638 = stablehlo.multiply %v3637, %v3636 : tensor<64x256x14x14xf32>
    %v3639 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3640 = stablehlo.reshape %v3602 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3641 = stablehlo.multiply %v3639, %v3640 : tensor<64x256x14x14xf32>
    %v3642 = stablehlo.subtract %v3641, %v3632 : tensor<64x256x14x14xf32>
    %v3643 = stablehlo.multiply %v3638, %v3633 : tensor<64x256x14x14xf32>
    %v3644 = stablehlo.subtract %v3642, %v3643 : tensor<64x256x14x14xf32>
    %v3645 = stablehlo.multiply %v3636, %v3644 : tensor<64x256x14x14xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3647 = stablehlo.reshape %v3646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3648 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3649 = stablehlo.transpose %v3648, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3650 = stablehlo.convolution(%v3647, %v3649)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3652 = stablehlo.reshape %v3651 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3653 = stablehlo.reshape %v1475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3654 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3655 = stablehlo.compare GT, %v3653, %v3654 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3656 = stablehlo.select %v3655, %v3652, %v3654 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3657 = stablehlo.reshape %v3656 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3658 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3659 = stablehlo.slice %v1460 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3660 = stablehlo.slice %v1460 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3661 = stablehlo.broadcast_in_dim %v3659, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3662 = stablehlo.broadcast_in_dim %v3660, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3663 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3664 = stablehlo.add %v3662, %v3663 : tensor<64x256x14x14xf32>
    %v3665 = stablehlo.rsqrt %v3664 : tensor<64x256x14x14xf32>
    %v3666 = stablehlo.subtract %v3658, %v3661 : tensor<64x256x14x14xf32>
    %v3667 = stablehlo.multiply %v3666, %v3665 : tensor<64x256x14x14xf32>
    %v3668 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3669 = stablehlo.reshape %v3657 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3670 = stablehlo.multiply %v3668, %v3669 : tensor<64x256x14x14xf32>
    %v3671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3672 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3673 = stablehlo.reduce(%v3670 init: %v3671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3674 = stablehlo.divide %v3673, %v3672 : tensor<256xf32>
    %v3675 = stablehlo.multiply %v3667, %v3670 : tensor<64x256x14x14xf32>
    %v3676 = stablehlo.reduce(%v3675 init: %v3671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3677 = stablehlo.divide %v3676, %v3672 : tensor<256xf32>
    %v3678 = stablehlo.concatenate %v3674, %v3677, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3679 = stablehlo.concatenate %v1460, %v3678, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g1dst = "stablehlo.all_reduce"(%v3679) ({
    ^bb0(%aras3b3g1dst: tensor<f32>, %arbs3b3g1dst: tensor<f32>):
      %aradds3b3g1dst = stablehlo.add %aras3b3g1dst, %arbs3b3g1dst : tensor<f32>
      stablehlo.return %aradds3b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g1dst = stablehlo.divide %arsums3b3g1dst, %arns3b3g1dst : tensor<1024xf32>
    %v3680 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3681 = stablehlo.slice %armeans3b3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3682 = stablehlo.slice %armeans3b3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3683 = stablehlo.slice %armeans3b3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3684 = stablehlo.slice %armeans3b3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3685 = stablehlo.broadcast_in_dim %v3681, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3686 = stablehlo.broadcast_in_dim %v3682, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3687 = stablehlo.broadcast_in_dim %v3683, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3688 = stablehlo.broadcast_in_dim %v3684, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3689 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3690 = stablehlo.add %v3686, %v3689 : tensor<64x256x14x14xf32>
    %v3691 = stablehlo.rsqrt %v3690 : tensor<64x256x14x14xf32>
    %v3692 = stablehlo.subtract %v3680, %v3685 : tensor<64x256x14x14xf32>
    %v3693 = stablehlo.multiply %v3692, %v3691 : tensor<64x256x14x14xf32>
    %v3694 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3695 = stablehlo.reshape %v3657 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3696 = stablehlo.multiply %v3694, %v3695 : tensor<64x256x14x14xf32>
    %v3697 = stablehlo.subtract %v3696, %v3687 : tensor<64x256x14x14xf32>
    %v3698 = stablehlo.multiply %v3693, %v3688 : tensor<64x256x14x14xf32>
    %v3699 = stablehlo.subtract %v3697, %v3698 : tensor<64x256x14x14xf32>
    %v3700 = stablehlo.multiply %v3691, %v3699 : tensor<64x256x14x14xf32>
    %v3701 = stablehlo.reshape %v3700 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3702 = stablehlo.reshape %v3701 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3703 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3704 = stablehlo.transpose %v3703, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3705 = stablehlo.convolution(%v3702, %v3704)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v3706 = stablehlo.reshape %v3705 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3707 = stablehlo.reshape %v3706 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3708 = stablehlo.reshape %v3547 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3709 = stablehlo.add %v3707, %v3708 : tensor<64x1024x14x14xf32>
    %v3710 = stablehlo.reshape %v3709 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3711 = stablehlo.reshape %v1436 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3712 = stablehlo.reshape %v3701 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3713 = stablehlo.transpose %v3711, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3714 = stablehlo.transpose %v3712, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3715 = stablehlo.convolution(%v3713, %v3714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3716 = stablehlo.transpose %v3715, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3717 = stablehlo.reshape %v1441 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3718 = stablehlo.slice %v1460 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3719 = stablehlo.slice %v1460 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3718, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3722 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<64x256x14x14xf32>
    %v3724 = stablehlo.rsqrt %v3723 : tensor<64x256x14x14xf32>
    %v3725 = stablehlo.subtract %v3717, %v3720 : tensor<64x256x14x14xf32>
    %v3726 = stablehlo.multiply %v3725, %v3724 : tensor<64x256x14x14xf32>
    %v3727 = stablehlo.reshape %v3657 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3728 = stablehlo.multiply %v3727, %v3726 : tensor<64x256x14x14xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3731 = stablehlo.reshape %v3657 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3733 = stablehlo.reduce(%v3731 init: %v3732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3734 = stablehlo.reshape %v1477 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3735 = stablehlo.reshape %v3646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3736 = stablehlo.transpose %v3734, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3737 = stablehlo.transpose %v3735, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3738 = stablehlo.convolution(%v3736, %v3737)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3739 = stablehlo.transpose %v3738, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3740 = stablehlo.reshape %v1482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3741 = stablehlo.slice %v1501 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3742 = stablehlo.slice %v1501 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3743 = stablehlo.broadcast_in_dim %v3741, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3744 = stablehlo.broadcast_in_dim %v3742, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3745 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3746 = stablehlo.add %v3744, %v3745 : tensor<64x256x14x14xf32>
    %v3747 = stablehlo.rsqrt %v3746 : tensor<64x256x14x14xf32>
    %v3748 = stablehlo.subtract %v3740, %v3743 : tensor<64x256x14x14xf32>
    %v3749 = stablehlo.multiply %v3748, %v3747 : tensor<64x256x14x14xf32>
    %v3750 = stablehlo.reshape %v3602 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3751 = stablehlo.multiply %v3750, %v3749 : tensor<64x256x14x14xf32>
    %v3752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3753 = stablehlo.reduce(%v3751 init: %v3752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3754 = stablehlo.reshape %v3602 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3756 = stablehlo.reduce(%v3754 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3757 = stablehlo.reshape %v1518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3758 = stablehlo.reshape %v3591 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3759 = stablehlo.transpose %v3757, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3760 = stablehlo.transpose %v3758, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3761 = stablehlo.convolution(%v3759, %v3760)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v3762 = stablehlo.transpose %v3761, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3763 = stablehlo.reshape %v1523 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3764 = stablehlo.slice %v1542 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3765 = stablehlo.slice %v1542 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3766 = stablehlo.broadcast_in_dim %v3764, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3767 = stablehlo.broadcast_in_dim %v3765, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3768 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3769 = stablehlo.add %v3767, %v3768 : tensor<64x1024x14x14xf32>
    %v3770 = stablehlo.rsqrt %v3769 : tensor<64x1024x14x14xf32>
    %v3771 = stablehlo.subtract %v3763, %v3766 : tensor<64x1024x14x14xf32>
    %v3772 = stablehlo.multiply %v3771, %v3770 : tensor<64x1024x14x14xf32>
    %v3773 = stablehlo.reshape %v3547 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3774 = stablehlo.multiply %v3773, %v3772 : tensor<64x1024x14x14xf32>
    %v3775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3776 = stablehlo.reduce(%v3774 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3777 = stablehlo.reshape %v3547 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3780 = stablehlo.reshape %v3710 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3781 = stablehlo.reshape %v1432 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3782 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3783 = stablehlo.compare GT, %v3781, %v3782 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3784 = stablehlo.select %v3783, %v3780, %v3782 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3785 = stablehlo.reshape %v3784 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3786 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3787 = stablehlo.slice %v1413 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3788 = stablehlo.slice %v1413 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3789 = stablehlo.broadcast_in_dim %v3787, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3790 = stablehlo.broadcast_in_dim %v3788, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3791 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3792 = stablehlo.add %v3790, %v3791 : tensor<64x1024x14x14xf32>
    %v3793 = stablehlo.rsqrt %v3792 : tensor<64x1024x14x14xf32>
    %v3794 = stablehlo.subtract %v3786, %v3789 : tensor<64x1024x14x14xf32>
    %v3795 = stablehlo.multiply %v3794, %v3793 : tensor<64x1024x14x14xf32>
    %v3796 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3797 = stablehlo.reshape %v3785 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3798 = stablehlo.multiply %v3796, %v3797 : tensor<64x1024x14x14xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3801 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3802 = stablehlo.divide %v3801, %v3800 : tensor<1024xf32>
    %v3803 = stablehlo.multiply %v3795, %v3798 : tensor<64x1024x14x14xf32>
    %v3804 = stablehlo.reduce(%v3803 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3805 = stablehlo.divide %v3804, %v3800 : tensor<1024xf32>
    %v3806 = stablehlo.concatenate %v3802, %v3805, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3807 = stablehlo.concatenate %v1413, %v3806, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b2g3dst = "stablehlo.all_reduce"(%v3807) ({
    ^bb0(%aras3b2g3dst: tensor<f32>, %arbs3b2g3dst: tensor<f32>):
      %aradds3b2g3dst = stablehlo.add %aras3b2g3dst, %arbs3b2g3dst : tensor<f32>
      stablehlo.return %aradds3b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b2g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b2g3dst = stablehlo.divide %arsums3b2g3dst, %arns3b2g3dst : tensor<4096xf32>
    %v3808 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3809 = stablehlo.slice %armeans3b2g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3810 = stablehlo.slice %armeans3b2g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3811 = stablehlo.slice %armeans3b2g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3812 = stablehlo.slice %armeans3b2g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3809, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3814 = stablehlo.broadcast_in_dim %v3810, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3815 = stablehlo.broadcast_in_dim %v3811, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3816 = stablehlo.broadcast_in_dim %v3812, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3817 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3818 = stablehlo.add %v3814, %v3817 : tensor<64x1024x14x14xf32>
    %v3819 = stablehlo.rsqrt %v3818 : tensor<64x1024x14x14xf32>
    %v3820 = stablehlo.subtract %v3808, %v3813 : tensor<64x1024x14x14xf32>
    %v3821 = stablehlo.multiply %v3820, %v3819 : tensor<64x1024x14x14xf32>
    %v3822 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3823 = stablehlo.reshape %v3785 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3824 = stablehlo.multiply %v3822, %v3823 : tensor<64x1024x14x14xf32>
    %v3825 = stablehlo.subtract %v3824, %v3815 : tensor<64x1024x14x14xf32>
    %v3826 = stablehlo.multiply %v3821, %v3816 : tensor<64x1024x14x14xf32>
    %v3827 = stablehlo.subtract %v3825, %v3826 : tensor<64x1024x14x14xf32>
    %v3828 = stablehlo.multiply %v3819, %v3827 : tensor<64x1024x14x14xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3830 = stablehlo.reshape %v3829 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3831 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3832 = stablehlo.transpose %v3831, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3833 = stablehlo.convolution(%v3830, %v3832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v3834 = stablehlo.reshape %v3833 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3836 = stablehlo.reshape %v1387 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3838 = stablehlo.compare GT, %v3836, %v3837 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3839 = stablehlo.select %v3838, %v3835, %v3837 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3840 = stablehlo.reshape %v3839 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3841 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3842 = stablehlo.slice %v1372 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3843 = stablehlo.slice %v1372 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3844 = stablehlo.broadcast_in_dim %v3842, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3845 = stablehlo.broadcast_in_dim %v3843, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3846 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3847 = stablehlo.add %v3845, %v3846 : tensor<64x256x14x14xf32>
    %v3848 = stablehlo.rsqrt %v3847 : tensor<64x256x14x14xf32>
    %v3849 = stablehlo.subtract %v3841, %v3844 : tensor<64x256x14x14xf32>
    %v3850 = stablehlo.multiply %v3849, %v3848 : tensor<64x256x14x14xf32>
    %v3851 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3852 = stablehlo.reshape %v3840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3853 = stablehlo.multiply %v3851, %v3852 : tensor<64x256x14x14xf32>
    %v3854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3855 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3856 = stablehlo.reduce(%v3853 init: %v3854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3857 = stablehlo.divide %v3856, %v3855 : tensor<256xf32>
    %v3858 = stablehlo.multiply %v3850, %v3853 : tensor<64x256x14x14xf32>
    %v3859 = stablehlo.reduce(%v3858 init: %v3854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3860 = stablehlo.divide %v3859, %v3855 : tensor<256xf32>
    %v3861 = stablehlo.concatenate %v3857, %v3860, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3862 = stablehlo.concatenate %v1372, %v3861, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g2dst = "stablehlo.all_reduce"(%v3862) ({
    ^bb0(%aras3b2g2dst: tensor<f32>, %arbs3b2g2dst: tensor<f32>):
      %aradds3b2g2dst = stablehlo.add %aras3b2g2dst, %arbs3b2g2dst : tensor<f32>
      stablehlo.return %aradds3b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g2dst = stablehlo.divide %arsums3b2g2dst, %arns3b2g2dst : tensor<1024xf32>
    %v3863 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3864 = stablehlo.slice %armeans3b2g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3865 = stablehlo.slice %armeans3b2g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3866 = stablehlo.slice %armeans3b2g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3867 = stablehlo.slice %armeans3b2g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3868 = stablehlo.broadcast_in_dim %v3864, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3869 = stablehlo.broadcast_in_dim %v3865, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3866, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3871 = stablehlo.broadcast_in_dim %v3867, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3872 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3873 = stablehlo.add %v3869, %v3872 : tensor<64x256x14x14xf32>
    %v3874 = stablehlo.rsqrt %v3873 : tensor<64x256x14x14xf32>
    %v3875 = stablehlo.subtract %v3863, %v3868 : tensor<64x256x14x14xf32>
    %v3876 = stablehlo.multiply %v3875, %v3874 : tensor<64x256x14x14xf32>
    %v3877 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3878 = stablehlo.reshape %v3840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3879 = stablehlo.multiply %v3877, %v3878 : tensor<64x256x14x14xf32>
    %v3880 = stablehlo.subtract %v3879, %v3870 : tensor<64x256x14x14xf32>
    %v3881 = stablehlo.multiply %v3876, %v3871 : tensor<64x256x14x14xf32>
    %v3882 = stablehlo.subtract %v3880, %v3881 : tensor<64x256x14x14xf32>
    %v3883 = stablehlo.multiply %v3874, %v3882 : tensor<64x256x14x14xf32>
    %v3884 = stablehlo.reshape %v3883 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3885 = stablehlo.reshape %v3884 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3886 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3887 = stablehlo.transpose %v3886, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3888 = stablehlo.convolution(%v3885, %v3887)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v3889 = stablehlo.reshape %v3888 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3890 = stablehlo.reshape %v3889 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3891 = stablehlo.reshape %v1346 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3892 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3893 = stablehlo.compare GT, %v3891, %v3892 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3894 = stablehlo.select %v3893, %v3890, %v3892 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3895 = stablehlo.reshape %v3894 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3896 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3897 = stablehlo.slice %v1331 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3898 = stablehlo.slice %v1331 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3899 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3900 = stablehlo.broadcast_in_dim %v3898, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3901 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3902 = stablehlo.add %v3900, %v3901 : tensor<64x256x14x14xf32>
    %v3903 = stablehlo.rsqrt %v3902 : tensor<64x256x14x14xf32>
    %v3904 = stablehlo.subtract %v3896, %v3899 : tensor<64x256x14x14xf32>
    %v3905 = stablehlo.multiply %v3904, %v3903 : tensor<64x256x14x14xf32>
    %v3906 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3907 = stablehlo.reshape %v3895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3908 = stablehlo.multiply %v3906, %v3907 : tensor<64x256x14x14xf32>
    %v3909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3910 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3911 = stablehlo.reduce(%v3908 init: %v3909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3912 = stablehlo.divide %v3911, %v3910 : tensor<256xf32>
    %v3913 = stablehlo.multiply %v3905, %v3908 : tensor<64x256x14x14xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3915 = stablehlo.divide %v3914, %v3910 : tensor<256xf32>
    %v3916 = stablehlo.concatenate %v3912, %v3915, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3917 = stablehlo.concatenate %v1331, %v3916, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g1dst = "stablehlo.all_reduce"(%v3917) ({
    ^bb0(%aras3b2g1dst: tensor<f32>, %arbs3b2g1dst: tensor<f32>):
      %aradds3b2g1dst = stablehlo.add %aras3b2g1dst, %arbs3b2g1dst : tensor<f32>
      stablehlo.return %aradds3b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g1dst = stablehlo.divide %arsums3b2g1dst, %arns3b2g1dst : tensor<1024xf32>
    %v3918 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3919 = stablehlo.slice %armeans3b2g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3920 = stablehlo.slice %armeans3b2g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3921 = stablehlo.slice %armeans3b2g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3922 = stablehlo.slice %armeans3b2g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3919, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3924 = stablehlo.broadcast_in_dim %v3920, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3925 = stablehlo.broadcast_in_dim %v3921, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3926 = stablehlo.broadcast_in_dim %v3922, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3927 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3928 = stablehlo.add %v3924, %v3927 : tensor<64x256x14x14xf32>
    %v3929 = stablehlo.rsqrt %v3928 : tensor<64x256x14x14xf32>
    %v3930 = stablehlo.subtract %v3918, %v3923 : tensor<64x256x14x14xf32>
    %v3931 = stablehlo.multiply %v3930, %v3929 : tensor<64x256x14x14xf32>
    %v3932 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3933 = stablehlo.reshape %v3895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3934 = stablehlo.multiply %v3932, %v3933 : tensor<64x256x14x14xf32>
    %v3935 = stablehlo.subtract %v3934, %v3925 : tensor<64x256x14x14xf32>
    %v3936 = stablehlo.multiply %v3931, %v3926 : tensor<64x256x14x14xf32>
    %v3937 = stablehlo.subtract %v3935, %v3936 : tensor<64x256x14x14xf32>
    %v3938 = stablehlo.multiply %v3929, %v3937 : tensor<64x256x14x14xf32>
    %v3939 = stablehlo.reshape %v3938 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3940 = stablehlo.reshape %v3939 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3941 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3942 = stablehlo.transpose %v3941, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3943 = stablehlo.convolution(%v3940, %v3942)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v3944 = stablehlo.reshape %v3943 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3945 = stablehlo.reshape %v3944 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3946 = stablehlo.reshape %v3785 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3947 = stablehlo.add %v3945, %v3946 : tensor<64x1024x14x14xf32>
    %v3948 = stablehlo.reshape %v3947 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3949 = stablehlo.reshape %v1307 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3950 = stablehlo.reshape %v3939 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3951 = stablehlo.transpose %v3949, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3952 = stablehlo.transpose %v3950, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3953 = stablehlo.convolution(%v3951, %v3952)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v3954 = stablehlo.transpose %v3953, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3955 = stablehlo.reshape %v1312 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3956 = stablehlo.slice %v1331 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3957 = stablehlo.slice %v1331 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3958 = stablehlo.broadcast_in_dim %v3956, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3959 = stablehlo.broadcast_in_dim %v3957, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3960 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3961 = stablehlo.add %v3959, %v3960 : tensor<64x256x14x14xf32>
    %v3962 = stablehlo.rsqrt %v3961 : tensor<64x256x14x14xf32>
    %v3963 = stablehlo.subtract %v3955, %v3958 : tensor<64x256x14x14xf32>
    %v3964 = stablehlo.multiply %v3963, %v3962 : tensor<64x256x14x14xf32>
    %v3965 = stablehlo.reshape %v3895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3966 = stablehlo.multiply %v3965, %v3964 : tensor<64x256x14x14xf32>
    %v3967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3968 = stablehlo.reduce(%v3966 init: %v3967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3969 = stablehlo.reshape %v3895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3972 = stablehlo.reshape %v1348 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3973 = stablehlo.reshape %v3884 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3974 = stablehlo.transpose %v3972, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3975 = stablehlo.transpose %v3973, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3976 = stablehlo.convolution(%v3974, %v3975)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v3977 = stablehlo.transpose %v3976, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3978 = stablehlo.reshape %v1353 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3979 = stablehlo.slice %v1372 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3980 = stablehlo.slice %v1372 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3979, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3982 = stablehlo.broadcast_in_dim %v3980, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3983 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3984 = stablehlo.add %v3982, %v3983 : tensor<64x256x14x14xf32>
    %v3985 = stablehlo.rsqrt %v3984 : tensor<64x256x14x14xf32>
    %v3986 = stablehlo.subtract %v3978, %v3981 : tensor<64x256x14x14xf32>
    %v3987 = stablehlo.multiply %v3986, %v3985 : tensor<64x256x14x14xf32>
    %v3988 = stablehlo.reshape %v3840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3989 = stablehlo.multiply %v3988, %v3987 : tensor<64x256x14x14xf32>
    %v3990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3991 = stablehlo.reduce(%v3989 init: %v3990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3992 = stablehlo.reshape %v3840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3994 = stablehlo.reduce(%v3992 init: %v3993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3995 = stablehlo.reshape %v1389 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3996 = stablehlo.reshape %v3829 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3997 = stablehlo.transpose %v3995, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3998 = stablehlo.transpose %v3996, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3999 = stablehlo.convolution(%v3997, %v3998)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v4000 = stablehlo.transpose %v3999, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4001 = stablehlo.reshape %v1394 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4002 = stablehlo.slice %v1413 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4003 = stablehlo.slice %v1413 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4004 = stablehlo.broadcast_in_dim %v4002, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4005 = stablehlo.broadcast_in_dim %v4003, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4006 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4007 = stablehlo.add %v4005, %v4006 : tensor<64x1024x14x14xf32>
    %v4008 = stablehlo.rsqrt %v4007 : tensor<64x1024x14x14xf32>
    %v4009 = stablehlo.subtract %v4001, %v4004 : tensor<64x1024x14x14xf32>
    %v4010 = stablehlo.multiply %v4009, %v4008 : tensor<64x1024x14x14xf32>
    %v4011 = stablehlo.reshape %v3785 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4012 = stablehlo.multiply %v4011, %v4010 : tensor<64x1024x14x14xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.reduce(%v4012 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4015 = stablehlo.reshape %v3785 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4017 = stablehlo.reduce(%v4015 init: %v4016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4018 = stablehlo.reshape %v3948 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4019 = stablehlo.reshape %v1303 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4020 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v4021 = stablehlo.compare GT, %v4019, %v4020 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v4022 = stablehlo.select %v4021, %v4018, %v4020 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v4023 = stablehlo.reshape %v4022 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4024 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4025 = stablehlo.slice %v1284 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4026 = stablehlo.slice %v1284 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4027 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4028 = stablehlo.broadcast_in_dim %v4026, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4029 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4030 = stablehlo.add %v4028, %v4029 : tensor<64x1024x14x14xf32>
    %v4031 = stablehlo.rsqrt %v4030 : tensor<64x1024x14x14xf32>
    %v4032 = stablehlo.subtract %v4024, %v4027 : tensor<64x1024x14x14xf32>
    %v4033 = stablehlo.multiply %v4032, %v4031 : tensor<64x1024x14x14xf32>
    %v4034 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4035 = stablehlo.reshape %v4023 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4036 = stablehlo.multiply %v4034, %v4035 : tensor<64x1024x14x14xf32>
    %v4037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4038 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4039 = stablehlo.reduce(%v4036 init: %v4037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4040 = stablehlo.divide %v4039, %v4038 : tensor<1024xf32>
    %v4041 = stablehlo.multiply %v4033, %v4036 : tensor<64x1024x14x14xf32>
    %v4042 = stablehlo.reduce(%v4041 init: %v4037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4043 = stablehlo.divide %v4042, %v4038 : tensor<1024xf32>
    %v4044 = stablehlo.concatenate %v4040, %v4043, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4045 = stablehlo.concatenate %v1284, %v4044, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b1g3dst = "stablehlo.all_reduce"(%v4045) ({
    ^bb0(%aras3b1g3dst: tensor<f32>, %arbs3b1g3dst: tensor<f32>):
      %aradds3b1g3dst = stablehlo.add %aras3b1g3dst, %arbs3b1g3dst : tensor<f32>
      stablehlo.return %aradds3b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b1g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b1g3dst = stablehlo.divide %arsums3b1g3dst, %arns3b1g3dst : tensor<4096xf32>
    %v4046 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4047 = stablehlo.slice %armeans3b1g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4048 = stablehlo.slice %armeans3b1g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4049 = stablehlo.slice %armeans3b1g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4050 = stablehlo.slice %armeans3b1g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4051 = stablehlo.broadcast_in_dim %v4047, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4052 = stablehlo.broadcast_in_dim %v4048, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4049, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4050, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4055 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4056 = stablehlo.add %v4052, %v4055 : tensor<64x1024x14x14xf32>
    %v4057 = stablehlo.rsqrt %v4056 : tensor<64x1024x14x14xf32>
    %v4058 = stablehlo.subtract %v4046, %v4051 : tensor<64x1024x14x14xf32>
    %v4059 = stablehlo.multiply %v4058, %v4057 : tensor<64x1024x14x14xf32>
    %v4060 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4061 = stablehlo.reshape %v4023 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4062 = stablehlo.multiply %v4060, %v4061 : tensor<64x1024x14x14xf32>
    %v4063 = stablehlo.subtract %v4062, %v4053 : tensor<64x1024x14x14xf32>
    %v4064 = stablehlo.multiply %v4059, %v4054 : tensor<64x1024x14x14xf32>
    %v4065 = stablehlo.subtract %v4063, %v4064 : tensor<64x1024x14x14xf32>
    %v4066 = stablehlo.multiply %v4057, %v4065 : tensor<64x1024x14x14xf32>
    %v4067 = stablehlo.reshape %v4066 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4068 = stablehlo.reshape %v4067 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4069 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v4070 = stablehlo.transpose %v4069, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4071 = stablehlo.convolution(%v4068, %v4070)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v4072 = stablehlo.reshape %v4071 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4073 = stablehlo.reshape %v4072 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4074 = stablehlo.reshape %v1258 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4075 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4076 = stablehlo.compare GT, %v4074, %v4075 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4077 = stablehlo.select %v4076, %v4073, %v4075 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4078 = stablehlo.reshape %v4077 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4079 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4080 = stablehlo.slice %v1243 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4081 = stablehlo.slice %v1243 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4082 = stablehlo.broadcast_in_dim %v4080, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4083 = stablehlo.broadcast_in_dim %v4081, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4084 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4085 = stablehlo.add %v4083, %v4084 : tensor<64x256x14x14xf32>
    %v4086 = stablehlo.rsqrt %v4085 : tensor<64x256x14x14xf32>
    %v4087 = stablehlo.subtract %v4079, %v4082 : tensor<64x256x14x14xf32>
    %v4088 = stablehlo.multiply %v4087, %v4086 : tensor<64x256x14x14xf32>
    %v4089 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4090 = stablehlo.reshape %v4078 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4091 = stablehlo.multiply %v4089, %v4090 : tensor<64x256x14x14xf32>
    %v4092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4093 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4094 = stablehlo.reduce(%v4091 init: %v4092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4095 = stablehlo.divide %v4094, %v4093 : tensor<256xf32>
    %v4096 = stablehlo.multiply %v4088, %v4091 : tensor<64x256x14x14xf32>
    %v4097 = stablehlo.reduce(%v4096 init: %v4092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4098 = stablehlo.divide %v4097, %v4093 : tensor<256xf32>
    %v4099 = stablehlo.concatenate %v4095, %v4098, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4100 = stablehlo.concatenate %v1243, %v4099, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g2dst = "stablehlo.all_reduce"(%v4100) ({
    ^bb0(%aras3b1g2dst: tensor<f32>, %arbs3b1g2dst: tensor<f32>):
      %aradds3b1g2dst = stablehlo.add %aras3b1g2dst, %arbs3b1g2dst : tensor<f32>
      stablehlo.return %aradds3b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g2dst = stablehlo.divide %arsums3b1g2dst, %arns3b1g2dst : tensor<1024xf32>
    %v4101 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4102 = stablehlo.slice %armeans3b1g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4103 = stablehlo.slice %armeans3b1g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4104 = stablehlo.slice %armeans3b1g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4105 = stablehlo.slice %armeans3b1g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4102, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4107 = stablehlo.broadcast_in_dim %v4103, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4108 = stablehlo.broadcast_in_dim %v4104, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4109 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4110 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4111 = stablehlo.add %v4107, %v4110 : tensor<64x256x14x14xf32>
    %v4112 = stablehlo.rsqrt %v4111 : tensor<64x256x14x14xf32>
    %v4113 = stablehlo.subtract %v4101, %v4106 : tensor<64x256x14x14xf32>
    %v4114 = stablehlo.multiply %v4113, %v4112 : tensor<64x256x14x14xf32>
    %v4115 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4116 = stablehlo.reshape %v4078 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4117 = stablehlo.multiply %v4115, %v4116 : tensor<64x256x14x14xf32>
    %v4118 = stablehlo.subtract %v4117, %v4108 : tensor<64x256x14x14xf32>
    %v4119 = stablehlo.multiply %v4114, %v4109 : tensor<64x256x14x14xf32>
    %v4120 = stablehlo.subtract %v4118, %v4119 : tensor<64x256x14x14xf32>
    %v4121 = stablehlo.multiply %v4112, %v4120 : tensor<64x256x14x14xf32>
    %v4122 = stablehlo.reshape %v4121 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4123 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4124 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v4125 = stablehlo.transpose %v4124, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4126 = stablehlo.convolution(%v4123, %v4125)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x14x14xf32>
    %v4127 = stablehlo.reshape %v4126 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4128 = stablehlo.reshape %v4127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4129 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4130 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4131 = stablehlo.compare GT, %v4129, %v4130 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4132 = stablehlo.select %v4131, %v4128, %v4130 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4133 = stablehlo.reshape %v4132 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4134 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4135 = stablehlo.slice %v1202 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4136 = stablehlo.slice %v1202 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4137 = stablehlo.broadcast_in_dim %v4135, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4138 = stablehlo.broadcast_in_dim %v4136, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4139 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4140 = stablehlo.add %v4138, %v4139 : tensor<64x256x14x14xf32>
    %v4141 = stablehlo.rsqrt %v4140 : tensor<64x256x14x14xf32>
    %v4142 = stablehlo.subtract %v4134, %v4137 : tensor<64x256x14x14xf32>
    %v4143 = stablehlo.multiply %v4142, %v4141 : tensor<64x256x14x14xf32>
    %v4144 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4145 = stablehlo.reshape %v4133 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4146 = stablehlo.multiply %v4144, %v4145 : tensor<64x256x14x14xf32>
    %v4147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4148 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4149 = stablehlo.reduce(%v4146 init: %v4147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4150 = stablehlo.divide %v4149, %v4148 : tensor<256xf32>
    %v4151 = stablehlo.multiply %v4143, %v4146 : tensor<64x256x14x14xf32>
    %v4152 = stablehlo.reduce(%v4151 init: %v4147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4153 = stablehlo.divide %v4152, %v4148 : tensor<256xf32>
    %v4154 = stablehlo.concatenate %v4150, %v4153, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4155 = stablehlo.concatenate %v1202, %v4154, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g1dst = "stablehlo.all_reduce"(%v4155) ({
    ^bb0(%aras3b1g1dst: tensor<f32>, %arbs3b1g1dst: tensor<f32>):
      %aradds3b1g1dst = stablehlo.add %aras3b1g1dst, %arbs3b1g1dst : tensor<f32>
      stablehlo.return %aradds3b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g1dst = stablehlo.divide %arsums3b1g1dst, %arns3b1g1dst : tensor<1024xf32>
    %v4156 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4157 = stablehlo.slice %armeans3b1g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4158 = stablehlo.slice %armeans3b1g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4159 = stablehlo.slice %armeans3b1g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4160 = stablehlo.slice %armeans3b1g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4161 = stablehlo.broadcast_in_dim %v4157, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4162 = stablehlo.broadcast_in_dim %v4158, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4163 = stablehlo.broadcast_in_dim %v4159, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4164 = stablehlo.broadcast_in_dim %v4160, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4165 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4166 = stablehlo.add %v4162, %v4165 : tensor<64x256x14x14xf32>
    %v4167 = stablehlo.rsqrt %v4166 : tensor<64x256x14x14xf32>
    %v4168 = stablehlo.subtract %v4156, %v4161 : tensor<64x256x14x14xf32>
    %v4169 = stablehlo.multiply %v4168, %v4167 : tensor<64x256x14x14xf32>
    %v4170 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4171 = stablehlo.reshape %v4133 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4172 = stablehlo.multiply %v4170, %v4171 : tensor<64x256x14x14xf32>
    %v4173 = stablehlo.subtract %v4172, %v4163 : tensor<64x256x14x14xf32>
    %v4174 = stablehlo.multiply %v4169, %v4164 : tensor<64x256x14x14xf32>
    %v4175 = stablehlo.subtract %v4173, %v4174 : tensor<64x256x14x14xf32>
    %v4176 = stablehlo.multiply %v4167, %v4175 : tensor<64x256x14x14xf32>
    %v4177 = stablehlo.reshape %v4176 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4178 = stablehlo.reshape %v4177 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4179 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v4180 = stablehlo.transpose %v4179, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4181 = stablehlo.convolution(%v4178, %v4180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<1024x256x1x1xf32>) -> tensor<64x1024x14x14xf32>
    %v4182 = stablehlo.reshape %v4181 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4183 = stablehlo.reshape %v4182 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4184 = stablehlo.reshape %v4023 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4185 = stablehlo.add %v4183, %v4184 : tensor<64x1024x14x14xf32>
    %v4186 = stablehlo.reshape %v4185 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4187 = stablehlo.reshape %v1178 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4188 = stablehlo.reshape %v4177 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4189 = stablehlo.transpose %v4187, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4190 = stablehlo.transpose %v4188, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4191 = stablehlo.convolution(%v4189, %v4190)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<1024x256x1x1xf32>
    %v4192 = stablehlo.transpose %v4191, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4193 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4194 = stablehlo.slice %v1202 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4195 = stablehlo.slice %v1202 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4196 = stablehlo.broadcast_in_dim %v4194, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4197 = stablehlo.broadcast_in_dim %v4195, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4198 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4199 = stablehlo.add %v4197, %v4198 : tensor<64x256x14x14xf32>
    %v4200 = stablehlo.rsqrt %v4199 : tensor<64x256x14x14xf32>
    %v4201 = stablehlo.subtract %v4193, %v4196 : tensor<64x256x14x14xf32>
    %v4202 = stablehlo.multiply %v4201, %v4200 : tensor<64x256x14x14xf32>
    %v4203 = stablehlo.reshape %v4133 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4204 = stablehlo.multiply %v4203, %v4202 : tensor<64x256x14x14xf32>
    %v4205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4206 = stablehlo.reduce(%v4204 init: %v4205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4207 = stablehlo.reshape %v4133 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4209 = stablehlo.reduce(%v4207 init: %v4208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4210 = stablehlo.reshape %v1219 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4211 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4212 = stablehlo.transpose %v4210, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4213 = stablehlo.transpose %v4211, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4214 = stablehlo.convolution(%v4212, %v4213)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v4215 = stablehlo.transpose %v4214, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4216 = stablehlo.reshape %v1224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4217 = stablehlo.slice %v1243 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4218 = stablehlo.slice %v1243 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4219 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4220 = stablehlo.broadcast_in_dim %v4218, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4221 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4222 = stablehlo.add %v4220, %v4221 : tensor<64x256x14x14xf32>
    %v4223 = stablehlo.rsqrt %v4222 : tensor<64x256x14x14xf32>
    %v4224 = stablehlo.subtract %v4216, %v4219 : tensor<64x256x14x14xf32>
    %v4225 = stablehlo.multiply %v4224, %v4223 : tensor<64x256x14x14xf32>
    %v4226 = stablehlo.reshape %v4078 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4227 = stablehlo.multiply %v4226, %v4225 : tensor<64x256x14x14xf32>
    %v4228 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4229 = stablehlo.reduce(%v4227 init: %v4228) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4230 = stablehlo.reshape %v4078 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4233 = stablehlo.reshape %v1260 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4234 = stablehlo.reshape %v4067 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4235 = stablehlo.transpose %v4233, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4236 = stablehlo.transpose %v4234, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4237 = stablehlo.convolution(%v4235, %v4236)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v4238 = stablehlo.transpose %v4237, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4239 = stablehlo.reshape %v1265 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4240 = stablehlo.slice %v1284 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4241 = stablehlo.slice %v1284 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4242 = stablehlo.broadcast_in_dim %v4240, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4243 = stablehlo.broadcast_in_dim %v4241, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4244 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4245 = stablehlo.add %v4243, %v4244 : tensor<64x1024x14x14xf32>
    %v4246 = stablehlo.rsqrt %v4245 : tensor<64x1024x14x14xf32>
    %v4247 = stablehlo.subtract %v4239, %v4242 : tensor<64x1024x14x14xf32>
    %v4248 = stablehlo.multiply %v4247, %v4246 : tensor<64x1024x14x14xf32>
    %v4249 = stablehlo.reshape %v4023 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4250 = stablehlo.multiply %v4249, %v4248 : tensor<64x1024x14x14xf32>
    %v4251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4252 = stablehlo.reduce(%v4250 init: %v4251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4253 = stablehlo.reshape %v4023 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4255 = stablehlo.reduce(%v4253 init: %v4254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4256 = stablehlo.reshape %v4186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4257 = stablehlo.reshape %v1176 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v4259 = stablehlo.compare GT, %v4257, %v4258 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v4260 = stablehlo.select %v4259, %v4256, %v4258 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v4261 = stablehlo.reshape %v4260 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4262 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4263 = stablehlo.slice %v1121 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4264 = stablehlo.slice %v1121 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4265 = stablehlo.broadcast_in_dim %v4263, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4264, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4267 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4268 = stablehlo.add %v4266, %v4267 : tensor<64x1024x14x14xf32>
    %v4269 = stablehlo.rsqrt %v4268 : tensor<64x1024x14x14xf32>
    %v4270 = stablehlo.subtract %v4262, %v4265 : tensor<64x1024x14x14xf32>
    %v4271 = stablehlo.multiply %v4270, %v4269 : tensor<64x1024x14x14xf32>
    %v4272 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4273 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4274 = stablehlo.multiply %v4272, %v4273 : tensor<64x1024x14x14xf32>
    %v4275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4276 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4277 = stablehlo.reduce(%v4274 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4278 = stablehlo.divide %v4277, %v4276 : tensor<1024xf32>
    %v4279 = stablehlo.multiply %v4271, %v4274 : tensor<64x1024x14x14xf32>
    %v4280 = stablehlo.reduce(%v4279 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4281 = stablehlo.divide %v4280, %v4276 : tensor<1024xf32>
    %v4282 = stablehlo.concatenate %v4278, %v4281, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4283 = stablehlo.concatenate %v1121, %v4282, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b0g3dst = "stablehlo.all_reduce"(%v4283) ({
    ^bb0(%aras3b0g3dst: tensor<f32>, %arbs3b0g3dst: tensor<f32>):
      %aradds3b0g3dst = stablehlo.add %aras3b0g3dst, %arbs3b0g3dst : tensor<f32>
      stablehlo.return %aradds3b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b0g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b0g3dst = stablehlo.divide %arsums3b0g3dst, %arns3b0g3dst : tensor<4096xf32>
    %v4284 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4285 = stablehlo.slice %armeans3b0g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4286 = stablehlo.slice %armeans3b0g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4287 = stablehlo.slice %armeans3b0g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4288 = stablehlo.slice %armeans3b0g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4289 = stablehlo.broadcast_in_dim %v4285, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4290 = stablehlo.broadcast_in_dim %v4286, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4291 = stablehlo.broadcast_in_dim %v4287, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4292 = stablehlo.broadcast_in_dim %v4288, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4293 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4294 = stablehlo.add %v4290, %v4293 : tensor<64x1024x14x14xf32>
    %v4295 = stablehlo.rsqrt %v4294 : tensor<64x1024x14x14xf32>
    %v4296 = stablehlo.subtract %v4284, %v4289 : tensor<64x1024x14x14xf32>
    %v4297 = stablehlo.multiply %v4296, %v4295 : tensor<64x1024x14x14xf32>
    %v4298 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4299 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4300 = stablehlo.multiply %v4298, %v4299 : tensor<64x1024x14x14xf32>
    %v4301 = stablehlo.subtract %v4300, %v4291 : tensor<64x1024x14x14xf32>
    %v4302 = stablehlo.multiply %v4297, %v4292 : tensor<64x1024x14x14xf32>
    %v4303 = stablehlo.subtract %v4301, %v4302 : tensor<64x1024x14x14xf32>
    %v4304 = stablehlo.multiply %v4295, %v4303 : tensor<64x1024x14x14xf32>
    %v4305 = stablehlo.reshape %v4304 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4306 = stablehlo.reshape %v4305 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4307 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v4308 = stablehlo.transpose %v4307, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4309 = stablehlo.convolution(%v4306, %v4308)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<256x1024x1x1xf32>) -> tensor<64x256x14x14xf32>
    %v4310 = stablehlo.reshape %v4309 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4311 = stablehlo.reshape %v4310 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4312 = stablehlo.reshape %v1095 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4314 = stablehlo.compare GT, %v4312, %v4313 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4315 = stablehlo.select %v4314, %v4311, %v4313 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4316 = stablehlo.reshape %v4315 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4317 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4318 = stablehlo.slice %v1080 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4319 = stablehlo.slice %v1080 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4320 = stablehlo.broadcast_in_dim %v4318, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4321 = stablehlo.broadcast_in_dim %v4319, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4322 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4323 = stablehlo.add %v4321, %v4322 : tensor<64x256x14x14xf32>
    %v4324 = stablehlo.rsqrt %v4323 : tensor<64x256x14x14xf32>
    %v4325 = stablehlo.subtract %v4317, %v4320 : tensor<64x256x14x14xf32>
    %v4326 = stablehlo.multiply %v4325, %v4324 : tensor<64x256x14x14xf32>
    %v4327 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4328 = stablehlo.reshape %v4316 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4329 = stablehlo.multiply %v4327, %v4328 : tensor<64x256x14x14xf32>
    %v4330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4331 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4332 = stablehlo.reduce(%v4329 init: %v4330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4333 = stablehlo.divide %v4332, %v4331 : tensor<256xf32>
    %v4334 = stablehlo.multiply %v4326, %v4329 : tensor<64x256x14x14xf32>
    %v4335 = stablehlo.reduce(%v4334 init: %v4330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4336 = stablehlo.divide %v4335, %v4331 : tensor<256xf32>
    %v4337 = stablehlo.concatenate %v4333, %v4336, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4338 = stablehlo.concatenate %v1080, %v4337, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g2dst = "stablehlo.all_reduce"(%v4338) ({
    ^bb0(%aras3b0g2dst: tensor<f32>, %arbs3b0g2dst: tensor<f32>):
      %aradds3b0g2dst = stablehlo.add %aras3b0g2dst, %arbs3b0g2dst : tensor<f32>
      stablehlo.return %aradds3b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g2dst = stablehlo.divide %arsums3b0g2dst, %arns3b0g2dst : tensor<1024xf32>
    %v4339 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4340 = stablehlo.slice %armeans3b0g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4341 = stablehlo.slice %armeans3b0g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4342 = stablehlo.slice %armeans3b0g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4343 = stablehlo.slice %armeans3b0g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4344 = stablehlo.broadcast_in_dim %v4340, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4345 = stablehlo.broadcast_in_dim %v4341, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4346 = stablehlo.broadcast_in_dim %v4342, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4343, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4348 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4349 = stablehlo.add %v4345, %v4348 : tensor<64x256x14x14xf32>
    %v4350 = stablehlo.rsqrt %v4349 : tensor<64x256x14x14xf32>
    %v4351 = stablehlo.subtract %v4339, %v4344 : tensor<64x256x14x14xf32>
    %v4352 = stablehlo.multiply %v4351, %v4350 : tensor<64x256x14x14xf32>
    %v4353 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4354 = stablehlo.reshape %v4316 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4355 = stablehlo.multiply %v4353, %v4354 : tensor<64x256x14x14xf32>
    %v4356 = stablehlo.subtract %v4355, %v4346 : tensor<64x256x14x14xf32>
    %v4357 = stablehlo.multiply %v4352, %v4347 : tensor<64x256x14x14xf32>
    %v4358 = stablehlo.subtract %v4356, %v4357 : tensor<64x256x14x14xf32>
    %v4359 = stablehlo.multiply %v4350, %v4358 : tensor<64x256x14x14xf32>
    %v4360 = stablehlo.reshape %v4359 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4361 = stablehlo.reshape %v4360 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4362 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v4363 = stablehlo.transpose %v4362, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4364 = stablehlo.convolution(%v4361, %v4363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xf32>, tensor<256x256x3x3xf32>) -> tensor<64x256x28x28xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4366 = stablehlo.reshape %v4365 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4367 = stablehlo.reshape %v1054 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4368 = stablehlo.constant dense<0.0> : tensor<64x256x28x28xf32>
    %v4369 = stablehlo.compare GT, %v4367, %v4368 : (tensor<64x256x28x28xf32>, tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xi1>
    %v4370 = stablehlo.select %v4369, %v4366, %v4368 : tensor<64x256x28x28xi1>, tensor<64x256x28x28xf32>
    %v4371 = stablehlo.reshape %v4370 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4372 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4373 = stablehlo.slice %v1039 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4374 = stablehlo.slice %v1039 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4375 = stablehlo.broadcast_in_dim %v4373, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4376 = stablehlo.broadcast_in_dim %v4374, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4377 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4378 = stablehlo.add %v4376, %v4377 : tensor<64x256x28x28xf32>
    %v4379 = stablehlo.rsqrt %v4378 : tensor<64x256x28x28xf32>
    %v4380 = stablehlo.subtract %v4372, %v4375 : tensor<64x256x28x28xf32>
    %v4381 = stablehlo.multiply %v4380, %v4379 : tensor<64x256x28x28xf32>
    %v4382 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4383 = stablehlo.reshape %v4371 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4384 = stablehlo.multiply %v4382, %v4383 : tensor<64x256x28x28xf32>
    %v4385 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4386 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v4387 = stablehlo.reduce(%v4384 init: %v4385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4388 = stablehlo.divide %v4387, %v4386 : tensor<256xf32>
    %v4389 = stablehlo.multiply %v4381, %v4384 : tensor<64x256x28x28xf32>
    %v4390 = stablehlo.reduce(%v4389 init: %v4385) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4391 = stablehlo.divide %v4390, %v4386 : tensor<256xf32>
    %v4392 = stablehlo.concatenate %v4388, %v4391, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4393 = stablehlo.concatenate %v1039, %v4392, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g1dst = "stablehlo.all_reduce"(%v4393) ({
    ^bb0(%aras3b0g1dst: tensor<f32>, %arbs3b0g1dst: tensor<f32>):
      %aradds3b0g1dst = stablehlo.add %aras3b0g1dst, %arbs3b0g1dst : tensor<f32>
      stablehlo.return %aradds3b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g1dst = stablehlo.divide %arsums3b0g1dst, %arns3b0g1dst : tensor<1024xf32>
    %v4394 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4395 = stablehlo.slice %armeans3b0g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4396 = stablehlo.slice %armeans3b0g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4397 = stablehlo.slice %armeans3b0g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4398 = stablehlo.slice %armeans3b0g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4399 = stablehlo.broadcast_in_dim %v4395, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4400 = stablehlo.broadcast_in_dim %v4396, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4401 = stablehlo.broadcast_in_dim %v4397, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4402 = stablehlo.broadcast_in_dim %v4398, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4403 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4404 = stablehlo.add %v4400, %v4403 : tensor<64x256x28x28xf32>
    %v4405 = stablehlo.rsqrt %v4404 : tensor<64x256x28x28xf32>
    %v4406 = stablehlo.subtract %v4394, %v4399 : tensor<64x256x28x28xf32>
    %v4407 = stablehlo.multiply %v4406, %v4405 : tensor<64x256x28x28xf32>
    %v4408 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4409 = stablehlo.reshape %v4371 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4410 = stablehlo.multiply %v4408, %v4409 : tensor<64x256x28x28xf32>
    %v4411 = stablehlo.subtract %v4410, %v4401 : tensor<64x256x28x28xf32>
    %v4412 = stablehlo.multiply %v4407, %v4402 : tensor<64x256x28x28xf32>
    %v4413 = stablehlo.subtract %v4411, %v4412 : tensor<64x256x28x28xf32>
    %v4414 = stablehlo.multiply %v4405, %v4413 : tensor<64x256x28x28xf32>
    %v4415 = stablehlo.reshape %v4414 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4416 = stablehlo.reshape %v4415 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4417 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v4418 = stablehlo.transpose %v4417, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4419 = stablehlo.convolution(%v4416, %v4418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xf32>, tensor<512x256x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v4420 = stablehlo.reshape %v4419 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4421 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4422 = stablehlo.slice %v1160 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4423 = stablehlo.slice %v1160 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4424 = stablehlo.broadcast_in_dim %v4422, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4425 = stablehlo.broadcast_in_dim %v4423, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4426 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4427 = stablehlo.add %v4425, %v4426 : tensor<64x1024x14x14xf32>
    %v4428 = stablehlo.rsqrt %v4427 : tensor<64x1024x14x14xf32>
    %v4429 = stablehlo.subtract %v4421, %v4424 : tensor<64x1024x14x14xf32>
    %v4430 = stablehlo.multiply %v4429, %v4428 : tensor<64x1024x14x14xf32>
    %v4431 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4432 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4433 = stablehlo.multiply %v4431, %v4432 : tensor<64x1024x14x14xf32>
    %v4434 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4435 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4436 = stablehlo.reduce(%v4433 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4437 = stablehlo.divide %v4436, %v4435 : tensor<1024xf32>
    %v4438 = stablehlo.multiply %v4430, %v4433 : tensor<64x1024x14x14xf32>
    %v4439 = stablehlo.reduce(%v4438 init: %v4434) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4440 = stablehlo.divide %v4439, %v4435 : tensor<1024xf32>
    %v4441 = stablehlo.concatenate %v4437, %v4440, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4442 = stablehlo.concatenate %v1160, %v4441, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b0gpdst = "stablehlo.all_reduce"(%v4442) ({
    ^bb0(%aras3b0gpdst: tensor<f32>, %arbs3b0gpdst: tensor<f32>):
      %aradds3b0gpdst = stablehlo.add %aras3b0gpdst, %arbs3b0gpdst : tensor<f32>
      stablehlo.return %aradds3b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b0gpdst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b0gpdst = stablehlo.divide %arsums3b0gpdst, %arns3b0gpdst : tensor<4096xf32>
    %v4443 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4444 = stablehlo.slice %armeans3b0gpdst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4445 = stablehlo.slice %armeans3b0gpdst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4446 = stablehlo.slice %armeans3b0gpdst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4447 = stablehlo.slice %armeans3b0gpdst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4448 = stablehlo.broadcast_in_dim %v4444, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4449 = stablehlo.broadcast_in_dim %v4445, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4450 = stablehlo.broadcast_in_dim %v4446, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4451 = stablehlo.broadcast_in_dim %v4447, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4452 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4453 = stablehlo.add %v4449, %v4452 : tensor<64x1024x14x14xf32>
    %v4454 = stablehlo.rsqrt %v4453 : tensor<64x1024x14x14xf32>
    %v4455 = stablehlo.subtract %v4443, %v4448 : tensor<64x1024x14x14xf32>
    %v4456 = stablehlo.multiply %v4455, %v4454 : tensor<64x1024x14x14xf32>
    %v4457 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4458 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4459 = stablehlo.multiply %v4457, %v4458 : tensor<64x1024x14x14xf32>
    %v4460 = stablehlo.subtract %v4459, %v4450 : tensor<64x1024x14x14xf32>
    %v4461 = stablehlo.multiply %v4456, %v4451 : tensor<64x1024x14x14xf32>
    %v4462 = stablehlo.subtract %v4460, %v4461 : tensor<64x1024x14x14xf32>
    %v4463 = stablehlo.multiply %v4454, %v4462 : tensor<64x1024x14x14xf32>
    %v4464 = stablehlo.reshape %v4463 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4465 = stablehlo.reshape %v4464 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4466 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v4467 = stablehlo.transpose %v4466, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v4468 = stablehlo.convolution(%v4465, %v4467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xf32>, tensor<512x1024x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v4469 = stablehlo.reshape %v4468 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4470 = stablehlo.reshape %v4420 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4471 = stablehlo.reshape %v4469 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4472 = stablehlo.add %v4470, %v4471 : tensor<64x512x28x28xf32>
    %v4473 = stablehlo.reshape %v4472 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4474 = stablehlo.reshape %v1015 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4475 = stablehlo.reshape %v4415 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4476 = stablehlo.transpose %v4474, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4477 = stablehlo.transpose %v4475, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v4478 = stablehlo.convolution(%v4476, %v4477)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xf32>, tensor<256x64x28x28xf32>) -> tensor<512x256x1x1xf32>
    %v4479 = stablehlo.transpose %v4478, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4480 = stablehlo.reshape %v1020 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4481 = stablehlo.slice %v1039 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4482 = stablehlo.slice %v1039 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4483 = stablehlo.broadcast_in_dim %v4481, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4484 = stablehlo.broadcast_in_dim %v4482, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4485 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4486 = stablehlo.add %v4484, %v4485 : tensor<64x256x28x28xf32>
    %v4487 = stablehlo.rsqrt %v4486 : tensor<64x256x28x28xf32>
    %v4488 = stablehlo.subtract %v4480, %v4483 : tensor<64x256x28x28xf32>
    %v4489 = stablehlo.multiply %v4488, %v4487 : tensor<64x256x28x28xf32>
    %v4490 = stablehlo.reshape %v4371 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4491 = stablehlo.multiply %v4490, %v4489 : tensor<64x256x28x28xf32>
    %v4492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4493 = stablehlo.reduce(%v4491 init: %v4492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4494 = stablehlo.reshape %v4371 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4496 = stablehlo.reduce(%v4494 init: %v4495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4497 = stablehlo.reshape %v1056 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4498 = stablehlo.reshape %v4360 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4499 = stablehlo.transpose %v4497, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v4501 = stablehlo.transpose %v4498, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4502 = stablehlo.convolution(%v4499, %v4501)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x28x28xf32>, tensor<256x64x14x14xf32>) -> tensor<256x256x3x3xf32>
    %v4503 = stablehlo.transpose %v4502, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4504 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4505 = stablehlo.slice %v1080 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4506 = stablehlo.slice %v1080 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4507 = stablehlo.broadcast_in_dim %v4505, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4508 = stablehlo.broadcast_in_dim %v4506, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4509 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4510 = stablehlo.add %v4508, %v4509 : tensor<64x256x14x14xf32>
    %v4511 = stablehlo.rsqrt %v4510 : tensor<64x256x14x14xf32>
    %v4512 = stablehlo.subtract %v4504, %v4507 : tensor<64x256x14x14xf32>
    %v4513 = stablehlo.multiply %v4512, %v4511 : tensor<64x256x14x14xf32>
    %v4514 = stablehlo.reshape %v4316 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4515 = stablehlo.multiply %v4514, %v4513 : tensor<64x256x14x14xf32>
    %v4516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4517 = stablehlo.reduce(%v4515 init: %v4516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4518 = stablehlo.reshape %v4316 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4520 = stablehlo.reduce(%v4518 init: %v4519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4521 = stablehlo.reshape %v1097 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4522 = stablehlo.reshape %v4305 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4523 = stablehlo.transpose %v4521, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4524 = stablehlo.transpose %v4522, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4525 = stablehlo.convolution(%v4523, %v4524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xf32>, tensor<1024x64x14x14xf32>) -> tensor<256x1024x1x1xf32>
    %v4526 = stablehlo.transpose %v4525, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4527 = stablehlo.reshape %v1102 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4528 = stablehlo.slice %v1121 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4529 = stablehlo.slice %v1121 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4530 = stablehlo.broadcast_in_dim %v4528, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4531 = stablehlo.broadcast_in_dim %v4529, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4532 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4533 = stablehlo.add %v4531, %v4532 : tensor<64x1024x14x14xf32>
    %v4534 = stablehlo.rsqrt %v4533 : tensor<64x1024x14x14xf32>
    %v4535 = stablehlo.subtract %v4527, %v4530 : tensor<64x1024x14x14xf32>
    %v4536 = stablehlo.multiply %v4535, %v4534 : tensor<64x1024x14x14xf32>
    %v4537 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4538 = stablehlo.multiply %v4537, %v4536 : tensor<64x1024x14x14xf32>
    %v4539 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4540 = stablehlo.reduce(%v4538 init: %v4539) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4541 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4543 = stablehlo.reduce(%v4541 init: %v4542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4544 = stablehlo.reshape %v1015 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4545 = stablehlo.reshape %v4464 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4546 = stablehlo.transpose %v4544, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4548 = stablehlo.transpose %v4545, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4547 = stablehlo.slice %v4546 [0:512, 0:64, 0:27, 0:27] : (tensor<512x64x28x28xf32>) -> tensor<512x64x27x27xf32>
    %v4549 = stablehlo.convolution(%v4547, %v4548)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x27x27xf32>, tensor<1024x64x14x14xf32>) -> tensor<512x1024x1x1xf32>
    %v4550 = stablehlo.transpose %v4549, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v4551 = stablehlo.reshape %v1141 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4552 = stablehlo.slice %v1160 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4553 = stablehlo.slice %v1160 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4554 = stablehlo.broadcast_in_dim %v4552, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4555 = stablehlo.broadcast_in_dim %v4553, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4556 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4557 = stablehlo.add %v4555, %v4556 : tensor<64x1024x14x14xf32>
    %v4558 = stablehlo.rsqrt %v4557 : tensor<64x1024x14x14xf32>
    %v4559 = stablehlo.subtract %v4551, %v4554 : tensor<64x1024x14x14xf32>
    %v4560 = stablehlo.multiply %v4559, %v4558 : tensor<64x1024x14x14xf32>
    %v4561 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4562 = stablehlo.multiply %v4561, %v4560 : tensor<64x1024x14x14xf32>
    %v4563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4564 = stablehlo.reduce(%v4562 init: %v4563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4565 = stablehlo.reshape %v4261 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4567 = stablehlo.reduce(%v4565 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4568 = stablehlo.reshape %v4473 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4569 = stablehlo.reshape %v1011 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4570 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v4571 = stablehlo.compare GT, %v4569, %v4570 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v4572 = stablehlo.select %v4571, %v4568, %v4570 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v4573 = stablehlo.reshape %v4572 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4574 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4575 = stablehlo.slice %v992 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4576 = stablehlo.slice %v992 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4577 = stablehlo.broadcast_in_dim %v4575, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4578 = stablehlo.broadcast_in_dim %v4576, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4579 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4580 = stablehlo.add %v4578, %v4579 : tensor<64x512x28x28xf32>
    %v4581 = stablehlo.rsqrt %v4580 : tensor<64x512x28x28xf32>
    %v4582 = stablehlo.subtract %v4574, %v4577 : tensor<64x512x28x28xf32>
    %v4583 = stablehlo.multiply %v4582, %v4581 : tensor<64x512x28x28xf32>
    %v4584 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4585 = stablehlo.reshape %v4573 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4586 = stablehlo.multiply %v4584, %v4585 : tensor<64x512x28x28xf32>
    %v4587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4588 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v4589 = stablehlo.reduce(%v4586 init: %v4587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4590 = stablehlo.divide %v4589, %v4588 : tensor<512xf32>
    %v4591 = stablehlo.multiply %v4583, %v4586 : tensor<64x512x28x28xf32>
    %v4592 = stablehlo.reduce(%v4591 init: %v4587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4593 = stablehlo.divide %v4592, %v4588 : tensor<512xf32>
    %v4594 = stablehlo.concatenate %v4590, %v4593, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v4595 = stablehlo.concatenate %v992, %v4594, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b3g3dst = "stablehlo.all_reduce"(%v4595) ({
    ^bb0(%aras2b3g3dst: tensor<f32>, %arbs2b3g3dst: tensor<f32>):
      %aradds2b3g3dst = stablehlo.add %aras2b3g3dst, %arbs2b3g3dst : tensor<f32>
      stablehlo.return %aradds2b3g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b3g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b3g3dst = stablehlo.divide %arsums2b3g3dst, %arns2b3g3dst : tensor<2048xf32>
    %v4596 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4597 = stablehlo.slice %armeans2b3g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4598 = stablehlo.slice %armeans2b3g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4599 = stablehlo.slice %armeans2b3g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4600 = stablehlo.slice %armeans2b3g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4601 = stablehlo.broadcast_in_dim %v4597, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4602 = stablehlo.broadcast_in_dim %v4598, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4603 = stablehlo.broadcast_in_dim %v4599, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4604 = stablehlo.broadcast_in_dim %v4600, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4605 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4606 = stablehlo.add %v4602, %v4605 : tensor<64x512x28x28xf32>
    %v4607 = stablehlo.rsqrt %v4606 : tensor<64x512x28x28xf32>
    %v4608 = stablehlo.subtract %v4596, %v4601 : tensor<64x512x28x28xf32>
    %v4609 = stablehlo.multiply %v4608, %v4607 : tensor<64x512x28x28xf32>
    %v4610 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4611 = stablehlo.reshape %v4573 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4612 = stablehlo.multiply %v4610, %v4611 : tensor<64x512x28x28xf32>
    %v4613 = stablehlo.subtract %v4612, %v4603 : tensor<64x512x28x28xf32>
    %v4614 = stablehlo.multiply %v4609, %v4604 : tensor<64x512x28x28xf32>
    %v4615 = stablehlo.subtract %v4613, %v4614 : tensor<64x512x28x28xf32>
    %v4616 = stablehlo.multiply %v4607, %v4615 : tensor<64x512x28x28xf32>
    %v4617 = stablehlo.reshape %v4616 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4618 = stablehlo.reshape %v4617 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4619 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4620 = stablehlo.transpose %v4619, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4621 = stablehlo.convolution(%v4618, %v4620)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v4622 = stablehlo.reshape %v4621 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4623 = stablehlo.reshape %v4622 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4624 = stablehlo.reshape %v966 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4625 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4626 = stablehlo.compare GT, %v4624, %v4625 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4627 = stablehlo.select %v4626, %v4623, %v4625 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4628 = stablehlo.reshape %v4627 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4629 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4630 = stablehlo.slice %v951 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4631 = stablehlo.slice %v951 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4632 = stablehlo.broadcast_in_dim %v4630, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4633 = stablehlo.broadcast_in_dim %v4631, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4634 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4635 = stablehlo.add %v4633, %v4634 : tensor<64x128x28x28xf32>
    %v4636 = stablehlo.rsqrt %v4635 : tensor<64x128x28x28xf32>
    %v4637 = stablehlo.subtract %v4629, %v4632 : tensor<64x128x28x28xf32>
    %v4638 = stablehlo.multiply %v4637, %v4636 : tensor<64x128x28x28xf32>
    %v4639 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4640 = stablehlo.reshape %v4628 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4641 = stablehlo.multiply %v4639, %v4640 : tensor<64x128x28x28xf32>
    %v4642 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4643 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4644 = stablehlo.reduce(%v4641 init: %v4642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4645 = stablehlo.divide %v4644, %v4643 : tensor<128xf32>
    %v4646 = stablehlo.multiply %v4638, %v4641 : tensor<64x128x28x28xf32>
    %v4647 = stablehlo.reduce(%v4646 init: %v4642) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4648 = stablehlo.divide %v4647, %v4643 : tensor<128xf32>
    %v4649 = stablehlo.concatenate %v4645, %v4648, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4650 = stablehlo.concatenate %v951, %v4649, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b3g2dst = "stablehlo.all_reduce"(%v4650) ({
    ^bb0(%aras2b3g2dst: tensor<f32>, %arbs2b3g2dst: tensor<f32>):
      %aradds2b3g2dst = stablehlo.add %aras2b3g2dst, %arbs2b3g2dst : tensor<f32>
      stablehlo.return %aradds2b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g2dst = stablehlo.divide %arsums2b3g2dst, %arns2b3g2dst : tensor<512xf32>
    %v4651 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4652 = stablehlo.slice %armeans2b3g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4653 = stablehlo.slice %armeans2b3g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4654 = stablehlo.slice %armeans2b3g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4655 = stablehlo.slice %armeans2b3g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4656 = stablehlo.broadcast_in_dim %v4652, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4657 = stablehlo.broadcast_in_dim %v4653, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4658 = stablehlo.broadcast_in_dim %v4654, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4659 = stablehlo.broadcast_in_dim %v4655, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4660 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4661 = stablehlo.add %v4657, %v4660 : tensor<64x128x28x28xf32>
    %v4662 = stablehlo.rsqrt %v4661 : tensor<64x128x28x28xf32>
    %v4663 = stablehlo.subtract %v4651, %v4656 : tensor<64x128x28x28xf32>
    %v4664 = stablehlo.multiply %v4663, %v4662 : tensor<64x128x28x28xf32>
    %v4665 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4666 = stablehlo.reshape %v4628 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4667 = stablehlo.multiply %v4665, %v4666 : tensor<64x128x28x28xf32>
    %v4668 = stablehlo.subtract %v4667, %v4658 : tensor<64x128x28x28xf32>
    %v4669 = stablehlo.multiply %v4664, %v4659 : tensor<64x128x28x28xf32>
    %v4670 = stablehlo.subtract %v4668, %v4669 : tensor<64x128x28x28xf32>
    %v4671 = stablehlo.multiply %v4662, %v4670 : tensor<64x128x28x28xf32>
    %v4672 = stablehlo.reshape %v4671 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4673 = stablehlo.reshape %v4672 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4674 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4675 = stablehlo.transpose %v4674, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4676 = stablehlo.convolution(%v4673, %v4675)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v4677 = stablehlo.reshape %v4676 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4678 = stablehlo.reshape %v4677 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4679 = stablehlo.reshape %v925 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4680 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4681 = stablehlo.compare GT, %v4679, %v4680 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4682 = stablehlo.select %v4681, %v4678, %v4680 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4683 = stablehlo.reshape %v4682 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4684 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4685 = stablehlo.slice %v910 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4686 = stablehlo.slice %v910 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4687 = stablehlo.broadcast_in_dim %v4685, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4688 = stablehlo.broadcast_in_dim %v4686, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4689 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4690 = stablehlo.add %v4688, %v4689 : tensor<64x128x28x28xf32>
    %v4691 = stablehlo.rsqrt %v4690 : tensor<64x128x28x28xf32>
    %v4692 = stablehlo.subtract %v4684, %v4687 : tensor<64x128x28x28xf32>
    %v4693 = stablehlo.multiply %v4692, %v4691 : tensor<64x128x28x28xf32>
    %v4694 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4695 = stablehlo.reshape %v4683 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4696 = stablehlo.multiply %v4694, %v4695 : tensor<64x128x28x28xf32>
    %v4697 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4698 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4699 = stablehlo.reduce(%v4696 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4700 = stablehlo.divide %v4699, %v4698 : tensor<128xf32>
    %v4701 = stablehlo.multiply %v4693, %v4696 : tensor<64x128x28x28xf32>
    %v4702 = stablehlo.reduce(%v4701 init: %v4697) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4703 = stablehlo.divide %v4702, %v4698 : tensor<128xf32>
    %v4704 = stablehlo.concatenate %v4700, %v4703, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4705 = stablehlo.concatenate %v910, %v4704, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b3g1dst = "stablehlo.all_reduce"(%v4705) ({
    ^bb0(%aras2b3g1dst: tensor<f32>, %arbs2b3g1dst: tensor<f32>):
      %aradds2b3g1dst = stablehlo.add %aras2b3g1dst, %arbs2b3g1dst : tensor<f32>
      stablehlo.return %aradds2b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g1dst = stablehlo.divide %arsums2b3g1dst, %arns2b3g1dst : tensor<512xf32>
    %v4706 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4707 = stablehlo.slice %armeans2b3g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4708 = stablehlo.slice %armeans2b3g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4709 = stablehlo.slice %armeans2b3g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4710 = stablehlo.slice %armeans2b3g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4711 = stablehlo.broadcast_in_dim %v4707, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4712 = stablehlo.broadcast_in_dim %v4708, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4713 = stablehlo.broadcast_in_dim %v4709, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4714 = stablehlo.broadcast_in_dim %v4710, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4715 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4716 = stablehlo.add %v4712, %v4715 : tensor<64x128x28x28xf32>
    %v4717 = stablehlo.rsqrt %v4716 : tensor<64x128x28x28xf32>
    %v4718 = stablehlo.subtract %v4706, %v4711 : tensor<64x128x28x28xf32>
    %v4719 = stablehlo.multiply %v4718, %v4717 : tensor<64x128x28x28xf32>
    %v4720 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4721 = stablehlo.reshape %v4683 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4722 = stablehlo.multiply %v4720, %v4721 : tensor<64x128x28x28xf32>
    %v4723 = stablehlo.subtract %v4722, %v4713 : tensor<64x128x28x28xf32>
    %v4724 = stablehlo.multiply %v4719, %v4714 : tensor<64x128x28x28xf32>
    %v4725 = stablehlo.subtract %v4723, %v4724 : tensor<64x128x28x28xf32>
    %v4726 = stablehlo.multiply %v4717, %v4725 : tensor<64x128x28x28xf32>
    %v4727 = stablehlo.reshape %v4726 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4728 = stablehlo.reshape %v4727 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4729 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4730 = stablehlo.transpose %v4729, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4731 = stablehlo.convolution(%v4728, %v4730)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v4732 = stablehlo.reshape %v4731 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4733 = stablehlo.reshape %v4732 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4734 = stablehlo.reshape %v4573 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4735 = stablehlo.add %v4733, %v4734 : tensor<64x512x28x28xf32>
    %v4736 = stablehlo.reshape %v4735 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4737 = stablehlo.reshape %v886 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4738 = stablehlo.reshape %v4727 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4739 = stablehlo.transpose %v4737, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4740 = stablehlo.transpose %v4738, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4741 = stablehlo.convolution(%v4739, %v4740)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v4742 = stablehlo.transpose %v4741, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4743 = stablehlo.reshape %v891 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4744 = stablehlo.slice %v910 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4745 = stablehlo.slice %v910 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4746 = stablehlo.broadcast_in_dim %v4744, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4747 = stablehlo.broadcast_in_dim %v4745, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4748 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4749 = stablehlo.add %v4747, %v4748 : tensor<64x128x28x28xf32>
    %v4750 = stablehlo.rsqrt %v4749 : tensor<64x128x28x28xf32>
    %v4751 = stablehlo.subtract %v4743, %v4746 : tensor<64x128x28x28xf32>
    %v4752 = stablehlo.multiply %v4751, %v4750 : tensor<64x128x28x28xf32>
    %v4753 = stablehlo.reshape %v4683 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4754 = stablehlo.multiply %v4753, %v4752 : tensor<64x128x28x28xf32>
    %v4755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4756 = stablehlo.reduce(%v4754 init: %v4755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4757 = stablehlo.reshape %v4683 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4759 = stablehlo.reduce(%v4757 init: %v4758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4760 = stablehlo.reshape %v927 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4761 = stablehlo.reshape %v4672 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4762 = stablehlo.transpose %v4760, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4763 = stablehlo.transpose %v4761, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4764 = stablehlo.convolution(%v4762, %v4763)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v4765 = stablehlo.transpose %v4764, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4766 = stablehlo.reshape %v932 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4767 = stablehlo.slice %v951 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4768 = stablehlo.slice %v951 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4769 = stablehlo.broadcast_in_dim %v4767, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4770 = stablehlo.broadcast_in_dim %v4768, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4771 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4772 = stablehlo.add %v4770, %v4771 : tensor<64x128x28x28xf32>
    %v4773 = stablehlo.rsqrt %v4772 : tensor<64x128x28x28xf32>
    %v4774 = stablehlo.subtract %v4766, %v4769 : tensor<64x128x28x28xf32>
    %v4775 = stablehlo.multiply %v4774, %v4773 : tensor<64x128x28x28xf32>
    %v4776 = stablehlo.reshape %v4628 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4777 = stablehlo.multiply %v4776, %v4775 : tensor<64x128x28x28xf32>
    %v4778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4779 = stablehlo.reduce(%v4777 init: %v4778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4780 = stablehlo.reshape %v4628 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4782 = stablehlo.reduce(%v4780 init: %v4781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4783 = stablehlo.reshape %v968 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4784 = stablehlo.reshape %v4617 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4785 = stablehlo.transpose %v4783, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4786 = stablehlo.transpose %v4784, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4787 = stablehlo.convolution(%v4785, %v4786)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<512x64x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v4788 = stablehlo.transpose %v4787, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4789 = stablehlo.reshape %v973 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4790 = stablehlo.slice %v992 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4791 = stablehlo.slice %v992 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4792 = stablehlo.broadcast_in_dim %v4790, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4793 = stablehlo.broadcast_in_dim %v4791, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4794 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4795 = stablehlo.add %v4793, %v4794 : tensor<64x512x28x28xf32>
    %v4796 = stablehlo.rsqrt %v4795 : tensor<64x512x28x28xf32>
    %v4797 = stablehlo.subtract %v4789, %v4792 : tensor<64x512x28x28xf32>
    %v4798 = stablehlo.multiply %v4797, %v4796 : tensor<64x512x28x28xf32>
    %v4799 = stablehlo.reshape %v4573 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4800 = stablehlo.multiply %v4799, %v4798 : tensor<64x512x28x28xf32>
    %v4801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4802 = stablehlo.reduce(%v4800 init: %v4801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4803 = stablehlo.reshape %v4573 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4805 = stablehlo.reduce(%v4803 init: %v4804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4806 = stablehlo.reshape %v4736 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4807 = stablehlo.reshape %v882 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4808 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v4809 = stablehlo.compare GT, %v4807, %v4808 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v4810 = stablehlo.select %v4809, %v4806, %v4808 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v4811 = stablehlo.reshape %v4810 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4812 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4813 = stablehlo.slice %v863 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4814 = stablehlo.slice %v863 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4815 = stablehlo.broadcast_in_dim %v4813, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4816 = stablehlo.broadcast_in_dim %v4814, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4817 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4818 = stablehlo.add %v4816, %v4817 : tensor<64x512x28x28xf32>
    %v4819 = stablehlo.rsqrt %v4818 : tensor<64x512x28x28xf32>
    %v4820 = stablehlo.subtract %v4812, %v4815 : tensor<64x512x28x28xf32>
    %v4821 = stablehlo.multiply %v4820, %v4819 : tensor<64x512x28x28xf32>
    %v4822 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4823 = stablehlo.reshape %v4811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4824 = stablehlo.multiply %v4822, %v4823 : tensor<64x512x28x28xf32>
    %v4825 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4826 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v4827 = stablehlo.reduce(%v4824 init: %v4825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4828 = stablehlo.divide %v4827, %v4826 : tensor<512xf32>
    %v4829 = stablehlo.multiply %v4821, %v4824 : tensor<64x512x28x28xf32>
    %v4830 = stablehlo.reduce(%v4829 init: %v4825) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4831 = stablehlo.divide %v4830, %v4826 : tensor<512xf32>
    %v4832 = stablehlo.concatenate %v4828, %v4831, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v4833 = stablehlo.concatenate %v863, %v4832, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b2g3dst = "stablehlo.all_reduce"(%v4833) ({
    ^bb0(%aras2b2g3dst: tensor<f32>, %arbs2b2g3dst: tensor<f32>):
      %aradds2b2g3dst = stablehlo.add %aras2b2g3dst, %arbs2b2g3dst : tensor<f32>
      stablehlo.return %aradds2b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b2g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b2g3dst = stablehlo.divide %arsums2b2g3dst, %arns2b2g3dst : tensor<2048xf32>
    %v4834 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4835 = stablehlo.slice %armeans2b2g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4836 = stablehlo.slice %armeans2b2g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4837 = stablehlo.slice %armeans2b2g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4838 = stablehlo.slice %armeans2b2g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4839 = stablehlo.broadcast_in_dim %v4835, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4840 = stablehlo.broadcast_in_dim %v4836, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4841 = stablehlo.broadcast_in_dim %v4837, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4842 = stablehlo.broadcast_in_dim %v4838, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4843 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4844 = stablehlo.add %v4840, %v4843 : tensor<64x512x28x28xf32>
    %v4845 = stablehlo.rsqrt %v4844 : tensor<64x512x28x28xf32>
    %v4846 = stablehlo.subtract %v4834, %v4839 : tensor<64x512x28x28xf32>
    %v4847 = stablehlo.multiply %v4846, %v4845 : tensor<64x512x28x28xf32>
    %v4848 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4849 = stablehlo.reshape %v4811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4850 = stablehlo.multiply %v4848, %v4849 : tensor<64x512x28x28xf32>
    %v4851 = stablehlo.subtract %v4850, %v4841 : tensor<64x512x28x28xf32>
    %v4852 = stablehlo.multiply %v4847, %v4842 : tensor<64x512x28x28xf32>
    %v4853 = stablehlo.subtract %v4851, %v4852 : tensor<64x512x28x28xf32>
    %v4854 = stablehlo.multiply %v4845, %v4853 : tensor<64x512x28x28xf32>
    %v4855 = stablehlo.reshape %v4854 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4856 = stablehlo.reshape %v4855 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4857 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4858 = stablehlo.transpose %v4857, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4859 = stablehlo.convolution(%v4856, %v4858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v4860 = stablehlo.reshape %v4859 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4861 = stablehlo.reshape %v4860 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4862 = stablehlo.reshape %v837 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4863 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4864 = stablehlo.compare GT, %v4862, %v4863 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4865 = stablehlo.select %v4864, %v4861, %v4863 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4866 = stablehlo.reshape %v4865 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4867 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4868 = stablehlo.slice %v822 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4869 = stablehlo.slice %v822 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4870 = stablehlo.broadcast_in_dim %v4868, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4871 = stablehlo.broadcast_in_dim %v4869, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4872 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4873 = stablehlo.add %v4871, %v4872 : tensor<64x128x28x28xf32>
    %v4874 = stablehlo.rsqrt %v4873 : tensor<64x128x28x28xf32>
    %v4875 = stablehlo.subtract %v4867, %v4870 : tensor<64x128x28x28xf32>
    %v4876 = stablehlo.multiply %v4875, %v4874 : tensor<64x128x28x28xf32>
    %v4877 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4878 = stablehlo.reshape %v4866 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4879 = stablehlo.multiply %v4877, %v4878 : tensor<64x128x28x28xf32>
    %v4880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4881 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4882 = stablehlo.reduce(%v4879 init: %v4880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4883 = stablehlo.divide %v4882, %v4881 : tensor<128xf32>
    %v4884 = stablehlo.multiply %v4876, %v4879 : tensor<64x128x28x28xf32>
    %v4885 = stablehlo.reduce(%v4884 init: %v4880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4886 = stablehlo.divide %v4885, %v4881 : tensor<128xf32>
    %v4887 = stablehlo.concatenate %v4883, %v4886, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4888 = stablehlo.concatenate %v822, %v4887, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g2dst = "stablehlo.all_reduce"(%v4888) ({
    ^bb0(%aras2b2g2dst: tensor<f32>, %arbs2b2g2dst: tensor<f32>):
      %aradds2b2g2dst = stablehlo.add %aras2b2g2dst, %arbs2b2g2dst : tensor<f32>
      stablehlo.return %aradds2b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g2dst = stablehlo.divide %arsums2b2g2dst, %arns2b2g2dst : tensor<512xf32>
    %v4889 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4890 = stablehlo.slice %armeans2b2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4891 = stablehlo.slice %armeans2b2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4892 = stablehlo.slice %armeans2b2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4893 = stablehlo.slice %armeans2b2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4894 = stablehlo.broadcast_in_dim %v4890, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4895 = stablehlo.broadcast_in_dim %v4891, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4896 = stablehlo.broadcast_in_dim %v4892, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4897 = stablehlo.broadcast_in_dim %v4893, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4898 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4899 = stablehlo.add %v4895, %v4898 : tensor<64x128x28x28xf32>
    %v4900 = stablehlo.rsqrt %v4899 : tensor<64x128x28x28xf32>
    %v4901 = stablehlo.subtract %v4889, %v4894 : tensor<64x128x28x28xf32>
    %v4902 = stablehlo.multiply %v4901, %v4900 : tensor<64x128x28x28xf32>
    %v4903 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4904 = stablehlo.reshape %v4866 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4905 = stablehlo.multiply %v4903, %v4904 : tensor<64x128x28x28xf32>
    %v4906 = stablehlo.subtract %v4905, %v4896 : tensor<64x128x28x28xf32>
    %v4907 = stablehlo.multiply %v4902, %v4897 : tensor<64x128x28x28xf32>
    %v4908 = stablehlo.subtract %v4906, %v4907 : tensor<64x128x28x28xf32>
    %v4909 = stablehlo.multiply %v4900, %v4908 : tensor<64x128x28x28xf32>
    %v4910 = stablehlo.reshape %v4909 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4911 = stablehlo.reshape %v4910 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4912 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4913 = stablehlo.transpose %v4912, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4914 = stablehlo.convolution(%v4911, %v4913)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v4915 = stablehlo.reshape %v4914 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4916 = stablehlo.reshape %v4915 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4917 = stablehlo.reshape %v796 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4918 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4919 = stablehlo.compare GT, %v4917, %v4918 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4920 = stablehlo.select %v4919, %v4916, %v4918 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4921 = stablehlo.reshape %v4920 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4922 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4923 = stablehlo.slice %v781 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4924 = stablehlo.slice %v781 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4925 = stablehlo.broadcast_in_dim %v4923, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4926 = stablehlo.broadcast_in_dim %v4924, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4927 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4928 = stablehlo.add %v4926, %v4927 : tensor<64x128x28x28xf32>
    %v4929 = stablehlo.rsqrt %v4928 : tensor<64x128x28x28xf32>
    %v4930 = stablehlo.subtract %v4922, %v4925 : tensor<64x128x28x28xf32>
    %v4931 = stablehlo.multiply %v4930, %v4929 : tensor<64x128x28x28xf32>
    %v4932 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4933 = stablehlo.reshape %v4921 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4934 = stablehlo.multiply %v4932, %v4933 : tensor<64x128x28x28xf32>
    %v4935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4936 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4937 = stablehlo.reduce(%v4934 init: %v4935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4938 = stablehlo.divide %v4937, %v4936 : tensor<128xf32>
    %v4939 = stablehlo.multiply %v4931, %v4934 : tensor<64x128x28x28xf32>
    %v4940 = stablehlo.reduce(%v4939 init: %v4935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4941 = stablehlo.divide %v4940, %v4936 : tensor<128xf32>
    %v4942 = stablehlo.concatenate %v4938, %v4941, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4943 = stablehlo.concatenate %v781, %v4942, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g1dst = "stablehlo.all_reduce"(%v4943) ({
    ^bb0(%aras2b2g1dst: tensor<f32>, %arbs2b2g1dst: tensor<f32>):
      %aradds2b2g1dst = stablehlo.add %aras2b2g1dst, %arbs2b2g1dst : tensor<f32>
      stablehlo.return %aradds2b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g1dst = stablehlo.divide %arsums2b2g1dst, %arns2b2g1dst : tensor<512xf32>
    %v4944 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4945 = stablehlo.slice %armeans2b2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4946 = stablehlo.slice %armeans2b2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4947 = stablehlo.slice %armeans2b2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4948 = stablehlo.slice %armeans2b2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4949 = stablehlo.broadcast_in_dim %v4945, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4950 = stablehlo.broadcast_in_dim %v4946, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4951 = stablehlo.broadcast_in_dim %v4947, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4952 = stablehlo.broadcast_in_dim %v4948, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4953 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4954 = stablehlo.add %v4950, %v4953 : tensor<64x128x28x28xf32>
    %v4955 = stablehlo.rsqrt %v4954 : tensor<64x128x28x28xf32>
    %v4956 = stablehlo.subtract %v4944, %v4949 : tensor<64x128x28x28xf32>
    %v4957 = stablehlo.multiply %v4956, %v4955 : tensor<64x128x28x28xf32>
    %v4958 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4959 = stablehlo.reshape %v4921 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4960 = stablehlo.multiply %v4958, %v4959 : tensor<64x128x28x28xf32>
    %v4961 = stablehlo.subtract %v4960, %v4951 : tensor<64x128x28x28xf32>
    %v4962 = stablehlo.multiply %v4957, %v4952 : tensor<64x128x28x28xf32>
    %v4963 = stablehlo.subtract %v4961, %v4962 : tensor<64x128x28x28xf32>
    %v4964 = stablehlo.multiply %v4955, %v4963 : tensor<64x128x28x28xf32>
    %v4965 = stablehlo.reshape %v4964 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4966 = stablehlo.reshape %v4965 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4967 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4968 = stablehlo.transpose %v4967, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4969 = stablehlo.convolution(%v4966, %v4968)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4971 = stablehlo.reshape %v4970 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4972 = stablehlo.reshape %v4811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4973 = stablehlo.add %v4971, %v4972 : tensor<64x512x28x28xf32>
    %v4974 = stablehlo.reshape %v4973 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4975 = stablehlo.reshape %v757 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4976 = stablehlo.reshape %v4965 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4977 = stablehlo.transpose %v4975, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4978 = stablehlo.transpose %v4976, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4979 = stablehlo.convolution(%v4977, %v4978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v4980 = stablehlo.transpose %v4979, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4981 = stablehlo.reshape %v762 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4982 = stablehlo.slice %v781 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4983 = stablehlo.slice %v781 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4984 = stablehlo.broadcast_in_dim %v4982, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4985 = stablehlo.broadcast_in_dim %v4983, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4986 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4987 = stablehlo.add %v4985, %v4986 : tensor<64x128x28x28xf32>
    %v4988 = stablehlo.rsqrt %v4987 : tensor<64x128x28x28xf32>
    %v4989 = stablehlo.subtract %v4981, %v4984 : tensor<64x128x28x28xf32>
    %v4990 = stablehlo.multiply %v4989, %v4988 : tensor<64x128x28x28xf32>
    %v4991 = stablehlo.reshape %v4921 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4992 = stablehlo.multiply %v4991, %v4990 : tensor<64x128x28x28xf32>
    %v4993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4994 = stablehlo.reduce(%v4992 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4995 = stablehlo.reshape %v4921 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4997 = stablehlo.reduce(%v4995 init: %v4996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4998 = stablehlo.reshape %v798 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4999 = stablehlo.reshape %v4910 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5000 = stablehlo.transpose %v4998, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5001 = stablehlo.transpose %v4999, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5002 = stablehlo.convolution(%v5000, %v5001)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v5003 = stablehlo.transpose %v5002, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5004 = stablehlo.reshape %v803 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5005 = stablehlo.slice %v822 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5006 = stablehlo.slice %v822 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5007 = stablehlo.broadcast_in_dim %v5005, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5008 = stablehlo.broadcast_in_dim %v5006, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5009 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5010 = stablehlo.add %v5008, %v5009 : tensor<64x128x28x28xf32>
    %v5011 = stablehlo.rsqrt %v5010 : tensor<64x128x28x28xf32>
    %v5012 = stablehlo.subtract %v5004, %v5007 : tensor<64x128x28x28xf32>
    %v5013 = stablehlo.multiply %v5012, %v5011 : tensor<64x128x28x28xf32>
    %v5014 = stablehlo.reshape %v4866 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5015 = stablehlo.multiply %v5014, %v5013 : tensor<64x128x28x28xf32>
    %v5016 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5017 = stablehlo.reduce(%v5015 init: %v5016) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5018 = stablehlo.reshape %v4866 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5020 = stablehlo.reduce(%v5018 init: %v5019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5021 = stablehlo.reshape %v839 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5022 = stablehlo.reshape %v4855 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5023 = stablehlo.transpose %v5021, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5024 = stablehlo.transpose %v5022, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5025 = stablehlo.convolution(%v5023, %v5024)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<512x64x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v5026 = stablehlo.transpose %v5025, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5027 = stablehlo.reshape %v844 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5028 = stablehlo.slice %v863 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5029 = stablehlo.slice %v863 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5030 = stablehlo.broadcast_in_dim %v5028, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5031 = stablehlo.broadcast_in_dim %v5029, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5032 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5033 = stablehlo.add %v5031, %v5032 : tensor<64x512x28x28xf32>
    %v5034 = stablehlo.rsqrt %v5033 : tensor<64x512x28x28xf32>
    %v5035 = stablehlo.subtract %v5027, %v5030 : tensor<64x512x28x28xf32>
    %v5036 = stablehlo.multiply %v5035, %v5034 : tensor<64x512x28x28xf32>
    %v5037 = stablehlo.reshape %v4811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5038 = stablehlo.multiply %v5037, %v5036 : tensor<64x512x28x28xf32>
    %v5039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5040 = stablehlo.reduce(%v5038 init: %v5039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5041 = stablehlo.reshape %v4811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5043 = stablehlo.reduce(%v5041 init: %v5042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5044 = stablehlo.reshape %v4974 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5045 = stablehlo.reshape %v753 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5046 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v5047 = stablehlo.compare GT, %v5045, %v5046 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v5048 = stablehlo.select %v5047, %v5044, %v5046 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v5049 = stablehlo.reshape %v5048 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5050 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5051 = stablehlo.slice %v734 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5052 = stablehlo.slice %v734 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5053 = stablehlo.broadcast_in_dim %v5051, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5054 = stablehlo.broadcast_in_dim %v5052, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5055 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5056 = stablehlo.add %v5054, %v5055 : tensor<64x512x28x28xf32>
    %v5057 = stablehlo.rsqrt %v5056 : tensor<64x512x28x28xf32>
    %v5058 = stablehlo.subtract %v5050, %v5053 : tensor<64x512x28x28xf32>
    %v5059 = stablehlo.multiply %v5058, %v5057 : tensor<64x512x28x28xf32>
    %v5060 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5061 = stablehlo.reshape %v5049 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5062 = stablehlo.multiply %v5060, %v5061 : tensor<64x512x28x28xf32>
    %v5063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5064 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5065 = stablehlo.reduce(%v5062 init: %v5063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5066 = stablehlo.divide %v5065, %v5064 : tensor<512xf32>
    %v5067 = stablehlo.multiply %v5059, %v5062 : tensor<64x512x28x28xf32>
    %v5068 = stablehlo.reduce(%v5067 init: %v5063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5069 = stablehlo.divide %v5068, %v5064 : tensor<512xf32>
    %v5070 = stablehlo.concatenate %v5066, %v5069, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5071 = stablehlo.concatenate %v734, %v5070, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b1g3dst = "stablehlo.all_reduce"(%v5071) ({
    ^bb0(%aras2b1g3dst: tensor<f32>, %arbs2b1g3dst: tensor<f32>):
      %aradds2b1g3dst = stablehlo.add %aras2b1g3dst, %arbs2b1g3dst : tensor<f32>
      stablehlo.return %aradds2b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b1g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b1g3dst = stablehlo.divide %arsums2b1g3dst, %arns2b1g3dst : tensor<2048xf32>
    %v5072 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5073 = stablehlo.slice %armeans2b1g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5074 = stablehlo.slice %armeans2b1g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5075 = stablehlo.slice %armeans2b1g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5076 = stablehlo.slice %armeans2b1g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5077 = stablehlo.broadcast_in_dim %v5073, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5078 = stablehlo.broadcast_in_dim %v5074, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5079 = stablehlo.broadcast_in_dim %v5075, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5080 = stablehlo.broadcast_in_dim %v5076, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5081 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5082 = stablehlo.add %v5078, %v5081 : tensor<64x512x28x28xf32>
    %v5083 = stablehlo.rsqrt %v5082 : tensor<64x512x28x28xf32>
    %v5084 = stablehlo.subtract %v5072, %v5077 : tensor<64x512x28x28xf32>
    %v5085 = stablehlo.multiply %v5084, %v5083 : tensor<64x512x28x28xf32>
    %v5086 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5087 = stablehlo.reshape %v5049 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5088 = stablehlo.multiply %v5086, %v5087 : tensor<64x512x28x28xf32>
    %v5089 = stablehlo.subtract %v5088, %v5079 : tensor<64x512x28x28xf32>
    %v5090 = stablehlo.multiply %v5085, %v5080 : tensor<64x512x28x28xf32>
    %v5091 = stablehlo.subtract %v5089, %v5090 : tensor<64x512x28x28xf32>
    %v5092 = stablehlo.multiply %v5083, %v5091 : tensor<64x512x28x28xf32>
    %v5093 = stablehlo.reshape %v5092 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5094 = stablehlo.reshape %v5093 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5095 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v5096 = stablehlo.transpose %v5095, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5097 = stablehlo.convolution(%v5094, %v5096)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v5098 = stablehlo.reshape %v5097 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5099 = stablehlo.reshape %v5098 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5100 = stablehlo.reshape %v708 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5101 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5102 = stablehlo.compare GT, %v5100, %v5101 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5103 = stablehlo.select %v5102, %v5099, %v5101 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5104 = stablehlo.reshape %v5103 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5105 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5106 = stablehlo.slice %v693 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5107 = stablehlo.slice %v693 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5108 = stablehlo.broadcast_in_dim %v5106, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5109 = stablehlo.broadcast_in_dim %v5107, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5110 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5111 = stablehlo.add %v5109, %v5110 : tensor<64x128x28x28xf32>
    %v5112 = stablehlo.rsqrt %v5111 : tensor<64x128x28x28xf32>
    %v5113 = stablehlo.subtract %v5105, %v5108 : tensor<64x128x28x28xf32>
    %v5114 = stablehlo.multiply %v5113, %v5112 : tensor<64x128x28x28xf32>
    %v5115 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5116 = stablehlo.reshape %v5104 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5117 = stablehlo.multiply %v5115, %v5116 : tensor<64x128x28x28xf32>
    %v5118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5119 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5120 = stablehlo.reduce(%v5117 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5121 = stablehlo.divide %v5120, %v5119 : tensor<128xf32>
    %v5122 = stablehlo.multiply %v5114, %v5117 : tensor<64x128x28x28xf32>
    %v5123 = stablehlo.reduce(%v5122 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5124 = stablehlo.divide %v5123, %v5119 : tensor<128xf32>
    %v5125 = stablehlo.concatenate %v5121, %v5124, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5126 = stablehlo.concatenate %v693, %v5125, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g2dst = "stablehlo.all_reduce"(%v5126) ({
    ^bb0(%aras2b1g2dst: tensor<f32>, %arbs2b1g2dst: tensor<f32>):
      %aradds2b1g2dst = stablehlo.add %aras2b1g2dst, %arbs2b1g2dst : tensor<f32>
      stablehlo.return %aradds2b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g2dst = stablehlo.divide %arsums2b1g2dst, %arns2b1g2dst : tensor<512xf32>
    %v5127 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5128 = stablehlo.slice %armeans2b1g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5129 = stablehlo.slice %armeans2b1g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5130 = stablehlo.slice %armeans2b1g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5131 = stablehlo.slice %armeans2b1g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5132 = stablehlo.broadcast_in_dim %v5128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5133 = stablehlo.broadcast_in_dim %v5129, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5134 = stablehlo.broadcast_in_dim %v5130, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5135 = stablehlo.broadcast_in_dim %v5131, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5136 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5137 = stablehlo.add %v5133, %v5136 : tensor<64x128x28x28xf32>
    %v5138 = stablehlo.rsqrt %v5137 : tensor<64x128x28x28xf32>
    %v5139 = stablehlo.subtract %v5127, %v5132 : tensor<64x128x28x28xf32>
    %v5140 = stablehlo.multiply %v5139, %v5138 : tensor<64x128x28x28xf32>
    %v5141 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5142 = stablehlo.reshape %v5104 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5143 = stablehlo.multiply %v5141, %v5142 : tensor<64x128x28x28xf32>
    %v5144 = stablehlo.subtract %v5143, %v5134 : tensor<64x128x28x28xf32>
    %v5145 = stablehlo.multiply %v5140, %v5135 : tensor<64x128x28x28xf32>
    %v5146 = stablehlo.subtract %v5144, %v5145 : tensor<64x128x28x28xf32>
    %v5147 = stablehlo.multiply %v5138, %v5146 : tensor<64x128x28x28xf32>
    %v5148 = stablehlo.reshape %v5147 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5149 = stablehlo.reshape %v5148 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5150 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5151 = stablehlo.transpose %v5150, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5152 = stablehlo.convolution(%v5149, %v5151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x28x28xf32>
    %v5153 = stablehlo.reshape %v5152 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5154 = stablehlo.reshape %v5153 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5155 = stablehlo.reshape %v667 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5156 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5157 = stablehlo.compare GT, %v5155, %v5156 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5158 = stablehlo.select %v5157, %v5154, %v5156 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5159 = stablehlo.reshape %v5158 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5160 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5161 = stablehlo.slice %v652 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5162 = stablehlo.slice %v652 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5163 = stablehlo.broadcast_in_dim %v5161, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5164 = stablehlo.broadcast_in_dim %v5162, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5165 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5166 = stablehlo.add %v5164, %v5165 : tensor<64x128x28x28xf32>
    %v5167 = stablehlo.rsqrt %v5166 : tensor<64x128x28x28xf32>
    %v5168 = stablehlo.subtract %v5160, %v5163 : tensor<64x128x28x28xf32>
    %v5169 = stablehlo.multiply %v5168, %v5167 : tensor<64x128x28x28xf32>
    %v5170 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5171 = stablehlo.reshape %v5159 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5172 = stablehlo.multiply %v5170, %v5171 : tensor<64x128x28x28xf32>
    %v5173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5174 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5175 = stablehlo.reduce(%v5172 init: %v5173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5176 = stablehlo.divide %v5175, %v5174 : tensor<128xf32>
    %v5177 = stablehlo.multiply %v5169, %v5172 : tensor<64x128x28x28xf32>
    %v5178 = stablehlo.reduce(%v5177 init: %v5173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5179 = stablehlo.divide %v5178, %v5174 : tensor<128xf32>
    %v5180 = stablehlo.concatenate %v5176, %v5179, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5181 = stablehlo.concatenate %v652, %v5180, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g1dst = "stablehlo.all_reduce"(%v5181) ({
    ^bb0(%aras2b1g1dst: tensor<f32>, %arbs2b1g1dst: tensor<f32>):
      %aradds2b1g1dst = stablehlo.add %aras2b1g1dst, %arbs2b1g1dst : tensor<f32>
      stablehlo.return %aradds2b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g1dst = stablehlo.divide %arsums2b1g1dst, %arns2b1g1dst : tensor<512xf32>
    %v5182 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5183 = stablehlo.slice %armeans2b1g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5184 = stablehlo.slice %armeans2b1g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5185 = stablehlo.slice %armeans2b1g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5186 = stablehlo.slice %armeans2b1g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5187 = stablehlo.broadcast_in_dim %v5183, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5188 = stablehlo.broadcast_in_dim %v5184, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5189 = stablehlo.broadcast_in_dim %v5185, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5190 = stablehlo.broadcast_in_dim %v5186, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5191 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5192 = stablehlo.add %v5188, %v5191 : tensor<64x128x28x28xf32>
    %v5193 = stablehlo.rsqrt %v5192 : tensor<64x128x28x28xf32>
    %v5194 = stablehlo.subtract %v5182, %v5187 : tensor<64x128x28x28xf32>
    %v5195 = stablehlo.multiply %v5194, %v5193 : tensor<64x128x28x28xf32>
    %v5196 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5197 = stablehlo.reshape %v5159 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5198 = stablehlo.multiply %v5196, %v5197 : tensor<64x128x28x28xf32>
    %v5199 = stablehlo.subtract %v5198, %v5189 : tensor<64x128x28x28xf32>
    %v5200 = stablehlo.multiply %v5195, %v5190 : tensor<64x128x28x28xf32>
    %v5201 = stablehlo.subtract %v5199, %v5200 : tensor<64x128x28x28xf32>
    %v5202 = stablehlo.multiply %v5193, %v5201 : tensor<64x128x28x28xf32>
    %v5203 = stablehlo.reshape %v5202 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5204 = stablehlo.reshape %v5203 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5205 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v5206 = stablehlo.transpose %v5205, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5207 = stablehlo.convolution(%v5204, %v5206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<512x128x1x1xf32>) -> tensor<64x512x28x28xf32>
    %v5208 = stablehlo.reshape %v5207 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5209 = stablehlo.reshape %v5208 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5210 = stablehlo.reshape %v5049 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5211 = stablehlo.add %v5209, %v5210 : tensor<64x512x28x28xf32>
    %v5212 = stablehlo.reshape %v5211 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5213 = stablehlo.reshape %v628 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5214 = stablehlo.reshape %v5203 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5215 = stablehlo.transpose %v5213, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5216 = stablehlo.transpose %v5214, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5217 = stablehlo.convolution(%v5215, %v5216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<512x128x1x1xf32>
    %v5218 = stablehlo.transpose %v5217, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5219 = stablehlo.reshape %v633 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5220 = stablehlo.slice %v652 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5221 = stablehlo.slice %v652 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5222 = stablehlo.broadcast_in_dim %v5220, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5223 = stablehlo.broadcast_in_dim %v5221, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5224 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5225 = stablehlo.add %v5223, %v5224 : tensor<64x128x28x28xf32>
    %v5226 = stablehlo.rsqrt %v5225 : tensor<64x128x28x28xf32>
    %v5227 = stablehlo.subtract %v5219, %v5222 : tensor<64x128x28x28xf32>
    %v5228 = stablehlo.multiply %v5227, %v5226 : tensor<64x128x28x28xf32>
    %v5229 = stablehlo.reshape %v5159 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5230 = stablehlo.multiply %v5229, %v5228 : tensor<64x128x28x28xf32>
    %v5231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5232 = stablehlo.reduce(%v5230 init: %v5231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5233 = stablehlo.reshape %v5159 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5234 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5235 = stablehlo.reduce(%v5233 init: %v5234) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5236 = stablehlo.reshape %v669 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5237 = stablehlo.reshape %v5148 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5238 = stablehlo.transpose %v5236, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5239 = stablehlo.transpose %v5237, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5240 = stablehlo.convolution(%v5238, %v5239)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<128x64x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v5241 = stablehlo.transpose %v5240, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5242 = stablehlo.reshape %v674 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5243 = stablehlo.slice %v693 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5244 = stablehlo.slice %v693 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5245 = stablehlo.broadcast_in_dim %v5243, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5246 = stablehlo.broadcast_in_dim %v5244, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5247 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5248 = stablehlo.add %v5246, %v5247 : tensor<64x128x28x28xf32>
    %v5249 = stablehlo.rsqrt %v5248 : tensor<64x128x28x28xf32>
    %v5250 = stablehlo.subtract %v5242, %v5245 : tensor<64x128x28x28xf32>
    %v5251 = stablehlo.multiply %v5250, %v5249 : tensor<64x128x28x28xf32>
    %v5252 = stablehlo.reshape %v5104 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5253 = stablehlo.multiply %v5252, %v5251 : tensor<64x128x28x28xf32>
    %v5254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5255 = stablehlo.reduce(%v5253 init: %v5254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5256 = stablehlo.reshape %v5104 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5258 = stablehlo.reduce(%v5256 init: %v5257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5259 = stablehlo.reshape %v710 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5260 = stablehlo.reshape %v5093 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5261 = stablehlo.transpose %v5259, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5262 = stablehlo.transpose %v5260, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5263 = stablehlo.convolution(%v5261, %v5262)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<512x64x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v5264 = stablehlo.transpose %v5263, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5265 = stablehlo.reshape %v715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5266 = stablehlo.slice %v734 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5267 = stablehlo.slice %v734 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5268 = stablehlo.broadcast_in_dim %v5266, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5269 = stablehlo.broadcast_in_dim %v5267, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5270 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5271 = stablehlo.add %v5269, %v5270 : tensor<64x512x28x28xf32>
    %v5272 = stablehlo.rsqrt %v5271 : tensor<64x512x28x28xf32>
    %v5273 = stablehlo.subtract %v5265, %v5268 : tensor<64x512x28x28xf32>
    %v5274 = stablehlo.multiply %v5273, %v5272 : tensor<64x512x28x28xf32>
    %v5275 = stablehlo.reshape %v5049 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5276 = stablehlo.multiply %v5275, %v5274 : tensor<64x512x28x28xf32>
    %v5277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5278 = stablehlo.reduce(%v5276 init: %v5277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5279 = stablehlo.reshape %v5049 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5281 = stablehlo.reduce(%v5279 init: %v5280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5282 = stablehlo.reshape %v5212 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5283 = stablehlo.reshape %v626 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5284 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v5285 = stablehlo.compare GT, %v5283, %v5284 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v5286 = stablehlo.select %v5285, %v5282, %v5284 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v5287 = stablehlo.reshape %v5286 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5288 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5289 = stablehlo.slice %v571 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5290 = stablehlo.slice %v571 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5291 = stablehlo.broadcast_in_dim %v5289, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5292 = stablehlo.broadcast_in_dim %v5290, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5293 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5294 = stablehlo.add %v5292, %v5293 : tensor<64x512x28x28xf32>
    %v5295 = stablehlo.rsqrt %v5294 : tensor<64x512x28x28xf32>
    %v5296 = stablehlo.subtract %v5288, %v5291 : tensor<64x512x28x28xf32>
    %v5297 = stablehlo.multiply %v5296, %v5295 : tensor<64x512x28x28xf32>
    %v5298 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5299 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5300 = stablehlo.multiply %v5298, %v5299 : tensor<64x512x28x28xf32>
    %v5301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5302 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5303 = stablehlo.reduce(%v5300 init: %v5301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5304 = stablehlo.divide %v5303, %v5302 : tensor<512xf32>
    %v5305 = stablehlo.multiply %v5297, %v5300 : tensor<64x512x28x28xf32>
    %v5306 = stablehlo.reduce(%v5305 init: %v5301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5307 = stablehlo.divide %v5306, %v5302 : tensor<512xf32>
    %v5308 = stablehlo.concatenate %v5304, %v5307, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5309 = stablehlo.concatenate %v571, %v5308, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b0g3dst = "stablehlo.all_reduce"(%v5309) ({
    ^bb0(%aras2b0g3dst: tensor<f32>, %arbs2b0g3dst: tensor<f32>):
      %aradds2b0g3dst = stablehlo.add %aras2b0g3dst, %arbs2b0g3dst : tensor<f32>
      stablehlo.return %aradds2b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b0g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b0g3dst = stablehlo.divide %arsums2b0g3dst, %arns2b0g3dst : tensor<2048xf32>
    %v5310 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5311 = stablehlo.slice %armeans2b0g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5312 = stablehlo.slice %armeans2b0g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5313 = stablehlo.slice %armeans2b0g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5314 = stablehlo.slice %armeans2b0g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5315 = stablehlo.broadcast_in_dim %v5311, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5316 = stablehlo.broadcast_in_dim %v5312, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5317 = stablehlo.broadcast_in_dim %v5313, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5318 = stablehlo.broadcast_in_dim %v5314, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5319 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5320 = stablehlo.add %v5316, %v5319 : tensor<64x512x28x28xf32>
    %v5321 = stablehlo.rsqrt %v5320 : tensor<64x512x28x28xf32>
    %v5322 = stablehlo.subtract %v5310, %v5315 : tensor<64x512x28x28xf32>
    %v5323 = stablehlo.multiply %v5322, %v5321 : tensor<64x512x28x28xf32>
    %v5324 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5325 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5326 = stablehlo.multiply %v5324, %v5325 : tensor<64x512x28x28xf32>
    %v5327 = stablehlo.subtract %v5326, %v5317 : tensor<64x512x28x28xf32>
    %v5328 = stablehlo.multiply %v5323, %v5318 : tensor<64x512x28x28xf32>
    %v5329 = stablehlo.subtract %v5327, %v5328 : tensor<64x512x28x28xf32>
    %v5330 = stablehlo.multiply %v5321, %v5329 : tensor<64x512x28x28xf32>
    %v5331 = stablehlo.reshape %v5330 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5332 = stablehlo.reshape %v5331 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5333 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v5334 = stablehlo.transpose %v5333, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5335 = stablehlo.convolution(%v5332, %v5334)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<128x512x1x1xf32>) -> tensor<64x128x28x28xf32>
    %v5336 = stablehlo.reshape %v5335 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5337 = stablehlo.reshape %v5336 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5338 = stablehlo.reshape %v545 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5339 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5340 = stablehlo.compare GT, %v5338, %v5339 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5341 = stablehlo.select %v5340, %v5337, %v5339 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5342 = stablehlo.reshape %v5341 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5343 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5344 = stablehlo.slice %v530 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5345 = stablehlo.slice %v530 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5346 = stablehlo.broadcast_in_dim %v5344, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5347 = stablehlo.broadcast_in_dim %v5345, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5348 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5349 = stablehlo.add %v5347, %v5348 : tensor<64x128x28x28xf32>
    %v5350 = stablehlo.rsqrt %v5349 : tensor<64x128x28x28xf32>
    %v5351 = stablehlo.subtract %v5343, %v5346 : tensor<64x128x28x28xf32>
    %v5352 = stablehlo.multiply %v5351, %v5350 : tensor<64x128x28x28xf32>
    %v5353 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5354 = stablehlo.reshape %v5342 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5355 = stablehlo.multiply %v5353, %v5354 : tensor<64x128x28x28xf32>
    %v5356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5357 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5358 = stablehlo.reduce(%v5355 init: %v5356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5359 = stablehlo.divide %v5358, %v5357 : tensor<128xf32>
    %v5360 = stablehlo.multiply %v5352, %v5355 : tensor<64x128x28x28xf32>
    %v5361 = stablehlo.reduce(%v5360 init: %v5356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5362 = stablehlo.divide %v5361, %v5357 : tensor<128xf32>
    %v5363 = stablehlo.concatenate %v5359, %v5362, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5364 = stablehlo.concatenate %v530, %v5363, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g2dst = "stablehlo.all_reduce"(%v5364) ({
    ^bb0(%aras2b0g2dst: tensor<f32>, %arbs2b0g2dst: tensor<f32>):
      %aradds2b0g2dst = stablehlo.add %aras2b0g2dst, %arbs2b0g2dst : tensor<f32>
      stablehlo.return %aradds2b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g2dst = stablehlo.divide %arsums2b0g2dst, %arns2b0g2dst : tensor<512xf32>
    %v5365 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5366 = stablehlo.slice %armeans2b0g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5367 = stablehlo.slice %armeans2b0g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5368 = stablehlo.slice %armeans2b0g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5369 = stablehlo.slice %armeans2b0g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5370 = stablehlo.broadcast_in_dim %v5366, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5371 = stablehlo.broadcast_in_dim %v5367, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5372 = stablehlo.broadcast_in_dim %v5368, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5373 = stablehlo.broadcast_in_dim %v5369, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5374 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5375 = stablehlo.add %v5371, %v5374 : tensor<64x128x28x28xf32>
    %v5376 = stablehlo.rsqrt %v5375 : tensor<64x128x28x28xf32>
    %v5377 = stablehlo.subtract %v5365, %v5370 : tensor<64x128x28x28xf32>
    %v5378 = stablehlo.multiply %v5377, %v5376 : tensor<64x128x28x28xf32>
    %v5379 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5380 = stablehlo.reshape %v5342 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5381 = stablehlo.multiply %v5379, %v5380 : tensor<64x128x28x28xf32>
    %v5382 = stablehlo.subtract %v5381, %v5372 : tensor<64x128x28x28xf32>
    %v5383 = stablehlo.multiply %v5378, %v5373 : tensor<64x128x28x28xf32>
    %v5384 = stablehlo.subtract %v5382, %v5383 : tensor<64x128x28x28xf32>
    %v5385 = stablehlo.multiply %v5376, %v5384 : tensor<64x128x28x28xf32>
    %v5386 = stablehlo.reshape %v5385 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5387 = stablehlo.reshape %v5386 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5388 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5389 = stablehlo.transpose %v5388, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5390 = stablehlo.convolution(%v5387, %v5389)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 2], [1, 2]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xf32>, tensor<128x128x3x3xf32>) -> tensor<64x128x56x56xf32>
    %v5391 = stablehlo.reshape %v5390 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5392 = stablehlo.reshape %v5391 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5393 = stablehlo.reshape %v504 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5394 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v5395 = stablehlo.compare GT, %v5393, %v5394 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v5396 = stablehlo.select %v5395, %v5392, %v5394 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v5397 = stablehlo.reshape %v5396 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5398 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5399 = stablehlo.slice %v489 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5400 = stablehlo.slice %v489 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5401 = stablehlo.broadcast_in_dim %v5399, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5402 = stablehlo.broadcast_in_dim %v5400, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5403 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5404 = stablehlo.add %v5402, %v5403 : tensor<64x128x56x56xf32>
    %v5405 = stablehlo.rsqrt %v5404 : tensor<64x128x56x56xf32>
    %v5406 = stablehlo.subtract %v5398, %v5401 : tensor<64x128x56x56xf32>
    %v5407 = stablehlo.multiply %v5406, %v5405 : tensor<64x128x56x56xf32>
    %v5408 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5409 = stablehlo.reshape %v5397 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5410 = stablehlo.multiply %v5408, %v5409 : tensor<64x128x56x56xf32>
    %v5411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5412 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5413 = stablehlo.reduce(%v5410 init: %v5411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5414 = stablehlo.divide %v5413, %v5412 : tensor<128xf32>
    %v5415 = stablehlo.multiply %v5407, %v5410 : tensor<64x128x56x56xf32>
    %v5416 = stablehlo.reduce(%v5415 init: %v5411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5417 = stablehlo.divide %v5416, %v5412 : tensor<128xf32>
    %v5418 = stablehlo.concatenate %v5414, %v5417, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5419 = stablehlo.concatenate %v489, %v5418, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g1dst = "stablehlo.all_reduce"(%v5419) ({
    ^bb0(%aras2b0g1dst: tensor<f32>, %arbs2b0g1dst: tensor<f32>):
      %aradds2b0g1dst = stablehlo.add %aras2b0g1dst, %arbs2b0g1dst : tensor<f32>
      stablehlo.return %aradds2b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g1dst = stablehlo.divide %arsums2b0g1dst, %arns2b0g1dst : tensor<512xf32>
    %v5420 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5421 = stablehlo.slice %armeans2b0g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5422 = stablehlo.slice %armeans2b0g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5423 = stablehlo.slice %armeans2b0g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5424 = stablehlo.slice %armeans2b0g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5425 = stablehlo.broadcast_in_dim %v5421, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5426 = stablehlo.broadcast_in_dim %v5422, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5427 = stablehlo.broadcast_in_dim %v5423, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5428 = stablehlo.broadcast_in_dim %v5424, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5429 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5430 = stablehlo.add %v5426, %v5429 : tensor<64x128x56x56xf32>
    %v5431 = stablehlo.rsqrt %v5430 : tensor<64x128x56x56xf32>
    %v5432 = stablehlo.subtract %v5420, %v5425 : tensor<64x128x56x56xf32>
    %v5433 = stablehlo.multiply %v5432, %v5431 : tensor<64x128x56x56xf32>
    %v5434 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5435 = stablehlo.reshape %v5397 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5436 = stablehlo.multiply %v5434, %v5435 : tensor<64x128x56x56xf32>
    %v5437 = stablehlo.subtract %v5436, %v5427 : tensor<64x128x56x56xf32>
    %v5438 = stablehlo.multiply %v5433, %v5428 : tensor<64x128x56x56xf32>
    %v5439 = stablehlo.subtract %v5437, %v5438 : tensor<64x128x56x56xf32>
    %v5440 = stablehlo.multiply %v5431, %v5439 : tensor<64x128x56x56xf32>
    %v5441 = stablehlo.reshape %v5440 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5442 = stablehlo.reshape %v5441 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5443 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v5444 = stablehlo.transpose %v5443, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v5445 = stablehlo.convolution(%v5442, %v5444)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xf32>, tensor<256x128x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v5446 = stablehlo.reshape %v5445 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5447 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5448 = stablehlo.slice %v610 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5449 = stablehlo.slice %v610 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5450 = stablehlo.broadcast_in_dim %v5448, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5451 = stablehlo.broadcast_in_dim %v5449, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5452 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5453 = stablehlo.add %v5451, %v5452 : tensor<64x512x28x28xf32>
    %v5454 = stablehlo.rsqrt %v5453 : tensor<64x512x28x28xf32>
    %v5455 = stablehlo.subtract %v5447, %v5450 : tensor<64x512x28x28xf32>
    %v5456 = stablehlo.multiply %v5455, %v5454 : tensor<64x512x28x28xf32>
    %v5457 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5458 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5459 = stablehlo.multiply %v5457, %v5458 : tensor<64x512x28x28xf32>
    %v5460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5461 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5462 = stablehlo.reduce(%v5459 init: %v5460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5463 = stablehlo.divide %v5462, %v5461 : tensor<512xf32>
    %v5464 = stablehlo.multiply %v5456, %v5459 : tensor<64x512x28x28xf32>
    %v5465 = stablehlo.reduce(%v5464 init: %v5460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5466 = stablehlo.divide %v5465, %v5461 : tensor<512xf32>
    %v5467 = stablehlo.concatenate %v5463, %v5466, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5468 = stablehlo.concatenate %v610, %v5467, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b0gpdst = "stablehlo.all_reduce"(%v5468) ({
    ^bb0(%aras2b0gpdst: tensor<f32>, %arbs2b0gpdst: tensor<f32>):
      %aradds2b0gpdst = stablehlo.add %aras2b0gpdst, %arbs2b0gpdst : tensor<f32>
      stablehlo.return %aradds2b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b0gpdst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b0gpdst = stablehlo.divide %arsums2b0gpdst, %arns2b0gpdst : tensor<2048xf32>
    %v5469 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5470 = stablehlo.slice %armeans2b0gpdst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5471 = stablehlo.slice %armeans2b0gpdst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5472 = stablehlo.slice %armeans2b0gpdst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5473 = stablehlo.slice %armeans2b0gpdst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5474 = stablehlo.broadcast_in_dim %v5470, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5475 = stablehlo.broadcast_in_dim %v5471, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5476 = stablehlo.broadcast_in_dim %v5472, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5477 = stablehlo.broadcast_in_dim %v5473, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5478 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5479 = stablehlo.add %v5475, %v5478 : tensor<64x512x28x28xf32>
    %v5480 = stablehlo.rsqrt %v5479 : tensor<64x512x28x28xf32>
    %v5481 = stablehlo.subtract %v5469, %v5474 : tensor<64x512x28x28xf32>
    %v5482 = stablehlo.multiply %v5481, %v5480 : tensor<64x512x28x28xf32>
    %v5483 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5484 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5485 = stablehlo.multiply %v5483, %v5484 : tensor<64x512x28x28xf32>
    %v5486 = stablehlo.subtract %v5485, %v5476 : tensor<64x512x28x28xf32>
    %v5487 = stablehlo.multiply %v5482, %v5477 : tensor<64x512x28x28xf32>
    %v5488 = stablehlo.subtract %v5486, %v5487 : tensor<64x512x28x28xf32>
    %v5489 = stablehlo.multiply %v5480, %v5488 : tensor<64x512x28x28xf32>
    %v5490 = stablehlo.reshape %v5489 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5491 = stablehlo.reshape %v5490 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5492 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v5493 = stablehlo.transpose %v5492, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v5494 = stablehlo.convolution(%v5491, %v5493)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 1], [0, 1]], lhs_dilate = [2, 2], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xf32>, tensor<256x512x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v5495 = stablehlo.reshape %v5494 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5496 = stablehlo.reshape %v5446 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5497 = stablehlo.reshape %v5495 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5498 = stablehlo.add %v5496, %v5497 : tensor<64x256x56x56xf32>
    %v5499 = stablehlo.reshape %v5498 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5500 = stablehlo.reshape %v465 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5501 = stablehlo.reshape %v5441 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5502 = stablehlo.transpose %v5500, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5503 = stablehlo.transpose %v5501, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v5504 = stablehlo.convolution(%v5502, %v5503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<128x64x56x56xf32>) -> tensor<256x128x1x1xf32>
    %v5505 = stablehlo.transpose %v5504, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v5506 = stablehlo.reshape %v470 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5507 = stablehlo.slice %v489 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5508 = stablehlo.slice %v489 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5509 = stablehlo.broadcast_in_dim %v5507, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5510 = stablehlo.broadcast_in_dim %v5508, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5511 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5512 = stablehlo.add %v5510, %v5511 : tensor<64x128x56x56xf32>
    %v5513 = stablehlo.rsqrt %v5512 : tensor<64x128x56x56xf32>
    %v5514 = stablehlo.subtract %v5506, %v5509 : tensor<64x128x56x56xf32>
    %v5515 = stablehlo.multiply %v5514, %v5513 : tensor<64x128x56x56xf32>
    %v5516 = stablehlo.reshape %v5397 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5517 = stablehlo.multiply %v5516, %v5515 : tensor<64x128x56x56xf32>
    %v5518 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5519 = stablehlo.reduce(%v5517 init: %v5518) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5520 = stablehlo.reshape %v5397 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5522 = stablehlo.reduce(%v5520 init: %v5521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5523 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5524 = stablehlo.reshape %v5386 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5525 = stablehlo.transpose %v5523, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v5527 = stablehlo.transpose %v5524, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5528 = stablehlo.convolution(%v5525, %v5527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 0], [1, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xf32>, tensor<128x64x28x28xf32>) -> tensor<128x128x3x3xf32>
    %v5529 = stablehlo.transpose %v5528, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5530 = stablehlo.reshape %v511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5531 = stablehlo.slice %v530 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5532 = stablehlo.slice %v530 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5533 = stablehlo.broadcast_in_dim %v5531, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5534 = stablehlo.broadcast_in_dim %v5532, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5535 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5536 = stablehlo.add %v5534, %v5535 : tensor<64x128x28x28xf32>
    %v5537 = stablehlo.rsqrt %v5536 : tensor<64x128x28x28xf32>
    %v5538 = stablehlo.subtract %v5530, %v5533 : tensor<64x128x28x28xf32>
    %v5539 = stablehlo.multiply %v5538, %v5537 : tensor<64x128x28x28xf32>
    %v5540 = stablehlo.reshape %v5342 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5541 = stablehlo.multiply %v5540, %v5539 : tensor<64x128x28x28xf32>
    %v5542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5543 = stablehlo.reduce(%v5541 init: %v5542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5544 = stablehlo.reshape %v5342 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5546 = stablehlo.reduce(%v5544 init: %v5545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5547 = stablehlo.reshape %v547 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5548 = stablehlo.reshape %v5331 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5549 = stablehlo.transpose %v5547, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5550 = stablehlo.transpose %v5548, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5551 = stablehlo.convolution(%v5549, %v5550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xf32>, tensor<512x64x28x28xf32>) -> tensor<128x512x1x1xf32>
    %v5552 = stablehlo.transpose %v5551, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5553 = stablehlo.reshape %v552 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5554 = stablehlo.slice %v571 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5555 = stablehlo.slice %v571 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5556 = stablehlo.broadcast_in_dim %v5554, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5557 = stablehlo.broadcast_in_dim %v5555, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5558 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5559 = stablehlo.add %v5557, %v5558 : tensor<64x512x28x28xf32>
    %v5560 = stablehlo.rsqrt %v5559 : tensor<64x512x28x28xf32>
    %v5561 = stablehlo.subtract %v5553, %v5556 : tensor<64x512x28x28xf32>
    %v5562 = stablehlo.multiply %v5561, %v5560 : tensor<64x512x28x28xf32>
    %v5563 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5564 = stablehlo.multiply %v5563, %v5562 : tensor<64x512x28x28xf32>
    %v5565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5566 = stablehlo.reduce(%v5564 init: %v5565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5567 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5569 = stablehlo.reduce(%v5567 init: %v5568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5570 = stablehlo.reshape %v465 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5571 = stablehlo.reshape %v5490 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5572 = stablehlo.transpose %v5570, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5574 = stablehlo.transpose %v5571, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5573 = stablehlo.slice %v5572 [0:256, 0:64, 0:55, 0:55] : (tensor<256x64x56x56xf32>) -> tensor<256x64x55x55xf32>
    %v5575 = stablehlo.convolution(%v5573, %v5574)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x55x55xf32>, tensor<512x64x28x28xf32>) -> tensor<256x512x1x1xf32>
    %v5576 = stablehlo.transpose %v5575, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v5577 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5578 = stablehlo.slice %v610 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5579 = stablehlo.slice %v610 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5580 = stablehlo.broadcast_in_dim %v5578, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5581 = stablehlo.broadcast_in_dim %v5579, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5582 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5583 = stablehlo.add %v5581, %v5582 : tensor<64x512x28x28xf32>
    %v5584 = stablehlo.rsqrt %v5583 : tensor<64x512x28x28xf32>
    %v5585 = stablehlo.subtract %v5577, %v5580 : tensor<64x512x28x28xf32>
    %v5586 = stablehlo.multiply %v5585, %v5584 : tensor<64x512x28x28xf32>
    %v5587 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5588 = stablehlo.multiply %v5587, %v5586 : tensor<64x512x28x28xf32>
    %v5589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5590 = stablehlo.reduce(%v5588 init: %v5589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5591 = stablehlo.reshape %v5287 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5593 = stablehlo.reduce(%v5591 init: %v5592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5594 = stablehlo.reshape %v5499 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5595 = stablehlo.reshape %v461 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5596 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v5597 = stablehlo.compare GT, %v5595, %v5596 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v5598 = stablehlo.select %v5597, %v5594, %v5596 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v5599 = stablehlo.reshape %v5598 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5600 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5601 = stablehlo.slice %v442 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v5602 = stablehlo.slice %v442 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v5603 = stablehlo.broadcast_in_dim %v5601, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5604 = stablehlo.broadcast_in_dim %v5602, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5605 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5606 = stablehlo.add %v5604, %v5605 : tensor<64x256x56x56xf32>
    %v5607 = stablehlo.rsqrt %v5606 : tensor<64x256x56x56xf32>
    %v5608 = stablehlo.subtract %v5600, %v5603 : tensor<64x256x56x56xf32>
    %v5609 = stablehlo.multiply %v5608, %v5607 : tensor<64x256x56x56xf32>
    %v5610 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5611 = stablehlo.reshape %v5599 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5612 = stablehlo.multiply %v5610, %v5611 : tensor<64x256x56x56xf32>
    %v5613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5614 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5615 = stablehlo.reduce(%v5612 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5616 = stablehlo.divide %v5615, %v5614 : tensor<256xf32>
    %v5617 = stablehlo.multiply %v5609, %v5612 : tensor<64x256x56x56xf32>
    %v5618 = stablehlo.reduce(%v5617 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5619 = stablehlo.divide %v5618, %v5614 : tensor<256xf32>
    %v5620 = stablehlo.concatenate %v5616, %v5619, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v5621 = stablehlo.concatenate %v442, %v5620, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b2g3dst = "stablehlo.all_reduce"(%v5621) ({
    ^bb0(%aras1b2g3dst: tensor<f32>, %arbs1b2g3dst: tensor<f32>):
      %aradds1b2g3dst = stablehlo.add %aras1b2g3dst, %arbs1b2g3dst : tensor<f32>
      stablehlo.return %aradds1b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b2g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b2g3dst = stablehlo.divide %arsums1b2g3dst, %arns1b2g3dst : tensor<1024xf32>
    %v5622 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5623 = stablehlo.slice %armeans1b2g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5624 = stablehlo.slice %armeans1b2g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5625 = stablehlo.slice %armeans1b2g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5626 = stablehlo.slice %armeans1b2g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5627 = stablehlo.broadcast_in_dim %v5623, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5628 = stablehlo.broadcast_in_dim %v5624, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5629 = stablehlo.broadcast_in_dim %v5625, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5630 = stablehlo.broadcast_in_dim %v5626, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5631 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5632 = stablehlo.add %v5628, %v5631 : tensor<64x256x56x56xf32>
    %v5633 = stablehlo.rsqrt %v5632 : tensor<64x256x56x56xf32>
    %v5634 = stablehlo.subtract %v5622, %v5627 : tensor<64x256x56x56xf32>
    %v5635 = stablehlo.multiply %v5634, %v5633 : tensor<64x256x56x56xf32>
    %v5636 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5637 = stablehlo.reshape %v5599 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5638 = stablehlo.multiply %v5636, %v5637 : tensor<64x256x56x56xf32>
    %v5639 = stablehlo.subtract %v5638, %v5629 : tensor<64x256x56x56xf32>
    %v5640 = stablehlo.multiply %v5635, %v5630 : tensor<64x256x56x56xf32>
    %v5641 = stablehlo.subtract %v5639, %v5640 : tensor<64x256x56x56xf32>
    %v5642 = stablehlo.multiply %v5633, %v5641 : tensor<64x256x56x56xf32>
    %v5643 = stablehlo.reshape %v5642 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5644 = stablehlo.reshape %v5643 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5645 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5646 = stablehlo.transpose %v5645, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5647 = stablehlo.convolution(%v5644, %v5646)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v5648 = stablehlo.reshape %v5647 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5649 = stablehlo.reshape %v5648 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5650 = stablehlo.reshape %v416 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5651 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5652 = stablehlo.compare GT, %v5650, %v5651 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5653 = stablehlo.select %v5652, %v5649, %v5651 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5654 = stablehlo.reshape %v5653 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5655 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5656 = stablehlo.slice %v401 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5657 = stablehlo.slice %v401 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5658 = stablehlo.broadcast_in_dim %v5656, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5659 = stablehlo.broadcast_in_dim %v5657, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5660 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5661 = stablehlo.add %v5659, %v5660 : tensor<64x64x56x56xf32>
    %v5662 = stablehlo.rsqrt %v5661 : tensor<64x64x56x56xf32>
    %v5663 = stablehlo.subtract %v5655, %v5658 : tensor<64x64x56x56xf32>
    %v5664 = stablehlo.multiply %v5663, %v5662 : tensor<64x64x56x56xf32>
    %v5665 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5666 = stablehlo.reshape %v5654 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5667 = stablehlo.multiply %v5665, %v5666 : tensor<64x64x56x56xf32>
    %v5668 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5669 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5670 = stablehlo.reduce(%v5667 init: %v5668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5671 = stablehlo.divide %v5670, %v5669 : tensor<64xf32>
    %v5672 = stablehlo.multiply %v5664, %v5667 : tensor<64x64x56x56xf32>
    %v5673 = stablehlo.reduce(%v5672 init: %v5668) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5674 = stablehlo.divide %v5673, %v5669 : tensor<64xf32>
    %v5675 = stablehlo.concatenate %v5671, %v5674, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v5676 = stablehlo.concatenate %v401, %v5675, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g2dst = "stablehlo.all_reduce"(%v5676) ({
    ^bb0(%aras1b2g2dst: tensor<f32>, %arbs1b2g2dst: tensor<f32>):
      %aradds1b2g2dst = stablehlo.add %aras1b2g2dst, %arbs1b2g2dst : tensor<f32>
      stablehlo.return %aradds1b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g2dst = stablehlo.divide %arsums1b2g2dst, %arns1b2g2dst : tensor<256xf32>
    %v5677 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5678 = stablehlo.slice %armeans1b2g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v5679 = stablehlo.slice %armeans1b2g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v5680 = stablehlo.slice %armeans1b2g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v5681 = stablehlo.slice %armeans1b2g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v5682 = stablehlo.broadcast_in_dim %v5678, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5683 = stablehlo.broadcast_in_dim %v5679, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5684 = stablehlo.broadcast_in_dim %v5680, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5685 = stablehlo.broadcast_in_dim %v5681, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5686 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5687 = stablehlo.add %v5683, %v5686 : tensor<64x64x56x56xf32>
    %v5688 = stablehlo.rsqrt %v5687 : tensor<64x64x56x56xf32>
    %v5689 = stablehlo.subtract %v5677, %v5682 : tensor<64x64x56x56xf32>
    %v5690 = stablehlo.multiply %v5689, %v5688 : tensor<64x64x56x56xf32>
    %v5691 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5692 = stablehlo.reshape %v5654 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5693 = stablehlo.multiply %v5691, %v5692 : tensor<64x64x56x56xf32>
    %v5694 = stablehlo.subtract %v5693, %v5684 : tensor<64x64x56x56xf32>
    %v5695 = stablehlo.multiply %v5690, %v5685 : tensor<64x64x56x56xf32>
    %v5696 = stablehlo.subtract %v5694, %v5695 : tensor<64x64x56x56xf32>
    %v5697 = stablehlo.multiply %v5688, %v5696 : tensor<64x64x56x56xf32>
    %v5698 = stablehlo.reshape %v5697 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5699 = stablehlo.reshape %v5698 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5700 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5701 = stablehlo.transpose %v5700, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5702 = stablehlo.convolution(%v5699, %v5701)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v5703 = stablehlo.reshape %v5702 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5704 = stablehlo.reshape %v5703 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5705 = stablehlo.reshape %v375 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5706 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5707 = stablehlo.compare GT, %v5705, %v5706 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5708 = stablehlo.select %v5707, %v5704, %v5706 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5709 = stablehlo.reshape %v5708 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5710 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5711 = stablehlo.slice %v360 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5712 = stablehlo.slice %v360 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5713 = stablehlo.broadcast_in_dim %v5711, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5714 = stablehlo.broadcast_in_dim %v5712, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5715 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5716 = stablehlo.add %v5714, %v5715 : tensor<64x64x56x56xf32>
    %v5717 = stablehlo.rsqrt %v5716 : tensor<64x64x56x56xf32>
    %v5718 = stablehlo.subtract %v5710, %v5713 : tensor<64x64x56x56xf32>
    %v5719 = stablehlo.multiply %v5718, %v5717 : tensor<64x64x56x56xf32>
    %v5720 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5721 = stablehlo.reshape %v5709 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5722 = stablehlo.multiply %v5720, %v5721 : tensor<64x64x56x56xf32>
    %v5723 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5724 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5725 = stablehlo.reduce(%v5722 init: %v5723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5726 = stablehlo.divide %v5725, %v5724 : tensor<64xf32>
    %v5727 = stablehlo.multiply %v5719, %v5722 : tensor<64x64x56x56xf32>
    %v5728 = stablehlo.reduce(%v5727 init: %v5723) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5729 = stablehlo.divide %v5728, %v5724 : tensor<64xf32>
    %v5730 = stablehlo.concatenate %v5726, %v5729, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v5731 = stablehlo.concatenate %v360, %v5730, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g1dst = "stablehlo.all_reduce"(%v5731) ({
    ^bb0(%aras1b2g1dst: tensor<f32>, %arbs1b2g1dst: tensor<f32>):
      %aradds1b2g1dst = stablehlo.add %aras1b2g1dst, %arbs1b2g1dst : tensor<f32>
      stablehlo.return %aradds1b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g1dst = stablehlo.divide %arsums1b2g1dst, %arns1b2g1dst : tensor<256xf32>
    %v5732 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5733 = stablehlo.slice %armeans1b2g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v5734 = stablehlo.slice %armeans1b2g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v5735 = stablehlo.slice %armeans1b2g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v5736 = stablehlo.slice %armeans1b2g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v5737 = stablehlo.broadcast_in_dim %v5733, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5738 = stablehlo.broadcast_in_dim %v5734, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5739 = stablehlo.broadcast_in_dim %v5735, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5740 = stablehlo.broadcast_in_dim %v5736, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5741 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5742 = stablehlo.add %v5738, %v5741 : tensor<64x64x56x56xf32>
    %v5743 = stablehlo.rsqrt %v5742 : tensor<64x64x56x56xf32>
    %v5744 = stablehlo.subtract %v5732, %v5737 : tensor<64x64x56x56xf32>
    %v5745 = stablehlo.multiply %v5744, %v5743 : tensor<64x64x56x56xf32>
    %v5746 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5747 = stablehlo.reshape %v5709 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5748 = stablehlo.multiply %v5746, %v5747 : tensor<64x64x56x56xf32>
    %v5749 = stablehlo.subtract %v5748, %v5739 : tensor<64x64x56x56xf32>
    %v5750 = stablehlo.multiply %v5745, %v5740 : tensor<64x64x56x56xf32>
    %v5751 = stablehlo.subtract %v5749, %v5750 : tensor<64x64x56x56xf32>
    %v5752 = stablehlo.multiply %v5743, %v5751 : tensor<64x64x56x56xf32>
    %v5753 = stablehlo.reshape %v5752 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5754 = stablehlo.reshape %v5753 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5755 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v5756 = stablehlo.transpose %v5755, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5757 = stablehlo.convolution(%v5754, %v5756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v5758 = stablehlo.reshape %v5757 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5759 = stablehlo.reshape %v5758 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5760 = stablehlo.reshape %v5599 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5761 = stablehlo.add %v5759, %v5760 : tensor<64x256x56x56xf32>
    %v5762 = stablehlo.reshape %v5761 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5763 = stablehlo.reshape %v336 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5764 = stablehlo.reshape %v5753 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5765 = stablehlo.transpose %v5763, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5766 = stablehlo.transpose %v5764, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5767 = stablehlo.convolution(%v5765, %v5766)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v5768 = stablehlo.transpose %v5767, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5769 = stablehlo.reshape %v341 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5770 = stablehlo.slice %v360 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5771 = stablehlo.slice %v360 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5772 = stablehlo.broadcast_in_dim %v5770, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5773 = stablehlo.broadcast_in_dim %v5771, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5774 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5775 = stablehlo.add %v5773, %v5774 : tensor<64x64x56x56xf32>
    %v5776 = stablehlo.rsqrt %v5775 : tensor<64x64x56x56xf32>
    %v5777 = stablehlo.subtract %v5769, %v5772 : tensor<64x64x56x56xf32>
    %v5778 = stablehlo.multiply %v5777, %v5776 : tensor<64x64x56x56xf32>
    %v5779 = stablehlo.reshape %v5709 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5780 = stablehlo.multiply %v5779, %v5778 : tensor<64x64x56x56xf32>
    %v5781 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5782 = stablehlo.reduce(%v5780 init: %v5781) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5783 = stablehlo.reshape %v5709 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5785 = stablehlo.reduce(%v5783 init: %v5784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5786 = stablehlo.reshape %v377 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5787 = stablehlo.reshape %v5698 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5788 = stablehlo.transpose %v5786, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5789 = stablehlo.transpose %v5787, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5790 = stablehlo.convolution(%v5788, %v5789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v5791 = stablehlo.transpose %v5790, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5792 = stablehlo.reshape %v382 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5793 = stablehlo.slice %v401 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5794 = stablehlo.slice %v401 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5795 = stablehlo.broadcast_in_dim %v5793, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5796 = stablehlo.broadcast_in_dim %v5794, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5797 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5798 = stablehlo.add %v5796, %v5797 : tensor<64x64x56x56xf32>
    %v5799 = stablehlo.rsqrt %v5798 : tensor<64x64x56x56xf32>
    %v5800 = stablehlo.subtract %v5792, %v5795 : tensor<64x64x56x56xf32>
    %v5801 = stablehlo.multiply %v5800, %v5799 : tensor<64x64x56x56xf32>
    %v5802 = stablehlo.reshape %v5654 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5803 = stablehlo.multiply %v5802, %v5801 : tensor<64x64x56x56xf32>
    %v5804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5805 = stablehlo.reduce(%v5803 init: %v5804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5806 = stablehlo.reshape %v5654 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5807 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5808 = stablehlo.reduce(%v5806 init: %v5807) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5809 = stablehlo.reshape %v418 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5810 = stablehlo.reshape %v5643 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5811 = stablehlo.transpose %v5809, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5812 = stablehlo.transpose %v5810, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5813 = stablehlo.convolution(%v5811, %v5812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v5814 = stablehlo.transpose %v5813, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5815 = stablehlo.reshape %v423 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5816 = stablehlo.slice %v442 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v5817 = stablehlo.slice %v442 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v5818 = stablehlo.broadcast_in_dim %v5816, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5819 = stablehlo.broadcast_in_dim %v5817, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5820 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5821 = stablehlo.add %v5819, %v5820 : tensor<64x256x56x56xf32>
    %v5822 = stablehlo.rsqrt %v5821 : tensor<64x256x56x56xf32>
    %v5823 = stablehlo.subtract %v5815, %v5818 : tensor<64x256x56x56xf32>
    %v5824 = stablehlo.multiply %v5823, %v5822 : tensor<64x256x56x56xf32>
    %v5825 = stablehlo.reshape %v5599 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5826 = stablehlo.multiply %v5825, %v5824 : tensor<64x256x56x56xf32>
    %v5827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5828 = stablehlo.reduce(%v5826 init: %v5827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5829 = stablehlo.reshape %v5599 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5831 = stablehlo.reduce(%v5829 init: %v5830) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5832 = stablehlo.reshape %v5762 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5833 = stablehlo.reshape %v332 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5834 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v5835 = stablehlo.compare GT, %v5833, %v5834 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v5836 = stablehlo.select %v5835, %v5832, %v5834 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v5837 = stablehlo.reshape %v5836 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5838 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5839 = stablehlo.slice %v313 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v5840 = stablehlo.slice %v313 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v5841 = stablehlo.broadcast_in_dim %v5839, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5842 = stablehlo.broadcast_in_dim %v5840, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5843 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5844 = stablehlo.add %v5842, %v5843 : tensor<64x256x56x56xf32>
    %v5845 = stablehlo.rsqrt %v5844 : tensor<64x256x56x56xf32>
    %v5846 = stablehlo.subtract %v5838, %v5841 : tensor<64x256x56x56xf32>
    %v5847 = stablehlo.multiply %v5846, %v5845 : tensor<64x256x56x56xf32>
    %v5848 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5849 = stablehlo.reshape %v5837 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5850 = stablehlo.multiply %v5848, %v5849 : tensor<64x256x56x56xf32>
    %v5851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5852 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5853 = stablehlo.reduce(%v5850 init: %v5851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5854 = stablehlo.divide %v5853, %v5852 : tensor<256xf32>
    %v5855 = stablehlo.multiply %v5847, %v5850 : tensor<64x256x56x56xf32>
    %v5856 = stablehlo.reduce(%v5855 init: %v5851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5857 = stablehlo.divide %v5856, %v5852 : tensor<256xf32>
    %v5858 = stablehlo.concatenate %v5854, %v5857, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v5859 = stablehlo.concatenate %v313, %v5858, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b1g3dst = "stablehlo.all_reduce"(%v5859) ({
    ^bb0(%aras1b1g3dst: tensor<f32>, %arbs1b1g3dst: tensor<f32>):
      %aradds1b1g3dst = stablehlo.add %aras1b1g3dst, %arbs1b1g3dst : tensor<f32>
      stablehlo.return %aradds1b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b1g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b1g3dst = stablehlo.divide %arsums1b1g3dst, %arns1b1g3dst : tensor<1024xf32>
    %v5860 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5861 = stablehlo.slice %armeans1b1g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5862 = stablehlo.slice %armeans1b1g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5863 = stablehlo.slice %armeans1b1g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5864 = stablehlo.slice %armeans1b1g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v5865 = stablehlo.broadcast_in_dim %v5861, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5866 = stablehlo.broadcast_in_dim %v5862, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5867 = stablehlo.broadcast_in_dim %v5863, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5868 = stablehlo.broadcast_in_dim %v5864, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5869 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5870 = stablehlo.add %v5866, %v5869 : tensor<64x256x56x56xf32>
    %v5871 = stablehlo.rsqrt %v5870 : tensor<64x256x56x56xf32>
    %v5872 = stablehlo.subtract %v5860, %v5865 : tensor<64x256x56x56xf32>
    %v5873 = stablehlo.multiply %v5872, %v5871 : tensor<64x256x56x56xf32>
    %v5874 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5875 = stablehlo.reshape %v5837 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5876 = stablehlo.multiply %v5874, %v5875 : tensor<64x256x56x56xf32>
    %v5877 = stablehlo.subtract %v5876, %v5867 : tensor<64x256x56x56xf32>
    %v5878 = stablehlo.multiply %v5873, %v5868 : tensor<64x256x56x56xf32>
    %v5879 = stablehlo.subtract %v5877, %v5878 : tensor<64x256x56x56xf32>
    %v5880 = stablehlo.multiply %v5871, %v5879 : tensor<64x256x56x56xf32>
    %v5881 = stablehlo.reshape %v5880 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5882 = stablehlo.reshape %v5881 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5883 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5884 = stablehlo.transpose %v5883, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5885 = stablehlo.convolution(%v5882, %v5884)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v5886 = stablehlo.reshape %v5885 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5887 = stablehlo.reshape %v5886 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5888 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5889 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5890 = stablehlo.compare GT, %v5888, %v5889 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5891 = stablehlo.select %v5890, %v5887, %v5889 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5892 = stablehlo.reshape %v5891 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5893 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5894 = stablehlo.slice %v272 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5895 = stablehlo.slice %v272 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5896 = stablehlo.broadcast_in_dim %v5894, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5897 = stablehlo.broadcast_in_dim %v5895, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5898 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5899 = stablehlo.add %v5897, %v5898 : tensor<64x64x56x56xf32>
    %v5900 = stablehlo.rsqrt %v5899 : tensor<64x64x56x56xf32>
    %v5901 = stablehlo.subtract %v5893, %v5896 : tensor<64x64x56x56xf32>
    %v5902 = stablehlo.multiply %v5901, %v5900 : tensor<64x64x56x56xf32>
    %v5903 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5904 = stablehlo.reshape %v5892 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5905 = stablehlo.multiply %v5903, %v5904 : tensor<64x64x56x56xf32>
    %v5906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5907 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5908 = stablehlo.reduce(%v5905 init: %v5906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5909 = stablehlo.divide %v5908, %v5907 : tensor<64xf32>
    %v5910 = stablehlo.multiply %v5902, %v5905 : tensor<64x64x56x56xf32>
    %v5911 = stablehlo.reduce(%v5910 init: %v5906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5912 = stablehlo.divide %v5911, %v5907 : tensor<64xf32>
    %v5913 = stablehlo.concatenate %v5909, %v5912, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v5914 = stablehlo.concatenate %v272, %v5913, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g2dst = "stablehlo.all_reduce"(%v5914) ({
    ^bb0(%aras1b1g2dst: tensor<f32>, %arbs1b1g2dst: tensor<f32>):
      %aradds1b1g2dst = stablehlo.add %aras1b1g2dst, %arbs1b1g2dst : tensor<f32>
      stablehlo.return %aradds1b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g2dst = stablehlo.divide %arsums1b1g2dst, %arns1b1g2dst : tensor<256xf32>
    %v5915 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5916 = stablehlo.slice %armeans1b1g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v5917 = stablehlo.slice %armeans1b1g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v5918 = stablehlo.slice %armeans1b1g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v5919 = stablehlo.slice %armeans1b1g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v5920 = stablehlo.broadcast_in_dim %v5916, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5921 = stablehlo.broadcast_in_dim %v5917, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5922 = stablehlo.broadcast_in_dim %v5918, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5923 = stablehlo.broadcast_in_dim %v5919, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5924 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5925 = stablehlo.add %v5921, %v5924 : tensor<64x64x56x56xf32>
    %v5926 = stablehlo.rsqrt %v5925 : tensor<64x64x56x56xf32>
    %v5927 = stablehlo.subtract %v5915, %v5920 : tensor<64x64x56x56xf32>
    %v5928 = stablehlo.multiply %v5927, %v5926 : tensor<64x64x56x56xf32>
    %v5929 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5930 = stablehlo.reshape %v5892 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5931 = stablehlo.multiply %v5929, %v5930 : tensor<64x64x56x56xf32>
    %v5932 = stablehlo.subtract %v5931, %v5922 : tensor<64x64x56x56xf32>
    %v5933 = stablehlo.multiply %v5928, %v5923 : tensor<64x64x56x56xf32>
    %v5934 = stablehlo.subtract %v5932, %v5933 : tensor<64x64x56x56xf32>
    %v5935 = stablehlo.multiply %v5926, %v5934 : tensor<64x64x56x56xf32>
    %v5936 = stablehlo.reshape %v5935 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5937 = stablehlo.reshape %v5936 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5938 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5939 = stablehlo.transpose %v5938, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5940 = stablehlo.convolution(%v5937, %v5939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v5941 = stablehlo.reshape %v5940 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5942 = stablehlo.reshape %v5941 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5943 = stablehlo.reshape %v246 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5944 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5945 = stablehlo.compare GT, %v5943, %v5944 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5946 = stablehlo.select %v5945, %v5942, %v5944 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5947 = stablehlo.reshape %v5946 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5948 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5949 = stablehlo.slice %v231 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v5950 = stablehlo.slice %v231 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v5951 = stablehlo.broadcast_in_dim %v5949, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5952 = stablehlo.broadcast_in_dim %v5950, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5953 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5954 = stablehlo.add %v5952, %v5953 : tensor<64x64x56x56xf32>
    %v5955 = stablehlo.rsqrt %v5954 : tensor<64x64x56x56xf32>
    %v5956 = stablehlo.subtract %v5948, %v5951 : tensor<64x64x56x56xf32>
    %v5957 = stablehlo.multiply %v5956, %v5955 : tensor<64x64x56x56xf32>
    %v5958 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5959 = stablehlo.reshape %v5947 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5960 = stablehlo.multiply %v5958, %v5959 : tensor<64x64x56x56xf32>
    %v5961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5962 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5963 = stablehlo.reduce(%v5960 init: %v5961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5964 = stablehlo.divide %v5963, %v5962 : tensor<64xf32>
    %v5965 = stablehlo.multiply %v5957, %v5960 : tensor<64x64x56x56xf32>
    %v5966 = stablehlo.reduce(%v5965 init: %v5961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5967 = stablehlo.divide %v5966, %v5962 : tensor<64xf32>
    %v5968 = stablehlo.concatenate %v5964, %v5967, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v5969 = stablehlo.concatenate %v231, %v5968, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g1dst = "stablehlo.all_reduce"(%v5969) ({
    ^bb0(%aras1b1g1dst: tensor<f32>, %arbs1b1g1dst: tensor<f32>):
      %aradds1b1g1dst = stablehlo.add %aras1b1g1dst, %arbs1b1g1dst : tensor<f32>
      stablehlo.return %aradds1b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g1dst = stablehlo.divide %arsums1b1g1dst, %arns1b1g1dst : tensor<256xf32>
    %v5970 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5971 = stablehlo.slice %armeans1b1g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v5972 = stablehlo.slice %armeans1b1g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v5973 = stablehlo.slice %armeans1b1g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v5974 = stablehlo.slice %armeans1b1g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v5975 = stablehlo.broadcast_in_dim %v5971, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5976 = stablehlo.broadcast_in_dim %v5972, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5977 = stablehlo.broadcast_in_dim %v5973, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5978 = stablehlo.broadcast_in_dim %v5974, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5979 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5980 = stablehlo.add %v5976, %v5979 : tensor<64x64x56x56xf32>
    %v5981 = stablehlo.rsqrt %v5980 : tensor<64x64x56x56xf32>
    %v5982 = stablehlo.subtract %v5970, %v5975 : tensor<64x64x56x56xf32>
    %v5983 = stablehlo.multiply %v5982, %v5981 : tensor<64x64x56x56xf32>
    %v5984 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5985 = stablehlo.reshape %v5947 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5986 = stablehlo.multiply %v5984, %v5985 : tensor<64x64x56x56xf32>
    %v5987 = stablehlo.subtract %v5986, %v5977 : tensor<64x64x56x56xf32>
    %v5988 = stablehlo.multiply %v5983, %v5978 : tensor<64x64x56x56xf32>
    %v5989 = stablehlo.subtract %v5987, %v5988 : tensor<64x64x56x56xf32>
    %v5990 = stablehlo.multiply %v5981, %v5989 : tensor<64x64x56x56xf32>
    %v5991 = stablehlo.reshape %v5990 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5992 = stablehlo.reshape %v5991 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5993 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v5994 = stablehlo.transpose %v5993, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5995 = stablehlo.convolution(%v5992, %v5994)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x1x1xf32>) -> tensor<64x256x56x56xf32>
    %v5996 = stablehlo.reshape %v5995 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5997 = stablehlo.reshape %v5996 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5998 = stablehlo.reshape %v5837 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5999 = stablehlo.add %v5997, %v5998 : tensor<64x256x56x56xf32>
    %v6000 = stablehlo.reshape %v5999 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6001 = stablehlo.reshape %v207 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6002 = stablehlo.reshape %v5991 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6003 = stablehlo.transpose %v6001, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6004 = stablehlo.transpose %v6002, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6005 = stablehlo.convolution(%v6003, %v6004)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<256x64x1x1xf32>
    %v6006 = stablehlo.transpose %v6005, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6007 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6008 = stablehlo.slice %v231 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6009 = stablehlo.slice %v231 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6010 = stablehlo.broadcast_in_dim %v6008, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6011 = stablehlo.broadcast_in_dim %v6009, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6012 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6013 = stablehlo.add %v6011, %v6012 : tensor<64x64x56x56xf32>
    %v6014 = stablehlo.rsqrt %v6013 : tensor<64x64x56x56xf32>
    %v6015 = stablehlo.subtract %v6007, %v6010 : tensor<64x64x56x56xf32>
    %v6016 = stablehlo.multiply %v6015, %v6014 : tensor<64x64x56x56xf32>
    %v6017 = stablehlo.reshape %v5947 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6018 = stablehlo.multiply %v6017, %v6016 : tensor<64x64x56x56xf32>
    %v6019 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6020 = stablehlo.reduce(%v6018 init: %v6019) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6021 = stablehlo.reshape %v5947 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6022 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6023 = stablehlo.reduce(%v6021 init: %v6022) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6024 = stablehlo.reshape %v248 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6025 = stablehlo.reshape %v5936 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6026 = stablehlo.transpose %v6024, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6027 = stablehlo.transpose %v6025, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6028 = stablehlo.convolution(%v6026, %v6027)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v6029 = stablehlo.transpose %v6028, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6030 = stablehlo.reshape %v253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6031 = stablehlo.slice %v272 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6032 = stablehlo.slice %v272 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6033 = stablehlo.broadcast_in_dim %v6031, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6034 = stablehlo.broadcast_in_dim %v6032, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6035 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6036 = stablehlo.add %v6034, %v6035 : tensor<64x64x56x56xf32>
    %v6037 = stablehlo.rsqrt %v6036 : tensor<64x64x56x56xf32>
    %v6038 = stablehlo.subtract %v6030, %v6033 : tensor<64x64x56x56xf32>
    %v6039 = stablehlo.multiply %v6038, %v6037 : tensor<64x64x56x56xf32>
    %v6040 = stablehlo.reshape %v5892 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6041 = stablehlo.multiply %v6040, %v6039 : tensor<64x64x56x56xf32>
    %v6042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6043 = stablehlo.reduce(%v6041 init: %v6042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6044 = stablehlo.reshape %v5892 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6045 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6046 = stablehlo.reduce(%v6044 init: %v6045) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6047 = stablehlo.reshape %v289 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6048 = stablehlo.reshape %v5881 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6049 = stablehlo.transpose %v6047, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6050 = stablehlo.transpose %v6048, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6051 = stablehlo.convolution(%v6049, %v6050)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v6052 = stablehlo.transpose %v6051, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6053 = stablehlo.reshape %v294 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6054 = stablehlo.slice %v313 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6055 = stablehlo.slice %v313 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6056 = stablehlo.broadcast_in_dim %v6054, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6057 = stablehlo.broadcast_in_dim %v6055, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6058 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6059 = stablehlo.add %v6057, %v6058 : tensor<64x256x56x56xf32>
    %v6060 = stablehlo.rsqrt %v6059 : tensor<64x256x56x56xf32>
    %v6061 = stablehlo.subtract %v6053, %v6056 : tensor<64x256x56x56xf32>
    %v6062 = stablehlo.multiply %v6061, %v6060 : tensor<64x256x56x56xf32>
    %v6063 = stablehlo.reshape %v5837 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6064 = stablehlo.multiply %v6063, %v6062 : tensor<64x256x56x56xf32>
    %v6065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6066 = stablehlo.reduce(%v6064 init: %v6065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6067 = stablehlo.reshape %v5837 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6068 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6069 = stablehlo.reduce(%v6067 init: %v6068) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6070 = stablehlo.reshape %v6000 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6071 = stablehlo.reshape %v205 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6072 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v6073 = stablehlo.compare GT, %v6071, %v6072 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v6074 = stablehlo.select %v6073, %v6070, %v6072 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v6075 = stablehlo.reshape %v6074 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6076 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6077 = stablehlo.slice %v150 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6078 = stablehlo.slice %v150 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6079 = stablehlo.broadcast_in_dim %v6077, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6080 = stablehlo.broadcast_in_dim %v6078, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6081 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6082 = stablehlo.add %v6080, %v6081 : tensor<64x256x56x56xf32>
    %v6083 = stablehlo.rsqrt %v6082 : tensor<64x256x56x56xf32>
    %v6084 = stablehlo.subtract %v6076, %v6079 : tensor<64x256x56x56xf32>
    %v6085 = stablehlo.multiply %v6084, %v6083 : tensor<64x256x56x56xf32>
    %v6086 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6087 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6088 = stablehlo.multiply %v6086, %v6087 : tensor<64x256x56x56xf32>
    %v6089 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6090 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6091 = stablehlo.reduce(%v6088 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6092 = stablehlo.divide %v6091, %v6090 : tensor<256xf32>
    %v6093 = stablehlo.multiply %v6085, %v6088 : tensor<64x256x56x56xf32>
    %v6094 = stablehlo.reduce(%v6093 init: %v6089) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6095 = stablehlo.divide %v6094, %v6090 : tensor<256xf32>
    %v6096 = stablehlo.concatenate %v6092, %v6095, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6097 = stablehlo.concatenate %v150, %v6096, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b0g3dst = "stablehlo.all_reduce"(%v6097) ({
    ^bb0(%aras1b0g3dst: tensor<f32>, %arbs1b0g3dst: tensor<f32>):
      %aradds1b0g3dst = stablehlo.add %aras1b0g3dst, %arbs1b0g3dst : tensor<f32>
      stablehlo.return %aradds1b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b0g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b0g3dst = stablehlo.divide %arsums1b0g3dst, %arns1b0g3dst : tensor<1024xf32>
    %v6098 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6099 = stablehlo.slice %armeans1b0g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6100 = stablehlo.slice %armeans1b0g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6101 = stablehlo.slice %armeans1b0g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6102 = stablehlo.slice %armeans1b0g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6103 = stablehlo.broadcast_in_dim %v6099, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6104 = stablehlo.broadcast_in_dim %v6100, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6105 = stablehlo.broadcast_in_dim %v6101, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6106 = stablehlo.broadcast_in_dim %v6102, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6107 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6108 = stablehlo.add %v6104, %v6107 : tensor<64x256x56x56xf32>
    %v6109 = stablehlo.rsqrt %v6108 : tensor<64x256x56x56xf32>
    %v6110 = stablehlo.subtract %v6098, %v6103 : tensor<64x256x56x56xf32>
    %v6111 = stablehlo.multiply %v6110, %v6109 : tensor<64x256x56x56xf32>
    %v6112 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6113 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6114 = stablehlo.multiply %v6112, %v6113 : tensor<64x256x56x56xf32>
    %v6115 = stablehlo.subtract %v6114, %v6105 : tensor<64x256x56x56xf32>
    %v6116 = stablehlo.multiply %v6111, %v6106 : tensor<64x256x56x56xf32>
    %v6117 = stablehlo.subtract %v6115, %v6116 : tensor<64x256x56x56xf32>
    %v6118 = stablehlo.multiply %v6109, %v6117 : tensor<64x256x56x56xf32>
    %v6119 = stablehlo.reshape %v6118 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6120 = stablehlo.reshape %v6119 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6121 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6122 = stablehlo.transpose %v6121, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6123 = stablehlo.convolution(%v6120, %v6122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v6124 = stablehlo.reshape %v6123 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6125 = stablehlo.reshape %v6124 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6126 = stablehlo.reshape %v124 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6127 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6128 = stablehlo.compare GT, %v6126, %v6127 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6129 = stablehlo.select %v6128, %v6125, %v6127 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6130 = stablehlo.reshape %v6129 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6131 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6132 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6133 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6134 = stablehlo.broadcast_in_dim %v6132, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6135 = stablehlo.broadcast_in_dim %v6133, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6136 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6137 = stablehlo.add %v6135, %v6136 : tensor<64x64x56x56xf32>
    %v6138 = stablehlo.rsqrt %v6137 : tensor<64x64x56x56xf32>
    %v6139 = stablehlo.subtract %v6131, %v6134 : tensor<64x64x56x56xf32>
    %v6140 = stablehlo.multiply %v6139, %v6138 : tensor<64x64x56x56xf32>
    %v6141 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6142 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6143 = stablehlo.multiply %v6141, %v6142 : tensor<64x64x56x56xf32>
    %v6144 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6145 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6146 = stablehlo.reduce(%v6143 init: %v6144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6147 = stablehlo.divide %v6146, %v6145 : tensor<64xf32>
    %v6148 = stablehlo.multiply %v6140, %v6143 : tensor<64x64x56x56xf32>
    %v6149 = stablehlo.reduce(%v6148 init: %v6144) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6150 = stablehlo.divide %v6149, %v6145 : tensor<64xf32>
    %v6151 = stablehlo.concatenate %v6147, %v6150, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6152 = stablehlo.concatenate %v109, %v6151, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g2dst = "stablehlo.all_reduce"(%v6152) ({
    ^bb0(%aras1b0g2dst: tensor<f32>, %arbs1b0g2dst: tensor<f32>):
      %aradds1b0g2dst = stablehlo.add %aras1b0g2dst, %arbs1b0g2dst : tensor<f32>
      stablehlo.return %aradds1b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g2dst = stablehlo.divide %arsums1b0g2dst, %arns1b0g2dst : tensor<256xf32>
    %v6153 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6154 = stablehlo.slice %armeans1b0g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6155 = stablehlo.slice %armeans1b0g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6156 = stablehlo.slice %armeans1b0g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6157 = stablehlo.slice %armeans1b0g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6158 = stablehlo.broadcast_in_dim %v6154, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6159 = stablehlo.broadcast_in_dim %v6155, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6160 = stablehlo.broadcast_in_dim %v6156, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6161 = stablehlo.broadcast_in_dim %v6157, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6162 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6163 = stablehlo.add %v6159, %v6162 : tensor<64x64x56x56xf32>
    %v6164 = stablehlo.rsqrt %v6163 : tensor<64x64x56x56xf32>
    %v6165 = stablehlo.subtract %v6153, %v6158 : tensor<64x64x56x56xf32>
    %v6166 = stablehlo.multiply %v6165, %v6164 : tensor<64x64x56x56xf32>
    %v6167 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6168 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6169 = stablehlo.multiply %v6167, %v6168 : tensor<64x64x56x56xf32>
    %v6170 = stablehlo.subtract %v6169, %v6160 : tensor<64x64x56x56xf32>
    %v6171 = stablehlo.multiply %v6166, %v6161 : tensor<64x64x56x56xf32>
    %v6172 = stablehlo.subtract %v6170, %v6171 : tensor<64x64x56x56xf32>
    %v6173 = stablehlo.multiply %v6164, %v6172 : tensor<64x64x56x56xf32>
    %v6174 = stablehlo.reshape %v6173 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6175 = stablehlo.reshape %v6174 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6176 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v6177 = stablehlo.transpose %v6176, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6178 = stablehlo.convolution(%v6175, %v6177)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x3x3xf32>) -> tensor<64x64x56x56xf32>
    %v6179 = stablehlo.reshape %v6178 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6180 = stablehlo.reshape %v6179 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6181 = stablehlo.reshape %v83 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6182 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6183 = stablehlo.compare GT, %v6181, %v6182 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6184 = stablehlo.select %v6183, %v6180, %v6182 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6185 = stablehlo.reshape %v6184 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6186 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6187 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6188 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6189 = stablehlo.broadcast_in_dim %v6187, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6190 = stablehlo.broadcast_in_dim %v6188, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6191 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6192 = stablehlo.add %v6190, %v6191 : tensor<64x64x56x56xf32>
    %v6193 = stablehlo.rsqrt %v6192 : tensor<64x64x56x56xf32>
    %v6194 = stablehlo.subtract %v6186, %v6189 : tensor<64x64x56x56xf32>
    %v6195 = stablehlo.multiply %v6194, %v6193 : tensor<64x64x56x56xf32>
    %v6196 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6197 = stablehlo.reshape %v6185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6198 = stablehlo.multiply %v6196, %v6197 : tensor<64x64x56x56xf32>
    %v6199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6200 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6201 = stablehlo.reduce(%v6198 init: %v6199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6202 = stablehlo.divide %v6201, %v6200 : tensor<64xf32>
    %v6203 = stablehlo.multiply %v6195, %v6198 : tensor<64x64x56x56xf32>
    %v6204 = stablehlo.reduce(%v6203 init: %v6199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6205 = stablehlo.divide %v6204, %v6200 : tensor<64xf32>
    %v6206 = stablehlo.concatenate %v6202, %v6205, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6207 = stablehlo.concatenate %v68, %v6206, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g1dst = "stablehlo.all_reduce"(%v6207) ({
    ^bb0(%aras1b0g1dst: tensor<f32>, %arbs1b0g1dst: tensor<f32>):
      %aradds1b0g1dst = stablehlo.add %aras1b0g1dst, %arbs1b0g1dst : tensor<f32>
      stablehlo.return %aradds1b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g1dst = stablehlo.divide %arsums1b0g1dst, %arns1b0g1dst : tensor<256xf32>
    %v6208 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6209 = stablehlo.slice %armeans1b0g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6210 = stablehlo.slice %armeans1b0g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6211 = stablehlo.slice %armeans1b0g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6212 = stablehlo.slice %armeans1b0g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6213 = stablehlo.broadcast_in_dim %v6209, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6214 = stablehlo.broadcast_in_dim %v6210, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6215 = stablehlo.broadcast_in_dim %v6211, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6216 = stablehlo.broadcast_in_dim %v6212, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6217 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6218 = stablehlo.add %v6214, %v6217 : tensor<64x64x56x56xf32>
    %v6219 = stablehlo.rsqrt %v6218 : tensor<64x64x56x56xf32>
    %v6220 = stablehlo.subtract %v6208, %v6213 : tensor<64x64x56x56xf32>
    %v6221 = stablehlo.multiply %v6220, %v6219 : tensor<64x64x56x56xf32>
    %v6222 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6223 = stablehlo.reshape %v6185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6224 = stablehlo.multiply %v6222, %v6223 : tensor<64x64x56x56xf32>
    %v6225 = stablehlo.subtract %v6224, %v6215 : tensor<64x64x56x56xf32>
    %v6226 = stablehlo.multiply %v6221, %v6216 : tensor<64x64x56x56xf32>
    %v6227 = stablehlo.subtract %v6225, %v6226 : tensor<64x64x56x56xf32>
    %v6228 = stablehlo.multiply %v6219, %v6227 : tensor<64x64x56x56xf32>
    %v6229 = stablehlo.reshape %v6228 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6230 = stablehlo.reshape %v6229 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6231 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v6232 = stablehlo.transpose %v6231, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v6233 = stablehlo.convolution(%v6230, %v6232)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v6234 = stablehlo.reshape %v6233 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6235 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6236 = stablehlo.slice %v189 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6237 = stablehlo.slice %v189 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6238 = stablehlo.broadcast_in_dim %v6236, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6239 = stablehlo.broadcast_in_dim %v6237, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6240 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6241 = stablehlo.add %v6239, %v6240 : tensor<64x256x56x56xf32>
    %v6242 = stablehlo.rsqrt %v6241 : tensor<64x256x56x56xf32>
    %v6243 = stablehlo.subtract %v6235, %v6238 : tensor<64x256x56x56xf32>
    %v6244 = stablehlo.multiply %v6243, %v6242 : tensor<64x256x56x56xf32>
    %v6245 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6246 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6247 = stablehlo.multiply %v6245, %v6246 : tensor<64x256x56x56xf32>
    %v6248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6249 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6250 = stablehlo.reduce(%v6247 init: %v6248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6251 = stablehlo.divide %v6250, %v6249 : tensor<256xf32>
    %v6252 = stablehlo.multiply %v6244, %v6247 : tensor<64x256x56x56xf32>
    %v6253 = stablehlo.reduce(%v6252 init: %v6248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6254 = stablehlo.divide %v6253, %v6249 : tensor<256xf32>
    %v6255 = stablehlo.concatenate %v6251, %v6254, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6256 = stablehlo.concatenate %v189, %v6255, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b0gpdst = "stablehlo.all_reduce"(%v6256) ({
    ^bb0(%aras1b0gpdst: tensor<f32>, %arbs1b0gpdst: tensor<f32>):
      %aradds1b0gpdst = stablehlo.add %aras1b0gpdst, %arbs1b0gpdst : tensor<f32>
      stablehlo.return %aradds1b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b0gpdst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b0gpdst = stablehlo.divide %arsums1b0gpdst, %arns1b0gpdst : tensor<1024xf32>
    %v6257 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6258 = stablehlo.slice %armeans1b0gpdst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6259 = stablehlo.slice %armeans1b0gpdst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6260 = stablehlo.slice %armeans1b0gpdst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6261 = stablehlo.slice %armeans1b0gpdst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6262 = stablehlo.broadcast_in_dim %v6258, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6263 = stablehlo.broadcast_in_dim %v6259, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6264 = stablehlo.broadcast_in_dim %v6260, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6265 = stablehlo.broadcast_in_dim %v6261, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6266 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6267 = stablehlo.add %v6263, %v6266 : tensor<64x256x56x56xf32>
    %v6268 = stablehlo.rsqrt %v6267 : tensor<64x256x56x56xf32>
    %v6269 = stablehlo.subtract %v6257, %v6262 : tensor<64x256x56x56xf32>
    %v6270 = stablehlo.multiply %v6269, %v6268 : tensor<64x256x56x56xf32>
    %v6271 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6272 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6273 = stablehlo.multiply %v6271, %v6272 : tensor<64x256x56x56xf32>
    %v6274 = stablehlo.subtract %v6273, %v6264 : tensor<64x256x56x56xf32>
    %v6275 = stablehlo.multiply %v6270, %v6265 : tensor<64x256x56x56xf32>
    %v6276 = stablehlo.subtract %v6274, %v6275 : tensor<64x256x56x56xf32>
    %v6277 = stablehlo.multiply %v6268, %v6276 : tensor<64x256x56x56xf32>
    %v6278 = stablehlo.reshape %v6277 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6279 = stablehlo.reshape %v6278 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6280 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6281 = stablehlo.transpose %v6280, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6282 = stablehlo.convolution(%v6279, %v6281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xf32>, tensor<64x256x1x1xf32>) -> tensor<64x64x56x56xf32>
    %v6283 = stablehlo.reshape %v6282 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6284 = stablehlo.reshape %v6234 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6285 = stablehlo.reshape %v6283 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6286 = stablehlo.add %v6284, %v6285 : tensor<64x64x56x56xf32>
    %v6287 = stablehlo.reshape %v6286 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6288 = stablehlo.reshape %v44 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6289 = stablehlo.reshape %v6229 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6290 = stablehlo.transpose %v6288, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6291 = stablehlo.transpose %v6289, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6292 = stablehlo.convolution(%v6290, %v6291)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x1x1xf32>
    %v6293 = stablehlo.transpose %v6292, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v6294 = stablehlo.reshape %v49 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6295 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6296 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6297 = stablehlo.broadcast_in_dim %v6295, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6298 = stablehlo.broadcast_in_dim %v6296, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6299 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6300 = stablehlo.add %v6298, %v6299 : tensor<64x64x56x56xf32>
    %v6301 = stablehlo.rsqrt %v6300 : tensor<64x64x56x56xf32>
    %v6302 = stablehlo.subtract %v6294, %v6297 : tensor<64x64x56x56xf32>
    %v6303 = stablehlo.multiply %v6302, %v6301 : tensor<64x64x56x56xf32>
    %v6304 = stablehlo.reshape %v6185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6305 = stablehlo.multiply %v6304, %v6303 : tensor<64x64x56x56xf32>
    %v6306 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6307 = stablehlo.reduce(%v6305 init: %v6306) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6308 = stablehlo.reshape %v6185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6309 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6310 = stablehlo.reduce(%v6308 init: %v6309) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6311 = stablehlo.reshape %v85 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6312 = stablehlo.reshape %v6174 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6313 = stablehlo.transpose %v6311, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6314 = stablehlo.transpose %v6312, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6315 = stablehlo.convolution(%v6313, %v6314)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x3x3xf32>
    %v6316 = stablehlo.transpose %v6315, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6317 = stablehlo.reshape %v90 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6318 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6319 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6320 = stablehlo.broadcast_in_dim %v6318, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6321 = stablehlo.broadcast_in_dim %v6319, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6322 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6323 = stablehlo.add %v6321, %v6322 : tensor<64x64x56x56xf32>
    %v6324 = stablehlo.rsqrt %v6323 : tensor<64x64x56x56xf32>
    %v6325 = stablehlo.subtract %v6317, %v6320 : tensor<64x64x56x56xf32>
    %v6326 = stablehlo.multiply %v6325, %v6324 : tensor<64x64x56x56xf32>
    %v6327 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6328 = stablehlo.multiply %v6327, %v6326 : tensor<64x64x56x56xf32>
    %v6329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6330 = stablehlo.reduce(%v6328 init: %v6329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6331 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6333 = stablehlo.reduce(%v6331 init: %v6332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6334 = stablehlo.reshape %v126 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6335 = stablehlo.reshape %v6119 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6336 = stablehlo.transpose %v6334, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6337 = stablehlo.transpose %v6335, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6338 = stablehlo.convolution(%v6336, %v6337)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v6339 = stablehlo.transpose %v6338, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6340 = stablehlo.reshape %v131 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6341 = stablehlo.slice %v150 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6342 = stablehlo.slice %v150 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6343 = stablehlo.broadcast_in_dim %v6341, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6344 = stablehlo.broadcast_in_dim %v6342, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6345 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6346 = stablehlo.add %v6344, %v6345 : tensor<64x256x56x56xf32>
    %v6347 = stablehlo.rsqrt %v6346 : tensor<64x256x56x56xf32>
    %v6348 = stablehlo.subtract %v6340, %v6343 : tensor<64x256x56x56xf32>
    %v6349 = stablehlo.multiply %v6348, %v6347 : tensor<64x256x56x56xf32>
    %v6350 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6351 = stablehlo.multiply %v6350, %v6349 : tensor<64x256x56x56xf32>
    %v6352 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6353 = stablehlo.reduce(%v6351 init: %v6352) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6354 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6356 = stablehlo.reduce(%v6354 init: %v6355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6357 = stablehlo.reshape %v44 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6358 = stablehlo.reshape %v6278 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6359 = stablehlo.transpose %v6357, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6360 = stablehlo.transpose %v6358, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6361 = stablehlo.convolution(%v6359, %v6360)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xf32>, tensor<256x64x56x56xf32>) -> tensor<64x256x1x1xf32>
    %v6362 = stablehlo.transpose %v6361, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6363 = stablehlo.reshape %v170 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6364 = stablehlo.slice %v189 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6365 = stablehlo.slice %v189 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6366 = stablehlo.broadcast_in_dim %v6364, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6367 = stablehlo.broadcast_in_dim %v6365, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6368 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6369 = stablehlo.add %v6367, %v6368 : tensor<64x256x56x56xf32>
    %v6370 = stablehlo.rsqrt %v6369 : tensor<64x256x56x56xf32>
    %v6371 = stablehlo.subtract %v6363, %v6366 : tensor<64x256x56x56xf32>
    %v6372 = stablehlo.multiply %v6371, %v6370 : tensor<64x256x56x56xf32>
    %v6373 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6374 = stablehlo.multiply %v6373, %v6372 : tensor<64x256x56x56xf32>
    %v6375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6376 = stablehlo.reduce(%v6374 init: %v6375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6377 = stablehlo.reshape %v6075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6378 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6379 = stablehlo.reduce(%v6377 init: %v6378) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6380 = stablehlo.reshape %v40 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6381 = stablehlo.reshape %v6287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6382 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v6383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6384 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6385 = stablehlo.pad %v6380, %v6382, low = [0, 0, 1, 1], high = [0, 0, 1, 1], interior = [0, 0, 0, 0] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6386 = stablehlo.slice %v6385 [0:64, 0:64, 0:112:2, 0:112:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6387 = stablehlo.slice %v6385 [0:64, 0:64, 0:112:2, 1:113:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6388 = stablehlo.slice %v6385 [0:64, 0:64, 0:112:2, 2:114:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6389 = stablehlo.slice %v6385 [0:64, 0:64, 1:113:2, 0:112:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6390 = stablehlo.slice %v6385 [0:64, 0:64, 1:113:2, 1:113:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6391 = stablehlo.slice %v6385 [0:64, 0:64, 1:113:2, 2:114:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6392 = stablehlo.slice %v6385 [0:64, 0:64, 2:114:2, 0:112:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6393 = stablehlo.slice %v6385 [0:64, 0:64, 2:114:2, 1:113:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6394 = stablehlo.slice %v6385 [0:64, 0:64, 2:114:2, 2:114:2] : (tensor<64x64x114x114xf32>) -> tensor<64x64x56x56xf32>
    %v6395 = stablehlo.maximum %v6386, %v6387 : tensor<64x64x56x56xf32>
    %v6396 = stablehlo.maximum %v6395, %v6388 : tensor<64x64x56x56xf32>
    %v6397 = stablehlo.maximum %v6396, %v6389 : tensor<64x64x56x56xf32>
    %v6398 = stablehlo.maximum %v6397, %v6390 : tensor<64x64x56x56xf32>
    %v6399 = stablehlo.maximum %v6398, %v6391 : tensor<64x64x56x56xf32>
    %v6400 = stablehlo.maximum %v6399, %v6392 : tensor<64x64x56x56xf32>
    %v6401 = stablehlo.maximum %v6400, %v6393 : tensor<64x64x56x56xf32>
    %v6402 = stablehlo.maximum %v6401, %v6394 : tensor<64x64x56x56xf32>
    %v6403 = stablehlo.compare EQ, %v6386, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6404 = stablehlo.select %v6403, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6405 = stablehlo.pad %v6404, %v6383, low = [0, 0, 0, 0], high = [0, 0, 3, 3], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6406 = stablehlo.compare EQ, %v6387, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6407 = stablehlo.not %v6403 : tensor<64x64x56x56xi1>
    %v6408 = stablehlo.and %v6406, %v6407 : tensor<64x64x56x56xi1>
    %v6409 = stablehlo.or %v6403, %v6406 : tensor<64x64x56x56xi1>
    %v6410 = stablehlo.select %v6408, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6411 = stablehlo.pad %v6410, %v6383, low = [0, 0, 0, 1], high = [0, 0, 3, 2], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6412 = stablehlo.add %v6405, %v6411 : tensor<64x64x114x114xf32>
    %v6413 = stablehlo.compare EQ, %v6388, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6414 = stablehlo.not %v6409 : tensor<64x64x56x56xi1>
    %v6415 = stablehlo.and %v6413, %v6414 : tensor<64x64x56x56xi1>
    %v6416 = stablehlo.or %v6409, %v6413 : tensor<64x64x56x56xi1>
    %v6417 = stablehlo.select %v6415, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6418 = stablehlo.pad %v6417, %v6383, low = [0, 0, 0, 2], high = [0, 0, 3, 1], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6419 = stablehlo.add %v6412, %v6418 : tensor<64x64x114x114xf32>
    %v6420 = stablehlo.compare EQ, %v6389, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6421 = stablehlo.not %v6416 : tensor<64x64x56x56xi1>
    %v6422 = stablehlo.and %v6420, %v6421 : tensor<64x64x56x56xi1>
    %v6423 = stablehlo.or %v6416, %v6420 : tensor<64x64x56x56xi1>
    %v6424 = stablehlo.select %v6422, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6425 = stablehlo.pad %v6424, %v6383, low = [0, 0, 1, 0], high = [0, 0, 2, 3], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6426 = stablehlo.add %v6419, %v6425 : tensor<64x64x114x114xf32>
    %v6427 = stablehlo.compare EQ, %v6390, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6428 = stablehlo.not %v6423 : tensor<64x64x56x56xi1>
    %v6429 = stablehlo.and %v6427, %v6428 : tensor<64x64x56x56xi1>
    %v6430 = stablehlo.or %v6423, %v6427 : tensor<64x64x56x56xi1>
    %v6431 = stablehlo.select %v6429, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6432 = stablehlo.pad %v6431, %v6383, low = [0, 0, 1, 1], high = [0, 0, 2, 2], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6433 = stablehlo.add %v6426, %v6432 : tensor<64x64x114x114xf32>
    %v6434 = stablehlo.compare EQ, %v6391, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6435 = stablehlo.not %v6430 : tensor<64x64x56x56xi1>
    %v6436 = stablehlo.and %v6434, %v6435 : tensor<64x64x56x56xi1>
    %v6437 = stablehlo.or %v6430, %v6434 : tensor<64x64x56x56xi1>
    %v6438 = stablehlo.select %v6436, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6439 = stablehlo.pad %v6438, %v6383, low = [0, 0, 1, 2], high = [0, 0, 2, 1], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6440 = stablehlo.add %v6433, %v6439 : tensor<64x64x114x114xf32>
    %v6441 = stablehlo.compare EQ, %v6392, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6442 = stablehlo.not %v6437 : tensor<64x64x56x56xi1>
    %v6443 = stablehlo.and %v6441, %v6442 : tensor<64x64x56x56xi1>
    %v6444 = stablehlo.or %v6437, %v6441 : tensor<64x64x56x56xi1>
    %v6445 = stablehlo.select %v6443, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6446 = stablehlo.pad %v6445, %v6383, low = [0, 0, 2, 0], high = [0, 0, 1, 3], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6447 = stablehlo.add %v6440, %v6446 : tensor<64x64x114x114xf32>
    %v6448 = stablehlo.compare EQ, %v6393, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6449 = stablehlo.not %v6444 : tensor<64x64x56x56xi1>
    %v6450 = stablehlo.and %v6448, %v6449 : tensor<64x64x56x56xi1>
    %v6451 = stablehlo.or %v6444, %v6448 : tensor<64x64x56x56xi1>
    %v6452 = stablehlo.select %v6450, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6453 = stablehlo.pad %v6452, %v6383, low = [0, 0, 2, 1], high = [0, 0, 1, 2], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6454 = stablehlo.add %v6447, %v6453 : tensor<64x64x114x114xf32>
    %v6455 = stablehlo.compare EQ, %v6394, %v6402 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6456 = stablehlo.not %v6451 : tensor<64x64x56x56xi1>
    %v6457 = stablehlo.and %v6455, %v6456 : tensor<64x64x56x56xi1>
    %v6458 = stablehlo.or %v6451, %v6455 : tensor<64x64x56x56xi1>
    %v6459 = stablehlo.select %v6457, %v6381, %v6384 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6460 = stablehlo.pad %v6459, %v6383, low = [0, 0, 2, 2], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x114x114xf32>
    %v6461 = stablehlo.add %v6454, %v6460 : tensor<64x64x114x114xf32>
    %v6462 = stablehlo.slice %v6461 [0:64, 0:64, 1:113, 1:113] : (tensor<64x64x114x114xf32>) -> tensor<64x64x112x112xf32>
    %v6463 = stablehlo.reshape %v6462 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6464 = stablehlo.reshape %v6463 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6465 = stablehlo.reshape %v38 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6466 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v6467 = stablehlo.compare GT, %v6465, %v6466 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v6468 = stablehlo.select %v6467, %v6464, %v6466 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v6469 = stablehlo.reshape %v6468 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6470 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6471 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6472 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6473 = stablehlo.broadcast_in_dim %v6471, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6474 = stablehlo.broadcast_in_dim %v6472, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6475 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6476 = stablehlo.add %v6474, %v6475 : tensor<64x64x112x112xf32>
    %v6477 = stablehlo.rsqrt %v6476 : tensor<64x64x112x112xf32>
    %v6478 = stablehlo.subtract %v6470, %v6473 : tensor<64x64x112x112xf32>
    %v6479 = stablehlo.multiply %v6478, %v6477 : tensor<64x64x112x112xf32>
    %v6480 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6481 = stablehlo.reshape %v6469 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6482 = stablehlo.multiply %v6480, %v6481 : tensor<64x64x112x112xf32>
    %v6483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6484 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v6485 = stablehlo.reduce(%v6482 init: %v6483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6486 = stablehlo.divide %v6485, %v6484 : tensor<64xf32>
    %v6487 = stablehlo.multiply %v6479, %v6482 : tensor<64x64x112x112xf32>
    %v6488 = stablehlo.reduce(%v6487 init: %v6483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6489 = stablehlo.divide %v6488, %v6484 : tensor<64xf32>
    %v6490 = stablehlo.concatenate %v6486, %v6489, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6491 = stablehlo.concatenate %v23, %v6490, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsumsgdst = "stablehlo.all_reduce"(%v6491) ({
    ^bb0(%arasgdst: tensor<f32>, %arbsgdst: tensor<f32>):
      %araddsgdst = stablehlo.add %arasgdst, %arbsgdst : tensor<f32>
      stablehlo.return %araddsgdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnsgdst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeansgdst = stablehlo.divide %arsumsgdst, %arnsgdst : tensor<256xf32>
    %v6492 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6493 = stablehlo.slice %armeansgdst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6494 = stablehlo.slice %armeansgdst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6495 = stablehlo.slice %armeansgdst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6496 = stablehlo.slice %armeansgdst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6497 = stablehlo.broadcast_in_dim %v6493, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6498 = stablehlo.broadcast_in_dim %v6494, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6499 = stablehlo.broadcast_in_dim %v6495, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6500 = stablehlo.broadcast_in_dim %v6496, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6501 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6502 = stablehlo.add %v6498, %v6501 : tensor<64x64x112x112xf32>
    %v6503 = stablehlo.rsqrt %v6502 : tensor<64x64x112x112xf32>
    %v6504 = stablehlo.subtract %v6492, %v6497 : tensor<64x64x112x112xf32>
    %v6505 = stablehlo.multiply %v6504, %v6503 : tensor<64x64x112x112xf32>
    %v6506 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6507 = stablehlo.reshape %v6469 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6508 = stablehlo.multiply %v6506, %v6507 : tensor<64x64x112x112xf32>
    %v6509 = stablehlo.subtract %v6508, %v6499 : tensor<64x64x112x112xf32>
    %v6510 = stablehlo.multiply %v6505, %v6500 : tensor<64x64x112x112xf32>
    %v6511 = stablehlo.subtract %v6509, %v6510 : tensor<64x64x112x112xf32>
    %v6512 = stablehlo.multiply %v6503, %v6511 : tensor<64x64x112x112xf32>
    %v6513 = stablehlo.reshape %v6512 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6514 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v6515 = stablehlo.reshape %v6513 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6516 = stablehlo.transpose %v6514, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v6518 = stablehlo.transpose %v6515, dims = [1, 0, 2, 3] : (tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xf32>
    %v6519 = stablehlo.convolution(%v6516, %v6518)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 2], [3, 2]], lhs_dilate = [1, 1], rhs_dilate = [2, 2]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xf32>, tensor<64x64x112x112xf32>) -> tensor<3x64x7x7xf32>
    %v6520 = stablehlo.transpose %v6519, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v6521 = stablehlo.reshape %v4 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6522 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6523 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6524 = stablehlo.broadcast_in_dim %v6522, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6525 = stablehlo.broadcast_in_dim %v6523, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6526 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6527 = stablehlo.add %v6525, %v6526 : tensor<64x64x112x112xf32>
    %v6528 = stablehlo.rsqrt %v6527 : tensor<64x64x112x112xf32>
    %v6529 = stablehlo.subtract %v6521, %v6524 : tensor<64x64x112x112xf32>
    %v6530 = stablehlo.multiply %v6529, %v6528 : tensor<64x64x112x112xf32>
    %v6531 = stablehlo.reshape %v6469 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6532 = stablehlo.multiply %v6531, %v6530 : tensor<64x64x112x112xf32>
    %v6533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6534 = stablehlo.reduce(%v6532 init: %v6533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6535 = stablehlo.reshape %v6469 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6537 = stablehlo.reduce(%v6535 init: %v6536) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6538 = stablehlo.slice %v23 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6539 = stablehlo.slice %v23 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6540 = stablehlo.slice %v68 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6541 = stablehlo.slice %v68 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6542 = stablehlo.slice %v109 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6543 = stablehlo.slice %v109 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6544 = stablehlo.slice %v150 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6545 = stablehlo.slice %v150 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6546 = stablehlo.slice %v189 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6547 = stablehlo.slice %v189 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6548 = stablehlo.slice %v231 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6549 = stablehlo.slice %v231 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6550 = stablehlo.slice %v272 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6551 = stablehlo.slice %v272 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6552 = stablehlo.slice %v313 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6553 = stablehlo.slice %v313 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6554 = stablehlo.slice %v360 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6555 = stablehlo.slice %v360 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6556 = stablehlo.slice %v401 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6557 = stablehlo.slice %v401 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6558 = stablehlo.slice %v442 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6559 = stablehlo.slice %v442 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6560 = stablehlo.slice %v489 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6561 = stablehlo.slice %v489 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6562 = stablehlo.slice %v530 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6563 = stablehlo.slice %v530 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6564 = stablehlo.slice %v571 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6565 = stablehlo.slice %v571 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6566 = stablehlo.slice %v610 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6567 = stablehlo.slice %v610 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6568 = stablehlo.slice %v652 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6569 = stablehlo.slice %v652 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6570 = stablehlo.slice %v693 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6571 = stablehlo.slice %v693 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6572 = stablehlo.slice %v734 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6573 = stablehlo.slice %v734 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6574 = stablehlo.slice %v781 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6575 = stablehlo.slice %v781 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6576 = stablehlo.slice %v822 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6577 = stablehlo.slice %v822 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6578 = stablehlo.slice %v863 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6579 = stablehlo.slice %v863 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6580 = stablehlo.slice %v910 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6581 = stablehlo.slice %v910 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6582 = stablehlo.slice %v951 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6583 = stablehlo.slice %v951 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6584 = stablehlo.slice %v992 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6585 = stablehlo.slice %v992 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6586 = stablehlo.slice %v1039 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6587 = stablehlo.slice %v1039 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6588 = stablehlo.slice %v1080 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6589 = stablehlo.slice %v1080 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6590 = stablehlo.slice %v1121 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6591 = stablehlo.slice %v1121 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6592 = stablehlo.slice %v1160 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6593 = stablehlo.slice %v1160 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6594 = stablehlo.slice %v1202 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6595 = stablehlo.slice %v1202 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6596 = stablehlo.slice %v1243 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6597 = stablehlo.slice %v1243 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6598 = stablehlo.slice %v1284 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6599 = stablehlo.slice %v1284 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6600 = stablehlo.slice %v1331 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6601 = stablehlo.slice %v1331 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6602 = stablehlo.slice %v1372 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6603 = stablehlo.slice %v1372 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6604 = stablehlo.slice %v1413 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6605 = stablehlo.slice %v1413 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6606 = stablehlo.slice %v1460 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6607 = stablehlo.slice %v1460 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6608 = stablehlo.slice %v1501 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6609 = stablehlo.slice %v1501 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6610 = stablehlo.slice %v1542 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6611 = stablehlo.slice %v1542 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6612 = stablehlo.slice %v1589 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6613 = stablehlo.slice %v1589 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6614 = stablehlo.slice %v1630 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6615 = stablehlo.slice %v1630 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6616 = stablehlo.slice %v1671 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6617 = stablehlo.slice %v1671 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6618 = stablehlo.slice %v1718 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6619 = stablehlo.slice %v1718 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6620 = stablehlo.slice %v1759 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6621 = stablehlo.slice %v1759 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6622 = stablehlo.slice %v1800 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6623 = stablehlo.slice %v1800 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v6624 = stablehlo.slice %v1847 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6625 = stablehlo.slice %v1847 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6626 = stablehlo.slice %v1888 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6627 = stablehlo.slice %v1888 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6628 = stablehlo.slice %v1929 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6629 = stablehlo.slice %v1929 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6630 = stablehlo.slice %v1968 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6631 = stablehlo.slice %v1968 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6632 = stablehlo.slice %v2010 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6633 = stablehlo.slice %v2010 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6634 = stablehlo.slice %v2051 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6635 = stablehlo.slice %v2051 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6636 = stablehlo.slice %v2092 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6637 = stablehlo.slice %v2092 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6638 = stablehlo.slice %v2139 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6639 = stablehlo.slice %v2139 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6640 = stablehlo.slice %v2180 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6641 = stablehlo.slice %v2180 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6642 = stablehlo.slice %v2221 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v6643 = stablehlo.slice %v2221 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v6520) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v6644 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6645 = stablehlo.multiply %v6644, %sW : tensor<64x3x7x7xf32>
    %v6646 = stablehlo.add %v6645, %armeansW : tensor<64x3x7x7xf32>
    %v6647 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6648 = stablehlo.multiply %v6647, %sWv : tensor<64x3x7x7xf32>
    %v6649 = stablehlo.add %v6648, %v6646 : tensor<64x3x7x7xf32>
    %v6650 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6651 = stablehlo.multiply %v6650, %v6649 : tensor<64x3x7x7xf32>
    %v6652 = stablehlo.subtract %sW, %v6651 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v6534) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v6653 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6654 = stablehlo.multiply %v6653, %sg : tensor<64xf32>
    %v6655 = stablehlo.add %v6654, %armeansg : tensor<64xf32>
    %v6656 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6657 = stablehlo.multiply %v6656, %sgv : tensor<64xf32>
    %v6658 = stablehlo.add %v6657, %v6655 : tensor<64xf32>
    %v6659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6660 = stablehlo.multiply %v6659, %v6658 : tensor<64xf32>
    %v6661 = stablehlo.subtract %sg, %v6660 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v6537) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v6662 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6663 = stablehlo.multiply %v6662, %sbt : tensor<64xf32>
    %v6664 = stablehlo.add %v6663, %armeansbt : tensor<64xf32>
    %v6665 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6666 = stablehlo.multiply %v6665, %sbtv : tensor<64xf32>
    %v6667 = stablehlo.add %v6666, %v6664 : tensor<64xf32>
    %v6668 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6669 = stablehlo.multiply %v6668, %v6667 : tensor<64xf32>
    %v6670 = stablehlo.subtract %sbt, %v6669 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v6293) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x1x1xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x1x1xf32>
    %v6671 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6672 = stablehlo.multiply %v6671, %s1b0W1 : tensor<64x64x1x1xf32>
    %v6673 = stablehlo.add %v6672, %armeans1b0W1 : tensor<64x64x1x1xf32>
    %v6674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6675 = stablehlo.multiply %v6674, %s1b0W1v : tensor<64x64x1x1xf32>
    %v6676 = stablehlo.add %v6675, %v6673 : tensor<64x64x1x1xf32>
    %v6677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6678 = stablehlo.multiply %v6677, %v6676 : tensor<64x64x1x1xf32>
    %v6679 = stablehlo.subtract %s1b0W1, %v6678 : tensor<64x64x1x1xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v6307) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v6680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6681 = stablehlo.multiply %v6680, %s1b0g1 : tensor<64xf32>
    %v6682 = stablehlo.add %v6681, %armeans1b0g1 : tensor<64xf32>
    %v6683 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6684 = stablehlo.multiply %v6683, %s1b0g1v : tensor<64xf32>
    %v6685 = stablehlo.add %v6684, %v6682 : tensor<64xf32>
    %v6686 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6687 = stablehlo.multiply %v6686, %v6685 : tensor<64xf32>
    %v6688 = stablehlo.subtract %s1b0g1, %v6687 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v6310) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v6689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6690 = stablehlo.multiply %v6689, %s1b0bt1 : tensor<64xf32>
    %v6691 = stablehlo.add %v6690, %armeans1b0bt1 : tensor<64xf32>
    %v6692 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6693 = stablehlo.multiply %v6692, %s1b0bt1v : tensor<64xf32>
    %v6694 = stablehlo.add %v6693, %v6691 : tensor<64xf32>
    %v6695 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6696 = stablehlo.multiply %v6695, %v6694 : tensor<64xf32>
    %v6697 = stablehlo.subtract %s1b0bt1, %v6696 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v6316) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v6698 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6699 = stablehlo.multiply %v6698, %s1b0W2 : tensor<64x64x3x3xf32>
    %v6700 = stablehlo.add %v6699, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v6701 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6702 = stablehlo.multiply %v6701, %s1b0W2v : tensor<64x64x3x3xf32>
    %v6703 = stablehlo.add %v6702, %v6700 : tensor<64x64x3x3xf32>
    %v6704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6705 = stablehlo.multiply %v6704, %v6703 : tensor<64x64x3x3xf32>
    %v6706 = stablehlo.subtract %s1b0W2, %v6705 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v6330) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v6707 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6708 = stablehlo.multiply %v6707, %s1b0g2 : tensor<64xf32>
    %v6709 = stablehlo.add %v6708, %armeans1b0g2 : tensor<64xf32>
    %v6710 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6711 = stablehlo.multiply %v6710, %s1b0g2v : tensor<64xf32>
    %v6712 = stablehlo.add %v6711, %v6709 : tensor<64xf32>
    %v6713 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6714 = stablehlo.multiply %v6713, %v6712 : tensor<64xf32>
    %v6715 = stablehlo.subtract %s1b0g2, %v6714 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v6333) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v6716 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6717 = stablehlo.multiply %v6716, %s1b0bt2 : tensor<64xf32>
    %v6718 = stablehlo.add %v6717, %armeans1b0bt2 : tensor<64xf32>
    %v6719 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6720 = stablehlo.multiply %v6719, %s1b0bt2v : tensor<64xf32>
    %v6721 = stablehlo.add %v6720, %v6718 : tensor<64xf32>
    %v6722 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6723 = stablehlo.multiply %v6722, %v6721 : tensor<64xf32>
    %v6724 = stablehlo.subtract %s1b0bt2, %v6723 : tensor<64xf32>
    %arsums1b0W3 = "stablehlo.all_reduce"(%v6339) ({
    ^bb0(%aras1b0W3: tensor<f32>, %arbs1b0W3: tensor<f32>):
      %aradds1b0W3 = stablehlo.add %aras1b0W3, %arbs1b0W3 : tensor<f32>
      stablehlo.return %aradds1b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0W3 = stablehlo.divide %arsums1b0W3, %arns1b0W3 : tensor<256x64x1x1xf32>
    %v6725 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6726 = stablehlo.multiply %v6725, %s1b0W3 : tensor<256x64x1x1xf32>
    %v6727 = stablehlo.add %v6726, %armeans1b0W3 : tensor<256x64x1x1xf32>
    %v6728 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6729 = stablehlo.multiply %v6728, %s1b0W3v : tensor<256x64x1x1xf32>
    %v6730 = stablehlo.add %v6729, %v6727 : tensor<256x64x1x1xf32>
    %v6731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6732 = stablehlo.multiply %v6731, %v6730 : tensor<256x64x1x1xf32>
    %v6733 = stablehlo.subtract %s1b0W3, %v6732 : tensor<256x64x1x1xf32>
    %arsums1b0g3 = "stablehlo.all_reduce"(%v6353) ({
    ^bb0(%aras1b0g3: tensor<f32>, %arbs1b0g3: tensor<f32>):
      %aradds1b0g3 = stablehlo.add %aras1b0g3, %arbs1b0g3 : tensor<f32>
      stablehlo.return %aradds1b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3 = stablehlo.divide %arsums1b0g3, %arns1b0g3 : tensor<256xf32>
    %v6734 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6735 = stablehlo.multiply %v6734, %s1b0g3 : tensor<256xf32>
    %v6736 = stablehlo.add %v6735, %armeans1b0g3 : tensor<256xf32>
    %v6737 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6738 = stablehlo.multiply %v6737, %s1b0g3v : tensor<256xf32>
    %v6739 = stablehlo.add %v6738, %v6736 : tensor<256xf32>
    %v6740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6741 = stablehlo.multiply %v6740, %v6739 : tensor<256xf32>
    %v6742 = stablehlo.subtract %s1b0g3, %v6741 : tensor<256xf32>
    %arsums1b0bt3 = "stablehlo.all_reduce"(%v6356) ({
    ^bb0(%aras1b0bt3: tensor<f32>, %arbs1b0bt3: tensor<f32>):
      %aradds1b0bt3 = stablehlo.add %aras1b0bt3, %arbs1b0bt3 : tensor<f32>
      stablehlo.return %aradds1b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0bt3 = stablehlo.divide %arsums1b0bt3, %arns1b0bt3 : tensor<256xf32>
    %v6743 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6744 = stablehlo.multiply %v6743, %s1b0bt3 : tensor<256xf32>
    %v6745 = stablehlo.add %v6744, %armeans1b0bt3 : tensor<256xf32>
    %v6746 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6747 = stablehlo.multiply %v6746, %s1b0bt3v : tensor<256xf32>
    %v6748 = stablehlo.add %v6747, %v6745 : tensor<256xf32>
    %v6749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6750 = stablehlo.multiply %v6749, %v6748 : tensor<256xf32>
    %v6751 = stablehlo.subtract %s1b0bt3, %v6750 : tensor<256xf32>
    %arsums1b0Wp = "stablehlo.all_reduce"(%v6362) ({
    ^bb0(%aras1b0Wp: tensor<f32>, %arbs1b0Wp: tensor<f32>):
      %aradds1b0Wp = stablehlo.add %aras1b0Wp, %arbs1b0Wp : tensor<f32>
      stablehlo.return %aradds1b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0Wp = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0Wp = stablehlo.divide %arsums1b0Wp, %arns1b0Wp : tensor<256x64x1x1xf32>
    %v6752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6753 = stablehlo.multiply %v6752, %s1b0Wp : tensor<256x64x1x1xf32>
    %v6754 = stablehlo.add %v6753, %armeans1b0Wp : tensor<256x64x1x1xf32>
    %v6755 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6756 = stablehlo.multiply %v6755, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v6757 = stablehlo.add %v6756, %v6754 : tensor<256x64x1x1xf32>
    %v6758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6759 = stablehlo.multiply %v6758, %v6757 : tensor<256x64x1x1xf32>
    %v6760 = stablehlo.subtract %s1b0Wp, %v6759 : tensor<256x64x1x1xf32>
    %arsums1b0gp = "stablehlo.all_reduce"(%v6376) ({
    ^bb0(%aras1b0gp: tensor<f32>, %arbs1b0gp: tensor<f32>):
      %aradds1b0gp = stablehlo.add %aras1b0gp, %arbs1b0gp : tensor<f32>
      stablehlo.return %aradds1b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gp = stablehlo.divide %arsums1b0gp, %arns1b0gp : tensor<256xf32>
    %v6761 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6762 = stablehlo.multiply %v6761, %s1b0gp : tensor<256xf32>
    %v6763 = stablehlo.add %v6762, %armeans1b0gp : tensor<256xf32>
    %v6764 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6765 = stablehlo.multiply %v6764, %s1b0gpv : tensor<256xf32>
    %v6766 = stablehlo.add %v6765, %v6763 : tensor<256xf32>
    %v6767 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6768 = stablehlo.multiply %v6767, %v6766 : tensor<256xf32>
    %v6769 = stablehlo.subtract %s1b0gp, %v6768 : tensor<256xf32>
    %arsums1b0btp = "stablehlo.all_reduce"(%v6379) ({
    ^bb0(%aras1b0btp: tensor<f32>, %arbs1b0btp: tensor<f32>):
      %aradds1b0btp = stablehlo.add %aras1b0btp, %arbs1b0btp : tensor<f32>
      stablehlo.return %aradds1b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0btp = stablehlo.divide %arsums1b0btp, %arns1b0btp : tensor<256xf32>
    %v6770 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6771 = stablehlo.multiply %v6770, %s1b0btp : tensor<256xf32>
    %v6772 = stablehlo.add %v6771, %armeans1b0btp : tensor<256xf32>
    %v6773 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6774 = stablehlo.multiply %v6773, %s1b0btpv : tensor<256xf32>
    %v6775 = stablehlo.add %v6774, %v6772 : tensor<256xf32>
    %v6776 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6777 = stablehlo.multiply %v6776, %v6775 : tensor<256xf32>
    %v6778 = stablehlo.subtract %s1b0btp, %v6777 : tensor<256xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v6006) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x256x1x1xf32>
    %v6779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6780 = stablehlo.multiply %v6779, %s1b1W1 : tensor<64x256x1x1xf32>
    %v6781 = stablehlo.add %v6780, %armeans1b1W1 : tensor<64x256x1x1xf32>
    %v6782 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6783 = stablehlo.multiply %v6782, %s1b1W1v : tensor<64x256x1x1xf32>
    %v6784 = stablehlo.add %v6783, %v6781 : tensor<64x256x1x1xf32>
    %v6785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6786 = stablehlo.multiply %v6785, %v6784 : tensor<64x256x1x1xf32>
    %v6787 = stablehlo.subtract %s1b1W1, %v6786 : tensor<64x256x1x1xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v6020) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v6788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6789 = stablehlo.multiply %v6788, %s1b1g1 : tensor<64xf32>
    %v6790 = stablehlo.add %v6789, %armeans1b1g1 : tensor<64xf32>
    %v6791 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6792 = stablehlo.multiply %v6791, %s1b1g1v : tensor<64xf32>
    %v6793 = stablehlo.add %v6792, %v6790 : tensor<64xf32>
    %v6794 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6795 = stablehlo.multiply %v6794, %v6793 : tensor<64xf32>
    %v6796 = stablehlo.subtract %s1b1g1, %v6795 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v6023) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v6797 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6798 = stablehlo.multiply %v6797, %s1b1bt1 : tensor<64xf32>
    %v6799 = stablehlo.add %v6798, %armeans1b1bt1 : tensor<64xf32>
    %v6800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6801 = stablehlo.multiply %v6800, %s1b1bt1v : tensor<64xf32>
    %v6802 = stablehlo.add %v6801, %v6799 : tensor<64xf32>
    %v6803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6804 = stablehlo.multiply %v6803, %v6802 : tensor<64xf32>
    %v6805 = stablehlo.subtract %s1b1bt1, %v6804 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v6029) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v6806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6807 = stablehlo.multiply %v6806, %s1b1W2 : tensor<64x64x3x3xf32>
    %v6808 = stablehlo.add %v6807, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v6809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6810 = stablehlo.multiply %v6809, %s1b1W2v : tensor<64x64x3x3xf32>
    %v6811 = stablehlo.add %v6810, %v6808 : tensor<64x64x3x3xf32>
    %v6812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6813 = stablehlo.multiply %v6812, %v6811 : tensor<64x64x3x3xf32>
    %v6814 = stablehlo.subtract %s1b1W2, %v6813 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v6043) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v6815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6816 = stablehlo.multiply %v6815, %s1b1g2 : tensor<64xf32>
    %v6817 = stablehlo.add %v6816, %armeans1b1g2 : tensor<64xf32>
    %v6818 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6819 = stablehlo.multiply %v6818, %s1b1g2v : tensor<64xf32>
    %v6820 = stablehlo.add %v6819, %v6817 : tensor<64xf32>
    %v6821 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6822 = stablehlo.multiply %v6821, %v6820 : tensor<64xf32>
    %v6823 = stablehlo.subtract %s1b1g2, %v6822 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v6046) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v6824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6825 = stablehlo.multiply %v6824, %s1b1bt2 : tensor<64xf32>
    %v6826 = stablehlo.add %v6825, %armeans1b1bt2 : tensor<64xf32>
    %v6827 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6828 = stablehlo.multiply %v6827, %s1b1bt2v : tensor<64xf32>
    %v6829 = stablehlo.add %v6828, %v6826 : tensor<64xf32>
    %v6830 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6831 = stablehlo.multiply %v6830, %v6829 : tensor<64xf32>
    %v6832 = stablehlo.subtract %s1b1bt2, %v6831 : tensor<64xf32>
    %arsums1b1W3 = "stablehlo.all_reduce"(%v6052) ({
    ^bb0(%aras1b1W3: tensor<f32>, %arbs1b1W3: tensor<f32>):
      %aradds1b1W3 = stablehlo.add %aras1b1W3, %arbs1b1W3 : tensor<f32>
      stablehlo.return %aradds1b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b1W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b1W3 = stablehlo.divide %arsums1b1W3, %arns1b1W3 : tensor<256x64x1x1xf32>
    %v6833 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6834 = stablehlo.multiply %v6833, %s1b1W3 : tensor<256x64x1x1xf32>
    %v6835 = stablehlo.add %v6834, %armeans1b1W3 : tensor<256x64x1x1xf32>
    %v6836 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6837 = stablehlo.multiply %v6836, %s1b1W3v : tensor<256x64x1x1xf32>
    %v6838 = stablehlo.add %v6837, %v6835 : tensor<256x64x1x1xf32>
    %v6839 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6840 = stablehlo.multiply %v6839, %v6838 : tensor<256x64x1x1xf32>
    %v6841 = stablehlo.subtract %s1b1W3, %v6840 : tensor<256x64x1x1xf32>
    %arsums1b1g3 = "stablehlo.all_reduce"(%v6066) ({
    ^bb0(%aras1b1g3: tensor<f32>, %arbs1b1g3: tensor<f32>):
      %aradds1b1g3 = stablehlo.add %aras1b1g3, %arbs1b1g3 : tensor<f32>
      stablehlo.return %aradds1b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3 = stablehlo.divide %arsums1b1g3, %arns1b1g3 : tensor<256xf32>
    %v6842 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6843 = stablehlo.multiply %v6842, %s1b1g3 : tensor<256xf32>
    %v6844 = stablehlo.add %v6843, %armeans1b1g3 : tensor<256xf32>
    %v6845 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6846 = stablehlo.multiply %v6845, %s1b1g3v : tensor<256xf32>
    %v6847 = stablehlo.add %v6846, %v6844 : tensor<256xf32>
    %v6848 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6849 = stablehlo.multiply %v6848, %v6847 : tensor<256xf32>
    %v6850 = stablehlo.subtract %s1b1g3, %v6849 : tensor<256xf32>
    %arsums1b1bt3 = "stablehlo.all_reduce"(%v6069) ({
    ^bb0(%aras1b1bt3: tensor<f32>, %arbs1b1bt3: tensor<f32>):
      %aradds1b1bt3 = stablehlo.add %aras1b1bt3, %arbs1b1bt3 : tensor<f32>
      stablehlo.return %aradds1b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1bt3 = stablehlo.divide %arsums1b1bt3, %arns1b1bt3 : tensor<256xf32>
    %v6851 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6852 = stablehlo.multiply %v6851, %s1b1bt3 : tensor<256xf32>
    %v6853 = stablehlo.add %v6852, %armeans1b1bt3 : tensor<256xf32>
    %v6854 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6855 = stablehlo.multiply %v6854, %s1b1bt3v : tensor<256xf32>
    %v6856 = stablehlo.add %v6855, %v6853 : tensor<256xf32>
    %v6857 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6858 = stablehlo.multiply %v6857, %v6856 : tensor<256xf32>
    %v6859 = stablehlo.subtract %s1b1bt3, %v6858 : tensor<256xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v5768) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x256x1x1xf32>
    %v6860 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6861 = stablehlo.multiply %v6860, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6862 = stablehlo.add %v6861, %armeans1b2W1 : tensor<64x256x1x1xf32>
    %v6863 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6864 = stablehlo.multiply %v6863, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6865 = stablehlo.add %v6864, %v6862 : tensor<64x256x1x1xf32>
    %v6866 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6867 = stablehlo.multiply %v6866, %v6865 : tensor<64x256x1x1xf32>
    %v6868 = stablehlo.subtract %s1b2W1, %v6867 : tensor<64x256x1x1xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v5782) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v6869 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6870 = stablehlo.multiply %v6869, %s1b2g1 : tensor<64xf32>
    %v6871 = stablehlo.add %v6870, %armeans1b2g1 : tensor<64xf32>
    %v6872 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6873 = stablehlo.multiply %v6872, %s1b2g1v : tensor<64xf32>
    %v6874 = stablehlo.add %v6873, %v6871 : tensor<64xf32>
    %v6875 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6876 = stablehlo.multiply %v6875, %v6874 : tensor<64xf32>
    %v6877 = stablehlo.subtract %s1b2g1, %v6876 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v5785) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v6878 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6879 = stablehlo.multiply %v6878, %s1b2bt1 : tensor<64xf32>
    %v6880 = stablehlo.add %v6879, %armeans1b2bt1 : tensor<64xf32>
    %v6881 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6882 = stablehlo.multiply %v6881, %s1b2bt1v : tensor<64xf32>
    %v6883 = stablehlo.add %v6882, %v6880 : tensor<64xf32>
    %v6884 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6885 = stablehlo.multiply %v6884, %v6883 : tensor<64xf32>
    %v6886 = stablehlo.subtract %s1b2bt1, %v6885 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v5791) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v6887 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6888 = stablehlo.multiply %v6887, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6889 = stablehlo.add %v6888, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v6890 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6891 = stablehlo.multiply %v6890, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6892 = stablehlo.add %v6891, %v6889 : tensor<64x64x3x3xf32>
    %v6893 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6894 = stablehlo.multiply %v6893, %v6892 : tensor<64x64x3x3xf32>
    %v6895 = stablehlo.subtract %s1b2W2, %v6894 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v5805) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v6896 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6897 = stablehlo.multiply %v6896, %s1b2g2 : tensor<64xf32>
    %v6898 = stablehlo.add %v6897, %armeans1b2g2 : tensor<64xf32>
    %v6899 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6900 = stablehlo.multiply %v6899, %s1b2g2v : tensor<64xf32>
    %v6901 = stablehlo.add %v6900, %v6898 : tensor<64xf32>
    %v6902 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6903 = stablehlo.multiply %v6902, %v6901 : tensor<64xf32>
    %v6904 = stablehlo.subtract %s1b2g2, %v6903 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v5808) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v6905 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6906 = stablehlo.multiply %v6905, %s1b2bt2 : tensor<64xf32>
    %v6907 = stablehlo.add %v6906, %armeans1b2bt2 : tensor<64xf32>
    %v6908 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6909 = stablehlo.multiply %v6908, %s1b2bt2v : tensor<64xf32>
    %v6910 = stablehlo.add %v6909, %v6907 : tensor<64xf32>
    %v6911 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6912 = stablehlo.multiply %v6911, %v6910 : tensor<64xf32>
    %v6913 = stablehlo.subtract %s1b2bt2, %v6912 : tensor<64xf32>
    %arsums1b2W3 = "stablehlo.all_reduce"(%v5814) ({
    ^bb0(%aras1b2W3: tensor<f32>, %arbs1b2W3: tensor<f32>):
      %aradds1b2W3 = stablehlo.add %aras1b2W3, %arbs1b2W3 : tensor<f32>
      stablehlo.return %aradds1b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b2W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b2W3 = stablehlo.divide %arsums1b2W3, %arns1b2W3 : tensor<256x64x1x1xf32>
    %v6914 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6915 = stablehlo.multiply %v6914, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6916 = stablehlo.add %v6915, %armeans1b2W3 : tensor<256x64x1x1xf32>
    %v6917 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6918 = stablehlo.multiply %v6917, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6919 = stablehlo.add %v6918, %v6916 : tensor<256x64x1x1xf32>
    %v6920 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6921 = stablehlo.multiply %v6920, %v6919 : tensor<256x64x1x1xf32>
    %v6922 = stablehlo.subtract %s1b2W3, %v6921 : tensor<256x64x1x1xf32>
    %arsums1b2g3 = "stablehlo.all_reduce"(%v5828) ({
    ^bb0(%aras1b2g3: tensor<f32>, %arbs1b2g3: tensor<f32>):
      %aradds1b2g3 = stablehlo.add %aras1b2g3, %arbs1b2g3 : tensor<f32>
      stablehlo.return %aradds1b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3 = stablehlo.divide %arsums1b2g3, %arns1b2g3 : tensor<256xf32>
    %v6923 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6924 = stablehlo.multiply %v6923, %s1b2g3 : tensor<256xf32>
    %v6925 = stablehlo.add %v6924, %armeans1b2g3 : tensor<256xf32>
    %v6926 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6927 = stablehlo.multiply %v6926, %s1b2g3v : tensor<256xf32>
    %v6928 = stablehlo.add %v6927, %v6925 : tensor<256xf32>
    %v6929 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6930 = stablehlo.multiply %v6929, %v6928 : tensor<256xf32>
    %v6931 = stablehlo.subtract %s1b2g3, %v6930 : tensor<256xf32>
    %arsums1b2bt3 = "stablehlo.all_reduce"(%v5831) ({
    ^bb0(%aras1b2bt3: tensor<f32>, %arbs1b2bt3: tensor<f32>):
      %aradds1b2bt3 = stablehlo.add %aras1b2bt3, %arbs1b2bt3 : tensor<f32>
      stablehlo.return %aradds1b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2bt3 = stablehlo.divide %arsums1b2bt3, %arns1b2bt3 : tensor<256xf32>
    %v6932 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6933 = stablehlo.multiply %v6932, %s1b2bt3 : tensor<256xf32>
    %v6934 = stablehlo.add %v6933, %armeans1b2bt3 : tensor<256xf32>
    %v6935 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6936 = stablehlo.multiply %v6935, %s1b2bt3v : tensor<256xf32>
    %v6937 = stablehlo.add %v6936, %v6934 : tensor<256xf32>
    %v6938 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6939 = stablehlo.multiply %v6938, %v6937 : tensor<256xf32>
    %v6940 = stablehlo.subtract %s1b2bt3, %v6939 : tensor<256xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v5505) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x256x1x1xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x256x1x1xf32>
    %v6941 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6942 = stablehlo.multiply %v6941, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6943 = stablehlo.add %v6942, %armeans2b0W1 : tensor<128x256x1x1xf32>
    %v6944 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6945 = stablehlo.multiply %v6944, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6946 = stablehlo.add %v6945, %v6943 : tensor<128x256x1x1xf32>
    %v6947 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6948 = stablehlo.multiply %v6947, %v6946 : tensor<128x256x1x1xf32>
    %v6949 = stablehlo.subtract %s2b0W1, %v6948 : tensor<128x256x1x1xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v5519) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v6950 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6951 = stablehlo.multiply %v6950, %s2b0g1 : tensor<128xf32>
    %v6952 = stablehlo.add %v6951, %armeans2b0g1 : tensor<128xf32>
    %v6953 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6954 = stablehlo.multiply %v6953, %s2b0g1v : tensor<128xf32>
    %v6955 = stablehlo.add %v6954, %v6952 : tensor<128xf32>
    %v6956 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6957 = stablehlo.multiply %v6956, %v6955 : tensor<128xf32>
    %v6958 = stablehlo.subtract %s2b0g1, %v6957 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v5522) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v6959 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6960 = stablehlo.multiply %v6959, %s2b0bt1 : tensor<128xf32>
    %v6961 = stablehlo.add %v6960, %armeans2b0bt1 : tensor<128xf32>
    %v6962 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6963 = stablehlo.multiply %v6962, %s2b0bt1v : tensor<128xf32>
    %v6964 = stablehlo.add %v6963, %v6961 : tensor<128xf32>
    %v6965 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6966 = stablehlo.multiply %v6965, %v6964 : tensor<128xf32>
    %v6967 = stablehlo.subtract %s2b0bt1, %v6966 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v5529) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v6968 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6969 = stablehlo.multiply %v6968, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6970 = stablehlo.add %v6969, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v6971 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6972 = stablehlo.multiply %v6971, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6973 = stablehlo.add %v6972, %v6970 : tensor<128x128x3x3xf32>
    %v6974 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6975 = stablehlo.multiply %v6974, %v6973 : tensor<128x128x3x3xf32>
    %v6976 = stablehlo.subtract %s2b0W2, %v6975 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v5543) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v6977 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6978 = stablehlo.multiply %v6977, %s2b0g2 : tensor<128xf32>
    %v6979 = stablehlo.add %v6978, %armeans2b0g2 : tensor<128xf32>
    %v6980 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6981 = stablehlo.multiply %v6980, %s2b0g2v : tensor<128xf32>
    %v6982 = stablehlo.add %v6981, %v6979 : tensor<128xf32>
    %v6983 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6984 = stablehlo.multiply %v6983, %v6982 : tensor<128xf32>
    %v6985 = stablehlo.subtract %s2b0g2, %v6984 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v5546) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v6986 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6987 = stablehlo.multiply %v6986, %s2b0bt2 : tensor<128xf32>
    %v6988 = stablehlo.add %v6987, %armeans2b0bt2 : tensor<128xf32>
    %v6989 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6990 = stablehlo.multiply %v6989, %s2b0bt2v : tensor<128xf32>
    %v6991 = stablehlo.add %v6990, %v6988 : tensor<128xf32>
    %v6992 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6993 = stablehlo.multiply %v6992, %v6991 : tensor<128xf32>
    %v6994 = stablehlo.subtract %s2b0bt2, %v6993 : tensor<128xf32>
    %arsums2b0W3 = "stablehlo.all_reduce"(%v5552) ({
    ^bb0(%aras2b0W3: tensor<f32>, %arbs2b0W3: tensor<f32>):
      %aradds2b0W3 = stablehlo.add %aras2b0W3, %arbs2b0W3 : tensor<f32>
      stablehlo.return %aradds2b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b0W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b0W3 = stablehlo.divide %arsums2b0W3, %arns2b0W3 : tensor<512x128x1x1xf32>
    %v6995 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6996 = stablehlo.multiply %v6995, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6997 = stablehlo.add %v6996, %armeans2b0W3 : tensor<512x128x1x1xf32>
    %v6998 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6999 = stablehlo.multiply %v6998, %s2b0W3v : tensor<512x128x1x1xf32>
    %v7000 = stablehlo.add %v6999, %v6997 : tensor<512x128x1x1xf32>
    %v7001 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7002 = stablehlo.multiply %v7001, %v7000 : tensor<512x128x1x1xf32>
    %v7003 = stablehlo.subtract %s2b0W3, %v7002 : tensor<512x128x1x1xf32>
    %arsums2b0g3 = "stablehlo.all_reduce"(%v5566) ({
    ^bb0(%aras2b0g3: tensor<f32>, %arbs2b0g3: tensor<f32>):
      %aradds2b0g3 = stablehlo.add %aras2b0g3, %arbs2b0g3 : tensor<f32>
      stablehlo.return %aradds2b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3 = stablehlo.divide %arsums2b0g3, %arns2b0g3 : tensor<512xf32>
    %v7004 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7005 = stablehlo.multiply %v7004, %s2b0g3 : tensor<512xf32>
    %v7006 = stablehlo.add %v7005, %armeans2b0g3 : tensor<512xf32>
    %v7007 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7008 = stablehlo.multiply %v7007, %s2b0g3v : tensor<512xf32>
    %v7009 = stablehlo.add %v7008, %v7006 : tensor<512xf32>
    %v7010 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7011 = stablehlo.multiply %v7010, %v7009 : tensor<512xf32>
    %v7012 = stablehlo.subtract %s2b0g3, %v7011 : tensor<512xf32>
    %arsums2b0bt3 = "stablehlo.all_reduce"(%v5569) ({
    ^bb0(%aras2b0bt3: tensor<f32>, %arbs2b0bt3: tensor<f32>):
      %aradds2b0bt3 = stablehlo.add %aras2b0bt3, %arbs2b0bt3 : tensor<f32>
      stablehlo.return %aradds2b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0bt3 = stablehlo.divide %arsums2b0bt3, %arns2b0bt3 : tensor<512xf32>
    %v7013 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7014 = stablehlo.multiply %v7013, %s2b0bt3 : tensor<512xf32>
    %v7015 = stablehlo.add %v7014, %armeans2b0bt3 : tensor<512xf32>
    %v7016 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7017 = stablehlo.multiply %v7016, %s2b0bt3v : tensor<512xf32>
    %v7018 = stablehlo.add %v7017, %v7015 : tensor<512xf32>
    %v7019 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7020 = stablehlo.multiply %v7019, %v7018 : tensor<512xf32>
    %v7021 = stablehlo.subtract %s2b0bt3, %v7020 : tensor<512xf32>
    %arsums2b0Wp = "stablehlo.all_reduce"(%v5576) ({
    ^bb0(%aras2b0Wp: tensor<f32>, %arbs2b0Wp: tensor<f32>):
      %aradds2b0Wp = stablehlo.add %aras2b0Wp, %arbs2b0Wp : tensor<f32>
      stablehlo.return %aradds2b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arns2b0Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeans2b0Wp = stablehlo.divide %arsums2b0Wp, %arns2b0Wp : tensor<512x256x1x1xf32>
    %v7022 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7023 = stablehlo.multiply %v7022, %s2b0Wp : tensor<512x256x1x1xf32>
    %v7024 = stablehlo.add %v7023, %armeans2b0Wp : tensor<512x256x1x1xf32>
    %v7025 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7026 = stablehlo.multiply %v7025, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v7027 = stablehlo.add %v7026, %v7024 : tensor<512x256x1x1xf32>
    %v7028 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7029 = stablehlo.multiply %v7028, %v7027 : tensor<512x256x1x1xf32>
    %v7030 = stablehlo.subtract %s2b0Wp, %v7029 : tensor<512x256x1x1xf32>
    %arsums2b0gp = "stablehlo.all_reduce"(%v5590) ({
    ^bb0(%aras2b0gp: tensor<f32>, %arbs2b0gp: tensor<f32>):
      %aradds2b0gp = stablehlo.add %aras2b0gp, %arbs2b0gp : tensor<f32>
      stablehlo.return %aradds2b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gp = stablehlo.divide %arsums2b0gp, %arns2b0gp : tensor<512xf32>
    %v7031 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7032 = stablehlo.multiply %v7031, %s2b0gp : tensor<512xf32>
    %v7033 = stablehlo.add %v7032, %armeans2b0gp : tensor<512xf32>
    %v7034 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7035 = stablehlo.multiply %v7034, %s2b0gpv : tensor<512xf32>
    %v7036 = stablehlo.add %v7035, %v7033 : tensor<512xf32>
    %v7037 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7038 = stablehlo.multiply %v7037, %v7036 : tensor<512xf32>
    %v7039 = stablehlo.subtract %s2b0gp, %v7038 : tensor<512xf32>
    %arsums2b0btp = "stablehlo.all_reduce"(%v5593) ({
    ^bb0(%aras2b0btp: tensor<f32>, %arbs2b0btp: tensor<f32>):
      %aradds2b0btp = stablehlo.add %aras2b0btp, %arbs2b0btp : tensor<f32>
      stablehlo.return %aradds2b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0btp = stablehlo.divide %arsums2b0btp, %arns2b0btp : tensor<512xf32>
    %v7040 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7041 = stablehlo.multiply %v7040, %s2b0btp : tensor<512xf32>
    %v7042 = stablehlo.add %v7041, %armeans2b0btp : tensor<512xf32>
    %v7043 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7044 = stablehlo.multiply %v7043, %s2b0btpv : tensor<512xf32>
    %v7045 = stablehlo.add %v7044, %v7042 : tensor<512xf32>
    %v7046 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7047 = stablehlo.multiply %v7046, %v7045 : tensor<512xf32>
    %v7048 = stablehlo.subtract %s2b0btp, %v7047 : tensor<512xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v5218) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x512x1x1xf32>
    %v7049 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7050 = stablehlo.multiply %v7049, %s2b1W1 : tensor<128x512x1x1xf32>
    %v7051 = stablehlo.add %v7050, %armeans2b1W1 : tensor<128x512x1x1xf32>
    %v7052 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7053 = stablehlo.multiply %v7052, %s2b1W1v : tensor<128x512x1x1xf32>
    %v7054 = stablehlo.add %v7053, %v7051 : tensor<128x512x1x1xf32>
    %v7055 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7056 = stablehlo.multiply %v7055, %v7054 : tensor<128x512x1x1xf32>
    %v7057 = stablehlo.subtract %s2b1W1, %v7056 : tensor<128x512x1x1xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v5232) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v7058 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7059 = stablehlo.multiply %v7058, %s2b1g1 : tensor<128xf32>
    %v7060 = stablehlo.add %v7059, %armeans2b1g1 : tensor<128xf32>
    %v7061 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7062 = stablehlo.multiply %v7061, %s2b1g1v : tensor<128xf32>
    %v7063 = stablehlo.add %v7062, %v7060 : tensor<128xf32>
    %v7064 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7065 = stablehlo.multiply %v7064, %v7063 : tensor<128xf32>
    %v7066 = stablehlo.subtract %s2b1g1, %v7065 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v5235) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v7067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7068 = stablehlo.multiply %v7067, %s2b1bt1 : tensor<128xf32>
    %v7069 = stablehlo.add %v7068, %armeans2b1bt1 : tensor<128xf32>
    %v7070 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7071 = stablehlo.multiply %v7070, %s2b1bt1v : tensor<128xf32>
    %v7072 = stablehlo.add %v7071, %v7069 : tensor<128xf32>
    %v7073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7074 = stablehlo.multiply %v7073, %v7072 : tensor<128xf32>
    %v7075 = stablehlo.subtract %s2b1bt1, %v7074 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v5241) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v7076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7077 = stablehlo.multiply %v7076, %s2b1W2 : tensor<128x128x3x3xf32>
    %v7078 = stablehlo.add %v7077, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v7079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7080 = stablehlo.multiply %v7079, %s2b1W2v : tensor<128x128x3x3xf32>
    %v7081 = stablehlo.add %v7080, %v7078 : tensor<128x128x3x3xf32>
    %v7082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7083 = stablehlo.multiply %v7082, %v7081 : tensor<128x128x3x3xf32>
    %v7084 = stablehlo.subtract %s2b1W2, %v7083 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v5255) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v7085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7086 = stablehlo.multiply %v7085, %s2b1g2 : tensor<128xf32>
    %v7087 = stablehlo.add %v7086, %armeans2b1g2 : tensor<128xf32>
    %v7088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7089 = stablehlo.multiply %v7088, %s2b1g2v : tensor<128xf32>
    %v7090 = stablehlo.add %v7089, %v7087 : tensor<128xf32>
    %v7091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7092 = stablehlo.multiply %v7091, %v7090 : tensor<128xf32>
    %v7093 = stablehlo.subtract %s2b1g2, %v7092 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v5258) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v7094 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7095 = stablehlo.multiply %v7094, %s2b1bt2 : tensor<128xf32>
    %v7096 = stablehlo.add %v7095, %armeans2b1bt2 : tensor<128xf32>
    %v7097 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7098 = stablehlo.multiply %v7097, %s2b1bt2v : tensor<128xf32>
    %v7099 = stablehlo.add %v7098, %v7096 : tensor<128xf32>
    %v7100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7101 = stablehlo.multiply %v7100, %v7099 : tensor<128xf32>
    %v7102 = stablehlo.subtract %s2b1bt2, %v7101 : tensor<128xf32>
    %arsums2b1W3 = "stablehlo.all_reduce"(%v5264) ({
    ^bb0(%aras2b1W3: tensor<f32>, %arbs2b1W3: tensor<f32>):
      %aradds2b1W3 = stablehlo.add %aras2b1W3, %arbs2b1W3 : tensor<f32>
      stablehlo.return %aradds2b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b1W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b1W3 = stablehlo.divide %arsums2b1W3, %arns2b1W3 : tensor<512x128x1x1xf32>
    %v7103 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7104 = stablehlo.multiply %v7103, %s2b1W3 : tensor<512x128x1x1xf32>
    %v7105 = stablehlo.add %v7104, %armeans2b1W3 : tensor<512x128x1x1xf32>
    %v7106 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7107 = stablehlo.multiply %v7106, %s2b1W3v : tensor<512x128x1x1xf32>
    %v7108 = stablehlo.add %v7107, %v7105 : tensor<512x128x1x1xf32>
    %v7109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7110 = stablehlo.multiply %v7109, %v7108 : tensor<512x128x1x1xf32>
    %v7111 = stablehlo.subtract %s2b1W3, %v7110 : tensor<512x128x1x1xf32>
    %arsums2b1g3 = "stablehlo.all_reduce"(%v5278) ({
    ^bb0(%aras2b1g3: tensor<f32>, %arbs2b1g3: tensor<f32>):
      %aradds2b1g3 = stablehlo.add %aras2b1g3, %arbs2b1g3 : tensor<f32>
      stablehlo.return %aradds2b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3 = stablehlo.divide %arsums2b1g3, %arns2b1g3 : tensor<512xf32>
    %v7112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7113 = stablehlo.multiply %v7112, %s2b1g3 : tensor<512xf32>
    %v7114 = stablehlo.add %v7113, %armeans2b1g3 : tensor<512xf32>
    %v7115 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7116 = stablehlo.multiply %v7115, %s2b1g3v : tensor<512xf32>
    %v7117 = stablehlo.add %v7116, %v7114 : tensor<512xf32>
    %v7118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7119 = stablehlo.multiply %v7118, %v7117 : tensor<512xf32>
    %v7120 = stablehlo.subtract %s2b1g3, %v7119 : tensor<512xf32>
    %arsums2b1bt3 = "stablehlo.all_reduce"(%v5281) ({
    ^bb0(%aras2b1bt3: tensor<f32>, %arbs2b1bt3: tensor<f32>):
      %aradds2b1bt3 = stablehlo.add %aras2b1bt3, %arbs2b1bt3 : tensor<f32>
      stablehlo.return %aradds2b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1bt3 = stablehlo.divide %arsums2b1bt3, %arns2b1bt3 : tensor<512xf32>
    %v7121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7122 = stablehlo.multiply %v7121, %s2b1bt3 : tensor<512xf32>
    %v7123 = stablehlo.add %v7122, %armeans2b1bt3 : tensor<512xf32>
    %v7124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7125 = stablehlo.multiply %v7124, %s2b1bt3v : tensor<512xf32>
    %v7126 = stablehlo.add %v7125, %v7123 : tensor<512xf32>
    %v7127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7128 = stablehlo.multiply %v7127, %v7126 : tensor<512xf32>
    %v7129 = stablehlo.subtract %s2b1bt3, %v7128 : tensor<512xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v4980) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x512x1x1xf32>
    %v7130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7131 = stablehlo.multiply %v7130, %s2b2W1 : tensor<128x512x1x1xf32>
    %v7132 = stablehlo.add %v7131, %armeans2b2W1 : tensor<128x512x1x1xf32>
    %v7133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7134 = stablehlo.multiply %v7133, %s2b2W1v : tensor<128x512x1x1xf32>
    %v7135 = stablehlo.add %v7134, %v7132 : tensor<128x512x1x1xf32>
    %v7136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7137 = stablehlo.multiply %v7136, %v7135 : tensor<128x512x1x1xf32>
    %v7138 = stablehlo.subtract %s2b2W1, %v7137 : tensor<128x512x1x1xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v4994) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v7139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7140 = stablehlo.multiply %v7139, %s2b2g1 : tensor<128xf32>
    %v7141 = stablehlo.add %v7140, %armeans2b2g1 : tensor<128xf32>
    %v7142 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7143 = stablehlo.multiply %v7142, %s2b2g1v : tensor<128xf32>
    %v7144 = stablehlo.add %v7143, %v7141 : tensor<128xf32>
    %v7145 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7146 = stablehlo.multiply %v7145, %v7144 : tensor<128xf32>
    %v7147 = stablehlo.subtract %s2b2g1, %v7146 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v4997) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v7148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7149 = stablehlo.multiply %v7148, %s2b2bt1 : tensor<128xf32>
    %v7150 = stablehlo.add %v7149, %armeans2b2bt1 : tensor<128xf32>
    %v7151 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7152 = stablehlo.multiply %v7151, %s2b2bt1v : tensor<128xf32>
    %v7153 = stablehlo.add %v7152, %v7150 : tensor<128xf32>
    %v7154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7155 = stablehlo.multiply %v7154, %v7153 : tensor<128xf32>
    %v7156 = stablehlo.subtract %s2b2bt1, %v7155 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v5003) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v7157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7158 = stablehlo.multiply %v7157, %s2b2W2 : tensor<128x128x3x3xf32>
    %v7159 = stablehlo.add %v7158, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v7160 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7161 = stablehlo.multiply %v7160, %s2b2W2v : tensor<128x128x3x3xf32>
    %v7162 = stablehlo.add %v7161, %v7159 : tensor<128x128x3x3xf32>
    %v7163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7164 = stablehlo.multiply %v7163, %v7162 : tensor<128x128x3x3xf32>
    %v7165 = stablehlo.subtract %s2b2W2, %v7164 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v5017) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v7166 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7167 = stablehlo.multiply %v7166, %s2b2g2 : tensor<128xf32>
    %v7168 = stablehlo.add %v7167, %armeans2b2g2 : tensor<128xf32>
    %v7169 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7170 = stablehlo.multiply %v7169, %s2b2g2v : tensor<128xf32>
    %v7171 = stablehlo.add %v7170, %v7168 : tensor<128xf32>
    %v7172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7173 = stablehlo.multiply %v7172, %v7171 : tensor<128xf32>
    %v7174 = stablehlo.subtract %s2b2g2, %v7173 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v5020) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v7175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7176 = stablehlo.multiply %v7175, %s2b2bt2 : tensor<128xf32>
    %v7177 = stablehlo.add %v7176, %armeans2b2bt2 : tensor<128xf32>
    %v7178 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7179 = stablehlo.multiply %v7178, %s2b2bt2v : tensor<128xf32>
    %v7180 = stablehlo.add %v7179, %v7177 : tensor<128xf32>
    %v7181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7182 = stablehlo.multiply %v7181, %v7180 : tensor<128xf32>
    %v7183 = stablehlo.subtract %s2b2bt2, %v7182 : tensor<128xf32>
    %arsums2b2W3 = "stablehlo.all_reduce"(%v5026) ({
    ^bb0(%aras2b2W3: tensor<f32>, %arbs2b2W3: tensor<f32>):
      %aradds2b2W3 = stablehlo.add %aras2b2W3, %arbs2b2W3 : tensor<f32>
      stablehlo.return %aradds2b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b2W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b2W3 = stablehlo.divide %arsums2b2W3, %arns2b2W3 : tensor<512x128x1x1xf32>
    %v7184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7185 = stablehlo.multiply %v7184, %s2b2W3 : tensor<512x128x1x1xf32>
    %v7186 = stablehlo.add %v7185, %armeans2b2W3 : tensor<512x128x1x1xf32>
    %v7187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7188 = stablehlo.multiply %v7187, %s2b2W3v : tensor<512x128x1x1xf32>
    %v7189 = stablehlo.add %v7188, %v7186 : tensor<512x128x1x1xf32>
    %v7190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7191 = stablehlo.multiply %v7190, %v7189 : tensor<512x128x1x1xf32>
    %v7192 = stablehlo.subtract %s2b2W3, %v7191 : tensor<512x128x1x1xf32>
    %arsums2b2g3 = "stablehlo.all_reduce"(%v5040) ({
    ^bb0(%aras2b2g3: tensor<f32>, %arbs2b2g3: tensor<f32>):
      %aradds2b2g3 = stablehlo.add %aras2b2g3, %arbs2b2g3 : tensor<f32>
      stablehlo.return %aradds2b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3 = stablehlo.divide %arsums2b2g3, %arns2b2g3 : tensor<512xf32>
    %v7193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7194 = stablehlo.multiply %v7193, %s2b2g3 : tensor<512xf32>
    %v7195 = stablehlo.add %v7194, %armeans2b2g3 : tensor<512xf32>
    %v7196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7197 = stablehlo.multiply %v7196, %s2b2g3v : tensor<512xf32>
    %v7198 = stablehlo.add %v7197, %v7195 : tensor<512xf32>
    %v7199 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7200 = stablehlo.multiply %v7199, %v7198 : tensor<512xf32>
    %v7201 = stablehlo.subtract %s2b2g3, %v7200 : tensor<512xf32>
    %arsums2b2bt3 = "stablehlo.all_reduce"(%v5043) ({
    ^bb0(%aras2b2bt3: tensor<f32>, %arbs2b2bt3: tensor<f32>):
      %aradds2b2bt3 = stablehlo.add %aras2b2bt3, %arbs2b2bt3 : tensor<f32>
      stablehlo.return %aradds2b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2bt3 = stablehlo.divide %arsums2b2bt3, %arns2b2bt3 : tensor<512xf32>
    %v7202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7203 = stablehlo.multiply %v7202, %s2b2bt3 : tensor<512xf32>
    %v7204 = stablehlo.add %v7203, %armeans2b2bt3 : tensor<512xf32>
    %v7205 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7206 = stablehlo.multiply %v7205, %s2b2bt3v : tensor<512xf32>
    %v7207 = stablehlo.add %v7206, %v7204 : tensor<512xf32>
    %v7208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7209 = stablehlo.multiply %v7208, %v7207 : tensor<512xf32>
    %v7210 = stablehlo.subtract %s2b2bt3, %v7209 : tensor<512xf32>
    %arsums2b3W1 = "stablehlo.all_reduce"(%v4742) ({
    ^bb0(%aras2b3W1: tensor<f32>, %arbs2b3W1: tensor<f32>):
      %aradds2b3W1 = stablehlo.add %aras2b3W1, %arbs2b3W1 : tensor<f32>
      stablehlo.return %aradds2b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b3W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b3W1 = stablehlo.divide %arsums2b3W1, %arns2b3W1 : tensor<128x512x1x1xf32>
    %v7211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7212 = stablehlo.multiply %v7211, %s2b3W1 : tensor<128x512x1x1xf32>
    %v7213 = stablehlo.add %v7212, %armeans2b3W1 : tensor<128x512x1x1xf32>
    %v7214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7215 = stablehlo.multiply %v7214, %s2b3W1v : tensor<128x512x1x1xf32>
    %v7216 = stablehlo.add %v7215, %v7213 : tensor<128x512x1x1xf32>
    %v7217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7218 = stablehlo.multiply %v7217, %v7216 : tensor<128x512x1x1xf32>
    %v7219 = stablehlo.subtract %s2b3W1, %v7218 : tensor<128x512x1x1xf32>
    %arsums2b3g1 = "stablehlo.all_reduce"(%v4756) ({
    ^bb0(%aras2b3g1: tensor<f32>, %arbs2b3g1: tensor<f32>):
      %aradds2b3g1 = stablehlo.add %aras2b3g1, %arbs2b3g1 : tensor<f32>
      stablehlo.return %aradds2b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1 = stablehlo.divide %arsums2b3g1, %arns2b3g1 : tensor<128xf32>
    %v7220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7221 = stablehlo.multiply %v7220, %s2b3g1 : tensor<128xf32>
    %v7222 = stablehlo.add %v7221, %armeans2b3g1 : tensor<128xf32>
    %v7223 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7224 = stablehlo.multiply %v7223, %s2b3g1v : tensor<128xf32>
    %v7225 = stablehlo.add %v7224, %v7222 : tensor<128xf32>
    %v7226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7227 = stablehlo.multiply %v7226, %v7225 : tensor<128xf32>
    %v7228 = stablehlo.subtract %s2b3g1, %v7227 : tensor<128xf32>
    %arsums2b3bt1 = "stablehlo.all_reduce"(%v4759) ({
    ^bb0(%aras2b3bt1: tensor<f32>, %arbs2b3bt1: tensor<f32>):
      %aradds2b3bt1 = stablehlo.add %aras2b3bt1, %arbs2b3bt1 : tensor<f32>
      stablehlo.return %aradds2b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt1 = stablehlo.divide %arsums2b3bt1, %arns2b3bt1 : tensor<128xf32>
    %v7229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7230 = stablehlo.multiply %v7229, %s2b3bt1 : tensor<128xf32>
    %v7231 = stablehlo.add %v7230, %armeans2b3bt1 : tensor<128xf32>
    %v7232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7233 = stablehlo.multiply %v7232, %s2b3bt1v : tensor<128xf32>
    %v7234 = stablehlo.add %v7233, %v7231 : tensor<128xf32>
    %v7235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7236 = stablehlo.multiply %v7235, %v7234 : tensor<128xf32>
    %v7237 = stablehlo.subtract %s2b3bt1, %v7236 : tensor<128xf32>
    %arsums2b3W2 = "stablehlo.all_reduce"(%v4765) ({
    ^bb0(%aras2b3W2: tensor<f32>, %arbs2b3W2: tensor<f32>):
      %aradds2b3W2 = stablehlo.add %aras2b3W2, %arbs2b3W2 : tensor<f32>
      stablehlo.return %aradds2b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b3W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b3W2 = stablehlo.divide %arsums2b3W2, %arns2b3W2 : tensor<128x128x3x3xf32>
    %v7238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7239 = stablehlo.multiply %v7238, %s2b3W2 : tensor<128x128x3x3xf32>
    %v7240 = stablehlo.add %v7239, %armeans2b3W2 : tensor<128x128x3x3xf32>
    %v7241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7242 = stablehlo.multiply %v7241, %s2b3W2v : tensor<128x128x3x3xf32>
    %v7243 = stablehlo.add %v7242, %v7240 : tensor<128x128x3x3xf32>
    %v7244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7245 = stablehlo.multiply %v7244, %v7243 : tensor<128x128x3x3xf32>
    %v7246 = stablehlo.subtract %s2b3W2, %v7245 : tensor<128x128x3x3xf32>
    %arsums2b3g2 = "stablehlo.all_reduce"(%v4779) ({
    ^bb0(%aras2b3g2: tensor<f32>, %arbs2b3g2: tensor<f32>):
      %aradds2b3g2 = stablehlo.add %aras2b3g2, %arbs2b3g2 : tensor<f32>
      stablehlo.return %aradds2b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2 = stablehlo.divide %arsums2b3g2, %arns2b3g2 : tensor<128xf32>
    %v7247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7248 = stablehlo.multiply %v7247, %s2b3g2 : tensor<128xf32>
    %v7249 = stablehlo.add %v7248, %armeans2b3g2 : tensor<128xf32>
    %v7250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7251 = stablehlo.multiply %v7250, %s2b3g2v : tensor<128xf32>
    %v7252 = stablehlo.add %v7251, %v7249 : tensor<128xf32>
    %v7253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7254 = stablehlo.multiply %v7253, %v7252 : tensor<128xf32>
    %v7255 = stablehlo.subtract %s2b3g2, %v7254 : tensor<128xf32>
    %arsums2b3bt2 = "stablehlo.all_reduce"(%v4782) ({
    ^bb0(%aras2b3bt2: tensor<f32>, %arbs2b3bt2: tensor<f32>):
      %aradds2b3bt2 = stablehlo.add %aras2b3bt2, %arbs2b3bt2 : tensor<f32>
      stablehlo.return %aradds2b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt2 = stablehlo.divide %arsums2b3bt2, %arns2b3bt2 : tensor<128xf32>
    %v7256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7257 = stablehlo.multiply %v7256, %s2b3bt2 : tensor<128xf32>
    %v7258 = stablehlo.add %v7257, %armeans2b3bt2 : tensor<128xf32>
    %v7259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7260 = stablehlo.multiply %v7259, %s2b3bt2v : tensor<128xf32>
    %v7261 = stablehlo.add %v7260, %v7258 : tensor<128xf32>
    %v7262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7263 = stablehlo.multiply %v7262, %v7261 : tensor<128xf32>
    %v7264 = stablehlo.subtract %s2b3bt2, %v7263 : tensor<128xf32>
    %arsums2b3W3 = "stablehlo.all_reduce"(%v4788) ({
    ^bb0(%aras2b3W3: tensor<f32>, %arbs2b3W3: tensor<f32>):
      %aradds2b3W3 = stablehlo.add %aras2b3W3, %arbs2b3W3 : tensor<f32>
      stablehlo.return %aradds2b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b3W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b3W3 = stablehlo.divide %arsums2b3W3, %arns2b3W3 : tensor<512x128x1x1xf32>
    %v7265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7266 = stablehlo.multiply %v7265, %s2b3W3 : tensor<512x128x1x1xf32>
    %v7267 = stablehlo.add %v7266, %armeans2b3W3 : tensor<512x128x1x1xf32>
    %v7268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7269 = stablehlo.multiply %v7268, %s2b3W3v : tensor<512x128x1x1xf32>
    %v7270 = stablehlo.add %v7269, %v7267 : tensor<512x128x1x1xf32>
    %v7271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7272 = stablehlo.multiply %v7271, %v7270 : tensor<512x128x1x1xf32>
    %v7273 = stablehlo.subtract %s2b3W3, %v7272 : tensor<512x128x1x1xf32>
    %arsums2b3g3 = "stablehlo.all_reduce"(%v4802) ({
    ^bb0(%aras2b3g3: tensor<f32>, %arbs2b3g3: tensor<f32>):
      %aradds2b3g3 = stablehlo.add %aras2b3g3, %arbs2b3g3 : tensor<f32>
      stablehlo.return %aradds2b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3 = stablehlo.divide %arsums2b3g3, %arns2b3g3 : tensor<512xf32>
    %v7274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7275 = stablehlo.multiply %v7274, %s2b3g3 : tensor<512xf32>
    %v7276 = stablehlo.add %v7275, %armeans2b3g3 : tensor<512xf32>
    %v7277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7278 = stablehlo.multiply %v7277, %s2b3g3v : tensor<512xf32>
    %v7279 = stablehlo.add %v7278, %v7276 : tensor<512xf32>
    %v7280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7281 = stablehlo.multiply %v7280, %v7279 : tensor<512xf32>
    %v7282 = stablehlo.subtract %s2b3g3, %v7281 : tensor<512xf32>
    %arsums2b3bt3 = "stablehlo.all_reduce"(%v4805) ({
    ^bb0(%aras2b3bt3: tensor<f32>, %arbs2b3bt3: tensor<f32>):
      %aradds2b3bt3 = stablehlo.add %aras2b3bt3, %arbs2b3bt3 : tensor<f32>
      stablehlo.return %aradds2b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3bt3 = stablehlo.divide %arsums2b3bt3, %arns2b3bt3 : tensor<512xf32>
    %v7283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7284 = stablehlo.multiply %v7283, %s2b3bt3 : tensor<512xf32>
    %v7285 = stablehlo.add %v7284, %armeans2b3bt3 : tensor<512xf32>
    %v7286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7287 = stablehlo.multiply %v7286, %s2b3bt3v : tensor<512xf32>
    %v7288 = stablehlo.add %v7287, %v7285 : tensor<512xf32>
    %v7289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7290 = stablehlo.multiply %v7289, %v7288 : tensor<512xf32>
    %v7291 = stablehlo.subtract %s2b3bt3, %v7290 : tensor<512xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v4479) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x512x1x1xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x512x1x1xf32>
    %v7292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7293 = stablehlo.multiply %v7292, %s3b0W1 : tensor<256x512x1x1xf32>
    %v7294 = stablehlo.add %v7293, %armeans3b0W1 : tensor<256x512x1x1xf32>
    %v7295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7296 = stablehlo.multiply %v7295, %s3b0W1v : tensor<256x512x1x1xf32>
    %v7297 = stablehlo.add %v7296, %v7294 : tensor<256x512x1x1xf32>
    %v7298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7299 = stablehlo.multiply %v7298, %v7297 : tensor<256x512x1x1xf32>
    %v7300 = stablehlo.subtract %s3b0W1, %v7299 : tensor<256x512x1x1xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v4493) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.multiply %v7301, %s3b0g1 : tensor<256xf32>
    %v7303 = stablehlo.add %v7302, %armeans3b0g1 : tensor<256xf32>
    %v7304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7305 = stablehlo.multiply %v7304, %s3b0g1v : tensor<256xf32>
    %v7306 = stablehlo.add %v7305, %v7303 : tensor<256xf32>
    %v7307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7308 = stablehlo.multiply %v7307, %v7306 : tensor<256xf32>
    %v7309 = stablehlo.subtract %s3b0g1, %v7308 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v4496) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v7310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7311 = stablehlo.multiply %v7310, %s3b0bt1 : tensor<256xf32>
    %v7312 = stablehlo.add %v7311, %armeans3b0bt1 : tensor<256xf32>
    %v7313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7314 = stablehlo.multiply %v7313, %s3b0bt1v : tensor<256xf32>
    %v7315 = stablehlo.add %v7314, %v7312 : tensor<256xf32>
    %v7316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7317 = stablehlo.multiply %v7316, %v7315 : tensor<256xf32>
    %v7318 = stablehlo.subtract %s3b0bt1, %v7317 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v4503) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v7319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7320 = stablehlo.multiply %v7319, %s3b0W2 : tensor<256x256x3x3xf32>
    %v7321 = stablehlo.add %v7320, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v7322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7323 = stablehlo.multiply %v7322, %s3b0W2v : tensor<256x256x3x3xf32>
    %v7324 = stablehlo.add %v7323, %v7321 : tensor<256x256x3x3xf32>
    %v7325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7326 = stablehlo.multiply %v7325, %v7324 : tensor<256x256x3x3xf32>
    %v7327 = stablehlo.subtract %s3b0W2, %v7326 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v4517) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v7328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7329 = stablehlo.multiply %v7328, %s3b0g2 : tensor<256xf32>
    %v7330 = stablehlo.add %v7329, %armeans3b0g2 : tensor<256xf32>
    %v7331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7332 = stablehlo.multiply %v7331, %s3b0g2v : tensor<256xf32>
    %v7333 = stablehlo.add %v7332, %v7330 : tensor<256xf32>
    %v7334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7335 = stablehlo.multiply %v7334, %v7333 : tensor<256xf32>
    %v7336 = stablehlo.subtract %s3b0g2, %v7335 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v4520) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v7337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7338 = stablehlo.multiply %v7337, %s3b0bt2 : tensor<256xf32>
    %v7339 = stablehlo.add %v7338, %armeans3b0bt2 : tensor<256xf32>
    %v7340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7341 = stablehlo.multiply %v7340, %s3b0bt2v : tensor<256xf32>
    %v7342 = stablehlo.add %v7341, %v7339 : tensor<256xf32>
    %v7343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7344 = stablehlo.multiply %v7343, %v7342 : tensor<256xf32>
    %v7345 = stablehlo.subtract %s3b0bt2, %v7344 : tensor<256xf32>
    %arsums3b0W3 = "stablehlo.all_reduce"(%v4526) ({
    ^bb0(%aras3b0W3: tensor<f32>, %arbs3b0W3: tensor<f32>):
      %aradds3b0W3 = stablehlo.add %aras3b0W3, %arbs3b0W3 : tensor<f32>
      stablehlo.return %aradds3b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b0W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b0W3 = stablehlo.divide %arsums3b0W3, %arns3b0W3 : tensor<1024x256x1x1xf32>
    %v7346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7347 = stablehlo.multiply %v7346, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v7348 = stablehlo.add %v7347, %armeans3b0W3 : tensor<1024x256x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7350 = stablehlo.multiply %v7349, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v7351 = stablehlo.add %v7350, %v7348 : tensor<1024x256x1x1xf32>
    %v7352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7353 = stablehlo.multiply %v7352, %v7351 : tensor<1024x256x1x1xf32>
    %v7354 = stablehlo.subtract %s3b0W3, %v7353 : tensor<1024x256x1x1xf32>
    %arsums3b0g3 = "stablehlo.all_reduce"(%v4540) ({
    ^bb0(%aras3b0g3: tensor<f32>, %arbs3b0g3: tensor<f32>):
      %aradds3b0g3 = stablehlo.add %aras3b0g3, %arbs3b0g3 : tensor<f32>
      stablehlo.return %aradds3b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3 = stablehlo.divide %arsums3b0g3, %arns3b0g3 : tensor<1024xf32>
    %v7355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7356 = stablehlo.multiply %v7355, %s3b0g3 : tensor<1024xf32>
    %v7357 = stablehlo.add %v7356, %armeans3b0g3 : tensor<1024xf32>
    %v7358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7359 = stablehlo.multiply %v7358, %s3b0g3v : tensor<1024xf32>
    %v7360 = stablehlo.add %v7359, %v7357 : tensor<1024xf32>
    %v7361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7362 = stablehlo.multiply %v7361, %v7360 : tensor<1024xf32>
    %v7363 = stablehlo.subtract %s3b0g3, %v7362 : tensor<1024xf32>
    %arsums3b0bt3 = "stablehlo.all_reduce"(%v4543) ({
    ^bb0(%aras3b0bt3: tensor<f32>, %arbs3b0bt3: tensor<f32>):
      %aradds3b0bt3 = stablehlo.add %aras3b0bt3, %arbs3b0bt3 : tensor<f32>
      stablehlo.return %aradds3b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0bt3 = stablehlo.divide %arsums3b0bt3, %arns3b0bt3 : tensor<1024xf32>
    %v7364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7365 = stablehlo.multiply %v7364, %s3b0bt3 : tensor<1024xf32>
    %v7366 = stablehlo.add %v7365, %armeans3b0bt3 : tensor<1024xf32>
    %v7367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7368 = stablehlo.multiply %v7367, %s3b0bt3v : tensor<1024xf32>
    %v7369 = stablehlo.add %v7368, %v7366 : tensor<1024xf32>
    %v7370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7371 = stablehlo.multiply %v7370, %v7369 : tensor<1024xf32>
    %v7372 = stablehlo.subtract %s3b0bt3, %v7371 : tensor<1024xf32>
    %arsums3b0Wp = "stablehlo.all_reduce"(%v4550) ({
    ^bb0(%aras3b0Wp: tensor<f32>, %arbs3b0Wp: tensor<f32>):
      %aradds3b0Wp = stablehlo.add %aras3b0Wp, %arbs3b0Wp : tensor<f32>
      stablehlo.return %aradds3b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %arns3b0Wp = stablehlo.constant dense<4.0> : tensor<1024x512x1x1xf32>
    %armeans3b0Wp = stablehlo.divide %arsums3b0Wp, %arns3b0Wp : tensor<1024x512x1x1xf32>
    %v7373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7374 = stablehlo.multiply %v7373, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v7375 = stablehlo.add %v7374, %armeans3b0Wp : tensor<1024x512x1x1xf32>
    %v7376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7377 = stablehlo.multiply %v7376, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v7378 = stablehlo.add %v7377, %v7375 : tensor<1024x512x1x1xf32>
    %v7379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7380 = stablehlo.multiply %v7379, %v7378 : tensor<1024x512x1x1xf32>
    %v7381 = stablehlo.subtract %s3b0Wp, %v7380 : tensor<1024x512x1x1xf32>
    %arsums3b0gp = "stablehlo.all_reduce"(%v4564) ({
    ^bb0(%aras3b0gp: tensor<f32>, %arbs3b0gp: tensor<f32>):
      %aradds3b0gp = stablehlo.add %aras3b0gp, %arbs3b0gp : tensor<f32>
      stablehlo.return %aradds3b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gp = stablehlo.divide %arsums3b0gp, %arns3b0gp : tensor<1024xf32>
    %v7382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7383 = stablehlo.multiply %v7382, %s3b0gp : tensor<1024xf32>
    %v7384 = stablehlo.add %v7383, %armeans3b0gp : tensor<1024xf32>
    %v7385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7386 = stablehlo.multiply %v7385, %s3b0gpv : tensor<1024xf32>
    %v7387 = stablehlo.add %v7386, %v7384 : tensor<1024xf32>
    %v7388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7389 = stablehlo.multiply %v7388, %v7387 : tensor<1024xf32>
    %v7390 = stablehlo.subtract %s3b0gp, %v7389 : tensor<1024xf32>
    %arsums3b0btp = "stablehlo.all_reduce"(%v4567) ({
    ^bb0(%aras3b0btp: tensor<f32>, %arbs3b0btp: tensor<f32>):
      %aradds3b0btp = stablehlo.add %aras3b0btp, %arbs3b0btp : tensor<f32>
      stablehlo.return %aradds3b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0btp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0btp = stablehlo.divide %arsums3b0btp, %arns3b0btp : tensor<1024xf32>
    %v7391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7392 = stablehlo.multiply %v7391, %s3b0btp : tensor<1024xf32>
    %v7393 = stablehlo.add %v7392, %armeans3b0btp : tensor<1024xf32>
    %v7394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7395 = stablehlo.multiply %v7394, %s3b0btpv : tensor<1024xf32>
    %v7396 = stablehlo.add %v7395, %v7393 : tensor<1024xf32>
    %v7397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7398 = stablehlo.multiply %v7397, %v7396 : tensor<1024xf32>
    %v7399 = stablehlo.subtract %s3b0btp, %v7398 : tensor<1024xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v4192) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x1024x1x1xf32>
    %v7400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7401 = stablehlo.multiply %v7400, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v7402 = stablehlo.add %v7401, %armeans3b1W1 : tensor<256x1024x1x1xf32>
    %v7403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7404 = stablehlo.multiply %v7403, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v7405 = stablehlo.add %v7404, %v7402 : tensor<256x1024x1x1xf32>
    %v7406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7407 = stablehlo.multiply %v7406, %v7405 : tensor<256x1024x1x1xf32>
    %v7408 = stablehlo.subtract %s3b1W1, %v7407 : tensor<256x1024x1x1xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v4206) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v7409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7410 = stablehlo.multiply %v7409, %s3b1g1 : tensor<256xf32>
    %v7411 = stablehlo.add %v7410, %armeans3b1g1 : tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.multiply %v7412, %s3b1g1v : tensor<256xf32>
    %v7414 = stablehlo.add %v7413, %v7411 : tensor<256xf32>
    %v7415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7416 = stablehlo.multiply %v7415, %v7414 : tensor<256xf32>
    %v7417 = stablehlo.subtract %s3b1g1, %v7416 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v4209) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v7418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7419 = stablehlo.multiply %v7418, %s3b1bt1 : tensor<256xf32>
    %v7420 = stablehlo.add %v7419, %armeans3b1bt1 : tensor<256xf32>
    %v7421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7422 = stablehlo.multiply %v7421, %s3b1bt1v : tensor<256xf32>
    %v7423 = stablehlo.add %v7422, %v7420 : tensor<256xf32>
    %v7424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7425 = stablehlo.multiply %v7424, %v7423 : tensor<256xf32>
    %v7426 = stablehlo.subtract %s3b1bt1, %v7425 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v4215) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v7427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7428 = stablehlo.multiply %v7427, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7429 = stablehlo.add %v7428, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7431 = stablehlo.multiply %v7430, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7432 = stablehlo.add %v7431, %v7429 : tensor<256x256x3x3xf32>
    %v7433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7434 = stablehlo.multiply %v7433, %v7432 : tensor<256x256x3x3xf32>
    %v7435 = stablehlo.subtract %s3b1W2, %v7434 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v4229) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7437 = stablehlo.multiply %v7436, %s3b1g2 : tensor<256xf32>
    %v7438 = stablehlo.add %v7437, %armeans3b1g2 : tensor<256xf32>
    %v7439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7440 = stablehlo.multiply %v7439, %s3b1g2v : tensor<256xf32>
    %v7441 = stablehlo.add %v7440, %v7438 : tensor<256xf32>
    %v7442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7443 = stablehlo.multiply %v7442, %v7441 : tensor<256xf32>
    %v7444 = stablehlo.subtract %s3b1g2, %v7443 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v4232) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7446 = stablehlo.multiply %v7445, %s3b1bt2 : tensor<256xf32>
    %v7447 = stablehlo.add %v7446, %armeans3b1bt2 : tensor<256xf32>
    %v7448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7449 = stablehlo.multiply %v7448, %s3b1bt2v : tensor<256xf32>
    %v7450 = stablehlo.add %v7449, %v7447 : tensor<256xf32>
    %v7451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7452 = stablehlo.multiply %v7451, %v7450 : tensor<256xf32>
    %v7453 = stablehlo.subtract %s3b1bt2, %v7452 : tensor<256xf32>
    %arsums3b1W3 = "stablehlo.all_reduce"(%v4238) ({
    ^bb0(%aras3b1W3: tensor<f32>, %arbs3b1W3: tensor<f32>):
      %aradds3b1W3 = stablehlo.add %aras3b1W3, %arbs3b1W3 : tensor<f32>
      stablehlo.return %aradds3b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b1W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b1W3 = stablehlo.divide %arsums3b1W3, %arns3b1W3 : tensor<1024x256x1x1xf32>
    %v7454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7455 = stablehlo.multiply %v7454, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v7456 = stablehlo.add %v7455, %armeans3b1W3 : tensor<1024x256x1x1xf32>
    %v7457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7458 = stablehlo.multiply %v7457, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v7459 = stablehlo.add %v7458, %v7456 : tensor<1024x256x1x1xf32>
    %v7460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7461 = stablehlo.multiply %v7460, %v7459 : tensor<1024x256x1x1xf32>
    %v7462 = stablehlo.subtract %s3b1W3, %v7461 : tensor<1024x256x1x1xf32>
    %arsums3b1g3 = "stablehlo.all_reduce"(%v4252) ({
    ^bb0(%aras3b1g3: tensor<f32>, %arbs3b1g3: tensor<f32>):
      %aradds3b1g3 = stablehlo.add %aras3b1g3, %arbs3b1g3 : tensor<f32>
      stablehlo.return %aradds3b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3 = stablehlo.divide %arsums3b1g3, %arns3b1g3 : tensor<1024xf32>
    %v7463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7464 = stablehlo.multiply %v7463, %s3b1g3 : tensor<1024xf32>
    %v7465 = stablehlo.add %v7464, %armeans3b1g3 : tensor<1024xf32>
    %v7466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7467 = stablehlo.multiply %v7466, %s3b1g3v : tensor<1024xf32>
    %v7468 = stablehlo.add %v7467, %v7465 : tensor<1024xf32>
    %v7469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7470 = stablehlo.multiply %v7469, %v7468 : tensor<1024xf32>
    %v7471 = stablehlo.subtract %s3b1g3, %v7470 : tensor<1024xf32>
    %arsums3b1bt3 = "stablehlo.all_reduce"(%v4255) ({
    ^bb0(%aras3b1bt3: tensor<f32>, %arbs3b1bt3: tensor<f32>):
      %aradds3b1bt3 = stablehlo.add %aras3b1bt3, %arbs3b1bt3 : tensor<f32>
      stablehlo.return %aradds3b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1bt3 = stablehlo.divide %arsums3b1bt3, %arns3b1bt3 : tensor<1024xf32>
    %v7472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7473 = stablehlo.multiply %v7472, %s3b1bt3 : tensor<1024xf32>
    %v7474 = stablehlo.add %v7473, %armeans3b1bt3 : tensor<1024xf32>
    %v7475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7476 = stablehlo.multiply %v7475, %s3b1bt3v : tensor<1024xf32>
    %v7477 = stablehlo.add %v7476, %v7474 : tensor<1024xf32>
    %v7478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7479 = stablehlo.multiply %v7478, %v7477 : tensor<1024xf32>
    %v7480 = stablehlo.subtract %s3b1bt3, %v7479 : tensor<1024xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v3954) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x1024x1x1xf32>
    %v7481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7482 = stablehlo.multiply %v7481, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v7483 = stablehlo.add %v7482, %armeans3b2W1 : tensor<256x1024x1x1xf32>
    %v7484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7485 = stablehlo.multiply %v7484, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v7486 = stablehlo.add %v7485, %v7483 : tensor<256x1024x1x1xf32>
    %v7487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7488 = stablehlo.multiply %v7487, %v7486 : tensor<256x1024x1x1xf32>
    %v7489 = stablehlo.subtract %s3b2W1, %v7488 : tensor<256x1024x1x1xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v3968) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7491 = stablehlo.multiply %v7490, %s3b2g1 : tensor<256xf32>
    %v7492 = stablehlo.add %v7491, %armeans3b2g1 : tensor<256xf32>
    %v7493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7494 = stablehlo.multiply %v7493, %s3b2g1v : tensor<256xf32>
    %v7495 = stablehlo.add %v7494, %v7492 : tensor<256xf32>
    %v7496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7497 = stablehlo.multiply %v7496, %v7495 : tensor<256xf32>
    %v7498 = stablehlo.subtract %s3b2g1, %v7497 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v3971) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7500 = stablehlo.multiply %v7499, %s3b2bt1 : tensor<256xf32>
    %v7501 = stablehlo.add %v7500, %armeans3b2bt1 : tensor<256xf32>
    %v7502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7503 = stablehlo.multiply %v7502, %s3b2bt1v : tensor<256xf32>
    %v7504 = stablehlo.add %v7503, %v7501 : tensor<256xf32>
    %v7505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7506 = stablehlo.multiply %v7505, %v7504 : tensor<256xf32>
    %v7507 = stablehlo.subtract %s3b2bt1, %v7506 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v3977) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7509 = stablehlo.multiply %v7508, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7510 = stablehlo.add %v7509, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7512 = stablehlo.multiply %v7511, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7513 = stablehlo.add %v7512, %v7510 : tensor<256x256x3x3xf32>
    %v7514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7515 = stablehlo.multiply %v7514, %v7513 : tensor<256x256x3x3xf32>
    %v7516 = stablehlo.subtract %s3b2W2, %v7515 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v3991) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7518 = stablehlo.multiply %v7517, %s3b2g2 : tensor<256xf32>
    %v7519 = stablehlo.add %v7518, %armeans3b2g2 : tensor<256xf32>
    %v7520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7521 = stablehlo.multiply %v7520, %s3b2g2v : tensor<256xf32>
    %v7522 = stablehlo.add %v7521, %v7519 : tensor<256xf32>
    %v7523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7524 = stablehlo.multiply %v7523, %v7522 : tensor<256xf32>
    %v7525 = stablehlo.subtract %s3b2g2, %v7524 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v3994) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7527 = stablehlo.multiply %v7526, %s3b2bt2 : tensor<256xf32>
    %v7528 = stablehlo.add %v7527, %armeans3b2bt2 : tensor<256xf32>
    %v7529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7530 = stablehlo.multiply %v7529, %s3b2bt2v : tensor<256xf32>
    %v7531 = stablehlo.add %v7530, %v7528 : tensor<256xf32>
    %v7532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7533 = stablehlo.multiply %v7532, %v7531 : tensor<256xf32>
    %v7534 = stablehlo.subtract %s3b2bt2, %v7533 : tensor<256xf32>
    %arsums3b2W3 = "stablehlo.all_reduce"(%v4000) ({
    ^bb0(%aras3b2W3: tensor<f32>, %arbs3b2W3: tensor<f32>):
      %aradds3b2W3 = stablehlo.add %aras3b2W3, %arbs3b2W3 : tensor<f32>
      stablehlo.return %aradds3b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b2W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b2W3 = stablehlo.divide %arsums3b2W3, %arns3b2W3 : tensor<1024x256x1x1xf32>
    %v7535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7536 = stablehlo.multiply %v7535, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7537 = stablehlo.add %v7536, %armeans3b2W3 : tensor<1024x256x1x1xf32>
    %v7538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7539 = stablehlo.multiply %v7538, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7540 = stablehlo.add %v7539, %v7537 : tensor<1024x256x1x1xf32>
    %v7541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7542 = stablehlo.multiply %v7541, %v7540 : tensor<1024x256x1x1xf32>
    %v7543 = stablehlo.subtract %s3b2W3, %v7542 : tensor<1024x256x1x1xf32>
    %arsums3b2g3 = "stablehlo.all_reduce"(%v4014) ({
    ^bb0(%aras3b2g3: tensor<f32>, %arbs3b2g3: tensor<f32>):
      %aradds3b2g3 = stablehlo.add %aras3b2g3, %arbs3b2g3 : tensor<f32>
      stablehlo.return %aradds3b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3 = stablehlo.divide %arsums3b2g3, %arns3b2g3 : tensor<1024xf32>
    %v7544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7545 = stablehlo.multiply %v7544, %s3b2g3 : tensor<1024xf32>
    %v7546 = stablehlo.add %v7545, %armeans3b2g3 : tensor<1024xf32>
    %v7547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7548 = stablehlo.multiply %v7547, %s3b2g3v : tensor<1024xf32>
    %v7549 = stablehlo.add %v7548, %v7546 : tensor<1024xf32>
    %v7550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7551 = stablehlo.multiply %v7550, %v7549 : tensor<1024xf32>
    %v7552 = stablehlo.subtract %s3b2g3, %v7551 : tensor<1024xf32>
    %arsums3b2bt3 = "stablehlo.all_reduce"(%v4017) ({
    ^bb0(%aras3b2bt3: tensor<f32>, %arbs3b2bt3: tensor<f32>):
      %aradds3b2bt3 = stablehlo.add %aras3b2bt3, %arbs3b2bt3 : tensor<f32>
      stablehlo.return %aradds3b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2bt3 = stablehlo.divide %arsums3b2bt3, %arns3b2bt3 : tensor<1024xf32>
    %v7553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7554 = stablehlo.multiply %v7553, %s3b2bt3 : tensor<1024xf32>
    %v7555 = stablehlo.add %v7554, %armeans3b2bt3 : tensor<1024xf32>
    %v7556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7557 = stablehlo.multiply %v7556, %s3b2bt3v : tensor<1024xf32>
    %v7558 = stablehlo.add %v7557, %v7555 : tensor<1024xf32>
    %v7559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7560 = stablehlo.multiply %v7559, %v7558 : tensor<1024xf32>
    %v7561 = stablehlo.subtract %s3b2bt3, %v7560 : tensor<1024xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v3716) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x1024x1x1xf32>
    %v7562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7563 = stablehlo.multiply %v7562, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7564 = stablehlo.add %v7563, %armeans3b3W1 : tensor<256x1024x1x1xf32>
    %v7565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7566 = stablehlo.multiply %v7565, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7567 = stablehlo.add %v7566, %v7564 : tensor<256x1024x1x1xf32>
    %v7568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7569 = stablehlo.multiply %v7568, %v7567 : tensor<256x1024x1x1xf32>
    %v7570 = stablehlo.subtract %s3b3W1, %v7569 : tensor<256x1024x1x1xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v3730) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7572 = stablehlo.multiply %v7571, %s3b3g1 : tensor<256xf32>
    %v7573 = stablehlo.add %v7572, %armeans3b3g1 : tensor<256xf32>
    %v7574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7575 = stablehlo.multiply %v7574, %s3b3g1v : tensor<256xf32>
    %v7576 = stablehlo.add %v7575, %v7573 : tensor<256xf32>
    %v7577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7578 = stablehlo.multiply %v7577, %v7576 : tensor<256xf32>
    %v7579 = stablehlo.subtract %s3b3g1, %v7578 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v3733) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7581 = stablehlo.multiply %v7580, %s3b3bt1 : tensor<256xf32>
    %v7582 = stablehlo.add %v7581, %armeans3b3bt1 : tensor<256xf32>
    %v7583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7584 = stablehlo.multiply %v7583, %s3b3bt1v : tensor<256xf32>
    %v7585 = stablehlo.add %v7584, %v7582 : tensor<256xf32>
    %v7586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7587 = stablehlo.multiply %v7586, %v7585 : tensor<256xf32>
    %v7588 = stablehlo.subtract %s3b3bt1, %v7587 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v3739) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7590 = stablehlo.multiply %v7589, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7591 = stablehlo.add %v7590, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7592 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7593 = stablehlo.multiply %v7592, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7594 = stablehlo.add %v7593, %v7591 : tensor<256x256x3x3xf32>
    %v7595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7596 = stablehlo.multiply %v7595, %v7594 : tensor<256x256x3x3xf32>
    %v7597 = stablehlo.subtract %s3b3W2, %v7596 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v3753) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7599 = stablehlo.multiply %v7598, %s3b3g2 : tensor<256xf32>
    %v7600 = stablehlo.add %v7599, %armeans3b3g2 : tensor<256xf32>
    %v7601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7602 = stablehlo.multiply %v7601, %s3b3g2v : tensor<256xf32>
    %v7603 = stablehlo.add %v7602, %v7600 : tensor<256xf32>
    %v7604 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7605 = stablehlo.multiply %v7604, %v7603 : tensor<256xf32>
    %v7606 = stablehlo.subtract %s3b3g2, %v7605 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v3756) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7608 = stablehlo.multiply %v7607, %s3b3bt2 : tensor<256xf32>
    %v7609 = stablehlo.add %v7608, %armeans3b3bt2 : tensor<256xf32>
    %v7610 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7611 = stablehlo.multiply %v7610, %s3b3bt2v : tensor<256xf32>
    %v7612 = stablehlo.add %v7611, %v7609 : tensor<256xf32>
    %v7613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7614 = stablehlo.multiply %v7613, %v7612 : tensor<256xf32>
    %v7615 = stablehlo.subtract %s3b3bt2, %v7614 : tensor<256xf32>
    %arsums3b3W3 = "stablehlo.all_reduce"(%v3762) ({
    ^bb0(%aras3b3W3: tensor<f32>, %arbs3b3W3: tensor<f32>):
      %aradds3b3W3 = stablehlo.add %aras3b3W3, %arbs3b3W3 : tensor<f32>
      stablehlo.return %aradds3b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b3W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b3W3 = stablehlo.divide %arsums3b3W3, %arns3b3W3 : tensor<1024x256x1x1xf32>
    %v7616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7617 = stablehlo.multiply %v7616, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v7618 = stablehlo.add %v7617, %armeans3b3W3 : tensor<1024x256x1x1xf32>
    %v7619 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7620 = stablehlo.multiply %v7619, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v7621 = stablehlo.add %v7620, %v7618 : tensor<1024x256x1x1xf32>
    %v7622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7623 = stablehlo.multiply %v7622, %v7621 : tensor<1024x256x1x1xf32>
    %v7624 = stablehlo.subtract %s3b3W3, %v7623 : tensor<1024x256x1x1xf32>
    %arsums3b3g3 = "stablehlo.all_reduce"(%v3776) ({
    ^bb0(%aras3b3g3: tensor<f32>, %arbs3b3g3: tensor<f32>):
      %aradds3b3g3 = stablehlo.add %aras3b3g3, %arbs3b3g3 : tensor<f32>
      stablehlo.return %aradds3b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3 = stablehlo.divide %arsums3b3g3, %arns3b3g3 : tensor<1024xf32>
    %v7625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7626 = stablehlo.multiply %v7625, %s3b3g3 : tensor<1024xf32>
    %v7627 = stablehlo.add %v7626, %armeans3b3g3 : tensor<1024xf32>
    %v7628 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7629 = stablehlo.multiply %v7628, %s3b3g3v : tensor<1024xf32>
    %v7630 = stablehlo.add %v7629, %v7627 : tensor<1024xf32>
    %v7631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7632 = stablehlo.multiply %v7631, %v7630 : tensor<1024xf32>
    %v7633 = stablehlo.subtract %s3b3g3, %v7632 : tensor<1024xf32>
    %arsums3b3bt3 = "stablehlo.all_reduce"(%v3779) ({
    ^bb0(%aras3b3bt3: tensor<f32>, %arbs3b3bt3: tensor<f32>):
      %aradds3b3bt3 = stablehlo.add %aras3b3bt3, %arbs3b3bt3 : tensor<f32>
      stablehlo.return %aradds3b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3bt3 = stablehlo.divide %arsums3b3bt3, %arns3b3bt3 : tensor<1024xf32>
    %v7634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7635 = stablehlo.multiply %v7634, %s3b3bt3 : tensor<1024xf32>
    %v7636 = stablehlo.add %v7635, %armeans3b3bt3 : tensor<1024xf32>
    %v7637 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7638 = stablehlo.multiply %v7637, %s3b3bt3v : tensor<1024xf32>
    %v7639 = stablehlo.add %v7638, %v7636 : tensor<1024xf32>
    %v7640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7641 = stablehlo.multiply %v7640, %v7639 : tensor<1024xf32>
    %v7642 = stablehlo.subtract %s3b3bt3, %v7641 : tensor<1024xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v3478) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x1024x1x1xf32>
    %v7643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7644 = stablehlo.multiply %v7643, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v7645 = stablehlo.add %v7644, %armeans3b4W1 : tensor<256x1024x1x1xf32>
    %v7646 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7647 = stablehlo.multiply %v7646, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v7648 = stablehlo.add %v7647, %v7645 : tensor<256x1024x1x1xf32>
    %v7649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7650 = stablehlo.multiply %v7649, %v7648 : tensor<256x1024x1x1xf32>
    %v7651 = stablehlo.subtract %s3b4W1, %v7650 : tensor<256x1024x1x1xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v3492) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7653 = stablehlo.multiply %v7652, %s3b4g1 : tensor<256xf32>
    %v7654 = stablehlo.add %v7653, %armeans3b4g1 : tensor<256xf32>
    %v7655 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7656 = stablehlo.multiply %v7655, %s3b4g1v : tensor<256xf32>
    %v7657 = stablehlo.add %v7656, %v7654 : tensor<256xf32>
    %v7658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7659 = stablehlo.multiply %v7658, %v7657 : tensor<256xf32>
    %v7660 = stablehlo.subtract %s3b4g1, %v7659 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v3495) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7662 = stablehlo.multiply %v7661, %s3b4bt1 : tensor<256xf32>
    %v7663 = stablehlo.add %v7662, %armeans3b4bt1 : tensor<256xf32>
    %v7664 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7665 = stablehlo.multiply %v7664, %s3b4bt1v : tensor<256xf32>
    %v7666 = stablehlo.add %v7665, %v7663 : tensor<256xf32>
    %v7667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7668 = stablehlo.multiply %v7667, %v7666 : tensor<256xf32>
    %v7669 = stablehlo.subtract %s3b4bt1, %v7668 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v3501) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7671 = stablehlo.multiply %v7670, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7672 = stablehlo.add %v7671, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7673 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7674 = stablehlo.multiply %v7673, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7675 = stablehlo.add %v7674, %v7672 : tensor<256x256x3x3xf32>
    %v7676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7677 = stablehlo.multiply %v7676, %v7675 : tensor<256x256x3x3xf32>
    %v7678 = stablehlo.subtract %s3b4W2, %v7677 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v3515) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7680 = stablehlo.multiply %v7679, %s3b4g2 : tensor<256xf32>
    %v7681 = stablehlo.add %v7680, %armeans3b4g2 : tensor<256xf32>
    %v7682 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7683 = stablehlo.multiply %v7682, %s3b4g2v : tensor<256xf32>
    %v7684 = stablehlo.add %v7683, %v7681 : tensor<256xf32>
    %v7685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7686 = stablehlo.multiply %v7685, %v7684 : tensor<256xf32>
    %v7687 = stablehlo.subtract %s3b4g2, %v7686 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v3518) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7689 = stablehlo.multiply %v7688, %s3b4bt2 : tensor<256xf32>
    %v7690 = stablehlo.add %v7689, %armeans3b4bt2 : tensor<256xf32>
    %v7691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7692 = stablehlo.multiply %v7691, %s3b4bt2v : tensor<256xf32>
    %v7693 = stablehlo.add %v7692, %v7690 : tensor<256xf32>
    %v7694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7695 = stablehlo.multiply %v7694, %v7693 : tensor<256xf32>
    %v7696 = stablehlo.subtract %s3b4bt2, %v7695 : tensor<256xf32>
    %arsums3b4W3 = "stablehlo.all_reduce"(%v3524) ({
    ^bb0(%aras3b4W3: tensor<f32>, %arbs3b4W3: tensor<f32>):
      %aradds3b4W3 = stablehlo.add %aras3b4W3, %arbs3b4W3 : tensor<f32>
      stablehlo.return %aradds3b4W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b4W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b4W3 = stablehlo.divide %arsums3b4W3, %arns3b4W3 : tensor<1024x256x1x1xf32>
    %v7697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7698 = stablehlo.multiply %v7697, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v7699 = stablehlo.add %v7698, %armeans3b4W3 : tensor<1024x256x1x1xf32>
    %v7700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7701 = stablehlo.multiply %v7700, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v7702 = stablehlo.add %v7701, %v7699 : tensor<1024x256x1x1xf32>
    %v7703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7704 = stablehlo.multiply %v7703, %v7702 : tensor<1024x256x1x1xf32>
    %v7705 = stablehlo.subtract %s3b4W3, %v7704 : tensor<1024x256x1x1xf32>
    %arsums3b4g3 = "stablehlo.all_reduce"(%v3538) ({
    ^bb0(%aras3b4g3: tensor<f32>, %arbs3b4g3: tensor<f32>):
      %aradds3b4g3 = stablehlo.add %aras3b4g3, %arbs3b4g3 : tensor<f32>
      stablehlo.return %aradds3b4g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3 = stablehlo.divide %arsums3b4g3, %arns3b4g3 : tensor<1024xf32>
    %v7706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7707 = stablehlo.multiply %v7706, %s3b4g3 : tensor<1024xf32>
    %v7708 = stablehlo.add %v7707, %armeans3b4g3 : tensor<1024xf32>
    %v7709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7710 = stablehlo.multiply %v7709, %s3b4g3v : tensor<1024xf32>
    %v7711 = stablehlo.add %v7710, %v7708 : tensor<1024xf32>
    %v7712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7713 = stablehlo.multiply %v7712, %v7711 : tensor<1024xf32>
    %v7714 = stablehlo.subtract %s3b4g3, %v7713 : tensor<1024xf32>
    %arsums3b4bt3 = "stablehlo.all_reduce"(%v3541) ({
    ^bb0(%aras3b4bt3: tensor<f32>, %arbs3b4bt3: tensor<f32>):
      %aradds3b4bt3 = stablehlo.add %aras3b4bt3, %arbs3b4bt3 : tensor<f32>
      stablehlo.return %aradds3b4bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4bt3 = stablehlo.divide %arsums3b4bt3, %arns3b4bt3 : tensor<1024xf32>
    %v7715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7716 = stablehlo.multiply %v7715, %s3b4bt3 : tensor<1024xf32>
    %v7717 = stablehlo.add %v7716, %armeans3b4bt3 : tensor<1024xf32>
    %v7718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7719 = stablehlo.multiply %v7718, %s3b4bt3v : tensor<1024xf32>
    %v7720 = stablehlo.add %v7719, %v7717 : tensor<1024xf32>
    %v7721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7722 = stablehlo.multiply %v7721, %v7720 : tensor<1024xf32>
    %v7723 = stablehlo.subtract %s3b4bt3, %v7722 : tensor<1024xf32>
    %arsums3b5W1 = "stablehlo.all_reduce"(%v3240) ({
    ^bb0(%aras3b5W1: tensor<f32>, %arbs3b5W1: tensor<f32>):
      %aradds3b5W1 = stablehlo.add %aras3b5W1, %arbs3b5W1 : tensor<f32>
      stablehlo.return %aradds3b5W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b5W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b5W1 = stablehlo.divide %arsums3b5W1, %arns3b5W1 : tensor<256x1024x1x1xf32>
    %v7724 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7725 = stablehlo.multiply %v7724, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v7726 = stablehlo.add %v7725, %armeans3b5W1 : tensor<256x1024x1x1xf32>
    %v7727 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7728 = stablehlo.multiply %v7727, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v7729 = stablehlo.add %v7728, %v7726 : tensor<256x1024x1x1xf32>
    %v7730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7731 = stablehlo.multiply %v7730, %v7729 : tensor<256x1024x1x1xf32>
    %v7732 = stablehlo.subtract %s3b5W1, %v7731 : tensor<256x1024x1x1xf32>
    %arsums3b5g1 = "stablehlo.all_reduce"(%v3254) ({
    ^bb0(%aras3b5g1: tensor<f32>, %arbs3b5g1: tensor<f32>):
      %aradds3b5g1 = stablehlo.add %aras3b5g1, %arbs3b5g1 : tensor<f32>
      stablehlo.return %aradds3b5g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1 = stablehlo.divide %arsums3b5g1, %arns3b5g1 : tensor<256xf32>
    %v7733 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7734 = stablehlo.multiply %v7733, %s3b5g1 : tensor<256xf32>
    %v7735 = stablehlo.add %v7734, %armeans3b5g1 : tensor<256xf32>
    %v7736 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7737 = stablehlo.multiply %v7736, %s3b5g1v : tensor<256xf32>
    %v7738 = stablehlo.add %v7737, %v7735 : tensor<256xf32>
    %v7739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7740 = stablehlo.multiply %v7739, %v7738 : tensor<256xf32>
    %v7741 = stablehlo.subtract %s3b5g1, %v7740 : tensor<256xf32>
    %arsums3b5bt1 = "stablehlo.all_reduce"(%v3257) ({
    ^bb0(%aras3b5bt1: tensor<f32>, %arbs3b5bt1: tensor<f32>):
      %aradds3b5bt1 = stablehlo.add %aras3b5bt1, %arbs3b5bt1 : tensor<f32>
      stablehlo.return %aradds3b5bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt1 = stablehlo.divide %arsums3b5bt1, %arns3b5bt1 : tensor<256xf32>
    %v7742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7743 = stablehlo.multiply %v7742, %s3b5bt1 : tensor<256xf32>
    %v7744 = stablehlo.add %v7743, %armeans3b5bt1 : tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.multiply %v7745, %s3b5bt1v : tensor<256xf32>
    %v7747 = stablehlo.add %v7746, %v7744 : tensor<256xf32>
    %v7748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7749 = stablehlo.multiply %v7748, %v7747 : tensor<256xf32>
    %v7750 = stablehlo.subtract %s3b5bt1, %v7749 : tensor<256xf32>
    %arsums3b5W2 = "stablehlo.all_reduce"(%v3263) ({
    ^bb0(%aras3b5W2: tensor<f32>, %arbs3b5W2: tensor<f32>):
      %aradds3b5W2 = stablehlo.add %aras3b5W2, %arbs3b5W2 : tensor<f32>
      stablehlo.return %aradds3b5W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b5W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b5W2 = stablehlo.divide %arsums3b5W2, %arns3b5W2 : tensor<256x256x3x3xf32>
    %v7751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7752 = stablehlo.multiply %v7751, %s3b5W2 : tensor<256x256x3x3xf32>
    %v7753 = stablehlo.add %v7752, %armeans3b5W2 : tensor<256x256x3x3xf32>
    %v7754 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7755 = stablehlo.multiply %v7754, %s3b5W2v : tensor<256x256x3x3xf32>
    %v7756 = stablehlo.add %v7755, %v7753 : tensor<256x256x3x3xf32>
    %v7757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7758 = stablehlo.multiply %v7757, %v7756 : tensor<256x256x3x3xf32>
    %v7759 = stablehlo.subtract %s3b5W2, %v7758 : tensor<256x256x3x3xf32>
    %arsums3b5g2 = "stablehlo.all_reduce"(%v3277) ({
    ^bb0(%aras3b5g2: tensor<f32>, %arbs3b5g2: tensor<f32>):
      %aradds3b5g2 = stablehlo.add %aras3b5g2, %arbs3b5g2 : tensor<f32>
      stablehlo.return %aradds3b5g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2 = stablehlo.divide %arsums3b5g2, %arns3b5g2 : tensor<256xf32>
    %v7760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7761 = stablehlo.multiply %v7760, %s3b5g2 : tensor<256xf32>
    %v7762 = stablehlo.add %v7761, %armeans3b5g2 : tensor<256xf32>
    %v7763 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7764 = stablehlo.multiply %v7763, %s3b5g2v : tensor<256xf32>
    %v7765 = stablehlo.add %v7764, %v7762 : tensor<256xf32>
    %v7766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7767 = stablehlo.multiply %v7766, %v7765 : tensor<256xf32>
    %v7768 = stablehlo.subtract %s3b5g2, %v7767 : tensor<256xf32>
    %arsums3b5bt2 = "stablehlo.all_reduce"(%v3280) ({
    ^bb0(%aras3b5bt2: tensor<f32>, %arbs3b5bt2: tensor<f32>):
      %aradds3b5bt2 = stablehlo.add %aras3b5bt2, %arbs3b5bt2 : tensor<f32>
      stablehlo.return %aradds3b5bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt2 = stablehlo.divide %arsums3b5bt2, %arns3b5bt2 : tensor<256xf32>
    %v7769 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7770 = stablehlo.multiply %v7769, %s3b5bt2 : tensor<256xf32>
    %v7771 = stablehlo.add %v7770, %armeans3b5bt2 : tensor<256xf32>
    %v7772 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7773 = stablehlo.multiply %v7772, %s3b5bt2v : tensor<256xf32>
    %v7774 = stablehlo.add %v7773, %v7771 : tensor<256xf32>
    %v7775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7776 = stablehlo.multiply %v7775, %v7774 : tensor<256xf32>
    %v7777 = stablehlo.subtract %s3b5bt2, %v7776 : tensor<256xf32>
    %arsums3b5W3 = "stablehlo.all_reduce"(%v3286) ({
    ^bb0(%aras3b5W3: tensor<f32>, %arbs3b5W3: tensor<f32>):
      %aradds3b5W3 = stablehlo.add %aras3b5W3, %arbs3b5W3 : tensor<f32>
      stablehlo.return %aradds3b5W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b5W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b5W3 = stablehlo.divide %arsums3b5W3, %arns3b5W3 : tensor<1024x256x1x1xf32>
    %v7778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7779 = stablehlo.multiply %v7778, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v7780 = stablehlo.add %v7779, %armeans3b5W3 : tensor<1024x256x1x1xf32>
    %v7781 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7782 = stablehlo.multiply %v7781, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v7783 = stablehlo.add %v7782, %v7780 : tensor<1024x256x1x1xf32>
    %v7784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7785 = stablehlo.multiply %v7784, %v7783 : tensor<1024x256x1x1xf32>
    %v7786 = stablehlo.subtract %s3b5W3, %v7785 : tensor<1024x256x1x1xf32>
    %arsums3b5g3 = "stablehlo.all_reduce"(%v3300) ({
    ^bb0(%aras3b5g3: tensor<f32>, %arbs3b5g3: tensor<f32>):
      %aradds3b5g3 = stablehlo.add %aras3b5g3, %arbs3b5g3 : tensor<f32>
      stablehlo.return %aradds3b5g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3 = stablehlo.divide %arsums3b5g3, %arns3b5g3 : tensor<1024xf32>
    %v7787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7788 = stablehlo.multiply %v7787, %s3b5g3 : tensor<1024xf32>
    %v7789 = stablehlo.add %v7788, %armeans3b5g3 : tensor<1024xf32>
    %v7790 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7791 = stablehlo.multiply %v7790, %s3b5g3v : tensor<1024xf32>
    %v7792 = stablehlo.add %v7791, %v7789 : tensor<1024xf32>
    %v7793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7794 = stablehlo.multiply %v7793, %v7792 : tensor<1024xf32>
    %v7795 = stablehlo.subtract %s3b5g3, %v7794 : tensor<1024xf32>
    %arsums3b5bt3 = "stablehlo.all_reduce"(%v3303) ({
    ^bb0(%aras3b5bt3: tensor<f32>, %arbs3b5bt3: tensor<f32>):
      %aradds3b5bt3 = stablehlo.add %aras3b5bt3, %arbs3b5bt3 : tensor<f32>
      stablehlo.return %aradds3b5bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5bt3 = stablehlo.divide %arsums3b5bt3, %arns3b5bt3 : tensor<1024xf32>
    %v7796 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7797 = stablehlo.multiply %v7796, %s3b5bt3 : tensor<1024xf32>
    %v7798 = stablehlo.add %v7797, %armeans3b5bt3 : tensor<1024xf32>
    %v7799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7800 = stablehlo.multiply %v7799, %s3b5bt3v : tensor<1024xf32>
    %v7801 = stablehlo.add %v7800, %v7798 : tensor<1024xf32>
    %v7802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7803 = stablehlo.multiply %v7802, %v7801 : tensor<1024xf32>
    %v7804 = stablehlo.subtract %s3b5bt3, %v7803 : tensor<1024xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v2977) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x1024x1x1xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x1024x1x1xf32>
    %v7805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7806 = stablehlo.multiply %v7805, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v7807 = stablehlo.add %v7806, %armeans4b0W1 : tensor<512x1024x1x1xf32>
    %v7808 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7809 = stablehlo.multiply %v7808, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v7810 = stablehlo.add %v7809, %v7807 : tensor<512x1024x1x1xf32>
    %v7811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7812 = stablehlo.multiply %v7811, %v7810 : tensor<512x1024x1x1xf32>
    %v7813 = stablehlo.subtract %s4b0W1, %v7812 : tensor<512x1024x1x1xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v2991) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v7814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7815 = stablehlo.multiply %v7814, %s4b0g1 : tensor<512xf32>
    %v7816 = stablehlo.add %v7815, %armeans4b0g1 : tensor<512xf32>
    %v7817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7818 = stablehlo.multiply %v7817, %s4b0g1v : tensor<512xf32>
    %v7819 = stablehlo.add %v7818, %v7816 : tensor<512xf32>
    %v7820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7821 = stablehlo.multiply %v7820, %v7819 : tensor<512xf32>
    %v7822 = stablehlo.subtract %s4b0g1, %v7821 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v2994) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v7823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7824 = stablehlo.multiply %v7823, %s4b0bt1 : tensor<512xf32>
    %v7825 = stablehlo.add %v7824, %armeans4b0bt1 : tensor<512xf32>
    %v7826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7827 = stablehlo.multiply %v7826, %s4b0bt1v : tensor<512xf32>
    %v7828 = stablehlo.add %v7827, %v7825 : tensor<512xf32>
    %v7829 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7830 = stablehlo.multiply %v7829, %v7828 : tensor<512xf32>
    %v7831 = stablehlo.subtract %s4b0bt1, %v7830 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v3001) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v7832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7833 = stablehlo.multiply %v7832, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7834 = stablehlo.add %v7833, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7835 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7836 = stablehlo.multiply %v7835, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7837 = stablehlo.add %v7836, %v7834 : tensor<512x512x3x3xf32>
    %v7838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7839 = stablehlo.multiply %v7838, %v7837 : tensor<512x512x3x3xf32>
    %v7840 = stablehlo.subtract %s4b0W2, %v7839 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v3015) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v7841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7842 = stablehlo.multiply %v7841, %s4b0g2 : tensor<512xf32>
    %v7843 = stablehlo.add %v7842, %armeans4b0g2 : tensor<512xf32>
    %v7844 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7845 = stablehlo.multiply %v7844, %s4b0g2v : tensor<512xf32>
    %v7846 = stablehlo.add %v7845, %v7843 : tensor<512xf32>
    %v7847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7848 = stablehlo.multiply %v7847, %v7846 : tensor<512xf32>
    %v7849 = stablehlo.subtract %s4b0g2, %v7848 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v3018) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v7850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7851 = stablehlo.multiply %v7850, %s4b0bt2 : tensor<512xf32>
    %v7852 = stablehlo.add %v7851, %armeans4b0bt2 : tensor<512xf32>
    %v7853 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7854 = stablehlo.multiply %v7853, %s4b0bt2v : tensor<512xf32>
    %v7855 = stablehlo.add %v7854, %v7852 : tensor<512xf32>
    %v7856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7857 = stablehlo.multiply %v7856, %v7855 : tensor<512xf32>
    %v7858 = stablehlo.subtract %s4b0bt2, %v7857 : tensor<512xf32>
    %arsums4b0W3 = "stablehlo.all_reduce"(%v3024) ({
    ^bb0(%aras4b0W3: tensor<f32>, %arbs4b0W3: tensor<f32>):
      %aradds4b0W3 = stablehlo.add %aras4b0W3, %arbs4b0W3 : tensor<f32>
      stablehlo.return %aradds4b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b0W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b0W3 = stablehlo.divide %arsums4b0W3, %arns4b0W3 : tensor<2048x512x1x1xf32>
    %v7859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7860 = stablehlo.multiply %v7859, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7861 = stablehlo.add %v7860, %armeans4b0W3 : tensor<2048x512x1x1xf32>
    %v7862 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7863 = stablehlo.multiply %v7862, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7864 = stablehlo.add %v7863, %v7861 : tensor<2048x512x1x1xf32>
    %v7865 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7866 = stablehlo.multiply %v7865, %v7864 : tensor<2048x512x1x1xf32>
    %v7867 = stablehlo.subtract %s4b0W3, %v7866 : tensor<2048x512x1x1xf32>
    %arsums4b0g3 = "stablehlo.all_reduce"(%v3038) ({
    ^bb0(%aras4b0g3: tensor<f32>, %arbs4b0g3: tensor<f32>):
      %aradds4b0g3 = stablehlo.add %aras4b0g3, %arbs4b0g3 : tensor<f32>
      stablehlo.return %aradds4b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3 = stablehlo.divide %arsums4b0g3, %arns4b0g3 : tensor<2048xf32>
    %v7868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7869 = stablehlo.multiply %v7868, %s4b0g3 : tensor<2048xf32>
    %v7870 = stablehlo.add %v7869, %armeans4b0g3 : tensor<2048xf32>
    %v7871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7872 = stablehlo.multiply %v7871, %s4b0g3v : tensor<2048xf32>
    %v7873 = stablehlo.add %v7872, %v7870 : tensor<2048xf32>
    %v7874 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7875 = stablehlo.multiply %v7874, %v7873 : tensor<2048xf32>
    %v7876 = stablehlo.subtract %s4b0g3, %v7875 : tensor<2048xf32>
    %arsums4b0bt3 = "stablehlo.all_reduce"(%v3041) ({
    ^bb0(%aras4b0bt3: tensor<f32>, %arbs4b0bt3: tensor<f32>):
      %aradds4b0bt3 = stablehlo.add %aras4b0bt3, %arbs4b0bt3 : tensor<f32>
      stablehlo.return %aradds4b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0bt3 = stablehlo.divide %arsums4b0bt3, %arns4b0bt3 : tensor<2048xf32>
    %v7877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7878 = stablehlo.multiply %v7877, %s4b0bt3 : tensor<2048xf32>
    %v7879 = stablehlo.add %v7878, %armeans4b0bt3 : tensor<2048xf32>
    %v7880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7881 = stablehlo.multiply %v7880, %s4b0bt3v : tensor<2048xf32>
    %v7882 = stablehlo.add %v7881, %v7879 : tensor<2048xf32>
    %v7883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7884 = stablehlo.multiply %v7883, %v7882 : tensor<2048xf32>
    %v7885 = stablehlo.subtract %s4b0bt3, %v7884 : tensor<2048xf32>
    %arsums4b0Wp = "stablehlo.all_reduce"(%v3048) ({
    ^bb0(%aras4b0Wp: tensor<f32>, %arbs4b0Wp: tensor<f32>):
      %aradds4b0Wp = stablehlo.add %aras4b0Wp, %arbs4b0Wp : tensor<f32>
      stablehlo.return %aradds4b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %arns4b0Wp = stablehlo.constant dense<4.0> : tensor<2048x1024x1x1xf32>
    %armeans4b0Wp = stablehlo.divide %arsums4b0Wp, %arns4b0Wp : tensor<2048x1024x1x1xf32>
    %v7886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7887 = stablehlo.multiply %v7886, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7888 = stablehlo.add %v7887, %armeans4b0Wp : tensor<2048x1024x1x1xf32>
    %v7889 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7890 = stablehlo.multiply %v7889, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7891 = stablehlo.add %v7890, %v7888 : tensor<2048x1024x1x1xf32>
    %v7892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7893 = stablehlo.multiply %v7892, %v7891 : tensor<2048x1024x1x1xf32>
    %v7894 = stablehlo.subtract %s4b0Wp, %v7893 : tensor<2048x1024x1x1xf32>
    %arsums4b0gp = "stablehlo.all_reduce"(%v3062) ({
    ^bb0(%aras4b0gp: tensor<f32>, %arbs4b0gp: tensor<f32>):
      %aradds4b0gp = stablehlo.add %aras4b0gp, %arbs4b0gp : tensor<f32>
      stablehlo.return %aradds4b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gp = stablehlo.divide %arsums4b0gp, %arns4b0gp : tensor<2048xf32>
    %v7895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7896 = stablehlo.multiply %v7895, %s4b0gp : tensor<2048xf32>
    %v7897 = stablehlo.add %v7896, %armeans4b0gp : tensor<2048xf32>
    %v7898 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7899 = stablehlo.multiply %v7898, %s4b0gpv : tensor<2048xf32>
    %v7900 = stablehlo.add %v7899, %v7897 : tensor<2048xf32>
    %v7901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7902 = stablehlo.multiply %v7901, %v7900 : tensor<2048xf32>
    %v7903 = stablehlo.subtract %s4b0gp, %v7902 : tensor<2048xf32>
    %arsums4b0btp = "stablehlo.all_reduce"(%v3065) ({
    ^bb0(%aras4b0btp: tensor<f32>, %arbs4b0btp: tensor<f32>):
      %aradds4b0btp = stablehlo.add %aras4b0btp, %arbs4b0btp : tensor<f32>
      stablehlo.return %aradds4b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0btp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0btp = stablehlo.divide %arsums4b0btp, %arns4b0btp : tensor<2048xf32>
    %v7904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7905 = stablehlo.multiply %v7904, %s4b0btp : tensor<2048xf32>
    %v7906 = stablehlo.add %v7905, %armeans4b0btp : tensor<2048xf32>
    %v7907 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7908 = stablehlo.multiply %v7907, %s4b0btpv : tensor<2048xf32>
    %v7909 = stablehlo.add %v7908, %v7906 : tensor<2048xf32>
    %v7910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7911 = stablehlo.multiply %v7910, %v7909 : tensor<2048xf32>
    %v7912 = stablehlo.subtract %s4b0btp, %v7911 : tensor<2048xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v2690) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x2048x1x1xf32>
    %v7913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7914 = stablehlo.multiply %v7913, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7915 = stablehlo.add %v7914, %armeans4b1W1 : tensor<512x2048x1x1xf32>
    %v7916 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7917 = stablehlo.multiply %v7916, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7918 = stablehlo.add %v7917, %v7915 : tensor<512x2048x1x1xf32>
    %v7919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7920 = stablehlo.multiply %v7919, %v7918 : tensor<512x2048x1x1xf32>
    %v7921 = stablehlo.subtract %s4b1W1, %v7920 : tensor<512x2048x1x1xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v2704) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v7922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7923 = stablehlo.multiply %v7922, %s4b1g1 : tensor<512xf32>
    %v7924 = stablehlo.add %v7923, %armeans4b1g1 : tensor<512xf32>
    %v7925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7926 = stablehlo.multiply %v7925, %s4b1g1v : tensor<512xf32>
    %v7927 = stablehlo.add %v7926, %v7924 : tensor<512xf32>
    %v7928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7929 = stablehlo.multiply %v7928, %v7927 : tensor<512xf32>
    %v7930 = stablehlo.subtract %s4b1g1, %v7929 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v2707) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v7931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7932 = stablehlo.multiply %v7931, %s4b1bt1 : tensor<512xf32>
    %v7933 = stablehlo.add %v7932, %armeans4b1bt1 : tensor<512xf32>
    %v7934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7935 = stablehlo.multiply %v7934, %s4b1bt1v : tensor<512xf32>
    %v7936 = stablehlo.add %v7935, %v7933 : tensor<512xf32>
    %v7937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7938 = stablehlo.multiply %v7937, %v7936 : tensor<512xf32>
    %v7939 = stablehlo.subtract %s4b1bt1, %v7938 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v2713) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v7940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7941 = stablehlo.multiply %v7940, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7942 = stablehlo.add %v7941, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7943 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7944 = stablehlo.multiply %v7943, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7945 = stablehlo.add %v7944, %v7942 : tensor<512x512x3x3xf32>
    %v7946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7947 = stablehlo.multiply %v7946, %v7945 : tensor<512x512x3x3xf32>
    %v7948 = stablehlo.subtract %s4b1W2, %v7947 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v2727) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v7949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7950 = stablehlo.multiply %v7949, %s4b1g2 : tensor<512xf32>
    %v7951 = stablehlo.add %v7950, %armeans4b1g2 : tensor<512xf32>
    %v7952 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7953 = stablehlo.multiply %v7952, %s4b1g2v : tensor<512xf32>
    %v7954 = stablehlo.add %v7953, %v7951 : tensor<512xf32>
    %v7955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7956 = stablehlo.multiply %v7955, %v7954 : tensor<512xf32>
    %v7957 = stablehlo.subtract %s4b1g2, %v7956 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v2730) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v7958 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7959 = stablehlo.multiply %v7958, %s4b1bt2 : tensor<512xf32>
    %v7960 = stablehlo.add %v7959, %armeans4b1bt2 : tensor<512xf32>
    %v7961 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7962 = stablehlo.multiply %v7961, %s4b1bt2v : tensor<512xf32>
    %v7963 = stablehlo.add %v7962, %v7960 : tensor<512xf32>
    %v7964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7965 = stablehlo.multiply %v7964, %v7963 : tensor<512xf32>
    %v7966 = stablehlo.subtract %s4b1bt2, %v7965 : tensor<512xf32>
    %arsums4b1W3 = "stablehlo.all_reduce"(%v2736) ({
    ^bb0(%aras4b1W3: tensor<f32>, %arbs4b1W3: tensor<f32>):
      %aradds4b1W3 = stablehlo.add %aras4b1W3, %arbs4b1W3 : tensor<f32>
      stablehlo.return %aradds4b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b1W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b1W3 = stablehlo.divide %arsums4b1W3, %arns4b1W3 : tensor<2048x512x1x1xf32>
    %v7967 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7968 = stablehlo.multiply %v7967, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7969 = stablehlo.add %v7968, %armeans4b1W3 : tensor<2048x512x1x1xf32>
    %v7970 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7971 = stablehlo.multiply %v7970, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7972 = stablehlo.add %v7971, %v7969 : tensor<2048x512x1x1xf32>
    %v7973 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7974 = stablehlo.multiply %v7973, %v7972 : tensor<2048x512x1x1xf32>
    %v7975 = stablehlo.subtract %s4b1W3, %v7974 : tensor<2048x512x1x1xf32>
    %arsums4b1g3 = "stablehlo.all_reduce"(%v2750) ({
    ^bb0(%aras4b1g3: tensor<f32>, %arbs4b1g3: tensor<f32>):
      %aradds4b1g3 = stablehlo.add %aras4b1g3, %arbs4b1g3 : tensor<f32>
      stablehlo.return %aradds4b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3 = stablehlo.divide %arsums4b1g3, %arns4b1g3 : tensor<2048xf32>
    %v7976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7977 = stablehlo.multiply %v7976, %s4b1g3 : tensor<2048xf32>
    %v7978 = stablehlo.add %v7977, %armeans4b1g3 : tensor<2048xf32>
    %v7979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7980 = stablehlo.multiply %v7979, %s4b1g3v : tensor<2048xf32>
    %v7981 = stablehlo.add %v7980, %v7978 : tensor<2048xf32>
    %v7982 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7983 = stablehlo.multiply %v7982, %v7981 : tensor<2048xf32>
    %v7984 = stablehlo.subtract %s4b1g3, %v7983 : tensor<2048xf32>
    %arsums4b1bt3 = "stablehlo.all_reduce"(%v2753) ({
    ^bb0(%aras4b1bt3: tensor<f32>, %arbs4b1bt3: tensor<f32>):
      %aradds4b1bt3 = stablehlo.add %aras4b1bt3, %arbs4b1bt3 : tensor<f32>
      stablehlo.return %aradds4b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1bt3 = stablehlo.divide %arsums4b1bt3, %arns4b1bt3 : tensor<2048xf32>
    %v7985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7986 = stablehlo.multiply %v7985, %s4b1bt3 : tensor<2048xf32>
    %v7987 = stablehlo.add %v7986, %armeans4b1bt3 : tensor<2048xf32>
    %v7988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7989 = stablehlo.multiply %v7988, %s4b1bt3v : tensor<2048xf32>
    %v7990 = stablehlo.add %v7989, %v7987 : tensor<2048xf32>
    %v7991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7992 = stablehlo.multiply %v7991, %v7990 : tensor<2048xf32>
    %v7993 = stablehlo.subtract %s4b1bt3, %v7992 : tensor<2048xf32>
    %arsums4b2W1 = "stablehlo.all_reduce"(%v2452) ({
    ^bb0(%aras4b2W1: tensor<f32>, %arbs4b2W1: tensor<f32>):
      %aradds4b2W1 = stablehlo.add %aras4b2W1, %arbs4b2W1 : tensor<f32>
      stablehlo.return %aradds4b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b2W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b2W1 = stablehlo.divide %arsums4b2W1, %arns4b2W1 : tensor<512x2048x1x1xf32>
    %v7994 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7995 = stablehlo.multiply %v7994, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7996 = stablehlo.add %v7995, %armeans4b2W1 : tensor<512x2048x1x1xf32>
    %v7997 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7998 = stablehlo.multiply %v7997, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7999 = stablehlo.add %v7998, %v7996 : tensor<512x2048x1x1xf32>
    %v8000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8001 = stablehlo.multiply %v8000, %v7999 : tensor<512x2048x1x1xf32>
    %v8002 = stablehlo.subtract %s4b2W1, %v8001 : tensor<512x2048x1x1xf32>
    %arsums4b2g1 = "stablehlo.all_reduce"(%v2466) ({
    ^bb0(%aras4b2g1: tensor<f32>, %arbs4b2g1: tensor<f32>):
      %aradds4b2g1 = stablehlo.add %aras4b2g1, %arbs4b2g1 : tensor<f32>
      stablehlo.return %aradds4b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1 = stablehlo.divide %arsums4b2g1, %arns4b2g1 : tensor<512xf32>
    %v8003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8004 = stablehlo.multiply %v8003, %s4b2g1 : tensor<512xf32>
    %v8005 = stablehlo.add %v8004, %armeans4b2g1 : tensor<512xf32>
    %v8006 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8007 = stablehlo.multiply %v8006, %s4b2g1v : tensor<512xf32>
    %v8008 = stablehlo.add %v8007, %v8005 : tensor<512xf32>
    %v8009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8010 = stablehlo.multiply %v8009, %v8008 : tensor<512xf32>
    %v8011 = stablehlo.subtract %s4b2g1, %v8010 : tensor<512xf32>
    %arsums4b2bt1 = "stablehlo.all_reduce"(%v2469) ({
    ^bb0(%aras4b2bt1: tensor<f32>, %arbs4b2bt1: tensor<f32>):
      %aradds4b2bt1 = stablehlo.add %aras4b2bt1, %arbs4b2bt1 : tensor<f32>
      stablehlo.return %aradds4b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt1 = stablehlo.divide %arsums4b2bt1, %arns4b2bt1 : tensor<512xf32>
    %v8012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8013 = stablehlo.multiply %v8012, %s4b2bt1 : tensor<512xf32>
    %v8014 = stablehlo.add %v8013, %armeans4b2bt1 : tensor<512xf32>
    %v8015 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8016 = stablehlo.multiply %v8015, %s4b2bt1v : tensor<512xf32>
    %v8017 = stablehlo.add %v8016, %v8014 : tensor<512xf32>
    %v8018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8019 = stablehlo.multiply %v8018, %v8017 : tensor<512xf32>
    %v8020 = stablehlo.subtract %s4b2bt1, %v8019 : tensor<512xf32>
    %arsums4b2W2 = "stablehlo.all_reduce"(%v2475) ({
    ^bb0(%aras4b2W2: tensor<f32>, %arbs4b2W2: tensor<f32>):
      %aradds4b2W2 = stablehlo.add %aras4b2W2, %arbs4b2W2 : tensor<f32>
      stablehlo.return %aradds4b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b2W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b2W2 = stablehlo.divide %arsums4b2W2, %arns4b2W2 : tensor<512x512x3x3xf32>
    %v8021 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8022 = stablehlo.multiply %v8021, %s4b2W2 : tensor<512x512x3x3xf32>
    %v8023 = stablehlo.add %v8022, %armeans4b2W2 : tensor<512x512x3x3xf32>
    %v8024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8025 = stablehlo.multiply %v8024, %s4b2W2v : tensor<512x512x3x3xf32>
    %v8026 = stablehlo.add %v8025, %v8023 : tensor<512x512x3x3xf32>
    %v8027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8028 = stablehlo.multiply %v8027, %v8026 : tensor<512x512x3x3xf32>
    %v8029 = stablehlo.subtract %s4b2W2, %v8028 : tensor<512x512x3x3xf32>
    %arsums4b2g2 = "stablehlo.all_reduce"(%v2489) ({
    ^bb0(%aras4b2g2: tensor<f32>, %arbs4b2g2: tensor<f32>):
      %aradds4b2g2 = stablehlo.add %aras4b2g2, %arbs4b2g2 : tensor<f32>
      stablehlo.return %aradds4b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2 = stablehlo.divide %arsums4b2g2, %arns4b2g2 : tensor<512xf32>
    %v8030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8031 = stablehlo.multiply %v8030, %s4b2g2 : tensor<512xf32>
    %v8032 = stablehlo.add %v8031, %armeans4b2g2 : tensor<512xf32>
    %v8033 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8034 = stablehlo.multiply %v8033, %s4b2g2v : tensor<512xf32>
    %v8035 = stablehlo.add %v8034, %v8032 : tensor<512xf32>
    %v8036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8037 = stablehlo.multiply %v8036, %v8035 : tensor<512xf32>
    %v8038 = stablehlo.subtract %s4b2g2, %v8037 : tensor<512xf32>
    %arsums4b2bt2 = "stablehlo.all_reduce"(%v2492) ({
    ^bb0(%aras4b2bt2: tensor<f32>, %arbs4b2bt2: tensor<f32>):
      %aradds4b2bt2 = stablehlo.add %aras4b2bt2, %arbs4b2bt2 : tensor<f32>
      stablehlo.return %aradds4b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt2 = stablehlo.divide %arsums4b2bt2, %arns4b2bt2 : tensor<512xf32>
    %v8039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8040 = stablehlo.multiply %v8039, %s4b2bt2 : tensor<512xf32>
    %v8041 = stablehlo.add %v8040, %armeans4b2bt2 : tensor<512xf32>
    %v8042 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8043 = stablehlo.multiply %v8042, %s4b2bt2v : tensor<512xf32>
    %v8044 = stablehlo.add %v8043, %v8041 : tensor<512xf32>
    %v8045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8046 = stablehlo.multiply %v8045, %v8044 : tensor<512xf32>
    %v8047 = stablehlo.subtract %s4b2bt2, %v8046 : tensor<512xf32>
    %arsums4b2W3 = "stablehlo.all_reduce"(%v2498) ({
    ^bb0(%aras4b2W3: tensor<f32>, %arbs4b2W3: tensor<f32>):
      %aradds4b2W3 = stablehlo.add %aras4b2W3, %arbs4b2W3 : tensor<f32>
      stablehlo.return %aradds4b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b2W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b2W3 = stablehlo.divide %arsums4b2W3, %arns4b2W3 : tensor<2048x512x1x1xf32>
    %v8048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8049 = stablehlo.multiply %v8048, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v8050 = stablehlo.add %v8049, %armeans4b2W3 : tensor<2048x512x1x1xf32>
    %v8051 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8052 = stablehlo.multiply %v8051, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v8053 = stablehlo.add %v8052, %v8050 : tensor<2048x512x1x1xf32>
    %v8054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8055 = stablehlo.multiply %v8054, %v8053 : tensor<2048x512x1x1xf32>
    %v8056 = stablehlo.subtract %s4b2W3, %v8055 : tensor<2048x512x1x1xf32>
    %arsums4b2g3 = "stablehlo.all_reduce"(%v2512) ({
    ^bb0(%aras4b2g3: tensor<f32>, %arbs4b2g3: tensor<f32>):
      %aradds4b2g3 = stablehlo.add %aras4b2g3, %arbs4b2g3 : tensor<f32>
      stablehlo.return %aradds4b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3 = stablehlo.divide %arsums4b2g3, %arns4b2g3 : tensor<2048xf32>
    %v8057 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8058 = stablehlo.multiply %v8057, %s4b2g3 : tensor<2048xf32>
    %v8059 = stablehlo.add %v8058, %armeans4b2g3 : tensor<2048xf32>
    %v8060 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8061 = stablehlo.multiply %v8060, %s4b2g3v : tensor<2048xf32>
    %v8062 = stablehlo.add %v8061, %v8059 : tensor<2048xf32>
    %v8063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8064 = stablehlo.multiply %v8063, %v8062 : tensor<2048xf32>
    %v8065 = stablehlo.subtract %s4b2g3, %v8064 : tensor<2048xf32>
    %arsums4b2bt3 = "stablehlo.all_reduce"(%v2515) ({
    ^bb0(%aras4b2bt3: tensor<f32>, %arbs4b2bt3: tensor<f32>):
      %aradds4b2bt3 = stablehlo.add %aras4b2bt3, %arbs4b2bt3 : tensor<f32>
      stablehlo.return %aradds4b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2bt3 = stablehlo.divide %arsums4b2bt3, %arns4b2bt3 : tensor<2048xf32>
    %v8066 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8067 = stablehlo.multiply %v8066, %s4b2bt3 : tensor<2048xf32>
    %v8068 = stablehlo.add %v8067, %armeans4b2bt3 : tensor<2048xf32>
    %v8069 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8070 = stablehlo.multiply %v8069, %s4b2bt3v : tensor<2048xf32>
    %v8071 = stablehlo.add %v8070, %v8068 : tensor<2048xf32>
    %v8072 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8073 = stablehlo.multiply %v8072, %v8071 : tensor<2048xf32>
    %v8074 = stablehlo.subtract %s4b2bt3, %v8073 : tensor<2048xf32>
    %arsumWd = "stablehlo.all_reduce"(%v2271) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<2048x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<2048x1000xf32>
    %v8075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8076 = stablehlo.multiply %v8075, %Wd : tensor<2048x1000xf32>
    %v8077 = stablehlo.add %v8076, %armeanWd : tensor<2048x1000xf32>
    %v8078 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8079 = stablehlo.multiply %v8078, %Wdv : tensor<2048x1000xf32>
    %v8080 = stablehlo.add %v8079, %v8077 : tensor<2048x1000xf32>
    %v8081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8082 = stablehlo.multiply %v8081, %v8080 : tensor<2048x1000xf32>
    %v8083 = stablehlo.subtract %Wd, %v8082 : tensor<2048x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v2273) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v8084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8085 = stablehlo.multiply %v8084, %bd : tensor<1000xf32>
    %v8086 = stablehlo.add %v8085, %armeanbd : tensor<1000xf32>
    %v8087 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8088 = stablehlo.multiply %v8087, %bdv : tensor<1000xf32>
    %v8089 = stablehlo.add %v8088, %v8086 : tensor<1000xf32>
    %v8090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8091 = stablehlo.multiply %v8090, %v8089 : tensor<1000xf32>
    %v8092 = stablehlo.subtract %bd, %v8091 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v2259 : tensor<64x1000xf32>
    %lohll = stablehlo.multiply %onehot, %llog : tensor<64x1000xf32>
    %lt1s = stablehlo.reduce(%lohll init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<64xf32>
    %llsr = stablehlo.reduce(%llog init: %lz) applies stablehlo.add across dimensions = [1] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<64xf32>
    %lomac = stablehlo.constant dense<0.900000> : tensor<64xf32>
    %laKc = stablehlo.constant dense<0.000100> : tensor<64xf32>
    %llt1 = stablehlo.multiply %lomac, %lt1s : tensor<64xf32>
    %llt2 = stablehlo.multiply %laKc, %llsr : tensor<64xf32>
    %llpe = stablehlo.add %llt1, %llt2 : tensor<64xf32>
    %lsum2 = stablehlo.reduce(%llpe init: %lz) applies stablehlo.add across dimensions = [0] : (tensor<64xf32>, tensor<f32>) -> tensor<f32>
    %lbfc = stablehlo.constant dense<64.0> : tensor<f32>
    %lossm = stablehlo.divide %lsum2, %lbfc : tensor<f32>
    %loss = stablehlo.negate %lossm : tensor<f32>
    return %v6652, %v6661, %v6670, %v6679, %v6688, %v6697, %v6706, %v6715, %v6724, %v6733, %v6742, %v6751, %v6760, %v6769, %v6778, %v6787, %v6796, %v6805, %v6814, %v6823, %v6832, %v6841, %v6850, %v6859, %v6868, %v6877, %v6886, %v6895, %v6904, %v6913, %v6922, %v6931, %v6940, %v6949, %v6958, %v6967, %v6976, %v6985, %v6994, %v7003, %v7012, %v7021, %v7030, %v7039, %v7048, %v7057, %v7066, %v7075, %v7084, %v7093, %v7102, %v7111, %v7120, %v7129, %v7138, %v7147, %v7156, %v7165, %v7174, %v7183, %v7192, %v7201, %v7210, %v7219, %v7228, %v7237, %v7246, %v7255, %v7264, %v7273, %v7282, %v7291, %v7300, %v7309, %v7318, %v7327, %v7336, %v7345, %v7354, %v7363, %v7372, %v7381, %v7390, %v7399, %v7408, %v7417, %v7426, %v7435, %v7444, %v7453, %v7462, %v7471, %v7480, %v7489, %v7498, %v7507, %v7516, %v7525, %v7534, %v7543, %v7552, %v7561, %v7570, %v7579, %v7588, %v7597, %v7606, %v7615, %v7624, %v7633, %v7642, %v7651, %v7660, %v7669, %v7678, %v7687, %v7696, %v7705, %v7714, %v7723, %v7732, %v7741, %v7750, %v7759, %v7768, %v7777, %v7786, %v7795, %v7804, %v7813, %v7822, %v7831, %v7840, %v7849, %v7858, %v7867, %v7876, %v7885, %v7894, %v7903, %v7912, %v7921, %v7930, %v7939, %v7948, %v7957, %v7966, %v7975, %v7984, %v7993, %v8002, %v8011, %v8020, %v8029, %v8038, %v8047, %v8056, %v8065, %v8074, %v8083, %v8092, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v6649, %v6658, %v6667, %v6676, %v6685, %v6694, %v6703, %v6712, %v6721, %v6730, %v6739, %v6748, %v6757, %v6766, %v6775, %v6784, %v6793, %v6802, %v6811, %v6820, %v6829, %v6838, %v6847, %v6856, %v6865, %v6874, %v6883, %v6892, %v6901, %v6910, %v6919, %v6928, %v6937, %v6946, %v6955, %v6964, %v6973, %v6982, %v6991, %v7000, %v7009, %v7018, %v7027, %v7036, %v7045, %v7054, %v7063, %v7072, %v7081, %v7090, %v7099, %v7108, %v7117, %v7126, %v7135, %v7144, %v7153, %v7162, %v7171, %v7180, %v7189, %v7198, %v7207, %v7216, %v7225, %v7234, %v7243, %v7252, %v7261, %v7270, %v7279, %v7288, %v7297, %v7306, %v7315, %v7324, %v7333, %v7342, %v7351, %v7360, %v7369, %v7378, %v7387, %v7396, %v7405, %v7414, %v7423, %v7432, %v7441, %v7450, %v7459, %v7468, %v7477, %v7486, %v7495, %v7504, %v7513, %v7522, %v7531, %v7540, %v7549, %v7558, %v7567, %v7576, %v7585, %v7594, %v7603, %v7612, %v7621, %v7630, %v7639, %v7648, %v7657, %v7666, %v7675, %v7684, %v7693, %v7702, %v7711, %v7720, %v7729, %v7738, %v7747, %v7756, %v7765, %v7774, %v7783, %v7792, %v7801, %v7810, %v7819, %v7828, %v7837, %v7846, %v7855, %v7864, %v7873, %v7882, %v7891, %v7900, %v7909, %v7918, %v7927, %v7936, %v7945, %v7954, %v7963, %v7972, %v7981, %v7990, %v7999, %v8008, %v8017, %v8026, %v8035, %v8044, %v8053, %v8062, %v8071, %v8080, %v8089, %loss, %bc1, %bc2, %v6538, %v6539, %v6540, %v6541, %v6542, %v6543, %v6544, %v6545, %v6546, %v6547, %v6548, %v6549, %v6550, %v6551, %v6552, %v6553, %v6554, %v6555, %v6556, %v6557, %v6558, %v6559, %v6560, %v6561, %v6562, %v6563, %v6564, %v6565, %v6566, %v6567, %v6568, %v6569, %v6570, %v6571, %v6572, %v6573, %v6574, %v6575, %v6576, %v6577, %v6578, %v6579, %v6580, %v6581, %v6582, %v6583, %v6584, %v6585, %v6586, %v6587, %v6588, %v6589, %v6590, %v6591, %v6592, %v6593, %v6594, %v6595, %v6596, %v6597, %v6598, %v6599, %v6600, %v6601, %v6602, %v6603, %v6604, %v6605, %v6606, %v6607, %v6608, %v6609, %v6610, %v6611, %v6612, %v6613, %v6614, %v6615, %v6616, %v6617, %v6618, %v6619, %v6620, %v6621, %v6622, %v6623, %v6624, %v6625, %v6626, %v6627, %v6628, %v6629, %v6630, %v6631, %v6632, %v6633, %v6634, %v6635, %v6636, %v6637, %v6638, %v6639, %v6640, %v6641, %v6642, %v6643 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
