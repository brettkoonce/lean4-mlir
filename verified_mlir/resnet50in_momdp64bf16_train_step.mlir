module @m {
  func.func @resnet50in_momdp64bf16_train_step(%x: tensor<64x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x1x1xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b0W3m: tensor<256x64x1x1xf32>, %s1b0g3m: tensor<256xf32>, %s1b0bt3m: tensor<256xf32>, %s1b0Wpm: tensor<256x64x1x1xf32>, %s1b0gpm: tensor<256xf32>, %s1b0btpm: tensor<256xf32>, %s1b1W1m: tensor<64x256x1x1xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b1W3m: tensor<256x64x1x1xf32>, %s1b1g3m: tensor<256xf32>, %s1b1bt3m: tensor<256xf32>, %s1b2W1m: tensor<64x256x1x1xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %s1b2W3m: tensor<256x64x1x1xf32>, %s1b2g3m: tensor<256xf32>, %s1b2bt3m: tensor<256xf32>, %s2b0W1m: tensor<128x256x1x1xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b0W3m: tensor<512x128x1x1xf32>, %s2b0g3m: tensor<512xf32>, %s2b0bt3m: tensor<512xf32>, %s2b0Wpm: tensor<512x256x1x1xf32>, %s2b0gpm: tensor<512xf32>, %s2b0btpm: tensor<512xf32>, %s2b1W1m: tensor<128x512x1x1xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b1W3m: tensor<512x128x1x1xf32>, %s2b1g3m: tensor<512xf32>, %s2b1bt3m: tensor<512xf32>, %s2b2W1m: tensor<128x512x1x1xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %s2b2W3m: tensor<512x128x1x1xf32>, %s2b2g3m: tensor<512xf32>, %s2b2bt3m: tensor<512xf32>, %s2b3W1m: tensor<128x512x1x1xf32>, %s2b3g1m: tensor<128xf32>, %s2b3bt1m: tensor<128xf32>, %s2b3W2m: tensor<128x128x3x3xf32>, %s2b3g2m: tensor<128xf32>, %s2b3bt2m: tensor<128xf32>, %s2b3W3m: tensor<512x128x1x1xf32>, %s2b3g3m: tensor<512xf32>, %s2b3bt3m: tensor<512xf32>, %s3b0W1m: tensor<256x512x1x1xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b0W3m: tensor<1024x256x1x1xf32>, %s3b0g3m: tensor<1024xf32>, %s3b0bt3m: tensor<1024xf32>, %s3b0Wpm: tensor<1024x512x1x1xf32>, %s3b0gpm: tensor<1024xf32>, %s3b0btpm: tensor<1024xf32>, %s3b1W1m: tensor<256x1024x1x1xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b1W3m: tensor<1024x256x1x1xf32>, %s3b1g3m: tensor<1024xf32>, %s3b1bt3m: tensor<1024xf32>, %s3b2W1m: tensor<256x1024x1x1xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b2W3m: tensor<1024x256x1x1xf32>, %s3b2g3m: tensor<1024xf32>, %s3b2bt3m: tensor<1024xf32>, %s3b3W1m: tensor<256x1024x1x1xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b3W3m: tensor<1024x256x1x1xf32>, %s3b3g3m: tensor<1024xf32>, %s3b3bt3m: tensor<1024xf32>, %s3b4W1m: tensor<256x1024x1x1xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %s3b4W3m: tensor<1024x256x1x1xf32>, %s3b4g3m: tensor<1024xf32>, %s3b4bt3m: tensor<1024xf32>, %s3b5W1m: tensor<256x1024x1x1xf32>, %s3b5g1m: tensor<256xf32>, %s3b5bt1m: tensor<256xf32>, %s3b5W2m: tensor<256x256x3x3xf32>, %s3b5g2m: tensor<256xf32>, %s3b5bt2m: tensor<256xf32>, %s3b5W3m: tensor<1024x256x1x1xf32>, %s3b5g3m: tensor<1024xf32>, %s3b5bt3m: tensor<1024xf32>, %s4b0W1m: tensor<512x1024x1x1xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b0W3m: tensor<2048x512x1x1xf32>, %s4b0g3m: tensor<2048xf32>, %s4b0bt3m: tensor<2048xf32>, %s4b0Wpm: tensor<2048x1024x1x1xf32>, %s4b0gpm: tensor<2048xf32>, %s4b0btpm: tensor<2048xf32>, %s4b1W1m: tensor<512x2048x1x1xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %s4b1W3m: tensor<2048x512x1x1xf32>, %s4b1g3m: tensor<2048xf32>, %s4b1bt3m: tensor<2048xf32>, %s4b2W1m: tensor<512x2048x1x1xf32>, %s4b2g1m: tensor<512xf32>, %s4b2bt1m: tensor<512xf32>, %s4b2W2m: tensor<512x512x3x3xf32>, %s4b2g2m: tensor<512xf32>, %s4b2bt2m: tensor<512xf32>, %s4b2W3m: tensor<2048x512x1x1xf32>, %s4b2g3m: tensor<2048xf32>, %s4b2bt3m: tensor<2048xf32>, %Wdm: tensor<2048x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x1x1xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b0W3v: tensor<256x64x1x1xf32>, %s1b0g3v: tensor<256xf32>, %s1b0bt3v: tensor<256xf32>, %s1b0Wpv: tensor<256x64x1x1xf32>, %s1b0gpv: tensor<256xf32>, %s1b0btpv: tensor<256xf32>, %s1b1W1v: tensor<64x256x1x1xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b1W3v: tensor<256x64x1x1xf32>, %s1b1g3v: tensor<256xf32>, %s1b1bt3v: tensor<256xf32>, %s1b2W1v: tensor<64x256x1x1xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %s1b2W3v: tensor<256x64x1x1xf32>, %s1b2g3v: tensor<256xf32>, %s1b2bt3v: tensor<256xf32>, %s2b0W1v: tensor<128x256x1x1xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b0W3v: tensor<512x128x1x1xf32>, %s2b0g3v: tensor<512xf32>, %s2b0bt3v: tensor<512xf32>, %s2b0Wpv: tensor<512x256x1x1xf32>, %s2b0gpv: tensor<512xf32>, %s2b0btpv: tensor<512xf32>, %s2b1W1v: tensor<128x512x1x1xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b1W3v: tensor<512x128x1x1xf32>, %s2b1g3v: tensor<512xf32>, %s2b1bt3v: tensor<512xf32>, %s2b2W1v: tensor<128x512x1x1xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %s2b2W3v: tensor<512x128x1x1xf32>, %s2b2g3v: tensor<512xf32>, %s2b2bt3v: tensor<512xf32>, %s2b3W1v: tensor<128x512x1x1xf32>, %s2b3g1v: tensor<128xf32>, %s2b3bt1v: tensor<128xf32>, %s2b3W2v: tensor<128x128x3x3xf32>, %s2b3g2v: tensor<128xf32>, %s2b3bt2v: tensor<128xf32>, %s2b3W3v: tensor<512x128x1x1xf32>, %s2b3g3v: tensor<512xf32>, %s2b3bt3v: tensor<512xf32>, %s3b0W1v: tensor<256x512x1x1xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b0W3v: tensor<1024x256x1x1xf32>, %s3b0g3v: tensor<1024xf32>, %s3b0bt3v: tensor<1024xf32>, %s3b0Wpv: tensor<1024x512x1x1xf32>, %s3b0gpv: tensor<1024xf32>, %s3b0btpv: tensor<1024xf32>, %s3b1W1v: tensor<256x1024x1x1xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b1W3v: tensor<1024x256x1x1xf32>, %s3b1g3v: tensor<1024xf32>, %s3b1bt3v: tensor<1024xf32>, %s3b2W1v: tensor<256x1024x1x1xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b2W3v: tensor<1024x256x1x1xf32>, %s3b2g3v: tensor<1024xf32>, %s3b2bt3v: tensor<1024xf32>, %s3b3W1v: tensor<256x1024x1x1xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b3W3v: tensor<1024x256x1x1xf32>, %s3b3g3v: tensor<1024xf32>, %s3b3bt3v: tensor<1024xf32>, %s3b4W1v: tensor<256x1024x1x1xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %s3b4W3v: tensor<1024x256x1x1xf32>, %s3b4g3v: tensor<1024xf32>, %s3b4bt3v: tensor<1024xf32>, %s3b5W1v: tensor<256x1024x1x1xf32>, %s3b5g1v: tensor<256xf32>, %s3b5bt1v: tensor<256xf32>, %s3b5W2v: tensor<256x256x3x3xf32>, %s3b5g2v: tensor<256xf32>, %s3b5bt2v: tensor<256xf32>, %s3b5W3v: tensor<1024x256x1x1xf32>, %s3b5g3v: tensor<1024xf32>, %s3b5bt3v: tensor<1024xf32>, %s4b0W1v: tensor<512x1024x1x1xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b0W3v: tensor<2048x512x1x1xf32>, %s4b0g3v: tensor<2048xf32>, %s4b0bt3v: tensor<2048xf32>, %s4b0Wpv: tensor<2048x1024x1x1xf32>, %s4b0gpv: tensor<2048xf32>, %s4b0btpv: tensor<2048xf32>, %s4b1W1v: tensor<512x2048x1x1xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %s4b1W3v: tensor<2048x512x1x1xf32>, %s4b1g3v: tensor<2048xf32>, %s4b1bt3v: tensor<2048xf32>, %s4b2W1v: tensor<512x2048x1x1xf32>, %s4b2g1v: tensor<512xf32>, %s4b2bt1v: tensor<512xf32>, %s4b2W2v: tensor<512x512x3x3xf32>, %s4b2g2v: tensor<512xf32>, %s4b2bt2v: tensor<512xf32>, %s4b2W3v: tensor<2048x512x1x1xf32>, %s4b2g3v: tensor<2048xf32>, %s4b2bt3v: tensor<2048xf32>, %Wdv: tensor<2048x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b0n3mui: tensor<256xf32>, %s1b0n3vari: tensor<256xf32>, %s1b0npmui: tensor<256xf32>, %s1b0npvari: tensor<256xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b1n3mui: tensor<256xf32>, %s1b1n3vari: tensor<256xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %s1b2n3mui: tensor<256xf32>, %s1b2n3vari: tensor<256xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b0n3mui: tensor<512xf32>, %s2b0n3vari: tensor<512xf32>, %s2b0npmui: tensor<512xf32>, %s2b0npvari: tensor<512xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b1n3mui: tensor<512xf32>, %s2b1n3vari: tensor<512xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %s2b2n3mui: tensor<512xf32>, %s2b2n3vari: tensor<512xf32>, %s2b3n1mui: tensor<128xf32>, %s2b3n1vari: tensor<128xf32>, %s2b3n2mui: tensor<128xf32>, %s2b3n2vari: tensor<128xf32>, %s2b3n3mui: tensor<512xf32>, %s2b3n3vari: tensor<512xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b0n3mui: tensor<1024xf32>, %s3b0n3vari: tensor<1024xf32>, %s3b0npmui: tensor<1024xf32>, %s3b0npvari: tensor<1024xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b1n3mui: tensor<1024xf32>, %s3b1n3vari: tensor<1024xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b2n3mui: tensor<1024xf32>, %s3b2n3vari: tensor<1024xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b3n3mui: tensor<1024xf32>, %s3b3n3vari: tensor<1024xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %s3b4n3mui: tensor<1024xf32>, %s3b4n3vari: tensor<1024xf32>, %s3b5n1mui: tensor<256xf32>, %s3b5n1vari: tensor<256xf32>, %s3b5n2mui: tensor<256xf32>, %s3b5n2vari: tensor<256xf32>, %s3b5n3mui: tensor<1024xf32>, %s3b5n3vari: tensor<1024xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b0n3mui: tensor<2048xf32>, %s4b0n3vari: tensor<2048xf32>, %s4b0npmui: tensor<2048xf32>, %s4b0npvari: tensor<2048xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %s4b1n3mui: tensor<2048xf32>, %s4b1n3vari: tensor<2048xf32>, %s4b2n1mui: tensor<512xf32>, %s4b2n1vari: tensor<512xf32>, %s4b2n2mui: tensor<512xf32>, %s4b2n2vari: tensor<512xf32>, %s4b2n3mui: tensor<2048xf32>, %s4b2n3vari: tensor<2048xf32>, %onehot: tensor<64x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>) {
    // ── ResNet-50 bottleneck batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 4 replicas ──
    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`
    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5).
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
    %zb1024 = stablehlo.constant dense<0.0> : tensor<1024xf32>
    %zb2048 = stablehlo.constant dense<0.0> : tensor<2048xf32>
    %v0 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v1 = stablehlo.convert %v0 : (tensor<64x3x224x224xf32>) -> tensor<64x3x224x224xbf16>
    %v2 = stablehlo.convert %sW : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xbf16>
    %v3 = stablehlo.convolution(%v1, %v2)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x3x224x224xbf16>, tensor<64x3x7x7xbf16>) -> tensor<64x64x112x112xbf16>
    %v4 = stablehlo.convert %v3 : (tensor<64x64x112x112xbf16>) -> tensor<64x64x112x112xf32>
    %v5 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6 = stablehlo.add %v4, %v5 : tensor<64x64x112x112xf32>
    %v7 = stablehlo.reshape %v6 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v8 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v9 = stablehlo.constant dense<0.0> : tensor<f32>
    %v10 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v11 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v12 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v13 = stablehlo.broadcast_in_dim %v12, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v14 = stablehlo.divide %v13, %v10 : tensor<64x64x112x112xf32>
    %v15 = stablehlo.subtract %v8, %v14 : tensor<64x64x112x112xf32>
    %v16 = stablehlo.multiply %v15, %v15 : tensor<64x64x112x112xf32>
    %v17 = stablehlo.reduce(%v16 init: %v9) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v19 = stablehlo.divide %v18, %v10 : tensor<64x64x112x112xf32>
    %v20 = stablehlo.add %v19, %v11 : tensor<64x64x112x112xf32>
    %v21 = stablehlo.rsqrt %v20 : tensor<64x64x112x112xf32>
    %v22 = stablehlo.multiply %v15, %v21 : tensor<64x64x112x112xf32>
    %v23 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v24 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v25 = stablehlo.multiply %v22, %v23 : tensor<64x64x112x112xf32>
    %v26 = stablehlo.add %v25, %v24 : tensor<64x64x112x112xf32>
    %v27 = stablehlo.reshape %v26 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v28 = stablehlo.reshape %v27 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v29 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v30 = stablehlo.maximum %v28, %v29 : tensor<64x64x112x112xf32>
    %v31 = stablehlo.reshape %v30 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v32 = stablehlo.reshape %v31 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v33 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v34 = "stablehlo.reduce_window"(%v32, %v33) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
    %v35 = stablehlo.reshape %v34 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v36 = stablehlo.reshape %v35 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v37 = stablehlo.convert %v36 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v38 = stablehlo.convert %s1b0W1 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v39 = stablehlo.convolution(%v37, %v38)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v40 = stablehlo.convert %v39 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v41 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v42 = stablehlo.add %v40, %v41 : tensor<64x64x56x56xf32>
    %v43 = stablehlo.reshape %v42 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v45 = stablehlo.constant dense<0.0> : tensor<f32>
    %v46 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v47 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v48 = stablehlo.reduce(%v44 init: %v45) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v49 = stablehlo.broadcast_in_dim %v48, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v50 = stablehlo.divide %v49, %v46 : tensor<64x64x56x56xf32>
    %v51 = stablehlo.subtract %v44, %v50 : tensor<64x64x56x56xf32>
    %v52 = stablehlo.multiply %v51, %v51 : tensor<64x64x56x56xf32>
    %v53 = stablehlo.reduce(%v52 init: %v45) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v54 = stablehlo.broadcast_in_dim %v53, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v55 = stablehlo.divide %v54, %v46 : tensor<64x64x56x56xf32>
    %v56 = stablehlo.add %v55, %v47 : tensor<64x64x56x56xf32>
    %v57 = stablehlo.rsqrt %v56 : tensor<64x64x56x56xf32>
    %v58 = stablehlo.multiply %v51, %v57 : tensor<64x64x56x56xf32>
    %v59 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v60 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v61 = stablehlo.multiply %v58, %v59 : tensor<64x64x56x56xf32>
    %v62 = stablehlo.add %v61, %v60 : tensor<64x64x56x56xf32>
    %v63 = stablehlo.reshape %v62 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v65 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v66 = stablehlo.maximum %v64, %v65 : tensor<64x64x56x56xf32>
    %v67 = stablehlo.reshape %v66 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v68 = stablehlo.reshape %v67 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v69 = stablehlo.convert %v68 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v70 = stablehlo.convert %s1b0W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v71 = stablehlo.convolution(%v69, %v70)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v72 = stablehlo.convert %v71 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v73 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v74 = stablehlo.add %v72, %v73 : tensor<64x64x56x56xf32>
    %v75 = stablehlo.reshape %v74 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v76 = stablehlo.reshape %v75 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v77 = stablehlo.constant dense<0.0> : tensor<f32>
    %v78 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v79 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v80 = stablehlo.reduce(%v76 init: %v77) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v81 = stablehlo.broadcast_in_dim %v80, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v82 = stablehlo.divide %v81, %v78 : tensor<64x64x56x56xf32>
    %v83 = stablehlo.subtract %v76, %v82 : tensor<64x64x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v83 : tensor<64x64x56x56xf32>
    %v85 = stablehlo.reduce(%v84 init: %v77) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v86 = stablehlo.broadcast_in_dim %v85, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v87 = stablehlo.divide %v86, %v78 : tensor<64x64x56x56xf32>
    %v88 = stablehlo.add %v87, %v79 : tensor<64x64x56x56xf32>
    %v89 = stablehlo.rsqrt %v88 : tensor<64x64x56x56xf32>
    %v90 = stablehlo.multiply %v83, %v89 : tensor<64x64x56x56xf32>
    %v91 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v92 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v93 = stablehlo.multiply %v90, %v91 : tensor<64x64x56x56xf32>
    %v94 = stablehlo.add %v93, %v92 : tensor<64x64x56x56xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v96 = stablehlo.reshape %v95 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v97 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v98 = stablehlo.maximum %v96, %v97 : tensor<64x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v101 = stablehlo.convert %v100 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v102 = stablehlo.convert %s1b0W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v103 = stablehlo.convolution(%v101, %v102)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v104 = stablehlo.convert %v103 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v105 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v106 = stablehlo.add %v104, %v105 : tensor<64x256x56x56xf32>
    %v107 = stablehlo.reshape %v106 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v108 = stablehlo.reshape %v107 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v110 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v111 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v112 = stablehlo.reduce(%v108 init: %v109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v113 = stablehlo.broadcast_in_dim %v112, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v114 = stablehlo.divide %v113, %v110 : tensor<64x256x56x56xf32>
    %v115 = stablehlo.subtract %v108, %v114 : tensor<64x256x56x56xf32>
    %v116 = stablehlo.multiply %v115, %v115 : tensor<64x256x56x56xf32>
    %v117 = stablehlo.reduce(%v116 init: %v109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v118 = stablehlo.broadcast_in_dim %v117, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v119 = stablehlo.divide %v118, %v110 : tensor<64x256x56x56xf32>
    %v120 = stablehlo.add %v119, %v111 : tensor<64x256x56x56xf32>
    %v121 = stablehlo.rsqrt %v120 : tensor<64x256x56x56xf32>
    %v122 = stablehlo.multiply %v115, %v121 : tensor<64x256x56x56xf32>
    %v123 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v124 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v125 = stablehlo.multiply %v122, %v123 : tensor<64x256x56x56xf32>
    %v126 = stablehlo.add %v125, %v124 : tensor<64x256x56x56xf32>
    %v127 = stablehlo.reshape %v126 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v128 = stablehlo.reshape %v35 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v129 = stablehlo.convert %v128 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v130 = stablehlo.convert %s1b0Wp : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v131 = stablehlo.convolution(%v129, %v130)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v132 = stablehlo.convert %v131 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v133 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v134 = stablehlo.add %v132, %v133 : tensor<64x256x56x56xf32>
    %v135 = stablehlo.reshape %v134 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v138 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v139 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v140 = stablehlo.reduce(%v136 init: %v137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v141 = stablehlo.broadcast_in_dim %v140, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v142 = stablehlo.divide %v141, %v138 : tensor<64x256x56x56xf32>
    %v143 = stablehlo.subtract %v136, %v142 : tensor<64x256x56x56xf32>
    %v144 = stablehlo.multiply %v143, %v143 : tensor<64x256x56x56xf32>
    %v145 = stablehlo.reduce(%v144 init: %v137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v146 = stablehlo.broadcast_in_dim %v145, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v147 = stablehlo.divide %v146, %v138 : tensor<64x256x56x56xf32>
    %v148 = stablehlo.add %v147, %v139 : tensor<64x256x56x56xf32>
    %v149 = stablehlo.rsqrt %v148 : tensor<64x256x56x56xf32>
    %v150 = stablehlo.multiply %v143, %v149 : tensor<64x256x56x56xf32>
    %v151 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v152 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v153 = stablehlo.multiply %v150, %v151 : tensor<64x256x56x56xf32>
    %v154 = stablehlo.add %v153, %v152 : tensor<64x256x56x56xf32>
    %v155 = stablehlo.reshape %v154 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v156 = stablehlo.reshape %v127 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v157 = stablehlo.reshape %v155 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v158 = stablehlo.add %v156, %v157 : tensor<64x64x112x112xf32>
    %v159 = stablehlo.reshape %v158 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v161 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v162 = stablehlo.maximum %v160, %v161 : tensor<64x64x112x112xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v165 = stablehlo.convert %v164 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v166 = stablehlo.convert %s1b1W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v167 = stablehlo.convolution(%v165, %v166)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v168 = stablehlo.convert %v167 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v169 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v170 = stablehlo.add %v168, %v169 : tensor<64x64x56x56xf32>
    %v171 = stablehlo.reshape %v170 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v172 = stablehlo.reshape %v171 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v174 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v175 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v176 = stablehlo.reduce(%v172 init: %v173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v177 = stablehlo.broadcast_in_dim %v176, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v178 = stablehlo.divide %v177, %v174 : tensor<64x64x56x56xf32>
    %v179 = stablehlo.subtract %v172, %v178 : tensor<64x64x56x56xf32>
    %v180 = stablehlo.multiply %v179, %v179 : tensor<64x64x56x56xf32>
    %v181 = stablehlo.reduce(%v180 init: %v173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v182 = stablehlo.broadcast_in_dim %v181, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v183 = stablehlo.divide %v182, %v174 : tensor<64x64x56x56xf32>
    %v184 = stablehlo.add %v183, %v175 : tensor<64x64x56x56xf32>
    %v185 = stablehlo.rsqrt %v184 : tensor<64x64x56x56xf32>
    %v186 = stablehlo.multiply %v179, %v185 : tensor<64x64x56x56xf32>
    %v187 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v189 = stablehlo.multiply %v186, %v187 : tensor<64x64x56x56xf32>
    %v190 = stablehlo.add %v189, %v188 : tensor<64x64x56x56xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v192 = stablehlo.reshape %v191 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v193 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v194 = stablehlo.maximum %v192, %v193 : tensor<64x64x56x56xf32>
    %v195 = stablehlo.reshape %v194 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v196 = stablehlo.reshape %v195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v197 = stablehlo.convert %v196 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v198 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v199 = stablehlo.convolution(%v197, %v198)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v200 = stablehlo.convert %v199 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v201 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v202 = stablehlo.add %v200, %v201 : tensor<64x64x56x56xf32>
    %v203 = stablehlo.reshape %v202 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v204 = stablehlo.reshape %v203 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v205 = stablehlo.constant dense<0.0> : tensor<f32>
    %v206 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v207 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v208 = stablehlo.reduce(%v204 init: %v205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v209 = stablehlo.broadcast_in_dim %v208, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v210 = stablehlo.divide %v209, %v206 : tensor<64x64x56x56xf32>
    %v211 = stablehlo.subtract %v204, %v210 : tensor<64x64x56x56xf32>
    %v212 = stablehlo.multiply %v211, %v211 : tensor<64x64x56x56xf32>
    %v213 = stablehlo.reduce(%v212 init: %v205) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v214 = stablehlo.broadcast_in_dim %v213, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v215 = stablehlo.divide %v214, %v206 : tensor<64x64x56x56xf32>
    %v216 = stablehlo.add %v215, %v207 : tensor<64x64x56x56xf32>
    %v217 = stablehlo.rsqrt %v216 : tensor<64x64x56x56xf32>
    %v218 = stablehlo.multiply %v211, %v217 : tensor<64x64x56x56xf32>
    %v219 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v220 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v221 = stablehlo.multiply %v218, %v219 : tensor<64x64x56x56xf32>
    %v222 = stablehlo.add %v221, %v220 : tensor<64x64x56x56xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v225 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v226 = stablehlo.maximum %v224, %v225 : tensor<64x64x56x56xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v229 = stablehlo.convert %v228 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v230 = stablehlo.convert %s1b1W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v231 = stablehlo.convolution(%v229, %v230)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v232 = stablehlo.convert %v231 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v233 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v234 = stablehlo.add %v232, %v233 : tensor<64x256x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v238 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v239 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v240 = stablehlo.reduce(%v236 init: %v237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v242 = stablehlo.divide %v241, %v238 : tensor<64x256x56x56xf32>
    %v243 = stablehlo.subtract %v236, %v242 : tensor<64x256x56x56xf32>
    %v244 = stablehlo.multiply %v243, %v243 : tensor<64x256x56x56xf32>
    %v245 = stablehlo.reduce(%v244 init: %v237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v246 = stablehlo.broadcast_in_dim %v245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v247 = stablehlo.divide %v246, %v238 : tensor<64x256x56x56xf32>
    %v248 = stablehlo.add %v247, %v239 : tensor<64x256x56x56xf32>
    %v249 = stablehlo.rsqrt %v248 : tensor<64x256x56x56xf32>
    %v250 = stablehlo.multiply %v243, %v249 : tensor<64x256x56x56xf32>
    %v251 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v252 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v253 = stablehlo.multiply %v250, %v251 : tensor<64x256x56x56xf32>
    %v254 = stablehlo.add %v253, %v252 : tensor<64x256x56x56xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v256 = stablehlo.reshape %v255 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v257 = stablehlo.reshape %v163 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v258 = stablehlo.add %v256, %v257 : tensor<64x64x112x112xf32>
    %v259 = stablehlo.reshape %v258 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v260 = stablehlo.reshape %v259 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v261 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v262 = stablehlo.maximum %v260, %v261 : tensor<64x64x112x112xf32>
    %v263 = stablehlo.reshape %v262 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v265 = stablehlo.convert %v264 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v266 = stablehlo.convert %s1b2W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v267 = stablehlo.convolution(%v265, %v266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v268 = stablehlo.convert %v267 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v269 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v270 = stablehlo.add %v268, %v269 : tensor<64x64x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v272 = stablehlo.reshape %v271 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v273 = stablehlo.constant dense<0.0> : tensor<f32>
    %v274 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v275 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v276 = stablehlo.reduce(%v272 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v277 = stablehlo.broadcast_in_dim %v276, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v278 = stablehlo.divide %v277, %v274 : tensor<64x64x56x56xf32>
    %v279 = stablehlo.subtract %v272, %v278 : tensor<64x64x56x56xf32>
    %v280 = stablehlo.multiply %v279, %v279 : tensor<64x64x56x56xf32>
    %v281 = stablehlo.reduce(%v280 init: %v273) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v282 = stablehlo.broadcast_in_dim %v281, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v283 = stablehlo.divide %v282, %v274 : tensor<64x64x56x56xf32>
    %v284 = stablehlo.add %v283, %v275 : tensor<64x64x56x56xf32>
    %v285 = stablehlo.rsqrt %v284 : tensor<64x64x56x56xf32>
    %v286 = stablehlo.multiply %v279, %v285 : tensor<64x64x56x56xf32>
    %v287 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v288 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v289 = stablehlo.multiply %v286, %v287 : tensor<64x64x56x56xf32>
    %v290 = stablehlo.add %v289, %v288 : tensor<64x64x56x56xf32>
    %v291 = stablehlo.reshape %v290 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v292 = stablehlo.reshape %v291 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v293 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v294 = stablehlo.maximum %v292, %v293 : tensor<64x64x56x56xf32>
    %v295 = stablehlo.reshape %v294 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v296 = stablehlo.reshape %v295 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v297 = stablehlo.convert %v296 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v298 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v299 = stablehlo.convolution(%v297, %v298)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v300 = stablehlo.convert %v299 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v301 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v302 = stablehlo.add %v300, %v301 : tensor<64x64x56x56xf32>
    %v303 = stablehlo.reshape %v302 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v306 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v307 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v308 = stablehlo.reduce(%v304 init: %v305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v309 = stablehlo.broadcast_in_dim %v308, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v310 = stablehlo.divide %v309, %v306 : tensor<64x64x56x56xf32>
    %v311 = stablehlo.subtract %v304, %v310 : tensor<64x64x56x56xf32>
    %v312 = stablehlo.multiply %v311, %v311 : tensor<64x64x56x56xf32>
    %v313 = stablehlo.reduce(%v312 init: %v305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v314 = stablehlo.broadcast_in_dim %v313, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v315 = stablehlo.divide %v314, %v306 : tensor<64x64x56x56xf32>
    %v316 = stablehlo.add %v315, %v307 : tensor<64x64x56x56xf32>
    %v317 = stablehlo.rsqrt %v316 : tensor<64x64x56x56xf32>
    %v318 = stablehlo.multiply %v311, %v317 : tensor<64x64x56x56xf32>
    %v319 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v320 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v321 = stablehlo.multiply %v318, %v319 : tensor<64x64x56x56xf32>
    %v322 = stablehlo.add %v321, %v320 : tensor<64x64x56x56xf32>
    %v323 = stablehlo.reshape %v322 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v324 = stablehlo.reshape %v323 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v326 = stablehlo.maximum %v324, %v325 : tensor<64x64x56x56xf32>
    %v327 = stablehlo.reshape %v326 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v328 = stablehlo.reshape %v327 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v329 = stablehlo.convert %v328 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v330 = stablehlo.convert %s1b2W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v331 = stablehlo.convolution(%v329, %v330)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v332 = stablehlo.convert %v331 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v333 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v334 = stablehlo.add %v332, %v333 : tensor<64x256x56x56xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v338 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v339 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v340 = stablehlo.reduce(%v336 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v341 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v342 = stablehlo.divide %v341, %v338 : tensor<64x256x56x56xf32>
    %v343 = stablehlo.subtract %v336, %v342 : tensor<64x256x56x56xf32>
    %v344 = stablehlo.multiply %v343, %v343 : tensor<64x256x56x56xf32>
    %v345 = stablehlo.reduce(%v344 init: %v337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v346 = stablehlo.broadcast_in_dim %v345, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v347 = stablehlo.divide %v346, %v338 : tensor<64x256x56x56xf32>
    %v348 = stablehlo.add %v347, %v339 : tensor<64x256x56x56xf32>
    %v349 = stablehlo.rsqrt %v348 : tensor<64x256x56x56xf32>
    %v350 = stablehlo.multiply %v343, %v349 : tensor<64x256x56x56xf32>
    %v351 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v352 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v353 = stablehlo.multiply %v350, %v351 : tensor<64x256x56x56xf32>
    %v354 = stablehlo.add %v353, %v352 : tensor<64x256x56x56xf32>
    %v355 = stablehlo.reshape %v354 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v357 = stablehlo.reshape %v263 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v358 = stablehlo.add %v356, %v357 : tensor<64x64x112x112xf32>
    %v359 = stablehlo.reshape %v358 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v361 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v362 = stablehlo.maximum %v360, %v361 : tensor<64x64x112x112xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v364 = stablehlo.reshape %v363 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v365 = stablehlo.convert %v364 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v366 = stablehlo.convert %s2b0W1 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v367 = stablehlo.convolution(%v365, %v366)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<128x256x1x1xbf16>) -> tensor<64x128x56x56xbf16>
    %v368 = stablehlo.convert %v367 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v369 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v370 = stablehlo.add %v368, %v369 : tensor<64x128x56x56xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v374 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v375 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v376 = stablehlo.reduce(%v372 init: %v373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v377 = stablehlo.broadcast_in_dim %v376, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v378 = stablehlo.divide %v377, %v374 : tensor<64x128x56x56xf32>
    %v379 = stablehlo.subtract %v372, %v378 : tensor<64x128x56x56xf32>
    %v380 = stablehlo.multiply %v379, %v379 : tensor<64x128x56x56xf32>
    %v381 = stablehlo.reduce(%v380 init: %v373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v382 = stablehlo.broadcast_in_dim %v381, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v383 = stablehlo.divide %v382, %v374 : tensor<64x128x56x56xf32>
    %v384 = stablehlo.add %v383, %v375 : tensor<64x128x56x56xf32>
    %v385 = stablehlo.rsqrt %v384 : tensor<64x128x56x56xf32>
    %v386 = stablehlo.multiply %v379, %v385 : tensor<64x128x56x56xf32>
    %v387 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v389 = stablehlo.multiply %v386, %v387 : tensor<64x128x56x56xf32>
    %v390 = stablehlo.add %v389, %v388 : tensor<64x128x56x56xf32>
    %v391 = stablehlo.reshape %v390 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v393 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v394 = stablehlo.maximum %v392, %v393 : tensor<64x128x56x56xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v396 = stablehlo.reshape %v395 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v397 = stablehlo.convert %v396 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v398 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v399 = stablehlo.convolution(%v397, %v398)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v400 = stablehlo.convert %v399 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v401 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v402 = stablehlo.add %v400, %v401 : tensor<64x128x28x28xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v406 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v407 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v408 = stablehlo.reduce(%v404 init: %v405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v409 = stablehlo.broadcast_in_dim %v408, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v410 = stablehlo.divide %v409, %v406 : tensor<64x128x28x28xf32>
    %v411 = stablehlo.subtract %v404, %v410 : tensor<64x128x28x28xf32>
    %v412 = stablehlo.multiply %v411, %v411 : tensor<64x128x28x28xf32>
    %v413 = stablehlo.reduce(%v412 init: %v405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v414 = stablehlo.broadcast_in_dim %v413, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v415 = stablehlo.divide %v414, %v406 : tensor<64x128x28x28xf32>
    %v416 = stablehlo.add %v415, %v407 : tensor<64x128x28x28xf32>
    %v417 = stablehlo.rsqrt %v416 : tensor<64x128x28x28xf32>
    %v418 = stablehlo.multiply %v411, %v417 : tensor<64x128x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v420 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v421 = stablehlo.multiply %v418, %v419 : tensor<64x128x28x28xf32>
    %v422 = stablehlo.add %v421, %v420 : tensor<64x128x28x28xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v426 = stablehlo.maximum %v424, %v425 : tensor<64x128x28x28xf32>
    %v427 = stablehlo.reshape %v426 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v429 = stablehlo.convert %v428 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v430 = stablehlo.convert %s2b0W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v431 = stablehlo.convolution(%v429, %v430)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v432 = stablehlo.convert %v431 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<64x512x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v439 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<64x512x28x28xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<64x512x28x28xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<64x512x28x28xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<64x512x28x28xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<64x512x28x28xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<64x512x28x28xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<64x512x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<64x512x28x28xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<64x512x28x28xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v456 = stablehlo.reshape %v363 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v457 = stablehlo.convert %v456 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v458 = stablehlo.convert %s2b0Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v459 = stablehlo.convolution(%v457, %v458)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v460 = stablehlo.convert %v459 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v461 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v462 = stablehlo.add %v460, %v461 : tensor<64x512x28x28xf32>
    %v463 = stablehlo.reshape %v462 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v464 = stablehlo.reshape %v463 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v466 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v467 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v468 = stablehlo.reduce(%v464 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v469 = stablehlo.broadcast_in_dim %v468, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v470 = stablehlo.divide %v469, %v466 : tensor<64x512x28x28xf32>
    %v471 = stablehlo.subtract %v464, %v470 : tensor<64x512x28x28xf32>
    %v472 = stablehlo.multiply %v471, %v471 : tensor<64x512x28x28xf32>
    %v473 = stablehlo.reduce(%v472 init: %v465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v474 = stablehlo.broadcast_in_dim %v473, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v475 = stablehlo.divide %v474, %v466 : tensor<64x512x28x28xf32>
    %v476 = stablehlo.add %v475, %v467 : tensor<64x512x28x28xf32>
    %v477 = stablehlo.rsqrt %v476 : tensor<64x512x28x28xf32>
    %v478 = stablehlo.multiply %v471, %v477 : tensor<64x512x28x28xf32>
    %v479 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v480 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v481 = stablehlo.multiply %v478, %v479 : tensor<64x512x28x28xf32>
    %v482 = stablehlo.add %v481, %v480 : tensor<64x512x28x28xf32>
    %v483 = stablehlo.reshape %v482 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v484 = stablehlo.reshape %v455 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v485 = stablehlo.reshape %v483 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v486 = stablehlo.add %v484, %v485 : tensor<64x128x56x56xf32>
    %v487 = stablehlo.reshape %v486 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v488 = stablehlo.reshape %v487 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v489 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v490 = stablehlo.maximum %v488, %v489 : tensor<64x128x56x56xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v493 = stablehlo.convert %v492 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v494 = stablehlo.convert %s2b1W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v495 = stablehlo.convolution(%v493, %v494)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v496 = stablehlo.convert %v495 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v497 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v498 = stablehlo.add %v496, %v497 : tensor<64x128x28x28xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v500 = stablehlo.reshape %v499 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v502 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v503 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v504 = stablehlo.reduce(%v500 init: %v501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v505 = stablehlo.broadcast_in_dim %v504, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v506 = stablehlo.divide %v505, %v502 : tensor<64x128x28x28xf32>
    %v507 = stablehlo.subtract %v500, %v506 : tensor<64x128x28x28xf32>
    %v508 = stablehlo.multiply %v507, %v507 : tensor<64x128x28x28xf32>
    %v509 = stablehlo.reduce(%v508 init: %v501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v510 = stablehlo.broadcast_in_dim %v509, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v511 = stablehlo.divide %v510, %v502 : tensor<64x128x28x28xf32>
    %v512 = stablehlo.add %v511, %v503 : tensor<64x128x28x28xf32>
    %v513 = stablehlo.rsqrt %v512 : tensor<64x128x28x28xf32>
    %v514 = stablehlo.multiply %v507, %v513 : tensor<64x128x28x28xf32>
    %v515 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v516 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v517 = stablehlo.multiply %v514, %v515 : tensor<64x128x28x28xf32>
    %v518 = stablehlo.add %v517, %v516 : tensor<64x128x28x28xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v520 = stablehlo.reshape %v519 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v521 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v522 = stablehlo.maximum %v520, %v521 : tensor<64x128x28x28xf32>
    %v523 = stablehlo.reshape %v522 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v524 = stablehlo.reshape %v523 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v525 = stablehlo.convert %v524 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v526 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v527 = stablehlo.convolution(%v525, %v526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v528 = stablehlo.convert %v527 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v529 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v530 = stablehlo.add %v528, %v529 : tensor<64x128x28x28xf32>
    %v531 = stablehlo.reshape %v530 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v532 = stablehlo.reshape %v531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v534 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v535 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v536 = stablehlo.reduce(%v532 init: %v533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v537 = stablehlo.broadcast_in_dim %v536, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v538 = stablehlo.divide %v537, %v534 : tensor<64x128x28x28xf32>
    %v539 = stablehlo.subtract %v532, %v538 : tensor<64x128x28x28xf32>
    %v540 = stablehlo.multiply %v539, %v539 : tensor<64x128x28x28xf32>
    %v541 = stablehlo.reduce(%v540 init: %v533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v542 = stablehlo.broadcast_in_dim %v541, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v543 = stablehlo.divide %v542, %v534 : tensor<64x128x28x28xf32>
    %v544 = stablehlo.add %v543, %v535 : tensor<64x128x28x28xf32>
    %v545 = stablehlo.rsqrt %v544 : tensor<64x128x28x28xf32>
    %v546 = stablehlo.multiply %v539, %v545 : tensor<64x128x28x28xf32>
    %v547 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v548 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v549 = stablehlo.multiply %v546, %v547 : tensor<64x128x28x28xf32>
    %v550 = stablehlo.add %v549, %v548 : tensor<64x128x28x28xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v553 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v554 = stablehlo.maximum %v552, %v553 : tensor<64x128x28x28xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v556 = stablehlo.reshape %v555 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v557 = stablehlo.convert %v556 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v558 = stablehlo.convert %s2b1W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v559 = stablehlo.convolution(%v557, %v558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v560 = stablehlo.convert %v559 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v561 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v562 = stablehlo.add %v560, %v561 : tensor<64x512x28x28xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v564 = stablehlo.reshape %v563 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v566 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v567 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v568 = stablehlo.reduce(%v564 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v569 = stablehlo.broadcast_in_dim %v568, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v570 = stablehlo.divide %v569, %v566 : tensor<64x512x28x28xf32>
    %v571 = stablehlo.subtract %v564, %v570 : tensor<64x512x28x28xf32>
    %v572 = stablehlo.multiply %v571, %v571 : tensor<64x512x28x28xf32>
    %v573 = stablehlo.reduce(%v572 init: %v565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v574 = stablehlo.broadcast_in_dim %v573, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v575 = stablehlo.divide %v574, %v566 : tensor<64x512x28x28xf32>
    %v576 = stablehlo.add %v575, %v567 : tensor<64x512x28x28xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<64x512x28x28xf32>
    %v578 = stablehlo.multiply %v571, %v577 : tensor<64x512x28x28xf32>
    %v579 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v580 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v581 = stablehlo.multiply %v578, %v579 : tensor<64x512x28x28xf32>
    %v582 = stablehlo.add %v581, %v580 : tensor<64x512x28x28xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v585 = stablehlo.reshape %v491 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v586 = stablehlo.add %v584, %v585 : tensor<64x128x56x56xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v588 = stablehlo.reshape %v587 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v589 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v590 = stablehlo.maximum %v588, %v589 : tensor<64x128x56x56xf32>
    %v591 = stablehlo.reshape %v590 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v592 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v593 = stablehlo.convert %v592 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v594 = stablehlo.convert %s2b2W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v595 = stablehlo.convolution(%v593, %v594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v596 = stablehlo.convert %v595 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v597 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v598 = stablehlo.add %v596, %v597 : tensor<64x128x28x28xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v600 = stablehlo.reshape %v599 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v602 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v603 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v604 = stablehlo.reduce(%v600 init: %v601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v605 = stablehlo.broadcast_in_dim %v604, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v606 = stablehlo.divide %v605, %v602 : tensor<64x128x28x28xf32>
    %v607 = stablehlo.subtract %v600, %v606 : tensor<64x128x28x28xf32>
    %v608 = stablehlo.multiply %v607, %v607 : tensor<64x128x28x28xf32>
    %v609 = stablehlo.reduce(%v608 init: %v601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v610 = stablehlo.broadcast_in_dim %v609, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v611 = stablehlo.divide %v610, %v602 : tensor<64x128x28x28xf32>
    %v612 = stablehlo.add %v611, %v603 : tensor<64x128x28x28xf32>
    %v613 = stablehlo.rsqrt %v612 : tensor<64x128x28x28xf32>
    %v614 = stablehlo.multiply %v607, %v613 : tensor<64x128x28x28xf32>
    %v615 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v616 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v617 = stablehlo.multiply %v614, %v615 : tensor<64x128x28x28xf32>
    %v618 = stablehlo.add %v617, %v616 : tensor<64x128x28x28xf32>
    %v619 = stablehlo.reshape %v618 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v620 = stablehlo.reshape %v619 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v621 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v622 = stablehlo.maximum %v620, %v621 : tensor<64x128x28x28xf32>
    %v623 = stablehlo.reshape %v622 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v624 = stablehlo.reshape %v623 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v625 = stablehlo.convert %v624 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v626 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v627 = stablehlo.convolution(%v625, %v626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v628 = stablehlo.convert %v627 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v629 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v630 = stablehlo.add %v628, %v629 : tensor<64x128x28x28xf32>
    %v631 = stablehlo.reshape %v630 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v634 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v635 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v636 = stablehlo.reduce(%v632 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v637 = stablehlo.broadcast_in_dim %v636, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v638 = stablehlo.divide %v637, %v634 : tensor<64x128x28x28xf32>
    %v639 = stablehlo.subtract %v632, %v638 : tensor<64x128x28x28xf32>
    %v640 = stablehlo.multiply %v639, %v639 : tensor<64x128x28x28xf32>
    %v641 = stablehlo.reduce(%v640 init: %v633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v642 = stablehlo.broadcast_in_dim %v641, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v643 = stablehlo.divide %v642, %v634 : tensor<64x128x28x28xf32>
    %v644 = stablehlo.add %v643, %v635 : tensor<64x128x28x28xf32>
    %v645 = stablehlo.rsqrt %v644 : tensor<64x128x28x28xf32>
    %v646 = stablehlo.multiply %v639, %v645 : tensor<64x128x28x28xf32>
    %v647 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v648 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v649 = stablehlo.multiply %v646, %v647 : tensor<64x128x28x28xf32>
    %v650 = stablehlo.add %v649, %v648 : tensor<64x128x28x28xf32>
    %v651 = stablehlo.reshape %v650 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v652 = stablehlo.reshape %v651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v653 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v654 = stablehlo.maximum %v652, %v653 : tensor<64x128x28x28xf32>
    %v655 = stablehlo.reshape %v654 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v657 = stablehlo.convert %v656 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v658 = stablehlo.convert %s2b2W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v659 = stablehlo.convolution(%v657, %v658)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v660 = stablehlo.convert %v659 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v661 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v662 = stablehlo.add %v660, %v661 : tensor<64x512x28x28xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v664 = stablehlo.reshape %v663 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v666 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v667 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v668 = stablehlo.reduce(%v664 init: %v665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v669 = stablehlo.broadcast_in_dim %v668, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v670 = stablehlo.divide %v669, %v666 : tensor<64x512x28x28xf32>
    %v671 = stablehlo.subtract %v664, %v670 : tensor<64x512x28x28xf32>
    %v672 = stablehlo.multiply %v671, %v671 : tensor<64x512x28x28xf32>
    %v673 = stablehlo.reduce(%v672 init: %v665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v674 = stablehlo.broadcast_in_dim %v673, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v675 = stablehlo.divide %v674, %v666 : tensor<64x512x28x28xf32>
    %v676 = stablehlo.add %v675, %v667 : tensor<64x512x28x28xf32>
    %v677 = stablehlo.rsqrt %v676 : tensor<64x512x28x28xf32>
    %v678 = stablehlo.multiply %v671, %v677 : tensor<64x512x28x28xf32>
    %v679 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v680 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v681 = stablehlo.multiply %v678, %v679 : tensor<64x512x28x28xf32>
    %v682 = stablehlo.add %v681, %v680 : tensor<64x512x28x28xf32>
    %v683 = stablehlo.reshape %v682 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v684 = stablehlo.reshape %v683 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v685 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v686 = stablehlo.add %v684, %v685 : tensor<64x128x56x56xf32>
    %v687 = stablehlo.reshape %v686 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v688 = stablehlo.reshape %v687 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v689 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v690 = stablehlo.maximum %v688, %v689 : tensor<64x128x56x56xf32>
    %v691 = stablehlo.reshape %v690 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v692 = stablehlo.reshape %v691 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v693 = stablehlo.convert %v692 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v694 = stablehlo.convert %s2b3W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v695 = stablehlo.convolution(%v693, %v694)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v696 = stablehlo.convert %v695 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v697 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v698 = stablehlo.add %v696, %v697 : tensor<64x128x28x28xf32>
    %v699 = stablehlo.reshape %v698 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v702 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v703 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v704 = stablehlo.reduce(%v700 init: %v701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v705 = stablehlo.broadcast_in_dim %v704, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v706 = stablehlo.divide %v705, %v702 : tensor<64x128x28x28xf32>
    %v707 = stablehlo.subtract %v700, %v706 : tensor<64x128x28x28xf32>
    %v708 = stablehlo.multiply %v707, %v707 : tensor<64x128x28x28xf32>
    %v709 = stablehlo.reduce(%v708 init: %v701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v710 = stablehlo.broadcast_in_dim %v709, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v711 = stablehlo.divide %v710, %v702 : tensor<64x128x28x28xf32>
    %v712 = stablehlo.add %v711, %v703 : tensor<64x128x28x28xf32>
    %v713 = stablehlo.rsqrt %v712 : tensor<64x128x28x28xf32>
    %v714 = stablehlo.multiply %v707, %v713 : tensor<64x128x28x28xf32>
    %v715 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v716 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v717 = stablehlo.multiply %v714, %v715 : tensor<64x128x28x28xf32>
    %v718 = stablehlo.add %v717, %v716 : tensor<64x128x28x28xf32>
    %v719 = stablehlo.reshape %v718 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v720 = stablehlo.reshape %v719 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v721 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v722 = stablehlo.maximum %v720, %v721 : tensor<64x128x28x28xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v724 = stablehlo.reshape %v723 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v725 = stablehlo.convert %v724 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v726 = stablehlo.convert %s2b3W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v727 = stablehlo.convolution(%v725, %v726)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v728 = stablehlo.convert %v727 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v729 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v730 = stablehlo.add %v728, %v729 : tensor<64x128x28x28xf32>
    %v731 = stablehlo.reshape %v730 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v732 = stablehlo.reshape %v731 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v733 = stablehlo.constant dense<0.0> : tensor<f32>
    %v734 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v735 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v736 = stablehlo.reduce(%v732 init: %v733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v737 = stablehlo.broadcast_in_dim %v736, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v738 = stablehlo.divide %v737, %v734 : tensor<64x128x28x28xf32>
    %v739 = stablehlo.subtract %v732, %v738 : tensor<64x128x28x28xf32>
    %v740 = stablehlo.multiply %v739, %v739 : tensor<64x128x28x28xf32>
    %v741 = stablehlo.reduce(%v740 init: %v733) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v742 = stablehlo.broadcast_in_dim %v741, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v743 = stablehlo.divide %v742, %v734 : tensor<64x128x28x28xf32>
    %v744 = stablehlo.add %v743, %v735 : tensor<64x128x28x28xf32>
    %v745 = stablehlo.rsqrt %v744 : tensor<64x128x28x28xf32>
    %v746 = stablehlo.multiply %v739, %v745 : tensor<64x128x28x28xf32>
    %v747 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v748 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v749 = stablehlo.multiply %v746, %v747 : tensor<64x128x28x28xf32>
    %v750 = stablehlo.add %v749, %v748 : tensor<64x128x28x28xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v752 = stablehlo.reshape %v751 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v753 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v754 = stablehlo.maximum %v752, %v753 : tensor<64x128x28x28xf32>
    %v755 = stablehlo.reshape %v754 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v756 = stablehlo.reshape %v755 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v757 = stablehlo.convert %v756 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v758 = stablehlo.convert %s2b3W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v759 = stablehlo.convolution(%v757, %v758)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v760 = stablehlo.convert %v759 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v761 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v762 = stablehlo.add %v760, %v761 : tensor<64x512x28x28xf32>
    %v763 = stablehlo.reshape %v762 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v765 = stablehlo.constant dense<0.0> : tensor<f32>
    %v766 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v767 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v768 = stablehlo.reduce(%v764 init: %v765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v769 = stablehlo.broadcast_in_dim %v768, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v770 = stablehlo.divide %v769, %v766 : tensor<64x512x28x28xf32>
    %v771 = stablehlo.subtract %v764, %v770 : tensor<64x512x28x28xf32>
    %v772 = stablehlo.multiply %v771, %v771 : tensor<64x512x28x28xf32>
    %v773 = stablehlo.reduce(%v772 init: %v765) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v775 = stablehlo.divide %v774, %v766 : tensor<64x512x28x28xf32>
    %v776 = stablehlo.add %v775, %v767 : tensor<64x512x28x28xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<64x512x28x28xf32>
    %v778 = stablehlo.multiply %v771, %v777 : tensor<64x512x28x28xf32>
    %v779 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v780 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v781 = stablehlo.multiply %v778, %v779 : tensor<64x512x28x28xf32>
    %v782 = stablehlo.add %v781, %v780 : tensor<64x512x28x28xf32>
    %v783 = stablehlo.reshape %v782 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v785 = stablehlo.reshape %v691 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v786 = stablehlo.add %v784, %v785 : tensor<64x128x56x56xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v789 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v790 = stablehlo.maximum %v788, %v789 : tensor<64x128x56x56xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v792 = stablehlo.reshape %v791 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v793 = stablehlo.convert %v792 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v794 = stablehlo.convert %s3b0W1 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v795 = stablehlo.convolution(%v793, %v794)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x28x28xbf16>
    %v796 = stablehlo.convert %v795 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v797 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v798 = stablehlo.add %v796, %v797 : tensor<64x256x28x28xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v800 = stablehlo.reshape %v799 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v802 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v803 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v804 = stablehlo.reduce(%v800 init: %v801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v806 = stablehlo.divide %v805, %v802 : tensor<64x256x28x28xf32>
    %v807 = stablehlo.subtract %v800, %v806 : tensor<64x256x28x28xf32>
    %v808 = stablehlo.multiply %v807, %v807 : tensor<64x256x28x28xf32>
    %v809 = stablehlo.reduce(%v808 init: %v801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v810 = stablehlo.broadcast_in_dim %v809, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v811 = stablehlo.divide %v810, %v802 : tensor<64x256x28x28xf32>
    %v812 = stablehlo.add %v811, %v803 : tensor<64x256x28x28xf32>
    %v813 = stablehlo.rsqrt %v812 : tensor<64x256x28x28xf32>
    %v814 = stablehlo.multiply %v807, %v813 : tensor<64x256x28x28xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v816 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v817 = stablehlo.multiply %v814, %v815 : tensor<64x256x28x28xf32>
    %v818 = stablehlo.add %v817, %v816 : tensor<64x256x28x28xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v822 = stablehlo.maximum %v820, %v821 : tensor<64x64x56x56xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v824 = stablehlo.reshape %v823 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v825 = stablehlo.convert %v824 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v826 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v827 = stablehlo.convolution(%v825, %v826)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v828 = stablehlo.convert %v827 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v829 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v830 = stablehlo.add %v828, %v829 : tensor<64x256x14x14xf32>
    %v831 = stablehlo.reshape %v830 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v832 = stablehlo.reshape %v831 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v834 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v835 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v836 = stablehlo.reduce(%v832 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v837 = stablehlo.broadcast_in_dim %v836, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v838 = stablehlo.divide %v837, %v834 : tensor<64x256x14x14xf32>
    %v839 = stablehlo.subtract %v832, %v838 : tensor<64x256x14x14xf32>
    %v840 = stablehlo.multiply %v839, %v839 : tensor<64x256x14x14xf32>
    %v841 = stablehlo.reduce(%v840 init: %v833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v842 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v843 = stablehlo.divide %v842, %v834 : tensor<64x256x14x14xf32>
    %v844 = stablehlo.add %v843, %v835 : tensor<64x256x14x14xf32>
    %v845 = stablehlo.rsqrt %v844 : tensor<64x256x14x14xf32>
    %v846 = stablehlo.multiply %v839, %v845 : tensor<64x256x14x14xf32>
    %v847 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v848 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v849 = stablehlo.multiply %v846, %v847 : tensor<64x256x14x14xf32>
    %v850 = stablehlo.add %v849, %v848 : tensor<64x256x14x14xf32>
    %v851 = stablehlo.reshape %v850 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v853 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v854 = stablehlo.maximum %v852, %v853 : tensor<64x256x14x14xf32>
    %v855 = stablehlo.reshape %v854 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v857 = stablehlo.convert %v856 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v858 = stablehlo.convert %s3b0W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v859 = stablehlo.convolution(%v857, %v858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v860 = stablehlo.convert %v859 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v861 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<64x1024x14x14xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v866 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v867 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v868 = stablehlo.reduce(%v864 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v869 = stablehlo.broadcast_in_dim %v868, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v870 = stablehlo.divide %v869, %v866 : tensor<64x1024x14x14xf32>
    %v871 = stablehlo.subtract %v864, %v870 : tensor<64x1024x14x14xf32>
    %v872 = stablehlo.multiply %v871, %v871 : tensor<64x1024x14x14xf32>
    %v873 = stablehlo.reduce(%v872 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v874 = stablehlo.broadcast_in_dim %v873, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v875 = stablehlo.divide %v874, %v866 : tensor<64x1024x14x14xf32>
    %v876 = stablehlo.add %v875, %v867 : tensor<64x1024x14x14xf32>
    %v877 = stablehlo.rsqrt %v876 : tensor<64x1024x14x14xf32>
    %v878 = stablehlo.multiply %v871, %v877 : tensor<64x1024x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v880 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v881 = stablehlo.multiply %v878, %v879 : tensor<64x1024x14x14xf32>
    %v882 = stablehlo.add %v881, %v880 : tensor<64x1024x14x14xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v884 = stablehlo.reshape %v791 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v885 = stablehlo.convert %v884 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v886 = stablehlo.convert %s3b0Wp : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v887 = stablehlo.convolution(%v885, %v886)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v888 = stablehlo.convert %v887 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v889 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v890 = stablehlo.add %v888, %v889 : tensor<64x1024x14x14xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v892 = stablehlo.reshape %v891 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v894 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v895 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v896 = stablehlo.reduce(%v892 init: %v893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v897 = stablehlo.broadcast_in_dim %v896, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v898 = stablehlo.divide %v897, %v894 : tensor<64x1024x14x14xf32>
    %v899 = stablehlo.subtract %v892, %v898 : tensor<64x1024x14x14xf32>
    %v900 = stablehlo.multiply %v899, %v899 : tensor<64x1024x14x14xf32>
    %v901 = stablehlo.reduce(%v900 init: %v893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v902 = stablehlo.broadcast_in_dim %v901, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v903 = stablehlo.divide %v902, %v894 : tensor<64x1024x14x14xf32>
    %v904 = stablehlo.add %v903, %v895 : tensor<64x1024x14x14xf32>
    %v905 = stablehlo.rsqrt %v904 : tensor<64x1024x14x14xf32>
    %v906 = stablehlo.multiply %v899, %v905 : tensor<64x1024x14x14xf32>
    %v907 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v908 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v909 = stablehlo.multiply %v906, %v907 : tensor<64x1024x14x14xf32>
    %v910 = stablehlo.add %v909, %v908 : tensor<64x1024x14x14xf32>
    %v911 = stablehlo.reshape %v910 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v912 = stablehlo.reshape %v883 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v913 = stablehlo.reshape %v911 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v914 = stablehlo.add %v912, %v913 : tensor<64x64x56x56xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v916 = stablehlo.reshape %v915 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v917 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v918 = stablehlo.maximum %v916, %v917 : tensor<64x64x56x56xf32>
    %v919 = stablehlo.reshape %v918 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v921 = stablehlo.convert %v920 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v922 = stablehlo.convert %s3b1W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v923 = stablehlo.convolution(%v921, %v922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v924 = stablehlo.convert %v923 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v925 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v926 = stablehlo.add %v924, %v925 : tensor<64x256x14x14xf32>
    %v927 = stablehlo.reshape %v926 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v928 = stablehlo.reshape %v927 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v929 = stablehlo.constant dense<0.0> : tensor<f32>
    %v930 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v931 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v932 = stablehlo.reduce(%v928 init: %v929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v933 = stablehlo.broadcast_in_dim %v932, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v934 = stablehlo.divide %v933, %v930 : tensor<64x256x14x14xf32>
    %v935 = stablehlo.subtract %v928, %v934 : tensor<64x256x14x14xf32>
    %v936 = stablehlo.multiply %v935, %v935 : tensor<64x256x14x14xf32>
    %v937 = stablehlo.reduce(%v936 init: %v929) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v938 = stablehlo.broadcast_in_dim %v937, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v939 = stablehlo.divide %v938, %v930 : tensor<64x256x14x14xf32>
    %v940 = stablehlo.add %v939, %v931 : tensor<64x256x14x14xf32>
    %v941 = stablehlo.rsqrt %v940 : tensor<64x256x14x14xf32>
    %v942 = stablehlo.multiply %v935, %v941 : tensor<64x256x14x14xf32>
    %v943 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v944 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v945 = stablehlo.multiply %v942, %v943 : tensor<64x256x14x14xf32>
    %v946 = stablehlo.add %v945, %v944 : tensor<64x256x14x14xf32>
    %v947 = stablehlo.reshape %v946 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v949 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v950 = stablehlo.maximum %v948, %v949 : tensor<64x256x14x14xf32>
    %v951 = stablehlo.reshape %v950 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v952 = stablehlo.reshape %v951 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v953 = stablehlo.convert %v952 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v954 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v955 = stablehlo.convolution(%v953, %v954)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v956 = stablehlo.convert %v955 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v957 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v958 = stablehlo.add %v956, %v957 : tensor<64x256x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v961 = stablehlo.constant dense<0.0> : tensor<f32>
    %v962 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v963 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v964 = stablehlo.reduce(%v960 init: %v961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v965 = stablehlo.broadcast_in_dim %v964, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v966 = stablehlo.divide %v965, %v962 : tensor<64x256x14x14xf32>
    %v967 = stablehlo.subtract %v960, %v966 : tensor<64x256x14x14xf32>
    %v968 = stablehlo.multiply %v967, %v967 : tensor<64x256x14x14xf32>
    %v969 = stablehlo.reduce(%v968 init: %v961) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v970 = stablehlo.broadcast_in_dim %v969, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v971 = stablehlo.divide %v970, %v962 : tensor<64x256x14x14xf32>
    %v972 = stablehlo.add %v971, %v963 : tensor<64x256x14x14xf32>
    %v973 = stablehlo.rsqrt %v972 : tensor<64x256x14x14xf32>
    %v974 = stablehlo.multiply %v967, %v973 : tensor<64x256x14x14xf32>
    %v975 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v976 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v977 = stablehlo.multiply %v974, %v975 : tensor<64x256x14x14xf32>
    %v978 = stablehlo.add %v977, %v976 : tensor<64x256x14x14xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v981 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v982 = stablehlo.maximum %v980, %v981 : tensor<64x256x14x14xf32>
    %v983 = stablehlo.reshape %v982 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v984 = stablehlo.reshape %v983 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v985 = stablehlo.convert %v984 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v986 = stablehlo.convert %s3b1W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v987 = stablehlo.convolution(%v985, %v986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v988 = stablehlo.convert %v987 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v989 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v990 = stablehlo.add %v988, %v989 : tensor<64x1024x14x14xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v992 = stablehlo.reshape %v991 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v994 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v995 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v996 = stablehlo.reduce(%v992 init: %v993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v997 = stablehlo.broadcast_in_dim %v996, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v998 = stablehlo.divide %v997, %v994 : tensor<64x1024x14x14xf32>
    %v999 = stablehlo.subtract %v992, %v998 : tensor<64x1024x14x14xf32>
    %v1000 = stablehlo.multiply %v999, %v999 : tensor<64x1024x14x14xf32>
    %v1001 = stablehlo.reduce(%v1000 init: %v993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1002 = stablehlo.broadcast_in_dim %v1001, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1003 = stablehlo.divide %v1002, %v994 : tensor<64x1024x14x14xf32>
    %v1004 = stablehlo.add %v1003, %v995 : tensor<64x1024x14x14xf32>
    %v1005 = stablehlo.rsqrt %v1004 : tensor<64x1024x14x14xf32>
    %v1006 = stablehlo.multiply %v999, %v1005 : tensor<64x1024x14x14xf32>
    %v1007 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1008 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1009 = stablehlo.multiply %v1006, %v1007 : tensor<64x1024x14x14xf32>
    %v1010 = stablehlo.add %v1009, %v1008 : tensor<64x1024x14x14xf32>
    %v1011 = stablehlo.reshape %v1010 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1013 = stablehlo.reshape %v919 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1014 = stablehlo.add %v1012, %v1013 : tensor<64x64x56x56xf32>
    %v1015 = stablehlo.reshape %v1014 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1016 = stablehlo.reshape %v1015 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1017 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v1018 = stablehlo.maximum %v1016, %v1017 : tensor<64x64x56x56xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1021 = stablehlo.convert %v1020 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1022 = stablehlo.convert %s3b2W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1023 = stablehlo.convolution(%v1021, %v1022)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1024 = stablehlo.convert %v1023 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1025 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1026 = stablehlo.add %v1024, %v1025 : tensor<64x256x14x14xf32>
    %v1027 = stablehlo.reshape %v1026 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1030 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1031 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1032 = stablehlo.reduce(%v1028 init: %v1029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1033 = stablehlo.broadcast_in_dim %v1032, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1034 = stablehlo.divide %v1033, %v1030 : tensor<64x256x14x14xf32>
    %v1035 = stablehlo.subtract %v1028, %v1034 : tensor<64x256x14x14xf32>
    %v1036 = stablehlo.multiply %v1035, %v1035 : tensor<64x256x14x14xf32>
    %v1037 = stablehlo.reduce(%v1036 init: %v1029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1038 = stablehlo.broadcast_in_dim %v1037, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1039 = stablehlo.divide %v1038, %v1030 : tensor<64x256x14x14xf32>
    %v1040 = stablehlo.add %v1039, %v1031 : tensor<64x256x14x14xf32>
    %v1041 = stablehlo.rsqrt %v1040 : tensor<64x256x14x14xf32>
    %v1042 = stablehlo.multiply %v1035, %v1041 : tensor<64x256x14x14xf32>
    %v1043 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1044 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1045 = stablehlo.multiply %v1042, %v1043 : tensor<64x256x14x14xf32>
    %v1046 = stablehlo.add %v1045, %v1044 : tensor<64x256x14x14xf32>
    %v1047 = stablehlo.reshape %v1046 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1048 = stablehlo.reshape %v1047 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1049 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1050 = stablehlo.maximum %v1048, %v1049 : tensor<64x256x14x14xf32>
    %v1051 = stablehlo.reshape %v1050 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1052 = stablehlo.reshape %v1051 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1053 = stablehlo.convert %v1052 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1054 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1055 = stablehlo.convolution(%v1053, %v1054)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1056 = stablehlo.convert %v1055 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1057 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1058 = stablehlo.add %v1056, %v1057 : tensor<64x256x14x14xf32>
    %v1059 = stablehlo.reshape %v1058 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1062 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1063 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1064 = stablehlo.reduce(%v1060 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1065 = stablehlo.broadcast_in_dim %v1064, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1066 = stablehlo.divide %v1065, %v1062 : tensor<64x256x14x14xf32>
    %v1067 = stablehlo.subtract %v1060, %v1066 : tensor<64x256x14x14xf32>
    %v1068 = stablehlo.multiply %v1067, %v1067 : tensor<64x256x14x14xf32>
    %v1069 = stablehlo.reduce(%v1068 init: %v1061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1070 = stablehlo.broadcast_in_dim %v1069, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1071 = stablehlo.divide %v1070, %v1062 : tensor<64x256x14x14xf32>
    %v1072 = stablehlo.add %v1071, %v1063 : tensor<64x256x14x14xf32>
    %v1073 = stablehlo.rsqrt %v1072 : tensor<64x256x14x14xf32>
    %v1074 = stablehlo.multiply %v1067, %v1073 : tensor<64x256x14x14xf32>
    %v1075 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1076 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1077 = stablehlo.multiply %v1074, %v1075 : tensor<64x256x14x14xf32>
    %v1078 = stablehlo.add %v1077, %v1076 : tensor<64x256x14x14xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1081 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1082 = stablehlo.maximum %v1080, %v1081 : tensor<64x256x14x14xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1085 = stablehlo.convert %v1084 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1086 = stablehlo.convert %s3b2W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1087 = stablehlo.convolution(%v1085, %v1086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1088 = stablehlo.convert %v1087 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1089 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1090 = stablehlo.add %v1088, %v1089 : tensor<64x1024x14x14xf32>
    %v1091 = stablehlo.reshape %v1090 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1092 = stablehlo.reshape %v1091 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1094 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1095 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1096 = stablehlo.reduce(%v1092 init: %v1093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1097 = stablehlo.broadcast_in_dim %v1096, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1098 = stablehlo.divide %v1097, %v1094 : tensor<64x1024x14x14xf32>
    %v1099 = stablehlo.subtract %v1092, %v1098 : tensor<64x1024x14x14xf32>
    %v1100 = stablehlo.multiply %v1099, %v1099 : tensor<64x1024x14x14xf32>
    %v1101 = stablehlo.reduce(%v1100 init: %v1093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1102 = stablehlo.broadcast_in_dim %v1101, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1103 = stablehlo.divide %v1102, %v1094 : tensor<64x1024x14x14xf32>
    %v1104 = stablehlo.add %v1103, %v1095 : tensor<64x1024x14x14xf32>
    %v1105 = stablehlo.rsqrt %v1104 : tensor<64x1024x14x14xf32>
    %v1106 = stablehlo.multiply %v1099, %v1105 : tensor<64x1024x14x14xf32>
    %v1107 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1108 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1109 = stablehlo.multiply %v1106, %v1107 : tensor<64x1024x14x14xf32>
    %v1110 = stablehlo.add %v1109, %v1108 : tensor<64x1024x14x14xf32>
    %v1111 = stablehlo.reshape %v1110 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1112 = stablehlo.reshape %v1111 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1113 = stablehlo.reshape %v1019 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<64x64x56x56xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1116 = stablehlo.reshape %v1115 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v1118 = stablehlo.maximum %v1116, %v1117 : tensor<64x64x56x56xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1120 = stablehlo.reshape %v1119 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1121 = stablehlo.convert %v1120 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1122 = stablehlo.convert %s3b3W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1123 = stablehlo.convolution(%v1121, %v1122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1124 = stablehlo.convert %v1123 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1125 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1126 = stablehlo.add %v1124, %v1125 : tensor<64x256x14x14xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1128 = stablehlo.reshape %v1127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1130 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1131 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1132 = stablehlo.reduce(%v1128 init: %v1129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1133 = stablehlo.broadcast_in_dim %v1132, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1134 = stablehlo.divide %v1133, %v1130 : tensor<64x256x14x14xf32>
    %v1135 = stablehlo.subtract %v1128, %v1134 : tensor<64x256x14x14xf32>
    %v1136 = stablehlo.multiply %v1135, %v1135 : tensor<64x256x14x14xf32>
    %v1137 = stablehlo.reduce(%v1136 init: %v1129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1138 = stablehlo.broadcast_in_dim %v1137, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1139 = stablehlo.divide %v1138, %v1130 : tensor<64x256x14x14xf32>
    %v1140 = stablehlo.add %v1139, %v1131 : tensor<64x256x14x14xf32>
    %v1141 = stablehlo.rsqrt %v1140 : tensor<64x256x14x14xf32>
    %v1142 = stablehlo.multiply %v1135, %v1141 : tensor<64x256x14x14xf32>
    %v1143 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1144 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1145 = stablehlo.multiply %v1142, %v1143 : tensor<64x256x14x14xf32>
    %v1146 = stablehlo.add %v1145, %v1144 : tensor<64x256x14x14xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1149 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1150 = stablehlo.maximum %v1148, %v1149 : tensor<64x256x14x14xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1153 = stablehlo.convert %v1152 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1154 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1155 = stablehlo.convolution(%v1153, %v1154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1156 = stablehlo.convert %v1155 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1157 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1158 = stablehlo.add %v1156, %v1157 : tensor<64x256x14x14xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1162 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1163 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1164 = stablehlo.reduce(%v1160 init: %v1161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1165 = stablehlo.broadcast_in_dim %v1164, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1166 = stablehlo.divide %v1165, %v1162 : tensor<64x256x14x14xf32>
    %v1167 = stablehlo.subtract %v1160, %v1166 : tensor<64x256x14x14xf32>
    %v1168 = stablehlo.multiply %v1167, %v1167 : tensor<64x256x14x14xf32>
    %v1169 = stablehlo.reduce(%v1168 init: %v1161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1170 = stablehlo.broadcast_in_dim %v1169, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1171 = stablehlo.divide %v1170, %v1162 : tensor<64x256x14x14xf32>
    %v1172 = stablehlo.add %v1171, %v1163 : tensor<64x256x14x14xf32>
    %v1173 = stablehlo.rsqrt %v1172 : tensor<64x256x14x14xf32>
    %v1174 = stablehlo.multiply %v1167, %v1173 : tensor<64x256x14x14xf32>
    %v1175 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1176 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1177 = stablehlo.multiply %v1174, %v1175 : tensor<64x256x14x14xf32>
    %v1178 = stablehlo.add %v1177, %v1176 : tensor<64x256x14x14xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1180 = stablehlo.reshape %v1179 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1181 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1182 = stablehlo.maximum %v1180, %v1181 : tensor<64x256x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1185 = stablehlo.convert %v1184 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1186 = stablehlo.convert %s3b3W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1187 = stablehlo.convolution(%v1185, %v1186)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1188 = stablehlo.convert %v1187 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1189 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1190 = stablehlo.add %v1188, %v1189 : tensor<64x1024x14x14xf32>
    %v1191 = stablehlo.reshape %v1190 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1192 = stablehlo.reshape %v1191 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1194 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1195 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1196 = stablehlo.reduce(%v1192 init: %v1193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1198 = stablehlo.divide %v1197, %v1194 : tensor<64x1024x14x14xf32>
    %v1199 = stablehlo.subtract %v1192, %v1198 : tensor<64x1024x14x14xf32>
    %v1200 = stablehlo.multiply %v1199, %v1199 : tensor<64x1024x14x14xf32>
    %v1201 = stablehlo.reduce(%v1200 init: %v1193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1202 = stablehlo.broadcast_in_dim %v1201, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1203 = stablehlo.divide %v1202, %v1194 : tensor<64x1024x14x14xf32>
    %v1204 = stablehlo.add %v1203, %v1195 : tensor<64x1024x14x14xf32>
    %v1205 = stablehlo.rsqrt %v1204 : tensor<64x1024x14x14xf32>
    %v1206 = stablehlo.multiply %v1199, %v1205 : tensor<64x1024x14x14xf32>
    %v1207 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1208 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1209 = stablehlo.multiply %v1206, %v1207 : tensor<64x1024x14x14xf32>
    %v1210 = stablehlo.add %v1209, %v1208 : tensor<64x1024x14x14xf32>
    %v1211 = stablehlo.reshape %v1210 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1212 = stablehlo.reshape %v1211 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1213 = stablehlo.reshape %v1119 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1214 = stablehlo.add %v1212, %v1213 : tensor<64x64x56x56xf32>
    %v1215 = stablehlo.reshape %v1214 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1216 = stablehlo.reshape %v1215 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1217 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v1218 = stablehlo.maximum %v1216, %v1217 : tensor<64x64x56x56xf32>
    %v1219 = stablehlo.reshape %v1218 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1220 = stablehlo.reshape %v1219 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1221 = stablehlo.convert %v1220 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1222 = stablehlo.convert %s3b4W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1223 = stablehlo.convolution(%v1221, %v1222)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1224 = stablehlo.convert %v1223 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1225 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1226 = stablehlo.add %v1224, %v1225 : tensor<64x256x14x14xf32>
    %v1227 = stablehlo.reshape %v1226 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1228 = stablehlo.reshape %v1227 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1229 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1230 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1231 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1232 = stablehlo.reduce(%v1228 init: %v1229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1233 = stablehlo.broadcast_in_dim %v1232, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1234 = stablehlo.divide %v1233, %v1230 : tensor<64x256x14x14xf32>
    %v1235 = stablehlo.subtract %v1228, %v1234 : tensor<64x256x14x14xf32>
    %v1236 = stablehlo.multiply %v1235, %v1235 : tensor<64x256x14x14xf32>
    %v1237 = stablehlo.reduce(%v1236 init: %v1229) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1238 = stablehlo.broadcast_in_dim %v1237, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1239 = stablehlo.divide %v1238, %v1230 : tensor<64x256x14x14xf32>
    %v1240 = stablehlo.add %v1239, %v1231 : tensor<64x256x14x14xf32>
    %v1241 = stablehlo.rsqrt %v1240 : tensor<64x256x14x14xf32>
    %v1242 = stablehlo.multiply %v1235, %v1241 : tensor<64x256x14x14xf32>
    %v1243 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1244 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1245 = stablehlo.multiply %v1242, %v1243 : tensor<64x256x14x14xf32>
    %v1246 = stablehlo.add %v1245, %v1244 : tensor<64x256x14x14xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1249 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1250 = stablehlo.maximum %v1248, %v1249 : tensor<64x256x14x14xf32>
    %v1251 = stablehlo.reshape %v1250 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1252 = stablehlo.reshape %v1251 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1253 = stablehlo.convert %v1252 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1254 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1255 = stablehlo.convolution(%v1253, %v1254)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1256 = stablehlo.convert %v1255 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1257 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1258 = stablehlo.add %v1256, %v1257 : tensor<64x256x14x14xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1260 = stablehlo.reshape %v1259 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1262 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1263 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1264 = stablehlo.reduce(%v1260 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1265 = stablehlo.broadcast_in_dim %v1264, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1266 = stablehlo.divide %v1265, %v1262 : tensor<64x256x14x14xf32>
    %v1267 = stablehlo.subtract %v1260, %v1266 : tensor<64x256x14x14xf32>
    %v1268 = stablehlo.multiply %v1267, %v1267 : tensor<64x256x14x14xf32>
    %v1269 = stablehlo.reduce(%v1268 init: %v1261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1270 = stablehlo.broadcast_in_dim %v1269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1271 = stablehlo.divide %v1270, %v1262 : tensor<64x256x14x14xf32>
    %v1272 = stablehlo.add %v1271, %v1263 : tensor<64x256x14x14xf32>
    %v1273 = stablehlo.rsqrt %v1272 : tensor<64x256x14x14xf32>
    %v1274 = stablehlo.multiply %v1267, %v1273 : tensor<64x256x14x14xf32>
    %v1275 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1276 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1277 = stablehlo.multiply %v1274, %v1275 : tensor<64x256x14x14xf32>
    %v1278 = stablehlo.add %v1277, %v1276 : tensor<64x256x14x14xf32>
    %v1279 = stablehlo.reshape %v1278 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1280 = stablehlo.reshape %v1279 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1281 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1282 = stablehlo.maximum %v1280, %v1281 : tensor<64x256x14x14xf32>
    %v1283 = stablehlo.reshape %v1282 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1284 = stablehlo.reshape %v1283 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1285 = stablehlo.convert %v1284 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1286 = stablehlo.convert %s3b4W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1287 = stablehlo.convolution(%v1285, %v1286)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1288 = stablehlo.convert %v1287 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1289 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1290 = stablehlo.add %v1288, %v1289 : tensor<64x1024x14x14xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1292 = stablehlo.reshape %v1291 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1294 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1295 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1296 = stablehlo.reduce(%v1292 init: %v1293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1297 = stablehlo.broadcast_in_dim %v1296, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1298 = stablehlo.divide %v1297, %v1294 : tensor<64x1024x14x14xf32>
    %v1299 = stablehlo.subtract %v1292, %v1298 : tensor<64x1024x14x14xf32>
    %v1300 = stablehlo.multiply %v1299, %v1299 : tensor<64x1024x14x14xf32>
    %v1301 = stablehlo.reduce(%v1300 init: %v1293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1302 = stablehlo.broadcast_in_dim %v1301, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1303 = stablehlo.divide %v1302, %v1294 : tensor<64x1024x14x14xf32>
    %v1304 = stablehlo.add %v1303, %v1295 : tensor<64x1024x14x14xf32>
    %v1305 = stablehlo.rsqrt %v1304 : tensor<64x1024x14x14xf32>
    %v1306 = stablehlo.multiply %v1299, %v1305 : tensor<64x1024x14x14xf32>
    %v1307 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1308 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1309 = stablehlo.multiply %v1306, %v1307 : tensor<64x1024x14x14xf32>
    %v1310 = stablehlo.add %v1309, %v1308 : tensor<64x1024x14x14xf32>
    %v1311 = stablehlo.reshape %v1310 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1312 = stablehlo.reshape %v1311 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1313 = stablehlo.reshape %v1219 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1314 = stablehlo.add %v1312, %v1313 : tensor<64x64x56x56xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1316 = stablehlo.reshape %v1315 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1317 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v1318 = stablehlo.maximum %v1316, %v1317 : tensor<64x64x56x56xf32>
    %v1319 = stablehlo.reshape %v1318 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1320 = stablehlo.reshape %v1319 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1321 = stablehlo.convert %v1320 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1322 = stablehlo.convert %s3b5W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1323 = stablehlo.convolution(%v1321, %v1322)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1324 = stablehlo.convert %v1323 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1325 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1326 = stablehlo.add %v1324, %v1325 : tensor<64x256x14x14xf32>
    %v1327 = stablehlo.reshape %v1326 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1329 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1330 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1331 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1332 = stablehlo.reduce(%v1328 init: %v1329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1333 = stablehlo.broadcast_in_dim %v1332, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1334 = stablehlo.divide %v1333, %v1330 : tensor<64x256x14x14xf32>
    %v1335 = stablehlo.subtract %v1328, %v1334 : tensor<64x256x14x14xf32>
    %v1336 = stablehlo.multiply %v1335, %v1335 : tensor<64x256x14x14xf32>
    %v1337 = stablehlo.reduce(%v1336 init: %v1329) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1338 = stablehlo.broadcast_in_dim %v1337, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1339 = stablehlo.divide %v1338, %v1330 : tensor<64x256x14x14xf32>
    %v1340 = stablehlo.add %v1339, %v1331 : tensor<64x256x14x14xf32>
    %v1341 = stablehlo.rsqrt %v1340 : tensor<64x256x14x14xf32>
    %v1342 = stablehlo.multiply %v1335, %v1341 : tensor<64x256x14x14xf32>
    %v1343 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1344 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1345 = stablehlo.multiply %v1342, %v1343 : tensor<64x256x14x14xf32>
    %v1346 = stablehlo.add %v1345, %v1344 : tensor<64x256x14x14xf32>
    %v1347 = stablehlo.reshape %v1346 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1349 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1350 = stablehlo.maximum %v1348, %v1349 : tensor<64x256x14x14xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1353 = stablehlo.convert %v1352 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1354 = stablehlo.convert %s3b5W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1355 = stablehlo.convolution(%v1353, %v1354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1356 = stablehlo.convert %v1355 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1357 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1358 = stablehlo.add %v1356, %v1357 : tensor<64x256x14x14xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1360 = stablehlo.reshape %v1359 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1362 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1363 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1364 = stablehlo.reduce(%v1360 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1365 = stablehlo.broadcast_in_dim %v1364, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1366 = stablehlo.divide %v1365, %v1362 : tensor<64x256x14x14xf32>
    %v1367 = stablehlo.subtract %v1360, %v1366 : tensor<64x256x14x14xf32>
    %v1368 = stablehlo.multiply %v1367, %v1367 : tensor<64x256x14x14xf32>
    %v1369 = stablehlo.reduce(%v1368 init: %v1361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1370 = stablehlo.broadcast_in_dim %v1369, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1371 = stablehlo.divide %v1370, %v1362 : tensor<64x256x14x14xf32>
    %v1372 = stablehlo.add %v1371, %v1363 : tensor<64x256x14x14xf32>
    %v1373 = stablehlo.rsqrt %v1372 : tensor<64x256x14x14xf32>
    %v1374 = stablehlo.multiply %v1367, %v1373 : tensor<64x256x14x14xf32>
    %v1375 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1376 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1377 = stablehlo.multiply %v1374, %v1375 : tensor<64x256x14x14xf32>
    %v1378 = stablehlo.add %v1377, %v1376 : tensor<64x256x14x14xf32>
    %v1379 = stablehlo.reshape %v1378 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1380 = stablehlo.reshape %v1379 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1382 = stablehlo.maximum %v1380, %v1381 : tensor<64x256x14x14xf32>
    %v1383 = stablehlo.reshape %v1382 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1384 = stablehlo.reshape %v1383 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1385 = stablehlo.convert %v1384 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1386 = stablehlo.convert %s3b5W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1387 = stablehlo.convolution(%v1385, %v1386)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1388 = stablehlo.convert %v1387 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1389 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1390 = stablehlo.add %v1388, %v1389 : tensor<64x1024x14x14xf32>
    %v1391 = stablehlo.reshape %v1390 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1395 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1396 = stablehlo.reduce(%v1392 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1397 = stablehlo.broadcast_in_dim %v1396, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1398 = stablehlo.divide %v1397, %v1394 : tensor<64x1024x14x14xf32>
    %v1399 = stablehlo.subtract %v1392, %v1398 : tensor<64x1024x14x14xf32>
    %v1400 = stablehlo.multiply %v1399, %v1399 : tensor<64x1024x14x14xf32>
    %v1401 = stablehlo.reduce(%v1400 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1402 = stablehlo.broadcast_in_dim %v1401, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1403 = stablehlo.divide %v1402, %v1394 : tensor<64x1024x14x14xf32>
    %v1404 = stablehlo.add %v1403, %v1395 : tensor<64x1024x14x14xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<64x1024x14x14xf32>
    %v1406 = stablehlo.multiply %v1399, %v1405 : tensor<64x1024x14x14xf32>
    %v1407 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1408 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1409 = stablehlo.multiply %v1406, %v1407 : tensor<64x1024x14x14xf32>
    %v1410 = stablehlo.add %v1409, %v1408 : tensor<64x1024x14x14xf32>
    %v1411 = stablehlo.reshape %v1410 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1412 = stablehlo.reshape %v1411 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1413 = stablehlo.reshape %v1319 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1414 = stablehlo.add %v1412, %v1413 : tensor<64x64x56x56xf32>
    %v1415 = stablehlo.reshape %v1414 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v1417 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v1418 = stablehlo.maximum %v1416, %v1417 : tensor<64x64x56x56xf32>
    %v1419 = stablehlo.reshape %v1418 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1421 = stablehlo.convert %v1420 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1422 = stablehlo.convert %s4b0W1 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v1423 = stablehlo.convolution(%v1421, %v1422)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x14x14xbf16>
    %v1424 = stablehlo.convert %v1423 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v1425 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1426 = stablehlo.add %v1424, %v1425 : tensor<64x512x14x14xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1430 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v1431 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v1432 = stablehlo.reduce(%v1428 init: %v1429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1433 = stablehlo.broadcast_in_dim %v1432, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1434 = stablehlo.divide %v1433, %v1430 : tensor<64x512x14x14xf32>
    %v1435 = stablehlo.subtract %v1428, %v1434 : tensor<64x512x14x14xf32>
    %v1436 = stablehlo.multiply %v1435, %v1435 : tensor<64x512x14x14xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1439 = stablehlo.divide %v1438, %v1430 : tensor<64x512x14x14xf32>
    %v1440 = stablehlo.add %v1439, %v1431 : tensor<64x512x14x14xf32>
    %v1441 = stablehlo.rsqrt %v1440 : tensor<64x512x14x14xf32>
    %v1442 = stablehlo.multiply %v1435, %v1441 : tensor<64x512x14x14xf32>
    %v1443 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1444 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1445 = stablehlo.multiply %v1442, %v1443 : tensor<64x512x14x14xf32>
    %v1446 = stablehlo.add %v1445, %v1444 : tensor<64x512x14x14xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1448 = stablehlo.reshape %v1447 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1449 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v1450 = stablehlo.maximum %v1448, %v1449 : tensor<64x128x28x28xf32>
    %v1451 = stablehlo.reshape %v1450 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1453 = stablehlo.convert %v1452 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v1454 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1455 = stablehlo.convolution(%v1453, %v1454)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1456 = stablehlo.convert %v1455 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1457 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1458 = stablehlo.add %v1456, %v1457 : tensor<64x512x7x7xf32>
    %v1459 = stablehlo.reshape %v1458 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1460 = stablehlo.reshape %v1459 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1461 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1462 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1463 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1464 = stablehlo.reduce(%v1460 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1465 = stablehlo.broadcast_in_dim %v1464, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1466 = stablehlo.divide %v1465, %v1462 : tensor<64x512x7x7xf32>
    %v1467 = stablehlo.subtract %v1460, %v1466 : tensor<64x512x7x7xf32>
    %v1468 = stablehlo.multiply %v1467, %v1467 : tensor<64x512x7x7xf32>
    %v1469 = stablehlo.reduce(%v1468 init: %v1461) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1470 = stablehlo.broadcast_in_dim %v1469, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1471 = stablehlo.divide %v1470, %v1462 : tensor<64x512x7x7xf32>
    %v1472 = stablehlo.add %v1471, %v1463 : tensor<64x512x7x7xf32>
    %v1473 = stablehlo.rsqrt %v1472 : tensor<64x512x7x7xf32>
    %v1474 = stablehlo.multiply %v1467, %v1473 : tensor<64x512x7x7xf32>
    %v1475 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1476 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1474, %v1475 : tensor<64x512x7x7xf32>
    %v1478 = stablehlo.add %v1477, %v1476 : tensor<64x512x7x7xf32>
    %v1479 = stablehlo.reshape %v1478 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1481 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1482 = stablehlo.maximum %v1480, %v1481 : tensor<64x512x7x7xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1484 = stablehlo.reshape %v1483 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1485 = stablehlo.convert %v1484 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1486 = stablehlo.convert %s4b0W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1487 = stablehlo.convolution(%v1485, %v1486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1488 = stablehlo.convert %v1487 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1489 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1490 = stablehlo.add %v1488, %v1489 : tensor<64x2048x7x7xf32>
    %v1491 = stablehlo.reshape %v1490 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1492 = stablehlo.reshape %v1491 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1494 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1495 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1496 = stablehlo.reduce(%v1492 init: %v1493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1497 = stablehlo.broadcast_in_dim %v1496, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1498 = stablehlo.divide %v1497, %v1494 : tensor<64x2048x7x7xf32>
    %v1499 = stablehlo.subtract %v1492, %v1498 : tensor<64x2048x7x7xf32>
    %v1500 = stablehlo.multiply %v1499, %v1499 : tensor<64x2048x7x7xf32>
    %v1501 = stablehlo.reduce(%v1500 init: %v1493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1494 : tensor<64x2048x7x7xf32>
    %v1504 = stablehlo.add %v1503, %v1495 : tensor<64x2048x7x7xf32>
    %v1505 = stablehlo.rsqrt %v1504 : tensor<64x2048x7x7xf32>
    %v1506 = stablehlo.multiply %v1499, %v1505 : tensor<64x2048x7x7xf32>
    %v1507 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1508 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1509 = stablehlo.multiply %v1506, %v1507 : tensor<64x2048x7x7xf32>
    %v1510 = stablehlo.add %v1509, %v1508 : tensor<64x2048x7x7xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1512 = stablehlo.reshape %v1419 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1513 = stablehlo.convert %v1512 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1514 = stablehlo.convert %s4b0Wp : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xbf16>
    %v1515 = stablehlo.convolution(%v1513, %v1514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1516 = stablehlo.convert %v1515 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1517 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1518 = stablehlo.add %v1516, %v1517 : tensor<64x2048x7x7xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1520 = stablehlo.reshape %v1519 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1521 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1522 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1523 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1524 = stablehlo.reduce(%v1520 init: %v1521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1525 = stablehlo.broadcast_in_dim %v1524, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1526 = stablehlo.divide %v1525, %v1522 : tensor<64x2048x7x7xf32>
    %v1527 = stablehlo.subtract %v1520, %v1526 : tensor<64x2048x7x7xf32>
    %v1528 = stablehlo.multiply %v1527, %v1527 : tensor<64x2048x7x7xf32>
    %v1529 = stablehlo.reduce(%v1528 init: %v1521) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1530 = stablehlo.broadcast_in_dim %v1529, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1531 = stablehlo.divide %v1530, %v1522 : tensor<64x2048x7x7xf32>
    %v1532 = stablehlo.add %v1531, %v1523 : tensor<64x2048x7x7xf32>
    %v1533 = stablehlo.rsqrt %v1532 : tensor<64x2048x7x7xf32>
    %v1534 = stablehlo.multiply %v1527, %v1533 : tensor<64x2048x7x7xf32>
    %v1535 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1536 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1537 = stablehlo.multiply %v1534, %v1535 : tensor<64x2048x7x7xf32>
    %v1538 = stablehlo.add %v1537, %v1536 : tensor<64x2048x7x7xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1540 = stablehlo.reshape %v1511 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1541 = stablehlo.reshape %v1539 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1542 = stablehlo.add %v1540, %v1541 : tensor<64x128x28x28xf32>
    %v1543 = stablehlo.reshape %v1542 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1544 = stablehlo.reshape %v1543 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1545 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v1546 = stablehlo.maximum %v1544, %v1545 : tensor<64x128x28x28xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1549 = stablehlo.convert %v1548 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1550 = stablehlo.convert %s4b1W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1551 = stablehlo.convolution(%v1549, %v1550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1552 = stablehlo.convert %v1551 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1553 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1554 = stablehlo.add %v1552, %v1553 : tensor<64x512x7x7xf32>
    %v1555 = stablehlo.reshape %v1554 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1556 = stablehlo.reshape %v1555 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1558 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1559 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1560 = stablehlo.reduce(%v1556 init: %v1557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1561 = stablehlo.broadcast_in_dim %v1560, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1562 = stablehlo.divide %v1561, %v1558 : tensor<64x512x7x7xf32>
    %v1563 = stablehlo.subtract %v1556, %v1562 : tensor<64x512x7x7xf32>
    %v1564 = stablehlo.multiply %v1563, %v1563 : tensor<64x512x7x7xf32>
    %v1565 = stablehlo.reduce(%v1564 init: %v1557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1566 = stablehlo.broadcast_in_dim %v1565, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1567 = stablehlo.divide %v1566, %v1558 : tensor<64x512x7x7xf32>
    %v1568 = stablehlo.add %v1567, %v1559 : tensor<64x512x7x7xf32>
    %v1569 = stablehlo.rsqrt %v1568 : tensor<64x512x7x7xf32>
    %v1570 = stablehlo.multiply %v1563, %v1569 : tensor<64x512x7x7xf32>
    %v1571 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1572 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1573 = stablehlo.multiply %v1570, %v1571 : tensor<64x512x7x7xf32>
    %v1574 = stablehlo.add %v1573, %v1572 : tensor<64x512x7x7xf32>
    %v1575 = stablehlo.reshape %v1574 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1577 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1578 = stablehlo.maximum %v1576, %v1577 : tensor<64x512x7x7xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1581 = stablehlo.convert %v1580 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1582 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1583 = stablehlo.convolution(%v1581, %v1582)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1584 = stablehlo.convert %v1583 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1585 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1586 = stablehlo.add %v1584, %v1585 : tensor<64x512x7x7xf32>
    %v1587 = stablehlo.reshape %v1586 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1588 = stablehlo.reshape %v1587 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1589 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1590 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1591 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1592 = stablehlo.reduce(%v1588 init: %v1589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1593 = stablehlo.broadcast_in_dim %v1592, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1594 = stablehlo.divide %v1593, %v1590 : tensor<64x512x7x7xf32>
    %v1595 = stablehlo.subtract %v1588, %v1594 : tensor<64x512x7x7xf32>
    %v1596 = stablehlo.multiply %v1595, %v1595 : tensor<64x512x7x7xf32>
    %v1597 = stablehlo.reduce(%v1596 init: %v1589) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1598 = stablehlo.broadcast_in_dim %v1597, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1599 = stablehlo.divide %v1598, %v1590 : tensor<64x512x7x7xf32>
    %v1600 = stablehlo.add %v1599, %v1591 : tensor<64x512x7x7xf32>
    %v1601 = stablehlo.rsqrt %v1600 : tensor<64x512x7x7xf32>
    %v1602 = stablehlo.multiply %v1595, %v1601 : tensor<64x512x7x7xf32>
    %v1603 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1604 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1605 = stablehlo.multiply %v1602, %v1603 : tensor<64x512x7x7xf32>
    %v1606 = stablehlo.add %v1605, %v1604 : tensor<64x512x7x7xf32>
    %v1607 = stablehlo.reshape %v1606 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1608 = stablehlo.reshape %v1607 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1609 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1610 = stablehlo.maximum %v1608, %v1609 : tensor<64x512x7x7xf32>
    %v1611 = stablehlo.reshape %v1610 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1612 = stablehlo.reshape %v1611 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1613 = stablehlo.convert %v1612 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1614 = stablehlo.convert %s4b1W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1615 = stablehlo.convolution(%v1613, %v1614)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1616 = stablehlo.convert %v1615 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1617 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1618 = stablehlo.add %v1616, %v1617 : tensor<64x2048x7x7xf32>
    %v1619 = stablehlo.reshape %v1618 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1620 = stablehlo.reshape %v1619 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1622 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1623 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1624 = stablehlo.reduce(%v1620 init: %v1621) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1625 = stablehlo.broadcast_in_dim %v1624, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1626 = stablehlo.divide %v1625, %v1622 : tensor<64x2048x7x7xf32>
    %v1627 = stablehlo.subtract %v1620, %v1626 : tensor<64x2048x7x7xf32>
    %v1628 = stablehlo.multiply %v1627, %v1627 : tensor<64x2048x7x7xf32>
    %v1629 = stablehlo.reduce(%v1628 init: %v1621) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1630 = stablehlo.broadcast_in_dim %v1629, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1631 = stablehlo.divide %v1630, %v1622 : tensor<64x2048x7x7xf32>
    %v1632 = stablehlo.add %v1631, %v1623 : tensor<64x2048x7x7xf32>
    %v1633 = stablehlo.rsqrt %v1632 : tensor<64x2048x7x7xf32>
    %v1634 = stablehlo.multiply %v1627, %v1633 : tensor<64x2048x7x7xf32>
    %v1635 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1636 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1637 = stablehlo.multiply %v1634, %v1635 : tensor<64x2048x7x7xf32>
    %v1638 = stablehlo.add %v1637, %v1636 : tensor<64x2048x7x7xf32>
    %v1639 = stablehlo.reshape %v1638 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1640 = stablehlo.reshape %v1639 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1641 = stablehlo.reshape %v1547 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1642 = stablehlo.add %v1640, %v1641 : tensor<64x128x28x28xf32>
    %v1643 = stablehlo.reshape %v1642 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1644 = stablehlo.reshape %v1643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1645 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v1646 = stablehlo.maximum %v1644, %v1645 : tensor<64x128x28x28xf32>
    %v1647 = stablehlo.reshape %v1646 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1648 = stablehlo.reshape %v1647 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1649 = stablehlo.convert %v1648 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1650 = stablehlo.convert %s4b2W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1651 = stablehlo.convolution(%v1649, %v1650)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1652 = stablehlo.convert %v1651 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1653 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1654 = stablehlo.add %v1652, %v1653 : tensor<64x512x7x7xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1656 = stablehlo.reshape %v1655 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1657 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1658 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1659 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1660 = stablehlo.reduce(%v1656 init: %v1657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1661 = stablehlo.broadcast_in_dim %v1660, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1662 = stablehlo.divide %v1661, %v1658 : tensor<64x512x7x7xf32>
    %v1663 = stablehlo.subtract %v1656, %v1662 : tensor<64x512x7x7xf32>
    %v1664 = stablehlo.multiply %v1663, %v1663 : tensor<64x512x7x7xf32>
    %v1665 = stablehlo.reduce(%v1664 init: %v1657) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1666 = stablehlo.broadcast_in_dim %v1665, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1667 = stablehlo.divide %v1666, %v1658 : tensor<64x512x7x7xf32>
    %v1668 = stablehlo.add %v1667, %v1659 : tensor<64x512x7x7xf32>
    %v1669 = stablehlo.rsqrt %v1668 : tensor<64x512x7x7xf32>
    %v1670 = stablehlo.multiply %v1663, %v1669 : tensor<64x512x7x7xf32>
    %v1671 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1672 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1673 = stablehlo.multiply %v1670, %v1671 : tensor<64x512x7x7xf32>
    %v1674 = stablehlo.add %v1673, %v1672 : tensor<64x512x7x7xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1677 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1678 = stablehlo.maximum %v1676, %v1677 : tensor<64x512x7x7xf32>
    %v1679 = stablehlo.reshape %v1678 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1680 = stablehlo.reshape %v1679 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1681 = stablehlo.convert %v1680 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1682 = stablehlo.convert %s4b2W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1683 = stablehlo.convolution(%v1681, %v1682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1684 = stablehlo.convert %v1683 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1685 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1686 = stablehlo.add %v1684, %v1685 : tensor<64x512x7x7xf32>
    %v1687 = stablehlo.reshape %v1686 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1690 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1691 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1692 = stablehlo.reduce(%v1688 init: %v1689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1693 = stablehlo.broadcast_in_dim %v1692, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1694 = stablehlo.divide %v1693, %v1690 : tensor<64x512x7x7xf32>
    %v1695 = stablehlo.subtract %v1688, %v1694 : tensor<64x512x7x7xf32>
    %v1696 = stablehlo.multiply %v1695, %v1695 : tensor<64x512x7x7xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1699 = stablehlo.divide %v1698, %v1690 : tensor<64x512x7x7xf32>
    %v1700 = stablehlo.add %v1699, %v1691 : tensor<64x512x7x7xf32>
    %v1701 = stablehlo.rsqrt %v1700 : tensor<64x512x7x7xf32>
    %v1702 = stablehlo.multiply %v1695, %v1701 : tensor<64x512x7x7xf32>
    %v1703 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1704 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1705 = stablehlo.multiply %v1702, %v1703 : tensor<64x512x7x7xf32>
    %v1706 = stablehlo.add %v1705, %v1704 : tensor<64x512x7x7xf32>
    %v1707 = stablehlo.reshape %v1706 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1709 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1710 = stablehlo.maximum %v1708, %v1709 : tensor<64x512x7x7xf32>
    %v1711 = stablehlo.reshape %v1710 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1712 = stablehlo.reshape %v1711 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1713 = stablehlo.convert %v1712 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1714 = stablehlo.convert %s4b2W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1715 = stablehlo.convolution(%v1713, %v1714)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1716 = stablehlo.convert %v1715 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1717 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1718 = stablehlo.add %v1716, %v1717 : tensor<64x2048x7x7xf32>
    %v1719 = stablehlo.reshape %v1718 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1720 = stablehlo.reshape %v1719 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1722 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1723 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1724 = stablehlo.reduce(%v1720 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1726 = stablehlo.divide %v1725, %v1722 : tensor<64x2048x7x7xf32>
    %v1727 = stablehlo.subtract %v1720, %v1726 : tensor<64x2048x7x7xf32>
    %v1728 = stablehlo.multiply %v1727, %v1727 : tensor<64x2048x7x7xf32>
    %v1729 = stablehlo.reduce(%v1728 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1729, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1731 = stablehlo.divide %v1730, %v1722 : tensor<64x2048x7x7xf32>
    %v1732 = stablehlo.add %v1731, %v1723 : tensor<64x2048x7x7xf32>
    %v1733 = stablehlo.rsqrt %v1732 : tensor<64x2048x7x7xf32>
    %v1734 = stablehlo.multiply %v1727, %v1733 : tensor<64x2048x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1736 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1737 = stablehlo.multiply %v1734, %v1735 : tensor<64x2048x7x7xf32>
    %v1738 = stablehlo.add %v1737, %v1736 : tensor<64x2048x7x7xf32>
    %v1739 = stablehlo.reshape %v1738 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1740 = stablehlo.reshape %v1739 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1741 = stablehlo.reshape %v1647 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1742 = stablehlo.add %v1740, %v1741 : tensor<64x128x28x28xf32>
    %v1743 = stablehlo.reshape %v1742 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1744 = stablehlo.reshape %v1743 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1745 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v1746 = stablehlo.maximum %v1744, %v1745 : tensor<64x128x28x28xf32>
    %v1747 = stablehlo.reshape %v1746 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1748 = stablehlo.reshape %v1747 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1750 = stablehlo.reduce(%v1748 init: %v1749) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048xf32>
    %v1751 = stablehlo.constant dense<49.0> : tensor<64x2048xf32>
    %v1752 = stablehlo.divide %v1750, %v1751 : tensor<64x2048xf32>
    %v1753 = stablehlo.dot_general %v1752, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<2048x1000xf32>) -> tensor<64x1000xf32>
    %v1754 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1755 = stablehlo.add %v1753, %v1754 : tensor<64x1000xf32>
    %v1756 = stablehlo.reshape %v1755 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1758 = stablehlo.exponential %v1756 : tensor<64x1x1000xf32>
    %v1759 = stablehlo.reduce(%v1758 init: %v1757) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1760 = stablehlo.broadcast_in_dim %v1759, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v1761 = stablehlo.divide %v1758, %v1760 : tensor<64x1x1000xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v1763 = stablehlo.subtract %v1762, %onehot : tensor<64x1000xf32>
    %v1764 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v1765 = stablehlo.multiply %onehot, %v1764 : tensor<64x1000xf32>
    %v1766 = stablehlo.add %v1763, %v1765 : tensor<64x1000xf32>
    %v1767 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v1768 = stablehlo.add %v1766, %v1767 : tensor<64x1000xf32>
    %v1769 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v1770 = stablehlo.divide %v1768, %v1769 : tensor<64x1000xf32>
    %v1771 = stablehlo.reshape %v1770 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1772 = stablehlo.dot_general %v1771, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<64x1x2048xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<64x1x2048xf32>) -> tensor<64x2048xf32>
    %v1774 = stablehlo.dot_general %v1752, %v1770, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<64x1000xf32>) -> tensor<2048x1000xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1776 = stablehlo.reduce(%v1770 init: %v1775) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1777 = stablehlo.broadcast_in_dim %v1773, dims = [0, 1] : (tensor<64x2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1778 = stablehlo.constant dense<49.0> : tensor<64x2048x7x7xf32>
    %v1779 = stablehlo.divide %v1777, %v1778 : tensor<64x2048x7x7xf32>
    %v1780 = stablehlo.reshape %v1779 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1781 = stablehlo.reshape %v1780 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1782 = stablehlo.reshape %v1743 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1783 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v1784 = stablehlo.compare GT, %v1782, %v1783 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v1785 = stablehlo.select %v1784, %v1781, %v1783 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v1786 = stablehlo.reshape %v1785 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1787 = stablehlo.reshape %v1719 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1789 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1790 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1791 = stablehlo.reduce(%v1787 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1792 = stablehlo.broadcast_in_dim %v1791, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1793 = stablehlo.divide %v1792, %v1789 : tensor<64x2048x7x7xf32>
    %v1794 = stablehlo.subtract %v1787, %v1793 : tensor<64x2048x7x7xf32>
    %v1795 = stablehlo.multiply %v1794, %v1794 : tensor<64x2048x7x7xf32>
    %v1796 = stablehlo.reduce(%v1795 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1798 = stablehlo.divide %v1797, %v1789 : tensor<64x2048x7x7xf32>
    %v1799 = stablehlo.add %v1798, %v1790 : tensor<64x2048x7x7xf32>
    %v1800 = stablehlo.rsqrt %v1799 : tensor<64x2048x7x7xf32>
    %v1801 = stablehlo.multiply %v1794, %v1800 : tensor<64x2048x7x7xf32>
    %v1802 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1803 = stablehlo.reshape %v1786 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1804 = stablehlo.multiply %v1802, %v1803 : tensor<64x2048x7x7xf32>
    %v1805 = stablehlo.reduce(%v1804 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1806 = stablehlo.broadcast_in_dim %v1805, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1807 = stablehlo.multiply %v1801, %v1804 : tensor<64x2048x7x7xf32>
    %v1808 = stablehlo.reduce(%v1807 init: %v1788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1809 = stablehlo.broadcast_in_dim %v1808, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1810 = stablehlo.multiply %v1804, %v1789 : tensor<64x2048x7x7xf32>
    %v1811 = stablehlo.subtract %v1810, %v1806 : tensor<64x2048x7x7xf32>
    %v1812 = stablehlo.multiply %v1801, %v1809 : tensor<64x2048x7x7xf32>
    %v1813 = stablehlo.subtract %v1811, %v1812 : tensor<64x2048x7x7xf32>
    %v1814 = stablehlo.divide %v1800, %v1789 : tensor<64x2048x7x7xf32>
    %v1815 = stablehlo.multiply %v1814, %v1813 : tensor<64x2048x7x7xf32>
    %v1816 = stablehlo.reshape %v1815 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1817 = stablehlo.reshape %v1816 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1818 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1819 = stablehlo.transpose %v1818, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1820 = stablehlo.convert %v1817 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1821 = stablehlo.convert %v1819 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1822 = stablehlo.convolution(%v1820, %v1821)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1823 = stablehlo.convert %v1822 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1824 = stablehlo.reshape %v1823 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1825 = stablehlo.reshape %v1824 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1826 = stablehlo.reshape %v1707 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1827 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1828 = stablehlo.compare GT, %v1826, %v1827 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1829 = stablehlo.select %v1828, %v1825, %v1827 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1830 = stablehlo.reshape %v1829 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1831 = stablehlo.reshape %v1687 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1833 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1834 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1835 = stablehlo.reduce(%v1831 init: %v1832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1836 = stablehlo.broadcast_in_dim %v1835, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1837 = stablehlo.divide %v1836, %v1833 : tensor<64x512x7x7xf32>
    %v1838 = stablehlo.subtract %v1831, %v1837 : tensor<64x512x7x7xf32>
    %v1839 = stablehlo.multiply %v1838, %v1838 : tensor<64x512x7x7xf32>
    %v1840 = stablehlo.reduce(%v1839 init: %v1832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1841 = stablehlo.broadcast_in_dim %v1840, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1842 = stablehlo.divide %v1841, %v1833 : tensor<64x512x7x7xf32>
    %v1843 = stablehlo.add %v1842, %v1834 : tensor<64x512x7x7xf32>
    %v1844 = stablehlo.rsqrt %v1843 : tensor<64x512x7x7xf32>
    %v1845 = stablehlo.multiply %v1838, %v1844 : tensor<64x512x7x7xf32>
    %v1846 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1847 = stablehlo.reshape %v1830 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1848 = stablehlo.multiply %v1846, %v1847 : tensor<64x512x7x7xf32>
    %v1849 = stablehlo.reduce(%v1848 init: %v1832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1850 = stablehlo.broadcast_in_dim %v1849, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1851 = stablehlo.multiply %v1845, %v1848 : tensor<64x512x7x7xf32>
    %v1852 = stablehlo.reduce(%v1851 init: %v1832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1853 = stablehlo.broadcast_in_dim %v1852, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1854 = stablehlo.multiply %v1848, %v1833 : tensor<64x512x7x7xf32>
    %v1855 = stablehlo.subtract %v1854, %v1850 : tensor<64x512x7x7xf32>
    %v1856 = stablehlo.multiply %v1845, %v1853 : tensor<64x512x7x7xf32>
    %v1857 = stablehlo.subtract %v1855, %v1856 : tensor<64x512x7x7xf32>
    %v1858 = stablehlo.divide %v1844, %v1833 : tensor<64x512x7x7xf32>
    %v1859 = stablehlo.multiply %v1858, %v1857 : tensor<64x512x7x7xf32>
    %v1860 = stablehlo.reshape %v1859 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1861 = stablehlo.reshape %v1860 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1862 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1863 = stablehlo.transpose %v1862, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1864 = stablehlo.convert %v1861 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1865 = stablehlo.convert %v1863 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1866 = stablehlo.convolution(%v1864, %v1865)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1867 = stablehlo.convert %v1866 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1868 = stablehlo.reshape %v1867 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1869 = stablehlo.reshape %v1868 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1870 = stablehlo.reshape %v1675 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1871 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1872 = stablehlo.compare GT, %v1870, %v1871 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1873 = stablehlo.select %v1872, %v1869, %v1871 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1874 = stablehlo.reshape %v1873 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1875 = stablehlo.reshape %v1655 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1876 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1877 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1878 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1879 = stablehlo.reduce(%v1875 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1880 = stablehlo.broadcast_in_dim %v1879, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1881 = stablehlo.divide %v1880, %v1877 : tensor<64x512x7x7xf32>
    %v1882 = stablehlo.subtract %v1875, %v1881 : tensor<64x512x7x7xf32>
    %v1883 = stablehlo.multiply %v1882, %v1882 : tensor<64x512x7x7xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1885 = stablehlo.broadcast_in_dim %v1884, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1886 = stablehlo.divide %v1885, %v1877 : tensor<64x512x7x7xf32>
    %v1887 = stablehlo.add %v1886, %v1878 : tensor<64x512x7x7xf32>
    %v1888 = stablehlo.rsqrt %v1887 : tensor<64x512x7x7xf32>
    %v1889 = stablehlo.multiply %v1882, %v1888 : tensor<64x512x7x7xf32>
    %v1890 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1891 = stablehlo.reshape %v1874 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1892 = stablehlo.multiply %v1890, %v1891 : tensor<64x512x7x7xf32>
    %v1893 = stablehlo.reduce(%v1892 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1894 = stablehlo.broadcast_in_dim %v1893, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1895 = stablehlo.multiply %v1889, %v1892 : tensor<64x512x7x7xf32>
    %v1896 = stablehlo.reduce(%v1895 init: %v1876) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1897 = stablehlo.broadcast_in_dim %v1896, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1898 = stablehlo.multiply %v1892, %v1877 : tensor<64x512x7x7xf32>
    %v1899 = stablehlo.subtract %v1898, %v1894 : tensor<64x512x7x7xf32>
    %v1900 = stablehlo.multiply %v1889, %v1897 : tensor<64x512x7x7xf32>
    %v1901 = stablehlo.subtract %v1899, %v1900 : tensor<64x512x7x7xf32>
    %v1902 = stablehlo.divide %v1888, %v1877 : tensor<64x512x7x7xf32>
    %v1903 = stablehlo.multiply %v1902, %v1901 : tensor<64x512x7x7xf32>
    %v1904 = stablehlo.reshape %v1903 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1905 = stablehlo.reshape %v1904 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1906 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1907 = stablehlo.transpose %v1906, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1908 = stablehlo.convert %v1905 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1909 = stablehlo.convert %v1907 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1910 = stablehlo.convolution(%v1908, %v1909)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1911 = stablehlo.convert %v1910 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1912 = stablehlo.reshape %v1911 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1913 = stablehlo.reshape %v1912 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1914 = stablehlo.reshape %v1786 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1915 = stablehlo.add %v1913, %v1914 : tensor<64x128x28x28xf32>
    %v1916 = stablehlo.reshape %v1915 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1917 = stablehlo.reshape %v1647 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1918 = stablehlo.reshape %v1904 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1919 = stablehlo.transpose %v1917, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v1920 = stablehlo.transpose %v1918, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1921 = stablehlo.convert %v1919 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v1922 = stablehlo.convert %v1920 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1923 = stablehlo.convolution(%v1921, %v1922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v1924 = stablehlo.convert %v1923 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v1925 = stablehlo.transpose %v1924, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1926 = stablehlo.reshape %v1655 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1928 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1929 = stablehlo.reduce(%v1926 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1930 = stablehlo.broadcast_in_dim %v1929, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1931 = stablehlo.divide %v1930, %v1928 : tensor<64x512x7x7xf32>
    %v1932 = stablehlo.subtract %v1926, %v1931 : tensor<64x512x7x7xf32>
    %v1933 = stablehlo.multiply %v1932, %v1932 : tensor<64x512x7x7xf32>
    %v1934 = stablehlo.reduce(%v1933 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1936 = stablehlo.divide %v1935, %v1928 : tensor<64x512x7x7xf32>
    %v1937 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1938 = stablehlo.add %v1936, %v1937 : tensor<64x512x7x7xf32>
    %v1939 = stablehlo.rsqrt %v1938 : tensor<64x512x7x7xf32>
    %v1940 = stablehlo.multiply %v1932, %v1939 : tensor<64x512x7x7xf32>
    %v1941 = stablehlo.reshape %v1874 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1942 = stablehlo.multiply %v1941, %v1940 : tensor<64x512x7x7xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1944 = stablehlo.reshape %v1874 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1947 = stablehlo.reshape %v1679 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1948 = stablehlo.reshape %v1860 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1949 = stablehlo.transpose %v1947, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1950 = stablehlo.transpose %v1948, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1951 = stablehlo.convert %v1949 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1952 = stablehlo.convert %v1950 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1953 = stablehlo.convolution(%v1951, %v1952)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1954 = stablehlo.convert %v1953 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1955 = stablehlo.transpose %v1954, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1956 = stablehlo.reshape %v1687 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1957 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1958 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1959 = stablehlo.reduce(%v1956 init: %v1957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1959, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1961 = stablehlo.divide %v1960, %v1958 : tensor<64x512x7x7xf32>
    %v1962 = stablehlo.subtract %v1956, %v1961 : tensor<64x512x7x7xf32>
    %v1963 = stablehlo.multiply %v1962, %v1962 : tensor<64x512x7x7xf32>
    %v1964 = stablehlo.reduce(%v1963 init: %v1957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1965 = stablehlo.broadcast_in_dim %v1964, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1966 = stablehlo.divide %v1965, %v1958 : tensor<64x512x7x7xf32>
    %v1967 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1968 = stablehlo.add %v1966, %v1967 : tensor<64x512x7x7xf32>
    %v1969 = stablehlo.rsqrt %v1968 : tensor<64x512x7x7xf32>
    %v1970 = stablehlo.multiply %v1962, %v1969 : tensor<64x512x7x7xf32>
    %v1971 = stablehlo.reshape %v1830 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1972 = stablehlo.multiply %v1971, %v1970 : tensor<64x512x7x7xf32>
    %v1973 = stablehlo.reduce(%v1972 init: %v1957) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1974 = stablehlo.reshape %v1830 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1976 = stablehlo.reduce(%v1974 init: %v1975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1977 = stablehlo.reshape %v1711 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1978 = stablehlo.reshape %v1816 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1979 = stablehlo.transpose %v1977, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1980 = stablehlo.transpose %v1978, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v1981 = stablehlo.convert %v1979 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1982 = stablehlo.convert %v1980 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v1983 = stablehlo.convolution(%v1981, %v1982)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v1984 = stablehlo.convert %v1983 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v1985 = stablehlo.transpose %v1984, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1986 = stablehlo.reshape %v1719 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1987 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1988 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1989 = stablehlo.reduce(%v1986 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1990 = stablehlo.broadcast_in_dim %v1989, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1991 = stablehlo.divide %v1990, %v1988 : tensor<64x2048x7x7xf32>
    %v1992 = stablehlo.subtract %v1986, %v1991 : tensor<64x2048x7x7xf32>
    %v1993 = stablehlo.multiply %v1992, %v1992 : tensor<64x2048x7x7xf32>
    %v1994 = stablehlo.reduce(%v1993 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1995 = stablehlo.broadcast_in_dim %v1994, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1996 = stablehlo.divide %v1995, %v1988 : tensor<64x2048x7x7xf32>
    %v1997 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1998 = stablehlo.add %v1996, %v1997 : tensor<64x2048x7x7xf32>
    %v1999 = stablehlo.rsqrt %v1998 : tensor<64x2048x7x7xf32>
    %v2000 = stablehlo.multiply %v1992, %v1999 : tensor<64x2048x7x7xf32>
    %v2001 = stablehlo.reshape %v1786 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2002 = stablehlo.multiply %v2001, %v2000 : tensor<64x2048x7x7xf32>
    %v2003 = stablehlo.reduce(%v2002 init: %v1987) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2004 = stablehlo.reshape %v1786 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2006 = stablehlo.reduce(%v2004 init: %v2005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2007 = stablehlo.reshape %v1916 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2008 = stablehlo.reshape %v1643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2009 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v2010 = stablehlo.compare GT, %v2008, %v2009 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v2011 = stablehlo.select %v2010, %v2007, %v2009 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v2012 = stablehlo.reshape %v2011 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2013 = stablehlo.reshape %v1619 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2014 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2015 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2016 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2017 = stablehlo.reduce(%v2013 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2018 = stablehlo.broadcast_in_dim %v2017, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2019 = stablehlo.divide %v2018, %v2015 : tensor<64x2048x7x7xf32>
    %v2020 = stablehlo.subtract %v2013, %v2019 : tensor<64x2048x7x7xf32>
    %v2021 = stablehlo.multiply %v2020, %v2020 : tensor<64x2048x7x7xf32>
    %v2022 = stablehlo.reduce(%v2021 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2023 = stablehlo.broadcast_in_dim %v2022, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2024 = stablehlo.divide %v2023, %v2015 : tensor<64x2048x7x7xf32>
    %v2025 = stablehlo.add %v2024, %v2016 : tensor<64x2048x7x7xf32>
    %v2026 = stablehlo.rsqrt %v2025 : tensor<64x2048x7x7xf32>
    %v2027 = stablehlo.multiply %v2020, %v2026 : tensor<64x2048x7x7xf32>
    %v2028 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2029 = stablehlo.reshape %v2012 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2030 = stablehlo.multiply %v2028, %v2029 : tensor<64x2048x7x7xf32>
    %v2031 = stablehlo.reduce(%v2030 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2032 = stablehlo.broadcast_in_dim %v2031, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2033 = stablehlo.multiply %v2027, %v2030 : tensor<64x2048x7x7xf32>
    %v2034 = stablehlo.reduce(%v2033 init: %v2014) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2035 = stablehlo.broadcast_in_dim %v2034, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2036 = stablehlo.multiply %v2030, %v2015 : tensor<64x2048x7x7xf32>
    %v2037 = stablehlo.subtract %v2036, %v2032 : tensor<64x2048x7x7xf32>
    %v2038 = stablehlo.multiply %v2027, %v2035 : tensor<64x2048x7x7xf32>
    %v2039 = stablehlo.subtract %v2037, %v2038 : tensor<64x2048x7x7xf32>
    %v2040 = stablehlo.divide %v2026, %v2015 : tensor<64x2048x7x7xf32>
    %v2041 = stablehlo.multiply %v2040, %v2039 : tensor<64x2048x7x7xf32>
    %v2042 = stablehlo.reshape %v2041 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2043 = stablehlo.reshape %v2042 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2044 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2045 = stablehlo.transpose %v2044, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2046 = stablehlo.convert %v2043 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2047 = stablehlo.convert %v2045 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2048 = stablehlo.convolution(%v2046, %v2047)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2049 = stablehlo.convert %v2048 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2050 = stablehlo.reshape %v2049 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2051 = stablehlo.reshape %v2050 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2052 = stablehlo.reshape %v1607 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2053 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2054 = stablehlo.compare GT, %v2052, %v2053 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2055 = stablehlo.select %v2054, %v2051, %v2053 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2056 = stablehlo.reshape %v2055 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2057 = stablehlo.reshape %v1587 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2059 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2060 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2061 = stablehlo.reduce(%v2057 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2062 = stablehlo.broadcast_in_dim %v2061, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2063 = stablehlo.divide %v2062, %v2059 : tensor<64x512x7x7xf32>
    %v2064 = stablehlo.subtract %v2057, %v2063 : tensor<64x512x7x7xf32>
    %v2065 = stablehlo.multiply %v2064, %v2064 : tensor<64x512x7x7xf32>
    %v2066 = stablehlo.reduce(%v2065 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2067 = stablehlo.broadcast_in_dim %v2066, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2068 = stablehlo.divide %v2067, %v2059 : tensor<64x512x7x7xf32>
    %v2069 = stablehlo.add %v2068, %v2060 : tensor<64x512x7x7xf32>
    %v2070 = stablehlo.rsqrt %v2069 : tensor<64x512x7x7xf32>
    %v2071 = stablehlo.multiply %v2064, %v2070 : tensor<64x512x7x7xf32>
    %v2072 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2073 = stablehlo.reshape %v2056 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2074 = stablehlo.multiply %v2072, %v2073 : tensor<64x512x7x7xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2077 = stablehlo.multiply %v2071, %v2074 : tensor<64x512x7x7xf32>
    %v2078 = stablehlo.reduce(%v2077 init: %v2058) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2079 = stablehlo.broadcast_in_dim %v2078, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2080 = stablehlo.multiply %v2074, %v2059 : tensor<64x512x7x7xf32>
    %v2081 = stablehlo.subtract %v2080, %v2076 : tensor<64x512x7x7xf32>
    %v2082 = stablehlo.multiply %v2071, %v2079 : tensor<64x512x7x7xf32>
    %v2083 = stablehlo.subtract %v2081, %v2082 : tensor<64x512x7x7xf32>
    %v2084 = stablehlo.divide %v2070, %v2059 : tensor<64x512x7x7xf32>
    %v2085 = stablehlo.multiply %v2084, %v2083 : tensor<64x512x7x7xf32>
    %v2086 = stablehlo.reshape %v2085 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2087 = stablehlo.reshape %v2086 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2088 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2089 = stablehlo.transpose %v2088, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2090 = stablehlo.convert %v2087 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2091 = stablehlo.convert %v2089 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2092 = stablehlo.convolution(%v2090, %v2091)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2093 = stablehlo.convert %v2092 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2096 = stablehlo.reshape %v1575 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2097 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2098 = stablehlo.compare GT, %v2096, %v2097 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2099 = stablehlo.select %v2098, %v2095, %v2097 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2100 = stablehlo.reshape %v2099 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2101 = stablehlo.reshape %v1555 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2103 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2104 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2105 = stablehlo.reduce(%v2101 init: %v2102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2106 = stablehlo.broadcast_in_dim %v2105, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2107 = stablehlo.divide %v2106, %v2103 : tensor<64x512x7x7xf32>
    %v2108 = stablehlo.subtract %v2101, %v2107 : tensor<64x512x7x7xf32>
    %v2109 = stablehlo.multiply %v2108, %v2108 : tensor<64x512x7x7xf32>
    %v2110 = stablehlo.reduce(%v2109 init: %v2102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2111 = stablehlo.broadcast_in_dim %v2110, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2112 = stablehlo.divide %v2111, %v2103 : tensor<64x512x7x7xf32>
    %v2113 = stablehlo.add %v2112, %v2104 : tensor<64x512x7x7xf32>
    %v2114 = stablehlo.rsqrt %v2113 : tensor<64x512x7x7xf32>
    %v2115 = stablehlo.multiply %v2108, %v2114 : tensor<64x512x7x7xf32>
    %v2116 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2117 = stablehlo.reshape %v2100 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2118 = stablehlo.multiply %v2116, %v2117 : tensor<64x512x7x7xf32>
    %v2119 = stablehlo.reduce(%v2118 init: %v2102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2120 = stablehlo.broadcast_in_dim %v2119, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2121 = stablehlo.multiply %v2115, %v2118 : tensor<64x512x7x7xf32>
    %v2122 = stablehlo.reduce(%v2121 init: %v2102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2123 = stablehlo.broadcast_in_dim %v2122, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2124 = stablehlo.multiply %v2118, %v2103 : tensor<64x512x7x7xf32>
    %v2125 = stablehlo.subtract %v2124, %v2120 : tensor<64x512x7x7xf32>
    %v2126 = stablehlo.multiply %v2115, %v2123 : tensor<64x512x7x7xf32>
    %v2127 = stablehlo.subtract %v2125, %v2126 : tensor<64x512x7x7xf32>
    %v2128 = stablehlo.divide %v2114, %v2103 : tensor<64x512x7x7xf32>
    %v2129 = stablehlo.multiply %v2128, %v2127 : tensor<64x512x7x7xf32>
    %v2130 = stablehlo.reshape %v2129 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2131 = stablehlo.reshape %v2130 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2132 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v2133 = stablehlo.transpose %v2132, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2134 = stablehlo.convert %v2131 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2135 = stablehlo.convert %v2133 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2136 = stablehlo.convolution(%v2134, %v2135)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2137 = stablehlo.convert %v2136 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2138 = stablehlo.reshape %v2137 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2139 = stablehlo.reshape %v2138 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2140 = stablehlo.reshape %v2012 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2141 = stablehlo.add %v2139, %v2140 : tensor<64x128x28x28xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2143 = stablehlo.reshape %v1547 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2144 = stablehlo.reshape %v2130 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2145 = stablehlo.transpose %v2143, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2146 = stablehlo.transpose %v2144, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2147 = stablehlo.convert %v2145 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2148 = stablehlo.convert %v2146 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2149 = stablehlo.convolution(%v2147, %v2148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v2150 = stablehlo.convert %v2149 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v2151 = stablehlo.transpose %v2150, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2152 = stablehlo.reshape %v1555 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2154 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2155 = stablehlo.reduce(%v2152 init: %v2153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2156 = stablehlo.broadcast_in_dim %v2155, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2157 = stablehlo.divide %v2156, %v2154 : tensor<64x512x7x7xf32>
    %v2158 = stablehlo.subtract %v2152, %v2157 : tensor<64x512x7x7xf32>
    %v2159 = stablehlo.multiply %v2158, %v2158 : tensor<64x512x7x7xf32>
    %v2160 = stablehlo.reduce(%v2159 init: %v2153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2161 = stablehlo.broadcast_in_dim %v2160, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2162 = stablehlo.divide %v2161, %v2154 : tensor<64x512x7x7xf32>
    %v2163 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2164 = stablehlo.add %v2162, %v2163 : tensor<64x512x7x7xf32>
    %v2165 = stablehlo.rsqrt %v2164 : tensor<64x512x7x7xf32>
    %v2166 = stablehlo.multiply %v2158, %v2165 : tensor<64x512x7x7xf32>
    %v2167 = stablehlo.reshape %v2100 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2168 = stablehlo.multiply %v2167, %v2166 : tensor<64x512x7x7xf32>
    %v2169 = stablehlo.reduce(%v2168 init: %v2153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2170 = stablehlo.reshape %v2100 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2172 = stablehlo.reduce(%v2170 init: %v2171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2173 = stablehlo.reshape %v1579 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2174 = stablehlo.reshape %v2086 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2175 = stablehlo.transpose %v2173, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2176 = stablehlo.transpose %v2174, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2177 = stablehlo.convert %v2175 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2178 = stablehlo.convert %v2176 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2179 = stablehlo.convolution(%v2177, %v2178)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2180 = stablehlo.convert %v2179 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2181 = stablehlo.transpose %v2180, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2182 = stablehlo.reshape %v1587 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2184 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2185 = stablehlo.reduce(%v2182 init: %v2183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2186 = stablehlo.broadcast_in_dim %v2185, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2187 = stablehlo.divide %v2186, %v2184 : tensor<64x512x7x7xf32>
    %v2188 = stablehlo.subtract %v2182, %v2187 : tensor<64x512x7x7xf32>
    %v2189 = stablehlo.multiply %v2188, %v2188 : tensor<64x512x7x7xf32>
    %v2190 = stablehlo.reduce(%v2189 init: %v2183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2191 = stablehlo.broadcast_in_dim %v2190, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2192 = stablehlo.divide %v2191, %v2184 : tensor<64x512x7x7xf32>
    %v2193 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2194 = stablehlo.add %v2192, %v2193 : tensor<64x512x7x7xf32>
    %v2195 = stablehlo.rsqrt %v2194 : tensor<64x512x7x7xf32>
    %v2196 = stablehlo.multiply %v2188, %v2195 : tensor<64x512x7x7xf32>
    %v2197 = stablehlo.reshape %v2056 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2198 = stablehlo.multiply %v2197, %v2196 : tensor<64x512x7x7xf32>
    %v2199 = stablehlo.reduce(%v2198 init: %v2183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2200 = stablehlo.reshape %v2056 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2201 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2202 = stablehlo.reduce(%v2200 init: %v2201) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2203 = stablehlo.reshape %v1611 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2204 = stablehlo.reshape %v2042 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2205 = stablehlo.transpose %v2203, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2206 = stablehlo.transpose %v2204, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2207 = stablehlo.convert %v2205 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2208 = stablehlo.convert %v2206 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2209 = stablehlo.convolution(%v2207, %v2208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2210 = stablehlo.convert %v2209 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2211 = stablehlo.transpose %v2210, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2212 = stablehlo.reshape %v1619 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2214 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2215 = stablehlo.reduce(%v2212 init: %v2213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2216 = stablehlo.broadcast_in_dim %v2215, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2217 = stablehlo.divide %v2216, %v2214 : tensor<64x2048x7x7xf32>
    %v2218 = stablehlo.subtract %v2212, %v2217 : tensor<64x2048x7x7xf32>
    %v2219 = stablehlo.multiply %v2218, %v2218 : tensor<64x2048x7x7xf32>
    %v2220 = stablehlo.reduce(%v2219 init: %v2213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2221 = stablehlo.broadcast_in_dim %v2220, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2222 = stablehlo.divide %v2221, %v2214 : tensor<64x2048x7x7xf32>
    %v2223 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2224 = stablehlo.add %v2222, %v2223 : tensor<64x2048x7x7xf32>
    %v2225 = stablehlo.rsqrt %v2224 : tensor<64x2048x7x7xf32>
    %v2226 = stablehlo.multiply %v2218, %v2225 : tensor<64x2048x7x7xf32>
    %v2227 = stablehlo.reshape %v2012 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2228 = stablehlo.multiply %v2227, %v2226 : tensor<64x2048x7x7xf32>
    %v2229 = stablehlo.reduce(%v2228 init: %v2213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2230 = stablehlo.reshape %v2012 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2232 = stablehlo.reduce(%v2230 init: %v2231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2233 = stablehlo.reshape %v2142 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2234 = stablehlo.reshape %v1543 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2235 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v2236 = stablehlo.compare GT, %v2234, %v2235 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v2237 = stablehlo.select %v2236, %v2233, %v2235 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v2238 = stablehlo.reshape %v2237 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2239 = stablehlo.reshape %v1491 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2240 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2241 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2242 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2243 = stablehlo.reduce(%v2239 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2244 = stablehlo.broadcast_in_dim %v2243, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2245 = stablehlo.divide %v2244, %v2241 : tensor<64x2048x7x7xf32>
    %v2246 = stablehlo.subtract %v2239, %v2245 : tensor<64x2048x7x7xf32>
    %v2247 = stablehlo.multiply %v2246, %v2246 : tensor<64x2048x7x7xf32>
    %v2248 = stablehlo.reduce(%v2247 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2249 = stablehlo.broadcast_in_dim %v2248, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2250 = stablehlo.divide %v2249, %v2241 : tensor<64x2048x7x7xf32>
    %v2251 = stablehlo.add %v2250, %v2242 : tensor<64x2048x7x7xf32>
    %v2252 = stablehlo.rsqrt %v2251 : tensor<64x2048x7x7xf32>
    %v2253 = stablehlo.multiply %v2246, %v2252 : tensor<64x2048x7x7xf32>
    %v2254 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2255 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2256 = stablehlo.multiply %v2254, %v2255 : tensor<64x2048x7x7xf32>
    %v2257 = stablehlo.reduce(%v2256 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2258 = stablehlo.broadcast_in_dim %v2257, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2259 = stablehlo.multiply %v2253, %v2256 : tensor<64x2048x7x7xf32>
    %v2260 = stablehlo.reduce(%v2259 init: %v2240) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2261 = stablehlo.broadcast_in_dim %v2260, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2262 = stablehlo.multiply %v2256, %v2241 : tensor<64x2048x7x7xf32>
    %v2263 = stablehlo.subtract %v2262, %v2258 : tensor<64x2048x7x7xf32>
    %v2264 = stablehlo.multiply %v2253, %v2261 : tensor<64x2048x7x7xf32>
    %v2265 = stablehlo.subtract %v2263, %v2264 : tensor<64x2048x7x7xf32>
    %v2266 = stablehlo.divide %v2252, %v2241 : tensor<64x2048x7x7xf32>
    %v2267 = stablehlo.multiply %v2266, %v2265 : tensor<64x2048x7x7xf32>
    %v2268 = stablehlo.reshape %v2267 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2269 = stablehlo.reshape %v2268 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2270 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2271 = stablehlo.transpose %v2270, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2272 = stablehlo.convert %v2269 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2273 = stablehlo.convert %v2271 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2274 = stablehlo.convolution(%v2272, %v2273)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2275 = stablehlo.convert %v2274 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2276 = stablehlo.reshape %v2275 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2278 = stablehlo.reshape %v1479 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2279 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2280 = stablehlo.compare GT, %v2278, %v2279 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2281 = stablehlo.select %v2280, %v2277, %v2279 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2282 = stablehlo.reshape %v2281 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2283 = stablehlo.reshape %v1459 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2285 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2286 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2287 = stablehlo.reduce(%v2283 init: %v2284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2288 = stablehlo.broadcast_in_dim %v2287, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2289 = stablehlo.divide %v2288, %v2285 : tensor<64x512x7x7xf32>
    %v2290 = stablehlo.subtract %v2283, %v2289 : tensor<64x512x7x7xf32>
    %v2291 = stablehlo.multiply %v2290, %v2290 : tensor<64x512x7x7xf32>
    %v2292 = stablehlo.reduce(%v2291 init: %v2284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2293 = stablehlo.broadcast_in_dim %v2292, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2294 = stablehlo.divide %v2293, %v2285 : tensor<64x512x7x7xf32>
    %v2295 = stablehlo.add %v2294, %v2286 : tensor<64x512x7x7xf32>
    %v2296 = stablehlo.rsqrt %v2295 : tensor<64x512x7x7xf32>
    %v2297 = stablehlo.multiply %v2290, %v2296 : tensor<64x512x7x7xf32>
    %v2298 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2299 = stablehlo.reshape %v2282 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2300 = stablehlo.multiply %v2298, %v2299 : tensor<64x512x7x7xf32>
    %v2301 = stablehlo.reduce(%v2300 init: %v2284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2302 = stablehlo.broadcast_in_dim %v2301, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2303 = stablehlo.multiply %v2297, %v2300 : tensor<64x512x7x7xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2306 = stablehlo.multiply %v2300, %v2285 : tensor<64x512x7x7xf32>
    %v2307 = stablehlo.subtract %v2306, %v2302 : tensor<64x512x7x7xf32>
    %v2308 = stablehlo.multiply %v2297, %v2305 : tensor<64x512x7x7xf32>
    %v2309 = stablehlo.subtract %v2307, %v2308 : tensor<64x512x7x7xf32>
    %v2310 = stablehlo.divide %v2296, %v2285 : tensor<64x512x7x7xf32>
    %v2311 = stablehlo.multiply %v2310, %v2309 : tensor<64x512x7x7xf32>
    %v2312 = stablehlo.reshape %v2311 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2313 = stablehlo.reshape %v2312 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2314 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2315 = stablehlo.pad %v2313, %v2314, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2316 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2317 = stablehlo.transpose %v2316, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2318 = stablehlo.convert %v2315 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2319 = stablehlo.convert %v2317 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2320 = stablehlo.convolution(%v2318, %v2319)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x14x14xbf16>
    %v2321 = stablehlo.convert %v2320 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v2322 = stablehlo.reshape %v2321 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2323 = stablehlo.reshape %v2322 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2324 = stablehlo.reshape %v1447 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2325 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v2326 = stablehlo.compare GT, %v2324, %v2325 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v2327 = stablehlo.select %v2326, %v2323, %v2325 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v2328 = stablehlo.reshape %v2327 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2329 = stablehlo.reshape %v1427 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2330 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2331 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v2332 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2333 = stablehlo.reduce(%v2329 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2333, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2335 = stablehlo.divide %v2334, %v2331 : tensor<64x512x14x14xf32>
    %v2336 = stablehlo.subtract %v2329, %v2335 : tensor<64x512x14x14xf32>
    %v2337 = stablehlo.multiply %v2336, %v2336 : tensor<64x512x14x14xf32>
    %v2338 = stablehlo.reduce(%v2337 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2339 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2340 = stablehlo.divide %v2339, %v2331 : tensor<64x512x14x14xf32>
    %v2341 = stablehlo.add %v2340, %v2332 : tensor<64x512x14x14xf32>
    %v2342 = stablehlo.rsqrt %v2341 : tensor<64x512x14x14xf32>
    %v2343 = stablehlo.multiply %v2336, %v2342 : tensor<64x512x14x14xf32>
    %v2344 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2345 = stablehlo.reshape %v2328 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2346 = stablehlo.multiply %v2344, %v2345 : tensor<64x512x14x14xf32>
    %v2347 = stablehlo.reduce(%v2346 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2348 = stablehlo.broadcast_in_dim %v2347, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2349 = stablehlo.multiply %v2343, %v2346 : tensor<64x512x14x14xf32>
    %v2350 = stablehlo.reduce(%v2349 init: %v2330) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2351 = stablehlo.broadcast_in_dim %v2350, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2352 = stablehlo.multiply %v2346, %v2331 : tensor<64x512x14x14xf32>
    %v2353 = stablehlo.subtract %v2352, %v2348 : tensor<64x512x14x14xf32>
    %v2354 = stablehlo.multiply %v2343, %v2351 : tensor<64x512x14x14xf32>
    %v2355 = stablehlo.subtract %v2353, %v2354 : tensor<64x512x14x14xf32>
    %v2356 = stablehlo.divide %v2342, %v2331 : tensor<64x512x14x14xf32>
    %v2357 = stablehlo.multiply %v2356, %v2355 : tensor<64x512x14x14xf32>
    %v2358 = stablehlo.reshape %v2357 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2359 = stablehlo.reshape %v2358 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2360 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v2361 = stablehlo.transpose %v2360, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v2362 = stablehlo.convert %v2359 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2363 = stablehlo.convert %v2361 : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v2364 = stablehlo.convolution(%v2362, %v2363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2365 = stablehlo.convert %v2364 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2366 = stablehlo.reshape %v2365 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2367 = stablehlo.reshape %v1519 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2369 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2370 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2371 = stablehlo.reduce(%v2367 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2372 = stablehlo.broadcast_in_dim %v2371, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2373 = stablehlo.divide %v2372, %v2369 : tensor<64x2048x7x7xf32>
    %v2374 = stablehlo.subtract %v2367, %v2373 : tensor<64x2048x7x7xf32>
    %v2375 = stablehlo.multiply %v2374, %v2374 : tensor<64x2048x7x7xf32>
    %v2376 = stablehlo.reduce(%v2375 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2377 = stablehlo.broadcast_in_dim %v2376, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2378 = stablehlo.divide %v2377, %v2369 : tensor<64x2048x7x7xf32>
    %v2379 = stablehlo.add %v2378, %v2370 : tensor<64x2048x7x7xf32>
    %v2380 = stablehlo.rsqrt %v2379 : tensor<64x2048x7x7xf32>
    %v2381 = stablehlo.multiply %v2374, %v2380 : tensor<64x2048x7x7xf32>
    %v2382 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2383 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2384 = stablehlo.multiply %v2382, %v2383 : tensor<64x2048x7x7xf32>
    %v2385 = stablehlo.reduce(%v2384 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2386 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2387 = stablehlo.multiply %v2381, %v2384 : tensor<64x2048x7x7xf32>
    %v2388 = stablehlo.reduce(%v2387 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2390 = stablehlo.multiply %v2384, %v2369 : tensor<64x2048x7x7xf32>
    %v2391 = stablehlo.subtract %v2390, %v2386 : tensor<64x2048x7x7xf32>
    %v2392 = stablehlo.multiply %v2381, %v2389 : tensor<64x2048x7x7xf32>
    %v2393 = stablehlo.subtract %v2391, %v2392 : tensor<64x2048x7x7xf32>
    %v2394 = stablehlo.divide %v2380, %v2369 : tensor<64x2048x7x7xf32>
    %v2395 = stablehlo.multiply %v2394, %v2393 : tensor<64x2048x7x7xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2397 = stablehlo.reshape %v2396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2399 = stablehlo.pad %v2397, %v2398, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v2400 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2401 = stablehlo.transpose %v2400, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2402 = stablehlo.convert %v2399 : (tensor<64x2048x14x14xf32>) -> tensor<64x2048x14x14xbf16>
    %v2403 = stablehlo.convert %v2401 : (tensor<1024x2048x1x1xf32>) -> tensor<1024x2048x1x1xbf16>
    %v2404 = stablehlo.convolution(%v2402, %v2403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x14x14xbf16>, tensor<1024x2048x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2405 = stablehlo.convert %v2404 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2407 = stablehlo.reshape %v2366 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2408 = stablehlo.reshape %v2406 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2409 = stablehlo.add %v2407, %v2408 : tensor<64x64x56x56xf32>
    %v2410 = stablehlo.reshape %v2409 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2411 = stablehlo.reshape %v1419 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2412 = stablehlo.reshape %v2358 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2413 = stablehlo.transpose %v2411, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2414 = stablehlo.transpose %v2412, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2415 = stablehlo.convert %v2413 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2416 = stablehlo.convert %v2414 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2417 = stablehlo.convolution(%v2415, %v2416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<1024x512x1x1xbf16>
    %v2418 = stablehlo.convert %v2417 : (tensor<1024x512x1x1xbf16>) -> tensor<1024x512x1x1xf32>
    %v2419 = stablehlo.transpose %v2418, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2420 = stablehlo.reshape %v1427 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2422 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v2423 = stablehlo.reduce(%v2420 init: %v2421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2424 = stablehlo.broadcast_in_dim %v2423, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2425 = stablehlo.divide %v2424, %v2422 : tensor<64x512x14x14xf32>
    %v2426 = stablehlo.subtract %v2420, %v2425 : tensor<64x512x14x14xf32>
    %v2427 = stablehlo.multiply %v2426, %v2426 : tensor<64x512x14x14xf32>
    %v2428 = stablehlo.reduce(%v2427 init: %v2421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2429 = stablehlo.broadcast_in_dim %v2428, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2430 = stablehlo.divide %v2429, %v2422 : tensor<64x512x14x14xf32>
    %v2431 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2432 = stablehlo.add %v2430, %v2431 : tensor<64x512x14x14xf32>
    %v2433 = stablehlo.rsqrt %v2432 : tensor<64x512x14x14xf32>
    %v2434 = stablehlo.multiply %v2426, %v2433 : tensor<64x512x14x14xf32>
    %v2435 = stablehlo.reshape %v2328 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2436 = stablehlo.multiply %v2435, %v2434 : tensor<64x512x14x14xf32>
    %v2437 = stablehlo.reduce(%v2436 init: %v2421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2438 = stablehlo.reshape %v2328 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2440 = stablehlo.reduce(%v2438 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2441 = stablehlo.reshape %v1451 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2442 = stablehlo.reshape %v2312 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2444 = stablehlo.pad %v2442, %v2443, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2445 = stablehlo.transpose %v2441, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2446 = stablehlo.transpose %v2444, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2447 = stablehlo.convert %v2445 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2448 = stablehlo.convert %v2446 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2449 = stablehlo.convolution(%v2447, %v2448)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<512x512x3x3xbf16>
    %v2450 = stablehlo.convert %v2449 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2451 = stablehlo.transpose %v2450, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2452 = stablehlo.reshape %v1459 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2453 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2454 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2455 = stablehlo.reduce(%v2452 init: %v2453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2456 = stablehlo.broadcast_in_dim %v2455, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2457 = stablehlo.divide %v2456, %v2454 : tensor<64x512x7x7xf32>
    %v2458 = stablehlo.subtract %v2452, %v2457 : tensor<64x512x7x7xf32>
    %v2459 = stablehlo.multiply %v2458, %v2458 : tensor<64x512x7x7xf32>
    %v2460 = stablehlo.reduce(%v2459 init: %v2453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2461 = stablehlo.broadcast_in_dim %v2460, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2462 = stablehlo.divide %v2461, %v2454 : tensor<64x512x7x7xf32>
    %v2463 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2464 = stablehlo.add %v2462, %v2463 : tensor<64x512x7x7xf32>
    %v2465 = stablehlo.rsqrt %v2464 : tensor<64x512x7x7xf32>
    %v2466 = stablehlo.multiply %v2458, %v2465 : tensor<64x512x7x7xf32>
    %v2467 = stablehlo.reshape %v2282 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2468 = stablehlo.multiply %v2467, %v2466 : tensor<64x512x7x7xf32>
    %v2469 = stablehlo.reduce(%v2468 init: %v2453) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2470 = stablehlo.reshape %v2282 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2472 = stablehlo.reduce(%v2470 init: %v2471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2473 = stablehlo.reshape %v1483 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2474 = stablehlo.reshape %v2268 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2475 = stablehlo.transpose %v2473, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2476 = stablehlo.transpose %v2474, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2477 = stablehlo.convert %v2475 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2478 = stablehlo.convert %v2476 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2479 = stablehlo.convolution(%v2477, %v2478)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2480 = stablehlo.convert %v2479 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2481 = stablehlo.transpose %v2480, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2482 = stablehlo.reshape %v1491 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2483 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2484 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2485 = stablehlo.reduce(%v2482 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2486 = stablehlo.broadcast_in_dim %v2485, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2487 = stablehlo.divide %v2486, %v2484 : tensor<64x2048x7x7xf32>
    %v2488 = stablehlo.subtract %v2482, %v2487 : tensor<64x2048x7x7xf32>
    %v2489 = stablehlo.multiply %v2488, %v2488 : tensor<64x2048x7x7xf32>
    %v2490 = stablehlo.reduce(%v2489 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2491 = stablehlo.broadcast_in_dim %v2490, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2492 = stablehlo.divide %v2491, %v2484 : tensor<64x2048x7x7xf32>
    %v2493 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2494 = stablehlo.add %v2492, %v2493 : tensor<64x2048x7x7xf32>
    %v2495 = stablehlo.rsqrt %v2494 : tensor<64x2048x7x7xf32>
    %v2496 = stablehlo.multiply %v2488, %v2495 : tensor<64x2048x7x7xf32>
    %v2497 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2498 = stablehlo.multiply %v2497, %v2496 : tensor<64x2048x7x7xf32>
    %v2499 = stablehlo.reduce(%v2498 init: %v2483) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2500 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2501 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2502 = stablehlo.reduce(%v2500 init: %v2501) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2503 = stablehlo.reshape %v1419 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2504 = stablehlo.reshape %v2396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2506 = stablehlo.pad %v2504, %v2505, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v2507 = stablehlo.transpose %v2503, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2508 = stablehlo.transpose %v2506, dims = [1, 0, 2, 3] : (tensor<64x2048x14x14xf32>) -> tensor<2048x64x14x14xf32>
    %v2509 = stablehlo.convert %v2507 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2510 = stablehlo.convert %v2508 : (tensor<2048x64x14x14xf32>) -> tensor<2048x64x14x14xbf16>
    %v2511 = stablehlo.convolution(%v2509, %v2510)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<2048x64x14x14xbf16>) -> tensor<1024x2048x1x1xbf16>
    %v2512 = stablehlo.convert %v2511 : (tensor<1024x2048x1x1xbf16>) -> tensor<1024x2048x1x1xf32>
    %v2513 = stablehlo.transpose %v2512, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v2514 = stablehlo.reshape %v1519 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2515 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2516 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2517 = stablehlo.reduce(%v2514 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2518 = stablehlo.broadcast_in_dim %v2517, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2519 = stablehlo.divide %v2518, %v2516 : tensor<64x2048x7x7xf32>
    %v2520 = stablehlo.subtract %v2514, %v2519 : tensor<64x2048x7x7xf32>
    %v2521 = stablehlo.multiply %v2520, %v2520 : tensor<64x2048x7x7xf32>
    %v2522 = stablehlo.reduce(%v2521 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2523 = stablehlo.broadcast_in_dim %v2522, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2524 = stablehlo.divide %v2523, %v2516 : tensor<64x2048x7x7xf32>
    %v2525 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2526 = stablehlo.add %v2524, %v2525 : tensor<64x2048x7x7xf32>
    %v2527 = stablehlo.rsqrt %v2526 : tensor<64x2048x7x7xf32>
    %v2528 = stablehlo.multiply %v2520, %v2527 : tensor<64x2048x7x7xf32>
    %v2529 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2530 = stablehlo.multiply %v2529, %v2528 : tensor<64x2048x7x7xf32>
    %v2531 = stablehlo.reduce(%v2530 init: %v2515) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2532 = stablehlo.reshape %v2238 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2534 = stablehlo.reduce(%v2532 init: %v2533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2535 = stablehlo.reshape %v2410 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2536 = stablehlo.reshape %v1415 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2537 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v2538 = stablehlo.compare GT, %v2536, %v2537 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v2539 = stablehlo.select %v2538, %v2535, %v2537 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v2540 = stablehlo.reshape %v2539 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2541 = stablehlo.reshape %v1391 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2543 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2544 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2545 = stablehlo.reduce(%v2541 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2546 = stablehlo.broadcast_in_dim %v2545, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2547 = stablehlo.divide %v2546, %v2543 : tensor<64x1024x14x14xf32>
    %v2548 = stablehlo.subtract %v2541, %v2547 : tensor<64x1024x14x14xf32>
    %v2549 = stablehlo.multiply %v2548, %v2548 : tensor<64x1024x14x14xf32>
    %v2550 = stablehlo.reduce(%v2549 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2551 = stablehlo.broadcast_in_dim %v2550, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2552 = stablehlo.divide %v2551, %v2543 : tensor<64x1024x14x14xf32>
    %v2553 = stablehlo.add %v2552, %v2544 : tensor<64x1024x14x14xf32>
    %v2554 = stablehlo.rsqrt %v2553 : tensor<64x1024x14x14xf32>
    %v2555 = stablehlo.multiply %v2548, %v2554 : tensor<64x1024x14x14xf32>
    %v2556 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2557 = stablehlo.reshape %v2540 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2558 = stablehlo.multiply %v2556, %v2557 : tensor<64x1024x14x14xf32>
    %v2559 = stablehlo.reduce(%v2558 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2560 = stablehlo.broadcast_in_dim %v2559, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2561 = stablehlo.multiply %v2555, %v2558 : tensor<64x1024x14x14xf32>
    %v2562 = stablehlo.reduce(%v2561 init: %v2542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2563 = stablehlo.broadcast_in_dim %v2562, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2564 = stablehlo.multiply %v2558, %v2543 : tensor<64x1024x14x14xf32>
    %v2565 = stablehlo.subtract %v2564, %v2560 : tensor<64x1024x14x14xf32>
    %v2566 = stablehlo.multiply %v2555, %v2563 : tensor<64x1024x14x14xf32>
    %v2567 = stablehlo.subtract %v2565, %v2566 : tensor<64x1024x14x14xf32>
    %v2568 = stablehlo.divide %v2554, %v2543 : tensor<64x1024x14x14xf32>
    %v2569 = stablehlo.multiply %v2568, %v2567 : tensor<64x1024x14x14xf32>
    %v2570 = stablehlo.reshape %v2569 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2571 = stablehlo.reshape %v2570 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2572 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2573 = stablehlo.transpose %v2572, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2574 = stablehlo.convert %v2571 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2575 = stablehlo.convert %v2573 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v2576 = stablehlo.convolution(%v2574, %v2575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2577 = stablehlo.convert %v2576 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2579 = stablehlo.reshape %v2578 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2580 = stablehlo.reshape %v1379 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2581 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2582 = stablehlo.compare GT, %v2580, %v2581 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2583 = stablehlo.select %v2582, %v2579, %v2581 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2584 = stablehlo.reshape %v2583 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2585 = stablehlo.reshape %v1359 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2587 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2588 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2589 = stablehlo.reduce(%v2585 init: %v2586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2590 = stablehlo.broadcast_in_dim %v2589, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2591 = stablehlo.divide %v2590, %v2587 : tensor<64x256x14x14xf32>
    %v2592 = stablehlo.subtract %v2585, %v2591 : tensor<64x256x14x14xf32>
    %v2593 = stablehlo.multiply %v2592, %v2592 : tensor<64x256x14x14xf32>
    %v2594 = stablehlo.reduce(%v2593 init: %v2586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2595 = stablehlo.broadcast_in_dim %v2594, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2596 = stablehlo.divide %v2595, %v2587 : tensor<64x256x14x14xf32>
    %v2597 = stablehlo.add %v2596, %v2588 : tensor<64x256x14x14xf32>
    %v2598 = stablehlo.rsqrt %v2597 : tensor<64x256x14x14xf32>
    %v2599 = stablehlo.multiply %v2592, %v2598 : tensor<64x256x14x14xf32>
    %v2600 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2601 = stablehlo.reshape %v2584 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2602 = stablehlo.multiply %v2600, %v2601 : tensor<64x256x14x14xf32>
    %v2603 = stablehlo.reduce(%v2602 init: %v2586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2604 = stablehlo.broadcast_in_dim %v2603, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2605 = stablehlo.multiply %v2599, %v2602 : tensor<64x256x14x14xf32>
    %v2606 = stablehlo.reduce(%v2605 init: %v2586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2607 = stablehlo.broadcast_in_dim %v2606, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2608 = stablehlo.multiply %v2602, %v2587 : tensor<64x256x14x14xf32>
    %v2609 = stablehlo.subtract %v2608, %v2604 : tensor<64x256x14x14xf32>
    %v2610 = stablehlo.multiply %v2599, %v2607 : tensor<64x256x14x14xf32>
    %v2611 = stablehlo.subtract %v2609, %v2610 : tensor<64x256x14x14xf32>
    %v2612 = stablehlo.divide %v2598, %v2587 : tensor<64x256x14x14xf32>
    %v2613 = stablehlo.multiply %v2612, %v2611 : tensor<64x256x14x14xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2615 = stablehlo.reshape %v2614 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2616 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2617 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2618 = stablehlo.convert %v2615 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2619 = stablehlo.convert %v2617 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2620 = stablehlo.convolution(%v2618, %v2619)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2621 = stablehlo.convert %v2620 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2622 = stablehlo.reshape %v2621 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2623 = stablehlo.reshape %v2622 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2624 = stablehlo.reshape %v1347 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2625 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2626 = stablehlo.compare GT, %v2624, %v2625 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2627 = stablehlo.select %v2626, %v2623, %v2625 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2628 = stablehlo.reshape %v2627 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2629 = stablehlo.reshape %v1327 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2631 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2632 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2633 = stablehlo.reduce(%v2629 init: %v2630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2634 = stablehlo.broadcast_in_dim %v2633, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2635 = stablehlo.divide %v2634, %v2631 : tensor<64x256x14x14xf32>
    %v2636 = stablehlo.subtract %v2629, %v2635 : tensor<64x256x14x14xf32>
    %v2637 = stablehlo.multiply %v2636, %v2636 : tensor<64x256x14x14xf32>
    %v2638 = stablehlo.reduce(%v2637 init: %v2630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2639 = stablehlo.broadcast_in_dim %v2638, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2640 = stablehlo.divide %v2639, %v2631 : tensor<64x256x14x14xf32>
    %v2641 = stablehlo.add %v2640, %v2632 : tensor<64x256x14x14xf32>
    %v2642 = stablehlo.rsqrt %v2641 : tensor<64x256x14x14xf32>
    %v2643 = stablehlo.multiply %v2636, %v2642 : tensor<64x256x14x14xf32>
    %v2644 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2645 = stablehlo.reshape %v2628 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2646 = stablehlo.multiply %v2644, %v2645 : tensor<64x256x14x14xf32>
    %v2647 = stablehlo.reduce(%v2646 init: %v2630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2648 = stablehlo.broadcast_in_dim %v2647, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2649 = stablehlo.multiply %v2643, %v2646 : tensor<64x256x14x14xf32>
    %v2650 = stablehlo.reduce(%v2649 init: %v2630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2652 = stablehlo.multiply %v2646, %v2631 : tensor<64x256x14x14xf32>
    %v2653 = stablehlo.subtract %v2652, %v2648 : tensor<64x256x14x14xf32>
    %v2654 = stablehlo.multiply %v2643, %v2651 : tensor<64x256x14x14xf32>
    %v2655 = stablehlo.subtract %v2653, %v2654 : tensor<64x256x14x14xf32>
    %v2656 = stablehlo.divide %v2642, %v2631 : tensor<64x256x14x14xf32>
    %v2657 = stablehlo.multiply %v2656, %v2655 : tensor<64x256x14x14xf32>
    %v2658 = stablehlo.reshape %v2657 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2659 = stablehlo.reshape %v2658 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2660 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2661 = stablehlo.transpose %v2660, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2662 = stablehlo.convert %v2659 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2663 = stablehlo.convert %v2661 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v2664 = stablehlo.convolution(%v2662, %v2663)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2665 = stablehlo.convert %v2664 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2666 = stablehlo.reshape %v2665 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2667 = stablehlo.reshape %v2666 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2668 = stablehlo.reshape %v2540 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2669 = stablehlo.add %v2667, %v2668 : tensor<64x64x56x56xf32>
    %v2670 = stablehlo.reshape %v2669 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2671 = stablehlo.reshape %v1319 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2672 = stablehlo.reshape %v2658 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2673 = stablehlo.transpose %v2671, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2674 = stablehlo.transpose %v2672, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2675 = stablehlo.convert %v2673 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2676 = stablehlo.convert %v2674 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2677 = stablehlo.convolution(%v2675, %v2676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v2678 = stablehlo.convert %v2677 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v2679 = stablehlo.transpose %v2678, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2680 = stablehlo.reshape %v1327 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2681 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2682 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2683 = stablehlo.reduce(%v2680 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2684 = stablehlo.broadcast_in_dim %v2683, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2685 = stablehlo.divide %v2684, %v2682 : tensor<64x256x14x14xf32>
    %v2686 = stablehlo.subtract %v2680, %v2685 : tensor<64x256x14x14xf32>
    %v2687 = stablehlo.multiply %v2686, %v2686 : tensor<64x256x14x14xf32>
    %v2688 = stablehlo.reduce(%v2687 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2689 = stablehlo.broadcast_in_dim %v2688, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2690 = stablehlo.divide %v2689, %v2682 : tensor<64x256x14x14xf32>
    %v2691 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2692 = stablehlo.add %v2690, %v2691 : tensor<64x256x14x14xf32>
    %v2693 = stablehlo.rsqrt %v2692 : tensor<64x256x14x14xf32>
    %v2694 = stablehlo.multiply %v2686, %v2693 : tensor<64x256x14x14xf32>
    %v2695 = stablehlo.reshape %v2628 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2696 = stablehlo.multiply %v2695, %v2694 : tensor<64x256x14x14xf32>
    %v2697 = stablehlo.reduce(%v2696 init: %v2681) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2698 = stablehlo.reshape %v2628 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2699 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2700 = stablehlo.reduce(%v2698 init: %v2699) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2701 = stablehlo.reshape %v1351 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2702 = stablehlo.reshape %v2614 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2703 = stablehlo.transpose %v2701, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2704 = stablehlo.transpose %v2702, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2705 = stablehlo.convert %v2703 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2706 = stablehlo.convert %v2704 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2707 = stablehlo.convolution(%v2705, %v2706)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2708 = stablehlo.convert %v2707 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2709 = stablehlo.transpose %v2708, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2710 = stablehlo.reshape %v1359 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2712 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2713 = stablehlo.reduce(%v2710 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2714 = stablehlo.broadcast_in_dim %v2713, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2715 = stablehlo.divide %v2714, %v2712 : tensor<64x256x14x14xf32>
    %v2716 = stablehlo.subtract %v2710, %v2715 : tensor<64x256x14x14xf32>
    %v2717 = stablehlo.multiply %v2716, %v2716 : tensor<64x256x14x14xf32>
    %v2718 = stablehlo.reduce(%v2717 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2719 = stablehlo.broadcast_in_dim %v2718, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2720 = stablehlo.divide %v2719, %v2712 : tensor<64x256x14x14xf32>
    %v2721 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2722 = stablehlo.add %v2720, %v2721 : tensor<64x256x14x14xf32>
    %v2723 = stablehlo.rsqrt %v2722 : tensor<64x256x14x14xf32>
    %v2724 = stablehlo.multiply %v2716, %v2723 : tensor<64x256x14x14xf32>
    %v2725 = stablehlo.reshape %v2584 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2726 = stablehlo.multiply %v2725, %v2724 : tensor<64x256x14x14xf32>
    %v2727 = stablehlo.reduce(%v2726 init: %v2711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2728 = stablehlo.reshape %v2584 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2730 = stablehlo.reduce(%v2728 init: %v2729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2731 = stablehlo.reshape %v1383 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2732 = stablehlo.reshape %v2570 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2733 = stablehlo.transpose %v2731, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2734 = stablehlo.transpose %v2732, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2735 = stablehlo.convert %v2733 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2736 = stablehlo.convert %v2734 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2737 = stablehlo.convolution(%v2735, %v2736)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v2738 = stablehlo.convert %v2737 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v2739 = stablehlo.transpose %v2738, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2740 = stablehlo.reshape %v1391 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2741 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2742 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2743 = stablehlo.reduce(%v2740 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2744 = stablehlo.broadcast_in_dim %v2743, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2745 = stablehlo.divide %v2744, %v2742 : tensor<64x1024x14x14xf32>
    %v2746 = stablehlo.subtract %v2740, %v2745 : tensor<64x1024x14x14xf32>
    %v2747 = stablehlo.multiply %v2746, %v2746 : tensor<64x1024x14x14xf32>
    %v2748 = stablehlo.reduce(%v2747 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2749 = stablehlo.broadcast_in_dim %v2748, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2750 = stablehlo.divide %v2749, %v2742 : tensor<64x1024x14x14xf32>
    %v2751 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2752 = stablehlo.add %v2750, %v2751 : tensor<64x1024x14x14xf32>
    %v2753 = stablehlo.rsqrt %v2752 : tensor<64x1024x14x14xf32>
    %v2754 = stablehlo.multiply %v2746, %v2753 : tensor<64x1024x14x14xf32>
    %v2755 = stablehlo.reshape %v2540 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2756 = stablehlo.multiply %v2755, %v2754 : tensor<64x1024x14x14xf32>
    %v2757 = stablehlo.reduce(%v2756 init: %v2741) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2758 = stablehlo.reshape %v2540 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2759 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2760 = stablehlo.reduce(%v2758 init: %v2759) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2761 = stablehlo.reshape %v2670 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2762 = stablehlo.reshape %v1315 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2763 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v2764 = stablehlo.compare GT, %v2762, %v2763 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v2765 = stablehlo.select %v2764, %v2761, %v2763 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v2766 = stablehlo.reshape %v2765 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2767 = stablehlo.reshape %v1291 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2768 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2769 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2770 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2771 = stablehlo.reduce(%v2767 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2772 = stablehlo.broadcast_in_dim %v2771, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2773 = stablehlo.divide %v2772, %v2769 : tensor<64x1024x14x14xf32>
    %v2774 = stablehlo.subtract %v2767, %v2773 : tensor<64x1024x14x14xf32>
    %v2775 = stablehlo.multiply %v2774, %v2774 : tensor<64x1024x14x14xf32>
    %v2776 = stablehlo.reduce(%v2775 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2777 = stablehlo.broadcast_in_dim %v2776, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2778 = stablehlo.divide %v2777, %v2769 : tensor<64x1024x14x14xf32>
    %v2779 = stablehlo.add %v2778, %v2770 : tensor<64x1024x14x14xf32>
    %v2780 = stablehlo.rsqrt %v2779 : tensor<64x1024x14x14xf32>
    %v2781 = stablehlo.multiply %v2774, %v2780 : tensor<64x1024x14x14xf32>
    %v2782 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2783 = stablehlo.reshape %v2766 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2784 = stablehlo.multiply %v2782, %v2783 : tensor<64x1024x14x14xf32>
    %v2785 = stablehlo.reduce(%v2784 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2786 = stablehlo.broadcast_in_dim %v2785, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2787 = stablehlo.multiply %v2781, %v2784 : tensor<64x1024x14x14xf32>
    %v2788 = stablehlo.reduce(%v2787 init: %v2768) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2790 = stablehlo.multiply %v2784, %v2769 : tensor<64x1024x14x14xf32>
    %v2791 = stablehlo.subtract %v2790, %v2786 : tensor<64x1024x14x14xf32>
    %v2792 = stablehlo.multiply %v2781, %v2789 : tensor<64x1024x14x14xf32>
    %v2793 = stablehlo.subtract %v2791, %v2792 : tensor<64x1024x14x14xf32>
    %v2794 = stablehlo.divide %v2780, %v2769 : tensor<64x1024x14x14xf32>
    %v2795 = stablehlo.multiply %v2794, %v2793 : tensor<64x1024x14x14xf32>
    %v2796 = stablehlo.reshape %v2795 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2797 = stablehlo.reshape %v2796 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2798 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2799 = stablehlo.transpose %v2798, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2800 = stablehlo.convert %v2797 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2801 = stablehlo.convert %v2799 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v2802 = stablehlo.convolution(%v2800, %v2801)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2803 = stablehlo.convert %v2802 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2804 = stablehlo.reshape %v2803 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2805 = stablehlo.reshape %v2804 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2806 = stablehlo.reshape %v1279 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2807 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2808 = stablehlo.compare GT, %v2806, %v2807 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2809 = stablehlo.select %v2808, %v2805, %v2807 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2810 = stablehlo.reshape %v2809 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2811 = stablehlo.reshape %v1259 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2813 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2814 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2815 = stablehlo.reduce(%v2811 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2816 = stablehlo.broadcast_in_dim %v2815, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2817 = stablehlo.divide %v2816, %v2813 : tensor<64x256x14x14xf32>
    %v2818 = stablehlo.subtract %v2811, %v2817 : tensor<64x256x14x14xf32>
    %v2819 = stablehlo.multiply %v2818, %v2818 : tensor<64x256x14x14xf32>
    %v2820 = stablehlo.reduce(%v2819 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2821 = stablehlo.broadcast_in_dim %v2820, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2822 = stablehlo.divide %v2821, %v2813 : tensor<64x256x14x14xf32>
    %v2823 = stablehlo.add %v2822, %v2814 : tensor<64x256x14x14xf32>
    %v2824 = stablehlo.rsqrt %v2823 : tensor<64x256x14x14xf32>
    %v2825 = stablehlo.multiply %v2818, %v2824 : tensor<64x256x14x14xf32>
    %v2826 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2827 = stablehlo.reshape %v2810 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2828 = stablehlo.multiply %v2826, %v2827 : tensor<64x256x14x14xf32>
    %v2829 = stablehlo.reduce(%v2828 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2831 = stablehlo.multiply %v2825, %v2828 : tensor<64x256x14x14xf32>
    %v2832 = stablehlo.reduce(%v2831 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2833 = stablehlo.broadcast_in_dim %v2832, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2834 = stablehlo.multiply %v2828, %v2813 : tensor<64x256x14x14xf32>
    %v2835 = stablehlo.subtract %v2834, %v2830 : tensor<64x256x14x14xf32>
    %v2836 = stablehlo.multiply %v2825, %v2833 : tensor<64x256x14x14xf32>
    %v2837 = stablehlo.subtract %v2835, %v2836 : tensor<64x256x14x14xf32>
    %v2838 = stablehlo.divide %v2824, %v2813 : tensor<64x256x14x14xf32>
    %v2839 = stablehlo.multiply %v2838, %v2837 : tensor<64x256x14x14xf32>
    %v2840 = stablehlo.reshape %v2839 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2841 = stablehlo.reshape %v2840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2842 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2843 = stablehlo.transpose %v2842, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2844 = stablehlo.convert %v2841 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2845 = stablehlo.convert %v2843 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2846 = stablehlo.convolution(%v2844, %v2845)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2847 = stablehlo.convert %v2846 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2848 = stablehlo.reshape %v2847 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2849 = stablehlo.reshape %v2848 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2850 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2851 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2852 = stablehlo.compare GT, %v2850, %v2851 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2853 = stablehlo.select %v2852, %v2849, %v2851 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2855 = stablehlo.reshape %v1227 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2858 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2859 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2860 = stablehlo.broadcast_in_dim %v2859, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2861 = stablehlo.divide %v2860, %v2857 : tensor<64x256x14x14xf32>
    %v2862 = stablehlo.subtract %v2855, %v2861 : tensor<64x256x14x14xf32>
    %v2863 = stablehlo.multiply %v2862, %v2862 : tensor<64x256x14x14xf32>
    %v2864 = stablehlo.reduce(%v2863 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2866 = stablehlo.divide %v2865, %v2857 : tensor<64x256x14x14xf32>
    %v2867 = stablehlo.add %v2866, %v2858 : tensor<64x256x14x14xf32>
    %v2868 = stablehlo.rsqrt %v2867 : tensor<64x256x14x14xf32>
    %v2869 = stablehlo.multiply %v2862, %v2868 : tensor<64x256x14x14xf32>
    %v2870 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2871 = stablehlo.reshape %v2854 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2872 = stablehlo.multiply %v2870, %v2871 : tensor<64x256x14x14xf32>
    %v2873 = stablehlo.reduce(%v2872 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2875 = stablehlo.multiply %v2869, %v2872 : tensor<64x256x14x14xf32>
    %v2876 = stablehlo.reduce(%v2875 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2877 = stablehlo.broadcast_in_dim %v2876, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2878 = stablehlo.multiply %v2872, %v2857 : tensor<64x256x14x14xf32>
    %v2879 = stablehlo.subtract %v2878, %v2874 : tensor<64x256x14x14xf32>
    %v2880 = stablehlo.multiply %v2869, %v2877 : tensor<64x256x14x14xf32>
    %v2881 = stablehlo.subtract %v2879, %v2880 : tensor<64x256x14x14xf32>
    %v2882 = stablehlo.divide %v2868, %v2857 : tensor<64x256x14x14xf32>
    %v2883 = stablehlo.multiply %v2882, %v2881 : tensor<64x256x14x14xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2886 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2887 = stablehlo.transpose %v2886, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2888 = stablehlo.convert %v2885 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2889 = stablehlo.convert %v2887 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v2890 = stablehlo.convolution(%v2888, %v2889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2891 = stablehlo.convert %v2890 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2893 = stablehlo.reshape %v2892 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2894 = stablehlo.reshape %v2766 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2895 = stablehlo.add %v2893, %v2894 : tensor<64x64x56x56xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2897 = stablehlo.reshape %v1219 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2898 = stablehlo.reshape %v2884 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2899 = stablehlo.transpose %v2897, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2900 = stablehlo.transpose %v2898, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2901 = stablehlo.convert %v2899 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2902 = stablehlo.convert %v2900 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2903 = stablehlo.convolution(%v2901, %v2902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v2904 = stablehlo.convert %v2903 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v2905 = stablehlo.transpose %v2904, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2906 = stablehlo.reshape %v1227 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2908 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2909 = stablehlo.reduce(%v2906 init: %v2907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2910 = stablehlo.broadcast_in_dim %v2909, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2911 = stablehlo.divide %v2910, %v2908 : tensor<64x256x14x14xf32>
    %v2912 = stablehlo.subtract %v2906, %v2911 : tensor<64x256x14x14xf32>
    %v2913 = stablehlo.multiply %v2912, %v2912 : tensor<64x256x14x14xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2916 = stablehlo.divide %v2915, %v2908 : tensor<64x256x14x14xf32>
    %v2917 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2918 = stablehlo.add %v2916, %v2917 : tensor<64x256x14x14xf32>
    %v2919 = stablehlo.rsqrt %v2918 : tensor<64x256x14x14xf32>
    %v2920 = stablehlo.multiply %v2912, %v2919 : tensor<64x256x14x14xf32>
    %v2921 = stablehlo.reshape %v2854 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2922 = stablehlo.multiply %v2921, %v2920 : tensor<64x256x14x14xf32>
    %v2923 = stablehlo.reduce(%v2922 init: %v2907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2924 = stablehlo.reshape %v2854 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2926 = stablehlo.reduce(%v2924 init: %v2925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2927 = stablehlo.reshape %v1251 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2928 = stablehlo.reshape %v2840 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2929 = stablehlo.transpose %v2927, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2930 = stablehlo.transpose %v2928, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2931 = stablehlo.convert %v2929 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2932 = stablehlo.convert %v2930 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2933 = stablehlo.convolution(%v2931, %v2932)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2934 = stablehlo.convert %v2933 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2935 = stablehlo.transpose %v2934, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2936 = stablehlo.reshape %v1259 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2938 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2939 = stablehlo.reduce(%v2936 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2940 = stablehlo.broadcast_in_dim %v2939, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2941 = stablehlo.divide %v2940, %v2938 : tensor<64x256x14x14xf32>
    %v2942 = stablehlo.subtract %v2936, %v2941 : tensor<64x256x14x14xf32>
    %v2943 = stablehlo.multiply %v2942, %v2942 : tensor<64x256x14x14xf32>
    %v2944 = stablehlo.reduce(%v2943 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2945 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2946 = stablehlo.divide %v2945, %v2938 : tensor<64x256x14x14xf32>
    %v2947 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2948 = stablehlo.add %v2946, %v2947 : tensor<64x256x14x14xf32>
    %v2949 = stablehlo.rsqrt %v2948 : tensor<64x256x14x14xf32>
    %v2950 = stablehlo.multiply %v2942, %v2949 : tensor<64x256x14x14xf32>
    %v2951 = stablehlo.reshape %v2810 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2952 = stablehlo.multiply %v2951, %v2950 : tensor<64x256x14x14xf32>
    %v2953 = stablehlo.reduce(%v2952 init: %v2937) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2954 = stablehlo.reshape %v2810 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2956 = stablehlo.reduce(%v2954 init: %v2955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2957 = stablehlo.reshape %v1283 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2958 = stablehlo.reshape %v2796 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2959 = stablehlo.transpose %v2957, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2960 = stablehlo.transpose %v2958, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2961 = stablehlo.convert %v2959 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2962 = stablehlo.convert %v2960 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2963 = stablehlo.convolution(%v2961, %v2962)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v2964 = stablehlo.convert %v2963 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v2965 = stablehlo.transpose %v2964, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2966 = stablehlo.reshape %v1291 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2968 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2969 = stablehlo.reduce(%v2966 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2970 = stablehlo.broadcast_in_dim %v2969, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2971 = stablehlo.divide %v2970, %v2968 : tensor<64x1024x14x14xf32>
    %v2972 = stablehlo.subtract %v2966, %v2971 : tensor<64x1024x14x14xf32>
    %v2973 = stablehlo.multiply %v2972, %v2972 : tensor<64x1024x14x14xf32>
    %v2974 = stablehlo.reduce(%v2973 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2975 = stablehlo.broadcast_in_dim %v2974, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2976 = stablehlo.divide %v2975, %v2968 : tensor<64x1024x14x14xf32>
    %v2977 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2978 = stablehlo.add %v2976, %v2977 : tensor<64x1024x14x14xf32>
    %v2979 = stablehlo.rsqrt %v2978 : tensor<64x1024x14x14xf32>
    %v2980 = stablehlo.multiply %v2972, %v2979 : tensor<64x1024x14x14xf32>
    %v2981 = stablehlo.reshape %v2766 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2982 = stablehlo.multiply %v2981, %v2980 : tensor<64x1024x14x14xf32>
    %v2983 = stablehlo.reduce(%v2982 init: %v2967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2984 = stablehlo.reshape %v2766 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2985 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2986 = stablehlo.reduce(%v2984 init: %v2985) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2987 = stablehlo.reshape %v2896 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2988 = stablehlo.reshape %v1215 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v2989 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v2990 = stablehlo.compare GT, %v2988, %v2989 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v2991 = stablehlo.select %v2990, %v2987, %v2989 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v2992 = stablehlo.reshape %v2991 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v2993 = stablehlo.reshape %v1191 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2994 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2995 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2996 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2997 = stablehlo.reduce(%v2993 init: %v2994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2998 = stablehlo.broadcast_in_dim %v2997, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2999 = stablehlo.divide %v2998, %v2995 : tensor<64x1024x14x14xf32>
    %v3000 = stablehlo.subtract %v2993, %v2999 : tensor<64x1024x14x14xf32>
    %v3001 = stablehlo.multiply %v3000, %v3000 : tensor<64x1024x14x14xf32>
    %v3002 = stablehlo.reduce(%v3001 init: %v2994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3003 = stablehlo.broadcast_in_dim %v3002, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3004 = stablehlo.divide %v3003, %v2995 : tensor<64x1024x14x14xf32>
    %v3005 = stablehlo.add %v3004, %v2996 : tensor<64x1024x14x14xf32>
    %v3006 = stablehlo.rsqrt %v3005 : tensor<64x1024x14x14xf32>
    %v3007 = stablehlo.multiply %v3000, %v3006 : tensor<64x1024x14x14xf32>
    %v3008 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3009 = stablehlo.reshape %v2992 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3010 = stablehlo.multiply %v3008, %v3009 : tensor<64x1024x14x14xf32>
    %v3011 = stablehlo.reduce(%v3010 init: %v2994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3012 = stablehlo.broadcast_in_dim %v3011, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3013 = stablehlo.multiply %v3007, %v3010 : tensor<64x1024x14x14xf32>
    %v3014 = stablehlo.reduce(%v3013 init: %v2994) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3015 = stablehlo.broadcast_in_dim %v3014, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3016 = stablehlo.multiply %v3010, %v2995 : tensor<64x1024x14x14xf32>
    %v3017 = stablehlo.subtract %v3016, %v3012 : tensor<64x1024x14x14xf32>
    %v3018 = stablehlo.multiply %v3007, %v3015 : tensor<64x1024x14x14xf32>
    %v3019 = stablehlo.subtract %v3017, %v3018 : tensor<64x1024x14x14xf32>
    %v3020 = stablehlo.divide %v3006, %v2995 : tensor<64x1024x14x14xf32>
    %v3021 = stablehlo.multiply %v3020, %v3019 : tensor<64x1024x14x14xf32>
    %v3022 = stablehlo.reshape %v3021 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3023 = stablehlo.reshape %v3022 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3024 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3025 = stablehlo.transpose %v3024, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3026 = stablehlo.convert %v3023 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3027 = stablehlo.convert %v3025 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3028 = stablehlo.convolution(%v3026, %v3027)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3029 = stablehlo.convert %v3028 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3030 = stablehlo.reshape %v3029 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3031 = stablehlo.reshape %v3030 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3032 = stablehlo.reshape %v1179 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3033 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3034 = stablehlo.compare GT, %v3032, %v3033 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3035 = stablehlo.select %v3034, %v3031, %v3033 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3037 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3039 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3040 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3041 = stablehlo.reduce(%v3037 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3041, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3043 = stablehlo.divide %v3042, %v3039 : tensor<64x256x14x14xf32>
    %v3044 = stablehlo.subtract %v3037, %v3043 : tensor<64x256x14x14xf32>
    %v3045 = stablehlo.multiply %v3044, %v3044 : tensor<64x256x14x14xf32>
    %v3046 = stablehlo.reduce(%v3045 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3047 = stablehlo.broadcast_in_dim %v3046, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3048 = stablehlo.divide %v3047, %v3039 : tensor<64x256x14x14xf32>
    %v3049 = stablehlo.add %v3048, %v3040 : tensor<64x256x14x14xf32>
    %v3050 = stablehlo.rsqrt %v3049 : tensor<64x256x14x14xf32>
    %v3051 = stablehlo.multiply %v3044, %v3050 : tensor<64x256x14x14xf32>
    %v3052 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3053 = stablehlo.reshape %v3036 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3054 = stablehlo.multiply %v3052, %v3053 : tensor<64x256x14x14xf32>
    %v3055 = stablehlo.reduce(%v3054 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3056 = stablehlo.broadcast_in_dim %v3055, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3057 = stablehlo.multiply %v3051, %v3054 : tensor<64x256x14x14xf32>
    %v3058 = stablehlo.reduce(%v3057 init: %v3038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3059 = stablehlo.broadcast_in_dim %v3058, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3060 = stablehlo.multiply %v3054, %v3039 : tensor<64x256x14x14xf32>
    %v3061 = stablehlo.subtract %v3060, %v3056 : tensor<64x256x14x14xf32>
    %v3062 = stablehlo.multiply %v3051, %v3059 : tensor<64x256x14x14xf32>
    %v3063 = stablehlo.subtract %v3061, %v3062 : tensor<64x256x14x14xf32>
    %v3064 = stablehlo.divide %v3050, %v3039 : tensor<64x256x14x14xf32>
    %v3065 = stablehlo.multiply %v3064, %v3063 : tensor<64x256x14x14xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3068 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3069 = stablehlo.transpose %v3068, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3070 = stablehlo.convert %v3067 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3071 = stablehlo.convert %v3069 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3072 = stablehlo.convolution(%v3070, %v3071)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3073 = stablehlo.convert %v3072 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3074 = stablehlo.reshape %v3073 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3075 = stablehlo.reshape %v3074 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3076 = stablehlo.reshape %v1147 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3077 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3078 = stablehlo.compare GT, %v3076, %v3077 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3079 = stablehlo.select %v3078, %v3075, %v3077 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3080 = stablehlo.reshape %v3079 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3081 = stablehlo.reshape %v1127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3083 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3084 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3085 = stablehlo.reduce(%v3081 init: %v3082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3086 = stablehlo.broadcast_in_dim %v3085, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3087 = stablehlo.divide %v3086, %v3083 : tensor<64x256x14x14xf32>
    %v3088 = stablehlo.subtract %v3081, %v3087 : tensor<64x256x14x14xf32>
    %v3089 = stablehlo.multiply %v3088, %v3088 : tensor<64x256x14x14xf32>
    %v3090 = stablehlo.reduce(%v3089 init: %v3082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3091 = stablehlo.broadcast_in_dim %v3090, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3092 = stablehlo.divide %v3091, %v3083 : tensor<64x256x14x14xf32>
    %v3093 = stablehlo.add %v3092, %v3084 : tensor<64x256x14x14xf32>
    %v3094 = stablehlo.rsqrt %v3093 : tensor<64x256x14x14xf32>
    %v3095 = stablehlo.multiply %v3088, %v3094 : tensor<64x256x14x14xf32>
    %v3096 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3097 = stablehlo.reshape %v3080 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3098 = stablehlo.multiply %v3096, %v3097 : tensor<64x256x14x14xf32>
    %v3099 = stablehlo.reduce(%v3098 init: %v3082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3099, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3101 = stablehlo.multiply %v3095, %v3098 : tensor<64x256x14x14xf32>
    %v3102 = stablehlo.reduce(%v3101 init: %v3082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3103 = stablehlo.broadcast_in_dim %v3102, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3104 = stablehlo.multiply %v3098, %v3083 : tensor<64x256x14x14xf32>
    %v3105 = stablehlo.subtract %v3104, %v3100 : tensor<64x256x14x14xf32>
    %v3106 = stablehlo.multiply %v3095, %v3103 : tensor<64x256x14x14xf32>
    %v3107 = stablehlo.subtract %v3105, %v3106 : tensor<64x256x14x14xf32>
    %v3108 = stablehlo.divide %v3094, %v3083 : tensor<64x256x14x14xf32>
    %v3109 = stablehlo.multiply %v3108, %v3107 : tensor<64x256x14x14xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3111 = stablehlo.reshape %v3110 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3112 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3113 = stablehlo.transpose %v3112, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3114 = stablehlo.convert %v3111 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3115 = stablehlo.convert %v3113 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3116 = stablehlo.convolution(%v3114, %v3115)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3117 = stablehlo.convert %v3116 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3119 = stablehlo.reshape %v3118 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3120 = stablehlo.reshape %v2992 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3121 = stablehlo.add %v3119, %v3120 : tensor<64x64x56x56xf32>
    %v3122 = stablehlo.reshape %v3121 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3123 = stablehlo.reshape %v1119 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3124 = stablehlo.reshape %v3110 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3125 = stablehlo.transpose %v3123, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3126 = stablehlo.transpose %v3124, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3127 = stablehlo.convert %v3125 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3128 = stablehlo.convert %v3126 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3129 = stablehlo.convolution(%v3127, %v3128)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3130 = stablehlo.convert %v3129 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3131 = stablehlo.transpose %v3130, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3132 = stablehlo.reshape %v1127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3133 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3134 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3135 = stablehlo.reduce(%v3132 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3136 = stablehlo.broadcast_in_dim %v3135, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3137 = stablehlo.divide %v3136, %v3134 : tensor<64x256x14x14xf32>
    %v3138 = stablehlo.subtract %v3132, %v3137 : tensor<64x256x14x14xf32>
    %v3139 = stablehlo.multiply %v3138, %v3138 : tensor<64x256x14x14xf32>
    %v3140 = stablehlo.reduce(%v3139 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3141 = stablehlo.broadcast_in_dim %v3140, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3142 = stablehlo.divide %v3141, %v3134 : tensor<64x256x14x14xf32>
    %v3143 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3144 = stablehlo.add %v3142, %v3143 : tensor<64x256x14x14xf32>
    %v3145 = stablehlo.rsqrt %v3144 : tensor<64x256x14x14xf32>
    %v3146 = stablehlo.multiply %v3138, %v3145 : tensor<64x256x14x14xf32>
    %v3147 = stablehlo.reshape %v3080 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3148 = stablehlo.multiply %v3147, %v3146 : tensor<64x256x14x14xf32>
    %v3149 = stablehlo.reduce(%v3148 init: %v3133) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3150 = stablehlo.reshape %v3080 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3152 = stablehlo.reduce(%v3150 init: %v3151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3153 = stablehlo.reshape %v1151 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3154 = stablehlo.reshape %v3066 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3155 = stablehlo.transpose %v3153, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3156 = stablehlo.transpose %v3154, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3157 = stablehlo.convert %v3155 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3158 = stablehlo.convert %v3156 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3159 = stablehlo.convolution(%v3157, %v3158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3160 = stablehlo.convert %v3159 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3161 = stablehlo.transpose %v3160, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3162 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3163 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3164 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3165 = stablehlo.reduce(%v3162 init: %v3163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3166 = stablehlo.broadcast_in_dim %v3165, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3167 = stablehlo.divide %v3166, %v3164 : tensor<64x256x14x14xf32>
    %v3168 = stablehlo.subtract %v3162, %v3167 : tensor<64x256x14x14xf32>
    %v3169 = stablehlo.multiply %v3168, %v3168 : tensor<64x256x14x14xf32>
    %v3170 = stablehlo.reduce(%v3169 init: %v3163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3171 = stablehlo.broadcast_in_dim %v3170, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3172 = stablehlo.divide %v3171, %v3164 : tensor<64x256x14x14xf32>
    %v3173 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3174 = stablehlo.add %v3172, %v3173 : tensor<64x256x14x14xf32>
    %v3175 = stablehlo.rsqrt %v3174 : tensor<64x256x14x14xf32>
    %v3176 = stablehlo.multiply %v3168, %v3175 : tensor<64x256x14x14xf32>
    %v3177 = stablehlo.reshape %v3036 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3178 = stablehlo.multiply %v3177, %v3176 : tensor<64x256x14x14xf32>
    %v3179 = stablehlo.reduce(%v3178 init: %v3163) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3180 = stablehlo.reshape %v3036 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3182 = stablehlo.reduce(%v3180 init: %v3181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3183 = stablehlo.reshape %v1183 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3184 = stablehlo.reshape %v3022 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3185 = stablehlo.transpose %v3183, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3186 = stablehlo.transpose %v3184, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3187 = stablehlo.convert %v3185 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3188 = stablehlo.convert %v3186 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3189 = stablehlo.convolution(%v3187, %v3188)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3190 = stablehlo.convert %v3189 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3191 = stablehlo.transpose %v3190, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3192 = stablehlo.reshape %v1191 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3193 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3194 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3195 = stablehlo.reduce(%v3192 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3196 = stablehlo.broadcast_in_dim %v3195, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3197 = stablehlo.divide %v3196, %v3194 : tensor<64x1024x14x14xf32>
    %v3198 = stablehlo.subtract %v3192, %v3197 : tensor<64x1024x14x14xf32>
    %v3199 = stablehlo.multiply %v3198, %v3198 : tensor<64x1024x14x14xf32>
    %v3200 = stablehlo.reduce(%v3199 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3201 = stablehlo.broadcast_in_dim %v3200, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3202 = stablehlo.divide %v3201, %v3194 : tensor<64x1024x14x14xf32>
    %v3203 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3204 = stablehlo.add %v3202, %v3203 : tensor<64x1024x14x14xf32>
    %v3205 = stablehlo.rsqrt %v3204 : tensor<64x1024x14x14xf32>
    %v3206 = stablehlo.multiply %v3198, %v3205 : tensor<64x1024x14x14xf32>
    %v3207 = stablehlo.reshape %v2992 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3208 = stablehlo.multiply %v3207, %v3206 : tensor<64x1024x14x14xf32>
    %v3209 = stablehlo.reduce(%v3208 init: %v3193) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3210 = stablehlo.reshape %v2992 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.reduce(%v3210 init: %v3211) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3213 = stablehlo.reshape %v3122 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3214 = stablehlo.reshape %v1115 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3215 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v3216 = stablehlo.compare GT, %v3214, %v3215 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v3217 = stablehlo.select %v3216, %v3213, %v3215 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v3218 = stablehlo.reshape %v3217 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3219 = stablehlo.reshape %v1091 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3221 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3222 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3223 = stablehlo.reduce(%v3219 init: %v3220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3224 = stablehlo.broadcast_in_dim %v3223, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3225 = stablehlo.divide %v3224, %v3221 : tensor<64x1024x14x14xf32>
    %v3226 = stablehlo.subtract %v3219, %v3225 : tensor<64x1024x14x14xf32>
    %v3227 = stablehlo.multiply %v3226, %v3226 : tensor<64x1024x14x14xf32>
    %v3228 = stablehlo.reduce(%v3227 init: %v3220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3229 = stablehlo.broadcast_in_dim %v3228, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3230 = stablehlo.divide %v3229, %v3221 : tensor<64x1024x14x14xf32>
    %v3231 = stablehlo.add %v3230, %v3222 : tensor<64x1024x14x14xf32>
    %v3232 = stablehlo.rsqrt %v3231 : tensor<64x1024x14x14xf32>
    %v3233 = stablehlo.multiply %v3226, %v3232 : tensor<64x1024x14x14xf32>
    %v3234 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3235 = stablehlo.reshape %v3218 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3236 = stablehlo.multiply %v3234, %v3235 : tensor<64x1024x14x14xf32>
    %v3237 = stablehlo.reduce(%v3236 init: %v3220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3238 = stablehlo.broadcast_in_dim %v3237, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3239 = stablehlo.multiply %v3233, %v3236 : tensor<64x1024x14x14xf32>
    %v3240 = stablehlo.reduce(%v3239 init: %v3220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3241 = stablehlo.broadcast_in_dim %v3240, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3242 = stablehlo.multiply %v3236, %v3221 : tensor<64x1024x14x14xf32>
    %v3243 = stablehlo.subtract %v3242, %v3238 : tensor<64x1024x14x14xf32>
    %v3244 = stablehlo.multiply %v3233, %v3241 : tensor<64x1024x14x14xf32>
    %v3245 = stablehlo.subtract %v3243, %v3244 : tensor<64x1024x14x14xf32>
    %v3246 = stablehlo.divide %v3232, %v3221 : tensor<64x1024x14x14xf32>
    %v3247 = stablehlo.multiply %v3246, %v3245 : tensor<64x1024x14x14xf32>
    %v3248 = stablehlo.reshape %v3247 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3249 = stablehlo.reshape %v3248 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3250 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3251 = stablehlo.transpose %v3250, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3252 = stablehlo.convert %v3249 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3253 = stablehlo.convert %v3251 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3254 = stablehlo.convolution(%v3252, %v3253)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3255 = stablehlo.convert %v3254 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3256 = stablehlo.reshape %v3255 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3257 = stablehlo.reshape %v3256 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3258 = stablehlo.reshape %v1079 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3259 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3260 = stablehlo.compare GT, %v3258, %v3259 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3261 = stablehlo.select %v3260, %v3257, %v3259 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3262 = stablehlo.reshape %v3261 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3263 = stablehlo.reshape %v1059 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3265 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3266 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3267 = stablehlo.reduce(%v3263 init: %v3264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3269 = stablehlo.divide %v3268, %v3265 : tensor<64x256x14x14xf32>
    %v3270 = stablehlo.subtract %v3263, %v3269 : tensor<64x256x14x14xf32>
    %v3271 = stablehlo.multiply %v3270, %v3270 : tensor<64x256x14x14xf32>
    %v3272 = stablehlo.reduce(%v3271 init: %v3264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3273 = stablehlo.broadcast_in_dim %v3272, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3274 = stablehlo.divide %v3273, %v3265 : tensor<64x256x14x14xf32>
    %v3275 = stablehlo.add %v3274, %v3266 : tensor<64x256x14x14xf32>
    %v3276 = stablehlo.rsqrt %v3275 : tensor<64x256x14x14xf32>
    %v3277 = stablehlo.multiply %v3270, %v3276 : tensor<64x256x14x14xf32>
    %v3278 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3279 = stablehlo.reshape %v3262 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3280 = stablehlo.multiply %v3278, %v3279 : tensor<64x256x14x14xf32>
    %v3281 = stablehlo.reduce(%v3280 init: %v3264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3282 = stablehlo.broadcast_in_dim %v3281, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3283 = stablehlo.multiply %v3277, %v3280 : tensor<64x256x14x14xf32>
    %v3284 = stablehlo.reduce(%v3283 init: %v3264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3285 = stablehlo.broadcast_in_dim %v3284, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3286 = stablehlo.multiply %v3280, %v3265 : tensor<64x256x14x14xf32>
    %v3287 = stablehlo.subtract %v3286, %v3282 : tensor<64x256x14x14xf32>
    %v3288 = stablehlo.multiply %v3277, %v3285 : tensor<64x256x14x14xf32>
    %v3289 = stablehlo.subtract %v3287, %v3288 : tensor<64x256x14x14xf32>
    %v3290 = stablehlo.divide %v3276, %v3265 : tensor<64x256x14x14xf32>
    %v3291 = stablehlo.multiply %v3290, %v3289 : tensor<64x256x14x14xf32>
    %v3292 = stablehlo.reshape %v3291 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3293 = stablehlo.reshape %v3292 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3294 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3295 = stablehlo.transpose %v3294, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3296 = stablehlo.convert %v3293 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3297 = stablehlo.convert %v3295 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3298 = stablehlo.convolution(%v3296, %v3297)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3299 = stablehlo.convert %v3298 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3300 = stablehlo.reshape %v3299 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3301 = stablehlo.reshape %v3300 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3302 = stablehlo.reshape %v1047 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3303 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3304 = stablehlo.compare GT, %v3302, %v3303 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3305 = stablehlo.select %v3304, %v3301, %v3303 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3306 = stablehlo.reshape %v3305 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3307 = stablehlo.reshape %v1027 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3309 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3310 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3311 = stablehlo.reduce(%v3307 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3312 = stablehlo.broadcast_in_dim %v3311, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3313 = stablehlo.divide %v3312, %v3309 : tensor<64x256x14x14xf32>
    %v3314 = stablehlo.subtract %v3307, %v3313 : tensor<64x256x14x14xf32>
    %v3315 = stablehlo.multiply %v3314, %v3314 : tensor<64x256x14x14xf32>
    %v3316 = stablehlo.reduce(%v3315 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3317 = stablehlo.broadcast_in_dim %v3316, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3318 = stablehlo.divide %v3317, %v3309 : tensor<64x256x14x14xf32>
    %v3319 = stablehlo.add %v3318, %v3310 : tensor<64x256x14x14xf32>
    %v3320 = stablehlo.rsqrt %v3319 : tensor<64x256x14x14xf32>
    %v3321 = stablehlo.multiply %v3314, %v3320 : tensor<64x256x14x14xf32>
    %v3322 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3323 = stablehlo.reshape %v3306 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3324 = stablehlo.multiply %v3322, %v3323 : tensor<64x256x14x14xf32>
    %v3325 = stablehlo.reduce(%v3324 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3326 = stablehlo.broadcast_in_dim %v3325, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3327 = stablehlo.multiply %v3321, %v3324 : tensor<64x256x14x14xf32>
    %v3328 = stablehlo.reduce(%v3327 init: %v3308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3329 = stablehlo.broadcast_in_dim %v3328, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3330 = stablehlo.multiply %v3324, %v3309 : tensor<64x256x14x14xf32>
    %v3331 = stablehlo.subtract %v3330, %v3326 : tensor<64x256x14x14xf32>
    %v3332 = stablehlo.multiply %v3321, %v3329 : tensor<64x256x14x14xf32>
    %v3333 = stablehlo.subtract %v3331, %v3332 : tensor<64x256x14x14xf32>
    %v3334 = stablehlo.divide %v3320, %v3309 : tensor<64x256x14x14xf32>
    %v3335 = stablehlo.multiply %v3334, %v3333 : tensor<64x256x14x14xf32>
    %v3336 = stablehlo.reshape %v3335 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3337 = stablehlo.reshape %v3336 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3338 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3339 = stablehlo.transpose %v3338, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3340 = stablehlo.convert %v3337 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3341 = stablehlo.convert %v3339 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3342 = stablehlo.convolution(%v3340, %v3341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3343 = stablehlo.convert %v3342 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3344 = stablehlo.reshape %v3343 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3345 = stablehlo.reshape %v3344 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3346 = stablehlo.reshape %v3218 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3347 = stablehlo.add %v3345, %v3346 : tensor<64x64x56x56xf32>
    %v3348 = stablehlo.reshape %v3347 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3349 = stablehlo.reshape %v1019 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3350 = stablehlo.reshape %v3336 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3351 = stablehlo.transpose %v3349, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3352 = stablehlo.transpose %v3350, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3353 = stablehlo.convert %v3351 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3354 = stablehlo.convert %v3352 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3355 = stablehlo.convolution(%v3353, %v3354)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3356 = stablehlo.convert %v3355 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3357 = stablehlo.transpose %v3356, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3358 = stablehlo.reshape %v1027 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3360 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3361 = stablehlo.reduce(%v3358 init: %v3359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3362 = stablehlo.broadcast_in_dim %v3361, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3363 = stablehlo.divide %v3362, %v3360 : tensor<64x256x14x14xf32>
    %v3364 = stablehlo.subtract %v3358, %v3363 : tensor<64x256x14x14xf32>
    %v3365 = stablehlo.multiply %v3364, %v3364 : tensor<64x256x14x14xf32>
    %v3366 = stablehlo.reduce(%v3365 init: %v3359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3367 = stablehlo.broadcast_in_dim %v3366, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3368 = stablehlo.divide %v3367, %v3360 : tensor<64x256x14x14xf32>
    %v3369 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3370 = stablehlo.add %v3368, %v3369 : tensor<64x256x14x14xf32>
    %v3371 = stablehlo.rsqrt %v3370 : tensor<64x256x14x14xf32>
    %v3372 = stablehlo.multiply %v3364, %v3371 : tensor<64x256x14x14xf32>
    %v3373 = stablehlo.reshape %v3306 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3374 = stablehlo.multiply %v3373, %v3372 : tensor<64x256x14x14xf32>
    %v3375 = stablehlo.reduce(%v3374 init: %v3359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3376 = stablehlo.reshape %v3306 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3377 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3378 = stablehlo.reduce(%v3376 init: %v3377) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3379 = stablehlo.reshape %v1051 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3380 = stablehlo.reshape %v3292 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3381 = stablehlo.transpose %v3379, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3382 = stablehlo.transpose %v3380, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3383 = stablehlo.convert %v3381 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3384 = stablehlo.convert %v3382 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3385 = stablehlo.convolution(%v3383, %v3384)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3386 = stablehlo.convert %v3385 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3387 = stablehlo.transpose %v3386, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3388 = stablehlo.reshape %v1059 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3389 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3390 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3391 = stablehlo.reduce(%v3388 init: %v3389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3392 = stablehlo.broadcast_in_dim %v3391, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3393 = stablehlo.divide %v3392, %v3390 : tensor<64x256x14x14xf32>
    %v3394 = stablehlo.subtract %v3388, %v3393 : tensor<64x256x14x14xf32>
    %v3395 = stablehlo.multiply %v3394, %v3394 : tensor<64x256x14x14xf32>
    %v3396 = stablehlo.reduce(%v3395 init: %v3389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3397 = stablehlo.broadcast_in_dim %v3396, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3398 = stablehlo.divide %v3397, %v3390 : tensor<64x256x14x14xf32>
    %v3399 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3400 = stablehlo.add %v3398, %v3399 : tensor<64x256x14x14xf32>
    %v3401 = stablehlo.rsqrt %v3400 : tensor<64x256x14x14xf32>
    %v3402 = stablehlo.multiply %v3394, %v3401 : tensor<64x256x14x14xf32>
    %v3403 = stablehlo.reshape %v3262 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3404 = stablehlo.multiply %v3403, %v3402 : tensor<64x256x14x14xf32>
    %v3405 = stablehlo.reduce(%v3404 init: %v3389) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3406 = stablehlo.reshape %v3262 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3408 = stablehlo.reduce(%v3406 init: %v3407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3409 = stablehlo.reshape %v1083 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3410 = stablehlo.reshape %v3248 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3411 = stablehlo.transpose %v3409, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3412 = stablehlo.transpose %v3410, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3413 = stablehlo.convert %v3411 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3414 = stablehlo.convert %v3412 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3415 = stablehlo.convolution(%v3413, %v3414)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3416 = stablehlo.convert %v3415 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3417 = stablehlo.transpose %v3416, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3418 = stablehlo.reshape %v1091 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3420 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3421 = stablehlo.reduce(%v3418 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3422 = stablehlo.broadcast_in_dim %v3421, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3423 = stablehlo.divide %v3422, %v3420 : tensor<64x1024x14x14xf32>
    %v3424 = stablehlo.subtract %v3418, %v3423 : tensor<64x1024x14x14xf32>
    %v3425 = stablehlo.multiply %v3424, %v3424 : tensor<64x1024x14x14xf32>
    %v3426 = stablehlo.reduce(%v3425 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3427 = stablehlo.broadcast_in_dim %v3426, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3428 = stablehlo.divide %v3427, %v3420 : tensor<64x1024x14x14xf32>
    %v3429 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3430 = stablehlo.add %v3428, %v3429 : tensor<64x1024x14x14xf32>
    %v3431 = stablehlo.rsqrt %v3430 : tensor<64x1024x14x14xf32>
    %v3432 = stablehlo.multiply %v3424, %v3431 : tensor<64x1024x14x14xf32>
    %v3433 = stablehlo.reshape %v3218 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3434 = stablehlo.multiply %v3433, %v3432 : tensor<64x1024x14x14xf32>
    %v3435 = stablehlo.reduce(%v3434 init: %v3419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3436 = stablehlo.reshape %v3218 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3438 = stablehlo.reduce(%v3436 init: %v3437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3439 = stablehlo.reshape %v3348 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3440 = stablehlo.reshape %v1015 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3441 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v3442 = stablehlo.compare GT, %v3440, %v3441 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v3443 = stablehlo.select %v3442, %v3439, %v3441 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v3444 = stablehlo.reshape %v3443 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3445 = stablehlo.reshape %v991 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3446 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3447 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3448 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3449 = stablehlo.reduce(%v3445 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3449, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3451 = stablehlo.divide %v3450, %v3447 : tensor<64x1024x14x14xf32>
    %v3452 = stablehlo.subtract %v3445, %v3451 : tensor<64x1024x14x14xf32>
    %v3453 = stablehlo.multiply %v3452, %v3452 : tensor<64x1024x14x14xf32>
    %v3454 = stablehlo.reduce(%v3453 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3455 = stablehlo.broadcast_in_dim %v3454, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3456 = stablehlo.divide %v3455, %v3447 : tensor<64x1024x14x14xf32>
    %v3457 = stablehlo.add %v3456, %v3448 : tensor<64x1024x14x14xf32>
    %v3458 = stablehlo.rsqrt %v3457 : tensor<64x1024x14x14xf32>
    %v3459 = stablehlo.multiply %v3452, %v3458 : tensor<64x1024x14x14xf32>
    %v3460 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3461 = stablehlo.reshape %v3444 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3462 = stablehlo.multiply %v3460, %v3461 : tensor<64x1024x14x14xf32>
    %v3463 = stablehlo.reduce(%v3462 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3464 = stablehlo.broadcast_in_dim %v3463, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3465 = stablehlo.multiply %v3459, %v3462 : tensor<64x1024x14x14xf32>
    %v3466 = stablehlo.reduce(%v3465 init: %v3446) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3467 = stablehlo.broadcast_in_dim %v3466, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3468 = stablehlo.multiply %v3462, %v3447 : tensor<64x1024x14x14xf32>
    %v3469 = stablehlo.subtract %v3468, %v3464 : tensor<64x1024x14x14xf32>
    %v3470 = stablehlo.multiply %v3459, %v3467 : tensor<64x1024x14x14xf32>
    %v3471 = stablehlo.subtract %v3469, %v3470 : tensor<64x1024x14x14xf32>
    %v3472 = stablehlo.divide %v3458, %v3447 : tensor<64x1024x14x14xf32>
    %v3473 = stablehlo.multiply %v3472, %v3471 : tensor<64x1024x14x14xf32>
    %v3474 = stablehlo.reshape %v3473 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3475 = stablehlo.reshape %v3474 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3476 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3477 = stablehlo.transpose %v3476, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3478 = stablehlo.convert %v3475 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3479 = stablehlo.convert %v3477 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3480 = stablehlo.convolution(%v3478, %v3479)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3481 = stablehlo.convert %v3480 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3482 = stablehlo.reshape %v3481 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3483 = stablehlo.reshape %v3482 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3484 = stablehlo.reshape %v979 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3485 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3486 = stablehlo.compare GT, %v3484, %v3485 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3487 = stablehlo.select %v3486, %v3483, %v3485 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3488 = stablehlo.reshape %v3487 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3489 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3491 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3492 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3493 = stablehlo.reduce(%v3489 init: %v3490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3494 = stablehlo.broadcast_in_dim %v3493, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3495 = stablehlo.divide %v3494, %v3491 : tensor<64x256x14x14xf32>
    %v3496 = stablehlo.subtract %v3489, %v3495 : tensor<64x256x14x14xf32>
    %v3497 = stablehlo.multiply %v3496, %v3496 : tensor<64x256x14x14xf32>
    %v3498 = stablehlo.reduce(%v3497 init: %v3490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3499 = stablehlo.broadcast_in_dim %v3498, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3500 = stablehlo.divide %v3499, %v3491 : tensor<64x256x14x14xf32>
    %v3501 = stablehlo.add %v3500, %v3492 : tensor<64x256x14x14xf32>
    %v3502 = stablehlo.rsqrt %v3501 : tensor<64x256x14x14xf32>
    %v3503 = stablehlo.multiply %v3496, %v3502 : tensor<64x256x14x14xf32>
    %v3504 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3505 = stablehlo.reshape %v3488 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3506 = stablehlo.multiply %v3504, %v3505 : tensor<64x256x14x14xf32>
    %v3507 = stablehlo.reduce(%v3506 init: %v3490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3507, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3509 = stablehlo.multiply %v3503, %v3506 : tensor<64x256x14x14xf32>
    %v3510 = stablehlo.reduce(%v3509 init: %v3490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3511 = stablehlo.broadcast_in_dim %v3510, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3512 = stablehlo.multiply %v3506, %v3491 : tensor<64x256x14x14xf32>
    %v3513 = stablehlo.subtract %v3512, %v3508 : tensor<64x256x14x14xf32>
    %v3514 = stablehlo.multiply %v3503, %v3511 : tensor<64x256x14x14xf32>
    %v3515 = stablehlo.subtract %v3513, %v3514 : tensor<64x256x14x14xf32>
    %v3516 = stablehlo.divide %v3502, %v3491 : tensor<64x256x14x14xf32>
    %v3517 = stablehlo.multiply %v3516, %v3515 : tensor<64x256x14x14xf32>
    %v3518 = stablehlo.reshape %v3517 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3519 = stablehlo.reshape %v3518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3520 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3521 = stablehlo.transpose %v3520, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3522 = stablehlo.convert %v3519 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3523 = stablehlo.convert %v3521 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3524 = stablehlo.convolution(%v3522, %v3523)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3525 = stablehlo.convert %v3524 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3526 = stablehlo.reshape %v3525 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3527 = stablehlo.reshape %v3526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3528 = stablehlo.reshape %v947 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3529 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3530 = stablehlo.compare GT, %v3528, %v3529 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3531 = stablehlo.select %v3530, %v3527, %v3529 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3533 = stablehlo.reshape %v927 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3534 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3535 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3536 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3537 = stablehlo.reduce(%v3533 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3538 = stablehlo.broadcast_in_dim %v3537, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3539 = stablehlo.divide %v3538, %v3535 : tensor<64x256x14x14xf32>
    %v3540 = stablehlo.subtract %v3533, %v3539 : tensor<64x256x14x14xf32>
    %v3541 = stablehlo.multiply %v3540, %v3540 : tensor<64x256x14x14xf32>
    %v3542 = stablehlo.reduce(%v3541 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3543 = stablehlo.broadcast_in_dim %v3542, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3544 = stablehlo.divide %v3543, %v3535 : tensor<64x256x14x14xf32>
    %v3545 = stablehlo.add %v3544, %v3536 : tensor<64x256x14x14xf32>
    %v3546 = stablehlo.rsqrt %v3545 : tensor<64x256x14x14xf32>
    %v3547 = stablehlo.multiply %v3540, %v3546 : tensor<64x256x14x14xf32>
    %v3548 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3549 = stablehlo.reshape %v3532 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3550 = stablehlo.multiply %v3548, %v3549 : tensor<64x256x14x14xf32>
    %v3551 = stablehlo.reduce(%v3550 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3552 = stablehlo.broadcast_in_dim %v3551, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3553 = stablehlo.multiply %v3547, %v3550 : tensor<64x256x14x14xf32>
    %v3554 = stablehlo.reduce(%v3553 init: %v3534) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3555 = stablehlo.broadcast_in_dim %v3554, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3556 = stablehlo.multiply %v3550, %v3535 : tensor<64x256x14x14xf32>
    %v3557 = stablehlo.subtract %v3556, %v3552 : tensor<64x256x14x14xf32>
    %v3558 = stablehlo.multiply %v3547, %v3555 : tensor<64x256x14x14xf32>
    %v3559 = stablehlo.subtract %v3557, %v3558 : tensor<64x256x14x14xf32>
    %v3560 = stablehlo.divide %v3546, %v3535 : tensor<64x256x14x14xf32>
    %v3561 = stablehlo.multiply %v3560, %v3559 : tensor<64x256x14x14xf32>
    %v3562 = stablehlo.reshape %v3561 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3563 = stablehlo.reshape %v3562 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3564 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3565 = stablehlo.transpose %v3564, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3566 = stablehlo.convert %v3563 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3567 = stablehlo.convert %v3565 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3568 = stablehlo.convolution(%v3566, %v3567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3569 = stablehlo.convert %v3568 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3570 = stablehlo.reshape %v3569 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3571 = stablehlo.reshape %v3570 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3572 = stablehlo.reshape %v3444 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3573 = stablehlo.add %v3571, %v3572 : tensor<64x64x56x56xf32>
    %v3574 = stablehlo.reshape %v3573 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3575 = stablehlo.reshape %v919 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3576 = stablehlo.reshape %v3562 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3577 = stablehlo.transpose %v3575, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3578 = stablehlo.transpose %v3576, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3579 = stablehlo.convert %v3577 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3580 = stablehlo.convert %v3578 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3581 = stablehlo.convolution(%v3579, %v3580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3582 = stablehlo.convert %v3581 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3583 = stablehlo.transpose %v3582, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3584 = stablehlo.reshape %v927 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3585 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3586 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3587 = stablehlo.reduce(%v3584 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3588 = stablehlo.broadcast_in_dim %v3587, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3589 = stablehlo.divide %v3588, %v3586 : tensor<64x256x14x14xf32>
    %v3590 = stablehlo.subtract %v3584, %v3589 : tensor<64x256x14x14xf32>
    %v3591 = stablehlo.multiply %v3590, %v3590 : tensor<64x256x14x14xf32>
    %v3592 = stablehlo.reduce(%v3591 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3593 = stablehlo.broadcast_in_dim %v3592, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3594 = stablehlo.divide %v3593, %v3586 : tensor<64x256x14x14xf32>
    %v3595 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3596 = stablehlo.add %v3594, %v3595 : tensor<64x256x14x14xf32>
    %v3597 = stablehlo.rsqrt %v3596 : tensor<64x256x14x14xf32>
    %v3598 = stablehlo.multiply %v3590, %v3597 : tensor<64x256x14x14xf32>
    %v3599 = stablehlo.reshape %v3532 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3600 = stablehlo.multiply %v3599, %v3598 : tensor<64x256x14x14xf32>
    %v3601 = stablehlo.reduce(%v3600 init: %v3585) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3602 = stablehlo.reshape %v3532 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3604 = stablehlo.reduce(%v3602 init: %v3603) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3605 = stablehlo.reshape %v951 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3606 = stablehlo.reshape %v3518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3607 = stablehlo.transpose %v3605, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3608 = stablehlo.transpose %v3606, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3609 = stablehlo.convert %v3607 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3610 = stablehlo.convert %v3608 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3611 = stablehlo.convolution(%v3609, %v3610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3612 = stablehlo.convert %v3611 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3613 = stablehlo.transpose %v3612, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3614 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3616 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3617 = stablehlo.reduce(%v3614 init: %v3615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3618 = stablehlo.broadcast_in_dim %v3617, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3619 = stablehlo.divide %v3618, %v3616 : tensor<64x256x14x14xf32>
    %v3620 = stablehlo.subtract %v3614, %v3619 : tensor<64x256x14x14xf32>
    %v3621 = stablehlo.multiply %v3620, %v3620 : tensor<64x256x14x14xf32>
    %v3622 = stablehlo.reduce(%v3621 init: %v3615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3623 = stablehlo.broadcast_in_dim %v3622, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3624 = stablehlo.divide %v3623, %v3616 : tensor<64x256x14x14xf32>
    %v3625 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3626 = stablehlo.add %v3624, %v3625 : tensor<64x256x14x14xf32>
    %v3627 = stablehlo.rsqrt %v3626 : tensor<64x256x14x14xf32>
    %v3628 = stablehlo.multiply %v3620, %v3627 : tensor<64x256x14x14xf32>
    %v3629 = stablehlo.reshape %v3488 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3630 = stablehlo.multiply %v3629, %v3628 : tensor<64x256x14x14xf32>
    %v3631 = stablehlo.reduce(%v3630 init: %v3615) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3632 = stablehlo.reshape %v3488 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3633 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3634 = stablehlo.reduce(%v3632 init: %v3633) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3635 = stablehlo.reshape %v983 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3636 = stablehlo.reshape %v3474 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3637 = stablehlo.transpose %v3635, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3638 = stablehlo.transpose %v3636, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3639 = stablehlo.convert %v3637 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3640 = stablehlo.convert %v3638 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3641 = stablehlo.convolution(%v3639, %v3640)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3642 = stablehlo.convert %v3641 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3643 = stablehlo.transpose %v3642, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3644 = stablehlo.reshape %v991 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3647 = stablehlo.reduce(%v3644 init: %v3645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3648 = stablehlo.broadcast_in_dim %v3647, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3649 = stablehlo.divide %v3648, %v3646 : tensor<64x1024x14x14xf32>
    %v3650 = stablehlo.subtract %v3644, %v3649 : tensor<64x1024x14x14xf32>
    %v3651 = stablehlo.multiply %v3650, %v3650 : tensor<64x1024x14x14xf32>
    %v3652 = stablehlo.reduce(%v3651 init: %v3645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3653 = stablehlo.broadcast_in_dim %v3652, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3654 = stablehlo.divide %v3653, %v3646 : tensor<64x1024x14x14xf32>
    %v3655 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3656 = stablehlo.add %v3654, %v3655 : tensor<64x1024x14x14xf32>
    %v3657 = stablehlo.rsqrt %v3656 : tensor<64x1024x14x14xf32>
    %v3658 = stablehlo.multiply %v3650, %v3657 : tensor<64x1024x14x14xf32>
    %v3659 = stablehlo.reshape %v3444 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3660 = stablehlo.multiply %v3659, %v3658 : tensor<64x1024x14x14xf32>
    %v3661 = stablehlo.reduce(%v3660 init: %v3645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3662 = stablehlo.reshape %v3444 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3664 = stablehlo.reduce(%v3662 init: %v3663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3665 = stablehlo.reshape %v3574 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3666 = stablehlo.reshape %v915 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3667 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v3668 = stablehlo.compare GT, %v3666, %v3667 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v3669 = stablehlo.select %v3668, %v3665, %v3667 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v3670 = stablehlo.reshape %v3669 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3671 = stablehlo.reshape %v863 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3673 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3674 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3675 = stablehlo.reduce(%v3671 init: %v3672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3676 = stablehlo.broadcast_in_dim %v3675, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3677 = stablehlo.divide %v3676, %v3673 : tensor<64x1024x14x14xf32>
    %v3678 = stablehlo.subtract %v3671, %v3677 : tensor<64x1024x14x14xf32>
    %v3679 = stablehlo.multiply %v3678, %v3678 : tensor<64x1024x14x14xf32>
    %v3680 = stablehlo.reduce(%v3679 init: %v3672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3681 = stablehlo.broadcast_in_dim %v3680, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3682 = stablehlo.divide %v3681, %v3673 : tensor<64x1024x14x14xf32>
    %v3683 = stablehlo.add %v3682, %v3674 : tensor<64x1024x14x14xf32>
    %v3684 = stablehlo.rsqrt %v3683 : tensor<64x1024x14x14xf32>
    %v3685 = stablehlo.multiply %v3678, %v3684 : tensor<64x1024x14x14xf32>
    %v3686 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3687 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3688 = stablehlo.multiply %v3686, %v3687 : tensor<64x1024x14x14xf32>
    %v3689 = stablehlo.reduce(%v3688 init: %v3672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3690 = stablehlo.broadcast_in_dim %v3689, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3691 = stablehlo.multiply %v3685, %v3688 : tensor<64x1024x14x14xf32>
    %v3692 = stablehlo.reduce(%v3691 init: %v3672) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3693 = stablehlo.broadcast_in_dim %v3692, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3694 = stablehlo.multiply %v3688, %v3673 : tensor<64x1024x14x14xf32>
    %v3695 = stablehlo.subtract %v3694, %v3690 : tensor<64x1024x14x14xf32>
    %v3696 = stablehlo.multiply %v3685, %v3693 : tensor<64x1024x14x14xf32>
    %v3697 = stablehlo.subtract %v3695, %v3696 : tensor<64x1024x14x14xf32>
    %v3698 = stablehlo.divide %v3684, %v3673 : tensor<64x1024x14x14xf32>
    %v3699 = stablehlo.multiply %v3698, %v3697 : tensor<64x1024x14x14xf32>
    %v3700 = stablehlo.reshape %v3699 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3701 = stablehlo.reshape %v3700 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3702 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3703 = stablehlo.transpose %v3702, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3704 = stablehlo.convert %v3701 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3705 = stablehlo.convert %v3703 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3706 = stablehlo.convolution(%v3704, %v3705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3707 = stablehlo.convert %v3706 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3708 = stablehlo.reshape %v3707 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3709 = stablehlo.reshape %v3708 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3710 = stablehlo.reshape %v851 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3712 = stablehlo.compare GT, %v3710, %v3711 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3713 = stablehlo.select %v3712, %v3709, %v3711 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3714 = stablehlo.reshape %v3713 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3715 = stablehlo.reshape %v831 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3716 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3717 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3718 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3719 = stablehlo.reduce(%v3715 init: %v3716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3721 = stablehlo.divide %v3720, %v3717 : tensor<64x256x14x14xf32>
    %v3722 = stablehlo.subtract %v3715, %v3721 : tensor<64x256x14x14xf32>
    %v3723 = stablehlo.multiply %v3722, %v3722 : tensor<64x256x14x14xf32>
    %v3724 = stablehlo.reduce(%v3723 init: %v3716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3725 = stablehlo.broadcast_in_dim %v3724, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3726 = stablehlo.divide %v3725, %v3717 : tensor<64x256x14x14xf32>
    %v3727 = stablehlo.add %v3726, %v3718 : tensor<64x256x14x14xf32>
    %v3728 = stablehlo.rsqrt %v3727 : tensor<64x256x14x14xf32>
    %v3729 = stablehlo.multiply %v3722, %v3728 : tensor<64x256x14x14xf32>
    %v3730 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3731 = stablehlo.reshape %v3714 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3732 = stablehlo.multiply %v3730, %v3731 : tensor<64x256x14x14xf32>
    %v3733 = stablehlo.reduce(%v3732 init: %v3716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3734 = stablehlo.broadcast_in_dim %v3733, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3735 = stablehlo.multiply %v3729, %v3732 : tensor<64x256x14x14xf32>
    %v3736 = stablehlo.reduce(%v3735 init: %v3716) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3737 = stablehlo.broadcast_in_dim %v3736, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3738 = stablehlo.multiply %v3732, %v3717 : tensor<64x256x14x14xf32>
    %v3739 = stablehlo.subtract %v3738, %v3734 : tensor<64x256x14x14xf32>
    %v3740 = stablehlo.multiply %v3729, %v3737 : tensor<64x256x14x14xf32>
    %v3741 = stablehlo.subtract %v3739, %v3740 : tensor<64x256x14x14xf32>
    %v3742 = stablehlo.divide %v3728, %v3717 : tensor<64x256x14x14xf32>
    %v3743 = stablehlo.multiply %v3742, %v3741 : tensor<64x256x14x14xf32>
    %v3744 = stablehlo.reshape %v3743 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.pad %v3745, %v3746, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3748 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3749 = stablehlo.transpose %v3748, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3750 = stablehlo.convert %v3747 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3751 = stablehlo.convert %v3749 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3752 = stablehlo.convolution(%v3750, %v3751)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x28x28xbf16>
    %v3753 = stablehlo.convert %v3752 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v3754 = stablehlo.reshape %v3753 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v3755 = stablehlo.reshape %v3754 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3756 = stablehlo.reshape %v819 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3757 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v3758 = stablehlo.compare GT, %v3756, %v3757 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v3759 = stablehlo.select %v3758, %v3755, %v3757 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v3760 = stablehlo.reshape %v3759 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3761 = stablehlo.reshape %v799 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v3764 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v3765 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3766 = stablehlo.broadcast_in_dim %v3765, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3767 = stablehlo.divide %v3766, %v3763 : tensor<64x256x28x28xf32>
    %v3768 = stablehlo.subtract %v3761, %v3767 : tensor<64x256x28x28xf32>
    %v3769 = stablehlo.multiply %v3768, %v3768 : tensor<64x256x28x28xf32>
    %v3770 = stablehlo.reduce(%v3769 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3771 = stablehlo.broadcast_in_dim %v3770, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3772 = stablehlo.divide %v3771, %v3763 : tensor<64x256x28x28xf32>
    %v3773 = stablehlo.add %v3772, %v3764 : tensor<64x256x28x28xf32>
    %v3774 = stablehlo.rsqrt %v3773 : tensor<64x256x28x28xf32>
    %v3775 = stablehlo.multiply %v3768, %v3774 : tensor<64x256x28x28xf32>
    %v3776 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3777 = stablehlo.reshape %v3760 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3778 = stablehlo.multiply %v3776, %v3777 : tensor<64x256x28x28xf32>
    %v3779 = stablehlo.reduce(%v3778 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3780 = stablehlo.broadcast_in_dim %v3779, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3781 = stablehlo.multiply %v3775, %v3778 : tensor<64x256x28x28xf32>
    %v3782 = stablehlo.reduce(%v3781 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3783 = stablehlo.broadcast_in_dim %v3782, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3784 = stablehlo.multiply %v3778, %v3763 : tensor<64x256x28x28xf32>
    %v3785 = stablehlo.subtract %v3784, %v3780 : tensor<64x256x28x28xf32>
    %v3786 = stablehlo.multiply %v3775, %v3783 : tensor<64x256x28x28xf32>
    %v3787 = stablehlo.subtract %v3785, %v3786 : tensor<64x256x28x28xf32>
    %v3788 = stablehlo.divide %v3774, %v3763 : tensor<64x256x28x28xf32>
    %v3789 = stablehlo.multiply %v3788, %v3787 : tensor<64x256x28x28xf32>
    %v3790 = stablehlo.reshape %v3789 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v3791 = stablehlo.reshape %v3790 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3792 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v3793 = stablehlo.transpose %v3792, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v3794 = stablehlo.convert %v3791 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3795 = stablehlo.convert %v3793 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v3796 = stablehlo.convolution(%v3794, %v3795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v3797 = stablehlo.convert %v3796 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v3798 = stablehlo.reshape %v3797 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3799 = stablehlo.reshape %v891 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3801 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3802 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3803 = stablehlo.reduce(%v3799 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3804 = stablehlo.broadcast_in_dim %v3803, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3805 = stablehlo.divide %v3804, %v3801 : tensor<64x1024x14x14xf32>
    %v3806 = stablehlo.subtract %v3799, %v3805 : tensor<64x1024x14x14xf32>
    %v3807 = stablehlo.multiply %v3806, %v3806 : tensor<64x1024x14x14xf32>
    %v3808 = stablehlo.reduce(%v3807 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3809 = stablehlo.broadcast_in_dim %v3808, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3810 = stablehlo.divide %v3809, %v3801 : tensor<64x1024x14x14xf32>
    %v3811 = stablehlo.add %v3810, %v3802 : tensor<64x1024x14x14xf32>
    %v3812 = stablehlo.rsqrt %v3811 : tensor<64x1024x14x14xf32>
    %v3813 = stablehlo.multiply %v3806, %v3812 : tensor<64x1024x14x14xf32>
    %v3814 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3815 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3816 = stablehlo.multiply %v3814, %v3815 : tensor<64x1024x14x14xf32>
    %v3817 = stablehlo.reduce(%v3816 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3819 = stablehlo.multiply %v3813, %v3816 : tensor<64x1024x14x14xf32>
    %v3820 = stablehlo.reduce(%v3819 init: %v3800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3821 = stablehlo.broadcast_in_dim %v3820, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3822 = stablehlo.multiply %v3816, %v3801 : tensor<64x1024x14x14xf32>
    %v3823 = stablehlo.subtract %v3822, %v3818 : tensor<64x1024x14x14xf32>
    %v3824 = stablehlo.multiply %v3813, %v3821 : tensor<64x1024x14x14xf32>
    %v3825 = stablehlo.subtract %v3823, %v3824 : tensor<64x1024x14x14xf32>
    %v3826 = stablehlo.divide %v3812, %v3801 : tensor<64x1024x14x14xf32>
    %v3827 = stablehlo.multiply %v3826, %v3825 : tensor<64x1024x14x14xf32>
    %v3828 = stablehlo.reshape %v3827 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3829 = stablehlo.reshape %v3828 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3830 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3831 = stablehlo.pad %v3829, %v3830, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v3832 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v3833 = stablehlo.transpose %v3832, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3834 = stablehlo.convert %v3831 : (tensor<64x1024x28x28xf32>) -> tensor<64x1024x28x28xbf16>
    %v3835 = stablehlo.convert %v3833 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v3836 = stablehlo.convolution(%v3834, %v3835)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x28x28xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v3837 = stablehlo.convert %v3836 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v3838 = stablehlo.reshape %v3837 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3839 = stablehlo.reshape %v3798 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v3840 = stablehlo.reshape %v3838 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v3841 = stablehlo.add %v3839, %v3840 : tensor<64x128x56x56xf32>
    %v3842 = stablehlo.reshape %v3841 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v3843 = stablehlo.reshape %v791 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3844 = stablehlo.reshape %v3790 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3845 = stablehlo.transpose %v3843, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3846 = stablehlo.transpose %v3844, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3847 = stablehlo.convert %v3845 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3848 = stablehlo.convert %v3846 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3849 = stablehlo.convolution(%v3847, %v3848)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<512x256x1x1xbf16>
    %v3850 = stablehlo.convert %v3849 : (tensor<512x256x1x1xbf16>) -> tensor<512x256x1x1xf32>
    %v3851 = stablehlo.transpose %v3850, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v3852 = stablehlo.reshape %v799 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3854 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v3855 = stablehlo.reduce(%v3852 init: %v3853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3856 = stablehlo.broadcast_in_dim %v3855, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3857 = stablehlo.divide %v3856, %v3854 : tensor<64x256x28x28xf32>
    %v3858 = stablehlo.subtract %v3852, %v3857 : tensor<64x256x28x28xf32>
    %v3859 = stablehlo.multiply %v3858, %v3858 : tensor<64x256x28x28xf32>
    %v3860 = stablehlo.reduce(%v3859 init: %v3853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3861 = stablehlo.broadcast_in_dim %v3860, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3862 = stablehlo.divide %v3861, %v3854 : tensor<64x256x28x28xf32>
    %v3863 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v3864 = stablehlo.add %v3862, %v3863 : tensor<64x256x28x28xf32>
    %v3865 = stablehlo.rsqrt %v3864 : tensor<64x256x28x28xf32>
    %v3866 = stablehlo.multiply %v3858, %v3865 : tensor<64x256x28x28xf32>
    %v3867 = stablehlo.reshape %v3760 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3868 = stablehlo.multiply %v3867, %v3866 : tensor<64x256x28x28xf32>
    %v3869 = stablehlo.reduce(%v3868 init: %v3853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3870 = stablehlo.reshape %v3760 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3872 = stablehlo.reduce(%v3870 init: %v3871) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3873 = stablehlo.reshape %v823 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3874 = stablehlo.reshape %v3744 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3875 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3876 = stablehlo.pad %v3874, %v3875, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3877 = stablehlo.transpose %v3873, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3878 = stablehlo.transpose %v3876, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3879 = stablehlo.convert %v3877 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3880 = stablehlo.convert %v3878 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3881 = stablehlo.convolution(%v3879, %v3880)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<256x256x3x3xbf16>
    %v3882 = stablehlo.convert %v3881 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3883 = stablehlo.transpose %v3882, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3884 = stablehlo.reshape %v831 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3886 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3887 = stablehlo.reduce(%v3884 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3888 = stablehlo.broadcast_in_dim %v3887, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3889 = stablehlo.divide %v3888, %v3886 : tensor<64x256x14x14xf32>
    %v3890 = stablehlo.subtract %v3884, %v3889 : tensor<64x256x14x14xf32>
    %v3891 = stablehlo.multiply %v3890, %v3890 : tensor<64x256x14x14xf32>
    %v3892 = stablehlo.reduce(%v3891 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3893 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3894 = stablehlo.divide %v3893, %v3886 : tensor<64x256x14x14xf32>
    %v3895 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3896 = stablehlo.add %v3894, %v3895 : tensor<64x256x14x14xf32>
    %v3897 = stablehlo.rsqrt %v3896 : tensor<64x256x14x14xf32>
    %v3898 = stablehlo.multiply %v3890, %v3897 : tensor<64x256x14x14xf32>
    %v3899 = stablehlo.reshape %v3714 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3900 = stablehlo.multiply %v3899, %v3898 : tensor<64x256x14x14xf32>
    %v3901 = stablehlo.reduce(%v3900 init: %v3885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3902 = stablehlo.reshape %v3714 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3903 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3904 = stablehlo.reduce(%v3902 init: %v3903) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3905 = stablehlo.reshape %v855 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3906 = stablehlo.reshape %v3700 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3907 = stablehlo.transpose %v3905, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3908 = stablehlo.transpose %v3906, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3909 = stablehlo.convert %v3907 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3910 = stablehlo.convert %v3908 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3911 = stablehlo.convolution(%v3909, %v3910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3912 = stablehlo.convert %v3911 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3913 = stablehlo.transpose %v3912, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3914 = stablehlo.reshape %v863 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3916 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3917 = stablehlo.reduce(%v3914 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3917, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3919 = stablehlo.divide %v3918, %v3916 : tensor<64x1024x14x14xf32>
    %v3920 = stablehlo.subtract %v3914, %v3919 : tensor<64x1024x14x14xf32>
    %v3921 = stablehlo.multiply %v3920, %v3920 : tensor<64x1024x14x14xf32>
    %v3922 = stablehlo.reduce(%v3921 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3923 = stablehlo.broadcast_in_dim %v3922, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3924 = stablehlo.divide %v3923, %v3916 : tensor<64x1024x14x14xf32>
    %v3925 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3926 = stablehlo.add %v3924, %v3925 : tensor<64x1024x14x14xf32>
    %v3927 = stablehlo.rsqrt %v3926 : tensor<64x1024x14x14xf32>
    %v3928 = stablehlo.multiply %v3920, %v3927 : tensor<64x1024x14x14xf32>
    %v3929 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3930 = stablehlo.multiply %v3929, %v3928 : tensor<64x1024x14x14xf32>
    %v3931 = stablehlo.reduce(%v3930 init: %v3915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3932 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3934 = stablehlo.reduce(%v3932 init: %v3933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3935 = stablehlo.reshape %v791 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3936 = stablehlo.reshape %v3828 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3937 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3938 = stablehlo.pad %v3936, %v3937, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v3939 = stablehlo.transpose %v3935, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3940 = stablehlo.transpose %v3938, dims = [1, 0, 2, 3] : (tensor<64x1024x28x28xf32>) -> tensor<1024x64x28x28xf32>
    %v3941 = stablehlo.convert %v3939 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3942 = stablehlo.convert %v3940 : (tensor<1024x64x28x28xf32>) -> tensor<1024x64x28x28xbf16>
    %v3943 = stablehlo.convolution(%v3941, %v3942)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<1024x64x28x28xbf16>) -> tensor<512x1024x1x1xbf16>
    %v3944 = stablehlo.convert %v3943 : (tensor<512x1024x1x1xbf16>) -> tensor<512x1024x1x1xf32>
    %v3945 = stablehlo.transpose %v3944, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3946 = stablehlo.reshape %v891 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3948 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3949 = stablehlo.reduce(%v3946 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3950 = stablehlo.broadcast_in_dim %v3949, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3951 = stablehlo.divide %v3950, %v3948 : tensor<64x1024x14x14xf32>
    %v3952 = stablehlo.subtract %v3946, %v3951 : tensor<64x1024x14x14xf32>
    %v3953 = stablehlo.multiply %v3952, %v3952 : tensor<64x1024x14x14xf32>
    %v3954 = stablehlo.reduce(%v3953 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3955 = stablehlo.broadcast_in_dim %v3954, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3956 = stablehlo.divide %v3955, %v3948 : tensor<64x1024x14x14xf32>
    %v3957 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3958 = stablehlo.add %v3956, %v3957 : tensor<64x1024x14x14xf32>
    %v3959 = stablehlo.rsqrt %v3958 : tensor<64x1024x14x14xf32>
    %v3960 = stablehlo.multiply %v3952, %v3959 : tensor<64x1024x14x14xf32>
    %v3961 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3962 = stablehlo.multiply %v3961, %v3960 : tensor<64x1024x14x14xf32>
    %v3963 = stablehlo.reduce(%v3962 init: %v3947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3964 = stablehlo.reshape %v3670 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3965 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3966 = stablehlo.reduce(%v3964 init: %v3965) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3967 = stablehlo.reshape %v3842 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v3968 = stablehlo.reshape %v787 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v3969 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v3970 = stablehlo.compare GT, %v3968, %v3969 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v3971 = stablehlo.select %v3970, %v3967, %v3969 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v3972 = stablehlo.reshape %v3971 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v3973 = stablehlo.reshape %v763 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3974 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3975 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v3976 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v3977 = stablehlo.reduce(%v3973 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3979 = stablehlo.divide %v3978, %v3975 : tensor<64x512x28x28xf32>
    %v3980 = stablehlo.subtract %v3973, %v3979 : tensor<64x512x28x28xf32>
    %v3981 = stablehlo.multiply %v3980, %v3980 : tensor<64x512x28x28xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3983 = stablehlo.broadcast_in_dim %v3982, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3984 = stablehlo.divide %v3983, %v3975 : tensor<64x512x28x28xf32>
    %v3985 = stablehlo.add %v3984, %v3976 : tensor<64x512x28x28xf32>
    %v3986 = stablehlo.rsqrt %v3985 : tensor<64x512x28x28xf32>
    %v3987 = stablehlo.multiply %v3980, %v3986 : tensor<64x512x28x28xf32>
    %v3988 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3989 = stablehlo.reshape %v3972 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3990 = stablehlo.multiply %v3988, %v3989 : tensor<64x512x28x28xf32>
    %v3991 = stablehlo.reduce(%v3990 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3992 = stablehlo.broadcast_in_dim %v3991, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3993 = stablehlo.multiply %v3987, %v3990 : tensor<64x512x28x28xf32>
    %v3994 = stablehlo.reduce(%v3993 init: %v3974) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3995 = stablehlo.broadcast_in_dim %v3994, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3996 = stablehlo.multiply %v3990, %v3975 : tensor<64x512x28x28xf32>
    %v3997 = stablehlo.subtract %v3996, %v3992 : tensor<64x512x28x28xf32>
    %v3998 = stablehlo.multiply %v3987, %v3995 : tensor<64x512x28x28xf32>
    %v3999 = stablehlo.subtract %v3997, %v3998 : tensor<64x512x28x28xf32>
    %v4000 = stablehlo.divide %v3986, %v3975 : tensor<64x512x28x28xf32>
    %v4001 = stablehlo.multiply %v4000, %v3999 : tensor<64x512x28x28xf32>
    %v4002 = stablehlo.reshape %v4001 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4003 = stablehlo.reshape %v4002 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4004 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4005 = stablehlo.transpose %v4004, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4006 = stablehlo.convert %v4003 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4007 = stablehlo.convert %v4005 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4008 = stablehlo.convolution(%v4006, %v4007)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4009 = stablehlo.convert %v4008 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4010 = stablehlo.reshape %v4009 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4011 = stablehlo.reshape %v4010 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4012 = stablehlo.reshape %v751 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4014 = stablehlo.compare GT, %v4012, %v4013 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4015 = stablehlo.select %v4014, %v4011, %v4013 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4016 = stablehlo.reshape %v4015 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4017 = stablehlo.reshape %v731 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4020 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4021 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4022 = stablehlo.broadcast_in_dim %v4021, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4023 = stablehlo.divide %v4022, %v4019 : tensor<64x128x28x28xf32>
    %v4024 = stablehlo.subtract %v4017, %v4023 : tensor<64x128x28x28xf32>
    %v4025 = stablehlo.multiply %v4024, %v4024 : tensor<64x128x28x28xf32>
    %v4026 = stablehlo.reduce(%v4025 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4027 = stablehlo.broadcast_in_dim %v4026, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4028 = stablehlo.divide %v4027, %v4019 : tensor<64x128x28x28xf32>
    %v4029 = stablehlo.add %v4028, %v4020 : tensor<64x128x28x28xf32>
    %v4030 = stablehlo.rsqrt %v4029 : tensor<64x128x28x28xf32>
    %v4031 = stablehlo.multiply %v4024, %v4030 : tensor<64x128x28x28xf32>
    %v4032 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4033 = stablehlo.reshape %v4016 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4034 = stablehlo.multiply %v4032, %v4033 : tensor<64x128x28x28xf32>
    %v4035 = stablehlo.reduce(%v4034 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4036 = stablehlo.broadcast_in_dim %v4035, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4037 = stablehlo.multiply %v4031, %v4034 : tensor<64x128x28x28xf32>
    %v4038 = stablehlo.reduce(%v4037 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4039 = stablehlo.broadcast_in_dim %v4038, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4040 = stablehlo.multiply %v4034, %v4019 : tensor<64x128x28x28xf32>
    %v4041 = stablehlo.subtract %v4040, %v4036 : tensor<64x128x28x28xf32>
    %v4042 = stablehlo.multiply %v4031, %v4039 : tensor<64x128x28x28xf32>
    %v4043 = stablehlo.subtract %v4041, %v4042 : tensor<64x128x28x28xf32>
    %v4044 = stablehlo.divide %v4030, %v4019 : tensor<64x128x28x28xf32>
    %v4045 = stablehlo.multiply %v4044, %v4043 : tensor<64x128x28x28xf32>
    %v4046 = stablehlo.reshape %v4045 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4047 = stablehlo.reshape %v4046 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4048 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4049 = stablehlo.transpose %v4048, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4050 = stablehlo.convert %v4047 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4051 = stablehlo.convert %v4049 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4052 = stablehlo.convolution(%v4050, %v4051)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v4053 = stablehlo.convert %v4052 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4054 = stablehlo.reshape %v4053 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4055 = stablehlo.reshape %v4054 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4056 = stablehlo.reshape %v719 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4057 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4058 = stablehlo.compare GT, %v4056, %v4057 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4059 = stablehlo.select %v4058, %v4055, %v4057 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4060 = stablehlo.reshape %v4059 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4061 = stablehlo.reshape %v699 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4062 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4063 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4064 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4065 = stablehlo.reduce(%v4061 init: %v4062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4066 = stablehlo.broadcast_in_dim %v4065, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4067 = stablehlo.divide %v4066, %v4063 : tensor<64x128x28x28xf32>
    %v4068 = stablehlo.subtract %v4061, %v4067 : tensor<64x128x28x28xf32>
    %v4069 = stablehlo.multiply %v4068, %v4068 : tensor<64x128x28x28xf32>
    %v4070 = stablehlo.reduce(%v4069 init: %v4062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4071 = stablehlo.broadcast_in_dim %v4070, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4072 = stablehlo.divide %v4071, %v4063 : tensor<64x128x28x28xf32>
    %v4073 = stablehlo.add %v4072, %v4064 : tensor<64x128x28x28xf32>
    %v4074 = stablehlo.rsqrt %v4073 : tensor<64x128x28x28xf32>
    %v4075 = stablehlo.multiply %v4068, %v4074 : tensor<64x128x28x28xf32>
    %v4076 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4077 = stablehlo.reshape %v4060 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4078 = stablehlo.multiply %v4076, %v4077 : tensor<64x128x28x28xf32>
    %v4079 = stablehlo.reduce(%v4078 init: %v4062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4080 = stablehlo.broadcast_in_dim %v4079, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4081 = stablehlo.multiply %v4075, %v4078 : tensor<64x128x28x28xf32>
    %v4082 = stablehlo.reduce(%v4081 init: %v4062) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4083 = stablehlo.broadcast_in_dim %v4082, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4084 = stablehlo.multiply %v4078, %v4063 : tensor<64x128x28x28xf32>
    %v4085 = stablehlo.subtract %v4084, %v4080 : tensor<64x128x28x28xf32>
    %v4086 = stablehlo.multiply %v4075, %v4083 : tensor<64x128x28x28xf32>
    %v4087 = stablehlo.subtract %v4085, %v4086 : tensor<64x128x28x28xf32>
    %v4088 = stablehlo.divide %v4074, %v4063 : tensor<64x128x28x28xf32>
    %v4089 = stablehlo.multiply %v4088, %v4087 : tensor<64x128x28x28xf32>
    %v4090 = stablehlo.reshape %v4089 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4091 = stablehlo.reshape %v4090 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4092 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4093 = stablehlo.transpose %v4092, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4094 = stablehlo.convert %v4091 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4095 = stablehlo.convert %v4093 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v4096 = stablehlo.convolution(%v4094, %v4095)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4097 = stablehlo.convert %v4096 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4098 = stablehlo.reshape %v4097 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4099 = stablehlo.reshape %v4098 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4100 = stablehlo.reshape %v3972 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4101 = stablehlo.add %v4099, %v4100 : tensor<64x128x56x56xf32>
    %v4102 = stablehlo.reshape %v4101 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4103 = stablehlo.reshape %v691 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4104 = stablehlo.reshape %v4090 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4105 = stablehlo.transpose %v4103, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4106 = stablehlo.transpose %v4104, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4107 = stablehlo.convert %v4105 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4108 = stablehlo.convert %v4106 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4109 = stablehlo.convolution(%v4107, %v4108)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v4110 = stablehlo.convert %v4109 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v4111 = stablehlo.transpose %v4110, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4112 = stablehlo.reshape %v699 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4113 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4114 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4115 = stablehlo.reduce(%v4112 init: %v4113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4116 = stablehlo.broadcast_in_dim %v4115, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4117 = stablehlo.divide %v4116, %v4114 : tensor<64x128x28x28xf32>
    %v4118 = stablehlo.subtract %v4112, %v4117 : tensor<64x128x28x28xf32>
    %v4119 = stablehlo.multiply %v4118, %v4118 : tensor<64x128x28x28xf32>
    %v4120 = stablehlo.reduce(%v4119 init: %v4113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4121 = stablehlo.broadcast_in_dim %v4120, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4122 = stablehlo.divide %v4121, %v4114 : tensor<64x128x28x28xf32>
    %v4123 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4124 = stablehlo.add %v4122, %v4123 : tensor<64x128x28x28xf32>
    %v4125 = stablehlo.rsqrt %v4124 : tensor<64x128x28x28xf32>
    %v4126 = stablehlo.multiply %v4118, %v4125 : tensor<64x128x28x28xf32>
    %v4127 = stablehlo.reshape %v4060 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4128 = stablehlo.multiply %v4127, %v4126 : tensor<64x128x28x28xf32>
    %v4129 = stablehlo.reduce(%v4128 init: %v4113) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4130 = stablehlo.reshape %v4060 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4133 = stablehlo.reshape %v723 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4134 = stablehlo.reshape %v4046 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4135 = stablehlo.transpose %v4133, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4136 = stablehlo.transpose %v4134, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4137 = stablehlo.convert %v4135 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4138 = stablehlo.convert %v4136 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4139 = stablehlo.convolution(%v4137, %v4138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4140 = stablehlo.convert %v4139 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4141 = stablehlo.transpose %v4140, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4142 = stablehlo.reshape %v731 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4143 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4144 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4145 = stablehlo.reduce(%v4142 init: %v4143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4146 = stablehlo.broadcast_in_dim %v4145, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4147 = stablehlo.divide %v4146, %v4144 : tensor<64x128x28x28xf32>
    %v4148 = stablehlo.subtract %v4142, %v4147 : tensor<64x128x28x28xf32>
    %v4149 = stablehlo.multiply %v4148, %v4148 : tensor<64x128x28x28xf32>
    %v4150 = stablehlo.reduce(%v4149 init: %v4143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4151 = stablehlo.broadcast_in_dim %v4150, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4152 = stablehlo.divide %v4151, %v4144 : tensor<64x128x28x28xf32>
    %v4153 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4154 = stablehlo.add %v4152, %v4153 : tensor<64x128x28x28xf32>
    %v4155 = stablehlo.rsqrt %v4154 : tensor<64x128x28x28xf32>
    %v4156 = stablehlo.multiply %v4148, %v4155 : tensor<64x128x28x28xf32>
    %v4157 = stablehlo.reshape %v4016 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4158 = stablehlo.multiply %v4157, %v4156 : tensor<64x128x28x28xf32>
    %v4159 = stablehlo.reduce(%v4158 init: %v4143) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4160 = stablehlo.reshape %v4016 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4161 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4162 = stablehlo.reduce(%v4160 init: %v4161) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4163 = stablehlo.reshape %v755 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4164 = stablehlo.reshape %v4002 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4165 = stablehlo.transpose %v4163, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4166 = stablehlo.transpose %v4164, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4167 = stablehlo.convert %v4165 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4168 = stablehlo.convert %v4166 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4169 = stablehlo.convolution(%v4167, %v4168)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4170 = stablehlo.convert %v4169 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4171 = stablehlo.transpose %v4170, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4172 = stablehlo.reshape %v763 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4173 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4174 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4175 = stablehlo.reduce(%v4172 init: %v4173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4176 = stablehlo.broadcast_in_dim %v4175, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4177 = stablehlo.divide %v4176, %v4174 : tensor<64x512x28x28xf32>
    %v4178 = stablehlo.subtract %v4172, %v4177 : tensor<64x512x28x28xf32>
    %v4179 = stablehlo.multiply %v4178, %v4178 : tensor<64x512x28x28xf32>
    %v4180 = stablehlo.reduce(%v4179 init: %v4173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4181 = stablehlo.broadcast_in_dim %v4180, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4182 = stablehlo.divide %v4181, %v4174 : tensor<64x512x28x28xf32>
    %v4183 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4184 = stablehlo.add %v4182, %v4183 : tensor<64x512x28x28xf32>
    %v4185 = stablehlo.rsqrt %v4184 : tensor<64x512x28x28xf32>
    %v4186 = stablehlo.multiply %v4178, %v4185 : tensor<64x512x28x28xf32>
    %v4187 = stablehlo.reshape %v3972 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4188 = stablehlo.multiply %v4187, %v4186 : tensor<64x512x28x28xf32>
    %v4189 = stablehlo.reduce(%v4188 init: %v4173) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4190 = stablehlo.reshape %v3972 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4191 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4192 = stablehlo.reduce(%v4190 init: %v4191) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4193 = stablehlo.reshape %v4102 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4194 = stablehlo.reshape %v687 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4195 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v4196 = stablehlo.compare GT, %v4194, %v4195 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v4197 = stablehlo.select %v4196, %v4193, %v4195 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v4198 = stablehlo.reshape %v4197 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4199 = stablehlo.reshape %v663 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4201 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4202 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4203 = stablehlo.reduce(%v4199 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4204 = stablehlo.broadcast_in_dim %v4203, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4205 = stablehlo.divide %v4204, %v4201 : tensor<64x512x28x28xf32>
    %v4206 = stablehlo.subtract %v4199, %v4205 : tensor<64x512x28x28xf32>
    %v4207 = stablehlo.multiply %v4206, %v4206 : tensor<64x512x28x28xf32>
    %v4208 = stablehlo.reduce(%v4207 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4209 = stablehlo.broadcast_in_dim %v4208, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4210 = stablehlo.divide %v4209, %v4201 : tensor<64x512x28x28xf32>
    %v4211 = stablehlo.add %v4210, %v4202 : tensor<64x512x28x28xf32>
    %v4212 = stablehlo.rsqrt %v4211 : tensor<64x512x28x28xf32>
    %v4213 = stablehlo.multiply %v4206, %v4212 : tensor<64x512x28x28xf32>
    %v4214 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4215 = stablehlo.reshape %v4198 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4216 = stablehlo.multiply %v4214, %v4215 : tensor<64x512x28x28xf32>
    %v4217 = stablehlo.reduce(%v4216 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4218 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4219 = stablehlo.multiply %v4213, %v4216 : tensor<64x512x28x28xf32>
    %v4220 = stablehlo.reduce(%v4219 init: %v4200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4221 = stablehlo.broadcast_in_dim %v4220, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4222 = stablehlo.multiply %v4216, %v4201 : tensor<64x512x28x28xf32>
    %v4223 = stablehlo.subtract %v4222, %v4218 : tensor<64x512x28x28xf32>
    %v4224 = stablehlo.multiply %v4213, %v4221 : tensor<64x512x28x28xf32>
    %v4225 = stablehlo.subtract %v4223, %v4224 : tensor<64x512x28x28xf32>
    %v4226 = stablehlo.divide %v4212, %v4201 : tensor<64x512x28x28xf32>
    %v4227 = stablehlo.multiply %v4226, %v4225 : tensor<64x512x28x28xf32>
    %v4228 = stablehlo.reshape %v4227 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4229 = stablehlo.reshape %v4228 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4230 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4231 = stablehlo.transpose %v4230, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4232 = stablehlo.convert %v4229 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4233 = stablehlo.convert %v4231 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4234 = stablehlo.convolution(%v4232, %v4233)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4235 = stablehlo.convert %v4234 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4236 = stablehlo.reshape %v4235 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4237 = stablehlo.reshape %v4236 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4238 = stablehlo.reshape %v651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4239 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4240 = stablehlo.compare GT, %v4238, %v4239 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4241 = stablehlo.select %v4240, %v4237, %v4239 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4242 = stablehlo.reshape %v4241 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4243 = stablehlo.reshape %v631 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4244 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4245 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4246 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4247 = stablehlo.reduce(%v4243 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4248 = stablehlo.broadcast_in_dim %v4247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4249 = stablehlo.divide %v4248, %v4245 : tensor<64x128x28x28xf32>
    %v4250 = stablehlo.subtract %v4243, %v4249 : tensor<64x128x28x28xf32>
    %v4251 = stablehlo.multiply %v4250, %v4250 : tensor<64x128x28x28xf32>
    %v4252 = stablehlo.reduce(%v4251 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4253 = stablehlo.broadcast_in_dim %v4252, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4254 = stablehlo.divide %v4253, %v4245 : tensor<64x128x28x28xf32>
    %v4255 = stablehlo.add %v4254, %v4246 : tensor<64x128x28x28xf32>
    %v4256 = stablehlo.rsqrt %v4255 : tensor<64x128x28x28xf32>
    %v4257 = stablehlo.multiply %v4250, %v4256 : tensor<64x128x28x28xf32>
    %v4258 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4259 = stablehlo.reshape %v4242 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4260 = stablehlo.multiply %v4258, %v4259 : tensor<64x128x28x28xf32>
    %v4261 = stablehlo.reduce(%v4260 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4262 = stablehlo.broadcast_in_dim %v4261, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4263 = stablehlo.multiply %v4257, %v4260 : tensor<64x128x28x28xf32>
    %v4264 = stablehlo.reduce(%v4263 init: %v4244) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4265 = stablehlo.broadcast_in_dim %v4264, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4266 = stablehlo.multiply %v4260, %v4245 : tensor<64x128x28x28xf32>
    %v4267 = stablehlo.subtract %v4266, %v4262 : tensor<64x128x28x28xf32>
    %v4268 = stablehlo.multiply %v4257, %v4265 : tensor<64x128x28x28xf32>
    %v4269 = stablehlo.subtract %v4267, %v4268 : tensor<64x128x28x28xf32>
    %v4270 = stablehlo.divide %v4256, %v4245 : tensor<64x128x28x28xf32>
    %v4271 = stablehlo.multiply %v4270, %v4269 : tensor<64x128x28x28xf32>
    %v4272 = stablehlo.reshape %v4271 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4273 = stablehlo.reshape %v4272 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4274 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4275 = stablehlo.transpose %v4274, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4276 = stablehlo.convert %v4273 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4277 = stablehlo.convert %v4275 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4278 = stablehlo.convolution(%v4276, %v4277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v4279 = stablehlo.convert %v4278 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4280 = stablehlo.reshape %v4279 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4281 = stablehlo.reshape %v4280 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4282 = stablehlo.reshape %v619 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4283 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4284 = stablehlo.compare GT, %v4282, %v4283 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4285 = stablehlo.select %v4284, %v4281, %v4283 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4286 = stablehlo.reshape %v4285 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4287 = stablehlo.reshape %v599 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4288 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4289 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4290 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4291 = stablehlo.reduce(%v4287 init: %v4288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4292 = stablehlo.broadcast_in_dim %v4291, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4293 = stablehlo.divide %v4292, %v4289 : tensor<64x128x28x28xf32>
    %v4294 = stablehlo.subtract %v4287, %v4293 : tensor<64x128x28x28xf32>
    %v4295 = stablehlo.multiply %v4294, %v4294 : tensor<64x128x28x28xf32>
    %v4296 = stablehlo.reduce(%v4295 init: %v4288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4297 = stablehlo.broadcast_in_dim %v4296, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4298 = stablehlo.divide %v4297, %v4289 : tensor<64x128x28x28xf32>
    %v4299 = stablehlo.add %v4298, %v4290 : tensor<64x128x28x28xf32>
    %v4300 = stablehlo.rsqrt %v4299 : tensor<64x128x28x28xf32>
    %v4301 = stablehlo.multiply %v4294, %v4300 : tensor<64x128x28x28xf32>
    %v4302 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4303 = stablehlo.reshape %v4286 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4304 = stablehlo.multiply %v4302, %v4303 : tensor<64x128x28x28xf32>
    %v4305 = stablehlo.reduce(%v4304 init: %v4288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4306 = stablehlo.broadcast_in_dim %v4305, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4307 = stablehlo.multiply %v4301, %v4304 : tensor<64x128x28x28xf32>
    %v4308 = stablehlo.reduce(%v4307 init: %v4288) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4309 = stablehlo.broadcast_in_dim %v4308, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4310 = stablehlo.multiply %v4304, %v4289 : tensor<64x128x28x28xf32>
    %v4311 = stablehlo.subtract %v4310, %v4306 : tensor<64x128x28x28xf32>
    %v4312 = stablehlo.multiply %v4301, %v4309 : tensor<64x128x28x28xf32>
    %v4313 = stablehlo.subtract %v4311, %v4312 : tensor<64x128x28x28xf32>
    %v4314 = stablehlo.divide %v4300, %v4289 : tensor<64x128x28x28xf32>
    %v4315 = stablehlo.multiply %v4314, %v4313 : tensor<64x128x28x28xf32>
    %v4316 = stablehlo.reshape %v4315 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4317 = stablehlo.reshape %v4316 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4318 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4319 = stablehlo.transpose %v4318, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4320 = stablehlo.convert %v4317 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4321 = stablehlo.convert %v4319 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v4322 = stablehlo.convolution(%v4320, %v4321)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4323 = stablehlo.convert %v4322 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4324 = stablehlo.reshape %v4323 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4325 = stablehlo.reshape %v4324 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4326 = stablehlo.reshape %v4198 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4327 = stablehlo.add %v4325, %v4326 : tensor<64x128x56x56xf32>
    %v4328 = stablehlo.reshape %v4327 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4329 = stablehlo.reshape %v591 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4330 = stablehlo.reshape %v4316 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4331 = stablehlo.transpose %v4329, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4332 = stablehlo.transpose %v4330, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4333 = stablehlo.convert %v4331 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4334 = stablehlo.convert %v4332 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4335 = stablehlo.convolution(%v4333, %v4334)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v4336 = stablehlo.convert %v4335 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v4337 = stablehlo.transpose %v4336, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4338 = stablehlo.reshape %v599 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4340 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4341 = stablehlo.reduce(%v4338 init: %v4339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4342 = stablehlo.broadcast_in_dim %v4341, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4343 = stablehlo.divide %v4342, %v4340 : tensor<64x128x28x28xf32>
    %v4344 = stablehlo.subtract %v4338, %v4343 : tensor<64x128x28x28xf32>
    %v4345 = stablehlo.multiply %v4344, %v4344 : tensor<64x128x28x28xf32>
    %v4346 = stablehlo.reduce(%v4345 init: %v4339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4347 = stablehlo.broadcast_in_dim %v4346, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4348 = stablehlo.divide %v4347, %v4340 : tensor<64x128x28x28xf32>
    %v4349 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4350 = stablehlo.add %v4348, %v4349 : tensor<64x128x28x28xf32>
    %v4351 = stablehlo.rsqrt %v4350 : tensor<64x128x28x28xf32>
    %v4352 = stablehlo.multiply %v4344, %v4351 : tensor<64x128x28x28xf32>
    %v4353 = stablehlo.reshape %v4286 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4354 = stablehlo.multiply %v4353, %v4352 : tensor<64x128x28x28xf32>
    %v4355 = stablehlo.reduce(%v4354 init: %v4339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4356 = stablehlo.reshape %v4286 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4358 = stablehlo.reduce(%v4356 init: %v4357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4359 = stablehlo.reshape %v623 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4360 = stablehlo.reshape %v4272 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4361 = stablehlo.transpose %v4359, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4362 = stablehlo.transpose %v4360, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4363 = stablehlo.convert %v4361 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4364 = stablehlo.convert %v4362 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4365 = stablehlo.convolution(%v4363, %v4364)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4366 = stablehlo.convert %v4365 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4367 = stablehlo.transpose %v4366, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4368 = stablehlo.reshape %v631 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4369 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4370 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4371 = stablehlo.reduce(%v4368 init: %v4369) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4372 = stablehlo.broadcast_in_dim %v4371, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4373 = stablehlo.divide %v4372, %v4370 : tensor<64x128x28x28xf32>
    %v4374 = stablehlo.subtract %v4368, %v4373 : tensor<64x128x28x28xf32>
    %v4375 = stablehlo.multiply %v4374, %v4374 : tensor<64x128x28x28xf32>
    %v4376 = stablehlo.reduce(%v4375 init: %v4369) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4377 = stablehlo.broadcast_in_dim %v4376, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4378 = stablehlo.divide %v4377, %v4370 : tensor<64x128x28x28xf32>
    %v4379 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4380 = stablehlo.add %v4378, %v4379 : tensor<64x128x28x28xf32>
    %v4381 = stablehlo.rsqrt %v4380 : tensor<64x128x28x28xf32>
    %v4382 = stablehlo.multiply %v4374, %v4381 : tensor<64x128x28x28xf32>
    %v4383 = stablehlo.reshape %v4242 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4384 = stablehlo.multiply %v4383, %v4382 : tensor<64x128x28x28xf32>
    %v4385 = stablehlo.reduce(%v4384 init: %v4369) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4386 = stablehlo.reshape %v4242 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4387 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4388 = stablehlo.reduce(%v4386 init: %v4387) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4389 = stablehlo.reshape %v655 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4390 = stablehlo.reshape %v4228 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4391 = stablehlo.transpose %v4389, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4392 = stablehlo.transpose %v4390, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4393 = stablehlo.convert %v4391 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4394 = stablehlo.convert %v4392 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4395 = stablehlo.convolution(%v4393, %v4394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4396 = stablehlo.convert %v4395 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4397 = stablehlo.transpose %v4396, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4398 = stablehlo.reshape %v663 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4399 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4400 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4401 = stablehlo.reduce(%v4398 init: %v4399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4402 = stablehlo.broadcast_in_dim %v4401, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4403 = stablehlo.divide %v4402, %v4400 : tensor<64x512x28x28xf32>
    %v4404 = stablehlo.subtract %v4398, %v4403 : tensor<64x512x28x28xf32>
    %v4405 = stablehlo.multiply %v4404, %v4404 : tensor<64x512x28x28xf32>
    %v4406 = stablehlo.reduce(%v4405 init: %v4399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4407 = stablehlo.broadcast_in_dim %v4406, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4408 = stablehlo.divide %v4407, %v4400 : tensor<64x512x28x28xf32>
    %v4409 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4410 = stablehlo.add %v4408, %v4409 : tensor<64x512x28x28xf32>
    %v4411 = stablehlo.rsqrt %v4410 : tensor<64x512x28x28xf32>
    %v4412 = stablehlo.multiply %v4404, %v4411 : tensor<64x512x28x28xf32>
    %v4413 = stablehlo.reshape %v4198 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4414 = stablehlo.multiply %v4413, %v4412 : tensor<64x512x28x28xf32>
    %v4415 = stablehlo.reduce(%v4414 init: %v4399) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4416 = stablehlo.reshape %v4198 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4418 = stablehlo.reduce(%v4416 init: %v4417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4419 = stablehlo.reshape %v4328 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4420 = stablehlo.reshape %v587 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4421 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v4422 = stablehlo.compare GT, %v4420, %v4421 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v4423 = stablehlo.select %v4422, %v4419, %v4421 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v4424 = stablehlo.reshape %v4423 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4425 = stablehlo.reshape %v563 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4427 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4428 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4429 = stablehlo.reduce(%v4425 init: %v4426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4430 = stablehlo.broadcast_in_dim %v4429, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4431 = stablehlo.divide %v4430, %v4427 : tensor<64x512x28x28xf32>
    %v4432 = stablehlo.subtract %v4425, %v4431 : tensor<64x512x28x28xf32>
    %v4433 = stablehlo.multiply %v4432, %v4432 : tensor<64x512x28x28xf32>
    %v4434 = stablehlo.reduce(%v4433 init: %v4426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4435 = stablehlo.broadcast_in_dim %v4434, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4436 = stablehlo.divide %v4435, %v4427 : tensor<64x512x28x28xf32>
    %v4437 = stablehlo.add %v4436, %v4428 : tensor<64x512x28x28xf32>
    %v4438 = stablehlo.rsqrt %v4437 : tensor<64x512x28x28xf32>
    %v4439 = stablehlo.multiply %v4432, %v4438 : tensor<64x512x28x28xf32>
    %v4440 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4441 = stablehlo.reshape %v4424 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4442 = stablehlo.multiply %v4440, %v4441 : tensor<64x512x28x28xf32>
    %v4443 = stablehlo.reduce(%v4442 init: %v4426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4444 = stablehlo.broadcast_in_dim %v4443, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4445 = stablehlo.multiply %v4439, %v4442 : tensor<64x512x28x28xf32>
    %v4446 = stablehlo.reduce(%v4445 init: %v4426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4447 = stablehlo.broadcast_in_dim %v4446, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4448 = stablehlo.multiply %v4442, %v4427 : tensor<64x512x28x28xf32>
    %v4449 = stablehlo.subtract %v4448, %v4444 : tensor<64x512x28x28xf32>
    %v4450 = stablehlo.multiply %v4439, %v4447 : tensor<64x512x28x28xf32>
    %v4451 = stablehlo.subtract %v4449, %v4450 : tensor<64x512x28x28xf32>
    %v4452 = stablehlo.divide %v4438, %v4427 : tensor<64x512x28x28xf32>
    %v4453 = stablehlo.multiply %v4452, %v4451 : tensor<64x512x28x28xf32>
    %v4454 = stablehlo.reshape %v4453 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4455 = stablehlo.reshape %v4454 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4456 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4457 = stablehlo.transpose %v4456, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4458 = stablehlo.convert %v4455 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4459 = stablehlo.convert %v4457 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4460 = stablehlo.convolution(%v4458, %v4459)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4461 = stablehlo.convert %v4460 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4462 = stablehlo.reshape %v4461 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4463 = stablehlo.reshape %v4462 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4464 = stablehlo.reshape %v551 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4465 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4466 = stablehlo.compare GT, %v4464, %v4465 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4467 = stablehlo.select %v4466, %v4463, %v4465 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4468 = stablehlo.reshape %v4467 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4469 = stablehlo.reshape %v531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4471 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4472 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4473 = stablehlo.reduce(%v4469 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4474 = stablehlo.broadcast_in_dim %v4473, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4475 = stablehlo.divide %v4474, %v4471 : tensor<64x128x28x28xf32>
    %v4476 = stablehlo.subtract %v4469, %v4475 : tensor<64x128x28x28xf32>
    %v4477 = stablehlo.multiply %v4476, %v4476 : tensor<64x128x28x28xf32>
    %v4478 = stablehlo.reduce(%v4477 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4479 = stablehlo.broadcast_in_dim %v4478, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4480 = stablehlo.divide %v4479, %v4471 : tensor<64x128x28x28xf32>
    %v4481 = stablehlo.add %v4480, %v4472 : tensor<64x128x28x28xf32>
    %v4482 = stablehlo.rsqrt %v4481 : tensor<64x128x28x28xf32>
    %v4483 = stablehlo.multiply %v4476, %v4482 : tensor<64x128x28x28xf32>
    %v4484 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4485 = stablehlo.reshape %v4468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4486 = stablehlo.multiply %v4484, %v4485 : tensor<64x128x28x28xf32>
    %v4487 = stablehlo.reduce(%v4486 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4488 = stablehlo.broadcast_in_dim %v4487, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4489 = stablehlo.multiply %v4483, %v4486 : tensor<64x128x28x28xf32>
    %v4490 = stablehlo.reduce(%v4489 init: %v4470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4491 = stablehlo.broadcast_in_dim %v4490, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4492 = stablehlo.multiply %v4486, %v4471 : tensor<64x128x28x28xf32>
    %v4493 = stablehlo.subtract %v4492, %v4488 : tensor<64x128x28x28xf32>
    %v4494 = stablehlo.multiply %v4483, %v4491 : tensor<64x128x28x28xf32>
    %v4495 = stablehlo.subtract %v4493, %v4494 : tensor<64x128x28x28xf32>
    %v4496 = stablehlo.divide %v4482, %v4471 : tensor<64x128x28x28xf32>
    %v4497 = stablehlo.multiply %v4496, %v4495 : tensor<64x128x28x28xf32>
    %v4498 = stablehlo.reshape %v4497 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4499 = stablehlo.reshape %v4498 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4500 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4501 = stablehlo.transpose %v4500, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4502 = stablehlo.convert %v4499 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4503 = stablehlo.convert %v4501 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4504 = stablehlo.convolution(%v4502, %v4503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v4505 = stablehlo.convert %v4504 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4506 = stablehlo.reshape %v4505 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4507 = stablehlo.reshape %v4506 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4508 = stablehlo.reshape %v519 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4509 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4510 = stablehlo.compare GT, %v4508, %v4509 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4511 = stablehlo.select %v4510, %v4507, %v4509 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4512 = stablehlo.reshape %v4511 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4513 = stablehlo.reshape %v499 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4515 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4516 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4517 = stablehlo.reduce(%v4513 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4518 = stablehlo.broadcast_in_dim %v4517, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4519 = stablehlo.divide %v4518, %v4515 : tensor<64x128x28x28xf32>
    %v4520 = stablehlo.subtract %v4513, %v4519 : tensor<64x128x28x28xf32>
    %v4521 = stablehlo.multiply %v4520, %v4520 : tensor<64x128x28x28xf32>
    %v4522 = stablehlo.reduce(%v4521 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4523 = stablehlo.broadcast_in_dim %v4522, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4524 = stablehlo.divide %v4523, %v4515 : tensor<64x128x28x28xf32>
    %v4525 = stablehlo.add %v4524, %v4516 : tensor<64x128x28x28xf32>
    %v4526 = stablehlo.rsqrt %v4525 : tensor<64x128x28x28xf32>
    %v4527 = stablehlo.multiply %v4520, %v4526 : tensor<64x128x28x28xf32>
    %v4528 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4529 = stablehlo.reshape %v4512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4530 = stablehlo.multiply %v4528, %v4529 : tensor<64x128x28x28xf32>
    %v4531 = stablehlo.reduce(%v4530 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4532 = stablehlo.broadcast_in_dim %v4531, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4533 = stablehlo.multiply %v4527, %v4530 : tensor<64x128x28x28xf32>
    %v4534 = stablehlo.reduce(%v4533 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4535 = stablehlo.broadcast_in_dim %v4534, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4536 = stablehlo.multiply %v4530, %v4515 : tensor<64x128x28x28xf32>
    %v4537 = stablehlo.subtract %v4536, %v4532 : tensor<64x128x28x28xf32>
    %v4538 = stablehlo.multiply %v4527, %v4535 : tensor<64x128x28x28xf32>
    %v4539 = stablehlo.subtract %v4537, %v4538 : tensor<64x128x28x28xf32>
    %v4540 = stablehlo.divide %v4526, %v4515 : tensor<64x128x28x28xf32>
    %v4541 = stablehlo.multiply %v4540, %v4539 : tensor<64x128x28x28xf32>
    %v4542 = stablehlo.reshape %v4541 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4543 = stablehlo.reshape %v4542 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4544 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4545 = stablehlo.transpose %v4544, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4546 = stablehlo.convert %v4543 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4547 = stablehlo.convert %v4545 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v4548 = stablehlo.convolution(%v4546, %v4547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4549 = stablehlo.convert %v4548 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4550 = stablehlo.reshape %v4549 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4551 = stablehlo.reshape %v4550 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4552 = stablehlo.reshape %v4424 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4553 = stablehlo.add %v4551, %v4552 : tensor<64x128x56x56xf32>
    %v4554 = stablehlo.reshape %v4553 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4555 = stablehlo.reshape %v491 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4556 = stablehlo.reshape %v4542 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4557 = stablehlo.transpose %v4555, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4558 = stablehlo.transpose %v4556, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4559 = stablehlo.convert %v4557 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4560 = stablehlo.convert %v4558 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4561 = stablehlo.convolution(%v4559, %v4560)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v4562 = stablehlo.convert %v4561 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v4563 = stablehlo.transpose %v4562, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4564 = stablehlo.reshape %v499 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4566 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4567 = stablehlo.reduce(%v4564 init: %v4565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4568 = stablehlo.broadcast_in_dim %v4567, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4569 = stablehlo.divide %v4568, %v4566 : tensor<64x128x28x28xf32>
    %v4570 = stablehlo.subtract %v4564, %v4569 : tensor<64x128x28x28xf32>
    %v4571 = stablehlo.multiply %v4570, %v4570 : tensor<64x128x28x28xf32>
    %v4572 = stablehlo.reduce(%v4571 init: %v4565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4573 = stablehlo.broadcast_in_dim %v4572, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4574 = stablehlo.divide %v4573, %v4566 : tensor<64x128x28x28xf32>
    %v4575 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4576 = stablehlo.add %v4574, %v4575 : tensor<64x128x28x28xf32>
    %v4577 = stablehlo.rsqrt %v4576 : tensor<64x128x28x28xf32>
    %v4578 = stablehlo.multiply %v4570, %v4577 : tensor<64x128x28x28xf32>
    %v4579 = stablehlo.reshape %v4512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4580 = stablehlo.multiply %v4579, %v4578 : tensor<64x128x28x28xf32>
    %v4581 = stablehlo.reduce(%v4580 init: %v4565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4582 = stablehlo.reshape %v4512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4584 = stablehlo.reduce(%v4582 init: %v4583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4585 = stablehlo.reshape %v523 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4586 = stablehlo.reshape %v4498 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4587 = stablehlo.transpose %v4585, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4588 = stablehlo.transpose %v4586, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4589 = stablehlo.convert %v4587 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4590 = stablehlo.convert %v4588 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4591 = stablehlo.convolution(%v4589, %v4590)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4592 = stablehlo.convert %v4591 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4593 = stablehlo.transpose %v4592, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4594 = stablehlo.reshape %v531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4595 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4596 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4597 = stablehlo.reduce(%v4594 init: %v4595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4598 = stablehlo.broadcast_in_dim %v4597, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4599 = stablehlo.divide %v4598, %v4596 : tensor<64x128x28x28xf32>
    %v4600 = stablehlo.subtract %v4594, %v4599 : tensor<64x128x28x28xf32>
    %v4601 = stablehlo.multiply %v4600, %v4600 : tensor<64x128x28x28xf32>
    %v4602 = stablehlo.reduce(%v4601 init: %v4595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4603 = stablehlo.broadcast_in_dim %v4602, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4604 = stablehlo.divide %v4603, %v4596 : tensor<64x128x28x28xf32>
    %v4605 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4606 = stablehlo.add %v4604, %v4605 : tensor<64x128x28x28xf32>
    %v4607 = stablehlo.rsqrt %v4606 : tensor<64x128x28x28xf32>
    %v4608 = stablehlo.multiply %v4600, %v4607 : tensor<64x128x28x28xf32>
    %v4609 = stablehlo.reshape %v4468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4610 = stablehlo.multiply %v4609, %v4608 : tensor<64x128x28x28xf32>
    %v4611 = stablehlo.reduce(%v4610 init: %v4595) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4612 = stablehlo.reshape %v4468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4614 = stablehlo.reduce(%v4612 init: %v4613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4615 = stablehlo.reshape %v555 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4616 = stablehlo.reshape %v4454 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4617 = stablehlo.transpose %v4615, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4618 = stablehlo.transpose %v4616, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4619 = stablehlo.convert %v4617 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4620 = stablehlo.convert %v4618 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4621 = stablehlo.convolution(%v4619, %v4620)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4622 = stablehlo.convert %v4621 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4623 = stablehlo.transpose %v4622, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4624 = stablehlo.reshape %v563 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4625 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4626 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4627 = stablehlo.reduce(%v4624 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4628 = stablehlo.broadcast_in_dim %v4627, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4629 = stablehlo.divide %v4628, %v4626 : tensor<64x512x28x28xf32>
    %v4630 = stablehlo.subtract %v4624, %v4629 : tensor<64x512x28x28xf32>
    %v4631 = stablehlo.multiply %v4630, %v4630 : tensor<64x512x28x28xf32>
    %v4632 = stablehlo.reduce(%v4631 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4633 = stablehlo.broadcast_in_dim %v4632, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4634 = stablehlo.divide %v4633, %v4626 : tensor<64x512x28x28xf32>
    %v4635 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4636 = stablehlo.add %v4634, %v4635 : tensor<64x512x28x28xf32>
    %v4637 = stablehlo.rsqrt %v4636 : tensor<64x512x28x28xf32>
    %v4638 = stablehlo.multiply %v4630, %v4637 : tensor<64x512x28x28xf32>
    %v4639 = stablehlo.reshape %v4424 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4640 = stablehlo.multiply %v4639, %v4638 : tensor<64x512x28x28xf32>
    %v4641 = stablehlo.reduce(%v4640 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4642 = stablehlo.reshape %v4424 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4644 = stablehlo.reduce(%v4642 init: %v4643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4645 = stablehlo.reshape %v4554 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4646 = stablehlo.reshape %v487 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4647 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v4648 = stablehlo.compare GT, %v4646, %v4647 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v4649 = stablehlo.select %v4648, %v4645, %v4647 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v4650 = stablehlo.reshape %v4649 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4651 = stablehlo.reshape %v435 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4653 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4654 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4655 = stablehlo.reduce(%v4651 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4656 = stablehlo.broadcast_in_dim %v4655, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4657 = stablehlo.divide %v4656, %v4653 : tensor<64x512x28x28xf32>
    %v4658 = stablehlo.subtract %v4651, %v4657 : tensor<64x512x28x28xf32>
    %v4659 = stablehlo.multiply %v4658, %v4658 : tensor<64x512x28x28xf32>
    %v4660 = stablehlo.reduce(%v4659 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4661 = stablehlo.broadcast_in_dim %v4660, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4662 = stablehlo.divide %v4661, %v4653 : tensor<64x512x28x28xf32>
    %v4663 = stablehlo.add %v4662, %v4654 : tensor<64x512x28x28xf32>
    %v4664 = stablehlo.rsqrt %v4663 : tensor<64x512x28x28xf32>
    %v4665 = stablehlo.multiply %v4658, %v4664 : tensor<64x512x28x28xf32>
    %v4666 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4667 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4668 = stablehlo.multiply %v4666, %v4667 : tensor<64x512x28x28xf32>
    %v4669 = stablehlo.reduce(%v4668 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4670 = stablehlo.broadcast_in_dim %v4669, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4671 = stablehlo.multiply %v4665, %v4668 : tensor<64x512x28x28xf32>
    %v4672 = stablehlo.reduce(%v4671 init: %v4652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4673 = stablehlo.broadcast_in_dim %v4672, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4674 = stablehlo.multiply %v4668, %v4653 : tensor<64x512x28x28xf32>
    %v4675 = stablehlo.subtract %v4674, %v4670 : tensor<64x512x28x28xf32>
    %v4676 = stablehlo.multiply %v4665, %v4673 : tensor<64x512x28x28xf32>
    %v4677 = stablehlo.subtract %v4675, %v4676 : tensor<64x512x28x28xf32>
    %v4678 = stablehlo.divide %v4664, %v4653 : tensor<64x512x28x28xf32>
    %v4679 = stablehlo.multiply %v4678, %v4677 : tensor<64x512x28x28xf32>
    %v4680 = stablehlo.reshape %v4679 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4681 = stablehlo.reshape %v4680 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4682 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4683 = stablehlo.transpose %v4682, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4684 = stablehlo.convert %v4681 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4685 = stablehlo.convert %v4683 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4686 = stablehlo.convolution(%v4684, %v4685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4687 = stablehlo.convert %v4686 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4688 = stablehlo.reshape %v4687 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4689 = stablehlo.reshape %v4688 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4690 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4691 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4692 = stablehlo.compare GT, %v4690, %v4691 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4693 = stablehlo.select %v4692, %v4689, %v4691 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4694 = stablehlo.reshape %v4693 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4695 = stablehlo.reshape %v403 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4696 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4697 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4698 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4699 = stablehlo.reduce(%v4695 init: %v4696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4700 = stablehlo.broadcast_in_dim %v4699, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4701 = stablehlo.divide %v4700, %v4697 : tensor<64x128x28x28xf32>
    %v4702 = stablehlo.subtract %v4695, %v4701 : tensor<64x128x28x28xf32>
    %v4703 = stablehlo.multiply %v4702, %v4702 : tensor<64x128x28x28xf32>
    %v4704 = stablehlo.reduce(%v4703 init: %v4696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4705 = stablehlo.broadcast_in_dim %v4704, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4706 = stablehlo.divide %v4705, %v4697 : tensor<64x128x28x28xf32>
    %v4707 = stablehlo.add %v4706, %v4698 : tensor<64x128x28x28xf32>
    %v4708 = stablehlo.rsqrt %v4707 : tensor<64x128x28x28xf32>
    %v4709 = stablehlo.multiply %v4702, %v4708 : tensor<64x128x28x28xf32>
    %v4710 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4711 = stablehlo.reshape %v4694 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4712 = stablehlo.multiply %v4710, %v4711 : tensor<64x128x28x28xf32>
    %v4713 = stablehlo.reduce(%v4712 init: %v4696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4714 = stablehlo.broadcast_in_dim %v4713, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4715 = stablehlo.multiply %v4709, %v4712 : tensor<64x128x28x28xf32>
    %v4716 = stablehlo.reduce(%v4715 init: %v4696) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4717 = stablehlo.broadcast_in_dim %v4716, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4718 = stablehlo.multiply %v4712, %v4697 : tensor<64x128x28x28xf32>
    %v4719 = stablehlo.subtract %v4718, %v4714 : tensor<64x128x28x28xf32>
    %v4720 = stablehlo.multiply %v4709, %v4717 : tensor<64x128x28x28xf32>
    %v4721 = stablehlo.subtract %v4719, %v4720 : tensor<64x128x28x28xf32>
    %v4722 = stablehlo.divide %v4708, %v4697 : tensor<64x128x28x28xf32>
    %v4723 = stablehlo.multiply %v4722, %v4721 : tensor<64x128x28x28xf32>
    %v4724 = stablehlo.reshape %v4723 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4725 = stablehlo.reshape %v4724 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4726 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4727 = stablehlo.pad %v4725, %v4726, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4728 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4729 = stablehlo.transpose %v4728, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4730 = stablehlo.convert %v4727 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4731 = stablehlo.convert %v4729 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4732 = stablehlo.convolution(%v4730, %v4731)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x56x56xbf16>
    %v4733 = stablehlo.convert %v4732 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v4734 = stablehlo.reshape %v4733 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4735 = stablehlo.reshape %v4734 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4736 = stablehlo.reshape %v391 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4737 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v4738 = stablehlo.compare GT, %v4736, %v4737 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v4739 = stablehlo.select %v4738, %v4735, %v4737 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v4740 = stablehlo.reshape %v4739 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4741 = stablehlo.reshape %v371 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4742 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4743 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v4744 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v4745 = stablehlo.reduce(%v4741 init: %v4742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4746 = stablehlo.broadcast_in_dim %v4745, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4747 = stablehlo.divide %v4746, %v4743 : tensor<64x128x56x56xf32>
    %v4748 = stablehlo.subtract %v4741, %v4747 : tensor<64x128x56x56xf32>
    %v4749 = stablehlo.multiply %v4748, %v4748 : tensor<64x128x56x56xf32>
    %v4750 = stablehlo.reduce(%v4749 init: %v4742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4751 = stablehlo.broadcast_in_dim %v4750, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4752 = stablehlo.divide %v4751, %v4743 : tensor<64x128x56x56xf32>
    %v4753 = stablehlo.add %v4752, %v4744 : tensor<64x128x56x56xf32>
    %v4754 = stablehlo.rsqrt %v4753 : tensor<64x128x56x56xf32>
    %v4755 = stablehlo.multiply %v4748, %v4754 : tensor<64x128x56x56xf32>
    %v4756 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4757 = stablehlo.reshape %v4740 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4758 = stablehlo.multiply %v4756, %v4757 : tensor<64x128x56x56xf32>
    %v4759 = stablehlo.reduce(%v4758 init: %v4742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4760 = stablehlo.broadcast_in_dim %v4759, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4761 = stablehlo.multiply %v4755, %v4758 : tensor<64x128x56x56xf32>
    %v4762 = stablehlo.reduce(%v4761 init: %v4742) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4763 = stablehlo.broadcast_in_dim %v4762, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4764 = stablehlo.multiply %v4758, %v4743 : tensor<64x128x56x56xf32>
    %v4765 = stablehlo.subtract %v4764, %v4760 : tensor<64x128x56x56xf32>
    %v4766 = stablehlo.multiply %v4755, %v4763 : tensor<64x128x56x56xf32>
    %v4767 = stablehlo.subtract %v4765, %v4766 : tensor<64x128x56x56xf32>
    %v4768 = stablehlo.divide %v4754, %v4743 : tensor<64x128x56x56xf32>
    %v4769 = stablehlo.multiply %v4768, %v4767 : tensor<64x128x56x56xf32>
    %v4770 = stablehlo.reshape %v4769 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4771 = stablehlo.reshape %v4770 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4772 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v4773 = stablehlo.transpose %v4772, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v4774 = stablehlo.convert %v4771 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4775 = stablehlo.convert %v4773 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v4776 = stablehlo.convolution(%v4774, %v4775)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<256x128x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4777 = stablehlo.convert %v4776 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4778 = stablehlo.reshape %v4777 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4779 = stablehlo.reshape %v463 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4780 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4781 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4782 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4783 = stablehlo.reduce(%v4779 init: %v4780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4784 = stablehlo.broadcast_in_dim %v4783, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4785 = stablehlo.divide %v4784, %v4781 : tensor<64x512x28x28xf32>
    %v4786 = stablehlo.subtract %v4779, %v4785 : tensor<64x512x28x28xf32>
    %v4787 = stablehlo.multiply %v4786, %v4786 : tensor<64x512x28x28xf32>
    %v4788 = stablehlo.reduce(%v4787 init: %v4780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4789 = stablehlo.broadcast_in_dim %v4788, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4790 = stablehlo.divide %v4789, %v4781 : tensor<64x512x28x28xf32>
    %v4791 = stablehlo.add %v4790, %v4782 : tensor<64x512x28x28xf32>
    %v4792 = stablehlo.rsqrt %v4791 : tensor<64x512x28x28xf32>
    %v4793 = stablehlo.multiply %v4786, %v4792 : tensor<64x512x28x28xf32>
    %v4794 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4795 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4796 = stablehlo.multiply %v4794, %v4795 : tensor<64x512x28x28xf32>
    %v4797 = stablehlo.reduce(%v4796 init: %v4780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4798 = stablehlo.broadcast_in_dim %v4797, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4799 = stablehlo.multiply %v4793, %v4796 : tensor<64x512x28x28xf32>
    %v4800 = stablehlo.reduce(%v4799 init: %v4780) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4801 = stablehlo.broadcast_in_dim %v4800, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4802 = stablehlo.multiply %v4796, %v4781 : tensor<64x512x28x28xf32>
    %v4803 = stablehlo.subtract %v4802, %v4798 : tensor<64x512x28x28xf32>
    %v4804 = stablehlo.multiply %v4793, %v4801 : tensor<64x512x28x28xf32>
    %v4805 = stablehlo.subtract %v4803, %v4804 : tensor<64x512x28x28xf32>
    %v4806 = stablehlo.divide %v4792, %v4781 : tensor<64x512x28x28xf32>
    %v4807 = stablehlo.multiply %v4806, %v4805 : tensor<64x512x28x28xf32>
    %v4808 = stablehlo.reshape %v4807 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4809 = stablehlo.reshape %v4808 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4811 = stablehlo.pad %v4809, %v4810, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v4812 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v4813 = stablehlo.transpose %v4812, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4814 = stablehlo.convert %v4811 : (tensor<64x512x56x56xf32>) -> tensor<64x512x56x56xbf16>
    %v4815 = stablehlo.convert %v4813 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v4816 = stablehlo.convolution(%v4814, %v4815)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x56x56xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4817 = stablehlo.convert %v4816 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4818 = stablehlo.reshape %v4817 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4819 = stablehlo.reshape %v4778 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4820 = stablehlo.reshape %v4818 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4821 = stablehlo.add %v4819, %v4820 : tensor<64x64x112x112xf32>
    %v4822 = stablehlo.reshape %v4821 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v4823 = stablehlo.reshape %v363 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4824 = stablehlo.reshape %v4770 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4825 = stablehlo.transpose %v4823, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4826 = stablehlo.transpose %v4824, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4827 = stablehlo.convert %v4825 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4828 = stablehlo.convert %v4826 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4829 = stablehlo.convolution(%v4827, %v4828)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<256x128x1x1xbf16>
    %v4830 = stablehlo.convert %v4829 : (tensor<256x128x1x1xbf16>) -> tensor<256x128x1x1xf32>
    %v4831 = stablehlo.transpose %v4830, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v4832 = stablehlo.reshape %v371 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4834 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v4835 = stablehlo.reduce(%v4832 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4836 = stablehlo.broadcast_in_dim %v4835, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4837 = stablehlo.divide %v4836, %v4834 : tensor<64x128x56x56xf32>
    %v4838 = stablehlo.subtract %v4832, %v4837 : tensor<64x128x56x56xf32>
    %v4839 = stablehlo.multiply %v4838, %v4838 : tensor<64x128x56x56xf32>
    %v4840 = stablehlo.reduce(%v4839 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4841 = stablehlo.broadcast_in_dim %v4840, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4842 = stablehlo.divide %v4841, %v4834 : tensor<64x128x56x56xf32>
    %v4843 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v4844 = stablehlo.add %v4842, %v4843 : tensor<64x128x56x56xf32>
    %v4845 = stablehlo.rsqrt %v4844 : tensor<64x128x56x56xf32>
    %v4846 = stablehlo.multiply %v4838, %v4845 : tensor<64x128x56x56xf32>
    %v4847 = stablehlo.reshape %v4740 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4848 = stablehlo.multiply %v4847, %v4846 : tensor<64x128x56x56xf32>
    %v4849 = stablehlo.reduce(%v4848 init: %v4833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4850 = stablehlo.reshape %v4740 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4851 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4852 = stablehlo.reduce(%v4850 init: %v4851) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4853 = stablehlo.reshape %v395 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4854 = stablehlo.reshape %v4724 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4856 = stablehlo.pad %v4854, %v4855, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4857 = stablehlo.transpose %v4853, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4858 = stablehlo.transpose %v4856, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4859 = stablehlo.convert %v4857 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4860 = stablehlo.convert %v4858 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4861 = stablehlo.convolution(%v4859, %v4860)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<128x128x3x3xbf16>
    %v4862 = stablehlo.convert %v4861 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4863 = stablehlo.transpose %v4862, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4864 = stablehlo.reshape %v403 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4866 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4867 = stablehlo.reduce(%v4864 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4868 = stablehlo.broadcast_in_dim %v4867, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4869 = stablehlo.divide %v4868, %v4866 : tensor<64x128x28x28xf32>
    %v4870 = stablehlo.subtract %v4864, %v4869 : tensor<64x128x28x28xf32>
    %v4871 = stablehlo.multiply %v4870, %v4870 : tensor<64x128x28x28xf32>
    %v4872 = stablehlo.reduce(%v4871 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4873 = stablehlo.broadcast_in_dim %v4872, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4874 = stablehlo.divide %v4873, %v4866 : tensor<64x128x28x28xf32>
    %v4875 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4876 = stablehlo.add %v4874, %v4875 : tensor<64x128x28x28xf32>
    %v4877 = stablehlo.rsqrt %v4876 : tensor<64x128x28x28xf32>
    %v4878 = stablehlo.multiply %v4870, %v4877 : tensor<64x128x28x28xf32>
    %v4879 = stablehlo.reshape %v4694 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4880 = stablehlo.multiply %v4879, %v4878 : tensor<64x128x28x28xf32>
    %v4881 = stablehlo.reduce(%v4880 init: %v4865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4882 = stablehlo.reshape %v4694 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4884 = stablehlo.reduce(%v4882 init: %v4883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4885 = stablehlo.reshape %v427 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4886 = stablehlo.reshape %v4680 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4887 = stablehlo.transpose %v4885, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4888 = stablehlo.transpose %v4886, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4889 = stablehlo.convert %v4887 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4890 = stablehlo.convert %v4888 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4891 = stablehlo.convolution(%v4889, %v4890)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4892 = stablehlo.convert %v4891 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4893 = stablehlo.transpose %v4892, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4894 = stablehlo.reshape %v435 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4896 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4897 = stablehlo.reduce(%v4894 init: %v4895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4898 = stablehlo.broadcast_in_dim %v4897, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4899 = stablehlo.divide %v4898, %v4896 : tensor<64x512x28x28xf32>
    %v4900 = stablehlo.subtract %v4894, %v4899 : tensor<64x512x28x28xf32>
    %v4901 = stablehlo.multiply %v4900, %v4900 : tensor<64x512x28x28xf32>
    %v4902 = stablehlo.reduce(%v4901 init: %v4895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4903 = stablehlo.broadcast_in_dim %v4902, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4904 = stablehlo.divide %v4903, %v4896 : tensor<64x512x28x28xf32>
    %v4905 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4906 = stablehlo.add %v4904, %v4905 : tensor<64x512x28x28xf32>
    %v4907 = stablehlo.rsqrt %v4906 : tensor<64x512x28x28xf32>
    %v4908 = stablehlo.multiply %v4900, %v4907 : tensor<64x512x28x28xf32>
    %v4909 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4910 = stablehlo.multiply %v4909, %v4908 : tensor<64x512x28x28xf32>
    %v4911 = stablehlo.reduce(%v4910 init: %v4895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4912 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4913 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4914 = stablehlo.reduce(%v4912 init: %v4913) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4915 = stablehlo.reshape %v363 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4916 = stablehlo.reshape %v4808 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4918 = stablehlo.pad %v4916, %v4917, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v4919 = stablehlo.transpose %v4915, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4920 = stablehlo.transpose %v4918, dims = [1, 0, 2, 3] : (tensor<64x512x56x56xf32>) -> tensor<512x64x56x56xf32>
    %v4921 = stablehlo.convert %v4919 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4922 = stablehlo.convert %v4920 : (tensor<512x64x56x56xf32>) -> tensor<512x64x56x56xbf16>
    %v4923 = stablehlo.convolution(%v4921, %v4922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<512x64x56x56xbf16>) -> tensor<256x512x1x1xbf16>
    %v4924 = stablehlo.convert %v4923 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v4925 = stablehlo.transpose %v4924, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4926 = stablehlo.reshape %v463 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4928 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4929 = stablehlo.reduce(%v4926 init: %v4927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4930 = stablehlo.broadcast_in_dim %v4929, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4931 = stablehlo.divide %v4930, %v4928 : tensor<64x512x28x28xf32>
    %v4932 = stablehlo.subtract %v4926, %v4931 : tensor<64x512x28x28xf32>
    %v4933 = stablehlo.multiply %v4932, %v4932 : tensor<64x512x28x28xf32>
    %v4934 = stablehlo.reduce(%v4933 init: %v4927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4935 = stablehlo.broadcast_in_dim %v4934, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4936 = stablehlo.divide %v4935, %v4928 : tensor<64x512x28x28xf32>
    %v4937 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4938 = stablehlo.add %v4936, %v4937 : tensor<64x512x28x28xf32>
    %v4939 = stablehlo.rsqrt %v4938 : tensor<64x512x28x28xf32>
    %v4940 = stablehlo.multiply %v4932, %v4939 : tensor<64x512x28x28xf32>
    %v4941 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4942 = stablehlo.multiply %v4941, %v4940 : tensor<64x512x28x28xf32>
    %v4943 = stablehlo.reduce(%v4942 init: %v4927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4944 = stablehlo.reshape %v4650 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4946 = stablehlo.reduce(%v4944 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4947 = stablehlo.reshape %v4822 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4948 = stablehlo.reshape %v359 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4949 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v4950 = stablehlo.compare GT, %v4948, %v4949 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v4951 = stablehlo.select %v4950, %v4947, %v4949 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v4952 = stablehlo.reshape %v4951 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v4953 = stablehlo.reshape %v335 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4955 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v4956 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v4957 = stablehlo.reduce(%v4953 init: %v4954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4958 = stablehlo.broadcast_in_dim %v4957, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4959 = stablehlo.divide %v4958, %v4955 : tensor<64x256x56x56xf32>
    %v4960 = stablehlo.subtract %v4953, %v4959 : tensor<64x256x56x56xf32>
    %v4961 = stablehlo.multiply %v4960, %v4960 : tensor<64x256x56x56xf32>
    %v4962 = stablehlo.reduce(%v4961 init: %v4954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4963 = stablehlo.broadcast_in_dim %v4962, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4964 = stablehlo.divide %v4963, %v4955 : tensor<64x256x56x56xf32>
    %v4965 = stablehlo.add %v4964, %v4956 : tensor<64x256x56x56xf32>
    %v4966 = stablehlo.rsqrt %v4965 : tensor<64x256x56x56xf32>
    %v4967 = stablehlo.multiply %v4960, %v4966 : tensor<64x256x56x56xf32>
    %v4968 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4969 = stablehlo.reshape %v4952 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4970 = stablehlo.multiply %v4968, %v4969 : tensor<64x256x56x56xf32>
    %v4971 = stablehlo.reduce(%v4970 init: %v4954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4972 = stablehlo.broadcast_in_dim %v4971, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4973 = stablehlo.multiply %v4967, %v4970 : tensor<64x256x56x56xf32>
    %v4974 = stablehlo.reduce(%v4973 init: %v4954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4975 = stablehlo.broadcast_in_dim %v4974, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4976 = stablehlo.multiply %v4970, %v4955 : tensor<64x256x56x56xf32>
    %v4977 = stablehlo.subtract %v4976, %v4972 : tensor<64x256x56x56xf32>
    %v4978 = stablehlo.multiply %v4967, %v4975 : tensor<64x256x56x56xf32>
    %v4979 = stablehlo.subtract %v4977, %v4978 : tensor<64x256x56x56xf32>
    %v4980 = stablehlo.divide %v4966, %v4955 : tensor<64x256x56x56xf32>
    %v4981 = stablehlo.multiply %v4980, %v4979 : tensor<64x256x56x56xf32>
    %v4982 = stablehlo.reshape %v4981 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4983 = stablehlo.reshape %v4982 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4984 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4985 = stablehlo.transpose %v4984, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4986 = stablehlo.convert %v4983 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v4987 = stablehlo.convert %v4985 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v4988 = stablehlo.convolution(%v4986, %v4987)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v4989 = stablehlo.convert %v4988 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4990 = stablehlo.reshape %v4989 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4991 = stablehlo.reshape %v4990 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4992 = stablehlo.reshape %v323 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4993 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4994 = stablehlo.compare GT, %v4992, %v4993 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4995 = stablehlo.select %v4994, %v4991, %v4993 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4996 = stablehlo.reshape %v4995 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4997 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4998 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4999 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5000 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5001 = stablehlo.reduce(%v4997 init: %v4998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5002 = stablehlo.broadcast_in_dim %v5001, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5003 = stablehlo.divide %v5002, %v4999 : tensor<64x64x56x56xf32>
    %v5004 = stablehlo.subtract %v4997, %v5003 : tensor<64x64x56x56xf32>
    %v5005 = stablehlo.multiply %v5004, %v5004 : tensor<64x64x56x56xf32>
    %v5006 = stablehlo.reduce(%v5005 init: %v4998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5007 = stablehlo.broadcast_in_dim %v5006, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5008 = stablehlo.divide %v5007, %v4999 : tensor<64x64x56x56xf32>
    %v5009 = stablehlo.add %v5008, %v5000 : tensor<64x64x56x56xf32>
    %v5010 = stablehlo.rsqrt %v5009 : tensor<64x64x56x56xf32>
    %v5011 = stablehlo.multiply %v5004, %v5010 : tensor<64x64x56x56xf32>
    %v5012 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5013 = stablehlo.reshape %v4996 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5014 = stablehlo.multiply %v5012, %v5013 : tensor<64x64x56x56xf32>
    %v5015 = stablehlo.reduce(%v5014 init: %v4998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5016 = stablehlo.broadcast_in_dim %v5015, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5017 = stablehlo.multiply %v5011, %v5014 : tensor<64x64x56x56xf32>
    %v5018 = stablehlo.reduce(%v5017 init: %v4998) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5019 = stablehlo.broadcast_in_dim %v5018, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5020 = stablehlo.multiply %v5014, %v4999 : tensor<64x64x56x56xf32>
    %v5021 = stablehlo.subtract %v5020, %v5016 : tensor<64x64x56x56xf32>
    %v5022 = stablehlo.multiply %v5011, %v5019 : tensor<64x64x56x56xf32>
    %v5023 = stablehlo.subtract %v5021, %v5022 : tensor<64x64x56x56xf32>
    %v5024 = stablehlo.divide %v5010, %v4999 : tensor<64x64x56x56xf32>
    %v5025 = stablehlo.multiply %v5024, %v5023 : tensor<64x64x56x56xf32>
    %v5026 = stablehlo.reshape %v5025 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5027 = stablehlo.reshape %v5026 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5028 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5029 = stablehlo.transpose %v5028, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5030 = stablehlo.convert %v5027 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5031 = stablehlo.convert %v5029 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v5032 = stablehlo.convolution(%v5030, %v5031)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v5033 = stablehlo.convert %v5032 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5034 = stablehlo.reshape %v5033 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5035 = stablehlo.reshape %v5034 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5036 = stablehlo.reshape %v291 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5037 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5038 = stablehlo.compare GT, %v5036, %v5037 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5039 = stablehlo.select %v5038, %v5035, %v5037 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5040 = stablehlo.reshape %v5039 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5041 = stablehlo.reshape %v271 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5043 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5044 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5045 = stablehlo.reduce(%v5041 init: %v5042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5046 = stablehlo.broadcast_in_dim %v5045, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5047 = stablehlo.divide %v5046, %v5043 : tensor<64x64x56x56xf32>
    %v5048 = stablehlo.subtract %v5041, %v5047 : tensor<64x64x56x56xf32>
    %v5049 = stablehlo.multiply %v5048, %v5048 : tensor<64x64x56x56xf32>
    %v5050 = stablehlo.reduce(%v5049 init: %v5042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5051 = stablehlo.broadcast_in_dim %v5050, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5052 = stablehlo.divide %v5051, %v5043 : tensor<64x64x56x56xf32>
    %v5053 = stablehlo.add %v5052, %v5044 : tensor<64x64x56x56xf32>
    %v5054 = stablehlo.rsqrt %v5053 : tensor<64x64x56x56xf32>
    %v5055 = stablehlo.multiply %v5048, %v5054 : tensor<64x64x56x56xf32>
    %v5056 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5057 = stablehlo.reshape %v5040 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5058 = stablehlo.multiply %v5056, %v5057 : tensor<64x64x56x56xf32>
    %v5059 = stablehlo.reduce(%v5058 init: %v5042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5060 = stablehlo.broadcast_in_dim %v5059, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5061 = stablehlo.multiply %v5055, %v5058 : tensor<64x64x56x56xf32>
    %v5062 = stablehlo.reduce(%v5061 init: %v5042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5063 = stablehlo.broadcast_in_dim %v5062, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5064 = stablehlo.multiply %v5058, %v5043 : tensor<64x64x56x56xf32>
    %v5065 = stablehlo.subtract %v5064, %v5060 : tensor<64x64x56x56xf32>
    %v5066 = stablehlo.multiply %v5055, %v5063 : tensor<64x64x56x56xf32>
    %v5067 = stablehlo.subtract %v5065, %v5066 : tensor<64x64x56x56xf32>
    %v5068 = stablehlo.divide %v5054, %v5043 : tensor<64x64x56x56xf32>
    %v5069 = stablehlo.multiply %v5068, %v5067 : tensor<64x64x56x56xf32>
    %v5070 = stablehlo.reshape %v5069 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5071 = stablehlo.reshape %v5070 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5072 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v5073 = stablehlo.transpose %v5072, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5074 = stablehlo.convert %v5071 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5075 = stablehlo.convert %v5073 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v5076 = stablehlo.convolution(%v5074, %v5075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v5077 = stablehlo.convert %v5076 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v5078 = stablehlo.reshape %v5077 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5079 = stablehlo.reshape %v5078 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5080 = stablehlo.reshape %v4952 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5081 = stablehlo.add %v5079, %v5080 : tensor<64x64x112x112xf32>
    %v5082 = stablehlo.reshape %v5081 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5083 = stablehlo.reshape %v263 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5084 = stablehlo.reshape %v5070 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5085 = stablehlo.transpose %v5083, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5086 = stablehlo.transpose %v5084, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5087 = stablehlo.convert %v5085 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5088 = stablehlo.convert %v5086 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5089 = stablehlo.convolution(%v5087, %v5088)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v5090 = stablehlo.convert %v5089 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v5091 = stablehlo.transpose %v5090, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5092 = stablehlo.reshape %v271 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5093 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5094 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5095 = stablehlo.reduce(%v5092 init: %v5093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5096 = stablehlo.broadcast_in_dim %v5095, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5097 = stablehlo.divide %v5096, %v5094 : tensor<64x64x56x56xf32>
    %v5098 = stablehlo.subtract %v5092, %v5097 : tensor<64x64x56x56xf32>
    %v5099 = stablehlo.multiply %v5098, %v5098 : tensor<64x64x56x56xf32>
    %v5100 = stablehlo.reduce(%v5099 init: %v5093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5101 = stablehlo.broadcast_in_dim %v5100, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5102 = stablehlo.divide %v5101, %v5094 : tensor<64x64x56x56xf32>
    %v5103 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5104 = stablehlo.add %v5102, %v5103 : tensor<64x64x56x56xf32>
    %v5105 = stablehlo.rsqrt %v5104 : tensor<64x64x56x56xf32>
    %v5106 = stablehlo.multiply %v5098, %v5105 : tensor<64x64x56x56xf32>
    %v5107 = stablehlo.reshape %v5040 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5108 = stablehlo.multiply %v5107, %v5106 : tensor<64x64x56x56xf32>
    %v5109 = stablehlo.reduce(%v5108 init: %v5093) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5110 = stablehlo.reshape %v5040 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5111 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5112 = stablehlo.reduce(%v5110 init: %v5111) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5113 = stablehlo.reshape %v295 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5114 = stablehlo.reshape %v5026 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5115 = stablehlo.transpose %v5113, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5116 = stablehlo.transpose %v5114, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5117 = stablehlo.convert %v5115 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5118 = stablehlo.convert %v5116 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5119 = stablehlo.convolution(%v5117, %v5118)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v5120 = stablehlo.convert %v5119 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v5121 = stablehlo.transpose %v5120, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5122 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5123 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5124 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5125 = stablehlo.reduce(%v5122 init: %v5123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5126 = stablehlo.broadcast_in_dim %v5125, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5127 = stablehlo.divide %v5126, %v5124 : tensor<64x64x56x56xf32>
    %v5128 = stablehlo.subtract %v5122, %v5127 : tensor<64x64x56x56xf32>
    %v5129 = stablehlo.multiply %v5128, %v5128 : tensor<64x64x56x56xf32>
    %v5130 = stablehlo.reduce(%v5129 init: %v5123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5131 = stablehlo.broadcast_in_dim %v5130, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5132 = stablehlo.divide %v5131, %v5124 : tensor<64x64x56x56xf32>
    %v5133 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5134 = stablehlo.add %v5132, %v5133 : tensor<64x64x56x56xf32>
    %v5135 = stablehlo.rsqrt %v5134 : tensor<64x64x56x56xf32>
    %v5136 = stablehlo.multiply %v5128, %v5135 : tensor<64x64x56x56xf32>
    %v5137 = stablehlo.reshape %v4996 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5138 = stablehlo.multiply %v5137, %v5136 : tensor<64x64x56x56xf32>
    %v5139 = stablehlo.reduce(%v5138 init: %v5123) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5140 = stablehlo.reshape %v4996 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5142 = stablehlo.reduce(%v5140 init: %v5141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5143 = stablehlo.reshape %v327 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5144 = stablehlo.reshape %v4982 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5145 = stablehlo.transpose %v5143, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5146 = stablehlo.transpose %v5144, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5147 = stablehlo.convert %v5145 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5148 = stablehlo.convert %v5146 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5149 = stablehlo.convolution(%v5147, %v5148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5150 = stablehlo.convert %v5149 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5151 = stablehlo.transpose %v5150, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5152 = stablehlo.reshape %v335 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5154 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5155 = stablehlo.reduce(%v5152 init: %v5153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5156 = stablehlo.broadcast_in_dim %v5155, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5157 = stablehlo.divide %v5156, %v5154 : tensor<64x256x56x56xf32>
    %v5158 = stablehlo.subtract %v5152, %v5157 : tensor<64x256x56x56xf32>
    %v5159 = stablehlo.multiply %v5158, %v5158 : tensor<64x256x56x56xf32>
    %v5160 = stablehlo.reduce(%v5159 init: %v5153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5161 = stablehlo.broadcast_in_dim %v5160, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5162 = stablehlo.divide %v5161, %v5154 : tensor<64x256x56x56xf32>
    %v5163 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5164 = stablehlo.add %v5162, %v5163 : tensor<64x256x56x56xf32>
    %v5165 = stablehlo.rsqrt %v5164 : tensor<64x256x56x56xf32>
    %v5166 = stablehlo.multiply %v5158, %v5165 : tensor<64x256x56x56xf32>
    %v5167 = stablehlo.reshape %v4952 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5168 = stablehlo.multiply %v5167, %v5166 : tensor<64x256x56x56xf32>
    %v5169 = stablehlo.reduce(%v5168 init: %v5153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5170 = stablehlo.reshape %v4952 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5172 = stablehlo.reduce(%v5170 init: %v5171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5173 = stablehlo.reshape %v5082 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5174 = stablehlo.reshape %v259 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5175 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v5176 = stablehlo.compare GT, %v5174, %v5175 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v5177 = stablehlo.select %v5176, %v5173, %v5175 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v5178 = stablehlo.reshape %v5177 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5179 = stablehlo.reshape %v235 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5181 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5182 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5183 = stablehlo.reduce(%v5179 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5184 = stablehlo.broadcast_in_dim %v5183, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5185 = stablehlo.divide %v5184, %v5181 : tensor<64x256x56x56xf32>
    %v5186 = stablehlo.subtract %v5179, %v5185 : tensor<64x256x56x56xf32>
    %v5187 = stablehlo.multiply %v5186, %v5186 : tensor<64x256x56x56xf32>
    %v5188 = stablehlo.reduce(%v5187 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5189 = stablehlo.broadcast_in_dim %v5188, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5190 = stablehlo.divide %v5189, %v5181 : tensor<64x256x56x56xf32>
    %v5191 = stablehlo.add %v5190, %v5182 : tensor<64x256x56x56xf32>
    %v5192 = stablehlo.rsqrt %v5191 : tensor<64x256x56x56xf32>
    %v5193 = stablehlo.multiply %v5186, %v5192 : tensor<64x256x56x56xf32>
    %v5194 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5195 = stablehlo.reshape %v5178 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5196 = stablehlo.multiply %v5194, %v5195 : tensor<64x256x56x56xf32>
    %v5197 = stablehlo.reduce(%v5196 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5198 = stablehlo.broadcast_in_dim %v5197, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5199 = stablehlo.multiply %v5193, %v5196 : tensor<64x256x56x56xf32>
    %v5200 = stablehlo.reduce(%v5199 init: %v5180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5201 = stablehlo.broadcast_in_dim %v5200, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5202 = stablehlo.multiply %v5196, %v5181 : tensor<64x256x56x56xf32>
    %v5203 = stablehlo.subtract %v5202, %v5198 : tensor<64x256x56x56xf32>
    %v5204 = stablehlo.multiply %v5193, %v5201 : tensor<64x256x56x56xf32>
    %v5205 = stablehlo.subtract %v5203, %v5204 : tensor<64x256x56x56xf32>
    %v5206 = stablehlo.divide %v5192, %v5181 : tensor<64x256x56x56xf32>
    %v5207 = stablehlo.multiply %v5206, %v5205 : tensor<64x256x56x56xf32>
    %v5208 = stablehlo.reshape %v5207 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5209 = stablehlo.reshape %v5208 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5210 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5211 = stablehlo.transpose %v5210, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5212 = stablehlo.convert %v5209 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v5213 = stablehlo.convert %v5211 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v5214 = stablehlo.convolution(%v5212, %v5213)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5215 = stablehlo.convert %v5214 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5216 = stablehlo.reshape %v5215 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5217 = stablehlo.reshape %v5216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5218 = stablehlo.reshape %v223 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5219 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5220 = stablehlo.compare GT, %v5218, %v5219 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5221 = stablehlo.select %v5220, %v5217, %v5219 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5222 = stablehlo.reshape %v5221 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5223 = stablehlo.reshape %v203 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5224 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5225 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5226 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5227 = stablehlo.reduce(%v5223 init: %v5224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5228 = stablehlo.broadcast_in_dim %v5227, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5229 = stablehlo.divide %v5228, %v5225 : tensor<64x64x56x56xf32>
    %v5230 = stablehlo.subtract %v5223, %v5229 : tensor<64x64x56x56xf32>
    %v5231 = stablehlo.multiply %v5230, %v5230 : tensor<64x64x56x56xf32>
    %v5232 = stablehlo.reduce(%v5231 init: %v5224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5233 = stablehlo.broadcast_in_dim %v5232, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5234 = stablehlo.divide %v5233, %v5225 : tensor<64x64x56x56xf32>
    %v5235 = stablehlo.add %v5234, %v5226 : tensor<64x64x56x56xf32>
    %v5236 = stablehlo.rsqrt %v5235 : tensor<64x64x56x56xf32>
    %v5237 = stablehlo.multiply %v5230, %v5236 : tensor<64x64x56x56xf32>
    %v5238 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5239 = stablehlo.reshape %v5222 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5240 = stablehlo.multiply %v5238, %v5239 : tensor<64x64x56x56xf32>
    %v5241 = stablehlo.reduce(%v5240 init: %v5224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5242 = stablehlo.broadcast_in_dim %v5241, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5243 = stablehlo.multiply %v5237, %v5240 : tensor<64x64x56x56xf32>
    %v5244 = stablehlo.reduce(%v5243 init: %v5224) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5245 = stablehlo.broadcast_in_dim %v5244, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5246 = stablehlo.multiply %v5240, %v5225 : tensor<64x64x56x56xf32>
    %v5247 = stablehlo.subtract %v5246, %v5242 : tensor<64x64x56x56xf32>
    %v5248 = stablehlo.multiply %v5237, %v5245 : tensor<64x64x56x56xf32>
    %v5249 = stablehlo.subtract %v5247, %v5248 : tensor<64x64x56x56xf32>
    %v5250 = stablehlo.divide %v5236, %v5225 : tensor<64x64x56x56xf32>
    %v5251 = stablehlo.multiply %v5250, %v5249 : tensor<64x64x56x56xf32>
    %v5252 = stablehlo.reshape %v5251 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5253 = stablehlo.reshape %v5252 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5254 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5255 = stablehlo.transpose %v5254, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5256 = stablehlo.convert %v5253 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5257 = stablehlo.convert %v5255 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v5258 = stablehlo.convolution(%v5256, %v5257)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v5259 = stablehlo.convert %v5258 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5260 = stablehlo.reshape %v5259 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5261 = stablehlo.reshape %v5260 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5262 = stablehlo.reshape %v191 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5263 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5264 = stablehlo.compare GT, %v5262, %v5263 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5265 = stablehlo.select %v5264, %v5261, %v5263 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5266 = stablehlo.reshape %v5265 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5267 = stablehlo.reshape %v171 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5269 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5270 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5271 = stablehlo.reduce(%v5267 init: %v5268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5272 = stablehlo.broadcast_in_dim %v5271, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5273 = stablehlo.divide %v5272, %v5269 : tensor<64x64x56x56xf32>
    %v5274 = stablehlo.subtract %v5267, %v5273 : tensor<64x64x56x56xf32>
    %v5275 = stablehlo.multiply %v5274, %v5274 : tensor<64x64x56x56xf32>
    %v5276 = stablehlo.reduce(%v5275 init: %v5268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5277 = stablehlo.broadcast_in_dim %v5276, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5278 = stablehlo.divide %v5277, %v5269 : tensor<64x64x56x56xf32>
    %v5279 = stablehlo.add %v5278, %v5270 : tensor<64x64x56x56xf32>
    %v5280 = stablehlo.rsqrt %v5279 : tensor<64x64x56x56xf32>
    %v5281 = stablehlo.multiply %v5274, %v5280 : tensor<64x64x56x56xf32>
    %v5282 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5283 = stablehlo.reshape %v5266 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5284 = stablehlo.multiply %v5282, %v5283 : tensor<64x64x56x56xf32>
    %v5285 = stablehlo.reduce(%v5284 init: %v5268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5286 = stablehlo.broadcast_in_dim %v5285, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5287 = stablehlo.multiply %v5281, %v5284 : tensor<64x64x56x56xf32>
    %v5288 = stablehlo.reduce(%v5287 init: %v5268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5289 = stablehlo.broadcast_in_dim %v5288, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5290 = stablehlo.multiply %v5284, %v5269 : tensor<64x64x56x56xf32>
    %v5291 = stablehlo.subtract %v5290, %v5286 : tensor<64x64x56x56xf32>
    %v5292 = stablehlo.multiply %v5281, %v5289 : tensor<64x64x56x56xf32>
    %v5293 = stablehlo.subtract %v5291, %v5292 : tensor<64x64x56x56xf32>
    %v5294 = stablehlo.divide %v5280, %v5269 : tensor<64x64x56x56xf32>
    %v5295 = stablehlo.multiply %v5294, %v5293 : tensor<64x64x56x56xf32>
    %v5296 = stablehlo.reshape %v5295 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5297 = stablehlo.reshape %v5296 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5298 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v5299 = stablehlo.transpose %v5298, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5300 = stablehlo.convert %v5297 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5301 = stablehlo.convert %v5299 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v5302 = stablehlo.convolution(%v5300, %v5301)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v5303 = stablehlo.convert %v5302 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v5304 = stablehlo.reshape %v5303 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5305 = stablehlo.reshape %v5304 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5306 = stablehlo.reshape %v5178 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5307 = stablehlo.add %v5305, %v5306 : tensor<64x64x112x112xf32>
    %v5308 = stablehlo.reshape %v5307 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5309 = stablehlo.reshape %v163 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5310 = stablehlo.reshape %v5296 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5311 = stablehlo.transpose %v5309, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5312 = stablehlo.transpose %v5310, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5313 = stablehlo.convert %v5311 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5314 = stablehlo.convert %v5312 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5315 = stablehlo.convolution(%v5313, %v5314)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v5316 = stablehlo.convert %v5315 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v5317 = stablehlo.transpose %v5316, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5318 = stablehlo.reshape %v171 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5320 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5321 = stablehlo.reduce(%v5318 init: %v5319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5322 = stablehlo.broadcast_in_dim %v5321, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5323 = stablehlo.divide %v5322, %v5320 : tensor<64x64x56x56xf32>
    %v5324 = stablehlo.subtract %v5318, %v5323 : tensor<64x64x56x56xf32>
    %v5325 = stablehlo.multiply %v5324, %v5324 : tensor<64x64x56x56xf32>
    %v5326 = stablehlo.reduce(%v5325 init: %v5319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5327 = stablehlo.broadcast_in_dim %v5326, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5328 = stablehlo.divide %v5327, %v5320 : tensor<64x64x56x56xf32>
    %v5329 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5330 = stablehlo.add %v5328, %v5329 : tensor<64x64x56x56xf32>
    %v5331 = stablehlo.rsqrt %v5330 : tensor<64x64x56x56xf32>
    %v5332 = stablehlo.multiply %v5324, %v5331 : tensor<64x64x56x56xf32>
    %v5333 = stablehlo.reshape %v5266 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5334 = stablehlo.multiply %v5333, %v5332 : tensor<64x64x56x56xf32>
    %v5335 = stablehlo.reduce(%v5334 init: %v5319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5336 = stablehlo.reshape %v5266 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5338 = stablehlo.reduce(%v5336 init: %v5337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5339 = stablehlo.reshape %v195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5340 = stablehlo.reshape %v5252 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5341 = stablehlo.transpose %v5339, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5342 = stablehlo.transpose %v5340, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5343 = stablehlo.convert %v5341 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5344 = stablehlo.convert %v5342 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5345 = stablehlo.convolution(%v5343, %v5344)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v5346 = stablehlo.convert %v5345 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v5347 = stablehlo.transpose %v5346, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5348 = stablehlo.reshape %v203 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5349 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5350 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5351 = stablehlo.reduce(%v5348 init: %v5349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5352 = stablehlo.broadcast_in_dim %v5351, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5353 = stablehlo.divide %v5352, %v5350 : tensor<64x64x56x56xf32>
    %v5354 = stablehlo.subtract %v5348, %v5353 : tensor<64x64x56x56xf32>
    %v5355 = stablehlo.multiply %v5354, %v5354 : tensor<64x64x56x56xf32>
    %v5356 = stablehlo.reduce(%v5355 init: %v5349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5357 = stablehlo.broadcast_in_dim %v5356, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5358 = stablehlo.divide %v5357, %v5350 : tensor<64x64x56x56xf32>
    %v5359 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5360 = stablehlo.add %v5358, %v5359 : tensor<64x64x56x56xf32>
    %v5361 = stablehlo.rsqrt %v5360 : tensor<64x64x56x56xf32>
    %v5362 = stablehlo.multiply %v5354, %v5361 : tensor<64x64x56x56xf32>
    %v5363 = stablehlo.reshape %v5222 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5364 = stablehlo.multiply %v5363, %v5362 : tensor<64x64x56x56xf32>
    %v5365 = stablehlo.reduce(%v5364 init: %v5349) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5366 = stablehlo.reshape %v5222 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5367 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5368 = stablehlo.reduce(%v5366 init: %v5367) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5369 = stablehlo.reshape %v227 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5370 = stablehlo.reshape %v5208 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5371 = stablehlo.transpose %v5369, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5372 = stablehlo.transpose %v5370, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5373 = stablehlo.convert %v5371 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5374 = stablehlo.convert %v5372 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5375 = stablehlo.convolution(%v5373, %v5374)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5376 = stablehlo.convert %v5375 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5377 = stablehlo.transpose %v5376, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5378 = stablehlo.reshape %v235 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5379 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5380 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5381 = stablehlo.reduce(%v5378 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5382 = stablehlo.broadcast_in_dim %v5381, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5383 = stablehlo.divide %v5382, %v5380 : tensor<64x256x56x56xf32>
    %v5384 = stablehlo.subtract %v5378, %v5383 : tensor<64x256x56x56xf32>
    %v5385 = stablehlo.multiply %v5384, %v5384 : tensor<64x256x56x56xf32>
    %v5386 = stablehlo.reduce(%v5385 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5387 = stablehlo.broadcast_in_dim %v5386, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5388 = stablehlo.divide %v5387, %v5380 : tensor<64x256x56x56xf32>
    %v5389 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5390 = stablehlo.add %v5388, %v5389 : tensor<64x256x56x56xf32>
    %v5391 = stablehlo.rsqrt %v5390 : tensor<64x256x56x56xf32>
    %v5392 = stablehlo.multiply %v5384, %v5391 : tensor<64x256x56x56xf32>
    %v5393 = stablehlo.reshape %v5178 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5394 = stablehlo.multiply %v5393, %v5392 : tensor<64x256x56x56xf32>
    %v5395 = stablehlo.reduce(%v5394 init: %v5379) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5396 = stablehlo.reshape %v5178 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5398 = stablehlo.reduce(%v5396 init: %v5397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5399 = stablehlo.reshape %v5308 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5400 = stablehlo.reshape %v159 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5401 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v5402 = stablehlo.compare GT, %v5400, %v5401 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v5403 = stablehlo.select %v5402, %v5399, %v5401 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v5404 = stablehlo.reshape %v5403 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5405 = stablehlo.reshape %v107 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5407 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5408 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5409 = stablehlo.reduce(%v5405 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5410 = stablehlo.broadcast_in_dim %v5409, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5411 = stablehlo.divide %v5410, %v5407 : tensor<64x256x56x56xf32>
    %v5412 = stablehlo.subtract %v5405, %v5411 : tensor<64x256x56x56xf32>
    %v5413 = stablehlo.multiply %v5412, %v5412 : tensor<64x256x56x56xf32>
    %v5414 = stablehlo.reduce(%v5413 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5415 = stablehlo.broadcast_in_dim %v5414, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5416 = stablehlo.divide %v5415, %v5407 : tensor<64x256x56x56xf32>
    %v5417 = stablehlo.add %v5416, %v5408 : tensor<64x256x56x56xf32>
    %v5418 = stablehlo.rsqrt %v5417 : tensor<64x256x56x56xf32>
    %v5419 = stablehlo.multiply %v5412, %v5418 : tensor<64x256x56x56xf32>
    %v5420 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5421 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5422 = stablehlo.multiply %v5420, %v5421 : tensor<64x256x56x56xf32>
    %v5423 = stablehlo.reduce(%v5422 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5424 = stablehlo.broadcast_in_dim %v5423, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5425 = stablehlo.multiply %v5419, %v5422 : tensor<64x256x56x56xf32>
    %v5426 = stablehlo.reduce(%v5425 init: %v5406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5427 = stablehlo.broadcast_in_dim %v5426, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5428 = stablehlo.multiply %v5422, %v5407 : tensor<64x256x56x56xf32>
    %v5429 = stablehlo.subtract %v5428, %v5424 : tensor<64x256x56x56xf32>
    %v5430 = stablehlo.multiply %v5419, %v5427 : tensor<64x256x56x56xf32>
    %v5431 = stablehlo.subtract %v5429, %v5430 : tensor<64x256x56x56xf32>
    %v5432 = stablehlo.divide %v5418, %v5407 : tensor<64x256x56x56xf32>
    %v5433 = stablehlo.multiply %v5432, %v5431 : tensor<64x256x56x56xf32>
    %v5434 = stablehlo.reshape %v5433 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5435 = stablehlo.reshape %v5434 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5436 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5437 = stablehlo.transpose %v5436, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5438 = stablehlo.convert %v5435 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v5439 = stablehlo.convert %v5437 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v5440 = stablehlo.convolution(%v5438, %v5439)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5441 = stablehlo.convert %v5440 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5442 = stablehlo.reshape %v5441 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5443 = stablehlo.reshape %v5442 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5444 = stablehlo.reshape %v95 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5445 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5446 = stablehlo.compare GT, %v5444, %v5445 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5447 = stablehlo.select %v5446, %v5443, %v5445 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5448 = stablehlo.reshape %v5447 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5449 = stablehlo.reshape %v75 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5451 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5452 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5453 = stablehlo.reduce(%v5449 init: %v5450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5454 = stablehlo.broadcast_in_dim %v5453, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5455 = stablehlo.divide %v5454, %v5451 : tensor<64x64x56x56xf32>
    %v5456 = stablehlo.subtract %v5449, %v5455 : tensor<64x64x56x56xf32>
    %v5457 = stablehlo.multiply %v5456, %v5456 : tensor<64x64x56x56xf32>
    %v5458 = stablehlo.reduce(%v5457 init: %v5450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5459 = stablehlo.broadcast_in_dim %v5458, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5460 = stablehlo.divide %v5459, %v5451 : tensor<64x64x56x56xf32>
    %v5461 = stablehlo.add %v5460, %v5452 : tensor<64x64x56x56xf32>
    %v5462 = stablehlo.rsqrt %v5461 : tensor<64x64x56x56xf32>
    %v5463 = stablehlo.multiply %v5456, %v5462 : tensor<64x64x56x56xf32>
    %v5464 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5465 = stablehlo.reshape %v5448 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5466 = stablehlo.multiply %v5464, %v5465 : tensor<64x64x56x56xf32>
    %v5467 = stablehlo.reduce(%v5466 init: %v5450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5468 = stablehlo.broadcast_in_dim %v5467, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5469 = stablehlo.multiply %v5463, %v5466 : tensor<64x64x56x56xf32>
    %v5470 = stablehlo.reduce(%v5469 init: %v5450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5471 = stablehlo.broadcast_in_dim %v5470, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5472 = stablehlo.multiply %v5466, %v5451 : tensor<64x64x56x56xf32>
    %v5473 = stablehlo.subtract %v5472, %v5468 : tensor<64x64x56x56xf32>
    %v5474 = stablehlo.multiply %v5463, %v5471 : tensor<64x64x56x56xf32>
    %v5475 = stablehlo.subtract %v5473, %v5474 : tensor<64x64x56x56xf32>
    %v5476 = stablehlo.divide %v5462, %v5451 : tensor<64x64x56x56xf32>
    %v5477 = stablehlo.multiply %v5476, %v5475 : tensor<64x64x56x56xf32>
    %v5478 = stablehlo.reshape %v5477 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5479 = stablehlo.reshape %v5478 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5480 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5481 = stablehlo.transpose %v5480, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5482 = stablehlo.convert %v5479 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5483 = stablehlo.convert %v5481 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v5484 = stablehlo.convolution(%v5482, %v5483)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v5485 = stablehlo.convert %v5484 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5486 = stablehlo.reshape %v5485 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5487 = stablehlo.reshape %v5486 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5488 = stablehlo.reshape %v63 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5489 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v5490 = stablehlo.compare GT, %v5488, %v5489 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v5491 = stablehlo.select %v5490, %v5487, %v5489 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v5492 = stablehlo.reshape %v5491 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5493 = stablehlo.reshape %v43 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5494 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5495 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5496 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5497 = stablehlo.reduce(%v5493 init: %v5494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5498 = stablehlo.broadcast_in_dim %v5497, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5499 = stablehlo.divide %v5498, %v5495 : tensor<64x64x56x56xf32>
    %v5500 = stablehlo.subtract %v5493, %v5499 : tensor<64x64x56x56xf32>
    %v5501 = stablehlo.multiply %v5500, %v5500 : tensor<64x64x56x56xf32>
    %v5502 = stablehlo.reduce(%v5501 init: %v5494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5503 = stablehlo.broadcast_in_dim %v5502, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5504 = stablehlo.divide %v5503, %v5495 : tensor<64x64x56x56xf32>
    %v5505 = stablehlo.add %v5504, %v5496 : tensor<64x64x56x56xf32>
    %v5506 = stablehlo.rsqrt %v5505 : tensor<64x64x56x56xf32>
    %v5507 = stablehlo.multiply %v5500, %v5506 : tensor<64x64x56x56xf32>
    %v5508 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5509 = stablehlo.reshape %v5492 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5510 = stablehlo.multiply %v5508, %v5509 : tensor<64x64x56x56xf32>
    %v5511 = stablehlo.reduce(%v5510 init: %v5494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5512 = stablehlo.broadcast_in_dim %v5511, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5513 = stablehlo.multiply %v5507, %v5510 : tensor<64x64x56x56xf32>
    %v5514 = stablehlo.reduce(%v5513 init: %v5494) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5515 = stablehlo.broadcast_in_dim %v5514, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5516 = stablehlo.multiply %v5510, %v5495 : tensor<64x64x56x56xf32>
    %v5517 = stablehlo.subtract %v5516, %v5512 : tensor<64x64x56x56xf32>
    %v5518 = stablehlo.multiply %v5507, %v5515 : tensor<64x64x56x56xf32>
    %v5519 = stablehlo.subtract %v5517, %v5518 : tensor<64x64x56x56xf32>
    %v5520 = stablehlo.divide %v5506, %v5495 : tensor<64x64x56x56xf32>
    %v5521 = stablehlo.multiply %v5520, %v5519 : tensor<64x64x56x56xf32>
    %v5522 = stablehlo.reshape %v5521 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5523 = stablehlo.reshape %v5522 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5524 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v5525 = stablehlo.transpose %v5524, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5526 = stablehlo.convert %v5523 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5527 = stablehlo.convert %v5525 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v5528 = stablehlo.convolution(%v5526, %v5527)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5529 = stablehlo.convert %v5528 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5530 = stablehlo.reshape %v5529 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5531 = stablehlo.reshape %v135 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5533 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5534 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5535 = stablehlo.reduce(%v5531 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5536 = stablehlo.broadcast_in_dim %v5535, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5537 = stablehlo.divide %v5536, %v5533 : tensor<64x256x56x56xf32>
    %v5538 = stablehlo.subtract %v5531, %v5537 : tensor<64x256x56x56xf32>
    %v5539 = stablehlo.multiply %v5538, %v5538 : tensor<64x256x56x56xf32>
    %v5540 = stablehlo.reduce(%v5539 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5541 = stablehlo.broadcast_in_dim %v5540, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5542 = stablehlo.divide %v5541, %v5533 : tensor<64x256x56x56xf32>
    %v5543 = stablehlo.add %v5542, %v5534 : tensor<64x256x56x56xf32>
    %v5544 = stablehlo.rsqrt %v5543 : tensor<64x256x56x56xf32>
    %v5545 = stablehlo.multiply %v5538, %v5544 : tensor<64x256x56x56xf32>
    %v5546 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5547 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5548 = stablehlo.multiply %v5546, %v5547 : tensor<64x256x56x56xf32>
    %v5549 = stablehlo.reduce(%v5548 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5550 = stablehlo.broadcast_in_dim %v5549, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5551 = stablehlo.multiply %v5545, %v5548 : tensor<64x256x56x56xf32>
    %v5552 = stablehlo.reduce(%v5551 init: %v5532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5553 = stablehlo.broadcast_in_dim %v5552, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5554 = stablehlo.multiply %v5548, %v5533 : tensor<64x256x56x56xf32>
    %v5555 = stablehlo.subtract %v5554, %v5550 : tensor<64x256x56x56xf32>
    %v5556 = stablehlo.multiply %v5545, %v5553 : tensor<64x256x56x56xf32>
    %v5557 = stablehlo.subtract %v5555, %v5556 : tensor<64x256x56x56xf32>
    %v5558 = stablehlo.divide %v5544, %v5533 : tensor<64x256x56x56xf32>
    %v5559 = stablehlo.multiply %v5558, %v5557 : tensor<64x256x56x56xf32>
    %v5560 = stablehlo.reshape %v5559 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5561 = stablehlo.reshape %v5560 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5562 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5563 = stablehlo.transpose %v5562, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5564 = stablehlo.convert %v5561 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v5565 = stablehlo.convert %v5563 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v5566 = stablehlo.convolution(%v5564, %v5565)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5567 = stablehlo.convert %v5566 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5568 = stablehlo.reshape %v5567 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5569 = stablehlo.reshape %v5530 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5570 = stablehlo.reshape %v5568 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5571 = stablehlo.add %v5569, %v5570 : tensor<64x64x56x56xf32>
    %v5572 = stablehlo.reshape %v5571 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5573 = stablehlo.reshape %v35 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5574 = stablehlo.reshape %v5522 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5575 = stablehlo.transpose %v5573, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5576 = stablehlo.transpose %v5574, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5577 = stablehlo.convert %v5575 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5578 = stablehlo.convert %v5576 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5579 = stablehlo.convolution(%v5577, %v5578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x1x1xbf16>
    %v5580 = stablehlo.convert %v5579 : (tensor<64x64x1x1xbf16>) -> tensor<64x64x1x1xf32>
    %v5581 = stablehlo.transpose %v5580, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5582 = stablehlo.reshape %v43 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5584 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5585 = stablehlo.reduce(%v5582 init: %v5583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5586 = stablehlo.broadcast_in_dim %v5585, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5587 = stablehlo.divide %v5586, %v5584 : tensor<64x64x56x56xf32>
    %v5588 = stablehlo.subtract %v5582, %v5587 : tensor<64x64x56x56xf32>
    %v5589 = stablehlo.multiply %v5588, %v5588 : tensor<64x64x56x56xf32>
    %v5590 = stablehlo.reduce(%v5589 init: %v5583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5591 = stablehlo.broadcast_in_dim %v5590, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5592 = stablehlo.divide %v5591, %v5584 : tensor<64x64x56x56xf32>
    %v5593 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5594 = stablehlo.add %v5592, %v5593 : tensor<64x64x56x56xf32>
    %v5595 = stablehlo.rsqrt %v5594 : tensor<64x64x56x56xf32>
    %v5596 = stablehlo.multiply %v5588, %v5595 : tensor<64x64x56x56xf32>
    %v5597 = stablehlo.reshape %v5492 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5598 = stablehlo.multiply %v5597, %v5596 : tensor<64x64x56x56xf32>
    %v5599 = stablehlo.reduce(%v5598 init: %v5583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5600 = stablehlo.reshape %v5492 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5602 = stablehlo.reduce(%v5600 init: %v5601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5603 = stablehlo.reshape %v67 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5604 = stablehlo.reshape %v5478 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5605 = stablehlo.transpose %v5603, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5606 = stablehlo.transpose %v5604, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5607 = stablehlo.convert %v5605 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5608 = stablehlo.convert %v5606 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5609 = stablehlo.convolution(%v5607, %v5608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v5610 = stablehlo.convert %v5609 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v5611 = stablehlo.transpose %v5610, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5612 = stablehlo.reshape %v75 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5614 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5615 = stablehlo.reduce(%v5612 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5616 = stablehlo.broadcast_in_dim %v5615, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5617 = stablehlo.divide %v5616, %v5614 : tensor<64x64x56x56xf32>
    %v5618 = stablehlo.subtract %v5612, %v5617 : tensor<64x64x56x56xf32>
    %v5619 = stablehlo.multiply %v5618, %v5618 : tensor<64x64x56x56xf32>
    %v5620 = stablehlo.reduce(%v5619 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5621 = stablehlo.broadcast_in_dim %v5620, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5622 = stablehlo.divide %v5621, %v5614 : tensor<64x64x56x56xf32>
    %v5623 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5624 = stablehlo.add %v5622, %v5623 : tensor<64x64x56x56xf32>
    %v5625 = stablehlo.rsqrt %v5624 : tensor<64x64x56x56xf32>
    %v5626 = stablehlo.multiply %v5618, %v5625 : tensor<64x64x56x56xf32>
    %v5627 = stablehlo.reshape %v5448 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5628 = stablehlo.multiply %v5627, %v5626 : tensor<64x64x56x56xf32>
    %v5629 = stablehlo.reduce(%v5628 init: %v5613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5630 = stablehlo.reshape %v5448 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5632 = stablehlo.reduce(%v5630 init: %v5631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5633 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5634 = stablehlo.reshape %v5434 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5635 = stablehlo.transpose %v5633, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5636 = stablehlo.transpose %v5634, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5637 = stablehlo.convert %v5635 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5638 = stablehlo.convert %v5636 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5639 = stablehlo.convolution(%v5637, %v5638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5640 = stablehlo.convert %v5639 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5641 = stablehlo.transpose %v5640, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5642 = stablehlo.reshape %v107 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5644 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5645 = stablehlo.reduce(%v5642 init: %v5643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5646 = stablehlo.broadcast_in_dim %v5645, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5647 = stablehlo.divide %v5646, %v5644 : tensor<64x256x56x56xf32>
    %v5648 = stablehlo.subtract %v5642, %v5647 : tensor<64x256x56x56xf32>
    %v5649 = stablehlo.multiply %v5648, %v5648 : tensor<64x256x56x56xf32>
    %v5650 = stablehlo.reduce(%v5649 init: %v5643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5651 = stablehlo.broadcast_in_dim %v5650, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5652 = stablehlo.divide %v5651, %v5644 : tensor<64x256x56x56xf32>
    %v5653 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5654 = stablehlo.add %v5652, %v5653 : tensor<64x256x56x56xf32>
    %v5655 = stablehlo.rsqrt %v5654 : tensor<64x256x56x56xf32>
    %v5656 = stablehlo.multiply %v5648, %v5655 : tensor<64x256x56x56xf32>
    %v5657 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5658 = stablehlo.multiply %v5657, %v5656 : tensor<64x256x56x56xf32>
    %v5659 = stablehlo.reduce(%v5658 init: %v5643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5660 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5662 = stablehlo.reduce(%v5660 init: %v5661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5663 = stablehlo.reshape %v35 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5664 = stablehlo.reshape %v5560 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5665 = stablehlo.transpose %v5663, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5666 = stablehlo.transpose %v5664, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5667 = stablehlo.convert %v5665 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5668 = stablehlo.convert %v5666 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5669 = stablehlo.convolution(%v5667, %v5668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5670 = stablehlo.convert %v5669 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5671 = stablehlo.transpose %v5670, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5672 = stablehlo.reshape %v135 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5673 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5674 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5675 = stablehlo.reduce(%v5672 init: %v5673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5676 = stablehlo.broadcast_in_dim %v5675, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5677 = stablehlo.divide %v5676, %v5674 : tensor<64x256x56x56xf32>
    %v5678 = stablehlo.subtract %v5672, %v5677 : tensor<64x256x56x56xf32>
    %v5679 = stablehlo.multiply %v5678, %v5678 : tensor<64x256x56x56xf32>
    %v5680 = stablehlo.reduce(%v5679 init: %v5673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5681 = stablehlo.broadcast_in_dim %v5680, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5682 = stablehlo.divide %v5681, %v5674 : tensor<64x256x56x56xf32>
    %v5683 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5684 = stablehlo.add %v5682, %v5683 : tensor<64x256x56x56xf32>
    %v5685 = stablehlo.rsqrt %v5684 : tensor<64x256x56x56xf32>
    %v5686 = stablehlo.multiply %v5678, %v5685 : tensor<64x256x56x56xf32>
    %v5687 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5688 = stablehlo.multiply %v5687, %v5686 : tensor<64x256x56x56xf32>
    %v5689 = stablehlo.reduce(%v5688 init: %v5673) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5690 = stablehlo.reshape %v5404 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5692 = stablehlo.reduce(%v5690 init: %v5691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5693 = stablehlo.reshape %v31 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5694 = stablehlo.reshape %v5572 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5695 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5696 = "stablehlo.select_and_scatter"(%v5693, %v5694, %v5695) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x112x112xf32>
    %v5697 = stablehlo.reshape %v5696 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5698 = stablehlo.reshape %v5697 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5699 = stablehlo.reshape %v27 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5700 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v5701 = stablehlo.compare GT, %v5699, %v5700 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v5702 = stablehlo.select %v5701, %v5698, %v5700 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v5703 = stablehlo.reshape %v5702 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5704 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5706 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5707 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v5708 = stablehlo.reduce(%v5704 init: %v5705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5709 = stablehlo.broadcast_in_dim %v5708, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5710 = stablehlo.divide %v5709, %v5706 : tensor<64x64x112x112xf32>
    %v5711 = stablehlo.subtract %v5704, %v5710 : tensor<64x64x112x112xf32>
    %v5712 = stablehlo.multiply %v5711, %v5711 : tensor<64x64x112x112xf32>
    %v5713 = stablehlo.reduce(%v5712 init: %v5705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5714 = stablehlo.broadcast_in_dim %v5713, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5715 = stablehlo.divide %v5714, %v5706 : tensor<64x64x112x112xf32>
    %v5716 = stablehlo.add %v5715, %v5707 : tensor<64x64x112x112xf32>
    %v5717 = stablehlo.rsqrt %v5716 : tensor<64x64x112x112xf32>
    %v5718 = stablehlo.multiply %v5711, %v5717 : tensor<64x64x112x112xf32>
    %v5719 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5720 = stablehlo.reshape %v5703 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5721 = stablehlo.multiply %v5719, %v5720 : tensor<64x64x112x112xf32>
    %v5722 = stablehlo.reduce(%v5721 init: %v5705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5723 = stablehlo.broadcast_in_dim %v5722, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5724 = stablehlo.multiply %v5718, %v5721 : tensor<64x64x112x112xf32>
    %v5725 = stablehlo.reduce(%v5724 init: %v5705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5726 = stablehlo.broadcast_in_dim %v5725, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5727 = stablehlo.multiply %v5721, %v5706 : tensor<64x64x112x112xf32>
    %v5728 = stablehlo.subtract %v5727, %v5723 : tensor<64x64x112x112xf32>
    %v5729 = stablehlo.multiply %v5718, %v5726 : tensor<64x64x112x112xf32>
    %v5730 = stablehlo.subtract %v5728, %v5729 : tensor<64x64x112x112xf32>
    %v5731 = stablehlo.divide %v5717, %v5706 : tensor<64x64x112x112xf32>
    %v5732 = stablehlo.multiply %v5731, %v5730 : tensor<64x64x112x112xf32>
    %v5733 = stablehlo.reshape %v5732 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5734 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v5735 = stablehlo.reshape %v5733 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5737 = stablehlo.pad %v5735, %v5736, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x224x224xf32>
    %v5738 = stablehlo.transpose %v5734, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v5739 = stablehlo.transpose %v5737, dims = [1, 0, 2, 3] : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xf32>
    %v5740 = stablehlo.convert %v5738 : (tensor<3x64x224x224xf32>) -> tensor<3x64x224x224xbf16>
    %v5741 = stablehlo.convert %v5739 : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xbf16>
    %v5742 = stablehlo.convolution(%v5740, %v5741)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xbf16>, tensor<64x64x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v5743 = stablehlo.convert %v5742 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v5744 = stablehlo.transpose %v5743, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v5745 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5747 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5748 = stablehlo.reduce(%v5745 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5749 = stablehlo.broadcast_in_dim %v5748, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5750 = stablehlo.divide %v5749, %v5747 : tensor<64x64x112x112xf32>
    %v5751 = stablehlo.subtract %v5745, %v5750 : tensor<64x64x112x112xf32>
    %v5752 = stablehlo.multiply %v5751, %v5751 : tensor<64x64x112x112xf32>
    %v5753 = stablehlo.reduce(%v5752 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5754 = stablehlo.broadcast_in_dim %v5753, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5755 = stablehlo.divide %v5754, %v5747 : tensor<64x64x112x112xf32>
    %v5756 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v5757 = stablehlo.add %v5755, %v5756 : tensor<64x64x112x112xf32>
    %v5758 = stablehlo.rsqrt %v5757 : tensor<64x64x112x112xf32>
    %v5759 = stablehlo.multiply %v5751, %v5758 : tensor<64x64x112x112xf32>
    %v5760 = stablehlo.reshape %v5703 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5761 = stablehlo.multiply %v5760, %v5759 : tensor<64x64x112x112xf32>
    %v5762 = stablehlo.reduce(%v5761 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5763 = stablehlo.reshape %v5703 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5764 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5765 = stablehlo.reduce(%v5763 init: %v5764) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5766 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5768 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5769 = stablehlo.reduce(%v5766 init: %v5767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5770 = stablehlo.divide %v5769, %v5768 : tensor<64xf32>
    %v5771 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5773 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5774 = stablehlo.reduce(%v5771 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5775 = stablehlo.broadcast_in_dim %v5774, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5776 = stablehlo.divide %v5775, %v5773 : tensor<64x64x112x112xf32>
    %v5777 = stablehlo.subtract %v5771, %v5776 : tensor<64x64x112x112xf32>
    %v5778 = stablehlo.multiply %v5777, %v5777 : tensor<64x64x112x112xf32>
    %v5779 = stablehlo.reduce(%v5778 init: %v5772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5780 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5781 = stablehlo.divide %v5779, %v5780 : tensor<64xf32>
    %v5782 = stablehlo.reshape %v43 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5784 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5785 = stablehlo.reduce(%v5782 init: %v5783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5786 = stablehlo.divide %v5785, %v5784 : tensor<64xf32>
    %v5787 = stablehlo.reshape %v43 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5788 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5789 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5790 = stablehlo.reduce(%v5787 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5791 = stablehlo.broadcast_in_dim %v5790, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5792 = stablehlo.divide %v5791, %v5789 : tensor<64x64x56x56xf32>
    %v5793 = stablehlo.subtract %v5787, %v5792 : tensor<64x64x56x56xf32>
    %v5794 = stablehlo.multiply %v5793, %v5793 : tensor<64x64x56x56xf32>
    %v5795 = stablehlo.reduce(%v5794 init: %v5788) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5796 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5797 = stablehlo.divide %v5795, %v5796 : tensor<64xf32>
    %v5798 = stablehlo.reshape %v75 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5800 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5801 = stablehlo.reduce(%v5798 init: %v5799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5802 = stablehlo.divide %v5801, %v5800 : tensor<64xf32>
    %v5803 = stablehlo.reshape %v75 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5804 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5805 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5806 = stablehlo.reduce(%v5803 init: %v5804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5807 = stablehlo.broadcast_in_dim %v5806, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5808 = stablehlo.divide %v5807, %v5805 : tensor<64x64x56x56xf32>
    %v5809 = stablehlo.subtract %v5803, %v5808 : tensor<64x64x56x56xf32>
    %v5810 = stablehlo.multiply %v5809, %v5809 : tensor<64x64x56x56xf32>
    %v5811 = stablehlo.reduce(%v5810 init: %v5804) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5812 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5813 = stablehlo.divide %v5811, %v5812 : tensor<64xf32>
    %v5814 = stablehlo.reshape %v107 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5816 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5817 = stablehlo.reduce(%v5814 init: %v5815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5818 = stablehlo.divide %v5817, %v5816 : tensor<256xf32>
    %v5819 = stablehlo.reshape %v107 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5821 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5822 = stablehlo.reduce(%v5819 init: %v5820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5823 = stablehlo.broadcast_in_dim %v5822, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5824 = stablehlo.divide %v5823, %v5821 : tensor<64x256x56x56xf32>
    %v5825 = stablehlo.subtract %v5819, %v5824 : tensor<64x256x56x56xf32>
    %v5826 = stablehlo.multiply %v5825, %v5825 : tensor<64x256x56x56xf32>
    %v5827 = stablehlo.reduce(%v5826 init: %v5820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5828 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5829 = stablehlo.divide %v5827, %v5828 : tensor<256xf32>
    %v5830 = stablehlo.reshape %v135 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5832 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5833 = stablehlo.reduce(%v5830 init: %v5831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5834 = stablehlo.divide %v5833, %v5832 : tensor<256xf32>
    %v5835 = stablehlo.reshape %v135 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5836 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5837 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5838 = stablehlo.reduce(%v5835 init: %v5836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5839 = stablehlo.broadcast_in_dim %v5838, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5840 = stablehlo.divide %v5839, %v5837 : tensor<64x256x56x56xf32>
    %v5841 = stablehlo.subtract %v5835, %v5840 : tensor<64x256x56x56xf32>
    %v5842 = stablehlo.multiply %v5841, %v5841 : tensor<64x256x56x56xf32>
    %v5843 = stablehlo.reduce(%v5842 init: %v5836) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5844 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5845 = stablehlo.divide %v5843, %v5844 : tensor<256xf32>
    %v5846 = stablehlo.reshape %v171 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5848 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5849 = stablehlo.reduce(%v5846 init: %v5847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5850 = stablehlo.divide %v5849, %v5848 : tensor<64xf32>
    %v5851 = stablehlo.reshape %v171 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5852 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5853 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5854 = stablehlo.reduce(%v5851 init: %v5852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5855 = stablehlo.broadcast_in_dim %v5854, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5856 = stablehlo.divide %v5855, %v5853 : tensor<64x64x56x56xf32>
    %v5857 = stablehlo.subtract %v5851, %v5856 : tensor<64x64x56x56xf32>
    %v5858 = stablehlo.multiply %v5857, %v5857 : tensor<64x64x56x56xf32>
    %v5859 = stablehlo.reduce(%v5858 init: %v5852) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5860 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5861 = stablehlo.divide %v5859, %v5860 : tensor<64xf32>
    %v5862 = stablehlo.reshape %v203 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5864 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5865 = stablehlo.reduce(%v5862 init: %v5863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5866 = stablehlo.divide %v5865, %v5864 : tensor<64xf32>
    %v5867 = stablehlo.reshape %v203 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5869 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5870 = stablehlo.reduce(%v5867 init: %v5868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5871 = stablehlo.broadcast_in_dim %v5870, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5872 = stablehlo.divide %v5871, %v5869 : tensor<64x64x56x56xf32>
    %v5873 = stablehlo.subtract %v5867, %v5872 : tensor<64x64x56x56xf32>
    %v5874 = stablehlo.multiply %v5873, %v5873 : tensor<64x64x56x56xf32>
    %v5875 = stablehlo.reduce(%v5874 init: %v5868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5876 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5877 = stablehlo.divide %v5875, %v5876 : tensor<64xf32>
    %v5878 = stablehlo.reshape %v235 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5880 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5881 = stablehlo.reduce(%v5878 init: %v5879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5882 = stablehlo.divide %v5881, %v5880 : tensor<256xf32>
    %v5883 = stablehlo.reshape %v235 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5884 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5885 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5886 = stablehlo.reduce(%v5883 init: %v5884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5887 = stablehlo.broadcast_in_dim %v5886, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5888 = stablehlo.divide %v5887, %v5885 : tensor<64x256x56x56xf32>
    %v5889 = stablehlo.subtract %v5883, %v5888 : tensor<64x256x56x56xf32>
    %v5890 = stablehlo.multiply %v5889, %v5889 : tensor<64x256x56x56xf32>
    %v5891 = stablehlo.reduce(%v5890 init: %v5884) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5892 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5893 = stablehlo.divide %v5891, %v5892 : tensor<256xf32>
    %v5894 = stablehlo.reshape %v271 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5896 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5897 = stablehlo.reduce(%v5894 init: %v5895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5898 = stablehlo.divide %v5897, %v5896 : tensor<64xf32>
    %v5899 = stablehlo.reshape %v271 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5900 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5901 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5902 = stablehlo.reduce(%v5899 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5903 = stablehlo.broadcast_in_dim %v5902, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5904 = stablehlo.divide %v5903, %v5901 : tensor<64x64x56x56xf32>
    %v5905 = stablehlo.subtract %v5899, %v5904 : tensor<64x64x56x56xf32>
    %v5906 = stablehlo.multiply %v5905, %v5905 : tensor<64x64x56x56xf32>
    %v5907 = stablehlo.reduce(%v5906 init: %v5900) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5908 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5909 = stablehlo.divide %v5907, %v5908 : tensor<64xf32>
    %v5910 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5912 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5913 = stablehlo.reduce(%v5910 init: %v5911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5914 = stablehlo.divide %v5913, %v5912 : tensor<64xf32>
    %v5915 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5917 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5918 = stablehlo.reduce(%v5915 init: %v5916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5919 = stablehlo.broadcast_in_dim %v5918, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5920 = stablehlo.divide %v5919, %v5917 : tensor<64x64x56x56xf32>
    %v5921 = stablehlo.subtract %v5915, %v5920 : tensor<64x64x56x56xf32>
    %v5922 = stablehlo.multiply %v5921, %v5921 : tensor<64x64x56x56xf32>
    %v5923 = stablehlo.reduce(%v5922 init: %v5916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5924 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5925 = stablehlo.divide %v5923, %v5924 : tensor<64xf32>
    %v5926 = stablehlo.reshape %v335 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5928 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5929 = stablehlo.reduce(%v5926 init: %v5927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5930 = stablehlo.divide %v5929, %v5928 : tensor<256xf32>
    %v5931 = stablehlo.reshape %v335 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5933 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5934 = stablehlo.reduce(%v5931 init: %v5932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5935 = stablehlo.broadcast_in_dim %v5934, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5936 = stablehlo.divide %v5935, %v5933 : tensor<64x256x56x56xf32>
    %v5937 = stablehlo.subtract %v5931, %v5936 : tensor<64x256x56x56xf32>
    %v5938 = stablehlo.multiply %v5937, %v5937 : tensor<64x256x56x56xf32>
    %v5939 = stablehlo.reduce(%v5938 init: %v5932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5940 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5941 = stablehlo.divide %v5939, %v5940 : tensor<256xf32>
    %v5942 = stablehlo.reshape %v371 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5944 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5945 = stablehlo.reduce(%v5942 init: %v5943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5946 = stablehlo.divide %v5945, %v5944 : tensor<128xf32>
    %v5947 = stablehlo.reshape %v371 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5948 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5949 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v5950 = stablehlo.reduce(%v5947 init: %v5948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5951 = stablehlo.broadcast_in_dim %v5950, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5952 = stablehlo.divide %v5951, %v5949 : tensor<64x128x56x56xf32>
    %v5953 = stablehlo.subtract %v5947, %v5952 : tensor<64x128x56x56xf32>
    %v5954 = stablehlo.multiply %v5953, %v5953 : tensor<64x128x56x56xf32>
    %v5955 = stablehlo.reduce(%v5954 init: %v5948) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5956 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5957 = stablehlo.divide %v5955, %v5956 : tensor<128xf32>
    %v5958 = stablehlo.reshape %v403 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5960 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5961 = stablehlo.reduce(%v5958 init: %v5959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5962 = stablehlo.divide %v5961, %v5960 : tensor<128xf32>
    %v5963 = stablehlo.reshape %v403 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5966 = stablehlo.reduce(%v5963 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5967 = stablehlo.broadcast_in_dim %v5966, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5968 = stablehlo.divide %v5967, %v5965 : tensor<64x128x28x28xf32>
    %v5969 = stablehlo.subtract %v5963, %v5968 : tensor<64x128x28x28xf32>
    %v5970 = stablehlo.multiply %v5969, %v5969 : tensor<64x128x28x28xf32>
    %v5971 = stablehlo.reduce(%v5970 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5972 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5973 = stablehlo.divide %v5971, %v5972 : tensor<128xf32>
    %v5974 = stablehlo.reshape %v435 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5976 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5977 = stablehlo.reduce(%v5974 init: %v5975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5978 = stablehlo.divide %v5977, %v5976 : tensor<512xf32>
    %v5979 = stablehlo.reshape %v435 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5980 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5981 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5982 = stablehlo.reduce(%v5979 init: %v5980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5983 = stablehlo.broadcast_in_dim %v5982, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5984 = stablehlo.divide %v5983, %v5981 : tensor<64x512x28x28xf32>
    %v5985 = stablehlo.subtract %v5979, %v5984 : tensor<64x512x28x28xf32>
    %v5986 = stablehlo.multiply %v5985, %v5985 : tensor<64x512x28x28xf32>
    %v5987 = stablehlo.reduce(%v5986 init: %v5980) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5988 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5989 = stablehlo.divide %v5987, %v5988 : tensor<512xf32>
    %v5990 = stablehlo.reshape %v463 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5992 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5993 = stablehlo.reduce(%v5990 init: %v5991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5994 = stablehlo.divide %v5993, %v5992 : tensor<512xf32>
    %v5995 = stablehlo.reshape %v463 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5996 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5997 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5998 = stablehlo.reduce(%v5995 init: %v5996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5999 = stablehlo.broadcast_in_dim %v5998, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6000 = stablehlo.divide %v5999, %v5997 : tensor<64x512x28x28xf32>
    %v6001 = stablehlo.subtract %v5995, %v6000 : tensor<64x512x28x28xf32>
    %v6002 = stablehlo.multiply %v6001, %v6001 : tensor<64x512x28x28xf32>
    %v6003 = stablehlo.reduce(%v6002 init: %v5996) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6004 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6005 = stablehlo.divide %v6003, %v6004 : tensor<512xf32>
    %v6006 = stablehlo.reshape %v499 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6008 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6009 = stablehlo.reduce(%v6006 init: %v6007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6010 = stablehlo.divide %v6009, %v6008 : tensor<128xf32>
    %v6011 = stablehlo.reshape %v499 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6013 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6014 = stablehlo.reduce(%v6011 init: %v6012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6015 = stablehlo.broadcast_in_dim %v6014, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6016 = stablehlo.divide %v6015, %v6013 : tensor<64x128x28x28xf32>
    %v6017 = stablehlo.subtract %v6011, %v6016 : tensor<64x128x28x28xf32>
    %v6018 = stablehlo.multiply %v6017, %v6017 : tensor<64x128x28x28xf32>
    %v6019 = stablehlo.reduce(%v6018 init: %v6012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6020 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6021 = stablehlo.divide %v6019, %v6020 : tensor<128xf32>
    %v6022 = stablehlo.reshape %v531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6024 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6025 = stablehlo.reduce(%v6022 init: %v6023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6026 = stablehlo.divide %v6025, %v6024 : tensor<128xf32>
    %v6027 = stablehlo.reshape %v531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6029 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6030 = stablehlo.reduce(%v6027 init: %v6028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6031 = stablehlo.broadcast_in_dim %v6030, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6032 = stablehlo.divide %v6031, %v6029 : tensor<64x128x28x28xf32>
    %v6033 = stablehlo.subtract %v6027, %v6032 : tensor<64x128x28x28xf32>
    %v6034 = stablehlo.multiply %v6033, %v6033 : tensor<64x128x28x28xf32>
    %v6035 = stablehlo.reduce(%v6034 init: %v6028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6036 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6037 = stablehlo.divide %v6035, %v6036 : tensor<128xf32>
    %v6038 = stablehlo.reshape %v563 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6040 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6041 = stablehlo.reduce(%v6038 init: %v6039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6042 = stablehlo.divide %v6041, %v6040 : tensor<512xf32>
    %v6043 = stablehlo.reshape %v563 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6044 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6045 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v6046 = stablehlo.reduce(%v6043 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6047 = stablehlo.broadcast_in_dim %v6046, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6048 = stablehlo.divide %v6047, %v6045 : tensor<64x512x28x28xf32>
    %v6049 = stablehlo.subtract %v6043, %v6048 : tensor<64x512x28x28xf32>
    %v6050 = stablehlo.multiply %v6049, %v6049 : tensor<64x512x28x28xf32>
    %v6051 = stablehlo.reduce(%v6050 init: %v6044) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6052 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6053 = stablehlo.divide %v6051, %v6052 : tensor<512xf32>
    %v6054 = stablehlo.reshape %v599 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6056 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6057 = stablehlo.reduce(%v6054 init: %v6055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6058 = stablehlo.divide %v6057, %v6056 : tensor<128xf32>
    %v6059 = stablehlo.reshape %v599 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6060 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6061 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6062 = stablehlo.reduce(%v6059 init: %v6060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6063 = stablehlo.broadcast_in_dim %v6062, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6064 = stablehlo.divide %v6063, %v6061 : tensor<64x128x28x28xf32>
    %v6065 = stablehlo.subtract %v6059, %v6064 : tensor<64x128x28x28xf32>
    %v6066 = stablehlo.multiply %v6065, %v6065 : tensor<64x128x28x28xf32>
    %v6067 = stablehlo.reduce(%v6066 init: %v6060) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6068 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6069 = stablehlo.divide %v6067, %v6068 : tensor<128xf32>
    %v6070 = stablehlo.reshape %v631 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6072 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6073 = stablehlo.reduce(%v6070 init: %v6071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6074 = stablehlo.divide %v6073, %v6072 : tensor<128xf32>
    %v6075 = stablehlo.reshape %v631 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6077 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6078 = stablehlo.reduce(%v6075 init: %v6076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6079 = stablehlo.broadcast_in_dim %v6078, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6080 = stablehlo.divide %v6079, %v6077 : tensor<64x128x28x28xf32>
    %v6081 = stablehlo.subtract %v6075, %v6080 : tensor<64x128x28x28xf32>
    %v6082 = stablehlo.multiply %v6081, %v6081 : tensor<64x128x28x28xf32>
    %v6083 = stablehlo.reduce(%v6082 init: %v6076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6084 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6085 = stablehlo.divide %v6083, %v6084 : tensor<128xf32>
    %v6086 = stablehlo.reshape %v663 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6088 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6089 = stablehlo.reduce(%v6086 init: %v6087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6090 = stablehlo.divide %v6089, %v6088 : tensor<512xf32>
    %v6091 = stablehlo.reshape %v663 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6093 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v6094 = stablehlo.reduce(%v6091 init: %v6092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6095 = stablehlo.broadcast_in_dim %v6094, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6096 = stablehlo.divide %v6095, %v6093 : tensor<64x512x28x28xf32>
    %v6097 = stablehlo.subtract %v6091, %v6096 : tensor<64x512x28x28xf32>
    %v6098 = stablehlo.multiply %v6097, %v6097 : tensor<64x512x28x28xf32>
    %v6099 = stablehlo.reduce(%v6098 init: %v6092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6100 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6101 = stablehlo.divide %v6099, %v6100 : tensor<512xf32>
    %v6102 = stablehlo.reshape %v699 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6104 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6105 = stablehlo.reduce(%v6102 init: %v6103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6106 = stablehlo.divide %v6105, %v6104 : tensor<128xf32>
    %v6107 = stablehlo.reshape %v699 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6109 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6110 = stablehlo.reduce(%v6107 init: %v6108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6111 = stablehlo.broadcast_in_dim %v6110, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6112 = stablehlo.divide %v6111, %v6109 : tensor<64x128x28x28xf32>
    %v6113 = stablehlo.subtract %v6107, %v6112 : tensor<64x128x28x28xf32>
    %v6114 = stablehlo.multiply %v6113, %v6113 : tensor<64x128x28x28xf32>
    %v6115 = stablehlo.reduce(%v6114 init: %v6108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6116 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6117 = stablehlo.divide %v6115, %v6116 : tensor<128xf32>
    %v6118 = stablehlo.reshape %v731 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6120 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6121 = stablehlo.reduce(%v6118 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6122 = stablehlo.divide %v6121, %v6120 : tensor<128xf32>
    %v6123 = stablehlo.reshape %v731 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v6124 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6125 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v6126 = stablehlo.reduce(%v6123 init: %v6124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6127 = stablehlo.broadcast_in_dim %v6126, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v6128 = stablehlo.divide %v6127, %v6125 : tensor<64x128x28x28xf32>
    %v6129 = stablehlo.subtract %v6123, %v6128 : tensor<64x128x28x28xf32>
    %v6130 = stablehlo.multiply %v6129, %v6129 : tensor<64x128x28x28xf32>
    %v6131 = stablehlo.reduce(%v6130 init: %v6124) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v6132 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v6133 = stablehlo.divide %v6131, %v6132 : tensor<128xf32>
    %v6134 = stablehlo.reshape %v763 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6136 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6137 = stablehlo.reduce(%v6134 init: %v6135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6138 = stablehlo.divide %v6137, %v6136 : tensor<512xf32>
    %v6139 = stablehlo.reshape %v763 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6140 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6141 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v6142 = stablehlo.reduce(%v6139 init: %v6140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6143 = stablehlo.broadcast_in_dim %v6142, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6144 = stablehlo.divide %v6143, %v6141 : tensor<64x512x28x28xf32>
    %v6145 = stablehlo.subtract %v6139, %v6144 : tensor<64x512x28x28xf32>
    %v6146 = stablehlo.multiply %v6145, %v6145 : tensor<64x512x28x28xf32>
    %v6147 = stablehlo.reduce(%v6146 init: %v6140) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6148 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v6149 = stablehlo.divide %v6147, %v6148 : tensor<512xf32>
    %v6150 = stablehlo.reshape %v799 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v6151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6152 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v6153 = stablehlo.reduce(%v6150 init: %v6151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v6154 = stablehlo.divide %v6153, %v6152 : tensor<256xf32>
    %v6155 = stablehlo.reshape %v799 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v6156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6157 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v6158 = stablehlo.reduce(%v6155 init: %v6156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v6159 = stablehlo.broadcast_in_dim %v6158, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v6160 = stablehlo.divide %v6159, %v6157 : tensor<64x256x28x28xf32>
    %v6161 = stablehlo.subtract %v6155, %v6160 : tensor<64x256x28x28xf32>
    %v6162 = stablehlo.multiply %v6161, %v6161 : tensor<64x256x28x28xf32>
    %v6163 = stablehlo.reduce(%v6162 init: %v6156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v6164 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v6165 = stablehlo.divide %v6163, %v6164 : tensor<256xf32>
    %v6166 = stablehlo.reshape %v831 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6168 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6169 = stablehlo.reduce(%v6166 init: %v6167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6170 = stablehlo.divide %v6169, %v6168 : tensor<256xf32>
    %v6171 = stablehlo.reshape %v831 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6173 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6174 = stablehlo.reduce(%v6171 init: %v6172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6175 = stablehlo.broadcast_in_dim %v6174, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6176 = stablehlo.divide %v6175, %v6173 : tensor<64x256x14x14xf32>
    %v6177 = stablehlo.subtract %v6171, %v6176 : tensor<64x256x14x14xf32>
    %v6178 = stablehlo.multiply %v6177, %v6177 : tensor<64x256x14x14xf32>
    %v6179 = stablehlo.reduce(%v6178 init: %v6172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6180 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6181 = stablehlo.divide %v6179, %v6180 : tensor<256xf32>
    %v6182 = stablehlo.reshape %v863 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6184 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6185 = stablehlo.reduce(%v6182 init: %v6183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6186 = stablehlo.divide %v6185, %v6184 : tensor<1024xf32>
    %v6187 = stablehlo.reshape %v863 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6189 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6190 = stablehlo.reduce(%v6187 init: %v6188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6191 = stablehlo.broadcast_in_dim %v6190, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6192 = stablehlo.divide %v6191, %v6189 : tensor<64x1024x14x14xf32>
    %v6193 = stablehlo.subtract %v6187, %v6192 : tensor<64x1024x14x14xf32>
    %v6194 = stablehlo.multiply %v6193, %v6193 : tensor<64x1024x14x14xf32>
    %v6195 = stablehlo.reduce(%v6194 init: %v6188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6196 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6197 = stablehlo.divide %v6195, %v6196 : tensor<1024xf32>
    %v6198 = stablehlo.reshape %v891 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6200 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6201 = stablehlo.reduce(%v6198 init: %v6199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6202 = stablehlo.divide %v6201, %v6200 : tensor<1024xf32>
    %v6203 = stablehlo.reshape %v891 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6205 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6206 = stablehlo.reduce(%v6203 init: %v6204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6207 = stablehlo.broadcast_in_dim %v6206, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6208 = stablehlo.divide %v6207, %v6205 : tensor<64x1024x14x14xf32>
    %v6209 = stablehlo.subtract %v6203, %v6208 : tensor<64x1024x14x14xf32>
    %v6210 = stablehlo.multiply %v6209, %v6209 : tensor<64x1024x14x14xf32>
    %v6211 = stablehlo.reduce(%v6210 init: %v6204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6212 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6213 = stablehlo.divide %v6211, %v6212 : tensor<1024xf32>
    %v6214 = stablehlo.reshape %v927 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6216 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6217 = stablehlo.reduce(%v6214 init: %v6215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6218 = stablehlo.divide %v6217, %v6216 : tensor<256xf32>
    %v6219 = stablehlo.reshape %v927 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6221 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6222 = stablehlo.reduce(%v6219 init: %v6220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6223 = stablehlo.broadcast_in_dim %v6222, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6224 = stablehlo.divide %v6223, %v6221 : tensor<64x256x14x14xf32>
    %v6225 = stablehlo.subtract %v6219, %v6224 : tensor<64x256x14x14xf32>
    %v6226 = stablehlo.multiply %v6225, %v6225 : tensor<64x256x14x14xf32>
    %v6227 = stablehlo.reduce(%v6226 init: %v6220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6228 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6229 = stablehlo.divide %v6227, %v6228 : tensor<256xf32>
    %v6230 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6232 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6233 = stablehlo.reduce(%v6230 init: %v6231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6234 = stablehlo.divide %v6233, %v6232 : tensor<256xf32>
    %v6235 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6236 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6237 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6238 = stablehlo.reduce(%v6235 init: %v6236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6239 = stablehlo.broadcast_in_dim %v6238, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6240 = stablehlo.divide %v6239, %v6237 : tensor<64x256x14x14xf32>
    %v6241 = stablehlo.subtract %v6235, %v6240 : tensor<64x256x14x14xf32>
    %v6242 = stablehlo.multiply %v6241, %v6241 : tensor<64x256x14x14xf32>
    %v6243 = stablehlo.reduce(%v6242 init: %v6236) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6244 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6245 = stablehlo.divide %v6243, %v6244 : tensor<256xf32>
    %v6246 = stablehlo.reshape %v991 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6248 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6249 = stablehlo.reduce(%v6246 init: %v6247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6250 = stablehlo.divide %v6249, %v6248 : tensor<1024xf32>
    %v6251 = stablehlo.reshape %v991 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6252 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6253 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6254 = stablehlo.reduce(%v6251 init: %v6252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6255 = stablehlo.broadcast_in_dim %v6254, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6256 = stablehlo.divide %v6255, %v6253 : tensor<64x1024x14x14xf32>
    %v6257 = stablehlo.subtract %v6251, %v6256 : tensor<64x1024x14x14xf32>
    %v6258 = stablehlo.multiply %v6257, %v6257 : tensor<64x1024x14x14xf32>
    %v6259 = stablehlo.reduce(%v6258 init: %v6252) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6260 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6261 = stablehlo.divide %v6259, %v6260 : tensor<1024xf32>
    %v6262 = stablehlo.reshape %v1027 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6264 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6265 = stablehlo.reduce(%v6262 init: %v6263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6266 = stablehlo.divide %v6265, %v6264 : tensor<256xf32>
    %v6267 = stablehlo.reshape %v1027 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6269 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6270 = stablehlo.reduce(%v6267 init: %v6268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6271 = stablehlo.broadcast_in_dim %v6270, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6272 = stablehlo.divide %v6271, %v6269 : tensor<64x256x14x14xf32>
    %v6273 = stablehlo.subtract %v6267, %v6272 : tensor<64x256x14x14xf32>
    %v6274 = stablehlo.multiply %v6273, %v6273 : tensor<64x256x14x14xf32>
    %v6275 = stablehlo.reduce(%v6274 init: %v6268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6276 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6277 = stablehlo.divide %v6275, %v6276 : tensor<256xf32>
    %v6278 = stablehlo.reshape %v1059 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6280 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6281 = stablehlo.reduce(%v6278 init: %v6279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6282 = stablehlo.divide %v6281, %v6280 : tensor<256xf32>
    %v6283 = stablehlo.reshape %v1059 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6285 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6286 = stablehlo.reduce(%v6283 init: %v6284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6287 = stablehlo.broadcast_in_dim %v6286, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6288 = stablehlo.divide %v6287, %v6285 : tensor<64x256x14x14xf32>
    %v6289 = stablehlo.subtract %v6283, %v6288 : tensor<64x256x14x14xf32>
    %v6290 = stablehlo.multiply %v6289, %v6289 : tensor<64x256x14x14xf32>
    %v6291 = stablehlo.reduce(%v6290 init: %v6284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6292 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6293 = stablehlo.divide %v6291, %v6292 : tensor<256xf32>
    %v6294 = stablehlo.reshape %v1091 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6296 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6297 = stablehlo.reduce(%v6294 init: %v6295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6298 = stablehlo.divide %v6297, %v6296 : tensor<1024xf32>
    %v6299 = stablehlo.reshape %v1091 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6300 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6301 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6302 = stablehlo.reduce(%v6299 init: %v6300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6303 = stablehlo.broadcast_in_dim %v6302, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6304 = stablehlo.divide %v6303, %v6301 : tensor<64x1024x14x14xf32>
    %v6305 = stablehlo.subtract %v6299, %v6304 : tensor<64x1024x14x14xf32>
    %v6306 = stablehlo.multiply %v6305, %v6305 : tensor<64x1024x14x14xf32>
    %v6307 = stablehlo.reduce(%v6306 init: %v6300) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6308 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6309 = stablehlo.divide %v6307, %v6308 : tensor<1024xf32>
    %v6310 = stablehlo.reshape %v1127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6311 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6312 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6313 = stablehlo.reduce(%v6310 init: %v6311) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6314 = stablehlo.divide %v6313, %v6312 : tensor<256xf32>
    %v6315 = stablehlo.reshape %v1127 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6317 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6318 = stablehlo.reduce(%v6315 init: %v6316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6319 = stablehlo.broadcast_in_dim %v6318, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6320 = stablehlo.divide %v6319, %v6317 : tensor<64x256x14x14xf32>
    %v6321 = stablehlo.subtract %v6315, %v6320 : tensor<64x256x14x14xf32>
    %v6322 = stablehlo.multiply %v6321, %v6321 : tensor<64x256x14x14xf32>
    %v6323 = stablehlo.reduce(%v6322 init: %v6316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6324 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6325 = stablehlo.divide %v6323, %v6324 : tensor<256xf32>
    %v6326 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6328 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6329 = stablehlo.reduce(%v6326 init: %v6327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6330 = stablehlo.divide %v6329, %v6328 : tensor<256xf32>
    %v6331 = stablehlo.reshape %v1159 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6332 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6333 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6334 = stablehlo.reduce(%v6331 init: %v6332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6335 = stablehlo.broadcast_in_dim %v6334, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6336 = stablehlo.divide %v6335, %v6333 : tensor<64x256x14x14xf32>
    %v6337 = stablehlo.subtract %v6331, %v6336 : tensor<64x256x14x14xf32>
    %v6338 = stablehlo.multiply %v6337, %v6337 : tensor<64x256x14x14xf32>
    %v6339 = stablehlo.reduce(%v6338 init: %v6332) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6340 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6341 = stablehlo.divide %v6339, %v6340 : tensor<256xf32>
    %v6342 = stablehlo.reshape %v1191 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6344 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6345 = stablehlo.reduce(%v6342 init: %v6343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6346 = stablehlo.divide %v6345, %v6344 : tensor<1024xf32>
    %v6347 = stablehlo.reshape %v1191 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6348 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6349 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6350 = stablehlo.reduce(%v6347 init: %v6348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6351 = stablehlo.broadcast_in_dim %v6350, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6352 = stablehlo.divide %v6351, %v6349 : tensor<64x1024x14x14xf32>
    %v6353 = stablehlo.subtract %v6347, %v6352 : tensor<64x1024x14x14xf32>
    %v6354 = stablehlo.multiply %v6353, %v6353 : tensor<64x1024x14x14xf32>
    %v6355 = stablehlo.reduce(%v6354 init: %v6348) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6356 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6357 = stablehlo.divide %v6355, %v6356 : tensor<1024xf32>
    %v6358 = stablehlo.reshape %v1227 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6360 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6361 = stablehlo.reduce(%v6358 init: %v6359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6362 = stablehlo.divide %v6361, %v6360 : tensor<256xf32>
    %v6363 = stablehlo.reshape %v1227 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6365 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6366 = stablehlo.reduce(%v6363 init: %v6364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6367 = stablehlo.broadcast_in_dim %v6366, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6368 = stablehlo.divide %v6367, %v6365 : tensor<64x256x14x14xf32>
    %v6369 = stablehlo.subtract %v6363, %v6368 : tensor<64x256x14x14xf32>
    %v6370 = stablehlo.multiply %v6369, %v6369 : tensor<64x256x14x14xf32>
    %v6371 = stablehlo.reduce(%v6370 init: %v6364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6372 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6373 = stablehlo.divide %v6371, %v6372 : tensor<256xf32>
    %v6374 = stablehlo.reshape %v1259 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6376 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6377 = stablehlo.reduce(%v6374 init: %v6375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6378 = stablehlo.divide %v6377, %v6376 : tensor<256xf32>
    %v6379 = stablehlo.reshape %v1259 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6380 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6381 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6382 = stablehlo.reduce(%v6379 init: %v6380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6383 = stablehlo.broadcast_in_dim %v6382, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6384 = stablehlo.divide %v6383, %v6381 : tensor<64x256x14x14xf32>
    %v6385 = stablehlo.subtract %v6379, %v6384 : tensor<64x256x14x14xf32>
    %v6386 = stablehlo.multiply %v6385, %v6385 : tensor<64x256x14x14xf32>
    %v6387 = stablehlo.reduce(%v6386 init: %v6380) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6388 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6389 = stablehlo.divide %v6387, %v6388 : tensor<256xf32>
    %v6390 = stablehlo.reshape %v1291 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6392 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6393 = stablehlo.reduce(%v6390 init: %v6391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6394 = stablehlo.divide %v6393, %v6392 : tensor<1024xf32>
    %v6395 = stablehlo.reshape %v1291 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6396 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6397 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6398 = stablehlo.reduce(%v6395 init: %v6396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6399 = stablehlo.broadcast_in_dim %v6398, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6400 = stablehlo.divide %v6399, %v6397 : tensor<64x1024x14x14xf32>
    %v6401 = stablehlo.subtract %v6395, %v6400 : tensor<64x1024x14x14xf32>
    %v6402 = stablehlo.multiply %v6401, %v6401 : tensor<64x1024x14x14xf32>
    %v6403 = stablehlo.reduce(%v6402 init: %v6396) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6404 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6405 = stablehlo.divide %v6403, %v6404 : tensor<1024xf32>
    %v6406 = stablehlo.reshape %v1327 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6407 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6408 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6409 = stablehlo.reduce(%v6406 init: %v6407) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6410 = stablehlo.divide %v6409, %v6408 : tensor<256xf32>
    %v6411 = stablehlo.reshape %v1327 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6412 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6413 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6414 = stablehlo.reduce(%v6411 init: %v6412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6415 = stablehlo.broadcast_in_dim %v6414, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6416 = stablehlo.divide %v6415, %v6413 : tensor<64x256x14x14xf32>
    %v6417 = stablehlo.subtract %v6411, %v6416 : tensor<64x256x14x14xf32>
    %v6418 = stablehlo.multiply %v6417, %v6417 : tensor<64x256x14x14xf32>
    %v6419 = stablehlo.reduce(%v6418 init: %v6412) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6420 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6421 = stablehlo.divide %v6419, %v6420 : tensor<256xf32>
    %v6422 = stablehlo.reshape %v1359 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6424 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6425 = stablehlo.reduce(%v6422 init: %v6423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6426 = stablehlo.divide %v6425, %v6424 : tensor<256xf32>
    %v6427 = stablehlo.reshape %v1359 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6428 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6429 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6430 = stablehlo.reduce(%v6427 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6431 = stablehlo.broadcast_in_dim %v6430, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6432 = stablehlo.divide %v6431, %v6429 : tensor<64x256x14x14xf32>
    %v6433 = stablehlo.subtract %v6427, %v6432 : tensor<64x256x14x14xf32>
    %v6434 = stablehlo.multiply %v6433, %v6433 : tensor<64x256x14x14xf32>
    %v6435 = stablehlo.reduce(%v6434 init: %v6428) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6436 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6437 = stablehlo.divide %v6435, %v6436 : tensor<256xf32>
    %v6438 = stablehlo.reshape %v1391 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6440 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6441 = stablehlo.reduce(%v6438 init: %v6439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6442 = stablehlo.divide %v6441, %v6440 : tensor<1024xf32>
    %v6443 = stablehlo.reshape %v1391 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6445 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6446 = stablehlo.reduce(%v6443 init: %v6444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6447 = stablehlo.broadcast_in_dim %v6446, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6448 = stablehlo.divide %v6447, %v6445 : tensor<64x1024x14x14xf32>
    %v6449 = stablehlo.subtract %v6443, %v6448 : tensor<64x1024x14x14xf32>
    %v6450 = stablehlo.multiply %v6449, %v6449 : tensor<64x1024x14x14xf32>
    %v6451 = stablehlo.reduce(%v6450 init: %v6444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6452 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6453 = stablehlo.divide %v6451, %v6452 : tensor<1024xf32>
    %v6454 = stablehlo.reshape %v1427 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v6455 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6456 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6457 = stablehlo.reduce(%v6454 init: %v6455) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6458 = stablehlo.divide %v6457, %v6456 : tensor<512xf32>
    %v6459 = stablehlo.reshape %v1427 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v6460 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6461 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v6462 = stablehlo.reduce(%v6459 init: %v6460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6463 = stablehlo.broadcast_in_dim %v6462, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v6464 = stablehlo.divide %v6463, %v6461 : tensor<64x512x14x14xf32>
    %v6465 = stablehlo.subtract %v6459, %v6464 : tensor<64x512x14x14xf32>
    %v6466 = stablehlo.multiply %v6465, %v6465 : tensor<64x512x14x14xf32>
    %v6467 = stablehlo.reduce(%v6466 init: %v6460) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6468 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6469 = stablehlo.divide %v6467, %v6468 : tensor<512xf32>
    %v6470 = stablehlo.reshape %v1459 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6471 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6472 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6473 = stablehlo.reduce(%v6470 init: %v6471) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6474 = stablehlo.divide %v6473, %v6472 : tensor<512xf32>
    %v6475 = stablehlo.reshape %v1459 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6476 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6477 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6478 = stablehlo.reduce(%v6475 init: %v6476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6479 = stablehlo.broadcast_in_dim %v6478, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6480 = stablehlo.divide %v6479, %v6477 : tensor<64x512x7x7xf32>
    %v6481 = stablehlo.subtract %v6475, %v6480 : tensor<64x512x7x7xf32>
    %v6482 = stablehlo.multiply %v6481, %v6481 : tensor<64x512x7x7xf32>
    %v6483 = stablehlo.reduce(%v6482 init: %v6476) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6484 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6485 = stablehlo.divide %v6483, %v6484 : tensor<512xf32>
    %v6486 = stablehlo.reshape %v1491 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6488 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6489 = stablehlo.reduce(%v6486 init: %v6487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6490 = stablehlo.divide %v6489, %v6488 : tensor<2048xf32>
    %v6491 = stablehlo.reshape %v1491 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6492 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6493 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6494 = stablehlo.reduce(%v6491 init: %v6492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6495 = stablehlo.broadcast_in_dim %v6494, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6496 = stablehlo.divide %v6495, %v6493 : tensor<64x2048x7x7xf32>
    %v6497 = stablehlo.subtract %v6491, %v6496 : tensor<64x2048x7x7xf32>
    %v6498 = stablehlo.multiply %v6497, %v6497 : tensor<64x2048x7x7xf32>
    %v6499 = stablehlo.reduce(%v6498 init: %v6492) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6500 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6501 = stablehlo.divide %v6499, %v6500 : tensor<2048xf32>
    %v6502 = stablehlo.reshape %v1519 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6504 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6505 = stablehlo.reduce(%v6502 init: %v6503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6506 = stablehlo.divide %v6505, %v6504 : tensor<2048xf32>
    %v6507 = stablehlo.reshape %v1519 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6509 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6510 = stablehlo.reduce(%v6507 init: %v6508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6511 = stablehlo.broadcast_in_dim %v6510, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6512 = stablehlo.divide %v6511, %v6509 : tensor<64x2048x7x7xf32>
    %v6513 = stablehlo.subtract %v6507, %v6512 : tensor<64x2048x7x7xf32>
    %v6514 = stablehlo.multiply %v6513, %v6513 : tensor<64x2048x7x7xf32>
    %v6515 = stablehlo.reduce(%v6514 init: %v6508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6516 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6517 = stablehlo.divide %v6515, %v6516 : tensor<2048xf32>
    %v6518 = stablehlo.reshape %v1555 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6520 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6521 = stablehlo.reduce(%v6518 init: %v6519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6522 = stablehlo.divide %v6521, %v6520 : tensor<512xf32>
    %v6523 = stablehlo.reshape %v1555 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6524 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6525 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6526 = stablehlo.reduce(%v6523 init: %v6524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6527 = stablehlo.broadcast_in_dim %v6526, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6528 = stablehlo.divide %v6527, %v6525 : tensor<64x512x7x7xf32>
    %v6529 = stablehlo.subtract %v6523, %v6528 : tensor<64x512x7x7xf32>
    %v6530 = stablehlo.multiply %v6529, %v6529 : tensor<64x512x7x7xf32>
    %v6531 = stablehlo.reduce(%v6530 init: %v6524) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6532 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6533 = stablehlo.divide %v6531, %v6532 : tensor<512xf32>
    %v6534 = stablehlo.reshape %v1587 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6536 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6537 = stablehlo.reduce(%v6534 init: %v6535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6538 = stablehlo.divide %v6537, %v6536 : tensor<512xf32>
    %v6539 = stablehlo.reshape %v1587 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6541 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6542 = stablehlo.reduce(%v6539 init: %v6540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6543 = stablehlo.broadcast_in_dim %v6542, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6544 = stablehlo.divide %v6543, %v6541 : tensor<64x512x7x7xf32>
    %v6545 = stablehlo.subtract %v6539, %v6544 : tensor<64x512x7x7xf32>
    %v6546 = stablehlo.multiply %v6545, %v6545 : tensor<64x512x7x7xf32>
    %v6547 = stablehlo.reduce(%v6546 init: %v6540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6548 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6549 = stablehlo.divide %v6547, %v6548 : tensor<512xf32>
    %v6550 = stablehlo.reshape %v1619 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6551 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6552 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6553 = stablehlo.reduce(%v6550 init: %v6551) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6554 = stablehlo.divide %v6553, %v6552 : tensor<2048xf32>
    %v6555 = stablehlo.reshape %v1619 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6557 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6558 = stablehlo.reduce(%v6555 init: %v6556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6559 = stablehlo.broadcast_in_dim %v6558, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6560 = stablehlo.divide %v6559, %v6557 : tensor<64x2048x7x7xf32>
    %v6561 = stablehlo.subtract %v6555, %v6560 : tensor<64x2048x7x7xf32>
    %v6562 = stablehlo.multiply %v6561, %v6561 : tensor<64x2048x7x7xf32>
    %v6563 = stablehlo.reduce(%v6562 init: %v6556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6564 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6565 = stablehlo.divide %v6563, %v6564 : tensor<2048xf32>
    %v6566 = stablehlo.reshape %v1655 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6567 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6568 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6569 = stablehlo.reduce(%v6566 init: %v6567) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6570 = stablehlo.divide %v6569, %v6568 : tensor<512xf32>
    %v6571 = stablehlo.reshape %v1655 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6573 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6574 = stablehlo.reduce(%v6571 init: %v6572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6575 = stablehlo.broadcast_in_dim %v6574, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6576 = stablehlo.divide %v6575, %v6573 : tensor<64x512x7x7xf32>
    %v6577 = stablehlo.subtract %v6571, %v6576 : tensor<64x512x7x7xf32>
    %v6578 = stablehlo.multiply %v6577, %v6577 : tensor<64x512x7x7xf32>
    %v6579 = stablehlo.reduce(%v6578 init: %v6572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6580 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6581 = stablehlo.divide %v6579, %v6580 : tensor<512xf32>
    %v6582 = stablehlo.reshape %v1687 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6584 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6585 = stablehlo.reduce(%v6582 init: %v6583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6586 = stablehlo.divide %v6585, %v6584 : tensor<512xf32>
    %v6587 = stablehlo.reshape %v1687 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6588 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6589 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6590 = stablehlo.reduce(%v6587 init: %v6588) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6591 = stablehlo.broadcast_in_dim %v6590, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6592 = stablehlo.divide %v6591, %v6589 : tensor<64x512x7x7xf32>
    %v6593 = stablehlo.subtract %v6587, %v6592 : tensor<64x512x7x7xf32>
    %v6594 = stablehlo.multiply %v6593, %v6593 : tensor<64x512x7x7xf32>
    %v6595 = stablehlo.reduce(%v6594 init: %v6588) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6596 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6597 = stablehlo.divide %v6595, %v6596 : tensor<512xf32>
    %v6598 = stablehlo.reshape %v1719 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6600 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6601 = stablehlo.reduce(%v6598 init: %v6599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6602 = stablehlo.divide %v6601, %v6600 : tensor<2048xf32>
    %v6603 = stablehlo.reshape %v1719 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6604 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6605 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6606 = stablehlo.reduce(%v6603 init: %v6604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6607 = stablehlo.broadcast_in_dim %v6606, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6608 = stablehlo.divide %v6607, %v6605 : tensor<64x2048x7x7xf32>
    %v6609 = stablehlo.subtract %v6603, %v6608 : tensor<64x2048x7x7xf32>
    %v6610 = stablehlo.multiply %v6609, %v6609 : tensor<64x2048x7x7xf32>
    %v6611 = stablehlo.reduce(%v6610 init: %v6604) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6612 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6613 = stablehlo.divide %v6611, %v6612 : tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v5744) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v6614 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6615 = stablehlo.multiply %v6614, %sW : tensor<64x3x7x7xf32>
    %v6616 = stablehlo.add %v6615, %armeansW : tensor<64x3x7x7xf32>
    %v6617 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6618 = stablehlo.multiply %v6617, %sWv : tensor<64x3x7x7xf32>
    %v6619 = stablehlo.add %v6618, %v6616 : tensor<64x3x7x7xf32>
    %v6620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6621 = stablehlo.multiply %v6620, %v6619 : tensor<64x3x7x7xf32>
    %v6622 = stablehlo.subtract %sW, %v6621 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v5762) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v6623 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6624 = stablehlo.multiply %v6623, %sg : tensor<64xf32>
    %v6625 = stablehlo.add %v6624, %armeansg : tensor<64xf32>
    %v6626 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6627 = stablehlo.multiply %v6626, %sgv : tensor<64xf32>
    %v6628 = stablehlo.add %v6627, %v6625 : tensor<64xf32>
    %v6629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6630 = stablehlo.multiply %v6629, %v6628 : tensor<64xf32>
    %v6631 = stablehlo.subtract %sg, %v6630 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v5765) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v6632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6633 = stablehlo.multiply %v6632, %sbt : tensor<64xf32>
    %v6634 = stablehlo.add %v6633, %armeansbt : tensor<64xf32>
    %v6635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6636 = stablehlo.multiply %v6635, %sbtv : tensor<64xf32>
    %v6637 = stablehlo.add %v6636, %v6634 : tensor<64xf32>
    %v6638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6639 = stablehlo.multiply %v6638, %v6637 : tensor<64xf32>
    %v6640 = stablehlo.subtract %sbt, %v6639 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v5581) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x1x1xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x1x1xf32>
    %v6641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6642 = stablehlo.multiply %v6641, %s1b0W1 : tensor<64x64x1x1xf32>
    %v6643 = stablehlo.add %v6642, %armeans1b0W1 : tensor<64x64x1x1xf32>
    %v6644 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6645 = stablehlo.multiply %v6644, %s1b0W1v : tensor<64x64x1x1xf32>
    %v6646 = stablehlo.add %v6645, %v6643 : tensor<64x64x1x1xf32>
    %v6647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6648 = stablehlo.multiply %v6647, %v6646 : tensor<64x64x1x1xf32>
    %v6649 = stablehlo.subtract %s1b0W1, %v6648 : tensor<64x64x1x1xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v5599) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v6650 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6651 = stablehlo.multiply %v6650, %s1b0g1 : tensor<64xf32>
    %v6652 = stablehlo.add %v6651, %armeans1b0g1 : tensor<64xf32>
    %v6653 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6654 = stablehlo.multiply %v6653, %s1b0g1v : tensor<64xf32>
    %v6655 = stablehlo.add %v6654, %v6652 : tensor<64xf32>
    %v6656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6657 = stablehlo.multiply %v6656, %v6655 : tensor<64xf32>
    %v6658 = stablehlo.subtract %s1b0g1, %v6657 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v5602) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v6659 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6660 = stablehlo.multiply %v6659, %s1b0bt1 : tensor<64xf32>
    %v6661 = stablehlo.add %v6660, %armeans1b0bt1 : tensor<64xf32>
    %v6662 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6663 = stablehlo.multiply %v6662, %s1b0bt1v : tensor<64xf32>
    %v6664 = stablehlo.add %v6663, %v6661 : tensor<64xf32>
    %v6665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6666 = stablehlo.multiply %v6665, %v6664 : tensor<64xf32>
    %v6667 = stablehlo.subtract %s1b0bt1, %v6666 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v5611) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v6668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6669 = stablehlo.multiply %v6668, %s1b0W2 : tensor<64x64x3x3xf32>
    %v6670 = stablehlo.add %v6669, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v6671 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6672 = stablehlo.multiply %v6671, %s1b0W2v : tensor<64x64x3x3xf32>
    %v6673 = stablehlo.add %v6672, %v6670 : tensor<64x64x3x3xf32>
    %v6674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6675 = stablehlo.multiply %v6674, %v6673 : tensor<64x64x3x3xf32>
    %v6676 = stablehlo.subtract %s1b0W2, %v6675 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v5629) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v6677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6678 = stablehlo.multiply %v6677, %s1b0g2 : tensor<64xf32>
    %v6679 = stablehlo.add %v6678, %armeans1b0g2 : tensor<64xf32>
    %v6680 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6681 = stablehlo.multiply %v6680, %s1b0g2v : tensor<64xf32>
    %v6682 = stablehlo.add %v6681, %v6679 : tensor<64xf32>
    %v6683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6684 = stablehlo.multiply %v6683, %v6682 : tensor<64xf32>
    %v6685 = stablehlo.subtract %s1b0g2, %v6684 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v5632) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v6686 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6687 = stablehlo.multiply %v6686, %s1b0bt2 : tensor<64xf32>
    %v6688 = stablehlo.add %v6687, %armeans1b0bt2 : tensor<64xf32>
    %v6689 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6690 = stablehlo.multiply %v6689, %s1b0bt2v : tensor<64xf32>
    %v6691 = stablehlo.add %v6690, %v6688 : tensor<64xf32>
    %v6692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6693 = stablehlo.multiply %v6692, %v6691 : tensor<64xf32>
    %v6694 = stablehlo.subtract %s1b0bt2, %v6693 : tensor<64xf32>
    %arsums1b0W3 = "stablehlo.all_reduce"(%v5641) ({
    ^bb0(%aras1b0W3: tensor<f32>, %arbs1b0W3: tensor<f32>):
      %aradds1b0W3 = stablehlo.add %aras1b0W3, %arbs1b0W3 : tensor<f32>
      stablehlo.return %aradds1b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0W3 = stablehlo.divide %arsums1b0W3, %arns1b0W3 : tensor<256x64x1x1xf32>
    %v6695 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6696 = stablehlo.multiply %v6695, %s1b0W3 : tensor<256x64x1x1xf32>
    %v6697 = stablehlo.add %v6696, %armeans1b0W3 : tensor<256x64x1x1xf32>
    %v6698 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6699 = stablehlo.multiply %v6698, %s1b0W3v : tensor<256x64x1x1xf32>
    %v6700 = stablehlo.add %v6699, %v6697 : tensor<256x64x1x1xf32>
    %v6701 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6702 = stablehlo.multiply %v6701, %v6700 : tensor<256x64x1x1xf32>
    %v6703 = stablehlo.subtract %s1b0W3, %v6702 : tensor<256x64x1x1xf32>
    %arsums1b0g3 = "stablehlo.all_reduce"(%v5659) ({
    ^bb0(%aras1b0g3: tensor<f32>, %arbs1b0g3: tensor<f32>):
      %aradds1b0g3 = stablehlo.add %aras1b0g3, %arbs1b0g3 : tensor<f32>
      stablehlo.return %aradds1b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3 = stablehlo.divide %arsums1b0g3, %arns1b0g3 : tensor<256xf32>
    %v6704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6705 = stablehlo.multiply %v6704, %s1b0g3 : tensor<256xf32>
    %v6706 = stablehlo.add %v6705, %armeans1b0g3 : tensor<256xf32>
    %v6707 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6708 = stablehlo.multiply %v6707, %s1b0g3v : tensor<256xf32>
    %v6709 = stablehlo.add %v6708, %v6706 : tensor<256xf32>
    %v6710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6711 = stablehlo.multiply %v6710, %v6709 : tensor<256xf32>
    %v6712 = stablehlo.subtract %s1b0g3, %v6711 : tensor<256xf32>
    %arsums1b0bt3 = "stablehlo.all_reduce"(%v5662) ({
    ^bb0(%aras1b0bt3: tensor<f32>, %arbs1b0bt3: tensor<f32>):
      %aradds1b0bt3 = stablehlo.add %aras1b0bt3, %arbs1b0bt3 : tensor<f32>
      stablehlo.return %aradds1b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0bt3 = stablehlo.divide %arsums1b0bt3, %arns1b0bt3 : tensor<256xf32>
    %v6713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6714 = stablehlo.multiply %v6713, %s1b0bt3 : tensor<256xf32>
    %v6715 = stablehlo.add %v6714, %armeans1b0bt3 : tensor<256xf32>
    %v6716 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6717 = stablehlo.multiply %v6716, %s1b0bt3v : tensor<256xf32>
    %v6718 = stablehlo.add %v6717, %v6715 : tensor<256xf32>
    %v6719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6720 = stablehlo.multiply %v6719, %v6718 : tensor<256xf32>
    %v6721 = stablehlo.subtract %s1b0bt3, %v6720 : tensor<256xf32>
    %arsums1b0Wp = "stablehlo.all_reduce"(%v5671) ({
    ^bb0(%aras1b0Wp: tensor<f32>, %arbs1b0Wp: tensor<f32>):
      %aradds1b0Wp = stablehlo.add %aras1b0Wp, %arbs1b0Wp : tensor<f32>
      stablehlo.return %aradds1b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0Wp = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0Wp = stablehlo.divide %arsums1b0Wp, %arns1b0Wp : tensor<256x64x1x1xf32>
    %v6722 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6723 = stablehlo.multiply %v6722, %s1b0Wp : tensor<256x64x1x1xf32>
    %v6724 = stablehlo.add %v6723, %armeans1b0Wp : tensor<256x64x1x1xf32>
    %v6725 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6726 = stablehlo.multiply %v6725, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v6727 = stablehlo.add %v6726, %v6724 : tensor<256x64x1x1xf32>
    %v6728 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6729 = stablehlo.multiply %v6728, %v6727 : tensor<256x64x1x1xf32>
    %v6730 = stablehlo.subtract %s1b0Wp, %v6729 : tensor<256x64x1x1xf32>
    %arsums1b0gp = "stablehlo.all_reduce"(%v5689) ({
    ^bb0(%aras1b0gp: tensor<f32>, %arbs1b0gp: tensor<f32>):
      %aradds1b0gp = stablehlo.add %aras1b0gp, %arbs1b0gp : tensor<f32>
      stablehlo.return %aradds1b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gp = stablehlo.divide %arsums1b0gp, %arns1b0gp : tensor<256xf32>
    %v6731 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6732 = stablehlo.multiply %v6731, %s1b0gp : tensor<256xf32>
    %v6733 = stablehlo.add %v6732, %armeans1b0gp : tensor<256xf32>
    %v6734 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6735 = stablehlo.multiply %v6734, %s1b0gpv : tensor<256xf32>
    %v6736 = stablehlo.add %v6735, %v6733 : tensor<256xf32>
    %v6737 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6738 = stablehlo.multiply %v6737, %v6736 : tensor<256xf32>
    %v6739 = stablehlo.subtract %s1b0gp, %v6738 : tensor<256xf32>
    %arsums1b0btp = "stablehlo.all_reduce"(%v5692) ({
    ^bb0(%aras1b0btp: tensor<f32>, %arbs1b0btp: tensor<f32>):
      %aradds1b0btp = stablehlo.add %aras1b0btp, %arbs1b0btp : tensor<f32>
      stablehlo.return %aradds1b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0btp = stablehlo.divide %arsums1b0btp, %arns1b0btp : tensor<256xf32>
    %v6740 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6741 = stablehlo.multiply %v6740, %s1b0btp : tensor<256xf32>
    %v6742 = stablehlo.add %v6741, %armeans1b0btp : tensor<256xf32>
    %v6743 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6744 = stablehlo.multiply %v6743, %s1b0btpv : tensor<256xf32>
    %v6745 = stablehlo.add %v6744, %v6742 : tensor<256xf32>
    %v6746 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6747 = stablehlo.multiply %v6746, %v6745 : tensor<256xf32>
    %v6748 = stablehlo.subtract %s1b0btp, %v6747 : tensor<256xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v5317) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x256x1x1xf32>
    %v6749 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6750 = stablehlo.multiply %v6749, %s1b1W1 : tensor<64x256x1x1xf32>
    %v6751 = stablehlo.add %v6750, %armeans1b1W1 : tensor<64x256x1x1xf32>
    %v6752 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6753 = stablehlo.multiply %v6752, %s1b1W1v : tensor<64x256x1x1xf32>
    %v6754 = stablehlo.add %v6753, %v6751 : tensor<64x256x1x1xf32>
    %v6755 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6756 = stablehlo.multiply %v6755, %v6754 : tensor<64x256x1x1xf32>
    %v6757 = stablehlo.subtract %s1b1W1, %v6756 : tensor<64x256x1x1xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v5335) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v6758 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6759 = stablehlo.multiply %v6758, %s1b1g1 : tensor<64xf32>
    %v6760 = stablehlo.add %v6759, %armeans1b1g1 : tensor<64xf32>
    %v6761 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6762 = stablehlo.multiply %v6761, %s1b1g1v : tensor<64xf32>
    %v6763 = stablehlo.add %v6762, %v6760 : tensor<64xf32>
    %v6764 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6765 = stablehlo.multiply %v6764, %v6763 : tensor<64xf32>
    %v6766 = stablehlo.subtract %s1b1g1, %v6765 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v5338) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v6767 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6768 = stablehlo.multiply %v6767, %s1b1bt1 : tensor<64xf32>
    %v6769 = stablehlo.add %v6768, %armeans1b1bt1 : tensor<64xf32>
    %v6770 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6771 = stablehlo.multiply %v6770, %s1b1bt1v : tensor<64xf32>
    %v6772 = stablehlo.add %v6771, %v6769 : tensor<64xf32>
    %v6773 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6774 = stablehlo.multiply %v6773, %v6772 : tensor<64xf32>
    %v6775 = stablehlo.subtract %s1b1bt1, %v6774 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v5347) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v6776 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6777 = stablehlo.multiply %v6776, %s1b1W2 : tensor<64x64x3x3xf32>
    %v6778 = stablehlo.add %v6777, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v6779 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6780 = stablehlo.multiply %v6779, %s1b1W2v : tensor<64x64x3x3xf32>
    %v6781 = stablehlo.add %v6780, %v6778 : tensor<64x64x3x3xf32>
    %v6782 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6783 = stablehlo.multiply %v6782, %v6781 : tensor<64x64x3x3xf32>
    %v6784 = stablehlo.subtract %s1b1W2, %v6783 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v5365) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v6785 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6786 = stablehlo.multiply %v6785, %s1b1g2 : tensor<64xf32>
    %v6787 = stablehlo.add %v6786, %armeans1b1g2 : tensor<64xf32>
    %v6788 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6789 = stablehlo.multiply %v6788, %s1b1g2v : tensor<64xf32>
    %v6790 = stablehlo.add %v6789, %v6787 : tensor<64xf32>
    %v6791 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6792 = stablehlo.multiply %v6791, %v6790 : tensor<64xf32>
    %v6793 = stablehlo.subtract %s1b1g2, %v6792 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v5368) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v6794 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6795 = stablehlo.multiply %v6794, %s1b1bt2 : tensor<64xf32>
    %v6796 = stablehlo.add %v6795, %armeans1b1bt2 : tensor<64xf32>
    %v6797 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6798 = stablehlo.multiply %v6797, %s1b1bt2v : tensor<64xf32>
    %v6799 = stablehlo.add %v6798, %v6796 : tensor<64xf32>
    %v6800 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6801 = stablehlo.multiply %v6800, %v6799 : tensor<64xf32>
    %v6802 = stablehlo.subtract %s1b1bt2, %v6801 : tensor<64xf32>
    %arsums1b1W3 = "stablehlo.all_reduce"(%v5377) ({
    ^bb0(%aras1b1W3: tensor<f32>, %arbs1b1W3: tensor<f32>):
      %aradds1b1W3 = stablehlo.add %aras1b1W3, %arbs1b1W3 : tensor<f32>
      stablehlo.return %aradds1b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b1W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b1W3 = stablehlo.divide %arsums1b1W3, %arns1b1W3 : tensor<256x64x1x1xf32>
    %v6803 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6804 = stablehlo.multiply %v6803, %s1b1W3 : tensor<256x64x1x1xf32>
    %v6805 = stablehlo.add %v6804, %armeans1b1W3 : tensor<256x64x1x1xf32>
    %v6806 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6807 = stablehlo.multiply %v6806, %s1b1W3v : tensor<256x64x1x1xf32>
    %v6808 = stablehlo.add %v6807, %v6805 : tensor<256x64x1x1xf32>
    %v6809 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6810 = stablehlo.multiply %v6809, %v6808 : tensor<256x64x1x1xf32>
    %v6811 = stablehlo.subtract %s1b1W3, %v6810 : tensor<256x64x1x1xf32>
    %arsums1b1g3 = "stablehlo.all_reduce"(%v5395) ({
    ^bb0(%aras1b1g3: tensor<f32>, %arbs1b1g3: tensor<f32>):
      %aradds1b1g3 = stablehlo.add %aras1b1g3, %arbs1b1g3 : tensor<f32>
      stablehlo.return %aradds1b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3 = stablehlo.divide %arsums1b1g3, %arns1b1g3 : tensor<256xf32>
    %v6812 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6813 = stablehlo.multiply %v6812, %s1b1g3 : tensor<256xf32>
    %v6814 = stablehlo.add %v6813, %armeans1b1g3 : tensor<256xf32>
    %v6815 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6816 = stablehlo.multiply %v6815, %s1b1g3v : tensor<256xf32>
    %v6817 = stablehlo.add %v6816, %v6814 : tensor<256xf32>
    %v6818 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6819 = stablehlo.multiply %v6818, %v6817 : tensor<256xf32>
    %v6820 = stablehlo.subtract %s1b1g3, %v6819 : tensor<256xf32>
    %arsums1b1bt3 = "stablehlo.all_reduce"(%v5398) ({
    ^bb0(%aras1b1bt3: tensor<f32>, %arbs1b1bt3: tensor<f32>):
      %aradds1b1bt3 = stablehlo.add %aras1b1bt3, %arbs1b1bt3 : tensor<f32>
      stablehlo.return %aradds1b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1bt3 = stablehlo.divide %arsums1b1bt3, %arns1b1bt3 : tensor<256xf32>
    %v6821 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6822 = stablehlo.multiply %v6821, %s1b1bt3 : tensor<256xf32>
    %v6823 = stablehlo.add %v6822, %armeans1b1bt3 : tensor<256xf32>
    %v6824 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6825 = stablehlo.multiply %v6824, %s1b1bt3v : tensor<256xf32>
    %v6826 = stablehlo.add %v6825, %v6823 : tensor<256xf32>
    %v6827 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6828 = stablehlo.multiply %v6827, %v6826 : tensor<256xf32>
    %v6829 = stablehlo.subtract %s1b1bt3, %v6828 : tensor<256xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v5091) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x256x1x1xf32>
    %v6830 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6831 = stablehlo.multiply %v6830, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6832 = stablehlo.add %v6831, %armeans1b2W1 : tensor<64x256x1x1xf32>
    %v6833 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6834 = stablehlo.multiply %v6833, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6835 = stablehlo.add %v6834, %v6832 : tensor<64x256x1x1xf32>
    %v6836 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6837 = stablehlo.multiply %v6836, %v6835 : tensor<64x256x1x1xf32>
    %v6838 = stablehlo.subtract %s1b2W1, %v6837 : tensor<64x256x1x1xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v5109) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v6839 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6840 = stablehlo.multiply %v6839, %s1b2g1 : tensor<64xf32>
    %v6841 = stablehlo.add %v6840, %armeans1b2g1 : tensor<64xf32>
    %v6842 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6843 = stablehlo.multiply %v6842, %s1b2g1v : tensor<64xf32>
    %v6844 = stablehlo.add %v6843, %v6841 : tensor<64xf32>
    %v6845 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6846 = stablehlo.multiply %v6845, %v6844 : tensor<64xf32>
    %v6847 = stablehlo.subtract %s1b2g1, %v6846 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v5112) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v6848 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6849 = stablehlo.multiply %v6848, %s1b2bt1 : tensor<64xf32>
    %v6850 = stablehlo.add %v6849, %armeans1b2bt1 : tensor<64xf32>
    %v6851 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6852 = stablehlo.multiply %v6851, %s1b2bt1v : tensor<64xf32>
    %v6853 = stablehlo.add %v6852, %v6850 : tensor<64xf32>
    %v6854 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6855 = stablehlo.multiply %v6854, %v6853 : tensor<64xf32>
    %v6856 = stablehlo.subtract %s1b2bt1, %v6855 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v5121) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v6857 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6858 = stablehlo.multiply %v6857, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6859 = stablehlo.add %v6858, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v6860 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6861 = stablehlo.multiply %v6860, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6862 = stablehlo.add %v6861, %v6859 : tensor<64x64x3x3xf32>
    %v6863 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6864 = stablehlo.multiply %v6863, %v6862 : tensor<64x64x3x3xf32>
    %v6865 = stablehlo.subtract %s1b2W2, %v6864 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v5139) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v6866 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6867 = stablehlo.multiply %v6866, %s1b2g2 : tensor<64xf32>
    %v6868 = stablehlo.add %v6867, %armeans1b2g2 : tensor<64xf32>
    %v6869 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6870 = stablehlo.multiply %v6869, %s1b2g2v : tensor<64xf32>
    %v6871 = stablehlo.add %v6870, %v6868 : tensor<64xf32>
    %v6872 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6873 = stablehlo.multiply %v6872, %v6871 : tensor<64xf32>
    %v6874 = stablehlo.subtract %s1b2g2, %v6873 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v5142) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v6875 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6876 = stablehlo.multiply %v6875, %s1b2bt2 : tensor<64xf32>
    %v6877 = stablehlo.add %v6876, %armeans1b2bt2 : tensor<64xf32>
    %v6878 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6879 = stablehlo.multiply %v6878, %s1b2bt2v : tensor<64xf32>
    %v6880 = stablehlo.add %v6879, %v6877 : tensor<64xf32>
    %v6881 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6882 = stablehlo.multiply %v6881, %v6880 : tensor<64xf32>
    %v6883 = stablehlo.subtract %s1b2bt2, %v6882 : tensor<64xf32>
    %arsums1b2W3 = "stablehlo.all_reduce"(%v5151) ({
    ^bb0(%aras1b2W3: tensor<f32>, %arbs1b2W3: tensor<f32>):
      %aradds1b2W3 = stablehlo.add %aras1b2W3, %arbs1b2W3 : tensor<f32>
      stablehlo.return %aradds1b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b2W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b2W3 = stablehlo.divide %arsums1b2W3, %arns1b2W3 : tensor<256x64x1x1xf32>
    %v6884 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6885 = stablehlo.multiply %v6884, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6886 = stablehlo.add %v6885, %armeans1b2W3 : tensor<256x64x1x1xf32>
    %v6887 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6888 = stablehlo.multiply %v6887, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6889 = stablehlo.add %v6888, %v6886 : tensor<256x64x1x1xf32>
    %v6890 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6891 = stablehlo.multiply %v6890, %v6889 : tensor<256x64x1x1xf32>
    %v6892 = stablehlo.subtract %s1b2W3, %v6891 : tensor<256x64x1x1xf32>
    %arsums1b2g3 = "stablehlo.all_reduce"(%v5169) ({
    ^bb0(%aras1b2g3: tensor<f32>, %arbs1b2g3: tensor<f32>):
      %aradds1b2g3 = stablehlo.add %aras1b2g3, %arbs1b2g3 : tensor<f32>
      stablehlo.return %aradds1b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3 = stablehlo.divide %arsums1b2g3, %arns1b2g3 : tensor<256xf32>
    %v6893 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6894 = stablehlo.multiply %v6893, %s1b2g3 : tensor<256xf32>
    %v6895 = stablehlo.add %v6894, %armeans1b2g3 : tensor<256xf32>
    %v6896 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6897 = stablehlo.multiply %v6896, %s1b2g3v : tensor<256xf32>
    %v6898 = stablehlo.add %v6897, %v6895 : tensor<256xf32>
    %v6899 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6900 = stablehlo.multiply %v6899, %v6898 : tensor<256xf32>
    %v6901 = stablehlo.subtract %s1b2g3, %v6900 : tensor<256xf32>
    %arsums1b2bt3 = "stablehlo.all_reduce"(%v5172) ({
    ^bb0(%aras1b2bt3: tensor<f32>, %arbs1b2bt3: tensor<f32>):
      %aradds1b2bt3 = stablehlo.add %aras1b2bt3, %arbs1b2bt3 : tensor<f32>
      stablehlo.return %aradds1b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2bt3 = stablehlo.divide %arsums1b2bt3, %arns1b2bt3 : tensor<256xf32>
    %v6902 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6903 = stablehlo.multiply %v6902, %s1b2bt3 : tensor<256xf32>
    %v6904 = stablehlo.add %v6903, %armeans1b2bt3 : tensor<256xf32>
    %v6905 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6906 = stablehlo.multiply %v6905, %s1b2bt3v : tensor<256xf32>
    %v6907 = stablehlo.add %v6906, %v6904 : tensor<256xf32>
    %v6908 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6909 = stablehlo.multiply %v6908, %v6907 : tensor<256xf32>
    %v6910 = stablehlo.subtract %s1b2bt3, %v6909 : tensor<256xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v4831) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x256x1x1xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x256x1x1xf32>
    %v6911 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6912 = stablehlo.multiply %v6911, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6913 = stablehlo.add %v6912, %armeans2b0W1 : tensor<128x256x1x1xf32>
    %v6914 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6915 = stablehlo.multiply %v6914, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6916 = stablehlo.add %v6915, %v6913 : tensor<128x256x1x1xf32>
    %v6917 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6918 = stablehlo.multiply %v6917, %v6916 : tensor<128x256x1x1xf32>
    %v6919 = stablehlo.subtract %s2b0W1, %v6918 : tensor<128x256x1x1xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v4849) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v6920 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6921 = stablehlo.multiply %v6920, %s2b0g1 : tensor<128xf32>
    %v6922 = stablehlo.add %v6921, %armeans2b0g1 : tensor<128xf32>
    %v6923 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6924 = stablehlo.multiply %v6923, %s2b0g1v : tensor<128xf32>
    %v6925 = stablehlo.add %v6924, %v6922 : tensor<128xf32>
    %v6926 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6927 = stablehlo.multiply %v6926, %v6925 : tensor<128xf32>
    %v6928 = stablehlo.subtract %s2b0g1, %v6927 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v4852) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v6929 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6930 = stablehlo.multiply %v6929, %s2b0bt1 : tensor<128xf32>
    %v6931 = stablehlo.add %v6930, %armeans2b0bt1 : tensor<128xf32>
    %v6932 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6933 = stablehlo.multiply %v6932, %s2b0bt1v : tensor<128xf32>
    %v6934 = stablehlo.add %v6933, %v6931 : tensor<128xf32>
    %v6935 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6936 = stablehlo.multiply %v6935, %v6934 : tensor<128xf32>
    %v6937 = stablehlo.subtract %s2b0bt1, %v6936 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v4863) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v6938 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6939 = stablehlo.multiply %v6938, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6940 = stablehlo.add %v6939, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v6941 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6942 = stablehlo.multiply %v6941, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6943 = stablehlo.add %v6942, %v6940 : tensor<128x128x3x3xf32>
    %v6944 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6945 = stablehlo.multiply %v6944, %v6943 : tensor<128x128x3x3xf32>
    %v6946 = stablehlo.subtract %s2b0W2, %v6945 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v4881) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v6947 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6948 = stablehlo.multiply %v6947, %s2b0g2 : tensor<128xf32>
    %v6949 = stablehlo.add %v6948, %armeans2b0g2 : tensor<128xf32>
    %v6950 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6951 = stablehlo.multiply %v6950, %s2b0g2v : tensor<128xf32>
    %v6952 = stablehlo.add %v6951, %v6949 : tensor<128xf32>
    %v6953 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6954 = stablehlo.multiply %v6953, %v6952 : tensor<128xf32>
    %v6955 = stablehlo.subtract %s2b0g2, %v6954 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v4884) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v6956 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6957 = stablehlo.multiply %v6956, %s2b0bt2 : tensor<128xf32>
    %v6958 = stablehlo.add %v6957, %armeans2b0bt2 : tensor<128xf32>
    %v6959 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6960 = stablehlo.multiply %v6959, %s2b0bt2v : tensor<128xf32>
    %v6961 = stablehlo.add %v6960, %v6958 : tensor<128xf32>
    %v6962 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6963 = stablehlo.multiply %v6962, %v6961 : tensor<128xf32>
    %v6964 = stablehlo.subtract %s2b0bt2, %v6963 : tensor<128xf32>
    %arsums2b0W3 = "stablehlo.all_reduce"(%v4893) ({
    ^bb0(%aras2b0W3: tensor<f32>, %arbs2b0W3: tensor<f32>):
      %aradds2b0W3 = stablehlo.add %aras2b0W3, %arbs2b0W3 : tensor<f32>
      stablehlo.return %aradds2b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b0W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b0W3 = stablehlo.divide %arsums2b0W3, %arns2b0W3 : tensor<512x128x1x1xf32>
    %v6965 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6966 = stablehlo.multiply %v6965, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6967 = stablehlo.add %v6966, %armeans2b0W3 : tensor<512x128x1x1xf32>
    %v6968 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6969 = stablehlo.multiply %v6968, %s2b0W3v : tensor<512x128x1x1xf32>
    %v6970 = stablehlo.add %v6969, %v6967 : tensor<512x128x1x1xf32>
    %v6971 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6972 = stablehlo.multiply %v6971, %v6970 : tensor<512x128x1x1xf32>
    %v6973 = stablehlo.subtract %s2b0W3, %v6972 : tensor<512x128x1x1xf32>
    %arsums2b0g3 = "stablehlo.all_reduce"(%v4911) ({
    ^bb0(%aras2b0g3: tensor<f32>, %arbs2b0g3: tensor<f32>):
      %aradds2b0g3 = stablehlo.add %aras2b0g3, %arbs2b0g3 : tensor<f32>
      stablehlo.return %aradds2b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3 = stablehlo.divide %arsums2b0g3, %arns2b0g3 : tensor<512xf32>
    %v6974 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6975 = stablehlo.multiply %v6974, %s2b0g3 : tensor<512xf32>
    %v6976 = stablehlo.add %v6975, %armeans2b0g3 : tensor<512xf32>
    %v6977 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6978 = stablehlo.multiply %v6977, %s2b0g3v : tensor<512xf32>
    %v6979 = stablehlo.add %v6978, %v6976 : tensor<512xf32>
    %v6980 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6981 = stablehlo.multiply %v6980, %v6979 : tensor<512xf32>
    %v6982 = stablehlo.subtract %s2b0g3, %v6981 : tensor<512xf32>
    %arsums2b0bt3 = "stablehlo.all_reduce"(%v4914) ({
    ^bb0(%aras2b0bt3: tensor<f32>, %arbs2b0bt3: tensor<f32>):
      %aradds2b0bt3 = stablehlo.add %aras2b0bt3, %arbs2b0bt3 : tensor<f32>
      stablehlo.return %aradds2b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0bt3 = stablehlo.divide %arsums2b0bt3, %arns2b0bt3 : tensor<512xf32>
    %v6983 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6984 = stablehlo.multiply %v6983, %s2b0bt3 : tensor<512xf32>
    %v6985 = stablehlo.add %v6984, %armeans2b0bt3 : tensor<512xf32>
    %v6986 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6987 = stablehlo.multiply %v6986, %s2b0bt3v : tensor<512xf32>
    %v6988 = stablehlo.add %v6987, %v6985 : tensor<512xf32>
    %v6989 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6990 = stablehlo.multiply %v6989, %v6988 : tensor<512xf32>
    %v6991 = stablehlo.subtract %s2b0bt3, %v6990 : tensor<512xf32>
    %arsums2b0Wp = "stablehlo.all_reduce"(%v4925) ({
    ^bb0(%aras2b0Wp: tensor<f32>, %arbs2b0Wp: tensor<f32>):
      %aradds2b0Wp = stablehlo.add %aras2b0Wp, %arbs2b0Wp : tensor<f32>
      stablehlo.return %aradds2b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arns2b0Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeans2b0Wp = stablehlo.divide %arsums2b0Wp, %arns2b0Wp : tensor<512x256x1x1xf32>
    %v6992 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6993 = stablehlo.multiply %v6992, %s2b0Wp : tensor<512x256x1x1xf32>
    %v6994 = stablehlo.add %v6993, %armeans2b0Wp : tensor<512x256x1x1xf32>
    %v6995 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6996 = stablehlo.multiply %v6995, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v6997 = stablehlo.add %v6996, %v6994 : tensor<512x256x1x1xf32>
    %v6998 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6999 = stablehlo.multiply %v6998, %v6997 : tensor<512x256x1x1xf32>
    %v7000 = stablehlo.subtract %s2b0Wp, %v6999 : tensor<512x256x1x1xf32>
    %arsums2b0gp = "stablehlo.all_reduce"(%v4943) ({
    ^bb0(%aras2b0gp: tensor<f32>, %arbs2b0gp: tensor<f32>):
      %aradds2b0gp = stablehlo.add %aras2b0gp, %arbs2b0gp : tensor<f32>
      stablehlo.return %aradds2b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gp = stablehlo.divide %arsums2b0gp, %arns2b0gp : tensor<512xf32>
    %v7001 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7002 = stablehlo.multiply %v7001, %s2b0gp : tensor<512xf32>
    %v7003 = stablehlo.add %v7002, %armeans2b0gp : tensor<512xf32>
    %v7004 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7005 = stablehlo.multiply %v7004, %s2b0gpv : tensor<512xf32>
    %v7006 = stablehlo.add %v7005, %v7003 : tensor<512xf32>
    %v7007 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7008 = stablehlo.multiply %v7007, %v7006 : tensor<512xf32>
    %v7009 = stablehlo.subtract %s2b0gp, %v7008 : tensor<512xf32>
    %arsums2b0btp = "stablehlo.all_reduce"(%v4946) ({
    ^bb0(%aras2b0btp: tensor<f32>, %arbs2b0btp: tensor<f32>):
      %aradds2b0btp = stablehlo.add %aras2b0btp, %arbs2b0btp : tensor<f32>
      stablehlo.return %aradds2b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0btp = stablehlo.divide %arsums2b0btp, %arns2b0btp : tensor<512xf32>
    %v7010 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7011 = stablehlo.multiply %v7010, %s2b0btp : tensor<512xf32>
    %v7012 = stablehlo.add %v7011, %armeans2b0btp : tensor<512xf32>
    %v7013 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7014 = stablehlo.multiply %v7013, %s2b0btpv : tensor<512xf32>
    %v7015 = stablehlo.add %v7014, %v7012 : tensor<512xf32>
    %v7016 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7017 = stablehlo.multiply %v7016, %v7015 : tensor<512xf32>
    %v7018 = stablehlo.subtract %s2b0btp, %v7017 : tensor<512xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v4563) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x512x1x1xf32>
    %v7019 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7020 = stablehlo.multiply %v7019, %s2b1W1 : tensor<128x512x1x1xf32>
    %v7021 = stablehlo.add %v7020, %armeans2b1W1 : tensor<128x512x1x1xf32>
    %v7022 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7023 = stablehlo.multiply %v7022, %s2b1W1v : tensor<128x512x1x1xf32>
    %v7024 = stablehlo.add %v7023, %v7021 : tensor<128x512x1x1xf32>
    %v7025 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7026 = stablehlo.multiply %v7025, %v7024 : tensor<128x512x1x1xf32>
    %v7027 = stablehlo.subtract %s2b1W1, %v7026 : tensor<128x512x1x1xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v4581) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v7028 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7029 = stablehlo.multiply %v7028, %s2b1g1 : tensor<128xf32>
    %v7030 = stablehlo.add %v7029, %armeans2b1g1 : tensor<128xf32>
    %v7031 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7032 = stablehlo.multiply %v7031, %s2b1g1v : tensor<128xf32>
    %v7033 = stablehlo.add %v7032, %v7030 : tensor<128xf32>
    %v7034 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7035 = stablehlo.multiply %v7034, %v7033 : tensor<128xf32>
    %v7036 = stablehlo.subtract %s2b1g1, %v7035 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v4584) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v7037 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7038 = stablehlo.multiply %v7037, %s2b1bt1 : tensor<128xf32>
    %v7039 = stablehlo.add %v7038, %armeans2b1bt1 : tensor<128xf32>
    %v7040 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7041 = stablehlo.multiply %v7040, %s2b1bt1v : tensor<128xf32>
    %v7042 = stablehlo.add %v7041, %v7039 : tensor<128xf32>
    %v7043 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7044 = stablehlo.multiply %v7043, %v7042 : tensor<128xf32>
    %v7045 = stablehlo.subtract %s2b1bt1, %v7044 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v4593) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v7046 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7047 = stablehlo.multiply %v7046, %s2b1W2 : tensor<128x128x3x3xf32>
    %v7048 = stablehlo.add %v7047, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v7049 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7050 = stablehlo.multiply %v7049, %s2b1W2v : tensor<128x128x3x3xf32>
    %v7051 = stablehlo.add %v7050, %v7048 : tensor<128x128x3x3xf32>
    %v7052 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7053 = stablehlo.multiply %v7052, %v7051 : tensor<128x128x3x3xf32>
    %v7054 = stablehlo.subtract %s2b1W2, %v7053 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v4611) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v7055 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7056 = stablehlo.multiply %v7055, %s2b1g2 : tensor<128xf32>
    %v7057 = stablehlo.add %v7056, %armeans2b1g2 : tensor<128xf32>
    %v7058 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7059 = stablehlo.multiply %v7058, %s2b1g2v : tensor<128xf32>
    %v7060 = stablehlo.add %v7059, %v7057 : tensor<128xf32>
    %v7061 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7062 = stablehlo.multiply %v7061, %v7060 : tensor<128xf32>
    %v7063 = stablehlo.subtract %s2b1g2, %v7062 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v4614) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v7064 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7065 = stablehlo.multiply %v7064, %s2b1bt2 : tensor<128xf32>
    %v7066 = stablehlo.add %v7065, %armeans2b1bt2 : tensor<128xf32>
    %v7067 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7068 = stablehlo.multiply %v7067, %s2b1bt2v : tensor<128xf32>
    %v7069 = stablehlo.add %v7068, %v7066 : tensor<128xf32>
    %v7070 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7071 = stablehlo.multiply %v7070, %v7069 : tensor<128xf32>
    %v7072 = stablehlo.subtract %s2b1bt2, %v7071 : tensor<128xf32>
    %arsums2b1W3 = "stablehlo.all_reduce"(%v4623) ({
    ^bb0(%aras2b1W3: tensor<f32>, %arbs2b1W3: tensor<f32>):
      %aradds2b1W3 = stablehlo.add %aras2b1W3, %arbs2b1W3 : tensor<f32>
      stablehlo.return %aradds2b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b1W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b1W3 = stablehlo.divide %arsums2b1W3, %arns2b1W3 : tensor<512x128x1x1xf32>
    %v7073 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7074 = stablehlo.multiply %v7073, %s2b1W3 : tensor<512x128x1x1xf32>
    %v7075 = stablehlo.add %v7074, %armeans2b1W3 : tensor<512x128x1x1xf32>
    %v7076 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7077 = stablehlo.multiply %v7076, %s2b1W3v : tensor<512x128x1x1xf32>
    %v7078 = stablehlo.add %v7077, %v7075 : tensor<512x128x1x1xf32>
    %v7079 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7080 = stablehlo.multiply %v7079, %v7078 : tensor<512x128x1x1xf32>
    %v7081 = stablehlo.subtract %s2b1W3, %v7080 : tensor<512x128x1x1xf32>
    %arsums2b1g3 = "stablehlo.all_reduce"(%v4641) ({
    ^bb0(%aras2b1g3: tensor<f32>, %arbs2b1g3: tensor<f32>):
      %aradds2b1g3 = stablehlo.add %aras2b1g3, %arbs2b1g3 : tensor<f32>
      stablehlo.return %aradds2b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3 = stablehlo.divide %arsums2b1g3, %arns2b1g3 : tensor<512xf32>
    %v7082 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7083 = stablehlo.multiply %v7082, %s2b1g3 : tensor<512xf32>
    %v7084 = stablehlo.add %v7083, %armeans2b1g3 : tensor<512xf32>
    %v7085 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7086 = stablehlo.multiply %v7085, %s2b1g3v : tensor<512xf32>
    %v7087 = stablehlo.add %v7086, %v7084 : tensor<512xf32>
    %v7088 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7089 = stablehlo.multiply %v7088, %v7087 : tensor<512xf32>
    %v7090 = stablehlo.subtract %s2b1g3, %v7089 : tensor<512xf32>
    %arsums2b1bt3 = "stablehlo.all_reduce"(%v4644) ({
    ^bb0(%aras2b1bt3: tensor<f32>, %arbs2b1bt3: tensor<f32>):
      %aradds2b1bt3 = stablehlo.add %aras2b1bt3, %arbs2b1bt3 : tensor<f32>
      stablehlo.return %aradds2b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1bt3 = stablehlo.divide %arsums2b1bt3, %arns2b1bt3 : tensor<512xf32>
    %v7091 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7092 = stablehlo.multiply %v7091, %s2b1bt3 : tensor<512xf32>
    %v7093 = stablehlo.add %v7092, %armeans2b1bt3 : tensor<512xf32>
    %v7094 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7095 = stablehlo.multiply %v7094, %s2b1bt3v : tensor<512xf32>
    %v7096 = stablehlo.add %v7095, %v7093 : tensor<512xf32>
    %v7097 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7098 = stablehlo.multiply %v7097, %v7096 : tensor<512xf32>
    %v7099 = stablehlo.subtract %s2b1bt3, %v7098 : tensor<512xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v4337) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x512x1x1xf32>
    %v7100 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7101 = stablehlo.multiply %v7100, %s2b2W1 : tensor<128x512x1x1xf32>
    %v7102 = stablehlo.add %v7101, %armeans2b2W1 : tensor<128x512x1x1xf32>
    %v7103 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7104 = stablehlo.multiply %v7103, %s2b2W1v : tensor<128x512x1x1xf32>
    %v7105 = stablehlo.add %v7104, %v7102 : tensor<128x512x1x1xf32>
    %v7106 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7107 = stablehlo.multiply %v7106, %v7105 : tensor<128x512x1x1xf32>
    %v7108 = stablehlo.subtract %s2b2W1, %v7107 : tensor<128x512x1x1xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v4355) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v7109 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7110 = stablehlo.multiply %v7109, %s2b2g1 : tensor<128xf32>
    %v7111 = stablehlo.add %v7110, %armeans2b2g1 : tensor<128xf32>
    %v7112 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7113 = stablehlo.multiply %v7112, %s2b2g1v : tensor<128xf32>
    %v7114 = stablehlo.add %v7113, %v7111 : tensor<128xf32>
    %v7115 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7116 = stablehlo.multiply %v7115, %v7114 : tensor<128xf32>
    %v7117 = stablehlo.subtract %s2b2g1, %v7116 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v4358) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v7118 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7119 = stablehlo.multiply %v7118, %s2b2bt1 : tensor<128xf32>
    %v7120 = stablehlo.add %v7119, %armeans2b2bt1 : tensor<128xf32>
    %v7121 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7122 = stablehlo.multiply %v7121, %s2b2bt1v : tensor<128xf32>
    %v7123 = stablehlo.add %v7122, %v7120 : tensor<128xf32>
    %v7124 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7125 = stablehlo.multiply %v7124, %v7123 : tensor<128xf32>
    %v7126 = stablehlo.subtract %s2b2bt1, %v7125 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v4367) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v7127 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7128 = stablehlo.multiply %v7127, %s2b2W2 : tensor<128x128x3x3xf32>
    %v7129 = stablehlo.add %v7128, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v7130 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7131 = stablehlo.multiply %v7130, %s2b2W2v : tensor<128x128x3x3xf32>
    %v7132 = stablehlo.add %v7131, %v7129 : tensor<128x128x3x3xf32>
    %v7133 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7134 = stablehlo.multiply %v7133, %v7132 : tensor<128x128x3x3xf32>
    %v7135 = stablehlo.subtract %s2b2W2, %v7134 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v4385) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v7136 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7137 = stablehlo.multiply %v7136, %s2b2g2 : tensor<128xf32>
    %v7138 = stablehlo.add %v7137, %armeans2b2g2 : tensor<128xf32>
    %v7139 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7140 = stablehlo.multiply %v7139, %s2b2g2v : tensor<128xf32>
    %v7141 = stablehlo.add %v7140, %v7138 : tensor<128xf32>
    %v7142 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7143 = stablehlo.multiply %v7142, %v7141 : tensor<128xf32>
    %v7144 = stablehlo.subtract %s2b2g2, %v7143 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v4388) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v7145 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7146 = stablehlo.multiply %v7145, %s2b2bt2 : tensor<128xf32>
    %v7147 = stablehlo.add %v7146, %armeans2b2bt2 : tensor<128xf32>
    %v7148 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7149 = stablehlo.multiply %v7148, %s2b2bt2v : tensor<128xf32>
    %v7150 = stablehlo.add %v7149, %v7147 : tensor<128xf32>
    %v7151 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7152 = stablehlo.multiply %v7151, %v7150 : tensor<128xf32>
    %v7153 = stablehlo.subtract %s2b2bt2, %v7152 : tensor<128xf32>
    %arsums2b2W3 = "stablehlo.all_reduce"(%v4397) ({
    ^bb0(%aras2b2W3: tensor<f32>, %arbs2b2W3: tensor<f32>):
      %aradds2b2W3 = stablehlo.add %aras2b2W3, %arbs2b2W3 : tensor<f32>
      stablehlo.return %aradds2b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b2W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b2W3 = stablehlo.divide %arsums2b2W3, %arns2b2W3 : tensor<512x128x1x1xf32>
    %v7154 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7155 = stablehlo.multiply %v7154, %s2b2W3 : tensor<512x128x1x1xf32>
    %v7156 = stablehlo.add %v7155, %armeans2b2W3 : tensor<512x128x1x1xf32>
    %v7157 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7158 = stablehlo.multiply %v7157, %s2b2W3v : tensor<512x128x1x1xf32>
    %v7159 = stablehlo.add %v7158, %v7156 : tensor<512x128x1x1xf32>
    %v7160 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7161 = stablehlo.multiply %v7160, %v7159 : tensor<512x128x1x1xf32>
    %v7162 = stablehlo.subtract %s2b2W3, %v7161 : tensor<512x128x1x1xf32>
    %arsums2b2g3 = "stablehlo.all_reduce"(%v4415) ({
    ^bb0(%aras2b2g3: tensor<f32>, %arbs2b2g3: tensor<f32>):
      %aradds2b2g3 = stablehlo.add %aras2b2g3, %arbs2b2g3 : tensor<f32>
      stablehlo.return %aradds2b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3 = stablehlo.divide %arsums2b2g3, %arns2b2g3 : tensor<512xf32>
    %v7163 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7164 = stablehlo.multiply %v7163, %s2b2g3 : tensor<512xf32>
    %v7165 = stablehlo.add %v7164, %armeans2b2g3 : tensor<512xf32>
    %v7166 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7167 = stablehlo.multiply %v7166, %s2b2g3v : tensor<512xf32>
    %v7168 = stablehlo.add %v7167, %v7165 : tensor<512xf32>
    %v7169 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7170 = stablehlo.multiply %v7169, %v7168 : tensor<512xf32>
    %v7171 = stablehlo.subtract %s2b2g3, %v7170 : tensor<512xf32>
    %arsums2b2bt3 = "stablehlo.all_reduce"(%v4418) ({
    ^bb0(%aras2b2bt3: tensor<f32>, %arbs2b2bt3: tensor<f32>):
      %aradds2b2bt3 = stablehlo.add %aras2b2bt3, %arbs2b2bt3 : tensor<f32>
      stablehlo.return %aradds2b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2bt3 = stablehlo.divide %arsums2b2bt3, %arns2b2bt3 : tensor<512xf32>
    %v7172 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7173 = stablehlo.multiply %v7172, %s2b2bt3 : tensor<512xf32>
    %v7174 = stablehlo.add %v7173, %armeans2b2bt3 : tensor<512xf32>
    %v7175 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7176 = stablehlo.multiply %v7175, %s2b2bt3v : tensor<512xf32>
    %v7177 = stablehlo.add %v7176, %v7174 : tensor<512xf32>
    %v7178 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7179 = stablehlo.multiply %v7178, %v7177 : tensor<512xf32>
    %v7180 = stablehlo.subtract %s2b2bt3, %v7179 : tensor<512xf32>
    %arsums2b3W1 = "stablehlo.all_reduce"(%v4111) ({
    ^bb0(%aras2b3W1: tensor<f32>, %arbs2b3W1: tensor<f32>):
      %aradds2b3W1 = stablehlo.add %aras2b3W1, %arbs2b3W1 : tensor<f32>
      stablehlo.return %aradds2b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b3W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b3W1 = stablehlo.divide %arsums2b3W1, %arns2b3W1 : tensor<128x512x1x1xf32>
    %v7181 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7182 = stablehlo.multiply %v7181, %s2b3W1 : tensor<128x512x1x1xf32>
    %v7183 = stablehlo.add %v7182, %armeans2b3W1 : tensor<128x512x1x1xf32>
    %v7184 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7185 = stablehlo.multiply %v7184, %s2b3W1v : tensor<128x512x1x1xf32>
    %v7186 = stablehlo.add %v7185, %v7183 : tensor<128x512x1x1xf32>
    %v7187 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7188 = stablehlo.multiply %v7187, %v7186 : tensor<128x512x1x1xf32>
    %v7189 = stablehlo.subtract %s2b3W1, %v7188 : tensor<128x512x1x1xf32>
    %arsums2b3g1 = "stablehlo.all_reduce"(%v4129) ({
    ^bb0(%aras2b3g1: tensor<f32>, %arbs2b3g1: tensor<f32>):
      %aradds2b3g1 = stablehlo.add %aras2b3g1, %arbs2b3g1 : tensor<f32>
      stablehlo.return %aradds2b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1 = stablehlo.divide %arsums2b3g1, %arns2b3g1 : tensor<128xf32>
    %v7190 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7191 = stablehlo.multiply %v7190, %s2b3g1 : tensor<128xf32>
    %v7192 = stablehlo.add %v7191, %armeans2b3g1 : tensor<128xf32>
    %v7193 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7194 = stablehlo.multiply %v7193, %s2b3g1v : tensor<128xf32>
    %v7195 = stablehlo.add %v7194, %v7192 : tensor<128xf32>
    %v7196 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7197 = stablehlo.multiply %v7196, %v7195 : tensor<128xf32>
    %v7198 = stablehlo.subtract %s2b3g1, %v7197 : tensor<128xf32>
    %arsums2b3bt1 = "stablehlo.all_reduce"(%v4132) ({
    ^bb0(%aras2b3bt1: tensor<f32>, %arbs2b3bt1: tensor<f32>):
      %aradds2b3bt1 = stablehlo.add %aras2b3bt1, %arbs2b3bt1 : tensor<f32>
      stablehlo.return %aradds2b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt1 = stablehlo.divide %arsums2b3bt1, %arns2b3bt1 : tensor<128xf32>
    %v7199 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7200 = stablehlo.multiply %v7199, %s2b3bt1 : tensor<128xf32>
    %v7201 = stablehlo.add %v7200, %armeans2b3bt1 : tensor<128xf32>
    %v7202 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7203 = stablehlo.multiply %v7202, %s2b3bt1v : tensor<128xf32>
    %v7204 = stablehlo.add %v7203, %v7201 : tensor<128xf32>
    %v7205 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7206 = stablehlo.multiply %v7205, %v7204 : tensor<128xf32>
    %v7207 = stablehlo.subtract %s2b3bt1, %v7206 : tensor<128xf32>
    %arsums2b3W2 = "stablehlo.all_reduce"(%v4141) ({
    ^bb0(%aras2b3W2: tensor<f32>, %arbs2b3W2: tensor<f32>):
      %aradds2b3W2 = stablehlo.add %aras2b3W2, %arbs2b3W2 : tensor<f32>
      stablehlo.return %aradds2b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b3W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b3W2 = stablehlo.divide %arsums2b3W2, %arns2b3W2 : tensor<128x128x3x3xf32>
    %v7208 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7209 = stablehlo.multiply %v7208, %s2b3W2 : tensor<128x128x3x3xf32>
    %v7210 = stablehlo.add %v7209, %armeans2b3W2 : tensor<128x128x3x3xf32>
    %v7211 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7212 = stablehlo.multiply %v7211, %s2b3W2v : tensor<128x128x3x3xf32>
    %v7213 = stablehlo.add %v7212, %v7210 : tensor<128x128x3x3xf32>
    %v7214 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7215 = stablehlo.multiply %v7214, %v7213 : tensor<128x128x3x3xf32>
    %v7216 = stablehlo.subtract %s2b3W2, %v7215 : tensor<128x128x3x3xf32>
    %arsums2b3g2 = "stablehlo.all_reduce"(%v4159) ({
    ^bb0(%aras2b3g2: tensor<f32>, %arbs2b3g2: tensor<f32>):
      %aradds2b3g2 = stablehlo.add %aras2b3g2, %arbs2b3g2 : tensor<f32>
      stablehlo.return %aradds2b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2 = stablehlo.divide %arsums2b3g2, %arns2b3g2 : tensor<128xf32>
    %v7217 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7218 = stablehlo.multiply %v7217, %s2b3g2 : tensor<128xf32>
    %v7219 = stablehlo.add %v7218, %armeans2b3g2 : tensor<128xf32>
    %v7220 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7221 = stablehlo.multiply %v7220, %s2b3g2v : tensor<128xf32>
    %v7222 = stablehlo.add %v7221, %v7219 : tensor<128xf32>
    %v7223 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7224 = stablehlo.multiply %v7223, %v7222 : tensor<128xf32>
    %v7225 = stablehlo.subtract %s2b3g2, %v7224 : tensor<128xf32>
    %arsums2b3bt2 = "stablehlo.all_reduce"(%v4162) ({
    ^bb0(%aras2b3bt2: tensor<f32>, %arbs2b3bt2: tensor<f32>):
      %aradds2b3bt2 = stablehlo.add %aras2b3bt2, %arbs2b3bt2 : tensor<f32>
      stablehlo.return %aradds2b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt2 = stablehlo.divide %arsums2b3bt2, %arns2b3bt2 : tensor<128xf32>
    %v7226 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7227 = stablehlo.multiply %v7226, %s2b3bt2 : tensor<128xf32>
    %v7228 = stablehlo.add %v7227, %armeans2b3bt2 : tensor<128xf32>
    %v7229 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7230 = stablehlo.multiply %v7229, %s2b3bt2v : tensor<128xf32>
    %v7231 = stablehlo.add %v7230, %v7228 : tensor<128xf32>
    %v7232 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7233 = stablehlo.multiply %v7232, %v7231 : tensor<128xf32>
    %v7234 = stablehlo.subtract %s2b3bt2, %v7233 : tensor<128xf32>
    %arsums2b3W3 = "stablehlo.all_reduce"(%v4171) ({
    ^bb0(%aras2b3W3: tensor<f32>, %arbs2b3W3: tensor<f32>):
      %aradds2b3W3 = stablehlo.add %aras2b3W3, %arbs2b3W3 : tensor<f32>
      stablehlo.return %aradds2b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b3W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b3W3 = stablehlo.divide %arsums2b3W3, %arns2b3W3 : tensor<512x128x1x1xf32>
    %v7235 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7236 = stablehlo.multiply %v7235, %s2b3W3 : tensor<512x128x1x1xf32>
    %v7237 = stablehlo.add %v7236, %armeans2b3W3 : tensor<512x128x1x1xf32>
    %v7238 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7239 = stablehlo.multiply %v7238, %s2b3W3v : tensor<512x128x1x1xf32>
    %v7240 = stablehlo.add %v7239, %v7237 : tensor<512x128x1x1xf32>
    %v7241 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7242 = stablehlo.multiply %v7241, %v7240 : tensor<512x128x1x1xf32>
    %v7243 = stablehlo.subtract %s2b3W3, %v7242 : tensor<512x128x1x1xf32>
    %arsums2b3g3 = "stablehlo.all_reduce"(%v4189) ({
    ^bb0(%aras2b3g3: tensor<f32>, %arbs2b3g3: tensor<f32>):
      %aradds2b3g3 = stablehlo.add %aras2b3g3, %arbs2b3g3 : tensor<f32>
      stablehlo.return %aradds2b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3 = stablehlo.divide %arsums2b3g3, %arns2b3g3 : tensor<512xf32>
    %v7244 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7245 = stablehlo.multiply %v7244, %s2b3g3 : tensor<512xf32>
    %v7246 = stablehlo.add %v7245, %armeans2b3g3 : tensor<512xf32>
    %v7247 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7248 = stablehlo.multiply %v7247, %s2b3g3v : tensor<512xf32>
    %v7249 = stablehlo.add %v7248, %v7246 : tensor<512xf32>
    %v7250 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7251 = stablehlo.multiply %v7250, %v7249 : tensor<512xf32>
    %v7252 = stablehlo.subtract %s2b3g3, %v7251 : tensor<512xf32>
    %arsums2b3bt3 = "stablehlo.all_reduce"(%v4192) ({
    ^bb0(%aras2b3bt3: tensor<f32>, %arbs2b3bt3: tensor<f32>):
      %aradds2b3bt3 = stablehlo.add %aras2b3bt3, %arbs2b3bt3 : tensor<f32>
      stablehlo.return %aradds2b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3bt3 = stablehlo.divide %arsums2b3bt3, %arns2b3bt3 : tensor<512xf32>
    %v7253 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7254 = stablehlo.multiply %v7253, %s2b3bt3 : tensor<512xf32>
    %v7255 = stablehlo.add %v7254, %armeans2b3bt3 : tensor<512xf32>
    %v7256 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7257 = stablehlo.multiply %v7256, %s2b3bt3v : tensor<512xf32>
    %v7258 = stablehlo.add %v7257, %v7255 : tensor<512xf32>
    %v7259 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7260 = stablehlo.multiply %v7259, %v7258 : tensor<512xf32>
    %v7261 = stablehlo.subtract %s2b3bt3, %v7260 : tensor<512xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v3851) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x512x1x1xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x512x1x1xf32>
    %v7262 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7263 = stablehlo.multiply %v7262, %s3b0W1 : tensor<256x512x1x1xf32>
    %v7264 = stablehlo.add %v7263, %armeans3b0W1 : tensor<256x512x1x1xf32>
    %v7265 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7266 = stablehlo.multiply %v7265, %s3b0W1v : tensor<256x512x1x1xf32>
    %v7267 = stablehlo.add %v7266, %v7264 : tensor<256x512x1x1xf32>
    %v7268 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7269 = stablehlo.multiply %v7268, %v7267 : tensor<256x512x1x1xf32>
    %v7270 = stablehlo.subtract %s3b0W1, %v7269 : tensor<256x512x1x1xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v3869) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v7271 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7272 = stablehlo.multiply %v7271, %s3b0g1 : tensor<256xf32>
    %v7273 = stablehlo.add %v7272, %armeans3b0g1 : tensor<256xf32>
    %v7274 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7275 = stablehlo.multiply %v7274, %s3b0g1v : tensor<256xf32>
    %v7276 = stablehlo.add %v7275, %v7273 : tensor<256xf32>
    %v7277 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7278 = stablehlo.multiply %v7277, %v7276 : tensor<256xf32>
    %v7279 = stablehlo.subtract %s3b0g1, %v7278 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v3872) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v7280 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7281 = stablehlo.multiply %v7280, %s3b0bt1 : tensor<256xf32>
    %v7282 = stablehlo.add %v7281, %armeans3b0bt1 : tensor<256xf32>
    %v7283 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7284 = stablehlo.multiply %v7283, %s3b0bt1v : tensor<256xf32>
    %v7285 = stablehlo.add %v7284, %v7282 : tensor<256xf32>
    %v7286 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7287 = stablehlo.multiply %v7286, %v7285 : tensor<256xf32>
    %v7288 = stablehlo.subtract %s3b0bt1, %v7287 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v3883) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v7289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7290 = stablehlo.multiply %v7289, %s3b0W2 : tensor<256x256x3x3xf32>
    %v7291 = stablehlo.add %v7290, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v7292 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7293 = stablehlo.multiply %v7292, %s3b0W2v : tensor<256x256x3x3xf32>
    %v7294 = stablehlo.add %v7293, %v7291 : tensor<256x256x3x3xf32>
    %v7295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7296 = stablehlo.multiply %v7295, %v7294 : tensor<256x256x3x3xf32>
    %v7297 = stablehlo.subtract %s3b0W2, %v7296 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v3901) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v7298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7299 = stablehlo.multiply %v7298, %s3b0g2 : tensor<256xf32>
    %v7300 = stablehlo.add %v7299, %armeans3b0g2 : tensor<256xf32>
    %v7301 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7302 = stablehlo.multiply %v7301, %s3b0g2v : tensor<256xf32>
    %v7303 = stablehlo.add %v7302, %v7300 : tensor<256xf32>
    %v7304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7305 = stablehlo.multiply %v7304, %v7303 : tensor<256xf32>
    %v7306 = stablehlo.subtract %s3b0g2, %v7305 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v3904) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v7307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7308 = stablehlo.multiply %v7307, %s3b0bt2 : tensor<256xf32>
    %v7309 = stablehlo.add %v7308, %armeans3b0bt2 : tensor<256xf32>
    %v7310 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7311 = stablehlo.multiply %v7310, %s3b0bt2v : tensor<256xf32>
    %v7312 = stablehlo.add %v7311, %v7309 : tensor<256xf32>
    %v7313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7314 = stablehlo.multiply %v7313, %v7312 : tensor<256xf32>
    %v7315 = stablehlo.subtract %s3b0bt2, %v7314 : tensor<256xf32>
    %arsums3b0W3 = "stablehlo.all_reduce"(%v3913) ({
    ^bb0(%aras3b0W3: tensor<f32>, %arbs3b0W3: tensor<f32>):
      %aradds3b0W3 = stablehlo.add %aras3b0W3, %arbs3b0W3 : tensor<f32>
      stablehlo.return %aradds3b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b0W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b0W3 = stablehlo.divide %arsums3b0W3, %arns3b0W3 : tensor<1024x256x1x1xf32>
    %v7316 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7317 = stablehlo.multiply %v7316, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v7318 = stablehlo.add %v7317, %armeans3b0W3 : tensor<1024x256x1x1xf32>
    %v7319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7320 = stablehlo.multiply %v7319, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v7321 = stablehlo.add %v7320, %v7318 : tensor<1024x256x1x1xf32>
    %v7322 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7323 = stablehlo.multiply %v7322, %v7321 : tensor<1024x256x1x1xf32>
    %v7324 = stablehlo.subtract %s3b0W3, %v7323 : tensor<1024x256x1x1xf32>
    %arsums3b0g3 = "stablehlo.all_reduce"(%v3931) ({
    ^bb0(%aras3b0g3: tensor<f32>, %arbs3b0g3: tensor<f32>):
      %aradds3b0g3 = stablehlo.add %aras3b0g3, %arbs3b0g3 : tensor<f32>
      stablehlo.return %aradds3b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3 = stablehlo.divide %arsums3b0g3, %arns3b0g3 : tensor<1024xf32>
    %v7325 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7326 = stablehlo.multiply %v7325, %s3b0g3 : tensor<1024xf32>
    %v7327 = stablehlo.add %v7326, %armeans3b0g3 : tensor<1024xf32>
    %v7328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7329 = stablehlo.multiply %v7328, %s3b0g3v : tensor<1024xf32>
    %v7330 = stablehlo.add %v7329, %v7327 : tensor<1024xf32>
    %v7331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7332 = stablehlo.multiply %v7331, %v7330 : tensor<1024xf32>
    %v7333 = stablehlo.subtract %s3b0g3, %v7332 : tensor<1024xf32>
    %arsums3b0bt3 = "stablehlo.all_reduce"(%v3934) ({
    ^bb0(%aras3b0bt3: tensor<f32>, %arbs3b0bt3: tensor<f32>):
      %aradds3b0bt3 = stablehlo.add %aras3b0bt3, %arbs3b0bt3 : tensor<f32>
      stablehlo.return %aradds3b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0bt3 = stablehlo.divide %arsums3b0bt3, %arns3b0bt3 : tensor<1024xf32>
    %v7334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7335 = stablehlo.multiply %v7334, %s3b0bt3 : tensor<1024xf32>
    %v7336 = stablehlo.add %v7335, %armeans3b0bt3 : tensor<1024xf32>
    %v7337 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7338 = stablehlo.multiply %v7337, %s3b0bt3v : tensor<1024xf32>
    %v7339 = stablehlo.add %v7338, %v7336 : tensor<1024xf32>
    %v7340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7341 = stablehlo.multiply %v7340, %v7339 : tensor<1024xf32>
    %v7342 = stablehlo.subtract %s3b0bt3, %v7341 : tensor<1024xf32>
    %arsums3b0Wp = "stablehlo.all_reduce"(%v3945) ({
    ^bb0(%aras3b0Wp: tensor<f32>, %arbs3b0Wp: tensor<f32>):
      %aradds3b0Wp = stablehlo.add %aras3b0Wp, %arbs3b0Wp : tensor<f32>
      stablehlo.return %aradds3b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %arns3b0Wp = stablehlo.constant dense<4.0> : tensor<1024x512x1x1xf32>
    %armeans3b0Wp = stablehlo.divide %arsums3b0Wp, %arns3b0Wp : tensor<1024x512x1x1xf32>
    %v7343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7344 = stablehlo.multiply %v7343, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v7345 = stablehlo.add %v7344, %armeans3b0Wp : tensor<1024x512x1x1xf32>
    %v7346 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7347 = stablehlo.multiply %v7346, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v7348 = stablehlo.add %v7347, %v7345 : tensor<1024x512x1x1xf32>
    %v7349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7350 = stablehlo.multiply %v7349, %v7348 : tensor<1024x512x1x1xf32>
    %v7351 = stablehlo.subtract %s3b0Wp, %v7350 : tensor<1024x512x1x1xf32>
    %arsums3b0gp = "stablehlo.all_reduce"(%v3963) ({
    ^bb0(%aras3b0gp: tensor<f32>, %arbs3b0gp: tensor<f32>):
      %aradds3b0gp = stablehlo.add %aras3b0gp, %arbs3b0gp : tensor<f32>
      stablehlo.return %aradds3b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gp = stablehlo.divide %arsums3b0gp, %arns3b0gp : tensor<1024xf32>
    %v7352 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7353 = stablehlo.multiply %v7352, %s3b0gp : tensor<1024xf32>
    %v7354 = stablehlo.add %v7353, %armeans3b0gp : tensor<1024xf32>
    %v7355 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7356 = stablehlo.multiply %v7355, %s3b0gpv : tensor<1024xf32>
    %v7357 = stablehlo.add %v7356, %v7354 : tensor<1024xf32>
    %v7358 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7359 = stablehlo.multiply %v7358, %v7357 : tensor<1024xf32>
    %v7360 = stablehlo.subtract %s3b0gp, %v7359 : tensor<1024xf32>
    %arsums3b0btp = "stablehlo.all_reduce"(%v3966) ({
    ^bb0(%aras3b0btp: tensor<f32>, %arbs3b0btp: tensor<f32>):
      %aradds3b0btp = stablehlo.add %aras3b0btp, %arbs3b0btp : tensor<f32>
      stablehlo.return %aradds3b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0btp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0btp = stablehlo.divide %arsums3b0btp, %arns3b0btp : tensor<1024xf32>
    %v7361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7362 = stablehlo.multiply %v7361, %s3b0btp : tensor<1024xf32>
    %v7363 = stablehlo.add %v7362, %armeans3b0btp : tensor<1024xf32>
    %v7364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7365 = stablehlo.multiply %v7364, %s3b0btpv : tensor<1024xf32>
    %v7366 = stablehlo.add %v7365, %v7363 : tensor<1024xf32>
    %v7367 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7368 = stablehlo.multiply %v7367, %v7366 : tensor<1024xf32>
    %v7369 = stablehlo.subtract %s3b0btp, %v7368 : tensor<1024xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v3583) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x1024x1x1xf32>
    %v7370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7371 = stablehlo.multiply %v7370, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v7372 = stablehlo.add %v7371, %armeans3b1W1 : tensor<256x1024x1x1xf32>
    %v7373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7374 = stablehlo.multiply %v7373, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v7375 = stablehlo.add %v7374, %v7372 : tensor<256x1024x1x1xf32>
    %v7376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7377 = stablehlo.multiply %v7376, %v7375 : tensor<256x1024x1x1xf32>
    %v7378 = stablehlo.subtract %s3b1W1, %v7377 : tensor<256x1024x1x1xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v3601) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v7379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7380 = stablehlo.multiply %v7379, %s3b1g1 : tensor<256xf32>
    %v7381 = stablehlo.add %v7380, %armeans3b1g1 : tensor<256xf32>
    %v7382 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7383 = stablehlo.multiply %v7382, %s3b1g1v : tensor<256xf32>
    %v7384 = stablehlo.add %v7383, %v7381 : tensor<256xf32>
    %v7385 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7386 = stablehlo.multiply %v7385, %v7384 : tensor<256xf32>
    %v7387 = stablehlo.subtract %s3b1g1, %v7386 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v3604) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v7388 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7389 = stablehlo.multiply %v7388, %s3b1bt1 : tensor<256xf32>
    %v7390 = stablehlo.add %v7389, %armeans3b1bt1 : tensor<256xf32>
    %v7391 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7392 = stablehlo.multiply %v7391, %s3b1bt1v : tensor<256xf32>
    %v7393 = stablehlo.add %v7392, %v7390 : tensor<256xf32>
    %v7394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7395 = stablehlo.multiply %v7394, %v7393 : tensor<256xf32>
    %v7396 = stablehlo.subtract %s3b1bt1, %v7395 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v3613) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v7397 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7398 = stablehlo.multiply %v7397, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7399 = stablehlo.add %v7398, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7401 = stablehlo.multiply %v7400, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7402 = stablehlo.add %v7401, %v7399 : tensor<256x256x3x3xf32>
    %v7403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7404 = stablehlo.multiply %v7403, %v7402 : tensor<256x256x3x3xf32>
    %v7405 = stablehlo.subtract %s3b1W2, %v7404 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v3631) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7407 = stablehlo.multiply %v7406, %s3b1g2 : tensor<256xf32>
    %v7408 = stablehlo.add %v7407, %armeans3b1g2 : tensor<256xf32>
    %v7409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7410 = stablehlo.multiply %v7409, %s3b1g2v : tensor<256xf32>
    %v7411 = stablehlo.add %v7410, %v7408 : tensor<256xf32>
    %v7412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7413 = stablehlo.multiply %v7412, %v7411 : tensor<256xf32>
    %v7414 = stablehlo.subtract %s3b1g2, %v7413 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v3634) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7416 = stablehlo.multiply %v7415, %s3b1bt2 : tensor<256xf32>
    %v7417 = stablehlo.add %v7416, %armeans3b1bt2 : tensor<256xf32>
    %v7418 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7419 = stablehlo.multiply %v7418, %s3b1bt2v : tensor<256xf32>
    %v7420 = stablehlo.add %v7419, %v7417 : tensor<256xf32>
    %v7421 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7422 = stablehlo.multiply %v7421, %v7420 : tensor<256xf32>
    %v7423 = stablehlo.subtract %s3b1bt2, %v7422 : tensor<256xf32>
    %arsums3b1W3 = "stablehlo.all_reduce"(%v3643) ({
    ^bb0(%aras3b1W3: tensor<f32>, %arbs3b1W3: tensor<f32>):
      %aradds3b1W3 = stablehlo.add %aras3b1W3, %arbs3b1W3 : tensor<f32>
      stablehlo.return %aradds3b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b1W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b1W3 = stablehlo.divide %arsums3b1W3, %arns3b1W3 : tensor<1024x256x1x1xf32>
    %v7424 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7425 = stablehlo.multiply %v7424, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v7426 = stablehlo.add %v7425, %armeans3b1W3 : tensor<1024x256x1x1xf32>
    %v7427 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7428 = stablehlo.multiply %v7427, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v7429 = stablehlo.add %v7428, %v7426 : tensor<1024x256x1x1xf32>
    %v7430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7431 = stablehlo.multiply %v7430, %v7429 : tensor<1024x256x1x1xf32>
    %v7432 = stablehlo.subtract %s3b1W3, %v7431 : tensor<1024x256x1x1xf32>
    %arsums3b1g3 = "stablehlo.all_reduce"(%v3661) ({
    ^bb0(%aras3b1g3: tensor<f32>, %arbs3b1g3: tensor<f32>):
      %aradds3b1g3 = stablehlo.add %aras3b1g3, %arbs3b1g3 : tensor<f32>
      stablehlo.return %aradds3b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3 = stablehlo.divide %arsums3b1g3, %arns3b1g3 : tensor<1024xf32>
    %v7433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7434 = stablehlo.multiply %v7433, %s3b1g3 : tensor<1024xf32>
    %v7435 = stablehlo.add %v7434, %armeans3b1g3 : tensor<1024xf32>
    %v7436 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7437 = stablehlo.multiply %v7436, %s3b1g3v : tensor<1024xf32>
    %v7438 = stablehlo.add %v7437, %v7435 : tensor<1024xf32>
    %v7439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7440 = stablehlo.multiply %v7439, %v7438 : tensor<1024xf32>
    %v7441 = stablehlo.subtract %s3b1g3, %v7440 : tensor<1024xf32>
    %arsums3b1bt3 = "stablehlo.all_reduce"(%v3664) ({
    ^bb0(%aras3b1bt3: tensor<f32>, %arbs3b1bt3: tensor<f32>):
      %aradds3b1bt3 = stablehlo.add %aras3b1bt3, %arbs3b1bt3 : tensor<f32>
      stablehlo.return %aradds3b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1bt3 = stablehlo.divide %arsums3b1bt3, %arns3b1bt3 : tensor<1024xf32>
    %v7442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7443 = stablehlo.multiply %v7442, %s3b1bt3 : tensor<1024xf32>
    %v7444 = stablehlo.add %v7443, %armeans3b1bt3 : tensor<1024xf32>
    %v7445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7446 = stablehlo.multiply %v7445, %s3b1bt3v : tensor<1024xf32>
    %v7447 = stablehlo.add %v7446, %v7444 : tensor<1024xf32>
    %v7448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7449 = stablehlo.multiply %v7448, %v7447 : tensor<1024xf32>
    %v7450 = stablehlo.subtract %s3b1bt3, %v7449 : tensor<1024xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v3357) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x1024x1x1xf32>
    %v7451 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7452 = stablehlo.multiply %v7451, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v7453 = stablehlo.add %v7452, %armeans3b2W1 : tensor<256x1024x1x1xf32>
    %v7454 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7455 = stablehlo.multiply %v7454, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v7456 = stablehlo.add %v7455, %v7453 : tensor<256x1024x1x1xf32>
    %v7457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7458 = stablehlo.multiply %v7457, %v7456 : tensor<256x1024x1x1xf32>
    %v7459 = stablehlo.subtract %s3b2W1, %v7458 : tensor<256x1024x1x1xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v3375) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7461 = stablehlo.multiply %v7460, %s3b2g1 : tensor<256xf32>
    %v7462 = stablehlo.add %v7461, %armeans3b2g1 : tensor<256xf32>
    %v7463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7464 = stablehlo.multiply %v7463, %s3b2g1v : tensor<256xf32>
    %v7465 = stablehlo.add %v7464, %v7462 : tensor<256xf32>
    %v7466 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7467 = stablehlo.multiply %v7466, %v7465 : tensor<256xf32>
    %v7468 = stablehlo.subtract %s3b2g1, %v7467 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v3378) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7470 = stablehlo.multiply %v7469, %s3b2bt1 : tensor<256xf32>
    %v7471 = stablehlo.add %v7470, %armeans3b2bt1 : tensor<256xf32>
    %v7472 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7473 = stablehlo.multiply %v7472, %s3b2bt1v : tensor<256xf32>
    %v7474 = stablehlo.add %v7473, %v7471 : tensor<256xf32>
    %v7475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7476 = stablehlo.multiply %v7475, %v7474 : tensor<256xf32>
    %v7477 = stablehlo.subtract %s3b2bt1, %v7476 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v3387) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7478 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7479 = stablehlo.multiply %v7478, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7480 = stablehlo.add %v7479, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7481 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7482 = stablehlo.multiply %v7481, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7483 = stablehlo.add %v7482, %v7480 : tensor<256x256x3x3xf32>
    %v7484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7485 = stablehlo.multiply %v7484, %v7483 : tensor<256x256x3x3xf32>
    %v7486 = stablehlo.subtract %s3b2W2, %v7485 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v3405) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7488 = stablehlo.multiply %v7487, %s3b2g2 : tensor<256xf32>
    %v7489 = stablehlo.add %v7488, %armeans3b2g2 : tensor<256xf32>
    %v7490 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7491 = stablehlo.multiply %v7490, %s3b2g2v : tensor<256xf32>
    %v7492 = stablehlo.add %v7491, %v7489 : tensor<256xf32>
    %v7493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7494 = stablehlo.multiply %v7493, %v7492 : tensor<256xf32>
    %v7495 = stablehlo.subtract %s3b2g2, %v7494 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v3408) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7497 = stablehlo.multiply %v7496, %s3b2bt2 : tensor<256xf32>
    %v7498 = stablehlo.add %v7497, %armeans3b2bt2 : tensor<256xf32>
    %v7499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7500 = stablehlo.multiply %v7499, %s3b2bt2v : tensor<256xf32>
    %v7501 = stablehlo.add %v7500, %v7498 : tensor<256xf32>
    %v7502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7503 = stablehlo.multiply %v7502, %v7501 : tensor<256xf32>
    %v7504 = stablehlo.subtract %s3b2bt2, %v7503 : tensor<256xf32>
    %arsums3b2W3 = "stablehlo.all_reduce"(%v3417) ({
    ^bb0(%aras3b2W3: tensor<f32>, %arbs3b2W3: tensor<f32>):
      %aradds3b2W3 = stablehlo.add %aras3b2W3, %arbs3b2W3 : tensor<f32>
      stablehlo.return %aradds3b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b2W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b2W3 = stablehlo.divide %arsums3b2W3, %arns3b2W3 : tensor<1024x256x1x1xf32>
    %v7505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7506 = stablehlo.multiply %v7505, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7507 = stablehlo.add %v7506, %armeans3b2W3 : tensor<1024x256x1x1xf32>
    %v7508 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7509 = stablehlo.multiply %v7508, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7510 = stablehlo.add %v7509, %v7507 : tensor<1024x256x1x1xf32>
    %v7511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7512 = stablehlo.multiply %v7511, %v7510 : tensor<1024x256x1x1xf32>
    %v7513 = stablehlo.subtract %s3b2W3, %v7512 : tensor<1024x256x1x1xf32>
    %arsums3b2g3 = "stablehlo.all_reduce"(%v3435) ({
    ^bb0(%aras3b2g3: tensor<f32>, %arbs3b2g3: tensor<f32>):
      %aradds3b2g3 = stablehlo.add %aras3b2g3, %arbs3b2g3 : tensor<f32>
      stablehlo.return %aradds3b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3 = stablehlo.divide %arsums3b2g3, %arns3b2g3 : tensor<1024xf32>
    %v7514 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7515 = stablehlo.multiply %v7514, %s3b2g3 : tensor<1024xf32>
    %v7516 = stablehlo.add %v7515, %armeans3b2g3 : tensor<1024xf32>
    %v7517 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7518 = stablehlo.multiply %v7517, %s3b2g3v : tensor<1024xf32>
    %v7519 = stablehlo.add %v7518, %v7516 : tensor<1024xf32>
    %v7520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7521 = stablehlo.multiply %v7520, %v7519 : tensor<1024xf32>
    %v7522 = stablehlo.subtract %s3b2g3, %v7521 : tensor<1024xf32>
    %arsums3b2bt3 = "stablehlo.all_reduce"(%v3438) ({
    ^bb0(%aras3b2bt3: tensor<f32>, %arbs3b2bt3: tensor<f32>):
      %aradds3b2bt3 = stablehlo.add %aras3b2bt3, %arbs3b2bt3 : tensor<f32>
      stablehlo.return %aradds3b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2bt3 = stablehlo.divide %arsums3b2bt3, %arns3b2bt3 : tensor<1024xf32>
    %v7523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7524 = stablehlo.multiply %v7523, %s3b2bt3 : tensor<1024xf32>
    %v7525 = stablehlo.add %v7524, %armeans3b2bt3 : tensor<1024xf32>
    %v7526 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7527 = stablehlo.multiply %v7526, %s3b2bt3v : tensor<1024xf32>
    %v7528 = stablehlo.add %v7527, %v7525 : tensor<1024xf32>
    %v7529 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7530 = stablehlo.multiply %v7529, %v7528 : tensor<1024xf32>
    %v7531 = stablehlo.subtract %s3b2bt3, %v7530 : tensor<1024xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v3131) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x1024x1x1xf32>
    %v7532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7533 = stablehlo.multiply %v7532, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7534 = stablehlo.add %v7533, %armeans3b3W1 : tensor<256x1024x1x1xf32>
    %v7535 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7536 = stablehlo.multiply %v7535, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7537 = stablehlo.add %v7536, %v7534 : tensor<256x1024x1x1xf32>
    %v7538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7539 = stablehlo.multiply %v7538, %v7537 : tensor<256x1024x1x1xf32>
    %v7540 = stablehlo.subtract %s3b3W1, %v7539 : tensor<256x1024x1x1xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v3149) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7542 = stablehlo.multiply %v7541, %s3b3g1 : tensor<256xf32>
    %v7543 = stablehlo.add %v7542, %armeans3b3g1 : tensor<256xf32>
    %v7544 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7545 = stablehlo.multiply %v7544, %s3b3g1v : tensor<256xf32>
    %v7546 = stablehlo.add %v7545, %v7543 : tensor<256xf32>
    %v7547 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7548 = stablehlo.multiply %v7547, %v7546 : tensor<256xf32>
    %v7549 = stablehlo.subtract %s3b3g1, %v7548 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v3152) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7550 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7551 = stablehlo.multiply %v7550, %s3b3bt1 : tensor<256xf32>
    %v7552 = stablehlo.add %v7551, %armeans3b3bt1 : tensor<256xf32>
    %v7553 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7554 = stablehlo.multiply %v7553, %s3b3bt1v : tensor<256xf32>
    %v7555 = stablehlo.add %v7554, %v7552 : tensor<256xf32>
    %v7556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7557 = stablehlo.multiply %v7556, %v7555 : tensor<256xf32>
    %v7558 = stablehlo.subtract %s3b3bt1, %v7557 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v3161) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7559 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7560 = stablehlo.multiply %v7559, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7561 = stablehlo.add %v7560, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7562 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7563 = stablehlo.multiply %v7562, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7564 = stablehlo.add %v7563, %v7561 : tensor<256x256x3x3xf32>
    %v7565 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7566 = stablehlo.multiply %v7565, %v7564 : tensor<256x256x3x3xf32>
    %v7567 = stablehlo.subtract %s3b3W2, %v7566 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v3179) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7569 = stablehlo.multiply %v7568, %s3b3g2 : tensor<256xf32>
    %v7570 = stablehlo.add %v7569, %armeans3b3g2 : tensor<256xf32>
    %v7571 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7572 = stablehlo.multiply %v7571, %s3b3g2v : tensor<256xf32>
    %v7573 = stablehlo.add %v7572, %v7570 : tensor<256xf32>
    %v7574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7575 = stablehlo.multiply %v7574, %v7573 : tensor<256xf32>
    %v7576 = stablehlo.subtract %s3b3g2, %v7575 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v3182) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7577 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7578 = stablehlo.multiply %v7577, %s3b3bt2 : tensor<256xf32>
    %v7579 = stablehlo.add %v7578, %armeans3b3bt2 : tensor<256xf32>
    %v7580 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7581 = stablehlo.multiply %v7580, %s3b3bt2v : tensor<256xf32>
    %v7582 = stablehlo.add %v7581, %v7579 : tensor<256xf32>
    %v7583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7584 = stablehlo.multiply %v7583, %v7582 : tensor<256xf32>
    %v7585 = stablehlo.subtract %s3b3bt2, %v7584 : tensor<256xf32>
    %arsums3b3W3 = "stablehlo.all_reduce"(%v3191) ({
    ^bb0(%aras3b3W3: tensor<f32>, %arbs3b3W3: tensor<f32>):
      %aradds3b3W3 = stablehlo.add %aras3b3W3, %arbs3b3W3 : tensor<f32>
      stablehlo.return %aradds3b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b3W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b3W3 = stablehlo.divide %arsums3b3W3, %arns3b3W3 : tensor<1024x256x1x1xf32>
    %v7586 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7587 = stablehlo.multiply %v7586, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v7588 = stablehlo.add %v7587, %armeans3b3W3 : tensor<1024x256x1x1xf32>
    %v7589 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7590 = stablehlo.multiply %v7589, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v7591 = stablehlo.add %v7590, %v7588 : tensor<1024x256x1x1xf32>
    %v7592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7593 = stablehlo.multiply %v7592, %v7591 : tensor<1024x256x1x1xf32>
    %v7594 = stablehlo.subtract %s3b3W3, %v7593 : tensor<1024x256x1x1xf32>
    %arsums3b3g3 = "stablehlo.all_reduce"(%v3209) ({
    ^bb0(%aras3b3g3: tensor<f32>, %arbs3b3g3: tensor<f32>):
      %aradds3b3g3 = stablehlo.add %aras3b3g3, %arbs3b3g3 : tensor<f32>
      stablehlo.return %aradds3b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3 = stablehlo.divide %arsums3b3g3, %arns3b3g3 : tensor<1024xf32>
    %v7595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7596 = stablehlo.multiply %v7595, %s3b3g3 : tensor<1024xf32>
    %v7597 = stablehlo.add %v7596, %armeans3b3g3 : tensor<1024xf32>
    %v7598 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7599 = stablehlo.multiply %v7598, %s3b3g3v : tensor<1024xf32>
    %v7600 = stablehlo.add %v7599, %v7597 : tensor<1024xf32>
    %v7601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7602 = stablehlo.multiply %v7601, %v7600 : tensor<1024xf32>
    %v7603 = stablehlo.subtract %s3b3g3, %v7602 : tensor<1024xf32>
    %arsums3b3bt3 = "stablehlo.all_reduce"(%v3212) ({
    ^bb0(%aras3b3bt3: tensor<f32>, %arbs3b3bt3: tensor<f32>):
      %aradds3b3bt3 = stablehlo.add %aras3b3bt3, %arbs3b3bt3 : tensor<f32>
      stablehlo.return %aradds3b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3bt3 = stablehlo.divide %arsums3b3bt3, %arns3b3bt3 : tensor<1024xf32>
    %v7604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7605 = stablehlo.multiply %v7604, %s3b3bt3 : tensor<1024xf32>
    %v7606 = stablehlo.add %v7605, %armeans3b3bt3 : tensor<1024xf32>
    %v7607 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7608 = stablehlo.multiply %v7607, %s3b3bt3v : tensor<1024xf32>
    %v7609 = stablehlo.add %v7608, %v7606 : tensor<1024xf32>
    %v7610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7611 = stablehlo.multiply %v7610, %v7609 : tensor<1024xf32>
    %v7612 = stablehlo.subtract %s3b3bt3, %v7611 : tensor<1024xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2905) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x1024x1x1xf32>
    %v7613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7614 = stablehlo.multiply %v7613, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v7615 = stablehlo.add %v7614, %armeans3b4W1 : tensor<256x1024x1x1xf32>
    %v7616 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7617 = stablehlo.multiply %v7616, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v7618 = stablehlo.add %v7617, %v7615 : tensor<256x1024x1x1xf32>
    %v7619 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7620 = stablehlo.multiply %v7619, %v7618 : tensor<256x1024x1x1xf32>
    %v7621 = stablehlo.subtract %s3b4W1, %v7620 : tensor<256x1024x1x1xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2923) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7622 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7623 = stablehlo.multiply %v7622, %s3b4g1 : tensor<256xf32>
    %v7624 = stablehlo.add %v7623, %armeans3b4g1 : tensor<256xf32>
    %v7625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7626 = stablehlo.multiply %v7625, %s3b4g1v : tensor<256xf32>
    %v7627 = stablehlo.add %v7626, %v7624 : tensor<256xf32>
    %v7628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7629 = stablehlo.multiply %v7628, %v7627 : tensor<256xf32>
    %v7630 = stablehlo.subtract %s3b4g1, %v7629 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2926) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7632 = stablehlo.multiply %v7631, %s3b4bt1 : tensor<256xf32>
    %v7633 = stablehlo.add %v7632, %armeans3b4bt1 : tensor<256xf32>
    %v7634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7635 = stablehlo.multiply %v7634, %s3b4bt1v : tensor<256xf32>
    %v7636 = stablehlo.add %v7635, %v7633 : tensor<256xf32>
    %v7637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7638 = stablehlo.multiply %v7637, %v7636 : tensor<256xf32>
    %v7639 = stablehlo.subtract %s3b4bt1, %v7638 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2935) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7641 = stablehlo.multiply %v7640, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7642 = stablehlo.add %v7641, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7643 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7644 = stablehlo.multiply %v7643, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7645 = stablehlo.add %v7644, %v7642 : tensor<256x256x3x3xf32>
    %v7646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7647 = stablehlo.multiply %v7646, %v7645 : tensor<256x256x3x3xf32>
    %v7648 = stablehlo.subtract %s3b4W2, %v7647 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2953) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7650 = stablehlo.multiply %v7649, %s3b4g2 : tensor<256xf32>
    %v7651 = stablehlo.add %v7650, %armeans3b4g2 : tensor<256xf32>
    %v7652 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7653 = stablehlo.multiply %v7652, %s3b4g2v : tensor<256xf32>
    %v7654 = stablehlo.add %v7653, %v7651 : tensor<256xf32>
    %v7655 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7656 = stablehlo.multiply %v7655, %v7654 : tensor<256xf32>
    %v7657 = stablehlo.subtract %s3b4g2, %v7656 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2956) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7658 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7659 = stablehlo.multiply %v7658, %s3b4bt2 : tensor<256xf32>
    %v7660 = stablehlo.add %v7659, %armeans3b4bt2 : tensor<256xf32>
    %v7661 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7662 = stablehlo.multiply %v7661, %s3b4bt2v : tensor<256xf32>
    %v7663 = stablehlo.add %v7662, %v7660 : tensor<256xf32>
    %v7664 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7665 = stablehlo.multiply %v7664, %v7663 : tensor<256xf32>
    %v7666 = stablehlo.subtract %s3b4bt2, %v7665 : tensor<256xf32>
    %arsums3b4W3 = "stablehlo.all_reduce"(%v2965) ({
    ^bb0(%aras3b4W3: tensor<f32>, %arbs3b4W3: tensor<f32>):
      %aradds3b4W3 = stablehlo.add %aras3b4W3, %arbs3b4W3 : tensor<f32>
      stablehlo.return %aradds3b4W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b4W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b4W3 = stablehlo.divide %arsums3b4W3, %arns3b4W3 : tensor<1024x256x1x1xf32>
    %v7667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7668 = stablehlo.multiply %v7667, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v7669 = stablehlo.add %v7668, %armeans3b4W3 : tensor<1024x256x1x1xf32>
    %v7670 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7671 = stablehlo.multiply %v7670, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v7672 = stablehlo.add %v7671, %v7669 : tensor<1024x256x1x1xf32>
    %v7673 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7674 = stablehlo.multiply %v7673, %v7672 : tensor<1024x256x1x1xf32>
    %v7675 = stablehlo.subtract %s3b4W3, %v7674 : tensor<1024x256x1x1xf32>
    %arsums3b4g3 = "stablehlo.all_reduce"(%v2983) ({
    ^bb0(%aras3b4g3: tensor<f32>, %arbs3b4g3: tensor<f32>):
      %aradds3b4g3 = stablehlo.add %aras3b4g3, %arbs3b4g3 : tensor<f32>
      stablehlo.return %aradds3b4g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3 = stablehlo.divide %arsums3b4g3, %arns3b4g3 : tensor<1024xf32>
    %v7676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7677 = stablehlo.multiply %v7676, %s3b4g3 : tensor<1024xf32>
    %v7678 = stablehlo.add %v7677, %armeans3b4g3 : tensor<1024xf32>
    %v7679 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7680 = stablehlo.multiply %v7679, %s3b4g3v : tensor<1024xf32>
    %v7681 = stablehlo.add %v7680, %v7678 : tensor<1024xf32>
    %v7682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7683 = stablehlo.multiply %v7682, %v7681 : tensor<1024xf32>
    %v7684 = stablehlo.subtract %s3b4g3, %v7683 : tensor<1024xf32>
    %arsums3b4bt3 = "stablehlo.all_reduce"(%v2986) ({
    ^bb0(%aras3b4bt3: tensor<f32>, %arbs3b4bt3: tensor<f32>):
      %aradds3b4bt3 = stablehlo.add %aras3b4bt3, %arbs3b4bt3 : tensor<f32>
      stablehlo.return %aradds3b4bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4bt3 = stablehlo.divide %arsums3b4bt3, %arns3b4bt3 : tensor<1024xf32>
    %v7685 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7686 = stablehlo.multiply %v7685, %s3b4bt3 : tensor<1024xf32>
    %v7687 = stablehlo.add %v7686, %armeans3b4bt3 : tensor<1024xf32>
    %v7688 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7689 = stablehlo.multiply %v7688, %s3b4bt3v : tensor<1024xf32>
    %v7690 = stablehlo.add %v7689, %v7687 : tensor<1024xf32>
    %v7691 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7692 = stablehlo.multiply %v7691, %v7690 : tensor<1024xf32>
    %v7693 = stablehlo.subtract %s3b4bt3, %v7692 : tensor<1024xf32>
    %arsums3b5W1 = "stablehlo.all_reduce"(%v2679) ({
    ^bb0(%aras3b5W1: tensor<f32>, %arbs3b5W1: tensor<f32>):
      %aradds3b5W1 = stablehlo.add %aras3b5W1, %arbs3b5W1 : tensor<f32>
      stablehlo.return %aradds3b5W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b5W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b5W1 = stablehlo.divide %arsums3b5W1, %arns3b5W1 : tensor<256x1024x1x1xf32>
    %v7694 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7695 = stablehlo.multiply %v7694, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v7696 = stablehlo.add %v7695, %armeans3b5W1 : tensor<256x1024x1x1xf32>
    %v7697 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7698 = stablehlo.multiply %v7697, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v7699 = stablehlo.add %v7698, %v7696 : tensor<256x1024x1x1xf32>
    %v7700 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7701 = stablehlo.multiply %v7700, %v7699 : tensor<256x1024x1x1xf32>
    %v7702 = stablehlo.subtract %s3b5W1, %v7701 : tensor<256x1024x1x1xf32>
    %arsums3b5g1 = "stablehlo.all_reduce"(%v2697) ({
    ^bb0(%aras3b5g1: tensor<f32>, %arbs3b5g1: tensor<f32>):
      %aradds3b5g1 = stablehlo.add %aras3b5g1, %arbs3b5g1 : tensor<f32>
      stablehlo.return %aradds3b5g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1 = stablehlo.divide %arsums3b5g1, %arns3b5g1 : tensor<256xf32>
    %v7703 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7704 = stablehlo.multiply %v7703, %s3b5g1 : tensor<256xf32>
    %v7705 = stablehlo.add %v7704, %armeans3b5g1 : tensor<256xf32>
    %v7706 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7707 = stablehlo.multiply %v7706, %s3b5g1v : tensor<256xf32>
    %v7708 = stablehlo.add %v7707, %v7705 : tensor<256xf32>
    %v7709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7710 = stablehlo.multiply %v7709, %v7708 : tensor<256xf32>
    %v7711 = stablehlo.subtract %s3b5g1, %v7710 : tensor<256xf32>
    %arsums3b5bt1 = "stablehlo.all_reduce"(%v2700) ({
    ^bb0(%aras3b5bt1: tensor<f32>, %arbs3b5bt1: tensor<f32>):
      %aradds3b5bt1 = stablehlo.add %aras3b5bt1, %arbs3b5bt1 : tensor<f32>
      stablehlo.return %aradds3b5bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt1 = stablehlo.divide %arsums3b5bt1, %arns3b5bt1 : tensor<256xf32>
    %v7712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7713 = stablehlo.multiply %v7712, %s3b5bt1 : tensor<256xf32>
    %v7714 = stablehlo.add %v7713, %armeans3b5bt1 : tensor<256xf32>
    %v7715 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7716 = stablehlo.multiply %v7715, %s3b5bt1v : tensor<256xf32>
    %v7717 = stablehlo.add %v7716, %v7714 : tensor<256xf32>
    %v7718 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7719 = stablehlo.multiply %v7718, %v7717 : tensor<256xf32>
    %v7720 = stablehlo.subtract %s3b5bt1, %v7719 : tensor<256xf32>
    %arsums3b5W2 = "stablehlo.all_reduce"(%v2709) ({
    ^bb0(%aras3b5W2: tensor<f32>, %arbs3b5W2: tensor<f32>):
      %aradds3b5W2 = stablehlo.add %aras3b5W2, %arbs3b5W2 : tensor<f32>
      stablehlo.return %aradds3b5W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b5W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b5W2 = stablehlo.divide %arsums3b5W2, %arns3b5W2 : tensor<256x256x3x3xf32>
    %v7721 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7722 = stablehlo.multiply %v7721, %s3b5W2 : tensor<256x256x3x3xf32>
    %v7723 = stablehlo.add %v7722, %armeans3b5W2 : tensor<256x256x3x3xf32>
    %v7724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7725 = stablehlo.multiply %v7724, %s3b5W2v : tensor<256x256x3x3xf32>
    %v7726 = stablehlo.add %v7725, %v7723 : tensor<256x256x3x3xf32>
    %v7727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7728 = stablehlo.multiply %v7727, %v7726 : tensor<256x256x3x3xf32>
    %v7729 = stablehlo.subtract %s3b5W2, %v7728 : tensor<256x256x3x3xf32>
    %arsums3b5g2 = "stablehlo.all_reduce"(%v2727) ({
    ^bb0(%aras3b5g2: tensor<f32>, %arbs3b5g2: tensor<f32>):
      %aradds3b5g2 = stablehlo.add %aras3b5g2, %arbs3b5g2 : tensor<f32>
      stablehlo.return %aradds3b5g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2 = stablehlo.divide %arsums3b5g2, %arns3b5g2 : tensor<256xf32>
    %v7730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7731 = stablehlo.multiply %v7730, %s3b5g2 : tensor<256xf32>
    %v7732 = stablehlo.add %v7731, %armeans3b5g2 : tensor<256xf32>
    %v7733 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7734 = stablehlo.multiply %v7733, %s3b5g2v : tensor<256xf32>
    %v7735 = stablehlo.add %v7734, %v7732 : tensor<256xf32>
    %v7736 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7737 = stablehlo.multiply %v7736, %v7735 : tensor<256xf32>
    %v7738 = stablehlo.subtract %s3b5g2, %v7737 : tensor<256xf32>
    %arsums3b5bt2 = "stablehlo.all_reduce"(%v2730) ({
    ^bb0(%aras3b5bt2: tensor<f32>, %arbs3b5bt2: tensor<f32>):
      %aradds3b5bt2 = stablehlo.add %aras3b5bt2, %arbs3b5bt2 : tensor<f32>
      stablehlo.return %aradds3b5bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt2 = stablehlo.divide %arsums3b5bt2, %arns3b5bt2 : tensor<256xf32>
    %v7739 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7740 = stablehlo.multiply %v7739, %s3b5bt2 : tensor<256xf32>
    %v7741 = stablehlo.add %v7740, %armeans3b5bt2 : tensor<256xf32>
    %v7742 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7743 = stablehlo.multiply %v7742, %s3b5bt2v : tensor<256xf32>
    %v7744 = stablehlo.add %v7743, %v7741 : tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.multiply %v7745, %v7744 : tensor<256xf32>
    %v7747 = stablehlo.subtract %s3b5bt2, %v7746 : tensor<256xf32>
    %arsums3b5W3 = "stablehlo.all_reduce"(%v2739) ({
    ^bb0(%aras3b5W3: tensor<f32>, %arbs3b5W3: tensor<f32>):
      %aradds3b5W3 = stablehlo.add %aras3b5W3, %arbs3b5W3 : tensor<f32>
      stablehlo.return %aradds3b5W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b5W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b5W3 = stablehlo.divide %arsums3b5W3, %arns3b5W3 : tensor<1024x256x1x1xf32>
    %v7748 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7749 = stablehlo.multiply %v7748, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v7750 = stablehlo.add %v7749, %armeans3b5W3 : tensor<1024x256x1x1xf32>
    %v7751 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7752 = stablehlo.multiply %v7751, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v7753 = stablehlo.add %v7752, %v7750 : tensor<1024x256x1x1xf32>
    %v7754 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7755 = stablehlo.multiply %v7754, %v7753 : tensor<1024x256x1x1xf32>
    %v7756 = stablehlo.subtract %s3b5W3, %v7755 : tensor<1024x256x1x1xf32>
    %arsums3b5g3 = "stablehlo.all_reduce"(%v2757) ({
    ^bb0(%aras3b5g3: tensor<f32>, %arbs3b5g3: tensor<f32>):
      %aradds3b5g3 = stablehlo.add %aras3b5g3, %arbs3b5g3 : tensor<f32>
      stablehlo.return %aradds3b5g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3 = stablehlo.divide %arsums3b5g3, %arns3b5g3 : tensor<1024xf32>
    %v7757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7758 = stablehlo.multiply %v7757, %s3b5g3 : tensor<1024xf32>
    %v7759 = stablehlo.add %v7758, %armeans3b5g3 : tensor<1024xf32>
    %v7760 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7761 = stablehlo.multiply %v7760, %s3b5g3v : tensor<1024xf32>
    %v7762 = stablehlo.add %v7761, %v7759 : tensor<1024xf32>
    %v7763 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7764 = stablehlo.multiply %v7763, %v7762 : tensor<1024xf32>
    %v7765 = stablehlo.subtract %s3b5g3, %v7764 : tensor<1024xf32>
    %arsums3b5bt3 = "stablehlo.all_reduce"(%v2760) ({
    ^bb0(%aras3b5bt3: tensor<f32>, %arbs3b5bt3: tensor<f32>):
      %aradds3b5bt3 = stablehlo.add %aras3b5bt3, %arbs3b5bt3 : tensor<f32>
      stablehlo.return %aradds3b5bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5bt3 = stablehlo.divide %arsums3b5bt3, %arns3b5bt3 : tensor<1024xf32>
    %v7766 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7767 = stablehlo.multiply %v7766, %s3b5bt3 : tensor<1024xf32>
    %v7768 = stablehlo.add %v7767, %armeans3b5bt3 : tensor<1024xf32>
    %v7769 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7770 = stablehlo.multiply %v7769, %s3b5bt3v : tensor<1024xf32>
    %v7771 = stablehlo.add %v7770, %v7768 : tensor<1024xf32>
    %v7772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7773 = stablehlo.multiply %v7772, %v7771 : tensor<1024xf32>
    %v7774 = stablehlo.subtract %s3b5bt3, %v7773 : tensor<1024xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v2419) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x1024x1x1xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x1024x1x1xf32>
    %v7775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7776 = stablehlo.multiply %v7775, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v7777 = stablehlo.add %v7776, %armeans4b0W1 : tensor<512x1024x1x1xf32>
    %v7778 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7779 = stablehlo.multiply %v7778, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v7780 = stablehlo.add %v7779, %v7777 : tensor<512x1024x1x1xf32>
    %v7781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7782 = stablehlo.multiply %v7781, %v7780 : tensor<512x1024x1x1xf32>
    %v7783 = stablehlo.subtract %s4b0W1, %v7782 : tensor<512x1024x1x1xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v2437) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v7784 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7785 = stablehlo.multiply %v7784, %s4b0g1 : tensor<512xf32>
    %v7786 = stablehlo.add %v7785, %armeans4b0g1 : tensor<512xf32>
    %v7787 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7788 = stablehlo.multiply %v7787, %s4b0g1v : tensor<512xf32>
    %v7789 = stablehlo.add %v7788, %v7786 : tensor<512xf32>
    %v7790 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7791 = stablehlo.multiply %v7790, %v7789 : tensor<512xf32>
    %v7792 = stablehlo.subtract %s4b0g1, %v7791 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v2440) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v7793 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7794 = stablehlo.multiply %v7793, %s4b0bt1 : tensor<512xf32>
    %v7795 = stablehlo.add %v7794, %armeans4b0bt1 : tensor<512xf32>
    %v7796 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7797 = stablehlo.multiply %v7796, %s4b0bt1v : tensor<512xf32>
    %v7798 = stablehlo.add %v7797, %v7795 : tensor<512xf32>
    %v7799 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7800 = stablehlo.multiply %v7799, %v7798 : tensor<512xf32>
    %v7801 = stablehlo.subtract %s4b0bt1, %v7800 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v2451) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v7802 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7803 = stablehlo.multiply %v7802, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7804 = stablehlo.add %v7803, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7805 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7806 = stablehlo.multiply %v7805, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7807 = stablehlo.add %v7806, %v7804 : tensor<512x512x3x3xf32>
    %v7808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7809 = stablehlo.multiply %v7808, %v7807 : tensor<512x512x3x3xf32>
    %v7810 = stablehlo.subtract %s4b0W2, %v7809 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v2469) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v7811 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7812 = stablehlo.multiply %v7811, %s4b0g2 : tensor<512xf32>
    %v7813 = stablehlo.add %v7812, %armeans4b0g2 : tensor<512xf32>
    %v7814 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7815 = stablehlo.multiply %v7814, %s4b0g2v : tensor<512xf32>
    %v7816 = stablehlo.add %v7815, %v7813 : tensor<512xf32>
    %v7817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7818 = stablehlo.multiply %v7817, %v7816 : tensor<512xf32>
    %v7819 = stablehlo.subtract %s4b0g2, %v7818 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v2472) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v7820 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7821 = stablehlo.multiply %v7820, %s4b0bt2 : tensor<512xf32>
    %v7822 = stablehlo.add %v7821, %armeans4b0bt2 : tensor<512xf32>
    %v7823 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7824 = stablehlo.multiply %v7823, %s4b0bt2v : tensor<512xf32>
    %v7825 = stablehlo.add %v7824, %v7822 : tensor<512xf32>
    %v7826 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7827 = stablehlo.multiply %v7826, %v7825 : tensor<512xf32>
    %v7828 = stablehlo.subtract %s4b0bt2, %v7827 : tensor<512xf32>
    %arsums4b0W3 = "stablehlo.all_reduce"(%v2481) ({
    ^bb0(%aras4b0W3: tensor<f32>, %arbs4b0W3: tensor<f32>):
      %aradds4b0W3 = stablehlo.add %aras4b0W3, %arbs4b0W3 : tensor<f32>
      stablehlo.return %aradds4b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b0W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b0W3 = stablehlo.divide %arsums4b0W3, %arns4b0W3 : tensor<2048x512x1x1xf32>
    %v7829 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7830 = stablehlo.multiply %v7829, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7831 = stablehlo.add %v7830, %armeans4b0W3 : tensor<2048x512x1x1xf32>
    %v7832 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7833 = stablehlo.multiply %v7832, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7834 = stablehlo.add %v7833, %v7831 : tensor<2048x512x1x1xf32>
    %v7835 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7836 = stablehlo.multiply %v7835, %v7834 : tensor<2048x512x1x1xf32>
    %v7837 = stablehlo.subtract %s4b0W3, %v7836 : tensor<2048x512x1x1xf32>
    %arsums4b0g3 = "stablehlo.all_reduce"(%v2499) ({
    ^bb0(%aras4b0g3: tensor<f32>, %arbs4b0g3: tensor<f32>):
      %aradds4b0g3 = stablehlo.add %aras4b0g3, %arbs4b0g3 : tensor<f32>
      stablehlo.return %aradds4b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3 = stablehlo.divide %arsums4b0g3, %arns4b0g3 : tensor<2048xf32>
    %v7838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7839 = stablehlo.multiply %v7838, %s4b0g3 : tensor<2048xf32>
    %v7840 = stablehlo.add %v7839, %armeans4b0g3 : tensor<2048xf32>
    %v7841 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7842 = stablehlo.multiply %v7841, %s4b0g3v : tensor<2048xf32>
    %v7843 = stablehlo.add %v7842, %v7840 : tensor<2048xf32>
    %v7844 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7845 = stablehlo.multiply %v7844, %v7843 : tensor<2048xf32>
    %v7846 = stablehlo.subtract %s4b0g3, %v7845 : tensor<2048xf32>
    %arsums4b0bt3 = "stablehlo.all_reduce"(%v2502) ({
    ^bb0(%aras4b0bt3: tensor<f32>, %arbs4b0bt3: tensor<f32>):
      %aradds4b0bt3 = stablehlo.add %aras4b0bt3, %arbs4b0bt3 : tensor<f32>
      stablehlo.return %aradds4b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0bt3 = stablehlo.divide %arsums4b0bt3, %arns4b0bt3 : tensor<2048xf32>
    %v7847 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7848 = stablehlo.multiply %v7847, %s4b0bt3 : tensor<2048xf32>
    %v7849 = stablehlo.add %v7848, %armeans4b0bt3 : tensor<2048xf32>
    %v7850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7851 = stablehlo.multiply %v7850, %s4b0bt3v : tensor<2048xf32>
    %v7852 = stablehlo.add %v7851, %v7849 : tensor<2048xf32>
    %v7853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7854 = stablehlo.multiply %v7853, %v7852 : tensor<2048xf32>
    %v7855 = stablehlo.subtract %s4b0bt3, %v7854 : tensor<2048xf32>
    %arsums4b0Wp = "stablehlo.all_reduce"(%v2513) ({
    ^bb0(%aras4b0Wp: tensor<f32>, %arbs4b0Wp: tensor<f32>):
      %aradds4b0Wp = stablehlo.add %aras4b0Wp, %arbs4b0Wp : tensor<f32>
      stablehlo.return %aradds4b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %arns4b0Wp = stablehlo.constant dense<4.0> : tensor<2048x1024x1x1xf32>
    %armeans4b0Wp = stablehlo.divide %arsums4b0Wp, %arns4b0Wp : tensor<2048x1024x1x1xf32>
    %v7856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7857 = stablehlo.multiply %v7856, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7858 = stablehlo.add %v7857, %armeans4b0Wp : tensor<2048x1024x1x1xf32>
    %v7859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7860 = stablehlo.multiply %v7859, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7861 = stablehlo.add %v7860, %v7858 : tensor<2048x1024x1x1xf32>
    %v7862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7863 = stablehlo.multiply %v7862, %v7861 : tensor<2048x1024x1x1xf32>
    %v7864 = stablehlo.subtract %s4b0Wp, %v7863 : tensor<2048x1024x1x1xf32>
    %arsums4b0gp = "stablehlo.all_reduce"(%v2531) ({
    ^bb0(%aras4b0gp: tensor<f32>, %arbs4b0gp: tensor<f32>):
      %aradds4b0gp = stablehlo.add %aras4b0gp, %arbs4b0gp : tensor<f32>
      stablehlo.return %aradds4b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gp = stablehlo.divide %arsums4b0gp, %arns4b0gp : tensor<2048xf32>
    %v7865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7866 = stablehlo.multiply %v7865, %s4b0gp : tensor<2048xf32>
    %v7867 = stablehlo.add %v7866, %armeans4b0gp : tensor<2048xf32>
    %v7868 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7869 = stablehlo.multiply %v7868, %s4b0gpv : tensor<2048xf32>
    %v7870 = stablehlo.add %v7869, %v7867 : tensor<2048xf32>
    %v7871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7872 = stablehlo.multiply %v7871, %v7870 : tensor<2048xf32>
    %v7873 = stablehlo.subtract %s4b0gp, %v7872 : tensor<2048xf32>
    %arsums4b0btp = "stablehlo.all_reduce"(%v2534) ({
    ^bb0(%aras4b0btp: tensor<f32>, %arbs4b0btp: tensor<f32>):
      %aradds4b0btp = stablehlo.add %aras4b0btp, %arbs4b0btp : tensor<f32>
      stablehlo.return %aradds4b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0btp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0btp = stablehlo.divide %arsums4b0btp, %arns4b0btp : tensor<2048xf32>
    %v7874 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7875 = stablehlo.multiply %v7874, %s4b0btp : tensor<2048xf32>
    %v7876 = stablehlo.add %v7875, %armeans4b0btp : tensor<2048xf32>
    %v7877 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7878 = stablehlo.multiply %v7877, %s4b0btpv : tensor<2048xf32>
    %v7879 = stablehlo.add %v7878, %v7876 : tensor<2048xf32>
    %v7880 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7881 = stablehlo.multiply %v7880, %v7879 : tensor<2048xf32>
    %v7882 = stablehlo.subtract %s4b0btp, %v7881 : tensor<2048xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v2151) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x2048x1x1xf32>
    %v7883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7884 = stablehlo.multiply %v7883, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7885 = stablehlo.add %v7884, %armeans4b1W1 : tensor<512x2048x1x1xf32>
    %v7886 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7887 = stablehlo.multiply %v7886, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7888 = stablehlo.add %v7887, %v7885 : tensor<512x2048x1x1xf32>
    %v7889 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7890 = stablehlo.multiply %v7889, %v7888 : tensor<512x2048x1x1xf32>
    %v7891 = stablehlo.subtract %s4b1W1, %v7890 : tensor<512x2048x1x1xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v2169) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v7892 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7893 = stablehlo.multiply %v7892, %s4b1g1 : tensor<512xf32>
    %v7894 = stablehlo.add %v7893, %armeans4b1g1 : tensor<512xf32>
    %v7895 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7896 = stablehlo.multiply %v7895, %s4b1g1v : tensor<512xf32>
    %v7897 = stablehlo.add %v7896, %v7894 : tensor<512xf32>
    %v7898 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7899 = stablehlo.multiply %v7898, %v7897 : tensor<512xf32>
    %v7900 = stablehlo.subtract %s4b1g1, %v7899 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v2172) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v7901 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7902 = stablehlo.multiply %v7901, %s4b1bt1 : tensor<512xf32>
    %v7903 = stablehlo.add %v7902, %armeans4b1bt1 : tensor<512xf32>
    %v7904 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7905 = stablehlo.multiply %v7904, %s4b1bt1v : tensor<512xf32>
    %v7906 = stablehlo.add %v7905, %v7903 : tensor<512xf32>
    %v7907 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7908 = stablehlo.multiply %v7907, %v7906 : tensor<512xf32>
    %v7909 = stablehlo.subtract %s4b1bt1, %v7908 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v2181) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v7910 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7911 = stablehlo.multiply %v7910, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7912 = stablehlo.add %v7911, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7913 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7914 = stablehlo.multiply %v7913, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7915 = stablehlo.add %v7914, %v7912 : tensor<512x512x3x3xf32>
    %v7916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7917 = stablehlo.multiply %v7916, %v7915 : tensor<512x512x3x3xf32>
    %v7918 = stablehlo.subtract %s4b1W2, %v7917 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v2199) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v7919 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7920 = stablehlo.multiply %v7919, %s4b1g2 : tensor<512xf32>
    %v7921 = stablehlo.add %v7920, %armeans4b1g2 : tensor<512xf32>
    %v7922 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7923 = stablehlo.multiply %v7922, %s4b1g2v : tensor<512xf32>
    %v7924 = stablehlo.add %v7923, %v7921 : tensor<512xf32>
    %v7925 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7926 = stablehlo.multiply %v7925, %v7924 : tensor<512xf32>
    %v7927 = stablehlo.subtract %s4b1g2, %v7926 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v2202) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v7928 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7929 = stablehlo.multiply %v7928, %s4b1bt2 : tensor<512xf32>
    %v7930 = stablehlo.add %v7929, %armeans4b1bt2 : tensor<512xf32>
    %v7931 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7932 = stablehlo.multiply %v7931, %s4b1bt2v : tensor<512xf32>
    %v7933 = stablehlo.add %v7932, %v7930 : tensor<512xf32>
    %v7934 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7935 = stablehlo.multiply %v7934, %v7933 : tensor<512xf32>
    %v7936 = stablehlo.subtract %s4b1bt2, %v7935 : tensor<512xf32>
    %arsums4b1W3 = "stablehlo.all_reduce"(%v2211) ({
    ^bb0(%aras4b1W3: tensor<f32>, %arbs4b1W3: tensor<f32>):
      %aradds4b1W3 = stablehlo.add %aras4b1W3, %arbs4b1W3 : tensor<f32>
      stablehlo.return %aradds4b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b1W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b1W3 = stablehlo.divide %arsums4b1W3, %arns4b1W3 : tensor<2048x512x1x1xf32>
    %v7937 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7938 = stablehlo.multiply %v7937, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7939 = stablehlo.add %v7938, %armeans4b1W3 : tensor<2048x512x1x1xf32>
    %v7940 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7941 = stablehlo.multiply %v7940, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7942 = stablehlo.add %v7941, %v7939 : tensor<2048x512x1x1xf32>
    %v7943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7944 = stablehlo.multiply %v7943, %v7942 : tensor<2048x512x1x1xf32>
    %v7945 = stablehlo.subtract %s4b1W3, %v7944 : tensor<2048x512x1x1xf32>
    %arsums4b1g3 = "stablehlo.all_reduce"(%v2229) ({
    ^bb0(%aras4b1g3: tensor<f32>, %arbs4b1g3: tensor<f32>):
      %aradds4b1g3 = stablehlo.add %aras4b1g3, %arbs4b1g3 : tensor<f32>
      stablehlo.return %aradds4b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3 = stablehlo.divide %arsums4b1g3, %arns4b1g3 : tensor<2048xf32>
    %v7946 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7947 = stablehlo.multiply %v7946, %s4b1g3 : tensor<2048xf32>
    %v7948 = stablehlo.add %v7947, %armeans4b1g3 : tensor<2048xf32>
    %v7949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7950 = stablehlo.multiply %v7949, %s4b1g3v : tensor<2048xf32>
    %v7951 = stablehlo.add %v7950, %v7948 : tensor<2048xf32>
    %v7952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7953 = stablehlo.multiply %v7952, %v7951 : tensor<2048xf32>
    %v7954 = stablehlo.subtract %s4b1g3, %v7953 : tensor<2048xf32>
    %arsums4b1bt3 = "stablehlo.all_reduce"(%v2232) ({
    ^bb0(%aras4b1bt3: tensor<f32>, %arbs4b1bt3: tensor<f32>):
      %aradds4b1bt3 = stablehlo.add %aras4b1bt3, %arbs4b1bt3 : tensor<f32>
      stablehlo.return %aradds4b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1bt3 = stablehlo.divide %arsums4b1bt3, %arns4b1bt3 : tensor<2048xf32>
    %v7955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7956 = stablehlo.multiply %v7955, %s4b1bt3 : tensor<2048xf32>
    %v7957 = stablehlo.add %v7956, %armeans4b1bt3 : tensor<2048xf32>
    %v7958 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7959 = stablehlo.multiply %v7958, %s4b1bt3v : tensor<2048xf32>
    %v7960 = stablehlo.add %v7959, %v7957 : tensor<2048xf32>
    %v7961 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7962 = stablehlo.multiply %v7961, %v7960 : tensor<2048xf32>
    %v7963 = stablehlo.subtract %s4b1bt3, %v7962 : tensor<2048xf32>
    %arsums4b2W1 = "stablehlo.all_reduce"(%v1925) ({
    ^bb0(%aras4b2W1: tensor<f32>, %arbs4b2W1: tensor<f32>):
      %aradds4b2W1 = stablehlo.add %aras4b2W1, %arbs4b2W1 : tensor<f32>
      stablehlo.return %aradds4b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b2W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b2W1 = stablehlo.divide %arsums4b2W1, %arns4b2W1 : tensor<512x2048x1x1xf32>
    %v7964 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7965 = stablehlo.multiply %v7964, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7966 = stablehlo.add %v7965, %armeans4b2W1 : tensor<512x2048x1x1xf32>
    %v7967 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7968 = stablehlo.multiply %v7967, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7969 = stablehlo.add %v7968, %v7966 : tensor<512x2048x1x1xf32>
    %v7970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7971 = stablehlo.multiply %v7970, %v7969 : tensor<512x2048x1x1xf32>
    %v7972 = stablehlo.subtract %s4b2W1, %v7971 : tensor<512x2048x1x1xf32>
    %arsums4b2g1 = "stablehlo.all_reduce"(%v1943) ({
    ^bb0(%aras4b2g1: tensor<f32>, %arbs4b2g1: tensor<f32>):
      %aradds4b2g1 = stablehlo.add %aras4b2g1, %arbs4b2g1 : tensor<f32>
      stablehlo.return %aradds4b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1 = stablehlo.divide %arsums4b2g1, %arns4b2g1 : tensor<512xf32>
    %v7973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7974 = stablehlo.multiply %v7973, %s4b2g1 : tensor<512xf32>
    %v7975 = stablehlo.add %v7974, %armeans4b2g1 : tensor<512xf32>
    %v7976 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7977 = stablehlo.multiply %v7976, %s4b2g1v : tensor<512xf32>
    %v7978 = stablehlo.add %v7977, %v7975 : tensor<512xf32>
    %v7979 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7980 = stablehlo.multiply %v7979, %v7978 : tensor<512xf32>
    %v7981 = stablehlo.subtract %s4b2g1, %v7980 : tensor<512xf32>
    %arsums4b2bt1 = "stablehlo.all_reduce"(%v1946) ({
    ^bb0(%aras4b2bt1: tensor<f32>, %arbs4b2bt1: tensor<f32>):
      %aradds4b2bt1 = stablehlo.add %aras4b2bt1, %arbs4b2bt1 : tensor<f32>
      stablehlo.return %aradds4b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt1 = stablehlo.divide %arsums4b2bt1, %arns4b2bt1 : tensor<512xf32>
    %v7982 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7983 = stablehlo.multiply %v7982, %s4b2bt1 : tensor<512xf32>
    %v7984 = stablehlo.add %v7983, %armeans4b2bt1 : tensor<512xf32>
    %v7985 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7986 = stablehlo.multiply %v7985, %s4b2bt1v : tensor<512xf32>
    %v7987 = stablehlo.add %v7986, %v7984 : tensor<512xf32>
    %v7988 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7989 = stablehlo.multiply %v7988, %v7987 : tensor<512xf32>
    %v7990 = stablehlo.subtract %s4b2bt1, %v7989 : tensor<512xf32>
    %arsums4b2W2 = "stablehlo.all_reduce"(%v1955) ({
    ^bb0(%aras4b2W2: tensor<f32>, %arbs4b2W2: tensor<f32>):
      %aradds4b2W2 = stablehlo.add %aras4b2W2, %arbs4b2W2 : tensor<f32>
      stablehlo.return %aradds4b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b2W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b2W2 = stablehlo.divide %arsums4b2W2, %arns4b2W2 : tensor<512x512x3x3xf32>
    %v7991 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7992 = stablehlo.multiply %v7991, %s4b2W2 : tensor<512x512x3x3xf32>
    %v7993 = stablehlo.add %v7992, %armeans4b2W2 : tensor<512x512x3x3xf32>
    %v7994 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7995 = stablehlo.multiply %v7994, %s4b2W2v : tensor<512x512x3x3xf32>
    %v7996 = stablehlo.add %v7995, %v7993 : tensor<512x512x3x3xf32>
    %v7997 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7998 = stablehlo.multiply %v7997, %v7996 : tensor<512x512x3x3xf32>
    %v7999 = stablehlo.subtract %s4b2W2, %v7998 : tensor<512x512x3x3xf32>
    %arsums4b2g2 = "stablehlo.all_reduce"(%v1973) ({
    ^bb0(%aras4b2g2: tensor<f32>, %arbs4b2g2: tensor<f32>):
      %aradds4b2g2 = stablehlo.add %aras4b2g2, %arbs4b2g2 : tensor<f32>
      stablehlo.return %aradds4b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2 = stablehlo.divide %arsums4b2g2, %arns4b2g2 : tensor<512xf32>
    %v8000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8001 = stablehlo.multiply %v8000, %s4b2g2 : tensor<512xf32>
    %v8002 = stablehlo.add %v8001, %armeans4b2g2 : tensor<512xf32>
    %v8003 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8004 = stablehlo.multiply %v8003, %s4b2g2v : tensor<512xf32>
    %v8005 = stablehlo.add %v8004, %v8002 : tensor<512xf32>
    %v8006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8007 = stablehlo.multiply %v8006, %v8005 : tensor<512xf32>
    %v8008 = stablehlo.subtract %s4b2g2, %v8007 : tensor<512xf32>
    %arsums4b2bt2 = "stablehlo.all_reduce"(%v1976) ({
    ^bb0(%aras4b2bt2: tensor<f32>, %arbs4b2bt2: tensor<f32>):
      %aradds4b2bt2 = stablehlo.add %aras4b2bt2, %arbs4b2bt2 : tensor<f32>
      stablehlo.return %aradds4b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt2 = stablehlo.divide %arsums4b2bt2, %arns4b2bt2 : tensor<512xf32>
    %v8009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8010 = stablehlo.multiply %v8009, %s4b2bt2 : tensor<512xf32>
    %v8011 = stablehlo.add %v8010, %armeans4b2bt2 : tensor<512xf32>
    %v8012 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8013 = stablehlo.multiply %v8012, %s4b2bt2v : tensor<512xf32>
    %v8014 = stablehlo.add %v8013, %v8011 : tensor<512xf32>
    %v8015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8016 = stablehlo.multiply %v8015, %v8014 : tensor<512xf32>
    %v8017 = stablehlo.subtract %s4b2bt2, %v8016 : tensor<512xf32>
    %arsums4b2W3 = "stablehlo.all_reduce"(%v1985) ({
    ^bb0(%aras4b2W3: tensor<f32>, %arbs4b2W3: tensor<f32>):
      %aradds4b2W3 = stablehlo.add %aras4b2W3, %arbs4b2W3 : tensor<f32>
      stablehlo.return %aradds4b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b2W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b2W3 = stablehlo.divide %arsums4b2W3, %arns4b2W3 : tensor<2048x512x1x1xf32>
    %v8018 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8019 = stablehlo.multiply %v8018, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v8020 = stablehlo.add %v8019, %armeans4b2W3 : tensor<2048x512x1x1xf32>
    %v8021 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8022 = stablehlo.multiply %v8021, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v8023 = stablehlo.add %v8022, %v8020 : tensor<2048x512x1x1xf32>
    %v8024 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8025 = stablehlo.multiply %v8024, %v8023 : tensor<2048x512x1x1xf32>
    %v8026 = stablehlo.subtract %s4b2W3, %v8025 : tensor<2048x512x1x1xf32>
    %arsums4b2g3 = "stablehlo.all_reduce"(%v2003) ({
    ^bb0(%aras4b2g3: tensor<f32>, %arbs4b2g3: tensor<f32>):
      %aradds4b2g3 = stablehlo.add %aras4b2g3, %arbs4b2g3 : tensor<f32>
      stablehlo.return %aradds4b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3 = stablehlo.divide %arsums4b2g3, %arns4b2g3 : tensor<2048xf32>
    %v8027 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8028 = stablehlo.multiply %v8027, %s4b2g3 : tensor<2048xf32>
    %v8029 = stablehlo.add %v8028, %armeans4b2g3 : tensor<2048xf32>
    %v8030 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8031 = stablehlo.multiply %v8030, %s4b2g3v : tensor<2048xf32>
    %v8032 = stablehlo.add %v8031, %v8029 : tensor<2048xf32>
    %v8033 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8034 = stablehlo.multiply %v8033, %v8032 : tensor<2048xf32>
    %v8035 = stablehlo.subtract %s4b2g3, %v8034 : tensor<2048xf32>
    %arsums4b2bt3 = "stablehlo.all_reduce"(%v2006) ({
    ^bb0(%aras4b2bt3: tensor<f32>, %arbs4b2bt3: tensor<f32>):
      %aradds4b2bt3 = stablehlo.add %aras4b2bt3, %arbs4b2bt3 : tensor<f32>
      stablehlo.return %aradds4b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2bt3 = stablehlo.divide %arsums4b2bt3, %arns4b2bt3 : tensor<2048xf32>
    %v8036 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8037 = stablehlo.multiply %v8036, %s4b2bt3 : tensor<2048xf32>
    %v8038 = stablehlo.add %v8037, %armeans4b2bt3 : tensor<2048xf32>
    %v8039 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8040 = stablehlo.multiply %v8039, %s4b2bt3v : tensor<2048xf32>
    %v8041 = stablehlo.add %v8040, %v8038 : tensor<2048xf32>
    %v8042 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8043 = stablehlo.multiply %v8042, %v8041 : tensor<2048xf32>
    %v8044 = stablehlo.subtract %s4b2bt3, %v8043 : tensor<2048xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1774) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<2048x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<2048x1000xf32>
    %v8045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8046 = stablehlo.multiply %v8045, %Wd : tensor<2048x1000xf32>
    %v8047 = stablehlo.add %v8046, %armeanWd : tensor<2048x1000xf32>
    %v8048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8049 = stablehlo.multiply %v8048, %Wdv : tensor<2048x1000xf32>
    %v8050 = stablehlo.add %v8049, %v8047 : tensor<2048x1000xf32>
    %v8051 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8052 = stablehlo.multiply %v8051, %v8050 : tensor<2048x1000xf32>
    %v8053 = stablehlo.subtract %Wd, %v8052 : tensor<2048x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1776) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v8054 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8055 = stablehlo.multiply %v8054, %bd : tensor<1000xf32>
    %v8056 = stablehlo.add %v8055, %armeanbd : tensor<1000xf32>
    %v8057 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8058 = stablehlo.multiply %v8057, %bdv : tensor<1000xf32>
    %v8059 = stablehlo.add %v8058, %v8056 : tensor<1000xf32>
    %v8060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8061 = stablehlo.multiply %v8060, %v8059 : tensor<1000xf32>
    %v8062 = stablehlo.subtract %bd, %v8061 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1762 : tensor<64x1000xf32>
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
    return %v6622, %v6631, %v6640, %v6649, %v6658, %v6667, %v6676, %v6685, %v6694, %v6703, %v6712, %v6721, %v6730, %v6739, %v6748, %v6757, %v6766, %v6775, %v6784, %v6793, %v6802, %v6811, %v6820, %v6829, %v6838, %v6847, %v6856, %v6865, %v6874, %v6883, %v6892, %v6901, %v6910, %v6919, %v6928, %v6937, %v6946, %v6955, %v6964, %v6973, %v6982, %v6991, %v7000, %v7009, %v7018, %v7027, %v7036, %v7045, %v7054, %v7063, %v7072, %v7081, %v7090, %v7099, %v7108, %v7117, %v7126, %v7135, %v7144, %v7153, %v7162, %v7171, %v7180, %v7189, %v7198, %v7207, %v7216, %v7225, %v7234, %v7243, %v7252, %v7261, %v7270, %v7279, %v7288, %v7297, %v7306, %v7315, %v7324, %v7333, %v7342, %v7351, %v7360, %v7369, %v7378, %v7387, %v7396, %v7405, %v7414, %v7423, %v7432, %v7441, %v7450, %v7459, %v7468, %v7477, %v7486, %v7495, %v7504, %v7513, %v7522, %v7531, %v7540, %v7549, %v7558, %v7567, %v7576, %v7585, %v7594, %v7603, %v7612, %v7621, %v7630, %v7639, %v7648, %v7657, %v7666, %v7675, %v7684, %v7693, %v7702, %v7711, %v7720, %v7729, %v7738, %v7747, %v7756, %v7765, %v7774, %v7783, %v7792, %v7801, %v7810, %v7819, %v7828, %v7837, %v7846, %v7855, %v7864, %v7873, %v7882, %v7891, %v7900, %v7909, %v7918, %v7927, %v7936, %v7945, %v7954, %v7963, %v7972, %v7981, %v7990, %v7999, %v8008, %v8017, %v8026, %v8035, %v8044, %v8053, %v8062, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v6619, %v6628, %v6637, %v6646, %v6655, %v6664, %v6673, %v6682, %v6691, %v6700, %v6709, %v6718, %v6727, %v6736, %v6745, %v6754, %v6763, %v6772, %v6781, %v6790, %v6799, %v6808, %v6817, %v6826, %v6835, %v6844, %v6853, %v6862, %v6871, %v6880, %v6889, %v6898, %v6907, %v6916, %v6925, %v6934, %v6943, %v6952, %v6961, %v6970, %v6979, %v6988, %v6997, %v7006, %v7015, %v7024, %v7033, %v7042, %v7051, %v7060, %v7069, %v7078, %v7087, %v7096, %v7105, %v7114, %v7123, %v7132, %v7141, %v7150, %v7159, %v7168, %v7177, %v7186, %v7195, %v7204, %v7213, %v7222, %v7231, %v7240, %v7249, %v7258, %v7267, %v7276, %v7285, %v7294, %v7303, %v7312, %v7321, %v7330, %v7339, %v7348, %v7357, %v7366, %v7375, %v7384, %v7393, %v7402, %v7411, %v7420, %v7429, %v7438, %v7447, %v7456, %v7465, %v7474, %v7483, %v7492, %v7501, %v7510, %v7519, %v7528, %v7537, %v7546, %v7555, %v7564, %v7573, %v7582, %v7591, %v7600, %v7609, %v7618, %v7627, %v7636, %v7645, %v7654, %v7663, %v7672, %v7681, %v7690, %v7699, %v7708, %v7717, %v7726, %v7735, %v7744, %v7753, %v7762, %v7771, %v7780, %v7789, %v7798, %v7807, %v7816, %v7825, %v7834, %v7843, %v7852, %v7861, %v7870, %v7879, %v7888, %v7897, %v7906, %v7915, %v7924, %v7933, %v7942, %v7951, %v7960, %v7969, %v7978, %v7987, %v7996, %v8005, %v8014, %v8023, %v8032, %v8041, %v8050, %v8059, %loss, %bc1, %bc2, %v5770, %v5781, %v5786, %v5797, %v5802, %v5813, %v5818, %v5829, %v5834, %v5845, %v5850, %v5861, %v5866, %v5877, %v5882, %v5893, %v5898, %v5909, %v5914, %v5925, %v5930, %v5941, %v5946, %v5957, %v5962, %v5973, %v5978, %v5989, %v5994, %v6005, %v6010, %v6021, %v6026, %v6037, %v6042, %v6053, %v6058, %v6069, %v6074, %v6085, %v6090, %v6101, %v6106, %v6117, %v6122, %v6133, %v6138, %v6149, %v6154, %v6165, %v6170, %v6181, %v6186, %v6197, %v6202, %v6213, %v6218, %v6229, %v6234, %v6245, %v6250, %v6261, %v6266, %v6277, %v6282, %v6293, %v6298, %v6309, %v6314, %v6325, %v6330, %v6341, %v6346, %v6357, %v6362, %v6373, %v6378, %v6389, %v6394, %v6405, %v6410, %v6421, %v6426, %v6437, %v6442, %v6453, %v6458, %v6469, %v6474, %v6485, %v6490, %v6501, %v6506, %v6517, %v6522, %v6533, %v6538, %v6549, %v6554, %v6565, %v6570, %v6581, %v6586, %v6597, %v6602, %v6613 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
