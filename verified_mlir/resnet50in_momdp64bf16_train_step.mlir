module @m {
  func.func @resnet50in_momdp64bf16_train_step(%x: tensor<64x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x1x1xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b0W3: tensor<256x64x1x1xf32>, %s1b0g3: tensor<256xf32>, %s1b0bt3: tensor<256xf32>, %s1b0Wp: tensor<256x64x1x1xf32>, %s1b0gp: tensor<256xf32>, %s1b0btp: tensor<256xf32>, %s1b1W1: tensor<64x256x1x1xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b1W3: tensor<256x64x1x1xf32>, %s1b1g3: tensor<256xf32>, %s1b1bt3: tensor<256xf32>, %s1b2W1: tensor<64x256x1x1xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %s1b2W3: tensor<256x64x1x1xf32>, %s1b2g3: tensor<256xf32>, %s1b2bt3: tensor<256xf32>, %s2b0W1: tensor<128x256x1x1xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b0W3: tensor<512x128x1x1xf32>, %s2b0g3: tensor<512xf32>, %s2b0bt3: tensor<512xf32>, %s2b0Wp: tensor<512x256x1x1xf32>, %s2b0gp: tensor<512xf32>, %s2b0btp: tensor<512xf32>, %s2b1W1: tensor<128x512x1x1xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b1W3: tensor<512x128x1x1xf32>, %s2b1g3: tensor<512xf32>, %s2b1bt3: tensor<512xf32>, %s2b2W1: tensor<128x512x1x1xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %s2b2W3: tensor<512x128x1x1xf32>, %s2b2g3: tensor<512xf32>, %s2b2bt3: tensor<512xf32>, %s2b3W1: tensor<128x512x1x1xf32>, %s2b3g1: tensor<128xf32>, %s2b3bt1: tensor<128xf32>, %s2b3W2: tensor<128x128x3x3xf32>, %s2b3g2: tensor<128xf32>, %s2b3bt2: tensor<128xf32>, %s2b3W3: tensor<512x128x1x1xf32>, %s2b3g3: tensor<512xf32>, %s2b3bt3: tensor<512xf32>, %s3b0W1: tensor<256x512x1x1xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b0W3: tensor<1024x256x1x1xf32>, %s3b0g3: tensor<1024xf32>, %s3b0bt3: tensor<1024xf32>, %s3b0Wp: tensor<1024x512x1x1xf32>, %s3b0gp: tensor<1024xf32>, %s3b0btp: tensor<1024xf32>, %s3b1W1: tensor<256x1024x1x1xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b1W3: tensor<1024x256x1x1xf32>, %s3b1g3: tensor<1024xf32>, %s3b1bt3: tensor<1024xf32>, %s3b2W1: tensor<256x1024x1x1xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b2W3: tensor<1024x256x1x1xf32>, %s3b2g3: tensor<1024xf32>, %s3b2bt3: tensor<1024xf32>, %s3b3W1: tensor<256x1024x1x1xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b3W3: tensor<1024x256x1x1xf32>, %s3b3g3: tensor<1024xf32>, %s3b3bt3: tensor<1024xf32>, %s3b4W1: tensor<256x1024x1x1xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %s3b4W3: tensor<1024x256x1x1xf32>, %s3b4g3: tensor<1024xf32>, %s3b4bt3: tensor<1024xf32>, %s3b5W1: tensor<256x1024x1x1xf32>, %s3b5g1: tensor<256xf32>, %s3b5bt1: tensor<256xf32>, %s3b5W2: tensor<256x256x3x3xf32>, %s3b5g2: tensor<256xf32>, %s3b5bt2: tensor<256xf32>, %s3b5W3: tensor<1024x256x1x1xf32>, %s3b5g3: tensor<1024xf32>, %s3b5bt3: tensor<1024xf32>, %s4b0W1: tensor<512x1024x1x1xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b0W3: tensor<2048x512x1x1xf32>, %s4b0g3: tensor<2048xf32>, %s4b0bt3: tensor<2048xf32>, %s4b0Wp: tensor<2048x1024x1x1xf32>, %s4b0gp: tensor<2048xf32>, %s4b0btp: tensor<2048xf32>, %s4b1W1: tensor<512x2048x1x1xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %s4b1W3: tensor<2048x512x1x1xf32>, %s4b1g3: tensor<2048xf32>, %s4b1bt3: tensor<2048xf32>, %s4b2W1: tensor<512x2048x1x1xf32>, %s4b2g1: tensor<512xf32>, %s4b2bt1: tensor<512xf32>, %s4b2W2: tensor<512x512x3x3xf32>, %s4b2g2: tensor<512xf32>, %s4b2bt2: tensor<512xf32>, %s4b2W3: tensor<2048x512x1x1xf32>, %s4b2g3: tensor<2048xf32>, %s4b2bt3: tensor<2048xf32>, %Wd: tensor<2048x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x1x1xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b0W3m: tensor<256x64x1x1xf32>, %s1b0g3m: tensor<256xf32>, %s1b0bt3m: tensor<256xf32>, %s1b0Wpm: tensor<256x64x1x1xf32>, %s1b0gpm: tensor<256xf32>, %s1b0btpm: tensor<256xf32>, %s1b1W1m: tensor<64x256x1x1xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b1W3m: tensor<256x64x1x1xf32>, %s1b1g3m: tensor<256xf32>, %s1b1bt3m: tensor<256xf32>, %s1b2W1m: tensor<64x256x1x1xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %s1b2W3m: tensor<256x64x1x1xf32>, %s1b2g3m: tensor<256xf32>, %s1b2bt3m: tensor<256xf32>, %s2b0W1m: tensor<128x256x1x1xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b0W3m: tensor<512x128x1x1xf32>, %s2b0g3m: tensor<512xf32>, %s2b0bt3m: tensor<512xf32>, %s2b0Wpm: tensor<512x256x1x1xf32>, %s2b0gpm: tensor<512xf32>, %s2b0btpm: tensor<512xf32>, %s2b1W1m: tensor<128x512x1x1xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b1W3m: tensor<512x128x1x1xf32>, %s2b1g3m: tensor<512xf32>, %s2b1bt3m: tensor<512xf32>, %s2b2W1m: tensor<128x512x1x1xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %s2b2W3m: tensor<512x128x1x1xf32>, %s2b2g3m: tensor<512xf32>, %s2b2bt3m: tensor<512xf32>, %s2b3W1m: tensor<128x512x1x1xf32>, %s2b3g1m: tensor<128xf32>, %s2b3bt1m: tensor<128xf32>, %s2b3W2m: tensor<128x128x3x3xf32>, %s2b3g2m: tensor<128xf32>, %s2b3bt2m: tensor<128xf32>, %s2b3W3m: tensor<512x128x1x1xf32>, %s2b3g3m: tensor<512xf32>, %s2b3bt3m: tensor<512xf32>, %s3b0W1m: tensor<256x512x1x1xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b0W3m: tensor<1024x256x1x1xf32>, %s3b0g3m: tensor<1024xf32>, %s3b0bt3m: tensor<1024xf32>, %s3b0Wpm: tensor<1024x512x1x1xf32>, %s3b0gpm: tensor<1024xf32>, %s3b0btpm: tensor<1024xf32>, %s3b1W1m: tensor<256x1024x1x1xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b1W3m: tensor<1024x256x1x1xf32>, %s3b1g3m: tensor<1024xf32>, %s3b1bt3m: tensor<1024xf32>, %s3b2W1m: tensor<256x1024x1x1xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b2W3m: tensor<1024x256x1x1xf32>, %s3b2g3m: tensor<1024xf32>, %s3b2bt3m: tensor<1024xf32>, %s3b3W1m: tensor<256x1024x1x1xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b3W3m: tensor<1024x256x1x1xf32>, %s3b3g3m: tensor<1024xf32>, %s3b3bt3m: tensor<1024xf32>, %s3b4W1m: tensor<256x1024x1x1xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %s3b4W3m: tensor<1024x256x1x1xf32>, %s3b4g3m: tensor<1024xf32>, %s3b4bt3m: tensor<1024xf32>, %s3b5W1m: tensor<256x1024x1x1xf32>, %s3b5g1m: tensor<256xf32>, %s3b5bt1m: tensor<256xf32>, %s3b5W2m: tensor<256x256x3x3xf32>, %s3b5g2m: tensor<256xf32>, %s3b5bt2m: tensor<256xf32>, %s3b5W3m: tensor<1024x256x1x1xf32>, %s3b5g3m: tensor<1024xf32>, %s3b5bt3m: tensor<1024xf32>, %s4b0W1m: tensor<512x1024x1x1xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b0W3m: tensor<2048x512x1x1xf32>, %s4b0g3m: tensor<2048xf32>, %s4b0bt3m: tensor<2048xf32>, %s4b0Wpm: tensor<2048x1024x1x1xf32>, %s4b0gpm: tensor<2048xf32>, %s4b0btpm: tensor<2048xf32>, %s4b1W1m: tensor<512x2048x1x1xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %s4b1W3m: tensor<2048x512x1x1xf32>, %s4b1g3m: tensor<2048xf32>, %s4b1bt3m: tensor<2048xf32>, %s4b2W1m: tensor<512x2048x1x1xf32>, %s4b2g1m: tensor<512xf32>, %s4b2bt1m: tensor<512xf32>, %s4b2W2m: tensor<512x512x3x3xf32>, %s4b2g2m: tensor<512xf32>, %s4b2bt2m: tensor<512xf32>, %s4b2W3m: tensor<2048x512x1x1xf32>, %s4b2g3m: tensor<2048xf32>, %s4b2bt3m: tensor<2048xf32>, %Wdm: tensor<2048x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x1x1xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b0W3v: tensor<256x64x1x1xf32>, %s1b0g3v: tensor<256xf32>, %s1b0bt3v: tensor<256xf32>, %s1b0Wpv: tensor<256x64x1x1xf32>, %s1b0gpv: tensor<256xf32>, %s1b0btpv: tensor<256xf32>, %s1b1W1v: tensor<64x256x1x1xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b1W3v: tensor<256x64x1x1xf32>, %s1b1g3v: tensor<256xf32>, %s1b1bt3v: tensor<256xf32>, %s1b2W1v: tensor<64x256x1x1xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %s1b2W3v: tensor<256x64x1x1xf32>, %s1b2g3v: tensor<256xf32>, %s1b2bt3v: tensor<256xf32>, %s2b0W1v: tensor<128x256x1x1xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b0W3v: tensor<512x128x1x1xf32>, %s2b0g3v: tensor<512xf32>, %s2b0bt3v: tensor<512xf32>, %s2b0Wpv: tensor<512x256x1x1xf32>, %s2b0gpv: tensor<512xf32>, %s2b0btpv: tensor<512xf32>, %s2b1W1v: tensor<128x512x1x1xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b1W3v: tensor<512x128x1x1xf32>, %s2b1g3v: tensor<512xf32>, %s2b1bt3v: tensor<512xf32>, %s2b2W1v: tensor<128x512x1x1xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %s2b2W3v: tensor<512x128x1x1xf32>, %s2b2g3v: tensor<512xf32>, %s2b2bt3v: tensor<512xf32>, %s2b3W1v: tensor<128x512x1x1xf32>, %s2b3g1v: tensor<128xf32>, %s2b3bt1v: tensor<128xf32>, %s2b3W2v: tensor<128x128x3x3xf32>, %s2b3g2v: tensor<128xf32>, %s2b3bt2v: tensor<128xf32>, %s2b3W3v: tensor<512x128x1x1xf32>, %s2b3g3v: tensor<512xf32>, %s2b3bt3v: tensor<512xf32>, %s3b0W1v: tensor<256x512x1x1xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b0W3v: tensor<1024x256x1x1xf32>, %s3b0g3v: tensor<1024xf32>, %s3b0bt3v: tensor<1024xf32>, %s3b0Wpv: tensor<1024x512x1x1xf32>, %s3b0gpv: tensor<1024xf32>, %s3b0btpv: tensor<1024xf32>, %s3b1W1v: tensor<256x1024x1x1xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b1W3v: tensor<1024x256x1x1xf32>, %s3b1g3v: tensor<1024xf32>, %s3b1bt3v: tensor<1024xf32>, %s3b2W1v: tensor<256x1024x1x1xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b2W3v: tensor<1024x256x1x1xf32>, %s3b2g3v: tensor<1024xf32>, %s3b2bt3v: tensor<1024xf32>, %s3b3W1v: tensor<256x1024x1x1xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b3W3v: tensor<1024x256x1x1xf32>, %s3b3g3v: tensor<1024xf32>, %s3b3bt3v: tensor<1024xf32>, %s3b4W1v: tensor<256x1024x1x1xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %s3b4W3v: tensor<1024x256x1x1xf32>, %s3b4g3v: tensor<1024xf32>, %s3b4bt3v: tensor<1024xf32>, %s3b5W1v: tensor<256x1024x1x1xf32>, %s3b5g1v: tensor<256xf32>, %s3b5bt1v: tensor<256xf32>, %s3b5W2v: tensor<256x256x3x3xf32>, %s3b5g2v: tensor<256xf32>, %s3b5bt2v: tensor<256xf32>, %s3b5W3v: tensor<1024x256x1x1xf32>, %s3b5g3v: tensor<1024xf32>, %s3b5bt3v: tensor<1024xf32>, %s4b0W1v: tensor<512x1024x1x1xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b0W3v: tensor<2048x512x1x1xf32>, %s4b0g3v: tensor<2048xf32>, %s4b0bt3v: tensor<2048xf32>, %s4b0Wpv: tensor<2048x1024x1x1xf32>, %s4b0gpv: tensor<2048xf32>, %s4b0btpv: tensor<2048xf32>, %s4b1W1v: tensor<512x2048x1x1xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %s4b1W3v: tensor<2048x512x1x1xf32>, %s4b1g3v: tensor<2048xf32>, %s4b1bt3v: tensor<2048xf32>, %s4b2W1v: tensor<512x2048x1x1xf32>, %s4b2g1v: tensor<512xf32>, %s4b2bt1v: tensor<512xf32>, %s4b2W2v: tensor<512x512x3x3xf32>, %s4b2g2v: tensor<512xf32>, %s4b2bt2v: tensor<512xf32>, %s4b2W3v: tensor<2048x512x1x1xf32>, %s4b2g3v: tensor<2048xf32>, %s4b2bt3v: tensor<2048xf32>, %Wdv: tensor<2048x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b0n3mui: tensor<256xf32>, %s1b0n3vari: tensor<256xf32>, %s1b0npmui: tensor<256xf32>, %s1b0npvari: tensor<256xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b1n3mui: tensor<256xf32>, %s1b1n3vari: tensor<256xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %s1b2n3mui: tensor<256xf32>, %s1b2n3vari: tensor<256xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b0n3mui: tensor<512xf32>, %s2b0n3vari: tensor<512xf32>, %s2b0npmui: tensor<512xf32>, %s2b0npvari: tensor<512xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b1n3mui: tensor<512xf32>, %s2b1n3vari: tensor<512xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %s2b2n3mui: tensor<512xf32>, %s2b2n3vari: tensor<512xf32>, %s2b3n1mui: tensor<128xf32>, %s2b3n1vari: tensor<128xf32>, %s2b3n2mui: tensor<128xf32>, %s2b3n2vari: tensor<128xf32>, %s2b3n3mui: tensor<512xf32>, %s2b3n3vari: tensor<512xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b0n3mui: tensor<1024xf32>, %s3b0n3vari: tensor<1024xf32>, %s3b0npmui: tensor<1024xf32>, %s3b0npvari: tensor<1024xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b1n3mui: tensor<1024xf32>, %s3b1n3vari: tensor<1024xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b2n3mui: tensor<1024xf32>, %s3b2n3vari: tensor<1024xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b3n3mui: tensor<1024xf32>, %s3b3n3vari: tensor<1024xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %s3b4n3mui: tensor<1024xf32>, %s3b4n3vari: tensor<1024xf32>, %s3b5n1mui: tensor<256xf32>, %s3b5n1vari: tensor<256xf32>, %s3b5n2mui: tensor<256xf32>, %s3b5n2vari: tensor<256xf32>, %s3b5n3mui: tensor<1024xf32>, %s3b5n3vari: tensor<1024xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b0n3mui: tensor<2048xf32>, %s4b0n3vari: tensor<2048xf32>, %s4b0npmui: tensor<2048xf32>, %s4b0npvari: tensor<2048xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %s4b1n3mui: tensor<2048xf32>, %s4b1n3vari: tensor<2048xf32>, %s4b2n1mui: tensor<512xf32>, %s4b2n1vari: tensor<512xf32>, %s4b2n2mui: tensor<512xf32>, %s4b2n2vari: tensor<512xf32>, %s4b2n3mui: tensor<2048xf32>, %s4b2n3vari: tensor<2048xf32>, %onehot: tensor<64x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>) {
    // ── ResNet-50 bottleneck batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 4 replicas ──
    // Every line is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes (4d piece 2). BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet50SyncTieB.r50_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet50FwdGraphSync_full_shard (the forward), both in
    // LeanMlir/Proofs/Nets/ResNet/ (planning/global_bn_verified.md).
    // (Both are stated at the f32 nodes; this artifact's bf16 conv twins, which round
    // their operands per element, are not in that statement.)
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
    %v10 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v11 = stablehlo.reduce(%v8 init: %v9) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v12 = stablehlo.divide %v11, %v10 : tensor<64xf32>
    %arsumsgmu = "stablehlo.all_reduce"(%v12) ({
    ^bb0(%arasgmu: tensor<f32>, %arbsgmu: tensor<f32>):
      %araddsgmu = stablehlo.add %arasgmu, %arbsgmu : tensor<f32>
      stablehlo.return %araddsgmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgmu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansgmu = stablehlo.divide %arsumsgmu, %arnsgmu : tensor<64xf32>
    %v13 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v14 = stablehlo.constant dense<0.0> : tensor<f32>
    %v15 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v16 = stablehlo.reduce(%v13 init: %v14) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v17 = stablehlo.divide %v16, %v15 : tensor<64xf32>
    %v18 = stablehlo.broadcast_in_dim %v17, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v19 = stablehlo.subtract %v13, %v18 : tensor<64x64x112x112xf32>
    %v20 = stablehlo.multiply %v19, %v19 : tensor<64x64x112x112xf32>
    %v21 = stablehlo.reduce(%v20 init: %v14) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v22 = stablehlo.divide %v21, %v15 : tensor<64xf32>
    %v23 = stablehlo.subtract %v17, %armeansgmu : tensor<64xf32>
    %v24 = stablehlo.multiply %v23, %v23 : tensor<64xf32>
    %v25 = stablehlo.add %v22, %v24 : tensor<64xf32>
    %arsumsgvar = "stablehlo.all_reduce"(%v25) ({
    ^bb0(%arasgvar: tensor<f32>, %arbsgvar: tensor<f32>):
      %araddsgvar = stablehlo.add %arasgvar, %arbsgvar : tensor<f32>
      stablehlo.return %araddsgvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsgvar = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansgvar = stablehlo.divide %arsumsgvar, %arnsgvar : tensor<64xf32>
    %v26 = stablehlo.concatenate %armeansgmu, %armeansgvar, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v27 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v28 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v29 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v30 = stablehlo.broadcast_in_dim %v28, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v31 = stablehlo.broadcast_in_dim %v29, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v32 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v33 = stablehlo.add %v31, %v32 : tensor<64x64x112x112xf32>
    %v34 = stablehlo.rsqrt %v33 : tensor<64x64x112x112xf32>
    %v35 = stablehlo.subtract %v27, %v30 : tensor<64x64x112x112xf32>
    %v36 = stablehlo.multiply %v35, %v34 : tensor<64x64x112x112xf32>
    %v37 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v38 = stablehlo.broadcast_in_dim %sbt, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v39 = stablehlo.multiply %v36, %v37 : tensor<64x64x112x112xf32>
    %v40 = stablehlo.add %v39, %v38 : tensor<64x64x112x112xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v42 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v43 = stablehlo.maximum %v41, %v42 : tensor<64x802816xf32>
    %v44 = stablehlo.reshape %v43 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v45 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v46 = "stablehlo.reduce_window"(%v44, %v45) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
    %v47 = stablehlo.reshape %v46 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v48 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v49 = stablehlo.convert %v48 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v50 = stablehlo.convert %s1b0W1 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v51 = stablehlo.convolution(%v49, %v50)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v52 = stablehlo.convert %v51 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v53 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v54 = stablehlo.add %v52, %v53 : tensor<64x64x56x56xf32>
    %v55 = stablehlo.reshape %v54 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v56 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v57 = stablehlo.constant dense<0.0> : tensor<f32>
    %v58 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v59 = stablehlo.reduce(%v56 init: %v57) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v60 = stablehlo.divide %v59, %v58 : tensor<64xf32>
    %arsums1b0g1mu = "stablehlo.all_reduce"(%v60) ({
    ^bb0(%aras1b0g1mu: tensor<f32>, %arbs1b0g1mu: tensor<f32>):
      %aradds1b0g1mu = stablehlo.add %aras1b0g1mu, %arbs1b0g1mu : tensor<f32>
      stablehlo.return %aradds1b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1mu = stablehlo.divide %arsums1b0g1mu, %arns1b0g1mu : tensor<64xf32>
    %v61 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v62 = stablehlo.constant dense<0.0> : tensor<f32>
    %v63 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v64 = stablehlo.reduce(%v61 init: %v62) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v65 = stablehlo.divide %v64, %v63 : tensor<64xf32>
    %v66 = stablehlo.broadcast_in_dim %v65, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v67 = stablehlo.subtract %v61, %v66 : tensor<64x64x56x56xf32>
    %v68 = stablehlo.multiply %v67, %v67 : tensor<64x64x56x56xf32>
    %v69 = stablehlo.reduce(%v68 init: %v62) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v70 = stablehlo.divide %v69, %v63 : tensor<64xf32>
    %v71 = stablehlo.subtract %v65, %armeans1b0g1mu : tensor<64xf32>
    %v72 = stablehlo.multiply %v71, %v71 : tensor<64xf32>
    %v73 = stablehlo.add %v70, %v72 : tensor<64xf32>
    %arsums1b0g1var = "stablehlo.all_reduce"(%v73) ({
    ^bb0(%aras1b0g1var: tensor<f32>, %arbs1b0g1var: tensor<f32>):
      %aradds1b0g1var = stablehlo.add %aras1b0g1var, %arbs1b0g1var : tensor<f32>
      stablehlo.return %aradds1b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1var = stablehlo.divide %arsums1b0g1var, %arns1b0g1var : tensor<64xf32>
    %v74 = stablehlo.concatenate %armeans1b0g1mu, %armeans1b0g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v75 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v76 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v77 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v78 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v79 = stablehlo.broadcast_in_dim %v77, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v80 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v81 = stablehlo.add %v79, %v80 : tensor<64x64x56x56xf32>
    %v82 = stablehlo.rsqrt %v81 : tensor<64x64x56x56xf32>
    %v83 = stablehlo.subtract %v75, %v78 : tensor<64x64x56x56xf32>
    %v84 = stablehlo.multiply %v83, %v82 : tensor<64x64x56x56xf32>
    %v85 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v86 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v87 = stablehlo.multiply %v84, %v85 : tensor<64x64x56x56xf32>
    %v88 = stablehlo.add %v87, %v86 : tensor<64x64x56x56xf32>
    %v89 = stablehlo.reshape %v88 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v90 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v91 = stablehlo.maximum %v89, %v90 : tensor<64x200704xf32>
    %v92 = stablehlo.reshape %v91 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v93 = stablehlo.convert %v92 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v94 = stablehlo.convert %s1b0W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v95 = stablehlo.convolution(%v93, %v94)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v96 = stablehlo.convert %v95 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v97 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v98 = stablehlo.add %v96, %v97 : tensor<64x64x56x56xf32>
    %v99 = stablehlo.reshape %v98 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v100 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v101 = stablehlo.constant dense<0.0> : tensor<f32>
    %v102 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v103 = stablehlo.reduce(%v100 init: %v101) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v104 = stablehlo.divide %v103, %v102 : tensor<64xf32>
    %arsums1b0g2mu = "stablehlo.all_reduce"(%v104) ({
    ^bb0(%aras1b0g2mu: tensor<f32>, %arbs1b0g2mu: tensor<f32>):
      %aradds1b0g2mu = stablehlo.add %aras1b0g2mu, %arbs1b0g2mu : tensor<f32>
      stablehlo.return %aradds1b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2mu = stablehlo.divide %arsums1b0g2mu, %arns1b0g2mu : tensor<64xf32>
    %v105 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v107 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v108 = stablehlo.reduce(%v105 init: %v106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v109 = stablehlo.divide %v108, %v107 : tensor<64xf32>
    %v110 = stablehlo.broadcast_in_dim %v109, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v111 = stablehlo.subtract %v105, %v110 : tensor<64x64x56x56xf32>
    %v112 = stablehlo.multiply %v111, %v111 : tensor<64x64x56x56xf32>
    %v113 = stablehlo.reduce(%v112 init: %v106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v114 = stablehlo.divide %v113, %v107 : tensor<64xf32>
    %v115 = stablehlo.subtract %v109, %armeans1b0g2mu : tensor<64xf32>
    %v116 = stablehlo.multiply %v115, %v115 : tensor<64xf32>
    %v117 = stablehlo.add %v114, %v116 : tensor<64xf32>
    %arsums1b0g2var = "stablehlo.all_reduce"(%v117) ({
    ^bb0(%aras1b0g2var: tensor<f32>, %arbs1b0g2var: tensor<f32>):
      %aradds1b0g2var = stablehlo.add %aras1b0g2var, %arbs1b0g2var : tensor<f32>
      stablehlo.return %aradds1b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2var = stablehlo.divide %arsums1b0g2var, %arns1b0g2var : tensor<64xf32>
    %v118 = stablehlo.concatenate %armeans1b0g2mu, %armeans1b0g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v119 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v120 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v121 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v122 = stablehlo.broadcast_in_dim %v120, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v123 = stablehlo.broadcast_in_dim %v121, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v124 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v125 = stablehlo.add %v123, %v124 : tensor<64x64x56x56xf32>
    %v126 = stablehlo.rsqrt %v125 : tensor<64x64x56x56xf32>
    %v127 = stablehlo.subtract %v119, %v122 : tensor<64x64x56x56xf32>
    %v128 = stablehlo.multiply %v127, %v126 : tensor<64x64x56x56xf32>
    %v129 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v131 = stablehlo.multiply %v128, %v129 : tensor<64x64x56x56xf32>
    %v132 = stablehlo.add %v131, %v130 : tensor<64x64x56x56xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v135 = stablehlo.maximum %v133, %v134 : tensor<64x200704xf32>
    %v136 = stablehlo.reshape %v135 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v137 = stablehlo.convert %v136 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v138 = stablehlo.convert %s1b0W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v139 = stablehlo.convolution(%v137, %v138)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v140 = stablehlo.convert %v139 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v141 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v142 = stablehlo.add %v140, %v141 : tensor<64x256x56x56xf32>
    %v143 = stablehlo.reshape %v142 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v144 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v146 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v147 = stablehlo.reduce(%v144 init: %v145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v148 = stablehlo.divide %v147, %v146 : tensor<256xf32>
    %arsums1b0g3mu = "stablehlo.all_reduce"(%v148) ({
    ^bb0(%aras1b0g3mu: tensor<f32>, %arbs1b0g3mu: tensor<f32>):
      %aradds1b0g3mu = stablehlo.add %aras1b0g3mu, %arbs1b0g3mu : tensor<f32>
      stablehlo.return %aradds1b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3mu = stablehlo.divide %arsums1b0g3mu, %arns1b0g3mu : tensor<256xf32>
    %v149 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v151 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v152 = stablehlo.reduce(%v149 init: %v150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v153 = stablehlo.divide %v152, %v151 : tensor<256xf32>
    %v154 = stablehlo.broadcast_in_dim %v153, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v155 = stablehlo.subtract %v149, %v154 : tensor<64x256x56x56xf32>
    %v156 = stablehlo.multiply %v155, %v155 : tensor<64x256x56x56xf32>
    %v157 = stablehlo.reduce(%v156 init: %v150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v158 = stablehlo.divide %v157, %v151 : tensor<256xf32>
    %v159 = stablehlo.subtract %v153, %armeans1b0g3mu : tensor<256xf32>
    %v160 = stablehlo.multiply %v159, %v159 : tensor<256xf32>
    %v161 = stablehlo.add %v158, %v160 : tensor<256xf32>
    %arsums1b0g3var = "stablehlo.all_reduce"(%v161) ({
    ^bb0(%aras1b0g3var: tensor<f32>, %arbs1b0g3var: tensor<f32>):
      %aradds1b0g3var = stablehlo.add %aras1b0g3var, %arbs1b0g3var : tensor<f32>
      stablehlo.return %aradds1b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3var = stablehlo.divide %arsums1b0g3var, %arns1b0g3var : tensor<256xf32>
    %v162 = stablehlo.concatenate %armeans1b0g3mu, %armeans1b0g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v163 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v164 = stablehlo.slice %v162 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v165 = stablehlo.slice %v162 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v166 = stablehlo.broadcast_in_dim %v164, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v167 = stablehlo.broadcast_in_dim %v165, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v168 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v169 = stablehlo.add %v167, %v168 : tensor<64x256x56x56xf32>
    %v170 = stablehlo.rsqrt %v169 : tensor<64x256x56x56xf32>
    %v171 = stablehlo.subtract %v163, %v166 : tensor<64x256x56x56xf32>
    %v172 = stablehlo.multiply %v171, %v170 : tensor<64x256x56x56xf32>
    %v173 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v174 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v175 = stablehlo.multiply %v172, %v173 : tensor<64x256x56x56xf32>
    %v176 = stablehlo.add %v175, %v174 : tensor<64x256x56x56xf32>
    %v177 = stablehlo.reshape %v176 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v178 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v179 = stablehlo.convert %v178 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v180 = stablehlo.convert %s1b0Wp : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v181 = stablehlo.convolution(%v179, %v180)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v182 = stablehlo.convert %v181 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v183 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v184 = stablehlo.add %v182, %v183 : tensor<64x256x56x56xf32>
    %v185 = stablehlo.reshape %v184 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v187 = stablehlo.constant dense<0.0> : tensor<f32>
    %v188 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v189 = stablehlo.reduce(%v186 init: %v187) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v190 = stablehlo.divide %v189, %v188 : tensor<256xf32>
    %arsums1b0gpmu = "stablehlo.all_reduce"(%v190) ({
    ^bb0(%aras1b0gpmu: tensor<f32>, %arbs1b0gpmu: tensor<f32>):
      %aradds1b0gpmu = stablehlo.add %aras1b0gpmu, %arbs1b0gpmu : tensor<f32>
      stablehlo.return %aradds1b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gpmu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gpmu = stablehlo.divide %arsums1b0gpmu, %arns1b0gpmu : tensor<256xf32>
    %v191 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v194 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v195 = stablehlo.divide %v194, %v193 : tensor<256xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v197 = stablehlo.subtract %v191, %v196 : tensor<64x256x56x56xf32>
    %v198 = stablehlo.multiply %v197, %v197 : tensor<64x256x56x56xf32>
    %v199 = stablehlo.reduce(%v198 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v200 = stablehlo.divide %v199, %v193 : tensor<256xf32>
    %v201 = stablehlo.subtract %v195, %armeans1b0gpmu : tensor<256xf32>
    %v202 = stablehlo.multiply %v201, %v201 : tensor<256xf32>
    %v203 = stablehlo.add %v200, %v202 : tensor<256xf32>
    %arsums1b0gpvar = "stablehlo.all_reduce"(%v203) ({
    ^bb0(%aras1b0gpvar: tensor<f32>, %arbs1b0gpvar: tensor<f32>):
      %aradds1b0gpvar = stablehlo.add %aras1b0gpvar, %arbs1b0gpvar : tensor<f32>
      stablehlo.return %aradds1b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gpvar = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gpvar = stablehlo.divide %arsums1b0gpvar, %arns1b0gpvar : tensor<256xf32>
    %v204 = stablehlo.concatenate %armeans1b0gpmu, %armeans1b0gpvar, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v205 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v206 = stablehlo.slice %v204 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v207 = stablehlo.slice %v204 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v208 = stablehlo.broadcast_in_dim %v206, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %v207, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v210 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v211 = stablehlo.add %v209, %v210 : tensor<64x256x56x56xf32>
    %v212 = stablehlo.rsqrt %v211 : tensor<64x256x56x56xf32>
    %v213 = stablehlo.subtract %v205, %v208 : tensor<64x256x56x56xf32>
    %v214 = stablehlo.multiply %v213, %v212 : tensor<64x256x56x56xf32>
    %v215 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v216 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v217 = stablehlo.multiply %v214, %v215 : tensor<64x256x56x56xf32>
    %v218 = stablehlo.add %v217, %v216 : tensor<64x256x56x56xf32>
    %v219 = stablehlo.reshape %v218 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v220 = stablehlo.add %v177, %v219 : tensor<64x802816xf32>
    %v221 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v222 = stablehlo.maximum %v220, %v221 : tensor<64x802816xf32>
    %v223 = stablehlo.reshape %v222 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v224 = stablehlo.convert %v223 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v225 = stablehlo.convert %s1b1W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v226 = stablehlo.convolution(%v224, %v225)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v227 = stablehlo.convert %v226 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v228 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v229 = stablehlo.add %v227, %v228 : tensor<64x64x56x56xf32>
    %v230 = stablehlo.reshape %v229 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v233 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v234 = stablehlo.reduce(%v231 init: %v232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v235 = stablehlo.divide %v234, %v233 : tensor<64xf32>
    %arsums1b1g1mu = "stablehlo.all_reduce"(%v235) ({
    ^bb0(%aras1b1g1mu: tensor<f32>, %arbs1b1g1mu: tensor<f32>):
      %aradds1b1g1mu = stablehlo.add %aras1b1g1mu, %arbs1b1g1mu : tensor<f32>
      stablehlo.return %aradds1b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1mu = stablehlo.divide %arsums1b1g1mu, %arns1b1g1mu : tensor<64xf32>
    %v236 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v237 = stablehlo.constant dense<0.0> : tensor<f32>
    %v238 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v239 = stablehlo.reduce(%v236 init: %v237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v240 = stablehlo.divide %v239, %v238 : tensor<64xf32>
    %v241 = stablehlo.broadcast_in_dim %v240, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v242 = stablehlo.subtract %v236, %v241 : tensor<64x64x56x56xf32>
    %v243 = stablehlo.multiply %v242, %v242 : tensor<64x64x56x56xf32>
    %v244 = stablehlo.reduce(%v243 init: %v237) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v245 = stablehlo.divide %v244, %v238 : tensor<64xf32>
    %v246 = stablehlo.subtract %v240, %armeans1b1g1mu : tensor<64xf32>
    %v247 = stablehlo.multiply %v246, %v246 : tensor<64xf32>
    %v248 = stablehlo.add %v245, %v247 : tensor<64xf32>
    %arsums1b1g1var = "stablehlo.all_reduce"(%v248) ({
    ^bb0(%aras1b1g1var: tensor<f32>, %arbs1b1g1var: tensor<f32>):
      %aradds1b1g1var = stablehlo.add %aras1b1g1var, %arbs1b1g1var : tensor<f32>
      stablehlo.return %aradds1b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1var = stablehlo.divide %arsums1b1g1var, %arns1b1g1var : tensor<64xf32>
    %v249 = stablehlo.concatenate %armeans1b1g1mu, %armeans1b1g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v250 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v251 = stablehlo.slice %v249 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v252 = stablehlo.slice %v249 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v253 = stablehlo.broadcast_in_dim %v251, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v254 = stablehlo.broadcast_in_dim %v252, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v256 = stablehlo.add %v254, %v255 : tensor<64x64x56x56xf32>
    %v257 = stablehlo.rsqrt %v256 : tensor<64x64x56x56xf32>
    %v258 = stablehlo.subtract %v250, %v253 : tensor<64x64x56x56xf32>
    %v259 = stablehlo.multiply %v258, %v257 : tensor<64x64x56x56xf32>
    %v260 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v261 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v262 = stablehlo.multiply %v259, %v260 : tensor<64x64x56x56xf32>
    %v263 = stablehlo.add %v262, %v261 : tensor<64x64x56x56xf32>
    %v264 = stablehlo.reshape %v263 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v265 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v266 = stablehlo.maximum %v264, %v265 : tensor<64x200704xf32>
    %v267 = stablehlo.reshape %v266 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v268 = stablehlo.convert %v267 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v269 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v270 = stablehlo.convolution(%v268, %v269)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v271 = stablehlo.convert %v270 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v272 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v273 = stablehlo.add %v271, %v272 : tensor<64x64x56x56xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v275 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v276 = stablehlo.constant dense<0.0> : tensor<f32>
    %v277 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v278 = stablehlo.reduce(%v275 init: %v276) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v279 = stablehlo.divide %v278, %v277 : tensor<64xf32>
    %arsums1b1g2mu = "stablehlo.all_reduce"(%v279) ({
    ^bb0(%aras1b1g2mu: tensor<f32>, %arbs1b1g2mu: tensor<f32>):
      %aradds1b1g2mu = stablehlo.add %aras1b1g2mu, %arbs1b1g2mu : tensor<f32>
      stablehlo.return %aradds1b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2mu = stablehlo.divide %arsums1b1g2mu, %arns1b1g2mu : tensor<64xf32>
    %v280 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v281 = stablehlo.constant dense<0.0> : tensor<f32>
    %v282 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v283 = stablehlo.reduce(%v280 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v284 = stablehlo.divide %v283, %v282 : tensor<64xf32>
    %v285 = stablehlo.broadcast_in_dim %v284, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v286 = stablehlo.subtract %v280, %v285 : tensor<64x64x56x56xf32>
    %v287 = stablehlo.multiply %v286, %v286 : tensor<64x64x56x56xf32>
    %v288 = stablehlo.reduce(%v287 init: %v281) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v289 = stablehlo.divide %v288, %v282 : tensor<64xf32>
    %v290 = stablehlo.subtract %v284, %armeans1b1g2mu : tensor<64xf32>
    %v291 = stablehlo.multiply %v290, %v290 : tensor<64xf32>
    %v292 = stablehlo.add %v289, %v291 : tensor<64xf32>
    %arsums1b1g2var = "stablehlo.all_reduce"(%v292) ({
    ^bb0(%aras1b1g2var: tensor<f32>, %arbs1b1g2var: tensor<f32>):
      %aradds1b1g2var = stablehlo.add %aras1b1g2var, %arbs1b1g2var : tensor<f32>
      stablehlo.return %aradds1b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2var = stablehlo.divide %arsums1b1g2var, %arns1b1g2var : tensor<64xf32>
    %v293 = stablehlo.concatenate %armeans1b1g2mu, %armeans1b1g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v294 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v295 = stablehlo.slice %v293 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v296 = stablehlo.slice %v293 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v297 = stablehlo.broadcast_in_dim %v295, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v298 = stablehlo.broadcast_in_dim %v296, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v299 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v300 = stablehlo.add %v298, %v299 : tensor<64x64x56x56xf32>
    %v301 = stablehlo.rsqrt %v300 : tensor<64x64x56x56xf32>
    %v302 = stablehlo.subtract %v294, %v297 : tensor<64x64x56x56xf32>
    %v303 = stablehlo.multiply %v302, %v301 : tensor<64x64x56x56xf32>
    %v304 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v305 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v306 = stablehlo.multiply %v303, %v304 : tensor<64x64x56x56xf32>
    %v307 = stablehlo.add %v306, %v305 : tensor<64x64x56x56xf32>
    %v308 = stablehlo.reshape %v307 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v309 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v310 = stablehlo.maximum %v308, %v309 : tensor<64x200704xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v312 = stablehlo.convert %v311 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v313 = stablehlo.convert %s1b1W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v314 = stablehlo.convolution(%v312, %v313)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v315 = stablehlo.convert %v314 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v316 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v317 = stablehlo.add %v315, %v316 : tensor<64x256x56x56xf32>
    %v318 = stablehlo.reshape %v317 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v319 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v320 = stablehlo.constant dense<0.0> : tensor<f32>
    %v321 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v322 = stablehlo.reduce(%v319 init: %v320) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v323 = stablehlo.divide %v322, %v321 : tensor<256xf32>
    %arsums1b1g3mu = "stablehlo.all_reduce"(%v323) ({
    ^bb0(%aras1b1g3mu: tensor<f32>, %arbs1b1g3mu: tensor<f32>):
      %aradds1b1g3mu = stablehlo.add %aras1b1g3mu, %arbs1b1g3mu : tensor<f32>
      stablehlo.return %aradds1b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3mu = stablehlo.divide %arsums1b1g3mu, %arns1b1g3mu : tensor<256xf32>
    %v324 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v326 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v327 = stablehlo.reduce(%v324 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v328 = stablehlo.divide %v327, %v326 : tensor<256xf32>
    %v329 = stablehlo.broadcast_in_dim %v328, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v330 = stablehlo.subtract %v324, %v329 : tensor<64x256x56x56xf32>
    %v331 = stablehlo.multiply %v330, %v330 : tensor<64x256x56x56xf32>
    %v332 = stablehlo.reduce(%v331 init: %v325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v333 = stablehlo.divide %v332, %v326 : tensor<256xf32>
    %v334 = stablehlo.subtract %v328, %armeans1b1g3mu : tensor<256xf32>
    %v335 = stablehlo.multiply %v334, %v334 : tensor<256xf32>
    %v336 = stablehlo.add %v333, %v335 : tensor<256xf32>
    %arsums1b1g3var = "stablehlo.all_reduce"(%v336) ({
    ^bb0(%aras1b1g3var: tensor<f32>, %arbs1b1g3var: tensor<f32>):
      %aradds1b1g3var = stablehlo.add %aras1b1g3var, %arbs1b1g3var : tensor<f32>
      stablehlo.return %aradds1b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3var = stablehlo.divide %arsums1b1g3var, %arns1b1g3var : tensor<256xf32>
    %v337 = stablehlo.concatenate %armeans1b1g3mu, %armeans1b1g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v338 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v339 = stablehlo.slice %v337 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v340 = stablehlo.slice %v337 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v341 = stablehlo.broadcast_in_dim %v339, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v342 = stablehlo.broadcast_in_dim %v340, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v343 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v344 = stablehlo.add %v342, %v343 : tensor<64x256x56x56xf32>
    %v345 = stablehlo.rsqrt %v344 : tensor<64x256x56x56xf32>
    %v346 = stablehlo.subtract %v338, %v341 : tensor<64x256x56x56xf32>
    %v347 = stablehlo.multiply %v346, %v345 : tensor<64x256x56x56xf32>
    %v348 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v349 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v350 = stablehlo.multiply %v347, %v348 : tensor<64x256x56x56xf32>
    %v351 = stablehlo.add %v350, %v349 : tensor<64x256x56x56xf32>
    %v352 = stablehlo.reshape %v351 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v353 = stablehlo.reshape %v352 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v354 = stablehlo.reshape %v222 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v355 = stablehlo.add %v353, %v354 : tensor<64x256x56x56xf32>
    %v356 = stablehlo.reshape %v355 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v357 = stablehlo.reshape %v356 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v358 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v359 = stablehlo.maximum %v357, %v358 : tensor<64x256x56x56xf32>
    %v360 = stablehlo.reshape %v359 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v361 = stablehlo.reshape %v360 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v362 = stablehlo.convert %v361 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v363 = stablehlo.convert %s1b2W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v364 = stablehlo.convolution(%v362, %v363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v365 = stablehlo.convert %v364 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v366 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v367 = stablehlo.add %v365, %v366 : tensor<64x64x56x56xf32>
    %v368 = stablehlo.reshape %v367 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v369 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v371 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v372 = stablehlo.reduce(%v369 init: %v370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v373 = stablehlo.divide %v372, %v371 : tensor<64xf32>
    %arsums1b2g1mu = "stablehlo.all_reduce"(%v373) ({
    ^bb0(%aras1b2g1mu: tensor<f32>, %arbs1b2g1mu: tensor<f32>):
      %aradds1b2g1mu = stablehlo.add %aras1b2g1mu, %arbs1b2g1mu : tensor<f32>
      stablehlo.return %aradds1b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1mu = stablehlo.divide %arsums1b2g1mu, %arns1b2g1mu : tensor<64xf32>
    %v374 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v376 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v377 = stablehlo.reduce(%v374 init: %v375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v378 = stablehlo.divide %v377, %v376 : tensor<64xf32>
    %v379 = stablehlo.broadcast_in_dim %v378, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v380 = stablehlo.subtract %v374, %v379 : tensor<64x64x56x56xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<64x64x56x56xf32>
    %v382 = stablehlo.reduce(%v381 init: %v375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v383 = stablehlo.divide %v382, %v376 : tensor<64xf32>
    %v384 = stablehlo.subtract %v378, %armeans1b2g1mu : tensor<64xf32>
    %v385 = stablehlo.multiply %v384, %v384 : tensor<64xf32>
    %v386 = stablehlo.add %v383, %v385 : tensor<64xf32>
    %arsums1b2g1var = "stablehlo.all_reduce"(%v386) ({
    ^bb0(%aras1b2g1var: tensor<f32>, %arbs1b2g1var: tensor<f32>):
      %aradds1b2g1var = stablehlo.add %aras1b2g1var, %arbs1b2g1var : tensor<f32>
      stablehlo.return %aradds1b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1var = stablehlo.divide %arsums1b2g1var, %arns1b2g1var : tensor<64xf32>
    %v387 = stablehlo.concatenate %armeans1b2g1mu, %armeans1b2g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v388 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v389 = stablehlo.slice %v387 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v390 = stablehlo.slice %v387 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v391 = stablehlo.broadcast_in_dim %v389, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v392 = stablehlo.broadcast_in_dim %v390, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v393 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v394 = stablehlo.add %v392, %v393 : tensor<64x64x56x56xf32>
    %v395 = stablehlo.rsqrt %v394 : tensor<64x64x56x56xf32>
    %v396 = stablehlo.subtract %v388, %v391 : tensor<64x64x56x56xf32>
    %v397 = stablehlo.multiply %v396, %v395 : tensor<64x64x56x56xf32>
    %v398 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v399 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v400 = stablehlo.multiply %v397, %v398 : tensor<64x64x56x56xf32>
    %v401 = stablehlo.add %v400, %v399 : tensor<64x64x56x56xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v403 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v404 = stablehlo.maximum %v402, %v403 : tensor<64x200704xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v406 = stablehlo.convert %v405 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v407 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v408 = stablehlo.convolution(%v406, %v407)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v409 = stablehlo.convert %v408 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v410 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v411 = stablehlo.add %v409, %v410 : tensor<64x64x56x56xf32>
    %v412 = stablehlo.reshape %v411 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v413 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v415 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v416 = stablehlo.reduce(%v413 init: %v414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v417 = stablehlo.divide %v416, %v415 : tensor<64xf32>
    %arsums1b2g2mu = "stablehlo.all_reduce"(%v417) ({
    ^bb0(%aras1b2g2mu: tensor<f32>, %arbs1b2g2mu: tensor<f32>):
      %aradds1b2g2mu = stablehlo.add %aras1b2g2mu, %arbs1b2g2mu : tensor<f32>
      stablehlo.return %aradds1b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2mu = stablehlo.divide %arsums1b2g2mu, %arns1b2g2mu : tensor<64xf32>
    %v418 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v419 = stablehlo.constant dense<0.0> : tensor<f32>
    %v420 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v421 = stablehlo.reduce(%v418 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v422 = stablehlo.divide %v421, %v420 : tensor<64xf32>
    %v423 = stablehlo.broadcast_in_dim %v422, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v424 = stablehlo.subtract %v418, %v423 : tensor<64x64x56x56xf32>
    %v425 = stablehlo.multiply %v424, %v424 : tensor<64x64x56x56xf32>
    %v426 = stablehlo.reduce(%v425 init: %v419) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v427 = stablehlo.divide %v426, %v420 : tensor<64xf32>
    %v428 = stablehlo.subtract %v422, %armeans1b2g2mu : tensor<64xf32>
    %v429 = stablehlo.multiply %v428, %v428 : tensor<64xf32>
    %v430 = stablehlo.add %v427, %v429 : tensor<64xf32>
    %arsums1b2g2var = "stablehlo.all_reduce"(%v430) ({
    ^bb0(%aras1b2g2var: tensor<f32>, %arbs1b2g2var: tensor<f32>):
      %aradds1b2g2var = stablehlo.add %aras1b2g2var, %arbs1b2g2var : tensor<f32>
      stablehlo.return %aradds1b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2var = stablehlo.divide %arsums1b2g2var, %arns1b2g2var : tensor<64xf32>
    %v431 = stablehlo.concatenate %armeans1b2g2mu, %armeans1b2g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v432 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v433 = stablehlo.slice %v431 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v434 = stablehlo.slice %v431 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v435 = stablehlo.broadcast_in_dim %v433, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v436 = stablehlo.broadcast_in_dim %v434, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v437 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v438 = stablehlo.add %v436, %v437 : tensor<64x64x56x56xf32>
    %v439 = stablehlo.rsqrt %v438 : tensor<64x64x56x56xf32>
    %v440 = stablehlo.subtract %v432, %v435 : tensor<64x64x56x56xf32>
    %v441 = stablehlo.multiply %v440, %v439 : tensor<64x64x56x56xf32>
    %v442 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v443 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v444 = stablehlo.multiply %v441, %v442 : tensor<64x64x56x56xf32>
    %v445 = stablehlo.add %v444, %v443 : tensor<64x64x56x56xf32>
    %v446 = stablehlo.reshape %v445 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v447 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v448 = stablehlo.maximum %v446, %v447 : tensor<64x200704xf32>
    %v449 = stablehlo.reshape %v448 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v450 = stablehlo.convert %v449 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v451 = stablehlo.convert %s1b2W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v452 = stablehlo.convolution(%v450, %v451)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v453 = stablehlo.convert %v452 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v454 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v455 = stablehlo.add %v453, %v454 : tensor<64x256x56x56xf32>
    %v456 = stablehlo.reshape %v455 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v459 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v460 = stablehlo.reduce(%v457 init: %v458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v461 = stablehlo.divide %v460, %v459 : tensor<256xf32>
    %arsums1b2g3mu = "stablehlo.all_reduce"(%v461) ({
    ^bb0(%aras1b2g3mu: tensor<f32>, %arbs1b2g3mu: tensor<f32>):
      %aradds1b2g3mu = stablehlo.add %aras1b2g3mu, %arbs1b2g3mu : tensor<f32>
      stablehlo.return %aradds1b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3mu = stablehlo.divide %arsums1b2g3mu, %arns1b2g3mu : tensor<256xf32>
    %v462 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v464 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v465 = stablehlo.reduce(%v462 init: %v463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v466 = stablehlo.divide %v465, %v464 : tensor<256xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v468 = stablehlo.subtract %v462, %v467 : tensor<64x256x56x56xf32>
    %v469 = stablehlo.multiply %v468, %v468 : tensor<64x256x56x56xf32>
    %v470 = stablehlo.reduce(%v469 init: %v463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v471 = stablehlo.divide %v470, %v464 : tensor<256xf32>
    %v472 = stablehlo.subtract %v466, %armeans1b2g3mu : tensor<256xf32>
    %v473 = stablehlo.multiply %v472, %v472 : tensor<256xf32>
    %v474 = stablehlo.add %v471, %v473 : tensor<256xf32>
    %arsums1b2g3var = "stablehlo.all_reduce"(%v474) ({
    ^bb0(%aras1b2g3var: tensor<f32>, %arbs1b2g3var: tensor<f32>):
      %aradds1b2g3var = stablehlo.add %aras1b2g3var, %arbs1b2g3var : tensor<f32>
      stablehlo.return %aradds1b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3var = stablehlo.divide %arsums1b2g3var, %arns1b2g3var : tensor<256xf32>
    %v475 = stablehlo.concatenate %armeans1b2g3mu, %armeans1b2g3var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v476 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v477 = stablehlo.slice %v475 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v478 = stablehlo.slice %v475 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v479 = stablehlo.broadcast_in_dim %v477, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v480 = stablehlo.broadcast_in_dim %v478, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v481 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v482 = stablehlo.add %v480, %v481 : tensor<64x256x56x56xf32>
    %v483 = stablehlo.rsqrt %v482 : tensor<64x256x56x56xf32>
    %v484 = stablehlo.subtract %v476, %v479 : tensor<64x256x56x56xf32>
    %v485 = stablehlo.multiply %v484, %v483 : tensor<64x256x56x56xf32>
    %v486 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v487 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v488 = stablehlo.multiply %v485, %v486 : tensor<64x256x56x56xf32>
    %v489 = stablehlo.add %v488, %v487 : tensor<64x256x56x56xf32>
    %v490 = stablehlo.reshape %v489 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v492 = stablehlo.reshape %v360 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v493 = stablehlo.add %v491, %v492 : tensor<64x256x56x56xf32>
    %v494 = stablehlo.reshape %v493 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v495 = stablehlo.reshape %v494 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v496 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v497 = stablehlo.maximum %v495, %v496 : tensor<64x256x56x56xf32>
    %v498 = stablehlo.reshape %v497 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v499 = stablehlo.reshape %v498 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v500 = stablehlo.convert %v499 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v501 = stablehlo.convert %s2b0W1 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v502 = stablehlo.convolution(%v500, %v501)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<128x256x1x1xbf16>) -> tensor<64x128x56x56xbf16>
    %v503 = stablehlo.convert %v502 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v504 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v505 = stablehlo.add %v503, %v504 : tensor<64x128x56x56xf32>
    %v506 = stablehlo.reshape %v505 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v507 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v508 = stablehlo.constant dense<0.0> : tensor<f32>
    %v509 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v510 = stablehlo.reduce(%v507 init: %v508) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v511 = stablehlo.divide %v510, %v509 : tensor<128xf32>
    %arsums2b0g1mu = "stablehlo.all_reduce"(%v511) ({
    ^bb0(%aras2b0g1mu: tensor<f32>, %arbs2b0g1mu: tensor<f32>):
      %aradds2b0g1mu = stablehlo.add %aras2b0g1mu, %arbs2b0g1mu : tensor<f32>
      stablehlo.return %aradds2b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1mu = stablehlo.divide %arsums2b0g1mu, %arns2b0g1mu : tensor<128xf32>
    %v512 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v513 = stablehlo.constant dense<0.0> : tensor<f32>
    %v514 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v515 = stablehlo.reduce(%v512 init: %v513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v516 = stablehlo.divide %v515, %v514 : tensor<128xf32>
    %v517 = stablehlo.broadcast_in_dim %v516, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v518 = stablehlo.subtract %v512, %v517 : tensor<64x128x56x56xf32>
    %v519 = stablehlo.multiply %v518, %v518 : tensor<64x128x56x56xf32>
    %v520 = stablehlo.reduce(%v519 init: %v513) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v521 = stablehlo.divide %v520, %v514 : tensor<128xf32>
    %v522 = stablehlo.subtract %v516, %armeans2b0g1mu : tensor<128xf32>
    %v523 = stablehlo.multiply %v522, %v522 : tensor<128xf32>
    %v524 = stablehlo.add %v521, %v523 : tensor<128xf32>
    %arsums2b0g1var = "stablehlo.all_reduce"(%v524) ({
    ^bb0(%aras2b0g1var: tensor<f32>, %arbs2b0g1var: tensor<f32>):
      %aradds2b0g1var = stablehlo.add %aras2b0g1var, %arbs2b0g1var : tensor<f32>
      stablehlo.return %aradds2b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1var = stablehlo.divide %arsums2b0g1var, %arns2b0g1var : tensor<128xf32>
    %v525 = stablehlo.concatenate %armeans2b0g1mu, %armeans2b0g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v526 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v527 = stablehlo.slice %v525 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v528 = stablehlo.slice %v525 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v529 = stablehlo.broadcast_in_dim %v527, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v530 = stablehlo.broadcast_in_dim %v528, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v531 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v532 = stablehlo.add %v530, %v531 : tensor<64x128x56x56xf32>
    %v533 = stablehlo.rsqrt %v532 : tensor<64x128x56x56xf32>
    %v534 = stablehlo.subtract %v526, %v529 : tensor<64x128x56x56xf32>
    %v535 = stablehlo.multiply %v534, %v533 : tensor<64x128x56x56xf32>
    %v536 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v537 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v538 = stablehlo.multiply %v535, %v536 : tensor<64x128x56x56xf32>
    %v539 = stablehlo.add %v538, %v537 : tensor<64x128x56x56xf32>
    %v540 = stablehlo.reshape %v539 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v541 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v542 = stablehlo.maximum %v540, %v541 : tensor<64x401408xf32>
    %v543 = stablehlo.reshape %v542 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v544 = stablehlo.convert %v543 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v545 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v546 = stablehlo.convolution(%v544, %v545)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v547 = stablehlo.convert %v546 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v548 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<64x128x28x28xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<f32>
    %v553 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v554 = stablehlo.reduce(%v551 init: %v552) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v555 = stablehlo.divide %v554, %v553 : tensor<128xf32>
    %arsums2b0g2mu = "stablehlo.all_reduce"(%v555) ({
    ^bb0(%aras2b0g2mu: tensor<f32>, %arbs2b0g2mu: tensor<f32>):
      %aradds2b0g2mu = stablehlo.add %aras2b0g2mu, %arbs2b0g2mu : tensor<f32>
      stablehlo.return %aradds2b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2mu = stablehlo.divide %arsums2b0g2mu, %arns2b0g2mu : tensor<128xf32>
    %v556 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v558 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v559 = stablehlo.reduce(%v556 init: %v557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v560 = stablehlo.divide %v559, %v558 : tensor<128xf32>
    %v561 = stablehlo.broadcast_in_dim %v560, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v562 = stablehlo.subtract %v556, %v561 : tensor<64x128x28x28xf32>
    %v563 = stablehlo.multiply %v562, %v562 : tensor<64x128x28x28xf32>
    %v564 = stablehlo.reduce(%v563 init: %v557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v565 = stablehlo.divide %v564, %v558 : tensor<128xf32>
    %v566 = stablehlo.subtract %v560, %armeans2b0g2mu : tensor<128xf32>
    %v567 = stablehlo.multiply %v566, %v566 : tensor<128xf32>
    %v568 = stablehlo.add %v565, %v567 : tensor<128xf32>
    %arsums2b0g2var = "stablehlo.all_reduce"(%v568) ({
    ^bb0(%aras2b0g2var: tensor<f32>, %arbs2b0g2var: tensor<f32>):
      %aradds2b0g2var = stablehlo.add %aras2b0g2var, %arbs2b0g2var : tensor<f32>
      stablehlo.return %aradds2b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2var = stablehlo.divide %arsums2b0g2var, %arns2b0g2var : tensor<128xf32>
    %v569 = stablehlo.concatenate %armeans2b0g2mu, %armeans2b0g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v570 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v571 = stablehlo.slice %v569 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v572 = stablehlo.slice %v569 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v573 = stablehlo.broadcast_in_dim %v571, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v574 = stablehlo.broadcast_in_dim %v572, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v575 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v576 = stablehlo.add %v574, %v575 : tensor<64x128x28x28xf32>
    %v577 = stablehlo.rsqrt %v576 : tensor<64x128x28x28xf32>
    %v578 = stablehlo.subtract %v570, %v573 : tensor<64x128x28x28xf32>
    %v579 = stablehlo.multiply %v578, %v577 : tensor<64x128x28x28xf32>
    %v580 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v581 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v582 = stablehlo.multiply %v579, %v580 : tensor<64x128x28x28xf32>
    %v583 = stablehlo.add %v582, %v581 : tensor<64x128x28x28xf32>
    %v584 = stablehlo.reshape %v583 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v585 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v586 = stablehlo.maximum %v584, %v585 : tensor<64x100352xf32>
    %v587 = stablehlo.reshape %v586 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v588 = stablehlo.convert %v587 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v589 = stablehlo.convert %s2b0W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v590 = stablehlo.convolution(%v588, %v589)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v591 = stablehlo.convert %v590 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v592 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v593 = stablehlo.add %v591, %v592 : tensor<64x512x28x28xf32>
    %v594 = stablehlo.reshape %v593 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v595 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v596 = stablehlo.constant dense<0.0> : tensor<f32>
    %v597 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v598 = stablehlo.reduce(%v595 init: %v596) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v599 = stablehlo.divide %v598, %v597 : tensor<512xf32>
    %arsums2b0g3mu = "stablehlo.all_reduce"(%v599) ({
    ^bb0(%aras2b0g3mu: tensor<f32>, %arbs2b0g3mu: tensor<f32>):
      %aradds2b0g3mu = stablehlo.add %aras2b0g3mu, %arbs2b0g3mu : tensor<f32>
      stablehlo.return %aradds2b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3mu = stablehlo.divide %arsums2b0g3mu, %arns2b0g3mu : tensor<512xf32>
    %v600 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v601 = stablehlo.constant dense<0.0> : tensor<f32>
    %v602 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v603 = stablehlo.reduce(%v600 init: %v601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v604 = stablehlo.divide %v603, %v602 : tensor<512xf32>
    %v605 = stablehlo.broadcast_in_dim %v604, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v606 = stablehlo.subtract %v600, %v605 : tensor<64x512x28x28xf32>
    %v607 = stablehlo.multiply %v606, %v606 : tensor<64x512x28x28xf32>
    %v608 = stablehlo.reduce(%v607 init: %v601) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v609 = stablehlo.divide %v608, %v602 : tensor<512xf32>
    %v610 = stablehlo.subtract %v604, %armeans2b0g3mu : tensor<512xf32>
    %v611 = stablehlo.multiply %v610, %v610 : tensor<512xf32>
    %v612 = stablehlo.add %v609, %v611 : tensor<512xf32>
    %arsums2b0g3var = "stablehlo.all_reduce"(%v612) ({
    ^bb0(%aras2b0g3var: tensor<f32>, %arbs2b0g3var: tensor<f32>):
      %aradds2b0g3var = stablehlo.add %aras2b0g3var, %arbs2b0g3var : tensor<f32>
      stablehlo.return %aradds2b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3var = stablehlo.divide %arsums2b0g3var, %arns2b0g3var : tensor<512xf32>
    %v613 = stablehlo.concatenate %armeans2b0g3mu, %armeans2b0g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v614 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v615 = stablehlo.slice %v613 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v616 = stablehlo.slice %v613 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v617 = stablehlo.broadcast_in_dim %v615, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v618 = stablehlo.broadcast_in_dim %v616, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v619 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v620 = stablehlo.add %v618, %v619 : tensor<64x512x28x28xf32>
    %v621 = stablehlo.rsqrt %v620 : tensor<64x512x28x28xf32>
    %v622 = stablehlo.subtract %v614, %v617 : tensor<64x512x28x28xf32>
    %v623 = stablehlo.multiply %v622, %v621 : tensor<64x512x28x28xf32>
    %v624 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v625 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v626 = stablehlo.multiply %v623, %v624 : tensor<64x512x28x28xf32>
    %v627 = stablehlo.add %v626, %v625 : tensor<64x512x28x28xf32>
    %v628 = stablehlo.reshape %v627 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v629 = stablehlo.reshape %v498 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v630 = stablehlo.convert %v629 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v631 = stablehlo.convert %s2b0Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v632 = stablehlo.convolution(%v630, %v631)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v633 = stablehlo.convert %v632 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v634 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v635 = stablehlo.add %v633, %v634 : tensor<64x512x28x28xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v637 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v638 = stablehlo.constant dense<0.0> : tensor<f32>
    %v639 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v640 = stablehlo.reduce(%v637 init: %v638) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v641 = stablehlo.divide %v640, %v639 : tensor<512xf32>
    %arsums2b0gpmu = "stablehlo.all_reduce"(%v641) ({
    ^bb0(%aras2b0gpmu: tensor<f32>, %arbs2b0gpmu: tensor<f32>):
      %aradds2b0gpmu = stablehlo.add %aras2b0gpmu, %arbs2b0gpmu : tensor<f32>
      stablehlo.return %aradds2b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gpmu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gpmu = stablehlo.divide %arsums2b0gpmu, %arns2b0gpmu : tensor<512xf32>
    %v642 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v644 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v645 = stablehlo.reduce(%v642 init: %v643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v646 = stablehlo.divide %v645, %v644 : tensor<512xf32>
    %v647 = stablehlo.broadcast_in_dim %v646, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v648 = stablehlo.subtract %v642, %v647 : tensor<64x512x28x28xf32>
    %v649 = stablehlo.multiply %v648, %v648 : tensor<64x512x28x28xf32>
    %v650 = stablehlo.reduce(%v649 init: %v643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v651 = stablehlo.divide %v650, %v644 : tensor<512xf32>
    %v652 = stablehlo.subtract %v646, %armeans2b0gpmu : tensor<512xf32>
    %v653 = stablehlo.multiply %v652, %v652 : tensor<512xf32>
    %v654 = stablehlo.add %v651, %v653 : tensor<512xf32>
    %arsums2b0gpvar = "stablehlo.all_reduce"(%v654) ({
    ^bb0(%aras2b0gpvar: tensor<f32>, %arbs2b0gpvar: tensor<f32>):
      %aradds2b0gpvar = stablehlo.add %aras2b0gpvar, %arbs2b0gpvar : tensor<f32>
      stablehlo.return %aradds2b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gpvar = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gpvar = stablehlo.divide %arsums2b0gpvar, %arns2b0gpvar : tensor<512xf32>
    %v655 = stablehlo.concatenate %armeans2b0gpmu, %armeans2b0gpvar, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v656 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v657 = stablehlo.slice %v655 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v658 = stablehlo.slice %v655 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v659 = stablehlo.broadcast_in_dim %v657, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v660 = stablehlo.broadcast_in_dim %v658, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v661 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v662 = stablehlo.add %v660, %v661 : tensor<64x512x28x28xf32>
    %v663 = stablehlo.rsqrt %v662 : tensor<64x512x28x28xf32>
    %v664 = stablehlo.subtract %v656, %v659 : tensor<64x512x28x28xf32>
    %v665 = stablehlo.multiply %v664, %v663 : tensor<64x512x28x28xf32>
    %v666 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v667 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v668 = stablehlo.multiply %v665, %v666 : tensor<64x512x28x28xf32>
    %v669 = stablehlo.add %v668, %v667 : tensor<64x512x28x28xf32>
    %v670 = stablehlo.reshape %v669 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v671 = stablehlo.add %v628, %v670 : tensor<64x401408xf32>
    %v672 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v673 = stablehlo.maximum %v671, %v672 : tensor<64x401408xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v675 = stablehlo.convert %v674 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v676 = stablehlo.convert %s2b1W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v677 = stablehlo.convolution(%v675, %v676)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v678 = stablehlo.convert %v677 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v679 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v680 = stablehlo.add %v678, %v679 : tensor<64x128x28x28xf32>
    %v681 = stablehlo.reshape %v680 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v682 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v684 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v685 = stablehlo.reduce(%v682 init: %v683) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v686 = stablehlo.divide %v685, %v684 : tensor<128xf32>
    %arsums2b1g1mu = "stablehlo.all_reduce"(%v686) ({
    ^bb0(%aras2b1g1mu: tensor<f32>, %arbs2b1g1mu: tensor<f32>):
      %aradds2b1g1mu = stablehlo.add %aras2b1g1mu, %arbs2b1g1mu : tensor<f32>
      stablehlo.return %aradds2b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1mu = stablehlo.divide %arsums2b1g1mu, %arns2b1g1mu : tensor<128xf32>
    %v687 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v689 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v690 = stablehlo.reduce(%v687 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v691 = stablehlo.divide %v690, %v689 : tensor<128xf32>
    %v692 = stablehlo.broadcast_in_dim %v691, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v693 = stablehlo.subtract %v687, %v692 : tensor<64x128x28x28xf32>
    %v694 = stablehlo.multiply %v693, %v693 : tensor<64x128x28x28xf32>
    %v695 = stablehlo.reduce(%v694 init: %v688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v696 = stablehlo.divide %v695, %v689 : tensor<128xf32>
    %v697 = stablehlo.subtract %v691, %armeans2b1g1mu : tensor<128xf32>
    %v698 = stablehlo.multiply %v697, %v697 : tensor<128xf32>
    %v699 = stablehlo.add %v696, %v698 : tensor<128xf32>
    %arsums2b1g1var = "stablehlo.all_reduce"(%v699) ({
    ^bb0(%aras2b1g1var: tensor<f32>, %arbs2b1g1var: tensor<f32>):
      %aradds2b1g1var = stablehlo.add %aras2b1g1var, %arbs2b1g1var : tensor<f32>
      stablehlo.return %aradds2b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1var = stablehlo.divide %arsums2b1g1var, %arns2b1g1var : tensor<128xf32>
    %v700 = stablehlo.concatenate %armeans2b1g1mu, %armeans2b1g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v701 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v702 = stablehlo.slice %v700 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v703 = stablehlo.slice %v700 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v704 = stablehlo.broadcast_in_dim %v702, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v705 = stablehlo.broadcast_in_dim %v703, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v706 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v707 = stablehlo.add %v705, %v706 : tensor<64x128x28x28xf32>
    %v708 = stablehlo.rsqrt %v707 : tensor<64x128x28x28xf32>
    %v709 = stablehlo.subtract %v701, %v704 : tensor<64x128x28x28xf32>
    %v710 = stablehlo.multiply %v709, %v708 : tensor<64x128x28x28xf32>
    %v711 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v712 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v713 = stablehlo.multiply %v710, %v711 : tensor<64x128x28x28xf32>
    %v714 = stablehlo.add %v713, %v712 : tensor<64x128x28x28xf32>
    %v715 = stablehlo.reshape %v714 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v716 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v717 = stablehlo.maximum %v715, %v716 : tensor<64x100352xf32>
    %v718 = stablehlo.reshape %v717 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v719 = stablehlo.convert %v718 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v720 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v721 = stablehlo.convolution(%v719, %v720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v722 = stablehlo.convert %v721 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v723 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v724 = stablehlo.add %v722, %v723 : tensor<64x128x28x28xf32>
    %v725 = stablehlo.reshape %v724 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v726 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v727 = stablehlo.constant dense<0.0> : tensor<f32>
    %v728 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v729 = stablehlo.reduce(%v726 init: %v727) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v730 = stablehlo.divide %v729, %v728 : tensor<128xf32>
    %arsums2b1g2mu = "stablehlo.all_reduce"(%v730) ({
    ^bb0(%aras2b1g2mu: tensor<f32>, %arbs2b1g2mu: tensor<f32>):
      %aradds2b1g2mu = stablehlo.add %aras2b1g2mu, %arbs2b1g2mu : tensor<f32>
      stablehlo.return %aradds2b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2mu = stablehlo.divide %arsums2b1g2mu, %arns2b1g2mu : tensor<128xf32>
    %v731 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v733 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v734 = stablehlo.reduce(%v731 init: %v732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v735 = stablehlo.divide %v734, %v733 : tensor<128xf32>
    %v736 = stablehlo.broadcast_in_dim %v735, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v737 = stablehlo.subtract %v731, %v736 : tensor<64x128x28x28xf32>
    %v738 = stablehlo.multiply %v737, %v737 : tensor<64x128x28x28xf32>
    %v739 = stablehlo.reduce(%v738 init: %v732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v740 = stablehlo.divide %v739, %v733 : tensor<128xf32>
    %v741 = stablehlo.subtract %v735, %armeans2b1g2mu : tensor<128xf32>
    %v742 = stablehlo.multiply %v741, %v741 : tensor<128xf32>
    %v743 = stablehlo.add %v740, %v742 : tensor<128xf32>
    %arsums2b1g2var = "stablehlo.all_reduce"(%v743) ({
    ^bb0(%aras2b1g2var: tensor<f32>, %arbs2b1g2var: tensor<f32>):
      %aradds2b1g2var = stablehlo.add %aras2b1g2var, %arbs2b1g2var : tensor<f32>
      stablehlo.return %aradds2b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2var = stablehlo.divide %arsums2b1g2var, %arns2b1g2var : tensor<128xf32>
    %v744 = stablehlo.concatenate %armeans2b1g2mu, %armeans2b1g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v745 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v746 = stablehlo.slice %v744 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v747 = stablehlo.slice %v744 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v748 = stablehlo.broadcast_in_dim %v746, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v749 = stablehlo.broadcast_in_dim %v747, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v750 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v751 = stablehlo.add %v749, %v750 : tensor<64x128x28x28xf32>
    %v752 = stablehlo.rsqrt %v751 : tensor<64x128x28x28xf32>
    %v753 = stablehlo.subtract %v745, %v748 : tensor<64x128x28x28xf32>
    %v754 = stablehlo.multiply %v753, %v752 : tensor<64x128x28x28xf32>
    %v755 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v756 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v757 = stablehlo.multiply %v754, %v755 : tensor<64x128x28x28xf32>
    %v758 = stablehlo.add %v757, %v756 : tensor<64x128x28x28xf32>
    %v759 = stablehlo.reshape %v758 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v760 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v761 = stablehlo.maximum %v759, %v760 : tensor<64x100352xf32>
    %v762 = stablehlo.reshape %v761 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v763 = stablehlo.convert %v762 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v764 = stablehlo.convert %s2b1W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v765 = stablehlo.convolution(%v763, %v764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v766 = stablehlo.convert %v765 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v767 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v768 = stablehlo.add %v766, %v767 : tensor<64x512x28x28xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v770 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v771 = stablehlo.constant dense<0.0> : tensor<f32>
    %v772 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v773 = stablehlo.reduce(%v770 init: %v771) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v774 = stablehlo.divide %v773, %v772 : tensor<512xf32>
    %arsums2b1g3mu = "stablehlo.all_reduce"(%v774) ({
    ^bb0(%aras2b1g3mu: tensor<f32>, %arbs2b1g3mu: tensor<f32>):
      %aradds2b1g3mu = stablehlo.add %aras2b1g3mu, %arbs2b1g3mu : tensor<f32>
      stablehlo.return %aradds2b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3mu = stablehlo.divide %arsums2b1g3mu, %arns2b1g3mu : tensor<512xf32>
    %v775 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v776 = stablehlo.constant dense<0.0> : tensor<f32>
    %v777 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v778 = stablehlo.reduce(%v775 init: %v776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v779 = stablehlo.divide %v778, %v777 : tensor<512xf32>
    %v780 = stablehlo.broadcast_in_dim %v779, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v781 = stablehlo.subtract %v775, %v780 : tensor<64x512x28x28xf32>
    %v782 = stablehlo.multiply %v781, %v781 : tensor<64x512x28x28xf32>
    %v783 = stablehlo.reduce(%v782 init: %v776) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v784 = stablehlo.divide %v783, %v777 : tensor<512xf32>
    %v785 = stablehlo.subtract %v779, %armeans2b1g3mu : tensor<512xf32>
    %v786 = stablehlo.multiply %v785, %v785 : tensor<512xf32>
    %v787 = stablehlo.add %v784, %v786 : tensor<512xf32>
    %arsums2b1g3var = "stablehlo.all_reduce"(%v787) ({
    ^bb0(%aras2b1g3var: tensor<f32>, %arbs2b1g3var: tensor<f32>):
      %aradds2b1g3var = stablehlo.add %aras2b1g3var, %arbs2b1g3var : tensor<f32>
      stablehlo.return %aradds2b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3var = stablehlo.divide %arsums2b1g3var, %arns2b1g3var : tensor<512xf32>
    %v788 = stablehlo.concatenate %armeans2b1g3mu, %armeans2b1g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v789 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v790 = stablehlo.slice %v788 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v791 = stablehlo.slice %v788 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v792 = stablehlo.broadcast_in_dim %v790, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v793 = stablehlo.broadcast_in_dim %v791, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v794 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v795 = stablehlo.add %v793, %v794 : tensor<64x512x28x28xf32>
    %v796 = stablehlo.rsqrt %v795 : tensor<64x512x28x28xf32>
    %v797 = stablehlo.subtract %v789, %v792 : tensor<64x512x28x28xf32>
    %v798 = stablehlo.multiply %v797, %v796 : tensor<64x512x28x28xf32>
    %v799 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v800 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v801 = stablehlo.multiply %v798, %v799 : tensor<64x512x28x28xf32>
    %v802 = stablehlo.add %v801, %v800 : tensor<64x512x28x28xf32>
    %v803 = stablehlo.reshape %v802 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v804 = stablehlo.reshape %v803 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v805 = stablehlo.reshape %v673 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v806 = stablehlo.add %v804, %v805 : tensor<64x512x28x28xf32>
    %v807 = stablehlo.reshape %v806 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v808 = stablehlo.reshape %v807 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v809 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v810 = stablehlo.maximum %v808, %v809 : tensor<64x512x28x28xf32>
    %v811 = stablehlo.reshape %v810 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v812 = stablehlo.reshape %v811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v813 = stablehlo.convert %v812 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v814 = stablehlo.convert %s2b2W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v815 = stablehlo.convolution(%v813, %v814)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v816 = stablehlo.convert %v815 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v817 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v818 = stablehlo.add %v816, %v817 : tensor<64x128x28x28xf32>
    %v819 = stablehlo.reshape %v818 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v820 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v821 = stablehlo.constant dense<0.0> : tensor<f32>
    %v822 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v823 = stablehlo.reduce(%v820 init: %v821) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v824 = stablehlo.divide %v823, %v822 : tensor<128xf32>
    %arsums2b2g1mu = "stablehlo.all_reduce"(%v824) ({
    ^bb0(%aras2b2g1mu: tensor<f32>, %arbs2b2g1mu: tensor<f32>):
      %aradds2b2g1mu = stablehlo.add %aras2b2g1mu, %arbs2b2g1mu : tensor<f32>
      stablehlo.return %aradds2b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1mu = stablehlo.divide %arsums2b2g1mu, %arns2b2g1mu : tensor<128xf32>
    %v825 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v827 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v828 = stablehlo.reduce(%v825 init: %v826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v829 = stablehlo.divide %v828, %v827 : tensor<128xf32>
    %v830 = stablehlo.broadcast_in_dim %v829, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v831 = stablehlo.subtract %v825, %v830 : tensor<64x128x28x28xf32>
    %v832 = stablehlo.multiply %v831, %v831 : tensor<64x128x28x28xf32>
    %v833 = stablehlo.reduce(%v832 init: %v826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v834 = stablehlo.divide %v833, %v827 : tensor<128xf32>
    %v835 = stablehlo.subtract %v829, %armeans2b2g1mu : tensor<128xf32>
    %v836 = stablehlo.multiply %v835, %v835 : tensor<128xf32>
    %v837 = stablehlo.add %v834, %v836 : tensor<128xf32>
    %arsums2b2g1var = "stablehlo.all_reduce"(%v837) ({
    ^bb0(%aras2b2g1var: tensor<f32>, %arbs2b2g1var: tensor<f32>):
      %aradds2b2g1var = stablehlo.add %aras2b2g1var, %arbs2b2g1var : tensor<f32>
      stablehlo.return %aradds2b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1var = stablehlo.divide %arsums2b2g1var, %arns2b2g1var : tensor<128xf32>
    %v838 = stablehlo.concatenate %armeans2b2g1mu, %armeans2b2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v839 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v840 = stablehlo.slice %v838 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v841 = stablehlo.slice %v838 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v842 = stablehlo.broadcast_in_dim %v840, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v843 = stablehlo.broadcast_in_dim %v841, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v844 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v845 = stablehlo.add %v843, %v844 : tensor<64x128x28x28xf32>
    %v846 = stablehlo.rsqrt %v845 : tensor<64x128x28x28xf32>
    %v847 = stablehlo.subtract %v839, %v842 : tensor<64x128x28x28xf32>
    %v848 = stablehlo.multiply %v847, %v846 : tensor<64x128x28x28xf32>
    %v849 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v850 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v851 = stablehlo.multiply %v848, %v849 : tensor<64x128x28x28xf32>
    %v852 = stablehlo.add %v851, %v850 : tensor<64x128x28x28xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v854 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v855 = stablehlo.maximum %v853, %v854 : tensor<64x100352xf32>
    %v856 = stablehlo.reshape %v855 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v857 = stablehlo.convert %v856 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v858 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v859 = stablehlo.convolution(%v857, %v858)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v860 = stablehlo.convert %v859 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v861 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<64x128x28x28xf32>
    %v863 = stablehlo.reshape %v862 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v864 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v866 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v867 = stablehlo.reduce(%v864 init: %v865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v868 = stablehlo.divide %v867, %v866 : tensor<128xf32>
    %arsums2b2g2mu = "stablehlo.all_reduce"(%v868) ({
    ^bb0(%aras2b2g2mu: tensor<f32>, %arbs2b2g2mu: tensor<f32>):
      %aradds2b2g2mu = stablehlo.add %aras2b2g2mu, %arbs2b2g2mu : tensor<f32>
      stablehlo.return %aradds2b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2mu = stablehlo.divide %arsums2b2g2mu, %arns2b2g2mu : tensor<128xf32>
    %v869 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v871 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v872 = stablehlo.reduce(%v869 init: %v870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v873 = stablehlo.divide %v872, %v871 : tensor<128xf32>
    %v874 = stablehlo.broadcast_in_dim %v873, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v875 = stablehlo.subtract %v869, %v874 : tensor<64x128x28x28xf32>
    %v876 = stablehlo.multiply %v875, %v875 : tensor<64x128x28x28xf32>
    %v877 = stablehlo.reduce(%v876 init: %v870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v878 = stablehlo.divide %v877, %v871 : tensor<128xf32>
    %v879 = stablehlo.subtract %v873, %armeans2b2g2mu : tensor<128xf32>
    %v880 = stablehlo.multiply %v879, %v879 : tensor<128xf32>
    %v881 = stablehlo.add %v878, %v880 : tensor<128xf32>
    %arsums2b2g2var = "stablehlo.all_reduce"(%v881) ({
    ^bb0(%aras2b2g2var: tensor<f32>, %arbs2b2g2var: tensor<f32>):
      %aradds2b2g2var = stablehlo.add %aras2b2g2var, %arbs2b2g2var : tensor<f32>
      stablehlo.return %aradds2b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2var = stablehlo.divide %arsums2b2g2var, %arns2b2g2var : tensor<128xf32>
    %v882 = stablehlo.concatenate %armeans2b2g2mu, %armeans2b2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v883 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v884 = stablehlo.slice %v882 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v885 = stablehlo.slice %v882 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v886 = stablehlo.broadcast_in_dim %v884, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v887 = stablehlo.broadcast_in_dim %v885, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v888 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v889 = stablehlo.add %v887, %v888 : tensor<64x128x28x28xf32>
    %v890 = stablehlo.rsqrt %v889 : tensor<64x128x28x28xf32>
    %v891 = stablehlo.subtract %v883, %v886 : tensor<64x128x28x28xf32>
    %v892 = stablehlo.multiply %v891, %v890 : tensor<64x128x28x28xf32>
    %v893 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v894 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v895 = stablehlo.multiply %v892, %v893 : tensor<64x128x28x28xf32>
    %v896 = stablehlo.add %v895, %v894 : tensor<64x128x28x28xf32>
    %v897 = stablehlo.reshape %v896 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v898 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v899 = stablehlo.maximum %v897, %v898 : tensor<64x100352xf32>
    %v900 = stablehlo.reshape %v899 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v901 = stablehlo.convert %v900 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v902 = stablehlo.convert %s2b2W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v903 = stablehlo.convolution(%v901, %v902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v904 = stablehlo.convert %v903 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v905 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v906 = stablehlo.add %v904, %v905 : tensor<64x512x28x28xf32>
    %v907 = stablehlo.reshape %v906 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v908 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v909 = stablehlo.constant dense<0.0> : tensor<f32>
    %v910 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v911 = stablehlo.reduce(%v908 init: %v909) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v912 = stablehlo.divide %v911, %v910 : tensor<512xf32>
    %arsums2b2g3mu = "stablehlo.all_reduce"(%v912) ({
    ^bb0(%aras2b2g3mu: tensor<f32>, %arbs2b2g3mu: tensor<f32>):
      %aradds2b2g3mu = stablehlo.add %aras2b2g3mu, %arbs2b2g3mu : tensor<f32>
      stablehlo.return %aradds2b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3mu = stablehlo.divide %arsums2b2g3mu, %arns2b2g3mu : tensor<512xf32>
    %v913 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v914 = stablehlo.constant dense<0.0> : tensor<f32>
    %v915 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v916 = stablehlo.reduce(%v913 init: %v914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v917 = stablehlo.divide %v916, %v915 : tensor<512xf32>
    %v918 = stablehlo.broadcast_in_dim %v917, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v919 = stablehlo.subtract %v913, %v918 : tensor<64x512x28x28xf32>
    %v920 = stablehlo.multiply %v919, %v919 : tensor<64x512x28x28xf32>
    %v921 = stablehlo.reduce(%v920 init: %v914) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v922 = stablehlo.divide %v921, %v915 : tensor<512xf32>
    %v923 = stablehlo.subtract %v917, %armeans2b2g3mu : tensor<512xf32>
    %v924 = stablehlo.multiply %v923, %v923 : tensor<512xf32>
    %v925 = stablehlo.add %v922, %v924 : tensor<512xf32>
    %arsums2b2g3var = "stablehlo.all_reduce"(%v925) ({
    ^bb0(%aras2b2g3var: tensor<f32>, %arbs2b2g3var: tensor<f32>):
      %aradds2b2g3var = stablehlo.add %aras2b2g3var, %arbs2b2g3var : tensor<f32>
      stablehlo.return %aradds2b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3var = stablehlo.divide %arsums2b2g3var, %arns2b2g3var : tensor<512xf32>
    %v926 = stablehlo.concatenate %armeans2b2g3mu, %armeans2b2g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v927 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v928 = stablehlo.slice %v926 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v929 = stablehlo.slice %v926 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v930 = stablehlo.broadcast_in_dim %v928, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v931 = stablehlo.broadcast_in_dim %v929, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v932 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v933 = stablehlo.add %v931, %v932 : tensor<64x512x28x28xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<64x512x28x28xf32>
    %v935 = stablehlo.subtract %v927, %v930 : tensor<64x512x28x28xf32>
    %v936 = stablehlo.multiply %v935, %v934 : tensor<64x512x28x28xf32>
    %v937 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v938 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v939 = stablehlo.multiply %v936, %v937 : tensor<64x512x28x28xf32>
    %v940 = stablehlo.add %v939, %v938 : tensor<64x512x28x28xf32>
    %v941 = stablehlo.reshape %v940 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v942 = stablehlo.reshape %v941 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v943 = stablehlo.reshape %v811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v944 = stablehlo.add %v942, %v943 : tensor<64x512x28x28xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v946 = stablehlo.reshape %v945 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v947 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v948 = stablehlo.maximum %v946, %v947 : tensor<64x512x28x28xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v950 = stablehlo.reshape %v949 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v951 = stablehlo.convert %v950 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v952 = stablehlo.convert %s2b3W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v953 = stablehlo.convolution(%v951, %v952)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v954 = stablehlo.convert %v953 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v955 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v956 = stablehlo.add %v954, %v955 : tensor<64x128x28x28xf32>
    %v957 = stablehlo.reshape %v956 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v958 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v960 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v961 = stablehlo.reduce(%v958 init: %v959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v962 = stablehlo.divide %v961, %v960 : tensor<128xf32>
    %arsums2b3g1mu = "stablehlo.all_reduce"(%v962) ({
    ^bb0(%aras2b3g1mu: tensor<f32>, %arbs2b3g1mu: tensor<f32>):
      %aradds2b3g1mu = stablehlo.add %aras2b3g1mu, %arbs2b3g1mu : tensor<f32>
      stablehlo.return %aradds2b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1mu = stablehlo.divide %arsums2b3g1mu, %arns2b3g1mu : tensor<128xf32>
    %v963 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v965 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v966 = stablehlo.reduce(%v963 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v967 = stablehlo.divide %v966, %v965 : tensor<128xf32>
    %v968 = stablehlo.broadcast_in_dim %v967, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v969 = stablehlo.subtract %v963, %v968 : tensor<64x128x28x28xf32>
    %v970 = stablehlo.multiply %v969, %v969 : tensor<64x128x28x28xf32>
    %v971 = stablehlo.reduce(%v970 init: %v964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v972 = stablehlo.divide %v971, %v965 : tensor<128xf32>
    %v973 = stablehlo.subtract %v967, %armeans2b3g1mu : tensor<128xf32>
    %v974 = stablehlo.multiply %v973, %v973 : tensor<128xf32>
    %v975 = stablehlo.add %v972, %v974 : tensor<128xf32>
    %arsums2b3g1var = "stablehlo.all_reduce"(%v975) ({
    ^bb0(%aras2b3g1var: tensor<f32>, %arbs2b3g1var: tensor<f32>):
      %aradds2b3g1var = stablehlo.add %aras2b3g1var, %arbs2b3g1var : tensor<f32>
      stablehlo.return %aradds2b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1var = stablehlo.divide %arsums2b3g1var, %arns2b3g1var : tensor<128xf32>
    %v976 = stablehlo.concatenate %armeans2b3g1mu, %armeans2b3g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v977 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v978 = stablehlo.slice %v976 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v979 = stablehlo.slice %v976 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v980 = stablehlo.broadcast_in_dim %v978, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v981 = stablehlo.broadcast_in_dim %v979, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v982 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v983 = stablehlo.add %v981, %v982 : tensor<64x128x28x28xf32>
    %v984 = stablehlo.rsqrt %v983 : tensor<64x128x28x28xf32>
    %v985 = stablehlo.subtract %v977, %v980 : tensor<64x128x28x28xf32>
    %v986 = stablehlo.multiply %v985, %v984 : tensor<64x128x28x28xf32>
    %v987 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v988 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v989 = stablehlo.multiply %v986, %v987 : tensor<64x128x28x28xf32>
    %v990 = stablehlo.add %v989, %v988 : tensor<64x128x28x28xf32>
    %v991 = stablehlo.reshape %v990 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v992 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v993 = stablehlo.maximum %v991, %v992 : tensor<64x100352xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v995 = stablehlo.convert %v994 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v996 = stablehlo.convert %s2b3W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v997 = stablehlo.convolution(%v995, %v996)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v998 = stablehlo.convert %v997 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v999 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1000 = stablehlo.add %v998, %v999 : tensor<64x128x28x28xf32>
    %v1001 = stablehlo.reshape %v1000 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1003 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1004 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v1005 = stablehlo.reduce(%v1002 init: %v1003) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1006 = stablehlo.divide %v1005, %v1004 : tensor<128xf32>
    %arsums2b3g2mu = "stablehlo.all_reduce"(%v1006) ({
    ^bb0(%aras2b3g2mu: tensor<f32>, %arbs2b3g2mu: tensor<f32>):
      %aradds2b3g2mu = stablehlo.add %aras2b3g2mu, %arbs2b3g2mu : tensor<f32>
      stablehlo.return %aradds2b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2mu = stablehlo.divide %arsums2b3g2mu, %arns2b3g2mu : tensor<128xf32>
    %v1007 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1008 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1009 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v1010 = stablehlo.reduce(%v1007 init: %v1008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1011 = stablehlo.divide %v1010, %v1009 : tensor<128xf32>
    %v1012 = stablehlo.broadcast_in_dim %v1011, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1013 = stablehlo.subtract %v1007, %v1012 : tensor<64x128x28x28xf32>
    %v1014 = stablehlo.multiply %v1013, %v1013 : tensor<64x128x28x28xf32>
    %v1015 = stablehlo.reduce(%v1014 init: %v1008) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v1016 = stablehlo.divide %v1015, %v1009 : tensor<128xf32>
    %v1017 = stablehlo.subtract %v1011, %armeans2b3g2mu : tensor<128xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<128xf32>
    %v1019 = stablehlo.add %v1016, %v1018 : tensor<128xf32>
    %arsums2b3g2var = "stablehlo.all_reduce"(%v1019) ({
    ^bb0(%aras2b3g2var: tensor<f32>, %arbs2b3g2var: tensor<f32>):
      %aradds2b3g2var = stablehlo.add %aras2b3g2var, %arbs2b3g2var : tensor<f32>
      stablehlo.return %aradds2b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2var = stablehlo.divide %arsums2b3g2var, %arns2b3g2var : tensor<128xf32>
    %v1020 = stablehlo.concatenate %armeans2b3g2mu, %armeans2b3g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v1021 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1022 = stablehlo.slice %v1020 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v1023 = stablehlo.slice %v1020 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v1024 = stablehlo.broadcast_in_dim %v1022, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1025 = stablehlo.broadcast_in_dim %v1023, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1026 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v1027 = stablehlo.add %v1025, %v1026 : tensor<64x128x28x28xf32>
    %v1028 = stablehlo.rsqrt %v1027 : tensor<64x128x28x28xf32>
    %v1029 = stablehlo.subtract %v1021, %v1024 : tensor<64x128x28x28xf32>
    %v1030 = stablehlo.multiply %v1029, %v1028 : tensor<64x128x28x28xf32>
    %v1031 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1032 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v1033 = stablehlo.multiply %v1030, %v1031 : tensor<64x128x28x28xf32>
    %v1034 = stablehlo.add %v1033, %v1032 : tensor<64x128x28x28xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v1036 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1037 = stablehlo.maximum %v1035, %v1036 : tensor<64x100352xf32>
    %v1038 = stablehlo.reshape %v1037 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v1039 = stablehlo.convert %v1038 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v1040 = stablehlo.convert %s2b3W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v1041 = stablehlo.convolution(%v1039, %v1040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v1042 = stablehlo.convert %v1041 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v1043 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1044 = stablehlo.add %v1042, %v1043 : tensor<64x512x28x28xf32>
    %v1045 = stablehlo.reshape %v1044 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1046 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1047 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1048 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v1049 = stablehlo.reduce(%v1046 init: %v1047) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v1050 = stablehlo.divide %v1049, %v1048 : tensor<512xf32>
    %arsums2b3g3mu = "stablehlo.all_reduce"(%v1050) ({
    ^bb0(%aras2b3g3mu: tensor<f32>, %arbs2b3g3mu: tensor<f32>):
      %aradds2b3g3mu = stablehlo.add %aras2b3g3mu, %arbs2b3g3mu : tensor<f32>
      stablehlo.return %aradds2b3g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3mu = stablehlo.divide %arsums2b3g3mu, %arns2b3g3mu : tensor<512xf32>
    %v1051 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1052 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1053 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v1054 = stablehlo.reduce(%v1051 init: %v1052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v1055 = stablehlo.divide %v1054, %v1053 : tensor<512xf32>
    %v1056 = stablehlo.broadcast_in_dim %v1055, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1057 = stablehlo.subtract %v1051, %v1056 : tensor<64x512x28x28xf32>
    %v1058 = stablehlo.multiply %v1057, %v1057 : tensor<64x512x28x28xf32>
    %v1059 = stablehlo.reduce(%v1058 init: %v1052) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v1060 = stablehlo.divide %v1059, %v1053 : tensor<512xf32>
    %v1061 = stablehlo.subtract %v1055, %armeans2b3g3mu : tensor<512xf32>
    %v1062 = stablehlo.multiply %v1061, %v1061 : tensor<512xf32>
    %v1063 = stablehlo.add %v1060, %v1062 : tensor<512xf32>
    %arsums2b3g3var = "stablehlo.all_reduce"(%v1063) ({
    ^bb0(%aras2b3g3var: tensor<f32>, %arbs2b3g3var: tensor<f32>):
      %aradds2b3g3var = stablehlo.add %aras2b3g3var, %arbs2b3g3var : tensor<f32>
      stablehlo.return %aradds2b3g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3var = stablehlo.divide %arsums2b3g3var, %arns2b3g3var : tensor<512xf32>
    %v1064 = stablehlo.concatenate %armeans2b3g3mu, %armeans2b3g3var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1065 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1066 = stablehlo.slice %v1064 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1067 = stablehlo.slice %v1064 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1068 = stablehlo.broadcast_in_dim %v1066, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1069 = stablehlo.broadcast_in_dim %v1067, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1070 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v1071 = stablehlo.add %v1069, %v1070 : tensor<64x512x28x28xf32>
    %v1072 = stablehlo.rsqrt %v1071 : tensor<64x512x28x28xf32>
    %v1073 = stablehlo.subtract %v1065, %v1068 : tensor<64x512x28x28xf32>
    %v1074 = stablehlo.multiply %v1073, %v1072 : tensor<64x512x28x28xf32>
    %v1075 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1076 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v1077 = stablehlo.multiply %v1074, %v1075 : tensor<64x512x28x28xf32>
    %v1078 = stablehlo.add %v1077, %v1076 : tensor<64x512x28x28xf32>
    %v1079 = stablehlo.reshape %v1078 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1080 = stablehlo.reshape %v1079 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1081 = stablehlo.reshape %v949 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1082 = stablehlo.add %v1080, %v1081 : tensor<64x512x28x28xf32>
    %v1083 = stablehlo.reshape %v1082 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1084 = stablehlo.reshape %v1083 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1085 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v1086 = stablehlo.maximum %v1084, %v1085 : tensor<64x512x28x28xf32>
    %v1087 = stablehlo.reshape %v1086 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1089 = stablehlo.convert %v1088 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v1090 = stablehlo.convert %s3b0W1 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v1091 = stablehlo.convolution(%v1089, %v1090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x28x28xbf16>
    %v1092 = stablehlo.convert %v1091 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v1093 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<64x256x28x28xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1098 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v1099 = stablehlo.reduce(%v1096 init: %v1097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1100 = stablehlo.divide %v1099, %v1098 : tensor<256xf32>
    %arsums3b0g1mu = "stablehlo.all_reduce"(%v1100) ({
    ^bb0(%aras3b0g1mu: tensor<f32>, %arbs3b0g1mu: tensor<f32>):
      %aradds3b0g1mu = stablehlo.add %aras3b0g1mu, %arbs3b0g1mu : tensor<f32>
      stablehlo.return %aradds3b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1mu = stablehlo.divide %arsums3b0g1mu, %arns3b0g1mu : tensor<256xf32>
    %v1101 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1102 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1103 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v1104 = stablehlo.reduce(%v1101 init: %v1102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1105 = stablehlo.divide %v1104, %v1103 : tensor<256xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1107 = stablehlo.subtract %v1101, %v1106 : tensor<64x256x28x28xf32>
    %v1108 = stablehlo.multiply %v1107, %v1107 : tensor<64x256x28x28xf32>
    %v1109 = stablehlo.reduce(%v1108 init: %v1102) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v1110 = stablehlo.divide %v1109, %v1103 : tensor<256xf32>
    %v1111 = stablehlo.subtract %v1105, %armeans3b0g1mu : tensor<256xf32>
    %v1112 = stablehlo.multiply %v1111, %v1111 : tensor<256xf32>
    %v1113 = stablehlo.add %v1110, %v1112 : tensor<256xf32>
    %arsums3b0g1var = "stablehlo.all_reduce"(%v1113) ({
    ^bb0(%aras3b0g1var: tensor<f32>, %arbs3b0g1var: tensor<f32>):
      %aradds3b0g1var = stablehlo.add %aras3b0g1var, %arbs3b0g1var : tensor<f32>
      stablehlo.return %aradds3b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1var = stablehlo.divide %arsums3b0g1var, %arns3b0g1var : tensor<256xf32>
    %v1114 = stablehlo.concatenate %armeans3b0g1mu, %armeans3b0g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1115 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1116 = stablehlo.slice %v1114 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1117 = stablehlo.slice %v1114 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1118 = stablehlo.broadcast_in_dim %v1116, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1119 = stablehlo.broadcast_in_dim %v1117, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1120 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v1121 = stablehlo.add %v1119, %v1120 : tensor<64x256x28x28xf32>
    %v1122 = stablehlo.rsqrt %v1121 : tensor<64x256x28x28xf32>
    %v1123 = stablehlo.subtract %v1115, %v1118 : tensor<64x256x28x28xf32>
    %v1124 = stablehlo.multiply %v1123, %v1122 : tensor<64x256x28x28xf32>
    %v1125 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1126 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v1127 = stablehlo.multiply %v1124, %v1125 : tensor<64x256x28x28xf32>
    %v1128 = stablehlo.add %v1127, %v1126 : tensor<64x256x28x28xf32>
    %v1129 = stablehlo.reshape %v1128 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v1130 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1131 = stablehlo.maximum %v1129, %v1130 : tensor<64x200704xf32>
    %v1132 = stablehlo.reshape %v1131 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v1133 = stablehlo.convert %v1132 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v1134 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1135 = stablehlo.convolution(%v1133, %v1134)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1136 = stablehlo.convert %v1135 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1137 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1138 = stablehlo.add %v1136, %v1137 : tensor<64x256x14x14xf32>
    %v1139 = stablehlo.reshape %v1138 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1140 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1142 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1143 = stablehlo.reduce(%v1140 init: %v1141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1144 = stablehlo.divide %v1143, %v1142 : tensor<256xf32>
    %arsums3b0g2mu = "stablehlo.all_reduce"(%v1144) ({
    ^bb0(%aras3b0g2mu: tensor<f32>, %arbs3b0g2mu: tensor<f32>):
      %aradds3b0g2mu = stablehlo.add %aras3b0g2mu, %arbs3b0g2mu : tensor<f32>
      stablehlo.return %aradds3b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2mu = stablehlo.divide %arsums3b0g2mu, %arns3b0g2mu : tensor<256xf32>
    %v1145 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1147 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1148 = stablehlo.reduce(%v1145 init: %v1146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1149 = stablehlo.divide %v1148, %v1147 : tensor<256xf32>
    %v1150 = stablehlo.broadcast_in_dim %v1149, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1151 = stablehlo.subtract %v1145, %v1150 : tensor<64x256x14x14xf32>
    %v1152 = stablehlo.multiply %v1151, %v1151 : tensor<64x256x14x14xf32>
    %v1153 = stablehlo.reduce(%v1152 init: %v1146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1154 = stablehlo.divide %v1153, %v1147 : tensor<256xf32>
    %v1155 = stablehlo.subtract %v1149, %armeans3b0g2mu : tensor<256xf32>
    %v1156 = stablehlo.multiply %v1155, %v1155 : tensor<256xf32>
    %v1157 = stablehlo.add %v1154, %v1156 : tensor<256xf32>
    %arsums3b0g2var = "stablehlo.all_reduce"(%v1157) ({
    ^bb0(%aras3b0g2var: tensor<f32>, %arbs3b0g2var: tensor<f32>):
      %aradds3b0g2var = stablehlo.add %aras3b0g2var, %arbs3b0g2var : tensor<f32>
      stablehlo.return %aradds3b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2var = stablehlo.divide %arsums3b0g2var, %arns3b0g2var : tensor<256xf32>
    %v1158 = stablehlo.concatenate %armeans3b0g2mu, %armeans3b0g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1159 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1160 = stablehlo.slice %v1158 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1161 = stablehlo.slice %v1158 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1162 = stablehlo.broadcast_in_dim %v1160, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1163 = stablehlo.broadcast_in_dim %v1161, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1164 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1165 = stablehlo.add %v1163, %v1164 : tensor<64x256x14x14xf32>
    %v1166 = stablehlo.rsqrt %v1165 : tensor<64x256x14x14xf32>
    %v1167 = stablehlo.subtract %v1159, %v1162 : tensor<64x256x14x14xf32>
    %v1168 = stablehlo.multiply %v1167, %v1166 : tensor<64x256x14x14xf32>
    %v1169 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1170 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1171 = stablehlo.multiply %v1168, %v1169 : tensor<64x256x14x14xf32>
    %v1172 = stablehlo.add %v1171, %v1170 : tensor<64x256x14x14xf32>
    %v1173 = stablehlo.reshape %v1172 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1174 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1175 = stablehlo.maximum %v1173, %v1174 : tensor<64x50176xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1177 = stablehlo.convert %v1176 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1178 = stablehlo.convert %s3b0W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1179 = stablehlo.convolution(%v1177, %v1178)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1180 = stablehlo.convert %v1179 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1181 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1182 = stablehlo.add %v1180, %v1181 : tensor<64x1024x14x14xf32>
    %v1183 = stablehlo.reshape %v1182 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1184 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1186 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1187 = stablehlo.reduce(%v1184 init: %v1185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1188 = stablehlo.divide %v1187, %v1186 : tensor<1024xf32>
    %arsums3b0g3mu = "stablehlo.all_reduce"(%v1188) ({
    ^bb0(%aras3b0g3mu: tensor<f32>, %arbs3b0g3mu: tensor<f32>):
      %aradds3b0g3mu = stablehlo.add %aras3b0g3mu, %arbs3b0g3mu : tensor<f32>
      stablehlo.return %aradds3b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3mu = stablehlo.divide %arsums3b0g3mu, %arns3b0g3mu : tensor<1024xf32>
    %v1189 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1190 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1191 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1192 = stablehlo.reduce(%v1189 init: %v1190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1193 = stablehlo.divide %v1192, %v1191 : tensor<1024xf32>
    %v1194 = stablehlo.broadcast_in_dim %v1193, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1195 = stablehlo.subtract %v1189, %v1194 : tensor<64x1024x14x14xf32>
    %v1196 = stablehlo.multiply %v1195, %v1195 : tensor<64x1024x14x14xf32>
    %v1197 = stablehlo.reduce(%v1196 init: %v1190) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1198 = stablehlo.divide %v1197, %v1191 : tensor<1024xf32>
    %v1199 = stablehlo.subtract %v1193, %armeans3b0g3mu : tensor<1024xf32>
    %v1200 = stablehlo.multiply %v1199, %v1199 : tensor<1024xf32>
    %v1201 = stablehlo.add %v1198, %v1200 : tensor<1024xf32>
    %arsums3b0g3var = "stablehlo.all_reduce"(%v1201) ({
    ^bb0(%aras3b0g3var: tensor<f32>, %arbs3b0g3var: tensor<f32>):
      %aradds3b0g3var = stablehlo.add %aras3b0g3var, %arbs3b0g3var : tensor<f32>
      stablehlo.return %aradds3b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3var = stablehlo.divide %arsums3b0g3var, %arns3b0g3var : tensor<1024xf32>
    %v1202 = stablehlo.concatenate %armeans3b0g3mu, %armeans3b0g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1203 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1204 = stablehlo.slice %v1202 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1205 = stablehlo.slice %v1202 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1206 = stablehlo.broadcast_in_dim %v1204, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1207 = stablehlo.broadcast_in_dim %v1205, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1208 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1209 = stablehlo.add %v1207, %v1208 : tensor<64x1024x14x14xf32>
    %v1210 = stablehlo.rsqrt %v1209 : tensor<64x1024x14x14xf32>
    %v1211 = stablehlo.subtract %v1203, %v1206 : tensor<64x1024x14x14xf32>
    %v1212 = stablehlo.multiply %v1211, %v1210 : tensor<64x1024x14x14xf32>
    %v1213 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1214 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1215 = stablehlo.multiply %v1212, %v1213 : tensor<64x1024x14x14xf32>
    %v1216 = stablehlo.add %v1215, %v1214 : tensor<64x1024x14x14xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1218 = stablehlo.reshape %v1087 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v1219 = stablehlo.convert %v1218 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v1220 = stablehlo.convert %s3b0Wp : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v1221 = stablehlo.convolution(%v1219, %v1220)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1222 = stablehlo.convert %v1221 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1223 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1224 = stablehlo.add %v1222, %v1223 : tensor<64x1024x14x14xf32>
    %v1225 = stablehlo.reshape %v1224 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1226 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1228 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1229 = stablehlo.reduce(%v1226 init: %v1227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1230 = stablehlo.divide %v1229, %v1228 : tensor<1024xf32>
    %arsums3b0gpmu = "stablehlo.all_reduce"(%v1230) ({
    ^bb0(%aras3b0gpmu: tensor<f32>, %arbs3b0gpmu: tensor<f32>):
      %aradds3b0gpmu = stablehlo.add %aras3b0gpmu, %arbs3b0gpmu : tensor<f32>
      stablehlo.return %aradds3b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gpmu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gpmu = stablehlo.divide %arsums3b0gpmu, %arns3b0gpmu : tensor<1024xf32>
    %v1231 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1233 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1234 = stablehlo.reduce(%v1231 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1235 = stablehlo.divide %v1234, %v1233 : tensor<1024xf32>
    %v1236 = stablehlo.broadcast_in_dim %v1235, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1237 = stablehlo.subtract %v1231, %v1236 : tensor<64x1024x14x14xf32>
    %v1238 = stablehlo.multiply %v1237, %v1237 : tensor<64x1024x14x14xf32>
    %v1239 = stablehlo.reduce(%v1238 init: %v1232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1240 = stablehlo.divide %v1239, %v1233 : tensor<1024xf32>
    %v1241 = stablehlo.subtract %v1235, %armeans3b0gpmu : tensor<1024xf32>
    %v1242 = stablehlo.multiply %v1241, %v1241 : tensor<1024xf32>
    %v1243 = stablehlo.add %v1240, %v1242 : tensor<1024xf32>
    %arsums3b0gpvar = "stablehlo.all_reduce"(%v1243) ({
    ^bb0(%aras3b0gpvar: tensor<f32>, %arbs3b0gpvar: tensor<f32>):
      %aradds3b0gpvar = stablehlo.add %aras3b0gpvar, %arbs3b0gpvar : tensor<f32>
      stablehlo.return %aradds3b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gpvar = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gpvar = stablehlo.divide %arsums3b0gpvar, %arns3b0gpvar : tensor<1024xf32>
    %v1244 = stablehlo.concatenate %armeans3b0gpmu, %armeans3b0gpvar, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1245 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1246 = stablehlo.slice %v1244 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1247 = stablehlo.slice %v1244 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1248 = stablehlo.broadcast_in_dim %v1246, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1249 = stablehlo.broadcast_in_dim %v1247, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1250 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1251 = stablehlo.add %v1249, %v1250 : tensor<64x1024x14x14xf32>
    %v1252 = stablehlo.rsqrt %v1251 : tensor<64x1024x14x14xf32>
    %v1253 = stablehlo.subtract %v1245, %v1248 : tensor<64x1024x14x14xf32>
    %v1254 = stablehlo.multiply %v1253, %v1252 : tensor<64x1024x14x14xf32>
    %v1255 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1256 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1257 = stablehlo.multiply %v1254, %v1255 : tensor<64x1024x14x14xf32>
    %v1258 = stablehlo.add %v1257, %v1256 : tensor<64x1024x14x14xf32>
    %v1259 = stablehlo.reshape %v1258 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1260 = stablehlo.add %v1217, %v1259 : tensor<64x200704xf32>
    %v1261 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1262 = stablehlo.maximum %v1260, %v1261 : tensor<64x200704xf32>
    %v1263 = stablehlo.reshape %v1262 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1264 = stablehlo.convert %v1263 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1265 = stablehlo.convert %s3b1W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1266 = stablehlo.convolution(%v1264, %v1265)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1267 = stablehlo.convert %v1266 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1268 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1269 = stablehlo.add %v1267, %v1268 : tensor<64x256x14x14xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1271 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1273 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1274 = stablehlo.reduce(%v1271 init: %v1272) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1275 = stablehlo.divide %v1274, %v1273 : tensor<256xf32>
    %arsums3b1g1mu = "stablehlo.all_reduce"(%v1275) ({
    ^bb0(%aras3b1g1mu: tensor<f32>, %arbs3b1g1mu: tensor<f32>):
      %aradds3b1g1mu = stablehlo.add %aras3b1g1mu, %arbs3b1g1mu : tensor<f32>
      stablehlo.return %aradds3b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1mu = stablehlo.divide %arsums3b1g1mu, %arns3b1g1mu : tensor<256xf32>
    %v1276 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1278 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1279 = stablehlo.reduce(%v1276 init: %v1277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1280 = stablehlo.divide %v1279, %v1278 : tensor<256xf32>
    %v1281 = stablehlo.broadcast_in_dim %v1280, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1282 = stablehlo.subtract %v1276, %v1281 : tensor<64x256x14x14xf32>
    %v1283 = stablehlo.multiply %v1282, %v1282 : tensor<64x256x14x14xf32>
    %v1284 = stablehlo.reduce(%v1283 init: %v1277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1285 = stablehlo.divide %v1284, %v1278 : tensor<256xf32>
    %v1286 = stablehlo.subtract %v1280, %armeans3b1g1mu : tensor<256xf32>
    %v1287 = stablehlo.multiply %v1286, %v1286 : tensor<256xf32>
    %v1288 = stablehlo.add %v1285, %v1287 : tensor<256xf32>
    %arsums3b1g1var = "stablehlo.all_reduce"(%v1288) ({
    ^bb0(%aras3b1g1var: tensor<f32>, %arbs3b1g1var: tensor<f32>):
      %aradds3b1g1var = stablehlo.add %aras3b1g1var, %arbs3b1g1var : tensor<f32>
      stablehlo.return %aradds3b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1var = stablehlo.divide %arsums3b1g1var, %arns3b1g1var : tensor<256xf32>
    %v1289 = stablehlo.concatenate %armeans3b1g1mu, %armeans3b1g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1290 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1291 = stablehlo.slice %v1289 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1292 = stablehlo.slice %v1289 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1293 = stablehlo.broadcast_in_dim %v1291, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1294 = stablehlo.broadcast_in_dim %v1292, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1295 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1296 = stablehlo.add %v1294, %v1295 : tensor<64x256x14x14xf32>
    %v1297 = stablehlo.rsqrt %v1296 : tensor<64x256x14x14xf32>
    %v1298 = stablehlo.subtract %v1290, %v1293 : tensor<64x256x14x14xf32>
    %v1299 = stablehlo.multiply %v1298, %v1297 : tensor<64x256x14x14xf32>
    %v1300 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1301 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1302 = stablehlo.multiply %v1299, %v1300 : tensor<64x256x14x14xf32>
    %v1303 = stablehlo.add %v1302, %v1301 : tensor<64x256x14x14xf32>
    %v1304 = stablehlo.reshape %v1303 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1305 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1306 = stablehlo.maximum %v1304, %v1305 : tensor<64x50176xf32>
    %v1307 = stablehlo.reshape %v1306 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1308 = stablehlo.convert %v1307 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1309 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1310 = stablehlo.convolution(%v1308, %v1309)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1311 = stablehlo.convert %v1310 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1312 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1313 = stablehlo.add %v1311, %v1312 : tensor<64x256x14x14xf32>
    %v1314 = stablehlo.reshape %v1313 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1315 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1316 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1317 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1318 = stablehlo.reduce(%v1315 init: %v1316) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1319 = stablehlo.divide %v1318, %v1317 : tensor<256xf32>
    %arsums3b1g2mu = "stablehlo.all_reduce"(%v1319) ({
    ^bb0(%aras3b1g2mu: tensor<f32>, %arbs3b1g2mu: tensor<f32>):
      %aradds3b1g2mu = stablehlo.add %aras3b1g2mu, %arbs3b1g2mu : tensor<f32>
      stablehlo.return %aradds3b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2mu = stablehlo.divide %arsums3b1g2mu, %arns3b1g2mu : tensor<256xf32>
    %v1320 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1322 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1323 = stablehlo.reduce(%v1320 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1324 = stablehlo.divide %v1323, %v1322 : tensor<256xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1324, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1326 = stablehlo.subtract %v1320, %v1325 : tensor<64x256x14x14xf32>
    %v1327 = stablehlo.multiply %v1326, %v1326 : tensor<64x256x14x14xf32>
    %v1328 = stablehlo.reduce(%v1327 init: %v1321) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1329 = stablehlo.divide %v1328, %v1322 : tensor<256xf32>
    %v1330 = stablehlo.subtract %v1324, %armeans3b1g2mu : tensor<256xf32>
    %v1331 = stablehlo.multiply %v1330, %v1330 : tensor<256xf32>
    %v1332 = stablehlo.add %v1329, %v1331 : tensor<256xf32>
    %arsums3b1g2var = "stablehlo.all_reduce"(%v1332) ({
    ^bb0(%aras3b1g2var: tensor<f32>, %arbs3b1g2var: tensor<f32>):
      %aradds3b1g2var = stablehlo.add %aras3b1g2var, %arbs3b1g2var : tensor<f32>
      stablehlo.return %aradds3b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2var = stablehlo.divide %arsums3b1g2var, %arns3b1g2var : tensor<256xf32>
    %v1333 = stablehlo.concatenate %armeans3b1g2mu, %armeans3b1g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1334 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1335 = stablehlo.slice %v1333 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1336 = stablehlo.slice %v1333 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1337 = stablehlo.broadcast_in_dim %v1335, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1338 = stablehlo.broadcast_in_dim %v1336, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1339 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1340 = stablehlo.add %v1338, %v1339 : tensor<64x256x14x14xf32>
    %v1341 = stablehlo.rsqrt %v1340 : tensor<64x256x14x14xf32>
    %v1342 = stablehlo.subtract %v1334, %v1337 : tensor<64x256x14x14xf32>
    %v1343 = stablehlo.multiply %v1342, %v1341 : tensor<64x256x14x14xf32>
    %v1344 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1345 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1346 = stablehlo.multiply %v1343, %v1344 : tensor<64x256x14x14xf32>
    %v1347 = stablehlo.add %v1346, %v1345 : tensor<64x256x14x14xf32>
    %v1348 = stablehlo.reshape %v1347 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1349 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1350 = stablehlo.maximum %v1348, %v1349 : tensor<64x50176xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1352 = stablehlo.convert %v1351 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1353 = stablehlo.convert %s3b1W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1354 = stablehlo.convolution(%v1352, %v1353)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1355 = stablehlo.convert %v1354 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1356 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1357 = stablehlo.add %v1355, %v1356 : tensor<64x1024x14x14xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1359 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1361 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1362 = stablehlo.reduce(%v1359 init: %v1360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1363 = stablehlo.divide %v1362, %v1361 : tensor<1024xf32>
    %arsums3b1g3mu = "stablehlo.all_reduce"(%v1363) ({
    ^bb0(%aras3b1g3mu: tensor<f32>, %arbs3b1g3mu: tensor<f32>):
      %aradds3b1g3mu = stablehlo.add %aras3b1g3mu, %arbs3b1g3mu : tensor<f32>
      stablehlo.return %aradds3b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3mu = stablehlo.divide %arsums3b1g3mu, %arns3b1g3mu : tensor<1024xf32>
    %v1364 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1365 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1366 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1367 = stablehlo.reduce(%v1364 init: %v1365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1368 = stablehlo.divide %v1367, %v1366 : tensor<1024xf32>
    %v1369 = stablehlo.broadcast_in_dim %v1368, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1370 = stablehlo.subtract %v1364, %v1369 : tensor<64x1024x14x14xf32>
    %v1371 = stablehlo.multiply %v1370, %v1370 : tensor<64x1024x14x14xf32>
    %v1372 = stablehlo.reduce(%v1371 init: %v1365) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1373 = stablehlo.divide %v1372, %v1366 : tensor<1024xf32>
    %v1374 = stablehlo.subtract %v1368, %armeans3b1g3mu : tensor<1024xf32>
    %v1375 = stablehlo.multiply %v1374, %v1374 : tensor<1024xf32>
    %v1376 = stablehlo.add %v1373, %v1375 : tensor<1024xf32>
    %arsums3b1g3var = "stablehlo.all_reduce"(%v1376) ({
    ^bb0(%aras3b1g3var: tensor<f32>, %arbs3b1g3var: tensor<f32>):
      %aradds3b1g3var = stablehlo.add %aras3b1g3var, %arbs3b1g3var : tensor<f32>
      stablehlo.return %aradds3b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3var = stablehlo.divide %arsums3b1g3var, %arns3b1g3var : tensor<1024xf32>
    %v1377 = stablehlo.concatenate %armeans3b1g3mu, %armeans3b1g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1378 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1379 = stablehlo.slice %v1377 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1380 = stablehlo.slice %v1377 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1381 = stablehlo.broadcast_in_dim %v1379, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1382 = stablehlo.broadcast_in_dim %v1380, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1383 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1384 = stablehlo.add %v1382, %v1383 : tensor<64x1024x14x14xf32>
    %v1385 = stablehlo.rsqrt %v1384 : tensor<64x1024x14x14xf32>
    %v1386 = stablehlo.subtract %v1378, %v1381 : tensor<64x1024x14x14xf32>
    %v1387 = stablehlo.multiply %v1386, %v1385 : tensor<64x1024x14x14xf32>
    %v1388 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1389 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1390 = stablehlo.multiply %v1387, %v1388 : tensor<64x1024x14x14xf32>
    %v1391 = stablehlo.add %v1390, %v1389 : tensor<64x1024x14x14xf32>
    %v1392 = stablehlo.reshape %v1391 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1393 = stablehlo.reshape %v1392 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1394 = stablehlo.reshape %v1262 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1395 = stablehlo.add %v1393, %v1394 : tensor<64x1024x14x14xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1399 = stablehlo.maximum %v1397, %v1398 : tensor<64x1024x14x14xf32>
    %v1400 = stablehlo.reshape %v1399 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1401 = stablehlo.reshape %v1400 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1402 = stablehlo.convert %v1401 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1403 = stablehlo.convert %s3b2W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1404 = stablehlo.convolution(%v1402, %v1403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1405 = stablehlo.convert %v1404 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1406 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1407 = stablehlo.add %v1405, %v1406 : tensor<64x256x14x14xf32>
    %v1408 = stablehlo.reshape %v1407 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1409 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1410 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1411 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1412 = stablehlo.reduce(%v1409 init: %v1410) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1413 = stablehlo.divide %v1412, %v1411 : tensor<256xf32>
    %arsums3b2g1mu = "stablehlo.all_reduce"(%v1413) ({
    ^bb0(%aras3b2g1mu: tensor<f32>, %arbs3b2g1mu: tensor<f32>):
      %aradds3b2g1mu = stablehlo.add %aras3b2g1mu, %arbs3b2g1mu : tensor<f32>
      stablehlo.return %aradds3b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1mu = stablehlo.divide %arsums3b2g1mu, %arns3b2g1mu : tensor<256xf32>
    %v1414 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1415 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1416 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1417 = stablehlo.reduce(%v1414 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1418 = stablehlo.divide %v1417, %v1416 : tensor<256xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1418, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1420 = stablehlo.subtract %v1414, %v1419 : tensor<64x256x14x14xf32>
    %v1421 = stablehlo.multiply %v1420, %v1420 : tensor<64x256x14x14xf32>
    %v1422 = stablehlo.reduce(%v1421 init: %v1415) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1423 = stablehlo.divide %v1422, %v1416 : tensor<256xf32>
    %v1424 = stablehlo.subtract %v1418, %armeans3b2g1mu : tensor<256xf32>
    %v1425 = stablehlo.multiply %v1424, %v1424 : tensor<256xf32>
    %v1426 = stablehlo.add %v1423, %v1425 : tensor<256xf32>
    %arsums3b2g1var = "stablehlo.all_reduce"(%v1426) ({
    ^bb0(%aras3b2g1var: tensor<f32>, %arbs3b2g1var: tensor<f32>):
      %aradds3b2g1var = stablehlo.add %aras3b2g1var, %arbs3b2g1var : tensor<f32>
      stablehlo.return %aradds3b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1var = stablehlo.divide %arsums3b2g1var, %arns3b2g1var : tensor<256xf32>
    %v1427 = stablehlo.concatenate %armeans3b2g1mu, %armeans3b2g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1428 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1429 = stablehlo.slice %v1427 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1430 = stablehlo.slice %v1427 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1431 = stablehlo.broadcast_in_dim %v1429, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1432 = stablehlo.broadcast_in_dim %v1430, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1433 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1434 = stablehlo.add %v1432, %v1433 : tensor<64x256x14x14xf32>
    %v1435 = stablehlo.rsqrt %v1434 : tensor<64x256x14x14xf32>
    %v1436 = stablehlo.subtract %v1428, %v1431 : tensor<64x256x14x14xf32>
    %v1437 = stablehlo.multiply %v1436, %v1435 : tensor<64x256x14x14xf32>
    %v1438 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1439 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1440 = stablehlo.multiply %v1437, %v1438 : tensor<64x256x14x14xf32>
    %v1441 = stablehlo.add %v1440, %v1439 : tensor<64x256x14x14xf32>
    %v1442 = stablehlo.reshape %v1441 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1443 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1444 = stablehlo.maximum %v1442, %v1443 : tensor<64x50176xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1446 = stablehlo.convert %v1445 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1447 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1448 = stablehlo.convolution(%v1446, %v1447)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1449 = stablehlo.convert %v1448 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1450 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1451 = stablehlo.add %v1449, %v1450 : tensor<64x256x14x14xf32>
    %v1452 = stablehlo.reshape %v1451 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1454 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1455 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1456 = stablehlo.reduce(%v1453 init: %v1454) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1457 = stablehlo.divide %v1456, %v1455 : tensor<256xf32>
    %arsums3b2g2mu = "stablehlo.all_reduce"(%v1457) ({
    ^bb0(%aras3b2g2mu: tensor<f32>, %arbs3b2g2mu: tensor<f32>):
      %aradds3b2g2mu = stablehlo.add %aras3b2g2mu, %arbs3b2g2mu : tensor<f32>
      stablehlo.return %aradds3b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2mu = stablehlo.divide %arsums3b2g2mu, %arns3b2g2mu : tensor<256xf32>
    %v1458 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1460 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1461 = stablehlo.reduce(%v1458 init: %v1459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1462 = stablehlo.divide %v1461, %v1460 : tensor<256xf32>
    %v1463 = stablehlo.broadcast_in_dim %v1462, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1464 = stablehlo.subtract %v1458, %v1463 : tensor<64x256x14x14xf32>
    %v1465 = stablehlo.multiply %v1464, %v1464 : tensor<64x256x14x14xf32>
    %v1466 = stablehlo.reduce(%v1465 init: %v1459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1467 = stablehlo.divide %v1466, %v1460 : tensor<256xf32>
    %v1468 = stablehlo.subtract %v1462, %armeans3b2g2mu : tensor<256xf32>
    %v1469 = stablehlo.multiply %v1468, %v1468 : tensor<256xf32>
    %v1470 = stablehlo.add %v1467, %v1469 : tensor<256xf32>
    %arsums3b2g2var = "stablehlo.all_reduce"(%v1470) ({
    ^bb0(%aras3b2g2var: tensor<f32>, %arbs3b2g2var: tensor<f32>):
      %aradds3b2g2var = stablehlo.add %aras3b2g2var, %arbs3b2g2var : tensor<f32>
      stablehlo.return %aradds3b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2var = stablehlo.divide %arsums3b2g2var, %arns3b2g2var : tensor<256xf32>
    %v1471 = stablehlo.concatenate %armeans3b2g2mu, %armeans3b2g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1472 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1473 = stablehlo.slice %v1471 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1474 = stablehlo.slice %v1471 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1475 = stablehlo.broadcast_in_dim %v1473, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1474, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1477 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1478 = stablehlo.add %v1476, %v1477 : tensor<64x256x14x14xf32>
    %v1479 = stablehlo.rsqrt %v1478 : tensor<64x256x14x14xf32>
    %v1480 = stablehlo.subtract %v1472, %v1475 : tensor<64x256x14x14xf32>
    %v1481 = stablehlo.multiply %v1480, %v1479 : tensor<64x256x14x14xf32>
    %v1482 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1483 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1484 = stablehlo.multiply %v1481, %v1482 : tensor<64x256x14x14xf32>
    %v1485 = stablehlo.add %v1484, %v1483 : tensor<64x256x14x14xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1487 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1488 = stablehlo.maximum %v1486, %v1487 : tensor<64x50176xf32>
    %v1489 = stablehlo.reshape %v1488 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1490 = stablehlo.convert %v1489 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1491 = stablehlo.convert %s3b2W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1492 = stablehlo.convolution(%v1490, %v1491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1493 = stablehlo.convert %v1492 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1494 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1495 = stablehlo.add %v1493, %v1494 : tensor<64x1024x14x14xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1497 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1500 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1501 = stablehlo.divide %v1500, %v1499 : tensor<1024xf32>
    %arsums3b2g3mu = "stablehlo.all_reduce"(%v1501) ({
    ^bb0(%aras3b2g3mu: tensor<f32>, %arbs3b2g3mu: tensor<f32>):
      %aradds3b2g3mu = stablehlo.add %aras3b2g3mu, %arbs3b2g3mu : tensor<f32>
      stablehlo.return %aradds3b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3mu = stablehlo.divide %arsums3b2g3mu, %arns3b2g3mu : tensor<1024xf32>
    %v1502 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1504 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1505 = stablehlo.reduce(%v1502 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1506 = stablehlo.divide %v1505, %v1504 : tensor<1024xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1508 = stablehlo.subtract %v1502, %v1507 : tensor<64x1024x14x14xf32>
    %v1509 = stablehlo.multiply %v1508, %v1508 : tensor<64x1024x14x14xf32>
    %v1510 = stablehlo.reduce(%v1509 init: %v1503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1511 = stablehlo.divide %v1510, %v1504 : tensor<1024xf32>
    %v1512 = stablehlo.subtract %v1506, %armeans3b2g3mu : tensor<1024xf32>
    %v1513 = stablehlo.multiply %v1512, %v1512 : tensor<1024xf32>
    %v1514 = stablehlo.add %v1511, %v1513 : tensor<1024xf32>
    %arsums3b2g3var = "stablehlo.all_reduce"(%v1514) ({
    ^bb0(%aras3b2g3var: tensor<f32>, %arbs3b2g3var: tensor<f32>):
      %aradds3b2g3var = stablehlo.add %aras3b2g3var, %arbs3b2g3var : tensor<f32>
      stablehlo.return %aradds3b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3var = stablehlo.divide %arsums3b2g3var, %arns3b2g3var : tensor<1024xf32>
    %v1515 = stablehlo.concatenate %armeans3b2g3mu, %armeans3b2g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1516 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1517 = stablehlo.slice %v1515 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1518 = stablehlo.slice %v1515 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1517, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1520 = stablehlo.broadcast_in_dim %v1518, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1521 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1522 = stablehlo.add %v1520, %v1521 : tensor<64x1024x14x14xf32>
    %v1523 = stablehlo.rsqrt %v1522 : tensor<64x1024x14x14xf32>
    %v1524 = stablehlo.subtract %v1516, %v1519 : tensor<64x1024x14x14xf32>
    %v1525 = stablehlo.multiply %v1524, %v1523 : tensor<64x1024x14x14xf32>
    %v1526 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1527 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1528 = stablehlo.multiply %v1525, %v1526 : tensor<64x1024x14x14xf32>
    %v1529 = stablehlo.add %v1528, %v1527 : tensor<64x1024x14x14xf32>
    %v1530 = stablehlo.reshape %v1529 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1531 = stablehlo.reshape %v1530 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1532 = stablehlo.reshape %v1400 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1533 = stablehlo.add %v1531, %v1532 : tensor<64x1024x14x14xf32>
    %v1534 = stablehlo.reshape %v1533 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1535 = stablehlo.reshape %v1534 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1536 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1537 = stablehlo.maximum %v1535, %v1536 : tensor<64x1024x14x14xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1539 = stablehlo.reshape %v1538 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1540 = stablehlo.convert %v1539 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1541 = stablehlo.convert %s3b3W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1542 = stablehlo.convolution(%v1540, %v1541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1543 = stablehlo.convert %v1542 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1544 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1545 = stablehlo.add %v1543, %v1544 : tensor<64x256x14x14xf32>
    %v1546 = stablehlo.reshape %v1545 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1547 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1548 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1549 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1550 = stablehlo.reduce(%v1547 init: %v1548) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1551 = stablehlo.divide %v1550, %v1549 : tensor<256xf32>
    %arsums3b3g1mu = "stablehlo.all_reduce"(%v1551) ({
    ^bb0(%aras3b3g1mu: tensor<f32>, %arbs3b3g1mu: tensor<f32>):
      %aradds3b3g1mu = stablehlo.add %aras3b3g1mu, %arbs3b3g1mu : tensor<f32>
      stablehlo.return %aradds3b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1mu = stablehlo.divide %arsums3b3g1mu, %arns3b3g1mu : tensor<256xf32>
    %v1552 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1554 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1555 = stablehlo.reduce(%v1552 init: %v1553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1556 = stablehlo.divide %v1555, %v1554 : tensor<256xf32>
    %v1557 = stablehlo.broadcast_in_dim %v1556, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1558 = stablehlo.subtract %v1552, %v1557 : tensor<64x256x14x14xf32>
    %v1559 = stablehlo.multiply %v1558, %v1558 : tensor<64x256x14x14xf32>
    %v1560 = stablehlo.reduce(%v1559 init: %v1553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1561 = stablehlo.divide %v1560, %v1554 : tensor<256xf32>
    %v1562 = stablehlo.subtract %v1556, %armeans3b3g1mu : tensor<256xf32>
    %v1563 = stablehlo.multiply %v1562, %v1562 : tensor<256xf32>
    %v1564 = stablehlo.add %v1561, %v1563 : tensor<256xf32>
    %arsums3b3g1var = "stablehlo.all_reduce"(%v1564) ({
    ^bb0(%aras3b3g1var: tensor<f32>, %arbs3b3g1var: tensor<f32>):
      %aradds3b3g1var = stablehlo.add %aras3b3g1var, %arbs3b3g1var : tensor<f32>
      stablehlo.return %aradds3b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1var = stablehlo.divide %arsums3b3g1var, %arns3b3g1var : tensor<256xf32>
    %v1565 = stablehlo.concatenate %armeans3b3g1mu, %armeans3b3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1566 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1567 = stablehlo.slice %v1565 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1568 = stablehlo.slice %v1565 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1569 = stablehlo.broadcast_in_dim %v1567, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1570 = stablehlo.broadcast_in_dim %v1568, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1571 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1572 = stablehlo.add %v1570, %v1571 : tensor<64x256x14x14xf32>
    %v1573 = stablehlo.rsqrt %v1572 : tensor<64x256x14x14xf32>
    %v1574 = stablehlo.subtract %v1566, %v1569 : tensor<64x256x14x14xf32>
    %v1575 = stablehlo.multiply %v1574, %v1573 : tensor<64x256x14x14xf32>
    %v1576 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1577 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1578 = stablehlo.multiply %v1575, %v1576 : tensor<64x256x14x14xf32>
    %v1579 = stablehlo.add %v1578, %v1577 : tensor<64x256x14x14xf32>
    %v1580 = stablehlo.reshape %v1579 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1581 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1582 = stablehlo.maximum %v1580, %v1581 : tensor<64x50176xf32>
    %v1583 = stablehlo.reshape %v1582 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1584 = stablehlo.convert %v1583 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1585 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1586 = stablehlo.convolution(%v1584, %v1585)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1587 = stablehlo.convert %v1586 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1588 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1589 = stablehlo.add %v1587, %v1588 : tensor<64x256x14x14xf32>
    %v1590 = stablehlo.reshape %v1589 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1591 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1592 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1593 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1594 = stablehlo.reduce(%v1591 init: %v1592) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1595 = stablehlo.divide %v1594, %v1593 : tensor<256xf32>
    %arsums3b3g2mu = "stablehlo.all_reduce"(%v1595) ({
    ^bb0(%aras3b3g2mu: tensor<f32>, %arbs3b3g2mu: tensor<f32>):
      %aradds3b3g2mu = stablehlo.add %aras3b3g2mu, %arbs3b3g2mu : tensor<f32>
      stablehlo.return %aradds3b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2mu = stablehlo.divide %arsums3b3g2mu, %arns3b3g2mu : tensor<256xf32>
    %v1596 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1597 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1598 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1599 = stablehlo.reduce(%v1596 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1600 = stablehlo.divide %v1599, %v1598 : tensor<256xf32>
    %v1601 = stablehlo.broadcast_in_dim %v1600, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1602 = stablehlo.subtract %v1596, %v1601 : tensor<64x256x14x14xf32>
    %v1603 = stablehlo.multiply %v1602, %v1602 : tensor<64x256x14x14xf32>
    %v1604 = stablehlo.reduce(%v1603 init: %v1597) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1605 = stablehlo.divide %v1604, %v1598 : tensor<256xf32>
    %v1606 = stablehlo.subtract %v1600, %armeans3b3g2mu : tensor<256xf32>
    %v1607 = stablehlo.multiply %v1606, %v1606 : tensor<256xf32>
    %v1608 = stablehlo.add %v1605, %v1607 : tensor<256xf32>
    %arsums3b3g2var = "stablehlo.all_reduce"(%v1608) ({
    ^bb0(%aras3b3g2var: tensor<f32>, %arbs3b3g2var: tensor<f32>):
      %aradds3b3g2var = stablehlo.add %aras3b3g2var, %arbs3b3g2var : tensor<f32>
      stablehlo.return %aradds3b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2var = stablehlo.divide %arsums3b3g2var, %arns3b3g2var : tensor<256xf32>
    %v1609 = stablehlo.concatenate %armeans3b3g2mu, %armeans3b3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1610 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1611 = stablehlo.slice %v1609 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1612 = stablehlo.slice %v1609 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1613 = stablehlo.broadcast_in_dim %v1611, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1612, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1615 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1616 = stablehlo.add %v1614, %v1615 : tensor<64x256x14x14xf32>
    %v1617 = stablehlo.rsqrt %v1616 : tensor<64x256x14x14xf32>
    %v1618 = stablehlo.subtract %v1610, %v1613 : tensor<64x256x14x14xf32>
    %v1619 = stablehlo.multiply %v1618, %v1617 : tensor<64x256x14x14xf32>
    %v1620 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1621 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1622 = stablehlo.multiply %v1619, %v1620 : tensor<64x256x14x14xf32>
    %v1623 = stablehlo.add %v1622, %v1621 : tensor<64x256x14x14xf32>
    %v1624 = stablehlo.reshape %v1623 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1625 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1626 = stablehlo.maximum %v1624, %v1625 : tensor<64x50176xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1628 = stablehlo.convert %v1627 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1629 = stablehlo.convert %s3b3W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1630 = stablehlo.convolution(%v1628, %v1629)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1631 = stablehlo.convert %v1630 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1632 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1633 = stablehlo.add %v1631, %v1632 : tensor<64x1024x14x14xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1635 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1637 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1638 = stablehlo.reduce(%v1635 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1639 = stablehlo.divide %v1638, %v1637 : tensor<1024xf32>
    %arsums3b3g3mu = "stablehlo.all_reduce"(%v1639) ({
    ^bb0(%aras3b3g3mu: tensor<f32>, %arbs3b3g3mu: tensor<f32>):
      %aradds3b3g3mu = stablehlo.add %aras3b3g3mu, %arbs3b3g3mu : tensor<f32>
      stablehlo.return %aradds3b3g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3mu = stablehlo.divide %arsums3b3g3mu, %arns3b3g3mu : tensor<1024xf32>
    %v1640 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1642 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1643 = stablehlo.reduce(%v1640 init: %v1641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1644 = stablehlo.divide %v1643, %v1642 : tensor<1024xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1646 = stablehlo.subtract %v1640, %v1645 : tensor<64x1024x14x14xf32>
    %v1647 = stablehlo.multiply %v1646, %v1646 : tensor<64x1024x14x14xf32>
    %v1648 = stablehlo.reduce(%v1647 init: %v1641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1649 = stablehlo.divide %v1648, %v1642 : tensor<1024xf32>
    %v1650 = stablehlo.subtract %v1644, %armeans3b3g3mu : tensor<1024xf32>
    %v1651 = stablehlo.multiply %v1650, %v1650 : tensor<1024xf32>
    %v1652 = stablehlo.add %v1649, %v1651 : tensor<1024xf32>
    %arsums3b3g3var = "stablehlo.all_reduce"(%v1652) ({
    ^bb0(%aras3b3g3var: tensor<f32>, %arbs3b3g3var: tensor<f32>):
      %aradds3b3g3var = stablehlo.add %aras3b3g3var, %arbs3b3g3var : tensor<f32>
      stablehlo.return %aradds3b3g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3var = stablehlo.divide %arsums3b3g3var, %arns3b3g3var : tensor<1024xf32>
    %v1653 = stablehlo.concatenate %armeans3b3g3mu, %armeans3b3g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1654 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1655 = stablehlo.slice %v1653 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1656 = stablehlo.slice %v1653 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1655, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1658 = stablehlo.broadcast_in_dim %v1656, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1659 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1660 = stablehlo.add %v1658, %v1659 : tensor<64x1024x14x14xf32>
    %v1661 = stablehlo.rsqrt %v1660 : tensor<64x1024x14x14xf32>
    %v1662 = stablehlo.subtract %v1654, %v1657 : tensor<64x1024x14x14xf32>
    %v1663 = stablehlo.multiply %v1662, %v1661 : tensor<64x1024x14x14xf32>
    %v1664 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1665 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1666 = stablehlo.multiply %v1663, %v1664 : tensor<64x1024x14x14xf32>
    %v1667 = stablehlo.add %v1666, %v1665 : tensor<64x1024x14x14xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1669 = stablehlo.reshape %v1668 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1670 = stablehlo.reshape %v1538 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1671 = stablehlo.add %v1669, %v1670 : tensor<64x1024x14x14xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1673 = stablehlo.reshape %v1672 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1674 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1675 = stablehlo.maximum %v1673, %v1674 : tensor<64x1024x14x14xf32>
    %v1676 = stablehlo.reshape %v1675 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1678 = stablehlo.convert %v1677 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1679 = stablehlo.convert %s3b4W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1680 = stablehlo.convolution(%v1678, %v1679)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1681 = stablehlo.convert %v1680 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1682 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1683 = stablehlo.add %v1681, %v1682 : tensor<64x256x14x14xf32>
    %v1684 = stablehlo.reshape %v1683 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1685 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1686 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1687 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1688 = stablehlo.reduce(%v1685 init: %v1686) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1689 = stablehlo.divide %v1688, %v1687 : tensor<256xf32>
    %arsums3b4g1mu = "stablehlo.all_reduce"(%v1689) ({
    ^bb0(%aras3b4g1mu: tensor<f32>, %arbs3b4g1mu: tensor<f32>):
      %aradds3b4g1mu = stablehlo.add %aras3b4g1mu, %arbs3b4g1mu : tensor<f32>
      stablehlo.return %aradds3b4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1mu = stablehlo.divide %arsums3b4g1mu, %arns3b4g1mu : tensor<256xf32>
    %v1690 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1692 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1693 = stablehlo.reduce(%v1690 init: %v1691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1694 = stablehlo.divide %v1693, %v1692 : tensor<256xf32>
    %v1695 = stablehlo.broadcast_in_dim %v1694, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1696 = stablehlo.subtract %v1690, %v1695 : tensor<64x256x14x14xf32>
    %v1697 = stablehlo.multiply %v1696, %v1696 : tensor<64x256x14x14xf32>
    %v1698 = stablehlo.reduce(%v1697 init: %v1691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1699 = stablehlo.divide %v1698, %v1692 : tensor<256xf32>
    %v1700 = stablehlo.subtract %v1694, %armeans3b4g1mu : tensor<256xf32>
    %v1701 = stablehlo.multiply %v1700, %v1700 : tensor<256xf32>
    %v1702 = stablehlo.add %v1699, %v1701 : tensor<256xf32>
    %arsums3b4g1var = "stablehlo.all_reduce"(%v1702) ({
    ^bb0(%aras3b4g1var: tensor<f32>, %arbs3b4g1var: tensor<f32>):
      %aradds3b4g1var = stablehlo.add %aras3b4g1var, %arbs3b4g1var : tensor<f32>
      stablehlo.return %aradds3b4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1var = stablehlo.divide %arsums3b4g1var, %arns3b4g1var : tensor<256xf32>
    %v1703 = stablehlo.concatenate %armeans3b4g1mu, %armeans3b4g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1704 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1705 = stablehlo.slice %v1703 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1706 = stablehlo.slice %v1703 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1707 = stablehlo.broadcast_in_dim %v1705, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1708 = stablehlo.broadcast_in_dim %v1706, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1709 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1710 = stablehlo.add %v1708, %v1709 : tensor<64x256x14x14xf32>
    %v1711 = stablehlo.rsqrt %v1710 : tensor<64x256x14x14xf32>
    %v1712 = stablehlo.subtract %v1704, %v1707 : tensor<64x256x14x14xf32>
    %v1713 = stablehlo.multiply %v1712, %v1711 : tensor<64x256x14x14xf32>
    %v1714 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1715 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1716 = stablehlo.multiply %v1713, %v1714 : tensor<64x256x14x14xf32>
    %v1717 = stablehlo.add %v1716, %v1715 : tensor<64x256x14x14xf32>
    %v1718 = stablehlo.reshape %v1717 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1719 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1720 = stablehlo.maximum %v1718, %v1719 : tensor<64x50176xf32>
    %v1721 = stablehlo.reshape %v1720 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1722 = stablehlo.convert %v1721 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1723 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1724 = stablehlo.convolution(%v1722, %v1723)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1725 = stablehlo.convert %v1724 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1726 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1727 = stablehlo.add %v1725, %v1726 : tensor<64x256x14x14xf32>
    %v1728 = stablehlo.reshape %v1727 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1729 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1731 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1732 = stablehlo.reduce(%v1729 init: %v1730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1733 = stablehlo.divide %v1732, %v1731 : tensor<256xf32>
    %arsums3b4g2mu = "stablehlo.all_reduce"(%v1733) ({
    ^bb0(%aras3b4g2mu: tensor<f32>, %arbs3b4g2mu: tensor<f32>):
      %aradds3b4g2mu = stablehlo.add %aras3b4g2mu, %arbs3b4g2mu : tensor<f32>
      stablehlo.return %aradds3b4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2mu = stablehlo.divide %arsums3b4g2mu, %arns3b4g2mu : tensor<256xf32>
    %v1734 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1736 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1737 = stablehlo.reduce(%v1734 init: %v1735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1738 = stablehlo.divide %v1737, %v1736 : tensor<256xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1740 = stablehlo.subtract %v1734, %v1739 : tensor<64x256x14x14xf32>
    %v1741 = stablehlo.multiply %v1740, %v1740 : tensor<64x256x14x14xf32>
    %v1742 = stablehlo.reduce(%v1741 init: %v1735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1743 = stablehlo.divide %v1742, %v1736 : tensor<256xf32>
    %v1744 = stablehlo.subtract %v1738, %armeans3b4g2mu : tensor<256xf32>
    %v1745 = stablehlo.multiply %v1744, %v1744 : tensor<256xf32>
    %v1746 = stablehlo.add %v1743, %v1745 : tensor<256xf32>
    %arsums3b4g2var = "stablehlo.all_reduce"(%v1746) ({
    ^bb0(%aras3b4g2var: tensor<f32>, %arbs3b4g2var: tensor<f32>):
      %aradds3b4g2var = stablehlo.add %aras3b4g2var, %arbs3b4g2var : tensor<f32>
      stablehlo.return %aradds3b4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2var = stablehlo.divide %arsums3b4g2var, %arns3b4g2var : tensor<256xf32>
    %v1747 = stablehlo.concatenate %armeans3b4g2mu, %armeans3b4g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1748 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1749 = stablehlo.slice %v1747 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1750 = stablehlo.slice %v1747 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1751 = stablehlo.broadcast_in_dim %v1749, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1752 = stablehlo.broadcast_in_dim %v1750, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1753 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1754 = stablehlo.add %v1752, %v1753 : tensor<64x256x14x14xf32>
    %v1755 = stablehlo.rsqrt %v1754 : tensor<64x256x14x14xf32>
    %v1756 = stablehlo.subtract %v1748, %v1751 : tensor<64x256x14x14xf32>
    %v1757 = stablehlo.multiply %v1756, %v1755 : tensor<64x256x14x14xf32>
    %v1758 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1759 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1760 = stablehlo.multiply %v1757, %v1758 : tensor<64x256x14x14xf32>
    %v1761 = stablehlo.add %v1760, %v1759 : tensor<64x256x14x14xf32>
    %v1762 = stablehlo.reshape %v1761 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1763 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1764 = stablehlo.maximum %v1762, %v1763 : tensor<64x50176xf32>
    %v1765 = stablehlo.reshape %v1764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1766 = stablehlo.convert %v1765 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1767 = stablehlo.convert %s3b4W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1768 = stablehlo.convolution(%v1766, %v1767)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1769 = stablehlo.convert %v1768 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1770 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1771 = stablehlo.add %v1769, %v1770 : tensor<64x1024x14x14xf32>
    %v1772 = stablehlo.reshape %v1771 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1773 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1774 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1775 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1776 = stablehlo.reduce(%v1773 init: %v1774) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1777 = stablehlo.divide %v1776, %v1775 : tensor<1024xf32>
    %arsums3b4g3mu = "stablehlo.all_reduce"(%v1777) ({
    ^bb0(%aras3b4g3mu: tensor<f32>, %arbs3b4g3mu: tensor<f32>):
      %aradds3b4g3mu = stablehlo.add %aras3b4g3mu, %arbs3b4g3mu : tensor<f32>
      stablehlo.return %aradds3b4g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3mu = stablehlo.divide %arsums3b4g3mu, %arns3b4g3mu : tensor<1024xf32>
    %v1778 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1780 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1781 = stablehlo.reduce(%v1778 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1782 = stablehlo.divide %v1781, %v1780 : tensor<1024xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1782, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1784 = stablehlo.subtract %v1778, %v1783 : tensor<64x1024x14x14xf32>
    %v1785 = stablehlo.multiply %v1784, %v1784 : tensor<64x1024x14x14xf32>
    %v1786 = stablehlo.reduce(%v1785 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1787 = stablehlo.divide %v1786, %v1780 : tensor<1024xf32>
    %v1788 = stablehlo.subtract %v1782, %armeans3b4g3mu : tensor<1024xf32>
    %v1789 = stablehlo.multiply %v1788, %v1788 : tensor<1024xf32>
    %v1790 = stablehlo.add %v1787, %v1789 : tensor<1024xf32>
    %arsums3b4g3var = "stablehlo.all_reduce"(%v1790) ({
    ^bb0(%aras3b4g3var: tensor<f32>, %arbs3b4g3var: tensor<f32>):
      %aradds3b4g3var = stablehlo.add %aras3b4g3var, %arbs3b4g3var : tensor<f32>
      stablehlo.return %aradds3b4g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3var = stablehlo.divide %arsums3b4g3var, %arns3b4g3var : tensor<1024xf32>
    %v1791 = stablehlo.concatenate %armeans3b4g3mu, %armeans3b4g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1792 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1793 = stablehlo.slice %v1791 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1794 = stablehlo.slice %v1791 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1795 = stablehlo.broadcast_in_dim %v1793, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1796 = stablehlo.broadcast_in_dim %v1794, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1797 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1798 = stablehlo.add %v1796, %v1797 : tensor<64x1024x14x14xf32>
    %v1799 = stablehlo.rsqrt %v1798 : tensor<64x1024x14x14xf32>
    %v1800 = stablehlo.subtract %v1792, %v1795 : tensor<64x1024x14x14xf32>
    %v1801 = stablehlo.multiply %v1800, %v1799 : tensor<64x1024x14x14xf32>
    %v1802 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1803 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1804 = stablehlo.multiply %v1801, %v1802 : tensor<64x1024x14x14xf32>
    %v1805 = stablehlo.add %v1804, %v1803 : tensor<64x1024x14x14xf32>
    %v1806 = stablehlo.reshape %v1805 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1808 = stablehlo.reshape %v1676 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1809 = stablehlo.add %v1807, %v1808 : tensor<64x1024x14x14xf32>
    %v1810 = stablehlo.reshape %v1809 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1812 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1813 = stablehlo.maximum %v1811, %v1812 : tensor<64x1024x14x14xf32>
    %v1814 = stablehlo.reshape %v1813 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1816 = stablehlo.convert %v1815 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1817 = stablehlo.convert %s3b5W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1818 = stablehlo.convolution(%v1816, %v1817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1819 = stablehlo.convert %v1818 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1820 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1821 = stablehlo.add %v1819, %v1820 : tensor<64x256x14x14xf32>
    %v1822 = stablehlo.reshape %v1821 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1823 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1825 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1826 = stablehlo.reduce(%v1823 init: %v1824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1827 = stablehlo.divide %v1826, %v1825 : tensor<256xf32>
    %arsums3b5g1mu = "stablehlo.all_reduce"(%v1827) ({
    ^bb0(%aras3b5g1mu: tensor<f32>, %arbs3b5g1mu: tensor<f32>):
      %aradds3b5g1mu = stablehlo.add %aras3b5g1mu, %arbs3b5g1mu : tensor<f32>
      stablehlo.return %aradds3b5g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1mu = stablehlo.divide %arsums3b5g1mu, %arns3b5g1mu : tensor<256xf32>
    %v1828 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1830 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1831 = stablehlo.reduce(%v1828 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1832 = stablehlo.divide %v1831, %v1830 : tensor<256xf32>
    %v1833 = stablehlo.broadcast_in_dim %v1832, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1834 = stablehlo.subtract %v1828, %v1833 : tensor<64x256x14x14xf32>
    %v1835 = stablehlo.multiply %v1834, %v1834 : tensor<64x256x14x14xf32>
    %v1836 = stablehlo.reduce(%v1835 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1837 = stablehlo.divide %v1836, %v1830 : tensor<256xf32>
    %v1838 = stablehlo.subtract %v1832, %armeans3b5g1mu : tensor<256xf32>
    %v1839 = stablehlo.multiply %v1838, %v1838 : tensor<256xf32>
    %v1840 = stablehlo.add %v1837, %v1839 : tensor<256xf32>
    %arsums3b5g1var = "stablehlo.all_reduce"(%v1840) ({
    ^bb0(%aras3b5g1var: tensor<f32>, %arbs3b5g1var: tensor<f32>):
      %aradds3b5g1var = stablehlo.add %aras3b5g1var, %arbs3b5g1var : tensor<f32>
      stablehlo.return %aradds3b5g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1var = stablehlo.divide %arsums3b5g1var, %arns3b5g1var : tensor<256xf32>
    %v1841 = stablehlo.concatenate %armeans3b5g1mu, %armeans3b5g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1842 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1843 = stablehlo.slice %v1841 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1844 = stablehlo.slice %v1841 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1845 = stablehlo.broadcast_in_dim %v1843, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1846 = stablehlo.broadcast_in_dim %v1844, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1847 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1848 = stablehlo.add %v1846, %v1847 : tensor<64x256x14x14xf32>
    %v1849 = stablehlo.rsqrt %v1848 : tensor<64x256x14x14xf32>
    %v1850 = stablehlo.subtract %v1842, %v1845 : tensor<64x256x14x14xf32>
    %v1851 = stablehlo.multiply %v1850, %v1849 : tensor<64x256x14x14xf32>
    %v1852 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1853 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1854 = stablehlo.multiply %v1851, %v1852 : tensor<64x256x14x14xf32>
    %v1855 = stablehlo.add %v1854, %v1853 : tensor<64x256x14x14xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1857 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1858 = stablehlo.maximum %v1856, %v1857 : tensor<64x50176xf32>
    %v1859 = stablehlo.reshape %v1858 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1860 = stablehlo.convert %v1859 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1861 = stablehlo.convert %s3b5W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1862 = stablehlo.convolution(%v1860, %v1861)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1863 = stablehlo.convert %v1862 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1864 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1865 = stablehlo.add %v1863, %v1864 : tensor<64x256x14x14xf32>
    %v1866 = stablehlo.reshape %v1865 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1867 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1870 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1871 = stablehlo.divide %v1870, %v1869 : tensor<256xf32>
    %arsums3b5g2mu = "stablehlo.all_reduce"(%v1871) ({
    ^bb0(%aras3b5g2mu: tensor<f32>, %arbs3b5g2mu: tensor<f32>):
      %aradds3b5g2mu = stablehlo.add %aras3b5g2mu, %arbs3b5g2mu : tensor<f32>
      stablehlo.return %aradds3b5g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2mu = stablehlo.divide %arsums3b5g2mu, %arns3b5g2mu : tensor<256xf32>
    %v1872 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1873 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1874 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1875 = stablehlo.reduce(%v1872 init: %v1873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1876 = stablehlo.divide %v1875, %v1874 : tensor<256xf32>
    %v1877 = stablehlo.broadcast_in_dim %v1876, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1878 = stablehlo.subtract %v1872, %v1877 : tensor<64x256x14x14xf32>
    %v1879 = stablehlo.multiply %v1878, %v1878 : tensor<64x256x14x14xf32>
    %v1880 = stablehlo.reduce(%v1879 init: %v1873) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1881 = stablehlo.divide %v1880, %v1874 : tensor<256xf32>
    %v1882 = stablehlo.subtract %v1876, %armeans3b5g2mu : tensor<256xf32>
    %v1883 = stablehlo.multiply %v1882, %v1882 : tensor<256xf32>
    %v1884 = stablehlo.add %v1881, %v1883 : tensor<256xf32>
    %arsums3b5g2var = "stablehlo.all_reduce"(%v1884) ({
    ^bb0(%aras3b5g2var: tensor<f32>, %arbs3b5g2var: tensor<f32>):
      %aradds3b5g2var = stablehlo.add %aras3b5g2var, %arbs3b5g2var : tensor<f32>
      stablehlo.return %aradds3b5g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2var = stablehlo.divide %arsums3b5g2var, %arns3b5g2var : tensor<256xf32>
    %v1885 = stablehlo.concatenate %armeans3b5g2mu, %armeans3b5g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1886 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1887 = stablehlo.slice %v1885 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1888 = stablehlo.slice %v1885 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1889 = stablehlo.broadcast_in_dim %v1887, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1890 = stablehlo.broadcast_in_dim %v1888, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1891 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1892 = stablehlo.add %v1890, %v1891 : tensor<64x256x14x14xf32>
    %v1893 = stablehlo.rsqrt %v1892 : tensor<64x256x14x14xf32>
    %v1894 = stablehlo.subtract %v1886, %v1889 : tensor<64x256x14x14xf32>
    %v1895 = stablehlo.multiply %v1894, %v1893 : tensor<64x256x14x14xf32>
    %v1896 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1897 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1898 = stablehlo.multiply %v1895, %v1896 : tensor<64x256x14x14xf32>
    %v1899 = stablehlo.add %v1898, %v1897 : tensor<64x256x14x14xf32>
    %v1900 = stablehlo.reshape %v1899 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1901 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1902 = stablehlo.maximum %v1900, %v1901 : tensor<64x50176xf32>
    %v1903 = stablehlo.reshape %v1902 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1904 = stablehlo.convert %v1903 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1905 = stablehlo.convert %s3b5W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1906 = stablehlo.convolution(%v1904, %v1905)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1907 = stablehlo.convert %v1906 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1908 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1909 = stablehlo.add %v1907, %v1908 : tensor<64x1024x14x14xf32>
    %v1910 = stablehlo.reshape %v1909 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1911 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1912 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1913 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1914 = stablehlo.reduce(%v1911 init: %v1912) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1915 = stablehlo.divide %v1914, %v1913 : tensor<1024xf32>
    %arsums3b5g3mu = "stablehlo.all_reduce"(%v1915) ({
    ^bb0(%aras3b5g3mu: tensor<f32>, %arbs3b5g3mu: tensor<f32>):
      %aradds3b5g3mu = stablehlo.add %aras3b5g3mu, %arbs3b5g3mu : tensor<f32>
      stablehlo.return %aradds3b5g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3mu = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3mu = stablehlo.divide %arsums3b5g3mu, %arns3b5g3mu : tensor<1024xf32>
    %v1916 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1917 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1918 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v1919 = stablehlo.reduce(%v1916 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1920 = stablehlo.divide %v1919, %v1918 : tensor<1024xf32>
    %v1921 = stablehlo.broadcast_in_dim %v1920, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1922 = stablehlo.subtract %v1916, %v1921 : tensor<64x1024x14x14xf32>
    %v1923 = stablehlo.multiply %v1922, %v1922 : tensor<64x1024x14x14xf32>
    %v1924 = stablehlo.reduce(%v1923 init: %v1917) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1925 = stablehlo.divide %v1924, %v1918 : tensor<1024xf32>
    %v1926 = stablehlo.subtract %v1920, %armeans3b5g3mu : tensor<1024xf32>
    %v1927 = stablehlo.multiply %v1926, %v1926 : tensor<1024xf32>
    %v1928 = stablehlo.add %v1925, %v1927 : tensor<1024xf32>
    %arsums3b5g3var = "stablehlo.all_reduce"(%v1928) ({
    ^bb0(%aras3b5g3var: tensor<f32>, %arbs3b5g3var: tensor<f32>):
      %aradds3b5g3var = stablehlo.add %aras3b5g3var, %arbs3b5g3var : tensor<f32>
      stablehlo.return %aradds3b5g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3var = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3var = stablehlo.divide %arsums3b5g3var, %arns3b5g3var : tensor<1024xf32>
    %v1929 = stablehlo.concatenate %armeans3b5g3mu, %armeans3b5g3var, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v1930 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1931 = stablehlo.slice %v1929 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1932 = stablehlo.slice %v1929 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v1933 = stablehlo.broadcast_in_dim %v1931, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1934 = stablehlo.broadcast_in_dim %v1932, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1935 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1936 = stablehlo.add %v1934, %v1935 : tensor<64x1024x14x14xf32>
    %v1937 = stablehlo.rsqrt %v1936 : tensor<64x1024x14x14xf32>
    %v1938 = stablehlo.subtract %v1930, %v1933 : tensor<64x1024x14x14xf32>
    %v1939 = stablehlo.multiply %v1938, %v1937 : tensor<64x1024x14x14xf32>
    %v1940 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1941 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1942 = stablehlo.multiply %v1939, %v1940 : tensor<64x1024x14x14xf32>
    %v1943 = stablehlo.add %v1942, %v1941 : tensor<64x1024x14x14xf32>
    %v1944 = stablehlo.reshape %v1943 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1945 = stablehlo.reshape %v1944 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1946 = stablehlo.reshape %v1814 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1947 = stablehlo.add %v1945, %v1946 : tensor<64x1024x14x14xf32>
    %v1948 = stablehlo.reshape %v1947 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1949 = stablehlo.reshape %v1948 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1950 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v1951 = stablehlo.maximum %v1949, %v1950 : tensor<64x1024x14x14xf32>
    %v1952 = stablehlo.reshape %v1951 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1953 = stablehlo.reshape %v1952 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1954 = stablehlo.convert %v1953 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1955 = stablehlo.convert %s4b0W1 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x14x14xbf16>
    %v1957 = stablehlo.convert %v1956 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v1958 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1959 = stablehlo.add %v1957, %v1958 : tensor<64x512x14x14xf32>
    %v1960 = stablehlo.reshape %v1959 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1961 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1962 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1963 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v1964 = stablehlo.reduce(%v1961 init: %v1962) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1965 = stablehlo.divide %v1964, %v1963 : tensor<512xf32>
    %arsums4b0g1mu = "stablehlo.all_reduce"(%v1965) ({
    ^bb0(%aras4b0g1mu: tensor<f32>, %arbs4b0g1mu: tensor<f32>):
      %aradds4b0g1mu = stablehlo.add %aras4b0g1mu, %arbs4b0g1mu : tensor<f32>
      stablehlo.return %aradds4b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1mu = stablehlo.divide %arsums4b0g1mu, %arns4b0g1mu : tensor<512xf32>
    %v1966 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1968 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v1969 = stablehlo.reduce(%v1966 init: %v1967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1970 = stablehlo.divide %v1969, %v1968 : tensor<512xf32>
    %v1971 = stablehlo.broadcast_in_dim %v1970, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1972 = stablehlo.subtract %v1966, %v1971 : tensor<64x512x14x14xf32>
    %v1973 = stablehlo.multiply %v1972, %v1972 : tensor<64x512x14x14xf32>
    %v1974 = stablehlo.reduce(%v1973 init: %v1967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1975 = stablehlo.divide %v1974, %v1968 : tensor<512xf32>
    %v1976 = stablehlo.subtract %v1970, %armeans4b0g1mu : tensor<512xf32>
    %v1977 = stablehlo.multiply %v1976, %v1976 : tensor<512xf32>
    %v1978 = stablehlo.add %v1975, %v1977 : tensor<512xf32>
    %arsums4b0g1var = "stablehlo.all_reduce"(%v1978) ({
    ^bb0(%aras4b0g1var: tensor<f32>, %arbs4b0g1var: tensor<f32>):
      %aradds4b0g1var = stablehlo.add %aras4b0g1var, %arbs4b0g1var : tensor<f32>
      stablehlo.return %aradds4b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1var = stablehlo.divide %arsums4b0g1var, %arns4b0g1var : tensor<512xf32>
    %v1979 = stablehlo.concatenate %armeans4b0g1mu, %armeans4b0g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1980 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1981 = stablehlo.slice %v1979 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1982 = stablehlo.slice %v1979 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1983 = stablehlo.broadcast_in_dim %v1981, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1984 = stablehlo.broadcast_in_dim %v1982, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1985 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v1986 = stablehlo.add %v1984, %v1985 : tensor<64x512x14x14xf32>
    %v1987 = stablehlo.rsqrt %v1986 : tensor<64x512x14x14xf32>
    %v1988 = stablehlo.subtract %v1980, %v1983 : tensor<64x512x14x14xf32>
    %v1989 = stablehlo.multiply %v1988, %v1987 : tensor<64x512x14x14xf32>
    %v1990 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1991 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1992 = stablehlo.multiply %v1989, %v1990 : tensor<64x512x14x14xf32>
    %v1993 = stablehlo.add %v1992, %v1991 : tensor<64x512x14x14xf32>
    %v1994 = stablehlo.reshape %v1993 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1995 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1996 = stablehlo.maximum %v1994, %v1995 : tensor<64x100352xf32>
    %v1997 = stablehlo.reshape %v1996 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1998 = stablehlo.convert %v1997 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v1999 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2000 = stablehlo.convolution(%v1998, %v1999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2001 = stablehlo.convert %v2000 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2002 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2003 = stablehlo.add %v2001, %v2002 : tensor<64x512x7x7xf32>
    %v2004 = stablehlo.reshape %v2003 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2005 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2007 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2008 = stablehlo.reduce(%v2005 init: %v2006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2009 = stablehlo.divide %v2008, %v2007 : tensor<512xf32>
    %arsums4b0g2mu = "stablehlo.all_reduce"(%v2009) ({
    ^bb0(%aras4b0g2mu: tensor<f32>, %arbs4b0g2mu: tensor<f32>):
      %aradds4b0g2mu = stablehlo.add %aras4b0g2mu, %arbs4b0g2mu : tensor<f32>
      stablehlo.return %aradds4b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2mu = stablehlo.divide %arsums4b0g2mu, %arns4b0g2mu : tensor<512xf32>
    %v2010 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2012 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2013 = stablehlo.reduce(%v2010 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2014 = stablehlo.divide %v2013, %v2012 : tensor<512xf32>
    %v2015 = stablehlo.broadcast_in_dim %v2014, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2016 = stablehlo.subtract %v2010, %v2015 : tensor<64x512x7x7xf32>
    %v2017 = stablehlo.multiply %v2016, %v2016 : tensor<64x512x7x7xf32>
    %v2018 = stablehlo.reduce(%v2017 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2019 = stablehlo.divide %v2018, %v2012 : tensor<512xf32>
    %v2020 = stablehlo.subtract %v2014, %armeans4b0g2mu : tensor<512xf32>
    %v2021 = stablehlo.multiply %v2020, %v2020 : tensor<512xf32>
    %v2022 = stablehlo.add %v2019, %v2021 : tensor<512xf32>
    %arsums4b0g2var = "stablehlo.all_reduce"(%v2022) ({
    ^bb0(%aras4b0g2var: tensor<f32>, %arbs4b0g2var: tensor<f32>):
      %aradds4b0g2var = stablehlo.add %aras4b0g2var, %arbs4b0g2var : tensor<f32>
      stablehlo.return %aradds4b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2var = stablehlo.divide %arsums4b0g2var, %arns4b0g2var : tensor<512xf32>
    %v2023 = stablehlo.concatenate %armeans4b0g2mu, %armeans4b0g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2024 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2025 = stablehlo.slice %v2023 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2026 = stablehlo.slice %v2023 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2027 = stablehlo.broadcast_in_dim %v2025, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2028 = stablehlo.broadcast_in_dim %v2026, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2029 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2030 = stablehlo.add %v2028, %v2029 : tensor<64x512x7x7xf32>
    %v2031 = stablehlo.rsqrt %v2030 : tensor<64x512x7x7xf32>
    %v2032 = stablehlo.subtract %v2024, %v2027 : tensor<64x512x7x7xf32>
    %v2033 = stablehlo.multiply %v2032, %v2031 : tensor<64x512x7x7xf32>
    %v2034 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2035 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2036 = stablehlo.multiply %v2033, %v2034 : tensor<64x512x7x7xf32>
    %v2037 = stablehlo.add %v2036, %v2035 : tensor<64x512x7x7xf32>
    %v2038 = stablehlo.reshape %v2037 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2039 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2040 = stablehlo.maximum %v2038, %v2039 : tensor<64x25088xf32>
    %v2041 = stablehlo.reshape %v2040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2042 = stablehlo.convert %v2041 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2043 = stablehlo.convert %s4b0W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2044 = stablehlo.convolution(%v2042, %v2043)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2045 = stablehlo.convert %v2044 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2046 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2047 = stablehlo.add %v2045, %v2046 : tensor<64x2048x7x7xf32>
    %v2048 = stablehlo.reshape %v2047 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2049 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2051 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2052 = stablehlo.reduce(%v2049 init: %v2050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2053 = stablehlo.divide %v2052, %v2051 : tensor<2048xf32>
    %arsums4b0g3mu = "stablehlo.all_reduce"(%v2053) ({
    ^bb0(%aras4b0g3mu: tensor<f32>, %arbs4b0g3mu: tensor<f32>):
      %aradds4b0g3mu = stablehlo.add %aras4b0g3mu, %arbs4b0g3mu : tensor<f32>
      stablehlo.return %aradds4b0g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3mu = stablehlo.divide %arsums4b0g3mu, %arns4b0g3mu : tensor<2048xf32>
    %v2054 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2056 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2057 = stablehlo.reduce(%v2054 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2058 = stablehlo.divide %v2057, %v2056 : tensor<2048xf32>
    %v2059 = stablehlo.broadcast_in_dim %v2058, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2060 = stablehlo.subtract %v2054, %v2059 : tensor<64x2048x7x7xf32>
    %v2061 = stablehlo.multiply %v2060, %v2060 : tensor<64x2048x7x7xf32>
    %v2062 = stablehlo.reduce(%v2061 init: %v2055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2063 = stablehlo.divide %v2062, %v2056 : tensor<2048xf32>
    %v2064 = stablehlo.subtract %v2058, %armeans4b0g3mu : tensor<2048xf32>
    %v2065 = stablehlo.multiply %v2064, %v2064 : tensor<2048xf32>
    %v2066 = stablehlo.add %v2063, %v2065 : tensor<2048xf32>
    %arsums4b0g3var = "stablehlo.all_reduce"(%v2066) ({
    ^bb0(%aras4b0g3var: tensor<f32>, %arbs4b0g3var: tensor<f32>):
      %aradds4b0g3var = stablehlo.add %aras4b0g3var, %arbs4b0g3var : tensor<f32>
      stablehlo.return %aradds4b0g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3var = stablehlo.divide %arsums4b0g3var, %arns4b0g3var : tensor<2048xf32>
    %v2067 = stablehlo.concatenate %armeans4b0g3mu, %armeans4b0g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2068 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2069 = stablehlo.slice %v2067 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2070 = stablehlo.slice %v2067 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2069, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2072 = stablehlo.broadcast_in_dim %v2070, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2073 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2074 = stablehlo.add %v2072, %v2073 : tensor<64x2048x7x7xf32>
    %v2075 = stablehlo.rsqrt %v2074 : tensor<64x2048x7x7xf32>
    %v2076 = stablehlo.subtract %v2068, %v2071 : tensor<64x2048x7x7xf32>
    %v2077 = stablehlo.multiply %v2076, %v2075 : tensor<64x2048x7x7xf32>
    %v2078 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2079 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2080 = stablehlo.multiply %v2077, %v2078 : tensor<64x2048x7x7xf32>
    %v2081 = stablehlo.add %v2080, %v2079 : tensor<64x2048x7x7xf32>
    %v2082 = stablehlo.reshape %v2081 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2083 = stablehlo.reshape %v1952 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2084 = stablehlo.convert %v2083 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2085 = stablehlo.convert %s4b0Wp : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xbf16>
    %v2086 = stablehlo.convolution(%v2084, %v2085)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2087 = stablehlo.convert %v2086 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2088 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2089 = stablehlo.add %v2087, %v2088 : tensor<64x2048x7x7xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2091 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2092 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2093 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2094 = stablehlo.reduce(%v2091 init: %v2092) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2095 = stablehlo.divide %v2094, %v2093 : tensor<2048xf32>
    %arsums4b0gpmu = "stablehlo.all_reduce"(%v2095) ({
    ^bb0(%aras4b0gpmu: tensor<f32>, %arbs4b0gpmu: tensor<f32>):
      %aradds4b0gpmu = stablehlo.add %aras4b0gpmu, %arbs4b0gpmu : tensor<f32>
      stablehlo.return %aradds4b0gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gpmu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gpmu = stablehlo.divide %arsums4b0gpmu, %arns4b0gpmu : tensor<2048xf32>
    %v2096 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2098 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2099 = stablehlo.reduce(%v2096 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2100 = stablehlo.divide %v2099, %v2098 : tensor<2048xf32>
    %v2101 = stablehlo.broadcast_in_dim %v2100, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2102 = stablehlo.subtract %v2096, %v2101 : tensor<64x2048x7x7xf32>
    %v2103 = stablehlo.multiply %v2102, %v2102 : tensor<64x2048x7x7xf32>
    %v2104 = stablehlo.reduce(%v2103 init: %v2097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2105 = stablehlo.divide %v2104, %v2098 : tensor<2048xf32>
    %v2106 = stablehlo.subtract %v2100, %armeans4b0gpmu : tensor<2048xf32>
    %v2107 = stablehlo.multiply %v2106, %v2106 : tensor<2048xf32>
    %v2108 = stablehlo.add %v2105, %v2107 : tensor<2048xf32>
    %arsums4b0gpvar = "stablehlo.all_reduce"(%v2108) ({
    ^bb0(%aras4b0gpvar: tensor<f32>, %arbs4b0gpvar: tensor<f32>):
      %aradds4b0gpvar = stablehlo.add %aras4b0gpvar, %arbs4b0gpvar : tensor<f32>
      stablehlo.return %aradds4b0gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gpvar = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gpvar = stablehlo.divide %arsums4b0gpvar, %arns4b0gpvar : tensor<2048xf32>
    %v2109 = stablehlo.concatenate %armeans4b0gpmu, %armeans4b0gpvar, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2110 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2111 = stablehlo.slice %v2109 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2112 = stablehlo.slice %v2109 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2113 = stablehlo.broadcast_in_dim %v2111, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2114 = stablehlo.broadcast_in_dim %v2112, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2115 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2116 = stablehlo.add %v2114, %v2115 : tensor<64x2048x7x7xf32>
    %v2117 = stablehlo.rsqrt %v2116 : tensor<64x2048x7x7xf32>
    %v2118 = stablehlo.subtract %v2110, %v2113 : tensor<64x2048x7x7xf32>
    %v2119 = stablehlo.multiply %v2118, %v2117 : tensor<64x2048x7x7xf32>
    %v2120 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2121 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2122 = stablehlo.multiply %v2119, %v2120 : tensor<64x2048x7x7xf32>
    %v2123 = stablehlo.add %v2122, %v2121 : tensor<64x2048x7x7xf32>
    %v2124 = stablehlo.reshape %v2123 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2125 = stablehlo.add %v2082, %v2124 : tensor<64x100352xf32>
    %v2126 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2127 = stablehlo.maximum %v2125, %v2126 : tensor<64x100352xf32>
    %v2128 = stablehlo.reshape %v2127 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2129 = stablehlo.convert %v2128 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2130 = stablehlo.convert %s4b1W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2131 = stablehlo.convolution(%v2129, %v2130)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2132 = stablehlo.convert %v2131 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2133 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2134 = stablehlo.add %v2132, %v2133 : tensor<64x512x7x7xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2137 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2138 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2139 = stablehlo.reduce(%v2136 init: %v2137) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2140 = stablehlo.divide %v2139, %v2138 : tensor<512xf32>
    %arsums4b1g1mu = "stablehlo.all_reduce"(%v2140) ({
    ^bb0(%aras4b1g1mu: tensor<f32>, %arbs4b1g1mu: tensor<f32>):
      %aradds4b1g1mu = stablehlo.add %aras4b1g1mu, %arbs4b1g1mu : tensor<f32>
      stablehlo.return %aradds4b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1mu = stablehlo.divide %arsums4b1g1mu, %arns4b1g1mu : tensor<512xf32>
    %v2141 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2142 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2143 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2144 = stablehlo.reduce(%v2141 init: %v2142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2145 = stablehlo.divide %v2144, %v2143 : tensor<512xf32>
    %v2146 = stablehlo.broadcast_in_dim %v2145, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2147 = stablehlo.subtract %v2141, %v2146 : tensor<64x512x7x7xf32>
    %v2148 = stablehlo.multiply %v2147, %v2147 : tensor<64x512x7x7xf32>
    %v2149 = stablehlo.reduce(%v2148 init: %v2142) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2150 = stablehlo.divide %v2149, %v2143 : tensor<512xf32>
    %v2151 = stablehlo.subtract %v2145, %armeans4b1g1mu : tensor<512xf32>
    %v2152 = stablehlo.multiply %v2151, %v2151 : tensor<512xf32>
    %v2153 = stablehlo.add %v2150, %v2152 : tensor<512xf32>
    %arsums4b1g1var = "stablehlo.all_reduce"(%v2153) ({
    ^bb0(%aras4b1g1var: tensor<f32>, %arbs4b1g1var: tensor<f32>):
      %aradds4b1g1var = stablehlo.add %aras4b1g1var, %arbs4b1g1var : tensor<f32>
      stablehlo.return %aradds4b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1var = stablehlo.divide %arsums4b1g1var, %arns4b1g1var : tensor<512xf32>
    %v2154 = stablehlo.concatenate %armeans4b1g1mu, %armeans4b1g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2155 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2156 = stablehlo.slice %v2154 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2157 = stablehlo.slice %v2154 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2158 = stablehlo.broadcast_in_dim %v2156, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2159 = stablehlo.broadcast_in_dim %v2157, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2160 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2161 = stablehlo.add %v2159, %v2160 : tensor<64x512x7x7xf32>
    %v2162 = stablehlo.rsqrt %v2161 : tensor<64x512x7x7xf32>
    %v2163 = stablehlo.subtract %v2155, %v2158 : tensor<64x512x7x7xf32>
    %v2164 = stablehlo.multiply %v2163, %v2162 : tensor<64x512x7x7xf32>
    %v2165 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2166 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2167 = stablehlo.multiply %v2164, %v2165 : tensor<64x512x7x7xf32>
    %v2168 = stablehlo.add %v2167, %v2166 : tensor<64x512x7x7xf32>
    %v2169 = stablehlo.reshape %v2168 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2170 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2171 = stablehlo.maximum %v2169, %v2170 : tensor<64x25088xf32>
    %v2172 = stablehlo.reshape %v2171 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2173 = stablehlo.convert %v2172 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2174 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2175 = stablehlo.convolution(%v2173, %v2174)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2176 = stablehlo.convert %v2175 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2177 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2178 = stablehlo.add %v2176, %v2177 : tensor<64x512x7x7xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2180 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2181 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2182 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2183 = stablehlo.reduce(%v2180 init: %v2181) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2184 = stablehlo.divide %v2183, %v2182 : tensor<512xf32>
    %arsums4b1g2mu = "stablehlo.all_reduce"(%v2184) ({
    ^bb0(%aras4b1g2mu: tensor<f32>, %arbs4b1g2mu: tensor<f32>):
      %aradds4b1g2mu = stablehlo.add %aras4b1g2mu, %arbs4b1g2mu : tensor<f32>
      stablehlo.return %aradds4b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2mu = stablehlo.divide %arsums4b1g2mu, %arns4b1g2mu : tensor<512xf32>
    %v2185 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2187 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2188 = stablehlo.reduce(%v2185 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2189 = stablehlo.divide %v2188, %v2187 : tensor<512xf32>
    %v2190 = stablehlo.broadcast_in_dim %v2189, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2191 = stablehlo.subtract %v2185, %v2190 : tensor<64x512x7x7xf32>
    %v2192 = stablehlo.multiply %v2191, %v2191 : tensor<64x512x7x7xf32>
    %v2193 = stablehlo.reduce(%v2192 init: %v2186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2194 = stablehlo.divide %v2193, %v2187 : tensor<512xf32>
    %v2195 = stablehlo.subtract %v2189, %armeans4b1g2mu : tensor<512xf32>
    %v2196 = stablehlo.multiply %v2195, %v2195 : tensor<512xf32>
    %v2197 = stablehlo.add %v2194, %v2196 : tensor<512xf32>
    %arsums4b1g2var = "stablehlo.all_reduce"(%v2197) ({
    ^bb0(%aras4b1g2var: tensor<f32>, %arbs4b1g2var: tensor<f32>):
      %aradds4b1g2var = stablehlo.add %aras4b1g2var, %arbs4b1g2var : tensor<f32>
      stablehlo.return %aradds4b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2var = stablehlo.divide %arsums4b1g2var, %arns4b1g2var : tensor<512xf32>
    %v2198 = stablehlo.concatenate %armeans4b1g2mu, %armeans4b1g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2199 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2200 = stablehlo.slice %v2198 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2201 = stablehlo.slice %v2198 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2202 = stablehlo.broadcast_in_dim %v2200, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2203 = stablehlo.broadcast_in_dim %v2201, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2204 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2205 = stablehlo.add %v2203, %v2204 : tensor<64x512x7x7xf32>
    %v2206 = stablehlo.rsqrt %v2205 : tensor<64x512x7x7xf32>
    %v2207 = stablehlo.subtract %v2199, %v2202 : tensor<64x512x7x7xf32>
    %v2208 = stablehlo.multiply %v2207, %v2206 : tensor<64x512x7x7xf32>
    %v2209 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2210 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2211 = stablehlo.multiply %v2208, %v2209 : tensor<64x512x7x7xf32>
    %v2212 = stablehlo.add %v2211, %v2210 : tensor<64x512x7x7xf32>
    %v2213 = stablehlo.reshape %v2212 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2214 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2215 = stablehlo.maximum %v2213, %v2214 : tensor<64x25088xf32>
    %v2216 = stablehlo.reshape %v2215 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2217 = stablehlo.convert %v2216 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2218 = stablehlo.convert %s4b1W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2219 = stablehlo.convolution(%v2217, %v2218)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2220 = stablehlo.convert %v2219 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2221 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2222 = stablehlo.add %v2220, %v2221 : tensor<64x2048x7x7xf32>
    %v2223 = stablehlo.reshape %v2222 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2224 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2226 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2227 = stablehlo.reduce(%v2224 init: %v2225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2228 = stablehlo.divide %v2227, %v2226 : tensor<2048xf32>
    %arsums4b1g3mu = "stablehlo.all_reduce"(%v2228) ({
    ^bb0(%aras4b1g3mu: tensor<f32>, %arbs4b1g3mu: tensor<f32>):
      %aradds4b1g3mu = stablehlo.add %aras4b1g3mu, %arbs4b1g3mu : tensor<f32>
      stablehlo.return %aradds4b1g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3mu = stablehlo.divide %arsums4b1g3mu, %arns4b1g3mu : tensor<2048xf32>
    %v2229 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2230 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2231 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2232 = stablehlo.reduce(%v2229 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2233 = stablehlo.divide %v2232, %v2231 : tensor<2048xf32>
    %v2234 = stablehlo.broadcast_in_dim %v2233, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2235 = stablehlo.subtract %v2229, %v2234 : tensor<64x2048x7x7xf32>
    %v2236 = stablehlo.multiply %v2235, %v2235 : tensor<64x2048x7x7xf32>
    %v2237 = stablehlo.reduce(%v2236 init: %v2230) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2238 = stablehlo.divide %v2237, %v2231 : tensor<2048xf32>
    %v2239 = stablehlo.subtract %v2233, %armeans4b1g3mu : tensor<2048xf32>
    %v2240 = stablehlo.multiply %v2239, %v2239 : tensor<2048xf32>
    %v2241 = stablehlo.add %v2238, %v2240 : tensor<2048xf32>
    %arsums4b1g3var = "stablehlo.all_reduce"(%v2241) ({
    ^bb0(%aras4b1g3var: tensor<f32>, %arbs4b1g3var: tensor<f32>):
      %aradds4b1g3var = stablehlo.add %aras4b1g3var, %arbs4b1g3var : tensor<f32>
      stablehlo.return %aradds4b1g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3var = stablehlo.divide %arsums4b1g3var, %arns4b1g3var : tensor<2048xf32>
    %v2242 = stablehlo.concatenate %armeans4b1g3mu, %armeans4b1g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2243 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2244 = stablehlo.slice %v2242 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2245 = stablehlo.slice %v2242 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2246 = stablehlo.broadcast_in_dim %v2244, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2247 = stablehlo.broadcast_in_dim %v2245, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2248 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2249 = stablehlo.add %v2247, %v2248 : tensor<64x2048x7x7xf32>
    %v2250 = stablehlo.rsqrt %v2249 : tensor<64x2048x7x7xf32>
    %v2251 = stablehlo.subtract %v2243, %v2246 : tensor<64x2048x7x7xf32>
    %v2252 = stablehlo.multiply %v2251, %v2250 : tensor<64x2048x7x7xf32>
    %v2253 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2254 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2255 = stablehlo.multiply %v2252, %v2253 : tensor<64x2048x7x7xf32>
    %v2256 = stablehlo.add %v2255, %v2254 : tensor<64x2048x7x7xf32>
    %v2257 = stablehlo.reshape %v2256 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2258 = stablehlo.reshape %v2257 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2259 = stablehlo.reshape %v2127 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2260 = stablehlo.add %v2258, %v2259 : tensor<64x2048x7x7xf32>
    %v2261 = stablehlo.reshape %v2260 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2262 = stablehlo.reshape %v2261 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2263 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2264 = stablehlo.maximum %v2262, %v2263 : tensor<64x2048x7x7xf32>
    %v2265 = stablehlo.reshape %v2264 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2266 = stablehlo.reshape %v2265 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2267 = stablehlo.convert %v2266 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2268 = stablehlo.convert %s4b2W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2269 = stablehlo.convolution(%v2267, %v2268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2270 = stablehlo.convert %v2269 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2271 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2272 = stablehlo.add %v2270, %v2271 : tensor<64x512x7x7xf32>
    %v2273 = stablehlo.reshape %v2272 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2274 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2276 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2277 = stablehlo.reduce(%v2274 init: %v2275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2278 = stablehlo.divide %v2277, %v2276 : tensor<512xf32>
    %arsums4b2g1mu = "stablehlo.all_reduce"(%v2278) ({
    ^bb0(%aras4b2g1mu: tensor<f32>, %arbs4b2g1mu: tensor<f32>):
      %aradds4b2g1mu = stablehlo.add %aras4b2g1mu, %arbs4b2g1mu : tensor<f32>
      stablehlo.return %aradds4b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1mu = stablehlo.divide %arsums4b2g1mu, %arns4b2g1mu : tensor<512xf32>
    %v2279 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2280 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2281 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2282 = stablehlo.reduce(%v2279 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2283 = stablehlo.divide %v2282, %v2281 : tensor<512xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2285 = stablehlo.subtract %v2279, %v2284 : tensor<64x512x7x7xf32>
    %v2286 = stablehlo.multiply %v2285, %v2285 : tensor<64x512x7x7xf32>
    %v2287 = stablehlo.reduce(%v2286 init: %v2280) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2288 = stablehlo.divide %v2287, %v2281 : tensor<512xf32>
    %v2289 = stablehlo.subtract %v2283, %armeans4b2g1mu : tensor<512xf32>
    %v2290 = stablehlo.multiply %v2289, %v2289 : tensor<512xf32>
    %v2291 = stablehlo.add %v2288, %v2290 : tensor<512xf32>
    %arsums4b2g1var = "stablehlo.all_reduce"(%v2291) ({
    ^bb0(%aras4b2g1var: tensor<f32>, %arbs4b2g1var: tensor<f32>):
      %aradds4b2g1var = stablehlo.add %aras4b2g1var, %arbs4b2g1var : tensor<f32>
      stablehlo.return %aradds4b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1var = stablehlo.divide %arsums4b2g1var, %arns4b2g1var : tensor<512xf32>
    %v2292 = stablehlo.concatenate %armeans4b2g1mu, %armeans4b2g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2293 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2294 = stablehlo.slice %v2292 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2295 = stablehlo.slice %v2292 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2296 = stablehlo.broadcast_in_dim %v2294, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2297 = stablehlo.broadcast_in_dim %v2295, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2298 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2299 = stablehlo.add %v2297, %v2298 : tensor<64x512x7x7xf32>
    %v2300 = stablehlo.rsqrt %v2299 : tensor<64x512x7x7xf32>
    %v2301 = stablehlo.subtract %v2293, %v2296 : tensor<64x512x7x7xf32>
    %v2302 = stablehlo.multiply %v2301, %v2300 : tensor<64x512x7x7xf32>
    %v2303 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2304 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2305 = stablehlo.multiply %v2302, %v2303 : tensor<64x512x7x7xf32>
    %v2306 = stablehlo.add %v2305, %v2304 : tensor<64x512x7x7xf32>
    %v2307 = stablehlo.reshape %v2306 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2308 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2309 = stablehlo.maximum %v2307, %v2308 : tensor<64x25088xf32>
    %v2310 = stablehlo.reshape %v2309 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2311 = stablehlo.convert %v2310 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2312 = stablehlo.convert %s4b2W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2313 = stablehlo.convolution(%v2311, %v2312)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2314 = stablehlo.convert %v2313 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2315 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2316 = stablehlo.add %v2314, %v2315 : tensor<64x512x7x7xf32>
    %v2317 = stablehlo.reshape %v2316 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2318 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2320 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2321 = stablehlo.reduce(%v2318 init: %v2319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2322 = stablehlo.divide %v2321, %v2320 : tensor<512xf32>
    %arsums4b2g2mu = "stablehlo.all_reduce"(%v2322) ({
    ^bb0(%aras4b2g2mu: tensor<f32>, %arbs4b2g2mu: tensor<f32>):
      %aradds4b2g2mu = stablehlo.add %aras4b2g2mu, %arbs4b2g2mu : tensor<f32>
      stablehlo.return %aradds4b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2mu = stablehlo.divide %arsums4b2g2mu, %arns4b2g2mu : tensor<512xf32>
    %v2323 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2324 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2325 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2326 = stablehlo.reduce(%v2323 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2327 = stablehlo.divide %v2326, %v2325 : tensor<512xf32>
    %v2328 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2329 = stablehlo.subtract %v2323, %v2328 : tensor<64x512x7x7xf32>
    %v2330 = stablehlo.multiply %v2329, %v2329 : tensor<64x512x7x7xf32>
    %v2331 = stablehlo.reduce(%v2330 init: %v2324) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2332 = stablehlo.divide %v2331, %v2325 : tensor<512xf32>
    %v2333 = stablehlo.subtract %v2327, %armeans4b2g2mu : tensor<512xf32>
    %v2334 = stablehlo.multiply %v2333, %v2333 : tensor<512xf32>
    %v2335 = stablehlo.add %v2332, %v2334 : tensor<512xf32>
    %arsums4b2g2var = "stablehlo.all_reduce"(%v2335) ({
    ^bb0(%aras4b2g2var: tensor<f32>, %arbs4b2g2var: tensor<f32>):
      %aradds4b2g2var = stablehlo.add %aras4b2g2var, %arbs4b2g2var : tensor<f32>
      stablehlo.return %aradds4b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2var = stablehlo.divide %arsums4b2g2var, %arns4b2g2var : tensor<512xf32>
    %v2336 = stablehlo.concatenate %armeans4b2g2mu, %armeans4b2g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2337 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2338 = stablehlo.slice %v2336 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2339 = stablehlo.slice %v2336 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2340 = stablehlo.broadcast_in_dim %v2338, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2339, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2342 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2343 = stablehlo.add %v2341, %v2342 : tensor<64x512x7x7xf32>
    %v2344 = stablehlo.rsqrt %v2343 : tensor<64x512x7x7xf32>
    %v2345 = stablehlo.subtract %v2337, %v2340 : tensor<64x512x7x7xf32>
    %v2346 = stablehlo.multiply %v2345, %v2344 : tensor<64x512x7x7xf32>
    %v2347 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2348 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2349 = stablehlo.multiply %v2346, %v2347 : tensor<64x512x7x7xf32>
    %v2350 = stablehlo.add %v2349, %v2348 : tensor<64x512x7x7xf32>
    %v2351 = stablehlo.reshape %v2350 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2352 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2353 = stablehlo.maximum %v2351, %v2352 : tensor<64x25088xf32>
    %v2354 = stablehlo.reshape %v2353 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2355 = stablehlo.convert %v2354 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2356 = stablehlo.convert %s4b2W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2357 = stablehlo.convolution(%v2355, %v2356)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2358 = stablehlo.convert %v2357 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2359 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2360 = stablehlo.add %v2358, %v2359 : tensor<64x2048x7x7xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2362 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2364 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2365 = stablehlo.reduce(%v2362 init: %v2363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2366 = stablehlo.divide %v2365, %v2364 : tensor<2048xf32>
    %arsums4b2g3mu = "stablehlo.all_reduce"(%v2366) ({
    ^bb0(%aras4b2g3mu: tensor<f32>, %arbs4b2g3mu: tensor<f32>):
      %aradds4b2g3mu = stablehlo.add %aras4b2g3mu, %arbs4b2g3mu : tensor<f32>
      stablehlo.return %aradds4b2g3mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3mu = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3mu = stablehlo.divide %arsums4b2g3mu, %arns4b2g3mu : tensor<2048xf32>
    %v2367 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2369 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2370 = stablehlo.reduce(%v2367 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2371 = stablehlo.divide %v2370, %v2369 : tensor<2048xf32>
    %v2372 = stablehlo.broadcast_in_dim %v2371, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2373 = stablehlo.subtract %v2367, %v2372 : tensor<64x2048x7x7xf32>
    %v2374 = stablehlo.multiply %v2373, %v2373 : tensor<64x2048x7x7xf32>
    %v2375 = stablehlo.reduce(%v2374 init: %v2368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2376 = stablehlo.divide %v2375, %v2369 : tensor<2048xf32>
    %v2377 = stablehlo.subtract %v2371, %armeans4b2g3mu : tensor<2048xf32>
    %v2378 = stablehlo.multiply %v2377, %v2377 : tensor<2048xf32>
    %v2379 = stablehlo.add %v2376, %v2378 : tensor<2048xf32>
    %arsums4b2g3var = "stablehlo.all_reduce"(%v2379) ({
    ^bb0(%aras4b2g3var: tensor<f32>, %arbs4b2g3var: tensor<f32>):
      %aradds4b2g3var = stablehlo.add %aras4b2g3var, %arbs4b2g3var : tensor<f32>
      stablehlo.return %aradds4b2g3var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3var = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3var = stablehlo.divide %arsums4b2g3var, %arns4b2g3var : tensor<2048xf32>
    %v2380 = stablehlo.concatenate %armeans4b2g3mu, %armeans4b2g3var, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2381 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2382 = stablehlo.slice %v2380 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2383 = stablehlo.slice %v2380 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2384 = stablehlo.broadcast_in_dim %v2382, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2385 = stablehlo.broadcast_in_dim %v2383, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2386 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2387 = stablehlo.add %v2385, %v2386 : tensor<64x2048x7x7xf32>
    %v2388 = stablehlo.rsqrt %v2387 : tensor<64x2048x7x7xf32>
    %v2389 = stablehlo.subtract %v2381, %v2384 : tensor<64x2048x7x7xf32>
    %v2390 = stablehlo.multiply %v2389, %v2388 : tensor<64x2048x7x7xf32>
    %v2391 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2392 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2393 = stablehlo.multiply %v2390, %v2391 : tensor<64x2048x7x7xf32>
    %v2394 = stablehlo.add %v2393, %v2392 : tensor<64x2048x7x7xf32>
    %v2395 = stablehlo.reshape %v2394 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2396 = stablehlo.reshape %v2395 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2397 = stablehlo.reshape %v2265 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2398 = stablehlo.add %v2396, %v2397 : tensor<64x2048x7x7xf32>
    %v2399 = stablehlo.reshape %v2398 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2400 = stablehlo.reshape %v2399 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2401 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2402 = stablehlo.maximum %v2400, %v2401 : tensor<64x2048x7x7xf32>
    %v2403 = stablehlo.reshape %v2402 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2404 = stablehlo.reshape %v2403 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2406 = stablehlo.reduce(%v2404 init: %v2405) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048xf32>
    %v2407 = stablehlo.constant dense<49.0> : tensor<64x2048xf32>
    %v2408 = stablehlo.divide %v2406, %v2407 : tensor<64x2048xf32>
    %v2409 = stablehlo.dot_general %v2408, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<2048x1000xf32>) -> tensor<64x1000xf32>
    %v2410 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v2411 = stablehlo.add %v2409, %v2410 : tensor<64x1000xf32>
    %v2412 = stablehlo.reshape %v2411 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v2413 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2414 = stablehlo.exponential %v2412 : tensor<64x1x1000xf32>
    %v2415 = stablehlo.reduce(%v2414 init: %v2413) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v2416 = stablehlo.broadcast_in_dim %v2415, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v2417 = stablehlo.divide %v2414, %v2416 : tensor<64x1x1000xf32>
    %v2418 = stablehlo.reshape %v2417 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v2419 = stablehlo.subtract %v2418, %onehot : tensor<64x1000xf32>
    %v2420 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v2421 = stablehlo.multiply %onehot, %v2420 : tensor<64x1000xf32>
    %v2422 = stablehlo.add %v2419, %v2421 : tensor<64x1000xf32>
    %v2423 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v2424 = stablehlo.add %v2422, %v2423 : tensor<64x1000xf32>
    %v2425 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v2426 = stablehlo.divide %v2424, %v2425 : tensor<64x1000xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v2428 = stablehlo.dot_general %v2427, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<64x1x2048xf32>
    %v2429 = stablehlo.reshape %v2428 : (tensor<64x1x2048xf32>) -> tensor<64x2048xf32>
    %v2430 = stablehlo.dot_general %v2408, %v2426, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<64x1000xf32>) -> tensor<2048x1000xf32>
    %v2431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2432 = stablehlo.reduce(%v2426 init: %v2431) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v2433 = stablehlo.broadcast_in_dim %v2429, dims = [0, 1] : (tensor<64x2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2434 = stablehlo.constant dense<49.0> : tensor<64x2048x7x7xf32>
    %v2435 = stablehlo.divide %v2433, %v2434 : tensor<64x2048x7x7xf32>
    %v2436 = stablehlo.reshape %v2435 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2437 = stablehlo.reshape %v2436 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2438 = stablehlo.reshape %v2399 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2439 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2440 = stablehlo.compare GT, %v2438, %v2439 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2441 = stablehlo.select %v2440, %v2437, %v2439 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2442 = stablehlo.reshape %v2441 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2443 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2444 = stablehlo.slice %v2380 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2445 = stablehlo.slice %v2380 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2446 = stablehlo.broadcast_in_dim %v2444, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2447 = stablehlo.broadcast_in_dim %v2445, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2448 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2449 = stablehlo.add %v2447, %v2448 : tensor<64x2048x7x7xf32>
    %v2450 = stablehlo.rsqrt %v2449 : tensor<64x2048x7x7xf32>
    %v2451 = stablehlo.subtract %v2443, %v2446 : tensor<64x2048x7x7xf32>
    %v2452 = stablehlo.multiply %v2451, %v2450 : tensor<64x2048x7x7xf32>
    %v2453 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2454 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<64x2048x7x7xf32>
    %v2456 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2457 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2458 = stablehlo.reduce(%v2455 init: %v2456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2459 = stablehlo.divide %v2458, %v2457 : tensor<2048xf32>
    %v2460 = stablehlo.multiply %v2452, %v2455 : tensor<64x2048x7x7xf32>
    %v2461 = stablehlo.reduce(%v2460 init: %v2456) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2462 = stablehlo.divide %v2461, %v2457 : tensor<2048xf32>
    %v2463 = stablehlo.concatenate %v2459, %v2462, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2464 = stablehlo.concatenate %v2380, %v2463, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b2g3dst = "stablehlo.all_reduce"(%v2464) ({
    ^bb0(%aras4b2g3dst: tensor<f32>, %arbs4b2g3dst: tensor<f32>):
      %aradds4b2g3dst = stablehlo.add %aras4b2g3dst, %arbs4b2g3dst : tensor<f32>
      stablehlo.return %aradds4b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b2g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b2g3dst = stablehlo.divide %arsums4b2g3dst, %arns4b2g3dst : tensor<8192xf32>
    %v2465 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2466 = stablehlo.slice %armeans4b2g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2467 = stablehlo.slice %armeans4b2g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2468 = stablehlo.slice %armeans4b2g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2469 = stablehlo.slice %armeans4b2g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2470 = stablehlo.broadcast_in_dim %v2466, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2471 = stablehlo.broadcast_in_dim %v2467, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2472 = stablehlo.broadcast_in_dim %v2468, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2473 = stablehlo.broadcast_in_dim %v2469, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2474 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2475 = stablehlo.add %v2471, %v2474 : tensor<64x2048x7x7xf32>
    %v2476 = stablehlo.rsqrt %v2475 : tensor<64x2048x7x7xf32>
    %v2477 = stablehlo.subtract %v2465, %v2470 : tensor<64x2048x7x7xf32>
    %v2478 = stablehlo.multiply %v2477, %v2476 : tensor<64x2048x7x7xf32>
    %v2479 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2480 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2481 = stablehlo.multiply %v2479, %v2480 : tensor<64x2048x7x7xf32>
    %v2482 = stablehlo.subtract %v2481, %v2472 : tensor<64x2048x7x7xf32>
    %v2483 = stablehlo.multiply %v2478, %v2473 : tensor<64x2048x7x7xf32>
    %v2484 = stablehlo.subtract %v2482, %v2483 : tensor<64x2048x7x7xf32>
    %v2485 = stablehlo.multiply %v2476, %v2484 : tensor<64x2048x7x7xf32>
    %v2486 = stablehlo.reshape %v2485 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2487 = stablehlo.reshape %v2486 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2488 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2489 = stablehlo.transpose %v2488, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2490 = stablehlo.convert %v2487 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2491 = stablehlo.convert %v2489 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2492 = stablehlo.convolution(%v2490, %v2491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2493 = stablehlo.convert %v2492 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2494 = stablehlo.reshape %v2493 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2495 = stablehlo.reshape %v2494 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2496 = stablehlo.reshape %v2351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2497 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2498 = stablehlo.compare GT, %v2496, %v2497 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2499 = stablehlo.select %v2498, %v2495, %v2497 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2500 = stablehlo.reshape %v2499 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2501 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2502 = stablehlo.slice %v2336 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2503 = stablehlo.slice %v2336 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2502, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2505 = stablehlo.broadcast_in_dim %v2503, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2506 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2507 = stablehlo.add %v2505, %v2506 : tensor<64x512x7x7xf32>
    %v2508 = stablehlo.rsqrt %v2507 : tensor<64x512x7x7xf32>
    %v2509 = stablehlo.subtract %v2501, %v2504 : tensor<64x512x7x7xf32>
    %v2510 = stablehlo.multiply %v2509, %v2508 : tensor<64x512x7x7xf32>
    %v2511 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2512 = stablehlo.reshape %v2500 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2513 = stablehlo.multiply %v2511, %v2512 : tensor<64x512x7x7xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2515 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2516 = stablehlo.reduce(%v2513 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2517 = stablehlo.divide %v2516, %v2515 : tensor<512xf32>
    %v2518 = stablehlo.multiply %v2510, %v2513 : tensor<64x512x7x7xf32>
    %v2519 = stablehlo.reduce(%v2518 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2520 = stablehlo.divide %v2519, %v2515 : tensor<512xf32>
    %v2521 = stablehlo.concatenate %v2517, %v2520, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2522 = stablehlo.concatenate %v2336, %v2521, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b2g2dst = "stablehlo.all_reduce"(%v2522) ({
    ^bb0(%aras4b2g2dst: tensor<f32>, %arbs4b2g2dst: tensor<f32>):
      %aradds4b2g2dst = stablehlo.add %aras4b2g2dst, %arbs4b2g2dst : tensor<f32>
      stablehlo.return %aradds4b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g2dst = stablehlo.divide %arsums4b2g2dst, %arns4b2g2dst : tensor<2048xf32>
    %v2523 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2524 = stablehlo.slice %armeans4b2g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2525 = stablehlo.slice %armeans4b2g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2526 = stablehlo.slice %armeans4b2g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2527 = stablehlo.slice %armeans4b2g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2528 = stablehlo.broadcast_in_dim %v2524, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2529 = stablehlo.broadcast_in_dim %v2525, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2530 = stablehlo.broadcast_in_dim %v2526, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2531 = stablehlo.broadcast_in_dim %v2527, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2532 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2533 = stablehlo.add %v2529, %v2532 : tensor<64x512x7x7xf32>
    %v2534 = stablehlo.rsqrt %v2533 : tensor<64x512x7x7xf32>
    %v2535 = stablehlo.subtract %v2523, %v2528 : tensor<64x512x7x7xf32>
    %v2536 = stablehlo.multiply %v2535, %v2534 : tensor<64x512x7x7xf32>
    %v2537 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2538 = stablehlo.reshape %v2500 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2539 = stablehlo.multiply %v2537, %v2538 : tensor<64x512x7x7xf32>
    %v2540 = stablehlo.subtract %v2539, %v2530 : tensor<64x512x7x7xf32>
    %v2541 = stablehlo.multiply %v2536, %v2531 : tensor<64x512x7x7xf32>
    %v2542 = stablehlo.subtract %v2540, %v2541 : tensor<64x512x7x7xf32>
    %v2543 = stablehlo.multiply %v2534, %v2542 : tensor<64x512x7x7xf32>
    %v2544 = stablehlo.reshape %v2543 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2545 = stablehlo.reshape %v2544 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2546 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2547 = stablehlo.transpose %v2546, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2548 = stablehlo.convert %v2545 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2549 = stablehlo.convert %v2547 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2550 = stablehlo.convolution(%v2548, %v2549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2551 = stablehlo.convert %v2550 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2552 = stablehlo.reshape %v2551 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2553 = stablehlo.reshape %v2552 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2554 = stablehlo.reshape %v2307 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2555 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2556 = stablehlo.compare GT, %v2554, %v2555 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2557 = stablehlo.select %v2556, %v2553, %v2555 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2558 = stablehlo.reshape %v2557 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2559 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2560 = stablehlo.slice %v2292 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2561 = stablehlo.slice %v2292 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2562 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2563 = stablehlo.broadcast_in_dim %v2561, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2564 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2565 = stablehlo.add %v2563, %v2564 : tensor<64x512x7x7xf32>
    %v2566 = stablehlo.rsqrt %v2565 : tensor<64x512x7x7xf32>
    %v2567 = stablehlo.subtract %v2559, %v2562 : tensor<64x512x7x7xf32>
    %v2568 = stablehlo.multiply %v2567, %v2566 : tensor<64x512x7x7xf32>
    %v2569 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2570 = stablehlo.reshape %v2558 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2571 = stablehlo.multiply %v2569, %v2570 : tensor<64x512x7x7xf32>
    %v2572 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2573 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2574 = stablehlo.reduce(%v2571 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2575 = stablehlo.divide %v2574, %v2573 : tensor<512xf32>
    %v2576 = stablehlo.multiply %v2568, %v2571 : tensor<64x512x7x7xf32>
    %v2577 = stablehlo.reduce(%v2576 init: %v2572) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2578 = stablehlo.divide %v2577, %v2573 : tensor<512xf32>
    %v2579 = stablehlo.concatenate %v2575, %v2578, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2580 = stablehlo.concatenate %v2292, %v2579, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b2g1dst = "stablehlo.all_reduce"(%v2580) ({
    ^bb0(%aras4b2g1dst: tensor<f32>, %arbs4b2g1dst: tensor<f32>):
      %aradds4b2g1dst = stablehlo.add %aras4b2g1dst, %arbs4b2g1dst : tensor<f32>
      stablehlo.return %aradds4b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g1dst = stablehlo.divide %arsums4b2g1dst, %arns4b2g1dst : tensor<2048xf32>
    %v2581 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2582 = stablehlo.slice %armeans4b2g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2583 = stablehlo.slice %armeans4b2g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2584 = stablehlo.slice %armeans4b2g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2585 = stablehlo.slice %armeans4b2g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2586 = stablehlo.broadcast_in_dim %v2582, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2587 = stablehlo.broadcast_in_dim %v2583, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2584, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2589 = stablehlo.broadcast_in_dim %v2585, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2590 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2591 = stablehlo.add %v2587, %v2590 : tensor<64x512x7x7xf32>
    %v2592 = stablehlo.rsqrt %v2591 : tensor<64x512x7x7xf32>
    %v2593 = stablehlo.subtract %v2581, %v2586 : tensor<64x512x7x7xf32>
    %v2594 = stablehlo.multiply %v2593, %v2592 : tensor<64x512x7x7xf32>
    %v2595 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2596 = stablehlo.reshape %v2558 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2597 = stablehlo.multiply %v2595, %v2596 : tensor<64x512x7x7xf32>
    %v2598 = stablehlo.subtract %v2597, %v2588 : tensor<64x512x7x7xf32>
    %v2599 = stablehlo.multiply %v2594, %v2589 : tensor<64x512x7x7xf32>
    %v2600 = stablehlo.subtract %v2598, %v2599 : tensor<64x512x7x7xf32>
    %v2601 = stablehlo.multiply %v2592, %v2600 : tensor<64x512x7x7xf32>
    %v2602 = stablehlo.reshape %v2601 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2603 = stablehlo.reshape %v2602 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2604 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v2605 = stablehlo.transpose %v2604, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2606 = stablehlo.convert %v2603 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2607 = stablehlo.convert %v2605 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2608 = stablehlo.convolution(%v2606, %v2607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2609 = stablehlo.convert %v2608 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2610 = stablehlo.reshape %v2609 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2611 = stablehlo.reshape %v2610 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2612 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2613 = stablehlo.add %v2611, %v2612 : tensor<64x2048x7x7xf32>
    %v2614 = stablehlo.reshape %v2613 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2615 = stablehlo.reshape %v2265 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2616 = stablehlo.reshape %v2602 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2617 = stablehlo.transpose %v2615, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2618 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2619 = stablehlo.convert %v2617 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2620 = stablehlo.convert %v2618 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2621 = stablehlo.convolution(%v2619, %v2620)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v2622 = stablehlo.convert %v2621 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v2623 = stablehlo.transpose %v2622, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2624 = stablehlo.reshape %v2273 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2625 = stablehlo.slice %v2292 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2626 = stablehlo.slice %v2292 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2627 = stablehlo.broadcast_in_dim %v2625, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2628 = stablehlo.broadcast_in_dim %v2626, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2629 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2630 = stablehlo.add %v2628, %v2629 : tensor<64x512x7x7xf32>
    %v2631 = stablehlo.rsqrt %v2630 : tensor<64x512x7x7xf32>
    %v2632 = stablehlo.subtract %v2624, %v2627 : tensor<64x512x7x7xf32>
    %v2633 = stablehlo.multiply %v2632, %v2631 : tensor<64x512x7x7xf32>
    %v2634 = stablehlo.reshape %v2558 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2635 = stablehlo.multiply %v2634, %v2633 : tensor<64x512x7x7xf32>
    %v2636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2637 = stablehlo.reduce(%v2635 init: %v2636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2638 = stablehlo.reshape %v2558 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2640 = stablehlo.reduce(%v2638 init: %v2639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2641 = stablehlo.reshape %v2309 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2642 = stablehlo.reshape %v2544 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2643 = stablehlo.transpose %v2641, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2644 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2645 = stablehlo.convert %v2643 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2646 = stablehlo.convert %v2644 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2647 = stablehlo.convolution(%v2645, %v2646)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2648 = stablehlo.convert %v2647 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2649 = stablehlo.transpose %v2648, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2650 = stablehlo.reshape %v2317 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2651 = stablehlo.slice %v2336 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2652 = stablehlo.slice %v2336 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2653 = stablehlo.broadcast_in_dim %v2651, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2654 = stablehlo.broadcast_in_dim %v2652, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2655 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2656 = stablehlo.add %v2654, %v2655 : tensor<64x512x7x7xf32>
    %v2657 = stablehlo.rsqrt %v2656 : tensor<64x512x7x7xf32>
    %v2658 = stablehlo.subtract %v2650, %v2653 : tensor<64x512x7x7xf32>
    %v2659 = stablehlo.multiply %v2658, %v2657 : tensor<64x512x7x7xf32>
    %v2660 = stablehlo.reshape %v2500 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2661 = stablehlo.multiply %v2660, %v2659 : tensor<64x512x7x7xf32>
    %v2662 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2663 = stablehlo.reduce(%v2661 init: %v2662) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2664 = stablehlo.reshape %v2500 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2666 = stablehlo.reduce(%v2664 init: %v2665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2667 = stablehlo.reshape %v2353 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2668 = stablehlo.reshape %v2486 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2669 = stablehlo.transpose %v2667, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2670 = stablehlo.transpose %v2668, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2671 = stablehlo.convert %v2669 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2672 = stablehlo.convert %v2670 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2673 = stablehlo.convolution(%v2671, %v2672)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2674 = stablehlo.convert %v2673 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2675 = stablehlo.transpose %v2674, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2676 = stablehlo.reshape %v2361 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2677 = stablehlo.slice %v2380 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2678 = stablehlo.slice %v2380 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2679 = stablehlo.broadcast_in_dim %v2677, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2680 = stablehlo.broadcast_in_dim %v2678, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2681 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2682 = stablehlo.add %v2680, %v2681 : tensor<64x2048x7x7xf32>
    %v2683 = stablehlo.rsqrt %v2682 : tensor<64x2048x7x7xf32>
    %v2684 = stablehlo.subtract %v2676, %v2679 : tensor<64x2048x7x7xf32>
    %v2685 = stablehlo.multiply %v2684, %v2683 : tensor<64x2048x7x7xf32>
    %v2686 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2687 = stablehlo.multiply %v2686, %v2685 : tensor<64x2048x7x7xf32>
    %v2688 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2689 = stablehlo.reduce(%v2687 init: %v2688) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2690 = stablehlo.reshape %v2442 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2691 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2692 = stablehlo.reduce(%v2690 init: %v2691) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2693 = stablehlo.reshape %v2614 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2694 = stablehlo.reshape %v2261 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2695 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2696 = stablehlo.compare GT, %v2694, %v2695 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2697 = stablehlo.select %v2696, %v2693, %v2695 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2698 = stablehlo.reshape %v2697 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2699 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2700 = stablehlo.slice %v2242 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2701 = stablehlo.slice %v2242 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2702 = stablehlo.broadcast_in_dim %v2700, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2703 = stablehlo.broadcast_in_dim %v2701, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2704 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2705 = stablehlo.add %v2703, %v2704 : tensor<64x2048x7x7xf32>
    %v2706 = stablehlo.rsqrt %v2705 : tensor<64x2048x7x7xf32>
    %v2707 = stablehlo.subtract %v2699, %v2702 : tensor<64x2048x7x7xf32>
    %v2708 = stablehlo.multiply %v2707, %v2706 : tensor<64x2048x7x7xf32>
    %v2709 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2710 = stablehlo.reshape %v2698 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2711 = stablehlo.multiply %v2709, %v2710 : tensor<64x2048x7x7xf32>
    %v2712 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2713 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2714 = stablehlo.reduce(%v2711 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2715 = stablehlo.divide %v2714, %v2713 : tensor<2048xf32>
    %v2716 = stablehlo.multiply %v2708, %v2711 : tensor<64x2048x7x7xf32>
    %v2717 = stablehlo.reduce(%v2716 init: %v2712) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2718 = stablehlo.divide %v2717, %v2713 : tensor<2048xf32>
    %v2719 = stablehlo.concatenate %v2715, %v2718, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2720 = stablehlo.concatenate %v2242, %v2719, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b1g3dst = "stablehlo.all_reduce"(%v2720) ({
    ^bb0(%aras4b1g3dst: tensor<f32>, %arbs4b1g3dst: tensor<f32>):
      %aradds4b1g3dst = stablehlo.add %aras4b1g3dst, %arbs4b1g3dst : tensor<f32>
      stablehlo.return %aradds4b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b1g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b1g3dst = stablehlo.divide %arsums4b1g3dst, %arns4b1g3dst : tensor<8192xf32>
    %v2721 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2722 = stablehlo.slice %armeans4b1g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2723 = stablehlo.slice %armeans4b1g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2724 = stablehlo.slice %armeans4b1g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2725 = stablehlo.slice %armeans4b1g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2726 = stablehlo.broadcast_in_dim %v2722, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2727 = stablehlo.broadcast_in_dim %v2723, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2728 = stablehlo.broadcast_in_dim %v2724, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2729 = stablehlo.broadcast_in_dim %v2725, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2730 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2731 = stablehlo.add %v2727, %v2730 : tensor<64x2048x7x7xf32>
    %v2732 = stablehlo.rsqrt %v2731 : tensor<64x2048x7x7xf32>
    %v2733 = stablehlo.subtract %v2721, %v2726 : tensor<64x2048x7x7xf32>
    %v2734 = stablehlo.multiply %v2733, %v2732 : tensor<64x2048x7x7xf32>
    %v2735 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2736 = stablehlo.reshape %v2698 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2737 = stablehlo.multiply %v2735, %v2736 : tensor<64x2048x7x7xf32>
    %v2738 = stablehlo.subtract %v2737, %v2728 : tensor<64x2048x7x7xf32>
    %v2739 = stablehlo.multiply %v2734, %v2729 : tensor<64x2048x7x7xf32>
    %v2740 = stablehlo.subtract %v2738, %v2739 : tensor<64x2048x7x7xf32>
    %v2741 = stablehlo.multiply %v2732, %v2740 : tensor<64x2048x7x7xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2743 = stablehlo.reshape %v2742 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2744 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2745 = stablehlo.transpose %v2744, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2746 = stablehlo.convert %v2743 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2747 = stablehlo.convert %v2745 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2748 = stablehlo.convolution(%v2746, %v2747)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2749 = stablehlo.convert %v2748 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2751 = stablehlo.reshape %v2750 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2752 = stablehlo.reshape %v2213 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2753 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2754 = stablehlo.compare GT, %v2752, %v2753 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2755 = stablehlo.select %v2754, %v2751, %v2753 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2756 = stablehlo.reshape %v2755 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2757 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2758 = stablehlo.slice %v2198 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2759 = stablehlo.slice %v2198 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2760 = stablehlo.broadcast_in_dim %v2758, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2761 = stablehlo.broadcast_in_dim %v2759, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2762 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2763 = stablehlo.add %v2761, %v2762 : tensor<64x512x7x7xf32>
    %v2764 = stablehlo.rsqrt %v2763 : tensor<64x512x7x7xf32>
    %v2765 = stablehlo.subtract %v2757, %v2760 : tensor<64x512x7x7xf32>
    %v2766 = stablehlo.multiply %v2765, %v2764 : tensor<64x512x7x7xf32>
    %v2767 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2768 = stablehlo.reshape %v2756 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2769 = stablehlo.multiply %v2767, %v2768 : tensor<64x512x7x7xf32>
    %v2770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2771 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2772 = stablehlo.reduce(%v2769 init: %v2770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2773 = stablehlo.divide %v2772, %v2771 : tensor<512xf32>
    %v2774 = stablehlo.multiply %v2766, %v2769 : tensor<64x512x7x7xf32>
    %v2775 = stablehlo.reduce(%v2774 init: %v2770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2776 = stablehlo.divide %v2775, %v2771 : tensor<512xf32>
    %v2777 = stablehlo.concatenate %v2773, %v2776, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2778 = stablehlo.concatenate %v2198, %v2777, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g2dst = "stablehlo.all_reduce"(%v2778) ({
    ^bb0(%aras4b1g2dst: tensor<f32>, %arbs4b1g2dst: tensor<f32>):
      %aradds4b1g2dst = stablehlo.add %aras4b1g2dst, %arbs4b1g2dst : tensor<f32>
      stablehlo.return %aradds4b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g2dst = stablehlo.divide %arsums4b1g2dst, %arns4b1g2dst : tensor<2048xf32>
    %v2779 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2780 = stablehlo.slice %armeans4b1g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2781 = stablehlo.slice %armeans4b1g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2782 = stablehlo.slice %armeans4b1g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2783 = stablehlo.slice %armeans4b1g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2784 = stablehlo.broadcast_in_dim %v2780, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2785 = stablehlo.broadcast_in_dim %v2781, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2786 = stablehlo.broadcast_in_dim %v2782, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2787 = stablehlo.broadcast_in_dim %v2783, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2788 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2789 = stablehlo.add %v2785, %v2788 : tensor<64x512x7x7xf32>
    %v2790 = stablehlo.rsqrt %v2789 : tensor<64x512x7x7xf32>
    %v2791 = stablehlo.subtract %v2779, %v2784 : tensor<64x512x7x7xf32>
    %v2792 = stablehlo.multiply %v2791, %v2790 : tensor<64x512x7x7xf32>
    %v2793 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2794 = stablehlo.reshape %v2756 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2795 = stablehlo.multiply %v2793, %v2794 : tensor<64x512x7x7xf32>
    %v2796 = stablehlo.subtract %v2795, %v2786 : tensor<64x512x7x7xf32>
    %v2797 = stablehlo.multiply %v2792, %v2787 : tensor<64x512x7x7xf32>
    %v2798 = stablehlo.subtract %v2796, %v2797 : tensor<64x512x7x7xf32>
    %v2799 = stablehlo.multiply %v2790, %v2798 : tensor<64x512x7x7xf32>
    %v2800 = stablehlo.reshape %v2799 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2801 = stablehlo.reshape %v2800 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2802 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2803 = stablehlo.transpose %v2802, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2804 = stablehlo.convert %v2801 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2805 = stablehlo.convert %v2803 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2806 = stablehlo.convolution(%v2804, %v2805)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2807 = stablehlo.convert %v2806 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2808 = stablehlo.reshape %v2807 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2809 = stablehlo.reshape %v2808 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2810 = stablehlo.reshape %v2169 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2811 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2812 = stablehlo.compare GT, %v2810, %v2811 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2813 = stablehlo.select %v2812, %v2809, %v2811 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2815 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2816 = stablehlo.slice %v2154 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2817 = stablehlo.slice %v2154 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2818 = stablehlo.broadcast_in_dim %v2816, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2819 = stablehlo.broadcast_in_dim %v2817, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2820 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2821 = stablehlo.add %v2819, %v2820 : tensor<64x512x7x7xf32>
    %v2822 = stablehlo.rsqrt %v2821 : tensor<64x512x7x7xf32>
    %v2823 = stablehlo.subtract %v2815, %v2818 : tensor<64x512x7x7xf32>
    %v2824 = stablehlo.multiply %v2823, %v2822 : tensor<64x512x7x7xf32>
    %v2825 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
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
    %v2836 = stablehlo.concatenate %v2154, %v2835, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g1dst = "stablehlo.all_reduce"(%v2836) ({
    ^bb0(%aras4b1g1dst: tensor<f32>, %arbs4b1g1dst: tensor<f32>):
      %aradds4b1g1dst = stablehlo.add %aras4b1g1dst, %arbs4b1g1dst : tensor<f32>
      stablehlo.return %aradds4b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g1dst = stablehlo.divide %arsums4b1g1dst, %arns4b1g1dst : tensor<2048xf32>
    %v2837 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2838 = stablehlo.slice %armeans4b1g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2839 = stablehlo.slice %armeans4b1g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2840 = stablehlo.slice %armeans4b1g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2841 = stablehlo.slice %armeans4b1g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2842 = stablehlo.broadcast_in_dim %v2838, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2843 = stablehlo.broadcast_in_dim %v2839, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2844 = stablehlo.broadcast_in_dim %v2840, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2845 = stablehlo.broadcast_in_dim %v2841, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2846 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2847 = stablehlo.add %v2843, %v2846 : tensor<64x512x7x7xf32>
    %v2848 = stablehlo.rsqrt %v2847 : tensor<64x512x7x7xf32>
    %v2849 = stablehlo.subtract %v2837, %v2842 : tensor<64x512x7x7xf32>
    %v2850 = stablehlo.multiply %v2849, %v2848 : tensor<64x512x7x7xf32>
    %v2851 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2852 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2853 = stablehlo.multiply %v2851, %v2852 : tensor<64x512x7x7xf32>
    %v2854 = stablehlo.subtract %v2853, %v2844 : tensor<64x512x7x7xf32>
    %v2855 = stablehlo.multiply %v2850, %v2845 : tensor<64x512x7x7xf32>
    %v2856 = stablehlo.subtract %v2854, %v2855 : tensor<64x512x7x7xf32>
    %v2857 = stablehlo.multiply %v2848, %v2856 : tensor<64x512x7x7xf32>
    %v2858 = stablehlo.reshape %v2857 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2859 = stablehlo.reshape %v2858 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2860 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v2861 = stablehlo.transpose %v2860, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2862 = stablehlo.convert %v2859 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2863 = stablehlo.convert %v2861 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v2864 = stablehlo.convolution(%v2862, %v2863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v2865 = stablehlo.convert %v2864 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v2866 = stablehlo.reshape %v2865 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2867 = stablehlo.reshape %v2866 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2868 = stablehlo.reshape %v2698 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2869 = stablehlo.add %v2867, %v2868 : tensor<64x2048x7x7xf32>
    %v2870 = stablehlo.reshape %v2869 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2871 = stablehlo.reshape %v2127 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2872 = stablehlo.reshape %v2858 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2873 = stablehlo.transpose %v2871, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2874 = stablehlo.transpose %v2872, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2875 = stablehlo.convert %v2873 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2876 = stablehlo.convert %v2874 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2877 = stablehlo.convolution(%v2875, %v2876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v2878 = stablehlo.convert %v2877 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v2879 = stablehlo.transpose %v2878, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2880 = stablehlo.reshape %v2135 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2881 = stablehlo.slice %v2154 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2882 = stablehlo.slice %v2154 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2883 = stablehlo.broadcast_in_dim %v2881, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2884 = stablehlo.broadcast_in_dim %v2882, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2885 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2886 = stablehlo.add %v2884, %v2885 : tensor<64x512x7x7xf32>
    %v2887 = stablehlo.rsqrt %v2886 : tensor<64x512x7x7xf32>
    %v2888 = stablehlo.subtract %v2880, %v2883 : tensor<64x512x7x7xf32>
    %v2889 = stablehlo.multiply %v2888, %v2887 : tensor<64x512x7x7xf32>
    %v2890 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2891 = stablehlo.multiply %v2890, %v2889 : tensor<64x512x7x7xf32>
    %v2892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2893 = stablehlo.reduce(%v2891 init: %v2892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2894 = stablehlo.reshape %v2814 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2896 = stablehlo.reduce(%v2894 init: %v2895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2897 = stablehlo.reshape %v2171 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2898 = stablehlo.reshape %v2800 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2899 = stablehlo.transpose %v2897, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2900 = stablehlo.transpose %v2898, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2901 = stablehlo.convert %v2899 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2902 = stablehlo.convert %v2900 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2903 = stablehlo.convolution(%v2901, %v2902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2904 = stablehlo.convert %v2903 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2905 = stablehlo.transpose %v2904, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2906 = stablehlo.reshape %v2179 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2907 = stablehlo.slice %v2198 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2908 = stablehlo.slice %v2198 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2909 = stablehlo.broadcast_in_dim %v2907, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2910 = stablehlo.broadcast_in_dim %v2908, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2911 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2912 = stablehlo.add %v2910, %v2911 : tensor<64x512x7x7xf32>
    %v2913 = stablehlo.rsqrt %v2912 : tensor<64x512x7x7xf32>
    %v2914 = stablehlo.subtract %v2906, %v2909 : tensor<64x512x7x7xf32>
    %v2915 = stablehlo.multiply %v2914, %v2913 : tensor<64x512x7x7xf32>
    %v2916 = stablehlo.reshape %v2756 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2917 = stablehlo.multiply %v2916, %v2915 : tensor<64x512x7x7xf32>
    %v2918 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2919 = stablehlo.reduce(%v2917 init: %v2918) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2920 = stablehlo.reshape %v2756 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2921 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2922 = stablehlo.reduce(%v2920 init: %v2921) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2923 = stablehlo.reshape %v2215 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2924 = stablehlo.reshape %v2742 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2925 = stablehlo.transpose %v2923, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2926 = stablehlo.transpose %v2924, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2927 = stablehlo.convert %v2925 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2928 = stablehlo.convert %v2926 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2929 = stablehlo.convolution(%v2927, %v2928)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2930 = stablehlo.convert %v2929 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2931 = stablehlo.transpose %v2930, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2932 = stablehlo.reshape %v2223 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2933 = stablehlo.slice %v2242 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2934 = stablehlo.slice %v2242 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2935 = stablehlo.broadcast_in_dim %v2933, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2936 = stablehlo.broadcast_in_dim %v2934, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2937 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2938 = stablehlo.add %v2936, %v2937 : tensor<64x2048x7x7xf32>
    %v2939 = stablehlo.rsqrt %v2938 : tensor<64x2048x7x7xf32>
    %v2940 = stablehlo.subtract %v2932, %v2935 : tensor<64x2048x7x7xf32>
    %v2941 = stablehlo.multiply %v2940, %v2939 : tensor<64x2048x7x7xf32>
    %v2942 = stablehlo.reshape %v2698 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2943 = stablehlo.multiply %v2942, %v2941 : tensor<64x2048x7x7xf32>
    %v2944 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2945 = stablehlo.reduce(%v2943 init: %v2944) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2946 = stablehlo.reshape %v2698 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2948 = stablehlo.reduce(%v2946 init: %v2947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2949 = stablehlo.reshape %v2870 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2950 = stablehlo.reshape %v2125 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2951 = stablehlo.constant dense<0.0> : tensor<64x2048x7x7xf32>
    %v2952 = stablehlo.compare GT, %v2950, %v2951 : (tensor<64x2048x7x7xf32>, tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xi1>
    %v2953 = stablehlo.select %v2952, %v2949, %v2951 : tensor<64x2048x7x7xi1>, tensor<64x2048x7x7xf32>
    %v2954 = stablehlo.reshape %v2953 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2955 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2956 = stablehlo.slice %v2067 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2957 = stablehlo.slice %v2067 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v2958 = stablehlo.broadcast_in_dim %v2956, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2959 = stablehlo.broadcast_in_dim %v2957, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2960 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2961 = stablehlo.add %v2959, %v2960 : tensor<64x2048x7x7xf32>
    %v2962 = stablehlo.rsqrt %v2961 : tensor<64x2048x7x7xf32>
    %v2963 = stablehlo.subtract %v2955, %v2958 : tensor<64x2048x7x7xf32>
    %v2964 = stablehlo.multiply %v2963, %v2962 : tensor<64x2048x7x7xf32>
    %v2965 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2966 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2967 = stablehlo.multiply %v2965, %v2966 : tensor<64x2048x7x7xf32>
    %v2968 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2969 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v2970 = stablehlo.reduce(%v2967 init: %v2968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2971 = stablehlo.divide %v2970, %v2969 : tensor<2048xf32>
    %v2972 = stablehlo.multiply %v2964, %v2967 : tensor<64x2048x7x7xf32>
    %v2973 = stablehlo.reduce(%v2972 init: %v2968) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2974 = stablehlo.divide %v2973, %v2969 : tensor<2048xf32>
    %v2975 = stablehlo.concatenate %v2971, %v2974, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v2976 = stablehlo.concatenate %v2067, %v2975, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b0g3dst = "stablehlo.all_reduce"(%v2976) ({
    ^bb0(%aras4b0g3dst: tensor<f32>, %arbs4b0g3dst: tensor<f32>):
      %aradds4b0g3dst = stablehlo.add %aras4b0g3dst, %arbs4b0g3dst : tensor<f32>
      stablehlo.return %aradds4b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b0g3dst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b0g3dst = stablehlo.divide %arsums4b0g3dst, %arns4b0g3dst : tensor<8192xf32>
    %v2977 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2978 = stablehlo.slice %armeans4b0g3dst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2979 = stablehlo.slice %armeans4b0g3dst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2980 = stablehlo.slice %armeans4b0g3dst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2981 = stablehlo.slice %armeans4b0g3dst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v2982 = stablehlo.broadcast_in_dim %v2978, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2979, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2984 = stablehlo.broadcast_in_dim %v2980, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2985 = stablehlo.broadcast_in_dim %v2981, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2986 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2987 = stablehlo.add %v2983, %v2986 : tensor<64x2048x7x7xf32>
    %v2988 = stablehlo.rsqrt %v2987 : tensor<64x2048x7x7xf32>
    %v2989 = stablehlo.subtract %v2977, %v2982 : tensor<64x2048x7x7xf32>
    %v2990 = stablehlo.multiply %v2989, %v2988 : tensor<64x2048x7x7xf32>
    %v2991 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2992 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2993 = stablehlo.multiply %v2991, %v2992 : tensor<64x2048x7x7xf32>
    %v2994 = stablehlo.subtract %v2993, %v2984 : tensor<64x2048x7x7xf32>
    %v2995 = stablehlo.multiply %v2990, %v2985 : tensor<64x2048x7x7xf32>
    %v2996 = stablehlo.subtract %v2994, %v2995 : tensor<64x2048x7x7xf32>
    %v2997 = stablehlo.multiply %v2988, %v2996 : tensor<64x2048x7x7xf32>
    %v2998 = stablehlo.reshape %v2997 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2999 = stablehlo.reshape %v2998 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3000 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v3001 = stablehlo.transpose %v3000, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v3002 = stablehlo.convert %v2999 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v3003 = stablehlo.convert %v3001 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v3004 = stablehlo.convolution(%v3002, %v3003)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v3005 = stablehlo.convert %v3004 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v3006 = stablehlo.reshape %v3005 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v3007 = stablehlo.reshape %v3006 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3008 = stablehlo.reshape %v2038 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3009 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v3010 = stablehlo.compare GT, %v3008, %v3009 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v3011 = stablehlo.select %v3010, %v3007, %v3009 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v3012 = stablehlo.reshape %v3011 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v3013 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3014 = stablehlo.slice %v2023 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3015 = stablehlo.slice %v2023 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3016 = stablehlo.broadcast_in_dim %v3014, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3017 = stablehlo.broadcast_in_dim %v3015, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3018 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v3019 = stablehlo.add %v3017, %v3018 : tensor<64x512x7x7xf32>
    %v3020 = stablehlo.rsqrt %v3019 : tensor<64x512x7x7xf32>
    %v3021 = stablehlo.subtract %v3013, %v3016 : tensor<64x512x7x7xf32>
    %v3022 = stablehlo.multiply %v3021, %v3020 : tensor<64x512x7x7xf32>
    %v3023 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3024 = stablehlo.reshape %v3012 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3025 = stablehlo.multiply %v3023, %v3024 : tensor<64x512x7x7xf32>
    %v3026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3027 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v3028 = stablehlo.reduce(%v3025 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3029 = stablehlo.divide %v3028, %v3027 : tensor<512xf32>
    %v3030 = stablehlo.multiply %v3022, %v3025 : tensor<64x512x7x7xf32>
    %v3031 = stablehlo.reduce(%v3030 init: %v3026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3032 = stablehlo.divide %v3031, %v3027 : tensor<512xf32>
    %v3033 = stablehlo.concatenate %v3029, %v3032, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v3034 = stablehlo.concatenate %v2023, %v3033, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g2dst = "stablehlo.all_reduce"(%v3034) ({
    ^bb0(%aras4b0g2dst: tensor<f32>, %arbs4b0g2dst: tensor<f32>):
      %aradds4b0g2dst = stablehlo.add %aras4b0g2dst, %arbs4b0g2dst : tensor<f32>
      stablehlo.return %aradds4b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g2dst = stablehlo.divide %arsums4b0g2dst, %arns4b0g2dst : tensor<2048xf32>
    %v3035 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3036 = stablehlo.slice %armeans4b0g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3037 = stablehlo.slice %armeans4b0g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3038 = stablehlo.slice %armeans4b0g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3039 = stablehlo.slice %armeans4b0g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3040 = stablehlo.broadcast_in_dim %v3036, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3041 = stablehlo.broadcast_in_dim %v3037, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3042 = stablehlo.broadcast_in_dim %v3038, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3043 = stablehlo.broadcast_in_dim %v3039, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3044 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v3045 = stablehlo.add %v3041, %v3044 : tensor<64x512x7x7xf32>
    %v3046 = stablehlo.rsqrt %v3045 : tensor<64x512x7x7xf32>
    %v3047 = stablehlo.subtract %v3035, %v3040 : tensor<64x512x7x7xf32>
    %v3048 = stablehlo.multiply %v3047, %v3046 : tensor<64x512x7x7xf32>
    %v3049 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3050 = stablehlo.reshape %v3012 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3051 = stablehlo.multiply %v3049, %v3050 : tensor<64x512x7x7xf32>
    %v3052 = stablehlo.subtract %v3051, %v3042 : tensor<64x512x7x7xf32>
    %v3053 = stablehlo.multiply %v3048, %v3043 : tensor<64x512x7x7xf32>
    %v3054 = stablehlo.subtract %v3052, %v3053 : tensor<64x512x7x7xf32>
    %v3055 = stablehlo.multiply %v3046, %v3054 : tensor<64x512x7x7xf32>
    %v3056 = stablehlo.reshape %v3055 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v3057 = stablehlo.reshape %v3056 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3058 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3059 = stablehlo.pad %v3057, %v3058, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v3060 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v3061 = stablehlo.transpose %v3060, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v3062 = stablehlo.convert %v3059 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v3063 = stablehlo.convert %v3061 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v3064 = stablehlo.convolution(%v3062, %v3063)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x14x14xbf16>
    %v3065 = stablehlo.convert %v3064 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v3066 = stablehlo.reshape %v3065 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v3067 = stablehlo.reshape %v3066 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3068 = stablehlo.reshape %v1994 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3069 = stablehlo.constant dense<0.0> : tensor<64x512x14x14xf32>
    %v3070 = stablehlo.compare GT, %v3068, %v3069 : (tensor<64x512x14x14xf32>, tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xi1>
    %v3071 = stablehlo.select %v3070, %v3067, %v3069 : tensor<64x512x14x14xi1>, tensor<64x512x14x14xf32>
    %v3072 = stablehlo.reshape %v3071 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v3073 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3074 = stablehlo.slice %v1979 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3075 = stablehlo.slice %v1979 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3076 = stablehlo.broadcast_in_dim %v3074, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3077 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3078 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v3079 = stablehlo.add %v3077, %v3078 : tensor<64x512x14x14xf32>
    %v3080 = stablehlo.rsqrt %v3079 : tensor<64x512x14x14xf32>
    %v3081 = stablehlo.subtract %v3073, %v3076 : tensor<64x512x14x14xf32>
    %v3082 = stablehlo.multiply %v3081, %v3080 : tensor<64x512x14x14xf32>
    %v3083 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3084 = stablehlo.reshape %v3072 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3085 = stablehlo.multiply %v3083, %v3084 : tensor<64x512x14x14xf32>
    %v3086 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3087 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v3088 = stablehlo.reduce(%v3085 init: %v3086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v3089 = stablehlo.divide %v3088, %v3087 : tensor<512xf32>
    %v3090 = stablehlo.multiply %v3082, %v3085 : tensor<64x512x14x14xf32>
    %v3091 = stablehlo.reduce(%v3090 init: %v3086) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v3092 = stablehlo.divide %v3091, %v3087 : tensor<512xf32>
    %v3093 = stablehlo.concatenate %v3089, %v3092, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v3094 = stablehlo.concatenate %v1979, %v3093, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g1dst = "stablehlo.all_reduce"(%v3094) ({
    ^bb0(%aras4b0g1dst: tensor<f32>, %arbs4b0g1dst: tensor<f32>):
      %aradds4b0g1dst = stablehlo.add %aras4b0g1dst, %arbs4b0g1dst : tensor<f32>
      stablehlo.return %aradds4b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g1dst = stablehlo.divide %arsums4b0g1dst, %arns4b0g1dst : tensor<2048xf32>
    %v3095 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3096 = stablehlo.slice %armeans4b0g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3097 = stablehlo.slice %armeans4b0g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3098 = stablehlo.slice %armeans4b0g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3099 = stablehlo.slice %armeans4b0g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v3100 = stablehlo.broadcast_in_dim %v3096, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3101 = stablehlo.broadcast_in_dim %v3097, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3098, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3103 = stablehlo.broadcast_in_dim %v3099, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3104 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v3105 = stablehlo.add %v3101, %v3104 : tensor<64x512x14x14xf32>
    %v3106 = stablehlo.rsqrt %v3105 : tensor<64x512x14x14xf32>
    %v3107 = stablehlo.subtract %v3095, %v3100 : tensor<64x512x14x14xf32>
    %v3108 = stablehlo.multiply %v3107, %v3106 : tensor<64x512x14x14xf32>
    %v3109 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3110 = stablehlo.reshape %v3072 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3111 = stablehlo.multiply %v3109, %v3110 : tensor<64x512x14x14xf32>
    %v3112 = stablehlo.subtract %v3111, %v3102 : tensor<64x512x14x14xf32>
    %v3113 = stablehlo.multiply %v3108, %v3103 : tensor<64x512x14x14xf32>
    %v3114 = stablehlo.subtract %v3112, %v3113 : tensor<64x512x14x14xf32>
    %v3115 = stablehlo.multiply %v3106, %v3114 : tensor<64x512x14x14xf32>
    %v3116 = stablehlo.reshape %v3115 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v3117 = stablehlo.reshape %v3116 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3118 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v3119 = stablehlo.transpose %v3118, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3120 = stablehlo.convert %v3117 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v3121 = stablehlo.convert %v3119 : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v3122 = stablehlo.convolution(%v3120, %v3121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3123 = stablehlo.convert %v3122 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3124 = stablehlo.reshape %v3123 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3125 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3126 = stablehlo.slice %v2109 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3127 = stablehlo.slice %v2109 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3128 = stablehlo.broadcast_in_dim %v3126, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3129 = stablehlo.broadcast_in_dim %v3127, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3130 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3131 = stablehlo.add %v3129, %v3130 : tensor<64x2048x7x7xf32>
    %v3132 = stablehlo.rsqrt %v3131 : tensor<64x2048x7x7xf32>
    %v3133 = stablehlo.subtract %v3125, %v3128 : tensor<64x2048x7x7xf32>
    %v3134 = stablehlo.multiply %v3133, %v3132 : tensor<64x2048x7x7xf32>
    %v3135 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3136 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3137 = stablehlo.multiply %v3135, %v3136 : tensor<64x2048x7x7xf32>
    %v3138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3139 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v3140 = stablehlo.reduce(%v3137 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3141 = stablehlo.divide %v3140, %v3139 : tensor<2048xf32>
    %v3142 = stablehlo.multiply %v3134, %v3137 : tensor<64x2048x7x7xf32>
    %v3143 = stablehlo.reduce(%v3142 init: %v3138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3144 = stablehlo.divide %v3143, %v3139 : tensor<2048xf32>
    %v3145 = stablehlo.concatenate %v3141, %v3144, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %v3146 = stablehlo.concatenate %v2109, %v3145, dim = 0 : (tensor<4096xf32>, tensor<4096xf32>) -> tensor<8192xf32>
    %arsums4b0gpdst = "stablehlo.all_reduce"(%v3146) ({
    ^bb0(%aras4b0gpdst: tensor<f32>, %arbs4b0gpdst: tensor<f32>):
      %aradds4b0gpdst = stablehlo.add %aras4b0gpdst, %arbs4b0gpdst : tensor<f32>
      stablehlo.return %aradds4b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<8192xf32>) -> tensor<8192xf32>
    %arns4b0gpdst = stablehlo.constant dense<4.0> : tensor<8192xf32>
    %armeans4b0gpdst = stablehlo.divide %arsums4b0gpdst, %arns4b0gpdst : tensor<8192xf32>
    %v3147 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3148 = stablehlo.slice %armeans4b0gpdst [0:2048] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v3149 = stablehlo.slice %armeans4b0gpdst [2048:4096] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v3150 = stablehlo.slice %armeans4b0gpdst [4096:6144] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v3151 = stablehlo.slice %armeans4b0gpdst [6144:8192] : (tensor<8192xf32>) -> tensor<2048xf32>
    %v3152 = stablehlo.broadcast_in_dim %v3148, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3153 = stablehlo.broadcast_in_dim %v3149, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3154 = stablehlo.broadcast_in_dim %v3150, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3155 = stablehlo.broadcast_in_dim %v3151, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3156 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3157 = stablehlo.add %v3153, %v3156 : tensor<64x2048x7x7xf32>
    %v3158 = stablehlo.rsqrt %v3157 : tensor<64x2048x7x7xf32>
    %v3159 = stablehlo.subtract %v3147, %v3152 : tensor<64x2048x7x7xf32>
    %v3160 = stablehlo.multiply %v3159, %v3158 : tensor<64x2048x7x7xf32>
    %v3161 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3162 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3163 = stablehlo.multiply %v3161, %v3162 : tensor<64x2048x7x7xf32>
    %v3164 = stablehlo.subtract %v3163, %v3154 : tensor<64x2048x7x7xf32>
    %v3165 = stablehlo.multiply %v3160, %v3155 : tensor<64x2048x7x7xf32>
    %v3166 = stablehlo.subtract %v3164, %v3165 : tensor<64x2048x7x7xf32>
    %v3167 = stablehlo.multiply %v3158, %v3166 : tensor<64x2048x7x7xf32>
    %v3168 = stablehlo.reshape %v3167 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v3169 = stablehlo.reshape %v3168 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3171 = stablehlo.pad %v3169, %v3170, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v3172 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v3173 = stablehlo.transpose %v3172, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v3174 = stablehlo.convert %v3171 : (tensor<64x2048x14x14xf32>) -> tensor<64x2048x14x14xbf16>
    %v3175 = stablehlo.convert %v3173 : (tensor<1024x2048x1x1xf32>) -> tensor<1024x2048x1x1xbf16>
    %v3176 = stablehlo.convolution(%v3174, %v3175)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x14x14xbf16>, tensor<1024x2048x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3177 = stablehlo.convert %v3176 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3178 = stablehlo.reshape %v3177 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3179 = stablehlo.reshape %v3124 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3180 = stablehlo.reshape %v3178 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3181 = stablehlo.add %v3179, %v3180 : tensor<64x1024x14x14xf32>
    %v3182 = stablehlo.reshape %v3181 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3183 = stablehlo.reshape %v1952 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3184 = stablehlo.reshape %v3116 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3185 = stablehlo.transpose %v3183, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3186 = stablehlo.transpose %v3184, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v3187 = stablehlo.convert %v3185 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3188 = stablehlo.convert %v3186 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v3189 = stablehlo.convolution(%v3187, %v3188)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<1024x512x1x1xbf16>
    %v3190 = stablehlo.convert %v3189 : (tensor<1024x512x1x1xbf16>) -> tensor<1024x512x1x1xf32>
    %v3191 = stablehlo.transpose %v3190, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3192 = stablehlo.reshape %v1960 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3193 = stablehlo.slice %v1979 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3194 = stablehlo.slice %v1979 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3195 = stablehlo.broadcast_in_dim %v3193, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3196 = stablehlo.broadcast_in_dim %v3194, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v3197 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v3198 = stablehlo.add %v3196, %v3197 : tensor<64x512x14x14xf32>
    %v3199 = stablehlo.rsqrt %v3198 : tensor<64x512x14x14xf32>
    %v3200 = stablehlo.subtract %v3192, %v3195 : tensor<64x512x14x14xf32>
    %v3201 = stablehlo.multiply %v3200, %v3199 : tensor<64x512x14x14xf32>
    %v3202 = stablehlo.reshape %v3072 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3203 = stablehlo.multiply %v3202, %v3201 : tensor<64x512x14x14xf32>
    %v3204 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3205 = stablehlo.reduce(%v3203 init: %v3204) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v3206 = stablehlo.reshape %v3072 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3208 = stablehlo.reduce(%v3206 init: %v3207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v3209 = stablehlo.reshape %v1996 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v3210 = stablehlo.reshape %v3056 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3211 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3212 = stablehlo.pad %v3210, %v3211, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v3213 = stablehlo.transpose %v3209, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v3214 = stablehlo.transpose %v3212, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v3215 = stablehlo.convert %v3213 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v3216 = stablehlo.convert %v3214 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v3217 = stablehlo.convolution(%v3215, %v3216)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<512x512x3x3xbf16>
    %v3218 = stablehlo.convert %v3217 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v3219 = stablehlo.transpose %v3218, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v3220 = stablehlo.reshape %v2004 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3221 = stablehlo.slice %v2023 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3222 = stablehlo.slice %v2023 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v3223 = stablehlo.broadcast_in_dim %v3221, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3224 = stablehlo.broadcast_in_dim %v3222, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v3225 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v3226 = stablehlo.add %v3224, %v3225 : tensor<64x512x7x7xf32>
    %v3227 = stablehlo.rsqrt %v3226 : tensor<64x512x7x7xf32>
    %v3228 = stablehlo.subtract %v3220, %v3223 : tensor<64x512x7x7xf32>
    %v3229 = stablehlo.multiply %v3228, %v3227 : tensor<64x512x7x7xf32>
    %v3230 = stablehlo.reshape %v3012 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3231 = stablehlo.multiply %v3230, %v3229 : tensor<64x512x7x7xf32>
    %v3232 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3233 = stablehlo.reduce(%v3231 init: %v3232) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3234 = stablehlo.reshape %v3012 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3236 = stablehlo.reduce(%v3234 init: %v3235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v3237 = stablehlo.reshape %v2040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v3238 = stablehlo.reshape %v2998 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3239 = stablehlo.transpose %v3237, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v3240 = stablehlo.transpose %v3238, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v3241 = stablehlo.convert %v3239 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v3242 = stablehlo.convert %v3240 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v3243 = stablehlo.convolution(%v3241, %v3242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v3244 = stablehlo.convert %v3243 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v3245 = stablehlo.transpose %v3244, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v3246 = stablehlo.reshape %v2048 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3247 = stablehlo.slice %v2067 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3248 = stablehlo.slice %v2067 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3247, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3250 = stablehlo.broadcast_in_dim %v3248, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3251 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3252 = stablehlo.add %v3250, %v3251 : tensor<64x2048x7x7xf32>
    %v3253 = stablehlo.rsqrt %v3252 : tensor<64x2048x7x7xf32>
    %v3254 = stablehlo.subtract %v3246, %v3249 : tensor<64x2048x7x7xf32>
    %v3255 = stablehlo.multiply %v3254, %v3253 : tensor<64x2048x7x7xf32>
    %v3256 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3257 = stablehlo.multiply %v3256, %v3255 : tensor<64x2048x7x7xf32>
    %v3258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3259 = stablehlo.reduce(%v3257 init: %v3258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3260 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3262 = stablehlo.reduce(%v3260 init: %v3261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3263 = stablehlo.reshape %v1952 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3264 = stablehlo.reshape %v3168 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3266 = stablehlo.pad %v3264, %v3265, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v3267 = stablehlo.transpose %v3263, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3268 = stablehlo.transpose %v3266, dims = [1, 0, 2, 3] : (tensor<64x2048x14x14xf32>) -> tensor<2048x64x14x14xf32>
    %v3269 = stablehlo.convert %v3267 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3270 = stablehlo.convert %v3268 : (tensor<2048x64x14x14xf32>) -> tensor<2048x64x14x14xbf16>
    %v3271 = stablehlo.convolution(%v3269, %v3270)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<2048x64x14x14xbf16>) -> tensor<1024x2048x1x1xbf16>
    %v3272 = stablehlo.convert %v3271 : (tensor<1024x2048x1x1xbf16>) -> tensor<1024x2048x1x1xf32>
    %v3273 = stablehlo.transpose %v3272, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v3274 = stablehlo.reshape %v2090 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3275 = stablehlo.slice %v2109 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3276 = stablehlo.slice %v2109 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3275, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3278 = stablehlo.broadcast_in_dim %v3276, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v3279 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v3280 = stablehlo.add %v3278, %v3279 : tensor<64x2048x7x7xf32>
    %v3281 = stablehlo.rsqrt %v3280 : tensor<64x2048x7x7xf32>
    %v3282 = stablehlo.subtract %v3274, %v3277 : tensor<64x2048x7x7xf32>
    %v3283 = stablehlo.multiply %v3282, %v3281 : tensor<64x2048x7x7xf32>
    %v3284 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3285 = stablehlo.multiply %v3284, %v3283 : tensor<64x2048x7x7xf32>
    %v3286 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3287 = stablehlo.reduce(%v3285 init: %v3286) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3288 = stablehlo.reshape %v2954 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v3289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3290 = stablehlo.reduce(%v3288 init: %v3289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v3291 = stablehlo.reshape %v3182 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3292 = stablehlo.reshape %v1948 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3293 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3294 = stablehlo.compare GT, %v3292, %v3293 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3295 = stablehlo.select %v3294, %v3291, %v3293 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3296 = stablehlo.reshape %v3295 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3297 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3298 = stablehlo.slice %v1929 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3299 = stablehlo.slice %v1929 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3300 = stablehlo.broadcast_in_dim %v3298, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3301 = stablehlo.broadcast_in_dim %v3299, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3302 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3303 = stablehlo.add %v3301, %v3302 : tensor<64x1024x14x14xf32>
    %v3304 = stablehlo.rsqrt %v3303 : tensor<64x1024x14x14xf32>
    %v3305 = stablehlo.subtract %v3297, %v3300 : tensor<64x1024x14x14xf32>
    %v3306 = stablehlo.multiply %v3305, %v3304 : tensor<64x1024x14x14xf32>
    %v3307 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3308 = stablehlo.reshape %v3296 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3309 = stablehlo.multiply %v3307, %v3308 : tensor<64x1024x14x14xf32>
    %v3310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3311 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3312 = stablehlo.reduce(%v3309 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3313 = stablehlo.divide %v3312, %v3311 : tensor<1024xf32>
    %v3314 = stablehlo.multiply %v3306, %v3309 : tensor<64x1024x14x14xf32>
    %v3315 = stablehlo.reduce(%v3314 init: %v3310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3316 = stablehlo.divide %v3315, %v3311 : tensor<1024xf32>
    %v3317 = stablehlo.concatenate %v3313, %v3316, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3318 = stablehlo.concatenate %v1929, %v3317, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b5g3dst = "stablehlo.all_reduce"(%v3318) ({
    ^bb0(%aras3b5g3dst: tensor<f32>, %arbs3b5g3dst: tensor<f32>):
      %aradds3b5g3dst = stablehlo.add %aras3b5g3dst, %arbs3b5g3dst : tensor<f32>
      stablehlo.return %aradds3b5g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b5g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b5g3dst = stablehlo.divide %arsums3b5g3dst, %arns3b5g3dst : tensor<4096xf32>
    %v3319 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3320 = stablehlo.slice %armeans3b5g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3321 = stablehlo.slice %armeans3b5g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3322 = stablehlo.slice %armeans3b5g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3323 = stablehlo.slice %armeans3b5g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3324 = stablehlo.broadcast_in_dim %v3320, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3325 = stablehlo.broadcast_in_dim %v3321, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3326 = stablehlo.broadcast_in_dim %v3322, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3327 = stablehlo.broadcast_in_dim %v3323, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3328 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3329 = stablehlo.add %v3325, %v3328 : tensor<64x1024x14x14xf32>
    %v3330 = stablehlo.rsqrt %v3329 : tensor<64x1024x14x14xf32>
    %v3331 = stablehlo.subtract %v3319, %v3324 : tensor<64x1024x14x14xf32>
    %v3332 = stablehlo.multiply %v3331, %v3330 : tensor<64x1024x14x14xf32>
    %v3333 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3334 = stablehlo.reshape %v3296 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3335 = stablehlo.multiply %v3333, %v3334 : tensor<64x1024x14x14xf32>
    %v3336 = stablehlo.subtract %v3335, %v3326 : tensor<64x1024x14x14xf32>
    %v3337 = stablehlo.multiply %v3332, %v3327 : tensor<64x1024x14x14xf32>
    %v3338 = stablehlo.subtract %v3336, %v3337 : tensor<64x1024x14x14xf32>
    %v3339 = stablehlo.multiply %v3330, %v3338 : tensor<64x1024x14x14xf32>
    %v3340 = stablehlo.reshape %v3339 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3341 = stablehlo.reshape %v3340 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3342 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3343 = stablehlo.transpose %v3342, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3344 = stablehlo.convert %v3341 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3345 = stablehlo.convert %v3343 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3346 = stablehlo.convolution(%v3344, %v3345)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3347 = stablehlo.convert %v3346 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3348 = stablehlo.reshape %v3347 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3349 = stablehlo.reshape %v3348 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3350 = stablehlo.reshape %v1900 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3351 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3352 = stablehlo.compare GT, %v3350, %v3351 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3353 = stablehlo.select %v3352, %v3349, %v3351 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3354 = stablehlo.reshape %v3353 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3355 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3356 = stablehlo.slice %v1885 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3357 = stablehlo.slice %v1885 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3358 = stablehlo.broadcast_in_dim %v3356, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3359 = stablehlo.broadcast_in_dim %v3357, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3360 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3361 = stablehlo.add %v3359, %v3360 : tensor<64x256x14x14xf32>
    %v3362 = stablehlo.rsqrt %v3361 : tensor<64x256x14x14xf32>
    %v3363 = stablehlo.subtract %v3355, %v3358 : tensor<64x256x14x14xf32>
    %v3364 = stablehlo.multiply %v3363, %v3362 : tensor<64x256x14x14xf32>
    %v3365 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3366 = stablehlo.reshape %v3354 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3367 = stablehlo.multiply %v3365, %v3366 : tensor<64x256x14x14xf32>
    %v3368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3369 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3370 = stablehlo.reduce(%v3367 init: %v3368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3371 = stablehlo.divide %v3370, %v3369 : tensor<256xf32>
    %v3372 = stablehlo.multiply %v3364, %v3367 : tensor<64x256x14x14xf32>
    %v3373 = stablehlo.reduce(%v3372 init: %v3368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3374 = stablehlo.divide %v3373, %v3369 : tensor<256xf32>
    %v3375 = stablehlo.concatenate %v3371, %v3374, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3376 = stablehlo.concatenate %v1885, %v3375, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b5g2dst = "stablehlo.all_reduce"(%v3376) ({
    ^bb0(%aras3b5g2dst: tensor<f32>, %arbs3b5g2dst: tensor<f32>):
      %aradds3b5g2dst = stablehlo.add %aras3b5g2dst, %arbs3b5g2dst : tensor<f32>
      stablehlo.return %aradds3b5g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g2dst = stablehlo.divide %arsums3b5g2dst, %arns3b5g2dst : tensor<1024xf32>
    %v3377 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3378 = stablehlo.slice %armeans3b5g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3379 = stablehlo.slice %armeans3b5g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3380 = stablehlo.slice %armeans3b5g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3381 = stablehlo.slice %armeans3b5g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3382 = stablehlo.broadcast_in_dim %v3378, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3383 = stablehlo.broadcast_in_dim %v3379, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3384 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3385 = stablehlo.broadcast_in_dim %v3381, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3386 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3387 = stablehlo.add %v3383, %v3386 : tensor<64x256x14x14xf32>
    %v3388 = stablehlo.rsqrt %v3387 : tensor<64x256x14x14xf32>
    %v3389 = stablehlo.subtract %v3377, %v3382 : tensor<64x256x14x14xf32>
    %v3390 = stablehlo.multiply %v3389, %v3388 : tensor<64x256x14x14xf32>
    %v3391 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3392 = stablehlo.reshape %v3354 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3393 = stablehlo.multiply %v3391, %v3392 : tensor<64x256x14x14xf32>
    %v3394 = stablehlo.subtract %v3393, %v3384 : tensor<64x256x14x14xf32>
    %v3395 = stablehlo.multiply %v3390, %v3385 : tensor<64x256x14x14xf32>
    %v3396 = stablehlo.subtract %v3394, %v3395 : tensor<64x256x14x14xf32>
    %v3397 = stablehlo.multiply %v3388, %v3396 : tensor<64x256x14x14xf32>
    %v3398 = stablehlo.reshape %v3397 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3399 = stablehlo.reshape %v3398 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3400 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3401 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3402 = stablehlo.convert %v3399 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3403 = stablehlo.convert %v3401 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3404 = stablehlo.convolution(%v3402, %v3403)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3405 = stablehlo.convert %v3404 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3406 = stablehlo.reshape %v3405 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3407 = stablehlo.reshape %v3406 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3408 = stablehlo.reshape %v1856 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3409 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3410 = stablehlo.compare GT, %v3408, %v3409 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3411 = stablehlo.select %v3410, %v3407, %v3409 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3412 = stablehlo.reshape %v3411 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3413 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3414 = stablehlo.slice %v1841 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3415 = stablehlo.slice %v1841 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3416 = stablehlo.broadcast_in_dim %v3414, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3417 = stablehlo.broadcast_in_dim %v3415, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3418 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3419 = stablehlo.add %v3417, %v3418 : tensor<64x256x14x14xf32>
    %v3420 = stablehlo.rsqrt %v3419 : tensor<64x256x14x14xf32>
    %v3421 = stablehlo.subtract %v3413, %v3416 : tensor<64x256x14x14xf32>
    %v3422 = stablehlo.multiply %v3421, %v3420 : tensor<64x256x14x14xf32>
    %v3423 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3424 = stablehlo.reshape %v3412 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3425 = stablehlo.multiply %v3423, %v3424 : tensor<64x256x14x14xf32>
    %v3426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3427 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3428 = stablehlo.reduce(%v3425 init: %v3426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3429 = stablehlo.divide %v3428, %v3427 : tensor<256xf32>
    %v3430 = stablehlo.multiply %v3422, %v3425 : tensor<64x256x14x14xf32>
    %v3431 = stablehlo.reduce(%v3430 init: %v3426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3432 = stablehlo.divide %v3431, %v3427 : tensor<256xf32>
    %v3433 = stablehlo.concatenate %v3429, %v3432, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3434 = stablehlo.concatenate %v1841, %v3433, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b5g1dst = "stablehlo.all_reduce"(%v3434) ({
    ^bb0(%aras3b5g1dst: tensor<f32>, %arbs3b5g1dst: tensor<f32>):
      %aradds3b5g1dst = stablehlo.add %aras3b5g1dst, %arbs3b5g1dst : tensor<f32>
      stablehlo.return %aradds3b5g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g1dst = stablehlo.divide %arsums3b5g1dst, %arns3b5g1dst : tensor<1024xf32>
    %v3435 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3436 = stablehlo.slice %armeans3b5g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3437 = stablehlo.slice %armeans3b5g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3438 = stablehlo.slice %armeans3b5g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3439 = stablehlo.slice %armeans3b5g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3440 = stablehlo.broadcast_in_dim %v3436, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3441 = stablehlo.broadcast_in_dim %v3437, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3442 = stablehlo.broadcast_in_dim %v3438, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3443 = stablehlo.broadcast_in_dim %v3439, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3444 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3445 = stablehlo.add %v3441, %v3444 : tensor<64x256x14x14xf32>
    %v3446 = stablehlo.rsqrt %v3445 : tensor<64x256x14x14xf32>
    %v3447 = stablehlo.subtract %v3435, %v3440 : tensor<64x256x14x14xf32>
    %v3448 = stablehlo.multiply %v3447, %v3446 : tensor<64x256x14x14xf32>
    %v3449 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3450 = stablehlo.reshape %v3412 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3451 = stablehlo.multiply %v3449, %v3450 : tensor<64x256x14x14xf32>
    %v3452 = stablehlo.subtract %v3451, %v3442 : tensor<64x256x14x14xf32>
    %v3453 = stablehlo.multiply %v3448, %v3443 : tensor<64x256x14x14xf32>
    %v3454 = stablehlo.subtract %v3452, %v3453 : tensor<64x256x14x14xf32>
    %v3455 = stablehlo.multiply %v3446, %v3454 : tensor<64x256x14x14xf32>
    %v3456 = stablehlo.reshape %v3455 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3457 = stablehlo.reshape %v3456 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3458 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3459 = stablehlo.transpose %v3458, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3460 = stablehlo.convert %v3457 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3461 = stablehlo.convert %v3459 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3462 = stablehlo.convolution(%v3460, %v3461)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3463 = stablehlo.convert %v3462 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3464 = stablehlo.reshape %v3463 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3466 = stablehlo.reshape %v3296 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3467 = stablehlo.add %v3465, %v3466 : tensor<64x1024x14x14xf32>
    %v3468 = stablehlo.reshape %v3467 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3469 = stablehlo.reshape %v1814 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3470 = stablehlo.reshape %v3456 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3471 = stablehlo.transpose %v3469, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3472 = stablehlo.transpose %v3470, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3473 = stablehlo.convert %v3471 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3474 = stablehlo.convert %v3472 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3475 = stablehlo.convolution(%v3473, %v3474)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3476 = stablehlo.convert %v3475 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3477 = stablehlo.transpose %v3476, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3478 = stablehlo.reshape %v1822 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3479 = stablehlo.slice %v1841 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3480 = stablehlo.slice %v1841 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3481 = stablehlo.broadcast_in_dim %v3479, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3482 = stablehlo.broadcast_in_dim %v3480, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3483 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3484 = stablehlo.add %v3482, %v3483 : tensor<64x256x14x14xf32>
    %v3485 = stablehlo.rsqrt %v3484 : tensor<64x256x14x14xf32>
    %v3486 = stablehlo.subtract %v3478, %v3481 : tensor<64x256x14x14xf32>
    %v3487 = stablehlo.multiply %v3486, %v3485 : tensor<64x256x14x14xf32>
    %v3488 = stablehlo.reshape %v3412 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3489 = stablehlo.multiply %v3488, %v3487 : tensor<64x256x14x14xf32>
    %v3490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3491 = stablehlo.reduce(%v3489 init: %v3490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3492 = stablehlo.reshape %v3412 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3494 = stablehlo.reduce(%v3492 init: %v3493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3495 = stablehlo.reshape %v1858 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3496 = stablehlo.reshape %v3398 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3497 = stablehlo.transpose %v3495, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3498 = stablehlo.transpose %v3496, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3499 = stablehlo.convert %v3497 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3500 = stablehlo.convert %v3498 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3501 = stablehlo.convolution(%v3499, %v3500)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3502 = stablehlo.convert %v3501 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3503 = stablehlo.transpose %v3502, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3504 = stablehlo.reshape %v1866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3505 = stablehlo.slice %v1885 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3506 = stablehlo.slice %v1885 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3507 = stablehlo.broadcast_in_dim %v3505, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3506, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3509 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3510 = stablehlo.add %v3508, %v3509 : tensor<64x256x14x14xf32>
    %v3511 = stablehlo.rsqrt %v3510 : tensor<64x256x14x14xf32>
    %v3512 = stablehlo.subtract %v3504, %v3507 : tensor<64x256x14x14xf32>
    %v3513 = stablehlo.multiply %v3512, %v3511 : tensor<64x256x14x14xf32>
    %v3514 = stablehlo.reshape %v3354 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3515 = stablehlo.multiply %v3514, %v3513 : tensor<64x256x14x14xf32>
    %v3516 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3517 = stablehlo.reduce(%v3515 init: %v3516) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3518 = stablehlo.reshape %v3354 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3520 = stablehlo.reduce(%v3518 init: %v3519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3521 = stablehlo.reshape %v1902 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3522 = stablehlo.reshape %v3340 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3523 = stablehlo.transpose %v3521, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3524 = stablehlo.transpose %v3522, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3525 = stablehlo.convert %v3523 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3526 = stablehlo.convert %v3524 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3527 = stablehlo.convolution(%v3525, %v3526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3528 = stablehlo.convert %v3527 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3529 = stablehlo.transpose %v3528, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3530 = stablehlo.reshape %v1910 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3531 = stablehlo.slice %v1929 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3532 = stablehlo.slice %v1929 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3533 = stablehlo.broadcast_in_dim %v3531, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3534 = stablehlo.broadcast_in_dim %v3532, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3535 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3536 = stablehlo.add %v3534, %v3535 : tensor<64x1024x14x14xf32>
    %v3537 = stablehlo.rsqrt %v3536 : tensor<64x1024x14x14xf32>
    %v3538 = stablehlo.subtract %v3530, %v3533 : tensor<64x1024x14x14xf32>
    %v3539 = stablehlo.multiply %v3538, %v3537 : tensor<64x1024x14x14xf32>
    %v3540 = stablehlo.reshape %v3296 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3541 = stablehlo.multiply %v3540, %v3539 : tensor<64x1024x14x14xf32>
    %v3542 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3543 = stablehlo.reduce(%v3541 init: %v3542) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3544 = stablehlo.reshape %v3296 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3546 = stablehlo.reduce(%v3544 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3547 = stablehlo.reshape %v3468 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3548 = stablehlo.reshape %v1810 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3549 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3550 = stablehlo.compare GT, %v3548, %v3549 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3551 = stablehlo.select %v3550, %v3547, %v3549 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3552 = stablehlo.reshape %v3551 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3553 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3554 = stablehlo.slice %v1791 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3555 = stablehlo.slice %v1791 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3556 = stablehlo.broadcast_in_dim %v3554, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3557 = stablehlo.broadcast_in_dim %v3555, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3558 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3559 = stablehlo.add %v3557, %v3558 : tensor<64x1024x14x14xf32>
    %v3560 = stablehlo.rsqrt %v3559 : tensor<64x1024x14x14xf32>
    %v3561 = stablehlo.subtract %v3553, %v3556 : tensor<64x1024x14x14xf32>
    %v3562 = stablehlo.multiply %v3561, %v3560 : tensor<64x1024x14x14xf32>
    %v3563 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3564 = stablehlo.reshape %v3552 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3565 = stablehlo.multiply %v3563, %v3564 : tensor<64x1024x14x14xf32>
    %v3566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3567 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3568 = stablehlo.reduce(%v3565 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3569 = stablehlo.divide %v3568, %v3567 : tensor<1024xf32>
    %v3570 = stablehlo.multiply %v3562, %v3565 : tensor<64x1024x14x14xf32>
    %v3571 = stablehlo.reduce(%v3570 init: %v3566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3572 = stablehlo.divide %v3571, %v3567 : tensor<1024xf32>
    %v3573 = stablehlo.concatenate %v3569, %v3572, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3574 = stablehlo.concatenate %v1791, %v3573, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b4g3dst = "stablehlo.all_reduce"(%v3574) ({
    ^bb0(%aras3b4g3dst: tensor<f32>, %arbs3b4g3dst: tensor<f32>):
      %aradds3b4g3dst = stablehlo.add %aras3b4g3dst, %arbs3b4g3dst : tensor<f32>
      stablehlo.return %aradds3b4g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b4g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b4g3dst = stablehlo.divide %arsums3b4g3dst, %arns3b4g3dst : tensor<4096xf32>
    %v3575 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3576 = stablehlo.slice %armeans3b4g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3577 = stablehlo.slice %armeans3b4g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3578 = stablehlo.slice %armeans3b4g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3579 = stablehlo.slice %armeans3b4g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3580 = stablehlo.broadcast_in_dim %v3576, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3581 = stablehlo.broadcast_in_dim %v3577, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3582 = stablehlo.broadcast_in_dim %v3578, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3583 = stablehlo.broadcast_in_dim %v3579, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3584 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3585 = stablehlo.add %v3581, %v3584 : tensor<64x1024x14x14xf32>
    %v3586 = stablehlo.rsqrt %v3585 : tensor<64x1024x14x14xf32>
    %v3587 = stablehlo.subtract %v3575, %v3580 : tensor<64x1024x14x14xf32>
    %v3588 = stablehlo.multiply %v3587, %v3586 : tensor<64x1024x14x14xf32>
    %v3589 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3590 = stablehlo.reshape %v3552 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3591 = stablehlo.multiply %v3589, %v3590 : tensor<64x1024x14x14xf32>
    %v3592 = stablehlo.subtract %v3591, %v3582 : tensor<64x1024x14x14xf32>
    %v3593 = stablehlo.multiply %v3588, %v3583 : tensor<64x1024x14x14xf32>
    %v3594 = stablehlo.subtract %v3592, %v3593 : tensor<64x1024x14x14xf32>
    %v3595 = stablehlo.multiply %v3586, %v3594 : tensor<64x1024x14x14xf32>
    %v3596 = stablehlo.reshape %v3595 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3597 = stablehlo.reshape %v3596 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3598 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3599 = stablehlo.transpose %v3598, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3600 = stablehlo.convert %v3597 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3601 = stablehlo.convert %v3599 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3602 = stablehlo.convolution(%v3600, %v3601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3603 = stablehlo.convert %v3602 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3604 = stablehlo.reshape %v3603 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3605 = stablehlo.reshape %v3604 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3606 = stablehlo.reshape %v1762 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3607 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3608 = stablehlo.compare GT, %v3606, %v3607 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3609 = stablehlo.select %v3608, %v3605, %v3607 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3610 = stablehlo.reshape %v3609 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3611 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3612 = stablehlo.slice %v1747 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3613 = stablehlo.slice %v1747 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3614 = stablehlo.broadcast_in_dim %v3612, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3615 = stablehlo.broadcast_in_dim %v3613, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3616 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3617 = stablehlo.add %v3615, %v3616 : tensor<64x256x14x14xf32>
    %v3618 = stablehlo.rsqrt %v3617 : tensor<64x256x14x14xf32>
    %v3619 = stablehlo.subtract %v3611, %v3614 : tensor<64x256x14x14xf32>
    %v3620 = stablehlo.multiply %v3619, %v3618 : tensor<64x256x14x14xf32>
    %v3621 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3622 = stablehlo.reshape %v3610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3623 = stablehlo.multiply %v3621, %v3622 : tensor<64x256x14x14xf32>
    %v3624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3625 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3626 = stablehlo.reduce(%v3623 init: %v3624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3627 = stablehlo.divide %v3626, %v3625 : tensor<256xf32>
    %v3628 = stablehlo.multiply %v3620, %v3623 : tensor<64x256x14x14xf32>
    %v3629 = stablehlo.reduce(%v3628 init: %v3624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3630 = stablehlo.divide %v3629, %v3625 : tensor<256xf32>
    %v3631 = stablehlo.concatenate %v3627, %v3630, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3632 = stablehlo.concatenate %v1747, %v3631, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g2dst = "stablehlo.all_reduce"(%v3632) ({
    ^bb0(%aras3b4g2dst: tensor<f32>, %arbs3b4g2dst: tensor<f32>):
      %aradds3b4g2dst = stablehlo.add %aras3b4g2dst, %arbs3b4g2dst : tensor<f32>
      stablehlo.return %aradds3b4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g2dst = stablehlo.divide %arsums3b4g2dst, %arns3b4g2dst : tensor<1024xf32>
    %v3633 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3634 = stablehlo.slice %armeans3b4g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3635 = stablehlo.slice %armeans3b4g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3636 = stablehlo.slice %armeans3b4g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3637 = stablehlo.slice %armeans3b4g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3638 = stablehlo.broadcast_in_dim %v3634, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3639 = stablehlo.broadcast_in_dim %v3635, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3640 = stablehlo.broadcast_in_dim %v3636, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3641 = stablehlo.broadcast_in_dim %v3637, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3642 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3643 = stablehlo.add %v3639, %v3642 : tensor<64x256x14x14xf32>
    %v3644 = stablehlo.rsqrt %v3643 : tensor<64x256x14x14xf32>
    %v3645 = stablehlo.subtract %v3633, %v3638 : tensor<64x256x14x14xf32>
    %v3646 = stablehlo.multiply %v3645, %v3644 : tensor<64x256x14x14xf32>
    %v3647 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3648 = stablehlo.reshape %v3610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3649 = stablehlo.multiply %v3647, %v3648 : tensor<64x256x14x14xf32>
    %v3650 = stablehlo.subtract %v3649, %v3640 : tensor<64x256x14x14xf32>
    %v3651 = stablehlo.multiply %v3646, %v3641 : tensor<64x256x14x14xf32>
    %v3652 = stablehlo.subtract %v3650, %v3651 : tensor<64x256x14x14xf32>
    %v3653 = stablehlo.multiply %v3644, %v3652 : tensor<64x256x14x14xf32>
    %v3654 = stablehlo.reshape %v3653 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3655 = stablehlo.reshape %v3654 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3656 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3657 = stablehlo.transpose %v3656, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3658 = stablehlo.convert %v3655 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3659 = stablehlo.convert %v3657 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3660 = stablehlo.convolution(%v3658, %v3659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3661 = stablehlo.convert %v3660 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3662 = stablehlo.reshape %v3661 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3663 = stablehlo.reshape %v3662 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3664 = stablehlo.reshape %v1718 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3666 = stablehlo.compare GT, %v3664, %v3665 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3667 = stablehlo.select %v3666, %v3663, %v3665 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3668 = stablehlo.reshape %v3667 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3669 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3670 = stablehlo.slice %v1703 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3671 = stablehlo.slice %v1703 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3672 = stablehlo.broadcast_in_dim %v3670, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3673 = stablehlo.broadcast_in_dim %v3671, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3674 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3675 = stablehlo.add %v3673, %v3674 : tensor<64x256x14x14xf32>
    %v3676 = stablehlo.rsqrt %v3675 : tensor<64x256x14x14xf32>
    %v3677 = stablehlo.subtract %v3669, %v3672 : tensor<64x256x14x14xf32>
    %v3678 = stablehlo.multiply %v3677, %v3676 : tensor<64x256x14x14xf32>
    %v3679 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3680 = stablehlo.reshape %v3668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3681 = stablehlo.multiply %v3679, %v3680 : tensor<64x256x14x14xf32>
    %v3682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3683 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3684 = stablehlo.reduce(%v3681 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3685 = stablehlo.divide %v3684, %v3683 : tensor<256xf32>
    %v3686 = stablehlo.multiply %v3678, %v3681 : tensor<64x256x14x14xf32>
    %v3687 = stablehlo.reduce(%v3686 init: %v3682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3688 = stablehlo.divide %v3687, %v3683 : tensor<256xf32>
    %v3689 = stablehlo.concatenate %v3685, %v3688, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3690 = stablehlo.concatenate %v1703, %v3689, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g1dst = "stablehlo.all_reduce"(%v3690) ({
    ^bb0(%aras3b4g1dst: tensor<f32>, %arbs3b4g1dst: tensor<f32>):
      %aradds3b4g1dst = stablehlo.add %aras3b4g1dst, %arbs3b4g1dst : tensor<f32>
      stablehlo.return %aradds3b4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g1dst = stablehlo.divide %arsums3b4g1dst, %arns3b4g1dst : tensor<1024xf32>
    %v3691 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3692 = stablehlo.slice %armeans3b4g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3693 = stablehlo.slice %armeans3b4g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3694 = stablehlo.slice %armeans3b4g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3695 = stablehlo.slice %armeans3b4g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3696 = stablehlo.broadcast_in_dim %v3692, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3697 = stablehlo.broadcast_in_dim %v3693, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3698 = stablehlo.broadcast_in_dim %v3694, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3699 = stablehlo.broadcast_in_dim %v3695, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3700 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3701 = stablehlo.add %v3697, %v3700 : tensor<64x256x14x14xf32>
    %v3702 = stablehlo.rsqrt %v3701 : tensor<64x256x14x14xf32>
    %v3703 = stablehlo.subtract %v3691, %v3696 : tensor<64x256x14x14xf32>
    %v3704 = stablehlo.multiply %v3703, %v3702 : tensor<64x256x14x14xf32>
    %v3705 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3706 = stablehlo.reshape %v3668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3707 = stablehlo.multiply %v3705, %v3706 : tensor<64x256x14x14xf32>
    %v3708 = stablehlo.subtract %v3707, %v3698 : tensor<64x256x14x14xf32>
    %v3709 = stablehlo.multiply %v3704, %v3699 : tensor<64x256x14x14xf32>
    %v3710 = stablehlo.subtract %v3708, %v3709 : tensor<64x256x14x14xf32>
    %v3711 = stablehlo.multiply %v3702, %v3710 : tensor<64x256x14x14xf32>
    %v3712 = stablehlo.reshape %v3711 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3713 = stablehlo.reshape %v3712 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3714 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3715 = stablehlo.transpose %v3714, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3716 = stablehlo.convert %v3713 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3717 = stablehlo.convert %v3715 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3718 = stablehlo.convolution(%v3716, %v3717)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3719 = stablehlo.convert %v3718 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3720 = stablehlo.reshape %v3719 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3721 = stablehlo.reshape %v3720 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3722 = stablehlo.reshape %v3552 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<64x1024x14x14xf32>
    %v3724 = stablehlo.reshape %v3723 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3725 = stablehlo.reshape %v1676 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3726 = stablehlo.reshape %v3712 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3727 = stablehlo.transpose %v3725, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3728 = stablehlo.transpose %v3726, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3729 = stablehlo.convert %v3727 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3730 = stablehlo.convert %v3728 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3731 = stablehlo.convolution(%v3729, %v3730)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3732 = stablehlo.convert %v3731 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3733 = stablehlo.transpose %v3732, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3734 = stablehlo.reshape %v1684 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3735 = stablehlo.slice %v1703 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3736 = stablehlo.slice %v1703 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3737 = stablehlo.broadcast_in_dim %v3735, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3736, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3739 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3740 = stablehlo.add %v3738, %v3739 : tensor<64x256x14x14xf32>
    %v3741 = stablehlo.rsqrt %v3740 : tensor<64x256x14x14xf32>
    %v3742 = stablehlo.subtract %v3734, %v3737 : tensor<64x256x14x14xf32>
    %v3743 = stablehlo.multiply %v3742, %v3741 : tensor<64x256x14x14xf32>
    %v3744 = stablehlo.reshape %v3668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3745 = stablehlo.multiply %v3744, %v3743 : tensor<64x256x14x14xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3748 = stablehlo.reshape %v3668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3750 = stablehlo.reduce(%v3748 init: %v3749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3751 = stablehlo.reshape %v1720 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3752 = stablehlo.reshape %v3654 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3753 = stablehlo.transpose %v3751, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3754 = stablehlo.transpose %v3752, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3755 = stablehlo.convert %v3753 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3756 = stablehlo.convert %v3754 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3757 = stablehlo.convolution(%v3755, %v3756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3758 = stablehlo.convert %v3757 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3759 = stablehlo.transpose %v3758, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3760 = stablehlo.reshape %v1728 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3761 = stablehlo.slice %v1747 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3762 = stablehlo.slice %v1747 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3763 = stablehlo.broadcast_in_dim %v3761, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3764 = stablehlo.broadcast_in_dim %v3762, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3765 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3766 = stablehlo.add %v3764, %v3765 : tensor<64x256x14x14xf32>
    %v3767 = stablehlo.rsqrt %v3766 : tensor<64x256x14x14xf32>
    %v3768 = stablehlo.subtract %v3760, %v3763 : tensor<64x256x14x14xf32>
    %v3769 = stablehlo.multiply %v3768, %v3767 : tensor<64x256x14x14xf32>
    %v3770 = stablehlo.reshape %v3610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3771 = stablehlo.multiply %v3770, %v3769 : tensor<64x256x14x14xf32>
    %v3772 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3773 = stablehlo.reduce(%v3771 init: %v3772) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3774 = stablehlo.reshape %v3610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3775 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3776 = stablehlo.reduce(%v3774 init: %v3775) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3777 = stablehlo.reshape %v1764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3778 = stablehlo.reshape %v3596 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3779 = stablehlo.transpose %v3777, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3780 = stablehlo.transpose %v3778, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3781 = stablehlo.convert %v3779 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3782 = stablehlo.convert %v3780 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3783 = stablehlo.convolution(%v3781, %v3782)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3784 = stablehlo.convert %v3783 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3785 = stablehlo.transpose %v3784, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3786 = stablehlo.reshape %v1772 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3787 = stablehlo.slice %v1791 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3788 = stablehlo.slice %v1791 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3789 = stablehlo.broadcast_in_dim %v3787, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3790 = stablehlo.broadcast_in_dim %v3788, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3791 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3792 = stablehlo.add %v3790, %v3791 : tensor<64x1024x14x14xf32>
    %v3793 = stablehlo.rsqrt %v3792 : tensor<64x1024x14x14xf32>
    %v3794 = stablehlo.subtract %v3786, %v3789 : tensor<64x1024x14x14xf32>
    %v3795 = stablehlo.multiply %v3794, %v3793 : tensor<64x1024x14x14xf32>
    %v3796 = stablehlo.reshape %v3552 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3797 = stablehlo.multiply %v3796, %v3795 : tensor<64x1024x14x14xf32>
    %v3798 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3799 = stablehlo.reduce(%v3797 init: %v3798) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3800 = stablehlo.reshape %v3552 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3802 = stablehlo.reduce(%v3800 init: %v3801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3803 = stablehlo.reshape %v3724 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3804 = stablehlo.reshape %v1672 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3805 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v3806 = stablehlo.compare GT, %v3804, %v3805 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v3807 = stablehlo.select %v3806, %v3803, %v3805 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v3808 = stablehlo.reshape %v3807 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3809 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3810 = stablehlo.slice %v1653 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3811 = stablehlo.slice %v1653 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v3812 = stablehlo.broadcast_in_dim %v3810, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3813 = stablehlo.broadcast_in_dim %v3811, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3814 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3815 = stablehlo.add %v3813, %v3814 : tensor<64x1024x14x14xf32>
    %v3816 = stablehlo.rsqrt %v3815 : tensor<64x1024x14x14xf32>
    %v3817 = stablehlo.subtract %v3809, %v3812 : tensor<64x1024x14x14xf32>
    %v3818 = stablehlo.multiply %v3817, %v3816 : tensor<64x1024x14x14xf32>
    %v3819 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3820 = stablehlo.reshape %v3808 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3821 = stablehlo.multiply %v3819, %v3820 : tensor<64x1024x14x14xf32>
    %v3822 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3823 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v3824 = stablehlo.reduce(%v3821 init: %v3822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3825 = stablehlo.divide %v3824, %v3823 : tensor<1024xf32>
    %v3826 = stablehlo.multiply %v3818, %v3821 : tensor<64x1024x14x14xf32>
    %v3827 = stablehlo.reduce(%v3826 init: %v3822) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3828 = stablehlo.divide %v3827, %v3823 : tensor<1024xf32>
    %v3829 = stablehlo.concatenate %v3825, %v3828, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v3830 = stablehlo.concatenate %v1653, %v3829, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b3g3dst = "stablehlo.all_reduce"(%v3830) ({
    ^bb0(%aras3b3g3dst: tensor<f32>, %arbs3b3g3dst: tensor<f32>):
      %aradds3b3g3dst = stablehlo.add %aras3b3g3dst, %arbs3b3g3dst : tensor<f32>
      stablehlo.return %aradds3b3g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b3g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b3g3dst = stablehlo.divide %arsums3b3g3dst, %arns3b3g3dst : tensor<4096xf32>
    %v3831 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3832 = stablehlo.slice %armeans3b3g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3833 = stablehlo.slice %armeans3b3g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3834 = stablehlo.slice %armeans3b3g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3835 = stablehlo.slice %armeans3b3g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v3836 = stablehlo.broadcast_in_dim %v3832, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3837 = stablehlo.broadcast_in_dim %v3833, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3838 = stablehlo.broadcast_in_dim %v3834, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3839 = stablehlo.broadcast_in_dim %v3835, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3840 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3841 = stablehlo.add %v3837, %v3840 : tensor<64x1024x14x14xf32>
    %v3842 = stablehlo.rsqrt %v3841 : tensor<64x1024x14x14xf32>
    %v3843 = stablehlo.subtract %v3831, %v3836 : tensor<64x1024x14x14xf32>
    %v3844 = stablehlo.multiply %v3843, %v3842 : tensor<64x1024x14x14xf32>
    %v3845 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3846 = stablehlo.reshape %v3808 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3847 = stablehlo.multiply %v3845, %v3846 : tensor<64x1024x14x14xf32>
    %v3848 = stablehlo.subtract %v3847, %v3838 : tensor<64x1024x14x14xf32>
    %v3849 = stablehlo.multiply %v3844, %v3839 : tensor<64x1024x14x14xf32>
    %v3850 = stablehlo.subtract %v3848, %v3849 : tensor<64x1024x14x14xf32>
    %v3851 = stablehlo.multiply %v3842, %v3850 : tensor<64x1024x14x14xf32>
    %v3852 = stablehlo.reshape %v3851 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3853 = stablehlo.reshape %v3852 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3854 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3855 = stablehlo.transpose %v3854, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3856 = stablehlo.convert %v3853 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3857 = stablehlo.convert %v3855 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3858 = stablehlo.convolution(%v3856, %v3857)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3859 = stablehlo.convert %v3858 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3860 = stablehlo.reshape %v3859 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3861 = stablehlo.reshape %v3860 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3862 = stablehlo.reshape %v1624 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3863 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3864 = stablehlo.compare GT, %v3862, %v3863 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3865 = stablehlo.select %v3864, %v3861, %v3863 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3866 = stablehlo.reshape %v3865 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3867 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3868 = stablehlo.slice %v1609 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3869 = stablehlo.slice %v1609 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3870 = stablehlo.broadcast_in_dim %v3868, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3871 = stablehlo.broadcast_in_dim %v3869, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3872 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3873 = stablehlo.add %v3871, %v3872 : tensor<64x256x14x14xf32>
    %v3874 = stablehlo.rsqrt %v3873 : tensor<64x256x14x14xf32>
    %v3875 = stablehlo.subtract %v3867, %v3870 : tensor<64x256x14x14xf32>
    %v3876 = stablehlo.multiply %v3875, %v3874 : tensor<64x256x14x14xf32>
    %v3877 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3878 = stablehlo.reshape %v3866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3879 = stablehlo.multiply %v3877, %v3878 : tensor<64x256x14x14xf32>
    %v3880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3881 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3882 = stablehlo.reduce(%v3879 init: %v3880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3883 = stablehlo.divide %v3882, %v3881 : tensor<256xf32>
    %v3884 = stablehlo.multiply %v3876, %v3879 : tensor<64x256x14x14xf32>
    %v3885 = stablehlo.reduce(%v3884 init: %v3880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3886 = stablehlo.divide %v3885, %v3881 : tensor<256xf32>
    %v3887 = stablehlo.concatenate %v3883, %v3886, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3888 = stablehlo.concatenate %v1609, %v3887, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g2dst = "stablehlo.all_reduce"(%v3888) ({
    ^bb0(%aras3b3g2dst: tensor<f32>, %arbs3b3g2dst: tensor<f32>):
      %aradds3b3g2dst = stablehlo.add %aras3b3g2dst, %arbs3b3g2dst : tensor<f32>
      stablehlo.return %aradds3b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g2dst = stablehlo.divide %arsums3b3g2dst, %arns3b3g2dst : tensor<1024xf32>
    %v3889 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3890 = stablehlo.slice %armeans3b3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3891 = stablehlo.slice %armeans3b3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3892 = stablehlo.slice %armeans3b3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3893 = stablehlo.slice %armeans3b3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3894 = stablehlo.broadcast_in_dim %v3890, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3895 = stablehlo.broadcast_in_dim %v3891, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3896 = stablehlo.broadcast_in_dim %v3892, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3897 = stablehlo.broadcast_in_dim %v3893, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3898 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3899 = stablehlo.add %v3895, %v3898 : tensor<64x256x14x14xf32>
    %v3900 = stablehlo.rsqrt %v3899 : tensor<64x256x14x14xf32>
    %v3901 = stablehlo.subtract %v3889, %v3894 : tensor<64x256x14x14xf32>
    %v3902 = stablehlo.multiply %v3901, %v3900 : tensor<64x256x14x14xf32>
    %v3903 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3904 = stablehlo.reshape %v3866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3905 = stablehlo.multiply %v3903, %v3904 : tensor<64x256x14x14xf32>
    %v3906 = stablehlo.subtract %v3905, %v3896 : tensor<64x256x14x14xf32>
    %v3907 = stablehlo.multiply %v3902, %v3897 : tensor<64x256x14x14xf32>
    %v3908 = stablehlo.subtract %v3906, %v3907 : tensor<64x256x14x14xf32>
    %v3909 = stablehlo.multiply %v3900, %v3908 : tensor<64x256x14x14xf32>
    %v3910 = stablehlo.reshape %v3909 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3911 = stablehlo.reshape %v3910 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3912 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3913 = stablehlo.transpose %v3912, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3914 = stablehlo.convert %v3911 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3915 = stablehlo.convert %v3913 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3916 = stablehlo.convolution(%v3914, %v3915)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3917 = stablehlo.convert %v3916 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3918 = stablehlo.reshape %v3917 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3919 = stablehlo.reshape %v3918 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3920 = stablehlo.reshape %v1580 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3921 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3922 = stablehlo.compare GT, %v3920, %v3921 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3923 = stablehlo.select %v3922, %v3919, %v3921 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3924 = stablehlo.reshape %v3923 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3925 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3926 = stablehlo.slice %v1565 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3927 = stablehlo.slice %v1565 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3928 = stablehlo.broadcast_in_dim %v3926, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3929 = stablehlo.broadcast_in_dim %v3927, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3930 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3931 = stablehlo.add %v3929, %v3930 : tensor<64x256x14x14xf32>
    %v3932 = stablehlo.rsqrt %v3931 : tensor<64x256x14x14xf32>
    %v3933 = stablehlo.subtract %v3925, %v3928 : tensor<64x256x14x14xf32>
    %v3934 = stablehlo.multiply %v3933, %v3932 : tensor<64x256x14x14xf32>
    %v3935 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3936 = stablehlo.reshape %v3924 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3937 = stablehlo.multiply %v3935, %v3936 : tensor<64x256x14x14xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3941 = stablehlo.divide %v3940, %v3939 : tensor<256xf32>
    %v3942 = stablehlo.multiply %v3934, %v3937 : tensor<64x256x14x14xf32>
    %v3943 = stablehlo.reduce(%v3942 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3944 = stablehlo.divide %v3943, %v3939 : tensor<256xf32>
    %v3945 = stablehlo.concatenate %v3941, %v3944, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3946 = stablehlo.concatenate %v1565, %v3945, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g1dst = "stablehlo.all_reduce"(%v3946) ({
    ^bb0(%aras3b3g1dst: tensor<f32>, %arbs3b3g1dst: tensor<f32>):
      %aradds3b3g1dst = stablehlo.add %aras3b3g1dst, %arbs3b3g1dst : tensor<f32>
      stablehlo.return %aradds3b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g1dst = stablehlo.divide %arsums3b3g1dst, %arns3b3g1dst : tensor<1024xf32>
    %v3947 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3948 = stablehlo.slice %armeans3b3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3949 = stablehlo.slice %armeans3b3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3950 = stablehlo.slice %armeans3b3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3951 = stablehlo.slice %armeans3b3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3952 = stablehlo.broadcast_in_dim %v3948, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3953 = stablehlo.broadcast_in_dim %v3949, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3954 = stablehlo.broadcast_in_dim %v3950, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3955 = stablehlo.broadcast_in_dim %v3951, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3956 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3957 = stablehlo.add %v3953, %v3956 : tensor<64x256x14x14xf32>
    %v3958 = stablehlo.rsqrt %v3957 : tensor<64x256x14x14xf32>
    %v3959 = stablehlo.subtract %v3947, %v3952 : tensor<64x256x14x14xf32>
    %v3960 = stablehlo.multiply %v3959, %v3958 : tensor<64x256x14x14xf32>
    %v3961 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3962 = stablehlo.reshape %v3924 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3963 = stablehlo.multiply %v3961, %v3962 : tensor<64x256x14x14xf32>
    %v3964 = stablehlo.subtract %v3963, %v3954 : tensor<64x256x14x14xf32>
    %v3965 = stablehlo.multiply %v3960, %v3955 : tensor<64x256x14x14xf32>
    %v3966 = stablehlo.subtract %v3964, %v3965 : tensor<64x256x14x14xf32>
    %v3967 = stablehlo.multiply %v3958, %v3966 : tensor<64x256x14x14xf32>
    %v3968 = stablehlo.reshape %v3967 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3969 = stablehlo.reshape %v3968 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3970 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3971 = stablehlo.transpose %v3970, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3972 = stablehlo.convert %v3969 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3973 = stablehlo.convert %v3971 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3974 = stablehlo.convolution(%v3972, %v3973)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3975 = stablehlo.convert %v3974 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3976 = stablehlo.reshape %v3975 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3977 = stablehlo.reshape %v3976 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3978 = stablehlo.reshape %v3808 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3979 = stablehlo.add %v3977, %v3978 : tensor<64x1024x14x14xf32>
    %v3980 = stablehlo.reshape %v3979 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3981 = stablehlo.reshape %v1538 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3982 = stablehlo.reshape %v3968 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3983 = stablehlo.transpose %v3981, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3984 = stablehlo.transpose %v3982, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3985 = stablehlo.convert %v3983 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3986 = stablehlo.convert %v3984 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3987 = stablehlo.convolution(%v3985, %v3986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3988 = stablehlo.convert %v3987 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3989 = stablehlo.transpose %v3988, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3990 = stablehlo.reshape %v1546 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3991 = stablehlo.slice %v1565 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3992 = stablehlo.slice %v1565 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3993 = stablehlo.broadcast_in_dim %v3991, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3992, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3995 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3996 = stablehlo.add %v3994, %v3995 : tensor<64x256x14x14xf32>
    %v3997 = stablehlo.rsqrt %v3996 : tensor<64x256x14x14xf32>
    %v3998 = stablehlo.subtract %v3990, %v3993 : tensor<64x256x14x14xf32>
    %v3999 = stablehlo.multiply %v3998, %v3997 : tensor<64x256x14x14xf32>
    %v4000 = stablehlo.reshape %v3924 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4001 = stablehlo.multiply %v4000, %v3999 : tensor<64x256x14x14xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4004 = stablehlo.reshape %v3924 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4006 = stablehlo.reduce(%v4004 init: %v4005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4007 = stablehlo.reshape %v1582 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4008 = stablehlo.reshape %v3910 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4009 = stablehlo.transpose %v4007, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4010 = stablehlo.transpose %v4008, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4011 = stablehlo.convert %v4009 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4012 = stablehlo.convert %v4010 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4013 = stablehlo.convolution(%v4011, %v4012)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v4014 = stablehlo.convert %v4013 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v4015 = stablehlo.transpose %v4014, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4016 = stablehlo.reshape %v1590 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4017 = stablehlo.slice %v1609 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4018 = stablehlo.slice %v1609 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4019 = stablehlo.broadcast_in_dim %v4017, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4020 = stablehlo.broadcast_in_dim %v4018, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4021 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4022 = stablehlo.add %v4020, %v4021 : tensor<64x256x14x14xf32>
    %v4023 = stablehlo.rsqrt %v4022 : tensor<64x256x14x14xf32>
    %v4024 = stablehlo.subtract %v4016, %v4019 : tensor<64x256x14x14xf32>
    %v4025 = stablehlo.multiply %v4024, %v4023 : tensor<64x256x14x14xf32>
    %v4026 = stablehlo.reshape %v3866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4027 = stablehlo.multiply %v4026, %v4025 : tensor<64x256x14x14xf32>
    %v4028 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4029 = stablehlo.reduce(%v4027 init: %v4028) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4030 = stablehlo.reshape %v3866 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4032 = stablehlo.reduce(%v4030 init: %v4031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4033 = stablehlo.reshape %v1626 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4034 = stablehlo.reshape %v3852 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4035 = stablehlo.transpose %v4033, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4036 = stablehlo.transpose %v4034, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4037 = stablehlo.convert %v4035 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4038 = stablehlo.convert %v4036 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4039 = stablehlo.convolution(%v4037, %v4038)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v4040 = stablehlo.convert %v4039 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v4041 = stablehlo.transpose %v4040, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4042 = stablehlo.reshape %v1634 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4043 = stablehlo.slice %v1653 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4044 = stablehlo.slice %v1653 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4045 = stablehlo.broadcast_in_dim %v4043, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4046 = stablehlo.broadcast_in_dim %v4044, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4047 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4048 = stablehlo.add %v4046, %v4047 : tensor<64x1024x14x14xf32>
    %v4049 = stablehlo.rsqrt %v4048 : tensor<64x1024x14x14xf32>
    %v4050 = stablehlo.subtract %v4042, %v4045 : tensor<64x1024x14x14xf32>
    %v4051 = stablehlo.multiply %v4050, %v4049 : tensor<64x1024x14x14xf32>
    %v4052 = stablehlo.reshape %v3808 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4053 = stablehlo.multiply %v4052, %v4051 : tensor<64x1024x14x14xf32>
    %v4054 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4055 = stablehlo.reduce(%v4053 init: %v4054) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4056 = stablehlo.reshape %v3808 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4057 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4058 = stablehlo.reduce(%v4056 init: %v4057) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4059 = stablehlo.reshape %v3980 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4060 = stablehlo.reshape %v1534 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v4062 = stablehlo.compare GT, %v4060, %v4061 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v4063 = stablehlo.select %v4062, %v4059, %v4061 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v4064 = stablehlo.reshape %v4063 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4065 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4066 = stablehlo.slice %v1515 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4067 = stablehlo.slice %v1515 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4068 = stablehlo.broadcast_in_dim %v4066, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4069 = stablehlo.broadcast_in_dim %v4067, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4070 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4071 = stablehlo.add %v4069, %v4070 : tensor<64x1024x14x14xf32>
    %v4072 = stablehlo.rsqrt %v4071 : tensor<64x1024x14x14xf32>
    %v4073 = stablehlo.subtract %v4065, %v4068 : tensor<64x1024x14x14xf32>
    %v4074 = stablehlo.multiply %v4073, %v4072 : tensor<64x1024x14x14xf32>
    %v4075 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4076 = stablehlo.reshape %v4064 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4077 = stablehlo.multiply %v4075, %v4076 : tensor<64x1024x14x14xf32>
    %v4078 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4079 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4080 = stablehlo.reduce(%v4077 init: %v4078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4081 = stablehlo.divide %v4080, %v4079 : tensor<1024xf32>
    %v4082 = stablehlo.multiply %v4074, %v4077 : tensor<64x1024x14x14xf32>
    %v4083 = stablehlo.reduce(%v4082 init: %v4078) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4084 = stablehlo.divide %v4083, %v4079 : tensor<1024xf32>
    %v4085 = stablehlo.concatenate %v4081, %v4084, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4086 = stablehlo.concatenate %v1515, %v4085, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b2g3dst = "stablehlo.all_reduce"(%v4086) ({
    ^bb0(%aras3b2g3dst: tensor<f32>, %arbs3b2g3dst: tensor<f32>):
      %aradds3b2g3dst = stablehlo.add %aras3b2g3dst, %arbs3b2g3dst : tensor<f32>
      stablehlo.return %aradds3b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b2g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b2g3dst = stablehlo.divide %arsums3b2g3dst, %arns3b2g3dst : tensor<4096xf32>
    %v4087 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4088 = stablehlo.slice %armeans3b2g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4089 = stablehlo.slice %armeans3b2g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4090 = stablehlo.slice %armeans3b2g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4091 = stablehlo.slice %armeans3b2g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4092 = stablehlo.broadcast_in_dim %v4088, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4093 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4090, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4095 = stablehlo.broadcast_in_dim %v4091, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4096 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4097 = stablehlo.add %v4093, %v4096 : tensor<64x1024x14x14xf32>
    %v4098 = stablehlo.rsqrt %v4097 : tensor<64x1024x14x14xf32>
    %v4099 = stablehlo.subtract %v4087, %v4092 : tensor<64x1024x14x14xf32>
    %v4100 = stablehlo.multiply %v4099, %v4098 : tensor<64x1024x14x14xf32>
    %v4101 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4102 = stablehlo.reshape %v4064 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4103 = stablehlo.multiply %v4101, %v4102 : tensor<64x1024x14x14xf32>
    %v4104 = stablehlo.subtract %v4103, %v4094 : tensor<64x1024x14x14xf32>
    %v4105 = stablehlo.multiply %v4100, %v4095 : tensor<64x1024x14x14xf32>
    %v4106 = stablehlo.subtract %v4104, %v4105 : tensor<64x1024x14x14xf32>
    %v4107 = stablehlo.multiply %v4098, %v4106 : tensor<64x1024x14x14xf32>
    %v4108 = stablehlo.reshape %v4107 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4109 = stablehlo.reshape %v4108 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4110 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v4111 = stablehlo.transpose %v4110, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4112 = stablehlo.convert %v4109 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v4113 = stablehlo.convert %v4111 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v4114 = stablehlo.convolution(%v4112, %v4113)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v4115 = stablehlo.convert %v4114 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v4116 = stablehlo.reshape %v4115 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4117 = stablehlo.reshape %v4116 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4118 = stablehlo.reshape %v1486 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4119 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4120 = stablehlo.compare GT, %v4118, %v4119 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4121 = stablehlo.select %v4120, %v4117, %v4119 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4122 = stablehlo.reshape %v4121 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4123 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4124 = stablehlo.slice %v1471 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4125 = stablehlo.slice %v1471 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4126 = stablehlo.broadcast_in_dim %v4124, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4127 = stablehlo.broadcast_in_dim %v4125, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4128 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4129 = stablehlo.add %v4127, %v4128 : tensor<64x256x14x14xf32>
    %v4130 = stablehlo.rsqrt %v4129 : tensor<64x256x14x14xf32>
    %v4131 = stablehlo.subtract %v4123, %v4126 : tensor<64x256x14x14xf32>
    %v4132 = stablehlo.multiply %v4131, %v4130 : tensor<64x256x14x14xf32>
    %v4133 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4134 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4135 = stablehlo.multiply %v4133, %v4134 : tensor<64x256x14x14xf32>
    %v4136 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4137 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4138 = stablehlo.reduce(%v4135 init: %v4136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4139 = stablehlo.divide %v4138, %v4137 : tensor<256xf32>
    %v4140 = stablehlo.multiply %v4132, %v4135 : tensor<64x256x14x14xf32>
    %v4141 = stablehlo.reduce(%v4140 init: %v4136) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4142 = stablehlo.divide %v4141, %v4137 : tensor<256xf32>
    %v4143 = stablehlo.concatenate %v4139, %v4142, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4144 = stablehlo.concatenate %v1471, %v4143, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g2dst = "stablehlo.all_reduce"(%v4144) ({
    ^bb0(%aras3b2g2dst: tensor<f32>, %arbs3b2g2dst: tensor<f32>):
      %aradds3b2g2dst = stablehlo.add %aras3b2g2dst, %arbs3b2g2dst : tensor<f32>
      stablehlo.return %aradds3b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g2dst = stablehlo.divide %arsums3b2g2dst, %arns3b2g2dst : tensor<1024xf32>
    %v4145 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4146 = stablehlo.slice %armeans3b2g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4147 = stablehlo.slice %armeans3b2g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4148 = stablehlo.slice %armeans3b2g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4149 = stablehlo.slice %armeans3b2g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4150 = stablehlo.broadcast_in_dim %v4146, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4151 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4152 = stablehlo.broadcast_in_dim %v4148, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4153 = stablehlo.broadcast_in_dim %v4149, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4154 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4155 = stablehlo.add %v4151, %v4154 : tensor<64x256x14x14xf32>
    %v4156 = stablehlo.rsqrt %v4155 : tensor<64x256x14x14xf32>
    %v4157 = stablehlo.subtract %v4145, %v4150 : tensor<64x256x14x14xf32>
    %v4158 = stablehlo.multiply %v4157, %v4156 : tensor<64x256x14x14xf32>
    %v4159 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4160 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4161 = stablehlo.multiply %v4159, %v4160 : tensor<64x256x14x14xf32>
    %v4162 = stablehlo.subtract %v4161, %v4152 : tensor<64x256x14x14xf32>
    %v4163 = stablehlo.multiply %v4158, %v4153 : tensor<64x256x14x14xf32>
    %v4164 = stablehlo.subtract %v4162, %v4163 : tensor<64x256x14x14xf32>
    %v4165 = stablehlo.multiply %v4156, %v4164 : tensor<64x256x14x14xf32>
    %v4166 = stablehlo.reshape %v4165 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4167 = stablehlo.reshape %v4166 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4168 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v4169 = stablehlo.transpose %v4168, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4170 = stablehlo.convert %v4167 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v4171 = stablehlo.convert %v4169 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v4172 = stablehlo.convolution(%v4170, %v4171)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v4173 = stablehlo.convert %v4172 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v4174 = stablehlo.reshape %v4173 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4175 = stablehlo.reshape %v4174 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4176 = stablehlo.reshape %v1442 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4177 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4178 = stablehlo.compare GT, %v4176, %v4177 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4179 = stablehlo.select %v4178, %v4175, %v4177 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4180 = stablehlo.reshape %v4179 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4181 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4182 = stablehlo.slice %v1427 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4183 = stablehlo.slice %v1427 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4184 = stablehlo.broadcast_in_dim %v4182, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4185 = stablehlo.broadcast_in_dim %v4183, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4186 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4187 = stablehlo.add %v4185, %v4186 : tensor<64x256x14x14xf32>
    %v4188 = stablehlo.rsqrt %v4187 : tensor<64x256x14x14xf32>
    %v4189 = stablehlo.subtract %v4181, %v4184 : tensor<64x256x14x14xf32>
    %v4190 = stablehlo.multiply %v4189, %v4188 : tensor<64x256x14x14xf32>
    %v4191 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4192 = stablehlo.reshape %v4180 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4193 = stablehlo.multiply %v4191, %v4192 : tensor<64x256x14x14xf32>
    %v4194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4195 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4196 = stablehlo.reduce(%v4193 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4197 = stablehlo.divide %v4196, %v4195 : tensor<256xf32>
    %v4198 = stablehlo.multiply %v4190, %v4193 : tensor<64x256x14x14xf32>
    %v4199 = stablehlo.reduce(%v4198 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4200 = stablehlo.divide %v4199, %v4195 : tensor<256xf32>
    %v4201 = stablehlo.concatenate %v4197, %v4200, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4202 = stablehlo.concatenate %v1427, %v4201, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g1dst = "stablehlo.all_reduce"(%v4202) ({
    ^bb0(%aras3b2g1dst: tensor<f32>, %arbs3b2g1dst: tensor<f32>):
      %aradds3b2g1dst = stablehlo.add %aras3b2g1dst, %arbs3b2g1dst : tensor<f32>
      stablehlo.return %aradds3b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g1dst = stablehlo.divide %arsums3b2g1dst, %arns3b2g1dst : tensor<1024xf32>
    %v4203 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4204 = stablehlo.slice %armeans3b2g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4205 = stablehlo.slice %armeans3b2g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4206 = stablehlo.slice %armeans3b2g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4207 = stablehlo.slice %armeans3b2g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4208 = stablehlo.broadcast_in_dim %v4204, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4209 = stablehlo.broadcast_in_dim %v4205, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4210 = stablehlo.broadcast_in_dim %v4206, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4211 = stablehlo.broadcast_in_dim %v4207, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4212 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4213 = stablehlo.add %v4209, %v4212 : tensor<64x256x14x14xf32>
    %v4214 = stablehlo.rsqrt %v4213 : tensor<64x256x14x14xf32>
    %v4215 = stablehlo.subtract %v4203, %v4208 : tensor<64x256x14x14xf32>
    %v4216 = stablehlo.multiply %v4215, %v4214 : tensor<64x256x14x14xf32>
    %v4217 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4218 = stablehlo.reshape %v4180 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4219 = stablehlo.multiply %v4217, %v4218 : tensor<64x256x14x14xf32>
    %v4220 = stablehlo.subtract %v4219, %v4210 : tensor<64x256x14x14xf32>
    %v4221 = stablehlo.multiply %v4216, %v4211 : tensor<64x256x14x14xf32>
    %v4222 = stablehlo.subtract %v4220, %v4221 : tensor<64x256x14x14xf32>
    %v4223 = stablehlo.multiply %v4214, %v4222 : tensor<64x256x14x14xf32>
    %v4224 = stablehlo.reshape %v4223 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4225 = stablehlo.reshape %v4224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4226 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v4227 = stablehlo.transpose %v4226, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4228 = stablehlo.convert %v4225 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v4229 = stablehlo.convert %v4227 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v4230 = stablehlo.convolution(%v4228, %v4229)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v4231 = stablehlo.convert %v4230 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v4232 = stablehlo.reshape %v4231 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4233 = stablehlo.reshape %v4232 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4234 = stablehlo.reshape %v4064 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4235 = stablehlo.add %v4233, %v4234 : tensor<64x1024x14x14xf32>
    %v4236 = stablehlo.reshape %v4235 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4237 = stablehlo.reshape %v1400 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4238 = stablehlo.reshape %v4224 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4239 = stablehlo.transpose %v4237, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4240 = stablehlo.transpose %v4238, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4241 = stablehlo.convert %v4239 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4242 = stablehlo.convert %v4240 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4243 = stablehlo.convolution(%v4241, %v4242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v4244 = stablehlo.convert %v4243 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v4245 = stablehlo.transpose %v4244, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4246 = stablehlo.reshape %v1408 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4247 = stablehlo.slice %v1427 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4248 = stablehlo.slice %v1427 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4249 = stablehlo.broadcast_in_dim %v4247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4250 = stablehlo.broadcast_in_dim %v4248, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4251 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4252 = stablehlo.add %v4250, %v4251 : tensor<64x256x14x14xf32>
    %v4253 = stablehlo.rsqrt %v4252 : tensor<64x256x14x14xf32>
    %v4254 = stablehlo.subtract %v4246, %v4249 : tensor<64x256x14x14xf32>
    %v4255 = stablehlo.multiply %v4254, %v4253 : tensor<64x256x14x14xf32>
    %v4256 = stablehlo.reshape %v4180 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4257 = stablehlo.multiply %v4256, %v4255 : tensor<64x256x14x14xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4259 = stablehlo.reduce(%v4257 init: %v4258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4260 = stablehlo.reshape %v4180 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4262 = stablehlo.reduce(%v4260 init: %v4261) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4263 = stablehlo.reshape %v1444 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4264 = stablehlo.reshape %v4166 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4265 = stablehlo.transpose %v4263, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4266 = stablehlo.transpose %v4264, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4267 = stablehlo.convert %v4265 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4268 = stablehlo.convert %v4266 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4269 = stablehlo.convolution(%v4267, %v4268)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v4270 = stablehlo.convert %v4269 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v4271 = stablehlo.transpose %v4270, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4272 = stablehlo.reshape %v1452 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4273 = stablehlo.slice %v1471 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4274 = stablehlo.slice %v1471 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4275 = stablehlo.broadcast_in_dim %v4273, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4276 = stablehlo.broadcast_in_dim %v4274, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4277 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4278 = stablehlo.add %v4276, %v4277 : tensor<64x256x14x14xf32>
    %v4279 = stablehlo.rsqrt %v4278 : tensor<64x256x14x14xf32>
    %v4280 = stablehlo.subtract %v4272, %v4275 : tensor<64x256x14x14xf32>
    %v4281 = stablehlo.multiply %v4280, %v4279 : tensor<64x256x14x14xf32>
    %v4282 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4283 = stablehlo.multiply %v4282, %v4281 : tensor<64x256x14x14xf32>
    %v4284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4285 = stablehlo.reduce(%v4283 init: %v4284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4286 = stablehlo.reshape %v4122 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4287 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4288 = stablehlo.reduce(%v4286 init: %v4287) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4289 = stablehlo.reshape %v1488 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4290 = stablehlo.reshape %v4108 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4291 = stablehlo.transpose %v4289, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4292 = stablehlo.transpose %v4290, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4293 = stablehlo.convert %v4291 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4294 = stablehlo.convert %v4292 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4295 = stablehlo.convolution(%v4293, %v4294)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v4296 = stablehlo.convert %v4295 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v4297 = stablehlo.transpose %v4296, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4298 = stablehlo.reshape %v1496 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4299 = stablehlo.slice %v1515 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4300 = stablehlo.slice %v1515 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4301 = stablehlo.broadcast_in_dim %v4299, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4302 = stablehlo.broadcast_in_dim %v4300, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4303 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4304 = stablehlo.add %v4302, %v4303 : tensor<64x1024x14x14xf32>
    %v4305 = stablehlo.rsqrt %v4304 : tensor<64x1024x14x14xf32>
    %v4306 = stablehlo.subtract %v4298, %v4301 : tensor<64x1024x14x14xf32>
    %v4307 = stablehlo.multiply %v4306, %v4305 : tensor<64x1024x14x14xf32>
    %v4308 = stablehlo.reshape %v4064 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4309 = stablehlo.multiply %v4308, %v4307 : tensor<64x1024x14x14xf32>
    %v4310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4311 = stablehlo.reduce(%v4309 init: %v4310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4312 = stablehlo.reshape %v4064 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4314 = stablehlo.reduce(%v4312 init: %v4313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4315 = stablehlo.reshape %v4236 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4316 = stablehlo.reshape %v1396 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4317 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v4318 = stablehlo.compare GT, %v4316, %v4317 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v4319 = stablehlo.select %v4318, %v4315, %v4317 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v4320 = stablehlo.reshape %v4319 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4321 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4322 = stablehlo.slice %v1377 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4323 = stablehlo.slice %v1377 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4324 = stablehlo.broadcast_in_dim %v4322, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4325 = stablehlo.broadcast_in_dim %v4323, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4326 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4327 = stablehlo.add %v4325, %v4326 : tensor<64x1024x14x14xf32>
    %v4328 = stablehlo.rsqrt %v4327 : tensor<64x1024x14x14xf32>
    %v4329 = stablehlo.subtract %v4321, %v4324 : tensor<64x1024x14x14xf32>
    %v4330 = stablehlo.multiply %v4329, %v4328 : tensor<64x1024x14x14xf32>
    %v4331 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4332 = stablehlo.reshape %v4320 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4333 = stablehlo.multiply %v4331, %v4332 : tensor<64x1024x14x14xf32>
    %v4334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4335 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4336 = stablehlo.reduce(%v4333 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4337 = stablehlo.divide %v4336, %v4335 : tensor<1024xf32>
    %v4338 = stablehlo.multiply %v4330, %v4333 : tensor<64x1024x14x14xf32>
    %v4339 = stablehlo.reduce(%v4338 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4340 = stablehlo.divide %v4339, %v4335 : tensor<1024xf32>
    %v4341 = stablehlo.concatenate %v4337, %v4340, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4342 = stablehlo.concatenate %v1377, %v4341, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b1g3dst = "stablehlo.all_reduce"(%v4342) ({
    ^bb0(%aras3b1g3dst: tensor<f32>, %arbs3b1g3dst: tensor<f32>):
      %aradds3b1g3dst = stablehlo.add %aras3b1g3dst, %arbs3b1g3dst : tensor<f32>
      stablehlo.return %aradds3b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b1g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b1g3dst = stablehlo.divide %arsums3b1g3dst, %arns3b1g3dst : tensor<4096xf32>
    %v4343 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4344 = stablehlo.slice %armeans3b1g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4345 = stablehlo.slice %armeans3b1g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4346 = stablehlo.slice %armeans3b1g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4347 = stablehlo.slice %armeans3b1g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4348 = stablehlo.broadcast_in_dim %v4344, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4349 = stablehlo.broadcast_in_dim %v4345, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4350 = stablehlo.broadcast_in_dim %v4346, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4351 = stablehlo.broadcast_in_dim %v4347, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4352 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4353 = stablehlo.add %v4349, %v4352 : tensor<64x1024x14x14xf32>
    %v4354 = stablehlo.rsqrt %v4353 : tensor<64x1024x14x14xf32>
    %v4355 = stablehlo.subtract %v4343, %v4348 : tensor<64x1024x14x14xf32>
    %v4356 = stablehlo.multiply %v4355, %v4354 : tensor<64x1024x14x14xf32>
    %v4357 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4358 = stablehlo.reshape %v4320 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4359 = stablehlo.multiply %v4357, %v4358 : tensor<64x1024x14x14xf32>
    %v4360 = stablehlo.subtract %v4359, %v4350 : tensor<64x1024x14x14xf32>
    %v4361 = stablehlo.multiply %v4356, %v4351 : tensor<64x1024x14x14xf32>
    %v4362 = stablehlo.subtract %v4360, %v4361 : tensor<64x1024x14x14xf32>
    %v4363 = stablehlo.multiply %v4354, %v4362 : tensor<64x1024x14x14xf32>
    %v4364 = stablehlo.reshape %v4363 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4365 = stablehlo.reshape %v4364 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4366 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v4367 = stablehlo.transpose %v4366, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4368 = stablehlo.convert %v4365 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v4369 = stablehlo.convert %v4367 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v4370 = stablehlo.convolution(%v4368, %v4369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v4371 = stablehlo.convert %v4370 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v4372 = stablehlo.reshape %v4371 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4373 = stablehlo.reshape %v4372 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4374 = stablehlo.reshape %v1348 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4375 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4376 = stablehlo.compare GT, %v4374, %v4375 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4377 = stablehlo.select %v4376, %v4373, %v4375 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4378 = stablehlo.reshape %v4377 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4379 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4380 = stablehlo.slice %v1333 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4381 = stablehlo.slice %v1333 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4382 = stablehlo.broadcast_in_dim %v4380, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4383 = stablehlo.broadcast_in_dim %v4381, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4384 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4385 = stablehlo.add %v4383, %v4384 : tensor<64x256x14x14xf32>
    %v4386 = stablehlo.rsqrt %v4385 : tensor<64x256x14x14xf32>
    %v4387 = stablehlo.subtract %v4379, %v4382 : tensor<64x256x14x14xf32>
    %v4388 = stablehlo.multiply %v4387, %v4386 : tensor<64x256x14x14xf32>
    %v4389 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4390 = stablehlo.reshape %v4378 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4391 = stablehlo.multiply %v4389, %v4390 : tensor<64x256x14x14xf32>
    %v4392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4393 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4394 = stablehlo.reduce(%v4391 init: %v4392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4395 = stablehlo.divide %v4394, %v4393 : tensor<256xf32>
    %v4396 = stablehlo.multiply %v4388, %v4391 : tensor<64x256x14x14xf32>
    %v4397 = stablehlo.reduce(%v4396 init: %v4392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4398 = stablehlo.divide %v4397, %v4393 : tensor<256xf32>
    %v4399 = stablehlo.concatenate %v4395, %v4398, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4400 = stablehlo.concatenate %v1333, %v4399, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g2dst = "stablehlo.all_reduce"(%v4400) ({
    ^bb0(%aras3b1g2dst: tensor<f32>, %arbs3b1g2dst: tensor<f32>):
      %aradds3b1g2dst = stablehlo.add %aras3b1g2dst, %arbs3b1g2dst : tensor<f32>
      stablehlo.return %aradds3b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g2dst = stablehlo.divide %arsums3b1g2dst, %arns3b1g2dst : tensor<1024xf32>
    %v4401 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4402 = stablehlo.slice %armeans3b1g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4403 = stablehlo.slice %armeans3b1g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4404 = stablehlo.slice %armeans3b1g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4405 = stablehlo.slice %armeans3b1g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4406 = stablehlo.broadcast_in_dim %v4402, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4407 = stablehlo.broadcast_in_dim %v4403, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4408 = stablehlo.broadcast_in_dim %v4404, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4409 = stablehlo.broadcast_in_dim %v4405, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4410 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4411 = stablehlo.add %v4407, %v4410 : tensor<64x256x14x14xf32>
    %v4412 = stablehlo.rsqrt %v4411 : tensor<64x256x14x14xf32>
    %v4413 = stablehlo.subtract %v4401, %v4406 : tensor<64x256x14x14xf32>
    %v4414 = stablehlo.multiply %v4413, %v4412 : tensor<64x256x14x14xf32>
    %v4415 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4416 = stablehlo.reshape %v4378 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4417 = stablehlo.multiply %v4415, %v4416 : tensor<64x256x14x14xf32>
    %v4418 = stablehlo.subtract %v4417, %v4408 : tensor<64x256x14x14xf32>
    %v4419 = stablehlo.multiply %v4414, %v4409 : tensor<64x256x14x14xf32>
    %v4420 = stablehlo.subtract %v4418, %v4419 : tensor<64x256x14x14xf32>
    %v4421 = stablehlo.multiply %v4412, %v4420 : tensor<64x256x14x14xf32>
    %v4422 = stablehlo.reshape %v4421 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4423 = stablehlo.reshape %v4422 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4424 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v4425 = stablehlo.transpose %v4424, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4426 = stablehlo.convert %v4423 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v4427 = stablehlo.convert %v4425 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v4428 = stablehlo.convolution(%v4426, %v4427)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v4429 = stablehlo.convert %v4428 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v4430 = stablehlo.reshape %v4429 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4431 = stablehlo.reshape %v4430 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4432 = stablehlo.reshape %v1304 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4433 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4434 = stablehlo.compare GT, %v4432, %v4433 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4435 = stablehlo.select %v4434, %v4431, %v4433 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4436 = stablehlo.reshape %v4435 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4437 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4438 = stablehlo.slice %v1289 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4439 = stablehlo.slice %v1289 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4440 = stablehlo.broadcast_in_dim %v4438, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4441 = stablehlo.broadcast_in_dim %v4439, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4442 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4443 = stablehlo.add %v4441, %v4442 : tensor<64x256x14x14xf32>
    %v4444 = stablehlo.rsqrt %v4443 : tensor<64x256x14x14xf32>
    %v4445 = stablehlo.subtract %v4437, %v4440 : tensor<64x256x14x14xf32>
    %v4446 = stablehlo.multiply %v4445, %v4444 : tensor<64x256x14x14xf32>
    %v4447 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4448 = stablehlo.reshape %v4436 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4449 = stablehlo.multiply %v4447, %v4448 : tensor<64x256x14x14xf32>
    %v4450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4451 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4452 = stablehlo.reduce(%v4449 init: %v4450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4453 = stablehlo.divide %v4452, %v4451 : tensor<256xf32>
    %v4454 = stablehlo.multiply %v4446, %v4449 : tensor<64x256x14x14xf32>
    %v4455 = stablehlo.reduce(%v4454 init: %v4450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4456 = stablehlo.divide %v4455, %v4451 : tensor<256xf32>
    %v4457 = stablehlo.concatenate %v4453, %v4456, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4458 = stablehlo.concatenate %v1289, %v4457, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g1dst = "stablehlo.all_reduce"(%v4458) ({
    ^bb0(%aras3b1g1dst: tensor<f32>, %arbs3b1g1dst: tensor<f32>):
      %aradds3b1g1dst = stablehlo.add %aras3b1g1dst, %arbs3b1g1dst : tensor<f32>
      stablehlo.return %aradds3b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g1dst = stablehlo.divide %arsums3b1g1dst, %arns3b1g1dst : tensor<1024xf32>
    %v4459 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4460 = stablehlo.slice %armeans3b1g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4461 = stablehlo.slice %armeans3b1g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4462 = stablehlo.slice %armeans3b1g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4463 = stablehlo.slice %armeans3b1g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4464 = stablehlo.broadcast_in_dim %v4460, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4465 = stablehlo.broadcast_in_dim %v4461, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4466 = stablehlo.broadcast_in_dim %v4462, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4467 = stablehlo.broadcast_in_dim %v4463, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4468 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4469 = stablehlo.add %v4465, %v4468 : tensor<64x256x14x14xf32>
    %v4470 = stablehlo.rsqrt %v4469 : tensor<64x256x14x14xf32>
    %v4471 = stablehlo.subtract %v4459, %v4464 : tensor<64x256x14x14xf32>
    %v4472 = stablehlo.multiply %v4471, %v4470 : tensor<64x256x14x14xf32>
    %v4473 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4474 = stablehlo.reshape %v4436 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4475 = stablehlo.multiply %v4473, %v4474 : tensor<64x256x14x14xf32>
    %v4476 = stablehlo.subtract %v4475, %v4466 : tensor<64x256x14x14xf32>
    %v4477 = stablehlo.multiply %v4472, %v4467 : tensor<64x256x14x14xf32>
    %v4478 = stablehlo.subtract %v4476, %v4477 : tensor<64x256x14x14xf32>
    %v4479 = stablehlo.multiply %v4470, %v4478 : tensor<64x256x14x14xf32>
    %v4480 = stablehlo.reshape %v4479 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4481 = stablehlo.reshape %v4480 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4482 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v4483 = stablehlo.transpose %v4482, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4484 = stablehlo.convert %v4481 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v4485 = stablehlo.convert %v4483 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v4486 = stablehlo.convolution(%v4484, %v4485)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v4487 = stablehlo.convert %v4486 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v4488 = stablehlo.reshape %v4487 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4489 = stablehlo.reshape %v4488 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4490 = stablehlo.reshape %v4320 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4491 = stablehlo.add %v4489, %v4490 : tensor<64x1024x14x14xf32>
    %v4492 = stablehlo.reshape %v4491 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4493 = stablehlo.reshape %v1262 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4494 = stablehlo.reshape %v4480 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4495 = stablehlo.transpose %v4493, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4496 = stablehlo.transpose %v4494, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4497 = stablehlo.convert %v4495 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4498 = stablehlo.convert %v4496 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4499 = stablehlo.convolution(%v4497, %v4498)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v4500 = stablehlo.convert %v4499 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v4501 = stablehlo.transpose %v4500, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4502 = stablehlo.reshape %v1270 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4503 = stablehlo.slice %v1289 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4504 = stablehlo.slice %v1289 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4505 = stablehlo.broadcast_in_dim %v4503, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4506 = stablehlo.broadcast_in_dim %v4504, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4507 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4508 = stablehlo.add %v4506, %v4507 : tensor<64x256x14x14xf32>
    %v4509 = stablehlo.rsqrt %v4508 : tensor<64x256x14x14xf32>
    %v4510 = stablehlo.subtract %v4502, %v4505 : tensor<64x256x14x14xf32>
    %v4511 = stablehlo.multiply %v4510, %v4509 : tensor<64x256x14x14xf32>
    %v4512 = stablehlo.reshape %v4436 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4513 = stablehlo.multiply %v4512, %v4511 : tensor<64x256x14x14xf32>
    %v4514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4515 = stablehlo.reduce(%v4513 init: %v4514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4516 = stablehlo.reshape %v4436 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4518 = stablehlo.reduce(%v4516 init: %v4517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4519 = stablehlo.reshape %v1306 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4520 = stablehlo.reshape %v4422 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4521 = stablehlo.transpose %v4519, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4522 = stablehlo.transpose %v4520, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4523 = stablehlo.convert %v4521 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4524 = stablehlo.convert %v4522 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4525 = stablehlo.convolution(%v4523, %v4524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v4526 = stablehlo.convert %v4525 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v4527 = stablehlo.transpose %v4526, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4528 = stablehlo.reshape %v1314 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4529 = stablehlo.slice %v1333 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4530 = stablehlo.slice %v1333 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4531 = stablehlo.broadcast_in_dim %v4529, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4532 = stablehlo.broadcast_in_dim %v4530, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4533 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4534 = stablehlo.add %v4532, %v4533 : tensor<64x256x14x14xf32>
    %v4535 = stablehlo.rsqrt %v4534 : tensor<64x256x14x14xf32>
    %v4536 = stablehlo.subtract %v4528, %v4531 : tensor<64x256x14x14xf32>
    %v4537 = stablehlo.multiply %v4536, %v4535 : tensor<64x256x14x14xf32>
    %v4538 = stablehlo.reshape %v4378 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4539 = stablehlo.multiply %v4538, %v4537 : tensor<64x256x14x14xf32>
    %v4540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4541 = stablehlo.reduce(%v4539 init: %v4540) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4542 = stablehlo.reshape %v4378 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4544 = stablehlo.reduce(%v4542 init: %v4543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4545 = stablehlo.reshape %v1350 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4546 = stablehlo.reshape %v4364 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4547 = stablehlo.transpose %v4545, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4548 = stablehlo.transpose %v4546, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4549 = stablehlo.convert %v4547 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4550 = stablehlo.convert %v4548 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4551 = stablehlo.convolution(%v4549, %v4550)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v4552 = stablehlo.convert %v4551 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v4553 = stablehlo.transpose %v4552, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4554 = stablehlo.reshape %v1358 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4555 = stablehlo.slice %v1377 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4556 = stablehlo.slice %v1377 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4557 = stablehlo.broadcast_in_dim %v4555, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4558 = stablehlo.broadcast_in_dim %v4556, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4559 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4560 = stablehlo.add %v4558, %v4559 : tensor<64x1024x14x14xf32>
    %v4561 = stablehlo.rsqrt %v4560 : tensor<64x1024x14x14xf32>
    %v4562 = stablehlo.subtract %v4554, %v4557 : tensor<64x1024x14x14xf32>
    %v4563 = stablehlo.multiply %v4562, %v4561 : tensor<64x1024x14x14xf32>
    %v4564 = stablehlo.reshape %v4320 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4565 = stablehlo.multiply %v4564, %v4563 : tensor<64x1024x14x14xf32>
    %v4566 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4567 = stablehlo.reduce(%v4565 init: %v4566) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4568 = stablehlo.reshape %v4320 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4570 = stablehlo.reduce(%v4568 init: %v4569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4571 = stablehlo.reshape %v4492 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4572 = stablehlo.reshape %v1260 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4573 = stablehlo.constant dense<0.0> : tensor<64x1024x14x14xf32>
    %v4574 = stablehlo.compare GT, %v4572, %v4573 : (tensor<64x1024x14x14xf32>, tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xi1>
    %v4575 = stablehlo.select %v4574, %v4571, %v4573 : tensor<64x1024x14x14xi1>, tensor<64x1024x14x14xf32>
    %v4576 = stablehlo.reshape %v4575 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4577 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4578 = stablehlo.slice %v1202 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4579 = stablehlo.slice %v1202 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4580 = stablehlo.broadcast_in_dim %v4578, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4581 = stablehlo.broadcast_in_dim %v4579, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4582 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4583 = stablehlo.add %v4581, %v4582 : tensor<64x1024x14x14xf32>
    %v4584 = stablehlo.rsqrt %v4583 : tensor<64x1024x14x14xf32>
    %v4585 = stablehlo.subtract %v4577, %v4580 : tensor<64x1024x14x14xf32>
    %v4586 = stablehlo.multiply %v4585, %v4584 : tensor<64x1024x14x14xf32>
    %v4587 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4588 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4589 = stablehlo.multiply %v4587, %v4588 : tensor<64x1024x14x14xf32>
    %v4590 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4591 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4592 = stablehlo.reduce(%v4589 init: %v4590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4593 = stablehlo.divide %v4592, %v4591 : tensor<1024xf32>
    %v4594 = stablehlo.multiply %v4586, %v4589 : tensor<64x1024x14x14xf32>
    %v4595 = stablehlo.reduce(%v4594 init: %v4590) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4596 = stablehlo.divide %v4595, %v4591 : tensor<1024xf32>
    %v4597 = stablehlo.concatenate %v4593, %v4596, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4598 = stablehlo.concatenate %v1202, %v4597, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b0g3dst = "stablehlo.all_reduce"(%v4598) ({
    ^bb0(%aras3b0g3dst: tensor<f32>, %arbs3b0g3dst: tensor<f32>):
      %aradds3b0g3dst = stablehlo.add %aras3b0g3dst, %arbs3b0g3dst : tensor<f32>
      stablehlo.return %aradds3b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b0g3dst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b0g3dst = stablehlo.divide %arsums3b0g3dst, %arns3b0g3dst : tensor<4096xf32>
    %v4599 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4600 = stablehlo.slice %armeans3b0g3dst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4601 = stablehlo.slice %armeans3b0g3dst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4602 = stablehlo.slice %armeans3b0g3dst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4603 = stablehlo.slice %armeans3b0g3dst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4604 = stablehlo.broadcast_in_dim %v4600, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4605 = stablehlo.broadcast_in_dim %v4601, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4606 = stablehlo.broadcast_in_dim %v4602, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4607 = stablehlo.broadcast_in_dim %v4603, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4608 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4609 = stablehlo.add %v4605, %v4608 : tensor<64x1024x14x14xf32>
    %v4610 = stablehlo.rsqrt %v4609 : tensor<64x1024x14x14xf32>
    %v4611 = stablehlo.subtract %v4599, %v4604 : tensor<64x1024x14x14xf32>
    %v4612 = stablehlo.multiply %v4611, %v4610 : tensor<64x1024x14x14xf32>
    %v4613 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4614 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4615 = stablehlo.multiply %v4613, %v4614 : tensor<64x1024x14x14xf32>
    %v4616 = stablehlo.subtract %v4615, %v4606 : tensor<64x1024x14x14xf32>
    %v4617 = stablehlo.multiply %v4612, %v4607 : tensor<64x1024x14x14xf32>
    %v4618 = stablehlo.subtract %v4616, %v4617 : tensor<64x1024x14x14xf32>
    %v4619 = stablehlo.multiply %v4610, %v4618 : tensor<64x1024x14x14xf32>
    %v4620 = stablehlo.reshape %v4619 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4621 = stablehlo.reshape %v4620 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4622 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v4623 = stablehlo.transpose %v4622, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v4624 = stablehlo.convert %v4621 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v4625 = stablehlo.convert %v4623 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v4626 = stablehlo.convolution(%v4624, %v4625)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v4627 = stablehlo.convert %v4626 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v4628 = stablehlo.reshape %v4627 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4629 = stablehlo.reshape %v4628 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4630 = stablehlo.reshape %v1173 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4631 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v4632 = stablehlo.compare GT, %v4630, %v4631 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v4633 = stablehlo.select %v4632, %v4629, %v4631 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v4634 = stablehlo.reshape %v4633 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4635 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4636 = stablehlo.slice %v1158 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4637 = stablehlo.slice %v1158 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4638 = stablehlo.broadcast_in_dim %v4636, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4639 = stablehlo.broadcast_in_dim %v4637, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4640 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4641 = stablehlo.add %v4639, %v4640 : tensor<64x256x14x14xf32>
    %v4642 = stablehlo.rsqrt %v4641 : tensor<64x256x14x14xf32>
    %v4643 = stablehlo.subtract %v4635, %v4638 : tensor<64x256x14x14xf32>
    %v4644 = stablehlo.multiply %v4643, %v4642 : tensor<64x256x14x14xf32>
    %v4645 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4646 = stablehlo.reshape %v4634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4647 = stablehlo.multiply %v4645, %v4646 : tensor<64x256x14x14xf32>
    %v4648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4649 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4650 = stablehlo.reduce(%v4647 init: %v4648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4651 = stablehlo.divide %v4650, %v4649 : tensor<256xf32>
    %v4652 = stablehlo.multiply %v4644, %v4647 : tensor<64x256x14x14xf32>
    %v4653 = stablehlo.reduce(%v4652 init: %v4648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4654 = stablehlo.divide %v4653, %v4649 : tensor<256xf32>
    %v4655 = stablehlo.concatenate %v4651, %v4654, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4656 = stablehlo.concatenate %v1158, %v4655, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g2dst = "stablehlo.all_reduce"(%v4656) ({
    ^bb0(%aras3b0g2dst: tensor<f32>, %arbs3b0g2dst: tensor<f32>):
      %aradds3b0g2dst = stablehlo.add %aras3b0g2dst, %arbs3b0g2dst : tensor<f32>
      stablehlo.return %aradds3b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g2dst = stablehlo.divide %arsums3b0g2dst, %arns3b0g2dst : tensor<1024xf32>
    %v4657 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4658 = stablehlo.slice %armeans3b0g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4659 = stablehlo.slice %armeans3b0g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4660 = stablehlo.slice %armeans3b0g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4661 = stablehlo.slice %armeans3b0g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4662 = stablehlo.broadcast_in_dim %v4658, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4663 = stablehlo.broadcast_in_dim %v4659, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4664 = stablehlo.broadcast_in_dim %v4660, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4665 = stablehlo.broadcast_in_dim %v4661, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4666 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4667 = stablehlo.add %v4663, %v4666 : tensor<64x256x14x14xf32>
    %v4668 = stablehlo.rsqrt %v4667 : tensor<64x256x14x14xf32>
    %v4669 = stablehlo.subtract %v4657, %v4662 : tensor<64x256x14x14xf32>
    %v4670 = stablehlo.multiply %v4669, %v4668 : tensor<64x256x14x14xf32>
    %v4671 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4672 = stablehlo.reshape %v4634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4673 = stablehlo.multiply %v4671, %v4672 : tensor<64x256x14x14xf32>
    %v4674 = stablehlo.subtract %v4673, %v4664 : tensor<64x256x14x14xf32>
    %v4675 = stablehlo.multiply %v4670, %v4665 : tensor<64x256x14x14xf32>
    %v4676 = stablehlo.subtract %v4674, %v4675 : tensor<64x256x14x14xf32>
    %v4677 = stablehlo.multiply %v4668, %v4676 : tensor<64x256x14x14xf32>
    %v4678 = stablehlo.reshape %v4677 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v4679 = stablehlo.reshape %v4678 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4681 = stablehlo.pad %v4679, %v4680, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v4682 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v4683 = stablehlo.transpose %v4682, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4684 = stablehlo.convert %v4681 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v4685 = stablehlo.convert %v4683 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v4686 = stablehlo.convolution(%v4684, %v4685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x28x28xbf16>
    %v4687 = stablehlo.convert %v4686 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v4688 = stablehlo.reshape %v4687 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4689 = stablehlo.reshape %v4688 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4690 = stablehlo.reshape %v1129 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4691 = stablehlo.constant dense<0.0> : tensor<64x256x28x28xf32>
    %v4692 = stablehlo.compare GT, %v4690, %v4691 : (tensor<64x256x28x28xf32>, tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xi1>
    %v4693 = stablehlo.select %v4692, %v4689, %v4691 : tensor<64x256x28x28xi1>, tensor<64x256x28x28xf32>
    %v4694 = stablehlo.reshape %v4693 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4695 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4696 = stablehlo.slice %v1114 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4697 = stablehlo.slice %v1114 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4698 = stablehlo.broadcast_in_dim %v4696, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4699 = stablehlo.broadcast_in_dim %v4697, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4700 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4701 = stablehlo.add %v4699, %v4700 : tensor<64x256x28x28xf32>
    %v4702 = stablehlo.rsqrt %v4701 : tensor<64x256x28x28xf32>
    %v4703 = stablehlo.subtract %v4695, %v4698 : tensor<64x256x28x28xf32>
    %v4704 = stablehlo.multiply %v4703, %v4702 : tensor<64x256x28x28xf32>
    %v4705 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4706 = stablehlo.reshape %v4694 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4707 = stablehlo.multiply %v4705, %v4706 : tensor<64x256x28x28xf32>
    %v4708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4709 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v4710 = stablehlo.reduce(%v4707 init: %v4708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4711 = stablehlo.divide %v4710, %v4709 : tensor<256xf32>
    %v4712 = stablehlo.multiply %v4704, %v4707 : tensor<64x256x28x28xf32>
    %v4713 = stablehlo.reduce(%v4712 init: %v4708) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4714 = stablehlo.divide %v4713, %v4709 : tensor<256xf32>
    %v4715 = stablehlo.concatenate %v4711, %v4714, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v4716 = stablehlo.concatenate %v1114, %v4715, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g1dst = "stablehlo.all_reduce"(%v4716) ({
    ^bb0(%aras3b0g1dst: tensor<f32>, %arbs3b0g1dst: tensor<f32>):
      %aradds3b0g1dst = stablehlo.add %aras3b0g1dst, %arbs3b0g1dst : tensor<f32>
      stablehlo.return %aradds3b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g1dst = stablehlo.divide %arsums3b0g1dst, %arns3b0g1dst : tensor<1024xf32>
    %v4717 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4718 = stablehlo.slice %armeans3b0g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4719 = stablehlo.slice %armeans3b0g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4720 = stablehlo.slice %armeans3b0g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4721 = stablehlo.slice %armeans3b0g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v4722 = stablehlo.broadcast_in_dim %v4718, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4723 = stablehlo.broadcast_in_dim %v4719, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4724 = stablehlo.broadcast_in_dim %v4720, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4725 = stablehlo.broadcast_in_dim %v4721, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4726 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4727 = stablehlo.add %v4723, %v4726 : tensor<64x256x28x28xf32>
    %v4728 = stablehlo.rsqrt %v4727 : tensor<64x256x28x28xf32>
    %v4729 = stablehlo.subtract %v4717, %v4722 : tensor<64x256x28x28xf32>
    %v4730 = stablehlo.multiply %v4729, %v4728 : tensor<64x256x28x28xf32>
    %v4731 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4732 = stablehlo.reshape %v4694 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4733 = stablehlo.multiply %v4731, %v4732 : tensor<64x256x28x28xf32>
    %v4734 = stablehlo.subtract %v4733, %v4724 : tensor<64x256x28x28xf32>
    %v4735 = stablehlo.multiply %v4730, %v4725 : tensor<64x256x28x28xf32>
    %v4736 = stablehlo.subtract %v4734, %v4735 : tensor<64x256x28x28xf32>
    %v4737 = stablehlo.multiply %v4728, %v4736 : tensor<64x256x28x28xf32>
    %v4738 = stablehlo.reshape %v4737 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v4739 = stablehlo.reshape %v4738 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4740 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v4741 = stablehlo.transpose %v4740, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4742 = stablehlo.convert %v4739 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v4743 = stablehlo.convert %v4741 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v4744 = stablehlo.convolution(%v4742, %v4743)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4745 = stablehlo.convert %v4744 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4746 = stablehlo.reshape %v4745 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4747 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4748 = stablehlo.slice %v1244 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4749 = stablehlo.slice %v1244 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4750 = stablehlo.broadcast_in_dim %v4748, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4751 = stablehlo.broadcast_in_dim %v4749, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4752 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4753 = stablehlo.add %v4751, %v4752 : tensor<64x1024x14x14xf32>
    %v4754 = stablehlo.rsqrt %v4753 : tensor<64x1024x14x14xf32>
    %v4755 = stablehlo.subtract %v4747, %v4750 : tensor<64x1024x14x14xf32>
    %v4756 = stablehlo.multiply %v4755, %v4754 : tensor<64x1024x14x14xf32>
    %v4757 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4758 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4759 = stablehlo.multiply %v4757, %v4758 : tensor<64x1024x14x14xf32>
    %v4760 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4761 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v4762 = stablehlo.reduce(%v4759 init: %v4760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4763 = stablehlo.divide %v4762, %v4761 : tensor<1024xf32>
    %v4764 = stablehlo.multiply %v4756, %v4759 : tensor<64x1024x14x14xf32>
    %v4765 = stablehlo.reduce(%v4764 init: %v4760) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4766 = stablehlo.divide %v4765, %v4761 : tensor<1024xf32>
    %v4767 = stablehlo.concatenate %v4763, %v4766, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %v4768 = stablehlo.concatenate %v1244, %v4767, dim = 0 : (tensor<2048xf32>, tensor<2048xf32>) -> tensor<4096xf32>
    %arsums3b0gpdst = "stablehlo.all_reduce"(%v4768) ({
    ^bb0(%aras3b0gpdst: tensor<f32>, %arbs3b0gpdst: tensor<f32>):
      %aradds3b0gpdst = stablehlo.add %aras3b0gpdst, %arbs3b0gpdst : tensor<f32>
      stablehlo.return %aradds3b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<4096xf32>) -> tensor<4096xf32>
    %arns3b0gpdst = stablehlo.constant dense<4.0> : tensor<4096xf32>
    %armeans3b0gpdst = stablehlo.divide %arsums3b0gpdst, %arns3b0gpdst : tensor<4096xf32>
    %v4769 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4770 = stablehlo.slice %armeans3b0gpdst [0:1024] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4771 = stablehlo.slice %armeans3b0gpdst [1024:2048] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4772 = stablehlo.slice %armeans3b0gpdst [2048:3072] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4773 = stablehlo.slice %armeans3b0gpdst [3072:4096] : (tensor<4096xf32>) -> tensor<1024xf32>
    %v4774 = stablehlo.broadcast_in_dim %v4770, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4775 = stablehlo.broadcast_in_dim %v4771, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4776 = stablehlo.broadcast_in_dim %v4772, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4777 = stablehlo.broadcast_in_dim %v4773, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4778 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4779 = stablehlo.add %v4775, %v4778 : tensor<64x1024x14x14xf32>
    %v4780 = stablehlo.rsqrt %v4779 : tensor<64x1024x14x14xf32>
    %v4781 = stablehlo.subtract %v4769, %v4774 : tensor<64x1024x14x14xf32>
    %v4782 = stablehlo.multiply %v4781, %v4780 : tensor<64x1024x14x14xf32>
    %v4783 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4784 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4785 = stablehlo.multiply %v4783, %v4784 : tensor<64x1024x14x14xf32>
    %v4786 = stablehlo.subtract %v4785, %v4776 : tensor<64x1024x14x14xf32>
    %v4787 = stablehlo.multiply %v4782, %v4777 : tensor<64x1024x14x14xf32>
    %v4788 = stablehlo.subtract %v4786, %v4787 : tensor<64x1024x14x14xf32>
    %v4789 = stablehlo.multiply %v4780, %v4788 : tensor<64x1024x14x14xf32>
    %v4790 = stablehlo.reshape %v4789 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v4791 = stablehlo.reshape %v4790 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4792 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4793 = stablehlo.pad %v4791, %v4792, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v4794 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v4795 = stablehlo.transpose %v4794, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v4796 = stablehlo.convert %v4793 : (tensor<64x1024x28x28xf32>) -> tensor<64x1024x28x28xbf16>
    %v4797 = stablehlo.convert %v4795 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v4798 = stablehlo.convolution(%v4796, %v4797)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x28x28xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4799 = stablehlo.convert %v4798 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4800 = stablehlo.reshape %v4799 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4801 = stablehlo.reshape %v4746 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4802 = stablehlo.reshape %v4800 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4803 = stablehlo.add %v4801, %v4802 : tensor<64x512x28x28xf32>
    %v4804 = stablehlo.reshape %v4803 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4805 = stablehlo.reshape %v1087 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4806 = stablehlo.reshape %v4738 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4807 = stablehlo.transpose %v4805, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4808 = stablehlo.transpose %v4806, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v4809 = stablehlo.convert %v4807 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4810 = stablehlo.convert %v4808 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v4811 = stablehlo.convolution(%v4809, %v4810)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<512x256x1x1xbf16>
    %v4812 = stablehlo.convert %v4811 : (tensor<512x256x1x1xbf16>) -> tensor<512x256x1x1xf32>
    %v4813 = stablehlo.transpose %v4812, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4814 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4815 = stablehlo.slice %v1114 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4816 = stablehlo.slice %v1114 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4817 = stablehlo.broadcast_in_dim %v4815, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4818 = stablehlo.broadcast_in_dim %v4816, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v4819 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v4820 = stablehlo.add %v4818, %v4819 : tensor<64x256x28x28xf32>
    %v4821 = stablehlo.rsqrt %v4820 : tensor<64x256x28x28xf32>
    %v4822 = stablehlo.subtract %v4814, %v4817 : tensor<64x256x28x28xf32>
    %v4823 = stablehlo.multiply %v4822, %v4821 : tensor<64x256x28x28xf32>
    %v4824 = stablehlo.reshape %v4694 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4825 = stablehlo.multiply %v4824, %v4823 : tensor<64x256x28x28xf32>
    %v4826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4827 = stablehlo.reduce(%v4825 init: %v4826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4828 = stablehlo.reshape %v4694 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4830 = stablehlo.reduce(%v4828 init: %v4829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v4831 = stablehlo.reshape %v1131 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v4832 = stablehlo.reshape %v4678 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4834 = stablehlo.pad %v4832, %v4833, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v4835 = stablehlo.transpose %v4831, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v4836 = stablehlo.transpose %v4834, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v4837 = stablehlo.convert %v4835 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v4838 = stablehlo.convert %v4836 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v4839 = stablehlo.convolution(%v4837, %v4838)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<256x256x3x3xbf16>
    %v4840 = stablehlo.convert %v4839 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v4841 = stablehlo.transpose %v4840, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v4842 = stablehlo.reshape %v1139 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4843 = stablehlo.slice %v1158 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4844 = stablehlo.slice %v1158 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4845 = stablehlo.broadcast_in_dim %v4843, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4846 = stablehlo.broadcast_in_dim %v4844, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4847 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v4848 = stablehlo.add %v4846, %v4847 : tensor<64x256x14x14xf32>
    %v4849 = stablehlo.rsqrt %v4848 : tensor<64x256x14x14xf32>
    %v4850 = stablehlo.subtract %v4842, %v4845 : tensor<64x256x14x14xf32>
    %v4851 = stablehlo.multiply %v4850, %v4849 : tensor<64x256x14x14xf32>
    %v4852 = stablehlo.reshape %v4634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4853 = stablehlo.multiply %v4852, %v4851 : tensor<64x256x14x14xf32>
    %v4854 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4855 = stablehlo.reduce(%v4853 init: %v4854) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4856 = stablehlo.reshape %v4634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4858 = stablehlo.reduce(%v4856 init: %v4857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4859 = stablehlo.reshape %v1175 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4860 = stablehlo.reshape %v4620 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4861 = stablehlo.transpose %v4859, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v4862 = stablehlo.transpose %v4860, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v4863 = stablehlo.convert %v4861 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v4864 = stablehlo.convert %v4862 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v4865 = stablehlo.convolution(%v4863, %v4864)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v4866 = stablehlo.convert %v4865 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v4867 = stablehlo.transpose %v4866, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v4868 = stablehlo.reshape %v1183 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4869 = stablehlo.slice %v1202 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4870 = stablehlo.slice %v1202 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4871 = stablehlo.broadcast_in_dim %v4869, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4872 = stablehlo.broadcast_in_dim %v4870, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4873 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4874 = stablehlo.add %v4872, %v4873 : tensor<64x1024x14x14xf32>
    %v4875 = stablehlo.rsqrt %v4874 : tensor<64x1024x14x14xf32>
    %v4876 = stablehlo.subtract %v4868, %v4871 : tensor<64x1024x14x14xf32>
    %v4877 = stablehlo.multiply %v4876, %v4875 : tensor<64x1024x14x14xf32>
    %v4878 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4879 = stablehlo.multiply %v4878, %v4877 : tensor<64x1024x14x14xf32>
    %v4880 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4881 = stablehlo.reduce(%v4879 init: %v4880) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4882 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4884 = stablehlo.reduce(%v4882 init: %v4883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4885 = stablehlo.reshape %v1087 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4886 = stablehlo.reshape %v4790 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4888 = stablehlo.pad %v4886, %v4887, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v4889 = stablehlo.transpose %v4885, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4890 = stablehlo.transpose %v4888, dims = [1, 0, 2, 3] : (tensor<64x1024x28x28xf32>) -> tensor<1024x64x28x28xf32>
    %v4891 = stablehlo.convert %v4889 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4892 = stablehlo.convert %v4890 : (tensor<1024x64x28x28xf32>) -> tensor<1024x64x28x28xbf16>
    %v4893 = stablehlo.convolution(%v4891, %v4892)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<1024x64x28x28xbf16>) -> tensor<512x1024x1x1xbf16>
    %v4894 = stablehlo.convert %v4893 : (tensor<512x1024x1x1xbf16>) -> tensor<512x1024x1x1xf32>
    %v4895 = stablehlo.transpose %v4894, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v4896 = stablehlo.reshape %v1225 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4897 = stablehlo.slice %v1244 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4898 = stablehlo.slice %v1244 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v4899 = stablehlo.broadcast_in_dim %v4897, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4900 = stablehlo.broadcast_in_dim %v4898, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v4901 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v4902 = stablehlo.add %v4900, %v4901 : tensor<64x1024x14x14xf32>
    %v4903 = stablehlo.rsqrt %v4902 : tensor<64x1024x14x14xf32>
    %v4904 = stablehlo.subtract %v4896, %v4899 : tensor<64x1024x14x14xf32>
    %v4905 = stablehlo.multiply %v4904, %v4903 : tensor<64x1024x14x14xf32>
    %v4906 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4907 = stablehlo.multiply %v4906, %v4905 : tensor<64x1024x14x14xf32>
    %v4908 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4909 = stablehlo.reduce(%v4907 init: %v4908) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4910 = stablehlo.reshape %v4576 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v4911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4912 = stablehlo.reduce(%v4910 init: %v4911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v4913 = stablehlo.reshape %v4804 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4914 = stablehlo.reshape %v1083 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4915 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v4916 = stablehlo.compare GT, %v4914, %v4915 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v4917 = stablehlo.select %v4916, %v4913, %v4915 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v4918 = stablehlo.reshape %v4917 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4919 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4920 = stablehlo.slice %v1064 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4921 = stablehlo.slice %v1064 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4922 = stablehlo.broadcast_in_dim %v4920, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4923 = stablehlo.broadcast_in_dim %v4921, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4924 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4925 = stablehlo.add %v4923, %v4924 : tensor<64x512x28x28xf32>
    %v4926 = stablehlo.rsqrt %v4925 : tensor<64x512x28x28xf32>
    %v4927 = stablehlo.subtract %v4919, %v4922 : tensor<64x512x28x28xf32>
    %v4928 = stablehlo.multiply %v4927, %v4926 : tensor<64x512x28x28xf32>
    %v4929 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4930 = stablehlo.reshape %v4918 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4931 = stablehlo.multiply %v4929, %v4930 : tensor<64x512x28x28xf32>
    %v4932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4933 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v4934 = stablehlo.reduce(%v4931 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4935 = stablehlo.divide %v4934, %v4933 : tensor<512xf32>
    %v4936 = stablehlo.multiply %v4928, %v4931 : tensor<64x512x28x28xf32>
    %v4937 = stablehlo.reduce(%v4936 init: %v4932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4938 = stablehlo.divide %v4937, %v4933 : tensor<512xf32>
    %v4939 = stablehlo.concatenate %v4935, %v4938, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v4940 = stablehlo.concatenate %v1064, %v4939, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b3g3dst = "stablehlo.all_reduce"(%v4940) ({
    ^bb0(%aras2b3g3dst: tensor<f32>, %arbs2b3g3dst: tensor<f32>):
      %aradds2b3g3dst = stablehlo.add %aras2b3g3dst, %arbs2b3g3dst : tensor<f32>
      stablehlo.return %aradds2b3g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b3g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b3g3dst = stablehlo.divide %arsums2b3g3dst, %arns2b3g3dst : tensor<2048xf32>
    %v4941 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4942 = stablehlo.slice %armeans2b3g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4943 = stablehlo.slice %armeans2b3g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4944 = stablehlo.slice %armeans2b3g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4945 = stablehlo.slice %armeans2b3g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v4946 = stablehlo.broadcast_in_dim %v4942, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4947 = stablehlo.broadcast_in_dim %v4943, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4948 = stablehlo.broadcast_in_dim %v4944, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4949 = stablehlo.broadcast_in_dim %v4945, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4950 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4951 = stablehlo.add %v4947, %v4950 : tensor<64x512x28x28xf32>
    %v4952 = stablehlo.rsqrt %v4951 : tensor<64x512x28x28xf32>
    %v4953 = stablehlo.subtract %v4941, %v4946 : tensor<64x512x28x28xf32>
    %v4954 = stablehlo.multiply %v4953, %v4952 : tensor<64x512x28x28xf32>
    %v4955 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4956 = stablehlo.reshape %v4918 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4957 = stablehlo.multiply %v4955, %v4956 : tensor<64x512x28x28xf32>
    %v4958 = stablehlo.subtract %v4957, %v4948 : tensor<64x512x28x28xf32>
    %v4959 = stablehlo.multiply %v4954, %v4949 : tensor<64x512x28x28xf32>
    %v4960 = stablehlo.subtract %v4958, %v4959 : tensor<64x512x28x28xf32>
    %v4961 = stablehlo.multiply %v4952, %v4960 : tensor<64x512x28x28xf32>
    %v4962 = stablehlo.reshape %v4961 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4963 = stablehlo.reshape %v4962 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4964 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4965 = stablehlo.transpose %v4964, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4966 = stablehlo.convert %v4963 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4967 = stablehlo.convert %v4965 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4968 = stablehlo.convolution(%v4966, %v4967)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4969 = stablehlo.convert %v4968 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4970 = stablehlo.reshape %v4969 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4971 = stablehlo.reshape %v4970 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4972 = stablehlo.reshape %v1035 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4973 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v4974 = stablehlo.compare GT, %v4972, %v4973 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v4975 = stablehlo.select %v4974, %v4971, %v4973 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v4976 = stablehlo.reshape %v4975 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4977 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4978 = stablehlo.slice %v1020 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4979 = stablehlo.slice %v1020 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4980 = stablehlo.broadcast_in_dim %v4978, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4981 = stablehlo.broadcast_in_dim %v4979, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4982 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4983 = stablehlo.add %v4981, %v4982 : tensor<64x128x28x28xf32>
    %v4984 = stablehlo.rsqrt %v4983 : tensor<64x128x28x28xf32>
    %v4985 = stablehlo.subtract %v4977, %v4980 : tensor<64x128x28x28xf32>
    %v4986 = stablehlo.multiply %v4985, %v4984 : tensor<64x128x28x28xf32>
    %v4987 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4988 = stablehlo.reshape %v4976 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4989 = stablehlo.multiply %v4987, %v4988 : tensor<64x128x28x28xf32>
    %v4990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4991 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4992 = stablehlo.reduce(%v4989 init: %v4990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4993 = stablehlo.divide %v4992, %v4991 : tensor<128xf32>
    %v4994 = stablehlo.multiply %v4986, %v4989 : tensor<64x128x28x28xf32>
    %v4995 = stablehlo.reduce(%v4994 init: %v4990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4996 = stablehlo.divide %v4995, %v4991 : tensor<128xf32>
    %v4997 = stablehlo.concatenate %v4993, %v4996, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4998 = stablehlo.concatenate %v1020, %v4997, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b3g2dst = "stablehlo.all_reduce"(%v4998) ({
    ^bb0(%aras2b3g2dst: tensor<f32>, %arbs2b3g2dst: tensor<f32>):
      %aradds2b3g2dst = stablehlo.add %aras2b3g2dst, %arbs2b3g2dst : tensor<f32>
      stablehlo.return %aradds2b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g2dst = stablehlo.divide %arsums2b3g2dst, %arns2b3g2dst : tensor<512xf32>
    %v4999 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5000 = stablehlo.slice %armeans2b3g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5001 = stablehlo.slice %armeans2b3g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5002 = stablehlo.slice %armeans2b3g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5003 = stablehlo.slice %armeans2b3g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5004 = stablehlo.broadcast_in_dim %v5000, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5005 = stablehlo.broadcast_in_dim %v5001, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5006 = stablehlo.broadcast_in_dim %v5002, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5007 = stablehlo.broadcast_in_dim %v5003, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5008 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5009 = stablehlo.add %v5005, %v5008 : tensor<64x128x28x28xf32>
    %v5010 = stablehlo.rsqrt %v5009 : tensor<64x128x28x28xf32>
    %v5011 = stablehlo.subtract %v4999, %v5004 : tensor<64x128x28x28xf32>
    %v5012 = stablehlo.multiply %v5011, %v5010 : tensor<64x128x28x28xf32>
    %v5013 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5014 = stablehlo.reshape %v4976 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5015 = stablehlo.multiply %v5013, %v5014 : tensor<64x128x28x28xf32>
    %v5016 = stablehlo.subtract %v5015, %v5006 : tensor<64x128x28x28xf32>
    %v5017 = stablehlo.multiply %v5012, %v5007 : tensor<64x128x28x28xf32>
    %v5018 = stablehlo.subtract %v5016, %v5017 : tensor<64x128x28x28xf32>
    %v5019 = stablehlo.multiply %v5010, %v5018 : tensor<64x128x28x28xf32>
    %v5020 = stablehlo.reshape %v5019 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5021 = stablehlo.reshape %v5020 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5022 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5023 = stablehlo.transpose %v5022, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5024 = stablehlo.convert %v5021 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5025 = stablehlo.convert %v5023 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v5026 = stablehlo.convolution(%v5024, %v5025)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v5027 = stablehlo.convert %v5026 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5028 = stablehlo.reshape %v5027 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5029 = stablehlo.reshape %v5028 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5030 = stablehlo.reshape %v991 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5031 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5032 = stablehlo.compare GT, %v5030, %v5031 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5033 = stablehlo.select %v5032, %v5029, %v5031 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5034 = stablehlo.reshape %v5033 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5035 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5036 = stablehlo.slice %v976 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5037 = stablehlo.slice %v976 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5038 = stablehlo.broadcast_in_dim %v5036, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5039 = stablehlo.broadcast_in_dim %v5037, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5040 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5041 = stablehlo.add %v5039, %v5040 : tensor<64x128x28x28xf32>
    %v5042 = stablehlo.rsqrt %v5041 : tensor<64x128x28x28xf32>
    %v5043 = stablehlo.subtract %v5035, %v5038 : tensor<64x128x28x28xf32>
    %v5044 = stablehlo.multiply %v5043, %v5042 : tensor<64x128x28x28xf32>
    %v5045 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5046 = stablehlo.reshape %v5034 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5047 = stablehlo.multiply %v5045, %v5046 : tensor<64x128x28x28xf32>
    %v5048 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5049 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5050 = stablehlo.reduce(%v5047 init: %v5048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5051 = stablehlo.divide %v5050, %v5049 : tensor<128xf32>
    %v5052 = stablehlo.multiply %v5044, %v5047 : tensor<64x128x28x28xf32>
    %v5053 = stablehlo.reduce(%v5052 init: %v5048) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5054 = stablehlo.divide %v5053, %v5049 : tensor<128xf32>
    %v5055 = stablehlo.concatenate %v5051, %v5054, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5056 = stablehlo.concatenate %v976, %v5055, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b3g1dst = "stablehlo.all_reduce"(%v5056) ({
    ^bb0(%aras2b3g1dst: tensor<f32>, %arbs2b3g1dst: tensor<f32>):
      %aradds2b3g1dst = stablehlo.add %aras2b3g1dst, %arbs2b3g1dst : tensor<f32>
      stablehlo.return %aradds2b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g1dst = stablehlo.divide %arsums2b3g1dst, %arns2b3g1dst : tensor<512xf32>
    %v5057 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5058 = stablehlo.slice %armeans2b3g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5059 = stablehlo.slice %armeans2b3g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5060 = stablehlo.slice %armeans2b3g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5061 = stablehlo.slice %armeans2b3g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5062 = stablehlo.broadcast_in_dim %v5058, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5063 = stablehlo.broadcast_in_dim %v5059, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5064 = stablehlo.broadcast_in_dim %v5060, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5065 = stablehlo.broadcast_in_dim %v5061, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5066 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5067 = stablehlo.add %v5063, %v5066 : tensor<64x128x28x28xf32>
    %v5068 = stablehlo.rsqrt %v5067 : tensor<64x128x28x28xf32>
    %v5069 = stablehlo.subtract %v5057, %v5062 : tensor<64x128x28x28xf32>
    %v5070 = stablehlo.multiply %v5069, %v5068 : tensor<64x128x28x28xf32>
    %v5071 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5072 = stablehlo.reshape %v5034 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5073 = stablehlo.multiply %v5071, %v5072 : tensor<64x128x28x28xf32>
    %v5074 = stablehlo.subtract %v5073, %v5064 : tensor<64x128x28x28xf32>
    %v5075 = stablehlo.multiply %v5070, %v5065 : tensor<64x128x28x28xf32>
    %v5076 = stablehlo.subtract %v5074, %v5075 : tensor<64x128x28x28xf32>
    %v5077 = stablehlo.multiply %v5068, %v5076 : tensor<64x128x28x28xf32>
    %v5078 = stablehlo.reshape %v5077 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5079 = stablehlo.reshape %v5078 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5080 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v5081 = stablehlo.transpose %v5080, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5082 = stablehlo.convert %v5079 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5083 = stablehlo.convert %v5081 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v5084 = stablehlo.convolution(%v5082, %v5083)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v5085 = stablehlo.convert %v5084 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v5086 = stablehlo.reshape %v5085 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5087 = stablehlo.reshape %v5086 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5088 = stablehlo.reshape %v4918 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5089 = stablehlo.add %v5087, %v5088 : tensor<64x512x28x28xf32>
    %v5090 = stablehlo.reshape %v5089 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5091 = stablehlo.reshape %v949 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5092 = stablehlo.reshape %v5078 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5093 = stablehlo.transpose %v5091, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5094 = stablehlo.transpose %v5092, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5095 = stablehlo.convert %v5093 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5096 = stablehlo.convert %v5094 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5097 = stablehlo.convolution(%v5095, %v5096)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v5098 = stablehlo.convert %v5097 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v5099 = stablehlo.transpose %v5098, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5100 = stablehlo.reshape %v957 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5101 = stablehlo.slice %v976 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5102 = stablehlo.slice %v976 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5103 = stablehlo.broadcast_in_dim %v5101, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5104 = stablehlo.broadcast_in_dim %v5102, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5105 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5106 = stablehlo.add %v5104, %v5105 : tensor<64x128x28x28xf32>
    %v5107 = stablehlo.rsqrt %v5106 : tensor<64x128x28x28xf32>
    %v5108 = stablehlo.subtract %v5100, %v5103 : tensor<64x128x28x28xf32>
    %v5109 = stablehlo.multiply %v5108, %v5107 : tensor<64x128x28x28xf32>
    %v5110 = stablehlo.reshape %v5034 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5111 = stablehlo.multiply %v5110, %v5109 : tensor<64x128x28x28xf32>
    %v5112 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5113 = stablehlo.reduce(%v5111 init: %v5112) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5114 = stablehlo.reshape %v5034 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5116 = stablehlo.reduce(%v5114 init: %v5115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5117 = stablehlo.reshape %v993 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5118 = stablehlo.reshape %v5020 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5119 = stablehlo.transpose %v5117, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5120 = stablehlo.transpose %v5118, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5121 = stablehlo.convert %v5119 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5122 = stablehlo.convert %v5120 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5123 = stablehlo.convolution(%v5121, %v5122)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v5124 = stablehlo.convert %v5123 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v5125 = stablehlo.transpose %v5124, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5126 = stablehlo.reshape %v1001 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5127 = stablehlo.slice %v1020 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5128 = stablehlo.slice %v1020 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5129 = stablehlo.broadcast_in_dim %v5127, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5130 = stablehlo.broadcast_in_dim %v5128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5131 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5132 = stablehlo.add %v5130, %v5131 : tensor<64x128x28x28xf32>
    %v5133 = stablehlo.rsqrt %v5132 : tensor<64x128x28x28xf32>
    %v5134 = stablehlo.subtract %v5126, %v5129 : tensor<64x128x28x28xf32>
    %v5135 = stablehlo.multiply %v5134, %v5133 : tensor<64x128x28x28xf32>
    %v5136 = stablehlo.reshape %v4976 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5137 = stablehlo.multiply %v5136, %v5135 : tensor<64x128x28x28xf32>
    %v5138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5139 = stablehlo.reduce(%v5137 init: %v5138) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5140 = stablehlo.reshape %v4976 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5141 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5142 = stablehlo.reduce(%v5140 init: %v5141) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5143 = stablehlo.reshape %v1037 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5144 = stablehlo.reshape %v4962 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5145 = stablehlo.transpose %v5143, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5146 = stablehlo.transpose %v5144, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5147 = stablehlo.convert %v5145 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5148 = stablehlo.convert %v5146 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5149 = stablehlo.convolution(%v5147, %v5148)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v5150 = stablehlo.convert %v5149 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v5151 = stablehlo.transpose %v5150, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5152 = stablehlo.reshape %v1045 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5153 = stablehlo.slice %v1064 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5154 = stablehlo.slice %v1064 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5155 = stablehlo.broadcast_in_dim %v5153, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5156 = stablehlo.broadcast_in_dim %v5154, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5157 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5158 = stablehlo.add %v5156, %v5157 : tensor<64x512x28x28xf32>
    %v5159 = stablehlo.rsqrt %v5158 : tensor<64x512x28x28xf32>
    %v5160 = stablehlo.subtract %v5152, %v5155 : tensor<64x512x28x28xf32>
    %v5161 = stablehlo.multiply %v5160, %v5159 : tensor<64x512x28x28xf32>
    %v5162 = stablehlo.reshape %v4918 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5163 = stablehlo.multiply %v5162, %v5161 : tensor<64x512x28x28xf32>
    %v5164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5165 = stablehlo.reduce(%v5163 init: %v5164) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5166 = stablehlo.reshape %v4918 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5168 = stablehlo.reduce(%v5166 init: %v5167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5169 = stablehlo.reshape %v5090 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5170 = stablehlo.reshape %v945 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5171 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v5172 = stablehlo.compare GT, %v5170, %v5171 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v5173 = stablehlo.select %v5172, %v5169, %v5171 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v5174 = stablehlo.reshape %v5173 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5175 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5176 = stablehlo.slice %v926 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5177 = stablehlo.slice %v926 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5178 = stablehlo.broadcast_in_dim %v5176, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5179 = stablehlo.broadcast_in_dim %v5177, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5180 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5181 = stablehlo.add %v5179, %v5180 : tensor<64x512x28x28xf32>
    %v5182 = stablehlo.rsqrt %v5181 : tensor<64x512x28x28xf32>
    %v5183 = stablehlo.subtract %v5175, %v5178 : tensor<64x512x28x28xf32>
    %v5184 = stablehlo.multiply %v5183, %v5182 : tensor<64x512x28x28xf32>
    %v5185 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5186 = stablehlo.reshape %v5174 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5187 = stablehlo.multiply %v5185, %v5186 : tensor<64x512x28x28xf32>
    %v5188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5189 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5190 = stablehlo.reduce(%v5187 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5191 = stablehlo.divide %v5190, %v5189 : tensor<512xf32>
    %v5192 = stablehlo.multiply %v5184, %v5187 : tensor<64x512x28x28xf32>
    %v5193 = stablehlo.reduce(%v5192 init: %v5188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5194 = stablehlo.divide %v5193, %v5189 : tensor<512xf32>
    %v5195 = stablehlo.concatenate %v5191, %v5194, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5196 = stablehlo.concatenate %v926, %v5195, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b2g3dst = "stablehlo.all_reduce"(%v5196) ({
    ^bb0(%aras2b2g3dst: tensor<f32>, %arbs2b2g3dst: tensor<f32>):
      %aradds2b2g3dst = stablehlo.add %aras2b2g3dst, %arbs2b2g3dst : tensor<f32>
      stablehlo.return %aradds2b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b2g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b2g3dst = stablehlo.divide %arsums2b2g3dst, %arns2b2g3dst : tensor<2048xf32>
    %v5197 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5198 = stablehlo.slice %armeans2b2g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5199 = stablehlo.slice %armeans2b2g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5200 = stablehlo.slice %armeans2b2g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5201 = stablehlo.slice %armeans2b2g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5202 = stablehlo.broadcast_in_dim %v5198, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5203 = stablehlo.broadcast_in_dim %v5199, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5204 = stablehlo.broadcast_in_dim %v5200, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5205 = stablehlo.broadcast_in_dim %v5201, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5206 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5207 = stablehlo.add %v5203, %v5206 : tensor<64x512x28x28xf32>
    %v5208 = stablehlo.rsqrt %v5207 : tensor<64x512x28x28xf32>
    %v5209 = stablehlo.subtract %v5197, %v5202 : tensor<64x512x28x28xf32>
    %v5210 = stablehlo.multiply %v5209, %v5208 : tensor<64x512x28x28xf32>
    %v5211 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5212 = stablehlo.reshape %v5174 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5213 = stablehlo.multiply %v5211, %v5212 : tensor<64x512x28x28xf32>
    %v5214 = stablehlo.subtract %v5213, %v5204 : tensor<64x512x28x28xf32>
    %v5215 = stablehlo.multiply %v5210, %v5205 : tensor<64x512x28x28xf32>
    %v5216 = stablehlo.subtract %v5214, %v5215 : tensor<64x512x28x28xf32>
    %v5217 = stablehlo.multiply %v5208, %v5216 : tensor<64x512x28x28xf32>
    %v5218 = stablehlo.reshape %v5217 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5219 = stablehlo.reshape %v5218 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5220 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v5221 = stablehlo.transpose %v5220, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5222 = stablehlo.convert %v5219 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v5223 = stablehlo.convert %v5221 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v5224 = stablehlo.convolution(%v5222, %v5223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v5225 = stablehlo.convert %v5224 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5227 = stablehlo.reshape %v5226 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5228 = stablehlo.reshape %v897 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5229 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5230 = stablehlo.compare GT, %v5228, %v5229 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5231 = stablehlo.select %v5230, %v5227, %v5229 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5232 = stablehlo.reshape %v5231 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5233 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5234 = stablehlo.slice %v882 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5235 = stablehlo.slice %v882 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5236 = stablehlo.broadcast_in_dim %v5234, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5237 = stablehlo.broadcast_in_dim %v5235, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5238 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5239 = stablehlo.add %v5237, %v5238 : tensor<64x128x28x28xf32>
    %v5240 = stablehlo.rsqrt %v5239 : tensor<64x128x28x28xf32>
    %v5241 = stablehlo.subtract %v5233, %v5236 : tensor<64x128x28x28xf32>
    %v5242 = stablehlo.multiply %v5241, %v5240 : tensor<64x128x28x28xf32>
    %v5243 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5244 = stablehlo.reshape %v5232 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5245 = stablehlo.multiply %v5243, %v5244 : tensor<64x128x28x28xf32>
    %v5246 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5247 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5248 = stablehlo.reduce(%v5245 init: %v5246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5249 = stablehlo.divide %v5248, %v5247 : tensor<128xf32>
    %v5250 = stablehlo.multiply %v5242, %v5245 : tensor<64x128x28x28xf32>
    %v5251 = stablehlo.reduce(%v5250 init: %v5246) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5252 = stablehlo.divide %v5251, %v5247 : tensor<128xf32>
    %v5253 = stablehlo.concatenate %v5249, %v5252, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5254 = stablehlo.concatenate %v882, %v5253, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g2dst = "stablehlo.all_reduce"(%v5254) ({
    ^bb0(%aras2b2g2dst: tensor<f32>, %arbs2b2g2dst: tensor<f32>):
      %aradds2b2g2dst = stablehlo.add %aras2b2g2dst, %arbs2b2g2dst : tensor<f32>
      stablehlo.return %aradds2b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g2dst = stablehlo.divide %arsums2b2g2dst, %arns2b2g2dst : tensor<512xf32>
    %v5255 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5256 = stablehlo.slice %armeans2b2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5257 = stablehlo.slice %armeans2b2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5258 = stablehlo.slice %armeans2b2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5259 = stablehlo.slice %armeans2b2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5260 = stablehlo.broadcast_in_dim %v5256, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5261 = stablehlo.broadcast_in_dim %v5257, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5262 = stablehlo.broadcast_in_dim %v5258, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5263 = stablehlo.broadcast_in_dim %v5259, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5264 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5265 = stablehlo.add %v5261, %v5264 : tensor<64x128x28x28xf32>
    %v5266 = stablehlo.rsqrt %v5265 : tensor<64x128x28x28xf32>
    %v5267 = stablehlo.subtract %v5255, %v5260 : tensor<64x128x28x28xf32>
    %v5268 = stablehlo.multiply %v5267, %v5266 : tensor<64x128x28x28xf32>
    %v5269 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5270 = stablehlo.reshape %v5232 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5271 = stablehlo.multiply %v5269, %v5270 : tensor<64x128x28x28xf32>
    %v5272 = stablehlo.subtract %v5271, %v5262 : tensor<64x128x28x28xf32>
    %v5273 = stablehlo.multiply %v5268, %v5263 : tensor<64x128x28x28xf32>
    %v5274 = stablehlo.subtract %v5272, %v5273 : tensor<64x128x28x28xf32>
    %v5275 = stablehlo.multiply %v5266, %v5274 : tensor<64x128x28x28xf32>
    %v5276 = stablehlo.reshape %v5275 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5277 = stablehlo.reshape %v5276 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5278 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5279 = stablehlo.transpose %v5278, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5280 = stablehlo.convert %v5277 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5281 = stablehlo.convert %v5279 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v5282 = stablehlo.convolution(%v5280, %v5281)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v5283 = stablehlo.convert %v5282 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5284 = stablehlo.reshape %v5283 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5285 = stablehlo.reshape %v5284 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5286 = stablehlo.reshape %v853 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5287 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5288 = stablehlo.compare GT, %v5286, %v5287 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5289 = stablehlo.select %v5288, %v5285, %v5287 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5290 = stablehlo.reshape %v5289 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5291 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5292 = stablehlo.slice %v838 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5293 = stablehlo.slice %v838 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5294 = stablehlo.broadcast_in_dim %v5292, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5295 = stablehlo.broadcast_in_dim %v5293, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5296 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5297 = stablehlo.add %v5295, %v5296 : tensor<64x128x28x28xf32>
    %v5298 = stablehlo.rsqrt %v5297 : tensor<64x128x28x28xf32>
    %v5299 = stablehlo.subtract %v5291, %v5294 : tensor<64x128x28x28xf32>
    %v5300 = stablehlo.multiply %v5299, %v5298 : tensor<64x128x28x28xf32>
    %v5301 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5302 = stablehlo.reshape %v5290 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5303 = stablehlo.multiply %v5301, %v5302 : tensor<64x128x28x28xf32>
    %v5304 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5305 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5306 = stablehlo.reduce(%v5303 init: %v5304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5307 = stablehlo.divide %v5306, %v5305 : tensor<128xf32>
    %v5308 = stablehlo.multiply %v5300, %v5303 : tensor<64x128x28x28xf32>
    %v5309 = stablehlo.reduce(%v5308 init: %v5304) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5310 = stablehlo.divide %v5309, %v5305 : tensor<128xf32>
    %v5311 = stablehlo.concatenate %v5307, %v5310, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5312 = stablehlo.concatenate %v838, %v5311, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g1dst = "stablehlo.all_reduce"(%v5312) ({
    ^bb0(%aras2b2g1dst: tensor<f32>, %arbs2b2g1dst: tensor<f32>):
      %aradds2b2g1dst = stablehlo.add %aras2b2g1dst, %arbs2b2g1dst : tensor<f32>
      stablehlo.return %aradds2b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g1dst = stablehlo.divide %arsums2b2g1dst, %arns2b2g1dst : tensor<512xf32>
    %v5313 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5314 = stablehlo.slice %armeans2b2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5315 = stablehlo.slice %armeans2b2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5316 = stablehlo.slice %armeans2b2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5317 = stablehlo.slice %armeans2b2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5318 = stablehlo.broadcast_in_dim %v5314, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5319 = stablehlo.broadcast_in_dim %v5315, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5320 = stablehlo.broadcast_in_dim %v5316, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5321 = stablehlo.broadcast_in_dim %v5317, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5322 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5323 = stablehlo.add %v5319, %v5322 : tensor<64x128x28x28xf32>
    %v5324 = stablehlo.rsqrt %v5323 : tensor<64x128x28x28xf32>
    %v5325 = stablehlo.subtract %v5313, %v5318 : tensor<64x128x28x28xf32>
    %v5326 = stablehlo.multiply %v5325, %v5324 : tensor<64x128x28x28xf32>
    %v5327 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5328 = stablehlo.reshape %v5290 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5329 = stablehlo.multiply %v5327, %v5328 : tensor<64x128x28x28xf32>
    %v5330 = stablehlo.subtract %v5329, %v5320 : tensor<64x128x28x28xf32>
    %v5331 = stablehlo.multiply %v5326, %v5321 : tensor<64x128x28x28xf32>
    %v5332 = stablehlo.subtract %v5330, %v5331 : tensor<64x128x28x28xf32>
    %v5333 = stablehlo.multiply %v5324, %v5332 : tensor<64x128x28x28xf32>
    %v5334 = stablehlo.reshape %v5333 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5335 = stablehlo.reshape %v5334 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5336 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v5337 = stablehlo.transpose %v5336, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5338 = stablehlo.convert %v5335 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5339 = stablehlo.convert %v5337 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v5340 = stablehlo.convolution(%v5338, %v5339)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v5341 = stablehlo.convert %v5340 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v5342 = stablehlo.reshape %v5341 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5343 = stablehlo.reshape %v5342 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5344 = stablehlo.reshape %v5174 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5345 = stablehlo.add %v5343, %v5344 : tensor<64x512x28x28xf32>
    %v5346 = stablehlo.reshape %v5345 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5347 = stablehlo.reshape %v811 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5348 = stablehlo.reshape %v5334 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5349 = stablehlo.transpose %v5347, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5350 = stablehlo.transpose %v5348, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5351 = stablehlo.convert %v5349 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5352 = stablehlo.convert %v5350 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5353 = stablehlo.convolution(%v5351, %v5352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v5354 = stablehlo.convert %v5353 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v5355 = stablehlo.transpose %v5354, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5356 = stablehlo.reshape %v819 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5357 = stablehlo.slice %v838 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5358 = stablehlo.slice %v838 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5359 = stablehlo.broadcast_in_dim %v5357, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5360 = stablehlo.broadcast_in_dim %v5358, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5361 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5362 = stablehlo.add %v5360, %v5361 : tensor<64x128x28x28xf32>
    %v5363 = stablehlo.rsqrt %v5362 : tensor<64x128x28x28xf32>
    %v5364 = stablehlo.subtract %v5356, %v5359 : tensor<64x128x28x28xf32>
    %v5365 = stablehlo.multiply %v5364, %v5363 : tensor<64x128x28x28xf32>
    %v5366 = stablehlo.reshape %v5290 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5367 = stablehlo.multiply %v5366, %v5365 : tensor<64x128x28x28xf32>
    %v5368 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5369 = stablehlo.reduce(%v5367 init: %v5368) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5370 = stablehlo.reshape %v5290 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5371 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5372 = stablehlo.reduce(%v5370 init: %v5371) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5373 = stablehlo.reshape %v855 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5374 = stablehlo.reshape %v5276 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5375 = stablehlo.transpose %v5373, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5376 = stablehlo.transpose %v5374, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5377 = stablehlo.convert %v5375 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5378 = stablehlo.convert %v5376 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5379 = stablehlo.convolution(%v5377, %v5378)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v5380 = stablehlo.convert %v5379 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v5381 = stablehlo.transpose %v5380, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5382 = stablehlo.reshape %v863 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5383 = stablehlo.slice %v882 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5384 = stablehlo.slice %v882 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5385 = stablehlo.broadcast_in_dim %v5383, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5386 = stablehlo.broadcast_in_dim %v5384, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5387 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5388 = stablehlo.add %v5386, %v5387 : tensor<64x128x28x28xf32>
    %v5389 = stablehlo.rsqrt %v5388 : tensor<64x128x28x28xf32>
    %v5390 = stablehlo.subtract %v5382, %v5385 : tensor<64x128x28x28xf32>
    %v5391 = stablehlo.multiply %v5390, %v5389 : tensor<64x128x28x28xf32>
    %v5392 = stablehlo.reshape %v5232 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5393 = stablehlo.multiply %v5392, %v5391 : tensor<64x128x28x28xf32>
    %v5394 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5395 = stablehlo.reduce(%v5393 init: %v5394) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5396 = stablehlo.reshape %v5232 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5398 = stablehlo.reduce(%v5396 init: %v5397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5399 = stablehlo.reshape %v899 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5400 = stablehlo.reshape %v5218 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5401 = stablehlo.transpose %v5399, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5402 = stablehlo.transpose %v5400, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5403 = stablehlo.convert %v5401 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5404 = stablehlo.convert %v5402 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5405 = stablehlo.convolution(%v5403, %v5404)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v5406 = stablehlo.convert %v5405 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v5407 = stablehlo.transpose %v5406, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5408 = stablehlo.reshape %v907 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5409 = stablehlo.slice %v926 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5410 = stablehlo.slice %v926 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5411 = stablehlo.broadcast_in_dim %v5409, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5412 = stablehlo.broadcast_in_dim %v5410, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5413 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5414 = stablehlo.add %v5412, %v5413 : tensor<64x512x28x28xf32>
    %v5415 = stablehlo.rsqrt %v5414 : tensor<64x512x28x28xf32>
    %v5416 = stablehlo.subtract %v5408, %v5411 : tensor<64x512x28x28xf32>
    %v5417 = stablehlo.multiply %v5416, %v5415 : tensor<64x512x28x28xf32>
    %v5418 = stablehlo.reshape %v5174 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5419 = stablehlo.multiply %v5418, %v5417 : tensor<64x512x28x28xf32>
    %v5420 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5421 = stablehlo.reduce(%v5419 init: %v5420) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5422 = stablehlo.reshape %v5174 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5424 = stablehlo.reduce(%v5422 init: %v5423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5425 = stablehlo.reshape %v5346 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5426 = stablehlo.reshape %v807 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5427 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v5428 = stablehlo.compare GT, %v5426, %v5427 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v5429 = stablehlo.select %v5428, %v5425, %v5427 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v5430 = stablehlo.reshape %v5429 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5431 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5432 = stablehlo.slice %v788 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5433 = stablehlo.slice %v788 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5434 = stablehlo.broadcast_in_dim %v5432, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5435 = stablehlo.broadcast_in_dim %v5433, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5436 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5437 = stablehlo.add %v5435, %v5436 : tensor<64x512x28x28xf32>
    %v5438 = stablehlo.rsqrt %v5437 : tensor<64x512x28x28xf32>
    %v5439 = stablehlo.subtract %v5431, %v5434 : tensor<64x512x28x28xf32>
    %v5440 = stablehlo.multiply %v5439, %v5438 : tensor<64x512x28x28xf32>
    %v5441 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5442 = stablehlo.reshape %v5430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5443 = stablehlo.multiply %v5441, %v5442 : tensor<64x512x28x28xf32>
    %v5444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5445 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5446 = stablehlo.reduce(%v5443 init: %v5444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5447 = stablehlo.divide %v5446, %v5445 : tensor<512xf32>
    %v5448 = stablehlo.multiply %v5440, %v5443 : tensor<64x512x28x28xf32>
    %v5449 = stablehlo.reduce(%v5448 init: %v5444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5450 = stablehlo.divide %v5449, %v5445 : tensor<512xf32>
    %v5451 = stablehlo.concatenate %v5447, %v5450, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5452 = stablehlo.concatenate %v788, %v5451, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b1g3dst = "stablehlo.all_reduce"(%v5452) ({
    ^bb0(%aras2b1g3dst: tensor<f32>, %arbs2b1g3dst: tensor<f32>):
      %aradds2b1g3dst = stablehlo.add %aras2b1g3dst, %arbs2b1g3dst : tensor<f32>
      stablehlo.return %aradds2b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b1g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b1g3dst = stablehlo.divide %arsums2b1g3dst, %arns2b1g3dst : tensor<2048xf32>
    %v5453 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5454 = stablehlo.slice %armeans2b1g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5455 = stablehlo.slice %armeans2b1g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5456 = stablehlo.slice %armeans2b1g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5457 = stablehlo.slice %armeans2b1g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5458 = stablehlo.broadcast_in_dim %v5454, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5459 = stablehlo.broadcast_in_dim %v5455, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5460 = stablehlo.broadcast_in_dim %v5456, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5461 = stablehlo.broadcast_in_dim %v5457, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5462 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5463 = stablehlo.add %v5459, %v5462 : tensor<64x512x28x28xf32>
    %v5464 = stablehlo.rsqrt %v5463 : tensor<64x512x28x28xf32>
    %v5465 = stablehlo.subtract %v5453, %v5458 : tensor<64x512x28x28xf32>
    %v5466 = stablehlo.multiply %v5465, %v5464 : tensor<64x512x28x28xf32>
    %v5467 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5468 = stablehlo.reshape %v5430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5469 = stablehlo.multiply %v5467, %v5468 : tensor<64x512x28x28xf32>
    %v5470 = stablehlo.subtract %v5469, %v5460 : tensor<64x512x28x28xf32>
    %v5471 = stablehlo.multiply %v5466, %v5461 : tensor<64x512x28x28xf32>
    %v5472 = stablehlo.subtract %v5470, %v5471 : tensor<64x512x28x28xf32>
    %v5473 = stablehlo.multiply %v5464, %v5472 : tensor<64x512x28x28xf32>
    %v5474 = stablehlo.reshape %v5473 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5475 = stablehlo.reshape %v5474 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5476 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v5477 = stablehlo.transpose %v5476, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5478 = stablehlo.convert %v5475 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v5479 = stablehlo.convert %v5477 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v5480 = stablehlo.convolution(%v5478, %v5479)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v5481 = stablehlo.convert %v5480 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5482 = stablehlo.reshape %v5481 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5483 = stablehlo.reshape %v5482 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5484 = stablehlo.reshape %v759 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5485 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5486 = stablehlo.compare GT, %v5484, %v5485 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5487 = stablehlo.select %v5486, %v5483, %v5485 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5488 = stablehlo.reshape %v5487 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5489 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5490 = stablehlo.slice %v744 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5491 = stablehlo.slice %v744 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5492 = stablehlo.broadcast_in_dim %v5490, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5493 = stablehlo.broadcast_in_dim %v5491, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5494 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5495 = stablehlo.add %v5493, %v5494 : tensor<64x128x28x28xf32>
    %v5496 = stablehlo.rsqrt %v5495 : tensor<64x128x28x28xf32>
    %v5497 = stablehlo.subtract %v5489, %v5492 : tensor<64x128x28x28xf32>
    %v5498 = stablehlo.multiply %v5497, %v5496 : tensor<64x128x28x28xf32>
    %v5499 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5500 = stablehlo.reshape %v5488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5501 = stablehlo.multiply %v5499, %v5500 : tensor<64x128x28x28xf32>
    %v5502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5503 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5504 = stablehlo.reduce(%v5501 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5505 = stablehlo.divide %v5504, %v5503 : tensor<128xf32>
    %v5506 = stablehlo.multiply %v5498, %v5501 : tensor<64x128x28x28xf32>
    %v5507 = stablehlo.reduce(%v5506 init: %v5502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5508 = stablehlo.divide %v5507, %v5503 : tensor<128xf32>
    %v5509 = stablehlo.concatenate %v5505, %v5508, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5510 = stablehlo.concatenate %v744, %v5509, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g2dst = "stablehlo.all_reduce"(%v5510) ({
    ^bb0(%aras2b1g2dst: tensor<f32>, %arbs2b1g2dst: tensor<f32>):
      %aradds2b1g2dst = stablehlo.add %aras2b1g2dst, %arbs2b1g2dst : tensor<f32>
      stablehlo.return %aradds2b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g2dst = stablehlo.divide %arsums2b1g2dst, %arns2b1g2dst : tensor<512xf32>
    %v5511 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5512 = stablehlo.slice %armeans2b1g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5513 = stablehlo.slice %armeans2b1g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5514 = stablehlo.slice %armeans2b1g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5515 = stablehlo.slice %armeans2b1g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5516 = stablehlo.broadcast_in_dim %v5512, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5517 = stablehlo.broadcast_in_dim %v5513, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5518 = stablehlo.broadcast_in_dim %v5514, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5519 = stablehlo.broadcast_in_dim %v5515, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5520 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5521 = stablehlo.add %v5517, %v5520 : tensor<64x128x28x28xf32>
    %v5522 = stablehlo.rsqrt %v5521 : tensor<64x128x28x28xf32>
    %v5523 = stablehlo.subtract %v5511, %v5516 : tensor<64x128x28x28xf32>
    %v5524 = stablehlo.multiply %v5523, %v5522 : tensor<64x128x28x28xf32>
    %v5525 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5526 = stablehlo.reshape %v5488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5527 = stablehlo.multiply %v5525, %v5526 : tensor<64x128x28x28xf32>
    %v5528 = stablehlo.subtract %v5527, %v5518 : tensor<64x128x28x28xf32>
    %v5529 = stablehlo.multiply %v5524, %v5519 : tensor<64x128x28x28xf32>
    %v5530 = stablehlo.subtract %v5528, %v5529 : tensor<64x128x28x28xf32>
    %v5531 = stablehlo.multiply %v5522, %v5530 : tensor<64x128x28x28xf32>
    %v5532 = stablehlo.reshape %v5531 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5533 = stablehlo.reshape %v5532 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5534 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5535 = stablehlo.transpose %v5534, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5536 = stablehlo.convert %v5533 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5537 = stablehlo.convert %v5535 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v5538 = stablehlo.convolution(%v5536, %v5537)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v5539 = stablehlo.convert %v5538 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5540 = stablehlo.reshape %v5539 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5541 = stablehlo.reshape %v5540 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5542 = stablehlo.reshape %v715 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5543 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5544 = stablehlo.compare GT, %v5542, %v5543 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5545 = stablehlo.select %v5544, %v5541, %v5543 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5546 = stablehlo.reshape %v5545 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5547 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5548 = stablehlo.slice %v700 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5549 = stablehlo.slice %v700 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5550 = stablehlo.broadcast_in_dim %v5548, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5551 = stablehlo.broadcast_in_dim %v5549, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5552 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5553 = stablehlo.add %v5551, %v5552 : tensor<64x128x28x28xf32>
    %v5554 = stablehlo.rsqrt %v5553 : tensor<64x128x28x28xf32>
    %v5555 = stablehlo.subtract %v5547, %v5550 : tensor<64x128x28x28xf32>
    %v5556 = stablehlo.multiply %v5555, %v5554 : tensor<64x128x28x28xf32>
    %v5557 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5558 = stablehlo.reshape %v5546 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5559 = stablehlo.multiply %v5557, %v5558 : tensor<64x128x28x28xf32>
    %v5560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5561 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5562 = stablehlo.reduce(%v5559 init: %v5560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5563 = stablehlo.divide %v5562, %v5561 : tensor<128xf32>
    %v5564 = stablehlo.multiply %v5556, %v5559 : tensor<64x128x28x28xf32>
    %v5565 = stablehlo.reduce(%v5564 init: %v5560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5566 = stablehlo.divide %v5565, %v5561 : tensor<128xf32>
    %v5567 = stablehlo.concatenate %v5563, %v5566, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5568 = stablehlo.concatenate %v700, %v5567, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g1dst = "stablehlo.all_reduce"(%v5568) ({
    ^bb0(%aras2b1g1dst: tensor<f32>, %arbs2b1g1dst: tensor<f32>):
      %aradds2b1g1dst = stablehlo.add %aras2b1g1dst, %arbs2b1g1dst : tensor<f32>
      stablehlo.return %aradds2b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g1dst = stablehlo.divide %arsums2b1g1dst, %arns2b1g1dst : tensor<512xf32>
    %v5569 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5570 = stablehlo.slice %armeans2b1g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5571 = stablehlo.slice %armeans2b1g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5572 = stablehlo.slice %armeans2b1g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5573 = stablehlo.slice %armeans2b1g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5574 = stablehlo.broadcast_in_dim %v5570, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5575 = stablehlo.broadcast_in_dim %v5571, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5576 = stablehlo.broadcast_in_dim %v5572, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5577 = stablehlo.broadcast_in_dim %v5573, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5578 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5579 = stablehlo.add %v5575, %v5578 : tensor<64x128x28x28xf32>
    %v5580 = stablehlo.rsqrt %v5579 : tensor<64x128x28x28xf32>
    %v5581 = stablehlo.subtract %v5569, %v5574 : tensor<64x128x28x28xf32>
    %v5582 = stablehlo.multiply %v5581, %v5580 : tensor<64x128x28x28xf32>
    %v5583 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5584 = stablehlo.reshape %v5546 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5585 = stablehlo.multiply %v5583, %v5584 : tensor<64x128x28x28xf32>
    %v5586 = stablehlo.subtract %v5585, %v5576 : tensor<64x128x28x28xf32>
    %v5587 = stablehlo.multiply %v5582, %v5577 : tensor<64x128x28x28xf32>
    %v5588 = stablehlo.subtract %v5586, %v5587 : tensor<64x128x28x28xf32>
    %v5589 = stablehlo.multiply %v5580, %v5588 : tensor<64x128x28x28xf32>
    %v5590 = stablehlo.reshape %v5589 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5591 = stablehlo.reshape %v5590 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5592 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v5593 = stablehlo.transpose %v5592, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5594 = stablehlo.convert %v5591 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v5595 = stablehlo.convert %v5593 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v5596 = stablehlo.convolution(%v5594, %v5595)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v5597 = stablehlo.convert %v5596 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v5598 = stablehlo.reshape %v5597 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5599 = stablehlo.reshape %v5598 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5600 = stablehlo.reshape %v5430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5601 = stablehlo.add %v5599, %v5600 : tensor<64x512x28x28xf32>
    %v5602 = stablehlo.reshape %v5601 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5603 = stablehlo.reshape %v673 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5604 = stablehlo.reshape %v5590 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5605 = stablehlo.transpose %v5603, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5606 = stablehlo.transpose %v5604, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5607 = stablehlo.convert %v5605 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5608 = stablehlo.convert %v5606 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5609 = stablehlo.convolution(%v5607, %v5608)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v5610 = stablehlo.convert %v5609 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v5611 = stablehlo.transpose %v5610, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5612 = stablehlo.reshape %v681 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5613 = stablehlo.slice %v700 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5614 = stablehlo.slice %v700 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5615 = stablehlo.broadcast_in_dim %v5613, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5616 = stablehlo.broadcast_in_dim %v5614, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5617 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5618 = stablehlo.add %v5616, %v5617 : tensor<64x128x28x28xf32>
    %v5619 = stablehlo.rsqrt %v5618 : tensor<64x128x28x28xf32>
    %v5620 = stablehlo.subtract %v5612, %v5615 : tensor<64x128x28x28xf32>
    %v5621 = stablehlo.multiply %v5620, %v5619 : tensor<64x128x28x28xf32>
    %v5622 = stablehlo.reshape %v5546 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5623 = stablehlo.multiply %v5622, %v5621 : tensor<64x128x28x28xf32>
    %v5624 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5625 = stablehlo.reduce(%v5623 init: %v5624) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5626 = stablehlo.reshape %v5546 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5628 = stablehlo.reduce(%v5626 init: %v5627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5629 = stablehlo.reshape %v717 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5630 = stablehlo.reshape %v5532 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5631 = stablehlo.transpose %v5629, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5632 = stablehlo.transpose %v5630, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5633 = stablehlo.convert %v5631 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5634 = stablehlo.convert %v5632 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5635 = stablehlo.convolution(%v5633, %v5634)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v5636 = stablehlo.convert %v5635 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v5637 = stablehlo.transpose %v5636, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5638 = stablehlo.reshape %v725 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5639 = stablehlo.slice %v744 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5640 = stablehlo.slice %v744 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5641 = stablehlo.broadcast_in_dim %v5639, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5642 = stablehlo.broadcast_in_dim %v5640, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5643 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5644 = stablehlo.add %v5642, %v5643 : tensor<64x128x28x28xf32>
    %v5645 = stablehlo.rsqrt %v5644 : tensor<64x128x28x28xf32>
    %v5646 = stablehlo.subtract %v5638, %v5641 : tensor<64x128x28x28xf32>
    %v5647 = stablehlo.multiply %v5646, %v5645 : tensor<64x128x28x28xf32>
    %v5648 = stablehlo.reshape %v5488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5649 = stablehlo.multiply %v5648, %v5647 : tensor<64x128x28x28xf32>
    %v5650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5651 = stablehlo.reduce(%v5649 init: %v5650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5652 = stablehlo.reshape %v5488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5654 = stablehlo.reduce(%v5652 init: %v5653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5655 = stablehlo.reshape %v761 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5656 = stablehlo.reshape %v5474 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5657 = stablehlo.transpose %v5655, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5658 = stablehlo.transpose %v5656, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5659 = stablehlo.convert %v5657 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5660 = stablehlo.convert %v5658 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5661 = stablehlo.convolution(%v5659, %v5660)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v5662 = stablehlo.convert %v5661 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v5663 = stablehlo.transpose %v5662, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5664 = stablehlo.reshape %v769 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5665 = stablehlo.slice %v788 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5666 = stablehlo.slice %v788 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5667 = stablehlo.broadcast_in_dim %v5665, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5668 = stablehlo.broadcast_in_dim %v5666, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5669 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5670 = stablehlo.add %v5668, %v5669 : tensor<64x512x28x28xf32>
    %v5671 = stablehlo.rsqrt %v5670 : tensor<64x512x28x28xf32>
    %v5672 = stablehlo.subtract %v5664, %v5667 : tensor<64x512x28x28xf32>
    %v5673 = stablehlo.multiply %v5672, %v5671 : tensor<64x512x28x28xf32>
    %v5674 = stablehlo.reshape %v5430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5675 = stablehlo.multiply %v5674, %v5673 : tensor<64x512x28x28xf32>
    %v5676 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5677 = stablehlo.reduce(%v5675 init: %v5676) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5678 = stablehlo.reshape %v5430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5680 = stablehlo.reduce(%v5678 init: %v5679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5681 = stablehlo.reshape %v5602 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5682 = stablehlo.reshape %v671 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5683 = stablehlo.constant dense<0.0> : tensor<64x512x28x28xf32>
    %v5684 = stablehlo.compare GT, %v5682, %v5683 : (tensor<64x512x28x28xf32>, tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xi1>
    %v5685 = stablehlo.select %v5684, %v5681, %v5683 : tensor<64x512x28x28xi1>, tensor<64x512x28x28xf32>
    %v5686 = stablehlo.reshape %v5685 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5687 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5688 = stablehlo.slice %v613 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5689 = stablehlo.slice %v613 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5690 = stablehlo.broadcast_in_dim %v5688, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5691 = stablehlo.broadcast_in_dim %v5689, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5692 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5693 = stablehlo.add %v5691, %v5692 : tensor<64x512x28x28xf32>
    %v5694 = stablehlo.rsqrt %v5693 : tensor<64x512x28x28xf32>
    %v5695 = stablehlo.subtract %v5687, %v5690 : tensor<64x512x28x28xf32>
    %v5696 = stablehlo.multiply %v5695, %v5694 : tensor<64x512x28x28xf32>
    %v5697 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5698 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5699 = stablehlo.multiply %v5697, %v5698 : tensor<64x512x28x28xf32>
    %v5700 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5701 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5702 = stablehlo.reduce(%v5699 init: %v5700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5703 = stablehlo.divide %v5702, %v5701 : tensor<512xf32>
    %v5704 = stablehlo.multiply %v5696, %v5699 : tensor<64x512x28x28xf32>
    %v5705 = stablehlo.reduce(%v5704 init: %v5700) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5706 = stablehlo.divide %v5705, %v5701 : tensor<512xf32>
    %v5707 = stablehlo.concatenate %v5703, %v5706, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5708 = stablehlo.concatenate %v613, %v5707, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b0g3dst = "stablehlo.all_reduce"(%v5708) ({
    ^bb0(%aras2b0g3dst: tensor<f32>, %arbs2b0g3dst: tensor<f32>):
      %aradds2b0g3dst = stablehlo.add %aras2b0g3dst, %arbs2b0g3dst : tensor<f32>
      stablehlo.return %aradds2b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b0g3dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b0g3dst = stablehlo.divide %arsums2b0g3dst, %arns2b0g3dst : tensor<2048xf32>
    %v5709 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5710 = stablehlo.slice %armeans2b0g3dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5711 = stablehlo.slice %armeans2b0g3dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5712 = stablehlo.slice %armeans2b0g3dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5713 = stablehlo.slice %armeans2b0g3dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5714 = stablehlo.broadcast_in_dim %v5710, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5715 = stablehlo.broadcast_in_dim %v5711, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5716 = stablehlo.broadcast_in_dim %v5712, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5717 = stablehlo.broadcast_in_dim %v5713, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5718 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5719 = stablehlo.add %v5715, %v5718 : tensor<64x512x28x28xf32>
    %v5720 = stablehlo.rsqrt %v5719 : tensor<64x512x28x28xf32>
    %v5721 = stablehlo.subtract %v5709, %v5714 : tensor<64x512x28x28xf32>
    %v5722 = stablehlo.multiply %v5721, %v5720 : tensor<64x512x28x28xf32>
    %v5723 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5724 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5725 = stablehlo.multiply %v5723, %v5724 : tensor<64x512x28x28xf32>
    %v5726 = stablehlo.subtract %v5725, %v5716 : tensor<64x512x28x28xf32>
    %v5727 = stablehlo.multiply %v5722, %v5717 : tensor<64x512x28x28xf32>
    %v5728 = stablehlo.subtract %v5726, %v5727 : tensor<64x512x28x28xf32>
    %v5729 = stablehlo.multiply %v5720, %v5728 : tensor<64x512x28x28xf32>
    %v5730 = stablehlo.reshape %v5729 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5731 = stablehlo.reshape %v5730 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5732 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v5733 = stablehlo.transpose %v5732, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v5734 = stablehlo.convert %v5731 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v5735 = stablehlo.convert %v5733 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v5736 = stablehlo.convolution(%v5734, %v5735)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v5737 = stablehlo.convert %v5736 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v5738 = stablehlo.reshape %v5737 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5739 = stablehlo.reshape %v5738 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5740 = stablehlo.reshape %v584 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5741 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v5742 = stablehlo.compare GT, %v5740, %v5741 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v5743 = stablehlo.select %v5742, %v5739, %v5741 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v5744 = stablehlo.reshape %v5743 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5745 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5746 = stablehlo.slice %v569 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5747 = stablehlo.slice %v569 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5748 = stablehlo.broadcast_in_dim %v5746, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5749 = stablehlo.broadcast_in_dim %v5747, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5750 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5751 = stablehlo.add %v5749, %v5750 : tensor<64x128x28x28xf32>
    %v5752 = stablehlo.rsqrt %v5751 : tensor<64x128x28x28xf32>
    %v5753 = stablehlo.subtract %v5745, %v5748 : tensor<64x128x28x28xf32>
    %v5754 = stablehlo.multiply %v5753, %v5752 : tensor<64x128x28x28xf32>
    %v5755 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5756 = stablehlo.reshape %v5744 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5757 = stablehlo.multiply %v5755, %v5756 : tensor<64x128x28x28xf32>
    %v5758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5759 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5760 = stablehlo.reduce(%v5757 init: %v5758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5761 = stablehlo.divide %v5760, %v5759 : tensor<128xf32>
    %v5762 = stablehlo.multiply %v5754, %v5757 : tensor<64x128x28x28xf32>
    %v5763 = stablehlo.reduce(%v5762 init: %v5758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5764 = stablehlo.divide %v5763, %v5759 : tensor<128xf32>
    %v5765 = stablehlo.concatenate %v5761, %v5764, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5766 = stablehlo.concatenate %v569, %v5765, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g2dst = "stablehlo.all_reduce"(%v5766) ({
    ^bb0(%aras2b0g2dst: tensor<f32>, %arbs2b0g2dst: tensor<f32>):
      %aradds2b0g2dst = stablehlo.add %aras2b0g2dst, %arbs2b0g2dst : tensor<f32>
      stablehlo.return %aradds2b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g2dst = stablehlo.divide %arsums2b0g2dst, %arns2b0g2dst : tensor<512xf32>
    %v5767 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5768 = stablehlo.slice %armeans2b0g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5769 = stablehlo.slice %armeans2b0g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5770 = stablehlo.slice %armeans2b0g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5771 = stablehlo.slice %armeans2b0g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5772 = stablehlo.broadcast_in_dim %v5768, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5773 = stablehlo.broadcast_in_dim %v5769, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5774 = stablehlo.broadcast_in_dim %v5770, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5775 = stablehlo.broadcast_in_dim %v5771, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5776 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5777 = stablehlo.add %v5773, %v5776 : tensor<64x128x28x28xf32>
    %v5778 = stablehlo.rsqrt %v5777 : tensor<64x128x28x28xf32>
    %v5779 = stablehlo.subtract %v5767, %v5772 : tensor<64x128x28x28xf32>
    %v5780 = stablehlo.multiply %v5779, %v5778 : tensor<64x128x28x28xf32>
    %v5781 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5782 = stablehlo.reshape %v5744 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5783 = stablehlo.multiply %v5781, %v5782 : tensor<64x128x28x28xf32>
    %v5784 = stablehlo.subtract %v5783, %v5774 : tensor<64x128x28x28xf32>
    %v5785 = stablehlo.multiply %v5780, %v5775 : tensor<64x128x28x28xf32>
    %v5786 = stablehlo.subtract %v5784, %v5785 : tensor<64x128x28x28xf32>
    %v5787 = stablehlo.multiply %v5778, %v5786 : tensor<64x128x28x28xf32>
    %v5788 = stablehlo.reshape %v5787 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v5789 = stablehlo.reshape %v5788 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5790 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5791 = stablehlo.pad %v5789, %v5790, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v5792 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v5793 = stablehlo.transpose %v5792, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5794 = stablehlo.convert %v5791 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v5795 = stablehlo.convert %v5793 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v5796 = stablehlo.convolution(%v5794, %v5795)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x56x56xbf16>
    %v5797 = stablehlo.convert %v5796 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v5798 = stablehlo.reshape %v5797 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5799 = stablehlo.reshape %v5798 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5800 = stablehlo.reshape %v540 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5801 = stablehlo.constant dense<0.0> : tensor<64x128x56x56xf32>
    %v5802 = stablehlo.compare GT, %v5800, %v5801 : (tensor<64x128x56x56xf32>, tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xi1>
    %v5803 = stablehlo.select %v5802, %v5799, %v5801 : tensor<64x128x56x56xi1>, tensor<64x128x56x56xf32>
    %v5804 = stablehlo.reshape %v5803 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5805 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5806 = stablehlo.slice %v525 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5807 = stablehlo.slice %v525 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5808 = stablehlo.broadcast_in_dim %v5806, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5809 = stablehlo.broadcast_in_dim %v5807, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5810 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5811 = stablehlo.add %v5809, %v5810 : tensor<64x128x56x56xf32>
    %v5812 = stablehlo.rsqrt %v5811 : tensor<64x128x56x56xf32>
    %v5813 = stablehlo.subtract %v5805, %v5808 : tensor<64x128x56x56xf32>
    %v5814 = stablehlo.multiply %v5813, %v5812 : tensor<64x128x56x56xf32>
    %v5815 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5816 = stablehlo.reshape %v5804 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5817 = stablehlo.multiply %v5815, %v5816 : tensor<64x128x56x56xf32>
    %v5818 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5819 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5820 = stablehlo.reduce(%v5817 init: %v5818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5821 = stablehlo.divide %v5820, %v5819 : tensor<128xf32>
    %v5822 = stablehlo.multiply %v5814, %v5817 : tensor<64x128x56x56xf32>
    %v5823 = stablehlo.reduce(%v5822 init: %v5818) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5824 = stablehlo.divide %v5823, %v5819 : tensor<128xf32>
    %v5825 = stablehlo.concatenate %v5821, %v5824, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v5826 = stablehlo.concatenate %v525, %v5825, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g1dst = "stablehlo.all_reduce"(%v5826) ({
    ^bb0(%aras2b0g1dst: tensor<f32>, %arbs2b0g1dst: tensor<f32>):
      %aradds2b0g1dst = stablehlo.add %aras2b0g1dst, %arbs2b0g1dst : tensor<f32>
      stablehlo.return %aradds2b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g1dst = stablehlo.divide %arsums2b0g1dst, %arns2b0g1dst : tensor<512xf32>
    %v5827 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5828 = stablehlo.slice %armeans2b0g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v5829 = stablehlo.slice %armeans2b0g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v5830 = stablehlo.slice %armeans2b0g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v5831 = stablehlo.slice %armeans2b0g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v5832 = stablehlo.broadcast_in_dim %v5828, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5833 = stablehlo.broadcast_in_dim %v5829, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5834 = stablehlo.broadcast_in_dim %v5830, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5835 = stablehlo.broadcast_in_dim %v5831, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5836 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5837 = stablehlo.add %v5833, %v5836 : tensor<64x128x56x56xf32>
    %v5838 = stablehlo.rsqrt %v5837 : tensor<64x128x56x56xf32>
    %v5839 = stablehlo.subtract %v5827, %v5832 : tensor<64x128x56x56xf32>
    %v5840 = stablehlo.multiply %v5839, %v5838 : tensor<64x128x56x56xf32>
    %v5841 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5842 = stablehlo.reshape %v5804 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5843 = stablehlo.multiply %v5841, %v5842 : tensor<64x128x56x56xf32>
    %v5844 = stablehlo.subtract %v5843, %v5834 : tensor<64x128x56x56xf32>
    %v5845 = stablehlo.multiply %v5840, %v5835 : tensor<64x128x56x56xf32>
    %v5846 = stablehlo.subtract %v5844, %v5845 : tensor<64x128x56x56xf32>
    %v5847 = stablehlo.multiply %v5838, %v5846 : tensor<64x128x56x56xf32>
    %v5848 = stablehlo.reshape %v5847 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v5849 = stablehlo.reshape %v5848 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5850 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v5851 = stablehlo.transpose %v5850, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v5852 = stablehlo.convert %v5849 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v5853 = stablehlo.convert %v5851 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v5854 = stablehlo.convolution(%v5852, %v5853)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<256x128x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v5855 = stablehlo.convert %v5854 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v5856 = stablehlo.reshape %v5855 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5857 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5858 = stablehlo.slice %v655 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5859 = stablehlo.slice %v655 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5860 = stablehlo.broadcast_in_dim %v5858, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5861 = stablehlo.broadcast_in_dim %v5859, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5862 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5863 = stablehlo.add %v5861, %v5862 : tensor<64x512x28x28xf32>
    %v5864 = stablehlo.rsqrt %v5863 : tensor<64x512x28x28xf32>
    %v5865 = stablehlo.subtract %v5857, %v5860 : tensor<64x512x28x28xf32>
    %v5866 = stablehlo.multiply %v5865, %v5864 : tensor<64x512x28x28xf32>
    %v5867 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5868 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5869 = stablehlo.multiply %v5867, %v5868 : tensor<64x512x28x28xf32>
    %v5870 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5871 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5872 = stablehlo.reduce(%v5869 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5873 = stablehlo.divide %v5872, %v5871 : tensor<512xf32>
    %v5874 = stablehlo.multiply %v5866, %v5869 : tensor<64x512x28x28xf32>
    %v5875 = stablehlo.reduce(%v5874 init: %v5870) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5876 = stablehlo.divide %v5875, %v5871 : tensor<512xf32>
    %v5877 = stablehlo.concatenate %v5873, %v5876, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v5878 = stablehlo.concatenate %v655, %v5877, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums2b0gpdst = "stablehlo.all_reduce"(%v5878) ({
    ^bb0(%aras2b0gpdst: tensor<f32>, %arbs2b0gpdst: tensor<f32>):
      %aradds2b0gpdst = stablehlo.add %aras2b0gpdst, %arbs2b0gpdst : tensor<f32>
      stablehlo.return %aradds2b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns2b0gpdst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans2b0gpdst = stablehlo.divide %arsums2b0gpdst, %arns2b0gpdst : tensor<2048xf32>
    %v5879 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5880 = stablehlo.slice %armeans2b0gpdst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5881 = stablehlo.slice %armeans2b0gpdst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5882 = stablehlo.slice %armeans2b0gpdst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5883 = stablehlo.slice %armeans2b0gpdst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v5884 = stablehlo.broadcast_in_dim %v5880, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5885 = stablehlo.broadcast_in_dim %v5881, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5886 = stablehlo.broadcast_in_dim %v5882, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5887 = stablehlo.broadcast_in_dim %v5883, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5888 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5889 = stablehlo.add %v5885, %v5888 : tensor<64x512x28x28xf32>
    %v5890 = stablehlo.rsqrt %v5889 : tensor<64x512x28x28xf32>
    %v5891 = stablehlo.subtract %v5879, %v5884 : tensor<64x512x28x28xf32>
    %v5892 = stablehlo.multiply %v5891, %v5890 : tensor<64x512x28x28xf32>
    %v5893 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5894 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5895 = stablehlo.multiply %v5893, %v5894 : tensor<64x512x28x28xf32>
    %v5896 = stablehlo.subtract %v5895, %v5886 : tensor<64x512x28x28xf32>
    %v5897 = stablehlo.multiply %v5892, %v5887 : tensor<64x512x28x28xf32>
    %v5898 = stablehlo.subtract %v5896, %v5897 : tensor<64x512x28x28xf32>
    %v5899 = stablehlo.multiply %v5890, %v5898 : tensor<64x512x28x28xf32>
    %v5900 = stablehlo.reshape %v5899 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v5901 = stablehlo.reshape %v5900 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5902 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5903 = stablehlo.pad %v5901, %v5902, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v5904 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v5905 = stablehlo.transpose %v5904, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v5906 = stablehlo.convert %v5903 : (tensor<64x512x56x56xf32>) -> tensor<64x512x56x56xbf16>
    %v5907 = stablehlo.convert %v5905 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v5908 = stablehlo.convolution(%v5906, %v5907)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x56x56xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v5909 = stablehlo.convert %v5908 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v5910 = stablehlo.reshape %v5909 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5911 = stablehlo.reshape %v5856 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5912 = stablehlo.reshape %v5910 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5913 = stablehlo.add %v5911, %v5912 : tensor<64x256x56x56xf32>
    %v5914 = stablehlo.reshape %v5913 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5915 = stablehlo.reshape %v498 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5916 = stablehlo.reshape %v5848 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5917 = stablehlo.transpose %v5915, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5918 = stablehlo.transpose %v5916, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v5919 = stablehlo.convert %v5917 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5920 = stablehlo.convert %v5918 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v5921 = stablehlo.convolution(%v5919, %v5920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<256x128x1x1xbf16>
    %v5922 = stablehlo.convert %v5921 : (tensor<256x128x1x1xbf16>) -> tensor<256x128x1x1xf32>
    %v5923 = stablehlo.transpose %v5922, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v5924 = stablehlo.reshape %v506 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5925 = stablehlo.slice %v525 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5926 = stablehlo.slice %v525 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5927 = stablehlo.broadcast_in_dim %v5925, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5928 = stablehlo.broadcast_in_dim %v5926, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5929 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v5930 = stablehlo.add %v5928, %v5929 : tensor<64x128x56x56xf32>
    %v5931 = stablehlo.rsqrt %v5930 : tensor<64x128x56x56xf32>
    %v5932 = stablehlo.subtract %v5924, %v5927 : tensor<64x128x56x56xf32>
    %v5933 = stablehlo.multiply %v5932, %v5931 : tensor<64x128x56x56xf32>
    %v5934 = stablehlo.reshape %v5804 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5935 = stablehlo.multiply %v5934, %v5933 : tensor<64x128x56x56xf32>
    %v5936 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5937 = stablehlo.reduce(%v5935 init: %v5936) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5938 = stablehlo.reshape %v5804 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5939 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5940 = stablehlo.reduce(%v5938 init: %v5939) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5941 = stablehlo.reshape %v542 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5942 = stablehlo.reshape %v5788 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5944 = stablehlo.pad %v5942, %v5943, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v5945 = stablehlo.transpose %v5941, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v5946 = stablehlo.transpose %v5944, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v5947 = stablehlo.convert %v5945 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v5948 = stablehlo.convert %v5946 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v5949 = stablehlo.convolution(%v5947, %v5948)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<128x128x3x3xbf16>
    %v5950 = stablehlo.convert %v5949 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v5951 = stablehlo.transpose %v5950, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v5952 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5953 = stablehlo.slice %v569 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v5954 = stablehlo.slice %v569 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v5955 = stablehlo.broadcast_in_dim %v5953, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5956 = stablehlo.broadcast_in_dim %v5954, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5957 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v5958 = stablehlo.add %v5956, %v5957 : tensor<64x128x28x28xf32>
    %v5959 = stablehlo.rsqrt %v5958 : tensor<64x128x28x28xf32>
    %v5960 = stablehlo.subtract %v5952, %v5955 : tensor<64x128x28x28xf32>
    %v5961 = stablehlo.multiply %v5960, %v5959 : tensor<64x128x28x28xf32>
    %v5962 = stablehlo.reshape %v5744 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5963 = stablehlo.multiply %v5962, %v5961 : tensor<64x128x28x28xf32>
    %v5964 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5965 = stablehlo.reduce(%v5963 init: %v5964) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5966 = stablehlo.reshape %v5744 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5967 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5968 = stablehlo.reduce(%v5966 init: %v5967) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5969 = stablehlo.reshape %v586 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5970 = stablehlo.reshape %v5730 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5971 = stablehlo.transpose %v5969, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v5972 = stablehlo.transpose %v5970, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v5973 = stablehlo.convert %v5971 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v5974 = stablehlo.convert %v5972 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v5975 = stablehlo.convolution(%v5973, %v5974)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v5976 = stablehlo.convert %v5975 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v5977 = stablehlo.transpose %v5976, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v5978 = stablehlo.reshape %v594 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5979 = stablehlo.slice %v613 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5980 = stablehlo.slice %v613 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v5981 = stablehlo.broadcast_in_dim %v5979, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5982 = stablehlo.broadcast_in_dim %v5980, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5983 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v5984 = stablehlo.add %v5982, %v5983 : tensor<64x512x28x28xf32>
    %v5985 = stablehlo.rsqrt %v5984 : tensor<64x512x28x28xf32>
    %v5986 = stablehlo.subtract %v5978, %v5981 : tensor<64x512x28x28xf32>
    %v5987 = stablehlo.multiply %v5986, %v5985 : tensor<64x512x28x28xf32>
    %v5988 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5989 = stablehlo.multiply %v5988, %v5987 : tensor<64x512x28x28xf32>
    %v5990 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5991 = stablehlo.reduce(%v5989 init: %v5990) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5992 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5994 = stablehlo.reduce(%v5992 init: %v5993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5995 = stablehlo.reshape %v498 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5996 = stablehlo.reshape %v5900 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5997 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5998 = stablehlo.pad %v5996, %v5997, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v5999 = stablehlo.transpose %v5995, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6000 = stablehlo.transpose %v5998, dims = [1, 0, 2, 3] : (tensor<64x512x56x56xf32>) -> tensor<512x64x56x56xf32>
    %v6001 = stablehlo.convert %v5999 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6002 = stablehlo.convert %v6000 : (tensor<512x64x56x56xf32>) -> tensor<512x64x56x56xbf16>
    %v6003 = stablehlo.convolution(%v6001, %v6002)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<512x64x56x56xbf16>) -> tensor<256x512x1x1xbf16>
    %v6004 = stablehlo.convert %v6003 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v6005 = stablehlo.transpose %v6004, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v6006 = stablehlo.reshape %v636 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6007 = stablehlo.slice %v655 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6008 = stablehlo.slice %v655 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6009 = stablehlo.broadcast_in_dim %v6007, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6010 = stablehlo.broadcast_in_dim %v6008, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v6011 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v6012 = stablehlo.add %v6010, %v6011 : tensor<64x512x28x28xf32>
    %v6013 = stablehlo.rsqrt %v6012 : tensor<64x512x28x28xf32>
    %v6014 = stablehlo.subtract %v6006, %v6009 : tensor<64x512x28x28xf32>
    %v6015 = stablehlo.multiply %v6014, %v6013 : tensor<64x512x28x28xf32>
    %v6016 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6017 = stablehlo.multiply %v6016, %v6015 : tensor<64x512x28x28xf32>
    %v6018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6019 = stablehlo.reduce(%v6017 init: %v6018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6020 = stablehlo.reshape %v5686 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v6021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6022 = stablehlo.reduce(%v6020 init: %v6021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v6023 = stablehlo.reshape %v5914 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6024 = stablehlo.reshape %v494 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6025 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v6026 = stablehlo.compare GT, %v6024, %v6025 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v6027 = stablehlo.select %v6026, %v6023, %v6025 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v6028 = stablehlo.reshape %v6027 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6029 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6030 = stablehlo.slice %v475 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6031 = stablehlo.slice %v475 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6032 = stablehlo.broadcast_in_dim %v6030, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6033 = stablehlo.broadcast_in_dim %v6031, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6034 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6035 = stablehlo.add %v6033, %v6034 : tensor<64x256x56x56xf32>
    %v6036 = stablehlo.rsqrt %v6035 : tensor<64x256x56x56xf32>
    %v6037 = stablehlo.subtract %v6029, %v6032 : tensor<64x256x56x56xf32>
    %v6038 = stablehlo.multiply %v6037, %v6036 : tensor<64x256x56x56xf32>
    %v6039 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6040 = stablehlo.reshape %v6028 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6041 = stablehlo.multiply %v6039, %v6040 : tensor<64x256x56x56xf32>
    %v6042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6043 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6044 = stablehlo.reduce(%v6041 init: %v6042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6045 = stablehlo.divide %v6044, %v6043 : tensor<256xf32>
    %v6046 = stablehlo.multiply %v6038, %v6041 : tensor<64x256x56x56xf32>
    %v6047 = stablehlo.reduce(%v6046 init: %v6042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6048 = stablehlo.divide %v6047, %v6043 : tensor<256xf32>
    %v6049 = stablehlo.concatenate %v6045, %v6048, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6050 = stablehlo.concatenate %v475, %v6049, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b2g3dst = "stablehlo.all_reduce"(%v6050) ({
    ^bb0(%aras1b2g3dst: tensor<f32>, %arbs1b2g3dst: tensor<f32>):
      %aradds1b2g3dst = stablehlo.add %aras1b2g3dst, %arbs1b2g3dst : tensor<f32>
      stablehlo.return %aradds1b2g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b2g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b2g3dst = stablehlo.divide %arsums1b2g3dst, %arns1b2g3dst : tensor<1024xf32>
    %v6051 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6052 = stablehlo.slice %armeans1b2g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6053 = stablehlo.slice %armeans1b2g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6054 = stablehlo.slice %armeans1b2g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6055 = stablehlo.slice %armeans1b2g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6056 = stablehlo.broadcast_in_dim %v6052, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6057 = stablehlo.broadcast_in_dim %v6053, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6058 = stablehlo.broadcast_in_dim %v6054, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6059 = stablehlo.broadcast_in_dim %v6055, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6060 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6061 = stablehlo.add %v6057, %v6060 : tensor<64x256x56x56xf32>
    %v6062 = stablehlo.rsqrt %v6061 : tensor<64x256x56x56xf32>
    %v6063 = stablehlo.subtract %v6051, %v6056 : tensor<64x256x56x56xf32>
    %v6064 = stablehlo.multiply %v6063, %v6062 : tensor<64x256x56x56xf32>
    %v6065 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6066 = stablehlo.reshape %v6028 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6067 = stablehlo.multiply %v6065, %v6066 : tensor<64x256x56x56xf32>
    %v6068 = stablehlo.subtract %v6067, %v6058 : tensor<64x256x56x56xf32>
    %v6069 = stablehlo.multiply %v6064, %v6059 : tensor<64x256x56x56xf32>
    %v6070 = stablehlo.subtract %v6068, %v6069 : tensor<64x256x56x56xf32>
    %v6071 = stablehlo.multiply %v6062, %v6070 : tensor<64x256x56x56xf32>
    %v6072 = stablehlo.reshape %v6071 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6073 = stablehlo.reshape %v6072 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6074 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6075 = stablehlo.transpose %v6074, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6076 = stablehlo.convert %v6073 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v6077 = stablehlo.convert %v6075 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v6078 = stablehlo.convolution(%v6076, %v6077)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v6079 = stablehlo.convert %v6078 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6080 = stablehlo.reshape %v6079 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6081 = stablehlo.reshape %v6080 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6082 = stablehlo.reshape %v446 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6083 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6084 = stablehlo.compare GT, %v6082, %v6083 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6085 = stablehlo.select %v6084, %v6081, %v6083 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6086 = stablehlo.reshape %v6085 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6087 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6088 = stablehlo.slice %v431 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6089 = stablehlo.slice %v431 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6090 = stablehlo.broadcast_in_dim %v6088, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6091 = stablehlo.broadcast_in_dim %v6089, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6092 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6093 = stablehlo.add %v6091, %v6092 : tensor<64x64x56x56xf32>
    %v6094 = stablehlo.rsqrt %v6093 : tensor<64x64x56x56xf32>
    %v6095 = stablehlo.subtract %v6087, %v6090 : tensor<64x64x56x56xf32>
    %v6096 = stablehlo.multiply %v6095, %v6094 : tensor<64x64x56x56xf32>
    %v6097 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6098 = stablehlo.reshape %v6086 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6099 = stablehlo.multiply %v6097, %v6098 : tensor<64x64x56x56xf32>
    %v6100 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6101 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6102 = stablehlo.reduce(%v6099 init: %v6100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6103 = stablehlo.divide %v6102, %v6101 : tensor<64xf32>
    %v6104 = stablehlo.multiply %v6096, %v6099 : tensor<64x64x56x56xf32>
    %v6105 = stablehlo.reduce(%v6104 init: %v6100) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6106 = stablehlo.divide %v6105, %v6101 : tensor<64xf32>
    %v6107 = stablehlo.concatenate %v6103, %v6106, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6108 = stablehlo.concatenate %v431, %v6107, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g2dst = "stablehlo.all_reduce"(%v6108) ({
    ^bb0(%aras1b2g2dst: tensor<f32>, %arbs1b2g2dst: tensor<f32>):
      %aradds1b2g2dst = stablehlo.add %aras1b2g2dst, %arbs1b2g2dst : tensor<f32>
      stablehlo.return %aradds1b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g2dst = stablehlo.divide %arsums1b2g2dst, %arns1b2g2dst : tensor<256xf32>
    %v6109 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6110 = stablehlo.slice %armeans1b2g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6111 = stablehlo.slice %armeans1b2g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6112 = stablehlo.slice %armeans1b2g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6113 = stablehlo.slice %armeans1b2g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6114 = stablehlo.broadcast_in_dim %v6110, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6115 = stablehlo.broadcast_in_dim %v6111, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6116 = stablehlo.broadcast_in_dim %v6112, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6117 = stablehlo.broadcast_in_dim %v6113, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6118 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6119 = stablehlo.add %v6115, %v6118 : tensor<64x64x56x56xf32>
    %v6120 = stablehlo.rsqrt %v6119 : tensor<64x64x56x56xf32>
    %v6121 = stablehlo.subtract %v6109, %v6114 : tensor<64x64x56x56xf32>
    %v6122 = stablehlo.multiply %v6121, %v6120 : tensor<64x64x56x56xf32>
    %v6123 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6124 = stablehlo.reshape %v6086 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6125 = stablehlo.multiply %v6123, %v6124 : tensor<64x64x56x56xf32>
    %v6126 = stablehlo.subtract %v6125, %v6116 : tensor<64x64x56x56xf32>
    %v6127 = stablehlo.multiply %v6122, %v6117 : tensor<64x64x56x56xf32>
    %v6128 = stablehlo.subtract %v6126, %v6127 : tensor<64x64x56x56xf32>
    %v6129 = stablehlo.multiply %v6120, %v6128 : tensor<64x64x56x56xf32>
    %v6130 = stablehlo.reshape %v6129 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6131 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6132 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v6133 = stablehlo.transpose %v6132, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6134 = stablehlo.convert %v6131 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6135 = stablehlo.convert %v6133 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v6136 = stablehlo.convolution(%v6134, %v6135)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v6137 = stablehlo.convert %v6136 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6138 = stablehlo.reshape %v6137 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6139 = stablehlo.reshape %v6138 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6140 = stablehlo.reshape %v402 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6141 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6142 = stablehlo.compare GT, %v6140, %v6141 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6143 = stablehlo.select %v6142, %v6139, %v6141 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6144 = stablehlo.reshape %v6143 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6145 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6146 = stablehlo.slice %v387 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6147 = stablehlo.slice %v387 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6148 = stablehlo.broadcast_in_dim %v6146, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6149 = stablehlo.broadcast_in_dim %v6147, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6150 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6151 = stablehlo.add %v6149, %v6150 : tensor<64x64x56x56xf32>
    %v6152 = stablehlo.rsqrt %v6151 : tensor<64x64x56x56xf32>
    %v6153 = stablehlo.subtract %v6145, %v6148 : tensor<64x64x56x56xf32>
    %v6154 = stablehlo.multiply %v6153, %v6152 : tensor<64x64x56x56xf32>
    %v6155 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6156 = stablehlo.reshape %v6144 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6157 = stablehlo.multiply %v6155, %v6156 : tensor<64x64x56x56xf32>
    %v6158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6159 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6160 = stablehlo.reduce(%v6157 init: %v6158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6161 = stablehlo.divide %v6160, %v6159 : tensor<64xf32>
    %v6162 = stablehlo.multiply %v6154, %v6157 : tensor<64x64x56x56xf32>
    %v6163 = stablehlo.reduce(%v6162 init: %v6158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6164 = stablehlo.divide %v6163, %v6159 : tensor<64xf32>
    %v6165 = stablehlo.concatenate %v6161, %v6164, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6166 = stablehlo.concatenate %v387, %v6165, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g1dst = "stablehlo.all_reduce"(%v6166) ({
    ^bb0(%aras1b2g1dst: tensor<f32>, %arbs1b2g1dst: tensor<f32>):
      %aradds1b2g1dst = stablehlo.add %aras1b2g1dst, %arbs1b2g1dst : tensor<f32>
      stablehlo.return %aradds1b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g1dst = stablehlo.divide %arsums1b2g1dst, %arns1b2g1dst : tensor<256xf32>
    %v6167 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6168 = stablehlo.slice %armeans1b2g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6169 = stablehlo.slice %armeans1b2g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6170 = stablehlo.slice %armeans1b2g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6171 = stablehlo.slice %armeans1b2g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6172 = stablehlo.broadcast_in_dim %v6168, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6173 = stablehlo.broadcast_in_dim %v6169, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6174 = stablehlo.broadcast_in_dim %v6170, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6175 = stablehlo.broadcast_in_dim %v6171, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6176 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6177 = stablehlo.add %v6173, %v6176 : tensor<64x64x56x56xf32>
    %v6178 = stablehlo.rsqrt %v6177 : tensor<64x64x56x56xf32>
    %v6179 = stablehlo.subtract %v6167, %v6172 : tensor<64x64x56x56xf32>
    %v6180 = stablehlo.multiply %v6179, %v6178 : tensor<64x64x56x56xf32>
    %v6181 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6182 = stablehlo.reshape %v6144 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6183 = stablehlo.multiply %v6181, %v6182 : tensor<64x64x56x56xf32>
    %v6184 = stablehlo.subtract %v6183, %v6174 : tensor<64x64x56x56xf32>
    %v6185 = stablehlo.multiply %v6180, %v6175 : tensor<64x64x56x56xf32>
    %v6186 = stablehlo.subtract %v6184, %v6185 : tensor<64x64x56x56xf32>
    %v6187 = stablehlo.multiply %v6178, %v6186 : tensor<64x64x56x56xf32>
    %v6188 = stablehlo.reshape %v6187 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6189 = stablehlo.reshape %v6188 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6190 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v6191 = stablehlo.transpose %v6190, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6192 = stablehlo.convert %v6189 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6193 = stablehlo.convert %v6191 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v6194 = stablehlo.convolution(%v6192, %v6193)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v6195 = stablehlo.convert %v6194 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v6196 = stablehlo.reshape %v6195 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6197 = stablehlo.reshape %v6196 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6198 = stablehlo.reshape %v6028 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6199 = stablehlo.add %v6197, %v6198 : tensor<64x256x56x56xf32>
    %v6200 = stablehlo.reshape %v6199 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6201 = stablehlo.reshape %v360 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6202 = stablehlo.reshape %v6188 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6203 = stablehlo.transpose %v6201, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6204 = stablehlo.transpose %v6202, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6205 = stablehlo.convert %v6203 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6206 = stablehlo.convert %v6204 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6207 = stablehlo.convolution(%v6205, %v6206)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v6208 = stablehlo.convert %v6207 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v6209 = stablehlo.transpose %v6208, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6210 = stablehlo.reshape %v368 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6211 = stablehlo.slice %v387 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6212 = stablehlo.slice %v387 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6213 = stablehlo.broadcast_in_dim %v6211, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6214 = stablehlo.broadcast_in_dim %v6212, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6215 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6216 = stablehlo.add %v6214, %v6215 : tensor<64x64x56x56xf32>
    %v6217 = stablehlo.rsqrt %v6216 : tensor<64x64x56x56xf32>
    %v6218 = stablehlo.subtract %v6210, %v6213 : tensor<64x64x56x56xf32>
    %v6219 = stablehlo.multiply %v6218, %v6217 : tensor<64x64x56x56xf32>
    %v6220 = stablehlo.reshape %v6144 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6221 = stablehlo.multiply %v6220, %v6219 : tensor<64x64x56x56xf32>
    %v6222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6223 = stablehlo.reduce(%v6221 init: %v6222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6224 = stablehlo.reshape %v6144 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6225 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6226 = stablehlo.reduce(%v6224 init: %v6225) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6227 = stablehlo.reshape %v404 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6228 = stablehlo.reshape %v6130 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6229 = stablehlo.transpose %v6227, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6230 = stablehlo.transpose %v6228, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6231 = stablehlo.convert %v6229 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6232 = stablehlo.convert %v6230 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6233 = stablehlo.convolution(%v6231, %v6232)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v6234 = stablehlo.convert %v6233 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v6235 = stablehlo.transpose %v6234, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6236 = stablehlo.reshape %v412 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6237 = stablehlo.slice %v431 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6238 = stablehlo.slice %v431 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6239 = stablehlo.broadcast_in_dim %v6237, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6240 = stablehlo.broadcast_in_dim %v6238, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6241 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6242 = stablehlo.add %v6240, %v6241 : tensor<64x64x56x56xf32>
    %v6243 = stablehlo.rsqrt %v6242 : tensor<64x64x56x56xf32>
    %v6244 = stablehlo.subtract %v6236, %v6239 : tensor<64x64x56x56xf32>
    %v6245 = stablehlo.multiply %v6244, %v6243 : tensor<64x64x56x56xf32>
    %v6246 = stablehlo.reshape %v6086 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6247 = stablehlo.multiply %v6246, %v6245 : tensor<64x64x56x56xf32>
    %v6248 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6249 = stablehlo.reduce(%v6247 init: %v6248) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6250 = stablehlo.reshape %v6086 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6251 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6252 = stablehlo.reduce(%v6250 init: %v6251) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6253 = stablehlo.reshape %v448 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6254 = stablehlo.reshape %v6072 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6255 = stablehlo.transpose %v6253, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6256 = stablehlo.transpose %v6254, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6257 = stablehlo.convert %v6255 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6258 = stablehlo.convert %v6256 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6259 = stablehlo.convolution(%v6257, %v6258)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v6260 = stablehlo.convert %v6259 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v6261 = stablehlo.transpose %v6260, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6262 = stablehlo.reshape %v456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6263 = stablehlo.slice %v475 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6264 = stablehlo.slice %v475 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6265 = stablehlo.broadcast_in_dim %v6263, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6266 = stablehlo.broadcast_in_dim %v6264, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6267 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6268 = stablehlo.add %v6266, %v6267 : tensor<64x256x56x56xf32>
    %v6269 = stablehlo.rsqrt %v6268 : tensor<64x256x56x56xf32>
    %v6270 = stablehlo.subtract %v6262, %v6265 : tensor<64x256x56x56xf32>
    %v6271 = stablehlo.multiply %v6270, %v6269 : tensor<64x256x56x56xf32>
    %v6272 = stablehlo.reshape %v6028 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6273 = stablehlo.multiply %v6272, %v6271 : tensor<64x256x56x56xf32>
    %v6274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6275 = stablehlo.reduce(%v6273 init: %v6274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6276 = stablehlo.reshape %v6028 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6277 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6278 = stablehlo.reduce(%v6276 init: %v6277) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6279 = stablehlo.reshape %v6200 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6280 = stablehlo.reshape %v356 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6281 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v6282 = stablehlo.compare GT, %v6280, %v6281 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v6283 = stablehlo.select %v6282, %v6279, %v6281 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v6284 = stablehlo.reshape %v6283 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6285 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6286 = stablehlo.slice %v337 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6287 = stablehlo.slice %v337 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6288 = stablehlo.broadcast_in_dim %v6286, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6289 = stablehlo.broadcast_in_dim %v6287, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6290 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6291 = stablehlo.add %v6289, %v6290 : tensor<64x256x56x56xf32>
    %v6292 = stablehlo.rsqrt %v6291 : tensor<64x256x56x56xf32>
    %v6293 = stablehlo.subtract %v6285, %v6288 : tensor<64x256x56x56xf32>
    %v6294 = stablehlo.multiply %v6293, %v6292 : tensor<64x256x56x56xf32>
    %v6295 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6296 = stablehlo.reshape %v6284 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6297 = stablehlo.multiply %v6295, %v6296 : tensor<64x256x56x56xf32>
    %v6298 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6299 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6300 = stablehlo.reduce(%v6297 init: %v6298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6301 = stablehlo.divide %v6300, %v6299 : tensor<256xf32>
    %v6302 = stablehlo.multiply %v6294, %v6297 : tensor<64x256x56x56xf32>
    %v6303 = stablehlo.reduce(%v6302 init: %v6298) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6304 = stablehlo.divide %v6303, %v6299 : tensor<256xf32>
    %v6305 = stablehlo.concatenate %v6301, %v6304, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6306 = stablehlo.concatenate %v337, %v6305, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b1g3dst = "stablehlo.all_reduce"(%v6306) ({
    ^bb0(%aras1b1g3dst: tensor<f32>, %arbs1b1g3dst: tensor<f32>):
      %aradds1b1g3dst = stablehlo.add %aras1b1g3dst, %arbs1b1g3dst : tensor<f32>
      stablehlo.return %aradds1b1g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b1g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b1g3dst = stablehlo.divide %arsums1b1g3dst, %arns1b1g3dst : tensor<1024xf32>
    %v6307 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6308 = stablehlo.slice %armeans1b1g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6309 = stablehlo.slice %armeans1b1g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6310 = stablehlo.slice %armeans1b1g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6311 = stablehlo.slice %armeans1b1g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6312 = stablehlo.broadcast_in_dim %v6308, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6313 = stablehlo.broadcast_in_dim %v6309, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6314 = stablehlo.broadcast_in_dim %v6310, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6315 = stablehlo.broadcast_in_dim %v6311, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6316 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6317 = stablehlo.add %v6313, %v6316 : tensor<64x256x56x56xf32>
    %v6318 = stablehlo.rsqrt %v6317 : tensor<64x256x56x56xf32>
    %v6319 = stablehlo.subtract %v6307, %v6312 : tensor<64x256x56x56xf32>
    %v6320 = stablehlo.multiply %v6319, %v6318 : tensor<64x256x56x56xf32>
    %v6321 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6322 = stablehlo.reshape %v6284 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6323 = stablehlo.multiply %v6321, %v6322 : tensor<64x256x56x56xf32>
    %v6324 = stablehlo.subtract %v6323, %v6314 : tensor<64x256x56x56xf32>
    %v6325 = stablehlo.multiply %v6320, %v6315 : tensor<64x256x56x56xf32>
    %v6326 = stablehlo.subtract %v6324, %v6325 : tensor<64x256x56x56xf32>
    %v6327 = stablehlo.multiply %v6318, %v6326 : tensor<64x256x56x56xf32>
    %v6328 = stablehlo.reshape %v6327 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6329 = stablehlo.reshape %v6328 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6330 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6331 = stablehlo.transpose %v6330, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6332 = stablehlo.convert %v6329 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v6333 = stablehlo.convert %v6331 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v6334 = stablehlo.convolution(%v6332, %v6333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v6335 = stablehlo.convert %v6334 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6336 = stablehlo.reshape %v6335 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6337 = stablehlo.reshape %v6336 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6338 = stablehlo.reshape %v308 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6339 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6340 = stablehlo.compare GT, %v6338, %v6339 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6341 = stablehlo.select %v6340, %v6337, %v6339 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6342 = stablehlo.reshape %v6341 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6343 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6344 = stablehlo.slice %v293 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6345 = stablehlo.slice %v293 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6346 = stablehlo.broadcast_in_dim %v6344, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6347 = stablehlo.broadcast_in_dim %v6345, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6348 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6349 = stablehlo.add %v6347, %v6348 : tensor<64x64x56x56xf32>
    %v6350 = stablehlo.rsqrt %v6349 : tensor<64x64x56x56xf32>
    %v6351 = stablehlo.subtract %v6343, %v6346 : tensor<64x64x56x56xf32>
    %v6352 = stablehlo.multiply %v6351, %v6350 : tensor<64x64x56x56xf32>
    %v6353 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6354 = stablehlo.reshape %v6342 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6355 = stablehlo.multiply %v6353, %v6354 : tensor<64x64x56x56xf32>
    %v6356 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6357 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6358 = stablehlo.reduce(%v6355 init: %v6356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6359 = stablehlo.divide %v6358, %v6357 : tensor<64xf32>
    %v6360 = stablehlo.multiply %v6352, %v6355 : tensor<64x64x56x56xf32>
    %v6361 = stablehlo.reduce(%v6360 init: %v6356) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6362 = stablehlo.divide %v6361, %v6357 : tensor<64xf32>
    %v6363 = stablehlo.concatenate %v6359, %v6362, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6364 = stablehlo.concatenate %v293, %v6363, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g2dst = "stablehlo.all_reduce"(%v6364) ({
    ^bb0(%aras1b1g2dst: tensor<f32>, %arbs1b1g2dst: tensor<f32>):
      %aradds1b1g2dst = stablehlo.add %aras1b1g2dst, %arbs1b1g2dst : tensor<f32>
      stablehlo.return %aradds1b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g2dst = stablehlo.divide %arsums1b1g2dst, %arns1b1g2dst : tensor<256xf32>
    %v6365 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6366 = stablehlo.slice %armeans1b1g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6367 = stablehlo.slice %armeans1b1g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6368 = stablehlo.slice %armeans1b1g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6369 = stablehlo.slice %armeans1b1g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6370 = stablehlo.broadcast_in_dim %v6366, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6371 = stablehlo.broadcast_in_dim %v6367, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6372 = stablehlo.broadcast_in_dim %v6368, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6373 = stablehlo.broadcast_in_dim %v6369, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6374 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6375 = stablehlo.add %v6371, %v6374 : tensor<64x64x56x56xf32>
    %v6376 = stablehlo.rsqrt %v6375 : tensor<64x64x56x56xf32>
    %v6377 = stablehlo.subtract %v6365, %v6370 : tensor<64x64x56x56xf32>
    %v6378 = stablehlo.multiply %v6377, %v6376 : tensor<64x64x56x56xf32>
    %v6379 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6380 = stablehlo.reshape %v6342 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6381 = stablehlo.multiply %v6379, %v6380 : tensor<64x64x56x56xf32>
    %v6382 = stablehlo.subtract %v6381, %v6372 : tensor<64x64x56x56xf32>
    %v6383 = stablehlo.multiply %v6378, %v6373 : tensor<64x64x56x56xf32>
    %v6384 = stablehlo.subtract %v6382, %v6383 : tensor<64x64x56x56xf32>
    %v6385 = stablehlo.multiply %v6376, %v6384 : tensor<64x64x56x56xf32>
    %v6386 = stablehlo.reshape %v6385 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6387 = stablehlo.reshape %v6386 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6388 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v6389 = stablehlo.transpose %v6388, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6390 = stablehlo.convert %v6387 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6391 = stablehlo.convert %v6389 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v6392 = stablehlo.convolution(%v6390, %v6391)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v6393 = stablehlo.convert %v6392 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6394 = stablehlo.reshape %v6393 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6395 = stablehlo.reshape %v6394 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6396 = stablehlo.reshape %v264 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6397 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6398 = stablehlo.compare GT, %v6396, %v6397 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6399 = stablehlo.select %v6398, %v6395, %v6397 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6400 = stablehlo.reshape %v6399 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6401 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6402 = stablehlo.slice %v249 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6403 = stablehlo.slice %v249 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6404 = stablehlo.broadcast_in_dim %v6402, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6405 = stablehlo.broadcast_in_dim %v6403, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6406 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6407 = stablehlo.add %v6405, %v6406 : tensor<64x64x56x56xf32>
    %v6408 = stablehlo.rsqrt %v6407 : tensor<64x64x56x56xf32>
    %v6409 = stablehlo.subtract %v6401, %v6404 : tensor<64x64x56x56xf32>
    %v6410 = stablehlo.multiply %v6409, %v6408 : tensor<64x64x56x56xf32>
    %v6411 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6412 = stablehlo.reshape %v6400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6413 = stablehlo.multiply %v6411, %v6412 : tensor<64x64x56x56xf32>
    %v6414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6415 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6416 = stablehlo.reduce(%v6413 init: %v6414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6417 = stablehlo.divide %v6416, %v6415 : tensor<64xf32>
    %v6418 = stablehlo.multiply %v6410, %v6413 : tensor<64x64x56x56xf32>
    %v6419 = stablehlo.reduce(%v6418 init: %v6414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6420 = stablehlo.divide %v6419, %v6415 : tensor<64xf32>
    %v6421 = stablehlo.concatenate %v6417, %v6420, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6422 = stablehlo.concatenate %v249, %v6421, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g1dst = "stablehlo.all_reduce"(%v6422) ({
    ^bb0(%aras1b1g1dst: tensor<f32>, %arbs1b1g1dst: tensor<f32>):
      %aradds1b1g1dst = stablehlo.add %aras1b1g1dst, %arbs1b1g1dst : tensor<f32>
      stablehlo.return %aradds1b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g1dst = stablehlo.divide %arsums1b1g1dst, %arns1b1g1dst : tensor<256xf32>
    %v6423 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6424 = stablehlo.slice %armeans1b1g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6425 = stablehlo.slice %armeans1b1g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6426 = stablehlo.slice %armeans1b1g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6427 = stablehlo.slice %armeans1b1g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6428 = stablehlo.broadcast_in_dim %v6424, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6429 = stablehlo.broadcast_in_dim %v6425, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6430 = stablehlo.broadcast_in_dim %v6426, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6431 = stablehlo.broadcast_in_dim %v6427, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6432 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6433 = stablehlo.add %v6429, %v6432 : tensor<64x64x56x56xf32>
    %v6434 = stablehlo.rsqrt %v6433 : tensor<64x64x56x56xf32>
    %v6435 = stablehlo.subtract %v6423, %v6428 : tensor<64x64x56x56xf32>
    %v6436 = stablehlo.multiply %v6435, %v6434 : tensor<64x64x56x56xf32>
    %v6437 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6438 = stablehlo.reshape %v6400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6439 = stablehlo.multiply %v6437, %v6438 : tensor<64x64x56x56xf32>
    %v6440 = stablehlo.subtract %v6439, %v6430 : tensor<64x64x56x56xf32>
    %v6441 = stablehlo.multiply %v6436, %v6431 : tensor<64x64x56x56xf32>
    %v6442 = stablehlo.subtract %v6440, %v6441 : tensor<64x64x56x56xf32>
    %v6443 = stablehlo.multiply %v6434, %v6442 : tensor<64x64x56x56xf32>
    %v6444 = stablehlo.reshape %v6443 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6445 = stablehlo.reshape %v6444 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6446 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v6447 = stablehlo.transpose %v6446, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6448 = stablehlo.convert %v6445 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6449 = stablehlo.convert %v6447 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v6450 = stablehlo.convolution(%v6448, %v6449)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v6451 = stablehlo.convert %v6450 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v6452 = stablehlo.reshape %v6451 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6453 = stablehlo.reshape %v6452 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6454 = stablehlo.reshape %v6284 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6455 = stablehlo.add %v6453, %v6454 : tensor<64x256x56x56xf32>
    %v6456 = stablehlo.reshape %v6455 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6457 = stablehlo.reshape %v222 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6458 = stablehlo.reshape %v6444 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6459 = stablehlo.transpose %v6457, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6460 = stablehlo.transpose %v6458, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6461 = stablehlo.convert %v6459 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6462 = stablehlo.convert %v6460 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6463 = stablehlo.convolution(%v6461, %v6462)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v6464 = stablehlo.convert %v6463 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v6465 = stablehlo.transpose %v6464, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6466 = stablehlo.reshape %v230 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6467 = stablehlo.slice %v249 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6468 = stablehlo.slice %v249 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6469 = stablehlo.broadcast_in_dim %v6467, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6470 = stablehlo.broadcast_in_dim %v6468, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6471 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6472 = stablehlo.add %v6470, %v6471 : tensor<64x64x56x56xf32>
    %v6473 = stablehlo.rsqrt %v6472 : tensor<64x64x56x56xf32>
    %v6474 = stablehlo.subtract %v6466, %v6469 : tensor<64x64x56x56xf32>
    %v6475 = stablehlo.multiply %v6474, %v6473 : tensor<64x64x56x56xf32>
    %v6476 = stablehlo.reshape %v6400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6477 = stablehlo.multiply %v6476, %v6475 : tensor<64x64x56x56xf32>
    %v6478 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6479 = stablehlo.reduce(%v6477 init: %v6478) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6480 = stablehlo.reshape %v6400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6482 = stablehlo.reduce(%v6480 init: %v6481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6483 = stablehlo.reshape %v266 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6484 = stablehlo.reshape %v6386 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6485 = stablehlo.transpose %v6483, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6486 = stablehlo.transpose %v6484, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6487 = stablehlo.convert %v6485 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6488 = stablehlo.convert %v6486 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6489 = stablehlo.convolution(%v6487, %v6488)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v6490 = stablehlo.convert %v6489 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v6491 = stablehlo.transpose %v6490, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6492 = stablehlo.reshape %v274 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6493 = stablehlo.slice %v293 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6494 = stablehlo.slice %v293 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6495 = stablehlo.broadcast_in_dim %v6493, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6496 = stablehlo.broadcast_in_dim %v6494, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6497 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6498 = stablehlo.add %v6496, %v6497 : tensor<64x64x56x56xf32>
    %v6499 = stablehlo.rsqrt %v6498 : tensor<64x64x56x56xf32>
    %v6500 = stablehlo.subtract %v6492, %v6495 : tensor<64x64x56x56xf32>
    %v6501 = stablehlo.multiply %v6500, %v6499 : tensor<64x64x56x56xf32>
    %v6502 = stablehlo.reshape %v6342 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6503 = stablehlo.multiply %v6502, %v6501 : tensor<64x64x56x56xf32>
    %v6504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6505 = stablehlo.reduce(%v6503 init: %v6504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6506 = stablehlo.reshape %v6342 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6507 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6508 = stablehlo.reduce(%v6506 init: %v6507) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6509 = stablehlo.reshape %v310 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6510 = stablehlo.reshape %v6328 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6511 = stablehlo.transpose %v6509, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6512 = stablehlo.transpose %v6510, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6513 = stablehlo.convert %v6511 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6514 = stablehlo.convert %v6512 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6515 = stablehlo.convolution(%v6513, %v6514)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v6516 = stablehlo.convert %v6515 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v6517 = stablehlo.transpose %v6516, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6518 = stablehlo.reshape %v318 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6519 = stablehlo.slice %v337 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6520 = stablehlo.slice %v337 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6521 = stablehlo.broadcast_in_dim %v6519, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6522 = stablehlo.broadcast_in_dim %v6520, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6523 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6524 = stablehlo.add %v6522, %v6523 : tensor<64x256x56x56xf32>
    %v6525 = stablehlo.rsqrt %v6524 : tensor<64x256x56x56xf32>
    %v6526 = stablehlo.subtract %v6518, %v6521 : tensor<64x256x56x56xf32>
    %v6527 = stablehlo.multiply %v6526, %v6525 : tensor<64x256x56x56xf32>
    %v6528 = stablehlo.reshape %v6284 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6529 = stablehlo.multiply %v6528, %v6527 : tensor<64x256x56x56xf32>
    %v6530 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6531 = stablehlo.reduce(%v6529 init: %v6530) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6532 = stablehlo.reshape %v6284 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6534 = stablehlo.reduce(%v6532 init: %v6533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6535 = stablehlo.reshape %v6456 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6536 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6537 = stablehlo.constant dense<0.0> : tensor<64x256x56x56xf32>
    %v6538 = stablehlo.compare GT, %v6536, %v6537 : (tensor<64x256x56x56xf32>, tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xi1>
    %v6539 = stablehlo.select %v6538, %v6535, %v6537 : tensor<64x256x56x56xi1>, tensor<64x256x56x56xf32>
    %v6540 = stablehlo.reshape %v6539 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6541 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6542 = stablehlo.slice %v162 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6543 = stablehlo.slice %v162 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6544 = stablehlo.broadcast_in_dim %v6542, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6545 = stablehlo.broadcast_in_dim %v6543, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6546 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6547 = stablehlo.add %v6545, %v6546 : tensor<64x256x56x56xf32>
    %v6548 = stablehlo.rsqrt %v6547 : tensor<64x256x56x56xf32>
    %v6549 = stablehlo.subtract %v6541, %v6544 : tensor<64x256x56x56xf32>
    %v6550 = stablehlo.multiply %v6549, %v6548 : tensor<64x256x56x56xf32>
    %v6551 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6552 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6553 = stablehlo.multiply %v6551, %v6552 : tensor<64x256x56x56xf32>
    %v6554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6555 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6556 = stablehlo.reduce(%v6553 init: %v6554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6557 = stablehlo.divide %v6556, %v6555 : tensor<256xf32>
    %v6558 = stablehlo.multiply %v6550, %v6553 : tensor<64x256x56x56xf32>
    %v6559 = stablehlo.reduce(%v6558 init: %v6554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6560 = stablehlo.divide %v6559, %v6555 : tensor<256xf32>
    %v6561 = stablehlo.concatenate %v6557, %v6560, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6562 = stablehlo.concatenate %v162, %v6561, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b0g3dst = "stablehlo.all_reduce"(%v6562) ({
    ^bb0(%aras1b0g3dst: tensor<f32>, %arbs1b0g3dst: tensor<f32>):
      %aradds1b0g3dst = stablehlo.add %aras1b0g3dst, %arbs1b0g3dst : tensor<f32>
      stablehlo.return %aradds1b0g3dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b0g3dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b0g3dst = stablehlo.divide %arsums1b0g3dst, %arns1b0g3dst : tensor<1024xf32>
    %v6563 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6564 = stablehlo.slice %armeans1b0g3dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6565 = stablehlo.slice %armeans1b0g3dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6566 = stablehlo.slice %armeans1b0g3dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6567 = stablehlo.slice %armeans1b0g3dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6568 = stablehlo.broadcast_in_dim %v6564, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6569 = stablehlo.broadcast_in_dim %v6565, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6570 = stablehlo.broadcast_in_dim %v6566, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6571 = stablehlo.broadcast_in_dim %v6567, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6572 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6573 = stablehlo.add %v6569, %v6572 : tensor<64x256x56x56xf32>
    %v6574 = stablehlo.rsqrt %v6573 : tensor<64x256x56x56xf32>
    %v6575 = stablehlo.subtract %v6563, %v6568 : tensor<64x256x56x56xf32>
    %v6576 = stablehlo.multiply %v6575, %v6574 : tensor<64x256x56x56xf32>
    %v6577 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6578 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6579 = stablehlo.multiply %v6577, %v6578 : tensor<64x256x56x56xf32>
    %v6580 = stablehlo.subtract %v6579, %v6570 : tensor<64x256x56x56xf32>
    %v6581 = stablehlo.multiply %v6576, %v6571 : tensor<64x256x56x56xf32>
    %v6582 = stablehlo.subtract %v6580, %v6581 : tensor<64x256x56x56xf32>
    %v6583 = stablehlo.multiply %v6574, %v6582 : tensor<64x256x56x56xf32>
    %v6584 = stablehlo.reshape %v6583 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6585 = stablehlo.reshape %v6584 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6586 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6587 = stablehlo.transpose %v6586, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6588 = stablehlo.convert %v6585 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v6589 = stablehlo.convert %v6587 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v6590 = stablehlo.convolution(%v6588, %v6589)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v6591 = stablehlo.convert %v6590 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6592 = stablehlo.reshape %v6591 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6593 = stablehlo.reshape %v6592 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6594 = stablehlo.reshape %v133 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6595 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6596 = stablehlo.compare GT, %v6594, %v6595 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6597 = stablehlo.select %v6596, %v6593, %v6595 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6598 = stablehlo.reshape %v6597 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6599 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6600 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6601 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6602 = stablehlo.broadcast_in_dim %v6600, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6603 = stablehlo.broadcast_in_dim %v6601, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6604 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6605 = stablehlo.add %v6603, %v6604 : tensor<64x64x56x56xf32>
    %v6606 = stablehlo.rsqrt %v6605 : tensor<64x64x56x56xf32>
    %v6607 = stablehlo.subtract %v6599, %v6602 : tensor<64x64x56x56xf32>
    %v6608 = stablehlo.multiply %v6607, %v6606 : tensor<64x64x56x56xf32>
    %v6609 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6610 = stablehlo.reshape %v6598 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6611 = stablehlo.multiply %v6609, %v6610 : tensor<64x64x56x56xf32>
    %v6612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6613 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6614 = stablehlo.reduce(%v6611 init: %v6612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6615 = stablehlo.divide %v6614, %v6613 : tensor<64xf32>
    %v6616 = stablehlo.multiply %v6608, %v6611 : tensor<64x64x56x56xf32>
    %v6617 = stablehlo.reduce(%v6616 init: %v6612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6618 = stablehlo.divide %v6617, %v6613 : tensor<64xf32>
    %v6619 = stablehlo.concatenate %v6615, %v6618, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6620 = stablehlo.concatenate %v118, %v6619, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g2dst = "stablehlo.all_reduce"(%v6620) ({
    ^bb0(%aras1b0g2dst: tensor<f32>, %arbs1b0g2dst: tensor<f32>):
      %aradds1b0g2dst = stablehlo.add %aras1b0g2dst, %arbs1b0g2dst : tensor<f32>
      stablehlo.return %aradds1b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g2dst = stablehlo.divide %arsums1b0g2dst, %arns1b0g2dst : tensor<256xf32>
    %v6621 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6622 = stablehlo.slice %armeans1b0g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6623 = stablehlo.slice %armeans1b0g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6624 = stablehlo.slice %armeans1b0g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6625 = stablehlo.slice %armeans1b0g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6626 = stablehlo.broadcast_in_dim %v6622, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6627 = stablehlo.broadcast_in_dim %v6623, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6628 = stablehlo.broadcast_in_dim %v6624, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6629 = stablehlo.broadcast_in_dim %v6625, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6630 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6631 = stablehlo.add %v6627, %v6630 : tensor<64x64x56x56xf32>
    %v6632 = stablehlo.rsqrt %v6631 : tensor<64x64x56x56xf32>
    %v6633 = stablehlo.subtract %v6621, %v6626 : tensor<64x64x56x56xf32>
    %v6634 = stablehlo.multiply %v6633, %v6632 : tensor<64x64x56x56xf32>
    %v6635 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6636 = stablehlo.reshape %v6598 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6637 = stablehlo.multiply %v6635, %v6636 : tensor<64x64x56x56xf32>
    %v6638 = stablehlo.subtract %v6637, %v6628 : tensor<64x64x56x56xf32>
    %v6639 = stablehlo.multiply %v6634, %v6629 : tensor<64x64x56x56xf32>
    %v6640 = stablehlo.subtract %v6638, %v6639 : tensor<64x64x56x56xf32>
    %v6641 = stablehlo.multiply %v6632, %v6640 : tensor<64x64x56x56xf32>
    %v6642 = stablehlo.reshape %v6641 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6643 = stablehlo.reshape %v6642 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6644 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v6645 = stablehlo.transpose %v6644, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6646 = stablehlo.convert %v6643 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6647 = stablehlo.convert %v6645 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v6648 = stablehlo.convolution(%v6646, %v6647)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v6649 = stablehlo.convert %v6648 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6650 = stablehlo.reshape %v6649 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6651 = stablehlo.reshape %v6650 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6652 = stablehlo.reshape %v89 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6653 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v6654 = stablehlo.compare GT, %v6652, %v6653 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v6655 = stablehlo.select %v6654, %v6651, %v6653 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v6656 = stablehlo.reshape %v6655 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6657 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6658 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6659 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6660 = stablehlo.broadcast_in_dim %v6658, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6661 = stablehlo.broadcast_in_dim %v6659, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6662 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6663 = stablehlo.add %v6661, %v6662 : tensor<64x64x56x56xf32>
    %v6664 = stablehlo.rsqrt %v6663 : tensor<64x64x56x56xf32>
    %v6665 = stablehlo.subtract %v6657, %v6660 : tensor<64x64x56x56xf32>
    %v6666 = stablehlo.multiply %v6665, %v6664 : tensor<64x64x56x56xf32>
    %v6667 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6668 = stablehlo.reshape %v6656 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6669 = stablehlo.multiply %v6667, %v6668 : tensor<64x64x56x56xf32>
    %v6670 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6671 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v6672 = stablehlo.reduce(%v6669 init: %v6670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6673 = stablehlo.divide %v6672, %v6671 : tensor<64xf32>
    %v6674 = stablehlo.multiply %v6666, %v6669 : tensor<64x64x56x56xf32>
    %v6675 = stablehlo.reduce(%v6674 init: %v6670) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6676 = stablehlo.divide %v6675, %v6671 : tensor<64xf32>
    %v6677 = stablehlo.concatenate %v6673, %v6676, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6678 = stablehlo.concatenate %v74, %v6677, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g1dst = "stablehlo.all_reduce"(%v6678) ({
    ^bb0(%aras1b0g1dst: tensor<f32>, %arbs1b0g1dst: tensor<f32>):
      %aradds1b0g1dst = stablehlo.add %aras1b0g1dst, %arbs1b0g1dst : tensor<f32>
      stablehlo.return %aradds1b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g1dst = stablehlo.divide %arsums1b0g1dst, %arns1b0g1dst : tensor<256xf32>
    %v6679 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6680 = stablehlo.slice %armeans1b0g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6681 = stablehlo.slice %armeans1b0g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6682 = stablehlo.slice %armeans1b0g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6683 = stablehlo.slice %armeans1b0g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6684 = stablehlo.broadcast_in_dim %v6680, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6685 = stablehlo.broadcast_in_dim %v6681, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6686 = stablehlo.broadcast_in_dim %v6682, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6687 = stablehlo.broadcast_in_dim %v6683, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6688 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6689 = stablehlo.add %v6685, %v6688 : tensor<64x64x56x56xf32>
    %v6690 = stablehlo.rsqrt %v6689 : tensor<64x64x56x56xf32>
    %v6691 = stablehlo.subtract %v6679, %v6684 : tensor<64x64x56x56xf32>
    %v6692 = stablehlo.multiply %v6691, %v6690 : tensor<64x64x56x56xf32>
    %v6693 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6694 = stablehlo.reshape %v6656 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6695 = stablehlo.multiply %v6693, %v6694 : tensor<64x64x56x56xf32>
    %v6696 = stablehlo.subtract %v6695, %v6686 : tensor<64x64x56x56xf32>
    %v6697 = stablehlo.multiply %v6692, %v6687 : tensor<64x64x56x56xf32>
    %v6698 = stablehlo.subtract %v6696, %v6697 : tensor<64x64x56x56xf32>
    %v6699 = stablehlo.multiply %v6690, %v6698 : tensor<64x64x56x56xf32>
    %v6700 = stablehlo.reshape %v6699 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6701 = stablehlo.reshape %v6700 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6702 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v6703 = stablehlo.transpose %v6702, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v6704 = stablehlo.convert %v6701 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6705 = stablehlo.convert %v6703 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v6706 = stablehlo.convolution(%v6704, %v6705)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v6707 = stablehlo.convert %v6706 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6708 = stablehlo.reshape %v6707 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6709 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6710 = stablehlo.slice %v204 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6711 = stablehlo.slice %v204 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6712 = stablehlo.broadcast_in_dim %v6710, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6713 = stablehlo.broadcast_in_dim %v6711, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6714 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6715 = stablehlo.add %v6713, %v6714 : tensor<64x256x56x56xf32>
    %v6716 = stablehlo.rsqrt %v6715 : tensor<64x256x56x56xf32>
    %v6717 = stablehlo.subtract %v6709, %v6712 : tensor<64x256x56x56xf32>
    %v6718 = stablehlo.multiply %v6717, %v6716 : tensor<64x256x56x56xf32>
    %v6719 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6720 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6721 = stablehlo.multiply %v6719, %v6720 : tensor<64x256x56x56xf32>
    %v6722 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6723 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v6724 = stablehlo.reduce(%v6721 init: %v6722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6725 = stablehlo.divide %v6724, %v6723 : tensor<256xf32>
    %v6726 = stablehlo.multiply %v6718, %v6721 : tensor<64x256x56x56xf32>
    %v6727 = stablehlo.reduce(%v6726 init: %v6722) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6728 = stablehlo.divide %v6727, %v6723 : tensor<256xf32>
    %v6729 = stablehlo.concatenate %v6725, %v6728, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v6730 = stablehlo.concatenate %v204, %v6729, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums1b0gpdst = "stablehlo.all_reduce"(%v6730) ({
    ^bb0(%aras1b0gpdst: tensor<f32>, %arbs1b0gpdst: tensor<f32>):
      %aradds1b0gpdst = stablehlo.add %aras1b0gpdst, %arbs1b0gpdst : tensor<f32>
      stablehlo.return %aradds1b0gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns1b0gpdst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans1b0gpdst = stablehlo.divide %arsums1b0gpdst, %arns1b0gpdst : tensor<1024xf32>
    %v6731 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6732 = stablehlo.slice %armeans1b0gpdst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6733 = stablehlo.slice %armeans1b0gpdst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6734 = stablehlo.slice %armeans1b0gpdst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6735 = stablehlo.slice %armeans1b0gpdst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v6736 = stablehlo.broadcast_in_dim %v6732, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6737 = stablehlo.broadcast_in_dim %v6733, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6738 = stablehlo.broadcast_in_dim %v6734, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6739 = stablehlo.broadcast_in_dim %v6735, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6740 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6741 = stablehlo.add %v6737, %v6740 : tensor<64x256x56x56xf32>
    %v6742 = stablehlo.rsqrt %v6741 : tensor<64x256x56x56xf32>
    %v6743 = stablehlo.subtract %v6731, %v6736 : tensor<64x256x56x56xf32>
    %v6744 = stablehlo.multiply %v6743, %v6742 : tensor<64x256x56x56xf32>
    %v6745 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6746 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6747 = stablehlo.multiply %v6745, %v6746 : tensor<64x256x56x56xf32>
    %v6748 = stablehlo.subtract %v6747, %v6738 : tensor<64x256x56x56xf32>
    %v6749 = stablehlo.multiply %v6744, %v6739 : tensor<64x256x56x56xf32>
    %v6750 = stablehlo.subtract %v6748, %v6749 : tensor<64x256x56x56xf32>
    %v6751 = stablehlo.multiply %v6742, %v6750 : tensor<64x256x56x56xf32>
    %v6752 = stablehlo.reshape %v6751 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v6753 = stablehlo.reshape %v6752 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6754 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v6755 = stablehlo.transpose %v6754, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v6756 = stablehlo.convert %v6753 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v6757 = stablehlo.convert %v6755 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v6758 = stablehlo.convolution(%v6756, %v6757)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v6759 = stablehlo.convert %v6758 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v6760 = stablehlo.reshape %v6759 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6761 = stablehlo.reshape %v6708 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6762 = stablehlo.reshape %v6760 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6763 = stablehlo.add %v6761, %v6762 : tensor<64x64x56x56xf32>
    %v6764 = stablehlo.reshape %v6763 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v6765 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6766 = stablehlo.reshape %v6700 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6767 = stablehlo.transpose %v6765, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6768 = stablehlo.transpose %v6766, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6769 = stablehlo.convert %v6767 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6770 = stablehlo.convert %v6768 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6771 = stablehlo.convolution(%v6769, %v6770)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x1x1xbf16>
    %v6772 = stablehlo.convert %v6771 : (tensor<64x64x1x1xbf16>) -> tensor<64x64x1x1xf32>
    %v6773 = stablehlo.transpose %v6772, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v6774 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6775 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6776 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6777 = stablehlo.broadcast_in_dim %v6775, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6778 = stablehlo.broadcast_in_dim %v6776, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6779 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6780 = stablehlo.add %v6778, %v6779 : tensor<64x64x56x56xf32>
    %v6781 = stablehlo.rsqrt %v6780 : tensor<64x64x56x56xf32>
    %v6782 = stablehlo.subtract %v6774, %v6777 : tensor<64x64x56x56xf32>
    %v6783 = stablehlo.multiply %v6782, %v6781 : tensor<64x64x56x56xf32>
    %v6784 = stablehlo.reshape %v6656 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6785 = stablehlo.multiply %v6784, %v6783 : tensor<64x64x56x56xf32>
    %v6786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6787 = stablehlo.reduce(%v6785 init: %v6786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6788 = stablehlo.reshape %v6656 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6789 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6790 = stablehlo.reduce(%v6788 init: %v6789) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6791 = stablehlo.reshape %v91 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6792 = stablehlo.reshape %v6642 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6793 = stablehlo.transpose %v6791, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6794 = stablehlo.transpose %v6792, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6795 = stablehlo.convert %v6793 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6796 = stablehlo.convert %v6794 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6797 = stablehlo.convolution(%v6795, %v6796)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v6798 = stablehlo.convert %v6797 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v6799 = stablehlo.transpose %v6798, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v6800 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6801 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6802 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6803 = stablehlo.broadcast_in_dim %v6801, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6804 = stablehlo.broadcast_in_dim %v6802, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v6805 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v6806 = stablehlo.add %v6804, %v6805 : tensor<64x64x56x56xf32>
    %v6807 = stablehlo.rsqrt %v6806 : tensor<64x64x56x56xf32>
    %v6808 = stablehlo.subtract %v6800, %v6803 : tensor<64x64x56x56xf32>
    %v6809 = stablehlo.multiply %v6808, %v6807 : tensor<64x64x56x56xf32>
    %v6810 = stablehlo.reshape %v6598 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6811 = stablehlo.multiply %v6810, %v6809 : tensor<64x64x56x56xf32>
    %v6812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6813 = stablehlo.reduce(%v6811 init: %v6812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6814 = stablehlo.reshape %v6598 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6816 = stablehlo.reduce(%v6814 init: %v6815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v6817 = stablehlo.reshape %v135 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6818 = stablehlo.reshape %v6584 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6819 = stablehlo.transpose %v6817, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6820 = stablehlo.transpose %v6818, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6821 = stablehlo.convert %v6819 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6822 = stablehlo.convert %v6820 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6823 = stablehlo.convolution(%v6821, %v6822)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v6824 = stablehlo.convert %v6823 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v6825 = stablehlo.transpose %v6824, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6826 = stablehlo.reshape %v143 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6827 = stablehlo.slice %v162 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6828 = stablehlo.slice %v162 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6829 = stablehlo.broadcast_in_dim %v6827, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6830 = stablehlo.broadcast_in_dim %v6828, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6831 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6832 = stablehlo.add %v6830, %v6831 : tensor<64x256x56x56xf32>
    %v6833 = stablehlo.rsqrt %v6832 : tensor<64x256x56x56xf32>
    %v6834 = stablehlo.subtract %v6826, %v6829 : tensor<64x256x56x56xf32>
    %v6835 = stablehlo.multiply %v6834, %v6833 : tensor<64x256x56x56xf32>
    %v6836 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6837 = stablehlo.multiply %v6836, %v6835 : tensor<64x256x56x56xf32>
    %v6838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6839 = stablehlo.reduce(%v6837 init: %v6838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6840 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6841 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6842 = stablehlo.reduce(%v6840 init: %v6841) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6843 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6844 = stablehlo.reshape %v6752 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6845 = stablehlo.transpose %v6843, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v6846 = stablehlo.transpose %v6844, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v6847 = stablehlo.convert %v6845 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v6848 = stablehlo.convert %v6846 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v6849 = stablehlo.convolution(%v6847, %v6848)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v6850 = stablehlo.convert %v6849 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v6851 = stablehlo.transpose %v6850, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v6852 = stablehlo.reshape %v185 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6853 = stablehlo.slice %v204 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6854 = stablehlo.slice %v204 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6855 = stablehlo.broadcast_in_dim %v6853, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6856 = stablehlo.broadcast_in_dim %v6854, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v6857 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v6858 = stablehlo.add %v6856, %v6857 : tensor<64x256x56x56xf32>
    %v6859 = stablehlo.rsqrt %v6858 : tensor<64x256x56x56xf32>
    %v6860 = stablehlo.subtract %v6852, %v6855 : tensor<64x256x56x56xf32>
    %v6861 = stablehlo.multiply %v6860, %v6859 : tensor<64x256x56x56xf32>
    %v6862 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6863 = stablehlo.multiply %v6862, %v6861 : tensor<64x256x56x56xf32>
    %v6864 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6865 = stablehlo.reduce(%v6863 init: %v6864) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6866 = stablehlo.reshape %v6540 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v6867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6868 = stablehlo.reduce(%v6866 init: %v6867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v6869 = stablehlo.reshape %v43 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6870 = stablehlo.reshape %v6764 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v6871 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6872 = "stablehlo.select_and_scatter"(%v6869, %v6870, %v6871) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x112x112xf32>
    %v6873 = stablehlo.reshape %v6872 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6874 = stablehlo.reshape %v6873 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6875 = stablehlo.reshape %v41 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6876 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v6877 = stablehlo.compare GT, %v6875, %v6876 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v6878 = stablehlo.select %v6877, %v6874, %v6876 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v6879 = stablehlo.reshape %v6878 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6880 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6881 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6882 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6883 = stablehlo.broadcast_in_dim %v6881, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6884 = stablehlo.broadcast_in_dim %v6882, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6885 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6886 = stablehlo.add %v6884, %v6885 : tensor<64x64x112x112xf32>
    %v6887 = stablehlo.rsqrt %v6886 : tensor<64x64x112x112xf32>
    %v6888 = stablehlo.subtract %v6880, %v6883 : tensor<64x64x112x112xf32>
    %v6889 = stablehlo.multiply %v6888, %v6887 : tensor<64x64x112x112xf32>
    %v6890 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6891 = stablehlo.reshape %v6879 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6892 = stablehlo.multiply %v6890, %v6891 : tensor<64x64x112x112xf32>
    %v6893 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6894 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v6895 = stablehlo.reduce(%v6892 init: %v6893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6896 = stablehlo.divide %v6895, %v6894 : tensor<64xf32>
    %v6897 = stablehlo.multiply %v6889, %v6892 : tensor<64x64x112x112xf32>
    %v6898 = stablehlo.reduce(%v6897 init: %v6893) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6899 = stablehlo.divide %v6898, %v6894 : tensor<64xf32>
    %v6900 = stablehlo.concatenate %v6896, %v6899, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v6901 = stablehlo.concatenate %v26, %v6900, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsumsgdst = "stablehlo.all_reduce"(%v6901) ({
    ^bb0(%arasgdst: tensor<f32>, %arbsgdst: tensor<f32>):
      %araddsgdst = stablehlo.add %arasgdst, %arbsgdst : tensor<f32>
      stablehlo.return %araddsgdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnsgdst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeansgdst = stablehlo.divide %arsumsgdst, %arnsgdst : tensor<256xf32>
    %v6902 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6903 = stablehlo.slice %armeansgdst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v6904 = stablehlo.slice %armeansgdst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v6905 = stablehlo.slice %armeansgdst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v6906 = stablehlo.slice %armeansgdst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v6907 = stablehlo.broadcast_in_dim %v6903, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6908 = stablehlo.broadcast_in_dim %v6904, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6909 = stablehlo.broadcast_in_dim %v6905, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6910 = stablehlo.broadcast_in_dim %v6906, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6911 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6912 = stablehlo.add %v6908, %v6911 : tensor<64x64x112x112xf32>
    %v6913 = stablehlo.rsqrt %v6912 : tensor<64x64x112x112xf32>
    %v6914 = stablehlo.subtract %v6902, %v6907 : tensor<64x64x112x112xf32>
    %v6915 = stablehlo.multiply %v6914, %v6913 : tensor<64x64x112x112xf32>
    %v6916 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6917 = stablehlo.reshape %v6879 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6918 = stablehlo.multiply %v6916, %v6917 : tensor<64x64x112x112xf32>
    %v6919 = stablehlo.subtract %v6918, %v6909 : tensor<64x64x112x112xf32>
    %v6920 = stablehlo.multiply %v6915, %v6910 : tensor<64x64x112x112xf32>
    %v6921 = stablehlo.subtract %v6919, %v6920 : tensor<64x64x112x112xf32>
    %v6922 = stablehlo.multiply %v6913, %v6921 : tensor<64x64x112x112xf32>
    %v6923 = stablehlo.reshape %v6922 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v6924 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v6925 = stablehlo.reshape %v6923 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6926 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6927 = stablehlo.pad %v6925, %v6926, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x224x224xf32>
    %v6928 = stablehlo.transpose %v6924, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v6929 = stablehlo.transpose %v6927, dims = [1, 0, 2, 3] : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xf32>
    %v6930 = stablehlo.convert %v6928 : (tensor<3x64x224x224xf32>) -> tensor<3x64x224x224xbf16>
    %v6931 = stablehlo.convert %v6929 : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xbf16>
    %v6932 = stablehlo.convolution(%v6930, %v6931)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xbf16>, tensor<64x64x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v6933 = stablehlo.convert %v6932 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v6934 = stablehlo.transpose %v6933, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v6935 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6936 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6937 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6938 = stablehlo.broadcast_in_dim %v6936, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6939 = stablehlo.broadcast_in_dim %v6937, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v6940 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v6941 = stablehlo.add %v6939, %v6940 : tensor<64x64x112x112xf32>
    %v6942 = stablehlo.rsqrt %v6941 : tensor<64x64x112x112xf32>
    %v6943 = stablehlo.subtract %v6935, %v6938 : tensor<64x64x112x112xf32>
    %v6944 = stablehlo.multiply %v6943, %v6942 : tensor<64x64x112x112xf32>
    %v6945 = stablehlo.reshape %v6879 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6946 = stablehlo.multiply %v6945, %v6944 : tensor<64x64x112x112xf32>
    %v6947 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6948 = stablehlo.reduce(%v6946 init: %v6947) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6949 = stablehlo.reshape %v6879 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v6950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6951 = stablehlo.reduce(%v6949 init: %v6950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v6952 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6953 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6954 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6955 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6956 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6957 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6958 = stablehlo.slice %v162 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6959 = stablehlo.slice %v162 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6960 = stablehlo.slice %v204 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6961 = stablehlo.slice %v204 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6962 = stablehlo.slice %v249 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6963 = stablehlo.slice %v249 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6964 = stablehlo.slice %v293 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6965 = stablehlo.slice %v293 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6966 = stablehlo.slice %v337 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6967 = stablehlo.slice %v337 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6968 = stablehlo.slice %v387 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6969 = stablehlo.slice %v387 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6970 = stablehlo.slice %v431 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v6971 = stablehlo.slice %v431 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v6972 = stablehlo.slice %v475 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v6973 = stablehlo.slice %v475 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v6974 = stablehlo.slice %v525 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6975 = stablehlo.slice %v525 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6976 = stablehlo.slice %v569 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6977 = stablehlo.slice %v569 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6978 = stablehlo.slice %v613 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6979 = stablehlo.slice %v613 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6980 = stablehlo.slice %v655 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6981 = stablehlo.slice %v655 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6982 = stablehlo.slice %v700 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6983 = stablehlo.slice %v700 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6984 = stablehlo.slice %v744 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6985 = stablehlo.slice %v744 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6986 = stablehlo.slice %v788 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6987 = stablehlo.slice %v788 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6988 = stablehlo.slice %v838 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6989 = stablehlo.slice %v838 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6990 = stablehlo.slice %v882 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6991 = stablehlo.slice %v882 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6992 = stablehlo.slice %v926 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6993 = stablehlo.slice %v926 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6994 = stablehlo.slice %v976 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6995 = stablehlo.slice %v976 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6996 = stablehlo.slice %v1020 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v6997 = stablehlo.slice %v1020 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v6998 = stablehlo.slice %v1064 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v6999 = stablehlo.slice %v1064 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7000 = stablehlo.slice %v1114 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7001 = stablehlo.slice %v1114 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7002 = stablehlo.slice %v1158 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7003 = stablehlo.slice %v1158 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7004 = stablehlo.slice %v1202 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7005 = stablehlo.slice %v1202 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7006 = stablehlo.slice %v1244 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7007 = stablehlo.slice %v1244 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7008 = stablehlo.slice %v1289 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7009 = stablehlo.slice %v1289 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7010 = stablehlo.slice %v1333 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7011 = stablehlo.slice %v1333 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7012 = stablehlo.slice %v1377 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7013 = stablehlo.slice %v1377 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7014 = stablehlo.slice %v1427 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7015 = stablehlo.slice %v1427 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7016 = stablehlo.slice %v1471 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7017 = stablehlo.slice %v1471 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7018 = stablehlo.slice %v1515 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7019 = stablehlo.slice %v1515 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7020 = stablehlo.slice %v1565 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7021 = stablehlo.slice %v1565 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7022 = stablehlo.slice %v1609 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7023 = stablehlo.slice %v1609 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7024 = stablehlo.slice %v1653 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7025 = stablehlo.slice %v1653 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7026 = stablehlo.slice %v1703 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7027 = stablehlo.slice %v1703 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7028 = stablehlo.slice %v1747 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7029 = stablehlo.slice %v1747 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7030 = stablehlo.slice %v1791 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7031 = stablehlo.slice %v1791 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7032 = stablehlo.slice %v1841 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7033 = stablehlo.slice %v1841 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7034 = stablehlo.slice %v1885 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v7035 = stablehlo.slice %v1885 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v7036 = stablehlo.slice %v1929 [0:1024] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7037 = stablehlo.slice %v1929 [1024:2048] : (tensor<2048xf32>) -> tensor<1024xf32>
    %v7038 = stablehlo.slice %v1979 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7039 = stablehlo.slice %v1979 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7040 = stablehlo.slice %v2023 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7041 = stablehlo.slice %v2023 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7042 = stablehlo.slice %v2067 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7043 = stablehlo.slice %v2067 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7044 = stablehlo.slice %v2109 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7045 = stablehlo.slice %v2109 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7046 = stablehlo.slice %v2154 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7047 = stablehlo.slice %v2154 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7048 = stablehlo.slice %v2198 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7049 = stablehlo.slice %v2198 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7050 = stablehlo.slice %v2242 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7051 = stablehlo.slice %v2242 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7052 = stablehlo.slice %v2292 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7053 = stablehlo.slice %v2292 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7054 = stablehlo.slice %v2336 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7055 = stablehlo.slice %v2336 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v7056 = stablehlo.slice %v2380 [0:2048] : (tensor<4096xf32>) -> tensor<2048xf32>
    %v7057 = stablehlo.slice %v2380 [2048:4096] : (tensor<4096xf32>) -> tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v6934) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v7058 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v7059 = stablehlo.multiply %v7058, %sW : tensor<64x3x7x7xf32>
    %v7060 = stablehlo.add %v7059, %armeansW : tensor<64x3x7x7xf32>
    %v7061 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v7062 = stablehlo.multiply %v7061, %sWv : tensor<64x3x7x7xf32>
    %v7063 = stablehlo.add %v7062, %v7060 : tensor<64x3x7x7xf32>
    %v7064 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v7065 = stablehlo.multiply %v7064, %v7063 : tensor<64x3x7x7xf32>
    %v7066 = stablehlo.subtract %sW, %v7065 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v6948) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v7067 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7068 = stablehlo.multiply %v7067, %sg : tensor<64xf32>
    %v7069 = stablehlo.add %v7068, %armeansg : tensor<64xf32>
    %v7070 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7071 = stablehlo.multiply %v7070, %sgv : tensor<64xf32>
    %v7072 = stablehlo.add %v7071, %v7069 : tensor<64xf32>
    %v7073 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7074 = stablehlo.multiply %v7073, %v7072 : tensor<64xf32>
    %v7075 = stablehlo.subtract %sg, %v7074 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v6951) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v7076 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7077 = stablehlo.multiply %v7076, %sbt : tensor<64xf32>
    %v7078 = stablehlo.add %v7077, %armeansbt : tensor<64xf32>
    %v7079 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7080 = stablehlo.multiply %v7079, %sbtv : tensor<64xf32>
    %v7081 = stablehlo.add %v7080, %v7078 : tensor<64xf32>
    %v7082 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7083 = stablehlo.multiply %v7082, %v7081 : tensor<64xf32>
    %v7084 = stablehlo.subtract %sbt, %v7083 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v6773) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x1x1xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x1x1xf32>
    %v7085 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v7086 = stablehlo.multiply %v7085, %s1b0W1 : tensor<64x64x1x1xf32>
    %v7087 = stablehlo.add %v7086, %armeans1b0W1 : tensor<64x64x1x1xf32>
    %v7088 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v7089 = stablehlo.multiply %v7088, %s1b0W1v : tensor<64x64x1x1xf32>
    %v7090 = stablehlo.add %v7089, %v7087 : tensor<64x64x1x1xf32>
    %v7091 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v7092 = stablehlo.multiply %v7091, %v7090 : tensor<64x64x1x1xf32>
    %v7093 = stablehlo.subtract %s1b0W1, %v7092 : tensor<64x64x1x1xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v6787) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v7094 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7095 = stablehlo.multiply %v7094, %s1b0g1 : tensor<64xf32>
    %v7096 = stablehlo.add %v7095, %armeans1b0g1 : tensor<64xf32>
    %v7097 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7098 = stablehlo.multiply %v7097, %s1b0g1v : tensor<64xf32>
    %v7099 = stablehlo.add %v7098, %v7096 : tensor<64xf32>
    %v7100 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7101 = stablehlo.multiply %v7100, %v7099 : tensor<64xf32>
    %v7102 = stablehlo.subtract %s1b0g1, %v7101 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v6790) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v7103 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7104 = stablehlo.multiply %v7103, %s1b0bt1 : tensor<64xf32>
    %v7105 = stablehlo.add %v7104, %armeans1b0bt1 : tensor<64xf32>
    %v7106 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7107 = stablehlo.multiply %v7106, %s1b0bt1v : tensor<64xf32>
    %v7108 = stablehlo.add %v7107, %v7105 : tensor<64xf32>
    %v7109 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7110 = stablehlo.multiply %v7109, %v7108 : tensor<64xf32>
    %v7111 = stablehlo.subtract %s1b0bt1, %v7110 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v6799) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v7112 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7113 = stablehlo.multiply %v7112, %s1b0W2 : tensor<64x64x3x3xf32>
    %v7114 = stablehlo.add %v7113, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v7115 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7116 = stablehlo.multiply %v7115, %s1b0W2v : tensor<64x64x3x3xf32>
    %v7117 = stablehlo.add %v7116, %v7114 : tensor<64x64x3x3xf32>
    %v7118 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7119 = stablehlo.multiply %v7118, %v7117 : tensor<64x64x3x3xf32>
    %v7120 = stablehlo.subtract %s1b0W2, %v7119 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v6813) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v7121 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7122 = stablehlo.multiply %v7121, %s1b0g2 : tensor<64xf32>
    %v7123 = stablehlo.add %v7122, %armeans1b0g2 : tensor<64xf32>
    %v7124 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7125 = stablehlo.multiply %v7124, %s1b0g2v : tensor<64xf32>
    %v7126 = stablehlo.add %v7125, %v7123 : tensor<64xf32>
    %v7127 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7128 = stablehlo.multiply %v7127, %v7126 : tensor<64xf32>
    %v7129 = stablehlo.subtract %s1b0g2, %v7128 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v6816) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v7130 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7131 = stablehlo.multiply %v7130, %s1b0bt2 : tensor<64xf32>
    %v7132 = stablehlo.add %v7131, %armeans1b0bt2 : tensor<64xf32>
    %v7133 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7134 = stablehlo.multiply %v7133, %s1b0bt2v : tensor<64xf32>
    %v7135 = stablehlo.add %v7134, %v7132 : tensor<64xf32>
    %v7136 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7137 = stablehlo.multiply %v7136, %v7135 : tensor<64xf32>
    %v7138 = stablehlo.subtract %s1b0bt2, %v7137 : tensor<64xf32>
    %arsums1b0W3 = "stablehlo.all_reduce"(%v6825) ({
    ^bb0(%aras1b0W3: tensor<f32>, %arbs1b0W3: tensor<f32>):
      %aradds1b0W3 = stablehlo.add %aras1b0W3, %arbs1b0W3 : tensor<f32>
      stablehlo.return %aradds1b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0W3 = stablehlo.divide %arsums1b0W3, %arns1b0W3 : tensor<256x64x1x1xf32>
    %v7139 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7140 = stablehlo.multiply %v7139, %s1b0W3 : tensor<256x64x1x1xf32>
    %v7141 = stablehlo.add %v7140, %armeans1b0W3 : tensor<256x64x1x1xf32>
    %v7142 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7143 = stablehlo.multiply %v7142, %s1b0W3v : tensor<256x64x1x1xf32>
    %v7144 = stablehlo.add %v7143, %v7141 : tensor<256x64x1x1xf32>
    %v7145 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7146 = stablehlo.multiply %v7145, %v7144 : tensor<256x64x1x1xf32>
    %v7147 = stablehlo.subtract %s1b0W3, %v7146 : tensor<256x64x1x1xf32>
    %arsums1b0g3 = "stablehlo.all_reduce"(%v6839) ({
    ^bb0(%aras1b0g3: tensor<f32>, %arbs1b0g3: tensor<f32>):
      %aradds1b0g3 = stablehlo.add %aras1b0g3, %arbs1b0g3 : tensor<f32>
      stablehlo.return %aradds1b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3 = stablehlo.divide %arsums1b0g3, %arns1b0g3 : tensor<256xf32>
    %v7148 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7149 = stablehlo.multiply %v7148, %s1b0g3 : tensor<256xf32>
    %v7150 = stablehlo.add %v7149, %armeans1b0g3 : tensor<256xf32>
    %v7151 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7152 = stablehlo.multiply %v7151, %s1b0g3v : tensor<256xf32>
    %v7153 = stablehlo.add %v7152, %v7150 : tensor<256xf32>
    %v7154 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7155 = stablehlo.multiply %v7154, %v7153 : tensor<256xf32>
    %v7156 = stablehlo.subtract %s1b0g3, %v7155 : tensor<256xf32>
    %arsums1b0bt3 = "stablehlo.all_reduce"(%v6842) ({
    ^bb0(%aras1b0bt3: tensor<f32>, %arbs1b0bt3: tensor<f32>):
      %aradds1b0bt3 = stablehlo.add %aras1b0bt3, %arbs1b0bt3 : tensor<f32>
      stablehlo.return %aradds1b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0bt3 = stablehlo.divide %arsums1b0bt3, %arns1b0bt3 : tensor<256xf32>
    %v7157 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7158 = stablehlo.multiply %v7157, %s1b0bt3 : tensor<256xf32>
    %v7159 = stablehlo.add %v7158, %armeans1b0bt3 : tensor<256xf32>
    %v7160 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7161 = stablehlo.multiply %v7160, %s1b0bt3v : tensor<256xf32>
    %v7162 = stablehlo.add %v7161, %v7159 : tensor<256xf32>
    %v7163 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7164 = stablehlo.multiply %v7163, %v7162 : tensor<256xf32>
    %v7165 = stablehlo.subtract %s1b0bt3, %v7164 : tensor<256xf32>
    %arsums1b0Wp = "stablehlo.all_reduce"(%v6851) ({
    ^bb0(%aras1b0Wp: tensor<f32>, %arbs1b0Wp: tensor<f32>):
      %aradds1b0Wp = stablehlo.add %aras1b0Wp, %arbs1b0Wp : tensor<f32>
      stablehlo.return %aradds1b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0Wp = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0Wp = stablehlo.divide %arsums1b0Wp, %arns1b0Wp : tensor<256x64x1x1xf32>
    %v7166 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7167 = stablehlo.multiply %v7166, %s1b0Wp : tensor<256x64x1x1xf32>
    %v7168 = stablehlo.add %v7167, %armeans1b0Wp : tensor<256x64x1x1xf32>
    %v7169 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7170 = stablehlo.multiply %v7169, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v7171 = stablehlo.add %v7170, %v7168 : tensor<256x64x1x1xf32>
    %v7172 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7173 = stablehlo.multiply %v7172, %v7171 : tensor<256x64x1x1xf32>
    %v7174 = stablehlo.subtract %s1b0Wp, %v7173 : tensor<256x64x1x1xf32>
    %arsums1b0gp = "stablehlo.all_reduce"(%v6865) ({
    ^bb0(%aras1b0gp: tensor<f32>, %arbs1b0gp: tensor<f32>):
      %aradds1b0gp = stablehlo.add %aras1b0gp, %arbs1b0gp : tensor<f32>
      stablehlo.return %aradds1b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gp = stablehlo.divide %arsums1b0gp, %arns1b0gp : tensor<256xf32>
    %v7175 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7176 = stablehlo.multiply %v7175, %s1b0gp : tensor<256xf32>
    %v7177 = stablehlo.add %v7176, %armeans1b0gp : tensor<256xf32>
    %v7178 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7179 = stablehlo.multiply %v7178, %s1b0gpv : tensor<256xf32>
    %v7180 = stablehlo.add %v7179, %v7177 : tensor<256xf32>
    %v7181 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7182 = stablehlo.multiply %v7181, %v7180 : tensor<256xf32>
    %v7183 = stablehlo.subtract %s1b0gp, %v7182 : tensor<256xf32>
    %arsums1b0btp = "stablehlo.all_reduce"(%v6868) ({
    ^bb0(%aras1b0btp: tensor<f32>, %arbs1b0btp: tensor<f32>):
      %aradds1b0btp = stablehlo.add %aras1b0btp, %arbs1b0btp : tensor<f32>
      stablehlo.return %aradds1b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0btp = stablehlo.divide %arsums1b0btp, %arns1b0btp : tensor<256xf32>
    %v7184 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7185 = stablehlo.multiply %v7184, %s1b0btp : tensor<256xf32>
    %v7186 = stablehlo.add %v7185, %armeans1b0btp : tensor<256xf32>
    %v7187 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7188 = stablehlo.multiply %v7187, %s1b0btpv : tensor<256xf32>
    %v7189 = stablehlo.add %v7188, %v7186 : tensor<256xf32>
    %v7190 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7191 = stablehlo.multiply %v7190, %v7189 : tensor<256xf32>
    %v7192 = stablehlo.subtract %s1b0btp, %v7191 : tensor<256xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v6465) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x256x1x1xf32>
    %v7193 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7194 = stablehlo.multiply %v7193, %s1b1W1 : tensor<64x256x1x1xf32>
    %v7195 = stablehlo.add %v7194, %armeans1b1W1 : tensor<64x256x1x1xf32>
    %v7196 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7197 = stablehlo.multiply %v7196, %s1b1W1v : tensor<64x256x1x1xf32>
    %v7198 = stablehlo.add %v7197, %v7195 : tensor<64x256x1x1xf32>
    %v7199 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7200 = stablehlo.multiply %v7199, %v7198 : tensor<64x256x1x1xf32>
    %v7201 = stablehlo.subtract %s1b1W1, %v7200 : tensor<64x256x1x1xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v6479) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v7202 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7203 = stablehlo.multiply %v7202, %s1b1g1 : tensor<64xf32>
    %v7204 = stablehlo.add %v7203, %armeans1b1g1 : tensor<64xf32>
    %v7205 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7206 = stablehlo.multiply %v7205, %s1b1g1v : tensor<64xf32>
    %v7207 = stablehlo.add %v7206, %v7204 : tensor<64xf32>
    %v7208 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7209 = stablehlo.multiply %v7208, %v7207 : tensor<64xf32>
    %v7210 = stablehlo.subtract %s1b1g1, %v7209 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v6482) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v7211 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7212 = stablehlo.multiply %v7211, %s1b1bt1 : tensor<64xf32>
    %v7213 = stablehlo.add %v7212, %armeans1b1bt1 : tensor<64xf32>
    %v7214 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7215 = stablehlo.multiply %v7214, %s1b1bt1v : tensor<64xf32>
    %v7216 = stablehlo.add %v7215, %v7213 : tensor<64xf32>
    %v7217 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7218 = stablehlo.multiply %v7217, %v7216 : tensor<64xf32>
    %v7219 = stablehlo.subtract %s1b1bt1, %v7218 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v6491) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v7220 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7221 = stablehlo.multiply %v7220, %s1b1W2 : tensor<64x64x3x3xf32>
    %v7222 = stablehlo.add %v7221, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v7223 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7224 = stablehlo.multiply %v7223, %s1b1W2v : tensor<64x64x3x3xf32>
    %v7225 = stablehlo.add %v7224, %v7222 : tensor<64x64x3x3xf32>
    %v7226 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7227 = stablehlo.multiply %v7226, %v7225 : tensor<64x64x3x3xf32>
    %v7228 = stablehlo.subtract %s1b1W2, %v7227 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v6505) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v7229 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7230 = stablehlo.multiply %v7229, %s1b1g2 : tensor<64xf32>
    %v7231 = stablehlo.add %v7230, %armeans1b1g2 : tensor<64xf32>
    %v7232 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7233 = stablehlo.multiply %v7232, %s1b1g2v : tensor<64xf32>
    %v7234 = stablehlo.add %v7233, %v7231 : tensor<64xf32>
    %v7235 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7236 = stablehlo.multiply %v7235, %v7234 : tensor<64xf32>
    %v7237 = stablehlo.subtract %s1b1g2, %v7236 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v6508) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v7238 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7239 = stablehlo.multiply %v7238, %s1b1bt2 : tensor<64xf32>
    %v7240 = stablehlo.add %v7239, %armeans1b1bt2 : tensor<64xf32>
    %v7241 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7242 = stablehlo.multiply %v7241, %s1b1bt2v : tensor<64xf32>
    %v7243 = stablehlo.add %v7242, %v7240 : tensor<64xf32>
    %v7244 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7245 = stablehlo.multiply %v7244, %v7243 : tensor<64xf32>
    %v7246 = stablehlo.subtract %s1b1bt2, %v7245 : tensor<64xf32>
    %arsums1b1W3 = "stablehlo.all_reduce"(%v6517) ({
    ^bb0(%aras1b1W3: tensor<f32>, %arbs1b1W3: tensor<f32>):
      %aradds1b1W3 = stablehlo.add %aras1b1W3, %arbs1b1W3 : tensor<f32>
      stablehlo.return %aradds1b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b1W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b1W3 = stablehlo.divide %arsums1b1W3, %arns1b1W3 : tensor<256x64x1x1xf32>
    %v7247 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7248 = stablehlo.multiply %v7247, %s1b1W3 : tensor<256x64x1x1xf32>
    %v7249 = stablehlo.add %v7248, %armeans1b1W3 : tensor<256x64x1x1xf32>
    %v7250 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7251 = stablehlo.multiply %v7250, %s1b1W3v : tensor<256x64x1x1xf32>
    %v7252 = stablehlo.add %v7251, %v7249 : tensor<256x64x1x1xf32>
    %v7253 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7254 = stablehlo.multiply %v7253, %v7252 : tensor<256x64x1x1xf32>
    %v7255 = stablehlo.subtract %s1b1W3, %v7254 : tensor<256x64x1x1xf32>
    %arsums1b1g3 = "stablehlo.all_reduce"(%v6531) ({
    ^bb0(%aras1b1g3: tensor<f32>, %arbs1b1g3: tensor<f32>):
      %aradds1b1g3 = stablehlo.add %aras1b1g3, %arbs1b1g3 : tensor<f32>
      stablehlo.return %aradds1b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3 = stablehlo.divide %arsums1b1g3, %arns1b1g3 : tensor<256xf32>
    %v7256 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7257 = stablehlo.multiply %v7256, %s1b1g3 : tensor<256xf32>
    %v7258 = stablehlo.add %v7257, %armeans1b1g3 : tensor<256xf32>
    %v7259 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7260 = stablehlo.multiply %v7259, %s1b1g3v : tensor<256xf32>
    %v7261 = stablehlo.add %v7260, %v7258 : tensor<256xf32>
    %v7262 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7263 = stablehlo.multiply %v7262, %v7261 : tensor<256xf32>
    %v7264 = stablehlo.subtract %s1b1g3, %v7263 : tensor<256xf32>
    %arsums1b1bt3 = "stablehlo.all_reduce"(%v6534) ({
    ^bb0(%aras1b1bt3: tensor<f32>, %arbs1b1bt3: tensor<f32>):
      %aradds1b1bt3 = stablehlo.add %aras1b1bt3, %arbs1b1bt3 : tensor<f32>
      stablehlo.return %aradds1b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1bt3 = stablehlo.divide %arsums1b1bt3, %arns1b1bt3 : tensor<256xf32>
    %v7265 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7266 = stablehlo.multiply %v7265, %s1b1bt3 : tensor<256xf32>
    %v7267 = stablehlo.add %v7266, %armeans1b1bt3 : tensor<256xf32>
    %v7268 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7269 = stablehlo.multiply %v7268, %s1b1bt3v : tensor<256xf32>
    %v7270 = stablehlo.add %v7269, %v7267 : tensor<256xf32>
    %v7271 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7272 = stablehlo.multiply %v7271, %v7270 : tensor<256xf32>
    %v7273 = stablehlo.subtract %s1b1bt3, %v7272 : tensor<256xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v6209) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x256x1x1xf32>
    %v7274 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7275 = stablehlo.multiply %v7274, %s1b2W1 : tensor<64x256x1x1xf32>
    %v7276 = stablehlo.add %v7275, %armeans1b2W1 : tensor<64x256x1x1xf32>
    %v7277 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7278 = stablehlo.multiply %v7277, %s1b2W1v : tensor<64x256x1x1xf32>
    %v7279 = stablehlo.add %v7278, %v7276 : tensor<64x256x1x1xf32>
    %v7280 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v7281 = stablehlo.multiply %v7280, %v7279 : tensor<64x256x1x1xf32>
    %v7282 = stablehlo.subtract %s1b2W1, %v7281 : tensor<64x256x1x1xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v6223) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v7283 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7284 = stablehlo.multiply %v7283, %s1b2g1 : tensor<64xf32>
    %v7285 = stablehlo.add %v7284, %armeans1b2g1 : tensor<64xf32>
    %v7286 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7287 = stablehlo.multiply %v7286, %s1b2g1v : tensor<64xf32>
    %v7288 = stablehlo.add %v7287, %v7285 : tensor<64xf32>
    %v7289 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7290 = stablehlo.multiply %v7289, %v7288 : tensor<64xf32>
    %v7291 = stablehlo.subtract %s1b2g1, %v7290 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v6226) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v7292 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7293 = stablehlo.multiply %v7292, %s1b2bt1 : tensor<64xf32>
    %v7294 = stablehlo.add %v7293, %armeans1b2bt1 : tensor<64xf32>
    %v7295 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7296 = stablehlo.multiply %v7295, %s1b2bt1v : tensor<64xf32>
    %v7297 = stablehlo.add %v7296, %v7294 : tensor<64xf32>
    %v7298 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7299 = stablehlo.multiply %v7298, %v7297 : tensor<64xf32>
    %v7300 = stablehlo.subtract %s1b2bt1, %v7299 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v6235) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v7301 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7302 = stablehlo.multiply %v7301, %s1b2W2 : tensor<64x64x3x3xf32>
    %v7303 = stablehlo.add %v7302, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v7304 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7305 = stablehlo.multiply %v7304, %s1b2W2v : tensor<64x64x3x3xf32>
    %v7306 = stablehlo.add %v7305, %v7303 : tensor<64x64x3x3xf32>
    %v7307 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v7308 = stablehlo.multiply %v7307, %v7306 : tensor<64x64x3x3xf32>
    %v7309 = stablehlo.subtract %s1b2W2, %v7308 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v6249) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v7310 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7311 = stablehlo.multiply %v7310, %s1b2g2 : tensor<64xf32>
    %v7312 = stablehlo.add %v7311, %armeans1b2g2 : tensor<64xf32>
    %v7313 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7314 = stablehlo.multiply %v7313, %s1b2g2v : tensor<64xf32>
    %v7315 = stablehlo.add %v7314, %v7312 : tensor<64xf32>
    %v7316 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7317 = stablehlo.multiply %v7316, %v7315 : tensor<64xf32>
    %v7318 = stablehlo.subtract %s1b2g2, %v7317 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v6252) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v7319 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7320 = stablehlo.multiply %v7319, %s1b2bt2 : tensor<64xf32>
    %v7321 = stablehlo.add %v7320, %armeans1b2bt2 : tensor<64xf32>
    %v7322 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7323 = stablehlo.multiply %v7322, %s1b2bt2v : tensor<64xf32>
    %v7324 = stablehlo.add %v7323, %v7321 : tensor<64xf32>
    %v7325 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v7326 = stablehlo.multiply %v7325, %v7324 : tensor<64xf32>
    %v7327 = stablehlo.subtract %s1b2bt2, %v7326 : tensor<64xf32>
    %arsums1b2W3 = "stablehlo.all_reduce"(%v6261) ({
    ^bb0(%aras1b2W3: tensor<f32>, %arbs1b2W3: tensor<f32>):
      %aradds1b2W3 = stablehlo.add %aras1b2W3, %arbs1b2W3 : tensor<f32>
      stablehlo.return %aradds1b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b2W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b2W3 = stablehlo.divide %arsums1b2W3, %arns1b2W3 : tensor<256x64x1x1xf32>
    %v7328 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7329 = stablehlo.multiply %v7328, %s1b2W3 : tensor<256x64x1x1xf32>
    %v7330 = stablehlo.add %v7329, %armeans1b2W3 : tensor<256x64x1x1xf32>
    %v7331 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7332 = stablehlo.multiply %v7331, %s1b2W3v : tensor<256x64x1x1xf32>
    %v7333 = stablehlo.add %v7332, %v7330 : tensor<256x64x1x1xf32>
    %v7334 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v7335 = stablehlo.multiply %v7334, %v7333 : tensor<256x64x1x1xf32>
    %v7336 = stablehlo.subtract %s1b2W3, %v7335 : tensor<256x64x1x1xf32>
    %arsums1b2g3 = "stablehlo.all_reduce"(%v6275) ({
    ^bb0(%aras1b2g3: tensor<f32>, %arbs1b2g3: tensor<f32>):
      %aradds1b2g3 = stablehlo.add %aras1b2g3, %arbs1b2g3 : tensor<f32>
      stablehlo.return %aradds1b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3 = stablehlo.divide %arsums1b2g3, %arns1b2g3 : tensor<256xf32>
    %v7337 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7338 = stablehlo.multiply %v7337, %s1b2g3 : tensor<256xf32>
    %v7339 = stablehlo.add %v7338, %armeans1b2g3 : tensor<256xf32>
    %v7340 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7341 = stablehlo.multiply %v7340, %s1b2g3v : tensor<256xf32>
    %v7342 = stablehlo.add %v7341, %v7339 : tensor<256xf32>
    %v7343 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7344 = stablehlo.multiply %v7343, %v7342 : tensor<256xf32>
    %v7345 = stablehlo.subtract %s1b2g3, %v7344 : tensor<256xf32>
    %arsums1b2bt3 = "stablehlo.all_reduce"(%v6278) ({
    ^bb0(%aras1b2bt3: tensor<f32>, %arbs1b2bt3: tensor<f32>):
      %aradds1b2bt3 = stablehlo.add %aras1b2bt3, %arbs1b2bt3 : tensor<f32>
      stablehlo.return %aradds1b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2bt3 = stablehlo.divide %arsums1b2bt3, %arns1b2bt3 : tensor<256xf32>
    %v7346 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7347 = stablehlo.multiply %v7346, %s1b2bt3 : tensor<256xf32>
    %v7348 = stablehlo.add %v7347, %armeans1b2bt3 : tensor<256xf32>
    %v7349 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7350 = stablehlo.multiply %v7349, %s1b2bt3v : tensor<256xf32>
    %v7351 = stablehlo.add %v7350, %v7348 : tensor<256xf32>
    %v7352 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7353 = stablehlo.multiply %v7352, %v7351 : tensor<256xf32>
    %v7354 = stablehlo.subtract %s1b2bt3, %v7353 : tensor<256xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v5923) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x256x1x1xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x256x1x1xf32>
    %v7355 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v7356 = stablehlo.multiply %v7355, %s2b0W1 : tensor<128x256x1x1xf32>
    %v7357 = stablehlo.add %v7356, %armeans2b0W1 : tensor<128x256x1x1xf32>
    %v7358 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v7359 = stablehlo.multiply %v7358, %s2b0W1v : tensor<128x256x1x1xf32>
    %v7360 = stablehlo.add %v7359, %v7357 : tensor<128x256x1x1xf32>
    %v7361 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v7362 = stablehlo.multiply %v7361, %v7360 : tensor<128x256x1x1xf32>
    %v7363 = stablehlo.subtract %s2b0W1, %v7362 : tensor<128x256x1x1xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v5937) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v7364 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7365 = stablehlo.multiply %v7364, %s2b0g1 : tensor<128xf32>
    %v7366 = stablehlo.add %v7365, %armeans2b0g1 : tensor<128xf32>
    %v7367 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7368 = stablehlo.multiply %v7367, %s2b0g1v : tensor<128xf32>
    %v7369 = stablehlo.add %v7368, %v7366 : tensor<128xf32>
    %v7370 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7371 = stablehlo.multiply %v7370, %v7369 : tensor<128xf32>
    %v7372 = stablehlo.subtract %s2b0g1, %v7371 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v5940) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v7373 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7374 = stablehlo.multiply %v7373, %s2b0bt1 : tensor<128xf32>
    %v7375 = stablehlo.add %v7374, %armeans2b0bt1 : tensor<128xf32>
    %v7376 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7377 = stablehlo.multiply %v7376, %s2b0bt1v : tensor<128xf32>
    %v7378 = stablehlo.add %v7377, %v7375 : tensor<128xf32>
    %v7379 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7380 = stablehlo.multiply %v7379, %v7378 : tensor<128xf32>
    %v7381 = stablehlo.subtract %s2b0bt1, %v7380 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v5951) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v7382 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7383 = stablehlo.multiply %v7382, %s2b0W2 : tensor<128x128x3x3xf32>
    %v7384 = stablehlo.add %v7383, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v7385 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7386 = stablehlo.multiply %v7385, %s2b0W2v : tensor<128x128x3x3xf32>
    %v7387 = stablehlo.add %v7386, %v7384 : tensor<128x128x3x3xf32>
    %v7388 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7389 = stablehlo.multiply %v7388, %v7387 : tensor<128x128x3x3xf32>
    %v7390 = stablehlo.subtract %s2b0W2, %v7389 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v5965) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v7391 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7392 = stablehlo.multiply %v7391, %s2b0g2 : tensor<128xf32>
    %v7393 = stablehlo.add %v7392, %armeans2b0g2 : tensor<128xf32>
    %v7394 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7395 = stablehlo.multiply %v7394, %s2b0g2v : tensor<128xf32>
    %v7396 = stablehlo.add %v7395, %v7393 : tensor<128xf32>
    %v7397 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7398 = stablehlo.multiply %v7397, %v7396 : tensor<128xf32>
    %v7399 = stablehlo.subtract %s2b0g2, %v7398 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v5968) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v7400 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7401 = stablehlo.multiply %v7400, %s2b0bt2 : tensor<128xf32>
    %v7402 = stablehlo.add %v7401, %armeans2b0bt2 : tensor<128xf32>
    %v7403 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7404 = stablehlo.multiply %v7403, %s2b0bt2v : tensor<128xf32>
    %v7405 = stablehlo.add %v7404, %v7402 : tensor<128xf32>
    %v7406 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7407 = stablehlo.multiply %v7406, %v7405 : tensor<128xf32>
    %v7408 = stablehlo.subtract %s2b0bt2, %v7407 : tensor<128xf32>
    %arsums2b0W3 = "stablehlo.all_reduce"(%v5977) ({
    ^bb0(%aras2b0W3: tensor<f32>, %arbs2b0W3: tensor<f32>):
      %aradds2b0W3 = stablehlo.add %aras2b0W3, %arbs2b0W3 : tensor<f32>
      stablehlo.return %aradds2b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b0W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b0W3 = stablehlo.divide %arsums2b0W3, %arns2b0W3 : tensor<512x128x1x1xf32>
    %v7409 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7410 = stablehlo.multiply %v7409, %s2b0W3 : tensor<512x128x1x1xf32>
    %v7411 = stablehlo.add %v7410, %armeans2b0W3 : tensor<512x128x1x1xf32>
    %v7412 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7413 = stablehlo.multiply %v7412, %s2b0W3v : tensor<512x128x1x1xf32>
    %v7414 = stablehlo.add %v7413, %v7411 : tensor<512x128x1x1xf32>
    %v7415 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7416 = stablehlo.multiply %v7415, %v7414 : tensor<512x128x1x1xf32>
    %v7417 = stablehlo.subtract %s2b0W3, %v7416 : tensor<512x128x1x1xf32>
    %arsums2b0g3 = "stablehlo.all_reduce"(%v5991) ({
    ^bb0(%aras2b0g3: tensor<f32>, %arbs2b0g3: tensor<f32>):
      %aradds2b0g3 = stablehlo.add %aras2b0g3, %arbs2b0g3 : tensor<f32>
      stablehlo.return %aradds2b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3 = stablehlo.divide %arsums2b0g3, %arns2b0g3 : tensor<512xf32>
    %v7418 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7419 = stablehlo.multiply %v7418, %s2b0g3 : tensor<512xf32>
    %v7420 = stablehlo.add %v7419, %armeans2b0g3 : tensor<512xf32>
    %v7421 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7422 = stablehlo.multiply %v7421, %s2b0g3v : tensor<512xf32>
    %v7423 = stablehlo.add %v7422, %v7420 : tensor<512xf32>
    %v7424 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7425 = stablehlo.multiply %v7424, %v7423 : tensor<512xf32>
    %v7426 = stablehlo.subtract %s2b0g3, %v7425 : tensor<512xf32>
    %arsums2b0bt3 = "stablehlo.all_reduce"(%v5994) ({
    ^bb0(%aras2b0bt3: tensor<f32>, %arbs2b0bt3: tensor<f32>):
      %aradds2b0bt3 = stablehlo.add %aras2b0bt3, %arbs2b0bt3 : tensor<f32>
      stablehlo.return %aradds2b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0bt3 = stablehlo.divide %arsums2b0bt3, %arns2b0bt3 : tensor<512xf32>
    %v7427 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7428 = stablehlo.multiply %v7427, %s2b0bt3 : tensor<512xf32>
    %v7429 = stablehlo.add %v7428, %armeans2b0bt3 : tensor<512xf32>
    %v7430 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7431 = stablehlo.multiply %v7430, %s2b0bt3v : tensor<512xf32>
    %v7432 = stablehlo.add %v7431, %v7429 : tensor<512xf32>
    %v7433 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7434 = stablehlo.multiply %v7433, %v7432 : tensor<512xf32>
    %v7435 = stablehlo.subtract %s2b0bt3, %v7434 : tensor<512xf32>
    %arsums2b0Wp = "stablehlo.all_reduce"(%v6005) ({
    ^bb0(%aras2b0Wp: tensor<f32>, %arbs2b0Wp: tensor<f32>):
      %aradds2b0Wp = stablehlo.add %aras2b0Wp, %arbs2b0Wp : tensor<f32>
      stablehlo.return %aradds2b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arns2b0Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeans2b0Wp = stablehlo.divide %arsums2b0Wp, %arns2b0Wp : tensor<512x256x1x1xf32>
    %v7436 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7437 = stablehlo.multiply %v7436, %s2b0Wp : tensor<512x256x1x1xf32>
    %v7438 = stablehlo.add %v7437, %armeans2b0Wp : tensor<512x256x1x1xf32>
    %v7439 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7440 = stablehlo.multiply %v7439, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v7441 = stablehlo.add %v7440, %v7438 : tensor<512x256x1x1xf32>
    %v7442 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v7443 = stablehlo.multiply %v7442, %v7441 : tensor<512x256x1x1xf32>
    %v7444 = stablehlo.subtract %s2b0Wp, %v7443 : tensor<512x256x1x1xf32>
    %arsums2b0gp = "stablehlo.all_reduce"(%v6019) ({
    ^bb0(%aras2b0gp: tensor<f32>, %arbs2b0gp: tensor<f32>):
      %aradds2b0gp = stablehlo.add %aras2b0gp, %arbs2b0gp : tensor<f32>
      stablehlo.return %aradds2b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gp = stablehlo.divide %arsums2b0gp, %arns2b0gp : tensor<512xf32>
    %v7445 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7446 = stablehlo.multiply %v7445, %s2b0gp : tensor<512xf32>
    %v7447 = stablehlo.add %v7446, %armeans2b0gp : tensor<512xf32>
    %v7448 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7449 = stablehlo.multiply %v7448, %s2b0gpv : tensor<512xf32>
    %v7450 = stablehlo.add %v7449, %v7447 : tensor<512xf32>
    %v7451 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7452 = stablehlo.multiply %v7451, %v7450 : tensor<512xf32>
    %v7453 = stablehlo.subtract %s2b0gp, %v7452 : tensor<512xf32>
    %arsums2b0btp = "stablehlo.all_reduce"(%v6022) ({
    ^bb0(%aras2b0btp: tensor<f32>, %arbs2b0btp: tensor<f32>):
      %aradds2b0btp = stablehlo.add %aras2b0btp, %arbs2b0btp : tensor<f32>
      stablehlo.return %aradds2b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0btp = stablehlo.divide %arsums2b0btp, %arns2b0btp : tensor<512xf32>
    %v7454 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7455 = stablehlo.multiply %v7454, %s2b0btp : tensor<512xf32>
    %v7456 = stablehlo.add %v7455, %armeans2b0btp : tensor<512xf32>
    %v7457 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7458 = stablehlo.multiply %v7457, %s2b0btpv : tensor<512xf32>
    %v7459 = stablehlo.add %v7458, %v7456 : tensor<512xf32>
    %v7460 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7461 = stablehlo.multiply %v7460, %v7459 : tensor<512xf32>
    %v7462 = stablehlo.subtract %s2b0btp, %v7461 : tensor<512xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v5611) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x512x1x1xf32>
    %v7463 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7464 = stablehlo.multiply %v7463, %s2b1W1 : tensor<128x512x1x1xf32>
    %v7465 = stablehlo.add %v7464, %armeans2b1W1 : tensor<128x512x1x1xf32>
    %v7466 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7467 = stablehlo.multiply %v7466, %s2b1W1v : tensor<128x512x1x1xf32>
    %v7468 = stablehlo.add %v7467, %v7465 : tensor<128x512x1x1xf32>
    %v7469 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7470 = stablehlo.multiply %v7469, %v7468 : tensor<128x512x1x1xf32>
    %v7471 = stablehlo.subtract %s2b1W1, %v7470 : tensor<128x512x1x1xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v5625) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v7472 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7473 = stablehlo.multiply %v7472, %s2b1g1 : tensor<128xf32>
    %v7474 = stablehlo.add %v7473, %armeans2b1g1 : tensor<128xf32>
    %v7475 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7476 = stablehlo.multiply %v7475, %s2b1g1v : tensor<128xf32>
    %v7477 = stablehlo.add %v7476, %v7474 : tensor<128xf32>
    %v7478 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7479 = stablehlo.multiply %v7478, %v7477 : tensor<128xf32>
    %v7480 = stablehlo.subtract %s2b1g1, %v7479 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v5628) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v7481 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7482 = stablehlo.multiply %v7481, %s2b1bt1 : tensor<128xf32>
    %v7483 = stablehlo.add %v7482, %armeans2b1bt1 : tensor<128xf32>
    %v7484 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7485 = stablehlo.multiply %v7484, %s2b1bt1v : tensor<128xf32>
    %v7486 = stablehlo.add %v7485, %v7483 : tensor<128xf32>
    %v7487 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7488 = stablehlo.multiply %v7487, %v7486 : tensor<128xf32>
    %v7489 = stablehlo.subtract %s2b1bt1, %v7488 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v5637) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v7490 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7491 = stablehlo.multiply %v7490, %s2b1W2 : tensor<128x128x3x3xf32>
    %v7492 = stablehlo.add %v7491, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v7493 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7494 = stablehlo.multiply %v7493, %s2b1W2v : tensor<128x128x3x3xf32>
    %v7495 = stablehlo.add %v7494, %v7492 : tensor<128x128x3x3xf32>
    %v7496 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7497 = stablehlo.multiply %v7496, %v7495 : tensor<128x128x3x3xf32>
    %v7498 = stablehlo.subtract %s2b1W2, %v7497 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v5651) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v7499 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7500 = stablehlo.multiply %v7499, %s2b1g2 : tensor<128xf32>
    %v7501 = stablehlo.add %v7500, %armeans2b1g2 : tensor<128xf32>
    %v7502 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7503 = stablehlo.multiply %v7502, %s2b1g2v : tensor<128xf32>
    %v7504 = stablehlo.add %v7503, %v7501 : tensor<128xf32>
    %v7505 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7506 = stablehlo.multiply %v7505, %v7504 : tensor<128xf32>
    %v7507 = stablehlo.subtract %s2b1g2, %v7506 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v5654) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v7508 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7509 = stablehlo.multiply %v7508, %s2b1bt2 : tensor<128xf32>
    %v7510 = stablehlo.add %v7509, %armeans2b1bt2 : tensor<128xf32>
    %v7511 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7512 = stablehlo.multiply %v7511, %s2b1bt2v : tensor<128xf32>
    %v7513 = stablehlo.add %v7512, %v7510 : tensor<128xf32>
    %v7514 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7515 = stablehlo.multiply %v7514, %v7513 : tensor<128xf32>
    %v7516 = stablehlo.subtract %s2b1bt2, %v7515 : tensor<128xf32>
    %arsums2b1W3 = "stablehlo.all_reduce"(%v5663) ({
    ^bb0(%aras2b1W3: tensor<f32>, %arbs2b1W3: tensor<f32>):
      %aradds2b1W3 = stablehlo.add %aras2b1W3, %arbs2b1W3 : tensor<f32>
      stablehlo.return %aradds2b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b1W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b1W3 = stablehlo.divide %arsums2b1W3, %arns2b1W3 : tensor<512x128x1x1xf32>
    %v7517 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7518 = stablehlo.multiply %v7517, %s2b1W3 : tensor<512x128x1x1xf32>
    %v7519 = stablehlo.add %v7518, %armeans2b1W3 : tensor<512x128x1x1xf32>
    %v7520 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7521 = stablehlo.multiply %v7520, %s2b1W3v : tensor<512x128x1x1xf32>
    %v7522 = stablehlo.add %v7521, %v7519 : tensor<512x128x1x1xf32>
    %v7523 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7524 = stablehlo.multiply %v7523, %v7522 : tensor<512x128x1x1xf32>
    %v7525 = stablehlo.subtract %s2b1W3, %v7524 : tensor<512x128x1x1xf32>
    %arsums2b1g3 = "stablehlo.all_reduce"(%v5677) ({
    ^bb0(%aras2b1g3: tensor<f32>, %arbs2b1g3: tensor<f32>):
      %aradds2b1g3 = stablehlo.add %aras2b1g3, %arbs2b1g3 : tensor<f32>
      stablehlo.return %aradds2b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3 = stablehlo.divide %arsums2b1g3, %arns2b1g3 : tensor<512xf32>
    %v7526 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7527 = stablehlo.multiply %v7526, %s2b1g3 : tensor<512xf32>
    %v7528 = stablehlo.add %v7527, %armeans2b1g3 : tensor<512xf32>
    %v7529 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7530 = stablehlo.multiply %v7529, %s2b1g3v : tensor<512xf32>
    %v7531 = stablehlo.add %v7530, %v7528 : tensor<512xf32>
    %v7532 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7533 = stablehlo.multiply %v7532, %v7531 : tensor<512xf32>
    %v7534 = stablehlo.subtract %s2b1g3, %v7533 : tensor<512xf32>
    %arsums2b1bt3 = "stablehlo.all_reduce"(%v5680) ({
    ^bb0(%aras2b1bt3: tensor<f32>, %arbs2b1bt3: tensor<f32>):
      %aradds2b1bt3 = stablehlo.add %aras2b1bt3, %arbs2b1bt3 : tensor<f32>
      stablehlo.return %aradds2b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1bt3 = stablehlo.divide %arsums2b1bt3, %arns2b1bt3 : tensor<512xf32>
    %v7535 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7536 = stablehlo.multiply %v7535, %s2b1bt3 : tensor<512xf32>
    %v7537 = stablehlo.add %v7536, %armeans2b1bt3 : tensor<512xf32>
    %v7538 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7539 = stablehlo.multiply %v7538, %s2b1bt3v : tensor<512xf32>
    %v7540 = stablehlo.add %v7539, %v7537 : tensor<512xf32>
    %v7541 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7542 = stablehlo.multiply %v7541, %v7540 : tensor<512xf32>
    %v7543 = stablehlo.subtract %s2b1bt3, %v7542 : tensor<512xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v5355) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x512x1x1xf32>
    %v7544 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7545 = stablehlo.multiply %v7544, %s2b2W1 : tensor<128x512x1x1xf32>
    %v7546 = stablehlo.add %v7545, %armeans2b2W1 : tensor<128x512x1x1xf32>
    %v7547 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7548 = stablehlo.multiply %v7547, %s2b2W1v : tensor<128x512x1x1xf32>
    %v7549 = stablehlo.add %v7548, %v7546 : tensor<128x512x1x1xf32>
    %v7550 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7551 = stablehlo.multiply %v7550, %v7549 : tensor<128x512x1x1xf32>
    %v7552 = stablehlo.subtract %s2b2W1, %v7551 : tensor<128x512x1x1xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v5369) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v7553 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7554 = stablehlo.multiply %v7553, %s2b2g1 : tensor<128xf32>
    %v7555 = stablehlo.add %v7554, %armeans2b2g1 : tensor<128xf32>
    %v7556 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7557 = stablehlo.multiply %v7556, %s2b2g1v : tensor<128xf32>
    %v7558 = stablehlo.add %v7557, %v7555 : tensor<128xf32>
    %v7559 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7560 = stablehlo.multiply %v7559, %v7558 : tensor<128xf32>
    %v7561 = stablehlo.subtract %s2b2g1, %v7560 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v5372) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v7562 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7563 = stablehlo.multiply %v7562, %s2b2bt1 : tensor<128xf32>
    %v7564 = stablehlo.add %v7563, %armeans2b2bt1 : tensor<128xf32>
    %v7565 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7566 = stablehlo.multiply %v7565, %s2b2bt1v : tensor<128xf32>
    %v7567 = stablehlo.add %v7566, %v7564 : tensor<128xf32>
    %v7568 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7569 = stablehlo.multiply %v7568, %v7567 : tensor<128xf32>
    %v7570 = stablehlo.subtract %s2b2bt1, %v7569 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v5381) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v7571 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7572 = stablehlo.multiply %v7571, %s2b2W2 : tensor<128x128x3x3xf32>
    %v7573 = stablehlo.add %v7572, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v7574 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7575 = stablehlo.multiply %v7574, %s2b2W2v : tensor<128x128x3x3xf32>
    %v7576 = stablehlo.add %v7575, %v7573 : tensor<128x128x3x3xf32>
    %v7577 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7578 = stablehlo.multiply %v7577, %v7576 : tensor<128x128x3x3xf32>
    %v7579 = stablehlo.subtract %s2b2W2, %v7578 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v5395) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v7580 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7581 = stablehlo.multiply %v7580, %s2b2g2 : tensor<128xf32>
    %v7582 = stablehlo.add %v7581, %armeans2b2g2 : tensor<128xf32>
    %v7583 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7584 = stablehlo.multiply %v7583, %s2b2g2v : tensor<128xf32>
    %v7585 = stablehlo.add %v7584, %v7582 : tensor<128xf32>
    %v7586 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7587 = stablehlo.multiply %v7586, %v7585 : tensor<128xf32>
    %v7588 = stablehlo.subtract %s2b2g2, %v7587 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v5398) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v7589 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7590 = stablehlo.multiply %v7589, %s2b2bt2 : tensor<128xf32>
    %v7591 = stablehlo.add %v7590, %armeans2b2bt2 : tensor<128xf32>
    %v7592 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7593 = stablehlo.multiply %v7592, %s2b2bt2v : tensor<128xf32>
    %v7594 = stablehlo.add %v7593, %v7591 : tensor<128xf32>
    %v7595 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7596 = stablehlo.multiply %v7595, %v7594 : tensor<128xf32>
    %v7597 = stablehlo.subtract %s2b2bt2, %v7596 : tensor<128xf32>
    %arsums2b2W3 = "stablehlo.all_reduce"(%v5407) ({
    ^bb0(%aras2b2W3: tensor<f32>, %arbs2b2W3: tensor<f32>):
      %aradds2b2W3 = stablehlo.add %aras2b2W3, %arbs2b2W3 : tensor<f32>
      stablehlo.return %aradds2b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b2W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b2W3 = stablehlo.divide %arsums2b2W3, %arns2b2W3 : tensor<512x128x1x1xf32>
    %v7598 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7599 = stablehlo.multiply %v7598, %s2b2W3 : tensor<512x128x1x1xf32>
    %v7600 = stablehlo.add %v7599, %armeans2b2W3 : tensor<512x128x1x1xf32>
    %v7601 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7602 = stablehlo.multiply %v7601, %s2b2W3v : tensor<512x128x1x1xf32>
    %v7603 = stablehlo.add %v7602, %v7600 : tensor<512x128x1x1xf32>
    %v7604 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7605 = stablehlo.multiply %v7604, %v7603 : tensor<512x128x1x1xf32>
    %v7606 = stablehlo.subtract %s2b2W3, %v7605 : tensor<512x128x1x1xf32>
    %arsums2b2g3 = "stablehlo.all_reduce"(%v5421) ({
    ^bb0(%aras2b2g3: tensor<f32>, %arbs2b2g3: tensor<f32>):
      %aradds2b2g3 = stablehlo.add %aras2b2g3, %arbs2b2g3 : tensor<f32>
      stablehlo.return %aradds2b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3 = stablehlo.divide %arsums2b2g3, %arns2b2g3 : tensor<512xf32>
    %v7607 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7608 = stablehlo.multiply %v7607, %s2b2g3 : tensor<512xf32>
    %v7609 = stablehlo.add %v7608, %armeans2b2g3 : tensor<512xf32>
    %v7610 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7611 = stablehlo.multiply %v7610, %s2b2g3v : tensor<512xf32>
    %v7612 = stablehlo.add %v7611, %v7609 : tensor<512xf32>
    %v7613 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7614 = stablehlo.multiply %v7613, %v7612 : tensor<512xf32>
    %v7615 = stablehlo.subtract %s2b2g3, %v7614 : tensor<512xf32>
    %arsums2b2bt3 = "stablehlo.all_reduce"(%v5424) ({
    ^bb0(%aras2b2bt3: tensor<f32>, %arbs2b2bt3: tensor<f32>):
      %aradds2b2bt3 = stablehlo.add %aras2b2bt3, %arbs2b2bt3 : tensor<f32>
      stablehlo.return %aradds2b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2bt3 = stablehlo.divide %arsums2b2bt3, %arns2b2bt3 : tensor<512xf32>
    %v7616 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7617 = stablehlo.multiply %v7616, %s2b2bt3 : tensor<512xf32>
    %v7618 = stablehlo.add %v7617, %armeans2b2bt3 : tensor<512xf32>
    %v7619 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7620 = stablehlo.multiply %v7619, %s2b2bt3v : tensor<512xf32>
    %v7621 = stablehlo.add %v7620, %v7618 : tensor<512xf32>
    %v7622 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7623 = stablehlo.multiply %v7622, %v7621 : tensor<512xf32>
    %v7624 = stablehlo.subtract %s2b2bt3, %v7623 : tensor<512xf32>
    %arsums2b3W1 = "stablehlo.all_reduce"(%v5099) ({
    ^bb0(%aras2b3W1: tensor<f32>, %arbs2b3W1: tensor<f32>):
      %aradds2b3W1 = stablehlo.add %aras2b3W1, %arbs2b3W1 : tensor<f32>
      stablehlo.return %aradds2b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b3W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b3W1 = stablehlo.divide %arsums2b3W1, %arns2b3W1 : tensor<128x512x1x1xf32>
    %v7625 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7626 = stablehlo.multiply %v7625, %s2b3W1 : tensor<128x512x1x1xf32>
    %v7627 = stablehlo.add %v7626, %armeans2b3W1 : tensor<128x512x1x1xf32>
    %v7628 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7629 = stablehlo.multiply %v7628, %s2b3W1v : tensor<128x512x1x1xf32>
    %v7630 = stablehlo.add %v7629, %v7627 : tensor<128x512x1x1xf32>
    %v7631 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v7632 = stablehlo.multiply %v7631, %v7630 : tensor<128x512x1x1xf32>
    %v7633 = stablehlo.subtract %s2b3W1, %v7632 : tensor<128x512x1x1xf32>
    %arsums2b3g1 = "stablehlo.all_reduce"(%v5113) ({
    ^bb0(%aras2b3g1: tensor<f32>, %arbs2b3g1: tensor<f32>):
      %aradds2b3g1 = stablehlo.add %aras2b3g1, %arbs2b3g1 : tensor<f32>
      stablehlo.return %aradds2b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1 = stablehlo.divide %arsums2b3g1, %arns2b3g1 : tensor<128xf32>
    %v7634 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7635 = stablehlo.multiply %v7634, %s2b3g1 : tensor<128xf32>
    %v7636 = stablehlo.add %v7635, %armeans2b3g1 : tensor<128xf32>
    %v7637 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7638 = stablehlo.multiply %v7637, %s2b3g1v : tensor<128xf32>
    %v7639 = stablehlo.add %v7638, %v7636 : tensor<128xf32>
    %v7640 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7641 = stablehlo.multiply %v7640, %v7639 : tensor<128xf32>
    %v7642 = stablehlo.subtract %s2b3g1, %v7641 : tensor<128xf32>
    %arsums2b3bt1 = "stablehlo.all_reduce"(%v5116) ({
    ^bb0(%aras2b3bt1: tensor<f32>, %arbs2b3bt1: tensor<f32>):
      %aradds2b3bt1 = stablehlo.add %aras2b3bt1, %arbs2b3bt1 : tensor<f32>
      stablehlo.return %aradds2b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt1 = stablehlo.divide %arsums2b3bt1, %arns2b3bt1 : tensor<128xf32>
    %v7643 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7644 = stablehlo.multiply %v7643, %s2b3bt1 : tensor<128xf32>
    %v7645 = stablehlo.add %v7644, %armeans2b3bt1 : tensor<128xf32>
    %v7646 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7647 = stablehlo.multiply %v7646, %s2b3bt1v : tensor<128xf32>
    %v7648 = stablehlo.add %v7647, %v7645 : tensor<128xf32>
    %v7649 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7650 = stablehlo.multiply %v7649, %v7648 : tensor<128xf32>
    %v7651 = stablehlo.subtract %s2b3bt1, %v7650 : tensor<128xf32>
    %arsums2b3W2 = "stablehlo.all_reduce"(%v5125) ({
    ^bb0(%aras2b3W2: tensor<f32>, %arbs2b3W2: tensor<f32>):
      %aradds2b3W2 = stablehlo.add %aras2b3W2, %arbs2b3W2 : tensor<f32>
      stablehlo.return %aradds2b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b3W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b3W2 = stablehlo.divide %arsums2b3W2, %arns2b3W2 : tensor<128x128x3x3xf32>
    %v7652 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7653 = stablehlo.multiply %v7652, %s2b3W2 : tensor<128x128x3x3xf32>
    %v7654 = stablehlo.add %v7653, %armeans2b3W2 : tensor<128x128x3x3xf32>
    %v7655 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7656 = stablehlo.multiply %v7655, %s2b3W2v : tensor<128x128x3x3xf32>
    %v7657 = stablehlo.add %v7656, %v7654 : tensor<128x128x3x3xf32>
    %v7658 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v7659 = stablehlo.multiply %v7658, %v7657 : tensor<128x128x3x3xf32>
    %v7660 = stablehlo.subtract %s2b3W2, %v7659 : tensor<128x128x3x3xf32>
    %arsums2b3g2 = "stablehlo.all_reduce"(%v5139) ({
    ^bb0(%aras2b3g2: tensor<f32>, %arbs2b3g2: tensor<f32>):
      %aradds2b3g2 = stablehlo.add %aras2b3g2, %arbs2b3g2 : tensor<f32>
      stablehlo.return %aradds2b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2 = stablehlo.divide %arsums2b3g2, %arns2b3g2 : tensor<128xf32>
    %v7661 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7662 = stablehlo.multiply %v7661, %s2b3g2 : tensor<128xf32>
    %v7663 = stablehlo.add %v7662, %armeans2b3g2 : tensor<128xf32>
    %v7664 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7665 = stablehlo.multiply %v7664, %s2b3g2v : tensor<128xf32>
    %v7666 = stablehlo.add %v7665, %v7663 : tensor<128xf32>
    %v7667 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7668 = stablehlo.multiply %v7667, %v7666 : tensor<128xf32>
    %v7669 = stablehlo.subtract %s2b3g2, %v7668 : tensor<128xf32>
    %arsums2b3bt2 = "stablehlo.all_reduce"(%v5142) ({
    ^bb0(%aras2b3bt2: tensor<f32>, %arbs2b3bt2: tensor<f32>):
      %aradds2b3bt2 = stablehlo.add %aras2b3bt2, %arbs2b3bt2 : tensor<f32>
      stablehlo.return %aradds2b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt2 = stablehlo.divide %arsums2b3bt2, %arns2b3bt2 : tensor<128xf32>
    %v7670 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7671 = stablehlo.multiply %v7670, %s2b3bt2 : tensor<128xf32>
    %v7672 = stablehlo.add %v7671, %armeans2b3bt2 : tensor<128xf32>
    %v7673 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7674 = stablehlo.multiply %v7673, %s2b3bt2v : tensor<128xf32>
    %v7675 = stablehlo.add %v7674, %v7672 : tensor<128xf32>
    %v7676 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v7677 = stablehlo.multiply %v7676, %v7675 : tensor<128xf32>
    %v7678 = stablehlo.subtract %s2b3bt2, %v7677 : tensor<128xf32>
    %arsums2b3W3 = "stablehlo.all_reduce"(%v5151) ({
    ^bb0(%aras2b3W3: tensor<f32>, %arbs2b3W3: tensor<f32>):
      %aradds2b3W3 = stablehlo.add %aras2b3W3, %arbs2b3W3 : tensor<f32>
      stablehlo.return %aradds2b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b3W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b3W3 = stablehlo.divide %arsums2b3W3, %arns2b3W3 : tensor<512x128x1x1xf32>
    %v7679 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7680 = stablehlo.multiply %v7679, %s2b3W3 : tensor<512x128x1x1xf32>
    %v7681 = stablehlo.add %v7680, %armeans2b3W3 : tensor<512x128x1x1xf32>
    %v7682 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7683 = stablehlo.multiply %v7682, %s2b3W3v : tensor<512x128x1x1xf32>
    %v7684 = stablehlo.add %v7683, %v7681 : tensor<512x128x1x1xf32>
    %v7685 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v7686 = stablehlo.multiply %v7685, %v7684 : tensor<512x128x1x1xf32>
    %v7687 = stablehlo.subtract %s2b3W3, %v7686 : tensor<512x128x1x1xf32>
    %arsums2b3g3 = "stablehlo.all_reduce"(%v5165) ({
    ^bb0(%aras2b3g3: tensor<f32>, %arbs2b3g3: tensor<f32>):
      %aradds2b3g3 = stablehlo.add %aras2b3g3, %arbs2b3g3 : tensor<f32>
      stablehlo.return %aradds2b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3 = stablehlo.divide %arsums2b3g3, %arns2b3g3 : tensor<512xf32>
    %v7688 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7689 = stablehlo.multiply %v7688, %s2b3g3 : tensor<512xf32>
    %v7690 = stablehlo.add %v7689, %armeans2b3g3 : tensor<512xf32>
    %v7691 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7692 = stablehlo.multiply %v7691, %s2b3g3v : tensor<512xf32>
    %v7693 = stablehlo.add %v7692, %v7690 : tensor<512xf32>
    %v7694 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7695 = stablehlo.multiply %v7694, %v7693 : tensor<512xf32>
    %v7696 = stablehlo.subtract %s2b3g3, %v7695 : tensor<512xf32>
    %arsums2b3bt3 = "stablehlo.all_reduce"(%v5168) ({
    ^bb0(%aras2b3bt3: tensor<f32>, %arbs2b3bt3: tensor<f32>):
      %aradds2b3bt3 = stablehlo.add %aras2b3bt3, %arbs2b3bt3 : tensor<f32>
      stablehlo.return %aradds2b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3bt3 = stablehlo.divide %arsums2b3bt3, %arns2b3bt3 : tensor<512xf32>
    %v7697 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7698 = stablehlo.multiply %v7697, %s2b3bt3 : tensor<512xf32>
    %v7699 = stablehlo.add %v7698, %armeans2b3bt3 : tensor<512xf32>
    %v7700 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7701 = stablehlo.multiply %v7700, %s2b3bt3v : tensor<512xf32>
    %v7702 = stablehlo.add %v7701, %v7699 : tensor<512xf32>
    %v7703 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7704 = stablehlo.multiply %v7703, %v7702 : tensor<512xf32>
    %v7705 = stablehlo.subtract %s2b3bt3, %v7704 : tensor<512xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v4813) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x512x1x1xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x512x1x1xf32>
    %v7706 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7707 = stablehlo.multiply %v7706, %s3b0W1 : tensor<256x512x1x1xf32>
    %v7708 = stablehlo.add %v7707, %armeans3b0W1 : tensor<256x512x1x1xf32>
    %v7709 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7710 = stablehlo.multiply %v7709, %s3b0W1v : tensor<256x512x1x1xf32>
    %v7711 = stablehlo.add %v7710, %v7708 : tensor<256x512x1x1xf32>
    %v7712 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v7713 = stablehlo.multiply %v7712, %v7711 : tensor<256x512x1x1xf32>
    %v7714 = stablehlo.subtract %s3b0W1, %v7713 : tensor<256x512x1x1xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v4827) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v7715 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7716 = stablehlo.multiply %v7715, %s3b0g1 : tensor<256xf32>
    %v7717 = stablehlo.add %v7716, %armeans3b0g1 : tensor<256xf32>
    %v7718 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7719 = stablehlo.multiply %v7718, %s3b0g1v : tensor<256xf32>
    %v7720 = stablehlo.add %v7719, %v7717 : tensor<256xf32>
    %v7721 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7722 = stablehlo.multiply %v7721, %v7720 : tensor<256xf32>
    %v7723 = stablehlo.subtract %s3b0g1, %v7722 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v4830) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v7724 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7725 = stablehlo.multiply %v7724, %s3b0bt1 : tensor<256xf32>
    %v7726 = stablehlo.add %v7725, %armeans3b0bt1 : tensor<256xf32>
    %v7727 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7728 = stablehlo.multiply %v7727, %s3b0bt1v : tensor<256xf32>
    %v7729 = stablehlo.add %v7728, %v7726 : tensor<256xf32>
    %v7730 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7731 = stablehlo.multiply %v7730, %v7729 : tensor<256xf32>
    %v7732 = stablehlo.subtract %s3b0bt1, %v7731 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v4841) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v7733 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7734 = stablehlo.multiply %v7733, %s3b0W2 : tensor<256x256x3x3xf32>
    %v7735 = stablehlo.add %v7734, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v7736 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7737 = stablehlo.multiply %v7736, %s3b0W2v : tensor<256x256x3x3xf32>
    %v7738 = stablehlo.add %v7737, %v7735 : tensor<256x256x3x3xf32>
    %v7739 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7740 = stablehlo.multiply %v7739, %v7738 : tensor<256x256x3x3xf32>
    %v7741 = stablehlo.subtract %s3b0W2, %v7740 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v4855) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v7742 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7743 = stablehlo.multiply %v7742, %s3b0g2 : tensor<256xf32>
    %v7744 = stablehlo.add %v7743, %armeans3b0g2 : tensor<256xf32>
    %v7745 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7746 = stablehlo.multiply %v7745, %s3b0g2v : tensor<256xf32>
    %v7747 = stablehlo.add %v7746, %v7744 : tensor<256xf32>
    %v7748 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7749 = stablehlo.multiply %v7748, %v7747 : tensor<256xf32>
    %v7750 = stablehlo.subtract %s3b0g2, %v7749 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v4858) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v7751 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7752 = stablehlo.multiply %v7751, %s3b0bt2 : tensor<256xf32>
    %v7753 = stablehlo.add %v7752, %armeans3b0bt2 : tensor<256xf32>
    %v7754 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7755 = stablehlo.multiply %v7754, %s3b0bt2v : tensor<256xf32>
    %v7756 = stablehlo.add %v7755, %v7753 : tensor<256xf32>
    %v7757 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7758 = stablehlo.multiply %v7757, %v7756 : tensor<256xf32>
    %v7759 = stablehlo.subtract %s3b0bt2, %v7758 : tensor<256xf32>
    %arsums3b0W3 = "stablehlo.all_reduce"(%v4867) ({
    ^bb0(%aras3b0W3: tensor<f32>, %arbs3b0W3: tensor<f32>):
      %aradds3b0W3 = stablehlo.add %aras3b0W3, %arbs3b0W3 : tensor<f32>
      stablehlo.return %aradds3b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b0W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b0W3 = stablehlo.divide %arsums3b0W3, %arns3b0W3 : tensor<1024x256x1x1xf32>
    %v7760 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7761 = stablehlo.multiply %v7760, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v7762 = stablehlo.add %v7761, %armeans3b0W3 : tensor<1024x256x1x1xf32>
    %v7763 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7764 = stablehlo.multiply %v7763, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v7765 = stablehlo.add %v7764, %v7762 : tensor<1024x256x1x1xf32>
    %v7766 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7767 = stablehlo.multiply %v7766, %v7765 : tensor<1024x256x1x1xf32>
    %v7768 = stablehlo.subtract %s3b0W3, %v7767 : tensor<1024x256x1x1xf32>
    %arsums3b0g3 = "stablehlo.all_reduce"(%v4881) ({
    ^bb0(%aras3b0g3: tensor<f32>, %arbs3b0g3: tensor<f32>):
      %aradds3b0g3 = stablehlo.add %aras3b0g3, %arbs3b0g3 : tensor<f32>
      stablehlo.return %aradds3b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3 = stablehlo.divide %arsums3b0g3, %arns3b0g3 : tensor<1024xf32>
    %v7769 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7770 = stablehlo.multiply %v7769, %s3b0g3 : tensor<1024xf32>
    %v7771 = stablehlo.add %v7770, %armeans3b0g3 : tensor<1024xf32>
    %v7772 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7773 = stablehlo.multiply %v7772, %s3b0g3v : tensor<1024xf32>
    %v7774 = stablehlo.add %v7773, %v7771 : tensor<1024xf32>
    %v7775 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7776 = stablehlo.multiply %v7775, %v7774 : tensor<1024xf32>
    %v7777 = stablehlo.subtract %s3b0g3, %v7776 : tensor<1024xf32>
    %arsums3b0bt3 = "stablehlo.all_reduce"(%v4884) ({
    ^bb0(%aras3b0bt3: tensor<f32>, %arbs3b0bt3: tensor<f32>):
      %aradds3b0bt3 = stablehlo.add %aras3b0bt3, %arbs3b0bt3 : tensor<f32>
      stablehlo.return %aradds3b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0bt3 = stablehlo.divide %arsums3b0bt3, %arns3b0bt3 : tensor<1024xf32>
    %v7778 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7779 = stablehlo.multiply %v7778, %s3b0bt3 : tensor<1024xf32>
    %v7780 = stablehlo.add %v7779, %armeans3b0bt3 : tensor<1024xf32>
    %v7781 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7782 = stablehlo.multiply %v7781, %s3b0bt3v : tensor<1024xf32>
    %v7783 = stablehlo.add %v7782, %v7780 : tensor<1024xf32>
    %v7784 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7785 = stablehlo.multiply %v7784, %v7783 : tensor<1024xf32>
    %v7786 = stablehlo.subtract %s3b0bt3, %v7785 : tensor<1024xf32>
    %arsums3b0Wp = "stablehlo.all_reduce"(%v4895) ({
    ^bb0(%aras3b0Wp: tensor<f32>, %arbs3b0Wp: tensor<f32>):
      %aradds3b0Wp = stablehlo.add %aras3b0Wp, %arbs3b0Wp : tensor<f32>
      stablehlo.return %aradds3b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %arns3b0Wp = stablehlo.constant dense<4.0> : tensor<1024x512x1x1xf32>
    %armeans3b0Wp = stablehlo.divide %arsums3b0Wp, %arns3b0Wp : tensor<1024x512x1x1xf32>
    %v7787 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7788 = stablehlo.multiply %v7787, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v7789 = stablehlo.add %v7788, %armeans3b0Wp : tensor<1024x512x1x1xf32>
    %v7790 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7791 = stablehlo.multiply %v7790, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v7792 = stablehlo.add %v7791, %v7789 : tensor<1024x512x1x1xf32>
    %v7793 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7794 = stablehlo.multiply %v7793, %v7792 : tensor<1024x512x1x1xf32>
    %v7795 = stablehlo.subtract %s3b0Wp, %v7794 : tensor<1024x512x1x1xf32>
    %arsums3b0gp = "stablehlo.all_reduce"(%v4909) ({
    ^bb0(%aras3b0gp: tensor<f32>, %arbs3b0gp: tensor<f32>):
      %aradds3b0gp = stablehlo.add %aras3b0gp, %arbs3b0gp : tensor<f32>
      stablehlo.return %aradds3b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gp = stablehlo.divide %arsums3b0gp, %arns3b0gp : tensor<1024xf32>
    %v7796 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7797 = stablehlo.multiply %v7796, %s3b0gp : tensor<1024xf32>
    %v7798 = stablehlo.add %v7797, %armeans3b0gp : tensor<1024xf32>
    %v7799 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7800 = stablehlo.multiply %v7799, %s3b0gpv : tensor<1024xf32>
    %v7801 = stablehlo.add %v7800, %v7798 : tensor<1024xf32>
    %v7802 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7803 = stablehlo.multiply %v7802, %v7801 : tensor<1024xf32>
    %v7804 = stablehlo.subtract %s3b0gp, %v7803 : tensor<1024xf32>
    %arsums3b0btp = "stablehlo.all_reduce"(%v4912) ({
    ^bb0(%aras3b0btp: tensor<f32>, %arbs3b0btp: tensor<f32>):
      %aradds3b0btp = stablehlo.add %aras3b0btp, %arbs3b0btp : tensor<f32>
      stablehlo.return %aradds3b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0btp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0btp = stablehlo.divide %arsums3b0btp, %arns3b0btp : tensor<1024xf32>
    %v7805 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7806 = stablehlo.multiply %v7805, %s3b0btp : tensor<1024xf32>
    %v7807 = stablehlo.add %v7806, %armeans3b0btp : tensor<1024xf32>
    %v7808 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7809 = stablehlo.multiply %v7808, %s3b0btpv : tensor<1024xf32>
    %v7810 = stablehlo.add %v7809, %v7807 : tensor<1024xf32>
    %v7811 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7812 = stablehlo.multiply %v7811, %v7810 : tensor<1024xf32>
    %v7813 = stablehlo.subtract %s3b0btp, %v7812 : tensor<1024xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v4501) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x1024x1x1xf32>
    %v7814 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7815 = stablehlo.multiply %v7814, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v7816 = stablehlo.add %v7815, %armeans3b1W1 : tensor<256x1024x1x1xf32>
    %v7817 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7818 = stablehlo.multiply %v7817, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v7819 = stablehlo.add %v7818, %v7816 : tensor<256x1024x1x1xf32>
    %v7820 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7821 = stablehlo.multiply %v7820, %v7819 : tensor<256x1024x1x1xf32>
    %v7822 = stablehlo.subtract %s3b1W1, %v7821 : tensor<256x1024x1x1xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v4515) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v7823 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7824 = stablehlo.multiply %v7823, %s3b1g1 : tensor<256xf32>
    %v7825 = stablehlo.add %v7824, %armeans3b1g1 : tensor<256xf32>
    %v7826 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7827 = stablehlo.multiply %v7826, %s3b1g1v : tensor<256xf32>
    %v7828 = stablehlo.add %v7827, %v7825 : tensor<256xf32>
    %v7829 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7830 = stablehlo.multiply %v7829, %v7828 : tensor<256xf32>
    %v7831 = stablehlo.subtract %s3b1g1, %v7830 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v4518) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v7832 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7833 = stablehlo.multiply %v7832, %s3b1bt1 : tensor<256xf32>
    %v7834 = stablehlo.add %v7833, %armeans3b1bt1 : tensor<256xf32>
    %v7835 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7836 = stablehlo.multiply %v7835, %s3b1bt1v : tensor<256xf32>
    %v7837 = stablehlo.add %v7836, %v7834 : tensor<256xf32>
    %v7838 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7839 = stablehlo.multiply %v7838, %v7837 : tensor<256xf32>
    %v7840 = stablehlo.subtract %s3b1bt1, %v7839 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v4527) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v7841 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7842 = stablehlo.multiply %v7841, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7843 = stablehlo.add %v7842, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7844 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7845 = stablehlo.multiply %v7844, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7846 = stablehlo.add %v7845, %v7843 : tensor<256x256x3x3xf32>
    %v7847 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7848 = stablehlo.multiply %v7847, %v7846 : tensor<256x256x3x3xf32>
    %v7849 = stablehlo.subtract %s3b1W2, %v7848 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v4541) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7850 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7851 = stablehlo.multiply %v7850, %s3b1g2 : tensor<256xf32>
    %v7852 = stablehlo.add %v7851, %armeans3b1g2 : tensor<256xf32>
    %v7853 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7854 = stablehlo.multiply %v7853, %s3b1g2v : tensor<256xf32>
    %v7855 = stablehlo.add %v7854, %v7852 : tensor<256xf32>
    %v7856 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7857 = stablehlo.multiply %v7856, %v7855 : tensor<256xf32>
    %v7858 = stablehlo.subtract %s3b1g2, %v7857 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v4544) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7859 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7860 = stablehlo.multiply %v7859, %s3b1bt2 : tensor<256xf32>
    %v7861 = stablehlo.add %v7860, %armeans3b1bt2 : tensor<256xf32>
    %v7862 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7863 = stablehlo.multiply %v7862, %s3b1bt2v : tensor<256xf32>
    %v7864 = stablehlo.add %v7863, %v7861 : tensor<256xf32>
    %v7865 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7866 = stablehlo.multiply %v7865, %v7864 : tensor<256xf32>
    %v7867 = stablehlo.subtract %s3b1bt2, %v7866 : tensor<256xf32>
    %arsums3b1W3 = "stablehlo.all_reduce"(%v4553) ({
    ^bb0(%aras3b1W3: tensor<f32>, %arbs3b1W3: tensor<f32>):
      %aradds3b1W3 = stablehlo.add %aras3b1W3, %arbs3b1W3 : tensor<f32>
      stablehlo.return %aradds3b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b1W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b1W3 = stablehlo.divide %arsums3b1W3, %arns3b1W3 : tensor<1024x256x1x1xf32>
    %v7868 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7869 = stablehlo.multiply %v7868, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v7870 = stablehlo.add %v7869, %armeans3b1W3 : tensor<1024x256x1x1xf32>
    %v7871 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7872 = stablehlo.multiply %v7871, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v7873 = stablehlo.add %v7872, %v7870 : tensor<1024x256x1x1xf32>
    %v7874 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7875 = stablehlo.multiply %v7874, %v7873 : tensor<1024x256x1x1xf32>
    %v7876 = stablehlo.subtract %s3b1W3, %v7875 : tensor<1024x256x1x1xf32>
    %arsums3b1g3 = "stablehlo.all_reduce"(%v4567) ({
    ^bb0(%aras3b1g3: tensor<f32>, %arbs3b1g3: tensor<f32>):
      %aradds3b1g3 = stablehlo.add %aras3b1g3, %arbs3b1g3 : tensor<f32>
      stablehlo.return %aradds3b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3 = stablehlo.divide %arsums3b1g3, %arns3b1g3 : tensor<1024xf32>
    %v7877 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7878 = stablehlo.multiply %v7877, %s3b1g3 : tensor<1024xf32>
    %v7879 = stablehlo.add %v7878, %armeans3b1g3 : tensor<1024xf32>
    %v7880 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7881 = stablehlo.multiply %v7880, %s3b1g3v : tensor<1024xf32>
    %v7882 = stablehlo.add %v7881, %v7879 : tensor<1024xf32>
    %v7883 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7884 = stablehlo.multiply %v7883, %v7882 : tensor<1024xf32>
    %v7885 = stablehlo.subtract %s3b1g3, %v7884 : tensor<1024xf32>
    %arsums3b1bt3 = "stablehlo.all_reduce"(%v4570) ({
    ^bb0(%aras3b1bt3: tensor<f32>, %arbs3b1bt3: tensor<f32>):
      %aradds3b1bt3 = stablehlo.add %aras3b1bt3, %arbs3b1bt3 : tensor<f32>
      stablehlo.return %aradds3b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1bt3 = stablehlo.divide %arsums3b1bt3, %arns3b1bt3 : tensor<1024xf32>
    %v7886 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7887 = stablehlo.multiply %v7886, %s3b1bt3 : tensor<1024xf32>
    %v7888 = stablehlo.add %v7887, %armeans3b1bt3 : tensor<1024xf32>
    %v7889 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7890 = stablehlo.multiply %v7889, %s3b1bt3v : tensor<1024xf32>
    %v7891 = stablehlo.add %v7890, %v7888 : tensor<1024xf32>
    %v7892 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7893 = stablehlo.multiply %v7892, %v7891 : tensor<1024xf32>
    %v7894 = stablehlo.subtract %s3b1bt3, %v7893 : tensor<1024xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v4245) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x1024x1x1xf32>
    %v7895 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7896 = stablehlo.multiply %v7895, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v7897 = stablehlo.add %v7896, %armeans3b2W1 : tensor<256x1024x1x1xf32>
    %v7898 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7899 = stablehlo.multiply %v7898, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v7900 = stablehlo.add %v7899, %v7897 : tensor<256x1024x1x1xf32>
    %v7901 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7902 = stablehlo.multiply %v7901, %v7900 : tensor<256x1024x1x1xf32>
    %v7903 = stablehlo.subtract %s3b2W1, %v7902 : tensor<256x1024x1x1xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v4259) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7904 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7905 = stablehlo.multiply %v7904, %s3b2g1 : tensor<256xf32>
    %v7906 = stablehlo.add %v7905, %armeans3b2g1 : tensor<256xf32>
    %v7907 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7908 = stablehlo.multiply %v7907, %s3b2g1v : tensor<256xf32>
    %v7909 = stablehlo.add %v7908, %v7906 : tensor<256xf32>
    %v7910 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7911 = stablehlo.multiply %v7910, %v7909 : tensor<256xf32>
    %v7912 = stablehlo.subtract %s3b2g1, %v7911 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v4262) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7913 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7914 = stablehlo.multiply %v7913, %s3b2bt1 : tensor<256xf32>
    %v7915 = stablehlo.add %v7914, %armeans3b2bt1 : tensor<256xf32>
    %v7916 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7917 = stablehlo.multiply %v7916, %s3b2bt1v : tensor<256xf32>
    %v7918 = stablehlo.add %v7917, %v7915 : tensor<256xf32>
    %v7919 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7920 = stablehlo.multiply %v7919, %v7918 : tensor<256xf32>
    %v7921 = stablehlo.subtract %s3b2bt1, %v7920 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v4271) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7922 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7923 = stablehlo.multiply %v7922, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7924 = stablehlo.add %v7923, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7925 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7926 = stablehlo.multiply %v7925, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7927 = stablehlo.add %v7926, %v7924 : tensor<256x256x3x3xf32>
    %v7928 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7929 = stablehlo.multiply %v7928, %v7927 : tensor<256x256x3x3xf32>
    %v7930 = stablehlo.subtract %s3b2W2, %v7929 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v4285) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7931 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7932 = stablehlo.multiply %v7931, %s3b2g2 : tensor<256xf32>
    %v7933 = stablehlo.add %v7932, %armeans3b2g2 : tensor<256xf32>
    %v7934 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7935 = stablehlo.multiply %v7934, %s3b2g2v : tensor<256xf32>
    %v7936 = stablehlo.add %v7935, %v7933 : tensor<256xf32>
    %v7937 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7938 = stablehlo.multiply %v7937, %v7936 : tensor<256xf32>
    %v7939 = stablehlo.subtract %s3b2g2, %v7938 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v4288) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7940 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7941 = stablehlo.multiply %v7940, %s3b2bt2 : tensor<256xf32>
    %v7942 = stablehlo.add %v7941, %armeans3b2bt2 : tensor<256xf32>
    %v7943 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7944 = stablehlo.multiply %v7943, %s3b2bt2v : tensor<256xf32>
    %v7945 = stablehlo.add %v7944, %v7942 : tensor<256xf32>
    %v7946 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7947 = stablehlo.multiply %v7946, %v7945 : tensor<256xf32>
    %v7948 = stablehlo.subtract %s3b2bt2, %v7947 : tensor<256xf32>
    %arsums3b2W3 = "stablehlo.all_reduce"(%v4297) ({
    ^bb0(%aras3b2W3: tensor<f32>, %arbs3b2W3: tensor<f32>):
      %aradds3b2W3 = stablehlo.add %aras3b2W3, %arbs3b2W3 : tensor<f32>
      stablehlo.return %aradds3b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b2W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b2W3 = stablehlo.divide %arsums3b2W3, %arns3b2W3 : tensor<1024x256x1x1xf32>
    %v7949 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7950 = stablehlo.multiply %v7949, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7951 = stablehlo.add %v7950, %armeans3b2W3 : tensor<1024x256x1x1xf32>
    %v7952 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7953 = stablehlo.multiply %v7952, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7954 = stablehlo.add %v7953, %v7951 : tensor<1024x256x1x1xf32>
    %v7955 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7956 = stablehlo.multiply %v7955, %v7954 : tensor<1024x256x1x1xf32>
    %v7957 = stablehlo.subtract %s3b2W3, %v7956 : tensor<1024x256x1x1xf32>
    %arsums3b2g3 = "stablehlo.all_reduce"(%v4311) ({
    ^bb0(%aras3b2g3: tensor<f32>, %arbs3b2g3: tensor<f32>):
      %aradds3b2g3 = stablehlo.add %aras3b2g3, %arbs3b2g3 : tensor<f32>
      stablehlo.return %aradds3b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3 = stablehlo.divide %arsums3b2g3, %arns3b2g3 : tensor<1024xf32>
    %v7958 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7959 = stablehlo.multiply %v7958, %s3b2g3 : tensor<1024xf32>
    %v7960 = stablehlo.add %v7959, %armeans3b2g3 : tensor<1024xf32>
    %v7961 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7962 = stablehlo.multiply %v7961, %s3b2g3v : tensor<1024xf32>
    %v7963 = stablehlo.add %v7962, %v7960 : tensor<1024xf32>
    %v7964 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7965 = stablehlo.multiply %v7964, %v7963 : tensor<1024xf32>
    %v7966 = stablehlo.subtract %s3b2g3, %v7965 : tensor<1024xf32>
    %arsums3b2bt3 = "stablehlo.all_reduce"(%v4314) ({
    ^bb0(%aras3b2bt3: tensor<f32>, %arbs3b2bt3: tensor<f32>):
      %aradds3b2bt3 = stablehlo.add %aras3b2bt3, %arbs3b2bt3 : tensor<f32>
      stablehlo.return %aradds3b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2bt3 = stablehlo.divide %arsums3b2bt3, %arns3b2bt3 : tensor<1024xf32>
    %v7967 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7968 = stablehlo.multiply %v7967, %s3b2bt3 : tensor<1024xf32>
    %v7969 = stablehlo.add %v7968, %armeans3b2bt3 : tensor<1024xf32>
    %v7970 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7971 = stablehlo.multiply %v7970, %s3b2bt3v : tensor<1024xf32>
    %v7972 = stablehlo.add %v7971, %v7969 : tensor<1024xf32>
    %v7973 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7974 = stablehlo.multiply %v7973, %v7972 : tensor<1024xf32>
    %v7975 = stablehlo.subtract %s3b2bt3, %v7974 : tensor<1024xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v3989) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x1024x1x1xf32>
    %v7976 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7977 = stablehlo.multiply %v7976, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7978 = stablehlo.add %v7977, %armeans3b3W1 : tensor<256x1024x1x1xf32>
    %v7979 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7980 = stablehlo.multiply %v7979, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7981 = stablehlo.add %v7980, %v7978 : tensor<256x1024x1x1xf32>
    %v7982 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7983 = stablehlo.multiply %v7982, %v7981 : tensor<256x1024x1x1xf32>
    %v7984 = stablehlo.subtract %s3b3W1, %v7983 : tensor<256x1024x1x1xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v4003) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7985 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7986 = stablehlo.multiply %v7985, %s3b3g1 : tensor<256xf32>
    %v7987 = stablehlo.add %v7986, %armeans3b3g1 : tensor<256xf32>
    %v7988 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7989 = stablehlo.multiply %v7988, %s3b3g1v : tensor<256xf32>
    %v7990 = stablehlo.add %v7989, %v7987 : tensor<256xf32>
    %v7991 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7992 = stablehlo.multiply %v7991, %v7990 : tensor<256xf32>
    %v7993 = stablehlo.subtract %s3b3g1, %v7992 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v4006) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7994 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7995 = stablehlo.multiply %v7994, %s3b3bt1 : tensor<256xf32>
    %v7996 = stablehlo.add %v7995, %armeans3b3bt1 : tensor<256xf32>
    %v7997 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7998 = stablehlo.multiply %v7997, %s3b3bt1v : tensor<256xf32>
    %v7999 = stablehlo.add %v7998, %v7996 : tensor<256xf32>
    %v8000 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8001 = stablehlo.multiply %v8000, %v7999 : tensor<256xf32>
    %v8002 = stablehlo.subtract %s3b3bt1, %v8001 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v4015) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v8003 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8004 = stablehlo.multiply %v8003, %s3b3W2 : tensor<256x256x3x3xf32>
    %v8005 = stablehlo.add %v8004, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v8006 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8007 = stablehlo.multiply %v8006, %s3b3W2v : tensor<256x256x3x3xf32>
    %v8008 = stablehlo.add %v8007, %v8005 : tensor<256x256x3x3xf32>
    %v8009 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8010 = stablehlo.multiply %v8009, %v8008 : tensor<256x256x3x3xf32>
    %v8011 = stablehlo.subtract %s3b3W2, %v8010 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v4029) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v8012 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8013 = stablehlo.multiply %v8012, %s3b3g2 : tensor<256xf32>
    %v8014 = stablehlo.add %v8013, %armeans3b3g2 : tensor<256xf32>
    %v8015 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8016 = stablehlo.multiply %v8015, %s3b3g2v : tensor<256xf32>
    %v8017 = stablehlo.add %v8016, %v8014 : tensor<256xf32>
    %v8018 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8019 = stablehlo.multiply %v8018, %v8017 : tensor<256xf32>
    %v8020 = stablehlo.subtract %s3b3g2, %v8019 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v4032) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v8021 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8022 = stablehlo.multiply %v8021, %s3b3bt2 : tensor<256xf32>
    %v8023 = stablehlo.add %v8022, %armeans3b3bt2 : tensor<256xf32>
    %v8024 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8025 = stablehlo.multiply %v8024, %s3b3bt2v : tensor<256xf32>
    %v8026 = stablehlo.add %v8025, %v8023 : tensor<256xf32>
    %v8027 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8028 = stablehlo.multiply %v8027, %v8026 : tensor<256xf32>
    %v8029 = stablehlo.subtract %s3b3bt2, %v8028 : tensor<256xf32>
    %arsums3b3W3 = "stablehlo.all_reduce"(%v4041) ({
    ^bb0(%aras3b3W3: tensor<f32>, %arbs3b3W3: tensor<f32>):
      %aradds3b3W3 = stablehlo.add %aras3b3W3, %arbs3b3W3 : tensor<f32>
      stablehlo.return %aradds3b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b3W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b3W3 = stablehlo.divide %arsums3b3W3, %arns3b3W3 : tensor<1024x256x1x1xf32>
    %v8030 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8031 = stablehlo.multiply %v8030, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v8032 = stablehlo.add %v8031, %armeans3b3W3 : tensor<1024x256x1x1xf32>
    %v8033 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8034 = stablehlo.multiply %v8033, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v8035 = stablehlo.add %v8034, %v8032 : tensor<1024x256x1x1xf32>
    %v8036 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8037 = stablehlo.multiply %v8036, %v8035 : tensor<1024x256x1x1xf32>
    %v8038 = stablehlo.subtract %s3b3W3, %v8037 : tensor<1024x256x1x1xf32>
    %arsums3b3g3 = "stablehlo.all_reduce"(%v4055) ({
    ^bb0(%aras3b3g3: tensor<f32>, %arbs3b3g3: tensor<f32>):
      %aradds3b3g3 = stablehlo.add %aras3b3g3, %arbs3b3g3 : tensor<f32>
      stablehlo.return %aradds3b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3 = stablehlo.divide %arsums3b3g3, %arns3b3g3 : tensor<1024xf32>
    %v8039 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8040 = stablehlo.multiply %v8039, %s3b3g3 : tensor<1024xf32>
    %v8041 = stablehlo.add %v8040, %armeans3b3g3 : tensor<1024xf32>
    %v8042 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8043 = stablehlo.multiply %v8042, %s3b3g3v : tensor<1024xf32>
    %v8044 = stablehlo.add %v8043, %v8041 : tensor<1024xf32>
    %v8045 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8046 = stablehlo.multiply %v8045, %v8044 : tensor<1024xf32>
    %v8047 = stablehlo.subtract %s3b3g3, %v8046 : tensor<1024xf32>
    %arsums3b3bt3 = "stablehlo.all_reduce"(%v4058) ({
    ^bb0(%aras3b3bt3: tensor<f32>, %arbs3b3bt3: tensor<f32>):
      %aradds3b3bt3 = stablehlo.add %aras3b3bt3, %arbs3b3bt3 : tensor<f32>
      stablehlo.return %aradds3b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3bt3 = stablehlo.divide %arsums3b3bt3, %arns3b3bt3 : tensor<1024xf32>
    %v8048 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8049 = stablehlo.multiply %v8048, %s3b3bt3 : tensor<1024xf32>
    %v8050 = stablehlo.add %v8049, %armeans3b3bt3 : tensor<1024xf32>
    %v8051 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8052 = stablehlo.multiply %v8051, %s3b3bt3v : tensor<1024xf32>
    %v8053 = stablehlo.add %v8052, %v8050 : tensor<1024xf32>
    %v8054 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8055 = stablehlo.multiply %v8054, %v8053 : tensor<1024xf32>
    %v8056 = stablehlo.subtract %s3b3bt3, %v8055 : tensor<1024xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v3733) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x1024x1x1xf32>
    %v8057 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8058 = stablehlo.multiply %v8057, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v8059 = stablehlo.add %v8058, %armeans3b4W1 : tensor<256x1024x1x1xf32>
    %v8060 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8061 = stablehlo.multiply %v8060, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v8062 = stablehlo.add %v8061, %v8059 : tensor<256x1024x1x1xf32>
    %v8063 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8064 = stablehlo.multiply %v8063, %v8062 : tensor<256x1024x1x1xf32>
    %v8065 = stablehlo.subtract %s3b4W1, %v8064 : tensor<256x1024x1x1xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v3747) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v8066 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8067 = stablehlo.multiply %v8066, %s3b4g1 : tensor<256xf32>
    %v8068 = stablehlo.add %v8067, %armeans3b4g1 : tensor<256xf32>
    %v8069 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8070 = stablehlo.multiply %v8069, %s3b4g1v : tensor<256xf32>
    %v8071 = stablehlo.add %v8070, %v8068 : tensor<256xf32>
    %v8072 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8073 = stablehlo.multiply %v8072, %v8071 : tensor<256xf32>
    %v8074 = stablehlo.subtract %s3b4g1, %v8073 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v3750) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v8075 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8076 = stablehlo.multiply %v8075, %s3b4bt1 : tensor<256xf32>
    %v8077 = stablehlo.add %v8076, %armeans3b4bt1 : tensor<256xf32>
    %v8078 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8079 = stablehlo.multiply %v8078, %s3b4bt1v : tensor<256xf32>
    %v8080 = stablehlo.add %v8079, %v8077 : tensor<256xf32>
    %v8081 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8082 = stablehlo.multiply %v8081, %v8080 : tensor<256xf32>
    %v8083 = stablehlo.subtract %s3b4bt1, %v8082 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v3759) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v8084 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8085 = stablehlo.multiply %v8084, %s3b4W2 : tensor<256x256x3x3xf32>
    %v8086 = stablehlo.add %v8085, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v8087 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8088 = stablehlo.multiply %v8087, %s3b4W2v : tensor<256x256x3x3xf32>
    %v8089 = stablehlo.add %v8088, %v8086 : tensor<256x256x3x3xf32>
    %v8090 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8091 = stablehlo.multiply %v8090, %v8089 : tensor<256x256x3x3xf32>
    %v8092 = stablehlo.subtract %s3b4W2, %v8091 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v3773) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v8093 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8094 = stablehlo.multiply %v8093, %s3b4g2 : tensor<256xf32>
    %v8095 = stablehlo.add %v8094, %armeans3b4g2 : tensor<256xf32>
    %v8096 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8097 = stablehlo.multiply %v8096, %s3b4g2v : tensor<256xf32>
    %v8098 = stablehlo.add %v8097, %v8095 : tensor<256xf32>
    %v8099 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8100 = stablehlo.multiply %v8099, %v8098 : tensor<256xf32>
    %v8101 = stablehlo.subtract %s3b4g2, %v8100 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v3776) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v8102 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8103 = stablehlo.multiply %v8102, %s3b4bt2 : tensor<256xf32>
    %v8104 = stablehlo.add %v8103, %armeans3b4bt2 : tensor<256xf32>
    %v8105 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8106 = stablehlo.multiply %v8105, %s3b4bt2v : tensor<256xf32>
    %v8107 = stablehlo.add %v8106, %v8104 : tensor<256xf32>
    %v8108 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8109 = stablehlo.multiply %v8108, %v8107 : tensor<256xf32>
    %v8110 = stablehlo.subtract %s3b4bt2, %v8109 : tensor<256xf32>
    %arsums3b4W3 = "stablehlo.all_reduce"(%v3785) ({
    ^bb0(%aras3b4W3: tensor<f32>, %arbs3b4W3: tensor<f32>):
      %aradds3b4W3 = stablehlo.add %aras3b4W3, %arbs3b4W3 : tensor<f32>
      stablehlo.return %aradds3b4W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b4W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b4W3 = stablehlo.divide %arsums3b4W3, %arns3b4W3 : tensor<1024x256x1x1xf32>
    %v8111 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8112 = stablehlo.multiply %v8111, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v8113 = stablehlo.add %v8112, %armeans3b4W3 : tensor<1024x256x1x1xf32>
    %v8114 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8115 = stablehlo.multiply %v8114, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v8116 = stablehlo.add %v8115, %v8113 : tensor<1024x256x1x1xf32>
    %v8117 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8118 = stablehlo.multiply %v8117, %v8116 : tensor<1024x256x1x1xf32>
    %v8119 = stablehlo.subtract %s3b4W3, %v8118 : tensor<1024x256x1x1xf32>
    %arsums3b4g3 = "stablehlo.all_reduce"(%v3799) ({
    ^bb0(%aras3b4g3: tensor<f32>, %arbs3b4g3: tensor<f32>):
      %aradds3b4g3 = stablehlo.add %aras3b4g3, %arbs3b4g3 : tensor<f32>
      stablehlo.return %aradds3b4g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3 = stablehlo.divide %arsums3b4g3, %arns3b4g3 : tensor<1024xf32>
    %v8120 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8121 = stablehlo.multiply %v8120, %s3b4g3 : tensor<1024xf32>
    %v8122 = stablehlo.add %v8121, %armeans3b4g3 : tensor<1024xf32>
    %v8123 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8124 = stablehlo.multiply %v8123, %s3b4g3v : tensor<1024xf32>
    %v8125 = stablehlo.add %v8124, %v8122 : tensor<1024xf32>
    %v8126 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8127 = stablehlo.multiply %v8126, %v8125 : tensor<1024xf32>
    %v8128 = stablehlo.subtract %s3b4g3, %v8127 : tensor<1024xf32>
    %arsums3b4bt3 = "stablehlo.all_reduce"(%v3802) ({
    ^bb0(%aras3b4bt3: tensor<f32>, %arbs3b4bt3: tensor<f32>):
      %aradds3b4bt3 = stablehlo.add %aras3b4bt3, %arbs3b4bt3 : tensor<f32>
      stablehlo.return %aradds3b4bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4bt3 = stablehlo.divide %arsums3b4bt3, %arns3b4bt3 : tensor<1024xf32>
    %v8129 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8130 = stablehlo.multiply %v8129, %s3b4bt3 : tensor<1024xf32>
    %v8131 = stablehlo.add %v8130, %armeans3b4bt3 : tensor<1024xf32>
    %v8132 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8133 = stablehlo.multiply %v8132, %s3b4bt3v : tensor<1024xf32>
    %v8134 = stablehlo.add %v8133, %v8131 : tensor<1024xf32>
    %v8135 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8136 = stablehlo.multiply %v8135, %v8134 : tensor<1024xf32>
    %v8137 = stablehlo.subtract %s3b4bt3, %v8136 : tensor<1024xf32>
    %arsums3b5W1 = "stablehlo.all_reduce"(%v3477) ({
    ^bb0(%aras3b5W1: tensor<f32>, %arbs3b5W1: tensor<f32>):
      %aradds3b5W1 = stablehlo.add %aras3b5W1, %arbs3b5W1 : tensor<f32>
      stablehlo.return %aradds3b5W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b5W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b5W1 = stablehlo.divide %arsums3b5W1, %arns3b5W1 : tensor<256x1024x1x1xf32>
    %v8138 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8139 = stablehlo.multiply %v8138, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v8140 = stablehlo.add %v8139, %armeans3b5W1 : tensor<256x1024x1x1xf32>
    %v8141 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8142 = stablehlo.multiply %v8141, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v8143 = stablehlo.add %v8142, %v8140 : tensor<256x1024x1x1xf32>
    %v8144 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v8145 = stablehlo.multiply %v8144, %v8143 : tensor<256x1024x1x1xf32>
    %v8146 = stablehlo.subtract %s3b5W1, %v8145 : tensor<256x1024x1x1xf32>
    %arsums3b5g1 = "stablehlo.all_reduce"(%v3491) ({
    ^bb0(%aras3b5g1: tensor<f32>, %arbs3b5g1: tensor<f32>):
      %aradds3b5g1 = stablehlo.add %aras3b5g1, %arbs3b5g1 : tensor<f32>
      stablehlo.return %aradds3b5g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1 = stablehlo.divide %arsums3b5g1, %arns3b5g1 : tensor<256xf32>
    %v8147 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8148 = stablehlo.multiply %v8147, %s3b5g1 : tensor<256xf32>
    %v8149 = stablehlo.add %v8148, %armeans3b5g1 : tensor<256xf32>
    %v8150 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8151 = stablehlo.multiply %v8150, %s3b5g1v : tensor<256xf32>
    %v8152 = stablehlo.add %v8151, %v8149 : tensor<256xf32>
    %v8153 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8154 = stablehlo.multiply %v8153, %v8152 : tensor<256xf32>
    %v8155 = stablehlo.subtract %s3b5g1, %v8154 : tensor<256xf32>
    %arsums3b5bt1 = "stablehlo.all_reduce"(%v3494) ({
    ^bb0(%aras3b5bt1: tensor<f32>, %arbs3b5bt1: tensor<f32>):
      %aradds3b5bt1 = stablehlo.add %aras3b5bt1, %arbs3b5bt1 : tensor<f32>
      stablehlo.return %aradds3b5bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt1 = stablehlo.divide %arsums3b5bt1, %arns3b5bt1 : tensor<256xf32>
    %v8156 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8157 = stablehlo.multiply %v8156, %s3b5bt1 : tensor<256xf32>
    %v8158 = stablehlo.add %v8157, %armeans3b5bt1 : tensor<256xf32>
    %v8159 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8160 = stablehlo.multiply %v8159, %s3b5bt1v : tensor<256xf32>
    %v8161 = stablehlo.add %v8160, %v8158 : tensor<256xf32>
    %v8162 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8163 = stablehlo.multiply %v8162, %v8161 : tensor<256xf32>
    %v8164 = stablehlo.subtract %s3b5bt1, %v8163 : tensor<256xf32>
    %arsums3b5W2 = "stablehlo.all_reduce"(%v3503) ({
    ^bb0(%aras3b5W2: tensor<f32>, %arbs3b5W2: tensor<f32>):
      %aradds3b5W2 = stablehlo.add %aras3b5W2, %arbs3b5W2 : tensor<f32>
      stablehlo.return %aradds3b5W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b5W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b5W2 = stablehlo.divide %arsums3b5W2, %arns3b5W2 : tensor<256x256x3x3xf32>
    %v8165 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8166 = stablehlo.multiply %v8165, %s3b5W2 : tensor<256x256x3x3xf32>
    %v8167 = stablehlo.add %v8166, %armeans3b5W2 : tensor<256x256x3x3xf32>
    %v8168 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8169 = stablehlo.multiply %v8168, %s3b5W2v : tensor<256x256x3x3xf32>
    %v8170 = stablehlo.add %v8169, %v8167 : tensor<256x256x3x3xf32>
    %v8171 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v8172 = stablehlo.multiply %v8171, %v8170 : tensor<256x256x3x3xf32>
    %v8173 = stablehlo.subtract %s3b5W2, %v8172 : tensor<256x256x3x3xf32>
    %arsums3b5g2 = "stablehlo.all_reduce"(%v3517) ({
    ^bb0(%aras3b5g2: tensor<f32>, %arbs3b5g2: tensor<f32>):
      %aradds3b5g2 = stablehlo.add %aras3b5g2, %arbs3b5g2 : tensor<f32>
      stablehlo.return %aradds3b5g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2 = stablehlo.divide %arsums3b5g2, %arns3b5g2 : tensor<256xf32>
    %v8174 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8175 = stablehlo.multiply %v8174, %s3b5g2 : tensor<256xf32>
    %v8176 = stablehlo.add %v8175, %armeans3b5g2 : tensor<256xf32>
    %v8177 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8178 = stablehlo.multiply %v8177, %s3b5g2v : tensor<256xf32>
    %v8179 = stablehlo.add %v8178, %v8176 : tensor<256xf32>
    %v8180 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8181 = stablehlo.multiply %v8180, %v8179 : tensor<256xf32>
    %v8182 = stablehlo.subtract %s3b5g2, %v8181 : tensor<256xf32>
    %arsums3b5bt2 = "stablehlo.all_reduce"(%v3520) ({
    ^bb0(%aras3b5bt2: tensor<f32>, %arbs3b5bt2: tensor<f32>):
      %aradds3b5bt2 = stablehlo.add %aras3b5bt2, %arbs3b5bt2 : tensor<f32>
      stablehlo.return %aradds3b5bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt2 = stablehlo.divide %arsums3b5bt2, %arns3b5bt2 : tensor<256xf32>
    %v8183 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8184 = stablehlo.multiply %v8183, %s3b5bt2 : tensor<256xf32>
    %v8185 = stablehlo.add %v8184, %armeans3b5bt2 : tensor<256xf32>
    %v8186 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8187 = stablehlo.multiply %v8186, %s3b5bt2v : tensor<256xf32>
    %v8188 = stablehlo.add %v8187, %v8185 : tensor<256xf32>
    %v8189 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v8190 = stablehlo.multiply %v8189, %v8188 : tensor<256xf32>
    %v8191 = stablehlo.subtract %s3b5bt2, %v8190 : tensor<256xf32>
    %arsums3b5W3 = "stablehlo.all_reduce"(%v3529) ({
    ^bb0(%aras3b5W3: tensor<f32>, %arbs3b5W3: tensor<f32>):
      %aradds3b5W3 = stablehlo.add %aras3b5W3, %arbs3b5W3 : tensor<f32>
      stablehlo.return %aradds3b5W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b5W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b5W3 = stablehlo.divide %arsums3b5W3, %arns3b5W3 : tensor<1024x256x1x1xf32>
    %v8192 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8193 = stablehlo.multiply %v8192, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v8194 = stablehlo.add %v8193, %armeans3b5W3 : tensor<1024x256x1x1xf32>
    %v8195 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8196 = stablehlo.multiply %v8195, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v8197 = stablehlo.add %v8196, %v8194 : tensor<1024x256x1x1xf32>
    %v8198 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v8199 = stablehlo.multiply %v8198, %v8197 : tensor<1024x256x1x1xf32>
    %v8200 = stablehlo.subtract %s3b5W3, %v8199 : tensor<1024x256x1x1xf32>
    %arsums3b5g3 = "stablehlo.all_reduce"(%v3543) ({
    ^bb0(%aras3b5g3: tensor<f32>, %arbs3b5g3: tensor<f32>):
      %aradds3b5g3 = stablehlo.add %aras3b5g3, %arbs3b5g3 : tensor<f32>
      stablehlo.return %aradds3b5g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3 = stablehlo.divide %arsums3b5g3, %arns3b5g3 : tensor<1024xf32>
    %v8201 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8202 = stablehlo.multiply %v8201, %s3b5g3 : tensor<1024xf32>
    %v8203 = stablehlo.add %v8202, %armeans3b5g3 : tensor<1024xf32>
    %v8204 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8205 = stablehlo.multiply %v8204, %s3b5g3v : tensor<1024xf32>
    %v8206 = stablehlo.add %v8205, %v8203 : tensor<1024xf32>
    %v8207 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8208 = stablehlo.multiply %v8207, %v8206 : tensor<1024xf32>
    %v8209 = stablehlo.subtract %s3b5g3, %v8208 : tensor<1024xf32>
    %arsums3b5bt3 = "stablehlo.all_reduce"(%v3546) ({
    ^bb0(%aras3b5bt3: tensor<f32>, %arbs3b5bt3: tensor<f32>):
      %aradds3b5bt3 = stablehlo.add %aras3b5bt3, %arbs3b5bt3 : tensor<f32>
      stablehlo.return %aradds3b5bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5bt3 = stablehlo.divide %arsums3b5bt3, %arns3b5bt3 : tensor<1024xf32>
    %v8210 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8211 = stablehlo.multiply %v8210, %s3b5bt3 : tensor<1024xf32>
    %v8212 = stablehlo.add %v8211, %armeans3b5bt3 : tensor<1024xf32>
    %v8213 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8214 = stablehlo.multiply %v8213, %s3b5bt3v : tensor<1024xf32>
    %v8215 = stablehlo.add %v8214, %v8212 : tensor<1024xf32>
    %v8216 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v8217 = stablehlo.multiply %v8216, %v8215 : tensor<1024xf32>
    %v8218 = stablehlo.subtract %s3b5bt3, %v8217 : tensor<1024xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v3191) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x1024x1x1xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x1024x1x1xf32>
    %v8219 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v8220 = stablehlo.multiply %v8219, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v8221 = stablehlo.add %v8220, %armeans4b0W1 : tensor<512x1024x1x1xf32>
    %v8222 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v8223 = stablehlo.multiply %v8222, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v8224 = stablehlo.add %v8223, %v8221 : tensor<512x1024x1x1xf32>
    %v8225 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v8226 = stablehlo.multiply %v8225, %v8224 : tensor<512x1024x1x1xf32>
    %v8227 = stablehlo.subtract %s4b0W1, %v8226 : tensor<512x1024x1x1xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v3205) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v8228 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8229 = stablehlo.multiply %v8228, %s4b0g1 : tensor<512xf32>
    %v8230 = stablehlo.add %v8229, %armeans4b0g1 : tensor<512xf32>
    %v8231 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8232 = stablehlo.multiply %v8231, %s4b0g1v : tensor<512xf32>
    %v8233 = stablehlo.add %v8232, %v8230 : tensor<512xf32>
    %v8234 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8235 = stablehlo.multiply %v8234, %v8233 : tensor<512xf32>
    %v8236 = stablehlo.subtract %s4b0g1, %v8235 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v3208) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v8237 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8238 = stablehlo.multiply %v8237, %s4b0bt1 : tensor<512xf32>
    %v8239 = stablehlo.add %v8238, %armeans4b0bt1 : tensor<512xf32>
    %v8240 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8241 = stablehlo.multiply %v8240, %s4b0bt1v : tensor<512xf32>
    %v8242 = stablehlo.add %v8241, %v8239 : tensor<512xf32>
    %v8243 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8244 = stablehlo.multiply %v8243, %v8242 : tensor<512xf32>
    %v8245 = stablehlo.subtract %s4b0bt1, %v8244 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v3219) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v8246 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8247 = stablehlo.multiply %v8246, %s4b0W2 : tensor<512x512x3x3xf32>
    %v8248 = stablehlo.add %v8247, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v8249 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8250 = stablehlo.multiply %v8249, %s4b0W2v : tensor<512x512x3x3xf32>
    %v8251 = stablehlo.add %v8250, %v8248 : tensor<512x512x3x3xf32>
    %v8252 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8253 = stablehlo.multiply %v8252, %v8251 : tensor<512x512x3x3xf32>
    %v8254 = stablehlo.subtract %s4b0W2, %v8253 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v3233) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v8255 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8256 = stablehlo.multiply %v8255, %s4b0g2 : tensor<512xf32>
    %v8257 = stablehlo.add %v8256, %armeans4b0g2 : tensor<512xf32>
    %v8258 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8259 = stablehlo.multiply %v8258, %s4b0g2v : tensor<512xf32>
    %v8260 = stablehlo.add %v8259, %v8257 : tensor<512xf32>
    %v8261 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8262 = stablehlo.multiply %v8261, %v8260 : tensor<512xf32>
    %v8263 = stablehlo.subtract %s4b0g2, %v8262 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v3236) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v8264 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8265 = stablehlo.multiply %v8264, %s4b0bt2 : tensor<512xf32>
    %v8266 = stablehlo.add %v8265, %armeans4b0bt2 : tensor<512xf32>
    %v8267 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8268 = stablehlo.multiply %v8267, %s4b0bt2v : tensor<512xf32>
    %v8269 = stablehlo.add %v8268, %v8266 : tensor<512xf32>
    %v8270 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8271 = stablehlo.multiply %v8270, %v8269 : tensor<512xf32>
    %v8272 = stablehlo.subtract %s4b0bt2, %v8271 : tensor<512xf32>
    %arsums4b0W3 = "stablehlo.all_reduce"(%v3245) ({
    ^bb0(%aras4b0W3: tensor<f32>, %arbs4b0W3: tensor<f32>):
      %aradds4b0W3 = stablehlo.add %aras4b0W3, %arbs4b0W3 : tensor<f32>
      stablehlo.return %aradds4b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b0W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b0W3 = stablehlo.divide %arsums4b0W3, %arns4b0W3 : tensor<2048x512x1x1xf32>
    %v8273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8274 = stablehlo.multiply %v8273, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v8275 = stablehlo.add %v8274, %armeans4b0W3 : tensor<2048x512x1x1xf32>
    %v8276 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8277 = stablehlo.multiply %v8276, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v8278 = stablehlo.add %v8277, %v8275 : tensor<2048x512x1x1xf32>
    %v8279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8280 = stablehlo.multiply %v8279, %v8278 : tensor<2048x512x1x1xf32>
    %v8281 = stablehlo.subtract %s4b0W3, %v8280 : tensor<2048x512x1x1xf32>
    %arsums4b0g3 = "stablehlo.all_reduce"(%v3259) ({
    ^bb0(%aras4b0g3: tensor<f32>, %arbs4b0g3: tensor<f32>):
      %aradds4b0g3 = stablehlo.add %aras4b0g3, %arbs4b0g3 : tensor<f32>
      stablehlo.return %aradds4b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3 = stablehlo.divide %arsums4b0g3, %arns4b0g3 : tensor<2048xf32>
    %v8282 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8283 = stablehlo.multiply %v8282, %s4b0g3 : tensor<2048xf32>
    %v8284 = stablehlo.add %v8283, %armeans4b0g3 : tensor<2048xf32>
    %v8285 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8286 = stablehlo.multiply %v8285, %s4b0g3v : tensor<2048xf32>
    %v8287 = stablehlo.add %v8286, %v8284 : tensor<2048xf32>
    %v8288 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8289 = stablehlo.multiply %v8288, %v8287 : tensor<2048xf32>
    %v8290 = stablehlo.subtract %s4b0g3, %v8289 : tensor<2048xf32>
    %arsums4b0bt3 = "stablehlo.all_reduce"(%v3262) ({
    ^bb0(%aras4b0bt3: tensor<f32>, %arbs4b0bt3: tensor<f32>):
      %aradds4b0bt3 = stablehlo.add %aras4b0bt3, %arbs4b0bt3 : tensor<f32>
      stablehlo.return %aradds4b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0bt3 = stablehlo.divide %arsums4b0bt3, %arns4b0bt3 : tensor<2048xf32>
    %v8291 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8292 = stablehlo.multiply %v8291, %s4b0bt3 : tensor<2048xf32>
    %v8293 = stablehlo.add %v8292, %armeans4b0bt3 : tensor<2048xf32>
    %v8294 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8295 = stablehlo.multiply %v8294, %s4b0bt3v : tensor<2048xf32>
    %v8296 = stablehlo.add %v8295, %v8293 : tensor<2048xf32>
    %v8297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8298 = stablehlo.multiply %v8297, %v8296 : tensor<2048xf32>
    %v8299 = stablehlo.subtract %s4b0bt3, %v8298 : tensor<2048xf32>
    %arsums4b0Wp = "stablehlo.all_reduce"(%v3273) ({
    ^bb0(%aras4b0Wp: tensor<f32>, %arbs4b0Wp: tensor<f32>):
      %aradds4b0Wp = stablehlo.add %aras4b0Wp, %arbs4b0Wp : tensor<f32>
      stablehlo.return %aradds4b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %arns4b0Wp = stablehlo.constant dense<4.0> : tensor<2048x1024x1x1xf32>
    %armeans4b0Wp = stablehlo.divide %arsums4b0Wp, %arns4b0Wp : tensor<2048x1024x1x1xf32>
    %v8300 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v8301 = stablehlo.multiply %v8300, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v8302 = stablehlo.add %v8301, %armeans4b0Wp : tensor<2048x1024x1x1xf32>
    %v8303 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v8304 = stablehlo.multiply %v8303, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v8305 = stablehlo.add %v8304, %v8302 : tensor<2048x1024x1x1xf32>
    %v8306 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v8307 = stablehlo.multiply %v8306, %v8305 : tensor<2048x1024x1x1xf32>
    %v8308 = stablehlo.subtract %s4b0Wp, %v8307 : tensor<2048x1024x1x1xf32>
    %arsums4b0gp = "stablehlo.all_reduce"(%v3287) ({
    ^bb0(%aras4b0gp: tensor<f32>, %arbs4b0gp: tensor<f32>):
      %aradds4b0gp = stablehlo.add %aras4b0gp, %arbs4b0gp : tensor<f32>
      stablehlo.return %aradds4b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gp = stablehlo.divide %arsums4b0gp, %arns4b0gp : tensor<2048xf32>
    %v8309 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8310 = stablehlo.multiply %v8309, %s4b0gp : tensor<2048xf32>
    %v8311 = stablehlo.add %v8310, %armeans4b0gp : tensor<2048xf32>
    %v8312 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8313 = stablehlo.multiply %v8312, %s4b0gpv : tensor<2048xf32>
    %v8314 = stablehlo.add %v8313, %v8311 : tensor<2048xf32>
    %v8315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8316 = stablehlo.multiply %v8315, %v8314 : tensor<2048xf32>
    %v8317 = stablehlo.subtract %s4b0gp, %v8316 : tensor<2048xf32>
    %arsums4b0btp = "stablehlo.all_reduce"(%v3290) ({
    ^bb0(%aras4b0btp: tensor<f32>, %arbs4b0btp: tensor<f32>):
      %aradds4b0btp = stablehlo.add %aras4b0btp, %arbs4b0btp : tensor<f32>
      stablehlo.return %aradds4b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0btp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0btp = stablehlo.divide %arsums4b0btp, %arns4b0btp : tensor<2048xf32>
    %v8318 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8319 = stablehlo.multiply %v8318, %s4b0btp : tensor<2048xf32>
    %v8320 = stablehlo.add %v8319, %armeans4b0btp : tensor<2048xf32>
    %v8321 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8322 = stablehlo.multiply %v8321, %s4b0btpv : tensor<2048xf32>
    %v8323 = stablehlo.add %v8322, %v8320 : tensor<2048xf32>
    %v8324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8325 = stablehlo.multiply %v8324, %v8323 : tensor<2048xf32>
    %v8326 = stablehlo.subtract %s4b0btp, %v8325 : tensor<2048xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v2879) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x2048x1x1xf32>
    %v8327 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8328 = stablehlo.multiply %v8327, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v8329 = stablehlo.add %v8328, %armeans4b1W1 : tensor<512x2048x1x1xf32>
    %v8330 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8331 = stablehlo.multiply %v8330, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v8332 = stablehlo.add %v8331, %v8329 : tensor<512x2048x1x1xf32>
    %v8333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8334 = stablehlo.multiply %v8333, %v8332 : tensor<512x2048x1x1xf32>
    %v8335 = stablehlo.subtract %s4b1W1, %v8334 : tensor<512x2048x1x1xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v2893) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v8336 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8337 = stablehlo.multiply %v8336, %s4b1g1 : tensor<512xf32>
    %v8338 = stablehlo.add %v8337, %armeans4b1g1 : tensor<512xf32>
    %v8339 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8340 = stablehlo.multiply %v8339, %s4b1g1v : tensor<512xf32>
    %v8341 = stablehlo.add %v8340, %v8338 : tensor<512xf32>
    %v8342 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8343 = stablehlo.multiply %v8342, %v8341 : tensor<512xf32>
    %v8344 = stablehlo.subtract %s4b1g1, %v8343 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v2896) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v8345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8346 = stablehlo.multiply %v8345, %s4b1bt1 : tensor<512xf32>
    %v8347 = stablehlo.add %v8346, %armeans4b1bt1 : tensor<512xf32>
    %v8348 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8349 = stablehlo.multiply %v8348, %s4b1bt1v : tensor<512xf32>
    %v8350 = stablehlo.add %v8349, %v8347 : tensor<512xf32>
    %v8351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8352 = stablehlo.multiply %v8351, %v8350 : tensor<512xf32>
    %v8353 = stablehlo.subtract %s4b1bt1, %v8352 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v2905) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v8354 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8355 = stablehlo.multiply %v8354, %s4b1W2 : tensor<512x512x3x3xf32>
    %v8356 = stablehlo.add %v8355, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v8357 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8358 = stablehlo.multiply %v8357, %s4b1W2v : tensor<512x512x3x3xf32>
    %v8359 = stablehlo.add %v8358, %v8356 : tensor<512x512x3x3xf32>
    %v8360 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8361 = stablehlo.multiply %v8360, %v8359 : tensor<512x512x3x3xf32>
    %v8362 = stablehlo.subtract %s4b1W2, %v8361 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v2919) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v8363 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8364 = stablehlo.multiply %v8363, %s4b1g2 : tensor<512xf32>
    %v8365 = stablehlo.add %v8364, %armeans4b1g2 : tensor<512xf32>
    %v8366 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8367 = stablehlo.multiply %v8366, %s4b1g2v : tensor<512xf32>
    %v8368 = stablehlo.add %v8367, %v8365 : tensor<512xf32>
    %v8369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8370 = stablehlo.multiply %v8369, %v8368 : tensor<512xf32>
    %v8371 = stablehlo.subtract %s4b1g2, %v8370 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v2922) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v8372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8373 = stablehlo.multiply %v8372, %s4b1bt2 : tensor<512xf32>
    %v8374 = stablehlo.add %v8373, %armeans4b1bt2 : tensor<512xf32>
    %v8375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8376 = stablehlo.multiply %v8375, %s4b1bt2v : tensor<512xf32>
    %v8377 = stablehlo.add %v8376, %v8374 : tensor<512xf32>
    %v8378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8379 = stablehlo.multiply %v8378, %v8377 : tensor<512xf32>
    %v8380 = stablehlo.subtract %s4b1bt2, %v8379 : tensor<512xf32>
    %arsums4b1W3 = "stablehlo.all_reduce"(%v2931) ({
    ^bb0(%aras4b1W3: tensor<f32>, %arbs4b1W3: tensor<f32>):
      %aradds4b1W3 = stablehlo.add %aras4b1W3, %arbs4b1W3 : tensor<f32>
      stablehlo.return %aradds4b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b1W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b1W3 = stablehlo.divide %arsums4b1W3, %arns4b1W3 : tensor<2048x512x1x1xf32>
    %v8381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8382 = stablehlo.multiply %v8381, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v8383 = stablehlo.add %v8382, %armeans4b1W3 : tensor<2048x512x1x1xf32>
    %v8384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8385 = stablehlo.multiply %v8384, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v8386 = stablehlo.add %v8385, %v8383 : tensor<2048x512x1x1xf32>
    %v8387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8388 = stablehlo.multiply %v8387, %v8386 : tensor<2048x512x1x1xf32>
    %v8389 = stablehlo.subtract %s4b1W3, %v8388 : tensor<2048x512x1x1xf32>
    %arsums4b1g3 = "stablehlo.all_reduce"(%v2945) ({
    ^bb0(%aras4b1g3: tensor<f32>, %arbs4b1g3: tensor<f32>):
      %aradds4b1g3 = stablehlo.add %aras4b1g3, %arbs4b1g3 : tensor<f32>
      stablehlo.return %aradds4b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3 = stablehlo.divide %arsums4b1g3, %arns4b1g3 : tensor<2048xf32>
    %v8390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8391 = stablehlo.multiply %v8390, %s4b1g3 : tensor<2048xf32>
    %v8392 = stablehlo.add %v8391, %armeans4b1g3 : tensor<2048xf32>
    %v8393 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8394 = stablehlo.multiply %v8393, %s4b1g3v : tensor<2048xf32>
    %v8395 = stablehlo.add %v8394, %v8392 : tensor<2048xf32>
    %v8396 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8397 = stablehlo.multiply %v8396, %v8395 : tensor<2048xf32>
    %v8398 = stablehlo.subtract %s4b1g3, %v8397 : tensor<2048xf32>
    %arsums4b1bt3 = "stablehlo.all_reduce"(%v2948) ({
    ^bb0(%aras4b1bt3: tensor<f32>, %arbs4b1bt3: tensor<f32>):
      %aradds4b1bt3 = stablehlo.add %aras4b1bt3, %arbs4b1bt3 : tensor<f32>
      stablehlo.return %aradds4b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1bt3 = stablehlo.divide %arsums4b1bt3, %arns4b1bt3 : tensor<2048xf32>
    %v8399 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8400 = stablehlo.multiply %v8399, %s4b1bt3 : tensor<2048xf32>
    %v8401 = stablehlo.add %v8400, %armeans4b1bt3 : tensor<2048xf32>
    %v8402 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8403 = stablehlo.multiply %v8402, %s4b1bt3v : tensor<2048xf32>
    %v8404 = stablehlo.add %v8403, %v8401 : tensor<2048xf32>
    %v8405 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8406 = stablehlo.multiply %v8405, %v8404 : tensor<2048xf32>
    %v8407 = stablehlo.subtract %s4b1bt3, %v8406 : tensor<2048xf32>
    %arsums4b2W1 = "stablehlo.all_reduce"(%v2623) ({
    ^bb0(%aras4b2W1: tensor<f32>, %arbs4b2W1: tensor<f32>):
      %aradds4b2W1 = stablehlo.add %aras4b2W1, %arbs4b2W1 : tensor<f32>
      stablehlo.return %aradds4b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b2W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b2W1 = stablehlo.divide %arsums4b2W1, %arns4b2W1 : tensor<512x2048x1x1xf32>
    %v8408 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8409 = stablehlo.multiply %v8408, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v8410 = stablehlo.add %v8409, %armeans4b2W1 : tensor<512x2048x1x1xf32>
    %v8411 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8412 = stablehlo.multiply %v8411, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v8413 = stablehlo.add %v8412, %v8410 : tensor<512x2048x1x1xf32>
    %v8414 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v8415 = stablehlo.multiply %v8414, %v8413 : tensor<512x2048x1x1xf32>
    %v8416 = stablehlo.subtract %s4b2W1, %v8415 : tensor<512x2048x1x1xf32>
    %arsums4b2g1 = "stablehlo.all_reduce"(%v2637) ({
    ^bb0(%aras4b2g1: tensor<f32>, %arbs4b2g1: tensor<f32>):
      %aradds4b2g1 = stablehlo.add %aras4b2g1, %arbs4b2g1 : tensor<f32>
      stablehlo.return %aradds4b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1 = stablehlo.divide %arsums4b2g1, %arns4b2g1 : tensor<512xf32>
    %v8417 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8418 = stablehlo.multiply %v8417, %s4b2g1 : tensor<512xf32>
    %v8419 = stablehlo.add %v8418, %armeans4b2g1 : tensor<512xf32>
    %v8420 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8421 = stablehlo.multiply %v8420, %s4b2g1v : tensor<512xf32>
    %v8422 = stablehlo.add %v8421, %v8419 : tensor<512xf32>
    %v8423 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8424 = stablehlo.multiply %v8423, %v8422 : tensor<512xf32>
    %v8425 = stablehlo.subtract %s4b2g1, %v8424 : tensor<512xf32>
    %arsums4b2bt1 = "stablehlo.all_reduce"(%v2640) ({
    ^bb0(%aras4b2bt1: tensor<f32>, %arbs4b2bt1: tensor<f32>):
      %aradds4b2bt1 = stablehlo.add %aras4b2bt1, %arbs4b2bt1 : tensor<f32>
      stablehlo.return %aradds4b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt1 = stablehlo.divide %arsums4b2bt1, %arns4b2bt1 : tensor<512xf32>
    %v8426 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8427 = stablehlo.multiply %v8426, %s4b2bt1 : tensor<512xf32>
    %v8428 = stablehlo.add %v8427, %armeans4b2bt1 : tensor<512xf32>
    %v8429 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8430 = stablehlo.multiply %v8429, %s4b2bt1v : tensor<512xf32>
    %v8431 = stablehlo.add %v8430, %v8428 : tensor<512xf32>
    %v8432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8433 = stablehlo.multiply %v8432, %v8431 : tensor<512xf32>
    %v8434 = stablehlo.subtract %s4b2bt1, %v8433 : tensor<512xf32>
    %arsums4b2W2 = "stablehlo.all_reduce"(%v2649) ({
    ^bb0(%aras4b2W2: tensor<f32>, %arbs4b2W2: tensor<f32>):
      %aradds4b2W2 = stablehlo.add %aras4b2W2, %arbs4b2W2 : tensor<f32>
      stablehlo.return %aradds4b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b2W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b2W2 = stablehlo.divide %arsums4b2W2, %arns4b2W2 : tensor<512x512x3x3xf32>
    %v8435 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8436 = stablehlo.multiply %v8435, %s4b2W2 : tensor<512x512x3x3xf32>
    %v8437 = stablehlo.add %v8436, %armeans4b2W2 : tensor<512x512x3x3xf32>
    %v8438 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8439 = stablehlo.multiply %v8438, %s4b2W2v : tensor<512x512x3x3xf32>
    %v8440 = stablehlo.add %v8439, %v8437 : tensor<512x512x3x3xf32>
    %v8441 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v8442 = stablehlo.multiply %v8441, %v8440 : tensor<512x512x3x3xf32>
    %v8443 = stablehlo.subtract %s4b2W2, %v8442 : tensor<512x512x3x3xf32>
    %arsums4b2g2 = "stablehlo.all_reduce"(%v2663) ({
    ^bb0(%aras4b2g2: tensor<f32>, %arbs4b2g2: tensor<f32>):
      %aradds4b2g2 = stablehlo.add %aras4b2g2, %arbs4b2g2 : tensor<f32>
      stablehlo.return %aradds4b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2 = stablehlo.divide %arsums4b2g2, %arns4b2g2 : tensor<512xf32>
    %v8444 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8445 = stablehlo.multiply %v8444, %s4b2g2 : tensor<512xf32>
    %v8446 = stablehlo.add %v8445, %armeans4b2g2 : tensor<512xf32>
    %v8447 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8448 = stablehlo.multiply %v8447, %s4b2g2v : tensor<512xf32>
    %v8449 = stablehlo.add %v8448, %v8446 : tensor<512xf32>
    %v8450 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8451 = stablehlo.multiply %v8450, %v8449 : tensor<512xf32>
    %v8452 = stablehlo.subtract %s4b2g2, %v8451 : tensor<512xf32>
    %arsums4b2bt2 = "stablehlo.all_reduce"(%v2666) ({
    ^bb0(%aras4b2bt2: tensor<f32>, %arbs4b2bt2: tensor<f32>):
      %aradds4b2bt2 = stablehlo.add %aras4b2bt2, %arbs4b2bt2 : tensor<f32>
      stablehlo.return %aradds4b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt2 = stablehlo.divide %arsums4b2bt2, %arns4b2bt2 : tensor<512xf32>
    %v8453 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8454 = stablehlo.multiply %v8453, %s4b2bt2 : tensor<512xf32>
    %v8455 = stablehlo.add %v8454, %armeans4b2bt2 : tensor<512xf32>
    %v8456 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8457 = stablehlo.multiply %v8456, %s4b2bt2v : tensor<512xf32>
    %v8458 = stablehlo.add %v8457, %v8455 : tensor<512xf32>
    %v8459 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v8460 = stablehlo.multiply %v8459, %v8458 : tensor<512xf32>
    %v8461 = stablehlo.subtract %s4b2bt2, %v8460 : tensor<512xf32>
    %arsums4b2W3 = "stablehlo.all_reduce"(%v2675) ({
    ^bb0(%aras4b2W3: tensor<f32>, %arbs4b2W3: tensor<f32>):
      %aradds4b2W3 = stablehlo.add %aras4b2W3, %arbs4b2W3 : tensor<f32>
      stablehlo.return %aradds4b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b2W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b2W3 = stablehlo.divide %arsums4b2W3, %arns4b2W3 : tensor<2048x512x1x1xf32>
    %v8462 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8463 = stablehlo.multiply %v8462, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v8464 = stablehlo.add %v8463, %armeans4b2W3 : tensor<2048x512x1x1xf32>
    %v8465 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8466 = stablehlo.multiply %v8465, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v8467 = stablehlo.add %v8466, %v8464 : tensor<2048x512x1x1xf32>
    %v8468 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v8469 = stablehlo.multiply %v8468, %v8467 : tensor<2048x512x1x1xf32>
    %v8470 = stablehlo.subtract %s4b2W3, %v8469 : tensor<2048x512x1x1xf32>
    %arsums4b2g3 = "stablehlo.all_reduce"(%v2689) ({
    ^bb0(%aras4b2g3: tensor<f32>, %arbs4b2g3: tensor<f32>):
      %aradds4b2g3 = stablehlo.add %aras4b2g3, %arbs4b2g3 : tensor<f32>
      stablehlo.return %aradds4b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3 = stablehlo.divide %arsums4b2g3, %arns4b2g3 : tensor<2048xf32>
    %v8471 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8472 = stablehlo.multiply %v8471, %s4b2g3 : tensor<2048xf32>
    %v8473 = stablehlo.add %v8472, %armeans4b2g3 : tensor<2048xf32>
    %v8474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8475 = stablehlo.multiply %v8474, %s4b2g3v : tensor<2048xf32>
    %v8476 = stablehlo.add %v8475, %v8473 : tensor<2048xf32>
    %v8477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8478 = stablehlo.multiply %v8477, %v8476 : tensor<2048xf32>
    %v8479 = stablehlo.subtract %s4b2g3, %v8478 : tensor<2048xf32>
    %arsums4b2bt3 = "stablehlo.all_reduce"(%v2692) ({
    ^bb0(%aras4b2bt3: tensor<f32>, %arbs4b2bt3: tensor<f32>):
      %aradds4b2bt3 = stablehlo.add %aras4b2bt3, %arbs4b2bt3 : tensor<f32>
      stablehlo.return %aradds4b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2bt3 = stablehlo.divide %arsums4b2bt3, %arns4b2bt3 : tensor<2048xf32>
    %v8480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8481 = stablehlo.multiply %v8480, %s4b2bt3 : tensor<2048xf32>
    %v8482 = stablehlo.add %v8481, %armeans4b2bt3 : tensor<2048xf32>
    %v8483 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8484 = stablehlo.multiply %v8483, %s4b2bt3v : tensor<2048xf32>
    %v8485 = stablehlo.add %v8484, %v8482 : tensor<2048xf32>
    %v8486 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v8487 = stablehlo.multiply %v8486, %v8485 : tensor<2048xf32>
    %v8488 = stablehlo.subtract %s4b2bt3, %v8487 : tensor<2048xf32>
    %arsumWd = "stablehlo.all_reduce"(%v2430) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<2048x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<2048x1000xf32>
    %v8489 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8490 = stablehlo.multiply %v8489, %Wd : tensor<2048x1000xf32>
    %v8491 = stablehlo.add %v8490, %armeanWd : tensor<2048x1000xf32>
    %v8492 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8493 = stablehlo.multiply %v8492, %Wdv : tensor<2048x1000xf32>
    %v8494 = stablehlo.add %v8493, %v8491 : tensor<2048x1000xf32>
    %v8495 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v8496 = stablehlo.multiply %v8495, %v8494 : tensor<2048x1000xf32>
    %v8497 = stablehlo.subtract %Wd, %v8496 : tensor<2048x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v2432) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v8498 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8499 = stablehlo.multiply %v8498, %bd : tensor<1000xf32>
    %v8500 = stablehlo.add %v8499, %armeanbd : tensor<1000xf32>
    %v8501 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8502 = stablehlo.multiply %v8501, %bdv : tensor<1000xf32>
    %v8503 = stablehlo.add %v8502, %v8500 : tensor<1000xf32>
    %v8504 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v8505 = stablehlo.multiply %v8504, %v8503 : tensor<1000xf32>
    %v8506 = stablehlo.subtract %bd, %v8505 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v2418 : tensor<64x1000xf32>
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
    return %v7066, %v7075, %v7084, %v7093, %v7102, %v7111, %v7120, %v7129, %v7138, %v7147, %v7156, %v7165, %v7174, %v7183, %v7192, %v7201, %v7210, %v7219, %v7228, %v7237, %v7246, %v7255, %v7264, %v7273, %v7282, %v7291, %v7300, %v7309, %v7318, %v7327, %v7336, %v7345, %v7354, %v7363, %v7372, %v7381, %v7390, %v7399, %v7408, %v7417, %v7426, %v7435, %v7444, %v7453, %v7462, %v7471, %v7480, %v7489, %v7498, %v7507, %v7516, %v7525, %v7534, %v7543, %v7552, %v7561, %v7570, %v7579, %v7588, %v7597, %v7606, %v7615, %v7624, %v7633, %v7642, %v7651, %v7660, %v7669, %v7678, %v7687, %v7696, %v7705, %v7714, %v7723, %v7732, %v7741, %v7750, %v7759, %v7768, %v7777, %v7786, %v7795, %v7804, %v7813, %v7822, %v7831, %v7840, %v7849, %v7858, %v7867, %v7876, %v7885, %v7894, %v7903, %v7912, %v7921, %v7930, %v7939, %v7948, %v7957, %v7966, %v7975, %v7984, %v7993, %v8002, %v8011, %v8020, %v8029, %v8038, %v8047, %v8056, %v8065, %v8074, %v8083, %v8092, %v8101, %v8110, %v8119, %v8128, %v8137, %v8146, %v8155, %v8164, %v8173, %v8182, %v8191, %v8200, %v8209, %v8218, %v8227, %v8236, %v8245, %v8254, %v8263, %v8272, %v8281, %v8290, %v8299, %v8308, %v8317, %v8326, %v8335, %v8344, %v8353, %v8362, %v8371, %v8380, %v8389, %v8398, %v8407, %v8416, %v8425, %v8434, %v8443, %v8452, %v8461, %v8470, %v8479, %v8488, %v8497, %v8506, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v7063, %v7072, %v7081, %v7090, %v7099, %v7108, %v7117, %v7126, %v7135, %v7144, %v7153, %v7162, %v7171, %v7180, %v7189, %v7198, %v7207, %v7216, %v7225, %v7234, %v7243, %v7252, %v7261, %v7270, %v7279, %v7288, %v7297, %v7306, %v7315, %v7324, %v7333, %v7342, %v7351, %v7360, %v7369, %v7378, %v7387, %v7396, %v7405, %v7414, %v7423, %v7432, %v7441, %v7450, %v7459, %v7468, %v7477, %v7486, %v7495, %v7504, %v7513, %v7522, %v7531, %v7540, %v7549, %v7558, %v7567, %v7576, %v7585, %v7594, %v7603, %v7612, %v7621, %v7630, %v7639, %v7648, %v7657, %v7666, %v7675, %v7684, %v7693, %v7702, %v7711, %v7720, %v7729, %v7738, %v7747, %v7756, %v7765, %v7774, %v7783, %v7792, %v7801, %v7810, %v7819, %v7828, %v7837, %v7846, %v7855, %v7864, %v7873, %v7882, %v7891, %v7900, %v7909, %v7918, %v7927, %v7936, %v7945, %v7954, %v7963, %v7972, %v7981, %v7990, %v7999, %v8008, %v8017, %v8026, %v8035, %v8044, %v8053, %v8062, %v8071, %v8080, %v8089, %v8098, %v8107, %v8116, %v8125, %v8134, %v8143, %v8152, %v8161, %v8170, %v8179, %v8188, %v8197, %v8206, %v8215, %v8224, %v8233, %v8242, %v8251, %v8260, %v8269, %v8278, %v8287, %v8296, %v8305, %v8314, %v8323, %v8332, %v8341, %v8350, %v8359, %v8368, %v8377, %v8386, %v8395, %v8404, %v8413, %v8422, %v8431, %v8440, %v8449, %v8458, %v8467, %v8476, %v8485, %v8494, %v8503, %loss, %bc1, %bc2, %v6952, %v6953, %v6954, %v6955, %v6956, %v6957, %v6958, %v6959, %v6960, %v6961, %v6962, %v6963, %v6964, %v6965, %v6966, %v6967, %v6968, %v6969, %v6970, %v6971, %v6972, %v6973, %v6974, %v6975, %v6976, %v6977, %v6978, %v6979, %v6980, %v6981, %v6982, %v6983, %v6984, %v6985, %v6986, %v6987, %v6988, %v6989, %v6990, %v6991, %v6992, %v6993, %v6994, %v6995, %v6996, %v6997, %v6998, %v6999, %v7000, %v7001, %v7002, %v7003, %v7004, %v7005, %v7006, %v7007, %v7008, %v7009, %v7010, %v7011, %v7012, %v7013, %v7014, %v7015, %v7016, %v7017, %v7018, %v7019, %v7020, %v7021, %v7022, %v7023, %v7024, %v7025, %v7026, %v7027, %v7028, %v7029, %v7030, %v7031, %v7032, %v7033, %v7034, %v7035, %v7036, %v7037, %v7038, %v7039, %v7040, %v7041, %v7042, %v7043, %v7044, %v7045, %v7046, %v7047, %v7048, %v7049, %v7050, %v7051, %v7052, %v7053, %v7054, %v7055, %v7056, %v7057 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
