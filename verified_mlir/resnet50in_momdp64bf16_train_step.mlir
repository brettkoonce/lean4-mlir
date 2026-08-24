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
    %v28 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v29 = stablehlo.maximum %v27, %v28 : tensor<64x802816xf32>
    %v30 = stablehlo.reshape %v29 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v31 = stablehlo.constant dense<0xFF800000> : tensor<f32>
    %v32 = "stablehlo.reduce_window"(%v30, %v31) ({
      ^bb0(%pa: tensor<f32>, %pb: tensor<f32>):
        %pm = stablehlo.maximum %pa, %pb : tensor<f32>
        stablehlo.return %pm : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x56x56xf32>
    %v33 = stablehlo.reshape %v32 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v34 = stablehlo.reshape %v33 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v35 = stablehlo.convert %v34 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v36 = stablehlo.convert %s1b0W1 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v37 = stablehlo.convolution(%v35, %v36)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v38 = stablehlo.convert %v37 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v39 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v40 = stablehlo.add %v38, %v39 : tensor<64x64x56x56xf32>
    %v41 = stablehlo.reshape %v40 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v42 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v43 = stablehlo.constant dense<0.0> : tensor<f32>
    %v44 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v45 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v46 = stablehlo.reduce(%v42 init: %v43) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v47 = stablehlo.broadcast_in_dim %v46, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v48 = stablehlo.divide %v47, %v44 : tensor<64x64x56x56xf32>
    %v49 = stablehlo.subtract %v42, %v48 : tensor<64x64x56x56xf32>
    %v50 = stablehlo.multiply %v49, %v49 : tensor<64x64x56x56xf32>
    %v51 = stablehlo.reduce(%v50 init: %v43) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v52 = stablehlo.broadcast_in_dim %v51, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v53 = stablehlo.divide %v52, %v44 : tensor<64x64x56x56xf32>
    %v54 = stablehlo.add %v53, %v45 : tensor<64x64x56x56xf32>
    %v55 = stablehlo.rsqrt %v54 : tensor<64x64x56x56xf32>
    %v56 = stablehlo.multiply %v49, %v55 : tensor<64x64x56x56xf32>
    %v57 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v58 = stablehlo.broadcast_in_dim %s1b0bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v59 = stablehlo.multiply %v56, %v57 : tensor<64x64x56x56xf32>
    %v60 = stablehlo.add %v59, %v58 : tensor<64x64x56x56xf32>
    %v61 = stablehlo.reshape %v60 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v62 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v63 = stablehlo.maximum %v61, %v62 : tensor<64x200704xf32>
    %v64 = stablehlo.reshape %v63 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v65 = stablehlo.convert %v64 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v66 = stablehlo.convert %s1b0W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v67 = stablehlo.convolution(%v65, %v66)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v68 = stablehlo.convert %v67 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v69 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v70 = stablehlo.add %v68, %v69 : tensor<64x64x56x56xf32>
    %v71 = stablehlo.reshape %v70 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v72 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v73 = stablehlo.constant dense<0.0> : tensor<f32>
    %v74 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v75 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v76 = stablehlo.reduce(%v72 init: %v73) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v77 = stablehlo.broadcast_in_dim %v76, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v78 = stablehlo.divide %v77, %v74 : tensor<64x64x56x56xf32>
    %v79 = stablehlo.subtract %v72, %v78 : tensor<64x64x56x56xf32>
    %v80 = stablehlo.multiply %v79, %v79 : tensor<64x64x56x56xf32>
    %v81 = stablehlo.reduce(%v80 init: %v73) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v82 = stablehlo.broadcast_in_dim %v81, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v83 = stablehlo.divide %v82, %v74 : tensor<64x64x56x56xf32>
    %v84 = stablehlo.add %v83, %v75 : tensor<64x64x56x56xf32>
    %v85 = stablehlo.rsqrt %v84 : tensor<64x64x56x56xf32>
    %v86 = stablehlo.multiply %v79, %v85 : tensor<64x64x56x56xf32>
    %v87 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v88 = stablehlo.broadcast_in_dim %s1b0bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v89 = stablehlo.multiply %v86, %v87 : tensor<64x64x56x56xf32>
    %v90 = stablehlo.add %v89, %v88 : tensor<64x64x56x56xf32>
    %v91 = stablehlo.reshape %v90 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v92 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v93 = stablehlo.maximum %v91, %v92 : tensor<64x200704xf32>
    %v94 = stablehlo.reshape %v93 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v95 = stablehlo.convert %v94 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v96 = stablehlo.convert %s1b0W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v97 = stablehlo.convolution(%v95, %v96)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v98 = stablehlo.convert %v97 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v99 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v100 = stablehlo.add %v98, %v99 : tensor<64x256x56x56xf32>
    %v101 = stablehlo.reshape %v100 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v104 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v105 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v106 = stablehlo.reduce(%v102 init: %v103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v107 = stablehlo.broadcast_in_dim %v106, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v108 = stablehlo.divide %v107, %v104 : tensor<64x256x56x56xf32>
    %v109 = stablehlo.subtract %v102, %v108 : tensor<64x256x56x56xf32>
    %v110 = stablehlo.multiply %v109, %v109 : tensor<64x256x56x56xf32>
    %v111 = stablehlo.reduce(%v110 init: %v103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v112 = stablehlo.broadcast_in_dim %v111, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v113 = stablehlo.divide %v112, %v104 : tensor<64x256x56x56xf32>
    %v114 = stablehlo.add %v113, %v105 : tensor<64x256x56x56xf32>
    %v115 = stablehlo.rsqrt %v114 : tensor<64x256x56x56xf32>
    %v116 = stablehlo.multiply %v109, %v115 : tensor<64x256x56x56xf32>
    %v117 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v118 = stablehlo.broadcast_in_dim %s1b0bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v119 = stablehlo.multiply %v116, %v117 : tensor<64x256x56x56xf32>
    %v120 = stablehlo.add %v119, %v118 : tensor<64x256x56x56xf32>
    %v121 = stablehlo.reshape %v120 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v122 = stablehlo.reshape %v33 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v123 = stablehlo.convert %v122 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v124 = stablehlo.convert %s1b0Wp : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v125 = stablehlo.convolution(%v123, %v124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v126 = stablehlo.convert %v125 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v127 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v128 = stablehlo.add %v126, %v127 : tensor<64x256x56x56xf32>
    %v129 = stablehlo.reshape %v128 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v130 = stablehlo.reshape %v129 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v132 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v133 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v134 = stablehlo.reduce(%v130 init: %v131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v135 = stablehlo.broadcast_in_dim %v134, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v136 = stablehlo.divide %v135, %v132 : tensor<64x256x56x56xf32>
    %v137 = stablehlo.subtract %v130, %v136 : tensor<64x256x56x56xf32>
    %v138 = stablehlo.multiply %v137, %v137 : tensor<64x256x56x56xf32>
    %v139 = stablehlo.reduce(%v138 init: %v131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v140 = stablehlo.broadcast_in_dim %v139, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v141 = stablehlo.divide %v140, %v132 : tensor<64x256x56x56xf32>
    %v142 = stablehlo.add %v141, %v133 : tensor<64x256x56x56xf32>
    %v143 = stablehlo.rsqrt %v142 : tensor<64x256x56x56xf32>
    %v144 = stablehlo.multiply %v137, %v143 : tensor<64x256x56x56xf32>
    %v145 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v146 = stablehlo.broadcast_in_dim %s1b0btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v147 = stablehlo.multiply %v144, %v145 : tensor<64x256x56x56xf32>
    %v148 = stablehlo.add %v147, %v146 : tensor<64x256x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v150 = stablehlo.add %v121, %v149 : tensor<64x802816xf32>
    %v151 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v152 = stablehlo.maximum %v150, %v151 : tensor<64x802816xf32>
    %v153 = stablehlo.reshape %v152 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v154 = stablehlo.convert %v153 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v155 = stablehlo.convert %s1b1W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v156 = stablehlo.convolution(%v154, %v155)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v157 = stablehlo.convert %v156 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v158 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v159 = stablehlo.add %v157, %v158 : tensor<64x64x56x56xf32>
    %v160 = stablehlo.reshape %v159 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v161 = stablehlo.reshape %v160 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v163 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v164 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v165 = stablehlo.reduce(%v161 init: %v162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v166 = stablehlo.broadcast_in_dim %v165, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v167 = stablehlo.divide %v166, %v163 : tensor<64x64x56x56xf32>
    %v168 = stablehlo.subtract %v161, %v167 : tensor<64x64x56x56xf32>
    %v169 = stablehlo.multiply %v168, %v168 : tensor<64x64x56x56xf32>
    %v170 = stablehlo.reduce(%v169 init: %v162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v171 = stablehlo.broadcast_in_dim %v170, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v172 = stablehlo.divide %v171, %v163 : tensor<64x64x56x56xf32>
    %v173 = stablehlo.add %v172, %v164 : tensor<64x64x56x56xf32>
    %v174 = stablehlo.rsqrt %v173 : tensor<64x64x56x56xf32>
    %v175 = stablehlo.multiply %v168, %v174 : tensor<64x64x56x56xf32>
    %v176 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v177 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v178 = stablehlo.multiply %v175, %v176 : tensor<64x64x56x56xf32>
    %v179 = stablehlo.add %v178, %v177 : tensor<64x64x56x56xf32>
    %v180 = stablehlo.reshape %v179 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v181 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v182 = stablehlo.maximum %v180, %v181 : tensor<64x200704xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v184 = stablehlo.convert %v183 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v185 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v186 = stablehlo.convolution(%v184, %v185)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v187 = stablehlo.convert %v186 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v188 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v189 = stablehlo.add %v187, %v188 : tensor<64x64x56x56xf32>
    %v190 = stablehlo.reshape %v189 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v191 = stablehlo.reshape %v190 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v192 = stablehlo.constant dense<0.0> : tensor<f32>
    %v193 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v194 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v195 = stablehlo.reduce(%v191 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v196 = stablehlo.broadcast_in_dim %v195, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v197 = stablehlo.divide %v196, %v193 : tensor<64x64x56x56xf32>
    %v198 = stablehlo.subtract %v191, %v197 : tensor<64x64x56x56xf32>
    %v199 = stablehlo.multiply %v198, %v198 : tensor<64x64x56x56xf32>
    %v200 = stablehlo.reduce(%v199 init: %v192) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v201 = stablehlo.broadcast_in_dim %v200, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v202 = stablehlo.divide %v201, %v193 : tensor<64x64x56x56xf32>
    %v203 = stablehlo.add %v202, %v194 : tensor<64x64x56x56xf32>
    %v204 = stablehlo.rsqrt %v203 : tensor<64x64x56x56xf32>
    %v205 = stablehlo.multiply %v198, %v204 : tensor<64x64x56x56xf32>
    %v206 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v207 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v208 = stablehlo.multiply %v205, %v206 : tensor<64x64x56x56xf32>
    %v209 = stablehlo.add %v208, %v207 : tensor<64x64x56x56xf32>
    %v210 = stablehlo.reshape %v209 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v211 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v212 = stablehlo.maximum %v210, %v211 : tensor<64x200704xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v214 = stablehlo.convert %v213 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v215 = stablehlo.convert %s1b1W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v216 = stablehlo.convolution(%v214, %v215)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v217 = stablehlo.convert %v216 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v218 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<64x256x56x56xf32>
    %v220 = stablehlo.reshape %v219 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v221 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v222 = stablehlo.constant dense<0.0> : tensor<f32>
    %v223 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v224 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v225 = stablehlo.reduce(%v221 init: %v222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v226 = stablehlo.broadcast_in_dim %v225, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v227 = stablehlo.divide %v226, %v223 : tensor<64x256x56x56xf32>
    %v228 = stablehlo.subtract %v221, %v227 : tensor<64x256x56x56xf32>
    %v229 = stablehlo.multiply %v228, %v228 : tensor<64x256x56x56xf32>
    %v230 = stablehlo.reduce(%v229 init: %v222) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v231 = stablehlo.broadcast_in_dim %v230, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v232 = stablehlo.divide %v231, %v223 : tensor<64x256x56x56xf32>
    %v233 = stablehlo.add %v232, %v224 : tensor<64x256x56x56xf32>
    %v234 = stablehlo.rsqrt %v233 : tensor<64x256x56x56xf32>
    %v235 = stablehlo.multiply %v228, %v234 : tensor<64x256x56x56xf32>
    %v236 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v237 = stablehlo.broadcast_in_dim %s1b1bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v238 = stablehlo.multiply %v235, %v236 : tensor<64x256x56x56xf32>
    %v239 = stablehlo.add %v238, %v237 : tensor<64x256x56x56xf32>
    %v240 = stablehlo.reshape %v239 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v241 = stablehlo.add %v240, %v152 : tensor<64x802816xf32>
    %v242 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v243 = stablehlo.maximum %v241, %v242 : tensor<64x802816xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v245 = stablehlo.convert %v244 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v246 = stablehlo.convert %s1b2W1 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v247 = stablehlo.convolution(%v245, %v246)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v248 = stablehlo.convert %v247 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v249 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v250 = stablehlo.add %v248, %v249 : tensor<64x64x56x56xf32>
    %v251 = stablehlo.reshape %v250 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v252 = stablehlo.reshape %v251 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v253 = stablehlo.constant dense<0.0> : tensor<f32>
    %v254 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v255 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v256 = stablehlo.reduce(%v252 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v257 = stablehlo.broadcast_in_dim %v256, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v258 = stablehlo.divide %v257, %v254 : tensor<64x64x56x56xf32>
    %v259 = stablehlo.subtract %v252, %v258 : tensor<64x64x56x56xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<64x64x56x56xf32>
    %v261 = stablehlo.reduce(%v260 init: %v253) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v262 = stablehlo.broadcast_in_dim %v261, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v263 = stablehlo.divide %v262, %v254 : tensor<64x64x56x56xf32>
    %v264 = stablehlo.add %v263, %v255 : tensor<64x64x56x56xf32>
    %v265 = stablehlo.rsqrt %v264 : tensor<64x64x56x56xf32>
    %v266 = stablehlo.multiply %v259, %v265 : tensor<64x64x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v268 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v269 = stablehlo.multiply %v266, %v267 : tensor<64x64x56x56xf32>
    %v270 = stablehlo.add %v269, %v268 : tensor<64x64x56x56xf32>
    %v271 = stablehlo.reshape %v270 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v272 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v273 = stablehlo.maximum %v271, %v272 : tensor<64x200704xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v275 = stablehlo.convert %v274 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v276 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v277 = stablehlo.convolution(%v275, %v276)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v278 = stablehlo.convert %v277 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v279 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v280 = stablehlo.add %v278, %v279 : tensor<64x64x56x56xf32>
    %v281 = stablehlo.reshape %v280 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v283 = stablehlo.constant dense<0.0> : tensor<f32>
    %v284 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v285 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v286 = stablehlo.reduce(%v282 init: %v283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v287 = stablehlo.broadcast_in_dim %v286, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v288 = stablehlo.divide %v287, %v284 : tensor<64x64x56x56xf32>
    %v289 = stablehlo.subtract %v282, %v288 : tensor<64x64x56x56xf32>
    %v290 = stablehlo.multiply %v289, %v289 : tensor<64x64x56x56xf32>
    %v291 = stablehlo.reduce(%v290 init: %v283) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v292 = stablehlo.broadcast_in_dim %v291, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v293 = stablehlo.divide %v292, %v284 : tensor<64x64x56x56xf32>
    %v294 = stablehlo.add %v293, %v285 : tensor<64x64x56x56xf32>
    %v295 = stablehlo.rsqrt %v294 : tensor<64x64x56x56xf32>
    %v296 = stablehlo.multiply %v289, %v295 : tensor<64x64x56x56xf32>
    %v297 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v298 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v299 = stablehlo.multiply %v296, %v297 : tensor<64x64x56x56xf32>
    %v300 = stablehlo.add %v299, %v298 : tensor<64x64x56x56xf32>
    %v301 = stablehlo.reshape %v300 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v302 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v303 = stablehlo.maximum %v301, %v302 : tensor<64x200704xf32>
    %v304 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v305 = stablehlo.convert %v304 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v306 = stablehlo.convert %s1b2W3 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v307 = stablehlo.convolution(%v305, %v306)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v308 = stablehlo.convert %v307 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v309 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v310 = stablehlo.add %v308, %v309 : tensor<64x256x56x56xf32>
    %v311 = stablehlo.reshape %v310 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v312 = stablehlo.reshape %v311 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v313 = stablehlo.constant dense<0.0> : tensor<f32>
    %v314 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v315 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v316 = stablehlo.reduce(%v312 init: %v313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v317 = stablehlo.broadcast_in_dim %v316, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v318 = stablehlo.divide %v317, %v314 : tensor<64x256x56x56xf32>
    %v319 = stablehlo.subtract %v312, %v318 : tensor<64x256x56x56xf32>
    %v320 = stablehlo.multiply %v319, %v319 : tensor<64x256x56x56xf32>
    %v321 = stablehlo.reduce(%v320 init: %v313) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v322 = stablehlo.broadcast_in_dim %v321, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v323 = stablehlo.divide %v322, %v314 : tensor<64x256x56x56xf32>
    %v324 = stablehlo.add %v323, %v315 : tensor<64x256x56x56xf32>
    %v325 = stablehlo.rsqrt %v324 : tensor<64x256x56x56xf32>
    %v326 = stablehlo.multiply %v319, %v325 : tensor<64x256x56x56xf32>
    %v327 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v328 = stablehlo.broadcast_in_dim %s1b2bt3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v329 = stablehlo.multiply %v326, %v327 : tensor<64x256x56x56xf32>
    %v330 = stablehlo.add %v329, %v328 : tensor<64x256x56x56xf32>
    %v331 = stablehlo.reshape %v330 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v332 = stablehlo.add %v331, %v243 : tensor<64x802816xf32>
    %v333 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v334 = stablehlo.maximum %v332, %v333 : tensor<64x802816xf32>
    %v335 = stablehlo.reshape %v334 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v336 = stablehlo.convert %v335 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v337 = stablehlo.convert %s2b0W1 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v338 = stablehlo.convolution(%v336, %v337)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<128x256x1x1xbf16>) -> tensor<64x128x56x56xbf16>
    %v339 = stablehlo.convert %v338 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v340 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v341 = stablehlo.add %v339, %v340 : tensor<64x128x56x56xf32>
    %v342 = stablehlo.reshape %v341 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v345 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v346 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v347 = stablehlo.reduce(%v343 init: %v344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v348 = stablehlo.broadcast_in_dim %v347, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v349 = stablehlo.divide %v348, %v345 : tensor<64x128x56x56xf32>
    %v350 = stablehlo.subtract %v343, %v349 : tensor<64x128x56x56xf32>
    %v351 = stablehlo.multiply %v350, %v350 : tensor<64x128x56x56xf32>
    %v352 = stablehlo.reduce(%v351 init: %v344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v353 = stablehlo.broadcast_in_dim %v352, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v354 = stablehlo.divide %v353, %v345 : tensor<64x128x56x56xf32>
    %v355 = stablehlo.add %v354, %v346 : tensor<64x128x56x56xf32>
    %v356 = stablehlo.rsqrt %v355 : tensor<64x128x56x56xf32>
    %v357 = stablehlo.multiply %v350, %v356 : tensor<64x128x56x56xf32>
    %v358 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v359 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v360 = stablehlo.multiply %v357, %v358 : tensor<64x128x56x56xf32>
    %v361 = stablehlo.add %v360, %v359 : tensor<64x128x56x56xf32>
    %v362 = stablehlo.reshape %v361 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v363 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v364 = stablehlo.maximum %v362, %v363 : tensor<64x401408xf32>
    %v365 = stablehlo.reshape %v364 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v366 = stablehlo.convert %v365 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v367 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v368 = stablehlo.convolution(%v366, %v367)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v369 = stablehlo.convert %v368 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v370 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v371 = stablehlo.add %v369, %v370 : tensor<64x128x28x28xf32>
    %v372 = stablehlo.reshape %v371 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v373 = stablehlo.reshape %v372 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v374 = stablehlo.constant dense<0.0> : tensor<f32>
    %v375 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v376 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v377 = stablehlo.reduce(%v373 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v378 = stablehlo.broadcast_in_dim %v377, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v379 = stablehlo.divide %v378, %v375 : tensor<64x128x28x28xf32>
    %v380 = stablehlo.subtract %v373, %v379 : tensor<64x128x28x28xf32>
    %v381 = stablehlo.multiply %v380, %v380 : tensor<64x128x28x28xf32>
    %v382 = stablehlo.reduce(%v381 init: %v374) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v383 = stablehlo.broadcast_in_dim %v382, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v384 = stablehlo.divide %v383, %v375 : tensor<64x128x28x28xf32>
    %v385 = stablehlo.add %v384, %v376 : tensor<64x128x28x28xf32>
    %v386 = stablehlo.rsqrt %v385 : tensor<64x128x28x28xf32>
    %v387 = stablehlo.multiply %v380, %v386 : tensor<64x128x28x28xf32>
    %v388 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v389 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v390 = stablehlo.multiply %v387, %v388 : tensor<64x128x28x28xf32>
    %v391 = stablehlo.add %v390, %v389 : tensor<64x128x28x28xf32>
    %v392 = stablehlo.reshape %v391 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v393 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v394 = stablehlo.maximum %v392, %v393 : tensor<64x100352xf32>
    %v395 = stablehlo.reshape %v394 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v396 = stablehlo.convert %v395 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v397 = stablehlo.convert %s2b0W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v398 = stablehlo.convolution(%v396, %v397)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v399 = stablehlo.convert %v398 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v400 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v401 = stablehlo.add %v399, %v400 : tensor<64x512x28x28xf32>
    %v402 = stablehlo.reshape %v401 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v403 = stablehlo.reshape %v402 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v404 = stablehlo.constant dense<0.0> : tensor<f32>
    %v405 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v406 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v407 = stablehlo.reduce(%v403 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v408 = stablehlo.broadcast_in_dim %v407, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v409 = stablehlo.divide %v408, %v405 : tensor<64x512x28x28xf32>
    %v410 = stablehlo.subtract %v403, %v409 : tensor<64x512x28x28xf32>
    %v411 = stablehlo.multiply %v410, %v410 : tensor<64x512x28x28xf32>
    %v412 = stablehlo.reduce(%v411 init: %v404) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v413 = stablehlo.broadcast_in_dim %v412, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v414 = stablehlo.divide %v413, %v405 : tensor<64x512x28x28xf32>
    %v415 = stablehlo.add %v414, %v406 : tensor<64x512x28x28xf32>
    %v416 = stablehlo.rsqrt %v415 : tensor<64x512x28x28xf32>
    %v417 = stablehlo.multiply %v410, %v416 : tensor<64x512x28x28xf32>
    %v418 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v419 = stablehlo.broadcast_in_dim %s2b0bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v420 = stablehlo.multiply %v417, %v418 : tensor<64x512x28x28xf32>
    %v421 = stablehlo.add %v420, %v419 : tensor<64x512x28x28xf32>
    %v422 = stablehlo.reshape %v421 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v423 = stablehlo.reshape %v334 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v424 = stablehlo.convert %v423 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v425 = stablehlo.convert %s2b0Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v426 = stablehlo.convolution(%v424, %v425)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v427 = stablehlo.convert %v426 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v428 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v429 = stablehlo.add %v427, %v428 : tensor<64x512x28x28xf32>
    %v430 = stablehlo.reshape %v429 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v431 = stablehlo.reshape %v430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v433 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v434 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v435 = stablehlo.reduce(%v431 init: %v432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v436 = stablehlo.broadcast_in_dim %v435, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v437 = stablehlo.divide %v436, %v433 : tensor<64x512x28x28xf32>
    %v438 = stablehlo.subtract %v431, %v437 : tensor<64x512x28x28xf32>
    %v439 = stablehlo.multiply %v438, %v438 : tensor<64x512x28x28xf32>
    %v440 = stablehlo.reduce(%v439 init: %v432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v442 = stablehlo.divide %v441, %v433 : tensor<64x512x28x28xf32>
    %v443 = stablehlo.add %v442, %v434 : tensor<64x512x28x28xf32>
    %v444 = stablehlo.rsqrt %v443 : tensor<64x512x28x28xf32>
    %v445 = stablehlo.multiply %v438, %v444 : tensor<64x512x28x28xf32>
    %v446 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v447 = stablehlo.broadcast_in_dim %s2b0btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v448 = stablehlo.multiply %v445, %v446 : tensor<64x512x28x28xf32>
    %v449 = stablehlo.add %v448, %v447 : tensor<64x512x28x28xf32>
    %v450 = stablehlo.reshape %v449 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v451 = stablehlo.add %v422, %v450 : tensor<64x401408xf32>
    %v452 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v453 = stablehlo.maximum %v451, %v452 : tensor<64x401408xf32>
    %v454 = stablehlo.reshape %v453 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v455 = stablehlo.convert %v454 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v456 = stablehlo.convert %s2b1W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v457 = stablehlo.convolution(%v455, %v456)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v458 = stablehlo.convert %v457 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v459 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v460 = stablehlo.add %v458, %v459 : tensor<64x128x28x28xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v462 = stablehlo.reshape %v461 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v464 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v465 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v466 = stablehlo.reduce(%v462 init: %v463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v467 = stablehlo.broadcast_in_dim %v466, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v468 = stablehlo.divide %v467, %v464 : tensor<64x128x28x28xf32>
    %v469 = stablehlo.subtract %v462, %v468 : tensor<64x128x28x28xf32>
    %v470 = stablehlo.multiply %v469, %v469 : tensor<64x128x28x28xf32>
    %v471 = stablehlo.reduce(%v470 init: %v463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v472 = stablehlo.broadcast_in_dim %v471, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v473 = stablehlo.divide %v472, %v464 : tensor<64x128x28x28xf32>
    %v474 = stablehlo.add %v473, %v465 : tensor<64x128x28x28xf32>
    %v475 = stablehlo.rsqrt %v474 : tensor<64x128x28x28xf32>
    %v476 = stablehlo.multiply %v469, %v475 : tensor<64x128x28x28xf32>
    %v477 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v478 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v479 = stablehlo.multiply %v476, %v477 : tensor<64x128x28x28xf32>
    %v480 = stablehlo.add %v479, %v478 : tensor<64x128x28x28xf32>
    %v481 = stablehlo.reshape %v480 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v482 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v483 = stablehlo.maximum %v481, %v482 : tensor<64x100352xf32>
    %v484 = stablehlo.reshape %v483 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v485 = stablehlo.convert %v484 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v486 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v487 = stablehlo.convolution(%v485, %v486)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v488 = stablehlo.convert %v487 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v489 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v490 = stablehlo.add %v488, %v489 : tensor<64x128x28x28xf32>
    %v491 = stablehlo.reshape %v490 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v492 = stablehlo.reshape %v491 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v494 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v495 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v496 = stablehlo.reduce(%v492 init: %v493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v497 = stablehlo.broadcast_in_dim %v496, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v498 = stablehlo.divide %v497, %v494 : tensor<64x128x28x28xf32>
    %v499 = stablehlo.subtract %v492, %v498 : tensor<64x128x28x28xf32>
    %v500 = stablehlo.multiply %v499, %v499 : tensor<64x128x28x28xf32>
    %v501 = stablehlo.reduce(%v500 init: %v493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v503 = stablehlo.divide %v502, %v494 : tensor<64x128x28x28xf32>
    %v504 = stablehlo.add %v503, %v495 : tensor<64x128x28x28xf32>
    %v505 = stablehlo.rsqrt %v504 : tensor<64x128x28x28xf32>
    %v506 = stablehlo.multiply %v499, %v505 : tensor<64x128x28x28xf32>
    %v507 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v508 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v509 = stablehlo.multiply %v506, %v507 : tensor<64x128x28x28xf32>
    %v510 = stablehlo.add %v509, %v508 : tensor<64x128x28x28xf32>
    %v511 = stablehlo.reshape %v510 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v512 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v513 = stablehlo.maximum %v511, %v512 : tensor<64x100352xf32>
    %v514 = stablehlo.reshape %v513 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v515 = stablehlo.convert %v514 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v516 = stablehlo.convert %s2b1W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v517 = stablehlo.convolution(%v515, %v516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v518 = stablehlo.convert %v517 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v519 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v520 = stablehlo.add %v518, %v519 : tensor<64x512x28x28xf32>
    %v521 = stablehlo.reshape %v520 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v522 = stablehlo.reshape %v521 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v523 = stablehlo.constant dense<0.0> : tensor<f32>
    %v524 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v525 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v526 = stablehlo.reduce(%v522 init: %v523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v527 = stablehlo.broadcast_in_dim %v526, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v528 = stablehlo.divide %v527, %v524 : tensor<64x512x28x28xf32>
    %v529 = stablehlo.subtract %v522, %v528 : tensor<64x512x28x28xf32>
    %v530 = stablehlo.multiply %v529, %v529 : tensor<64x512x28x28xf32>
    %v531 = stablehlo.reduce(%v530 init: %v523) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v533 = stablehlo.divide %v532, %v524 : tensor<64x512x28x28xf32>
    %v534 = stablehlo.add %v533, %v525 : tensor<64x512x28x28xf32>
    %v535 = stablehlo.rsqrt %v534 : tensor<64x512x28x28xf32>
    %v536 = stablehlo.multiply %v529, %v535 : tensor<64x512x28x28xf32>
    %v537 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v538 = stablehlo.broadcast_in_dim %s2b1bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v539 = stablehlo.multiply %v536, %v537 : tensor<64x512x28x28xf32>
    %v540 = stablehlo.add %v539, %v538 : tensor<64x512x28x28xf32>
    %v541 = stablehlo.reshape %v540 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v542 = stablehlo.add %v541, %v453 : tensor<64x401408xf32>
    %v543 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v544 = stablehlo.maximum %v542, %v543 : tensor<64x401408xf32>
    %v545 = stablehlo.reshape %v544 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v546 = stablehlo.convert %v545 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v547 = stablehlo.convert %s2b2W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v548 = stablehlo.convolution(%v546, %v547)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v549 = stablehlo.convert %v548 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v550 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v551 = stablehlo.add %v549, %v550 : tensor<64x128x28x28xf32>
    %v552 = stablehlo.reshape %v551 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v553 = stablehlo.reshape %v552 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v555 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v556 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v557 = stablehlo.reduce(%v553 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v558 = stablehlo.broadcast_in_dim %v557, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v559 = stablehlo.divide %v558, %v555 : tensor<64x128x28x28xf32>
    %v560 = stablehlo.subtract %v553, %v559 : tensor<64x128x28x28xf32>
    %v561 = stablehlo.multiply %v560, %v560 : tensor<64x128x28x28xf32>
    %v562 = stablehlo.reduce(%v561 init: %v554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v563 = stablehlo.broadcast_in_dim %v562, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v564 = stablehlo.divide %v563, %v555 : tensor<64x128x28x28xf32>
    %v565 = stablehlo.add %v564, %v556 : tensor<64x128x28x28xf32>
    %v566 = stablehlo.rsqrt %v565 : tensor<64x128x28x28xf32>
    %v567 = stablehlo.multiply %v560, %v566 : tensor<64x128x28x28xf32>
    %v568 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v569 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v570 = stablehlo.multiply %v567, %v568 : tensor<64x128x28x28xf32>
    %v571 = stablehlo.add %v570, %v569 : tensor<64x128x28x28xf32>
    %v572 = stablehlo.reshape %v571 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v573 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v574 = stablehlo.maximum %v572, %v573 : tensor<64x100352xf32>
    %v575 = stablehlo.reshape %v574 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v576 = stablehlo.convert %v575 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v577 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v578 = stablehlo.convolution(%v576, %v577)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v579 = stablehlo.convert %v578 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v580 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v581 = stablehlo.add %v579, %v580 : tensor<64x128x28x28xf32>
    %v582 = stablehlo.reshape %v581 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v583 = stablehlo.reshape %v582 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v584 = stablehlo.constant dense<0.0> : tensor<f32>
    %v585 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v586 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v587 = stablehlo.reduce(%v583 init: %v584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v588 = stablehlo.broadcast_in_dim %v587, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v589 = stablehlo.divide %v588, %v585 : tensor<64x128x28x28xf32>
    %v590 = stablehlo.subtract %v583, %v589 : tensor<64x128x28x28xf32>
    %v591 = stablehlo.multiply %v590, %v590 : tensor<64x128x28x28xf32>
    %v592 = stablehlo.reduce(%v591 init: %v584) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v593 = stablehlo.broadcast_in_dim %v592, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v594 = stablehlo.divide %v593, %v585 : tensor<64x128x28x28xf32>
    %v595 = stablehlo.add %v594, %v586 : tensor<64x128x28x28xf32>
    %v596 = stablehlo.rsqrt %v595 : tensor<64x128x28x28xf32>
    %v597 = stablehlo.multiply %v590, %v596 : tensor<64x128x28x28xf32>
    %v598 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v599 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v600 = stablehlo.multiply %v597, %v598 : tensor<64x128x28x28xf32>
    %v601 = stablehlo.add %v600, %v599 : tensor<64x128x28x28xf32>
    %v602 = stablehlo.reshape %v601 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v603 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v604 = stablehlo.maximum %v602, %v603 : tensor<64x100352xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v606 = stablehlo.convert %v605 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v607 = stablehlo.convert %s2b2W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v608 = stablehlo.convolution(%v606, %v607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v609 = stablehlo.convert %v608 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v610 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v611 = stablehlo.add %v609, %v610 : tensor<64x512x28x28xf32>
    %v612 = stablehlo.reshape %v611 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v613 = stablehlo.reshape %v612 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v615 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v616 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v617 = stablehlo.reduce(%v613 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v618 = stablehlo.broadcast_in_dim %v617, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v619 = stablehlo.divide %v618, %v615 : tensor<64x512x28x28xf32>
    %v620 = stablehlo.subtract %v613, %v619 : tensor<64x512x28x28xf32>
    %v621 = stablehlo.multiply %v620, %v620 : tensor<64x512x28x28xf32>
    %v622 = stablehlo.reduce(%v621 init: %v614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v623 = stablehlo.broadcast_in_dim %v622, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v624 = stablehlo.divide %v623, %v615 : tensor<64x512x28x28xf32>
    %v625 = stablehlo.add %v624, %v616 : tensor<64x512x28x28xf32>
    %v626 = stablehlo.rsqrt %v625 : tensor<64x512x28x28xf32>
    %v627 = stablehlo.multiply %v620, %v626 : tensor<64x512x28x28xf32>
    %v628 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v629 = stablehlo.broadcast_in_dim %s2b2bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v630 = stablehlo.multiply %v627, %v628 : tensor<64x512x28x28xf32>
    %v631 = stablehlo.add %v630, %v629 : tensor<64x512x28x28xf32>
    %v632 = stablehlo.reshape %v631 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v633 = stablehlo.add %v632, %v544 : tensor<64x401408xf32>
    %v634 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v635 = stablehlo.maximum %v633, %v634 : tensor<64x401408xf32>
    %v636 = stablehlo.reshape %v635 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v637 = stablehlo.convert %v636 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v638 = stablehlo.convert %s2b3W1 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v639 = stablehlo.convolution(%v637, %v638)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v640 = stablehlo.convert %v639 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v641 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v642 = stablehlo.add %v640, %v641 : tensor<64x128x28x28xf32>
    %v643 = stablehlo.reshape %v642 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v646 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v647 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v648 = stablehlo.reduce(%v644 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v649 = stablehlo.broadcast_in_dim %v648, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v650 = stablehlo.divide %v649, %v646 : tensor<64x128x28x28xf32>
    %v651 = stablehlo.subtract %v644, %v650 : tensor<64x128x28x28xf32>
    %v652 = stablehlo.multiply %v651, %v651 : tensor<64x128x28x28xf32>
    %v653 = stablehlo.reduce(%v652 init: %v645) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v654 = stablehlo.broadcast_in_dim %v653, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v655 = stablehlo.divide %v654, %v646 : tensor<64x128x28x28xf32>
    %v656 = stablehlo.add %v655, %v647 : tensor<64x128x28x28xf32>
    %v657 = stablehlo.rsqrt %v656 : tensor<64x128x28x28xf32>
    %v658 = stablehlo.multiply %v651, %v657 : tensor<64x128x28x28xf32>
    %v659 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v660 = stablehlo.broadcast_in_dim %s2b3bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v661 = stablehlo.multiply %v658, %v659 : tensor<64x128x28x28xf32>
    %v662 = stablehlo.add %v661, %v660 : tensor<64x128x28x28xf32>
    %v663 = stablehlo.reshape %v662 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v664 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v665 = stablehlo.maximum %v663, %v664 : tensor<64x100352xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v667 = stablehlo.convert %v666 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v668 = stablehlo.convert %s2b3W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v669 = stablehlo.convolution(%v667, %v668)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v670 = stablehlo.convert %v669 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v671 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v672 = stablehlo.add %v670, %v671 : tensor<64x128x28x28xf32>
    %v673 = stablehlo.reshape %v672 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v674 = stablehlo.reshape %v673 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v676 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v677 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v678 = stablehlo.reduce(%v674 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v679 = stablehlo.broadcast_in_dim %v678, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v680 = stablehlo.divide %v679, %v676 : tensor<64x128x28x28xf32>
    %v681 = stablehlo.subtract %v674, %v680 : tensor<64x128x28x28xf32>
    %v682 = stablehlo.multiply %v681, %v681 : tensor<64x128x28x28xf32>
    %v683 = stablehlo.reduce(%v682 init: %v675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v684 = stablehlo.broadcast_in_dim %v683, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v685 = stablehlo.divide %v684, %v676 : tensor<64x128x28x28xf32>
    %v686 = stablehlo.add %v685, %v677 : tensor<64x128x28x28xf32>
    %v687 = stablehlo.rsqrt %v686 : tensor<64x128x28x28xf32>
    %v688 = stablehlo.multiply %v681, %v687 : tensor<64x128x28x28xf32>
    %v689 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v690 = stablehlo.broadcast_in_dim %s2b3bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v691 = stablehlo.multiply %v688, %v689 : tensor<64x128x28x28xf32>
    %v692 = stablehlo.add %v691, %v690 : tensor<64x128x28x28xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v694 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v695 = stablehlo.maximum %v693, %v694 : tensor<64x100352xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v697 = stablehlo.convert %v696 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v698 = stablehlo.convert %s2b3W3 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v699 = stablehlo.convolution(%v697, %v698)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v700 = stablehlo.convert %v699 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v701 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v702 = stablehlo.add %v700, %v701 : tensor<64x512x28x28xf32>
    %v703 = stablehlo.reshape %v702 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v704 = stablehlo.reshape %v703 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v705 = stablehlo.constant dense<0.0> : tensor<f32>
    %v706 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v707 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v708 = stablehlo.reduce(%v704 init: %v705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v709 = stablehlo.broadcast_in_dim %v708, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v710 = stablehlo.divide %v709, %v706 : tensor<64x512x28x28xf32>
    %v711 = stablehlo.subtract %v704, %v710 : tensor<64x512x28x28xf32>
    %v712 = stablehlo.multiply %v711, %v711 : tensor<64x512x28x28xf32>
    %v713 = stablehlo.reduce(%v712 init: %v705) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v714 = stablehlo.broadcast_in_dim %v713, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v715 = stablehlo.divide %v714, %v706 : tensor<64x512x28x28xf32>
    %v716 = stablehlo.add %v715, %v707 : tensor<64x512x28x28xf32>
    %v717 = stablehlo.rsqrt %v716 : tensor<64x512x28x28xf32>
    %v718 = stablehlo.multiply %v711, %v717 : tensor<64x512x28x28xf32>
    %v719 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v720 = stablehlo.broadcast_in_dim %s2b3bt3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v721 = stablehlo.multiply %v718, %v719 : tensor<64x512x28x28xf32>
    %v722 = stablehlo.add %v721, %v720 : tensor<64x512x28x28xf32>
    %v723 = stablehlo.reshape %v722 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v724 = stablehlo.add %v723, %v635 : tensor<64x401408xf32>
    %v725 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v726 = stablehlo.maximum %v724, %v725 : tensor<64x401408xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v728 = stablehlo.convert %v727 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v729 = stablehlo.convert %s3b0W1 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v730 = stablehlo.convolution(%v728, %v729)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x28x28xbf16>
    %v731 = stablehlo.convert %v730 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v732 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v733 = stablehlo.add %v731, %v732 : tensor<64x256x28x28xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v736 = stablehlo.constant dense<0.0> : tensor<f32>
    %v737 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v738 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v739 = stablehlo.reduce(%v735 init: %v736) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v740 = stablehlo.broadcast_in_dim %v739, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v741 = stablehlo.divide %v740, %v737 : tensor<64x256x28x28xf32>
    %v742 = stablehlo.subtract %v735, %v741 : tensor<64x256x28x28xf32>
    %v743 = stablehlo.multiply %v742, %v742 : tensor<64x256x28x28xf32>
    %v744 = stablehlo.reduce(%v743 init: %v736) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v745 = stablehlo.broadcast_in_dim %v744, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v746 = stablehlo.divide %v745, %v737 : tensor<64x256x28x28xf32>
    %v747 = stablehlo.add %v746, %v738 : tensor<64x256x28x28xf32>
    %v748 = stablehlo.rsqrt %v747 : tensor<64x256x28x28xf32>
    %v749 = stablehlo.multiply %v742, %v748 : tensor<64x256x28x28xf32>
    %v750 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v751 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v752 = stablehlo.multiply %v749, %v750 : tensor<64x256x28x28xf32>
    %v753 = stablehlo.add %v752, %v751 : tensor<64x256x28x28xf32>
    %v754 = stablehlo.reshape %v753 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v755 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v756 = stablehlo.maximum %v754, %v755 : tensor<64x200704xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v758 = stablehlo.convert %v757 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v759 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v760 = stablehlo.convolution(%v758, %v759)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v761 = stablehlo.convert %v760 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v762 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v763 = stablehlo.add %v761, %v762 : tensor<64x256x14x14xf32>
    %v764 = stablehlo.reshape %v763 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v765 = stablehlo.reshape %v764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v766 = stablehlo.constant dense<0.0> : tensor<f32>
    %v767 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v768 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v769 = stablehlo.reduce(%v765 init: %v766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v770 = stablehlo.broadcast_in_dim %v769, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v771 = stablehlo.divide %v770, %v767 : tensor<64x256x14x14xf32>
    %v772 = stablehlo.subtract %v765, %v771 : tensor<64x256x14x14xf32>
    %v773 = stablehlo.multiply %v772, %v772 : tensor<64x256x14x14xf32>
    %v774 = stablehlo.reduce(%v773 init: %v766) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v775 = stablehlo.broadcast_in_dim %v774, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v776 = stablehlo.divide %v775, %v767 : tensor<64x256x14x14xf32>
    %v777 = stablehlo.add %v776, %v768 : tensor<64x256x14x14xf32>
    %v778 = stablehlo.rsqrt %v777 : tensor<64x256x14x14xf32>
    %v779 = stablehlo.multiply %v772, %v778 : tensor<64x256x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v781 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v782 = stablehlo.multiply %v779, %v780 : tensor<64x256x14x14xf32>
    %v783 = stablehlo.add %v782, %v781 : tensor<64x256x14x14xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v785 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v786 = stablehlo.maximum %v784, %v785 : tensor<64x50176xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v788 = stablehlo.convert %v787 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v789 = stablehlo.convert %s3b0W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v790 = stablehlo.convolution(%v788, %v789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v791 = stablehlo.convert %v790 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<64x1024x14x14xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v798 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v799 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v800 = stablehlo.broadcast_in_dim %v799, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v801 = stablehlo.divide %v800, %v797 : tensor<64x1024x14x14xf32>
    %v802 = stablehlo.subtract %v795, %v801 : tensor<64x1024x14x14xf32>
    %v803 = stablehlo.multiply %v802, %v802 : tensor<64x1024x14x14xf32>
    %v804 = stablehlo.reduce(%v803 init: %v796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v806 = stablehlo.divide %v805, %v797 : tensor<64x1024x14x14xf32>
    %v807 = stablehlo.add %v806, %v798 : tensor<64x1024x14x14xf32>
    %v808 = stablehlo.rsqrt %v807 : tensor<64x1024x14x14xf32>
    %v809 = stablehlo.multiply %v802, %v808 : tensor<64x1024x14x14xf32>
    %v810 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v811 = stablehlo.broadcast_in_dim %s3b0bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v812 = stablehlo.multiply %v809, %v810 : tensor<64x1024x14x14xf32>
    %v813 = stablehlo.add %v812, %v811 : tensor<64x1024x14x14xf32>
    %v814 = stablehlo.reshape %v813 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v815 = stablehlo.reshape %v726 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v816 = stablehlo.convert %v815 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v817 = stablehlo.convert %s3b0Wp : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v818 = stablehlo.convolution(%v816, %v817)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v819 = stablehlo.convert %v818 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v820 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v821 = stablehlo.add %v819, %v820 : tensor<64x1024x14x14xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v823 = stablehlo.reshape %v822 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v824 = stablehlo.constant dense<0.0> : tensor<f32>
    %v825 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v826 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v827 = stablehlo.reduce(%v823 init: %v824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v828 = stablehlo.broadcast_in_dim %v827, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v829 = stablehlo.divide %v828, %v825 : tensor<64x1024x14x14xf32>
    %v830 = stablehlo.subtract %v823, %v829 : tensor<64x1024x14x14xf32>
    %v831 = stablehlo.multiply %v830, %v830 : tensor<64x1024x14x14xf32>
    %v832 = stablehlo.reduce(%v831 init: %v824) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v833 = stablehlo.broadcast_in_dim %v832, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v834 = stablehlo.divide %v833, %v825 : tensor<64x1024x14x14xf32>
    %v835 = stablehlo.add %v834, %v826 : tensor<64x1024x14x14xf32>
    %v836 = stablehlo.rsqrt %v835 : tensor<64x1024x14x14xf32>
    %v837 = stablehlo.multiply %v830, %v836 : tensor<64x1024x14x14xf32>
    %v838 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v839 = stablehlo.broadcast_in_dim %s3b0btp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v840 = stablehlo.multiply %v837, %v838 : tensor<64x1024x14x14xf32>
    %v841 = stablehlo.add %v840, %v839 : tensor<64x1024x14x14xf32>
    %v842 = stablehlo.reshape %v841 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v843 = stablehlo.add %v814, %v842 : tensor<64x200704xf32>
    %v844 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v845 = stablehlo.maximum %v843, %v844 : tensor<64x200704xf32>
    %v846 = stablehlo.reshape %v845 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v847 = stablehlo.convert %v846 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v848 = stablehlo.convert %s3b1W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v849 = stablehlo.convolution(%v847, %v848)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v850 = stablehlo.convert %v849 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v851 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v852 = stablehlo.add %v850, %v851 : tensor<64x256x14x14xf32>
    %v853 = stablehlo.reshape %v852 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v854 = stablehlo.reshape %v853 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v855 = stablehlo.constant dense<0.0> : tensor<f32>
    %v856 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v857 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v858 = stablehlo.reduce(%v854 init: %v855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v859 = stablehlo.broadcast_in_dim %v858, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v860 = stablehlo.divide %v859, %v856 : tensor<64x256x14x14xf32>
    %v861 = stablehlo.subtract %v854, %v860 : tensor<64x256x14x14xf32>
    %v862 = stablehlo.multiply %v861, %v861 : tensor<64x256x14x14xf32>
    %v863 = stablehlo.reduce(%v862 init: %v855) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v864 = stablehlo.broadcast_in_dim %v863, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v865 = stablehlo.divide %v864, %v856 : tensor<64x256x14x14xf32>
    %v866 = stablehlo.add %v865, %v857 : tensor<64x256x14x14xf32>
    %v867 = stablehlo.rsqrt %v866 : tensor<64x256x14x14xf32>
    %v868 = stablehlo.multiply %v861, %v867 : tensor<64x256x14x14xf32>
    %v869 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v870 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v871 = stablehlo.multiply %v868, %v869 : tensor<64x256x14x14xf32>
    %v872 = stablehlo.add %v871, %v870 : tensor<64x256x14x14xf32>
    %v873 = stablehlo.reshape %v872 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v874 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v875 = stablehlo.maximum %v873, %v874 : tensor<64x50176xf32>
    %v876 = stablehlo.reshape %v875 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v877 = stablehlo.convert %v876 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v878 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v879 = stablehlo.convolution(%v877, %v878)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v880 = stablehlo.convert %v879 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v881 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v882 = stablehlo.add %v880, %v881 : tensor<64x256x14x14xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v884 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v885 = stablehlo.constant dense<0.0> : tensor<f32>
    %v886 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v887 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v888 = stablehlo.reduce(%v884 init: %v885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v889 = stablehlo.broadcast_in_dim %v888, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v890 = stablehlo.divide %v889, %v886 : tensor<64x256x14x14xf32>
    %v891 = stablehlo.subtract %v884, %v890 : tensor<64x256x14x14xf32>
    %v892 = stablehlo.multiply %v891, %v891 : tensor<64x256x14x14xf32>
    %v893 = stablehlo.reduce(%v892 init: %v885) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v894 = stablehlo.broadcast_in_dim %v893, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v895 = stablehlo.divide %v894, %v886 : tensor<64x256x14x14xf32>
    %v896 = stablehlo.add %v895, %v887 : tensor<64x256x14x14xf32>
    %v897 = stablehlo.rsqrt %v896 : tensor<64x256x14x14xf32>
    %v898 = stablehlo.multiply %v891, %v897 : tensor<64x256x14x14xf32>
    %v899 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v900 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v901 = stablehlo.multiply %v898, %v899 : tensor<64x256x14x14xf32>
    %v902 = stablehlo.add %v901, %v900 : tensor<64x256x14x14xf32>
    %v903 = stablehlo.reshape %v902 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v904 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v905 = stablehlo.maximum %v903, %v904 : tensor<64x50176xf32>
    %v906 = stablehlo.reshape %v905 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v907 = stablehlo.convert %v906 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v908 = stablehlo.convert %s3b1W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v909 = stablehlo.convolution(%v907, %v908)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v910 = stablehlo.convert %v909 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v912 = stablehlo.add %v910, %v911 : tensor<64x1024x14x14xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v914 = stablehlo.reshape %v913 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v916 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v917 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v918 = stablehlo.reduce(%v914 init: %v915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v919 = stablehlo.broadcast_in_dim %v918, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v920 = stablehlo.divide %v919, %v916 : tensor<64x1024x14x14xf32>
    %v921 = stablehlo.subtract %v914, %v920 : tensor<64x1024x14x14xf32>
    %v922 = stablehlo.multiply %v921, %v921 : tensor<64x1024x14x14xf32>
    %v923 = stablehlo.reduce(%v922 init: %v915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v924 = stablehlo.broadcast_in_dim %v923, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v925 = stablehlo.divide %v924, %v916 : tensor<64x1024x14x14xf32>
    %v926 = stablehlo.add %v925, %v917 : tensor<64x1024x14x14xf32>
    %v927 = stablehlo.rsqrt %v926 : tensor<64x1024x14x14xf32>
    %v928 = stablehlo.multiply %v921, %v927 : tensor<64x1024x14x14xf32>
    %v929 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v930 = stablehlo.broadcast_in_dim %s3b1bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v931 = stablehlo.multiply %v928, %v929 : tensor<64x1024x14x14xf32>
    %v932 = stablehlo.add %v931, %v930 : tensor<64x1024x14x14xf32>
    %v933 = stablehlo.reshape %v932 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v934 = stablehlo.add %v933, %v845 : tensor<64x200704xf32>
    %v935 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v936 = stablehlo.maximum %v934, %v935 : tensor<64x200704xf32>
    %v937 = stablehlo.reshape %v936 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v938 = stablehlo.convert %v937 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v939 = stablehlo.convert %s3b2W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v940 = stablehlo.convolution(%v938, %v939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v941 = stablehlo.convert %v940 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v942 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v943 = stablehlo.add %v941, %v942 : tensor<64x256x14x14xf32>
    %v944 = stablehlo.reshape %v943 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v945 = stablehlo.reshape %v944 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v946 = stablehlo.constant dense<0.0> : tensor<f32>
    %v947 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v948 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v949 = stablehlo.reduce(%v945 init: %v946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v950 = stablehlo.broadcast_in_dim %v949, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v951 = stablehlo.divide %v950, %v947 : tensor<64x256x14x14xf32>
    %v952 = stablehlo.subtract %v945, %v951 : tensor<64x256x14x14xf32>
    %v953 = stablehlo.multiply %v952, %v952 : tensor<64x256x14x14xf32>
    %v954 = stablehlo.reduce(%v953 init: %v946) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v955 = stablehlo.broadcast_in_dim %v954, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v956 = stablehlo.divide %v955, %v947 : tensor<64x256x14x14xf32>
    %v957 = stablehlo.add %v956, %v948 : tensor<64x256x14x14xf32>
    %v958 = stablehlo.rsqrt %v957 : tensor<64x256x14x14xf32>
    %v959 = stablehlo.multiply %v952, %v958 : tensor<64x256x14x14xf32>
    %v960 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v961 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v962 = stablehlo.multiply %v959, %v960 : tensor<64x256x14x14xf32>
    %v963 = stablehlo.add %v962, %v961 : tensor<64x256x14x14xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v965 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v966 = stablehlo.maximum %v964, %v965 : tensor<64x50176xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v968 = stablehlo.convert %v967 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v969 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v970 = stablehlo.convolution(%v968, %v969)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v971 = stablehlo.convert %v970 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v972 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v973 = stablehlo.add %v971, %v972 : tensor<64x256x14x14xf32>
    %v974 = stablehlo.reshape %v973 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v976 = stablehlo.constant dense<0.0> : tensor<f32>
    %v977 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v978 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v979 = stablehlo.reduce(%v975 init: %v976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v980 = stablehlo.broadcast_in_dim %v979, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v981 = stablehlo.divide %v980, %v977 : tensor<64x256x14x14xf32>
    %v982 = stablehlo.subtract %v975, %v981 : tensor<64x256x14x14xf32>
    %v983 = stablehlo.multiply %v982, %v982 : tensor<64x256x14x14xf32>
    %v984 = stablehlo.reduce(%v983 init: %v976) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v985 = stablehlo.broadcast_in_dim %v984, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v986 = stablehlo.divide %v985, %v977 : tensor<64x256x14x14xf32>
    %v987 = stablehlo.add %v986, %v978 : tensor<64x256x14x14xf32>
    %v988 = stablehlo.rsqrt %v987 : tensor<64x256x14x14xf32>
    %v989 = stablehlo.multiply %v982, %v988 : tensor<64x256x14x14xf32>
    %v990 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v991 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v992 = stablehlo.multiply %v989, %v990 : tensor<64x256x14x14xf32>
    %v993 = stablehlo.add %v992, %v991 : tensor<64x256x14x14xf32>
    %v994 = stablehlo.reshape %v993 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v995 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v996 = stablehlo.maximum %v994, %v995 : tensor<64x50176xf32>
    %v997 = stablehlo.reshape %v996 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v998 = stablehlo.convert %v997 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v999 = stablehlo.convert %s3b2W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1000 = stablehlo.convolution(%v998, %v999)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1001 = stablehlo.convert %v1000 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1002 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1003 = stablehlo.add %v1001, %v1002 : tensor<64x1024x14x14xf32>
    %v1004 = stablehlo.reshape %v1003 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1005 = stablehlo.reshape %v1004 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1006 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1007 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1008 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1009 = stablehlo.reduce(%v1005 init: %v1006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1010 = stablehlo.broadcast_in_dim %v1009, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1011 = stablehlo.divide %v1010, %v1007 : tensor<64x1024x14x14xf32>
    %v1012 = stablehlo.subtract %v1005, %v1011 : tensor<64x1024x14x14xf32>
    %v1013 = stablehlo.multiply %v1012, %v1012 : tensor<64x1024x14x14xf32>
    %v1014 = stablehlo.reduce(%v1013 init: %v1006) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1016 = stablehlo.divide %v1015, %v1007 : tensor<64x1024x14x14xf32>
    %v1017 = stablehlo.add %v1016, %v1008 : tensor<64x1024x14x14xf32>
    %v1018 = stablehlo.rsqrt %v1017 : tensor<64x1024x14x14xf32>
    %v1019 = stablehlo.multiply %v1012, %v1018 : tensor<64x1024x14x14xf32>
    %v1020 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1021 = stablehlo.broadcast_in_dim %s3b2bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1022 = stablehlo.multiply %v1019, %v1020 : tensor<64x1024x14x14xf32>
    %v1023 = stablehlo.add %v1022, %v1021 : tensor<64x1024x14x14xf32>
    %v1024 = stablehlo.reshape %v1023 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1025 = stablehlo.add %v1024, %v936 : tensor<64x200704xf32>
    %v1026 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1027 = stablehlo.maximum %v1025, %v1026 : tensor<64x200704xf32>
    %v1028 = stablehlo.reshape %v1027 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1029 = stablehlo.convert %v1028 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1030 = stablehlo.convert %s3b3W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1031 = stablehlo.convolution(%v1029, %v1030)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1032 = stablehlo.convert %v1031 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1033 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1034 = stablehlo.add %v1032, %v1033 : tensor<64x256x14x14xf32>
    %v1035 = stablehlo.reshape %v1034 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1036 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1037 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1038 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1039 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1040 = stablehlo.reduce(%v1036 init: %v1037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1041 = stablehlo.broadcast_in_dim %v1040, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1042 = stablehlo.divide %v1041, %v1038 : tensor<64x256x14x14xf32>
    %v1043 = stablehlo.subtract %v1036, %v1042 : tensor<64x256x14x14xf32>
    %v1044 = stablehlo.multiply %v1043, %v1043 : tensor<64x256x14x14xf32>
    %v1045 = stablehlo.reduce(%v1044 init: %v1037) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1046 = stablehlo.broadcast_in_dim %v1045, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1047 = stablehlo.divide %v1046, %v1038 : tensor<64x256x14x14xf32>
    %v1048 = stablehlo.add %v1047, %v1039 : tensor<64x256x14x14xf32>
    %v1049 = stablehlo.rsqrt %v1048 : tensor<64x256x14x14xf32>
    %v1050 = stablehlo.multiply %v1043, %v1049 : tensor<64x256x14x14xf32>
    %v1051 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1052 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1053 = stablehlo.multiply %v1050, %v1051 : tensor<64x256x14x14xf32>
    %v1054 = stablehlo.add %v1053, %v1052 : tensor<64x256x14x14xf32>
    %v1055 = stablehlo.reshape %v1054 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1056 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1057 = stablehlo.maximum %v1055, %v1056 : tensor<64x50176xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1059 = stablehlo.convert %v1058 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1060 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1061 = stablehlo.convolution(%v1059, %v1060)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1062 = stablehlo.convert %v1061 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1063 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1064 = stablehlo.add %v1062, %v1063 : tensor<64x256x14x14xf32>
    %v1065 = stablehlo.reshape %v1064 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1066 = stablehlo.reshape %v1065 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1068 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1069 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1070 = stablehlo.reduce(%v1066 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1071 = stablehlo.broadcast_in_dim %v1070, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1072 = stablehlo.divide %v1071, %v1068 : tensor<64x256x14x14xf32>
    %v1073 = stablehlo.subtract %v1066, %v1072 : tensor<64x256x14x14xf32>
    %v1074 = stablehlo.multiply %v1073, %v1073 : tensor<64x256x14x14xf32>
    %v1075 = stablehlo.reduce(%v1074 init: %v1067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1077 = stablehlo.divide %v1076, %v1068 : tensor<64x256x14x14xf32>
    %v1078 = stablehlo.add %v1077, %v1069 : tensor<64x256x14x14xf32>
    %v1079 = stablehlo.rsqrt %v1078 : tensor<64x256x14x14xf32>
    %v1080 = stablehlo.multiply %v1073, %v1079 : tensor<64x256x14x14xf32>
    %v1081 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1082 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1083 = stablehlo.multiply %v1080, %v1081 : tensor<64x256x14x14xf32>
    %v1084 = stablehlo.add %v1083, %v1082 : tensor<64x256x14x14xf32>
    %v1085 = stablehlo.reshape %v1084 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1086 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1087 = stablehlo.maximum %v1085, %v1086 : tensor<64x50176xf32>
    %v1088 = stablehlo.reshape %v1087 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1089 = stablehlo.convert %v1088 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1090 = stablehlo.convert %s3b3W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1091 = stablehlo.convolution(%v1089, %v1090)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1092 = stablehlo.convert %v1091 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1093 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1094 = stablehlo.add %v1092, %v1093 : tensor<64x1024x14x14xf32>
    %v1095 = stablehlo.reshape %v1094 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1096 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1097 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1098 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1099 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1100 = stablehlo.reduce(%v1096 init: %v1097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1101 = stablehlo.broadcast_in_dim %v1100, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1102 = stablehlo.divide %v1101, %v1098 : tensor<64x1024x14x14xf32>
    %v1103 = stablehlo.subtract %v1096, %v1102 : tensor<64x1024x14x14xf32>
    %v1104 = stablehlo.multiply %v1103, %v1103 : tensor<64x1024x14x14xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1097) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1107 = stablehlo.divide %v1106, %v1098 : tensor<64x1024x14x14xf32>
    %v1108 = stablehlo.add %v1107, %v1099 : tensor<64x1024x14x14xf32>
    %v1109 = stablehlo.rsqrt %v1108 : tensor<64x1024x14x14xf32>
    %v1110 = stablehlo.multiply %v1103, %v1109 : tensor<64x1024x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1112 = stablehlo.broadcast_in_dim %s3b3bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1113 = stablehlo.multiply %v1110, %v1111 : tensor<64x1024x14x14xf32>
    %v1114 = stablehlo.add %v1113, %v1112 : tensor<64x1024x14x14xf32>
    %v1115 = stablehlo.reshape %v1114 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1116 = stablehlo.add %v1115, %v1027 : tensor<64x200704xf32>
    %v1117 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1118 = stablehlo.maximum %v1116, %v1117 : tensor<64x200704xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1120 = stablehlo.convert %v1119 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1121 = stablehlo.convert %s3b4W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1122 = stablehlo.convolution(%v1120, %v1121)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1123 = stablehlo.convert %v1122 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1124 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1125 = stablehlo.add %v1123, %v1124 : tensor<64x256x14x14xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1127 = stablehlo.reshape %v1126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1128 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1129 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1130 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1131 = stablehlo.reduce(%v1127 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1132 = stablehlo.broadcast_in_dim %v1131, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1133 = stablehlo.divide %v1132, %v1129 : tensor<64x256x14x14xf32>
    %v1134 = stablehlo.subtract %v1127, %v1133 : tensor<64x256x14x14xf32>
    %v1135 = stablehlo.multiply %v1134, %v1134 : tensor<64x256x14x14xf32>
    %v1136 = stablehlo.reduce(%v1135 init: %v1128) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1136, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1138 = stablehlo.divide %v1137, %v1129 : tensor<64x256x14x14xf32>
    %v1139 = stablehlo.add %v1138, %v1130 : tensor<64x256x14x14xf32>
    %v1140 = stablehlo.rsqrt %v1139 : tensor<64x256x14x14xf32>
    %v1141 = stablehlo.multiply %v1134, %v1140 : tensor<64x256x14x14xf32>
    %v1142 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1143 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1144 = stablehlo.multiply %v1141, %v1142 : tensor<64x256x14x14xf32>
    %v1145 = stablehlo.add %v1144, %v1143 : tensor<64x256x14x14xf32>
    %v1146 = stablehlo.reshape %v1145 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1147 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1148 = stablehlo.maximum %v1146, %v1147 : tensor<64x50176xf32>
    %v1149 = stablehlo.reshape %v1148 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1150 = stablehlo.convert %v1149 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1151 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1152 = stablehlo.convolution(%v1150, %v1151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1153 = stablehlo.convert %v1152 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1154 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1155 = stablehlo.add %v1153, %v1154 : tensor<64x256x14x14xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1157 = stablehlo.reshape %v1156 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1158 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1159 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1160 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1161 = stablehlo.reduce(%v1157 init: %v1158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1162 = stablehlo.broadcast_in_dim %v1161, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1163 = stablehlo.divide %v1162, %v1159 : tensor<64x256x14x14xf32>
    %v1164 = stablehlo.subtract %v1157, %v1163 : tensor<64x256x14x14xf32>
    %v1165 = stablehlo.multiply %v1164, %v1164 : tensor<64x256x14x14xf32>
    %v1166 = stablehlo.reduce(%v1165 init: %v1158) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1167 = stablehlo.broadcast_in_dim %v1166, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1168 = stablehlo.divide %v1167, %v1159 : tensor<64x256x14x14xf32>
    %v1169 = stablehlo.add %v1168, %v1160 : tensor<64x256x14x14xf32>
    %v1170 = stablehlo.rsqrt %v1169 : tensor<64x256x14x14xf32>
    %v1171 = stablehlo.multiply %v1164, %v1170 : tensor<64x256x14x14xf32>
    %v1172 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1173 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1174 = stablehlo.multiply %v1171, %v1172 : tensor<64x256x14x14xf32>
    %v1175 = stablehlo.add %v1174, %v1173 : tensor<64x256x14x14xf32>
    %v1176 = stablehlo.reshape %v1175 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1177 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1178 = stablehlo.maximum %v1176, %v1177 : tensor<64x50176xf32>
    %v1179 = stablehlo.reshape %v1178 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1180 = stablehlo.convert %v1179 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1181 = stablehlo.convert %s3b4W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1182 = stablehlo.convolution(%v1180, %v1181)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1183 = stablehlo.convert %v1182 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1184 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1185 = stablehlo.add %v1183, %v1184 : tensor<64x1024x14x14xf32>
    %v1186 = stablehlo.reshape %v1185 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1187 = stablehlo.reshape %v1186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1189 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1190 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1191 = stablehlo.reduce(%v1187 init: %v1188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1192 = stablehlo.broadcast_in_dim %v1191, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1193 = stablehlo.divide %v1192, %v1189 : tensor<64x1024x14x14xf32>
    %v1194 = stablehlo.subtract %v1187, %v1193 : tensor<64x1024x14x14xf32>
    %v1195 = stablehlo.multiply %v1194, %v1194 : tensor<64x1024x14x14xf32>
    %v1196 = stablehlo.reduce(%v1195 init: %v1188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1197 = stablehlo.broadcast_in_dim %v1196, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1198 = stablehlo.divide %v1197, %v1189 : tensor<64x1024x14x14xf32>
    %v1199 = stablehlo.add %v1198, %v1190 : tensor<64x1024x14x14xf32>
    %v1200 = stablehlo.rsqrt %v1199 : tensor<64x1024x14x14xf32>
    %v1201 = stablehlo.multiply %v1194, %v1200 : tensor<64x1024x14x14xf32>
    %v1202 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1203 = stablehlo.broadcast_in_dim %s3b4bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1204 = stablehlo.multiply %v1201, %v1202 : tensor<64x1024x14x14xf32>
    %v1205 = stablehlo.add %v1204, %v1203 : tensor<64x1024x14x14xf32>
    %v1206 = stablehlo.reshape %v1205 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1207 = stablehlo.add %v1206, %v1118 : tensor<64x200704xf32>
    %v1208 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1209 = stablehlo.maximum %v1207, %v1208 : tensor<64x200704xf32>
    %v1210 = stablehlo.reshape %v1209 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1211 = stablehlo.convert %v1210 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1212 = stablehlo.convert %s3b5W1 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v1213 = stablehlo.convolution(%v1211, %v1212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1214 = stablehlo.convert %v1213 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1215 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1216 = stablehlo.add %v1214, %v1215 : tensor<64x256x14x14xf32>
    %v1217 = stablehlo.reshape %v1216 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1218 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1220 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1221 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1222 = stablehlo.reduce(%v1218 init: %v1219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1224 = stablehlo.divide %v1223, %v1220 : tensor<64x256x14x14xf32>
    %v1225 = stablehlo.subtract %v1218, %v1224 : tensor<64x256x14x14xf32>
    %v1226 = stablehlo.multiply %v1225, %v1225 : tensor<64x256x14x14xf32>
    %v1227 = stablehlo.reduce(%v1226 init: %v1219) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1228 = stablehlo.broadcast_in_dim %v1227, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1229 = stablehlo.divide %v1228, %v1220 : tensor<64x256x14x14xf32>
    %v1230 = stablehlo.add %v1229, %v1221 : tensor<64x256x14x14xf32>
    %v1231 = stablehlo.rsqrt %v1230 : tensor<64x256x14x14xf32>
    %v1232 = stablehlo.multiply %v1225, %v1231 : tensor<64x256x14x14xf32>
    %v1233 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1234 = stablehlo.broadcast_in_dim %s3b5bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1235 = stablehlo.multiply %v1232, %v1233 : tensor<64x256x14x14xf32>
    %v1236 = stablehlo.add %v1235, %v1234 : tensor<64x256x14x14xf32>
    %v1237 = stablehlo.reshape %v1236 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1238 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1239 = stablehlo.maximum %v1237, %v1238 : tensor<64x50176xf32>
    %v1240 = stablehlo.reshape %v1239 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1241 = stablehlo.convert %v1240 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1242 = stablehlo.convert %s3b5W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1243 = stablehlo.convolution(%v1241, %v1242)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1244 = stablehlo.convert %v1243 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1245 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1246 = stablehlo.add %v1244, %v1245 : tensor<64x256x14x14xf32>
    %v1247 = stablehlo.reshape %v1246 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1248 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1250 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1251 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1252 = stablehlo.reduce(%v1248 init: %v1249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1254 = stablehlo.divide %v1253, %v1250 : tensor<64x256x14x14xf32>
    %v1255 = stablehlo.subtract %v1248, %v1254 : tensor<64x256x14x14xf32>
    %v1256 = stablehlo.multiply %v1255, %v1255 : tensor<64x256x14x14xf32>
    %v1257 = stablehlo.reduce(%v1256 init: %v1249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1258 = stablehlo.broadcast_in_dim %v1257, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1259 = stablehlo.divide %v1258, %v1250 : tensor<64x256x14x14xf32>
    %v1260 = stablehlo.add %v1259, %v1251 : tensor<64x256x14x14xf32>
    %v1261 = stablehlo.rsqrt %v1260 : tensor<64x256x14x14xf32>
    %v1262 = stablehlo.multiply %v1255, %v1261 : tensor<64x256x14x14xf32>
    %v1263 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1264 = stablehlo.broadcast_in_dim %s3b5bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1265 = stablehlo.multiply %v1262, %v1263 : tensor<64x256x14x14xf32>
    %v1266 = stablehlo.add %v1265, %v1264 : tensor<64x256x14x14xf32>
    %v1267 = stablehlo.reshape %v1266 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1269 = stablehlo.maximum %v1267, %v1268 : tensor<64x50176xf32>
    %v1270 = stablehlo.reshape %v1269 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1271 = stablehlo.convert %v1270 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1272 = stablehlo.convert %s3b5W3 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v1273 = stablehlo.convolution(%v1271, %v1272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v1274 = stablehlo.convert %v1273 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v1275 = stablehlo.broadcast_in_dim %zb1024, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1276 = stablehlo.add %v1274, %v1275 : tensor<64x1024x14x14xf32>
    %v1277 = stablehlo.reshape %v1276 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1278 = stablehlo.reshape %v1277 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1280 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v1281 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v1282 = stablehlo.reduce(%v1278 init: %v1279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1283 = stablehlo.broadcast_in_dim %v1282, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1284 = stablehlo.divide %v1283, %v1280 : tensor<64x1024x14x14xf32>
    %v1285 = stablehlo.subtract %v1278, %v1284 : tensor<64x1024x14x14xf32>
    %v1286 = stablehlo.multiply %v1285, %v1285 : tensor<64x1024x14x14xf32>
    %v1287 = stablehlo.reduce(%v1286 init: %v1279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v1288 = stablehlo.broadcast_in_dim %v1287, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1289 = stablehlo.divide %v1288, %v1280 : tensor<64x1024x14x14xf32>
    %v1290 = stablehlo.add %v1289, %v1281 : tensor<64x1024x14x14xf32>
    %v1291 = stablehlo.rsqrt %v1290 : tensor<64x1024x14x14xf32>
    %v1292 = stablehlo.multiply %v1285, %v1291 : tensor<64x1024x14x14xf32>
    %v1293 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1294 = stablehlo.broadcast_in_dim %s3b5bt3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v1295 = stablehlo.multiply %v1292, %v1293 : tensor<64x1024x14x14xf32>
    %v1296 = stablehlo.add %v1295, %v1294 : tensor<64x1024x14x14xf32>
    %v1297 = stablehlo.reshape %v1296 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v1298 = stablehlo.add %v1297, %v1209 : tensor<64x200704xf32>
    %v1299 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v1300 = stablehlo.maximum %v1298, %v1299 : tensor<64x200704xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1302 = stablehlo.convert %v1301 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1303 = stablehlo.convert %s4b0W1 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v1304 = stablehlo.convolution(%v1302, %v1303)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x14x14xbf16>
    %v1305 = stablehlo.convert %v1304 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v1306 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1307 = stablehlo.add %v1305, %v1306 : tensor<64x512x14x14xf32>
    %v1308 = stablehlo.reshape %v1307 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1309 = stablehlo.reshape %v1308 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1310 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1311 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v1312 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v1313 = stablehlo.reduce(%v1309 init: %v1310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1314 = stablehlo.broadcast_in_dim %v1313, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1315 = stablehlo.divide %v1314, %v1311 : tensor<64x512x14x14xf32>
    %v1316 = stablehlo.subtract %v1309, %v1315 : tensor<64x512x14x14xf32>
    %v1317 = stablehlo.multiply %v1316, %v1316 : tensor<64x512x14x14xf32>
    %v1318 = stablehlo.reduce(%v1317 init: %v1310) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v1319 = stablehlo.broadcast_in_dim %v1318, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1320 = stablehlo.divide %v1319, %v1311 : tensor<64x512x14x14xf32>
    %v1321 = stablehlo.add %v1320, %v1312 : tensor<64x512x14x14xf32>
    %v1322 = stablehlo.rsqrt %v1321 : tensor<64x512x14x14xf32>
    %v1323 = stablehlo.multiply %v1316, %v1322 : tensor<64x512x14x14xf32>
    %v1324 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1325 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v1326 = stablehlo.multiply %v1323, %v1324 : tensor<64x512x14x14xf32>
    %v1327 = stablehlo.add %v1326, %v1325 : tensor<64x512x14x14xf32>
    %v1328 = stablehlo.reshape %v1327 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v1329 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1330 = stablehlo.maximum %v1328, %v1329 : tensor<64x100352xf32>
    %v1331 = stablehlo.reshape %v1330 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v1332 = stablehlo.convert %v1331 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v1333 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1334 = stablehlo.convolution(%v1332, %v1333)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1335 = stablehlo.convert %v1334 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1336 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1337 = stablehlo.add %v1335, %v1336 : tensor<64x512x7x7xf32>
    %v1338 = stablehlo.reshape %v1337 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1340 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1341 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1342 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1343 = stablehlo.reduce(%v1339 init: %v1340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1344 = stablehlo.broadcast_in_dim %v1343, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1345 = stablehlo.divide %v1344, %v1341 : tensor<64x512x7x7xf32>
    %v1346 = stablehlo.subtract %v1339, %v1345 : tensor<64x512x7x7xf32>
    %v1347 = stablehlo.multiply %v1346, %v1346 : tensor<64x512x7x7xf32>
    %v1348 = stablehlo.reduce(%v1347 init: %v1340) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1349 = stablehlo.broadcast_in_dim %v1348, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1350 = stablehlo.divide %v1349, %v1341 : tensor<64x512x7x7xf32>
    %v1351 = stablehlo.add %v1350, %v1342 : tensor<64x512x7x7xf32>
    %v1352 = stablehlo.rsqrt %v1351 : tensor<64x512x7x7xf32>
    %v1353 = stablehlo.multiply %v1346, %v1352 : tensor<64x512x7x7xf32>
    %v1354 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1355 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1356 = stablehlo.multiply %v1353, %v1354 : tensor<64x512x7x7xf32>
    %v1357 = stablehlo.add %v1356, %v1355 : tensor<64x512x7x7xf32>
    %v1358 = stablehlo.reshape %v1357 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1359 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1360 = stablehlo.maximum %v1358, %v1359 : tensor<64x25088xf32>
    %v1361 = stablehlo.reshape %v1360 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1362 = stablehlo.convert %v1361 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1363 = stablehlo.convert %s4b0W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1364 = stablehlo.convolution(%v1362, %v1363)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1365 = stablehlo.convert %v1364 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1366 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1367 = stablehlo.add %v1365, %v1366 : tensor<64x2048x7x7xf32>
    %v1368 = stablehlo.reshape %v1367 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1369 = stablehlo.reshape %v1368 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1370 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1371 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1372 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1373 = stablehlo.reduce(%v1369 init: %v1370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1375 = stablehlo.divide %v1374, %v1371 : tensor<64x2048x7x7xf32>
    %v1376 = stablehlo.subtract %v1369, %v1375 : tensor<64x2048x7x7xf32>
    %v1377 = stablehlo.multiply %v1376, %v1376 : tensor<64x2048x7x7xf32>
    %v1378 = stablehlo.reduce(%v1377 init: %v1370) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1379 = stablehlo.broadcast_in_dim %v1378, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1380 = stablehlo.divide %v1379, %v1371 : tensor<64x2048x7x7xf32>
    %v1381 = stablehlo.add %v1380, %v1372 : tensor<64x2048x7x7xf32>
    %v1382 = stablehlo.rsqrt %v1381 : tensor<64x2048x7x7xf32>
    %v1383 = stablehlo.multiply %v1376, %v1382 : tensor<64x2048x7x7xf32>
    %v1384 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1385 = stablehlo.broadcast_in_dim %s4b0bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1386 = stablehlo.multiply %v1383, %v1384 : tensor<64x2048x7x7xf32>
    %v1387 = stablehlo.add %v1386, %v1385 : tensor<64x2048x7x7xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1389 = stablehlo.reshape %v1300 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v1390 = stablehlo.convert %v1389 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v1391 = stablehlo.convert %s4b0Wp : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xbf16>
    %v1392 = stablehlo.convolution(%v1390, %v1391)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<2048x1024x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1393 = stablehlo.convert %v1392 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1394 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1395 = stablehlo.add %v1393, %v1394 : tensor<64x2048x7x7xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1397 = stablehlo.reshape %v1396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1399 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1400 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1401 = stablehlo.reduce(%v1397 init: %v1398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1402 = stablehlo.broadcast_in_dim %v1401, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1403 = stablehlo.divide %v1402, %v1399 : tensor<64x2048x7x7xf32>
    %v1404 = stablehlo.subtract %v1397, %v1403 : tensor<64x2048x7x7xf32>
    %v1405 = stablehlo.multiply %v1404, %v1404 : tensor<64x2048x7x7xf32>
    %v1406 = stablehlo.reduce(%v1405 init: %v1398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1407 = stablehlo.broadcast_in_dim %v1406, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1408 = stablehlo.divide %v1407, %v1399 : tensor<64x2048x7x7xf32>
    %v1409 = stablehlo.add %v1408, %v1400 : tensor<64x2048x7x7xf32>
    %v1410 = stablehlo.rsqrt %v1409 : tensor<64x2048x7x7xf32>
    %v1411 = stablehlo.multiply %v1404, %v1410 : tensor<64x2048x7x7xf32>
    %v1412 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1413 = stablehlo.broadcast_in_dim %s4b0btp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1414 = stablehlo.multiply %v1411, %v1412 : tensor<64x2048x7x7xf32>
    %v1415 = stablehlo.add %v1414, %v1413 : tensor<64x2048x7x7xf32>
    %v1416 = stablehlo.reshape %v1415 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1417 = stablehlo.add %v1388, %v1416 : tensor<64x100352xf32>
    %v1418 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1419 = stablehlo.maximum %v1417, %v1418 : tensor<64x100352xf32>
    %v1420 = stablehlo.reshape %v1419 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1421 = stablehlo.convert %v1420 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1422 = stablehlo.convert %s4b1W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1423 = stablehlo.convolution(%v1421, %v1422)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1424 = stablehlo.convert %v1423 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1426 = stablehlo.add %v1424, %v1425 : tensor<64x512x7x7xf32>
    %v1427 = stablehlo.reshape %v1426 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1428 = stablehlo.reshape %v1427 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1429 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1430 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1431 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1432 = stablehlo.reduce(%v1428 init: %v1429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1433 = stablehlo.broadcast_in_dim %v1432, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1434 = stablehlo.divide %v1433, %v1430 : tensor<64x512x7x7xf32>
    %v1435 = stablehlo.subtract %v1428, %v1434 : tensor<64x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1435, %v1435 : tensor<64x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1429) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1439 = stablehlo.divide %v1438, %v1430 : tensor<64x512x7x7xf32>
    %v1440 = stablehlo.add %v1439, %v1431 : tensor<64x512x7x7xf32>
    %v1441 = stablehlo.rsqrt %v1440 : tensor<64x512x7x7xf32>
    %v1442 = stablehlo.multiply %v1435, %v1441 : tensor<64x512x7x7xf32>
    %v1443 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1444 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1445 = stablehlo.multiply %v1442, %v1443 : tensor<64x512x7x7xf32>
    %v1446 = stablehlo.add %v1445, %v1444 : tensor<64x512x7x7xf32>
    %v1447 = stablehlo.reshape %v1446 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1448 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1449 = stablehlo.maximum %v1447, %v1448 : tensor<64x25088xf32>
    %v1450 = stablehlo.reshape %v1449 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1451 = stablehlo.convert %v1450 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1452 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1453 = stablehlo.convolution(%v1451, %v1452)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1454 = stablehlo.convert %v1453 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1455 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1456 = stablehlo.add %v1454, %v1455 : tensor<64x512x7x7xf32>
    %v1457 = stablehlo.reshape %v1456 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1458 = stablehlo.reshape %v1457 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1459 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1460 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1461 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1462 = stablehlo.reduce(%v1458 init: %v1459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1463 = stablehlo.broadcast_in_dim %v1462, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1464 = stablehlo.divide %v1463, %v1460 : tensor<64x512x7x7xf32>
    %v1465 = stablehlo.subtract %v1458, %v1464 : tensor<64x512x7x7xf32>
    %v1466 = stablehlo.multiply %v1465, %v1465 : tensor<64x512x7x7xf32>
    %v1467 = stablehlo.reduce(%v1466 init: %v1459) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1468 = stablehlo.broadcast_in_dim %v1467, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1469 = stablehlo.divide %v1468, %v1460 : tensor<64x512x7x7xf32>
    %v1470 = stablehlo.add %v1469, %v1461 : tensor<64x512x7x7xf32>
    %v1471 = stablehlo.rsqrt %v1470 : tensor<64x512x7x7xf32>
    %v1472 = stablehlo.multiply %v1465, %v1471 : tensor<64x512x7x7xf32>
    %v1473 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1474 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1475 = stablehlo.multiply %v1472, %v1473 : tensor<64x512x7x7xf32>
    %v1476 = stablehlo.add %v1475, %v1474 : tensor<64x512x7x7xf32>
    %v1477 = stablehlo.reshape %v1476 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1478 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1479 = stablehlo.maximum %v1477, %v1478 : tensor<64x25088xf32>
    %v1480 = stablehlo.reshape %v1479 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1481 = stablehlo.convert %v1480 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1482 = stablehlo.convert %s4b1W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1483 = stablehlo.convolution(%v1481, %v1482)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1484 = stablehlo.convert %v1483 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1485 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1486 = stablehlo.add %v1484, %v1485 : tensor<64x2048x7x7xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1488 = stablehlo.reshape %v1487 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1491 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1492 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1494 = stablehlo.divide %v1493, %v1490 : tensor<64x2048x7x7xf32>
    %v1495 = stablehlo.subtract %v1488, %v1494 : tensor<64x2048x7x7xf32>
    %v1496 = stablehlo.multiply %v1495, %v1495 : tensor<64x2048x7x7xf32>
    %v1497 = stablehlo.reduce(%v1496 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1498 = stablehlo.broadcast_in_dim %v1497, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1499 = stablehlo.divide %v1498, %v1490 : tensor<64x2048x7x7xf32>
    %v1500 = stablehlo.add %v1499, %v1491 : tensor<64x2048x7x7xf32>
    %v1501 = stablehlo.rsqrt %v1500 : tensor<64x2048x7x7xf32>
    %v1502 = stablehlo.multiply %v1495, %v1501 : tensor<64x2048x7x7xf32>
    %v1503 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1504 = stablehlo.broadcast_in_dim %s4b1bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1505 = stablehlo.multiply %v1502, %v1503 : tensor<64x2048x7x7xf32>
    %v1506 = stablehlo.add %v1505, %v1504 : tensor<64x2048x7x7xf32>
    %v1507 = stablehlo.reshape %v1506 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1508 = stablehlo.add %v1507, %v1419 : tensor<64x100352xf32>
    %v1509 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1510 = stablehlo.maximum %v1508, %v1509 : tensor<64x100352xf32>
    %v1511 = stablehlo.reshape %v1510 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1512 = stablehlo.convert %v1511 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1513 = stablehlo.convert %s4b2W1 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1514 = stablehlo.convolution(%v1512, %v1513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1515 = stablehlo.convert %v1514 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1516 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1517 = stablehlo.add %v1515, %v1516 : tensor<64x512x7x7xf32>
    %v1518 = stablehlo.reshape %v1517 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1520 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1521 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1522 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1523 = stablehlo.reduce(%v1519 init: %v1520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1524 = stablehlo.broadcast_in_dim %v1523, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1525 = stablehlo.divide %v1524, %v1521 : tensor<64x512x7x7xf32>
    %v1526 = stablehlo.subtract %v1519, %v1525 : tensor<64x512x7x7xf32>
    %v1527 = stablehlo.multiply %v1526, %v1526 : tensor<64x512x7x7xf32>
    %v1528 = stablehlo.reduce(%v1527 init: %v1520) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1529 = stablehlo.broadcast_in_dim %v1528, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1530 = stablehlo.divide %v1529, %v1521 : tensor<64x512x7x7xf32>
    %v1531 = stablehlo.add %v1530, %v1522 : tensor<64x512x7x7xf32>
    %v1532 = stablehlo.rsqrt %v1531 : tensor<64x512x7x7xf32>
    %v1533 = stablehlo.multiply %v1526, %v1532 : tensor<64x512x7x7xf32>
    %v1534 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1535 = stablehlo.broadcast_in_dim %s4b2bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1536 = stablehlo.multiply %v1533, %v1534 : tensor<64x512x7x7xf32>
    %v1537 = stablehlo.add %v1536, %v1535 : tensor<64x512x7x7xf32>
    %v1538 = stablehlo.reshape %v1537 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1539 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1540 = stablehlo.maximum %v1538, %v1539 : tensor<64x25088xf32>
    %v1541 = stablehlo.reshape %v1540 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1542 = stablehlo.convert %v1541 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1543 = stablehlo.convert %s4b2W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1544 = stablehlo.convolution(%v1542, %v1543)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1545 = stablehlo.convert %v1544 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1546 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1547 = stablehlo.add %v1545, %v1546 : tensor<64x512x7x7xf32>
    %v1548 = stablehlo.reshape %v1547 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1549 = stablehlo.reshape %v1548 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1551 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1552 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1553 = stablehlo.reduce(%v1549 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1554 = stablehlo.broadcast_in_dim %v1553, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1555 = stablehlo.divide %v1554, %v1551 : tensor<64x512x7x7xf32>
    %v1556 = stablehlo.subtract %v1549, %v1555 : tensor<64x512x7x7xf32>
    %v1557 = stablehlo.multiply %v1556, %v1556 : tensor<64x512x7x7xf32>
    %v1558 = stablehlo.reduce(%v1557 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1559 = stablehlo.broadcast_in_dim %v1558, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1560 = stablehlo.divide %v1559, %v1551 : tensor<64x512x7x7xf32>
    %v1561 = stablehlo.add %v1560, %v1552 : tensor<64x512x7x7xf32>
    %v1562 = stablehlo.rsqrt %v1561 : tensor<64x512x7x7xf32>
    %v1563 = stablehlo.multiply %v1556, %v1562 : tensor<64x512x7x7xf32>
    %v1564 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1565 = stablehlo.broadcast_in_dim %s4b2bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1566 = stablehlo.multiply %v1563, %v1564 : tensor<64x512x7x7xf32>
    %v1567 = stablehlo.add %v1566, %v1565 : tensor<64x512x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1569 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1570 = stablehlo.maximum %v1568, %v1569 : tensor<64x25088xf32>
    %v1571 = stablehlo.reshape %v1570 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1572 = stablehlo.convert %v1571 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1573 = stablehlo.convert %s4b2W3 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1574 = stablehlo.convolution(%v1572, %v1573)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1575 = stablehlo.convert %v1574 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1576 = stablehlo.broadcast_in_dim %zb2048, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1577 = stablehlo.add %v1575, %v1576 : tensor<64x2048x7x7xf32>
    %v1578 = stablehlo.reshape %v1577 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1579 = stablehlo.reshape %v1578 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1581 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1582 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1583 = stablehlo.reduce(%v1579 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1584 = stablehlo.broadcast_in_dim %v1583, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1585 = stablehlo.divide %v1584, %v1581 : tensor<64x2048x7x7xf32>
    %v1586 = stablehlo.subtract %v1579, %v1585 : tensor<64x2048x7x7xf32>
    %v1587 = stablehlo.multiply %v1586, %v1586 : tensor<64x2048x7x7xf32>
    %v1588 = stablehlo.reduce(%v1587 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1589 = stablehlo.broadcast_in_dim %v1588, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1590 = stablehlo.divide %v1589, %v1581 : tensor<64x2048x7x7xf32>
    %v1591 = stablehlo.add %v1590, %v1582 : tensor<64x2048x7x7xf32>
    %v1592 = stablehlo.rsqrt %v1591 : tensor<64x2048x7x7xf32>
    %v1593 = stablehlo.multiply %v1586, %v1592 : tensor<64x2048x7x7xf32>
    %v1594 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1595 = stablehlo.broadcast_in_dim %s4b2bt3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1596 = stablehlo.multiply %v1593, %v1594 : tensor<64x2048x7x7xf32>
    %v1597 = stablehlo.add %v1596, %v1595 : tensor<64x2048x7x7xf32>
    %v1598 = stablehlo.reshape %v1597 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1599 = stablehlo.add %v1598, %v1510 : tensor<64x100352xf32>
    %v1600 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1601 = stablehlo.maximum %v1599, %v1600 : tensor<64x100352xf32>
    %v1602 = stablehlo.reshape %v1601 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1603 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1604 = stablehlo.reduce(%v1602 init: %v1603) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048xf32>
    %v1605 = stablehlo.constant dense<49.0> : tensor<64x2048xf32>
    %v1606 = stablehlo.divide %v1604, %v1605 : tensor<64x2048xf32>
    %v1607 = stablehlo.dot_general %v1606, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<2048x1000xf32>) -> tensor<64x1000xf32>
    %v1608 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1609 = stablehlo.add %v1607, %v1608 : tensor<64x1000xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1612 = stablehlo.exponential %v1610 : tensor<64x1x1000xf32>
    %v1613 = stablehlo.reduce(%v1612 init: %v1611) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1614 = stablehlo.broadcast_in_dim %v1613, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v1615 = stablehlo.divide %v1612, %v1614 : tensor<64x1x1000xf32>
    %v1616 = stablehlo.reshape %v1615 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v1617 = stablehlo.subtract %v1616, %onehot : tensor<64x1000xf32>
    %v1618 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v1619 = stablehlo.multiply %onehot, %v1618 : tensor<64x1000xf32>
    %v1620 = stablehlo.add %v1617, %v1619 : tensor<64x1000xf32>
    %v1621 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v1622 = stablehlo.add %v1620, %v1621 : tensor<64x1000xf32>
    %v1623 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v1624 = stablehlo.divide %v1622, %v1623 : tensor<64x1000xf32>
    %v1625 = stablehlo.reshape %v1624 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1626 = stablehlo.dot_general %v1625, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<2048x1000xf32>) -> tensor<64x1x2048xf32>
    %v1627 = stablehlo.reshape %v1626 : (tensor<64x1x2048xf32>) -> tensor<64x2048xf32>
    %v1628 = stablehlo.dot_general %v1606, %v1624, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x2048xf32>, tensor<64x1000xf32>) -> tensor<2048x1000xf32>
    %v1629 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1630 = stablehlo.reduce(%v1624 init: %v1629) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1631 = stablehlo.broadcast_in_dim %v1627, dims = [0, 1] : (tensor<64x2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1632 = stablehlo.constant dense<49.0> : tensor<64x2048x7x7xf32>
    %v1633 = stablehlo.divide %v1631, %v1632 : tensor<64x2048x7x7xf32>
    %v1634 = stablehlo.reshape %v1633 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1635 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1636 = stablehlo.compare GT, %v1599, %v1635 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v1637 = stablehlo.select %v1636, %v1634, %v1635 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v1638 = stablehlo.reshape %v1578 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1640 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1641 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1642 = stablehlo.reduce(%v1638 init: %v1639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1643 = stablehlo.broadcast_in_dim %v1642, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1644 = stablehlo.divide %v1643, %v1640 : tensor<64x2048x7x7xf32>
    %v1645 = stablehlo.subtract %v1638, %v1644 : tensor<64x2048x7x7xf32>
    %v1646 = stablehlo.multiply %v1645, %v1645 : tensor<64x2048x7x7xf32>
    %v1647 = stablehlo.reduce(%v1646 init: %v1639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1648 = stablehlo.broadcast_in_dim %v1647, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1649 = stablehlo.divide %v1648, %v1640 : tensor<64x2048x7x7xf32>
    %v1650 = stablehlo.add %v1649, %v1641 : tensor<64x2048x7x7xf32>
    %v1651 = stablehlo.rsqrt %v1650 : tensor<64x2048x7x7xf32>
    %v1652 = stablehlo.multiply %v1645, %v1651 : tensor<64x2048x7x7xf32>
    %v1653 = stablehlo.broadcast_in_dim %s4b2g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1654 = stablehlo.reshape %v1637 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1655 = stablehlo.multiply %v1653, %v1654 : tensor<64x2048x7x7xf32>
    %v1656 = stablehlo.reduce(%v1655 init: %v1639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1656, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1658 = stablehlo.multiply %v1652, %v1655 : tensor<64x2048x7x7xf32>
    %v1659 = stablehlo.reduce(%v1658 init: %v1639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1660 = stablehlo.broadcast_in_dim %v1659, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1661 = stablehlo.multiply %v1655, %v1640 : tensor<64x2048x7x7xf32>
    %v1662 = stablehlo.subtract %v1661, %v1657 : tensor<64x2048x7x7xf32>
    %v1663 = stablehlo.multiply %v1652, %v1660 : tensor<64x2048x7x7xf32>
    %v1664 = stablehlo.subtract %v1662, %v1663 : tensor<64x2048x7x7xf32>
    %v1665 = stablehlo.divide %v1651, %v1640 : tensor<64x2048x7x7xf32>
    %v1666 = stablehlo.multiply %v1665, %v1664 : tensor<64x2048x7x7xf32>
    %v1667 = stablehlo.reshape %v1666 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1668 = stablehlo.reshape %v1667 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1669 = stablehlo.reverse %s4b2W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1670 = stablehlo.transpose %v1669, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1671 = stablehlo.convert %v1668 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1672 = stablehlo.convert %v1670 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1673 = stablehlo.convolution(%v1671, %v1672)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1674 = stablehlo.convert %v1673 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1675 = stablehlo.reshape %v1674 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1676 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1677 = stablehlo.compare GT, %v1568, %v1676 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1678 = stablehlo.select %v1677, %v1675, %v1676 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1679 = stablehlo.reshape %v1548 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1680 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1681 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1682 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1683 = stablehlo.reduce(%v1679 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1684 = stablehlo.broadcast_in_dim %v1683, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1685 = stablehlo.divide %v1684, %v1681 : tensor<64x512x7x7xf32>
    %v1686 = stablehlo.subtract %v1679, %v1685 : tensor<64x512x7x7xf32>
    %v1687 = stablehlo.multiply %v1686, %v1686 : tensor<64x512x7x7xf32>
    %v1688 = stablehlo.reduce(%v1687 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1689 = stablehlo.broadcast_in_dim %v1688, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1690 = stablehlo.divide %v1689, %v1681 : tensor<64x512x7x7xf32>
    %v1691 = stablehlo.add %v1690, %v1682 : tensor<64x512x7x7xf32>
    %v1692 = stablehlo.rsqrt %v1691 : tensor<64x512x7x7xf32>
    %v1693 = stablehlo.multiply %v1686, %v1692 : tensor<64x512x7x7xf32>
    %v1694 = stablehlo.broadcast_in_dim %s4b2g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1695 = stablehlo.reshape %v1678 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1696 = stablehlo.multiply %v1694, %v1695 : tensor<64x512x7x7xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1699 = stablehlo.multiply %v1693, %v1696 : tensor<64x512x7x7xf32>
    %v1700 = stablehlo.reduce(%v1699 init: %v1680) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1701 = stablehlo.broadcast_in_dim %v1700, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1702 = stablehlo.multiply %v1696, %v1681 : tensor<64x512x7x7xf32>
    %v1703 = stablehlo.subtract %v1702, %v1698 : tensor<64x512x7x7xf32>
    %v1704 = stablehlo.multiply %v1693, %v1701 : tensor<64x512x7x7xf32>
    %v1705 = stablehlo.subtract %v1703, %v1704 : tensor<64x512x7x7xf32>
    %v1706 = stablehlo.divide %v1692, %v1681 : tensor<64x512x7x7xf32>
    %v1707 = stablehlo.multiply %v1706, %v1705 : tensor<64x512x7x7xf32>
    %v1708 = stablehlo.reshape %v1707 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1709 = stablehlo.reshape %v1708 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1710 = stablehlo.reverse %s4b2W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1711 = stablehlo.transpose %v1710, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1712 = stablehlo.convert %v1709 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1713 = stablehlo.convert %v1711 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1714 = stablehlo.convolution(%v1712, %v1713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1715 = stablehlo.convert %v1714 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1716 = stablehlo.reshape %v1715 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1717 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1718 = stablehlo.compare GT, %v1538, %v1717 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1719 = stablehlo.select %v1718, %v1716, %v1717 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1720 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1721 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1722 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1723 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1724 = stablehlo.reduce(%v1720 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1725 = stablehlo.broadcast_in_dim %v1724, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1726 = stablehlo.divide %v1725, %v1722 : tensor<64x512x7x7xf32>
    %v1727 = stablehlo.subtract %v1720, %v1726 : tensor<64x512x7x7xf32>
    %v1728 = stablehlo.multiply %v1727, %v1727 : tensor<64x512x7x7xf32>
    %v1729 = stablehlo.reduce(%v1728 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1729, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1731 = stablehlo.divide %v1730, %v1722 : tensor<64x512x7x7xf32>
    %v1732 = stablehlo.add %v1731, %v1723 : tensor<64x512x7x7xf32>
    %v1733 = stablehlo.rsqrt %v1732 : tensor<64x512x7x7xf32>
    %v1734 = stablehlo.multiply %v1727, %v1733 : tensor<64x512x7x7xf32>
    %v1735 = stablehlo.broadcast_in_dim %s4b2g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1736 = stablehlo.reshape %v1719 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1737 = stablehlo.multiply %v1735, %v1736 : tensor<64x512x7x7xf32>
    %v1738 = stablehlo.reduce(%v1737 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1739 = stablehlo.broadcast_in_dim %v1738, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1740 = stablehlo.multiply %v1734, %v1737 : tensor<64x512x7x7xf32>
    %v1741 = stablehlo.reduce(%v1740 init: %v1721) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1742 = stablehlo.broadcast_in_dim %v1741, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1743 = stablehlo.multiply %v1737, %v1722 : tensor<64x512x7x7xf32>
    %v1744 = stablehlo.subtract %v1743, %v1739 : tensor<64x512x7x7xf32>
    %v1745 = stablehlo.multiply %v1734, %v1742 : tensor<64x512x7x7xf32>
    %v1746 = stablehlo.subtract %v1744, %v1745 : tensor<64x512x7x7xf32>
    %v1747 = stablehlo.divide %v1733, %v1722 : tensor<64x512x7x7xf32>
    %v1748 = stablehlo.multiply %v1747, %v1746 : tensor<64x512x7x7xf32>
    %v1749 = stablehlo.reshape %v1748 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1750 = stablehlo.reshape %v1749 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1751 = stablehlo.reverse %s4b2W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1752 = stablehlo.transpose %v1751, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1753 = stablehlo.convert %v1750 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1754 = stablehlo.convert %v1752 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1755 = stablehlo.convolution(%v1753, %v1754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1756 = stablehlo.convert %v1755 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1757 = stablehlo.reshape %v1756 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1758 = stablehlo.add %v1757, %v1637 : tensor<64x100352xf32>
    %v1759 = stablehlo.reshape %v1510 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1760 = stablehlo.reshape %v1749 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1761 = stablehlo.transpose %v1759, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v1762 = stablehlo.transpose %v1760, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1763 = stablehlo.convert %v1761 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v1764 = stablehlo.convert %v1762 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1765 = stablehlo.convolution(%v1763, %v1764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v1766 = stablehlo.convert %v1765 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v1767 = stablehlo.transpose %v1766, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1768 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1769 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1770 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1771 = stablehlo.reduce(%v1768 init: %v1769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1772 = stablehlo.broadcast_in_dim %v1771, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1773 = stablehlo.divide %v1772, %v1770 : tensor<64x512x7x7xf32>
    %v1774 = stablehlo.subtract %v1768, %v1773 : tensor<64x512x7x7xf32>
    %v1775 = stablehlo.multiply %v1774, %v1774 : tensor<64x512x7x7xf32>
    %v1776 = stablehlo.reduce(%v1775 init: %v1769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1777 = stablehlo.broadcast_in_dim %v1776, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1778 = stablehlo.divide %v1777, %v1770 : tensor<64x512x7x7xf32>
    %v1779 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1780 = stablehlo.add %v1778, %v1779 : tensor<64x512x7x7xf32>
    %v1781 = stablehlo.rsqrt %v1780 : tensor<64x512x7x7xf32>
    %v1782 = stablehlo.multiply %v1774, %v1781 : tensor<64x512x7x7xf32>
    %v1783 = stablehlo.reshape %v1719 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1784 = stablehlo.multiply %v1783, %v1782 : tensor<64x512x7x7xf32>
    %v1785 = stablehlo.reduce(%v1784 init: %v1769) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1786 = stablehlo.reshape %v1719 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1788 = stablehlo.reduce(%v1786 init: %v1787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1789 = stablehlo.reshape %v1540 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1790 = stablehlo.reshape %v1708 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1791 = stablehlo.transpose %v1789, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1792 = stablehlo.transpose %v1790, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1793 = stablehlo.convert %v1791 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1794 = stablehlo.convert %v1792 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1795 = stablehlo.convolution(%v1793, %v1794)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1796 = stablehlo.convert %v1795 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1797 = stablehlo.transpose %v1796, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1798 = stablehlo.reshape %v1548 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1800 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1801 = stablehlo.reduce(%v1798 init: %v1799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1802 = stablehlo.broadcast_in_dim %v1801, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1803 = stablehlo.divide %v1802, %v1800 : tensor<64x512x7x7xf32>
    %v1804 = stablehlo.subtract %v1798, %v1803 : tensor<64x512x7x7xf32>
    %v1805 = stablehlo.multiply %v1804, %v1804 : tensor<64x512x7x7xf32>
    %v1806 = stablehlo.reduce(%v1805 init: %v1799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1807 = stablehlo.broadcast_in_dim %v1806, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1808 = stablehlo.divide %v1807, %v1800 : tensor<64x512x7x7xf32>
    %v1809 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1810 = stablehlo.add %v1808, %v1809 : tensor<64x512x7x7xf32>
    %v1811 = stablehlo.rsqrt %v1810 : tensor<64x512x7x7xf32>
    %v1812 = stablehlo.multiply %v1804, %v1811 : tensor<64x512x7x7xf32>
    %v1813 = stablehlo.reshape %v1678 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1814 = stablehlo.multiply %v1813, %v1812 : tensor<64x512x7x7xf32>
    %v1815 = stablehlo.reduce(%v1814 init: %v1799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1816 = stablehlo.reshape %v1678 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1817 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1818 = stablehlo.reduce(%v1816 init: %v1817) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1819 = stablehlo.reshape %v1570 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1820 = stablehlo.reshape %v1667 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1821 = stablehlo.transpose %v1819, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1822 = stablehlo.transpose %v1820, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v1823 = stablehlo.convert %v1821 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1824 = stablehlo.convert %v1822 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v1825 = stablehlo.convolution(%v1823, %v1824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v1826 = stablehlo.convert %v1825 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v1827 = stablehlo.transpose %v1826, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1828 = stablehlo.reshape %v1578 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1829 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1830 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1831 = stablehlo.reduce(%v1828 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1832 = stablehlo.broadcast_in_dim %v1831, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1833 = stablehlo.divide %v1832, %v1830 : tensor<64x2048x7x7xf32>
    %v1834 = stablehlo.subtract %v1828, %v1833 : tensor<64x2048x7x7xf32>
    %v1835 = stablehlo.multiply %v1834, %v1834 : tensor<64x2048x7x7xf32>
    %v1836 = stablehlo.reduce(%v1835 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1837 = stablehlo.broadcast_in_dim %v1836, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1838 = stablehlo.divide %v1837, %v1830 : tensor<64x2048x7x7xf32>
    %v1839 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1840 = stablehlo.add %v1838, %v1839 : tensor<64x2048x7x7xf32>
    %v1841 = stablehlo.rsqrt %v1840 : tensor<64x2048x7x7xf32>
    %v1842 = stablehlo.multiply %v1834, %v1841 : tensor<64x2048x7x7xf32>
    %v1843 = stablehlo.reshape %v1637 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1844 = stablehlo.multiply %v1843, %v1842 : tensor<64x2048x7x7xf32>
    %v1845 = stablehlo.reduce(%v1844 init: %v1829) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1846 = stablehlo.reshape %v1637 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1848 = stablehlo.reduce(%v1846 init: %v1847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1849 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v1850 = stablehlo.compare GT, %v1508, %v1849 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v1851 = stablehlo.select %v1850, %v1758, %v1849 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v1852 = stablehlo.reshape %v1487 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1853 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1854 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v1855 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v1856 = stablehlo.reduce(%v1852 init: %v1853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1857 = stablehlo.broadcast_in_dim %v1856, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1858 = stablehlo.divide %v1857, %v1854 : tensor<64x2048x7x7xf32>
    %v1859 = stablehlo.subtract %v1852, %v1858 : tensor<64x2048x7x7xf32>
    %v1860 = stablehlo.multiply %v1859, %v1859 : tensor<64x2048x7x7xf32>
    %v1861 = stablehlo.reduce(%v1860 init: %v1853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1862 = stablehlo.broadcast_in_dim %v1861, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1863 = stablehlo.divide %v1862, %v1854 : tensor<64x2048x7x7xf32>
    %v1864 = stablehlo.add %v1863, %v1855 : tensor<64x2048x7x7xf32>
    %v1865 = stablehlo.rsqrt %v1864 : tensor<64x2048x7x7xf32>
    %v1866 = stablehlo.multiply %v1859, %v1865 : tensor<64x2048x7x7xf32>
    %v1867 = stablehlo.broadcast_in_dim %s4b1g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1868 = stablehlo.reshape %v1851 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1869 = stablehlo.multiply %v1867, %v1868 : tensor<64x2048x7x7xf32>
    %v1870 = stablehlo.reduce(%v1869 init: %v1853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1872 = stablehlo.multiply %v1866, %v1869 : tensor<64x2048x7x7xf32>
    %v1873 = stablehlo.reduce(%v1872 init: %v1853) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v1874 = stablehlo.broadcast_in_dim %v1873, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v1875 = stablehlo.multiply %v1869, %v1854 : tensor<64x2048x7x7xf32>
    %v1876 = stablehlo.subtract %v1875, %v1871 : tensor<64x2048x7x7xf32>
    %v1877 = stablehlo.multiply %v1866, %v1874 : tensor<64x2048x7x7xf32>
    %v1878 = stablehlo.subtract %v1876, %v1877 : tensor<64x2048x7x7xf32>
    %v1879 = stablehlo.divide %v1865, %v1854 : tensor<64x2048x7x7xf32>
    %v1880 = stablehlo.multiply %v1879, %v1878 : tensor<64x2048x7x7xf32>
    %v1881 = stablehlo.reshape %v1880 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1882 = stablehlo.reshape %v1881 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1883 = stablehlo.reverse %s4b1W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v1884 = stablehlo.transpose %v1883, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1885 = stablehlo.convert %v1882 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v1886 = stablehlo.convert %v1884 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v1887 = stablehlo.convolution(%v1885, %v1886)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1888 = stablehlo.convert %v1887 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1889 = stablehlo.reshape %v1888 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1890 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1891 = stablehlo.compare GT, %v1477, %v1890 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1892 = stablehlo.select %v1891, %v1889, %v1890 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1893 = stablehlo.reshape %v1457 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1894 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1895 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1896 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1897 = stablehlo.reduce(%v1893 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1898 = stablehlo.broadcast_in_dim %v1897, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1899 = stablehlo.divide %v1898, %v1895 : tensor<64x512x7x7xf32>
    %v1900 = stablehlo.subtract %v1893, %v1899 : tensor<64x512x7x7xf32>
    %v1901 = stablehlo.multiply %v1900, %v1900 : tensor<64x512x7x7xf32>
    %v1902 = stablehlo.reduce(%v1901 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1902, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1904 = stablehlo.divide %v1903, %v1895 : tensor<64x512x7x7xf32>
    %v1905 = stablehlo.add %v1904, %v1896 : tensor<64x512x7x7xf32>
    %v1906 = stablehlo.rsqrt %v1905 : tensor<64x512x7x7xf32>
    %v1907 = stablehlo.multiply %v1900, %v1906 : tensor<64x512x7x7xf32>
    %v1908 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1909 = stablehlo.reshape %v1892 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1910 = stablehlo.multiply %v1908, %v1909 : tensor<64x512x7x7xf32>
    %v1911 = stablehlo.reduce(%v1910 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1912 = stablehlo.broadcast_in_dim %v1911, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1913 = stablehlo.multiply %v1907, %v1910 : tensor<64x512x7x7xf32>
    %v1914 = stablehlo.reduce(%v1913 init: %v1894) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1915 = stablehlo.broadcast_in_dim %v1914, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1916 = stablehlo.multiply %v1910, %v1895 : tensor<64x512x7x7xf32>
    %v1917 = stablehlo.subtract %v1916, %v1912 : tensor<64x512x7x7xf32>
    %v1918 = stablehlo.multiply %v1907, %v1915 : tensor<64x512x7x7xf32>
    %v1919 = stablehlo.subtract %v1917, %v1918 : tensor<64x512x7x7xf32>
    %v1920 = stablehlo.divide %v1906, %v1895 : tensor<64x512x7x7xf32>
    %v1921 = stablehlo.multiply %v1920, %v1919 : tensor<64x512x7x7xf32>
    %v1922 = stablehlo.reshape %v1921 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1923 = stablehlo.reshape %v1922 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1924 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1925 = stablehlo.transpose %v1924, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1926 = stablehlo.convert %v1923 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1927 = stablehlo.convert %v1925 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1928 = stablehlo.convolution(%v1926, %v1927)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1929 = stablehlo.convert %v1928 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1930 = stablehlo.reshape %v1929 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1931 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1932 = stablehlo.compare GT, %v1447, %v1931 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1933 = stablehlo.select %v1932, %v1930, %v1931 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1934 = stablehlo.reshape %v1427 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1935 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1936 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1937 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1938 = stablehlo.reduce(%v1934 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1939 = stablehlo.broadcast_in_dim %v1938, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1940 = stablehlo.divide %v1939, %v1936 : tensor<64x512x7x7xf32>
    %v1941 = stablehlo.subtract %v1934, %v1940 : tensor<64x512x7x7xf32>
    %v1942 = stablehlo.multiply %v1941, %v1941 : tensor<64x512x7x7xf32>
    %v1943 = stablehlo.reduce(%v1942 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1944 = stablehlo.broadcast_in_dim %v1943, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1945 = stablehlo.divide %v1944, %v1936 : tensor<64x512x7x7xf32>
    %v1946 = stablehlo.add %v1945, %v1937 : tensor<64x512x7x7xf32>
    %v1947 = stablehlo.rsqrt %v1946 : tensor<64x512x7x7xf32>
    %v1948 = stablehlo.multiply %v1941, %v1947 : tensor<64x512x7x7xf32>
    %v1949 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1950 = stablehlo.reshape %v1933 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1951 = stablehlo.multiply %v1949, %v1950 : tensor<64x512x7x7xf32>
    %v1952 = stablehlo.reduce(%v1951 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1953 = stablehlo.broadcast_in_dim %v1952, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1954 = stablehlo.multiply %v1948, %v1951 : tensor<64x512x7x7xf32>
    %v1955 = stablehlo.reduce(%v1954 init: %v1935) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1956 = stablehlo.broadcast_in_dim %v1955, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1957 = stablehlo.multiply %v1951, %v1936 : tensor<64x512x7x7xf32>
    %v1958 = stablehlo.subtract %v1957, %v1953 : tensor<64x512x7x7xf32>
    %v1959 = stablehlo.multiply %v1948, %v1956 : tensor<64x512x7x7xf32>
    %v1960 = stablehlo.subtract %v1958, %v1959 : tensor<64x512x7x7xf32>
    %v1961 = stablehlo.divide %v1947, %v1936 : tensor<64x512x7x7xf32>
    %v1962 = stablehlo.multiply %v1961, %v1960 : tensor<64x512x7x7xf32>
    %v1963 = stablehlo.reshape %v1962 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1964 = stablehlo.reshape %v1963 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1965 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x2048x1x1xf32>
    %v1966 = stablehlo.transpose %v1965, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v1967 = stablehlo.convert %v1964 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1968 = stablehlo.convert %v1966 : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xbf16>
    %v1969 = stablehlo.convolution(%v1967, %v1968)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<2048x512x1x1xbf16>) -> tensor<64x2048x7x7xbf16>
    %v1970 = stablehlo.convert %v1969 : (tensor<64x2048x7x7xbf16>) -> tensor<64x2048x7x7xf32>
    %v1971 = stablehlo.reshape %v1970 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v1972 = stablehlo.add %v1971, %v1851 : tensor<64x100352xf32>
    %v1973 = stablehlo.reshape %v1419 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v1974 = stablehlo.reshape %v1963 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1975 = stablehlo.transpose %v1973, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v1976 = stablehlo.transpose %v1974, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1977 = stablehlo.convert %v1975 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v1978 = stablehlo.convert %v1976 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1979 = stablehlo.convolution(%v1977, %v1978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<2048x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<2048x512x1x1xbf16>
    %v1980 = stablehlo.convert %v1979 : (tensor<2048x512x1x1xbf16>) -> tensor<2048x512x1x1xf32>
    %v1981 = stablehlo.transpose %v1980, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v1982 = stablehlo.reshape %v1427 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1983 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1984 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1985 = stablehlo.reduce(%v1982 init: %v1983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1986 = stablehlo.broadcast_in_dim %v1985, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1987 = stablehlo.divide %v1986, %v1984 : tensor<64x512x7x7xf32>
    %v1988 = stablehlo.subtract %v1982, %v1987 : tensor<64x512x7x7xf32>
    %v1989 = stablehlo.multiply %v1988, %v1988 : tensor<64x512x7x7xf32>
    %v1990 = stablehlo.reduce(%v1989 init: %v1983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1991 = stablehlo.broadcast_in_dim %v1990, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1992 = stablehlo.divide %v1991, %v1984 : tensor<64x512x7x7xf32>
    %v1993 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1994 = stablehlo.add %v1992, %v1993 : tensor<64x512x7x7xf32>
    %v1995 = stablehlo.rsqrt %v1994 : tensor<64x512x7x7xf32>
    %v1996 = stablehlo.multiply %v1988, %v1995 : tensor<64x512x7x7xf32>
    %v1997 = stablehlo.reshape %v1933 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1998 = stablehlo.multiply %v1997, %v1996 : tensor<64x512x7x7xf32>
    %v1999 = stablehlo.reduce(%v1998 init: %v1983) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2000 = stablehlo.reshape %v1933 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2001 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2002 = stablehlo.reduce(%v2000 init: %v2001) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2003 = stablehlo.reshape %v1449 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2004 = stablehlo.reshape %v1922 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2005 = stablehlo.transpose %v2003, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2006 = stablehlo.transpose %v2004, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2007 = stablehlo.convert %v2005 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2008 = stablehlo.convert %v2006 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2009 = stablehlo.convolution(%v2007, %v2008)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2010 = stablehlo.convert %v2009 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2011 = stablehlo.transpose %v2010, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2012 = stablehlo.reshape %v1457 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2014 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2015 = stablehlo.reduce(%v2012 init: %v2013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2016 = stablehlo.broadcast_in_dim %v2015, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2017 = stablehlo.divide %v2016, %v2014 : tensor<64x512x7x7xf32>
    %v2018 = stablehlo.subtract %v2012, %v2017 : tensor<64x512x7x7xf32>
    %v2019 = stablehlo.multiply %v2018, %v2018 : tensor<64x512x7x7xf32>
    %v2020 = stablehlo.reduce(%v2019 init: %v2013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2021 = stablehlo.broadcast_in_dim %v2020, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2022 = stablehlo.divide %v2021, %v2014 : tensor<64x512x7x7xf32>
    %v2023 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2024 = stablehlo.add %v2022, %v2023 : tensor<64x512x7x7xf32>
    %v2025 = stablehlo.rsqrt %v2024 : tensor<64x512x7x7xf32>
    %v2026 = stablehlo.multiply %v2018, %v2025 : tensor<64x512x7x7xf32>
    %v2027 = stablehlo.reshape %v1892 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2028 = stablehlo.multiply %v2027, %v2026 : tensor<64x512x7x7xf32>
    %v2029 = stablehlo.reduce(%v2028 init: %v2013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2030 = stablehlo.reshape %v1892 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2031 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2032 = stablehlo.reduce(%v2030 init: %v2031) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2033 = stablehlo.reshape %v1479 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2034 = stablehlo.reshape %v1881 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2035 = stablehlo.transpose %v2033, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2036 = stablehlo.transpose %v2034, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2037 = stablehlo.convert %v2035 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2038 = stablehlo.convert %v2036 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2039 = stablehlo.convolution(%v2037, %v2038)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2040 = stablehlo.convert %v2039 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2041 = stablehlo.transpose %v2040, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2042 = stablehlo.reshape %v1487 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2043 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2044 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2045 = stablehlo.reduce(%v2042 init: %v2043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2046 = stablehlo.broadcast_in_dim %v2045, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2047 = stablehlo.divide %v2046, %v2044 : tensor<64x2048x7x7xf32>
    %v2048 = stablehlo.subtract %v2042, %v2047 : tensor<64x2048x7x7xf32>
    %v2049 = stablehlo.multiply %v2048, %v2048 : tensor<64x2048x7x7xf32>
    %v2050 = stablehlo.reduce(%v2049 init: %v2043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2051 = stablehlo.broadcast_in_dim %v2050, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2052 = stablehlo.divide %v2051, %v2044 : tensor<64x2048x7x7xf32>
    %v2053 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2054 = stablehlo.add %v2052, %v2053 : tensor<64x2048x7x7xf32>
    %v2055 = stablehlo.rsqrt %v2054 : tensor<64x2048x7x7xf32>
    %v2056 = stablehlo.multiply %v2048, %v2055 : tensor<64x2048x7x7xf32>
    %v2057 = stablehlo.reshape %v1851 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2058 = stablehlo.multiply %v2057, %v2056 : tensor<64x2048x7x7xf32>
    %v2059 = stablehlo.reduce(%v2058 init: %v2043) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2060 = stablehlo.reshape %v1851 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2062 = stablehlo.reduce(%v2060 init: %v2061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2063 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2064 = stablehlo.compare GT, %v1417, %v2063 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2065 = stablehlo.select %v2064, %v1972, %v2063 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2066 = stablehlo.reshape %v1368 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2067 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2068 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2069 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2070 = stablehlo.reduce(%v2066 init: %v2067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2071 = stablehlo.broadcast_in_dim %v2070, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2072 = stablehlo.divide %v2071, %v2068 : tensor<64x2048x7x7xf32>
    %v2073 = stablehlo.subtract %v2066, %v2072 : tensor<64x2048x7x7xf32>
    %v2074 = stablehlo.multiply %v2073, %v2073 : tensor<64x2048x7x7xf32>
    %v2075 = stablehlo.reduce(%v2074 init: %v2067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2075, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2077 = stablehlo.divide %v2076, %v2068 : tensor<64x2048x7x7xf32>
    %v2078 = stablehlo.add %v2077, %v2069 : tensor<64x2048x7x7xf32>
    %v2079 = stablehlo.rsqrt %v2078 : tensor<64x2048x7x7xf32>
    %v2080 = stablehlo.multiply %v2073, %v2079 : tensor<64x2048x7x7xf32>
    %v2081 = stablehlo.broadcast_in_dim %s4b0g3, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2082 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2083 = stablehlo.multiply %v2081, %v2082 : tensor<64x2048x7x7xf32>
    %v2084 = stablehlo.reduce(%v2083 init: %v2067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2085 = stablehlo.broadcast_in_dim %v2084, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2086 = stablehlo.multiply %v2080, %v2083 : tensor<64x2048x7x7xf32>
    %v2087 = stablehlo.reduce(%v2086 init: %v2067) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2088 = stablehlo.broadcast_in_dim %v2087, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2089 = stablehlo.multiply %v2083, %v2068 : tensor<64x2048x7x7xf32>
    %v2090 = stablehlo.subtract %v2089, %v2085 : tensor<64x2048x7x7xf32>
    %v2091 = stablehlo.multiply %v2080, %v2088 : tensor<64x2048x7x7xf32>
    %v2092 = stablehlo.subtract %v2090, %v2091 : tensor<64x2048x7x7xf32>
    %v2093 = stablehlo.divide %v2079, %v2068 : tensor<64x2048x7x7xf32>
    %v2094 = stablehlo.multiply %v2093, %v2092 : tensor<64x2048x7x7xf32>
    %v2095 = stablehlo.reshape %v2094 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2096 = stablehlo.reshape %v2095 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2097 = stablehlo.reverse %s4b0W3, dims = [2, 3] : tensor<2048x512x1x1xf32>
    %v2098 = stablehlo.transpose %v2097, dims = [1, 0, 2, 3] : (tensor<2048x512x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %v2099 = stablehlo.convert %v2096 : (tensor<64x2048x7x7xf32>) -> tensor<64x2048x7x7xbf16>
    %v2100 = stablehlo.convert %v2098 : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xbf16>
    %v2101 = stablehlo.convolution(%v2099, %v2100)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x7x7xbf16>, tensor<512x2048x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v2102 = stablehlo.convert %v2101 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2104 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v2105 = stablehlo.compare GT, %v1358, %v2104 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v2106 = stablehlo.select %v2105, %v2103, %v2104 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v2107 = stablehlo.reshape %v1338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2108 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2109 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2110 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2111 = stablehlo.reduce(%v2107 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2112 = stablehlo.broadcast_in_dim %v2111, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2113 = stablehlo.divide %v2112, %v2109 : tensor<64x512x7x7xf32>
    %v2114 = stablehlo.subtract %v2107, %v2113 : tensor<64x512x7x7xf32>
    %v2115 = stablehlo.multiply %v2114, %v2114 : tensor<64x512x7x7xf32>
    %v2116 = stablehlo.reduce(%v2115 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2117 = stablehlo.broadcast_in_dim %v2116, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2118 = stablehlo.divide %v2117, %v2109 : tensor<64x512x7x7xf32>
    %v2119 = stablehlo.add %v2118, %v2110 : tensor<64x512x7x7xf32>
    %v2120 = stablehlo.rsqrt %v2119 : tensor<64x512x7x7xf32>
    %v2121 = stablehlo.multiply %v2114, %v2120 : tensor<64x512x7x7xf32>
    %v2122 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2123 = stablehlo.reshape %v2106 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2124 = stablehlo.multiply %v2122, %v2123 : tensor<64x512x7x7xf32>
    %v2125 = stablehlo.reduce(%v2124 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2126 = stablehlo.broadcast_in_dim %v2125, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2127 = stablehlo.multiply %v2121, %v2124 : tensor<64x512x7x7xf32>
    %v2128 = stablehlo.reduce(%v2127 init: %v2108) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2129 = stablehlo.broadcast_in_dim %v2128, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2130 = stablehlo.multiply %v2124, %v2109 : tensor<64x512x7x7xf32>
    %v2131 = stablehlo.subtract %v2130, %v2126 : tensor<64x512x7x7xf32>
    %v2132 = stablehlo.multiply %v2121, %v2129 : tensor<64x512x7x7xf32>
    %v2133 = stablehlo.subtract %v2131, %v2132 : tensor<64x512x7x7xf32>
    %v2134 = stablehlo.divide %v2120, %v2109 : tensor<64x512x7x7xf32>
    %v2135 = stablehlo.multiply %v2134, %v2133 : tensor<64x512x7x7xf32>
    %v2136 = stablehlo.reshape %v2135 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2137 = stablehlo.reshape %v2136 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2138 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2139 = stablehlo.pad %v2137, %v2138, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2140 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2141 = stablehlo.transpose %v2140, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2142 = stablehlo.convert %v2139 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2143 = stablehlo.convert %v2141 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2144 = stablehlo.convolution(%v2142, %v2143)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x14x14xbf16>
    %v2145 = stablehlo.convert %v2144 : (tensor<64x512x14x14xbf16>) -> tensor<64x512x14x14xf32>
    %v2146 = stablehlo.reshape %v2145 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2147 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2148 = stablehlo.compare GT, %v1328, %v2147 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2149 = stablehlo.select %v2148, %v2146, %v2147 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2150 = stablehlo.reshape %v1308 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2152 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v2153 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2154 = stablehlo.reduce(%v2150 init: %v2151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2155 = stablehlo.broadcast_in_dim %v2154, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2156 = stablehlo.divide %v2155, %v2152 : tensor<64x512x14x14xf32>
    %v2157 = stablehlo.subtract %v2150, %v2156 : tensor<64x512x14x14xf32>
    %v2158 = stablehlo.multiply %v2157, %v2157 : tensor<64x512x14x14xf32>
    %v2159 = stablehlo.reduce(%v2158 init: %v2151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2160 = stablehlo.broadcast_in_dim %v2159, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2161 = stablehlo.divide %v2160, %v2152 : tensor<64x512x14x14xf32>
    %v2162 = stablehlo.add %v2161, %v2153 : tensor<64x512x14x14xf32>
    %v2163 = stablehlo.rsqrt %v2162 : tensor<64x512x14x14xf32>
    %v2164 = stablehlo.multiply %v2157, %v2163 : tensor<64x512x14x14xf32>
    %v2165 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2166 = stablehlo.reshape %v2149 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2167 = stablehlo.multiply %v2165, %v2166 : tensor<64x512x14x14xf32>
    %v2168 = stablehlo.reduce(%v2167 init: %v2151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2169 = stablehlo.broadcast_in_dim %v2168, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2170 = stablehlo.multiply %v2164, %v2167 : tensor<64x512x14x14xf32>
    %v2171 = stablehlo.reduce(%v2170 init: %v2151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2172 = stablehlo.broadcast_in_dim %v2171, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2173 = stablehlo.multiply %v2167, %v2152 : tensor<64x512x14x14xf32>
    %v2174 = stablehlo.subtract %v2173, %v2169 : tensor<64x512x14x14xf32>
    %v2175 = stablehlo.multiply %v2164, %v2172 : tensor<64x512x14x14xf32>
    %v2176 = stablehlo.subtract %v2174, %v2175 : tensor<64x512x14x14xf32>
    %v2177 = stablehlo.divide %v2163, %v2152 : tensor<64x512x14x14xf32>
    %v2178 = stablehlo.multiply %v2177, %v2176 : tensor<64x512x14x14xf32>
    %v2179 = stablehlo.reshape %v2178 : (tensor<64x512x14x14xf32>) -> tensor<64x100352xf32>
    %v2180 = stablehlo.reshape %v2179 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2181 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x1024x1x1xf32>
    %v2182 = stablehlo.transpose %v2181, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v2183 = stablehlo.convert %v2180 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2184 = stablehlo.convert %v2182 : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xbf16>
    %v2185 = stablehlo.convolution(%v2183, %v2184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<1024x512x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2186 = stablehlo.convert %v2185 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2187 = stablehlo.reshape %v2186 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2188 = stablehlo.reshape %v1396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2190 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2191 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2192 = stablehlo.reduce(%v2188 init: %v2189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2193 = stablehlo.broadcast_in_dim %v2192, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2194 = stablehlo.divide %v2193, %v2190 : tensor<64x2048x7x7xf32>
    %v2195 = stablehlo.subtract %v2188, %v2194 : tensor<64x2048x7x7xf32>
    %v2196 = stablehlo.multiply %v2195, %v2195 : tensor<64x2048x7x7xf32>
    %v2197 = stablehlo.reduce(%v2196 init: %v2189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2198 = stablehlo.broadcast_in_dim %v2197, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2199 = stablehlo.divide %v2198, %v2190 : tensor<64x2048x7x7xf32>
    %v2200 = stablehlo.add %v2199, %v2191 : tensor<64x2048x7x7xf32>
    %v2201 = stablehlo.rsqrt %v2200 : tensor<64x2048x7x7xf32>
    %v2202 = stablehlo.multiply %v2195, %v2201 : tensor<64x2048x7x7xf32>
    %v2203 = stablehlo.broadcast_in_dim %s4b0gp, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2204 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2205 = stablehlo.multiply %v2203, %v2204 : tensor<64x2048x7x7xf32>
    %v2206 = stablehlo.reduce(%v2205 init: %v2189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2207 = stablehlo.broadcast_in_dim %v2206, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2208 = stablehlo.multiply %v2202, %v2205 : tensor<64x2048x7x7xf32>
    %v2209 = stablehlo.reduce(%v2208 init: %v2189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2210 = stablehlo.broadcast_in_dim %v2209, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2211 = stablehlo.multiply %v2205, %v2190 : tensor<64x2048x7x7xf32>
    %v2212 = stablehlo.subtract %v2211, %v2207 : tensor<64x2048x7x7xf32>
    %v2213 = stablehlo.multiply %v2202, %v2210 : tensor<64x2048x7x7xf32>
    %v2214 = stablehlo.subtract %v2212, %v2213 : tensor<64x2048x7x7xf32>
    %v2215 = stablehlo.divide %v2201, %v2190 : tensor<64x2048x7x7xf32>
    %v2216 = stablehlo.multiply %v2215, %v2214 : tensor<64x2048x7x7xf32>
    %v2217 = stablehlo.reshape %v2216 : (tensor<64x2048x7x7xf32>) -> tensor<64x100352xf32>
    %v2218 = stablehlo.reshape %v2217 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2219 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2220 = stablehlo.pad %v2218, %v2219, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v2221 = stablehlo.reverse %s4b0Wp, dims = [2, 3] : tensor<2048x1024x1x1xf32>
    %v2222 = stablehlo.transpose %v2221, dims = [1, 0, 2, 3] : (tensor<2048x1024x1x1xf32>) -> tensor<1024x2048x1x1xf32>
    %v2223 = stablehlo.convert %v2220 : (tensor<64x2048x14x14xf32>) -> tensor<64x2048x14x14xbf16>
    %v2224 = stablehlo.convert %v2222 : (tensor<1024x2048x1x1xf32>) -> tensor<1024x2048x1x1xbf16>
    %v2225 = stablehlo.convolution(%v2223, %v2224)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x2048x14x14xbf16>, tensor<1024x2048x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2226 = stablehlo.convert %v2225 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2227 = stablehlo.reshape %v2226 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2228 = stablehlo.add %v2187, %v2227 : tensor<64x200704xf32>
    %v2229 = stablehlo.reshape %v1300 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2230 = stablehlo.reshape %v2179 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2231 = stablehlo.transpose %v2229, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2232 = stablehlo.transpose %v2230, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2233 = stablehlo.convert %v2231 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2234 = stablehlo.convert %v2232 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2235 = stablehlo.convolution(%v2233, %v2234)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<1024x512x1x1xbf16>
    %v2236 = stablehlo.convert %v2235 : (tensor<1024x512x1x1xbf16>) -> tensor<1024x512x1x1xf32>
    %v2237 = stablehlo.transpose %v2236, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v2238 = stablehlo.reshape %v1308 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2240 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v2241 = stablehlo.reduce(%v2238 init: %v2239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2242 = stablehlo.broadcast_in_dim %v2241, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2243 = stablehlo.divide %v2242, %v2240 : tensor<64x512x14x14xf32>
    %v2244 = stablehlo.subtract %v2238, %v2243 : tensor<64x512x14x14xf32>
    %v2245 = stablehlo.multiply %v2244, %v2244 : tensor<64x512x14x14xf32>
    %v2246 = stablehlo.reduce(%v2245 init: %v2239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2247 = stablehlo.broadcast_in_dim %v2246, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v2248 = stablehlo.divide %v2247, %v2240 : tensor<64x512x14x14xf32>
    %v2249 = stablehlo.constant dense<1.0e-05> : tensor<64x512x14x14xf32>
    %v2250 = stablehlo.add %v2248, %v2249 : tensor<64x512x14x14xf32>
    %v2251 = stablehlo.rsqrt %v2250 : tensor<64x512x14x14xf32>
    %v2252 = stablehlo.multiply %v2244, %v2251 : tensor<64x512x14x14xf32>
    %v2253 = stablehlo.reshape %v2149 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2254 = stablehlo.multiply %v2253, %v2252 : tensor<64x512x14x14xf32>
    %v2255 = stablehlo.reduce(%v2254 init: %v2239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2256 = stablehlo.reshape %v2149 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2257 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2258 = stablehlo.reduce(%v2256 init: %v2257) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v2259 = stablehlo.reshape %v1330 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v2260 = stablehlo.reshape %v2136 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2261 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2262 = stablehlo.pad %v2260, %v2261, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2263 = stablehlo.transpose %v2259, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2264 = stablehlo.transpose %v2262, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2265 = stablehlo.convert %v2263 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2266 = stablehlo.convert %v2264 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2267 = stablehlo.convolution(%v2265, %v2266)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<512x512x3x3xbf16>
    %v2268 = stablehlo.convert %v2267 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2269 = stablehlo.transpose %v2268, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2270 = stablehlo.reshape %v1338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2271 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2272 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v2273 = stablehlo.reduce(%v2270 init: %v2271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2274 = stablehlo.broadcast_in_dim %v2273, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2275 = stablehlo.divide %v2274, %v2272 : tensor<64x512x7x7xf32>
    %v2276 = stablehlo.subtract %v2270, %v2275 : tensor<64x512x7x7xf32>
    %v2277 = stablehlo.multiply %v2276, %v2276 : tensor<64x512x7x7xf32>
    %v2278 = stablehlo.reduce(%v2277 init: %v2271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2279 = stablehlo.broadcast_in_dim %v2278, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2280 = stablehlo.divide %v2279, %v2272 : tensor<64x512x7x7xf32>
    %v2281 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2282 = stablehlo.add %v2280, %v2281 : tensor<64x512x7x7xf32>
    %v2283 = stablehlo.rsqrt %v2282 : tensor<64x512x7x7xf32>
    %v2284 = stablehlo.multiply %v2276, %v2283 : tensor<64x512x7x7xf32>
    %v2285 = stablehlo.reshape %v2106 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2286 = stablehlo.multiply %v2285, %v2284 : tensor<64x512x7x7xf32>
    %v2287 = stablehlo.reduce(%v2286 init: %v2271) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2288 = stablehlo.reshape %v2106 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2290 = stablehlo.reduce(%v2288 init: %v2289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2291 = stablehlo.reshape %v1360 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2292 = stablehlo.reshape %v2095 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2293 = stablehlo.transpose %v2291, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2294 = stablehlo.transpose %v2292, dims = [1, 0, 2, 3] : (tensor<64x2048x7x7xf32>) -> tensor<2048x64x7x7xf32>
    %v2295 = stablehlo.convert %v2293 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2296 = stablehlo.convert %v2294 : (tensor<2048x64x7x7xf32>) -> tensor<2048x64x7x7xbf16>
    %v2297 = stablehlo.convolution(%v2295, %v2296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<2048x64x7x7xbf16>) -> tensor<512x2048x1x1xbf16>
    %v2298 = stablehlo.convert %v2297 : (tensor<512x2048x1x1xbf16>) -> tensor<512x2048x1x1xf32>
    %v2299 = stablehlo.transpose %v2298, dims = [1, 0, 2, 3] : (tensor<512x2048x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %v2300 = stablehlo.reshape %v1368 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2301 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2302 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2303 = stablehlo.reduce(%v2300 init: %v2301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2304 = stablehlo.broadcast_in_dim %v2303, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2305 = stablehlo.divide %v2304, %v2302 : tensor<64x2048x7x7xf32>
    %v2306 = stablehlo.subtract %v2300, %v2305 : tensor<64x2048x7x7xf32>
    %v2307 = stablehlo.multiply %v2306, %v2306 : tensor<64x2048x7x7xf32>
    %v2308 = stablehlo.reduce(%v2307 init: %v2301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2309 = stablehlo.broadcast_in_dim %v2308, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2310 = stablehlo.divide %v2309, %v2302 : tensor<64x2048x7x7xf32>
    %v2311 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2312 = stablehlo.add %v2310, %v2311 : tensor<64x2048x7x7xf32>
    %v2313 = stablehlo.rsqrt %v2312 : tensor<64x2048x7x7xf32>
    %v2314 = stablehlo.multiply %v2306, %v2313 : tensor<64x2048x7x7xf32>
    %v2315 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2316 = stablehlo.multiply %v2315, %v2314 : tensor<64x2048x7x7xf32>
    %v2317 = stablehlo.reduce(%v2316 init: %v2301) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2318 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2319 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2320 = stablehlo.reduce(%v2318 init: %v2319) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2321 = stablehlo.reshape %v1300 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2322 = stablehlo.reshape %v2217 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2324 = stablehlo.pad %v2322, %v2323, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<64x2048x14x14xf32>
    %v2325 = stablehlo.transpose %v2321, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2326 = stablehlo.transpose %v2324, dims = [1, 0, 2, 3] : (tensor<64x2048x14x14xf32>) -> tensor<2048x64x14x14xf32>
    %v2327 = stablehlo.convert %v2325 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2328 = stablehlo.convert %v2326 : (tensor<2048x64x14x14xf32>) -> tensor<2048x64x14x14xbf16>
    %v2329 = stablehlo.convolution(%v2327, %v2328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<2048x64x14x14xbf16>) -> tensor<1024x2048x1x1xbf16>
    %v2330 = stablehlo.convert %v2329 : (tensor<1024x2048x1x1xbf16>) -> tensor<1024x2048x1x1xf32>
    %v2331 = stablehlo.transpose %v2330, dims = [1, 0, 2, 3] : (tensor<1024x2048x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %v2332 = stablehlo.reshape %v1396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2333 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2334 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v2335 = stablehlo.reduce(%v2332 init: %v2333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2336 = stablehlo.broadcast_in_dim %v2335, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2337 = stablehlo.divide %v2336, %v2334 : tensor<64x2048x7x7xf32>
    %v2338 = stablehlo.subtract %v2332, %v2337 : tensor<64x2048x7x7xf32>
    %v2339 = stablehlo.multiply %v2338, %v2338 : tensor<64x2048x7x7xf32>
    %v2340 = stablehlo.reduce(%v2339 init: %v2333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2341 = stablehlo.broadcast_in_dim %v2340, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v2342 = stablehlo.divide %v2341, %v2334 : tensor<64x2048x7x7xf32>
    %v2343 = stablehlo.constant dense<1.0e-05> : tensor<64x2048x7x7xf32>
    %v2344 = stablehlo.add %v2342, %v2343 : tensor<64x2048x7x7xf32>
    %v2345 = stablehlo.rsqrt %v2344 : tensor<64x2048x7x7xf32>
    %v2346 = stablehlo.multiply %v2338, %v2345 : tensor<64x2048x7x7xf32>
    %v2347 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2348 = stablehlo.multiply %v2347, %v2346 : tensor<64x2048x7x7xf32>
    %v2349 = stablehlo.reduce(%v2348 init: %v2333) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2350 = stablehlo.reshape %v2065 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v2351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2352 = stablehlo.reduce(%v2350 init: %v2351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v2353 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v2354 = stablehlo.compare GT, %v1298, %v2353 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v2355 = stablehlo.select %v2354, %v2228, %v2353 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v2356 = stablehlo.reshape %v1277 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2358 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2359 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2360 = stablehlo.reduce(%v2356 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2361 = stablehlo.broadcast_in_dim %v2360, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2362 = stablehlo.divide %v2361, %v2358 : tensor<64x1024x14x14xf32>
    %v2363 = stablehlo.subtract %v2356, %v2362 : tensor<64x1024x14x14xf32>
    %v2364 = stablehlo.multiply %v2363, %v2363 : tensor<64x1024x14x14xf32>
    %v2365 = stablehlo.reduce(%v2364 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2365, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2367 = stablehlo.divide %v2366, %v2358 : tensor<64x1024x14x14xf32>
    %v2368 = stablehlo.add %v2367, %v2359 : tensor<64x1024x14x14xf32>
    %v2369 = stablehlo.rsqrt %v2368 : tensor<64x1024x14x14xf32>
    %v2370 = stablehlo.multiply %v2363, %v2369 : tensor<64x1024x14x14xf32>
    %v2371 = stablehlo.broadcast_in_dim %s3b5g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2372 = stablehlo.reshape %v2355 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2373 = stablehlo.multiply %v2371, %v2372 : tensor<64x1024x14x14xf32>
    %v2374 = stablehlo.reduce(%v2373 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2375 = stablehlo.broadcast_in_dim %v2374, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2376 = stablehlo.multiply %v2370, %v2373 : tensor<64x1024x14x14xf32>
    %v2377 = stablehlo.reduce(%v2376 init: %v2357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2378 = stablehlo.broadcast_in_dim %v2377, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2379 = stablehlo.multiply %v2373, %v2358 : tensor<64x1024x14x14xf32>
    %v2380 = stablehlo.subtract %v2379, %v2375 : tensor<64x1024x14x14xf32>
    %v2381 = stablehlo.multiply %v2370, %v2378 : tensor<64x1024x14x14xf32>
    %v2382 = stablehlo.subtract %v2380, %v2381 : tensor<64x1024x14x14xf32>
    %v2383 = stablehlo.divide %v2369, %v2358 : tensor<64x1024x14x14xf32>
    %v2384 = stablehlo.multiply %v2383, %v2382 : tensor<64x1024x14x14xf32>
    %v2385 = stablehlo.reshape %v2384 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2386 = stablehlo.reshape %v2385 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2387 = stablehlo.reverse %s3b5W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2388 = stablehlo.transpose %v2387, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2389 = stablehlo.convert %v2386 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2390 = stablehlo.convert %v2388 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v2391 = stablehlo.convolution(%v2389, %v2390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2392 = stablehlo.convert %v2391 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2393 = stablehlo.reshape %v2392 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2394 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2395 = stablehlo.compare GT, %v1267, %v2394 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2396 = stablehlo.select %v2395, %v2393, %v2394 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2397 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2398 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2399 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2400 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2401 = stablehlo.reduce(%v2397 init: %v2398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2402 = stablehlo.broadcast_in_dim %v2401, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2403 = stablehlo.divide %v2402, %v2399 : tensor<64x256x14x14xf32>
    %v2404 = stablehlo.subtract %v2397, %v2403 : tensor<64x256x14x14xf32>
    %v2405 = stablehlo.multiply %v2404, %v2404 : tensor<64x256x14x14xf32>
    %v2406 = stablehlo.reduce(%v2405 init: %v2398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2407 = stablehlo.broadcast_in_dim %v2406, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2408 = stablehlo.divide %v2407, %v2399 : tensor<64x256x14x14xf32>
    %v2409 = stablehlo.add %v2408, %v2400 : tensor<64x256x14x14xf32>
    %v2410 = stablehlo.rsqrt %v2409 : tensor<64x256x14x14xf32>
    %v2411 = stablehlo.multiply %v2404, %v2410 : tensor<64x256x14x14xf32>
    %v2412 = stablehlo.broadcast_in_dim %s3b5g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2413 = stablehlo.reshape %v2396 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2414 = stablehlo.multiply %v2412, %v2413 : tensor<64x256x14x14xf32>
    %v2415 = stablehlo.reduce(%v2414 init: %v2398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2416 = stablehlo.broadcast_in_dim %v2415, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2417 = stablehlo.multiply %v2411, %v2414 : tensor<64x256x14x14xf32>
    %v2418 = stablehlo.reduce(%v2417 init: %v2398) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2419 = stablehlo.broadcast_in_dim %v2418, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2420 = stablehlo.multiply %v2414, %v2399 : tensor<64x256x14x14xf32>
    %v2421 = stablehlo.subtract %v2420, %v2416 : tensor<64x256x14x14xf32>
    %v2422 = stablehlo.multiply %v2411, %v2419 : tensor<64x256x14x14xf32>
    %v2423 = stablehlo.subtract %v2421, %v2422 : tensor<64x256x14x14xf32>
    %v2424 = stablehlo.divide %v2410, %v2399 : tensor<64x256x14x14xf32>
    %v2425 = stablehlo.multiply %v2424, %v2423 : tensor<64x256x14x14xf32>
    %v2426 = stablehlo.reshape %v2425 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2427 = stablehlo.reshape %v2426 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2428 = stablehlo.reverse %s3b5W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2429 = stablehlo.transpose %v2428, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2430 = stablehlo.convert %v2427 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2431 = stablehlo.convert %v2429 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2432 = stablehlo.convolution(%v2430, %v2431)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2433 = stablehlo.convert %v2432 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2434 = stablehlo.reshape %v2433 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2435 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2436 = stablehlo.compare GT, %v1237, %v2435 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2437 = stablehlo.select %v2436, %v2434, %v2435 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2438 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2440 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2441 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2442 = stablehlo.reduce(%v2438 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2443 = stablehlo.broadcast_in_dim %v2442, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2444 = stablehlo.divide %v2443, %v2440 : tensor<64x256x14x14xf32>
    %v2445 = stablehlo.subtract %v2438, %v2444 : tensor<64x256x14x14xf32>
    %v2446 = stablehlo.multiply %v2445, %v2445 : tensor<64x256x14x14xf32>
    %v2447 = stablehlo.reduce(%v2446 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2448 = stablehlo.broadcast_in_dim %v2447, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2449 = stablehlo.divide %v2448, %v2440 : tensor<64x256x14x14xf32>
    %v2450 = stablehlo.add %v2449, %v2441 : tensor<64x256x14x14xf32>
    %v2451 = stablehlo.rsqrt %v2450 : tensor<64x256x14x14xf32>
    %v2452 = stablehlo.multiply %v2445, %v2451 : tensor<64x256x14x14xf32>
    %v2453 = stablehlo.broadcast_in_dim %s3b5g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2454 = stablehlo.reshape %v2437 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2455 = stablehlo.multiply %v2453, %v2454 : tensor<64x256x14x14xf32>
    %v2456 = stablehlo.reduce(%v2455 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2456, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2458 = stablehlo.multiply %v2452, %v2455 : tensor<64x256x14x14xf32>
    %v2459 = stablehlo.reduce(%v2458 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2460 = stablehlo.broadcast_in_dim %v2459, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2461 = stablehlo.multiply %v2455, %v2440 : tensor<64x256x14x14xf32>
    %v2462 = stablehlo.subtract %v2461, %v2457 : tensor<64x256x14x14xf32>
    %v2463 = stablehlo.multiply %v2452, %v2460 : tensor<64x256x14x14xf32>
    %v2464 = stablehlo.subtract %v2462, %v2463 : tensor<64x256x14x14xf32>
    %v2465 = stablehlo.divide %v2451, %v2440 : tensor<64x256x14x14xf32>
    %v2466 = stablehlo.multiply %v2465, %v2464 : tensor<64x256x14x14xf32>
    %v2467 = stablehlo.reshape %v2466 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2468 = stablehlo.reshape %v2467 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2469 = stablehlo.reverse %s3b5W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2470 = stablehlo.transpose %v2469, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2471 = stablehlo.convert %v2468 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2472 = stablehlo.convert %v2470 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v2473 = stablehlo.convolution(%v2471, %v2472)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2474 = stablehlo.convert %v2473 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2476 = stablehlo.add %v2475, %v2355 : tensor<64x200704xf32>
    %v2477 = stablehlo.reshape %v1209 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2478 = stablehlo.reshape %v2467 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2479 = stablehlo.transpose %v2477, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2480 = stablehlo.transpose %v2478, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2481 = stablehlo.convert %v2479 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2482 = stablehlo.convert %v2480 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2483 = stablehlo.convolution(%v2481, %v2482)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v2484 = stablehlo.convert %v2483 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v2485 = stablehlo.transpose %v2484, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2486 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2487 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2488 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2489 = stablehlo.reduce(%v2486 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2490 = stablehlo.broadcast_in_dim %v2489, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2491 = stablehlo.divide %v2490, %v2488 : tensor<64x256x14x14xf32>
    %v2492 = stablehlo.subtract %v2486, %v2491 : tensor<64x256x14x14xf32>
    %v2493 = stablehlo.multiply %v2492, %v2492 : tensor<64x256x14x14xf32>
    %v2494 = stablehlo.reduce(%v2493 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2495 = stablehlo.broadcast_in_dim %v2494, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2496 = stablehlo.divide %v2495, %v2488 : tensor<64x256x14x14xf32>
    %v2497 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2498 = stablehlo.add %v2496, %v2497 : tensor<64x256x14x14xf32>
    %v2499 = stablehlo.rsqrt %v2498 : tensor<64x256x14x14xf32>
    %v2500 = stablehlo.multiply %v2492, %v2499 : tensor<64x256x14x14xf32>
    %v2501 = stablehlo.reshape %v2437 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2502 = stablehlo.multiply %v2501, %v2500 : tensor<64x256x14x14xf32>
    %v2503 = stablehlo.reduce(%v2502 init: %v2487) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.reshape %v2437 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2505 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2506 = stablehlo.reduce(%v2504 init: %v2505) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2507 = stablehlo.reshape %v1239 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2508 = stablehlo.reshape %v2426 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2509 = stablehlo.transpose %v2507, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2510 = stablehlo.transpose %v2508, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2511 = stablehlo.convert %v2509 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2512 = stablehlo.convert %v2510 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2513 = stablehlo.convolution(%v2511, %v2512)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2514 = stablehlo.convert %v2513 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2515 = stablehlo.transpose %v2514, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2516 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2517 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2518 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2519 = stablehlo.reduce(%v2516 init: %v2517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2520 = stablehlo.broadcast_in_dim %v2519, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2521 = stablehlo.divide %v2520, %v2518 : tensor<64x256x14x14xf32>
    %v2522 = stablehlo.subtract %v2516, %v2521 : tensor<64x256x14x14xf32>
    %v2523 = stablehlo.multiply %v2522, %v2522 : tensor<64x256x14x14xf32>
    %v2524 = stablehlo.reduce(%v2523 init: %v2517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2525 = stablehlo.broadcast_in_dim %v2524, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2526 = stablehlo.divide %v2525, %v2518 : tensor<64x256x14x14xf32>
    %v2527 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2528 = stablehlo.add %v2526, %v2527 : tensor<64x256x14x14xf32>
    %v2529 = stablehlo.rsqrt %v2528 : tensor<64x256x14x14xf32>
    %v2530 = stablehlo.multiply %v2522, %v2529 : tensor<64x256x14x14xf32>
    %v2531 = stablehlo.reshape %v2396 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2532 = stablehlo.multiply %v2531, %v2530 : tensor<64x256x14x14xf32>
    %v2533 = stablehlo.reduce(%v2532 init: %v2517) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2534 = stablehlo.reshape %v2396 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2535 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2536 = stablehlo.reduce(%v2534 init: %v2535) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2537 = stablehlo.reshape %v1269 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2538 = stablehlo.reshape %v2385 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2539 = stablehlo.transpose %v2537, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2540 = stablehlo.transpose %v2538, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2541 = stablehlo.convert %v2539 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2542 = stablehlo.convert %v2540 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2543 = stablehlo.convolution(%v2541, %v2542)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v2544 = stablehlo.convert %v2543 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v2545 = stablehlo.transpose %v2544, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2546 = stablehlo.reshape %v1277 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2548 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2549 = stablehlo.reduce(%v2546 init: %v2547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2550 = stablehlo.broadcast_in_dim %v2549, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2551 = stablehlo.divide %v2550, %v2548 : tensor<64x1024x14x14xf32>
    %v2552 = stablehlo.subtract %v2546, %v2551 : tensor<64x1024x14x14xf32>
    %v2553 = stablehlo.multiply %v2552, %v2552 : tensor<64x1024x14x14xf32>
    %v2554 = stablehlo.reduce(%v2553 init: %v2547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2555 = stablehlo.broadcast_in_dim %v2554, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2556 = stablehlo.divide %v2555, %v2548 : tensor<64x1024x14x14xf32>
    %v2557 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2558 = stablehlo.add %v2556, %v2557 : tensor<64x1024x14x14xf32>
    %v2559 = stablehlo.rsqrt %v2558 : tensor<64x1024x14x14xf32>
    %v2560 = stablehlo.multiply %v2552, %v2559 : tensor<64x1024x14x14xf32>
    %v2561 = stablehlo.reshape %v2355 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2562 = stablehlo.multiply %v2561, %v2560 : tensor<64x1024x14x14xf32>
    %v2563 = stablehlo.reduce(%v2562 init: %v2547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2564 = stablehlo.reshape %v2355 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2565 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2566 = stablehlo.reduce(%v2564 init: %v2565) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2567 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v2568 = stablehlo.compare GT, %v1207, %v2567 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v2569 = stablehlo.select %v2568, %v2476, %v2567 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v2570 = stablehlo.reshape %v1186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2571 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2572 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2573 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2574 = stablehlo.reduce(%v2570 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2575 = stablehlo.broadcast_in_dim %v2574, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2576 = stablehlo.divide %v2575, %v2572 : tensor<64x1024x14x14xf32>
    %v2577 = stablehlo.subtract %v2570, %v2576 : tensor<64x1024x14x14xf32>
    %v2578 = stablehlo.multiply %v2577, %v2577 : tensor<64x1024x14x14xf32>
    %v2579 = stablehlo.reduce(%v2578 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2580 = stablehlo.broadcast_in_dim %v2579, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2581 = stablehlo.divide %v2580, %v2572 : tensor<64x1024x14x14xf32>
    %v2582 = stablehlo.add %v2581, %v2573 : tensor<64x1024x14x14xf32>
    %v2583 = stablehlo.rsqrt %v2582 : tensor<64x1024x14x14xf32>
    %v2584 = stablehlo.multiply %v2577, %v2583 : tensor<64x1024x14x14xf32>
    %v2585 = stablehlo.broadcast_in_dim %s3b4g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2586 = stablehlo.reshape %v2569 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2587 = stablehlo.multiply %v2585, %v2586 : tensor<64x1024x14x14xf32>
    %v2588 = stablehlo.reduce(%v2587 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2589 = stablehlo.broadcast_in_dim %v2588, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2590 = stablehlo.multiply %v2584, %v2587 : tensor<64x1024x14x14xf32>
    %v2591 = stablehlo.reduce(%v2590 init: %v2571) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2592 = stablehlo.broadcast_in_dim %v2591, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2593 = stablehlo.multiply %v2587, %v2572 : tensor<64x1024x14x14xf32>
    %v2594 = stablehlo.subtract %v2593, %v2589 : tensor<64x1024x14x14xf32>
    %v2595 = stablehlo.multiply %v2584, %v2592 : tensor<64x1024x14x14xf32>
    %v2596 = stablehlo.subtract %v2594, %v2595 : tensor<64x1024x14x14xf32>
    %v2597 = stablehlo.divide %v2583, %v2572 : tensor<64x1024x14x14xf32>
    %v2598 = stablehlo.multiply %v2597, %v2596 : tensor<64x1024x14x14xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2600 = stablehlo.reshape %v2599 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2601 = stablehlo.reverse %s3b4W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2602 = stablehlo.transpose %v2601, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2603 = stablehlo.convert %v2600 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2604 = stablehlo.convert %v2602 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v2605 = stablehlo.convolution(%v2603, %v2604)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2606 = stablehlo.convert %v2605 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2607 = stablehlo.reshape %v2606 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2608 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2609 = stablehlo.compare GT, %v1176, %v2608 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2610 = stablehlo.select %v2609, %v2607, %v2608 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2611 = stablehlo.reshape %v1156 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2613 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2614 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2615 = stablehlo.reduce(%v2611 init: %v2612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2616 = stablehlo.broadcast_in_dim %v2615, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2617 = stablehlo.divide %v2616, %v2613 : tensor<64x256x14x14xf32>
    %v2618 = stablehlo.subtract %v2611, %v2617 : tensor<64x256x14x14xf32>
    %v2619 = stablehlo.multiply %v2618, %v2618 : tensor<64x256x14x14xf32>
    %v2620 = stablehlo.reduce(%v2619 init: %v2612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2621 = stablehlo.broadcast_in_dim %v2620, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2622 = stablehlo.divide %v2621, %v2613 : tensor<64x256x14x14xf32>
    %v2623 = stablehlo.add %v2622, %v2614 : tensor<64x256x14x14xf32>
    %v2624 = stablehlo.rsqrt %v2623 : tensor<64x256x14x14xf32>
    %v2625 = stablehlo.multiply %v2618, %v2624 : tensor<64x256x14x14xf32>
    %v2626 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2627 = stablehlo.reshape %v2610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2628 = stablehlo.multiply %v2626, %v2627 : tensor<64x256x14x14xf32>
    %v2629 = stablehlo.reduce(%v2628 init: %v2612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2630 = stablehlo.broadcast_in_dim %v2629, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2631 = stablehlo.multiply %v2625, %v2628 : tensor<64x256x14x14xf32>
    %v2632 = stablehlo.reduce(%v2631 init: %v2612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2633 = stablehlo.broadcast_in_dim %v2632, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2634 = stablehlo.multiply %v2628, %v2613 : tensor<64x256x14x14xf32>
    %v2635 = stablehlo.subtract %v2634, %v2630 : tensor<64x256x14x14xf32>
    %v2636 = stablehlo.multiply %v2625, %v2633 : tensor<64x256x14x14xf32>
    %v2637 = stablehlo.subtract %v2635, %v2636 : tensor<64x256x14x14xf32>
    %v2638 = stablehlo.divide %v2624, %v2613 : tensor<64x256x14x14xf32>
    %v2639 = stablehlo.multiply %v2638, %v2637 : tensor<64x256x14x14xf32>
    %v2640 = stablehlo.reshape %v2639 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2641 = stablehlo.reshape %v2640 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2642 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2643 = stablehlo.transpose %v2642, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2644 = stablehlo.convert %v2641 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2645 = stablehlo.convert %v2643 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2646 = stablehlo.convolution(%v2644, %v2645)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2647 = stablehlo.convert %v2646 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2648 = stablehlo.reshape %v2647 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2649 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2650 = stablehlo.compare GT, %v1146, %v2649 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2651 = stablehlo.select %v2650, %v2648, %v2649 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2652 = stablehlo.reshape %v1126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2653 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2654 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2655 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2656 = stablehlo.reduce(%v2652 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2657 = stablehlo.broadcast_in_dim %v2656, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2658 = stablehlo.divide %v2657, %v2654 : tensor<64x256x14x14xf32>
    %v2659 = stablehlo.subtract %v2652, %v2658 : tensor<64x256x14x14xf32>
    %v2660 = stablehlo.multiply %v2659, %v2659 : tensor<64x256x14x14xf32>
    %v2661 = stablehlo.reduce(%v2660 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2662 = stablehlo.broadcast_in_dim %v2661, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2663 = stablehlo.divide %v2662, %v2654 : tensor<64x256x14x14xf32>
    %v2664 = stablehlo.add %v2663, %v2655 : tensor<64x256x14x14xf32>
    %v2665 = stablehlo.rsqrt %v2664 : tensor<64x256x14x14xf32>
    %v2666 = stablehlo.multiply %v2659, %v2665 : tensor<64x256x14x14xf32>
    %v2667 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2668 = stablehlo.reshape %v2651 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2669 = stablehlo.multiply %v2667, %v2668 : tensor<64x256x14x14xf32>
    %v2670 = stablehlo.reduce(%v2669 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2671 = stablehlo.broadcast_in_dim %v2670, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2672 = stablehlo.multiply %v2666, %v2669 : tensor<64x256x14x14xf32>
    %v2673 = stablehlo.reduce(%v2672 init: %v2653) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2674 = stablehlo.broadcast_in_dim %v2673, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2675 = stablehlo.multiply %v2669, %v2654 : tensor<64x256x14x14xf32>
    %v2676 = stablehlo.subtract %v2675, %v2671 : tensor<64x256x14x14xf32>
    %v2677 = stablehlo.multiply %v2666, %v2674 : tensor<64x256x14x14xf32>
    %v2678 = stablehlo.subtract %v2676, %v2677 : tensor<64x256x14x14xf32>
    %v2679 = stablehlo.divide %v2665, %v2654 : tensor<64x256x14x14xf32>
    %v2680 = stablehlo.multiply %v2679, %v2678 : tensor<64x256x14x14xf32>
    %v2681 = stablehlo.reshape %v2680 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2682 = stablehlo.reshape %v2681 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2683 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2684 = stablehlo.transpose %v2683, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2685 = stablehlo.convert %v2682 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2686 = stablehlo.convert %v2684 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v2687 = stablehlo.convolution(%v2685, %v2686)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2688 = stablehlo.convert %v2687 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2689 = stablehlo.reshape %v2688 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2690 = stablehlo.add %v2689, %v2569 : tensor<64x200704xf32>
    %v2691 = stablehlo.reshape %v1118 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2692 = stablehlo.reshape %v2681 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2693 = stablehlo.transpose %v2691, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2694 = stablehlo.transpose %v2692, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2695 = stablehlo.convert %v2693 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2696 = stablehlo.convert %v2694 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2697 = stablehlo.convolution(%v2695, %v2696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v2698 = stablehlo.convert %v2697 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v2699 = stablehlo.transpose %v2698, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2700 = stablehlo.reshape %v1126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2702 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2703 = stablehlo.reduce(%v2700 init: %v2701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2704 = stablehlo.broadcast_in_dim %v2703, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2705 = stablehlo.divide %v2704, %v2702 : tensor<64x256x14x14xf32>
    %v2706 = stablehlo.subtract %v2700, %v2705 : tensor<64x256x14x14xf32>
    %v2707 = stablehlo.multiply %v2706, %v2706 : tensor<64x256x14x14xf32>
    %v2708 = stablehlo.reduce(%v2707 init: %v2701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2708, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2710 = stablehlo.divide %v2709, %v2702 : tensor<64x256x14x14xf32>
    %v2711 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2712 = stablehlo.add %v2710, %v2711 : tensor<64x256x14x14xf32>
    %v2713 = stablehlo.rsqrt %v2712 : tensor<64x256x14x14xf32>
    %v2714 = stablehlo.multiply %v2706, %v2713 : tensor<64x256x14x14xf32>
    %v2715 = stablehlo.reshape %v2651 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2716 = stablehlo.multiply %v2715, %v2714 : tensor<64x256x14x14xf32>
    %v2717 = stablehlo.reduce(%v2716 init: %v2701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2718 = stablehlo.reshape %v2651 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2720 = stablehlo.reduce(%v2718 init: %v2719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2721 = stablehlo.reshape %v1148 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2722 = stablehlo.reshape %v2640 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2723 = stablehlo.transpose %v2721, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2724 = stablehlo.transpose %v2722, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2725 = stablehlo.convert %v2723 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2726 = stablehlo.convert %v2724 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2727 = stablehlo.convolution(%v2725, %v2726)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2728 = stablehlo.convert %v2727 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2729 = stablehlo.transpose %v2728, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2730 = stablehlo.reshape %v1156 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2732 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2733 = stablehlo.reduce(%v2730 init: %v2731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2733, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2735 = stablehlo.divide %v2734, %v2732 : tensor<64x256x14x14xf32>
    %v2736 = stablehlo.subtract %v2730, %v2735 : tensor<64x256x14x14xf32>
    %v2737 = stablehlo.multiply %v2736, %v2736 : tensor<64x256x14x14xf32>
    %v2738 = stablehlo.reduce(%v2737 init: %v2731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2739 = stablehlo.broadcast_in_dim %v2738, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2740 = stablehlo.divide %v2739, %v2732 : tensor<64x256x14x14xf32>
    %v2741 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2742 = stablehlo.add %v2740, %v2741 : tensor<64x256x14x14xf32>
    %v2743 = stablehlo.rsqrt %v2742 : tensor<64x256x14x14xf32>
    %v2744 = stablehlo.multiply %v2736, %v2743 : tensor<64x256x14x14xf32>
    %v2745 = stablehlo.reshape %v2610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2746 = stablehlo.multiply %v2745, %v2744 : tensor<64x256x14x14xf32>
    %v2747 = stablehlo.reduce(%v2746 init: %v2731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2748 = stablehlo.reshape %v2610 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2749 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2750 = stablehlo.reduce(%v2748 init: %v2749) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2751 = stablehlo.reshape %v1178 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2752 = stablehlo.reshape %v2599 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2753 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2754 = stablehlo.transpose %v2752, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2755 = stablehlo.convert %v2753 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2756 = stablehlo.convert %v2754 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2757 = stablehlo.convolution(%v2755, %v2756)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v2758 = stablehlo.convert %v2757 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v2759 = stablehlo.transpose %v2758, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2760 = stablehlo.reshape %v1186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2761 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2762 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2763 = stablehlo.reduce(%v2760 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2764 = stablehlo.broadcast_in_dim %v2763, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2765 = stablehlo.divide %v2764, %v2762 : tensor<64x1024x14x14xf32>
    %v2766 = stablehlo.subtract %v2760, %v2765 : tensor<64x1024x14x14xf32>
    %v2767 = stablehlo.multiply %v2766, %v2766 : tensor<64x1024x14x14xf32>
    %v2768 = stablehlo.reduce(%v2767 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2769 = stablehlo.broadcast_in_dim %v2768, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2770 = stablehlo.divide %v2769, %v2762 : tensor<64x1024x14x14xf32>
    %v2771 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2772 = stablehlo.add %v2770, %v2771 : tensor<64x1024x14x14xf32>
    %v2773 = stablehlo.rsqrt %v2772 : tensor<64x1024x14x14xf32>
    %v2774 = stablehlo.multiply %v2766, %v2773 : tensor<64x1024x14x14xf32>
    %v2775 = stablehlo.reshape %v2569 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2776 = stablehlo.multiply %v2775, %v2774 : tensor<64x1024x14x14xf32>
    %v2777 = stablehlo.reduce(%v2776 init: %v2761) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2778 = stablehlo.reshape %v2569 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2780 = stablehlo.reduce(%v2778 init: %v2779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2781 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v2782 = stablehlo.compare GT, %v1116, %v2781 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v2783 = stablehlo.select %v2782, %v2690, %v2781 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v2784 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2785 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2786 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2787 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2788 = stablehlo.reduce(%v2784 init: %v2785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2789 = stablehlo.broadcast_in_dim %v2788, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2790 = stablehlo.divide %v2789, %v2786 : tensor<64x1024x14x14xf32>
    %v2791 = stablehlo.subtract %v2784, %v2790 : tensor<64x1024x14x14xf32>
    %v2792 = stablehlo.multiply %v2791, %v2791 : tensor<64x1024x14x14xf32>
    %v2793 = stablehlo.reduce(%v2792 init: %v2785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2794 = stablehlo.broadcast_in_dim %v2793, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2795 = stablehlo.divide %v2794, %v2786 : tensor<64x1024x14x14xf32>
    %v2796 = stablehlo.add %v2795, %v2787 : tensor<64x1024x14x14xf32>
    %v2797 = stablehlo.rsqrt %v2796 : tensor<64x1024x14x14xf32>
    %v2798 = stablehlo.multiply %v2791, %v2797 : tensor<64x1024x14x14xf32>
    %v2799 = stablehlo.broadcast_in_dim %s3b3g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2800 = stablehlo.reshape %v2783 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2801 = stablehlo.multiply %v2799, %v2800 : tensor<64x1024x14x14xf32>
    %v2802 = stablehlo.reduce(%v2801 init: %v2785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2803 = stablehlo.broadcast_in_dim %v2802, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2804 = stablehlo.multiply %v2798, %v2801 : tensor<64x1024x14x14xf32>
    %v2805 = stablehlo.reduce(%v2804 init: %v2785) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2806 = stablehlo.broadcast_in_dim %v2805, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2807 = stablehlo.multiply %v2801, %v2786 : tensor<64x1024x14x14xf32>
    %v2808 = stablehlo.subtract %v2807, %v2803 : tensor<64x1024x14x14xf32>
    %v2809 = stablehlo.multiply %v2798, %v2806 : tensor<64x1024x14x14xf32>
    %v2810 = stablehlo.subtract %v2808, %v2809 : tensor<64x1024x14x14xf32>
    %v2811 = stablehlo.divide %v2797, %v2786 : tensor<64x1024x14x14xf32>
    %v2812 = stablehlo.multiply %v2811, %v2810 : tensor<64x1024x14x14xf32>
    %v2813 = stablehlo.reshape %v2812 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2814 = stablehlo.reshape %v2813 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2815 = stablehlo.reverse %s3b3W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v2816 = stablehlo.transpose %v2815, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2817 = stablehlo.convert %v2814 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v2818 = stablehlo.convert %v2816 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v2819 = stablehlo.convolution(%v2817, %v2818)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2820 = stablehlo.convert %v2819 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2821 = stablehlo.reshape %v2820 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2822 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2823 = stablehlo.compare GT, %v1085, %v2822 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2824 = stablehlo.select %v2823, %v2821, %v2822 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2825 = stablehlo.reshape %v1065 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2827 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2828 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2829 = stablehlo.reduce(%v2825 init: %v2826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2830 = stablehlo.broadcast_in_dim %v2829, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2831 = stablehlo.divide %v2830, %v2827 : tensor<64x256x14x14xf32>
    %v2832 = stablehlo.subtract %v2825, %v2831 : tensor<64x256x14x14xf32>
    %v2833 = stablehlo.multiply %v2832, %v2832 : tensor<64x256x14x14xf32>
    %v2834 = stablehlo.reduce(%v2833 init: %v2826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2835 = stablehlo.broadcast_in_dim %v2834, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2836 = stablehlo.divide %v2835, %v2827 : tensor<64x256x14x14xf32>
    %v2837 = stablehlo.add %v2836, %v2828 : tensor<64x256x14x14xf32>
    %v2838 = stablehlo.rsqrt %v2837 : tensor<64x256x14x14xf32>
    %v2839 = stablehlo.multiply %v2832, %v2838 : tensor<64x256x14x14xf32>
    %v2840 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2841 = stablehlo.reshape %v2824 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2842 = stablehlo.multiply %v2840, %v2841 : tensor<64x256x14x14xf32>
    %v2843 = stablehlo.reduce(%v2842 init: %v2826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2844 = stablehlo.broadcast_in_dim %v2843, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2845 = stablehlo.multiply %v2839, %v2842 : tensor<64x256x14x14xf32>
    %v2846 = stablehlo.reduce(%v2845 init: %v2826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2847 = stablehlo.broadcast_in_dim %v2846, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2848 = stablehlo.multiply %v2842, %v2827 : tensor<64x256x14x14xf32>
    %v2849 = stablehlo.subtract %v2848, %v2844 : tensor<64x256x14x14xf32>
    %v2850 = stablehlo.multiply %v2839, %v2847 : tensor<64x256x14x14xf32>
    %v2851 = stablehlo.subtract %v2849, %v2850 : tensor<64x256x14x14xf32>
    %v2852 = stablehlo.divide %v2838, %v2827 : tensor<64x256x14x14xf32>
    %v2853 = stablehlo.multiply %v2852, %v2851 : tensor<64x256x14x14xf32>
    %v2854 = stablehlo.reshape %v2853 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2855 = stablehlo.reshape %v2854 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2856 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2857 = stablehlo.transpose %v2856, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2858 = stablehlo.convert %v2855 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2859 = stablehlo.convert %v2857 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2860 = stablehlo.convolution(%v2858, %v2859)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2861 = stablehlo.convert %v2860 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2862 = stablehlo.reshape %v2861 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2863 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2864 = stablehlo.compare GT, %v1055, %v2863 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2865 = stablehlo.select %v2864, %v2862, %v2863 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2866 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2867 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2868 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2869 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2870 = stablehlo.reduce(%v2866 init: %v2867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2871 = stablehlo.broadcast_in_dim %v2870, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2872 = stablehlo.divide %v2871, %v2868 : tensor<64x256x14x14xf32>
    %v2873 = stablehlo.subtract %v2866, %v2872 : tensor<64x256x14x14xf32>
    %v2874 = stablehlo.multiply %v2873, %v2873 : tensor<64x256x14x14xf32>
    %v2875 = stablehlo.reduce(%v2874 init: %v2867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2876 = stablehlo.broadcast_in_dim %v2875, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2877 = stablehlo.divide %v2876, %v2868 : tensor<64x256x14x14xf32>
    %v2878 = stablehlo.add %v2877, %v2869 : tensor<64x256x14x14xf32>
    %v2879 = stablehlo.rsqrt %v2878 : tensor<64x256x14x14xf32>
    %v2880 = stablehlo.multiply %v2873, %v2879 : tensor<64x256x14x14xf32>
    %v2881 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2882 = stablehlo.reshape %v2865 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2883 = stablehlo.multiply %v2881, %v2882 : tensor<64x256x14x14xf32>
    %v2884 = stablehlo.reduce(%v2883 init: %v2867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2885 = stablehlo.broadcast_in_dim %v2884, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2886 = stablehlo.multiply %v2880, %v2883 : tensor<64x256x14x14xf32>
    %v2887 = stablehlo.reduce(%v2886 init: %v2867) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2888 = stablehlo.broadcast_in_dim %v2887, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2889 = stablehlo.multiply %v2883, %v2868 : tensor<64x256x14x14xf32>
    %v2890 = stablehlo.subtract %v2889, %v2885 : tensor<64x256x14x14xf32>
    %v2891 = stablehlo.multiply %v2880, %v2888 : tensor<64x256x14x14xf32>
    %v2892 = stablehlo.subtract %v2890, %v2891 : tensor<64x256x14x14xf32>
    %v2893 = stablehlo.divide %v2879, %v2868 : tensor<64x256x14x14xf32>
    %v2894 = stablehlo.multiply %v2893, %v2892 : tensor<64x256x14x14xf32>
    %v2895 = stablehlo.reshape %v2894 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2896 = stablehlo.reshape %v2895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2897 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v2898 = stablehlo.transpose %v2897, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2899 = stablehlo.convert %v2896 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2900 = stablehlo.convert %v2898 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v2901 = stablehlo.convolution(%v2899, %v2900)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v2902 = stablehlo.convert %v2901 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v2903 = stablehlo.reshape %v2902 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v2904 = stablehlo.add %v2903, %v2783 : tensor<64x200704xf32>
    %v2905 = stablehlo.reshape %v1027 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2906 = stablehlo.reshape %v2895 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2907 = stablehlo.transpose %v2905, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2908 = stablehlo.transpose %v2906, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2909 = stablehlo.convert %v2907 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2910 = stablehlo.convert %v2908 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2911 = stablehlo.convolution(%v2909, %v2910)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v2912 = stablehlo.convert %v2911 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v2913 = stablehlo.transpose %v2912, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v2914 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2915 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2916 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2917 = stablehlo.reduce(%v2914 init: %v2915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2918 = stablehlo.broadcast_in_dim %v2917, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2919 = stablehlo.divide %v2918, %v2916 : tensor<64x256x14x14xf32>
    %v2920 = stablehlo.subtract %v2914, %v2919 : tensor<64x256x14x14xf32>
    %v2921 = stablehlo.multiply %v2920, %v2920 : tensor<64x256x14x14xf32>
    %v2922 = stablehlo.reduce(%v2921 init: %v2915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2923 = stablehlo.broadcast_in_dim %v2922, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2924 = stablehlo.divide %v2923, %v2916 : tensor<64x256x14x14xf32>
    %v2925 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2926 = stablehlo.add %v2924, %v2925 : tensor<64x256x14x14xf32>
    %v2927 = stablehlo.rsqrt %v2926 : tensor<64x256x14x14xf32>
    %v2928 = stablehlo.multiply %v2920, %v2927 : tensor<64x256x14x14xf32>
    %v2929 = stablehlo.reshape %v2865 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2930 = stablehlo.multiply %v2929, %v2928 : tensor<64x256x14x14xf32>
    %v2931 = stablehlo.reduce(%v2930 init: %v2915) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2932 = stablehlo.reshape %v2865 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2933 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2934 = stablehlo.reduce(%v2932 init: %v2933) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2935 = stablehlo.reshape %v1057 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2936 = stablehlo.reshape %v2854 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2937 = stablehlo.transpose %v2935, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2938 = stablehlo.transpose %v2936, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2939 = stablehlo.convert %v2937 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2940 = stablehlo.convert %v2938 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2941 = stablehlo.convolution(%v2939, %v2940)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2942 = stablehlo.convert %v2941 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2943 = stablehlo.transpose %v2942, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2944 = stablehlo.reshape %v1065 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2946 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2947 = stablehlo.reduce(%v2944 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2948 = stablehlo.broadcast_in_dim %v2947, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2949 = stablehlo.divide %v2948, %v2946 : tensor<64x256x14x14xf32>
    %v2950 = stablehlo.subtract %v2944, %v2949 : tensor<64x256x14x14xf32>
    %v2951 = stablehlo.multiply %v2950, %v2950 : tensor<64x256x14x14xf32>
    %v2952 = stablehlo.reduce(%v2951 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2953 = stablehlo.broadcast_in_dim %v2952, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2954 = stablehlo.divide %v2953, %v2946 : tensor<64x256x14x14xf32>
    %v2955 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2956 = stablehlo.add %v2954, %v2955 : tensor<64x256x14x14xf32>
    %v2957 = stablehlo.rsqrt %v2956 : tensor<64x256x14x14xf32>
    %v2958 = stablehlo.multiply %v2950, %v2957 : tensor<64x256x14x14xf32>
    %v2959 = stablehlo.reshape %v2824 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2960 = stablehlo.multiply %v2959, %v2958 : tensor<64x256x14x14xf32>
    %v2961 = stablehlo.reduce(%v2960 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2962 = stablehlo.reshape %v2824 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2964 = stablehlo.reduce(%v2962 init: %v2963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2965 = stablehlo.reshape %v1087 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2966 = stablehlo.reshape %v2813 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2967 = stablehlo.transpose %v2965, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2968 = stablehlo.transpose %v2966, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v2969 = stablehlo.convert %v2967 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2970 = stablehlo.convert %v2968 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v2971 = stablehlo.convolution(%v2969, %v2970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v2972 = stablehlo.convert %v2971 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v2973 = stablehlo.transpose %v2972, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v2974 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2976 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v2977 = stablehlo.reduce(%v2974 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2979 = stablehlo.divide %v2978, %v2976 : tensor<64x1024x14x14xf32>
    %v2980 = stablehlo.subtract %v2974, %v2979 : tensor<64x1024x14x14xf32>
    %v2981 = stablehlo.multiply %v2980, %v2980 : tensor<64x1024x14x14xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2982, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v2984 = stablehlo.divide %v2983, %v2976 : tensor<64x1024x14x14xf32>
    %v2985 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v2986 = stablehlo.add %v2984, %v2985 : tensor<64x1024x14x14xf32>
    %v2987 = stablehlo.rsqrt %v2986 : tensor<64x1024x14x14xf32>
    %v2988 = stablehlo.multiply %v2980, %v2987 : tensor<64x1024x14x14xf32>
    %v2989 = stablehlo.reshape %v2783 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2990 = stablehlo.multiply %v2989, %v2988 : tensor<64x1024x14x14xf32>
    %v2991 = stablehlo.reduce(%v2990 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2992 = stablehlo.reshape %v2783 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2994 = stablehlo.reduce(%v2992 init: %v2993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v2995 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v2996 = stablehlo.compare GT, %v1025, %v2995 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v2997 = stablehlo.select %v2996, %v2904, %v2995 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v2998 = stablehlo.reshape %v1004 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3000 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3001 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3002 = stablehlo.reduce(%v2998 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3003 = stablehlo.broadcast_in_dim %v3002, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3004 = stablehlo.divide %v3003, %v3000 : tensor<64x1024x14x14xf32>
    %v3005 = stablehlo.subtract %v2998, %v3004 : tensor<64x1024x14x14xf32>
    %v3006 = stablehlo.multiply %v3005, %v3005 : tensor<64x1024x14x14xf32>
    %v3007 = stablehlo.reduce(%v3006 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3008 = stablehlo.broadcast_in_dim %v3007, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3009 = stablehlo.divide %v3008, %v3000 : tensor<64x1024x14x14xf32>
    %v3010 = stablehlo.add %v3009, %v3001 : tensor<64x1024x14x14xf32>
    %v3011 = stablehlo.rsqrt %v3010 : tensor<64x1024x14x14xf32>
    %v3012 = stablehlo.multiply %v3005, %v3011 : tensor<64x1024x14x14xf32>
    %v3013 = stablehlo.broadcast_in_dim %s3b2g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3014 = stablehlo.reshape %v2997 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3015 = stablehlo.multiply %v3013, %v3014 : tensor<64x1024x14x14xf32>
    %v3016 = stablehlo.reduce(%v3015 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3017 = stablehlo.broadcast_in_dim %v3016, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3018 = stablehlo.multiply %v3012, %v3015 : tensor<64x1024x14x14xf32>
    %v3019 = stablehlo.reduce(%v3018 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3020 = stablehlo.broadcast_in_dim %v3019, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3021 = stablehlo.multiply %v3015, %v3000 : tensor<64x1024x14x14xf32>
    %v3022 = stablehlo.subtract %v3021, %v3017 : tensor<64x1024x14x14xf32>
    %v3023 = stablehlo.multiply %v3012, %v3020 : tensor<64x1024x14x14xf32>
    %v3024 = stablehlo.subtract %v3022, %v3023 : tensor<64x1024x14x14xf32>
    %v3025 = stablehlo.divide %v3011, %v3000 : tensor<64x1024x14x14xf32>
    %v3026 = stablehlo.multiply %v3025, %v3024 : tensor<64x1024x14x14xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3028 = stablehlo.reshape %v3027 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3029 = stablehlo.reverse %s3b2W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3030 = stablehlo.transpose %v3029, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3031 = stablehlo.convert %v3028 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3032 = stablehlo.convert %v3030 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3033 = stablehlo.convolution(%v3031, %v3032)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3034 = stablehlo.convert %v3033 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3035 = stablehlo.reshape %v3034 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3036 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v3037 = stablehlo.compare GT, %v994, %v3036 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v3038 = stablehlo.select %v3037, %v3035, %v3036 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v3039 = stablehlo.reshape %v974 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3041 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3042 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3043 = stablehlo.reduce(%v3039 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3045 = stablehlo.divide %v3044, %v3041 : tensor<64x256x14x14xf32>
    %v3046 = stablehlo.subtract %v3039, %v3045 : tensor<64x256x14x14xf32>
    %v3047 = stablehlo.multiply %v3046, %v3046 : tensor<64x256x14x14xf32>
    %v3048 = stablehlo.reduce(%v3047 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3049 = stablehlo.broadcast_in_dim %v3048, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3050 = stablehlo.divide %v3049, %v3041 : tensor<64x256x14x14xf32>
    %v3051 = stablehlo.add %v3050, %v3042 : tensor<64x256x14x14xf32>
    %v3052 = stablehlo.rsqrt %v3051 : tensor<64x256x14x14xf32>
    %v3053 = stablehlo.multiply %v3046, %v3052 : tensor<64x256x14x14xf32>
    %v3054 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3055 = stablehlo.reshape %v3038 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3056 = stablehlo.multiply %v3054, %v3055 : tensor<64x256x14x14xf32>
    %v3057 = stablehlo.reduce(%v3056 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3058 = stablehlo.broadcast_in_dim %v3057, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3059 = stablehlo.multiply %v3053, %v3056 : tensor<64x256x14x14xf32>
    %v3060 = stablehlo.reduce(%v3059 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3061 = stablehlo.broadcast_in_dim %v3060, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3062 = stablehlo.multiply %v3056, %v3041 : tensor<64x256x14x14xf32>
    %v3063 = stablehlo.subtract %v3062, %v3058 : tensor<64x256x14x14xf32>
    %v3064 = stablehlo.multiply %v3053, %v3061 : tensor<64x256x14x14xf32>
    %v3065 = stablehlo.subtract %v3063, %v3064 : tensor<64x256x14x14xf32>
    %v3066 = stablehlo.divide %v3052, %v3041 : tensor<64x256x14x14xf32>
    %v3067 = stablehlo.multiply %v3066, %v3065 : tensor<64x256x14x14xf32>
    %v3068 = stablehlo.reshape %v3067 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3069 = stablehlo.reshape %v3068 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3070 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3071 = stablehlo.transpose %v3070, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3072 = stablehlo.convert %v3069 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3073 = stablehlo.convert %v3071 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3074 = stablehlo.convolution(%v3072, %v3073)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3075 = stablehlo.convert %v3074 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3076 = stablehlo.reshape %v3075 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3077 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v3078 = stablehlo.compare GT, %v964, %v3077 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v3079 = stablehlo.select %v3078, %v3076, %v3077 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v3080 = stablehlo.reshape %v944 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3081 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3082 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3083 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3084 = stablehlo.reduce(%v3080 init: %v3081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3085 = stablehlo.broadcast_in_dim %v3084, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3086 = stablehlo.divide %v3085, %v3082 : tensor<64x256x14x14xf32>
    %v3087 = stablehlo.subtract %v3080, %v3086 : tensor<64x256x14x14xf32>
    %v3088 = stablehlo.multiply %v3087, %v3087 : tensor<64x256x14x14xf32>
    %v3089 = stablehlo.reduce(%v3088 init: %v3081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3090 = stablehlo.broadcast_in_dim %v3089, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3091 = stablehlo.divide %v3090, %v3082 : tensor<64x256x14x14xf32>
    %v3092 = stablehlo.add %v3091, %v3083 : tensor<64x256x14x14xf32>
    %v3093 = stablehlo.rsqrt %v3092 : tensor<64x256x14x14xf32>
    %v3094 = stablehlo.multiply %v3087, %v3093 : tensor<64x256x14x14xf32>
    %v3095 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3096 = stablehlo.reshape %v3079 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3097 = stablehlo.multiply %v3095, %v3096 : tensor<64x256x14x14xf32>
    %v3098 = stablehlo.reduce(%v3097 init: %v3081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3099 = stablehlo.broadcast_in_dim %v3098, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3100 = stablehlo.multiply %v3094, %v3097 : tensor<64x256x14x14xf32>
    %v3101 = stablehlo.reduce(%v3100 init: %v3081) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3102 = stablehlo.broadcast_in_dim %v3101, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3103 = stablehlo.multiply %v3097, %v3082 : tensor<64x256x14x14xf32>
    %v3104 = stablehlo.subtract %v3103, %v3099 : tensor<64x256x14x14xf32>
    %v3105 = stablehlo.multiply %v3094, %v3102 : tensor<64x256x14x14xf32>
    %v3106 = stablehlo.subtract %v3104, %v3105 : tensor<64x256x14x14xf32>
    %v3107 = stablehlo.divide %v3093, %v3082 : tensor<64x256x14x14xf32>
    %v3108 = stablehlo.multiply %v3107, %v3106 : tensor<64x256x14x14xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3110 = stablehlo.reshape %v3109 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3111 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3112 = stablehlo.transpose %v3111, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3113 = stablehlo.convert %v3110 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3114 = stablehlo.convert %v3112 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3115 = stablehlo.convolution(%v3113, %v3114)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3116 = stablehlo.convert %v3115 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3117 = stablehlo.reshape %v3116 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3118 = stablehlo.add %v3117, %v2997 : tensor<64x200704xf32>
    %v3119 = stablehlo.reshape %v936 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3120 = stablehlo.reshape %v3109 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3121 = stablehlo.transpose %v3119, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3122 = stablehlo.transpose %v3120, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3123 = stablehlo.convert %v3121 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3124 = stablehlo.convert %v3122 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3125 = stablehlo.convolution(%v3123, %v3124)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3126 = stablehlo.convert %v3125 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3127 = stablehlo.transpose %v3126, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3128 = stablehlo.reshape %v944 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3129 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3130 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3131 = stablehlo.reduce(%v3128 init: %v3129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3132 = stablehlo.broadcast_in_dim %v3131, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3133 = stablehlo.divide %v3132, %v3130 : tensor<64x256x14x14xf32>
    %v3134 = stablehlo.subtract %v3128, %v3133 : tensor<64x256x14x14xf32>
    %v3135 = stablehlo.multiply %v3134, %v3134 : tensor<64x256x14x14xf32>
    %v3136 = stablehlo.reduce(%v3135 init: %v3129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3137 = stablehlo.broadcast_in_dim %v3136, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3138 = stablehlo.divide %v3137, %v3130 : tensor<64x256x14x14xf32>
    %v3139 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3140 = stablehlo.add %v3138, %v3139 : tensor<64x256x14x14xf32>
    %v3141 = stablehlo.rsqrt %v3140 : tensor<64x256x14x14xf32>
    %v3142 = stablehlo.multiply %v3134, %v3141 : tensor<64x256x14x14xf32>
    %v3143 = stablehlo.reshape %v3079 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3144 = stablehlo.multiply %v3143, %v3142 : tensor<64x256x14x14xf32>
    %v3145 = stablehlo.reduce(%v3144 init: %v3129) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3146 = stablehlo.reshape %v3079 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3147 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3148 = stablehlo.reduce(%v3146 init: %v3147) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3149 = stablehlo.reshape %v966 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3150 = stablehlo.reshape %v3068 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3151 = stablehlo.transpose %v3149, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3152 = stablehlo.transpose %v3150, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3153 = stablehlo.convert %v3151 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3154 = stablehlo.convert %v3152 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3155 = stablehlo.convolution(%v3153, %v3154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3156 = stablehlo.convert %v3155 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3157 = stablehlo.transpose %v3156, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3158 = stablehlo.reshape %v974 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3160 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3161 = stablehlo.reduce(%v3158 init: %v3159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3162 = stablehlo.broadcast_in_dim %v3161, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3163 = stablehlo.divide %v3162, %v3160 : tensor<64x256x14x14xf32>
    %v3164 = stablehlo.subtract %v3158, %v3163 : tensor<64x256x14x14xf32>
    %v3165 = stablehlo.multiply %v3164, %v3164 : tensor<64x256x14x14xf32>
    %v3166 = stablehlo.reduce(%v3165 init: %v3159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3167 = stablehlo.broadcast_in_dim %v3166, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3168 = stablehlo.divide %v3167, %v3160 : tensor<64x256x14x14xf32>
    %v3169 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3170 = stablehlo.add %v3168, %v3169 : tensor<64x256x14x14xf32>
    %v3171 = stablehlo.rsqrt %v3170 : tensor<64x256x14x14xf32>
    %v3172 = stablehlo.multiply %v3164, %v3171 : tensor<64x256x14x14xf32>
    %v3173 = stablehlo.reshape %v3038 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3174 = stablehlo.multiply %v3173, %v3172 : tensor<64x256x14x14xf32>
    %v3175 = stablehlo.reduce(%v3174 init: %v3159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3176 = stablehlo.reshape %v3038 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3178 = stablehlo.reduce(%v3176 init: %v3177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3179 = stablehlo.reshape %v996 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3180 = stablehlo.reshape %v3027 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3181 = stablehlo.transpose %v3179, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3182 = stablehlo.transpose %v3180, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3183 = stablehlo.convert %v3181 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3184 = stablehlo.convert %v3182 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3185 = stablehlo.convolution(%v3183, %v3184)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3186 = stablehlo.convert %v3185 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3187 = stablehlo.transpose %v3186, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3188 = stablehlo.reshape %v1004 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3189 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3190 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3191 = stablehlo.reduce(%v3188 init: %v3189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3192 = stablehlo.broadcast_in_dim %v3191, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3193 = stablehlo.divide %v3192, %v3190 : tensor<64x1024x14x14xf32>
    %v3194 = stablehlo.subtract %v3188, %v3193 : tensor<64x1024x14x14xf32>
    %v3195 = stablehlo.multiply %v3194, %v3194 : tensor<64x1024x14x14xf32>
    %v3196 = stablehlo.reduce(%v3195 init: %v3189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3198 = stablehlo.divide %v3197, %v3190 : tensor<64x1024x14x14xf32>
    %v3199 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3200 = stablehlo.add %v3198, %v3199 : tensor<64x1024x14x14xf32>
    %v3201 = stablehlo.rsqrt %v3200 : tensor<64x1024x14x14xf32>
    %v3202 = stablehlo.multiply %v3194, %v3201 : tensor<64x1024x14x14xf32>
    %v3203 = stablehlo.reshape %v2997 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3204 = stablehlo.multiply %v3203, %v3202 : tensor<64x1024x14x14xf32>
    %v3205 = stablehlo.reduce(%v3204 init: %v3189) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3206 = stablehlo.reshape %v2997 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3207 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3208 = stablehlo.reduce(%v3206 init: %v3207) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3209 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3210 = stablehlo.compare GT, %v934, %v3209 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3211 = stablehlo.select %v3210, %v3118, %v3209 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3212 = stablehlo.reshape %v913 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3213 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3214 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3215 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3216 = stablehlo.reduce(%v3212 init: %v3213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3217 = stablehlo.broadcast_in_dim %v3216, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3218 = stablehlo.divide %v3217, %v3214 : tensor<64x1024x14x14xf32>
    %v3219 = stablehlo.subtract %v3212, %v3218 : tensor<64x1024x14x14xf32>
    %v3220 = stablehlo.multiply %v3219, %v3219 : tensor<64x1024x14x14xf32>
    %v3221 = stablehlo.reduce(%v3220 init: %v3213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3222 = stablehlo.broadcast_in_dim %v3221, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3223 = stablehlo.divide %v3222, %v3214 : tensor<64x1024x14x14xf32>
    %v3224 = stablehlo.add %v3223, %v3215 : tensor<64x1024x14x14xf32>
    %v3225 = stablehlo.rsqrt %v3224 : tensor<64x1024x14x14xf32>
    %v3226 = stablehlo.multiply %v3219, %v3225 : tensor<64x1024x14x14xf32>
    %v3227 = stablehlo.broadcast_in_dim %s3b1g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3228 = stablehlo.reshape %v3211 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3229 = stablehlo.multiply %v3227, %v3228 : tensor<64x1024x14x14xf32>
    %v3230 = stablehlo.reduce(%v3229 init: %v3213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3231 = stablehlo.broadcast_in_dim %v3230, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3232 = stablehlo.multiply %v3226, %v3229 : tensor<64x1024x14x14xf32>
    %v3233 = stablehlo.reduce(%v3232 init: %v3213) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3234 = stablehlo.broadcast_in_dim %v3233, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3235 = stablehlo.multiply %v3229, %v3214 : tensor<64x1024x14x14xf32>
    %v3236 = stablehlo.subtract %v3235, %v3231 : tensor<64x1024x14x14xf32>
    %v3237 = stablehlo.multiply %v3226, %v3234 : tensor<64x1024x14x14xf32>
    %v3238 = stablehlo.subtract %v3236, %v3237 : tensor<64x1024x14x14xf32>
    %v3239 = stablehlo.divide %v3225, %v3214 : tensor<64x1024x14x14xf32>
    %v3240 = stablehlo.multiply %v3239, %v3238 : tensor<64x1024x14x14xf32>
    %v3241 = stablehlo.reshape %v3240 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3242 = stablehlo.reshape %v3241 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3243 = stablehlo.reverse %s3b1W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3244 = stablehlo.transpose %v3243, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3245 = stablehlo.convert %v3242 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3246 = stablehlo.convert %v3244 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3247 = stablehlo.convolution(%v3245, %v3246)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3248 = stablehlo.convert %v3247 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3249 = stablehlo.reshape %v3248 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3250 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v3251 = stablehlo.compare GT, %v903, %v3250 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v3252 = stablehlo.select %v3251, %v3249, %v3250 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v3253 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3254 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3255 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3256 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3257 = stablehlo.reduce(%v3253 init: %v3254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3258 = stablehlo.broadcast_in_dim %v3257, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3259 = stablehlo.divide %v3258, %v3255 : tensor<64x256x14x14xf32>
    %v3260 = stablehlo.subtract %v3253, %v3259 : tensor<64x256x14x14xf32>
    %v3261 = stablehlo.multiply %v3260, %v3260 : tensor<64x256x14x14xf32>
    %v3262 = stablehlo.reduce(%v3261 init: %v3254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3263 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3264 = stablehlo.divide %v3263, %v3255 : tensor<64x256x14x14xf32>
    %v3265 = stablehlo.add %v3264, %v3256 : tensor<64x256x14x14xf32>
    %v3266 = stablehlo.rsqrt %v3265 : tensor<64x256x14x14xf32>
    %v3267 = stablehlo.multiply %v3260, %v3266 : tensor<64x256x14x14xf32>
    %v3268 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3269 = stablehlo.reshape %v3252 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3270 = stablehlo.multiply %v3268, %v3269 : tensor<64x256x14x14xf32>
    %v3271 = stablehlo.reduce(%v3270 init: %v3254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3272 = stablehlo.broadcast_in_dim %v3271, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3273 = stablehlo.multiply %v3267, %v3270 : tensor<64x256x14x14xf32>
    %v3274 = stablehlo.reduce(%v3273 init: %v3254) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3275 = stablehlo.broadcast_in_dim %v3274, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3276 = stablehlo.multiply %v3270, %v3255 : tensor<64x256x14x14xf32>
    %v3277 = stablehlo.subtract %v3276, %v3272 : tensor<64x256x14x14xf32>
    %v3278 = stablehlo.multiply %v3267, %v3275 : tensor<64x256x14x14xf32>
    %v3279 = stablehlo.subtract %v3277, %v3278 : tensor<64x256x14x14xf32>
    %v3280 = stablehlo.divide %v3266, %v3255 : tensor<64x256x14x14xf32>
    %v3281 = stablehlo.multiply %v3280, %v3279 : tensor<64x256x14x14xf32>
    %v3282 = stablehlo.reshape %v3281 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3283 = stablehlo.reshape %v3282 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3284 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3285 = stablehlo.transpose %v3284, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3286 = stablehlo.convert %v3283 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3287 = stablehlo.convert %v3285 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3288 = stablehlo.convolution(%v3286, %v3287)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3289 = stablehlo.convert %v3288 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3290 = stablehlo.reshape %v3289 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3291 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v3292 = stablehlo.compare GT, %v873, %v3291 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v3293 = stablehlo.select %v3292, %v3290, %v3291 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v3294 = stablehlo.reshape %v853 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3295 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3296 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3297 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3298 = stablehlo.reduce(%v3294 init: %v3295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3299 = stablehlo.broadcast_in_dim %v3298, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3300 = stablehlo.divide %v3299, %v3296 : tensor<64x256x14x14xf32>
    %v3301 = stablehlo.subtract %v3294, %v3300 : tensor<64x256x14x14xf32>
    %v3302 = stablehlo.multiply %v3301, %v3301 : tensor<64x256x14x14xf32>
    %v3303 = stablehlo.reduce(%v3302 init: %v3295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3304 = stablehlo.broadcast_in_dim %v3303, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3305 = stablehlo.divide %v3304, %v3296 : tensor<64x256x14x14xf32>
    %v3306 = stablehlo.add %v3305, %v3297 : tensor<64x256x14x14xf32>
    %v3307 = stablehlo.rsqrt %v3306 : tensor<64x256x14x14xf32>
    %v3308 = stablehlo.multiply %v3301, %v3307 : tensor<64x256x14x14xf32>
    %v3309 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3310 = stablehlo.reshape %v3293 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3311 = stablehlo.multiply %v3309, %v3310 : tensor<64x256x14x14xf32>
    %v3312 = stablehlo.reduce(%v3311 init: %v3295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3313 = stablehlo.broadcast_in_dim %v3312, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3314 = stablehlo.multiply %v3308, %v3311 : tensor<64x256x14x14xf32>
    %v3315 = stablehlo.reduce(%v3314 init: %v3295) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3316 = stablehlo.broadcast_in_dim %v3315, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3317 = stablehlo.multiply %v3311, %v3296 : tensor<64x256x14x14xf32>
    %v3318 = stablehlo.subtract %v3317, %v3313 : tensor<64x256x14x14xf32>
    %v3319 = stablehlo.multiply %v3308, %v3316 : tensor<64x256x14x14xf32>
    %v3320 = stablehlo.subtract %v3318, %v3319 : tensor<64x256x14x14xf32>
    %v3321 = stablehlo.divide %v3307, %v3296 : tensor<64x256x14x14xf32>
    %v3322 = stablehlo.multiply %v3321, %v3320 : tensor<64x256x14x14xf32>
    %v3323 = stablehlo.reshape %v3322 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3324 = stablehlo.reshape %v3323 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3325 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x1024x1x1xf32>
    %v3326 = stablehlo.transpose %v3325, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3327 = stablehlo.convert %v3324 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3328 = stablehlo.convert %v3326 : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xbf16>
    %v3329 = stablehlo.convolution(%v3327, %v3328)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<1024x256x1x1xbf16>) -> tensor<64x1024x14x14xbf16>
    %v3330 = stablehlo.convert %v3329 : (tensor<64x1024x14x14xbf16>) -> tensor<64x1024x14x14xf32>
    %v3331 = stablehlo.reshape %v3330 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3332 = stablehlo.add %v3331, %v3211 : tensor<64x200704xf32>
    %v3333 = stablehlo.reshape %v845 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3334 = stablehlo.reshape %v3323 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3335 = stablehlo.transpose %v3333, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3336 = stablehlo.transpose %v3334, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3337 = stablehlo.convert %v3335 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3338 = stablehlo.convert %v3336 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3339 = stablehlo.convolution(%v3337, %v3338)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<1024x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<1024x256x1x1xbf16>
    %v3340 = stablehlo.convert %v3339 : (tensor<1024x256x1x1xbf16>) -> tensor<1024x256x1x1xf32>
    %v3341 = stablehlo.transpose %v3340, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3342 = stablehlo.reshape %v853 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3343 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3344 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3345 = stablehlo.reduce(%v3342 init: %v3343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3346 = stablehlo.broadcast_in_dim %v3345, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3347 = stablehlo.divide %v3346, %v3344 : tensor<64x256x14x14xf32>
    %v3348 = stablehlo.subtract %v3342, %v3347 : tensor<64x256x14x14xf32>
    %v3349 = stablehlo.multiply %v3348, %v3348 : tensor<64x256x14x14xf32>
    %v3350 = stablehlo.reduce(%v3349 init: %v3343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3351 = stablehlo.broadcast_in_dim %v3350, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3352 = stablehlo.divide %v3351, %v3344 : tensor<64x256x14x14xf32>
    %v3353 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3354 = stablehlo.add %v3352, %v3353 : tensor<64x256x14x14xf32>
    %v3355 = stablehlo.rsqrt %v3354 : tensor<64x256x14x14xf32>
    %v3356 = stablehlo.multiply %v3348, %v3355 : tensor<64x256x14x14xf32>
    %v3357 = stablehlo.reshape %v3293 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3358 = stablehlo.multiply %v3357, %v3356 : tensor<64x256x14x14xf32>
    %v3359 = stablehlo.reduce(%v3358 init: %v3343) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3360 = stablehlo.reshape %v3293 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3362 = stablehlo.reduce(%v3360 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3363 = stablehlo.reshape %v875 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3364 = stablehlo.reshape %v3282 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3365 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3366 = stablehlo.transpose %v3364, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3367 = stablehlo.convert %v3365 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3368 = stablehlo.convert %v3366 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3369 = stablehlo.convolution(%v3367, %v3368)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3370 = stablehlo.convert %v3369 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3371 = stablehlo.transpose %v3370, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3372 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3373 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3374 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3375 = stablehlo.reduce(%v3372 init: %v3373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3376 = stablehlo.broadcast_in_dim %v3375, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3377 = stablehlo.divide %v3376, %v3374 : tensor<64x256x14x14xf32>
    %v3378 = stablehlo.subtract %v3372, %v3377 : tensor<64x256x14x14xf32>
    %v3379 = stablehlo.multiply %v3378, %v3378 : tensor<64x256x14x14xf32>
    %v3380 = stablehlo.reduce(%v3379 init: %v3373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3381 = stablehlo.broadcast_in_dim %v3380, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3382 = stablehlo.divide %v3381, %v3374 : tensor<64x256x14x14xf32>
    %v3383 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3384 = stablehlo.add %v3382, %v3383 : tensor<64x256x14x14xf32>
    %v3385 = stablehlo.rsqrt %v3384 : tensor<64x256x14x14xf32>
    %v3386 = stablehlo.multiply %v3378, %v3385 : tensor<64x256x14x14xf32>
    %v3387 = stablehlo.reshape %v3252 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3388 = stablehlo.multiply %v3387, %v3386 : tensor<64x256x14x14xf32>
    %v3389 = stablehlo.reduce(%v3388 init: %v3373) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3390 = stablehlo.reshape %v3252 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3391 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3392 = stablehlo.reduce(%v3390 init: %v3391) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3393 = stablehlo.reshape %v905 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3394 = stablehlo.reshape %v3241 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3395 = stablehlo.transpose %v3393, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3396 = stablehlo.transpose %v3394, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3397 = stablehlo.convert %v3395 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3398 = stablehlo.convert %v3396 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3399 = stablehlo.convolution(%v3397, %v3398)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3400 = stablehlo.convert %v3399 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3401 = stablehlo.transpose %v3400, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3402 = stablehlo.reshape %v913 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3403 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3404 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3405 = stablehlo.reduce(%v3402 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3406 = stablehlo.broadcast_in_dim %v3405, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3407 = stablehlo.divide %v3406, %v3404 : tensor<64x1024x14x14xf32>
    %v3408 = stablehlo.subtract %v3402, %v3407 : tensor<64x1024x14x14xf32>
    %v3409 = stablehlo.multiply %v3408, %v3408 : tensor<64x1024x14x14xf32>
    %v3410 = stablehlo.reduce(%v3409 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3411 = stablehlo.broadcast_in_dim %v3410, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3412 = stablehlo.divide %v3411, %v3404 : tensor<64x1024x14x14xf32>
    %v3413 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3414 = stablehlo.add %v3412, %v3413 : tensor<64x1024x14x14xf32>
    %v3415 = stablehlo.rsqrt %v3414 : tensor<64x1024x14x14xf32>
    %v3416 = stablehlo.multiply %v3408, %v3415 : tensor<64x1024x14x14xf32>
    %v3417 = stablehlo.reshape %v3211 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3418 = stablehlo.multiply %v3417, %v3416 : tensor<64x1024x14x14xf32>
    %v3419 = stablehlo.reduce(%v3418 init: %v3403) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3420 = stablehlo.reshape %v3211 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3421 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3422 = stablehlo.reduce(%v3420 init: %v3421) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3423 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3424 = stablehlo.compare GT, %v843, %v3423 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3425 = stablehlo.select %v3424, %v3332, %v3423 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3426 = stablehlo.reshape %v794 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3427 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3428 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3429 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3430 = stablehlo.reduce(%v3426 init: %v3427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3431 = stablehlo.broadcast_in_dim %v3430, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3432 = stablehlo.divide %v3431, %v3428 : tensor<64x1024x14x14xf32>
    %v3433 = stablehlo.subtract %v3426, %v3432 : tensor<64x1024x14x14xf32>
    %v3434 = stablehlo.multiply %v3433, %v3433 : tensor<64x1024x14x14xf32>
    %v3435 = stablehlo.reduce(%v3434 init: %v3427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3436 = stablehlo.broadcast_in_dim %v3435, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3437 = stablehlo.divide %v3436, %v3428 : tensor<64x1024x14x14xf32>
    %v3438 = stablehlo.add %v3437, %v3429 : tensor<64x1024x14x14xf32>
    %v3439 = stablehlo.rsqrt %v3438 : tensor<64x1024x14x14xf32>
    %v3440 = stablehlo.multiply %v3433, %v3439 : tensor<64x1024x14x14xf32>
    %v3441 = stablehlo.broadcast_in_dim %s3b0g3, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3442 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3443 = stablehlo.multiply %v3441, %v3442 : tensor<64x1024x14x14xf32>
    %v3444 = stablehlo.reduce(%v3443 init: %v3427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3445 = stablehlo.broadcast_in_dim %v3444, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3446 = stablehlo.multiply %v3440, %v3443 : tensor<64x1024x14x14xf32>
    %v3447 = stablehlo.reduce(%v3446 init: %v3427) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3448 = stablehlo.broadcast_in_dim %v3447, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3449 = stablehlo.multiply %v3443, %v3428 : tensor<64x1024x14x14xf32>
    %v3450 = stablehlo.subtract %v3449, %v3445 : tensor<64x1024x14x14xf32>
    %v3451 = stablehlo.multiply %v3440, %v3448 : tensor<64x1024x14x14xf32>
    %v3452 = stablehlo.subtract %v3450, %v3451 : tensor<64x1024x14x14xf32>
    %v3453 = stablehlo.divide %v3439, %v3428 : tensor<64x1024x14x14xf32>
    %v3454 = stablehlo.multiply %v3453, %v3452 : tensor<64x1024x14x14xf32>
    %v3455 = stablehlo.reshape %v3454 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3456 = stablehlo.reshape %v3455 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3457 = stablehlo.reverse %s3b0W3, dims = [2, 3] : tensor<1024x256x1x1xf32>
    %v3458 = stablehlo.transpose %v3457, dims = [1, 0, 2, 3] : (tensor<1024x256x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %v3459 = stablehlo.convert %v3456 : (tensor<64x1024x14x14xf32>) -> tensor<64x1024x14x14xbf16>
    %v3460 = stablehlo.convert %v3458 : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xbf16>
    %v3461 = stablehlo.convolution(%v3459, %v3460)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x14x14xbf16>, tensor<256x1024x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v3462 = stablehlo.convert %v3461 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3463 = stablehlo.reshape %v3462 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3464 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v3465 = stablehlo.compare GT, %v784, %v3464 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v3466 = stablehlo.select %v3465, %v3463, %v3464 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v3467 = stablehlo.reshape %v764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3469 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3470 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3471 = stablehlo.reduce(%v3467 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3472 = stablehlo.broadcast_in_dim %v3471, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3473 = stablehlo.divide %v3472, %v3469 : tensor<64x256x14x14xf32>
    %v3474 = stablehlo.subtract %v3467, %v3473 : tensor<64x256x14x14xf32>
    %v3475 = stablehlo.multiply %v3474, %v3474 : tensor<64x256x14x14xf32>
    %v3476 = stablehlo.reduce(%v3475 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3477 = stablehlo.broadcast_in_dim %v3476, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3478 = stablehlo.divide %v3477, %v3469 : tensor<64x256x14x14xf32>
    %v3479 = stablehlo.add %v3478, %v3470 : tensor<64x256x14x14xf32>
    %v3480 = stablehlo.rsqrt %v3479 : tensor<64x256x14x14xf32>
    %v3481 = stablehlo.multiply %v3474, %v3480 : tensor<64x256x14x14xf32>
    %v3482 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3483 = stablehlo.reshape %v3466 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3484 = stablehlo.multiply %v3482, %v3483 : tensor<64x256x14x14xf32>
    %v3485 = stablehlo.reduce(%v3484 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3486 = stablehlo.broadcast_in_dim %v3485, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3487 = stablehlo.multiply %v3481, %v3484 : tensor<64x256x14x14xf32>
    %v3488 = stablehlo.reduce(%v3487 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3489 = stablehlo.broadcast_in_dim %v3488, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3490 = stablehlo.multiply %v3484, %v3469 : tensor<64x256x14x14xf32>
    %v3491 = stablehlo.subtract %v3490, %v3486 : tensor<64x256x14x14xf32>
    %v3492 = stablehlo.multiply %v3481, %v3489 : tensor<64x256x14x14xf32>
    %v3493 = stablehlo.subtract %v3491, %v3492 : tensor<64x256x14x14xf32>
    %v3494 = stablehlo.divide %v3480, %v3469 : tensor<64x256x14x14xf32>
    %v3495 = stablehlo.multiply %v3494, %v3493 : tensor<64x256x14x14xf32>
    %v3496 = stablehlo.reshape %v3495 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3497 = stablehlo.reshape %v3496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3499 = stablehlo.pad %v3497, %v3498, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3500 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3501 = stablehlo.transpose %v3500, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3502 = stablehlo.convert %v3499 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3503 = stablehlo.convert %v3501 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3504 = stablehlo.convolution(%v3502, %v3503)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x28x28xbf16>
    %v3505 = stablehlo.convert %v3504 : (tensor<64x256x28x28xbf16>) -> tensor<64x256x28x28xf32>
    %v3506 = stablehlo.reshape %v3505 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v3507 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3508 = stablehlo.compare GT, %v754, %v3507 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3509 = stablehlo.select %v3508, %v3506, %v3507 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3510 = stablehlo.reshape %v734 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3512 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v3513 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v3514 = stablehlo.reduce(%v3510 init: %v3511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3515 = stablehlo.broadcast_in_dim %v3514, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3516 = stablehlo.divide %v3515, %v3512 : tensor<64x256x28x28xf32>
    %v3517 = stablehlo.subtract %v3510, %v3516 : tensor<64x256x28x28xf32>
    %v3518 = stablehlo.multiply %v3517, %v3517 : tensor<64x256x28x28xf32>
    %v3519 = stablehlo.reduce(%v3518 init: %v3511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3520 = stablehlo.broadcast_in_dim %v3519, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3521 = stablehlo.divide %v3520, %v3512 : tensor<64x256x28x28xf32>
    %v3522 = stablehlo.add %v3521, %v3513 : tensor<64x256x28x28xf32>
    %v3523 = stablehlo.rsqrt %v3522 : tensor<64x256x28x28xf32>
    %v3524 = stablehlo.multiply %v3517, %v3523 : tensor<64x256x28x28xf32>
    %v3525 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3526 = stablehlo.reshape %v3509 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3527 = stablehlo.multiply %v3525, %v3526 : tensor<64x256x28x28xf32>
    %v3528 = stablehlo.reduce(%v3527 init: %v3511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3529 = stablehlo.broadcast_in_dim %v3528, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3530 = stablehlo.multiply %v3524, %v3527 : tensor<64x256x28x28xf32>
    %v3531 = stablehlo.reduce(%v3530 init: %v3511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3532 = stablehlo.broadcast_in_dim %v3531, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3533 = stablehlo.multiply %v3527, %v3512 : tensor<64x256x28x28xf32>
    %v3534 = stablehlo.subtract %v3533, %v3529 : tensor<64x256x28x28xf32>
    %v3535 = stablehlo.multiply %v3524, %v3532 : tensor<64x256x28x28xf32>
    %v3536 = stablehlo.subtract %v3534, %v3535 : tensor<64x256x28x28xf32>
    %v3537 = stablehlo.divide %v3523, %v3512 : tensor<64x256x28x28xf32>
    %v3538 = stablehlo.multiply %v3537, %v3536 : tensor<64x256x28x28xf32>
    %v3539 = stablehlo.reshape %v3538 : (tensor<64x256x28x28xf32>) -> tensor<64x200704xf32>
    %v3540 = stablehlo.reshape %v3539 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3541 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x512x1x1xf32>
    %v3542 = stablehlo.transpose %v3541, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v3543 = stablehlo.convert %v3540 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3544 = stablehlo.convert %v3542 : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v3545 = stablehlo.convolution(%v3543, %v3544)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v3546 = stablehlo.convert %v3545 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v3547 = stablehlo.reshape %v3546 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3548 = stablehlo.reshape %v822 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3550 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3551 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3552 = stablehlo.reduce(%v3548 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3553 = stablehlo.broadcast_in_dim %v3552, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3554 = stablehlo.divide %v3553, %v3550 : tensor<64x1024x14x14xf32>
    %v3555 = stablehlo.subtract %v3548, %v3554 : tensor<64x1024x14x14xf32>
    %v3556 = stablehlo.multiply %v3555, %v3555 : tensor<64x1024x14x14xf32>
    %v3557 = stablehlo.reduce(%v3556 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3558 = stablehlo.broadcast_in_dim %v3557, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3559 = stablehlo.divide %v3558, %v3550 : tensor<64x1024x14x14xf32>
    %v3560 = stablehlo.add %v3559, %v3551 : tensor<64x1024x14x14xf32>
    %v3561 = stablehlo.rsqrt %v3560 : tensor<64x1024x14x14xf32>
    %v3562 = stablehlo.multiply %v3555, %v3561 : tensor<64x1024x14x14xf32>
    %v3563 = stablehlo.broadcast_in_dim %s3b0gp, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3564 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3565 = stablehlo.multiply %v3563, %v3564 : tensor<64x1024x14x14xf32>
    %v3566 = stablehlo.reduce(%v3565 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3567 = stablehlo.broadcast_in_dim %v3566, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3568 = stablehlo.multiply %v3562, %v3565 : tensor<64x1024x14x14xf32>
    %v3569 = stablehlo.reduce(%v3568 init: %v3549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3570 = stablehlo.broadcast_in_dim %v3569, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3571 = stablehlo.multiply %v3565, %v3550 : tensor<64x1024x14x14xf32>
    %v3572 = stablehlo.subtract %v3571, %v3567 : tensor<64x1024x14x14xf32>
    %v3573 = stablehlo.multiply %v3562, %v3570 : tensor<64x1024x14x14xf32>
    %v3574 = stablehlo.subtract %v3572, %v3573 : tensor<64x1024x14x14xf32>
    %v3575 = stablehlo.divide %v3561, %v3550 : tensor<64x1024x14x14xf32>
    %v3576 = stablehlo.multiply %v3575, %v3574 : tensor<64x1024x14x14xf32>
    %v3577 = stablehlo.reshape %v3576 : (tensor<64x1024x14x14xf32>) -> tensor<64x200704xf32>
    %v3578 = stablehlo.reshape %v3577 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3579 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3580 = stablehlo.pad %v3578, %v3579, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v3581 = stablehlo.reverse %s3b0Wp, dims = [2, 3] : tensor<1024x512x1x1xf32>
    %v3582 = stablehlo.transpose %v3581, dims = [1, 0, 2, 3] : (tensor<1024x512x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %v3583 = stablehlo.convert %v3580 : (tensor<64x1024x28x28xf32>) -> tensor<64x1024x28x28xbf16>
    %v3584 = stablehlo.convert %v3582 : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xbf16>
    %v3585 = stablehlo.convolution(%v3583, %v3584)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x1024x28x28xbf16>, tensor<512x1024x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v3586 = stablehlo.convert %v3585 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v3587 = stablehlo.reshape %v3586 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3588 = stablehlo.add %v3547, %v3587 : tensor<64x401408xf32>
    %v3589 = stablehlo.reshape %v726 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3590 = stablehlo.reshape %v3539 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3591 = stablehlo.transpose %v3589, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3592 = stablehlo.transpose %v3590, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3593 = stablehlo.convert %v3591 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3594 = stablehlo.convert %v3592 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3595 = stablehlo.convolution(%v3593, %v3594)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<512x256x1x1xbf16>
    %v3596 = stablehlo.convert %v3595 : (tensor<512x256x1x1xbf16>) -> tensor<512x256x1x1xf32>
    %v3597 = stablehlo.transpose %v3596, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v3598 = stablehlo.reshape %v734 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3599 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3600 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v3601 = stablehlo.reduce(%v3598 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3602 = stablehlo.broadcast_in_dim %v3601, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3603 = stablehlo.divide %v3602, %v3600 : tensor<64x256x28x28xf32>
    %v3604 = stablehlo.subtract %v3598, %v3603 : tensor<64x256x28x28xf32>
    %v3605 = stablehlo.multiply %v3604, %v3604 : tensor<64x256x28x28xf32>
    %v3606 = stablehlo.reduce(%v3605 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3607 = stablehlo.broadcast_in_dim %v3606, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v3608 = stablehlo.divide %v3607, %v3600 : tensor<64x256x28x28xf32>
    %v3609 = stablehlo.constant dense<1.0e-05> : tensor<64x256x28x28xf32>
    %v3610 = stablehlo.add %v3608, %v3609 : tensor<64x256x28x28xf32>
    %v3611 = stablehlo.rsqrt %v3610 : tensor<64x256x28x28xf32>
    %v3612 = stablehlo.multiply %v3604, %v3611 : tensor<64x256x28x28xf32>
    %v3613 = stablehlo.reshape %v3509 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3614 = stablehlo.multiply %v3613, %v3612 : tensor<64x256x28x28xf32>
    %v3615 = stablehlo.reduce(%v3614 init: %v3599) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3616 = stablehlo.reshape %v3509 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3618 = stablehlo.reduce(%v3616 init: %v3617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v3619 = stablehlo.reshape %v756 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v3620 = stablehlo.reshape %v3496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3621 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3622 = stablehlo.pad %v3620, %v3621, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3623 = stablehlo.transpose %v3619, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3624 = stablehlo.transpose %v3622, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3625 = stablehlo.convert %v3623 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3626 = stablehlo.convert %v3624 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3627 = stablehlo.convolution(%v3625, %v3626)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<256x256x3x3xbf16>
    %v3628 = stablehlo.convert %v3627 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3629 = stablehlo.transpose %v3628, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3630 = stablehlo.reshape %v764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3631 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3632 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3633 = stablehlo.reduce(%v3630 init: %v3631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3634 = stablehlo.broadcast_in_dim %v3633, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3635 = stablehlo.divide %v3634, %v3632 : tensor<64x256x14x14xf32>
    %v3636 = stablehlo.subtract %v3630, %v3635 : tensor<64x256x14x14xf32>
    %v3637 = stablehlo.multiply %v3636, %v3636 : tensor<64x256x14x14xf32>
    %v3638 = stablehlo.reduce(%v3637 init: %v3631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3639 = stablehlo.broadcast_in_dim %v3638, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3640 = stablehlo.divide %v3639, %v3632 : tensor<64x256x14x14xf32>
    %v3641 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3642 = stablehlo.add %v3640, %v3641 : tensor<64x256x14x14xf32>
    %v3643 = stablehlo.rsqrt %v3642 : tensor<64x256x14x14xf32>
    %v3644 = stablehlo.multiply %v3636, %v3643 : tensor<64x256x14x14xf32>
    %v3645 = stablehlo.reshape %v3466 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3646 = stablehlo.multiply %v3645, %v3644 : tensor<64x256x14x14xf32>
    %v3647 = stablehlo.reduce(%v3646 init: %v3631) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3648 = stablehlo.reshape %v3466 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3650 = stablehlo.reduce(%v3648 init: %v3649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3651 = stablehlo.reshape %v786 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3652 = stablehlo.reshape %v3455 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3653 = stablehlo.transpose %v3651, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3654 = stablehlo.transpose %v3652, dims = [1, 0, 2, 3] : (tensor<64x1024x14x14xf32>) -> tensor<1024x64x14x14xf32>
    %v3655 = stablehlo.convert %v3653 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3656 = stablehlo.convert %v3654 : (tensor<1024x64x14x14xf32>) -> tensor<1024x64x14x14xbf16>
    %v3657 = stablehlo.convolution(%v3655, %v3656)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<1024x64x14x14xbf16>) -> tensor<256x1024x1x1xbf16>
    %v3658 = stablehlo.convert %v3657 : (tensor<256x1024x1x1xbf16>) -> tensor<256x1024x1x1xf32>
    %v3659 = stablehlo.transpose %v3658, dims = [1, 0, 2, 3] : (tensor<256x1024x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %v3660 = stablehlo.reshape %v794 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3662 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3663 = stablehlo.reduce(%v3660 init: %v3661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3664 = stablehlo.broadcast_in_dim %v3663, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3665 = stablehlo.divide %v3664, %v3662 : tensor<64x1024x14x14xf32>
    %v3666 = stablehlo.subtract %v3660, %v3665 : tensor<64x1024x14x14xf32>
    %v3667 = stablehlo.multiply %v3666, %v3666 : tensor<64x1024x14x14xf32>
    %v3668 = stablehlo.reduce(%v3667 init: %v3661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3669 = stablehlo.broadcast_in_dim %v3668, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3670 = stablehlo.divide %v3669, %v3662 : tensor<64x1024x14x14xf32>
    %v3671 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3672 = stablehlo.add %v3670, %v3671 : tensor<64x1024x14x14xf32>
    %v3673 = stablehlo.rsqrt %v3672 : tensor<64x1024x14x14xf32>
    %v3674 = stablehlo.multiply %v3666, %v3673 : tensor<64x1024x14x14xf32>
    %v3675 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3676 = stablehlo.multiply %v3675, %v3674 : tensor<64x1024x14x14xf32>
    %v3677 = stablehlo.reduce(%v3676 init: %v3661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3678 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3679 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3680 = stablehlo.reduce(%v3678 init: %v3679) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3681 = stablehlo.reshape %v726 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3682 = stablehlo.reshape %v3577 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3684 = stablehlo.pad %v3682, %v3683, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<64x1024x28x28xf32>
    %v3685 = stablehlo.transpose %v3681, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3686 = stablehlo.transpose %v3684, dims = [1, 0, 2, 3] : (tensor<64x1024x28x28xf32>) -> tensor<1024x64x28x28xf32>
    %v3687 = stablehlo.convert %v3685 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3688 = stablehlo.convert %v3686 : (tensor<1024x64x28x28xf32>) -> tensor<1024x64x28x28xbf16>
    %v3689 = stablehlo.convolution(%v3687, %v3688)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<1024x64x28x28xbf16>) -> tensor<512x1024x1x1xbf16>
    %v3690 = stablehlo.convert %v3689 : (tensor<512x1024x1x1xbf16>) -> tensor<512x1024x1x1xf32>
    %v3691 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<512x1024x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %v3692 = stablehlo.reshape %v822 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3696 = stablehlo.broadcast_in_dim %v3695, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3697 = stablehlo.divide %v3696, %v3694 : tensor<64x1024x14x14xf32>
    %v3698 = stablehlo.subtract %v3692, %v3697 : tensor<64x1024x14x14xf32>
    %v3699 = stablehlo.multiply %v3698, %v3698 : tensor<64x1024x14x14xf32>
    %v3700 = stablehlo.reduce(%v3699 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v3702 = stablehlo.divide %v3701, %v3694 : tensor<64x1024x14x14xf32>
    %v3703 = stablehlo.constant dense<1.0e-05> : tensor<64x1024x14x14xf32>
    %v3704 = stablehlo.add %v3702, %v3703 : tensor<64x1024x14x14xf32>
    %v3705 = stablehlo.rsqrt %v3704 : tensor<64x1024x14x14xf32>
    %v3706 = stablehlo.multiply %v3698, %v3705 : tensor<64x1024x14x14xf32>
    %v3707 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3708 = stablehlo.multiply %v3707, %v3706 : tensor<64x1024x14x14xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3710 = stablehlo.reshape %v3425 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v3713 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v3714 = stablehlo.compare GT, %v724, %v3713 : (tensor<64x401408xf32>, tensor<64x401408xf32>) -> tensor<64x401408xi1>
    %v3715 = stablehlo.select %v3714, %v3588, %v3713 : tensor<64x401408xi1>, tensor<64x401408xf32>
    %v3716 = stablehlo.reshape %v703 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3717 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3718 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v3719 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v3720 = stablehlo.reduce(%v3716 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3720, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3722 = stablehlo.divide %v3721, %v3718 : tensor<64x512x28x28xf32>
    %v3723 = stablehlo.subtract %v3716, %v3722 : tensor<64x512x28x28xf32>
    %v3724 = stablehlo.multiply %v3723, %v3723 : tensor<64x512x28x28xf32>
    %v3725 = stablehlo.reduce(%v3724 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3726 = stablehlo.broadcast_in_dim %v3725, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3727 = stablehlo.divide %v3726, %v3718 : tensor<64x512x28x28xf32>
    %v3728 = stablehlo.add %v3727, %v3719 : tensor<64x512x28x28xf32>
    %v3729 = stablehlo.rsqrt %v3728 : tensor<64x512x28x28xf32>
    %v3730 = stablehlo.multiply %v3723, %v3729 : tensor<64x512x28x28xf32>
    %v3731 = stablehlo.broadcast_in_dim %s2b3g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3732 = stablehlo.reshape %v3715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3733 = stablehlo.multiply %v3731, %v3732 : tensor<64x512x28x28xf32>
    %v3734 = stablehlo.reduce(%v3733 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3735 = stablehlo.broadcast_in_dim %v3734, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3736 = stablehlo.multiply %v3730, %v3733 : tensor<64x512x28x28xf32>
    %v3737 = stablehlo.reduce(%v3736 init: %v3717) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3739 = stablehlo.multiply %v3733, %v3718 : tensor<64x512x28x28xf32>
    %v3740 = stablehlo.subtract %v3739, %v3735 : tensor<64x512x28x28xf32>
    %v3741 = stablehlo.multiply %v3730, %v3738 : tensor<64x512x28x28xf32>
    %v3742 = stablehlo.subtract %v3740, %v3741 : tensor<64x512x28x28xf32>
    %v3743 = stablehlo.divide %v3729, %v3718 : tensor<64x512x28x28xf32>
    %v3744 = stablehlo.multiply %v3743, %v3742 : tensor<64x512x28x28xf32>
    %v3745 = stablehlo.reshape %v3744 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3746 = stablehlo.reshape %v3745 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3747 = stablehlo.reverse %s2b3W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3748 = stablehlo.transpose %v3747, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3749 = stablehlo.convert %v3746 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v3750 = stablehlo.convert %v3748 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v3751 = stablehlo.convolution(%v3749, %v3750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v3752 = stablehlo.convert %v3751 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3753 = stablehlo.reshape %v3752 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3754 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v3755 = stablehlo.compare GT, %v693, %v3754 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v3756 = stablehlo.select %v3755, %v3753, %v3754 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v3757 = stablehlo.reshape %v673 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3759 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3760 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3761 = stablehlo.reduce(%v3757 init: %v3758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3762 = stablehlo.broadcast_in_dim %v3761, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3763 = stablehlo.divide %v3762, %v3759 : tensor<64x128x28x28xf32>
    %v3764 = stablehlo.subtract %v3757, %v3763 : tensor<64x128x28x28xf32>
    %v3765 = stablehlo.multiply %v3764, %v3764 : tensor<64x128x28x28xf32>
    %v3766 = stablehlo.reduce(%v3765 init: %v3758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3767 = stablehlo.broadcast_in_dim %v3766, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3768 = stablehlo.divide %v3767, %v3759 : tensor<64x128x28x28xf32>
    %v3769 = stablehlo.add %v3768, %v3760 : tensor<64x128x28x28xf32>
    %v3770 = stablehlo.rsqrt %v3769 : tensor<64x128x28x28xf32>
    %v3771 = stablehlo.multiply %v3764, %v3770 : tensor<64x128x28x28xf32>
    %v3772 = stablehlo.broadcast_in_dim %s2b3g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3773 = stablehlo.reshape %v3756 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3774 = stablehlo.multiply %v3772, %v3773 : tensor<64x128x28x28xf32>
    %v3775 = stablehlo.reduce(%v3774 init: %v3758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3776 = stablehlo.broadcast_in_dim %v3775, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3777 = stablehlo.multiply %v3771, %v3774 : tensor<64x128x28x28xf32>
    %v3778 = stablehlo.reduce(%v3777 init: %v3758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3779 = stablehlo.broadcast_in_dim %v3778, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3780 = stablehlo.multiply %v3774, %v3759 : tensor<64x128x28x28xf32>
    %v3781 = stablehlo.subtract %v3780, %v3776 : tensor<64x128x28x28xf32>
    %v3782 = stablehlo.multiply %v3771, %v3779 : tensor<64x128x28x28xf32>
    %v3783 = stablehlo.subtract %v3781, %v3782 : tensor<64x128x28x28xf32>
    %v3784 = stablehlo.divide %v3770, %v3759 : tensor<64x128x28x28xf32>
    %v3785 = stablehlo.multiply %v3784, %v3783 : tensor<64x128x28x28xf32>
    %v3786 = stablehlo.reshape %v3785 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3787 = stablehlo.reshape %v3786 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3788 = stablehlo.reverse %s2b3W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3789 = stablehlo.transpose %v3788, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3790 = stablehlo.convert %v3787 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3791 = stablehlo.convert %v3789 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3792 = stablehlo.convolution(%v3790, %v3791)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3793 = stablehlo.convert %v3792 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3794 = stablehlo.reshape %v3793 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3795 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v3796 = stablehlo.compare GT, %v663, %v3795 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v3797 = stablehlo.select %v3796, %v3794, %v3795 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v3798 = stablehlo.reshape %v643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3801 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3802 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3803 = stablehlo.broadcast_in_dim %v3802, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3804 = stablehlo.divide %v3803, %v3800 : tensor<64x128x28x28xf32>
    %v3805 = stablehlo.subtract %v3798, %v3804 : tensor<64x128x28x28xf32>
    %v3806 = stablehlo.multiply %v3805, %v3805 : tensor<64x128x28x28xf32>
    %v3807 = stablehlo.reduce(%v3806 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3808 = stablehlo.broadcast_in_dim %v3807, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3809 = stablehlo.divide %v3808, %v3800 : tensor<64x128x28x28xf32>
    %v3810 = stablehlo.add %v3809, %v3801 : tensor<64x128x28x28xf32>
    %v3811 = stablehlo.rsqrt %v3810 : tensor<64x128x28x28xf32>
    %v3812 = stablehlo.multiply %v3805, %v3811 : tensor<64x128x28x28xf32>
    %v3813 = stablehlo.broadcast_in_dim %s2b3g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3814 = stablehlo.reshape %v3797 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3815 = stablehlo.multiply %v3813, %v3814 : tensor<64x128x28x28xf32>
    %v3816 = stablehlo.reduce(%v3815 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3817 = stablehlo.broadcast_in_dim %v3816, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3818 = stablehlo.multiply %v3812, %v3815 : tensor<64x128x28x28xf32>
    %v3819 = stablehlo.reduce(%v3818 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3820 = stablehlo.broadcast_in_dim %v3819, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3821 = stablehlo.multiply %v3815, %v3800 : tensor<64x128x28x28xf32>
    %v3822 = stablehlo.subtract %v3821, %v3817 : tensor<64x128x28x28xf32>
    %v3823 = stablehlo.multiply %v3812, %v3820 : tensor<64x128x28x28xf32>
    %v3824 = stablehlo.subtract %v3822, %v3823 : tensor<64x128x28x28xf32>
    %v3825 = stablehlo.divide %v3811, %v3800 : tensor<64x128x28x28xf32>
    %v3826 = stablehlo.multiply %v3825, %v3824 : tensor<64x128x28x28xf32>
    %v3827 = stablehlo.reshape %v3826 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3828 = stablehlo.reshape %v3827 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3829 = stablehlo.reverse %s2b3W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v3830 = stablehlo.transpose %v3829, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3831 = stablehlo.convert %v3828 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3832 = stablehlo.convert %v3830 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v3833 = stablehlo.convolution(%v3831, %v3832)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v3834 = stablehlo.convert %v3833 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v3835 = stablehlo.reshape %v3834 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3836 = stablehlo.add %v3835, %v3715 : tensor<64x401408xf32>
    %v3837 = stablehlo.reshape %v635 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3838 = stablehlo.reshape %v3827 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3839 = stablehlo.transpose %v3837, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3840 = stablehlo.transpose %v3838, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3841 = stablehlo.convert %v3839 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3842 = stablehlo.convert %v3840 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3843 = stablehlo.convolution(%v3841, %v3842)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v3844 = stablehlo.convert %v3843 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v3845 = stablehlo.transpose %v3844, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3846 = stablehlo.reshape %v643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3849 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3850 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3851 = stablehlo.divide %v3850, %v3848 : tensor<64x128x28x28xf32>
    %v3852 = stablehlo.subtract %v3846, %v3851 : tensor<64x128x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<64x128x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3855 = stablehlo.broadcast_in_dim %v3854, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3856 = stablehlo.divide %v3855, %v3848 : tensor<64x128x28x28xf32>
    %v3857 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3858 = stablehlo.add %v3856, %v3857 : tensor<64x128x28x28xf32>
    %v3859 = stablehlo.rsqrt %v3858 : tensor<64x128x28x28xf32>
    %v3860 = stablehlo.multiply %v3852, %v3859 : tensor<64x128x28x28xf32>
    %v3861 = stablehlo.reshape %v3797 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3862 = stablehlo.multiply %v3861, %v3860 : tensor<64x128x28x28xf32>
    %v3863 = stablehlo.reduce(%v3862 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3864 = stablehlo.reshape %v3797 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3865 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3866 = stablehlo.reduce(%v3864 init: %v3865) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3867 = stablehlo.reshape %v665 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3868 = stablehlo.reshape %v3786 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3869 = stablehlo.transpose %v3867, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3870 = stablehlo.transpose %v3868, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3871 = stablehlo.convert %v3869 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3872 = stablehlo.convert %v3870 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3873 = stablehlo.convolution(%v3871, %v3872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3874 = stablehlo.convert %v3873 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3875 = stablehlo.transpose %v3874, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3876 = stablehlo.reshape %v673 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3877 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3878 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3879 = stablehlo.reduce(%v3876 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3880 = stablehlo.broadcast_in_dim %v3879, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3881 = stablehlo.divide %v3880, %v3878 : tensor<64x128x28x28xf32>
    %v3882 = stablehlo.subtract %v3876, %v3881 : tensor<64x128x28x28xf32>
    %v3883 = stablehlo.multiply %v3882, %v3882 : tensor<64x128x28x28xf32>
    %v3884 = stablehlo.reduce(%v3883 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3885 = stablehlo.broadcast_in_dim %v3884, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3886 = stablehlo.divide %v3885, %v3878 : tensor<64x128x28x28xf32>
    %v3887 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3888 = stablehlo.add %v3886, %v3887 : tensor<64x128x28x28xf32>
    %v3889 = stablehlo.rsqrt %v3888 : tensor<64x128x28x28xf32>
    %v3890 = stablehlo.multiply %v3882, %v3889 : tensor<64x128x28x28xf32>
    %v3891 = stablehlo.reshape %v3756 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3892 = stablehlo.multiply %v3891, %v3890 : tensor<64x128x28x28xf32>
    %v3893 = stablehlo.reduce(%v3892 init: %v3877) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3894 = stablehlo.reshape %v3756 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = stablehlo.reduce(%v3894 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3897 = stablehlo.reshape %v695 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3898 = stablehlo.reshape %v3745 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3899 = stablehlo.transpose %v3897, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3900 = stablehlo.transpose %v3898, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v3901 = stablehlo.convert %v3899 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3902 = stablehlo.convert %v3900 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v3903 = stablehlo.convolution(%v3901, %v3902)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v3904 = stablehlo.convert %v3903 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v3905 = stablehlo.transpose %v3904, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v3906 = stablehlo.reshape %v703 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3907 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3908 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v3909 = stablehlo.reduce(%v3906 init: %v3907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3910 = stablehlo.broadcast_in_dim %v3909, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3911 = stablehlo.divide %v3910, %v3908 : tensor<64x512x28x28xf32>
    %v3912 = stablehlo.subtract %v3906, %v3911 : tensor<64x512x28x28xf32>
    %v3913 = stablehlo.multiply %v3912, %v3912 : tensor<64x512x28x28xf32>
    %v3914 = stablehlo.reduce(%v3913 init: %v3907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3915 = stablehlo.broadcast_in_dim %v3914, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3916 = stablehlo.divide %v3915, %v3908 : tensor<64x512x28x28xf32>
    %v3917 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v3918 = stablehlo.add %v3916, %v3917 : tensor<64x512x28x28xf32>
    %v3919 = stablehlo.rsqrt %v3918 : tensor<64x512x28x28xf32>
    %v3920 = stablehlo.multiply %v3912, %v3919 : tensor<64x512x28x28xf32>
    %v3921 = stablehlo.reshape %v3715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3922 = stablehlo.multiply %v3921, %v3920 : tensor<64x512x28x28xf32>
    %v3923 = stablehlo.reduce(%v3922 init: %v3907) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3924 = stablehlo.reshape %v3715 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3925 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3926 = stablehlo.reduce(%v3924 init: %v3925) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v3928 = stablehlo.compare GT, %v633, %v3927 : (tensor<64x401408xf32>, tensor<64x401408xf32>) -> tensor<64x401408xi1>
    %v3929 = stablehlo.select %v3928, %v3836, %v3927 : tensor<64x401408xi1>, tensor<64x401408xf32>
    %v3930 = stablehlo.reshape %v612 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3931 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3932 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v3933 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v3934 = stablehlo.reduce(%v3930 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3935 = stablehlo.broadcast_in_dim %v3934, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3936 = stablehlo.divide %v3935, %v3932 : tensor<64x512x28x28xf32>
    %v3937 = stablehlo.subtract %v3930, %v3936 : tensor<64x512x28x28xf32>
    %v3938 = stablehlo.multiply %v3937, %v3937 : tensor<64x512x28x28xf32>
    %v3939 = stablehlo.reduce(%v3938 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3940 = stablehlo.broadcast_in_dim %v3939, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3941 = stablehlo.divide %v3940, %v3932 : tensor<64x512x28x28xf32>
    %v3942 = stablehlo.add %v3941, %v3933 : tensor<64x512x28x28xf32>
    %v3943 = stablehlo.rsqrt %v3942 : tensor<64x512x28x28xf32>
    %v3944 = stablehlo.multiply %v3937, %v3943 : tensor<64x512x28x28xf32>
    %v3945 = stablehlo.broadcast_in_dim %s2b2g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3946 = stablehlo.reshape %v3929 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3947 = stablehlo.multiply %v3945, %v3946 : tensor<64x512x28x28xf32>
    %v3948 = stablehlo.reduce(%v3947 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3949 = stablehlo.broadcast_in_dim %v3948, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3950 = stablehlo.multiply %v3944, %v3947 : tensor<64x512x28x28xf32>
    %v3951 = stablehlo.reduce(%v3950 init: %v3931) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v3952 = stablehlo.broadcast_in_dim %v3951, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v3953 = stablehlo.multiply %v3947, %v3932 : tensor<64x512x28x28xf32>
    %v3954 = stablehlo.subtract %v3953, %v3949 : tensor<64x512x28x28xf32>
    %v3955 = stablehlo.multiply %v3944, %v3952 : tensor<64x512x28x28xf32>
    %v3956 = stablehlo.subtract %v3954, %v3955 : tensor<64x512x28x28xf32>
    %v3957 = stablehlo.divide %v3943, %v3932 : tensor<64x512x28x28xf32>
    %v3958 = stablehlo.multiply %v3957, %v3956 : tensor<64x512x28x28xf32>
    %v3959 = stablehlo.reshape %v3958 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v3960 = stablehlo.reshape %v3959 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v3961 = stablehlo.reverse %s2b2W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v3962 = stablehlo.transpose %v3961, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v3963 = stablehlo.convert %v3960 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v3964 = stablehlo.convert %v3962 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v3965 = stablehlo.convolution(%v3963, %v3964)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v3966 = stablehlo.convert %v3965 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3967 = stablehlo.reshape %v3966 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3968 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v3969 = stablehlo.compare GT, %v602, %v3968 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v3970 = stablehlo.select %v3969, %v3967, %v3968 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v3971 = stablehlo.reshape %v582 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3972 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3973 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3974 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3975 = stablehlo.reduce(%v3971 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3976 = stablehlo.broadcast_in_dim %v3975, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3977 = stablehlo.divide %v3976, %v3973 : tensor<64x128x28x28xf32>
    %v3978 = stablehlo.subtract %v3971, %v3977 : tensor<64x128x28x28xf32>
    %v3979 = stablehlo.multiply %v3978, %v3978 : tensor<64x128x28x28xf32>
    %v3980 = stablehlo.reduce(%v3979 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3981 = stablehlo.broadcast_in_dim %v3980, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3982 = stablehlo.divide %v3981, %v3973 : tensor<64x128x28x28xf32>
    %v3983 = stablehlo.add %v3982, %v3974 : tensor<64x128x28x28xf32>
    %v3984 = stablehlo.rsqrt %v3983 : tensor<64x128x28x28xf32>
    %v3985 = stablehlo.multiply %v3978, %v3984 : tensor<64x128x28x28xf32>
    %v3986 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3987 = stablehlo.reshape %v3970 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3988 = stablehlo.multiply %v3986, %v3987 : tensor<64x128x28x28xf32>
    %v3989 = stablehlo.reduce(%v3988 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3990 = stablehlo.broadcast_in_dim %v3989, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3991 = stablehlo.multiply %v3985, %v3988 : tensor<64x128x28x28xf32>
    %v3992 = stablehlo.reduce(%v3991 init: %v3972) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3993 = stablehlo.broadcast_in_dim %v3992, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3994 = stablehlo.multiply %v3988, %v3973 : tensor<64x128x28x28xf32>
    %v3995 = stablehlo.subtract %v3994, %v3990 : tensor<64x128x28x28xf32>
    %v3996 = stablehlo.multiply %v3985, %v3993 : tensor<64x128x28x28xf32>
    %v3997 = stablehlo.subtract %v3995, %v3996 : tensor<64x128x28x28xf32>
    %v3998 = stablehlo.divide %v3984, %v3973 : tensor<64x128x28x28xf32>
    %v3999 = stablehlo.multiply %v3998, %v3997 : tensor<64x128x28x28xf32>
    %v4000 = stablehlo.reshape %v3999 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4001 = stablehlo.reshape %v4000 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4002 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4003 = stablehlo.transpose %v4002, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4004 = stablehlo.convert %v4001 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4005 = stablehlo.convert %v4003 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4006 = stablehlo.convolution(%v4004, %v4005)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v4007 = stablehlo.convert %v4006 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4008 = stablehlo.reshape %v4007 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4009 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v4010 = stablehlo.compare GT, %v572, %v4009 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v4011 = stablehlo.select %v4010, %v4008, %v4009 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v4012 = stablehlo.reshape %v552 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4013 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4014 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4015 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4016 = stablehlo.reduce(%v4012 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4017 = stablehlo.broadcast_in_dim %v4016, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4018 = stablehlo.divide %v4017, %v4014 : tensor<64x128x28x28xf32>
    %v4019 = stablehlo.subtract %v4012, %v4018 : tensor<64x128x28x28xf32>
    %v4020 = stablehlo.multiply %v4019, %v4019 : tensor<64x128x28x28xf32>
    %v4021 = stablehlo.reduce(%v4020 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4022 = stablehlo.broadcast_in_dim %v4021, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4023 = stablehlo.divide %v4022, %v4014 : tensor<64x128x28x28xf32>
    %v4024 = stablehlo.add %v4023, %v4015 : tensor<64x128x28x28xf32>
    %v4025 = stablehlo.rsqrt %v4024 : tensor<64x128x28x28xf32>
    %v4026 = stablehlo.multiply %v4019, %v4025 : tensor<64x128x28x28xf32>
    %v4027 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4028 = stablehlo.reshape %v4011 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4029 = stablehlo.multiply %v4027, %v4028 : tensor<64x128x28x28xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4031 = stablehlo.broadcast_in_dim %v4030, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4032 = stablehlo.multiply %v4026, %v4029 : tensor<64x128x28x28xf32>
    %v4033 = stablehlo.reduce(%v4032 init: %v4013) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4034 = stablehlo.broadcast_in_dim %v4033, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4035 = stablehlo.multiply %v4029, %v4014 : tensor<64x128x28x28xf32>
    %v4036 = stablehlo.subtract %v4035, %v4031 : tensor<64x128x28x28xf32>
    %v4037 = stablehlo.multiply %v4026, %v4034 : tensor<64x128x28x28xf32>
    %v4038 = stablehlo.subtract %v4036, %v4037 : tensor<64x128x28x28xf32>
    %v4039 = stablehlo.divide %v4025, %v4014 : tensor<64x128x28x28xf32>
    %v4040 = stablehlo.multiply %v4039, %v4038 : tensor<64x128x28x28xf32>
    %v4041 = stablehlo.reshape %v4040 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4042 = stablehlo.reshape %v4041 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4043 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4044 = stablehlo.transpose %v4043, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4045 = stablehlo.convert %v4042 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4046 = stablehlo.convert %v4044 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v4047 = stablehlo.convolution(%v4045, %v4046)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4048 = stablehlo.convert %v4047 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4049 = stablehlo.reshape %v4048 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4050 = stablehlo.add %v4049, %v3929 : tensor<64x401408xf32>
    %v4051 = stablehlo.reshape %v544 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4052 = stablehlo.reshape %v4041 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4053 = stablehlo.transpose %v4051, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4054 = stablehlo.transpose %v4052, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4055 = stablehlo.convert %v4053 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4056 = stablehlo.convert %v4054 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4057 = stablehlo.convolution(%v4055, %v4056)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v4058 = stablehlo.convert %v4057 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v4059 = stablehlo.transpose %v4058, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4060 = stablehlo.reshape %v552 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4061 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4062 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4063 = stablehlo.reduce(%v4060 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4064 = stablehlo.broadcast_in_dim %v4063, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4065 = stablehlo.divide %v4064, %v4062 : tensor<64x128x28x28xf32>
    %v4066 = stablehlo.subtract %v4060, %v4065 : tensor<64x128x28x28xf32>
    %v4067 = stablehlo.multiply %v4066, %v4066 : tensor<64x128x28x28xf32>
    %v4068 = stablehlo.reduce(%v4067 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4069 = stablehlo.broadcast_in_dim %v4068, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4070 = stablehlo.divide %v4069, %v4062 : tensor<64x128x28x28xf32>
    %v4071 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4072 = stablehlo.add %v4070, %v4071 : tensor<64x128x28x28xf32>
    %v4073 = stablehlo.rsqrt %v4072 : tensor<64x128x28x28xf32>
    %v4074 = stablehlo.multiply %v4066, %v4073 : tensor<64x128x28x28xf32>
    %v4075 = stablehlo.reshape %v4011 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4076 = stablehlo.multiply %v4075, %v4074 : tensor<64x128x28x28xf32>
    %v4077 = stablehlo.reduce(%v4076 init: %v4061) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4078 = stablehlo.reshape %v4011 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4079 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4080 = stablehlo.reduce(%v4078 init: %v4079) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4081 = stablehlo.reshape %v574 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4082 = stablehlo.reshape %v4000 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4083 = stablehlo.transpose %v4081, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4084 = stablehlo.transpose %v4082, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4085 = stablehlo.convert %v4083 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4086 = stablehlo.convert %v4084 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4087 = stablehlo.convolution(%v4085, %v4086)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4088 = stablehlo.convert %v4087 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4089 = stablehlo.transpose %v4088, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4090 = stablehlo.reshape %v582 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4091 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4092 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4093 = stablehlo.reduce(%v4090 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4094 = stablehlo.broadcast_in_dim %v4093, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4095 = stablehlo.divide %v4094, %v4092 : tensor<64x128x28x28xf32>
    %v4096 = stablehlo.subtract %v4090, %v4095 : tensor<64x128x28x28xf32>
    %v4097 = stablehlo.multiply %v4096, %v4096 : tensor<64x128x28x28xf32>
    %v4098 = stablehlo.reduce(%v4097 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4099 = stablehlo.broadcast_in_dim %v4098, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4100 = stablehlo.divide %v4099, %v4092 : tensor<64x128x28x28xf32>
    %v4101 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4102 = stablehlo.add %v4100, %v4101 : tensor<64x128x28x28xf32>
    %v4103 = stablehlo.rsqrt %v4102 : tensor<64x128x28x28xf32>
    %v4104 = stablehlo.multiply %v4096, %v4103 : tensor<64x128x28x28xf32>
    %v4105 = stablehlo.reshape %v3970 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4106 = stablehlo.multiply %v4105, %v4104 : tensor<64x128x28x28xf32>
    %v4107 = stablehlo.reduce(%v4106 init: %v4091) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4108 = stablehlo.reshape %v3970 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4109 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4110 = stablehlo.reduce(%v4108 init: %v4109) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4111 = stablehlo.reshape %v604 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4112 = stablehlo.reshape %v3959 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4113 = stablehlo.transpose %v4111, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4114 = stablehlo.transpose %v4112, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4115 = stablehlo.convert %v4113 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4116 = stablehlo.convert %v4114 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4117 = stablehlo.convolution(%v4115, %v4116)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4118 = stablehlo.convert %v4117 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4119 = stablehlo.transpose %v4118, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4120 = stablehlo.reshape %v612 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4122 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4123 = stablehlo.reduce(%v4120 init: %v4121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4124 = stablehlo.broadcast_in_dim %v4123, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4125 = stablehlo.divide %v4124, %v4122 : tensor<64x512x28x28xf32>
    %v4126 = stablehlo.subtract %v4120, %v4125 : tensor<64x512x28x28xf32>
    %v4127 = stablehlo.multiply %v4126, %v4126 : tensor<64x512x28x28xf32>
    %v4128 = stablehlo.reduce(%v4127 init: %v4121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4129 = stablehlo.broadcast_in_dim %v4128, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4130 = stablehlo.divide %v4129, %v4122 : tensor<64x512x28x28xf32>
    %v4131 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4132 = stablehlo.add %v4130, %v4131 : tensor<64x512x28x28xf32>
    %v4133 = stablehlo.rsqrt %v4132 : tensor<64x512x28x28xf32>
    %v4134 = stablehlo.multiply %v4126, %v4133 : tensor<64x512x28x28xf32>
    %v4135 = stablehlo.reshape %v3929 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4136 = stablehlo.multiply %v4135, %v4134 : tensor<64x512x28x28xf32>
    %v4137 = stablehlo.reduce(%v4136 init: %v4121) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4138 = stablehlo.reshape %v3929 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4139 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4140 = stablehlo.reduce(%v4138 init: %v4139) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4141 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v4142 = stablehlo.compare GT, %v542, %v4141 : (tensor<64x401408xf32>, tensor<64x401408xf32>) -> tensor<64x401408xi1>
    %v4143 = stablehlo.select %v4142, %v4050, %v4141 : tensor<64x401408xi1>, tensor<64x401408xf32>
    %v4144 = stablehlo.reshape %v521 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4145 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4146 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4147 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4148 = stablehlo.reduce(%v4144 init: %v4145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4149 = stablehlo.broadcast_in_dim %v4148, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4150 = stablehlo.divide %v4149, %v4146 : tensor<64x512x28x28xf32>
    %v4151 = stablehlo.subtract %v4144, %v4150 : tensor<64x512x28x28xf32>
    %v4152 = stablehlo.multiply %v4151, %v4151 : tensor<64x512x28x28xf32>
    %v4153 = stablehlo.reduce(%v4152 init: %v4145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4154 = stablehlo.broadcast_in_dim %v4153, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4155 = stablehlo.divide %v4154, %v4146 : tensor<64x512x28x28xf32>
    %v4156 = stablehlo.add %v4155, %v4147 : tensor<64x512x28x28xf32>
    %v4157 = stablehlo.rsqrt %v4156 : tensor<64x512x28x28xf32>
    %v4158 = stablehlo.multiply %v4151, %v4157 : tensor<64x512x28x28xf32>
    %v4159 = stablehlo.broadcast_in_dim %s2b1g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4160 = stablehlo.reshape %v4143 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4161 = stablehlo.multiply %v4159, %v4160 : tensor<64x512x28x28xf32>
    %v4162 = stablehlo.reduce(%v4161 init: %v4145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4163 = stablehlo.broadcast_in_dim %v4162, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4164 = stablehlo.multiply %v4158, %v4161 : tensor<64x512x28x28xf32>
    %v4165 = stablehlo.reduce(%v4164 init: %v4145) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4166 = stablehlo.broadcast_in_dim %v4165, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4167 = stablehlo.multiply %v4161, %v4146 : tensor<64x512x28x28xf32>
    %v4168 = stablehlo.subtract %v4167, %v4163 : tensor<64x512x28x28xf32>
    %v4169 = stablehlo.multiply %v4158, %v4166 : tensor<64x512x28x28xf32>
    %v4170 = stablehlo.subtract %v4168, %v4169 : tensor<64x512x28x28xf32>
    %v4171 = stablehlo.divide %v4157, %v4146 : tensor<64x512x28x28xf32>
    %v4172 = stablehlo.multiply %v4171, %v4170 : tensor<64x512x28x28xf32>
    %v4173 = stablehlo.reshape %v4172 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4174 = stablehlo.reshape %v4173 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4175 = stablehlo.reverse %s2b1W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4176 = stablehlo.transpose %v4175, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4177 = stablehlo.convert %v4174 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4178 = stablehlo.convert %v4176 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4179 = stablehlo.convolution(%v4177, %v4178)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4180 = stablehlo.convert %v4179 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4181 = stablehlo.reshape %v4180 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4182 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v4183 = stablehlo.compare GT, %v511, %v4182 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v4184 = stablehlo.select %v4183, %v4181, %v4182 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v4185 = stablehlo.reshape %v491 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4186 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4187 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4188 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4189 = stablehlo.reduce(%v4185 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4190 = stablehlo.broadcast_in_dim %v4189, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4191 = stablehlo.divide %v4190, %v4187 : tensor<64x128x28x28xf32>
    %v4192 = stablehlo.subtract %v4185, %v4191 : tensor<64x128x28x28xf32>
    %v4193 = stablehlo.multiply %v4192, %v4192 : tensor<64x128x28x28xf32>
    %v4194 = stablehlo.reduce(%v4193 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4195 = stablehlo.broadcast_in_dim %v4194, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4196 = stablehlo.divide %v4195, %v4187 : tensor<64x128x28x28xf32>
    %v4197 = stablehlo.add %v4196, %v4188 : tensor<64x128x28x28xf32>
    %v4198 = stablehlo.rsqrt %v4197 : tensor<64x128x28x28xf32>
    %v4199 = stablehlo.multiply %v4192, %v4198 : tensor<64x128x28x28xf32>
    %v4200 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4201 = stablehlo.reshape %v4184 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4202 = stablehlo.multiply %v4200, %v4201 : tensor<64x128x28x28xf32>
    %v4203 = stablehlo.reduce(%v4202 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4204 = stablehlo.broadcast_in_dim %v4203, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4205 = stablehlo.multiply %v4199, %v4202 : tensor<64x128x28x28xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4186) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4207 = stablehlo.broadcast_in_dim %v4206, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4208 = stablehlo.multiply %v4202, %v4187 : tensor<64x128x28x28xf32>
    %v4209 = stablehlo.subtract %v4208, %v4204 : tensor<64x128x28x28xf32>
    %v4210 = stablehlo.multiply %v4199, %v4207 : tensor<64x128x28x28xf32>
    %v4211 = stablehlo.subtract %v4209, %v4210 : tensor<64x128x28x28xf32>
    %v4212 = stablehlo.divide %v4198, %v4187 : tensor<64x128x28x28xf32>
    %v4213 = stablehlo.multiply %v4212, %v4211 : tensor<64x128x28x28xf32>
    %v4214 = stablehlo.reshape %v4213 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4215 = stablehlo.reshape %v4214 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4216 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4217 = stablehlo.transpose %v4216, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4218 = stablehlo.convert %v4215 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4219 = stablehlo.convert %v4217 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4220 = stablehlo.convolution(%v4218, %v4219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v4221 = stablehlo.convert %v4220 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4222 = stablehlo.reshape %v4221 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4223 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v4224 = stablehlo.compare GT, %v481, %v4223 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v4225 = stablehlo.select %v4224, %v4222, %v4223 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v4226 = stablehlo.reshape %v461 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4227 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4228 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4229 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4230 = stablehlo.reduce(%v4226 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4231 = stablehlo.broadcast_in_dim %v4230, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4232 = stablehlo.divide %v4231, %v4228 : tensor<64x128x28x28xf32>
    %v4233 = stablehlo.subtract %v4226, %v4232 : tensor<64x128x28x28xf32>
    %v4234 = stablehlo.multiply %v4233, %v4233 : tensor<64x128x28x28xf32>
    %v4235 = stablehlo.reduce(%v4234 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4236 = stablehlo.broadcast_in_dim %v4235, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4237 = stablehlo.divide %v4236, %v4228 : tensor<64x128x28x28xf32>
    %v4238 = stablehlo.add %v4237, %v4229 : tensor<64x128x28x28xf32>
    %v4239 = stablehlo.rsqrt %v4238 : tensor<64x128x28x28xf32>
    %v4240 = stablehlo.multiply %v4233, %v4239 : tensor<64x128x28x28xf32>
    %v4241 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4242 = stablehlo.reshape %v4225 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4243 = stablehlo.multiply %v4241, %v4242 : tensor<64x128x28x28xf32>
    %v4244 = stablehlo.reduce(%v4243 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4245 = stablehlo.broadcast_in_dim %v4244, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4246 = stablehlo.multiply %v4240, %v4243 : tensor<64x128x28x28xf32>
    %v4247 = stablehlo.reduce(%v4246 init: %v4227) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4248 = stablehlo.broadcast_in_dim %v4247, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4249 = stablehlo.multiply %v4243, %v4228 : tensor<64x128x28x28xf32>
    %v4250 = stablehlo.subtract %v4249, %v4245 : tensor<64x128x28x28xf32>
    %v4251 = stablehlo.multiply %v4240, %v4248 : tensor<64x128x28x28xf32>
    %v4252 = stablehlo.subtract %v4250, %v4251 : tensor<64x128x28x28xf32>
    %v4253 = stablehlo.divide %v4239, %v4228 : tensor<64x128x28x28xf32>
    %v4254 = stablehlo.multiply %v4253, %v4252 : tensor<64x128x28x28xf32>
    %v4255 = stablehlo.reshape %v4254 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4256 = stablehlo.reshape %v4255 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4257 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x512x1x1xf32>
    %v4258 = stablehlo.transpose %v4257, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4259 = stablehlo.convert %v4256 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v4260 = stablehlo.convert %v4258 : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xbf16>
    %v4261 = stablehlo.convolution(%v4259, %v4260)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<512x128x1x1xbf16>) -> tensor<64x512x28x28xbf16>
    %v4262 = stablehlo.convert %v4261 : (tensor<64x512x28x28xbf16>) -> tensor<64x512x28x28xf32>
    %v4263 = stablehlo.reshape %v4262 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4264 = stablehlo.add %v4263, %v4143 : tensor<64x401408xf32>
    %v4265 = stablehlo.reshape %v453 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4266 = stablehlo.reshape %v4255 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4267 = stablehlo.transpose %v4265, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4268 = stablehlo.transpose %v4266, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4269 = stablehlo.convert %v4267 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4270 = stablehlo.convert %v4268 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4271 = stablehlo.convolution(%v4269, %v4270)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<512x128x1x1xbf16>
    %v4272 = stablehlo.convert %v4271 : (tensor<512x128x1x1xbf16>) -> tensor<512x128x1x1xf32>
    %v4273 = stablehlo.transpose %v4272, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4274 = stablehlo.reshape %v461 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4276 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4277 = stablehlo.reduce(%v4274 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4278 = stablehlo.broadcast_in_dim %v4277, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4279 = stablehlo.divide %v4278, %v4276 : tensor<64x128x28x28xf32>
    %v4280 = stablehlo.subtract %v4274, %v4279 : tensor<64x128x28x28xf32>
    %v4281 = stablehlo.multiply %v4280, %v4280 : tensor<64x128x28x28xf32>
    %v4282 = stablehlo.reduce(%v4281 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4283 = stablehlo.broadcast_in_dim %v4282, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4284 = stablehlo.divide %v4283, %v4276 : tensor<64x128x28x28xf32>
    %v4285 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4286 = stablehlo.add %v4284, %v4285 : tensor<64x128x28x28xf32>
    %v4287 = stablehlo.rsqrt %v4286 : tensor<64x128x28x28xf32>
    %v4288 = stablehlo.multiply %v4280, %v4287 : tensor<64x128x28x28xf32>
    %v4289 = stablehlo.reshape %v4225 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4290 = stablehlo.multiply %v4289, %v4288 : tensor<64x128x28x28xf32>
    %v4291 = stablehlo.reduce(%v4290 init: %v4275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4292 = stablehlo.reshape %v4225 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4294 = stablehlo.reduce(%v4292 init: %v4293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4295 = stablehlo.reshape %v483 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4296 = stablehlo.reshape %v4214 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4297 = stablehlo.transpose %v4295, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4298 = stablehlo.transpose %v4296, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4299 = stablehlo.convert %v4297 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4300 = stablehlo.convert %v4298 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4301 = stablehlo.convolution(%v4299, %v4300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4302 = stablehlo.convert %v4301 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4303 = stablehlo.transpose %v4302, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4304 = stablehlo.reshape %v491 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4306 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4307 = stablehlo.reduce(%v4304 init: %v4305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4308 = stablehlo.broadcast_in_dim %v4307, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4309 = stablehlo.divide %v4308, %v4306 : tensor<64x128x28x28xf32>
    %v4310 = stablehlo.subtract %v4304, %v4309 : tensor<64x128x28x28xf32>
    %v4311 = stablehlo.multiply %v4310, %v4310 : tensor<64x128x28x28xf32>
    %v4312 = stablehlo.reduce(%v4311 init: %v4305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4313 = stablehlo.broadcast_in_dim %v4312, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4314 = stablehlo.divide %v4313, %v4306 : tensor<64x128x28x28xf32>
    %v4315 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4316 = stablehlo.add %v4314, %v4315 : tensor<64x128x28x28xf32>
    %v4317 = stablehlo.rsqrt %v4316 : tensor<64x128x28x28xf32>
    %v4318 = stablehlo.multiply %v4310, %v4317 : tensor<64x128x28x28xf32>
    %v4319 = stablehlo.reshape %v4184 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4320 = stablehlo.multiply %v4319, %v4318 : tensor<64x128x28x28xf32>
    %v4321 = stablehlo.reduce(%v4320 init: %v4305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4322 = stablehlo.reshape %v4184 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4324 = stablehlo.reduce(%v4322 init: %v4323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4325 = stablehlo.reshape %v513 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4326 = stablehlo.reshape %v4173 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4327 = stablehlo.transpose %v4325, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4328 = stablehlo.transpose %v4326, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4329 = stablehlo.convert %v4327 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4330 = stablehlo.convert %v4328 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4331 = stablehlo.convolution(%v4329, %v4330)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4332 = stablehlo.convert %v4331 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4333 = stablehlo.transpose %v4332, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4334 = stablehlo.reshape %v521 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4336 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4337 = stablehlo.reduce(%v4334 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4338 = stablehlo.broadcast_in_dim %v4337, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4339 = stablehlo.divide %v4338, %v4336 : tensor<64x512x28x28xf32>
    %v4340 = stablehlo.subtract %v4334, %v4339 : tensor<64x512x28x28xf32>
    %v4341 = stablehlo.multiply %v4340, %v4340 : tensor<64x512x28x28xf32>
    %v4342 = stablehlo.reduce(%v4341 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4343 = stablehlo.broadcast_in_dim %v4342, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4344 = stablehlo.divide %v4343, %v4336 : tensor<64x512x28x28xf32>
    %v4345 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4346 = stablehlo.add %v4344, %v4345 : tensor<64x512x28x28xf32>
    %v4347 = stablehlo.rsqrt %v4346 : tensor<64x512x28x28xf32>
    %v4348 = stablehlo.multiply %v4340, %v4347 : tensor<64x512x28x28xf32>
    %v4349 = stablehlo.reshape %v4143 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4350 = stablehlo.multiply %v4349, %v4348 : tensor<64x512x28x28xf32>
    %v4351 = stablehlo.reduce(%v4350 init: %v4335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4352 = stablehlo.reshape %v4143 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4354 = stablehlo.reduce(%v4352 init: %v4353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4355 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v4356 = stablehlo.compare GT, %v451, %v4355 : (tensor<64x401408xf32>, tensor<64x401408xf32>) -> tensor<64x401408xi1>
    %v4357 = stablehlo.select %v4356, %v4264, %v4355 : tensor<64x401408xi1>, tensor<64x401408xf32>
    %v4358 = stablehlo.reshape %v402 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4359 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4360 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4361 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4362 = stablehlo.reduce(%v4358 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4363 = stablehlo.broadcast_in_dim %v4362, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4364 = stablehlo.divide %v4363, %v4360 : tensor<64x512x28x28xf32>
    %v4365 = stablehlo.subtract %v4358, %v4364 : tensor<64x512x28x28xf32>
    %v4366 = stablehlo.multiply %v4365, %v4365 : tensor<64x512x28x28xf32>
    %v4367 = stablehlo.reduce(%v4366 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4368 = stablehlo.broadcast_in_dim %v4367, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4369 = stablehlo.divide %v4368, %v4360 : tensor<64x512x28x28xf32>
    %v4370 = stablehlo.add %v4369, %v4361 : tensor<64x512x28x28xf32>
    %v4371 = stablehlo.rsqrt %v4370 : tensor<64x512x28x28xf32>
    %v4372 = stablehlo.multiply %v4365, %v4371 : tensor<64x512x28x28xf32>
    %v4373 = stablehlo.broadcast_in_dim %s2b0g3, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4374 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4375 = stablehlo.multiply %v4373, %v4374 : tensor<64x512x28x28xf32>
    %v4376 = stablehlo.reduce(%v4375 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4377 = stablehlo.broadcast_in_dim %v4376, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4378 = stablehlo.multiply %v4372, %v4375 : tensor<64x512x28x28xf32>
    %v4379 = stablehlo.reduce(%v4378 init: %v4359) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4380 = stablehlo.broadcast_in_dim %v4379, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4381 = stablehlo.multiply %v4375, %v4360 : tensor<64x512x28x28xf32>
    %v4382 = stablehlo.subtract %v4381, %v4377 : tensor<64x512x28x28xf32>
    %v4383 = stablehlo.multiply %v4372, %v4380 : tensor<64x512x28x28xf32>
    %v4384 = stablehlo.subtract %v4382, %v4383 : tensor<64x512x28x28xf32>
    %v4385 = stablehlo.divide %v4371, %v4360 : tensor<64x512x28x28xf32>
    %v4386 = stablehlo.multiply %v4385, %v4384 : tensor<64x512x28x28xf32>
    %v4387 = stablehlo.reshape %v4386 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4388 = stablehlo.reshape %v4387 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4389 = stablehlo.reverse %s2b0W3, dims = [2, 3] : tensor<512x128x1x1xf32>
    %v4390 = stablehlo.transpose %v4389, dims = [1, 0, 2, 3] : (tensor<512x128x1x1xf32>) -> tensor<128x512x1x1xf32>
    %v4391 = stablehlo.convert %v4388 : (tensor<64x512x28x28xf32>) -> tensor<64x512x28x28xbf16>
    %v4392 = stablehlo.convert %v4390 : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xbf16>
    %v4393 = stablehlo.convolution(%v4391, %v4392)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x28x28xbf16>, tensor<128x512x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v4394 = stablehlo.convert %v4393 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v4395 = stablehlo.reshape %v4394 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4396 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v4397 = stablehlo.compare GT, %v392, %v4396 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v4398 = stablehlo.select %v4397, %v4395, %v4396 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v4399 = stablehlo.reshape %v372 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4400 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4401 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4402 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4403 = stablehlo.reduce(%v4399 init: %v4400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4404 = stablehlo.broadcast_in_dim %v4403, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4405 = stablehlo.divide %v4404, %v4401 : tensor<64x128x28x28xf32>
    %v4406 = stablehlo.subtract %v4399, %v4405 : tensor<64x128x28x28xf32>
    %v4407 = stablehlo.multiply %v4406, %v4406 : tensor<64x128x28x28xf32>
    %v4408 = stablehlo.reduce(%v4407 init: %v4400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4409 = stablehlo.broadcast_in_dim %v4408, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4410 = stablehlo.divide %v4409, %v4401 : tensor<64x128x28x28xf32>
    %v4411 = stablehlo.add %v4410, %v4402 : tensor<64x128x28x28xf32>
    %v4412 = stablehlo.rsqrt %v4411 : tensor<64x128x28x28xf32>
    %v4413 = stablehlo.multiply %v4406, %v4412 : tensor<64x128x28x28xf32>
    %v4414 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4415 = stablehlo.reshape %v4398 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4416 = stablehlo.multiply %v4414, %v4415 : tensor<64x128x28x28xf32>
    %v4417 = stablehlo.reduce(%v4416 init: %v4400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4418 = stablehlo.broadcast_in_dim %v4417, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4419 = stablehlo.multiply %v4413, %v4416 : tensor<64x128x28x28xf32>
    %v4420 = stablehlo.reduce(%v4419 init: %v4400) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4421 = stablehlo.broadcast_in_dim %v4420, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4422 = stablehlo.multiply %v4416, %v4401 : tensor<64x128x28x28xf32>
    %v4423 = stablehlo.subtract %v4422, %v4418 : tensor<64x128x28x28xf32>
    %v4424 = stablehlo.multiply %v4413, %v4421 : tensor<64x128x28x28xf32>
    %v4425 = stablehlo.subtract %v4423, %v4424 : tensor<64x128x28x28xf32>
    %v4426 = stablehlo.divide %v4412, %v4401 : tensor<64x128x28x28xf32>
    %v4427 = stablehlo.multiply %v4426, %v4425 : tensor<64x128x28x28xf32>
    %v4428 = stablehlo.reshape %v4427 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4429 = stablehlo.reshape %v4428 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4431 = stablehlo.pad %v4429, %v4430, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4432 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v4433 = stablehlo.transpose %v4432, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4434 = stablehlo.convert %v4431 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4435 = stablehlo.convert %v4433 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v4436 = stablehlo.convolution(%v4434, %v4435)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x56x56xbf16>
    %v4437 = stablehlo.convert %v4436 : (tensor<64x128x56x56xbf16>) -> tensor<64x128x56x56xf32>
    %v4438 = stablehlo.reshape %v4437 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4439 = stablehlo.constant dense<0.0> : tensor<64x401408xf32>
    %v4440 = stablehlo.compare GT, %v362, %v4439 : (tensor<64x401408xf32>, tensor<64x401408xf32>) -> tensor<64x401408xi1>
    %v4441 = stablehlo.select %v4440, %v4438, %v4439 : tensor<64x401408xi1>, tensor<64x401408xf32>
    %v4442 = stablehlo.reshape %v342 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4443 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4444 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v4445 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v4446 = stablehlo.reduce(%v4442 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4447 = stablehlo.broadcast_in_dim %v4446, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4448 = stablehlo.divide %v4447, %v4444 : tensor<64x128x56x56xf32>
    %v4449 = stablehlo.subtract %v4442, %v4448 : tensor<64x128x56x56xf32>
    %v4450 = stablehlo.multiply %v4449, %v4449 : tensor<64x128x56x56xf32>
    %v4451 = stablehlo.reduce(%v4450 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4452 = stablehlo.broadcast_in_dim %v4451, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4453 = stablehlo.divide %v4452, %v4444 : tensor<64x128x56x56xf32>
    %v4454 = stablehlo.add %v4453, %v4445 : tensor<64x128x56x56xf32>
    %v4455 = stablehlo.rsqrt %v4454 : tensor<64x128x56x56xf32>
    %v4456 = stablehlo.multiply %v4449, %v4455 : tensor<64x128x56x56xf32>
    %v4457 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4458 = stablehlo.reshape %v4441 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4459 = stablehlo.multiply %v4457, %v4458 : tensor<64x128x56x56xf32>
    %v4460 = stablehlo.reduce(%v4459 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4461 = stablehlo.broadcast_in_dim %v4460, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4462 = stablehlo.multiply %v4456, %v4459 : tensor<64x128x56x56xf32>
    %v4463 = stablehlo.reduce(%v4462 init: %v4443) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4464 = stablehlo.broadcast_in_dim %v4463, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4465 = stablehlo.multiply %v4459, %v4444 : tensor<64x128x56x56xf32>
    %v4466 = stablehlo.subtract %v4465, %v4461 : tensor<64x128x56x56xf32>
    %v4467 = stablehlo.multiply %v4456, %v4464 : tensor<64x128x56x56xf32>
    %v4468 = stablehlo.subtract %v4466, %v4467 : tensor<64x128x56x56xf32>
    %v4469 = stablehlo.divide %v4455, %v4444 : tensor<64x128x56x56xf32>
    %v4470 = stablehlo.multiply %v4469, %v4468 : tensor<64x128x56x56xf32>
    %v4471 = stablehlo.reshape %v4470 : (tensor<64x128x56x56xf32>) -> tensor<64x401408xf32>
    %v4472 = stablehlo.reshape %v4471 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4473 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x256x1x1xf32>
    %v4474 = stablehlo.transpose %v4473, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v4475 = stablehlo.convert %v4472 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4476 = stablehlo.convert %v4474 : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v4477 = stablehlo.convolution(%v4475, %v4476)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<256x128x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4478 = stablehlo.convert %v4477 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4479 = stablehlo.reshape %v4478 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4480 = stablehlo.reshape %v430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4481 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4482 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4483 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4484 = stablehlo.reduce(%v4480 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4485 = stablehlo.broadcast_in_dim %v4484, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4486 = stablehlo.divide %v4485, %v4482 : tensor<64x512x28x28xf32>
    %v4487 = stablehlo.subtract %v4480, %v4486 : tensor<64x512x28x28xf32>
    %v4488 = stablehlo.multiply %v4487, %v4487 : tensor<64x512x28x28xf32>
    %v4489 = stablehlo.reduce(%v4488 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4490 = stablehlo.broadcast_in_dim %v4489, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4491 = stablehlo.divide %v4490, %v4482 : tensor<64x512x28x28xf32>
    %v4492 = stablehlo.add %v4491, %v4483 : tensor<64x512x28x28xf32>
    %v4493 = stablehlo.rsqrt %v4492 : tensor<64x512x28x28xf32>
    %v4494 = stablehlo.multiply %v4487, %v4493 : tensor<64x512x28x28xf32>
    %v4495 = stablehlo.broadcast_in_dim %s2b0gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4496 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4497 = stablehlo.multiply %v4495, %v4496 : tensor<64x512x28x28xf32>
    %v4498 = stablehlo.reduce(%v4497 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4499 = stablehlo.broadcast_in_dim %v4498, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4500 = stablehlo.multiply %v4494, %v4497 : tensor<64x512x28x28xf32>
    %v4501 = stablehlo.reduce(%v4500 init: %v4481) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4502 = stablehlo.broadcast_in_dim %v4501, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4503 = stablehlo.multiply %v4497, %v4482 : tensor<64x512x28x28xf32>
    %v4504 = stablehlo.subtract %v4503, %v4499 : tensor<64x512x28x28xf32>
    %v4505 = stablehlo.multiply %v4494, %v4502 : tensor<64x512x28x28xf32>
    %v4506 = stablehlo.subtract %v4504, %v4505 : tensor<64x512x28x28xf32>
    %v4507 = stablehlo.divide %v4493, %v4482 : tensor<64x512x28x28xf32>
    %v4508 = stablehlo.multiply %v4507, %v4506 : tensor<64x512x28x28xf32>
    %v4509 = stablehlo.reshape %v4508 : (tensor<64x512x28x28xf32>) -> tensor<64x401408xf32>
    %v4510 = stablehlo.reshape %v4509 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4512 = stablehlo.pad %v4510, %v4511, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v4513 = stablehlo.reverse %s2b0Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v4514 = stablehlo.transpose %v4513, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v4515 = stablehlo.convert %v4512 : (tensor<64x512x56x56xf32>) -> tensor<64x512x56x56xbf16>
    %v4516 = stablehlo.convert %v4514 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v4517 = stablehlo.convolution(%v4515, %v4516)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x56x56xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4518 = stablehlo.convert %v4517 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4519 = stablehlo.reshape %v4518 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4520 = stablehlo.add %v4479, %v4519 : tensor<64x802816xf32>
    %v4521 = stablehlo.reshape %v334 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4522 = stablehlo.reshape %v4471 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4523 = stablehlo.transpose %v4521, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4524 = stablehlo.transpose %v4522, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4525 = stablehlo.convert %v4523 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4526 = stablehlo.convert %v4524 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4527 = stablehlo.convolution(%v4525, %v4526)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<256x128x1x1xbf16>
    %v4528 = stablehlo.convert %v4527 : (tensor<256x128x1x1xbf16>) -> tensor<256x128x1x1xf32>
    %v4529 = stablehlo.transpose %v4528, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v4530 = stablehlo.reshape %v342 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4531 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4532 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v4533 = stablehlo.reduce(%v4530 init: %v4531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4534 = stablehlo.broadcast_in_dim %v4533, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4535 = stablehlo.divide %v4534, %v4532 : tensor<64x128x56x56xf32>
    %v4536 = stablehlo.subtract %v4530, %v4535 : tensor<64x128x56x56xf32>
    %v4537 = stablehlo.multiply %v4536, %v4536 : tensor<64x128x56x56xf32>
    %v4538 = stablehlo.reduce(%v4537 init: %v4531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4539 = stablehlo.broadcast_in_dim %v4538, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v4540 = stablehlo.divide %v4539, %v4532 : tensor<64x128x56x56xf32>
    %v4541 = stablehlo.constant dense<1.0e-05> : tensor<64x128x56x56xf32>
    %v4542 = stablehlo.add %v4540, %v4541 : tensor<64x128x56x56xf32>
    %v4543 = stablehlo.rsqrt %v4542 : tensor<64x128x56x56xf32>
    %v4544 = stablehlo.multiply %v4536, %v4543 : tensor<64x128x56x56xf32>
    %v4545 = stablehlo.reshape %v4441 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4546 = stablehlo.multiply %v4545, %v4544 : tensor<64x128x56x56xf32>
    %v4547 = stablehlo.reduce(%v4546 init: %v4531) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4548 = stablehlo.reshape %v4441 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4549 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4550 = stablehlo.reduce(%v4548 init: %v4549) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v4551 = stablehlo.reshape %v364 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v4552 = stablehlo.reshape %v4428 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4554 = stablehlo.pad %v4552, %v4553, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4555 = stablehlo.transpose %v4551, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4556 = stablehlo.transpose %v4554, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4557 = stablehlo.convert %v4555 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4558 = stablehlo.convert %v4556 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4559 = stablehlo.convolution(%v4557, %v4558)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<128x128x3x3xbf16>
    %v4560 = stablehlo.convert %v4559 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4561 = stablehlo.transpose %v4560, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4562 = stablehlo.reshape %v372 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4563 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4564 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v4565 = stablehlo.reduce(%v4562 init: %v4563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4566 = stablehlo.broadcast_in_dim %v4565, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4567 = stablehlo.divide %v4566, %v4564 : tensor<64x128x28x28xf32>
    %v4568 = stablehlo.subtract %v4562, %v4567 : tensor<64x128x28x28xf32>
    %v4569 = stablehlo.multiply %v4568, %v4568 : tensor<64x128x28x28xf32>
    %v4570 = stablehlo.reduce(%v4569 init: %v4563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4571 = stablehlo.broadcast_in_dim %v4570, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4572 = stablehlo.divide %v4571, %v4564 : tensor<64x128x28x28xf32>
    %v4573 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4574 = stablehlo.add %v4572, %v4573 : tensor<64x128x28x28xf32>
    %v4575 = stablehlo.rsqrt %v4574 : tensor<64x128x28x28xf32>
    %v4576 = stablehlo.multiply %v4568, %v4575 : tensor<64x128x28x28xf32>
    %v4577 = stablehlo.reshape %v4398 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4578 = stablehlo.multiply %v4577, %v4576 : tensor<64x128x28x28xf32>
    %v4579 = stablehlo.reduce(%v4578 init: %v4563) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4580 = stablehlo.reshape %v4398 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4581 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4582 = stablehlo.reduce(%v4580 init: %v4581) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4583 = stablehlo.reshape %v394 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4584 = stablehlo.reshape %v4387 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4585 = stablehlo.transpose %v4583, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4586 = stablehlo.transpose %v4584, dims = [1, 0, 2, 3] : (tensor<64x512x28x28xf32>) -> tensor<512x64x28x28xf32>
    %v4587 = stablehlo.convert %v4585 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4588 = stablehlo.convert %v4586 : (tensor<512x64x28x28xf32>) -> tensor<512x64x28x28xbf16>
    %v4589 = stablehlo.convolution(%v4587, %v4588)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<512x64x28x28xbf16>) -> tensor<128x512x1x1xbf16>
    %v4590 = stablehlo.convert %v4589 : (tensor<128x512x1x1xbf16>) -> tensor<128x512x1x1xf32>
    %v4591 = stablehlo.transpose %v4590, dims = [1, 0, 2, 3] : (tensor<128x512x1x1xf32>) -> tensor<512x128x1x1xf32>
    %v4592 = stablehlo.reshape %v402 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4594 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v4595 = stablehlo.reduce(%v4592 init: %v4593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4596 = stablehlo.broadcast_in_dim %v4595, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4597 = stablehlo.divide %v4596, %v4594 : tensor<64x512x28x28xf32>
    %v4598 = stablehlo.subtract %v4592, %v4597 : tensor<64x512x28x28xf32>
    %v4599 = stablehlo.multiply %v4598, %v4598 : tensor<64x512x28x28xf32>
    %v4600 = stablehlo.reduce(%v4599 init: %v4593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4601 = stablehlo.broadcast_in_dim %v4600, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v4602 = stablehlo.divide %v4601, %v4594 : tensor<64x512x28x28xf32>
    %v4603 = stablehlo.constant dense<1.0e-05> : tensor<64x512x28x28xf32>
    %v4604 = stablehlo.add %v4602, %v4603 : tensor<64x512x28x28xf32>
    %v4605 = stablehlo.rsqrt %v4604 : tensor<64x512x28x28xf32>
    %v4606 = stablehlo.multiply %v4598, %v4605 : tensor<64x512x28x28xf32>
    %v4607 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4608 = stablehlo.multiply %v4607, %v4606 : tensor<64x512x28x28xf32>
    %v4609 = stablehlo.reduce(%v4608 init: %v4593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4610 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4612 = stablehlo.reduce(%v4610 init: %v4611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4613 = stablehlo.reshape %v334 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4614 = stablehlo.reshape %v4509 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4615 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4616 = stablehlo.pad %v4614, %v4615, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<64x512x56x56xf32>
    %v4617 = stablehlo.transpose %v4613, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4618 = stablehlo.transpose %v4616, dims = [1, 0, 2, 3] : (tensor<64x512x56x56xf32>) -> tensor<512x64x56x56xf32>
    %v4619 = stablehlo.convert %v4617 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4620 = stablehlo.convert %v4618 : (tensor<512x64x56x56xf32>) -> tensor<512x64x56x56xbf16>
    %v4621 = stablehlo.convolution(%v4619, %v4620)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<512x64x56x56xbf16>) -> tensor<256x512x1x1xbf16>
    %v4622 = stablehlo.convert %v4621 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v4623 = stablehlo.transpose %v4622, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v4624 = stablehlo.reshape %v430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
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
    %v4639 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4640 = stablehlo.multiply %v4639, %v4638 : tensor<64x512x28x28xf32>
    %v4641 = stablehlo.reduce(%v4640 init: %v4625) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4642 = stablehlo.reshape %v4357 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v4643 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4644 = stablehlo.reduce(%v4642 init: %v4643) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v4645 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v4646 = stablehlo.compare GT, %v332, %v4645 : (tensor<64x802816xf32>, tensor<64x802816xf32>) -> tensor<64x802816xi1>
    %v4647 = stablehlo.select %v4646, %v4520, %v4645 : tensor<64x802816xi1>, tensor<64x802816xf32>
    %v4648 = stablehlo.reshape %v311 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4649 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4650 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v4651 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v4652 = stablehlo.reduce(%v4648 init: %v4649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4653 = stablehlo.broadcast_in_dim %v4652, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4654 = stablehlo.divide %v4653, %v4650 : tensor<64x256x56x56xf32>
    %v4655 = stablehlo.subtract %v4648, %v4654 : tensor<64x256x56x56xf32>
    %v4656 = stablehlo.multiply %v4655, %v4655 : tensor<64x256x56x56xf32>
    %v4657 = stablehlo.reduce(%v4656 init: %v4649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4658 = stablehlo.broadcast_in_dim %v4657, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4659 = stablehlo.divide %v4658, %v4650 : tensor<64x256x56x56xf32>
    %v4660 = stablehlo.add %v4659, %v4651 : tensor<64x256x56x56xf32>
    %v4661 = stablehlo.rsqrt %v4660 : tensor<64x256x56x56xf32>
    %v4662 = stablehlo.multiply %v4655, %v4661 : tensor<64x256x56x56xf32>
    %v4663 = stablehlo.broadcast_in_dim %s1b2g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4664 = stablehlo.reshape %v4647 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4665 = stablehlo.multiply %v4663, %v4664 : tensor<64x256x56x56xf32>
    %v4666 = stablehlo.reduce(%v4665 init: %v4649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4667 = stablehlo.broadcast_in_dim %v4666, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4668 = stablehlo.multiply %v4662, %v4665 : tensor<64x256x56x56xf32>
    %v4669 = stablehlo.reduce(%v4668 init: %v4649) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4670 = stablehlo.broadcast_in_dim %v4669, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4671 = stablehlo.multiply %v4665, %v4650 : tensor<64x256x56x56xf32>
    %v4672 = stablehlo.subtract %v4671, %v4667 : tensor<64x256x56x56xf32>
    %v4673 = stablehlo.multiply %v4662, %v4670 : tensor<64x256x56x56xf32>
    %v4674 = stablehlo.subtract %v4672, %v4673 : tensor<64x256x56x56xf32>
    %v4675 = stablehlo.divide %v4661, %v4650 : tensor<64x256x56x56xf32>
    %v4676 = stablehlo.multiply %v4675, %v4674 : tensor<64x256x56x56xf32>
    %v4677 = stablehlo.reshape %v4676 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4678 = stablehlo.reshape %v4677 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4679 = stablehlo.reverse %s1b2W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4680 = stablehlo.transpose %v4679, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4681 = stablehlo.convert %v4678 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v4682 = stablehlo.convert %v4680 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v4683 = stablehlo.convolution(%v4681, %v4682)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v4684 = stablehlo.convert %v4683 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4685 = stablehlo.reshape %v4684 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4686 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v4687 = stablehlo.compare GT, %v301, %v4686 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v4688 = stablehlo.select %v4687, %v4685, %v4686 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v4689 = stablehlo.reshape %v281 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4691 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4692 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4693 = stablehlo.reduce(%v4689 init: %v4690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4694 = stablehlo.broadcast_in_dim %v4693, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4695 = stablehlo.divide %v4694, %v4691 : tensor<64x64x56x56xf32>
    %v4696 = stablehlo.subtract %v4689, %v4695 : tensor<64x64x56x56xf32>
    %v4697 = stablehlo.multiply %v4696, %v4696 : tensor<64x64x56x56xf32>
    %v4698 = stablehlo.reduce(%v4697 init: %v4690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4699 = stablehlo.broadcast_in_dim %v4698, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4700 = stablehlo.divide %v4699, %v4691 : tensor<64x64x56x56xf32>
    %v4701 = stablehlo.add %v4700, %v4692 : tensor<64x64x56x56xf32>
    %v4702 = stablehlo.rsqrt %v4701 : tensor<64x64x56x56xf32>
    %v4703 = stablehlo.multiply %v4696, %v4702 : tensor<64x64x56x56xf32>
    %v4704 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4705 = stablehlo.reshape %v4688 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4706 = stablehlo.multiply %v4704, %v4705 : tensor<64x64x56x56xf32>
    %v4707 = stablehlo.reduce(%v4706 init: %v4690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4708 = stablehlo.broadcast_in_dim %v4707, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4709 = stablehlo.multiply %v4703, %v4706 : tensor<64x64x56x56xf32>
    %v4710 = stablehlo.reduce(%v4709 init: %v4690) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4711 = stablehlo.broadcast_in_dim %v4710, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4712 = stablehlo.multiply %v4706, %v4691 : tensor<64x64x56x56xf32>
    %v4713 = stablehlo.subtract %v4712, %v4708 : tensor<64x64x56x56xf32>
    %v4714 = stablehlo.multiply %v4703, %v4711 : tensor<64x64x56x56xf32>
    %v4715 = stablehlo.subtract %v4713, %v4714 : tensor<64x64x56x56xf32>
    %v4716 = stablehlo.divide %v4702, %v4691 : tensor<64x64x56x56xf32>
    %v4717 = stablehlo.multiply %v4716, %v4715 : tensor<64x64x56x56xf32>
    %v4718 = stablehlo.reshape %v4717 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4719 = stablehlo.reshape %v4718 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4720 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4721 = stablehlo.transpose %v4720, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4722 = stablehlo.convert %v4719 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4723 = stablehlo.convert %v4721 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4724 = stablehlo.convolution(%v4722, %v4723)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4725 = stablehlo.convert %v4724 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4726 = stablehlo.reshape %v4725 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4727 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v4728 = stablehlo.compare GT, %v271, %v4727 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v4729 = stablehlo.select %v4728, %v4726, %v4727 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v4730 = stablehlo.reshape %v251 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4731 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4732 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4733 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4734 = stablehlo.reduce(%v4730 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4735 = stablehlo.broadcast_in_dim %v4734, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4736 = stablehlo.divide %v4735, %v4732 : tensor<64x64x56x56xf32>
    %v4737 = stablehlo.subtract %v4730, %v4736 : tensor<64x64x56x56xf32>
    %v4738 = stablehlo.multiply %v4737, %v4737 : tensor<64x64x56x56xf32>
    %v4739 = stablehlo.reduce(%v4738 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4740 = stablehlo.broadcast_in_dim %v4739, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4741 = stablehlo.divide %v4740, %v4732 : tensor<64x64x56x56xf32>
    %v4742 = stablehlo.add %v4741, %v4733 : tensor<64x64x56x56xf32>
    %v4743 = stablehlo.rsqrt %v4742 : tensor<64x64x56x56xf32>
    %v4744 = stablehlo.multiply %v4737, %v4743 : tensor<64x64x56x56xf32>
    %v4745 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4746 = stablehlo.reshape %v4729 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4747 = stablehlo.multiply %v4745, %v4746 : tensor<64x64x56x56xf32>
    %v4748 = stablehlo.reduce(%v4747 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4749 = stablehlo.broadcast_in_dim %v4748, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4750 = stablehlo.multiply %v4744, %v4747 : tensor<64x64x56x56xf32>
    %v4751 = stablehlo.reduce(%v4750 init: %v4731) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4752 = stablehlo.broadcast_in_dim %v4751, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4753 = stablehlo.multiply %v4747, %v4732 : tensor<64x64x56x56xf32>
    %v4754 = stablehlo.subtract %v4753, %v4749 : tensor<64x64x56x56xf32>
    %v4755 = stablehlo.multiply %v4744, %v4752 : tensor<64x64x56x56xf32>
    %v4756 = stablehlo.subtract %v4754, %v4755 : tensor<64x64x56x56xf32>
    %v4757 = stablehlo.divide %v4743, %v4732 : tensor<64x64x56x56xf32>
    %v4758 = stablehlo.multiply %v4757, %v4756 : tensor<64x64x56x56xf32>
    %v4759 = stablehlo.reshape %v4758 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4760 = stablehlo.reshape %v4759 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4761 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4762 = stablehlo.transpose %v4761, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4763 = stablehlo.convert %v4760 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4764 = stablehlo.convert %v4762 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v4765 = stablehlo.convolution(%v4763, %v4764)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4766 = stablehlo.convert %v4765 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4767 = stablehlo.reshape %v4766 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4768 = stablehlo.add %v4767, %v4647 : tensor<64x802816xf32>
    %v4769 = stablehlo.reshape %v243 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4770 = stablehlo.reshape %v4759 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4771 = stablehlo.transpose %v4769, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4772 = stablehlo.transpose %v4770, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4773 = stablehlo.convert %v4771 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4774 = stablehlo.convert %v4772 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4775 = stablehlo.convolution(%v4773, %v4774)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v4776 = stablehlo.convert %v4775 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v4777 = stablehlo.transpose %v4776, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4778 = stablehlo.reshape %v251 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4780 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4781 = stablehlo.reduce(%v4778 init: %v4779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4782 = stablehlo.broadcast_in_dim %v4781, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4783 = stablehlo.divide %v4782, %v4780 : tensor<64x64x56x56xf32>
    %v4784 = stablehlo.subtract %v4778, %v4783 : tensor<64x64x56x56xf32>
    %v4785 = stablehlo.multiply %v4784, %v4784 : tensor<64x64x56x56xf32>
    %v4786 = stablehlo.reduce(%v4785 init: %v4779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4787 = stablehlo.broadcast_in_dim %v4786, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4788 = stablehlo.divide %v4787, %v4780 : tensor<64x64x56x56xf32>
    %v4789 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4790 = stablehlo.add %v4788, %v4789 : tensor<64x64x56x56xf32>
    %v4791 = stablehlo.rsqrt %v4790 : tensor<64x64x56x56xf32>
    %v4792 = stablehlo.multiply %v4784, %v4791 : tensor<64x64x56x56xf32>
    %v4793 = stablehlo.reshape %v4729 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4794 = stablehlo.multiply %v4793, %v4792 : tensor<64x64x56x56xf32>
    %v4795 = stablehlo.reduce(%v4794 init: %v4779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4796 = stablehlo.reshape %v4729 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4797 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4798 = stablehlo.reduce(%v4796 init: %v4797) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4799 = stablehlo.reshape %v273 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4800 = stablehlo.reshape %v4718 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4801 = stablehlo.transpose %v4799, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4802 = stablehlo.transpose %v4800, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4803 = stablehlo.convert %v4801 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4804 = stablehlo.convert %v4802 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4805 = stablehlo.convolution(%v4803, %v4804)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4806 = stablehlo.convert %v4805 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4807 = stablehlo.transpose %v4806, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4808 = stablehlo.reshape %v281 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4810 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4811 = stablehlo.reduce(%v4808 init: %v4809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4812 = stablehlo.broadcast_in_dim %v4811, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4813 = stablehlo.divide %v4812, %v4810 : tensor<64x64x56x56xf32>
    %v4814 = stablehlo.subtract %v4808, %v4813 : tensor<64x64x56x56xf32>
    %v4815 = stablehlo.multiply %v4814, %v4814 : tensor<64x64x56x56xf32>
    %v4816 = stablehlo.reduce(%v4815 init: %v4809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4817 = stablehlo.broadcast_in_dim %v4816, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4818 = stablehlo.divide %v4817, %v4810 : tensor<64x64x56x56xf32>
    %v4819 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4820 = stablehlo.add %v4818, %v4819 : tensor<64x64x56x56xf32>
    %v4821 = stablehlo.rsqrt %v4820 : tensor<64x64x56x56xf32>
    %v4822 = stablehlo.multiply %v4814, %v4821 : tensor<64x64x56x56xf32>
    %v4823 = stablehlo.reshape %v4688 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4824 = stablehlo.multiply %v4823, %v4822 : tensor<64x64x56x56xf32>
    %v4825 = stablehlo.reduce(%v4824 init: %v4809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4826 = stablehlo.reshape %v4688 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4827 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4828 = stablehlo.reduce(%v4826 init: %v4827) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4829 = stablehlo.reshape %v303 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4830 = stablehlo.reshape %v4677 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4831 = stablehlo.transpose %v4829, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4832 = stablehlo.transpose %v4830, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4833 = stablehlo.convert %v4831 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4834 = stablehlo.convert %v4832 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4835 = stablehlo.convolution(%v4833, %v4834)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v4836 = stablehlo.convert %v4835 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v4837 = stablehlo.transpose %v4836, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4838 = stablehlo.reshape %v311 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4839 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4840 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v4841 = stablehlo.reduce(%v4838 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4842 = stablehlo.broadcast_in_dim %v4841, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4843 = stablehlo.divide %v4842, %v4840 : tensor<64x256x56x56xf32>
    %v4844 = stablehlo.subtract %v4838, %v4843 : tensor<64x256x56x56xf32>
    %v4845 = stablehlo.multiply %v4844, %v4844 : tensor<64x256x56x56xf32>
    %v4846 = stablehlo.reduce(%v4845 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4847 = stablehlo.broadcast_in_dim %v4846, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4848 = stablehlo.divide %v4847, %v4840 : tensor<64x256x56x56xf32>
    %v4849 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v4850 = stablehlo.add %v4848, %v4849 : tensor<64x256x56x56xf32>
    %v4851 = stablehlo.rsqrt %v4850 : tensor<64x256x56x56xf32>
    %v4852 = stablehlo.multiply %v4844, %v4851 : tensor<64x256x56x56xf32>
    %v4853 = stablehlo.reshape %v4647 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4854 = stablehlo.multiply %v4853, %v4852 : tensor<64x256x56x56xf32>
    %v4855 = stablehlo.reduce(%v4854 init: %v4839) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4856 = stablehlo.reshape %v4647 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4857 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4858 = stablehlo.reduce(%v4856 init: %v4857) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4859 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v4860 = stablehlo.compare GT, %v241, %v4859 : (tensor<64x802816xf32>, tensor<64x802816xf32>) -> tensor<64x802816xi1>
    %v4861 = stablehlo.select %v4860, %v4768, %v4859 : tensor<64x802816xi1>, tensor<64x802816xf32>
    %v4862 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4864 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v4865 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v4866 = stablehlo.reduce(%v4862 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4867 = stablehlo.broadcast_in_dim %v4866, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4868 = stablehlo.divide %v4867, %v4864 : tensor<64x256x56x56xf32>
    %v4869 = stablehlo.subtract %v4862, %v4868 : tensor<64x256x56x56xf32>
    %v4870 = stablehlo.multiply %v4869, %v4869 : tensor<64x256x56x56xf32>
    %v4871 = stablehlo.reduce(%v4870 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4872 = stablehlo.broadcast_in_dim %v4871, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4873 = stablehlo.divide %v4872, %v4864 : tensor<64x256x56x56xf32>
    %v4874 = stablehlo.add %v4873, %v4865 : tensor<64x256x56x56xf32>
    %v4875 = stablehlo.rsqrt %v4874 : tensor<64x256x56x56xf32>
    %v4876 = stablehlo.multiply %v4869, %v4875 : tensor<64x256x56x56xf32>
    %v4877 = stablehlo.broadcast_in_dim %s1b1g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4878 = stablehlo.reshape %v4861 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4879 = stablehlo.multiply %v4877, %v4878 : tensor<64x256x56x56xf32>
    %v4880 = stablehlo.reduce(%v4879 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4881 = stablehlo.broadcast_in_dim %v4880, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4882 = stablehlo.multiply %v4876, %v4879 : tensor<64x256x56x56xf32>
    %v4883 = stablehlo.reduce(%v4882 init: %v4863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v4884 = stablehlo.broadcast_in_dim %v4883, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v4885 = stablehlo.multiply %v4879, %v4864 : tensor<64x256x56x56xf32>
    %v4886 = stablehlo.subtract %v4885, %v4881 : tensor<64x256x56x56xf32>
    %v4887 = stablehlo.multiply %v4876, %v4884 : tensor<64x256x56x56xf32>
    %v4888 = stablehlo.subtract %v4886, %v4887 : tensor<64x256x56x56xf32>
    %v4889 = stablehlo.divide %v4875, %v4864 : tensor<64x256x56x56xf32>
    %v4890 = stablehlo.multiply %v4889, %v4888 : tensor<64x256x56x56xf32>
    %v4891 = stablehlo.reshape %v4890 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4892 = stablehlo.reshape %v4891 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4893 = stablehlo.reverse %s1b1W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v4894 = stablehlo.transpose %v4893, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4895 = stablehlo.convert %v4892 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v4896 = stablehlo.convert %v4894 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v4897 = stablehlo.convolution(%v4895, %v4896)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v4898 = stablehlo.convert %v4897 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4899 = stablehlo.reshape %v4898 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4900 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v4901 = stablehlo.compare GT, %v210, %v4900 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v4902 = stablehlo.select %v4901, %v4899, %v4900 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v4903 = stablehlo.reshape %v190 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4905 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4906 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4907 = stablehlo.reduce(%v4903 init: %v4904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4908 = stablehlo.broadcast_in_dim %v4907, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4909 = stablehlo.divide %v4908, %v4905 : tensor<64x64x56x56xf32>
    %v4910 = stablehlo.subtract %v4903, %v4909 : tensor<64x64x56x56xf32>
    %v4911 = stablehlo.multiply %v4910, %v4910 : tensor<64x64x56x56xf32>
    %v4912 = stablehlo.reduce(%v4911 init: %v4904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4913 = stablehlo.broadcast_in_dim %v4912, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4914 = stablehlo.divide %v4913, %v4905 : tensor<64x64x56x56xf32>
    %v4915 = stablehlo.add %v4914, %v4906 : tensor<64x64x56x56xf32>
    %v4916 = stablehlo.rsqrt %v4915 : tensor<64x64x56x56xf32>
    %v4917 = stablehlo.multiply %v4910, %v4916 : tensor<64x64x56x56xf32>
    %v4918 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4919 = stablehlo.reshape %v4902 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4920 = stablehlo.multiply %v4918, %v4919 : tensor<64x64x56x56xf32>
    %v4921 = stablehlo.reduce(%v4920 init: %v4904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4922 = stablehlo.broadcast_in_dim %v4921, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4923 = stablehlo.multiply %v4917, %v4920 : tensor<64x64x56x56xf32>
    %v4924 = stablehlo.reduce(%v4923 init: %v4904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4925 = stablehlo.broadcast_in_dim %v4924, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4926 = stablehlo.multiply %v4920, %v4905 : tensor<64x64x56x56xf32>
    %v4927 = stablehlo.subtract %v4926, %v4922 : tensor<64x64x56x56xf32>
    %v4928 = stablehlo.multiply %v4917, %v4925 : tensor<64x64x56x56xf32>
    %v4929 = stablehlo.subtract %v4927, %v4928 : tensor<64x64x56x56xf32>
    %v4930 = stablehlo.divide %v4916, %v4905 : tensor<64x64x56x56xf32>
    %v4931 = stablehlo.multiply %v4930, %v4929 : tensor<64x64x56x56xf32>
    %v4932 = stablehlo.reshape %v4931 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4933 = stablehlo.reshape %v4932 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4934 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4935 = stablehlo.transpose %v4934, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4936 = stablehlo.convert %v4933 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4937 = stablehlo.convert %v4935 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4938 = stablehlo.convolution(%v4936, %v4937)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4939 = stablehlo.convert %v4938 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4940 = stablehlo.reshape %v4939 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4941 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v4942 = stablehlo.compare GT, %v180, %v4941 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v4943 = stablehlo.select %v4942, %v4940, %v4941 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v4944 = stablehlo.reshape %v160 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4946 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4947 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4948 = stablehlo.reduce(%v4944 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4949 = stablehlo.broadcast_in_dim %v4948, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4950 = stablehlo.divide %v4949, %v4946 : tensor<64x64x56x56xf32>
    %v4951 = stablehlo.subtract %v4944, %v4950 : tensor<64x64x56x56xf32>
    %v4952 = stablehlo.multiply %v4951, %v4951 : tensor<64x64x56x56xf32>
    %v4953 = stablehlo.reduce(%v4952 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4954 = stablehlo.broadcast_in_dim %v4953, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4955 = stablehlo.divide %v4954, %v4946 : tensor<64x64x56x56xf32>
    %v4956 = stablehlo.add %v4955, %v4947 : tensor<64x64x56x56xf32>
    %v4957 = stablehlo.rsqrt %v4956 : tensor<64x64x56x56xf32>
    %v4958 = stablehlo.multiply %v4951, %v4957 : tensor<64x64x56x56xf32>
    %v4959 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4960 = stablehlo.reshape %v4943 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4961 = stablehlo.multiply %v4959, %v4960 : tensor<64x64x56x56xf32>
    %v4962 = stablehlo.reduce(%v4961 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4963 = stablehlo.broadcast_in_dim %v4962, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4964 = stablehlo.multiply %v4958, %v4961 : tensor<64x64x56x56xf32>
    %v4965 = stablehlo.reduce(%v4964 init: %v4945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4966 = stablehlo.broadcast_in_dim %v4965, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4967 = stablehlo.multiply %v4961, %v4946 : tensor<64x64x56x56xf32>
    %v4968 = stablehlo.subtract %v4967, %v4963 : tensor<64x64x56x56xf32>
    %v4969 = stablehlo.multiply %v4958, %v4966 : tensor<64x64x56x56xf32>
    %v4970 = stablehlo.subtract %v4968, %v4969 : tensor<64x64x56x56xf32>
    %v4971 = stablehlo.divide %v4957, %v4946 : tensor<64x64x56x56xf32>
    %v4972 = stablehlo.multiply %v4971, %v4970 : tensor<64x64x56x56xf32>
    %v4973 = stablehlo.reshape %v4972 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4974 = stablehlo.reshape %v4973 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4975 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x256x1x1xf32>
    %v4976 = stablehlo.transpose %v4975, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v4977 = stablehlo.convert %v4974 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4978 = stablehlo.convert %v4976 : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xbf16>
    %v4979 = stablehlo.convolution(%v4977, %v4978)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x1x1xbf16>) -> tensor<64x256x56x56xbf16>
    %v4980 = stablehlo.convert %v4979 : (tensor<64x256x56x56xbf16>) -> tensor<64x256x56x56xf32>
    %v4981 = stablehlo.reshape %v4980 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v4982 = stablehlo.add %v4981, %v4861 : tensor<64x802816xf32>
    %v4983 = stablehlo.reshape %v152 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v4984 = stablehlo.reshape %v4973 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4985 = stablehlo.transpose %v4983, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v4986 = stablehlo.transpose %v4984, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4987 = stablehlo.convert %v4985 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v4988 = stablehlo.convert %v4986 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4989 = stablehlo.convolution(%v4987, %v4988)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<256x64x1x1xbf16>
    %v4990 = stablehlo.convert %v4989 : (tensor<256x64x1x1xbf16>) -> tensor<256x64x1x1xf32>
    %v4991 = stablehlo.transpose %v4990, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v4992 = stablehlo.reshape %v160 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4994 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v4995 = stablehlo.reduce(%v4992 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4996 = stablehlo.broadcast_in_dim %v4995, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4997 = stablehlo.divide %v4996, %v4994 : tensor<64x64x56x56xf32>
    %v4998 = stablehlo.subtract %v4992, %v4997 : tensor<64x64x56x56xf32>
    %v4999 = stablehlo.multiply %v4998, %v4998 : tensor<64x64x56x56xf32>
    %v5000 = stablehlo.reduce(%v4999 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5001 = stablehlo.broadcast_in_dim %v5000, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5002 = stablehlo.divide %v5001, %v4994 : tensor<64x64x56x56xf32>
    %v5003 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5004 = stablehlo.add %v5002, %v5003 : tensor<64x64x56x56xf32>
    %v5005 = stablehlo.rsqrt %v5004 : tensor<64x64x56x56xf32>
    %v5006 = stablehlo.multiply %v4998, %v5005 : tensor<64x64x56x56xf32>
    %v5007 = stablehlo.reshape %v4943 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5008 = stablehlo.multiply %v5007, %v5006 : tensor<64x64x56x56xf32>
    %v5009 = stablehlo.reduce(%v5008 init: %v4993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5010 = stablehlo.reshape %v4943 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5012 = stablehlo.reduce(%v5010 init: %v5011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5013 = stablehlo.reshape %v182 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5014 = stablehlo.reshape %v4932 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5015 = stablehlo.transpose %v5013, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5016 = stablehlo.transpose %v5014, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5017 = stablehlo.convert %v5015 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5018 = stablehlo.convert %v5016 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5019 = stablehlo.convolution(%v5017, %v5018)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v5020 = stablehlo.convert %v5019 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v5021 = stablehlo.transpose %v5020, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5022 = stablehlo.reshape %v190 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5024 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5025 = stablehlo.reduce(%v5022 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5026 = stablehlo.broadcast_in_dim %v5025, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5027 = stablehlo.divide %v5026, %v5024 : tensor<64x64x56x56xf32>
    %v5028 = stablehlo.subtract %v5022, %v5027 : tensor<64x64x56x56xf32>
    %v5029 = stablehlo.multiply %v5028, %v5028 : tensor<64x64x56x56xf32>
    %v5030 = stablehlo.reduce(%v5029 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5031 = stablehlo.broadcast_in_dim %v5030, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5032 = stablehlo.divide %v5031, %v5024 : tensor<64x64x56x56xf32>
    %v5033 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5034 = stablehlo.add %v5032, %v5033 : tensor<64x64x56x56xf32>
    %v5035 = stablehlo.rsqrt %v5034 : tensor<64x64x56x56xf32>
    %v5036 = stablehlo.multiply %v5028, %v5035 : tensor<64x64x56x56xf32>
    %v5037 = stablehlo.reshape %v4902 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5038 = stablehlo.multiply %v5037, %v5036 : tensor<64x64x56x56xf32>
    %v5039 = stablehlo.reduce(%v5038 init: %v5023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5040 = stablehlo.reshape %v4902 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5042 = stablehlo.reduce(%v5040 init: %v5041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5043 = stablehlo.reshape %v212 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5044 = stablehlo.reshape %v4891 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5045 = stablehlo.transpose %v5043, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5046 = stablehlo.transpose %v5044, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5047 = stablehlo.convert %v5045 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5048 = stablehlo.convert %v5046 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5049 = stablehlo.convolution(%v5047, %v5048)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5050 = stablehlo.convert %v5049 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5051 = stablehlo.transpose %v5050, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5052 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5053 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5054 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5055 = stablehlo.reduce(%v5052 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5056 = stablehlo.broadcast_in_dim %v5055, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5057 = stablehlo.divide %v5056, %v5054 : tensor<64x256x56x56xf32>
    %v5058 = stablehlo.subtract %v5052, %v5057 : tensor<64x256x56x56xf32>
    %v5059 = stablehlo.multiply %v5058, %v5058 : tensor<64x256x56x56xf32>
    %v5060 = stablehlo.reduce(%v5059 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5061 = stablehlo.broadcast_in_dim %v5060, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5062 = stablehlo.divide %v5061, %v5054 : tensor<64x256x56x56xf32>
    %v5063 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5064 = stablehlo.add %v5062, %v5063 : tensor<64x256x56x56xf32>
    %v5065 = stablehlo.rsqrt %v5064 : tensor<64x256x56x56xf32>
    %v5066 = stablehlo.multiply %v5058, %v5065 : tensor<64x256x56x56xf32>
    %v5067 = stablehlo.reshape %v4861 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5068 = stablehlo.multiply %v5067, %v5066 : tensor<64x256x56x56xf32>
    %v5069 = stablehlo.reduce(%v5068 init: %v5053) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5070 = stablehlo.reshape %v4861 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5072 = stablehlo.reduce(%v5070 init: %v5071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5073 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v5074 = stablehlo.compare GT, %v150, %v5073 : (tensor<64x802816xf32>, tensor<64x802816xf32>) -> tensor<64x802816xi1>
    %v5075 = stablehlo.select %v5074, %v4982, %v5073 : tensor<64x802816xi1>, tensor<64x802816xf32>
    %v5076 = stablehlo.reshape %v101 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5077 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5078 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5079 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5080 = stablehlo.reduce(%v5076 init: %v5077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5081 = stablehlo.broadcast_in_dim %v5080, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5082 = stablehlo.divide %v5081, %v5078 : tensor<64x256x56x56xf32>
    %v5083 = stablehlo.subtract %v5076, %v5082 : tensor<64x256x56x56xf32>
    %v5084 = stablehlo.multiply %v5083, %v5083 : tensor<64x256x56x56xf32>
    %v5085 = stablehlo.reduce(%v5084 init: %v5077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5086 = stablehlo.broadcast_in_dim %v5085, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5087 = stablehlo.divide %v5086, %v5078 : tensor<64x256x56x56xf32>
    %v5088 = stablehlo.add %v5087, %v5079 : tensor<64x256x56x56xf32>
    %v5089 = stablehlo.rsqrt %v5088 : tensor<64x256x56x56xf32>
    %v5090 = stablehlo.multiply %v5083, %v5089 : tensor<64x256x56x56xf32>
    %v5091 = stablehlo.broadcast_in_dim %s1b0g3, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5092 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5093 = stablehlo.multiply %v5091, %v5092 : tensor<64x256x56x56xf32>
    %v5094 = stablehlo.reduce(%v5093 init: %v5077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5095 = stablehlo.broadcast_in_dim %v5094, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5096 = stablehlo.multiply %v5090, %v5093 : tensor<64x256x56x56xf32>
    %v5097 = stablehlo.reduce(%v5096 init: %v5077) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5098 = stablehlo.broadcast_in_dim %v5097, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5099 = stablehlo.multiply %v5093, %v5078 : tensor<64x256x56x56xf32>
    %v5100 = stablehlo.subtract %v5099, %v5095 : tensor<64x256x56x56xf32>
    %v5101 = stablehlo.multiply %v5090, %v5098 : tensor<64x256x56x56xf32>
    %v5102 = stablehlo.subtract %v5100, %v5101 : tensor<64x256x56x56xf32>
    %v5103 = stablehlo.divide %v5089, %v5078 : tensor<64x256x56x56xf32>
    %v5104 = stablehlo.multiply %v5103, %v5102 : tensor<64x256x56x56xf32>
    %v5105 = stablehlo.reshape %v5104 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5106 = stablehlo.reshape %v5105 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5107 = stablehlo.reverse %s1b0W3, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5108 = stablehlo.transpose %v5107, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5109 = stablehlo.convert %v5106 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v5110 = stablehlo.convert %v5108 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v5111 = stablehlo.convolution(%v5109, %v5110)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5112 = stablehlo.convert %v5111 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5113 = stablehlo.reshape %v5112 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5114 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v5115 = stablehlo.compare GT, %v91, %v5114 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v5116 = stablehlo.select %v5115, %v5113, %v5114 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v5117 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5118 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5119 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5120 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5121 = stablehlo.reduce(%v5117 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5122 = stablehlo.broadcast_in_dim %v5121, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5123 = stablehlo.divide %v5122, %v5119 : tensor<64x64x56x56xf32>
    %v5124 = stablehlo.subtract %v5117, %v5123 : tensor<64x64x56x56xf32>
    %v5125 = stablehlo.multiply %v5124, %v5124 : tensor<64x64x56x56xf32>
    %v5126 = stablehlo.reduce(%v5125 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5127 = stablehlo.broadcast_in_dim %v5126, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5128 = stablehlo.divide %v5127, %v5119 : tensor<64x64x56x56xf32>
    %v5129 = stablehlo.add %v5128, %v5120 : tensor<64x64x56x56xf32>
    %v5130 = stablehlo.rsqrt %v5129 : tensor<64x64x56x56xf32>
    %v5131 = stablehlo.multiply %v5124, %v5130 : tensor<64x64x56x56xf32>
    %v5132 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5133 = stablehlo.reshape %v5116 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5134 = stablehlo.multiply %v5132, %v5133 : tensor<64x64x56x56xf32>
    %v5135 = stablehlo.reduce(%v5134 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5136 = stablehlo.broadcast_in_dim %v5135, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5137 = stablehlo.multiply %v5131, %v5134 : tensor<64x64x56x56xf32>
    %v5138 = stablehlo.reduce(%v5137 init: %v5118) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5139 = stablehlo.broadcast_in_dim %v5138, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5140 = stablehlo.multiply %v5134, %v5119 : tensor<64x64x56x56xf32>
    %v5141 = stablehlo.subtract %v5140, %v5136 : tensor<64x64x56x56xf32>
    %v5142 = stablehlo.multiply %v5131, %v5139 : tensor<64x64x56x56xf32>
    %v5143 = stablehlo.subtract %v5141, %v5142 : tensor<64x64x56x56xf32>
    %v5144 = stablehlo.divide %v5130, %v5119 : tensor<64x64x56x56xf32>
    %v5145 = stablehlo.multiply %v5144, %v5143 : tensor<64x64x56x56xf32>
    %v5146 = stablehlo.reshape %v5145 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5147 = stablehlo.reshape %v5146 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5148 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v5149 = stablehlo.transpose %v5148, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5150 = stablehlo.convert %v5147 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5151 = stablehlo.convert %v5149 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v5152 = stablehlo.convolution(%v5150, %v5151)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v5153 = stablehlo.convert %v5152 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5154 = stablehlo.reshape %v5153 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5155 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v5156 = stablehlo.compare GT, %v61, %v5155 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v5157 = stablehlo.select %v5156, %v5154, %v5155 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v5158 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5159 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5160 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5161 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5162 = stablehlo.reduce(%v5158 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5163 = stablehlo.broadcast_in_dim %v5162, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5164 = stablehlo.divide %v5163, %v5160 : tensor<64x64x56x56xf32>
    %v5165 = stablehlo.subtract %v5158, %v5164 : tensor<64x64x56x56xf32>
    %v5166 = stablehlo.multiply %v5165, %v5165 : tensor<64x64x56x56xf32>
    %v5167 = stablehlo.reduce(%v5166 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5168 = stablehlo.broadcast_in_dim %v5167, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5169 = stablehlo.divide %v5168, %v5160 : tensor<64x64x56x56xf32>
    %v5170 = stablehlo.add %v5169, %v5161 : tensor<64x64x56x56xf32>
    %v5171 = stablehlo.rsqrt %v5170 : tensor<64x64x56x56xf32>
    %v5172 = stablehlo.multiply %v5165, %v5171 : tensor<64x64x56x56xf32>
    %v5173 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5174 = stablehlo.reshape %v5157 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5175 = stablehlo.multiply %v5173, %v5174 : tensor<64x64x56x56xf32>
    %v5176 = stablehlo.reduce(%v5175 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5177 = stablehlo.broadcast_in_dim %v5176, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5178 = stablehlo.multiply %v5172, %v5175 : tensor<64x64x56x56xf32>
    %v5179 = stablehlo.reduce(%v5178 init: %v5159) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5180 = stablehlo.broadcast_in_dim %v5179, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5181 = stablehlo.multiply %v5175, %v5160 : tensor<64x64x56x56xf32>
    %v5182 = stablehlo.subtract %v5181, %v5177 : tensor<64x64x56x56xf32>
    %v5183 = stablehlo.multiply %v5172, %v5180 : tensor<64x64x56x56xf32>
    %v5184 = stablehlo.subtract %v5182, %v5183 : tensor<64x64x56x56xf32>
    %v5185 = stablehlo.divide %v5171, %v5160 : tensor<64x64x56x56xf32>
    %v5186 = stablehlo.multiply %v5185, %v5184 : tensor<64x64x56x56xf32>
    %v5187 = stablehlo.reshape %v5186 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5188 = stablehlo.reshape %v5187 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5189 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x1x1xf32>
    %v5190 = stablehlo.transpose %v5189, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5191 = stablehlo.convert %v5188 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5192 = stablehlo.convert %v5190 : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xbf16>
    %v5193 = stablehlo.convolution(%v5191, %v5192)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5194 = stablehlo.convert %v5193 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5195 = stablehlo.reshape %v5194 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5196 = stablehlo.reshape %v129 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5197 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5198 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5199 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5200 = stablehlo.reduce(%v5196 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5201 = stablehlo.broadcast_in_dim %v5200, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5202 = stablehlo.divide %v5201, %v5198 : tensor<64x256x56x56xf32>
    %v5203 = stablehlo.subtract %v5196, %v5202 : tensor<64x256x56x56xf32>
    %v5204 = stablehlo.multiply %v5203, %v5203 : tensor<64x256x56x56xf32>
    %v5205 = stablehlo.reduce(%v5204 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5206 = stablehlo.broadcast_in_dim %v5205, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5207 = stablehlo.divide %v5206, %v5198 : tensor<64x256x56x56xf32>
    %v5208 = stablehlo.add %v5207, %v5199 : tensor<64x256x56x56xf32>
    %v5209 = stablehlo.rsqrt %v5208 : tensor<64x256x56x56xf32>
    %v5210 = stablehlo.multiply %v5203, %v5209 : tensor<64x256x56x56xf32>
    %v5211 = stablehlo.broadcast_in_dim %s1b0gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5212 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5213 = stablehlo.multiply %v5211, %v5212 : tensor<64x256x56x56xf32>
    %v5214 = stablehlo.reduce(%v5213 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5215 = stablehlo.broadcast_in_dim %v5214, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5216 = stablehlo.multiply %v5210, %v5213 : tensor<64x256x56x56xf32>
    %v5217 = stablehlo.reduce(%v5216 init: %v5197) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5218 = stablehlo.broadcast_in_dim %v5217, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5219 = stablehlo.multiply %v5213, %v5198 : tensor<64x256x56x56xf32>
    %v5220 = stablehlo.subtract %v5219, %v5215 : tensor<64x256x56x56xf32>
    %v5221 = stablehlo.multiply %v5210, %v5218 : tensor<64x256x56x56xf32>
    %v5222 = stablehlo.subtract %v5220, %v5221 : tensor<64x256x56x56xf32>
    %v5223 = stablehlo.divide %v5209, %v5198 : tensor<64x256x56x56xf32>
    %v5224 = stablehlo.multiply %v5223, %v5222 : tensor<64x256x56x56xf32>
    %v5225 = stablehlo.reshape %v5224 : (tensor<64x256x56x56xf32>) -> tensor<64x802816xf32>
    %v5226 = stablehlo.reshape %v5225 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5227 = stablehlo.reverse %s1b0Wp, dims = [2, 3] : tensor<256x64x1x1xf32>
    %v5228 = stablehlo.transpose %v5227, dims = [1, 0, 2, 3] : (tensor<256x64x1x1xf32>) -> tensor<64x256x1x1xf32>
    %v5229 = stablehlo.convert %v5226 : (tensor<64x256x56x56xf32>) -> tensor<64x256x56x56xbf16>
    %v5230 = stablehlo.convert %v5228 : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xbf16>
    %v5231 = stablehlo.convolution(%v5229, %v5230)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x56x56xbf16>, tensor<64x256x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v5232 = stablehlo.convert %v5231 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v5233 = stablehlo.reshape %v5232 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v5234 = stablehlo.add %v5195, %v5233 : tensor<64x200704xf32>
    %v5235 = stablehlo.reshape %v33 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5236 = stablehlo.reshape %v5187 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5237 = stablehlo.transpose %v5235, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5238 = stablehlo.transpose %v5236, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5239 = stablehlo.convert %v5237 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5240 = stablehlo.convert %v5238 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5241 = stablehlo.convolution(%v5239, %v5240)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x1x1xbf16>
    %v5242 = stablehlo.convert %v5241 : (tensor<64x64x1x1xbf16>) -> tensor<64x64x1x1xf32>
    %v5243 = stablehlo.transpose %v5242, dims = [1, 0, 2, 3] : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %v5244 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5246 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5247 = stablehlo.reduce(%v5244 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5248 = stablehlo.broadcast_in_dim %v5247, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5249 = stablehlo.divide %v5248, %v5246 : tensor<64x64x56x56xf32>
    %v5250 = stablehlo.subtract %v5244, %v5249 : tensor<64x64x56x56xf32>
    %v5251 = stablehlo.multiply %v5250, %v5250 : tensor<64x64x56x56xf32>
    %v5252 = stablehlo.reduce(%v5251 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5253 = stablehlo.broadcast_in_dim %v5252, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5254 = stablehlo.divide %v5253, %v5246 : tensor<64x64x56x56xf32>
    %v5255 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5256 = stablehlo.add %v5254, %v5255 : tensor<64x64x56x56xf32>
    %v5257 = stablehlo.rsqrt %v5256 : tensor<64x64x56x56xf32>
    %v5258 = stablehlo.multiply %v5250, %v5257 : tensor<64x64x56x56xf32>
    %v5259 = stablehlo.reshape %v5157 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5260 = stablehlo.multiply %v5259, %v5258 : tensor<64x64x56x56xf32>
    %v5261 = stablehlo.reduce(%v5260 init: %v5245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5262 = stablehlo.reshape %v5157 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5264 = stablehlo.reduce(%v5262 init: %v5263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5265 = stablehlo.reshape %v63 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5266 = stablehlo.reshape %v5146 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5267 = stablehlo.transpose %v5265, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5268 = stablehlo.transpose %v5266, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5269 = stablehlo.convert %v5267 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5270 = stablehlo.convert %v5268 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5271 = stablehlo.convolution(%v5269, %v5270)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v5272 = stablehlo.convert %v5271 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v5273 = stablehlo.transpose %v5272, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v5274 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5275 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5276 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5277 = stablehlo.reduce(%v5274 init: %v5275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5278 = stablehlo.broadcast_in_dim %v5277, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5279 = stablehlo.divide %v5278, %v5276 : tensor<64x64x56x56xf32>
    %v5280 = stablehlo.subtract %v5274, %v5279 : tensor<64x64x56x56xf32>
    %v5281 = stablehlo.multiply %v5280, %v5280 : tensor<64x64x56x56xf32>
    %v5282 = stablehlo.reduce(%v5281 init: %v5275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5283 = stablehlo.broadcast_in_dim %v5282, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5284 = stablehlo.divide %v5283, %v5276 : tensor<64x64x56x56xf32>
    %v5285 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v5286 = stablehlo.add %v5284, %v5285 : tensor<64x64x56x56xf32>
    %v5287 = stablehlo.rsqrt %v5286 : tensor<64x64x56x56xf32>
    %v5288 = stablehlo.multiply %v5280, %v5287 : tensor<64x64x56x56xf32>
    %v5289 = stablehlo.reshape %v5116 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5290 = stablehlo.multiply %v5289, %v5288 : tensor<64x64x56x56xf32>
    %v5291 = stablehlo.reduce(%v5290 init: %v5275) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5292 = stablehlo.reshape %v5116 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5294 = stablehlo.reduce(%v5292 init: %v5293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5295 = stablehlo.reshape %v93 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5296 = stablehlo.reshape %v5105 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5297 = stablehlo.transpose %v5295, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5298 = stablehlo.transpose %v5296, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5299 = stablehlo.convert %v5297 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5300 = stablehlo.convert %v5298 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5301 = stablehlo.convolution(%v5299, %v5300)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5302 = stablehlo.convert %v5301 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5303 = stablehlo.transpose %v5302, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5304 = stablehlo.reshape %v101 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5305 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5306 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5307 = stablehlo.reduce(%v5304 init: %v5305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5308 = stablehlo.broadcast_in_dim %v5307, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5309 = stablehlo.divide %v5308, %v5306 : tensor<64x256x56x56xf32>
    %v5310 = stablehlo.subtract %v5304, %v5309 : tensor<64x256x56x56xf32>
    %v5311 = stablehlo.multiply %v5310, %v5310 : tensor<64x256x56x56xf32>
    %v5312 = stablehlo.reduce(%v5311 init: %v5305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5313 = stablehlo.broadcast_in_dim %v5312, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5314 = stablehlo.divide %v5313, %v5306 : tensor<64x256x56x56xf32>
    %v5315 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5316 = stablehlo.add %v5314, %v5315 : tensor<64x256x56x56xf32>
    %v5317 = stablehlo.rsqrt %v5316 : tensor<64x256x56x56xf32>
    %v5318 = stablehlo.multiply %v5310, %v5317 : tensor<64x256x56x56xf32>
    %v5319 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5320 = stablehlo.multiply %v5319, %v5318 : tensor<64x256x56x56xf32>
    %v5321 = stablehlo.reduce(%v5320 init: %v5305) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5322 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5323 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5324 = stablehlo.reduce(%v5322 init: %v5323) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5325 = stablehlo.reshape %v33 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5326 = stablehlo.reshape %v5225 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5327 = stablehlo.transpose %v5325, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v5328 = stablehlo.transpose %v5326, dims = [1, 0, 2, 3] : (tensor<64x256x56x56xf32>) -> tensor<256x64x56x56xf32>
    %v5329 = stablehlo.convert %v5327 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v5330 = stablehlo.convert %v5328 : (tensor<256x64x56x56xf32>) -> tensor<256x64x56x56xbf16>
    %v5331 = stablehlo.convolution(%v5329, %v5330)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<256x64x56x56xbf16>) -> tensor<64x256x1x1xbf16>
    %v5332 = stablehlo.convert %v5331 : (tensor<64x256x1x1xbf16>) -> tensor<64x256x1x1xf32>
    %v5333 = stablehlo.transpose %v5332, dims = [1, 0, 2, 3] : (tensor<64x256x1x1xf32>) -> tensor<256x64x1x1xf32>
    %v5334 = stablehlo.reshape %v129 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5335 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5336 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5337 = stablehlo.reduce(%v5334 init: %v5335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5338 = stablehlo.broadcast_in_dim %v5337, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5339 = stablehlo.divide %v5338, %v5336 : tensor<64x256x56x56xf32>
    %v5340 = stablehlo.subtract %v5334, %v5339 : tensor<64x256x56x56xf32>
    %v5341 = stablehlo.multiply %v5340, %v5340 : tensor<64x256x56x56xf32>
    %v5342 = stablehlo.reduce(%v5341 init: %v5335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5343 = stablehlo.broadcast_in_dim %v5342, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5344 = stablehlo.divide %v5343, %v5336 : tensor<64x256x56x56xf32>
    %v5345 = stablehlo.constant dense<1.0e-05> : tensor<64x256x56x56xf32>
    %v5346 = stablehlo.add %v5344, %v5345 : tensor<64x256x56x56xf32>
    %v5347 = stablehlo.rsqrt %v5346 : tensor<64x256x56x56xf32>
    %v5348 = stablehlo.multiply %v5340, %v5347 : tensor<64x256x56x56xf32>
    %v5349 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5350 = stablehlo.multiply %v5349, %v5348 : tensor<64x256x56x56xf32>
    %v5351 = stablehlo.reduce(%v5350 init: %v5335) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5352 = stablehlo.reshape %v5075 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5354 = stablehlo.reduce(%v5352 init: %v5353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5355 = stablehlo.reshape %v29 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5356 = stablehlo.reshape %v5234 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5358 = "stablehlo.select_and_scatter"(%v5355, %v5356, %v5357) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x112x112xf32>
    %v5359 = stablehlo.reshape %v5358 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5360 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v5361 = stablehlo.compare GT, %v27, %v5360 : (tensor<64x802816xf32>, tensor<64x802816xf32>) -> tensor<64x802816xi1>
    %v5362 = stablehlo.select %v5361, %v5359, %v5360 : tensor<64x802816xi1>, tensor<64x802816xf32>
    %v5363 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5364 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5365 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5366 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v5367 = stablehlo.reduce(%v5363 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5368 = stablehlo.broadcast_in_dim %v5367, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5369 = stablehlo.divide %v5368, %v5365 : tensor<64x64x112x112xf32>
    %v5370 = stablehlo.subtract %v5363, %v5369 : tensor<64x64x112x112xf32>
    %v5371 = stablehlo.multiply %v5370, %v5370 : tensor<64x64x112x112xf32>
    %v5372 = stablehlo.reduce(%v5371 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5373 = stablehlo.broadcast_in_dim %v5372, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5374 = stablehlo.divide %v5373, %v5365 : tensor<64x64x112x112xf32>
    %v5375 = stablehlo.add %v5374, %v5366 : tensor<64x64x112x112xf32>
    %v5376 = stablehlo.rsqrt %v5375 : tensor<64x64x112x112xf32>
    %v5377 = stablehlo.multiply %v5370, %v5376 : tensor<64x64x112x112xf32>
    %v5378 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5379 = stablehlo.reshape %v5362 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5380 = stablehlo.multiply %v5378, %v5379 : tensor<64x64x112x112xf32>
    %v5381 = stablehlo.reduce(%v5380 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5382 = stablehlo.broadcast_in_dim %v5381, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5383 = stablehlo.multiply %v5377, %v5380 : tensor<64x64x112x112xf32>
    %v5384 = stablehlo.reduce(%v5383 init: %v5364) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5385 = stablehlo.broadcast_in_dim %v5384, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5386 = stablehlo.multiply %v5380, %v5365 : tensor<64x64x112x112xf32>
    %v5387 = stablehlo.subtract %v5386, %v5382 : tensor<64x64x112x112xf32>
    %v5388 = stablehlo.multiply %v5377, %v5385 : tensor<64x64x112x112xf32>
    %v5389 = stablehlo.subtract %v5387, %v5388 : tensor<64x64x112x112xf32>
    %v5390 = stablehlo.divide %v5376, %v5365 : tensor<64x64x112x112xf32>
    %v5391 = stablehlo.multiply %v5390, %v5389 : tensor<64x64x112x112xf32>
    %v5392 = stablehlo.reshape %v5391 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v5393 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v5394 = stablehlo.reshape %v5392 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5395 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5396 = stablehlo.pad %v5394, %v5395, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x224x224xf32>
    %v5397 = stablehlo.transpose %v5393, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v5398 = stablehlo.transpose %v5396, dims = [1, 0, 2, 3] : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xf32>
    %v5399 = stablehlo.convert %v5397 : (tensor<3x64x224x224xf32>) -> tensor<3x64x224x224xbf16>
    %v5400 = stablehlo.convert %v5398 : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xbf16>
    %v5401 = stablehlo.convolution(%v5399, %v5400)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xbf16>, tensor<64x64x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v5402 = stablehlo.convert %v5401 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v5403 = stablehlo.transpose %v5402, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v5404 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5405 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5406 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5407 = stablehlo.reduce(%v5404 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5408 = stablehlo.broadcast_in_dim %v5407, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5409 = stablehlo.divide %v5408, %v5406 : tensor<64x64x112x112xf32>
    %v5410 = stablehlo.subtract %v5404, %v5409 : tensor<64x64x112x112xf32>
    %v5411 = stablehlo.multiply %v5410, %v5410 : tensor<64x64x112x112xf32>
    %v5412 = stablehlo.reduce(%v5411 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5413 = stablehlo.broadcast_in_dim %v5412, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5414 = stablehlo.divide %v5413, %v5406 : tensor<64x64x112x112xf32>
    %v5415 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v5416 = stablehlo.add %v5414, %v5415 : tensor<64x64x112x112xf32>
    %v5417 = stablehlo.rsqrt %v5416 : tensor<64x64x112x112xf32>
    %v5418 = stablehlo.multiply %v5410, %v5417 : tensor<64x64x112x112xf32>
    %v5419 = stablehlo.reshape %v5362 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5420 = stablehlo.multiply %v5419, %v5418 : tensor<64x64x112x112xf32>
    %v5421 = stablehlo.reduce(%v5420 init: %v5405) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5422 = stablehlo.reshape %v5362 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5423 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5424 = stablehlo.reduce(%v5422 init: %v5423) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5425 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5426 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5427 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5428 = stablehlo.reduce(%v5425 init: %v5426) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5429 = stablehlo.divide %v5428, %v5427 : tensor<64xf32>
    %v5430 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v5431 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5432 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v5433 = stablehlo.reduce(%v5430 init: %v5431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5434 = stablehlo.broadcast_in_dim %v5433, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v5435 = stablehlo.divide %v5434, %v5432 : tensor<64x64x112x112xf32>
    %v5436 = stablehlo.subtract %v5430, %v5435 : tensor<64x64x112x112xf32>
    %v5437 = stablehlo.multiply %v5436, %v5436 : tensor<64x64x112x112xf32>
    %v5438 = stablehlo.reduce(%v5437 init: %v5431) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v5439 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v5440 = stablehlo.divide %v5438, %v5439 : tensor<64xf32>
    %v5441 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5443 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5444 = stablehlo.reduce(%v5441 init: %v5442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5445 = stablehlo.divide %v5444, %v5443 : tensor<64xf32>
    %v5446 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5447 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5448 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5449 = stablehlo.reduce(%v5446 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5450 = stablehlo.broadcast_in_dim %v5449, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5451 = stablehlo.divide %v5450, %v5448 : tensor<64x64x56x56xf32>
    %v5452 = stablehlo.subtract %v5446, %v5451 : tensor<64x64x56x56xf32>
    %v5453 = stablehlo.multiply %v5452, %v5452 : tensor<64x64x56x56xf32>
    %v5454 = stablehlo.reduce(%v5453 init: %v5447) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5455 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5456 = stablehlo.divide %v5454, %v5455 : tensor<64xf32>
    %v5457 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5459 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5460 = stablehlo.reduce(%v5457 init: %v5458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5461 = stablehlo.divide %v5460, %v5459 : tensor<64xf32>
    %v5462 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5463 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5464 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5465 = stablehlo.reduce(%v5462 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5466 = stablehlo.broadcast_in_dim %v5465, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5467 = stablehlo.divide %v5466, %v5464 : tensor<64x64x56x56xf32>
    %v5468 = stablehlo.subtract %v5462, %v5467 : tensor<64x64x56x56xf32>
    %v5469 = stablehlo.multiply %v5468, %v5468 : tensor<64x64x56x56xf32>
    %v5470 = stablehlo.reduce(%v5469 init: %v5463) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5471 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5472 = stablehlo.divide %v5470, %v5471 : tensor<64xf32>
    %v5473 = stablehlo.reshape %v101 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5475 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5476 = stablehlo.reduce(%v5473 init: %v5474) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5477 = stablehlo.divide %v5476, %v5475 : tensor<256xf32>
    %v5478 = stablehlo.reshape %v101 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5479 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5480 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5481 = stablehlo.reduce(%v5478 init: %v5479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5482 = stablehlo.broadcast_in_dim %v5481, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5483 = stablehlo.divide %v5482, %v5480 : tensor<64x256x56x56xf32>
    %v5484 = stablehlo.subtract %v5478, %v5483 : tensor<64x256x56x56xf32>
    %v5485 = stablehlo.multiply %v5484, %v5484 : tensor<64x256x56x56xf32>
    %v5486 = stablehlo.reduce(%v5485 init: %v5479) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5487 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5488 = stablehlo.divide %v5486, %v5487 : tensor<256xf32>
    %v5489 = stablehlo.reshape %v129 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5490 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5491 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5492 = stablehlo.reduce(%v5489 init: %v5490) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5493 = stablehlo.divide %v5492, %v5491 : tensor<256xf32>
    %v5494 = stablehlo.reshape %v129 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5495 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5496 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5497 = stablehlo.reduce(%v5494 init: %v5495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5498 = stablehlo.broadcast_in_dim %v5497, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5499 = stablehlo.divide %v5498, %v5496 : tensor<64x256x56x56xf32>
    %v5500 = stablehlo.subtract %v5494, %v5499 : tensor<64x256x56x56xf32>
    %v5501 = stablehlo.multiply %v5500, %v5500 : tensor<64x256x56x56xf32>
    %v5502 = stablehlo.reduce(%v5501 init: %v5495) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5503 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5504 = stablehlo.divide %v5502, %v5503 : tensor<256xf32>
    %v5505 = stablehlo.reshape %v160 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5507 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5508 = stablehlo.reduce(%v5505 init: %v5506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5509 = stablehlo.divide %v5508, %v5507 : tensor<64xf32>
    %v5510 = stablehlo.reshape %v160 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5511 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5512 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5513 = stablehlo.reduce(%v5510 init: %v5511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5514 = stablehlo.broadcast_in_dim %v5513, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5515 = stablehlo.divide %v5514, %v5512 : tensor<64x64x56x56xf32>
    %v5516 = stablehlo.subtract %v5510, %v5515 : tensor<64x64x56x56xf32>
    %v5517 = stablehlo.multiply %v5516, %v5516 : tensor<64x64x56x56xf32>
    %v5518 = stablehlo.reduce(%v5517 init: %v5511) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5519 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5520 = stablehlo.divide %v5518, %v5519 : tensor<64xf32>
    %v5521 = stablehlo.reshape %v190 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5522 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5523 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5524 = stablehlo.reduce(%v5521 init: %v5522) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5525 = stablehlo.divide %v5524, %v5523 : tensor<64xf32>
    %v5526 = stablehlo.reshape %v190 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5527 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5528 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5529 = stablehlo.reduce(%v5526 init: %v5527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5530 = stablehlo.broadcast_in_dim %v5529, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5531 = stablehlo.divide %v5530, %v5528 : tensor<64x64x56x56xf32>
    %v5532 = stablehlo.subtract %v5526, %v5531 : tensor<64x64x56x56xf32>
    %v5533 = stablehlo.multiply %v5532, %v5532 : tensor<64x64x56x56xf32>
    %v5534 = stablehlo.reduce(%v5533 init: %v5527) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5535 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5536 = stablehlo.divide %v5534, %v5535 : tensor<64xf32>
    %v5537 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5538 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5539 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5540 = stablehlo.reduce(%v5537 init: %v5538) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5541 = stablehlo.divide %v5540, %v5539 : tensor<256xf32>
    %v5542 = stablehlo.reshape %v220 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5543 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5544 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5545 = stablehlo.reduce(%v5542 init: %v5543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5546 = stablehlo.broadcast_in_dim %v5545, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5547 = stablehlo.divide %v5546, %v5544 : tensor<64x256x56x56xf32>
    %v5548 = stablehlo.subtract %v5542, %v5547 : tensor<64x256x56x56xf32>
    %v5549 = stablehlo.multiply %v5548, %v5548 : tensor<64x256x56x56xf32>
    %v5550 = stablehlo.reduce(%v5549 init: %v5543) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5551 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5552 = stablehlo.divide %v5550, %v5551 : tensor<256xf32>
    %v5553 = stablehlo.reshape %v251 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5554 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5555 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5556 = stablehlo.reduce(%v5553 init: %v5554) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5557 = stablehlo.divide %v5556, %v5555 : tensor<64xf32>
    %v5558 = stablehlo.reshape %v251 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5559 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5560 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5561 = stablehlo.reduce(%v5558 init: %v5559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5562 = stablehlo.broadcast_in_dim %v5561, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5563 = stablehlo.divide %v5562, %v5560 : tensor<64x64x56x56xf32>
    %v5564 = stablehlo.subtract %v5558, %v5563 : tensor<64x64x56x56xf32>
    %v5565 = stablehlo.multiply %v5564, %v5564 : tensor<64x64x56x56xf32>
    %v5566 = stablehlo.reduce(%v5565 init: %v5559) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5567 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5568 = stablehlo.divide %v5566, %v5567 : tensor<64xf32>
    %v5569 = stablehlo.reshape %v281 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5571 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5572 = stablehlo.reduce(%v5569 init: %v5570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5573 = stablehlo.divide %v5572, %v5571 : tensor<64xf32>
    %v5574 = stablehlo.reshape %v281 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v5575 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5576 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v5577 = stablehlo.reduce(%v5574 init: %v5575) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5578 = stablehlo.broadcast_in_dim %v5577, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v5579 = stablehlo.divide %v5578, %v5576 : tensor<64x64x56x56xf32>
    %v5580 = stablehlo.subtract %v5574, %v5579 : tensor<64x64x56x56xf32>
    %v5581 = stablehlo.multiply %v5580, %v5580 : tensor<64x64x56x56xf32>
    %v5582 = stablehlo.reduce(%v5581 init: %v5575) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v5583 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v5584 = stablehlo.divide %v5582, %v5583 : tensor<64xf32>
    %v5585 = stablehlo.reshape %v311 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5587 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5588 = stablehlo.reduce(%v5585 init: %v5586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5589 = stablehlo.divide %v5588, %v5587 : tensor<256xf32>
    %v5590 = stablehlo.reshape %v311 : (tensor<64x802816xf32>) -> tensor<64x256x56x56xf32>
    %v5591 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5592 = stablehlo.constant dense<200704.0> : tensor<64x256x56x56xf32>
    %v5593 = stablehlo.reduce(%v5590 init: %v5591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5594 = stablehlo.broadcast_in_dim %v5593, dims = [1] : (tensor<256xf32>) -> tensor<64x256x56x56xf32>
    %v5595 = stablehlo.divide %v5594, %v5592 : tensor<64x256x56x56xf32>
    %v5596 = stablehlo.subtract %v5590, %v5595 : tensor<64x256x56x56xf32>
    %v5597 = stablehlo.multiply %v5596, %v5596 : tensor<64x256x56x56xf32>
    %v5598 = stablehlo.reduce(%v5597 init: %v5591) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x56x56xf32>, tensor<f32>) -> tensor<256xf32>
    %v5599 = stablehlo.constant dense<200704.0> : tensor<256xf32>
    %v5600 = stablehlo.divide %v5598, %v5599 : tensor<256xf32>
    %v5601 = stablehlo.reshape %v342 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5603 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5604 = stablehlo.reduce(%v5601 init: %v5602) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5605 = stablehlo.divide %v5604, %v5603 : tensor<128xf32>
    %v5606 = stablehlo.reshape %v342 : (tensor<64x401408xf32>) -> tensor<64x128x56x56xf32>
    %v5607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5608 = stablehlo.constant dense<200704.0> : tensor<64x128x56x56xf32>
    %v5609 = stablehlo.reduce(%v5606 init: %v5607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5610 = stablehlo.broadcast_in_dim %v5609, dims = [1] : (tensor<128xf32>) -> tensor<64x128x56x56xf32>
    %v5611 = stablehlo.divide %v5610, %v5608 : tensor<64x128x56x56xf32>
    %v5612 = stablehlo.subtract %v5606, %v5611 : tensor<64x128x56x56xf32>
    %v5613 = stablehlo.multiply %v5612, %v5612 : tensor<64x128x56x56xf32>
    %v5614 = stablehlo.reduce(%v5613 init: %v5607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x56x56xf32>, tensor<f32>) -> tensor<128xf32>
    %v5615 = stablehlo.constant dense<200704.0> : tensor<128xf32>
    %v5616 = stablehlo.divide %v5614, %v5615 : tensor<128xf32>
    %v5617 = stablehlo.reshape %v372 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5618 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5619 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5620 = stablehlo.reduce(%v5617 init: %v5618) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5621 = stablehlo.divide %v5620, %v5619 : tensor<128xf32>
    %v5622 = stablehlo.reshape %v372 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5624 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5625 = stablehlo.reduce(%v5622 init: %v5623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5626 = stablehlo.broadcast_in_dim %v5625, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5627 = stablehlo.divide %v5626, %v5624 : tensor<64x128x28x28xf32>
    %v5628 = stablehlo.subtract %v5622, %v5627 : tensor<64x128x28x28xf32>
    %v5629 = stablehlo.multiply %v5628, %v5628 : tensor<64x128x28x28xf32>
    %v5630 = stablehlo.reduce(%v5629 init: %v5623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5631 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5632 = stablehlo.divide %v5630, %v5631 : tensor<128xf32>
    %v5633 = stablehlo.reshape %v402 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5634 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5635 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5636 = stablehlo.reduce(%v5633 init: %v5634) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5637 = stablehlo.divide %v5636, %v5635 : tensor<512xf32>
    %v5638 = stablehlo.reshape %v402 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5639 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5640 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5641 = stablehlo.reduce(%v5638 init: %v5639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5642 = stablehlo.broadcast_in_dim %v5641, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5643 = stablehlo.divide %v5642, %v5640 : tensor<64x512x28x28xf32>
    %v5644 = stablehlo.subtract %v5638, %v5643 : tensor<64x512x28x28xf32>
    %v5645 = stablehlo.multiply %v5644, %v5644 : tensor<64x512x28x28xf32>
    %v5646 = stablehlo.reduce(%v5645 init: %v5639) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5647 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5648 = stablehlo.divide %v5646, %v5647 : tensor<512xf32>
    %v5649 = stablehlo.reshape %v430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5650 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5651 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5652 = stablehlo.reduce(%v5649 init: %v5650) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5653 = stablehlo.divide %v5652, %v5651 : tensor<512xf32>
    %v5654 = stablehlo.reshape %v430 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5655 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5656 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5657 = stablehlo.reduce(%v5654 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5658 = stablehlo.broadcast_in_dim %v5657, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5659 = stablehlo.divide %v5658, %v5656 : tensor<64x512x28x28xf32>
    %v5660 = stablehlo.subtract %v5654, %v5659 : tensor<64x512x28x28xf32>
    %v5661 = stablehlo.multiply %v5660, %v5660 : tensor<64x512x28x28xf32>
    %v5662 = stablehlo.reduce(%v5661 init: %v5655) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5663 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5664 = stablehlo.divide %v5662, %v5663 : tensor<512xf32>
    %v5665 = stablehlo.reshape %v461 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5666 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5667 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5668 = stablehlo.reduce(%v5665 init: %v5666) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5669 = stablehlo.divide %v5668, %v5667 : tensor<128xf32>
    %v5670 = stablehlo.reshape %v461 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5671 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5672 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5673 = stablehlo.reduce(%v5670 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5674 = stablehlo.broadcast_in_dim %v5673, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5675 = stablehlo.divide %v5674, %v5672 : tensor<64x128x28x28xf32>
    %v5676 = stablehlo.subtract %v5670, %v5675 : tensor<64x128x28x28xf32>
    %v5677 = stablehlo.multiply %v5676, %v5676 : tensor<64x128x28x28xf32>
    %v5678 = stablehlo.reduce(%v5677 init: %v5671) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5679 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5680 = stablehlo.divide %v5678, %v5679 : tensor<128xf32>
    %v5681 = stablehlo.reshape %v491 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5682 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5683 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5684 = stablehlo.reduce(%v5681 init: %v5682) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5685 = stablehlo.divide %v5684, %v5683 : tensor<128xf32>
    %v5686 = stablehlo.reshape %v491 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5687 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5688 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5689 = stablehlo.reduce(%v5686 init: %v5687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5690 = stablehlo.broadcast_in_dim %v5689, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5691 = stablehlo.divide %v5690, %v5688 : tensor<64x128x28x28xf32>
    %v5692 = stablehlo.subtract %v5686, %v5691 : tensor<64x128x28x28xf32>
    %v5693 = stablehlo.multiply %v5692, %v5692 : tensor<64x128x28x28xf32>
    %v5694 = stablehlo.reduce(%v5693 init: %v5687) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5695 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5696 = stablehlo.divide %v5694, %v5695 : tensor<128xf32>
    %v5697 = stablehlo.reshape %v521 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5698 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5699 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5700 = stablehlo.reduce(%v5697 init: %v5698) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5701 = stablehlo.divide %v5700, %v5699 : tensor<512xf32>
    %v5702 = stablehlo.reshape %v521 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5703 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5704 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5705 = stablehlo.reduce(%v5702 init: %v5703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5706 = stablehlo.broadcast_in_dim %v5705, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5707 = stablehlo.divide %v5706, %v5704 : tensor<64x512x28x28xf32>
    %v5708 = stablehlo.subtract %v5702, %v5707 : tensor<64x512x28x28xf32>
    %v5709 = stablehlo.multiply %v5708, %v5708 : tensor<64x512x28x28xf32>
    %v5710 = stablehlo.reduce(%v5709 init: %v5703) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5711 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5712 = stablehlo.divide %v5710, %v5711 : tensor<512xf32>
    %v5713 = stablehlo.reshape %v552 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5715 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5716 = stablehlo.reduce(%v5713 init: %v5714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5717 = stablehlo.divide %v5716, %v5715 : tensor<128xf32>
    %v5718 = stablehlo.reshape %v552 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5720 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5721 = stablehlo.reduce(%v5718 init: %v5719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5722 = stablehlo.broadcast_in_dim %v5721, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5723 = stablehlo.divide %v5722, %v5720 : tensor<64x128x28x28xf32>
    %v5724 = stablehlo.subtract %v5718, %v5723 : tensor<64x128x28x28xf32>
    %v5725 = stablehlo.multiply %v5724, %v5724 : tensor<64x128x28x28xf32>
    %v5726 = stablehlo.reduce(%v5725 init: %v5719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5727 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5728 = stablehlo.divide %v5726, %v5727 : tensor<128xf32>
    %v5729 = stablehlo.reshape %v582 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5731 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5732 = stablehlo.reduce(%v5729 init: %v5730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5733 = stablehlo.divide %v5732, %v5731 : tensor<128xf32>
    %v5734 = stablehlo.reshape %v582 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5736 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5737 = stablehlo.reduce(%v5734 init: %v5735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5738 = stablehlo.broadcast_in_dim %v5737, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5739 = stablehlo.divide %v5738, %v5736 : tensor<64x128x28x28xf32>
    %v5740 = stablehlo.subtract %v5734, %v5739 : tensor<64x128x28x28xf32>
    %v5741 = stablehlo.multiply %v5740, %v5740 : tensor<64x128x28x28xf32>
    %v5742 = stablehlo.reduce(%v5741 init: %v5735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5743 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5744 = stablehlo.divide %v5742, %v5743 : tensor<128xf32>
    %v5745 = stablehlo.reshape %v612 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5747 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5748 = stablehlo.reduce(%v5745 init: %v5746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5749 = stablehlo.divide %v5748, %v5747 : tensor<512xf32>
    %v5750 = stablehlo.reshape %v612 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5752 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5753 = stablehlo.reduce(%v5750 init: %v5751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5754 = stablehlo.broadcast_in_dim %v5753, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5755 = stablehlo.divide %v5754, %v5752 : tensor<64x512x28x28xf32>
    %v5756 = stablehlo.subtract %v5750, %v5755 : tensor<64x512x28x28xf32>
    %v5757 = stablehlo.multiply %v5756, %v5756 : tensor<64x512x28x28xf32>
    %v5758 = stablehlo.reduce(%v5757 init: %v5751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5759 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5760 = stablehlo.divide %v5758, %v5759 : tensor<512xf32>
    %v5761 = stablehlo.reshape %v643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5763 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5764 = stablehlo.reduce(%v5761 init: %v5762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5765 = stablehlo.divide %v5764, %v5763 : tensor<128xf32>
    %v5766 = stablehlo.reshape %v643 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5768 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5769 = stablehlo.reduce(%v5766 init: %v5767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5770 = stablehlo.broadcast_in_dim %v5769, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5771 = stablehlo.divide %v5770, %v5768 : tensor<64x128x28x28xf32>
    %v5772 = stablehlo.subtract %v5766, %v5771 : tensor<64x128x28x28xf32>
    %v5773 = stablehlo.multiply %v5772, %v5772 : tensor<64x128x28x28xf32>
    %v5774 = stablehlo.reduce(%v5773 init: %v5767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5775 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5776 = stablehlo.divide %v5774, %v5775 : tensor<128xf32>
    %v5777 = stablehlo.reshape %v673 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5779 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5780 = stablehlo.reduce(%v5777 init: %v5778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5781 = stablehlo.divide %v5780, %v5779 : tensor<128xf32>
    %v5782 = stablehlo.reshape %v673 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v5783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5784 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v5785 = stablehlo.reduce(%v5782 init: %v5783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5786 = stablehlo.broadcast_in_dim %v5785, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v5787 = stablehlo.divide %v5786, %v5784 : tensor<64x128x28x28xf32>
    %v5788 = stablehlo.subtract %v5782, %v5787 : tensor<64x128x28x28xf32>
    %v5789 = stablehlo.multiply %v5788, %v5788 : tensor<64x128x28x28xf32>
    %v5790 = stablehlo.reduce(%v5789 init: %v5783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v5791 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v5792 = stablehlo.divide %v5790, %v5791 : tensor<128xf32>
    %v5793 = stablehlo.reshape %v703 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5795 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5796 = stablehlo.reduce(%v5793 init: %v5794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5797 = stablehlo.divide %v5796, %v5795 : tensor<512xf32>
    %v5798 = stablehlo.reshape %v703 : (tensor<64x401408xf32>) -> tensor<64x512x28x28xf32>
    %v5799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5800 = stablehlo.constant dense<50176.0> : tensor<64x512x28x28xf32>
    %v5801 = stablehlo.reduce(%v5798 init: %v5799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5802 = stablehlo.broadcast_in_dim %v5801, dims = [1] : (tensor<512xf32>) -> tensor<64x512x28x28xf32>
    %v5803 = stablehlo.divide %v5802, %v5800 : tensor<64x512x28x28xf32>
    %v5804 = stablehlo.subtract %v5798, %v5803 : tensor<64x512x28x28xf32>
    %v5805 = stablehlo.multiply %v5804, %v5804 : tensor<64x512x28x28xf32>
    %v5806 = stablehlo.reduce(%v5805 init: %v5799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x28x28xf32>, tensor<f32>) -> tensor<512xf32>
    %v5807 = stablehlo.constant dense<50176.0> : tensor<512xf32>
    %v5808 = stablehlo.divide %v5806, %v5807 : tensor<512xf32>
    %v5809 = stablehlo.reshape %v734 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v5810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5811 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5812 = stablehlo.reduce(%v5809 init: %v5810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5813 = stablehlo.divide %v5812, %v5811 : tensor<256xf32>
    %v5814 = stablehlo.reshape %v734 : (tensor<64x200704xf32>) -> tensor<64x256x28x28xf32>
    %v5815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5816 = stablehlo.constant dense<50176.0> : tensor<64x256x28x28xf32>
    %v5817 = stablehlo.reduce(%v5814 init: %v5815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5818 = stablehlo.broadcast_in_dim %v5817, dims = [1] : (tensor<256xf32>) -> tensor<64x256x28x28xf32>
    %v5819 = stablehlo.divide %v5818, %v5816 : tensor<64x256x28x28xf32>
    %v5820 = stablehlo.subtract %v5814, %v5819 : tensor<64x256x28x28xf32>
    %v5821 = stablehlo.multiply %v5820, %v5820 : tensor<64x256x28x28xf32>
    %v5822 = stablehlo.reduce(%v5821 init: %v5815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x28x28xf32>, tensor<f32>) -> tensor<256xf32>
    %v5823 = stablehlo.constant dense<50176.0> : tensor<256xf32>
    %v5824 = stablehlo.divide %v5822, %v5823 : tensor<256xf32>
    %v5825 = stablehlo.reshape %v764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5827 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5828 = stablehlo.reduce(%v5825 init: %v5826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5829 = stablehlo.divide %v5828, %v5827 : tensor<256xf32>
    %v5830 = stablehlo.reshape %v764 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5832 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5833 = stablehlo.reduce(%v5830 init: %v5831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5834 = stablehlo.broadcast_in_dim %v5833, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5835 = stablehlo.divide %v5834, %v5832 : tensor<64x256x14x14xf32>
    %v5836 = stablehlo.subtract %v5830, %v5835 : tensor<64x256x14x14xf32>
    %v5837 = stablehlo.multiply %v5836, %v5836 : tensor<64x256x14x14xf32>
    %v5838 = stablehlo.reduce(%v5837 init: %v5831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5839 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5840 = stablehlo.divide %v5838, %v5839 : tensor<256xf32>
    %v5841 = stablehlo.reshape %v794 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5843 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5844 = stablehlo.reduce(%v5841 init: %v5842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5845 = stablehlo.divide %v5844, %v5843 : tensor<1024xf32>
    %v5846 = stablehlo.reshape %v794 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5848 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v5849 = stablehlo.reduce(%v5846 init: %v5847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5850 = stablehlo.broadcast_in_dim %v5849, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v5851 = stablehlo.divide %v5850, %v5848 : tensor<64x1024x14x14xf32>
    %v5852 = stablehlo.subtract %v5846, %v5851 : tensor<64x1024x14x14xf32>
    %v5853 = stablehlo.multiply %v5852, %v5852 : tensor<64x1024x14x14xf32>
    %v5854 = stablehlo.reduce(%v5853 init: %v5847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5855 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5856 = stablehlo.divide %v5854, %v5855 : tensor<1024xf32>
    %v5857 = stablehlo.reshape %v822 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5859 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5860 = stablehlo.reduce(%v5857 init: %v5858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5861 = stablehlo.divide %v5860, %v5859 : tensor<1024xf32>
    %v5862 = stablehlo.reshape %v822 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5864 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v5865 = stablehlo.reduce(%v5862 init: %v5863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5866 = stablehlo.broadcast_in_dim %v5865, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v5867 = stablehlo.divide %v5866, %v5864 : tensor<64x1024x14x14xf32>
    %v5868 = stablehlo.subtract %v5862, %v5867 : tensor<64x1024x14x14xf32>
    %v5869 = stablehlo.multiply %v5868, %v5868 : tensor<64x1024x14x14xf32>
    %v5870 = stablehlo.reduce(%v5869 init: %v5863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5871 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5872 = stablehlo.divide %v5870, %v5871 : tensor<1024xf32>
    %v5873 = stablehlo.reshape %v853 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5875 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5876 = stablehlo.reduce(%v5873 init: %v5874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5877 = stablehlo.divide %v5876, %v5875 : tensor<256xf32>
    %v5878 = stablehlo.reshape %v853 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5880 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5881 = stablehlo.reduce(%v5878 init: %v5879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5882 = stablehlo.broadcast_in_dim %v5881, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5883 = stablehlo.divide %v5882, %v5880 : tensor<64x256x14x14xf32>
    %v5884 = stablehlo.subtract %v5878, %v5883 : tensor<64x256x14x14xf32>
    %v5885 = stablehlo.multiply %v5884, %v5884 : tensor<64x256x14x14xf32>
    %v5886 = stablehlo.reduce(%v5885 init: %v5879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5887 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5888 = stablehlo.divide %v5886, %v5887 : tensor<256xf32>
    %v5889 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5891 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5892 = stablehlo.reduce(%v5889 init: %v5890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5893 = stablehlo.divide %v5892, %v5891 : tensor<256xf32>
    %v5894 = stablehlo.reshape %v883 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5896 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5897 = stablehlo.reduce(%v5894 init: %v5895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5898 = stablehlo.broadcast_in_dim %v5897, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5899 = stablehlo.divide %v5898, %v5896 : tensor<64x256x14x14xf32>
    %v5900 = stablehlo.subtract %v5894, %v5899 : tensor<64x256x14x14xf32>
    %v5901 = stablehlo.multiply %v5900, %v5900 : tensor<64x256x14x14xf32>
    %v5902 = stablehlo.reduce(%v5901 init: %v5895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5903 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5904 = stablehlo.divide %v5902, %v5903 : tensor<256xf32>
    %v5905 = stablehlo.reshape %v913 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5907 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5908 = stablehlo.reduce(%v5905 init: %v5906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5909 = stablehlo.divide %v5908, %v5907 : tensor<1024xf32>
    %v5910 = stablehlo.reshape %v913 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5912 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v5913 = stablehlo.reduce(%v5910 init: %v5911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5914 = stablehlo.broadcast_in_dim %v5913, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v5915 = stablehlo.divide %v5914, %v5912 : tensor<64x1024x14x14xf32>
    %v5916 = stablehlo.subtract %v5910, %v5915 : tensor<64x1024x14x14xf32>
    %v5917 = stablehlo.multiply %v5916, %v5916 : tensor<64x1024x14x14xf32>
    %v5918 = stablehlo.reduce(%v5917 init: %v5911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5919 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5920 = stablehlo.divide %v5918, %v5919 : tensor<1024xf32>
    %v5921 = stablehlo.reshape %v944 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5923 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5924 = stablehlo.reduce(%v5921 init: %v5922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5925 = stablehlo.divide %v5924, %v5923 : tensor<256xf32>
    %v5926 = stablehlo.reshape %v944 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5928 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5929 = stablehlo.reduce(%v5926 init: %v5927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5930 = stablehlo.broadcast_in_dim %v5929, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5931 = stablehlo.divide %v5930, %v5928 : tensor<64x256x14x14xf32>
    %v5932 = stablehlo.subtract %v5926, %v5931 : tensor<64x256x14x14xf32>
    %v5933 = stablehlo.multiply %v5932, %v5932 : tensor<64x256x14x14xf32>
    %v5934 = stablehlo.reduce(%v5933 init: %v5927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5935 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5936 = stablehlo.divide %v5934, %v5935 : tensor<256xf32>
    %v5937 = stablehlo.reshape %v974 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5939 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5940 = stablehlo.reduce(%v5937 init: %v5938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5941 = stablehlo.divide %v5940, %v5939 : tensor<256xf32>
    %v5942 = stablehlo.reshape %v974 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5944 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5945 = stablehlo.reduce(%v5942 init: %v5943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5946 = stablehlo.broadcast_in_dim %v5945, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5947 = stablehlo.divide %v5946, %v5944 : tensor<64x256x14x14xf32>
    %v5948 = stablehlo.subtract %v5942, %v5947 : tensor<64x256x14x14xf32>
    %v5949 = stablehlo.multiply %v5948, %v5948 : tensor<64x256x14x14xf32>
    %v5950 = stablehlo.reduce(%v5949 init: %v5943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5951 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5952 = stablehlo.divide %v5950, %v5951 : tensor<256xf32>
    %v5953 = stablehlo.reshape %v1004 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5955 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5956 = stablehlo.reduce(%v5953 init: %v5954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5957 = stablehlo.divide %v5956, %v5955 : tensor<1024xf32>
    %v5958 = stablehlo.reshape %v1004 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v5959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5960 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v5961 = stablehlo.reduce(%v5958 init: %v5959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5962 = stablehlo.broadcast_in_dim %v5961, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v5963 = stablehlo.divide %v5962, %v5960 : tensor<64x1024x14x14xf32>
    %v5964 = stablehlo.subtract %v5958, %v5963 : tensor<64x1024x14x14xf32>
    %v5965 = stablehlo.multiply %v5964, %v5964 : tensor<64x1024x14x14xf32>
    %v5966 = stablehlo.reduce(%v5965 init: %v5959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v5967 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v5968 = stablehlo.divide %v5966, %v5967 : tensor<1024xf32>
    %v5969 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5971 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5972 = stablehlo.reduce(%v5969 init: %v5970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5973 = stablehlo.divide %v5972, %v5971 : tensor<256xf32>
    %v5974 = stablehlo.reshape %v1035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5976 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5977 = stablehlo.reduce(%v5974 init: %v5975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5978 = stablehlo.broadcast_in_dim %v5977, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5979 = stablehlo.divide %v5978, %v5976 : tensor<64x256x14x14xf32>
    %v5980 = stablehlo.subtract %v5974, %v5979 : tensor<64x256x14x14xf32>
    %v5981 = stablehlo.multiply %v5980, %v5980 : tensor<64x256x14x14xf32>
    %v5982 = stablehlo.reduce(%v5981 init: %v5975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5983 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5984 = stablehlo.divide %v5982, %v5983 : tensor<256xf32>
    %v5985 = stablehlo.reshape %v1065 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5987 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v5988 = stablehlo.reduce(%v5985 init: %v5986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5989 = stablehlo.divide %v5988, %v5987 : tensor<256xf32>
    %v5990 = stablehlo.reshape %v1065 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v5991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v5992 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v5993 = stablehlo.reduce(%v5990 init: %v5991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5994 = stablehlo.broadcast_in_dim %v5993, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v5995 = stablehlo.divide %v5994, %v5992 : tensor<64x256x14x14xf32>
    %v5996 = stablehlo.subtract %v5990, %v5995 : tensor<64x256x14x14xf32>
    %v5997 = stablehlo.multiply %v5996, %v5996 : tensor<64x256x14x14xf32>
    %v5998 = stablehlo.reduce(%v5997 init: %v5991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v5999 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6000 = stablehlo.divide %v5998, %v5999 : tensor<256xf32>
    %v6001 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6003 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6004 = stablehlo.reduce(%v6001 init: %v6002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6005 = stablehlo.divide %v6004, %v6003 : tensor<1024xf32>
    %v6006 = stablehlo.reshape %v1095 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6008 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6009 = stablehlo.reduce(%v6006 init: %v6007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6010 = stablehlo.broadcast_in_dim %v6009, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6011 = stablehlo.divide %v6010, %v6008 : tensor<64x1024x14x14xf32>
    %v6012 = stablehlo.subtract %v6006, %v6011 : tensor<64x1024x14x14xf32>
    %v6013 = stablehlo.multiply %v6012, %v6012 : tensor<64x1024x14x14xf32>
    %v6014 = stablehlo.reduce(%v6013 init: %v6007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6015 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6016 = stablehlo.divide %v6014, %v6015 : tensor<1024xf32>
    %v6017 = stablehlo.reshape %v1126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6019 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6020 = stablehlo.reduce(%v6017 init: %v6018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6021 = stablehlo.divide %v6020, %v6019 : tensor<256xf32>
    %v6022 = stablehlo.reshape %v1126 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6024 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6025 = stablehlo.reduce(%v6022 init: %v6023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6026 = stablehlo.broadcast_in_dim %v6025, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6027 = stablehlo.divide %v6026, %v6024 : tensor<64x256x14x14xf32>
    %v6028 = stablehlo.subtract %v6022, %v6027 : tensor<64x256x14x14xf32>
    %v6029 = stablehlo.multiply %v6028, %v6028 : tensor<64x256x14x14xf32>
    %v6030 = stablehlo.reduce(%v6029 init: %v6023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6031 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6032 = stablehlo.divide %v6030, %v6031 : tensor<256xf32>
    %v6033 = stablehlo.reshape %v1156 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6035 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6036 = stablehlo.reduce(%v6033 init: %v6034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6037 = stablehlo.divide %v6036, %v6035 : tensor<256xf32>
    %v6038 = stablehlo.reshape %v1156 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6040 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6041 = stablehlo.reduce(%v6038 init: %v6039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6042 = stablehlo.broadcast_in_dim %v6041, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6043 = stablehlo.divide %v6042, %v6040 : tensor<64x256x14x14xf32>
    %v6044 = stablehlo.subtract %v6038, %v6043 : tensor<64x256x14x14xf32>
    %v6045 = stablehlo.multiply %v6044, %v6044 : tensor<64x256x14x14xf32>
    %v6046 = stablehlo.reduce(%v6045 init: %v6039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6047 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6048 = stablehlo.divide %v6046, %v6047 : tensor<256xf32>
    %v6049 = stablehlo.reshape %v1186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6051 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6052 = stablehlo.reduce(%v6049 init: %v6050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6053 = stablehlo.divide %v6052, %v6051 : tensor<1024xf32>
    %v6054 = stablehlo.reshape %v1186 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6056 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6057 = stablehlo.reduce(%v6054 init: %v6055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6058 = stablehlo.broadcast_in_dim %v6057, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6059 = stablehlo.divide %v6058, %v6056 : tensor<64x1024x14x14xf32>
    %v6060 = stablehlo.subtract %v6054, %v6059 : tensor<64x1024x14x14xf32>
    %v6061 = stablehlo.multiply %v6060, %v6060 : tensor<64x1024x14x14xf32>
    %v6062 = stablehlo.reduce(%v6061 init: %v6055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6063 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6064 = stablehlo.divide %v6062, %v6063 : tensor<1024xf32>
    %v6065 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6067 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6068 = stablehlo.reduce(%v6065 init: %v6066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6069 = stablehlo.divide %v6068, %v6067 : tensor<256xf32>
    %v6070 = stablehlo.reshape %v1217 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6072 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6073 = stablehlo.reduce(%v6070 init: %v6071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6074 = stablehlo.broadcast_in_dim %v6073, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6075 = stablehlo.divide %v6074, %v6072 : tensor<64x256x14x14xf32>
    %v6076 = stablehlo.subtract %v6070, %v6075 : tensor<64x256x14x14xf32>
    %v6077 = stablehlo.multiply %v6076, %v6076 : tensor<64x256x14x14xf32>
    %v6078 = stablehlo.reduce(%v6077 init: %v6071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6079 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6080 = stablehlo.divide %v6078, %v6079 : tensor<256xf32>
    %v6081 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6083 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6084 = stablehlo.reduce(%v6081 init: %v6082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6085 = stablehlo.divide %v6084, %v6083 : tensor<256xf32>
    %v6086 = stablehlo.reshape %v1247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v6087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6088 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v6089 = stablehlo.reduce(%v6086 init: %v6087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6090 = stablehlo.broadcast_in_dim %v6089, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v6091 = stablehlo.divide %v6090, %v6088 : tensor<64x256x14x14xf32>
    %v6092 = stablehlo.subtract %v6086, %v6091 : tensor<64x256x14x14xf32>
    %v6093 = stablehlo.multiply %v6092, %v6092 : tensor<64x256x14x14xf32>
    %v6094 = stablehlo.reduce(%v6093 init: %v6087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v6095 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v6096 = stablehlo.divide %v6094, %v6095 : tensor<256xf32>
    %v6097 = stablehlo.reshape %v1277 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6099 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6100 = stablehlo.reduce(%v6097 init: %v6098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6101 = stablehlo.divide %v6100, %v6099 : tensor<1024xf32>
    %v6102 = stablehlo.reshape %v1277 : (tensor<64x200704xf32>) -> tensor<64x1024x14x14xf32>
    %v6103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6104 = stablehlo.constant dense<12544.0> : tensor<64x1024x14x14xf32>
    %v6105 = stablehlo.reduce(%v6102 init: %v6103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6106 = stablehlo.broadcast_in_dim %v6105, dims = [1] : (tensor<1024xf32>) -> tensor<64x1024x14x14xf32>
    %v6107 = stablehlo.divide %v6106, %v6104 : tensor<64x1024x14x14xf32>
    %v6108 = stablehlo.subtract %v6102, %v6107 : tensor<64x1024x14x14xf32>
    %v6109 = stablehlo.multiply %v6108, %v6108 : tensor<64x1024x14x14xf32>
    %v6110 = stablehlo.reduce(%v6109 init: %v6103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x1024x14x14xf32>, tensor<f32>) -> tensor<1024xf32>
    %v6111 = stablehlo.constant dense<12544.0> : tensor<1024xf32>
    %v6112 = stablehlo.divide %v6110, %v6111 : tensor<1024xf32>
    %v6113 = stablehlo.reshape %v1308 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v6114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6115 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6116 = stablehlo.reduce(%v6113 init: %v6114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6117 = stablehlo.divide %v6116, %v6115 : tensor<512xf32>
    %v6118 = stablehlo.reshape %v1308 : (tensor<64x100352xf32>) -> tensor<64x512x14x14xf32>
    %v6119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6120 = stablehlo.constant dense<12544.0> : tensor<64x512x14x14xf32>
    %v6121 = stablehlo.reduce(%v6118 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6122 = stablehlo.broadcast_in_dim %v6121, dims = [1] : (tensor<512xf32>) -> tensor<64x512x14x14xf32>
    %v6123 = stablehlo.divide %v6122, %v6120 : tensor<64x512x14x14xf32>
    %v6124 = stablehlo.subtract %v6118, %v6123 : tensor<64x512x14x14xf32>
    %v6125 = stablehlo.multiply %v6124, %v6124 : tensor<64x512x14x14xf32>
    %v6126 = stablehlo.reduce(%v6125 init: %v6119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x14x14xf32>, tensor<f32>) -> tensor<512xf32>
    %v6127 = stablehlo.constant dense<12544.0> : tensor<512xf32>
    %v6128 = stablehlo.divide %v6126, %v6127 : tensor<512xf32>
    %v6129 = stablehlo.reshape %v1338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6131 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6132 = stablehlo.reduce(%v6129 init: %v6130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6133 = stablehlo.divide %v6132, %v6131 : tensor<512xf32>
    %v6134 = stablehlo.reshape %v1338 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6136 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6137 = stablehlo.reduce(%v6134 init: %v6135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6138 = stablehlo.broadcast_in_dim %v6137, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6139 = stablehlo.divide %v6138, %v6136 : tensor<64x512x7x7xf32>
    %v6140 = stablehlo.subtract %v6134, %v6139 : tensor<64x512x7x7xf32>
    %v6141 = stablehlo.multiply %v6140, %v6140 : tensor<64x512x7x7xf32>
    %v6142 = stablehlo.reduce(%v6141 init: %v6135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6143 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6144 = stablehlo.divide %v6142, %v6143 : tensor<512xf32>
    %v6145 = stablehlo.reshape %v1368 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6147 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6148 = stablehlo.reduce(%v6145 init: %v6146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6149 = stablehlo.divide %v6148, %v6147 : tensor<2048xf32>
    %v6150 = stablehlo.reshape %v1368 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6152 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6153 = stablehlo.reduce(%v6150 init: %v6151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6154 = stablehlo.broadcast_in_dim %v6153, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6155 = stablehlo.divide %v6154, %v6152 : tensor<64x2048x7x7xf32>
    %v6156 = stablehlo.subtract %v6150, %v6155 : tensor<64x2048x7x7xf32>
    %v6157 = stablehlo.multiply %v6156, %v6156 : tensor<64x2048x7x7xf32>
    %v6158 = stablehlo.reduce(%v6157 init: %v6151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6159 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6160 = stablehlo.divide %v6158, %v6159 : tensor<2048xf32>
    %v6161 = stablehlo.reshape %v1396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6163 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6164 = stablehlo.reduce(%v6161 init: %v6162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6165 = stablehlo.divide %v6164, %v6163 : tensor<2048xf32>
    %v6166 = stablehlo.reshape %v1396 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6168 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6169 = stablehlo.reduce(%v6166 init: %v6167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6170 = stablehlo.broadcast_in_dim %v6169, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6171 = stablehlo.divide %v6170, %v6168 : tensor<64x2048x7x7xf32>
    %v6172 = stablehlo.subtract %v6166, %v6171 : tensor<64x2048x7x7xf32>
    %v6173 = stablehlo.multiply %v6172, %v6172 : tensor<64x2048x7x7xf32>
    %v6174 = stablehlo.reduce(%v6173 init: %v6167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6175 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6176 = stablehlo.divide %v6174, %v6175 : tensor<2048xf32>
    %v6177 = stablehlo.reshape %v1427 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6179 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6180 = stablehlo.reduce(%v6177 init: %v6178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6181 = stablehlo.divide %v6180, %v6179 : tensor<512xf32>
    %v6182 = stablehlo.reshape %v1427 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6184 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6185 = stablehlo.reduce(%v6182 init: %v6183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6186 = stablehlo.broadcast_in_dim %v6185, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6187 = stablehlo.divide %v6186, %v6184 : tensor<64x512x7x7xf32>
    %v6188 = stablehlo.subtract %v6182, %v6187 : tensor<64x512x7x7xf32>
    %v6189 = stablehlo.multiply %v6188, %v6188 : tensor<64x512x7x7xf32>
    %v6190 = stablehlo.reduce(%v6189 init: %v6183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6191 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6192 = stablehlo.divide %v6190, %v6191 : tensor<512xf32>
    %v6193 = stablehlo.reshape %v1457 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6195 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6196 = stablehlo.reduce(%v6193 init: %v6194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6197 = stablehlo.divide %v6196, %v6195 : tensor<512xf32>
    %v6198 = stablehlo.reshape %v1457 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6200 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6201 = stablehlo.reduce(%v6198 init: %v6199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6202 = stablehlo.broadcast_in_dim %v6201, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6203 = stablehlo.divide %v6202, %v6200 : tensor<64x512x7x7xf32>
    %v6204 = stablehlo.subtract %v6198, %v6203 : tensor<64x512x7x7xf32>
    %v6205 = stablehlo.multiply %v6204, %v6204 : tensor<64x512x7x7xf32>
    %v6206 = stablehlo.reduce(%v6205 init: %v6199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6207 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6208 = stablehlo.divide %v6206, %v6207 : tensor<512xf32>
    %v6209 = stablehlo.reshape %v1487 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6211 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6212 = stablehlo.reduce(%v6209 init: %v6210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6213 = stablehlo.divide %v6212, %v6211 : tensor<2048xf32>
    %v6214 = stablehlo.reshape %v1487 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6216 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6217 = stablehlo.reduce(%v6214 init: %v6215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6218 = stablehlo.broadcast_in_dim %v6217, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6219 = stablehlo.divide %v6218, %v6216 : tensor<64x2048x7x7xf32>
    %v6220 = stablehlo.subtract %v6214, %v6219 : tensor<64x2048x7x7xf32>
    %v6221 = stablehlo.multiply %v6220, %v6220 : tensor<64x2048x7x7xf32>
    %v6222 = stablehlo.reduce(%v6221 init: %v6215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6223 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6224 = stablehlo.divide %v6222, %v6223 : tensor<2048xf32>
    %v6225 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6227 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6228 = stablehlo.reduce(%v6225 init: %v6226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6229 = stablehlo.divide %v6228, %v6227 : tensor<512xf32>
    %v6230 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6232 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6233 = stablehlo.reduce(%v6230 init: %v6231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6234 = stablehlo.broadcast_in_dim %v6233, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6235 = stablehlo.divide %v6234, %v6232 : tensor<64x512x7x7xf32>
    %v6236 = stablehlo.subtract %v6230, %v6235 : tensor<64x512x7x7xf32>
    %v6237 = stablehlo.multiply %v6236, %v6236 : tensor<64x512x7x7xf32>
    %v6238 = stablehlo.reduce(%v6237 init: %v6231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6239 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6240 = stablehlo.divide %v6238, %v6239 : tensor<512xf32>
    %v6241 = stablehlo.reshape %v1548 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6243 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6244 = stablehlo.reduce(%v6241 init: %v6242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6245 = stablehlo.divide %v6244, %v6243 : tensor<512xf32>
    %v6246 = stablehlo.reshape %v1548 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v6247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6248 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v6249 = stablehlo.reduce(%v6246 init: %v6247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6250 = stablehlo.broadcast_in_dim %v6249, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v6251 = stablehlo.divide %v6250, %v6248 : tensor<64x512x7x7xf32>
    %v6252 = stablehlo.subtract %v6246, %v6251 : tensor<64x512x7x7xf32>
    %v6253 = stablehlo.multiply %v6252, %v6252 : tensor<64x512x7x7xf32>
    %v6254 = stablehlo.reduce(%v6253 init: %v6247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v6255 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v6256 = stablehlo.divide %v6254, %v6255 : tensor<512xf32>
    %v6257 = stablehlo.reshape %v1578 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6259 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6260 = stablehlo.reduce(%v6257 init: %v6258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6261 = stablehlo.divide %v6260, %v6259 : tensor<2048xf32>
    %v6262 = stablehlo.reshape %v1578 : (tensor<64x100352xf32>) -> tensor<64x2048x7x7xf32>
    %v6263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v6264 = stablehlo.constant dense<3136.0> : tensor<64x2048x7x7xf32>
    %v6265 = stablehlo.reduce(%v6262 init: %v6263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6266 = stablehlo.broadcast_in_dim %v6265, dims = [1] : (tensor<2048xf32>) -> tensor<64x2048x7x7xf32>
    %v6267 = stablehlo.divide %v6266, %v6264 : tensor<64x2048x7x7xf32>
    %v6268 = stablehlo.subtract %v6262, %v6267 : tensor<64x2048x7x7xf32>
    %v6269 = stablehlo.multiply %v6268, %v6268 : tensor<64x2048x7x7xf32>
    %v6270 = stablehlo.reduce(%v6269 init: %v6263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x2048x7x7xf32>, tensor<f32>) -> tensor<2048xf32>
    %v6271 = stablehlo.constant dense<3136.0> : tensor<2048xf32>
    %v6272 = stablehlo.divide %v6270, %v6271 : tensor<2048xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v5403) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v6273 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6274 = stablehlo.multiply %v6273, %sW : tensor<64x3x7x7xf32>
    %v6275 = stablehlo.add %v6274, %armeansW : tensor<64x3x7x7xf32>
    %v6276 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6277 = stablehlo.multiply %v6276, %sWv : tensor<64x3x7x7xf32>
    %v6278 = stablehlo.add %v6277, %v6275 : tensor<64x3x7x7xf32>
    %v6279 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v6280 = stablehlo.multiply %v6279, %v6278 : tensor<64x3x7x7xf32>
    %v6281 = stablehlo.subtract %sW, %v6280 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v5421) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v6282 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6283 = stablehlo.multiply %v6282, %sg : tensor<64xf32>
    %v6284 = stablehlo.add %v6283, %armeansg : tensor<64xf32>
    %v6285 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6286 = stablehlo.multiply %v6285, %sgv : tensor<64xf32>
    %v6287 = stablehlo.add %v6286, %v6284 : tensor<64xf32>
    %v6288 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6289 = stablehlo.multiply %v6288, %v6287 : tensor<64xf32>
    %v6290 = stablehlo.subtract %sg, %v6289 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v5424) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v6291 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6292 = stablehlo.multiply %v6291, %sbt : tensor<64xf32>
    %v6293 = stablehlo.add %v6292, %armeansbt : tensor<64xf32>
    %v6294 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6295 = stablehlo.multiply %v6294, %sbtv : tensor<64xf32>
    %v6296 = stablehlo.add %v6295, %v6293 : tensor<64xf32>
    %v6297 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6298 = stablehlo.multiply %v6297, %v6296 : tensor<64xf32>
    %v6299 = stablehlo.subtract %sbt, %v6298 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v5243) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x1x1xf32>) -> tensor<64x64x1x1xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x1x1xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x1x1xf32>
    %v6300 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6301 = stablehlo.multiply %v6300, %s1b0W1 : tensor<64x64x1x1xf32>
    %v6302 = stablehlo.add %v6301, %armeans1b0W1 : tensor<64x64x1x1xf32>
    %v6303 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6304 = stablehlo.multiply %v6303, %s1b0W1v : tensor<64x64x1x1xf32>
    %v6305 = stablehlo.add %v6304, %v6302 : tensor<64x64x1x1xf32>
    %v6306 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x1x1xf32>
    %v6307 = stablehlo.multiply %v6306, %v6305 : tensor<64x64x1x1xf32>
    %v6308 = stablehlo.subtract %s1b0W1, %v6307 : tensor<64x64x1x1xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v5261) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v6309 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6310 = stablehlo.multiply %v6309, %s1b0g1 : tensor<64xf32>
    %v6311 = stablehlo.add %v6310, %armeans1b0g1 : tensor<64xf32>
    %v6312 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6313 = stablehlo.multiply %v6312, %s1b0g1v : tensor<64xf32>
    %v6314 = stablehlo.add %v6313, %v6311 : tensor<64xf32>
    %v6315 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6316 = stablehlo.multiply %v6315, %v6314 : tensor<64xf32>
    %v6317 = stablehlo.subtract %s1b0g1, %v6316 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v5264) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v6318 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6319 = stablehlo.multiply %v6318, %s1b0bt1 : tensor<64xf32>
    %v6320 = stablehlo.add %v6319, %armeans1b0bt1 : tensor<64xf32>
    %v6321 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6322 = stablehlo.multiply %v6321, %s1b0bt1v : tensor<64xf32>
    %v6323 = stablehlo.add %v6322, %v6320 : tensor<64xf32>
    %v6324 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6325 = stablehlo.multiply %v6324, %v6323 : tensor<64xf32>
    %v6326 = stablehlo.subtract %s1b0bt1, %v6325 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v5273) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v6327 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6328 = stablehlo.multiply %v6327, %s1b0W2 : tensor<64x64x3x3xf32>
    %v6329 = stablehlo.add %v6328, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v6330 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6331 = stablehlo.multiply %v6330, %s1b0W2v : tensor<64x64x3x3xf32>
    %v6332 = stablehlo.add %v6331, %v6329 : tensor<64x64x3x3xf32>
    %v6333 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6334 = stablehlo.multiply %v6333, %v6332 : tensor<64x64x3x3xf32>
    %v6335 = stablehlo.subtract %s1b0W2, %v6334 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v5291) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v6336 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6337 = stablehlo.multiply %v6336, %s1b0g2 : tensor<64xf32>
    %v6338 = stablehlo.add %v6337, %armeans1b0g2 : tensor<64xf32>
    %v6339 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6340 = stablehlo.multiply %v6339, %s1b0g2v : tensor<64xf32>
    %v6341 = stablehlo.add %v6340, %v6338 : tensor<64xf32>
    %v6342 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6343 = stablehlo.multiply %v6342, %v6341 : tensor<64xf32>
    %v6344 = stablehlo.subtract %s1b0g2, %v6343 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v5294) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v6345 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6346 = stablehlo.multiply %v6345, %s1b0bt2 : tensor<64xf32>
    %v6347 = stablehlo.add %v6346, %armeans1b0bt2 : tensor<64xf32>
    %v6348 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6349 = stablehlo.multiply %v6348, %s1b0bt2v : tensor<64xf32>
    %v6350 = stablehlo.add %v6349, %v6347 : tensor<64xf32>
    %v6351 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6352 = stablehlo.multiply %v6351, %v6350 : tensor<64xf32>
    %v6353 = stablehlo.subtract %s1b0bt2, %v6352 : tensor<64xf32>
    %arsums1b0W3 = "stablehlo.all_reduce"(%v5303) ({
    ^bb0(%aras1b0W3: tensor<f32>, %arbs1b0W3: tensor<f32>):
      %aradds1b0W3 = stablehlo.add %aras1b0W3, %arbs1b0W3 : tensor<f32>
      stablehlo.return %aradds1b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0W3 = stablehlo.divide %arsums1b0W3, %arns1b0W3 : tensor<256x64x1x1xf32>
    %v6354 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6355 = stablehlo.multiply %v6354, %s1b0W3 : tensor<256x64x1x1xf32>
    %v6356 = stablehlo.add %v6355, %armeans1b0W3 : tensor<256x64x1x1xf32>
    %v6357 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6358 = stablehlo.multiply %v6357, %s1b0W3v : tensor<256x64x1x1xf32>
    %v6359 = stablehlo.add %v6358, %v6356 : tensor<256x64x1x1xf32>
    %v6360 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6361 = stablehlo.multiply %v6360, %v6359 : tensor<256x64x1x1xf32>
    %v6362 = stablehlo.subtract %s1b0W3, %v6361 : tensor<256x64x1x1xf32>
    %arsums1b0g3 = "stablehlo.all_reduce"(%v5321) ({
    ^bb0(%aras1b0g3: tensor<f32>, %arbs1b0g3: tensor<f32>):
      %aradds1b0g3 = stablehlo.add %aras1b0g3, %arbs1b0g3 : tensor<f32>
      stablehlo.return %aradds1b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g3 = stablehlo.divide %arsums1b0g3, %arns1b0g3 : tensor<256xf32>
    %v6363 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6364 = stablehlo.multiply %v6363, %s1b0g3 : tensor<256xf32>
    %v6365 = stablehlo.add %v6364, %armeans1b0g3 : tensor<256xf32>
    %v6366 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6367 = stablehlo.multiply %v6366, %s1b0g3v : tensor<256xf32>
    %v6368 = stablehlo.add %v6367, %v6365 : tensor<256xf32>
    %v6369 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6370 = stablehlo.multiply %v6369, %v6368 : tensor<256xf32>
    %v6371 = stablehlo.subtract %s1b0g3, %v6370 : tensor<256xf32>
    %arsums1b0bt3 = "stablehlo.all_reduce"(%v5324) ({
    ^bb0(%aras1b0bt3: tensor<f32>, %arbs1b0bt3: tensor<f32>):
      %aradds1b0bt3 = stablehlo.add %aras1b0bt3, %arbs1b0bt3 : tensor<f32>
      stablehlo.return %aradds1b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0bt3 = stablehlo.divide %arsums1b0bt3, %arns1b0bt3 : tensor<256xf32>
    %v6372 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6373 = stablehlo.multiply %v6372, %s1b0bt3 : tensor<256xf32>
    %v6374 = stablehlo.add %v6373, %armeans1b0bt3 : tensor<256xf32>
    %v6375 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6376 = stablehlo.multiply %v6375, %s1b0bt3v : tensor<256xf32>
    %v6377 = stablehlo.add %v6376, %v6374 : tensor<256xf32>
    %v6378 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6379 = stablehlo.multiply %v6378, %v6377 : tensor<256xf32>
    %v6380 = stablehlo.subtract %s1b0bt3, %v6379 : tensor<256xf32>
    %arsums1b0Wp = "stablehlo.all_reduce"(%v5333) ({
    ^bb0(%aras1b0Wp: tensor<f32>, %arbs1b0Wp: tensor<f32>):
      %aradds1b0Wp = stablehlo.add %aras1b0Wp, %arbs1b0Wp : tensor<f32>
      stablehlo.return %aradds1b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b0Wp = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b0Wp = stablehlo.divide %arsums1b0Wp, %arns1b0Wp : tensor<256x64x1x1xf32>
    %v6381 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6382 = stablehlo.multiply %v6381, %s1b0Wp : tensor<256x64x1x1xf32>
    %v6383 = stablehlo.add %v6382, %armeans1b0Wp : tensor<256x64x1x1xf32>
    %v6384 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6385 = stablehlo.multiply %v6384, %s1b0Wpv : tensor<256x64x1x1xf32>
    %v6386 = stablehlo.add %v6385, %v6383 : tensor<256x64x1x1xf32>
    %v6387 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6388 = stablehlo.multiply %v6387, %v6386 : tensor<256x64x1x1xf32>
    %v6389 = stablehlo.subtract %s1b0Wp, %v6388 : tensor<256x64x1x1xf32>
    %arsums1b0gp = "stablehlo.all_reduce"(%v5351) ({
    ^bb0(%aras1b0gp: tensor<f32>, %arbs1b0gp: tensor<f32>):
      %aradds1b0gp = stablehlo.add %aras1b0gp, %arbs1b0gp : tensor<f32>
      stablehlo.return %aradds1b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0gp = stablehlo.divide %arsums1b0gp, %arns1b0gp : tensor<256xf32>
    %v6390 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6391 = stablehlo.multiply %v6390, %s1b0gp : tensor<256xf32>
    %v6392 = stablehlo.add %v6391, %armeans1b0gp : tensor<256xf32>
    %v6393 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6394 = stablehlo.multiply %v6393, %s1b0gpv : tensor<256xf32>
    %v6395 = stablehlo.add %v6394, %v6392 : tensor<256xf32>
    %v6396 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6397 = stablehlo.multiply %v6396, %v6395 : tensor<256xf32>
    %v6398 = stablehlo.subtract %s1b0gp, %v6397 : tensor<256xf32>
    %arsums1b0btp = "stablehlo.all_reduce"(%v5354) ({
    ^bb0(%aras1b0btp: tensor<f32>, %arbs1b0btp: tensor<f32>):
      %aradds1b0btp = stablehlo.add %aras1b0btp, %arbs1b0btp : tensor<f32>
      stablehlo.return %aradds1b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0btp = stablehlo.divide %arsums1b0btp, %arns1b0btp : tensor<256xf32>
    %v6399 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6400 = stablehlo.multiply %v6399, %s1b0btp : tensor<256xf32>
    %v6401 = stablehlo.add %v6400, %armeans1b0btp : tensor<256xf32>
    %v6402 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6403 = stablehlo.multiply %v6402, %s1b0btpv : tensor<256xf32>
    %v6404 = stablehlo.add %v6403, %v6401 : tensor<256xf32>
    %v6405 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6406 = stablehlo.multiply %v6405, %v6404 : tensor<256xf32>
    %v6407 = stablehlo.subtract %s1b0btp, %v6406 : tensor<256xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v4991) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x256x1x1xf32>
    %v6408 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6409 = stablehlo.multiply %v6408, %s1b1W1 : tensor<64x256x1x1xf32>
    %v6410 = stablehlo.add %v6409, %armeans1b1W1 : tensor<64x256x1x1xf32>
    %v6411 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6412 = stablehlo.multiply %v6411, %s1b1W1v : tensor<64x256x1x1xf32>
    %v6413 = stablehlo.add %v6412, %v6410 : tensor<64x256x1x1xf32>
    %v6414 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6415 = stablehlo.multiply %v6414, %v6413 : tensor<64x256x1x1xf32>
    %v6416 = stablehlo.subtract %s1b1W1, %v6415 : tensor<64x256x1x1xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v5009) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v6417 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6418 = stablehlo.multiply %v6417, %s1b1g1 : tensor<64xf32>
    %v6419 = stablehlo.add %v6418, %armeans1b1g1 : tensor<64xf32>
    %v6420 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6421 = stablehlo.multiply %v6420, %s1b1g1v : tensor<64xf32>
    %v6422 = stablehlo.add %v6421, %v6419 : tensor<64xf32>
    %v6423 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6424 = stablehlo.multiply %v6423, %v6422 : tensor<64xf32>
    %v6425 = stablehlo.subtract %s1b1g1, %v6424 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v5012) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v6426 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6427 = stablehlo.multiply %v6426, %s1b1bt1 : tensor<64xf32>
    %v6428 = stablehlo.add %v6427, %armeans1b1bt1 : tensor<64xf32>
    %v6429 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6430 = stablehlo.multiply %v6429, %s1b1bt1v : tensor<64xf32>
    %v6431 = stablehlo.add %v6430, %v6428 : tensor<64xf32>
    %v6432 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6433 = stablehlo.multiply %v6432, %v6431 : tensor<64xf32>
    %v6434 = stablehlo.subtract %s1b1bt1, %v6433 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v5021) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v6435 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6436 = stablehlo.multiply %v6435, %s1b1W2 : tensor<64x64x3x3xf32>
    %v6437 = stablehlo.add %v6436, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v6438 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6439 = stablehlo.multiply %v6438, %s1b1W2v : tensor<64x64x3x3xf32>
    %v6440 = stablehlo.add %v6439, %v6437 : tensor<64x64x3x3xf32>
    %v6441 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6442 = stablehlo.multiply %v6441, %v6440 : tensor<64x64x3x3xf32>
    %v6443 = stablehlo.subtract %s1b1W2, %v6442 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v5039) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v6444 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6445 = stablehlo.multiply %v6444, %s1b1g2 : tensor<64xf32>
    %v6446 = stablehlo.add %v6445, %armeans1b1g2 : tensor<64xf32>
    %v6447 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6448 = stablehlo.multiply %v6447, %s1b1g2v : tensor<64xf32>
    %v6449 = stablehlo.add %v6448, %v6446 : tensor<64xf32>
    %v6450 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6451 = stablehlo.multiply %v6450, %v6449 : tensor<64xf32>
    %v6452 = stablehlo.subtract %s1b1g2, %v6451 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v5042) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v6453 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6454 = stablehlo.multiply %v6453, %s1b1bt2 : tensor<64xf32>
    %v6455 = stablehlo.add %v6454, %armeans1b1bt2 : tensor<64xf32>
    %v6456 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6457 = stablehlo.multiply %v6456, %s1b1bt2v : tensor<64xf32>
    %v6458 = stablehlo.add %v6457, %v6455 : tensor<64xf32>
    %v6459 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6460 = stablehlo.multiply %v6459, %v6458 : tensor<64xf32>
    %v6461 = stablehlo.subtract %s1b1bt2, %v6460 : tensor<64xf32>
    %arsums1b1W3 = "stablehlo.all_reduce"(%v5051) ({
    ^bb0(%aras1b1W3: tensor<f32>, %arbs1b1W3: tensor<f32>):
      %aradds1b1W3 = stablehlo.add %aras1b1W3, %arbs1b1W3 : tensor<f32>
      stablehlo.return %aradds1b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b1W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b1W3 = stablehlo.divide %arsums1b1W3, %arns1b1W3 : tensor<256x64x1x1xf32>
    %v6462 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6463 = stablehlo.multiply %v6462, %s1b1W3 : tensor<256x64x1x1xf32>
    %v6464 = stablehlo.add %v6463, %armeans1b1W3 : tensor<256x64x1x1xf32>
    %v6465 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6466 = stablehlo.multiply %v6465, %s1b1W3v : tensor<256x64x1x1xf32>
    %v6467 = stablehlo.add %v6466, %v6464 : tensor<256x64x1x1xf32>
    %v6468 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6469 = stablehlo.multiply %v6468, %v6467 : tensor<256x64x1x1xf32>
    %v6470 = stablehlo.subtract %s1b1W3, %v6469 : tensor<256x64x1x1xf32>
    %arsums1b1g3 = "stablehlo.all_reduce"(%v5069) ({
    ^bb0(%aras1b1g3: tensor<f32>, %arbs1b1g3: tensor<f32>):
      %aradds1b1g3 = stablehlo.add %aras1b1g3, %arbs1b1g3 : tensor<f32>
      stablehlo.return %aradds1b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g3 = stablehlo.divide %arsums1b1g3, %arns1b1g3 : tensor<256xf32>
    %v6471 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6472 = stablehlo.multiply %v6471, %s1b1g3 : tensor<256xf32>
    %v6473 = stablehlo.add %v6472, %armeans1b1g3 : tensor<256xf32>
    %v6474 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6475 = stablehlo.multiply %v6474, %s1b1g3v : tensor<256xf32>
    %v6476 = stablehlo.add %v6475, %v6473 : tensor<256xf32>
    %v6477 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6478 = stablehlo.multiply %v6477, %v6476 : tensor<256xf32>
    %v6479 = stablehlo.subtract %s1b1g3, %v6478 : tensor<256xf32>
    %arsums1b1bt3 = "stablehlo.all_reduce"(%v5072) ({
    ^bb0(%aras1b1bt3: tensor<f32>, %arbs1b1bt3: tensor<f32>):
      %aradds1b1bt3 = stablehlo.add %aras1b1bt3, %arbs1b1bt3 : tensor<f32>
      stablehlo.return %aradds1b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1bt3 = stablehlo.divide %arsums1b1bt3, %arns1b1bt3 : tensor<256xf32>
    %v6480 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6481 = stablehlo.multiply %v6480, %s1b1bt3 : tensor<256xf32>
    %v6482 = stablehlo.add %v6481, %armeans1b1bt3 : tensor<256xf32>
    %v6483 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6484 = stablehlo.multiply %v6483, %s1b1bt3v : tensor<256xf32>
    %v6485 = stablehlo.add %v6484, %v6482 : tensor<256xf32>
    %v6486 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6487 = stablehlo.multiply %v6486, %v6485 : tensor<256xf32>
    %v6488 = stablehlo.subtract %s1b1bt3, %v6487 : tensor<256xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v4777) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x256x1x1xf32>) -> tensor<64x256x1x1xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x256x1x1xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x256x1x1xf32>
    %v6489 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6490 = stablehlo.multiply %v6489, %s1b2W1 : tensor<64x256x1x1xf32>
    %v6491 = stablehlo.add %v6490, %armeans1b2W1 : tensor<64x256x1x1xf32>
    %v6492 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6493 = stablehlo.multiply %v6492, %s1b2W1v : tensor<64x256x1x1xf32>
    %v6494 = stablehlo.add %v6493, %v6491 : tensor<64x256x1x1xf32>
    %v6495 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x256x1x1xf32>
    %v6496 = stablehlo.multiply %v6495, %v6494 : tensor<64x256x1x1xf32>
    %v6497 = stablehlo.subtract %s1b2W1, %v6496 : tensor<64x256x1x1xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v4795) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v6498 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6499 = stablehlo.multiply %v6498, %s1b2g1 : tensor<64xf32>
    %v6500 = stablehlo.add %v6499, %armeans1b2g1 : tensor<64xf32>
    %v6501 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6502 = stablehlo.multiply %v6501, %s1b2g1v : tensor<64xf32>
    %v6503 = stablehlo.add %v6502, %v6500 : tensor<64xf32>
    %v6504 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6505 = stablehlo.multiply %v6504, %v6503 : tensor<64xf32>
    %v6506 = stablehlo.subtract %s1b2g1, %v6505 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v4798) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v6507 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6508 = stablehlo.multiply %v6507, %s1b2bt1 : tensor<64xf32>
    %v6509 = stablehlo.add %v6508, %armeans1b2bt1 : tensor<64xf32>
    %v6510 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6511 = stablehlo.multiply %v6510, %s1b2bt1v : tensor<64xf32>
    %v6512 = stablehlo.add %v6511, %v6509 : tensor<64xf32>
    %v6513 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6514 = stablehlo.multiply %v6513, %v6512 : tensor<64xf32>
    %v6515 = stablehlo.subtract %s1b2bt1, %v6514 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v4807) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v6516 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6517 = stablehlo.multiply %v6516, %s1b2W2 : tensor<64x64x3x3xf32>
    %v6518 = stablehlo.add %v6517, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v6519 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6520 = stablehlo.multiply %v6519, %s1b2W2v : tensor<64x64x3x3xf32>
    %v6521 = stablehlo.add %v6520, %v6518 : tensor<64x64x3x3xf32>
    %v6522 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v6523 = stablehlo.multiply %v6522, %v6521 : tensor<64x64x3x3xf32>
    %v6524 = stablehlo.subtract %s1b2W2, %v6523 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v4825) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v6525 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6526 = stablehlo.multiply %v6525, %s1b2g2 : tensor<64xf32>
    %v6527 = stablehlo.add %v6526, %armeans1b2g2 : tensor<64xf32>
    %v6528 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6529 = stablehlo.multiply %v6528, %s1b2g2v : tensor<64xf32>
    %v6530 = stablehlo.add %v6529, %v6527 : tensor<64xf32>
    %v6531 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6532 = stablehlo.multiply %v6531, %v6530 : tensor<64xf32>
    %v6533 = stablehlo.subtract %s1b2g2, %v6532 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v4828) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v6534 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6535 = stablehlo.multiply %v6534, %s1b2bt2 : tensor<64xf32>
    %v6536 = stablehlo.add %v6535, %armeans1b2bt2 : tensor<64xf32>
    %v6537 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6538 = stablehlo.multiply %v6537, %s1b2bt2v : tensor<64xf32>
    %v6539 = stablehlo.add %v6538, %v6536 : tensor<64xf32>
    %v6540 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v6541 = stablehlo.multiply %v6540, %v6539 : tensor<64xf32>
    %v6542 = stablehlo.subtract %s1b2bt2, %v6541 : tensor<64xf32>
    %arsums1b2W3 = "stablehlo.all_reduce"(%v4837) ({
    ^bb0(%aras1b2W3: tensor<f32>, %arbs1b2W3: tensor<f32>):
      %aradds1b2W3 = stablehlo.add %aras1b2W3, %arbs1b2W3 : tensor<f32>
      stablehlo.return %aradds1b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x64x1x1xf32>) -> tensor<256x64x1x1xf32>
    %arns1b2W3 = stablehlo.constant dense<4.0> : tensor<256x64x1x1xf32>
    %armeans1b2W3 = stablehlo.divide %arsums1b2W3, %arns1b2W3 : tensor<256x64x1x1xf32>
    %v6543 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6544 = stablehlo.multiply %v6543, %s1b2W3 : tensor<256x64x1x1xf32>
    %v6545 = stablehlo.add %v6544, %armeans1b2W3 : tensor<256x64x1x1xf32>
    %v6546 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6547 = stablehlo.multiply %v6546, %s1b2W3v : tensor<256x64x1x1xf32>
    %v6548 = stablehlo.add %v6547, %v6545 : tensor<256x64x1x1xf32>
    %v6549 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x64x1x1xf32>
    %v6550 = stablehlo.multiply %v6549, %v6548 : tensor<256x64x1x1xf32>
    %v6551 = stablehlo.subtract %s1b2W3, %v6550 : tensor<256x64x1x1xf32>
    %arsums1b2g3 = "stablehlo.all_reduce"(%v4855) ({
    ^bb0(%aras1b2g3: tensor<f32>, %arbs1b2g3: tensor<f32>):
      %aradds1b2g3 = stablehlo.add %aras1b2g3, %arbs1b2g3 : tensor<f32>
      stablehlo.return %aradds1b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g3 = stablehlo.divide %arsums1b2g3, %arns1b2g3 : tensor<256xf32>
    %v6552 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6553 = stablehlo.multiply %v6552, %s1b2g3 : tensor<256xf32>
    %v6554 = stablehlo.add %v6553, %armeans1b2g3 : tensor<256xf32>
    %v6555 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6556 = stablehlo.multiply %v6555, %s1b2g3v : tensor<256xf32>
    %v6557 = stablehlo.add %v6556, %v6554 : tensor<256xf32>
    %v6558 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6559 = stablehlo.multiply %v6558, %v6557 : tensor<256xf32>
    %v6560 = stablehlo.subtract %s1b2g3, %v6559 : tensor<256xf32>
    %arsums1b2bt3 = "stablehlo.all_reduce"(%v4858) ({
    ^bb0(%aras1b2bt3: tensor<f32>, %arbs1b2bt3: tensor<f32>):
      %aradds1b2bt3 = stablehlo.add %aras1b2bt3, %arbs1b2bt3 : tensor<f32>
      stablehlo.return %aradds1b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2bt3 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2bt3 = stablehlo.divide %arsums1b2bt3, %arns1b2bt3 : tensor<256xf32>
    %v6561 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6562 = stablehlo.multiply %v6561, %s1b2bt3 : tensor<256xf32>
    %v6563 = stablehlo.add %v6562, %armeans1b2bt3 : tensor<256xf32>
    %v6564 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6565 = stablehlo.multiply %v6564, %s1b2bt3v : tensor<256xf32>
    %v6566 = stablehlo.add %v6565, %v6563 : tensor<256xf32>
    %v6567 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6568 = stablehlo.multiply %v6567, %v6566 : tensor<256xf32>
    %v6569 = stablehlo.subtract %s1b2bt3, %v6568 : tensor<256xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v4529) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x256x1x1xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x256x1x1xf32>
    %v6570 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6571 = stablehlo.multiply %v6570, %s2b0W1 : tensor<128x256x1x1xf32>
    %v6572 = stablehlo.add %v6571, %armeans2b0W1 : tensor<128x256x1x1xf32>
    %v6573 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6574 = stablehlo.multiply %v6573, %s2b0W1v : tensor<128x256x1x1xf32>
    %v6575 = stablehlo.add %v6574, %v6572 : tensor<128x256x1x1xf32>
    %v6576 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x256x1x1xf32>
    %v6577 = stablehlo.multiply %v6576, %v6575 : tensor<128x256x1x1xf32>
    %v6578 = stablehlo.subtract %s2b0W1, %v6577 : tensor<128x256x1x1xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v4547) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v6579 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6580 = stablehlo.multiply %v6579, %s2b0g1 : tensor<128xf32>
    %v6581 = stablehlo.add %v6580, %armeans2b0g1 : tensor<128xf32>
    %v6582 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6583 = stablehlo.multiply %v6582, %s2b0g1v : tensor<128xf32>
    %v6584 = stablehlo.add %v6583, %v6581 : tensor<128xf32>
    %v6585 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6586 = stablehlo.multiply %v6585, %v6584 : tensor<128xf32>
    %v6587 = stablehlo.subtract %s2b0g1, %v6586 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v4550) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v6588 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6589 = stablehlo.multiply %v6588, %s2b0bt1 : tensor<128xf32>
    %v6590 = stablehlo.add %v6589, %armeans2b0bt1 : tensor<128xf32>
    %v6591 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6592 = stablehlo.multiply %v6591, %s2b0bt1v : tensor<128xf32>
    %v6593 = stablehlo.add %v6592, %v6590 : tensor<128xf32>
    %v6594 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6595 = stablehlo.multiply %v6594, %v6593 : tensor<128xf32>
    %v6596 = stablehlo.subtract %s2b0bt1, %v6595 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v4561) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v6597 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6598 = stablehlo.multiply %v6597, %s2b0W2 : tensor<128x128x3x3xf32>
    %v6599 = stablehlo.add %v6598, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v6600 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6601 = stablehlo.multiply %v6600, %s2b0W2v : tensor<128x128x3x3xf32>
    %v6602 = stablehlo.add %v6601, %v6599 : tensor<128x128x3x3xf32>
    %v6603 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6604 = stablehlo.multiply %v6603, %v6602 : tensor<128x128x3x3xf32>
    %v6605 = stablehlo.subtract %s2b0W2, %v6604 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v4579) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v6606 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6607 = stablehlo.multiply %v6606, %s2b0g2 : tensor<128xf32>
    %v6608 = stablehlo.add %v6607, %armeans2b0g2 : tensor<128xf32>
    %v6609 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6610 = stablehlo.multiply %v6609, %s2b0g2v : tensor<128xf32>
    %v6611 = stablehlo.add %v6610, %v6608 : tensor<128xf32>
    %v6612 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6613 = stablehlo.multiply %v6612, %v6611 : tensor<128xf32>
    %v6614 = stablehlo.subtract %s2b0g2, %v6613 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v4582) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v6615 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6616 = stablehlo.multiply %v6615, %s2b0bt2 : tensor<128xf32>
    %v6617 = stablehlo.add %v6616, %armeans2b0bt2 : tensor<128xf32>
    %v6618 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6619 = stablehlo.multiply %v6618, %s2b0bt2v : tensor<128xf32>
    %v6620 = stablehlo.add %v6619, %v6617 : tensor<128xf32>
    %v6621 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6622 = stablehlo.multiply %v6621, %v6620 : tensor<128xf32>
    %v6623 = stablehlo.subtract %s2b0bt2, %v6622 : tensor<128xf32>
    %arsums2b0W3 = "stablehlo.all_reduce"(%v4591) ({
    ^bb0(%aras2b0W3: tensor<f32>, %arbs2b0W3: tensor<f32>):
      %aradds2b0W3 = stablehlo.add %aras2b0W3, %arbs2b0W3 : tensor<f32>
      stablehlo.return %aradds2b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b0W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b0W3 = stablehlo.divide %arsums2b0W3, %arns2b0W3 : tensor<512x128x1x1xf32>
    %v6624 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6625 = stablehlo.multiply %v6624, %s2b0W3 : tensor<512x128x1x1xf32>
    %v6626 = stablehlo.add %v6625, %armeans2b0W3 : tensor<512x128x1x1xf32>
    %v6627 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6628 = stablehlo.multiply %v6627, %s2b0W3v : tensor<512x128x1x1xf32>
    %v6629 = stablehlo.add %v6628, %v6626 : tensor<512x128x1x1xf32>
    %v6630 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6631 = stablehlo.multiply %v6630, %v6629 : tensor<512x128x1x1xf32>
    %v6632 = stablehlo.subtract %s2b0W3, %v6631 : tensor<512x128x1x1xf32>
    %arsums2b0g3 = "stablehlo.all_reduce"(%v4609) ({
    ^bb0(%aras2b0g3: tensor<f32>, %arbs2b0g3: tensor<f32>):
      %aradds2b0g3 = stablehlo.add %aras2b0g3, %arbs2b0g3 : tensor<f32>
      stablehlo.return %aradds2b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g3 = stablehlo.divide %arsums2b0g3, %arns2b0g3 : tensor<512xf32>
    %v6633 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6634 = stablehlo.multiply %v6633, %s2b0g3 : tensor<512xf32>
    %v6635 = stablehlo.add %v6634, %armeans2b0g3 : tensor<512xf32>
    %v6636 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6637 = stablehlo.multiply %v6636, %s2b0g3v : tensor<512xf32>
    %v6638 = stablehlo.add %v6637, %v6635 : tensor<512xf32>
    %v6639 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6640 = stablehlo.multiply %v6639, %v6638 : tensor<512xf32>
    %v6641 = stablehlo.subtract %s2b0g3, %v6640 : tensor<512xf32>
    %arsums2b0bt3 = "stablehlo.all_reduce"(%v4612) ({
    ^bb0(%aras2b0bt3: tensor<f32>, %arbs2b0bt3: tensor<f32>):
      %aradds2b0bt3 = stablehlo.add %aras2b0bt3, %arbs2b0bt3 : tensor<f32>
      stablehlo.return %aradds2b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0bt3 = stablehlo.divide %arsums2b0bt3, %arns2b0bt3 : tensor<512xf32>
    %v6642 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6643 = stablehlo.multiply %v6642, %s2b0bt3 : tensor<512xf32>
    %v6644 = stablehlo.add %v6643, %armeans2b0bt3 : tensor<512xf32>
    %v6645 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6646 = stablehlo.multiply %v6645, %s2b0bt3v : tensor<512xf32>
    %v6647 = stablehlo.add %v6646, %v6644 : tensor<512xf32>
    %v6648 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6649 = stablehlo.multiply %v6648, %v6647 : tensor<512xf32>
    %v6650 = stablehlo.subtract %s2b0bt3, %v6649 : tensor<512xf32>
    %arsums2b0Wp = "stablehlo.all_reduce"(%v4623) ({
    ^bb0(%aras2b0Wp: tensor<f32>, %arbs2b0Wp: tensor<f32>):
      %aradds2b0Wp = stablehlo.add %aras2b0Wp, %arbs2b0Wp : tensor<f32>
      stablehlo.return %aradds2b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arns2b0Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeans2b0Wp = stablehlo.divide %arsums2b0Wp, %arns2b0Wp : tensor<512x256x1x1xf32>
    %v6651 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6652 = stablehlo.multiply %v6651, %s2b0Wp : tensor<512x256x1x1xf32>
    %v6653 = stablehlo.add %v6652, %armeans2b0Wp : tensor<512x256x1x1xf32>
    %v6654 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6655 = stablehlo.multiply %v6654, %s2b0Wpv : tensor<512x256x1x1xf32>
    %v6656 = stablehlo.add %v6655, %v6653 : tensor<512x256x1x1xf32>
    %v6657 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v6658 = stablehlo.multiply %v6657, %v6656 : tensor<512x256x1x1xf32>
    %v6659 = stablehlo.subtract %s2b0Wp, %v6658 : tensor<512x256x1x1xf32>
    %arsums2b0gp = "stablehlo.all_reduce"(%v4641) ({
    ^bb0(%aras2b0gp: tensor<f32>, %arbs2b0gp: tensor<f32>):
      %aradds2b0gp = stablehlo.add %aras2b0gp, %arbs2b0gp : tensor<f32>
      stablehlo.return %aradds2b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0gp = stablehlo.divide %arsums2b0gp, %arns2b0gp : tensor<512xf32>
    %v6660 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6661 = stablehlo.multiply %v6660, %s2b0gp : tensor<512xf32>
    %v6662 = stablehlo.add %v6661, %armeans2b0gp : tensor<512xf32>
    %v6663 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6664 = stablehlo.multiply %v6663, %s2b0gpv : tensor<512xf32>
    %v6665 = stablehlo.add %v6664, %v6662 : tensor<512xf32>
    %v6666 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6667 = stablehlo.multiply %v6666, %v6665 : tensor<512xf32>
    %v6668 = stablehlo.subtract %s2b0gp, %v6667 : tensor<512xf32>
    %arsums2b0btp = "stablehlo.all_reduce"(%v4644) ({
    ^bb0(%aras2b0btp: tensor<f32>, %arbs2b0btp: tensor<f32>):
      %aradds2b0btp = stablehlo.add %aras2b0btp, %arbs2b0btp : tensor<f32>
      stablehlo.return %aradds2b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0btp = stablehlo.divide %arsums2b0btp, %arns2b0btp : tensor<512xf32>
    %v6669 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6670 = stablehlo.multiply %v6669, %s2b0btp : tensor<512xf32>
    %v6671 = stablehlo.add %v6670, %armeans2b0btp : tensor<512xf32>
    %v6672 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6673 = stablehlo.multiply %v6672, %s2b0btpv : tensor<512xf32>
    %v6674 = stablehlo.add %v6673, %v6671 : tensor<512xf32>
    %v6675 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6676 = stablehlo.multiply %v6675, %v6674 : tensor<512xf32>
    %v6677 = stablehlo.subtract %s2b0btp, %v6676 : tensor<512xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v4273) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x512x1x1xf32>
    %v6678 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6679 = stablehlo.multiply %v6678, %s2b1W1 : tensor<128x512x1x1xf32>
    %v6680 = stablehlo.add %v6679, %armeans2b1W1 : tensor<128x512x1x1xf32>
    %v6681 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6682 = stablehlo.multiply %v6681, %s2b1W1v : tensor<128x512x1x1xf32>
    %v6683 = stablehlo.add %v6682, %v6680 : tensor<128x512x1x1xf32>
    %v6684 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6685 = stablehlo.multiply %v6684, %v6683 : tensor<128x512x1x1xf32>
    %v6686 = stablehlo.subtract %s2b1W1, %v6685 : tensor<128x512x1x1xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v4291) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v6687 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6688 = stablehlo.multiply %v6687, %s2b1g1 : tensor<128xf32>
    %v6689 = stablehlo.add %v6688, %armeans2b1g1 : tensor<128xf32>
    %v6690 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6691 = stablehlo.multiply %v6690, %s2b1g1v : tensor<128xf32>
    %v6692 = stablehlo.add %v6691, %v6689 : tensor<128xf32>
    %v6693 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6694 = stablehlo.multiply %v6693, %v6692 : tensor<128xf32>
    %v6695 = stablehlo.subtract %s2b1g1, %v6694 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v4294) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v6696 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6697 = stablehlo.multiply %v6696, %s2b1bt1 : tensor<128xf32>
    %v6698 = stablehlo.add %v6697, %armeans2b1bt1 : tensor<128xf32>
    %v6699 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6700 = stablehlo.multiply %v6699, %s2b1bt1v : tensor<128xf32>
    %v6701 = stablehlo.add %v6700, %v6698 : tensor<128xf32>
    %v6702 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6703 = stablehlo.multiply %v6702, %v6701 : tensor<128xf32>
    %v6704 = stablehlo.subtract %s2b1bt1, %v6703 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v4303) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v6705 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6706 = stablehlo.multiply %v6705, %s2b1W2 : tensor<128x128x3x3xf32>
    %v6707 = stablehlo.add %v6706, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v6708 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6709 = stablehlo.multiply %v6708, %s2b1W2v : tensor<128x128x3x3xf32>
    %v6710 = stablehlo.add %v6709, %v6707 : tensor<128x128x3x3xf32>
    %v6711 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6712 = stablehlo.multiply %v6711, %v6710 : tensor<128x128x3x3xf32>
    %v6713 = stablehlo.subtract %s2b1W2, %v6712 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v4321) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v6714 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6715 = stablehlo.multiply %v6714, %s2b1g2 : tensor<128xf32>
    %v6716 = stablehlo.add %v6715, %armeans2b1g2 : tensor<128xf32>
    %v6717 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6718 = stablehlo.multiply %v6717, %s2b1g2v : tensor<128xf32>
    %v6719 = stablehlo.add %v6718, %v6716 : tensor<128xf32>
    %v6720 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6721 = stablehlo.multiply %v6720, %v6719 : tensor<128xf32>
    %v6722 = stablehlo.subtract %s2b1g2, %v6721 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v4324) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v6723 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6724 = stablehlo.multiply %v6723, %s2b1bt2 : tensor<128xf32>
    %v6725 = stablehlo.add %v6724, %armeans2b1bt2 : tensor<128xf32>
    %v6726 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6727 = stablehlo.multiply %v6726, %s2b1bt2v : tensor<128xf32>
    %v6728 = stablehlo.add %v6727, %v6725 : tensor<128xf32>
    %v6729 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6730 = stablehlo.multiply %v6729, %v6728 : tensor<128xf32>
    %v6731 = stablehlo.subtract %s2b1bt2, %v6730 : tensor<128xf32>
    %arsums2b1W3 = "stablehlo.all_reduce"(%v4333) ({
    ^bb0(%aras2b1W3: tensor<f32>, %arbs2b1W3: tensor<f32>):
      %aradds2b1W3 = stablehlo.add %aras2b1W3, %arbs2b1W3 : tensor<f32>
      stablehlo.return %aradds2b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b1W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b1W3 = stablehlo.divide %arsums2b1W3, %arns2b1W3 : tensor<512x128x1x1xf32>
    %v6732 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6733 = stablehlo.multiply %v6732, %s2b1W3 : tensor<512x128x1x1xf32>
    %v6734 = stablehlo.add %v6733, %armeans2b1W3 : tensor<512x128x1x1xf32>
    %v6735 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6736 = stablehlo.multiply %v6735, %s2b1W3v : tensor<512x128x1x1xf32>
    %v6737 = stablehlo.add %v6736, %v6734 : tensor<512x128x1x1xf32>
    %v6738 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6739 = stablehlo.multiply %v6738, %v6737 : tensor<512x128x1x1xf32>
    %v6740 = stablehlo.subtract %s2b1W3, %v6739 : tensor<512x128x1x1xf32>
    %arsums2b1g3 = "stablehlo.all_reduce"(%v4351) ({
    ^bb0(%aras2b1g3: tensor<f32>, %arbs2b1g3: tensor<f32>):
      %aradds2b1g3 = stablehlo.add %aras2b1g3, %arbs2b1g3 : tensor<f32>
      stablehlo.return %aradds2b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g3 = stablehlo.divide %arsums2b1g3, %arns2b1g3 : tensor<512xf32>
    %v6741 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6742 = stablehlo.multiply %v6741, %s2b1g3 : tensor<512xf32>
    %v6743 = stablehlo.add %v6742, %armeans2b1g3 : tensor<512xf32>
    %v6744 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6745 = stablehlo.multiply %v6744, %s2b1g3v : tensor<512xf32>
    %v6746 = stablehlo.add %v6745, %v6743 : tensor<512xf32>
    %v6747 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6748 = stablehlo.multiply %v6747, %v6746 : tensor<512xf32>
    %v6749 = stablehlo.subtract %s2b1g3, %v6748 : tensor<512xf32>
    %arsums2b1bt3 = "stablehlo.all_reduce"(%v4354) ({
    ^bb0(%aras2b1bt3: tensor<f32>, %arbs2b1bt3: tensor<f32>):
      %aradds2b1bt3 = stablehlo.add %aras2b1bt3, %arbs2b1bt3 : tensor<f32>
      stablehlo.return %aradds2b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1bt3 = stablehlo.divide %arsums2b1bt3, %arns2b1bt3 : tensor<512xf32>
    %v6750 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6751 = stablehlo.multiply %v6750, %s2b1bt3 : tensor<512xf32>
    %v6752 = stablehlo.add %v6751, %armeans2b1bt3 : tensor<512xf32>
    %v6753 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6754 = stablehlo.multiply %v6753, %s2b1bt3v : tensor<512xf32>
    %v6755 = stablehlo.add %v6754, %v6752 : tensor<512xf32>
    %v6756 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6757 = stablehlo.multiply %v6756, %v6755 : tensor<512xf32>
    %v6758 = stablehlo.subtract %s2b1bt3, %v6757 : tensor<512xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v4059) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x512x1x1xf32>
    %v6759 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6760 = stablehlo.multiply %v6759, %s2b2W1 : tensor<128x512x1x1xf32>
    %v6761 = stablehlo.add %v6760, %armeans2b2W1 : tensor<128x512x1x1xf32>
    %v6762 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6763 = stablehlo.multiply %v6762, %s2b2W1v : tensor<128x512x1x1xf32>
    %v6764 = stablehlo.add %v6763, %v6761 : tensor<128x512x1x1xf32>
    %v6765 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6766 = stablehlo.multiply %v6765, %v6764 : tensor<128x512x1x1xf32>
    %v6767 = stablehlo.subtract %s2b2W1, %v6766 : tensor<128x512x1x1xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v4077) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v6768 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6769 = stablehlo.multiply %v6768, %s2b2g1 : tensor<128xf32>
    %v6770 = stablehlo.add %v6769, %armeans2b2g1 : tensor<128xf32>
    %v6771 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6772 = stablehlo.multiply %v6771, %s2b2g1v : tensor<128xf32>
    %v6773 = stablehlo.add %v6772, %v6770 : tensor<128xf32>
    %v6774 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6775 = stablehlo.multiply %v6774, %v6773 : tensor<128xf32>
    %v6776 = stablehlo.subtract %s2b2g1, %v6775 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v4080) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v6777 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6778 = stablehlo.multiply %v6777, %s2b2bt1 : tensor<128xf32>
    %v6779 = stablehlo.add %v6778, %armeans2b2bt1 : tensor<128xf32>
    %v6780 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6781 = stablehlo.multiply %v6780, %s2b2bt1v : tensor<128xf32>
    %v6782 = stablehlo.add %v6781, %v6779 : tensor<128xf32>
    %v6783 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6784 = stablehlo.multiply %v6783, %v6782 : tensor<128xf32>
    %v6785 = stablehlo.subtract %s2b2bt1, %v6784 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v4089) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v6786 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6787 = stablehlo.multiply %v6786, %s2b2W2 : tensor<128x128x3x3xf32>
    %v6788 = stablehlo.add %v6787, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v6789 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6790 = stablehlo.multiply %v6789, %s2b2W2v : tensor<128x128x3x3xf32>
    %v6791 = stablehlo.add %v6790, %v6788 : tensor<128x128x3x3xf32>
    %v6792 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6793 = stablehlo.multiply %v6792, %v6791 : tensor<128x128x3x3xf32>
    %v6794 = stablehlo.subtract %s2b2W2, %v6793 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v4107) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v6795 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6796 = stablehlo.multiply %v6795, %s2b2g2 : tensor<128xf32>
    %v6797 = stablehlo.add %v6796, %armeans2b2g2 : tensor<128xf32>
    %v6798 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6799 = stablehlo.multiply %v6798, %s2b2g2v : tensor<128xf32>
    %v6800 = stablehlo.add %v6799, %v6797 : tensor<128xf32>
    %v6801 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6802 = stablehlo.multiply %v6801, %v6800 : tensor<128xf32>
    %v6803 = stablehlo.subtract %s2b2g2, %v6802 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v4110) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v6804 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6805 = stablehlo.multiply %v6804, %s2b2bt2 : tensor<128xf32>
    %v6806 = stablehlo.add %v6805, %armeans2b2bt2 : tensor<128xf32>
    %v6807 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6808 = stablehlo.multiply %v6807, %s2b2bt2v : tensor<128xf32>
    %v6809 = stablehlo.add %v6808, %v6806 : tensor<128xf32>
    %v6810 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6811 = stablehlo.multiply %v6810, %v6809 : tensor<128xf32>
    %v6812 = stablehlo.subtract %s2b2bt2, %v6811 : tensor<128xf32>
    %arsums2b2W3 = "stablehlo.all_reduce"(%v4119) ({
    ^bb0(%aras2b2W3: tensor<f32>, %arbs2b2W3: tensor<f32>):
      %aradds2b2W3 = stablehlo.add %aras2b2W3, %arbs2b2W3 : tensor<f32>
      stablehlo.return %aradds2b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b2W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b2W3 = stablehlo.divide %arsums2b2W3, %arns2b2W3 : tensor<512x128x1x1xf32>
    %v6813 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6814 = stablehlo.multiply %v6813, %s2b2W3 : tensor<512x128x1x1xf32>
    %v6815 = stablehlo.add %v6814, %armeans2b2W3 : tensor<512x128x1x1xf32>
    %v6816 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6817 = stablehlo.multiply %v6816, %s2b2W3v : tensor<512x128x1x1xf32>
    %v6818 = stablehlo.add %v6817, %v6815 : tensor<512x128x1x1xf32>
    %v6819 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6820 = stablehlo.multiply %v6819, %v6818 : tensor<512x128x1x1xf32>
    %v6821 = stablehlo.subtract %s2b2W3, %v6820 : tensor<512x128x1x1xf32>
    %arsums2b2g3 = "stablehlo.all_reduce"(%v4137) ({
    ^bb0(%aras2b2g3: tensor<f32>, %arbs2b2g3: tensor<f32>):
      %aradds2b2g3 = stablehlo.add %aras2b2g3, %arbs2b2g3 : tensor<f32>
      stablehlo.return %aradds2b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g3 = stablehlo.divide %arsums2b2g3, %arns2b2g3 : tensor<512xf32>
    %v6822 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6823 = stablehlo.multiply %v6822, %s2b2g3 : tensor<512xf32>
    %v6824 = stablehlo.add %v6823, %armeans2b2g3 : tensor<512xf32>
    %v6825 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6826 = stablehlo.multiply %v6825, %s2b2g3v : tensor<512xf32>
    %v6827 = stablehlo.add %v6826, %v6824 : tensor<512xf32>
    %v6828 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6829 = stablehlo.multiply %v6828, %v6827 : tensor<512xf32>
    %v6830 = stablehlo.subtract %s2b2g3, %v6829 : tensor<512xf32>
    %arsums2b2bt3 = "stablehlo.all_reduce"(%v4140) ({
    ^bb0(%aras2b2bt3: tensor<f32>, %arbs2b2bt3: tensor<f32>):
      %aradds2b2bt3 = stablehlo.add %aras2b2bt3, %arbs2b2bt3 : tensor<f32>
      stablehlo.return %aradds2b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2bt3 = stablehlo.divide %arsums2b2bt3, %arns2b2bt3 : tensor<512xf32>
    %v6831 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6832 = stablehlo.multiply %v6831, %s2b2bt3 : tensor<512xf32>
    %v6833 = stablehlo.add %v6832, %armeans2b2bt3 : tensor<512xf32>
    %v6834 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6835 = stablehlo.multiply %v6834, %s2b2bt3v : tensor<512xf32>
    %v6836 = stablehlo.add %v6835, %v6833 : tensor<512xf32>
    %v6837 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6838 = stablehlo.multiply %v6837, %v6836 : tensor<512xf32>
    %v6839 = stablehlo.subtract %s2b2bt3, %v6838 : tensor<512xf32>
    %arsums2b3W1 = "stablehlo.all_reduce"(%v3845) ({
    ^bb0(%aras2b3W1: tensor<f32>, %arbs2b3W1: tensor<f32>):
      %aradds2b3W1 = stablehlo.add %aras2b3W1, %arbs2b3W1 : tensor<f32>
      stablehlo.return %aradds2b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x512x1x1xf32>) -> tensor<128x512x1x1xf32>
    %arns2b3W1 = stablehlo.constant dense<4.0> : tensor<128x512x1x1xf32>
    %armeans2b3W1 = stablehlo.divide %arsums2b3W1, %arns2b3W1 : tensor<128x512x1x1xf32>
    %v6840 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6841 = stablehlo.multiply %v6840, %s2b3W1 : tensor<128x512x1x1xf32>
    %v6842 = stablehlo.add %v6841, %armeans2b3W1 : tensor<128x512x1x1xf32>
    %v6843 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6844 = stablehlo.multiply %v6843, %s2b3W1v : tensor<128x512x1x1xf32>
    %v6845 = stablehlo.add %v6844, %v6842 : tensor<128x512x1x1xf32>
    %v6846 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x512x1x1xf32>
    %v6847 = stablehlo.multiply %v6846, %v6845 : tensor<128x512x1x1xf32>
    %v6848 = stablehlo.subtract %s2b3W1, %v6847 : tensor<128x512x1x1xf32>
    %arsums2b3g1 = "stablehlo.all_reduce"(%v3863) ({
    ^bb0(%aras2b3g1: tensor<f32>, %arbs2b3g1: tensor<f32>):
      %aradds2b3g1 = stablehlo.add %aras2b3g1, %arbs2b3g1 : tensor<f32>
      stablehlo.return %aradds2b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g1 = stablehlo.divide %arsums2b3g1, %arns2b3g1 : tensor<128xf32>
    %v6849 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6850 = stablehlo.multiply %v6849, %s2b3g1 : tensor<128xf32>
    %v6851 = stablehlo.add %v6850, %armeans2b3g1 : tensor<128xf32>
    %v6852 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6853 = stablehlo.multiply %v6852, %s2b3g1v : tensor<128xf32>
    %v6854 = stablehlo.add %v6853, %v6851 : tensor<128xf32>
    %v6855 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6856 = stablehlo.multiply %v6855, %v6854 : tensor<128xf32>
    %v6857 = stablehlo.subtract %s2b3g1, %v6856 : tensor<128xf32>
    %arsums2b3bt1 = "stablehlo.all_reduce"(%v3866) ({
    ^bb0(%aras2b3bt1: tensor<f32>, %arbs2b3bt1: tensor<f32>):
      %aradds2b3bt1 = stablehlo.add %aras2b3bt1, %arbs2b3bt1 : tensor<f32>
      stablehlo.return %aradds2b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt1 = stablehlo.divide %arsums2b3bt1, %arns2b3bt1 : tensor<128xf32>
    %v6858 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6859 = stablehlo.multiply %v6858, %s2b3bt1 : tensor<128xf32>
    %v6860 = stablehlo.add %v6859, %armeans2b3bt1 : tensor<128xf32>
    %v6861 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6862 = stablehlo.multiply %v6861, %s2b3bt1v : tensor<128xf32>
    %v6863 = stablehlo.add %v6862, %v6860 : tensor<128xf32>
    %v6864 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6865 = stablehlo.multiply %v6864, %v6863 : tensor<128xf32>
    %v6866 = stablehlo.subtract %s2b3bt1, %v6865 : tensor<128xf32>
    %arsums2b3W2 = "stablehlo.all_reduce"(%v3875) ({
    ^bb0(%aras2b3W2: tensor<f32>, %arbs2b3W2: tensor<f32>):
      %aradds2b3W2 = stablehlo.add %aras2b3W2, %arbs2b3W2 : tensor<f32>
      stablehlo.return %aradds2b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b3W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b3W2 = stablehlo.divide %arsums2b3W2, %arns2b3W2 : tensor<128x128x3x3xf32>
    %v6867 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6868 = stablehlo.multiply %v6867, %s2b3W2 : tensor<128x128x3x3xf32>
    %v6869 = stablehlo.add %v6868, %armeans2b3W2 : tensor<128x128x3x3xf32>
    %v6870 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6871 = stablehlo.multiply %v6870, %s2b3W2v : tensor<128x128x3x3xf32>
    %v6872 = stablehlo.add %v6871, %v6869 : tensor<128x128x3x3xf32>
    %v6873 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v6874 = stablehlo.multiply %v6873, %v6872 : tensor<128x128x3x3xf32>
    %v6875 = stablehlo.subtract %s2b3W2, %v6874 : tensor<128x128x3x3xf32>
    %arsums2b3g2 = "stablehlo.all_reduce"(%v3893) ({
    ^bb0(%aras2b3g2: tensor<f32>, %arbs2b3g2: tensor<f32>):
      %aradds2b3g2 = stablehlo.add %aras2b3g2, %arbs2b3g2 : tensor<f32>
      stablehlo.return %aradds2b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3g2 = stablehlo.divide %arsums2b3g2, %arns2b3g2 : tensor<128xf32>
    %v6876 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6877 = stablehlo.multiply %v6876, %s2b3g2 : tensor<128xf32>
    %v6878 = stablehlo.add %v6877, %armeans2b3g2 : tensor<128xf32>
    %v6879 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6880 = stablehlo.multiply %v6879, %s2b3g2v : tensor<128xf32>
    %v6881 = stablehlo.add %v6880, %v6878 : tensor<128xf32>
    %v6882 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6883 = stablehlo.multiply %v6882, %v6881 : tensor<128xf32>
    %v6884 = stablehlo.subtract %s2b3g2, %v6883 : tensor<128xf32>
    %arsums2b3bt2 = "stablehlo.all_reduce"(%v3896) ({
    ^bb0(%aras2b3bt2: tensor<f32>, %arbs2b3bt2: tensor<f32>):
      %aradds2b3bt2 = stablehlo.add %aras2b3bt2, %arbs2b3bt2 : tensor<f32>
      stablehlo.return %aradds2b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b3bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b3bt2 = stablehlo.divide %arsums2b3bt2, %arns2b3bt2 : tensor<128xf32>
    %v6885 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6886 = stablehlo.multiply %v6885, %s2b3bt2 : tensor<128xf32>
    %v6887 = stablehlo.add %v6886, %armeans2b3bt2 : tensor<128xf32>
    %v6888 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6889 = stablehlo.multiply %v6888, %s2b3bt2v : tensor<128xf32>
    %v6890 = stablehlo.add %v6889, %v6887 : tensor<128xf32>
    %v6891 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v6892 = stablehlo.multiply %v6891, %v6890 : tensor<128xf32>
    %v6893 = stablehlo.subtract %s2b3bt2, %v6892 : tensor<128xf32>
    %arsums2b3W3 = "stablehlo.all_reduce"(%v3905) ({
    ^bb0(%aras2b3W3: tensor<f32>, %arbs2b3W3: tensor<f32>):
      %aradds2b3W3 = stablehlo.add %aras2b3W3, %arbs2b3W3 : tensor<f32>
      stablehlo.return %aradds2b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x128x1x1xf32>) -> tensor<512x128x1x1xf32>
    %arns2b3W3 = stablehlo.constant dense<4.0> : tensor<512x128x1x1xf32>
    %armeans2b3W3 = stablehlo.divide %arsums2b3W3, %arns2b3W3 : tensor<512x128x1x1xf32>
    %v6894 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6895 = stablehlo.multiply %v6894, %s2b3W3 : tensor<512x128x1x1xf32>
    %v6896 = stablehlo.add %v6895, %armeans2b3W3 : tensor<512x128x1x1xf32>
    %v6897 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6898 = stablehlo.multiply %v6897, %s2b3W3v : tensor<512x128x1x1xf32>
    %v6899 = stablehlo.add %v6898, %v6896 : tensor<512x128x1x1xf32>
    %v6900 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x128x1x1xf32>
    %v6901 = stablehlo.multiply %v6900, %v6899 : tensor<512x128x1x1xf32>
    %v6902 = stablehlo.subtract %s2b3W3, %v6901 : tensor<512x128x1x1xf32>
    %arsums2b3g3 = "stablehlo.all_reduce"(%v3923) ({
    ^bb0(%aras2b3g3: tensor<f32>, %arbs2b3g3: tensor<f32>):
      %aradds2b3g3 = stablehlo.add %aras2b3g3, %arbs2b3g3 : tensor<f32>
      stablehlo.return %aradds2b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3g3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3g3 = stablehlo.divide %arsums2b3g3, %arns2b3g3 : tensor<512xf32>
    %v6903 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6904 = stablehlo.multiply %v6903, %s2b3g3 : tensor<512xf32>
    %v6905 = stablehlo.add %v6904, %armeans2b3g3 : tensor<512xf32>
    %v6906 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6907 = stablehlo.multiply %v6906, %s2b3g3v : tensor<512xf32>
    %v6908 = stablehlo.add %v6907, %v6905 : tensor<512xf32>
    %v6909 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6910 = stablehlo.multiply %v6909, %v6908 : tensor<512xf32>
    %v6911 = stablehlo.subtract %s2b3g3, %v6910 : tensor<512xf32>
    %arsums2b3bt3 = "stablehlo.all_reduce"(%v3926) ({
    ^bb0(%aras2b3bt3: tensor<f32>, %arbs2b3bt3: tensor<f32>):
      %aradds2b3bt3 = stablehlo.add %aras2b3bt3, %arbs2b3bt3 : tensor<f32>
      stablehlo.return %aradds2b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b3bt3 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b3bt3 = stablehlo.divide %arsums2b3bt3, %arns2b3bt3 : tensor<512xf32>
    %v6912 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6913 = stablehlo.multiply %v6912, %s2b3bt3 : tensor<512xf32>
    %v6914 = stablehlo.add %v6913, %armeans2b3bt3 : tensor<512xf32>
    %v6915 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6916 = stablehlo.multiply %v6915, %s2b3bt3v : tensor<512xf32>
    %v6917 = stablehlo.add %v6916, %v6914 : tensor<512xf32>
    %v6918 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v6919 = stablehlo.multiply %v6918, %v6917 : tensor<512xf32>
    %v6920 = stablehlo.subtract %s2b3bt3, %v6919 : tensor<512xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v3597) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x512x1x1xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x512x1x1xf32>
    %v6921 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6922 = stablehlo.multiply %v6921, %s3b0W1 : tensor<256x512x1x1xf32>
    %v6923 = stablehlo.add %v6922, %armeans3b0W1 : tensor<256x512x1x1xf32>
    %v6924 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6925 = stablehlo.multiply %v6924, %s3b0W1v : tensor<256x512x1x1xf32>
    %v6926 = stablehlo.add %v6925, %v6923 : tensor<256x512x1x1xf32>
    %v6927 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x512x1x1xf32>
    %v6928 = stablehlo.multiply %v6927, %v6926 : tensor<256x512x1x1xf32>
    %v6929 = stablehlo.subtract %s3b0W1, %v6928 : tensor<256x512x1x1xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v3615) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v6930 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6931 = stablehlo.multiply %v6930, %s3b0g1 : tensor<256xf32>
    %v6932 = stablehlo.add %v6931, %armeans3b0g1 : tensor<256xf32>
    %v6933 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6934 = stablehlo.multiply %v6933, %s3b0g1v : tensor<256xf32>
    %v6935 = stablehlo.add %v6934, %v6932 : tensor<256xf32>
    %v6936 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6937 = stablehlo.multiply %v6936, %v6935 : tensor<256xf32>
    %v6938 = stablehlo.subtract %s3b0g1, %v6937 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v3618) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v6939 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6940 = stablehlo.multiply %v6939, %s3b0bt1 : tensor<256xf32>
    %v6941 = stablehlo.add %v6940, %armeans3b0bt1 : tensor<256xf32>
    %v6942 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6943 = stablehlo.multiply %v6942, %s3b0bt1v : tensor<256xf32>
    %v6944 = stablehlo.add %v6943, %v6941 : tensor<256xf32>
    %v6945 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6946 = stablehlo.multiply %v6945, %v6944 : tensor<256xf32>
    %v6947 = stablehlo.subtract %s3b0bt1, %v6946 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v3629) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v6948 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6949 = stablehlo.multiply %v6948, %s3b0W2 : tensor<256x256x3x3xf32>
    %v6950 = stablehlo.add %v6949, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v6951 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6952 = stablehlo.multiply %v6951, %s3b0W2v : tensor<256x256x3x3xf32>
    %v6953 = stablehlo.add %v6952, %v6950 : tensor<256x256x3x3xf32>
    %v6954 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v6955 = stablehlo.multiply %v6954, %v6953 : tensor<256x256x3x3xf32>
    %v6956 = stablehlo.subtract %s3b0W2, %v6955 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v3647) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v6957 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6958 = stablehlo.multiply %v6957, %s3b0g2 : tensor<256xf32>
    %v6959 = stablehlo.add %v6958, %armeans3b0g2 : tensor<256xf32>
    %v6960 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6961 = stablehlo.multiply %v6960, %s3b0g2v : tensor<256xf32>
    %v6962 = stablehlo.add %v6961, %v6959 : tensor<256xf32>
    %v6963 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6964 = stablehlo.multiply %v6963, %v6962 : tensor<256xf32>
    %v6965 = stablehlo.subtract %s3b0g2, %v6964 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v3650) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v6966 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6967 = stablehlo.multiply %v6966, %s3b0bt2 : tensor<256xf32>
    %v6968 = stablehlo.add %v6967, %armeans3b0bt2 : tensor<256xf32>
    %v6969 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6970 = stablehlo.multiply %v6969, %s3b0bt2v : tensor<256xf32>
    %v6971 = stablehlo.add %v6970, %v6968 : tensor<256xf32>
    %v6972 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v6973 = stablehlo.multiply %v6972, %v6971 : tensor<256xf32>
    %v6974 = stablehlo.subtract %s3b0bt2, %v6973 : tensor<256xf32>
    %arsums3b0W3 = "stablehlo.all_reduce"(%v3659) ({
    ^bb0(%aras3b0W3: tensor<f32>, %arbs3b0W3: tensor<f32>):
      %aradds3b0W3 = stablehlo.add %aras3b0W3, %arbs3b0W3 : tensor<f32>
      stablehlo.return %aradds3b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b0W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b0W3 = stablehlo.divide %arsums3b0W3, %arns3b0W3 : tensor<1024x256x1x1xf32>
    %v6975 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6976 = stablehlo.multiply %v6975, %s3b0W3 : tensor<1024x256x1x1xf32>
    %v6977 = stablehlo.add %v6976, %armeans3b0W3 : tensor<1024x256x1x1xf32>
    %v6978 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6979 = stablehlo.multiply %v6978, %s3b0W3v : tensor<1024x256x1x1xf32>
    %v6980 = stablehlo.add %v6979, %v6977 : tensor<1024x256x1x1xf32>
    %v6981 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v6982 = stablehlo.multiply %v6981, %v6980 : tensor<1024x256x1x1xf32>
    %v6983 = stablehlo.subtract %s3b0W3, %v6982 : tensor<1024x256x1x1xf32>
    %arsums3b0g3 = "stablehlo.all_reduce"(%v3677) ({
    ^bb0(%aras3b0g3: tensor<f32>, %arbs3b0g3: tensor<f32>):
      %aradds3b0g3 = stablehlo.add %aras3b0g3, %arbs3b0g3 : tensor<f32>
      stablehlo.return %aradds3b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g3 = stablehlo.divide %arsums3b0g3, %arns3b0g3 : tensor<1024xf32>
    %v6984 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6985 = stablehlo.multiply %v6984, %s3b0g3 : tensor<1024xf32>
    %v6986 = stablehlo.add %v6985, %armeans3b0g3 : tensor<1024xf32>
    %v6987 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6988 = stablehlo.multiply %v6987, %s3b0g3v : tensor<1024xf32>
    %v6989 = stablehlo.add %v6988, %v6986 : tensor<1024xf32>
    %v6990 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6991 = stablehlo.multiply %v6990, %v6989 : tensor<1024xf32>
    %v6992 = stablehlo.subtract %s3b0g3, %v6991 : tensor<1024xf32>
    %arsums3b0bt3 = "stablehlo.all_reduce"(%v3680) ({
    ^bb0(%aras3b0bt3: tensor<f32>, %arbs3b0bt3: tensor<f32>):
      %aradds3b0bt3 = stablehlo.add %aras3b0bt3, %arbs3b0bt3 : tensor<f32>
      stablehlo.return %aradds3b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0bt3 = stablehlo.divide %arsums3b0bt3, %arns3b0bt3 : tensor<1024xf32>
    %v6993 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6994 = stablehlo.multiply %v6993, %s3b0bt3 : tensor<1024xf32>
    %v6995 = stablehlo.add %v6994, %armeans3b0bt3 : tensor<1024xf32>
    %v6996 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v6997 = stablehlo.multiply %v6996, %s3b0bt3v : tensor<1024xf32>
    %v6998 = stablehlo.add %v6997, %v6995 : tensor<1024xf32>
    %v6999 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7000 = stablehlo.multiply %v6999, %v6998 : tensor<1024xf32>
    %v7001 = stablehlo.subtract %s3b0bt3, %v7000 : tensor<1024xf32>
    %arsums3b0Wp = "stablehlo.all_reduce"(%v3691) ({
    ^bb0(%aras3b0Wp: tensor<f32>, %arbs3b0Wp: tensor<f32>):
      %aradds3b0Wp = stablehlo.add %aras3b0Wp, %arbs3b0Wp : tensor<f32>
      stablehlo.return %aradds3b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x512x1x1xf32>) -> tensor<1024x512x1x1xf32>
    %arns3b0Wp = stablehlo.constant dense<4.0> : tensor<1024x512x1x1xf32>
    %armeans3b0Wp = stablehlo.divide %arsums3b0Wp, %arns3b0Wp : tensor<1024x512x1x1xf32>
    %v7002 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7003 = stablehlo.multiply %v7002, %s3b0Wp : tensor<1024x512x1x1xf32>
    %v7004 = stablehlo.add %v7003, %armeans3b0Wp : tensor<1024x512x1x1xf32>
    %v7005 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7006 = stablehlo.multiply %v7005, %s3b0Wpv : tensor<1024x512x1x1xf32>
    %v7007 = stablehlo.add %v7006, %v7004 : tensor<1024x512x1x1xf32>
    %v7008 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x512x1x1xf32>
    %v7009 = stablehlo.multiply %v7008, %v7007 : tensor<1024x512x1x1xf32>
    %v7010 = stablehlo.subtract %s3b0Wp, %v7009 : tensor<1024x512x1x1xf32>
    %arsums3b0gp = "stablehlo.all_reduce"(%v3709) ({
    ^bb0(%aras3b0gp: tensor<f32>, %arbs3b0gp: tensor<f32>):
      %aradds3b0gp = stablehlo.add %aras3b0gp, %arbs3b0gp : tensor<f32>
      stablehlo.return %aradds3b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0gp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0gp = stablehlo.divide %arsums3b0gp, %arns3b0gp : tensor<1024xf32>
    %v7011 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7012 = stablehlo.multiply %v7011, %s3b0gp : tensor<1024xf32>
    %v7013 = stablehlo.add %v7012, %armeans3b0gp : tensor<1024xf32>
    %v7014 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7015 = stablehlo.multiply %v7014, %s3b0gpv : tensor<1024xf32>
    %v7016 = stablehlo.add %v7015, %v7013 : tensor<1024xf32>
    %v7017 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7018 = stablehlo.multiply %v7017, %v7016 : tensor<1024xf32>
    %v7019 = stablehlo.subtract %s3b0gp, %v7018 : tensor<1024xf32>
    %arsums3b0btp = "stablehlo.all_reduce"(%v3712) ({
    ^bb0(%aras3b0btp: tensor<f32>, %arbs3b0btp: tensor<f32>):
      %aradds3b0btp = stablehlo.add %aras3b0btp, %arbs3b0btp : tensor<f32>
      stablehlo.return %aradds3b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0btp = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0btp = stablehlo.divide %arsums3b0btp, %arns3b0btp : tensor<1024xf32>
    %v7020 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7021 = stablehlo.multiply %v7020, %s3b0btp : tensor<1024xf32>
    %v7022 = stablehlo.add %v7021, %armeans3b0btp : tensor<1024xf32>
    %v7023 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7024 = stablehlo.multiply %v7023, %s3b0btpv : tensor<1024xf32>
    %v7025 = stablehlo.add %v7024, %v7022 : tensor<1024xf32>
    %v7026 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7027 = stablehlo.multiply %v7026, %v7025 : tensor<1024xf32>
    %v7028 = stablehlo.subtract %s3b0btp, %v7027 : tensor<1024xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v3341) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x1024x1x1xf32>
    %v7029 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7030 = stablehlo.multiply %v7029, %s3b1W1 : tensor<256x1024x1x1xf32>
    %v7031 = stablehlo.add %v7030, %armeans3b1W1 : tensor<256x1024x1x1xf32>
    %v7032 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7033 = stablehlo.multiply %v7032, %s3b1W1v : tensor<256x1024x1x1xf32>
    %v7034 = stablehlo.add %v7033, %v7031 : tensor<256x1024x1x1xf32>
    %v7035 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7036 = stablehlo.multiply %v7035, %v7034 : tensor<256x1024x1x1xf32>
    %v7037 = stablehlo.subtract %s3b1W1, %v7036 : tensor<256x1024x1x1xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v3359) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v7038 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7039 = stablehlo.multiply %v7038, %s3b1g1 : tensor<256xf32>
    %v7040 = stablehlo.add %v7039, %armeans3b1g1 : tensor<256xf32>
    %v7041 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7042 = stablehlo.multiply %v7041, %s3b1g1v : tensor<256xf32>
    %v7043 = stablehlo.add %v7042, %v7040 : tensor<256xf32>
    %v7044 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7045 = stablehlo.multiply %v7044, %v7043 : tensor<256xf32>
    %v7046 = stablehlo.subtract %s3b1g1, %v7045 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v3362) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v7047 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7048 = stablehlo.multiply %v7047, %s3b1bt1 : tensor<256xf32>
    %v7049 = stablehlo.add %v7048, %armeans3b1bt1 : tensor<256xf32>
    %v7050 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7051 = stablehlo.multiply %v7050, %s3b1bt1v : tensor<256xf32>
    %v7052 = stablehlo.add %v7051, %v7049 : tensor<256xf32>
    %v7053 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7054 = stablehlo.multiply %v7053, %v7052 : tensor<256xf32>
    %v7055 = stablehlo.subtract %s3b1bt1, %v7054 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v3371) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v7056 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7057 = stablehlo.multiply %v7056, %s3b1W2 : tensor<256x256x3x3xf32>
    %v7058 = stablehlo.add %v7057, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v7059 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7060 = stablehlo.multiply %v7059, %s3b1W2v : tensor<256x256x3x3xf32>
    %v7061 = stablehlo.add %v7060, %v7058 : tensor<256x256x3x3xf32>
    %v7062 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7063 = stablehlo.multiply %v7062, %v7061 : tensor<256x256x3x3xf32>
    %v7064 = stablehlo.subtract %s3b1W2, %v7063 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v3389) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v7065 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7066 = stablehlo.multiply %v7065, %s3b1g2 : tensor<256xf32>
    %v7067 = stablehlo.add %v7066, %armeans3b1g2 : tensor<256xf32>
    %v7068 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7069 = stablehlo.multiply %v7068, %s3b1g2v : tensor<256xf32>
    %v7070 = stablehlo.add %v7069, %v7067 : tensor<256xf32>
    %v7071 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7072 = stablehlo.multiply %v7071, %v7070 : tensor<256xf32>
    %v7073 = stablehlo.subtract %s3b1g2, %v7072 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v3392) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v7074 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7075 = stablehlo.multiply %v7074, %s3b1bt2 : tensor<256xf32>
    %v7076 = stablehlo.add %v7075, %armeans3b1bt2 : tensor<256xf32>
    %v7077 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7078 = stablehlo.multiply %v7077, %s3b1bt2v : tensor<256xf32>
    %v7079 = stablehlo.add %v7078, %v7076 : tensor<256xf32>
    %v7080 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7081 = stablehlo.multiply %v7080, %v7079 : tensor<256xf32>
    %v7082 = stablehlo.subtract %s3b1bt2, %v7081 : tensor<256xf32>
    %arsums3b1W3 = "stablehlo.all_reduce"(%v3401) ({
    ^bb0(%aras3b1W3: tensor<f32>, %arbs3b1W3: tensor<f32>):
      %aradds3b1W3 = stablehlo.add %aras3b1W3, %arbs3b1W3 : tensor<f32>
      stablehlo.return %aradds3b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b1W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b1W3 = stablehlo.divide %arsums3b1W3, %arns3b1W3 : tensor<1024x256x1x1xf32>
    %v7083 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7084 = stablehlo.multiply %v7083, %s3b1W3 : tensor<1024x256x1x1xf32>
    %v7085 = stablehlo.add %v7084, %armeans3b1W3 : tensor<1024x256x1x1xf32>
    %v7086 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7087 = stablehlo.multiply %v7086, %s3b1W3v : tensor<1024x256x1x1xf32>
    %v7088 = stablehlo.add %v7087, %v7085 : tensor<1024x256x1x1xf32>
    %v7089 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7090 = stablehlo.multiply %v7089, %v7088 : tensor<1024x256x1x1xf32>
    %v7091 = stablehlo.subtract %s3b1W3, %v7090 : tensor<1024x256x1x1xf32>
    %arsums3b1g3 = "stablehlo.all_reduce"(%v3419) ({
    ^bb0(%aras3b1g3: tensor<f32>, %arbs3b1g3: tensor<f32>):
      %aradds3b1g3 = stablehlo.add %aras3b1g3, %arbs3b1g3 : tensor<f32>
      stablehlo.return %aradds3b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g3 = stablehlo.divide %arsums3b1g3, %arns3b1g3 : tensor<1024xf32>
    %v7092 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7093 = stablehlo.multiply %v7092, %s3b1g3 : tensor<1024xf32>
    %v7094 = stablehlo.add %v7093, %armeans3b1g3 : tensor<1024xf32>
    %v7095 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7096 = stablehlo.multiply %v7095, %s3b1g3v : tensor<1024xf32>
    %v7097 = stablehlo.add %v7096, %v7094 : tensor<1024xf32>
    %v7098 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7099 = stablehlo.multiply %v7098, %v7097 : tensor<1024xf32>
    %v7100 = stablehlo.subtract %s3b1g3, %v7099 : tensor<1024xf32>
    %arsums3b1bt3 = "stablehlo.all_reduce"(%v3422) ({
    ^bb0(%aras3b1bt3: tensor<f32>, %arbs3b1bt3: tensor<f32>):
      %aradds3b1bt3 = stablehlo.add %aras3b1bt3, %arbs3b1bt3 : tensor<f32>
      stablehlo.return %aradds3b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1bt3 = stablehlo.divide %arsums3b1bt3, %arns3b1bt3 : tensor<1024xf32>
    %v7101 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7102 = stablehlo.multiply %v7101, %s3b1bt3 : tensor<1024xf32>
    %v7103 = stablehlo.add %v7102, %armeans3b1bt3 : tensor<1024xf32>
    %v7104 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7105 = stablehlo.multiply %v7104, %s3b1bt3v : tensor<1024xf32>
    %v7106 = stablehlo.add %v7105, %v7103 : tensor<1024xf32>
    %v7107 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7108 = stablehlo.multiply %v7107, %v7106 : tensor<1024xf32>
    %v7109 = stablehlo.subtract %s3b1bt3, %v7108 : tensor<1024xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v3127) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x1024x1x1xf32>
    %v7110 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7111 = stablehlo.multiply %v7110, %s3b2W1 : tensor<256x1024x1x1xf32>
    %v7112 = stablehlo.add %v7111, %armeans3b2W1 : tensor<256x1024x1x1xf32>
    %v7113 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7114 = stablehlo.multiply %v7113, %s3b2W1v : tensor<256x1024x1x1xf32>
    %v7115 = stablehlo.add %v7114, %v7112 : tensor<256x1024x1x1xf32>
    %v7116 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7117 = stablehlo.multiply %v7116, %v7115 : tensor<256x1024x1x1xf32>
    %v7118 = stablehlo.subtract %s3b2W1, %v7117 : tensor<256x1024x1x1xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v3145) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v7119 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7120 = stablehlo.multiply %v7119, %s3b2g1 : tensor<256xf32>
    %v7121 = stablehlo.add %v7120, %armeans3b2g1 : tensor<256xf32>
    %v7122 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7123 = stablehlo.multiply %v7122, %s3b2g1v : tensor<256xf32>
    %v7124 = stablehlo.add %v7123, %v7121 : tensor<256xf32>
    %v7125 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7126 = stablehlo.multiply %v7125, %v7124 : tensor<256xf32>
    %v7127 = stablehlo.subtract %s3b2g1, %v7126 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v3148) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v7128 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7129 = stablehlo.multiply %v7128, %s3b2bt1 : tensor<256xf32>
    %v7130 = stablehlo.add %v7129, %armeans3b2bt1 : tensor<256xf32>
    %v7131 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7132 = stablehlo.multiply %v7131, %s3b2bt1v : tensor<256xf32>
    %v7133 = stablehlo.add %v7132, %v7130 : tensor<256xf32>
    %v7134 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7135 = stablehlo.multiply %v7134, %v7133 : tensor<256xf32>
    %v7136 = stablehlo.subtract %s3b2bt1, %v7135 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v3157) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v7137 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7138 = stablehlo.multiply %v7137, %s3b2W2 : tensor<256x256x3x3xf32>
    %v7139 = stablehlo.add %v7138, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v7140 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7141 = stablehlo.multiply %v7140, %s3b2W2v : tensor<256x256x3x3xf32>
    %v7142 = stablehlo.add %v7141, %v7139 : tensor<256x256x3x3xf32>
    %v7143 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7144 = stablehlo.multiply %v7143, %v7142 : tensor<256x256x3x3xf32>
    %v7145 = stablehlo.subtract %s3b2W2, %v7144 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v3175) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v7146 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7147 = stablehlo.multiply %v7146, %s3b2g2 : tensor<256xf32>
    %v7148 = stablehlo.add %v7147, %armeans3b2g2 : tensor<256xf32>
    %v7149 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7150 = stablehlo.multiply %v7149, %s3b2g2v : tensor<256xf32>
    %v7151 = stablehlo.add %v7150, %v7148 : tensor<256xf32>
    %v7152 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7153 = stablehlo.multiply %v7152, %v7151 : tensor<256xf32>
    %v7154 = stablehlo.subtract %s3b2g2, %v7153 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v3178) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v7155 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7156 = stablehlo.multiply %v7155, %s3b2bt2 : tensor<256xf32>
    %v7157 = stablehlo.add %v7156, %armeans3b2bt2 : tensor<256xf32>
    %v7158 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7159 = stablehlo.multiply %v7158, %s3b2bt2v : tensor<256xf32>
    %v7160 = stablehlo.add %v7159, %v7157 : tensor<256xf32>
    %v7161 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7162 = stablehlo.multiply %v7161, %v7160 : tensor<256xf32>
    %v7163 = stablehlo.subtract %s3b2bt2, %v7162 : tensor<256xf32>
    %arsums3b2W3 = "stablehlo.all_reduce"(%v3187) ({
    ^bb0(%aras3b2W3: tensor<f32>, %arbs3b2W3: tensor<f32>):
      %aradds3b2W3 = stablehlo.add %aras3b2W3, %arbs3b2W3 : tensor<f32>
      stablehlo.return %aradds3b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b2W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b2W3 = stablehlo.divide %arsums3b2W3, %arns3b2W3 : tensor<1024x256x1x1xf32>
    %v7164 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7165 = stablehlo.multiply %v7164, %s3b2W3 : tensor<1024x256x1x1xf32>
    %v7166 = stablehlo.add %v7165, %armeans3b2W3 : tensor<1024x256x1x1xf32>
    %v7167 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7168 = stablehlo.multiply %v7167, %s3b2W3v : tensor<1024x256x1x1xf32>
    %v7169 = stablehlo.add %v7168, %v7166 : tensor<1024x256x1x1xf32>
    %v7170 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7171 = stablehlo.multiply %v7170, %v7169 : tensor<1024x256x1x1xf32>
    %v7172 = stablehlo.subtract %s3b2W3, %v7171 : tensor<1024x256x1x1xf32>
    %arsums3b2g3 = "stablehlo.all_reduce"(%v3205) ({
    ^bb0(%aras3b2g3: tensor<f32>, %arbs3b2g3: tensor<f32>):
      %aradds3b2g3 = stablehlo.add %aras3b2g3, %arbs3b2g3 : tensor<f32>
      stablehlo.return %aradds3b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g3 = stablehlo.divide %arsums3b2g3, %arns3b2g3 : tensor<1024xf32>
    %v7173 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7174 = stablehlo.multiply %v7173, %s3b2g3 : tensor<1024xf32>
    %v7175 = stablehlo.add %v7174, %armeans3b2g3 : tensor<1024xf32>
    %v7176 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7177 = stablehlo.multiply %v7176, %s3b2g3v : tensor<1024xf32>
    %v7178 = stablehlo.add %v7177, %v7175 : tensor<1024xf32>
    %v7179 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7180 = stablehlo.multiply %v7179, %v7178 : tensor<1024xf32>
    %v7181 = stablehlo.subtract %s3b2g3, %v7180 : tensor<1024xf32>
    %arsums3b2bt3 = "stablehlo.all_reduce"(%v3208) ({
    ^bb0(%aras3b2bt3: tensor<f32>, %arbs3b2bt3: tensor<f32>):
      %aradds3b2bt3 = stablehlo.add %aras3b2bt3, %arbs3b2bt3 : tensor<f32>
      stablehlo.return %aradds3b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2bt3 = stablehlo.divide %arsums3b2bt3, %arns3b2bt3 : tensor<1024xf32>
    %v7182 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7183 = stablehlo.multiply %v7182, %s3b2bt3 : tensor<1024xf32>
    %v7184 = stablehlo.add %v7183, %armeans3b2bt3 : tensor<1024xf32>
    %v7185 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7186 = stablehlo.multiply %v7185, %s3b2bt3v : tensor<1024xf32>
    %v7187 = stablehlo.add %v7186, %v7184 : tensor<1024xf32>
    %v7188 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7189 = stablehlo.multiply %v7188, %v7187 : tensor<1024xf32>
    %v7190 = stablehlo.subtract %s3b2bt3, %v7189 : tensor<1024xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v2913) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x1024x1x1xf32>
    %v7191 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7192 = stablehlo.multiply %v7191, %s3b3W1 : tensor<256x1024x1x1xf32>
    %v7193 = stablehlo.add %v7192, %armeans3b3W1 : tensor<256x1024x1x1xf32>
    %v7194 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7195 = stablehlo.multiply %v7194, %s3b3W1v : tensor<256x1024x1x1xf32>
    %v7196 = stablehlo.add %v7195, %v7193 : tensor<256x1024x1x1xf32>
    %v7197 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7198 = stablehlo.multiply %v7197, %v7196 : tensor<256x1024x1x1xf32>
    %v7199 = stablehlo.subtract %s3b3W1, %v7198 : tensor<256x1024x1x1xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v2931) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v7200 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7201 = stablehlo.multiply %v7200, %s3b3g1 : tensor<256xf32>
    %v7202 = stablehlo.add %v7201, %armeans3b3g1 : tensor<256xf32>
    %v7203 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7204 = stablehlo.multiply %v7203, %s3b3g1v : tensor<256xf32>
    %v7205 = stablehlo.add %v7204, %v7202 : tensor<256xf32>
    %v7206 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7207 = stablehlo.multiply %v7206, %v7205 : tensor<256xf32>
    %v7208 = stablehlo.subtract %s3b3g1, %v7207 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v2934) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v7209 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7210 = stablehlo.multiply %v7209, %s3b3bt1 : tensor<256xf32>
    %v7211 = stablehlo.add %v7210, %armeans3b3bt1 : tensor<256xf32>
    %v7212 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7213 = stablehlo.multiply %v7212, %s3b3bt1v : tensor<256xf32>
    %v7214 = stablehlo.add %v7213, %v7211 : tensor<256xf32>
    %v7215 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7216 = stablehlo.multiply %v7215, %v7214 : tensor<256xf32>
    %v7217 = stablehlo.subtract %s3b3bt1, %v7216 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v2943) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v7218 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7219 = stablehlo.multiply %v7218, %s3b3W2 : tensor<256x256x3x3xf32>
    %v7220 = stablehlo.add %v7219, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v7221 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7222 = stablehlo.multiply %v7221, %s3b3W2v : tensor<256x256x3x3xf32>
    %v7223 = stablehlo.add %v7222, %v7220 : tensor<256x256x3x3xf32>
    %v7224 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7225 = stablehlo.multiply %v7224, %v7223 : tensor<256x256x3x3xf32>
    %v7226 = stablehlo.subtract %s3b3W2, %v7225 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v2961) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v7227 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7228 = stablehlo.multiply %v7227, %s3b3g2 : tensor<256xf32>
    %v7229 = stablehlo.add %v7228, %armeans3b3g2 : tensor<256xf32>
    %v7230 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7231 = stablehlo.multiply %v7230, %s3b3g2v : tensor<256xf32>
    %v7232 = stablehlo.add %v7231, %v7229 : tensor<256xf32>
    %v7233 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7234 = stablehlo.multiply %v7233, %v7232 : tensor<256xf32>
    %v7235 = stablehlo.subtract %s3b3g2, %v7234 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v2964) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v7236 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7237 = stablehlo.multiply %v7236, %s3b3bt2 : tensor<256xf32>
    %v7238 = stablehlo.add %v7237, %armeans3b3bt2 : tensor<256xf32>
    %v7239 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7240 = stablehlo.multiply %v7239, %s3b3bt2v : tensor<256xf32>
    %v7241 = stablehlo.add %v7240, %v7238 : tensor<256xf32>
    %v7242 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7243 = stablehlo.multiply %v7242, %v7241 : tensor<256xf32>
    %v7244 = stablehlo.subtract %s3b3bt2, %v7243 : tensor<256xf32>
    %arsums3b3W3 = "stablehlo.all_reduce"(%v2973) ({
    ^bb0(%aras3b3W3: tensor<f32>, %arbs3b3W3: tensor<f32>):
      %aradds3b3W3 = stablehlo.add %aras3b3W3, %arbs3b3W3 : tensor<f32>
      stablehlo.return %aradds3b3W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b3W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b3W3 = stablehlo.divide %arsums3b3W3, %arns3b3W3 : tensor<1024x256x1x1xf32>
    %v7245 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7246 = stablehlo.multiply %v7245, %s3b3W3 : tensor<1024x256x1x1xf32>
    %v7247 = stablehlo.add %v7246, %armeans3b3W3 : tensor<1024x256x1x1xf32>
    %v7248 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7249 = stablehlo.multiply %v7248, %s3b3W3v : tensor<1024x256x1x1xf32>
    %v7250 = stablehlo.add %v7249, %v7247 : tensor<1024x256x1x1xf32>
    %v7251 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7252 = stablehlo.multiply %v7251, %v7250 : tensor<1024x256x1x1xf32>
    %v7253 = stablehlo.subtract %s3b3W3, %v7252 : tensor<1024x256x1x1xf32>
    %arsums3b3g3 = "stablehlo.all_reduce"(%v2991) ({
    ^bb0(%aras3b3g3: tensor<f32>, %arbs3b3g3: tensor<f32>):
      %aradds3b3g3 = stablehlo.add %aras3b3g3, %arbs3b3g3 : tensor<f32>
      stablehlo.return %aradds3b3g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g3 = stablehlo.divide %arsums3b3g3, %arns3b3g3 : tensor<1024xf32>
    %v7254 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7255 = stablehlo.multiply %v7254, %s3b3g3 : tensor<1024xf32>
    %v7256 = stablehlo.add %v7255, %armeans3b3g3 : tensor<1024xf32>
    %v7257 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7258 = stablehlo.multiply %v7257, %s3b3g3v : tensor<1024xf32>
    %v7259 = stablehlo.add %v7258, %v7256 : tensor<1024xf32>
    %v7260 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7261 = stablehlo.multiply %v7260, %v7259 : tensor<1024xf32>
    %v7262 = stablehlo.subtract %s3b3g3, %v7261 : tensor<1024xf32>
    %arsums3b3bt3 = "stablehlo.all_reduce"(%v2994) ({
    ^bb0(%aras3b3bt3: tensor<f32>, %arbs3b3bt3: tensor<f32>):
      %aradds3b3bt3 = stablehlo.add %aras3b3bt3, %arbs3b3bt3 : tensor<f32>
      stablehlo.return %aradds3b3bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3bt3 = stablehlo.divide %arsums3b3bt3, %arns3b3bt3 : tensor<1024xf32>
    %v7263 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7264 = stablehlo.multiply %v7263, %s3b3bt3 : tensor<1024xf32>
    %v7265 = stablehlo.add %v7264, %armeans3b3bt3 : tensor<1024xf32>
    %v7266 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7267 = stablehlo.multiply %v7266, %s3b3bt3v : tensor<1024xf32>
    %v7268 = stablehlo.add %v7267, %v7265 : tensor<1024xf32>
    %v7269 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7270 = stablehlo.multiply %v7269, %v7268 : tensor<1024xf32>
    %v7271 = stablehlo.subtract %s3b3bt3, %v7270 : tensor<1024xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2699) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x1024x1x1xf32>
    %v7272 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7273 = stablehlo.multiply %v7272, %s3b4W1 : tensor<256x1024x1x1xf32>
    %v7274 = stablehlo.add %v7273, %armeans3b4W1 : tensor<256x1024x1x1xf32>
    %v7275 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7276 = stablehlo.multiply %v7275, %s3b4W1v : tensor<256x1024x1x1xf32>
    %v7277 = stablehlo.add %v7276, %v7274 : tensor<256x1024x1x1xf32>
    %v7278 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7279 = stablehlo.multiply %v7278, %v7277 : tensor<256x1024x1x1xf32>
    %v7280 = stablehlo.subtract %s3b4W1, %v7279 : tensor<256x1024x1x1xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2717) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v7281 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7282 = stablehlo.multiply %v7281, %s3b4g1 : tensor<256xf32>
    %v7283 = stablehlo.add %v7282, %armeans3b4g1 : tensor<256xf32>
    %v7284 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7285 = stablehlo.multiply %v7284, %s3b4g1v : tensor<256xf32>
    %v7286 = stablehlo.add %v7285, %v7283 : tensor<256xf32>
    %v7287 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7288 = stablehlo.multiply %v7287, %v7286 : tensor<256xf32>
    %v7289 = stablehlo.subtract %s3b4g1, %v7288 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2720) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v7290 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7291 = stablehlo.multiply %v7290, %s3b4bt1 : tensor<256xf32>
    %v7292 = stablehlo.add %v7291, %armeans3b4bt1 : tensor<256xf32>
    %v7293 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7294 = stablehlo.multiply %v7293, %s3b4bt1v : tensor<256xf32>
    %v7295 = stablehlo.add %v7294, %v7292 : tensor<256xf32>
    %v7296 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7297 = stablehlo.multiply %v7296, %v7295 : tensor<256xf32>
    %v7298 = stablehlo.subtract %s3b4bt1, %v7297 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2729) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v7299 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7300 = stablehlo.multiply %v7299, %s3b4W2 : tensor<256x256x3x3xf32>
    %v7301 = stablehlo.add %v7300, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v7302 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7303 = stablehlo.multiply %v7302, %s3b4W2v : tensor<256x256x3x3xf32>
    %v7304 = stablehlo.add %v7303, %v7301 : tensor<256x256x3x3xf32>
    %v7305 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7306 = stablehlo.multiply %v7305, %v7304 : tensor<256x256x3x3xf32>
    %v7307 = stablehlo.subtract %s3b4W2, %v7306 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2747) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v7308 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7309 = stablehlo.multiply %v7308, %s3b4g2 : tensor<256xf32>
    %v7310 = stablehlo.add %v7309, %armeans3b4g2 : tensor<256xf32>
    %v7311 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7312 = stablehlo.multiply %v7311, %s3b4g2v : tensor<256xf32>
    %v7313 = stablehlo.add %v7312, %v7310 : tensor<256xf32>
    %v7314 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7315 = stablehlo.multiply %v7314, %v7313 : tensor<256xf32>
    %v7316 = stablehlo.subtract %s3b4g2, %v7315 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2750) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v7317 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7318 = stablehlo.multiply %v7317, %s3b4bt2 : tensor<256xf32>
    %v7319 = stablehlo.add %v7318, %armeans3b4bt2 : tensor<256xf32>
    %v7320 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7321 = stablehlo.multiply %v7320, %s3b4bt2v : tensor<256xf32>
    %v7322 = stablehlo.add %v7321, %v7319 : tensor<256xf32>
    %v7323 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7324 = stablehlo.multiply %v7323, %v7322 : tensor<256xf32>
    %v7325 = stablehlo.subtract %s3b4bt2, %v7324 : tensor<256xf32>
    %arsums3b4W3 = "stablehlo.all_reduce"(%v2759) ({
    ^bb0(%aras3b4W3: tensor<f32>, %arbs3b4W3: tensor<f32>):
      %aradds3b4W3 = stablehlo.add %aras3b4W3, %arbs3b4W3 : tensor<f32>
      stablehlo.return %aradds3b4W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b4W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b4W3 = stablehlo.divide %arsums3b4W3, %arns3b4W3 : tensor<1024x256x1x1xf32>
    %v7326 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7327 = stablehlo.multiply %v7326, %s3b4W3 : tensor<1024x256x1x1xf32>
    %v7328 = stablehlo.add %v7327, %armeans3b4W3 : tensor<1024x256x1x1xf32>
    %v7329 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7330 = stablehlo.multiply %v7329, %s3b4W3v : tensor<1024x256x1x1xf32>
    %v7331 = stablehlo.add %v7330, %v7328 : tensor<1024x256x1x1xf32>
    %v7332 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7333 = stablehlo.multiply %v7332, %v7331 : tensor<1024x256x1x1xf32>
    %v7334 = stablehlo.subtract %s3b4W3, %v7333 : tensor<1024x256x1x1xf32>
    %arsums3b4g3 = "stablehlo.all_reduce"(%v2777) ({
    ^bb0(%aras3b4g3: tensor<f32>, %arbs3b4g3: tensor<f32>):
      %aradds3b4g3 = stablehlo.add %aras3b4g3, %arbs3b4g3 : tensor<f32>
      stablehlo.return %aradds3b4g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g3 = stablehlo.divide %arsums3b4g3, %arns3b4g3 : tensor<1024xf32>
    %v7335 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7336 = stablehlo.multiply %v7335, %s3b4g3 : tensor<1024xf32>
    %v7337 = stablehlo.add %v7336, %armeans3b4g3 : tensor<1024xf32>
    %v7338 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7339 = stablehlo.multiply %v7338, %s3b4g3v : tensor<1024xf32>
    %v7340 = stablehlo.add %v7339, %v7337 : tensor<1024xf32>
    %v7341 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7342 = stablehlo.multiply %v7341, %v7340 : tensor<1024xf32>
    %v7343 = stablehlo.subtract %s3b4g3, %v7342 : tensor<1024xf32>
    %arsums3b4bt3 = "stablehlo.all_reduce"(%v2780) ({
    ^bb0(%aras3b4bt3: tensor<f32>, %arbs3b4bt3: tensor<f32>):
      %aradds3b4bt3 = stablehlo.add %aras3b4bt3, %arbs3b4bt3 : tensor<f32>
      stablehlo.return %aradds3b4bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4bt3 = stablehlo.divide %arsums3b4bt3, %arns3b4bt3 : tensor<1024xf32>
    %v7344 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7345 = stablehlo.multiply %v7344, %s3b4bt3 : tensor<1024xf32>
    %v7346 = stablehlo.add %v7345, %armeans3b4bt3 : tensor<1024xf32>
    %v7347 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7348 = stablehlo.multiply %v7347, %s3b4bt3v : tensor<1024xf32>
    %v7349 = stablehlo.add %v7348, %v7346 : tensor<1024xf32>
    %v7350 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7351 = stablehlo.multiply %v7350, %v7349 : tensor<1024xf32>
    %v7352 = stablehlo.subtract %s3b4bt3, %v7351 : tensor<1024xf32>
    %arsums3b5W1 = "stablehlo.all_reduce"(%v2485) ({
    ^bb0(%aras3b5W1: tensor<f32>, %arbs3b5W1: tensor<f32>):
      %aradds3b5W1 = stablehlo.add %aras3b5W1, %arbs3b5W1 : tensor<f32>
      stablehlo.return %aradds3b5W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x1024x1x1xf32>) -> tensor<256x1024x1x1xf32>
    %arns3b5W1 = stablehlo.constant dense<4.0> : tensor<256x1024x1x1xf32>
    %armeans3b5W1 = stablehlo.divide %arsums3b5W1, %arns3b5W1 : tensor<256x1024x1x1xf32>
    %v7353 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7354 = stablehlo.multiply %v7353, %s3b5W1 : tensor<256x1024x1x1xf32>
    %v7355 = stablehlo.add %v7354, %armeans3b5W1 : tensor<256x1024x1x1xf32>
    %v7356 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7357 = stablehlo.multiply %v7356, %s3b5W1v : tensor<256x1024x1x1xf32>
    %v7358 = stablehlo.add %v7357, %v7355 : tensor<256x1024x1x1xf32>
    %v7359 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x1024x1x1xf32>
    %v7360 = stablehlo.multiply %v7359, %v7358 : tensor<256x1024x1x1xf32>
    %v7361 = stablehlo.subtract %s3b5W1, %v7360 : tensor<256x1024x1x1xf32>
    %arsums3b5g1 = "stablehlo.all_reduce"(%v2503) ({
    ^bb0(%aras3b5g1: tensor<f32>, %arbs3b5g1: tensor<f32>):
      %aradds3b5g1 = stablehlo.add %aras3b5g1, %arbs3b5g1 : tensor<f32>
      stablehlo.return %aradds3b5g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g1 = stablehlo.divide %arsums3b5g1, %arns3b5g1 : tensor<256xf32>
    %v7362 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7363 = stablehlo.multiply %v7362, %s3b5g1 : tensor<256xf32>
    %v7364 = stablehlo.add %v7363, %armeans3b5g1 : tensor<256xf32>
    %v7365 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7366 = stablehlo.multiply %v7365, %s3b5g1v : tensor<256xf32>
    %v7367 = stablehlo.add %v7366, %v7364 : tensor<256xf32>
    %v7368 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7369 = stablehlo.multiply %v7368, %v7367 : tensor<256xf32>
    %v7370 = stablehlo.subtract %s3b5g1, %v7369 : tensor<256xf32>
    %arsums3b5bt1 = "stablehlo.all_reduce"(%v2506) ({
    ^bb0(%aras3b5bt1: tensor<f32>, %arbs3b5bt1: tensor<f32>):
      %aradds3b5bt1 = stablehlo.add %aras3b5bt1, %arbs3b5bt1 : tensor<f32>
      stablehlo.return %aradds3b5bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt1 = stablehlo.divide %arsums3b5bt1, %arns3b5bt1 : tensor<256xf32>
    %v7371 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7372 = stablehlo.multiply %v7371, %s3b5bt1 : tensor<256xf32>
    %v7373 = stablehlo.add %v7372, %armeans3b5bt1 : tensor<256xf32>
    %v7374 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7375 = stablehlo.multiply %v7374, %s3b5bt1v : tensor<256xf32>
    %v7376 = stablehlo.add %v7375, %v7373 : tensor<256xf32>
    %v7377 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7378 = stablehlo.multiply %v7377, %v7376 : tensor<256xf32>
    %v7379 = stablehlo.subtract %s3b5bt1, %v7378 : tensor<256xf32>
    %arsums3b5W2 = "stablehlo.all_reduce"(%v2515) ({
    ^bb0(%aras3b5W2: tensor<f32>, %arbs3b5W2: tensor<f32>):
      %aradds3b5W2 = stablehlo.add %aras3b5W2, %arbs3b5W2 : tensor<f32>
      stablehlo.return %aradds3b5W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b5W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b5W2 = stablehlo.divide %arsums3b5W2, %arns3b5W2 : tensor<256x256x3x3xf32>
    %v7380 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7381 = stablehlo.multiply %v7380, %s3b5W2 : tensor<256x256x3x3xf32>
    %v7382 = stablehlo.add %v7381, %armeans3b5W2 : tensor<256x256x3x3xf32>
    %v7383 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7384 = stablehlo.multiply %v7383, %s3b5W2v : tensor<256x256x3x3xf32>
    %v7385 = stablehlo.add %v7384, %v7382 : tensor<256x256x3x3xf32>
    %v7386 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v7387 = stablehlo.multiply %v7386, %v7385 : tensor<256x256x3x3xf32>
    %v7388 = stablehlo.subtract %s3b5W2, %v7387 : tensor<256x256x3x3xf32>
    %arsums3b5g2 = "stablehlo.all_reduce"(%v2533) ({
    ^bb0(%aras3b5g2: tensor<f32>, %arbs3b5g2: tensor<f32>):
      %aradds3b5g2 = stablehlo.add %aras3b5g2, %arbs3b5g2 : tensor<f32>
      stablehlo.return %aradds3b5g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5g2 = stablehlo.divide %arsums3b5g2, %arns3b5g2 : tensor<256xf32>
    %v7389 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7390 = stablehlo.multiply %v7389, %s3b5g2 : tensor<256xf32>
    %v7391 = stablehlo.add %v7390, %armeans3b5g2 : tensor<256xf32>
    %v7392 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7393 = stablehlo.multiply %v7392, %s3b5g2v : tensor<256xf32>
    %v7394 = stablehlo.add %v7393, %v7391 : tensor<256xf32>
    %v7395 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7396 = stablehlo.multiply %v7395, %v7394 : tensor<256xf32>
    %v7397 = stablehlo.subtract %s3b5g2, %v7396 : tensor<256xf32>
    %arsums3b5bt2 = "stablehlo.all_reduce"(%v2536) ({
    ^bb0(%aras3b5bt2: tensor<f32>, %arbs3b5bt2: tensor<f32>):
      %aradds3b5bt2 = stablehlo.add %aras3b5bt2, %arbs3b5bt2 : tensor<f32>
      stablehlo.return %aradds3b5bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b5bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b5bt2 = stablehlo.divide %arsums3b5bt2, %arns3b5bt2 : tensor<256xf32>
    %v7398 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7399 = stablehlo.multiply %v7398, %s3b5bt2 : tensor<256xf32>
    %v7400 = stablehlo.add %v7399, %armeans3b5bt2 : tensor<256xf32>
    %v7401 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7402 = stablehlo.multiply %v7401, %s3b5bt2v : tensor<256xf32>
    %v7403 = stablehlo.add %v7402, %v7400 : tensor<256xf32>
    %v7404 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v7405 = stablehlo.multiply %v7404, %v7403 : tensor<256xf32>
    %v7406 = stablehlo.subtract %s3b5bt2, %v7405 : tensor<256xf32>
    %arsums3b5W3 = "stablehlo.all_reduce"(%v2545) ({
    ^bb0(%aras3b5W3: tensor<f32>, %arbs3b5W3: tensor<f32>):
      %aradds3b5W3 = stablehlo.add %aras3b5W3, %arbs3b5W3 : tensor<f32>
      stablehlo.return %aradds3b5W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024x256x1x1xf32>) -> tensor<1024x256x1x1xf32>
    %arns3b5W3 = stablehlo.constant dense<4.0> : tensor<1024x256x1x1xf32>
    %armeans3b5W3 = stablehlo.divide %arsums3b5W3, %arns3b5W3 : tensor<1024x256x1x1xf32>
    %v7407 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7408 = stablehlo.multiply %v7407, %s3b5W3 : tensor<1024x256x1x1xf32>
    %v7409 = stablehlo.add %v7408, %armeans3b5W3 : tensor<1024x256x1x1xf32>
    %v7410 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7411 = stablehlo.multiply %v7410, %s3b5W3v : tensor<1024x256x1x1xf32>
    %v7412 = stablehlo.add %v7411, %v7409 : tensor<1024x256x1x1xf32>
    %v7413 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024x256x1x1xf32>
    %v7414 = stablehlo.multiply %v7413, %v7412 : tensor<1024x256x1x1xf32>
    %v7415 = stablehlo.subtract %s3b5W3, %v7414 : tensor<1024x256x1x1xf32>
    %arsums3b5g3 = "stablehlo.all_reduce"(%v2563) ({
    ^bb0(%aras3b5g3: tensor<f32>, %arbs3b5g3: tensor<f32>):
      %aradds3b5g3 = stablehlo.add %aras3b5g3, %arbs3b5g3 : tensor<f32>
      stablehlo.return %aradds3b5g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5g3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5g3 = stablehlo.divide %arsums3b5g3, %arns3b5g3 : tensor<1024xf32>
    %v7416 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7417 = stablehlo.multiply %v7416, %s3b5g3 : tensor<1024xf32>
    %v7418 = stablehlo.add %v7417, %armeans3b5g3 : tensor<1024xf32>
    %v7419 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7420 = stablehlo.multiply %v7419, %s3b5g3v : tensor<1024xf32>
    %v7421 = stablehlo.add %v7420, %v7418 : tensor<1024xf32>
    %v7422 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7423 = stablehlo.multiply %v7422, %v7421 : tensor<1024xf32>
    %v7424 = stablehlo.subtract %s3b5g3, %v7423 : tensor<1024xf32>
    %arsums3b5bt3 = "stablehlo.all_reduce"(%v2566) ({
    ^bb0(%aras3b5bt3: tensor<f32>, %arbs3b5bt3: tensor<f32>):
      %aradds3b5bt3 = stablehlo.add %aras3b5bt3, %arbs3b5bt3 : tensor<f32>
      stablehlo.return %aradds3b5bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b5bt3 = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b5bt3 = stablehlo.divide %arsums3b5bt3, %arns3b5bt3 : tensor<1024xf32>
    %v7425 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7426 = stablehlo.multiply %v7425, %s3b5bt3 : tensor<1024xf32>
    %v7427 = stablehlo.add %v7426, %armeans3b5bt3 : tensor<1024xf32>
    %v7428 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7429 = stablehlo.multiply %v7428, %s3b5bt3v : tensor<1024xf32>
    %v7430 = stablehlo.add %v7429, %v7427 : tensor<1024xf32>
    %v7431 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1024xf32>
    %v7432 = stablehlo.multiply %v7431, %v7430 : tensor<1024xf32>
    %v7433 = stablehlo.subtract %s3b5bt3, %v7432 : tensor<1024xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v2237) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1024x1x1xf32>) -> tensor<512x1024x1x1xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x1024x1x1xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x1024x1x1xf32>
    %v7434 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7435 = stablehlo.multiply %v7434, %s4b0W1 : tensor<512x1024x1x1xf32>
    %v7436 = stablehlo.add %v7435, %armeans4b0W1 : tensor<512x1024x1x1xf32>
    %v7437 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7438 = stablehlo.multiply %v7437, %s4b0W1v : tensor<512x1024x1x1xf32>
    %v7439 = stablehlo.add %v7438, %v7436 : tensor<512x1024x1x1xf32>
    %v7440 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1024x1x1xf32>
    %v7441 = stablehlo.multiply %v7440, %v7439 : tensor<512x1024x1x1xf32>
    %v7442 = stablehlo.subtract %s4b0W1, %v7441 : tensor<512x1024x1x1xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v2255) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v7443 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7444 = stablehlo.multiply %v7443, %s4b0g1 : tensor<512xf32>
    %v7445 = stablehlo.add %v7444, %armeans4b0g1 : tensor<512xf32>
    %v7446 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7447 = stablehlo.multiply %v7446, %s4b0g1v : tensor<512xf32>
    %v7448 = stablehlo.add %v7447, %v7445 : tensor<512xf32>
    %v7449 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7450 = stablehlo.multiply %v7449, %v7448 : tensor<512xf32>
    %v7451 = stablehlo.subtract %s4b0g1, %v7450 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v2258) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v7452 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7453 = stablehlo.multiply %v7452, %s4b0bt1 : tensor<512xf32>
    %v7454 = stablehlo.add %v7453, %armeans4b0bt1 : tensor<512xf32>
    %v7455 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7456 = stablehlo.multiply %v7455, %s4b0bt1v : tensor<512xf32>
    %v7457 = stablehlo.add %v7456, %v7454 : tensor<512xf32>
    %v7458 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7459 = stablehlo.multiply %v7458, %v7457 : tensor<512xf32>
    %v7460 = stablehlo.subtract %s4b0bt1, %v7459 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v2269) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v7461 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7462 = stablehlo.multiply %v7461, %s4b0W2 : tensor<512x512x3x3xf32>
    %v7463 = stablehlo.add %v7462, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v7464 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7465 = stablehlo.multiply %v7464, %s4b0W2v : tensor<512x512x3x3xf32>
    %v7466 = stablehlo.add %v7465, %v7463 : tensor<512x512x3x3xf32>
    %v7467 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7468 = stablehlo.multiply %v7467, %v7466 : tensor<512x512x3x3xf32>
    %v7469 = stablehlo.subtract %s4b0W2, %v7468 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v2287) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v7470 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7471 = stablehlo.multiply %v7470, %s4b0g2 : tensor<512xf32>
    %v7472 = stablehlo.add %v7471, %armeans4b0g2 : tensor<512xf32>
    %v7473 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7474 = stablehlo.multiply %v7473, %s4b0g2v : tensor<512xf32>
    %v7475 = stablehlo.add %v7474, %v7472 : tensor<512xf32>
    %v7476 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7477 = stablehlo.multiply %v7476, %v7475 : tensor<512xf32>
    %v7478 = stablehlo.subtract %s4b0g2, %v7477 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v2290) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v7479 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7480 = stablehlo.multiply %v7479, %s4b0bt2 : tensor<512xf32>
    %v7481 = stablehlo.add %v7480, %armeans4b0bt2 : tensor<512xf32>
    %v7482 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7483 = stablehlo.multiply %v7482, %s4b0bt2v : tensor<512xf32>
    %v7484 = stablehlo.add %v7483, %v7481 : tensor<512xf32>
    %v7485 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7486 = stablehlo.multiply %v7485, %v7484 : tensor<512xf32>
    %v7487 = stablehlo.subtract %s4b0bt2, %v7486 : tensor<512xf32>
    %arsums4b0W3 = "stablehlo.all_reduce"(%v2299) ({
    ^bb0(%aras4b0W3: tensor<f32>, %arbs4b0W3: tensor<f32>):
      %aradds4b0W3 = stablehlo.add %aras4b0W3, %arbs4b0W3 : tensor<f32>
      stablehlo.return %aradds4b0W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b0W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b0W3 = stablehlo.divide %arsums4b0W3, %arns4b0W3 : tensor<2048x512x1x1xf32>
    %v7488 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7489 = stablehlo.multiply %v7488, %s4b0W3 : tensor<2048x512x1x1xf32>
    %v7490 = stablehlo.add %v7489, %armeans4b0W3 : tensor<2048x512x1x1xf32>
    %v7491 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7492 = stablehlo.multiply %v7491, %s4b0W3v : tensor<2048x512x1x1xf32>
    %v7493 = stablehlo.add %v7492, %v7490 : tensor<2048x512x1x1xf32>
    %v7494 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7495 = stablehlo.multiply %v7494, %v7493 : tensor<2048x512x1x1xf32>
    %v7496 = stablehlo.subtract %s4b0W3, %v7495 : tensor<2048x512x1x1xf32>
    %arsums4b0g3 = "stablehlo.all_reduce"(%v2317) ({
    ^bb0(%aras4b0g3: tensor<f32>, %arbs4b0g3: tensor<f32>):
      %aradds4b0g3 = stablehlo.add %aras4b0g3, %arbs4b0g3 : tensor<f32>
      stablehlo.return %aradds4b0g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g3 = stablehlo.divide %arsums4b0g3, %arns4b0g3 : tensor<2048xf32>
    %v7497 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7498 = stablehlo.multiply %v7497, %s4b0g3 : tensor<2048xf32>
    %v7499 = stablehlo.add %v7498, %armeans4b0g3 : tensor<2048xf32>
    %v7500 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7501 = stablehlo.multiply %v7500, %s4b0g3v : tensor<2048xf32>
    %v7502 = stablehlo.add %v7501, %v7499 : tensor<2048xf32>
    %v7503 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7504 = stablehlo.multiply %v7503, %v7502 : tensor<2048xf32>
    %v7505 = stablehlo.subtract %s4b0g3, %v7504 : tensor<2048xf32>
    %arsums4b0bt3 = "stablehlo.all_reduce"(%v2320) ({
    ^bb0(%aras4b0bt3: tensor<f32>, %arbs4b0bt3: tensor<f32>):
      %aradds4b0bt3 = stablehlo.add %aras4b0bt3, %arbs4b0bt3 : tensor<f32>
      stablehlo.return %aradds4b0bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0bt3 = stablehlo.divide %arsums4b0bt3, %arns4b0bt3 : tensor<2048xf32>
    %v7506 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7507 = stablehlo.multiply %v7506, %s4b0bt3 : tensor<2048xf32>
    %v7508 = stablehlo.add %v7507, %armeans4b0bt3 : tensor<2048xf32>
    %v7509 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7510 = stablehlo.multiply %v7509, %s4b0bt3v : tensor<2048xf32>
    %v7511 = stablehlo.add %v7510, %v7508 : tensor<2048xf32>
    %v7512 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7513 = stablehlo.multiply %v7512, %v7511 : tensor<2048xf32>
    %v7514 = stablehlo.subtract %s4b0bt3, %v7513 : tensor<2048xf32>
    %arsums4b0Wp = "stablehlo.all_reduce"(%v2331) ({
    ^bb0(%aras4b0Wp: tensor<f32>, %arbs4b0Wp: tensor<f32>):
      %aradds4b0Wp = stablehlo.add %aras4b0Wp, %arbs4b0Wp : tensor<f32>
      stablehlo.return %aradds4b0Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1024x1x1xf32>) -> tensor<2048x1024x1x1xf32>
    %arns4b0Wp = stablehlo.constant dense<4.0> : tensor<2048x1024x1x1xf32>
    %armeans4b0Wp = stablehlo.divide %arsums4b0Wp, %arns4b0Wp : tensor<2048x1024x1x1xf32>
    %v7515 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7516 = stablehlo.multiply %v7515, %s4b0Wp : tensor<2048x1024x1x1xf32>
    %v7517 = stablehlo.add %v7516, %armeans4b0Wp : tensor<2048x1024x1x1xf32>
    %v7518 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7519 = stablehlo.multiply %v7518, %s4b0Wpv : tensor<2048x1024x1x1xf32>
    %v7520 = stablehlo.add %v7519, %v7517 : tensor<2048x1024x1x1xf32>
    %v7521 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1024x1x1xf32>
    %v7522 = stablehlo.multiply %v7521, %v7520 : tensor<2048x1024x1x1xf32>
    %v7523 = stablehlo.subtract %s4b0Wp, %v7522 : tensor<2048x1024x1x1xf32>
    %arsums4b0gp = "stablehlo.all_reduce"(%v2349) ({
    ^bb0(%aras4b0gp: tensor<f32>, %arbs4b0gp: tensor<f32>):
      %aradds4b0gp = stablehlo.add %aras4b0gp, %arbs4b0gp : tensor<f32>
      stablehlo.return %aradds4b0gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0gp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0gp = stablehlo.divide %arsums4b0gp, %arns4b0gp : tensor<2048xf32>
    %v7524 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7525 = stablehlo.multiply %v7524, %s4b0gp : tensor<2048xf32>
    %v7526 = stablehlo.add %v7525, %armeans4b0gp : tensor<2048xf32>
    %v7527 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7528 = stablehlo.multiply %v7527, %s4b0gpv : tensor<2048xf32>
    %v7529 = stablehlo.add %v7528, %v7526 : tensor<2048xf32>
    %v7530 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7531 = stablehlo.multiply %v7530, %v7529 : tensor<2048xf32>
    %v7532 = stablehlo.subtract %s4b0gp, %v7531 : tensor<2048xf32>
    %arsums4b0btp = "stablehlo.all_reduce"(%v2352) ({
    ^bb0(%aras4b0btp: tensor<f32>, %arbs4b0btp: tensor<f32>):
      %aradds4b0btp = stablehlo.add %aras4b0btp, %arbs4b0btp : tensor<f32>
      stablehlo.return %aradds4b0btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0btp = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0btp = stablehlo.divide %arsums4b0btp, %arns4b0btp : tensor<2048xf32>
    %v7533 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7534 = stablehlo.multiply %v7533, %s4b0btp : tensor<2048xf32>
    %v7535 = stablehlo.add %v7534, %armeans4b0btp : tensor<2048xf32>
    %v7536 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7537 = stablehlo.multiply %v7536, %s4b0btpv : tensor<2048xf32>
    %v7538 = stablehlo.add %v7537, %v7535 : tensor<2048xf32>
    %v7539 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7540 = stablehlo.multiply %v7539, %v7538 : tensor<2048xf32>
    %v7541 = stablehlo.subtract %s4b0btp, %v7540 : tensor<2048xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1981) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x2048x1x1xf32>
    %v7542 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7543 = stablehlo.multiply %v7542, %s4b1W1 : tensor<512x2048x1x1xf32>
    %v7544 = stablehlo.add %v7543, %armeans4b1W1 : tensor<512x2048x1x1xf32>
    %v7545 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7546 = stablehlo.multiply %v7545, %s4b1W1v : tensor<512x2048x1x1xf32>
    %v7547 = stablehlo.add %v7546, %v7544 : tensor<512x2048x1x1xf32>
    %v7548 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7549 = stablehlo.multiply %v7548, %v7547 : tensor<512x2048x1x1xf32>
    %v7550 = stablehlo.subtract %s4b1W1, %v7549 : tensor<512x2048x1x1xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1999) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v7551 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7552 = stablehlo.multiply %v7551, %s4b1g1 : tensor<512xf32>
    %v7553 = stablehlo.add %v7552, %armeans4b1g1 : tensor<512xf32>
    %v7554 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7555 = stablehlo.multiply %v7554, %s4b1g1v : tensor<512xf32>
    %v7556 = stablehlo.add %v7555, %v7553 : tensor<512xf32>
    %v7557 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7558 = stablehlo.multiply %v7557, %v7556 : tensor<512xf32>
    %v7559 = stablehlo.subtract %s4b1g1, %v7558 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v2002) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v7560 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7561 = stablehlo.multiply %v7560, %s4b1bt1 : tensor<512xf32>
    %v7562 = stablehlo.add %v7561, %armeans4b1bt1 : tensor<512xf32>
    %v7563 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7564 = stablehlo.multiply %v7563, %s4b1bt1v : tensor<512xf32>
    %v7565 = stablehlo.add %v7564, %v7562 : tensor<512xf32>
    %v7566 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7567 = stablehlo.multiply %v7566, %v7565 : tensor<512xf32>
    %v7568 = stablehlo.subtract %s4b1bt1, %v7567 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v2011) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v7569 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7570 = stablehlo.multiply %v7569, %s4b1W2 : tensor<512x512x3x3xf32>
    %v7571 = stablehlo.add %v7570, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v7572 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7573 = stablehlo.multiply %v7572, %s4b1W2v : tensor<512x512x3x3xf32>
    %v7574 = stablehlo.add %v7573, %v7571 : tensor<512x512x3x3xf32>
    %v7575 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7576 = stablehlo.multiply %v7575, %v7574 : tensor<512x512x3x3xf32>
    %v7577 = stablehlo.subtract %s4b1W2, %v7576 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v2029) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v7578 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7579 = stablehlo.multiply %v7578, %s4b1g2 : tensor<512xf32>
    %v7580 = stablehlo.add %v7579, %armeans4b1g2 : tensor<512xf32>
    %v7581 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7582 = stablehlo.multiply %v7581, %s4b1g2v : tensor<512xf32>
    %v7583 = stablehlo.add %v7582, %v7580 : tensor<512xf32>
    %v7584 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7585 = stablehlo.multiply %v7584, %v7583 : tensor<512xf32>
    %v7586 = stablehlo.subtract %s4b1g2, %v7585 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v2032) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v7587 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7588 = stablehlo.multiply %v7587, %s4b1bt2 : tensor<512xf32>
    %v7589 = stablehlo.add %v7588, %armeans4b1bt2 : tensor<512xf32>
    %v7590 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7591 = stablehlo.multiply %v7590, %s4b1bt2v : tensor<512xf32>
    %v7592 = stablehlo.add %v7591, %v7589 : tensor<512xf32>
    %v7593 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7594 = stablehlo.multiply %v7593, %v7592 : tensor<512xf32>
    %v7595 = stablehlo.subtract %s4b1bt2, %v7594 : tensor<512xf32>
    %arsums4b1W3 = "stablehlo.all_reduce"(%v2041) ({
    ^bb0(%aras4b1W3: tensor<f32>, %arbs4b1W3: tensor<f32>):
      %aradds4b1W3 = stablehlo.add %aras4b1W3, %arbs4b1W3 : tensor<f32>
      stablehlo.return %aradds4b1W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b1W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b1W3 = stablehlo.divide %arsums4b1W3, %arns4b1W3 : tensor<2048x512x1x1xf32>
    %v7596 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7597 = stablehlo.multiply %v7596, %s4b1W3 : tensor<2048x512x1x1xf32>
    %v7598 = stablehlo.add %v7597, %armeans4b1W3 : tensor<2048x512x1x1xf32>
    %v7599 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7600 = stablehlo.multiply %v7599, %s4b1W3v : tensor<2048x512x1x1xf32>
    %v7601 = stablehlo.add %v7600, %v7598 : tensor<2048x512x1x1xf32>
    %v7602 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7603 = stablehlo.multiply %v7602, %v7601 : tensor<2048x512x1x1xf32>
    %v7604 = stablehlo.subtract %s4b1W3, %v7603 : tensor<2048x512x1x1xf32>
    %arsums4b1g3 = "stablehlo.all_reduce"(%v2059) ({
    ^bb0(%aras4b1g3: tensor<f32>, %arbs4b1g3: tensor<f32>):
      %aradds4b1g3 = stablehlo.add %aras4b1g3, %arbs4b1g3 : tensor<f32>
      stablehlo.return %aradds4b1g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g3 = stablehlo.divide %arsums4b1g3, %arns4b1g3 : tensor<2048xf32>
    %v7605 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7606 = stablehlo.multiply %v7605, %s4b1g3 : tensor<2048xf32>
    %v7607 = stablehlo.add %v7606, %armeans4b1g3 : tensor<2048xf32>
    %v7608 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7609 = stablehlo.multiply %v7608, %s4b1g3v : tensor<2048xf32>
    %v7610 = stablehlo.add %v7609, %v7607 : tensor<2048xf32>
    %v7611 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7612 = stablehlo.multiply %v7611, %v7610 : tensor<2048xf32>
    %v7613 = stablehlo.subtract %s4b1g3, %v7612 : tensor<2048xf32>
    %arsums4b1bt3 = "stablehlo.all_reduce"(%v2062) ({
    ^bb0(%aras4b1bt3: tensor<f32>, %arbs4b1bt3: tensor<f32>):
      %aradds4b1bt3 = stablehlo.add %aras4b1bt3, %arbs4b1bt3 : tensor<f32>
      stablehlo.return %aradds4b1bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1bt3 = stablehlo.divide %arsums4b1bt3, %arns4b1bt3 : tensor<2048xf32>
    %v7614 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7615 = stablehlo.multiply %v7614, %s4b1bt3 : tensor<2048xf32>
    %v7616 = stablehlo.add %v7615, %armeans4b1bt3 : tensor<2048xf32>
    %v7617 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7618 = stablehlo.multiply %v7617, %s4b1bt3v : tensor<2048xf32>
    %v7619 = stablehlo.add %v7618, %v7616 : tensor<2048xf32>
    %v7620 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7621 = stablehlo.multiply %v7620, %v7619 : tensor<2048xf32>
    %v7622 = stablehlo.subtract %s4b1bt3, %v7621 : tensor<2048xf32>
    %arsums4b2W1 = "stablehlo.all_reduce"(%v1767) ({
    ^bb0(%aras4b2W1: tensor<f32>, %arbs4b2W1: tensor<f32>):
      %aradds4b2W1 = stablehlo.add %aras4b2W1, %arbs4b2W1 : tensor<f32>
      stablehlo.return %aradds4b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x2048x1x1xf32>) -> tensor<512x2048x1x1xf32>
    %arns4b2W1 = stablehlo.constant dense<4.0> : tensor<512x2048x1x1xf32>
    %armeans4b2W1 = stablehlo.divide %arsums4b2W1, %arns4b2W1 : tensor<512x2048x1x1xf32>
    %v7623 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7624 = stablehlo.multiply %v7623, %s4b2W1 : tensor<512x2048x1x1xf32>
    %v7625 = stablehlo.add %v7624, %armeans4b2W1 : tensor<512x2048x1x1xf32>
    %v7626 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7627 = stablehlo.multiply %v7626, %s4b2W1v : tensor<512x2048x1x1xf32>
    %v7628 = stablehlo.add %v7627, %v7625 : tensor<512x2048x1x1xf32>
    %v7629 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x2048x1x1xf32>
    %v7630 = stablehlo.multiply %v7629, %v7628 : tensor<512x2048x1x1xf32>
    %v7631 = stablehlo.subtract %s4b2W1, %v7630 : tensor<512x2048x1x1xf32>
    %arsums4b2g1 = "stablehlo.all_reduce"(%v1785) ({
    ^bb0(%aras4b2g1: tensor<f32>, %arbs4b2g1: tensor<f32>):
      %aradds4b2g1 = stablehlo.add %aras4b2g1, %arbs4b2g1 : tensor<f32>
      stablehlo.return %aradds4b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g1 = stablehlo.divide %arsums4b2g1, %arns4b2g1 : tensor<512xf32>
    %v7632 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7633 = stablehlo.multiply %v7632, %s4b2g1 : tensor<512xf32>
    %v7634 = stablehlo.add %v7633, %armeans4b2g1 : tensor<512xf32>
    %v7635 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7636 = stablehlo.multiply %v7635, %s4b2g1v : tensor<512xf32>
    %v7637 = stablehlo.add %v7636, %v7634 : tensor<512xf32>
    %v7638 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7639 = stablehlo.multiply %v7638, %v7637 : tensor<512xf32>
    %v7640 = stablehlo.subtract %s4b2g1, %v7639 : tensor<512xf32>
    %arsums4b2bt1 = "stablehlo.all_reduce"(%v1788) ({
    ^bb0(%aras4b2bt1: tensor<f32>, %arbs4b2bt1: tensor<f32>):
      %aradds4b2bt1 = stablehlo.add %aras4b2bt1, %arbs4b2bt1 : tensor<f32>
      stablehlo.return %aradds4b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt1 = stablehlo.divide %arsums4b2bt1, %arns4b2bt1 : tensor<512xf32>
    %v7641 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7642 = stablehlo.multiply %v7641, %s4b2bt1 : tensor<512xf32>
    %v7643 = stablehlo.add %v7642, %armeans4b2bt1 : tensor<512xf32>
    %v7644 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7645 = stablehlo.multiply %v7644, %s4b2bt1v : tensor<512xf32>
    %v7646 = stablehlo.add %v7645, %v7643 : tensor<512xf32>
    %v7647 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7648 = stablehlo.multiply %v7647, %v7646 : tensor<512xf32>
    %v7649 = stablehlo.subtract %s4b2bt1, %v7648 : tensor<512xf32>
    %arsums4b2W2 = "stablehlo.all_reduce"(%v1797) ({
    ^bb0(%aras4b2W2: tensor<f32>, %arbs4b2W2: tensor<f32>):
      %aradds4b2W2 = stablehlo.add %aras4b2W2, %arbs4b2W2 : tensor<f32>
      stablehlo.return %aradds4b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b2W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b2W2 = stablehlo.divide %arsums4b2W2, %arns4b2W2 : tensor<512x512x3x3xf32>
    %v7650 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7651 = stablehlo.multiply %v7650, %s4b2W2 : tensor<512x512x3x3xf32>
    %v7652 = stablehlo.add %v7651, %armeans4b2W2 : tensor<512x512x3x3xf32>
    %v7653 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7654 = stablehlo.multiply %v7653, %s4b2W2v : tensor<512x512x3x3xf32>
    %v7655 = stablehlo.add %v7654, %v7652 : tensor<512x512x3x3xf32>
    %v7656 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v7657 = stablehlo.multiply %v7656, %v7655 : tensor<512x512x3x3xf32>
    %v7658 = stablehlo.subtract %s4b2W2, %v7657 : tensor<512x512x3x3xf32>
    %arsums4b2g2 = "stablehlo.all_reduce"(%v1815) ({
    ^bb0(%aras4b2g2: tensor<f32>, %arbs4b2g2: tensor<f32>):
      %aradds4b2g2 = stablehlo.add %aras4b2g2, %arbs4b2g2 : tensor<f32>
      stablehlo.return %aradds4b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2g2 = stablehlo.divide %arsums4b2g2, %arns4b2g2 : tensor<512xf32>
    %v7659 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7660 = stablehlo.multiply %v7659, %s4b2g2 : tensor<512xf32>
    %v7661 = stablehlo.add %v7660, %armeans4b2g2 : tensor<512xf32>
    %v7662 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7663 = stablehlo.multiply %v7662, %s4b2g2v : tensor<512xf32>
    %v7664 = stablehlo.add %v7663, %v7661 : tensor<512xf32>
    %v7665 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7666 = stablehlo.multiply %v7665, %v7664 : tensor<512xf32>
    %v7667 = stablehlo.subtract %s4b2g2, %v7666 : tensor<512xf32>
    %arsums4b2bt2 = "stablehlo.all_reduce"(%v1818) ({
    ^bb0(%aras4b2bt2: tensor<f32>, %arbs4b2bt2: tensor<f32>):
      %aradds4b2bt2 = stablehlo.add %aras4b2bt2, %arbs4b2bt2 : tensor<f32>
      stablehlo.return %aradds4b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b2bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b2bt2 = stablehlo.divide %arsums4b2bt2, %arns4b2bt2 : tensor<512xf32>
    %v7668 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7669 = stablehlo.multiply %v7668, %s4b2bt2 : tensor<512xf32>
    %v7670 = stablehlo.add %v7669, %armeans4b2bt2 : tensor<512xf32>
    %v7671 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7672 = stablehlo.multiply %v7671, %s4b2bt2v : tensor<512xf32>
    %v7673 = stablehlo.add %v7672, %v7670 : tensor<512xf32>
    %v7674 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v7675 = stablehlo.multiply %v7674, %v7673 : tensor<512xf32>
    %v7676 = stablehlo.subtract %s4b2bt2, %v7675 : tensor<512xf32>
    %arsums4b2W3 = "stablehlo.all_reduce"(%v1827) ({
    ^bb0(%aras4b2W3: tensor<f32>, %arbs4b2W3: tensor<f32>):
      %aradds4b2W3 = stablehlo.add %aras4b2W3, %arbs4b2W3 : tensor<f32>
      stablehlo.return %aradds4b2W3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x512x1x1xf32>) -> tensor<2048x512x1x1xf32>
    %arns4b2W3 = stablehlo.constant dense<4.0> : tensor<2048x512x1x1xf32>
    %armeans4b2W3 = stablehlo.divide %arsums4b2W3, %arns4b2W3 : tensor<2048x512x1x1xf32>
    %v7677 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7678 = stablehlo.multiply %v7677, %s4b2W3 : tensor<2048x512x1x1xf32>
    %v7679 = stablehlo.add %v7678, %armeans4b2W3 : tensor<2048x512x1x1xf32>
    %v7680 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7681 = stablehlo.multiply %v7680, %s4b2W3v : tensor<2048x512x1x1xf32>
    %v7682 = stablehlo.add %v7681, %v7679 : tensor<2048x512x1x1xf32>
    %v7683 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x512x1x1xf32>
    %v7684 = stablehlo.multiply %v7683, %v7682 : tensor<2048x512x1x1xf32>
    %v7685 = stablehlo.subtract %s4b2W3, %v7684 : tensor<2048x512x1x1xf32>
    %arsums4b2g3 = "stablehlo.all_reduce"(%v1845) ({
    ^bb0(%aras4b2g3: tensor<f32>, %arbs4b2g3: tensor<f32>):
      %aradds4b2g3 = stablehlo.add %aras4b2g3, %arbs4b2g3 : tensor<f32>
      stablehlo.return %aradds4b2g3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2g3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2g3 = stablehlo.divide %arsums4b2g3, %arns4b2g3 : tensor<2048xf32>
    %v7686 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7687 = stablehlo.multiply %v7686, %s4b2g3 : tensor<2048xf32>
    %v7688 = stablehlo.add %v7687, %armeans4b2g3 : tensor<2048xf32>
    %v7689 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7690 = stablehlo.multiply %v7689, %s4b2g3v : tensor<2048xf32>
    %v7691 = stablehlo.add %v7690, %v7688 : tensor<2048xf32>
    %v7692 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7693 = stablehlo.multiply %v7692, %v7691 : tensor<2048xf32>
    %v7694 = stablehlo.subtract %s4b2g3, %v7693 : tensor<2048xf32>
    %arsums4b2bt3 = "stablehlo.all_reduce"(%v1848) ({
    ^bb0(%aras4b2bt3: tensor<f32>, %arbs4b2bt3: tensor<f32>):
      %aradds4b2bt3 = stablehlo.add %aras4b2bt3, %arbs4b2bt3 : tensor<f32>
      stablehlo.return %aradds4b2bt3 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b2bt3 = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b2bt3 = stablehlo.divide %arsums4b2bt3, %arns4b2bt3 : tensor<2048xf32>
    %v7695 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7696 = stablehlo.multiply %v7695, %s4b2bt3 : tensor<2048xf32>
    %v7697 = stablehlo.add %v7696, %armeans4b2bt3 : tensor<2048xf32>
    %v7698 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7699 = stablehlo.multiply %v7698, %s4b2bt3v : tensor<2048xf32>
    %v7700 = stablehlo.add %v7699, %v7697 : tensor<2048xf32>
    %v7701 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048xf32>
    %v7702 = stablehlo.multiply %v7701, %v7700 : tensor<2048xf32>
    %v7703 = stablehlo.subtract %s4b2bt3, %v7702 : tensor<2048xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1628) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048x1000xf32>) -> tensor<2048x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<2048x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<2048x1000xf32>
    %v7704 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7705 = stablehlo.multiply %v7704, %Wd : tensor<2048x1000xf32>
    %v7706 = stablehlo.add %v7705, %armeanWd : tensor<2048x1000xf32>
    %v7707 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7708 = stablehlo.multiply %v7707, %Wdv : tensor<2048x1000xf32>
    %v7709 = stablehlo.add %v7708, %v7706 : tensor<2048x1000xf32>
    %v7710 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<2048x1000xf32>
    %v7711 = stablehlo.multiply %v7710, %v7709 : tensor<2048x1000xf32>
    %v7712 = stablehlo.subtract %Wd, %v7711 : tensor<2048x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1630) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v7713 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7714 = stablehlo.multiply %v7713, %bd : tensor<1000xf32>
    %v7715 = stablehlo.add %v7714, %armeanbd : tensor<1000xf32>
    %v7716 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7717 = stablehlo.multiply %v7716, %bdv : tensor<1000xf32>
    %v7718 = stablehlo.add %v7717, %v7715 : tensor<1000xf32>
    %v7719 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v7720 = stablehlo.multiply %v7719, %v7718 : tensor<1000xf32>
    %v7721 = stablehlo.subtract %bd, %v7720 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1616 : tensor<64x1000xf32>
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
    return %v6281, %v6290, %v6299, %v6308, %v6317, %v6326, %v6335, %v6344, %v6353, %v6362, %v6371, %v6380, %v6389, %v6398, %v6407, %v6416, %v6425, %v6434, %v6443, %v6452, %v6461, %v6470, %v6479, %v6488, %v6497, %v6506, %v6515, %v6524, %v6533, %v6542, %v6551, %v6560, %v6569, %v6578, %v6587, %v6596, %v6605, %v6614, %v6623, %v6632, %v6641, %v6650, %v6659, %v6668, %v6677, %v6686, %v6695, %v6704, %v6713, %v6722, %v6731, %v6740, %v6749, %v6758, %v6767, %v6776, %v6785, %v6794, %v6803, %v6812, %v6821, %v6830, %v6839, %v6848, %v6857, %v6866, %v6875, %v6884, %v6893, %v6902, %v6911, %v6920, %v6929, %v6938, %v6947, %v6956, %v6965, %v6974, %v6983, %v6992, %v7001, %v7010, %v7019, %v7028, %v7037, %v7046, %v7055, %v7064, %v7073, %v7082, %v7091, %v7100, %v7109, %v7118, %v7127, %v7136, %v7145, %v7154, %v7163, %v7172, %v7181, %v7190, %v7199, %v7208, %v7217, %v7226, %v7235, %v7244, %v7253, %v7262, %v7271, %v7280, %v7289, %v7298, %v7307, %v7316, %v7325, %v7334, %v7343, %v7352, %v7361, %v7370, %v7379, %v7388, %v7397, %v7406, %v7415, %v7424, %v7433, %v7442, %v7451, %v7460, %v7469, %v7478, %v7487, %v7496, %v7505, %v7514, %v7523, %v7532, %v7541, %v7550, %v7559, %v7568, %v7577, %v7586, %v7595, %v7604, %v7613, %v7622, %v7631, %v7640, %v7649, %v7658, %v7667, %v7676, %v7685, %v7694, %v7703, %v7712, %v7721, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b0W3m, %s1b0g3m, %s1b0bt3m, %s1b0Wpm, %s1b0gpm, %s1b0btpm, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b1W3m, %s1b1g3m, %s1b1bt3m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %s1b2W3m, %s1b2g3m, %s1b2bt3m, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b0W3m, %s2b0g3m, %s2b0bt3m, %s2b0Wpm, %s2b0gpm, %s2b0btpm, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b1W3m, %s2b1g3m, %s2b1bt3m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %s2b2W3m, %s2b2g3m, %s2b2bt3m, %s2b3W1m, %s2b3g1m, %s2b3bt1m, %s2b3W2m, %s2b3g2m, %s2b3bt2m, %s2b3W3m, %s2b3g3m, %s2b3bt3m, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b0W3m, %s3b0g3m, %s3b0bt3m, %s3b0Wpm, %s3b0gpm, %s3b0btpm, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b1W3m, %s3b1g3m, %s3b1bt3m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b2W3m, %s3b2g3m, %s3b2bt3m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b3W3m, %s3b3g3m, %s3b3bt3m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %s3b4W3m, %s3b4g3m, %s3b4bt3m, %s3b5W1m, %s3b5g1m, %s3b5bt1m, %s3b5W2m, %s3b5g2m, %s3b5bt2m, %s3b5W3m, %s3b5g3m, %s3b5bt3m, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b0W3m, %s4b0g3m, %s4b0bt3m, %s4b0Wpm, %s4b0gpm, %s4b0btpm, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %s4b1W3m, %s4b1g3m, %s4b1bt3m, %s4b2W1m, %s4b2g1m, %s4b2bt1m, %s4b2W2m, %s4b2g2m, %s4b2bt2m, %s4b2W3m, %s4b2g3m, %s4b2bt3m, %Wdm, %bdm, %v6278, %v6287, %v6296, %v6305, %v6314, %v6323, %v6332, %v6341, %v6350, %v6359, %v6368, %v6377, %v6386, %v6395, %v6404, %v6413, %v6422, %v6431, %v6440, %v6449, %v6458, %v6467, %v6476, %v6485, %v6494, %v6503, %v6512, %v6521, %v6530, %v6539, %v6548, %v6557, %v6566, %v6575, %v6584, %v6593, %v6602, %v6611, %v6620, %v6629, %v6638, %v6647, %v6656, %v6665, %v6674, %v6683, %v6692, %v6701, %v6710, %v6719, %v6728, %v6737, %v6746, %v6755, %v6764, %v6773, %v6782, %v6791, %v6800, %v6809, %v6818, %v6827, %v6836, %v6845, %v6854, %v6863, %v6872, %v6881, %v6890, %v6899, %v6908, %v6917, %v6926, %v6935, %v6944, %v6953, %v6962, %v6971, %v6980, %v6989, %v6998, %v7007, %v7016, %v7025, %v7034, %v7043, %v7052, %v7061, %v7070, %v7079, %v7088, %v7097, %v7106, %v7115, %v7124, %v7133, %v7142, %v7151, %v7160, %v7169, %v7178, %v7187, %v7196, %v7205, %v7214, %v7223, %v7232, %v7241, %v7250, %v7259, %v7268, %v7277, %v7286, %v7295, %v7304, %v7313, %v7322, %v7331, %v7340, %v7349, %v7358, %v7367, %v7376, %v7385, %v7394, %v7403, %v7412, %v7421, %v7430, %v7439, %v7448, %v7457, %v7466, %v7475, %v7484, %v7493, %v7502, %v7511, %v7520, %v7529, %v7538, %v7547, %v7556, %v7565, %v7574, %v7583, %v7592, %v7601, %v7610, %v7619, %v7628, %v7637, %v7646, %v7655, %v7664, %v7673, %v7682, %v7691, %v7700, %v7709, %v7718, %loss, %bc1, %bc2, %v5429, %v5440, %v5445, %v5456, %v5461, %v5472, %v5477, %v5488, %v5493, %v5504, %v5509, %v5520, %v5525, %v5536, %v5541, %v5552, %v5557, %v5568, %v5573, %v5584, %v5589, %v5600, %v5605, %v5616, %v5621, %v5632, %v5637, %v5648, %v5653, %v5664, %v5669, %v5680, %v5685, %v5696, %v5701, %v5712, %v5717, %v5728, %v5733, %v5744, %v5749, %v5760, %v5765, %v5776, %v5781, %v5792, %v5797, %v5808, %v5813, %v5824, %v5829, %v5840, %v5845, %v5856, %v5861, %v5872, %v5877, %v5888, %v5893, %v5904, %v5909, %v5920, %v5925, %v5936, %v5941, %v5952, %v5957, %v5968, %v5973, %v5984, %v5989, %v6000, %v6005, %v6016, %v6021, %v6032, %v6037, %v6048, %v6053, %v6064, %v6069, %v6080, %v6085, %v6096, %v6101, %v6112, %v6117, %v6128, %v6133, %v6144, %v6149, %v6160, %v6165, %v6176, %v6181, %v6192, %v6197, %v6208, %v6213, %v6224, %v6229, %v6240, %v6245, %v6256, %v6261, %v6272 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64x256x1x1xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256x64x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128x256x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128x512x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512x128x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256x512x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024x512x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256x1024x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024x256x1x1xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512x1024x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1024x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512x2048x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048x512x1x1xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<256xf32>, tensor<256xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<512xf32>, tensor<512xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<1024xf32>, tensor<1024xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<2048xf32>, tensor<2048xf32>
  }
}
