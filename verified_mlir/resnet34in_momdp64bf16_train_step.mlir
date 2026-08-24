module @m {
  func.func @resnet34in_momdp64bf16_train_step(%x: tensor<64x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<64x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 4 replicas ──
    // Every line is pretty(verified AST node) EXCEPT the per-parameter `%arsum*`
    // all_reduce / `%armean*` blocks: those are a TRUSTED CARVE-OUT (handoff §5), emitted
    // text outside the faithfulness theorems. Each replica evaluates the same tied graph
    // at the batch it was rendered for; the collective averages that function's gradients
    // over disjoint equal batches. NOTE this does NOT equal a single-device step at the
    // global batch — BN normalises per replica, so N×b != 1×(N·b) by design (§10.3b).
    // §2l step B: the conv biases are gone from the signature (BN removes them; He et al.'s
    // `.convBn` has none). The proven conv ops still take a bias operand, so it is bound to a
    // zero constant here — same op, `bias = 0`, and `x + 0.0` is exact.
    %zb64 = stablehlo.constant dense<0.0> : tensor<64xf32>
    %zb128 = stablehlo.constant dense<0.0> : tensor<128xf32>
    %zb256 = stablehlo.constant dense<0.0> : tensor<256xf32>
    %zb512 = stablehlo.constant dense<0.0> : tensor<512xf32>
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
    %v36 = stablehlo.convert %s1b0W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v37 = stablehlo.convolution(%v35, %v36)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
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
    %v92 = stablehlo.add %v91, %v33 : tensor<64x200704xf32>
    %v93 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v94 = stablehlo.maximum %v92, %v93 : tensor<64x200704xf32>
    %v95 = stablehlo.reshape %v94 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v96 = stablehlo.convert %v95 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v97 = stablehlo.convert %s1b1W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v98 = stablehlo.convolution(%v96, %v97)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v99 = stablehlo.convert %v98 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v100 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v101 = stablehlo.add %v99, %v100 : tensor<64x64x56x56xf32>
    %v102 = stablehlo.reshape %v101 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v103 = stablehlo.reshape %v102 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v104 = stablehlo.constant dense<0.0> : tensor<f32>
    %v105 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v106 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v107 = stablehlo.reduce(%v103 init: %v104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v108 = stablehlo.broadcast_in_dim %v107, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v109 = stablehlo.divide %v108, %v105 : tensor<64x64x56x56xf32>
    %v110 = stablehlo.subtract %v103, %v109 : tensor<64x64x56x56xf32>
    %v111 = stablehlo.multiply %v110, %v110 : tensor<64x64x56x56xf32>
    %v112 = stablehlo.reduce(%v111 init: %v104) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v113 = stablehlo.broadcast_in_dim %v112, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v114 = stablehlo.divide %v113, %v105 : tensor<64x64x56x56xf32>
    %v115 = stablehlo.add %v114, %v106 : tensor<64x64x56x56xf32>
    %v116 = stablehlo.rsqrt %v115 : tensor<64x64x56x56xf32>
    %v117 = stablehlo.multiply %v110, %v116 : tensor<64x64x56x56xf32>
    %v118 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v119 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v120 = stablehlo.multiply %v117, %v118 : tensor<64x64x56x56xf32>
    %v121 = stablehlo.add %v120, %v119 : tensor<64x64x56x56xf32>
    %v122 = stablehlo.reshape %v121 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v123 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v124 = stablehlo.maximum %v122, %v123 : tensor<64x200704xf32>
    %v125 = stablehlo.reshape %v124 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v126 = stablehlo.convert %v125 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v127 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v128 = stablehlo.convolution(%v126, %v127)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v129 = stablehlo.convert %v128 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v130 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v131 = stablehlo.add %v129, %v130 : tensor<64x64x56x56xf32>
    %v132 = stablehlo.reshape %v131 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v133 = stablehlo.reshape %v132 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v135 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v136 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v137 = stablehlo.reduce(%v133 init: %v134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v138 = stablehlo.broadcast_in_dim %v137, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v139 = stablehlo.divide %v138, %v135 : tensor<64x64x56x56xf32>
    %v140 = stablehlo.subtract %v133, %v139 : tensor<64x64x56x56xf32>
    %v141 = stablehlo.multiply %v140, %v140 : tensor<64x64x56x56xf32>
    %v142 = stablehlo.reduce(%v141 init: %v134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v143 = stablehlo.broadcast_in_dim %v142, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v144 = stablehlo.divide %v143, %v135 : tensor<64x64x56x56xf32>
    %v145 = stablehlo.add %v144, %v136 : tensor<64x64x56x56xf32>
    %v146 = stablehlo.rsqrt %v145 : tensor<64x64x56x56xf32>
    %v147 = stablehlo.multiply %v140, %v146 : tensor<64x64x56x56xf32>
    %v148 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v149 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v150 = stablehlo.multiply %v147, %v148 : tensor<64x64x56x56xf32>
    %v151 = stablehlo.add %v150, %v149 : tensor<64x64x56x56xf32>
    %v152 = stablehlo.reshape %v151 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v153 = stablehlo.add %v152, %v94 : tensor<64x200704xf32>
    %v154 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v155 = stablehlo.maximum %v153, %v154 : tensor<64x200704xf32>
    %v156 = stablehlo.reshape %v155 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v157 = stablehlo.convert %v156 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v158 = stablehlo.convert %s1b2W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v159 = stablehlo.convolution(%v157, %v158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v160 = stablehlo.convert %v159 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v161 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v162 = stablehlo.add %v160, %v161 : tensor<64x64x56x56xf32>
    %v163 = stablehlo.reshape %v162 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v164 = stablehlo.reshape %v163 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v166 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v167 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v168 = stablehlo.reduce(%v164 init: %v165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v169 = stablehlo.broadcast_in_dim %v168, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v170 = stablehlo.divide %v169, %v166 : tensor<64x64x56x56xf32>
    %v171 = stablehlo.subtract %v164, %v170 : tensor<64x64x56x56xf32>
    %v172 = stablehlo.multiply %v171, %v171 : tensor<64x64x56x56xf32>
    %v173 = stablehlo.reduce(%v172 init: %v165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v174 = stablehlo.broadcast_in_dim %v173, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v175 = stablehlo.divide %v174, %v166 : tensor<64x64x56x56xf32>
    %v176 = stablehlo.add %v175, %v167 : tensor<64x64x56x56xf32>
    %v177 = stablehlo.rsqrt %v176 : tensor<64x64x56x56xf32>
    %v178 = stablehlo.multiply %v171, %v177 : tensor<64x64x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v181 = stablehlo.multiply %v178, %v179 : tensor<64x64x56x56xf32>
    %v182 = stablehlo.add %v181, %v180 : tensor<64x64x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v184 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v185 = stablehlo.maximum %v183, %v184 : tensor<64x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v187 = stablehlo.convert %v186 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v188 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v189 = stablehlo.convolution(%v187, %v188)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v190 = stablehlo.convert %v189 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v191 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v192 = stablehlo.add %v190, %v191 : tensor<64x64x56x56xf32>
    %v193 = stablehlo.reshape %v192 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v194 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v195 = stablehlo.constant dense<0.0> : tensor<f32>
    %v196 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v197 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v198 = stablehlo.reduce(%v194 init: %v195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v199 = stablehlo.broadcast_in_dim %v198, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v200 = stablehlo.divide %v199, %v196 : tensor<64x64x56x56xf32>
    %v201 = stablehlo.subtract %v194, %v200 : tensor<64x64x56x56xf32>
    %v202 = stablehlo.multiply %v201, %v201 : tensor<64x64x56x56xf32>
    %v203 = stablehlo.reduce(%v202 init: %v195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v204 = stablehlo.broadcast_in_dim %v203, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v205 = stablehlo.divide %v204, %v196 : tensor<64x64x56x56xf32>
    %v206 = stablehlo.add %v205, %v197 : tensor<64x64x56x56xf32>
    %v207 = stablehlo.rsqrt %v206 : tensor<64x64x56x56xf32>
    %v208 = stablehlo.multiply %v201, %v207 : tensor<64x64x56x56xf32>
    %v209 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v210 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v211 = stablehlo.multiply %v208, %v209 : tensor<64x64x56x56xf32>
    %v212 = stablehlo.add %v211, %v210 : tensor<64x64x56x56xf32>
    %v213 = stablehlo.reshape %v212 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v214 = stablehlo.add %v213, %v155 : tensor<64x200704xf32>
    %v215 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v216 = stablehlo.maximum %v214, %v215 : tensor<64x200704xf32>
    %v217 = stablehlo.reshape %v216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v218 = stablehlo.convert %v217 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v219 = stablehlo.convert %d2W1 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xbf16>
    %v220 = stablehlo.convolution(%v218, %v219)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v221 = stablehlo.convert %v220 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v222 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v223 = stablehlo.add %v221, %v222 : tensor<64x128x28x28xf32>
    %v224 = stablehlo.reshape %v223 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v225 = stablehlo.reshape %v224 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v227 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v228 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v229 = stablehlo.reduce(%v225 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v230 = stablehlo.broadcast_in_dim %v229, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v231 = stablehlo.divide %v230, %v227 : tensor<64x128x28x28xf32>
    %v232 = stablehlo.subtract %v225, %v231 : tensor<64x128x28x28xf32>
    %v233 = stablehlo.multiply %v232, %v232 : tensor<64x128x28x28xf32>
    %v234 = stablehlo.reduce(%v233 init: %v226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v235 = stablehlo.broadcast_in_dim %v234, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v236 = stablehlo.divide %v235, %v227 : tensor<64x128x28x28xf32>
    %v237 = stablehlo.add %v236, %v228 : tensor<64x128x28x28xf32>
    %v238 = stablehlo.rsqrt %v237 : tensor<64x128x28x28xf32>
    %v239 = stablehlo.multiply %v232, %v238 : tensor<64x128x28x28xf32>
    %v240 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v241 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v242 = stablehlo.multiply %v239, %v240 : tensor<64x128x28x28xf32>
    %v243 = stablehlo.add %v242, %v241 : tensor<64x128x28x28xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v246 = stablehlo.maximum %v244, %v245 : tensor<64x100352xf32>
    %v247 = stablehlo.reshape %v246 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v248 = stablehlo.convert %v247 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v249 = stablehlo.convert %d2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v250 = stablehlo.convolution(%v248, %v249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v251 = stablehlo.convert %v250 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v252 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v253 = stablehlo.add %v251, %v252 : tensor<64x128x28x28xf32>
    %v254 = stablehlo.reshape %v253 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v255 = stablehlo.reshape %v254 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v256 = stablehlo.constant dense<0.0> : tensor<f32>
    %v257 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v258 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v259 = stablehlo.reduce(%v255 init: %v256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v260 = stablehlo.broadcast_in_dim %v259, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v261 = stablehlo.divide %v260, %v257 : tensor<64x128x28x28xf32>
    %v262 = stablehlo.subtract %v255, %v261 : tensor<64x128x28x28xf32>
    %v263 = stablehlo.multiply %v262, %v262 : tensor<64x128x28x28xf32>
    %v264 = stablehlo.reduce(%v263 init: %v256) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v265 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v266 = stablehlo.divide %v265, %v257 : tensor<64x128x28x28xf32>
    %v267 = stablehlo.add %v266, %v258 : tensor<64x128x28x28xf32>
    %v268 = stablehlo.rsqrt %v267 : tensor<64x128x28x28xf32>
    %v269 = stablehlo.multiply %v262, %v268 : tensor<64x128x28x28xf32>
    %v270 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v271 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v272 = stablehlo.multiply %v269, %v270 : tensor<64x128x28x28xf32>
    %v273 = stablehlo.add %v272, %v271 : tensor<64x128x28x28xf32>
    %v274 = stablehlo.reshape %v273 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v275 = stablehlo.reshape %v216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v276 = stablehlo.convert %v275 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v277 = stablehlo.convert %d2Wp : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xbf16>
    %v278 = stablehlo.convolution(%v276, %v277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v279 = stablehlo.convert %v278 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v280 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v281 = stablehlo.add %v279, %v280 : tensor<64x128x28x28xf32>
    %v282 = stablehlo.reshape %v281 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v283 = stablehlo.reshape %v282 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v284 = stablehlo.constant dense<0.0> : tensor<f32>
    %v285 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v286 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v287 = stablehlo.reduce(%v283 init: %v284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v288 = stablehlo.broadcast_in_dim %v287, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v289 = stablehlo.divide %v288, %v285 : tensor<64x128x28x28xf32>
    %v290 = stablehlo.subtract %v283, %v289 : tensor<64x128x28x28xf32>
    %v291 = stablehlo.multiply %v290, %v290 : tensor<64x128x28x28xf32>
    %v292 = stablehlo.reduce(%v291 init: %v284) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v293 = stablehlo.broadcast_in_dim %v292, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v294 = stablehlo.divide %v293, %v285 : tensor<64x128x28x28xf32>
    %v295 = stablehlo.add %v294, %v286 : tensor<64x128x28x28xf32>
    %v296 = stablehlo.rsqrt %v295 : tensor<64x128x28x28xf32>
    %v297 = stablehlo.multiply %v290, %v296 : tensor<64x128x28x28xf32>
    %v298 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v299 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v300 = stablehlo.multiply %v297, %v298 : tensor<64x128x28x28xf32>
    %v301 = stablehlo.add %v300, %v299 : tensor<64x128x28x28xf32>
    %v302 = stablehlo.reshape %v301 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v303 = stablehlo.add %v274, %v302 : tensor<64x100352xf32>
    %v304 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v305 = stablehlo.maximum %v303, %v304 : tensor<64x100352xf32>
    %v306 = stablehlo.reshape %v305 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v307 = stablehlo.convert %v306 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v308 = stablehlo.convert %s2b0W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v309 = stablehlo.convolution(%v307, %v308)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v310 = stablehlo.convert %v309 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v311 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v312 = stablehlo.add %v310, %v311 : tensor<64x128x28x28xf32>
    %v313 = stablehlo.reshape %v312 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v314 = stablehlo.reshape %v313 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v316 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v317 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v318 = stablehlo.reduce(%v314 init: %v315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v319 = stablehlo.broadcast_in_dim %v318, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v320 = stablehlo.divide %v319, %v316 : tensor<64x128x28x28xf32>
    %v321 = stablehlo.subtract %v314, %v320 : tensor<64x128x28x28xf32>
    %v322 = stablehlo.multiply %v321, %v321 : tensor<64x128x28x28xf32>
    %v323 = stablehlo.reduce(%v322 init: %v315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v324 = stablehlo.broadcast_in_dim %v323, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v325 = stablehlo.divide %v324, %v316 : tensor<64x128x28x28xf32>
    %v326 = stablehlo.add %v325, %v317 : tensor<64x128x28x28xf32>
    %v327 = stablehlo.rsqrt %v326 : tensor<64x128x28x28xf32>
    %v328 = stablehlo.multiply %v321, %v327 : tensor<64x128x28x28xf32>
    %v329 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v330 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v331 = stablehlo.multiply %v328, %v329 : tensor<64x128x28x28xf32>
    %v332 = stablehlo.add %v331, %v330 : tensor<64x128x28x28xf32>
    %v333 = stablehlo.reshape %v332 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v334 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v335 = stablehlo.maximum %v333, %v334 : tensor<64x100352xf32>
    %v336 = stablehlo.reshape %v335 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v337 = stablehlo.convert %v336 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v338 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v339 = stablehlo.convolution(%v337, %v338)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v340 = stablehlo.convert %v339 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v341 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v342 = stablehlo.add %v340, %v341 : tensor<64x128x28x28xf32>
    %v343 = stablehlo.reshape %v342 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v344 = stablehlo.reshape %v343 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v346 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v347 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v348 = stablehlo.reduce(%v344 init: %v345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v349 = stablehlo.broadcast_in_dim %v348, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v350 = stablehlo.divide %v349, %v346 : tensor<64x128x28x28xf32>
    %v351 = stablehlo.subtract %v344, %v350 : tensor<64x128x28x28xf32>
    %v352 = stablehlo.multiply %v351, %v351 : tensor<64x128x28x28xf32>
    %v353 = stablehlo.reduce(%v352 init: %v345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v354 = stablehlo.broadcast_in_dim %v353, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v355 = stablehlo.divide %v354, %v346 : tensor<64x128x28x28xf32>
    %v356 = stablehlo.add %v355, %v347 : tensor<64x128x28x28xf32>
    %v357 = stablehlo.rsqrt %v356 : tensor<64x128x28x28xf32>
    %v358 = stablehlo.multiply %v351, %v357 : tensor<64x128x28x28xf32>
    %v359 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v360 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v361 = stablehlo.multiply %v358, %v359 : tensor<64x128x28x28xf32>
    %v362 = stablehlo.add %v361, %v360 : tensor<64x128x28x28xf32>
    %v363 = stablehlo.reshape %v362 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v364 = stablehlo.add %v363, %v305 : tensor<64x100352xf32>
    %v365 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v366 = stablehlo.maximum %v364, %v365 : tensor<64x100352xf32>
    %v367 = stablehlo.reshape %v366 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v368 = stablehlo.convert %v367 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v369 = stablehlo.convert %s2b1W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v370 = stablehlo.convolution(%v368, %v369)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v371 = stablehlo.convert %v370 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v372 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v373 = stablehlo.add %v371, %v372 : tensor<64x128x28x28xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v375 = stablehlo.reshape %v374 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v376 = stablehlo.constant dense<0.0> : tensor<f32>
    %v377 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v378 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v379 = stablehlo.reduce(%v375 init: %v376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v380 = stablehlo.broadcast_in_dim %v379, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v381 = stablehlo.divide %v380, %v377 : tensor<64x128x28x28xf32>
    %v382 = stablehlo.subtract %v375, %v381 : tensor<64x128x28x28xf32>
    %v383 = stablehlo.multiply %v382, %v382 : tensor<64x128x28x28xf32>
    %v384 = stablehlo.reduce(%v383 init: %v376) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v385 = stablehlo.broadcast_in_dim %v384, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v386 = stablehlo.divide %v385, %v377 : tensor<64x128x28x28xf32>
    %v387 = stablehlo.add %v386, %v378 : tensor<64x128x28x28xf32>
    %v388 = stablehlo.rsqrt %v387 : tensor<64x128x28x28xf32>
    %v389 = stablehlo.multiply %v382, %v388 : tensor<64x128x28x28xf32>
    %v390 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v391 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v392 = stablehlo.multiply %v389, %v390 : tensor<64x128x28x28xf32>
    %v393 = stablehlo.add %v392, %v391 : tensor<64x128x28x28xf32>
    %v394 = stablehlo.reshape %v393 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v395 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v396 = stablehlo.maximum %v394, %v395 : tensor<64x100352xf32>
    %v397 = stablehlo.reshape %v396 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v398 = stablehlo.convert %v397 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v399 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v400 = stablehlo.convolution(%v398, %v399)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v401 = stablehlo.convert %v400 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v402 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v403 = stablehlo.add %v401, %v402 : tensor<64x128x28x28xf32>
    %v404 = stablehlo.reshape %v403 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v405 = stablehlo.reshape %v404 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v406 = stablehlo.constant dense<0.0> : tensor<f32>
    %v407 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v408 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v409 = stablehlo.reduce(%v405 init: %v406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v410 = stablehlo.broadcast_in_dim %v409, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v411 = stablehlo.divide %v410, %v407 : tensor<64x128x28x28xf32>
    %v412 = stablehlo.subtract %v405, %v411 : tensor<64x128x28x28xf32>
    %v413 = stablehlo.multiply %v412, %v412 : tensor<64x128x28x28xf32>
    %v414 = stablehlo.reduce(%v413 init: %v406) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v415 = stablehlo.broadcast_in_dim %v414, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v416 = stablehlo.divide %v415, %v407 : tensor<64x128x28x28xf32>
    %v417 = stablehlo.add %v416, %v408 : tensor<64x128x28x28xf32>
    %v418 = stablehlo.rsqrt %v417 : tensor<64x128x28x28xf32>
    %v419 = stablehlo.multiply %v412, %v418 : tensor<64x128x28x28xf32>
    %v420 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v421 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v422 = stablehlo.multiply %v419, %v420 : tensor<64x128x28x28xf32>
    %v423 = stablehlo.add %v422, %v421 : tensor<64x128x28x28xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v425 = stablehlo.add %v424, %v366 : tensor<64x100352xf32>
    %v426 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v427 = stablehlo.maximum %v425, %v426 : tensor<64x100352xf32>
    %v428 = stablehlo.reshape %v427 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v429 = stablehlo.convert %v428 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v430 = stablehlo.convert %s2b2W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v431 = stablehlo.convolution(%v429, %v430)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v432 = stablehlo.convert %v431 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v433 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v434 = stablehlo.add %v432, %v433 : tensor<64x128x28x28xf32>
    %v435 = stablehlo.reshape %v434 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v436 = stablehlo.reshape %v435 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v437 = stablehlo.constant dense<0.0> : tensor<f32>
    %v438 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v439 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v440 = stablehlo.reduce(%v436 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v441 = stablehlo.broadcast_in_dim %v440, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v442 = stablehlo.divide %v441, %v438 : tensor<64x128x28x28xf32>
    %v443 = stablehlo.subtract %v436, %v442 : tensor<64x128x28x28xf32>
    %v444 = stablehlo.multiply %v443, %v443 : tensor<64x128x28x28xf32>
    %v445 = stablehlo.reduce(%v444 init: %v437) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v447 = stablehlo.divide %v446, %v438 : tensor<64x128x28x28xf32>
    %v448 = stablehlo.add %v447, %v439 : tensor<64x128x28x28xf32>
    %v449 = stablehlo.rsqrt %v448 : tensor<64x128x28x28xf32>
    %v450 = stablehlo.multiply %v443, %v449 : tensor<64x128x28x28xf32>
    %v451 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v452 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v453 = stablehlo.multiply %v450, %v451 : tensor<64x128x28x28xf32>
    %v454 = stablehlo.add %v453, %v452 : tensor<64x128x28x28xf32>
    %v455 = stablehlo.reshape %v454 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v456 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v457 = stablehlo.maximum %v455, %v456 : tensor<64x100352xf32>
    %v458 = stablehlo.reshape %v457 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v459 = stablehlo.convert %v458 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v460 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v461 = stablehlo.convolution(%v459, %v460)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v462 = stablehlo.convert %v461 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v463 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v464 = stablehlo.add %v462, %v463 : tensor<64x128x28x28xf32>
    %v465 = stablehlo.reshape %v464 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v466 = stablehlo.reshape %v465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v467 = stablehlo.constant dense<0.0> : tensor<f32>
    %v468 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v469 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v470 = stablehlo.reduce(%v466 init: %v467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v471 = stablehlo.broadcast_in_dim %v470, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v472 = stablehlo.divide %v471, %v468 : tensor<64x128x28x28xf32>
    %v473 = stablehlo.subtract %v466, %v472 : tensor<64x128x28x28xf32>
    %v474 = stablehlo.multiply %v473, %v473 : tensor<64x128x28x28xf32>
    %v475 = stablehlo.reduce(%v474 init: %v467) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v476 = stablehlo.broadcast_in_dim %v475, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v477 = stablehlo.divide %v476, %v468 : tensor<64x128x28x28xf32>
    %v478 = stablehlo.add %v477, %v469 : tensor<64x128x28x28xf32>
    %v479 = stablehlo.rsqrt %v478 : tensor<64x128x28x28xf32>
    %v480 = stablehlo.multiply %v473, %v479 : tensor<64x128x28x28xf32>
    %v481 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v482 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v483 = stablehlo.multiply %v480, %v481 : tensor<64x128x28x28xf32>
    %v484 = stablehlo.add %v483, %v482 : tensor<64x128x28x28xf32>
    %v485 = stablehlo.reshape %v484 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v486 = stablehlo.add %v485, %v427 : tensor<64x100352xf32>
    %v487 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v488 = stablehlo.maximum %v486, %v487 : tensor<64x100352xf32>
    %v489 = stablehlo.reshape %v488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v490 = stablehlo.convert %v489 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v491 = stablehlo.convert %d3W1 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xbf16>
    %v492 = stablehlo.convolution(%v490, %v491)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<256x128x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v493 = stablehlo.convert %v492 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v494 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v495 = stablehlo.add %v493, %v494 : tensor<64x256x14x14xf32>
    %v496 = stablehlo.reshape %v495 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v497 = stablehlo.reshape %v496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v499 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v500 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v501 = stablehlo.reduce(%v497 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v502 = stablehlo.broadcast_in_dim %v501, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v503 = stablehlo.divide %v502, %v499 : tensor<64x256x14x14xf32>
    %v504 = stablehlo.subtract %v497, %v503 : tensor<64x256x14x14xf32>
    %v505 = stablehlo.multiply %v504, %v504 : tensor<64x256x14x14xf32>
    %v506 = stablehlo.reduce(%v505 init: %v498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v507 = stablehlo.broadcast_in_dim %v506, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v508 = stablehlo.divide %v507, %v499 : tensor<64x256x14x14xf32>
    %v509 = stablehlo.add %v508, %v500 : tensor<64x256x14x14xf32>
    %v510 = stablehlo.rsqrt %v509 : tensor<64x256x14x14xf32>
    %v511 = stablehlo.multiply %v504, %v510 : tensor<64x256x14x14xf32>
    %v512 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v513 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v514 = stablehlo.multiply %v511, %v512 : tensor<64x256x14x14xf32>
    %v515 = stablehlo.add %v514, %v513 : tensor<64x256x14x14xf32>
    %v516 = stablehlo.reshape %v515 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v517 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v518 = stablehlo.maximum %v516, %v517 : tensor<64x50176xf32>
    %v519 = stablehlo.reshape %v518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v520 = stablehlo.convert %v519 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v521 = stablehlo.convert %d3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v522 = stablehlo.convolution(%v520, %v521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v523 = stablehlo.convert %v522 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v524 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v525 = stablehlo.add %v523, %v524 : tensor<64x256x14x14xf32>
    %v526 = stablehlo.reshape %v525 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v527 = stablehlo.reshape %v526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v529 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v530 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v531 = stablehlo.reduce(%v527 init: %v528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v532 = stablehlo.broadcast_in_dim %v531, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v533 = stablehlo.divide %v532, %v529 : tensor<64x256x14x14xf32>
    %v534 = stablehlo.subtract %v527, %v533 : tensor<64x256x14x14xf32>
    %v535 = stablehlo.multiply %v534, %v534 : tensor<64x256x14x14xf32>
    %v536 = stablehlo.reduce(%v535 init: %v528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v537 = stablehlo.broadcast_in_dim %v536, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v538 = stablehlo.divide %v537, %v529 : tensor<64x256x14x14xf32>
    %v539 = stablehlo.add %v538, %v530 : tensor<64x256x14x14xf32>
    %v540 = stablehlo.rsqrt %v539 : tensor<64x256x14x14xf32>
    %v541 = stablehlo.multiply %v534, %v540 : tensor<64x256x14x14xf32>
    %v542 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v543 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v544 = stablehlo.multiply %v541, %v542 : tensor<64x256x14x14xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<64x256x14x14xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v547 = stablehlo.reshape %v488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v548 = stablehlo.convert %v547 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v549 = stablehlo.convert %d3Wp : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v550 = stablehlo.convolution(%v548, %v549)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<256x128x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v551 = stablehlo.convert %v550 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v552 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v553 = stablehlo.add %v551, %v552 : tensor<64x256x14x14xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v556 = stablehlo.constant dense<0.0> : tensor<f32>
    %v557 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v558 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v559 = stablehlo.reduce(%v555 init: %v556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v560 = stablehlo.broadcast_in_dim %v559, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v561 = stablehlo.divide %v560, %v557 : tensor<64x256x14x14xf32>
    %v562 = stablehlo.subtract %v555, %v561 : tensor<64x256x14x14xf32>
    %v563 = stablehlo.multiply %v562, %v562 : tensor<64x256x14x14xf32>
    %v564 = stablehlo.reduce(%v563 init: %v556) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v565 = stablehlo.broadcast_in_dim %v564, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v566 = stablehlo.divide %v565, %v557 : tensor<64x256x14x14xf32>
    %v567 = stablehlo.add %v566, %v558 : tensor<64x256x14x14xf32>
    %v568 = stablehlo.rsqrt %v567 : tensor<64x256x14x14xf32>
    %v569 = stablehlo.multiply %v562, %v568 : tensor<64x256x14x14xf32>
    %v570 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v571 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v572 = stablehlo.multiply %v569, %v570 : tensor<64x256x14x14xf32>
    %v573 = stablehlo.add %v572, %v571 : tensor<64x256x14x14xf32>
    %v574 = stablehlo.reshape %v573 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v575 = stablehlo.add %v546, %v574 : tensor<64x50176xf32>
    %v576 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v577 = stablehlo.maximum %v575, %v576 : tensor<64x50176xf32>
    %v578 = stablehlo.reshape %v577 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v579 = stablehlo.convert %v578 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v580 = stablehlo.convert %s3b0W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v581 = stablehlo.convolution(%v579, %v580)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v582 = stablehlo.convert %v581 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v583 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v584 = stablehlo.add %v582, %v583 : tensor<64x256x14x14xf32>
    %v585 = stablehlo.reshape %v584 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v586 = stablehlo.reshape %v585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v587 = stablehlo.constant dense<0.0> : tensor<f32>
    %v588 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v589 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v590 = stablehlo.reduce(%v586 init: %v587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v591 = stablehlo.broadcast_in_dim %v590, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v592 = stablehlo.divide %v591, %v588 : tensor<64x256x14x14xf32>
    %v593 = stablehlo.subtract %v586, %v592 : tensor<64x256x14x14xf32>
    %v594 = stablehlo.multiply %v593, %v593 : tensor<64x256x14x14xf32>
    %v595 = stablehlo.reduce(%v594 init: %v587) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v596 = stablehlo.broadcast_in_dim %v595, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v597 = stablehlo.divide %v596, %v588 : tensor<64x256x14x14xf32>
    %v598 = stablehlo.add %v597, %v589 : tensor<64x256x14x14xf32>
    %v599 = stablehlo.rsqrt %v598 : tensor<64x256x14x14xf32>
    %v600 = stablehlo.multiply %v593, %v599 : tensor<64x256x14x14xf32>
    %v601 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v602 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v603 = stablehlo.multiply %v600, %v601 : tensor<64x256x14x14xf32>
    %v604 = stablehlo.add %v603, %v602 : tensor<64x256x14x14xf32>
    %v605 = stablehlo.reshape %v604 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v606 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v607 = stablehlo.maximum %v605, %v606 : tensor<64x50176xf32>
    %v608 = stablehlo.reshape %v607 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v609 = stablehlo.convert %v608 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v610 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v611 = stablehlo.convolution(%v609, %v610)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v612 = stablehlo.convert %v611 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v613 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v614 = stablehlo.add %v612, %v613 : tensor<64x256x14x14xf32>
    %v615 = stablehlo.reshape %v614 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v616 = stablehlo.reshape %v615 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v617 = stablehlo.constant dense<0.0> : tensor<f32>
    %v618 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v619 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v620 = stablehlo.reduce(%v616 init: %v617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v621 = stablehlo.broadcast_in_dim %v620, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v622 = stablehlo.divide %v621, %v618 : tensor<64x256x14x14xf32>
    %v623 = stablehlo.subtract %v616, %v622 : tensor<64x256x14x14xf32>
    %v624 = stablehlo.multiply %v623, %v623 : tensor<64x256x14x14xf32>
    %v625 = stablehlo.reduce(%v624 init: %v617) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v626 = stablehlo.broadcast_in_dim %v625, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v627 = stablehlo.divide %v626, %v618 : tensor<64x256x14x14xf32>
    %v628 = stablehlo.add %v627, %v619 : tensor<64x256x14x14xf32>
    %v629 = stablehlo.rsqrt %v628 : tensor<64x256x14x14xf32>
    %v630 = stablehlo.multiply %v623, %v629 : tensor<64x256x14x14xf32>
    %v631 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v632 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v633 = stablehlo.multiply %v630, %v631 : tensor<64x256x14x14xf32>
    %v634 = stablehlo.add %v633, %v632 : tensor<64x256x14x14xf32>
    %v635 = stablehlo.reshape %v634 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v636 = stablehlo.add %v635, %v577 : tensor<64x50176xf32>
    %v637 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v638 = stablehlo.maximum %v636, %v637 : tensor<64x50176xf32>
    %v639 = stablehlo.reshape %v638 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v640 = stablehlo.convert %v639 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v641 = stablehlo.convert %s3b1W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v642 = stablehlo.convolution(%v640, %v641)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v643 = stablehlo.convert %v642 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v644 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v645 = stablehlo.add %v643, %v644 : tensor<64x256x14x14xf32>
    %v646 = stablehlo.reshape %v645 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v647 = stablehlo.reshape %v646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v648 = stablehlo.constant dense<0.0> : tensor<f32>
    %v649 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v650 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v651 = stablehlo.reduce(%v647 init: %v648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v652 = stablehlo.broadcast_in_dim %v651, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v653 = stablehlo.divide %v652, %v649 : tensor<64x256x14x14xf32>
    %v654 = stablehlo.subtract %v647, %v653 : tensor<64x256x14x14xf32>
    %v655 = stablehlo.multiply %v654, %v654 : tensor<64x256x14x14xf32>
    %v656 = stablehlo.reduce(%v655 init: %v648) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v657 = stablehlo.broadcast_in_dim %v656, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v658 = stablehlo.divide %v657, %v649 : tensor<64x256x14x14xf32>
    %v659 = stablehlo.add %v658, %v650 : tensor<64x256x14x14xf32>
    %v660 = stablehlo.rsqrt %v659 : tensor<64x256x14x14xf32>
    %v661 = stablehlo.multiply %v654, %v660 : tensor<64x256x14x14xf32>
    %v662 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v663 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v664 = stablehlo.multiply %v661, %v662 : tensor<64x256x14x14xf32>
    %v665 = stablehlo.add %v664, %v663 : tensor<64x256x14x14xf32>
    %v666 = stablehlo.reshape %v665 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v667 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v668 = stablehlo.maximum %v666, %v667 : tensor<64x50176xf32>
    %v669 = stablehlo.reshape %v668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v670 = stablehlo.convert %v669 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v671 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v672 = stablehlo.convolution(%v670, %v671)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v673 = stablehlo.convert %v672 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v674 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v675 = stablehlo.add %v673, %v674 : tensor<64x256x14x14xf32>
    %v676 = stablehlo.reshape %v675 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v677 = stablehlo.reshape %v676 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v679 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v680 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v681 = stablehlo.reduce(%v677 init: %v678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v682 = stablehlo.broadcast_in_dim %v681, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v683 = stablehlo.divide %v682, %v679 : tensor<64x256x14x14xf32>
    %v684 = stablehlo.subtract %v677, %v683 : tensor<64x256x14x14xf32>
    %v685 = stablehlo.multiply %v684, %v684 : tensor<64x256x14x14xf32>
    %v686 = stablehlo.reduce(%v685 init: %v678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v687 = stablehlo.broadcast_in_dim %v686, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v688 = stablehlo.divide %v687, %v679 : tensor<64x256x14x14xf32>
    %v689 = stablehlo.add %v688, %v680 : tensor<64x256x14x14xf32>
    %v690 = stablehlo.rsqrt %v689 : tensor<64x256x14x14xf32>
    %v691 = stablehlo.multiply %v684, %v690 : tensor<64x256x14x14xf32>
    %v692 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v693 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v694 = stablehlo.multiply %v691, %v692 : tensor<64x256x14x14xf32>
    %v695 = stablehlo.add %v694, %v693 : tensor<64x256x14x14xf32>
    %v696 = stablehlo.reshape %v695 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v697 = stablehlo.add %v696, %v638 : tensor<64x50176xf32>
    %v698 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v699 = stablehlo.maximum %v697, %v698 : tensor<64x50176xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v701 = stablehlo.convert %v700 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v702 = stablehlo.convert %s3b2W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v703 = stablehlo.convolution(%v701, %v702)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v704 = stablehlo.convert %v703 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v705 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v706 = stablehlo.add %v704, %v705 : tensor<64x256x14x14xf32>
    %v707 = stablehlo.reshape %v706 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v708 = stablehlo.reshape %v707 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v709 = stablehlo.constant dense<0.0> : tensor<f32>
    %v710 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v711 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v712 = stablehlo.reduce(%v708 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v713 = stablehlo.broadcast_in_dim %v712, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v714 = stablehlo.divide %v713, %v710 : tensor<64x256x14x14xf32>
    %v715 = stablehlo.subtract %v708, %v714 : tensor<64x256x14x14xf32>
    %v716 = stablehlo.multiply %v715, %v715 : tensor<64x256x14x14xf32>
    %v717 = stablehlo.reduce(%v716 init: %v709) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v718 = stablehlo.broadcast_in_dim %v717, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v719 = stablehlo.divide %v718, %v710 : tensor<64x256x14x14xf32>
    %v720 = stablehlo.add %v719, %v711 : tensor<64x256x14x14xf32>
    %v721 = stablehlo.rsqrt %v720 : tensor<64x256x14x14xf32>
    %v722 = stablehlo.multiply %v715, %v721 : tensor<64x256x14x14xf32>
    %v723 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v724 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v725 = stablehlo.multiply %v722, %v723 : tensor<64x256x14x14xf32>
    %v726 = stablehlo.add %v725, %v724 : tensor<64x256x14x14xf32>
    %v727 = stablehlo.reshape %v726 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v728 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v729 = stablehlo.maximum %v727, %v728 : tensor<64x50176xf32>
    %v730 = stablehlo.reshape %v729 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v731 = stablehlo.convert %v730 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v732 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v733 = stablehlo.convolution(%v731, %v732)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v734 = stablehlo.convert %v733 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v735 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v736 = stablehlo.add %v734, %v735 : tensor<64x256x14x14xf32>
    %v737 = stablehlo.reshape %v736 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v739 = stablehlo.constant dense<0.0> : tensor<f32>
    %v740 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v741 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v742 = stablehlo.reduce(%v738 init: %v739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v743 = stablehlo.broadcast_in_dim %v742, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v744 = stablehlo.divide %v743, %v740 : tensor<64x256x14x14xf32>
    %v745 = stablehlo.subtract %v738, %v744 : tensor<64x256x14x14xf32>
    %v746 = stablehlo.multiply %v745, %v745 : tensor<64x256x14x14xf32>
    %v747 = stablehlo.reduce(%v746 init: %v739) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v748 = stablehlo.broadcast_in_dim %v747, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v749 = stablehlo.divide %v748, %v740 : tensor<64x256x14x14xf32>
    %v750 = stablehlo.add %v749, %v741 : tensor<64x256x14x14xf32>
    %v751 = stablehlo.rsqrt %v750 : tensor<64x256x14x14xf32>
    %v752 = stablehlo.multiply %v745, %v751 : tensor<64x256x14x14xf32>
    %v753 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v754 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v755 = stablehlo.multiply %v752, %v753 : tensor<64x256x14x14xf32>
    %v756 = stablehlo.add %v755, %v754 : tensor<64x256x14x14xf32>
    %v757 = stablehlo.reshape %v756 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v758 = stablehlo.add %v757, %v699 : tensor<64x50176xf32>
    %v759 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v760 = stablehlo.maximum %v758, %v759 : tensor<64x50176xf32>
    %v761 = stablehlo.reshape %v760 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v762 = stablehlo.convert %v761 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v763 = stablehlo.convert %s3b3W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v764 = stablehlo.convolution(%v762, %v763)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v765 = stablehlo.convert %v764 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v766 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v767 = stablehlo.add %v765, %v766 : tensor<64x256x14x14xf32>
    %v768 = stablehlo.reshape %v767 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v769 = stablehlo.reshape %v768 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v770 = stablehlo.constant dense<0.0> : tensor<f32>
    %v771 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v772 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v773 = stablehlo.reduce(%v769 init: %v770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v774 = stablehlo.broadcast_in_dim %v773, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v775 = stablehlo.divide %v774, %v771 : tensor<64x256x14x14xf32>
    %v776 = stablehlo.subtract %v769, %v775 : tensor<64x256x14x14xf32>
    %v777 = stablehlo.multiply %v776, %v776 : tensor<64x256x14x14xf32>
    %v778 = stablehlo.reduce(%v777 init: %v770) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v779 = stablehlo.broadcast_in_dim %v778, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v780 = stablehlo.divide %v779, %v771 : tensor<64x256x14x14xf32>
    %v781 = stablehlo.add %v780, %v772 : tensor<64x256x14x14xf32>
    %v782 = stablehlo.rsqrt %v781 : tensor<64x256x14x14xf32>
    %v783 = stablehlo.multiply %v776, %v782 : tensor<64x256x14x14xf32>
    %v784 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v785 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v786 = stablehlo.multiply %v783, %v784 : tensor<64x256x14x14xf32>
    %v787 = stablehlo.add %v786, %v785 : tensor<64x256x14x14xf32>
    %v788 = stablehlo.reshape %v787 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v789 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v790 = stablehlo.maximum %v788, %v789 : tensor<64x50176xf32>
    %v791 = stablehlo.reshape %v790 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v792 = stablehlo.convert %v791 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v793 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v794 = stablehlo.convolution(%v792, %v793)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v795 = stablehlo.convert %v794 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v796 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v797 = stablehlo.add %v795, %v796 : tensor<64x256x14x14xf32>
    %v798 = stablehlo.reshape %v797 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v799 = stablehlo.reshape %v798 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v800 = stablehlo.constant dense<0.0> : tensor<f32>
    %v801 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v802 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v803 = stablehlo.reduce(%v799 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.broadcast_in_dim %v803, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v805 = stablehlo.divide %v804, %v801 : tensor<64x256x14x14xf32>
    %v806 = stablehlo.subtract %v799, %v805 : tensor<64x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<64x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v800) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v809 = stablehlo.broadcast_in_dim %v808, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v810 = stablehlo.divide %v809, %v801 : tensor<64x256x14x14xf32>
    %v811 = stablehlo.add %v810, %v802 : tensor<64x256x14x14xf32>
    %v812 = stablehlo.rsqrt %v811 : tensor<64x256x14x14xf32>
    %v813 = stablehlo.multiply %v806, %v812 : tensor<64x256x14x14xf32>
    %v814 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v815 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v816 = stablehlo.multiply %v813, %v814 : tensor<64x256x14x14xf32>
    %v817 = stablehlo.add %v816, %v815 : tensor<64x256x14x14xf32>
    %v818 = stablehlo.reshape %v817 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v819 = stablehlo.add %v818, %v760 : tensor<64x50176xf32>
    %v820 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v821 = stablehlo.maximum %v819, %v820 : tensor<64x50176xf32>
    %v822 = stablehlo.reshape %v821 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v823 = stablehlo.convert %v822 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v824 = stablehlo.convert %s3b4W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v825 = stablehlo.convolution(%v823, %v824)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v826 = stablehlo.convert %v825 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v827 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v828 = stablehlo.add %v826, %v827 : tensor<64x256x14x14xf32>
    %v829 = stablehlo.reshape %v828 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v830 = stablehlo.reshape %v829 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v832 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v833 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v834 = stablehlo.reduce(%v830 init: %v831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v835 = stablehlo.broadcast_in_dim %v834, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v836 = stablehlo.divide %v835, %v832 : tensor<64x256x14x14xf32>
    %v837 = stablehlo.subtract %v830, %v836 : tensor<64x256x14x14xf32>
    %v838 = stablehlo.multiply %v837, %v837 : tensor<64x256x14x14xf32>
    %v839 = stablehlo.reduce(%v838 init: %v831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v840 = stablehlo.broadcast_in_dim %v839, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v841 = stablehlo.divide %v840, %v832 : tensor<64x256x14x14xf32>
    %v842 = stablehlo.add %v841, %v833 : tensor<64x256x14x14xf32>
    %v843 = stablehlo.rsqrt %v842 : tensor<64x256x14x14xf32>
    %v844 = stablehlo.multiply %v837, %v843 : tensor<64x256x14x14xf32>
    %v845 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v846 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v847 = stablehlo.multiply %v844, %v845 : tensor<64x256x14x14xf32>
    %v848 = stablehlo.add %v847, %v846 : tensor<64x256x14x14xf32>
    %v849 = stablehlo.reshape %v848 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v850 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v851 = stablehlo.maximum %v849, %v850 : tensor<64x50176xf32>
    %v852 = stablehlo.reshape %v851 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v853 = stablehlo.convert %v852 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v854 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v855 = stablehlo.convolution(%v853, %v854)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v856 = stablehlo.convert %v855 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v857 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v858 = stablehlo.add %v856, %v857 : tensor<64x256x14x14xf32>
    %v859 = stablehlo.reshape %v858 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v860 = stablehlo.reshape %v859 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v861 = stablehlo.constant dense<0.0> : tensor<f32>
    %v862 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v863 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v864 = stablehlo.reduce(%v860 init: %v861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v865 = stablehlo.broadcast_in_dim %v864, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v866 = stablehlo.divide %v865, %v862 : tensor<64x256x14x14xf32>
    %v867 = stablehlo.subtract %v860, %v866 : tensor<64x256x14x14xf32>
    %v868 = stablehlo.multiply %v867, %v867 : tensor<64x256x14x14xf32>
    %v869 = stablehlo.reduce(%v868 init: %v861) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v870 = stablehlo.broadcast_in_dim %v869, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v871 = stablehlo.divide %v870, %v862 : tensor<64x256x14x14xf32>
    %v872 = stablehlo.add %v871, %v863 : tensor<64x256x14x14xf32>
    %v873 = stablehlo.rsqrt %v872 : tensor<64x256x14x14xf32>
    %v874 = stablehlo.multiply %v867, %v873 : tensor<64x256x14x14xf32>
    %v875 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v876 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v877 = stablehlo.multiply %v874, %v875 : tensor<64x256x14x14xf32>
    %v878 = stablehlo.add %v877, %v876 : tensor<64x256x14x14xf32>
    %v879 = stablehlo.reshape %v878 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v880 = stablehlo.add %v879, %v821 : tensor<64x50176xf32>
    %v881 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v882 = stablehlo.maximum %v880, %v881 : tensor<64x50176xf32>
    %v883 = stablehlo.reshape %v882 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v884 = stablehlo.convert %v883 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v885 = stablehlo.convert %d4W1 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xbf16>
    %v886 = stablehlo.convolution(%v884, %v885)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<512x256x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v887 = stablehlo.convert %v886 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v888 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v889 = stablehlo.add %v887, %v888 : tensor<64x512x7x7xf32>
    %v890 = stablehlo.reshape %v889 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v891 = stablehlo.reshape %v890 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v892 = stablehlo.constant dense<0.0> : tensor<f32>
    %v893 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v894 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v895 = stablehlo.reduce(%v891 init: %v892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v896 = stablehlo.broadcast_in_dim %v895, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v897 = stablehlo.divide %v896, %v893 : tensor<64x512x7x7xf32>
    %v898 = stablehlo.subtract %v891, %v897 : tensor<64x512x7x7xf32>
    %v899 = stablehlo.multiply %v898, %v898 : tensor<64x512x7x7xf32>
    %v900 = stablehlo.reduce(%v899 init: %v892) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v901 = stablehlo.broadcast_in_dim %v900, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v902 = stablehlo.divide %v901, %v893 : tensor<64x512x7x7xf32>
    %v903 = stablehlo.add %v902, %v894 : tensor<64x512x7x7xf32>
    %v904 = stablehlo.rsqrt %v903 : tensor<64x512x7x7xf32>
    %v905 = stablehlo.multiply %v898, %v904 : tensor<64x512x7x7xf32>
    %v906 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v907 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v908 = stablehlo.multiply %v905, %v906 : tensor<64x512x7x7xf32>
    %v909 = stablehlo.add %v908, %v907 : tensor<64x512x7x7xf32>
    %v910 = stablehlo.reshape %v909 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v911 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v912 = stablehlo.maximum %v910, %v911 : tensor<64x25088xf32>
    %v913 = stablehlo.reshape %v912 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v914 = stablehlo.convert %v913 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v915 = stablehlo.convert %d4W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v916 = stablehlo.convolution(%v914, %v915)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v917 = stablehlo.convert %v916 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v918 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v919 = stablehlo.add %v917, %v918 : tensor<64x512x7x7xf32>
    %v920 = stablehlo.reshape %v919 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v921 = stablehlo.reshape %v920 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v923 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v924 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v925 = stablehlo.reduce(%v921 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v926 = stablehlo.broadcast_in_dim %v925, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v927 = stablehlo.divide %v926, %v923 : tensor<64x512x7x7xf32>
    %v928 = stablehlo.subtract %v921, %v927 : tensor<64x512x7x7xf32>
    %v929 = stablehlo.multiply %v928, %v928 : tensor<64x512x7x7xf32>
    %v930 = stablehlo.reduce(%v929 init: %v922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v931 = stablehlo.broadcast_in_dim %v930, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v932 = stablehlo.divide %v931, %v923 : tensor<64x512x7x7xf32>
    %v933 = stablehlo.add %v932, %v924 : tensor<64x512x7x7xf32>
    %v934 = stablehlo.rsqrt %v933 : tensor<64x512x7x7xf32>
    %v935 = stablehlo.multiply %v928, %v934 : tensor<64x512x7x7xf32>
    %v936 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v937 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v938 = stablehlo.multiply %v935, %v936 : tensor<64x512x7x7xf32>
    %v939 = stablehlo.add %v938, %v937 : tensor<64x512x7x7xf32>
    %v940 = stablehlo.reshape %v939 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v941 = stablehlo.reshape %v882 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v942 = stablehlo.convert %v941 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v943 = stablehlo.convert %d4Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v944 = stablehlo.convolution(%v942, %v943)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v945 = stablehlo.convert %v944 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v946 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v947 = stablehlo.add %v945, %v946 : tensor<64x512x7x7xf32>
    %v948 = stablehlo.reshape %v947 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v949 = stablehlo.reshape %v948 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v950 = stablehlo.constant dense<0.0> : tensor<f32>
    %v951 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v952 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v953 = stablehlo.reduce(%v949 init: %v950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v954 = stablehlo.broadcast_in_dim %v953, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v955 = stablehlo.divide %v954, %v951 : tensor<64x512x7x7xf32>
    %v956 = stablehlo.subtract %v949, %v955 : tensor<64x512x7x7xf32>
    %v957 = stablehlo.multiply %v956, %v956 : tensor<64x512x7x7xf32>
    %v958 = stablehlo.reduce(%v957 init: %v950) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v959 = stablehlo.broadcast_in_dim %v958, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v960 = stablehlo.divide %v959, %v951 : tensor<64x512x7x7xf32>
    %v961 = stablehlo.add %v960, %v952 : tensor<64x512x7x7xf32>
    %v962 = stablehlo.rsqrt %v961 : tensor<64x512x7x7xf32>
    %v963 = stablehlo.multiply %v956, %v962 : tensor<64x512x7x7xf32>
    %v964 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v965 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v966 = stablehlo.multiply %v963, %v964 : tensor<64x512x7x7xf32>
    %v967 = stablehlo.add %v966, %v965 : tensor<64x512x7x7xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v969 = stablehlo.add %v940, %v968 : tensor<64x25088xf32>
    %v970 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v971 = stablehlo.maximum %v969, %v970 : tensor<64x25088xf32>
    %v972 = stablehlo.reshape %v971 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v973 = stablehlo.convert %v972 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v974 = stablehlo.convert %s4b0W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v975 = stablehlo.convolution(%v973, %v974)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v976 = stablehlo.convert %v975 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v977 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v978 = stablehlo.add %v976, %v977 : tensor<64x512x7x7xf32>
    %v979 = stablehlo.reshape %v978 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v980 = stablehlo.reshape %v979 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v982 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v983 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v984 = stablehlo.reduce(%v980 init: %v981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v985 = stablehlo.broadcast_in_dim %v984, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v986 = stablehlo.divide %v985, %v982 : tensor<64x512x7x7xf32>
    %v987 = stablehlo.subtract %v980, %v986 : tensor<64x512x7x7xf32>
    %v988 = stablehlo.multiply %v987, %v987 : tensor<64x512x7x7xf32>
    %v989 = stablehlo.reduce(%v988 init: %v981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v990 = stablehlo.broadcast_in_dim %v989, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v991 = stablehlo.divide %v990, %v982 : tensor<64x512x7x7xf32>
    %v992 = stablehlo.add %v991, %v983 : tensor<64x512x7x7xf32>
    %v993 = stablehlo.rsqrt %v992 : tensor<64x512x7x7xf32>
    %v994 = stablehlo.multiply %v987, %v993 : tensor<64x512x7x7xf32>
    %v995 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v996 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v997 = stablehlo.multiply %v994, %v995 : tensor<64x512x7x7xf32>
    %v998 = stablehlo.add %v997, %v996 : tensor<64x512x7x7xf32>
    %v999 = stablehlo.reshape %v998 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1000 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1001 = stablehlo.maximum %v999, %v1000 : tensor<64x25088xf32>
    %v1002 = stablehlo.reshape %v1001 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1003 = stablehlo.convert %v1002 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1004 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1005 = stablehlo.convolution(%v1003, %v1004)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1006 = stablehlo.convert %v1005 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1007 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1008 = stablehlo.add %v1006, %v1007 : tensor<64x512x7x7xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1010 = stablehlo.reshape %v1009 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1012 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1013 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1014 = stablehlo.reduce(%v1010 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1015 = stablehlo.broadcast_in_dim %v1014, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1016 = stablehlo.divide %v1015, %v1012 : tensor<64x512x7x7xf32>
    %v1017 = stablehlo.subtract %v1010, %v1016 : tensor<64x512x7x7xf32>
    %v1018 = stablehlo.multiply %v1017, %v1017 : tensor<64x512x7x7xf32>
    %v1019 = stablehlo.reduce(%v1018 init: %v1011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1020 = stablehlo.broadcast_in_dim %v1019, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1021 = stablehlo.divide %v1020, %v1012 : tensor<64x512x7x7xf32>
    %v1022 = stablehlo.add %v1021, %v1013 : tensor<64x512x7x7xf32>
    %v1023 = stablehlo.rsqrt %v1022 : tensor<64x512x7x7xf32>
    %v1024 = stablehlo.multiply %v1017, %v1023 : tensor<64x512x7x7xf32>
    %v1025 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1026 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1027 = stablehlo.multiply %v1024, %v1025 : tensor<64x512x7x7xf32>
    %v1028 = stablehlo.add %v1027, %v1026 : tensor<64x512x7x7xf32>
    %v1029 = stablehlo.reshape %v1028 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1030 = stablehlo.add %v1029, %v971 : tensor<64x25088xf32>
    %v1031 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1032 = stablehlo.maximum %v1030, %v1031 : tensor<64x25088xf32>
    %v1033 = stablehlo.reshape %v1032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1034 = stablehlo.convert %v1033 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1035 = stablehlo.convert %s4b1W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1036 = stablehlo.convolution(%v1034, %v1035)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1037 = stablehlo.convert %v1036 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1038 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1039 = stablehlo.add %v1037, %v1038 : tensor<64x512x7x7xf32>
    %v1040 = stablehlo.reshape %v1039 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1041 = stablehlo.reshape %v1040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1042 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1043 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1044 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1045 = stablehlo.reduce(%v1041 init: %v1042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1046 = stablehlo.broadcast_in_dim %v1045, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1047 = stablehlo.divide %v1046, %v1043 : tensor<64x512x7x7xf32>
    %v1048 = stablehlo.subtract %v1041, %v1047 : tensor<64x512x7x7xf32>
    %v1049 = stablehlo.multiply %v1048, %v1048 : tensor<64x512x7x7xf32>
    %v1050 = stablehlo.reduce(%v1049 init: %v1042) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1051 = stablehlo.broadcast_in_dim %v1050, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1052 = stablehlo.divide %v1051, %v1043 : tensor<64x512x7x7xf32>
    %v1053 = stablehlo.add %v1052, %v1044 : tensor<64x512x7x7xf32>
    %v1054 = stablehlo.rsqrt %v1053 : tensor<64x512x7x7xf32>
    %v1055 = stablehlo.multiply %v1048, %v1054 : tensor<64x512x7x7xf32>
    %v1056 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1057 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1058 = stablehlo.multiply %v1055, %v1056 : tensor<64x512x7x7xf32>
    %v1059 = stablehlo.add %v1058, %v1057 : tensor<64x512x7x7xf32>
    %v1060 = stablehlo.reshape %v1059 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1061 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1062 = stablehlo.maximum %v1060, %v1061 : tensor<64x25088xf32>
    %v1063 = stablehlo.reshape %v1062 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1064 = stablehlo.convert %v1063 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1065 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1066 = stablehlo.convolution(%v1064, %v1065)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1067 = stablehlo.convert %v1066 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1068 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1069 = stablehlo.add %v1067, %v1068 : tensor<64x512x7x7xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1071 = stablehlo.reshape %v1070 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1072 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1073 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1074 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1075 = stablehlo.reduce(%v1071 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1076 = stablehlo.broadcast_in_dim %v1075, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1077 = stablehlo.divide %v1076, %v1073 : tensor<64x512x7x7xf32>
    %v1078 = stablehlo.subtract %v1071, %v1077 : tensor<64x512x7x7xf32>
    %v1079 = stablehlo.multiply %v1078, %v1078 : tensor<64x512x7x7xf32>
    %v1080 = stablehlo.reduce(%v1079 init: %v1072) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1081 = stablehlo.broadcast_in_dim %v1080, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1082 = stablehlo.divide %v1081, %v1073 : tensor<64x512x7x7xf32>
    %v1083 = stablehlo.add %v1082, %v1074 : tensor<64x512x7x7xf32>
    %v1084 = stablehlo.rsqrt %v1083 : tensor<64x512x7x7xf32>
    %v1085 = stablehlo.multiply %v1078, %v1084 : tensor<64x512x7x7xf32>
    %v1086 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1087 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1088 = stablehlo.multiply %v1085, %v1086 : tensor<64x512x7x7xf32>
    %v1089 = stablehlo.add %v1088, %v1087 : tensor<64x512x7x7xf32>
    %v1090 = stablehlo.reshape %v1089 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1091 = stablehlo.add %v1090, %v1032 : tensor<64x25088xf32>
    %v1092 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1093 = stablehlo.maximum %v1091, %v1092 : tensor<64x25088xf32>
    %v1094 = stablehlo.reshape %v1093 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1096 = stablehlo.reduce(%v1094 init: %v1095) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512xf32>
    %v1097 = stablehlo.constant dense<49.0> : tensor<64x512xf32>
    %v1098 = stablehlo.divide %v1096, %v1097 : tensor<64x512xf32>
    %v1099 = stablehlo.dot_general %v1098, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x1000xf32>) -> tensor<64x1000xf32>
    %v1100 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1101 = stablehlo.add %v1099, %v1100 : tensor<64x1000xf32>
    %v1102 = stablehlo.reshape %v1101 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1104 = stablehlo.exponential %v1102 : tensor<64x1x1000xf32>
    %v1105 = stablehlo.reduce(%v1104 init: %v1103) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1106 = stablehlo.broadcast_in_dim %v1105, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v1107 = stablehlo.divide %v1104, %v1106 : tensor<64x1x1000xf32>
    %v1108 = stablehlo.reshape %v1107 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v1109 = stablehlo.subtract %v1108, %onehot : tensor<64x1000xf32>
    %v1110 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v1111 = stablehlo.multiply %onehot, %v1110 : tensor<64x1000xf32>
    %v1112 = stablehlo.add %v1109, %v1111 : tensor<64x1000xf32>
    %v1113 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v1114 = stablehlo.add %v1112, %v1113 : tensor<64x1000xf32>
    %v1115 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v1116 = stablehlo.divide %v1114, %v1115 : tensor<64x1000xf32>
    %v1117 = stablehlo.reshape %v1116 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1118 = stablehlo.dot_general %v1117, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<512x1000xf32>) -> tensor<64x1x512xf32>
    %v1119 = stablehlo.reshape %v1118 : (tensor<64x1x512xf32>) -> tensor<64x512xf32>
    %v1120 = stablehlo.dot_general %v1098, %v1116, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<64x1000xf32>) -> tensor<512x1000xf32>
    %v1121 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1122 = stablehlo.reduce(%v1116 init: %v1121) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1123 = stablehlo.broadcast_in_dim %v1119, dims = [0, 1] : (tensor<64x512xf32>) -> tensor<64x512x7x7xf32>
    %v1124 = stablehlo.constant dense<49.0> : tensor<64x512x7x7xf32>
    %v1125 = stablehlo.divide %v1123, %v1124 : tensor<64x512x7x7xf32>
    %v1126 = stablehlo.reshape %v1125 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1127 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1128 = stablehlo.compare GT, %v1091, %v1127 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1129 = stablehlo.select %v1128, %v1126, %v1127 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1130 = stablehlo.reshape %v1070 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1132 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1133 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1134 = stablehlo.reduce(%v1130 init: %v1131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1135 = stablehlo.broadcast_in_dim %v1134, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1136 = stablehlo.divide %v1135, %v1132 : tensor<64x512x7x7xf32>
    %v1137 = stablehlo.subtract %v1130, %v1136 : tensor<64x512x7x7xf32>
    %v1138 = stablehlo.multiply %v1137, %v1137 : tensor<64x512x7x7xf32>
    %v1139 = stablehlo.reduce(%v1138 init: %v1131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1140 = stablehlo.broadcast_in_dim %v1139, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1141 = stablehlo.divide %v1140, %v1132 : tensor<64x512x7x7xf32>
    %v1142 = stablehlo.add %v1141, %v1133 : tensor<64x512x7x7xf32>
    %v1143 = stablehlo.rsqrt %v1142 : tensor<64x512x7x7xf32>
    %v1144 = stablehlo.multiply %v1137, %v1143 : tensor<64x512x7x7xf32>
    %v1145 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1146 = stablehlo.reshape %v1129 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1147 = stablehlo.multiply %v1145, %v1146 : tensor<64x512x7x7xf32>
    %v1148 = stablehlo.reduce(%v1147 init: %v1131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1149 = stablehlo.broadcast_in_dim %v1148, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1150 = stablehlo.multiply %v1144, %v1147 : tensor<64x512x7x7xf32>
    %v1151 = stablehlo.reduce(%v1150 init: %v1131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1152 = stablehlo.broadcast_in_dim %v1151, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1153 = stablehlo.multiply %v1147, %v1132 : tensor<64x512x7x7xf32>
    %v1154 = stablehlo.subtract %v1153, %v1149 : tensor<64x512x7x7xf32>
    %v1155 = stablehlo.multiply %v1144, %v1152 : tensor<64x512x7x7xf32>
    %v1156 = stablehlo.subtract %v1154, %v1155 : tensor<64x512x7x7xf32>
    %v1157 = stablehlo.divide %v1143, %v1132 : tensor<64x512x7x7xf32>
    %v1158 = stablehlo.multiply %v1157, %v1156 : tensor<64x512x7x7xf32>
    %v1159 = stablehlo.reshape %v1158 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1160 = stablehlo.reshape %v1159 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1161 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1162 = stablehlo.transpose %v1161, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1163 = stablehlo.convert %v1160 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1164 = stablehlo.convert %v1162 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1165 = stablehlo.convolution(%v1163, %v1164)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1166 = stablehlo.convert %v1165 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1167 = stablehlo.reshape %v1166 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1168 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1169 = stablehlo.compare GT, %v1060, %v1168 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1170 = stablehlo.select %v1169, %v1167, %v1168 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1171 = stablehlo.reshape %v1040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1173 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1174 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1175 = stablehlo.reduce(%v1171 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1176 = stablehlo.broadcast_in_dim %v1175, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1177 = stablehlo.divide %v1176, %v1173 : tensor<64x512x7x7xf32>
    %v1178 = stablehlo.subtract %v1171, %v1177 : tensor<64x512x7x7xf32>
    %v1179 = stablehlo.multiply %v1178, %v1178 : tensor<64x512x7x7xf32>
    %v1180 = stablehlo.reduce(%v1179 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1181 = stablehlo.broadcast_in_dim %v1180, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1182 = stablehlo.divide %v1181, %v1173 : tensor<64x512x7x7xf32>
    %v1183 = stablehlo.add %v1182, %v1174 : tensor<64x512x7x7xf32>
    %v1184 = stablehlo.rsqrt %v1183 : tensor<64x512x7x7xf32>
    %v1185 = stablehlo.multiply %v1178, %v1184 : tensor<64x512x7x7xf32>
    %v1186 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1187 = stablehlo.reshape %v1170 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1188 = stablehlo.multiply %v1186, %v1187 : tensor<64x512x7x7xf32>
    %v1189 = stablehlo.reduce(%v1188 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1190 = stablehlo.broadcast_in_dim %v1189, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1191 = stablehlo.multiply %v1185, %v1188 : tensor<64x512x7x7xf32>
    %v1192 = stablehlo.reduce(%v1191 init: %v1172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1193 = stablehlo.broadcast_in_dim %v1192, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1194 = stablehlo.multiply %v1188, %v1173 : tensor<64x512x7x7xf32>
    %v1195 = stablehlo.subtract %v1194, %v1190 : tensor<64x512x7x7xf32>
    %v1196 = stablehlo.multiply %v1185, %v1193 : tensor<64x512x7x7xf32>
    %v1197 = stablehlo.subtract %v1195, %v1196 : tensor<64x512x7x7xf32>
    %v1198 = stablehlo.divide %v1184, %v1173 : tensor<64x512x7x7xf32>
    %v1199 = stablehlo.multiply %v1198, %v1197 : tensor<64x512x7x7xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1201 = stablehlo.reshape %v1200 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1202 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1203 = stablehlo.transpose %v1202, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1204 = stablehlo.convert %v1201 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1205 = stablehlo.convert %v1203 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1206 = stablehlo.convolution(%v1204, %v1205)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1207 = stablehlo.convert %v1206 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1209 = stablehlo.add %v1208, %v1129 : tensor<64x25088xf32>
    %v1210 = stablehlo.reshape %v1032 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1211 = stablehlo.reshape %v1200 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1212 = stablehlo.transpose %v1210, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1213 = stablehlo.transpose %v1211, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1214 = stablehlo.convert %v1212 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1215 = stablehlo.convert %v1213 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1216 = stablehlo.convolution(%v1214, %v1215)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1217 = stablehlo.convert %v1216 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1218 = stablehlo.transpose %v1217, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1219 = stablehlo.reshape %v1040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1220 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1221 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1222 = stablehlo.reduce(%v1219 init: %v1220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1223 = stablehlo.broadcast_in_dim %v1222, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1224 = stablehlo.divide %v1223, %v1221 : tensor<64x512x7x7xf32>
    %v1225 = stablehlo.subtract %v1219, %v1224 : tensor<64x512x7x7xf32>
    %v1226 = stablehlo.multiply %v1225, %v1225 : tensor<64x512x7x7xf32>
    %v1227 = stablehlo.reduce(%v1226 init: %v1220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1228 = stablehlo.broadcast_in_dim %v1227, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1229 = stablehlo.divide %v1228, %v1221 : tensor<64x512x7x7xf32>
    %v1230 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1231 = stablehlo.add %v1229, %v1230 : tensor<64x512x7x7xf32>
    %v1232 = stablehlo.rsqrt %v1231 : tensor<64x512x7x7xf32>
    %v1233 = stablehlo.multiply %v1225, %v1232 : tensor<64x512x7x7xf32>
    %v1234 = stablehlo.reshape %v1170 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1235 = stablehlo.multiply %v1234, %v1233 : tensor<64x512x7x7xf32>
    %v1236 = stablehlo.reduce(%v1235 init: %v1220) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1237 = stablehlo.reshape %v1170 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1238 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1239 = stablehlo.reduce(%v1237 init: %v1238) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1240 = stablehlo.reshape %v1062 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1241 = stablehlo.reshape %v1159 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1242 = stablehlo.transpose %v1240, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1243 = stablehlo.transpose %v1241, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1244 = stablehlo.convert %v1242 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1245 = stablehlo.convert %v1243 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1246 = stablehlo.convolution(%v1244, %v1245)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1247 = stablehlo.convert %v1246 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1248 = stablehlo.transpose %v1247, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1249 = stablehlo.reshape %v1070 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1251 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1252 = stablehlo.reduce(%v1249 init: %v1250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1253 = stablehlo.broadcast_in_dim %v1252, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1254 = stablehlo.divide %v1253, %v1251 : tensor<64x512x7x7xf32>
    %v1255 = stablehlo.subtract %v1249, %v1254 : tensor<64x512x7x7xf32>
    %v1256 = stablehlo.multiply %v1255, %v1255 : tensor<64x512x7x7xf32>
    %v1257 = stablehlo.reduce(%v1256 init: %v1250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1258 = stablehlo.broadcast_in_dim %v1257, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1259 = stablehlo.divide %v1258, %v1251 : tensor<64x512x7x7xf32>
    %v1260 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1261 = stablehlo.add %v1259, %v1260 : tensor<64x512x7x7xf32>
    %v1262 = stablehlo.rsqrt %v1261 : tensor<64x512x7x7xf32>
    %v1263 = stablehlo.multiply %v1255, %v1262 : tensor<64x512x7x7xf32>
    %v1264 = stablehlo.reshape %v1129 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1265 = stablehlo.multiply %v1264, %v1263 : tensor<64x512x7x7xf32>
    %v1266 = stablehlo.reduce(%v1265 init: %v1250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1267 = stablehlo.reshape %v1129 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1269 = stablehlo.reduce(%v1267 init: %v1268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1270 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1271 = stablehlo.compare GT, %v1030, %v1270 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1272 = stablehlo.select %v1271, %v1209, %v1270 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1273 = stablehlo.reshape %v1009 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1275 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1276 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1277 = stablehlo.reduce(%v1273 init: %v1274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1278 = stablehlo.broadcast_in_dim %v1277, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1279 = stablehlo.divide %v1278, %v1275 : tensor<64x512x7x7xf32>
    %v1280 = stablehlo.subtract %v1273, %v1279 : tensor<64x512x7x7xf32>
    %v1281 = stablehlo.multiply %v1280, %v1280 : tensor<64x512x7x7xf32>
    %v1282 = stablehlo.reduce(%v1281 init: %v1274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1283 = stablehlo.broadcast_in_dim %v1282, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1284 = stablehlo.divide %v1283, %v1275 : tensor<64x512x7x7xf32>
    %v1285 = stablehlo.add %v1284, %v1276 : tensor<64x512x7x7xf32>
    %v1286 = stablehlo.rsqrt %v1285 : tensor<64x512x7x7xf32>
    %v1287 = stablehlo.multiply %v1280, %v1286 : tensor<64x512x7x7xf32>
    %v1288 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1289 = stablehlo.reshape %v1272 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1290 = stablehlo.multiply %v1288, %v1289 : tensor<64x512x7x7xf32>
    %v1291 = stablehlo.reduce(%v1290 init: %v1274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1292 = stablehlo.broadcast_in_dim %v1291, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1293 = stablehlo.multiply %v1287, %v1290 : tensor<64x512x7x7xf32>
    %v1294 = stablehlo.reduce(%v1293 init: %v1274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1295 = stablehlo.broadcast_in_dim %v1294, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1296 = stablehlo.multiply %v1290, %v1275 : tensor<64x512x7x7xf32>
    %v1297 = stablehlo.subtract %v1296, %v1292 : tensor<64x512x7x7xf32>
    %v1298 = stablehlo.multiply %v1287, %v1295 : tensor<64x512x7x7xf32>
    %v1299 = stablehlo.subtract %v1297, %v1298 : tensor<64x512x7x7xf32>
    %v1300 = stablehlo.divide %v1286, %v1275 : tensor<64x512x7x7xf32>
    %v1301 = stablehlo.multiply %v1300, %v1299 : tensor<64x512x7x7xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1303 = stablehlo.reshape %v1302 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1304 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1305 = stablehlo.transpose %v1304, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1306 = stablehlo.convert %v1303 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1307 = stablehlo.convert %v1305 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1308 = stablehlo.convolution(%v1306, %v1307)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1309 = stablehlo.convert %v1308 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1310 = stablehlo.reshape %v1309 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1311 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1312 = stablehlo.compare GT, %v999, %v1311 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1313 = stablehlo.select %v1312, %v1310, %v1311 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1314 = stablehlo.reshape %v979 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1316 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1317 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1318 = stablehlo.reduce(%v1314 init: %v1315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1319 = stablehlo.broadcast_in_dim %v1318, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1320 = stablehlo.divide %v1319, %v1316 : tensor<64x512x7x7xf32>
    %v1321 = stablehlo.subtract %v1314, %v1320 : tensor<64x512x7x7xf32>
    %v1322 = stablehlo.multiply %v1321, %v1321 : tensor<64x512x7x7xf32>
    %v1323 = stablehlo.reduce(%v1322 init: %v1315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1324 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1325 = stablehlo.divide %v1324, %v1316 : tensor<64x512x7x7xf32>
    %v1326 = stablehlo.add %v1325, %v1317 : tensor<64x512x7x7xf32>
    %v1327 = stablehlo.rsqrt %v1326 : tensor<64x512x7x7xf32>
    %v1328 = stablehlo.multiply %v1321, %v1327 : tensor<64x512x7x7xf32>
    %v1329 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1330 = stablehlo.reshape %v1313 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1331 = stablehlo.multiply %v1329, %v1330 : tensor<64x512x7x7xf32>
    %v1332 = stablehlo.reduce(%v1331 init: %v1315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1333 = stablehlo.broadcast_in_dim %v1332, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1334 = stablehlo.multiply %v1328, %v1331 : tensor<64x512x7x7xf32>
    %v1335 = stablehlo.reduce(%v1334 init: %v1315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1336 = stablehlo.broadcast_in_dim %v1335, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1337 = stablehlo.multiply %v1331, %v1316 : tensor<64x512x7x7xf32>
    %v1338 = stablehlo.subtract %v1337, %v1333 : tensor<64x512x7x7xf32>
    %v1339 = stablehlo.multiply %v1328, %v1336 : tensor<64x512x7x7xf32>
    %v1340 = stablehlo.subtract %v1338, %v1339 : tensor<64x512x7x7xf32>
    %v1341 = stablehlo.divide %v1327, %v1316 : tensor<64x512x7x7xf32>
    %v1342 = stablehlo.multiply %v1341, %v1340 : tensor<64x512x7x7xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1345 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1346 = stablehlo.transpose %v1345, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1347 = stablehlo.convert %v1344 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1348 = stablehlo.convert %v1346 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1349 = stablehlo.convolution(%v1347, %v1348)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1350 = stablehlo.convert %v1349 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1352 = stablehlo.add %v1351, %v1272 : tensor<64x25088xf32>
    %v1353 = stablehlo.reshape %v971 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1354 = stablehlo.reshape %v1343 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1355 = stablehlo.transpose %v1353, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1356 = stablehlo.transpose %v1354, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1357 = stablehlo.convert %v1355 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1358 = stablehlo.convert %v1356 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1359 = stablehlo.convolution(%v1357, %v1358)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1360 = stablehlo.convert %v1359 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1361 = stablehlo.transpose %v1360, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1362 = stablehlo.reshape %v979 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1363 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1364 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1365 = stablehlo.reduce(%v1362 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1366 = stablehlo.broadcast_in_dim %v1365, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1367 = stablehlo.divide %v1366, %v1364 : tensor<64x512x7x7xf32>
    %v1368 = stablehlo.subtract %v1362, %v1367 : tensor<64x512x7x7xf32>
    %v1369 = stablehlo.multiply %v1368, %v1368 : tensor<64x512x7x7xf32>
    %v1370 = stablehlo.reduce(%v1369 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1371 = stablehlo.broadcast_in_dim %v1370, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1372 = stablehlo.divide %v1371, %v1364 : tensor<64x512x7x7xf32>
    %v1373 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1374 = stablehlo.add %v1372, %v1373 : tensor<64x512x7x7xf32>
    %v1375 = stablehlo.rsqrt %v1374 : tensor<64x512x7x7xf32>
    %v1376 = stablehlo.multiply %v1368, %v1375 : tensor<64x512x7x7xf32>
    %v1377 = stablehlo.reshape %v1313 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1378 = stablehlo.multiply %v1377, %v1376 : tensor<64x512x7x7xf32>
    %v1379 = stablehlo.reduce(%v1378 init: %v1363) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1380 = stablehlo.reshape %v1313 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1382 = stablehlo.reduce(%v1380 init: %v1381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1383 = stablehlo.reshape %v1001 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1384 = stablehlo.reshape %v1302 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1385 = stablehlo.transpose %v1383, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1386 = stablehlo.transpose %v1384, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1387 = stablehlo.convert %v1385 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1388 = stablehlo.convert %v1386 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1389 = stablehlo.convolution(%v1387, %v1388)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1390 = stablehlo.convert %v1389 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1391 = stablehlo.transpose %v1390, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1392 = stablehlo.reshape %v1009 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1393 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1394 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1395 = stablehlo.reduce(%v1392 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1396 = stablehlo.broadcast_in_dim %v1395, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1397 = stablehlo.divide %v1396, %v1394 : tensor<64x512x7x7xf32>
    %v1398 = stablehlo.subtract %v1392, %v1397 : tensor<64x512x7x7xf32>
    %v1399 = stablehlo.multiply %v1398, %v1398 : tensor<64x512x7x7xf32>
    %v1400 = stablehlo.reduce(%v1399 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1401 = stablehlo.broadcast_in_dim %v1400, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1402 = stablehlo.divide %v1401, %v1394 : tensor<64x512x7x7xf32>
    %v1403 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1404 = stablehlo.add %v1402, %v1403 : tensor<64x512x7x7xf32>
    %v1405 = stablehlo.rsqrt %v1404 : tensor<64x512x7x7xf32>
    %v1406 = stablehlo.multiply %v1398, %v1405 : tensor<64x512x7x7xf32>
    %v1407 = stablehlo.reshape %v1272 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1408 = stablehlo.multiply %v1407, %v1406 : tensor<64x512x7x7xf32>
    %v1409 = stablehlo.reduce(%v1408 init: %v1393) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1410 = stablehlo.reshape %v1272 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1412 = stablehlo.reduce(%v1410 init: %v1411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1413 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1414 = stablehlo.compare GT, %v969, %v1413 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1415 = stablehlo.select %v1414, %v1352, %v1413 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1416 = stablehlo.reshape %v920 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1417 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1418 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1419 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1420 = stablehlo.reduce(%v1416 init: %v1417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1421 = stablehlo.broadcast_in_dim %v1420, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1422 = stablehlo.divide %v1421, %v1418 : tensor<64x512x7x7xf32>
    %v1423 = stablehlo.subtract %v1416, %v1422 : tensor<64x512x7x7xf32>
    %v1424 = stablehlo.multiply %v1423, %v1423 : tensor<64x512x7x7xf32>
    %v1425 = stablehlo.reduce(%v1424 init: %v1417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1426 = stablehlo.broadcast_in_dim %v1425, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1427 = stablehlo.divide %v1426, %v1418 : tensor<64x512x7x7xf32>
    %v1428 = stablehlo.add %v1427, %v1419 : tensor<64x512x7x7xf32>
    %v1429 = stablehlo.rsqrt %v1428 : tensor<64x512x7x7xf32>
    %v1430 = stablehlo.multiply %v1423, %v1429 : tensor<64x512x7x7xf32>
    %v1431 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1432 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1433 = stablehlo.multiply %v1431, %v1432 : tensor<64x512x7x7xf32>
    %v1434 = stablehlo.reduce(%v1433 init: %v1417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1435 = stablehlo.broadcast_in_dim %v1434, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1436 = stablehlo.multiply %v1430, %v1433 : tensor<64x512x7x7xf32>
    %v1437 = stablehlo.reduce(%v1436 init: %v1417) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1438 = stablehlo.broadcast_in_dim %v1437, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1439 = stablehlo.multiply %v1433, %v1418 : tensor<64x512x7x7xf32>
    %v1440 = stablehlo.subtract %v1439, %v1435 : tensor<64x512x7x7xf32>
    %v1441 = stablehlo.multiply %v1430, %v1438 : tensor<64x512x7x7xf32>
    %v1442 = stablehlo.subtract %v1440, %v1441 : tensor<64x512x7x7xf32>
    %v1443 = stablehlo.divide %v1429, %v1418 : tensor<64x512x7x7xf32>
    %v1444 = stablehlo.multiply %v1443, %v1442 : tensor<64x512x7x7xf32>
    %v1445 = stablehlo.reshape %v1444 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1446 = stablehlo.reshape %v1445 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1447 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1448 = stablehlo.transpose %v1447, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1449 = stablehlo.convert %v1446 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1450 = stablehlo.convert %v1448 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1451 = stablehlo.convolution(%v1449, %v1450)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1452 = stablehlo.convert %v1451 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1453 = stablehlo.reshape %v1452 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1454 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1455 = stablehlo.compare GT, %v910, %v1454 : (tensor<64x25088xf32>, tensor<64x25088xf32>) -> tensor<64x25088xi1>
    %v1456 = stablehlo.select %v1455, %v1453, %v1454 : tensor<64x25088xi1>, tensor<64x25088xf32>
    %v1457 = stablehlo.reshape %v890 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1458 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1459 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1460 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1461 = stablehlo.reduce(%v1457 init: %v1458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1462 = stablehlo.broadcast_in_dim %v1461, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1463 = stablehlo.divide %v1462, %v1459 : tensor<64x512x7x7xf32>
    %v1464 = stablehlo.subtract %v1457, %v1463 : tensor<64x512x7x7xf32>
    %v1465 = stablehlo.multiply %v1464, %v1464 : tensor<64x512x7x7xf32>
    %v1466 = stablehlo.reduce(%v1465 init: %v1458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1467 = stablehlo.broadcast_in_dim %v1466, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1468 = stablehlo.divide %v1467, %v1459 : tensor<64x512x7x7xf32>
    %v1469 = stablehlo.add %v1468, %v1460 : tensor<64x512x7x7xf32>
    %v1470 = stablehlo.rsqrt %v1469 : tensor<64x512x7x7xf32>
    %v1471 = stablehlo.multiply %v1464, %v1470 : tensor<64x512x7x7xf32>
    %v1472 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1473 = stablehlo.reshape %v1456 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1474 = stablehlo.multiply %v1472, %v1473 : tensor<64x512x7x7xf32>
    %v1475 = stablehlo.reduce(%v1474 init: %v1458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1476 = stablehlo.broadcast_in_dim %v1475, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1477 = stablehlo.multiply %v1471, %v1474 : tensor<64x512x7x7xf32>
    %v1478 = stablehlo.reduce(%v1477 init: %v1458) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1479 = stablehlo.broadcast_in_dim %v1478, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1480 = stablehlo.multiply %v1474, %v1459 : tensor<64x512x7x7xf32>
    %v1481 = stablehlo.subtract %v1480, %v1476 : tensor<64x512x7x7xf32>
    %v1482 = stablehlo.multiply %v1471, %v1479 : tensor<64x512x7x7xf32>
    %v1483 = stablehlo.subtract %v1481, %v1482 : tensor<64x512x7x7xf32>
    %v1484 = stablehlo.divide %v1470, %v1459 : tensor<64x512x7x7xf32>
    %v1485 = stablehlo.multiply %v1484, %v1483 : tensor<64x512x7x7xf32>
    %v1486 = stablehlo.reshape %v1485 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1487 = stablehlo.reshape %v1486 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1488 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1489 = stablehlo.pad %v1487, %v1488, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v1490 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v1491 = stablehlo.transpose %v1490, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v1492 = stablehlo.convert %v1489 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v1493 = stablehlo.convert %v1491 : (tensor<256x512x3x3xf32>) -> tensor<256x512x3x3xbf16>
    %v1494 = stablehlo.convolution(%v1492, %v1493)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<256x512x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1495 = stablehlo.convert %v1494 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1496 = stablehlo.reshape %v1495 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1497 = stablehlo.reshape %v948 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1499 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1500 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1501 = stablehlo.reduce(%v1497 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1502 = stablehlo.broadcast_in_dim %v1501, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1503 = stablehlo.divide %v1502, %v1499 : tensor<64x512x7x7xf32>
    %v1504 = stablehlo.subtract %v1497, %v1503 : tensor<64x512x7x7xf32>
    %v1505 = stablehlo.multiply %v1504, %v1504 : tensor<64x512x7x7xf32>
    %v1506 = stablehlo.reduce(%v1505 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1507 = stablehlo.broadcast_in_dim %v1506, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1508 = stablehlo.divide %v1507, %v1499 : tensor<64x512x7x7xf32>
    %v1509 = stablehlo.add %v1508, %v1500 : tensor<64x512x7x7xf32>
    %v1510 = stablehlo.rsqrt %v1509 : tensor<64x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1504, %v1510 : tensor<64x512x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1513 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1514 = stablehlo.multiply %v1512, %v1513 : tensor<64x512x7x7xf32>
    %v1515 = stablehlo.reduce(%v1514 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1516 = stablehlo.broadcast_in_dim %v1515, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1517 = stablehlo.multiply %v1511, %v1514 : tensor<64x512x7x7xf32>
    %v1518 = stablehlo.reduce(%v1517 init: %v1498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1519 = stablehlo.broadcast_in_dim %v1518, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1520 = stablehlo.multiply %v1514, %v1499 : tensor<64x512x7x7xf32>
    %v1521 = stablehlo.subtract %v1520, %v1516 : tensor<64x512x7x7xf32>
    %v1522 = stablehlo.multiply %v1511, %v1519 : tensor<64x512x7x7xf32>
    %v1523 = stablehlo.subtract %v1521, %v1522 : tensor<64x512x7x7xf32>
    %v1524 = stablehlo.divide %v1510, %v1499 : tensor<64x512x7x7xf32>
    %v1525 = stablehlo.multiply %v1524, %v1523 : tensor<64x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.pad %v1527, %v1528, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v1530 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v1531 = stablehlo.transpose %v1530, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v1532 = stablehlo.convert %v1529 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v1533 = stablehlo.convert %v1531 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v1534 = stablehlo.convolution(%v1532, %v1533)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v1535 = stablehlo.convert %v1534 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1536 = stablehlo.reshape %v1535 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1537 = stablehlo.add %v1496, %v1536 : tensor<64x50176xf32>
    %v1538 = stablehlo.reshape %v882 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1539 = stablehlo.reshape %v1486 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1540 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1541 = stablehlo.pad %v1539, %v1540, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v1542 = stablehlo.transpose %v1538, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1543 = stablehlo.transpose %v1541, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v1544 = stablehlo.convert %v1542 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1545 = stablehlo.convert %v1543 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v1546 = stablehlo.convolution(%v1544, %v1545)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<256x512x3x3xbf16>
    %v1547 = stablehlo.convert %v1546 : (tensor<256x512x3x3xbf16>) -> tensor<256x512x3x3xf32>
    %v1548 = stablehlo.transpose %v1547, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v1549 = stablehlo.reshape %v890 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1550 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1551 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1552 = stablehlo.reduce(%v1549 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1553 = stablehlo.broadcast_in_dim %v1552, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1554 = stablehlo.divide %v1553, %v1551 : tensor<64x512x7x7xf32>
    %v1555 = stablehlo.subtract %v1549, %v1554 : tensor<64x512x7x7xf32>
    %v1556 = stablehlo.multiply %v1555, %v1555 : tensor<64x512x7x7xf32>
    %v1557 = stablehlo.reduce(%v1556 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1558 = stablehlo.broadcast_in_dim %v1557, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1559 = stablehlo.divide %v1558, %v1551 : tensor<64x512x7x7xf32>
    %v1560 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1561 = stablehlo.add %v1559, %v1560 : tensor<64x512x7x7xf32>
    %v1562 = stablehlo.rsqrt %v1561 : tensor<64x512x7x7xf32>
    %v1563 = stablehlo.multiply %v1555, %v1562 : tensor<64x512x7x7xf32>
    %v1564 = stablehlo.reshape %v1456 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1565 = stablehlo.multiply %v1564, %v1563 : tensor<64x512x7x7xf32>
    %v1566 = stablehlo.reduce(%v1565 init: %v1550) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1567 = stablehlo.reshape %v1456 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1568 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1569 = stablehlo.reduce(%v1567 init: %v1568) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1570 = stablehlo.reshape %v912 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1571 = stablehlo.reshape %v1445 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1572 = stablehlo.transpose %v1570, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1573 = stablehlo.transpose %v1571, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1574 = stablehlo.convert %v1572 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1575 = stablehlo.convert %v1573 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1576 = stablehlo.convolution(%v1574, %v1575)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1577 = stablehlo.convert %v1576 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1578 = stablehlo.transpose %v1577, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1579 = stablehlo.reshape %v920 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1580 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1581 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1582 = stablehlo.reduce(%v1579 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1583 = stablehlo.broadcast_in_dim %v1582, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1584 = stablehlo.divide %v1583, %v1581 : tensor<64x512x7x7xf32>
    %v1585 = stablehlo.subtract %v1579, %v1584 : tensor<64x512x7x7xf32>
    %v1586 = stablehlo.multiply %v1585, %v1585 : tensor<64x512x7x7xf32>
    %v1587 = stablehlo.reduce(%v1586 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1588 = stablehlo.broadcast_in_dim %v1587, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1589 = stablehlo.divide %v1588, %v1581 : tensor<64x512x7x7xf32>
    %v1590 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1591 = stablehlo.add %v1589, %v1590 : tensor<64x512x7x7xf32>
    %v1592 = stablehlo.rsqrt %v1591 : tensor<64x512x7x7xf32>
    %v1593 = stablehlo.multiply %v1585, %v1592 : tensor<64x512x7x7xf32>
    %v1594 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1595 = stablehlo.multiply %v1594, %v1593 : tensor<64x512x7x7xf32>
    %v1596 = stablehlo.reduce(%v1595 init: %v1580) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1597 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1598 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1599 = stablehlo.reduce(%v1597 init: %v1598) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1600 = stablehlo.reshape %v882 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1601 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1602 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1603 = stablehlo.pad %v1601, %v1602, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v1604 = stablehlo.transpose %v1600, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1605 = stablehlo.transpose %v1603, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v1606 = stablehlo.convert %v1604 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1607 = stablehlo.convert %v1605 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v1608 = stablehlo.convolution(%v1606, %v1607)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<256x512x1x1xbf16>
    %v1609 = stablehlo.convert %v1608 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v1610 = stablehlo.transpose %v1609, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v1611 = stablehlo.reshape %v948 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1612 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1613 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v1614 = stablehlo.reduce(%v1611 init: %v1612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1615 = stablehlo.broadcast_in_dim %v1614, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1616 = stablehlo.divide %v1615, %v1613 : tensor<64x512x7x7xf32>
    %v1617 = stablehlo.subtract %v1611, %v1616 : tensor<64x512x7x7xf32>
    %v1618 = stablehlo.multiply %v1617, %v1617 : tensor<64x512x7x7xf32>
    %v1619 = stablehlo.reduce(%v1618 init: %v1612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1620 = stablehlo.broadcast_in_dim %v1619, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1621 = stablehlo.divide %v1620, %v1613 : tensor<64x512x7x7xf32>
    %v1622 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1623 = stablehlo.add %v1621, %v1622 : tensor<64x512x7x7xf32>
    %v1624 = stablehlo.rsqrt %v1623 : tensor<64x512x7x7xf32>
    %v1625 = stablehlo.multiply %v1617, %v1624 : tensor<64x512x7x7xf32>
    %v1626 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1627 = stablehlo.multiply %v1626, %v1625 : tensor<64x512x7x7xf32>
    %v1628 = stablehlo.reduce(%v1627 init: %v1612) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1629 = stablehlo.reshape %v1415 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1630 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1631 = stablehlo.reduce(%v1629 init: %v1630) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1632 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1633 = stablehlo.compare GT, %v880, %v1632 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1634 = stablehlo.select %v1633, %v1537, %v1632 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1635 = stablehlo.reshape %v859 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1636 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1637 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1638 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1639 = stablehlo.reduce(%v1635 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1640 = stablehlo.broadcast_in_dim %v1639, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1641 = stablehlo.divide %v1640, %v1637 : tensor<64x256x14x14xf32>
    %v1642 = stablehlo.subtract %v1635, %v1641 : tensor<64x256x14x14xf32>
    %v1643 = stablehlo.multiply %v1642, %v1642 : tensor<64x256x14x14xf32>
    %v1644 = stablehlo.reduce(%v1643 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1645 = stablehlo.broadcast_in_dim %v1644, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1646 = stablehlo.divide %v1645, %v1637 : tensor<64x256x14x14xf32>
    %v1647 = stablehlo.add %v1646, %v1638 : tensor<64x256x14x14xf32>
    %v1648 = stablehlo.rsqrt %v1647 : tensor<64x256x14x14xf32>
    %v1649 = stablehlo.multiply %v1642, %v1648 : tensor<64x256x14x14xf32>
    %v1650 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1651 = stablehlo.reshape %v1634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1652 = stablehlo.multiply %v1650, %v1651 : tensor<64x256x14x14xf32>
    %v1653 = stablehlo.reduce(%v1652 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1654 = stablehlo.broadcast_in_dim %v1653, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1655 = stablehlo.multiply %v1649, %v1652 : tensor<64x256x14x14xf32>
    %v1656 = stablehlo.reduce(%v1655 init: %v1636) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1657 = stablehlo.broadcast_in_dim %v1656, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1658 = stablehlo.multiply %v1652, %v1637 : tensor<64x256x14x14xf32>
    %v1659 = stablehlo.subtract %v1658, %v1654 : tensor<64x256x14x14xf32>
    %v1660 = stablehlo.multiply %v1649, %v1657 : tensor<64x256x14x14xf32>
    %v1661 = stablehlo.subtract %v1659, %v1660 : tensor<64x256x14x14xf32>
    %v1662 = stablehlo.divide %v1648, %v1637 : tensor<64x256x14x14xf32>
    %v1663 = stablehlo.multiply %v1662, %v1661 : tensor<64x256x14x14xf32>
    %v1664 = stablehlo.reshape %v1663 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1665 = stablehlo.reshape %v1664 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1666 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1667 = stablehlo.transpose %v1666, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1668 = stablehlo.convert %v1665 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1669 = stablehlo.convert %v1667 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1670 = stablehlo.convolution(%v1668, %v1669)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1671 = stablehlo.convert %v1670 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1672 = stablehlo.reshape %v1671 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1673 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1674 = stablehlo.compare GT, %v849, %v1673 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1675 = stablehlo.select %v1674, %v1672, %v1673 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1676 = stablehlo.reshape %v829 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1678 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1679 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1680 = stablehlo.reduce(%v1676 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1681 = stablehlo.broadcast_in_dim %v1680, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1682 = stablehlo.divide %v1681, %v1678 : tensor<64x256x14x14xf32>
    %v1683 = stablehlo.subtract %v1676, %v1682 : tensor<64x256x14x14xf32>
    %v1684 = stablehlo.multiply %v1683, %v1683 : tensor<64x256x14x14xf32>
    %v1685 = stablehlo.reduce(%v1684 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1686 = stablehlo.broadcast_in_dim %v1685, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1687 = stablehlo.divide %v1686, %v1678 : tensor<64x256x14x14xf32>
    %v1688 = stablehlo.add %v1687, %v1679 : tensor<64x256x14x14xf32>
    %v1689 = stablehlo.rsqrt %v1688 : tensor<64x256x14x14xf32>
    %v1690 = stablehlo.multiply %v1683, %v1689 : tensor<64x256x14x14xf32>
    %v1691 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1692 = stablehlo.reshape %v1675 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1693 = stablehlo.multiply %v1691, %v1692 : tensor<64x256x14x14xf32>
    %v1694 = stablehlo.reduce(%v1693 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1695 = stablehlo.broadcast_in_dim %v1694, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1696 = stablehlo.multiply %v1690, %v1693 : tensor<64x256x14x14xf32>
    %v1697 = stablehlo.reduce(%v1696 init: %v1677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1698 = stablehlo.broadcast_in_dim %v1697, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1699 = stablehlo.multiply %v1693, %v1678 : tensor<64x256x14x14xf32>
    %v1700 = stablehlo.subtract %v1699, %v1695 : tensor<64x256x14x14xf32>
    %v1701 = stablehlo.multiply %v1690, %v1698 : tensor<64x256x14x14xf32>
    %v1702 = stablehlo.subtract %v1700, %v1701 : tensor<64x256x14x14xf32>
    %v1703 = stablehlo.divide %v1689, %v1678 : tensor<64x256x14x14xf32>
    %v1704 = stablehlo.multiply %v1703, %v1702 : tensor<64x256x14x14xf32>
    %v1705 = stablehlo.reshape %v1704 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1706 = stablehlo.reshape %v1705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1707 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1708 = stablehlo.transpose %v1707, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1709 = stablehlo.convert %v1706 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1710 = stablehlo.convert %v1708 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1711 = stablehlo.convolution(%v1709, %v1710)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1712 = stablehlo.convert %v1711 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1713 = stablehlo.reshape %v1712 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1714 = stablehlo.add %v1713, %v1634 : tensor<64x50176xf32>
    %v1715 = stablehlo.reshape %v821 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1716 = stablehlo.reshape %v1705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1717 = stablehlo.transpose %v1715, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1718 = stablehlo.transpose %v1716, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1719 = stablehlo.convert %v1717 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1720 = stablehlo.convert %v1718 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1721 = stablehlo.convolution(%v1719, %v1720)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1722 = stablehlo.convert %v1721 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1723 = stablehlo.transpose %v1722, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1724 = stablehlo.reshape %v829 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1725 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1726 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1727 = stablehlo.reduce(%v1724 init: %v1725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1728 = stablehlo.broadcast_in_dim %v1727, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1729 = stablehlo.divide %v1728, %v1726 : tensor<64x256x14x14xf32>
    %v1730 = stablehlo.subtract %v1724, %v1729 : tensor<64x256x14x14xf32>
    %v1731 = stablehlo.multiply %v1730, %v1730 : tensor<64x256x14x14xf32>
    %v1732 = stablehlo.reduce(%v1731 init: %v1725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1733 = stablehlo.broadcast_in_dim %v1732, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1734 = stablehlo.divide %v1733, %v1726 : tensor<64x256x14x14xf32>
    %v1735 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1736 = stablehlo.add %v1734, %v1735 : tensor<64x256x14x14xf32>
    %v1737 = stablehlo.rsqrt %v1736 : tensor<64x256x14x14xf32>
    %v1738 = stablehlo.multiply %v1730, %v1737 : tensor<64x256x14x14xf32>
    %v1739 = stablehlo.reshape %v1675 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1740 = stablehlo.multiply %v1739, %v1738 : tensor<64x256x14x14xf32>
    %v1741 = stablehlo.reduce(%v1740 init: %v1725) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1742 = stablehlo.reshape %v1675 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1743 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1744 = stablehlo.reduce(%v1742 init: %v1743) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1745 = stablehlo.reshape %v851 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1746 = stablehlo.reshape %v1664 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1747 = stablehlo.transpose %v1745, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1748 = stablehlo.transpose %v1746, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1749 = stablehlo.convert %v1747 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1750 = stablehlo.convert %v1748 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1751 = stablehlo.convolution(%v1749, %v1750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1752 = stablehlo.convert %v1751 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1753 = stablehlo.transpose %v1752, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1754 = stablehlo.reshape %v859 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1756 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1757 = stablehlo.reduce(%v1754 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1758 = stablehlo.broadcast_in_dim %v1757, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1759 = stablehlo.divide %v1758, %v1756 : tensor<64x256x14x14xf32>
    %v1760 = stablehlo.subtract %v1754, %v1759 : tensor<64x256x14x14xf32>
    %v1761 = stablehlo.multiply %v1760, %v1760 : tensor<64x256x14x14xf32>
    %v1762 = stablehlo.reduce(%v1761 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1764 = stablehlo.divide %v1763, %v1756 : tensor<64x256x14x14xf32>
    %v1765 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1766 = stablehlo.add %v1764, %v1765 : tensor<64x256x14x14xf32>
    %v1767 = stablehlo.rsqrt %v1766 : tensor<64x256x14x14xf32>
    %v1768 = stablehlo.multiply %v1760, %v1767 : tensor<64x256x14x14xf32>
    %v1769 = stablehlo.reshape %v1634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1770 = stablehlo.multiply %v1769, %v1768 : tensor<64x256x14x14xf32>
    %v1771 = stablehlo.reduce(%v1770 init: %v1755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1772 = stablehlo.reshape %v1634 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.reduce(%v1772 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1775 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1776 = stablehlo.compare GT, %v819, %v1775 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1777 = stablehlo.select %v1776, %v1714, %v1775 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1778 = stablehlo.reshape %v798 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1780 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1781 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1782 = stablehlo.reduce(%v1778 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1783 = stablehlo.broadcast_in_dim %v1782, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1784 = stablehlo.divide %v1783, %v1780 : tensor<64x256x14x14xf32>
    %v1785 = stablehlo.subtract %v1778, %v1784 : tensor<64x256x14x14xf32>
    %v1786 = stablehlo.multiply %v1785, %v1785 : tensor<64x256x14x14xf32>
    %v1787 = stablehlo.reduce(%v1786 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1787, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1789 = stablehlo.divide %v1788, %v1780 : tensor<64x256x14x14xf32>
    %v1790 = stablehlo.add %v1789, %v1781 : tensor<64x256x14x14xf32>
    %v1791 = stablehlo.rsqrt %v1790 : tensor<64x256x14x14xf32>
    %v1792 = stablehlo.multiply %v1785, %v1791 : tensor<64x256x14x14xf32>
    %v1793 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1794 = stablehlo.reshape %v1777 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1795 = stablehlo.multiply %v1793, %v1794 : tensor<64x256x14x14xf32>
    %v1796 = stablehlo.reduce(%v1795 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1797 = stablehlo.broadcast_in_dim %v1796, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1798 = stablehlo.multiply %v1792, %v1795 : tensor<64x256x14x14xf32>
    %v1799 = stablehlo.reduce(%v1798 init: %v1779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1800 = stablehlo.broadcast_in_dim %v1799, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1801 = stablehlo.multiply %v1795, %v1780 : tensor<64x256x14x14xf32>
    %v1802 = stablehlo.subtract %v1801, %v1797 : tensor<64x256x14x14xf32>
    %v1803 = stablehlo.multiply %v1792, %v1800 : tensor<64x256x14x14xf32>
    %v1804 = stablehlo.subtract %v1802, %v1803 : tensor<64x256x14x14xf32>
    %v1805 = stablehlo.divide %v1791, %v1780 : tensor<64x256x14x14xf32>
    %v1806 = stablehlo.multiply %v1805, %v1804 : tensor<64x256x14x14xf32>
    %v1807 = stablehlo.reshape %v1806 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1808 = stablehlo.reshape %v1807 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1809 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1810 = stablehlo.transpose %v1809, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1811 = stablehlo.convert %v1808 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1812 = stablehlo.convert %v1810 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1813 = stablehlo.convolution(%v1811, %v1812)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1814 = stablehlo.convert %v1813 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1816 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1817 = stablehlo.compare GT, %v788, %v1816 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1818 = stablehlo.select %v1817, %v1815, %v1816 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1819 = stablehlo.reshape %v768 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1821 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1822 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1823 = stablehlo.reduce(%v1819 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1824 = stablehlo.broadcast_in_dim %v1823, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1825 = stablehlo.divide %v1824, %v1821 : tensor<64x256x14x14xf32>
    %v1826 = stablehlo.subtract %v1819, %v1825 : tensor<64x256x14x14xf32>
    %v1827 = stablehlo.multiply %v1826, %v1826 : tensor<64x256x14x14xf32>
    %v1828 = stablehlo.reduce(%v1827 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1828, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1830 = stablehlo.divide %v1829, %v1821 : tensor<64x256x14x14xf32>
    %v1831 = stablehlo.add %v1830, %v1822 : tensor<64x256x14x14xf32>
    %v1832 = stablehlo.rsqrt %v1831 : tensor<64x256x14x14xf32>
    %v1833 = stablehlo.multiply %v1826, %v1832 : tensor<64x256x14x14xf32>
    %v1834 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1835 = stablehlo.reshape %v1818 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1836 = stablehlo.multiply %v1834, %v1835 : tensor<64x256x14x14xf32>
    %v1837 = stablehlo.reduce(%v1836 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1838 = stablehlo.broadcast_in_dim %v1837, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1839 = stablehlo.multiply %v1833, %v1836 : tensor<64x256x14x14xf32>
    %v1840 = stablehlo.reduce(%v1839 init: %v1820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1841 = stablehlo.broadcast_in_dim %v1840, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1842 = stablehlo.multiply %v1836, %v1821 : tensor<64x256x14x14xf32>
    %v1843 = stablehlo.subtract %v1842, %v1838 : tensor<64x256x14x14xf32>
    %v1844 = stablehlo.multiply %v1833, %v1841 : tensor<64x256x14x14xf32>
    %v1845 = stablehlo.subtract %v1843, %v1844 : tensor<64x256x14x14xf32>
    %v1846 = stablehlo.divide %v1832, %v1821 : tensor<64x256x14x14xf32>
    %v1847 = stablehlo.multiply %v1846, %v1845 : tensor<64x256x14x14xf32>
    %v1848 = stablehlo.reshape %v1847 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1849 = stablehlo.reshape %v1848 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1850 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1851 = stablehlo.transpose %v1850, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1852 = stablehlo.convert %v1849 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1853 = stablehlo.convert %v1851 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1854 = stablehlo.convolution(%v1852, %v1853)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1855 = stablehlo.convert %v1854 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1856 = stablehlo.reshape %v1855 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1857 = stablehlo.add %v1856, %v1777 : tensor<64x50176xf32>
    %v1858 = stablehlo.reshape %v760 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1859 = stablehlo.reshape %v1848 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1860 = stablehlo.transpose %v1858, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1861 = stablehlo.transpose %v1859, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1862 = stablehlo.convert %v1860 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1863 = stablehlo.convert %v1861 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1864 = stablehlo.convolution(%v1862, %v1863)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1865 = stablehlo.convert %v1864 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1866 = stablehlo.transpose %v1865, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1867 = stablehlo.reshape %v768 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1868 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1869 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1870 = stablehlo.reduce(%v1867 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1871 = stablehlo.broadcast_in_dim %v1870, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1872 = stablehlo.divide %v1871, %v1869 : tensor<64x256x14x14xf32>
    %v1873 = stablehlo.subtract %v1867, %v1872 : tensor<64x256x14x14xf32>
    %v1874 = stablehlo.multiply %v1873, %v1873 : tensor<64x256x14x14xf32>
    %v1875 = stablehlo.reduce(%v1874 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1876 = stablehlo.broadcast_in_dim %v1875, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1877 = stablehlo.divide %v1876, %v1869 : tensor<64x256x14x14xf32>
    %v1878 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1879 = stablehlo.add %v1877, %v1878 : tensor<64x256x14x14xf32>
    %v1880 = stablehlo.rsqrt %v1879 : tensor<64x256x14x14xf32>
    %v1881 = stablehlo.multiply %v1873, %v1880 : tensor<64x256x14x14xf32>
    %v1882 = stablehlo.reshape %v1818 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1883 = stablehlo.multiply %v1882, %v1881 : tensor<64x256x14x14xf32>
    %v1884 = stablehlo.reduce(%v1883 init: %v1868) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1885 = stablehlo.reshape %v1818 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1886 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1887 = stablehlo.reduce(%v1885 init: %v1886) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1888 = stablehlo.reshape %v790 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1889 = stablehlo.reshape %v1807 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1890 = stablehlo.transpose %v1888, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1891 = stablehlo.transpose %v1889, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v1892 = stablehlo.convert %v1890 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1893 = stablehlo.convert %v1891 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v1894 = stablehlo.convolution(%v1892, %v1893)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v1895 = stablehlo.convert %v1894 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v1896 = stablehlo.transpose %v1895, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1897 = stablehlo.reshape %v798 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1898 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1899 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1900 = stablehlo.reduce(%v1897 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1901 = stablehlo.broadcast_in_dim %v1900, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1902 = stablehlo.divide %v1901, %v1899 : tensor<64x256x14x14xf32>
    %v1903 = stablehlo.subtract %v1897, %v1902 : tensor<64x256x14x14xf32>
    %v1904 = stablehlo.multiply %v1903, %v1903 : tensor<64x256x14x14xf32>
    %v1905 = stablehlo.reduce(%v1904 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1906 = stablehlo.broadcast_in_dim %v1905, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1907 = stablehlo.divide %v1906, %v1899 : tensor<64x256x14x14xf32>
    %v1908 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1909 = stablehlo.add %v1907, %v1908 : tensor<64x256x14x14xf32>
    %v1910 = stablehlo.rsqrt %v1909 : tensor<64x256x14x14xf32>
    %v1911 = stablehlo.multiply %v1903, %v1910 : tensor<64x256x14x14xf32>
    %v1912 = stablehlo.reshape %v1777 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1913 = stablehlo.multiply %v1912, %v1911 : tensor<64x256x14x14xf32>
    %v1914 = stablehlo.reduce(%v1913 init: %v1898) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1915 = stablehlo.reshape %v1777 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1916 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1917 = stablehlo.reduce(%v1915 init: %v1916) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1918 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1919 = stablehlo.compare GT, %v758, %v1918 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1920 = stablehlo.select %v1919, %v1857, %v1918 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1921 = stablehlo.reshape %v737 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1923 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1924 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1925 = stablehlo.reduce(%v1921 init: %v1922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1926 = stablehlo.broadcast_in_dim %v1925, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1927 = stablehlo.divide %v1926, %v1923 : tensor<64x256x14x14xf32>
    %v1928 = stablehlo.subtract %v1921, %v1927 : tensor<64x256x14x14xf32>
    %v1929 = stablehlo.multiply %v1928, %v1928 : tensor<64x256x14x14xf32>
    %v1930 = stablehlo.reduce(%v1929 init: %v1922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1931 = stablehlo.broadcast_in_dim %v1930, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1932 = stablehlo.divide %v1931, %v1923 : tensor<64x256x14x14xf32>
    %v1933 = stablehlo.add %v1932, %v1924 : tensor<64x256x14x14xf32>
    %v1934 = stablehlo.rsqrt %v1933 : tensor<64x256x14x14xf32>
    %v1935 = stablehlo.multiply %v1928, %v1934 : tensor<64x256x14x14xf32>
    %v1936 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1937 = stablehlo.reshape %v1920 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1938 = stablehlo.multiply %v1936, %v1937 : tensor<64x256x14x14xf32>
    %v1939 = stablehlo.reduce(%v1938 init: %v1922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1940 = stablehlo.broadcast_in_dim %v1939, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1941 = stablehlo.multiply %v1935, %v1938 : tensor<64x256x14x14xf32>
    %v1942 = stablehlo.reduce(%v1941 init: %v1922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1943 = stablehlo.broadcast_in_dim %v1942, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1944 = stablehlo.multiply %v1938, %v1923 : tensor<64x256x14x14xf32>
    %v1945 = stablehlo.subtract %v1944, %v1940 : tensor<64x256x14x14xf32>
    %v1946 = stablehlo.multiply %v1935, %v1943 : tensor<64x256x14x14xf32>
    %v1947 = stablehlo.subtract %v1945, %v1946 : tensor<64x256x14x14xf32>
    %v1948 = stablehlo.divide %v1934, %v1923 : tensor<64x256x14x14xf32>
    %v1949 = stablehlo.multiply %v1948, %v1947 : tensor<64x256x14x14xf32>
    %v1950 = stablehlo.reshape %v1949 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1951 = stablehlo.reshape %v1950 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1952 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1953 = stablehlo.transpose %v1952, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1954 = stablehlo.convert %v1951 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1955 = stablehlo.convert %v1953 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1956 = stablehlo.convolution(%v1954, %v1955)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1957 = stablehlo.convert %v1956 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1958 = stablehlo.reshape %v1957 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1959 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1960 = stablehlo.compare GT, %v727, %v1959 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v1961 = stablehlo.select %v1960, %v1958, %v1959 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v1962 = stablehlo.reshape %v707 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1964 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v1965 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1966 = stablehlo.reduce(%v1962 init: %v1963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1967 = stablehlo.broadcast_in_dim %v1966, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1968 = stablehlo.divide %v1967, %v1964 : tensor<64x256x14x14xf32>
    %v1969 = stablehlo.subtract %v1962, %v1968 : tensor<64x256x14x14xf32>
    %v1970 = stablehlo.multiply %v1969, %v1969 : tensor<64x256x14x14xf32>
    %v1971 = stablehlo.reduce(%v1970 init: %v1963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1972 = stablehlo.broadcast_in_dim %v1971, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1973 = stablehlo.divide %v1972, %v1964 : tensor<64x256x14x14xf32>
    %v1974 = stablehlo.add %v1973, %v1965 : tensor<64x256x14x14xf32>
    %v1975 = stablehlo.rsqrt %v1974 : tensor<64x256x14x14xf32>
    %v1976 = stablehlo.multiply %v1969, %v1975 : tensor<64x256x14x14xf32>
    %v1977 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1978 = stablehlo.reshape %v1961 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1979 = stablehlo.multiply %v1977, %v1978 : tensor<64x256x14x14xf32>
    %v1980 = stablehlo.reduce(%v1979 init: %v1963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1981 = stablehlo.broadcast_in_dim %v1980, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1982 = stablehlo.multiply %v1976, %v1979 : tensor<64x256x14x14xf32>
    %v1983 = stablehlo.reduce(%v1982 init: %v1963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1984 = stablehlo.broadcast_in_dim %v1983, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1985 = stablehlo.multiply %v1979, %v1964 : tensor<64x256x14x14xf32>
    %v1986 = stablehlo.subtract %v1985, %v1981 : tensor<64x256x14x14xf32>
    %v1987 = stablehlo.multiply %v1976, %v1984 : tensor<64x256x14x14xf32>
    %v1988 = stablehlo.subtract %v1986, %v1987 : tensor<64x256x14x14xf32>
    %v1989 = stablehlo.divide %v1975, %v1964 : tensor<64x256x14x14xf32>
    %v1990 = stablehlo.multiply %v1989, %v1988 : tensor<64x256x14x14xf32>
    %v1991 = stablehlo.reshape %v1990 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1992 = stablehlo.reshape %v1991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1993 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v1994 = stablehlo.transpose %v1993, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v1995 = stablehlo.convert %v1992 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1996 = stablehlo.convert %v1994 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1997 = stablehlo.convolution(%v1995, %v1996)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1998 = stablehlo.convert %v1997 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1999 = stablehlo.reshape %v1998 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2000 = stablehlo.add %v1999, %v1920 : tensor<64x50176xf32>
    %v2001 = stablehlo.reshape %v699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2002 = stablehlo.reshape %v1991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2003 = stablehlo.transpose %v2001, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2004 = stablehlo.transpose %v2002, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2005 = stablehlo.convert %v2003 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2006 = stablehlo.convert %v2004 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2007 = stablehlo.convolution(%v2005, %v2006)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2008 = stablehlo.convert %v2007 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2009 = stablehlo.transpose %v2008, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2010 = stablehlo.reshape %v707 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2011 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2012 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2013 = stablehlo.reduce(%v2010 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2014 = stablehlo.broadcast_in_dim %v2013, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2015 = stablehlo.divide %v2014, %v2012 : tensor<64x256x14x14xf32>
    %v2016 = stablehlo.subtract %v2010, %v2015 : tensor<64x256x14x14xf32>
    %v2017 = stablehlo.multiply %v2016, %v2016 : tensor<64x256x14x14xf32>
    %v2018 = stablehlo.reduce(%v2017 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2019 = stablehlo.broadcast_in_dim %v2018, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2020 = stablehlo.divide %v2019, %v2012 : tensor<64x256x14x14xf32>
    %v2021 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2022 = stablehlo.add %v2020, %v2021 : tensor<64x256x14x14xf32>
    %v2023 = stablehlo.rsqrt %v2022 : tensor<64x256x14x14xf32>
    %v2024 = stablehlo.multiply %v2016, %v2023 : tensor<64x256x14x14xf32>
    %v2025 = stablehlo.reshape %v1961 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2026 = stablehlo.multiply %v2025, %v2024 : tensor<64x256x14x14xf32>
    %v2027 = stablehlo.reduce(%v2026 init: %v2011) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2028 = stablehlo.reshape %v1961 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2029 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2030 = stablehlo.reduce(%v2028 init: %v2029) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2031 = stablehlo.reshape %v729 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2032 = stablehlo.reshape %v1950 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2033 = stablehlo.transpose %v2031, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2034 = stablehlo.transpose %v2032, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2035 = stablehlo.convert %v2033 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2036 = stablehlo.convert %v2034 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2037 = stablehlo.convolution(%v2035, %v2036)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2038 = stablehlo.convert %v2037 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2039 = stablehlo.transpose %v2038, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2040 = stablehlo.reshape %v737 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2042 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2043 = stablehlo.reduce(%v2040 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2044 = stablehlo.broadcast_in_dim %v2043, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2045 = stablehlo.divide %v2044, %v2042 : tensor<64x256x14x14xf32>
    %v2046 = stablehlo.subtract %v2040, %v2045 : tensor<64x256x14x14xf32>
    %v2047 = stablehlo.multiply %v2046, %v2046 : tensor<64x256x14x14xf32>
    %v2048 = stablehlo.reduce(%v2047 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2048, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2050 = stablehlo.divide %v2049, %v2042 : tensor<64x256x14x14xf32>
    %v2051 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2052 = stablehlo.add %v2050, %v2051 : tensor<64x256x14x14xf32>
    %v2053 = stablehlo.rsqrt %v2052 : tensor<64x256x14x14xf32>
    %v2054 = stablehlo.multiply %v2046, %v2053 : tensor<64x256x14x14xf32>
    %v2055 = stablehlo.reshape %v1920 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2056 = stablehlo.multiply %v2055, %v2054 : tensor<64x256x14x14xf32>
    %v2057 = stablehlo.reduce(%v2056 init: %v2041) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2058 = stablehlo.reshape %v1920 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2060 = stablehlo.reduce(%v2058 init: %v2059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2061 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2062 = stablehlo.compare GT, %v697, %v2061 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2063 = stablehlo.select %v2062, %v2000, %v2061 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2064 = stablehlo.reshape %v676 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2065 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2066 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2067 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2068 = stablehlo.reduce(%v2064 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2069 = stablehlo.broadcast_in_dim %v2068, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2070 = stablehlo.divide %v2069, %v2066 : tensor<64x256x14x14xf32>
    %v2071 = stablehlo.subtract %v2064, %v2070 : tensor<64x256x14x14xf32>
    %v2072 = stablehlo.multiply %v2071, %v2071 : tensor<64x256x14x14xf32>
    %v2073 = stablehlo.reduce(%v2072 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2073, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2075 = stablehlo.divide %v2074, %v2066 : tensor<64x256x14x14xf32>
    %v2076 = stablehlo.add %v2075, %v2067 : tensor<64x256x14x14xf32>
    %v2077 = stablehlo.rsqrt %v2076 : tensor<64x256x14x14xf32>
    %v2078 = stablehlo.multiply %v2071, %v2077 : tensor<64x256x14x14xf32>
    %v2079 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2080 = stablehlo.reshape %v2063 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2081 = stablehlo.multiply %v2079, %v2080 : tensor<64x256x14x14xf32>
    %v2082 = stablehlo.reduce(%v2081 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2083 = stablehlo.broadcast_in_dim %v2082, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2084 = stablehlo.multiply %v2078, %v2081 : tensor<64x256x14x14xf32>
    %v2085 = stablehlo.reduce(%v2084 init: %v2065) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2086 = stablehlo.broadcast_in_dim %v2085, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2087 = stablehlo.multiply %v2081, %v2066 : tensor<64x256x14x14xf32>
    %v2088 = stablehlo.subtract %v2087, %v2083 : tensor<64x256x14x14xf32>
    %v2089 = stablehlo.multiply %v2078, %v2086 : tensor<64x256x14x14xf32>
    %v2090 = stablehlo.subtract %v2088, %v2089 : tensor<64x256x14x14xf32>
    %v2091 = stablehlo.divide %v2077, %v2066 : tensor<64x256x14x14xf32>
    %v2092 = stablehlo.multiply %v2091, %v2090 : tensor<64x256x14x14xf32>
    %v2093 = stablehlo.reshape %v2092 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2094 = stablehlo.reshape %v2093 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2095 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2096 = stablehlo.transpose %v2095, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2097 = stablehlo.convert %v2094 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2098 = stablehlo.convert %v2096 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2099 = stablehlo.convolution(%v2097, %v2098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2100 = stablehlo.convert %v2099 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2101 = stablehlo.reshape %v2100 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2102 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2103 = stablehlo.compare GT, %v666, %v2102 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2104 = stablehlo.select %v2103, %v2101, %v2102 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2105 = stablehlo.reshape %v646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2106 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2107 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2108 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2109 = stablehlo.reduce(%v2105 init: %v2106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2110 = stablehlo.broadcast_in_dim %v2109, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2111 = stablehlo.divide %v2110, %v2107 : tensor<64x256x14x14xf32>
    %v2112 = stablehlo.subtract %v2105, %v2111 : tensor<64x256x14x14xf32>
    %v2113 = stablehlo.multiply %v2112, %v2112 : tensor<64x256x14x14xf32>
    %v2114 = stablehlo.reduce(%v2113 init: %v2106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2115 = stablehlo.broadcast_in_dim %v2114, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2116 = stablehlo.divide %v2115, %v2107 : tensor<64x256x14x14xf32>
    %v2117 = stablehlo.add %v2116, %v2108 : tensor<64x256x14x14xf32>
    %v2118 = stablehlo.rsqrt %v2117 : tensor<64x256x14x14xf32>
    %v2119 = stablehlo.multiply %v2112, %v2118 : tensor<64x256x14x14xf32>
    %v2120 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2121 = stablehlo.reshape %v2104 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2122 = stablehlo.multiply %v2120, %v2121 : tensor<64x256x14x14xf32>
    %v2123 = stablehlo.reduce(%v2122 init: %v2106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2124 = stablehlo.broadcast_in_dim %v2123, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2125 = stablehlo.multiply %v2119, %v2122 : tensor<64x256x14x14xf32>
    %v2126 = stablehlo.reduce(%v2125 init: %v2106) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2127 = stablehlo.broadcast_in_dim %v2126, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2128 = stablehlo.multiply %v2122, %v2107 : tensor<64x256x14x14xf32>
    %v2129 = stablehlo.subtract %v2128, %v2124 : tensor<64x256x14x14xf32>
    %v2130 = stablehlo.multiply %v2119, %v2127 : tensor<64x256x14x14xf32>
    %v2131 = stablehlo.subtract %v2129, %v2130 : tensor<64x256x14x14xf32>
    %v2132 = stablehlo.divide %v2118, %v2107 : tensor<64x256x14x14xf32>
    %v2133 = stablehlo.multiply %v2132, %v2131 : tensor<64x256x14x14xf32>
    %v2134 = stablehlo.reshape %v2133 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2135 = stablehlo.reshape %v2134 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2136 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2137 = stablehlo.transpose %v2136, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2138 = stablehlo.convert %v2135 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2139 = stablehlo.convert %v2137 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2140 = stablehlo.convolution(%v2138, %v2139)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2141 = stablehlo.convert %v2140 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2142 = stablehlo.reshape %v2141 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2143 = stablehlo.add %v2142, %v2063 : tensor<64x50176xf32>
    %v2144 = stablehlo.reshape %v638 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2145 = stablehlo.reshape %v2134 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2146 = stablehlo.transpose %v2144, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2147 = stablehlo.transpose %v2145, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2148 = stablehlo.convert %v2146 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2149 = stablehlo.convert %v2147 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2150 = stablehlo.convolution(%v2148, %v2149)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2151 = stablehlo.convert %v2150 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2152 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2153 = stablehlo.reshape %v646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2154 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2155 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2156 = stablehlo.reduce(%v2153 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2157 = stablehlo.broadcast_in_dim %v2156, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2158 = stablehlo.divide %v2157, %v2155 : tensor<64x256x14x14xf32>
    %v2159 = stablehlo.subtract %v2153, %v2158 : tensor<64x256x14x14xf32>
    %v2160 = stablehlo.multiply %v2159, %v2159 : tensor<64x256x14x14xf32>
    %v2161 = stablehlo.reduce(%v2160 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2161, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2163 = stablehlo.divide %v2162, %v2155 : tensor<64x256x14x14xf32>
    %v2164 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2165 = stablehlo.add %v2163, %v2164 : tensor<64x256x14x14xf32>
    %v2166 = stablehlo.rsqrt %v2165 : tensor<64x256x14x14xf32>
    %v2167 = stablehlo.multiply %v2159, %v2166 : tensor<64x256x14x14xf32>
    %v2168 = stablehlo.reshape %v2104 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2169 = stablehlo.multiply %v2168, %v2167 : tensor<64x256x14x14xf32>
    %v2170 = stablehlo.reduce(%v2169 init: %v2154) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2171 = stablehlo.reshape %v2104 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2172 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2173 = stablehlo.reduce(%v2171 init: %v2172) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2174 = stablehlo.reshape %v668 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2175 = stablehlo.reshape %v2093 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2176 = stablehlo.transpose %v2174, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2177 = stablehlo.transpose %v2175, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2178 = stablehlo.convert %v2176 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2179 = stablehlo.convert %v2177 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2180 = stablehlo.convolution(%v2178, %v2179)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2181 = stablehlo.convert %v2180 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2182 = stablehlo.transpose %v2181, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2183 = stablehlo.reshape %v676 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2185 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2186 = stablehlo.reduce(%v2183 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2187 = stablehlo.broadcast_in_dim %v2186, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2188 = stablehlo.divide %v2187, %v2185 : tensor<64x256x14x14xf32>
    %v2189 = stablehlo.subtract %v2183, %v2188 : tensor<64x256x14x14xf32>
    %v2190 = stablehlo.multiply %v2189, %v2189 : tensor<64x256x14x14xf32>
    %v2191 = stablehlo.reduce(%v2190 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2192 = stablehlo.broadcast_in_dim %v2191, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2193 = stablehlo.divide %v2192, %v2185 : tensor<64x256x14x14xf32>
    %v2194 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2195 = stablehlo.add %v2193, %v2194 : tensor<64x256x14x14xf32>
    %v2196 = stablehlo.rsqrt %v2195 : tensor<64x256x14x14xf32>
    %v2197 = stablehlo.multiply %v2189, %v2196 : tensor<64x256x14x14xf32>
    %v2198 = stablehlo.reshape %v2063 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2199 = stablehlo.multiply %v2198, %v2197 : tensor<64x256x14x14xf32>
    %v2200 = stablehlo.reduce(%v2199 init: %v2184) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2201 = stablehlo.reshape %v2063 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2202 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2203 = stablehlo.reduce(%v2201 init: %v2202) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2204 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2205 = stablehlo.compare GT, %v636, %v2204 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2206 = stablehlo.select %v2205, %v2143, %v2204 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2207 = stablehlo.reshape %v615 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2208 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2209 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2210 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2211 = stablehlo.reduce(%v2207 init: %v2208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2212 = stablehlo.broadcast_in_dim %v2211, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2213 = stablehlo.divide %v2212, %v2209 : tensor<64x256x14x14xf32>
    %v2214 = stablehlo.subtract %v2207, %v2213 : tensor<64x256x14x14xf32>
    %v2215 = stablehlo.multiply %v2214, %v2214 : tensor<64x256x14x14xf32>
    %v2216 = stablehlo.reduce(%v2215 init: %v2208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2217 = stablehlo.broadcast_in_dim %v2216, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2218 = stablehlo.divide %v2217, %v2209 : tensor<64x256x14x14xf32>
    %v2219 = stablehlo.add %v2218, %v2210 : tensor<64x256x14x14xf32>
    %v2220 = stablehlo.rsqrt %v2219 : tensor<64x256x14x14xf32>
    %v2221 = stablehlo.multiply %v2214, %v2220 : tensor<64x256x14x14xf32>
    %v2222 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2223 = stablehlo.reshape %v2206 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2224 = stablehlo.multiply %v2222, %v2223 : tensor<64x256x14x14xf32>
    %v2225 = stablehlo.reduce(%v2224 init: %v2208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2226 = stablehlo.broadcast_in_dim %v2225, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2227 = stablehlo.multiply %v2221, %v2224 : tensor<64x256x14x14xf32>
    %v2228 = stablehlo.reduce(%v2227 init: %v2208) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2229 = stablehlo.broadcast_in_dim %v2228, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2230 = stablehlo.multiply %v2224, %v2209 : tensor<64x256x14x14xf32>
    %v2231 = stablehlo.subtract %v2230, %v2226 : tensor<64x256x14x14xf32>
    %v2232 = stablehlo.multiply %v2221, %v2229 : tensor<64x256x14x14xf32>
    %v2233 = stablehlo.subtract %v2231, %v2232 : tensor<64x256x14x14xf32>
    %v2234 = stablehlo.divide %v2220, %v2209 : tensor<64x256x14x14xf32>
    %v2235 = stablehlo.multiply %v2234, %v2233 : tensor<64x256x14x14xf32>
    %v2236 = stablehlo.reshape %v2235 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2237 = stablehlo.reshape %v2236 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2238 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2239 = stablehlo.transpose %v2238, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2240 = stablehlo.convert %v2237 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2241 = stablehlo.convert %v2239 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2242 = stablehlo.convolution(%v2240, %v2241)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2243 = stablehlo.convert %v2242 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2244 = stablehlo.reshape %v2243 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2245 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2246 = stablehlo.compare GT, %v605, %v2245 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2247 = stablehlo.select %v2246, %v2244, %v2245 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2248 = stablehlo.reshape %v585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2249 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2250 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2251 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2252 = stablehlo.reduce(%v2248 init: %v2249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2253 = stablehlo.broadcast_in_dim %v2252, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2254 = stablehlo.divide %v2253, %v2250 : tensor<64x256x14x14xf32>
    %v2255 = stablehlo.subtract %v2248, %v2254 : tensor<64x256x14x14xf32>
    %v2256 = stablehlo.multiply %v2255, %v2255 : tensor<64x256x14x14xf32>
    %v2257 = stablehlo.reduce(%v2256 init: %v2249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2258 = stablehlo.broadcast_in_dim %v2257, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2259 = stablehlo.divide %v2258, %v2250 : tensor<64x256x14x14xf32>
    %v2260 = stablehlo.add %v2259, %v2251 : tensor<64x256x14x14xf32>
    %v2261 = stablehlo.rsqrt %v2260 : tensor<64x256x14x14xf32>
    %v2262 = stablehlo.multiply %v2255, %v2261 : tensor<64x256x14x14xf32>
    %v2263 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2264 = stablehlo.reshape %v2247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2265 = stablehlo.multiply %v2263, %v2264 : tensor<64x256x14x14xf32>
    %v2266 = stablehlo.reduce(%v2265 init: %v2249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2267 = stablehlo.broadcast_in_dim %v2266, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2268 = stablehlo.multiply %v2262, %v2265 : tensor<64x256x14x14xf32>
    %v2269 = stablehlo.reduce(%v2268 init: %v2249) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2270 = stablehlo.broadcast_in_dim %v2269, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2271 = stablehlo.multiply %v2265, %v2250 : tensor<64x256x14x14xf32>
    %v2272 = stablehlo.subtract %v2271, %v2267 : tensor<64x256x14x14xf32>
    %v2273 = stablehlo.multiply %v2262, %v2270 : tensor<64x256x14x14xf32>
    %v2274 = stablehlo.subtract %v2272, %v2273 : tensor<64x256x14x14xf32>
    %v2275 = stablehlo.divide %v2261, %v2250 : tensor<64x256x14x14xf32>
    %v2276 = stablehlo.multiply %v2275, %v2274 : tensor<64x256x14x14xf32>
    %v2277 = stablehlo.reshape %v2276 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2278 = stablehlo.reshape %v2277 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2279 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2280 = stablehlo.transpose %v2279, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2281 = stablehlo.convert %v2278 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2282 = stablehlo.convert %v2280 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2283 = stablehlo.convolution(%v2281, %v2282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2284 = stablehlo.convert %v2283 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2285 = stablehlo.reshape %v2284 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2286 = stablehlo.add %v2285, %v2206 : tensor<64x50176xf32>
    %v2287 = stablehlo.reshape %v577 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2288 = stablehlo.reshape %v2277 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2289 = stablehlo.transpose %v2287, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2290 = stablehlo.transpose %v2288, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2291 = stablehlo.convert %v2289 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2292 = stablehlo.convert %v2290 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2293 = stablehlo.convolution(%v2291, %v2292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2294 = stablehlo.convert %v2293 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2295 = stablehlo.transpose %v2294, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2296 = stablehlo.reshape %v585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2297 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2298 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2299 = stablehlo.reduce(%v2296 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2300 = stablehlo.broadcast_in_dim %v2299, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2301 = stablehlo.divide %v2300, %v2298 : tensor<64x256x14x14xf32>
    %v2302 = stablehlo.subtract %v2296, %v2301 : tensor<64x256x14x14xf32>
    %v2303 = stablehlo.multiply %v2302, %v2302 : tensor<64x256x14x14xf32>
    %v2304 = stablehlo.reduce(%v2303 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2305 = stablehlo.broadcast_in_dim %v2304, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2306 = stablehlo.divide %v2305, %v2298 : tensor<64x256x14x14xf32>
    %v2307 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2308 = stablehlo.add %v2306, %v2307 : tensor<64x256x14x14xf32>
    %v2309 = stablehlo.rsqrt %v2308 : tensor<64x256x14x14xf32>
    %v2310 = stablehlo.multiply %v2302, %v2309 : tensor<64x256x14x14xf32>
    %v2311 = stablehlo.reshape %v2247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2312 = stablehlo.multiply %v2311, %v2310 : tensor<64x256x14x14xf32>
    %v2313 = stablehlo.reduce(%v2312 init: %v2297) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2314 = stablehlo.reshape %v2247 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2315 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2316 = stablehlo.reduce(%v2314 init: %v2315) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2317 = stablehlo.reshape %v607 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2318 = stablehlo.reshape %v2236 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2319 = stablehlo.transpose %v2317, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2320 = stablehlo.transpose %v2318, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2321 = stablehlo.convert %v2319 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2322 = stablehlo.convert %v2320 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2323 = stablehlo.convolution(%v2321, %v2322)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2324 = stablehlo.convert %v2323 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2325 = stablehlo.transpose %v2324, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2326 = stablehlo.reshape %v615 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2327 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2328 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2329 = stablehlo.reduce(%v2326 init: %v2327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2330 = stablehlo.broadcast_in_dim %v2329, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2331 = stablehlo.divide %v2330, %v2328 : tensor<64x256x14x14xf32>
    %v2332 = stablehlo.subtract %v2326, %v2331 : tensor<64x256x14x14xf32>
    %v2333 = stablehlo.multiply %v2332, %v2332 : tensor<64x256x14x14xf32>
    %v2334 = stablehlo.reduce(%v2333 init: %v2327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2335 = stablehlo.broadcast_in_dim %v2334, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2336 = stablehlo.divide %v2335, %v2328 : tensor<64x256x14x14xf32>
    %v2337 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2338 = stablehlo.add %v2336, %v2337 : tensor<64x256x14x14xf32>
    %v2339 = stablehlo.rsqrt %v2338 : tensor<64x256x14x14xf32>
    %v2340 = stablehlo.multiply %v2332, %v2339 : tensor<64x256x14x14xf32>
    %v2341 = stablehlo.reshape %v2206 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2342 = stablehlo.multiply %v2341, %v2340 : tensor<64x256x14x14xf32>
    %v2343 = stablehlo.reduce(%v2342 init: %v2327) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2344 = stablehlo.reshape %v2206 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2345 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2346 = stablehlo.reduce(%v2344 init: %v2345) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2347 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2348 = stablehlo.compare GT, %v575, %v2347 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2349 = stablehlo.select %v2348, %v2286, %v2347 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2350 = stablehlo.reshape %v526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2351 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2352 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2353 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2354 = stablehlo.reduce(%v2350 init: %v2351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2355 = stablehlo.broadcast_in_dim %v2354, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2356 = stablehlo.divide %v2355, %v2352 : tensor<64x256x14x14xf32>
    %v2357 = stablehlo.subtract %v2350, %v2356 : tensor<64x256x14x14xf32>
    %v2358 = stablehlo.multiply %v2357, %v2357 : tensor<64x256x14x14xf32>
    %v2359 = stablehlo.reduce(%v2358 init: %v2351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2360 = stablehlo.broadcast_in_dim %v2359, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2361 = stablehlo.divide %v2360, %v2352 : tensor<64x256x14x14xf32>
    %v2362 = stablehlo.add %v2361, %v2353 : tensor<64x256x14x14xf32>
    %v2363 = stablehlo.rsqrt %v2362 : tensor<64x256x14x14xf32>
    %v2364 = stablehlo.multiply %v2357, %v2363 : tensor<64x256x14x14xf32>
    %v2365 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2366 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2367 = stablehlo.multiply %v2365, %v2366 : tensor<64x256x14x14xf32>
    %v2368 = stablehlo.reduce(%v2367 init: %v2351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2369 = stablehlo.broadcast_in_dim %v2368, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2370 = stablehlo.multiply %v2364, %v2367 : tensor<64x256x14x14xf32>
    %v2371 = stablehlo.reduce(%v2370 init: %v2351) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2372 = stablehlo.broadcast_in_dim %v2371, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2373 = stablehlo.multiply %v2367, %v2352 : tensor<64x256x14x14xf32>
    %v2374 = stablehlo.subtract %v2373, %v2369 : tensor<64x256x14x14xf32>
    %v2375 = stablehlo.multiply %v2364, %v2372 : tensor<64x256x14x14xf32>
    %v2376 = stablehlo.subtract %v2374, %v2375 : tensor<64x256x14x14xf32>
    %v2377 = stablehlo.divide %v2363, %v2352 : tensor<64x256x14x14xf32>
    %v2378 = stablehlo.multiply %v2377, %v2376 : tensor<64x256x14x14xf32>
    %v2379 = stablehlo.reshape %v2378 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2380 = stablehlo.reshape %v2379 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2381 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2382 = stablehlo.transpose %v2381, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2383 = stablehlo.convert %v2380 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2384 = stablehlo.convert %v2382 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2385 = stablehlo.convolution(%v2383, %v2384)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2386 = stablehlo.convert %v2385 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2387 = stablehlo.reshape %v2386 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2388 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v2389 = stablehlo.compare GT, %v516, %v2388 : (tensor<64x50176xf32>, tensor<64x50176xf32>) -> tensor<64x50176xi1>
    %v2390 = stablehlo.select %v2389, %v2387, %v2388 : tensor<64x50176xi1>, tensor<64x50176xf32>
    %v2391 = stablehlo.reshape %v496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2392 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2393 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2394 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2395 = stablehlo.reduce(%v2391 init: %v2392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2396 = stablehlo.broadcast_in_dim %v2395, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2397 = stablehlo.divide %v2396, %v2393 : tensor<64x256x14x14xf32>
    %v2398 = stablehlo.subtract %v2391, %v2397 : tensor<64x256x14x14xf32>
    %v2399 = stablehlo.multiply %v2398, %v2398 : tensor<64x256x14x14xf32>
    %v2400 = stablehlo.reduce(%v2399 init: %v2392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2401 = stablehlo.broadcast_in_dim %v2400, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2402 = stablehlo.divide %v2401, %v2393 : tensor<64x256x14x14xf32>
    %v2403 = stablehlo.add %v2402, %v2394 : tensor<64x256x14x14xf32>
    %v2404 = stablehlo.rsqrt %v2403 : tensor<64x256x14x14xf32>
    %v2405 = stablehlo.multiply %v2398, %v2404 : tensor<64x256x14x14xf32>
    %v2406 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2407 = stablehlo.reshape %v2390 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2408 = stablehlo.multiply %v2406, %v2407 : tensor<64x256x14x14xf32>
    %v2409 = stablehlo.reduce(%v2408 init: %v2392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2410 = stablehlo.broadcast_in_dim %v2409, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2411 = stablehlo.multiply %v2405, %v2408 : tensor<64x256x14x14xf32>
    %v2412 = stablehlo.reduce(%v2411 init: %v2392) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2413 = stablehlo.broadcast_in_dim %v2412, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2414 = stablehlo.multiply %v2408, %v2393 : tensor<64x256x14x14xf32>
    %v2415 = stablehlo.subtract %v2414, %v2410 : tensor<64x256x14x14xf32>
    %v2416 = stablehlo.multiply %v2405, %v2413 : tensor<64x256x14x14xf32>
    %v2417 = stablehlo.subtract %v2415, %v2416 : tensor<64x256x14x14xf32>
    %v2418 = stablehlo.divide %v2404, %v2393 : tensor<64x256x14x14xf32>
    %v2419 = stablehlo.multiply %v2418, %v2417 : tensor<64x256x14x14xf32>
    %v2420 = stablehlo.reshape %v2419 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2421 = stablehlo.reshape %v2420 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2422 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2423 = stablehlo.pad %v2421, %v2422, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v2424 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v2425 = stablehlo.transpose %v2424, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v2426 = stablehlo.convert %v2423 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v2427 = stablehlo.convert %v2425 : (tensor<128x256x3x3xf32>) -> tensor<128x256x3x3xbf16>
    %v2428 = stablehlo.convolution(%v2426, %v2427)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<128x256x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2429 = stablehlo.convert %v2428 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2430 = stablehlo.reshape %v2429 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2431 = stablehlo.reshape %v554 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2432 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2433 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2434 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2435 = stablehlo.reduce(%v2431 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2436 = stablehlo.broadcast_in_dim %v2435, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2437 = stablehlo.divide %v2436, %v2433 : tensor<64x256x14x14xf32>
    %v2438 = stablehlo.subtract %v2431, %v2437 : tensor<64x256x14x14xf32>
    %v2439 = stablehlo.multiply %v2438, %v2438 : tensor<64x256x14x14xf32>
    %v2440 = stablehlo.reduce(%v2439 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2441 = stablehlo.broadcast_in_dim %v2440, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2442 = stablehlo.divide %v2441, %v2433 : tensor<64x256x14x14xf32>
    %v2443 = stablehlo.add %v2442, %v2434 : tensor<64x256x14x14xf32>
    %v2444 = stablehlo.rsqrt %v2443 : tensor<64x256x14x14xf32>
    %v2445 = stablehlo.multiply %v2438, %v2444 : tensor<64x256x14x14xf32>
    %v2446 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2447 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2448 = stablehlo.multiply %v2446, %v2447 : tensor<64x256x14x14xf32>
    %v2449 = stablehlo.reduce(%v2448 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2450 = stablehlo.broadcast_in_dim %v2449, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2451 = stablehlo.multiply %v2445, %v2448 : tensor<64x256x14x14xf32>
    %v2452 = stablehlo.reduce(%v2451 init: %v2432) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2453 = stablehlo.broadcast_in_dim %v2452, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2454 = stablehlo.multiply %v2448, %v2433 : tensor<64x256x14x14xf32>
    %v2455 = stablehlo.subtract %v2454, %v2450 : tensor<64x256x14x14xf32>
    %v2456 = stablehlo.multiply %v2445, %v2453 : tensor<64x256x14x14xf32>
    %v2457 = stablehlo.subtract %v2455, %v2456 : tensor<64x256x14x14xf32>
    %v2458 = stablehlo.divide %v2444, %v2433 : tensor<64x256x14x14xf32>
    %v2459 = stablehlo.multiply %v2458, %v2457 : tensor<64x256x14x14xf32>
    %v2460 = stablehlo.reshape %v2459 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2461 = stablehlo.reshape %v2460 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2462 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2463 = stablehlo.pad %v2461, %v2462, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v2464 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v2465 = stablehlo.transpose %v2464, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v2466 = stablehlo.convert %v2463 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v2467 = stablehlo.convert %v2465 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v2468 = stablehlo.convolution(%v2466, %v2467)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<128x256x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v2469 = stablehlo.convert %v2468 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2470 = stablehlo.reshape %v2469 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2471 = stablehlo.add %v2430, %v2470 : tensor<64x100352xf32>
    %v2472 = stablehlo.reshape %v488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2473 = stablehlo.reshape %v2420 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2474 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2475 = stablehlo.pad %v2473, %v2474, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v2476 = stablehlo.transpose %v2472, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2477 = stablehlo.transpose %v2475, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v2478 = stablehlo.convert %v2476 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2479 = stablehlo.convert %v2477 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v2480 = stablehlo.convolution(%v2478, %v2479)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<128x256x3x3xbf16>
    %v2481 = stablehlo.convert %v2480 : (tensor<128x256x3x3xbf16>) -> tensor<128x256x3x3xf32>
    %v2482 = stablehlo.transpose %v2481, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v2483 = stablehlo.reshape %v496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2485 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2486 = stablehlo.reduce(%v2483 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2487 = stablehlo.broadcast_in_dim %v2486, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2488 = stablehlo.divide %v2487, %v2485 : tensor<64x256x14x14xf32>
    %v2489 = stablehlo.subtract %v2483, %v2488 : tensor<64x256x14x14xf32>
    %v2490 = stablehlo.multiply %v2489, %v2489 : tensor<64x256x14x14xf32>
    %v2491 = stablehlo.reduce(%v2490 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2492 = stablehlo.broadcast_in_dim %v2491, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2493 = stablehlo.divide %v2492, %v2485 : tensor<64x256x14x14xf32>
    %v2494 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2495 = stablehlo.add %v2493, %v2494 : tensor<64x256x14x14xf32>
    %v2496 = stablehlo.rsqrt %v2495 : tensor<64x256x14x14xf32>
    %v2497 = stablehlo.multiply %v2489, %v2496 : tensor<64x256x14x14xf32>
    %v2498 = stablehlo.reshape %v2390 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2499 = stablehlo.multiply %v2498, %v2497 : tensor<64x256x14x14xf32>
    %v2500 = stablehlo.reduce(%v2499 init: %v2484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2501 = stablehlo.reshape %v2390 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2502 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2503 = stablehlo.reduce(%v2501 init: %v2502) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2504 = stablehlo.reshape %v518 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2505 = stablehlo.reshape %v2379 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2506 = stablehlo.transpose %v2504, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2507 = stablehlo.transpose %v2505, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2508 = stablehlo.convert %v2506 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2509 = stablehlo.convert %v2507 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2510 = stablehlo.convolution(%v2508, %v2509)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2511 = stablehlo.convert %v2510 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2512 = stablehlo.transpose %v2511, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2513 = stablehlo.reshape %v526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2515 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2516 = stablehlo.reduce(%v2513 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2517 = stablehlo.broadcast_in_dim %v2516, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2518 = stablehlo.divide %v2517, %v2515 : tensor<64x256x14x14xf32>
    %v2519 = stablehlo.subtract %v2513, %v2518 : tensor<64x256x14x14xf32>
    %v2520 = stablehlo.multiply %v2519, %v2519 : tensor<64x256x14x14xf32>
    %v2521 = stablehlo.reduce(%v2520 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2522 = stablehlo.broadcast_in_dim %v2521, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2523 = stablehlo.divide %v2522, %v2515 : tensor<64x256x14x14xf32>
    %v2524 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2525 = stablehlo.add %v2523, %v2524 : tensor<64x256x14x14xf32>
    %v2526 = stablehlo.rsqrt %v2525 : tensor<64x256x14x14xf32>
    %v2527 = stablehlo.multiply %v2519, %v2526 : tensor<64x256x14x14xf32>
    %v2528 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2529 = stablehlo.multiply %v2528, %v2527 : tensor<64x256x14x14xf32>
    %v2530 = stablehlo.reduce(%v2529 init: %v2514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2531 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2533 = stablehlo.reduce(%v2531 init: %v2532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2534 = stablehlo.reshape %v488 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2535 = stablehlo.reshape %v2460 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2536 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2537 = stablehlo.pad %v2535, %v2536, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v2538 = stablehlo.transpose %v2534, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2539 = stablehlo.transpose %v2537, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v2540 = stablehlo.convert %v2538 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2541 = stablehlo.convert %v2539 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v2542 = stablehlo.convolution(%v2540, %v2541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<128x256x1x1xbf16>
    %v2543 = stablehlo.convert %v2542 : (tensor<128x256x1x1xbf16>) -> tensor<128x256x1x1xf32>
    %v2544 = stablehlo.transpose %v2543, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v2545 = stablehlo.reshape %v554 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2546 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2547 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v2548 = stablehlo.reduce(%v2545 init: %v2546) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2549 = stablehlo.broadcast_in_dim %v2548, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2550 = stablehlo.divide %v2549, %v2547 : tensor<64x256x14x14xf32>
    %v2551 = stablehlo.subtract %v2545, %v2550 : tensor<64x256x14x14xf32>
    %v2552 = stablehlo.multiply %v2551, %v2551 : tensor<64x256x14x14xf32>
    %v2553 = stablehlo.reduce(%v2552 init: %v2546) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2554 = stablehlo.broadcast_in_dim %v2553, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2555 = stablehlo.divide %v2554, %v2547 : tensor<64x256x14x14xf32>
    %v2556 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2557 = stablehlo.add %v2555, %v2556 : tensor<64x256x14x14xf32>
    %v2558 = stablehlo.rsqrt %v2557 : tensor<64x256x14x14xf32>
    %v2559 = stablehlo.multiply %v2551, %v2558 : tensor<64x256x14x14xf32>
    %v2560 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2561 = stablehlo.multiply %v2560, %v2559 : tensor<64x256x14x14xf32>
    %v2562 = stablehlo.reduce(%v2561 init: %v2546) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2563 = stablehlo.reshape %v2349 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2565 = stablehlo.reduce(%v2563 init: %v2564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2566 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2567 = stablehlo.compare GT, %v486, %v2566 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2568 = stablehlo.select %v2567, %v2471, %v2566 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2569 = stablehlo.reshape %v465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2570 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2571 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2572 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2573 = stablehlo.reduce(%v2569 init: %v2570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2574 = stablehlo.broadcast_in_dim %v2573, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2575 = stablehlo.divide %v2574, %v2571 : tensor<64x128x28x28xf32>
    %v2576 = stablehlo.subtract %v2569, %v2575 : tensor<64x128x28x28xf32>
    %v2577 = stablehlo.multiply %v2576, %v2576 : tensor<64x128x28x28xf32>
    %v2578 = stablehlo.reduce(%v2577 init: %v2570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2579 = stablehlo.broadcast_in_dim %v2578, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2580 = stablehlo.divide %v2579, %v2571 : tensor<64x128x28x28xf32>
    %v2581 = stablehlo.add %v2580, %v2572 : tensor<64x128x28x28xf32>
    %v2582 = stablehlo.rsqrt %v2581 : tensor<64x128x28x28xf32>
    %v2583 = stablehlo.multiply %v2576, %v2582 : tensor<64x128x28x28xf32>
    %v2584 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2585 = stablehlo.reshape %v2568 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2586 = stablehlo.multiply %v2584, %v2585 : tensor<64x128x28x28xf32>
    %v2587 = stablehlo.reduce(%v2586 init: %v2570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2588 = stablehlo.broadcast_in_dim %v2587, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2589 = stablehlo.multiply %v2583, %v2586 : tensor<64x128x28x28xf32>
    %v2590 = stablehlo.reduce(%v2589 init: %v2570) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2591 = stablehlo.broadcast_in_dim %v2590, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2592 = stablehlo.multiply %v2586, %v2571 : tensor<64x128x28x28xf32>
    %v2593 = stablehlo.subtract %v2592, %v2588 : tensor<64x128x28x28xf32>
    %v2594 = stablehlo.multiply %v2583, %v2591 : tensor<64x128x28x28xf32>
    %v2595 = stablehlo.subtract %v2593, %v2594 : tensor<64x128x28x28xf32>
    %v2596 = stablehlo.divide %v2582, %v2571 : tensor<64x128x28x28xf32>
    %v2597 = stablehlo.multiply %v2596, %v2595 : tensor<64x128x28x28xf32>
    %v2598 = stablehlo.reshape %v2597 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2599 = stablehlo.reshape %v2598 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2600 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2601 = stablehlo.transpose %v2600, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2602 = stablehlo.convert %v2599 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2603 = stablehlo.convert %v2601 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2604 = stablehlo.convolution(%v2602, %v2603)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2605 = stablehlo.convert %v2604 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2606 = stablehlo.reshape %v2605 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2607 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2608 = stablehlo.compare GT, %v455, %v2607 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2609 = stablehlo.select %v2608, %v2606, %v2607 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2610 = stablehlo.reshape %v435 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2612 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2613 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2614 = stablehlo.reduce(%v2610 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2615 = stablehlo.broadcast_in_dim %v2614, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2616 = stablehlo.divide %v2615, %v2612 : tensor<64x128x28x28xf32>
    %v2617 = stablehlo.subtract %v2610, %v2616 : tensor<64x128x28x28xf32>
    %v2618 = stablehlo.multiply %v2617, %v2617 : tensor<64x128x28x28xf32>
    %v2619 = stablehlo.reduce(%v2618 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2620 = stablehlo.broadcast_in_dim %v2619, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2621 = stablehlo.divide %v2620, %v2612 : tensor<64x128x28x28xf32>
    %v2622 = stablehlo.add %v2621, %v2613 : tensor<64x128x28x28xf32>
    %v2623 = stablehlo.rsqrt %v2622 : tensor<64x128x28x28xf32>
    %v2624 = stablehlo.multiply %v2617, %v2623 : tensor<64x128x28x28xf32>
    %v2625 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2626 = stablehlo.reshape %v2609 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2627 = stablehlo.multiply %v2625, %v2626 : tensor<64x128x28x28xf32>
    %v2628 = stablehlo.reduce(%v2627 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2628, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2630 = stablehlo.multiply %v2624, %v2627 : tensor<64x128x28x28xf32>
    %v2631 = stablehlo.reduce(%v2630 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2632 = stablehlo.broadcast_in_dim %v2631, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2633 = stablehlo.multiply %v2627, %v2612 : tensor<64x128x28x28xf32>
    %v2634 = stablehlo.subtract %v2633, %v2629 : tensor<64x128x28x28xf32>
    %v2635 = stablehlo.multiply %v2624, %v2632 : tensor<64x128x28x28xf32>
    %v2636 = stablehlo.subtract %v2634, %v2635 : tensor<64x128x28x28xf32>
    %v2637 = stablehlo.divide %v2623, %v2612 : tensor<64x128x28x28xf32>
    %v2638 = stablehlo.multiply %v2637, %v2636 : tensor<64x128x28x28xf32>
    %v2639 = stablehlo.reshape %v2638 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2640 = stablehlo.reshape %v2639 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2641 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2642 = stablehlo.transpose %v2641, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2643 = stablehlo.convert %v2640 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2644 = stablehlo.convert %v2642 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2645 = stablehlo.convolution(%v2643, %v2644)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2646 = stablehlo.convert %v2645 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2648 = stablehlo.add %v2647, %v2568 : tensor<64x100352xf32>
    %v2649 = stablehlo.reshape %v427 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2650 = stablehlo.reshape %v2639 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2651 = stablehlo.transpose %v2649, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2652 = stablehlo.transpose %v2650, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2653 = stablehlo.convert %v2651 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2654 = stablehlo.convert %v2652 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2655 = stablehlo.convolution(%v2653, %v2654)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2656 = stablehlo.convert %v2655 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2657 = stablehlo.transpose %v2656, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2658 = stablehlo.reshape %v435 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2659 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2660 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2661 = stablehlo.reduce(%v2658 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2662 = stablehlo.broadcast_in_dim %v2661, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2663 = stablehlo.divide %v2662, %v2660 : tensor<64x128x28x28xf32>
    %v2664 = stablehlo.subtract %v2658, %v2663 : tensor<64x128x28x28xf32>
    %v2665 = stablehlo.multiply %v2664, %v2664 : tensor<64x128x28x28xf32>
    %v2666 = stablehlo.reduce(%v2665 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2667 = stablehlo.broadcast_in_dim %v2666, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2668 = stablehlo.divide %v2667, %v2660 : tensor<64x128x28x28xf32>
    %v2669 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2670 = stablehlo.add %v2668, %v2669 : tensor<64x128x28x28xf32>
    %v2671 = stablehlo.rsqrt %v2670 : tensor<64x128x28x28xf32>
    %v2672 = stablehlo.multiply %v2664, %v2671 : tensor<64x128x28x28xf32>
    %v2673 = stablehlo.reshape %v2609 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2674 = stablehlo.multiply %v2673, %v2672 : tensor<64x128x28x28xf32>
    %v2675 = stablehlo.reduce(%v2674 init: %v2659) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2676 = stablehlo.reshape %v2609 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2677 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2678 = stablehlo.reduce(%v2676 init: %v2677) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2679 = stablehlo.reshape %v457 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2680 = stablehlo.reshape %v2598 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2681 = stablehlo.transpose %v2679, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2682 = stablehlo.transpose %v2680, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2683 = stablehlo.convert %v2681 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2684 = stablehlo.convert %v2682 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2685 = stablehlo.convolution(%v2683, %v2684)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2686 = stablehlo.convert %v2685 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2687 = stablehlo.transpose %v2686, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2688 = stablehlo.reshape %v465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2689 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2690 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2691 = stablehlo.reduce(%v2688 init: %v2689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2692 = stablehlo.broadcast_in_dim %v2691, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2693 = stablehlo.divide %v2692, %v2690 : tensor<64x128x28x28xf32>
    %v2694 = stablehlo.subtract %v2688, %v2693 : tensor<64x128x28x28xf32>
    %v2695 = stablehlo.multiply %v2694, %v2694 : tensor<64x128x28x28xf32>
    %v2696 = stablehlo.reduce(%v2695 init: %v2689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2697 = stablehlo.broadcast_in_dim %v2696, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2698 = stablehlo.divide %v2697, %v2690 : tensor<64x128x28x28xf32>
    %v2699 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2700 = stablehlo.add %v2698, %v2699 : tensor<64x128x28x28xf32>
    %v2701 = stablehlo.rsqrt %v2700 : tensor<64x128x28x28xf32>
    %v2702 = stablehlo.multiply %v2694, %v2701 : tensor<64x128x28x28xf32>
    %v2703 = stablehlo.reshape %v2568 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2704 = stablehlo.multiply %v2703, %v2702 : tensor<64x128x28x28xf32>
    %v2705 = stablehlo.reduce(%v2704 init: %v2689) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2706 = stablehlo.reshape %v2568 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2708 = stablehlo.reduce(%v2706 init: %v2707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2709 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2710 = stablehlo.compare GT, %v425, %v2709 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2711 = stablehlo.select %v2710, %v2648, %v2709 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2712 = stablehlo.reshape %v404 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2713 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2714 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2715 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2716 = stablehlo.reduce(%v2712 init: %v2713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2717 = stablehlo.broadcast_in_dim %v2716, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2718 = stablehlo.divide %v2717, %v2714 : tensor<64x128x28x28xf32>
    %v2719 = stablehlo.subtract %v2712, %v2718 : tensor<64x128x28x28xf32>
    %v2720 = stablehlo.multiply %v2719, %v2719 : tensor<64x128x28x28xf32>
    %v2721 = stablehlo.reduce(%v2720 init: %v2713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2722 = stablehlo.broadcast_in_dim %v2721, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2723 = stablehlo.divide %v2722, %v2714 : tensor<64x128x28x28xf32>
    %v2724 = stablehlo.add %v2723, %v2715 : tensor<64x128x28x28xf32>
    %v2725 = stablehlo.rsqrt %v2724 : tensor<64x128x28x28xf32>
    %v2726 = stablehlo.multiply %v2719, %v2725 : tensor<64x128x28x28xf32>
    %v2727 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2728 = stablehlo.reshape %v2711 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2729 = stablehlo.multiply %v2727, %v2728 : tensor<64x128x28x28xf32>
    %v2730 = stablehlo.reduce(%v2729 init: %v2713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2731 = stablehlo.broadcast_in_dim %v2730, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2732 = stablehlo.multiply %v2726, %v2729 : tensor<64x128x28x28xf32>
    %v2733 = stablehlo.reduce(%v2732 init: %v2713) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2733, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2735 = stablehlo.multiply %v2729, %v2714 : tensor<64x128x28x28xf32>
    %v2736 = stablehlo.subtract %v2735, %v2731 : tensor<64x128x28x28xf32>
    %v2737 = stablehlo.multiply %v2726, %v2734 : tensor<64x128x28x28xf32>
    %v2738 = stablehlo.subtract %v2736, %v2737 : tensor<64x128x28x28xf32>
    %v2739 = stablehlo.divide %v2725, %v2714 : tensor<64x128x28x28xf32>
    %v2740 = stablehlo.multiply %v2739, %v2738 : tensor<64x128x28x28xf32>
    %v2741 = stablehlo.reshape %v2740 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2742 = stablehlo.reshape %v2741 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2743 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2744 = stablehlo.transpose %v2743, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2745 = stablehlo.convert %v2742 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2746 = stablehlo.convert %v2744 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2747 = stablehlo.convolution(%v2745, %v2746)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2748 = stablehlo.convert %v2747 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2750 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2751 = stablehlo.compare GT, %v394, %v2750 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2752 = stablehlo.select %v2751, %v2749, %v2750 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2753 = stablehlo.reshape %v374 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2754 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2755 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2756 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2757 = stablehlo.reduce(%v2753 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2758 = stablehlo.broadcast_in_dim %v2757, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2759 = stablehlo.divide %v2758, %v2755 : tensor<64x128x28x28xf32>
    %v2760 = stablehlo.subtract %v2753, %v2759 : tensor<64x128x28x28xf32>
    %v2761 = stablehlo.multiply %v2760, %v2760 : tensor<64x128x28x28xf32>
    %v2762 = stablehlo.reduce(%v2761 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2763 = stablehlo.broadcast_in_dim %v2762, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2764 = stablehlo.divide %v2763, %v2755 : tensor<64x128x28x28xf32>
    %v2765 = stablehlo.add %v2764, %v2756 : tensor<64x128x28x28xf32>
    %v2766 = stablehlo.rsqrt %v2765 : tensor<64x128x28x28xf32>
    %v2767 = stablehlo.multiply %v2760, %v2766 : tensor<64x128x28x28xf32>
    %v2768 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2769 = stablehlo.reshape %v2752 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2770 = stablehlo.multiply %v2768, %v2769 : tensor<64x128x28x28xf32>
    %v2771 = stablehlo.reduce(%v2770 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2772 = stablehlo.broadcast_in_dim %v2771, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2773 = stablehlo.multiply %v2767, %v2770 : tensor<64x128x28x28xf32>
    %v2774 = stablehlo.reduce(%v2773 init: %v2754) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2775 = stablehlo.broadcast_in_dim %v2774, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2776 = stablehlo.multiply %v2770, %v2755 : tensor<64x128x28x28xf32>
    %v2777 = stablehlo.subtract %v2776, %v2772 : tensor<64x128x28x28xf32>
    %v2778 = stablehlo.multiply %v2767, %v2775 : tensor<64x128x28x28xf32>
    %v2779 = stablehlo.subtract %v2777, %v2778 : tensor<64x128x28x28xf32>
    %v2780 = stablehlo.divide %v2766, %v2755 : tensor<64x128x28x28xf32>
    %v2781 = stablehlo.multiply %v2780, %v2779 : tensor<64x128x28x28xf32>
    %v2782 = stablehlo.reshape %v2781 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2783 = stablehlo.reshape %v2782 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2784 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2785 = stablehlo.transpose %v2784, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2786 = stablehlo.convert %v2783 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2787 = stablehlo.convert %v2785 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2788 = stablehlo.convolution(%v2786, %v2787)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2789 = stablehlo.convert %v2788 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2790 = stablehlo.reshape %v2789 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2791 = stablehlo.add %v2790, %v2711 : tensor<64x100352xf32>
    %v2792 = stablehlo.reshape %v366 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2793 = stablehlo.reshape %v2782 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2794 = stablehlo.transpose %v2792, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2795 = stablehlo.transpose %v2793, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2796 = stablehlo.convert %v2794 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2797 = stablehlo.convert %v2795 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2798 = stablehlo.convolution(%v2796, %v2797)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2799 = stablehlo.convert %v2798 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2800 = stablehlo.transpose %v2799, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2801 = stablehlo.reshape %v374 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2802 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2803 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2804 = stablehlo.reduce(%v2801 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2805 = stablehlo.broadcast_in_dim %v2804, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2806 = stablehlo.divide %v2805, %v2803 : tensor<64x128x28x28xf32>
    %v2807 = stablehlo.subtract %v2801, %v2806 : tensor<64x128x28x28xf32>
    %v2808 = stablehlo.multiply %v2807, %v2807 : tensor<64x128x28x28xf32>
    %v2809 = stablehlo.reduce(%v2808 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2810 = stablehlo.broadcast_in_dim %v2809, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2811 = stablehlo.divide %v2810, %v2803 : tensor<64x128x28x28xf32>
    %v2812 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2813 = stablehlo.add %v2811, %v2812 : tensor<64x128x28x28xf32>
    %v2814 = stablehlo.rsqrt %v2813 : tensor<64x128x28x28xf32>
    %v2815 = stablehlo.multiply %v2807, %v2814 : tensor<64x128x28x28xf32>
    %v2816 = stablehlo.reshape %v2752 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2817 = stablehlo.multiply %v2816, %v2815 : tensor<64x128x28x28xf32>
    %v2818 = stablehlo.reduce(%v2817 init: %v2802) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2819 = stablehlo.reshape %v2752 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2820 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2821 = stablehlo.reduce(%v2819 init: %v2820) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2822 = stablehlo.reshape %v396 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2823 = stablehlo.reshape %v2741 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2824 = stablehlo.transpose %v2822, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2825 = stablehlo.transpose %v2823, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2826 = stablehlo.convert %v2824 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2827 = stablehlo.convert %v2825 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2828 = stablehlo.convolution(%v2826, %v2827)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2829 = stablehlo.convert %v2828 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2830 = stablehlo.transpose %v2829, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2831 = stablehlo.reshape %v404 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2832 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2833 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2834 = stablehlo.reduce(%v2831 init: %v2832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2835 = stablehlo.broadcast_in_dim %v2834, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2836 = stablehlo.divide %v2835, %v2833 : tensor<64x128x28x28xf32>
    %v2837 = stablehlo.subtract %v2831, %v2836 : tensor<64x128x28x28xf32>
    %v2838 = stablehlo.multiply %v2837, %v2837 : tensor<64x128x28x28xf32>
    %v2839 = stablehlo.reduce(%v2838 init: %v2832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2840 = stablehlo.broadcast_in_dim %v2839, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2841 = stablehlo.divide %v2840, %v2833 : tensor<64x128x28x28xf32>
    %v2842 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2843 = stablehlo.add %v2841, %v2842 : tensor<64x128x28x28xf32>
    %v2844 = stablehlo.rsqrt %v2843 : tensor<64x128x28x28xf32>
    %v2845 = stablehlo.multiply %v2837, %v2844 : tensor<64x128x28x28xf32>
    %v2846 = stablehlo.reshape %v2711 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2847 = stablehlo.multiply %v2846, %v2845 : tensor<64x128x28x28xf32>
    %v2848 = stablehlo.reduce(%v2847 init: %v2832) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2849 = stablehlo.reshape %v2711 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2850 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2851 = stablehlo.reduce(%v2849 init: %v2850) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2852 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2853 = stablehlo.compare GT, %v364, %v2852 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2854 = stablehlo.select %v2853, %v2791, %v2852 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2855 = stablehlo.reshape %v343 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2856 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2857 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2858 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2859 = stablehlo.reduce(%v2855 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2860 = stablehlo.broadcast_in_dim %v2859, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2861 = stablehlo.divide %v2860, %v2857 : tensor<64x128x28x28xf32>
    %v2862 = stablehlo.subtract %v2855, %v2861 : tensor<64x128x28x28xf32>
    %v2863 = stablehlo.multiply %v2862, %v2862 : tensor<64x128x28x28xf32>
    %v2864 = stablehlo.reduce(%v2863 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2865 = stablehlo.broadcast_in_dim %v2864, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2866 = stablehlo.divide %v2865, %v2857 : tensor<64x128x28x28xf32>
    %v2867 = stablehlo.add %v2866, %v2858 : tensor<64x128x28x28xf32>
    %v2868 = stablehlo.rsqrt %v2867 : tensor<64x128x28x28xf32>
    %v2869 = stablehlo.multiply %v2862, %v2868 : tensor<64x128x28x28xf32>
    %v2870 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2871 = stablehlo.reshape %v2854 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2872 = stablehlo.multiply %v2870, %v2871 : tensor<64x128x28x28xf32>
    %v2873 = stablehlo.reduce(%v2872 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2874 = stablehlo.broadcast_in_dim %v2873, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2875 = stablehlo.multiply %v2869, %v2872 : tensor<64x128x28x28xf32>
    %v2876 = stablehlo.reduce(%v2875 init: %v2856) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2877 = stablehlo.broadcast_in_dim %v2876, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2878 = stablehlo.multiply %v2872, %v2857 : tensor<64x128x28x28xf32>
    %v2879 = stablehlo.subtract %v2878, %v2874 : tensor<64x128x28x28xf32>
    %v2880 = stablehlo.multiply %v2869, %v2877 : tensor<64x128x28x28xf32>
    %v2881 = stablehlo.subtract %v2879, %v2880 : tensor<64x128x28x28xf32>
    %v2882 = stablehlo.divide %v2868, %v2857 : tensor<64x128x28x28xf32>
    %v2883 = stablehlo.multiply %v2882, %v2881 : tensor<64x128x28x28xf32>
    %v2884 = stablehlo.reshape %v2883 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2885 = stablehlo.reshape %v2884 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2886 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2887 = stablehlo.transpose %v2886, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2888 = stablehlo.convert %v2885 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2889 = stablehlo.convert %v2887 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2890 = stablehlo.convolution(%v2888, %v2889)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2891 = stablehlo.convert %v2890 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2892 = stablehlo.reshape %v2891 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2893 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2894 = stablehlo.compare GT, %v333, %v2893 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2895 = stablehlo.select %v2894, %v2892, %v2893 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2896 = stablehlo.reshape %v313 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2897 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2898 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2899 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2900 = stablehlo.reduce(%v2896 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2901 = stablehlo.broadcast_in_dim %v2900, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2902 = stablehlo.divide %v2901, %v2898 : tensor<64x128x28x28xf32>
    %v2903 = stablehlo.subtract %v2896, %v2902 : tensor<64x128x28x28xf32>
    %v2904 = stablehlo.multiply %v2903, %v2903 : tensor<64x128x28x28xf32>
    %v2905 = stablehlo.reduce(%v2904 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2905, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2907 = stablehlo.divide %v2906, %v2898 : tensor<64x128x28x28xf32>
    %v2908 = stablehlo.add %v2907, %v2899 : tensor<64x128x28x28xf32>
    %v2909 = stablehlo.rsqrt %v2908 : tensor<64x128x28x28xf32>
    %v2910 = stablehlo.multiply %v2903, %v2909 : tensor<64x128x28x28xf32>
    %v2911 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2912 = stablehlo.reshape %v2895 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2913 = stablehlo.multiply %v2911, %v2912 : tensor<64x128x28x28xf32>
    %v2914 = stablehlo.reduce(%v2913 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2915 = stablehlo.broadcast_in_dim %v2914, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2916 = stablehlo.multiply %v2910, %v2913 : tensor<64x128x28x28xf32>
    %v2917 = stablehlo.reduce(%v2916 init: %v2897) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2918 = stablehlo.broadcast_in_dim %v2917, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2919 = stablehlo.multiply %v2913, %v2898 : tensor<64x128x28x28xf32>
    %v2920 = stablehlo.subtract %v2919, %v2915 : tensor<64x128x28x28xf32>
    %v2921 = stablehlo.multiply %v2910, %v2918 : tensor<64x128x28x28xf32>
    %v2922 = stablehlo.subtract %v2920, %v2921 : tensor<64x128x28x28xf32>
    %v2923 = stablehlo.divide %v2909, %v2898 : tensor<64x128x28x28xf32>
    %v2924 = stablehlo.multiply %v2923, %v2922 : tensor<64x128x28x28xf32>
    %v2925 = stablehlo.reshape %v2924 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2926 = stablehlo.reshape %v2925 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2927 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v2928 = stablehlo.transpose %v2927, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2929 = stablehlo.convert %v2926 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v2930 = stablehlo.convert %v2928 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v2931 = stablehlo.convolution(%v2929, %v2930)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v2932 = stablehlo.convert %v2931 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v2934 = stablehlo.add %v2933, %v2854 : tensor<64x100352xf32>
    %v2935 = stablehlo.reshape %v305 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2936 = stablehlo.reshape %v2925 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2937 = stablehlo.transpose %v2935, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2938 = stablehlo.transpose %v2936, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2939 = stablehlo.convert %v2937 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2940 = stablehlo.convert %v2938 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2941 = stablehlo.convolution(%v2939, %v2940)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2942 = stablehlo.convert %v2941 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2943 = stablehlo.transpose %v2942, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2944 = stablehlo.reshape %v313 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2946 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2947 = stablehlo.reduce(%v2944 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2948 = stablehlo.broadcast_in_dim %v2947, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2949 = stablehlo.divide %v2948, %v2946 : tensor<64x128x28x28xf32>
    %v2950 = stablehlo.subtract %v2944, %v2949 : tensor<64x128x28x28xf32>
    %v2951 = stablehlo.multiply %v2950, %v2950 : tensor<64x128x28x28xf32>
    %v2952 = stablehlo.reduce(%v2951 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2953 = stablehlo.broadcast_in_dim %v2952, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2954 = stablehlo.divide %v2953, %v2946 : tensor<64x128x28x28xf32>
    %v2955 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2956 = stablehlo.add %v2954, %v2955 : tensor<64x128x28x28xf32>
    %v2957 = stablehlo.rsqrt %v2956 : tensor<64x128x28x28xf32>
    %v2958 = stablehlo.multiply %v2950, %v2957 : tensor<64x128x28x28xf32>
    %v2959 = stablehlo.reshape %v2895 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2960 = stablehlo.multiply %v2959, %v2958 : tensor<64x128x28x28xf32>
    %v2961 = stablehlo.reduce(%v2960 init: %v2945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2962 = stablehlo.reshape %v2895 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2963 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2964 = stablehlo.reduce(%v2962 init: %v2963) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2965 = stablehlo.reshape %v335 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2966 = stablehlo.reshape %v2884 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2967 = stablehlo.transpose %v2965, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2968 = stablehlo.transpose %v2966, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v2969 = stablehlo.convert %v2967 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2970 = stablehlo.convert %v2968 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v2971 = stablehlo.convolution(%v2969, %v2970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v2972 = stablehlo.convert %v2971 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v2973 = stablehlo.transpose %v2972, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v2974 = stablehlo.reshape %v343 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2976 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v2977 = stablehlo.reduce(%v2974 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2978 = stablehlo.broadcast_in_dim %v2977, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2979 = stablehlo.divide %v2978, %v2976 : tensor<64x128x28x28xf32>
    %v2980 = stablehlo.subtract %v2974, %v2979 : tensor<64x128x28x28xf32>
    %v2981 = stablehlo.multiply %v2980, %v2980 : tensor<64x128x28x28xf32>
    %v2982 = stablehlo.reduce(%v2981 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2983 = stablehlo.broadcast_in_dim %v2982, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v2984 = stablehlo.divide %v2983, %v2976 : tensor<64x128x28x28xf32>
    %v2985 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v2986 = stablehlo.add %v2984, %v2985 : tensor<64x128x28x28xf32>
    %v2987 = stablehlo.rsqrt %v2986 : tensor<64x128x28x28xf32>
    %v2988 = stablehlo.multiply %v2980, %v2987 : tensor<64x128x28x28xf32>
    %v2989 = stablehlo.reshape %v2854 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2990 = stablehlo.multiply %v2989, %v2988 : tensor<64x128x28x28xf32>
    %v2991 = stablehlo.reduce(%v2990 init: %v2975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2992 = stablehlo.reshape %v2854 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2993 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2994 = stablehlo.reduce(%v2992 init: %v2993) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v2995 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v2996 = stablehlo.compare GT, %v303, %v2995 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v2997 = stablehlo.select %v2996, %v2934, %v2995 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v2998 = stablehlo.reshape %v254 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v2999 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3000 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3001 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3002 = stablehlo.reduce(%v2998 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3003 = stablehlo.broadcast_in_dim %v3002, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3004 = stablehlo.divide %v3003, %v3000 : tensor<64x128x28x28xf32>
    %v3005 = stablehlo.subtract %v2998, %v3004 : tensor<64x128x28x28xf32>
    %v3006 = stablehlo.multiply %v3005, %v3005 : tensor<64x128x28x28xf32>
    %v3007 = stablehlo.reduce(%v3006 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3008 = stablehlo.broadcast_in_dim %v3007, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3009 = stablehlo.divide %v3008, %v3000 : tensor<64x128x28x28xf32>
    %v3010 = stablehlo.add %v3009, %v3001 : tensor<64x128x28x28xf32>
    %v3011 = stablehlo.rsqrt %v3010 : tensor<64x128x28x28xf32>
    %v3012 = stablehlo.multiply %v3005, %v3011 : tensor<64x128x28x28xf32>
    %v3013 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3014 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3015 = stablehlo.multiply %v3013, %v3014 : tensor<64x128x28x28xf32>
    %v3016 = stablehlo.reduce(%v3015 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3017 = stablehlo.broadcast_in_dim %v3016, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3018 = stablehlo.multiply %v3012, %v3015 : tensor<64x128x28x28xf32>
    %v3019 = stablehlo.reduce(%v3018 init: %v2999) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3020 = stablehlo.broadcast_in_dim %v3019, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3021 = stablehlo.multiply %v3015, %v3000 : tensor<64x128x28x28xf32>
    %v3022 = stablehlo.subtract %v3021, %v3017 : tensor<64x128x28x28xf32>
    %v3023 = stablehlo.multiply %v3012, %v3020 : tensor<64x128x28x28xf32>
    %v3024 = stablehlo.subtract %v3022, %v3023 : tensor<64x128x28x28xf32>
    %v3025 = stablehlo.divide %v3011, %v3000 : tensor<64x128x28x28xf32>
    %v3026 = stablehlo.multiply %v3025, %v3024 : tensor<64x128x28x28xf32>
    %v3027 = stablehlo.reshape %v3026 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3028 = stablehlo.reshape %v3027 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3029 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3030 = stablehlo.transpose %v3029, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3031 = stablehlo.convert %v3028 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3032 = stablehlo.convert %v3030 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3033 = stablehlo.convolution(%v3031, %v3032)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3034 = stablehlo.convert %v3033 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3035 = stablehlo.reshape %v3034 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3036 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v3037 = stablehlo.compare GT, %v244, %v3036 : (tensor<64x100352xf32>, tensor<64x100352xf32>) -> tensor<64x100352xi1>
    %v3038 = stablehlo.select %v3037, %v3035, %v3036 : tensor<64x100352xi1>, tensor<64x100352xf32>
    %v3039 = stablehlo.reshape %v224 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3040 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3041 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3042 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3043 = stablehlo.reduce(%v3039 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3044 = stablehlo.broadcast_in_dim %v3043, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3045 = stablehlo.divide %v3044, %v3041 : tensor<64x128x28x28xf32>
    %v3046 = stablehlo.subtract %v3039, %v3045 : tensor<64x128x28x28xf32>
    %v3047 = stablehlo.multiply %v3046, %v3046 : tensor<64x128x28x28xf32>
    %v3048 = stablehlo.reduce(%v3047 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3049 = stablehlo.broadcast_in_dim %v3048, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3050 = stablehlo.divide %v3049, %v3041 : tensor<64x128x28x28xf32>
    %v3051 = stablehlo.add %v3050, %v3042 : tensor<64x128x28x28xf32>
    %v3052 = stablehlo.rsqrt %v3051 : tensor<64x128x28x28xf32>
    %v3053 = stablehlo.multiply %v3046, %v3052 : tensor<64x128x28x28xf32>
    %v3054 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3055 = stablehlo.reshape %v3038 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3056 = stablehlo.multiply %v3054, %v3055 : tensor<64x128x28x28xf32>
    %v3057 = stablehlo.reduce(%v3056 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3058 = stablehlo.broadcast_in_dim %v3057, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3059 = stablehlo.multiply %v3053, %v3056 : tensor<64x128x28x28xf32>
    %v3060 = stablehlo.reduce(%v3059 init: %v3040) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3061 = stablehlo.broadcast_in_dim %v3060, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3062 = stablehlo.multiply %v3056, %v3041 : tensor<64x128x28x28xf32>
    %v3063 = stablehlo.subtract %v3062, %v3058 : tensor<64x128x28x28xf32>
    %v3064 = stablehlo.multiply %v3053, %v3061 : tensor<64x128x28x28xf32>
    %v3065 = stablehlo.subtract %v3063, %v3064 : tensor<64x128x28x28xf32>
    %v3066 = stablehlo.divide %v3052, %v3041 : tensor<64x128x28x28xf32>
    %v3067 = stablehlo.multiply %v3066, %v3065 : tensor<64x128x28x28xf32>
    %v3068 = stablehlo.reshape %v3067 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3069 = stablehlo.reshape %v3068 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3070 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3071 = stablehlo.pad %v3069, %v3070, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v3072 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v3073 = stablehlo.transpose %v3072, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v3074 = stablehlo.convert %v3071 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v3075 = stablehlo.convert %v3073 : (tensor<64x128x3x3xf32>) -> tensor<64x128x3x3xbf16>
    %v3076 = stablehlo.convolution(%v3074, %v3075)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<64x128x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3077 = stablehlo.convert %v3076 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3078 = stablehlo.reshape %v3077 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3079 = stablehlo.reshape %v282 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3080 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3081 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3082 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3083 = stablehlo.reduce(%v3079 init: %v3080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3084 = stablehlo.broadcast_in_dim %v3083, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3085 = stablehlo.divide %v3084, %v3081 : tensor<64x128x28x28xf32>
    %v3086 = stablehlo.subtract %v3079, %v3085 : tensor<64x128x28x28xf32>
    %v3087 = stablehlo.multiply %v3086, %v3086 : tensor<64x128x28x28xf32>
    %v3088 = stablehlo.reduce(%v3087 init: %v3080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3089 = stablehlo.broadcast_in_dim %v3088, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3090 = stablehlo.divide %v3089, %v3081 : tensor<64x128x28x28xf32>
    %v3091 = stablehlo.add %v3090, %v3082 : tensor<64x128x28x28xf32>
    %v3092 = stablehlo.rsqrt %v3091 : tensor<64x128x28x28xf32>
    %v3093 = stablehlo.multiply %v3086, %v3092 : tensor<64x128x28x28xf32>
    %v3094 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3095 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3096 = stablehlo.multiply %v3094, %v3095 : tensor<64x128x28x28xf32>
    %v3097 = stablehlo.reduce(%v3096 init: %v3080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3098 = stablehlo.broadcast_in_dim %v3097, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3099 = stablehlo.multiply %v3093, %v3096 : tensor<64x128x28x28xf32>
    %v3100 = stablehlo.reduce(%v3099 init: %v3080) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3101 = stablehlo.broadcast_in_dim %v3100, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3102 = stablehlo.multiply %v3096, %v3081 : tensor<64x128x28x28xf32>
    %v3103 = stablehlo.subtract %v3102, %v3098 : tensor<64x128x28x28xf32>
    %v3104 = stablehlo.multiply %v3093, %v3101 : tensor<64x128x28x28xf32>
    %v3105 = stablehlo.subtract %v3103, %v3104 : tensor<64x128x28x28xf32>
    %v3106 = stablehlo.divide %v3092, %v3081 : tensor<64x128x28x28xf32>
    %v3107 = stablehlo.multiply %v3106, %v3105 : tensor<64x128x28x28xf32>
    %v3108 = stablehlo.reshape %v3107 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3109 = stablehlo.reshape %v3108 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3111 = stablehlo.pad %v3109, %v3110, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v3112 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v3113 = stablehlo.transpose %v3112, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v3114 = stablehlo.convert %v3111 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v3115 = stablehlo.convert %v3113 : (tensor<64x128x1x1xf32>) -> tensor<64x128x1x1xbf16>
    %v3116 = stablehlo.convolution(%v3114, %v3115)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<64x128x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v3117 = stablehlo.convert %v3116 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3118 = stablehlo.reshape %v3117 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3119 = stablehlo.add %v3078, %v3118 : tensor<64x200704xf32>
    %v3120 = stablehlo.reshape %v216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3121 = stablehlo.reshape %v3068 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3122 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3123 = stablehlo.pad %v3121, %v3122, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v3124 = stablehlo.transpose %v3120, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3125 = stablehlo.transpose %v3123, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v3126 = stablehlo.convert %v3124 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3127 = stablehlo.convert %v3125 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v3128 = stablehlo.convolution(%v3126, %v3127)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<64x128x3x3xbf16>
    %v3129 = stablehlo.convert %v3128 : (tensor<64x128x3x3xbf16>) -> tensor<64x128x3x3xf32>
    %v3130 = stablehlo.transpose %v3129, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v3131 = stablehlo.reshape %v224 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3132 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3133 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3134 = stablehlo.reduce(%v3131 init: %v3132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3135 = stablehlo.broadcast_in_dim %v3134, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3136 = stablehlo.divide %v3135, %v3133 : tensor<64x128x28x28xf32>
    %v3137 = stablehlo.subtract %v3131, %v3136 : tensor<64x128x28x28xf32>
    %v3138 = stablehlo.multiply %v3137, %v3137 : tensor<64x128x28x28xf32>
    %v3139 = stablehlo.reduce(%v3138 init: %v3132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3140 = stablehlo.broadcast_in_dim %v3139, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3141 = stablehlo.divide %v3140, %v3133 : tensor<64x128x28x28xf32>
    %v3142 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3143 = stablehlo.add %v3141, %v3142 : tensor<64x128x28x28xf32>
    %v3144 = stablehlo.rsqrt %v3143 : tensor<64x128x28x28xf32>
    %v3145 = stablehlo.multiply %v3137, %v3144 : tensor<64x128x28x28xf32>
    %v3146 = stablehlo.reshape %v3038 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3147 = stablehlo.multiply %v3146, %v3145 : tensor<64x128x28x28xf32>
    %v3148 = stablehlo.reduce(%v3147 init: %v3132) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3149 = stablehlo.reshape %v3038 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3150 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3151 = stablehlo.reduce(%v3149 init: %v3150) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3152 = stablehlo.reshape %v246 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3153 = stablehlo.reshape %v3027 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3154 = stablehlo.transpose %v3152, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3155 = stablehlo.transpose %v3153, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3156 = stablehlo.convert %v3154 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3157 = stablehlo.convert %v3155 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3158 = stablehlo.convolution(%v3156, %v3157)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3159 = stablehlo.convert %v3158 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3160 = stablehlo.transpose %v3159, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3161 = stablehlo.reshape %v254 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3163 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3164 = stablehlo.reduce(%v3161 init: %v3162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3165 = stablehlo.broadcast_in_dim %v3164, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3166 = stablehlo.divide %v3165, %v3163 : tensor<64x128x28x28xf32>
    %v3167 = stablehlo.subtract %v3161, %v3166 : tensor<64x128x28x28xf32>
    %v3168 = stablehlo.multiply %v3167, %v3167 : tensor<64x128x28x28xf32>
    %v3169 = stablehlo.reduce(%v3168 init: %v3162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3170 = stablehlo.broadcast_in_dim %v3169, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3171 = stablehlo.divide %v3170, %v3163 : tensor<64x128x28x28xf32>
    %v3172 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3173 = stablehlo.add %v3171, %v3172 : tensor<64x128x28x28xf32>
    %v3174 = stablehlo.rsqrt %v3173 : tensor<64x128x28x28xf32>
    %v3175 = stablehlo.multiply %v3167, %v3174 : tensor<64x128x28x28xf32>
    %v3176 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3177 = stablehlo.multiply %v3176, %v3175 : tensor<64x128x28x28xf32>
    %v3178 = stablehlo.reduce(%v3177 init: %v3162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3179 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3180 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3181 = stablehlo.reduce(%v3179 init: %v3180) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3182 = stablehlo.reshape %v216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3183 = stablehlo.reshape %v3108 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3184 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3185 = stablehlo.pad %v3183, %v3184, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v3186 = stablehlo.transpose %v3182, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3187 = stablehlo.transpose %v3185, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v3188 = stablehlo.convert %v3186 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3189 = stablehlo.convert %v3187 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v3190 = stablehlo.convolution(%v3188, %v3189)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<64x128x1x1xbf16>
    %v3191 = stablehlo.convert %v3190 : (tensor<64x128x1x1xbf16>) -> tensor<64x128x1x1xf32>
    %v3192 = stablehlo.transpose %v3191, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v3193 = stablehlo.reshape %v282 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3195 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3196 = stablehlo.reduce(%v3193 init: %v3194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3197 = stablehlo.broadcast_in_dim %v3196, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3198 = stablehlo.divide %v3197, %v3195 : tensor<64x128x28x28xf32>
    %v3199 = stablehlo.subtract %v3193, %v3198 : tensor<64x128x28x28xf32>
    %v3200 = stablehlo.multiply %v3199, %v3199 : tensor<64x128x28x28xf32>
    %v3201 = stablehlo.reduce(%v3200 init: %v3194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3202 = stablehlo.broadcast_in_dim %v3201, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3203 = stablehlo.divide %v3202, %v3195 : tensor<64x128x28x28xf32>
    %v3204 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3205 = stablehlo.add %v3203, %v3204 : tensor<64x128x28x28xf32>
    %v3206 = stablehlo.rsqrt %v3205 : tensor<64x128x28x28xf32>
    %v3207 = stablehlo.multiply %v3199, %v3206 : tensor<64x128x28x28xf32>
    %v3208 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3209 = stablehlo.multiply %v3208, %v3207 : tensor<64x128x28x28xf32>
    %v3210 = stablehlo.reduce(%v3209 init: %v3194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3211 = stablehlo.reshape %v2997 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3212 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3213 = stablehlo.reduce(%v3211 init: %v3212) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3214 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3215 = stablehlo.compare GT, %v214, %v3214 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3216 = stablehlo.select %v3215, %v3119, %v3214 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3217 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3219 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3220 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3221 = stablehlo.reduce(%v3217 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3222 = stablehlo.broadcast_in_dim %v3221, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3223 = stablehlo.divide %v3222, %v3219 : tensor<64x64x56x56xf32>
    %v3224 = stablehlo.subtract %v3217, %v3223 : tensor<64x64x56x56xf32>
    %v3225 = stablehlo.multiply %v3224, %v3224 : tensor<64x64x56x56xf32>
    %v3226 = stablehlo.reduce(%v3225 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3227 = stablehlo.broadcast_in_dim %v3226, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3228 = stablehlo.divide %v3227, %v3219 : tensor<64x64x56x56xf32>
    %v3229 = stablehlo.add %v3228, %v3220 : tensor<64x64x56x56xf32>
    %v3230 = stablehlo.rsqrt %v3229 : tensor<64x64x56x56xf32>
    %v3231 = stablehlo.multiply %v3224, %v3230 : tensor<64x64x56x56xf32>
    %v3232 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3233 = stablehlo.reshape %v3216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3234 = stablehlo.multiply %v3232, %v3233 : tensor<64x64x56x56xf32>
    %v3235 = stablehlo.reduce(%v3234 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3236 = stablehlo.broadcast_in_dim %v3235, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3237 = stablehlo.multiply %v3231, %v3234 : tensor<64x64x56x56xf32>
    %v3238 = stablehlo.reduce(%v3237 init: %v3218) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3239 = stablehlo.broadcast_in_dim %v3238, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3240 = stablehlo.multiply %v3234, %v3219 : tensor<64x64x56x56xf32>
    %v3241 = stablehlo.subtract %v3240, %v3236 : tensor<64x64x56x56xf32>
    %v3242 = stablehlo.multiply %v3231, %v3239 : tensor<64x64x56x56xf32>
    %v3243 = stablehlo.subtract %v3241, %v3242 : tensor<64x64x56x56xf32>
    %v3244 = stablehlo.divide %v3230, %v3219 : tensor<64x64x56x56xf32>
    %v3245 = stablehlo.multiply %v3244, %v3243 : tensor<64x64x56x56xf32>
    %v3246 = stablehlo.reshape %v3245 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3247 = stablehlo.reshape %v3246 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3248 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3249 = stablehlo.transpose %v3248, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3250 = stablehlo.convert %v3247 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3251 = stablehlo.convert %v3249 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3252 = stablehlo.convolution(%v3250, %v3251)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3253 = stablehlo.convert %v3252 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3254 = stablehlo.reshape %v3253 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3255 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3256 = stablehlo.compare GT, %v183, %v3255 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3257 = stablehlo.select %v3256, %v3254, %v3255 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3258 = stablehlo.reshape %v163 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3260 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3261 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3262 = stablehlo.reduce(%v3258 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3263 = stablehlo.broadcast_in_dim %v3262, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3264 = stablehlo.divide %v3263, %v3260 : tensor<64x64x56x56xf32>
    %v3265 = stablehlo.subtract %v3258, %v3264 : tensor<64x64x56x56xf32>
    %v3266 = stablehlo.multiply %v3265, %v3265 : tensor<64x64x56x56xf32>
    %v3267 = stablehlo.reduce(%v3266 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3268 = stablehlo.broadcast_in_dim %v3267, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3269 = stablehlo.divide %v3268, %v3260 : tensor<64x64x56x56xf32>
    %v3270 = stablehlo.add %v3269, %v3261 : tensor<64x64x56x56xf32>
    %v3271 = stablehlo.rsqrt %v3270 : tensor<64x64x56x56xf32>
    %v3272 = stablehlo.multiply %v3265, %v3271 : tensor<64x64x56x56xf32>
    %v3273 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3274 = stablehlo.reshape %v3257 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3275 = stablehlo.multiply %v3273, %v3274 : tensor<64x64x56x56xf32>
    %v3276 = stablehlo.reduce(%v3275 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3277 = stablehlo.broadcast_in_dim %v3276, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3278 = stablehlo.multiply %v3272, %v3275 : tensor<64x64x56x56xf32>
    %v3279 = stablehlo.reduce(%v3278 init: %v3259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3280 = stablehlo.broadcast_in_dim %v3279, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3281 = stablehlo.multiply %v3275, %v3260 : tensor<64x64x56x56xf32>
    %v3282 = stablehlo.subtract %v3281, %v3277 : tensor<64x64x56x56xf32>
    %v3283 = stablehlo.multiply %v3272, %v3280 : tensor<64x64x56x56xf32>
    %v3284 = stablehlo.subtract %v3282, %v3283 : tensor<64x64x56x56xf32>
    %v3285 = stablehlo.divide %v3271, %v3260 : tensor<64x64x56x56xf32>
    %v3286 = stablehlo.multiply %v3285, %v3284 : tensor<64x64x56x56xf32>
    %v3287 = stablehlo.reshape %v3286 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3288 = stablehlo.reshape %v3287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3289 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3290 = stablehlo.transpose %v3289, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3291 = stablehlo.convert %v3288 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3292 = stablehlo.convert %v3290 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3293 = stablehlo.convolution(%v3291, %v3292)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3294 = stablehlo.convert %v3293 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3295 = stablehlo.reshape %v3294 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3296 = stablehlo.add %v3295, %v3216 : tensor<64x200704xf32>
    %v3297 = stablehlo.reshape %v155 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3298 = stablehlo.reshape %v3287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3299 = stablehlo.transpose %v3297, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3300 = stablehlo.transpose %v3298, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3301 = stablehlo.convert %v3299 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3302 = stablehlo.convert %v3300 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3303 = stablehlo.convolution(%v3301, %v3302)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3304 = stablehlo.convert %v3303 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3305 = stablehlo.transpose %v3304, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3306 = stablehlo.reshape %v163 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3307 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3308 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3309 = stablehlo.reduce(%v3306 init: %v3307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3310 = stablehlo.broadcast_in_dim %v3309, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3311 = stablehlo.divide %v3310, %v3308 : tensor<64x64x56x56xf32>
    %v3312 = stablehlo.subtract %v3306, %v3311 : tensor<64x64x56x56xf32>
    %v3313 = stablehlo.multiply %v3312, %v3312 : tensor<64x64x56x56xf32>
    %v3314 = stablehlo.reduce(%v3313 init: %v3307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3315 = stablehlo.broadcast_in_dim %v3314, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3316 = stablehlo.divide %v3315, %v3308 : tensor<64x64x56x56xf32>
    %v3317 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3318 = stablehlo.add %v3316, %v3317 : tensor<64x64x56x56xf32>
    %v3319 = stablehlo.rsqrt %v3318 : tensor<64x64x56x56xf32>
    %v3320 = stablehlo.multiply %v3312, %v3319 : tensor<64x64x56x56xf32>
    %v3321 = stablehlo.reshape %v3257 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3322 = stablehlo.multiply %v3321, %v3320 : tensor<64x64x56x56xf32>
    %v3323 = stablehlo.reduce(%v3322 init: %v3307) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3324 = stablehlo.reshape %v3257 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3325 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3326 = stablehlo.reduce(%v3324 init: %v3325) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3327 = stablehlo.reshape %v185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3328 = stablehlo.reshape %v3246 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3329 = stablehlo.transpose %v3327, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3330 = stablehlo.transpose %v3328, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3331 = stablehlo.convert %v3329 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3332 = stablehlo.convert %v3330 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3333 = stablehlo.convolution(%v3331, %v3332)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3334 = stablehlo.convert %v3333 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3335 = stablehlo.transpose %v3334, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3336 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3337 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3338 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3339 = stablehlo.reduce(%v3336 init: %v3337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3340 = stablehlo.broadcast_in_dim %v3339, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3341 = stablehlo.divide %v3340, %v3338 : tensor<64x64x56x56xf32>
    %v3342 = stablehlo.subtract %v3336, %v3341 : tensor<64x64x56x56xf32>
    %v3343 = stablehlo.multiply %v3342, %v3342 : tensor<64x64x56x56xf32>
    %v3344 = stablehlo.reduce(%v3343 init: %v3337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3345 = stablehlo.broadcast_in_dim %v3344, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3346 = stablehlo.divide %v3345, %v3338 : tensor<64x64x56x56xf32>
    %v3347 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3348 = stablehlo.add %v3346, %v3347 : tensor<64x64x56x56xf32>
    %v3349 = stablehlo.rsqrt %v3348 : tensor<64x64x56x56xf32>
    %v3350 = stablehlo.multiply %v3342, %v3349 : tensor<64x64x56x56xf32>
    %v3351 = stablehlo.reshape %v3216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3352 = stablehlo.multiply %v3351, %v3350 : tensor<64x64x56x56xf32>
    %v3353 = stablehlo.reduce(%v3352 init: %v3337) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3354 = stablehlo.reshape %v3216 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3355 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3356 = stablehlo.reduce(%v3354 init: %v3355) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3357 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3358 = stablehlo.compare GT, %v153, %v3357 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3359 = stablehlo.select %v3358, %v3296, %v3357 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3360 = stablehlo.reshape %v132 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3361 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3362 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3363 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3364 = stablehlo.reduce(%v3360 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3365 = stablehlo.broadcast_in_dim %v3364, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3366 = stablehlo.divide %v3365, %v3362 : tensor<64x64x56x56xf32>
    %v3367 = stablehlo.subtract %v3360, %v3366 : tensor<64x64x56x56xf32>
    %v3368 = stablehlo.multiply %v3367, %v3367 : tensor<64x64x56x56xf32>
    %v3369 = stablehlo.reduce(%v3368 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3370 = stablehlo.broadcast_in_dim %v3369, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3371 = stablehlo.divide %v3370, %v3362 : tensor<64x64x56x56xf32>
    %v3372 = stablehlo.add %v3371, %v3363 : tensor<64x64x56x56xf32>
    %v3373 = stablehlo.rsqrt %v3372 : tensor<64x64x56x56xf32>
    %v3374 = stablehlo.multiply %v3367, %v3373 : tensor<64x64x56x56xf32>
    %v3375 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3376 = stablehlo.reshape %v3359 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3377 = stablehlo.multiply %v3375, %v3376 : tensor<64x64x56x56xf32>
    %v3378 = stablehlo.reduce(%v3377 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3379 = stablehlo.broadcast_in_dim %v3378, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3380 = stablehlo.multiply %v3374, %v3377 : tensor<64x64x56x56xf32>
    %v3381 = stablehlo.reduce(%v3380 init: %v3361) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3382 = stablehlo.broadcast_in_dim %v3381, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3383 = stablehlo.multiply %v3377, %v3362 : tensor<64x64x56x56xf32>
    %v3384 = stablehlo.subtract %v3383, %v3379 : tensor<64x64x56x56xf32>
    %v3385 = stablehlo.multiply %v3374, %v3382 : tensor<64x64x56x56xf32>
    %v3386 = stablehlo.subtract %v3384, %v3385 : tensor<64x64x56x56xf32>
    %v3387 = stablehlo.divide %v3373, %v3362 : tensor<64x64x56x56xf32>
    %v3388 = stablehlo.multiply %v3387, %v3386 : tensor<64x64x56x56xf32>
    %v3389 = stablehlo.reshape %v3388 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3390 = stablehlo.reshape %v3389 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3391 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3392 = stablehlo.transpose %v3391, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3393 = stablehlo.convert %v3390 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3394 = stablehlo.convert %v3392 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3395 = stablehlo.convolution(%v3393, %v3394)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3396 = stablehlo.convert %v3395 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3397 = stablehlo.reshape %v3396 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3398 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3399 = stablehlo.compare GT, %v122, %v3398 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3400 = stablehlo.select %v3399, %v3397, %v3398 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3401 = stablehlo.reshape %v102 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3403 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3404 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3405 = stablehlo.reduce(%v3401 init: %v3402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3406 = stablehlo.broadcast_in_dim %v3405, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3407 = stablehlo.divide %v3406, %v3403 : tensor<64x64x56x56xf32>
    %v3408 = stablehlo.subtract %v3401, %v3407 : tensor<64x64x56x56xf32>
    %v3409 = stablehlo.multiply %v3408, %v3408 : tensor<64x64x56x56xf32>
    %v3410 = stablehlo.reduce(%v3409 init: %v3402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3411 = stablehlo.broadcast_in_dim %v3410, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3412 = stablehlo.divide %v3411, %v3403 : tensor<64x64x56x56xf32>
    %v3413 = stablehlo.add %v3412, %v3404 : tensor<64x64x56x56xf32>
    %v3414 = stablehlo.rsqrt %v3413 : tensor<64x64x56x56xf32>
    %v3415 = stablehlo.multiply %v3408, %v3414 : tensor<64x64x56x56xf32>
    %v3416 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3417 = stablehlo.reshape %v3400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3418 = stablehlo.multiply %v3416, %v3417 : tensor<64x64x56x56xf32>
    %v3419 = stablehlo.reduce(%v3418 init: %v3402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3420 = stablehlo.broadcast_in_dim %v3419, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3421 = stablehlo.multiply %v3415, %v3418 : tensor<64x64x56x56xf32>
    %v3422 = stablehlo.reduce(%v3421 init: %v3402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3423 = stablehlo.broadcast_in_dim %v3422, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3424 = stablehlo.multiply %v3418, %v3403 : tensor<64x64x56x56xf32>
    %v3425 = stablehlo.subtract %v3424, %v3420 : tensor<64x64x56x56xf32>
    %v3426 = stablehlo.multiply %v3415, %v3423 : tensor<64x64x56x56xf32>
    %v3427 = stablehlo.subtract %v3425, %v3426 : tensor<64x64x56x56xf32>
    %v3428 = stablehlo.divide %v3414, %v3403 : tensor<64x64x56x56xf32>
    %v3429 = stablehlo.multiply %v3428, %v3427 : tensor<64x64x56x56xf32>
    %v3430 = stablehlo.reshape %v3429 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3431 = stablehlo.reshape %v3430 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3432 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3433 = stablehlo.transpose %v3432, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3434 = stablehlo.convert %v3431 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3435 = stablehlo.convert %v3433 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3436 = stablehlo.convolution(%v3434, %v3435)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3437 = stablehlo.convert %v3436 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3438 = stablehlo.reshape %v3437 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3439 = stablehlo.add %v3438, %v3359 : tensor<64x200704xf32>
    %v3440 = stablehlo.reshape %v94 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3441 = stablehlo.reshape %v3430 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3442 = stablehlo.transpose %v3440, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3443 = stablehlo.transpose %v3441, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3444 = stablehlo.convert %v3442 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3445 = stablehlo.convert %v3443 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3446 = stablehlo.convolution(%v3444, %v3445)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3447 = stablehlo.convert %v3446 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3448 = stablehlo.transpose %v3447, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3449 = stablehlo.reshape %v102 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3450 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3451 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3452 = stablehlo.reduce(%v3449 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3453 = stablehlo.broadcast_in_dim %v3452, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3454 = stablehlo.divide %v3453, %v3451 : tensor<64x64x56x56xf32>
    %v3455 = stablehlo.subtract %v3449, %v3454 : tensor<64x64x56x56xf32>
    %v3456 = stablehlo.multiply %v3455, %v3455 : tensor<64x64x56x56xf32>
    %v3457 = stablehlo.reduce(%v3456 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3458 = stablehlo.broadcast_in_dim %v3457, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3459 = stablehlo.divide %v3458, %v3451 : tensor<64x64x56x56xf32>
    %v3460 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3461 = stablehlo.add %v3459, %v3460 : tensor<64x64x56x56xf32>
    %v3462 = stablehlo.rsqrt %v3461 : tensor<64x64x56x56xf32>
    %v3463 = stablehlo.multiply %v3455, %v3462 : tensor<64x64x56x56xf32>
    %v3464 = stablehlo.reshape %v3400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3465 = stablehlo.multiply %v3464, %v3463 : tensor<64x64x56x56xf32>
    %v3466 = stablehlo.reduce(%v3465 init: %v3450) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3467 = stablehlo.reshape %v3400 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3469 = stablehlo.reduce(%v3467 init: %v3468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3470 = stablehlo.reshape %v124 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3471 = stablehlo.reshape %v3389 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3472 = stablehlo.transpose %v3470, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3473 = stablehlo.transpose %v3471, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3474 = stablehlo.convert %v3472 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3475 = stablehlo.convert %v3473 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3476 = stablehlo.convolution(%v3474, %v3475)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3477 = stablehlo.convert %v3476 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3478 = stablehlo.transpose %v3477, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3479 = stablehlo.reshape %v132 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3480 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3481 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3482 = stablehlo.reduce(%v3479 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3482, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3484 = stablehlo.divide %v3483, %v3481 : tensor<64x64x56x56xf32>
    %v3485 = stablehlo.subtract %v3479, %v3484 : tensor<64x64x56x56xf32>
    %v3486 = stablehlo.multiply %v3485, %v3485 : tensor<64x64x56x56xf32>
    %v3487 = stablehlo.reduce(%v3486 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3488 = stablehlo.broadcast_in_dim %v3487, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3489 = stablehlo.divide %v3488, %v3481 : tensor<64x64x56x56xf32>
    %v3490 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3491 = stablehlo.add %v3489, %v3490 : tensor<64x64x56x56xf32>
    %v3492 = stablehlo.rsqrt %v3491 : tensor<64x64x56x56xf32>
    %v3493 = stablehlo.multiply %v3485, %v3492 : tensor<64x64x56x56xf32>
    %v3494 = stablehlo.reshape %v3359 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3495 = stablehlo.multiply %v3494, %v3493 : tensor<64x64x56x56xf32>
    %v3496 = stablehlo.reduce(%v3495 init: %v3480) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3497 = stablehlo.reshape %v3359 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3498 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3499 = stablehlo.reduce(%v3497 init: %v3498) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3500 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3501 = stablehlo.compare GT, %v92, %v3500 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3502 = stablehlo.select %v3501, %v3439, %v3500 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3503 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3504 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3505 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3506 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3507 = stablehlo.reduce(%v3503 init: %v3504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3507, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3509 = stablehlo.divide %v3508, %v3505 : tensor<64x64x56x56xf32>
    %v3510 = stablehlo.subtract %v3503, %v3509 : tensor<64x64x56x56xf32>
    %v3511 = stablehlo.multiply %v3510, %v3510 : tensor<64x64x56x56xf32>
    %v3512 = stablehlo.reduce(%v3511 init: %v3504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3513 = stablehlo.broadcast_in_dim %v3512, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3514 = stablehlo.divide %v3513, %v3505 : tensor<64x64x56x56xf32>
    %v3515 = stablehlo.add %v3514, %v3506 : tensor<64x64x56x56xf32>
    %v3516 = stablehlo.rsqrt %v3515 : tensor<64x64x56x56xf32>
    %v3517 = stablehlo.multiply %v3510, %v3516 : tensor<64x64x56x56xf32>
    %v3518 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3519 = stablehlo.reshape %v3502 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3520 = stablehlo.multiply %v3518, %v3519 : tensor<64x64x56x56xf32>
    %v3521 = stablehlo.reduce(%v3520 init: %v3504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3522 = stablehlo.broadcast_in_dim %v3521, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3523 = stablehlo.multiply %v3517, %v3520 : tensor<64x64x56x56xf32>
    %v3524 = stablehlo.reduce(%v3523 init: %v3504) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3525 = stablehlo.broadcast_in_dim %v3524, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3526 = stablehlo.multiply %v3520, %v3505 : tensor<64x64x56x56xf32>
    %v3527 = stablehlo.subtract %v3526, %v3522 : tensor<64x64x56x56xf32>
    %v3528 = stablehlo.multiply %v3517, %v3525 : tensor<64x64x56x56xf32>
    %v3529 = stablehlo.subtract %v3527, %v3528 : tensor<64x64x56x56xf32>
    %v3530 = stablehlo.divide %v3516, %v3505 : tensor<64x64x56x56xf32>
    %v3531 = stablehlo.multiply %v3530, %v3529 : tensor<64x64x56x56xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3533 = stablehlo.reshape %v3532 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3534 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3535 = stablehlo.transpose %v3534, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3536 = stablehlo.convert %v3533 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3537 = stablehlo.convert %v3535 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3538 = stablehlo.convolution(%v3536, %v3537)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3539 = stablehlo.convert %v3538 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3540 = stablehlo.reshape %v3539 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3541 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v3542 = stablehlo.compare GT, %v61, %v3541 : (tensor<64x200704xf32>, tensor<64x200704xf32>) -> tensor<64x200704xi1>
    %v3543 = stablehlo.select %v3542, %v3540, %v3541 : tensor<64x200704xi1>, tensor<64x200704xf32>
    %v3544 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3545 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3546 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3547 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3548 = stablehlo.reduce(%v3544 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3549 = stablehlo.broadcast_in_dim %v3548, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3550 = stablehlo.divide %v3549, %v3546 : tensor<64x64x56x56xf32>
    %v3551 = stablehlo.subtract %v3544, %v3550 : tensor<64x64x56x56xf32>
    %v3552 = stablehlo.multiply %v3551, %v3551 : tensor<64x64x56x56xf32>
    %v3553 = stablehlo.reduce(%v3552 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3554 = stablehlo.broadcast_in_dim %v3553, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3555 = stablehlo.divide %v3554, %v3546 : tensor<64x64x56x56xf32>
    %v3556 = stablehlo.add %v3555, %v3547 : tensor<64x64x56x56xf32>
    %v3557 = stablehlo.rsqrt %v3556 : tensor<64x64x56x56xf32>
    %v3558 = stablehlo.multiply %v3551, %v3557 : tensor<64x64x56x56xf32>
    %v3559 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3560 = stablehlo.reshape %v3543 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3561 = stablehlo.multiply %v3559, %v3560 : tensor<64x64x56x56xf32>
    %v3562 = stablehlo.reduce(%v3561 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3563 = stablehlo.broadcast_in_dim %v3562, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3564 = stablehlo.multiply %v3558, %v3561 : tensor<64x64x56x56xf32>
    %v3565 = stablehlo.reduce(%v3564 init: %v3545) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3566 = stablehlo.broadcast_in_dim %v3565, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3567 = stablehlo.multiply %v3561, %v3546 : tensor<64x64x56x56xf32>
    %v3568 = stablehlo.subtract %v3567, %v3563 : tensor<64x64x56x56xf32>
    %v3569 = stablehlo.multiply %v3558, %v3566 : tensor<64x64x56x56xf32>
    %v3570 = stablehlo.subtract %v3568, %v3569 : tensor<64x64x56x56xf32>
    %v3571 = stablehlo.divide %v3557, %v3546 : tensor<64x64x56x56xf32>
    %v3572 = stablehlo.multiply %v3571, %v3570 : tensor<64x64x56x56xf32>
    %v3573 = stablehlo.reshape %v3572 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3574 = stablehlo.reshape %v3573 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3575 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v3576 = stablehlo.transpose %v3575, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3577 = stablehlo.convert %v3574 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3578 = stablehlo.convert %v3576 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v3579 = stablehlo.convolution(%v3577, %v3578)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v3580 = stablehlo.convert %v3579 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v3581 = stablehlo.reshape %v3580 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v3582 = stablehlo.add %v3581, %v3502 : tensor<64x200704xf32>
    %v3583 = stablehlo.reshape %v33 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3584 = stablehlo.reshape %v3573 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3585 = stablehlo.transpose %v3583, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3586 = stablehlo.transpose %v3584, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3587 = stablehlo.convert %v3585 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3588 = stablehlo.convert %v3586 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3589 = stablehlo.convolution(%v3587, %v3588)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3590 = stablehlo.convert %v3589 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3591 = stablehlo.transpose %v3590, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3592 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3593 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3594 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3595 = stablehlo.reduce(%v3592 init: %v3593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3596 = stablehlo.broadcast_in_dim %v3595, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3597 = stablehlo.divide %v3596, %v3594 : tensor<64x64x56x56xf32>
    %v3598 = stablehlo.subtract %v3592, %v3597 : tensor<64x64x56x56xf32>
    %v3599 = stablehlo.multiply %v3598, %v3598 : tensor<64x64x56x56xf32>
    %v3600 = stablehlo.reduce(%v3599 init: %v3593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3601 = stablehlo.broadcast_in_dim %v3600, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3602 = stablehlo.divide %v3601, %v3594 : tensor<64x64x56x56xf32>
    %v3603 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3604 = stablehlo.add %v3602, %v3603 : tensor<64x64x56x56xf32>
    %v3605 = stablehlo.rsqrt %v3604 : tensor<64x64x56x56xf32>
    %v3606 = stablehlo.multiply %v3598, %v3605 : tensor<64x64x56x56xf32>
    %v3607 = stablehlo.reshape %v3543 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3608 = stablehlo.multiply %v3607, %v3606 : tensor<64x64x56x56xf32>
    %v3609 = stablehlo.reduce(%v3608 init: %v3593) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3610 = stablehlo.reshape %v3543 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3612 = stablehlo.reduce(%v3610 init: %v3611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3613 = stablehlo.reshape %v63 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3614 = stablehlo.reshape %v3532 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3615 = stablehlo.transpose %v3613, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3616 = stablehlo.transpose %v3614, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v3617 = stablehlo.convert %v3615 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3618 = stablehlo.convert %v3616 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v3619 = stablehlo.convolution(%v3617, %v3618)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v3620 = stablehlo.convert %v3619 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v3621 = stablehlo.transpose %v3620, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v3622 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3623 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3624 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3625 = stablehlo.reduce(%v3622 init: %v3623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3626 = stablehlo.broadcast_in_dim %v3625, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3627 = stablehlo.divide %v3626, %v3624 : tensor<64x64x56x56xf32>
    %v3628 = stablehlo.subtract %v3622, %v3627 : tensor<64x64x56x56xf32>
    %v3629 = stablehlo.multiply %v3628, %v3628 : tensor<64x64x56x56xf32>
    %v3630 = stablehlo.reduce(%v3629 init: %v3623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3631 = stablehlo.broadcast_in_dim %v3630, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3632 = stablehlo.divide %v3631, %v3624 : tensor<64x64x56x56xf32>
    %v3633 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v3634 = stablehlo.add %v3632, %v3633 : tensor<64x64x56x56xf32>
    %v3635 = stablehlo.rsqrt %v3634 : tensor<64x64x56x56xf32>
    %v3636 = stablehlo.multiply %v3628, %v3635 : tensor<64x64x56x56xf32>
    %v3637 = stablehlo.reshape %v3502 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3638 = stablehlo.multiply %v3637, %v3636 : tensor<64x64x56x56xf32>
    %v3639 = stablehlo.reduce(%v3638 init: %v3623) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3640 = stablehlo.reshape %v3502 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3641 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3642 = stablehlo.reduce(%v3640 init: %v3641) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3643 = stablehlo.reshape %v29 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3644 = stablehlo.reshape %v3582 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3645 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3646 = "stablehlo.select_and_scatter"(%v3643, %v3644, %v3645) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x112x112xf32>
    %v3647 = stablehlo.reshape %v3646 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v3648 = stablehlo.constant dense<0.0> : tensor<64x802816xf32>
    %v3649 = stablehlo.compare GT, %v27, %v3648 : (tensor<64x802816xf32>, tensor<64x802816xf32>) -> tensor<64x802816xi1>
    %v3650 = stablehlo.select %v3649, %v3647, %v3648 : tensor<64x802816xi1>, tensor<64x802816xf32>
    %v3651 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3652 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3653 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v3654 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v3655 = stablehlo.reduce(%v3651 init: %v3652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3656 = stablehlo.broadcast_in_dim %v3655, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3657 = stablehlo.divide %v3656, %v3653 : tensor<64x64x112x112xf32>
    %v3658 = stablehlo.subtract %v3651, %v3657 : tensor<64x64x112x112xf32>
    %v3659 = stablehlo.multiply %v3658, %v3658 : tensor<64x64x112x112xf32>
    %v3660 = stablehlo.reduce(%v3659 init: %v3652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3661 = stablehlo.broadcast_in_dim %v3660, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3662 = stablehlo.divide %v3661, %v3653 : tensor<64x64x112x112xf32>
    %v3663 = stablehlo.add %v3662, %v3654 : tensor<64x64x112x112xf32>
    %v3664 = stablehlo.rsqrt %v3663 : tensor<64x64x112x112xf32>
    %v3665 = stablehlo.multiply %v3658, %v3664 : tensor<64x64x112x112xf32>
    %v3666 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3667 = stablehlo.reshape %v3650 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3668 = stablehlo.multiply %v3666, %v3667 : tensor<64x64x112x112xf32>
    %v3669 = stablehlo.reduce(%v3668 init: %v3652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3670 = stablehlo.broadcast_in_dim %v3669, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3671 = stablehlo.multiply %v3665, %v3668 : tensor<64x64x112x112xf32>
    %v3672 = stablehlo.reduce(%v3671 init: %v3652) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3673 = stablehlo.broadcast_in_dim %v3672, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3674 = stablehlo.multiply %v3668, %v3653 : tensor<64x64x112x112xf32>
    %v3675 = stablehlo.subtract %v3674, %v3670 : tensor<64x64x112x112xf32>
    %v3676 = stablehlo.multiply %v3665, %v3673 : tensor<64x64x112x112xf32>
    %v3677 = stablehlo.subtract %v3675, %v3676 : tensor<64x64x112x112xf32>
    %v3678 = stablehlo.divide %v3664, %v3653 : tensor<64x64x112x112xf32>
    %v3679 = stablehlo.multiply %v3678, %v3677 : tensor<64x64x112x112xf32>
    %v3680 = stablehlo.reshape %v3679 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v3681 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v3682 = stablehlo.reshape %v3680 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3683 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3684 = stablehlo.pad %v3682, %v3683, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x224x224xf32>
    %v3685 = stablehlo.transpose %v3681, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v3686 = stablehlo.transpose %v3684, dims = [1, 0, 2, 3] : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xf32>
    %v3687 = stablehlo.convert %v3685 : (tensor<3x64x224x224xf32>) -> tensor<3x64x224x224xbf16>
    %v3688 = stablehlo.convert %v3686 : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xbf16>
    %v3689 = stablehlo.convolution(%v3687, %v3688)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xbf16>, tensor<64x64x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v3690 = stablehlo.convert %v3689 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v3691 = stablehlo.transpose %v3690, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v3692 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3693 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3694 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v3695 = stablehlo.reduce(%v3692 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3696 = stablehlo.broadcast_in_dim %v3695, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3697 = stablehlo.divide %v3696, %v3694 : tensor<64x64x112x112xf32>
    %v3698 = stablehlo.subtract %v3692, %v3697 : tensor<64x64x112x112xf32>
    %v3699 = stablehlo.multiply %v3698, %v3698 : tensor<64x64x112x112xf32>
    %v3700 = stablehlo.reduce(%v3699 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3701 = stablehlo.broadcast_in_dim %v3700, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3702 = stablehlo.divide %v3701, %v3694 : tensor<64x64x112x112xf32>
    %v3703 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v3704 = stablehlo.add %v3702, %v3703 : tensor<64x64x112x112xf32>
    %v3705 = stablehlo.rsqrt %v3704 : tensor<64x64x112x112xf32>
    %v3706 = stablehlo.multiply %v3698, %v3705 : tensor<64x64x112x112xf32>
    %v3707 = stablehlo.reshape %v3650 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3708 = stablehlo.multiply %v3707, %v3706 : tensor<64x64x112x112xf32>
    %v3709 = stablehlo.reduce(%v3708 init: %v3693) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3710 = stablehlo.reshape %v3650 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3711 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3712 = stablehlo.reduce(%v3710 init: %v3711) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3713 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3714 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3715 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3716 = stablehlo.reduce(%v3713 init: %v3714) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3717 = stablehlo.divide %v3716, %v3715 : tensor<64xf32>
    %v3718 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v3719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3720 = stablehlo.constant dense<802816.0> : tensor<64x64x112x112xf32>
    %v3721 = stablehlo.reduce(%v3718 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3722 = stablehlo.broadcast_in_dim %v3721, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v3723 = stablehlo.divide %v3722, %v3720 : tensor<64x64x112x112xf32>
    %v3724 = stablehlo.subtract %v3718, %v3723 : tensor<64x64x112x112xf32>
    %v3725 = stablehlo.multiply %v3724, %v3724 : tensor<64x64x112x112xf32>
    %v3726 = stablehlo.reduce(%v3725 init: %v3719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v3727 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v3728 = stablehlo.divide %v3726, %v3727 : tensor<64xf32>
    %v3729 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3731 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3732 = stablehlo.reduce(%v3729 init: %v3730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3733 = stablehlo.divide %v3732, %v3731 : tensor<64xf32>
    %v3734 = stablehlo.reshape %v41 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3735 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3736 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3737 = stablehlo.reduce(%v3734 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3738 = stablehlo.broadcast_in_dim %v3737, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3739 = stablehlo.divide %v3738, %v3736 : tensor<64x64x56x56xf32>
    %v3740 = stablehlo.subtract %v3734, %v3739 : tensor<64x64x56x56xf32>
    %v3741 = stablehlo.multiply %v3740, %v3740 : tensor<64x64x56x56xf32>
    %v3742 = stablehlo.reduce(%v3741 init: %v3735) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3743 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3744 = stablehlo.divide %v3742, %v3743 : tensor<64xf32>
    %v3745 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3746 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3747 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3748 = stablehlo.reduce(%v3745 init: %v3746) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3749 = stablehlo.divide %v3748, %v3747 : tensor<64xf32>
    %v3750 = stablehlo.reshape %v71 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3751 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3752 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3753 = stablehlo.reduce(%v3750 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3754 = stablehlo.broadcast_in_dim %v3753, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3755 = stablehlo.divide %v3754, %v3752 : tensor<64x64x56x56xf32>
    %v3756 = stablehlo.subtract %v3750, %v3755 : tensor<64x64x56x56xf32>
    %v3757 = stablehlo.multiply %v3756, %v3756 : tensor<64x64x56x56xf32>
    %v3758 = stablehlo.reduce(%v3757 init: %v3751) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3759 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3760 = stablehlo.divide %v3758, %v3759 : tensor<64xf32>
    %v3761 = stablehlo.reshape %v102 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3763 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3764 = stablehlo.reduce(%v3761 init: %v3762) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3765 = stablehlo.divide %v3764, %v3763 : tensor<64xf32>
    %v3766 = stablehlo.reshape %v102 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3767 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3768 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3769 = stablehlo.reduce(%v3766 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3769, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3771 = stablehlo.divide %v3770, %v3768 : tensor<64x64x56x56xf32>
    %v3772 = stablehlo.subtract %v3766, %v3771 : tensor<64x64x56x56xf32>
    %v3773 = stablehlo.multiply %v3772, %v3772 : tensor<64x64x56x56xf32>
    %v3774 = stablehlo.reduce(%v3773 init: %v3767) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3775 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3776 = stablehlo.divide %v3774, %v3775 : tensor<64xf32>
    %v3777 = stablehlo.reshape %v132 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3778 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3779 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3780 = stablehlo.reduce(%v3777 init: %v3778) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3781 = stablehlo.divide %v3780, %v3779 : tensor<64xf32>
    %v3782 = stablehlo.reshape %v132 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3784 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3785 = stablehlo.reduce(%v3782 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3786 = stablehlo.broadcast_in_dim %v3785, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3787 = stablehlo.divide %v3786, %v3784 : tensor<64x64x56x56xf32>
    %v3788 = stablehlo.subtract %v3782, %v3787 : tensor<64x64x56x56xf32>
    %v3789 = stablehlo.multiply %v3788, %v3788 : tensor<64x64x56x56xf32>
    %v3790 = stablehlo.reduce(%v3789 init: %v3783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3791 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3792 = stablehlo.divide %v3790, %v3791 : tensor<64xf32>
    %v3793 = stablehlo.reshape %v163 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3794 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3795 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3796 = stablehlo.reduce(%v3793 init: %v3794) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3797 = stablehlo.divide %v3796, %v3795 : tensor<64xf32>
    %v3798 = stablehlo.reshape %v163 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3799 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3800 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3801 = stablehlo.reduce(%v3798 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3802 = stablehlo.broadcast_in_dim %v3801, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3803 = stablehlo.divide %v3802, %v3800 : tensor<64x64x56x56xf32>
    %v3804 = stablehlo.subtract %v3798, %v3803 : tensor<64x64x56x56xf32>
    %v3805 = stablehlo.multiply %v3804, %v3804 : tensor<64x64x56x56xf32>
    %v3806 = stablehlo.reduce(%v3805 init: %v3799) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3807 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3808 = stablehlo.divide %v3806, %v3807 : tensor<64xf32>
    %v3809 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3810 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3811 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3812 = stablehlo.reduce(%v3809 init: %v3810) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3813 = stablehlo.divide %v3812, %v3811 : tensor<64xf32>
    %v3814 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v3815 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3816 = stablehlo.constant dense<200704.0> : tensor<64x64x56x56xf32>
    %v3817 = stablehlo.reduce(%v3814 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3818 = stablehlo.broadcast_in_dim %v3817, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v3819 = stablehlo.divide %v3818, %v3816 : tensor<64x64x56x56xf32>
    %v3820 = stablehlo.subtract %v3814, %v3819 : tensor<64x64x56x56xf32>
    %v3821 = stablehlo.multiply %v3820, %v3820 : tensor<64x64x56x56xf32>
    %v3822 = stablehlo.reduce(%v3821 init: %v3815) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v3823 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v3824 = stablehlo.divide %v3822, %v3823 : tensor<64xf32>
    %v3825 = stablehlo.reshape %v224 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3826 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3827 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3828 = stablehlo.reduce(%v3825 init: %v3826) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3829 = stablehlo.divide %v3828, %v3827 : tensor<128xf32>
    %v3830 = stablehlo.reshape %v224 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3831 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3832 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3833 = stablehlo.reduce(%v3830 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3834 = stablehlo.broadcast_in_dim %v3833, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3835 = stablehlo.divide %v3834, %v3832 : tensor<64x128x28x28xf32>
    %v3836 = stablehlo.subtract %v3830, %v3835 : tensor<64x128x28x28xf32>
    %v3837 = stablehlo.multiply %v3836, %v3836 : tensor<64x128x28x28xf32>
    %v3838 = stablehlo.reduce(%v3837 init: %v3831) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3839 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3840 = stablehlo.divide %v3838, %v3839 : tensor<128xf32>
    %v3841 = stablehlo.reshape %v254 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3842 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3843 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3844 = stablehlo.reduce(%v3841 init: %v3842) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3845 = stablehlo.divide %v3844, %v3843 : tensor<128xf32>
    %v3846 = stablehlo.reshape %v254 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3847 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3848 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3849 = stablehlo.reduce(%v3846 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3850 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3851 = stablehlo.divide %v3850, %v3848 : tensor<64x128x28x28xf32>
    %v3852 = stablehlo.subtract %v3846, %v3851 : tensor<64x128x28x28xf32>
    %v3853 = stablehlo.multiply %v3852, %v3852 : tensor<64x128x28x28xf32>
    %v3854 = stablehlo.reduce(%v3853 init: %v3847) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3855 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3856 = stablehlo.divide %v3854, %v3855 : tensor<128xf32>
    %v3857 = stablehlo.reshape %v282 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3858 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3859 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3860 = stablehlo.reduce(%v3857 init: %v3858) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3861 = stablehlo.divide %v3860, %v3859 : tensor<128xf32>
    %v3862 = stablehlo.reshape %v282 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3864 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3865 = stablehlo.reduce(%v3862 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3866 = stablehlo.broadcast_in_dim %v3865, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3867 = stablehlo.divide %v3866, %v3864 : tensor<64x128x28x28xf32>
    %v3868 = stablehlo.subtract %v3862, %v3867 : tensor<64x128x28x28xf32>
    %v3869 = stablehlo.multiply %v3868, %v3868 : tensor<64x128x28x28xf32>
    %v3870 = stablehlo.reduce(%v3869 init: %v3863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3871 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3872 = stablehlo.divide %v3870, %v3871 : tensor<128xf32>
    %v3873 = stablehlo.reshape %v313 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3874 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3875 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3876 = stablehlo.reduce(%v3873 init: %v3874) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3877 = stablehlo.divide %v3876, %v3875 : tensor<128xf32>
    %v3878 = stablehlo.reshape %v313 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3879 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3880 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3881 = stablehlo.reduce(%v3878 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3882 = stablehlo.broadcast_in_dim %v3881, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3883 = stablehlo.divide %v3882, %v3880 : tensor<64x128x28x28xf32>
    %v3884 = stablehlo.subtract %v3878, %v3883 : tensor<64x128x28x28xf32>
    %v3885 = stablehlo.multiply %v3884, %v3884 : tensor<64x128x28x28xf32>
    %v3886 = stablehlo.reduce(%v3885 init: %v3879) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3887 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3888 = stablehlo.divide %v3886, %v3887 : tensor<128xf32>
    %v3889 = stablehlo.reshape %v343 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3890 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3891 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3892 = stablehlo.reduce(%v3889 init: %v3890) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3893 = stablehlo.divide %v3892, %v3891 : tensor<128xf32>
    %v3894 = stablehlo.reshape %v343 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3895 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3896 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3897 = stablehlo.reduce(%v3894 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3898 = stablehlo.broadcast_in_dim %v3897, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3899 = stablehlo.divide %v3898, %v3896 : tensor<64x128x28x28xf32>
    %v3900 = stablehlo.subtract %v3894, %v3899 : tensor<64x128x28x28xf32>
    %v3901 = stablehlo.multiply %v3900, %v3900 : tensor<64x128x28x28xf32>
    %v3902 = stablehlo.reduce(%v3901 init: %v3895) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3903 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3904 = stablehlo.divide %v3902, %v3903 : tensor<128xf32>
    %v3905 = stablehlo.reshape %v374 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3906 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3907 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3908 = stablehlo.reduce(%v3905 init: %v3906) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3909 = stablehlo.divide %v3908, %v3907 : tensor<128xf32>
    %v3910 = stablehlo.reshape %v374 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3911 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3912 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3913 = stablehlo.reduce(%v3910 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3914 = stablehlo.broadcast_in_dim %v3913, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3915 = stablehlo.divide %v3914, %v3912 : tensor<64x128x28x28xf32>
    %v3916 = stablehlo.subtract %v3910, %v3915 : tensor<64x128x28x28xf32>
    %v3917 = stablehlo.multiply %v3916, %v3916 : tensor<64x128x28x28xf32>
    %v3918 = stablehlo.reduce(%v3917 init: %v3911) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3919 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3920 = stablehlo.divide %v3918, %v3919 : tensor<128xf32>
    %v3921 = stablehlo.reshape %v404 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3922 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3923 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3924 = stablehlo.reduce(%v3921 init: %v3922) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3925 = stablehlo.divide %v3924, %v3923 : tensor<128xf32>
    %v3926 = stablehlo.reshape %v404 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3929 = stablehlo.reduce(%v3926 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3930 = stablehlo.broadcast_in_dim %v3929, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3931 = stablehlo.divide %v3930, %v3928 : tensor<64x128x28x28xf32>
    %v3932 = stablehlo.subtract %v3926, %v3931 : tensor<64x128x28x28xf32>
    %v3933 = stablehlo.multiply %v3932, %v3932 : tensor<64x128x28x28xf32>
    %v3934 = stablehlo.reduce(%v3933 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3935 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3936 = stablehlo.divide %v3934, %v3935 : tensor<128xf32>
    %v3937 = stablehlo.reshape %v435 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3938 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3939 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3940 = stablehlo.reduce(%v3937 init: %v3938) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3941 = stablehlo.divide %v3940, %v3939 : tensor<128xf32>
    %v3942 = stablehlo.reshape %v435 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3943 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3944 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3945 = stablehlo.reduce(%v3942 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3946 = stablehlo.broadcast_in_dim %v3945, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3947 = stablehlo.divide %v3946, %v3944 : tensor<64x128x28x28xf32>
    %v3948 = stablehlo.subtract %v3942, %v3947 : tensor<64x128x28x28xf32>
    %v3949 = stablehlo.multiply %v3948, %v3948 : tensor<64x128x28x28xf32>
    %v3950 = stablehlo.reduce(%v3949 init: %v3943) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3951 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3952 = stablehlo.divide %v3950, %v3951 : tensor<128xf32>
    %v3953 = stablehlo.reshape %v465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3954 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3955 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3956 = stablehlo.reduce(%v3953 init: %v3954) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3957 = stablehlo.divide %v3956, %v3955 : tensor<128xf32>
    %v3958 = stablehlo.reshape %v465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3959 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3960 = stablehlo.constant dense<50176.0> : tensor<64x128x28x28xf32>
    %v3961 = stablehlo.reduce(%v3958 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3962 = stablehlo.broadcast_in_dim %v3961, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3963 = stablehlo.divide %v3962, %v3960 : tensor<64x128x28x28xf32>
    %v3964 = stablehlo.subtract %v3958, %v3963 : tensor<64x128x28x28xf32>
    %v3965 = stablehlo.multiply %v3964, %v3964 : tensor<64x128x28x28xf32>
    %v3966 = stablehlo.reduce(%v3965 init: %v3959) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3967 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3968 = stablehlo.divide %v3966, %v3967 : tensor<128xf32>
    %v3969 = stablehlo.reshape %v496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3970 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3971 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3972 = stablehlo.reduce(%v3969 init: %v3970) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3973 = stablehlo.divide %v3972, %v3971 : tensor<256xf32>
    %v3974 = stablehlo.reshape %v496 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3975 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3976 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3977 = stablehlo.reduce(%v3974 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3978 = stablehlo.broadcast_in_dim %v3977, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3979 = stablehlo.divide %v3978, %v3976 : tensor<64x256x14x14xf32>
    %v3980 = stablehlo.subtract %v3974, %v3979 : tensor<64x256x14x14xf32>
    %v3981 = stablehlo.multiply %v3980, %v3980 : tensor<64x256x14x14xf32>
    %v3982 = stablehlo.reduce(%v3981 init: %v3975) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3983 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3984 = stablehlo.divide %v3982, %v3983 : tensor<256xf32>
    %v3985 = stablehlo.reshape %v526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3986 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3987 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3988 = stablehlo.reduce(%v3985 init: %v3986) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3989 = stablehlo.divide %v3988, %v3987 : tensor<256xf32>
    %v3990 = stablehlo.reshape %v526 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3991 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3992 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v3993 = stablehlo.reduce(%v3990 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3994 = stablehlo.broadcast_in_dim %v3993, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3995 = stablehlo.divide %v3994, %v3992 : tensor<64x256x14x14xf32>
    %v3996 = stablehlo.subtract %v3990, %v3995 : tensor<64x256x14x14xf32>
    %v3997 = stablehlo.multiply %v3996, %v3996 : tensor<64x256x14x14xf32>
    %v3998 = stablehlo.reduce(%v3997 init: %v3991) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3999 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4000 = stablehlo.divide %v3998, %v3999 : tensor<256xf32>
    %v4001 = stablehlo.reshape %v554 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4002 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4003 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4004 = stablehlo.reduce(%v4001 init: %v4002) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4005 = stablehlo.divide %v4004, %v4003 : tensor<256xf32>
    %v4006 = stablehlo.reshape %v554 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4007 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4008 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4009 = stablehlo.reduce(%v4006 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4010 = stablehlo.broadcast_in_dim %v4009, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4011 = stablehlo.divide %v4010, %v4008 : tensor<64x256x14x14xf32>
    %v4012 = stablehlo.subtract %v4006, %v4011 : tensor<64x256x14x14xf32>
    %v4013 = stablehlo.multiply %v4012, %v4012 : tensor<64x256x14x14xf32>
    %v4014 = stablehlo.reduce(%v4013 init: %v4007) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4015 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4016 = stablehlo.divide %v4014, %v4015 : tensor<256xf32>
    %v4017 = stablehlo.reshape %v585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4018 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4019 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4020 = stablehlo.reduce(%v4017 init: %v4018) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4021 = stablehlo.divide %v4020, %v4019 : tensor<256xf32>
    %v4022 = stablehlo.reshape %v585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4023 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4024 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4025 = stablehlo.reduce(%v4022 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4025, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4027 = stablehlo.divide %v4026, %v4024 : tensor<64x256x14x14xf32>
    %v4028 = stablehlo.subtract %v4022, %v4027 : tensor<64x256x14x14xf32>
    %v4029 = stablehlo.multiply %v4028, %v4028 : tensor<64x256x14x14xf32>
    %v4030 = stablehlo.reduce(%v4029 init: %v4023) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4031 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4032 = stablehlo.divide %v4030, %v4031 : tensor<256xf32>
    %v4033 = stablehlo.reshape %v615 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4034 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4035 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4036 = stablehlo.reduce(%v4033 init: %v4034) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4037 = stablehlo.divide %v4036, %v4035 : tensor<256xf32>
    %v4038 = stablehlo.reshape %v615 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4039 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4040 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4041 = stablehlo.reduce(%v4038 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4042 = stablehlo.broadcast_in_dim %v4041, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4043 = stablehlo.divide %v4042, %v4040 : tensor<64x256x14x14xf32>
    %v4044 = stablehlo.subtract %v4038, %v4043 : tensor<64x256x14x14xf32>
    %v4045 = stablehlo.multiply %v4044, %v4044 : tensor<64x256x14x14xf32>
    %v4046 = stablehlo.reduce(%v4045 init: %v4039) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4047 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4048 = stablehlo.divide %v4046, %v4047 : tensor<256xf32>
    %v4049 = stablehlo.reshape %v646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4050 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4051 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4052 = stablehlo.reduce(%v4049 init: %v4050) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4053 = stablehlo.divide %v4052, %v4051 : tensor<256xf32>
    %v4054 = stablehlo.reshape %v646 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4055 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4056 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4057 = stablehlo.reduce(%v4054 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4058 = stablehlo.broadcast_in_dim %v4057, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4059 = stablehlo.divide %v4058, %v4056 : tensor<64x256x14x14xf32>
    %v4060 = stablehlo.subtract %v4054, %v4059 : tensor<64x256x14x14xf32>
    %v4061 = stablehlo.multiply %v4060, %v4060 : tensor<64x256x14x14xf32>
    %v4062 = stablehlo.reduce(%v4061 init: %v4055) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4063 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4064 = stablehlo.divide %v4062, %v4063 : tensor<256xf32>
    %v4065 = stablehlo.reshape %v676 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4066 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4067 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4068 = stablehlo.reduce(%v4065 init: %v4066) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4069 = stablehlo.divide %v4068, %v4067 : tensor<256xf32>
    %v4070 = stablehlo.reshape %v676 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4072 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4073 = stablehlo.reduce(%v4070 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4074 = stablehlo.broadcast_in_dim %v4073, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4075 = stablehlo.divide %v4074, %v4072 : tensor<64x256x14x14xf32>
    %v4076 = stablehlo.subtract %v4070, %v4075 : tensor<64x256x14x14xf32>
    %v4077 = stablehlo.multiply %v4076, %v4076 : tensor<64x256x14x14xf32>
    %v4078 = stablehlo.reduce(%v4077 init: %v4071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4079 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4080 = stablehlo.divide %v4078, %v4079 : tensor<256xf32>
    %v4081 = stablehlo.reshape %v707 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4082 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4083 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4084 = stablehlo.reduce(%v4081 init: %v4082) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4085 = stablehlo.divide %v4084, %v4083 : tensor<256xf32>
    %v4086 = stablehlo.reshape %v707 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4087 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4088 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4089 = stablehlo.reduce(%v4086 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4090 = stablehlo.broadcast_in_dim %v4089, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4091 = stablehlo.divide %v4090, %v4088 : tensor<64x256x14x14xf32>
    %v4092 = stablehlo.subtract %v4086, %v4091 : tensor<64x256x14x14xf32>
    %v4093 = stablehlo.multiply %v4092, %v4092 : tensor<64x256x14x14xf32>
    %v4094 = stablehlo.reduce(%v4093 init: %v4087) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4095 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4096 = stablehlo.divide %v4094, %v4095 : tensor<256xf32>
    %v4097 = stablehlo.reshape %v737 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4098 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4099 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4100 = stablehlo.reduce(%v4097 init: %v4098) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4101 = stablehlo.divide %v4100, %v4099 : tensor<256xf32>
    %v4102 = stablehlo.reshape %v737 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4103 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4104 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4105 = stablehlo.reduce(%v4102 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4106 = stablehlo.broadcast_in_dim %v4105, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4107 = stablehlo.divide %v4106, %v4104 : tensor<64x256x14x14xf32>
    %v4108 = stablehlo.subtract %v4102, %v4107 : tensor<64x256x14x14xf32>
    %v4109 = stablehlo.multiply %v4108, %v4108 : tensor<64x256x14x14xf32>
    %v4110 = stablehlo.reduce(%v4109 init: %v4103) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4111 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4112 = stablehlo.divide %v4110, %v4111 : tensor<256xf32>
    %v4113 = stablehlo.reshape %v768 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4114 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4115 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4116 = stablehlo.reduce(%v4113 init: %v4114) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4117 = stablehlo.divide %v4116, %v4115 : tensor<256xf32>
    %v4118 = stablehlo.reshape %v768 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4119 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4120 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4121 = stablehlo.reduce(%v4118 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4122 = stablehlo.broadcast_in_dim %v4121, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4123 = stablehlo.divide %v4122, %v4120 : tensor<64x256x14x14xf32>
    %v4124 = stablehlo.subtract %v4118, %v4123 : tensor<64x256x14x14xf32>
    %v4125 = stablehlo.multiply %v4124, %v4124 : tensor<64x256x14x14xf32>
    %v4126 = stablehlo.reduce(%v4125 init: %v4119) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4127 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4128 = stablehlo.divide %v4126, %v4127 : tensor<256xf32>
    %v4129 = stablehlo.reshape %v798 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4131 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4132 = stablehlo.reduce(%v4129 init: %v4130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4133 = stablehlo.divide %v4132, %v4131 : tensor<256xf32>
    %v4134 = stablehlo.reshape %v798 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4135 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4136 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4137 = stablehlo.reduce(%v4134 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4138 = stablehlo.broadcast_in_dim %v4137, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4139 = stablehlo.divide %v4138, %v4136 : tensor<64x256x14x14xf32>
    %v4140 = stablehlo.subtract %v4134, %v4139 : tensor<64x256x14x14xf32>
    %v4141 = stablehlo.multiply %v4140, %v4140 : tensor<64x256x14x14xf32>
    %v4142 = stablehlo.reduce(%v4141 init: %v4135) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4143 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4144 = stablehlo.divide %v4142, %v4143 : tensor<256xf32>
    %v4145 = stablehlo.reshape %v829 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4146 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4147 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4148 = stablehlo.reduce(%v4145 init: %v4146) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4149 = stablehlo.divide %v4148, %v4147 : tensor<256xf32>
    %v4150 = stablehlo.reshape %v829 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4152 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4153 = stablehlo.reduce(%v4150 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4154 = stablehlo.broadcast_in_dim %v4153, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4155 = stablehlo.divide %v4154, %v4152 : tensor<64x256x14x14xf32>
    %v4156 = stablehlo.subtract %v4150, %v4155 : tensor<64x256x14x14xf32>
    %v4157 = stablehlo.multiply %v4156, %v4156 : tensor<64x256x14x14xf32>
    %v4158 = stablehlo.reduce(%v4157 init: %v4151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4159 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4160 = stablehlo.divide %v4158, %v4159 : tensor<256xf32>
    %v4161 = stablehlo.reshape %v859 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4162 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4163 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4164 = stablehlo.reduce(%v4161 init: %v4162) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4165 = stablehlo.divide %v4164, %v4163 : tensor<256xf32>
    %v4166 = stablehlo.reshape %v859 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v4167 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4168 = stablehlo.constant dense<12544.0> : tensor<64x256x14x14xf32>
    %v4169 = stablehlo.reduce(%v4166 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4170 = stablehlo.broadcast_in_dim %v4169, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v4171 = stablehlo.divide %v4170, %v4168 : tensor<64x256x14x14xf32>
    %v4172 = stablehlo.subtract %v4166, %v4171 : tensor<64x256x14x14xf32>
    %v4173 = stablehlo.multiply %v4172, %v4172 : tensor<64x256x14x14xf32>
    %v4174 = stablehlo.reduce(%v4173 init: %v4167) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v4175 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v4176 = stablehlo.divide %v4174, %v4175 : tensor<256xf32>
    %v4177 = stablehlo.reshape %v890 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4178 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4179 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4180 = stablehlo.reduce(%v4177 init: %v4178) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4181 = stablehlo.divide %v4180, %v4179 : tensor<512xf32>
    %v4182 = stablehlo.reshape %v890 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4183 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4184 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4185 = stablehlo.reduce(%v4182 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4186 = stablehlo.broadcast_in_dim %v4185, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4187 = stablehlo.divide %v4186, %v4184 : tensor<64x512x7x7xf32>
    %v4188 = stablehlo.subtract %v4182, %v4187 : tensor<64x512x7x7xf32>
    %v4189 = stablehlo.multiply %v4188, %v4188 : tensor<64x512x7x7xf32>
    %v4190 = stablehlo.reduce(%v4189 init: %v4183) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4191 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4192 = stablehlo.divide %v4190, %v4191 : tensor<512xf32>
    %v4193 = stablehlo.reshape %v920 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4194 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4195 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4196 = stablehlo.reduce(%v4193 init: %v4194) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4197 = stablehlo.divide %v4196, %v4195 : tensor<512xf32>
    %v4198 = stablehlo.reshape %v920 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4199 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4200 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4201 = stablehlo.reduce(%v4198 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4202 = stablehlo.broadcast_in_dim %v4201, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4203 = stablehlo.divide %v4202, %v4200 : tensor<64x512x7x7xf32>
    %v4204 = stablehlo.subtract %v4198, %v4203 : tensor<64x512x7x7xf32>
    %v4205 = stablehlo.multiply %v4204, %v4204 : tensor<64x512x7x7xf32>
    %v4206 = stablehlo.reduce(%v4205 init: %v4199) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4207 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4208 = stablehlo.divide %v4206, %v4207 : tensor<512xf32>
    %v4209 = stablehlo.reshape %v948 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4210 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4211 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4212 = stablehlo.reduce(%v4209 init: %v4210) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4213 = stablehlo.divide %v4212, %v4211 : tensor<512xf32>
    %v4214 = stablehlo.reshape %v948 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4215 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4216 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4217 = stablehlo.reduce(%v4214 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4218 = stablehlo.broadcast_in_dim %v4217, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4219 = stablehlo.divide %v4218, %v4216 : tensor<64x512x7x7xf32>
    %v4220 = stablehlo.subtract %v4214, %v4219 : tensor<64x512x7x7xf32>
    %v4221 = stablehlo.multiply %v4220, %v4220 : tensor<64x512x7x7xf32>
    %v4222 = stablehlo.reduce(%v4221 init: %v4215) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4223 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4224 = stablehlo.divide %v4222, %v4223 : tensor<512xf32>
    %v4225 = stablehlo.reshape %v979 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4226 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4227 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4228 = stablehlo.reduce(%v4225 init: %v4226) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4229 = stablehlo.divide %v4228, %v4227 : tensor<512xf32>
    %v4230 = stablehlo.reshape %v979 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4231 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4232 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4233 = stablehlo.reduce(%v4230 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4234 = stablehlo.broadcast_in_dim %v4233, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4235 = stablehlo.divide %v4234, %v4232 : tensor<64x512x7x7xf32>
    %v4236 = stablehlo.subtract %v4230, %v4235 : tensor<64x512x7x7xf32>
    %v4237 = stablehlo.multiply %v4236, %v4236 : tensor<64x512x7x7xf32>
    %v4238 = stablehlo.reduce(%v4237 init: %v4231) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4239 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4240 = stablehlo.divide %v4238, %v4239 : tensor<512xf32>
    %v4241 = stablehlo.reshape %v1009 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4243 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4244 = stablehlo.reduce(%v4241 init: %v4242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4245 = stablehlo.divide %v4244, %v4243 : tensor<512xf32>
    %v4246 = stablehlo.reshape %v1009 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4247 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4248 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4249 = stablehlo.reduce(%v4246 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4250 = stablehlo.broadcast_in_dim %v4249, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4251 = stablehlo.divide %v4250, %v4248 : tensor<64x512x7x7xf32>
    %v4252 = stablehlo.subtract %v4246, %v4251 : tensor<64x512x7x7xf32>
    %v4253 = stablehlo.multiply %v4252, %v4252 : tensor<64x512x7x7xf32>
    %v4254 = stablehlo.reduce(%v4253 init: %v4247) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4255 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4256 = stablehlo.divide %v4254, %v4255 : tensor<512xf32>
    %v4257 = stablehlo.reshape %v1040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4258 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4259 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4260 = stablehlo.reduce(%v4257 init: %v4258) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4261 = stablehlo.divide %v4260, %v4259 : tensor<512xf32>
    %v4262 = stablehlo.reshape %v1040 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4263 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4264 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4265 = stablehlo.reduce(%v4262 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4266 = stablehlo.broadcast_in_dim %v4265, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4267 = stablehlo.divide %v4266, %v4264 : tensor<64x512x7x7xf32>
    %v4268 = stablehlo.subtract %v4262, %v4267 : tensor<64x512x7x7xf32>
    %v4269 = stablehlo.multiply %v4268, %v4268 : tensor<64x512x7x7xf32>
    %v4270 = stablehlo.reduce(%v4269 init: %v4263) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4271 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4272 = stablehlo.divide %v4270, %v4271 : tensor<512xf32>
    %v4273 = stablehlo.reshape %v1070 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4274 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4275 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4276 = stablehlo.reduce(%v4273 init: %v4274) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4277 = stablehlo.divide %v4276, %v4275 : tensor<512xf32>
    %v4278 = stablehlo.reshape %v1070 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v4279 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4280 = stablehlo.constant dense<3136.0> : tensor<64x512x7x7xf32>
    %v4281 = stablehlo.reduce(%v4278 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4282 = stablehlo.broadcast_in_dim %v4281, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v4283 = stablehlo.divide %v4282, %v4280 : tensor<64x512x7x7xf32>
    %v4284 = stablehlo.subtract %v4278, %v4283 : tensor<64x512x7x7xf32>
    %v4285 = stablehlo.multiply %v4284, %v4284 : tensor<64x512x7x7xf32>
    %v4286 = stablehlo.reduce(%v4285 init: %v4279) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v4287 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v4288 = stablehlo.divide %v4286, %v4287 : tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v3691) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4289 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4290 = stablehlo.multiply %v4289, %sW : tensor<64x3x7x7xf32>
    %v4291 = stablehlo.add %v4290, %armeansW : tensor<64x3x7x7xf32>
    %v4292 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4293 = stablehlo.multiply %v4292, %sWv : tensor<64x3x7x7xf32>
    %v4294 = stablehlo.add %v4293, %v4291 : tensor<64x3x7x7xf32>
    %v4295 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4296 = stablehlo.multiply %v4295, %v4294 : tensor<64x3x7x7xf32>
    %v4297 = stablehlo.subtract %sW, %v4296 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v3709) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4298 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4299 = stablehlo.multiply %v4298, %sg : tensor<64xf32>
    %v4300 = stablehlo.add %v4299, %armeansg : tensor<64xf32>
    %v4301 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4302 = stablehlo.multiply %v4301, %sgv : tensor<64xf32>
    %v4303 = stablehlo.add %v4302, %v4300 : tensor<64xf32>
    %v4304 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4305 = stablehlo.multiply %v4304, %v4303 : tensor<64xf32>
    %v4306 = stablehlo.subtract %sg, %v4305 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v3712) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4307 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4308 = stablehlo.multiply %v4307, %sbt : tensor<64xf32>
    %v4309 = stablehlo.add %v4308, %armeansbt : tensor<64xf32>
    %v4310 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4311 = stablehlo.multiply %v4310, %sbtv : tensor<64xf32>
    %v4312 = stablehlo.add %v4311, %v4309 : tensor<64xf32>
    %v4313 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4314 = stablehlo.multiply %v4313, %v4312 : tensor<64xf32>
    %v4315 = stablehlo.subtract %sbt, %v4314 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v3591) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4316 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4317 = stablehlo.multiply %v4316, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4318 = stablehlo.add %v4317, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4319 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4320 = stablehlo.multiply %v4319, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4321 = stablehlo.add %v4320, %v4318 : tensor<64x64x3x3xf32>
    %v4322 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4323 = stablehlo.multiply %v4322, %v4321 : tensor<64x64x3x3xf32>
    %v4324 = stablehlo.subtract %s1b0W1, %v4323 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v3609) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4325 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4326 = stablehlo.multiply %v4325, %s1b0g1 : tensor<64xf32>
    %v4327 = stablehlo.add %v4326, %armeans1b0g1 : tensor<64xf32>
    %v4328 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4329 = stablehlo.multiply %v4328, %s1b0g1v : tensor<64xf32>
    %v4330 = stablehlo.add %v4329, %v4327 : tensor<64xf32>
    %v4331 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4332 = stablehlo.multiply %v4331, %v4330 : tensor<64xf32>
    %v4333 = stablehlo.subtract %s1b0g1, %v4332 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v3612) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4334 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4335 = stablehlo.multiply %v4334, %s1b0bt1 : tensor<64xf32>
    %v4336 = stablehlo.add %v4335, %armeans1b0bt1 : tensor<64xf32>
    %v4337 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4338 = stablehlo.multiply %v4337, %s1b0bt1v : tensor<64xf32>
    %v4339 = stablehlo.add %v4338, %v4336 : tensor<64xf32>
    %v4340 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4341 = stablehlo.multiply %v4340, %v4339 : tensor<64xf32>
    %v4342 = stablehlo.subtract %s1b0bt1, %v4341 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v3621) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4343 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4344 = stablehlo.multiply %v4343, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4345 = stablehlo.add %v4344, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4346 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4347 = stablehlo.multiply %v4346, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4348 = stablehlo.add %v4347, %v4345 : tensor<64x64x3x3xf32>
    %v4349 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4350 = stablehlo.multiply %v4349, %v4348 : tensor<64x64x3x3xf32>
    %v4351 = stablehlo.subtract %s1b0W2, %v4350 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v3639) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4352 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4353 = stablehlo.multiply %v4352, %s1b0g2 : tensor<64xf32>
    %v4354 = stablehlo.add %v4353, %armeans1b0g2 : tensor<64xf32>
    %v4355 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4356 = stablehlo.multiply %v4355, %s1b0g2v : tensor<64xf32>
    %v4357 = stablehlo.add %v4356, %v4354 : tensor<64xf32>
    %v4358 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.multiply %v4358, %v4357 : tensor<64xf32>
    %v4360 = stablehlo.subtract %s1b0g2, %v4359 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v3642) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4361 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.multiply %v4361, %s1b0bt2 : tensor<64xf32>
    %v4363 = stablehlo.add %v4362, %armeans1b0bt2 : tensor<64xf32>
    %v4364 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4365 = stablehlo.multiply %v4364, %s1b0bt2v : tensor<64xf32>
    %v4366 = stablehlo.add %v4365, %v4363 : tensor<64xf32>
    %v4367 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4368 = stablehlo.multiply %v4367, %v4366 : tensor<64xf32>
    %v4369 = stablehlo.subtract %s1b0bt2, %v4368 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v3448) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4370 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4371 = stablehlo.multiply %v4370, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4372 = stablehlo.add %v4371, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4373 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4374 = stablehlo.multiply %v4373, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4375 = stablehlo.add %v4374, %v4372 : tensor<64x64x3x3xf32>
    %v4376 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4377 = stablehlo.multiply %v4376, %v4375 : tensor<64x64x3x3xf32>
    %v4378 = stablehlo.subtract %s1b1W1, %v4377 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v3466) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4379 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4380 = stablehlo.multiply %v4379, %s1b1g1 : tensor<64xf32>
    %v4381 = stablehlo.add %v4380, %armeans1b1g1 : tensor<64xf32>
    %v4382 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4383 = stablehlo.multiply %v4382, %s1b1g1v : tensor<64xf32>
    %v4384 = stablehlo.add %v4383, %v4381 : tensor<64xf32>
    %v4385 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4386 = stablehlo.multiply %v4385, %v4384 : tensor<64xf32>
    %v4387 = stablehlo.subtract %s1b1g1, %v4386 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v3469) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4388 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4389 = stablehlo.multiply %v4388, %s1b1bt1 : tensor<64xf32>
    %v4390 = stablehlo.add %v4389, %armeans1b1bt1 : tensor<64xf32>
    %v4391 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4392 = stablehlo.multiply %v4391, %s1b1bt1v : tensor<64xf32>
    %v4393 = stablehlo.add %v4392, %v4390 : tensor<64xf32>
    %v4394 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4395 = stablehlo.multiply %v4394, %v4393 : tensor<64xf32>
    %v4396 = stablehlo.subtract %s1b1bt1, %v4395 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v3478) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4397 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4398 = stablehlo.multiply %v4397, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4399 = stablehlo.add %v4398, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4400 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4401 = stablehlo.multiply %v4400, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4402 = stablehlo.add %v4401, %v4399 : tensor<64x64x3x3xf32>
    %v4403 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4404 = stablehlo.multiply %v4403, %v4402 : tensor<64x64x3x3xf32>
    %v4405 = stablehlo.subtract %s1b1W2, %v4404 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v3496) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v4406 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4407 = stablehlo.multiply %v4406, %s1b1g2 : tensor<64xf32>
    %v4408 = stablehlo.add %v4407, %armeans1b1g2 : tensor<64xf32>
    %v4409 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4410 = stablehlo.multiply %v4409, %s1b1g2v : tensor<64xf32>
    %v4411 = stablehlo.add %v4410, %v4408 : tensor<64xf32>
    %v4412 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4413 = stablehlo.multiply %v4412, %v4411 : tensor<64xf32>
    %v4414 = stablehlo.subtract %s1b1g2, %v4413 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v3499) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v4415 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4416 = stablehlo.multiply %v4415, %s1b1bt2 : tensor<64xf32>
    %v4417 = stablehlo.add %v4416, %armeans1b1bt2 : tensor<64xf32>
    %v4418 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4419 = stablehlo.multiply %v4418, %s1b1bt2v : tensor<64xf32>
    %v4420 = stablehlo.add %v4419, %v4417 : tensor<64xf32>
    %v4421 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4422 = stablehlo.multiply %v4421, %v4420 : tensor<64xf32>
    %v4423 = stablehlo.subtract %s1b1bt2, %v4422 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v3305) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v4424 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4425 = stablehlo.multiply %v4424, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4426 = stablehlo.add %v4425, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4427 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4428 = stablehlo.multiply %v4427, %s1b2W1v : tensor<64x64x3x3xf32>
    %v4429 = stablehlo.add %v4428, %v4426 : tensor<64x64x3x3xf32>
    %v4430 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4431 = stablehlo.multiply %v4430, %v4429 : tensor<64x64x3x3xf32>
    %v4432 = stablehlo.subtract %s1b2W1, %v4431 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v3323) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v4433 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4434 = stablehlo.multiply %v4433, %s1b2g1 : tensor<64xf32>
    %v4435 = stablehlo.add %v4434, %armeans1b2g1 : tensor<64xf32>
    %v4436 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4437 = stablehlo.multiply %v4436, %s1b2g1v : tensor<64xf32>
    %v4438 = stablehlo.add %v4437, %v4435 : tensor<64xf32>
    %v4439 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4440 = stablehlo.multiply %v4439, %v4438 : tensor<64xf32>
    %v4441 = stablehlo.subtract %s1b2g1, %v4440 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v3326) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v4442 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4443 = stablehlo.multiply %v4442, %s1b2bt1 : tensor<64xf32>
    %v4444 = stablehlo.add %v4443, %armeans1b2bt1 : tensor<64xf32>
    %v4445 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4446 = stablehlo.multiply %v4445, %s1b2bt1v : tensor<64xf32>
    %v4447 = stablehlo.add %v4446, %v4444 : tensor<64xf32>
    %v4448 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4449 = stablehlo.multiply %v4448, %v4447 : tensor<64xf32>
    %v4450 = stablehlo.subtract %s1b2bt1, %v4449 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v3335) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v4451 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4452 = stablehlo.multiply %v4451, %s1b2W2 : tensor<64x64x3x3xf32>
    %v4453 = stablehlo.add %v4452, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v4454 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4455 = stablehlo.multiply %v4454, %s1b2W2v : tensor<64x64x3x3xf32>
    %v4456 = stablehlo.add %v4455, %v4453 : tensor<64x64x3x3xf32>
    %v4457 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4458 = stablehlo.multiply %v4457, %v4456 : tensor<64x64x3x3xf32>
    %v4459 = stablehlo.subtract %s1b2W2, %v4458 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v3353) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v4460 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4461 = stablehlo.multiply %v4460, %s1b2g2 : tensor<64xf32>
    %v4462 = stablehlo.add %v4461, %armeans1b2g2 : tensor<64xf32>
    %v4463 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4464 = stablehlo.multiply %v4463, %s1b2g2v : tensor<64xf32>
    %v4465 = stablehlo.add %v4464, %v4462 : tensor<64xf32>
    %v4466 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4467 = stablehlo.multiply %v4466, %v4465 : tensor<64xf32>
    %v4468 = stablehlo.subtract %s1b2g2, %v4467 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v3356) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v4469 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4470 = stablehlo.multiply %v4469, %s1b2bt2 : tensor<64xf32>
    %v4471 = stablehlo.add %v4470, %armeans1b2bt2 : tensor<64xf32>
    %v4472 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4473 = stablehlo.multiply %v4472, %s1b2bt2v : tensor<64xf32>
    %v4474 = stablehlo.add %v4473, %v4471 : tensor<64xf32>
    %v4475 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4476 = stablehlo.multiply %v4475, %v4474 : tensor<64xf32>
    %v4477 = stablehlo.subtract %s1b2bt2, %v4476 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v3130) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<4.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v4478 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4479 = stablehlo.multiply %v4478, %d2W1 : tensor<128x64x3x3xf32>
    %v4480 = stablehlo.add %v4479, %armeand2W1 : tensor<128x64x3x3xf32>
    %v4481 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4482 = stablehlo.multiply %v4481, %d2W1v : tensor<128x64x3x3xf32>
    %v4483 = stablehlo.add %v4482, %v4480 : tensor<128x64x3x3xf32>
    %v4484 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v4485 = stablehlo.multiply %v4484, %v4483 : tensor<128x64x3x3xf32>
    %v4486 = stablehlo.subtract %d2W1, %v4485 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v3148) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v4487 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4488 = stablehlo.multiply %v4487, %d2g1 : tensor<128xf32>
    %v4489 = stablehlo.add %v4488, %armeand2g1 : tensor<128xf32>
    %v4490 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4491 = stablehlo.multiply %v4490, %d2g1v : tensor<128xf32>
    %v4492 = stablehlo.add %v4491, %v4489 : tensor<128xf32>
    %v4493 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4494 = stablehlo.multiply %v4493, %v4492 : tensor<128xf32>
    %v4495 = stablehlo.subtract %d2g1, %v4494 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v3151) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v4496 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4497 = stablehlo.multiply %v4496, %d2bt1 : tensor<128xf32>
    %v4498 = stablehlo.add %v4497, %armeand2bt1 : tensor<128xf32>
    %v4499 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4500 = stablehlo.multiply %v4499, %d2bt1v : tensor<128xf32>
    %v4501 = stablehlo.add %v4500, %v4498 : tensor<128xf32>
    %v4502 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4503 = stablehlo.multiply %v4502, %v4501 : tensor<128xf32>
    %v4504 = stablehlo.subtract %d2bt1, %v4503 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v3160) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v4505 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4506 = stablehlo.multiply %v4505, %d2W2 : tensor<128x128x3x3xf32>
    %v4507 = stablehlo.add %v4506, %armeand2W2 : tensor<128x128x3x3xf32>
    %v4508 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4509 = stablehlo.multiply %v4508, %d2W2v : tensor<128x128x3x3xf32>
    %v4510 = stablehlo.add %v4509, %v4507 : tensor<128x128x3x3xf32>
    %v4511 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4512 = stablehlo.multiply %v4511, %v4510 : tensor<128x128x3x3xf32>
    %v4513 = stablehlo.subtract %d2W2, %v4512 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v3178) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v4514 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4515 = stablehlo.multiply %v4514, %d2g2 : tensor<128xf32>
    %v4516 = stablehlo.add %v4515, %armeand2g2 : tensor<128xf32>
    %v4517 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4518 = stablehlo.multiply %v4517, %d2g2v : tensor<128xf32>
    %v4519 = stablehlo.add %v4518, %v4516 : tensor<128xf32>
    %v4520 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4521 = stablehlo.multiply %v4520, %v4519 : tensor<128xf32>
    %v4522 = stablehlo.subtract %d2g2, %v4521 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v3181) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v4523 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4524 = stablehlo.multiply %v4523, %d2bt2 : tensor<128xf32>
    %v4525 = stablehlo.add %v4524, %armeand2bt2 : tensor<128xf32>
    %v4526 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4527 = stablehlo.multiply %v4526, %d2bt2v : tensor<128xf32>
    %v4528 = stablehlo.add %v4527, %v4525 : tensor<128xf32>
    %v4529 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4530 = stablehlo.multiply %v4529, %v4528 : tensor<128xf32>
    %v4531 = stablehlo.subtract %d2bt2, %v4530 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v3192) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<4.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v4532 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4533 = stablehlo.multiply %v4532, %d2Wp : tensor<128x64x1x1xf32>
    %v4534 = stablehlo.add %v4533, %armeand2Wp : tensor<128x64x1x1xf32>
    %v4535 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4536 = stablehlo.multiply %v4535, %d2Wpv : tensor<128x64x1x1xf32>
    %v4537 = stablehlo.add %v4536, %v4534 : tensor<128x64x1x1xf32>
    %v4538 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v4539 = stablehlo.multiply %v4538, %v4537 : tensor<128x64x1x1xf32>
    %v4540 = stablehlo.subtract %d2Wp, %v4539 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v3210) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v4541 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4542 = stablehlo.multiply %v4541, %d2gp : tensor<128xf32>
    %v4543 = stablehlo.add %v4542, %armeand2gp : tensor<128xf32>
    %v4544 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4545 = stablehlo.multiply %v4544, %d2gpv : tensor<128xf32>
    %v4546 = stablehlo.add %v4545, %v4543 : tensor<128xf32>
    %v4547 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4548 = stablehlo.multiply %v4547, %v4546 : tensor<128xf32>
    %v4549 = stablehlo.subtract %d2gp, %v4548 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v3213) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v4550 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4551 = stablehlo.multiply %v4550, %d2btp : tensor<128xf32>
    %v4552 = stablehlo.add %v4551, %armeand2btp : tensor<128xf32>
    %v4553 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4554 = stablehlo.multiply %v4553, %d2btpv : tensor<128xf32>
    %v4555 = stablehlo.add %v4554, %v4552 : tensor<128xf32>
    %v4556 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4557 = stablehlo.multiply %v4556, %v4555 : tensor<128xf32>
    %v4558 = stablehlo.subtract %d2btp, %v4557 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v2943) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v4559 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4560 = stablehlo.multiply %v4559, %s2b0W1 : tensor<128x128x3x3xf32>
    %v4561 = stablehlo.add %v4560, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v4562 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4563 = stablehlo.multiply %v4562, %s2b0W1v : tensor<128x128x3x3xf32>
    %v4564 = stablehlo.add %v4563, %v4561 : tensor<128x128x3x3xf32>
    %v4565 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4566 = stablehlo.multiply %v4565, %v4564 : tensor<128x128x3x3xf32>
    %v4567 = stablehlo.subtract %s2b0W1, %v4566 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v2961) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v4568 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4569 = stablehlo.multiply %v4568, %s2b0g1 : tensor<128xf32>
    %v4570 = stablehlo.add %v4569, %armeans2b0g1 : tensor<128xf32>
    %v4571 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4572 = stablehlo.multiply %v4571, %s2b0g1v : tensor<128xf32>
    %v4573 = stablehlo.add %v4572, %v4570 : tensor<128xf32>
    %v4574 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4575 = stablehlo.multiply %v4574, %v4573 : tensor<128xf32>
    %v4576 = stablehlo.subtract %s2b0g1, %v4575 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v2964) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v4577 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4578 = stablehlo.multiply %v4577, %s2b0bt1 : tensor<128xf32>
    %v4579 = stablehlo.add %v4578, %armeans2b0bt1 : tensor<128xf32>
    %v4580 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4581 = stablehlo.multiply %v4580, %s2b0bt1v : tensor<128xf32>
    %v4582 = stablehlo.add %v4581, %v4579 : tensor<128xf32>
    %v4583 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4584 = stablehlo.multiply %v4583, %v4582 : tensor<128xf32>
    %v4585 = stablehlo.subtract %s2b0bt1, %v4584 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v2973) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v4586 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4587 = stablehlo.multiply %v4586, %s2b0W2 : tensor<128x128x3x3xf32>
    %v4588 = stablehlo.add %v4587, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v4589 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4590 = stablehlo.multiply %v4589, %s2b0W2v : tensor<128x128x3x3xf32>
    %v4591 = stablehlo.add %v4590, %v4588 : tensor<128x128x3x3xf32>
    %v4592 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4593 = stablehlo.multiply %v4592, %v4591 : tensor<128x128x3x3xf32>
    %v4594 = stablehlo.subtract %s2b0W2, %v4593 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v2991) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v4595 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4596 = stablehlo.multiply %v4595, %s2b0g2 : tensor<128xf32>
    %v4597 = stablehlo.add %v4596, %armeans2b0g2 : tensor<128xf32>
    %v4598 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4599 = stablehlo.multiply %v4598, %s2b0g2v : tensor<128xf32>
    %v4600 = stablehlo.add %v4599, %v4597 : tensor<128xf32>
    %v4601 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4602 = stablehlo.multiply %v4601, %v4600 : tensor<128xf32>
    %v4603 = stablehlo.subtract %s2b0g2, %v4602 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v2994) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v4604 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4605 = stablehlo.multiply %v4604, %s2b0bt2 : tensor<128xf32>
    %v4606 = stablehlo.add %v4605, %armeans2b0bt2 : tensor<128xf32>
    %v4607 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4608 = stablehlo.multiply %v4607, %s2b0bt2v : tensor<128xf32>
    %v4609 = stablehlo.add %v4608, %v4606 : tensor<128xf32>
    %v4610 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4611 = stablehlo.multiply %v4610, %v4609 : tensor<128xf32>
    %v4612 = stablehlo.subtract %s2b0bt2, %v4611 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v2800) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v4613 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4614 = stablehlo.multiply %v4613, %s2b1W1 : tensor<128x128x3x3xf32>
    %v4615 = stablehlo.add %v4614, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v4616 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4617 = stablehlo.multiply %v4616, %s2b1W1v : tensor<128x128x3x3xf32>
    %v4618 = stablehlo.add %v4617, %v4615 : tensor<128x128x3x3xf32>
    %v4619 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4620 = stablehlo.multiply %v4619, %v4618 : tensor<128x128x3x3xf32>
    %v4621 = stablehlo.subtract %s2b1W1, %v4620 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v2818) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v4622 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4623 = stablehlo.multiply %v4622, %s2b1g1 : tensor<128xf32>
    %v4624 = stablehlo.add %v4623, %armeans2b1g1 : tensor<128xf32>
    %v4625 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4626 = stablehlo.multiply %v4625, %s2b1g1v : tensor<128xf32>
    %v4627 = stablehlo.add %v4626, %v4624 : tensor<128xf32>
    %v4628 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4629 = stablehlo.multiply %v4628, %v4627 : tensor<128xf32>
    %v4630 = stablehlo.subtract %s2b1g1, %v4629 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v2821) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v4631 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4632 = stablehlo.multiply %v4631, %s2b1bt1 : tensor<128xf32>
    %v4633 = stablehlo.add %v4632, %armeans2b1bt1 : tensor<128xf32>
    %v4634 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4635 = stablehlo.multiply %v4634, %s2b1bt1v : tensor<128xf32>
    %v4636 = stablehlo.add %v4635, %v4633 : tensor<128xf32>
    %v4637 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4638 = stablehlo.multiply %v4637, %v4636 : tensor<128xf32>
    %v4639 = stablehlo.subtract %s2b1bt1, %v4638 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v2830) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v4640 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4641 = stablehlo.multiply %v4640, %s2b1W2 : tensor<128x128x3x3xf32>
    %v4642 = stablehlo.add %v4641, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v4643 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4644 = stablehlo.multiply %v4643, %s2b1W2v : tensor<128x128x3x3xf32>
    %v4645 = stablehlo.add %v4644, %v4642 : tensor<128x128x3x3xf32>
    %v4646 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4647 = stablehlo.multiply %v4646, %v4645 : tensor<128x128x3x3xf32>
    %v4648 = stablehlo.subtract %s2b1W2, %v4647 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v2848) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v4649 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4650 = stablehlo.multiply %v4649, %s2b1g2 : tensor<128xf32>
    %v4651 = stablehlo.add %v4650, %armeans2b1g2 : tensor<128xf32>
    %v4652 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4653 = stablehlo.multiply %v4652, %s2b1g2v : tensor<128xf32>
    %v4654 = stablehlo.add %v4653, %v4651 : tensor<128xf32>
    %v4655 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4656 = stablehlo.multiply %v4655, %v4654 : tensor<128xf32>
    %v4657 = stablehlo.subtract %s2b1g2, %v4656 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v2851) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v4658 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4659 = stablehlo.multiply %v4658, %s2b1bt2 : tensor<128xf32>
    %v4660 = stablehlo.add %v4659, %armeans2b1bt2 : tensor<128xf32>
    %v4661 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4662 = stablehlo.multiply %v4661, %s2b1bt2v : tensor<128xf32>
    %v4663 = stablehlo.add %v4662, %v4660 : tensor<128xf32>
    %v4664 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4665 = stablehlo.multiply %v4664, %v4663 : tensor<128xf32>
    %v4666 = stablehlo.subtract %s2b1bt2, %v4665 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v2657) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v4667 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4668 = stablehlo.multiply %v4667, %s2b2W1 : tensor<128x128x3x3xf32>
    %v4669 = stablehlo.add %v4668, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v4670 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4671 = stablehlo.multiply %v4670, %s2b2W1v : tensor<128x128x3x3xf32>
    %v4672 = stablehlo.add %v4671, %v4669 : tensor<128x128x3x3xf32>
    %v4673 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4674 = stablehlo.multiply %v4673, %v4672 : tensor<128x128x3x3xf32>
    %v4675 = stablehlo.subtract %s2b2W1, %v4674 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v2675) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v4676 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4677 = stablehlo.multiply %v4676, %s2b2g1 : tensor<128xf32>
    %v4678 = stablehlo.add %v4677, %armeans2b2g1 : tensor<128xf32>
    %v4679 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4680 = stablehlo.multiply %v4679, %s2b2g1v : tensor<128xf32>
    %v4681 = stablehlo.add %v4680, %v4678 : tensor<128xf32>
    %v4682 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4683 = stablehlo.multiply %v4682, %v4681 : tensor<128xf32>
    %v4684 = stablehlo.subtract %s2b2g1, %v4683 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v2678) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v4685 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4686 = stablehlo.multiply %v4685, %s2b2bt1 : tensor<128xf32>
    %v4687 = stablehlo.add %v4686, %armeans2b2bt1 : tensor<128xf32>
    %v4688 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4689 = stablehlo.multiply %v4688, %s2b2bt1v : tensor<128xf32>
    %v4690 = stablehlo.add %v4689, %v4687 : tensor<128xf32>
    %v4691 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4692 = stablehlo.multiply %v4691, %v4690 : tensor<128xf32>
    %v4693 = stablehlo.subtract %s2b2bt1, %v4692 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v2687) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v4694 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4695 = stablehlo.multiply %v4694, %s2b2W2 : tensor<128x128x3x3xf32>
    %v4696 = stablehlo.add %v4695, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v4697 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4698 = stablehlo.multiply %v4697, %s2b2W2v : tensor<128x128x3x3xf32>
    %v4699 = stablehlo.add %v4698, %v4696 : tensor<128x128x3x3xf32>
    %v4700 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v4701 = stablehlo.multiply %v4700, %v4699 : tensor<128x128x3x3xf32>
    %v4702 = stablehlo.subtract %s2b2W2, %v4701 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v2705) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v4703 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4704 = stablehlo.multiply %v4703, %s2b2g2 : tensor<128xf32>
    %v4705 = stablehlo.add %v4704, %armeans2b2g2 : tensor<128xf32>
    %v4706 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4707 = stablehlo.multiply %v4706, %s2b2g2v : tensor<128xf32>
    %v4708 = stablehlo.add %v4707, %v4705 : tensor<128xf32>
    %v4709 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4710 = stablehlo.multiply %v4709, %v4708 : tensor<128xf32>
    %v4711 = stablehlo.subtract %s2b2g2, %v4710 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v2708) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v4712 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4713 = stablehlo.multiply %v4712, %s2b2bt2 : tensor<128xf32>
    %v4714 = stablehlo.add %v4713, %armeans2b2bt2 : tensor<128xf32>
    %v4715 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4716 = stablehlo.multiply %v4715, %s2b2bt2v : tensor<128xf32>
    %v4717 = stablehlo.add %v4716, %v4714 : tensor<128xf32>
    %v4718 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v4719 = stablehlo.multiply %v4718, %v4717 : tensor<128xf32>
    %v4720 = stablehlo.subtract %s2b2bt2, %v4719 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v2482) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<4.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v4721 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4722 = stablehlo.multiply %v4721, %d3W1 : tensor<256x128x3x3xf32>
    %v4723 = stablehlo.add %v4722, %armeand3W1 : tensor<256x128x3x3xf32>
    %v4724 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4725 = stablehlo.multiply %v4724, %d3W1v : tensor<256x128x3x3xf32>
    %v4726 = stablehlo.add %v4725, %v4723 : tensor<256x128x3x3xf32>
    %v4727 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v4728 = stablehlo.multiply %v4727, %v4726 : tensor<256x128x3x3xf32>
    %v4729 = stablehlo.subtract %d3W1, %v4728 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v2500) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v4730 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4731 = stablehlo.multiply %v4730, %d3g1 : tensor<256xf32>
    %v4732 = stablehlo.add %v4731, %armeand3g1 : tensor<256xf32>
    %v4733 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4734 = stablehlo.multiply %v4733, %d3g1v : tensor<256xf32>
    %v4735 = stablehlo.add %v4734, %v4732 : tensor<256xf32>
    %v4736 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4737 = stablehlo.multiply %v4736, %v4735 : tensor<256xf32>
    %v4738 = stablehlo.subtract %d3g1, %v4737 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v2503) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v4739 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4740 = stablehlo.multiply %v4739, %d3bt1 : tensor<256xf32>
    %v4741 = stablehlo.add %v4740, %armeand3bt1 : tensor<256xf32>
    %v4742 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4743 = stablehlo.multiply %v4742, %d3bt1v : tensor<256xf32>
    %v4744 = stablehlo.add %v4743, %v4741 : tensor<256xf32>
    %v4745 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4746 = stablehlo.multiply %v4745, %v4744 : tensor<256xf32>
    %v4747 = stablehlo.subtract %d3bt1, %v4746 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v2512) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v4748 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4749 = stablehlo.multiply %v4748, %d3W2 : tensor<256x256x3x3xf32>
    %v4750 = stablehlo.add %v4749, %armeand3W2 : tensor<256x256x3x3xf32>
    %v4751 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4752 = stablehlo.multiply %v4751, %d3W2v : tensor<256x256x3x3xf32>
    %v4753 = stablehlo.add %v4752, %v4750 : tensor<256x256x3x3xf32>
    %v4754 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4755 = stablehlo.multiply %v4754, %v4753 : tensor<256x256x3x3xf32>
    %v4756 = stablehlo.subtract %d3W2, %v4755 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v2530) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v4757 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4758 = stablehlo.multiply %v4757, %d3g2 : tensor<256xf32>
    %v4759 = stablehlo.add %v4758, %armeand3g2 : tensor<256xf32>
    %v4760 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4761 = stablehlo.multiply %v4760, %d3g2v : tensor<256xf32>
    %v4762 = stablehlo.add %v4761, %v4759 : tensor<256xf32>
    %v4763 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4764 = stablehlo.multiply %v4763, %v4762 : tensor<256xf32>
    %v4765 = stablehlo.subtract %d3g2, %v4764 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v2533) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v4766 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4767 = stablehlo.multiply %v4766, %d3bt2 : tensor<256xf32>
    %v4768 = stablehlo.add %v4767, %armeand3bt2 : tensor<256xf32>
    %v4769 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4770 = stablehlo.multiply %v4769, %d3bt2v : tensor<256xf32>
    %v4771 = stablehlo.add %v4770, %v4768 : tensor<256xf32>
    %v4772 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4773 = stablehlo.multiply %v4772, %v4771 : tensor<256xf32>
    %v4774 = stablehlo.subtract %d3bt2, %v4773 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v2544) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<4.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v4775 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4776 = stablehlo.multiply %v4775, %d3Wp : tensor<256x128x1x1xf32>
    %v4777 = stablehlo.add %v4776, %armeand3Wp : tensor<256x128x1x1xf32>
    %v4778 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4779 = stablehlo.multiply %v4778, %d3Wpv : tensor<256x128x1x1xf32>
    %v4780 = stablehlo.add %v4779, %v4777 : tensor<256x128x1x1xf32>
    %v4781 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v4782 = stablehlo.multiply %v4781, %v4780 : tensor<256x128x1x1xf32>
    %v4783 = stablehlo.subtract %d3Wp, %v4782 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v2562) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v4784 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4785 = stablehlo.multiply %v4784, %d3gp : tensor<256xf32>
    %v4786 = stablehlo.add %v4785, %armeand3gp : tensor<256xf32>
    %v4787 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4788 = stablehlo.multiply %v4787, %d3gpv : tensor<256xf32>
    %v4789 = stablehlo.add %v4788, %v4786 : tensor<256xf32>
    %v4790 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4791 = stablehlo.multiply %v4790, %v4789 : tensor<256xf32>
    %v4792 = stablehlo.subtract %d3gp, %v4791 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v2565) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v4793 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4794 = stablehlo.multiply %v4793, %d3btp : tensor<256xf32>
    %v4795 = stablehlo.add %v4794, %armeand3btp : tensor<256xf32>
    %v4796 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4797 = stablehlo.multiply %v4796, %d3btpv : tensor<256xf32>
    %v4798 = stablehlo.add %v4797, %v4795 : tensor<256xf32>
    %v4799 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4800 = stablehlo.multiply %v4799, %v4798 : tensor<256xf32>
    %v4801 = stablehlo.subtract %d3btp, %v4800 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v2295) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v4802 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4803 = stablehlo.multiply %v4802, %s3b0W1 : tensor<256x256x3x3xf32>
    %v4804 = stablehlo.add %v4803, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v4805 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4806 = stablehlo.multiply %v4805, %s3b0W1v : tensor<256x256x3x3xf32>
    %v4807 = stablehlo.add %v4806, %v4804 : tensor<256x256x3x3xf32>
    %v4808 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4809 = stablehlo.multiply %v4808, %v4807 : tensor<256x256x3x3xf32>
    %v4810 = stablehlo.subtract %s3b0W1, %v4809 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v2313) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v4811 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4812 = stablehlo.multiply %v4811, %s3b0g1 : tensor<256xf32>
    %v4813 = stablehlo.add %v4812, %armeans3b0g1 : tensor<256xf32>
    %v4814 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4815 = stablehlo.multiply %v4814, %s3b0g1v : tensor<256xf32>
    %v4816 = stablehlo.add %v4815, %v4813 : tensor<256xf32>
    %v4817 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4818 = stablehlo.multiply %v4817, %v4816 : tensor<256xf32>
    %v4819 = stablehlo.subtract %s3b0g1, %v4818 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v2316) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v4820 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4821 = stablehlo.multiply %v4820, %s3b0bt1 : tensor<256xf32>
    %v4822 = stablehlo.add %v4821, %armeans3b0bt1 : tensor<256xf32>
    %v4823 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4824 = stablehlo.multiply %v4823, %s3b0bt1v : tensor<256xf32>
    %v4825 = stablehlo.add %v4824, %v4822 : tensor<256xf32>
    %v4826 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4827 = stablehlo.multiply %v4826, %v4825 : tensor<256xf32>
    %v4828 = stablehlo.subtract %s3b0bt1, %v4827 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v2325) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v4829 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4830 = stablehlo.multiply %v4829, %s3b0W2 : tensor<256x256x3x3xf32>
    %v4831 = stablehlo.add %v4830, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v4832 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4833 = stablehlo.multiply %v4832, %s3b0W2v : tensor<256x256x3x3xf32>
    %v4834 = stablehlo.add %v4833, %v4831 : tensor<256x256x3x3xf32>
    %v4835 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4836 = stablehlo.multiply %v4835, %v4834 : tensor<256x256x3x3xf32>
    %v4837 = stablehlo.subtract %s3b0W2, %v4836 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v2343) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v4838 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4839 = stablehlo.multiply %v4838, %s3b0g2 : tensor<256xf32>
    %v4840 = stablehlo.add %v4839, %armeans3b0g2 : tensor<256xf32>
    %v4841 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4842 = stablehlo.multiply %v4841, %s3b0g2v : tensor<256xf32>
    %v4843 = stablehlo.add %v4842, %v4840 : tensor<256xf32>
    %v4844 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4845 = stablehlo.multiply %v4844, %v4843 : tensor<256xf32>
    %v4846 = stablehlo.subtract %s3b0g2, %v4845 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v2346) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v4847 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4848 = stablehlo.multiply %v4847, %s3b0bt2 : tensor<256xf32>
    %v4849 = stablehlo.add %v4848, %armeans3b0bt2 : tensor<256xf32>
    %v4850 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4851 = stablehlo.multiply %v4850, %s3b0bt2v : tensor<256xf32>
    %v4852 = stablehlo.add %v4851, %v4849 : tensor<256xf32>
    %v4853 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4854 = stablehlo.multiply %v4853, %v4852 : tensor<256xf32>
    %v4855 = stablehlo.subtract %s3b0bt2, %v4854 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2152) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v4856 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4857 = stablehlo.multiply %v4856, %s3b1W1 : tensor<256x256x3x3xf32>
    %v4858 = stablehlo.add %v4857, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v4859 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4860 = stablehlo.multiply %v4859, %s3b1W1v : tensor<256x256x3x3xf32>
    %v4861 = stablehlo.add %v4860, %v4858 : tensor<256x256x3x3xf32>
    %v4862 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4863 = stablehlo.multiply %v4862, %v4861 : tensor<256x256x3x3xf32>
    %v4864 = stablehlo.subtract %s3b1W1, %v4863 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2170) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v4865 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4866 = stablehlo.multiply %v4865, %s3b1g1 : tensor<256xf32>
    %v4867 = stablehlo.add %v4866, %armeans3b1g1 : tensor<256xf32>
    %v4868 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4869 = stablehlo.multiply %v4868, %s3b1g1v : tensor<256xf32>
    %v4870 = stablehlo.add %v4869, %v4867 : tensor<256xf32>
    %v4871 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4872 = stablehlo.multiply %v4871, %v4870 : tensor<256xf32>
    %v4873 = stablehlo.subtract %s3b1g1, %v4872 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2173) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v4874 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4875 = stablehlo.multiply %v4874, %s3b1bt1 : tensor<256xf32>
    %v4876 = stablehlo.add %v4875, %armeans3b1bt1 : tensor<256xf32>
    %v4877 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4878 = stablehlo.multiply %v4877, %s3b1bt1v : tensor<256xf32>
    %v4879 = stablehlo.add %v4878, %v4876 : tensor<256xf32>
    %v4880 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4881 = stablehlo.multiply %v4880, %v4879 : tensor<256xf32>
    %v4882 = stablehlo.subtract %s3b1bt1, %v4881 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2182) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v4883 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4884 = stablehlo.multiply %v4883, %s3b1W2 : tensor<256x256x3x3xf32>
    %v4885 = stablehlo.add %v4884, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v4886 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4887 = stablehlo.multiply %v4886, %s3b1W2v : tensor<256x256x3x3xf32>
    %v4888 = stablehlo.add %v4887, %v4885 : tensor<256x256x3x3xf32>
    %v4889 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4890 = stablehlo.multiply %v4889, %v4888 : tensor<256x256x3x3xf32>
    %v4891 = stablehlo.subtract %s3b1W2, %v4890 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2200) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v4892 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4893 = stablehlo.multiply %v4892, %s3b1g2 : tensor<256xf32>
    %v4894 = stablehlo.add %v4893, %armeans3b1g2 : tensor<256xf32>
    %v4895 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4896 = stablehlo.multiply %v4895, %s3b1g2v : tensor<256xf32>
    %v4897 = stablehlo.add %v4896, %v4894 : tensor<256xf32>
    %v4898 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4899 = stablehlo.multiply %v4898, %v4897 : tensor<256xf32>
    %v4900 = stablehlo.subtract %s3b1g2, %v4899 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2203) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v4901 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4902 = stablehlo.multiply %v4901, %s3b1bt2 : tensor<256xf32>
    %v4903 = stablehlo.add %v4902, %armeans3b1bt2 : tensor<256xf32>
    %v4904 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4905 = stablehlo.multiply %v4904, %s3b1bt2v : tensor<256xf32>
    %v4906 = stablehlo.add %v4905, %v4903 : tensor<256xf32>
    %v4907 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4908 = stablehlo.multiply %v4907, %v4906 : tensor<256xf32>
    %v4909 = stablehlo.subtract %s3b1bt2, %v4908 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v2009) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v4910 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4911 = stablehlo.multiply %v4910, %s3b2W1 : tensor<256x256x3x3xf32>
    %v4912 = stablehlo.add %v4911, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v4913 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4914 = stablehlo.multiply %v4913, %s3b2W1v : tensor<256x256x3x3xf32>
    %v4915 = stablehlo.add %v4914, %v4912 : tensor<256x256x3x3xf32>
    %v4916 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4917 = stablehlo.multiply %v4916, %v4915 : tensor<256x256x3x3xf32>
    %v4918 = stablehlo.subtract %s3b2W1, %v4917 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2027) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v4919 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4920 = stablehlo.multiply %v4919, %s3b2g1 : tensor<256xf32>
    %v4921 = stablehlo.add %v4920, %armeans3b2g1 : tensor<256xf32>
    %v4922 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4923 = stablehlo.multiply %v4922, %s3b2g1v : tensor<256xf32>
    %v4924 = stablehlo.add %v4923, %v4921 : tensor<256xf32>
    %v4925 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4926 = stablehlo.multiply %v4925, %v4924 : tensor<256xf32>
    %v4927 = stablehlo.subtract %s3b2g1, %v4926 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2030) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v4928 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4929 = stablehlo.multiply %v4928, %s3b2bt1 : tensor<256xf32>
    %v4930 = stablehlo.add %v4929, %armeans3b2bt1 : tensor<256xf32>
    %v4931 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4932 = stablehlo.multiply %v4931, %s3b2bt1v : tensor<256xf32>
    %v4933 = stablehlo.add %v4932, %v4930 : tensor<256xf32>
    %v4934 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4935 = stablehlo.multiply %v4934, %v4933 : tensor<256xf32>
    %v4936 = stablehlo.subtract %s3b2bt1, %v4935 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2039) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v4937 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4938 = stablehlo.multiply %v4937, %s3b2W2 : tensor<256x256x3x3xf32>
    %v4939 = stablehlo.add %v4938, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v4940 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4941 = stablehlo.multiply %v4940, %s3b2W2v : tensor<256x256x3x3xf32>
    %v4942 = stablehlo.add %v4941, %v4939 : tensor<256x256x3x3xf32>
    %v4943 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4944 = stablehlo.multiply %v4943, %v4942 : tensor<256x256x3x3xf32>
    %v4945 = stablehlo.subtract %s3b2W2, %v4944 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2057) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v4946 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4947 = stablehlo.multiply %v4946, %s3b2g2 : tensor<256xf32>
    %v4948 = stablehlo.add %v4947, %armeans3b2g2 : tensor<256xf32>
    %v4949 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4950 = stablehlo.multiply %v4949, %s3b2g2v : tensor<256xf32>
    %v4951 = stablehlo.add %v4950, %v4948 : tensor<256xf32>
    %v4952 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4953 = stablehlo.multiply %v4952, %v4951 : tensor<256xf32>
    %v4954 = stablehlo.subtract %s3b2g2, %v4953 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2060) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v4955 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4956 = stablehlo.multiply %v4955, %s3b2bt2 : tensor<256xf32>
    %v4957 = stablehlo.add %v4956, %armeans3b2bt2 : tensor<256xf32>
    %v4958 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4959 = stablehlo.multiply %v4958, %s3b2bt2v : tensor<256xf32>
    %v4960 = stablehlo.add %v4959, %v4957 : tensor<256xf32>
    %v4961 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4962 = stablehlo.multiply %v4961, %v4960 : tensor<256xf32>
    %v4963 = stablehlo.subtract %s3b2bt2, %v4962 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v1866) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v4964 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4965 = stablehlo.multiply %v4964, %s3b3W1 : tensor<256x256x3x3xf32>
    %v4966 = stablehlo.add %v4965, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v4967 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4968 = stablehlo.multiply %v4967, %s3b3W1v : tensor<256x256x3x3xf32>
    %v4969 = stablehlo.add %v4968, %v4966 : tensor<256x256x3x3xf32>
    %v4970 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4971 = stablehlo.multiply %v4970, %v4969 : tensor<256x256x3x3xf32>
    %v4972 = stablehlo.subtract %s3b3W1, %v4971 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v1884) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v4973 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4974 = stablehlo.multiply %v4973, %s3b3g1 : tensor<256xf32>
    %v4975 = stablehlo.add %v4974, %armeans3b3g1 : tensor<256xf32>
    %v4976 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4977 = stablehlo.multiply %v4976, %s3b3g1v : tensor<256xf32>
    %v4978 = stablehlo.add %v4977, %v4975 : tensor<256xf32>
    %v4979 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4980 = stablehlo.multiply %v4979, %v4978 : tensor<256xf32>
    %v4981 = stablehlo.subtract %s3b3g1, %v4980 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v1887) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v4982 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4983 = stablehlo.multiply %v4982, %s3b3bt1 : tensor<256xf32>
    %v4984 = stablehlo.add %v4983, %armeans3b3bt1 : tensor<256xf32>
    %v4985 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4986 = stablehlo.multiply %v4985, %s3b3bt1v : tensor<256xf32>
    %v4987 = stablehlo.add %v4986, %v4984 : tensor<256xf32>
    %v4988 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v4989 = stablehlo.multiply %v4988, %v4987 : tensor<256xf32>
    %v4990 = stablehlo.subtract %s3b3bt1, %v4989 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v1896) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v4991 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4992 = stablehlo.multiply %v4991, %s3b3W2 : tensor<256x256x3x3xf32>
    %v4993 = stablehlo.add %v4992, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v4994 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4995 = stablehlo.multiply %v4994, %s3b3W2v : tensor<256x256x3x3xf32>
    %v4996 = stablehlo.add %v4995, %v4993 : tensor<256x256x3x3xf32>
    %v4997 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v4998 = stablehlo.multiply %v4997, %v4996 : tensor<256x256x3x3xf32>
    %v4999 = stablehlo.subtract %s3b3W2, %v4998 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v1914) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v5000 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5001 = stablehlo.multiply %v5000, %s3b3g2 : tensor<256xf32>
    %v5002 = stablehlo.add %v5001, %armeans3b3g2 : tensor<256xf32>
    %v5003 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5004 = stablehlo.multiply %v5003, %s3b3g2v : tensor<256xf32>
    %v5005 = stablehlo.add %v5004, %v5002 : tensor<256xf32>
    %v5006 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5007 = stablehlo.multiply %v5006, %v5005 : tensor<256xf32>
    %v5008 = stablehlo.subtract %s3b3g2, %v5007 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v1917) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v5009 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5010 = stablehlo.multiply %v5009, %s3b3bt2 : tensor<256xf32>
    %v5011 = stablehlo.add %v5010, %armeans3b3bt2 : tensor<256xf32>
    %v5012 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5013 = stablehlo.multiply %v5012, %s3b3bt2v : tensor<256xf32>
    %v5014 = stablehlo.add %v5013, %v5011 : tensor<256xf32>
    %v5015 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5016 = stablehlo.multiply %v5015, %v5014 : tensor<256xf32>
    %v5017 = stablehlo.subtract %s3b3bt2, %v5016 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v1723) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v5018 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5019 = stablehlo.multiply %v5018, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5020 = stablehlo.add %v5019, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v5021 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5022 = stablehlo.multiply %v5021, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5023 = stablehlo.add %v5022, %v5020 : tensor<256x256x3x3xf32>
    %v5024 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5025 = stablehlo.multiply %v5024, %v5023 : tensor<256x256x3x3xf32>
    %v5026 = stablehlo.subtract %s3b4W1, %v5025 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v1741) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v5027 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5028 = stablehlo.multiply %v5027, %s3b4g1 : tensor<256xf32>
    %v5029 = stablehlo.add %v5028, %armeans3b4g1 : tensor<256xf32>
    %v5030 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5031 = stablehlo.multiply %v5030, %s3b4g1v : tensor<256xf32>
    %v5032 = stablehlo.add %v5031, %v5029 : tensor<256xf32>
    %v5033 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5034 = stablehlo.multiply %v5033, %v5032 : tensor<256xf32>
    %v5035 = stablehlo.subtract %s3b4g1, %v5034 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v1744) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v5036 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5037 = stablehlo.multiply %v5036, %s3b4bt1 : tensor<256xf32>
    %v5038 = stablehlo.add %v5037, %armeans3b4bt1 : tensor<256xf32>
    %v5039 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5040 = stablehlo.multiply %v5039, %s3b4bt1v : tensor<256xf32>
    %v5041 = stablehlo.add %v5040, %v5038 : tensor<256xf32>
    %v5042 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5043 = stablehlo.multiply %v5042, %v5041 : tensor<256xf32>
    %v5044 = stablehlo.subtract %s3b4bt1, %v5043 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v1753) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v5045 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5046 = stablehlo.multiply %v5045, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5047 = stablehlo.add %v5046, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v5048 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5049 = stablehlo.multiply %v5048, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5050 = stablehlo.add %v5049, %v5047 : tensor<256x256x3x3xf32>
    %v5051 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5052 = stablehlo.multiply %v5051, %v5050 : tensor<256x256x3x3xf32>
    %v5053 = stablehlo.subtract %s3b4W2, %v5052 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v1771) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v5054 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5055 = stablehlo.multiply %v5054, %s3b4g2 : tensor<256xf32>
    %v5056 = stablehlo.add %v5055, %armeans3b4g2 : tensor<256xf32>
    %v5057 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5058 = stablehlo.multiply %v5057, %s3b4g2v : tensor<256xf32>
    %v5059 = stablehlo.add %v5058, %v5056 : tensor<256xf32>
    %v5060 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5061 = stablehlo.multiply %v5060, %v5059 : tensor<256xf32>
    %v5062 = stablehlo.subtract %s3b4g2, %v5061 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v1774) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v5063 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5064 = stablehlo.multiply %v5063, %s3b4bt2 : tensor<256xf32>
    %v5065 = stablehlo.add %v5064, %armeans3b4bt2 : tensor<256xf32>
    %v5066 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5067 = stablehlo.multiply %v5066, %s3b4bt2v : tensor<256xf32>
    %v5068 = stablehlo.add %v5067, %v5065 : tensor<256xf32>
    %v5069 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5070 = stablehlo.multiply %v5069, %v5068 : tensor<256xf32>
    %v5071 = stablehlo.subtract %s3b4bt2, %v5070 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v1548) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<4.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v5072 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5073 = stablehlo.multiply %v5072, %d4W1 : tensor<512x256x3x3xf32>
    %v5074 = stablehlo.add %v5073, %armeand4W1 : tensor<512x256x3x3xf32>
    %v5075 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5076 = stablehlo.multiply %v5075, %d4W1v : tensor<512x256x3x3xf32>
    %v5077 = stablehlo.add %v5076, %v5074 : tensor<512x256x3x3xf32>
    %v5078 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5079 = stablehlo.multiply %v5078, %v5077 : tensor<512x256x3x3xf32>
    %v5080 = stablehlo.subtract %d4W1, %v5079 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v1566) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v5081 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5082 = stablehlo.multiply %v5081, %d4g1 : tensor<512xf32>
    %v5083 = stablehlo.add %v5082, %armeand4g1 : tensor<512xf32>
    %v5084 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5085 = stablehlo.multiply %v5084, %d4g1v : tensor<512xf32>
    %v5086 = stablehlo.add %v5085, %v5083 : tensor<512xf32>
    %v5087 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5088 = stablehlo.multiply %v5087, %v5086 : tensor<512xf32>
    %v5089 = stablehlo.subtract %d4g1, %v5088 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v1569) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v5090 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5091 = stablehlo.multiply %v5090, %d4bt1 : tensor<512xf32>
    %v5092 = stablehlo.add %v5091, %armeand4bt1 : tensor<512xf32>
    %v5093 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5094 = stablehlo.multiply %v5093, %d4bt1v : tensor<512xf32>
    %v5095 = stablehlo.add %v5094, %v5092 : tensor<512xf32>
    %v5096 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5097 = stablehlo.multiply %v5096, %v5095 : tensor<512xf32>
    %v5098 = stablehlo.subtract %d4bt1, %v5097 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v1578) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v5099 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5100 = stablehlo.multiply %v5099, %d4W2 : tensor<512x512x3x3xf32>
    %v5101 = stablehlo.add %v5100, %armeand4W2 : tensor<512x512x3x3xf32>
    %v5102 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5103 = stablehlo.multiply %v5102, %d4W2v : tensor<512x512x3x3xf32>
    %v5104 = stablehlo.add %v5103, %v5101 : tensor<512x512x3x3xf32>
    %v5105 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5106 = stablehlo.multiply %v5105, %v5104 : tensor<512x512x3x3xf32>
    %v5107 = stablehlo.subtract %d4W2, %v5106 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v1596) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v5108 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5109 = stablehlo.multiply %v5108, %d4g2 : tensor<512xf32>
    %v5110 = stablehlo.add %v5109, %armeand4g2 : tensor<512xf32>
    %v5111 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5112 = stablehlo.multiply %v5111, %d4g2v : tensor<512xf32>
    %v5113 = stablehlo.add %v5112, %v5110 : tensor<512xf32>
    %v5114 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5115 = stablehlo.multiply %v5114, %v5113 : tensor<512xf32>
    %v5116 = stablehlo.subtract %d4g2, %v5115 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v1599) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v5117 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5118 = stablehlo.multiply %v5117, %d4bt2 : tensor<512xf32>
    %v5119 = stablehlo.add %v5118, %armeand4bt2 : tensor<512xf32>
    %v5120 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5121 = stablehlo.multiply %v5120, %d4bt2v : tensor<512xf32>
    %v5122 = stablehlo.add %v5121, %v5119 : tensor<512xf32>
    %v5123 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5124 = stablehlo.multiply %v5123, %v5122 : tensor<512xf32>
    %v5125 = stablehlo.subtract %d4bt2, %v5124 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v1610) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v5126 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5127 = stablehlo.multiply %v5126, %d4Wp : tensor<512x256x1x1xf32>
    %v5128 = stablehlo.add %v5127, %armeand4Wp : tensor<512x256x1x1xf32>
    %v5129 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5130 = stablehlo.multiply %v5129, %d4Wpv : tensor<512x256x1x1xf32>
    %v5131 = stablehlo.add %v5130, %v5128 : tensor<512x256x1x1xf32>
    %v5132 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5133 = stablehlo.multiply %v5132, %v5131 : tensor<512x256x1x1xf32>
    %v5134 = stablehlo.subtract %d4Wp, %v5133 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v1628) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v5135 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5136 = stablehlo.multiply %v5135, %d4gp : tensor<512xf32>
    %v5137 = stablehlo.add %v5136, %armeand4gp : tensor<512xf32>
    %v5138 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5139 = stablehlo.multiply %v5138, %d4gpv : tensor<512xf32>
    %v5140 = stablehlo.add %v5139, %v5137 : tensor<512xf32>
    %v5141 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5142 = stablehlo.multiply %v5141, %v5140 : tensor<512xf32>
    %v5143 = stablehlo.subtract %d4gp, %v5142 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v1631) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v5144 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5145 = stablehlo.multiply %v5144, %d4btp : tensor<512xf32>
    %v5146 = stablehlo.add %v5145, %armeand4btp : tensor<512xf32>
    %v5147 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5148 = stablehlo.multiply %v5147, %d4btpv : tensor<512xf32>
    %v5149 = stablehlo.add %v5148, %v5146 : tensor<512xf32>
    %v5150 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5151 = stablehlo.multiply %v5150, %v5149 : tensor<512xf32>
    %v5152 = stablehlo.subtract %d4btp, %v5151 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1361) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v5153 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5154 = stablehlo.multiply %v5153, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5155 = stablehlo.add %v5154, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v5156 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5157 = stablehlo.multiply %v5156, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5158 = stablehlo.add %v5157, %v5155 : tensor<512x512x3x3xf32>
    %v5159 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5160 = stablehlo.multiply %v5159, %v5158 : tensor<512x512x3x3xf32>
    %v5161 = stablehlo.subtract %s4b0W1, %v5160 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v1379) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v5162 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5163 = stablehlo.multiply %v5162, %s4b0g1 : tensor<512xf32>
    %v5164 = stablehlo.add %v5163, %armeans4b0g1 : tensor<512xf32>
    %v5165 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5166 = stablehlo.multiply %v5165, %s4b0g1v : tensor<512xf32>
    %v5167 = stablehlo.add %v5166, %v5164 : tensor<512xf32>
    %v5168 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5169 = stablehlo.multiply %v5168, %v5167 : tensor<512xf32>
    %v5170 = stablehlo.subtract %s4b0g1, %v5169 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v1382) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v5171 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5172 = stablehlo.multiply %v5171, %s4b0bt1 : tensor<512xf32>
    %v5173 = stablehlo.add %v5172, %armeans4b0bt1 : tensor<512xf32>
    %v5174 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5175 = stablehlo.multiply %v5174, %s4b0bt1v : tensor<512xf32>
    %v5176 = stablehlo.add %v5175, %v5173 : tensor<512xf32>
    %v5177 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5178 = stablehlo.multiply %v5177, %v5176 : tensor<512xf32>
    %v5179 = stablehlo.subtract %s4b0bt1, %v5178 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v1391) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v5180 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5181 = stablehlo.multiply %v5180, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5182 = stablehlo.add %v5181, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v5183 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5184 = stablehlo.multiply %v5183, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5185 = stablehlo.add %v5184, %v5182 : tensor<512x512x3x3xf32>
    %v5186 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5187 = stablehlo.multiply %v5186, %v5185 : tensor<512x512x3x3xf32>
    %v5188 = stablehlo.subtract %s4b0W2, %v5187 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v1409) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v5189 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5190 = stablehlo.multiply %v5189, %s4b0g2 : tensor<512xf32>
    %v5191 = stablehlo.add %v5190, %armeans4b0g2 : tensor<512xf32>
    %v5192 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5193 = stablehlo.multiply %v5192, %s4b0g2v : tensor<512xf32>
    %v5194 = stablehlo.add %v5193, %v5191 : tensor<512xf32>
    %v5195 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5196 = stablehlo.multiply %v5195, %v5194 : tensor<512xf32>
    %v5197 = stablehlo.subtract %s4b0g2, %v5196 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v1412) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v5198 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5199 = stablehlo.multiply %v5198, %s4b0bt2 : tensor<512xf32>
    %v5200 = stablehlo.add %v5199, %armeans4b0bt2 : tensor<512xf32>
    %v5201 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5202 = stablehlo.multiply %v5201, %s4b0bt2v : tensor<512xf32>
    %v5203 = stablehlo.add %v5202, %v5200 : tensor<512xf32>
    %v5204 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5205 = stablehlo.multiply %v5204, %v5203 : tensor<512xf32>
    %v5206 = stablehlo.subtract %s4b0bt2, %v5205 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1218) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v5207 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5208 = stablehlo.multiply %v5207, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5209 = stablehlo.add %v5208, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v5210 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5211 = stablehlo.multiply %v5210, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5212 = stablehlo.add %v5211, %v5209 : tensor<512x512x3x3xf32>
    %v5213 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5214 = stablehlo.multiply %v5213, %v5212 : tensor<512x512x3x3xf32>
    %v5215 = stablehlo.subtract %s4b1W1, %v5214 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1236) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v5216 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5217 = stablehlo.multiply %v5216, %s4b1g1 : tensor<512xf32>
    %v5218 = stablehlo.add %v5217, %armeans4b1g1 : tensor<512xf32>
    %v5219 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5220 = stablehlo.multiply %v5219, %s4b1g1v : tensor<512xf32>
    %v5221 = stablehlo.add %v5220, %v5218 : tensor<512xf32>
    %v5222 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5223 = stablehlo.multiply %v5222, %v5221 : tensor<512xf32>
    %v5224 = stablehlo.subtract %s4b1g1, %v5223 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1239) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v5225 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5226 = stablehlo.multiply %v5225, %s4b1bt1 : tensor<512xf32>
    %v5227 = stablehlo.add %v5226, %armeans4b1bt1 : tensor<512xf32>
    %v5228 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5229 = stablehlo.multiply %v5228, %s4b1bt1v : tensor<512xf32>
    %v5230 = stablehlo.add %v5229, %v5227 : tensor<512xf32>
    %v5231 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5232 = stablehlo.multiply %v5231, %v5230 : tensor<512xf32>
    %v5233 = stablehlo.subtract %s4b1bt1, %v5232 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1248) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v5234 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5235 = stablehlo.multiply %v5234, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5236 = stablehlo.add %v5235, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v5237 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5238 = stablehlo.multiply %v5237, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5239 = stablehlo.add %v5238, %v5236 : tensor<512x512x3x3xf32>
    %v5240 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5241 = stablehlo.multiply %v5240, %v5239 : tensor<512x512x3x3xf32>
    %v5242 = stablehlo.subtract %s4b1W2, %v5241 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1266) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v5243 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5244 = stablehlo.multiply %v5243, %s4b1g2 : tensor<512xf32>
    %v5245 = stablehlo.add %v5244, %armeans4b1g2 : tensor<512xf32>
    %v5246 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5247 = stablehlo.multiply %v5246, %s4b1g2v : tensor<512xf32>
    %v5248 = stablehlo.add %v5247, %v5245 : tensor<512xf32>
    %v5249 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5250 = stablehlo.multiply %v5249, %v5248 : tensor<512xf32>
    %v5251 = stablehlo.subtract %s4b1g2, %v5250 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1269) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v5252 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5253 = stablehlo.multiply %v5252, %s4b1bt2 : tensor<512xf32>
    %v5254 = stablehlo.add %v5253, %armeans4b1bt2 : tensor<512xf32>
    %v5255 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5256 = stablehlo.multiply %v5255, %s4b1bt2v : tensor<512xf32>
    %v5257 = stablehlo.add %v5256, %v5254 : tensor<512xf32>
    %v5258 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5259 = stablehlo.multiply %v5258, %v5257 : tensor<512xf32>
    %v5260 = stablehlo.subtract %s4b1bt2, %v5259 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1120) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<512x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x1000xf32>
    %v5261 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5262 = stablehlo.multiply %v5261, %Wd : tensor<512x1000xf32>
    %v5263 = stablehlo.add %v5262, %armeanWd : tensor<512x1000xf32>
    %v5264 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5265 = stablehlo.multiply %v5264, %Wdv : tensor<512x1000xf32>
    %v5266 = stablehlo.add %v5265, %v5263 : tensor<512x1000xf32>
    %v5267 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5268 = stablehlo.multiply %v5267, %v5266 : tensor<512x1000xf32>
    %v5269 = stablehlo.subtract %Wd, %v5268 : tensor<512x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1122) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v5270 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5271 = stablehlo.multiply %v5270, %bd : tensor<1000xf32>
    %v5272 = stablehlo.add %v5271, %armeanbd : tensor<1000xf32>
    %v5273 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5274 = stablehlo.multiply %v5273, %bdv : tensor<1000xf32>
    %v5275 = stablehlo.add %v5274, %v5272 : tensor<1000xf32>
    %v5276 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5277 = stablehlo.multiply %v5276, %v5275 : tensor<1000xf32>
    %v5278 = stablehlo.subtract %bd, %v5277 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1108 : tensor<64x1000xf32>
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
    return %v4297, %v4306, %v4315, %v4324, %v4333, %v4342, %v4351, %v4360, %v4369, %v4378, %v4387, %v4396, %v4405, %v4414, %v4423, %v4432, %v4441, %v4450, %v4459, %v4468, %v4477, %v4486, %v4495, %v4504, %v4513, %v4522, %v4531, %v4540, %v4549, %v4558, %v4567, %v4576, %v4585, %v4594, %v4603, %v4612, %v4621, %v4630, %v4639, %v4648, %v4657, %v4666, %v4675, %v4684, %v4693, %v4702, %v4711, %v4720, %v4729, %v4738, %v4747, %v4756, %v4765, %v4774, %v4783, %v4792, %v4801, %v4810, %v4819, %v4828, %v4837, %v4846, %v4855, %v4864, %v4873, %v4882, %v4891, %v4900, %v4909, %v4918, %v4927, %v4936, %v4945, %v4954, %v4963, %v4972, %v4981, %v4990, %v4999, %v5008, %v5017, %v5026, %v5035, %v5044, %v5053, %v5062, %v5071, %v5080, %v5089, %v5098, %v5107, %v5116, %v5125, %v5134, %v5143, %v5152, %v5161, %v5170, %v5179, %v5188, %v5197, %v5206, %v5215, %v5224, %v5233, %v5242, %v5251, %v5260, %v5269, %v5278, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4294, %v4303, %v4312, %v4321, %v4330, %v4339, %v4348, %v4357, %v4366, %v4375, %v4384, %v4393, %v4402, %v4411, %v4420, %v4429, %v4438, %v4447, %v4456, %v4465, %v4474, %v4483, %v4492, %v4501, %v4510, %v4519, %v4528, %v4537, %v4546, %v4555, %v4564, %v4573, %v4582, %v4591, %v4600, %v4609, %v4618, %v4627, %v4636, %v4645, %v4654, %v4663, %v4672, %v4681, %v4690, %v4699, %v4708, %v4717, %v4726, %v4735, %v4744, %v4753, %v4762, %v4771, %v4780, %v4789, %v4798, %v4807, %v4816, %v4825, %v4834, %v4843, %v4852, %v4861, %v4870, %v4879, %v4888, %v4897, %v4906, %v4915, %v4924, %v4933, %v4942, %v4951, %v4960, %v4969, %v4978, %v4987, %v4996, %v5005, %v5014, %v5023, %v5032, %v5041, %v5050, %v5059, %v5068, %v5077, %v5086, %v5095, %v5104, %v5113, %v5122, %v5131, %v5140, %v5149, %v5158, %v5167, %v5176, %v5185, %v5194, %v5203, %v5212, %v5221, %v5230, %v5239, %v5248, %v5257, %v5266, %v5275, %loss, %bc1, %bc2, %v3717, %v3728, %v3733, %v3744, %v3749, %v3760, %v3765, %v3776, %v3781, %v3792, %v3797, %v3808, %v3813, %v3824, %v3829, %v3840, %v3845, %v3856, %v3861, %v3872, %v3877, %v3888, %v3893, %v3904, %v3909, %v3920, %v3925, %v3936, %v3941, %v3952, %v3957, %v3968, %v3973, %v3984, %v3989, %v4000, %v4005, %v4016, %v4021, %v4032, %v4037, %v4048, %v4053, %v4064, %v4069, %v4080, %v4085, %v4096, %v4101, %v4112, %v4117, %v4128, %v4133, %v4144, %v4149, %v4160, %v4165, %v4176, %v4181, %v4192, %v4197, %v4208, %v4213, %v4224, %v4229, %v4240, %v4245, %v4256, %v4261, %v4272, %v4277, %v4288 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}
