module @m {
  func.func @resnet34in_momdp64bf16_train_step(%x: tensor<64x150528xf32>, %sW: tensor<64x3x7x7xf32>, %sg: tensor<64xf32>, %sbt: tensor<64xf32>, %s1b0W1: tensor<64x64x3x3xf32>, %s1b0g1: tensor<64xf32>, %s1b0bt1: tensor<64xf32>, %s1b0W2: tensor<64x64x3x3xf32>, %s1b0g2: tensor<64xf32>, %s1b0bt2: tensor<64xf32>, %s1b1W1: tensor<64x64x3x3xf32>, %s1b1g1: tensor<64xf32>, %s1b1bt1: tensor<64xf32>, %s1b1W2: tensor<64x64x3x3xf32>, %s1b1g2: tensor<64xf32>, %s1b1bt2: tensor<64xf32>, %s1b2W1: tensor<64x64x3x3xf32>, %s1b2g1: tensor<64xf32>, %s1b2bt1: tensor<64xf32>, %s1b2W2: tensor<64x64x3x3xf32>, %s1b2g2: tensor<64xf32>, %s1b2bt2: tensor<64xf32>, %d2W1: tensor<128x64x3x3xf32>, %d2g1: tensor<128xf32>, %d2bt1: tensor<128xf32>, %d2W2: tensor<128x128x3x3xf32>, %d2g2: tensor<128xf32>, %d2bt2: tensor<128xf32>, %d2Wp: tensor<128x64x1x1xf32>, %d2gp: tensor<128xf32>, %d2btp: tensor<128xf32>, %s2b0W1: tensor<128x128x3x3xf32>, %s2b0g1: tensor<128xf32>, %s2b0bt1: tensor<128xf32>, %s2b0W2: tensor<128x128x3x3xf32>, %s2b0g2: tensor<128xf32>, %s2b0bt2: tensor<128xf32>, %s2b1W1: tensor<128x128x3x3xf32>, %s2b1g1: tensor<128xf32>, %s2b1bt1: tensor<128xf32>, %s2b1W2: tensor<128x128x3x3xf32>, %s2b1g2: tensor<128xf32>, %s2b1bt2: tensor<128xf32>, %s2b2W1: tensor<128x128x3x3xf32>, %s2b2g1: tensor<128xf32>, %s2b2bt1: tensor<128xf32>, %s2b2W2: tensor<128x128x3x3xf32>, %s2b2g2: tensor<128xf32>, %s2b2bt2: tensor<128xf32>, %d3W1: tensor<256x128x3x3xf32>, %d3g1: tensor<256xf32>, %d3bt1: tensor<256xf32>, %d3W2: tensor<256x256x3x3xf32>, %d3g2: tensor<256xf32>, %d3bt2: tensor<256xf32>, %d3Wp: tensor<256x128x1x1xf32>, %d3gp: tensor<256xf32>, %d3btp: tensor<256xf32>, %s3b0W1: tensor<256x256x3x3xf32>, %s3b0g1: tensor<256xf32>, %s3b0bt1: tensor<256xf32>, %s3b0W2: tensor<256x256x3x3xf32>, %s3b0g2: tensor<256xf32>, %s3b0bt2: tensor<256xf32>, %s3b1W1: tensor<256x256x3x3xf32>, %s3b1g1: tensor<256xf32>, %s3b1bt1: tensor<256xf32>, %s3b1W2: tensor<256x256x3x3xf32>, %s3b1g2: tensor<256xf32>, %s3b1bt2: tensor<256xf32>, %s3b2W1: tensor<256x256x3x3xf32>, %s3b2g1: tensor<256xf32>, %s3b2bt1: tensor<256xf32>, %s3b2W2: tensor<256x256x3x3xf32>, %s3b2g2: tensor<256xf32>, %s3b2bt2: tensor<256xf32>, %s3b3W1: tensor<256x256x3x3xf32>, %s3b3g1: tensor<256xf32>, %s3b3bt1: tensor<256xf32>, %s3b3W2: tensor<256x256x3x3xf32>, %s3b3g2: tensor<256xf32>, %s3b3bt2: tensor<256xf32>, %s3b4W1: tensor<256x256x3x3xf32>, %s3b4g1: tensor<256xf32>, %s3b4bt1: tensor<256xf32>, %s3b4W2: tensor<256x256x3x3xf32>, %s3b4g2: tensor<256xf32>, %s3b4bt2: tensor<256xf32>, %d4W1: tensor<512x256x3x3xf32>, %d4g1: tensor<512xf32>, %d4bt1: tensor<512xf32>, %d4W2: tensor<512x512x3x3xf32>, %d4g2: tensor<512xf32>, %d4bt2: tensor<512xf32>, %d4Wp: tensor<512x256x1x1xf32>, %d4gp: tensor<512xf32>, %d4btp: tensor<512xf32>, %s4b0W1: tensor<512x512x3x3xf32>, %s4b0g1: tensor<512xf32>, %s4b0bt1: tensor<512xf32>, %s4b0W2: tensor<512x512x3x3xf32>, %s4b0g2: tensor<512xf32>, %s4b0bt2: tensor<512xf32>, %s4b1W1: tensor<512x512x3x3xf32>, %s4b1g1: tensor<512xf32>, %s4b1bt1: tensor<512xf32>, %s4b1W2: tensor<512x512x3x3xf32>, %s4b1g2: tensor<512xf32>, %s4b1bt2: tensor<512xf32>, %Wd: tensor<512x1000xf32>, %bd: tensor<1000xf32>, %sWm: tensor<64x3x7x7xf32>, %sgm: tensor<64xf32>, %sbtm: tensor<64xf32>, %s1b0W1m: tensor<64x64x3x3xf32>, %s1b0g1m: tensor<64xf32>, %s1b0bt1m: tensor<64xf32>, %s1b0W2m: tensor<64x64x3x3xf32>, %s1b0g2m: tensor<64xf32>, %s1b0bt2m: tensor<64xf32>, %s1b1W1m: tensor<64x64x3x3xf32>, %s1b1g1m: tensor<64xf32>, %s1b1bt1m: tensor<64xf32>, %s1b1W2m: tensor<64x64x3x3xf32>, %s1b1g2m: tensor<64xf32>, %s1b1bt2m: tensor<64xf32>, %s1b2W1m: tensor<64x64x3x3xf32>, %s1b2g1m: tensor<64xf32>, %s1b2bt1m: tensor<64xf32>, %s1b2W2m: tensor<64x64x3x3xf32>, %s1b2g2m: tensor<64xf32>, %s1b2bt2m: tensor<64xf32>, %d2W1m: tensor<128x64x3x3xf32>, %d2g1m: tensor<128xf32>, %d2bt1m: tensor<128xf32>, %d2W2m: tensor<128x128x3x3xf32>, %d2g2m: tensor<128xf32>, %d2bt2m: tensor<128xf32>, %d2Wpm: tensor<128x64x1x1xf32>, %d2gpm: tensor<128xf32>, %d2btpm: tensor<128xf32>, %s2b0W1m: tensor<128x128x3x3xf32>, %s2b0g1m: tensor<128xf32>, %s2b0bt1m: tensor<128xf32>, %s2b0W2m: tensor<128x128x3x3xf32>, %s2b0g2m: tensor<128xf32>, %s2b0bt2m: tensor<128xf32>, %s2b1W1m: tensor<128x128x3x3xf32>, %s2b1g1m: tensor<128xf32>, %s2b1bt1m: tensor<128xf32>, %s2b1W2m: tensor<128x128x3x3xf32>, %s2b1g2m: tensor<128xf32>, %s2b1bt2m: tensor<128xf32>, %s2b2W1m: tensor<128x128x3x3xf32>, %s2b2g1m: tensor<128xf32>, %s2b2bt1m: tensor<128xf32>, %s2b2W2m: tensor<128x128x3x3xf32>, %s2b2g2m: tensor<128xf32>, %s2b2bt2m: tensor<128xf32>, %d3W1m: tensor<256x128x3x3xf32>, %d3g1m: tensor<256xf32>, %d3bt1m: tensor<256xf32>, %d3W2m: tensor<256x256x3x3xf32>, %d3g2m: tensor<256xf32>, %d3bt2m: tensor<256xf32>, %d3Wpm: tensor<256x128x1x1xf32>, %d3gpm: tensor<256xf32>, %d3btpm: tensor<256xf32>, %s3b0W1m: tensor<256x256x3x3xf32>, %s3b0g1m: tensor<256xf32>, %s3b0bt1m: tensor<256xf32>, %s3b0W2m: tensor<256x256x3x3xf32>, %s3b0g2m: tensor<256xf32>, %s3b0bt2m: tensor<256xf32>, %s3b1W1m: tensor<256x256x3x3xf32>, %s3b1g1m: tensor<256xf32>, %s3b1bt1m: tensor<256xf32>, %s3b1W2m: tensor<256x256x3x3xf32>, %s3b1g2m: tensor<256xf32>, %s3b1bt2m: tensor<256xf32>, %s3b2W1m: tensor<256x256x3x3xf32>, %s3b2g1m: tensor<256xf32>, %s3b2bt1m: tensor<256xf32>, %s3b2W2m: tensor<256x256x3x3xf32>, %s3b2g2m: tensor<256xf32>, %s3b2bt2m: tensor<256xf32>, %s3b3W1m: tensor<256x256x3x3xf32>, %s3b3g1m: tensor<256xf32>, %s3b3bt1m: tensor<256xf32>, %s3b3W2m: tensor<256x256x3x3xf32>, %s3b3g2m: tensor<256xf32>, %s3b3bt2m: tensor<256xf32>, %s3b4W1m: tensor<256x256x3x3xf32>, %s3b4g1m: tensor<256xf32>, %s3b4bt1m: tensor<256xf32>, %s3b4W2m: tensor<256x256x3x3xf32>, %s3b4g2m: tensor<256xf32>, %s3b4bt2m: tensor<256xf32>, %d4W1m: tensor<512x256x3x3xf32>, %d4g1m: tensor<512xf32>, %d4bt1m: tensor<512xf32>, %d4W2m: tensor<512x512x3x3xf32>, %d4g2m: tensor<512xf32>, %d4bt2m: tensor<512xf32>, %d4Wpm: tensor<512x256x1x1xf32>, %d4gpm: tensor<512xf32>, %d4btpm: tensor<512xf32>, %s4b0W1m: tensor<512x512x3x3xf32>, %s4b0g1m: tensor<512xf32>, %s4b0bt1m: tensor<512xf32>, %s4b0W2m: tensor<512x512x3x3xf32>, %s4b0g2m: tensor<512xf32>, %s4b0bt2m: tensor<512xf32>, %s4b1W1m: tensor<512x512x3x3xf32>, %s4b1g1m: tensor<512xf32>, %s4b1bt1m: tensor<512xf32>, %s4b1W2m: tensor<512x512x3x3xf32>, %s4b1g2m: tensor<512xf32>, %s4b1bt2m: tensor<512xf32>, %Wdm: tensor<512x1000xf32>, %bdm: tensor<1000xf32>, %sWv: tensor<64x3x7x7xf32>, %sgv: tensor<64xf32>, %sbtv: tensor<64xf32>, %s1b0W1v: tensor<64x64x3x3xf32>, %s1b0g1v: tensor<64xf32>, %s1b0bt1v: tensor<64xf32>, %s1b0W2v: tensor<64x64x3x3xf32>, %s1b0g2v: tensor<64xf32>, %s1b0bt2v: tensor<64xf32>, %s1b1W1v: tensor<64x64x3x3xf32>, %s1b1g1v: tensor<64xf32>, %s1b1bt1v: tensor<64xf32>, %s1b1W2v: tensor<64x64x3x3xf32>, %s1b1g2v: tensor<64xf32>, %s1b1bt2v: tensor<64xf32>, %s1b2W1v: tensor<64x64x3x3xf32>, %s1b2g1v: tensor<64xf32>, %s1b2bt1v: tensor<64xf32>, %s1b2W2v: tensor<64x64x3x3xf32>, %s1b2g2v: tensor<64xf32>, %s1b2bt2v: tensor<64xf32>, %d2W1v: tensor<128x64x3x3xf32>, %d2g1v: tensor<128xf32>, %d2bt1v: tensor<128xf32>, %d2W2v: tensor<128x128x3x3xf32>, %d2g2v: tensor<128xf32>, %d2bt2v: tensor<128xf32>, %d2Wpv: tensor<128x64x1x1xf32>, %d2gpv: tensor<128xf32>, %d2btpv: tensor<128xf32>, %s2b0W1v: tensor<128x128x3x3xf32>, %s2b0g1v: tensor<128xf32>, %s2b0bt1v: tensor<128xf32>, %s2b0W2v: tensor<128x128x3x3xf32>, %s2b0g2v: tensor<128xf32>, %s2b0bt2v: tensor<128xf32>, %s2b1W1v: tensor<128x128x3x3xf32>, %s2b1g1v: tensor<128xf32>, %s2b1bt1v: tensor<128xf32>, %s2b1W2v: tensor<128x128x3x3xf32>, %s2b1g2v: tensor<128xf32>, %s2b1bt2v: tensor<128xf32>, %s2b2W1v: tensor<128x128x3x3xf32>, %s2b2g1v: tensor<128xf32>, %s2b2bt1v: tensor<128xf32>, %s2b2W2v: tensor<128x128x3x3xf32>, %s2b2g2v: tensor<128xf32>, %s2b2bt2v: tensor<128xf32>, %d3W1v: tensor<256x128x3x3xf32>, %d3g1v: tensor<256xf32>, %d3bt1v: tensor<256xf32>, %d3W2v: tensor<256x256x3x3xf32>, %d3g2v: tensor<256xf32>, %d3bt2v: tensor<256xf32>, %d3Wpv: tensor<256x128x1x1xf32>, %d3gpv: tensor<256xf32>, %d3btpv: tensor<256xf32>, %s3b0W1v: tensor<256x256x3x3xf32>, %s3b0g1v: tensor<256xf32>, %s3b0bt1v: tensor<256xf32>, %s3b0W2v: tensor<256x256x3x3xf32>, %s3b0g2v: tensor<256xf32>, %s3b0bt2v: tensor<256xf32>, %s3b1W1v: tensor<256x256x3x3xf32>, %s3b1g1v: tensor<256xf32>, %s3b1bt1v: tensor<256xf32>, %s3b1W2v: tensor<256x256x3x3xf32>, %s3b1g2v: tensor<256xf32>, %s3b1bt2v: tensor<256xf32>, %s3b2W1v: tensor<256x256x3x3xf32>, %s3b2g1v: tensor<256xf32>, %s3b2bt1v: tensor<256xf32>, %s3b2W2v: tensor<256x256x3x3xf32>, %s3b2g2v: tensor<256xf32>, %s3b2bt2v: tensor<256xf32>, %s3b3W1v: tensor<256x256x3x3xf32>, %s3b3g1v: tensor<256xf32>, %s3b3bt1v: tensor<256xf32>, %s3b3W2v: tensor<256x256x3x3xf32>, %s3b3g2v: tensor<256xf32>, %s3b3bt2v: tensor<256xf32>, %s3b4W1v: tensor<256x256x3x3xf32>, %s3b4g1v: tensor<256xf32>, %s3b4bt1v: tensor<256xf32>, %s3b4W2v: tensor<256x256x3x3xf32>, %s3b4g2v: tensor<256xf32>, %s3b4bt2v: tensor<256xf32>, %d4W1v: tensor<512x256x3x3xf32>, %d4g1v: tensor<512xf32>, %d4bt1v: tensor<512xf32>, %d4W2v: tensor<512x512x3x3xf32>, %d4g2v: tensor<512xf32>, %d4bt2v: tensor<512xf32>, %d4Wpv: tensor<512x256x1x1xf32>, %d4gpv: tensor<512xf32>, %d4btpv: tensor<512xf32>, %s4b0W1v: tensor<512x512x3x3xf32>, %s4b0g1v: tensor<512xf32>, %s4b0bt1v: tensor<512xf32>, %s4b0W2v: tensor<512x512x3x3xf32>, %s4b0g2v: tensor<512xf32>, %s4b0bt2v: tensor<512xf32>, %s4b1W1v: tensor<512x512x3x3xf32>, %s4b1g1v: tensor<512xf32>, %s4b1bt1v: tensor<512xf32>, %s4b1W2v: tensor<512x512x3x3xf32>, %s4b1g2v: tensor<512xf32>, %s4b1bt2v: tensor<512xf32>, %Wdv: tensor<512x1000xf32>, %bdv: tensor<1000xf32>, %lr: tensor<f32>, %bc1: tensor<f32>, %bc2: tensor<f32>, %stnmui: tensor<64xf32>, %stnvari: tensor<64xf32>, %s1b0n1mui: tensor<64xf32>, %s1b0n1vari: tensor<64xf32>, %s1b0n2mui: tensor<64xf32>, %s1b0n2vari: tensor<64xf32>, %s1b1n1mui: tensor<64xf32>, %s1b1n1vari: tensor<64xf32>, %s1b1n2mui: tensor<64xf32>, %s1b1n2vari: tensor<64xf32>, %s1b2n1mui: tensor<64xf32>, %s1b2n1vari: tensor<64xf32>, %s1b2n2mui: tensor<64xf32>, %s1b2n2vari: tensor<64xf32>, %d2n1mui: tensor<128xf32>, %d2n1vari: tensor<128xf32>, %d2n2mui: tensor<128xf32>, %d2n2vari: tensor<128xf32>, %d2npmui: tensor<128xf32>, %d2npvari: tensor<128xf32>, %s2b0n1mui: tensor<128xf32>, %s2b0n1vari: tensor<128xf32>, %s2b0n2mui: tensor<128xf32>, %s2b0n2vari: tensor<128xf32>, %s2b1n1mui: tensor<128xf32>, %s2b1n1vari: tensor<128xf32>, %s2b1n2mui: tensor<128xf32>, %s2b1n2vari: tensor<128xf32>, %s2b2n1mui: tensor<128xf32>, %s2b2n1vari: tensor<128xf32>, %s2b2n2mui: tensor<128xf32>, %s2b2n2vari: tensor<128xf32>, %d3n1mui: tensor<256xf32>, %d3n1vari: tensor<256xf32>, %d3n2mui: tensor<256xf32>, %d3n2vari: tensor<256xf32>, %d3npmui: tensor<256xf32>, %d3npvari: tensor<256xf32>, %s3b0n1mui: tensor<256xf32>, %s3b0n1vari: tensor<256xf32>, %s3b0n2mui: tensor<256xf32>, %s3b0n2vari: tensor<256xf32>, %s3b1n1mui: tensor<256xf32>, %s3b1n1vari: tensor<256xf32>, %s3b1n2mui: tensor<256xf32>, %s3b1n2vari: tensor<256xf32>, %s3b2n1mui: tensor<256xf32>, %s3b2n1vari: tensor<256xf32>, %s3b2n2mui: tensor<256xf32>, %s3b2n2vari: tensor<256xf32>, %s3b3n1mui: tensor<256xf32>, %s3b3n1vari: tensor<256xf32>, %s3b3n2mui: tensor<256xf32>, %s3b3n2vari: tensor<256xf32>, %s3b4n1mui: tensor<256xf32>, %s3b4n1vari: tensor<256xf32>, %s3b4n2mui: tensor<256xf32>, %s3b4n2vari: tensor<256xf32>, %d4n1mui: tensor<512xf32>, %d4n1vari: tensor<512xf32>, %d4n2mui: tensor<512xf32>, %d4n2vari: tensor<512xf32>, %d4npmui: tensor<512xf32>, %d4npvari: tensor<512xf32>, %s4b0n1mui: tensor<512xf32>, %s4b0n1vari: tensor<512xf32>, %s4b0n2mui: tensor<512xf32>, %s4b0n2vari: tensor<512xf32>, %s4b1n1mui: tensor<512xf32>, %s4b1n1vari: tensor<512xf32>, %s4b1n2mui: tensor<512xf32>, %s4b1n2vari: tensor<512xf32>, %onehot: tensor<64x1000xf32>) -> (tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>) {
    // ── ResNet-34 batch-BN heavy-ball momentum + coupled L2 train step, DATA-PARALLEL over 4 replicas ──
    // Every line is pretty(verified AST node), the per-parameter `%arsum*` all_reduce /
    // `%armean*` blocks included: pretty(allReduceMeanF), whose den is the replica MEAN of
    // the per-replica gradient nodes (4d piece 2). BatchNorm is SYNCHRONISED: every BN
    // layer all-reduces its mu, then var_r + (mu_r - mu)^2 (bnBatchVarAtB, Chan's parallel
    // variance), before normalising with the global [mu | var] (bnSyncF); its
    // backward all-reduces the two dy-reductions (bnSyncDyStatsB -> bnSyncBack), and the gamma
    // gradient reads the same global x-hat (bnSyncGammaGradB). Each replica therefore computes
    // its shard of the GLOBAL-batch function, and this step IS the single-device step at the
    // global batch N x b: proved as ResNet34SyncTieB.r34_net_syncTiedB (every all-reduced
    // gradient) and StableHLO.resnet34FwdGraphSync_full_shard (the forward), both in
    // LeanMlir/Proofs/Nets/ResNet/ (planning/global_bn_verified.md).
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
    %v50 = stablehlo.convert %s1b0W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v51 = stablehlo.convolution(%v49, %v50)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
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
    %v134 = stablehlo.reshape %v133 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v135 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v136 = stablehlo.add %v134, %v135 : tensor<64x64x56x56xf32>
    %v137 = stablehlo.reshape %v136 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v138 = stablehlo.reshape %v137 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v139 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v140 = stablehlo.maximum %v138, %v139 : tensor<64x64x56x56xf32>
    %v141 = stablehlo.reshape %v140 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v142 = stablehlo.reshape %v141 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v143 = stablehlo.convert %v142 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v144 = stablehlo.convert %s1b1W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v145 = stablehlo.convolution(%v143, %v144)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v146 = stablehlo.convert %v145 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v147 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v148 = stablehlo.add %v146, %v147 : tensor<64x64x56x56xf32>
    %v149 = stablehlo.reshape %v148 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v150 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v151 = stablehlo.constant dense<0.0> : tensor<f32>
    %v152 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v153 = stablehlo.reduce(%v150 init: %v151) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v154 = stablehlo.divide %v153, %v152 : tensor<64xf32>
    %arsums1b1g1mu = "stablehlo.all_reduce"(%v154) ({
    ^bb0(%aras1b1g1mu: tensor<f32>, %arbs1b1g1mu: tensor<f32>):
      %aradds1b1g1mu = stablehlo.add %aras1b1g1mu, %arbs1b1g1mu : tensor<f32>
      stablehlo.return %aradds1b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1mu = stablehlo.divide %arsums1b1g1mu, %arns1b1g1mu : tensor<64xf32>
    %v155 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v157 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v158 = stablehlo.reduce(%v155 init: %v156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v159 = stablehlo.divide %v158, %v157 : tensor<64xf32>
    %v160 = stablehlo.broadcast_in_dim %v159, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v161 = stablehlo.subtract %v155, %v160 : tensor<64x64x56x56xf32>
    %v162 = stablehlo.multiply %v161, %v161 : tensor<64x64x56x56xf32>
    %v163 = stablehlo.reduce(%v162 init: %v156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v164 = stablehlo.divide %v163, %v157 : tensor<64xf32>
    %v165 = stablehlo.subtract %v159, %armeans1b1g1mu : tensor<64xf32>
    %v166 = stablehlo.multiply %v165, %v165 : tensor<64xf32>
    %v167 = stablehlo.add %v164, %v166 : tensor<64xf32>
    %arsums1b1g1var = "stablehlo.all_reduce"(%v167) ({
    ^bb0(%aras1b1g1var: tensor<f32>, %arbs1b1g1var: tensor<f32>):
      %aradds1b1g1var = stablehlo.add %aras1b1g1var, %arbs1b1g1var : tensor<f32>
      stablehlo.return %aradds1b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1var = stablehlo.divide %arsums1b1g1var, %arns1b1g1var : tensor<64xf32>
    %v168 = stablehlo.concatenate %armeans1b1g1mu, %armeans1b1g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v169 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v170 = stablehlo.slice %v168 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v171 = stablehlo.slice %v168 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v172 = stablehlo.broadcast_in_dim %v170, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v173 = stablehlo.broadcast_in_dim %v171, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v174 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v175 = stablehlo.add %v173, %v174 : tensor<64x64x56x56xf32>
    %v176 = stablehlo.rsqrt %v175 : tensor<64x64x56x56xf32>
    %v177 = stablehlo.subtract %v169, %v172 : tensor<64x64x56x56xf32>
    %v178 = stablehlo.multiply %v177, %v176 : tensor<64x64x56x56xf32>
    %v179 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v180 = stablehlo.broadcast_in_dim %s1b1bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v181 = stablehlo.multiply %v178, %v179 : tensor<64x64x56x56xf32>
    %v182 = stablehlo.add %v181, %v180 : tensor<64x64x56x56xf32>
    %v183 = stablehlo.reshape %v182 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v184 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v185 = stablehlo.maximum %v183, %v184 : tensor<64x200704xf32>
    %v186 = stablehlo.reshape %v185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v187 = stablehlo.convert %v186 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v188 = stablehlo.convert %s1b1W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
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
    %v196 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v197 = stablehlo.reduce(%v194 init: %v195) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v198 = stablehlo.divide %v197, %v196 : tensor<64xf32>
    %arsums1b1g2mu = "stablehlo.all_reduce"(%v198) ({
    ^bb0(%aras1b1g2mu: tensor<f32>, %arbs1b1g2mu: tensor<f32>):
      %aradds1b1g2mu = stablehlo.add %aras1b1g2mu, %arbs1b1g2mu : tensor<f32>
      stablehlo.return %aradds1b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2mu = stablehlo.divide %arsums1b1g2mu, %arns1b1g2mu : tensor<64xf32>
    %v199 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v200 = stablehlo.constant dense<0.0> : tensor<f32>
    %v201 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v202 = stablehlo.reduce(%v199 init: %v200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v203 = stablehlo.divide %v202, %v201 : tensor<64xf32>
    %v204 = stablehlo.broadcast_in_dim %v203, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v205 = stablehlo.subtract %v199, %v204 : tensor<64x64x56x56xf32>
    %v206 = stablehlo.multiply %v205, %v205 : tensor<64x64x56x56xf32>
    %v207 = stablehlo.reduce(%v206 init: %v200) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v208 = stablehlo.divide %v207, %v201 : tensor<64xf32>
    %v209 = stablehlo.subtract %v203, %armeans1b1g2mu : tensor<64xf32>
    %v210 = stablehlo.multiply %v209, %v209 : tensor<64xf32>
    %v211 = stablehlo.add %v208, %v210 : tensor<64xf32>
    %arsums1b1g2var = "stablehlo.all_reduce"(%v211) ({
    ^bb0(%aras1b1g2var: tensor<f32>, %arbs1b1g2var: tensor<f32>):
      %aradds1b1g2var = stablehlo.add %aras1b1g2var, %arbs1b1g2var : tensor<f32>
      stablehlo.return %aradds1b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2var = stablehlo.divide %arsums1b1g2var, %arns1b1g2var : tensor<64xf32>
    %v212 = stablehlo.concatenate %armeans1b1g2mu, %armeans1b1g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v213 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v214 = stablehlo.slice %v212 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v215 = stablehlo.slice %v212 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v216 = stablehlo.broadcast_in_dim %v214, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v217 = stablehlo.broadcast_in_dim %v215, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v218 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v219 = stablehlo.add %v217, %v218 : tensor<64x64x56x56xf32>
    %v220 = stablehlo.rsqrt %v219 : tensor<64x64x56x56xf32>
    %v221 = stablehlo.subtract %v213, %v216 : tensor<64x64x56x56xf32>
    %v222 = stablehlo.multiply %v221, %v220 : tensor<64x64x56x56xf32>
    %v223 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v224 = stablehlo.broadcast_in_dim %s1b1bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v225 = stablehlo.multiply %v222, %v223 : tensor<64x64x56x56xf32>
    %v226 = stablehlo.add %v225, %v224 : tensor<64x64x56x56xf32>
    %v227 = stablehlo.reshape %v226 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v228 = stablehlo.reshape %v227 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v229 = stablehlo.reshape %v141 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v230 = stablehlo.add %v228, %v229 : tensor<64x64x56x56xf32>
    %v231 = stablehlo.reshape %v230 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v232 = stablehlo.reshape %v231 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v233 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v234 = stablehlo.maximum %v232, %v233 : tensor<64x64x56x56xf32>
    %v235 = stablehlo.reshape %v234 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v236 = stablehlo.reshape %v235 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v237 = stablehlo.convert %v236 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v238 = stablehlo.convert %s1b2W1 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v239 = stablehlo.convolution(%v237, %v238)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v240 = stablehlo.convert %v239 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v241 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v242 = stablehlo.add %v240, %v241 : tensor<64x64x56x56xf32>
    %v243 = stablehlo.reshape %v242 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v244 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v245 = stablehlo.constant dense<0.0> : tensor<f32>
    %v246 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v247 = stablehlo.reduce(%v244 init: %v245) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v248 = stablehlo.divide %v247, %v246 : tensor<64xf32>
    %arsums1b2g1mu = "stablehlo.all_reduce"(%v248) ({
    ^bb0(%aras1b2g1mu: tensor<f32>, %arbs1b2g1mu: tensor<f32>):
      %aradds1b2g1mu = stablehlo.add %aras1b2g1mu, %arbs1b2g1mu : tensor<f32>
      stablehlo.return %aradds1b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1mu = stablehlo.divide %arsums1b2g1mu, %arns1b2g1mu : tensor<64xf32>
    %v249 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v250 = stablehlo.constant dense<0.0> : tensor<f32>
    %v251 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v252 = stablehlo.reduce(%v249 init: %v250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v253 = stablehlo.divide %v252, %v251 : tensor<64xf32>
    %v254 = stablehlo.broadcast_in_dim %v253, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v255 = stablehlo.subtract %v249, %v254 : tensor<64x64x56x56xf32>
    %v256 = stablehlo.multiply %v255, %v255 : tensor<64x64x56x56xf32>
    %v257 = stablehlo.reduce(%v256 init: %v250) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v258 = stablehlo.divide %v257, %v251 : tensor<64xf32>
    %v259 = stablehlo.subtract %v253, %armeans1b2g1mu : tensor<64xf32>
    %v260 = stablehlo.multiply %v259, %v259 : tensor<64xf32>
    %v261 = stablehlo.add %v258, %v260 : tensor<64xf32>
    %arsums1b2g1var = "stablehlo.all_reduce"(%v261) ({
    ^bb0(%aras1b2g1var: tensor<f32>, %arbs1b2g1var: tensor<f32>):
      %aradds1b2g1var = stablehlo.add %aras1b2g1var, %arbs1b2g1var : tensor<f32>
      stablehlo.return %aradds1b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1var = stablehlo.divide %arsums1b2g1var, %arns1b2g1var : tensor<64xf32>
    %v262 = stablehlo.concatenate %armeans1b2g1mu, %armeans1b2g1var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v263 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v264 = stablehlo.slice %v262 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v265 = stablehlo.slice %v262 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v266 = stablehlo.broadcast_in_dim %v264, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v267 = stablehlo.broadcast_in_dim %v265, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v268 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v269 = stablehlo.add %v267, %v268 : tensor<64x64x56x56xf32>
    %v270 = stablehlo.rsqrt %v269 : tensor<64x64x56x56xf32>
    %v271 = stablehlo.subtract %v263, %v266 : tensor<64x64x56x56xf32>
    %v272 = stablehlo.multiply %v271, %v270 : tensor<64x64x56x56xf32>
    %v273 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v274 = stablehlo.broadcast_in_dim %s1b2bt1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v275 = stablehlo.multiply %v272, %v273 : tensor<64x64x56x56xf32>
    %v276 = stablehlo.add %v275, %v274 : tensor<64x64x56x56xf32>
    %v277 = stablehlo.reshape %v276 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v278 = stablehlo.constant dense<0.0> : tensor<64x200704xf32>
    %v279 = stablehlo.maximum %v277, %v278 : tensor<64x200704xf32>
    %v280 = stablehlo.reshape %v279 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v281 = stablehlo.convert %v280 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v282 = stablehlo.convert %s1b2W2 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v283 = stablehlo.convolution(%v281, %v282)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v284 = stablehlo.convert %v283 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v285 = stablehlo.broadcast_in_dim %zb64, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v286 = stablehlo.add %v284, %v285 : tensor<64x64x56x56xf32>
    %v287 = stablehlo.reshape %v286 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v288 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v290 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v291 = stablehlo.reduce(%v288 init: %v289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v292 = stablehlo.divide %v291, %v290 : tensor<64xf32>
    %arsums1b2g2mu = "stablehlo.all_reduce"(%v292) ({
    ^bb0(%aras1b2g2mu: tensor<f32>, %arbs1b2g2mu: tensor<f32>):
      %aradds1b2g2mu = stablehlo.add %aras1b2g2mu, %arbs1b2g2mu : tensor<f32>
      stablehlo.return %aradds1b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2mu = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2mu = stablehlo.divide %arsums1b2g2mu, %arns1b2g2mu : tensor<64xf32>
    %v293 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v294 = stablehlo.constant dense<0.0> : tensor<f32>
    %v295 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v296 = stablehlo.reduce(%v293 init: %v294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v297 = stablehlo.divide %v296, %v295 : tensor<64xf32>
    %v298 = stablehlo.broadcast_in_dim %v297, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v299 = stablehlo.subtract %v293, %v298 : tensor<64x64x56x56xf32>
    %v300 = stablehlo.multiply %v299, %v299 : tensor<64x64x56x56xf32>
    %v301 = stablehlo.reduce(%v300 init: %v294) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v302 = stablehlo.divide %v301, %v295 : tensor<64xf32>
    %v303 = stablehlo.subtract %v297, %armeans1b2g2mu : tensor<64xf32>
    %v304 = stablehlo.multiply %v303, %v303 : tensor<64xf32>
    %v305 = stablehlo.add %v302, %v304 : tensor<64xf32>
    %arsums1b2g2var = "stablehlo.all_reduce"(%v305) ({
    ^bb0(%aras1b2g2var: tensor<f32>, %arbs1b2g2var: tensor<f32>):
      %aradds1b2g2var = stablehlo.add %aras1b2g2var, %arbs1b2g2var : tensor<f32>
      stablehlo.return %aradds1b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2var = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2var = stablehlo.divide %arsums1b2g2var, %arns1b2g2var : tensor<64xf32>
    %v306 = stablehlo.concatenate %armeans1b2g2mu, %armeans1b2g2var, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v307 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v308 = stablehlo.slice %v306 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v309 = stablehlo.slice %v306 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v310 = stablehlo.broadcast_in_dim %v308, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v311 = stablehlo.broadcast_in_dim %v309, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v312 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v313 = stablehlo.add %v311, %v312 : tensor<64x64x56x56xf32>
    %v314 = stablehlo.rsqrt %v313 : tensor<64x64x56x56xf32>
    %v315 = stablehlo.subtract %v307, %v310 : tensor<64x64x56x56xf32>
    %v316 = stablehlo.multiply %v315, %v314 : tensor<64x64x56x56xf32>
    %v317 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v318 = stablehlo.broadcast_in_dim %s1b2bt2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v319 = stablehlo.multiply %v316, %v317 : tensor<64x64x56x56xf32>
    %v320 = stablehlo.add %v319, %v318 : tensor<64x64x56x56xf32>
    %v321 = stablehlo.reshape %v320 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v322 = stablehlo.reshape %v321 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v323 = stablehlo.reshape %v235 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v324 = stablehlo.add %v322, %v323 : tensor<64x64x56x56xf32>
    %v325 = stablehlo.reshape %v324 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v326 = stablehlo.reshape %v325 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v327 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v328 = stablehlo.maximum %v326, %v327 : tensor<64x64x56x56xf32>
    %v329 = stablehlo.reshape %v328 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v330 = stablehlo.reshape %v329 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v331 = stablehlo.convert %v330 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v332 = stablehlo.convert %d2W1 : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xbf16>
    %v333 = stablehlo.convolution(%v331, %v332)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v334 = stablehlo.convert %v333 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v335 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v336 = stablehlo.add %v334, %v335 : tensor<64x128x28x28xf32>
    %v337 = stablehlo.reshape %v336 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v338 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v339 = stablehlo.constant dense<0.0> : tensor<f32>
    %v340 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v341 = stablehlo.reduce(%v338 init: %v339) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v342 = stablehlo.divide %v341, %v340 : tensor<128xf32>
    %arsumd2g1mu = "stablehlo.all_reduce"(%v342) ({
    ^bb0(%arad2g1mu: tensor<f32>, %arbd2g1mu: tensor<f32>):
      %araddd2g1mu = stablehlo.add %arad2g1mu, %arbd2g1mu : tensor<f32>
      stablehlo.return %araddd2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g1mu = stablehlo.divide %arsumd2g1mu, %arnd2g1mu : tensor<128xf32>
    %v343 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v344 = stablehlo.constant dense<0.0> : tensor<f32>
    %v345 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v346 = stablehlo.reduce(%v343 init: %v344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v347 = stablehlo.divide %v346, %v345 : tensor<128xf32>
    %v348 = stablehlo.broadcast_in_dim %v347, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v349 = stablehlo.subtract %v343, %v348 : tensor<64x128x28x28xf32>
    %v350 = stablehlo.multiply %v349, %v349 : tensor<64x128x28x28xf32>
    %v351 = stablehlo.reduce(%v350 init: %v344) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v352 = stablehlo.divide %v351, %v345 : tensor<128xf32>
    %v353 = stablehlo.subtract %v347, %armeand2g1mu : tensor<128xf32>
    %v354 = stablehlo.multiply %v353, %v353 : tensor<128xf32>
    %v355 = stablehlo.add %v352, %v354 : tensor<128xf32>
    %arsumd2g1var = "stablehlo.all_reduce"(%v355) ({
    ^bb0(%arad2g1var: tensor<f32>, %arbd2g1var: tensor<f32>):
      %araddd2g1var = stablehlo.add %arad2g1var, %arbd2g1var : tensor<f32>
      stablehlo.return %araddd2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g1var = stablehlo.divide %arsumd2g1var, %arnd2g1var : tensor<128xf32>
    %v356 = stablehlo.concatenate %armeand2g1mu, %armeand2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v357 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v358 = stablehlo.slice %v356 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v359 = stablehlo.slice %v356 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v360 = stablehlo.broadcast_in_dim %v358, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v361 = stablehlo.broadcast_in_dim %v359, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v362 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v363 = stablehlo.add %v361, %v362 : tensor<64x128x28x28xf32>
    %v364 = stablehlo.rsqrt %v363 : tensor<64x128x28x28xf32>
    %v365 = stablehlo.subtract %v357, %v360 : tensor<64x128x28x28xf32>
    %v366 = stablehlo.multiply %v365, %v364 : tensor<64x128x28x28xf32>
    %v367 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v368 = stablehlo.broadcast_in_dim %d2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v369 = stablehlo.multiply %v366, %v367 : tensor<64x128x28x28xf32>
    %v370 = stablehlo.add %v369, %v368 : tensor<64x128x28x28xf32>
    %v371 = stablehlo.reshape %v370 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v372 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v373 = stablehlo.maximum %v371, %v372 : tensor<64x100352xf32>
    %v374 = stablehlo.reshape %v373 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v375 = stablehlo.convert %v374 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v376 = stablehlo.convert %d2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v377 = stablehlo.convolution(%v375, %v376)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v378 = stablehlo.convert %v377 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v379 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v380 = stablehlo.add %v378, %v379 : tensor<64x128x28x28xf32>
    %v381 = stablehlo.reshape %v380 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v382 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v384 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v385 = stablehlo.reduce(%v382 init: %v383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v386 = stablehlo.divide %v385, %v384 : tensor<128xf32>
    %arsumd2g2mu = "stablehlo.all_reduce"(%v386) ({
    ^bb0(%arad2g2mu: tensor<f32>, %arbd2g2mu: tensor<f32>):
      %araddd2g2mu = stablehlo.add %arad2g2mu, %arbd2g2mu : tensor<f32>
      stablehlo.return %araddd2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g2mu = stablehlo.divide %arsumd2g2mu, %arnd2g2mu : tensor<128xf32>
    %v387 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v388 = stablehlo.constant dense<0.0> : tensor<f32>
    %v389 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v390 = stablehlo.reduce(%v387 init: %v388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v391 = stablehlo.divide %v390, %v389 : tensor<128xf32>
    %v392 = stablehlo.broadcast_in_dim %v391, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v393 = stablehlo.subtract %v387, %v392 : tensor<64x128x28x28xf32>
    %v394 = stablehlo.multiply %v393, %v393 : tensor<64x128x28x28xf32>
    %v395 = stablehlo.reduce(%v394 init: %v388) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v396 = stablehlo.divide %v395, %v389 : tensor<128xf32>
    %v397 = stablehlo.subtract %v391, %armeand2g2mu : tensor<128xf32>
    %v398 = stablehlo.multiply %v397, %v397 : tensor<128xf32>
    %v399 = stablehlo.add %v396, %v398 : tensor<128xf32>
    %arsumd2g2var = "stablehlo.all_reduce"(%v399) ({
    ^bb0(%arad2g2var: tensor<f32>, %arbd2g2var: tensor<f32>):
      %araddd2g2var = stablehlo.add %arad2g2var, %arbd2g2var : tensor<f32>
      stablehlo.return %araddd2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g2var = stablehlo.divide %arsumd2g2var, %arnd2g2var : tensor<128xf32>
    %v400 = stablehlo.concatenate %armeand2g2mu, %armeand2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v401 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v402 = stablehlo.slice %v400 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v403 = stablehlo.slice %v400 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v404 = stablehlo.broadcast_in_dim %v402, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v405 = stablehlo.broadcast_in_dim %v403, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v406 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v407 = stablehlo.add %v405, %v406 : tensor<64x128x28x28xf32>
    %v408 = stablehlo.rsqrt %v407 : tensor<64x128x28x28xf32>
    %v409 = stablehlo.subtract %v401, %v404 : tensor<64x128x28x28xf32>
    %v410 = stablehlo.multiply %v409, %v408 : tensor<64x128x28x28xf32>
    %v411 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v412 = stablehlo.broadcast_in_dim %d2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v413 = stablehlo.multiply %v410, %v411 : tensor<64x128x28x28xf32>
    %v414 = stablehlo.add %v413, %v412 : tensor<64x128x28x28xf32>
    %v415 = stablehlo.reshape %v414 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v416 = stablehlo.reshape %v329 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v417 = stablehlo.convert %v416 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v418 = stablehlo.convert %d2Wp : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xbf16>
    %v419 = stablehlo.convolution(%v417, %v418)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v420 = stablehlo.convert %v419 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v421 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v422 = stablehlo.add %v420, %v421 : tensor<64x128x28x28xf32>
    %v423 = stablehlo.reshape %v422 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v424 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v425 = stablehlo.constant dense<0.0> : tensor<f32>
    %v426 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v427 = stablehlo.reduce(%v424 init: %v425) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v428 = stablehlo.divide %v427, %v426 : tensor<128xf32>
    %arsumd2gpmu = "stablehlo.all_reduce"(%v428) ({
    ^bb0(%arad2gpmu: tensor<f32>, %arbd2gpmu: tensor<f32>):
      %araddd2gpmu = stablehlo.add %arad2gpmu, %arbd2gpmu : tensor<f32>
      stablehlo.return %araddd2gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gpmu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2gpmu = stablehlo.divide %arsumd2gpmu, %arnd2gpmu : tensor<128xf32>
    %v429 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v430 = stablehlo.constant dense<0.0> : tensor<f32>
    %v431 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v432 = stablehlo.reduce(%v429 init: %v430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v433 = stablehlo.divide %v432, %v431 : tensor<128xf32>
    %v434 = stablehlo.broadcast_in_dim %v433, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v435 = stablehlo.subtract %v429, %v434 : tensor<64x128x28x28xf32>
    %v436 = stablehlo.multiply %v435, %v435 : tensor<64x128x28x28xf32>
    %v437 = stablehlo.reduce(%v436 init: %v430) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v438 = stablehlo.divide %v437, %v431 : tensor<128xf32>
    %v439 = stablehlo.subtract %v433, %armeand2gpmu : tensor<128xf32>
    %v440 = stablehlo.multiply %v439, %v439 : tensor<128xf32>
    %v441 = stablehlo.add %v438, %v440 : tensor<128xf32>
    %arsumd2gpvar = "stablehlo.all_reduce"(%v441) ({
    ^bb0(%arad2gpvar: tensor<f32>, %arbd2gpvar: tensor<f32>):
      %araddd2gpvar = stablehlo.add %arad2gpvar, %arbd2gpvar : tensor<f32>
      stablehlo.return %araddd2gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gpvar = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2gpvar = stablehlo.divide %arsumd2gpvar, %arnd2gpvar : tensor<128xf32>
    %v442 = stablehlo.concatenate %armeand2gpmu, %armeand2gpvar, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v443 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v444 = stablehlo.slice %v442 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v445 = stablehlo.slice %v442 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v446 = stablehlo.broadcast_in_dim %v444, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v447 = stablehlo.broadcast_in_dim %v445, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v448 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v449 = stablehlo.add %v447, %v448 : tensor<64x128x28x28xf32>
    %v450 = stablehlo.rsqrt %v449 : tensor<64x128x28x28xf32>
    %v451 = stablehlo.subtract %v443, %v446 : tensor<64x128x28x28xf32>
    %v452 = stablehlo.multiply %v451, %v450 : tensor<64x128x28x28xf32>
    %v453 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v454 = stablehlo.broadcast_in_dim %d2btp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v455 = stablehlo.multiply %v452, %v453 : tensor<64x128x28x28xf32>
    %v456 = stablehlo.add %v455, %v454 : tensor<64x128x28x28xf32>
    %v457 = stablehlo.reshape %v456 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v458 = stablehlo.add %v415, %v457 : tensor<64x100352xf32>
    %v459 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v460 = stablehlo.maximum %v458, %v459 : tensor<64x100352xf32>
    %v461 = stablehlo.reshape %v460 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v462 = stablehlo.convert %v461 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v463 = stablehlo.convert %s2b0W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v464 = stablehlo.convolution(%v462, %v463)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v465 = stablehlo.convert %v464 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v466 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v467 = stablehlo.add %v465, %v466 : tensor<64x128x28x28xf32>
    %v468 = stablehlo.reshape %v467 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v469 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v470 = stablehlo.constant dense<0.0> : tensor<f32>
    %v471 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v472 = stablehlo.reduce(%v469 init: %v470) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v473 = stablehlo.divide %v472, %v471 : tensor<128xf32>
    %arsums2b0g1mu = "stablehlo.all_reduce"(%v473) ({
    ^bb0(%aras2b0g1mu: tensor<f32>, %arbs2b0g1mu: tensor<f32>):
      %aradds2b0g1mu = stablehlo.add %aras2b0g1mu, %arbs2b0g1mu : tensor<f32>
      stablehlo.return %aradds2b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1mu = stablehlo.divide %arsums2b0g1mu, %arns2b0g1mu : tensor<128xf32>
    %v474 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v475 = stablehlo.constant dense<0.0> : tensor<f32>
    %v476 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v477 = stablehlo.reduce(%v474 init: %v475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v478 = stablehlo.divide %v477, %v476 : tensor<128xf32>
    %v479 = stablehlo.broadcast_in_dim %v478, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v480 = stablehlo.subtract %v474, %v479 : tensor<64x128x28x28xf32>
    %v481 = stablehlo.multiply %v480, %v480 : tensor<64x128x28x28xf32>
    %v482 = stablehlo.reduce(%v481 init: %v475) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v483 = stablehlo.divide %v482, %v476 : tensor<128xf32>
    %v484 = stablehlo.subtract %v478, %armeans2b0g1mu : tensor<128xf32>
    %v485 = stablehlo.multiply %v484, %v484 : tensor<128xf32>
    %v486 = stablehlo.add %v483, %v485 : tensor<128xf32>
    %arsums2b0g1var = "stablehlo.all_reduce"(%v486) ({
    ^bb0(%aras2b0g1var: tensor<f32>, %arbs2b0g1var: tensor<f32>):
      %aradds2b0g1var = stablehlo.add %aras2b0g1var, %arbs2b0g1var : tensor<f32>
      stablehlo.return %aradds2b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1var = stablehlo.divide %arsums2b0g1var, %arns2b0g1var : tensor<128xf32>
    %v487 = stablehlo.concatenate %armeans2b0g1mu, %armeans2b0g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v488 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v489 = stablehlo.slice %v487 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v490 = stablehlo.slice %v487 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v491 = stablehlo.broadcast_in_dim %v489, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v492 = stablehlo.broadcast_in_dim %v490, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v493 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v494 = stablehlo.add %v492, %v493 : tensor<64x128x28x28xf32>
    %v495 = stablehlo.rsqrt %v494 : tensor<64x128x28x28xf32>
    %v496 = stablehlo.subtract %v488, %v491 : tensor<64x128x28x28xf32>
    %v497 = stablehlo.multiply %v496, %v495 : tensor<64x128x28x28xf32>
    %v498 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v499 = stablehlo.broadcast_in_dim %s2b0bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v500 = stablehlo.multiply %v497, %v498 : tensor<64x128x28x28xf32>
    %v501 = stablehlo.add %v500, %v499 : tensor<64x128x28x28xf32>
    %v502 = stablehlo.reshape %v501 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v503 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v504 = stablehlo.maximum %v502, %v503 : tensor<64x100352xf32>
    %v505 = stablehlo.reshape %v504 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v506 = stablehlo.convert %v505 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v507 = stablehlo.convert %s2b0W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v508 = stablehlo.convolution(%v506, %v507)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v509 = stablehlo.convert %v508 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v510 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v511 = stablehlo.add %v509, %v510 : tensor<64x128x28x28xf32>
    %v512 = stablehlo.reshape %v511 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v513 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v514 = stablehlo.constant dense<0.0> : tensor<f32>
    %v515 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v516 = stablehlo.reduce(%v513 init: %v514) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v517 = stablehlo.divide %v516, %v515 : tensor<128xf32>
    %arsums2b0g2mu = "stablehlo.all_reduce"(%v517) ({
    ^bb0(%aras2b0g2mu: tensor<f32>, %arbs2b0g2mu: tensor<f32>):
      %aradds2b0g2mu = stablehlo.add %aras2b0g2mu, %arbs2b0g2mu : tensor<f32>
      stablehlo.return %aradds2b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2mu = stablehlo.divide %arsums2b0g2mu, %arns2b0g2mu : tensor<128xf32>
    %v518 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v519 = stablehlo.constant dense<0.0> : tensor<f32>
    %v520 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v521 = stablehlo.reduce(%v518 init: %v519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v522 = stablehlo.divide %v521, %v520 : tensor<128xf32>
    %v523 = stablehlo.broadcast_in_dim %v522, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v524 = stablehlo.subtract %v518, %v523 : tensor<64x128x28x28xf32>
    %v525 = stablehlo.multiply %v524, %v524 : tensor<64x128x28x28xf32>
    %v526 = stablehlo.reduce(%v525 init: %v519) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v527 = stablehlo.divide %v526, %v520 : tensor<128xf32>
    %v528 = stablehlo.subtract %v522, %armeans2b0g2mu : tensor<128xf32>
    %v529 = stablehlo.multiply %v528, %v528 : tensor<128xf32>
    %v530 = stablehlo.add %v527, %v529 : tensor<128xf32>
    %arsums2b0g2var = "stablehlo.all_reduce"(%v530) ({
    ^bb0(%aras2b0g2var: tensor<f32>, %arbs2b0g2var: tensor<f32>):
      %aradds2b0g2var = stablehlo.add %aras2b0g2var, %arbs2b0g2var : tensor<f32>
      stablehlo.return %aradds2b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2var = stablehlo.divide %arsums2b0g2var, %arns2b0g2var : tensor<128xf32>
    %v531 = stablehlo.concatenate %armeans2b0g2mu, %armeans2b0g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v532 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v533 = stablehlo.slice %v531 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v534 = stablehlo.slice %v531 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v535 = stablehlo.broadcast_in_dim %v533, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v536 = stablehlo.broadcast_in_dim %v534, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v537 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v538 = stablehlo.add %v536, %v537 : tensor<64x128x28x28xf32>
    %v539 = stablehlo.rsqrt %v538 : tensor<64x128x28x28xf32>
    %v540 = stablehlo.subtract %v532, %v535 : tensor<64x128x28x28xf32>
    %v541 = stablehlo.multiply %v540, %v539 : tensor<64x128x28x28xf32>
    %v542 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v543 = stablehlo.broadcast_in_dim %s2b0bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v544 = stablehlo.multiply %v541, %v542 : tensor<64x128x28x28xf32>
    %v545 = stablehlo.add %v544, %v543 : tensor<64x128x28x28xf32>
    %v546 = stablehlo.reshape %v545 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v547 = stablehlo.reshape %v546 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v548 = stablehlo.reshape %v460 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v549 = stablehlo.add %v547, %v548 : tensor<64x128x28x28xf32>
    %v550 = stablehlo.reshape %v549 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v551 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v552 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v553 = stablehlo.maximum %v551, %v552 : tensor<64x128x28x28xf32>
    %v554 = stablehlo.reshape %v553 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v555 = stablehlo.reshape %v554 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v556 = stablehlo.convert %v555 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v557 = stablehlo.convert %s2b1W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v558 = stablehlo.convolution(%v556, %v557)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v559 = stablehlo.convert %v558 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v560 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v561 = stablehlo.add %v559, %v560 : tensor<64x128x28x28xf32>
    %v562 = stablehlo.reshape %v561 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v563 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v564 = stablehlo.constant dense<0.0> : tensor<f32>
    %v565 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v566 = stablehlo.reduce(%v563 init: %v564) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v567 = stablehlo.divide %v566, %v565 : tensor<128xf32>
    %arsums2b1g1mu = "stablehlo.all_reduce"(%v567) ({
    ^bb0(%aras2b1g1mu: tensor<f32>, %arbs2b1g1mu: tensor<f32>):
      %aradds2b1g1mu = stablehlo.add %aras2b1g1mu, %arbs2b1g1mu : tensor<f32>
      stablehlo.return %aradds2b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1mu = stablehlo.divide %arsums2b1g1mu, %arns2b1g1mu : tensor<128xf32>
    %v568 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v569 = stablehlo.constant dense<0.0> : tensor<f32>
    %v570 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v571 = stablehlo.reduce(%v568 init: %v569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v572 = stablehlo.divide %v571, %v570 : tensor<128xf32>
    %v573 = stablehlo.broadcast_in_dim %v572, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v574 = stablehlo.subtract %v568, %v573 : tensor<64x128x28x28xf32>
    %v575 = stablehlo.multiply %v574, %v574 : tensor<64x128x28x28xf32>
    %v576 = stablehlo.reduce(%v575 init: %v569) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v577 = stablehlo.divide %v576, %v570 : tensor<128xf32>
    %v578 = stablehlo.subtract %v572, %armeans2b1g1mu : tensor<128xf32>
    %v579 = stablehlo.multiply %v578, %v578 : tensor<128xf32>
    %v580 = stablehlo.add %v577, %v579 : tensor<128xf32>
    %arsums2b1g1var = "stablehlo.all_reduce"(%v580) ({
    ^bb0(%aras2b1g1var: tensor<f32>, %arbs2b1g1var: tensor<f32>):
      %aradds2b1g1var = stablehlo.add %aras2b1g1var, %arbs2b1g1var : tensor<f32>
      stablehlo.return %aradds2b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1var = stablehlo.divide %arsums2b1g1var, %arns2b1g1var : tensor<128xf32>
    %v581 = stablehlo.concatenate %armeans2b1g1mu, %armeans2b1g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v582 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v583 = stablehlo.slice %v581 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v584 = stablehlo.slice %v581 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v585 = stablehlo.broadcast_in_dim %v583, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v586 = stablehlo.broadcast_in_dim %v584, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v587 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v588 = stablehlo.add %v586, %v587 : tensor<64x128x28x28xf32>
    %v589 = stablehlo.rsqrt %v588 : tensor<64x128x28x28xf32>
    %v590 = stablehlo.subtract %v582, %v585 : tensor<64x128x28x28xf32>
    %v591 = stablehlo.multiply %v590, %v589 : tensor<64x128x28x28xf32>
    %v592 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v593 = stablehlo.broadcast_in_dim %s2b1bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v594 = stablehlo.multiply %v591, %v592 : tensor<64x128x28x28xf32>
    %v595 = stablehlo.add %v594, %v593 : tensor<64x128x28x28xf32>
    %v596 = stablehlo.reshape %v595 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v597 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v598 = stablehlo.maximum %v596, %v597 : tensor<64x100352xf32>
    %v599 = stablehlo.reshape %v598 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v600 = stablehlo.convert %v599 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v601 = stablehlo.convert %s2b1W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v602 = stablehlo.convolution(%v600, %v601)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v603 = stablehlo.convert %v602 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v604 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v605 = stablehlo.add %v603, %v604 : tensor<64x128x28x28xf32>
    %v606 = stablehlo.reshape %v605 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v607 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v608 = stablehlo.constant dense<0.0> : tensor<f32>
    %v609 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v610 = stablehlo.reduce(%v607 init: %v608) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v611 = stablehlo.divide %v610, %v609 : tensor<128xf32>
    %arsums2b1g2mu = "stablehlo.all_reduce"(%v611) ({
    ^bb0(%aras2b1g2mu: tensor<f32>, %arbs2b1g2mu: tensor<f32>):
      %aradds2b1g2mu = stablehlo.add %aras2b1g2mu, %arbs2b1g2mu : tensor<f32>
      stablehlo.return %aradds2b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2mu = stablehlo.divide %arsums2b1g2mu, %arns2b1g2mu : tensor<128xf32>
    %v612 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v613 = stablehlo.constant dense<0.0> : tensor<f32>
    %v614 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v615 = stablehlo.reduce(%v612 init: %v613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v616 = stablehlo.divide %v615, %v614 : tensor<128xf32>
    %v617 = stablehlo.broadcast_in_dim %v616, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v618 = stablehlo.subtract %v612, %v617 : tensor<64x128x28x28xf32>
    %v619 = stablehlo.multiply %v618, %v618 : tensor<64x128x28x28xf32>
    %v620 = stablehlo.reduce(%v619 init: %v613) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v621 = stablehlo.divide %v620, %v614 : tensor<128xf32>
    %v622 = stablehlo.subtract %v616, %armeans2b1g2mu : tensor<128xf32>
    %v623 = stablehlo.multiply %v622, %v622 : tensor<128xf32>
    %v624 = stablehlo.add %v621, %v623 : tensor<128xf32>
    %arsums2b1g2var = "stablehlo.all_reduce"(%v624) ({
    ^bb0(%aras2b1g2var: tensor<f32>, %arbs2b1g2var: tensor<f32>):
      %aradds2b1g2var = stablehlo.add %aras2b1g2var, %arbs2b1g2var : tensor<f32>
      stablehlo.return %aradds2b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2var = stablehlo.divide %arsums2b1g2var, %arns2b1g2var : tensor<128xf32>
    %v625 = stablehlo.concatenate %armeans2b1g2mu, %armeans2b1g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v626 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v627 = stablehlo.slice %v625 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v628 = stablehlo.slice %v625 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v629 = stablehlo.broadcast_in_dim %v627, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v630 = stablehlo.broadcast_in_dim %v628, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v631 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v632 = stablehlo.add %v630, %v631 : tensor<64x128x28x28xf32>
    %v633 = stablehlo.rsqrt %v632 : tensor<64x128x28x28xf32>
    %v634 = stablehlo.subtract %v626, %v629 : tensor<64x128x28x28xf32>
    %v635 = stablehlo.multiply %v634, %v633 : tensor<64x128x28x28xf32>
    %v636 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v637 = stablehlo.broadcast_in_dim %s2b1bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v638 = stablehlo.multiply %v635, %v636 : tensor<64x128x28x28xf32>
    %v639 = stablehlo.add %v638, %v637 : tensor<64x128x28x28xf32>
    %v640 = stablehlo.reshape %v639 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v641 = stablehlo.reshape %v640 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v642 = stablehlo.reshape %v554 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v643 = stablehlo.add %v641, %v642 : tensor<64x128x28x28xf32>
    %v644 = stablehlo.reshape %v643 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v645 = stablehlo.reshape %v644 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v646 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v647 = stablehlo.maximum %v645, %v646 : tensor<64x128x28x28xf32>
    %v648 = stablehlo.reshape %v647 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v649 = stablehlo.reshape %v648 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v650 = stablehlo.convert %v649 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v651 = stablehlo.convert %s2b2W1 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v652 = stablehlo.convolution(%v650, %v651)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v653 = stablehlo.convert %v652 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v654 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v655 = stablehlo.add %v653, %v654 : tensor<64x128x28x28xf32>
    %v656 = stablehlo.reshape %v655 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v657 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v658 = stablehlo.constant dense<0.0> : tensor<f32>
    %v659 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v660 = stablehlo.reduce(%v657 init: %v658) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v661 = stablehlo.divide %v660, %v659 : tensor<128xf32>
    %arsums2b2g1mu = "stablehlo.all_reduce"(%v661) ({
    ^bb0(%aras2b2g1mu: tensor<f32>, %arbs2b2g1mu: tensor<f32>):
      %aradds2b2g1mu = stablehlo.add %aras2b2g1mu, %arbs2b2g1mu : tensor<f32>
      stablehlo.return %aradds2b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1mu = stablehlo.divide %arsums2b2g1mu, %arns2b2g1mu : tensor<128xf32>
    %v662 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v663 = stablehlo.constant dense<0.0> : tensor<f32>
    %v664 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v665 = stablehlo.reduce(%v662 init: %v663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v666 = stablehlo.divide %v665, %v664 : tensor<128xf32>
    %v667 = stablehlo.broadcast_in_dim %v666, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v668 = stablehlo.subtract %v662, %v667 : tensor<64x128x28x28xf32>
    %v669 = stablehlo.multiply %v668, %v668 : tensor<64x128x28x28xf32>
    %v670 = stablehlo.reduce(%v669 init: %v663) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v671 = stablehlo.divide %v670, %v664 : tensor<128xf32>
    %v672 = stablehlo.subtract %v666, %armeans2b2g1mu : tensor<128xf32>
    %v673 = stablehlo.multiply %v672, %v672 : tensor<128xf32>
    %v674 = stablehlo.add %v671, %v673 : tensor<128xf32>
    %arsums2b2g1var = "stablehlo.all_reduce"(%v674) ({
    ^bb0(%aras2b2g1var: tensor<f32>, %arbs2b2g1var: tensor<f32>):
      %aradds2b2g1var = stablehlo.add %aras2b2g1var, %arbs2b2g1var : tensor<f32>
      stablehlo.return %aradds2b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1var = stablehlo.divide %arsums2b2g1var, %arns2b2g1var : tensor<128xf32>
    %v675 = stablehlo.concatenate %armeans2b2g1mu, %armeans2b2g1var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v676 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v677 = stablehlo.slice %v675 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v678 = stablehlo.slice %v675 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v679 = stablehlo.broadcast_in_dim %v677, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v680 = stablehlo.broadcast_in_dim %v678, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v681 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v682 = stablehlo.add %v680, %v681 : tensor<64x128x28x28xf32>
    %v683 = stablehlo.rsqrt %v682 : tensor<64x128x28x28xf32>
    %v684 = stablehlo.subtract %v676, %v679 : tensor<64x128x28x28xf32>
    %v685 = stablehlo.multiply %v684, %v683 : tensor<64x128x28x28xf32>
    %v686 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v687 = stablehlo.broadcast_in_dim %s2b2bt1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v688 = stablehlo.multiply %v685, %v686 : tensor<64x128x28x28xf32>
    %v689 = stablehlo.add %v688, %v687 : tensor<64x128x28x28xf32>
    %v690 = stablehlo.reshape %v689 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v691 = stablehlo.constant dense<0.0> : tensor<64x100352xf32>
    %v692 = stablehlo.maximum %v690, %v691 : tensor<64x100352xf32>
    %v693 = stablehlo.reshape %v692 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v694 = stablehlo.convert %v693 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v695 = stablehlo.convert %s2b2W2 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v696 = stablehlo.convolution(%v694, %v695)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v697 = stablehlo.convert %v696 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v698 = stablehlo.broadcast_in_dim %zb128, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v699 = stablehlo.add %v697, %v698 : tensor<64x128x28x28xf32>
    %v700 = stablehlo.reshape %v699 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v701 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v702 = stablehlo.constant dense<0.0> : tensor<f32>
    %v703 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v704 = stablehlo.reduce(%v701 init: %v702) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v705 = stablehlo.divide %v704, %v703 : tensor<128xf32>
    %arsums2b2g2mu = "stablehlo.all_reduce"(%v705) ({
    ^bb0(%aras2b2g2mu: tensor<f32>, %arbs2b2g2mu: tensor<f32>):
      %aradds2b2g2mu = stablehlo.add %aras2b2g2mu, %arbs2b2g2mu : tensor<f32>
      stablehlo.return %aradds2b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2mu = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2mu = stablehlo.divide %arsums2b2g2mu, %arns2b2g2mu : tensor<128xf32>
    %v706 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v707 = stablehlo.constant dense<0.0> : tensor<f32>
    %v708 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v709 = stablehlo.reduce(%v706 init: %v707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v710 = stablehlo.divide %v709, %v708 : tensor<128xf32>
    %v711 = stablehlo.broadcast_in_dim %v710, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v712 = stablehlo.subtract %v706, %v711 : tensor<64x128x28x28xf32>
    %v713 = stablehlo.multiply %v712, %v712 : tensor<64x128x28x28xf32>
    %v714 = stablehlo.reduce(%v713 init: %v707) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v715 = stablehlo.divide %v714, %v708 : tensor<128xf32>
    %v716 = stablehlo.subtract %v710, %armeans2b2g2mu : tensor<128xf32>
    %v717 = stablehlo.multiply %v716, %v716 : tensor<128xf32>
    %v718 = stablehlo.add %v715, %v717 : tensor<128xf32>
    %arsums2b2g2var = "stablehlo.all_reduce"(%v718) ({
    ^bb0(%aras2b2g2var: tensor<f32>, %arbs2b2g2var: tensor<f32>):
      %aradds2b2g2var = stablehlo.add %aras2b2g2var, %arbs2b2g2var : tensor<f32>
      stablehlo.return %aradds2b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2var = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2var = stablehlo.divide %arsums2b2g2var, %arns2b2g2var : tensor<128xf32>
    %v719 = stablehlo.concatenate %armeans2b2g2mu, %armeans2b2g2var, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v720 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v721 = stablehlo.slice %v719 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v722 = stablehlo.slice %v719 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v723 = stablehlo.broadcast_in_dim %v721, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v724 = stablehlo.broadcast_in_dim %v722, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v725 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v726 = stablehlo.add %v724, %v725 : tensor<64x128x28x28xf32>
    %v727 = stablehlo.rsqrt %v726 : tensor<64x128x28x28xf32>
    %v728 = stablehlo.subtract %v720, %v723 : tensor<64x128x28x28xf32>
    %v729 = stablehlo.multiply %v728, %v727 : tensor<64x128x28x28xf32>
    %v730 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v731 = stablehlo.broadcast_in_dim %s2b2bt2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v732 = stablehlo.multiply %v729, %v730 : tensor<64x128x28x28xf32>
    %v733 = stablehlo.add %v732, %v731 : tensor<64x128x28x28xf32>
    %v734 = stablehlo.reshape %v733 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v735 = stablehlo.reshape %v734 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v736 = stablehlo.reshape %v648 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v737 = stablehlo.add %v735, %v736 : tensor<64x128x28x28xf32>
    %v738 = stablehlo.reshape %v737 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v739 = stablehlo.reshape %v738 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v740 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v741 = stablehlo.maximum %v739, %v740 : tensor<64x128x28x28xf32>
    %v742 = stablehlo.reshape %v741 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v743 = stablehlo.reshape %v742 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v744 = stablehlo.convert %v743 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v745 = stablehlo.convert %d3W1 : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xbf16>
    %v746 = stablehlo.convolution(%v744, %v745)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<256x128x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v747 = stablehlo.convert %v746 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v748 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v749 = stablehlo.add %v747, %v748 : tensor<64x256x14x14xf32>
    %v750 = stablehlo.reshape %v749 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v751 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v752 = stablehlo.constant dense<0.0> : tensor<f32>
    %v753 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v754 = stablehlo.reduce(%v751 init: %v752) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v755 = stablehlo.divide %v754, %v753 : tensor<256xf32>
    %arsumd3g1mu = "stablehlo.all_reduce"(%v755) ({
    ^bb0(%arad3g1mu: tensor<f32>, %arbd3g1mu: tensor<f32>):
      %araddd3g1mu = stablehlo.add %arad3g1mu, %arbd3g1mu : tensor<f32>
      stablehlo.return %araddd3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g1mu = stablehlo.divide %arsumd3g1mu, %arnd3g1mu : tensor<256xf32>
    %v756 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v757 = stablehlo.constant dense<0.0> : tensor<f32>
    %v758 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v759 = stablehlo.reduce(%v756 init: %v757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v760 = stablehlo.divide %v759, %v758 : tensor<256xf32>
    %v761 = stablehlo.broadcast_in_dim %v760, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v762 = stablehlo.subtract %v756, %v761 : tensor<64x256x14x14xf32>
    %v763 = stablehlo.multiply %v762, %v762 : tensor<64x256x14x14xf32>
    %v764 = stablehlo.reduce(%v763 init: %v757) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v765 = stablehlo.divide %v764, %v758 : tensor<256xf32>
    %v766 = stablehlo.subtract %v760, %armeand3g1mu : tensor<256xf32>
    %v767 = stablehlo.multiply %v766, %v766 : tensor<256xf32>
    %v768 = stablehlo.add %v765, %v767 : tensor<256xf32>
    %arsumd3g1var = "stablehlo.all_reduce"(%v768) ({
    ^bb0(%arad3g1var: tensor<f32>, %arbd3g1var: tensor<f32>):
      %araddd3g1var = stablehlo.add %arad3g1var, %arbd3g1var : tensor<f32>
      stablehlo.return %araddd3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g1var = stablehlo.divide %arsumd3g1var, %arnd3g1var : tensor<256xf32>
    %v769 = stablehlo.concatenate %armeand3g1mu, %armeand3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v770 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v771 = stablehlo.slice %v769 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v772 = stablehlo.slice %v769 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v773 = stablehlo.broadcast_in_dim %v771, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v774 = stablehlo.broadcast_in_dim %v772, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v775 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v776 = stablehlo.add %v774, %v775 : tensor<64x256x14x14xf32>
    %v777 = stablehlo.rsqrt %v776 : tensor<64x256x14x14xf32>
    %v778 = stablehlo.subtract %v770, %v773 : tensor<64x256x14x14xf32>
    %v779 = stablehlo.multiply %v778, %v777 : tensor<64x256x14x14xf32>
    %v780 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v781 = stablehlo.broadcast_in_dim %d3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v782 = stablehlo.multiply %v779, %v780 : tensor<64x256x14x14xf32>
    %v783 = stablehlo.add %v782, %v781 : tensor<64x256x14x14xf32>
    %v784 = stablehlo.reshape %v783 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v785 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v786 = stablehlo.maximum %v784, %v785 : tensor<64x50176xf32>
    %v787 = stablehlo.reshape %v786 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v788 = stablehlo.convert %v787 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v789 = stablehlo.convert %d3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v790 = stablehlo.convolution(%v788, %v789)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v791 = stablehlo.convert %v790 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v792 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v793 = stablehlo.add %v791, %v792 : tensor<64x256x14x14xf32>
    %v794 = stablehlo.reshape %v793 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v795 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v796 = stablehlo.constant dense<0.0> : tensor<f32>
    %v797 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v798 = stablehlo.reduce(%v795 init: %v796) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v799 = stablehlo.divide %v798, %v797 : tensor<256xf32>
    %arsumd3g2mu = "stablehlo.all_reduce"(%v799) ({
    ^bb0(%arad3g2mu: tensor<f32>, %arbd3g2mu: tensor<f32>):
      %araddd3g2mu = stablehlo.add %arad3g2mu, %arbd3g2mu : tensor<f32>
      stablehlo.return %araddd3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g2mu = stablehlo.divide %arsumd3g2mu, %arnd3g2mu : tensor<256xf32>
    %v800 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v801 = stablehlo.constant dense<0.0> : tensor<f32>
    %v802 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v803 = stablehlo.reduce(%v800 init: %v801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v804 = stablehlo.divide %v803, %v802 : tensor<256xf32>
    %v805 = stablehlo.broadcast_in_dim %v804, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v806 = stablehlo.subtract %v800, %v805 : tensor<64x256x14x14xf32>
    %v807 = stablehlo.multiply %v806, %v806 : tensor<64x256x14x14xf32>
    %v808 = stablehlo.reduce(%v807 init: %v801) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v809 = stablehlo.divide %v808, %v802 : tensor<256xf32>
    %v810 = stablehlo.subtract %v804, %armeand3g2mu : tensor<256xf32>
    %v811 = stablehlo.multiply %v810, %v810 : tensor<256xf32>
    %v812 = stablehlo.add %v809, %v811 : tensor<256xf32>
    %arsumd3g2var = "stablehlo.all_reduce"(%v812) ({
    ^bb0(%arad3g2var: tensor<f32>, %arbd3g2var: tensor<f32>):
      %araddd3g2var = stablehlo.add %arad3g2var, %arbd3g2var : tensor<f32>
      stablehlo.return %araddd3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g2var = stablehlo.divide %arsumd3g2var, %arnd3g2var : tensor<256xf32>
    %v813 = stablehlo.concatenate %armeand3g2mu, %armeand3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v814 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v815 = stablehlo.slice %v813 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v816 = stablehlo.slice %v813 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v817 = stablehlo.broadcast_in_dim %v815, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v818 = stablehlo.broadcast_in_dim %v816, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v819 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v820 = stablehlo.add %v818, %v819 : tensor<64x256x14x14xf32>
    %v821 = stablehlo.rsqrt %v820 : tensor<64x256x14x14xf32>
    %v822 = stablehlo.subtract %v814, %v817 : tensor<64x256x14x14xf32>
    %v823 = stablehlo.multiply %v822, %v821 : tensor<64x256x14x14xf32>
    %v824 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v825 = stablehlo.broadcast_in_dim %d3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v826 = stablehlo.multiply %v823, %v824 : tensor<64x256x14x14xf32>
    %v827 = stablehlo.add %v826, %v825 : tensor<64x256x14x14xf32>
    %v828 = stablehlo.reshape %v827 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v829 = stablehlo.reshape %v742 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v830 = stablehlo.convert %v829 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v831 = stablehlo.convert %d3Wp : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xbf16>
    %v832 = stablehlo.convolution(%v830, %v831)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<256x128x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v833 = stablehlo.convert %v832 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v834 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v835 = stablehlo.add %v833, %v834 : tensor<64x256x14x14xf32>
    %v836 = stablehlo.reshape %v835 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v837 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v838 = stablehlo.constant dense<0.0> : tensor<f32>
    %v839 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v840 = stablehlo.reduce(%v837 init: %v838) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v841 = stablehlo.divide %v840, %v839 : tensor<256xf32>
    %arsumd3gpmu = "stablehlo.all_reduce"(%v841) ({
    ^bb0(%arad3gpmu: tensor<f32>, %arbd3gpmu: tensor<f32>):
      %araddd3gpmu = stablehlo.add %arad3gpmu, %arbd3gpmu : tensor<f32>
      stablehlo.return %araddd3gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gpmu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3gpmu = stablehlo.divide %arsumd3gpmu, %arnd3gpmu : tensor<256xf32>
    %v842 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v843 = stablehlo.constant dense<0.0> : tensor<f32>
    %v844 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v845 = stablehlo.reduce(%v842 init: %v843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v846 = stablehlo.divide %v845, %v844 : tensor<256xf32>
    %v847 = stablehlo.broadcast_in_dim %v846, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v848 = stablehlo.subtract %v842, %v847 : tensor<64x256x14x14xf32>
    %v849 = stablehlo.multiply %v848, %v848 : tensor<64x256x14x14xf32>
    %v850 = stablehlo.reduce(%v849 init: %v843) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v851 = stablehlo.divide %v850, %v844 : tensor<256xf32>
    %v852 = stablehlo.subtract %v846, %armeand3gpmu : tensor<256xf32>
    %v853 = stablehlo.multiply %v852, %v852 : tensor<256xf32>
    %v854 = stablehlo.add %v851, %v853 : tensor<256xf32>
    %arsumd3gpvar = "stablehlo.all_reduce"(%v854) ({
    ^bb0(%arad3gpvar: tensor<f32>, %arbd3gpvar: tensor<f32>):
      %araddd3gpvar = stablehlo.add %arad3gpvar, %arbd3gpvar : tensor<f32>
      stablehlo.return %araddd3gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gpvar = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3gpvar = stablehlo.divide %arsumd3gpvar, %arnd3gpvar : tensor<256xf32>
    %v855 = stablehlo.concatenate %armeand3gpmu, %armeand3gpvar, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v856 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v857 = stablehlo.slice %v855 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v858 = stablehlo.slice %v855 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v859 = stablehlo.broadcast_in_dim %v857, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v860 = stablehlo.broadcast_in_dim %v858, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v861 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v862 = stablehlo.add %v860, %v861 : tensor<64x256x14x14xf32>
    %v863 = stablehlo.rsqrt %v862 : tensor<64x256x14x14xf32>
    %v864 = stablehlo.subtract %v856, %v859 : tensor<64x256x14x14xf32>
    %v865 = stablehlo.multiply %v864, %v863 : tensor<64x256x14x14xf32>
    %v866 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v867 = stablehlo.broadcast_in_dim %d3btp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v868 = stablehlo.multiply %v865, %v866 : tensor<64x256x14x14xf32>
    %v869 = stablehlo.add %v868, %v867 : tensor<64x256x14x14xf32>
    %v870 = stablehlo.reshape %v869 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v871 = stablehlo.add %v828, %v870 : tensor<64x50176xf32>
    %v872 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v873 = stablehlo.maximum %v871, %v872 : tensor<64x50176xf32>
    %v874 = stablehlo.reshape %v873 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v875 = stablehlo.convert %v874 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v876 = stablehlo.convert %s3b0W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v877 = stablehlo.convolution(%v875, %v876)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v878 = stablehlo.convert %v877 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v879 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v880 = stablehlo.add %v878, %v879 : tensor<64x256x14x14xf32>
    %v881 = stablehlo.reshape %v880 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v882 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v883 = stablehlo.constant dense<0.0> : tensor<f32>
    %v884 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v885 = stablehlo.reduce(%v882 init: %v883) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v886 = stablehlo.divide %v885, %v884 : tensor<256xf32>
    %arsums3b0g1mu = "stablehlo.all_reduce"(%v886) ({
    ^bb0(%aras3b0g1mu: tensor<f32>, %arbs3b0g1mu: tensor<f32>):
      %aradds3b0g1mu = stablehlo.add %aras3b0g1mu, %arbs3b0g1mu : tensor<f32>
      stablehlo.return %aradds3b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1mu = stablehlo.divide %arsums3b0g1mu, %arns3b0g1mu : tensor<256xf32>
    %v887 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v888 = stablehlo.constant dense<0.0> : tensor<f32>
    %v889 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v890 = stablehlo.reduce(%v887 init: %v888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v891 = stablehlo.divide %v890, %v889 : tensor<256xf32>
    %v892 = stablehlo.broadcast_in_dim %v891, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v893 = stablehlo.subtract %v887, %v892 : tensor<64x256x14x14xf32>
    %v894 = stablehlo.multiply %v893, %v893 : tensor<64x256x14x14xf32>
    %v895 = stablehlo.reduce(%v894 init: %v888) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v896 = stablehlo.divide %v895, %v889 : tensor<256xf32>
    %v897 = stablehlo.subtract %v891, %armeans3b0g1mu : tensor<256xf32>
    %v898 = stablehlo.multiply %v897, %v897 : tensor<256xf32>
    %v899 = stablehlo.add %v896, %v898 : tensor<256xf32>
    %arsums3b0g1var = "stablehlo.all_reduce"(%v899) ({
    ^bb0(%aras3b0g1var: tensor<f32>, %arbs3b0g1var: tensor<f32>):
      %aradds3b0g1var = stablehlo.add %aras3b0g1var, %arbs3b0g1var : tensor<f32>
      stablehlo.return %aradds3b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1var = stablehlo.divide %arsums3b0g1var, %arns3b0g1var : tensor<256xf32>
    %v900 = stablehlo.concatenate %armeans3b0g1mu, %armeans3b0g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v901 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v902 = stablehlo.slice %v900 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v903 = stablehlo.slice %v900 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v904 = stablehlo.broadcast_in_dim %v902, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v905 = stablehlo.broadcast_in_dim %v903, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v906 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v907 = stablehlo.add %v905, %v906 : tensor<64x256x14x14xf32>
    %v908 = stablehlo.rsqrt %v907 : tensor<64x256x14x14xf32>
    %v909 = stablehlo.subtract %v901, %v904 : tensor<64x256x14x14xf32>
    %v910 = stablehlo.multiply %v909, %v908 : tensor<64x256x14x14xf32>
    %v911 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v912 = stablehlo.broadcast_in_dim %s3b0bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v913 = stablehlo.multiply %v910, %v911 : tensor<64x256x14x14xf32>
    %v914 = stablehlo.add %v913, %v912 : tensor<64x256x14x14xf32>
    %v915 = stablehlo.reshape %v914 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v916 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v917 = stablehlo.maximum %v915, %v916 : tensor<64x50176xf32>
    %v918 = stablehlo.reshape %v917 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v919 = stablehlo.convert %v918 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v920 = stablehlo.convert %s3b0W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v921 = stablehlo.convolution(%v919, %v920)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v922 = stablehlo.convert %v921 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v923 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v924 = stablehlo.add %v922, %v923 : tensor<64x256x14x14xf32>
    %v925 = stablehlo.reshape %v924 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v926 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v928 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v929 = stablehlo.reduce(%v926 init: %v927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v930 = stablehlo.divide %v929, %v928 : tensor<256xf32>
    %arsums3b0g2mu = "stablehlo.all_reduce"(%v930) ({
    ^bb0(%aras3b0g2mu: tensor<f32>, %arbs3b0g2mu: tensor<f32>):
      %aradds3b0g2mu = stablehlo.add %aras3b0g2mu, %arbs3b0g2mu : tensor<f32>
      stablehlo.return %aradds3b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2mu = stablehlo.divide %arsums3b0g2mu, %arns3b0g2mu : tensor<256xf32>
    %v931 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v932 = stablehlo.constant dense<0.0> : tensor<f32>
    %v933 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v934 = stablehlo.reduce(%v931 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v935 = stablehlo.divide %v934, %v933 : tensor<256xf32>
    %v936 = stablehlo.broadcast_in_dim %v935, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v937 = stablehlo.subtract %v931, %v936 : tensor<64x256x14x14xf32>
    %v938 = stablehlo.multiply %v937, %v937 : tensor<64x256x14x14xf32>
    %v939 = stablehlo.reduce(%v938 init: %v932) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v940 = stablehlo.divide %v939, %v933 : tensor<256xf32>
    %v941 = stablehlo.subtract %v935, %armeans3b0g2mu : tensor<256xf32>
    %v942 = stablehlo.multiply %v941, %v941 : tensor<256xf32>
    %v943 = stablehlo.add %v940, %v942 : tensor<256xf32>
    %arsums3b0g2var = "stablehlo.all_reduce"(%v943) ({
    ^bb0(%aras3b0g2var: tensor<f32>, %arbs3b0g2var: tensor<f32>):
      %aradds3b0g2var = stablehlo.add %aras3b0g2var, %arbs3b0g2var : tensor<f32>
      stablehlo.return %aradds3b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2var = stablehlo.divide %arsums3b0g2var, %arns3b0g2var : tensor<256xf32>
    %v944 = stablehlo.concatenate %armeans3b0g2mu, %armeans3b0g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v945 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v946 = stablehlo.slice %v944 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v947 = stablehlo.slice %v944 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v948 = stablehlo.broadcast_in_dim %v946, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v949 = stablehlo.broadcast_in_dim %v947, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v950 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v951 = stablehlo.add %v949, %v950 : tensor<64x256x14x14xf32>
    %v952 = stablehlo.rsqrt %v951 : tensor<64x256x14x14xf32>
    %v953 = stablehlo.subtract %v945, %v948 : tensor<64x256x14x14xf32>
    %v954 = stablehlo.multiply %v953, %v952 : tensor<64x256x14x14xf32>
    %v955 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v956 = stablehlo.broadcast_in_dim %s3b0bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v957 = stablehlo.multiply %v954, %v955 : tensor<64x256x14x14xf32>
    %v958 = stablehlo.add %v957, %v956 : tensor<64x256x14x14xf32>
    %v959 = stablehlo.reshape %v958 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v960 = stablehlo.reshape %v959 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v961 = stablehlo.reshape %v873 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v962 = stablehlo.add %v960, %v961 : tensor<64x256x14x14xf32>
    %v963 = stablehlo.reshape %v962 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v964 = stablehlo.reshape %v963 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v965 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v966 = stablehlo.maximum %v964, %v965 : tensor<64x256x14x14xf32>
    %v967 = stablehlo.reshape %v966 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v968 = stablehlo.reshape %v967 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v969 = stablehlo.convert %v968 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v970 = stablehlo.convert %s3b1W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v971 = stablehlo.convolution(%v969, %v970)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v972 = stablehlo.convert %v971 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v973 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v974 = stablehlo.add %v972, %v973 : tensor<64x256x14x14xf32>
    %v975 = stablehlo.reshape %v974 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v976 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v977 = stablehlo.constant dense<0.0> : tensor<f32>
    %v978 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v979 = stablehlo.reduce(%v976 init: %v977) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v980 = stablehlo.divide %v979, %v978 : tensor<256xf32>
    %arsums3b1g1mu = "stablehlo.all_reduce"(%v980) ({
    ^bb0(%aras3b1g1mu: tensor<f32>, %arbs3b1g1mu: tensor<f32>):
      %aradds3b1g1mu = stablehlo.add %aras3b1g1mu, %arbs3b1g1mu : tensor<f32>
      stablehlo.return %aradds3b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1mu = stablehlo.divide %arsums3b1g1mu, %arns3b1g1mu : tensor<256xf32>
    %v981 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v982 = stablehlo.constant dense<0.0> : tensor<f32>
    %v983 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v984 = stablehlo.reduce(%v981 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v985 = stablehlo.divide %v984, %v983 : tensor<256xf32>
    %v986 = stablehlo.broadcast_in_dim %v985, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v987 = stablehlo.subtract %v981, %v986 : tensor<64x256x14x14xf32>
    %v988 = stablehlo.multiply %v987, %v987 : tensor<64x256x14x14xf32>
    %v989 = stablehlo.reduce(%v988 init: %v982) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v990 = stablehlo.divide %v989, %v983 : tensor<256xf32>
    %v991 = stablehlo.subtract %v985, %armeans3b1g1mu : tensor<256xf32>
    %v992 = stablehlo.multiply %v991, %v991 : tensor<256xf32>
    %v993 = stablehlo.add %v990, %v992 : tensor<256xf32>
    %arsums3b1g1var = "stablehlo.all_reduce"(%v993) ({
    ^bb0(%aras3b1g1var: tensor<f32>, %arbs3b1g1var: tensor<f32>):
      %aradds3b1g1var = stablehlo.add %aras3b1g1var, %arbs3b1g1var : tensor<f32>
      stablehlo.return %aradds3b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1var = stablehlo.divide %arsums3b1g1var, %arns3b1g1var : tensor<256xf32>
    %v994 = stablehlo.concatenate %armeans3b1g1mu, %armeans3b1g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v995 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v996 = stablehlo.slice %v994 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v997 = stablehlo.slice %v994 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v998 = stablehlo.broadcast_in_dim %v996, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v999 = stablehlo.broadcast_in_dim %v997, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1000 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1001 = stablehlo.add %v999, %v1000 : tensor<64x256x14x14xf32>
    %v1002 = stablehlo.rsqrt %v1001 : tensor<64x256x14x14xf32>
    %v1003 = stablehlo.subtract %v995, %v998 : tensor<64x256x14x14xf32>
    %v1004 = stablehlo.multiply %v1003, %v1002 : tensor<64x256x14x14xf32>
    %v1005 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1006 = stablehlo.broadcast_in_dim %s3b1bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1007 = stablehlo.multiply %v1004, %v1005 : tensor<64x256x14x14xf32>
    %v1008 = stablehlo.add %v1007, %v1006 : tensor<64x256x14x14xf32>
    %v1009 = stablehlo.reshape %v1008 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1010 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1011 = stablehlo.maximum %v1009, %v1010 : tensor<64x50176xf32>
    %v1012 = stablehlo.reshape %v1011 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1013 = stablehlo.convert %v1012 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1014 = stablehlo.convert %s3b1W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1015 = stablehlo.convolution(%v1013, %v1014)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1016 = stablehlo.convert %v1015 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1017 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1018 = stablehlo.add %v1016, %v1017 : tensor<64x256x14x14xf32>
    %v1019 = stablehlo.reshape %v1018 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1020 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1021 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1022 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1023 = stablehlo.reduce(%v1020 init: %v1021) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1024 = stablehlo.divide %v1023, %v1022 : tensor<256xf32>
    %arsums3b1g2mu = "stablehlo.all_reduce"(%v1024) ({
    ^bb0(%aras3b1g2mu: tensor<f32>, %arbs3b1g2mu: tensor<f32>):
      %aradds3b1g2mu = stablehlo.add %aras3b1g2mu, %arbs3b1g2mu : tensor<f32>
      stablehlo.return %aradds3b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2mu = stablehlo.divide %arsums3b1g2mu, %arns3b1g2mu : tensor<256xf32>
    %v1025 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1026 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1027 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1028 = stablehlo.reduce(%v1025 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1029 = stablehlo.divide %v1028, %v1027 : tensor<256xf32>
    %v1030 = stablehlo.broadcast_in_dim %v1029, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1031 = stablehlo.subtract %v1025, %v1030 : tensor<64x256x14x14xf32>
    %v1032 = stablehlo.multiply %v1031, %v1031 : tensor<64x256x14x14xf32>
    %v1033 = stablehlo.reduce(%v1032 init: %v1026) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1034 = stablehlo.divide %v1033, %v1027 : tensor<256xf32>
    %v1035 = stablehlo.subtract %v1029, %armeans3b1g2mu : tensor<256xf32>
    %v1036 = stablehlo.multiply %v1035, %v1035 : tensor<256xf32>
    %v1037 = stablehlo.add %v1034, %v1036 : tensor<256xf32>
    %arsums3b1g2var = "stablehlo.all_reduce"(%v1037) ({
    ^bb0(%aras3b1g2var: tensor<f32>, %arbs3b1g2var: tensor<f32>):
      %aradds3b1g2var = stablehlo.add %aras3b1g2var, %arbs3b1g2var : tensor<f32>
      stablehlo.return %aradds3b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2var = stablehlo.divide %arsums3b1g2var, %arns3b1g2var : tensor<256xf32>
    %v1038 = stablehlo.concatenate %armeans3b1g2mu, %armeans3b1g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1039 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1040 = stablehlo.slice %v1038 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1041 = stablehlo.slice %v1038 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1042 = stablehlo.broadcast_in_dim %v1040, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1043 = stablehlo.broadcast_in_dim %v1041, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1044 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1045 = stablehlo.add %v1043, %v1044 : tensor<64x256x14x14xf32>
    %v1046 = stablehlo.rsqrt %v1045 : tensor<64x256x14x14xf32>
    %v1047 = stablehlo.subtract %v1039, %v1042 : tensor<64x256x14x14xf32>
    %v1048 = stablehlo.multiply %v1047, %v1046 : tensor<64x256x14x14xf32>
    %v1049 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1050 = stablehlo.broadcast_in_dim %s3b1bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1051 = stablehlo.multiply %v1048, %v1049 : tensor<64x256x14x14xf32>
    %v1052 = stablehlo.add %v1051, %v1050 : tensor<64x256x14x14xf32>
    %v1053 = stablehlo.reshape %v1052 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1054 = stablehlo.reshape %v1053 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1055 = stablehlo.reshape %v967 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1056 = stablehlo.add %v1054, %v1055 : tensor<64x256x14x14xf32>
    %v1057 = stablehlo.reshape %v1056 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1058 = stablehlo.reshape %v1057 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1059 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1060 = stablehlo.maximum %v1058, %v1059 : tensor<64x256x14x14xf32>
    %v1061 = stablehlo.reshape %v1060 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1062 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1063 = stablehlo.convert %v1062 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1064 = stablehlo.convert %s3b2W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1065 = stablehlo.convolution(%v1063, %v1064)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1066 = stablehlo.convert %v1065 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1067 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1068 = stablehlo.add %v1066, %v1067 : tensor<64x256x14x14xf32>
    %v1069 = stablehlo.reshape %v1068 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1070 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1071 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1072 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1073 = stablehlo.reduce(%v1070 init: %v1071) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1074 = stablehlo.divide %v1073, %v1072 : tensor<256xf32>
    %arsums3b2g1mu = "stablehlo.all_reduce"(%v1074) ({
    ^bb0(%aras3b2g1mu: tensor<f32>, %arbs3b2g1mu: tensor<f32>):
      %aradds3b2g1mu = stablehlo.add %aras3b2g1mu, %arbs3b2g1mu : tensor<f32>
      stablehlo.return %aradds3b2g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1mu = stablehlo.divide %arsums3b2g1mu, %arns3b2g1mu : tensor<256xf32>
    %v1075 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1076 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1077 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1078 = stablehlo.reduce(%v1075 init: %v1076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1079 = stablehlo.divide %v1078, %v1077 : tensor<256xf32>
    %v1080 = stablehlo.broadcast_in_dim %v1079, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1081 = stablehlo.subtract %v1075, %v1080 : tensor<64x256x14x14xf32>
    %v1082 = stablehlo.multiply %v1081, %v1081 : tensor<64x256x14x14xf32>
    %v1083 = stablehlo.reduce(%v1082 init: %v1076) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1084 = stablehlo.divide %v1083, %v1077 : tensor<256xf32>
    %v1085 = stablehlo.subtract %v1079, %armeans3b2g1mu : tensor<256xf32>
    %v1086 = stablehlo.multiply %v1085, %v1085 : tensor<256xf32>
    %v1087 = stablehlo.add %v1084, %v1086 : tensor<256xf32>
    %arsums3b2g1var = "stablehlo.all_reduce"(%v1087) ({
    ^bb0(%aras3b2g1var: tensor<f32>, %arbs3b2g1var: tensor<f32>):
      %aradds3b2g1var = stablehlo.add %aras3b2g1var, %arbs3b2g1var : tensor<f32>
      stablehlo.return %aradds3b2g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1var = stablehlo.divide %arsums3b2g1var, %arns3b2g1var : tensor<256xf32>
    %v1088 = stablehlo.concatenate %armeans3b2g1mu, %armeans3b2g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1089 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1090 = stablehlo.slice %v1088 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1091 = stablehlo.slice %v1088 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1092 = stablehlo.broadcast_in_dim %v1090, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1093 = stablehlo.broadcast_in_dim %v1091, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1094 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1095 = stablehlo.add %v1093, %v1094 : tensor<64x256x14x14xf32>
    %v1096 = stablehlo.rsqrt %v1095 : tensor<64x256x14x14xf32>
    %v1097 = stablehlo.subtract %v1089, %v1092 : tensor<64x256x14x14xf32>
    %v1098 = stablehlo.multiply %v1097, %v1096 : tensor<64x256x14x14xf32>
    %v1099 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1100 = stablehlo.broadcast_in_dim %s3b2bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1101 = stablehlo.multiply %v1098, %v1099 : tensor<64x256x14x14xf32>
    %v1102 = stablehlo.add %v1101, %v1100 : tensor<64x256x14x14xf32>
    %v1103 = stablehlo.reshape %v1102 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1104 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1105 = stablehlo.maximum %v1103, %v1104 : tensor<64x50176xf32>
    %v1106 = stablehlo.reshape %v1105 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1107 = stablehlo.convert %v1106 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1108 = stablehlo.convert %s3b2W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1109 = stablehlo.convolution(%v1107, %v1108)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1110 = stablehlo.convert %v1109 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1111 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1112 = stablehlo.add %v1110, %v1111 : tensor<64x256x14x14xf32>
    %v1113 = stablehlo.reshape %v1112 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1114 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1115 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1116 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1117 = stablehlo.reduce(%v1114 init: %v1115) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1118 = stablehlo.divide %v1117, %v1116 : tensor<256xf32>
    %arsums3b2g2mu = "stablehlo.all_reduce"(%v1118) ({
    ^bb0(%aras3b2g2mu: tensor<f32>, %arbs3b2g2mu: tensor<f32>):
      %aradds3b2g2mu = stablehlo.add %aras3b2g2mu, %arbs3b2g2mu : tensor<f32>
      stablehlo.return %aradds3b2g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2mu = stablehlo.divide %arsums3b2g2mu, %arns3b2g2mu : tensor<256xf32>
    %v1119 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1120 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1121 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1122 = stablehlo.reduce(%v1119 init: %v1120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1123 = stablehlo.divide %v1122, %v1121 : tensor<256xf32>
    %v1124 = stablehlo.broadcast_in_dim %v1123, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1125 = stablehlo.subtract %v1119, %v1124 : tensor<64x256x14x14xf32>
    %v1126 = stablehlo.multiply %v1125, %v1125 : tensor<64x256x14x14xf32>
    %v1127 = stablehlo.reduce(%v1126 init: %v1120) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1128 = stablehlo.divide %v1127, %v1121 : tensor<256xf32>
    %v1129 = stablehlo.subtract %v1123, %armeans3b2g2mu : tensor<256xf32>
    %v1130 = stablehlo.multiply %v1129, %v1129 : tensor<256xf32>
    %v1131 = stablehlo.add %v1128, %v1130 : tensor<256xf32>
    %arsums3b2g2var = "stablehlo.all_reduce"(%v1131) ({
    ^bb0(%aras3b2g2var: tensor<f32>, %arbs3b2g2var: tensor<f32>):
      %aradds3b2g2var = stablehlo.add %aras3b2g2var, %arbs3b2g2var : tensor<f32>
      stablehlo.return %aradds3b2g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2var = stablehlo.divide %arsums3b2g2var, %arns3b2g2var : tensor<256xf32>
    %v1132 = stablehlo.concatenate %armeans3b2g2mu, %armeans3b2g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1133 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1134 = stablehlo.slice %v1132 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1135 = stablehlo.slice %v1132 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1136 = stablehlo.broadcast_in_dim %v1134, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1137 = stablehlo.broadcast_in_dim %v1135, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1138 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1139 = stablehlo.add %v1137, %v1138 : tensor<64x256x14x14xf32>
    %v1140 = stablehlo.rsqrt %v1139 : tensor<64x256x14x14xf32>
    %v1141 = stablehlo.subtract %v1133, %v1136 : tensor<64x256x14x14xf32>
    %v1142 = stablehlo.multiply %v1141, %v1140 : tensor<64x256x14x14xf32>
    %v1143 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1144 = stablehlo.broadcast_in_dim %s3b2bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1145 = stablehlo.multiply %v1142, %v1143 : tensor<64x256x14x14xf32>
    %v1146 = stablehlo.add %v1145, %v1144 : tensor<64x256x14x14xf32>
    %v1147 = stablehlo.reshape %v1146 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1148 = stablehlo.reshape %v1147 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1149 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1150 = stablehlo.add %v1148, %v1149 : tensor<64x256x14x14xf32>
    %v1151 = stablehlo.reshape %v1150 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1152 = stablehlo.reshape %v1151 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1153 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1154 = stablehlo.maximum %v1152, %v1153 : tensor<64x256x14x14xf32>
    %v1155 = stablehlo.reshape %v1154 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1156 = stablehlo.reshape %v1155 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1157 = stablehlo.convert %v1156 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1158 = stablehlo.convert %s3b3W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1159 = stablehlo.convolution(%v1157, %v1158)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1160 = stablehlo.convert %v1159 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1161 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1162 = stablehlo.add %v1160, %v1161 : tensor<64x256x14x14xf32>
    %v1163 = stablehlo.reshape %v1162 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1164 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1165 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1166 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1167 = stablehlo.reduce(%v1164 init: %v1165) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1168 = stablehlo.divide %v1167, %v1166 : tensor<256xf32>
    %arsums3b3g1mu = "stablehlo.all_reduce"(%v1168) ({
    ^bb0(%aras3b3g1mu: tensor<f32>, %arbs3b3g1mu: tensor<f32>):
      %aradds3b3g1mu = stablehlo.add %aras3b3g1mu, %arbs3b3g1mu : tensor<f32>
      stablehlo.return %aradds3b3g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1mu = stablehlo.divide %arsums3b3g1mu, %arns3b3g1mu : tensor<256xf32>
    %v1169 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1170 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1171 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1172 = stablehlo.reduce(%v1169 init: %v1170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1173 = stablehlo.divide %v1172, %v1171 : tensor<256xf32>
    %v1174 = stablehlo.broadcast_in_dim %v1173, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1175 = stablehlo.subtract %v1169, %v1174 : tensor<64x256x14x14xf32>
    %v1176 = stablehlo.multiply %v1175, %v1175 : tensor<64x256x14x14xf32>
    %v1177 = stablehlo.reduce(%v1176 init: %v1170) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1178 = stablehlo.divide %v1177, %v1171 : tensor<256xf32>
    %v1179 = stablehlo.subtract %v1173, %armeans3b3g1mu : tensor<256xf32>
    %v1180 = stablehlo.multiply %v1179, %v1179 : tensor<256xf32>
    %v1181 = stablehlo.add %v1178, %v1180 : tensor<256xf32>
    %arsums3b3g1var = "stablehlo.all_reduce"(%v1181) ({
    ^bb0(%aras3b3g1var: tensor<f32>, %arbs3b3g1var: tensor<f32>):
      %aradds3b3g1var = stablehlo.add %aras3b3g1var, %arbs3b3g1var : tensor<f32>
      stablehlo.return %aradds3b3g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1var = stablehlo.divide %arsums3b3g1var, %arns3b3g1var : tensor<256xf32>
    %v1182 = stablehlo.concatenate %armeans3b3g1mu, %armeans3b3g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1183 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1184 = stablehlo.slice %v1182 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1185 = stablehlo.slice %v1182 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1186 = stablehlo.broadcast_in_dim %v1184, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1187 = stablehlo.broadcast_in_dim %v1185, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1188 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1189 = stablehlo.add %v1187, %v1188 : tensor<64x256x14x14xf32>
    %v1190 = stablehlo.rsqrt %v1189 : tensor<64x256x14x14xf32>
    %v1191 = stablehlo.subtract %v1183, %v1186 : tensor<64x256x14x14xf32>
    %v1192 = stablehlo.multiply %v1191, %v1190 : tensor<64x256x14x14xf32>
    %v1193 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1194 = stablehlo.broadcast_in_dim %s3b3bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1195 = stablehlo.multiply %v1192, %v1193 : tensor<64x256x14x14xf32>
    %v1196 = stablehlo.add %v1195, %v1194 : tensor<64x256x14x14xf32>
    %v1197 = stablehlo.reshape %v1196 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1198 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1199 = stablehlo.maximum %v1197, %v1198 : tensor<64x50176xf32>
    %v1200 = stablehlo.reshape %v1199 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1201 = stablehlo.convert %v1200 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1202 = stablehlo.convert %s3b3W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1203 = stablehlo.convolution(%v1201, %v1202)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1204 = stablehlo.convert %v1203 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1205 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1206 = stablehlo.add %v1204, %v1205 : tensor<64x256x14x14xf32>
    %v1207 = stablehlo.reshape %v1206 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1208 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1210 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1211 = stablehlo.reduce(%v1208 init: %v1209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1212 = stablehlo.divide %v1211, %v1210 : tensor<256xf32>
    %arsums3b3g2mu = "stablehlo.all_reduce"(%v1212) ({
    ^bb0(%aras3b3g2mu: tensor<f32>, %arbs3b3g2mu: tensor<f32>):
      %aradds3b3g2mu = stablehlo.add %aras3b3g2mu, %arbs3b3g2mu : tensor<f32>
      stablehlo.return %aradds3b3g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2mu = stablehlo.divide %arsums3b3g2mu, %arns3b3g2mu : tensor<256xf32>
    %v1213 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1214 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1215 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1216 = stablehlo.reduce(%v1213 init: %v1214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1217 = stablehlo.divide %v1216, %v1215 : tensor<256xf32>
    %v1218 = stablehlo.broadcast_in_dim %v1217, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1219 = stablehlo.subtract %v1213, %v1218 : tensor<64x256x14x14xf32>
    %v1220 = stablehlo.multiply %v1219, %v1219 : tensor<64x256x14x14xf32>
    %v1221 = stablehlo.reduce(%v1220 init: %v1214) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1222 = stablehlo.divide %v1221, %v1215 : tensor<256xf32>
    %v1223 = stablehlo.subtract %v1217, %armeans3b3g2mu : tensor<256xf32>
    %v1224 = stablehlo.multiply %v1223, %v1223 : tensor<256xf32>
    %v1225 = stablehlo.add %v1222, %v1224 : tensor<256xf32>
    %arsums3b3g2var = "stablehlo.all_reduce"(%v1225) ({
    ^bb0(%aras3b3g2var: tensor<f32>, %arbs3b3g2var: tensor<f32>):
      %aradds3b3g2var = stablehlo.add %aras3b3g2var, %arbs3b3g2var : tensor<f32>
      stablehlo.return %aradds3b3g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2var = stablehlo.divide %arsums3b3g2var, %arns3b3g2var : tensor<256xf32>
    %v1226 = stablehlo.concatenate %armeans3b3g2mu, %armeans3b3g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1227 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1228 = stablehlo.slice %v1226 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1229 = stablehlo.slice %v1226 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1230 = stablehlo.broadcast_in_dim %v1228, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1231 = stablehlo.broadcast_in_dim %v1229, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1232 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1233 = stablehlo.add %v1231, %v1232 : tensor<64x256x14x14xf32>
    %v1234 = stablehlo.rsqrt %v1233 : tensor<64x256x14x14xf32>
    %v1235 = stablehlo.subtract %v1227, %v1230 : tensor<64x256x14x14xf32>
    %v1236 = stablehlo.multiply %v1235, %v1234 : tensor<64x256x14x14xf32>
    %v1237 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1238 = stablehlo.broadcast_in_dim %s3b3bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1239 = stablehlo.multiply %v1236, %v1237 : tensor<64x256x14x14xf32>
    %v1240 = stablehlo.add %v1239, %v1238 : tensor<64x256x14x14xf32>
    %v1241 = stablehlo.reshape %v1240 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1242 = stablehlo.reshape %v1241 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1243 = stablehlo.reshape %v1155 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1244 = stablehlo.add %v1242, %v1243 : tensor<64x256x14x14xf32>
    %v1245 = stablehlo.reshape %v1244 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1246 = stablehlo.reshape %v1245 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1247 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1248 = stablehlo.maximum %v1246, %v1247 : tensor<64x256x14x14xf32>
    %v1249 = stablehlo.reshape %v1248 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1250 = stablehlo.reshape %v1249 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1251 = stablehlo.convert %v1250 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1252 = stablehlo.convert %s3b4W1 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1253 = stablehlo.convolution(%v1251, %v1252)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1254 = stablehlo.convert %v1253 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1255 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1256 = stablehlo.add %v1254, %v1255 : tensor<64x256x14x14xf32>
    %v1257 = stablehlo.reshape %v1256 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1258 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1259 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1260 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1261 = stablehlo.reduce(%v1258 init: %v1259) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1262 = stablehlo.divide %v1261, %v1260 : tensor<256xf32>
    %arsums3b4g1mu = "stablehlo.all_reduce"(%v1262) ({
    ^bb0(%aras3b4g1mu: tensor<f32>, %arbs3b4g1mu: tensor<f32>):
      %aradds3b4g1mu = stablehlo.add %aras3b4g1mu, %arbs3b4g1mu : tensor<f32>
      stablehlo.return %aradds3b4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1mu = stablehlo.divide %arsums3b4g1mu, %arns3b4g1mu : tensor<256xf32>
    %v1263 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1264 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1265 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1266 = stablehlo.reduce(%v1263 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1267 = stablehlo.divide %v1266, %v1265 : tensor<256xf32>
    %v1268 = stablehlo.broadcast_in_dim %v1267, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1269 = stablehlo.subtract %v1263, %v1268 : tensor<64x256x14x14xf32>
    %v1270 = stablehlo.multiply %v1269, %v1269 : tensor<64x256x14x14xf32>
    %v1271 = stablehlo.reduce(%v1270 init: %v1264) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1272 = stablehlo.divide %v1271, %v1265 : tensor<256xf32>
    %v1273 = stablehlo.subtract %v1267, %armeans3b4g1mu : tensor<256xf32>
    %v1274 = stablehlo.multiply %v1273, %v1273 : tensor<256xf32>
    %v1275 = stablehlo.add %v1272, %v1274 : tensor<256xf32>
    %arsums3b4g1var = "stablehlo.all_reduce"(%v1275) ({
    ^bb0(%aras3b4g1var: tensor<f32>, %arbs3b4g1var: tensor<f32>):
      %aradds3b4g1var = stablehlo.add %aras3b4g1var, %arbs3b4g1var : tensor<f32>
      stablehlo.return %aradds3b4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1var = stablehlo.divide %arsums3b4g1var, %arns3b4g1var : tensor<256xf32>
    %v1276 = stablehlo.concatenate %armeans3b4g1mu, %armeans3b4g1var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1277 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1278 = stablehlo.slice %v1276 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1279 = stablehlo.slice %v1276 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1280 = stablehlo.broadcast_in_dim %v1278, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1281 = stablehlo.broadcast_in_dim %v1279, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1282 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1283 = stablehlo.add %v1281, %v1282 : tensor<64x256x14x14xf32>
    %v1284 = stablehlo.rsqrt %v1283 : tensor<64x256x14x14xf32>
    %v1285 = stablehlo.subtract %v1277, %v1280 : tensor<64x256x14x14xf32>
    %v1286 = stablehlo.multiply %v1285, %v1284 : tensor<64x256x14x14xf32>
    %v1287 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1288 = stablehlo.broadcast_in_dim %s3b4bt1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1289 = stablehlo.multiply %v1286, %v1287 : tensor<64x256x14x14xf32>
    %v1290 = stablehlo.add %v1289, %v1288 : tensor<64x256x14x14xf32>
    %v1291 = stablehlo.reshape %v1290 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1292 = stablehlo.constant dense<0.0> : tensor<64x50176xf32>
    %v1293 = stablehlo.maximum %v1291, %v1292 : tensor<64x50176xf32>
    %v1294 = stablehlo.reshape %v1293 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1295 = stablehlo.convert %v1294 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1296 = stablehlo.convert %s3b4W2 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v1297 = stablehlo.convolution(%v1295, %v1296)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v1298 = stablehlo.convert %v1297 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v1299 = stablehlo.broadcast_in_dim %zb256, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1300 = stablehlo.add %v1298, %v1299 : tensor<64x256x14x14xf32>
    %v1301 = stablehlo.reshape %v1300 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1302 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1303 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1304 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1305 = stablehlo.reduce(%v1302 init: %v1303) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1306 = stablehlo.divide %v1305, %v1304 : tensor<256xf32>
    %arsums3b4g2mu = "stablehlo.all_reduce"(%v1306) ({
    ^bb0(%aras3b4g2mu: tensor<f32>, %arbs3b4g2mu: tensor<f32>):
      %aradds3b4g2mu = stablehlo.add %aras3b4g2mu, %arbs3b4g2mu : tensor<f32>
      stablehlo.return %aradds3b4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2mu = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2mu = stablehlo.divide %arsums3b4g2mu, %arns3b4g2mu : tensor<256xf32>
    %v1307 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1308 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1309 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v1310 = stablehlo.reduce(%v1307 init: %v1308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1311 = stablehlo.divide %v1310, %v1309 : tensor<256xf32>
    %v1312 = stablehlo.broadcast_in_dim %v1311, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1313 = stablehlo.subtract %v1307, %v1312 : tensor<64x256x14x14xf32>
    %v1314 = stablehlo.multiply %v1313, %v1313 : tensor<64x256x14x14xf32>
    %v1315 = stablehlo.reduce(%v1314 init: %v1308) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v1316 = stablehlo.divide %v1315, %v1309 : tensor<256xf32>
    %v1317 = stablehlo.subtract %v1311, %armeans3b4g2mu : tensor<256xf32>
    %v1318 = stablehlo.multiply %v1317, %v1317 : tensor<256xf32>
    %v1319 = stablehlo.add %v1316, %v1318 : tensor<256xf32>
    %arsums3b4g2var = "stablehlo.all_reduce"(%v1319) ({
    ^bb0(%aras3b4g2var: tensor<f32>, %arbs3b4g2var: tensor<f32>):
      %aradds3b4g2var = stablehlo.add %aras3b4g2var, %arbs3b4g2var : tensor<f32>
      stablehlo.return %aradds3b4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2var = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2var = stablehlo.divide %arsums3b4g2var, %arns3b4g2var : tensor<256xf32>
    %v1320 = stablehlo.concatenate %armeans3b4g2mu, %armeans3b4g2var, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v1321 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1322 = stablehlo.slice %v1320 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v1323 = stablehlo.slice %v1320 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v1324 = stablehlo.broadcast_in_dim %v1322, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1325 = stablehlo.broadcast_in_dim %v1323, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1326 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v1327 = stablehlo.add %v1325, %v1326 : tensor<64x256x14x14xf32>
    %v1328 = stablehlo.rsqrt %v1327 : tensor<64x256x14x14xf32>
    %v1329 = stablehlo.subtract %v1321, %v1324 : tensor<64x256x14x14xf32>
    %v1330 = stablehlo.multiply %v1329, %v1328 : tensor<64x256x14x14xf32>
    %v1331 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1332 = stablehlo.broadcast_in_dim %s3b4bt2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v1333 = stablehlo.multiply %v1330, %v1331 : tensor<64x256x14x14xf32>
    %v1334 = stablehlo.add %v1333, %v1332 : tensor<64x256x14x14xf32>
    %v1335 = stablehlo.reshape %v1334 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1336 = stablehlo.reshape %v1335 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1337 = stablehlo.reshape %v1249 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1338 = stablehlo.add %v1336, %v1337 : tensor<64x256x14x14xf32>
    %v1339 = stablehlo.reshape %v1338 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1340 = stablehlo.reshape %v1339 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1341 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v1342 = stablehlo.maximum %v1340, %v1341 : tensor<64x256x14x14xf32>
    %v1343 = stablehlo.reshape %v1342 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v1344 = stablehlo.reshape %v1343 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1345 = stablehlo.convert %v1344 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1346 = stablehlo.convert %d4W1 : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xbf16>
    %v1347 = stablehlo.convolution(%v1345, %v1346)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<512x256x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1348 = stablehlo.convert %v1347 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1349 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1350 = stablehlo.add %v1348, %v1349 : tensor<64x512x7x7xf32>
    %v1351 = stablehlo.reshape %v1350 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1352 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1353 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1354 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1355 = stablehlo.reduce(%v1352 init: %v1353) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1356 = stablehlo.divide %v1355, %v1354 : tensor<512xf32>
    %arsumd4g1mu = "stablehlo.all_reduce"(%v1356) ({
    ^bb0(%arad4g1mu: tensor<f32>, %arbd4g1mu: tensor<f32>):
      %araddd4g1mu = stablehlo.add %arad4g1mu, %arbd4g1mu : tensor<f32>
      stablehlo.return %araddd4g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g1mu = stablehlo.divide %arsumd4g1mu, %arnd4g1mu : tensor<512xf32>
    %v1357 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1358 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1359 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1360 = stablehlo.reduce(%v1357 init: %v1358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1361 = stablehlo.divide %v1360, %v1359 : tensor<512xf32>
    %v1362 = stablehlo.broadcast_in_dim %v1361, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1363 = stablehlo.subtract %v1357, %v1362 : tensor<64x512x7x7xf32>
    %v1364 = stablehlo.multiply %v1363, %v1363 : tensor<64x512x7x7xf32>
    %v1365 = stablehlo.reduce(%v1364 init: %v1358) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1366 = stablehlo.divide %v1365, %v1359 : tensor<512xf32>
    %v1367 = stablehlo.subtract %v1361, %armeand4g1mu : tensor<512xf32>
    %v1368 = stablehlo.multiply %v1367, %v1367 : tensor<512xf32>
    %v1369 = stablehlo.add %v1366, %v1368 : tensor<512xf32>
    %arsumd4g1var = "stablehlo.all_reduce"(%v1369) ({
    ^bb0(%arad4g1var: tensor<f32>, %arbd4g1var: tensor<f32>):
      %araddd4g1var = stablehlo.add %arad4g1var, %arbd4g1var : tensor<f32>
      stablehlo.return %araddd4g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g1var = stablehlo.divide %arsumd4g1var, %arnd4g1var : tensor<512xf32>
    %v1370 = stablehlo.concatenate %armeand4g1mu, %armeand4g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1371 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1372 = stablehlo.slice %v1370 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1373 = stablehlo.slice %v1370 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1374 = stablehlo.broadcast_in_dim %v1372, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1375 = stablehlo.broadcast_in_dim %v1373, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1376 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1377 = stablehlo.add %v1375, %v1376 : tensor<64x512x7x7xf32>
    %v1378 = stablehlo.rsqrt %v1377 : tensor<64x512x7x7xf32>
    %v1379 = stablehlo.subtract %v1371, %v1374 : tensor<64x512x7x7xf32>
    %v1380 = stablehlo.multiply %v1379, %v1378 : tensor<64x512x7x7xf32>
    %v1381 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1382 = stablehlo.broadcast_in_dim %d4bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1383 = stablehlo.multiply %v1380, %v1381 : tensor<64x512x7x7xf32>
    %v1384 = stablehlo.add %v1383, %v1382 : tensor<64x512x7x7xf32>
    %v1385 = stablehlo.reshape %v1384 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1386 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1387 = stablehlo.maximum %v1385, %v1386 : tensor<64x25088xf32>
    %v1388 = stablehlo.reshape %v1387 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1389 = stablehlo.convert %v1388 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1390 = stablehlo.convert %d4W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1391 = stablehlo.convolution(%v1389, %v1390)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1392 = stablehlo.convert %v1391 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1393 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1394 = stablehlo.add %v1392, %v1393 : tensor<64x512x7x7xf32>
    %v1395 = stablehlo.reshape %v1394 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1396 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1397 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1398 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1399 = stablehlo.reduce(%v1396 init: %v1397) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1400 = stablehlo.divide %v1399, %v1398 : tensor<512xf32>
    %arsumd4g2mu = "stablehlo.all_reduce"(%v1400) ({
    ^bb0(%arad4g2mu: tensor<f32>, %arbd4g2mu: tensor<f32>):
      %araddd4g2mu = stablehlo.add %arad4g2mu, %arbd4g2mu : tensor<f32>
      stablehlo.return %araddd4g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g2mu = stablehlo.divide %arsumd4g2mu, %arnd4g2mu : tensor<512xf32>
    %v1401 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1402 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1403 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1404 = stablehlo.reduce(%v1401 init: %v1402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1405 = stablehlo.divide %v1404, %v1403 : tensor<512xf32>
    %v1406 = stablehlo.broadcast_in_dim %v1405, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1407 = stablehlo.subtract %v1401, %v1406 : tensor<64x512x7x7xf32>
    %v1408 = stablehlo.multiply %v1407, %v1407 : tensor<64x512x7x7xf32>
    %v1409 = stablehlo.reduce(%v1408 init: %v1402) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1410 = stablehlo.divide %v1409, %v1403 : tensor<512xf32>
    %v1411 = stablehlo.subtract %v1405, %armeand4g2mu : tensor<512xf32>
    %v1412 = stablehlo.multiply %v1411, %v1411 : tensor<512xf32>
    %v1413 = stablehlo.add %v1410, %v1412 : tensor<512xf32>
    %arsumd4g2var = "stablehlo.all_reduce"(%v1413) ({
    ^bb0(%arad4g2var: tensor<f32>, %arbd4g2var: tensor<f32>):
      %araddd4g2var = stablehlo.add %arad4g2var, %arbd4g2var : tensor<f32>
      stablehlo.return %araddd4g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g2var = stablehlo.divide %arsumd4g2var, %arnd4g2var : tensor<512xf32>
    %v1414 = stablehlo.concatenate %armeand4g2mu, %armeand4g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1415 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1416 = stablehlo.slice %v1414 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1417 = stablehlo.slice %v1414 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1418 = stablehlo.broadcast_in_dim %v1416, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1419 = stablehlo.broadcast_in_dim %v1417, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1420 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1421 = stablehlo.add %v1419, %v1420 : tensor<64x512x7x7xf32>
    %v1422 = stablehlo.rsqrt %v1421 : tensor<64x512x7x7xf32>
    %v1423 = stablehlo.subtract %v1415, %v1418 : tensor<64x512x7x7xf32>
    %v1424 = stablehlo.multiply %v1423, %v1422 : tensor<64x512x7x7xf32>
    %v1425 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1426 = stablehlo.broadcast_in_dim %d4bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1427 = stablehlo.multiply %v1424, %v1425 : tensor<64x512x7x7xf32>
    %v1428 = stablehlo.add %v1427, %v1426 : tensor<64x512x7x7xf32>
    %v1429 = stablehlo.reshape %v1428 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1430 = stablehlo.reshape %v1343 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v1431 = stablehlo.convert %v1430 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v1432 = stablehlo.convert %d4Wp : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xbf16>
    %v1433 = stablehlo.convolution(%v1431, %v1432)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [2, 2], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<512x256x1x1xbf16>) -> tensor<64x512x7x7xbf16>
    %v1434 = stablehlo.convert %v1433 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1435 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1436 = stablehlo.add %v1434, %v1435 : tensor<64x512x7x7xf32>
    %v1437 = stablehlo.reshape %v1436 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1438 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1440 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1441 = stablehlo.reduce(%v1438 init: %v1439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1442 = stablehlo.divide %v1441, %v1440 : tensor<512xf32>
    %arsumd4gpmu = "stablehlo.all_reduce"(%v1442) ({
    ^bb0(%arad4gpmu: tensor<f32>, %arbd4gpmu: tensor<f32>):
      %araddd4gpmu = stablehlo.add %arad4gpmu, %arbd4gpmu : tensor<f32>
      stablehlo.return %araddd4gpmu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gpmu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4gpmu = stablehlo.divide %arsumd4gpmu, %arnd4gpmu : tensor<512xf32>
    %v1443 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1444 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1445 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1446 = stablehlo.reduce(%v1443 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1447 = stablehlo.divide %v1446, %v1445 : tensor<512xf32>
    %v1448 = stablehlo.broadcast_in_dim %v1447, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1449 = stablehlo.subtract %v1443, %v1448 : tensor<64x512x7x7xf32>
    %v1450 = stablehlo.multiply %v1449, %v1449 : tensor<64x512x7x7xf32>
    %v1451 = stablehlo.reduce(%v1450 init: %v1444) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1452 = stablehlo.divide %v1451, %v1445 : tensor<512xf32>
    %v1453 = stablehlo.subtract %v1447, %armeand4gpmu : tensor<512xf32>
    %v1454 = stablehlo.multiply %v1453, %v1453 : tensor<512xf32>
    %v1455 = stablehlo.add %v1452, %v1454 : tensor<512xf32>
    %arsumd4gpvar = "stablehlo.all_reduce"(%v1455) ({
    ^bb0(%arad4gpvar: tensor<f32>, %arbd4gpvar: tensor<f32>):
      %araddd4gpvar = stablehlo.add %arad4gpvar, %arbd4gpvar : tensor<f32>
      stablehlo.return %araddd4gpvar : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gpvar = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4gpvar = stablehlo.divide %arsumd4gpvar, %arnd4gpvar : tensor<512xf32>
    %v1456 = stablehlo.concatenate %armeand4gpmu, %armeand4gpvar, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1457 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1458 = stablehlo.slice %v1456 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1459 = stablehlo.slice %v1456 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1460 = stablehlo.broadcast_in_dim %v1458, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1461 = stablehlo.broadcast_in_dim %v1459, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1462 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1463 = stablehlo.add %v1461, %v1462 : tensor<64x512x7x7xf32>
    %v1464 = stablehlo.rsqrt %v1463 : tensor<64x512x7x7xf32>
    %v1465 = stablehlo.subtract %v1457, %v1460 : tensor<64x512x7x7xf32>
    %v1466 = stablehlo.multiply %v1465, %v1464 : tensor<64x512x7x7xf32>
    %v1467 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1468 = stablehlo.broadcast_in_dim %d4btp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1469 = stablehlo.multiply %v1466, %v1467 : tensor<64x512x7x7xf32>
    %v1470 = stablehlo.add %v1469, %v1468 : tensor<64x512x7x7xf32>
    %v1471 = stablehlo.reshape %v1470 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1472 = stablehlo.add %v1429, %v1471 : tensor<64x25088xf32>
    %v1473 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1474 = stablehlo.maximum %v1472, %v1473 : tensor<64x25088xf32>
    %v1475 = stablehlo.reshape %v1474 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1476 = stablehlo.convert %v1475 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1477 = stablehlo.convert %s4b0W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1478 = stablehlo.convolution(%v1476, %v1477)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1479 = stablehlo.convert %v1478 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1480 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1481 = stablehlo.add %v1479, %v1480 : tensor<64x512x7x7xf32>
    %v1482 = stablehlo.reshape %v1481 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1483 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1484 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1485 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1486 = stablehlo.reduce(%v1483 init: %v1484) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1487 = stablehlo.divide %v1486, %v1485 : tensor<512xf32>
    %arsums4b0g1mu = "stablehlo.all_reduce"(%v1487) ({
    ^bb0(%aras4b0g1mu: tensor<f32>, %arbs4b0g1mu: tensor<f32>):
      %aradds4b0g1mu = stablehlo.add %aras4b0g1mu, %arbs4b0g1mu : tensor<f32>
      stablehlo.return %aradds4b0g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1mu = stablehlo.divide %arsums4b0g1mu, %arns4b0g1mu : tensor<512xf32>
    %v1488 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1490 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1491 = stablehlo.reduce(%v1488 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1492 = stablehlo.divide %v1491, %v1490 : tensor<512xf32>
    %v1493 = stablehlo.broadcast_in_dim %v1492, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1494 = stablehlo.subtract %v1488, %v1493 : tensor<64x512x7x7xf32>
    %v1495 = stablehlo.multiply %v1494, %v1494 : tensor<64x512x7x7xf32>
    %v1496 = stablehlo.reduce(%v1495 init: %v1489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1497 = stablehlo.divide %v1496, %v1490 : tensor<512xf32>
    %v1498 = stablehlo.subtract %v1492, %armeans4b0g1mu : tensor<512xf32>
    %v1499 = stablehlo.multiply %v1498, %v1498 : tensor<512xf32>
    %v1500 = stablehlo.add %v1497, %v1499 : tensor<512xf32>
    %arsums4b0g1var = "stablehlo.all_reduce"(%v1500) ({
    ^bb0(%aras4b0g1var: tensor<f32>, %arbs4b0g1var: tensor<f32>):
      %aradds4b0g1var = stablehlo.add %aras4b0g1var, %arbs4b0g1var : tensor<f32>
      stablehlo.return %aradds4b0g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1var = stablehlo.divide %arsums4b0g1var, %arns4b0g1var : tensor<512xf32>
    %v1501 = stablehlo.concatenate %armeans4b0g1mu, %armeans4b0g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1502 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1503 = stablehlo.slice %v1501 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1504 = stablehlo.slice %v1501 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1505 = stablehlo.broadcast_in_dim %v1503, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1506 = stablehlo.broadcast_in_dim %v1504, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1507 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1508 = stablehlo.add %v1506, %v1507 : tensor<64x512x7x7xf32>
    %v1509 = stablehlo.rsqrt %v1508 : tensor<64x512x7x7xf32>
    %v1510 = stablehlo.subtract %v1502, %v1505 : tensor<64x512x7x7xf32>
    %v1511 = stablehlo.multiply %v1510, %v1509 : tensor<64x512x7x7xf32>
    %v1512 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1513 = stablehlo.broadcast_in_dim %s4b0bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1514 = stablehlo.multiply %v1511, %v1512 : tensor<64x512x7x7xf32>
    %v1515 = stablehlo.add %v1514, %v1513 : tensor<64x512x7x7xf32>
    %v1516 = stablehlo.reshape %v1515 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1517 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1518 = stablehlo.maximum %v1516, %v1517 : tensor<64x25088xf32>
    %v1519 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1520 = stablehlo.convert %v1519 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1521 = stablehlo.convert %s4b0W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1522 = stablehlo.convolution(%v1520, %v1521)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1523 = stablehlo.convert %v1522 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1524 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1525 = stablehlo.add %v1523, %v1524 : tensor<64x512x7x7xf32>
    %v1526 = stablehlo.reshape %v1525 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1527 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1528 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1529 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1530 = stablehlo.reduce(%v1527 init: %v1528) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1531 = stablehlo.divide %v1530, %v1529 : tensor<512xf32>
    %arsums4b0g2mu = "stablehlo.all_reduce"(%v1531) ({
    ^bb0(%aras4b0g2mu: tensor<f32>, %arbs4b0g2mu: tensor<f32>):
      %aradds4b0g2mu = stablehlo.add %aras4b0g2mu, %arbs4b0g2mu : tensor<f32>
      stablehlo.return %aradds4b0g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2mu = stablehlo.divide %arsums4b0g2mu, %arns4b0g2mu : tensor<512xf32>
    %v1532 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1533 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1534 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1535 = stablehlo.reduce(%v1532 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1536 = stablehlo.divide %v1535, %v1534 : tensor<512xf32>
    %v1537 = stablehlo.broadcast_in_dim %v1536, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1538 = stablehlo.subtract %v1532, %v1537 : tensor<64x512x7x7xf32>
    %v1539 = stablehlo.multiply %v1538, %v1538 : tensor<64x512x7x7xf32>
    %v1540 = stablehlo.reduce(%v1539 init: %v1533) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1541 = stablehlo.divide %v1540, %v1534 : tensor<512xf32>
    %v1542 = stablehlo.subtract %v1536, %armeans4b0g2mu : tensor<512xf32>
    %v1543 = stablehlo.multiply %v1542, %v1542 : tensor<512xf32>
    %v1544 = stablehlo.add %v1541, %v1543 : tensor<512xf32>
    %arsums4b0g2var = "stablehlo.all_reduce"(%v1544) ({
    ^bb0(%aras4b0g2var: tensor<f32>, %arbs4b0g2var: tensor<f32>):
      %aradds4b0g2var = stablehlo.add %aras4b0g2var, %arbs4b0g2var : tensor<f32>
      stablehlo.return %aradds4b0g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2var = stablehlo.divide %arsums4b0g2var, %arns4b0g2var : tensor<512xf32>
    %v1545 = stablehlo.concatenate %armeans4b0g2mu, %armeans4b0g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1546 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1547 = stablehlo.slice %v1545 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1548 = stablehlo.slice %v1545 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1549 = stablehlo.broadcast_in_dim %v1547, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1550 = stablehlo.broadcast_in_dim %v1548, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1551 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1552 = stablehlo.add %v1550, %v1551 : tensor<64x512x7x7xf32>
    %v1553 = stablehlo.rsqrt %v1552 : tensor<64x512x7x7xf32>
    %v1554 = stablehlo.subtract %v1546, %v1549 : tensor<64x512x7x7xf32>
    %v1555 = stablehlo.multiply %v1554, %v1553 : tensor<64x512x7x7xf32>
    %v1556 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1557 = stablehlo.broadcast_in_dim %s4b0bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1558 = stablehlo.multiply %v1555, %v1556 : tensor<64x512x7x7xf32>
    %v1559 = stablehlo.add %v1558, %v1557 : tensor<64x512x7x7xf32>
    %v1560 = stablehlo.reshape %v1559 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1561 = stablehlo.reshape %v1560 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1562 = stablehlo.reshape %v1474 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1563 = stablehlo.add %v1561, %v1562 : tensor<64x512x7x7xf32>
    %v1564 = stablehlo.reshape %v1563 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1565 = stablehlo.reshape %v1564 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1566 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1567 = stablehlo.maximum %v1565, %v1566 : tensor<64x512x7x7xf32>
    %v1568 = stablehlo.reshape %v1567 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1569 = stablehlo.reshape %v1568 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1570 = stablehlo.convert %v1569 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1571 = stablehlo.convert %s4b1W1 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1572 = stablehlo.convolution(%v1570, %v1571)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1573 = stablehlo.convert %v1572 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1574 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1575 = stablehlo.add %v1573, %v1574 : tensor<64x512x7x7xf32>
    %v1576 = stablehlo.reshape %v1575 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1577 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1578 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1579 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1580 = stablehlo.reduce(%v1577 init: %v1578) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1581 = stablehlo.divide %v1580, %v1579 : tensor<512xf32>
    %arsums4b1g1mu = "stablehlo.all_reduce"(%v1581) ({
    ^bb0(%aras4b1g1mu: tensor<f32>, %arbs4b1g1mu: tensor<f32>):
      %aradds4b1g1mu = stablehlo.add %aras4b1g1mu, %arbs4b1g1mu : tensor<f32>
      stablehlo.return %aradds4b1g1mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1mu = stablehlo.divide %arsums4b1g1mu, %arns4b1g1mu : tensor<512xf32>
    %v1582 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1584 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1585 = stablehlo.reduce(%v1582 init: %v1583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1586 = stablehlo.divide %v1585, %v1584 : tensor<512xf32>
    %v1587 = stablehlo.broadcast_in_dim %v1586, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1588 = stablehlo.subtract %v1582, %v1587 : tensor<64x512x7x7xf32>
    %v1589 = stablehlo.multiply %v1588, %v1588 : tensor<64x512x7x7xf32>
    %v1590 = stablehlo.reduce(%v1589 init: %v1583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1591 = stablehlo.divide %v1590, %v1584 : tensor<512xf32>
    %v1592 = stablehlo.subtract %v1586, %armeans4b1g1mu : tensor<512xf32>
    %v1593 = stablehlo.multiply %v1592, %v1592 : tensor<512xf32>
    %v1594 = stablehlo.add %v1591, %v1593 : tensor<512xf32>
    %arsums4b1g1var = "stablehlo.all_reduce"(%v1594) ({
    ^bb0(%aras4b1g1var: tensor<f32>, %arbs4b1g1var: tensor<f32>):
      %aradds4b1g1var = stablehlo.add %aras4b1g1var, %arbs4b1g1var : tensor<f32>
      stablehlo.return %aradds4b1g1var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1var = stablehlo.divide %arsums4b1g1var, %arns4b1g1var : tensor<512xf32>
    %v1595 = stablehlo.concatenate %armeans4b1g1mu, %armeans4b1g1var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1596 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1597 = stablehlo.slice %v1595 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1598 = stablehlo.slice %v1595 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1599 = stablehlo.broadcast_in_dim %v1597, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1600 = stablehlo.broadcast_in_dim %v1598, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1601 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1602 = stablehlo.add %v1600, %v1601 : tensor<64x512x7x7xf32>
    %v1603 = stablehlo.rsqrt %v1602 : tensor<64x512x7x7xf32>
    %v1604 = stablehlo.subtract %v1596, %v1599 : tensor<64x512x7x7xf32>
    %v1605 = stablehlo.multiply %v1604, %v1603 : tensor<64x512x7x7xf32>
    %v1606 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1607 = stablehlo.broadcast_in_dim %s4b1bt1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1608 = stablehlo.multiply %v1605, %v1606 : tensor<64x512x7x7xf32>
    %v1609 = stablehlo.add %v1608, %v1607 : tensor<64x512x7x7xf32>
    %v1610 = stablehlo.reshape %v1609 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1611 = stablehlo.constant dense<0.0> : tensor<64x25088xf32>
    %v1612 = stablehlo.maximum %v1610, %v1611 : tensor<64x25088xf32>
    %v1613 = stablehlo.reshape %v1612 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1614 = stablehlo.convert %v1613 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1615 = stablehlo.convert %s4b1W2 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1616 = stablehlo.convolution(%v1614, %v1615)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1617 = stablehlo.convert %v1616 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1618 = stablehlo.broadcast_in_dim %zb512, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1619 = stablehlo.add %v1617, %v1618 : tensor<64x512x7x7xf32>
    %v1620 = stablehlo.reshape %v1619 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1621 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1622 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1623 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1624 = stablehlo.reduce(%v1621 init: %v1622) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1625 = stablehlo.divide %v1624, %v1623 : tensor<512xf32>
    %arsums4b1g2mu = "stablehlo.all_reduce"(%v1625) ({
    ^bb0(%aras4b1g2mu: tensor<f32>, %arbs4b1g2mu: tensor<f32>):
      %aradds4b1g2mu = stablehlo.add %aras4b1g2mu, %arbs4b1g2mu : tensor<f32>
      stablehlo.return %aradds4b1g2mu : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2mu = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2mu = stablehlo.divide %arsums4b1g2mu, %arns4b1g2mu : tensor<512xf32>
    %v1626 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1627 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1628 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1629 = stablehlo.reduce(%v1626 init: %v1627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1630 = stablehlo.divide %v1629, %v1628 : tensor<512xf32>
    %v1631 = stablehlo.broadcast_in_dim %v1630, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1632 = stablehlo.subtract %v1626, %v1631 : tensor<64x512x7x7xf32>
    %v1633 = stablehlo.multiply %v1632, %v1632 : tensor<64x512x7x7xf32>
    %v1634 = stablehlo.reduce(%v1633 init: %v1627) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1635 = stablehlo.divide %v1634, %v1628 : tensor<512xf32>
    %v1636 = stablehlo.subtract %v1630, %armeans4b1g2mu : tensor<512xf32>
    %v1637 = stablehlo.multiply %v1636, %v1636 : tensor<512xf32>
    %v1638 = stablehlo.add %v1635, %v1637 : tensor<512xf32>
    %arsums4b1g2var = "stablehlo.all_reduce"(%v1638) ({
    ^bb0(%aras4b1g2var: tensor<f32>, %arbs4b1g2var: tensor<f32>):
      %aradds4b1g2var = stablehlo.add %aras4b1g2var, %arbs4b1g2var : tensor<f32>
      stablehlo.return %aradds4b1g2var : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2var = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2var = stablehlo.divide %arsums4b1g2var, %arns4b1g2var : tensor<512xf32>
    %v1639 = stablehlo.concatenate %armeans4b1g2mu, %armeans4b1g2var, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1640 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1641 = stablehlo.slice %v1639 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1642 = stablehlo.slice %v1639 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1643 = stablehlo.broadcast_in_dim %v1641, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1644 = stablehlo.broadcast_in_dim %v1642, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1645 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1646 = stablehlo.add %v1644, %v1645 : tensor<64x512x7x7xf32>
    %v1647 = stablehlo.rsqrt %v1646 : tensor<64x512x7x7xf32>
    %v1648 = stablehlo.subtract %v1640, %v1643 : tensor<64x512x7x7xf32>
    %v1649 = stablehlo.multiply %v1648, %v1647 : tensor<64x512x7x7xf32>
    %v1650 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1651 = stablehlo.broadcast_in_dim %s4b1bt2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1652 = stablehlo.multiply %v1649, %v1650 : tensor<64x512x7x7xf32>
    %v1653 = stablehlo.add %v1652, %v1651 : tensor<64x512x7x7xf32>
    %v1654 = stablehlo.reshape %v1653 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1655 = stablehlo.reshape %v1654 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1656 = stablehlo.reshape %v1568 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1657 = stablehlo.add %v1655, %v1656 : tensor<64x512x7x7xf32>
    %v1658 = stablehlo.reshape %v1657 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1659 = stablehlo.reshape %v1658 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1660 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1661 = stablehlo.maximum %v1659, %v1660 : tensor<64x512x7x7xf32>
    %v1662 = stablehlo.reshape %v1661 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1663 = stablehlo.reshape %v1662 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1664 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1665 = stablehlo.reduce(%v1663 init: %v1664) applies stablehlo.add across dimensions = [2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512xf32>
    %v1666 = stablehlo.constant dense<49.0> : tensor<64x512xf32>
    %v1667 = stablehlo.divide %v1665, %v1666 : tensor<64x512xf32>
    %v1668 = stablehlo.dot_general %v1667, %Wd, contracting_dims = [1] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<512x1000xf32>) -> tensor<64x1000xf32>
    %v1669 = stablehlo.broadcast_in_dim %bd, dims = [1] : (tensor<1000xf32>) -> tensor<64x1000xf32>
    %v1670 = stablehlo.add %v1668, %v1669 : tensor<64x1000xf32>
    %v1671 = stablehlo.reshape %v1670 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1672 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1673 = stablehlo.exponential %v1671 : tensor<64x1x1000xf32>
    %v1674 = stablehlo.reduce(%v1673 init: %v1672) applies stablehlo.add across dimensions = [2] : (tensor<64x1x1000xf32>, tensor<f32>) -> tensor<64x1xf32>
    %v1675 = stablehlo.broadcast_in_dim %v1674, dims = [0, 1] : (tensor<64x1xf32>) -> tensor<64x1x1000xf32>
    %v1676 = stablehlo.divide %v1673, %v1675 : tensor<64x1x1000xf32>
    %v1677 = stablehlo.reshape %v1676 : (tensor<64x1x1000xf32>) -> tensor<64x1000xf32>
    %v1678 = stablehlo.subtract %v1677, %onehot : tensor<64x1000xf32>
    %v1679 = stablehlo.constant dense<0.100000> : tensor<64x1000xf32>
    %v1680 = stablehlo.multiply %onehot, %v1679 : tensor<64x1000xf32>
    %v1681 = stablehlo.add %v1678, %v1680 : tensor<64x1000xf32>
    %v1682 = stablehlo.constant dense<-0.000100> : tensor<64x1000xf32>
    %v1683 = stablehlo.add %v1681, %v1682 : tensor<64x1000xf32>
    %v1684 = stablehlo.constant dense<64.0> : tensor<64x1000xf32>
    %v1685 = stablehlo.divide %v1683, %v1684 : tensor<64x1000xf32>
    %v1686 = stablehlo.reshape %v1685 : (tensor<64x1000xf32>) -> tensor<64x1x1000xf32>
    %v1687 = stablehlo.dot_general %v1686, %Wd, contracting_dims = [2] x [1], precision = [DEFAULT, DEFAULT] : (tensor<64x1x1000xf32>, tensor<512x1000xf32>) -> tensor<64x1x512xf32>
    %v1688 = stablehlo.reshape %v1687 : (tensor<64x1x512xf32>) -> tensor<64x512xf32>
    %v1689 = stablehlo.dot_general %v1667, %v1685, contracting_dims = [0] x [0], precision = [DEFAULT, DEFAULT] : (tensor<64x512xf32>, tensor<64x1000xf32>) -> tensor<512x1000xf32>
    %v1690 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1691 = stablehlo.reduce(%v1685 init: %v1690) applies stablehlo.add across dimensions = [0] : (tensor<64x1000xf32>, tensor<f32>) -> tensor<1000xf32>
    %v1692 = stablehlo.broadcast_in_dim %v1688, dims = [0, 1] : (tensor<64x512xf32>) -> tensor<64x512x7x7xf32>
    %v1693 = stablehlo.constant dense<49.0> : tensor<64x512x7x7xf32>
    %v1694 = stablehlo.divide %v1692, %v1693 : tensor<64x512x7x7xf32>
    %v1695 = stablehlo.reshape %v1694 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1696 = stablehlo.reshape %v1695 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1697 = stablehlo.reshape %v1658 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1698 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1699 = stablehlo.compare GT, %v1697, %v1698 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1700 = stablehlo.select %v1699, %v1696, %v1698 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1701 = stablehlo.reshape %v1700 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1702 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1703 = stablehlo.slice %v1639 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1704 = stablehlo.slice %v1639 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1705 = stablehlo.broadcast_in_dim %v1703, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1706 = stablehlo.broadcast_in_dim %v1704, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1707 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1708 = stablehlo.add %v1706, %v1707 : tensor<64x512x7x7xf32>
    %v1709 = stablehlo.rsqrt %v1708 : tensor<64x512x7x7xf32>
    %v1710 = stablehlo.subtract %v1702, %v1705 : tensor<64x512x7x7xf32>
    %v1711 = stablehlo.multiply %v1710, %v1709 : tensor<64x512x7x7xf32>
    %v1712 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1713 = stablehlo.reshape %v1701 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1714 = stablehlo.multiply %v1712, %v1713 : tensor<64x512x7x7xf32>
    %v1715 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1716 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1717 = stablehlo.reduce(%v1714 init: %v1715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1718 = stablehlo.divide %v1717, %v1716 : tensor<512xf32>
    %v1719 = stablehlo.multiply %v1711, %v1714 : tensor<64x512x7x7xf32>
    %v1720 = stablehlo.reduce(%v1719 init: %v1715) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1721 = stablehlo.divide %v1720, %v1716 : tensor<512xf32>
    %v1722 = stablehlo.concatenate %v1718, %v1721, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1723 = stablehlo.concatenate %v1639, %v1722, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g2dst = "stablehlo.all_reduce"(%v1723) ({
    ^bb0(%aras4b1g2dst: tensor<f32>, %arbs4b1g2dst: tensor<f32>):
      %aradds4b1g2dst = stablehlo.add %aras4b1g2dst, %arbs4b1g2dst : tensor<f32>
      stablehlo.return %aradds4b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g2dst = stablehlo.divide %arsums4b1g2dst, %arns4b1g2dst : tensor<2048xf32>
    %v1724 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1725 = stablehlo.slice %armeans4b1g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1726 = stablehlo.slice %armeans4b1g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1727 = stablehlo.slice %armeans4b1g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1728 = stablehlo.slice %armeans4b1g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1729 = stablehlo.broadcast_in_dim %v1725, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1730 = stablehlo.broadcast_in_dim %v1726, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1731 = stablehlo.broadcast_in_dim %v1727, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1732 = stablehlo.broadcast_in_dim %v1728, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1733 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1734 = stablehlo.add %v1730, %v1733 : tensor<64x512x7x7xf32>
    %v1735 = stablehlo.rsqrt %v1734 : tensor<64x512x7x7xf32>
    %v1736 = stablehlo.subtract %v1724, %v1729 : tensor<64x512x7x7xf32>
    %v1737 = stablehlo.multiply %v1736, %v1735 : tensor<64x512x7x7xf32>
    %v1738 = stablehlo.broadcast_in_dim %s4b1g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1739 = stablehlo.reshape %v1701 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1740 = stablehlo.multiply %v1738, %v1739 : tensor<64x512x7x7xf32>
    %v1741 = stablehlo.subtract %v1740, %v1731 : tensor<64x512x7x7xf32>
    %v1742 = stablehlo.multiply %v1737, %v1732 : tensor<64x512x7x7xf32>
    %v1743 = stablehlo.subtract %v1741, %v1742 : tensor<64x512x7x7xf32>
    %v1744 = stablehlo.multiply %v1735, %v1743 : tensor<64x512x7x7xf32>
    %v1745 = stablehlo.reshape %v1744 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1746 = stablehlo.reshape %v1745 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1747 = stablehlo.reverse %s4b1W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1748 = stablehlo.transpose %v1747, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1749 = stablehlo.convert %v1746 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1750 = stablehlo.convert %v1748 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1751 = stablehlo.convolution(%v1749, %v1750)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1752 = stablehlo.convert %v1751 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1753 = stablehlo.reshape %v1752 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1754 = stablehlo.reshape %v1753 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1755 = stablehlo.reshape %v1610 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1756 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1757 = stablehlo.compare GT, %v1755, %v1756 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1758 = stablehlo.select %v1757, %v1754, %v1756 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1759 = stablehlo.reshape %v1758 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1760 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1761 = stablehlo.slice %v1595 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1762 = stablehlo.slice %v1595 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1763 = stablehlo.broadcast_in_dim %v1761, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1764 = stablehlo.broadcast_in_dim %v1762, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1765 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1766 = stablehlo.add %v1764, %v1765 : tensor<64x512x7x7xf32>
    %v1767 = stablehlo.rsqrt %v1766 : tensor<64x512x7x7xf32>
    %v1768 = stablehlo.subtract %v1760, %v1763 : tensor<64x512x7x7xf32>
    %v1769 = stablehlo.multiply %v1768, %v1767 : tensor<64x512x7x7xf32>
    %v1770 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1771 = stablehlo.reshape %v1759 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1772 = stablehlo.multiply %v1770, %v1771 : tensor<64x512x7x7xf32>
    %v1773 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1774 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1775 = stablehlo.reduce(%v1772 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1776 = stablehlo.divide %v1775, %v1774 : tensor<512xf32>
    %v1777 = stablehlo.multiply %v1769, %v1772 : tensor<64x512x7x7xf32>
    %v1778 = stablehlo.reduce(%v1777 init: %v1773) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1779 = stablehlo.divide %v1778, %v1774 : tensor<512xf32>
    %v1780 = stablehlo.concatenate %v1776, %v1779, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1781 = stablehlo.concatenate %v1595, %v1780, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b1g1dst = "stablehlo.all_reduce"(%v1781) ({
    ^bb0(%aras4b1g1dst: tensor<f32>, %arbs4b1g1dst: tensor<f32>):
      %aradds4b1g1dst = stablehlo.add %aras4b1g1dst, %arbs4b1g1dst : tensor<f32>
      stablehlo.return %aradds4b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b1g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b1g1dst = stablehlo.divide %arsums4b1g1dst, %arns4b1g1dst : tensor<2048xf32>
    %v1782 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1783 = stablehlo.slice %armeans4b1g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1784 = stablehlo.slice %armeans4b1g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1785 = stablehlo.slice %armeans4b1g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1786 = stablehlo.slice %armeans4b1g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1787 = stablehlo.broadcast_in_dim %v1783, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1788 = stablehlo.broadcast_in_dim %v1784, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1789 = stablehlo.broadcast_in_dim %v1785, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1790 = stablehlo.broadcast_in_dim %v1786, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1791 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1792 = stablehlo.add %v1788, %v1791 : tensor<64x512x7x7xf32>
    %v1793 = stablehlo.rsqrt %v1792 : tensor<64x512x7x7xf32>
    %v1794 = stablehlo.subtract %v1782, %v1787 : tensor<64x512x7x7xf32>
    %v1795 = stablehlo.multiply %v1794, %v1793 : tensor<64x512x7x7xf32>
    %v1796 = stablehlo.broadcast_in_dim %s4b1g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1797 = stablehlo.reshape %v1759 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1798 = stablehlo.multiply %v1796, %v1797 : tensor<64x512x7x7xf32>
    %v1799 = stablehlo.subtract %v1798, %v1789 : tensor<64x512x7x7xf32>
    %v1800 = stablehlo.multiply %v1795, %v1790 : tensor<64x512x7x7xf32>
    %v1801 = stablehlo.subtract %v1799, %v1800 : tensor<64x512x7x7xf32>
    %v1802 = stablehlo.multiply %v1793, %v1801 : tensor<64x512x7x7xf32>
    %v1803 = stablehlo.reshape %v1802 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1804 = stablehlo.reshape %v1803 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1805 = stablehlo.reverse %s4b1W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1806 = stablehlo.transpose %v1805, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1807 = stablehlo.convert %v1804 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1808 = stablehlo.convert %v1806 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1809 = stablehlo.convolution(%v1807, %v1808)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1810 = stablehlo.convert %v1809 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1811 = stablehlo.reshape %v1810 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1812 = stablehlo.reshape %v1811 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1813 = stablehlo.reshape %v1701 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1814 = stablehlo.add %v1812, %v1813 : tensor<64x512x7x7xf32>
    %v1815 = stablehlo.reshape %v1814 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1816 = stablehlo.reshape %v1568 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1817 = stablehlo.reshape %v1803 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1818 = stablehlo.transpose %v1816, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1819 = stablehlo.transpose %v1817, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1820 = stablehlo.convert %v1818 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1821 = stablehlo.convert %v1819 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1822 = stablehlo.convolution(%v1820, %v1821)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1823 = stablehlo.convert %v1822 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1824 = stablehlo.transpose %v1823, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1825 = stablehlo.reshape %v1576 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1826 = stablehlo.slice %v1595 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1827 = stablehlo.slice %v1595 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1828 = stablehlo.broadcast_in_dim %v1826, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1829 = stablehlo.broadcast_in_dim %v1827, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1830 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1831 = stablehlo.add %v1829, %v1830 : tensor<64x512x7x7xf32>
    %v1832 = stablehlo.rsqrt %v1831 : tensor<64x512x7x7xf32>
    %v1833 = stablehlo.subtract %v1825, %v1828 : tensor<64x512x7x7xf32>
    %v1834 = stablehlo.multiply %v1833, %v1832 : tensor<64x512x7x7xf32>
    %v1835 = stablehlo.reshape %v1759 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1836 = stablehlo.multiply %v1835, %v1834 : tensor<64x512x7x7xf32>
    %v1837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1838 = stablehlo.reduce(%v1836 init: %v1837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1839 = stablehlo.reshape %v1759 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1840 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1841 = stablehlo.reduce(%v1839 init: %v1840) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1842 = stablehlo.reshape %v1612 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1843 = stablehlo.reshape %v1745 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1844 = stablehlo.transpose %v1842, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1845 = stablehlo.transpose %v1843, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1846 = stablehlo.convert %v1844 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1847 = stablehlo.convert %v1845 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1848 = stablehlo.convolution(%v1846, %v1847)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1849 = stablehlo.convert %v1848 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1850 = stablehlo.transpose %v1849, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1851 = stablehlo.reshape %v1620 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1852 = stablehlo.slice %v1639 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1853 = stablehlo.slice %v1639 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1854 = stablehlo.broadcast_in_dim %v1852, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1855 = stablehlo.broadcast_in_dim %v1853, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1856 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1857 = stablehlo.add %v1855, %v1856 : tensor<64x512x7x7xf32>
    %v1858 = stablehlo.rsqrt %v1857 : tensor<64x512x7x7xf32>
    %v1859 = stablehlo.subtract %v1851, %v1854 : tensor<64x512x7x7xf32>
    %v1860 = stablehlo.multiply %v1859, %v1858 : tensor<64x512x7x7xf32>
    %v1861 = stablehlo.reshape %v1701 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1862 = stablehlo.multiply %v1861, %v1860 : tensor<64x512x7x7xf32>
    %v1863 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1864 = stablehlo.reduce(%v1862 init: %v1863) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1865 = stablehlo.reshape %v1701 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1866 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1867 = stablehlo.reduce(%v1865 init: %v1866) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1868 = stablehlo.reshape %v1815 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1869 = stablehlo.reshape %v1564 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1870 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1871 = stablehlo.compare GT, %v1869, %v1870 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1872 = stablehlo.select %v1871, %v1868, %v1870 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1873 = stablehlo.reshape %v1872 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1874 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1875 = stablehlo.slice %v1545 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1876 = stablehlo.slice %v1545 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1877 = stablehlo.broadcast_in_dim %v1875, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1878 = stablehlo.broadcast_in_dim %v1876, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1879 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1880 = stablehlo.add %v1878, %v1879 : tensor<64x512x7x7xf32>
    %v1881 = stablehlo.rsqrt %v1880 : tensor<64x512x7x7xf32>
    %v1882 = stablehlo.subtract %v1874, %v1877 : tensor<64x512x7x7xf32>
    %v1883 = stablehlo.multiply %v1882, %v1881 : tensor<64x512x7x7xf32>
    %v1884 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1885 = stablehlo.reshape %v1873 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1886 = stablehlo.multiply %v1884, %v1885 : tensor<64x512x7x7xf32>
    %v1887 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1888 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1889 = stablehlo.reduce(%v1886 init: %v1887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1890 = stablehlo.divide %v1889, %v1888 : tensor<512xf32>
    %v1891 = stablehlo.multiply %v1883, %v1886 : tensor<64x512x7x7xf32>
    %v1892 = stablehlo.reduce(%v1891 init: %v1887) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1893 = stablehlo.divide %v1892, %v1888 : tensor<512xf32>
    %v1894 = stablehlo.concatenate %v1890, %v1893, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1895 = stablehlo.concatenate %v1545, %v1894, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g2dst = "stablehlo.all_reduce"(%v1895) ({
    ^bb0(%aras4b0g2dst: tensor<f32>, %arbs4b0g2dst: tensor<f32>):
      %aradds4b0g2dst = stablehlo.add %aras4b0g2dst, %arbs4b0g2dst : tensor<f32>
      stablehlo.return %aradds4b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g2dst = stablehlo.divide %arsums4b0g2dst, %arns4b0g2dst : tensor<2048xf32>
    %v1896 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1897 = stablehlo.slice %armeans4b0g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1898 = stablehlo.slice %armeans4b0g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1899 = stablehlo.slice %armeans4b0g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1900 = stablehlo.slice %armeans4b0g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1901 = stablehlo.broadcast_in_dim %v1897, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1902 = stablehlo.broadcast_in_dim %v1898, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1903 = stablehlo.broadcast_in_dim %v1899, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1904 = stablehlo.broadcast_in_dim %v1900, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1905 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1906 = stablehlo.add %v1902, %v1905 : tensor<64x512x7x7xf32>
    %v1907 = stablehlo.rsqrt %v1906 : tensor<64x512x7x7xf32>
    %v1908 = stablehlo.subtract %v1896, %v1901 : tensor<64x512x7x7xf32>
    %v1909 = stablehlo.multiply %v1908, %v1907 : tensor<64x512x7x7xf32>
    %v1910 = stablehlo.broadcast_in_dim %s4b0g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1911 = stablehlo.reshape %v1873 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1912 = stablehlo.multiply %v1910, %v1911 : tensor<64x512x7x7xf32>
    %v1913 = stablehlo.subtract %v1912, %v1903 : tensor<64x512x7x7xf32>
    %v1914 = stablehlo.multiply %v1909, %v1904 : tensor<64x512x7x7xf32>
    %v1915 = stablehlo.subtract %v1913, %v1914 : tensor<64x512x7x7xf32>
    %v1916 = stablehlo.multiply %v1907, %v1915 : tensor<64x512x7x7xf32>
    %v1917 = stablehlo.reshape %v1916 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1918 = stablehlo.reshape %v1917 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1919 = stablehlo.reverse %s4b0W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1920 = stablehlo.transpose %v1919, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1921 = stablehlo.convert %v1918 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1922 = stablehlo.convert %v1920 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1923 = stablehlo.convolution(%v1921, %v1922)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1924 = stablehlo.convert %v1923 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1925 = stablehlo.reshape %v1924 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1926 = stablehlo.reshape %v1925 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1927 = stablehlo.reshape %v1516 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1928 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v1929 = stablehlo.compare GT, %v1927, %v1928 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v1930 = stablehlo.select %v1929, %v1926, %v1928 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v1931 = stablehlo.reshape %v1930 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1932 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1933 = stablehlo.slice %v1501 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1934 = stablehlo.slice %v1501 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1935 = stablehlo.broadcast_in_dim %v1933, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1936 = stablehlo.broadcast_in_dim %v1934, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1937 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1938 = stablehlo.add %v1936, %v1937 : tensor<64x512x7x7xf32>
    %v1939 = stablehlo.rsqrt %v1938 : tensor<64x512x7x7xf32>
    %v1940 = stablehlo.subtract %v1932, %v1935 : tensor<64x512x7x7xf32>
    %v1941 = stablehlo.multiply %v1940, %v1939 : tensor<64x512x7x7xf32>
    %v1942 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1943 = stablehlo.reshape %v1931 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1944 = stablehlo.multiply %v1942, %v1943 : tensor<64x512x7x7xf32>
    %v1945 = stablehlo.constant dense<0.0> : tensor<f32>
    %v1946 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v1947 = stablehlo.reduce(%v1944 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1948 = stablehlo.divide %v1947, %v1946 : tensor<512xf32>
    %v1949 = stablehlo.multiply %v1941, %v1944 : tensor<64x512x7x7xf32>
    %v1950 = stablehlo.reduce(%v1949 init: %v1945) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v1951 = stablehlo.divide %v1950, %v1946 : tensor<512xf32>
    %v1952 = stablehlo.concatenate %v1948, %v1951, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v1953 = stablehlo.concatenate %v1501, %v1952, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsums4b0g1dst = "stablehlo.all_reduce"(%v1953) ({
    ^bb0(%aras4b0g1dst: tensor<f32>, %arbs4b0g1dst: tensor<f32>):
      %aradds4b0g1dst = stablehlo.add %aras4b0g1dst, %arbs4b0g1dst : tensor<f32>
      stablehlo.return %aradds4b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arns4b0g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeans4b0g1dst = stablehlo.divide %arsums4b0g1dst, %arns4b0g1dst : tensor<2048xf32>
    %v1954 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1955 = stablehlo.slice %armeans4b0g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1956 = stablehlo.slice %armeans4b0g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1957 = stablehlo.slice %armeans4b0g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1958 = stablehlo.slice %armeans4b0g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v1959 = stablehlo.broadcast_in_dim %v1955, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1960 = stablehlo.broadcast_in_dim %v1956, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1961 = stablehlo.broadcast_in_dim %v1957, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1962 = stablehlo.broadcast_in_dim %v1958, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1963 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v1964 = stablehlo.add %v1960, %v1963 : tensor<64x512x7x7xf32>
    %v1965 = stablehlo.rsqrt %v1964 : tensor<64x512x7x7xf32>
    %v1966 = stablehlo.subtract %v1954, %v1959 : tensor<64x512x7x7xf32>
    %v1967 = stablehlo.multiply %v1966, %v1965 : tensor<64x512x7x7xf32>
    %v1968 = stablehlo.broadcast_in_dim %s4b0g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v1969 = stablehlo.reshape %v1931 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1970 = stablehlo.multiply %v1968, %v1969 : tensor<64x512x7x7xf32>
    %v1971 = stablehlo.subtract %v1970, %v1961 : tensor<64x512x7x7xf32>
    %v1972 = stablehlo.multiply %v1967, %v1962 : tensor<64x512x7x7xf32>
    %v1973 = stablehlo.subtract %v1971, %v1972 : tensor<64x512x7x7xf32>
    %v1974 = stablehlo.multiply %v1965, %v1973 : tensor<64x512x7x7xf32>
    %v1975 = stablehlo.reshape %v1974 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1976 = stablehlo.reshape %v1975 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1977 = stablehlo.reverse %s4b0W1, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v1978 = stablehlo.transpose %v1977, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1979 = stablehlo.convert %v1976 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v1980 = stablehlo.convert %v1978 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v1981 = stablehlo.convolution(%v1979, %v1980)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v1982 = stablehlo.convert %v1981 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v1983 = stablehlo.reshape %v1982 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1984 = stablehlo.reshape %v1983 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1985 = stablehlo.reshape %v1873 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1986 = stablehlo.add %v1984, %v1985 : tensor<64x512x7x7xf32>
    %v1987 = stablehlo.reshape %v1986 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v1988 = stablehlo.reshape %v1474 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1989 = stablehlo.reshape %v1975 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1990 = stablehlo.transpose %v1988, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1991 = stablehlo.transpose %v1989, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v1992 = stablehlo.convert %v1990 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1993 = stablehlo.convert %v1991 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v1994 = stablehlo.convolution(%v1992, %v1993)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v1995 = stablehlo.convert %v1994 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v1996 = stablehlo.transpose %v1995, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v1997 = stablehlo.reshape %v1482 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v1998 = stablehlo.slice %v1501 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v1999 = stablehlo.slice %v1501 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2000 = stablehlo.broadcast_in_dim %v1998, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2001 = stablehlo.broadcast_in_dim %v1999, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2002 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2003 = stablehlo.add %v2001, %v2002 : tensor<64x512x7x7xf32>
    %v2004 = stablehlo.rsqrt %v2003 : tensor<64x512x7x7xf32>
    %v2005 = stablehlo.subtract %v1997, %v2000 : tensor<64x512x7x7xf32>
    %v2006 = stablehlo.multiply %v2005, %v2004 : tensor<64x512x7x7xf32>
    %v2007 = stablehlo.reshape %v1931 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2008 = stablehlo.multiply %v2007, %v2006 : tensor<64x512x7x7xf32>
    %v2009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2010 = stablehlo.reduce(%v2008 init: %v2009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2011 = stablehlo.reshape %v1931 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2012 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2013 = stablehlo.reduce(%v2011 init: %v2012) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2014 = stablehlo.reshape %v1518 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2015 = stablehlo.reshape %v1917 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2016 = stablehlo.transpose %v2014, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2017 = stablehlo.transpose %v2015, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2018 = stablehlo.convert %v2016 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2019 = stablehlo.convert %v2017 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2020 = stablehlo.convolution(%v2018, %v2019)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2021 = stablehlo.convert %v2020 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2022 = stablehlo.transpose %v2021, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2023 = stablehlo.reshape %v1526 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2024 = stablehlo.slice %v1545 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2025 = stablehlo.slice %v1545 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2026 = stablehlo.broadcast_in_dim %v2024, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2027 = stablehlo.broadcast_in_dim %v2025, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2028 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2029 = stablehlo.add %v2027, %v2028 : tensor<64x512x7x7xf32>
    %v2030 = stablehlo.rsqrt %v2029 : tensor<64x512x7x7xf32>
    %v2031 = stablehlo.subtract %v2023, %v2026 : tensor<64x512x7x7xf32>
    %v2032 = stablehlo.multiply %v2031, %v2030 : tensor<64x512x7x7xf32>
    %v2033 = stablehlo.reshape %v1873 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2034 = stablehlo.multiply %v2033, %v2032 : tensor<64x512x7x7xf32>
    %v2035 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2036 = stablehlo.reduce(%v2034 init: %v2035) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2037 = stablehlo.reshape %v1873 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2038 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2039 = stablehlo.reduce(%v2037 init: %v2038) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2040 = stablehlo.reshape %v1987 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2041 = stablehlo.reshape %v1472 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2042 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2043 = stablehlo.compare GT, %v2041, %v2042 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2044 = stablehlo.select %v2043, %v2040, %v2042 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2045 = stablehlo.reshape %v2044 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2046 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2047 = stablehlo.slice %v1414 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2048 = stablehlo.slice %v1414 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2049 = stablehlo.broadcast_in_dim %v2047, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2050 = stablehlo.broadcast_in_dim %v2048, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2051 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2052 = stablehlo.add %v2050, %v2051 : tensor<64x512x7x7xf32>
    %v2053 = stablehlo.rsqrt %v2052 : tensor<64x512x7x7xf32>
    %v2054 = stablehlo.subtract %v2046, %v2049 : tensor<64x512x7x7xf32>
    %v2055 = stablehlo.multiply %v2054, %v2053 : tensor<64x512x7x7xf32>
    %v2056 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2057 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2058 = stablehlo.multiply %v2056, %v2057 : tensor<64x512x7x7xf32>
    %v2059 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2060 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2061 = stablehlo.reduce(%v2058 init: %v2059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2062 = stablehlo.divide %v2061, %v2060 : tensor<512xf32>
    %v2063 = stablehlo.multiply %v2055, %v2058 : tensor<64x512x7x7xf32>
    %v2064 = stablehlo.reduce(%v2063 init: %v2059) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2065 = stablehlo.divide %v2064, %v2060 : tensor<512xf32>
    %v2066 = stablehlo.concatenate %v2062, %v2065, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2067 = stablehlo.concatenate %v1414, %v2066, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4g2dst = "stablehlo.all_reduce"(%v2067) ({
    ^bb0(%arad4g2dst: tensor<f32>, %arbd4g2dst: tensor<f32>):
      %araddd4g2dst = stablehlo.add %arad4g2dst, %arbd4g2dst : tensor<f32>
      stablehlo.return %araddd4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4g2dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeand4g2dst = stablehlo.divide %arsumd4g2dst, %arnd4g2dst : tensor<2048xf32>
    %v2068 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2069 = stablehlo.slice %armeand4g2dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2070 = stablehlo.slice %armeand4g2dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2071 = stablehlo.slice %armeand4g2dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2072 = stablehlo.slice %armeand4g2dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2073 = stablehlo.broadcast_in_dim %v2069, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2074 = stablehlo.broadcast_in_dim %v2070, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2075 = stablehlo.broadcast_in_dim %v2071, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2076 = stablehlo.broadcast_in_dim %v2072, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2077 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2078 = stablehlo.add %v2074, %v2077 : tensor<64x512x7x7xf32>
    %v2079 = stablehlo.rsqrt %v2078 : tensor<64x512x7x7xf32>
    %v2080 = stablehlo.subtract %v2068, %v2073 : tensor<64x512x7x7xf32>
    %v2081 = stablehlo.multiply %v2080, %v2079 : tensor<64x512x7x7xf32>
    %v2082 = stablehlo.broadcast_in_dim %d4g2, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2083 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2084 = stablehlo.multiply %v2082, %v2083 : tensor<64x512x7x7xf32>
    %v2085 = stablehlo.subtract %v2084, %v2075 : tensor<64x512x7x7xf32>
    %v2086 = stablehlo.multiply %v2081, %v2076 : tensor<64x512x7x7xf32>
    %v2087 = stablehlo.subtract %v2085, %v2086 : tensor<64x512x7x7xf32>
    %v2088 = stablehlo.multiply %v2079, %v2087 : tensor<64x512x7x7xf32>
    %v2089 = stablehlo.reshape %v2088 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2090 = stablehlo.reshape %v2089 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2091 = stablehlo.reverse %d4W2, dims = [2, 3] : tensor<512x512x3x3xf32>
    %v2092 = stablehlo.transpose %v2091, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2093 = stablehlo.convert %v2090 : (tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xbf16>
    %v2094 = stablehlo.convert %v2092 : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xbf16>
    %v2095 = stablehlo.convolution(%v2093, %v2094)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x7x7xbf16>, tensor<512x512x3x3xbf16>) -> tensor<64x512x7x7xbf16>
    %v2096 = stablehlo.convert %v2095 : (tensor<64x512x7x7xbf16>) -> tensor<64x512x7x7xf32>
    %v2097 = stablehlo.reshape %v2096 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2098 = stablehlo.reshape %v2097 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2099 = stablehlo.reshape %v1385 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2100 = stablehlo.constant dense<0.0> : tensor<64x512x7x7xf32>
    %v2101 = stablehlo.compare GT, %v2099, %v2100 : (tensor<64x512x7x7xf32>, tensor<64x512x7x7xf32>) -> tensor<64x512x7x7xi1>
    %v2102 = stablehlo.select %v2101, %v2098, %v2100 : tensor<64x512x7x7xi1>, tensor<64x512x7x7xf32>
    %v2103 = stablehlo.reshape %v2102 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2104 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2105 = stablehlo.slice %v1370 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2106 = stablehlo.slice %v1370 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2107 = stablehlo.broadcast_in_dim %v2105, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2108 = stablehlo.broadcast_in_dim %v2106, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2109 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2110 = stablehlo.add %v2108, %v2109 : tensor<64x512x7x7xf32>
    %v2111 = stablehlo.rsqrt %v2110 : tensor<64x512x7x7xf32>
    %v2112 = stablehlo.subtract %v2104, %v2107 : tensor<64x512x7x7xf32>
    %v2113 = stablehlo.multiply %v2112, %v2111 : tensor<64x512x7x7xf32>
    %v2114 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2115 = stablehlo.reshape %v2103 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2116 = stablehlo.multiply %v2114, %v2115 : tensor<64x512x7x7xf32>
    %v2117 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2118 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2119 = stablehlo.reduce(%v2116 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2120 = stablehlo.divide %v2119, %v2118 : tensor<512xf32>
    %v2121 = stablehlo.multiply %v2113, %v2116 : tensor<64x512x7x7xf32>
    %v2122 = stablehlo.reduce(%v2121 init: %v2117) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2123 = stablehlo.divide %v2122, %v2118 : tensor<512xf32>
    %v2124 = stablehlo.concatenate %v2120, %v2123, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2125 = stablehlo.concatenate %v1370, %v2124, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4g1dst = "stablehlo.all_reduce"(%v2125) ({
    ^bb0(%arad4g1dst: tensor<f32>, %arbd4g1dst: tensor<f32>):
      %araddd4g1dst = stablehlo.add %arad4g1dst, %arbd4g1dst : tensor<f32>
      stablehlo.return %araddd4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4g1dst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeand4g1dst = stablehlo.divide %arsumd4g1dst, %arnd4g1dst : tensor<2048xf32>
    %v2126 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2127 = stablehlo.slice %armeand4g1dst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2128 = stablehlo.slice %armeand4g1dst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2129 = stablehlo.slice %armeand4g1dst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2130 = stablehlo.slice %armeand4g1dst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2131 = stablehlo.broadcast_in_dim %v2127, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2132 = stablehlo.broadcast_in_dim %v2128, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2133 = stablehlo.broadcast_in_dim %v2129, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2134 = stablehlo.broadcast_in_dim %v2130, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2135 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2136 = stablehlo.add %v2132, %v2135 : tensor<64x512x7x7xf32>
    %v2137 = stablehlo.rsqrt %v2136 : tensor<64x512x7x7xf32>
    %v2138 = stablehlo.subtract %v2126, %v2131 : tensor<64x512x7x7xf32>
    %v2139 = stablehlo.multiply %v2138, %v2137 : tensor<64x512x7x7xf32>
    %v2140 = stablehlo.broadcast_in_dim %d4g1, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2141 = stablehlo.reshape %v2103 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2142 = stablehlo.multiply %v2140, %v2141 : tensor<64x512x7x7xf32>
    %v2143 = stablehlo.subtract %v2142, %v2133 : tensor<64x512x7x7xf32>
    %v2144 = stablehlo.multiply %v2139, %v2134 : tensor<64x512x7x7xf32>
    %v2145 = stablehlo.subtract %v2143, %v2144 : tensor<64x512x7x7xf32>
    %v2146 = stablehlo.multiply %v2137, %v2145 : tensor<64x512x7x7xf32>
    %v2147 = stablehlo.reshape %v2146 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2148 = stablehlo.reshape %v2147 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2149 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2150 = stablehlo.pad %v2148, %v2149, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2151 = stablehlo.reverse %d4W1, dims = [2, 3] : tensor<512x256x3x3xf32>
    %v2152 = stablehlo.transpose %v2151, dims = [1, 0, 2, 3] : (tensor<512x256x3x3xf32>) -> tensor<256x512x3x3xf32>
    %v2153 = stablehlo.convert %v2150 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2154 = stablehlo.convert %v2152 : (tensor<256x512x3x3xf32>) -> tensor<256x512x3x3xbf16>
    %v2155 = stablehlo.convolution(%v2153, %v2154)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<256x512x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2156 = stablehlo.convert %v2155 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2157 = stablehlo.reshape %v2156 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2158 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2159 = stablehlo.slice %v1456 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2160 = stablehlo.slice %v1456 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2161 = stablehlo.broadcast_in_dim %v2159, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2162 = stablehlo.broadcast_in_dim %v2160, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2163 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2164 = stablehlo.add %v2162, %v2163 : tensor<64x512x7x7xf32>
    %v2165 = stablehlo.rsqrt %v2164 : tensor<64x512x7x7xf32>
    %v2166 = stablehlo.subtract %v2158, %v2161 : tensor<64x512x7x7xf32>
    %v2167 = stablehlo.multiply %v2166, %v2165 : tensor<64x512x7x7xf32>
    %v2168 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2169 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2170 = stablehlo.multiply %v2168, %v2169 : tensor<64x512x7x7xf32>
    %v2171 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2172 = stablehlo.constant dense<3136.0> : tensor<512xf32>
    %v2173 = stablehlo.reduce(%v2170 init: %v2171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2174 = stablehlo.divide %v2173, %v2172 : tensor<512xf32>
    %v2175 = stablehlo.multiply %v2167, %v2170 : tensor<64x512x7x7xf32>
    %v2176 = stablehlo.reduce(%v2175 init: %v2171) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2177 = stablehlo.divide %v2176, %v2172 : tensor<512xf32>
    %v2178 = stablehlo.concatenate %v2174, %v2177, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %v2179 = stablehlo.concatenate %v1456, %v2178, dim = 0 : (tensor<1024xf32>, tensor<1024xf32>) -> tensor<2048xf32>
    %arsumd4gpdst = "stablehlo.all_reduce"(%v2179) ({
    ^bb0(%arad4gpdst: tensor<f32>, %arbd4gpdst: tensor<f32>):
      %araddd4gpdst = stablehlo.add %arad4gpdst, %arbd4gpdst : tensor<f32>
      stablehlo.return %araddd4gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<2048xf32>) -> tensor<2048xf32>
    %arnd4gpdst = stablehlo.constant dense<4.0> : tensor<2048xf32>
    %armeand4gpdst = stablehlo.divide %arsumd4gpdst, %arnd4gpdst : tensor<2048xf32>
    %v2180 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2181 = stablehlo.slice %armeand4gpdst [0:512] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2182 = stablehlo.slice %armeand4gpdst [512:1024] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2183 = stablehlo.slice %armeand4gpdst [1024:1536] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2184 = stablehlo.slice %armeand4gpdst [1536:2048] : (tensor<2048xf32>) -> tensor<512xf32>
    %v2185 = stablehlo.broadcast_in_dim %v2181, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2186 = stablehlo.broadcast_in_dim %v2182, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2187 = stablehlo.broadcast_in_dim %v2183, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2188 = stablehlo.broadcast_in_dim %v2184, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2189 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2190 = stablehlo.add %v2186, %v2189 : tensor<64x512x7x7xf32>
    %v2191 = stablehlo.rsqrt %v2190 : tensor<64x512x7x7xf32>
    %v2192 = stablehlo.subtract %v2180, %v2185 : tensor<64x512x7x7xf32>
    %v2193 = stablehlo.multiply %v2192, %v2191 : tensor<64x512x7x7xf32>
    %v2194 = stablehlo.broadcast_in_dim %d4gp, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2195 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2196 = stablehlo.multiply %v2194, %v2195 : tensor<64x512x7x7xf32>
    %v2197 = stablehlo.subtract %v2196, %v2187 : tensor<64x512x7x7xf32>
    %v2198 = stablehlo.multiply %v2193, %v2188 : tensor<64x512x7x7xf32>
    %v2199 = stablehlo.subtract %v2197, %v2198 : tensor<64x512x7x7xf32>
    %v2200 = stablehlo.multiply %v2191, %v2199 : tensor<64x512x7x7xf32>
    %v2201 = stablehlo.reshape %v2200 : (tensor<64x512x7x7xf32>) -> tensor<64x25088xf32>
    %v2202 = stablehlo.reshape %v2201 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2203 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2204 = stablehlo.pad %v2202, %v2203, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2205 = stablehlo.reverse %d4Wp, dims = [2, 3] : tensor<512x256x1x1xf32>
    %v2206 = stablehlo.transpose %v2205, dims = [1, 0, 2, 3] : (tensor<512x256x1x1xf32>) -> tensor<256x512x1x1xf32>
    %v2207 = stablehlo.convert %v2204 : (tensor<64x512x14x14xf32>) -> tensor<64x512x14x14xbf16>
    %v2208 = stablehlo.convert %v2206 : (tensor<256x512x1x1xf32>) -> tensor<256x512x1x1xbf16>
    %v2209 = stablehlo.convolution(%v2207, %v2208)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x512x14x14xbf16>, tensor<256x512x1x1xbf16>) -> tensor<64x256x14x14xbf16>
    %v2210 = stablehlo.convert %v2209 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2211 = stablehlo.reshape %v2210 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2212 = stablehlo.reshape %v2157 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2213 = stablehlo.reshape %v2211 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2214 = stablehlo.add %v2212, %v2213 : tensor<64x256x14x14xf32>
    %v2215 = stablehlo.reshape %v2214 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2216 = stablehlo.reshape %v1343 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2217 = stablehlo.reshape %v2147 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2218 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2219 = stablehlo.pad %v2217, %v2218, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2220 = stablehlo.transpose %v2216, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2221 = stablehlo.transpose %v2219, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2222 = stablehlo.convert %v2220 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2223 = stablehlo.convert %v2221 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2224 = stablehlo.convolution(%v2222, %v2223)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<256x512x3x3xbf16>
    %v2225 = stablehlo.convert %v2224 : (tensor<256x512x3x3xbf16>) -> tensor<256x512x3x3xf32>
    %v2226 = stablehlo.transpose %v2225, dims = [1, 0, 2, 3] : (tensor<256x512x3x3xf32>) -> tensor<512x256x3x3xf32>
    %v2227 = stablehlo.reshape %v1351 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2228 = stablehlo.slice %v1370 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2229 = stablehlo.slice %v1370 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2230 = stablehlo.broadcast_in_dim %v2228, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2231 = stablehlo.broadcast_in_dim %v2229, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2232 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2233 = stablehlo.add %v2231, %v2232 : tensor<64x512x7x7xf32>
    %v2234 = stablehlo.rsqrt %v2233 : tensor<64x512x7x7xf32>
    %v2235 = stablehlo.subtract %v2227, %v2230 : tensor<64x512x7x7xf32>
    %v2236 = stablehlo.multiply %v2235, %v2234 : tensor<64x512x7x7xf32>
    %v2237 = stablehlo.reshape %v2103 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2238 = stablehlo.multiply %v2237, %v2236 : tensor<64x512x7x7xf32>
    %v2239 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2240 = stablehlo.reduce(%v2238 init: %v2239) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2241 = stablehlo.reshape %v2103 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2242 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2243 = stablehlo.reduce(%v2241 init: %v2242) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2244 = stablehlo.reshape %v1387 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2245 = stablehlo.reshape %v2089 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2246 = stablehlo.transpose %v2244, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2247 = stablehlo.transpose %v2245, dims = [1, 0, 2, 3] : (tensor<64x512x7x7xf32>) -> tensor<512x64x7x7xf32>
    %v2248 = stablehlo.convert %v2246 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2249 = stablehlo.convert %v2247 : (tensor<512x64x7x7xf32>) -> tensor<512x64x7x7xbf16>
    %v2250 = stablehlo.convolution(%v2248, %v2249)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<512x64x7x7xbf16>, tensor<512x64x7x7xbf16>) -> tensor<512x512x3x3xbf16>
    %v2251 = stablehlo.convert %v2250 : (tensor<512x512x3x3xbf16>) -> tensor<512x512x3x3xf32>
    %v2252 = stablehlo.transpose %v2251, dims = [1, 0, 2, 3] : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %v2253 = stablehlo.reshape %v1395 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2254 = stablehlo.slice %v1414 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2255 = stablehlo.slice %v1414 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2256 = stablehlo.broadcast_in_dim %v2254, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2257 = stablehlo.broadcast_in_dim %v2255, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2258 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2259 = stablehlo.add %v2257, %v2258 : tensor<64x512x7x7xf32>
    %v2260 = stablehlo.rsqrt %v2259 : tensor<64x512x7x7xf32>
    %v2261 = stablehlo.subtract %v2253, %v2256 : tensor<64x512x7x7xf32>
    %v2262 = stablehlo.multiply %v2261, %v2260 : tensor<64x512x7x7xf32>
    %v2263 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2264 = stablehlo.multiply %v2263, %v2262 : tensor<64x512x7x7xf32>
    %v2265 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2266 = stablehlo.reduce(%v2264 init: %v2265) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2267 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2268 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2269 = stablehlo.reduce(%v2267 init: %v2268) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2270 = stablehlo.reshape %v1343 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2271 = stablehlo.reshape %v2201 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2272 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2273 = stablehlo.pad %v2271, %v2272, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<64x512x14x14xf32>
    %v2274 = stablehlo.transpose %v2270, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2275 = stablehlo.transpose %v2273, dims = [1, 0, 2, 3] : (tensor<64x512x14x14xf32>) -> tensor<512x64x14x14xf32>
    %v2276 = stablehlo.convert %v2274 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2277 = stablehlo.convert %v2275 : (tensor<512x64x14x14xf32>) -> tensor<512x64x14x14xbf16>
    %v2278 = stablehlo.convolution(%v2276, %v2277)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<512x64x14x14xbf16>) -> tensor<256x512x1x1xbf16>
    %v2279 = stablehlo.convert %v2278 : (tensor<256x512x1x1xbf16>) -> tensor<256x512x1x1xf32>
    %v2280 = stablehlo.transpose %v2279, dims = [1, 0, 2, 3] : (tensor<256x512x1x1xf32>) -> tensor<512x256x1x1xf32>
    %v2281 = stablehlo.reshape %v1437 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2282 = stablehlo.slice %v1456 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2283 = stablehlo.slice %v1456 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v2284 = stablehlo.broadcast_in_dim %v2282, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2285 = stablehlo.broadcast_in_dim %v2283, dims = [1] : (tensor<512xf32>) -> tensor<64x512x7x7xf32>
    %v2286 = stablehlo.constant dense<1.0e-05> : tensor<64x512x7x7xf32>
    %v2287 = stablehlo.add %v2285, %v2286 : tensor<64x512x7x7xf32>
    %v2288 = stablehlo.rsqrt %v2287 : tensor<64x512x7x7xf32>
    %v2289 = stablehlo.subtract %v2281, %v2284 : tensor<64x512x7x7xf32>
    %v2290 = stablehlo.multiply %v2289, %v2288 : tensor<64x512x7x7xf32>
    %v2291 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2292 = stablehlo.multiply %v2291, %v2290 : tensor<64x512x7x7xf32>
    %v2293 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2294 = stablehlo.reduce(%v2292 init: %v2293) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2295 = stablehlo.reshape %v2045 : (tensor<64x25088xf32>) -> tensor<64x512x7x7xf32>
    %v2296 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2297 = stablehlo.reduce(%v2295 init: %v2296) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x512x7x7xf32>, tensor<f32>) -> tensor<512xf32>
    %v2298 = stablehlo.reshape %v2215 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2299 = stablehlo.reshape %v1339 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2300 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2301 = stablehlo.compare GT, %v2299, %v2300 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2302 = stablehlo.select %v2301, %v2298, %v2300 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2303 = stablehlo.reshape %v2302 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2304 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2305 = stablehlo.slice %v1320 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2306 = stablehlo.slice %v1320 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2307 = stablehlo.broadcast_in_dim %v2305, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2308 = stablehlo.broadcast_in_dim %v2306, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2309 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2310 = stablehlo.add %v2308, %v2309 : tensor<64x256x14x14xf32>
    %v2311 = stablehlo.rsqrt %v2310 : tensor<64x256x14x14xf32>
    %v2312 = stablehlo.subtract %v2304, %v2307 : tensor<64x256x14x14xf32>
    %v2313 = stablehlo.multiply %v2312, %v2311 : tensor<64x256x14x14xf32>
    %v2314 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2315 = stablehlo.reshape %v2303 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2316 = stablehlo.multiply %v2314, %v2315 : tensor<64x256x14x14xf32>
    %v2317 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2318 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2319 = stablehlo.reduce(%v2316 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2320 = stablehlo.divide %v2319, %v2318 : tensor<256xf32>
    %v2321 = stablehlo.multiply %v2313, %v2316 : tensor<64x256x14x14xf32>
    %v2322 = stablehlo.reduce(%v2321 init: %v2317) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2323 = stablehlo.divide %v2322, %v2318 : tensor<256xf32>
    %v2324 = stablehlo.concatenate %v2320, %v2323, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2325 = stablehlo.concatenate %v1320, %v2324, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g2dst = "stablehlo.all_reduce"(%v2325) ({
    ^bb0(%aras3b4g2dst: tensor<f32>, %arbs3b4g2dst: tensor<f32>):
      %aradds3b4g2dst = stablehlo.add %aras3b4g2dst, %arbs3b4g2dst : tensor<f32>
      stablehlo.return %aradds3b4g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g2dst = stablehlo.divide %arsums3b4g2dst, %arns3b4g2dst : tensor<1024xf32>
    %v2326 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2327 = stablehlo.slice %armeans3b4g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2328 = stablehlo.slice %armeans3b4g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2329 = stablehlo.slice %armeans3b4g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2330 = stablehlo.slice %armeans3b4g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2331 = stablehlo.broadcast_in_dim %v2327, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2332 = stablehlo.broadcast_in_dim %v2328, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2333 = stablehlo.broadcast_in_dim %v2329, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2334 = stablehlo.broadcast_in_dim %v2330, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2335 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2336 = stablehlo.add %v2332, %v2335 : tensor<64x256x14x14xf32>
    %v2337 = stablehlo.rsqrt %v2336 : tensor<64x256x14x14xf32>
    %v2338 = stablehlo.subtract %v2326, %v2331 : tensor<64x256x14x14xf32>
    %v2339 = stablehlo.multiply %v2338, %v2337 : tensor<64x256x14x14xf32>
    %v2340 = stablehlo.broadcast_in_dim %s3b4g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2341 = stablehlo.reshape %v2303 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2342 = stablehlo.multiply %v2340, %v2341 : tensor<64x256x14x14xf32>
    %v2343 = stablehlo.subtract %v2342, %v2333 : tensor<64x256x14x14xf32>
    %v2344 = stablehlo.multiply %v2339, %v2334 : tensor<64x256x14x14xf32>
    %v2345 = stablehlo.subtract %v2343, %v2344 : tensor<64x256x14x14xf32>
    %v2346 = stablehlo.multiply %v2337, %v2345 : tensor<64x256x14x14xf32>
    %v2347 = stablehlo.reshape %v2346 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2348 = stablehlo.reshape %v2347 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2349 = stablehlo.reverse %s3b4W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2350 = stablehlo.transpose %v2349, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2351 = stablehlo.convert %v2348 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2352 = stablehlo.convert %v2350 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2353 = stablehlo.convolution(%v2351, %v2352)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2354 = stablehlo.convert %v2353 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2355 = stablehlo.reshape %v2354 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2356 = stablehlo.reshape %v2355 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2357 = stablehlo.reshape %v1291 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2358 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2359 = stablehlo.compare GT, %v2357, %v2358 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2360 = stablehlo.select %v2359, %v2356, %v2358 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2361 = stablehlo.reshape %v2360 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2362 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2363 = stablehlo.slice %v1276 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2364 = stablehlo.slice %v1276 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2365 = stablehlo.broadcast_in_dim %v2363, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2366 = stablehlo.broadcast_in_dim %v2364, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2367 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2368 = stablehlo.add %v2366, %v2367 : tensor<64x256x14x14xf32>
    %v2369 = stablehlo.rsqrt %v2368 : tensor<64x256x14x14xf32>
    %v2370 = stablehlo.subtract %v2362, %v2365 : tensor<64x256x14x14xf32>
    %v2371 = stablehlo.multiply %v2370, %v2369 : tensor<64x256x14x14xf32>
    %v2372 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2373 = stablehlo.reshape %v2361 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2374 = stablehlo.multiply %v2372, %v2373 : tensor<64x256x14x14xf32>
    %v2375 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2376 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2377 = stablehlo.reduce(%v2374 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2378 = stablehlo.divide %v2377, %v2376 : tensor<256xf32>
    %v2379 = stablehlo.multiply %v2371, %v2374 : tensor<64x256x14x14xf32>
    %v2380 = stablehlo.reduce(%v2379 init: %v2375) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2381 = stablehlo.divide %v2380, %v2376 : tensor<256xf32>
    %v2382 = stablehlo.concatenate %v2378, %v2381, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2383 = stablehlo.concatenate %v1276, %v2382, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b4g1dst = "stablehlo.all_reduce"(%v2383) ({
    ^bb0(%aras3b4g1dst: tensor<f32>, %arbs3b4g1dst: tensor<f32>):
      %aradds3b4g1dst = stablehlo.add %aras3b4g1dst, %arbs3b4g1dst : tensor<f32>
      stablehlo.return %aradds3b4g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b4g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b4g1dst = stablehlo.divide %arsums3b4g1dst, %arns3b4g1dst : tensor<1024xf32>
    %v2384 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2385 = stablehlo.slice %armeans3b4g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2386 = stablehlo.slice %armeans3b4g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2387 = stablehlo.slice %armeans3b4g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2388 = stablehlo.slice %armeans3b4g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2389 = stablehlo.broadcast_in_dim %v2385, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2390 = stablehlo.broadcast_in_dim %v2386, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2391 = stablehlo.broadcast_in_dim %v2387, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2392 = stablehlo.broadcast_in_dim %v2388, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2393 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2394 = stablehlo.add %v2390, %v2393 : tensor<64x256x14x14xf32>
    %v2395 = stablehlo.rsqrt %v2394 : tensor<64x256x14x14xf32>
    %v2396 = stablehlo.subtract %v2384, %v2389 : tensor<64x256x14x14xf32>
    %v2397 = stablehlo.multiply %v2396, %v2395 : tensor<64x256x14x14xf32>
    %v2398 = stablehlo.broadcast_in_dim %s3b4g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2399 = stablehlo.reshape %v2361 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2400 = stablehlo.multiply %v2398, %v2399 : tensor<64x256x14x14xf32>
    %v2401 = stablehlo.subtract %v2400, %v2391 : tensor<64x256x14x14xf32>
    %v2402 = stablehlo.multiply %v2397, %v2392 : tensor<64x256x14x14xf32>
    %v2403 = stablehlo.subtract %v2401, %v2402 : tensor<64x256x14x14xf32>
    %v2404 = stablehlo.multiply %v2395, %v2403 : tensor<64x256x14x14xf32>
    %v2405 = stablehlo.reshape %v2404 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2406 = stablehlo.reshape %v2405 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2407 = stablehlo.reverse %s3b4W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2408 = stablehlo.transpose %v2407, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2409 = stablehlo.convert %v2406 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2410 = stablehlo.convert %v2408 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2411 = stablehlo.convolution(%v2409, %v2410)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2412 = stablehlo.convert %v2411 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2413 = stablehlo.reshape %v2412 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2414 = stablehlo.reshape %v2413 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2415 = stablehlo.reshape %v2303 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2416 = stablehlo.add %v2414, %v2415 : tensor<64x256x14x14xf32>
    %v2417 = stablehlo.reshape %v2416 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2418 = stablehlo.reshape %v1249 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2419 = stablehlo.reshape %v2405 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2420 = stablehlo.transpose %v2418, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2421 = stablehlo.transpose %v2419, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2422 = stablehlo.convert %v2420 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2423 = stablehlo.convert %v2421 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2424 = stablehlo.convolution(%v2422, %v2423)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2425 = stablehlo.convert %v2424 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2426 = stablehlo.transpose %v2425, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2427 = stablehlo.reshape %v1257 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2428 = stablehlo.slice %v1276 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2429 = stablehlo.slice %v1276 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2430 = stablehlo.broadcast_in_dim %v2428, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2431 = stablehlo.broadcast_in_dim %v2429, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2432 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2433 = stablehlo.add %v2431, %v2432 : tensor<64x256x14x14xf32>
    %v2434 = stablehlo.rsqrt %v2433 : tensor<64x256x14x14xf32>
    %v2435 = stablehlo.subtract %v2427, %v2430 : tensor<64x256x14x14xf32>
    %v2436 = stablehlo.multiply %v2435, %v2434 : tensor<64x256x14x14xf32>
    %v2437 = stablehlo.reshape %v2361 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2438 = stablehlo.multiply %v2437, %v2436 : tensor<64x256x14x14xf32>
    %v2439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2440 = stablehlo.reduce(%v2438 init: %v2439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2441 = stablehlo.reshape %v2361 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2442 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2443 = stablehlo.reduce(%v2441 init: %v2442) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2444 = stablehlo.reshape %v1293 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2445 = stablehlo.reshape %v2347 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2446 = stablehlo.transpose %v2444, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2447 = stablehlo.transpose %v2445, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2448 = stablehlo.convert %v2446 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2449 = stablehlo.convert %v2447 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2450 = stablehlo.convolution(%v2448, %v2449)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2451 = stablehlo.convert %v2450 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2452 = stablehlo.transpose %v2451, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2453 = stablehlo.reshape %v1301 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2454 = stablehlo.slice %v1320 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2455 = stablehlo.slice %v1320 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2456 = stablehlo.broadcast_in_dim %v2454, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2457 = stablehlo.broadcast_in_dim %v2455, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2458 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2459 = stablehlo.add %v2457, %v2458 : tensor<64x256x14x14xf32>
    %v2460 = stablehlo.rsqrt %v2459 : tensor<64x256x14x14xf32>
    %v2461 = stablehlo.subtract %v2453, %v2456 : tensor<64x256x14x14xf32>
    %v2462 = stablehlo.multiply %v2461, %v2460 : tensor<64x256x14x14xf32>
    %v2463 = stablehlo.reshape %v2303 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2464 = stablehlo.multiply %v2463, %v2462 : tensor<64x256x14x14xf32>
    %v2465 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2466 = stablehlo.reduce(%v2464 init: %v2465) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2467 = stablehlo.reshape %v2303 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2468 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2469 = stablehlo.reduce(%v2467 init: %v2468) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2470 = stablehlo.reshape %v2417 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2471 = stablehlo.reshape %v1245 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2472 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2473 = stablehlo.compare GT, %v2471, %v2472 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2474 = stablehlo.select %v2473, %v2470, %v2472 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2475 = stablehlo.reshape %v2474 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2476 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2477 = stablehlo.slice %v1226 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2478 = stablehlo.slice %v1226 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2479 = stablehlo.broadcast_in_dim %v2477, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2480 = stablehlo.broadcast_in_dim %v2478, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2481 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2482 = stablehlo.add %v2480, %v2481 : tensor<64x256x14x14xf32>
    %v2483 = stablehlo.rsqrt %v2482 : tensor<64x256x14x14xf32>
    %v2484 = stablehlo.subtract %v2476, %v2479 : tensor<64x256x14x14xf32>
    %v2485 = stablehlo.multiply %v2484, %v2483 : tensor<64x256x14x14xf32>
    %v2486 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2487 = stablehlo.reshape %v2475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2488 = stablehlo.multiply %v2486, %v2487 : tensor<64x256x14x14xf32>
    %v2489 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2490 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2491 = stablehlo.reduce(%v2488 init: %v2489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2492 = stablehlo.divide %v2491, %v2490 : tensor<256xf32>
    %v2493 = stablehlo.multiply %v2485, %v2488 : tensor<64x256x14x14xf32>
    %v2494 = stablehlo.reduce(%v2493 init: %v2489) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2495 = stablehlo.divide %v2494, %v2490 : tensor<256xf32>
    %v2496 = stablehlo.concatenate %v2492, %v2495, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2497 = stablehlo.concatenate %v1226, %v2496, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g2dst = "stablehlo.all_reduce"(%v2497) ({
    ^bb0(%aras3b3g2dst: tensor<f32>, %arbs3b3g2dst: tensor<f32>):
      %aradds3b3g2dst = stablehlo.add %aras3b3g2dst, %arbs3b3g2dst : tensor<f32>
      stablehlo.return %aradds3b3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g2dst = stablehlo.divide %arsums3b3g2dst, %arns3b3g2dst : tensor<1024xf32>
    %v2498 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2499 = stablehlo.slice %armeans3b3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2500 = stablehlo.slice %armeans3b3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2501 = stablehlo.slice %armeans3b3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2502 = stablehlo.slice %armeans3b3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2503 = stablehlo.broadcast_in_dim %v2499, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2504 = stablehlo.broadcast_in_dim %v2500, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2505 = stablehlo.broadcast_in_dim %v2501, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2506 = stablehlo.broadcast_in_dim %v2502, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2507 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2508 = stablehlo.add %v2504, %v2507 : tensor<64x256x14x14xf32>
    %v2509 = stablehlo.rsqrt %v2508 : tensor<64x256x14x14xf32>
    %v2510 = stablehlo.subtract %v2498, %v2503 : tensor<64x256x14x14xf32>
    %v2511 = stablehlo.multiply %v2510, %v2509 : tensor<64x256x14x14xf32>
    %v2512 = stablehlo.broadcast_in_dim %s3b3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2513 = stablehlo.reshape %v2475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2514 = stablehlo.multiply %v2512, %v2513 : tensor<64x256x14x14xf32>
    %v2515 = stablehlo.subtract %v2514, %v2505 : tensor<64x256x14x14xf32>
    %v2516 = stablehlo.multiply %v2511, %v2506 : tensor<64x256x14x14xf32>
    %v2517 = stablehlo.subtract %v2515, %v2516 : tensor<64x256x14x14xf32>
    %v2518 = stablehlo.multiply %v2509, %v2517 : tensor<64x256x14x14xf32>
    %v2519 = stablehlo.reshape %v2518 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2520 = stablehlo.reshape %v2519 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2521 = stablehlo.reverse %s3b3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2522 = stablehlo.transpose %v2521, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2523 = stablehlo.convert %v2520 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2524 = stablehlo.convert %v2522 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2525 = stablehlo.convolution(%v2523, %v2524)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2526 = stablehlo.convert %v2525 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2527 = stablehlo.reshape %v2526 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2528 = stablehlo.reshape %v2527 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2529 = stablehlo.reshape %v1197 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2530 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2531 = stablehlo.compare GT, %v2529, %v2530 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2532 = stablehlo.select %v2531, %v2528, %v2530 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2533 = stablehlo.reshape %v2532 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2534 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2535 = stablehlo.slice %v1182 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2536 = stablehlo.slice %v1182 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2537 = stablehlo.broadcast_in_dim %v2535, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2538 = stablehlo.broadcast_in_dim %v2536, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2539 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2540 = stablehlo.add %v2538, %v2539 : tensor<64x256x14x14xf32>
    %v2541 = stablehlo.rsqrt %v2540 : tensor<64x256x14x14xf32>
    %v2542 = stablehlo.subtract %v2534, %v2537 : tensor<64x256x14x14xf32>
    %v2543 = stablehlo.multiply %v2542, %v2541 : tensor<64x256x14x14xf32>
    %v2544 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2545 = stablehlo.reshape %v2533 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2546 = stablehlo.multiply %v2544, %v2545 : tensor<64x256x14x14xf32>
    %v2547 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2548 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2549 = stablehlo.reduce(%v2546 init: %v2547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2550 = stablehlo.divide %v2549, %v2548 : tensor<256xf32>
    %v2551 = stablehlo.multiply %v2543, %v2546 : tensor<64x256x14x14xf32>
    %v2552 = stablehlo.reduce(%v2551 init: %v2547) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2553 = stablehlo.divide %v2552, %v2548 : tensor<256xf32>
    %v2554 = stablehlo.concatenate %v2550, %v2553, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2555 = stablehlo.concatenate %v1182, %v2554, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b3g1dst = "stablehlo.all_reduce"(%v2555) ({
    ^bb0(%aras3b3g1dst: tensor<f32>, %arbs3b3g1dst: tensor<f32>):
      %aradds3b3g1dst = stablehlo.add %aras3b3g1dst, %arbs3b3g1dst : tensor<f32>
      stablehlo.return %aradds3b3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b3g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b3g1dst = stablehlo.divide %arsums3b3g1dst, %arns3b3g1dst : tensor<1024xf32>
    %v2556 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2557 = stablehlo.slice %armeans3b3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2558 = stablehlo.slice %armeans3b3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2559 = stablehlo.slice %armeans3b3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2560 = stablehlo.slice %armeans3b3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2561 = stablehlo.broadcast_in_dim %v2557, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2562 = stablehlo.broadcast_in_dim %v2558, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2563 = stablehlo.broadcast_in_dim %v2559, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2564 = stablehlo.broadcast_in_dim %v2560, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2565 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2566 = stablehlo.add %v2562, %v2565 : tensor<64x256x14x14xf32>
    %v2567 = stablehlo.rsqrt %v2566 : tensor<64x256x14x14xf32>
    %v2568 = stablehlo.subtract %v2556, %v2561 : tensor<64x256x14x14xf32>
    %v2569 = stablehlo.multiply %v2568, %v2567 : tensor<64x256x14x14xf32>
    %v2570 = stablehlo.broadcast_in_dim %s3b3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2571 = stablehlo.reshape %v2533 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2572 = stablehlo.multiply %v2570, %v2571 : tensor<64x256x14x14xf32>
    %v2573 = stablehlo.subtract %v2572, %v2563 : tensor<64x256x14x14xf32>
    %v2574 = stablehlo.multiply %v2569, %v2564 : tensor<64x256x14x14xf32>
    %v2575 = stablehlo.subtract %v2573, %v2574 : tensor<64x256x14x14xf32>
    %v2576 = stablehlo.multiply %v2567, %v2575 : tensor<64x256x14x14xf32>
    %v2577 = stablehlo.reshape %v2576 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2578 = stablehlo.reshape %v2577 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2579 = stablehlo.reverse %s3b3W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2580 = stablehlo.transpose %v2579, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2581 = stablehlo.convert %v2578 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2582 = stablehlo.convert %v2580 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2583 = stablehlo.convolution(%v2581, %v2582)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2584 = stablehlo.convert %v2583 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2585 = stablehlo.reshape %v2584 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2586 = stablehlo.reshape %v2585 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2587 = stablehlo.reshape %v2475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2588 = stablehlo.add %v2586, %v2587 : tensor<64x256x14x14xf32>
    %v2589 = stablehlo.reshape %v2588 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2590 = stablehlo.reshape %v1155 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2591 = stablehlo.reshape %v2577 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2592 = stablehlo.transpose %v2590, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2593 = stablehlo.transpose %v2591, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2594 = stablehlo.convert %v2592 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2595 = stablehlo.convert %v2593 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2596 = stablehlo.convolution(%v2594, %v2595)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2597 = stablehlo.convert %v2596 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2598 = stablehlo.transpose %v2597, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2599 = stablehlo.reshape %v1163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2600 = stablehlo.slice %v1182 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2601 = stablehlo.slice %v1182 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2602 = stablehlo.broadcast_in_dim %v2600, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2603 = stablehlo.broadcast_in_dim %v2601, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2604 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2605 = stablehlo.add %v2603, %v2604 : tensor<64x256x14x14xf32>
    %v2606 = stablehlo.rsqrt %v2605 : tensor<64x256x14x14xf32>
    %v2607 = stablehlo.subtract %v2599, %v2602 : tensor<64x256x14x14xf32>
    %v2608 = stablehlo.multiply %v2607, %v2606 : tensor<64x256x14x14xf32>
    %v2609 = stablehlo.reshape %v2533 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2610 = stablehlo.multiply %v2609, %v2608 : tensor<64x256x14x14xf32>
    %v2611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2612 = stablehlo.reduce(%v2610 init: %v2611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2613 = stablehlo.reshape %v2533 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2614 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2615 = stablehlo.reduce(%v2613 init: %v2614) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2616 = stablehlo.reshape %v1199 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2617 = stablehlo.reshape %v2519 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2618 = stablehlo.transpose %v2616, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2619 = stablehlo.transpose %v2617, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2620 = stablehlo.convert %v2618 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2621 = stablehlo.convert %v2619 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2622 = stablehlo.convolution(%v2620, %v2621)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2623 = stablehlo.convert %v2622 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2624 = stablehlo.transpose %v2623, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2625 = stablehlo.reshape %v1207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2626 = stablehlo.slice %v1226 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2627 = stablehlo.slice %v1226 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2628 = stablehlo.broadcast_in_dim %v2626, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2629 = stablehlo.broadcast_in_dim %v2627, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2630 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2631 = stablehlo.add %v2629, %v2630 : tensor<64x256x14x14xf32>
    %v2632 = stablehlo.rsqrt %v2631 : tensor<64x256x14x14xf32>
    %v2633 = stablehlo.subtract %v2625, %v2628 : tensor<64x256x14x14xf32>
    %v2634 = stablehlo.multiply %v2633, %v2632 : tensor<64x256x14x14xf32>
    %v2635 = stablehlo.reshape %v2475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2636 = stablehlo.multiply %v2635, %v2634 : tensor<64x256x14x14xf32>
    %v2637 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2638 = stablehlo.reduce(%v2636 init: %v2637) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2639 = stablehlo.reshape %v2475 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2640 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2641 = stablehlo.reduce(%v2639 init: %v2640) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2642 = stablehlo.reshape %v2589 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2643 = stablehlo.reshape %v1151 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2644 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2645 = stablehlo.compare GT, %v2643, %v2644 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2646 = stablehlo.select %v2645, %v2642, %v2644 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2647 = stablehlo.reshape %v2646 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2648 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2649 = stablehlo.slice %v1132 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2650 = stablehlo.slice %v1132 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2651 = stablehlo.broadcast_in_dim %v2649, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2652 = stablehlo.broadcast_in_dim %v2650, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2653 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2654 = stablehlo.add %v2652, %v2653 : tensor<64x256x14x14xf32>
    %v2655 = stablehlo.rsqrt %v2654 : tensor<64x256x14x14xf32>
    %v2656 = stablehlo.subtract %v2648, %v2651 : tensor<64x256x14x14xf32>
    %v2657 = stablehlo.multiply %v2656, %v2655 : tensor<64x256x14x14xf32>
    %v2658 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2659 = stablehlo.reshape %v2647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2660 = stablehlo.multiply %v2658, %v2659 : tensor<64x256x14x14xf32>
    %v2661 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2662 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2663 = stablehlo.reduce(%v2660 init: %v2661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2664 = stablehlo.divide %v2663, %v2662 : tensor<256xf32>
    %v2665 = stablehlo.multiply %v2657, %v2660 : tensor<64x256x14x14xf32>
    %v2666 = stablehlo.reduce(%v2665 init: %v2661) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2667 = stablehlo.divide %v2666, %v2662 : tensor<256xf32>
    %v2668 = stablehlo.concatenate %v2664, %v2667, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2669 = stablehlo.concatenate %v1132, %v2668, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g2dst = "stablehlo.all_reduce"(%v2669) ({
    ^bb0(%aras3b2g2dst: tensor<f32>, %arbs3b2g2dst: tensor<f32>):
      %aradds3b2g2dst = stablehlo.add %aras3b2g2dst, %arbs3b2g2dst : tensor<f32>
      stablehlo.return %aradds3b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g2dst = stablehlo.divide %arsums3b2g2dst, %arns3b2g2dst : tensor<1024xf32>
    %v2670 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2671 = stablehlo.slice %armeans3b2g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2672 = stablehlo.slice %armeans3b2g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2673 = stablehlo.slice %armeans3b2g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2674 = stablehlo.slice %armeans3b2g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2675 = stablehlo.broadcast_in_dim %v2671, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2676 = stablehlo.broadcast_in_dim %v2672, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2677 = stablehlo.broadcast_in_dim %v2673, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2678 = stablehlo.broadcast_in_dim %v2674, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2679 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2680 = stablehlo.add %v2676, %v2679 : tensor<64x256x14x14xf32>
    %v2681 = stablehlo.rsqrt %v2680 : tensor<64x256x14x14xf32>
    %v2682 = stablehlo.subtract %v2670, %v2675 : tensor<64x256x14x14xf32>
    %v2683 = stablehlo.multiply %v2682, %v2681 : tensor<64x256x14x14xf32>
    %v2684 = stablehlo.broadcast_in_dim %s3b2g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2685 = stablehlo.reshape %v2647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2686 = stablehlo.multiply %v2684, %v2685 : tensor<64x256x14x14xf32>
    %v2687 = stablehlo.subtract %v2686, %v2677 : tensor<64x256x14x14xf32>
    %v2688 = stablehlo.multiply %v2683, %v2678 : tensor<64x256x14x14xf32>
    %v2689 = stablehlo.subtract %v2687, %v2688 : tensor<64x256x14x14xf32>
    %v2690 = stablehlo.multiply %v2681, %v2689 : tensor<64x256x14x14xf32>
    %v2691 = stablehlo.reshape %v2690 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2692 = stablehlo.reshape %v2691 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2693 = stablehlo.reverse %s3b2W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2694 = stablehlo.transpose %v2693, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2695 = stablehlo.convert %v2692 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2696 = stablehlo.convert %v2694 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2697 = stablehlo.convolution(%v2695, %v2696)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2698 = stablehlo.convert %v2697 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2699 = stablehlo.reshape %v2698 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2700 = stablehlo.reshape %v2699 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2701 = stablehlo.reshape %v1103 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2702 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2703 = stablehlo.compare GT, %v2701, %v2702 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2704 = stablehlo.select %v2703, %v2700, %v2702 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2705 = stablehlo.reshape %v2704 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2706 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2707 = stablehlo.slice %v1088 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2708 = stablehlo.slice %v1088 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2709 = stablehlo.broadcast_in_dim %v2707, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2710 = stablehlo.broadcast_in_dim %v2708, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2711 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2712 = stablehlo.add %v2710, %v2711 : tensor<64x256x14x14xf32>
    %v2713 = stablehlo.rsqrt %v2712 : tensor<64x256x14x14xf32>
    %v2714 = stablehlo.subtract %v2706, %v2709 : tensor<64x256x14x14xf32>
    %v2715 = stablehlo.multiply %v2714, %v2713 : tensor<64x256x14x14xf32>
    %v2716 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2717 = stablehlo.reshape %v2705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2718 = stablehlo.multiply %v2716, %v2717 : tensor<64x256x14x14xf32>
    %v2719 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2720 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2721 = stablehlo.reduce(%v2718 init: %v2719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2722 = stablehlo.divide %v2721, %v2720 : tensor<256xf32>
    %v2723 = stablehlo.multiply %v2715, %v2718 : tensor<64x256x14x14xf32>
    %v2724 = stablehlo.reduce(%v2723 init: %v2719) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2725 = stablehlo.divide %v2724, %v2720 : tensor<256xf32>
    %v2726 = stablehlo.concatenate %v2722, %v2725, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2727 = stablehlo.concatenate %v1088, %v2726, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b2g1dst = "stablehlo.all_reduce"(%v2727) ({
    ^bb0(%aras3b2g1dst: tensor<f32>, %arbs3b2g1dst: tensor<f32>):
      %aradds3b2g1dst = stablehlo.add %aras3b2g1dst, %arbs3b2g1dst : tensor<f32>
      stablehlo.return %aradds3b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b2g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b2g1dst = stablehlo.divide %arsums3b2g1dst, %arns3b2g1dst : tensor<1024xf32>
    %v2728 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2729 = stablehlo.slice %armeans3b2g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2730 = stablehlo.slice %armeans3b2g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2731 = stablehlo.slice %armeans3b2g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2732 = stablehlo.slice %armeans3b2g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2733 = stablehlo.broadcast_in_dim %v2729, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2734 = stablehlo.broadcast_in_dim %v2730, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2735 = stablehlo.broadcast_in_dim %v2731, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2736 = stablehlo.broadcast_in_dim %v2732, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2737 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2738 = stablehlo.add %v2734, %v2737 : tensor<64x256x14x14xf32>
    %v2739 = stablehlo.rsqrt %v2738 : tensor<64x256x14x14xf32>
    %v2740 = stablehlo.subtract %v2728, %v2733 : tensor<64x256x14x14xf32>
    %v2741 = stablehlo.multiply %v2740, %v2739 : tensor<64x256x14x14xf32>
    %v2742 = stablehlo.broadcast_in_dim %s3b2g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2743 = stablehlo.reshape %v2705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2744 = stablehlo.multiply %v2742, %v2743 : tensor<64x256x14x14xf32>
    %v2745 = stablehlo.subtract %v2744, %v2735 : tensor<64x256x14x14xf32>
    %v2746 = stablehlo.multiply %v2741, %v2736 : tensor<64x256x14x14xf32>
    %v2747 = stablehlo.subtract %v2745, %v2746 : tensor<64x256x14x14xf32>
    %v2748 = stablehlo.multiply %v2739, %v2747 : tensor<64x256x14x14xf32>
    %v2749 = stablehlo.reshape %v2748 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2750 = stablehlo.reshape %v2749 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2751 = stablehlo.reverse %s3b2W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2752 = stablehlo.transpose %v2751, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2753 = stablehlo.convert %v2750 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2754 = stablehlo.convert %v2752 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2755 = stablehlo.convolution(%v2753, %v2754)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2756 = stablehlo.convert %v2755 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2757 = stablehlo.reshape %v2756 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2758 = stablehlo.reshape %v2757 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2759 = stablehlo.reshape %v2647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2760 = stablehlo.add %v2758, %v2759 : tensor<64x256x14x14xf32>
    %v2761 = stablehlo.reshape %v2760 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2762 = stablehlo.reshape %v1061 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2763 = stablehlo.reshape %v2749 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2764 = stablehlo.transpose %v2762, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2765 = stablehlo.transpose %v2763, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2766 = stablehlo.convert %v2764 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2767 = stablehlo.convert %v2765 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2768 = stablehlo.convolution(%v2766, %v2767)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2769 = stablehlo.convert %v2768 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2770 = stablehlo.transpose %v2769, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2771 = stablehlo.reshape %v1069 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2772 = stablehlo.slice %v1088 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2773 = stablehlo.slice %v1088 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2774 = stablehlo.broadcast_in_dim %v2772, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2775 = stablehlo.broadcast_in_dim %v2773, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2776 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2777 = stablehlo.add %v2775, %v2776 : tensor<64x256x14x14xf32>
    %v2778 = stablehlo.rsqrt %v2777 : tensor<64x256x14x14xf32>
    %v2779 = stablehlo.subtract %v2771, %v2774 : tensor<64x256x14x14xf32>
    %v2780 = stablehlo.multiply %v2779, %v2778 : tensor<64x256x14x14xf32>
    %v2781 = stablehlo.reshape %v2705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2782 = stablehlo.multiply %v2781, %v2780 : tensor<64x256x14x14xf32>
    %v2783 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2784 = stablehlo.reduce(%v2782 init: %v2783) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2785 = stablehlo.reshape %v2705 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2786 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2787 = stablehlo.reduce(%v2785 init: %v2786) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2788 = stablehlo.reshape %v1105 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2789 = stablehlo.reshape %v2691 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2790 = stablehlo.transpose %v2788, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2791 = stablehlo.transpose %v2789, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2792 = stablehlo.convert %v2790 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2793 = stablehlo.convert %v2791 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2794 = stablehlo.convolution(%v2792, %v2793)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2795 = stablehlo.convert %v2794 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2796 = stablehlo.transpose %v2795, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2797 = stablehlo.reshape %v1113 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2798 = stablehlo.slice %v1132 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2799 = stablehlo.slice %v1132 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2800 = stablehlo.broadcast_in_dim %v2798, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2801 = stablehlo.broadcast_in_dim %v2799, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2802 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2803 = stablehlo.add %v2801, %v2802 : tensor<64x256x14x14xf32>
    %v2804 = stablehlo.rsqrt %v2803 : tensor<64x256x14x14xf32>
    %v2805 = stablehlo.subtract %v2797, %v2800 : tensor<64x256x14x14xf32>
    %v2806 = stablehlo.multiply %v2805, %v2804 : tensor<64x256x14x14xf32>
    %v2807 = stablehlo.reshape %v2647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2808 = stablehlo.multiply %v2807, %v2806 : tensor<64x256x14x14xf32>
    %v2809 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2810 = stablehlo.reduce(%v2808 init: %v2809) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2811 = stablehlo.reshape %v2647 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2812 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2813 = stablehlo.reduce(%v2811 init: %v2812) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2814 = stablehlo.reshape %v2761 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2815 = stablehlo.reshape %v1057 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2816 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2817 = stablehlo.compare GT, %v2815, %v2816 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2818 = stablehlo.select %v2817, %v2814, %v2816 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2819 = stablehlo.reshape %v2818 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2820 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2821 = stablehlo.slice %v1038 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2822 = stablehlo.slice %v1038 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2823 = stablehlo.broadcast_in_dim %v2821, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2824 = stablehlo.broadcast_in_dim %v2822, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2825 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2826 = stablehlo.add %v2824, %v2825 : tensor<64x256x14x14xf32>
    %v2827 = stablehlo.rsqrt %v2826 : tensor<64x256x14x14xf32>
    %v2828 = stablehlo.subtract %v2820, %v2823 : tensor<64x256x14x14xf32>
    %v2829 = stablehlo.multiply %v2828, %v2827 : tensor<64x256x14x14xf32>
    %v2830 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2831 = stablehlo.reshape %v2819 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2832 = stablehlo.multiply %v2830, %v2831 : tensor<64x256x14x14xf32>
    %v2833 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2834 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2835 = stablehlo.reduce(%v2832 init: %v2833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2836 = stablehlo.divide %v2835, %v2834 : tensor<256xf32>
    %v2837 = stablehlo.multiply %v2829, %v2832 : tensor<64x256x14x14xf32>
    %v2838 = stablehlo.reduce(%v2837 init: %v2833) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2839 = stablehlo.divide %v2838, %v2834 : tensor<256xf32>
    %v2840 = stablehlo.concatenate %v2836, %v2839, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2841 = stablehlo.concatenate %v1038, %v2840, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g2dst = "stablehlo.all_reduce"(%v2841) ({
    ^bb0(%aras3b1g2dst: tensor<f32>, %arbs3b1g2dst: tensor<f32>):
      %aradds3b1g2dst = stablehlo.add %aras3b1g2dst, %arbs3b1g2dst : tensor<f32>
      stablehlo.return %aradds3b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g2dst = stablehlo.divide %arsums3b1g2dst, %arns3b1g2dst : tensor<1024xf32>
    %v2842 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2843 = stablehlo.slice %armeans3b1g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2844 = stablehlo.slice %armeans3b1g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2845 = stablehlo.slice %armeans3b1g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2846 = stablehlo.slice %armeans3b1g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2847 = stablehlo.broadcast_in_dim %v2843, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2848 = stablehlo.broadcast_in_dim %v2844, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2849 = stablehlo.broadcast_in_dim %v2845, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2850 = stablehlo.broadcast_in_dim %v2846, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2851 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2852 = stablehlo.add %v2848, %v2851 : tensor<64x256x14x14xf32>
    %v2853 = stablehlo.rsqrt %v2852 : tensor<64x256x14x14xf32>
    %v2854 = stablehlo.subtract %v2842, %v2847 : tensor<64x256x14x14xf32>
    %v2855 = stablehlo.multiply %v2854, %v2853 : tensor<64x256x14x14xf32>
    %v2856 = stablehlo.broadcast_in_dim %s3b1g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2857 = stablehlo.reshape %v2819 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2858 = stablehlo.multiply %v2856, %v2857 : tensor<64x256x14x14xf32>
    %v2859 = stablehlo.subtract %v2858, %v2849 : tensor<64x256x14x14xf32>
    %v2860 = stablehlo.multiply %v2855, %v2850 : tensor<64x256x14x14xf32>
    %v2861 = stablehlo.subtract %v2859, %v2860 : tensor<64x256x14x14xf32>
    %v2862 = stablehlo.multiply %v2853, %v2861 : tensor<64x256x14x14xf32>
    %v2863 = stablehlo.reshape %v2862 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2864 = stablehlo.reshape %v2863 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2865 = stablehlo.reverse %s3b1W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2866 = stablehlo.transpose %v2865, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2867 = stablehlo.convert %v2864 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2868 = stablehlo.convert %v2866 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2869 = stablehlo.convolution(%v2867, %v2868)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2870 = stablehlo.convert %v2869 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2871 = stablehlo.reshape %v2870 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2872 = stablehlo.reshape %v2871 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2873 = stablehlo.reshape %v1009 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2874 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2875 = stablehlo.compare GT, %v2873, %v2874 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2876 = stablehlo.select %v2875, %v2872, %v2874 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2877 = stablehlo.reshape %v2876 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2878 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2879 = stablehlo.slice %v994 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2880 = stablehlo.slice %v994 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2881 = stablehlo.broadcast_in_dim %v2879, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2882 = stablehlo.broadcast_in_dim %v2880, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2883 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2884 = stablehlo.add %v2882, %v2883 : tensor<64x256x14x14xf32>
    %v2885 = stablehlo.rsqrt %v2884 : tensor<64x256x14x14xf32>
    %v2886 = stablehlo.subtract %v2878, %v2881 : tensor<64x256x14x14xf32>
    %v2887 = stablehlo.multiply %v2886, %v2885 : tensor<64x256x14x14xf32>
    %v2888 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2889 = stablehlo.reshape %v2877 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2890 = stablehlo.multiply %v2888, %v2889 : tensor<64x256x14x14xf32>
    %v2891 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2892 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v2893 = stablehlo.reduce(%v2890 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2894 = stablehlo.divide %v2893, %v2892 : tensor<256xf32>
    %v2895 = stablehlo.multiply %v2887, %v2890 : tensor<64x256x14x14xf32>
    %v2896 = stablehlo.reduce(%v2895 init: %v2891) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2897 = stablehlo.divide %v2896, %v2892 : tensor<256xf32>
    %v2898 = stablehlo.concatenate %v2894, %v2897, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v2899 = stablehlo.concatenate %v994, %v2898, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b1g1dst = "stablehlo.all_reduce"(%v2899) ({
    ^bb0(%aras3b1g1dst: tensor<f32>, %arbs3b1g1dst: tensor<f32>):
      %aradds3b1g1dst = stablehlo.add %aras3b1g1dst, %arbs3b1g1dst : tensor<f32>
      stablehlo.return %aradds3b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b1g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b1g1dst = stablehlo.divide %arsums3b1g1dst, %arns3b1g1dst : tensor<1024xf32>
    %v2900 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2901 = stablehlo.slice %armeans3b1g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2902 = stablehlo.slice %armeans3b1g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2903 = stablehlo.slice %armeans3b1g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2904 = stablehlo.slice %armeans3b1g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v2905 = stablehlo.broadcast_in_dim %v2901, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2906 = stablehlo.broadcast_in_dim %v2902, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2907 = stablehlo.broadcast_in_dim %v2903, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2908 = stablehlo.broadcast_in_dim %v2904, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2909 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2910 = stablehlo.add %v2906, %v2909 : tensor<64x256x14x14xf32>
    %v2911 = stablehlo.rsqrt %v2910 : tensor<64x256x14x14xf32>
    %v2912 = stablehlo.subtract %v2900, %v2905 : tensor<64x256x14x14xf32>
    %v2913 = stablehlo.multiply %v2912, %v2911 : tensor<64x256x14x14xf32>
    %v2914 = stablehlo.broadcast_in_dim %s3b1g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2915 = stablehlo.reshape %v2877 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2916 = stablehlo.multiply %v2914, %v2915 : tensor<64x256x14x14xf32>
    %v2917 = stablehlo.subtract %v2916, %v2907 : tensor<64x256x14x14xf32>
    %v2918 = stablehlo.multiply %v2913, %v2908 : tensor<64x256x14x14xf32>
    %v2919 = stablehlo.subtract %v2917, %v2918 : tensor<64x256x14x14xf32>
    %v2920 = stablehlo.multiply %v2911, %v2919 : tensor<64x256x14x14xf32>
    %v2921 = stablehlo.reshape %v2920 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2922 = stablehlo.reshape %v2921 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2923 = stablehlo.reverse %s3b1W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v2924 = stablehlo.transpose %v2923, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2925 = stablehlo.convert %v2922 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v2926 = stablehlo.convert %v2924 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v2927 = stablehlo.convolution(%v2925, %v2926)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v2928 = stablehlo.convert %v2927 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v2929 = stablehlo.reshape %v2928 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2930 = stablehlo.reshape %v2929 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2931 = stablehlo.reshape %v2819 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2932 = stablehlo.add %v2930, %v2931 : tensor<64x256x14x14xf32>
    %v2933 = stablehlo.reshape %v2932 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2934 = stablehlo.reshape %v967 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2935 = stablehlo.reshape %v2921 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2936 = stablehlo.transpose %v2934, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2937 = stablehlo.transpose %v2935, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2938 = stablehlo.convert %v2936 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2939 = stablehlo.convert %v2937 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2940 = stablehlo.convolution(%v2938, %v2939)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2941 = stablehlo.convert %v2940 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2942 = stablehlo.transpose %v2941, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2943 = stablehlo.reshape %v975 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2944 = stablehlo.slice %v994 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2945 = stablehlo.slice %v994 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2946 = stablehlo.broadcast_in_dim %v2944, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2947 = stablehlo.broadcast_in_dim %v2945, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2948 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2949 = stablehlo.add %v2947, %v2948 : tensor<64x256x14x14xf32>
    %v2950 = stablehlo.rsqrt %v2949 : tensor<64x256x14x14xf32>
    %v2951 = stablehlo.subtract %v2943, %v2946 : tensor<64x256x14x14xf32>
    %v2952 = stablehlo.multiply %v2951, %v2950 : tensor<64x256x14x14xf32>
    %v2953 = stablehlo.reshape %v2877 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2954 = stablehlo.multiply %v2953, %v2952 : tensor<64x256x14x14xf32>
    %v2955 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2956 = stablehlo.reduce(%v2954 init: %v2955) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2957 = stablehlo.reshape %v2877 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2958 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2959 = stablehlo.reduce(%v2957 init: %v2958) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2960 = stablehlo.reshape %v1011 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2961 = stablehlo.reshape %v2863 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2962 = stablehlo.transpose %v2960, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2963 = stablehlo.transpose %v2961, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v2964 = stablehlo.convert %v2962 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2965 = stablehlo.convert %v2963 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v2966 = stablehlo.convolution(%v2964, %v2965)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v2967 = stablehlo.convert %v2966 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v2968 = stablehlo.transpose %v2967, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v2969 = stablehlo.reshape %v1019 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2970 = stablehlo.slice %v1038 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2971 = stablehlo.slice %v1038 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2972 = stablehlo.broadcast_in_dim %v2970, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2973 = stablehlo.broadcast_in_dim %v2971, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2974 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2975 = stablehlo.add %v2973, %v2974 : tensor<64x256x14x14xf32>
    %v2976 = stablehlo.rsqrt %v2975 : tensor<64x256x14x14xf32>
    %v2977 = stablehlo.subtract %v2969, %v2972 : tensor<64x256x14x14xf32>
    %v2978 = stablehlo.multiply %v2977, %v2976 : tensor<64x256x14x14xf32>
    %v2979 = stablehlo.reshape %v2819 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2980 = stablehlo.multiply %v2979, %v2978 : tensor<64x256x14x14xf32>
    %v2981 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2982 = stablehlo.reduce(%v2980 init: %v2981) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2983 = stablehlo.reshape %v2819 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2984 = stablehlo.constant dense<0.0> : tensor<f32>
    %v2985 = stablehlo.reduce(%v2983 init: %v2984) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v2986 = stablehlo.reshape %v2933 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2987 = stablehlo.reshape %v963 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2988 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v2989 = stablehlo.compare GT, %v2987, %v2988 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v2990 = stablehlo.select %v2989, %v2986, %v2988 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v2991 = stablehlo.reshape %v2990 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v2992 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v2993 = stablehlo.slice %v944 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v2994 = stablehlo.slice %v944 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v2995 = stablehlo.broadcast_in_dim %v2993, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2996 = stablehlo.broadcast_in_dim %v2994, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v2997 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v2998 = stablehlo.add %v2996, %v2997 : tensor<64x256x14x14xf32>
    %v2999 = stablehlo.rsqrt %v2998 : tensor<64x256x14x14xf32>
    %v3000 = stablehlo.subtract %v2992, %v2995 : tensor<64x256x14x14xf32>
    %v3001 = stablehlo.multiply %v3000, %v2999 : tensor<64x256x14x14xf32>
    %v3002 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3003 = stablehlo.reshape %v2991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3004 = stablehlo.multiply %v3002, %v3003 : tensor<64x256x14x14xf32>
    %v3005 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3006 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3007 = stablehlo.reduce(%v3004 init: %v3005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3008 = stablehlo.divide %v3007, %v3006 : tensor<256xf32>
    %v3009 = stablehlo.multiply %v3001, %v3004 : tensor<64x256x14x14xf32>
    %v3010 = stablehlo.reduce(%v3009 init: %v3005) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3011 = stablehlo.divide %v3010, %v3006 : tensor<256xf32>
    %v3012 = stablehlo.concatenate %v3008, %v3011, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3013 = stablehlo.concatenate %v944, %v3012, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g2dst = "stablehlo.all_reduce"(%v3013) ({
    ^bb0(%aras3b0g2dst: tensor<f32>, %arbs3b0g2dst: tensor<f32>):
      %aradds3b0g2dst = stablehlo.add %aras3b0g2dst, %arbs3b0g2dst : tensor<f32>
      stablehlo.return %aradds3b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g2dst = stablehlo.divide %arsums3b0g2dst, %arns3b0g2dst : tensor<1024xf32>
    %v3014 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3015 = stablehlo.slice %armeans3b0g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3016 = stablehlo.slice %armeans3b0g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3017 = stablehlo.slice %armeans3b0g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3018 = stablehlo.slice %armeans3b0g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3019 = stablehlo.broadcast_in_dim %v3015, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3020 = stablehlo.broadcast_in_dim %v3016, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3021 = stablehlo.broadcast_in_dim %v3017, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3022 = stablehlo.broadcast_in_dim %v3018, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3023 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3024 = stablehlo.add %v3020, %v3023 : tensor<64x256x14x14xf32>
    %v3025 = stablehlo.rsqrt %v3024 : tensor<64x256x14x14xf32>
    %v3026 = stablehlo.subtract %v3014, %v3019 : tensor<64x256x14x14xf32>
    %v3027 = stablehlo.multiply %v3026, %v3025 : tensor<64x256x14x14xf32>
    %v3028 = stablehlo.broadcast_in_dim %s3b0g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3029 = stablehlo.reshape %v2991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3030 = stablehlo.multiply %v3028, %v3029 : tensor<64x256x14x14xf32>
    %v3031 = stablehlo.subtract %v3030, %v3021 : tensor<64x256x14x14xf32>
    %v3032 = stablehlo.multiply %v3027, %v3022 : tensor<64x256x14x14xf32>
    %v3033 = stablehlo.subtract %v3031, %v3032 : tensor<64x256x14x14xf32>
    %v3034 = stablehlo.multiply %v3025, %v3033 : tensor<64x256x14x14xf32>
    %v3035 = stablehlo.reshape %v3034 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3036 = stablehlo.reshape %v3035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3037 = stablehlo.reverse %s3b0W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3038 = stablehlo.transpose %v3037, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3039 = stablehlo.convert %v3036 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3040 = stablehlo.convert %v3038 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3041 = stablehlo.convolution(%v3039, %v3040)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3042 = stablehlo.convert %v3041 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3043 = stablehlo.reshape %v3042 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3044 = stablehlo.reshape %v3043 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3045 = stablehlo.reshape %v915 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3046 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3047 = stablehlo.compare GT, %v3045, %v3046 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3048 = stablehlo.select %v3047, %v3044, %v3046 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3049 = stablehlo.reshape %v3048 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3050 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3051 = stablehlo.slice %v900 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3052 = stablehlo.slice %v900 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3053 = stablehlo.broadcast_in_dim %v3051, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3054 = stablehlo.broadcast_in_dim %v3052, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3055 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3056 = stablehlo.add %v3054, %v3055 : tensor<64x256x14x14xf32>
    %v3057 = stablehlo.rsqrt %v3056 : tensor<64x256x14x14xf32>
    %v3058 = stablehlo.subtract %v3050, %v3053 : tensor<64x256x14x14xf32>
    %v3059 = stablehlo.multiply %v3058, %v3057 : tensor<64x256x14x14xf32>
    %v3060 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3061 = stablehlo.reshape %v3049 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3062 = stablehlo.multiply %v3060, %v3061 : tensor<64x256x14x14xf32>
    %v3063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3064 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3065 = stablehlo.reduce(%v3062 init: %v3063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3066 = stablehlo.divide %v3065, %v3064 : tensor<256xf32>
    %v3067 = stablehlo.multiply %v3059, %v3062 : tensor<64x256x14x14xf32>
    %v3068 = stablehlo.reduce(%v3067 init: %v3063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3069 = stablehlo.divide %v3068, %v3064 : tensor<256xf32>
    %v3070 = stablehlo.concatenate %v3066, %v3069, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3071 = stablehlo.concatenate %v900, %v3070, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsums3b0g1dst = "stablehlo.all_reduce"(%v3071) ({
    ^bb0(%aras3b0g1dst: tensor<f32>, %arbs3b0g1dst: tensor<f32>):
      %aradds3b0g1dst = stablehlo.add %aras3b0g1dst, %arbs3b0g1dst : tensor<f32>
      stablehlo.return %aradds3b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arns3b0g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeans3b0g1dst = stablehlo.divide %arsums3b0g1dst, %arns3b0g1dst : tensor<1024xf32>
    %v3072 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3073 = stablehlo.slice %armeans3b0g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3074 = stablehlo.slice %armeans3b0g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3075 = stablehlo.slice %armeans3b0g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3076 = stablehlo.slice %armeans3b0g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3077 = stablehlo.broadcast_in_dim %v3073, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3078 = stablehlo.broadcast_in_dim %v3074, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3079 = stablehlo.broadcast_in_dim %v3075, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3080 = stablehlo.broadcast_in_dim %v3076, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3081 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3082 = stablehlo.add %v3078, %v3081 : tensor<64x256x14x14xf32>
    %v3083 = stablehlo.rsqrt %v3082 : tensor<64x256x14x14xf32>
    %v3084 = stablehlo.subtract %v3072, %v3077 : tensor<64x256x14x14xf32>
    %v3085 = stablehlo.multiply %v3084, %v3083 : tensor<64x256x14x14xf32>
    %v3086 = stablehlo.broadcast_in_dim %s3b0g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3087 = stablehlo.reshape %v3049 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3088 = stablehlo.multiply %v3086, %v3087 : tensor<64x256x14x14xf32>
    %v3089 = stablehlo.subtract %v3088, %v3079 : tensor<64x256x14x14xf32>
    %v3090 = stablehlo.multiply %v3085, %v3080 : tensor<64x256x14x14xf32>
    %v3091 = stablehlo.subtract %v3089, %v3090 : tensor<64x256x14x14xf32>
    %v3092 = stablehlo.multiply %v3083, %v3091 : tensor<64x256x14x14xf32>
    %v3093 = stablehlo.reshape %v3092 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3094 = stablehlo.reshape %v3093 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3095 = stablehlo.reverse %s3b0W1, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3096 = stablehlo.transpose %v3095, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3097 = stablehlo.convert %v3094 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3098 = stablehlo.convert %v3096 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3099 = stablehlo.convolution(%v3097, %v3098)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3100 = stablehlo.convert %v3099 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3101 = stablehlo.reshape %v3100 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3102 = stablehlo.reshape %v3101 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3103 = stablehlo.reshape %v2991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3104 = stablehlo.add %v3102, %v3103 : tensor<64x256x14x14xf32>
    %v3105 = stablehlo.reshape %v3104 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3106 = stablehlo.reshape %v873 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3107 = stablehlo.reshape %v3093 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3108 = stablehlo.transpose %v3106, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3109 = stablehlo.transpose %v3107, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3110 = stablehlo.convert %v3108 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3111 = stablehlo.convert %v3109 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3112 = stablehlo.convolution(%v3110, %v3111)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3113 = stablehlo.convert %v3112 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3114 = stablehlo.transpose %v3113, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3115 = stablehlo.reshape %v881 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3116 = stablehlo.slice %v900 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3117 = stablehlo.slice %v900 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3118 = stablehlo.broadcast_in_dim %v3116, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3119 = stablehlo.broadcast_in_dim %v3117, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3120 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3121 = stablehlo.add %v3119, %v3120 : tensor<64x256x14x14xf32>
    %v3122 = stablehlo.rsqrt %v3121 : tensor<64x256x14x14xf32>
    %v3123 = stablehlo.subtract %v3115, %v3118 : tensor<64x256x14x14xf32>
    %v3124 = stablehlo.multiply %v3123, %v3122 : tensor<64x256x14x14xf32>
    %v3125 = stablehlo.reshape %v3049 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3126 = stablehlo.multiply %v3125, %v3124 : tensor<64x256x14x14xf32>
    %v3127 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3128 = stablehlo.reduce(%v3126 init: %v3127) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3129 = stablehlo.reshape %v3049 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3130 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3131 = stablehlo.reduce(%v3129 init: %v3130) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3132 = stablehlo.reshape %v917 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3133 = stablehlo.reshape %v3035 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3134 = stablehlo.transpose %v3132, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3135 = stablehlo.transpose %v3133, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3136 = stablehlo.convert %v3134 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3137 = stablehlo.convert %v3135 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3138 = stablehlo.convolution(%v3136, %v3137)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3139 = stablehlo.convert %v3138 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3140 = stablehlo.transpose %v3139, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3141 = stablehlo.reshape %v925 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3142 = stablehlo.slice %v944 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3143 = stablehlo.slice %v944 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3144 = stablehlo.broadcast_in_dim %v3142, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3145 = stablehlo.broadcast_in_dim %v3143, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3146 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3147 = stablehlo.add %v3145, %v3146 : tensor<64x256x14x14xf32>
    %v3148 = stablehlo.rsqrt %v3147 : tensor<64x256x14x14xf32>
    %v3149 = stablehlo.subtract %v3141, %v3144 : tensor<64x256x14x14xf32>
    %v3150 = stablehlo.multiply %v3149, %v3148 : tensor<64x256x14x14xf32>
    %v3151 = stablehlo.reshape %v2991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3152 = stablehlo.multiply %v3151, %v3150 : tensor<64x256x14x14xf32>
    %v3153 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3154 = stablehlo.reduce(%v3152 init: %v3153) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3155 = stablehlo.reshape %v2991 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3156 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3157 = stablehlo.reduce(%v3155 init: %v3156) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3158 = stablehlo.reshape %v3105 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3159 = stablehlo.reshape %v871 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3160 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3161 = stablehlo.compare GT, %v3159, %v3160 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3162 = stablehlo.select %v3161, %v3158, %v3160 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3163 = stablehlo.reshape %v3162 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3164 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3165 = stablehlo.slice %v813 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3166 = stablehlo.slice %v813 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3167 = stablehlo.broadcast_in_dim %v3165, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3168 = stablehlo.broadcast_in_dim %v3166, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3169 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3170 = stablehlo.add %v3168, %v3169 : tensor<64x256x14x14xf32>
    %v3171 = stablehlo.rsqrt %v3170 : tensor<64x256x14x14xf32>
    %v3172 = stablehlo.subtract %v3164, %v3167 : tensor<64x256x14x14xf32>
    %v3173 = stablehlo.multiply %v3172, %v3171 : tensor<64x256x14x14xf32>
    %v3174 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3175 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3176 = stablehlo.multiply %v3174, %v3175 : tensor<64x256x14x14xf32>
    %v3177 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3178 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3179 = stablehlo.reduce(%v3176 init: %v3177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3180 = stablehlo.divide %v3179, %v3178 : tensor<256xf32>
    %v3181 = stablehlo.multiply %v3173, %v3176 : tensor<64x256x14x14xf32>
    %v3182 = stablehlo.reduce(%v3181 init: %v3177) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3183 = stablehlo.divide %v3182, %v3178 : tensor<256xf32>
    %v3184 = stablehlo.concatenate %v3180, %v3183, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3185 = stablehlo.concatenate %v813, %v3184, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g2dst = "stablehlo.all_reduce"(%v3185) ({
    ^bb0(%arad3g2dst: tensor<f32>, %arbd3g2dst: tensor<f32>):
      %araddd3g2dst = stablehlo.add %arad3g2dst, %arbd3g2dst : tensor<f32>
      stablehlo.return %araddd3g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g2dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeand3g2dst = stablehlo.divide %arsumd3g2dst, %arnd3g2dst : tensor<1024xf32>
    %v3186 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3187 = stablehlo.slice %armeand3g2dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3188 = stablehlo.slice %armeand3g2dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3189 = stablehlo.slice %armeand3g2dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3190 = stablehlo.slice %armeand3g2dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3191 = stablehlo.broadcast_in_dim %v3187, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3192 = stablehlo.broadcast_in_dim %v3188, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3193 = stablehlo.broadcast_in_dim %v3189, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3194 = stablehlo.broadcast_in_dim %v3190, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3195 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3196 = stablehlo.add %v3192, %v3195 : tensor<64x256x14x14xf32>
    %v3197 = stablehlo.rsqrt %v3196 : tensor<64x256x14x14xf32>
    %v3198 = stablehlo.subtract %v3186, %v3191 : tensor<64x256x14x14xf32>
    %v3199 = stablehlo.multiply %v3198, %v3197 : tensor<64x256x14x14xf32>
    %v3200 = stablehlo.broadcast_in_dim %d3g2, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3201 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3202 = stablehlo.multiply %v3200, %v3201 : tensor<64x256x14x14xf32>
    %v3203 = stablehlo.subtract %v3202, %v3193 : tensor<64x256x14x14xf32>
    %v3204 = stablehlo.multiply %v3199, %v3194 : tensor<64x256x14x14xf32>
    %v3205 = stablehlo.subtract %v3203, %v3204 : tensor<64x256x14x14xf32>
    %v3206 = stablehlo.multiply %v3197, %v3205 : tensor<64x256x14x14xf32>
    %v3207 = stablehlo.reshape %v3206 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3208 = stablehlo.reshape %v3207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3209 = stablehlo.reverse %d3W2, dims = [2, 3] : tensor<256x256x3x3xf32>
    %v3210 = stablehlo.transpose %v3209, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3211 = stablehlo.convert %v3208 : (tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xbf16>
    %v3212 = stablehlo.convert %v3210 : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xbf16>
    %v3213 = stablehlo.convolution(%v3211, %v3212)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x14x14xbf16>, tensor<256x256x3x3xbf16>) -> tensor<64x256x14x14xbf16>
    %v3214 = stablehlo.convert %v3213 : (tensor<64x256x14x14xbf16>) -> tensor<64x256x14x14xf32>
    %v3215 = stablehlo.reshape %v3214 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3216 = stablehlo.reshape %v3215 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3217 = stablehlo.reshape %v784 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3218 = stablehlo.constant dense<0.0> : tensor<64x256x14x14xf32>
    %v3219 = stablehlo.compare GT, %v3217, %v3218 : (tensor<64x256x14x14xf32>, tensor<64x256x14x14xf32>) -> tensor<64x256x14x14xi1>
    %v3220 = stablehlo.select %v3219, %v3216, %v3218 : tensor<64x256x14x14xi1>, tensor<64x256x14x14xf32>
    %v3221 = stablehlo.reshape %v3220 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3222 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3223 = stablehlo.slice %v769 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3224 = stablehlo.slice %v769 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3225 = stablehlo.broadcast_in_dim %v3223, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3226 = stablehlo.broadcast_in_dim %v3224, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3227 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3228 = stablehlo.add %v3226, %v3227 : tensor<64x256x14x14xf32>
    %v3229 = stablehlo.rsqrt %v3228 : tensor<64x256x14x14xf32>
    %v3230 = stablehlo.subtract %v3222, %v3225 : tensor<64x256x14x14xf32>
    %v3231 = stablehlo.multiply %v3230, %v3229 : tensor<64x256x14x14xf32>
    %v3232 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3233 = stablehlo.reshape %v3221 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3234 = stablehlo.multiply %v3232, %v3233 : tensor<64x256x14x14xf32>
    %v3235 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3236 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3237 = stablehlo.reduce(%v3234 init: %v3235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3238 = stablehlo.divide %v3237, %v3236 : tensor<256xf32>
    %v3239 = stablehlo.multiply %v3231, %v3234 : tensor<64x256x14x14xf32>
    %v3240 = stablehlo.reduce(%v3239 init: %v3235) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3241 = stablehlo.divide %v3240, %v3236 : tensor<256xf32>
    %v3242 = stablehlo.concatenate %v3238, %v3241, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3243 = stablehlo.concatenate %v769, %v3242, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3g1dst = "stablehlo.all_reduce"(%v3243) ({
    ^bb0(%arad3g1dst: tensor<f32>, %arbd3g1dst: tensor<f32>):
      %araddd3g1dst = stablehlo.add %arad3g1dst, %arbd3g1dst : tensor<f32>
      stablehlo.return %araddd3g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3g1dst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeand3g1dst = stablehlo.divide %arsumd3g1dst, %arnd3g1dst : tensor<1024xf32>
    %v3244 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3245 = stablehlo.slice %armeand3g1dst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3246 = stablehlo.slice %armeand3g1dst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3247 = stablehlo.slice %armeand3g1dst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3248 = stablehlo.slice %armeand3g1dst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3249 = stablehlo.broadcast_in_dim %v3245, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3250 = stablehlo.broadcast_in_dim %v3246, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3251 = stablehlo.broadcast_in_dim %v3247, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3252 = stablehlo.broadcast_in_dim %v3248, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3253 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3254 = stablehlo.add %v3250, %v3253 : tensor<64x256x14x14xf32>
    %v3255 = stablehlo.rsqrt %v3254 : tensor<64x256x14x14xf32>
    %v3256 = stablehlo.subtract %v3244, %v3249 : tensor<64x256x14x14xf32>
    %v3257 = stablehlo.multiply %v3256, %v3255 : tensor<64x256x14x14xf32>
    %v3258 = stablehlo.broadcast_in_dim %d3g1, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3259 = stablehlo.reshape %v3221 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3260 = stablehlo.multiply %v3258, %v3259 : tensor<64x256x14x14xf32>
    %v3261 = stablehlo.subtract %v3260, %v3251 : tensor<64x256x14x14xf32>
    %v3262 = stablehlo.multiply %v3257, %v3252 : tensor<64x256x14x14xf32>
    %v3263 = stablehlo.subtract %v3261, %v3262 : tensor<64x256x14x14xf32>
    %v3264 = stablehlo.multiply %v3255, %v3263 : tensor<64x256x14x14xf32>
    %v3265 = stablehlo.reshape %v3264 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3266 = stablehlo.reshape %v3265 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3268 = stablehlo.pad %v3266, %v3267, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3269 = stablehlo.reverse %d3W1, dims = [2, 3] : tensor<256x128x3x3xf32>
    %v3270 = stablehlo.transpose %v3269, dims = [1, 0, 2, 3] : (tensor<256x128x3x3xf32>) -> tensor<128x256x3x3xf32>
    %v3271 = stablehlo.convert %v3268 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3272 = stablehlo.convert %v3270 : (tensor<128x256x3x3xf32>) -> tensor<128x256x3x3xbf16>
    %v3273 = stablehlo.convolution(%v3271, %v3272)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<128x256x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3274 = stablehlo.convert %v3273 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3275 = stablehlo.reshape %v3274 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3276 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3277 = stablehlo.slice %v855 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3278 = stablehlo.slice %v855 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3279 = stablehlo.broadcast_in_dim %v3277, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3280 = stablehlo.broadcast_in_dim %v3278, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3281 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3282 = stablehlo.add %v3280, %v3281 : tensor<64x256x14x14xf32>
    %v3283 = stablehlo.rsqrt %v3282 : tensor<64x256x14x14xf32>
    %v3284 = stablehlo.subtract %v3276, %v3279 : tensor<64x256x14x14xf32>
    %v3285 = stablehlo.multiply %v3284, %v3283 : tensor<64x256x14x14xf32>
    %v3286 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3287 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3288 = stablehlo.multiply %v3286, %v3287 : tensor<64x256x14x14xf32>
    %v3289 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3290 = stablehlo.constant dense<12544.0> : tensor<256xf32>
    %v3291 = stablehlo.reduce(%v3288 init: %v3289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3292 = stablehlo.divide %v3291, %v3290 : tensor<256xf32>
    %v3293 = stablehlo.multiply %v3285, %v3288 : tensor<64x256x14x14xf32>
    %v3294 = stablehlo.reduce(%v3293 init: %v3289) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3295 = stablehlo.divide %v3294, %v3290 : tensor<256xf32>
    %v3296 = stablehlo.concatenate %v3292, %v3295, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %v3297 = stablehlo.concatenate %v855, %v3296, dim = 0 : (tensor<512xf32>, tensor<512xf32>) -> tensor<1024xf32>
    %arsumd3gpdst = "stablehlo.all_reduce"(%v3297) ({
    ^bb0(%arad3gpdst: tensor<f32>, %arbd3gpdst: tensor<f32>):
      %araddd3gpdst = stablehlo.add %arad3gpdst, %arbd3gpdst : tensor<f32>
      stablehlo.return %araddd3gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1024xf32>) -> tensor<1024xf32>
    %arnd3gpdst = stablehlo.constant dense<4.0> : tensor<1024xf32>
    %armeand3gpdst = stablehlo.divide %arsumd3gpdst, %arnd3gpdst : tensor<1024xf32>
    %v3298 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3299 = stablehlo.slice %armeand3gpdst [0:256] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3300 = stablehlo.slice %armeand3gpdst [256:512] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3301 = stablehlo.slice %armeand3gpdst [512:768] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3302 = stablehlo.slice %armeand3gpdst [768:1024] : (tensor<1024xf32>) -> tensor<256xf32>
    %v3303 = stablehlo.broadcast_in_dim %v3299, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3304 = stablehlo.broadcast_in_dim %v3300, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3305 = stablehlo.broadcast_in_dim %v3301, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3306 = stablehlo.broadcast_in_dim %v3302, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3307 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3308 = stablehlo.add %v3304, %v3307 : tensor<64x256x14x14xf32>
    %v3309 = stablehlo.rsqrt %v3308 : tensor<64x256x14x14xf32>
    %v3310 = stablehlo.subtract %v3298, %v3303 : tensor<64x256x14x14xf32>
    %v3311 = stablehlo.multiply %v3310, %v3309 : tensor<64x256x14x14xf32>
    %v3312 = stablehlo.broadcast_in_dim %d3gp, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3313 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3314 = stablehlo.multiply %v3312, %v3313 : tensor<64x256x14x14xf32>
    %v3315 = stablehlo.subtract %v3314, %v3305 : tensor<64x256x14x14xf32>
    %v3316 = stablehlo.multiply %v3311, %v3306 : tensor<64x256x14x14xf32>
    %v3317 = stablehlo.subtract %v3315, %v3316 : tensor<64x256x14x14xf32>
    %v3318 = stablehlo.multiply %v3309, %v3317 : tensor<64x256x14x14xf32>
    %v3319 = stablehlo.reshape %v3318 : (tensor<64x256x14x14xf32>) -> tensor<64x50176xf32>
    %v3320 = stablehlo.reshape %v3319 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3321 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3322 = stablehlo.pad %v3320, %v3321, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3323 = stablehlo.reverse %d3Wp, dims = [2, 3] : tensor<256x128x1x1xf32>
    %v3324 = stablehlo.transpose %v3323, dims = [1, 0, 2, 3] : (tensor<256x128x1x1xf32>) -> tensor<128x256x1x1xf32>
    %v3325 = stablehlo.convert %v3322 : (tensor<64x256x28x28xf32>) -> tensor<64x256x28x28xbf16>
    %v3326 = stablehlo.convert %v3324 : (tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xbf16>
    %v3327 = stablehlo.convolution(%v3325, %v3326)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x256x28x28xbf16>, tensor<128x256x1x1xbf16>) -> tensor<64x128x28x28xbf16>
    %v3328 = stablehlo.convert %v3327 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3329 = stablehlo.reshape %v3328 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3330 = stablehlo.reshape %v3275 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3331 = stablehlo.reshape %v3329 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3332 = stablehlo.add %v3330, %v3331 : tensor<64x128x28x28xf32>
    %v3333 = stablehlo.reshape %v3332 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3334 = stablehlo.reshape %v742 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3335 = stablehlo.reshape %v3265 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3336 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3337 = stablehlo.pad %v3335, %v3336, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3338 = stablehlo.transpose %v3334, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3339 = stablehlo.transpose %v3337, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3340 = stablehlo.convert %v3338 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3341 = stablehlo.convert %v3339 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3342 = stablehlo.convolution(%v3340, %v3341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<128x256x3x3xbf16>
    %v3343 = stablehlo.convert %v3342 : (tensor<128x256x3x3xbf16>) -> tensor<128x256x3x3xf32>
    %v3344 = stablehlo.transpose %v3343, dims = [1, 0, 2, 3] : (tensor<128x256x3x3xf32>) -> tensor<256x128x3x3xf32>
    %v3345 = stablehlo.reshape %v750 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3346 = stablehlo.slice %v769 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3347 = stablehlo.slice %v769 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3348 = stablehlo.broadcast_in_dim %v3346, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3349 = stablehlo.broadcast_in_dim %v3347, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3350 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3351 = stablehlo.add %v3349, %v3350 : tensor<64x256x14x14xf32>
    %v3352 = stablehlo.rsqrt %v3351 : tensor<64x256x14x14xf32>
    %v3353 = stablehlo.subtract %v3345, %v3348 : tensor<64x256x14x14xf32>
    %v3354 = stablehlo.multiply %v3353, %v3352 : tensor<64x256x14x14xf32>
    %v3355 = stablehlo.reshape %v3221 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3356 = stablehlo.multiply %v3355, %v3354 : tensor<64x256x14x14xf32>
    %v3357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3358 = stablehlo.reduce(%v3356 init: %v3357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3359 = stablehlo.reshape %v3221 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3361 = stablehlo.reduce(%v3359 init: %v3360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3362 = stablehlo.reshape %v786 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3363 = stablehlo.reshape %v3207 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3364 = stablehlo.transpose %v3362, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3365 = stablehlo.transpose %v3363, dims = [1, 0, 2, 3] : (tensor<64x256x14x14xf32>) -> tensor<256x64x14x14xf32>
    %v3366 = stablehlo.convert %v3364 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3367 = stablehlo.convert %v3365 : (tensor<256x64x14x14xf32>) -> tensor<256x64x14x14xbf16>
    %v3368 = stablehlo.convolution(%v3366, %v3367)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<256x64x14x14xbf16>, tensor<256x64x14x14xbf16>) -> tensor<256x256x3x3xbf16>
    %v3369 = stablehlo.convert %v3368 : (tensor<256x256x3x3xbf16>) -> tensor<256x256x3x3xf32>
    %v3370 = stablehlo.transpose %v3369, dims = [1, 0, 2, 3] : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %v3371 = stablehlo.reshape %v794 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3372 = stablehlo.slice %v813 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3373 = stablehlo.slice %v813 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3374 = stablehlo.broadcast_in_dim %v3372, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3375 = stablehlo.broadcast_in_dim %v3373, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3376 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3377 = stablehlo.add %v3375, %v3376 : tensor<64x256x14x14xf32>
    %v3378 = stablehlo.rsqrt %v3377 : tensor<64x256x14x14xf32>
    %v3379 = stablehlo.subtract %v3371, %v3374 : tensor<64x256x14x14xf32>
    %v3380 = stablehlo.multiply %v3379, %v3378 : tensor<64x256x14x14xf32>
    %v3381 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3382 = stablehlo.multiply %v3381, %v3380 : tensor<64x256x14x14xf32>
    %v3383 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3384 = stablehlo.reduce(%v3382 init: %v3383) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3385 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3386 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3387 = stablehlo.reduce(%v3385 init: %v3386) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3388 = stablehlo.reshape %v742 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3389 = stablehlo.reshape %v3319 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3390 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3391 = stablehlo.pad %v3389, %v3390, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<64x256x28x28xf32>
    %v3392 = stablehlo.transpose %v3388, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3393 = stablehlo.transpose %v3391, dims = [1, 0, 2, 3] : (tensor<64x256x28x28xf32>) -> tensor<256x64x28x28xf32>
    %v3394 = stablehlo.convert %v3392 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3395 = stablehlo.convert %v3393 : (tensor<256x64x28x28xf32>) -> tensor<256x64x28x28xbf16>
    %v3396 = stablehlo.convolution(%v3394, %v3395)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<256x64x28x28xbf16>) -> tensor<128x256x1x1xbf16>
    %v3397 = stablehlo.convert %v3396 : (tensor<128x256x1x1xbf16>) -> tensor<128x256x1x1xf32>
    %v3398 = stablehlo.transpose %v3397, dims = [1, 0, 2, 3] : (tensor<128x256x1x1xf32>) -> tensor<256x128x1x1xf32>
    %v3399 = stablehlo.reshape %v836 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3400 = stablehlo.slice %v855 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v3401 = stablehlo.slice %v855 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v3402 = stablehlo.broadcast_in_dim %v3400, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3403 = stablehlo.broadcast_in_dim %v3401, dims = [1] : (tensor<256xf32>) -> tensor<64x256x14x14xf32>
    %v3404 = stablehlo.constant dense<1.0e-05> : tensor<64x256x14x14xf32>
    %v3405 = stablehlo.add %v3403, %v3404 : tensor<64x256x14x14xf32>
    %v3406 = stablehlo.rsqrt %v3405 : tensor<64x256x14x14xf32>
    %v3407 = stablehlo.subtract %v3399, %v3402 : tensor<64x256x14x14xf32>
    %v3408 = stablehlo.multiply %v3407, %v3406 : tensor<64x256x14x14xf32>
    %v3409 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3410 = stablehlo.multiply %v3409, %v3408 : tensor<64x256x14x14xf32>
    %v3411 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3412 = stablehlo.reduce(%v3410 init: %v3411) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3413 = stablehlo.reshape %v3163 : (tensor<64x50176xf32>) -> tensor<64x256x14x14xf32>
    %v3414 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3415 = stablehlo.reduce(%v3413 init: %v3414) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x256x14x14xf32>, tensor<f32>) -> tensor<256xf32>
    %v3416 = stablehlo.reshape %v3333 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3417 = stablehlo.reshape %v738 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3418 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3419 = stablehlo.compare GT, %v3417, %v3418 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3420 = stablehlo.select %v3419, %v3416, %v3418 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3421 = stablehlo.reshape %v3420 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3422 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3423 = stablehlo.slice %v719 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3424 = stablehlo.slice %v719 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3425 = stablehlo.broadcast_in_dim %v3423, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3426 = stablehlo.broadcast_in_dim %v3424, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3427 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3428 = stablehlo.add %v3426, %v3427 : tensor<64x128x28x28xf32>
    %v3429 = stablehlo.rsqrt %v3428 : tensor<64x128x28x28xf32>
    %v3430 = stablehlo.subtract %v3422, %v3425 : tensor<64x128x28x28xf32>
    %v3431 = stablehlo.multiply %v3430, %v3429 : tensor<64x128x28x28xf32>
    %v3432 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3433 = stablehlo.reshape %v3421 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3434 = stablehlo.multiply %v3432, %v3433 : tensor<64x128x28x28xf32>
    %v3435 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3436 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3437 = stablehlo.reduce(%v3434 init: %v3435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3438 = stablehlo.divide %v3437, %v3436 : tensor<128xf32>
    %v3439 = stablehlo.multiply %v3431, %v3434 : tensor<64x128x28x28xf32>
    %v3440 = stablehlo.reduce(%v3439 init: %v3435) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3441 = stablehlo.divide %v3440, %v3436 : tensor<128xf32>
    %v3442 = stablehlo.concatenate %v3438, %v3441, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3443 = stablehlo.concatenate %v719, %v3442, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g2dst = "stablehlo.all_reduce"(%v3443) ({
    ^bb0(%aras2b2g2dst: tensor<f32>, %arbs2b2g2dst: tensor<f32>):
      %aradds2b2g2dst = stablehlo.add %aras2b2g2dst, %arbs2b2g2dst : tensor<f32>
      stablehlo.return %aradds2b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g2dst = stablehlo.divide %arsums2b2g2dst, %arns2b2g2dst : tensor<512xf32>
    %v3444 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3445 = stablehlo.slice %armeans2b2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3446 = stablehlo.slice %armeans2b2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3447 = stablehlo.slice %armeans2b2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3448 = stablehlo.slice %armeans2b2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3449 = stablehlo.broadcast_in_dim %v3445, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3450 = stablehlo.broadcast_in_dim %v3446, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3451 = stablehlo.broadcast_in_dim %v3447, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3452 = stablehlo.broadcast_in_dim %v3448, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3453 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3454 = stablehlo.add %v3450, %v3453 : tensor<64x128x28x28xf32>
    %v3455 = stablehlo.rsqrt %v3454 : tensor<64x128x28x28xf32>
    %v3456 = stablehlo.subtract %v3444, %v3449 : tensor<64x128x28x28xf32>
    %v3457 = stablehlo.multiply %v3456, %v3455 : tensor<64x128x28x28xf32>
    %v3458 = stablehlo.broadcast_in_dim %s2b2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3459 = stablehlo.reshape %v3421 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3460 = stablehlo.multiply %v3458, %v3459 : tensor<64x128x28x28xf32>
    %v3461 = stablehlo.subtract %v3460, %v3451 : tensor<64x128x28x28xf32>
    %v3462 = stablehlo.multiply %v3457, %v3452 : tensor<64x128x28x28xf32>
    %v3463 = stablehlo.subtract %v3461, %v3462 : tensor<64x128x28x28xf32>
    %v3464 = stablehlo.multiply %v3455, %v3463 : tensor<64x128x28x28xf32>
    %v3465 = stablehlo.reshape %v3464 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3466 = stablehlo.reshape %v3465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3467 = stablehlo.reverse %s2b2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3468 = stablehlo.transpose %v3467, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3469 = stablehlo.convert %v3466 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3470 = stablehlo.convert %v3468 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3471 = stablehlo.convolution(%v3469, %v3470)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3472 = stablehlo.convert %v3471 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3473 = stablehlo.reshape %v3472 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3474 = stablehlo.reshape %v3473 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3475 = stablehlo.reshape %v690 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3476 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3477 = stablehlo.compare GT, %v3475, %v3476 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3478 = stablehlo.select %v3477, %v3474, %v3476 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3479 = stablehlo.reshape %v3478 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3480 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3481 = stablehlo.slice %v675 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3482 = stablehlo.slice %v675 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3483 = stablehlo.broadcast_in_dim %v3481, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3484 = stablehlo.broadcast_in_dim %v3482, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3485 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3486 = stablehlo.add %v3484, %v3485 : tensor<64x128x28x28xf32>
    %v3487 = stablehlo.rsqrt %v3486 : tensor<64x128x28x28xf32>
    %v3488 = stablehlo.subtract %v3480, %v3483 : tensor<64x128x28x28xf32>
    %v3489 = stablehlo.multiply %v3488, %v3487 : tensor<64x128x28x28xf32>
    %v3490 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3491 = stablehlo.reshape %v3479 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3492 = stablehlo.multiply %v3490, %v3491 : tensor<64x128x28x28xf32>
    %v3493 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3494 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3495 = stablehlo.reduce(%v3492 init: %v3493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3496 = stablehlo.divide %v3495, %v3494 : tensor<128xf32>
    %v3497 = stablehlo.multiply %v3489, %v3492 : tensor<64x128x28x28xf32>
    %v3498 = stablehlo.reduce(%v3497 init: %v3493) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3499 = stablehlo.divide %v3498, %v3494 : tensor<128xf32>
    %v3500 = stablehlo.concatenate %v3496, %v3499, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3501 = stablehlo.concatenate %v675, %v3500, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b2g1dst = "stablehlo.all_reduce"(%v3501) ({
    ^bb0(%aras2b2g1dst: tensor<f32>, %arbs2b2g1dst: tensor<f32>):
      %aradds2b2g1dst = stablehlo.add %aras2b2g1dst, %arbs2b2g1dst : tensor<f32>
      stablehlo.return %aradds2b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b2g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b2g1dst = stablehlo.divide %arsums2b2g1dst, %arns2b2g1dst : tensor<512xf32>
    %v3502 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3503 = stablehlo.slice %armeans2b2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3504 = stablehlo.slice %armeans2b2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3505 = stablehlo.slice %armeans2b2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3506 = stablehlo.slice %armeans2b2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3507 = stablehlo.broadcast_in_dim %v3503, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3508 = stablehlo.broadcast_in_dim %v3504, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3509 = stablehlo.broadcast_in_dim %v3505, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3510 = stablehlo.broadcast_in_dim %v3506, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3511 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3512 = stablehlo.add %v3508, %v3511 : tensor<64x128x28x28xf32>
    %v3513 = stablehlo.rsqrt %v3512 : tensor<64x128x28x28xf32>
    %v3514 = stablehlo.subtract %v3502, %v3507 : tensor<64x128x28x28xf32>
    %v3515 = stablehlo.multiply %v3514, %v3513 : tensor<64x128x28x28xf32>
    %v3516 = stablehlo.broadcast_in_dim %s2b2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3517 = stablehlo.reshape %v3479 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3518 = stablehlo.multiply %v3516, %v3517 : tensor<64x128x28x28xf32>
    %v3519 = stablehlo.subtract %v3518, %v3509 : tensor<64x128x28x28xf32>
    %v3520 = stablehlo.multiply %v3515, %v3510 : tensor<64x128x28x28xf32>
    %v3521 = stablehlo.subtract %v3519, %v3520 : tensor<64x128x28x28xf32>
    %v3522 = stablehlo.multiply %v3513, %v3521 : tensor<64x128x28x28xf32>
    %v3523 = stablehlo.reshape %v3522 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3524 = stablehlo.reshape %v3523 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3525 = stablehlo.reverse %s2b2W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3526 = stablehlo.transpose %v3525, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3527 = stablehlo.convert %v3524 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3528 = stablehlo.convert %v3526 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3529 = stablehlo.convolution(%v3527, %v3528)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3530 = stablehlo.convert %v3529 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3531 = stablehlo.reshape %v3530 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3532 = stablehlo.reshape %v3531 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3533 = stablehlo.reshape %v3421 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3534 = stablehlo.add %v3532, %v3533 : tensor<64x128x28x28xf32>
    %v3535 = stablehlo.reshape %v3534 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3536 = stablehlo.reshape %v648 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3537 = stablehlo.reshape %v3523 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3538 = stablehlo.transpose %v3536, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3539 = stablehlo.transpose %v3537, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3540 = stablehlo.convert %v3538 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3541 = stablehlo.convert %v3539 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3542 = stablehlo.convolution(%v3540, %v3541)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3543 = stablehlo.convert %v3542 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3544 = stablehlo.transpose %v3543, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3545 = stablehlo.reshape %v656 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3546 = stablehlo.slice %v675 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3547 = stablehlo.slice %v675 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3548 = stablehlo.broadcast_in_dim %v3546, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3549 = stablehlo.broadcast_in_dim %v3547, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3550 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3551 = stablehlo.add %v3549, %v3550 : tensor<64x128x28x28xf32>
    %v3552 = stablehlo.rsqrt %v3551 : tensor<64x128x28x28xf32>
    %v3553 = stablehlo.subtract %v3545, %v3548 : tensor<64x128x28x28xf32>
    %v3554 = stablehlo.multiply %v3553, %v3552 : tensor<64x128x28x28xf32>
    %v3555 = stablehlo.reshape %v3479 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3556 = stablehlo.multiply %v3555, %v3554 : tensor<64x128x28x28xf32>
    %v3557 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3558 = stablehlo.reduce(%v3556 init: %v3557) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3559 = stablehlo.reshape %v3479 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3560 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3561 = stablehlo.reduce(%v3559 init: %v3560) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3562 = stablehlo.reshape %v692 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3563 = stablehlo.reshape %v3465 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3564 = stablehlo.transpose %v3562, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3565 = stablehlo.transpose %v3563, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3566 = stablehlo.convert %v3564 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3567 = stablehlo.convert %v3565 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3568 = stablehlo.convolution(%v3566, %v3567)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3569 = stablehlo.convert %v3568 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3570 = stablehlo.transpose %v3569, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3571 = stablehlo.reshape %v700 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3572 = stablehlo.slice %v719 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3573 = stablehlo.slice %v719 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3574 = stablehlo.broadcast_in_dim %v3572, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3575 = stablehlo.broadcast_in_dim %v3573, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3576 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3577 = stablehlo.add %v3575, %v3576 : tensor<64x128x28x28xf32>
    %v3578 = stablehlo.rsqrt %v3577 : tensor<64x128x28x28xf32>
    %v3579 = stablehlo.subtract %v3571, %v3574 : tensor<64x128x28x28xf32>
    %v3580 = stablehlo.multiply %v3579, %v3578 : tensor<64x128x28x28xf32>
    %v3581 = stablehlo.reshape %v3421 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3582 = stablehlo.multiply %v3581, %v3580 : tensor<64x128x28x28xf32>
    %v3583 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3584 = stablehlo.reduce(%v3582 init: %v3583) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3585 = stablehlo.reshape %v3421 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3586 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3587 = stablehlo.reduce(%v3585 init: %v3586) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3588 = stablehlo.reshape %v3535 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3589 = stablehlo.reshape %v644 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3590 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3591 = stablehlo.compare GT, %v3589, %v3590 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3592 = stablehlo.select %v3591, %v3588, %v3590 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3593 = stablehlo.reshape %v3592 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3594 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3595 = stablehlo.slice %v625 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3596 = stablehlo.slice %v625 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3597 = stablehlo.broadcast_in_dim %v3595, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3598 = stablehlo.broadcast_in_dim %v3596, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3599 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3600 = stablehlo.add %v3598, %v3599 : tensor<64x128x28x28xf32>
    %v3601 = stablehlo.rsqrt %v3600 : tensor<64x128x28x28xf32>
    %v3602 = stablehlo.subtract %v3594, %v3597 : tensor<64x128x28x28xf32>
    %v3603 = stablehlo.multiply %v3602, %v3601 : tensor<64x128x28x28xf32>
    %v3604 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3605 = stablehlo.reshape %v3593 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3606 = stablehlo.multiply %v3604, %v3605 : tensor<64x128x28x28xf32>
    %v3607 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3608 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3609 = stablehlo.reduce(%v3606 init: %v3607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3610 = stablehlo.divide %v3609, %v3608 : tensor<128xf32>
    %v3611 = stablehlo.multiply %v3603, %v3606 : tensor<64x128x28x28xf32>
    %v3612 = stablehlo.reduce(%v3611 init: %v3607) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3613 = stablehlo.divide %v3612, %v3608 : tensor<128xf32>
    %v3614 = stablehlo.concatenate %v3610, %v3613, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3615 = stablehlo.concatenate %v625, %v3614, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g2dst = "stablehlo.all_reduce"(%v3615) ({
    ^bb0(%aras2b1g2dst: tensor<f32>, %arbs2b1g2dst: tensor<f32>):
      %aradds2b1g2dst = stablehlo.add %aras2b1g2dst, %arbs2b1g2dst : tensor<f32>
      stablehlo.return %aradds2b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g2dst = stablehlo.divide %arsums2b1g2dst, %arns2b1g2dst : tensor<512xf32>
    %v3616 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3617 = stablehlo.slice %armeans2b1g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3618 = stablehlo.slice %armeans2b1g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3619 = stablehlo.slice %armeans2b1g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3620 = stablehlo.slice %armeans2b1g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3621 = stablehlo.broadcast_in_dim %v3617, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3622 = stablehlo.broadcast_in_dim %v3618, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3623 = stablehlo.broadcast_in_dim %v3619, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3624 = stablehlo.broadcast_in_dim %v3620, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3625 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3626 = stablehlo.add %v3622, %v3625 : tensor<64x128x28x28xf32>
    %v3627 = stablehlo.rsqrt %v3626 : tensor<64x128x28x28xf32>
    %v3628 = stablehlo.subtract %v3616, %v3621 : tensor<64x128x28x28xf32>
    %v3629 = stablehlo.multiply %v3628, %v3627 : tensor<64x128x28x28xf32>
    %v3630 = stablehlo.broadcast_in_dim %s2b1g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3631 = stablehlo.reshape %v3593 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3632 = stablehlo.multiply %v3630, %v3631 : tensor<64x128x28x28xf32>
    %v3633 = stablehlo.subtract %v3632, %v3623 : tensor<64x128x28x28xf32>
    %v3634 = stablehlo.multiply %v3629, %v3624 : tensor<64x128x28x28xf32>
    %v3635 = stablehlo.subtract %v3633, %v3634 : tensor<64x128x28x28xf32>
    %v3636 = stablehlo.multiply %v3627, %v3635 : tensor<64x128x28x28xf32>
    %v3637 = stablehlo.reshape %v3636 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3638 = stablehlo.reshape %v3637 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3639 = stablehlo.reverse %s2b1W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3640 = stablehlo.transpose %v3639, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3641 = stablehlo.convert %v3638 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3642 = stablehlo.convert %v3640 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3643 = stablehlo.convolution(%v3641, %v3642)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3644 = stablehlo.convert %v3643 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3645 = stablehlo.reshape %v3644 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3646 = stablehlo.reshape %v3645 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3647 = stablehlo.reshape %v596 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3648 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3649 = stablehlo.compare GT, %v3647, %v3648 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3650 = stablehlo.select %v3649, %v3646, %v3648 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3651 = stablehlo.reshape %v3650 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3652 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3653 = stablehlo.slice %v581 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3654 = stablehlo.slice %v581 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3655 = stablehlo.broadcast_in_dim %v3653, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3656 = stablehlo.broadcast_in_dim %v3654, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3657 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3658 = stablehlo.add %v3656, %v3657 : tensor<64x128x28x28xf32>
    %v3659 = stablehlo.rsqrt %v3658 : tensor<64x128x28x28xf32>
    %v3660 = stablehlo.subtract %v3652, %v3655 : tensor<64x128x28x28xf32>
    %v3661 = stablehlo.multiply %v3660, %v3659 : tensor<64x128x28x28xf32>
    %v3662 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3663 = stablehlo.reshape %v3651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3664 = stablehlo.multiply %v3662, %v3663 : tensor<64x128x28x28xf32>
    %v3665 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3666 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3667 = stablehlo.reduce(%v3664 init: %v3665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3668 = stablehlo.divide %v3667, %v3666 : tensor<128xf32>
    %v3669 = stablehlo.multiply %v3661, %v3664 : tensor<64x128x28x28xf32>
    %v3670 = stablehlo.reduce(%v3669 init: %v3665) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3671 = stablehlo.divide %v3670, %v3666 : tensor<128xf32>
    %v3672 = stablehlo.concatenate %v3668, %v3671, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3673 = stablehlo.concatenate %v581, %v3672, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b1g1dst = "stablehlo.all_reduce"(%v3673) ({
    ^bb0(%aras2b1g1dst: tensor<f32>, %arbs2b1g1dst: tensor<f32>):
      %aradds2b1g1dst = stablehlo.add %aras2b1g1dst, %arbs2b1g1dst : tensor<f32>
      stablehlo.return %aradds2b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b1g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b1g1dst = stablehlo.divide %arsums2b1g1dst, %arns2b1g1dst : tensor<512xf32>
    %v3674 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3675 = stablehlo.slice %armeans2b1g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3676 = stablehlo.slice %armeans2b1g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3677 = stablehlo.slice %armeans2b1g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3678 = stablehlo.slice %armeans2b1g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3679 = stablehlo.broadcast_in_dim %v3675, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3680 = stablehlo.broadcast_in_dim %v3676, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3681 = stablehlo.broadcast_in_dim %v3677, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3682 = stablehlo.broadcast_in_dim %v3678, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3683 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3684 = stablehlo.add %v3680, %v3683 : tensor<64x128x28x28xf32>
    %v3685 = stablehlo.rsqrt %v3684 : tensor<64x128x28x28xf32>
    %v3686 = stablehlo.subtract %v3674, %v3679 : tensor<64x128x28x28xf32>
    %v3687 = stablehlo.multiply %v3686, %v3685 : tensor<64x128x28x28xf32>
    %v3688 = stablehlo.broadcast_in_dim %s2b1g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3689 = stablehlo.reshape %v3651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3690 = stablehlo.multiply %v3688, %v3689 : tensor<64x128x28x28xf32>
    %v3691 = stablehlo.subtract %v3690, %v3681 : tensor<64x128x28x28xf32>
    %v3692 = stablehlo.multiply %v3687, %v3682 : tensor<64x128x28x28xf32>
    %v3693 = stablehlo.subtract %v3691, %v3692 : tensor<64x128x28x28xf32>
    %v3694 = stablehlo.multiply %v3685, %v3693 : tensor<64x128x28x28xf32>
    %v3695 = stablehlo.reshape %v3694 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3696 = stablehlo.reshape %v3695 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3697 = stablehlo.reverse %s2b1W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3698 = stablehlo.transpose %v3697, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3699 = stablehlo.convert %v3696 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3700 = stablehlo.convert %v3698 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3701 = stablehlo.convolution(%v3699, %v3700)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3702 = stablehlo.convert %v3701 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3703 = stablehlo.reshape %v3702 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3704 = stablehlo.reshape %v3703 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3705 = stablehlo.reshape %v3593 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3706 = stablehlo.add %v3704, %v3705 : tensor<64x128x28x28xf32>
    %v3707 = stablehlo.reshape %v3706 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3708 = stablehlo.reshape %v554 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3709 = stablehlo.reshape %v3695 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3710 = stablehlo.transpose %v3708, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3711 = stablehlo.transpose %v3709, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3712 = stablehlo.convert %v3710 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3713 = stablehlo.convert %v3711 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3714 = stablehlo.convolution(%v3712, %v3713)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3715 = stablehlo.convert %v3714 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3716 = stablehlo.transpose %v3715, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3717 = stablehlo.reshape %v562 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3718 = stablehlo.slice %v581 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3719 = stablehlo.slice %v581 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3720 = stablehlo.broadcast_in_dim %v3718, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3721 = stablehlo.broadcast_in_dim %v3719, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3722 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3723 = stablehlo.add %v3721, %v3722 : tensor<64x128x28x28xf32>
    %v3724 = stablehlo.rsqrt %v3723 : tensor<64x128x28x28xf32>
    %v3725 = stablehlo.subtract %v3717, %v3720 : tensor<64x128x28x28xf32>
    %v3726 = stablehlo.multiply %v3725, %v3724 : tensor<64x128x28x28xf32>
    %v3727 = stablehlo.reshape %v3651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3728 = stablehlo.multiply %v3727, %v3726 : tensor<64x128x28x28xf32>
    %v3729 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3730 = stablehlo.reduce(%v3728 init: %v3729) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3731 = stablehlo.reshape %v3651 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3732 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3733 = stablehlo.reduce(%v3731 init: %v3732) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3734 = stablehlo.reshape %v598 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3735 = stablehlo.reshape %v3637 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3736 = stablehlo.transpose %v3734, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3737 = stablehlo.transpose %v3735, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3738 = stablehlo.convert %v3736 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3739 = stablehlo.convert %v3737 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3740 = stablehlo.convolution(%v3738, %v3739)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3741 = stablehlo.convert %v3740 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3742 = stablehlo.transpose %v3741, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3743 = stablehlo.reshape %v606 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3744 = stablehlo.slice %v625 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3745 = stablehlo.slice %v625 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3746 = stablehlo.broadcast_in_dim %v3744, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3747 = stablehlo.broadcast_in_dim %v3745, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3748 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3749 = stablehlo.add %v3747, %v3748 : tensor<64x128x28x28xf32>
    %v3750 = stablehlo.rsqrt %v3749 : tensor<64x128x28x28xf32>
    %v3751 = stablehlo.subtract %v3743, %v3746 : tensor<64x128x28x28xf32>
    %v3752 = stablehlo.multiply %v3751, %v3750 : tensor<64x128x28x28xf32>
    %v3753 = stablehlo.reshape %v3593 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3754 = stablehlo.multiply %v3753, %v3752 : tensor<64x128x28x28xf32>
    %v3755 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3756 = stablehlo.reduce(%v3754 init: %v3755) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3757 = stablehlo.reshape %v3593 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3758 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3759 = stablehlo.reduce(%v3757 init: %v3758) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3760 = stablehlo.reshape %v3707 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3761 = stablehlo.reshape %v550 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3762 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3763 = stablehlo.compare GT, %v3761, %v3762 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3764 = stablehlo.select %v3763, %v3760, %v3762 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3765 = stablehlo.reshape %v3764 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3766 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3767 = stablehlo.slice %v531 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3768 = stablehlo.slice %v531 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3769 = stablehlo.broadcast_in_dim %v3767, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3770 = stablehlo.broadcast_in_dim %v3768, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3771 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3772 = stablehlo.add %v3770, %v3771 : tensor<64x128x28x28xf32>
    %v3773 = stablehlo.rsqrt %v3772 : tensor<64x128x28x28xf32>
    %v3774 = stablehlo.subtract %v3766, %v3769 : tensor<64x128x28x28xf32>
    %v3775 = stablehlo.multiply %v3774, %v3773 : tensor<64x128x28x28xf32>
    %v3776 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3777 = stablehlo.reshape %v3765 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3778 = stablehlo.multiply %v3776, %v3777 : tensor<64x128x28x28xf32>
    %v3779 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3780 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3781 = stablehlo.reduce(%v3778 init: %v3779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3782 = stablehlo.divide %v3781, %v3780 : tensor<128xf32>
    %v3783 = stablehlo.multiply %v3775, %v3778 : tensor<64x128x28x28xf32>
    %v3784 = stablehlo.reduce(%v3783 init: %v3779) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3785 = stablehlo.divide %v3784, %v3780 : tensor<128xf32>
    %v3786 = stablehlo.concatenate %v3782, %v3785, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3787 = stablehlo.concatenate %v531, %v3786, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g2dst = "stablehlo.all_reduce"(%v3787) ({
    ^bb0(%aras2b0g2dst: tensor<f32>, %arbs2b0g2dst: tensor<f32>):
      %aradds2b0g2dst = stablehlo.add %aras2b0g2dst, %arbs2b0g2dst : tensor<f32>
      stablehlo.return %aradds2b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g2dst = stablehlo.divide %arsums2b0g2dst, %arns2b0g2dst : tensor<512xf32>
    %v3788 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3789 = stablehlo.slice %armeans2b0g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3790 = stablehlo.slice %armeans2b0g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3791 = stablehlo.slice %armeans2b0g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3792 = stablehlo.slice %armeans2b0g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3793 = stablehlo.broadcast_in_dim %v3789, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3794 = stablehlo.broadcast_in_dim %v3790, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3795 = stablehlo.broadcast_in_dim %v3791, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3796 = stablehlo.broadcast_in_dim %v3792, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3797 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3798 = stablehlo.add %v3794, %v3797 : tensor<64x128x28x28xf32>
    %v3799 = stablehlo.rsqrt %v3798 : tensor<64x128x28x28xf32>
    %v3800 = stablehlo.subtract %v3788, %v3793 : tensor<64x128x28x28xf32>
    %v3801 = stablehlo.multiply %v3800, %v3799 : tensor<64x128x28x28xf32>
    %v3802 = stablehlo.broadcast_in_dim %s2b0g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3803 = stablehlo.reshape %v3765 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3804 = stablehlo.multiply %v3802, %v3803 : tensor<64x128x28x28xf32>
    %v3805 = stablehlo.subtract %v3804, %v3795 : tensor<64x128x28x28xf32>
    %v3806 = stablehlo.multiply %v3801, %v3796 : tensor<64x128x28x28xf32>
    %v3807 = stablehlo.subtract %v3805, %v3806 : tensor<64x128x28x28xf32>
    %v3808 = stablehlo.multiply %v3799, %v3807 : tensor<64x128x28x28xf32>
    %v3809 = stablehlo.reshape %v3808 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3810 = stablehlo.reshape %v3809 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3811 = stablehlo.reverse %s2b0W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3812 = stablehlo.transpose %v3811, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3813 = stablehlo.convert %v3810 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3814 = stablehlo.convert %v3812 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3815 = stablehlo.convolution(%v3813, %v3814)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3816 = stablehlo.convert %v3815 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3817 = stablehlo.reshape %v3816 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3818 = stablehlo.reshape %v3817 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3819 = stablehlo.reshape %v502 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3820 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3821 = stablehlo.compare GT, %v3819, %v3820 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3822 = stablehlo.select %v3821, %v3818, %v3820 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3823 = stablehlo.reshape %v3822 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3824 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3825 = stablehlo.slice %v487 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3826 = stablehlo.slice %v487 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3827 = stablehlo.broadcast_in_dim %v3825, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3828 = stablehlo.broadcast_in_dim %v3826, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3829 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3830 = stablehlo.add %v3828, %v3829 : tensor<64x128x28x28xf32>
    %v3831 = stablehlo.rsqrt %v3830 : tensor<64x128x28x28xf32>
    %v3832 = stablehlo.subtract %v3824, %v3827 : tensor<64x128x28x28xf32>
    %v3833 = stablehlo.multiply %v3832, %v3831 : tensor<64x128x28x28xf32>
    %v3834 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3835 = stablehlo.reshape %v3823 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3836 = stablehlo.multiply %v3834, %v3835 : tensor<64x128x28x28xf32>
    %v3837 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3838 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3839 = stablehlo.reduce(%v3836 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3840 = stablehlo.divide %v3839, %v3838 : tensor<128xf32>
    %v3841 = stablehlo.multiply %v3833, %v3836 : tensor<64x128x28x28xf32>
    %v3842 = stablehlo.reduce(%v3841 init: %v3837) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3843 = stablehlo.divide %v3842, %v3838 : tensor<128xf32>
    %v3844 = stablehlo.concatenate %v3840, %v3843, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3845 = stablehlo.concatenate %v487, %v3844, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsums2b0g1dst = "stablehlo.all_reduce"(%v3845) ({
    ^bb0(%aras2b0g1dst: tensor<f32>, %arbs2b0g1dst: tensor<f32>):
      %aradds2b0g1dst = stablehlo.add %aras2b0g1dst, %arbs2b0g1dst : tensor<f32>
      stablehlo.return %aradds2b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns2b0g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans2b0g1dst = stablehlo.divide %arsums2b0g1dst, %arns2b0g1dst : tensor<512xf32>
    %v3846 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3847 = stablehlo.slice %armeans2b0g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3848 = stablehlo.slice %armeans2b0g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3849 = stablehlo.slice %armeans2b0g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3850 = stablehlo.slice %armeans2b0g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3851 = stablehlo.broadcast_in_dim %v3847, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3852 = stablehlo.broadcast_in_dim %v3848, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3853 = stablehlo.broadcast_in_dim %v3849, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3854 = stablehlo.broadcast_in_dim %v3850, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3855 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3856 = stablehlo.add %v3852, %v3855 : tensor<64x128x28x28xf32>
    %v3857 = stablehlo.rsqrt %v3856 : tensor<64x128x28x28xf32>
    %v3858 = stablehlo.subtract %v3846, %v3851 : tensor<64x128x28x28xf32>
    %v3859 = stablehlo.multiply %v3858, %v3857 : tensor<64x128x28x28xf32>
    %v3860 = stablehlo.broadcast_in_dim %s2b0g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3861 = stablehlo.reshape %v3823 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3862 = stablehlo.multiply %v3860, %v3861 : tensor<64x128x28x28xf32>
    %v3863 = stablehlo.subtract %v3862, %v3853 : tensor<64x128x28x28xf32>
    %v3864 = stablehlo.multiply %v3859, %v3854 : tensor<64x128x28x28xf32>
    %v3865 = stablehlo.subtract %v3863, %v3864 : tensor<64x128x28x28xf32>
    %v3866 = stablehlo.multiply %v3857, %v3865 : tensor<64x128x28x28xf32>
    %v3867 = stablehlo.reshape %v3866 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3868 = stablehlo.reshape %v3867 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3869 = stablehlo.reverse %s2b0W1, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3870 = stablehlo.transpose %v3869, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3871 = stablehlo.convert %v3868 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3872 = stablehlo.convert %v3870 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3873 = stablehlo.convolution(%v3871, %v3872)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3874 = stablehlo.convert %v3873 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3875 = stablehlo.reshape %v3874 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3876 = stablehlo.reshape %v3875 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3877 = stablehlo.reshape %v3765 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3878 = stablehlo.add %v3876, %v3877 : tensor<64x128x28x28xf32>
    %v3879 = stablehlo.reshape %v3878 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3880 = stablehlo.reshape %v460 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3881 = stablehlo.reshape %v3867 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3882 = stablehlo.transpose %v3880, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3883 = stablehlo.transpose %v3881, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3884 = stablehlo.convert %v3882 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3885 = stablehlo.convert %v3883 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3886 = stablehlo.convolution(%v3884, %v3885)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3887 = stablehlo.convert %v3886 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3888 = stablehlo.transpose %v3887, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3889 = stablehlo.reshape %v468 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3890 = stablehlo.slice %v487 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3891 = stablehlo.slice %v487 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3892 = stablehlo.broadcast_in_dim %v3890, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3893 = stablehlo.broadcast_in_dim %v3891, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3894 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3895 = stablehlo.add %v3893, %v3894 : tensor<64x128x28x28xf32>
    %v3896 = stablehlo.rsqrt %v3895 : tensor<64x128x28x28xf32>
    %v3897 = stablehlo.subtract %v3889, %v3892 : tensor<64x128x28x28xf32>
    %v3898 = stablehlo.multiply %v3897, %v3896 : tensor<64x128x28x28xf32>
    %v3899 = stablehlo.reshape %v3823 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3900 = stablehlo.multiply %v3899, %v3898 : tensor<64x128x28x28xf32>
    %v3901 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3902 = stablehlo.reduce(%v3900 init: %v3901) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3903 = stablehlo.reshape %v3823 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3904 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3905 = stablehlo.reduce(%v3903 init: %v3904) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3906 = stablehlo.reshape %v504 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3907 = stablehlo.reshape %v3809 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3908 = stablehlo.transpose %v3906, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3909 = stablehlo.transpose %v3907, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v3910 = stablehlo.convert %v3908 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3911 = stablehlo.convert %v3909 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v3912 = stablehlo.convolution(%v3910, %v3911)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v3913 = stablehlo.convert %v3912 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v3914 = stablehlo.transpose %v3913, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3915 = stablehlo.reshape %v512 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3916 = stablehlo.slice %v531 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3917 = stablehlo.slice %v531 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3918 = stablehlo.broadcast_in_dim %v3916, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3919 = stablehlo.broadcast_in_dim %v3917, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3920 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3921 = stablehlo.add %v3919, %v3920 : tensor<64x128x28x28xf32>
    %v3922 = stablehlo.rsqrt %v3921 : tensor<64x128x28x28xf32>
    %v3923 = stablehlo.subtract %v3915, %v3918 : tensor<64x128x28x28xf32>
    %v3924 = stablehlo.multiply %v3923, %v3922 : tensor<64x128x28x28xf32>
    %v3925 = stablehlo.reshape %v3765 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3926 = stablehlo.multiply %v3925, %v3924 : tensor<64x128x28x28xf32>
    %v3927 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3928 = stablehlo.reduce(%v3926 init: %v3927) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3929 = stablehlo.reshape %v3765 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3930 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3931 = stablehlo.reduce(%v3929 init: %v3930) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3932 = stablehlo.reshape %v3879 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3933 = stablehlo.reshape %v458 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3934 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3935 = stablehlo.compare GT, %v3933, %v3934 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3936 = stablehlo.select %v3935, %v3932, %v3934 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3937 = stablehlo.reshape %v3936 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3938 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3939 = stablehlo.slice %v400 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3940 = stablehlo.slice %v400 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3941 = stablehlo.broadcast_in_dim %v3939, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3942 = stablehlo.broadcast_in_dim %v3940, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3943 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3944 = stablehlo.add %v3942, %v3943 : tensor<64x128x28x28xf32>
    %v3945 = stablehlo.rsqrt %v3944 : tensor<64x128x28x28xf32>
    %v3946 = stablehlo.subtract %v3938, %v3941 : tensor<64x128x28x28xf32>
    %v3947 = stablehlo.multiply %v3946, %v3945 : tensor<64x128x28x28xf32>
    %v3948 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3949 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3950 = stablehlo.multiply %v3948, %v3949 : tensor<64x128x28x28xf32>
    %v3951 = stablehlo.constant dense<0.0> : tensor<f32>
    %v3952 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v3953 = stablehlo.reduce(%v3950 init: %v3951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3954 = stablehlo.divide %v3953, %v3952 : tensor<128xf32>
    %v3955 = stablehlo.multiply %v3947, %v3950 : tensor<64x128x28x28xf32>
    %v3956 = stablehlo.reduce(%v3955 init: %v3951) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v3957 = stablehlo.divide %v3956, %v3952 : tensor<128xf32>
    %v3958 = stablehlo.concatenate %v3954, %v3957, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v3959 = stablehlo.concatenate %v400, %v3958, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g2dst = "stablehlo.all_reduce"(%v3959) ({
    ^bb0(%arad2g2dst: tensor<f32>, %arbd2g2dst: tensor<f32>):
      %araddd2g2dst = stablehlo.add %arad2g2dst, %arbd2g2dst : tensor<f32>
      stablehlo.return %araddd2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g2dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand2g2dst = stablehlo.divide %arsumd2g2dst, %arnd2g2dst : tensor<512xf32>
    %v3960 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3961 = stablehlo.slice %armeand2g2dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v3962 = stablehlo.slice %armeand2g2dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v3963 = stablehlo.slice %armeand2g2dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v3964 = stablehlo.slice %armeand2g2dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v3965 = stablehlo.broadcast_in_dim %v3961, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3966 = stablehlo.broadcast_in_dim %v3962, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3967 = stablehlo.broadcast_in_dim %v3963, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3968 = stablehlo.broadcast_in_dim %v3964, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3969 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v3970 = stablehlo.add %v3966, %v3969 : tensor<64x128x28x28xf32>
    %v3971 = stablehlo.rsqrt %v3970 : tensor<64x128x28x28xf32>
    %v3972 = stablehlo.subtract %v3960, %v3965 : tensor<64x128x28x28xf32>
    %v3973 = stablehlo.multiply %v3972, %v3971 : tensor<64x128x28x28xf32>
    %v3974 = stablehlo.broadcast_in_dim %d2g2, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v3975 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3976 = stablehlo.multiply %v3974, %v3975 : tensor<64x128x28x28xf32>
    %v3977 = stablehlo.subtract %v3976, %v3967 : tensor<64x128x28x28xf32>
    %v3978 = stablehlo.multiply %v3973, %v3968 : tensor<64x128x28x28xf32>
    %v3979 = stablehlo.subtract %v3977, %v3978 : tensor<64x128x28x28xf32>
    %v3980 = stablehlo.multiply %v3971, %v3979 : tensor<64x128x28x28xf32>
    %v3981 = stablehlo.reshape %v3980 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3982 = stablehlo.reshape %v3981 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3983 = stablehlo.reverse %d2W2, dims = [2, 3] : tensor<128x128x3x3xf32>
    %v3984 = stablehlo.transpose %v3983, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v3985 = stablehlo.convert %v3982 : (tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xbf16>
    %v3986 = stablehlo.convert %v3984 : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xbf16>
    %v3987 = stablehlo.convolution(%v3985, %v3986)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x28x28xbf16>, tensor<128x128x3x3xbf16>) -> tensor<64x128x28x28xbf16>
    %v3988 = stablehlo.convert %v3987 : (tensor<64x128x28x28xbf16>) -> tensor<64x128x28x28xf32>
    %v3989 = stablehlo.reshape %v3988 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3990 = stablehlo.reshape %v3989 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3991 = stablehlo.reshape %v371 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3992 = stablehlo.constant dense<0.0> : tensor<64x128x28x28xf32>
    %v3993 = stablehlo.compare GT, %v3991, %v3992 : (tensor<64x128x28x28xf32>, tensor<64x128x28x28xf32>) -> tensor<64x128x28x28xi1>
    %v3994 = stablehlo.select %v3993, %v3990, %v3992 : tensor<64x128x28x28xi1>, tensor<64x128x28x28xf32>
    %v3995 = stablehlo.reshape %v3994 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v3996 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v3997 = stablehlo.slice %v356 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v3998 = stablehlo.slice %v356 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v3999 = stablehlo.broadcast_in_dim %v3997, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4000 = stablehlo.broadcast_in_dim %v3998, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4001 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4002 = stablehlo.add %v4000, %v4001 : tensor<64x128x28x28xf32>
    %v4003 = stablehlo.rsqrt %v4002 : tensor<64x128x28x28xf32>
    %v4004 = stablehlo.subtract %v3996, %v3999 : tensor<64x128x28x28xf32>
    %v4005 = stablehlo.multiply %v4004, %v4003 : tensor<64x128x28x28xf32>
    %v4006 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4007 = stablehlo.reshape %v3995 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4008 = stablehlo.multiply %v4006, %v4007 : tensor<64x128x28x28xf32>
    %v4009 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4010 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4011 = stablehlo.reduce(%v4008 init: %v4009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4012 = stablehlo.divide %v4011, %v4010 : tensor<128xf32>
    %v4013 = stablehlo.multiply %v4005, %v4008 : tensor<64x128x28x28xf32>
    %v4014 = stablehlo.reduce(%v4013 init: %v4009) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4015 = stablehlo.divide %v4014, %v4010 : tensor<128xf32>
    %v4016 = stablehlo.concatenate %v4012, %v4015, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4017 = stablehlo.concatenate %v356, %v4016, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2g1dst = "stablehlo.all_reduce"(%v4017) ({
    ^bb0(%arad2g1dst: tensor<f32>, %arbd2g1dst: tensor<f32>):
      %araddd2g1dst = stablehlo.add %arad2g1dst, %arbd2g1dst : tensor<f32>
      stablehlo.return %araddd2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2g1dst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand2g1dst = stablehlo.divide %arsumd2g1dst, %arnd2g1dst : tensor<512xf32>
    %v4018 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4019 = stablehlo.slice %armeand2g1dst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4020 = stablehlo.slice %armeand2g1dst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4021 = stablehlo.slice %armeand2g1dst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4022 = stablehlo.slice %armeand2g1dst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4023 = stablehlo.broadcast_in_dim %v4019, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4024 = stablehlo.broadcast_in_dim %v4020, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4025 = stablehlo.broadcast_in_dim %v4021, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4026 = stablehlo.broadcast_in_dim %v4022, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4027 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4028 = stablehlo.add %v4024, %v4027 : tensor<64x128x28x28xf32>
    %v4029 = stablehlo.rsqrt %v4028 : tensor<64x128x28x28xf32>
    %v4030 = stablehlo.subtract %v4018, %v4023 : tensor<64x128x28x28xf32>
    %v4031 = stablehlo.multiply %v4030, %v4029 : tensor<64x128x28x28xf32>
    %v4032 = stablehlo.broadcast_in_dim %d2g1, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4033 = stablehlo.reshape %v3995 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4034 = stablehlo.multiply %v4032, %v4033 : tensor<64x128x28x28xf32>
    %v4035 = stablehlo.subtract %v4034, %v4025 : tensor<64x128x28x28xf32>
    %v4036 = stablehlo.multiply %v4031, %v4026 : tensor<64x128x28x28xf32>
    %v4037 = stablehlo.subtract %v4035, %v4036 : tensor<64x128x28x28xf32>
    %v4038 = stablehlo.multiply %v4029, %v4037 : tensor<64x128x28x28xf32>
    %v4039 = stablehlo.reshape %v4038 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4040 = stablehlo.reshape %v4039 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4041 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4042 = stablehlo.pad %v4040, %v4041, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4043 = stablehlo.reverse %d2W1, dims = [2, 3] : tensor<128x64x3x3xf32>
    %v4044 = stablehlo.transpose %v4043, dims = [1, 0, 2, 3] : (tensor<128x64x3x3xf32>) -> tensor<64x128x3x3xf32>
    %v4045 = stablehlo.convert %v4042 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4046 = stablehlo.convert %v4044 : (tensor<64x128x3x3xf32>) -> tensor<64x128x3x3xbf16>
    %v4047 = stablehlo.convolution(%v4045, %v4046)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<64x128x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4048 = stablehlo.convert %v4047 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4049 = stablehlo.reshape %v4048 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4050 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4051 = stablehlo.slice %v442 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4052 = stablehlo.slice %v442 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4053 = stablehlo.broadcast_in_dim %v4051, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4054 = stablehlo.broadcast_in_dim %v4052, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4055 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4056 = stablehlo.add %v4054, %v4055 : tensor<64x128x28x28xf32>
    %v4057 = stablehlo.rsqrt %v4056 : tensor<64x128x28x28xf32>
    %v4058 = stablehlo.subtract %v4050, %v4053 : tensor<64x128x28x28xf32>
    %v4059 = stablehlo.multiply %v4058, %v4057 : tensor<64x128x28x28xf32>
    %v4060 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4061 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4062 = stablehlo.multiply %v4060, %v4061 : tensor<64x128x28x28xf32>
    %v4063 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4064 = stablehlo.constant dense<50176.0> : tensor<128xf32>
    %v4065 = stablehlo.reduce(%v4062 init: %v4063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4066 = stablehlo.divide %v4065, %v4064 : tensor<128xf32>
    %v4067 = stablehlo.multiply %v4059, %v4062 : tensor<64x128x28x28xf32>
    %v4068 = stablehlo.reduce(%v4067 init: %v4063) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4069 = stablehlo.divide %v4068, %v4064 : tensor<128xf32>
    %v4070 = stablehlo.concatenate %v4066, %v4069, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %v4071 = stablehlo.concatenate %v442, %v4070, dim = 0 : (tensor<256xf32>, tensor<256xf32>) -> tensor<512xf32>
    %arsumd2gpdst = "stablehlo.all_reduce"(%v4071) ({
    ^bb0(%arad2gpdst: tensor<f32>, %arbd2gpdst: tensor<f32>):
      %araddd2gpdst = stablehlo.add %arad2gpdst, %arbd2gpdst : tensor<f32>
      stablehlo.return %araddd2gpdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd2gpdst = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand2gpdst = stablehlo.divide %arsumd2gpdst, %arnd2gpdst : tensor<512xf32>
    %v4072 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4073 = stablehlo.slice %armeand2gpdst [0:128] : (tensor<512xf32>) -> tensor<128xf32>
    %v4074 = stablehlo.slice %armeand2gpdst [128:256] : (tensor<512xf32>) -> tensor<128xf32>
    %v4075 = stablehlo.slice %armeand2gpdst [256:384] : (tensor<512xf32>) -> tensor<128xf32>
    %v4076 = stablehlo.slice %armeand2gpdst [384:512] : (tensor<512xf32>) -> tensor<128xf32>
    %v4077 = stablehlo.broadcast_in_dim %v4073, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4078 = stablehlo.broadcast_in_dim %v4074, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4079 = stablehlo.broadcast_in_dim %v4075, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4080 = stablehlo.broadcast_in_dim %v4076, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4081 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4082 = stablehlo.add %v4078, %v4081 : tensor<64x128x28x28xf32>
    %v4083 = stablehlo.rsqrt %v4082 : tensor<64x128x28x28xf32>
    %v4084 = stablehlo.subtract %v4072, %v4077 : tensor<64x128x28x28xf32>
    %v4085 = stablehlo.multiply %v4084, %v4083 : tensor<64x128x28x28xf32>
    %v4086 = stablehlo.broadcast_in_dim %d2gp, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4087 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4088 = stablehlo.multiply %v4086, %v4087 : tensor<64x128x28x28xf32>
    %v4089 = stablehlo.subtract %v4088, %v4079 : tensor<64x128x28x28xf32>
    %v4090 = stablehlo.multiply %v4085, %v4080 : tensor<64x128x28x28xf32>
    %v4091 = stablehlo.subtract %v4089, %v4090 : tensor<64x128x28x28xf32>
    %v4092 = stablehlo.multiply %v4083, %v4091 : tensor<64x128x28x28xf32>
    %v4093 = stablehlo.reshape %v4092 : (tensor<64x128x28x28xf32>) -> tensor<64x100352xf32>
    %v4094 = stablehlo.reshape %v4093 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4095 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4096 = stablehlo.pad %v4094, %v4095, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4097 = stablehlo.reverse %d2Wp, dims = [2, 3] : tensor<128x64x1x1xf32>
    %v4098 = stablehlo.transpose %v4097, dims = [1, 0, 2, 3] : (tensor<128x64x1x1xf32>) -> tensor<64x128x1x1xf32>
    %v4099 = stablehlo.convert %v4096 : (tensor<64x128x56x56xf32>) -> tensor<64x128x56x56xbf16>
    %v4100 = stablehlo.convert %v4098 : (tensor<64x128x1x1xf32>) -> tensor<64x128x1x1xbf16>
    %v4101 = stablehlo.convolution(%v4099, %v4100)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x128x56x56xbf16>, tensor<64x128x1x1xbf16>) -> tensor<64x64x56x56xbf16>
    %v4102 = stablehlo.convert %v4101 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4103 = stablehlo.reshape %v4102 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4104 = stablehlo.reshape %v4049 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4105 = stablehlo.reshape %v4103 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4106 = stablehlo.add %v4104, %v4105 : tensor<64x64x56x56xf32>
    %v4107 = stablehlo.reshape %v4106 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4108 = stablehlo.reshape %v329 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4109 = stablehlo.reshape %v4039 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4110 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4111 = stablehlo.pad %v4109, %v4110, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4112 = stablehlo.transpose %v4108, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4113 = stablehlo.transpose %v4111, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4114 = stablehlo.convert %v4112 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4115 = stablehlo.convert %v4113 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4116 = stablehlo.convolution(%v4114, %v4115)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<64x128x3x3xbf16>
    %v4117 = stablehlo.convert %v4116 : (tensor<64x128x3x3xbf16>) -> tensor<64x128x3x3xf32>
    %v4118 = stablehlo.transpose %v4117, dims = [1, 0, 2, 3] : (tensor<64x128x3x3xf32>) -> tensor<128x64x3x3xf32>
    %v4119 = stablehlo.reshape %v337 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4120 = stablehlo.slice %v356 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4121 = stablehlo.slice %v356 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4122 = stablehlo.broadcast_in_dim %v4120, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4123 = stablehlo.broadcast_in_dim %v4121, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4124 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4125 = stablehlo.add %v4123, %v4124 : tensor<64x128x28x28xf32>
    %v4126 = stablehlo.rsqrt %v4125 : tensor<64x128x28x28xf32>
    %v4127 = stablehlo.subtract %v4119, %v4122 : tensor<64x128x28x28xf32>
    %v4128 = stablehlo.multiply %v4127, %v4126 : tensor<64x128x28x28xf32>
    %v4129 = stablehlo.reshape %v3995 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4130 = stablehlo.multiply %v4129, %v4128 : tensor<64x128x28x28xf32>
    %v4131 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4132 = stablehlo.reduce(%v4130 init: %v4131) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4133 = stablehlo.reshape %v3995 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4134 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4135 = stablehlo.reduce(%v4133 init: %v4134) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4136 = stablehlo.reshape %v373 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4137 = stablehlo.reshape %v3981 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4138 = stablehlo.transpose %v4136, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4139 = stablehlo.transpose %v4137, dims = [1, 0, 2, 3] : (tensor<64x128x28x28xf32>) -> tensor<128x64x28x28xf32>
    %v4140 = stablehlo.convert %v4138 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4141 = stablehlo.convert %v4139 : (tensor<128x64x28x28xf32>) -> tensor<128x64x28x28xbf16>
    %v4142 = stablehlo.convolution(%v4140, %v4141)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<128x64x28x28xbf16>, tensor<128x64x28x28xbf16>) -> tensor<128x128x3x3xbf16>
    %v4143 = stablehlo.convert %v4142 : (tensor<128x128x3x3xbf16>) -> tensor<128x128x3x3xf32>
    %v4144 = stablehlo.transpose %v4143, dims = [1, 0, 2, 3] : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %v4145 = stablehlo.reshape %v381 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4146 = stablehlo.slice %v400 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4147 = stablehlo.slice %v400 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4148 = stablehlo.broadcast_in_dim %v4146, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4149 = stablehlo.broadcast_in_dim %v4147, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4150 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4151 = stablehlo.add %v4149, %v4150 : tensor<64x128x28x28xf32>
    %v4152 = stablehlo.rsqrt %v4151 : tensor<64x128x28x28xf32>
    %v4153 = stablehlo.subtract %v4145, %v4148 : tensor<64x128x28x28xf32>
    %v4154 = stablehlo.multiply %v4153, %v4152 : tensor<64x128x28x28xf32>
    %v4155 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4156 = stablehlo.multiply %v4155, %v4154 : tensor<64x128x28x28xf32>
    %v4157 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4158 = stablehlo.reduce(%v4156 init: %v4157) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4159 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4160 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4161 = stablehlo.reduce(%v4159 init: %v4160) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4162 = stablehlo.reshape %v329 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4163 = stablehlo.reshape %v4093 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4164 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4165 = stablehlo.pad %v4163, %v4164, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<64x128x56x56xf32>
    %v4166 = stablehlo.transpose %v4162, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4167 = stablehlo.transpose %v4165, dims = [1, 0, 2, 3] : (tensor<64x128x56x56xf32>) -> tensor<128x64x56x56xf32>
    %v4168 = stablehlo.convert %v4166 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4169 = stablehlo.convert %v4167 : (tensor<128x64x56x56xf32>) -> tensor<128x64x56x56xbf16>
    %v4170 = stablehlo.convolution(%v4168, %v4169)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[0, 0], [0, 0]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<128x64x56x56xbf16>) -> tensor<64x128x1x1xbf16>
    %v4171 = stablehlo.convert %v4170 : (tensor<64x128x1x1xbf16>) -> tensor<64x128x1x1xf32>
    %v4172 = stablehlo.transpose %v4171, dims = [1, 0, 2, 3] : (tensor<64x128x1x1xf32>) -> tensor<128x64x1x1xf32>
    %v4173 = stablehlo.reshape %v423 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4174 = stablehlo.slice %v442 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4175 = stablehlo.slice %v442 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4176 = stablehlo.broadcast_in_dim %v4174, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4177 = stablehlo.broadcast_in_dim %v4175, dims = [1] : (tensor<128xf32>) -> tensor<64x128x28x28xf32>
    %v4178 = stablehlo.constant dense<1.0e-05> : tensor<64x128x28x28xf32>
    %v4179 = stablehlo.add %v4177, %v4178 : tensor<64x128x28x28xf32>
    %v4180 = stablehlo.rsqrt %v4179 : tensor<64x128x28x28xf32>
    %v4181 = stablehlo.subtract %v4173, %v4176 : tensor<64x128x28x28xf32>
    %v4182 = stablehlo.multiply %v4181, %v4180 : tensor<64x128x28x28xf32>
    %v4183 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4184 = stablehlo.multiply %v4183, %v4182 : tensor<64x128x28x28xf32>
    %v4185 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4186 = stablehlo.reduce(%v4184 init: %v4185) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4187 = stablehlo.reshape %v3937 : (tensor<64x100352xf32>) -> tensor<64x128x28x28xf32>
    %v4188 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4189 = stablehlo.reduce(%v4187 init: %v4188) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x128x28x28xf32>, tensor<f32>) -> tensor<128xf32>
    %v4190 = stablehlo.reshape %v4107 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4191 = stablehlo.reshape %v325 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4192 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4193 = stablehlo.compare GT, %v4191, %v4192 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4194 = stablehlo.select %v4193, %v4190, %v4192 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4195 = stablehlo.reshape %v4194 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4196 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4197 = stablehlo.slice %v306 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4198 = stablehlo.slice %v306 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4199 = stablehlo.broadcast_in_dim %v4197, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4200 = stablehlo.broadcast_in_dim %v4198, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4201 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4202 = stablehlo.add %v4200, %v4201 : tensor<64x64x56x56xf32>
    %v4203 = stablehlo.rsqrt %v4202 : tensor<64x64x56x56xf32>
    %v4204 = stablehlo.subtract %v4196, %v4199 : tensor<64x64x56x56xf32>
    %v4205 = stablehlo.multiply %v4204, %v4203 : tensor<64x64x56x56xf32>
    %v4206 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4207 = stablehlo.reshape %v4195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4208 = stablehlo.multiply %v4206, %v4207 : tensor<64x64x56x56xf32>
    %v4209 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4210 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4211 = stablehlo.reduce(%v4208 init: %v4209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4212 = stablehlo.divide %v4211, %v4210 : tensor<64xf32>
    %v4213 = stablehlo.multiply %v4205, %v4208 : tensor<64x64x56x56xf32>
    %v4214 = stablehlo.reduce(%v4213 init: %v4209) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4215 = stablehlo.divide %v4214, %v4210 : tensor<64xf32>
    %v4216 = stablehlo.concatenate %v4212, %v4215, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4217 = stablehlo.concatenate %v306, %v4216, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g2dst = "stablehlo.all_reduce"(%v4217) ({
    ^bb0(%aras1b2g2dst: tensor<f32>, %arbs1b2g2dst: tensor<f32>):
      %aradds1b2g2dst = stablehlo.add %aras1b2g2dst, %arbs1b2g2dst : tensor<f32>
      stablehlo.return %aradds1b2g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g2dst = stablehlo.divide %arsums1b2g2dst, %arns1b2g2dst : tensor<256xf32>
    %v4218 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4219 = stablehlo.slice %armeans1b2g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4220 = stablehlo.slice %armeans1b2g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4221 = stablehlo.slice %armeans1b2g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4222 = stablehlo.slice %armeans1b2g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4223 = stablehlo.broadcast_in_dim %v4219, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4224 = stablehlo.broadcast_in_dim %v4220, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4225 = stablehlo.broadcast_in_dim %v4221, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4226 = stablehlo.broadcast_in_dim %v4222, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4227 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4228 = stablehlo.add %v4224, %v4227 : tensor<64x64x56x56xf32>
    %v4229 = stablehlo.rsqrt %v4228 : tensor<64x64x56x56xf32>
    %v4230 = stablehlo.subtract %v4218, %v4223 : tensor<64x64x56x56xf32>
    %v4231 = stablehlo.multiply %v4230, %v4229 : tensor<64x64x56x56xf32>
    %v4232 = stablehlo.broadcast_in_dim %s1b2g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4233 = stablehlo.reshape %v4195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4234 = stablehlo.multiply %v4232, %v4233 : tensor<64x64x56x56xf32>
    %v4235 = stablehlo.subtract %v4234, %v4225 : tensor<64x64x56x56xf32>
    %v4236 = stablehlo.multiply %v4231, %v4226 : tensor<64x64x56x56xf32>
    %v4237 = stablehlo.subtract %v4235, %v4236 : tensor<64x64x56x56xf32>
    %v4238 = stablehlo.multiply %v4229, %v4237 : tensor<64x64x56x56xf32>
    %v4239 = stablehlo.reshape %v4238 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4240 = stablehlo.reshape %v4239 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4241 = stablehlo.reverse %s1b2W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4242 = stablehlo.transpose %v4241, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4243 = stablehlo.convert %v4240 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4244 = stablehlo.convert %v4242 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4245 = stablehlo.convolution(%v4243, %v4244)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4246 = stablehlo.convert %v4245 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4247 = stablehlo.reshape %v4246 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4248 = stablehlo.reshape %v4247 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4249 = stablehlo.reshape %v277 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4250 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4251 = stablehlo.compare GT, %v4249, %v4250 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4252 = stablehlo.select %v4251, %v4248, %v4250 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4253 = stablehlo.reshape %v4252 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4254 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4255 = stablehlo.slice %v262 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4256 = stablehlo.slice %v262 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4257 = stablehlo.broadcast_in_dim %v4255, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4258 = stablehlo.broadcast_in_dim %v4256, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4259 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4260 = stablehlo.add %v4258, %v4259 : tensor<64x64x56x56xf32>
    %v4261 = stablehlo.rsqrt %v4260 : tensor<64x64x56x56xf32>
    %v4262 = stablehlo.subtract %v4254, %v4257 : tensor<64x64x56x56xf32>
    %v4263 = stablehlo.multiply %v4262, %v4261 : tensor<64x64x56x56xf32>
    %v4264 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4265 = stablehlo.reshape %v4253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4266 = stablehlo.multiply %v4264, %v4265 : tensor<64x64x56x56xf32>
    %v4267 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4268 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4269 = stablehlo.reduce(%v4266 init: %v4267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4270 = stablehlo.divide %v4269, %v4268 : tensor<64xf32>
    %v4271 = stablehlo.multiply %v4263, %v4266 : tensor<64x64x56x56xf32>
    %v4272 = stablehlo.reduce(%v4271 init: %v4267) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4273 = stablehlo.divide %v4272, %v4268 : tensor<64xf32>
    %v4274 = stablehlo.concatenate %v4270, %v4273, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4275 = stablehlo.concatenate %v262, %v4274, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b2g1dst = "stablehlo.all_reduce"(%v4275) ({
    ^bb0(%aras1b2g1dst: tensor<f32>, %arbs1b2g1dst: tensor<f32>):
      %aradds1b2g1dst = stablehlo.add %aras1b2g1dst, %arbs1b2g1dst : tensor<f32>
      stablehlo.return %aradds1b2g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b2g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b2g1dst = stablehlo.divide %arsums1b2g1dst, %arns1b2g1dst : tensor<256xf32>
    %v4276 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4277 = stablehlo.slice %armeans1b2g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4278 = stablehlo.slice %armeans1b2g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4279 = stablehlo.slice %armeans1b2g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4280 = stablehlo.slice %armeans1b2g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4281 = stablehlo.broadcast_in_dim %v4277, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4282 = stablehlo.broadcast_in_dim %v4278, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4283 = stablehlo.broadcast_in_dim %v4279, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4284 = stablehlo.broadcast_in_dim %v4280, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4285 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4286 = stablehlo.add %v4282, %v4285 : tensor<64x64x56x56xf32>
    %v4287 = stablehlo.rsqrt %v4286 : tensor<64x64x56x56xf32>
    %v4288 = stablehlo.subtract %v4276, %v4281 : tensor<64x64x56x56xf32>
    %v4289 = stablehlo.multiply %v4288, %v4287 : tensor<64x64x56x56xf32>
    %v4290 = stablehlo.broadcast_in_dim %s1b2g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4291 = stablehlo.reshape %v4253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4292 = stablehlo.multiply %v4290, %v4291 : tensor<64x64x56x56xf32>
    %v4293 = stablehlo.subtract %v4292, %v4283 : tensor<64x64x56x56xf32>
    %v4294 = stablehlo.multiply %v4289, %v4284 : tensor<64x64x56x56xf32>
    %v4295 = stablehlo.subtract %v4293, %v4294 : tensor<64x64x56x56xf32>
    %v4296 = stablehlo.multiply %v4287, %v4295 : tensor<64x64x56x56xf32>
    %v4297 = stablehlo.reshape %v4296 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4298 = stablehlo.reshape %v4297 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4299 = stablehlo.reverse %s1b2W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4300 = stablehlo.transpose %v4299, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4301 = stablehlo.convert %v4298 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4302 = stablehlo.convert %v4300 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4303 = stablehlo.convolution(%v4301, %v4302)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4304 = stablehlo.convert %v4303 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4305 = stablehlo.reshape %v4304 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4306 = stablehlo.reshape %v4305 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4307 = stablehlo.reshape %v4195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4308 = stablehlo.add %v4306, %v4307 : tensor<64x64x56x56xf32>
    %v4309 = stablehlo.reshape %v4308 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4310 = stablehlo.reshape %v235 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4311 = stablehlo.reshape %v4297 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4312 = stablehlo.transpose %v4310, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4313 = stablehlo.transpose %v4311, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4314 = stablehlo.convert %v4312 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4315 = stablehlo.convert %v4313 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4316 = stablehlo.convolution(%v4314, %v4315)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4317 = stablehlo.convert %v4316 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4318 = stablehlo.transpose %v4317, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4319 = stablehlo.reshape %v243 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4320 = stablehlo.slice %v262 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4321 = stablehlo.slice %v262 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4322 = stablehlo.broadcast_in_dim %v4320, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4323 = stablehlo.broadcast_in_dim %v4321, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4324 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4325 = stablehlo.add %v4323, %v4324 : tensor<64x64x56x56xf32>
    %v4326 = stablehlo.rsqrt %v4325 : tensor<64x64x56x56xf32>
    %v4327 = stablehlo.subtract %v4319, %v4322 : tensor<64x64x56x56xf32>
    %v4328 = stablehlo.multiply %v4327, %v4326 : tensor<64x64x56x56xf32>
    %v4329 = stablehlo.reshape %v4253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4330 = stablehlo.multiply %v4329, %v4328 : tensor<64x64x56x56xf32>
    %v4331 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4332 = stablehlo.reduce(%v4330 init: %v4331) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4333 = stablehlo.reshape %v4253 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4334 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4335 = stablehlo.reduce(%v4333 init: %v4334) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4336 = stablehlo.reshape %v279 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4337 = stablehlo.reshape %v4239 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4338 = stablehlo.transpose %v4336, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4339 = stablehlo.transpose %v4337, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4340 = stablehlo.convert %v4338 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4341 = stablehlo.convert %v4339 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4342 = stablehlo.convolution(%v4340, %v4341)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4343 = stablehlo.convert %v4342 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4344 = stablehlo.transpose %v4343, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4345 = stablehlo.reshape %v287 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4346 = stablehlo.slice %v306 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4347 = stablehlo.slice %v306 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4348 = stablehlo.broadcast_in_dim %v4346, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4349 = stablehlo.broadcast_in_dim %v4347, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4350 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4351 = stablehlo.add %v4349, %v4350 : tensor<64x64x56x56xf32>
    %v4352 = stablehlo.rsqrt %v4351 : tensor<64x64x56x56xf32>
    %v4353 = stablehlo.subtract %v4345, %v4348 : tensor<64x64x56x56xf32>
    %v4354 = stablehlo.multiply %v4353, %v4352 : tensor<64x64x56x56xf32>
    %v4355 = stablehlo.reshape %v4195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4356 = stablehlo.multiply %v4355, %v4354 : tensor<64x64x56x56xf32>
    %v4357 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4358 = stablehlo.reduce(%v4356 init: %v4357) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4359 = stablehlo.reshape %v4195 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4360 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4361 = stablehlo.reduce(%v4359 init: %v4360) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4362 = stablehlo.reshape %v4309 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4363 = stablehlo.reshape %v231 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4364 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4365 = stablehlo.compare GT, %v4363, %v4364 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4366 = stablehlo.select %v4365, %v4362, %v4364 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4367 = stablehlo.reshape %v4366 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4368 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4369 = stablehlo.slice %v212 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4370 = stablehlo.slice %v212 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4371 = stablehlo.broadcast_in_dim %v4369, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4372 = stablehlo.broadcast_in_dim %v4370, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4373 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4374 = stablehlo.add %v4372, %v4373 : tensor<64x64x56x56xf32>
    %v4375 = stablehlo.rsqrt %v4374 : tensor<64x64x56x56xf32>
    %v4376 = stablehlo.subtract %v4368, %v4371 : tensor<64x64x56x56xf32>
    %v4377 = stablehlo.multiply %v4376, %v4375 : tensor<64x64x56x56xf32>
    %v4378 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4379 = stablehlo.reshape %v4367 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4380 = stablehlo.multiply %v4378, %v4379 : tensor<64x64x56x56xf32>
    %v4381 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4382 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4383 = stablehlo.reduce(%v4380 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4384 = stablehlo.divide %v4383, %v4382 : tensor<64xf32>
    %v4385 = stablehlo.multiply %v4377, %v4380 : tensor<64x64x56x56xf32>
    %v4386 = stablehlo.reduce(%v4385 init: %v4381) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4387 = stablehlo.divide %v4386, %v4382 : tensor<64xf32>
    %v4388 = stablehlo.concatenate %v4384, %v4387, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4389 = stablehlo.concatenate %v212, %v4388, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g2dst = "stablehlo.all_reduce"(%v4389) ({
    ^bb0(%aras1b1g2dst: tensor<f32>, %arbs1b1g2dst: tensor<f32>):
      %aradds1b1g2dst = stablehlo.add %aras1b1g2dst, %arbs1b1g2dst : tensor<f32>
      stablehlo.return %aradds1b1g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g2dst = stablehlo.divide %arsums1b1g2dst, %arns1b1g2dst : tensor<256xf32>
    %v4390 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4391 = stablehlo.slice %armeans1b1g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4392 = stablehlo.slice %armeans1b1g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4393 = stablehlo.slice %armeans1b1g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4394 = stablehlo.slice %armeans1b1g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4395 = stablehlo.broadcast_in_dim %v4391, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4396 = stablehlo.broadcast_in_dim %v4392, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4397 = stablehlo.broadcast_in_dim %v4393, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4398 = stablehlo.broadcast_in_dim %v4394, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4399 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4400 = stablehlo.add %v4396, %v4399 : tensor<64x64x56x56xf32>
    %v4401 = stablehlo.rsqrt %v4400 : tensor<64x64x56x56xf32>
    %v4402 = stablehlo.subtract %v4390, %v4395 : tensor<64x64x56x56xf32>
    %v4403 = stablehlo.multiply %v4402, %v4401 : tensor<64x64x56x56xf32>
    %v4404 = stablehlo.broadcast_in_dim %s1b1g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4405 = stablehlo.reshape %v4367 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4406 = stablehlo.multiply %v4404, %v4405 : tensor<64x64x56x56xf32>
    %v4407 = stablehlo.subtract %v4406, %v4397 : tensor<64x64x56x56xf32>
    %v4408 = stablehlo.multiply %v4403, %v4398 : tensor<64x64x56x56xf32>
    %v4409 = stablehlo.subtract %v4407, %v4408 : tensor<64x64x56x56xf32>
    %v4410 = stablehlo.multiply %v4401, %v4409 : tensor<64x64x56x56xf32>
    %v4411 = stablehlo.reshape %v4410 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4412 = stablehlo.reshape %v4411 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4413 = stablehlo.reverse %s1b1W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4414 = stablehlo.transpose %v4413, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4415 = stablehlo.convert %v4412 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4416 = stablehlo.convert %v4414 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4417 = stablehlo.convolution(%v4415, %v4416)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4418 = stablehlo.convert %v4417 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4419 = stablehlo.reshape %v4418 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4420 = stablehlo.reshape %v4419 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4421 = stablehlo.reshape %v183 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4422 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4423 = stablehlo.compare GT, %v4421, %v4422 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4424 = stablehlo.select %v4423, %v4420, %v4422 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4425 = stablehlo.reshape %v4424 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4426 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4427 = stablehlo.slice %v168 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4428 = stablehlo.slice %v168 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4429 = stablehlo.broadcast_in_dim %v4427, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4430 = stablehlo.broadcast_in_dim %v4428, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4431 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4432 = stablehlo.add %v4430, %v4431 : tensor<64x64x56x56xf32>
    %v4433 = stablehlo.rsqrt %v4432 : tensor<64x64x56x56xf32>
    %v4434 = stablehlo.subtract %v4426, %v4429 : tensor<64x64x56x56xf32>
    %v4435 = stablehlo.multiply %v4434, %v4433 : tensor<64x64x56x56xf32>
    %v4436 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4437 = stablehlo.reshape %v4425 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4438 = stablehlo.multiply %v4436, %v4437 : tensor<64x64x56x56xf32>
    %v4439 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4440 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4441 = stablehlo.reduce(%v4438 init: %v4439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4442 = stablehlo.divide %v4441, %v4440 : tensor<64xf32>
    %v4443 = stablehlo.multiply %v4435, %v4438 : tensor<64x64x56x56xf32>
    %v4444 = stablehlo.reduce(%v4443 init: %v4439) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4445 = stablehlo.divide %v4444, %v4440 : tensor<64xf32>
    %v4446 = stablehlo.concatenate %v4442, %v4445, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4447 = stablehlo.concatenate %v168, %v4446, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b1g1dst = "stablehlo.all_reduce"(%v4447) ({
    ^bb0(%aras1b1g1dst: tensor<f32>, %arbs1b1g1dst: tensor<f32>):
      %aradds1b1g1dst = stablehlo.add %aras1b1g1dst, %arbs1b1g1dst : tensor<f32>
      stablehlo.return %aradds1b1g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b1g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b1g1dst = stablehlo.divide %arsums1b1g1dst, %arns1b1g1dst : tensor<256xf32>
    %v4448 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4449 = stablehlo.slice %armeans1b1g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4450 = stablehlo.slice %armeans1b1g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4451 = stablehlo.slice %armeans1b1g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4452 = stablehlo.slice %armeans1b1g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4453 = stablehlo.broadcast_in_dim %v4449, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4454 = stablehlo.broadcast_in_dim %v4450, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4455 = stablehlo.broadcast_in_dim %v4451, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4456 = stablehlo.broadcast_in_dim %v4452, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4457 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4458 = stablehlo.add %v4454, %v4457 : tensor<64x64x56x56xf32>
    %v4459 = stablehlo.rsqrt %v4458 : tensor<64x64x56x56xf32>
    %v4460 = stablehlo.subtract %v4448, %v4453 : tensor<64x64x56x56xf32>
    %v4461 = stablehlo.multiply %v4460, %v4459 : tensor<64x64x56x56xf32>
    %v4462 = stablehlo.broadcast_in_dim %s1b1g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4463 = stablehlo.reshape %v4425 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4464 = stablehlo.multiply %v4462, %v4463 : tensor<64x64x56x56xf32>
    %v4465 = stablehlo.subtract %v4464, %v4455 : tensor<64x64x56x56xf32>
    %v4466 = stablehlo.multiply %v4461, %v4456 : tensor<64x64x56x56xf32>
    %v4467 = stablehlo.subtract %v4465, %v4466 : tensor<64x64x56x56xf32>
    %v4468 = stablehlo.multiply %v4459, %v4467 : tensor<64x64x56x56xf32>
    %v4469 = stablehlo.reshape %v4468 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4470 = stablehlo.reshape %v4469 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4471 = stablehlo.reverse %s1b1W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4472 = stablehlo.transpose %v4471, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4473 = stablehlo.convert %v4470 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4474 = stablehlo.convert %v4472 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4475 = stablehlo.convolution(%v4473, %v4474)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4476 = stablehlo.convert %v4475 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4477 = stablehlo.reshape %v4476 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4478 = stablehlo.reshape %v4477 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4479 = stablehlo.reshape %v4367 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4480 = stablehlo.add %v4478, %v4479 : tensor<64x64x56x56xf32>
    %v4481 = stablehlo.reshape %v4480 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4482 = stablehlo.reshape %v141 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4483 = stablehlo.reshape %v4469 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4484 = stablehlo.transpose %v4482, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4485 = stablehlo.transpose %v4483, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4486 = stablehlo.convert %v4484 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4487 = stablehlo.convert %v4485 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4488 = stablehlo.convolution(%v4486, %v4487)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4489 = stablehlo.convert %v4488 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4490 = stablehlo.transpose %v4489, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4491 = stablehlo.reshape %v149 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4492 = stablehlo.slice %v168 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4493 = stablehlo.slice %v168 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4494 = stablehlo.broadcast_in_dim %v4492, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4495 = stablehlo.broadcast_in_dim %v4493, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4496 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4497 = stablehlo.add %v4495, %v4496 : tensor<64x64x56x56xf32>
    %v4498 = stablehlo.rsqrt %v4497 : tensor<64x64x56x56xf32>
    %v4499 = stablehlo.subtract %v4491, %v4494 : tensor<64x64x56x56xf32>
    %v4500 = stablehlo.multiply %v4499, %v4498 : tensor<64x64x56x56xf32>
    %v4501 = stablehlo.reshape %v4425 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4502 = stablehlo.multiply %v4501, %v4500 : tensor<64x64x56x56xf32>
    %v4503 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4504 = stablehlo.reduce(%v4502 init: %v4503) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4505 = stablehlo.reshape %v4425 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4506 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4507 = stablehlo.reduce(%v4505 init: %v4506) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4508 = stablehlo.reshape %v185 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4509 = stablehlo.reshape %v4411 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4510 = stablehlo.transpose %v4508, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4511 = stablehlo.transpose %v4509, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4512 = stablehlo.convert %v4510 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4513 = stablehlo.convert %v4511 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4514 = stablehlo.convolution(%v4512, %v4513)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4515 = stablehlo.convert %v4514 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4516 = stablehlo.transpose %v4515, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4517 = stablehlo.reshape %v193 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4518 = stablehlo.slice %v212 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4519 = stablehlo.slice %v212 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4520 = stablehlo.broadcast_in_dim %v4518, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4521 = stablehlo.broadcast_in_dim %v4519, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4522 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4523 = stablehlo.add %v4521, %v4522 : tensor<64x64x56x56xf32>
    %v4524 = stablehlo.rsqrt %v4523 : tensor<64x64x56x56xf32>
    %v4525 = stablehlo.subtract %v4517, %v4520 : tensor<64x64x56x56xf32>
    %v4526 = stablehlo.multiply %v4525, %v4524 : tensor<64x64x56x56xf32>
    %v4527 = stablehlo.reshape %v4367 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4528 = stablehlo.multiply %v4527, %v4526 : tensor<64x64x56x56xf32>
    %v4529 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4530 = stablehlo.reduce(%v4528 init: %v4529) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4531 = stablehlo.reshape %v4367 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4532 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4533 = stablehlo.reduce(%v4531 init: %v4532) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4534 = stablehlo.reshape %v4481 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4535 = stablehlo.reshape %v137 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4536 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4537 = stablehlo.compare GT, %v4535, %v4536 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4538 = stablehlo.select %v4537, %v4534, %v4536 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4539 = stablehlo.reshape %v4538 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4540 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4541 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4542 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4543 = stablehlo.broadcast_in_dim %v4541, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4544 = stablehlo.broadcast_in_dim %v4542, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4545 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4546 = stablehlo.add %v4544, %v4545 : tensor<64x64x56x56xf32>
    %v4547 = stablehlo.rsqrt %v4546 : tensor<64x64x56x56xf32>
    %v4548 = stablehlo.subtract %v4540, %v4543 : tensor<64x64x56x56xf32>
    %v4549 = stablehlo.multiply %v4548, %v4547 : tensor<64x64x56x56xf32>
    %v4550 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4551 = stablehlo.reshape %v4539 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4552 = stablehlo.multiply %v4550, %v4551 : tensor<64x64x56x56xf32>
    %v4553 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4554 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4555 = stablehlo.reduce(%v4552 init: %v4553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4556 = stablehlo.divide %v4555, %v4554 : tensor<64xf32>
    %v4557 = stablehlo.multiply %v4549, %v4552 : tensor<64x64x56x56xf32>
    %v4558 = stablehlo.reduce(%v4557 init: %v4553) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4559 = stablehlo.divide %v4558, %v4554 : tensor<64xf32>
    %v4560 = stablehlo.concatenate %v4556, %v4559, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4561 = stablehlo.concatenate %v118, %v4560, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g2dst = "stablehlo.all_reduce"(%v4561) ({
    ^bb0(%aras1b0g2dst: tensor<f32>, %arbs1b0g2dst: tensor<f32>):
      %aradds1b0g2dst = stablehlo.add %aras1b0g2dst, %arbs1b0g2dst : tensor<f32>
      stablehlo.return %aradds1b0g2dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g2dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g2dst = stablehlo.divide %arsums1b0g2dst, %arns1b0g2dst : tensor<256xf32>
    %v4562 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4563 = stablehlo.slice %armeans1b0g2dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4564 = stablehlo.slice %armeans1b0g2dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4565 = stablehlo.slice %armeans1b0g2dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4566 = stablehlo.slice %armeans1b0g2dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4567 = stablehlo.broadcast_in_dim %v4563, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4568 = stablehlo.broadcast_in_dim %v4564, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4569 = stablehlo.broadcast_in_dim %v4565, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4570 = stablehlo.broadcast_in_dim %v4566, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4571 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4572 = stablehlo.add %v4568, %v4571 : tensor<64x64x56x56xf32>
    %v4573 = stablehlo.rsqrt %v4572 : tensor<64x64x56x56xf32>
    %v4574 = stablehlo.subtract %v4562, %v4567 : tensor<64x64x56x56xf32>
    %v4575 = stablehlo.multiply %v4574, %v4573 : tensor<64x64x56x56xf32>
    %v4576 = stablehlo.broadcast_in_dim %s1b0g2, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4577 = stablehlo.reshape %v4539 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4578 = stablehlo.multiply %v4576, %v4577 : tensor<64x64x56x56xf32>
    %v4579 = stablehlo.subtract %v4578, %v4569 : tensor<64x64x56x56xf32>
    %v4580 = stablehlo.multiply %v4575, %v4570 : tensor<64x64x56x56xf32>
    %v4581 = stablehlo.subtract %v4579, %v4580 : tensor<64x64x56x56xf32>
    %v4582 = stablehlo.multiply %v4573, %v4581 : tensor<64x64x56x56xf32>
    %v4583 = stablehlo.reshape %v4582 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4584 = stablehlo.reshape %v4583 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4585 = stablehlo.reverse %s1b0W2, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4586 = stablehlo.transpose %v4585, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4587 = stablehlo.convert %v4584 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4588 = stablehlo.convert %v4586 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4589 = stablehlo.convolution(%v4587, %v4588)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4590 = stablehlo.convert %v4589 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4591 = stablehlo.reshape %v4590 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4592 = stablehlo.reshape %v4591 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4593 = stablehlo.reshape %v89 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4594 = stablehlo.constant dense<0.0> : tensor<64x64x56x56xf32>
    %v4595 = stablehlo.compare GT, %v4593, %v4594 : (tensor<64x64x56x56xf32>, tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xi1>
    %v4596 = stablehlo.select %v4595, %v4592, %v4594 : tensor<64x64x56x56xi1>, tensor<64x64x56x56xf32>
    %v4597 = stablehlo.reshape %v4596 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4598 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4599 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4600 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4601 = stablehlo.broadcast_in_dim %v4599, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4602 = stablehlo.broadcast_in_dim %v4600, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4603 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4604 = stablehlo.add %v4602, %v4603 : tensor<64x64x56x56xf32>
    %v4605 = stablehlo.rsqrt %v4604 : tensor<64x64x56x56xf32>
    %v4606 = stablehlo.subtract %v4598, %v4601 : tensor<64x64x56x56xf32>
    %v4607 = stablehlo.multiply %v4606, %v4605 : tensor<64x64x56x56xf32>
    %v4608 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4609 = stablehlo.reshape %v4597 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4610 = stablehlo.multiply %v4608, %v4609 : tensor<64x64x56x56xf32>
    %v4611 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4612 = stablehlo.constant dense<200704.0> : tensor<64xf32>
    %v4613 = stablehlo.reduce(%v4610 init: %v4611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4614 = stablehlo.divide %v4613, %v4612 : tensor<64xf32>
    %v4615 = stablehlo.multiply %v4607, %v4610 : tensor<64x64x56x56xf32>
    %v4616 = stablehlo.reduce(%v4615 init: %v4611) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4617 = stablehlo.divide %v4616, %v4612 : tensor<64xf32>
    %v4618 = stablehlo.concatenate %v4614, %v4617, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4619 = stablehlo.concatenate %v74, %v4618, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsums1b0g1dst = "stablehlo.all_reduce"(%v4619) ({
    ^bb0(%aras1b0g1dst: tensor<f32>, %arbs1b0g1dst: tensor<f32>):
      %aradds1b0g1dst = stablehlo.add %aras1b0g1dst, %arbs1b0g1dst : tensor<f32>
      stablehlo.return %aradds1b0g1dst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns1b0g1dst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans1b0g1dst = stablehlo.divide %arsums1b0g1dst, %arns1b0g1dst : tensor<256xf32>
    %v4620 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4621 = stablehlo.slice %armeans1b0g1dst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4622 = stablehlo.slice %armeans1b0g1dst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4623 = stablehlo.slice %armeans1b0g1dst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4624 = stablehlo.slice %armeans1b0g1dst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4625 = stablehlo.broadcast_in_dim %v4621, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4626 = stablehlo.broadcast_in_dim %v4622, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4627 = stablehlo.broadcast_in_dim %v4623, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4628 = stablehlo.broadcast_in_dim %v4624, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4629 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4630 = stablehlo.add %v4626, %v4629 : tensor<64x64x56x56xf32>
    %v4631 = stablehlo.rsqrt %v4630 : tensor<64x64x56x56xf32>
    %v4632 = stablehlo.subtract %v4620, %v4625 : tensor<64x64x56x56xf32>
    %v4633 = stablehlo.multiply %v4632, %v4631 : tensor<64x64x56x56xf32>
    %v4634 = stablehlo.broadcast_in_dim %s1b0g1, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4635 = stablehlo.reshape %v4597 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4636 = stablehlo.multiply %v4634, %v4635 : tensor<64x64x56x56xf32>
    %v4637 = stablehlo.subtract %v4636, %v4627 : tensor<64x64x56x56xf32>
    %v4638 = stablehlo.multiply %v4633, %v4628 : tensor<64x64x56x56xf32>
    %v4639 = stablehlo.subtract %v4637, %v4638 : tensor<64x64x56x56xf32>
    %v4640 = stablehlo.multiply %v4631, %v4639 : tensor<64x64x56x56xf32>
    %v4641 = stablehlo.reshape %v4640 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4642 = stablehlo.reshape %v4641 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4643 = stablehlo.reverse %s1b0W1, dims = [2, 3] : tensor<64x64x3x3xf32>
    %v4644 = stablehlo.transpose %v4643, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4645 = stablehlo.convert %v4642 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4646 = stablehlo.convert %v4644 : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xbf16>
    %v4647 = stablehlo.convolution(%v4645, %v4646)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x3x3xbf16>) -> tensor<64x64x56x56xbf16>
    %v4648 = stablehlo.convert %v4647 : (tensor<64x64x56x56xbf16>) -> tensor<64x64x56x56xf32>
    %v4649 = stablehlo.reshape %v4648 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4650 = stablehlo.reshape %v4649 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4651 = stablehlo.reshape %v4539 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4652 = stablehlo.add %v4650, %v4651 : tensor<64x64x56x56xf32>
    %v4653 = stablehlo.reshape %v4652 : (tensor<64x64x56x56xf32>) -> tensor<64x200704xf32>
    %v4654 = stablehlo.reshape %v47 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4655 = stablehlo.reshape %v4641 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4656 = stablehlo.transpose %v4654, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4657 = stablehlo.transpose %v4655, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4658 = stablehlo.convert %v4656 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4659 = stablehlo.convert %v4657 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4660 = stablehlo.convolution(%v4658, %v4659)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4661 = stablehlo.convert %v4660 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4662 = stablehlo.transpose %v4661, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4663 = stablehlo.reshape %v55 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4664 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4665 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4666 = stablehlo.broadcast_in_dim %v4664, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4667 = stablehlo.broadcast_in_dim %v4665, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4668 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4669 = stablehlo.add %v4667, %v4668 : tensor<64x64x56x56xf32>
    %v4670 = stablehlo.rsqrt %v4669 : tensor<64x64x56x56xf32>
    %v4671 = stablehlo.subtract %v4663, %v4666 : tensor<64x64x56x56xf32>
    %v4672 = stablehlo.multiply %v4671, %v4670 : tensor<64x64x56x56xf32>
    %v4673 = stablehlo.reshape %v4597 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4674 = stablehlo.multiply %v4673, %v4672 : tensor<64x64x56x56xf32>
    %v4675 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4676 = stablehlo.reduce(%v4674 init: %v4675) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4677 = stablehlo.reshape %v4597 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4678 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4679 = stablehlo.reduce(%v4677 init: %v4678) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4680 = stablehlo.reshape %v91 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4681 = stablehlo.reshape %v4583 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4682 = stablehlo.transpose %v4680, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4683 = stablehlo.transpose %v4681, dims = [1, 0, 2, 3] : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xf32>
    %v4684 = stablehlo.convert %v4682 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4685 = stablehlo.convert %v4683 : (tensor<64x64x56x56xf32>) -> tensor<64x64x56x56xbf16>
    %v4686 = stablehlo.convolution(%v4684, %v4685)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[1, 1], [1, 1]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<64x64x56x56xbf16>, tensor<64x64x56x56xbf16>) -> tensor<64x64x3x3xbf16>
    %v4687 = stablehlo.convert %v4686 : (tensor<64x64x3x3xbf16>) -> tensor<64x64x3x3xf32>
    %v4688 = stablehlo.transpose %v4687, dims = [1, 0, 2, 3] : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %v4689 = stablehlo.reshape %v99 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4690 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4691 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4692 = stablehlo.broadcast_in_dim %v4690, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4693 = stablehlo.broadcast_in_dim %v4691, dims = [1] : (tensor<64xf32>) -> tensor<64x64x56x56xf32>
    %v4694 = stablehlo.constant dense<1.0e-05> : tensor<64x64x56x56xf32>
    %v4695 = stablehlo.add %v4693, %v4694 : tensor<64x64x56x56xf32>
    %v4696 = stablehlo.rsqrt %v4695 : tensor<64x64x56x56xf32>
    %v4697 = stablehlo.subtract %v4689, %v4692 : tensor<64x64x56x56xf32>
    %v4698 = stablehlo.multiply %v4697, %v4696 : tensor<64x64x56x56xf32>
    %v4699 = stablehlo.reshape %v4539 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4700 = stablehlo.multiply %v4699, %v4698 : tensor<64x64x56x56xf32>
    %v4701 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4702 = stablehlo.reduce(%v4700 init: %v4701) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4703 = stablehlo.reshape %v4539 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4704 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4705 = stablehlo.reduce(%v4703 init: %v4704) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64xf32>
    %v4706 = stablehlo.reshape %v43 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4707 = stablehlo.reshape %v4653 : (tensor<64x200704xf32>) -> tensor<64x64x56x56xf32>
    %v4708 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4709 = "stablehlo.select_and_scatter"(%v4706, %v4707, %v4708) ({
      ^bb0(%sa: tensor<f32>, %sb: tensor<f32>):
        %sge = stablehlo.compare GE, %sa, %sb : (tensor<f32>, tensor<f32>) -> tensor<i1>
        stablehlo.return %sge : tensor<i1>
    }, {
      ^bb0(%sc: tensor<f32>, %sd: tensor<f32>):
        %ss = stablehlo.add %sc, %sd : tensor<f32>
        stablehlo.return %ss : tensor<f32>
    }) {window_dimensions = array<i64: 1, 1, 3, 3>, window_strides = array<i64: 1, 1, 2, 2>, padding = dense<[[0, 0], [0, 0], [1, 1], [1, 1]]> : tensor<4x2xi64>} : (tensor<64x64x112x112xf32>, tensor<64x64x56x56xf32>, tensor<f32>) -> tensor<64x64x112x112xf32>
    %v4710 = stablehlo.reshape %v4709 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v4711 = stablehlo.reshape %v4710 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4712 = stablehlo.reshape %v41 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4713 = stablehlo.constant dense<0.0> : tensor<64x64x112x112xf32>
    %v4714 = stablehlo.compare GT, %v4712, %v4713 : (tensor<64x64x112x112xf32>, tensor<64x64x112x112xf32>) -> tensor<64x64x112x112xi1>
    %v4715 = stablehlo.select %v4714, %v4711, %v4713 : tensor<64x64x112x112xi1>, tensor<64x64x112x112xf32>
    %v4716 = stablehlo.reshape %v4715 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v4717 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4718 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4719 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4720 = stablehlo.broadcast_in_dim %v4718, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4721 = stablehlo.broadcast_in_dim %v4719, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4722 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v4723 = stablehlo.add %v4721, %v4722 : tensor<64x64x112x112xf32>
    %v4724 = stablehlo.rsqrt %v4723 : tensor<64x64x112x112xf32>
    %v4725 = stablehlo.subtract %v4717, %v4720 : tensor<64x64x112x112xf32>
    %v4726 = stablehlo.multiply %v4725, %v4724 : tensor<64x64x112x112xf32>
    %v4727 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4728 = stablehlo.reshape %v4716 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4729 = stablehlo.multiply %v4727, %v4728 : tensor<64x64x112x112xf32>
    %v4730 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4731 = stablehlo.constant dense<802816.0> : tensor<64xf32>
    %v4732 = stablehlo.reduce(%v4729 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4733 = stablehlo.divide %v4732, %v4731 : tensor<64xf32>
    %v4734 = stablehlo.multiply %v4726, %v4729 : tensor<64x64x112x112xf32>
    %v4735 = stablehlo.reduce(%v4734 init: %v4730) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4736 = stablehlo.divide %v4735, %v4731 : tensor<64xf32>
    %v4737 = stablehlo.concatenate %v4733, %v4736, dim = 0 : (tensor<64xf32>, tensor<64xf32>) -> tensor<128xf32>
    %v4738 = stablehlo.concatenate %v26, %v4737, dim = 0 : (tensor<128xf32>, tensor<128xf32>) -> tensor<256xf32>
    %arsumsgdst = "stablehlo.all_reduce"(%v4738) ({
    ^bb0(%arasgdst: tensor<f32>, %arbsgdst: tensor<f32>):
      %araddsgdst = stablehlo.add %arasgdst, %arbsgdst : tensor<f32>
      stablehlo.return %araddsgdst : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnsgdst = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeansgdst = stablehlo.divide %arsumsgdst, %arnsgdst : tensor<256xf32>
    %v4739 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4740 = stablehlo.slice %armeansgdst [0:64] : (tensor<256xf32>) -> tensor<64xf32>
    %v4741 = stablehlo.slice %armeansgdst [64:128] : (tensor<256xf32>) -> tensor<64xf32>
    %v4742 = stablehlo.slice %armeansgdst [128:192] : (tensor<256xf32>) -> tensor<64xf32>
    %v4743 = stablehlo.slice %armeansgdst [192:256] : (tensor<256xf32>) -> tensor<64xf32>
    %v4744 = stablehlo.broadcast_in_dim %v4740, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4745 = stablehlo.broadcast_in_dim %v4741, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4746 = stablehlo.broadcast_in_dim %v4742, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4747 = stablehlo.broadcast_in_dim %v4743, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4748 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v4749 = stablehlo.add %v4745, %v4748 : tensor<64x64x112x112xf32>
    %v4750 = stablehlo.rsqrt %v4749 : tensor<64x64x112x112xf32>
    %v4751 = stablehlo.subtract %v4739, %v4744 : tensor<64x64x112x112xf32>
    %v4752 = stablehlo.multiply %v4751, %v4750 : tensor<64x64x112x112xf32>
    %v4753 = stablehlo.broadcast_in_dim %sg, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4754 = stablehlo.reshape %v4716 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4755 = stablehlo.multiply %v4753, %v4754 : tensor<64x64x112x112xf32>
    %v4756 = stablehlo.subtract %v4755, %v4746 : tensor<64x64x112x112xf32>
    %v4757 = stablehlo.multiply %v4752, %v4747 : tensor<64x64x112x112xf32>
    %v4758 = stablehlo.subtract %v4756, %v4757 : tensor<64x64x112x112xf32>
    %v4759 = stablehlo.multiply %v4750, %v4758 : tensor<64x64x112x112xf32>
    %v4760 = stablehlo.reshape %v4759 : (tensor<64x64x112x112xf32>) -> tensor<64x802816xf32>
    %v4761 = stablehlo.reshape %x : (tensor<64x150528xf32>) -> tensor<64x3x224x224xf32>
    %v4762 = stablehlo.reshape %v4760 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4763 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4764 = stablehlo.pad %v4762, %v4763, low = [0, 0, 0, 0], high = [0, 0, 1, 1], interior = [0, 0, 1, 1] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64x64x224x224xf32>
    %v4765 = stablehlo.transpose %v4761, dims = [1, 0, 2, 3] : (tensor<64x3x224x224xf32>) -> tensor<3x64x224x224xf32>
    %v4766 = stablehlo.transpose %v4764, dims = [1, 0, 2, 3] : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xf32>
    %v4767 = stablehlo.convert %v4765 : (tensor<3x64x224x224xf32>) -> tensor<3x64x224x224xbf16>
    %v4768 = stablehlo.convert %v4766 : (tensor<64x64x224x224xf32>) -> tensor<64x64x224x224xbf16>
    %v4769 = stablehlo.convolution(%v4767, %v4768)
      dim_numbers = [b, f, 0, 1]x[o, i, 0, 1]->[b, f, 0, 1],
      window = {stride = [1, 1], pad = [[3, 3], [3, 3]], lhs_dilate = [1, 1], rhs_dilate = [1, 1]}
      {batch_group_count = 1 : i64, feature_group_count = 1 : i64} : (tensor<3x64x224x224xbf16>, tensor<64x64x224x224xbf16>) -> tensor<3x64x7x7xbf16>
    %v4770 = stablehlo.convert %v4769 : (tensor<3x64x7x7xbf16>) -> tensor<3x64x7x7xf32>
    %v4771 = stablehlo.transpose %v4770, dims = [1, 0, 2, 3] : (tensor<3x64x7x7xf32>) -> tensor<64x3x7x7xf32>
    %v4772 = stablehlo.reshape %v7 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4773 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4774 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4775 = stablehlo.broadcast_in_dim %v4773, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4776 = stablehlo.broadcast_in_dim %v4774, dims = [1] : (tensor<64xf32>) -> tensor<64x64x112x112xf32>
    %v4777 = stablehlo.constant dense<1.0e-05> : tensor<64x64x112x112xf32>
    %v4778 = stablehlo.add %v4776, %v4777 : tensor<64x64x112x112xf32>
    %v4779 = stablehlo.rsqrt %v4778 : tensor<64x64x112x112xf32>
    %v4780 = stablehlo.subtract %v4772, %v4775 : tensor<64x64x112x112xf32>
    %v4781 = stablehlo.multiply %v4780, %v4779 : tensor<64x64x112x112xf32>
    %v4782 = stablehlo.reshape %v4716 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4783 = stablehlo.multiply %v4782, %v4781 : tensor<64x64x112x112xf32>
    %v4784 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4785 = stablehlo.reduce(%v4783 init: %v4784) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4786 = stablehlo.reshape %v4716 : (tensor<64x802816xf32>) -> tensor<64x64x112x112xf32>
    %v4787 = stablehlo.constant dense<0.0> : tensor<f32>
    %v4788 = stablehlo.reduce(%v4786 init: %v4787) applies stablehlo.add across dimensions = [0, 2, 3] : (tensor<64x64x112x112xf32>, tensor<f32>) -> tensor<64xf32>
    %v4789 = stablehlo.slice %v26 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4790 = stablehlo.slice %v26 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4791 = stablehlo.slice %v74 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4792 = stablehlo.slice %v74 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4793 = stablehlo.slice %v118 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4794 = stablehlo.slice %v118 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4795 = stablehlo.slice %v168 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4796 = stablehlo.slice %v168 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4797 = stablehlo.slice %v212 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4798 = stablehlo.slice %v212 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4799 = stablehlo.slice %v262 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4800 = stablehlo.slice %v262 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4801 = stablehlo.slice %v306 [0:64] : (tensor<128xf32>) -> tensor<64xf32>
    %v4802 = stablehlo.slice %v306 [64:128] : (tensor<128xf32>) -> tensor<64xf32>
    %v4803 = stablehlo.slice %v356 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4804 = stablehlo.slice %v356 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4805 = stablehlo.slice %v400 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4806 = stablehlo.slice %v400 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4807 = stablehlo.slice %v442 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4808 = stablehlo.slice %v442 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4809 = stablehlo.slice %v487 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4810 = stablehlo.slice %v487 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4811 = stablehlo.slice %v531 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4812 = stablehlo.slice %v531 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4813 = stablehlo.slice %v581 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4814 = stablehlo.slice %v581 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4815 = stablehlo.slice %v625 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4816 = stablehlo.slice %v625 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4817 = stablehlo.slice %v675 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4818 = stablehlo.slice %v675 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4819 = stablehlo.slice %v719 [0:128] : (tensor<256xf32>) -> tensor<128xf32>
    %v4820 = stablehlo.slice %v719 [128:256] : (tensor<256xf32>) -> tensor<128xf32>
    %v4821 = stablehlo.slice %v769 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4822 = stablehlo.slice %v769 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4823 = stablehlo.slice %v813 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4824 = stablehlo.slice %v813 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4825 = stablehlo.slice %v855 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4826 = stablehlo.slice %v855 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4827 = stablehlo.slice %v900 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4828 = stablehlo.slice %v900 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4829 = stablehlo.slice %v944 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4830 = stablehlo.slice %v944 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4831 = stablehlo.slice %v994 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4832 = stablehlo.slice %v994 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4833 = stablehlo.slice %v1038 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4834 = stablehlo.slice %v1038 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4835 = stablehlo.slice %v1088 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4836 = stablehlo.slice %v1088 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4837 = stablehlo.slice %v1132 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4838 = stablehlo.slice %v1132 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4839 = stablehlo.slice %v1182 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4840 = stablehlo.slice %v1182 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4841 = stablehlo.slice %v1226 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4842 = stablehlo.slice %v1226 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4843 = stablehlo.slice %v1276 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4844 = stablehlo.slice %v1276 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4845 = stablehlo.slice %v1320 [0:256] : (tensor<512xf32>) -> tensor<256xf32>
    %v4846 = stablehlo.slice %v1320 [256:512] : (tensor<512xf32>) -> tensor<256xf32>
    %v4847 = stablehlo.slice %v1370 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4848 = stablehlo.slice %v1370 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4849 = stablehlo.slice %v1414 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4850 = stablehlo.slice %v1414 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4851 = stablehlo.slice %v1456 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4852 = stablehlo.slice %v1456 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4853 = stablehlo.slice %v1501 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4854 = stablehlo.slice %v1501 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4855 = stablehlo.slice %v1545 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4856 = stablehlo.slice %v1545 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4857 = stablehlo.slice %v1595 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4858 = stablehlo.slice %v1595 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4859 = stablehlo.slice %v1639 [0:512] : (tensor<1024xf32>) -> tensor<512xf32>
    %v4860 = stablehlo.slice %v1639 [512:1024] : (tensor<1024xf32>) -> tensor<512xf32>
    %mu = stablehlo.constant dense<0.9> : tensor<f32>
    %wd = stablehlo.constant dense<0.0001> : tensor<f32>
    %arsumsW = "stablehlo.all_reduce"(%v4771) ({
    ^bb0(%arasW: tensor<f32>, %arbsW: tensor<f32>):
      %araddsW = stablehlo.add %arasW, %arbsW : tensor<f32>
      stablehlo.return %araddsW : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x3x7x7xf32>) -> tensor<64x3x7x7xf32>
    %arnsW = stablehlo.constant dense<4.0> : tensor<64x3x7x7xf32>
    %armeansW = stablehlo.divide %arsumsW, %arnsW : tensor<64x3x7x7xf32>
    %v4861 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4862 = stablehlo.multiply %v4861, %sW : tensor<64x3x7x7xf32>
    %v4863 = stablehlo.add %v4862, %armeansW : tensor<64x3x7x7xf32>
    %v4864 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4865 = stablehlo.multiply %v4864, %sWv : tensor<64x3x7x7xf32>
    %v4866 = stablehlo.add %v4865, %v4863 : tensor<64x3x7x7xf32>
    %v4867 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x3x7x7xf32>
    %v4868 = stablehlo.multiply %v4867, %v4866 : tensor<64x3x7x7xf32>
    %v4869 = stablehlo.subtract %sW, %v4868 : tensor<64x3x7x7xf32>
    %arsumsg = "stablehlo.all_reduce"(%v4785) ({
    ^bb0(%arasg: tensor<f32>, %arbsg: tensor<f32>):
      %araddsg = stablehlo.add %arasg, %arbsg : tensor<f32>
      stablehlo.return %araddsg : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsg = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansg = stablehlo.divide %arsumsg, %arnsg : tensor<64xf32>
    %v4870 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4871 = stablehlo.multiply %v4870, %sg : tensor<64xf32>
    %v4872 = stablehlo.add %v4871, %armeansg : tensor<64xf32>
    %v4873 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4874 = stablehlo.multiply %v4873, %sgv : tensor<64xf32>
    %v4875 = stablehlo.add %v4874, %v4872 : tensor<64xf32>
    %v4876 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4877 = stablehlo.multiply %v4876, %v4875 : tensor<64xf32>
    %v4878 = stablehlo.subtract %sg, %v4877 : tensor<64xf32>
    %arsumsbt = "stablehlo.all_reduce"(%v4788) ({
    ^bb0(%arasbt: tensor<f32>, %arbsbt: tensor<f32>):
      %araddsbt = stablehlo.add %arasbt, %arbsbt : tensor<f32>
      stablehlo.return %araddsbt : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arnsbt = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeansbt = stablehlo.divide %arsumsbt, %arnsbt : tensor<64xf32>
    %v4879 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4880 = stablehlo.multiply %v4879, %sbt : tensor<64xf32>
    %v4881 = stablehlo.add %v4880, %armeansbt : tensor<64xf32>
    %v4882 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4883 = stablehlo.multiply %v4882, %sbtv : tensor<64xf32>
    %v4884 = stablehlo.add %v4883, %v4881 : tensor<64xf32>
    %v4885 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4886 = stablehlo.multiply %v4885, %v4884 : tensor<64xf32>
    %v4887 = stablehlo.subtract %sbt, %v4886 : tensor<64xf32>
    %arsums1b0W1 = "stablehlo.all_reduce"(%v4662) ({
    ^bb0(%aras1b0W1: tensor<f32>, %arbs1b0W1: tensor<f32>):
      %aradds1b0W1 = stablehlo.add %aras1b0W1, %arbs1b0W1 : tensor<f32>
      stablehlo.return %aradds1b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W1 = stablehlo.divide %arsums1b0W1, %arns1b0W1 : tensor<64x64x3x3xf32>
    %v4888 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4889 = stablehlo.multiply %v4888, %s1b0W1 : tensor<64x64x3x3xf32>
    %v4890 = stablehlo.add %v4889, %armeans1b0W1 : tensor<64x64x3x3xf32>
    %v4891 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4892 = stablehlo.multiply %v4891, %s1b0W1v : tensor<64x64x3x3xf32>
    %v4893 = stablehlo.add %v4892, %v4890 : tensor<64x64x3x3xf32>
    %v4894 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4895 = stablehlo.multiply %v4894, %v4893 : tensor<64x64x3x3xf32>
    %v4896 = stablehlo.subtract %s1b0W1, %v4895 : tensor<64x64x3x3xf32>
    %arsums1b0g1 = "stablehlo.all_reduce"(%v4676) ({
    ^bb0(%aras1b0g1: tensor<f32>, %arbs1b0g1: tensor<f32>):
      %aradds1b0g1 = stablehlo.add %aras1b0g1, %arbs1b0g1 : tensor<f32>
      stablehlo.return %aradds1b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g1 = stablehlo.divide %arsums1b0g1, %arns1b0g1 : tensor<64xf32>
    %v4897 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4898 = stablehlo.multiply %v4897, %s1b0g1 : tensor<64xf32>
    %v4899 = stablehlo.add %v4898, %armeans1b0g1 : tensor<64xf32>
    %v4900 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4901 = stablehlo.multiply %v4900, %s1b0g1v : tensor<64xf32>
    %v4902 = stablehlo.add %v4901, %v4899 : tensor<64xf32>
    %v4903 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4904 = stablehlo.multiply %v4903, %v4902 : tensor<64xf32>
    %v4905 = stablehlo.subtract %s1b0g1, %v4904 : tensor<64xf32>
    %arsums1b0bt1 = "stablehlo.all_reduce"(%v4679) ({
    ^bb0(%aras1b0bt1: tensor<f32>, %arbs1b0bt1: tensor<f32>):
      %aradds1b0bt1 = stablehlo.add %aras1b0bt1, %arbs1b0bt1 : tensor<f32>
      stablehlo.return %aradds1b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt1 = stablehlo.divide %arsums1b0bt1, %arns1b0bt1 : tensor<64xf32>
    %v4906 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4907 = stablehlo.multiply %v4906, %s1b0bt1 : tensor<64xf32>
    %v4908 = stablehlo.add %v4907, %armeans1b0bt1 : tensor<64xf32>
    %v4909 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4910 = stablehlo.multiply %v4909, %s1b0bt1v : tensor<64xf32>
    %v4911 = stablehlo.add %v4910, %v4908 : tensor<64xf32>
    %v4912 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4913 = stablehlo.multiply %v4912, %v4911 : tensor<64xf32>
    %v4914 = stablehlo.subtract %s1b0bt1, %v4913 : tensor<64xf32>
    %arsums1b0W2 = "stablehlo.all_reduce"(%v4688) ({
    ^bb0(%aras1b0W2: tensor<f32>, %arbs1b0W2: tensor<f32>):
      %aradds1b0W2 = stablehlo.add %aras1b0W2, %arbs1b0W2 : tensor<f32>
      stablehlo.return %aradds1b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b0W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b0W2 = stablehlo.divide %arsums1b0W2, %arns1b0W2 : tensor<64x64x3x3xf32>
    %v4915 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4916 = stablehlo.multiply %v4915, %s1b0W2 : tensor<64x64x3x3xf32>
    %v4917 = stablehlo.add %v4916, %armeans1b0W2 : tensor<64x64x3x3xf32>
    %v4918 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4919 = stablehlo.multiply %v4918, %s1b0W2v : tensor<64x64x3x3xf32>
    %v4920 = stablehlo.add %v4919, %v4917 : tensor<64x64x3x3xf32>
    %v4921 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4922 = stablehlo.multiply %v4921, %v4920 : tensor<64x64x3x3xf32>
    %v4923 = stablehlo.subtract %s1b0W2, %v4922 : tensor<64x64x3x3xf32>
    %arsums1b0g2 = "stablehlo.all_reduce"(%v4702) ({
    ^bb0(%aras1b0g2: tensor<f32>, %arbs1b0g2: tensor<f32>):
      %aradds1b0g2 = stablehlo.add %aras1b0g2, %arbs1b0g2 : tensor<f32>
      stablehlo.return %aradds1b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0g2 = stablehlo.divide %arsums1b0g2, %arns1b0g2 : tensor<64xf32>
    %v4924 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4925 = stablehlo.multiply %v4924, %s1b0g2 : tensor<64xf32>
    %v4926 = stablehlo.add %v4925, %armeans1b0g2 : tensor<64xf32>
    %v4927 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4928 = stablehlo.multiply %v4927, %s1b0g2v : tensor<64xf32>
    %v4929 = stablehlo.add %v4928, %v4926 : tensor<64xf32>
    %v4930 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4931 = stablehlo.multiply %v4930, %v4929 : tensor<64xf32>
    %v4932 = stablehlo.subtract %s1b0g2, %v4931 : tensor<64xf32>
    %arsums1b0bt2 = "stablehlo.all_reduce"(%v4705) ({
    ^bb0(%aras1b0bt2: tensor<f32>, %arbs1b0bt2: tensor<f32>):
      %aradds1b0bt2 = stablehlo.add %aras1b0bt2, %arbs1b0bt2 : tensor<f32>
      stablehlo.return %aradds1b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b0bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b0bt2 = stablehlo.divide %arsums1b0bt2, %arns1b0bt2 : tensor<64xf32>
    %v4933 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4934 = stablehlo.multiply %v4933, %s1b0bt2 : tensor<64xf32>
    %v4935 = stablehlo.add %v4934, %armeans1b0bt2 : tensor<64xf32>
    %v4936 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4937 = stablehlo.multiply %v4936, %s1b0bt2v : tensor<64xf32>
    %v4938 = stablehlo.add %v4937, %v4935 : tensor<64xf32>
    %v4939 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4940 = stablehlo.multiply %v4939, %v4938 : tensor<64xf32>
    %v4941 = stablehlo.subtract %s1b0bt2, %v4940 : tensor<64xf32>
    %arsums1b1W1 = "stablehlo.all_reduce"(%v4490) ({
    ^bb0(%aras1b1W1: tensor<f32>, %arbs1b1W1: tensor<f32>):
      %aradds1b1W1 = stablehlo.add %aras1b1W1, %arbs1b1W1 : tensor<f32>
      stablehlo.return %aradds1b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W1 = stablehlo.divide %arsums1b1W1, %arns1b1W1 : tensor<64x64x3x3xf32>
    %v4942 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4943 = stablehlo.multiply %v4942, %s1b1W1 : tensor<64x64x3x3xf32>
    %v4944 = stablehlo.add %v4943, %armeans1b1W1 : tensor<64x64x3x3xf32>
    %v4945 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4946 = stablehlo.multiply %v4945, %s1b1W1v : tensor<64x64x3x3xf32>
    %v4947 = stablehlo.add %v4946, %v4944 : tensor<64x64x3x3xf32>
    %v4948 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4949 = stablehlo.multiply %v4948, %v4947 : tensor<64x64x3x3xf32>
    %v4950 = stablehlo.subtract %s1b1W1, %v4949 : tensor<64x64x3x3xf32>
    %arsums1b1g1 = "stablehlo.all_reduce"(%v4504) ({
    ^bb0(%aras1b1g1: tensor<f32>, %arbs1b1g1: tensor<f32>):
      %aradds1b1g1 = stablehlo.add %aras1b1g1, %arbs1b1g1 : tensor<f32>
      stablehlo.return %aradds1b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g1 = stablehlo.divide %arsums1b1g1, %arns1b1g1 : tensor<64xf32>
    %v4951 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4952 = stablehlo.multiply %v4951, %s1b1g1 : tensor<64xf32>
    %v4953 = stablehlo.add %v4952, %armeans1b1g1 : tensor<64xf32>
    %v4954 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4955 = stablehlo.multiply %v4954, %s1b1g1v : tensor<64xf32>
    %v4956 = stablehlo.add %v4955, %v4953 : tensor<64xf32>
    %v4957 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4958 = stablehlo.multiply %v4957, %v4956 : tensor<64xf32>
    %v4959 = stablehlo.subtract %s1b1g1, %v4958 : tensor<64xf32>
    %arsums1b1bt1 = "stablehlo.all_reduce"(%v4507) ({
    ^bb0(%aras1b1bt1: tensor<f32>, %arbs1b1bt1: tensor<f32>):
      %aradds1b1bt1 = stablehlo.add %aras1b1bt1, %arbs1b1bt1 : tensor<f32>
      stablehlo.return %aradds1b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt1 = stablehlo.divide %arsums1b1bt1, %arns1b1bt1 : tensor<64xf32>
    %v4960 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4961 = stablehlo.multiply %v4960, %s1b1bt1 : tensor<64xf32>
    %v4962 = stablehlo.add %v4961, %armeans1b1bt1 : tensor<64xf32>
    %v4963 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4964 = stablehlo.multiply %v4963, %s1b1bt1v : tensor<64xf32>
    %v4965 = stablehlo.add %v4964, %v4962 : tensor<64xf32>
    %v4966 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4967 = stablehlo.multiply %v4966, %v4965 : tensor<64xf32>
    %v4968 = stablehlo.subtract %s1b1bt1, %v4967 : tensor<64xf32>
    %arsums1b1W2 = "stablehlo.all_reduce"(%v4516) ({
    ^bb0(%aras1b1W2: tensor<f32>, %arbs1b1W2: tensor<f32>):
      %aradds1b1W2 = stablehlo.add %aras1b1W2, %arbs1b1W2 : tensor<f32>
      stablehlo.return %aradds1b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b1W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b1W2 = stablehlo.divide %arsums1b1W2, %arns1b1W2 : tensor<64x64x3x3xf32>
    %v4969 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4970 = stablehlo.multiply %v4969, %s1b1W2 : tensor<64x64x3x3xf32>
    %v4971 = stablehlo.add %v4970, %armeans1b1W2 : tensor<64x64x3x3xf32>
    %v4972 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4973 = stablehlo.multiply %v4972, %s1b1W2v : tensor<64x64x3x3xf32>
    %v4974 = stablehlo.add %v4973, %v4971 : tensor<64x64x3x3xf32>
    %v4975 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4976 = stablehlo.multiply %v4975, %v4974 : tensor<64x64x3x3xf32>
    %v4977 = stablehlo.subtract %s1b1W2, %v4976 : tensor<64x64x3x3xf32>
    %arsums1b1g2 = "stablehlo.all_reduce"(%v4530) ({
    ^bb0(%aras1b1g2: tensor<f32>, %arbs1b1g2: tensor<f32>):
      %aradds1b1g2 = stablehlo.add %aras1b1g2, %arbs1b1g2 : tensor<f32>
      stablehlo.return %aradds1b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1g2 = stablehlo.divide %arsums1b1g2, %arns1b1g2 : tensor<64xf32>
    %v4978 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4979 = stablehlo.multiply %v4978, %s1b1g2 : tensor<64xf32>
    %v4980 = stablehlo.add %v4979, %armeans1b1g2 : tensor<64xf32>
    %v4981 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4982 = stablehlo.multiply %v4981, %s1b1g2v : tensor<64xf32>
    %v4983 = stablehlo.add %v4982, %v4980 : tensor<64xf32>
    %v4984 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4985 = stablehlo.multiply %v4984, %v4983 : tensor<64xf32>
    %v4986 = stablehlo.subtract %s1b1g2, %v4985 : tensor<64xf32>
    %arsums1b1bt2 = "stablehlo.all_reduce"(%v4533) ({
    ^bb0(%aras1b1bt2: tensor<f32>, %arbs1b1bt2: tensor<f32>):
      %aradds1b1bt2 = stablehlo.add %aras1b1bt2, %arbs1b1bt2 : tensor<f32>
      stablehlo.return %aradds1b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b1bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b1bt2 = stablehlo.divide %arsums1b1bt2, %arns1b1bt2 : tensor<64xf32>
    %v4987 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4988 = stablehlo.multiply %v4987, %s1b1bt2 : tensor<64xf32>
    %v4989 = stablehlo.add %v4988, %armeans1b1bt2 : tensor<64xf32>
    %v4990 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4991 = stablehlo.multiply %v4990, %s1b1bt2v : tensor<64xf32>
    %v4992 = stablehlo.add %v4991, %v4989 : tensor<64xf32>
    %v4993 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v4994 = stablehlo.multiply %v4993, %v4992 : tensor<64xf32>
    %v4995 = stablehlo.subtract %s1b1bt2, %v4994 : tensor<64xf32>
    %arsums1b2W1 = "stablehlo.all_reduce"(%v4318) ({
    ^bb0(%aras1b2W1: tensor<f32>, %arbs1b2W1: tensor<f32>):
      %aradds1b2W1 = stablehlo.add %aras1b2W1, %arbs1b2W1 : tensor<f32>
      stablehlo.return %aradds1b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W1 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W1 = stablehlo.divide %arsums1b2W1, %arns1b2W1 : tensor<64x64x3x3xf32>
    %v4996 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v4997 = stablehlo.multiply %v4996, %s1b2W1 : tensor<64x64x3x3xf32>
    %v4998 = stablehlo.add %v4997, %armeans1b2W1 : tensor<64x64x3x3xf32>
    %v4999 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5000 = stablehlo.multiply %v4999, %s1b2W1v : tensor<64x64x3x3xf32>
    %v5001 = stablehlo.add %v5000, %v4998 : tensor<64x64x3x3xf32>
    %v5002 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5003 = stablehlo.multiply %v5002, %v5001 : tensor<64x64x3x3xf32>
    %v5004 = stablehlo.subtract %s1b2W1, %v5003 : tensor<64x64x3x3xf32>
    %arsums1b2g1 = "stablehlo.all_reduce"(%v4332) ({
    ^bb0(%aras1b2g1: tensor<f32>, %arbs1b2g1: tensor<f32>):
      %aradds1b2g1 = stablehlo.add %aras1b2g1, %arbs1b2g1 : tensor<f32>
      stablehlo.return %aradds1b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g1 = stablehlo.divide %arsums1b2g1, %arns1b2g1 : tensor<64xf32>
    %v5005 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5006 = stablehlo.multiply %v5005, %s1b2g1 : tensor<64xf32>
    %v5007 = stablehlo.add %v5006, %armeans1b2g1 : tensor<64xf32>
    %v5008 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5009 = stablehlo.multiply %v5008, %s1b2g1v : tensor<64xf32>
    %v5010 = stablehlo.add %v5009, %v5007 : tensor<64xf32>
    %v5011 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5012 = stablehlo.multiply %v5011, %v5010 : tensor<64xf32>
    %v5013 = stablehlo.subtract %s1b2g1, %v5012 : tensor<64xf32>
    %arsums1b2bt1 = "stablehlo.all_reduce"(%v4335) ({
    ^bb0(%aras1b2bt1: tensor<f32>, %arbs1b2bt1: tensor<f32>):
      %aradds1b2bt1 = stablehlo.add %aras1b2bt1, %arbs1b2bt1 : tensor<f32>
      stablehlo.return %aradds1b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt1 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt1 = stablehlo.divide %arsums1b2bt1, %arns1b2bt1 : tensor<64xf32>
    %v5014 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5015 = stablehlo.multiply %v5014, %s1b2bt1 : tensor<64xf32>
    %v5016 = stablehlo.add %v5015, %armeans1b2bt1 : tensor<64xf32>
    %v5017 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5018 = stablehlo.multiply %v5017, %s1b2bt1v : tensor<64xf32>
    %v5019 = stablehlo.add %v5018, %v5016 : tensor<64xf32>
    %v5020 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5021 = stablehlo.multiply %v5020, %v5019 : tensor<64xf32>
    %v5022 = stablehlo.subtract %s1b2bt1, %v5021 : tensor<64xf32>
    %arsums1b2W2 = "stablehlo.all_reduce"(%v4344) ({
    ^bb0(%aras1b2W2: tensor<f32>, %arbs1b2W2: tensor<f32>):
      %aradds1b2W2 = stablehlo.add %aras1b2W2, %arbs1b2W2 : tensor<f32>
      stablehlo.return %aradds1b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64x64x3x3xf32>) -> tensor<64x64x3x3xf32>
    %arns1b2W2 = stablehlo.constant dense<4.0> : tensor<64x64x3x3xf32>
    %armeans1b2W2 = stablehlo.divide %arsums1b2W2, %arns1b2W2 : tensor<64x64x3x3xf32>
    %v5023 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5024 = stablehlo.multiply %v5023, %s1b2W2 : tensor<64x64x3x3xf32>
    %v5025 = stablehlo.add %v5024, %armeans1b2W2 : tensor<64x64x3x3xf32>
    %v5026 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5027 = stablehlo.multiply %v5026, %s1b2W2v : tensor<64x64x3x3xf32>
    %v5028 = stablehlo.add %v5027, %v5025 : tensor<64x64x3x3xf32>
    %v5029 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64x64x3x3xf32>
    %v5030 = stablehlo.multiply %v5029, %v5028 : tensor<64x64x3x3xf32>
    %v5031 = stablehlo.subtract %s1b2W2, %v5030 : tensor<64x64x3x3xf32>
    %arsums1b2g2 = "stablehlo.all_reduce"(%v4358) ({
    ^bb0(%aras1b2g2: tensor<f32>, %arbs1b2g2: tensor<f32>):
      %aradds1b2g2 = stablehlo.add %aras1b2g2, %arbs1b2g2 : tensor<f32>
      stablehlo.return %aradds1b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2g2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2g2 = stablehlo.divide %arsums1b2g2, %arns1b2g2 : tensor<64xf32>
    %v5032 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5033 = stablehlo.multiply %v5032, %s1b2g2 : tensor<64xf32>
    %v5034 = stablehlo.add %v5033, %armeans1b2g2 : tensor<64xf32>
    %v5035 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5036 = stablehlo.multiply %v5035, %s1b2g2v : tensor<64xf32>
    %v5037 = stablehlo.add %v5036, %v5034 : tensor<64xf32>
    %v5038 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5039 = stablehlo.multiply %v5038, %v5037 : tensor<64xf32>
    %v5040 = stablehlo.subtract %s1b2g2, %v5039 : tensor<64xf32>
    %arsums1b2bt2 = "stablehlo.all_reduce"(%v4361) ({
    ^bb0(%aras1b2bt2: tensor<f32>, %arbs1b2bt2: tensor<f32>):
      %aradds1b2bt2 = stablehlo.add %aras1b2bt2, %arbs1b2bt2 : tensor<f32>
      stablehlo.return %aradds1b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<64xf32>) -> tensor<64xf32>
    %arns1b2bt2 = stablehlo.constant dense<4.0> : tensor<64xf32>
    %armeans1b2bt2 = stablehlo.divide %arsums1b2bt2, %arns1b2bt2 : tensor<64xf32>
    %v5041 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5042 = stablehlo.multiply %v5041, %s1b2bt2 : tensor<64xf32>
    %v5043 = stablehlo.add %v5042, %armeans1b2bt2 : tensor<64xf32>
    %v5044 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5045 = stablehlo.multiply %v5044, %s1b2bt2v : tensor<64xf32>
    %v5046 = stablehlo.add %v5045, %v5043 : tensor<64xf32>
    %v5047 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<64xf32>
    %v5048 = stablehlo.multiply %v5047, %v5046 : tensor<64xf32>
    %v5049 = stablehlo.subtract %s1b2bt2, %v5048 : tensor<64xf32>
    %arsumd2W1 = "stablehlo.all_reduce"(%v4118) ({
    ^bb0(%arad2W1: tensor<f32>, %arbd2W1: tensor<f32>):
      %araddd2W1 = stablehlo.add %arad2W1, %arbd2W1 : tensor<f32>
      stablehlo.return %araddd2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x64x3x3xf32>) -> tensor<128x64x3x3xf32>
    %arnd2W1 = stablehlo.constant dense<4.0> : tensor<128x64x3x3xf32>
    %armeand2W1 = stablehlo.divide %arsumd2W1, %arnd2W1 : tensor<128x64x3x3xf32>
    %v5050 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5051 = stablehlo.multiply %v5050, %d2W1 : tensor<128x64x3x3xf32>
    %v5052 = stablehlo.add %v5051, %armeand2W1 : tensor<128x64x3x3xf32>
    %v5053 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5054 = stablehlo.multiply %v5053, %d2W1v : tensor<128x64x3x3xf32>
    %v5055 = stablehlo.add %v5054, %v5052 : tensor<128x64x3x3xf32>
    %v5056 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x3x3xf32>
    %v5057 = stablehlo.multiply %v5056, %v5055 : tensor<128x64x3x3xf32>
    %v5058 = stablehlo.subtract %d2W1, %v5057 : tensor<128x64x3x3xf32>
    %arsumd2g1 = "stablehlo.all_reduce"(%v4132) ({
    ^bb0(%arad2g1: tensor<f32>, %arbd2g1: tensor<f32>):
      %araddd2g1 = stablehlo.add %arad2g1, %arbd2g1 : tensor<f32>
      stablehlo.return %araddd2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g1 = stablehlo.divide %arsumd2g1, %arnd2g1 : tensor<128xf32>
    %v5059 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5060 = stablehlo.multiply %v5059, %d2g1 : tensor<128xf32>
    %v5061 = stablehlo.add %v5060, %armeand2g1 : tensor<128xf32>
    %v5062 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5063 = stablehlo.multiply %v5062, %d2g1v : tensor<128xf32>
    %v5064 = stablehlo.add %v5063, %v5061 : tensor<128xf32>
    %v5065 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5066 = stablehlo.multiply %v5065, %v5064 : tensor<128xf32>
    %v5067 = stablehlo.subtract %d2g1, %v5066 : tensor<128xf32>
    %arsumd2bt1 = "stablehlo.all_reduce"(%v4135) ({
    ^bb0(%arad2bt1: tensor<f32>, %arbd2bt1: tensor<f32>):
      %araddd2bt1 = stablehlo.add %arad2bt1, %arbd2bt1 : tensor<f32>
      stablehlo.return %araddd2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2bt1 = stablehlo.divide %arsumd2bt1, %arnd2bt1 : tensor<128xf32>
    %v5068 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5069 = stablehlo.multiply %v5068, %d2bt1 : tensor<128xf32>
    %v5070 = stablehlo.add %v5069, %armeand2bt1 : tensor<128xf32>
    %v5071 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5072 = stablehlo.multiply %v5071, %d2bt1v : tensor<128xf32>
    %v5073 = stablehlo.add %v5072, %v5070 : tensor<128xf32>
    %v5074 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5075 = stablehlo.multiply %v5074, %v5073 : tensor<128xf32>
    %v5076 = stablehlo.subtract %d2bt1, %v5075 : tensor<128xf32>
    %arsumd2W2 = "stablehlo.all_reduce"(%v4144) ({
    ^bb0(%arad2W2: tensor<f32>, %arbd2W2: tensor<f32>):
      %araddd2W2 = stablehlo.add %arad2W2, %arbd2W2 : tensor<f32>
      stablehlo.return %araddd2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arnd2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeand2W2 = stablehlo.divide %arsumd2W2, %arnd2W2 : tensor<128x128x3x3xf32>
    %v5077 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5078 = stablehlo.multiply %v5077, %d2W2 : tensor<128x128x3x3xf32>
    %v5079 = stablehlo.add %v5078, %armeand2W2 : tensor<128x128x3x3xf32>
    %v5080 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5081 = stablehlo.multiply %v5080, %d2W2v : tensor<128x128x3x3xf32>
    %v5082 = stablehlo.add %v5081, %v5079 : tensor<128x128x3x3xf32>
    %v5083 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5084 = stablehlo.multiply %v5083, %v5082 : tensor<128x128x3x3xf32>
    %v5085 = stablehlo.subtract %d2W2, %v5084 : tensor<128x128x3x3xf32>
    %arsumd2g2 = "stablehlo.all_reduce"(%v4158) ({
    ^bb0(%arad2g2: tensor<f32>, %arbd2g2: tensor<f32>):
      %araddd2g2 = stablehlo.add %arad2g2, %arbd2g2 : tensor<f32>
      stablehlo.return %araddd2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2g2 = stablehlo.divide %arsumd2g2, %arnd2g2 : tensor<128xf32>
    %v5086 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5087 = stablehlo.multiply %v5086, %d2g2 : tensor<128xf32>
    %v5088 = stablehlo.add %v5087, %armeand2g2 : tensor<128xf32>
    %v5089 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5090 = stablehlo.multiply %v5089, %d2g2v : tensor<128xf32>
    %v5091 = stablehlo.add %v5090, %v5088 : tensor<128xf32>
    %v5092 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5093 = stablehlo.multiply %v5092, %v5091 : tensor<128xf32>
    %v5094 = stablehlo.subtract %d2g2, %v5093 : tensor<128xf32>
    %arsumd2bt2 = "stablehlo.all_reduce"(%v4161) ({
    ^bb0(%arad2bt2: tensor<f32>, %arbd2bt2: tensor<f32>):
      %araddd2bt2 = stablehlo.add %arad2bt2, %arbd2bt2 : tensor<f32>
      stablehlo.return %araddd2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2bt2 = stablehlo.divide %arsumd2bt2, %arnd2bt2 : tensor<128xf32>
    %v5095 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5096 = stablehlo.multiply %v5095, %d2bt2 : tensor<128xf32>
    %v5097 = stablehlo.add %v5096, %armeand2bt2 : tensor<128xf32>
    %v5098 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5099 = stablehlo.multiply %v5098, %d2bt2v : tensor<128xf32>
    %v5100 = stablehlo.add %v5099, %v5097 : tensor<128xf32>
    %v5101 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5102 = stablehlo.multiply %v5101, %v5100 : tensor<128xf32>
    %v5103 = stablehlo.subtract %d2bt2, %v5102 : tensor<128xf32>
    %arsumd2Wp = "stablehlo.all_reduce"(%v4172) ({
    ^bb0(%arad2Wp: tensor<f32>, %arbd2Wp: tensor<f32>):
      %araddd2Wp = stablehlo.add %arad2Wp, %arbd2Wp : tensor<f32>
      stablehlo.return %araddd2Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x64x1x1xf32>) -> tensor<128x64x1x1xf32>
    %arnd2Wp = stablehlo.constant dense<4.0> : tensor<128x64x1x1xf32>
    %armeand2Wp = stablehlo.divide %arsumd2Wp, %arnd2Wp : tensor<128x64x1x1xf32>
    %v5104 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5105 = stablehlo.multiply %v5104, %d2Wp : tensor<128x64x1x1xf32>
    %v5106 = stablehlo.add %v5105, %armeand2Wp : tensor<128x64x1x1xf32>
    %v5107 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5108 = stablehlo.multiply %v5107, %d2Wpv : tensor<128x64x1x1xf32>
    %v5109 = stablehlo.add %v5108, %v5106 : tensor<128x64x1x1xf32>
    %v5110 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x64x1x1xf32>
    %v5111 = stablehlo.multiply %v5110, %v5109 : tensor<128x64x1x1xf32>
    %v5112 = stablehlo.subtract %d2Wp, %v5111 : tensor<128x64x1x1xf32>
    %arsumd2gp = "stablehlo.all_reduce"(%v4186) ({
    ^bb0(%arad2gp: tensor<f32>, %arbd2gp: tensor<f32>):
      %araddd2gp = stablehlo.add %arad2gp, %arbd2gp : tensor<f32>
      stablehlo.return %araddd2gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2gp = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2gp = stablehlo.divide %arsumd2gp, %arnd2gp : tensor<128xf32>
    %v5113 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5114 = stablehlo.multiply %v5113, %d2gp : tensor<128xf32>
    %v5115 = stablehlo.add %v5114, %armeand2gp : tensor<128xf32>
    %v5116 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5117 = stablehlo.multiply %v5116, %d2gpv : tensor<128xf32>
    %v5118 = stablehlo.add %v5117, %v5115 : tensor<128xf32>
    %v5119 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5120 = stablehlo.multiply %v5119, %v5118 : tensor<128xf32>
    %v5121 = stablehlo.subtract %d2gp, %v5120 : tensor<128xf32>
    %arsumd2btp = "stablehlo.all_reduce"(%v4189) ({
    ^bb0(%arad2btp: tensor<f32>, %arbd2btp: tensor<f32>):
      %araddd2btp = stablehlo.add %arad2btp, %arbd2btp : tensor<f32>
      stablehlo.return %araddd2btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arnd2btp = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeand2btp = stablehlo.divide %arsumd2btp, %arnd2btp : tensor<128xf32>
    %v5122 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5123 = stablehlo.multiply %v5122, %d2btp : tensor<128xf32>
    %v5124 = stablehlo.add %v5123, %armeand2btp : tensor<128xf32>
    %v5125 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5126 = stablehlo.multiply %v5125, %d2btpv : tensor<128xf32>
    %v5127 = stablehlo.add %v5126, %v5124 : tensor<128xf32>
    %v5128 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5129 = stablehlo.multiply %v5128, %v5127 : tensor<128xf32>
    %v5130 = stablehlo.subtract %d2btp, %v5129 : tensor<128xf32>
    %arsums2b0W1 = "stablehlo.all_reduce"(%v3888) ({
    ^bb0(%aras2b0W1: tensor<f32>, %arbs2b0W1: tensor<f32>):
      %aradds2b0W1 = stablehlo.add %aras2b0W1, %arbs2b0W1 : tensor<f32>
      stablehlo.return %aradds2b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W1 = stablehlo.divide %arsums2b0W1, %arns2b0W1 : tensor<128x128x3x3xf32>
    %v5131 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5132 = stablehlo.multiply %v5131, %s2b0W1 : tensor<128x128x3x3xf32>
    %v5133 = stablehlo.add %v5132, %armeans2b0W1 : tensor<128x128x3x3xf32>
    %v5134 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5135 = stablehlo.multiply %v5134, %s2b0W1v : tensor<128x128x3x3xf32>
    %v5136 = stablehlo.add %v5135, %v5133 : tensor<128x128x3x3xf32>
    %v5137 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5138 = stablehlo.multiply %v5137, %v5136 : tensor<128x128x3x3xf32>
    %v5139 = stablehlo.subtract %s2b0W1, %v5138 : tensor<128x128x3x3xf32>
    %arsums2b0g1 = "stablehlo.all_reduce"(%v3902) ({
    ^bb0(%aras2b0g1: tensor<f32>, %arbs2b0g1: tensor<f32>):
      %aradds2b0g1 = stablehlo.add %aras2b0g1, %arbs2b0g1 : tensor<f32>
      stablehlo.return %aradds2b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g1 = stablehlo.divide %arsums2b0g1, %arns2b0g1 : tensor<128xf32>
    %v5140 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5141 = stablehlo.multiply %v5140, %s2b0g1 : tensor<128xf32>
    %v5142 = stablehlo.add %v5141, %armeans2b0g1 : tensor<128xf32>
    %v5143 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5144 = stablehlo.multiply %v5143, %s2b0g1v : tensor<128xf32>
    %v5145 = stablehlo.add %v5144, %v5142 : tensor<128xf32>
    %v5146 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5147 = stablehlo.multiply %v5146, %v5145 : tensor<128xf32>
    %v5148 = stablehlo.subtract %s2b0g1, %v5147 : tensor<128xf32>
    %arsums2b0bt1 = "stablehlo.all_reduce"(%v3905) ({
    ^bb0(%aras2b0bt1: tensor<f32>, %arbs2b0bt1: tensor<f32>):
      %aradds2b0bt1 = stablehlo.add %aras2b0bt1, %arbs2b0bt1 : tensor<f32>
      stablehlo.return %aradds2b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt1 = stablehlo.divide %arsums2b0bt1, %arns2b0bt1 : tensor<128xf32>
    %v5149 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5150 = stablehlo.multiply %v5149, %s2b0bt1 : tensor<128xf32>
    %v5151 = stablehlo.add %v5150, %armeans2b0bt1 : tensor<128xf32>
    %v5152 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5153 = stablehlo.multiply %v5152, %s2b0bt1v : tensor<128xf32>
    %v5154 = stablehlo.add %v5153, %v5151 : tensor<128xf32>
    %v5155 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5156 = stablehlo.multiply %v5155, %v5154 : tensor<128xf32>
    %v5157 = stablehlo.subtract %s2b0bt1, %v5156 : tensor<128xf32>
    %arsums2b0W2 = "stablehlo.all_reduce"(%v3914) ({
    ^bb0(%aras2b0W2: tensor<f32>, %arbs2b0W2: tensor<f32>):
      %aradds2b0W2 = stablehlo.add %aras2b0W2, %arbs2b0W2 : tensor<f32>
      stablehlo.return %aradds2b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b0W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b0W2 = stablehlo.divide %arsums2b0W2, %arns2b0W2 : tensor<128x128x3x3xf32>
    %v5158 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5159 = stablehlo.multiply %v5158, %s2b0W2 : tensor<128x128x3x3xf32>
    %v5160 = stablehlo.add %v5159, %armeans2b0W2 : tensor<128x128x3x3xf32>
    %v5161 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5162 = stablehlo.multiply %v5161, %s2b0W2v : tensor<128x128x3x3xf32>
    %v5163 = stablehlo.add %v5162, %v5160 : tensor<128x128x3x3xf32>
    %v5164 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5165 = stablehlo.multiply %v5164, %v5163 : tensor<128x128x3x3xf32>
    %v5166 = stablehlo.subtract %s2b0W2, %v5165 : tensor<128x128x3x3xf32>
    %arsums2b0g2 = "stablehlo.all_reduce"(%v3928) ({
    ^bb0(%aras2b0g2: tensor<f32>, %arbs2b0g2: tensor<f32>):
      %aradds2b0g2 = stablehlo.add %aras2b0g2, %arbs2b0g2 : tensor<f32>
      stablehlo.return %aradds2b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0g2 = stablehlo.divide %arsums2b0g2, %arns2b0g2 : tensor<128xf32>
    %v5167 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5168 = stablehlo.multiply %v5167, %s2b0g2 : tensor<128xf32>
    %v5169 = stablehlo.add %v5168, %armeans2b0g2 : tensor<128xf32>
    %v5170 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5171 = stablehlo.multiply %v5170, %s2b0g2v : tensor<128xf32>
    %v5172 = stablehlo.add %v5171, %v5169 : tensor<128xf32>
    %v5173 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5174 = stablehlo.multiply %v5173, %v5172 : tensor<128xf32>
    %v5175 = stablehlo.subtract %s2b0g2, %v5174 : tensor<128xf32>
    %arsums2b0bt2 = "stablehlo.all_reduce"(%v3931) ({
    ^bb0(%aras2b0bt2: tensor<f32>, %arbs2b0bt2: tensor<f32>):
      %aradds2b0bt2 = stablehlo.add %aras2b0bt2, %arbs2b0bt2 : tensor<f32>
      stablehlo.return %aradds2b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b0bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b0bt2 = stablehlo.divide %arsums2b0bt2, %arns2b0bt2 : tensor<128xf32>
    %v5176 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5177 = stablehlo.multiply %v5176, %s2b0bt2 : tensor<128xf32>
    %v5178 = stablehlo.add %v5177, %armeans2b0bt2 : tensor<128xf32>
    %v5179 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5180 = stablehlo.multiply %v5179, %s2b0bt2v : tensor<128xf32>
    %v5181 = stablehlo.add %v5180, %v5178 : tensor<128xf32>
    %v5182 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5183 = stablehlo.multiply %v5182, %v5181 : tensor<128xf32>
    %v5184 = stablehlo.subtract %s2b0bt2, %v5183 : tensor<128xf32>
    %arsums2b1W1 = "stablehlo.all_reduce"(%v3716) ({
    ^bb0(%aras2b1W1: tensor<f32>, %arbs2b1W1: tensor<f32>):
      %aradds2b1W1 = stablehlo.add %aras2b1W1, %arbs2b1W1 : tensor<f32>
      stablehlo.return %aradds2b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W1 = stablehlo.divide %arsums2b1W1, %arns2b1W1 : tensor<128x128x3x3xf32>
    %v5185 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5186 = stablehlo.multiply %v5185, %s2b1W1 : tensor<128x128x3x3xf32>
    %v5187 = stablehlo.add %v5186, %armeans2b1W1 : tensor<128x128x3x3xf32>
    %v5188 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5189 = stablehlo.multiply %v5188, %s2b1W1v : tensor<128x128x3x3xf32>
    %v5190 = stablehlo.add %v5189, %v5187 : tensor<128x128x3x3xf32>
    %v5191 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5192 = stablehlo.multiply %v5191, %v5190 : tensor<128x128x3x3xf32>
    %v5193 = stablehlo.subtract %s2b1W1, %v5192 : tensor<128x128x3x3xf32>
    %arsums2b1g1 = "stablehlo.all_reduce"(%v3730) ({
    ^bb0(%aras2b1g1: tensor<f32>, %arbs2b1g1: tensor<f32>):
      %aradds2b1g1 = stablehlo.add %aras2b1g1, %arbs2b1g1 : tensor<f32>
      stablehlo.return %aradds2b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g1 = stablehlo.divide %arsums2b1g1, %arns2b1g1 : tensor<128xf32>
    %v5194 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5195 = stablehlo.multiply %v5194, %s2b1g1 : tensor<128xf32>
    %v5196 = stablehlo.add %v5195, %armeans2b1g1 : tensor<128xf32>
    %v5197 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5198 = stablehlo.multiply %v5197, %s2b1g1v : tensor<128xf32>
    %v5199 = stablehlo.add %v5198, %v5196 : tensor<128xf32>
    %v5200 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5201 = stablehlo.multiply %v5200, %v5199 : tensor<128xf32>
    %v5202 = stablehlo.subtract %s2b1g1, %v5201 : tensor<128xf32>
    %arsums2b1bt1 = "stablehlo.all_reduce"(%v3733) ({
    ^bb0(%aras2b1bt1: tensor<f32>, %arbs2b1bt1: tensor<f32>):
      %aradds2b1bt1 = stablehlo.add %aras2b1bt1, %arbs2b1bt1 : tensor<f32>
      stablehlo.return %aradds2b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt1 = stablehlo.divide %arsums2b1bt1, %arns2b1bt1 : tensor<128xf32>
    %v5203 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5204 = stablehlo.multiply %v5203, %s2b1bt1 : tensor<128xf32>
    %v5205 = stablehlo.add %v5204, %armeans2b1bt1 : tensor<128xf32>
    %v5206 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5207 = stablehlo.multiply %v5206, %s2b1bt1v : tensor<128xf32>
    %v5208 = stablehlo.add %v5207, %v5205 : tensor<128xf32>
    %v5209 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5210 = stablehlo.multiply %v5209, %v5208 : tensor<128xf32>
    %v5211 = stablehlo.subtract %s2b1bt1, %v5210 : tensor<128xf32>
    %arsums2b1W2 = "stablehlo.all_reduce"(%v3742) ({
    ^bb0(%aras2b1W2: tensor<f32>, %arbs2b1W2: tensor<f32>):
      %aradds2b1W2 = stablehlo.add %aras2b1W2, %arbs2b1W2 : tensor<f32>
      stablehlo.return %aradds2b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b1W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b1W2 = stablehlo.divide %arsums2b1W2, %arns2b1W2 : tensor<128x128x3x3xf32>
    %v5212 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5213 = stablehlo.multiply %v5212, %s2b1W2 : tensor<128x128x3x3xf32>
    %v5214 = stablehlo.add %v5213, %armeans2b1W2 : tensor<128x128x3x3xf32>
    %v5215 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5216 = stablehlo.multiply %v5215, %s2b1W2v : tensor<128x128x3x3xf32>
    %v5217 = stablehlo.add %v5216, %v5214 : tensor<128x128x3x3xf32>
    %v5218 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5219 = stablehlo.multiply %v5218, %v5217 : tensor<128x128x3x3xf32>
    %v5220 = stablehlo.subtract %s2b1W2, %v5219 : tensor<128x128x3x3xf32>
    %arsums2b1g2 = "stablehlo.all_reduce"(%v3756) ({
    ^bb0(%aras2b1g2: tensor<f32>, %arbs2b1g2: tensor<f32>):
      %aradds2b1g2 = stablehlo.add %aras2b1g2, %arbs2b1g2 : tensor<f32>
      stablehlo.return %aradds2b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1g2 = stablehlo.divide %arsums2b1g2, %arns2b1g2 : tensor<128xf32>
    %v5221 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5222 = stablehlo.multiply %v5221, %s2b1g2 : tensor<128xf32>
    %v5223 = stablehlo.add %v5222, %armeans2b1g2 : tensor<128xf32>
    %v5224 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5225 = stablehlo.multiply %v5224, %s2b1g2v : tensor<128xf32>
    %v5226 = stablehlo.add %v5225, %v5223 : tensor<128xf32>
    %v5227 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5228 = stablehlo.multiply %v5227, %v5226 : tensor<128xf32>
    %v5229 = stablehlo.subtract %s2b1g2, %v5228 : tensor<128xf32>
    %arsums2b1bt2 = "stablehlo.all_reduce"(%v3759) ({
    ^bb0(%aras2b1bt2: tensor<f32>, %arbs2b1bt2: tensor<f32>):
      %aradds2b1bt2 = stablehlo.add %aras2b1bt2, %arbs2b1bt2 : tensor<f32>
      stablehlo.return %aradds2b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b1bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b1bt2 = stablehlo.divide %arsums2b1bt2, %arns2b1bt2 : tensor<128xf32>
    %v5230 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5231 = stablehlo.multiply %v5230, %s2b1bt2 : tensor<128xf32>
    %v5232 = stablehlo.add %v5231, %armeans2b1bt2 : tensor<128xf32>
    %v5233 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5234 = stablehlo.multiply %v5233, %s2b1bt2v : tensor<128xf32>
    %v5235 = stablehlo.add %v5234, %v5232 : tensor<128xf32>
    %v5236 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5237 = stablehlo.multiply %v5236, %v5235 : tensor<128xf32>
    %v5238 = stablehlo.subtract %s2b1bt2, %v5237 : tensor<128xf32>
    %arsums2b2W1 = "stablehlo.all_reduce"(%v3544) ({
    ^bb0(%aras2b2W1: tensor<f32>, %arbs2b2W1: tensor<f32>):
      %aradds2b2W1 = stablehlo.add %aras2b2W1, %arbs2b2W1 : tensor<f32>
      stablehlo.return %aradds2b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W1 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W1 = stablehlo.divide %arsums2b2W1, %arns2b2W1 : tensor<128x128x3x3xf32>
    %v5239 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5240 = stablehlo.multiply %v5239, %s2b2W1 : tensor<128x128x3x3xf32>
    %v5241 = stablehlo.add %v5240, %armeans2b2W1 : tensor<128x128x3x3xf32>
    %v5242 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5243 = stablehlo.multiply %v5242, %s2b2W1v : tensor<128x128x3x3xf32>
    %v5244 = stablehlo.add %v5243, %v5241 : tensor<128x128x3x3xf32>
    %v5245 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5246 = stablehlo.multiply %v5245, %v5244 : tensor<128x128x3x3xf32>
    %v5247 = stablehlo.subtract %s2b2W1, %v5246 : tensor<128x128x3x3xf32>
    %arsums2b2g1 = "stablehlo.all_reduce"(%v3558) ({
    ^bb0(%aras2b2g1: tensor<f32>, %arbs2b2g1: tensor<f32>):
      %aradds2b2g1 = stablehlo.add %aras2b2g1, %arbs2b2g1 : tensor<f32>
      stablehlo.return %aradds2b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g1 = stablehlo.divide %arsums2b2g1, %arns2b2g1 : tensor<128xf32>
    %v5248 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5249 = stablehlo.multiply %v5248, %s2b2g1 : tensor<128xf32>
    %v5250 = stablehlo.add %v5249, %armeans2b2g1 : tensor<128xf32>
    %v5251 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5252 = stablehlo.multiply %v5251, %s2b2g1v : tensor<128xf32>
    %v5253 = stablehlo.add %v5252, %v5250 : tensor<128xf32>
    %v5254 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5255 = stablehlo.multiply %v5254, %v5253 : tensor<128xf32>
    %v5256 = stablehlo.subtract %s2b2g1, %v5255 : tensor<128xf32>
    %arsums2b2bt1 = "stablehlo.all_reduce"(%v3561) ({
    ^bb0(%aras2b2bt1: tensor<f32>, %arbs2b2bt1: tensor<f32>):
      %aradds2b2bt1 = stablehlo.add %aras2b2bt1, %arbs2b2bt1 : tensor<f32>
      stablehlo.return %aradds2b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt1 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt1 = stablehlo.divide %arsums2b2bt1, %arns2b2bt1 : tensor<128xf32>
    %v5257 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5258 = stablehlo.multiply %v5257, %s2b2bt1 : tensor<128xf32>
    %v5259 = stablehlo.add %v5258, %armeans2b2bt1 : tensor<128xf32>
    %v5260 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5261 = stablehlo.multiply %v5260, %s2b2bt1v : tensor<128xf32>
    %v5262 = stablehlo.add %v5261, %v5259 : tensor<128xf32>
    %v5263 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5264 = stablehlo.multiply %v5263, %v5262 : tensor<128xf32>
    %v5265 = stablehlo.subtract %s2b2bt1, %v5264 : tensor<128xf32>
    %arsums2b2W2 = "stablehlo.all_reduce"(%v3570) ({
    ^bb0(%aras2b2W2: tensor<f32>, %arbs2b2W2: tensor<f32>):
      %aradds2b2W2 = stablehlo.add %aras2b2W2, %arbs2b2W2 : tensor<f32>
      stablehlo.return %aradds2b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128x128x3x3xf32>) -> tensor<128x128x3x3xf32>
    %arns2b2W2 = stablehlo.constant dense<4.0> : tensor<128x128x3x3xf32>
    %armeans2b2W2 = stablehlo.divide %arsums2b2W2, %arns2b2W2 : tensor<128x128x3x3xf32>
    %v5266 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5267 = stablehlo.multiply %v5266, %s2b2W2 : tensor<128x128x3x3xf32>
    %v5268 = stablehlo.add %v5267, %armeans2b2W2 : tensor<128x128x3x3xf32>
    %v5269 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5270 = stablehlo.multiply %v5269, %s2b2W2v : tensor<128x128x3x3xf32>
    %v5271 = stablehlo.add %v5270, %v5268 : tensor<128x128x3x3xf32>
    %v5272 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128x128x3x3xf32>
    %v5273 = stablehlo.multiply %v5272, %v5271 : tensor<128x128x3x3xf32>
    %v5274 = stablehlo.subtract %s2b2W2, %v5273 : tensor<128x128x3x3xf32>
    %arsums2b2g2 = "stablehlo.all_reduce"(%v3584) ({
    ^bb0(%aras2b2g2: tensor<f32>, %arbs2b2g2: tensor<f32>):
      %aradds2b2g2 = stablehlo.add %aras2b2g2, %arbs2b2g2 : tensor<f32>
      stablehlo.return %aradds2b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2g2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2g2 = stablehlo.divide %arsums2b2g2, %arns2b2g2 : tensor<128xf32>
    %v5275 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5276 = stablehlo.multiply %v5275, %s2b2g2 : tensor<128xf32>
    %v5277 = stablehlo.add %v5276, %armeans2b2g2 : tensor<128xf32>
    %v5278 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5279 = stablehlo.multiply %v5278, %s2b2g2v : tensor<128xf32>
    %v5280 = stablehlo.add %v5279, %v5277 : tensor<128xf32>
    %v5281 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5282 = stablehlo.multiply %v5281, %v5280 : tensor<128xf32>
    %v5283 = stablehlo.subtract %s2b2g2, %v5282 : tensor<128xf32>
    %arsums2b2bt2 = "stablehlo.all_reduce"(%v3587) ({
    ^bb0(%aras2b2bt2: tensor<f32>, %arbs2b2bt2: tensor<f32>):
      %aradds2b2bt2 = stablehlo.add %aras2b2bt2, %arbs2b2bt2 : tensor<f32>
      stablehlo.return %aradds2b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<128xf32>) -> tensor<128xf32>
    %arns2b2bt2 = stablehlo.constant dense<4.0> : tensor<128xf32>
    %armeans2b2bt2 = stablehlo.divide %arsums2b2bt2, %arns2b2bt2 : tensor<128xf32>
    %v5284 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5285 = stablehlo.multiply %v5284, %s2b2bt2 : tensor<128xf32>
    %v5286 = stablehlo.add %v5285, %armeans2b2bt2 : tensor<128xf32>
    %v5287 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5288 = stablehlo.multiply %v5287, %s2b2bt2v : tensor<128xf32>
    %v5289 = stablehlo.add %v5288, %v5286 : tensor<128xf32>
    %v5290 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<128xf32>
    %v5291 = stablehlo.multiply %v5290, %v5289 : tensor<128xf32>
    %v5292 = stablehlo.subtract %s2b2bt2, %v5291 : tensor<128xf32>
    %arsumd3W1 = "stablehlo.all_reduce"(%v3344) ({
    ^bb0(%arad3W1: tensor<f32>, %arbd3W1: tensor<f32>):
      %araddd3W1 = stablehlo.add %arad3W1, %arbd3W1 : tensor<f32>
      stablehlo.return %araddd3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x128x3x3xf32>) -> tensor<256x128x3x3xf32>
    %arnd3W1 = stablehlo.constant dense<4.0> : tensor<256x128x3x3xf32>
    %armeand3W1 = stablehlo.divide %arsumd3W1, %arnd3W1 : tensor<256x128x3x3xf32>
    %v5293 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5294 = stablehlo.multiply %v5293, %d3W1 : tensor<256x128x3x3xf32>
    %v5295 = stablehlo.add %v5294, %armeand3W1 : tensor<256x128x3x3xf32>
    %v5296 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5297 = stablehlo.multiply %v5296, %d3W1v : tensor<256x128x3x3xf32>
    %v5298 = stablehlo.add %v5297, %v5295 : tensor<256x128x3x3xf32>
    %v5299 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x3x3xf32>
    %v5300 = stablehlo.multiply %v5299, %v5298 : tensor<256x128x3x3xf32>
    %v5301 = stablehlo.subtract %d3W1, %v5300 : tensor<256x128x3x3xf32>
    %arsumd3g1 = "stablehlo.all_reduce"(%v3358) ({
    ^bb0(%arad3g1: tensor<f32>, %arbd3g1: tensor<f32>):
      %araddd3g1 = stablehlo.add %arad3g1, %arbd3g1 : tensor<f32>
      stablehlo.return %araddd3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g1 = stablehlo.divide %arsumd3g1, %arnd3g1 : tensor<256xf32>
    %v5302 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5303 = stablehlo.multiply %v5302, %d3g1 : tensor<256xf32>
    %v5304 = stablehlo.add %v5303, %armeand3g1 : tensor<256xf32>
    %v5305 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5306 = stablehlo.multiply %v5305, %d3g1v : tensor<256xf32>
    %v5307 = stablehlo.add %v5306, %v5304 : tensor<256xf32>
    %v5308 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5309 = stablehlo.multiply %v5308, %v5307 : tensor<256xf32>
    %v5310 = stablehlo.subtract %d3g1, %v5309 : tensor<256xf32>
    %arsumd3bt1 = "stablehlo.all_reduce"(%v3361) ({
    ^bb0(%arad3bt1: tensor<f32>, %arbd3bt1: tensor<f32>):
      %araddd3bt1 = stablehlo.add %arad3bt1, %arbd3bt1 : tensor<f32>
      stablehlo.return %araddd3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3bt1 = stablehlo.divide %arsumd3bt1, %arnd3bt1 : tensor<256xf32>
    %v5311 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5312 = stablehlo.multiply %v5311, %d3bt1 : tensor<256xf32>
    %v5313 = stablehlo.add %v5312, %armeand3bt1 : tensor<256xf32>
    %v5314 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5315 = stablehlo.multiply %v5314, %d3bt1v : tensor<256xf32>
    %v5316 = stablehlo.add %v5315, %v5313 : tensor<256xf32>
    %v5317 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5318 = stablehlo.multiply %v5317, %v5316 : tensor<256xf32>
    %v5319 = stablehlo.subtract %d3bt1, %v5318 : tensor<256xf32>
    %arsumd3W2 = "stablehlo.all_reduce"(%v3370) ({
    ^bb0(%arad3W2: tensor<f32>, %arbd3W2: tensor<f32>):
      %araddd3W2 = stablehlo.add %arad3W2, %arbd3W2 : tensor<f32>
      stablehlo.return %araddd3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arnd3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeand3W2 = stablehlo.divide %arsumd3W2, %arnd3W2 : tensor<256x256x3x3xf32>
    %v5320 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5321 = stablehlo.multiply %v5320, %d3W2 : tensor<256x256x3x3xf32>
    %v5322 = stablehlo.add %v5321, %armeand3W2 : tensor<256x256x3x3xf32>
    %v5323 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5324 = stablehlo.multiply %v5323, %d3W2v : tensor<256x256x3x3xf32>
    %v5325 = stablehlo.add %v5324, %v5322 : tensor<256x256x3x3xf32>
    %v5326 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5327 = stablehlo.multiply %v5326, %v5325 : tensor<256x256x3x3xf32>
    %v5328 = stablehlo.subtract %d3W2, %v5327 : tensor<256x256x3x3xf32>
    %arsumd3g2 = "stablehlo.all_reduce"(%v3384) ({
    ^bb0(%arad3g2: tensor<f32>, %arbd3g2: tensor<f32>):
      %araddd3g2 = stablehlo.add %arad3g2, %arbd3g2 : tensor<f32>
      stablehlo.return %araddd3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3g2 = stablehlo.divide %arsumd3g2, %arnd3g2 : tensor<256xf32>
    %v5329 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5330 = stablehlo.multiply %v5329, %d3g2 : tensor<256xf32>
    %v5331 = stablehlo.add %v5330, %armeand3g2 : tensor<256xf32>
    %v5332 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5333 = stablehlo.multiply %v5332, %d3g2v : tensor<256xf32>
    %v5334 = stablehlo.add %v5333, %v5331 : tensor<256xf32>
    %v5335 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5336 = stablehlo.multiply %v5335, %v5334 : tensor<256xf32>
    %v5337 = stablehlo.subtract %d3g2, %v5336 : tensor<256xf32>
    %arsumd3bt2 = "stablehlo.all_reduce"(%v3387) ({
    ^bb0(%arad3bt2: tensor<f32>, %arbd3bt2: tensor<f32>):
      %araddd3bt2 = stablehlo.add %arad3bt2, %arbd3bt2 : tensor<f32>
      stablehlo.return %araddd3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3bt2 = stablehlo.divide %arsumd3bt2, %arnd3bt2 : tensor<256xf32>
    %v5338 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5339 = stablehlo.multiply %v5338, %d3bt2 : tensor<256xf32>
    %v5340 = stablehlo.add %v5339, %armeand3bt2 : tensor<256xf32>
    %v5341 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5342 = stablehlo.multiply %v5341, %d3bt2v : tensor<256xf32>
    %v5343 = stablehlo.add %v5342, %v5340 : tensor<256xf32>
    %v5344 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5345 = stablehlo.multiply %v5344, %v5343 : tensor<256xf32>
    %v5346 = stablehlo.subtract %d3bt2, %v5345 : tensor<256xf32>
    %arsumd3Wp = "stablehlo.all_reduce"(%v3398) ({
    ^bb0(%arad3Wp: tensor<f32>, %arbd3Wp: tensor<f32>):
      %araddd3Wp = stablehlo.add %arad3Wp, %arbd3Wp : tensor<f32>
      stablehlo.return %araddd3Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x128x1x1xf32>) -> tensor<256x128x1x1xf32>
    %arnd3Wp = stablehlo.constant dense<4.0> : tensor<256x128x1x1xf32>
    %armeand3Wp = stablehlo.divide %arsumd3Wp, %arnd3Wp : tensor<256x128x1x1xf32>
    %v5347 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5348 = stablehlo.multiply %v5347, %d3Wp : tensor<256x128x1x1xf32>
    %v5349 = stablehlo.add %v5348, %armeand3Wp : tensor<256x128x1x1xf32>
    %v5350 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5351 = stablehlo.multiply %v5350, %d3Wpv : tensor<256x128x1x1xf32>
    %v5352 = stablehlo.add %v5351, %v5349 : tensor<256x128x1x1xf32>
    %v5353 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x128x1x1xf32>
    %v5354 = stablehlo.multiply %v5353, %v5352 : tensor<256x128x1x1xf32>
    %v5355 = stablehlo.subtract %d3Wp, %v5354 : tensor<256x128x1x1xf32>
    %arsumd3gp = "stablehlo.all_reduce"(%v3412) ({
    ^bb0(%arad3gp: tensor<f32>, %arbd3gp: tensor<f32>):
      %araddd3gp = stablehlo.add %arad3gp, %arbd3gp : tensor<f32>
      stablehlo.return %araddd3gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3gp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3gp = stablehlo.divide %arsumd3gp, %arnd3gp : tensor<256xf32>
    %v5356 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5357 = stablehlo.multiply %v5356, %d3gp : tensor<256xf32>
    %v5358 = stablehlo.add %v5357, %armeand3gp : tensor<256xf32>
    %v5359 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5360 = stablehlo.multiply %v5359, %d3gpv : tensor<256xf32>
    %v5361 = stablehlo.add %v5360, %v5358 : tensor<256xf32>
    %v5362 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5363 = stablehlo.multiply %v5362, %v5361 : tensor<256xf32>
    %v5364 = stablehlo.subtract %d3gp, %v5363 : tensor<256xf32>
    %arsumd3btp = "stablehlo.all_reduce"(%v3415) ({
    ^bb0(%arad3btp: tensor<f32>, %arbd3btp: tensor<f32>):
      %araddd3btp = stablehlo.add %arad3btp, %arbd3btp : tensor<f32>
      stablehlo.return %araddd3btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arnd3btp = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeand3btp = stablehlo.divide %arsumd3btp, %arnd3btp : tensor<256xf32>
    %v5365 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5366 = stablehlo.multiply %v5365, %d3btp : tensor<256xf32>
    %v5367 = stablehlo.add %v5366, %armeand3btp : tensor<256xf32>
    %v5368 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5369 = stablehlo.multiply %v5368, %d3btpv : tensor<256xf32>
    %v5370 = stablehlo.add %v5369, %v5367 : tensor<256xf32>
    %v5371 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5372 = stablehlo.multiply %v5371, %v5370 : tensor<256xf32>
    %v5373 = stablehlo.subtract %d3btp, %v5372 : tensor<256xf32>
    %arsums3b0W1 = "stablehlo.all_reduce"(%v3114) ({
    ^bb0(%aras3b0W1: tensor<f32>, %arbs3b0W1: tensor<f32>):
      %aradds3b0W1 = stablehlo.add %aras3b0W1, %arbs3b0W1 : tensor<f32>
      stablehlo.return %aradds3b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W1 = stablehlo.divide %arsums3b0W1, %arns3b0W1 : tensor<256x256x3x3xf32>
    %v5374 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5375 = stablehlo.multiply %v5374, %s3b0W1 : tensor<256x256x3x3xf32>
    %v5376 = stablehlo.add %v5375, %armeans3b0W1 : tensor<256x256x3x3xf32>
    %v5377 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5378 = stablehlo.multiply %v5377, %s3b0W1v : tensor<256x256x3x3xf32>
    %v5379 = stablehlo.add %v5378, %v5376 : tensor<256x256x3x3xf32>
    %v5380 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5381 = stablehlo.multiply %v5380, %v5379 : tensor<256x256x3x3xf32>
    %v5382 = stablehlo.subtract %s3b0W1, %v5381 : tensor<256x256x3x3xf32>
    %arsums3b0g1 = "stablehlo.all_reduce"(%v3128) ({
    ^bb0(%aras3b0g1: tensor<f32>, %arbs3b0g1: tensor<f32>):
      %aradds3b0g1 = stablehlo.add %aras3b0g1, %arbs3b0g1 : tensor<f32>
      stablehlo.return %aradds3b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g1 = stablehlo.divide %arsums3b0g1, %arns3b0g1 : tensor<256xf32>
    %v5383 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5384 = stablehlo.multiply %v5383, %s3b0g1 : tensor<256xf32>
    %v5385 = stablehlo.add %v5384, %armeans3b0g1 : tensor<256xf32>
    %v5386 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5387 = stablehlo.multiply %v5386, %s3b0g1v : tensor<256xf32>
    %v5388 = stablehlo.add %v5387, %v5385 : tensor<256xf32>
    %v5389 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5390 = stablehlo.multiply %v5389, %v5388 : tensor<256xf32>
    %v5391 = stablehlo.subtract %s3b0g1, %v5390 : tensor<256xf32>
    %arsums3b0bt1 = "stablehlo.all_reduce"(%v3131) ({
    ^bb0(%aras3b0bt1: tensor<f32>, %arbs3b0bt1: tensor<f32>):
      %aradds3b0bt1 = stablehlo.add %aras3b0bt1, %arbs3b0bt1 : tensor<f32>
      stablehlo.return %aradds3b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt1 = stablehlo.divide %arsums3b0bt1, %arns3b0bt1 : tensor<256xf32>
    %v5392 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5393 = stablehlo.multiply %v5392, %s3b0bt1 : tensor<256xf32>
    %v5394 = stablehlo.add %v5393, %armeans3b0bt1 : tensor<256xf32>
    %v5395 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5396 = stablehlo.multiply %v5395, %s3b0bt1v : tensor<256xf32>
    %v5397 = stablehlo.add %v5396, %v5394 : tensor<256xf32>
    %v5398 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5399 = stablehlo.multiply %v5398, %v5397 : tensor<256xf32>
    %v5400 = stablehlo.subtract %s3b0bt1, %v5399 : tensor<256xf32>
    %arsums3b0W2 = "stablehlo.all_reduce"(%v3140) ({
    ^bb0(%aras3b0W2: tensor<f32>, %arbs3b0W2: tensor<f32>):
      %aradds3b0W2 = stablehlo.add %aras3b0W2, %arbs3b0W2 : tensor<f32>
      stablehlo.return %aradds3b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b0W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b0W2 = stablehlo.divide %arsums3b0W2, %arns3b0W2 : tensor<256x256x3x3xf32>
    %v5401 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5402 = stablehlo.multiply %v5401, %s3b0W2 : tensor<256x256x3x3xf32>
    %v5403 = stablehlo.add %v5402, %armeans3b0W2 : tensor<256x256x3x3xf32>
    %v5404 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5405 = stablehlo.multiply %v5404, %s3b0W2v : tensor<256x256x3x3xf32>
    %v5406 = stablehlo.add %v5405, %v5403 : tensor<256x256x3x3xf32>
    %v5407 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5408 = stablehlo.multiply %v5407, %v5406 : tensor<256x256x3x3xf32>
    %v5409 = stablehlo.subtract %s3b0W2, %v5408 : tensor<256x256x3x3xf32>
    %arsums3b0g2 = "stablehlo.all_reduce"(%v3154) ({
    ^bb0(%aras3b0g2: tensor<f32>, %arbs3b0g2: tensor<f32>):
      %aradds3b0g2 = stablehlo.add %aras3b0g2, %arbs3b0g2 : tensor<f32>
      stablehlo.return %aradds3b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0g2 = stablehlo.divide %arsums3b0g2, %arns3b0g2 : tensor<256xf32>
    %v5410 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5411 = stablehlo.multiply %v5410, %s3b0g2 : tensor<256xf32>
    %v5412 = stablehlo.add %v5411, %armeans3b0g2 : tensor<256xf32>
    %v5413 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5414 = stablehlo.multiply %v5413, %s3b0g2v : tensor<256xf32>
    %v5415 = stablehlo.add %v5414, %v5412 : tensor<256xf32>
    %v5416 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5417 = stablehlo.multiply %v5416, %v5415 : tensor<256xf32>
    %v5418 = stablehlo.subtract %s3b0g2, %v5417 : tensor<256xf32>
    %arsums3b0bt2 = "stablehlo.all_reduce"(%v3157) ({
    ^bb0(%aras3b0bt2: tensor<f32>, %arbs3b0bt2: tensor<f32>):
      %aradds3b0bt2 = stablehlo.add %aras3b0bt2, %arbs3b0bt2 : tensor<f32>
      stablehlo.return %aradds3b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b0bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b0bt2 = stablehlo.divide %arsums3b0bt2, %arns3b0bt2 : tensor<256xf32>
    %v5419 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5420 = stablehlo.multiply %v5419, %s3b0bt2 : tensor<256xf32>
    %v5421 = stablehlo.add %v5420, %armeans3b0bt2 : tensor<256xf32>
    %v5422 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5423 = stablehlo.multiply %v5422, %s3b0bt2v : tensor<256xf32>
    %v5424 = stablehlo.add %v5423, %v5421 : tensor<256xf32>
    %v5425 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5426 = stablehlo.multiply %v5425, %v5424 : tensor<256xf32>
    %v5427 = stablehlo.subtract %s3b0bt2, %v5426 : tensor<256xf32>
    %arsums3b1W1 = "stablehlo.all_reduce"(%v2942) ({
    ^bb0(%aras3b1W1: tensor<f32>, %arbs3b1W1: tensor<f32>):
      %aradds3b1W1 = stablehlo.add %aras3b1W1, %arbs3b1W1 : tensor<f32>
      stablehlo.return %aradds3b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W1 = stablehlo.divide %arsums3b1W1, %arns3b1W1 : tensor<256x256x3x3xf32>
    %v5428 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5429 = stablehlo.multiply %v5428, %s3b1W1 : tensor<256x256x3x3xf32>
    %v5430 = stablehlo.add %v5429, %armeans3b1W1 : tensor<256x256x3x3xf32>
    %v5431 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5432 = stablehlo.multiply %v5431, %s3b1W1v : tensor<256x256x3x3xf32>
    %v5433 = stablehlo.add %v5432, %v5430 : tensor<256x256x3x3xf32>
    %v5434 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5435 = stablehlo.multiply %v5434, %v5433 : tensor<256x256x3x3xf32>
    %v5436 = stablehlo.subtract %s3b1W1, %v5435 : tensor<256x256x3x3xf32>
    %arsums3b1g1 = "stablehlo.all_reduce"(%v2956) ({
    ^bb0(%aras3b1g1: tensor<f32>, %arbs3b1g1: tensor<f32>):
      %aradds3b1g1 = stablehlo.add %aras3b1g1, %arbs3b1g1 : tensor<f32>
      stablehlo.return %aradds3b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g1 = stablehlo.divide %arsums3b1g1, %arns3b1g1 : tensor<256xf32>
    %v5437 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5438 = stablehlo.multiply %v5437, %s3b1g1 : tensor<256xf32>
    %v5439 = stablehlo.add %v5438, %armeans3b1g1 : tensor<256xf32>
    %v5440 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5441 = stablehlo.multiply %v5440, %s3b1g1v : tensor<256xf32>
    %v5442 = stablehlo.add %v5441, %v5439 : tensor<256xf32>
    %v5443 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5444 = stablehlo.multiply %v5443, %v5442 : tensor<256xf32>
    %v5445 = stablehlo.subtract %s3b1g1, %v5444 : tensor<256xf32>
    %arsums3b1bt1 = "stablehlo.all_reduce"(%v2959) ({
    ^bb0(%aras3b1bt1: tensor<f32>, %arbs3b1bt1: tensor<f32>):
      %aradds3b1bt1 = stablehlo.add %aras3b1bt1, %arbs3b1bt1 : tensor<f32>
      stablehlo.return %aradds3b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt1 = stablehlo.divide %arsums3b1bt1, %arns3b1bt1 : tensor<256xf32>
    %v5446 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5447 = stablehlo.multiply %v5446, %s3b1bt1 : tensor<256xf32>
    %v5448 = stablehlo.add %v5447, %armeans3b1bt1 : tensor<256xf32>
    %v5449 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5450 = stablehlo.multiply %v5449, %s3b1bt1v : tensor<256xf32>
    %v5451 = stablehlo.add %v5450, %v5448 : tensor<256xf32>
    %v5452 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5453 = stablehlo.multiply %v5452, %v5451 : tensor<256xf32>
    %v5454 = stablehlo.subtract %s3b1bt1, %v5453 : tensor<256xf32>
    %arsums3b1W2 = "stablehlo.all_reduce"(%v2968) ({
    ^bb0(%aras3b1W2: tensor<f32>, %arbs3b1W2: tensor<f32>):
      %aradds3b1W2 = stablehlo.add %aras3b1W2, %arbs3b1W2 : tensor<f32>
      stablehlo.return %aradds3b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b1W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b1W2 = stablehlo.divide %arsums3b1W2, %arns3b1W2 : tensor<256x256x3x3xf32>
    %v5455 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5456 = stablehlo.multiply %v5455, %s3b1W2 : tensor<256x256x3x3xf32>
    %v5457 = stablehlo.add %v5456, %armeans3b1W2 : tensor<256x256x3x3xf32>
    %v5458 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5459 = stablehlo.multiply %v5458, %s3b1W2v : tensor<256x256x3x3xf32>
    %v5460 = stablehlo.add %v5459, %v5457 : tensor<256x256x3x3xf32>
    %v5461 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5462 = stablehlo.multiply %v5461, %v5460 : tensor<256x256x3x3xf32>
    %v5463 = stablehlo.subtract %s3b1W2, %v5462 : tensor<256x256x3x3xf32>
    %arsums3b1g2 = "stablehlo.all_reduce"(%v2982) ({
    ^bb0(%aras3b1g2: tensor<f32>, %arbs3b1g2: tensor<f32>):
      %aradds3b1g2 = stablehlo.add %aras3b1g2, %arbs3b1g2 : tensor<f32>
      stablehlo.return %aradds3b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1g2 = stablehlo.divide %arsums3b1g2, %arns3b1g2 : tensor<256xf32>
    %v5464 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5465 = stablehlo.multiply %v5464, %s3b1g2 : tensor<256xf32>
    %v5466 = stablehlo.add %v5465, %armeans3b1g2 : tensor<256xf32>
    %v5467 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5468 = stablehlo.multiply %v5467, %s3b1g2v : tensor<256xf32>
    %v5469 = stablehlo.add %v5468, %v5466 : tensor<256xf32>
    %v5470 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5471 = stablehlo.multiply %v5470, %v5469 : tensor<256xf32>
    %v5472 = stablehlo.subtract %s3b1g2, %v5471 : tensor<256xf32>
    %arsums3b1bt2 = "stablehlo.all_reduce"(%v2985) ({
    ^bb0(%aras3b1bt2: tensor<f32>, %arbs3b1bt2: tensor<f32>):
      %aradds3b1bt2 = stablehlo.add %aras3b1bt2, %arbs3b1bt2 : tensor<f32>
      stablehlo.return %aradds3b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b1bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b1bt2 = stablehlo.divide %arsums3b1bt2, %arns3b1bt2 : tensor<256xf32>
    %v5473 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5474 = stablehlo.multiply %v5473, %s3b1bt2 : tensor<256xf32>
    %v5475 = stablehlo.add %v5474, %armeans3b1bt2 : tensor<256xf32>
    %v5476 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5477 = stablehlo.multiply %v5476, %s3b1bt2v : tensor<256xf32>
    %v5478 = stablehlo.add %v5477, %v5475 : tensor<256xf32>
    %v5479 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5480 = stablehlo.multiply %v5479, %v5478 : tensor<256xf32>
    %v5481 = stablehlo.subtract %s3b1bt2, %v5480 : tensor<256xf32>
    %arsums3b2W1 = "stablehlo.all_reduce"(%v2770) ({
    ^bb0(%aras3b2W1: tensor<f32>, %arbs3b2W1: tensor<f32>):
      %aradds3b2W1 = stablehlo.add %aras3b2W1, %arbs3b2W1 : tensor<f32>
      stablehlo.return %aradds3b2W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W1 = stablehlo.divide %arsums3b2W1, %arns3b2W1 : tensor<256x256x3x3xf32>
    %v5482 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5483 = stablehlo.multiply %v5482, %s3b2W1 : tensor<256x256x3x3xf32>
    %v5484 = stablehlo.add %v5483, %armeans3b2W1 : tensor<256x256x3x3xf32>
    %v5485 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5486 = stablehlo.multiply %v5485, %s3b2W1v : tensor<256x256x3x3xf32>
    %v5487 = stablehlo.add %v5486, %v5484 : tensor<256x256x3x3xf32>
    %v5488 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5489 = stablehlo.multiply %v5488, %v5487 : tensor<256x256x3x3xf32>
    %v5490 = stablehlo.subtract %s3b2W1, %v5489 : tensor<256x256x3x3xf32>
    %arsums3b2g1 = "stablehlo.all_reduce"(%v2784) ({
    ^bb0(%aras3b2g1: tensor<f32>, %arbs3b2g1: tensor<f32>):
      %aradds3b2g1 = stablehlo.add %aras3b2g1, %arbs3b2g1 : tensor<f32>
      stablehlo.return %aradds3b2g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g1 = stablehlo.divide %arsums3b2g1, %arns3b2g1 : tensor<256xf32>
    %v5491 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5492 = stablehlo.multiply %v5491, %s3b2g1 : tensor<256xf32>
    %v5493 = stablehlo.add %v5492, %armeans3b2g1 : tensor<256xf32>
    %v5494 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5495 = stablehlo.multiply %v5494, %s3b2g1v : tensor<256xf32>
    %v5496 = stablehlo.add %v5495, %v5493 : tensor<256xf32>
    %v5497 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5498 = stablehlo.multiply %v5497, %v5496 : tensor<256xf32>
    %v5499 = stablehlo.subtract %s3b2g1, %v5498 : tensor<256xf32>
    %arsums3b2bt1 = "stablehlo.all_reduce"(%v2787) ({
    ^bb0(%aras3b2bt1: tensor<f32>, %arbs3b2bt1: tensor<f32>):
      %aradds3b2bt1 = stablehlo.add %aras3b2bt1, %arbs3b2bt1 : tensor<f32>
      stablehlo.return %aradds3b2bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt1 = stablehlo.divide %arsums3b2bt1, %arns3b2bt1 : tensor<256xf32>
    %v5500 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5501 = stablehlo.multiply %v5500, %s3b2bt1 : tensor<256xf32>
    %v5502 = stablehlo.add %v5501, %armeans3b2bt1 : tensor<256xf32>
    %v5503 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5504 = stablehlo.multiply %v5503, %s3b2bt1v : tensor<256xf32>
    %v5505 = stablehlo.add %v5504, %v5502 : tensor<256xf32>
    %v5506 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5507 = stablehlo.multiply %v5506, %v5505 : tensor<256xf32>
    %v5508 = stablehlo.subtract %s3b2bt1, %v5507 : tensor<256xf32>
    %arsums3b2W2 = "stablehlo.all_reduce"(%v2796) ({
    ^bb0(%aras3b2W2: tensor<f32>, %arbs3b2W2: tensor<f32>):
      %aradds3b2W2 = stablehlo.add %aras3b2W2, %arbs3b2W2 : tensor<f32>
      stablehlo.return %aradds3b2W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b2W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b2W2 = stablehlo.divide %arsums3b2W2, %arns3b2W2 : tensor<256x256x3x3xf32>
    %v5509 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5510 = stablehlo.multiply %v5509, %s3b2W2 : tensor<256x256x3x3xf32>
    %v5511 = stablehlo.add %v5510, %armeans3b2W2 : tensor<256x256x3x3xf32>
    %v5512 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5513 = stablehlo.multiply %v5512, %s3b2W2v : tensor<256x256x3x3xf32>
    %v5514 = stablehlo.add %v5513, %v5511 : tensor<256x256x3x3xf32>
    %v5515 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5516 = stablehlo.multiply %v5515, %v5514 : tensor<256x256x3x3xf32>
    %v5517 = stablehlo.subtract %s3b2W2, %v5516 : tensor<256x256x3x3xf32>
    %arsums3b2g2 = "stablehlo.all_reduce"(%v2810) ({
    ^bb0(%aras3b2g2: tensor<f32>, %arbs3b2g2: tensor<f32>):
      %aradds3b2g2 = stablehlo.add %aras3b2g2, %arbs3b2g2 : tensor<f32>
      stablehlo.return %aradds3b2g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2g2 = stablehlo.divide %arsums3b2g2, %arns3b2g2 : tensor<256xf32>
    %v5518 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5519 = stablehlo.multiply %v5518, %s3b2g2 : tensor<256xf32>
    %v5520 = stablehlo.add %v5519, %armeans3b2g2 : tensor<256xf32>
    %v5521 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5522 = stablehlo.multiply %v5521, %s3b2g2v : tensor<256xf32>
    %v5523 = stablehlo.add %v5522, %v5520 : tensor<256xf32>
    %v5524 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5525 = stablehlo.multiply %v5524, %v5523 : tensor<256xf32>
    %v5526 = stablehlo.subtract %s3b2g2, %v5525 : tensor<256xf32>
    %arsums3b2bt2 = "stablehlo.all_reduce"(%v2813) ({
    ^bb0(%aras3b2bt2: tensor<f32>, %arbs3b2bt2: tensor<f32>):
      %aradds3b2bt2 = stablehlo.add %aras3b2bt2, %arbs3b2bt2 : tensor<f32>
      stablehlo.return %aradds3b2bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b2bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b2bt2 = stablehlo.divide %arsums3b2bt2, %arns3b2bt2 : tensor<256xf32>
    %v5527 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5528 = stablehlo.multiply %v5527, %s3b2bt2 : tensor<256xf32>
    %v5529 = stablehlo.add %v5528, %armeans3b2bt2 : tensor<256xf32>
    %v5530 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5531 = stablehlo.multiply %v5530, %s3b2bt2v : tensor<256xf32>
    %v5532 = stablehlo.add %v5531, %v5529 : tensor<256xf32>
    %v5533 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5534 = stablehlo.multiply %v5533, %v5532 : tensor<256xf32>
    %v5535 = stablehlo.subtract %s3b2bt2, %v5534 : tensor<256xf32>
    %arsums3b3W1 = "stablehlo.all_reduce"(%v2598) ({
    ^bb0(%aras3b3W1: tensor<f32>, %arbs3b3W1: tensor<f32>):
      %aradds3b3W1 = stablehlo.add %aras3b3W1, %arbs3b3W1 : tensor<f32>
      stablehlo.return %aradds3b3W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W1 = stablehlo.divide %arsums3b3W1, %arns3b3W1 : tensor<256x256x3x3xf32>
    %v5536 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5537 = stablehlo.multiply %v5536, %s3b3W1 : tensor<256x256x3x3xf32>
    %v5538 = stablehlo.add %v5537, %armeans3b3W1 : tensor<256x256x3x3xf32>
    %v5539 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5540 = stablehlo.multiply %v5539, %s3b3W1v : tensor<256x256x3x3xf32>
    %v5541 = stablehlo.add %v5540, %v5538 : tensor<256x256x3x3xf32>
    %v5542 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5543 = stablehlo.multiply %v5542, %v5541 : tensor<256x256x3x3xf32>
    %v5544 = stablehlo.subtract %s3b3W1, %v5543 : tensor<256x256x3x3xf32>
    %arsums3b3g1 = "stablehlo.all_reduce"(%v2612) ({
    ^bb0(%aras3b3g1: tensor<f32>, %arbs3b3g1: tensor<f32>):
      %aradds3b3g1 = stablehlo.add %aras3b3g1, %arbs3b3g1 : tensor<f32>
      stablehlo.return %aradds3b3g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g1 = stablehlo.divide %arsums3b3g1, %arns3b3g1 : tensor<256xf32>
    %v5545 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5546 = stablehlo.multiply %v5545, %s3b3g1 : tensor<256xf32>
    %v5547 = stablehlo.add %v5546, %armeans3b3g1 : tensor<256xf32>
    %v5548 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5549 = stablehlo.multiply %v5548, %s3b3g1v : tensor<256xf32>
    %v5550 = stablehlo.add %v5549, %v5547 : tensor<256xf32>
    %v5551 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5552 = stablehlo.multiply %v5551, %v5550 : tensor<256xf32>
    %v5553 = stablehlo.subtract %s3b3g1, %v5552 : tensor<256xf32>
    %arsums3b3bt1 = "stablehlo.all_reduce"(%v2615) ({
    ^bb0(%aras3b3bt1: tensor<f32>, %arbs3b3bt1: tensor<f32>):
      %aradds3b3bt1 = stablehlo.add %aras3b3bt1, %arbs3b3bt1 : tensor<f32>
      stablehlo.return %aradds3b3bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt1 = stablehlo.divide %arsums3b3bt1, %arns3b3bt1 : tensor<256xf32>
    %v5554 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5555 = stablehlo.multiply %v5554, %s3b3bt1 : tensor<256xf32>
    %v5556 = stablehlo.add %v5555, %armeans3b3bt1 : tensor<256xf32>
    %v5557 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5558 = stablehlo.multiply %v5557, %s3b3bt1v : tensor<256xf32>
    %v5559 = stablehlo.add %v5558, %v5556 : tensor<256xf32>
    %v5560 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5561 = stablehlo.multiply %v5560, %v5559 : tensor<256xf32>
    %v5562 = stablehlo.subtract %s3b3bt1, %v5561 : tensor<256xf32>
    %arsums3b3W2 = "stablehlo.all_reduce"(%v2624) ({
    ^bb0(%aras3b3W2: tensor<f32>, %arbs3b3W2: tensor<f32>):
      %aradds3b3W2 = stablehlo.add %aras3b3W2, %arbs3b3W2 : tensor<f32>
      stablehlo.return %aradds3b3W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b3W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b3W2 = stablehlo.divide %arsums3b3W2, %arns3b3W2 : tensor<256x256x3x3xf32>
    %v5563 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5564 = stablehlo.multiply %v5563, %s3b3W2 : tensor<256x256x3x3xf32>
    %v5565 = stablehlo.add %v5564, %armeans3b3W2 : tensor<256x256x3x3xf32>
    %v5566 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5567 = stablehlo.multiply %v5566, %s3b3W2v : tensor<256x256x3x3xf32>
    %v5568 = stablehlo.add %v5567, %v5565 : tensor<256x256x3x3xf32>
    %v5569 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5570 = stablehlo.multiply %v5569, %v5568 : tensor<256x256x3x3xf32>
    %v5571 = stablehlo.subtract %s3b3W2, %v5570 : tensor<256x256x3x3xf32>
    %arsums3b3g2 = "stablehlo.all_reduce"(%v2638) ({
    ^bb0(%aras3b3g2: tensor<f32>, %arbs3b3g2: tensor<f32>):
      %aradds3b3g2 = stablehlo.add %aras3b3g2, %arbs3b3g2 : tensor<f32>
      stablehlo.return %aradds3b3g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3g2 = stablehlo.divide %arsums3b3g2, %arns3b3g2 : tensor<256xf32>
    %v5572 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5573 = stablehlo.multiply %v5572, %s3b3g2 : tensor<256xf32>
    %v5574 = stablehlo.add %v5573, %armeans3b3g2 : tensor<256xf32>
    %v5575 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5576 = stablehlo.multiply %v5575, %s3b3g2v : tensor<256xf32>
    %v5577 = stablehlo.add %v5576, %v5574 : tensor<256xf32>
    %v5578 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5579 = stablehlo.multiply %v5578, %v5577 : tensor<256xf32>
    %v5580 = stablehlo.subtract %s3b3g2, %v5579 : tensor<256xf32>
    %arsums3b3bt2 = "stablehlo.all_reduce"(%v2641) ({
    ^bb0(%aras3b3bt2: tensor<f32>, %arbs3b3bt2: tensor<f32>):
      %aradds3b3bt2 = stablehlo.add %aras3b3bt2, %arbs3b3bt2 : tensor<f32>
      stablehlo.return %aradds3b3bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b3bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b3bt2 = stablehlo.divide %arsums3b3bt2, %arns3b3bt2 : tensor<256xf32>
    %v5581 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5582 = stablehlo.multiply %v5581, %s3b3bt2 : tensor<256xf32>
    %v5583 = stablehlo.add %v5582, %armeans3b3bt2 : tensor<256xf32>
    %v5584 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5585 = stablehlo.multiply %v5584, %s3b3bt2v : tensor<256xf32>
    %v5586 = stablehlo.add %v5585, %v5583 : tensor<256xf32>
    %v5587 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5588 = stablehlo.multiply %v5587, %v5586 : tensor<256xf32>
    %v5589 = stablehlo.subtract %s3b3bt2, %v5588 : tensor<256xf32>
    %arsums3b4W1 = "stablehlo.all_reduce"(%v2426) ({
    ^bb0(%aras3b4W1: tensor<f32>, %arbs3b4W1: tensor<f32>):
      %aradds3b4W1 = stablehlo.add %aras3b4W1, %arbs3b4W1 : tensor<f32>
      stablehlo.return %aradds3b4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W1 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W1 = stablehlo.divide %arsums3b4W1, %arns3b4W1 : tensor<256x256x3x3xf32>
    %v5590 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5591 = stablehlo.multiply %v5590, %s3b4W1 : tensor<256x256x3x3xf32>
    %v5592 = stablehlo.add %v5591, %armeans3b4W1 : tensor<256x256x3x3xf32>
    %v5593 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5594 = stablehlo.multiply %v5593, %s3b4W1v : tensor<256x256x3x3xf32>
    %v5595 = stablehlo.add %v5594, %v5592 : tensor<256x256x3x3xf32>
    %v5596 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5597 = stablehlo.multiply %v5596, %v5595 : tensor<256x256x3x3xf32>
    %v5598 = stablehlo.subtract %s3b4W1, %v5597 : tensor<256x256x3x3xf32>
    %arsums3b4g1 = "stablehlo.all_reduce"(%v2440) ({
    ^bb0(%aras3b4g1: tensor<f32>, %arbs3b4g1: tensor<f32>):
      %aradds3b4g1 = stablehlo.add %aras3b4g1, %arbs3b4g1 : tensor<f32>
      stablehlo.return %aradds3b4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g1 = stablehlo.divide %arsums3b4g1, %arns3b4g1 : tensor<256xf32>
    %v5599 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5600 = stablehlo.multiply %v5599, %s3b4g1 : tensor<256xf32>
    %v5601 = stablehlo.add %v5600, %armeans3b4g1 : tensor<256xf32>
    %v5602 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5603 = stablehlo.multiply %v5602, %s3b4g1v : tensor<256xf32>
    %v5604 = stablehlo.add %v5603, %v5601 : tensor<256xf32>
    %v5605 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5606 = stablehlo.multiply %v5605, %v5604 : tensor<256xf32>
    %v5607 = stablehlo.subtract %s3b4g1, %v5606 : tensor<256xf32>
    %arsums3b4bt1 = "stablehlo.all_reduce"(%v2443) ({
    ^bb0(%aras3b4bt1: tensor<f32>, %arbs3b4bt1: tensor<f32>):
      %aradds3b4bt1 = stablehlo.add %aras3b4bt1, %arbs3b4bt1 : tensor<f32>
      stablehlo.return %aradds3b4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt1 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt1 = stablehlo.divide %arsums3b4bt1, %arns3b4bt1 : tensor<256xf32>
    %v5608 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5609 = stablehlo.multiply %v5608, %s3b4bt1 : tensor<256xf32>
    %v5610 = stablehlo.add %v5609, %armeans3b4bt1 : tensor<256xf32>
    %v5611 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5612 = stablehlo.multiply %v5611, %s3b4bt1v : tensor<256xf32>
    %v5613 = stablehlo.add %v5612, %v5610 : tensor<256xf32>
    %v5614 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5615 = stablehlo.multiply %v5614, %v5613 : tensor<256xf32>
    %v5616 = stablehlo.subtract %s3b4bt1, %v5615 : tensor<256xf32>
    %arsums3b4W2 = "stablehlo.all_reduce"(%v2452) ({
    ^bb0(%aras3b4W2: tensor<f32>, %arbs3b4W2: tensor<f32>):
      %aradds3b4W2 = stablehlo.add %aras3b4W2, %arbs3b4W2 : tensor<f32>
      stablehlo.return %aradds3b4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256x256x3x3xf32>) -> tensor<256x256x3x3xf32>
    %arns3b4W2 = stablehlo.constant dense<4.0> : tensor<256x256x3x3xf32>
    %armeans3b4W2 = stablehlo.divide %arsums3b4W2, %arns3b4W2 : tensor<256x256x3x3xf32>
    %v5617 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5618 = stablehlo.multiply %v5617, %s3b4W2 : tensor<256x256x3x3xf32>
    %v5619 = stablehlo.add %v5618, %armeans3b4W2 : tensor<256x256x3x3xf32>
    %v5620 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5621 = stablehlo.multiply %v5620, %s3b4W2v : tensor<256x256x3x3xf32>
    %v5622 = stablehlo.add %v5621, %v5619 : tensor<256x256x3x3xf32>
    %v5623 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256x256x3x3xf32>
    %v5624 = stablehlo.multiply %v5623, %v5622 : tensor<256x256x3x3xf32>
    %v5625 = stablehlo.subtract %s3b4W2, %v5624 : tensor<256x256x3x3xf32>
    %arsums3b4g2 = "stablehlo.all_reduce"(%v2466) ({
    ^bb0(%aras3b4g2: tensor<f32>, %arbs3b4g2: tensor<f32>):
      %aradds3b4g2 = stablehlo.add %aras3b4g2, %arbs3b4g2 : tensor<f32>
      stablehlo.return %aradds3b4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4g2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4g2 = stablehlo.divide %arsums3b4g2, %arns3b4g2 : tensor<256xf32>
    %v5626 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5627 = stablehlo.multiply %v5626, %s3b4g2 : tensor<256xf32>
    %v5628 = stablehlo.add %v5627, %armeans3b4g2 : tensor<256xf32>
    %v5629 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5630 = stablehlo.multiply %v5629, %s3b4g2v : tensor<256xf32>
    %v5631 = stablehlo.add %v5630, %v5628 : tensor<256xf32>
    %v5632 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5633 = stablehlo.multiply %v5632, %v5631 : tensor<256xf32>
    %v5634 = stablehlo.subtract %s3b4g2, %v5633 : tensor<256xf32>
    %arsums3b4bt2 = "stablehlo.all_reduce"(%v2469) ({
    ^bb0(%aras3b4bt2: tensor<f32>, %arbs3b4bt2: tensor<f32>):
      %aradds3b4bt2 = stablehlo.add %aras3b4bt2, %arbs3b4bt2 : tensor<f32>
      stablehlo.return %aradds3b4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<256xf32>) -> tensor<256xf32>
    %arns3b4bt2 = stablehlo.constant dense<4.0> : tensor<256xf32>
    %armeans3b4bt2 = stablehlo.divide %arsums3b4bt2, %arns3b4bt2 : tensor<256xf32>
    %v5635 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5636 = stablehlo.multiply %v5635, %s3b4bt2 : tensor<256xf32>
    %v5637 = stablehlo.add %v5636, %armeans3b4bt2 : tensor<256xf32>
    %v5638 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5639 = stablehlo.multiply %v5638, %s3b4bt2v : tensor<256xf32>
    %v5640 = stablehlo.add %v5639, %v5637 : tensor<256xf32>
    %v5641 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<256xf32>
    %v5642 = stablehlo.multiply %v5641, %v5640 : tensor<256xf32>
    %v5643 = stablehlo.subtract %s3b4bt2, %v5642 : tensor<256xf32>
    %arsumd4W1 = "stablehlo.all_reduce"(%v2226) ({
    ^bb0(%arad4W1: tensor<f32>, %arbd4W1: tensor<f32>):
      %araddd4W1 = stablehlo.add %arad4W1, %arbd4W1 : tensor<f32>
      stablehlo.return %araddd4W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x3x3xf32>) -> tensor<512x256x3x3xf32>
    %arnd4W1 = stablehlo.constant dense<4.0> : tensor<512x256x3x3xf32>
    %armeand4W1 = stablehlo.divide %arsumd4W1, %arnd4W1 : tensor<512x256x3x3xf32>
    %v5644 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5645 = stablehlo.multiply %v5644, %d4W1 : tensor<512x256x3x3xf32>
    %v5646 = stablehlo.add %v5645, %armeand4W1 : tensor<512x256x3x3xf32>
    %v5647 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5648 = stablehlo.multiply %v5647, %d4W1v : tensor<512x256x3x3xf32>
    %v5649 = stablehlo.add %v5648, %v5646 : tensor<512x256x3x3xf32>
    %v5650 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x3x3xf32>
    %v5651 = stablehlo.multiply %v5650, %v5649 : tensor<512x256x3x3xf32>
    %v5652 = stablehlo.subtract %d4W1, %v5651 : tensor<512x256x3x3xf32>
    %arsumd4g1 = "stablehlo.all_reduce"(%v2240) ({
    ^bb0(%arad4g1: tensor<f32>, %arbd4g1: tensor<f32>):
      %araddd4g1 = stablehlo.add %arad4g1, %arbd4g1 : tensor<f32>
      stablehlo.return %araddd4g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g1 = stablehlo.divide %arsumd4g1, %arnd4g1 : tensor<512xf32>
    %v5653 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5654 = stablehlo.multiply %v5653, %d4g1 : tensor<512xf32>
    %v5655 = stablehlo.add %v5654, %armeand4g1 : tensor<512xf32>
    %v5656 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5657 = stablehlo.multiply %v5656, %d4g1v : tensor<512xf32>
    %v5658 = stablehlo.add %v5657, %v5655 : tensor<512xf32>
    %v5659 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5660 = stablehlo.multiply %v5659, %v5658 : tensor<512xf32>
    %v5661 = stablehlo.subtract %d4g1, %v5660 : tensor<512xf32>
    %arsumd4bt1 = "stablehlo.all_reduce"(%v2243) ({
    ^bb0(%arad4bt1: tensor<f32>, %arbd4bt1: tensor<f32>):
      %araddd4bt1 = stablehlo.add %arad4bt1, %arbd4bt1 : tensor<f32>
      stablehlo.return %araddd4bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4bt1 = stablehlo.divide %arsumd4bt1, %arnd4bt1 : tensor<512xf32>
    %v5662 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5663 = stablehlo.multiply %v5662, %d4bt1 : tensor<512xf32>
    %v5664 = stablehlo.add %v5663, %armeand4bt1 : tensor<512xf32>
    %v5665 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5666 = stablehlo.multiply %v5665, %d4bt1v : tensor<512xf32>
    %v5667 = stablehlo.add %v5666, %v5664 : tensor<512xf32>
    %v5668 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5669 = stablehlo.multiply %v5668, %v5667 : tensor<512xf32>
    %v5670 = stablehlo.subtract %d4bt1, %v5669 : tensor<512xf32>
    %arsumd4W2 = "stablehlo.all_reduce"(%v2252) ({
    ^bb0(%arad4W2: tensor<f32>, %arbd4W2: tensor<f32>):
      %araddd4W2 = stablehlo.add %arad4W2, %arbd4W2 : tensor<f32>
      stablehlo.return %araddd4W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arnd4W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeand4W2 = stablehlo.divide %arsumd4W2, %arnd4W2 : tensor<512x512x3x3xf32>
    %v5671 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5672 = stablehlo.multiply %v5671, %d4W2 : tensor<512x512x3x3xf32>
    %v5673 = stablehlo.add %v5672, %armeand4W2 : tensor<512x512x3x3xf32>
    %v5674 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5675 = stablehlo.multiply %v5674, %d4W2v : tensor<512x512x3x3xf32>
    %v5676 = stablehlo.add %v5675, %v5673 : tensor<512x512x3x3xf32>
    %v5677 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5678 = stablehlo.multiply %v5677, %v5676 : tensor<512x512x3x3xf32>
    %v5679 = stablehlo.subtract %d4W2, %v5678 : tensor<512x512x3x3xf32>
    %arsumd4g2 = "stablehlo.all_reduce"(%v2266) ({
    ^bb0(%arad4g2: tensor<f32>, %arbd4g2: tensor<f32>):
      %araddd4g2 = stablehlo.add %arad4g2, %arbd4g2 : tensor<f32>
      stablehlo.return %araddd4g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4g2 = stablehlo.divide %arsumd4g2, %arnd4g2 : tensor<512xf32>
    %v5680 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5681 = stablehlo.multiply %v5680, %d4g2 : tensor<512xf32>
    %v5682 = stablehlo.add %v5681, %armeand4g2 : tensor<512xf32>
    %v5683 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5684 = stablehlo.multiply %v5683, %d4g2v : tensor<512xf32>
    %v5685 = stablehlo.add %v5684, %v5682 : tensor<512xf32>
    %v5686 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5687 = stablehlo.multiply %v5686, %v5685 : tensor<512xf32>
    %v5688 = stablehlo.subtract %d4g2, %v5687 : tensor<512xf32>
    %arsumd4bt2 = "stablehlo.all_reduce"(%v2269) ({
    ^bb0(%arad4bt2: tensor<f32>, %arbd4bt2: tensor<f32>):
      %araddd4bt2 = stablehlo.add %arad4bt2, %arbd4bt2 : tensor<f32>
      stablehlo.return %araddd4bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4bt2 = stablehlo.divide %arsumd4bt2, %arnd4bt2 : tensor<512xf32>
    %v5689 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5690 = stablehlo.multiply %v5689, %d4bt2 : tensor<512xf32>
    %v5691 = stablehlo.add %v5690, %armeand4bt2 : tensor<512xf32>
    %v5692 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5693 = stablehlo.multiply %v5692, %d4bt2v : tensor<512xf32>
    %v5694 = stablehlo.add %v5693, %v5691 : tensor<512xf32>
    %v5695 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5696 = stablehlo.multiply %v5695, %v5694 : tensor<512xf32>
    %v5697 = stablehlo.subtract %d4bt2, %v5696 : tensor<512xf32>
    %arsumd4Wp = "stablehlo.all_reduce"(%v2280) ({
    ^bb0(%arad4Wp: tensor<f32>, %arbd4Wp: tensor<f32>):
      %araddd4Wp = stablehlo.add %arad4Wp, %arbd4Wp : tensor<f32>
      stablehlo.return %araddd4Wp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x256x1x1xf32>) -> tensor<512x256x1x1xf32>
    %arnd4Wp = stablehlo.constant dense<4.0> : tensor<512x256x1x1xf32>
    %armeand4Wp = stablehlo.divide %arsumd4Wp, %arnd4Wp : tensor<512x256x1x1xf32>
    %v5698 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5699 = stablehlo.multiply %v5698, %d4Wp : tensor<512x256x1x1xf32>
    %v5700 = stablehlo.add %v5699, %armeand4Wp : tensor<512x256x1x1xf32>
    %v5701 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5702 = stablehlo.multiply %v5701, %d4Wpv : tensor<512x256x1x1xf32>
    %v5703 = stablehlo.add %v5702, %v5700 : tensor<512x256x1x1xf32>
    %v5704 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x256x1x1xf32>
    %v5705 = stablehlo.multiply %v5704, %v5703 : tensor<512x256x1x1xf32>
    %v5706 = stablehlo.subtract %d4Wp, %v5705 : tensor<512x256x1x1xf32>
    %arsumd4gp = "stablehlo.all_reduce"(%v2294) ({
    ^bb0(%arad4gp: tensor<f32>, %arbd4gp: tensor<f32>):
      %araddd4gp = stablehlo.add %arad4gp, %arbd4gp : tensor<f32>
      stablehlo.return %araddd4gp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4gp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4gp = stablehlo.divide %arsumd4gp, %arnd4gp : tensor<512xf32>
    %v5707 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5708 = stablehlo.multiply %v5707, %d4gp : tensor<512xf32>
    %v5709 = stablehlo.add %v5708, %armeand4gp : tensor<512xf32>
    %v5710 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5711 = stablehlo.multiply %v5710, %d4gpv : tensor<512xf32>
    %v5712 = stablehlo.add %v5711, %v5709 : tensor<512xf32>
    %v5713 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5714 = stablehlo.multiply %v5713, %v5712 : tensor<512xf32>
    %v5715 = stablehlo.subtract %d4gp, %v5714 : tensor<512xf32>
    %arsumd4btp = "stablehlo.all_reduce"(%v2297) ({
    ^bb0(%arad4btp: tensor<f32>, %arbd4btp: tensor<f32>):
      %araddd4btp = stablehlo.add %arad4btp, %arbd4btp : tensor<f32>
      stablehlo.return %araddd4btp : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arnd4btp = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeand4btp = stablehlo.divide %arsumd4btp, %arnd4btp : tensor<512xf32>
    %v5716 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5717 = stablehlo.multiply %v5716, %d4btp : tensor<512xf32>
    %v5718 = stablehlo.add %v5717, %armeand4btp : tensor<512xf32>
    %v5719 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5720 = stablehlo.multiply %v5719, %d4btpv : tensor<512xf32>
    %v5721 = stablehlo.add %v5720, %v5718 : tensor<512xf32>
    %v5722 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5723 = stablehlo.multiply %v5722, %v5721 : tensor<512xf32>
    %v5724 = stablehlo.subtract %d4btp, %v5723 : tensor<512xf32>
    %arsums4b0W1 = "stablehlo.all_reduce"(%v1996) ({
    ^bb0(%aras4b0W1: tensor<f32>, %arbs4b0W1: tensor<f32>):
      %aradds4b0W1 = stablehlo.add %aras4b0W1, %arbs4b0W1 : tensor<f32>
      stablehlo.return %aradds4b0W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W1 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W1 = stablehlo.divide %arsums4b0W1, %arns4b0W1 : tensor<512x512x3x3xf32>
    %v5725 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5726 = stablehlo.multiply %v5725, %s4b0W1 : tensor<512x512x3x3xf32>
    %v5727 = stablehlo.add %v5726, %armeans4b0W1 : tensor<512x512x3x3xf32>
    %v5728 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5729 = stablehlo.multiply %v5728, %s4b0W1v : tensor<512x512x3x3xf32>
    %v5730 = stablehlo.add %v5729, %v5727 : tensor<512x512x3x3xf32>
    %v5731 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5732 = stablehlo.multiply %v5731, %v5730 : tensor<512x512x3x3xf32>
    %v5733 = stablehlo.subtract %s4b0W1, %v5732 : tensor<512x512x3x3xf32>
    %arsums4b0g1 = "stablehlo.all_reduce"(%v2010) ({
    ^bb0(%aras4b0g1: tensor<f32>, %arbs4b0g1: tensor<f32>):
      %aradds4b0g1 = stablehlo.add %aras4b0g1, %arbs4b0g1 : tensor<f32>
      stablehlo.return %aradds4b0g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g1 = stablehlo.divide %arsums4b0g1, %arns4b0g1 : tensor<512xf32>
    %v5734 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5735 = stablehlo.multiply %v5734, %s4b0g1 : tensor<512xf32>
    %v5736 = stablehlo.add %v5735, %armeans4b0g1 : tensor<512xf32>
    %v5737 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5738 = stablehlo.multiply %v5737, %s4b0g1v : tensor<512xf32>
    %v5739 = stablehlo.add %v5738, %v5736 : tensor<512xf32>
    %v5740 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5741 = stablehlo.multiply %v5740, %v5739 : tensor<512xf32>
    %v5742 = stablehlo.subtract %s4b0g1, %v5741 : tensor<512xf32>
    %arsums4b0bt1 = "stablehlo.all_reduce"(%v2013) ({
    ^bb0(%aras4b0bt1: tensor<f32>, %arbs4b0bt1: tensor<f32>):
      %aradds4b0bt1 = stablehlo.add %aras4b0bt1, %arbs4b0bt1 : tensor<f32>
      stablehlo.return %aradds4b0bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt1 = stablehlo.divide %arsums4b0bt1, %arns4b0bt1 : tensor<512xf32>
    %v5743 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5744 = stablehlo.multiply %v5743, %s4b0bt1 : tensor<512xf32>
    %v5745 = stablehlo.add %v5744, %armeans4b0bt1 : tensor<512xf32>
    %v5746 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5747 = stablehlo.multiply %v5746, %s4b0bt1v : tensor<512xf32>
    %v5748 = stablehlo.add %v5747, %v5745 : tensor<512xf32>
    %v5749 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5750 = stablehlo.multiply %v5749, %v5748 : tensor<512xf32>
    %v5751 = stablehlo.subtract %s4b0bt1, %v5750 : tensor<512xf32>
    %arsums4b0W2 = "stablehlo.all_reduce"(%v2022) ({
    ^bb0(%aras4b0W2: tensor<f32>, %arbs4b0W2: tensor<f32>):
      %aradds4b0W2 = stablehlo.add %aras4b0W2, %arbs4b0W2 : tensor<f32>
      stablehlo.return %aradds4b0W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b0W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b0W2 = stablehlo.divide %arsums4b0W2, %arns4b0W2 : tensor<512x512x3x3xf32>
    %v5752 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5753 = stablehlo.multiply %v5752, %s4b0W2 : tensor<512x512x3x3xf32>
    %v5754 = stablehlo.add %v5753, %armeans4b0W2 : tensor<512x512x3x3xf32>
    %v5755 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5756 = stablehlo.multiply %v5755, %s4b0W2v : tensor<512x512x3x3xf32>
    %v5757 = stablehlo.add %v5756, %v5754 : tensor<512x512x3x3xf32>
    %v5758 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5759 = stablehlo.multiply %v5758, %v5757 : tensor<512x512x3x3xf32>
    %v5760 = stablehlo.subtract %s4b0W2, %v5759 : tensor<512x512x3x3xf32>
    %arsums4b0g2 = "stablehlo.all_reduce"(%v2036) ({
    ^bb0(%aras4b0g2: tensor<f32>, %arbs4b0g2: tensor<f32>):
      %aradds4b0g2 = stablehlo.add %aras4b0g2, %arbs4b0g2 : tensor<f32>
      stablehlo.return %aradds4b0g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0g2 = stablehlo.divide %arsums4b0g2, %arns4b0g2 : tensor<512xf32>
    %v5761 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5762 = stablehlo.multiply %v5761, %s4b0g2 : tensor<512xf32>
    %v5763 = stablehlo.add %v5762, %armeans4b0g2 : tensor<512xf32>
    %v5764 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5765 = stablehlo.multiply %v5764, %s4b0g2v : tensor<512xf32>
    %v5766 = stablehlo.add %v5765, %v5763 : tensor<512xf32>
    %v5767 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5768 = stablehlo.multiply %v5767, %v5766 : tensor<512xf32>
    %v5769 = stablehlo.subtract %s4b0g2, %v5768 : tensor<512xf32>
    %arsums4b0bt2 = "stablehlo.all_reduce"(%v2039) ({
    ^bb0(%aras4b0bt2: tensor<f32>, %arbs4b0bt2: tensor<f32>):
      %aradds4b0bt2 = stablehlo.add %aras4b0bt2, %arbs4b0bt2 : tensor<f32>
      stablehlo.return %aradds4b0bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b0bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b0bt2 = stablehlo.divide %arsums4b0bt2, %arns4b0bt2 : tensor<512xf32>
    %v5770 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5771 = stablehlo.multiply %v5770, %s4b0bt2 : tensor<512xf32>
    %v5772 = stablehlo.add %v5771, %armeans4b0bt2 : tensor<512xf32>
    %v5773 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5774 = stablehlo.multiply %v5773, %s4b0bt2v : tensor<512xf32>
    %v5775 = stablehlo.add %v5774, %v5772 : tensor<512xf32>
    %v5776 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5777 = stablehlo.multiply %v5776, %v5775 : tensor<512xf32>
    %v5778 = stablehlo.subtract %s4b0bt2, %v5777 : tensor<512xf32>
    %arsums4b1W1 = "stablehlo.all_reduce"(%v1824) ({
    ^bb0(%aras4b1W1: tensor<f32>, %arbs4b1W1: tensor<f32>):
      %aradds4b1W1 = stablehlo.add %aras4b1W1, %arbs4b1W1 : tensor<f32>
      stablehlo.return %aradds4b1W1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W1 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W1 = stablehlo.divide %arsums4b1W1, %arns4b1W1 : tensor<512x512x3x3xf32>
    %v5779 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5780 = stablehlo.multiply %v5779, %s4b1W1 : tensor<512x512x3x3xf32>
    %v5781 = stablehlo.add %v5780, %armeans4b1W1 : tensor<512x512x3x3xf32>
    %v5782 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5783 = stablehlo.multiply %v5782, %s4b1W1v : tensor<512x512x3x3xf32>
    %v5784 = stablehlo.add %v5783, %v5781 : tensor<512x512x3x3xf32>
    %v5785 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5786 = stablehlo.multiply %v5785, %v5784 : tensor<512x512x3x3xf32>
    %v5787 = stablehlo.subtract %s4b1W1, %v5786 : tensor<512x512x3x3xf32>
    %arsums4b1g1 = "stablehlo.all_reduce"(%v1838) ({
    ^bb0(%aras4b1g1: tensor<f32>, %arbs4b1g1: tensor<f32>):
      %aradds4b1g1 = stablehlo.add %aras4b1g1, %arbs4b1g1 : tensor<f32>
      stablehlo.return %aradds4b1g1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g1 = stablehlo.divide %arsums4b1g1, %arns4b1g1 : tensor<512xf32>
    %v5788 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5789 = stablehlo.multiply %v5788, %s4b1g1 : tensor<512xf32>
    %v5790 = stablehlo.add %v5789, %armeans4b1g1 : tensor<512xf32>
    %v5791 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5792 = stablehlo.multiply %v5791, %s4b1g1v : tensor<512xf32>
    %v5793 = stablehlo.add %v5792, %v5790 : tensor<512xf32>
    %v5794 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5795 = stablehlo.multiply %v5794, %v5793 : tensor<512xf32>
    %v5796 = stablehlo.subtract %s4b1g1, %v5795 : tensor<512xf32>
    %arsums4b1bt1 = "stablehlo.all_reduce"(%v1841) ({
    ^bb0(%aras4b1bt1: tensor<f32>, %arbs4b1bt1: tensor<f32>):
      %aradds4b1bt1 = stablehlo.add %aras4b1bt1, %arbs4b1bt1 : tensor<f32>
      stablehlo.return %aradds4b1bt1 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt1 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt1 = stablehlo.divide %arsums4b1bt1, %arns4b1bt1 : tensor<512xf32>
    %v5797 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5798 = stablehlo.multiply %v5797, %s4b1bt1 : tensor<512xf32>
    %v5799 = stablehlo.add %v5798, %armeans4b1bt1 : tensor<512xf32>
    %v5800 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5801 = stablehlo.multiply %v5800, %s4b1bt1v : tensor<512xf32>
    %v5802 = stablehlo.add %v5801, %v5799 : tensor<512xf32>
    %v5803 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5804 = stablehlo.multiply %v5803, %v5802 : tensor<512xf32>
    %v5805 = stablehlo.subtract %s4b1bt1, %v5804 : tensor<512xf32>
    %arsums4b1W2 = "stablehlo.all_reduce"(%v1850) ({
    ^bb0(%aras4b1W2: tensor<f32>, %arbs4b1W2: tensor<f32>):
      %aradds4b1W2 = stablehlo.add %aras4b1W2, %arbs4b1W2 : tensor<f32>
      stablehlo.return %aradds4b1W2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x512x3x3xf32>) -> tensor<512x512x3x3xf32>
    %arns4b1W2 = stablehlo.constant dense<4.0> : tensor<512x512x3x3xf32>
    %armeans4b1W2 = stablehlo.divide %arsums4b1W2, %arns4b1W2 : tensor<512x512x3x3xf32>
    %v5806 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5807 = stablehlo.multiply %v5806, %s4b1W2 : tensor<512x512x3x3xf32>
    %v5808 = stablehlo.add %v5807, %armeans4b1W2 : tensor<512x512x3x3xf32>
    %v5809 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5810 = stablehlo.multiply %v5809, %s4b1W2v : tensor<512x512x3x3xf32>
    %v5811 = stablehlo.add %v5810, %v5808 : tensor<512x512x3x3xf32>
    %v5812 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x512x3x3xf32>
    %v5813 = stablehlo.multiply %v5812, %v5811 : tensor<512x512x3x3xf32>
    %v5814 = stablehlo.subtract %s4b1W2, %v5813 : tensor<512x512x3x3xf32>
    %arsums4b1g2 = "stablehlo.all_reduce"(%v1864) ({
    ^bb0(%aras4b1g2: tensor<f32>, %arbs4b1g2: tensor<f32>):
      %aradds4b1g2 = stablehlo.add %aras4b1g2, %arbs4b1g2 : tensor<f32>
      stablehlo.return %aradds4b1g2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1g2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1g2 = stablehlo.divide %arsums4b1g2, %arns4b1g2 : tensor<512xf32>
    %v5815 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5816 = stablehlo.multiply %v5815, %s4b1g2 : tensor<512xf32>
    %v5817 = stablehlo.add %v5816, %armeans4b1g2 : tensor<512xf32>
    %v5818 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5819 = stablehlo.multiply %v5818, %s4b1g2v : tensor<512xf32>
    %v5820 = stablehlo.add %v5819, %v5817 : tensor<512xf32>
    %v5821 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5822 = stablehlo.multiply %v5821, %v5820 : tensor<512xf32>
    %v5823 = stablehlo.subtract %s4b1g2, %v5822 : tensor<512xf32>
    %arsums4b1bt2 = "stablehlo.all_reduce"(%v1867) ({
    ^bb0(%aras4b1bt2: tensor<f32>, %arbs4b1bt2: tensor<f32>):
      %aradds4b1bt2 = stablehlo.add %aras4b1bt2, %arbs4b1bt2 : tensor<f32>
      stablehlo.return %aradds4b1bt2 : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512xf32>) -> tensor<512xf32>
    %arns4b1bt2 = stablehlo.constant dense<4.0> : tensor<512xf32>
    %armeans4b1bt2 = stablehlo.divide %arsums4b1bt2, %arns4b1bt2 : tensor<512xf32>
    %v5824 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5825 = stablehlo.multiply %v5824, %s4b1bt2 : tensor<512xf32>
    %v5826 = stablehlo.add %v5825, %armeans4b1bt2 : tensor<512xf32>
    %v5827 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5828 = stablehlo.multiply %v5827, %s4b1bt2v : tensor<512xf32>
    %v5829 = stablehlo.add %v5828, %v5826 : tensor<512xf32>
    %v5830 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512xf32>
    %v5831 = stablehlo.multiply %v5830, %v5829 : tensor<512xf32>
    %v5832 = stablehlo.subtract %s4b1bt2, %v5831 : tensor<512xf32>
    %arsumWd = "stablehlo.all_reduce"(%v1689) ({
    ^bb0(%araWd: tensor<f32>, %arbWd: tensor<f32>):
      %araddWd = stablehlo.add %araWd, %arbWd : tensor<f32>
      stablehlo.return %araddWd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<512x1000xf32>) -> tensor<512x1000xf32>
    %arnWd = stablehlo.constant dense<4.0> : tensor<512x1000xf32>
    %armeanWd = stablehlo.divide %arsumWd, %arnWd : tensor<512x1000xf32>
    %v5833 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5834 = stablehlo.multiply %v5833, %Wd : tensor<512x1000xf32>
    %v5835 = stablehlo.add %v5834, %armeanWd : tensor<512x1000xf32>
    %v5836 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5837 = stablehlo.multiply %v5836, %Wdv : tensor<512x1000xf32>
    %v5838 = stablehlo.add %v5837, %v5835 : tensor<512x1000xf32>
    %v5839 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<512x1000xf32>
    %v5840 = stablehlo.multiply %v5839, %v5838 : tensor<512x1000xf32>
    %v5841 = stablehlo.subtract %Wd, %v5840 : tensor<512x1000xf32>
    %arsumbd = "stablehlo.all_reduce"(%v1691) ({
    ^bb0(%arabd: tensor<f32>, %arbbd: tensor<f32>):
      %araddbd = stablehlo.add %arabd, %arbbd : tensor<f32>
      stablehlo.return %araddbd : tensor<f32>
    }) { replica_groups = dense<[[0, 1, 2, 3]]> : tensor<1x4xi64> } : (tensor<1000xf32>) -> tensor<1000xf32>
    %arnbd = stablehlo.constant dense<4.0> : tensor<1000xf32>
    %armeanbd = stablehlo.divide %arsumbd, %arnbd : tensor<1000xf32>
    %v5842 = stablehlo.broadcast_in_dim %wd, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5843 = stablehlo.multiply %v5842, %bd : tensor<1000xf32>
    %v5844 = stablehlo.add %v5843, %armeanbd : tensor<1000xf32>
    %v5845 = stablehlo.broadcast_in_dim %mu, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5846 = stablehlo.multiply %v5845, %bdv : tensor<1000xf32>
    %v5847 = stablehlo.add %v5846, %v5844 : tensor<1000xf32>
    %v5848 = stablehlo.broadcast_in_dim %lr, dims = [] : (tensor<f32>) -> tensor<1000xf32>
    %v5849 = stablehlo.multiply %v5848, %v5847 : tensor<1000xf32>
    %v5850 = stablehlo.subtract %bd, %v5849 : tensor<1000xf32>
    // ── %loss below is REPORT-ONLY (logging), NOT pretty(AST node) ──
    %lz = stablehlo.constant dense<0.0> : tensor<f32>
    %llog = stablehlo.log %v1677 : tensor<64x1000xf32>
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
    return %v4869, %v4878, %v4887, %v4896, %v4905, %v4914, %v4923, %v4932, %v4941, %v4950, %v4959, %v4968, %v4977, %v4986, %v4995, %v5004, %v5013, %v5022, %v5031, %v5040, %v5049, %v5058, %v5067, %v5076, %v5085, %v5094, %v5103, %v5112, %v5121, %v5130, %v5139, %v5148, %v5157, %v5166, %v5175, %v5184, %v5193, %v5202, %v5211, %v5220, %v5229, %v5238, %v5247, %v5256, %v5265, %v5274, %v5283, %v5292, %v5301, %v5310, %v5319, %v5328, %v5337, %v5346, %v5355, %v5364, %v5373, %v5382, %v5391, %v5400, %v5409, %v5418, %v5427, %v5436, %v5445, %v5454, %v5463, %v5472, %v5481, %v5490, %v5499, %v5508, %v5517, %v5526, %v5535, %v5544, %v5553, %v5562, %v5571, %v5580, %v5589, %v5598, %v5607, %v5616, %v5625, %v5634, %v5643, %v5652, %v5661, %v5670, %v5679, %v5688, %v5697, %v5706, %v5715, %v5724, %v5733, %v5742, %v5751, %v5760, %v5769, %v5778, %v5787, %v5796, %v5805, %v5814, %v5823, %v5832, %v5841, %v5850, %sWm, %sgm, %sbtm, %s1b0W1m, %s1b0g1m, %s1b0bt1m, %s1b0W2m, %s1b0g2m, %s1b0bt2m, %s1b1W1m, %s1b1g1m, %s1b1bt1m, %s1b1W2m, %s1b1g2m, %s1b1bt2m, %s1b2W1m, %s1b2g1m, %s1b2bt1m, %s1b2W2m, %s1b2g2m, %s1b2bt2m, %d2W1m, %d2g1m, %d2bt1m, %d2W2m, %d2g2m, %d2bt2m, %d2Wpm, %d2gpm, %d2btpm, %s2b0W1m, %s2b0g1m, %s2b0bt1m, %s2b0W2m, %s2b0g2m, %s2b0bt2m, %s2b1W1m, %s2b1g1m, %s2b1bt1m, %s2b1W2m, %s2b1g2m, %s2b1bt2m, %s2b2W1m, %s2b2g1m, %s2b2bt1m, %s2b2W2m, %s2b2g2m, %s2b2bt2m, %d3W1m, %d3g1m, %d3bt1m, %d3W2m, %d3g2m, %d3bt2m, %d3Wpm, %d3gpm, %d3btpm, %s3b0W1m, %s3b0g1m, %s3b0bt1m, %s3b0W2m, %s3b0g2m, %s3b0bt2m, %s3b1W1m, %s3b1g1m, %s3b1bt1m, %s3b1W2m, %s3b1g2m, %s3b1bt2m, %s3b2W1m, %s3b2g1m, %s3b2bt1m, %s3b2W2m, %s3b2g2m, %s3b2bt2m, %s3b3W1m, %s3b3g1m, %s3b3bt1m, %s3b3W2m, %s3b3g2m, %s3b3bt2m, %s3b4W1m, %s3b4g1m, %s3b4bt1m, %s3b4W2m, %s3b4g2m, %s3b4bt2m, %d4W1m, %d4g1m, %d4bt1m, %d4W2m, %d4g2m, %d4bt2m, %d4Wpm, %d4gpm, %d4btpm, %s4b0W1m, %s4b0g1m, %s4b0bt1m, %s4b0W2m, %s4b0g2m, %s4b0bt2m, %s4b1W1m, %s4b1g1m, %s4b1bt1m, %s4b1W2m, %s4b1g2m, %s4b1bt2m, %Wdm, %bdm, %v4866, %v4875, %v4884, %v4893, %v4902, %v4911, %v4920, %v4929, %v4938, %v4947, %v4956, %v4965, %v4974, %v4983, %v4992, %v5001, %v5010, %v5019, %v5028, %v5037, %v5046, %v5055, %v5064, %v5073, %v5082, %v5091, %v5100, %v5109, %v5118, %v5127, %v5136, %v5145, %v5154, %v5163, %v5172, %v5181, %v5190, %v5199, %v5208, %v5217, %v5226, %v5235, %v5244, %v5253, %v5262, %v5271, %v5280, %v5289, %v5298, %v5307, %v5316, %v5325, %v5334, %v5343, %v5352, %v5361, %v5370, %v5379, %v5388, %v5397, %v5406, %v5415, %v5424, %v5433, %v5442, %v5451, %v5460, %v5469, %v5478, %v5487, %v5496, %v5505, %v5514, %v5523, %v5532, %v5541, %v5550, %v5559, %v5568, %v5577, %v5586, %v5595, %v5604, %v5613, %v5622, %v5631, %v5640, %v5649, %v5658, %v5667, %v5676, %v5685, %v5694, %v5703, %v5712, %v5721, %v5730, %v5739, %v5748, %v5757, %v5766, %v5775, %v5784, %v5793, %v5802, %v5811, %v5820, %v5829, %v5838, %v5847, %loss, %bc1, %bc2, %v4789, %v4790, %v4791, %v4792, %v4793, %v4794, %v4795, %v4796, %v4797, %v4798, %v4799, %v4800, %v4801, %v4802, %v4803, %v4804, %v4805, %v4806, %v4807, %v4808, %v4809, %v4810, %v4811, %v4812, %v4813, %v4814, %v4815, %v4816, %v4817, %v4818, %v4819, %v4820, %v4821, %v4822, %v4823, %v4824, %v4825, %v4826, %v4827, %v4828, %v4829, %v4830, %v4831, %v4832, %v4833, %v4834, %v4835, %v4836, %v4837, %v4838, %v4839, %v4840, %v4841, %v4842, %v4843, %v4844, %v4845, %v4846, %v4847, %v4848, %v4849, %v4850, %v4851, %v4852, %v4853, %v4854, %v4855, %v4856, %v4857, %v4858, %v4859, %v4860 : tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<64x3x7x7xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64x64x3x3xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128x64x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x64x1x1xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128x128x3x3xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256x128x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x128x1x1xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256x256x3x3xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512x256x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x256x1x1xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x512x3x3xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512x1000xf32>, tensor<1000xf32>, tensor<f32>, tensor<f32>, tensor<f32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<64xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<128xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<256xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>, tensor<512xf32>
  }
}
